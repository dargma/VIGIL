"""
VIGIL InternVL3.5-1B Steered Evaluation + Visualization.

Runs: baseline, blind, steered (multiple alphas), steered_blind.
Generates heatmaps and comparison plots.

Usage:
    python scripts/eval_internvl_steered.py --max-samples 500
"""

import os, sys, json, re, gc, argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from datetime import datetime
from datasets import load_from_disk

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model_registry import load_model
from src.steerer import ActivationSteerer

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


# ─── VLMEvalKit standard ───────────────────────────────────────────────────

def process_punctuation(inText):
    outText = inText
    punct = [';', r'/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\',
             '_', '-', '>', '<', '@', '`', ',', '?', '!']
    commaStrip = re.compile(r'(\d)(,)(\d)')
    periodStrip = re.compile(r'(?<!\d)\.(?!\d)')
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (
                re.search(commaStrip, inText) is not None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub('', outText, re.UNICODE)
    return outText

def YOrN_Extraction(output):
    s = output.lower()
    words = process_punctuation(s).split()
    if 'yes' in words and 'no' not in words:
        return 'Yes'
    if 'yes' not in words and 'no' in words:
        return 'No'
    return 'Unknown'

POPE_PROMPT = "{question} Please answer yes or no."
TRANSFORM = T.Compose([
    T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])


def pope_metrics(records):
    import pandas as pd
    def cal_f1(yt, yp):
        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        return f1, p, r
    df = pd.DataFrame(records)
    results = {}
    for split_name, sub in [("Overall", df)] + [
        (cat, df[df["category"] == cat]) for cat in sorted(df["category"].unique())
    ]:
        yt = np.array([1 if a == "Yes" else 0 for a in sub["answer"]])
        yp = np.array([1 if a == "Yes" else 0 for a in sub["extracted"]])
        score = np.array([1 if a == e else 0 for a, e in zip(sub["answer"], sub["extracted"])])
        f1, p, r = cal_f1(yt, yp)
        results[split_name] = {
            "acc": float(np.mean(score) * 100), "f1": float(f1 * 100),
            "precision": float(p * 100), "recall": float(r * 100),
            "n": len(sub), "n_unknown": int(sum(1 for e in sub["extracted"] if e == "Unknown")),
        }
    return results


def generate_one(model_info, image, question, blind=False):
    if blind:
        image = Image.new("RGB", image.size, (0, 0, 0))
    image_rgb = image.convert("RGB")
    pixel_values = TRANSFORM(image_rgb).unsqueeze(0).to(
        device=model_info["device"], dtype=torch.bfloat16
    )
    prompt = POPE_PROMPT.format(question=question)
    with torch.no_grad():
        response = model_info["model"].chat(
            model_info["tokenizer"], pixel_values, prompt,
            generation_config={"max_new_tokens": 64}
        )
    return response.strip()


def run_pope(model_info, dataset, label, blind=False, max_n=None):
    n = min(len(dataset), max_n or len(dataset))
    records = []
    print(f"\n[pope] {label} ({'blind' if blind else 'real'}) — {n} samples")
    for i in range(n):
        s = dataset[i]
        try:
            raw = generate_one(model_info, s["image"], s["question"], blind=blind)
        except Exception as e:
            print(f"  [{i}] ERR: {e}")
            raw = ""
        ext = YOrN_Extraction(raw)
        records.append({
            "index": i, "question": s["question"],
            "answer": s["answer"].strip().capitalize(),
            "prediction": raw, "extracted": ext,
            "category": s.get("category", "unknown"),
        })
        if (i + 1) % 200 == 0:
            acc = pope_metrics(records)["Overall"]["acc"]
            print(f"  [{i+1}/{n}] acc={acc:.1f}%")
    metrics = pope_metrics(records)
    o = metrics["Overall"]
    print(f"  → {label}: acc={o['acc']:.1f}%, F1={o['f1']:.1f}%, P={o['precision']:.1f}%, R={o['recall']:.1f}%")
    return records, metrics


def setup_steering(model_info, cal_data, alpha=3.0):
    """Set up steering using calibration data."""
    top_heads = cal_data["top_heads"]
    head_scores = cal_data["head_scores"]

    # Install steering hooks on top vision heads
    layers = model_info["get_layers_fn"]()
    hooks = []
    steering_info = {"alpha": alpha, "heads": top_heads[:10]}

    for li, hi in top_heads[:10]:  # top 10 heads
        key = f"{li},{hi}"
        if key not in head_scores:
            continue

        d = head_scores[key]["cohens_d"]
        correct_mean = head_scores[key]["correct_mean"]
        incorrect_mean = head_scores[key]["incorrect_mean"]

        # Steering direction: push toward correct activation level
        direction = 1.0 if correct_mean > incorrect_mean else -1.0

        layer = layers[li]
        attn = layer.self_attn
        o_proj = attn.o_proj

        def make_steer_hook(layer_idx, head_idx, dir_sign, steer_alpha, head_dim):
            def hook_fn(module, args):
                x = args[0]  # (batch, seq, hidden_size)
                batch, seq_len, hidden = x.shape
                num_heads = hidden // head_dim

                # Reshape to per-head
                x_heads = x.view(batch, seq_len, num_heads, head_dim)

                # Scale the target head
                scale = 1.0 + dir_sign * steer_alpha * 0.1  # gentle scaling
                x_heads[:, :, head_idx, :] = x_heads[:, :, head_idx, :] * scale

                return (x_heads.view(batch, seq_len, hidden),) + args[1:]
            return hook_fn

        handle = o_proj.register_forward_pre_hook(
            make_steer_hook(li, hi, direction, alpha, model_info["head_dim"])
        )
        hooks.append(handle)

    print(f"[steer] Installed {len(hooks)} steering hooks, alpha={alpha}")
    return hooks


def cleanup_steering(hooks):
    for h in hooks:
        h.remove()
    print("[steer] Removed steering hooks")


def generate_heatmap(cal_data, output_dir):
    """Generate Cohen's d heatmap of vision heads."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    head_scores = cal_data["head_scores"]
    num_layers = 28
    num_heads = 16

    heatmap = np.zeros((num_layers, num_heads))
    for key, v in head_scores.items():
        li, hi = [int(x) for x in key.split(",")]
        heatmap[li, hi] = v["cohens_d"]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(heatmap, aspect="auto", cmap="hot", interpolation="nearest")
    ax.set_xlabel("Head Index", fontsize=12)
    ax.set_ylabel("Layer Index", fontsize=12)
    ax.set_title("InternVL3.5-1B: Cohen's d per Attention Head\n(higher = more vision-specialized)", fontsize=14)
    plt.colorbar(im, ax=ax, label="Cohen's d")

    # Mark top 10 heads
    for li, hi in cal_data["top_heads"][:10]:
        ax.plot(hi, li, "w*", markersize=10, markeredgecolor="black")

    plt.tight_layout()
    fig.savefig(output_dir / "internvl_cohens_d_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved heatmap to {output_dir / 'internvl_cohens_d_heatmap.png'}")


def generate_comparison_plot(all_results, output_dir):
    """Generate bar chart comparing conditions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = []
    accs = []
    f1s = []
    for label, data in all_results.items():
        if "metrics" in data:
            o = data["metrics"]["Overall"]
            labels.append(label)
            accs.append(o["acc"])
            f1s.append(o["f1"])

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, accs, width, label="Accuracy", color="#2196F3")
    bars2 = ax.bar(x + width/2, f1s, width, label="F1", color="#FF9800")

    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("InternVL3.5-1B POPE Evaluation — All Conditions", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 100)

    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / "internvl_pope_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved comparison to {output_dir / 'internvl_pope_comparison.png'}")


def generate_cross_model_plot(output_dir):
    """Generate cross-model comparison if Qwen3 data exists."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = {
        "Qwen3-VL-2B": {"acc": 87.4, "f1": 87.2, "gap": 37.4, "precision": 88.7},
        "Qwen3+BoN+SFT": {"acc": 87.8, "f1": 87.4, "gap": 37.8, "precision": 90.3},
        "Qwen3+DAPO": {"acc": 87.8, "f1": 87.4, "gap": 37.8, "precision": 90.3},
    }

    # Will be filled with InternVL data during the run
    return models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--alphas", type=str, default="1,3,5,10",
                        help="Comma-separated steering alphas to test")
    parser.add_argument("--output-dir", type=str, default="lab/reports/multimodel/internvl3_5_1b")
    args = parser.parse_args()

    dataset = load_from_disk("data/eval/pope")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    alphas = [float(a) for a in args.alphas.split(",")]

    # Load calibration
    cal_path = Path("checkpoints/calibration/internvl3_5_1b/calibration.json")
    if not cal_path.exists():
        print("[ERROR] Calibration not found. Run calibrate_internvl.py first.")
        return
    with open(cal_path) as f:
        cal_data = json.load(f)

    # Generate heatmap
    generate_heatmap(cal_data, output_dir)

    # Load model
    print("=" * 60)
    print("InternVL3.5-1B Steered Evaluation")
    print("=" * 60)
    model_info = load_model("internvl3_5_1b")

    all_results = {}

    # ── Baseline ──
    recs, met = run_pope(model_info, dataset, "baseline", max_n=args.max_samples)
    all_results["baseline"] = {"records": recs, "metrics": met}

    # ── Blind ──
    recs_b, met_b = run_pope(model_info, dataset, "blind", blind=True, max_n=min(args.max_samples, 500))
    all_results["blind"] = {"records": recs_b, "metrics": met_b}

    # ── Steered at different alphas ──
    for alpha in alphas:
        label = f"steered_a{alpha:.0f}"
        hooks = setup_steering(model_info, cal_data, alpha=alpha)
        recs_s, met_s = run_pope(model_info, dataset, label, max_n=args.max_samples)
        all_results[label] = {"records": recs_s, "metrics": met_s}
        cleanup_steering(hooks)

    # ── Steered blind (best alpha) ──
    # Find best alpha
    best_alpha = max(alphas, key=lambda a: all_results[f"steered_a{a:.0f}"]["metrics"]["Overall"]["acc"])
    label_sb = f"steered_a{best_alpha:.0f}_blind"
    hooks = setup_steering(model_info, cal_data, alpha=best_alpha)
    recs_sb, met_sb = run_pope(model_info, dataset, label_sb, blind=True, max_n=min(args.max_samples, 500))
    all_results[label_sb] = {"records": recs_sb, "metrics": met_sb}
    cleanup_steering(hooks)

    # ── Generate plots ──
    generate_comparison_plot(all_results, output_dir)

    # ── Summary ──
    summary = {
        "model": "internvl3_5_1b",
        "timestamp": ts,
        "n_samples": args.max_samples,
        "conditions": {},
    }

    for label, data in all_results.items():
        if "metrics" in data:
            summary["conditions"][label] = data["metrics"]["Overall"]
            with open(output_dir / f"{label}_{ts}.json", "w") as f:
                json.dump({"metrics": data["metrics"]}, f, indent=2)

    # Compute blind gaps
    baseline_acc = summary["conditions"]["baseline"]["acc"]
    blind_acc = summary["conditions"]["blind"]["acc"]
    summary["blind_gap_baseline"] = baseline_acc - blind_acc

    best_steered_label = f"steered_a{best_alpha:.0f}"
    best_steered_blind_label = f"steered_a{best_alpha:.0f}_blind"
    if best_steered_label in summary["conditions"] and best_steered_blind_label in summary["conditions"]:
        s_acc = summary["conditions"][best_steered_label]["acc"]
        sb_acc = summary["conditions"][best_steered_blind_label]["acc"]
        summary["blind_gap_steered"] = s_acc - sb_acc

    with open(output_dir / f"summary_{ts}.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print table
    print(f"\n{'='*80}")
    print(f"InternVL3.5-1B POPE Eval — {args.max_samples} samples")
    print(f"{'='*80}")
    print(f"{'Condition':<25} {'Acc':>7} {'F1':>7} {'P':>7} {'R':>7} {'Unk':>5}")
    print("-" * 80)
    for c, m in summary["conditions"].items():
        print(f"{c:<25} {m['acc']:>6.1f}% {m['f1']:>6.1f}% "
              f"{m['precision']:>6.1f}% {m['recall']:>6.1f}% {m['n_unknown']:>5}")
    print(f"\nBlind Gap (baseline): {summary.get('blind_gap_baseline', 0):.1f}pp")
    if "blind_gap_steered" in summary:
        print(f"Blind Gap (steered):  {summary['blind_gap_steered']:.1f}pp")
    print("=" * 80)

    del model_info
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
