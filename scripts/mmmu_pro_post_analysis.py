"""
MMMU-Pro Steering Post-Analysis

Run AFTER eval_mmmu_pro_steering.py completes. Generates:
  1. Vision drift analysis: per-position activation Δ along the thinking chain
  2. Heatmap: layer × head activation change heatmap (steered vs unsteered)
  3. Vision-only vs all-heads steering comparison
  4. GRPO feasibility evidence plots

Usage:
    python scripts/mmmu_pro_post_analysis.py \
        --results-dir lab/reports/mmmu_pro_steering \
        --max-samples 20 --alphas 0,3,5
"""

import os, sys, json, re, gc, time, argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.steerer import SteeringHook, ActivationSteerer
from src.calibrator import CalibrationResult
from src.profiler import VisionHeadProfiler


# ─── Constants ──────────────────────────────────────────────────────────────

GEN_KWARGS = dict(
    max_new_tokens=4096, temperature=1.0,
    top_p=0.95, top_k=20, do_sample=True,
)


# ─── Shared utilities from steering script ──────────────────────────────────

def split_thinking(text):
    m = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    m = re.search(r'</think>', text)
    if m:
        return text[:m.start()].strip(), text[m.end():].strip()
    return "", text.strip()


def extract_mc_answer(raw_output, num_choices=10):
    _, answer_text = split_thinking(raw_output)
    if not answer_text:
        answer_text = raw_output
    valid = [chr(65 + i) for i in range(num_choices)]
    for pat in [
        r'(?:the\s+)?answer\s+is\s*[:\s]*\(?([A-J])\)?',
        r'(?:answer|option|choice)\s*[:\s]*\(?([A-J])\)?',
        r'\*\*([A-J])\*\*',
        r'^\s*\(?([A-J])\)?[\.\s]*$',
    ]:
        m = re.search(pat, answer_text, re.IGNORECASE | re.MULTILINE)
        if m and m.group(1).upper() in valid:
            return m.group(1).upper()
    found = re.findall(r'\b([A-J])\b', answer_text)
    for letter in reversed(found):
        if letter in valid:
            return letter
    return None


def load_mmmu_pro():
    from datasets import load_from_disk
    datasets = {}
    for name, path in [
        ("standard-10", "eval/mmmu_pro_standard10"),
        ("vision", "eval/mmmu_pro_vision"),
    ]:
        full = Path("data") / path
        if full.exists():
            datasets[name] = load_from_disk(str(full))
    return datasets


def parse_options(row):
    opts = row.get("options", [])
    if isinstance(opts, str):
        try: opts = json.loads(opts)
        except: opts = eval(opts) if opts.startswith("[") else []
    return opts if isinstance(opts, list) else []


def get_image(row):
    for key in ["image", "image_1", "question_image"]:
        img = row.get(key)
        if isinstance(img, Image.Image):
            return img
    return None


def load_model():
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    hf_id = "Qwen/Qwen3-VL-2B-Thinking"
    print(f"[model] Loading {hf_id}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(hf_id)
    model.eval()
    info = {
        "model": model, "processor": processor,
        "tokenizer": processor.tokenizer,
        "get_layers_fn": lambda: model.model.language_model.layers,
        "get_lm_head_fn": lambda: model.lm_head,
        "get_norm_fn": lambda: model.model.language_model.norm,
        "num_layers": 28, "num_heads": 16, "num_kv_heads": 8,
        "head_dim": 128, "hidden_size": 2048, "gqa": True,
        "steer_layers_start": 4,
        "device": next(model.parameters()).device,
    }
    return info


def prepare_inputs(model_info, image, prompt):
    from qwen_vl_utils import process_vision_info
    processor = model_info["processor"]
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True)
    if image is not None:
        imgs, _, _ = process_vision_info(messages, return_video_kwargs=True)
    else:
        imgs = None
    inputs = processor(text=[text], images=imgs, return_tensors="pt", padding=True)
    inputs = {k: v.to(model_info["device"]) for k, v in inputs.items()}
    return inputs


# ─── Analysis 1: Vision Drift Along Sequence ────────────────────────────────

def analyze_vision_drift(model_info, calibration, dataset, max_samples=20, alphas=[0, 5]):
    """
    For each sample, run forward pass with hooks and record per-position
    activation magnitude at vision heads. Compare steered vs unsteered.

    Returns: {alpha: [sample_trajectories]} where each trajectory is
             a dict with position-indexed activation magnitudes.
    """
    print("\n" + "="*60)
    print("Analysis 1: Vision Drift Along Reasoning Sequence")
    print("="*60)

    layers = model_info["get_layers_fn"]()
    top_heads = calibration.top_heads[:10]  # top-10 vision heads
    num_heads = model_info["num_heads"]
    head_dim = model_info["head_dim"]
    device = model_info["device"]

    drift_results = {}

    for alpha in alphas:
        steerer = None
        if alpha > 0:
            steerer = ActivationSteerer(model_info, calibration, steer_layers_start=4)
            steerer.steer(alpha)

        trajectories = []
        for idx in range(min(len(dataset), max_samples)):
            row = dataset[idx]
            image = get_image(row)
            if image is None:
                continue

            options = parse_options(row)
            opts_text = "\n".join(f"{chr(65+j)}. {o}" for j, o in enumerate(options))
            hint = row.get("hint", "") or ""
            hint_text = f"{hint}\n" if hint else ""
            prompt = f"{hint_text}{row.get('question', '')}\nOptions:\n{opts_text}\nPlease select the correct answer from the options above."

            # Collect per-position activations via hooks
            act_records = {}  # (layer, head) -> list of activation norms per position

            def make_hook(layer_idx):
                def hook_fn(module, args):
                    x = args[0] if isinstance(args, tuple) else args
                    if x.dim() == 3:
                        B, S, D = x.shape
                        x_heads = x.view(B, S, num_heads, head_dim)
                        for li, hi in top_heads:
                            if li == layer_idx:
                                # Per-position norm
                                norms = x_heads[0, :, hi, :].norm(dim=-1).detach().cpu().numpy()
                                key = (li, hi)
                                if key not in act_records:
                                    act_records[key] = []
                                act_records[key].append(norms)
                    return args
                return hook_fn

            hooks = []
            for li, hi in top_heads:
                layer = layers[li]
                h = layer.self_attn.o_proj.register_forward_pre_hook(make_hook(li))
                hooks.append(h)

            try:
                inputs = prepare_inputs(model_info, image, prompt)
                with torch.no_grad():
                    gen = model_info["model"].generate(**inputs, **GEN_KWARGS)
            except Exception as e:
                print(f"  [{idx}] ERR: {e}")
                for h in hooks:
                    h.remove()
                continue

            for h in hooks:
                h.remove()

            # Aggregate: average activation norm across vision heads per position
            all_norms = []
            for key, norm_list in act_records.items():
                if norm_list:
                    all_norms.append(norm_list[0])  # first (and likely only) call

            if all_norms:
                # Find common length (min across heads)
                min_len = min(len(n) for n in all_norms)
                stacked = np.stack([n[:min_len] for n in all_norms], axis=0)
                avg_trajectory = stacked.mean(axis=0)
                trajectories.append({
                    "idx": idx,
                    "trajectory": avg_trajectory.tolist(),
                    "seq_len": min_len,
                })

            if (idx + 1) % 5 == 0:
                print(f"  [α={alpha}] {idx+1}/{max_samples} drift samples collected")

        drift_results[alpha] = trajectories
        if steerer:
            steerer.cleanup()

    return drift_results


# ─── Analysis 2: Layer×Head Activation Heatmap ──────────────────────────────

def analyze_activation_heatmap(model_info, calibration, dataset, max_samples=10, alpha=5):
    """
    Compute per-head mean activation norm for:
    - Unsteered (α=0)
    - Steered (α=alpha)
    Returns: heatmap arrays (num_layers × num_heads)
    """
    print("\n" + "="*60)
    print("Analysis 2: Layer×Head Activation Heatmap")
    print("="*60)

    layers = model_info["get_layers_fn"]()
    num_layers = model_info["num_layers"]
    num_heads = model_info["num_heads"]
    head_dim = model_info["head_dim"]

    heatmaps = {}
    for a in [0, alpha]:
        steerer = None
        if a > 0:
            steerer = ActivationSteerer(model_info, calibration, steer_layers_start=4)
            steerer.steer(a)

        accum = np.zeros((num_layers, num_heads))
        counts = np.zeros((num_layers, num_heads))

        for idx in range(min(len(dataset), max_samples)):
            row = dataset[idx]
            image = get_image(row)
            if image is None:
                continue

            options = parse_options(row)
            opts_text = "\n".join(f"{chr(65+j)}. {o}" for j, o in enumerate(options))
            hint = row.get("hint", "") or ""
            hint_text = f"{hint}\n" if hint else ""
            prompt = f"{hint_text}{row.get('question', '')}\nOptions:\n{opts_text}\nPlease select the correct answer from the options above."

            act_data = {}

            def make_hook(layer_idx):
                def hook_fn(module, args):
                    x = args[0] if isinstance(args, tuple) else args
                    if x.dim() == 3:
                        B, S, D = x.shape
                        x_heads = x.view(B, S, num_heads, head_dim)
                        for hi in range(num_heads):
                            norm = x_heads[0, :, hi, :].norm(dim=-1).mean().item()
                            key = (layer_idx, hi)
                            act_data[key] = act_data.get(key, 0) + norm
                    return args
                return hook_fn

            hooks = []
            for li in range(num_layers):
                layer = layers[li]
                h = layer.self_attn.o_proj.register_forward_pre_hook(make_hook(li))
                hooks.append(h)

            try:
                inputs = prepare_inputs(model_info, image, prompt)
                with torch.no_grad():
                    # Forward only (no generation for speed)
                    _ = model_info["model"](**inputs)
            except Exception as e:
                print(f"  [{idx}] ERR: {e}")
                for h in hooks:
                    h.remove()
                continue

            for h in hooks:
                h.remove()

            for (li, hi), norm in act_data.items():
                accum[li, hi] += norm
                counts[li, hi] += 1

            if (idx + 1) % 5 == 0:
                print(f"  [α={a}] {idx+1}/{max_samples} heatmap samples")

        mask = counts > 0
        heatmaps[a] = np.divide(accum, counts, where=mask, out=np.zeros_like(accum))

        if steerer:
            steerer.cleanup()

    # Compute difference heatmap
    heatmaps["diff"] = heatmaps[alpha] - heatmaps[0]

    return heatmaps


# ─── Analysis 3: Vision-Only vs All-Heads Steering ─────────────────────────

def analyze_head_types(model_info, calibration, dataset, max_samples=30, alpha=5):
    """
    Compare steering with:
    1. Vision-only heads (top-K from calibration, high Δ)
    2. All top-K heads (including text-focused ones)
    3. Random heads (control)
    """
    print("\n" + "="*60)
    print("Analysis 3: Vision-Only vs All-Heads Steering")
    print("="*60)

    # Get vision heads (from calibration)
    vision_heads = calibration.top_heads[:20]

    # Get all heads sorted by Cohen's d (including low-Δ ones)
    all_scored = sorted(calibration.head_scores.items(), key=lambda x: abs(x[1]), reverse=True)
    all_heads = [h for h, s in all_scored[:20]]

    # Random heads (control)
    np.random.seed(42)
    random_heads = [(np.random.randint(4, 28), np.random.randint(0, 16)) for _ in range(20)]

    conditions = {
        "baseline": None,
        "vision_only": vision_heads,
        "all_top_k": all_heads,
        "random": random_heads,
    }

    results = {}
    for cond_name, head_list in conditions.items():
        steerer = None
        if head_list is not None:
            # Create a modified calibration with specific heads
            mod_cal = CalibrationResult(
                top_heads=head_list,
                head_scores=calibration.head_scores,
                steering_vectors=calibration.steering_vectors,
                correct_acts=calibration.correct_acts if hasattr(calibration, 'correct_acts') else {},
                incorrect_acts=calibration.incorrect_acts if hasattr(calibration, 'incorrect_acts') else {},
            )
            steerer = ActivationSteerer(model_info, mod_cal, steer_layers_start=4)
            steerer.steer(alpha)

        correct, total = 0, 0
        for idx in range(min(len(dataset), max_samples)):
            row = dataset[idx]
            image = get_image(row)
            if image is None:
                continue

            options = parse_options(row)
            opts_text = "\n".join(f"{chr(65+j)}. {o}" for j, o in enumerate(options))
            hint = row.get("hint", "") or ""
            hint_text = f"{hint}\n" if hint else ""
            prompt = f"{hint_text}{row.get('question', '')}\nOptions:\n{opts_text}\nPlease select the correct answer from the options above."

            try:
                inputs = prepare_inputs(model_info, image, prompt)
                with torch.no_grad():
                    gen = model_info["model"].generate(**inputs, **GEN_KWARGS)
                out = gen[0][inputs["input_ids"].shape[1]:]
                raw = model_info["processor"].tokenizer.decode(out, skip_special_tokens=False)
                for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
                    raw = raw.replace(tok, "")

                gt = str(row.get("answer", "")).strip().upper()
                pred = extract_mc_answer(raw, len(options) if options else 10)
                if pred == gt:
                    correct += 1
                total += 1
            except Exception as e:
                print(f"  [{cond_name}][{idx}] ERR: {e}")
                total += 1

            if (idx + 1) % 10 == 0:
                acc = correct / total * 100 if total > 0 else 0
                print(f"  [{cond_name}] {idx+1}/{max_samples} acc={acc:.1f}%")

        if steerer:
            steerer.cleanup()

        acc = correct / total * 100 if total > 0 else 0
        results[cond_name] = {
            "acc": acc, "correct": correct, "total": total,
            "num_heads": len(head_list) if head_list else 0,
            "heads": [(l, h) for l, h in head_list[:5]] if head_list else [],
        }
        print(f"  → {cond_name}: {acc:.1f}% ({correct}/{total})")

    return results


# ─── Visualization ──────────────────────────────────────────────────────────

def plot_results(drift_results, heatmaps, head_type_results, sweep_results, output_dir):
    """Generate all analysis figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    plt.style.use("seaborn-v0_8-whitegrid")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Fig 1: Vision Drift Curves ──
    if drift_results:
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(drift_results)))
        for (alpha, trajs), color in zip(sorted(drift_results.items()), colors):
            if not trajs:
                continue
            # Normalize trajectories to common length (percentile-based)
            max_len = max(t["seq_len"] for t in trajs)
            n_bins = 50
            binned = np.zeros((len(trajs), n_bins))
            for ti, t in enumerate(trajs):
                traj = np.array(t["trajectory"])
                # Resample to n_bins
                indices = np.linspace(0, len(traj)-1, n_bins).astype(int)
                binned[ti] = traj[indices]
            mean = binned.mean(axis=0)
            std = binned.std(axis=0)
            x = np.linspace(0, 100, n_bins)
            ax.plot(x, mean, label=f"α={alpha}", color=color, linewidth=2)
            ax.fill_between(x, mean-std, mean+std, alpha=0.15, color=color)

        ax.set_xlabel("Sequence Position (%)", fontsize=12)
        ax.set_ylabel("Mean Vision Head Activation Norm", fontsize=12)
        ax.set_title("Vision Attention Drift Along Reasoning Chain", fontsize=14)
        ax.legend(fontsize=11)
        plt.tight_layout()
        fig.savefig(output_dir / "fig1_vision_drift_curves.png", dpi=150)
        plt.close(fig)
        print(f"  Saved fig1_vision_drift_curves.png")

    # ── Fig 2: Activation Heatmap ──
    if heatmaps:
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))

        for ax, (key, title) in zip(axes, [
            (0, "Unsteered (α=0)"),
            (list(heatmaps.keys())[1] if len(heatmaps) > 1 else 0, f"Steered"),
            ("diff", "Difference (Steered - Unsteered)"),
        ]):
            if key not in heatmaps:
                continue
            data = heatmaps[key]
            if key == "diff":
                vmax = max(abs(data.min()), abs(data.max())) or 1
                im = ax.imshow(data, aspect="auto", cmap="RdBu_r",
                              vmin=-vmax, vmax=vmax)
            else:
                im = ax.imshow(data, aspect="auto", cmap="viridis")
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_xlabel("Head Index", fontsize=11)
            ax.set_ylabel("Layer", fontsize=11)
            ax.set_title(title, fontsize=12)

        plt.suptitle("Layer×Head Activation Heatmap", fontsize=14)
        plt.tight_layout()
        fig.savefig(output_dir / "fig2_activation_heatmap.png", dpi=150)
        plt.close(fig)
        print(f"  Saved fig2_activation_heatmap.png")

    # ── Fig 3: Head Type Comparison ──
    if head_type_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        names = list(head_type_results.keys())
        accs = [head_type_results[n]["acc"] for n in names]
        colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
        bars = ax.bar(names, accs, color=colors[:len(names)], width=0.6)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f"{acc:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title("Vision-Only vs All-Heads Steering (MMMU-Pro)", fontsize=14)
        ax.set_ylim(0, max(accs) * 1.15)
        plt.tight_layout()
        fig.savefig(output_dir / "fig3_head_type_comparison.png", dpi=150)
        plt.close(fig)
        print(f"  Saved fig3_head_type_comparison.png")

    # ── Fig 4: Alpha Sweep (from main results) ──
    if sweep_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        alphas_list = sorted(sweep_results.keys())
        accs = [sweep_results[a]["acc"] for a in alphas_list]
        ax.plot(alphas_list, accs, "o-", color="#4C72B0", linewidth=2, markersize=8)
        ax.axhline(y=accs[0], color="gray", linestyle="--", alpha=0.5,
                  label=f"Baseline={accs[0]:.1f}%")
        ax.axhline(y=42.5, color="red", linestyle="--", alpha=0.5,
                  label="Paper=42.5%")
        ax.set_xlabel("Steering Strength (α)", fontsize=12)
        ax.set_ylabel("MMMU-Pro Accuracy (%)", fontsize=12)
        ax.set_title("Steering Alpha Sweep — Qwen3-VL-2B-Thinking", fontsize=14)
        ax.legend(fontsize=11)
        plt.tight_layout()
        fig.savefig(output_dir / "fig4_alpha_sweep.png", dpi=150)
        plt.close(fig)
        print(f"  Saved fig4_alpha_sweep.png")


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="lab/reports/mmmu_pro_steering")
    parser.add_argument("--max-samples-drift", type=int, default=15,
                       help="Samples for vision drift analysis")
    parser.add_argument("--max-samples-heatmap", type=int, default=10,
                       help="Samples for heatmap analysis")
    parser.add_argument("--max-samples-heads", type=int, default=25,
                       help="Samples for head type comparison")
    parser.add_argument("--heatmap-alpha", type=float, default=5,
                       help="Alpha for heatmap steered condition")
    parser.add_argument("--drift-alphas", type=str, default="0,5",
                       help="Alphas for drift analysis")
    parser.add_argument("--head-alpha", type=float, default=5,
                       help="Alpha for head type comparison")
    parser.add_argument("--skip-drift", action="store_true")
    parser.add_argument("--skip-heatmap", action="store_true")
    parser.add_argument("--skip-heads", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load sweep results (from main steering script)
    sweep_results = {}
    for f in sorted(output_dir.glob("a*_standard-10_*.json")):
        try:
            data = json.loads(f.read_text())
            alpha = float(f.name.split("_")[0][1:])
            sweep_results[alpha] = data
        except Exception:
            pass

    # Also try full analysis file
    for f in sorted(output_dir.glob("steering_analysis_*.json")):
        try:
            data = json.loads(f.read_text())
            for k, v in data.items():
                if k.startswith("a") and "standard-10" in k:
                    alpha = float(k.split("_")[0][1:])
                    if alpha not in sweep_results and isinstance(v, dict):
                        sweep_results[alpha] = v
        except Exception:
            pass

    if sweep_results:
        print(f"[results] Loaded sweep results for alphas: {sorted(sweep_results.keys())}")
    else:
        print("[results] No sweep results found yet (will still run analyses)")

    # Load data + model
    mmmu_pro = load_mmmu_pro()
    dataset = mmmu_pro.get("standard-10")
    if dataset is None:
        print("ERROR: No MMMU-Pro standard-10 data found")
        return

    model_info = load_model()

    # Load calibration
    pope_cal = Path("checkpoints/calibration/qwen3_vl_2b")
    if not pope_cal.exists():
        print("ERROR: No POPE calibration found")
        return
    calibration = CalibrationResult.load(str(pope_cal))
    print(f"[cal] Loaded {len(calibration.top_heads)} heads")

    # Run analyses
    drift_results = {}
    heatmaps = {}
    head_type_results = {}

    if not args.skip_drift:
        drift_alphas = [float(a) for a in args.drift_alphas.split(",")]
        drift_results = analyze_vision_drift(
            model_info, calibration, dataset,
            max_samples=args.max_samples_drift, alphas=drift_alphas)
        with open(output_dir / "drift_analysis.json", "w") as f:
            json.dump({str(k): [{"idx": t["idx"], "seq_len": t["seq_len"],
                                  "trajectory": t["trajectory"]}
                                 for t in v]
                       for k, v in drift_results.items()}, f, indent=2)

    if not args.skip_heatmap:
        heatmaps = analyze_activation_heatmap(
            model_info, calibration, dataset,
            max_samples=args.max_samples_heatmap, alpha=args.heatmap_alpha)
        # Save heatmaps as numpy
        np.savez(output_dir / "heatmaps.npz",
                 **{str(k): v for k, v in heatmaps.items()})

    if not args.skip_heads:
        head_type_results = analyze_head_types(
            model_info, calibration, dataset,
            max_samples=args.max_samples_heads, alpha=args.head_alpha)
        with open(output_dir / "head_type_comparison.json", "w") as f:
            json.dump(head_type_results, f, indent=2)

    # Generate plots
    print("\n" + "="*60)
    print("Generating Figures")
    print("="*60)
    plot_results(drift_results, heatmaps, head_type_results, sweep_results, output_dir)

    # Summary
    print("\n" + "="*75)
    print("MMMU-Pro Post-Analysis Summary")
    print("="*75)

    if sweep_results:
        base = sweep_results.get(0, {}).get("acc", 0)
        best_a = max(sweep_results, key=lambda a: sweep_results[a].get("acc", 0))
        best = sweep_results[best_a].get("acc", 0)
        print(f"  Alpha Sweep: baseline={base:.1f}%, best={best:.1f}% (α={best_a}), Δ={best-base:+.1f}pp")

    if drift_results:
        for alpha, trajs in sorted(drift_results.items()):
            if trajs:
                # Compute drift magnitude (slope of trajectory)
                avg_traj = np.mean([t["trajectory"] for t in trajs], axis=0)
                # Fit linear trend
                x = np.arange(len(avg_traj))
                if len(x) > 1:
                    slope = np.polyfit(x, avg_traj, 1)[0]
                    print(f"  Drift α={alpha}: slope={slope:.4f} ({'decaying' if slope < 0 else 'stable/growing'})")

    if head_type_results:
        for name, r in head_type_results.items():
            print(f"  {name}: {r['acc']:.1f}%")

    print(f"\nFigures saved to: {output_dir}/")


if __name__ == "__main__":
    main()
