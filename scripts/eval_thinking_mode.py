"""
VIGIL: Thinking Mode Evaluation + Vision Drift Analysis.

Evaluates Qwen3-VL-2B in thinking mode (extended reasoning) and tracks
vision head activation across token positions to detect attention drift.

3 conditions tested:
1. Baseline (Qwen3-VL-2B-Instruct, thinking enabled)
2. BoN+SFT checkpoint (thinking enabled)
3. Blind test (black images) for both

Produces:
- Figure 1 candidate: token position vs vision head activation curve
- POPE accuracy + Blind Gap in thinking mode
- Drift metrics: decay_ratio, slope, lookback_count
"""

import sys, os, gc, json, time, argparse
from pathlib import Path
from datetime import datetime

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from PIL import Image


TOP_20_VISION_HEADS = [
    (5, 0), (4, 6), (23, 2), (2, 9), (5, 7),
    (11, 2), (2, 6), (8, 3), (2, 8), (4, 1),
    (10, 8), (5, 10), (7, 9), (13, 2), (4, 0),
    (2, 2), (5, 4), (1, 1), (9, 9), (6, 2),
]


def parse_args():
    p = argparse.ArgumentParser(description="VIGIL: Thinking Mode Eval + Drift Analysis")
    p.add_argument("--model-path", type=str, default=None,
                   help="Checkpoint path (default: base Qwen3-VL-2B-Instruct)")
    p.add_argument("--model-label", type=str, default="baseline",
                   help="Label for this model condition (e.g., baseline, bon_r1, bon_r2)")
    p.add_argument("--eval-samples", type=int, default=200)
    p.add_argument("--drift-samples", type=int, default=50)
    p.add_argument("--blind-test-samples", type=int, default=100)
    p.add_argument("--max-thinking-tokens", type=int, default=512,
                   help="Max tokens for thinking chain (longer = more drift data)")
    p.add_argument("--output-dir", type=str, default="lab/reports/thinking_mode")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading (thinking mode)
# ---------------------------------------------------------------------------
def load_model(model_path=None):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    base = "Qwen/Qwen3-VL-2B-Instruct"
    load_path = model_path if model_path else base
    print(f"[model] Loading {load_path} (thinking mode)...")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        load_path, torch_dtype=torch.float16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(base)
    model.eval()

    device = next(model.parameters()).device

    # get_layers_fn for VisionDriftAnalyzer
    def get_layers():
        return model.model.language_model.layers

    model_info = {
        "model": model, "processor": processor, "tokenizer": processor.tokenizer,
        "model_type": "qwen3_vl", "device": device,
        "num_layers": 28, "num_heads": 16, "num_kv_heads": 8,
        "head_dim": 128, "hidden_size": 2048,
        "get_layers_fn": get_layers,
    }
    return model, processor, model_info


def build_inputs(model_info, question, image, thinking=True):
    processor = model_info["processor"]
    messages = [{"role": "user", "content": []}]
    if image is not None:
        messages[0]["content"].append({"type": "image", "image": image})
    messages[0]["content"].append({"type": "text", "text": question})

    # Enable thinking by NOT disabling it — Qwen3 defaults to thinking when enabled
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=thinking,
    )
    inputs = processor(
        text=[text],
        images=[image] if image is not None else None,
        return_tensors="pt", padding=True,
    )
    return {k: v.to(model_info["device"]) for k, v in inputs.items()}


# ---------------------------------------------------------------------------
# Answer extraction from thinking output
# ---------------------------------------------------------------------------
def extract_answer_from_thinking(text):
    """Extract the final answer after </think> tag."""
    if "</think>" in text:
        answer_part = text.split("</think>")[-1].strip()
    else:
        answer_part = text.strip()
    return answer_part


def extract_yesno(text):
    """Extract yes/no from model output (handles thinking tags)."""
    text = extract_answer_from_thinking(text)
    t = text.strip().lower()
    if t.startswith("yes") or t.startswith("no"):
        return "yes" if t.startswith("yes") else "no"
    t50 = t[:50]
    has_yes = "yes" in t50
    has_no = "no" in t50
    if has_yes and has_no:
        return "yes" if t50.index("yes") < t50.index("no") else "no"
    if has_yes: return "yes"
    if has_no: return "no"
    return ""


# ---------------------------------------------------------------------------
# Vision Drift Tracking (step-by-step during generation)
# ---------------------------------------------------------------------------
class StepwiseDriftTracker:
    """Track vision head activations at each generation step.

    Instead of using generate() which batches forward passes,
    we do manual token-by-token generation to capture per-step activations.
    """

    def __init__(self, model_info, vision_heads):
        self.model_info = model_info
        self.vision_heads = vision_heads
        self.num_heads = model_info["num_heads"]
        self.head_dim = model_info["head_dim"]
        self._captured = {}
        self._hooks = []
        self.trajectory = []  # List of mean activation per step

    def install(self):
        self.remove()
        layers = self.model_info["get_layers_fn"]()
        target_layers = set(li for li, _ in self.vision_heads)
        for li in target_layers:
            o_proj = layers[li].self_attn.o_proj

            def make_hook(layer_idx):
                def hook_fn(module, args):
                    self._captured[layer_idx] = args[0].detach()
                return hook_fn

            handle = o_proj.register_forward_pre_hook(make_hook(li))
            self._hooks.append(handle)

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._captured.clear()

    def record_step(self):
        """Record activation norms for all vision heads at current step."""
        norms = []
        for (li, hi) in self.vision_heads:
            inp = self._captured.get(li)
            if inp is None:
                continue
            last = inp[0, -1, :].view(self.num_heads, self.head_dim)
            norms.append(last[hi].norm().item())
        if norms:
            self.trajectory.append(float(np.mean(norms)))
        self._captured.clear()

    def clear(self):
        self.trajectory.clear()
        self._captured.clear()

    def get_trajectory(self):
        return np.array(self.trajectory) if self.trajectory else np.array([])


def generate_with_drift_tracking(model, model_info, inputs, tracker, max_tokens=512):
    """Generate token-by-token while tracking drift at each step."""
    tracker.clear()
    tracker.install()

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")

    generated_ids = []
    past_key_values = None
    cur_attention_mask = attention_mask

    try:
        for step in range(max_tokens):
            with torch.no_grad():
                if step == 0:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=cur_attention_mask,
                        use_cache=True,
                    )
                else:
                    # Extend attention mask for new token
                    cur_attention_mask = torch.cat([
                        cur_attention_mask,
                        torch.ones(1, 1, device=cur_attention_mask.device,
                                   dtype=cur_attention_mask.dtype)
                    ], dim=1)
                    outputs = model(
                        input_ids=next_token.unsqueeze(0),
                        attention_mask=cur_attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

            tracker.record_step()
            past_key_values = outputs.past_key_values

            next_token = outputs.logits[0, -1, :].argmax()
            generated_ids.append(next_token.item())

            # Check for EOS
            eos_id = model_info["tokenizer"].eos_token_id
            if isinstance(eos_id, list):
                if next_token.item() in eos_id:
                    break
            elif next_token.item() == eos_id:
                break

    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
    finally:
        tracker.remove()

    text = model_info["tokenizer"].decode(generated_ids, skip_special_tokens=True)
    return text, tracker.get_trajectory()


# ---------------------------------------------------------------------------
# POPE Eval (thinking mode)
# ---------------------------------------------------------------------------
def eval_pope_thinking(model, model_info, n_samples=200, max_thinking=512):
    from src.data_loader import load_pope
    samples = load_pope("adversarial", limit=n_samples)

    correct = total = yes_count = no_count = 0
    thinking_lengths = []

    for i, s in enumerate(samples):
        try:
            inputs = build_inputs(
                model_info,
                s["question"] + "\nAnswer yes or no only.",
                s.get("image"),
                thinking=True,
            )
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_thinking + 64,  # thinking + answer
                    do_sample=False,
                    enable_thinking=True,
                )
            pred = model_info["tokenizer"].decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False,
            )

            # Track thinking chain length
            if "<think>" in pred and "</think>" in pred:
                think_part = pred.split("<think>")[1].split("</think>")[0]
                think_tokens = len(model_info["tokenizer"].encode(think_part))
                thinking_lengths.append(think_tokens)

            yn = extract_yesno(pred)
            if yn == "yes": yes_count += 1
            elif yn == "no": no_count += 1
            if yn == s["answer"].strip().lower():
                correct += 1
            total += 1

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(samples)}] acc={correct/total*100:.1f}%")
        except Exception as e:
            if i < 3:
                print(f"  Error at {i}: {e}")
            continue

    acc = correct / max(total, 1) * 100
    avg_think_len = float(np.mean(thinking_lengths)) if thinking_lengths else 0
    return {
        "acc": acc, "correct": correct, "total": total,
        "yes": yes_count, "no": no_count,
        "avg_thinking_tokens": avg_think_len,
        "max_thinking_tokens": max(thinking_lengths) if thinking_lengths else 0,
    }


# ---------------------------------------------------------------------------
# Blind Test (thinking mode)
# ---------------------------------------------------------------------------
def eval_blind_thinking(model, model_info, n_samples=100, max_thinking=512):
    from src.data_loader import load_pope
    samples = load_pope("adversarial", limit=n_samples)
    black_img = Image.new("RGB", (448, 448), (0, 0, 0))

    correct_real = correct_blind = total = 0
    for s in samples:
        try:
            prompt = s["question"] + "\nAnswer yes or no only."
            # Real image
            inputs_r = build_inputs(model_info, prompt, s.get("image"), thinking=True)
            with torch.no_grad():
                out_r = model.generate(**inputs_r, max_new_tokens=max_thinking + 64,
                                       do_sample=False, enable_thinking=True)
            pred_r = model_info["tokenizer"].decode(
                out_r[0][inputs_r["input_ids"].shape[1]:], skip_special_tokens=False)

            # Black image
            inputs_b = build_inputs(model_info, prompt, black_img, thinking=True)
            with torch.no_grad():
                out_b = model.generate(**inputs_b, max_new_tokens=max_thinking + 64,
                                       do_sample=False, enable_thinking=True)
            pred_b = model_info["tokenizer"].decode(
                out_b[0][inputs_b["input_ids"].shape[1]:], skip_special_tokens=False)

            gt = s["answer"].strip().lower()
            if extract_yesno(pred_r) == gt: correct_real += 1
            if extract_yesno(pred_b) == gt: correct_blind += 1
            total += 1
        except Exception:
            continue

    acc_r = correct_real / max(total, 1) * 100
    acc_b = correct_blind / max(total, 1) * 100
    return {"acc_real": acc_r, "acc_blind": acc_b, "gap": acc_r - acc_b, "total": total}


# ---------------------------------------------------------------------------
# Vision Drift Analysis
# ---------------------------------------------------------------------------
def run_drift_analysis(model, model_info, n_samples=50, max_thinking=512):
    """Run step-by-step generation and track vision head activation."""
    from src.data_loader import load_pope
    samples = load_pope("adversarial", limit=n_samples)

    tracker = StepwiseDriftTracker(model_info, TOP_20_VISION_HEADS)
    all_trajectories = []
    all_metrics = []

    for i, s in enumerate(samples):
        try:
            inputs = build_inputs(
                model_info,
                s["question"] + "\nAnswer yes or no only.",
                s.get("image"),
                thinking=True,
            )
            text, trajectory = generate_with_drift_tracking(
                model, model_info, inputs, tracker, max_tokens=max_thinking,
            )

            if len(trajectory) >= 4:
                # Compute drift metrics
                mid = len(trajectory) // 2
                first_half = trajectory[:mid].mean()
                second_half = trajectory[mid:].mean()
                decay = float(second_half / (first_half + 1e-8))
                x = np.arange(len(trajectory))
                slope = float(np.polyfit(x, trajectory, 1)[0])

                metrics = {
                    "decay_ratio": decay,
                    "slope": slope,
                    "length": len(trajectory),
                    "mean_activation": float(trajectory.mean()),
                }
                all_metrics.append(metrics)
                all_trajectories.append(trajectory.tolist())

            if (i + 1) % 10 == 0:
                avg_decay = np.mean([m["decay_ratio"] for m in all_metrics]) if all_metrics else 0
                avg_slope = np.mean([m["slope"] for m in all_metrics]) if all_metrics else 0
                print(f"  [drift {i+1}/{n_samples}] decay={avg_decay:.3f}, slope={avg_slope:.6f}")

        except Exception as e:
            if i < 3:
                print(f"  Drift error {i}: {e}")
            gc.collect()
            torch.cuda.empty_cache()
            continue

    summary = {}
    if all_metrics:
        summary = {
            "avg_decay_ratio": float(np.mean([m["decay_ratio"] for m in all_metrics])),
            "avg_slope": float(np.mean([m["slope"] for m in all_metrics])),
            "avg_length": float(np.mean([m["length"] for m in all_metrics])),
            "avg_activation": float(np.mean([m["mean_activation"] for m in all_metrics])),
            "n_valid": len(all_metrics),
        }
        print(f"\n  [drift summary] decay={summary['avg_decay_ratio']:.3f}, "
              f"slope={summary['avg_slope']:.6f}, n={summary['n_valid']}")

    return summary, all_trajectories


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_drift_curves(all_results, output_dir):
    """Generate Figure 1: Vision drift curves across conditions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"baseline": "#2196F3", "bon_r1": "#4CAF50", "bon_r2": "#FF9800"}
    labels = {"baseline": "Baseline (Instruct)", "bon_r1": "BoN+SFT Round 1", "bon_r2": "BoN+SFT Round 2"}

    for label, data in all_results.items():
        trajs = data.get("trajectories", [])
        if not trajs:
            continue

        # Pad/truncate to common length and average
        max_len = max(len(t) for t in trajs)
        padded = np.full((len(trajs), max_len), np.nan)
        for j, t in enumerate(trajs):
            padded[j, :len(t)] = t

        mean_traj = np.nanmean(padded, axis=0)
        std_traj = np.nanstd(padded, axis=0)

        x = np.arange(len(mean_traj))
        color = colors.get(label, "#9E9E9E")
        ax.plot(x, mean_traj, label=labels.get(label, label), color=color, linewidth=2)
        ax.fill_between(x, mean_traj - std_traj, mean_traj + std_traj, alpha=0.15, color=color)

        # Add decay annotation
        if label in data and "avg_decay_ratio" in data[label]:
            decay = data[label]["avg_decay_ratio"]
            ax.annotate(f"decay={decay:.2f}", xy=(len(mean_traj)-1, mean_traj[-1]),
                        fontsize=9, color=color)

    ax.set_xlabel("Token Position in Thinking Chain", fontsize=12)
    ax.set_ylabel("Mean Vision Head Activation (L2 norm)", fontsize=12)
    ax.set_title("Vision Attention Drift During Extended Reasoning", fontsize=14)
    ax.legend(fontsize=11, loc="upper right")
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    path = Path(output_dir) / "fig1_vision_drift_curve.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(path), dpi=150)
    plt.close()
    print(f"  Saved: {path}")
    return str(path)


def plot_thinking_comparison(all_results, output_dir):
    """Bar chart: POPE accuracy + Blind Gap across conditions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    labels = list(all_results.keys())
    display = {"baseline": "Baseline", "bon_r1": "BoN+SFT R1", "bon_r2": "BoN+SFT R2"}

    # POPE accuracy
    accs = [all_results[l].get("pope", {}).get("acc", 0) for l in labels]
    colors = ["#2196F3", "#4CAF50", "#FF9800"][:len(labels)]
    bars = axes[0].bar([display.get(l, l) for l in labels], accs, color=colors[:len(labels)])
    axes[0].set_ylabel("POPE Accuracy (%)", fontsize=12)
    axes[0].set_title("Thinking Mode: POPE Accuracy", fontsize=13)
    for bar, val in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f"{val:.1f}%", ha="center", fontsize=11)
    axes[0].set_ylim(0, 100)

    # Blind Gap
    gaps = [all_results[l].get("blind", {}).get("gap", 0) for l in labels]
    bars = axes[1].bar([display.get(l, l) for l in labels], gaps, color=colors[:len(labels)])
    axes[1].set_ylabel("Blind Test Gap (pp)", fontsize=12)
    axes[1].set_title("Thinking Mode: Blind Test Gap", fontsize=13)
    for bar, val in zip(bars, gaps):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f"{val:.1f}pp", ha="center", fontsize=11)

    plt.tight_layout()
    path = Path(output_dir) / "fig2_thinking_mode_comparison.png"
    plt.savefig(str(path), dpi=150)
    plt.close()
    print(f"  Saved: {path}")
    return str(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"VIGIL: Thinking Mode Eval + Vision Drift Analysis")
    print(f"  model: {args.model_path or 'baseline'}")
    print(f"  label: {args.model_label}")
    print(f"  max_thinking_tokens: {args.max_thinking_tokens}")
    print(f"{'='*60}")

    model, processor, model_info = load_model(args.model_path)

    # 1. POPE eval (thinking mode)
    print(f"\n[1] POPE Evaluation (thinking mode, {args.eval_samples} samples)...")
    pope = eval_pope_thinking(model, model_info, args.eval_samples, args.max_thinking_tokens)
    print(f"  POPE: {pope['acc']:.1f}% | avg_think_tokens: {pope['avg_thinking_tokens']:.0f}")

    # 2. Blind test (thinking mode)
    print(f"\n[2] Blind Test (thinking mode, {args.blind_test_samples} samples)...")
    blind = eval_blind_thinking(model, model_info, args.blind_test_samples, args.max_thinking_tokens)
    print(f"  Real: {blind['acc_real']:.1f}% | Blind: {blind['acc_blind']:.1f}% | Gap: {blind['gap']:.1f}pp")

    # 3. Vision drift analysis
    print(f"\n[3] Vision Drift Analysis ({args.drift_samples} samples)...")
    drift_summary, trajectories = run_drift_analysis(
        model, model_info, args.drift_samples, args.max_thinking_tokens,
    )

    # Save results
    results = {
        "timestamp": ts,
        "model_label": args.model_label,
        "model_path": args.model_path,
        "config": vars(args),
        "pope": pope,
        "blind": blind,
        "drift": drift_summary,
        "trajectories": trajectories,
    }

    results_path = output_dir / f"results_{args.model_label}_{ts}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {results_path}")

    # Load previous results if exist for combined plotting
    all_results = {args.model_label: results}

    # Try loading other conditions
    for prev_file in output_dir.glob("results_*.json"):
        if prev_file == results_path:
            continue
        try:
            with open(prev_file) as f:
                prev = json.load(f)
            prev_label = prev.get("model_label", "unknown")
            if prev_label not in all_results:
                all_results[prev_label] = prev
        except Exception:
            continue

    # Generate plots
    print(f"\n[4] Generating plots...")
    if any(r.get("trajectories") for r in all_results.values()):
        plot_drift_curves(all_results, str(output_dir))
    if len(all_results) >= 1:
        plot_thinking_comparison(all_results, str(output_dir))

    print(f"\n{'='*60}")
    print(f"Done. Label={args.model_label}")
    print(f"  POPE (thinking): {pope['acc']:.1f}%")
    print(f"  Blind Gap (thinking): {blind['gap']:.1f}pp")
    if drift_summary:
        print(f"  Drift decay: {drift_summary['avg_decay_ratio']:.3f}")
        print(f"  Drift slope: {drift_summary['avg_slope']:.6f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
