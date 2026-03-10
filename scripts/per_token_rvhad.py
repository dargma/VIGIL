"""
Per-Token R_vhad Analysis — Teacher-Forced Counterfactual

Measures per-token vision head activation differential:
  R_vhad(t_i) = Act(t_i | x_real, T_{<i}) - Act(t_i | x_black, T_{<i})

This is the key measurement for GRPO reward design:
  - WHERE in the thinking chain does vision drift occur?
  - Does steering prevent drift? At what cost?
  - What's the penalty threshold (slope breakpoint)?

Usage:
    PYTHONUNBUFFERED=1 python -u scripts/per_token_rvhad.py \
        --max-samples 20 --alphas 0,5 2>&1 | tee logs/per_token_rvhad.log
"""

import os, sys, json, re, time, argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.steerer import ActivationSteerer
from src.calibrator import CalibrationResult

HF_ID = "Qwen/Qwen3-VL-2B-Thinking"


def split_thinking(text):
    m = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    m = re.search(r'</think>', text)
    if m:
        return text[:m.start()].strip(), text[m.end():].strip()
    return "", text.strip()


def load_pope(max_samples=50):
    from datasets import load_dataset
    print(f"[data] Loading POPE (max {max_samples})...")
    ds = load_dataset("lmms-lab/POPE", split="test", streaming=True)
    samples = []
    for row in ds:
        if len(samples) >= max_samples:
            break
        image = row.get("image")
        if image is None:
            continue
        samples.append({
            "image": image,
            "question": row.get("question", ""),
            "answer": row.get("answer", "").strip().lower(),
            "category": row.get("category", "unknown"),
        })
    print(f"[data] {len(samples)} samples")
    return samples


def load_model():
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    print(f"[model] Loading {HF_ID}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        HF_ID, torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(HF_ID)
    model.eval()
    info = {
        "model": model, "processor": processor,
        "tokenizer": processor.tokenizer,
        "get_layers_fn": lambda: model.model.language_model.layers,
        "num_layers": 28, "num_heads": 16, "num_kv_heads": 8,
        "head_dim": 128, "hidden_size": 2048, "gqa": True,
        "steer_layers_start": 4,
        "device": next(model.parameters()).device,
    }
    return info


def prepare_inputs(model_info, image, prompt):
    """Build model inputs for a given image and prompt."""
    from qwen_vl_utils import process_vision_info
    processor = model_info["processor"]
    content = [{"type": "image", "image": image},
               {"type": "text", "text": prompt}]
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True)
    imgs, _, _ = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(text=[text], images=imgs, return_tensors="pt", padding=True)
    inputs = {k: v.to(model_info["device"]) for k, v in inputs.items()}
    return inputs


def collect_per_token_activations(model_info, inputs, calibration):
    """
    Forward pass with hooks. Collect per-token activation norms
    at vision heads.

    Returns: dict of (layer, head) -> np.array of shape (seq_len,)
    """
    layers = model_info["get_layers_fn"]()
    top_heads = calibration.top_heads[:10]
    num_heads = model_info["num_heads"]
    head_dim = model_info["head_dim"]

    act_norms = {}

    def make_hook(layer_idx):
        def hook_fn(module, args):
            x = args[0] if isinstance(args, tuple) else args
            if x.dim() == 3:
                B, S, D = x.shape
                x_heads = x.view(B, S, num_heads, head_dim)
                for li, hi in top_heads:
                    if li == layer_idx:
                        # Per-token L2 norm for this head
                        norms = x_heads[0, :, hi, :].norm(dim=-1).detach().cpu().numpy()
                        act_norms[(li, hi)] = norms
            return args
        return hook_fn

    hooks = []
    for li, hi in top_heads:
        h = layers[li].self_attn.o_proj.register_forward_pre_hook(make_hook(li))
        hooks.append(h)

    try:
        with torch.no_grad():
            outputs = model_info["model"](**inputs)
    finally:
        for h in hooks:
            h.remove()

    return act_norms


def per_token_rvhad(model_info, calibration, sample, steerer=None):
    """
    Compute per-token R_vhad for a single sample.

    Step 1: Generate response with REAL image → get token sequence T
    Step 2: Teacher-force T with REAL image → collect per-token activations
    Step 3: Teacher-force T with BLACK image → collect per-token activations
    Step 4: R_vhad(t_i) = Act(t_i | real) - Act(t_i | black)

    Returns dict with:
      - thinking_tokens: number of thinking tokens
      - answer_tokens: number of answer tokens
      - per_token_rvhad: list of R_vhad values per token position
      - per_head_rvhad: dict of (L,H) -> list of delta values
      - slope_thinking: slope of R_vhad during thinking
      - slope_answer: slope of R_vhad during answer
    """
    prompt = f"{sample['question']} Please answer yes or no."
    image = sample["image"]

    # Step 1: Generate response with real image
    inputs_gen = prepare_inputs(model_info, image, prompt)
    with torch.no_grad():
        gen = model_info["model"].generate(
            **inputs_gen, max_new_tokens=2048,
            temperature=1.0, top_p=0.95, top_k=20, do_sample=True)

    generated_ids = gen[0][inputs_gen["input_ids"].shape[1]:]  # only new tokens
    raw = model_info["processor"].tokenizer.decode(generated_ids, skip_special_tokens=False)
    for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
        raw = raw.replace(tok, "")

    thinking, answer = split_thinking(raw)

    # Find </think> token position in generated sequence
    think_end_id = model_info["tokenizer"].encode("</think>", add_special_tokens=False)
    gen_list = generated_ids.tolist()

    # Approximate: count tokens before </think>
    think_token_count = 0
    for i in range(len(gen_list)):
        # Check if this is the </think> token
        if gen_list[i] in think_end_id:
            think_token_count = i
            break

    if think_token_count == 0:
        think_token_count = len(gen_list) // 2  # fallback

    # Step 2: Teacher-force with REAL image
    # Concat prompt + generated tokens → forward pass
    full_ids_real = gen[0].unsqueeze(0)  # [1, prompt_len + gen_len]

    # Build attention mask
    real_inputs = {
        "input_ids": full_ids_real,
        "attention_mask": torch.ones_like(full_ids_real),
    }
    # Need to include image features
    real_inputs_full = prepare_inputs(model_info, image, prompt)
    # Replace input_ids with full sequence (prompt + generated)
    prompt_len = real_inputs_full["input_ids"].shape[1]
    real_inputs_full["input_ids"] = full_ids_real
    real_inputs_full["attention_mask"] = torch.ones_like(full_ids_real)

    act_real = collect_per_token_activations(model_info, real_inputs_full, calibration)

    # Step 3: Teacher-force with BLACK image
    black_image = Image.new('RGB', image.size, (0, 0, 0))
    black_inputs = prepare_inputs(model_info, black_image, prompt)
    black_prompt_len = black_inputs["input_ids"].shape[1]

    # Append generated tokens to black prompt
    # Note: prompt_len might differ slightly due to image token count
    # Use the generated token IDs appended to black prompt
    black_full_ids = torch.cat([
        black_inputs["input_ids"],
        generated_ids.unsqueeze(0).to(black_inputs["input_ids"].device)
    ], dim=1)
    black_inputs["input_ids"] = black_full_ids
    black_inputs["attention_mask"] = torch.ones_like(black_full_ids)

    act_black = collect_per_token_activations(model_info, black_inputs, calibration)

    # Step 4: Compute per-token R_vhad
    per_head_rvhad = {}
    head_weights = {}

    for (li, hi) in calibration.top_heads[:10]:
        key = (li, hi)
        if key in act_real and key in act_black:
            real_norms = act_real[key]
            black_norms = act_black[key]

            # Align: only compare the generated token positions
            # Real: positions [prompt_len:] correspond to generated tokens
            # Black: positions [black_prompt_len:] correspond to generated tokens
            real_gen = real_norms[prompt_len:]
            black_gen = black_norms[black_prompt_len:]

            # Take minimum length
            min_len = min(len(real_gen), len(black_gen))
            if min_len == 0:
                continue

            delta = real_gen[:min_len] - black_gen[:min_len]
            per_head_rvhad[f"L{li}H{hi}"] = delta.tolist()

            d = abs(calibration.head_scores.get(key, 1.0))
            head_weights[f"L{li}H{hi}"] = d

    if not per_head_rvhad:
        return None

    # Weighted average across heads
    all_deltas = []
    all_weights = []
    for hname, delta in per_head_rvhad.items():
        w = head_weights.get(hname, 1.0)
        all_deltas.append(np.array(delta) * w)
        all_weights.append(w)

    total_w = sum(all_weights)
    weighted_rvhad = sum(all_deltas) / total_w

    # Compute slopes
    gen_len = len(weighted_rvhad)
    think_end = min(think_token_count, gen_len)

    result = {
        "raw_text": raw[:300],
        "thinking_words": len(thinking.split()) if thinking else 0,
        "gen_tokens": gen_len,
        "think_tokens": think_end,
        "answer_tokens": gen_len - think_end,
        "per_token_rvhad": weighted_rvhad.tolist(),
        "per_head_rvhad": per_head_rvhad,
    }

    # Slopes
    if think_end > 5:
        x = np.arange(think_end)
        slope_think = float(np.polyfit(x, weighted_rvhad[:think_end], 1)[0])
        result["slope_thinking"] = slope_think

    if gen_len - think_end > 5:
        x = np.arange(gen_len - think_end)
        slope_ans = float(np.polyfit(x, weighted_rvhad[think_end:], 1)[0])
        result["slope_answer"] = slope_ans

    # Overall stats
    result["mean_rvhad_thinking"] = float(np.mean(weighted_rvhad[:think_end])) if think_end > 0 else 0
    result["mean_rvhad_answer"] = float(np.mean(weighted_rvhad[think_end:])) if gen_len > think_end else 0
    result["mean_rvhad_overall"] = float(np.mean(weighted_rvhad))

    return result


def plot_rvhad(results, output_dir, alpha_label="α=0"):
    """Plot per-token R_vhad trajectories."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid")
    output_dir = Path(output_dir)

    # Fig 1: Individual trajectories
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Top: All trajectories overlaid
    ax = axes[0]
    n_bins = 50
    binned = []
    for r in results:
        traj = np.array(r["per_token_rvhad"])
        if len(traj) < 5:
            continue
        indices = np.linspace(0, len(traj)-1, n_bins).astype(int)
        binned.append(traj[indices])

    if binned:
        binned = np.stack(binned)
        mean = binned.mean(axis=0)
        std = binned.std(axis=0)
        x = np.linspace(0, 100, n_bins)
        ax.plot(x, mean, "b-", linewidth=2, label="Mean R_vhad")
        ax.fill_between(x, mean-std, mean+std, alpha=0.2, color="blue")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        # Mark approximate thinking/answer boundary
        avg_think_pct = np.mean([r["think_tokens"] / r["gen_tokens"] * 100
                                 for r in results if r["gen_tokens"] > 0])
        ax.axvline(x=avg_think_pct, color="red", linestyle="--", alpha=0.7,
                  label=f"</think> (~{avg_think_pct:.0f}%)")

    ax.set_xlabel("Sequence Position (%)", fontsize=12)
    ax.set_ylabel("R_vhad (real - black activation)", fontsize=12)
    ax.set_title(f"Per-Token R_vhad — {alpha_label}", fontsize=14)
    ax.legend(fontsize=11)

    # Bottom: Slope statistics
    ax = axes[1]
    think_slopes = [r.get("slope_thinking", 0) for r in results if "slope_thinking" in r]
    answer_slopes = [r.get("slope_answer", 0) for r in results if "slope_answer" in r]

    if think_slopes and answer_slopes:
        x_pos = [0, 1]
        means = [np.mean(think_slopes), np.mean(answer_slopes)]
        stds = [np.std(think_slopes), np.std(answer_slopes)]
        colors = ["#4C72B0", "#55A868"]
        bars = ax.bar(x_pos, means, yerr=stds, color=colors, width=0.5, capsize=5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(["Thinking Phase", "Answer Phase"], fontsize=12)
        ax.set_ylabel("R_vhad Slope (drift rate)", fontsize=12)
        ax.set_title("Vision Drift Rate: Thinking vs Answer", fontsize=14)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f"{m:.4f}", ha="center", va="bottom", fontsize=11)

    plt.tight_layout()
    safe_label = alpha_label.replace("=", "").replace(" ", "_")
    fig.savefig(output_dir / f"per_token_rvhad_{safe_label}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved per_token_rvhad_{safe_label}.png")

    # Fig 2: Per-head breakdown
    fig, ax = plt.subplots(figsize=(14, 8))
    head_names = sorted(set(h for r in results for h in r.get("per_head_rvhad", {}).keys()))

    for i, hname in enumerate(head_names[:8]):  # top 8 heads
        all_trajs = []
        for r in results:
            if hname in r.get("per_head_rvhad", {}):
                traj = np.array(r["per_head_rvhad"][hname])
                if len(traj) > 5:
                    indices = np.linspace(0, len(traj)-1, n_bins).astype(int)
                    all_trajs.append(traj[indices])

        if all_trajs:
            stacked = np.stack(all_trajs)
            mean = stacked.mean(axis=0)
            x = np.linspace(0, 100, n_bins)
            ax.plot(x, mean, linewidth=1.5, label=hname, alpha=0.8)

    ax.set_xlabel("Sequence Position (%)", fontsize=12)
    ax.set_ylabel("R_vhad per head", fontsize=12)
    ax.set_title(f"Per-Head R_vhad Trajectories — {alpha_label}", fontsize=14)
    ax.legend(fontsize=9, ncol=2)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(output_dir / f"per_head_rvhad_{safe_label}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved per_head_rvhad_{safe_label}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--alphas", type=str, default="0,5",
                        help="Alphas to test (0=unsteered)")
    parser.add_argument("--output-dir", type=str,
                        default="lab/reports/pope_thinking_steering")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    alphas = [float(a) for a in args.alphas.split(",")]

    samples = load_pope(args.max_samples)
    model_info = load_model()

    pope_cal = Path("checkpoints/calibration/qwen3_vl_2b")
    calibration = CalibrationResult.load(str(pope_cal))
    print(f"[cal] {len(calibration.top_heads)} heads")

    all_results = {}

    for alpha in alphas:
        steerer = None
        if alpha > 0:
            steerer = ActivationSteerer(model_info, calibration, steer_layers_start=4)
            steerer.steer(alpha)

        label = f"α={alpha}"
        print(f"\n{'='*60}")
        print(f"Per-Token R_vhad: {label}")
        print(f"{'='*60}")

        results = []
        for i, s in enumerate(samples):
            try:
                r = per_token_rvhad(model_info, calibration, s, steerer)
                if r:
                    results.append(r)
                    print(f"  [{i+1}/{len(samples)}] think={r['think_tokens']}tok "
                          f"mean_rvhad={r['mean_rvhad_overall']:.4f} "
                          f"slope_think={r.get('slope_thinking', 'N/A'):.5f}"
                          if isinstance(r.get('slope_thinking'), float) else
                          f"  [{i+1}/{len(samples)}] think={r['think_tokens']}tok "
                          f"mean_rvhad={r['mean_rvhad_overall']:.4f}",
                          flush=True)
            except Exception as e:
                print(f"  [{i+1}] ERR: {e}", flush=True)

        if steerer:
            steerer.cleanup()

        all_results[alpha] = results

        # Save
        safe_results = []
        for r in results:
            safe = {k: v for k, v in r.items() if k != "per_head_rvhad"}
            safe_results.append(safe)
        with open(output_dir / f"per_token_rvhad_a{alpha}_{ts}.json", "w") as f:
            json.dump(safe_results, f, indent=2)

        # Plot
        plot_rvhad(results, output_dir, label)

    # Summary
    print(f"\n{'='*75}")
    print("Per-Token R_vhad Summary")
    print(f"{'='*75}")
    print(f"{'Alpha':>7} {'Mean R_vhad':>12} {'Think Slope':>12} {'Ans Slope':>12} "
          f"{'Think Mean':>12} {'Ans Mean':>12}")
    print("-" * 75)

    for alpha in alphas:
        results = all_results.get(alpha, [])
        if not results:
            continue

        mean_rvhad = np.mean([r["mean_rvhad_overall"] for r in results])
        think_slopes = [r.get("slope_thinking", 0) for r in results if "slope_thinking" in r]
        ans_slopes = [r.get("slope_answer", 0) for r in results if "slope_answer" in r]
        think_means = [r["mean_rvhad_thinking"] for r in results]
        ans_means = [r["mean_rvhad_answer"] for r in results]

        ts_mean = np.mean(think_slopes) if think_slopes else 0
        as_mean = np.mean(ans_slopes) if ans_slopes else 0
        tm = np.mean(think_means)
        am = np.mean(ans_means)

        print(f"{alpha:>7.0f} {mean_rvhad:>12.4f} {ts_mean:>12.5f} {as_mean:>12.5f} "
              f"{tm:>12.4f} {am:>12.4f}")

    # GRPO Design Implications
    print(f"\n{'='*75}")
    print("GRPO R_vhad Reward Design Implications")
    print(f"{'='*75}")

    if 0 in all_results and all_results[0]:
        baseline = all_results[0]
        think_slopes = [r.get("slope_thinking", 0) for r in baseline if "slope_thinking" in r]
        mean_slope = np.mean(think_slopes) if think_slopes else 0

        if mean_slope < -0.001:
            print(f"  Vision drift CONFIRMED: thinking slope = {mean_slope:.5f}")
            print(f"  → R_vhad should penalize negative drift (slope < threshold)")
            print(f"  → Suggested threshold: {mean_slope * 0.5:.5f} (half of observed drift)")
        elif mean_slope > 0.001:
            print(f"  Vision activation INCREASES during thinking: slope = {mean_slope:.5f}")
            print(f"  → Model actively refers back to image during reasoning")
            print(f"  → R_vhad reward: encourage this positive trend")
        else:
            print(f"  Vision activation STABLE: slope ≈ 0")
            print(f"  → R_vhad may not be necessary as primary reward")

    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()
