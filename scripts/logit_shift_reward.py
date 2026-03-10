"""
Logit-Shift Reward (LSR) Validation

Core idea: If the image changes but the model's next-token logits don't change,
the model is BLIND (ignoring the image). KL-divergence between logits with
real vs black image = direct measure of image influence on output.

This is MORE robust than activation pattern matching because:
- Logit changes directly affect output (not hackable)
- No need for pattern templates or velocity thresholds
- Single scalar reward per token position

Method:
  1. Real image + teacher-forced tokens → logits P_real
  2. Black image + same tokens → logits P_black
  3. LSR(t_i) = D_KL(P_real(t_i) || P_black(t_i))
  4. Overall LSR = mean over key positions (or weighted by position)

Validation:
  - Do CORRECT rollouts have higher LSR than INCORRECT ones?
  - If yes → LSR is a valid GRPO reward signal

Usage:
    PYTHONUNBUFFERED=1 python -u scripts/logit_shift_reward.py \
        --max-samples 30 2>&1 | tee logs/logit_shift_reward.log
"""

import os, sys, json, re, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

HF_ID = "Qwen/Qwen3-VL-2B-Thinking"


def split_thinking(text):
    m = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    m = re.search(r'</think>', text)
    if m:
        return text[:m.start()].strip(), text[m.end():].strip()
    return "", text.strip()


def extract_yes_no(raw):
    import string
    _, answer = split_thinking(raw)
    if not answer:
        answer = raw
    answer = answer.strip().lower()
    for p in string.punctuation:
        answer = answer.replace(p, " ")
    words = answer.split()
    for w in words[:5]:
        if w in ("yes", "true"): return "yes"
        if w in ("no", "false"): return "no"
    if "yes" in words: return "yes"
    if "no" in words: return "no"
    return None


def load_pope(max_samples=30):
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
    return model, processor


def prepare_inputs(processor, image, prompt, device):
    from qwen_vl_utils import process_vision_info
    content = [{"type": "image", "image": image},
               {"type": "text", "text": prompt}]
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True)
    imgs, _, _ = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(text=[text], images=imgs, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs


def compute_lsr(model, processor, sample, device):
    """
    Compute Logit-Shift Reward for a single sample.

    Steps:
      1. Generate response with real image
      2. Teacher-force with real image → get logits
      3. Teacher-force with black image → get logits
      4. KL-divergence at each token position
    """
    prompt = f"{sample['question']} Please answer yes or no."
    image = sample["image"]
    gt = sample["answer"]

    # Step 1: Generate with real image
    inputs_gen = prepare_inputs(processor, image, prompt, device)
    with torch.no_grad():
        gen = model.generate(
            **inputs_gen, max_new_tokens=512,
            temperature=1.0, top_p=0.95, top_k=20, do_sample=True)

    gen_ids = gen[0][inputs_gen["input_ids"].shape[1]:]
    raw = processor.tokenizer.decode(gen_ids, skip_special_tokens=False)
    for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
        raw = raw.replace(tok, "")

    pred = extract_yes_no(raw)
    is_correct = (pred == gt) if pred else False

    thinking, answer = split_thinking(raw)
    think_words = len(thinking.split()) if thinking else 0

    # Step 2: Teacher-force with REAL image
    real_inputs = prepare_inputs(processor, image, prompt, device)
    prompt_len = real_inputs["input_ids"].shape[1]

    # Append generated tokens
    full_ids = torch.cat([real_inputs["input_ids"],
                         gen_ids.unsqueeze(0).to(device)], dim=1)
    real_inputs["input_ids"] = full_ids
    real_inputs["attention_mask"] = torch.ones_like(full_ids)

    with torch.no_grad():
        out_real = model(**real_inputs)
    logits_real = out_real.logits[0, prompt_len-1:-1, :]  # shifted by 1 for next-token

    # Step 3: Teacher-force with BLACK image
    black_image = Image.new('RGB', image.size, (0, 0, 0))
    black_inputs = prepare_inputs(processor, black_image, prompt, device)
    black_prompt_len = black_inputs["input_ids"].shape[1]

    black_full_ids = torch.cat([black_inputs["input_ids"],
                               gen_ids.unsqueeze(0).to(device)], dim=1)
    black_inputs["input_ids"] = black_full_ids
    black_inputs["attention_mask"] = torch.ones_like(black_full_ids)

    with torch.no_grad():
        out_black = model(**black_inputs)
    logits_black = out_black.logits[0, black_prompt_len-1:-1, :]

    # Step 4: Per-token KL divergence
    # Align lengths (may differ due to image token count)
    min_len = min(logits_real.shape[0], logits_black.shape[0], len(gen_ids))
    if min_len == 0:
        return None

    lr = logits_real[:min_len].float()
    lb = logits_black[:min_len].float()

    # P_real and P_black as probability distributions
    p_real = F.softmax(lr, dim=-1)
    p_black = F.softmax(lb, dim=-1)

    # Per-token KL: D_KL(P_real || P_black)
    # = sum(P_real * log(P_real / P_black))
    kl_per_token = F.kl_div(
        F.log_softmax(lb, dim=-1),
        p_real,
        reduction='none'
    ).sum(dim=-1).cpu().numpy()

    # Also compute reverse KL and symmetric JS divergence
    kl_reverse = F.kl_div(
        F.log_softmax(lr, dim=-1),
        p_black,
        reduction='none'
    ).sum(dim=-1).cpu().numpy()

    # Jensen-Shannon divergence (symmetric, bounded)
    m = 0.5 * (p_real + p_black)
    js_per_token = 0.5 * (
        F.kl_div(torch.log(m + 1e-10), p_real, reduction='none').sum(dim=-1) +
        F.kl_div(torch.log(m + 1e-10), p_black, reduction='none').sum(dim=-1)
    ).cpu().numpy()

    # Find think/answer boundary
    think_end_token = 0
    think_end_ids = processor.tokenizer.encode("</think>", add_special_tokens=False)
    gen_list = gen_ids.tolist()
    for i in range(len(gen_list)):
        if gen_list[i] in think_end_ids:
            think_end_token = i
            break

    result = {
        "is_correct": is_correct,
        "pred": pred,
        "gt": gt,
        "think_words": think_words,
        "gen_tokens": int(min_len),
        "think_end_token": int(think_end_token),

        # Per-token KL divergence
        "kl_per_token": kl_per_token.tolist(),
        "js_per_token": js_per_token.tolist(),

        # Aggregate metrics
        "mean_kl": float(np.mean(kl_per_token)),
        "mean_js": float(np.mean(js_per_token)),
        "max_kl": float(np.max(kl_per_token)),

        # Think vs answer phase
        "mean_kl_thinking": float(np.mean(kl_per_token[:think_end_token]))
            if think_end_token > 0 else 0,
        "mean_kl_answer": float(np.mean(kl_per_token[think_end_token:]))
            if think_end_token < min_len else 0,

        # KL slope (drift of image influence)
        "kl_slope_thinking": float(np.polyfit(
            range(think_end_token), kl_per_token[:think_end_token], 1)[0])
            if think_end_token > 5 else 0,
    }

    return result


def plot_lsr(results, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid")

    correct = [r for r in results if r["is_correct"]]
    incorrect = [r for r in results if not r["is_correct"]]

    # Fig 1: KL distribution: correct vs incorrect
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Mean KL comparison
    ax = axes[0]
    correct_kl = [r["mean_kl"] for r in correct]
    incorrect_kl = [r["mean_kl"] for r in incorrect]
    data = [correct_kl, incorrect_kl]
    bp = ax.boxplot(data, labels=["Correct", "Incorrect"], patch_artist=True,
                   boxprops=dict(facecolor="#55A868", alpha=0.7))
    bp['boxes'][1].set_facecolor("#C44E52")
    ax.set_ylabel("Mean KL Divergence", fontsize=12)
    ax.set_title("LSR: Correct vs Incorrect Rollouts", fontsize=13)

    # Compute effect size
    if correct_kl and incorrect_kl:
        pooled_std = np.sqrt((np.var(correct_kl) + np.var(incorrect_kl)) / 2)
        if pooled_std > 0:
            cohens_d = (np.mean(correct_kl) - np.mean(incorrect_kl)) / pooled_std
            ax.text(0.05, 0.95, f"Cohen's d = {cohens_d:.2f}",
                   transform=ax.transAxes, fontsize=11, va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Per-token KL trajectory
    ax = axes[1]
    n_bins = 30
    for label, group, color in [("Correct", correct, "#55A868"),
                                  ("Incorrect", incorrect, "#C44E52")]:
        if not group:
            continue
        binned = []
        for r in group:
            traj = np.array(r["kl_per_token"])
            if len(traj) < 5:
                continue
            indices = np.linspace(0, len(traj)-1, n_bins).astype(int)
            binned.append(traj[indices])
        if not binned:
            continue
        binned = np.stack(binned)
        mean = binned.mean(axis=0)
        std = binned.std(axis=0)
        x = np.linspace(0, 100, n_bins)
        ax.plot(x, mean, color=color, linewidth=2, label=label)
        ax.fill_between(x, mean-std, mean+std, alpha=0.15, color=color)

    ax.set_xlabel("Sequence Position (%)", fontsize=12)
    ax.set_ylabel("KL Divergence", fontsize=12)
    ax.set_title("Per-Token Logit Shift: Trajectory", fontsize=13)
    ax.legend(fontsize=11)

    # Think vs Answer phase
    ax = axes[2]
    phases = ["Thinking", "Answer"]
    correct_vals = [np.mean([r["mean_kl_thinking"] for r in correct]) if correct else 0,
                    np.mean([r["mean_kl_answer"] for r in correct]) if correct else 0]
    incorrect_vals = [np.mean([r["mean_kl_thinking"] for r in incorrect]) if incorrect else 0,
                      np.mean([r["mean_kl_answer"] for r in incorrect]) if incorrect else 0]

    x = np.arange(len(phases))
    w = 0.35
    ax.bar(x - w/2, correct_vals, w, label="Correct", color="#55A868")
    ax.bar(x + w/2, incorrect_vals, w, label="Incorrect", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(phases, fontsize=12)
    ax.set_ylabel("Mean KL Divergence", fontsize=12)
    ax.set_title("LSR by Phase: Think vs Answer", fontsize=13)
    ax.legend(fontsize=11)

    plt.suptitle("Logit-Shift Reward (LSR) Validation", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / "lsr_validation.png", dpi=150)
    plt.close(fig)
    print(f"  Saved lsr_validation.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=30)
    parser.add_argument("--output-dir", type=str,
                        default="lab/reports/pope_thinking_steering")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    samples = load_pope(args.max_samples)
    model, processor = load_model()
    device = next(model.parameters()).device

    print(f"\n{'='*60}")
    print("Logit-Shift Reward (LSR) Validation")
    print(f"{'='*60}")

    results = []
    t0 = time.time()

    for i, s in enumerate(samples):
        try:
            r = compute_lsr(model, processor, s, device)
            if r:
                results.append(r)
                status = "✓" if r["is_correct"] else "✗"
                print(f"  [{i+1}/{len(samples)}] {status} "
                      f"KL={r['mean_kl']:.4f} "
                      f"KL_think={r['mean_kl_thinking']:.4f} "
                      f"KL_ans={r['mean_kl_answer']:.4f} "
                      f"think={r['think_words']}w",
                      flush=True)
        except Exception as e:
            print(f"  [{i+1}] ERR: {e}", flush=True)

    elapsed = (time.time() - t0) / 60

    # ── Analysis ──
    correct = [r for r in results if r["is_correct"]]
    incorrect = [r for r in results if not r["is_correct"]]

    print(f"\n{'='*75}")
    print("LSR Validation Results")
    print(f"{'='*75}")
    print(f"Samples: {len(results)} ({len(correct)} correct, {len(incorrect)} incorrect)")
    print(f"Time: {elapsed:.1f}m")

    if correct and incorrect:
        c_kl = np.mean([r["mean_kl"] for r in correct])
        i_kl = np.mean([r["mean_kl"] for r in incorrect])
        pooled_std = np.sqrt(
            (np.var([r["mean_kl"] for r in correct]) +
             np.var([r["mean_kl"] for r in incorrect])) / 2)
        d = (c_kl - i_kl) / pooled_std if pooled_std > 0 else 0

        print(f"\n  Correct mean KL:   {c_kl:.4f}")
        print(f"  Incorrect mean KL: {i_kl:.4f}")
        print(f"  Cohen's d:         {d:.3f}")

        # Phase breakdown
        c_think = np.mean([r["mean_kl_thinking"] for r in correct])
        i_think = np.mean([r["mean_kl_thinking"] for r in incorrect])
        c_ans = np.mean([r["mean_kl_answer"] for r in correct])
        i_ans = np.mean([r["mean_kl_answer"] for r in incorrect])

        print(f"\n  {'Phase':<15} {'Correct':>10} {'Incorrect':>10} {'Gap':>10}")
        print(f"  {'-'*45}")
        print(f"  {'Thinking':<15} {c_think:>10.4f} {i_think:>10.4f} {c_think-i_think:>+10.4f}")
        print(f"  {'Answer':<15} {c_ans:>10.4f} {i_ans:>10.4f} {c_ans-i_ans:>+10.4f}")

        print(f"\n{'='*75}")
        print("GRPO LSR Feasibility")
        print(f"{'='*75}")

        if d > 0.5:
            print(f"  ✓ STRONG SIGNAL: Cohen's d = {d:.3f} > 0.5")
            print(f"  → Correct rollouts have {'higher' if c_kl > i_kl else 'lower'} KL")
            print(f"  → LSR is a valid GRPO reward signal!")
            print(f"  → R_LSR = D_KL(P_real || P_black) as soft reward")
        elif d > 0.2:
            print(f"  ~ MODERATE: Cohen's d = {d:.3f}")
            print(f"  → LSR shows some discrimination, but noisy")
            print(f"  → May work as auxiliary reward (w < 0.3)")
        else:
            print(f"  ✗ WEAK: Cohen's d = {d:.3f}")
            print(f"  → LSR doesn't reliably separate correct/incorrect")
            print(f"  → Consider alternative reward signals")

        if c_kl > i_kl:
            print(f"\n  Direction: Correct answers USE images MORE (higher KL)")
            print(f"  → Reward = +KL (encourage image-dependent reasoning)")
        else:
            print(f"\n  Direction: Incorrect answers have HIGHER KL")
            print(f"  → Model may be 'confused' by images for wrong answers")
            print(f"  → Need more nuanced reward design")

    # Save
    safe_results = [{k: v for k, v in r.items()
                     if k not in ("kl_per_token", "js_per_token")}
                    for r in results]
    with open(output_dir / f"lsr_results_{ts}.json", "w") as f:
        json.dump(safe_results, f, indent=2)

    # Full results with trajectories
    with open(output_dir / f"lsr_trajectories_{ts}.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plots
    plot_lsr(results, output_dir)
    print(f"\nResults: {output_dir}/lsr_results_{ts}.json")


if __name__ == "__main__":
    main()
