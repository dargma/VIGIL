"""
In-depth analysis: Why does thinking mode sometimes hurt accuracy?

Analysis 1: Per-sample comparison (thinking correct vs short correct)
Analysis 2: Per-token LSR heatmap (KL divergence at each token position)
Analysis 3: Vision head activation trajectory (decay curve)
Analysis 4: Thinking chain length vs correctness correlation
Analysis 5: Disagreement case deep dive (representative examples)

Outputs:
  lab/reports/thinking_analysis/
    ├── disagreement_cases.json         # Per-sample thinking vs short comparison
    ├── fig1_lsr_heatmap_correct.png    # LSR heatmap for correct samples
    ├── fig2_lsr_heatmap_wrong.png      # LSR heatmap for wrong samples
    ├── fig3_vision_drift_curve.png     # Activation decay: correct vs wrong
    ├── fig4_length_vs_accuracy.png     # Thinking length vs correctness
    ├── fig5_disagreement_examples.png  # Deep dive on flip cases
    ├── fig6_lsr_distribution.png       # LSR distribution: correct vs wrong
    ├── ANALYSIS_REPORT.md              # Full report
"""
import os, sys, json, re, string, time, gc
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from qwen_vl_utils import process_vision_info

HF_ID = "Qwen/Qwen3-VL-2B-Thinking"
OUT_DIR = Path("lab/reports/thinking_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Utility Functions ───

def split_thinking(text):
    m = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if m: return m.group(1).strip(), text[m.end():].strip()
    m = re.search(r'</think>', text)
    if m: return text[:m.start()].strip(), text[m.end():].strip()
    return "", text.strip()

def extract_yes_no(raw):
    _, answer = split_thinking(raw)
    if not answer: answer = raw
    text = answer.strip().lower()
    for p in string.punctuation: text = text.replace(p, " ")
    words = text.split()
    for w in words[:5]:
        if w in ("yes", "true"): return "yes"
        if w in ("no", "false"): return "no"
    if "yes" in words: return "yes"
    if "no" in words: return "no"
    return None

def load_pope(max_per_split=100):
    """Load balanced POPE samples (100 per split = 300 total for analysis)."""
    from datasets import load_dataset
    SPLITS = ["random", "popular", "adversarial"]
    ds = load_dataset("lmms-lab/POPE", split="test", streaming=True)
    per_split = defaultdict(list)
    for row in ds:
        cat = row.get("category", "unknown")
        if cat not in SPLITS: continue
        if len(per_split[cat]) >= max_per_split:
            if all(len(per_split[s]) >= max_per_split for s in SPLITS): break
            continue
        per_split[cat].append({
            "image": row["image"], "question": row["question"],
            "answer": row["answer"].strip().lower(), "category": cat,
            "image_source": row.get("image_source", ""),
        })
    samples = []
    for s in SPLITS: samples.extend(per_split[s])
    print(f"Loaded {len(samples)} POPE samples ({max_per_split} per split)")
    return samples

def load_model():
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    print(f"Loading {HF_ID}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        HF_ID, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()
    processor = AutoProcessor.from_pretrained(HF_ID, trust_remote_code=True)
    return model, processor

def prepare_inputs(processor, image, question, device):
    content = [{"type": "image", "image": image}, {"type": "text", "text": question}]
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=True)
    imgs, _, _ = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(text=[text], images=imgs, return_tensors="pt", padding=True)
    return {k: v.to(device) for k, v in inputs.items()}

def prepare_inputs_no_thinking(processor, image, question, device):
    # Add "Answer directly with yes or no." for short mode
    q = question + " Answer directly with yes or no."
    content = [{"type": "image", "image": image}, {"type": "text", "text": q}]
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    imgs, _, _ = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(text=[text], images=imgs, return_tensors="pt", padding=True)
    return {k: v.to(device) for k, v in inputs.items()}

# ─── Analysis 1: Per-sample comparison ───

def run_comparison(model, processor, samples, device, max_samples=300):
    """Run both thinking and non-thinking on same samples, record per-sample results."""
    results = []
    t0 = time.time()

    for i, s in enumerate(samples[:max_samples]):
        try:
            q = s["question"] + " Please answer yes or no."
            gt = s["answer"]

            # --- Thinking mode ---
            inputs_t = prepare_inputs(processor, s["image"], q, device)
            with torch.no_grad():
                out_t = model.generate(**inputs_t, max_new_tokens=1024, do_sample=False)
            raw_t = processor.tokenizer.decode(out_t[0][inputs_t["input_ids"].shape[1]:], skip_special_tokens=False)
            for tok in ["<|im_end|>","<|endoftext|>","<|im_start|>"]: raw_t = raw_t.replace(tok, "")
            thinking_text, answer_t = split_thinking(raw_t)
            pred_t = extract_yes_no(raw_t)
            thinking_len = len(processor.tokenizer.encode(thinking_text)) if thinking_text else 0

            # --- Short (non-thinking) mode ---
            inputs_s = prepare_inputs_no_thinking(processor, s["image"], q, device)
            with torch.no_grad():
                out_s = model.generate(**inputs_s, max_new_tokens=64, do_sample=False)
            raw_s = processor.tokenizer.decode(out_s[0][inputs_s["input_ids"].shape[1]:], skip_special_tokens=False)
            for tok in ["<|im_end|>","<|endoftext|>","<|im_start|>"]: raw_s = raw_s.replace(tok, "")
            pred_s = extract_yes_no(raw_s)

            result = {
                "idx": i,
                "question": s["question"],
                "category": s["category"],
                "gt": gt,
                "pred_thinking": pred_t,
                "pred_short": pred_s,
                "correct_thinking": pred_t == gt,
                "correct_short": pred_s == gt,
                "thinking_text": thinking_text[:500],  # truncate for storage
                "answer_thinking": answer_t[:200],
                "answer_short": raw_s.strip()[:200],
                "thinking_len_tokens": thinking_len,
            }
            results.append(result)

            if (i+1) % 25 == 0:
                ct = sum(r["correct_thinking"] for r in results)
                cs = sum(r["correct_short"] for r in results)
                n = len(results)
                print(f"  [{i+1}/{max_samples}] thinking={ct/n:.1%} short={cs/n:.1%} ({time.time()-t0:.0f}s)", flush=True)

        except Exception as e:
            print(f"  ERROR at {i}: {e}", flush=True)

    return results

# ─── Analysis 2: Per-token LSR (KL divergence) ───

def compute_per_token_lsr(model, processor, sample, device):
    """Compute KL(P_real || P_black) at each generated token position during thinking."""
    q = sample["question"] + " Please answer yes or no."

    # Generate with thinking (real image)
    inputs_real = prepare_inputs(processor, sample["image"], q, device)
    with torch.no_grad():
        out = model.generate(**inputs_real, max_new_tokens=1024, do_sample=False,
                            return_dict_in_generate=True, output_scores=False)
    gen_ids = out.sequences[0][inputs_real["input_ids"].shape[1]:]

    # Now do teacher-forcing: feed generated tokens and get logits for real vs black
    full_ids_real = out.sequences  # [1, prompt_len + gen_len]

    with torch.no_grad():
        logits_real = model(**{**inputs_real, "input_ids": full_ids_real}).logits

    # Black image
    black = Image.new('RGB', sample["image"].size, (0,0,0))
    inputs_black = prepare_inputs(processor, black, q, device)

    # Use same generated tokens with black image
    prompt_len_black = inputs_black["input_ids"].shape[1]
    prompt_len_real = inputs_real["input_ids"].shape[1]

    # Reconstruct full sequence for black: black_prompt + gen_ids
    full_ids_black = torch.cat([inputs_black["input_ids"], gen_ids.unsqueeze(0)], dim=1)

    # Need to adjust attention mask and position ids for black
    try:
        inputs_black_full = {k: v for k, v in inputs_black.items()}
        inputs_black_full["input_ids"] = full_ids_black
        # Extend attention mask
        if "attention_mask" in inputs_black_full:
            gen_mask = torch.ones(1, len(gen_ids), device=device, dtype=inputs_black_full["attention_mask"].dtype)
            inputs_black_full["attention_mask"] = torch.cat([inputs_black_full["attention_mask"], gen_mask], dim=1)

        with torch.no_grad():
            logits_black = model(**inputs_black_full).logits
    except Exception as e:
        print(f"  Black forward failed: {e}")
        return None, None, None

    # Compute per-token KL divergence on generated tokens
    # logits_real: [1, prompt+gen, vocab], logits_black: [1, prompt+gen, vocab]
    # We want KL at each generated position
    gen_len = len(gen_ids)
    kl_per_token = []

    for t in range(gen_len):
        pos_real = prompt_len_real + t
        pos_black = prompt_len_black + t

        if pos_real >= logits_real.shape[1] or pos_black >= logits_black.shape[1]:
            break

        log_p_real = torch.log_softmax(logits_real[0, pos_real].float(), dim=-1)
        log_p_black = torch.log_softmax(logits_black[0, pos_black].float(), dim=-1)
        p_real = log_p_real.exp()

        kl = (p_real * (log_p_real - log_p_black)).sum().item()
        kl_per_token.append(max(0, kl))  # clamp negative KL (numerical)

    # Decode tokens for labeling
    tokens = [processor.tokenizer.decode([tid]) for tid in gen_ids[:len(kl_per_token)]]

    # Find think boundary
    raw = processor.tokenizer.decode(gen_ids, skip_special_tokens=False)
    think_end = raw.find("</think>")
    if think_end >= 0:
        think_end_tokens = len(processor.tokenizer.encode(raw[:think_end]))
    else:
        think_end_tokens = len(kl_per_token)

    return kl_per_token, tokens, think_end_tokens

# ─── Analysis 3: Vision Head Activation Trajectory ───

def get_vision_heads():
    """Top-20 vision heads from calibration (Cohen's d ranking)."""
    # From skills/SKILL_calibration_results.md
    return [
        (4, 5), (5, 3), (4, 12), (5, 8), (4, 1),
        (27, 7), (26, 3), (27, 15), (25, 11), (24, 9),
        (26, 14), (27, 1), (25, 6), (24, 13), (26, 8),
        (5, 14), (4, 9), (27, 4), (25, 2), (24, 0),
    ]

class ActivationTracker:
    """Track vision head activations during generation."""

    def __init__(self, model, vision_heads, num_heads=16, head_dim=128):
        self.vision_heads = vision_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self._hooks = []
        self._captured = {}
        self.trajectory = []  # per-step mean activation
        self.per_head_trajectory = defaultdict(list)

        layers = model.model.language_model.layers
        for li, hi in vision_heads:
            if li < len(layers):
                hook = layers[li].self_attn.o_proj.register_forward_pre_hook(
                    self._make_hook(li))
                self._hooks.append(hook)

    def _make_hook(self, layer_idx):
        def hook_fn(module, input):
            if isinstance(input, tuple): input = input[0]
            self._captured[layer_idx] = input.detach()
        return hook_fn

    def record_step(self):
        norms = []
        for (li, hi) in self.vision_heads:
            inp = self._captured.get(li)
            if inp is not None:
                last = inp[0, -1, :].view(self.num_heads, self.head_dim)
                n = last[hi].norm().item()
                norms.append(n)
                self.per_head_trajectory[(li, hi)].append(n)
        if norms:
            self.trajectory.append(float(np.mean(norms)))
        self._captured.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

def generate_with_tracking(model, processor, sample, device, vision_heads):
    """Generate token-by-token with activation tracking."""
    q = sample["question"] + " Please answer yes or no."
    inputs = prepare_inputs(processor, sample["image"], q, device)

    tracker = ActivationTracker(model, vision_heads)

    input_ids = inputs["input_ids"]
    gen_ids = []
    max_new = 512

    try:
        with torch.no_grad():
            for step in range(max_new):
                outputs = model(**{**inputs, "input_ids": input_ids})
                next_token = outputs.logits[0, -1].argmax()
                gen_ids.append(next_token.item())
                tracker.record_step()

                # Append token
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

                # Update attention mask
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = torch.cat([
                        inputs["attention_mask"],
                        torch.ones(1, 1, device=device, dtype=inputs["attention_mask"].dtype)
                    ], dim=1)

                # Check EOS
                if next_token.item() in [processor.tokenizer.eos_token_id,
                                          processor.tokenizer.convert_tokens_to_ids("<|im_end|>")]:
                    break
    finally:
        tracker.remove()

    raw = processor.tokenizer.decode(gen_ids, skip_special_tokens=False)
    return tracker.trajectory, tracker.per_head_trajectory, raw

# ─── Analysis 4 & 5: Plotting ───

def plot_lsr_heatmaps(lsr_data_correct, lsr_data_wrong, out_dir):
    """Plot LSR heatmaps for correct vs wrong thinking samples."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, data, title in [(axes[0], lsr_data_correct, "Correct (Thinking)"),
                             (axes[1], lsr_data_wrong, "Wrong (Thinking)")]:
        if not data:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=14)
            ax.set_title(title)
            continue

        # Pad to same length
        max_len = max(len(d["kl"]) for d in data)
        max_len = min(max_len, 200)  # cap at 200 tokens
        matrix = np.zeros((len(data), max_len))
        for i, d in enumerate(data):
            kl = d["kl"][:max_len]
            matrix[i, :len(kl)] = kl

        # Sort by thinking length
        lengths = [d.get("think_boundary", len(d["kl"])) for d in data]
        order = np.argsort(lengths)
        matrix = matrix[order]

        im = ax.imshow(matrix, aspect='auto', cmap='hot', interpolation='nearest',
                       vmin=0, vmax=np.percentile(matrix[matrix > 0], 95) if matrix.any() else 1)
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Sample (sorted by thinking length)")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="KL(real || black)")

    plt.tight_layout()
    plt.savefig(out_dir / "fig1_lsr_heatmap_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig1_lsr_heatmap_comparison.png")

def plot_drift_curves(trajectories_correct, trajectories_wrong, out_dir):
    """Plot vision head activation trajectories."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for trajs, label, color in [(trajectories_correct, "Correct", "green"),
                                 (trajectories_wrong, "Wrong", "red")]:
        if not trajs:
            continue
        # Normalize to same length (interpolate)
        max_len = max(len(t) for t in trajs)
        normalized = []
        for t in trajs:
            if len(t) < 5: continue
            x_old = np.linspace(0, 1, len(t))
            x_new = np.linspace(0, 1, 100)
            normalized.append(np.interp(x_new, x_old, t))

        if normalized:
            arr = np.array(normalized)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            x = np.linspace(0, 100, 100)
            ax.plot(x, mean, label=f"{label} (n={len(normalized)})", color=color, linewidth=2)
            ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel("Token Position (% of generation)", fontsize=12)
    ax.set_ylabel("Mean Vision Head Activation (L2 norm)", fontsize=12)
    ax.set_title("Vision Head Activation Trajectory: Correct vs Wrong", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fig3_vision_drift_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig3_vision_drift_curve.png")

def plot_length_vs_accuracy(results, out_dir):
    """Plot thinking chain length vs correctness."""
    lengths = [r["thinking_len_tokens"] for r in results if r["pred_thinking"] is not None]
    correct = [r["correct_thinking"] for r in results if r["pred_thinking"] is not None]

    # Bin by length
    bins = [0, 50, 100, 150, 200, 300, 500, 1000]
    bin_labels = ["0-50", "50-100", "100-150", "150-200", "200-300", "300-500", "500+"]
    bin_acc = []
    bin_counts = []

    for i in range(len(bins) - 1):
        mask = [(bins[i] <= l < bins[i+1]) for l in lengths]
        n = sum(mask)
        if n > 0:
            c = sum(c for c, m in zip(correct, mask) if m)
            bin_acc.append(c / n)
            bin_counts.append(n)
        else:
            bin_acc.append(0)
            bin_counts.append(0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    x = range(len(bin_labels))
    bars = ax1.bar(x, [a * 100 for a in bin_acc], color='steelblue', alpha=0.8)
    ax1.set_ylabel("Accuracy (%)", fontsize=12)
    ax1.set_title("Thinking Chain Length vs Accuracy", fontsize=14)
    ax1.axhline(y=sum(correct)/len(correct)*100, color='red', linestyle='--',
                label=f'Overall: {sum(correct)/len(correct)*100:.1f}%')
    ax1.legend()

    for bar, count in zip(bars, bin_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'n={count}', ha='center', fontsize=9)

    ax2.bar(x, bin_counts, color='gray', alpha=0.6)
    ax2.set_ylabel("Sample Count", fontsize=12)
    ax2.set_xlabel("Thinking Chain Length (tokens)", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(bin_labels, rotation=45)

    plt.tight_layout()
    plt.savefig(out_dir / "fig4_length_vs_accuracy.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig4_length_vs_accuracy.png")

def plot_lsr_distribution(lsr_correct, lsr_wrong, out_dir):
    """Plot distribution of mean LSR for correct vs wrong."""
    fig, ax = plt.subplots(figsize=(10, 6))

    if lsr_correct:
        means_c = [np.mean(d["kl"]) for d in lsr_correct if d["kl"]]
        ax.hist(means_c, bins=30, alpha=0.6, color='green', label=f'Correct (n={len(means_c)})', density=True)

    if lsr_wrong:
        means_w = [np.mean(d["kl"]) for d in lsr_wrong if d["kl"]]
        ax.hist(means_w, bins=30, alpha=0.6, color='red', label=f'Wrong (n={len(means_w)})', density=True)

    ax.set_xlabel("Mean KL(P_real || P_black)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("LSR Distribution: Correct vs Wrong Thinking Responses", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "fig6_lsr_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig6_lsr_distribution.png")

def plot_disagreement_examples(disagreements, out_dir):
    """Visualize representative disagreement cases."""
    # Cases where short is right but thinking is wrong
    short_wins = [d for d in disagreements if d["correct_short"] and not d["correct_thinking"]]
    think_wins = [d for d in disagreements if d["correct_thinking"] and not d["correct_short"]]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Table for short wins
    if short_wins[:8]:
        cell_text = []
        for d in short_wins[:8]:
            cell_text.append([
                d["question"][:60] + "..." if len(d["question"]) > 60 else d["question"],
                d["gt"],
                d["pred_short"] or "?",
                d["pred_thinking"] or "?",
                str(d["thinking_len_tokens"]),
                d["category"],
            ])
        table = axes[0].table(cellText=cell_text,
                              colLabels=["Question", "GT", "Short", "Think", "ThinkLen", "Split"],
                              loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
    axes[0].set_title(f"Short Correct, Thinking Wrong (n={len(short_wins)})", fontsize=13, fontweight='bold')
    axes[0].axis('off')

    # Table for think wins
    if think_wins[:8]:
        cell_text = []
        for d in think_wins[:8]:
            cell_text.append([
                d["question"][:60] + "..." if len(d["question"]) > 60 else d["question"],
                d["gt"],
                d["pred_short"] or "?",
                d["pred_thinking"] or "?",
                str(d["thinking_len_tokens"]),
                d["category"],
            ])
        table = axes[1].table(cellText=cell_text,
                              colLabels=["Question", "GT", "Short", "Think", "ThinkLen", "Split"],
                              loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
    axes[1].set_title(f"Thinking Correct, Short Wrong (n={len(think_wins)})", fontsize=13, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(out_dir / "fig5_disagreement_examples.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig5_disagreement_examples.png")

def generate_report(results, lsr_correct, lsr_wrong, disagreements, out_dir):
    """Generate markdown analysis report."""
    total = len(results)
    ct = sum(r["correct_thinking"] for r in results)
    cs = sum(r["correct_short"] for r in results)

    short_wins = [d for d in disagreements if d["correct_short"] and not d["correct_thinking"]]
    think_wins = [d for d in disagreements if d["correct_thinking"] and not d["correct_short"]]
    both_right = sum(1 for r in results if r["correct_thinking"] and r["correct_short"])
    both_wrong = sum(1 for r in results if not r["correct_thinking"] and not r["correct_short"])

    # Per-category breakdown
    cats = defaultdict(lambda: {"t_correct": 0, "s_correct": 0, "total": 0})
    for r in results:
        c = cats[r["category"]]
        c["total"] += 1
        if r["correct_thinking"]: c["t_correct"] += 1
        if r["correct_short"]: c["s_correct"] += 1

    # LSR stats
    lsr_mean_c = np.mean([np.mean(d["kl"]) for d in lsr_correct if d["kl"]]) if lsr_correct else 0
    lsr_mean_w = np.mean([np.mean(d["kl"]) for d in lsr_wrong if d["kl"]]) if lsr_wrong else 0

    # Thinking length stats
    lens_correct = [r["thinking_len_tokens"] for r in results if r["correct_thinking"]]
    lens_wrong = [r["thinking_len_tokens"] for r in results if not r["correct_thinking"] and r["pred_thinking"] is not None]

    report = f"""# Thinking vs Short Answer: In-Depth Analysis

**Date**: 2026-03-14
**Model**: Qwen3-VL-2B-Thinking (HF baseline, no training)
**Samples**: {total} POPE (balanced splits)

---

## 1. Overall Results

| Mode | Accuracy | N |
|------|----------|---|
| Short (non-thinking) | **{cs/total*100:.1f}%** | {total} |
| Thinking (reasoning) | **{ct/total*100:.1f}%** | {total} |
| Delta | **{(ct-cs)/total*100:+.1f}pp** | — |

## 2. Agreement Matrix

|  | Think ✓ | Think ✗ |
|--|---------|---------|
| **Short ✓** | {both_right} ({both_right/total*100:.1f}%) | {len(short_wins)} ({len(short_wins)/total*100:.1f}%) |
| **Short ✗** | {len(think_wins)} ({len(think_wins)/total*100:.1f}%) | {both_wrong} ({both_wrong/total*100:.1f}%) |

**Key insight**: {len(short_wins)} samples where short is right but thinking is wrong.
{len(think_wins)} samples where thinking is right but short is wrong.

## 3. Per-Split Breakdown

| Split | Thinking Acc | Short Acc | Delta |
|-------|-------------|-----------|-------|
"""
    for cat in ["random", "popular", "adversarial"]:
        c = cats[cat]
        if c["total"] > 0:
            ta = c["t_correct"]/c["total"]*100
            sa = c["s_correct"]/c["total"]*100
            report += f"| {cat} | {ta:.1f}% | {sa:.1f}% | {ta-sa:+.1f}pp |\n"

    lsr_ratio = f"{lsr_mean_c/lsr_mean_w:.2f}x" if lsr_mean_w > 0 else "N/A"
    lsr_interp = "Higher LSR for correct → model uses image more when it gets the answer right" if lsr_mean_c > lsr_mean_w else "Similar or lower LSR for correct → image usage doesn't predict correctness"

    report += f"""
## 4. LSR Analysis (KL Divergence)

| Metric | Correct | Wrong | Ratio |
|--------|---------|-------|-------|
| Mean KL(real∥black) | {lsr_mean_c:.4f} | {lsr_mean_w:.4f} | {lsr_ratio} |

**Interpretation**: {lsr_interp}

## 5. Thinking Length Analysis

| Metric | Correct Samples | Wrong Samples |
|--------|----------------|---------------|
| Mean length | {np.mean(lens_correct):.0f} tokens | {np.mean(lens_wrong):.0f} tokens |
| Median length | {np.median(lens_correct):.0f} tokens | {np.median(lens_wrong) if lens_wrong else 0:.0f} tokens |
| Max length | {max(lens_correct) if lens_correct else 0} tokens | {max(lens_wrong) if lens_wrong else 0} tokens |

## 6. Disagreement Case Study

### Cases where SHORT is right but THINKING is wrong:
"""
    for d in short_wins[:5]:
        report += f"""
**Q**: {d["question"]}
- **GT**: {d["gt"]} | **Short**: {d["pred_short"]} ✓ | **Think**: {d["pred_thinking"]} ✗
- **Thinking chain** ({d["thinking_len_tokens"]} tokens): {d["thinking_text"][:300]}...
- **Category**: {d["category"]}
"""

    report += "\n### Cases where THINKING is right but SHORT is wrong:\n"
    for d in think_wins[:5]:
        report += f"""
**Q**: {d["question"]}
- **GT**: {d["gt"]} | **Short**: {d["pred_short"]} ✗ | **Think**: {d["pred_thinking"]} ✓
- **Thinking chain** ({d["thinking_len_tokens"]} tokens): {d["thinking_text"][:300]}...
- **Category**: {d["category"]}
"""

    report += f"""
## 7. Hypothesis: Why Thinking Can Hurt

Based on the data:

1. **Visual Attention Drift (O(1/L))**: Longer thinking chains = more tokens between image encoding and answer = weaker image signal at decision time
2. **Overthinking simple questions**: POPE is binary yes/no. Short mode answers immediately based on visual features. Thinking mode can "reason itself into the wrong answer" by second-guessing
3. **Language prior dominance**: In long chains, language model priors (statistical co-occurrences) gradually override visual evidence
4. **LSR confirms drift**: If wrong answers have lower mean KL, it means the model's logits become less image-dependent during wrong reasoning

## 8. Implications for Training

1. **Self-play DPO**: Use thinking vs short as natural preference pairs for samples where they disagree
2. **Drift penalty**: Add R_drift = -slope(activation trajectory) to reward function
3. **Length-gated LSR**: Weight LSR reward higher for later tokens in thinking chain (where drift is worst)
4. **Selective thinking**: Train model to use thinking only when beneficial (complex questions), not for simple ones
"""

    with open(out_dir / "ANALYSIS_REPORT.md", "w") as f:
        f.write(report)
    print("  Saved ANALYSIS_REPORT.md")

# ─── Main ───

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--lsr-samples", type=int, default=50, help="Samples for LSR analysis (slow)")
    parser.add_argument("--drift-samples", type=int, default=20, help="Samples for drift tracking (very slow)")
    parser.add_argument("--skip-lsr", action="store_true")
    parser.add_argument("--skip-drift", action="store_true")
    args = parser.parse_args()

    samples = load_pope(args.max_samples // 3)
    model, processor = load_model()
    device = next(model.parameters()).device

    # ── Analysis 1: Per-sample comparison ──
    print("\n=== ANALYSIS 1: Thinking vs Short Comparison ===")
    results = run_comparison(model, processor, samples, device, args.max_samples)

    # Save raw results
    with open(OUT_DIR / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Identify disagreements
    disagreements = [r for r in results if r["correct_thinking"] != r["correct_short"]]
    print(f"\n  Disagreements: {len(disagreements)}/{len(results)} ({len(disagreements)/len(results)*100:.1f}%)")

    short_wins = [d for d in disagreements if d["correct_short"] and not d["correct_thinking"]]
    think_wins = [d for d in disagreements if d["correct_thinking"] and not d["correct_short"]]
    print(f"  Short wins: {len(short_wins)}, Think wins: {len(think_wins)}")

    # Plot disagreement table
    plot_disagreement_examples(disagreements, OUT_DIR)

    # Plot length vs accuracy
    plot_length_vs_accuracy(results, OUT_DIR)

    # ── Analysis 2: Per-token LSR ──
    lsr_correct = []
    lsr_wrong = []

    if not args.skip_lsr:
        print(f"\n=== ANALYSIS 2: Per-token LSR ({args.lsr_samples} samples) ===")
        # Select balanced correct/wrong samples
        correct_samples = [(i, r) for i, r in enumerate(results) if r["correct_thinking"]]
        wrong_samples = [(i, r) for i, r in enumerate(results) if not r["correct_thinking"] and r["pred_thinking"] is not None]

        n_each = min(args.lsr_samples // 2, len(correct_samples), len(wrong_samples))
        selected = correct_samples[:n_each] + wrong_samples[:n_each]

        for j, (idx, r) in enumerate(selected):
            try:
                kl, tokens, think_boundary = compute_per_token_lsr(model, processor, samples[idx], device)
                if kl is not None:
                    entry = {"kl": kl, "tokens": tokens, "think_boundary": think_boundary,
                             "idx": idx, "correct": r["correct_thinking"]}
                    if r["correct_thinking"]:
                        lsr_correct.append(entry)
                    else:
                        lsr_wrong.append(entry)
                if (j+1) % 10 == 0:
                    print(f"  LSR [{j+1}/{len(selected)}] correct={len(lsr_correct)} wrong={len(lsr_wrong)}", flush=True)
            except Exception as e:
                print(f"  LSR error at {idx}: {e}", flush=True)

        # Plot LSR heatmaps
        plot_lsr_heatmaps(lsr_correct, lsr_wrong, OUT_DIR)
        plot_lsr_distribution(lsr_correct, lsr_wrong, OUT_DIR)

    # ── Analysis 3: Vision head activation trajectory ──
    trajectories_correct = []
    trajectories_wrong = []

    if not args.skip_drift:
        print(f"\n=== ANALYSIS 3: Vision Drift Curves ({args.drift_samples} samples) ===")
        vision_heads = get_vision_heads()

        correct_idxs = [i for i, r in enumerate(results) if r["correct_thinking"]][:args.drift_samples // 2]
        wrong_idxs = [i for i, r in enumerate(results) if not r["correct_thinking"]][:args.drift_samples // 2]

        for j, idx in enumerate(correct_idxs + wrong_idxs):
            try:
                is_correct = results[idx]["correct_thinking"]
                traj, _, _ = generate_with_tracking(model, processor, samples[idx], device, vision_heads)
                if len(traj) > 5:
                    if is_correct:
                        trajectories_correct.append(traj)
                    else:
                        trajectories_wrong.append(traj)
                if (j+1) % 5 == 0:
                    print(f"  Drift [{j+1}/{len(correct_idxs)+len(wrong_idxs)}]", flush=True)
            except Exception as e:
                print(f"  Drift error at {idx}: {e}", flush=True)

        plot_drift_curves(trajectories_correct, trajectories_wrong, OUT_DIR)

    # ── Generate Report ──
    print("\n=== Generating Report ===")
    generate_report(results, lsr_correct, lsr_wrong, disagreements, OUT_DIR)

    # Summary
    total = len(results)
    ct = sum(r["correct_thinking"] for r in results)
    cs = sum(r["correct_short"] for r in results)
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Thinking accuracy: {ct/total*100:.1f}% ({ct}/{total})")
    print(f"Short accuracy:    {cs/total*100:.1f}% ({cs}/{total})")
    print(f"Delta:             {(ct-cs)/total*100:+.1f}pp")
    print(f"Disagreements:     {len(disagreements)} ({len(disagreements)/total*100:.1f}%)")
    print(f"Short wins:        {len(short_wins)}")
    print(f"Think wins:        {len(think_wins)}")
    if lsr_correct:
        print(f"Mean LSR correct:  {np.mean([np.mean(d['kl']) for d in lsr_correct]):.4f}")
    if lsr_wrong:
        print(f"Mean LSR wrong:    {np.mean([np.mean(d['kl']) for d in lsr_wrong]):.4f}")
    print(f"{'='*60}")
    print(f"Full report: {OUT_DIR / 'ANALYSIS_REPORT.md'}")

if __name__ == "__main__":
    main()
