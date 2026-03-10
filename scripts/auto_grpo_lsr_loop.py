"""
Automated GRPO-LSR Training Loop

Pipeline:
  Round N: GRPO training → Eval (POPE + Blind Gap) → Report → Git push
           → Analyze results → Propose improvements → Apply → Round N+1

Each round produces:
  - checkpoints/grpo_lsr/roundN/      (model checkpoint)
  - lab/reports/grpo_lsr/roundN/       (figures + JSON results)
  - Git commit with results

Usage:
    PYTHONUNBUFFERED=1 python -u scripts/auto_grpo_lsr_loop.py \
        --max-rounds 5 --steps-per-round 50 \
        2>&1 | tee logs/auto_grpo_lsr.log
"""

import os, sys, gc, json, re, time, random, argparse, string, subprocess
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from datetime import datetime
from collections import defaultdict, Counter
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Constants ──────────────────────────────────────────────────────────

HF_ID = "Qwen/Qwen3-VL-2B-Instruct"
POPE_SPLITS = ["random", "popular", "adversarial"]
PROJECT_ROOT = Path(__file__).parent.parent


# ── Answer Extraction ──────────────────────────────────────────────────

def extract_yes_no(raw):
    text = raw.strip().lower()
    for p in string.punctuation:
        text = text.replace(p, " ")
    words = text.split()
    for w in words[:5]:
        if w in ("yes", "true"): return "yes"
        if w in ("no", "false"): return "no"
    if "yes" in words: return "yes"
    if "no" in words: return "no"
    return None


def extract_answer(raw, qtype="short_answer"):
    text = raw.strip()
    if qtype == "yesno":
        return extract_yes_no(text)
    if qtype == "mc":
        for ch in text[:5]:
            if ch.upper() in "ABCDEFGH":
                return ch.upper()
        return text[:20]
    return text.split("\n")[0].strip()[:100]


# ── Data Loading ───────────────────────────────────────────────────────

def load_training_data(limit=500, seed=42):
    from datasets import load_dataset
    rng = random.Random(seed)
    samples = []

    # A-OKVQA (multiple choice — diverse answers)
    print("[data] Loading A-OKVQA train...")
    try:
        ds = load_dataset("HuggingFaceM4/A-OKVQA", split="train")
        for row in ds:
            img = row.get("image")
            if img is None:
                continue
            choices = row.get("choices", [])
            idx = row.get("correct_choice_idx", 0)
            if not choices:
                continue
            ans = choices[idx] if idx < len(choices) else choices[0]
            choice_str = "\n".join(
                f"({chr(65+i)}) {c}" for i, c in enumerate(choices))
            samples.append({
                "question": f"{row['question']}\n{choice_str}\nAnswer with the letter only.",
                "answer": chr(65 + idx),
                "image": img,
                "type": "mc",
                "source": "aokvqa",
            })
    except Exception as e:
        print(f"  A-OKVQA error: {e}")

    # VQAv2 short-answer (non-binary)
    print("[data] Loading VQAv2 train (non-binary)...")
    try:
        ds = load_dataset("HuggingFaceM4/VQAv2", split="train", streaming=True)
        count = 0
        for row in ds:
            img = row.get("image")
            if img is None:
                continue
            answers = row.get("answers", [])
            if not answers:
                continue
            # Get most common answer
            if isinstance(answers[0], dict):
                ans_list = [a["answer"] for a in answers]
            else:
                ans_list = answers
            ans = Counter(ans_list).most_common(1)[0][0]
            if ans.lower() in ("yes", "no"):
                continue
            samples.append({
                "question": row["question"],
                "answer": ans,
                "image": img,
                "type": "short_answer",
                "source": "vqav2",
            })
            count += 1
            if count >= limit:
                break
    except Exception as e:
        print(f"  VQAv2 error: {e}")

    rng.shuffle(samples)
    samples = samples[:limit]
    print(f"[data] {len(samples)} training samples "
          f"({sum(1 for s in samples if s['source']=='aokvqa')} aokvqa, "
          f"{sum(1 for s in samples if s['source']=='vqav2')} vqav2)")
    return samples


def load_pope_eval(max_samples=300):
    from datasets import load_dataset
    ds = load_dataset("lmms-lab/POPE", split="test", streaming=True)
    per_split = defaultdict(list)
    per_sample = max_samples // 3

    for row in ds:
        cat = row.get("category", "unknown")
        if cat not in POPE_SPLITS:
            continue
        if len(per_split[cat]) >= per_sample:
            if all(len(per_split[s]) >= per_sample for s in POPE_SPLITS):
                break
            continue
        per_split[cat].append({
            "image": row["image"],
            "question": row["question"],
            "answer": row["answer"].strip().lower(),
            "category": cat,
        })

    samples = []
    for s in POPE_SPLITS:
        samples.extend(per_split[s])
    return samples


# ── Model Loading ──────────────────────────────────────────────────────

def load_model(model_path=None, for_training=True):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from peft import get_peft_model, LoraConfig, TaskType

    path = model_path or HF_ID
    dtype = torch.bfloat16
    print(f"[model] Loading {path}...")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        path, dtype=dtype, device_map="auto")
    processor = AutoProcessor.from_pretrained(HF_ID, trust_remote_code=True)

    if for_training:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "o_proj", "k_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        model.train()
    else:
        model.eval()

    return model, processor


# ── Input Preparation ──────────────────────────────────────────────────

def prepare_inputs(processor, image, question, device):
    from qwen_vl_utils import process_vision_info
    content = [{"type": "image", "image": image},
               {"type": "text", "text": question}]
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False)
    imgs, _, _ = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(text=[text], images=imgs, return_tensors="pt",
                       padding=True)
    return {k: v.to(device) for k, v in inputs.items()}


# ── Candidate Generation ──────────────────────────────────────────────

def generate_candidates(model, processor, sample, group_size, temperature,
                        top_p, max_new_tokens, device):
    question = sample["question"]
    image = sample["image"]
    inputs = prepare_inputs(processor, image, question, device)
    prompt_len = inputs["input_ids"].shape[1]

    candidates = []
    candidate_ids_list = []

    for _ in range(group_size):
        try:
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    temperature=temperature, top_p=top_p, do_sample=True)
            gen_ids = out[0][prompt_len:]
            text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
            candidates.append(text.strip())
            candidate_ids_list.append(gen_ids.detach())
        except Exception:
            candidates.append("")
            candidate_ids_list.append(
                torch.tensor([], dtype=torch.long, device=device))

    return candidates, candidate_ids_list, prompt_len, inputs


# ── LSR Computation ────────────────────────────────────────────────────

def compute_lsr_for_candidate(model, processor, sample, candidate_ids,
                               prompt_len, device):
    if candidate_ids.numel() == 0:
        return 0.0

    image = sample["image"]
    question = sample["question"]

    # Teacher-force real
    real_inputs = prepare_inputs(processor, image, question, device)
    rpl = real_inputs["input_ids"].shape[1]
    rf = torch.cat([real_inputs["input_ids"],
                    candidate_ids.unsqueeze(0)], dim=1)
    real_inputs["input_ids"] = rf
    real_inputs["attention_mask"] = torch.ones_like(rf)
    with torch.no_grad():
        lr = model(**real_inputs).logits[0]

    # Teacher-force black
    black_image = Image.new('RGB', image.size, (0, 0, 0))
    black_inputs = prepare_inputs(processor, black_image, question, device)
    bpl = black_inputs["input_ids"].shape[1]
    bf = torch.cat([black_inputs["input_ids"],
                    candidate_ids.unsqueeze(0)], dim=1)
    black_inputs["input_ids"] = bf
    black_inputs["attention_mask"] = torch.ones_like(bf)
    with torch.no_grad():
        lb = model(**black_inputs).logits[0]

    # Extract at generated positions
    lr = lr[rpl - 1: rpl - 1 + len(candidate_ids)]
    lb = lb[bpl - 1: bpl - 1 + len(candidate_ids)]
    ml = min(lr.shape[0], lb.shape[0])
    if ml == 0:
        return 0.0

    kl = F.kl_div(
        F.log_softmax(lb[:ml].float(), dim=-1),
        F.softmax(lr[:ml].float(), dim=-1),
        reduction='none'
    ).sum(dim=-1)

    result = kl.mean().item()
    del lr, lb, kl, real_inputs, black_inputs
    return result


# ── Reward ─────────────────────────────────────────────────────────────

def compute_r_correct(prediction, ground_truth, qtype="short_answer"):
    if not prediction:
        return 0.0
    pred = prediction.strip().lower()
    gt = ground_truth.strip().lower()
    if qtype == "yesno":
        p = extract_yes_no(prediction)
        return 1.0 if p == gt else 0.0
    if qtype == "mc":
        return 1.0 if pred[:1] == gt[:1].lower() else 0.0
    pred_tokens = set(pred.split())
    gt_tokens = set(gt.split())
    if not gt_tokens:
        return 0.0
    overlap = pred_tokens & gt_tokens
    if not overlap:
        return 0.0
    p = len(overlap) / len(pred_tokens)
    r = len(overlap) / len(gt_tokens)
    return 2 * p * r / (p + r)


def compute_r_fluency(text, max_tokens=64):
    if not text or len(text) < 3 or len(set(text)) < 3:
        return 0.0
    words = text.split()
    if len(words) > max_tokens * 2:
        return max(0, 1.0 - (len(words) - max_tokens) / max_tokens * 0.5)
    return 1.0


def compute_rewards(model, processor, sample, candidates, cand_ids_list,
                    prompt_len, device, cfg):
    rewards = []
    details = []
    gt = sample["answer"]
    qtype = sample.get("type", "short_answer")

    for cand, cand_ids in zip(candidates, cand_ids_list):
        pred = extract_answer(cand, qtype)
        r_correct = compute_r_correct(pred, gt, qtype)
        r_fluency = compute_r_fluency(cand)

        try:
            r_lsr_raw = compute_lsr_for_candidate(
                model, processor, sample, cand_ids, prompt_len, device)
            r_lsr = min(r_lsr_raw / cfg["lsr_scale"], 1.0)
        except Exception:
            r_lsr_raw, r_lsr = 0.0, 0.0

        r_total = (cfg["w_correct"] * r_correct +
                   cfg["w_lsr"] * r_lsr +
                   cfg["w_fluency"] * r_fluency)

        rewards.append(r_total)
        details.append({
            "pred": pred, "gt": gt, "correct": r_correct,
            "lsr_raw": r_lsr_raw, "lsr_norm": r_lsr,
            "fluency": r_fluency, "total": r_total,
        })

    return rewards, details


# ── Log-Prob & GRPO Loss ───────────────────────────────────────────────

def compute_logprobs(model, inputs, candidate_ids, prompt_len):
    if candidate_ids.numel() == 0:
        return torch.tensor(0.0, device=candidate_ids.device), torch.tensor(0.0)

    full_ids = torch.cat([inputs["input_ids"][:, :prompt_len],
                          candidate_ids.unsqueeze(0)], dim=1)
    attn = torch.ones_like(full_ids)
    fwd = {k: v for k, v in inputs.items()
           if k not in ("input_ids", "attention_mask")}
    fwd["input_ids"] = full_ids
    fwd["attention_mask"] = attn

    out = model(**fwd)
    logits = out.logits[0, prompt_len - 1: prompt_len - 1 + len(candidate_ids)]
    log_probs = F.log_softmax(logits, dim=-1)
    token_lp = log_probs.gather(1, candidate_ids.unsqueeze(1)).squeeze(1)
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    return token_lp, entropy


def compute_ref_logprobs(model, inputs, candidate_ids, prompt_len):
    model.disable_adapter_layers()
    with torch.no_grad():
        lp, _ = compute_logprobs(model, inputs, candidate_ids, prompt_len)
    model.enable_adapter_layers()
    return lp.detach()


def compute_grpo_loss(model, inputs, cand_ids_list, prompt_len,
                      advantages, cfg):
    total_loss = torch.tensor(0.0, device=next(model.parameters()).device,
                              requires_grad=True)
    n_valid = 0
    stats = {"kl": [], "entropy": [], "ratio": []}

    for cand_ids, adv in zip(cand_ids_list, advantages):
        if cand_ids.numel() == 0 or abs(adv) < 1e-8:
            continue

        cur_lp, entropy = compute_logprobs(model, inputs, cand_ids, prompt_len)
        ref_lp = compute_ref_logprobs(model, inputs, cand_ids, prompt_len)

        kl = (torch.exp(ref_lp) * (ref_lp - cur_lp)).mean()
        if kl.item() > 10.0:
            continue

        log_ratio = (cur_lp - ref_lp).sum()
        ratio = torch.exp(log_ratio.clamp(-5, 5))
        clipped = torch.clamp(ratio, 1 - cfg["epsilon"], 1 + cfg["epsilon"])
        surr = torch.min(ratio * adv, clipped * adv)
        loss = -surr + cfg["beta_kl"] * kl - cfg["beta_entropy"] * entropy
        total_loss = total_loss + loss
        n_valid += 1

        stats["kl"].append(kl.item())
        stats["entropy"].append(entropy.item())
        stats["ratio"].append(ratio.item())

    if n_valid > 0:
        total_loss = total_loss / n_valid
    return total_loss, stats


# ── Evaluation ─────────────────────────────────────────────────────────

def evaluate_pope(model, processor, samples, device):
    model.eval()
    correct = total = yes_count = 0
    per_split = defaultdict(lambda: {"correct": 0, "total": 0})

    for s in samples:
        try:
            inputs = prepare_inputs(
                processor, s["image"],
                s["question"] + " Please answer yes or no.", device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=32,
                                     do_sample=False)
            gen = out[0][inputs["input_ids"].shape[1]:]
            text = processor.tokenizer.decode(gen, skip_special_tokens=True)
            pred = extract_yes_no(text)
            gt = s["answer"].strip().lower()
            cat = s.get("category", "unknown")

            is_correct = (pred == gt)
            if is_correct:
                correct += 1
                per_split[cat]["correct"] += 1
            if pred == "yes":
                yes_count += 1
            total += 1
            per_split[cat]["total"] += 1
        except Exception:
            total += 1

    model.train()
    acc = correct / total if total > 0 else 0
    split_accs = {}
    for cat, d in per_split.items():
        split_accs[cat] = d["correct"] / d["total"] if d["total"] > 0 else 0

    return {
        "acc": acc, "total": total,
        "yes_rate": yes_count / total if total > 0 else 0,
        "per_split": split_accs,
    }


def evaluate_blind(model, processor, samples, device, n=100):
    model.eval()
    real_correct = blind_correct = total = 0

    for s in samples[:n]:
        try:
            q = s["question"] + " Please answer yes or no."
            gt = s["answer"].strip().lower()

            inputs = prepare_inputs(processor, s["image"], q, device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=32,
                                     do_sample=False)
            pred_real = extract_yes_no(
                processor.tokenizer.decode(
                    out[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True))
            if pred_real == gt:
                real_correct += 1

            black = Image.new('RGB', s["image"].size, (0, 0, 0))
            inputs_b = prepare_inputs(processor, black, q, device)
            with torch.no_grad():
                out_b = model.generate(**inputs_b, max_new_tokens=32,
                                       do_sample=False)
            pred_blind = extract_yes_no(
                processor.tokenizer.decode(
                    out_b[0][inputs_b["input_ids"].shape[1]:],
                    skip_special_tokens=True))
            if pred_blind == gt:
                blind_correct += 1
            total += 1
        except Exception:
            total += 1

    model.train()
    ra = real_correct / total if total > 0 else 0
    ba = blind_correct / total if total > 0 else 0
    return {"real_acc": ra, "blind_acc": ba, "gap": ra - ba, "total": total}


# ── Report Generation ──────────────────────────────────────────────────

def generate_report(round_num, history, report_dir):
    """Generate markdown report + plots for a round."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid")

    report_dir.mkdir(parents=True, exist_ok=True)

    # Extract step-level data
    steps = [s for s in history["steps"] if not s.get("skipped")]
    if not steps:
        return

    step_nums = [s["step"] for s in steps]
    losses = [s["loss"] for s in steps]
    rewards = [s["mean_reward"] for s in steps]
    lsr_vals = [s.get("mean_lsr_raw", 0) for s in steps]
    correct_vals = [s.get("mean_correct", 0) for s in steps]

    # Fig 1: Training dynamics (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(step_nums, losses, 'b-', linewidth=1.5)
    axes[0, 0].set_ylabel("Loss", fontsize=12)
    axes[0, 0].set_title("GRPO Loss", fontsize=13)

    axes[0, 1].plot(step_nums, rewards, 'g-', linewidth=1.5)
    axes[0, 1].set_ylabel("Mean Reward", fontsize=12)
    axes[0, 1].set_title("Mean Reward", fontsize=13)

    axes[1, 0].plot(step_nums, lsr_vals, 'r-', linewidth=1.5)
    axes[1, 0].set_ylabel("Mean LSR (raw KL)", fontsize=12)
    axes[1, 0].set_xlabel("Step", fontsize=12)
    axes[1, 0].set_title("LSR (Image Influence)", fontsize=13)

    axes[1, 1].plot(step_nums, correct_vals, 'm-', linewidth=1.5)
    axes[1, 1].set_ylabel("Correctness Rate", fontsize=12)
    axes[1, 1].set_xlabel("Step", fontsize=12)
    axes[1, 1].set_title("Training Correctness", fontsize=13)

    plt.suptitle(f"GRPO-LSR Round {round_num}: Training Dynamics", fontsize=14)
    plt.tight_layout()
    fig.savefig(report_dir / "training_dynamics.png", dpi=150)
    plt.close(fig)

    # Fig 2: Eval progression (if multiple evals)
    evals = history.get("evals", [])
    if evals:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        eval_steps = [e["step"] for e in evals]
        pope_accs = [e["pope"]["acc"] * 100 for e in evals]
        gaps = [e["blind"]["gap"] * 100 for e in evals]

        # Add pre-training point
        pre = history.get("pre_eval", {})
        if pre:
            eval_steps = [0] + eval_steps
            pope_accs = [pre["pope"]["acc"] * 100] + pope_accs
            gaps = [pre["blind"]["gap"] * 100] + gaps

        axes[0].plot(eval_steps, pope_accs, 'bo-', linewidth=2, markersize=8)
        axes[0].set_ylabel("POPE Accuracy (%)", fontsize=12)
        axes[0].set_xlabel("Step", fontsize=12)
        axes[0].set_title("POPE Accuracy", fontsize=13)

        axes[1].plot(eval_steps, gaps, 'rs-', linewidth=2, markersize=8)
        axes[1].set_ylabel("Blind Gap (pp)", fontsize=12)
        axes[1].set_xlabel("Step", fontsize=12)
        axes[1].set_title("Blind Test Gap", fontsize=13)

        plt.suptitle(f"GRPO-LSR Round {round_num}: Evaluation", fontsize=14)
        plt.tight_layout()
        fig.savefig(report_dir / "eval_progression.png", dpi=150)
        plt.close(fig)

    # Markdown report
    pre = history.get("pre_eval", {})
    final = history.get("final_eval", {})
    cfg = history.get("config", {})

    skip_count = sum(1 for s in history["steps"] if s.get("skipped"))
    total_steps = len(history["steps"])

    md = f"""# GRPO-LSR Round {round_num} Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Model**: {cfg.get('model_path', HF_ID)}

## Configuration

| Parameter | Value |
|-----------|-------|
| Steps | {cfg.get('num_steps', 'N/A')} |
| Group Size | {cfg.get('group_size', 'N/A')} |
| Temperature | {cfg.get('temperature', 'N/A')} |
| LR | {cfg.get('lr', 'N/A')} |
| w_correct | {cfg.get('w_correct', 'N/A')} |
| w_lsr | {cfg.get('w_lsr', 'N/A')} |
| w_fluency | {cfg.get('w_fluency', 'N/A')} |
| LSR Scale | {cfg.get('lsr_scale', 'N/A')} |
| beta_kl | {cfg.get('beta_kl', 'N/A')} |
| beta_entropy | {cfg.get('beta_entropy', 'N/A')} |

## Training Summary

- **Total steps**: {total_steps} ({skip_count} skipped due to zero-variance, {total_steps - skip_count} effective)
- **Skip rate**: {skip_count/total_steps*100:.0f}%

## Results

| Metric | Pre-Training | Post-Training | Delta |
|--------|:------------:|:-------------:|:-----:|
| POPE Acc | {pre.get('pope',{}).get('acc',0)*100:.1f}% | {final.get('pope',{}).get('acc',0)*100:.1f}% | {(final.get('pope',{}).get('acc',0)-pre.get('pope',{}).get('acc',0))*100:+.1f}pp |
| Yes Rate | {pre.get('pope',{}).get('yes_rate',0)*100:.1f}% | {final.get('pope',{}).get('yes_rate',0)*100:.1f}% | {(final.get('pope',{}).get('yes_rate',0)-pre.get('pope',{}).get('yes_rate',0))*100:+.1f}pp |
| Blind Gap | {pre.get('blind',{}).get('gap',0)*100:.1f}pp | {final.get('blind',{}).get('gap',0)*100:.1f}pp | {(final.get('blind',{}).get('gap',0)-pre.get('blind',{}).get('gap',0))*100:+.1f}pp |
| Real Acc | {pre.get('blind',{}).get('real_acc',0)*100:.1f}% | {final.get('blind',{}).get('real_acc',0)*100:.1f}% | {(final.get('blind',{}).get('real_acc',0)-pre.get('blind',{}).get('real_acc',0))*100:+.1f}pp |

"""

    if final.get("pope", {}).get("per_split"):
        md += "### Per-Split POPE Accuracy\n\n"
        md += "| Split | Accuracy |\n|-------|----------|\n"
        for split, acc in sorted(final["pope"]["per_split"].items()):
            md += f"| {split} | {acc*100:.1f}% |\n"
        md += "\n"

    md += """## Figures

![Training Dynamics](training_dynamics.png)
![Evaluation](eval_progression.png)
"""

    with open(report_dir / "REPORT.md", "w") as f:
        f.write(md)

    print(f"  [report] Saved to {report_dir}")
    return md


# ── Auto-Improvement Analysis ─────────────────────────────────────────

def analyze_and_improve(round_num, history, current_cfg):
    """
    Analyze results from current round and propose config changes for next round.
    Returns (new_cfg, analysis_text).
    """
    new_cfg = dict(current_cfg)
    notes = []

    pre = history.get("pre_eval", {})
    final = history.get("final_eval", {})

    pre_acc = pre.get("pope", {}).get("acc", 0)
    final_acc = final.get("pope", {}).get("acc", 0)
    delta_acc = final_acc - pre_acc

    pre_gap = pre.get("blind", {}).get("gap", 0)
    final_gap = final.get("blind", {}).get("gap", 0)
    delta_gap = final_gap - pre_gap

    yes_rate = final.get("pope", {}).get("yes_rate", 0.5)

    # Count skips
    skip_rate = sum(1 for s in history["steps"] if s.get("skipped")) / \
                max(len(history["steps"]), 1)

    # Mean LSR from training
    active_steps = [s for s in history["steps"] if not s.get("skipped")]
    mean_lsr = np.mean([s.get("mean_lsr_raw", 0) for s in active_steps]) \
               if active_steps else 0

    # ── Rule-based improvements ──

    # 1. High skip rate → increase group size
    if skip_rate > 0.4:
        old_gs = new_cfg.get("group_size", 4)
        new_gs = min(old_gs + 2, 8)
        if new_gs != old_gs:
            new_cfg["group_size"] = new_gs
            notes.append(f"Skip rate {skip_rate:.0%} too high → "
                         f"group_size {old_gs}→{new_gs}")

    # 2. Also increase temperature if skip rate is high
    if skip_rate > 0.3:
        old_t = new_cfg.get("temperature", 1.2)
        new_t = min(old_t + 0.1, 1.5)
        if new_t != old_t:
            new_cfg["temperature"] = round(new_t, 1)
            notes.append(f"Low diversity → temperature {old_t}→{new_t}")

    # 3. Collapse detection
    if yes_rate > 0.85 or yes_rate < 0.15:
        new_cfg["beta_entropy"] = min(
            new_cfg.get("beta_entropy", 0.01) * 2, 0.1)
        new_cfg["lr"] = new_cfg.get("lr", 5e-6) * 0.5
        notes.append(f"Collapse risk (yes_rate={yes_rate:.1%}) → "
                     f"↑entropy bonus, ↓lr")

    # 4. If LSR is too high/low, adjust scale
    if mean_lsr > 0 and active_steps:
        if mean_lsr > 3.0:
            new_cfg["lsr_scale"] = round(new_cfg.get("lsr_scale", 2.0) * 1.5, 1)
            notes.append(f"LSR too high ({mean_lsr:.2f}) → ↑lsr_scale")
        elif mean_lsr < 0.1:
            new_cfg["lsr_scale"] = round(
                max(new_cfg.get("lsr_scale", 2.0) * 0.7, 0.5), 1)
            notes.append(f"LSR too low ({mean_lsr:.2f}) → ↓lsr_scale")

    # 5. If accuracy dropped, reduce lr
    if delta_acc < -0.02:
        new_cfg["lr"] = new_cfg.get("lr", 5e-6) * 0.5
        notes.append(f"Accuracy dropped {delta_acc*100:+.1f}pp → ↓lr by 50%")

    # 6. If accuracy improved, keep momentum
    if delta_acc > 0.01:
        notes.append(f"Accuracy improved {delta_acc*100:+.1f}pp — keeping config")

    # 7. If gap improved significantly, increase LSR weight
    if delta_gap > 0.02:
        old_w = new_cfg.get("w_lsr", 0.4)
        new_w = min(old_w + 0.05, 0.6)
        if new_w != old_w:
            new_cfg["w_lsr"] = round(new_w, 2)
            new_cfg["w_correct"] = round(1.0 - new_w -
                                          new_cfg.get("w_fluency", 0.2), 2)
            notes.append(f"Gap improved → ↑w_lsr {old_w}→{new_w}")

    if not notes:
        notes.append("No changes needed — results stable")

    analysis = f"Round {round_num} Analysis:\n" + "\n".join(f"  - {n}" for n in notes)
    return new_cfg, analysis


# ── Git Operations ─────────────────────────────────────────────────────

def git_commit_and_push(round_num, summary):
    """Commit results and push to remote."""
    try:
        os.chdir(PROJECT_ROOT)

        # Stage relevant files
        subprocess.run(["git", "add",
                        f"lab/reports/grpo_lsr/",
                        f"checkpoints/grpo_lsr/",
                        "lab/RESEARCH_JOURNAL.md",
                        "scripts/train_grpo_lsr.py",
                        "scripts/auto_grpo_lsr_loop.py",
                        "scripts/logit_shift_reward.py",
                        ], capture_output=True)

        msg = (f"GRPO-LSR Round {round_num}: {summary}\n\n"
               f"Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>")
        result = subprocess.run(
            ["git", "commit", "-m", msg],
            capture_output=True, text=True)
        print(f"  [git] commit: {result.stdout.strip()}")

        result = subprocess.run(
            ["git", "push", "origin", "main"],
            capture_output=True, text=True, timeout=60)
        print(f"  [git] push: {result.stdout.strip() or result.stderr.strip()}")
        return True
    except Exception as e:
        print(f"  [git] error: {e}")
        return False


# ── Single Round ───────────────────────────────────────────────────────

def run_round(round_num, cfg, train_data, eval_data, model_path=None):
    """Run a single GRPO-LSR training round. Returns (history, checkpoint_path)."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(cfg["output_dir"]) / f"round{round_num}"
    report_dir = Path("lab/reports/grpo_lsr") / f"round{round_num}"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  GRPO-LSR Round {round_num}")
    print(f"  Steps: {cfg['num_steps']}, Group: {cfg['group_size']}, "
          f"T: {cfg['temperature']}, LR: {cfg['lr']}")
    print(f"  Weights: correct={cfg['w_correct']}, lsr={cfg['w_lsr']}, "
          f"fluency={cfg['w_fluency']}")
    print(f"{'='*70}\n")

    # Load model
    model, processor = load_model(model_path, for_training=True)
    device = next(model.parameters()).device

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr"], weight_decay=0.01)

    # Pre-eval
    print("Pre-training eval...")
    pre_pope = evaluate_pope(model, processor, eval_data, device)
    pre_blind = evaluate_blind(model, processor, eval_data, device,
                               cfg.get("blind_samples", 100))
    print(f"  POPE: {pre_pope['acc']:.1%} | "
          f"Gap: {pre_blind['gap']:.1%}")

    history = {
        "round": round_num,
        "config": cfg,
        "pre_eval": {"pope": pre_pope, "blind": pre_blind},
        "steps": [],
        "evals": [],
    }

    # Training loop
    model.train()
    optimizer.zero_grad()
    best_acc = pre_pope["acc"]

    for step in range(cfg["num_steps"]):
        step_t0 = time.time()
        sample = train_data[step % len(train_data)]

        # Generate
        model.eval()
        try:
            candidates, cand_ids_list, prompt_len, inputs = \
                generate_candidates(
                    model, processor, sample, cfg["group_size"],
                    cfg["temperature"], cfg["top_p"],
                    cfg["max_new_tokens"], device)
        except torch.cuda.OutOfMemoryError:
            print(f"  [step {step+1}] OOM gen, skip")
            torch.cuda.empty_cache(); gc.collect()
            continue

        # Rewards
        try:
            rewards, details = compute_rewards(
                model, processor, sample, candidates, cand_ids_list,
                prompt_len, device, cfg)
        except torch.cuda.OutOfMemoryError:
            print(f"  [step {step+1}] OOM reward, skip")
            torch.cuda.empty_cache(); gc.collect()
            continue

        # Advantage
        rarr = np.array(rewards)
        rstd = rarr.std()

        if rstd < 1e-8:
            elapsed = time.time() - step_t0
            print(f"  [step {step+1}/{cfg['num_steps']}] "
                  f"SKIP r={rarr.mean():.3f} ({elapsed:.1f}s)")
            history["steps"].append({
                "step": step + 1, "skipped": True,
                "mean_reward": float(rarr.mean())})
            del candidates, cand_ids_list, inputs
            continue

        advantages = ((rarr - rarr.mean()) / (rstd + 1e-8)).tolist()

        # Loss
        model.train()
        try:
            loss, lstats = compute_grpo_loss(
                model, inputs, cand_ids_list, prompt_len, advantages, cfg)
            (loss / cfg["grad_accum"]).backward()

            if (step + 1) % cfg["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg.get("max_grad_norm", 1.0))
                optimizer.step()
                optimizer.zero_grad()

        except torch.cuda.OutOfMemoryError:
            print(f"  [step {step+1}] OOM loss, skip")
            optimizer.zero_grad()
            torch.cuda.empty_cache(); gc.collect()
            continue

        elapsed = time.time() - step_t0
        mc = np.mean([d["correct"] for d in details])
        ml = np.mean([d["lsr_raw"] for d in details])
        mk = np.mean(lstats["kl"]) if lstats["kl"] else 0

        history["steps"].append({
            "step": step + 1, "loss": loss.item(),
            "mean_reward": float(rarr.mean()),
            "reward_std": float(rstd),
            "mean_correct": float(mc),
            "mean_lsr_raw": float(ml),
            "mean_kl": float(mk),
            "elapsed": elapsed,
        })

        print(f"  [step {step+1}/{cfg['num_steps']}] "
              f"loss={loss.item():.4f} r={rarr.mean():.3f}±{rstd:.3f} "
              f"correct={mc:.2f} LSR={ml:.3f} ({elapsed:.1f}s)",
              flush=True)

        # Periodic eval
        if (step + 1) % cfg["eval_every"] == 0:
            pope_res = evaluate_pope(model, processor, eval_data, device)
            blind_res = evaluate_blind(model, processor, eval_data, device,
                                       cfg.get("blind_samples", 100))
            print(f"  === Eval step {step+1}: "
                  f"POPE={pope_res['acc']:.1%} "
                  f"Gap={blind_res['gap']:.1%} ===")
            history["evals"].append({
                "step": step + 1, "pope": pope_res, "blind": blind_res})

            if pope_res["acc"] > best_acc:
                best_acc = pope_res["acc"]
                model.save_pretrained(output_dir / "best")
                processor.save_pretrained(output_dir / "best")
                print(f"  ★ New best: {best_acc:.1%}")

        del loss, candidates, cand_ids_list, inputs
        torch.cuda.empty_cache()
        if step % 5 == 0:
            gc.collect()

    # Final eval
    print("\nFinal evaluation...")
    final_pope = evaluate_pope(model, processor, eval_data, device)
    final_blind = evaluate_blind(model, processor, eval_data, device,
                                  cfg.get("blind_samples", 100))
    print(f"  POPE: {final_pope['acc']:.1%} | Gap: {final_blind['gap']:.1%}")
    history["final_eval"] = {"pope": final_pope, "blind": final_blind}

    # Save checkpoint
    model.save_pretrained(output_dir / "final")
    processor.save_pretrained(output_dir / "final")

    # Save history
    with open(output_dir / f"history_{ts}.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(report_dir / f"history_{ts}.json", "w") as f:
        json.dump(history, f, indent=2)

    # Generate report
    generate_report(round_num, history, report_dir)

    # Cleanup model
    del model, optimizer
    torch.cuda.empty_cache()
    gc.collect()

    return history, str(output_dir / "final")


# ── Main Loop ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Auto GRPO-LSR Loop")
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--steps-per-round", type=int, default=50)
    parser.add_argument("--train-samples", type=int, default=500)
    parser.add_argument("--eval-samples", type=int, default=300)
    parser.add_argument("--blind-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str,
                        default="checkpoints/grpo_lsr")
    parser.add_argument("--no-git", action="store_true",
                        help="Skip git commit/push")
    # Initial config overrides
    parser.add_argument("--group-size", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=1.3)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--w-correct", type=float, default=0.4)
    parser.add_argument("--w-lsr", type=float, default=0.4)
    parser.add_argument("--w-fluency", type=float, default=0.2)
    parser.add_argument("--lsr-scale", type=float, default=2.0)
    parser.add_argument("--resume-round", type=int, default=1,
                        help="Start from this round number")
    parser.add_argument("--resume-model", type=str, default=None,
                        help="Model path to resume from")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load data once
    train_data = load_training_data(args.train_samples, args.seed)
    eval_data = load_pope_eval(args.eval_samples)
    print(f"\n[data] {len(train_data)} train, {len(eval_data)} eval\n")

    # Initial config
    cfg = {
        "num_steps": args.steps_per_round,
        "group_size": args.group_size,
        "temperature": args.temperature,
        "top_p": 0.95,
        "max_new_tokens": 64,
        "lr": args.lr,
        "beta_kl": 0.01,
        "beta_entropy": 0.01,
        "epsilon": 0.2,
        "grad_accum": 4,
        "max_grad_norm": 1.0,
        "w_correct": args.w_correct,
        "w_lsr": args.w_lsr,
        "w_fluency": args.w_fluency,
        "lsr_scale": args.lsr_scale,
        "eval_every": 10,
        "blind_samples": args.blind_samples,
        "output_dir": args.output_dir,
    }

    all_results = []
    model_path = args.resume_model

    for round_num in range(args.resume_round, args.resume_round + args.max_rounds):
        print(f"\n{'#'*70}")
        print(f"#  AUTO LOOP: Round {round_num} / "
              f"{args.resume_round + args.max_rounds - 1}")
        print(f"{'#'*70}")

        # Run round
        history, checkpoint = run_round(
            round_num, cfg, train_data, eval_data, model_path)

        # Summarize
        pre = history["pre_eval"]
        final = history["final_eval"]
        delta_acc = (final["pope"]["acc"] - pre["pope"]["acc"]) * 100
        delta_gap = (final["blind"]["gap"] - pre["blind"]["gap"]) * 100

        summary = (f"POPE {final['pope']['acc']*100:.1f}% "
                   f"({delta_acc:+.1f}pp), "
                   f"Gap {final['blind']['gap']*100:.1f}pp "
                   f"({delta_gap:+.1f}pp)")
        print(f"\n  Round {round_num} summary: {summary}")

        # Git commit + push
        if not args.no_git:
            # Append to research journal
            journal_path = PROJECT_ROOT / "lab" / "RESEARCH_JOURNAL.md"
            with open(journal_path, "a") as f:
                f.write(f"\n\n### GRPO-LSR Round {round_num} "
                        f"({datetime.now().strftime('%Y-%m-%d %H:%M')})\n\n")
                f.write(f"- Config: group={cfg['group_size']}, "
                        f"T={cfg['temperature']}, lr={cfg['lr']}, "
                        f"w_lsr={cfg['w_lsr']}\n")
                f.write(f"- Pre:  POPE={pre['pope']['acc']*100:.1f}%, "
                        f"Gap={pre['blind']['gap']*100:.1f}pp\n")
                f.write(f"- Post: POPE={final['pope']['acc']*100:.1f}%, "
                        f"Gap={final['blind']['gap']*100:.1f}pp\n")
                f.write(f"- Delta: POPE {delta_acc:+.1f}pp, "
                        f"Gap {delta_gap:+.1f}pp\n")

            git_commit_and_push(round_num, summary)

        all_results.append({
            "round": round_num,
            "config": dict(cfg),
            "pre_pope_acc": pre["pope"]["acc"],
            "final_pope_acc": final["pope"]["acc"],
            "delta_acc": delta_acc,
            "pre_gap": pre["blind"]["gap"],
            "final_gap": final["blind"]["gap"],
            "delta_gap": delta_gap,
            "checkpoint": checkpoint,
        })

        # Analyze and improve for next round
        new_cfg, analysis = analyze_and_improve(round_num, history, cfg)
        print(f"\n  {analysis}")

        cfg = new_cfg
        model_path = checkpoint  # Next round starts from this checkpoint

        # Early stopping: 3 rounds with no improvement
        if len(all_results) >= 3:
            last3 = all_results[-3:]
            if all(r["delta_acc"] <= 0 for r in last3):
                print("\n  ⚠ 3 consecutive rounds with no improvement. Stopping.")
                break

    # ── Final Summary ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("AUTO LOOP COMPLETE")
    print(f"{'='*70}")
    print(f"\n{'Round':<8} {'Pre POPE':>10} {'Post POPE':>10} {'Δ':>8} "
          f"{'Pre Gap':>10} {'Post Gap':>10} {'Δ':>8}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['round']:<8} {r['pre_pope_acc']*100:>9.1f}% "
              f"{r['final_pope_acc']*100:>9.1f}% {r['delta_acc']:>+7.1f}pp "
              f"{r['pre_gap']*100:>9.1f}pp {r['final_gap']*100:>9.1f}pp "
              f"{r['delta_gap']:>+7.1f}pp")

    # Save overall results
    overall_path = Path(args.output_dir) / "auto_loop_results.json"
    with open(overall_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults: {overall_path}")


if __name__ == "__main__":
    main()
