"""
Thinking-Mode GRPO with Logit-Shift Reward (LSR)

Key design decisions (from validation):
  - LSR signal is EXCLUSIVELY in thinking phase (KL~1.44 vs ~0.0003 in answer)
  - Use Qwen3-VL-2B-Thinking with enable_thinking=True
  - Compute LSR only on <think>...</think> tokens (thinking-phase masking)
  - Gated reward: R = R_correct*0.5 + R_correct*R_LSR*0.5
    (high LSR only rewarded when answer is correct — prevents reward hacking)
  - Force min reasoning chain via min_new_tokens

Pipeline:
  Round N → GRPO training → Eval → Report → Git push → Analyze → Round N+1

Usage:
    PYTHONUNBUFFERED=1 python -u scripts/auto_grpo_thinking_lsr.py \
        --max-rounds 3 --steps-per-round 50 \
        2>&1 | tee logs/auto_grpo_thinking_lsr.log
"""

import os, sys, gc, json, re, time, random, argparse, string, subprocess
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from datetime import datetime
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

HF_ID = "Qwen/Qwen3-VL-2B-Thinking"
POPE_SPLITS = ["random", "popular", "adversarial"]
PROJECT_ROOT = Path(__file__).parent.parent


# ══════════════════════════════════════════════════════════════════════
#  Answer Extraction
# ══════════════════════════════════════════════════════════════════════

def split_thinking(text):
    """Split output into (thinking, answer) parts."""
    m = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    m = re.search(r'</think>', text)
    if m:
        return text[:m.start()].strip(), text[m.end():].strip()
    return "", text.strip()


def extract_yes_no(raw):
    _, answer = split_thinking(raw)
    if not answer:
        answer = raw
    text = answer.strip().lower()
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
    _, answer = split_thinking(raw)
    if not answer:
        answer = raw
    text = answer.strip()
    if qtype == "yesno":
        return extract_yes_no(raw)
    if qtype == "mc":
        for ch in text[:5]:
            if ch.upper() in "ABCDEFGH":
                return ch.upper()
        return text[:20]
    return text.split("\n")[0].strip()[:100]


# ══════════════════════════════════════════════════════════════════════
#  Data Loading
# ══════════════════════════════════════════════════════════════════════

def load_training_data(limit=500, seed=42):
    """Load mixed training data (non-binary VQA)."""
    from datasets import load_dataset
    rng = random.Random(seed)
    samples = []

    # A-OKVQA (MC — good diversity for thinking)
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

    # VQAv2 short-answer (non-binary only)
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
    src_counts = Counter(s["source"] for s in samples)
    print(f"[data] {len(samples)} training samples "
          f"({', '.join(f'{k}={v}' for k, v in src_counts.items())})")
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


# ══════════════════════════════════════════════════════════════════════
#  Model Loading
# ══════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════
#  Input Preparation (Thinking Mode)
# ══════════════════════════════════════════════════════════════════════

def prepare_inputs(processor, image, question, device):
    """Prepare inputs with enable_thinking=True for reasoning chain."""
    from qwen_vl_utils import process_vision_info
    content = [{"type": "image", "image": image},
               {"type": "text", "text": question}]
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True)  # ← Force thinking mode
    imgs, _, _ = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(text=[text], images=imgs, return_tensors="pt",
                       padding=True)
    return {k: v.to(device) for k, v in inputs.items()}


# ══════════════════════════════════════════════════════════════════════
#  Thinking-Phase Token Masking
# ══════════════════════════════════════════════════════════════════════

def find_think_token_range(tokenizer, gen_ids):
    """
    Find the token index range for <think>...</think> in generated tokens.
    Returns (start, end) indices within gen_ids.
    """
    # Decode to find </think> position
    text = tokenizer.decode(gen_ids, skip_special_tokens=False)

    # Find </think> boundary
    think_end_str = "</think>"
    think_start_str = "<think>"

    # Token-level search: find </think> token ID
    think_end_ids = tokenizer.encode(think_end_str, add_special_tokens=False)
    think_start_ids = tokenizer.encode(think_start_str, add_special_tokens=False)

    gen_list = gen_ids.tolist()

    # Find start of thinking (first <think> token)
    start_idx = 0
    if think_start_ids:
        for i in range(len(gen_list)):
            if gen_list[i] == think_start_ids[0]:
                start_idx = i + 1  # After <think> token
                break

    # Find end of thinking (</think> token)
    end_idx = len(gen_list)
    if think_end_ids:
        for i in range(len(gen_list)):
            if gen_list[i] == think_end_ids[0]:
                end_idx = i  # Before </think> token
                break

    return start_idx, end_idx


# ══════════════════════════════════════════════════════════════════════
#  Candidate Generation (Thinking Mode)
# ══════════════════════════════════════════════════════════════════════

def generate_candidates(model, processor, sample, group_size, temperature,
                        top_p, max_new_tokens, min_think_tokens, device):
    """Generate candidates with forced thinking chains."""
    question = sample["question"]
    image = sample["image"]
    inputs = prepare_inputs(processor, image, question, device)
    prompt_len = inputs["input_ids"].shape[1]

    candidates = []
    candidate_ids_list = []
    think_ranges = []

    for _ in range(group_size):
        try:
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_think_tokens,  # Force reasoning
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                )
            gen_ids = out[0][prompt_len:]
            text = processor.tokenizer.decode(gen_ids, skip_special_tokens=False)

            # Clean up special tokens for answer extraction
            clean_text = text
            for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
                clean_text = clean_text.replace(tok, "")

            candidates.append(clean_text.strip())
            candidate_ids_list.append(gen_ids.detach())

            # Find thinking phase boundaries
            t_start, t_end = find_think_token_range(
                processor.tokenizer, gen_ids)
            think_ranges.append((t_start, t_end))

        except Exception as e:
            candidates.append("")
            candidate_ids_list.append(
                torch.tensor([], dtype=torch.long, device=device))
            think_ranges.append((0, 0))

    return candidates, candidate_ids_list, prompt_len, inputs, think_ranges


# ══════════════════════════════════════════════════════════════════════
#  LSR: Thinking-Phase Only
# ══════════════════════════════════════════════════════════════════════

def compute_thinking_lsr(model, processor, sample, candidate_ids,
                          think_range, device):
    """
    Compute LSR = D_KL(P_real || P_black) ONLY on thinking-phase tokens.

    Steps:
      1. Teacher-force with real image → logits_real
      2. Teacher-force with black image → logits_black
      3. KL divergence only at thinking token positions
    """
    if candidate_ids.numel() == 0:
        return 0.0, 0

    t_start, t_end = think_range
    if t_end <= t_start:
        return 0.0, 0

    image = sample["image"]
    question = sample["question"]

    # Teacher-force with REAL image
    real_inputs = prepare_inputs(processor, image, question, device)
    rpl = real_inputs["input_ids"].shape[1]
    rf = torch.cat([real_inputs["input_ids"],
                    candidate_ids.unsqueeze(0)], dim=1)
    real_inputs["input_ids"] = rf
    real_inputs["attention_mask"] = torch.ones_like(rf)
    with torch.no_grad():
        logits_real = model(**real_inputs).logits[0]

    # Teacher-force with BLACK image
    black_image = Image.new('RGB', image.size, (0, 0, 0))
    black_inputs = prepare_inputs(processor, black_image, question, device)
    bpl = black_inputs["input_ids"].shape[1]
    bf = torch.cat([black_inputs["input_ids"],
                    candidate_ids.unsqueeze(0)], dim=1)
    black_inputs["input_ids"] = bf
    black_inputs["attention_mask"] = torch.ones_like(bf)
    with torch.no_grad():
        logits_black = model(**black_inputs).logits[0]

    # Extract logits at generated token positions (next-token shift)
    lr = logits_real[rpl - 1: rpl - 1 + len(candidate_ids)]
    lb = logits_black[bpl - 1: bpl - 1 + len(candidate_ids)]

    ml = min(lr.shape[0], lb.shape[0], len(candidate_ids))
    if ml == 0:
        return 0.0, 0

    # Apply thinking-phase mask: only compute KL on [t_start, t_end)
    t_end_safe = min(t_end, ml)
    t_start_safe = min(t_start, t_end_safe)
    think_len = t_end_safe - t_start_safe

    if think_len <= 0:
        return 0.0, 0

    lr_think = lr[t_start_safe:t_end_safe].float()
    lb_think = lb[t_start_safe:t_end_safe].float()

    # Per-token KL: D_KL(P_real || P_black) on thinking tokens only
    kl_per_token = F.kl_div(
        F.log_softmax(lb_think, dim=-1),
        F.softmax(lr_think, dim=-1),
        reduction='none'
    ).sum(dim=-1)

    mean_kl = kl_per_token.mean().item()

    del logits_real, logits_black, lr, lb, lr_think, lb_think, kl_per_token
    del real_inputs, black_inputs
    return mean_kl, think_len


# ══════════════════════════════════════════════════════════════════════
#  Gated Reward
# ══════════════════════════════════════════════════════════════════════

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


def compute_gated_reward(r_correct, r_lsr_raw, lsr_scale):
    """
    Gated reward: R = R_correct * 0.5 + R_correct * R_LSR * 0.5

    - If correct=0: R=0 regardless of LSR → prevents reward hacking
    - If correct=1 & high LSR: R=1.0 (maximum reward)
    - If correct=1 & low LSR: R=0.5 (correct but not grounded)
    """
    r_lsr = min(r_lsr_raw / lsr_scale, 1.0)
    r_total = r_correct * 0.5 + r_correct * r_lsr * 0.5
    return r_total, r_lsr


def compute_rewards(model, processor, sample, candidates, cand_ids_list,
                    think_ranges, prompt_len, device, cfg):
    """Compute gated rewards for all candidates."""
    rewards = []
    details = []
    gt = sample["answer"]
    qtype = sample.get("type", "short_answer")

    for cand, cand_ids, t_range in zip(candidates, cand_ids_list, think_ranges):
        pred = extract_answer(cand, qtype)
        r_correct = compute_r_correct(pred, gt, qtype)

        # Thinking text for logging
        thinking, answer = split_thinking(cand)
        think_words = len(thinking.split()) if thinking else 0

        # LSR on thinking phase only
        try:
            r_lsr_raw, think_tokens = compute_thinking_lsr(
                model, processor, sample, cand_ids, t_range, device)
        except Exception:
            r_lsr_raw, think_tokens = 0.0, 0

        # Gated reward
        r_total, r_lsr_norm = compute_gated_reward(
            r_correct, r_lsr_raw, cfg["lsr_scale"])

        rewards.append(r_total)
        details.append({
            "pred": pred, "gt": gt, "correct": r_correct,
            "lsr_raw": r_lsr_raw, "lsr_norm": r_lsr_norm,
            "think_words": think_words, "think_tokens": think_tokens,
            "total": r_total,
        })

    return rewards, details


# ══════════════════════════════════════════════════════════════════════
#  Log-Prob & GRPO Loss
# ══════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════
#  Evaluation
# ══════════════════════════════════════════════════════════════════════

def evaluate_pope(model, processor, samples, device):
    """POPE eval with thinking mode."""
    model.eval()
    correct = total = yes_count = 0
    per_split = defaultdict(lambda: {"correct": 0, "total": 0})
    think_lengths = []

    for s in samples:
        try:
            inputs = prepare_inputs(
                processor, s["image"],
                s["question"] + " Please answer yes or no.", device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=512,
                                     do_sample=False)
            gen = out[0][inputs["input_ids"].shape[1]:]
            raw = processor.tokenizer.decode(gen, skip_special_tokens=False)
            for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
                raw = raw.replace(tok, "")

            thinking, answer = split_thinking(raw)
            think_lengths.append(len(thinking.split()) if thinking else 0)

            pred = extract_yes_no(raw)
            gt = s["answer"].strip().lower()
            cat = s.get("category", "unknown")

            if pred == gt:
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
    split_accs = {cat: d["correct"]/d["total"] if d["total"] > 0 else 0
                  for cat, d in per_split.items()}

    return {
        "acc": acc, "total": total,
        "yes_rate": yes_count / total if total > 0 else 0,
        "per_split": split_accs,
        "avg_think_words": float(np.mean(think_lengths)) if think_lengths else 0,
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
                out = model.generate(**inputs, max_new_tokens=512,
                                     do_sample=False)
            raw = processor.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=False)
            pred_real = extract_yes_no(raw)
            if pred_real == gt:
                real_correct += 1

            black = Image.new('RGB', s["image"].size, (0, 0, 0))
            inputs_b = prepare_inputs(processor, black, q, device)
            with torch.no_grad():
                out_b = model.generate(**inputs_b, max_new_tokens=512,
                                       do_sample=False)
            raw_b = processor.tokenizer.decode(
                out_b[0][inputs_b["input_ids"].shape[1]:],
                skip_special_tokens=False)
            pred_blind = extract_yes_no(raw_b)
            if pred_blind == gt:
                blind_correct += 1
            total += 1
        except Exception:
            total += 1

    model.train()
    ra = real_correct / total if total > 0 else 0
    ba = blind_correct / total if total > 0 else 0
    return {"real_acc": ra, "blind_acc": ba, "gap": ra - ba, "total": total}


# ══════════════════════════════════════════════════════════════════════
#  Report Generation
# ══════════════════════════════════════════════════════════════════════

def generate_report(round_num, history, report_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid")

    report_dir.mkdir(parents=True, exist_ok=True)

    steps = [s for s in history["steps"] if not s.get("skipped")]
    if not steps:
        return

    step_nums = [s["step"] for s in steps]
    losses = [s["loss"] for s in steps]
    lsr_vals = [s.get("mean_lsr_raw", 0) for s in steps]
    think_lens = [s.get("mean_think_words", 0) for s in steps]
    correct_vals = [s.get("mean_correct", 0) for s in steps]

    # Fig 1: Training dynamics (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(step_nums, losses, 'b-', linewidth=1.5)
    axes[0, 0].set_ylabel("Loss", fontsize=12)
    axes[0, 0].set_title("GRPO Loss", fontsize=13)

    axes[0, 1].plot(step_nums, lsr_vals, 'r-', linewidth=1.5)
    axes[0, 1].set_ylabel("Mean KL (Thinking Phase)", fontsize=12)
    axes[0, 1].set_title("LSR: Image Influence in Reasoning", fontsize=13)

    axes[1, 0].plot(step_nums, think_lens, 'g-', linewidth=1.5)
    axes[1, 0].set_ylabel("Avg Thinking Words", fontsize=12)
    axes[1, 0].set_xlabel("Step", fontsize=12)
    axes[1, 0].set_title("Reasoning Chain Length", fontsize=13)

    axes[1, 1].plot(step_nums, correct_vals, 'm-', linewidth=1.5)
    axes[1, 1].set_ylabel("Correctness", fontsize=12)
    axes[1, 1].set_xlabel("Step", fontsize=12)
    axes[1, 1].set_title("Training Correctness", fontsize=13)

    plt.suptitle(f"Thinking-GRPO-LSR Round {round_num}", fontsize=14)
    plt.tight_layout()
    fig.savefig(report_dir / "training_dynamics.png", dpi=150)
    plt.close(fig)

    # Fig 2: Eval progression
    evals = history.get("evals", [])
    pre = history.get("pre_eval", {})
    if evals:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        eval_steps = [0] + [e["step"] for e in evals]
        pope_accs = [pre.get("pope", {}).get("acc", 0) * 100] + \
                    [e["pope"]["acc"] * 100 for e in evals]
        gaps = [pre.get("blind", {}).get("gap", 0) * 100] + \
               [e["blind"]["gap"] * 100 for e in evals]
        think_w = [pre.get("pope", {}).get("avg_think_words", 0)] + \
                  [e["pope"].get("avg_think_words", 0) for e in evals]

        axes[0].plot(eval_steps, pope_accs, 'bo-', linewidth=2, markersize=8)
        axes[0].set_ylabel("POPE Accuracy (%)", fontsize=12)
        axes[0].set_xlabel("Step", fontsize=12)
        axes[0].set_title("POPE Accuracy", fontsize=13)

        axes[1].plot(eval_steps, gaps, 'rs-', linewidth=2, markersize=8)
        axes[1].set_ylabel("Blind Gap (pp)", fontsize=12)
        axes[1].set_xlabel("Step", fontsize=12)
        axes[1].set_title("Blind Test Gap", fontsize=13)

        axes[2].plot(eval_steps, think_w, 'g^-', linewidth=2, markersize=8)
        axes[2].set_ylabel("Avg Thinking Words", fontsize=12)
        axes[2].set_xlabel("Step", fontsize=12)
        axes[2].set_title("Reasoning Length (Eval)", fontsize=13)

        plt.suptitle(f"Thinking-GRPO-LSR Round {round_num}: Eval", fontsize=14)
        plt.tight_layout()
        fig.savefig(report_dir / "eval_progression.png", dpi=150)
        plt.close(fig)

    # Markdown report
    final = history.get("final_eval", {})
    cfg = history.get("config", {})
    skip_count = sum(1 for s in history["steps"] if s.get("skipped"))
    total_steps = len(history["steps"])

    md = f"""# Thinking-GRPO-LSR Round {round_num} Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Model**: Qwen3-VL-2B-Thinking (enable_thinking=True)

## Configuration

| Parameter | Value |
|-----------|-------|
| Steps | {cfg.get('num_steps', 'N/A')} |
| Group Size | {cfg.get('group_size', 'N/A')} |
| Temperature | {cfg.get('temperature', 'N/A')} |
| LR | {cfg.get('lr', 'N/A')} |
| LSR Scale | {cfg.get('lsr_scale', 'N/A')} |
| Min Think Tokens | {cfg.get('min_think_tokens', 'N/A')} |
| Reward | R_correct*0.5 + R_correct*R_LSR*0.5 (gated) |

## Training Summary

- **Total steps**: {total_steps} ({skip_count} skipped, {total_steps - skip_count} effective)
- **Skip rate**: {skip_count/max(total_steps,1)*100:.0f}%
- **Mean thinking words (train)**: {np.mean([s.get('mean_think_words',0) for s in steps]):.0f}
- **Mean KL thinking (train)**: {np.mean([s.get('mean_lsr_raw',0) for s in steps]):.3f}

## Results

| Metric | Pre | Post | Delta |
|--------|:---:|:----:|:-----:|
| POPE Acc | {pre.get('pope',{}).get('acc',0)*100:.1f}% | {final.get('pope',{}).get('acc',0)*100:.1f}% | {(final.get('pope',{}).get('acc',0)-pre.get('pope',{}).get('acc',0))*100:+.1f}pp |
| Blind Gap | {pre.get('blind',{}).get('gap',0)*100:.1f}pp | {final.get('blind',{}).get('gap',0)*100:.1f}pp | {(final.get('blind',{}).get('gap',0)-pre.get('blind',{}).get('gap',0))*100:+.1f}pp |
| Think Words | {pre.get('pope',{}).get('avg_think_words',0):.0f} | {final.get('pope',{}).get('avg_think_words',0):.0f} | {final.get('pope',{}).get('avg_think_words',0)-pre.get('pope',{}).get('avg_think_words',0):+.0f} |

## Figures

![Training Dynamics](training_dynamics.png)
![Evaluation](eval_progression.png)
"""

    with open(report_dir / "REPORT.md", "w") as f:
        f.write(md)
    print(f"  [report] Saved to {report_dir}")


# ══════════════════════════════════════════════════════════════════════
#  Auto-Improvement Analysis
# ══════════════════════════════════════════════════════════════════════

def analyze_and_improve(round_num, history, current_cfg):
    new_cfg = dict(current_cfg)
    notes = []

    pre = history.get("pre_eval", {})
    final = history.get("final_eval", {})

    pre_acc = pre.get("pope", {}).get("acc", 0)
    final_acc = final.get("pope", {}).get("acc", 0)
    delta_acc = final_acc - pre_acc

    pre_gap = pre.get("blind", {}).get("gap", 0)
    final_gap = final.get("blind", {}).get("gap", 0)

    yes_rate = final.get("pope", {}).get("yes_rate", 0.5)

    # Skip rate
    skip_rate = sum(1 for s in history["steps"] if s.get("skipped")) / \
                max(len(history["steps"]), 1)

    active = [s for s in history["steps"] if not s.get("skipped")]
    mean_lsr = np.mean([s.get("mean_lsr_raw", 0) for s in active]) \
               if active else 0
    mean_think = np.mean([s.get("mean_think_words", 0) for s in active]) \
                 if active else 0

    # High skip rate → bigger groups + higher temperature
    if skip_rate > 0.4:
        old_gs = new_cfg.get("group_size", 6)
        new_gs = min(old_gs + 2, 10)
        if new_gs != old_gs:
            new_cfg["group_size"] = new_gs
            notes.append(f"Skip rate {skip_rate:.0%} → group_size {old_gs}→{new_gs}")

    if skip_rate > 0.3:
        old_t = new_cfg.get("temperature", 1.3)
        new_t = min(old_t + 0.1, 1.6)
        if abs(new_t - old_t) > 0.01:
            new_cfg["temperature"] = round(new_t, 1)
            notes.append(f"Low diversity → temperature {old_t}→{new_t}")

    # Collapse
    if yes_rate > 0.85 or yes_rate < 0.15:
        new_cfg["beta_entropy"] = min(
            new_cfg.get("beta_entropy", 0.01) * 2, 0.1)
        new_cfg["lr"] = new_cfg.get("lr", 5e-6) * 0.5
        notes.append(f"Collapse risk (yes_rate={yes_rate:.1%}) → ↑entropy, ↓lr")

    # LSR scale tuning
    if mean_lsr > 3.0:
        new_cfg["lsr_scale"] = round(new_cfg.get("lsr_scale", 2.0) * 1.5, 1)
        notes.append(f"LSR too high ({mean_lsr:.2f}) → ↑lsr_scale")
    elif mean_lsr < 0.1 and active:
        new_cfg["lsr_scale"] = round(
            max(new_cfg.get("lsr_scale", 2.0) * 0.7, 0.5), 1)
        notes.append(f"LSR too low ({mean_lsr:.2f}) → ↓lsr_scale")

    # Short thinking → increase min_think_tokens
    if mean_think < 20:
        old_mt = new_cfg.get("min_think_tokens", 32)
        new_mt = min(old_mt + 16, 128)
        new_cfg["min_think_tokens"] = new_mt
        notes.append(f"Short thinking ({mean_think:.0f}w) → "
                     f"min_think_tokens {old_mt}→{new_mt}")

    # Accuracy dropped → reduce lr
    if delta_acc < -0.02:
        new_cfg["lr"] = new_cfg.get("lr", 5e-6) * 0.5
        notes.append(f"Accuracy dropped {delta_acc*100:+.1f}pp → ↓lr")

    if not notes:
        notes.append("No changes — results stable")

    analysis = f"Round {round_num} Analysis:\n" + \
               "\n".join(f"  - {n}" for n in notes)
    return new_cfg, analysis


# ══════════════════════════════════════════════════════════════════════
#  Git Operations
# ══════════════════════════════════════════════════════════════════════

def git_commit_and_push(round_num, summary):
    try:
        os.chdir(PROJECT_ROOT)
        subprocess.run(["git", "add",
                        "lab/reports/grpo_lsr_thinking/",
                        "checkpoints/grpo_lsr_thinking/",
                        "lab/RESEARCH_JOURNAL.md",
                        "scripts/auto_grpo_thinking_lsr.py",
                        "scripts/logit_shift_reward.py",
                        ], capture_output=True)

        msg = (f"Thinking-GRPO-LSR Round {round_num}: {summary}\n\n"
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


# ══════════════════════════════════════════════════════════════════════
#  Single Round
# ══════════════════════════════════════════════════════════════════════

def run_round(round_num, cfg, train_data, eval_data, model_path=None):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(cfg["output_dir"]) / f"round{round_num}"
    report_dir = Path("lab/reports/grpo_lsr_thinking") / f"round{round_num}"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  Thinking-GRPO-LSR Round {round_num}")
    print(f"  Steps: {cfg['num_steps']}, Group: {cfg['group_size']}, "
          f"T: {cfg['temperature']}, LR: {cfg['lr']}")
    print(f"  Reward: R_correct*0.5 + R_correct*R_LSR*0.5 (gated)")
    print(f"  min_think_tokens: {cfg['min_think_tokens']}, "
          f"max_new_tokens: {cfg['max_new_tokens']}")
    print(f"{'='*70}\n")

    model, processor = load_model(model_path, for_training=True)
    device = next(model.parameters()).device

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr"], weight_decay=0.01)

    # Pre-eval
    print("Pre-training eval...")
    pre_pope = evaluate_pope(model, processor, eval_data[:60], device)
    pre_blind = evaluate_blind(model, processor, eval_data, device,
                               min(cfg.get("blind_samples", 50), 50))
    print(f"  POPE: {pre_pope['acc']:.1%} | "
          f"Gap: {pre_blind['gap']:.1%} | "
          f"Think: {pre_pope['avg_think_words']:.0f}w")

    history = {
        "round": round_num,
        "config": cfg,
        "pre_eval": {"pope": pre_pope, "blind": pre_blind},
        "steps": [],
        "evals": [],
    }

    model.train()
    optimizer.zero_grad()
    best_acc = pre_pope["acc"]

    for step in range(cfg["num_steps"]):
        step_t0 = time.time()
        sample = train_data[step % len(train_data)]

        # Generate with thinking
        model.eval()
        try:
            candidates, cand_ids_list, prompt_len, inputs, think_ranges = \
                generate_candidates(
                    model, processor, sample, cfg["group_size"],
                    cfg["temperature"], cfg["top_p"],
                    cfg["max_new_tokens"], cfg["min_think_tokens"], device)
        except torch.cuda.OutOfMemoryError:
            print(f"  [step {step+1}] OOM gen, skip")
            torch.cuda.empty_cache(); gc.collect()
            continue

        # Gated rewards with thinking-phase LSR
        try:
            rewards, details = compute_rewards(
                model, processor, sample, candidates, cand_ids_list,
                think_ranges, prompt_len, device, cfg)
        except torch.cuda.OutOfMemoryError:
            print(f"  [step {step+1}] OOM reward, skip")
            torch.cuda.empty_cache(); gc.collect()
            continue

        # Advantage
        rarr = np.array(rewards)
        rstd = rarr.std()

        if rstd < 1e-8:
            elapsed = time.time() - step_t0
            mean_tw = np.mean([d["think_words"] for d in details])
            print(f"  [step {step+1}/{cfg['num_steps']}] "
                  f"SKIP r={rarr.mean():.3f} think={mean_tw:.0f}w "
                  f"({elapsed:.1f}s)")
            history["steps"].append({
                "step": step + 1, "skipped": True,
                "mean_reward": float(rarr.mean()),
                "mean_think_words": float(mean_tw),
            })
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
        mt = np.mean([d["think_words"] for d in details])
        mk = np.mean(lstats["kl"]) if lstats["kl"] else 0

        history["steps"].append({
            "step": step + 1, "loss": loss.item(),
            "mean_reward": float(rarr.mean()),
            "reward_std": float(rstd),
            "mean_correct": float(mc),
            "mean_lsr_raw": float(ml),
            "mean_think_words": float(mt),
            "mean_kl": float(mk),
            "elapsed": elapsed,
        })

        print(f"  [step {step+1}/{cfg['num_steps']}] "
              f"loss={loss.item():.4f} r={rarr.mean():.3f}±{rstd:.3f} "
              f"correct={mc:.2f} LSR={ml:.3f} "
              f"think={mt:.0f}w ({elapsed:.1f}s)", flush=True)

        # Periodic eval
        if (step + 1) % cfg["eval_every"] == 0:
            pope_res = evaluate_pope(
                model, processor, eval_data[:60], device)
            blind_res = evaluate_blind(
                model, processor, eval_data, device,
                min(cfg.get("blind_samples", 50), 50))
            print(f"  === Eval step {step+1}: "
                  f"POPE={pope_res['acc']:.1%} "
                  f"Gap={blind_res['gap']:.1%} "
                  f"Think={pope_res['avg_think_words']:.0f}w ===")
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
    final_pope = evaluate_pope(model, processor, eval_data[:60], device)
    final_blind = evaluate_blind(
        model, processor, eval_data, device,
        min(cfg.get("blind_samples", 50), 50))
    print(f"  POPE: {final_pope['acc']:.1%} | Gap: {final_blind['gap']:.1%} | "
          f"Think: {final_pope['avg_think_words']:.0f}w")
    history["final_eval"] = {"pope": final_pope, "blind": final_blind}

    model.save_pretrained(output_dir / "final")
    processor.save_pretrained(output_dir / "final")

    with open(output_dir / f"history_{ts}.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(report_dir / f"history_{ts}.json", "w") as f:
        json.dump(history, f, indent=2)

    generate_report(round_num, history, report_dir)

    del model, optimizer
    torch.cuda.empty_cache()
    gc.collect()

    return history, str(output_dir / "final")


# ══════════════════════════════════════════════════════════════════════
#  Main Loop
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Auto Thinking-GRPO-LSR Loop")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--steps-per-round", type=int, default=50)
    parser.add_argument("--train-samples", type=int, default=500)
    parser.add_argument("--eval-samples", type=int, default=300)
    parser.add_argument("--blind-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str,
                        default="checkpoints/grpo_lsr_thinking")
    parser.add_argument("--no-git", action="store_true")
    # Config
    parser.add_argument("--group-size", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=1.3)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--lsr-scale", type=float, default=2.0)
    parser.add_argument("--min-think-tokens", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--resume-round", type=int, default=1)
    parser.add_argument("--resume-model", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_data = load_training_data(args.train_samples, args.seed)
    eval_data = load_pope_eval(args.eval_samples)
    print(f"\n[data] {len(train_data)} train, {len(eval_data)} eval\n")

    cfg = {
        "num_steps": args.steps_per_round,
        "group_size": args.group_size,
        "temperature": args.temperature,
        "top_p": 0.95,
        "max_new_tokens": args.max_new_tokens,
        "min_think_tokens": args.min_think_tokens,
        "lr": args.lr,
        "beta_kl": 0.01,
        "beta_entropy": 0.01,
        "epsilon": 0.2,
        "grad_accum": 2,  # Smaller accum (longer sequences)
        "max_grad_norm": 1.0,
        "lsr_scale": args.lsr_scale,
        "eval_every": 10,
        "blind_samples": args.blind_samples,
        "output_dir": args.output_dir,
    }

    all_results = []
    model_path = args.resume_model

    for round_num in range(args.resume_round,
                           args.resume_round + args.max_rounds):
        print(f"\n{'#'*70}")
        print(f"#  AUTO LOOP: Thinking-GRPO-LSR Round {round_num}")
        print(f"{'#'*70}")

        history, checkpoint = run_round(
            round_num, cfg, train_data, eval_data, model_path)

        pre = history["pre_eval"]
        final = history["final_eval"]
        delta_acc = (final["pope"]["acc"] - pre["pope"]["acc"]) * 100
        delta_gap = (final["blind"]["gap"] - pre["blind"]["gap"]) * 100

        summary = (f"POPE {final['pope']['acc']*100:.1f}% "
                   f"({delta_acc:+.1f}pp), "
                   f"Gap {final['blind']['gap']*100:.1f}pp "
                   f"({delta_gap:+.1f}pp), "
                   f"Think {final['pope']['avg_think_words']:.0f}w")
        print(f"\n  Round {round_num} summary: {summary}")

        # Git
        if not args.no_git:
            journal_path = PROJECT_ROOT / "lab" / "RESEARCH_JOURNAL.md"
            with open(journal_path, "a") as f:
                f.write(f"\n\n### Thinking-GRPO-LSR Round {round_num} "
                        f"({datetime.now().strftime('%Y-%m-%d %H:%M')})\n\n")
                f.write(f"- Config: group={cfg['group_size']}, "
                        f"T={cfg['temperature']}, lr={cfg['lr']}, "
                        f"lsr_scale={cfg['lsr_scale']}\n")
                f.write(f"- Reward: R_correct*0.5 + R_correct*R_LSR*0.5 "
                        f"(gated, thinking-phase only)\n")
                f.write(f"- Pre:  POPE={pre['pope']['acc']*100:.1f}%, "
                        f"Gap={pre['blind']['gap']*100:.1f}pp, "
                        f"Think={pre['pope']['avg_think_words']:.0f}w\n")
                f.write(f"- Post: POPE={final['pope']['acc']*100:.1f}%, "
                        f"Gap={final['blind']['gap']*100:.1f}pp, "
                        f"Think={final['pope']['avg_think_words']:.0f}w\n")
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
            "pre_think_words": pre["pope"]["avg_think_words"],
            "final_think_words": final["pope"]["avg_think_words"],
            "checkpoint": checkpoint,
        })

        # Analyze and improve
        new_cfg, analysis = analyze_and_improve(round_num, history, cfg)
        print(f"\n  {analysis}")

        cfg = new_cfg
        model_path = checkpoint

        # Early stopping
        if len(all_results) >= 3:
            last3 = all_results[-3:]
            if all(r["delta_acc"] <= 0 for r in last3):
                print("\n  ⚠ 3 rounds no improvement. Stopping.")
                break

    # Final summary
    print(f"\n{'='*70}")
    print("THINKING-GRPO-LSR AUTO LOOP COMPLETE")
    print(f"{'='*70}")
    print(f"\n{'Round':<8} {'POPE':>8} {'Δ':>7} {'Gap':>8} {'Δ':>7} "
          f"{'Think':>8}")
    print("-" * 55)
    for r in all_results:
        print(f"{r['round']:<8} {r['final_pope_acc']*100:>7.1f}% "
              f"{r['delta_acc']:>+6.1f}pp "
              f"{r['final_gap']*100:>7.1f}pp "
              f"{r['delta_gap']:>+6.1f}pp "
              f"{r['final_think_words']:>7.0f}w")

    overall_path = Path(args.output_dir) / "auto_loop_results.json"
    with open(overall_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults: {overall_path}")


if __name__ == "__main__":
    main()
