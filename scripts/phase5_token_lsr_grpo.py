"""
Phase 5: Token-Level LSR-Weighted GRPO

Key innovation over Phase 2 GRPO-LSR:
  - Per-token LSR (KL divergence) used as token-level loss weight
  - token_weight(t) = 1.0 + alpha * normalized_LSR(t)
  - Decay penalty: penalizes LSR decrease over thinking chain
  - Result: model learns WHERE in the thinking chain to attend to images

Comparison with VPPO (ICLR 2026):
  - VPPO: binary mask (vision-dependent or not)
  - Ours: continuous weight proportional to LSR magnitude

Usage:
    PYTHONUNBUFFERED=1 python -u scripts/phase5_token_lsr_grpo.py \
        --steps 15 --alpha 0.5 --beta-decay 0.1 \
        2>&1 | tee logs/phase5_token_lsr_grpo.log
"""

import os, sys, gc, json, re, time, random, argparse, string
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
#  Utilities (from phase2_grpo_lsr.py)
# ══════════════════════════════════════════════════════════════════════

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

def extract_answer(raw, qtype="short_answer"):
    _, answer = split_thinking(raw)
    if not answer: answer = raw
    text = answer.strip()
    if qtype == "yesno": return extract_yes_no(raw)
    if qtype == "mc":
        for ch in text[:5]:
            if ch.upper() in "ABCDEFGH": return ch.upper()
        return text[:20]
    return text.split("\n")[0].strip()[:100]

def find_think_token_range(tokenizer, gen_ids):
    gen_list = gen_ids.tolist()
    think_end_ids = tokenizer.encode("</think>", add_special_tokens=False)
    think_start_ids = tokenizer.encode("<think>", add_special_tokens=False)
    start_idx = 0
    if think_start_ids:
        for i in range(len(gen_list)):
            if gen_list[i] == think_start_ids[0]:
                start_idx = i + 1; break
    end_idx = len(gen_list)
    if think_end_ids:
        for i in range(len(gen_list)):
            if gen_list[i] == think_end_ids[0]:
                end_idx = i; break
    return start_idx, end_idx


# ══════════════════════════════════════════════════════════════════════
#  Data Loading (from phase2_grpo_lsr.py)
# ══════════════════════════════════════════════════════════════════════

def load_training_data(limit=500, seed=42, include_pope=True):
    from datasets import load_dataset
    rng = random.Random(seed)
    samples = []

    # Load POPE-style binary VQA from GQA balanced-val (yes/no, same format as eval)
    if include_pope:
        print("[data] Loading GQA balanced-val (yes/no VQA)...")
        try:
            ds = load_dataset("lmms-lab/GQA", split="testdev_balanced",
                              streaming=True)
            count = 0
            for row in ds:
                img = row.get("image")
                if img is None: continue
                ans = str(row.get("answer", "")).strip().lower()
                if ans not in ("yes", "no"): continue
                samples.append({
                    "question": row["question"] + " Please answer yes or no.",
                    "answer": ans, "image": img,
                    "type": "yesno", "source": "gqa_yn",
                })
                count += 1
                if count >= limit // 2: break
        except Exception as e:
            print(f"  GQA error: {e}")

    print("[data] Loading A-OKVQA train...")
    try:
        ds = load_dataset("HuggingFaceM4/A-OKVQA", split="train")
        for row in ds:
            img = row.get("image")
            if img is None: continue
            choices = row.get("choices", [])
            idx = row.get("correct_choice_idx", 0)
            if not choices: continue
            choice_str = "\n".join(f"({chr(65+i)}) {c}" for i, c in enumerate(choices))
            samples.append({
                "question": f"{row['question']}\n{choice_str}\nAnswer with the letter only.",
                "answer": chr(65 + idx), "image": img,
                "type": "mc", "source": "aokvqa",
            })
    except Exception as e:
        print(f"  A-OKVQA error: {e}")

    print("[data] Loading VQAv2 train (non-binary)...")
    try:
        ds = load_dataset("HuggingFaceM4/VQAv2", split="train", streaming=True)
        count = 0
        for row in ds:
            img = row.get("image")
            if img is None: continue
            answers = row.get("answers", [])
            if not answers: continue
            if isinstance(answers[0], dict):
                ans_list = [a["answer"] for a in answers]
            else:
                ans_list = answers
            ans = Counter(ans_list).most_common(1)[0][0]
            if ans.lower() in ("yes", "no"): continue
            samples.append({
                "question": row["question"], "answer": ans, "image": img,
                "type": "short_answer", "source": "vqav2",
            })
            count += 1
            if count >= limit: break
    except Exception as e:
        print(f"  VQAv2 error: {e}")

    rng.shuffle(samples)
    samples = samples[:limit]
    src = Counter(s["source"] for s in samples)
    print(f"[data] {len(samples)} training samples "
          f"({', '.join(f'{k}={v}' for k, v in src.items())})")
    return samples

def load_pope_eval(max_samples=300):
    from datasets import load_dataset
    ds = load_dataset("lmms-lab/POPE", split="test", streaming=True)
    per_split = defaultdict(list)
    per_sample = max_samples // 3
    for row in ds:
        cat = row.get("category", "unknown")
        if cat not in POPE_SPLITS: continue
        if len(per_split[cat]) >= per_sample:
            if all(len(per_split[s]) >= per_sample for s in POPE_SPLITS): break
            continue
        per_split[cat].append({
            "image": row["image"], "question": row["question"],
            "answer": row["answer"].strip().lower(), "category": cat,
        })
    samples = []
    for s in POPE_SPLITS: samples.extend(per_split[s])
    return samples


# ══════════════════════════════════════════════════════════════════════
#  Model Loading
# ══════════════════════════════════════════════════════════════════════

def load_model(model_path=None, for_training=True):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    path = model_path or HF_ID
    print(f"[model] Loading {path} (full finetune, bfloat16)...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(HF_ID, trust_remote_code=True)
    if for_training:
        model.train()
        model.gradient_checkpointing_enable()
        for p in model.parameters(): p.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable: {trainable:,} params, gradient checkpointing ON")
    return model, processor, processor.tokenizer


# ══════════════════════════════════════════════════════════════════════
#  Input Preparation & Generation
# ══════════════════════════════════════════════════════════════════════

def prepare_inputs(processor, image, question, device):
    from qwen_vl_utils import process_vision_info
    content = [{"type": "image", "image": image},
               {"type": "text", "text": question}]
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True)
    imgs, _, _ = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(text=[text], images=imgs, return_tensors="pt", padding=True)
    return {k: v.to(device) for k, v in inputs.items()}

def generate_candidates(model, processor, sample, group_size, temperature,
                        top_p, max_new_tokens, min_think_tokens, device):
    question = sample["question"]
    image = sample["image"]
    inputs = prepare_inputs(processor, image, question, device)
    prompt_len = inputs["input_ids"].shape[1]
    candidates, candidate_ids_list, think_ranges = [], [], []
    for _ in range(group_size):
        try:
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    min_new_tokens=min_think_tokens,
                    temperature=temperature, top_p=top_p, do_sample=True)
            gen_ids = out[0][prompt_len:].clone()
            text = processor.tokenizer.decode(gen_ids, skip_special_tokens=False)
            for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
                text = text.replace(tok, "")
            candidates.append(text.strip())
            candidate_ids_list.append(gen_ids.detach())
            think_ranges.append(find_think_token_range(processor.tokenizer, gen_ids))
        except Exception:
            candidates.append("")
            candidate_ids_list.append(torch.tensor([], dtype=torch.long, device=device))
            think_ranges.append((0, 0))
    inputs = {k: v.clone() if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}
    return candidates, candidate_ids_list, prompt_len, inputs, think_ranges


# ══════════════════════════════════════════════════════════════════════
#  Token-Level LSR (KEY INNOVATION)
# ══════════════════════════════════════════════════════════════════════

def compute_token_lsr(model, processor, sample, candidate_ids,
                      think_range, device):
    """Compute per-token KL(P_real || P_black) for thinking tokens.

    Returns:
        kl_per_token: tensor of shape (think_len,) — per-token KL values
        mean_lsr: float — sequence-level mean for reward compatibility
        think_len: int — number of thinking tokens
    """
    if candidate_ids.numel() == 0:
        return torch.zeros(0, device=device), 0.0, 0

    t_start, t_end = think_range
    if t_end <= t_start:
        return torch.zeros(0, device=device), 0.0, 0

    image = sample["image"]
    question = sample["question"]
    candidate_ids = candidate_ids.clone().detach()

    # Teacher-force real image
    real_inputs = prepare_inputs(processor, image, question, device)
    rpl = real_inputs["input_ids"].shape[1]
    rf = torch.cat([real_inputs["input_ids"], candidate_ids.unsqueeze(0)], dim=1)
    real_inputs["input_ids"] = rf
    real_inputs["attention_mask"] = torch.ones_like(rf)
    with torch.no_grad():
        logits_real = model(**real_inputs).logits[0]

    # Teacher-force black image
    black_image = Image.new('RGB', image.size, (0, 0, 0))
    black_inputs = prepare_inputs(processor, black_image, question, device)
    bpl = black_inputs["input_ids"].shape[1]
    bf = torch.cat([black_inputs["input_ids"], candidate_ids.unsqueeze(0)], dim=1)
    black_inputs["input_ids"] = bf
    black_inputs["attention_mask"] = torch.ones_like(bf)
    with torch.no_grad():
        logits_black = model(**black_inputs).logits[0]

    # Extract logits for candidate tokens
    lr = logits_real[rpl - 1: rpl - 1 + len(candidate_ids)]
    lb = logits_black[bpl - 1: bpl - 1 + len(candidate_ids)]

    ml = min(lr.shape[0], lb.shape[0], len(candidate_ids))
    t_end_safe = min(t_end, ml)
    t_start_safe = min(t_start, t_end_safe)
    think_len = t_end_safe - t_start_safe

    if think_len <= 0:
        del logits_real, logits_black, real_inputs, black_inputs
        return torch.zeros(0, device=device), 0.0, 0

    lr_think = lr[t_start_safe:t_end_safe].float()
    lb_think = lb[t_start_safe:t_end_safe].float()

    # Per-token KL divergence
    kl_per_token = F.kl_div(
        F.log_softmax(lb_think, dim=-1),
        F.softmax(lr_think, dim=-1),
        reduction='none'
    ).sum(dim=-1)  # shape: (think_len,)

    mean_lsr = kl_per_token.mean().item()

    del logits_real, logits_black, lr, lb, lr_think, lb_think
    del real_inputs, black_inputs
    return kl_per_token, mean_lsr, think_len


def normalize_token_lsr(kl_per_token, eps=1e-6):
    """Normalize per-token KL so mean=1.0.
    Returns weights in [0, max_clip]."""
    if kl_per_token.numel() == 0:
        return kl_per_token
    mean_kl = kl_per_token.mean() + eps
    return kl_per_token / mean_kl


def compute_decay_penalty(kl_per_token, smooth_window=3):
    """Compute penalty for LSR decrease during thinking chain.
    Large penalty = model stops looking at image during reasoning."""
    if kl_per_token.numel() < 2:
        return 0.0

    kl = kl_per_token.float()

    # Optional smoothing to reduce noise
    if smooth_window > 1 and kl.numel() >= smooth_window:
        kernel = torch.ones(1, 1, smooth_window, device=kl.device) / smooth_window
        kl_smooth = F.conv1d(
            kl.unsqueeze(0).unsqueeze(0),
            kernel, padding=smooth_window // 2
        ).squeeze()[:kl.numel()]
    else:
        kl_smooth = kl

    # Gradient: positive = increasing LSR (good), negative = decreasing (bad)
    lsr_gradient = torch.diff(kl_smooth)

    # Penalize only decreases (negative gradients)
    decay = torch.clamp(-lsr_gradient, min=0).sum()
    decay_normalized = decay.item() / max(kl.numel() - 1, 1)
    return decay_normalized


# ══════════════════════════════════════════════════════════════════════
#  Correctness Reward
# ══════════════════════════════════════════════════════════════════════

def compute_r_correct(prediction, ground_truth, qtype="short_answer"):
    if not prediction: return 0.0
    pred = prediction.strip().lower()
    gt = ground_truth.strip().lower()
    if qtype == "yesno":
        return 1.0 if extract_yes_no(prediction) == gt else 0.0
    if qtype == "mc":
        return 1.0 if pred[:1] == gt[:1].lower() else 0.0
    pred_tokens = set(pred.split())
    gt_tokens = set(gt.split())
    if not gt_tokens: return 0.0
    overlap = pred_tokens & gt_tokens
    if not overlap: return 0.0
    p = len(overlap) / len(pred_tokens)
    r = len(overlap) / len(gt_tokens)
    return 2 * p * r / (p + r)


# ══════════════════════════════════════════════════════════════════════
#  Token-Weighted Rewards + Loss
# ══════════════════════════════════════════════════════════════════════

def compute_rewards_with_token_lsr(model, processor, sample, candidates,
                                    cand_ids_list, think_ranges, prompt_len,
                                    device, cfg):
    """Compute rewards AND per-token LSR weights for each candidate."""
    rewards, details, token_weights_list = [], [], []
    gt = sample["answer"]
    qtype = sample.get("type", "short_answer")

    for cand, cand_ids, t_range in zip(candidates, cand_ids_list, think_ranges):
        pred = extract_answer(cand, qtype)
        r_correct = compute_r_correct(pred, gt, qtype)

        try:
            kl_per_token, mean_lsr, think_len = compute_token_lsr(
                model, processor, sample, cand_ids, t_range, device)
        except Exception:
            kl_per_token = torch.zeros(0, device=device)
            mean_lsr, think_len = 0.0, 0

        # Decay penalty
        decay_pen = compute_decay_penalty(kl_per_token) if kl_per_token.numel() >= 2 else 0.0

        # Gated reward (same as Phase 2 + decay penalty)
        r_lsr = min(mean_lsr / cfg["lsr_scale"], 1.0)
        r_total = r_correct * 0.5 + r_correct * r_lsr * 0.5
        r_total -= cfg["beta_decay"] * decay_pen  # Decay penalty

        # Build per-token weights for this candidate
        # Only apply to thinking tokens; answer tokens get weight=1.0
        t_start, t_end = t_range
        n_tokens = cand_ids.numel()
        token_w = torch.ones(n_tokens, device=device)

        if kl_per_token.numel() >= 10:  # Minimum thinking length for weighting
            norm_lsr = normalize_token_lsr(kl_per_token)
            # Clip to prevent extreme weights
            norm_lsr = torch.clamp(norm_lsr, 0.0, 5.0)
            # Apply alpha-scaled weight to thinking token positions
            t_end_safe = min(t_end, n_tokens)
            t_start_safe = min(t_start, t_end_safe)
            w_len = min(norm_lsr.numel(), t_end_safe - t_start_safe)
            if w_len > 0:
                token_w[t_start_safe:t_start_safe + w_len] = (
                    1.0 + cfg["alpha"] * norm_lsr[:w_len])

        rewards.append(r_total)
        token_weights_list.append(token_w)
        details.append({
            "pred": pred, "gt": gt, "correct": r_correct,
            "lsr_raw": mean_lsr, "lsr_norm": r_lsr,
            "decay_penalty": decay_pen,
            "think_tokens": think_len,
            "total": r_total,
            "token_weight_mean": token_w.mean().item(),
            "token_weight_max": token_w.max().item(),
        })

    return rewards, details, token_weights_list


def compute_weighted_logprobs(model, inputs, candidate_ids, prompt_len,
                               token_weights=None):
    """Compute weighted mean of per-token log probs."""
    if candidate_ids.numel() == 0:
        dev = candidate_ids.device
        return torch.tensor(0.0, device=dev), torch.tensor(0.0, device=dev)

    candidate_ids = candidate_ids.clone().detach()
    full_ids = torch.cat([inputs["input_ids"][:, :prompt_len].clone(),
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

    # Entropy for collapse prevention
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()

    if token_weights is not None:
        # Align weights to token count
        w = token_weights[:len(token_lp)]
        if len(w) < len(token_lp):
            w = F.pad(w, (0, len(token_lp) - len(w)), value=1.0)
        weighted_lp = (token_lp * w).sum() / (w.sum() + 1e-8)
    else:
        weighted_lp = token_lp.mean()

    return weighted_lp, entropy


def compute_token_lsr_grpo_loss(model, inputs, cand_ids_list, prompt_len,
                                 advantages, token_weights_list, cfg):
    """GRPO loss with per-token LSR weighting."""
    total_loss = torch.tensor(0.0, device=next(model.parameters()).device,
                              requires_grad=True)
    n_valid = 0
    stats = {"entropy": [], "token_weight_mean": []}

    for cand_ids, adv, tw in zip(cand_ids_list, advantages, token_weights_list):
        if cand_ids.numel() == 0 or abs(adv) < 1e-8:
            continue

        weighted_lp, entropy = compute_weighted_logprobs(
            model, inputs, cand_ids, prompt_len, tw)

        # REINFORCE with token-level weighting
        policy_loss = -weighted_lp * adv
        loss = policy_loss - cfg["beta_entropy"] * entropy

        total_loss = total_loss + loss
        n_valid += 1
        stats["entropy"].append(entropy.item())
        stats["token_weight_mean"].append(tw.mean().item())

    if n_valid > 0:
        total_loss = total_loss / n_valid
    return total_loss, stats


# ══════════════════════════════════════════════════════════════════════
#  Evaluation (from phase2_grpo_lsr.py)
# ══════════════════════════════════════════════════════════════════════

def evaluate_pope(model, processor, samples, device, max_eval=60):
    was_training = model.training
    model.eval()
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    correct = total = 0
    think_lengths = []
    for s in samples[:max_eval]:
        try:
            q = s["question"] + " Please answer yes or no."
            inputs = prepare_inputs(processor, s["image"], q, device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            raw = processor.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
            for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
                raw = raw.replace(tok, "")
            pred = extract_yes_no(raw)
            gt = s["answer"]
            if pred == gt: correct += 1
            total += 1
            thinking, _ = split_thinking(raw)
            think_lengths.append(len(thinking.split()) if thinking else 0)
        except Exception:
            total += 1
    if was_training:
        model.train()
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
    acc = correct / total if total > 0 else 0
    return {"acc": acc, "total": total,
            "avg_think_words": np.mean(think_lengths) if think_lengths else 0}

def evaluate_blind(model, processor, samples, device, n=50):
    was_training = model.training
    model.eval()
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    real_c = blind_c = total = 0
    for s in samples[:n]:
        try:
            q = s["question"] + " Please answer yes or no."
            gt = s["answer"]
            inputs = prepare_inputs(processor, s["image"], q, device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            raw = processor.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
            if extract_yes_no(raw) == gt: real_c += 1
            black = Image.new('RGB', s["image"].size, (0, 0, 0))
            inputs_b = prepare_inputs(processor, black, q, device)
            with torch.no_grad():
                out_b = model.generate(**inputs_b, max_new_tokens=512, do_sample=False)
            raw_b = processor.tokenizer.decode(
                out_b[0][inputs_b["input_ids"].shape[1]:], skip_special_tokens=False)
            if extract_yes_no(raw_b) == gt: blind_c += 1
            total += 1
        except Exception:
            total += 1
    if was_training:
        model.train()
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
    ra = real_c / total if total > 0 else 0
    ba = blind_c / total if total > 0 else 0
    return {"real_acc": ra, "blind_acc": ba, "gap": ra - ba, "total": total}


# ══════════════════════════════════════════════════════════════════════
#  Report Generation
# ══════════════════════════════════════════════════════════════════════

def generate_report(history, report_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    steps = [s for s in history["steps"] if not s.get("skipped")]
    if not steps:
        return

    # Plot 1: Loss + Reward
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = [s["step"] for s in steps]
    ax1.plot(x, [s["loss"] for s in steps], 'b-o', label="Loss")
    ax1.set_xlabel("Step"); ax1.set_ylabel("Loss"); ax1.legend()
    ax1.set_title("Training Loss")

    ax2.plot(x, [s["mean_reward"] for s in steps], 'g-o', label="Reward")
    ax2.plot(x, [s.get("mean_decay_pen", 0) for s in steps], 'r--s',
             label="Decay Penalty", alpha=0.7)
    ax2.set_xlabel("Step"); ax2.set_ylabel("Value"); ax2.legend()
    ax2.set_title("Reward + Decay Penalty")
    plt.tight_layout()
    plt.savefig(report_dir / "fig1_training_curves.png", dpi=150)
    plt.close()

    # Plot 2: Token weight stats
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, [s.get("token_weight_mean", 1.0) for s in steps], 'b-o',
            label="Mean weight")
    ax.plot(x, [s.get("token_weight_max", 1.0) for s in steps], 'r--s',
            label="Max weight")
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel("Step"); ax.set_ylabel("Token Weight")
    ax.legend(); ax.set_title("Token-Level LSR Weights During Training")
    plt.tight_layout()
    plt.savefig(report_dir / "fig2_token_weights.png", dpi=150)
    plt.close()

    # Plot 3: Eval progression
    evals = history.get("evals", [])
    if evals:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ex = [e["step"] for e in evals]
        ax1.plot(ex, [e["pope"]["acc"] * 100 for e in evals], 'b-o')
        ax1.set_xlabel("Step"); ax1.set_ylabel("POPE Acc (%)")
        ax1.set_title("POPE Accuracy")
        ax2.plot(ex, [e["blind"]["gap"] * 100 for e in evals], 'g-o')
        ax2.set_xlabel("Step"); ax2.set_ylabel("Blind Gap (pp)")
        ax2.set_title("Blind Test Gap")
        plt.tight_layout()
        plt.savefig(report_dir / "fig3_eval_progression.png", dpi=150)
        plt.close()

    # Save history
    with open(report_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"  [report] Saved to {report_dir}")


# ══════════════════════════════════════════════════════════════════════
#  Main Training Loop
# ══════════════════════════════════════════════════════════════════════

def run_training(cfg, train_data, eval_data, model_path=None):
    output_dir = Path(cfg["output_dir"])
    # Use output_dir basename for per-run reports
    run_name = output_dir.name
    report_dir = Path("lab/reports/phase5_token_lsr_grpo") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  Phase 5: Token-Level LSR-Weighted GRPO")
    print(f"  alpha={cfg['alpha']} | beta_decay={cfg['beta_decay']} | "
          f"steps={cfg['num_steps']} | group={cfg['group_size']} | "
          f"T={cfg['temperature']} | lr={cfg['lr']}")
    print(f"{'='*70}\n")

    model, processor, tokenizer = load_model(model_path, for_training=True)
    device = next(model.parameters()).device

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr"], weight_decay=0.01)

    # Pre-eval
    print("Pre-training eval...")
    pre_pope = evaluate_pope(model, processor, eval_data, device, 60)
    pre_blind = evaluate_blind(model, processor, eval_data, device, 50)
    print(f"  POPE: {pre_pope['acc']:.1%} | Gap: {pre_blind['gap']:.1%}")

    history = {
        "config": cfg,
        "pre_eval": {"pope": pre_pope, "blind": pre_blind},
        "steps": [], "evals": [],
    }

    model.train()
    optimizer.zero_grad()
    best_acc = pre_pope["acc"]

    for step in range(cfg["num_steps"]):
        step_t0 = time.time()
        sample = train_data[step % len(train_data)]

        # Generate candidates
        model.eval()
        if hasattr(model, 'gradient_checkpointing_disable'):
            model.gradient_checkpointing_disable()
        try:
            candidates, cand_ids_list, prompt_len, inputs, think_ranges = \
                generate_candidates(
                    model, processor, sample, cfg["group_size"],
                    cfg["temperature"], cfg["top_p"],
                    cfg["max_new_tokens"], cfg["min_think_tokens"], device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); gc.collect()
            print(f"  [step {step+1}] OOM gen, skip"); continue

        # Rewards + token-level LSR weights
        try:
            rewards, details, token_weights_list = compute_rewards_with_token_lsr(
                model, processor, sample, candidates, cand_ids_list,
                think_ranges, prompt_len, device, cfg)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); gc.collect()
            print(f"  [step {step+1}] OOM reward, skip"); continue

        rarr = np.array(rewards)
        rstd = rarr.std()

        # Dynamic resampling (DAPO): if zero variance, try another sample
        if rstd < 1e-8:
            resample_tries = 0
            while rstd < 1e-8 and resample_tries < 3:
                resample_tries += 1
                alt_idx = (step * 7 + resample_tries * 13) % len(train_data)
                alt_sample = train_data[alt_idx]
                try:
                    model.eval()
                    if hasattr(model, 'gradient_checkpointing_disable'):
                        model.gradient_checkpointing_disable()
                    candidates, cand_ids_list, prompt_len, inputs, think_ranges = \
                        generate_candidates(
                            model, processor, alt_sample, cfg["group_size"],
                            cfg["temperature"], cfg["top_p"],
                            cfg["max_new_tokens"], cfg["min_think_tokens"], device)
                    rewards, details, token_weights_list = compute_rewards_with_token_lsr(
                        model, processor, alt_sample, candidates, cand_ids_list,
                        think_ranges, prompt_len, device, cfg)
                    sample = alt_sample  # use this sample for loss
                    rarr = np.array(rewards)
                    rstd = rarr.std()
                except Exception:
                    break

            if rstd < 1e-8:
                elapsed = time.time() - step_t0
                print(f"  [step {step+1}/{cfg['num_steps']}] "
                      f"SKIP r={rarr.mean():.3f} ({elapsed:.1f}s)")
                history["steps"].append({
                    "step": step + 1, "skipped": True,
                    "mean_reward": float(rarr.mean())})
                del candidates, cand_ids_list, inputs; continue

        advantages = ((rarr - rarr.mean()) / (rstd + 1e-8)).tolist()

        # Loss with token-level weighting
        model.train()
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        try:
            loss, lstats = compute_token_lsr_grpo_loss(
                model, inputs, cand_ids_list, prompt_len,
                advantages, token_weights_list, cfg)
            (loss / cfg["grad_accum"]).backward()

            if (step + 1) % cfg["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.get("max_grad_norm", 1.0))
                optimizer.step()
                optimizer.zero_grad()
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad()
            torch.cuda.empty_cache(); gc.collect()
            print(f"  [step {step+1}] OOM loss, skip"); continue

        elapsed = time.time() - step_t0
        mc = np.mean([d["correct"] for d in details])
        ml = np.mean([d["lsr_raw"] for d in details])
        md = np.mean([d["decay_penalty"] for d in details])
        tw_mean = np.mean([d["token_weight_mean"] for d in details])
        tw_max = np.mean([d["token_weight_max"] for d in details])

        history["steps"].append({
            "step": step + 1, "loss": loss.item(),
            "mean_reward": float(rarr.mean()),
            "reward_std": float(rstd),
            "mean_correct": float(mc),
            "mean_lsr_raw": float(ml),
            "mean_decay_pen": float(md),
            "token_weight_mean": float(tw_mean),
            "token_weight_max": float(tw_max),
            "mean_entropy": float(np.mean(lstats["entropy"])) if lstats["entropy"] else 0,
            "elapsed": elapsed,
        })

        print(f"  [step {step+1}/{cfg['num_steps']}] "
              f"loss={loss.item():.4f} r={rarr.mean():.3f}±{rstd:.3f} "
              f"correct={mc:.2f} LSR={ml:.3f} decay={md:.3f} "
              f"tw={tw_mean:.2f}/{tw_max:.1f} ({elapsed:.1f}s)", flush=True)

        # Eval
        if (step + 1) % cfg["eval_every"] == 0 or step + 1 == cfg["num_steps"]:
            pope_res = evaluate_pope(model, processor, eval_data, device, 60)
            blind_res = evaluate_blind(model, processor, eval_data, device, 50)
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

        del candidates, cand_ids_list, inputs, token_weights_list

    # Final eval
    history["final_eval"] = history["evals"][-1] if history["evals"] else {}

    # Save final model
    model.save_pretrained(output_dir / "final")
    processor.save_pretrained(output_dir / "final")

    # Generate report
    generate_report(history, report_dir)

    # Summary
    pre_acc = pre_pope["acc"]
    pre_gap = pre_blind["gap"]
    final = history.get("final_eval", {})
    f_acc = final.get("pope", {}).get("acc", 0)
    f_gap = final.get("blind", {}).get("gap", 0)

    print(f"\n{'='*70}")
    print(f"  Phase 5 COMPLETE")
    print(f"  POPE: {pre_acc:.1%} → {f_acc:.1%} ({(f_acc-pre_acc)*100:+.1f}pp)")
    print(f"  Gap:  {pre_gap:.1%} → {f_gap:.1%} ({(f_gap-pre_gap)*100:+.1f}pp)")
    print(f"  Best: {best_acc:.1%}")
    print(f"  Saved to {output_dir}")
    print(f"{'='*70}")

    del model; torch.cuda.empty_cache(); gc.collect()
    return history


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 5: Token-LSR GRPO")
    parser.add_argument("--model-path", type=str,
                        default="checkpoints/phase2_grpo_lsr/round4/best")
    parser.add_argument("--output-dir", type=str,
                        default="checkpoints/phase5_token_lsr_grpo")
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--group-size", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=1.3)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Token-level LSR weight scale (0=standard GRPO)")
    parser.add_argument("--beta-decay", type=float, default=0.1,
                        help="Decay penalty weight (0=no penalty)")
    parser.add_argument("--lsr-scale", type=float, default=2.0)
    parser.add_argument("--beta-entropy", type=float, default=0.01)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--min-think-tokens", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--train-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = vars(args)
    cfg["num_steps"] = cfg.pop("steps")

    train_data = load_training_data(args.train_samples, args.seed)
    eval_data = load_pope_eval(300)

    run_training(cfg, train_data, eval_data, args.model_path)


if __name__ == "__main__":
    main()
