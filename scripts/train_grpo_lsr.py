"""
GRPO Training with Logit-Shift Reward (LSR)

LSR = D_KL(P_real || P_black) — measures how much the image influences next-token
logit distributions. Correct rollouts have higher LSR (Cohen's d = 0.604, validated).

Pipeline per step:
  1. Sample → generate group_size candidates with real image
  2. Per candidate: teacher-force with real image → logits_real
  3. Per candidate: teacher-force with black image → logits_black
  4. R_LSR = mean D_KL(P_real || P_black) over generated tokens
  5. R_total = w_correct * R_correct + w_lsr * R_LSR + w_fluency * R_fluency
  6. GRPO advantage → clipped surrogate loss → update

Usage:
    PYTHONUNBUFFERED=1 python -u scripts/train_grpo_lsr.py \
        --train-samples 500 --num-steps 50 --eval-every 10 \
        2>&1 | tee logs/grpo_lsr.log
"""

import os, sys, gc, json, re, time, random, argparse, string
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from datetime import datetime
from collections import defaultdict
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Constants ──────────────────────────────────────────────────────────

HF_ID = "Qwen/Qwen3-VL-2B-Instruct"
POPE_SPLITS = ["random", "popular", "adversarial"]


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
    """Load mixed non-binary VQA training data."""
    from datasets import load_dataset

    rng = random.Random(seed)
    samples = []

    # TextVQA train (best for diverse answers)
    print("[data] Loading TextVQA train...")
    try:
        ds = load_dataset("textvqa", split="train", trust_remote_code=True)
        for row in ds:
            img = row.get("image")
            if img is None:
                continue
            answers = row.get("answers", [])
            if not answers:
                continue
            # Most common answer
            from collections import Counter
            ans = Counter(answers).most_common(1)[0][0]
            if ans.lower() in ("yes", "no"):
                continue
            samples.append({
                "question": row["question"],
                "answer": ans,
                "image": img,
                "type": "short_answer",
                "source": "textvqa",
            })
            if len(samples) >= limit * 2:
                break
    except Exception as e:
        print(f"  TextVQA error: {e}")

    # A-OKVQA (multiple choice — good diversity)
    if len(samples) < limit:
        print("[data] Loading A-OKVQA train...")
        try:
            ds = load_dataset("HuggingFaceM4/A-OKVQA", split="train",
                              trust_remote_code=True)
            for row in ds:
                img = row.get("image")
                if img is None:
                    continue
                choices = row.get("choices", [])
                idx = row.get("correct_choice_idx", 0)
                if not choices:
                    continue
                ans = choices[idx] if idx < len(choices) else choices[0]
                q = row["question"]
                choice_str = "\n".join(
                    f"({chr(65+i)}) {c}" for i, c in enumerate(choices))
                samples.append({
                    "question": f"{q}\n{choice_str}\nAnswer with the letter only.",
                    "answer": chr(65 + idx),
                    "image": img,
                    "type": "mc",
                    "source": "aokvqa",
                })
                if len(samples) >= limit * 2:
                    break
        except Exception as e:
            print(f"  A-OKVQA error: {e}")

    rng.shuffle(samples)
    samples = samples[:limit]
    print(f"[data] {len(samples)} training samples")
    return samples


def load_pope_eval(max_samples=200):
    """Load POPE for evaluation."""
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
    print(f"[model] Loading {path} (dtype={dtype})...")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        path, dtype=dtype, device_map="auto")
    processor = AutoProcessor.from_pretrained(
        HF_ID, trust_remote_code=True)

    if for_training:
        # LoRA for efficient training + reference model via adapter toggle
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
    """Generate group_size diverse candidates for a sample."""
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
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                )
            gen_ids = out[0][prompt_len:]
            text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
            candidates.append(text.strip())
            candidate_ids_list.append(gen_ids.detach())
        except Exception as e:
            candidates.append("")
            candidate_ids_list.append(torch.tensor([], dtype=torch.long,
                                                    device=device))

    return candidates, candidate_ids_list, prompt_len, inputs


# ── LSR Computation ────────────────────────────────────────────────────

def compute_lsr_for_candidate(model, processor, sample, candidate_ids,
                               prompt_len, device):
    """
    Compute Logit-Shift Reward for a single candidate.

    Returns mean D_KL(P_real || P_black) over generated tokens.
    Cost: 2 forward passes (teacher-force with real + black image).
    """
    if candidate_ids.numel() == 0:
        return 0.0

    image = sample["image"]
    question = sample["question"]

    # Teacher-force with REAL image
    real_inputs = prepare_inputs(processor, image, question, device)
    real_prompt_len = real_inputs["input_ids"].shape[1]
    real_full = torch.cat([real_inputs["input_ids"],
                           candidate_ids.unsqueeze(0)], dim=1)
    real_inputs["input_ids"] = real_full
    real_inputs["attention_mask"] = torch.ones_like(real_full)

    with torch.no_grad():
        logits_real = model(**real_inputs).logits[0]

    # Teacher-force with BLACK image
    black_image = Image.new('RGB', image.size, (0, 0, 0))
    black_inputs = prepare_inputs(processor, black_image, question, device)
    black_prompt_len = black_inputs["input_ids"].shape[1]
    black_full = torch.cat([black_inputs["input_ids"],
                            candidate_ids.unsqueeze(0)], dim=1)
    black_inputs["input_ids"] = black_full
    black_inputs["attention_mask"] = torch.ones_like(black_full)

    with torch.no_grad():
        logits_black = model(**black_inputs).logits[0]

    # Extract logits at generated token positions (shifted by 1)
    lr = logits_real[real_prompt_len - 1: real_prompt_len - 1 + len(candidate_ids)]
    lb = logits_black[black_prompt_len - 1: black_prompt_len - 1 + len(candidate_ids)]

    min_len = min(lr.shape[0], lb.shape[0])
    if min_len == 0:
        return 0.0

    lr = lr[:min_len].float()
    lb = lb[:min_len].float()

    # Per-token KL: D_KL(P_real || P_black)
    kl_per_token = F.kl_div(
        F.log_softmax(lb, dim=-1),
        F.softmax(lr, dim=-1),
        reduction='none'
    ).sum(dim=-1)

    mean_kl = kl_per_token.mean().item()

    # Cleanup
    del logits_real, logits_black, lr, lb, kl_per_token
    del real_inputs, black_inputs

    return mean_kl


# ── Reward Computation ─────────────────────────────────────────────────

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
    # Short answer: F1 token overlap
    pred_tokens = set(pred.split())
    gt_tokens = set(gt.split())
    if not gt_tokens:
        return 0.0
    overlap = pred_tokens & gt_tokens
    if not overlap:
        return 0.0
    precision = len(overlap) / len(pred_tokens)
    recall = len(overlap) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_r_fluency(text, max_tokens=64):
    if not text or len(text) < 3:
        return 0.0
    if len(set(text)) < 3:
        return 0.0
    words = text.split()
    if len(words) > max_tokens * 2:
        return max(0, 1.0 - (len(words) - max_tokens) / max_tokens * 0.5)
    return 1.0


def compute_rewards(model, processor, sample, candidates, candidate_ids_list,
                    prompt_len, device, w_correct, w_lsr, w_fluency,
                    lsr_scale):
    """Compute composite reward for each candidate in the group."""
    rewards = []
    details = []
    gt = sample["answer"]
    qtype = sample.get("type", "short_answer")

    for i, (cand, cand_ids) in enumerate(zip(candidates, candidate_ids_list)):
        pred = extract_answer(cand, qtype)
        r_correct = compute_r_correct(pred, gt, qtype)
        r_fluency = compute_r_fluency(cand)

        # LSR: teacher-force with real vs black → KL divergence
        try:
            r_lsr_raw = compute_lsr_for_candidate(
                model, processor, sample, cand_ids, prompt_len, device)
            # Normalize: typical KL range is 0-3, scale to ~[0,1]
            r_lsr = min(r_lsr_raw / lsr_scale, 1.0)
        except Exception as e:
            r_lsr_raw = 0.0
            r_lsr = 0.0

        r_total = (w_correct * r_correct +
                   w_lsr * r_lsr +
                   w_fluency * r_fluency)

        rewards.append(r_total)
        details.append({
            "pred": pred, "gt": gt, "correct": r_correct,
            "lsr_raw": r_lsr_raw, "lsr_norm": r_lsr,
            "fluency": r_fluency, "total": r_total,
        })

    return rewards, details


# ── Log-Prob Computation ───────────────────────────────────────────────

def compute_logprobs(model, inputs, candidate_ids, prompt_len):
    """Compute per-token log-probs for a candidate (with gradient)."""
    if candidate_ids.numel() == 0:
        return torch.tensor(0.0, device=candidate_ids.device), torch.tensor(0.0)

    full_ids = torch.cat([inputs["input_ids"][:, :prompt_len],
                          candidate_ids.unsqueeze(0)], dim=1)
    attn = torch.ones_like(full_ids)

    fwd_inputs = {k: v for k, v in inputs.items()
                  if k not in ("input_ids", "attention_mask")}
    fwd_inputs["input_ids"] = full_ids
    fwd_inputs["attention_mask"] = attn

    out = model(**fwd_inputs)
    logits = out.logits[0, prompt_len - 1: prompt_len - 1 + len(candidate_ids)]

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(
        1, candidate_ids.unsqueeze(1)).squeeze(1)

    # Entropy for regularization
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()

    return token_log_probs, entropy


def compute_ref_logprobs(model, inputs, candidate_ids, prompt_len):
    """Reference log-probs via LoRA disable."""
    model.disable_adapter_layers()
    with torch.no_grad():
        lp, _ = compute_logprobs(model, inputs, candidate_ids, prompt_len)
    model.enable_adapter_layers()
    return lp.detach()


# ── GRPO Loss ──────────────────────────────────────────────────────────

def compute_grpo_loss(model, inputs, candidate_ids_list, prompt_len,
                      advantages, beta_kl, beta_entropy, epsilon_clip):
    """Clipped surrogate GRPO loss with KL penalty."""
    total_loss = torch.tensor(0.0, device=next(model.parameters()).device,
                              requires_grad=True)
    n_valid = 0
    stats = {"kl": [], "entropy": [], "ratio": []}

    for cand_ids, adv in zip(candidate_ids_list, advantages):
        if cand_ids.numel() == 0 or abs(adv) < 1e-8:
            continue

        # Current policy log-probs (with gradient)
        cur_lp, entropy = compute_logprobs(
            model, inputs, cand_ids, prompt_len)

        # Reference log-probs (no gradient)
        ref_lp = compute_ref_logprobs(
            model, inputs, cand_ids, prompt_len)

        # KL divergence
        kl = (torch.exp(ref_lp) * (ref_lp - cur_lp)).mean()
        if kl.item() > 10.0:  # Skip extreme KL
            continue

        # Clipped surrogate
        log_ratio = (cur_lp - ref_lp).sum()
        ratio = torch.exp(log_ratio.clamp(-5, 5))
        clipped = torch.clamp(ratio, 1 - epsilon_clip, 1 + epsilon_clip)
        surr = torch.min(ratio * adv, clipped * adv)

        loss = -surr + beta_kl * kl - beta_entropy * entropy
        total_loss = total_loss + loss
        n_valid += 1

        stats["kl"].append(kl.item())
        stats["entropy"].append(entropy.item())
        stats["ratio"].append(ratio.item())

    if n_valid > 0:
        total_loss = total_loss / n_valid

    return total_loss, stats


# ── Evaluation ─────────────────────────────────────────────────────────

def evaluate_pope(model, processor, samples, device, batch_label="eval"):
    """Quick POPE evaluation."""
    model.eval()
    correct = 0
    total = 0
    yes_count = 0

    for i, s in enumerate(samples):
        try:
            inputs = prepare_inputs(processor, s["image"],
                                    s["question"] + " Please answer yes or no.",
                                    device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=32,
                                     do_sample=False)
            gen = out[0][inputs["input_ids"].shape[1]:]
            text = processor.tokenizer.decode(gen, skip_special_tokens=True)
            pred = extract_yes_no(text)
            gt = s["answer"].strip().lower()
            if pred == gt:
                correct += 1
            if pred == "yes":
                yes_count += 1
            total += 1
        except Exception:
            total += 1

    acc = correct / total if total > 0 else 0
    yes_rate = yes_count / total if total > 0 else 0
    model.train()
    return {"acc": acc, "total": total, "yes_rate": yes_rate}


def evaluate_blind(model, processor, samples, device, n=50):
    """Blind test: replace images with black → compute gap."""
    model.eval()
    real_correct = 0
    blind_correct = 0
    total = 0

    for s in samples[:n]:
        try:
            q = s["question"] + " Please answer yes or no."
            gt = s["answer"].strip().lower()

            # Real image
            inputs = prepare_inputs(processor, s["image"], q, device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=32,
                                     do_sample=False)
            gen = out[0][inputs["input_ids"].shape[1]:]
            pred_real = extract_yes_no(
                processor.tokenizer.decode(gen, skip_special_tokens=True))
            if pred_real == gt:
                real_correct += 1

            # Black image
            black = Image.new('RGB', s["image"].size, (0, 0, 0))
            inputs_b = prepare_inputs(processor, black, q, device)
            with torch.no_grad():
                out_b = model.generate(**inputs_b, max_new_tokens=32,
                                       do_sample=False)
            gen_b = out_b[0][inputs_b["input_ids"].shape[1]:]
            pred_blind = extract_yes_no(
                processor.tokenizer.decode(gen_b, skip_special_tokens=True))
            if pred_blind == gt:
                blind_correct += 1

            total += 1
        except Exception:
            total += 1

    model.train()
    real_acc = real_correct / total if total > 0 else 0
    blind_acc = blind_correct / total if total > 0 else 0
    gap = real_acc - blind_acc
    return {"real_acc": real_acc, "blind_acc": blind_acc, "gap": gap,
            "total": total}


# ── Main Training Loop ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GRPO with LSR reward")
    # Data
    parser.add_argument("--train-samples", type=int, default=500)
    parser.add_argument("--eval-samples", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    # Model
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str,
                        default="checkpoints/grpo_lsr")
    # GRPO
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta-kl", type=float, default=0.01)
    parser.add_argument("--beta-entropy", type=float, default=0.01)
    parser.add_argument("--epsilon-clip", type=float, default=0.2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    # Reward weights
    parser.add_argument("--w-correct", type=float, default=0.4)
    parser.add_argument("--w-lsr", type=float, default=0.4)
    parser.add_argument("--w-fluency", type=float, default=0.2)
    parser.add_argument("--lsr-scale", type=float, default=2.0,
                        help="Normalization: r_lsr = min(raw_kl / scale, 1.0)")
    # Eval
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--blind-samples", type=int, default=50)
    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint dir")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── Load data ──────────────────────────────────────────────────
    train_data = load_training_data(args.train_samples, args.seed)
    eval_data = load_pope_eval(args.eval_samples)
    print(f"[data] {len(train_data)} train, {len(eval_data)} eval")

    # ── Load model ─────────────────────────────────────────────────
    model, processor = load_model(args.model_path, for_training=True)
    device = next(model.parameters()).device

    # ── Optimizer ──────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01)

    # ── Pre-training eval ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Pre-training evaluation")
    print("=" * 60)
    pre_pope = evaluate_pope(model, processor, eval_data, device, "pre")
    pre_blind = evaluate_blind(model, processor, eval_data, device,
                               args.blind_samples)
    print(f"  POPE: {pre_pope['acc']:.1%} (yes_rate={pre_pope['yes_rate']:.1%})")
    print(f"  Blind: real={pre_blind['real_acc']:.1%} "
          f"blind={pre_blind['blind_acc']:.1%} "
          f"gap={pre_blind['gap']:.1%}")

    # ── Training loop ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"GRPO-LSR Training: {args.num_steps} steps, "
          f"group={args.group_size}, T={args.temperature}")
    print(f"Reward: w_correct={args.w_correct}, w_lsr={args.w_lsr}, "
          f"w_fluency={args.w_fluency}")
    print("=" * 60)

    history = {
        "steps": [],
        "pre_eval": {"pope": pre_pope, "blind": pre_blind},
        "evals": [],
    }
    best_acc = pre_pope["acc"]
    start_step = 0

    model.train()
    optimizer.zero_grad()
    accum_loss = 0.0
    accum_count = 0

    for step in range(start_step, args.num_steps):
        step_t0 = time.time()
        sample = train_data[step % len(train_data)]

        # ── Generate candidates ────────────────────────────────────
        model.eval()
        try:
            candidates, cand_ids_list, prompt_len, inputs = \
                generate_candidates(
                    model, processor, sample, args.group_size,
                    args.temperature, args.top_p,
                    args.max_new_tokens, device)
        except torch.cuda.OutOfMemoryError:
            print(f"  [step {step+1}] OOM during generation, skipping")
            torch.cuda.empty_cache()
            gc.collect()
            continue

        # ── Compute rewards ────────────────────────────────────────
        try:
            rewards, details = compute_rewards(
                model, processor, sample, candidates, cand_ids_list,
                prompt_len, device,
                args.w_correct, args.w_lsr, args.w_fluency,
                args.lsr_scale)
        except torch.cuda.OutOfMemoryError:
            print(f"  [step {step+1}] OOM during reward, skipping")
            torch.cuda.empty_cache()
            gc.collect()
            continue

        # ── Advantage ──────────────────────────────────────────────
        reward_arr = np.array(rewards)
        reward_std = reward_arr.std()

        if reward_std < 1e-8:
            # Zero variance → skip (no gradient signal)
            elapsed = time.time() - step_t0
            print(f"  [step {step+1}/{args.num_steps}] "
                  f"SKIP (zero-var) r={reward_arr.mean():.3f} "
                  f"({elapsed:.1f}s)")
            history["steps"].append({
                "step": step + 1, "skipped": True,
                "mean_reward": float(reward_arr.mean()),
            })
            continue

        advantages = ((reward_arr - reward_arr.mean()) /
                      (reward_std + 1e-8)).tolist()

        # ── GRPO loss ──────────────────────────────────────────────
        model.train()
        try:
            loss, loss_stats = compute_grpo_loss(
                model, inputs, cand_ids_list, prompt_len,
                advantages, args.beta_kl, args.beta_entropy,
                args.epsilon_clip)

            # Gradient accumulation
            (loss / args.grad_accum).backward()
            accum_loss += loss.item()
            accum_count += 1

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

        except torch.cuda.OutOfMemoryError:
            print(f"  [step {step+1}] OOM during loss, skipping")
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            gc.collect()
            continue

        elapsed = time.time() - step_t0

        # ── Logging ────────────────────────────────────────────────
        mean_correct = np.mean([d["correct"] for d in details])
        mean_lsr = np.mean([d["lsr_raw"] for d in details])
        mean_kl = np.mean(loss_stats["kl"]) if loss_stats["kl"] else 0

        step_info = {
            "step": step + 1,
            "loss": loss.item(),
            "mean_reward": float(reward_arr.mean()),
            "reward_std": float(reward_std),
            "mean_correct": float(mean_correct),
            "mean_lsr_raw": float(mean_lsr),
            "mean_kl": float(mean_kl),
            "elapsed": elapsed,
        }
        history["steps"].append(step_info)

        print(f"  [step {step+1}/{args.num_steps}] "
              f"loss={loss.item():.4f} "
              f"r={reward_arr.mean():.3f}±{reward_std:.3f} "
              f"correct={mean_correct:.2f} "
              f"LSR={mean_lsr:.3f} "
              f"KL={mean_kl:.4f} "
              f"({elapsed:.1f}s)", flush=True)

        # ── Periodic eval ──────────────────────────────────────────
        if (step + 1) % args.eval_every == 0:
            print(f"\n  === Eval at step {step+1} ===")
            pope_res = evaluate_pope(model, processor, eval_data, device)
            blind_res = evaluate_blind(model, processor, eval_data, device,
                                       args.blind_samples)
            print(f"  POPE: {pope_res['acc']:.1%} "
                  f"(yes={pope_res['yes_rate']:.1%})")
            print(f"  Blind: real={blind_res['real_acc']:.1%} "
                  f"blind={blind_res['blind_acc']:.1%} "
                  f"gap={blind_res['gap']:.1%}")

            eval_info = {
                "step": step + 1,
                "pope": pope_res,
                "blind": blind_res,
            }
            history["evals"].append(eval_info)

            # Save best
            if pope_res["acc"] > best_acc:
                best_acc = pope_res["acc"]
                model.save_pretrained(output_dir / "best")
                processor.save_pretrained(output_dir / "best")
                print(f"  ★ New best: {best_acc:.1%}")

            # Collapse detection
            if pope_res["yes_rate"] > 0.9 or pope_res["yes_rate"] < 0.1:
                print(f"  ⚠ COLLAPSE DETECTED (yes_rate={pope_res['yes_rate']:.1%})")
                model.save_pretrained(output_dir / "pre_collapse")
                # Don't stop — let it recover with entropy bonus

            print()

        # Cleanup
        del loss, candidates, cand_ids_list, inputs
        torch.cuda.empty_cache()
        if step % 5 == 0:
            gc.collect()

    # ── Final eval ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Final evaluation")
    print("=" * 60)
    final_pope = evaluate_pope(model, processor, eval_data, device, "final")
    final_blind = evaluate_blind(model, processor, eval_data, device,
                                  args.blind_samples)
    print(f"  POPE: {final_pope['acc']:.1%} (yes={final_pope['yes_rate']:.1%})")
    print(f"  Blind: real={final_blind['real_acc']:.1%} "
          f"blind={final_blind['blind_acc']:.1%} "
          f"gap={final_blind['gap']:.1%}")

    history["final_eval"] = {"pope": final_pope, "blind": final_blind}

    # ── Save ───────────────────────────────────────────────────────
    model.save_pretrained(output_dir / "final")
    processor.save_pretrained(output_dir / "final")

    with open(output_dir / f"history_{ts}.json", "w") as f:
        json.dump(history, f, indent=2)

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"  Steps: {args.num_steps}")
    print(f"  Pre  POPE: {pre_pope['acc']:.1%}, Gap: {pre_blind['gap']:.1%}")
    print(f"  Post POPE: {final_pope['acc']:.1%}, Gap: {final_blind['gap']:.1%}")
    print(f"  Best POPE: {best_acc:.1%}")
    print(f"  Δ POPE: {(final_pope['acc'] - pre_pope['acc'])*100:+.1f}pp")
    print(f"  Δ Gap:  {(final_blind['gap'] - pre_blind['gap'])*100:+.1f}pp")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
