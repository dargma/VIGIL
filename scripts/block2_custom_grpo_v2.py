"""
VIGIL Block 2 v2 — Custom GRPO with FULL model unfreezing (no LoRA).

Key improvements over v1:
  - Full model unfreezing (LoRA collapsed in Block 1 v1-v3)
  - Batch generation via num_return_sequences (fills GPU)
  - Better answer extraction (handles verbose responses)
  - Prompt engineering for concise answers
  - Multiple samples per optimizer step
  - Mixed data (non-binary) for reward diversity

Settings:
  A: R_correct only (lambda_iig=0)
  B: R_correct + IIG (lambda_iig=0.0615)
"""

import sys, os, gc, json, time, random, math, argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from copy import deepcopy

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    num_steps=50,
    group_size=8,
    lr=5e-7,
    beta_kl=0.02,
    beta_entropy=0.01,
    min_entropy=0.3,
    max_grad_norm=1.0,
    max_kl=0.3,
    epsilon_clip=0.2,
    max_grad_candidates=4,  # only backprop on top-K by |advantage|
    samples_per_step=2,  # multiple samples per optimizer step
    # Generation
    max_new_tokens=64,
    temperature=1.4,
    top_p=0.95,
    # IIG
    lambda_iig=0.0615,
    eps_iig=0.1,
    # Eval
    eval_every=10,
    eval_samples=200,
    blind_test_samples=100,
    # Data
    train_samples=2000,
    seed=42,
    # Paths
    output_dir="checkpoints/block2_v2",
    results_dir="lab/results/block2_v2",
)


def parse_args():
    p = argparse.ArgumentParser(description="VIGIL Block 2 v2: Custom GRPO (full unfreeze)")
    for k, v in DEFAULTS.items():
        flag = f"--{k.replace('_', '-')}"
        p.add_argument(flag, type=type(v), default=v)
    p.add_argument("--setting", choices=["A", "B"], default="B",
                   help="A = R_correct only, B = R_correct + IIG")
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_mixed_training_data(limit: int, seed: int):
    """Load VQAv2 (non-binary) + A-OKVQA (MC) + TextVQA (open-ended)."""
    from src.data_loader import (
        load_vqav2_train, load_aokvqa_train, load_textvqa_train,
        load_pope, check_image_overlap, remove_overlapping,
    )

    per_source = max(limit // 3, 200)
    vqav2 = load_vqav2_train(limit=per_source * 3)
    aokvqa = load_aokvqa_train(limit=per_source)
    textvqa = load_textvqa_train(limit=per_source)

    # Remove POPE-overlapping images
    pope = load_pope("adversarial", limit=3000)
    overlap = check_image_overlap(aokvqa, pope)
    if overlap:
        aokvqa = remove_overlapping(aokvqa, overlap)
    overlap_v2 = check_image_overlap(vqav2, pope)
    if overlap_v2:
        vqav2 = remove_overlapping(vqav2, overlap_v2)

    # Filter VQAv2: remove binary yes/no
    vqav2 = [s for s in vqav2 if s["answer"].strip().lower() not in ("yes", "no", "y", "n")]

    # Filter for valid images
    vqav2 = [s for s in vqav2 if s.get("image") is not None]
    aokvqa = [s for s in aokvqa if s.get("image") is not None]
    textvqa = [s for s in textvqa if s.get("image") is not None]

    # Balance: ~40% open-ended, ~30% MC, ~30% short-answer
    n_text = min(int(limit * 0.4), len(textvqa))
    n_mc = min(int(limit * 0.3), len(aokvqa))
    n_sa = min(limit - n_text - n_mc, len(vqav2))

    random.seed(seed)
    random.shuffle(vqav2); random.shuffle(aokvqa); random.shuffle(textvqa)

    combined = textvqa[:n_text] + aokvqa[:n_mc] + vqav2[:n_sa]
    random.shuffle(combined)

    # Format prompts for concise answers
    for s in combined:
        if s["type"] == "mc" and "choices" in s:
            letters = "ABCDEFGH"
            choices_str = "\n".join(
                f"{letters[i]}. {c}" for i, c in enumerate(s["choices"]) if i < len(letters)
            )
            s["prompt_text"] = f"{s['question']}\n{choices_str}\nAnswer with the letter only."
        elif s["type"] == "yesno":
            s["prompt_text"] = f"{s['question']}\nAnswer yes or no only."
        else:
            # Short answer / open-ended: ask for brief answer
            s["prompt_text"] = f"{s['question']}\nAnswer in a few words."

    type_dist = Counter(s["type"] for s in combined)
    source_dist = Counter(s["source"] for s in combined)
    print(f"[data] Mixed training set: {len(combined)} samples")
    print(f"  Types: {dict(type_dist)}")
    print(f"  Sources: {dict(source_dist)}")
    return combined


# ---------------------------------------------------------------------------
# Model loading — FULL UNFREEZE (no LoRA)
# ---------------------------------------------------------------------------
def load_model():
    """Load Qwen3-VL-2B-Instruct with all parameters trainable."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    print("[model] Loading Qwen3-VL-2B-Instruct (full unfreeze)...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",
        torch_dtype=torch.bfloat16,  # bf16 for training stability
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

    # All params trainable + gradient checkpointing
    model.train()
    for p in model.parameters():
        p.requires_grad = True
    model.gradient_checkpointing_enable()
    print("  Gradient checkpointing enabled")

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params/1e6:.1f}M, trainable: {trainable/1e6:.1f}M")

    device = next(model.parameters()).device

    model_info = {
        "model": model,
        "processor": processor,
        "tokenizer": processor.tokenizer,
        "model_type": "qwen3_vl",
        "device": device,
        "num_layers": 28,
        "num_heads": 16,
        "num_kv_heads": 8,
        "head_dim": 128,
        "hidden_size": 2048,
    }
    return model, processor, model_info


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def build_inputs(model_info, question: str, image):
    """Build model inputs for question + image."""
    processor = model_info["processor"]
    messages = [{"role": "user", "content": []}]
    if image is not None:
        messages[0]["content"].append({"type": "image", "image": image})
    messages[0]["content"].append({"type": "text", "text": question})
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image] if image is not None else None,
                       return_tensors="pt", padding=True)
    return {k: v.to(model_info["device"]) for k, v in inputs.items()}


# ---------------------------------------------------------------------------
# Batch generation — uses num_return_sequences to fill GPU
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_candidates_batch(
    model, model_info, sample: dict, group_size: int,
    temperature: float, top_p: float, max_new_tokens: int,
):
    """Generate group_size candidates in a single batched forward pass."""
    question = sample.get("prompt_text", sample["question"])
    image = sample.get("image")

    inputs = build_inputs(model_info, question, image)
    prompt_len = inputs["input_ids"].shape[1]

    try:
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=group_size,
        )
    except torch.cuda.OutOfMemoryError:
        gc.collect(); torch.cuda.empty_cache()
        # Fallback: half group size
        try:
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=group_size // 2,
            )
        except Exception:
            return [], [], prompt_len, inputs

    candidates = []
    candidate_ids_list = []
    for i in range(output_ids.shape[0]):
        gen_ids = output_ids[i, prompt_len:]
        text = model_info["tokenizer"].decode(gen_ids, skip_special_tokens=True)
        candidates.append(text)
        candidate_ids_list.append(gen_ids)

    return candidates, candidate_ids_list, prompt_len, inputs


# ---------------------------------------------------------------------------
# Better answer extraction
# ---------------------------------------------------------------------------
def extract_yesno(text: str) -> str:
    """Extract yes/no from potentially verbose response."""
    t = text.strip().lower()
    # Check common patterns
    if t.startswith("yes") or t.startswith("no"):
        return "yes" if t.startswith("yes") else "no"
    # Check within first 50 chars
    t50 = t[:50]
    has_yes = "yes" in t50
    has_no = "no" in t50
    if has_yes and has_no:
        return "yes" if t50.index("yes") < t50.index("no") else "no"
    if has_yes:
        return "yes"
    if has_no:
        return "no"
    return ""


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------
def compute_rewards_for_group(
    model_info, sample: dict, candidates: list,
    lambda_iig: float, eps_iig: float, use_iig: bool,
):
    """Compute rewards for all candidates."""
    gt = sample["answer"]
    qtype = sample["type"]

    rewards = []
    details = []

    # IIG computation
    iig_values = [0.0] * len(candidates)
    if use_iig and sample.get("image") is not None:
        try:
            from src.iig import compute_iig_batch_candidates
            question = sample.get("prompt_text", sample["question"])
            iig_values = compute_iig_batch_candidates(
                model_info, question, sample["image"], candidates,
            )
        except Exception as e:
            print(f"    [iig] Error: {e}")

    for i, cand in enumerate(candidates):
        if not cand.strip():
            rewards.append(0.0)
            details.append({"r_correct": 0.0, "iig": 0.0, "r_total": 0.0})
            continue

        # Better reward computation for non-binary
        from src.rewards import compute_r_correct
        r_correct = compute_r_correct(cand, gt, qtype)

        # For short_answer type, also give partial credit for containing the answer
        if qtype == "short_answer" and r_correct == 0:
            if gt.lower() in cand.lower():
                r_correct = 0.5  # partial credit for containing answer

        iig_val = iig_values[i]
        if use_iig:
            from src.iig import vigil_reward
            r_total = vigil_reward(r_correct, iig_val, lambda_iig, eps_iig)
        else:
            r_total = r_correct

        rewards.append(r_total)
        details.append({"r_correct": r_correct, "iig": iig_val, "r_total": r_total})

    return rewards, details


# ---------------------------------------------------------------------------
# Log-prob computation
# ---------------------------------------------------------------------------
def compute_logprobs(model, model_info, inputs, candidate_ids, prompt_len):
    """Compute per-token log-probs and entropy for a candidate."""
    if candidate_ids.numel() == 0:
        dev = model_info["device"]
        return torch.tensor([], device=dev), torch.tensor(0.0, device=dev)

    T = candidate_ids.shape[0]
    full_ids = torch.cat([inputs["input_ids"][0], candidate_ids]).unsqueeze(0)

    kw = {}
    for k, v in inputs.items():
        if k == "input_ids":
            continue
        if k == "attention_mask" and v is not None:
            ext = torch.ones(1, T, dtype=v.dtype, device=v.device)
            kw[k] = torch.cat([v, ext], dim=1)
        else:
            kw[k] = v

    logits = model(input_ids=full_ids, **kw).logits
    relevant = logits[0, prompt_len - 1: prompt_len - 1 + T, :]
    log_probs_all = F.log_softmax(relevant.float(), dim=-1)
    token_lp = log_probs_all.gather(-1, candidate_ids[:T].unsqueeze(-1)).squeeze(-1)
    probs = torch.exp(log_probs_all)
    entropy = -(probs * log_probs_all).sum(dim=-1).mean()
    return token_lp, entropy


# ---------------------------------------------------------------------------
# GRPO loss with KL against frozen ref logprobs
# ---------------------------------------------------------------------------
def compute_grpo_loss(
    model, model_info, inputs, candidate_ids_list, prompt_len,
    advantages, ref_logprobs_list,
    beta_kl, beta_entropy, epsilon_clip, max_kl,
    max_grad_candidates=4,
):
    """Compute clipped PPO loss with entropy bonus and KL penalty.
    Only processes top max_grad_candidates by |advantage| to save memory."""
    dev = model_info["device"]
    total_loss = torch.tensor(0.0, device=dev, requires_grad=True)
    pg_sum = ent_sum = kl_sum = 0.0
    n_valid = 0

    # Select top-K candidates by |advantage| for gradient computation
    indexed = list(enumerate(zip(candidate_ids_list, advantages, ref_logprobs_list)))
    indexed.sort(key=lambda x: abs(x[1][1]), reverse=True)
    indexed = indexed[:max_grad_candidates]

    for _, (cand_ids, adv, ref_lp) in indexed:
        if cand_ids.numel() == 0 or ref_lp.numel() == 0 or abs(adv) < 1e-8:
            continue

        T = cand_ids.shape[0]
        full_ids = torch.cat([inputs["input_ids"][0], cand_ids]).unsqueeze(0)
        kw = {}
        for k, v in inputs.items():
            if k == "input_ids":
                continue
            if k == "attention_mask" and v is not None:
                ext = torch.ones(1, T, dtype=v.dtype, device=v.device)
                kw[k] = torch.cat([v, ext], dim=1)
            else:
                kw[k] = v

        try:
            logits = model(input_ids=full_ids, **kw).logits
        except torch.cuda.OutOfMemoryError:
            gc.collect(); torch.cuda.empty_cache()
            continue

        relevant = logits[0, prompt_len - 1: prompt_len - 1 + T, :]
        log_probs_all = F.log_softmax(relevant.float(), dim=-1)
        cur_lp = log_probs_all.gather(-1, cand_ids[:T].unsqueeze(-1)).squeeze(-1)

        # Entropy
        probs = torch.exp(log_probs_all)
        entropy = -(probs * log_probs_all).sum(dim=-1).mean()

        # Align lengths
        min_len = min(cur_lp.shape[0], ref_lp.shape[0])
        cur_t = cur_lp[:min_len]
        ref_t = ref_lp[:min_len].detach()

        # KL
        kl = (torch.exp(ref_t) * (ref_t - cur_t)).mean()
        if kl.item() > max_kl:
            continue

        # Clipped surrogate
        log_ratio = cur_t - ref_t
        ratio = torch.exp(log_ratio)
        adv_t = torch.tensor(adv, device=dev, dtype=torch.float32)
        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 1 - epsilon_clip, 1 + epsilon_clip) * adv_t
        pg_loss = -torch.min(surr1, surr2).mean()

        cand_loss = pg_loss - beta_entropy * entropy + beta_kl * kl
        total_loss = total_loss + cand_loss
        pg_sum += pg_loss.item()
        ent_sum += entropy.item()
        kl_sum += kl.item()
        n_valid += 1

    if n_valid == 0:
        return None, {"pg_loss": 0, "entropy": 0, "kl": 0, "n_valid": 0}
    return total_loss / n_valid, {
        "pg_loss": pg_sum / n_valid,
        "entropy": ent_sum / n_valid,
        "kl": kl_sum / n_valid,
        "n_valid": n_valid,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def eval_pope(model, model_info, n_samples=200):
    """Evaluate on POPE adversarial."""
    from src.data_loader import load_pope
    model.eval()
    model.gradient_checkpointing_disable()
    samples = load_pope("adversarial", limit=n_samples)
    correct = total = yes_count = no_count = 0

    for s in samples:
        try:
            inputs = build_inputs(model_info, s["question"] + "\nAnswer yes or no only.", s.get("image"))
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
            pred = model_info["tokenizer"].decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            yn = extract_yesno(pred)
            if yn == "yes": yes_count += 1
            elif yn == "no": no_count += 1

            if yn == s["answer"].strip().lower():
                correct += 1
            total += 1
        except Exception:
            continue

    acc = correct / max(total, 1) * 100
    model.train()
    model.gradient_checkpointing_enable()
    return {"acc": acc, "correct": correct, "total": total,
            "yes": yes_count, "no": no_count,
            "yes_ratio": yes_count / max(total, 1),
            "no_ratio": no_count / max(total, 1)}


def eval_blind_test(model, model_info, n_samples=100):
    """Blind test: real vs black images. Measure Gap."""
    from src.data_loader import load_pope
    model.eval()
    model.gradient_checkpointing_disable()
    samples = load_pope("adversarial", limit=n_samples)
    black_img = Image.new("RGB", (448, 448), (0, 0, 0))
    correct_real = correct_blind = total = 0

    for s in samples:
        try:
            prompt = s["question"] + "\nAnswer yes or no only."
            # Real
            inputs_r = build_inputs(model_info, prompt, s.get("image"))
            with torch.no_grad():
                out_r = model.generate(**inputs_r, max_new_tokens=20, do_sample=False)
            pred_r = model_info["tokenizer"].decode(
                out_r[0][inputs_r["input_ids"].shape[1]:], skip_special_tokens=True)
            # Blind
            inputs_b = build_inputs(model_info, prompt, black_img)
            with torch.no_grad():
                out_b = model.generate(**inputs_b, max_new_tokens=20, do_sample=False)
            pred_b = model_info["tokenizer"].decode(
                out_b[0][inputs_b["input_ids"].shape[1]:], skip_special_tokens=True)

            gt = s["answer"].strip().lower()
            if extract_yesno(pred_r) == gt: correct_real += 1
            if extract_yesno(pred_b) == gt: correct_blind += 1
            total += 1
        except Exception:
            continue

    acc_r = correct_real / max(total, 1) * 100
    acc_b = correct_blind / max(total, 1) * 100
    model.train()
    model.gradient_checkpointing_enable()
    return {"acc_real": acc_r, "acc_blind": acc_b, "gap": acc_r - acc_b, "total": total}


def check_collapse(pope_result):
    total = pope_result["total"]
    if total == 0: return True
    yr = pope_result["yes"] / total
    nr = pope_result["no"] / total
    return yr > 0.90 or nr > 0.90


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    if args.setting == "A":
        args.lambda_iig = 0.0
        print("[config] Setting A: R_correct only")
    else:
        print(f"[config] Setting B: R_correct + IIG (lambda={args.lambda_iig})")
    use_iig = args.lambda_iig > 0

    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"{'='*70}")
    print(f"VIGIL Block 2 v2: Custom GRPO — Setting {args.setting} (full unfreeze)")
    print(f"{'='*70}")
    print(f"  steps={args.num_steps}, group={args.group_size}, samples/step={args.samples_per_step}")
    print(f"  lr={args.lr}, beta_kl={args.beta_kl}, beta_ent={args.beta_entropy}")
    print(f"  temp={args.temperature}, max_new_tokens={args.max_new_tokens}")
    print(f"  lambda_iig={args.lambda_iig}")
    print(f"{'='*70}\n")

    # Load data
    print("[1/4] Loading mixed training data...")
    train_data = load_mixed_training_data(args.train_samples, args.seed)

    # Load model
    print("\n[2/4] Loading model (full unfreeze, bf16)...")
    model, processor, model_info = load_model()
    device = model_info["device"]

    # Store reference model state for KL computation
    # We save ref log-probs per sample instead of keeping a full ref model copy
    print("  (KL computed against initial log-probs, no ref model copy needed)")

    # Optimizer — all parameters, low LR
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95)
    )
    print(f"  Optimizer: AdamW, lr={args.lr}")

    # GPU usage report
    mem_used = torch.cuda.memory_allocated() / 1024**3
    mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  GPU: {mem_used:.1f}/{mem_total:.1f} GB used")

    # Baseline eval
    print("\n[3/4] Baseline evaluation...")
    pope_base = eval_pope(model, model_info, n_samples=min(args.eval_samples, 200))
    blind_base = eval_blind_test(model, model_info, n_samples=min(args.blind_test_samples, 100))
    print(f"  POPE: {pope_base['acc']:.1f}% (yes={pope_base['yes']}, no={pope_base['no']}, total={pope_base['total']})")
    print(f"  Blind: real={blind_base['acc_real']:.1f}%, blind={blind_base['acc_blind']:.1f}%, Gap={blind_base['gap']:.1f}pp")

    # Training loop
    print(f"\n[4/4] Training for {args.num_steps} steps (x{args.samples_per_step} samples/step)...")
    log_history = []
    eval_history = [{"step": 0, "pope": pope_base, "blind": blind_base}]
    data_idx = 0
    total_skipped = 0
    total_groups = 0

    for step in range(1, args.num_steps + 1):
        step_t0 = time.time()
        optimizer.zero_grad()
        step_loss_sum = 0.0
        step_reward_sum = 0.0
        step_reward_std_sum = 0.0
        step_diversity_sum = 0.0
        step_entropy_sum = 0.0
        step_kl_sum = 0.0
        step_skipped = 0
        step_valid = 0

        for si in range(args.samples_per_step):
            sample = train_data[data_idx % len(train_data)]
            data_idx += 1

            # Skip no-image samples when IIG needed
            if use_iig and sample.get("image") is None:
                for _ in range(len(train_data)):
                    sample = train_data[data_idx % len(train_data)]
                    data_idx += 1
                    if sample.get("image") is not None:
                        break

            total_groups += 1

            # Generate candidates (batched, no grad checkpointing)
            model.eval()
            model.gradient_checkpointing_disable()
            candidates, cand_ids, prompt_len, inputs = generate_candidates_batch(
                model, model_info, sample, args.group_size,
                args.temperature, args.top_p, args.max_new_tokens,
            )
            model.train()
            model.gradient_checkpointing_enable()

            if len(candidates) == 0:
                step_skipped += 1
                total_skipped += 1
                continue

            # Compute rewards
            rewards, details = compute_rewards_for_group(
                model_info, sample, candidates,
                args.lambda_iig, args.eps_iig, use_iig,
            )

            rarr = np.array(rewards)
            rmean = rarr.mean()
            rstd = rarr.std()
            unique = len(set(c.strip().lower() for c in candidates if c.strip()))
            diversity = unique / max(len(candidates), 1)

            # Skip zero-variance
            if rstd < 1e-8:
                step_skipped += 1
                total_skipped += 1
                step_reward_sum += rmean
                step_diversity_sum += diversity
                continue

            # Advantages
            advantages = ((rarr - rmean) / (rstd + 1e-8)).tolist()

            # Reference log-probs (before update, under current model)
            # For KL, we use the log-probs computed before the policy gradient step
            ref_lps = []
            for cid in cand_ids:
                if cid.numel() > 0:
                    with torch.no_grad():
                        rlp, _ = compute_logprobs(model, model_info, inputs, cid, prompt_len)
                    ref_lps.append(rlp)
                else:
                    ref_lps.append(torch.tensor([], device=device))

            # GRPO loss (only top-K candidates by advantage)
            loss, stats = compute_grpo_loss(
                model, model_info, inputs, cand_ids, prompt_len,
                advantages, ref_lps,
                args.beta_kl, args.beta_entropy, args.epsilon_clip, args.max_kl,
                max_grad_candidates=args.max_grad_candidates,
            )

            if loss is not None:
                (loss / args.samples_per_step).backward()
                step_loss_sum += loss.item()
                step_entropy_sum += stats["entropy"]
                step_kl_sum += stats["kl"]
                step_valid += 1

            step_reward_sum += rmean
            step_reward_std_sum += rstd
            step_diversity_sum += diversity

            # Free memory
            del loss, inputs, cand_ids, candidates, ref_lps
            gc.collect()

        # Optimizer step
        if step_valid > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

        # Clear cache periodically
        if step % 3 == 0:
            torch.cuda.empty_cache()

        # Logging
        n_total = args.samples_per_step
        avg_loss = step_loss_sum / max(step_valid, 1)
        avg_reward = step_reward_sum / n_total
        avg_rstd = step_reward_std_sum / n_total
        avg_div = step_diversity_sum / n_total
        avg_ent = step_entropy_sum / max(step_valid, 1)
        avg_kl = step_kl_sum / max(step_valid, 1)
        skip_frac = total_skipped / max(total_groups, 1)
        elapsed = time.time() - step_t0

        step_info = {
            "step": step, "loss": avg_loss, "reward_mean": avg_reward,
            "reward_std": avg_rstd, "diversity": avg_div,
            "entropy": avg_ent, "kl": avg_kl,
            "valid": step_valid, "skipped": step_skipped, "time": elapsed,
        }
        log_history.append(step_info)

        print(
            f"  Step {step:3d} | L={avg_loss:.4f} | R={avg_reward:.3f}+/-{avg_rstd:.3f} | "
            f"div={avg_div:.2f} | H={avg_ent:.3f} | KL={avg_kl:.4f} | "
            f"valid={step_valid}/{n_total} | skip={skip_frac:.0%} | {elapsed:.1f}s"
        )

        # Periodic eval
        if step % args.eval_every == 0 or step == args.num_steps:
            print(f"\n  --- Eval at step {step} ---")
            model.eval()
            pope_res = eval_pope(model, model_info, n_samples=args.eval_samples)
            blind_res = eval_blind_test(model, model_info, n_samples=args.blind_test_samples)
            model.train()

            print(f"    POPE: {pope_res['acc']:.1f}% (yes={pope_res['yes']}, no={pope_res['no']})")
            print(f"    Blind: real={blind_res['acc_real']:.1f}%, blind={blind_res['acc_blind']:.1f}%, Gap={blind_res['gap']:.1f}pp")

            eval_history.append({"step": step, "pope": pope_res, "blind": blind_res})

            if check_collapse(pope_res):
                print(f"    ** COLLAPSE at step {step} **")

            # Save checkpoint
            ckpt = output_dir / f"step_{step:04d}"
            ckpt.mkdir(parents=True, exist_ok=True)
            torch.save({
                "step": step,
                "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(),
                "eval_history": eval_history,
            }, str(ckpt / "checkpoint.pt"))
            print(f"    Checkpoint: {ckpt}\n")

    # Final
    print(f"\n{'='*70}")
    print(f"Training complete: {args.num_steps} steps")
    print(f"{'='*70}")
    frac = total_skipped / max(total_groups, 1)
    print(f"  Groups: {total_groups}, skipped: {total_skipped} ({frac:.0%})")

    if len(eval_history) >= 2:
        p0 = eval_history[0]["pope"]["acc"]
        pf = eval_history[-1]["pope"]["acc"]
        g0 = eval_history[0]["blind"]["gap"]
        gf = eval_history[-1]["blind"]["gap"]
        print(f"  POPE: {p0:.1f}% -> {pf:.1f}% ({pf-p0:+.1f}pp)")
        print(f"  Gap:  {g0:.1f}pp -> {gf:.1f}pp ({gf-g0:+.1f}pp)")

    # Save final model
    final = output_dir / "final"
    final.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final))
    model_info["tokenizer"].save_pretrained(str(final))
    print(f"  Final model: {final}")

    # Save results
    results = {
        "setting": args.setting,
        "config": vars(args),
        "timestamp": ts,
        "log_history": log_history,
        "eval_history": _ser(eval_history),
    }
    rpath = results_dir / f"block2v2_{args.setting}_{ts}.json"
    with open(rpath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results: {rpath}")


def _ser(eh):
    out = []
    for e in eh:
        o = {"step": e["step"]}
        for k in ("pope", "blind"):
            if k in e:
                o[k] = {kk: float(vv) if isinstance(vv, (np.floating, float)) else int(vv)
                         for kk, vv in e[k].items()}
        out.append(o)
    return out


if __name__ == "__main__":
    main()
