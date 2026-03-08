"""
VIGIL Block 2 — Custom GRPO Training Loop (no TRL).

TRL's GRPOTrainer collapsed on binary VQA in 3 attempts (v1-v3).
Root causes: binary reward + small output space => zero variance => collapse.

This script implements a manual GRPO loop with:
  - Entropy bonus to prevent mode collapse
  - Mixed data (binary + open-ended + MC) for diverse reward signal
  - IIG (Image Information Gain) reward integration
  - Zero-variance group skipping
  - Dynamic temperature adjustment
  - KL penalty via LoRA adapter toggling (no separate ref model copy)

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

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    # Training
    num_steps=50,
    group_size=8,
    lr=2e-6,
    beta_kl=0.05,
    beta_entropy=0.01,
    min_entropy=0.3,
    entropy_boost_factor=2.0,
    max_grad_norm=1.0,
    max_kl=0.2,
    epsilon_clip=0.2,
    grad_accum=4,
    # Generation
    max_completion_length=128,
    temperature=1.2,
    top_p=0.95,
    dynamic_temp_threshold=0.8,   # diversity ratio below which temp is raised
    dynamic_temp_boost=0.3,
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
    # LoRA
    lora_r=16,
    lora_alpha=32,
    # Paths
    output_dir="checkpoints/block2",
    results_dir="lab/results/block2",
)


def parse_args():
    p = argparse.ArgumentParser(description="VIGIL Block 2: Custom GRPO")
    for k, v in DEFAULTS.items():
        flag = f"--{k.replace('_', '-')}"
        p.add_argument(flag, type=type(v), default=v)
    p.add_argument("--setting", choices=["A", "B"], default="B",
                   help="A = R_correct only, B = R_correct + IIG")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint dir to resume from")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading — mixed dataset
# ---------------------------------------------------------------------------
def load_mixed_training_data(limit: int, seed: int):
    """Load VQAv2 + A-OKVQA + TextVQA-train mixed together."""
    from src.data_loader import load_vqav2_train, load_aokvqa_train, load_textvqa_train, load_pope, check_image_overlap, remove_overlapping

    # Load each source proportionally
    per_source = max(limit // 3, 100)
    vqav2 = load_vqav2_train(limit=per_source)
    aokvqa = load_aokvqa_train(limit=per_source)
    textvqa = load_textvqa_train(limit=per_source)

    # Remove POPE-overlapping images from A-OKVQA
    pope = load_pope("adversarial", limit=3000)
    overlap = check_image_overlap(aokvqa, pope)
    if overlap:
        aokvqa = remove_overlapping(aokvqa, overlap)

    combined = vqav2 + aokvqa + textvqa
    random.seed(seed)
    random.shuffle(combined)
    combined = combined[:limit]

    # Add formatted question for MC samples
    for s in combined:
        if s["type"] == "mc" and "choices" in s:
            letters = "ABCDEFGH"
            choices_str = "\n".join(
                f"{letters[i]}. {c}" for i, c in enumerate(s["choices"]) if i < len(letters)
            )
            s["formatted_question"] = (
                f"{s['question']}\n{choices_str}\nAnswer with the letter only."
            )
        elif s["type"] == "yesno":
            s["formatted_question"] = f"{s['question']}\nAnswer with yes or no."
        else:
            s["formatted_question"] = f"{s['question']}\nAnswer briefly."

    type_dist = Counter(s["type"] for s in combined)
    source_dist = Counter(s["source"] for s in combined)
    print(f"[data] Mixed training set: {len(combined)} samples")
    print(f"  Types: {dict(type_dist)}")
    print(f"  Sources: {dict(source_dist)}")
    return combined


# ---------------------------------------------------------------------------
# Model loading with LoRA
# ---------------------------------------------------------------------------
def load_model_with_lora(lora_r: int, lora_alpha: int):
    """Load Qwen3-VL-2B-Instruct with LoRA via peft."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from peft import get_peft_model, LoraConfig

    print("[model] Loading Qwen3-VL-2B-Instruct...")
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "o_proj", "k_proj"],
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    device = next(model.parameters()).device

    # Build model_info dict compatible with src.model_registry / src.iig
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
    """Build model inputs for question + image using make_chat_prompt."""
    from src.model_registry import make_chat_prompt
    return make_chat_prompt(model_info, question, image)


# ---------------------------------------------------------------------------
# Generation — produce group_size candidates per sample
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_candidates(
    model, model_info, sample: dict, group_size: int,
    temperature: float, top_p: float, max_new_tokens: int,
):
    """Generate group_size candidates for a single sample.

    Returns:
        candidates: list[str] — decoded text for each candidate
        candidate_ids: list[Tensor] — token ids (without prompt) for each candidate
        prompt_len: int — length of the prompt in tokens
        inputs: dict — the prompt inputs (for log-prob computation later)
    """
    question = sample.get("formatted_question", sample["question"])
    image = sample.get("image")

    inputs = build_inputs(model_info, question, image)
    prompt_len = inputs["input_ids"].shape[1]

    candidates = []
    candidate_ids_list = []

    for _ in range(group_size):
        try:
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )
            gen_ids = output_ids[0, prompt_len:]  # strip prompt
            text = model_info["tokenizer"].decode(gen_ids, skip_special_tokens=True)
            candidates.append(text)
            candidate_ids_list.append(gen_ids)
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            candidates.append("")
            candidate_ids_list.append(torch.tensor([], dtype=torch.long, device=model_info["device"]))
        except Exception as e:
            print(f"    [gen] Error: {e}")
            candidates.append("")
            candidate_ids_list.append(torch.tensor([], dtype=torch.long, device=model_info["device"]))

    return candidates, candidate_ids_list, prompt_len, inputs


# ---------------------------------------------------------------------------
# Log-prob computation via teacher forcing
# ---------------------------------------------------------------------------
def compute_logprobs_for_candidate(
    model, model_info, inputs: dict, candidate_ids: torch.Tensor,
    prompt_len: int,
):
    """Teacher-force candidate tokens and extract per-token log-probs.

    Returns:
        log_probs: Tensor of shape (T,) — per-token log-probs
        entropy: scalar — mean per-token entropy of the policy distribution
    """
    if candidate_ids.numel() == 0:
        return torch.tensor([], device=model_info["device"]), torch.tensor(0.0, device=model_info["device"])

    T = candidate_ids.shape[0]
    full_ids = torch.cat([inputs["input_ids"][0], candidate_ids]).unsqueeze(0)

    # Build kwargs, extending attention_mask if present
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
        with torch.no_grad():
            logits = model(input_ids=full_ids, **kw).logits
    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        return torch.tensor([], device=model_info["device"]), torch.tensor(0.0, device=model_info["device"])

    # Logits at positions [prompt_len-1 : prompt_len-1+T] predict tokens at [prompt_len : prompt_len+T]
    relevant_logits = logits[0, prompt_len - 1 : prompt_len - 1 + T, :]  # (T, V)
    log_probs_all = F.log_softmax(relevant_logits.float(), dim=-1)  # (T, V)
    token_log_probs = log_probs_all.gather(-1, candidate_ids[:T].unsqueeze(-1)).squeeze(-1)  # (T,)

    # Entropy: H(p) = -sum(p * log p) averaged over tokens
    probs = torch.exp(log_probs_all)
    per_token_entropy = -(probs * log_probs_all).sum(dim=-1)  # (T,)
    mean_entropy = per_token_entropy.mean()

    return token_log_probs, mean_entropy


# ---------------------------------------------------------------------------
# Reference log-probs via LoRA adapter toggling
# ---------------------------------------------------------------------------
def compute_ref_logprobs(model, model_info, inputs, candidate_ids, prompt_len):
    """Compute log-probs under the reference (non-LoRA) model.

    Uses disable_adapter_layers() / enable_adapter_layers() to toggle LoRA.
    """
    model.disable_adapter_layers()
    try:
        ref_lp, _ = compute_logprobs_for_candidate(model, model_info, inputs, candidate_ids, prompt_len)
    finally:
        model.enable_adapter_layers()
    return ref_lp


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------
def compute_rewards_for_group(
    model_info, sample: dict, candidates: list,
    lambda_iig: float, eps_iig: float,
    use_iig: bool,
):
    """Compute rewards for all candidates in a group.

    Returns:
        rewards: list[float]
        reward_details: list[dict] with per-candidate breakdown
    """
    from src.rewards import compute_r_correct
    gt = sample["answer"]
    qtype = sample["type"]

    rewards = []
    details = []
    iig_values = []

    # Batch IIG computation if needed
    if use_iig and sample.get("image") is not None:
        try:
            from src.iig import compute_iig_batch_candidates
            question = sample.get("formatted_question", sample["question"])
            iig_values = compute_iig_batch_candidates(
                model_info, question, sample["image"],
                [c for c in candidates],
            )
        except Exception as e:
            print(f"    [iig] Batch IIG failed: {e}")
            iig_values = [0.0] * len(candidates)
    else:
        iig_values = [0.0] * len(candidates)

    for i, cand in enumerate(candidates):
        if not cand.strip():
            rewards.append(0.0)
            details.append({"r_correct": 0.0, "iig": 0.0, "r_total": 0.0, "empty": True})
            continue

        r_correct = compute_r_correct(cand, gt, qtype)
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
# GRPO policy gradient step
# ---------------------------------------------------------------------------
def compute_grpo_loss(
    model, model_info, inputs: dict, candidate_ids_list: list,
    prompt_len: int, advantages: list,
    beta_kl: float, beta_entropy: float, epsilon_clip: float, max_kl: float,
):
    """Compute clipped PPO-style policy gradient loss with entropy bonus and KL penalty.

    Returns:
        loss: scalar tensor (or None if skipped)
        stats: dict with metrics
    """
    total_loss = torch.tensor(0.0, device=model_info["device"], requires_grad=True)
    total_pg_loss = 0.0
    total_entropy = 0.0
    total_kl = 0.0
    n_valid = 0

    for i, (cand_ids, adv) in enumerate(zip(candidate_ids_list, advantages)):
        if cand_ids.numel() == 0:
            continue
        if abs(adv) < 1e-8:
            continue

        T = cand_ids.shape[0]

        # --- Current policy log-probs (with gradient) ---
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
            gc.collect()
            torch.cuda.empty_cache()
            continue

        relevant_logits = logits[0, prompt_len - 1 : prompt_len - 1 + T, :]
        log_probs_all = F.log_softmax(relevant_logits.float(), dim=-1)
        cur_lp = log_probs_all.gather(-1, cand_ids[:T].unsqueeze(-1)).squeeze(-1)

        # Entropy
        probs = torch.exp(log_probs_all)
        entropy = -(probs * log_probs_all).sum(dim=-1).mean()

        # --- Reference log-probs (no gradient) ---
        ref_lp = compute_ref_logprobs(model, model_info, inputs, cand_ids, prompt_len)

        if ref_lp.numel() == 0:
            continue

        # Ensure matching lengths
        min_len = min(cur_lp.shape[0], ref_lp.shape[0])
        cur_lp_t = cur_lp[:min_len]
        ref_lp_t = ref_lp[:min_len].detach()

        # KL divergence: mean per-token KL
        kl = (ref_lp_t.exp() * (ref_lp_t - cur_lp_t)).mean()

        # Skip if KL is too large (diverging badly)
        if kl.item() > max_kl:
            continue

        # Log-ratio for clipping
        log_ratio = cur_lp_t - ref_lp_t
        ratio = torch.exp(log_ratio)
        adv_t = torch.tensor(adv, device=model_info["device"], dtype=torch.float32)

        # Clipped surrogate
        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 1 - epsilon_clip, 1 + epsilon_clip) * adv_t
        pg_loss = -torch.min(surr1, surr2).mean()

        # Combine: policy gradient - entropy bonus + KL penalty
        candidate_loss = pg_loss - beta_entropy * entropy + beta_kl * kl

        total_loss = total_loss + candidate_loss
        total_pg_loss += pg_loss.item()
        total_entropy += entropy.item()
        total_kl += kl.item()
        n_valid += 1

    if n_valid == 0:
        return None, {"pg_loss": 0, "entropy": 0, "kl": 0, "n_valid": 0}

    avg_loss = total_loss / n_valid
    stats = {
        "pg_loss": total_pg_loss / n_valid,
        "entropy": total_entropy / n_valid,
        "kl": total_kl / n_valid,
        "n_valid": n_valid,
    }
    return avg_loss, stats


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def eval_pope(model, model_info, n_samples=200):
    """Evaluate on POPE adversarial split."""
    from src.data_loader import load_pope

    model.eval()
    samples = load_pope("adversarial", limit=n_samples)
    correct = total = yes_count = no_count = 0

    for s in samples:
        try:
            inputs = build_inputs(model_info, s["question"], s.get("image"))
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=30, do_sample=False)
            pred = model_info["tokenizer"].decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            p = pred.strip().lower()
            if "yes" in p:
                yes_count += 1
            elif "no" in p:
                no_count += 1

            # Check correctness
            has_yes = "yes" in p[:20]
            has_no = "no" in p[:20]
            if has_yes and has_no:
                yn = "yes" if p.index("yes") < p.index("no") else "no"
            elif has_yes:
                yn = "yes"
            elif has_no:
                yn = "no"
            else:
                yn = ""
            if yn == s["answer"].strip().lower():
                correct += 1
            total += 1
        except Exception:
            continue

    acc = correct / total * 100 if total > 0 else 0.0
    yes_ratio = yes_count / max(total, 1)
    no_ratio = no_count / max(total, 1)
    model.train()
    return {
        "acc": acc,
        "correct": correct,
        "total": total,
        "yes": yes_count,
        "no": no_count,
        "yes_ratio": yes_ratio,
        "no_ratio": no_ratio,
    }


def eval_blind_test(model, model_info, n_samples=100):
    """Blind test: replace all images with black images. Measure Gap."""
    from src.data_loader import load_pope

    model.eval()
    samples = load_pope("adversarial", limit=n_samples)
    black_img = Image.new("RGB", (448, 448), (0, 0, 0))

    correct_real = correct_blind = total = 0
    for s in samples:
        try:
            # Real image
            inputs_real = build_inputs(model_info, s["question"], s.get("image"))
            with torch.no_grad():
                out_real = model.generate(**inputs_real, max_new_tokens=30, do_sample=False)
            pred_real = model_info["tokenizer"].decode(
                out_real[0][inputs_real["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip().lower()

            # Black image
            inputs_blind = build_inputs(model_info, s["question"], black_img)
            with torch.no_grad():
                out_blind = model.generate(**inputs_blind, max_new_tokens=30, do_sample=False)
            pred_blind = model_info["tokenizer"].decode(
                out_blind[0][inputs_blind["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip().lower()

            gt = s["answer"].strip().lower()

            def _check(p, g):
                has_yes = "yes" in p[:20]
                has_no = "no" in p[:20]
                if has_yes and has_no:
                    yn = "yes" if p.index("yes") < p.index("no") else "no"
                elif has_yes:
                    yn = "yes"
                elif has_no:
                    yn = "no"
                else:
                    yn = ""
                return yn == g

            if _check(pred_real, gt):
                correct_real += 1
            if _check(pred_blind, gt):
                correct_blind += 1
            total += 1
        except Exception:
            continue

    acc_real = correct_real / max(total, 1) * 100
    acc_blind = correct_blind / max(total, 1) * 100
    gap = acc_real - acc_blind
    model.train()
    return {
        "acc_real": acc_real,
        "acc_blind": acc_blind,
        "gap": gap,
        "total": total,
    }


# ---------------------------------------------------------------------------
# Collapse detection
# ---------------------------------------------------------------------------
def check_collapse(pope_result: dict) -> bool:
    """Return True if collapse is detected (>90% same answer)."""
    total = pope_result["total"]
    if total == 0:
        return True
    yes_r = pope_result["yes"] / total
    no_r = pope_result["no"] / total
    collapsed = yes_r > 0.90 or no_r > 0.90
    if collapsed:
        print(f"  [COLLAPSE WARNING] yes={yes_r:.1%}, no={no_r:.1%}")
    return collapsed


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Override lambda_iig based on setting
    if args.setting == "A":
        args.lambda_iig = 0.0
        print("[config] Setting A: R_correct only (lambda_iig=0)")
    else:
        print(f"[config] Setting B: R_correct + IIG (lambda_iig={args.lambda_iig})")

    use_iig = args.lambda_iig > 0

    # Paths
    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"{'='*70}")
    print(f"VIGIL Block 2: Custom GRPO — Setting {args.setting}")
    print(f"{'='*70}")
    print(f"  num_steps={args.num_steps}, group_size={args.group_size}, lr={args.lr}")
    print(f"  beta_kl={args.beta_kl}, beta_entropy={args.beta_entropy}, temp={args.temperature}")
    print(f"  lambda_iig={args.lambda_iig}, eps_iig={args.eps_iig}")
    print(f"  train_samples={args.train_samples}, eval_every={args.eval_every}")
    print(f"  timestamp={ts}")
    print(f"{'='*70}\n")

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    print("[1/4] Loading mixed training data...")
    train_data = load_mixed_training_data(args.train_samples, args.seed)

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    print("\n[2/4] Loading model with LoRA...")
    model, processor, model_info = load_model_with_lora(args.lora_r, args.lora_alpha)
    device = model_info["device"]

    # Optimizer — only LoRA parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    print(f"  Optimizer: AdamW, {len(trainable_params)} parameter groups, lr={args.lr}")

    # -----------------------------------------------------------------------
    # Baseline eval
    # -----------------------------------------------------------------------
    print("\n[3/4] Baseline evaluation...")
    model.eval()
    pope_baseline = eval_pope(model, model_info, n_samples=min(args.eval_samples, 200))
    blind_baseline = eval_blind_test(model, model_info, n_samples=min(args.blind_test_samples, 100))
    print(f"  POPE baseline: {pope_baseline['acc']:.1f}% (yes={pope_baseline['yes']}, no={pope_baseline['no']}, total={pope_baseline['total']})")
    print(f"  Blind test baseline: real={blind_baseline['acc_real']:.1f}%, blind={blind_baseline['acc_blind']:.1f}%, Gap={blind_baseline['gap']:.1f}pp")
    model.train()

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    print(f"\n[4/4] Training for {args.num_steps} steps...")
    log_history = []
    eval_history = [{"step": 0, "pope": pope_baseline, "blind": blind_baseline}]

    current_beta_entropy = args.beta_entropy
    current_temperature = args.temperature
    data_idx = 0
    accum_loss = 0.0
    accum_steps = 0
    total_groups_skipped = 0
    total_groups = 0

    for step in range(1, args.num_steps + 1):
        step_t0 = time.time()

        # Cycle through training data
        sample = train_data[data_idx % len(train_data)]
        data_idx += 1

        # Skip samples without images if IIG is needed
        if use_iig and sample.get("image") is None:
            # Find next sample with image
            for _ in range(len(train_data)):
                sample = train_data[data_idx % len(train_data)]
                data_idx += 1
                if sample.get("image") is not None:
                    break

        # --- Generate candidates ---
        model.eval()
        candidates, candidate_ids_list, prompt_len, inputs = generate_candidates(
            model, model_info, sample, args.group_size,
            current_temperature, args.top_p, args.max_completion_length,
        )
        model.train()

        # --- Compute rewards ---
        rewards, reward_details = compute_rewards_for_group(
            model_info, sample, candidates,
            args.lambda_iig, args.eps_iig, use_iig,
        )

        total_groups += 1
        reward_arr = np.array(rewards)
        reward_mean = reward_arr.mean()
        reward_std = reward_arr.std()

        # --- Check group diversity ---
        unique_candidates = len(set(c.strip().lower() for c in candidates if c.strip()))
        diversity_ratio = unique_candidates / max(args.group_size, 1)

        # Dynamic temperature: if diversity too low, boost
        if diversity_ratio < args.dynamic_temp_threshold and step > 1:
            current_temperature = min(args.temperature + args.dynamic_temp_boost, 2.0)
        else:
            current_temperature = args.temperature

        # --- Skip zero-variance groups ---
        if reward_std < 1e-8:
            total_groups_skipped += 1
            step_stats = {
                "step": step, "skipped": True,
                "reward_mean": float(reward_mean), "reward_std": float(reward_std),
                "diversity": diversity_ratio, "type": sample["type"], "source": sample["source"],
            }
            log_history.append(step_stats)
            if step % 5 == 0:
                print(f"  Step {step:3d}: SKIPPED (zero variance) | "
                      f"R={reward_mean:.3f} | div={diversity_ratio:.2f} | type={sample['type']}")
            continue

        # --- Compute advantages ---
        advantages = ((reward_arr - reward_mean) / (reward_std + 1e-8)).tolist()

        # --- Policy gradient step ---
        loss, loss_stats = compute_grpo_loss(
            model, model_info, inputs, candidate_ids_list,
            prompt_len, advantages,
            args.beta_kl, current_beta_entropy, args.epsilon_clip, args.max_kl,
        )

        if loss is not None:
            # Gradient accumulation
            scaled_loss = loss / args.grad_accum
            scaled_loss.backward()
            accum_steps += 1

            if accum_steps >= args.grad_accum:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                accum_steps = 0

            accum_loss += loss.item()

            # Dynamic entropy: if entropy drops below threshold, boost coefficient
            if loss_stats["entropy"] < args.min_entropy and loss_stats["entropy"] > 0:
                current_beta_entropy = min(
                    args.beta_entropy * args.entropy_boost_factor,
                    0.1,  # cap at 0.1
                )
            else:
                current_beta_entropy = args.beta_entropy

        # --- Logging ---
        step_stats = {
            "step": step,
            "skipped": False,
            "loss": loss.item() if loss is not None else None,
            "pg_loss": loss_stats["pg_loss"],
            "entropy": loss_stats["entropy"],
            "kl": loss_stats["kl"],
            "n_valid": loss_stats["n_valid"],
            "reward_mean": float(reward_mean),
            "reward_std": float(reward_std),
            "diversity": diversity_ratio,
            "type": sample["type"],
            "source": sample["source"],
            "temperature": current_temperature,
            "beta_entropy": current_beta_entropy,
            "time": time.time() - step_t0,
        }
        log_history.append(step_stats)

        frac_skipped = total_groups_skipped / max(total_groups, 1)
        print(
            f"  Step {step:3d} | "
            f"L={loss.item():.4f} " if loss is not None else f"  Step {step:3d} | L=None  ",
            end="",
        )
        print(
            f"| pg={loss_stats['pg_loss']:.4f} | H={loss_stats['entropy']:.3f} | "
            f"KL={loss_stats['kl']:.4f} | R={reward_mean:.3f}+/-{reward_std:.3f} | "
            f"div={diversity_ratio:.2f} | skip={frac_skipped:.1%} | "
            f"type={sample['type']} | {time.time()-step_t0:.1f}s"
        )

        # --- Free memory ---
        del loss, inputs, candidate_ids_list, candidates
        gc.collect()
        if step % 5 == 0:
            torch.cuda.empty_cache()

        # --- Periodic eval ---
        if step % args.eval_every == 0 or step == args.num_steps:
            print(f"\n  --- Eval at step {step} ---")
            model.eval()
            pope_res = eval_pope(model, model_info, n_samples=args.eval_samples)
            blind_res = eval_blind_test(model, model_info, n_samples=args.blind_test_samples)
            model.train()

            print(f"    POPE: {pope_res['acc']:.1f}% (yes={pope_res['yes']}, no={pope_res['no']})")
            print(f"    Blind: real={blind_res['acc_real']:.1f}%, blind={blind_res['acc_blind']:.1f}%, Gap={blind_res['gap']:.1f}pp")

            eval_entry = {"step": step, "pope": pope_res, "blind": blind_res}
            eval_history.append(eval_entry)

            # Collapse check
            if check_collapse(pope_res):
                print(f"    [COLLAPSE DETECTED at step {step}]")
                print(f"    Saving emergency checkpoint and continuing with boosted entropy...")
                current_beta_entropy = min(current_beta_entropy * 3.0, 0.5)

            # Save checkpoint
            ckpt_path = output_dir / f"step_{step:04d}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt_path))
            print(f"    Checkpoint saved: {ckpt_path}")
            print()

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    # Flush remaining gradients
    if accum_steps > 0:
        torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

    print(f"\n{'='*70}")
    print(f"Training complete: {args.num_steps} steps")
    print(f"{'='*70}")

    frac_skipped = total_groups_skipped / max(total_groups, 1)
    print(f"  Groups total: {total_groups}, skipped: {total_groups_skipped} ({frac_skipped:.1%})")

    if len(eval_history) >= 2:
        first_pope = eval_history[0]["pope"]["acc"]
        last_pope = eval_history[-1]["pope"]["acc"]
        first_gap = eval_history[0]["blind"]["gap"]
        last_gap = eval_history[-1]["blind"]["gap"]
        print(f"  POPE: {first_pope:.1f}% -> {last_pope:.1f}% ({last_pope-first_pope:+.1f}pp)")
        print(f"  Gap:  {first_gap:.1f}pp -> {last_gap:.1f}pp ({last_gap-first_gap:+.1f}pp)")

        last = eval_history[-1]["pope"]
        if check_collapse(last):
            print(f"  FINAL: COLLAPSE (yes={last['yes']}, no={last['no']})")
        else:
            print(f"  FINAL: STABLE (yes_ratio={last['yes_ratio']:.1%})")

    # Save final checkpoint
    final_path = output_dir / "final"
    final_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_path))
    print(f"  Final checkpoint: {final_path}")

    # Save results
    results = {
        "setting": args.setting,
        "config": vars(args),
        "timestamp": ts,
        "log_history": log_history,
        "eval_history": _serialize_eval_history(eval_history),
        "total_groups": total_groups,
        "total_skipped": total_groups_skipped,
        "frac_skipped": frac_skipped,
    }
    results_path = results_dir / f"block2_{args.setting}_{ts}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results: {results_path}")


def _serialize_eval_history(eval_history):
    """Ensure eval history is JSON-serializable."""
    out = []
    for entry in eval_history:
        e = {"step": entry["step"]}
        for key in ("pope", "blind"):
            if key in entry:
                e[key] = {k: float(v) if isinstance(v, (np.floating, float)) else int(v)
                          for k, v in entry[key].items()}
        out.append(e)
    return out


if __name__ == "__main__":
    main()
