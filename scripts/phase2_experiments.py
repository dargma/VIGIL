"""
VIGIL Phase 2 Experiments — All axes in one script.

P2-01: Dual-Head Ablation (inference-only steering comparison)
P2-02: Steered Distillation (BoN+SFT with steering active)
DAPO: Custom DAPO with soft thresholding rewards

Usage:
    python scripts/phase2_experiments.py --exp p2_01  # Dual-head ablation
    python scripts/phase2_experiments.py --exp p2_02  # Steered distillation BoN+SFT
    python scripts/phase2_experiments.py --exp dapo   # DAPO training
    python scripts/phase2_experiments.py --exp all    # Run in order
"""

import os, sys, gc, json, re, time, argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from datetime import datetime
from datasets import load_from_disk

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.steerer import ActivationSteerer, SteeringHook
from src.calibrator import CalibrationResult

# ─── VLMEvalKit standard functions ─────────────────────────────────────────

def process_punctuation(inText):
    outText = inText
    punct = [';', r'/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\',
             '_', '-', '>', '<', '@', '`', ',', '?', '!']
    commaStrip = re.compile(r'(\d)(,)(\d)')
    periodStrip = re.compile(r'(?<!\d)\.(?!\d)')
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (
                re.search(commaStrip, inText) is not None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub('', outText, re.UNICODE)
    return outText


def YOrN_Extraction(output):
    s = output.lower()
    words = process_punctuation(s).split()
    if 'yes' in words and 'no' not in words:
        return 'Yes'
    if 'yes' not in words and 'no' in words:
        return 'No'
    return 'Unknown'


POPE_PROMPT = "{question} Please answer yes or no."


def pope_metrics(records):
    import pandas as pd
    def cal_f1(yt, yp):
        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        return f1, p, r

    df = pd.DataFrame(records)
    results = {}
    for split_name, sub in [("Overall", df)] + [
        (cat, df[df["category"] == cat]) for cat in sorted(df["category"].unique())
    ]:
        yt = np.array([1 if a == "Yes" else 0 for a in sub["answer"]])
        yp = np.array([1 if a == "Yes" else 0 for a in sub["extracted"]])
        score = np.array([1 if a == e else 0 for a, e in zip(sub["answer"], sub["extracted"])])
        f1, p, r = cal_f1(yt, yp)
        results[split_name] = {
            "acc": float(np.mean(score) * 100), "f1": float(f1 * 100),
            "precision": float(p * 100), "recall": float(r * 100),
            "n": len(sub), "n_unknown": int(sum(1 for e in sub["extracted"] if e == "Unknown")),
        }
    return results


# ─── Model utilities ──────────────────────────────────────────────────────

def load_qwen3vl(model_path=None, for_training=False):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    hf_id = model_path or "Qwen/Qwen3-VL-2B-Instruct"
    print(f"[model] Loading: {hf_id}")
    dtype = torch.bfloat16
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        hf_id, torch_dtype=dtype, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    if for_training:
        model.train()
        model.gradient_checkpointing_enable()
    else:
        model.eval()
    return model, processor


def generate_one(model, processor, image, question, blind=False, max_new_tokens=64):
    from qwen_vl_utils import process_vision_info
    if blind:
        image = Image.new("RGB", image.size, (0, 0, 0))
    prompt = POPE_PROMPT.format(question=question)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, _ = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(text=[text], images=images, videos=videos,
                       return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             temperature=0.01, top_p=0.8, top_k=20)
    out = gen[0][inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(out, skip_special_tokens=True).strip()


def generate_candidates(model, processor, image, question, n=8, temp=1.2, max_tokens=64):
    """Generate N diverse candidates with sampling."""
    from qwen_vl_utils import process_vision_info
    prompt = POPE_PROMPT.format(question=question)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, _ = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(text=[text], images=images, videos=videos,
                       return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    candidates = []
    for _ in range(n):
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=max_tokens,
                                 temperature=temp, top_p=0.95, do_sample=True)
        out = gen[0][inputs["input_ids"].shape[1]:]
        candidates.append(processor.tokenizer.decode(out, skip_special_tokens=True).strip())
    return candidates


def eval_pope(model, processor, dataset, label, blind=False, max_n=None):
    n = min(len(dataset), max_n or len(dataset))
    records = []
    print(f"\n[pope] {label} ({'blind' if blind else 'real'}) — {n} samples")
    for i in range(n):
        s = dataset[i]
        try:
            raw = generate_one(model, processor, s["image"], s["question"], blind=blind)
        except Exception as e:
            print(f"  [{i}] ERR: {e}")
            raw = ""
        ext = YOrN_Extraction(raw)
        records.append({
            "index": i, "question": s["question"],
            "answer": s["answer"].strip().capitalize(),
            "prediction": raw, "extracted": ext,
            "category": s.get("category", "unknown"),
        })
        if (i + 1) % 200 == 0:
            acc = pope_metrics(records)["Overall"]["acc"]
            print(f"  [{i+1}/{n}] acc={acc:.1f}%")
    metrics = pope_metrics(records)
    print(f"  → {label}: acc={metrics['Overall']['acc']:.1f}%, F1={metrics['Overall']['f1']:.1f}%")
    return records, metrics


def setup_steering(model, calibration, head_subset=None, alpha=5.0, steer_start=4):
    """Install steering hooks, optionally for a subset of heads."""
    model_info = {
        "get_layers_fn": lambda: model.model.language_model.layers,
        "num_heads": 16, "head_dim": 128,
    }

    if head_subset is not None:
        # Create filtered calibration
        filtered_cal = CalibrationResult(
            n_correct=calibration.n_correct,
            n_incorrect=calibration.n_incorrect,
        )
        filtered_cal.top_heads = [(li, hi) for li, hi in head_subset
                                  if (li, hi) in calibration.steering_vectors]
        filtered_cal.steering_vectors = {
            k: v for k, v in calibration.steering_vectors.items()
            if k in [(li, hi) for li, hi in head_subset]
        }
        filtered_cal.head_scores = {
            k: v for k, v in calibration.head_scores.items()
            if k in [(li, hi) for li, hi in head_subset]
        }
        calibration = filtered_cal

    steerer = ActivationSteerer(model_info, calibration, steer_layers_start=steer_start)
    steerer.steer(alpha)
    print(f"[steer] Active: {len(steerer.hooks)} hooks, alpha={alpha}")
    return steerer


# ═══════════════════════════════════════════════════════════════════════════
# P2-01: Dual-Head Ablation
# ═══════════════════════════════════════════════════════════════════════════

def run_p2_01(args):
    """Test feature-only vs decision-only vs all-heads steering (inference only)."""
    print("\n" + "="*60)
    print("P2-01: DUAL-HEAD ABLATION")
    print("="*60)

    dataset = load_from_disk("data/eval/pope")
    calibration = CalibrationResult.load("checkpoints/calibration/qwen3_vl_2b")
    model, processor = load_qwen3vl()

    # Partition heads
    all_heads = calibration.top_heads  # Top 20 by Cohen's d
    # Decision heads: early/mid layers (L0-13), high Cohen's d
    decision_heads = [(li, hi) for li, hi in all_heads if li <= 13]
    # Feature heads: late layers (L14+), any Cohen's d score
    feature_heads = [(li, hi) for li, hi in all_heads if li >= 14]

    print(f"\nHead partition:")
    print(f"  All: {len(all_heads)} heads — {all_heads}")
    print(f"  Decision (L0-13): {len(decision_heads)} heads — {decision_heads}")
    print(f"  Feature (L14+): {len(feature_heads)} heads — {feature_heads}")

    n_eval = args.eval_samples
    results = {}

    conditions = [
        ("baseline", None, 0.0),
        ("all_heads_a5", None, 5.0),
        ("all_heads_a3", None, 3.0),
        ("decision_only_a5", decision_heads, 5.0),
        ("decision_only_a3", decision_heads, 3.0),
        ("feature_only_a5", feature_heads, 5.0),
        ("feature_only_a10", feature_heads, 10.0),
    ]

    for label, heads, alpha in conditions:
        steerer = None
        if alpha > 0:
            steerer = setup_steering(model, calibration,
                                     head_subset=heads, alpha=alpha)

        # Real eval
        recs, met = eval_pope(model, processor, dataset, label, max_n=n_eval)
        results[label] = met

        # Blind eval
        recs_b, met_b = eval_pope(model, processor, dataset,
                                  f"{label}_blind", blind=True, max_n=n_eval)
        results[f"{label}_blind"] = met_b

        # Gap
        gap = met["Overall"]["acc"] - met_b["Overall"]["acc"]
        results[label]["Overall"]["blind_gap"] = gap
        print(f"  GAP: {gap:.1f}pp")

        if steerer:
            steerer.cleanup()

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"P2-01 Results (n={n_eval})")
    print(f"{'='*80}")
    print(f"{'Condition':<25} {'Acc':>7} {'F1':>7} {'P':>7} {'R':>7} {'Gap':>8}")
    print("-" * 80)
    for label, heads, alpha in conditions:
        m = results[label]["Overall"]
        gap = m.get("blind_gap", 0)
        print(f"{label:<25} {m['acc']:>6.1f}% {m['f1']:>6.1f}% "
              f"{m['precision']:>6.1f}% {m['recall']:>6.1f}% {gap:>6.1f}pp")
    print("="*80)

    # Save
    out_dir = Path(args.output_dir) / "p2_01"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(out_dir / f"results_{ts}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_dir / f'results_{ts}.json'}")

    del model
    torch.cuda.empty_cache()
    return results


# ═══════════════════════════════════════════════════════════════════════════
# P2-02: Steered Distillation (BoN+SFT with steering)
# ═══════════════════════════════════════════════════════════════════════════

def run_p2_02(args):
    """Generate BoN candidates with steering active, SFT to internalize."""
    print("\n" + "="*60)
    print("P2-02: STEERED DISTILLATION (BoN+SFT)")
    print("="*60)

    from src.iig import compute_iig

    dataset = load_from_disk("data/eval/pope")
    train_data = load_from_disk("data/training/vqav2")
    calibration = CalibrationResult.load("checkpoints/calibration/qwen3_vl_2b")

    model, processor = load_qwen3vl()

    # ── Phase 1: Generate steered candidates ──
    print("\n[P2-02] Phase 1: Generating steered candidates...")
    steerer = setup_steering(model, calibration, alpha=args.steer_alpha)

    candidates_data = []
    n_train = min(len(train_data), args.train_samples)

    for i in range(n_train):
        sample = train_data[i]
        question = sample["question"]
        gt = sample["answer"]
        image = sample["image"]

        try:
            cands = generate_candidates(
                model, processor, image, question,
                n=args.n_candidates, temp=1.2, max_tokens=64,
            )
        except Exception as e:
            print(f"  [{i}] Generate error: {e}")
            continue

        # Score each candidate
        best_score = -1
        best_cand = None
        for c in cands:
            ext = YOrN_Extraction(c)
            gt_ext = gt.strip().capitalize()
            correct = 1.0 if ext == gt_ext else 0.0

            # IIG scoring
            try:
                iig_val = compute_iig(model, processor, image, question, c)
                iig_reward = min(max(iig_val * args.lambda_iig, 0), 1.0)
            except Exception:
                iig_reward = 0.0

            score = correct + iig_reward
            if score > best_score:
                best_score = score
                best_cand = c

        if best_cand and best_score > 0:
            candidates_data.append({
                "question": question,
                "answer": best_cand,
                "gt": gt,
                "score": best_score,
                "image_idx": i,
            })

        if (i + 1) % 50 == 0:
            yield_rate = len(candidates_data) / (i + 1) * 100
            print(f"  [{i+1}/{n_train}] yield={yield_rate:.0f}% ({len(candidates_data)} candidates)")

    steerer.cleanup()
    print(f"\n[P2-02] Generated {len(candidates_data)} curated candidates")

    if len(candidates_data) < 50:
        print("[P2-02] Too few candidates, aborting SFT")
        del model
        torch.cuda.empty_cache()
        return None

    # ── Phase 2: SFT on curated candidates ──
    print("\n[P2-02] Phase 2: SFT on steered candidates...")

    model.train()
    model.gradient_checkpointing_enable()
    for p in model.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.sft_lr, weight_decay=0.01)

    for epoch in range(args.sft_epochs):
        total_loss = 0
        n_batches = 0

        for ci, cdata in enumerate(candidates_data):
            try:
                sample = train_data[cdata["image_idx"]]
                image = sample["image"]
                question = cdata["question"]
                answer = cdata["answer"]

                prompt = POPE_PROMPT.format(question=question)
                messages = [
                    {"role": "user", "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ]},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": answer},
                    ]},
                ]

                from qwen_vl_utils import process_vision_info
                text = processor.apply_chat_template(messages, tokenize=False)
                images, videos, _ = process_vision_info(messages, return_video_kwargs=True)
                inputs = processor(text=[text], images=images, videos=videos,
                                   return_tensors="pt", padding=True)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                # Create labels (mask prompt tokens)
                input_ids = inputs["input_ids"]
                labels = input_ids.clone()
                # Find where the assistant answer starts
                prompt_only = processor.apply_chat_template(
                    messages[:1], tokenize=False, add_generation_prompt=True
                )
                prompt_len = len(processor.tokenizer.encode(prompt_only))
                labels[0, :prompt_len] = -100

                outputs = model(**inputs, labels=labels)
                loss = outputs.loss / args.sft_grad_accum

                loss.backward()
                total_loss += loss.item() * args.sft_grad_accum

                if (ci + 1) % args.sft_grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    n_batches += 1

            except Exception as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                continue

        avg_loss = total_loss / max(len(candidates_data), 1)
        print(f"  Epoch {epoch+1}/{args.sft_epochs}: loss={avg_loss:.4f}")

    # Save checkpoint
    out_dir = Path(args.output_dir) / "p2_02_steered_distill"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    print(f"[P2-02] Saved to {out_dir}")

    # ── Phase 3: Eval (WITHOUT steering to test internalization) ──
    model.eval()
    print("\n[P2-02] Phase 3: Eval (unsteered, testing internalization)...")

    recs, met = eval_pope(model, processor, dataset, "steered_distill",
                          max_n=args.eval_samples)
    recs_b, met_b = eval_pope(model, processor, dataset, "steered_distill_blind",
                              blind=True, max_n=args.eval_samples)

    gap = met["Overall"]["acc"] - met_b["Overall"]["acc"]
    print(f"\n[P2-02] RESULT: acc={met['Overall']['acc']:.1f}%, gap={gap:.1f}pp")

    # Also eval WITH mild steering (α=1) to test stacking
    steerer = setup_steering(model, calibration, alpha=1.0)
    recs_s, met_s = eval_pope(model, processor, dataset, "steered_distill+steer",
                              max_n=args.eval_samples)
    steerer.cleanup()
    print(f"[P2-02] +steer(α=1): acc={met_s['Overall']['acc']:.1f}%")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(out_dir / f"results_{ts}.json", "w") as f:
        json.dump({
            "unsteered": met, "blind": met_b, "stacked": met_s,
            "gap": gap, "n_candidates": len(candidates_data),
        }, f, indent=2)

    del model
    torch.cuda.empty_cache()
    return met


# ═══════════════════════════════════════════════════════════════════════════
# DAPO with Soft Thresholding
# ═══════════════════════════════════════════════════════════════════════════

def run_dapo(args):
    """Custom DAPO loop with soft thresholding rewards.

    Key DAPO modifications vs GRPO:
    1. Asymmetric clipping: eps_low=0.2, eps_high=0.28
    2. KL coef=0.02 (small model needs some KL, unlike DAPO paper's 0.0)
    3. Dynamic sampling: skip groups with zero reward variance
    4. Overlong penalty: linear penalty near max_new_tokens
    5. Token-level loss normalization
    6. Soft thresholding rewards (sigmoid-based, temperature annealing)
    """
    print("\n" + "="*60)
    print("DAPO: Soft Thresholding RL Training")
    print("="*60)

    from src.soft_rewards import SoftVIGILReward, soft_correct, soft_iig
    from src.iig import compute_iig

    # Load mixed training data (non-binary for diversity)
    train_data = load_from_disk("data/training/vqav2")
    eval_data = load_from_disk("data/eval/pope")
    calibration = CalibrationResult.load("checkpoints/calibration/qwen3_vl_2b")

    # Start from BoN+SFT best checkpoint
    bon_path = "checkpoints/block2_bon/final"
    if Path(bon_path).exists():
        print(f"[DAPO] Starting from BoN+SFT checkpoint: {bon_path}")
        model, processor = load_qwen3vl(bon_path, for_training=True)
        ref_model, _ = load_qwen3vl(bon_path)
    else:
        print("[DAPO] No BoN+SFT checkpoint, starting from base model")
        model, processor = load_qwen3vl(for_training=True)
        ref_model, _ = load_qwen3vl()
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # DAPO hyperparameters
    eps_low = 0.2
    eps_high = 0.28
    kl_coef = args.kl_coef  # 0.02 for small model safety
    group_size = args.group_size
    lr = args.dapo_lr
    total_steps = args.dapo_steps
    max_new_tokens = 64

    # Soft reward
    soft_reward = SoftVIGILReward(
        w_correct=0.35, w_visual=0.45, w_gate=0.20,
        tau_iig=0.5, tau_gate=0.1, anneal_rate=0.8,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    out_dir = Path(args.output_dir) / "dapo"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_log = []
    n_train = min(len(train_data), 2000)
    indices = list(range(n_train))
    np.random.seed(42)

    print(f"\n[DAPO] Config: steps={total_steps}, group={group_size}, "
          f"lr={lr}, kl={kl_coef}, eps=[{eps_low},{eps_high}]")

    for step in range(total_steps):
        np.random.shuffle(indices)
        idx = indices[step % n_train]
        sample = train_data[idx]
        question = sample["question"]
        gt = sample["answer"].strip()
        image = sample["image"]

        # ── Generate group of candidates ──
        model.eval()
        try:
            candidates = generate_candidates(
                model, processor, image, question,
                n=group_size, temp=1.2, max_tokens=max_new_tokens,
            )
        except Exception as e:
            print(f"  Step {step}: generate error: {e}")
            torch.cuda.empty_cache()
            continue

        # ── Compute soft rewards ──
        rewards = []
        for cand in candidates:
            # Soft correctness
            r_correct = soft_correct(cand, gt)

            # IIG-based visual grounding
            try:
                iig_val = compute_iig(model, processor, image, question, cand)
            except Exception:
                iig_val = 0.0
            r_iig = soft_iig(iig_val, tau=soft_reward.tau_iig)

            # Overlong penalty (DAPO feature)
            cand_tokens = len(processor.tokenizer.encode(cand))
            if cand_tokens > max_new_tokens * 0.8:
                overlong_penalty = min(
                    (cand_tokens - max_new_tokens * 0.8) / (max_new_tokens * 0.2), 1.0
                ) * 0.3
            else:
                overlong_penalty = 0.0

            # Composite soft reward
            r_total = (
                soft_reward.w_correct * r_correct
                + soft_reward.w_visual * r_iig
                - overlong_penalty
            )
            rewards.append(r_total)

        rewards = np.array(rewards)
        mean_r = np.mean(rewards)
        std_r = np.std(rewards)

        # DAPO dynamic sampling: skip zero-variance groups
        if std_r < 1e-6:
            if step % 10 == 0:
                print(f"  Step {step}: zero variance (all rewards={mean_r:.3f}), skipping")
            continue

        # Advantages
        advantages = (rewards - mean_r) / (std_r + 1e-8)

        # ── DAPO policy update ──
        model.train()
        total_loss = 0.0

        for ci, (cand, adv) in enumerate(zip(candidates, advantages)):
            try:
                prompt = POPE_PROMPT.format(question=question)
                messages = [
                    {"role": "user", "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ]},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": cand},
                    ]},
                ]

                from qwen_vl_utils import process_vision_info
                text = processor.apply_chat_template(messages, tokenize=False)
                images_list, videos, _ = process_vision_info(messages, return_video_kwargs=True)
                inputs = processor(text=[text], images=images_list, videos=videos,
                                   return_tensors="pt", padding=True)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                input_ids = inputs["input_ids"]
                labels = input_ids.clone()
                prompt_only = processor.apply_chat_template(
                    messages[:1], tokenize=False, add_generation_prompt=True
                )
                prompt_len = len(processor.tokenizer.encode(prompt_only))
                labels[0, :prompt_len] = -100

                # Policy log probs
                outputs = model(**inputs, labels=labels)
                log_probs = -outputs.loss  # Cross-entropy = -log_prob (per token)

                # Reference log probs (for KL)
                with torch.no_grad():
                    ref_outputs = ref_model(**inputs, labels=labels)
                    ref_log_probs = -ref_outputs.loss

                # KL penalty
                kl = log_probs - ref_log_probs

                # DAPO asymmetric clipping
                ratio = torch.exp(log_probs - ref_log_probs)
                if adv >= 0:
                    clipped_ratio = torch.clamp(ratio, max=1 + eps_high)
                else:
                    clipped_ratio = torch.clamp(ratio, min=1 - eps_low)

                # Surrogate loss
                surrogate = torch.min(ratio * adv, clipped_ratio * adv)
                loss = -(surrogate - kl_coef * kl) / group_size

                loss.backward()
                total_loss += loss.item()

            except Exception as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                continue

        # Gradient step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        step_info = {
            "step": step,
            "mean_reward": float(mean_r),
            "std_reward": float(std_r),
            "loss": total_loss,
            "lr": scheduler.get_last_lr()[0],
        }
        results_log.append(step_info)

        if (step + 1) % 5 == 0:
            print(f"  Step {step+1}/{total_steps}: "
                  f"r={mean_r:.3f}±{std_r:.3f}, loss={total_loss:.4f}")

        # Periodic eval
        if (step + 1) % args.eval_every == 0:
            model.eval()
            recs, met = eval_pope(model, processor, eval_data,
                                  f"dapo_step{step+1}", max_n=200)
            recs_b, met_b = eval_pope(model, processor, eval_data,
                                      f"dapo_step{step+1}_blind", blind=True, max_n=100)
            gap = met["Overall"]["acc"] - met_b["Overall"]["acc"]
            print(f"  [EVAL] step {step+1}: acc={met['Overall']['acc']:.1f}%, gap={gap:.1f}pp")

            step_info["eval_acc"] = met["Overall"]["acc"]
            step_info["eval_gap"] = gap

            # Save checkpoint
            ckpt_dir = out_dir / f"step_{step+1}"
            ckpt_dir.mkdir(exist_ok=True)
            model.save_pretrained(str(ckpt_dir))

            # Collapse detection
            ext_counts = {}
            for r in recs:
                ext_counts[r["extracted"]] = ext_counts.get(r["extracted"], 0) + 1
            max_frac = max(ext_counts.values()) / len(recs) if recs else 0
            if max_frac > 0.9:
                dominant = max(ext_counts, key=ext_counts.get)
                print(f"  [WARN] Collapse detected: {dominant}={max_frac:.0%}. "
                      f"Reducing lr by 0.5x")
                for pg in optimizer.param_groups:
                    pg['lr'] *= 0.5

            model.train()

    # Final eval
    model.eval()
    recs, met = eval_pope(model, processor, eval_data, "dapo_final", max_n=500)
    recs_b, met_b = eval_pope(model, processor, eval_data, "dapo_final_blind",
                              blind=True, max_n=200)
    gap = met["Overall"]["acc"] - met_b["Overall"]["acc"]
    print(f"\n[DAPO] FINAL: acc={met['Overall']['acc']:.1f}%, gap={gap:.1f}pp")

    # Save
    model.save_pretrained(str(out_dir / "final"))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(out_dir / f"results_{ts}.json", "w") as f:
        json.dump({
            "final_metrics": met, "blind_metrics": met_b, "gap": gap,
            "training_log": results_log,
            "config": {
                "steps": total_steps, "group_size": group_size,
                "lr": lr, "kl_coef": kl_coef,
                "eps_low": eps_low, "eps_high": eps_high,
            }
        }, f, indent=2)

    del model, ref_model
    torch.cuda.empty_cache()
    return met


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="VIGIL Phase 2 Experiments")
    parser.add_argument("--exp", type=str, required=True,
                        choices=["p2_01", "p2_02", "dapo", "all"])
    parser.add_argument("--output-dir", type=str, default="checkpoints/phase2")
    parser.add_argument("--eval-samples", type=int, default=500)

    # P2-01 args
    # (none beyond eval-samples)

    # P2-02 args
    parser.add_argument("--steer-alpha", type=float, default=5.0)
    parser.add_argument("--train-samples", type=int, default=1000)
    parser.add_argument("--n-candidates", type=int, default=8)
    parser.add_argument("--lambda-iig", type=float, default=0.0615)
    parser.add_argument("--sft-epochs", type=int, default=2)
    parser.add_argument("--sft-lr", type=float, default=2e-6)
    parser.add_argument("--sft-grad-accum", type=int, default=8)

    # DAPO args
    parser.add_argument("--dapo-steps", type=int, default=50)
    parser.add_argument("--dapo-lr", type=float, default=5e-7)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--kl-coef", type=float, default=0.02)
    parser.add_argument("--eval-every", type=int, default=10)

    args = parser.parse_args()

    if args.exp == "p2_01" or args.exp == "all":
        run_p2_01(args)

    if args.exp == "p2_02" or args.exp == "all":
        run_p2_02(args)

    if args.exp == "dapo" or args.exp == "all":
        run_dapo(args)


if __name__ == "__main__":
    main()
