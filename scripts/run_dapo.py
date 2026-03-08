"""
VIGIL DAPO Training — Think mode first, Short-answer mode second.

Phase 1: Think-mode DAPO on TextVQA (extended reasoning, vision drift tracking)
Phase 2: Short-answer DAPO on VQAv2/POPE (binary/short answer, precision focus)

Usage:
    python scripts/run_dapo.py --phase think   # Think mode only
    python scripts/run_dapo.py --phase short   # Short answer only
    python scripts/run_dapo.py --phase both    # Think → Short (default)
"""

import os, sys, gc, json, re, time, copy, argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from datetime import datetime
from datasets import load_from_disk

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.steerer import ActivationSteerer
from src.calibrator import CalibrationResult
from src.soft_rewards import accuracy_reward, soft_iig, SoftVIGILReward


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

def load_model(model_path=None, for_training=False, enable_thinking=False):
    """Load Qwen3-VL-2B model.

    Args:
        model_path: HF ID or local path
        for_training: enable gradient checkpointing + train mode
        enable_thinking: use thinking variant or enable_thinking in processor
    """
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    if enable_thinking and model_path is None:
        hf_id = "Qwen/Qwen3-VL-2B-Thinking"
    else:
        hf_id = model_path or "Qwen/Qwen3-VL-2B-Instruct"

    print(f"[model] Loading: {hf_id}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(hf_id)

    if for_training:
        model.train()
        model.gradient_checkpointing_enable()
    else:
        model.eval()

    tokenizer = processor.tokenizer

    # Build model_info dict compatible with src/iig.py
    model_info = {
        "model": model,
        "processor": processor,
        "tokenizer": tokenizer,
        "device": next(model.parameters()).device,
        "model_type": "qwen3_vl",
        "get_layers_fn": lambda: model.model.language_model.layers,
        "num_heads": 16,
        "head_dim": 128,
    }

    return model, processor, model_info


def generate_one(model, processor, image, question, blind=False,
                 max_new_tokens=64, enable_thinking=False, prompt_template=None):
    """Generate a single response."""
    from qwen_vl_utils import process_vision_info
    if blind:
        image = Image.new("RGB", image.size, (0, 0, 0))

    if prompt_template:
        prompt = prompt_template.format(question=question)
    else:
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

    gen_kwargs = dict(max_new_tokens=max_new_tokens, temperature=0.01,
                      top_p=0.8, top_k=20, repetition_penalty=1.0)
    if enable_thinking:
        gen_kwargs["max_new_tokens"] = max(max_new_tokens, 512)

    with torch.no_grad():
        gen = model.generate(**inputs, **gen_kwargs)
    out = gen[0][inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(out, skip_special_tokens=True).strip()


def generate_group(model, processor, image, question, n=8, temp=1.2,
                   max_tokens=64, prompt_template=None, enable_thinking=False):
    """Generate N diverse candidates with sampling."""
    from qwen_vl_utils import process_vision_info

    if prompt_template:
        prompt = prompt_template.format(question=question)
    else:
        prompt = question

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, _ = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(text=[text], images=images, videos=videos,
                       return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    max_t = max(max_tokens, 512) if enable_thinking else max_tokens

    candidates = []
    for _ in range(n):
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=max_t,
                                 temperature=temp, top_p=0.95, do_sample=True)
        out = gen[0][inputs["input_ids"].shape[1]:]
        text_out = processor.tokenizer.decode(out, skip_special_tokens=True).strip()
        candidates.append(text_out)
    return candidates


def get_log_probs(model, processor, image, question, answer, prompt_template=None):
    """Get per-sequence mean log prob for a (question, answer) pair."""
    from qwen_vl_utils import process_vision_info

    if prompt_template:
        prompt = prompt_template.format(question=question)
    else:
        prompt = question

    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": answer},
        ]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False)
    images, videos, _ = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(text=[text], images=images, videos=videos,
                       return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Create labels (mask prompt tokens)
    input_ids = inputs["input_ids"]
    labels = input_ids.clone()
    prompt_only = processor.apply_chat_template(
        messages[:1], tokenize=False, add_generation_prompt=True
    )
    prompt_len = len(processor.tokenizer.encode(prompt_only))
    labels[0, :prompt_len] = -100

    outputs = model(**inputs, labels=labels)
    # outputs.loss is mean CE over non-masked tokens = -mean log prob
    return -outputs.loss, labels, input_ids


def eval_pope(model, processor, dataset, label, blind=False, max_n=None):
    """Standard POPE evaluation."""
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


# ═══════════════════════════════════════════════════════════════════════════
# DAPO Core Training Loop
# ═══════════════════════════════════════════════════════════════════════════

def dapo_train(
    model, processor, model_info, ref_model,
    train_data, eval_data, calibration,
    args, phase_name="dapo",
    prompt_template=None, enable_thinking=False,
    get_answer_fn=None, get_question_fn=None,
):
    """Core DAPO training loop.

    Args:
        model: policy model (for training)
        processor: tokenizer/processor
        model_info: dict for IIG computation
        ref_model: reference model (frozen)
        train_data: training dataset
        eval_data: POPE eval dataset
        calibration: CalibrationResult for steering eval
        args: parsed args
        phase_name: output subdirectory name
        prompt_template: e.g. "{question} Please answer yes or no."
        enable_thinking: generate with extended thinking
        get_answer_fn: extract ground truth from sample (default: sample["answer"])
        get_question_fn: extract question from sample (default: sample["question"])
    """
    from src.iig import compute_iig

    # Defaults
    if get_answer_fn is None:
        get_answer_fn = lambda s: s["answer"].strip()
    if get_question_fn is None:
        get_question_fn = lambda s: s["question"]

    # DAPO hyperparameters
    eps_low = 0.2
    eps_high = 0.28
    kl_coef = args.kl_coef
    group_size = args.group_size
    total_steps = args.dapo_steps
    max_new_tokens = 512 if enable_thinking else 64

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.dapo_lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    out_dir = Path(args.output_dir) / phase_name
    out_dir.mkdir(parents=True, exist_ok=True)

    results_log = []
    n_train = min(len(train_data), 2000)
    indices = list(range(n_train))
    np.random.seed(42)

    best_acc = 0.0
    no_improve_count = 0

    print(f"\n[DAPO-{phase_name}] Config: steps={total_steps}, group={group_size}, "
          f"lr={args.dapo_lr}, kl={kl_coef}, eps=[{eps_low},{eps_high}], "
          f"thinking={enable_thinking}")

    for step in range(total_steps):
        np.random.shuffle(indices)
        idx = indices[step % n_train]
        sample = train_data[idx]
        question = get_question_fn(sample)
        gt = get_answer_fn(sample)
        image = sample["image"]

        # ── Generate group of candidates ──
        model.eval()
        try:
            candidates = generate_group(
                model, processor, image, question,
                n=group_size, temp=1.2, max_tokens=max_new_tokens,
                prompt_template=prompt_template, enable_thinking=enable_thinking,
            )
        except Exception as e:
            print(f"  Step {step}: generate error: {e}")
            torch.cuda.empty_cache()
            continue

        # ── Compute rewards for each candidate ──
        rewards = []
        for cand in candidates:
            # 1. Accuracy reward (supports open-ended, yes/no, MC)
            r_acc = accuracy_reward(cand, gt)

            # 2. IIG-based visual grounding (soft)
            try:
                iig_val = compute_iig(model_info, question, image, cand)
            except Exception:
                iig_val = 0.0
            r_iig = soft_iig(iig_val, tau=0.5)

            # 3. Overlong penalty (DAPO feature)
            cand_tokens = len(processor.tokenizer.encode(cand))
            if cand_tokens > max_new_tokens * 0.8:
                overlong_pen = min(
                    (cand_tokens - max_new_tokens * 0.8) / (max_new_tokens * 0.2), 1.0
                ) * 0.3
            else:
                overlong_pen = 0.0

            # 4. Composite reward
            r_total = 0.4 * r_acc + 0.4 * r_iig - overlong_pen
            rewards.append(r_total)

        rewards = np.array(rewards)
        mean_r = np.mean(rewards)
        std_r = np.std(rewards)

        # DAPO dynamic sampling: skip zero-variance groups
        if std_r < 1e-6:
            if step % 10 == 0:
                print(f"  Step {step}: zero variance (r={mean_r:.3f}), skipping")
            continue

        # Advantages (group-relative)
        advantages = (rewards - mean_r) / (std_r + 1e-8)

        # ── DAPO policy update ──
        model.train()
        total_loss = 0.0

        for ci, (cand, adv) in enumerate(zip(candidates, advantages)):
            try:
                # Get policy log probs
                log_probs, labels, input_ids = get_log_probs(
                    model, processor, image, question, cand,
                    prompt_template=prompt_template,
                )

                # Get reference log probs
                with torch.no_grad():
                    ref_log_probs, _, _ = get_log_probs(
                        ref_model, processor, image, question, cand,
                        prompt_template=prompt_template,
                    )

                # KL penalty
                kl = log_probs - ref_log_probs

                # DAPO asymmetric clipping
                ratio = torch.exp(log_probs - ref_log_probs.detach())
                adv_t = torch.tensor(adv, dtype=ratio.dtype, device=ratio.device)
                if adv >= 0:
                    clipped_ratio = torch.clamp(ratio, max=1 + eps_high)
                else:
                    clipped_ratio = torch.clamp(ratio, min=1 - eps_low)

                surrogate = torch.min(ratio * adv_t, clipped_ratio * adv_t)
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
                  f"r={mean_r:.3f}±{std_r:.3f}, loss={total_loss:.4f}, "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

        # Periodic eval
        if (step + 1) % args.eval_every == 0:
            model.eval()

            recs, met = eval_pope(model, processor, eval_data,
                                  f"{phase_name}_step{step+1}", max_n=300)
            recs_b, met_b = eval_pope(model, processor, eval_data,
                                      f"{phase_name}_step{step+1}_blind",
                                      blind=True, max_n=150)
            gap = met["Overall"]["acc"] - met_b["Overall"]["acc"]
            acc = met["Overall"]["acc"]
            print(f"  [EVAL] step {step+1}: acc={acc:.1f}%, gap={gap:.1f}pp, "
                  f"F1={met['Overall']['f1']:.1f}%")

            step_info["eval_acc"] = acc
            step_info["eval_gap"] = gap
            step_info["eval_f1"] = met["Overall"]["f1"]

            # Save checkpoint
            ckpt_dir = out_dir / f"step_{step+1}"
            ckpt_dir.mkdir(exist_ok=True)
            model.save_pretrained(str(ckpt_dir))

            # Track best
            if acc > best_acc:
                best_acc = acc
                no_improve_count = 0
                model.save_pretrained(str(out_dir / "best"))
                print(f"  [BEST] New best: {acc:.1f}%")
            else:
                no_improve_count += 1

            # Collapse detection
            ext_counts = {}
            for r in recs:
                ext_counts[r["extracted"]] = ext_counts.get(r["extracted"], 0) + 1
            max_frac = max(ext_counts.values()) / len(recs) if recs else 0
            if max_frac > 0.9:
                dominant = max(ext_counts, key=ext_counts.get)
                print(f"  [WARN] Collapse: {dominant}={max_frac:.0%}. LR/=2")
                for pg in optimizer.param_groups:
                    pg['lr'] *= 0.5

            # 3× no improvement → change axis (reduce lr aggressively)
            if no_improve_count >= 3:
                print(f"  [WARN] {no_improve_count}× no improvement. LR/=5, early stop if next fails")
                for pg in optimizer.param_groups:
                    pg['lr'] *= 0.2
                if no_improve_count >= 4:
                    print(f"  [STOP] Early stopping after {no_improve_count}× no improvement")
                    break

            model.train()

    # ── Final eval ──
    model.eval()
    recs, met = eval_pope(model, processor, eval_data,
                          f"{phase_name}_final", max_n=500)
    recs_b, met_b = eval_pope(model, processor, eval_data,
                              f"{phase_name}_final_blind", blind=True, max_n=200)
    gap = met["Overall"]["acc"] - met_b["Overall"]["acc"]
    print(f"\n[DAPO-{phase_name}] FINAL: acc={met['Overall']['acc']:.1f}%, "
          f"gap={gap:.1f}pp, F1={met['Overall']['f1']:.1f}%")

    # Save final
    model.save_pretrained(str(out_dir / "final"))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(out_dir / f"results_{ts}.json", "w") as f:
        json.dump({
            "phase": phase_name,
            "final_metrics": met, "blind_metrics": met_b, "gap": gap,
            "best_acc": best_acc,
            "training_log": results_log,
            "config": {
                "steps": total_steps, "group_size": group_size,
                "lr": args.dapo_lr, "kl_coef": kl_coef,
                "eps_low": eps_low, "eps_high": eps_high,
                "enable_thinking": enable_thinking,
            }
        }, f, indent=2)
    print(f"[DAPO-{phase_name}] Results saved to {out_dir}")

    return met, results_log


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Think-Mode DAPO
# ═══════════════════════════════════════════════════════════════════════════

def run_think_dapo(args):
    """DAPO with thinking mode on TextVQA (extended reasoning chains)."""
    print("\n" + "=" * 60)
    print("DAPO Phase 1: THINK MODE (TextVQA)")
    print("=" * 60)

    # Load TextVQA training data (open-ended, needs reasoning)
    train_data = load_from_disk("data/training/textvqa_train")
    eval_data = load_from_disk("data/eval/pope")
    calibration = CalibrationResult.load("checkpoints/calibration/qwen3_vl_2b")

    # Load thinking model
    model, processor, model_info = load_model(enable_thinking=True, for_training=True)

    # Reference model (frozen copy)
    ref_model, _, _ = load_model(enable_thinking=True, for_training=False)
    for p in ref_model.parameters():
        p.requires_grad = False

    # TextVQA answer extraction: answers is a list, take most common
    def get_answer(sample):
        answers = sample.get("answers", [])
        if isinstance(answers, list) and len(answers) > 0:
            # Most common answer (VQA convention)
            from collections import Counter
            c = Counter(answers)
            return c.most_common(1)[0][0]
        return str(answers)

    met, log = dapo_train(
        model, processor, model_info, ref_model,
        train_data, eval_data, calibration,
        args, phase_name="dapo_think",
        prompt_template="{question} Answer the question concisely.",
        enable_thinking=True,
        get_answer_fn=get_answer,
    )

    del model, ref_model
    torch.cuda.empty_cache()
    gc.collect()
    return met


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Short-Answer DAPO
# ═══════════════════════════════════════════════════════════════════════════

def run_short_dapo(args):
    """DAPO on short-answer VQA (VQAv2 + POPE-style yes/no)."""
    print("\n" + "=" * 60)
    print("DAPO Phase 2: SHORT-ANSWER MODE (VQAv2)")
    print("=" * 60)

    train_data = load_from_disk("data/training/vqav2_train")
    eval_data = load_from_disk("data/eval/pope")
    calibration = CalibrationResult.load("checkpoints/calibration/qwen3_vl_2b")

    # Start from BoN+SFT best or base
    bon_path = "checkpoints/block2_bon/final"
    if Path(bon_path).exists():
        print(f"[DAPO] Starting from BoN+SFT: {bon_path}")
        model, processor, model_info = load_model(model_path=bon_path, for_training=True)
        ref_model, _, _ = load_model(model_path=bon_path, for_training=False)
    else:
        model, processor, model_info = load_model(for_training=True)
        ref_model, _, _ = load_model(for_training=False)

    for p in ref_model.parameters():
        p.requires_grad = False

    def get_answer(sample):
        return sample.get("multiple_choice_answer", sample.get("answer", "")).strip()

    met, log = dapo_train(
        model, processor, model_info, ref_model,
        train_data, eval_data, calibration,
        args, phase_name="dapo_short",
        prompt_template=POPE_PROMPT,
        enable_thinking=False,
        get_answer_fn=get_answer,
    )

    del model, ref_model
    torch.cuda.empty_cache()
    gc.collect()
    return met


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="VIGIL DAPO Training")
    parser.add_argument("--phase", type=str, default="both",
                        choices=["think", "short", "both"])
    parser.add_argument("--output-dir", type=str, default="checkpoints/dapo")

    # DAPO hyperparameters
    parser.add_argument("--dapo-steps", type=int, default=50)
    parser.add_argument("--dapo-lr", type=float, default=5e-7)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--kl-coef", type=float, default=0.02)
    parser.add_argument("--eval-every", type=int, default=10)

    args = parser.parse_args()

    start_time = time.time()

    if args.phase in ("think", "both"):
        print("\n" + "#" * 60)
        print("# PHASE 1: Think-Mode DAPO")
        print("#" * 60)
        run_think_dapo(args)

    if args.phase in ("short", "both"):
        print("\n" + "#" * 60)
        print("# PHASE 2: Short-Answer DAPO")
        print("#" * 60)
        run_short_dapo(args)

    elapsed = time.time() - start_time
    print(f"\n[DONE] Total time: {elapsed/3600:.1f} hours")


if __name__ == "__main__":
    main()
