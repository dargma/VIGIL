"""
VIGIL Block 2 — DPO training with IIG-based preference pairs.

Fallback strategy if GRPO collapses on binary VQA. DPO directly optimizes
chosen > rejected without group-relative advantage, so it is immune to the
zero-variance problem that kills GRPO on binary tasks.

Phase 1: Generate preference dataset
  - For each sample, generate K=8 candidates (temp=1.2)
  - Score with vigil_reward = R_correct + lambda * max(IIG,0) * (R_correct + eps)
  - Create preference pairs: chosen=highest reward, rejected=lowest reward
  - Skip ties (chosen_reward <= rejected_reward)

Phase 2: DPO training
  - TRL DPOTrainer with LoRA
  - beta=0.1, lr=2e-6, 50 steps
  - Eval every 10 steps: POPE-Adv 200 + Blind Test Gap

Usage:
    python scripts/block2_dpo_iig.py --phase 1 --limit 500
    python scripts/block2_dpo_iig.py --phase 2 --max-steps 50
    python scripts/block2_dpo_iig.py --phase 1 --phase 2 --limit 500  # both
"""

import sys
import os
import gc
import json
import time
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PREF_DIR = Path("data/training/dpo_iig_preferences")
RESULTS_DIR = Path("lab/results/block2_dpo")


def parse_args():
    p = argparse.ArgumentParser(description="VIGIL Block 2: DPO with IIG preferences")
    p.add_argument("--phase", type=int, nargs="+", default=[1],
                   help="Which phases to run: 1=generate prefs, 2=train DPO")
    # Phase 1 args
    p.add_argument("--limit", type=int, default=500,
                   help="Number of training samples for preference generation")
    p.add_argument("--K", type=int, default=8,
                   help="Number of candidates per sample")
    p.add_argument("--temperature", type=float, default=1.2)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--lam", type=float, default=0.0615,
                   help="IIG lambda (from Block 0 calibration)")
    p.add_argument("--eps", type=float, default=0.1,
                   help="Epsilon for vigil_reward reversal protection")
    p.add_argument("--data-source", type=str, default="mixed",
                   choices=["mixed", "textvqa", "pope"],
                   help="Data source for preference generation")
    # Phase 2 args
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=2e-6)
    p.add_argument("--beta", type=float, default=0.1,
                   help="DPO beta (KL regularization strength)")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--pope-eval-n", type=int, default=200)
    p.add_argument("--blind-eval-n", type=int, default=100)
    # Common
    p.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pref-dir", type=str, default=str(PREF_DIR))
    p.add_argument("--results-dir", type=str, default=str(RESULTS_DIR))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_model_and_processor(model_id: str):
    """Load Qwen3-VL model, processor, and build model_info dict."""
    import torch
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    print(f"[model] Loading {model_id}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    device = next(model.parameters()).device

    model_info = {
        "model": model,
        "processor": processor,
        "tokenizer": processor.tokenizer,
        "model_type": "qwen3_vl",
        "device": device,
        "num_heads": model.config.num_attention_heads
                     if hasattr(model.config, "num_attention_heads")
                     else 16,
        "head_dim": 128,
    }
    print(f"[model] Loaded on {device}, dtype={model.dtype}")
    return model, processor, model_info


def check_yesno(pred: str, gt: str) -> bool:
    p = pred.strip().lower()
    g = gt.strip().lower()
    has_yes = "yes" in p
    has_no = "no" in p
    if has_yes and has_no:
        yn = "yes" if p.index("yes") < p.index("no") else "no"
    elif has_yes:
        yn = "yes"
    elif has_no:
        yn = "no"
    else:
        yn = ""
    return yn == g


def eval_pope(model, processor, device, n_samples: int = 200) -> dict:
    """Quick POPE-Adversarial evaluation."""
    import torch
    from src.data_loader import load_pope
    from src.model_registry import make_chat_prompt

    model_info = {
        "model": model, "processor": processor,
        "tokenizer": processor.tokenizer, "model_type": "qwen3_vl",
        "device": device,
    }
    samples = load_pope("adversarial", limit=n_samples)
    correct = total = yes_count = no_count = 0
    for s in samples:
        if s.get("image") is None:
            continue
        try:
            inputs = make_chat_prompt(model_info, s["question"], s["image"])
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=30, do_sample=False)
            pred = processor.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            p = pred.strip().lower()
            if "yes" in p:
                yes_count += 1
            elif "no" in p:
                no_count += 1
            if check_yesno(pred, s["answer"]):
                correct += 1
            total += 1
        except Exception:
            continue
    acc = correct / total * 100 if total > 0 else 0.0
    return {"acc": acc, "correct": correct, "total": total,
            "yes": yes_count, "no": no_count}


def eval_blind_gap(model, processor, device, n_samples: int = 100) -> dict:
    """Blind test: real vs black image, return gap."""
    import torch
    from PIL import Image
    from src.data_loader import load_pope
    from src.model_registry import make_chat_prompt

    model_info = {
        "model": model, "processor": processor,
        "tokenizer": processor.tokenizer, "model_type": "qwen3_vl",
        "device": device,
    }
    black_img = Image.new("RGB", (448, 448), (0, 0, 0))
    samples = load_pope("adversarial", limit=n_samples)

    real_correct = blind_correct = total = 0
    for s in samples:
        if s.get("image") is None:
            continue
        try:
            # Real image
            inp_real = make_chat_prompt(model_info, s["question"], s["image"])
            with torch.no_grad():
                out_real = model.generate(**inp_real, max_new_tokens=30, do_sample=False)
            pred_real = processor.tokenizer.decode(
                out_real[0][inp_real["input_ids"].shape[1]:], skip_special_tokens=True)

            # Black image
            inp_black = make_chat_prompt(model_info, s["question"], black_img)
            with torch.no_grad():
                out_black = model.generate(**inp_black, max_new_tokens=30, do_sample=False)
            pred_black = processor.tokenizer.decode(
                out_black[0][inp_black["input_ids"].shape[1]:], skip_special_tokens=True)

            if check_yesno(pred_real, s["answer"]):
                real_correct += 1
            if check_yesno(pred_black, s["answer"]):
                blind_correct += 1
            total += 1
        except Exception:
            continue

    real_acc = real_correct / total * 100 if total > 0 else 0.0
    blind_acc = blind_correct / total * 100 if total > 0 else 0.0
    gap = real_acc - blind_acc
    return {"real_acc": real_acc, "blind_acc": blind_acc, "gap": gap, "total": total}


# ---------------------------------------------------------------------------
# Phase 1: Generate preference dataset
# ---------------------------------------------------------------------------

def load_training_data(source: str, limit: int, seed: int) -> List[Dict]:
    """Load training samples based on source selection."""
    random.seed(seed)

    if source == "mixed":
        # Try to load pre-built mixed dataset
        mixed_path = Path("data/training/mixed_grpo")
        if mixed_path.exists():
            from datasets import load_from_disk
            ds = load_from_disk(str(mixed_path))
            samples = []
            for row in ds:
                try:
                    prompt_data = json.loads(row["prompt"])
                    question = prompt_data[0]["content"] if prompt_data else ""
                except (json.JSONDecodeError, KeyError, IndexError):
                    question = row.get("prompt", "")
                samples.append({
                    "question": question,
                    "answer": row["answer"],
                    "image": row.get("image"),
                    "type": row.get("type", "short_answer"),
                    "source": row.get("source", "mixed"),
                })
            random.shuffle(samples)
            samples = [s for s in samples if s.get("image") is not None]
            samples = samples[:limit]
            print(f"[data] Loaded mixed dataset: {len(samples)} samples")
            return samples

        # Fallback: build on the fly from individual sources
        print("[data] Mixed dataset not found, building from sources...")
        from src.data_loader import (
            load_vqav2_train, load_aokvqa_train, load_textvqa_train,
            load_pope, check_image_overlap, remove_overlapping,
        )
        n_textvqa = int(limit * 0.4)
        n_aokvqa = int(limit * 0.3)
        n_vqav2 = limit - n_textvqa - n_aokvqa

        textvqa = load_textvqa_train(limit=n_textvqa * 3)
        aokvqa = load_aokvqa_train(limit=n_aokvqa * 3)
        vqav2 = load_vqav2_train(limit=n_vqav2 * 3)

        # Remove POPE overlap
        pope = load_pope("all", limit=None)
        overlap = check_image_overlap(aokvqa, pope)
        if overlap:
            aokvqa = remove_overlapping(aokvqa, overlap)
        overlap_v2 = check_image_overlap(vqav2, pope)
        if overlap_v2:
            vqav2 = remove_overlapping(vqav2, overlap_v2)

        # Filter binary from VQAv2
        vqav2 = [s for s in vqav2 if s["answer"] not in ("yes", "no")]

        # Filter missing images
        textvqa = [s for s in textvqa if s.get("image") is not None]
        aokvqa = [s for s in aokvqa if s.get("image") is not None]
        vqav2 = [s for s in vqav2 if s.get("image") is not None]

        random.shuffle(textvqa)
        random.shuffle(aokvqa)
        random.shuffle(vqav2)

        combined = (textvqa[:n_textvqa] + aokvqa[:n_aokvqa] + vqav2[:n_vqav2])
        random.shuffle(combined)
        print(f"[data] Built mixed: {len(combined)} samples")
        return combined[:limit]

    elif source == "textvqa":
        from src.data_loader import load_textvqa_train
        samples = load_textvqa_train(limit=limit * 2)
        samples = [s for s in samples if s.get("image") is not None]
        random.shuffle(samples)
        return samples[:limit]

    elif source == "pope":
        from src.data_loader import load_pope
        samples = load_pope("adversarial", limit=limit)
        samples = [s for s in samples if s.get("image") is not None]
        return samples

    else:
        raise ValueError(f"Unknown data source: {source}")


def generate_candidates(
    model, processor, model_info: dict,
    question: str, image,
    K: int, temperature: float, max_new_tokens: int,
) -> List[str]:
    """Generate K candidate responses for a single sample."""
    import torch
    from src.model_registry import make_chat_prompt

    candidates = []
    inputs = make_chat_prompt(model_info, question, image)

    for _ in range(K):
        try:
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                )
            text = processor.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()
            candidates.append(text if text else "")
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            import torch
            torch.cuda.empty_cache()
            candidates.append("")
        except Exception:
            candidates.append("")

    return candidates


def run_phase1(args):
    """Generate IIG-based preference pairs."""
    import torch
    from src.iig import compute_iig, vigil_reward
    from src.rewards import compute_r_correct

    pref_dir = Path(args.pref_dir)
    pref_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"VIGIL Block 2 — Phase 1: Generate IIG Preference Pairs")
    print(f"  samples={args.limit}, K={args.K}, temp={args.temperature}")
    print(f"  lambda={args.lam}, eps={args.eps}")
    print(f"  data_source={args.data_source}")
    print(f"{'='*60}\n")

    # Load model
    model, processor, model_info = load_model_and_processor(args.model_id)

    # Load data
    samples = load_training_data(args.data_source, args.limit, args.seed)
    print(f"\n[phase1] {len(samples)} samples loaded")

    # Generate preferences
    preference_pairs = []
    stats = {"total": 0, "pairs_created": 0, "ties_skipped": 0,
             "iig_errors": 0, "empty_gen": 0, "no_image": 0}

    t0 = time.time()
    for idx, sample in enumerate(samples):
        stats["total"] += 1

        if sample.get("image") is None:
            stats["no_image"] += 1
            continue

        question = sample["question"]
        answer = sample["answer"]
        q_type = sample.get("type", "short_answer")
        image = sample["image"]

        # Generate K candidates
        candidates = generate_candidates(
            model, processor, model_info,
            question, image,
            args.K, args.temperature, args.max_new_tokens,
        )

        # Filter empty
        valid_candidates = [(i, c) for i, c in enumerate(candidates) if c.strip()]
        if len(valid_candidates) < 2:
            stats["empty_gen"] += 1
            continue

        # Score each candidate
        scored = []
        for ci, cand in valid_candidates:
            r_correct = compute_r_correct(cand, answer, q_type)

            # Compute IIG (may fail on edge cases)
            try:
                iig_val = compute_iig(model_info, question, image, cand)
            except Exception:
                iig_val = 0.0
                stats["iig_errors"] += 1

            reward = vigil_reward(r_correct, iig_val, args.lam, args.eps)
            scored.append({
                "text": cand,
                "r_correct": r_correct,
                "iig": iig_val,
                "reward": reward,
            })

        # Sort by reward
        scored.sort(key=lambda x: x["reward"], reverse=True)
        chosen = scored[0]
        rejected = scored[-1]

        # Skip ties
        if chosen["reward"] <= rejected["reward"]:
            stats["ties_skipped"] += 1
            continue

        pair = {
            "prompt": question,
            "chosen": chosen["text"],
            "rejected": rejected["text"],
            "chosen_reward": chosen["reward"],
            "rejected_reward": rejected["reward"],
            "chosen_r_correct": chosen["r_correct"],
            "rejected_r_correct": rejected["r_correct"],
            "chosen_iig": chosen["iig"],
            "rejected_iig": rejected["iig"],
            "answer": answer,
            "type": q_type,
            "source": sample.get("source", "unknown"),
        }
        preference_pairs.append(pair)
        stats["pairs_created"] += 1

        # Progress
        if (idx + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (len(samples) - idx - 1) / rate if rate > 0 else 0
            print(f"  [{idx+1}/{len(samples)}] pairs={stats['pairs_created']}, "
                  f"ties={stats['ties_skipped']}, "
                  f"rate={rate:.1f} s/sample, ETA={eta/60:.1f}min")

        # Periodic GPU cleanup
        if (idx + 1) % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\n[phase1] Generation complete in {elapsed/60:.1f} min")
    print(f"  Total samples processed: {stats['total']}")
    print(f"  Preference pairs created: {stats['pairs_created']}")
    print(f"  Ties skipped: {stats['ties_skipped']}")
    print(f"  IIG errors: {stats['iig_errors']}")
    print(f"  Empty generations: {stats['empty_gen']}")
    print(f"  No image: {stats['no_image']}")

    if not preference_pairs:
        print("\n  ERROR: No preference pairs generated. Check data and model.")
        return

    # Save as HuggingFace Dataset
    from datasets import Dataset

    # DPO format requires 'prompt', 'chosen', 'rejected'
    # We store images separately since DPO trainer handles text only
    ds_records = []
    for pair in preference_pairs:
        ds_records.append({
            "prompt": [{"role": "user", "content": pair["prompt"]}],
            "chosen": [{"role": "assistant", "content": pair["chosen"]}],
            "rejected": [{"role": "assistant", "content": pair["rejected"]}],
            "chosen_reward": pair["chosen_reward"],
            "rejected_reward": pair["rejected_reward"],
            "chosen_iig": pair["chosen_iig"],
            "rejected_iig": pair["rejected_iig"],
            "answer": pair["answer"],
            "type": pair["type"],
            "source": pair["source"],
        })

    ds = Dataset.from_list(ds_records)
    ds.save_to_disk(str(pref_dir))
    print(f"\n  Saved {len(ds)} preference pairs to {pref_dir}")

    # Save detailed stats
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    reward_gaps = [p["chosen_reward"] - p["rejected_reward"] for p in preference_pairs]
    iig_chosen = [p["chosen_iig"] for p in preference_pairs]
    iig_rejected = [p["rejected_iig"] for p in preference_pairs]

    summary = {
        "phase": 1,
        "timestamp": datetime.now().isoformat(),
        "args": {k: v for k, v in vars(args).items()},
        "stats": stats,
        "reward_gap": {
            "mean": float(np.mean(reward_gaps)),
            "std": float(np.std(reward_gaps)),
            "min": float(np.min(reward_gaps)),
            "max": float(np.max(reward_gaps)),
            "median": float(np.median(reward_gaps)),
        },
        "iig_chosen_mean": float(np.mean(iig_chosen)),
        "iig_rejected_mean": float(np.mean(iig_rejected)),
        "elapsed_minutes": elapsed / 60,
    }
    summary_path = results_dir / "phase1_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary to {summary_path}")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Phase 2: DPO training
# ---------------------------------------------------------------------------

def run_phase2(args):
    """DPO training on IIG preference pairs."""
    import torch
    from datasets import load_from_disk
    from trl import DPOTrainer, DPOConfig
    from peft import LoraConfig
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    pref_dir = Path(args.pref_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("checkpoints/block2_dpo_iig")

    print(f"\n{'='*60}")
    print(f"VIGIL Block 2 — Phase 2: DPO Training")
    print(f"  lr={args.lr}, beta={args.beta}, steps={args.max_steps}")
    print(f"  LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  eval_every={args.eval_every}")
    print(f"{'='*60}\n")

    # Load preference dataset
    if not pref_dir.exists():
        print(f"ERROR: Preference dataset not found at {pref_dir}")
        print(f"Run --phase 1 first to generate preferences.")
        return
    pref_ds = load_from_disk(str(pref_dir))
    print(f"[data] Loaded {len(pref_ds)} preference pairs from {pref_dir}")

    if len(pref_ds) == 0:
        print("ERROR: Empty preference dataset. Aborting.")
        return

    # Load model
    print("[model] Loading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.float16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_id)
    device = next(model.parameters()).device

    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "o_proj", "k_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )

    # Baseline eval
    print("\n--- Baseline evaluation ---")
    pope_baseline = eval_pope(model, processor, device, args.pope_eval_n)
    blind_baseline = eval_blind_gap(model, processor, device, args.blind_eval_n)
    print(f"  POPE-Adv: {pope_baseline['acc']:.1f}% "
          f"(yes={pope_baseline['yes']}, no={pope_baseline['no']})")
    print(f"  Blind Gap: {blind_baseline['gap']:.1f}pp "
          f"(real={blind_baseline['real_acc']:.1f}%, "
          f"blind={blind_baseline['blind_acc']:.1f}%)")

    eval_log = [{
        "step": 0,
        "pope": pope_baseline,
        "blind": blind_baseline,
    }]

    # DPO config
    dpo_config = DPOConfig(
        output_dir=str(ckpt_dir),
        learning_rate=args.lr,
        beta=args.beta,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=args.max_steps,
        warmup_ratio=0.1,
        logging_steps=1,
        save_steps=args.eval_every,
        max_length=1024,
        max_prompt_length=512,
        bf16=False,
        fp16=True,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
    )

    # Create DPO trainer
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=pref_ds,
        processing_class=processor.tokenizer,
        peft_config=lora_config,
    )

    # Custom training loop with periodic eval
    # We use trainer.train() but hook into save_steps for eval
    print(f"\n--- Starting DPO training ({args.max_steps} steps) ---")
    t0 = time.time()

    # Train in chunks to allow intermediate eval
    steps_done = 0
    while steps_done < args.max_steps:
        chunk_steps = min(args.eval_every, args.max_steps - steps_done)

        # Adjust max_steps for this chunk
        trainer.args.max_steps = steps_done + chunk_steps

        try:
            trainer.train(resume_from_checkpoint=True if steps_done > 0 else False)
        except Exception as e:
            # Resume might fail on first call — retry without resume
            if steps_done > 0:
                print(f"  [warn] Resume failed ({e}), trying fresh train call...")
            try:
                trainer.train()
            except torch.cuda.OutOfMemoryError:
                print("  [OOM] Reducing batch size not possible. Saving and aborting.")
                break
            except Exception as e2:
                print(f"  [error] Training failed: {e2}")
                break

        steps_done += chunk_steps

        # Intermediate eval
        if steps_done < args.max_steps:
            print(f"\n--- Eval at step {steps_done} ---")
            model.eval()
            pope_result = eval_pope(model, processor, device, args.pope_eval_n)
            blind_result = eval_blind_gap(model, processor, device, args.blind_eval_n)
            print(f"  POPE-Adv: {pope_result['acc']:.1f}% "
                  f"(yes={pope_result['yes']}, no={pope_result['no']})")
            print(f"  Blind Gap: {blind_result['gap']:.1f}pp")
            eval_log.append({
                "step": steps_done,
                "pope": pope_result,
                "blind": blind_result,
            })
            model.train()

    # Final eval
    print(f"\n--- Final evaluation (step {steps_done}) ---")
    model.eval()
    pope_final = eval_pope(model, processor, device, args.pope_eval_n)
    blind_final = eval_blind_gap(model, processor, device, args.blind_eval_n)
    print(f"  POPE-Adv: {pope_final['acc']:.1f}% "
          f"(yes={pope_final['yes']}, no={pope_final['no']})")
    print(f"  Blind Gap: {blind_final['gap']:.1f}pp "
          f"(real={blind_final['real_acc']:.1f}%, "
          f"blind={blind_final['blind_acc']:.1f}%)")
    eval_log.append({
        "step": steps_done,
        "pope": pope_final,
        "blind": blind_final,
    })

    elapsed = time.time() - t0

    # Analyze results
    pope_delta = pope_final["acc"] - pope_baseline["acc"]
    gap_delta = blind_final["gap"] - blind_baseline["gap"]

    print(f"\n{'='*60}")
    print(f"Block 2 DPO Results Summary")
    print(f"{'='*60}")
    print(f"  POPE-Adv: {pope_baseline['acc']:.1f}% -> {pope_final['acc']:.1f}% "
          f"({pope_delta:+.1f}pp)")
    print(f"  Blind Gap: {blind_baseline['gap']:.1f}pp -> {blind_final['gap']:.1f}pp "
          f"({gap_delta:+.1f}pp)")

    if pope_delta < -10:
        print(f"\n  COLLAPSE DETECTED: POPE dropped {abs(pope_delta):.1f}pp")
        print(f"  DPO also collapses on this model. Consider:")
        print(f"  - Reduce beta (currently {args.beta})")
        print(f"  - Fewer steps")
        print(f"  - Better preference pairs (stricter filtering)")
    elif pope_delta < -3:
        print(f"\n  MODERATE DAMAGE: POPE dropped {abs(pope_delta):.1f}pp")
    else:
        print(f"\n  STABLE: POPE delta within tolerance")
        if gap_delta > 0:
            print(f"  Blind Gap IMPROVED by {gap_delta:.1f}pp (more image-dependent)")
        else:
            print(f"  Blind Gap unchanged or decreased ({gap_delta:+.1f}pp)")

    # Save results
    train_log = []
    try:
        train_log = trainer.state.log_history
    except Exception:
        pass

    results = {
        "phase": 2,
        "timestamp": datetime.now().isoformat(),
        "args": {k: str(v) for k, v in vars(args).items()},
        "baseline_pope": pope_baseline,
        "baseline_blind": blind_baseline,
        "final_pope": pope_final,
        "final_blind": blind_final,
        "pope_delta": pope_delta,
        "gap_delta": gap_delta,
        "eval_log": eval_log,
        "train_log": train_log,
        "elapsed_minutes": elapsed / 60,
        "steps_completed": steps_done,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = results_dir / f"phase2_dpo_{ts}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved results to {results_path}")

    # Save final model
    final_ckpt = ckpt_dir / "final"
    try:
        trainer.save_model(str(final_ckpt))
        print(f"  Saved model to {final_ckpt}")
    except Exception as e:
        print(f"  [warn] Could not save model: {e}")

    # Cleanup
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)

    phases = args.phase
    print(f"VIGIL Block 2 — DPO with IIG Preferences")
    print(f"  Phases to run: {phases}")
    print(f"  Timestamp: {datetime.now().isoformat()}\n")

    if 1 in phases:
        run_phase1(args)

    if 2 in phases:
        run_phase2(args)

    print("\nBlock 2 complete.")


if __name__ == "__main__":
    main()
