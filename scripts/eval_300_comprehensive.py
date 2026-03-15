"""
VIGIL Comprehensive 4-Benchmark Evaluation

Evaluates multiple model checkpoints on:
  1. POPE (300 samples) — yes/no visual QA, per-split breakdown
  2. TextVQA (100 samples) — open-ended visual QA, min-match accuracy
  3. Blind Test (100 samples) — POPE with black image, Gap = real_acc - blind_acc
  4. MME (300 image pairs) — pair-based scoring, Perception + Cognition

Usage:
    # Baseline only
    python scripts/eval_300_comprehensive.py --conditions baseline

    # Multiple conditions
    python scripts/eval_300_comprehensive.py \
        --conditions baseline \
        --conditions exp1:checkpoints/phase6c/gated_only/final \
        --conditions exp3:checkpoints/phase6c/gated_curriculum/final \
        --conditions exp6:checkpoints/phase7/exp6_fixed/final

    # Custom sample counts
    python scripts/eval_300_comprehensive.py \
        --conditions baseline \
        --pope-samples 600 --textvqa-samples 200 --blind-samples 200 --mme-pairs 500

Output: lab/reports/eval_300/results_{timestamp}.json + printed comparison table.
"""

import os
import sys
import gc
import json
import re
import time
import string
import argparse
import random
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from datetime import datetime
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

# ─── Constants ────────────────────────────────────────────────────────────────

HF_ID = "Qwen/Qwen3-VL-2B-Thinking"
PROJECT_ROOT = Path(__file__).parent.parent

POPE_DATA_PATH = PROJECT_ROOT / "data" / "eval" / "pope"
TEXTVQA_DATA_PATH = PROJECT_ROOT / "data" / "eval" / "textvqa_val"
MME_DATA_PATH = PROJECT_ROOT / "data" / "eval" / "mme"
OUTPUT_DIR = PROJECT_ROOT / "lab" / "reports" / "eval_300"

POPE_SPLITS = ["random", "popular", "adversarial"]

PERCEPTION_SUBTASKS = [
    "existence", "count", "position", "color", "posters",
    "celebrity", "scene", "landmark", "artwork", "OCR",
]
COGNITION_SUBTASKS = [
    "commonsense_reasoning", "numerical_calculation",
    "text_translation", "code_reasoning",
]


# ─── Text extraction utilities ────────────────────────────────────────────────

def split_thinking(text):
    """Split thinking tags from answer text."""
    m = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    m = re.search(r'</think>', text)
    if m:
        return text[:m.start()].strip(), text[m.end():].strip()
    return "", text.strip()


def extract_yes_no(raw):
    """Extract yes/no from model output, handling thinking tags.

    VLMEvalKit-compatible: process punctuation, word-level yes/no check.
    Returns 'yes', 'no', or None.
    """
    _, answer = split_thinking(raw)
    if not answer:
        answer = raw
    text = answer.strip().lower()
    for p in string.punctuation:
        text = text.replace(p, " ")
    words = text.split()
    # Check first 5 words for yes/no
    for w in words[:5]:
        if w in ("yes", "true"):
            return "yes"
        if w in ("no", "false"):
            return "no"
    # Fallback: anywhere in text
    if "yes" in words:
        return "yes"
    if "no" in words:
        return "no"
    return None


def extract_yes_no_mme(raw):
    """Extract Yes/No for MME scoring (capitalized to match MME ground truth)."""
    result = extract_yes_no(raw)
    if result == "yes":
        return "Yes"
    if result == "no":
        return "No"
    return "Unknown"


def extract_answer(raw, qtype="short_answer"):
    """Extract answer from model output based on question type."""
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
    # Short answer: first line, truncated
    return text.split("\n")[0].strip()[:100]


def textvqa_accuracy(pred, answers_all):
    """Compute TextVQA accuracy: min(match_count / 3, 1.0).

    Tries exact match, then substring containment in both directions.
    """
    if not pred:
        return 0.0
    pred_clean = pred.strip().lower()
    # Strip common prefixes
    for prefix in ["the answer is ", "it says ", "the text reads ",
                   "the brand is ", "it is ", "this is "]:
        if pred_clean.startswith(prefix):
            pred_clean = pred_clean[len(prefix):]
    # Exact match
    match_count = sum(1 for a in answers_all if a.strip().lower() == pred_clean)
    if match_count > 0:
        return min(match_count / 3.0, 1.0)
    # Pred in answer
    match_count = sum(1 for a in answers_all if a.strip().lower() in pred_clean)
    if match_count > 0:
        return min(match_count / 3.0, 1.0)
    # Answer in pred
    match_count = sum(1 for a in answers_all if pred_clean in a.strip().lower())
    return min(match_count / 3.0, 1.0)


# ─── Model loading ────────────────────────────────────────────────────────────

def load_model(model_path=None, dtype=torch.bfloat16):
    """Load Qwen3-VL-2B-Thinking, optionally from a fine-tuned checkpoint.

    Args:
        model_path: Path to fine-tuned checkpoint directory. If None, loads
                    the base HF model.
        dtype: Model dtype.

    Returns:
        (model, processor) tuple.
    """
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    load_path = model_path or HF_ID
    print(f"[eval] Loading model from: {load_path}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        load_path, torch_dtype=dtype, device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(HF_ID, trust_remote_code=True)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[eval] Model loaded: {n_params:.0f}M params, dtype={dtype}")
    return model, processor


def unload_model(model):
    """Free GPU memory."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─── Input preparation ────────────────────────────────────────────────────────

def prepare_inputs(processor, image, question, device, enable_thinking=True):
    """Prepare model inputs for a single image + question.

    Uses Qwen3-VL chat template with thinking mode enabled.
    """
    from qwen_vl_utils import process_vision_info

    content = [
        {"type": "image", "image": image},
        {"type": "text", "text": question},
    ]
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    imgs, _, _ = process_vision_info(messages, return_video_kwargs=True)

    inputs = processor(
        text=[text], images=imgs, return_tensors="pt", padding=True,
    )
    return {k: v.to(device) for k, v in inputs.items()}


@torch.no_grad()
def generate_answer(model, processor, image, question, max_new_tokens=256,
                    enable_thinking=True):
    """Generate a single answer. Returns raw output string."""
    device = next(model.parameters()).device
    inputs = prepare_inputs(processor, image, question, device,
                            enable_thinking=enable_thinking)
    generated_ids = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
    )
    input_len = inputs["input_ids"].shape[1]
    output_ids = generated_ids[0][input_len:]
    raw = processor.tokenizer.decode(output_ids, skip_special_tokens=False)
    # Clean special tokens
    for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
        raw = raw.replace(tok, "")
    return raw.strip()


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_pope_data(max_samples=300):
    """Load POPE dataset from local disk cache.

    Returns list of dicts with balanced splits (random/popular/adversarial).
    """
    from datasets import load_from_disk

    if POPE_DATA_PATH.exists():
        print(f"[data] Loading POPE from disk: {POPE_DATA_PATH}")
        ds = load_from_disk(str(POPE_DATA_PATH))
    else:
        print("[data] POPE not found locally, downloading...")
        from datasets import load_dataset
        ds = load_dataset("lmms-lab/POPE", split="test")
        ds.save_to_disk(str(POPE_DATA_PATH))

    per_split = defaultdict(list)
    per_sample = max_samples // 3

    for i in range(len(ds)):
        row = ds[i]
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

    print(f"[data] POPE: {len(samples)} samples "
          f"({', '.join(f'{s}={len(per_split[s])}' for s in POPE_SPLITS)})")
    return samples


def load_textvqa_data(max_samples=100):
    """Load TextVQA validation set from local disk cache."""
    from datasets import load_from_disk

    if TEXTVQA_DATA_PATH.exists():
        print(f"[data] Loading TextVQA from disk: {TEXTVQA_DATA_PATH}")
        ds = load_from_disk(str(TEXTVQA_DATA_PATH))
    else:
        print("[data] TextVQA not found locally, downloading...")
        from datasets import load_dataset
        ds = load_dataset("lmms-lab/textvqa", split="validation")
        ds.save_to_disk(str(TEXTVQA_DATA_PATH))

    samples = []
    for i in range(len(ds)):
        if len(samples) >= max_samples:
            break
        row = ds[i]
        img = row.get("image")
        if img is None:
            continue
        answers = row.get("answers", [])
        if not answers:
            continue
        ans = Counter(answers).most_common(1)[0][0]
        samples.append({
            "image": img,
            "question": row["question"],
            "answer": ans,
            "answers_all": answers,
        })

    print(f"[data] TextVQA: {len(samples)} samples")
    return samples


def load_mme_data(max_pairs=300):
    """Load MME dataset, grouped by question_id (image pairs).

    Returns list of pair groups, each containing 2 question dicts.
    """
    from datasets import load_from_disk

    if MME_DATA_PATH.exists():
        print(f"[data] Loading MME from disk: {MME_DATA_PATH}")
        ds = load_from_disk(str(MME_DATA_PATH))
    else:
        print("[data] MME not found locally, downloading...")
        from datasets import load_dataset
        ds = load_dataset("lmms-lab/MME", split="test")
        ds.save_to_disk(str(MME_DATA_PATH))

    grouped = defaultdict(list)
    for i in range(len(ds)):
        row = ds[i]
        grouped[row["question_id"]].append({
            "index": i,
            "question_id": row["question_id"],
            "question": row["question"],
            "answer": row["answer"].strip(),
            "category": row["category"],
            "image": row["image"],
        })

    pairs = list(grouped.values())
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    n_questions = sum(len(p) for p in pairs)
    cats = defaultdict(int)
    for p in pairs:
        cats[p[0]["category"]] += 1
    print(f"[data] MME: {len(pairs)} image pairs ({n_questions} questions), "
          f"{len(cats)} subtasks")
    return pairs


# ─── Benchmark runners ────────────────────────────────────────────────────────

def run_pope(model, processor, samples, label=""):
    """Run POPE evaluation. Returns dict with overall and per-split metrics."""
    print(f"\n  [{label}] POPE: evaluating {len(samples)} samples...")
    t0 = time.time()
    correct, total = 0, 0
    per_split = defaultdict(lambda: {"correct": 0, "total": 0})

    for i, s in enumerate(samples):
        try:
            raw = generate_answer(model, processor, s["image"], s["question"],
                                  max_new_tokens=256)
            pred = extract_yes_no(raw)
            is_correct = pred == s["answer"]
            if is_correct:
                correct += 1
            total += 1

            cat = s.get("category", "unknown")
            per_split[cat]["total"] += 1
            if is_correct:
                per_split[cat]["correct"] += 1

        except torch.cuda.OutOfMemoryError:
            print(f"    [{i}] OOM, skipping")
            torch.cuda.empty_cache()
            total += 1
        except Exception as e:
            print(f"    [{i}] Error: {e}")
            total += 1

        if (i + 1) % 100 == 0:
            acc = correct / max(total, 1) * 100
            elapsed = time.time() - t0
            print(f"    [{label}] POPE {i+1}/{len(samples)}: "
                  f"acc={acc:.1f}% ({elapsed:.0f}s)")

    overall_acc = correct / max(total, 1)
    split_results = {}
    for s in POPE_SPLITS:
        d = per_split[s]
        split_results[s] = {
            "acc": d["correct"] / max(d["total"], 1),
            "correct": d["correct"],
            "total": d["total"],
        }

    elapsed = time.time() - t0
    print(f"    [{label}] POPE done: acc={overall_acc*100:.1f}% "
          f"({correct}/{total}) in {elapsed:.0f}s")

    return {
        "acc": overall_acc,
        "correct": correct,
        "total": total,
        "per_split": split_results,
        "elapsed_s": round(elapsed, 1),
    }


def run_textvqa(model, processor, samples, label=""):
    """Run TextVQA evaluation. Returns dict with accuracy metrics."""
    print(f"\n  [{label}] TextVQA: evaluating {len(samples)} samples...")
    t0 = time.time()
    total_acc, total = 0.0, 0

    for i, s in enumerate(samples):
        try:
            raw = generate_answer(model, processor, s["image"],
                                  s["question"] + " Answer briefly.",
                                  max_new_tokens=512)
            pred = extract_answer(raw, "short_answer")
            acc = textvqa_accuracy(pred, s.get("answers_all", [s["answer"]]))
            total_acc += acc
            total += 1

        except torch.cuda.OutOfMemoryError:
            print(f"    [{i}] OOM, skipping")
            torch.cuda.empty_cache()
            total += 1
        except Exception as e:
            print(f"    [{i}] Error: {e}")
            total += 1

        if (i + 1) % 50 == 0:
            avg_acc = total_acc / max(total, 1) * 100
            elapsed = time.time() - t0
            print(f"    [{label}] TextVQA {i+1}/{len(samples)}: "
                  f"acc={avg_acc:.1f}% ({elapsed:.0f}s)")

    avg_acc = total_acc / max(total, 1)
    elapsed = time.time() - t0
    print(f"    [{label}] TextVQA done: acc={avg_acc*100:.1f}% "
          f"({total} samples) in {elapsed:.0f}s")

    return {
        "acc": avg_acc,
        "total": total,
        "elapsed_s": round(elapsed, 1),
    }


def run_blind(model, processor, samples, label=""):
    """Run Blind Test on POPE samples.

    Generates with real image and black image, computes Gap.
    """
    print(f"\n  [{label}] Blind Test: evaluating {len(samples)} samples...")
    t0 = time.time()
    real_correct, blind_correct, total = 0, 0, 0

    for i, s in enumerate(samples):
        try:
            # Real image
            raw_real = generate_answer(model, processor, s["image"],
                                       s["question"], max_new_tokens=256)
            pred_real = extract_yes_no(raw_real)
            if pred_real == s["answer"]:
                real_correct += 1

            # Black image
            black = Image.new('RGB', s["image"].size, (0, 0, 0))
            raw_blind = generate_answer(model, processor, black,
                                         s["question"], max_new_tokens=256)
            pred_blind = extract_yes_no(raw_blind)
            if pred_blind == s["answer"]:
                blind_correct += 1

            total += 1

        except torch.cuda.OutOfMemoryError:
            print(f"    [{i}] OOM, skipping")
            torch.cuda.empty_cache()
            total += 1
        except Exception as e:
            print(f"    [{i}] Error: {e}")
            total += 1

        if (i + 1) % 50 == 0:
            r_acc = real_correct / max(total, 1) * 100
            b_acc = blind_correct / max(total, 1) * 100
            gap = r_acc - b_acc
            elapsed = time.time() - t0
            print(f"    [{label}] Blind {i+1}/{len(samples)}: "
                  f"real={r_acc:.1f}% blind={b_acc:.1f}% gap={gap:.1f}pp "
                  f"({elapsed:.0f}s)")

    real_acc = real_correct / max(total, 1)
    blind_acc = blind_correct / max(total, 1)
    gap = real_acc - blind_acc
    elapsed = time.time() - t0
    print(f"    [{label}] Blind done: real={real_acc*100:.1f}% "
          f"blind={blind_acc*100:.1f}% gap={gap*100:.1f}pp in {elapsed:.0f}s")

    return {
        "real_acc": real_acc,
        "blind_acc": blind_acc,
        "gap": gap,
        "real_correct": real_correct,
        "blind_correct": blind_correct,
        "total": total,
        "elapsed_s": round(elapsed, 1),
    }


def run_mme(model, processor, pairs, label=""):
    """Run MME evaluation with pair-based scoring.

    Returns dict with per-subtask scores and Perception/Cognition totals.
    """
    print(f"\n  [{label}] MME: evaluating {len(pairs)} image pairs...")
    t0 = time.time()
    results = {}
    total_q, correct_q = 0, 0

    for pi, pair in enumerate(pairs):
        qid = pair[0]["question_id"]
        category = pair[0]["category"]
        pair_results = []

        for q_item in pair:
            try:
                raw = generate_answer(model, processor, q_item["image"],
                                      q_item["question"], max_new_tokens=2048)
                extracted = extract_yes_no_mme(raw)
            except torch.cuda.OutOfMemoryError:
                print(f"    [{pi}] OOM on {qid}, skipping")
                torch.cuda.empty_cache()
                extracted = "Unknown"
            except Exception as e:
                print(f"    [{pi}] Error on {qid}: {e}")
                extracted = "Unknown"

            gt = q_item["answer"]
            pair_results.append({
                "question_id": qid,
                "question": q_item["question"],
                "gt_answer": gt,
                "extracted": extracted,
                "correct": extracted == gt,
                "category": category,
            })
            total_q += 1
            if extracted == gt:
                correct_q += 1

        results[qid] = pair_results

        if (pi + 1) % 50 == 0 or (pi + 1) == len(pairs):
            acc = correct_q / max(total_q, 1) * 100
            elapsed = time.time() - t0
            rate = (pi + 1) / elapsed * 60 if elapsed > 0 else 0
            print(f"    [{label}] MME {pi+1}/{len(pairs)} images "
                  f"({rate:.0f} img/min), acc+={acc:.1f}%")

    # Compute pair-based scores
    report = compute_mme_scores(results)
    elapsed = time.time() - t0
    print(f"    [{label}] MME done in {elapsed:.0f}s")

    return {
        "report": report,
        "perception": report["_Perception"]["score"],
        "cognition": report["_Cognition"]["score"],
        "total": report["_Total"]["score"],
        "perception_max": report["_Perception"]["max_score"],
        "cognition_max": report["_Cognition"]["max_score"],
        "elapsed_s": round(elapsed, 1),
    }


def compute_mme_scores(results):
    """Compute MME pair-based scores.

    For each image, both questions must be correct to score 1 point.
    """
    subtask_scores = defaultdict(lambda: {"correct_pairs": 0, "total_pairs": 0,
                                          "correct_q": 0, "total_q": 0})
    for qid, pair_results in results.items():
        if not pair_results:
            continue
        category = pair_results[0]["category"]
        st = subtask_scores[category]
        st["total_pairs"] += 1

        all_correct = True
        for r in pair_results:
            st["total_q"] += 1
            if r["extracted"] == r["gt_answer"]:
                st["correct_q"] += 1
            else:
                all_correct = False

        if all_correct:
            st["correct_pairs"] += 1

    report = {}
    perception_total = 0
    cognition_total = 0

    for subtask in sorted(subtask_scores.keys()):
        s = subtask_scores[subtask]
        score = s["correct_pairs"]
        max_score = s["total_pairs"]
        acc_plus = s["correct_q"] / s["total_q"] * 100 if s["total_q"] > 0 else 0

        report[subtask] = {
            "score": score,
            "max_score": max_score,
            "acc_plus": round(acc_plus, 1),
            "correct_pairs": s["correct_pairs"],
            "total_pairs": s["total_pairs"],
            "correct_q": s["correct_q"],
            "total_q": s["total_q"],
        }

        if subtask in PERCEPTION_SUBTASKS:
            perception_total += score
        elif subtask in COGNITION_SUBTASKS:
            cognition_total += score

    perception_max = sum(report[s]["max_score"] for s in PERCEPTION_SUBTASKS
                         if s in report)
    cognition_max = sum(report[s]["max_score"] for s in COGNITION_SUBTASKS
                        if s in report)

    report["_Perception"] = {"score": perception_total, "max_score": perception_max}
    report["_Cognition"] = {"score": cognition_total, "max_score": cognition_max}
    report["_Total"] = {"score": perception_total + cognition_total,
                        "max_score": perception_max + cognition_max}

    return report


# ─── Comparison table ─────────────────────────────────────────────────────────

def print_comparison_table(all_results):
    """Print a formatted comparison table across all conditions and benchmarks."""
    print(f"\n{'=' * 90}")
    print("COMPREHENSIVE EVALUATION RESULTS")
    print(f"{'=' * 90}")

    # Header
    header = (f"{'Condition':<20} "
              f"{'POPE':>8} "
              f"{'TextVQA':>9} "
              f"{'Blind-R':>9} "
              f"{'Blind-B':>9} "
              f"{'Gap':>8} "
              f"{'MME-P':>7} "
              f"{'MME-C':>7} "
              f"{'MME-T':>7}")
    print(header)
    print("-" * 90)

    for label, results in all_results.items():
        pope_acc = results.get("pope", {}).get("acc", float("nan"))
        tvqa_acc = results.get("textvqa", {}).get("acc", float("nan"))
        real_acc = results.get("blind", {}).get("real_acc", float("nan"))
        blind_acc = results.get("blind", {}).get("blind_acc", float("nan"))
        gap = results.get("blind", {}).get("gap", float("nan"))
        mme_p = results.get("mme", {}).get("perception", float("nan"))
        mme_c = results.get("mme", {}).get("cognition", float("nan"))
        mme_t = results.get("mme", {}).get("total", float("nan"))

        def fmt_pct(v):
            if v != v:  # NaN check
                return "  —"
            return f"{v*100:6.1f}%"

        def fmt_pp(v):
            if v != v:
                return "  —"
            return f"{v*100:5.1f}pp"

        def fmt_int(v):
            if v != v:
                return "  —"
            return f"{int(v):>5}"

        row = (f"{label:<20} "
               f"{fmt_pct(pope_acc):>8} "
               f"{fmt_pct(tvqa_acc):>9} "
               f"{fmt_pct(real_acc):>9} "
               f"{fmt_pct(blind_acc):>9} "
               f"{fmt_pp(gap):>8} "
               f"{fmt_int(mme_p):>7} "
               f"{fmt_int(mme_c):>7} "
               f"{fmt_int(mme_t):>7}")
        print(row)

    print(f"{'=' * 90}")

    # POPE per-split breakdown
    has_pope = any("pope" in r and "per_split" in r.get("pope", {})
                   for r in all_results.values())
    if has_pope:
        print(f"\nPOPE Per-Split Breakdown:")
        print(f"{'Condition':<20} {'random':>10} {'popular':>10} {'adversarial':>12}")
        print("-" * 55)
        for label, results in all_results.items():
            pope = results.get("pope", {})
            splits = pope.get("per_split", {})
            parts = []
            for s in POPE_SPLITS:
                sd = splits.get(s, {})
                a = sd.get("acc", float("nan"))
                if a != a:
                    parts.append("  —")
                else:
                    parts.append(f"{a*100:6.1f}%")
            print(f"{label:<20} {parts[0]:>10} {parts[1]:>10} {parts[2]:>12}")
        print()


# ─── Parse conditions ─────────────────────────────────────────────────────────

def parse_conditions(condition_strings):
    """Parse condition strings into (label, model_path) tuples.

    Format: 'label' or 'label:path'. 'baseline' maps to None (base HF model).
    """
    conditions = []
    for s in condition_strings:
        if ":" in s:
            label, path = s.split(":", 1)
            conditions.append((label.strip(), path.strip()))
        else:
            label = s.strip()
            if label.lower() == "baseline":
                conditions.append(("baseline", None))
            else:
                conditions.append((label, None))
    return conditions


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_full_eval(conditions, pope_samples=300, textvqa_samples=100,
                  blind_samples=100, mme_pairs=300, benchmarks=None):
    """Run comprehensive evaluation across all conditions and benchmarks.

    Args:
        conditions: list of (label, model_path) tuples.
        pope_samples: number of POPE samples.
        textvqa_samples: number of TextVQA samples.
        blind_samples: number of Blind Test samples.
        mme_pairs: number of MME image pairs.
        benchmarks: list of benchmark names to run. None = all.

    Returns:
        dict mapping label -> benchmark results.
    """
    if benchmarks is None:
        benchmarks = ["pope", "textvqa", "blind", "mme"]

    # Pre-load all datasets (before loading model to catch data errors early)
    datasets = {}
    if "pope" in benchmarks:
        datasets["pope"] = load_pope_data(max_samples=pope_samples)
    if "textvqa" in benchmarks:
        datasets["textvqa"] = load_textvqa_data(max_samples=textvqa_samples)
    if "blind" in benchmarks:
        # Blind uses POPE samples
        if "pope" in datasets and blind_samples <= len(datasets["pope"]):
            datasets["blind"] = datasets["pope"][:blind_samples]
        else:
            datasets["blind"] = load_pope_data(max_samples=blind_samples)
    if "mme" in benchmarks:
        datasets["mme"] = load_mme_data(max_pairs=mme_pairs)

    all_results = {}

    for ci, (label, model_path) in enumerate(conditions):
        print(f"\n{'=' * 70}")
        print(f"CONDITION {ci+1}/{len(conditions)}: {label}")
        if model_path:
            print(f"  Model: {model_path}")
        else:
            print(f"  Model: {HF_ID} (base)")
        print(f"{'=' * 70}")

        # Verify checkpoint exists
        if model_path and not Path(model_path).exists():
            print(f"  WARNING: checkpoint not found at {model_path}, skipping")
            all_results[label] = {"error": f"checkpoint not found: {model_path}"}
            continue

        # Load model once for all benchmarks
        model, processor = load_model(model_path)
        condition_results = {"model_path": model_path or HF_ID}

        # Run each benchmark
        if "pope" in benchmarks:
            try:
                condition_results["pope"] = run_pope(
                    model, processor, datasets["pope"], label)
            except Exception as e:
                print(f"  POPE failed: {e}")
                condition_results["pope"] = {"error": str(e)}

        if "textvqa" in benchmarks:
            try:
                condition_results["textvqa"] = run_textvqa(
                    model, processor, datasets["textvqa"], label)
            except Exception as e:
                print(f"  TextVQA failed: {e}")
                condition_results["textvqa"] = {"error": str(e)}

        if "blind" in benchmarks:
            try:
                condition_results["blind"] = run_blind(
                    model, processor, datasets["blind"], label)
            except Exception as e:
                print(f"  Blind failed: {e}")
                condition_results["blind"] = {"error": str(e)}

        if "mme" in benchmarks:
            try:
                condition_results["mme"] = run_mme(
                    model, processor, datasets["mme"], label)
            except Exception as e:
                print(f"  MME failed: {e}")
                condition_results["mme"] = {"error": str(e)}

        all_results[label] = condition_results

        # Free GPU memory before next condition
        unload_model(model)

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="VIGIL Comprehensive 4-Benchmark Evaluation",
    )
    parser.add_argument(
        "--conditions", type=str, action="append", required=True,
        help="Condition spec: 'label' or 'label:checkpoint_path'. "
             "'baseline' uses base HF model. Can be repeated.",
    )
    parser.add_argument("--pope-samples", type=int, default=300,
                        help="Number of POPE samples (default: 300)")
    parser.add_argument("--textvqa-samples", type=int, default=100,
                        help="Number of TextVQA samples (default: 100)")
    parser.add_argument("--blind-samples", type=int, default=100,
                        help="Number of Blind Test samples (default: 100)")
    parser.add_argument("--mme-pairs", type=int, default=300,
                        help="Number of MME image pairs (default: 300)")
    parser.add_argument("--benchmarks", type=str, nargs="+", default=None,
                        choices=["pope", "textvqa", "blind", "mme"],
                        help="Benchmarks to run. Default: all four.")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Directory for results JSON.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Parse conditions
    conditions = parse_conditions(args.conditions)
    print(f"[eval] {len(conditions)} condition(s):")
    for label, path in conditions:
        print(f"  - {label}: {path or HF_ID}")

    print(f"[eval] Benchmarks: {args.benchmarks or ['pope', 'textvqa', 'blind', 'mme']}")
    print(f"[eval] Samples: POPE={args.pope_samples}, TextVQA={args.textvqa_samples}, "
          f"Blind={args.blind_samples}, MME={args.mme_pairs} pairs")

    # Run evaluation
    all_results = run_full_eval(
        conditions,
        pope_samples=args.pope_samples,
        textvqa_samples=args.textvqa_samples,
        blind_samples=args.blind_samples,
        mme_pairs=args.mme_pairs,
        benchmarks=args.benchmarks,
    )

    # Print comparison table
    print_comparison_table(all_results)

    # Save results
    result_file = output_dir / f"results_{timestamp}.json"
    save_data = {
        "timestamp": timestamp,
        "config": {
            "pope_samples": args.pope_samples,
            "textvqa_samples": args.textvqa_samples,
            "blind_samples": args.blind_samples,
            "mme_pairs": args.mme_pairs,
            "benchmarks": args.benchmarks or ["pope", "textvqa", "blind", "mme"],
            "conditions": {label: (path or HF_ID) for label, path in conditions},
        },
        "results": all_results,
    }

    with open(result_file, "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\n[eval] Results saved to {result_file}")
    return all_results


if __name__ == "__main__":
    main()
