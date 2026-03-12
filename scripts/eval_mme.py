"""
VIGIL MME Benchmark Evaluation — Qwen3-VL-2B-Thinking

MME evaluates 14 Perception subtasks and 4 Cognition subtasks.
Each image has 2 questions (one yes, one no). An image scores 1 point
only if BOTH questions are answered correctly. Subtask score = sum of
per-image points. Total Perception/Cognition = sum of subtask scores.

Usage:
    # Baseline (no checkpoint)
    python scripts/eval_mme.py

    # Best Phase 2 checkpoint
    python scripts/eval_mme.py --model-path checkpoints/phase2_grpo_lsr/round4/best

    # Limit samples for smoke test
    python scripts/eval_mme.py --max-samples 20

    # Compare baseline vs checkpoint
    python scripts/eval_mme.py --compare \
        --model-path checkpoints/phase2_grpo_lsr/round4/best
"""

import os
import sys
import json
import re
import argparse
import gc
import time
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Constants ────────────────────────────────────────────────────────────────

HF_ID = "Qwen/Qwen3-VL-2B-Thinking"
DATA_PATH = Path("data/eval/mme")

# MME subtask classification
PERCEPTION_SUBTASKS = [
    "existence", "count", "position", "color", "posters",
    "celebrity", "scene", "landmark", "artwork", "OCR",
]
COGNITION_SUBTASKS = [
    "commonsense_reasoning", "numerical_calculation",
    "text_translation", "code_reasoning",
]

# Thinking-mode generation settings (match Qwen3-VL-Thinking paper)
GEN_KWARGS = dict(
    max_new_tokens=2048,
    temperature=1.0,
    top_p=0.95,
    top_k=20,
    do_sample=True,
)


# ─── Answer extraction ───────────────────────────────────────────────────────

def split_thinking(text):
    """Split thinking tags from answer."""
    m = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    m = re.search(r'</think>', text)
    if m:
        return text[:m.start()].strip(), text[m.end():].strip()
    return "", text.strip()


def extract_yes_no(raw_output):
    """Extract yes/no from model output, handling thinking tags.

    Uses VLMEvalKit-compatible parsing: process punctuation, then
    check for yes/no words.
    """
    _, answer_text = split_thinking(raw_output)
    if not answer_text:
        answer_text = raw_output

    s = answer_text.lower().strip()

    # Remove common punctuation for word-level matching
    punct = [
        ';', r'/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\',
        '_', '-', '>', '<', '@', '`', ',', '?', '!',
    ]
    for p in punct:
        s = s.replace(p, ' ')
    s = re.sub(r'(?<!\d)\.(?!\d)', '', s)

    words = s.split()
    has_yes = 'yes' in words
    has_no = 'no' in words

    if has_yes and not has_no:
        return "Yes"
    if has_no and not has_yes:
        return "No"

    # Fallback: check first word
    first = words[0] if words else ""
    if first.startswith("yes"):
        return "Yes"
    if first.startswith("no"):
        return "No"

    return "Unknown"


# ─── Model loading ───────────────────────────────────────────────────────────

def load_model(model_path=None, dtype=torch.bfloat16):
    """Load Qwen3-VL-2B-Thinking, optionally from a fine-tuned checkpoint.

    Args:
        model_path: Path to fine-tuned checkpoint directory. If None, loads
                    the base HF model.
        dtype: Model dtype. bfloat16 recommended for A100.

    Returns:
        (model, processor) tuple.
    """
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    load_path = model_path or HF_ID
    print(f"[mme] Loading model from: {load_path}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        load_path, torch_dtype=dtype, device_map="auto",
    )
    # Always use base processor (tokenizer/chat template unchanged by fine-tuning)
    processor = AutoProcessor.from_pretrained(HF_ID)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[mme] Model loaded: {n_params:.0f}M params, dtype={dtype}")
    return model, processor


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_mme_data(max_samples=None):
    """Load MME dataset from local disk cache.

    Returns list of dicts grouped by question_id (image), each containing
    the two yes/no question variants.
    """
    from datasets import load_from_disk

    if DATA_PATH.exists():
        print(f"[mme] Loading from disk: {DATA_PATH}")
        ds = load_from_disk(str(DATA_PATH))
    else:
        print("[mme] Local data not found, downloading from HuggingFace...")
        from datasets import load_dataset
        ds = load_dataset("lmms-lab/MME", split="test")
        ds.save_to_disk(str(DATA_PATH))
        print(f"[mme] Saved to {DATA_PATH}")

    # Group by question_id (each image has 2 questions)
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

    # Convert to list and optionally limit
    pairs = list(grouped.values())
    if max_samples is not None:
        pairs = pairs[:max_samples]

    n_images = len(pairs)
    n_questions = sum(len(p) for p in pairs)
    cats = defaultdict(int)
    for p in pairs:
        cats[p[0]["category"]] += 1
    print(f"[mme] Loaded {n_images} images ({n_questions} questions), "
          f"{len(cats)} subtasks")
    return pairs


# ─── Generation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_answer(model, processor, image, question):
    """Generate answer for a single MME question.

    Uses thinking mode with Qwen3-VL chat template.
    """
    from qwen_vl_utils import process_vision_info

    # MME questions already end with "Please answer yes or no."
    # so we use the question text directly.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True,
    )
    images, videos, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True,
    )

    inputs = processor(
        text=[text],
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    generated_ids = model.generate(**inputs, **GEN_KWARGS)

    input_len = inputs["input_ids"].shape[1]
    output_ids = generated_ids[0][input_len:]
    response = processor.tokenizer.decode(output_ids, skip_special_tokens=True)
    return response.strip()


# ─── Scoring ──────────────────────────────────────────────────────────────────

def compute_mme_scores(results):
    """Compute MME pair-based scores.

    MME scoring: for each image, if BOTH questions (yes-variant and no-variant)
    are answered correctly, the image scores 1 point. The subtask score is the
    sum of per-image points. Additionally, per-question accuracy (acc+) is
    reported as the number of correct individual answers.

    Returns:
        dict with per-subtask and aggregate Perception/Cognition scores.
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

    # Build report
    report = {}
    perception_total = 0
    cognition_total = 0

    for subtask in sorted(subtask_scores.keys()):
        s = subtask_scores[subtask]
        # MME score = correct_pairs (each pair = 1 point)
        # acc+ = correct_q / total_q * 100 (per-question accuracy)
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

    # Aggregate
    perception_max = sum(report[s]["max_score"] for s in PERCEPTION_SUBTASKS
                         if s in report)
    cognition_max = sum(report[s]["max_score"] for s in COGNITION_SUBTASKS
                        if s in report)

    report["_Perception"] = {
        "score": perception_total,
        "max_score": perception_max,
    }
    report["_Cognition"] = {
        "score": cognition_total,
        "max_score": cognition_max,
    }
    report["_Total"] = {
        "score": perception_total + cognition_total,
        "max_score": perception_max + cognition_max,
    }

    return report


def print_scores(report, label=""):
    """Print formatted MME score table."""
    header = f"MME Results{' — ' + label if label else ''}"
    print(f"\n{'=' * 70}")
    print(header)
    print(f"{'=' * 70}")
    print(f"{'Subtask':<30} {'Score':>8} {'Max':>8} {'Acc+':>8}")
    print("-" * 70)

    # Perception subtasks
    print("  PERCEPTION:")
    for subtask in PERCEPTION_SUBTASKS:
        if subtask in report:
            s = report[subtask]
            print(f"    {subtask:<26} {s['score']:>8} {s['max_score']:>8} "
                  f"{s['acc_plus']:>7.1f}%")
    p = report["_Perception"]
    print(f"  {'PERCEPTION TOTAL':<28} {p['score']:>8} {p['max_score']:>8}")

    # Cognition subtasks
    print("  COGNITION:")
    for subtask in COGNITION_SUBTASKS:
        if subtask in report:
            s = report[subtask]
            print(f"    {subtask:<26} {s['score']:>8} {s['max_score']:>8} "
                  f"{s['acc_plus']:>7.1f}%")
    c = report["_Cognition"]
    print(f"  {'COGNITION TOTAL':<28} {c['score']:>8} {c['max_score']:>8}")

    t = report["_Total"]
    print(f"\n  {'TOTAL':<28} {t['score']:>8} {t['max_score']:>8}")
    print(f"{'=' * 70}")


# ─── Comparison chart ─────────────────────────────────────────────────────────

def generate_comparison_chart(results_list, output_path):
    """Generate bar chart comparing Perception/Cognition across conditions.

    Args:
        results_list: list of (label, report_dict) tuples.
        output_path: Path to save the figure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[mme] matplotlib not available, skipping chart")
        return

    labels = [r[0] for r in results_list]
    perception_scores = [r[1]["_Perception"]["score"] for r in results_list]
    cognition_scores = [r[1]["_Cognition"]["score"] for r in results_list]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, perception_scores, width, label="Perception",
                   color="#4C72B0", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, cognition_scores, width, label="Cognition",
                   color="#DD8452", edgecolor="black", linewidth=0.5)

    # Value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{int(bar.get_height())}", ha="center", va="bottom", fontsize=11)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{int(bar.get_height())}", ha="center", va="bottom", fontsize=11)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("MME Benchmark — Perception vs Cognition", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(perception_scores), 1) * 1.15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[mme] Chart saved to {output_path}")


def generate_subtask_chart(results_list, output_path):
    """Generate per-subtask bar chart for detailed comparison."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    all_subtasks = PERCEPTION_SUBTASKS + COGNITION_SUBTASKS

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7),
                                    gridspec_kw={"width_ratios": [10, 4]})

    x_p = np.arange(len(PERCEPTION_SUBTASKS))
    x_c = np.arange(len(COGNITION_SUBTASKS))
    n = len(results_list)
    width = 0.8 / max(n, 1)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

    for i, (label, report) in enumerate(results_list):
        offset = (i - n / 2 + 0.5) * width
        p_scores = [report.get(s, {}).get("score", 0) for s in PERCEPTION_SUBTASKS]
        c_scores = [report.get(s, {}).get("score", 0) for s in COGNITION_SUBTASKS]
        color = colors[i % len(colors)]

        ax1.bar(x_p + offset, p_scores, width, label=label, color=color,
                edgecolor="black", linewidth=0.3)
        ax2.bar(x_c + offset, c_scores, width, label=label, color=color,
                edgecolor="black", linewidth=0.3)

    ax1.set_xticks(x_p)
    ax1.set_xticklabels(PERCEPTION_SUBTASKS, rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("Score", fontsize=11)
    ax1.set_title("Perception Subtasks", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    ax2.set_xticks(x_c)
    ax2.set_xticklabels(COGNITION_SUBTASKS, rotation=45, ha="right", fontsize=9)
    ax2.set_title("Cognition Subtasks", fontsize=12)
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle("MME Per-Subtask Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[mme] Subtask chart saved to {output_path}")


# ─── Main evaluation loop ────────────────────────────────────────────────────

def run_eval(model, processor, pairs, label="baseline"):
    """Run MME evaluation on all image pairs.

    Args:
        model: Loaded model.
        processor: Loaded processor.
        pairs: List of image-pair groups from load_mme_data().
        label: Condition label for logging.

    Returns:
        (results_dict, report_dict) — raw results and computed scores.
    """
    results = {}
    total_q = 0
    correct_q = 0
    t0 = time.time()

    for pi, pair in enumerate(pairs):
        qid = pair[0]["question_id"]
        category = pair[0]["category"]
        pair_results = []

        for q_item in pair:
            try:
                raw_output = generate_answer(
                    model, processor, q_item["image"], q_item["question"],
                )
            except torch.cuda.OutOfMemoryError:
                print(f"  [{pi}] OOM on {qid}, skipping")
                torch.cuda.empty_cache()
                raw_output = ""
            except Exception as e:
                print(f"  [{pi}] Error on {qid}: {e}")
                raw_output = ""

            extracted = extract_yes_no(raw_output)
            gt = q_item["answer"]

            pair_results.append({
                "question_id": qid,
                "question": q_item["question"],
                "gt_answer": gt,
                "raw_output": raw_output,
                "extracted": extracted,
                "correct": extracted == gt,
                "category": category,
            })

            total_q += 1
            if extracted == gt:
                correct_q += 1

        results[qid] = pair_results

        # Progress logging
        if (pi + 1) % 50 == 0 or (pi + 1) == len(pairs):
            elapsed = time.time() - t0
            rate = (pi + 1) / elapsed * 60 if elapsed > 0 else 0
            acc = correct_q / total_q * 100 if total_q > 0 else 0
            print(f"  [{label}] {pi+1}/{len(pairs)} images "
                  f"({rate:.0f} img/min), acc+={acc:.1f}%")

    report = compute_mme_scores(results)
    elapsed = time.time() - t0
    print(f"  [{label}] Done in {elapsed:.0f}s")
    print_scores(report, label)

    return results, report


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VIGIL MME Benchmark Evaluation (Qwen3-VL-2B-Thinking)",
    )
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to fine-tuned checkpoint. If omitted, "
                             "uses base Qwen3-VL-2B-Thinking.")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max number of image pairs to evaluate. "
                             "Each pair has 2 questions.")
    parser.add_argument("--compare", action="store_true",
                        help="Run both baseline and checkpoint, then compare.")
    parser.add_argument("--output-dir", type=str, default="lab/reports/mme",
                        help="Directory for results JSON and charts.")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16"],
                        help="Model dtype.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    # Load data
    pairs = load_mme_data(max_samples=args.max_samples)

    if args.compare:
        # ── Compare baseline vs checkpoint ──
        all_reports = []

        # 1) Baseline
        print("\n" + "=" * 70)
        print("CONDITION: baseline")
        print("=" * 70)
        model, processor = load_model(None, dtype=dtype)
        results_base, report_base = run_eval(model, processor, pairs, "baseline")
        all_reports.append(("baseline", report_base))

        # Save baseline
        base_file = output_dir / f"mme_baseline_{timestamp}.json"
        with open(base_file, "w") as f:
            json.dump({
                "label": "baseline",
                "report": report_base,
                "config": {"model": HF_ID, "max_samples": args.max_samples},
            }, f, indent=2, default=str)

        # Free memory
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # 2) Checkpoint
        if args.model_path and Path(args.model_path).exists():
            print("\n" + "=" * 70)
            print(f"CONDITION: checkpoint ({args.model_path})")
            print("=" * 70)
            model, processor = load_model(args.model_path, dtype=dtype)
            ckpt_label = Path(args.model_path).name
            results_ckpt, report_ckpt = run_eval(
                model, processor, pairs, ckpt_label,
            )
            all_reports.append((ckpt_label, report_ckpt))

            ckpt_file = output_dir / f"mme_{ckpt_label}_{timestamp}.json"
            with open(ckpt_file, "w") as f:
                json.dump({
                    "label": ckpt_label,
                    "report": report_ckpt,
                    "config": {"model": args.model_path,
                               "max_samples": args.max_samples},
                }, f, indent=2, default=str)

            del model
            gc.collect()
            torch.cuda.empty_cache()

        # Generate comparison charts
        generate_comparison_chart(
            all_reports, output_dir / f"mme_comparison_{timestamp}.png",
        )
        generate_subtask_chart(
            all_reports, output_dir / f"mme_subtasks_{timestamp}.png",
        )

        # Print comparison summary
        print(f"\n{'=' * 70}")
        print("COMPARISON SUMMARY")
        print(f"{'=' * 70}")
        print(f"{'Condition':<20} {'Perception':>12} {'Cognition':>12} {'Total':>12}")
        print("-" * 70)
        for label, report in all_reports:
            p = report["_Perception"]["score"]
            c = report["_Cognition"]["score"]
            t = report["_Total"]["score"]
            print(f"{label:<20} {p:>12} {c:>12} {t:>12}")

        # Save combined summary
        summary_file = output_dir / f"mme_summary_{timestamp}.json"
        with open(summary_file, "w") as f:
            json.dump({
                "timestamp": timestamp,
                "conditions": {
                    label: {
                        "Perception": report["_Perception"]["score"],
                        "Cognition": report["_Cognition"]["score"],
                        "Total": report["_Total"]["score"],
                    }
                    for label, report in all_reports
                },
            }, f, indent=2)
        print(f"\n[mme] Summary saved to {summary_file}")

    else:
        # ── Single condition ──
        label = "baseline" if not args.model_path else Path(args.model_path).name
        model, processor = load_model(args.model_path, dtype=dtype)
        results, report = run_eval(model, processor, pairs, label)

        # Save results
        result_file = output_dir / f"mme_{label}_{timestamp}.json"
        with open(result_file, "w") as f:
            json.dump({
                "label": label,
                "report": report,
                "config": {
                    "model": args.model_path or HF_ID,
                    "max_samples": args.max_samples,
                    "dtype": args.dtype,
                },
            }, f, indent=2, default=str)
        print(f"\n[mme] Results saved to {result_file}")

        # Generate single-condition chart
        generate_comparison_chart(
            [(label, report)],
            output_dir / f"mme_{label}_{timestamp}.png",
        )


if __name__ == "__main__":
    main()
