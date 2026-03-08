"""
VIGIL Official Evaluation — uses VLMEvalKit prompts, parsing, and metrics.

Evaluates Qwen3-VL-2B (baseline / steered / BoN+SFT checkpoint) on POPE.
Uses exact prompts and YOrN_Extraction from VLMEvalKit for publication-quality scores.

Usage:
    # Baseline
    python scripts/eval_official.py --condition baseline

    # Steered
    python scripts/eval_official.py --condition steered --alpha 5.0

    # BoN+SFT checkpoint
    python scripts/eval_official.py --condition bon_r1 \
        --model-path checkpoints/block2_bon/final

    # Blind test (black images)
    python scripts/eval_official.py --condition baseline --blind

    # All conditions (full comparison)
    python scripts/eval_official.py --all-conditions
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from datetime import datetime
from datasets import load_from_disk

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# VIGIL modules
from src.steerer import ActivationSteerer
from src.calibrator import CalibrationResult


# ─── VLMEvalKit functions (inlined to avoid broken import chain) ────────────
# Source: vlmeval/smp/misc.py:process_punctuation
# Source: vlmeval/dataset/utils/yorn.py:YOrN_Extraction
# These are the EXACT functions from VLMEvalKit — do not modify.

import re

def process_punctuation(inText):
    """VLMEvalKit: vlmeval/smp/misc.py:33"""
    outText = inText
    punct = [
        ';', r'/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-',
        '>', '<', '@', '`', ',', '?', '!'
    ]
    commaStrip = re.compile(r'(\d)(,)(\d)')
    periodStrip = re.compile(r'(?<!\d)\.(?!\d)')
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(
                commaStrip, inText) is not None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub('', outText, re.UNICODE)
    return outText


def YOrN_Extraction(output):
    """VLMEvalKit: vlmeval/dataset/utils/yorn.py:249"""
    s = output.lower()
    words = process_punctuation(s).split()
    if 'yes' in words and 'no' not in words:
        return 'Yes'
    if 'yes' not in words and 'no' in words:
        return 'No'
    return 'Unknown'


# ─── VLMEvalKit-standard prompts ───────────────────────────────────────────
# From vlmeval/vlm/qwen3_vl/prompt.py:_build_yorn_prompt
POPE_PROMPT_TEMPLATE = "{question} Please answer yes or no."


def pope_rating(records):
    """Compute POPE metrics exactly as VLMEvalKit does.

    Replicates vlmeval/dataset/utils/yorn.py:POPE_rating().
    Returns dict with overall + per-category acc, F1, precision, recall.
    """
    def cal_f1(y_true, y_pred):
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return f1, precision, recall

    df = pd.DataFrame(records)
    results = {}

    # Overall
    y_true = np.array([1 if a == "Yes" else 0 for a in df["answer"]])
    y_pred = np.array([1 if a == "Yes" else 0 for a in df["extracted"]])
    score = np.array([1 if a == e else 0 for a, e in zip(df["answer"], df["extracted"])])
    f1, prec, rec = cal_f1(y_true, y_pred)
    results["Overall"] = {
        "acc": float(np.mean(score) * 100),
        "f1": float(f1 * 100),
        "precision": float(prec * 100),
        "recall": float(rec * 100),
        "n": len(df),
        "n_unknown": int(sum(1 for e in df["extracted"] if e == "Unknown")),
    }

    # Per-category (random, popular, adversarial)
    if "category" in df.columns:
        for cat in sorted(df["category"].unique()):
            sub = df[df["category"] == cat]
            yt = np.array([1 if a == "Yes" else 0 for a in sub["answer"]])
            yp = np.array([1 if a == "Yes" else 0 for a in sub["extracted"]])
            sc = np.array([1 if a == e else 0 for a, e in zip(sub["answer"], sub["extracted"])])
            f1, prec, rec = cal_f1(yt, yp)
            results[cat] = {
                "acc": float(np.mean(sc) * 100),
                "f1": float(f1 * 100),
                "precision": float(prec * 100),
                "recall": float(rec * 100),
                "n": len(sub),
                "n_unknown": int(sum(1 for e in sub["extracted"] if e == "Unknown")),
            }

    return results


def load_qwen3vl(model_path=None, dtype=torch.bfloat16):
    """Load Qwen3-VL-2B, optionally from a fine-tuned checkpoint."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    hf_id = model_path or "Qwen/Qwen3-VL-2B-Instruct"
    print(f"[eval] Loading model from: {hf_id}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        hf_id, torch_dtype=dtype, device_map="auto",
    )
    # Use original processor always (tokenizer/chat template unchanged by SFT)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    model.eval()

    return model, processor


def generate_answer(model, processor, image, question, blind=False, max_new_tokens=64):
    """Generate answer using VLMEvalKit-standard prompt format.

    Matches Qwen3VLChat.generate_inner_transformers() flow:
    - apply_chat_template with user message containing image + text
    - process_vision_info for pixel values
    - model.generate with greedy decoding (temperature=0.01)
    """
    from qwen_vl_utils import process_vision_info

    # Use black image for blind test
    if blind:
        image = Image.new("RGB", image.size, (0, 0, 0))

    # Build message in VLMEvalKit format
    prompt_text = POPE_PROMPT_TEMPLATE.format(question=question)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # Apply chat template (same as VLMEvalKit)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images, videos, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )

    inputs = processor(
        text=[text],
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.01,  # VLMEvalKit default for Qwen3VL
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.0,
        )

    # Decode only new tokens
    input_len = inputs["input_ids"].shape[1]
    output_ids = generated_ids[0][input_len:]
    response = processor.tokenizer.decode(output_ids, skip_special_tokens=True)
    return response.strip()


def setup_steering(model, alpha=5.0):
    """Load calibration and install steering hooks."""
    cal_path = Path("checkpoints/calibration/qwen3_vl_2b")
    if not cal_path.exists():
        print(f"[eval] No calibration at {cal_path}, skipping steering")
        return None

    calibration = CalibrationResult.load(str(cal_path))

    # Build model_info dict for steerer
    model_info = {
        "get_layers_fn": lambda: model.model.language_model.layers,
        "get_lm_head_fn": lambda: model.lm_head,
        "get_norm_fn": lambda: model.model.language_model.norm,
        "num_heads": 16,  # Qwen3-VL-2B
        "head_dim": 128,
        "num_kv_heads": 8,
        "hidden_size": 2048,
        "gqa": True,
        "steer_layers_start": 4,  # Skip DeepStack layers 1-3
    }

    steerer = ActivationSteerer(
        model_info, calibration, steer_layers_start=4,
    )
    steerer.steer(alpha)
    print(f"[eval] Steering active with alpha={alpha}")
    return steerer


def eval_pope(
    model, processor, dataset, condition,
    blind=False, steerer=None, max_samples=None
):
    """Run POPE evaluation with VLMEvalKit-standard prompts and parsing."""
    records = []
    n = len(dataset)
    if max_samples:
        n = min(n, max_samples)

    print(f"\n[eval] POPE {condition} ({'blind' if blind else 'real'}) — {n} samples")

    for i in range(n):
        sample = dataset[i]
        question = sample["question"]
        gt_answer = sample["answer"].strip().capitalize()  # "Yes" or "No"
        category = sample.get("category", "unknown")
        image = sample["image"]

        try:
            raw_pred = generate_answer(
                model, processor, image, question, blind=blind
            )
        except Exception as e:
            print(f"  [{i}] ERROR: {e}")
            raw_pred = ""

        # Official VLMEvalKit parsing
        extracted = YOrN_Extraction(raw_pred)

        records.append({
            "index": i,
            "question": question,
            "answer": gt_answer,
            "prediction": raw_pred,
            "extracted": extracted,
            "category": category,
        })

        if (i + 1) % 100 == 0:
            interim = pope_rating(records)
            acc = interim["Overall"]["acc"]
            print(f"  [{i+1}/{n}] acc={acc:.1f}%")

    # Compute final metrics
    metrics = pope_rating(records)

    print(f"\n[eval] === {condition} {'(blind)' if blind else ''} Results ===")
    for split, m in metrics.items():
        print(f"  {split}: acc={m['acc']:.1f}%, F1={m['f1']:.1f}%, "
              f"P={m['precision']:.1f}%, R={m['recall']:.1f}%, "
              f"unknown={m['n_unknown']}/{m['n']}")

    return records, metrics


def run_all_conditions(args):
    """Run baseline, steered, bon_r1, + blind variants."""
    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("lab/reports/official_eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_from_disk("data/eval/pope")
    n = args.max_samples or len(dataset)

    conditions = [
        ("baseline", None, False, False),
        ("baseline_blind", None, True, False),
        ("steered", None, False, True),
        ("steered_blind", None, True, True),
    ]

    # Add BoN checkpoints if they exist
    bon_paths = [
        ("bon_r1", "checkpoints/block2_bon/final"),
        ("bon_r2", "checkpoints/block2_bon/round2"),
    ]
    for name, path in bon_paths:
        if Path(path).exists():
            conditions.append((name, path, False, False))
            conditions.append((f"{name}_blind", path, True, False))
            # Also steered variants
            conditions.append((f"{name}_steered", path, False, True))

    for cond_name, model_path, blind, steer in conditions:
        print(f"\n{'='*60}")
        print(f"Condition: {cond_name}")
        print(f"{'='*60}")

        # Load model
        model, processor = load_qwen3vl(model_path)
        steerer = None
        if steer:
            steerer = setup_steering(model, alpha=args.alpha)

        records, metrics = eval_pope(
            model, processor, dataset, cond_name,
            blind=blind, steerer=steerer, max_samples=n,
        )

        all_results[cond_name] = {
            "metrics": metrics,
            "config": {
                "model_path": model_path,
                "blind": blind,
                "steer": steer,
                "alpha": args.alpha if steer else None,
                "n_samples": len(records),
            },
        }

        # Save per-condition results
        cond_file = output_dir / f"{cond_name}_{timestamp}.json"
        with open(cond_file, "w") as f:
            json.dump({"records": records, "metrics": metrics}, f, indent=2)

        # Cleanup
        if steerer:
            steerer.cleanup()
        del model
        torch.cuda.empty_cache()

    # Save comparison summary
    summary = {
        "timestamp": timestamp,
        "conditions": {},
    }
    for cond_name, data in all_results.items():
        summary["conditions"][cond_name] = data["metrics"]["Overall"]

    # Compute blind test gaps
    for base in ["baseline", "steered", "bon_r1", "bon_r2", "bon_r1_steered"]:
        blind_key = f"{base}_blind"
        if base in summary["conditions"] and blind_key in summary["conditions"]:
            real_acc = summary["conditions"][base]["acc"]
            blind_acc = summary["conditions"][blind_key]["acc"]
            gap = real_acc - blind_acc
            summary["conditions"][base]["blind_gap"] = gap
            print(f"\n[gap] {base}: real={real_acc:.1f}%, blind={blind_acc:.1f}%, gap={gap:.1f}pp")

    summary_file = output_dir / f"comparison_{timestamp}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[eval] Summary saved to {summary_file}")

    # Print comparison table
    print_comparison_table(summary)

    return all_results


def print_comparison_table(summary):
    """Print publication-quality comparison table."""
    print(f"\n{'='*80}")
    print("POPE Official Evaluation (VLMEvalKit prompts + parsing)")
    print(f"{'='*80}")
    print(f"{'Condition':<25} {'Acc':>7} {'F1':>7} {'Prec':>7} {'Rec':>7} {'Gap':>7}")
    print("-" * 80)
    for cond, m in summary["conditions"].items():
        gap_str = f"{m.get('blind_gap', 0):.1f}pp" if "blind_gap" in m else "-"
        print(f"{cond:<25} {m['acc']:>6.1f}% {m['f1']:>6.1f}% "
              f"{m['precision']:>6.1f}% {m['recall']:>6.1f}% {gap_str:>7}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="VIGIL Official POPE Evaluation")
    parser.add_argument("--condition", type=str, default="baseline",
                        help="Condition: baseline, steered, bon_r1, bon_r2")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to fine-tuned checkpoint")
    parser.add_argument("--alpha", type=float, default=5.0,
                        help="Steering strength")
    parser.add_argument("--blind", action="store_true",
                        help="Use black images (blind test)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples")
    parser.add_argument("--all-conditions", action="store_true",
                        help="Run all conditions sequentially")
    parser.add_argument("--output-dir", type=str, default="lab/reports/official_eval",
                        help="Output directory for results")
    args = parser.parse_args()

    if args.all_conditions:
        run_all_conditions(args)
        return

    # Single condition
    dataset = load_from_disk("data/eval/pope")
    model, processor = load_qwen3vl(args.model_path)

    steerer = None
    if args.condition in ("steered",):
        steerer = setup_steering(model, alpha=args.alpha)

    records, metrics = eval_pope(
        model, processor, dataset, args.condition,
        blind=args.blind, steerer=steerer, max_samples=args.max_samples,
    )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"{args.condition}{'_blind' if args.blind else ''}_{ts}.json"
    with open(result_file, "w") as f:
        json.dump({"records": records, "metrics": metrics}, f, indent=2)
    print(f"\n[eval] Saved to {result_file}")

    if steerer:
        steerer.cleanup()


if __name__ == "__main__":
    main()
