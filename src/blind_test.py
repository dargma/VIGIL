"""
VIGIL Blind Test — the killer experiment.

Replace all test images with black images. Measure Gap = acc(real) - acc(black).
- Baseline gap ~16
- R_correct-only GRPO gap ~7 (blind reasoner)
- VIGIL full reward gap target ~19 (more image-dependent)
"""

import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


def make_black_image(size=(224, 224)):
    return Image.new("RGB", size, (0, 0, 0))


@torch.no_grad()
def run_blind_test(
    model_info: Dict[str, Any],
    samples: List[Dict],
    generate_fn,
    image_size: tuple = (224, 224),
    max_samples: int = 500,
) -> Dict[str, Any]:
    """Run blind test evaluation.

    Args:
        model_info: loaded model info dict
        samples: list of {question, answer, image, ...}
        generate_fn: callable(model_info, question, image) -> prediction_str
        image_size: size for black image
        max_samples: max samples to evaluate

    Returns:
        dict with acc_real, acc_black, gap, per_sample results
    """
    model = model_info["model"]
    model.eval()
    black_image = make_black_image(image_size)

    results_real = []
    results_black = []

    for i, sample in enumerate(samples[:max_samples]):
        try:
            question = sample["question"]
            gt = sample["answer"].strip().lower()
            image = sample.get("image")

            # Real image prediction
            pred_real = generate_fn(model_info, question, image).strip().lower()
            correct_real = _check_correct(pred_real, gt, sample.get("type", "yesno"))

            # Black image prediction
            pred_black = generate_fn(model_info, question, black_image).strip().lower()
            correct_black = _check_correct(pred_black, gt, sample.get("type", "yesno"))

            results_real.append(correct_real)
            results_black.append(correct_black)

            if (i + 1) % 50 == 0:
                acc_r = np.mean(results_real) * 100
                acc_b = np.mean(results_black) * 100
                print(f"[blind_test] {i+1}/{min(len(samples), max_samples)}: "
                      f"real={acc_r:.1f}%, black={acc_b:.1f}%, gap={acc_r - acc_b:.1f}pp")

        except Exception as e:
            print(f"[blind_test] Skip sample {i}: {e}")
            continue

    acc_real = np.mean(results_real) * 100 if results_real else 0.0
    acc_black = np.mean(results_black) * 100 if results_black else 0.0
    gap = acc_real - acc_black

    summary = {
        "acc_real": float(acc_real),
        "acc_black": float(acc_black),
        "gap": float(gap),
        "n_samples": len(results_real),
        "model": model_info.get("model_key", "unknown"),
        "timestamp": datetime.now().isoformat(),
    }

    print(f"\n[blind_test] RESULTS for {summary['model']}:")
    print(f"  Accuracy (real):  {acc_real:.1f}%")
    print(f"  Accuracy (black): {acc_black:.1f}%")
    print(f"  Gap:              {gap:.1f}pp")
    print(f"  N samples:        {len(results_real)}")

    return summary


def _check_correct(prediction: str, ground_truth: str, q_type: str = "yesno") -> bool:
    pred = prediction.strip().lower()
    gt = ground_truth.strip().lower()

    if q_type == "yesno":
        pred_yn = "yes" if "yes" in pred[:10] else ("no" if "no" in pred[:10] else "")
        return pred_yn == gt

    elif q_type == "mc":
        return pred[:1] == gt[:1] if pred and gt else False

    else:
        return pred.startswith(gt[:5]) if gt else False


def save_blind_test_results(results: Dict[str, Any], output_dir: str):
    """Save blind test results to JSON."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_key = results.get("model", "unknown")
    fname = out / f"blind_test_{model_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[blind_test] Saved results to {fname}")


def compare_blind_tests(results_list: List[Dict[str, Any]]) -> str:
    """Generate comparison table from multiple blind test results."""
    lines = [
        "| Model/Config | Acc (Real) | Acc (Black) | Gap | N |",
        "|-------------|-----------|------------|-----|---|",
    ]
    for r in results_list:
        lines.append(
            f"| {r.get('model', '?')} | {r['acc_real']:.1f}% | "
            f"{r['acc_black']:.1f}% | {r['gap']:.1f}pp | {r['n_samples']} |"
        )
    return "\n".join(lines)
