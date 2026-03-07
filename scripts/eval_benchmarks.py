"""
VIGIL Evaluation Script — run benchmarks across eval modes.

Usage:
    python scripts/eval_benchmarks.py --model qwen3_vl_2b --calibration-dir ... --mode greedy_baseline
    python scripts/eval_benchmarks.py --model qwen3_vl_2b --calibration-dir ... --mode steered_only --alpha 1.0
    python scripts/eval_benchmarks.py --model qwen3_vl_2b --calibration-dir ... --mode blind_test
"""

import sys
import json
import argparse
import torch
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_registry import load_model, make_chat_prompt
from src.calibrator import CalibrationResult
from src.steerer import ActivationSteerer, AgreementMonitor
from src.data_loader import load_pope
from src.blind_test import run_blind_test, save_blind_test_results


@torch.no_grad()
def generate_answer(model_info, question, image, steerer=None, steer_mode="uniform"):
    """Generate an answer, optionally with steering.

    Args:
        steer_mode: "uniform" (same alpha for all heads) or "proportional" (by Cohen's d)
    """
    inputs = make_chat_prompt(model_info, question, image)
    model = model_info["model"]

    if steerer:
        if steer_mode == "proportional":
            steerer.steer_proportional(global_alpha=1.0)
        else:
            steerer.steer(alpha=1.0)

    # Generate
    input_ids = inputs.get("input_ids")
    gen_kwargs = {k: v for k, v in inputs.items()}
    gen_kwargs["max_new_tokens"] = 64
    gen_kwargs["do_sample"] = False
    output_ids = model.generate(**gen_kwargs)

    if steerer:
        steerer.release()

    # Decode only the new tokens
    new_ids = output_ids[0, input_ids.shape[-1]:]
    answer = model_info["tokenizer"].decode(new_ids, skip_special_tokens=True)
    return answer.strip()


def eval_pope(model_info, split, steerer=None, steer_mode="uniform", limit=500):
    """Evaluate on POPE benchmark."""
    samples = load_pope(split=split, limit=limit)
    correct = 0
    total = 0

    for i, sample in enumerate(samples):
        try:
            pred = generate_answer(
                model_info, sample["question"], sample.get("image"),
                steerer, steer_mode,
            )
            gt = sample["answer"]
            pred_yn = "yes" if "yes" in pred.lower()[:10] else "no"
            if pred_yn == gt:
                correct += 1
            total += 1

            if (i + 1) % 100 == 0:
                print(f"  POPE-{split} [{i+1}/{len(samples)}]: "
                      f"acc={correct/total*100:.1f}%")
        except Exception as e:
            print(f"  Skip {i}: {e}")

    acc = correct / total * 100 if total > 0 else 0
    print(f"  POPE-{split}: {acc:.1f}% ({correct}/{total})")
    return {"benchmark": f"pope_{split}", "accuracy": acc, "correct": correct, "total": total}


def main():
    parser = argparse.ArgumentParser(description="VIGIL Evaluation")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--calibration-dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Trained model checkpoint (for grpo/dapo modes)")
    parser.add_argument("--mode", type=str, default="greedy_baseline",
                        choices=["greedy_baseline", "steered_only", "steered_proportional",
                                 "grpo_steered", "dapo_steered", "blind_test"])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="lab/reports")
    args = parser.parse_args()

    # Load model
    model_info = load_model(args.model, dtype=torch.float16)

    # Setup steering if needed
    steerer = None
    agreement_monitor = None
    if args.mode in ("steered_only", "steered_proportional", "grpo_steered", "dapo_steered") and args.calibration_dir:
        calibration = CalibrationResult.load(args.calibration_dir)
        steer_start = model_info.get("steer_layers_start", 0)
        steerer = ActivationSteerer(model_info, calibration, steer_start)
        agreement_monitor = AgreementMonitor(model_info)

    # Run evaluation
    results = {
        "model": args.model,
        "mode": args.mode,
        "timestamp": datetime.now().isoformat(),
        "benchmarks": {},
    }

    if args.mode == "blind_test":
        if not args.calibration_dir:
            print("Blind test doesn't require calibration, using baseline model")
        pope_samples = load_pope(split="adversarial", limit=args.limit)

        def gen_fn(mi, q, img):
            return generate_answer(mi, q, img)

        bt_result = run_blind_test(model_info, pope_samples, gen_fn, max_samples=args.limit)
        save_blind_test_results(bt_result, args.output_dir)
        results["benchmarks"]["blind_test"] = bt_result
    else:
        steer_mode = "proportional" if args.mode == "steered_proportional" else "uniform"
        for split in ["random", "popular", "adversarial"]:
            r = eval_pope(model_info, split, steerer, steer_mode, limit=args.limit)
            results["benchmarks"][f"pope_{split}"] = r

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"eval_{args.model}_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {fname}")

    # Cleanup
    if steerer:
        steerer.cleanup()


if __name__ == "__main__":
    main()
