"""
VIGIL Calibration Script — run calibration on a target model.

Usage:
    python scripts/calibrate.py --model qwen3_vl_2b --output-dir checkpoints/calibration/qwen3_vl_2b
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_registry import load_model, make_chat_prompt
from src.calibrator import SteeringCalibrator
from src.data_loader import build_calibration_set


def process_sample(model_info, sample):
    """Convert sample to model inputs + ground truth string."""
    question = sample["question"]
    image = sample.get("image")
    inputs = make_chat_prompt(model_info, question, image)
    gt = sample["answer"]
    return inputs, gt


def main():
    parser = argparse.ArgumentParser(description="VIGIL Calibration")
    parser.add_argument("--model", type=str, required=True,
                        help="Model key from configs/models.yaml")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for calibration results")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--limit-per-source", type=int, default=500)
    args = parser.parse_args()

    # Load model
    import torch
    model_info = load_model(args.model, dtype=torch.float16)

    # Load calibration data
    samples = build_calibration_set(limit_per_source=args.limit_per_source)

    # Run calibration
    calibrator = SteeringCalibrator(
        model_info=model_info,
        top_k=args.top_k,
    )
    result = calibrator.calibrate(
        samples=samples,
        process_fn=process_sample,
        max_samples=args.max_samples,
    )

    # Save
    result.save(args.output_dir)
    print(f"\nCalibration complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
