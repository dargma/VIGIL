"""
VIGIL GRPO/DAPO Training Script.

Usage:
    python scripts/train_grpo.py --model qwen3_vl_2b --calibration-dir checkpoints/calibration/qwen3_vl_2b
    python scripts/train_grpo.py --model qwen3_vl_2b --calibration-dir ... --dapo  # DAPO mode
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_registry import load_model
from src.calibrator import CalibrationResult
from src.data_loader import build_training_set, load_pope
from src.trainer import setup_grpo_training, setup_dapo_training, load_training_config


def main():
    parser = argparse.ArgumentParser(description="VIGIL GRPO/DAPO Training")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--calibration-dir", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dapo", action="store_true", help="Use DAPO instead of GRPO")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    # Load model
    model_info = load_model(args.model)

    # Load calibration
    calibration = CalibrationResult.load(args.calibration_dir)

    # Build training set (with POPE overlap check)
    pope_samples = load_pope(split="all", limit=3000)
    train_samples = build_training_set(pope_samples=pope_samples, limit=args.train_limit)

    # Load config
    config = load_training_config(args.config)

    # Setup trainer
    if args.dapo:
        setup = setup_dapo_training(model_info, calibration, train_samples, config)
        print("[VIGIL] Starting DAPO training...")
    else:
        setup = setup_grpo_training(model_info, calibration, train_samples, config)
        print("[VIGIL] Starting GRPO training...")

    trainer = setup["trainer"]
    reward_fn = setup["reward_fn"]

    try:
        trainer.train(resume_from_checkpoint=args.resume)
    except KeyboardInterrupt:
        print("\n[VIGIL] Training interrupted. Saving checkpoint...")
        trainer.save_model()
    finally:
        reward_fn.cleanup()
        stats = reward_fn.get_reward_stats()
        print(f"\n[VIGIL] Reward stats: {stats}")


if __name__ == "__main__":
    main()
