"""
VIGIL Blind Test Runner — the killer experiment.

Runs evaluation with real images vs black images, computes Gap metric.
Can compare multiple model configurations.

Usage:
    python scripts/run_blind_test.py --model qwen3_vl_2b --limit 500
    python scripts/run_blind_test.py --model qwen3_vl_2b --checkpoint checkpoints/grpo_qwen3_vl_2b
"""

import sys
import json
import argparse
import torch
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_registry import load_model, make_chat_prompt
from src.data_loader import load_pope
from src.blind_test import run_blind_test, save_blind_test_results, compare_blind_tests


@torch.no_grad()
def simple_generate(model_info, question, image, max_new_tokens=64):
    """Simple generation without steering."""
    inputs = make_chat_prompt(model_info, question, image)
    output_ids = model_info["model"].generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
    )
    input_len = inputs["input_ids"].shape[-1]
    new_ids = output_ids[0, input_len:]
    return model_info["tokenizer"].decode(new_ids, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="VIGIL Blind Test")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="lab/reports")
    parser.add_argument("--pope-split", type=str, default="adversarial")
    args = parser.parse_args()

    model_info = load_model(args.model)
    samples = load_pope(split=args.pope_split, limit=args.limit)

    def gen_fn(mi, q, img):
        return simple_generate(mi, q, img)

    result = run_blind_test(model_info, samples, gen_fn, max_samples=args.limit)
    result["checkpoint"] = args.checkpoint or "baseline"
    result["pope_split"] = args.pope_split

    save_blind_test_results(result, args.output_dir)

    # Print interpretation
    gap = result["gap"]
    print(f"\n{'='*50}")
    print(f"BLIND TEST INTERPRETATION:")
    if gap > 18:
        print(f"  Gap = {gap:.1f}pp — STRONG visual grounding (target zone)")
    elif gap > 12:
        print(f"  Gap = {gap:.1f}pp — Moderate visual grounding (baseline level)")
    elif gap > 5:
        print(f"  Gap = {gap:.1f}pp — WEAK visual grounding (partial blind reasoner)")
    else:
        print(f"  Gap = {gap:.1f}pp — BLIND REASONER (model ignores images)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
