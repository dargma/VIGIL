"""
VIGIL — Prepare mixed-format training dataset for GRPO/DPO.

Addresses binary VQA collapse by creating a balanced mix:
  ~40% open-ended (TextVQA)
  ~30% multiple-choice (A-OKVQA)
  ~30% short-answer (VQAv2 non-binary only — no yes/no)

Usage:
    python scripts/prepare_mixed_data.py --limit 5000 --seed 42
"""

import sys
import os
import argparse
import random
import json
import time
from pathlib import Path
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUTPUT_DIR = Path("data/training/mixed_grpo")


def parse_args():
    p = argparse.ArgumentParser(description="Prepare mixed GRPO dataset")
    p.add_argument("--limit", type=int, default=5000,
                   help="Total samples in final dataset")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    return p.parse_args()


def has_valid_image(sample: dict) -> bool:
    """Check if sample has a usable PIL image."""
    img = sample.get("image")
    if img is None:
        return False
    try:
        from PIL import Image
        if isinstance(img, Image.Image):
            # Quick sanity — try to access size
            _ = img.size
            return True
    except Exception:
        pass
    return False


def format_for_grpo(sample: dict) -> dict:
    """Convert a data_loader sample to GRPO-ready format."""
    q = sample["question"]
    return {
        "prompt": [{"role": "user", "content": q}],
        "image": sample["image"],
        "answer": sample["answer"],
        "type": sample["type"],
        "source": sample["source"],
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    t0 = time.time()
    output_dir = Path(args.output_dir)

    print(f"{'='*60}")
    print(f"VIGIL — Prepare Mixed GRPO Dataset")
    print(f"  limit={args.limit}, seed={args.seed}")
    print(f"  output={output_dir}")
    print(f"{'='*60}\n")

    # ---- Load raw data ----
    from src.data_loader import (
        load_vqav2_train, load_aokvqa_train, load_textvqa_train,
        load_pope, check_image_overlap, remove_overlapping,
    )

    # Load generous amounts; we will subsample later
    raw_limit = max(args.limit * 3, 30000)
    print("[1/5] Loading raw datasets...")
    vqav2_raw = load_vqav2_train(limit=raw_limit)
    aokvqa_raw = load_aokvqa_train(limit=raw_limit)
    textvqa_raw = load_textvqa_train(limit=raw_limit)

    # ---- POPE overlap check ----
    print("\n[2/5] Checking POPE image overlap...")
    pope_samples = load_pope("all", limit=None)
    overlap_ids = check_image_overlap(aokvqa_raw, pope_samples)
    if overlap_ids:
        aokvqa_raw = remove_overlapping(aokvqa_raw, overlap_ids)
    # Also check VQAv2 (uses COCO images too)
    overlap_ids_v2 = check_image_overlap(vqav2_raw, pope_samples)
    if overlap_ids_v2:
        vqav2_raw = remove_overlapping(vqav2_raw, overlap_ids_v2)

    # ---- Filter VQAv2: exclude binary yes/no ----
    print("\n[3/5] Filtering VQAv2 (removing yes/no answers)...")
    vqav2_before = len(vqav2_raw)
    vqav2_filtered = [
        s for s in vqav2_raw
        if s["answer"].strip().lower() not in ("yes", "no", "y", "n")
    ]
    vqav2_binary_removed = vqav2_before - len(vqav2_filtered)
    print(f"  VQAv2: {vqav2_before} -> {len(vqav2_filtered)} "
          f"(removed {vqav2_binary_removed} binary samples, "
          f"{vqav2_binary_removed/max(vqav2_before,1)*100:.1f}%)")

    # ---- Filter out samples without valid images ----
    print("\n[4/5] Filtering samples without valid images...")
    def filter_images(samples, name):
        before = len(samples)
        filtered = [s for s in samples if has_valid_image(s)]
        print(f"  {name}: {before} -> {len(filtered)} "
              f"(dropped {before - len(filtered)} without image)")
        return filtered

    textvqa = filter_images(textvqa_raw, "TextVQA")
    aokvqa = filter_images(aokvqa_raw, "A-OKVQA")
    vqav2 = filter_images(vqav2_filtered, "VQAv2")

    # ---- Compute target counts ----
    total = args.limit
    n_textvqa = int(total * 0.40)   # ~40% open-ended
    n_aokvqa = int(total * 0.30)    # ~30% multiple-choice
    n_vqav2 = total - n_textvqa - n_aokvqa  # ~30% short-answer (non-binary)

    # Clamp to available
    n_textvqa = min(n_textvqa, len(textvqa))
    n_aokvqa = min(n_aokvqa, len(aokvqa))
    n_vqav2 = min(n_vqav2, len(vqav2))

    # If some source is short, redistribute surplus to others
    actual_total = n_textvqa + n_aokvqa + n_vqav2
    if actual_total < total:
        deficit = total - actual_total
        # Try to fill from whichever has headroom
        for _ in range(deficit):
            if n_textvqa < len(textvqa):
                n_textvqa += 1
            elif n_vqav2 < len(vqav2):
                n_vqav2 += 1
            elif n_aokvqa < len(aokvqa):
                n_aokvqa += 1
            else:
                break

    print(f"\n  Target allocation:")
    print(f"    TextVQA (open-ended):    {n_textvqa}")
    print(f"    A-OKVQA (multiple-choice): {n_aokvqa}")
    print(f"    VQAv2 (short-answer):    {n_vqav2}")
    print(f"    Total:                   {n_textvqa + n_aokvqa + n_vqav2}")

    # ---- Sample and format ----
    random.shuffle(textvqa)
    random.shuffle(aokvqa)
    random.shuffle(vqav2)

    # Update type field for clarity
    for s in textvqa:
        s["type"] = "open_ended"
    for s in aokvqa:
        s["type"] = "mc"
    for s in vqav2:
        s["type"] = "short_answer"

    selected = (
        [format_for_grpo(s) for s in textvqa[:n_textvqa]]
        + [format_for_grpo(s) for s in aokvqa[:n_aokvqa]]
        + [format_for_grpo(s) for s in vqav2[:n_vqav2]]
    )
    random.shuffle(selected)

    # ---- Statistics ----
    print(f"\n[5/5] Final dataset statistics:")
    print(f"  Total samples: {len(selected)}")

    type_counts = Counter(s["type"] for s in selected)
    source_counts = Counter(s["source"] for s in selected)
    ans_lengths = [len(s["answer"].split()) for s in selected]

    print(f"\n  By type:")
    for t, c in sorted(type_counts.items()):
        print(f"    {t}: {c} ({c/len(selected)*100:.1f}%)")

    print(f"\n  By source:")
    for src, c in sorted(source_counts.items()):
        print(f"    {src}: {c} ({c/len(selected)*100:.1f}%)")

    print(f"\n  Answer length (words):")
    import numpy as np
    arr = np.array(ans_lengths)
    print(f"    mean={arr.mean():.1f}, median={np.median(arr):.1f}, "
          f"min={arr.min()}, max={arr.max()}, p90={np.percentile(arr,90):.0f}")

    # Verify zero binary in VQAv2 portion
    vqav2_binary_check = sum(
        1 for s in selected
        if s["source"] == "vqav2" and s["answer"] in ("yes", "no")
    )
    print(f"\n  VQAv2 binary answers remaining: {vqav2_binary_check} (should be 0)")

    # ---- Save as HuggingFace Dataset ----
    print(f"\n  Saving to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    from datasets import Dataset, Features, Value, Image as HFImage

    # Build records — images stored as PIL in the Arrow dataset
    records = []
    for s in selected:
        records.append({
            "prompt": json.dumps(s["prompt"]),  # serialize list-of-dicts
            "image": s["image"],
            "answer": s["answer"],
            "type": s["type"],
            "source": s["source"],
        })

    features = Features({
        "prompt": Value("string"),
        "image": HFImage(),
        "answer": Value("string"),
        "type": Value("string"),
        "source": Value("string"),
    })

    ds = Dataset.from_list(records, features=features)
    ds.save_to_disk(str(output_dir))
    print(f"  Saved {len(ds)} samples to {output_dir}")

    # Also save a lightweight JSON index (without images) for quick inspection
    index_path = output_dir / "index.json"
    index_records = [
        {"answer": s["answer"], "type": s["type"], "source": s["source"]}
        for s in selected
    ]
    with open(index_path, "w") as f:
        json.dump({
            "total": len(selected),
            "type_distribution": dict(type_counts),
            "source_distribution": dict(source_counts),
            "answer_length_stats": {
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "min": int(arr.min()),
                "max": int(arr.max()),
                "p90": float(np.percentile(arr, 90)),
            },
            "created": datetime.now().isoformat(),
            "seed": args.seed,
            "limit": args.limit,
        }, f, indent=2)
    print(f"  Saved index to {index_path}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s.")


if __name__ == "__main__":
    main()
