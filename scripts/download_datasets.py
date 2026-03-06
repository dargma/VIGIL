"""
VIGIL Dataset Downloader — download and cache all required datasets to Drive.

Datasets:
  Calibration: GQA balanced val, TextVQA val
  Training: VQAv2 train, A-OKVQA train, TextVQA train
  Eval: POPE, MMBench, MME, MMMU
"""

import os
import json
import time
import traceback
from pathlib import Path
from datasets import load_dataset

DATA_DIR = Path("/content/drive/MyDrive/VIGIL/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

STATUS = {}


def download_with_retry(name, load_fn, max_retries=3):
    """Download dataset with retry on failure."""
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            print(f"\n{'='*60}")
            print(f"[{name}] Downloading (attempt {attempt+1}/{max_retries})...")
            ds = load_fn()
            elapsed = time.time() - t0

            # Get size info
            if hasattr(ds, 'num_rows'):
                n_rows = ds.num_rows
            elif hasattr(ds, '__len__'):
                n_rows = len(ds)
            else:
                n_rows = "streaming"

            print(f"[{name}] Done: {n_rows} rows in {elapsed:.1f}s")
            STATUS[name] = {"status": "ok", "rows": str(n_rows), "time": f"{elapsed:.1f}s"}
            return ds
        except Exception as e:
            print(f"[{name}] Attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1:
                STATUS[name] = {"status": "FAILED", "error": str(e)}
                traceback.print_exc()
                return None
            time.sleep(5)


def save_to_disk(ds, name, path):
    """Save dataset to disk in Arrow format."""
    out = DATA_DIR / path
    out.mkdir(parents=True, exist_ok=True)
    try:
        ds.save_to_disk(str(out))
        print(f"[{name}] Saved to {out}")
    except Exception as e:
        print(f"[{name}] Failed to save: {e}")
        # Fallback: save as JSONL (without images)
        try:
            jsonl_path = out / "data.jsonl"
            with open(jsonl_path, "w") as f:
                for row in ds:
                    # Skip image columns for JSONL fallback
                    record = {k: v for k, v in row.items() if not isinstance(v, bytes) and k != "image"}
                    f.write(json.dumps(record, default=str) + "\n")
            print(f"[{name}] Fallback saved as JSONL to {jsonl_path}")
        except Exception as e2:
            print(f"[{name}] Fallback also failed: {e2}")


def main():
    print("VIGIL Dataset Downloader")
    print(f"Target: {DATA_DIR}")
    print(f"Disk free: check df -h")

    # ============================================================
    # 1. CALIBRATION DATASETS
    # ============================================================

    # GQA balanced val
    gqa = download_with_retry(
        "GQA-balanced-val",
        lambda: load_dataset("lmms-lab/GQA", split="testdev_balanced"),
    )
    if gqa:
        save_to_disk(gqa, "GQA-balanced-val", "calibration/gqa_balanced_val")

    # TextVQA val
    textvqa_val = download_with_retry(
        "TextVQA-val",
        lambda: load_dataset("textvqa", split="validation"),
    )
    if textvqa_val:
        save_to_disk(textvqa_val, "TextVQA-val", "calibration/textvqa_val")

    # ============================================================
    # 2. TRAINING DATASETS
    # ============================================================

    # VQAv2 train (large — stream and take first 20K)
    print(f"\n{'='*60}")
    print("[VQAv2-train] Downloading via streaming (20K samples)...")
    try:
        t0 = time.time()
        ds_stream = load_dataset("HuggingFaceM4/VQAv2", split="train", streaming=True)
        records = []
        for i, row in enumerate(ds_stream):
            records.append({
                "question": row.get("question", ""),
                "answers": row.get("answers", []),
                "image_id": row.get("image_id", ""),
                "question_id": row.get("question_id", ""),
            })
            if len(records) >= 20000:
                break
            if (i + 1) % 5000 == 0:
                print(f"  ... {i+1} samples collected")

        # Save as JSONL
        out = DATA_DIR / "training" / "vqav2_train"
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "data.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r, default=str) + "\n")
        elapsed = time.time() - t0
        print(f"[VQAv2-train] Done: {len(records)} rows in {elapsed:.1f}s (JSONL, no images)")
        STATUS["VQAv2-train"] = {"status": "ok", "rows": len(records), "time": f"{elapsed:.1f}s", "note": "JSONL no images"}
    except Exception as e:
        print(f"[VQAv2-train] FAILED: {e}")
        STATUS["VQAv2-train"] = {"status": "FAILED", "error": str(e)}

    # A-OKVQA train
    aokvqa = download_with_retry(
        "A-OKVQA-train",
        lambda: load_dataset("HuggingFaceM4/A-OKVQA", split="train"),
    )
    if aokvqa:
        save_to_disk(aokvqa, "A-OKVQA-train", "training/aokvqa_train")

    # TextVQA train
    textvqa_train = download_with_retry(
        "TextVQA-train",
        lambda: load_dataset("textvqa", split="train"),
    )
    if textvqa_train:
        save_to_disk(textvqa_train, "TextVQA-train", "training/textvqa_train")

    # ============================================================
    # 3. EVALUATION DATASETS
    # ============================================================

    # POPE
    pope = download_with_retry(
        "POPE",
        lambda: load_dataset("lmms-lab/POPE", split="test"),
    )
    if pope:
        save_to_disk(pope, "POPE", "eval/pope")

    # MMBench
    mmbench = download_with_retry(
        "MMBench",
        lambda: load_dataset("lmms-lab/MMBench", "en", split="dev"),
    )
    if mmbench:
        save_to_disk(mmbench, "MMBench", "eval/mmbench")

    # MME
    mme = download_with_retry(
        "MME",
        lambda: load_dataset("lmms-lab/MME", split="test"),
    )
    if mme:
        save_to_disk(mme, "MME", "eval/mme")

    # MMMU
    mmmu = download_with_retry(
        "MMMU",
        lambda: load_dataset("MMMU/MMMU", "Accounting", split="validation"),
    )
    if mmmu:
        save_to_disk(mmmu, "MMMU-sample", "eval/mmmu_sample")

    # Full MMMU (all subjects)
    try:
        print(f"\n{'='*60}")
        print("[MMMU-full] Downloading all subjects...")
        t0 = time.time()
        mmmu_full = load_dataset("MMMU/MMMU", split="validation")
        elapsed = time.time() - t0
        n = len(mmmu_full) if hasattr(mmmu_full, '__len__') else "?"
        print(f"[MMMU-full] Done: {n} rows in {elapsed:.1f}s")
        save_to_disk(mmmu_full, "MMMU-full", "eval/mmmu")
        STATUS["MMMU-full"] = {"status": "ok", "rows": str(n), "time": f"{elapsed:.1f}s"}
    except Exception as e:
        print(f"[MMMU-full] Failed (single subject already saved): {e}")
        STATUS["MMMU-full"] = {"status": "partial", "error": str(e)}

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    for name, info in STATUS.items():
        status = info["status"]
        rows = info.get("rows", "?")
        time_s = info.get("time", "?")
        icon = "OK" if status == "ok" else ("PARTIAL" if status == "partial" else "FAIL")
        print(f"  [{icon}] {name}: {rows} rows ({time_s})")

    # Save status
    with open(DATA_DIR / "download_status.json", "w") as f:
        json.dump(STATUS, f, indent=2)
    print(f"\nStatus saved to {DATA_DIR / 'download_status.json'}")

    # Disk usage
    os.system(f"du -sh {DATA_DIR}")


if __name__ == "__main__":
    main()
