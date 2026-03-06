"""
VIGIL Dataset Downloader v2 — fixed configs for GQA, TextVQA, VQAv2, MMMU.
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
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            print(f"\n{'='*60}")
            print(f"[{name}] Downloading (attempt {attempt+1}/{max_retries})...")
            ds = load_fn()
            elapsed = time.time() - t0
            n_rows = len(ds) if hasattr(ds, '__len__') else "?"
            print(f"[{name}] Done: {n_rows} rows in {elapsed:.1f}s")
            STATUS[name] = {"status": "ok", "rows": str(n_rows), "time": f"{elapsed:.1f}s"}
            return ds
        except Exception as e:
            print(f"[{name}] Attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1:
                STATUS[name] = {"status": "FAILED", "error": str(e)}
                traceback.print_exc()
                return None
            time.sleep(3)


def save_to_disk(ds, name, path):
    out = DATA_DIR / path
    out.mkdir(parents=True, exist_ok=True)
    try:
        ds.save_to_disk(str(out))
        print(f"[{name}] Saved to {out}")
    except Exception as e:
        print(f"[{name}] Arrow save failed: {e}, trying JSONL...")
        try:
            jsonl_path = out / "data.jsonl"
            with open(jsonl_path, "w") as f:
                for row in ds:
                    record = {}
                    for k, v in row.items():
                        if k == "image" or isinstance(v, bytes):
                            continue
                        try:
                            json.dumps(v)
                            record[k] = v
                        except (TypeError, ValueError):
                            record[k] = str(v)
                    f.write(json.dumps(record) + "\n")
            print(f"[{name}] Saved JSONL to {jsonl_path}")
        except Exception as e2:
            print(f"[{name}] JSONL fallback also failed: {e2}")


def main():
    print("VIGIL Dataset Downloader v2 (fixing failed downloads)")
    print(f"Target: {DATA_DIR}\n")

    # ============================================================
    # 1. GQA — needs config "testdev_balanced_instructions"
    # ============================================================
    gqa = download_with_retry(
        "GQA-balanced-val",
        lambda: load_dataset("lmms-lab/GQA", "testdev_balanced_instructions", split="testdev"),
    )
    if gqa:
        save_to_disk(gqa, "GQA-balanced-val", "calibration/gqa_balanced_val")

    # ============================================================
    # 2. TextVQA — use lmms-lab version
    # ============================================================
    textvqa_val = download_with_retry(
        "TextVQA-val",
        lambda: load_dataset("lmms-lab/textvqa", split="validation"),
    )
    if textvqa_val:
        save_to_disk(textvqa_val, "TextVQA-val", "calibration/textvqa_val")

    # TextVQA train
    textvqa_train = download_with_retry(
        "TextVQA-train",
        lambda: load_dataset("lmms-lab/textvqa", split="train"),
    )
    if textvqa_train:
        save_to_disk(textvqa_train, "TextVQA-train", "training/textvqa_train")

    # ============================================================
    # 3. VQAv2 — try lmms-lab version or merve version
    # ============================================================
    vqav2 = download_with_retry(
        "VQAv2-train",
        lambda: load_dataset("merve/vqav2-small", split="validation"),
    )
    if vqav2:
        save_to_disk(vqav2, "VQAv2-train", "training/vqav2_train")
    else:
        # Fallback: try another source
        vqav2 = download_with_retry(
            "VQAv2-train-alt",
            lambda: load_dataset("lmms-lab/VQAv2", "vqav2_val", split="validation"),
        )
        if vqav2:
            save_to_disk(vqav2, "VQAv2-train-alt", "training/vqav2_train")

    # ============================================================
    # 4. MMMU — download all subjects
    # ============================================================
    mmmu_subjects = [
        'Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art',
        'Art_Theory', 'Basic_Medical_Science', 'Biology', 'Chemistry',
        'Clinical_Medicine', 'Computer_Science', 'Design',
        'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics',
        'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature',
        'Manage', 'Marketing', 'Materials', 'Math', 'Mechanical_Engineering',
        'Music', 'Pharmacy', 'Physics', 'Psychology', 'Public_Health', 'Sociology',
    ]

    print(f"\n{'='*60}")
    print(f"[MMMU] Downloading {len(mmmu_subjects)} subjects...")
    all_mmmu = []
    for subj in mmmu_subjects:
        try:
            ds = load_dataset("MMMU/MMMU", subj, split="validation")
            all_mmmu.append(ds)
            print(f"  {subj}: {len(ds)} samples")
        except Exception as e:
            print(f"  {subj}: FAILED ({e})")

    if all_mmmu:
        from datasets import concatenate_datasets
        mmmu_full = concatenate_datasets(all_mmmu)
        print(f"[MMMU] Total: {len(mmmu_full)} samples across {len(all_mmmu)} subjects")
        save_to_disk(mmmu_full, "MMMU-full", "eval/mmmu")
        STATUS["MMMU-full"] = {"status": "ok", "rows": len(mmmu_full)}
    else:
        STATUS["MMMU-full"] = {"status": "FAILED"}

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*60}")
    print("DOWNLOAD v2 SUMMARY")
    print(f"{'='*60}")
    for name, info in STATUS.items():
        status = info["status"]
        rows = info.get("rows", "?")
        icon = "OK" if status == "ok" else "FAIL"
        print(f"  [{icon}] {name}: {rows} rows")

    # Merge with existing status
    existing = {}
    status_file = DATA_DIR / "download_status.json"
    if status_file.exists():
        with open(status_file) as f:
            existing = json.load(f)
    existing.update(STATUS)
    with open(status_file, "w") as f:
        json.dump(existing, f, indent=2)

    os.system(f"du -sh {DATA_DIR}")
    os.system(f"du -sh {DATA_DIR}/*")


if __name__ == "__main__":
    main()
