"""
Session 14 Tasks 1-2: Phase 6 Round 2 + Alpha=0.0 Ablation
Waits for GPU to be free, then runs sequentially.

Usage:
    PYTHONUNBUFFERED=1 python -u scripts/run_tasks12.py 2>&1 | tee logs/session14_tasks12.log
"""

import subprocess
import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime


def gpu_mem_used_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return 99999


def wait_for_gpu(max_mem_mb=1000, poll_s=30, timeout_s=7200):
    """Wait until GPU memory drops below threshold."""
    t0 = time.time()
    while gpu_mem_used_mb() > max_mem_mb:
        elapsed = time.time() - t0
        if elapsed > timeout_s:
            print(f"[WARN] GPU not free after {timeout_s}s, proceeding anyway")
            break
        print(f"  GPU mem={gpu_mem_used_mb()}MB, waiting... ({elapsed:.0f}s)")
        time.sleep(poll_s)
    print(f"  GPU free (mem={gpu_mem_used_mb()}MB)")


def run_task(name, cmd, log_file):
    print(f"\n{'='*70}")
    print(f"  TASK: {name}")
    print(f"  TIME: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*70}\n")

    t0 = time.time()
    with open(log_file, "w") as lf:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
            lf.flush()
        proc.wait()

    elapsed = time.time() - t0
    status = "OK" if proc.returncode == 0 else f"FAIL (rc={proc.returncode})"
    print(f"\n  [{name}] {status} in {elapsed:.0f}s\n")
    return {"name": name, "status": status, "elapsed_s": round(elapsed)}


def main():
    print("[session14] Waiting for GPU to be free...")
    wait_for_gpu()

    results = []

    # Task 1: Phase 6 Round 2 from textvqa_v2 checkpoint
    r1 = run_task(
        "Phase6_Round2_GDPO_VPPO",
        [
            sys.executable, "-u", "scripts/phase6_head_mask_grpo.py",
            "--model-path", "checkpoints/phase6_head_mask/textvqa_v2/final",
            "--output-dir", "checkpoints/phase6_head_mask/round2_gdpo_vppo",
            "--steps", "30",
            "--alpha", "0.5",
            "--gdpo",
            "--vppo-mask",
            "--lr", "2e-6",
            "--eval-every", "5",
            "--train-samples", "500",
            "--seed", "43",
        ],
        "logs/phase6_round2_gdpo_vppo.log",
    )
    results.append(r1)

    # Task 2: Alpha=0.0 ablation
    r2 = run_task(
        "Alpha0_Ablation_GDPO_VPPO",
        [
            sys.executable, "-u", "scripts/phase6_head_mask_grpo.py",
            "--output-dir", "checkpoints/phase6_head_mask/alpha0_ablation",
            "--steps", "20",
            "--alpha", "0.0",
            "--gdpo",
            "--vppo-mask",
            "--lr", "2e-6",
            "--eval-every", "5",
            "--train-samples", "500",
            "--seed", "42",
        ],
        "logs/phase6_alpha0_ablation.log",
    )
    results.append(r2)

    # Save summary
    summary_file = Path("lab/reports/session14_tasks12.json")
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print("  TASKS 1-2 COMPLETE")
    print(f"{'='*70}")
    for r in results:
        print(f"  {r['name']:<35} {r['status']}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
