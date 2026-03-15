"""
Session 14 Orchestrator: Run 3 GPU experiments sequentially.

Task 1: Phase 6 Round 2 — GDPO+VPPO from textvqa_v2 checkpoint (30 steps)
Task 2: Alpha=0.0 ablation — GDPO+VPPO without head weighting (20 steps)
Task 3: MME eval — baseline vs Phase 2 R4 best checkpoint

Usage:
    PYTHONUNBUFFERED=1 python -u scripts/run_all_session14.py 2>&1 | tee logs/session14_all.log
"""

import subprocess
import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

RESULTS_FILE = Path("lab/reports/session14_results.json")
RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)


def run_task(name, cmd, log_file):
    """Run a task, stream output, return success/failure."""
    print(f"\n{'='*70}")
    print(f"  TASK: {name}")
    print(f"  CMD:  {' '.join(cmd)}")
    print(f"  LOG:  {log_file}")
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
    return {
        "name": name,
        "status": status,
        "elapsed_s": round(elapsed),
        "returncode": proc.returncode,
        "log": str(log_file),
    }


def main():
    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Task 1: Phase 6 Round 2 from textvqa_v2 checkpoint ──
    task1 = run_task(
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
        LOG_DIR / "phase6_round2_gdpo_vppo.log",
    )
    all_results.append(task1)

    # ── Task 2: Alpha=0.0 ablation (no head weighting) ──
    task2 = run_task(
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
        LOG_DIR / "phase6_alpha0_ablation.log",
    )
    all_results.append(task2)

    # ── Task 3: MME eval (baseline vs best checkpoint) ──
    task3 = run_task(
        "MME_Eval_Compare",
        [
            sys.executable, "-u", "scripts/eval_mme.py",
            "--compare",
            "--model-path", "checkpoints/phase2_grpo_lsr/round4/best",
            "--output-dir", "lab/reports/mme",
        ],
        LOG_DIR / "mme_eval_compare.log",
    )
    all_results.append(task3)

    # ── Summary ──
    print(f"\n{'='*70}")
    print("  SESSION 14 — ALL TASKS COMPLETE")
    print(f"{'='*70}")
    for r in all_results:
        print(f"  {r['name']:<35} {r['status']:<15} {r['elapsed_s']:>5}s")
    print(f"{'='*70}\n")

    # Save results
    summary = {
        "timestamp": timestamp,
        "tasks": all_results,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
