"""
Phase 6c: Gated Head-LSR + Curriculum experiments.

Experiment 1: Gated Head-LSR only (15 steps)
Experiment 2: Curriculum only (30 steps)
Experiment 3: Gated + Curriculum combined (30 steps)

Usage:
    PYTHONUNBUFFERED=1 python -u scripts/run_phase6c.py 2>&1 | tee logs/phase6c_all.log
"""

import subprocess
import sys
import time
import json
from pathlib import Path
from datetime import datetime


def run(name, args, log_file):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*70}\n")

    cmd = [sys.executable, "-u", "scripts/phase6_head_mask_grpo.py"] + args
    t0 = time.time()
    with open(log_file, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
            lf.flush()
        proc.wait()

    elapsed = time.time() - t0
    ok = proc.returncode == 0
    print(f"\n  [{name}] {'OK' if ok else 'FAIL'} in {elapsed:.0f}s\n")
    return {"name": name, "ok": ok, "elapsed": round(elapsed)}


def main():
    results = []

    # Experiment 1: Gated Head-LSR only
    r1 = run("Exp1_Gated_HeadLSR", [
        "--output-dir", "checkpoints/phase6c/gated_only",
        "--steps", "15",
        "--alpha", "0.5",
        "--gdpo", "--vppo-mask",
        "--gated-head-lsr",
        "--lr", "2e-6",
        "--eval-every", "5",
        "--train-samples", "500",
        "--seed", "42",
    ], "logs/phase6c_gated_only.log")
    results.append(r1)

    # Experiment 2: Curriculum only (no gating)
    r2 = run("Exp2_Curriculum_Only", [
        "--output-dir", "checkpoints/phase6c/curriculum_only",
        "--steps", "30",
        "--alpha", "0.5",
        "--gdpo", "--vppo-mask",
        "--curriculum",
        "--curriculum-phases", "10,20,30",
        "--curriculum-thresholds", "100,200,999",
        "--lr", "2e-6",
        "--eval-every", "5",
        "--train-samples", "500",
        "--seed", "42",
    ], "logs/phase6c_curriculum_only.log")
    results.append(r2)

    # Experiment 3: Gated + Curriculum combined
    r3 = run("Exp3_Gated_Curriculum", [
        "--output-dir", "checkpoints/phase6c/gated_curriculum",
        "--steps", "30",
        "--alpha", "0.5",
        "--gdpo", "--vppo-mask",
        "--gated-head-lsr",
        "--curriculum",
        "--curriculum-phases", "10,20,30",
        "--curriculum-thresholds", "100,200,999",
        "--lr", "2e-6",
        "--eval-every", "5",
        "--train-samples", "500",
        "--seed", "42",
    ], "logs/phase6c_gated_curriculum.log")
    results.append(r3)

    # Summary
    print(f"\n{'='*70}")
    print("  PHASE 6C COMPLETE")
    print(f"{'='*70}")
    for r in results:
        print(f"  {r['name']:<35} {'OK' if r['ok'] else 'FAIL':<6} {r['elapsed']:>5}s")
    print(f"{'='*70}")

    out = Path("lab/reports/phase6c")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
