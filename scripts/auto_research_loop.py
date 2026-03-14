"""
Auto-Research Loop for VIGIL

Runs experiments sequentially, logs results, keeps improvements.
Designed to run overnight without human intervention.

Experiment queue:
  1. Phase 6b GDPO+VPPO (30 steps) — currently running
  2. Phase 7 BoN+SFT on TextVQA with head scoring (500 samples)
  3. Phase 7 Round 2 from best checkpoint
  4. Multi-benchmark eval (POPE 300, TextVQA 75, Blind 50)

Usage:
    python scripts/auto_research_loop.py 2>&1 | tee logs/auto_research.log
"""

import os, sys, json, time, subprocess
from pathlib import Path
from datetime import datetime

PROJECT = Path(__file__).parent.parent
RESULTS_TSV = PROJECT / "lab" / "results.tsv"
JOURNAL = PROJECT / "lab" / "RESEARCH_JOURNAL.md"


def log_result(experiment, metrics, status="completed"):
    """Append to results.tsv (machine-parseable experiment log)."""
    if not RESULTS_TSV.exists():
        with open(RESULTS_TSV, "w") as f:
            f.write("timestamp\texperiment\ttextvqa\tpope\tgap\tstatus\tnotes\n")

    tvqa = metrics.get("textvqa", 0)
    pope = metrics.get("pope", 0)
    gap = metrics.get("gap", 0)
    notes = metrics.get("notes", "")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    with open(RESULTS_TSV, "a") as f:
        f.write(f"{ts}\t{experiment}\t{tvqa:.3f}\t{pope:.3f}\t{gap:.3f}\t{status}\t{notes}\n")
    print(f"[log] {experiment}: tvqa={tvqa:.1%} pope={pope:.1%} gap={gap:.1%} [{status}]")


def run_experiment(name, cmd, timeout_min=120):
    """Run an experiment with timeout, return exit code."""
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {name}")
    print(f"  CMD: {cmd[:100]}...")
    print(f"  Timeout: {timeout_min} min")
    print(f"  Start: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*70}\n")

    log_file = PROJECT / "logs" / f"{name}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            cmd, shell=True, timeout=timeout_min * 60,
            cwd=str(PROJECT),
            stdout=open(log_file, "w"),
            stderr=subprocess.STDOUT,
        )
        return result.returncode
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {name} exceeded {timeout_min} min")
        return -1
    except Exception as e:
        print(f"  [ERROR] {name}: {e}")
        return -2


def parse_results(log_file):
    """Parse final metrics from a log file."""
    metrics = {"textvqa": 0, "pope": 0, "gap": 0}
    try:
        with open(log_file) as f:
            lines = f.readlines()
        for line in reversed(lines):
            if "TextVQA:" in line and "→" in line:
                # Parse "TextVQA: 72.7% → 75.0%"
                parts = line.split("→")
                if len(parts) >= 2:
                    val = parts[1].strip().split("%")[0].split("(")[0].strip()
                    metrics["textvqa"] = float(val) / 100
            if "POPE:" in line and "→" in line:
                parts = line.split("→")
                if len(parts) >= 2:
                    val = parts[1].strip().split("%")[0].split("(")[0].strip()
                    metrics["pope"] = float(val) / 100
            if "Gap:" in line and "→" in line:
                parts = line.split("→")
                if len(parts) >= 2:
                    val = parts[1].strip().split("%")[0].split("(")[0].strip()
                    metrics["gap"] = float(val) / 100
            # Also try "TextVQA=X% POPE=Y% Gap=Z%"
            if "TextVQA=" in line and "POPE=" in line:
                import re
                m_tvqa = re.search(r'TextVQA=(\d+\.?\d*)%', line)
                m_pope = re.search(r'POPE=(\d+\.?\d*)%', line)
                m_gap = re.search(r'Gap=(\d+\.?\d*)%', line)
                if m_tvqa: metrics["textvqa"] = float(m_tvqa.group(1)) / 100
                if m_pope: metrics["pope"] = float(m_pope.group(1)) / 100
                if m_gap: metrics["gap"] = float(m_gap.group(1)) / 100
    except Exception:
        pass
    return metrics


def wait_for_gpu(check_interval=30, max_wait=600):
    """Wait until GPU has < 1GB used."""
    import subprocess
    waited = 0
    while waited < max_wait:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                text=True)
            used = int(out.strip().split("\n")[0])
            if used < 1000:
                return True
        except Exception:
            pass
        time.sleep(check_interval)
        waited += check_interval
    return False


def main():
    print(f"\n{'#'*70}")
    print(f"  VIGIL AUTO-RESEARCH LOOP")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Logging: {RESULTS_TSV}")
    print(f"{'#'*70}\n")

    # ── Experiment 1: Phase 6b (may already be running) ──
    phase6b_log = PROJECT / "logs" / "phase6b_gdpo_vppo_r2.log"
    if phase6b_log.exists():
        # Check if it completed
        with open(phase6b_log) as f:
            content = f.read()
        if "COMPLETE" in content:
            print("[skip] Phase 6b already completed")
            metrics = parse_results(phase6b_log)
            log_result("phase6b_gdpo_vppo", metrics, "completed")
        else:
            print("[wait] Phase 6b still running, waiting for GPU...")
            wait_for_gpu(check_interval=60, max_wait=3600)
            metrics = parse_results(phase6b_log)
            log_result("phase6b_gdpo_vppo", metrics,
                       "completed" if metrics["pope"] > 0 else "interrupted")
    else:
        print("[skip] No Phase 6b log found")

    # ── Experiment 2: Phase 7 BoN+SFT (main experiment) ──
    print("\n[queue] Phase 7: BoN+SFT on TextVQA with Head-Level Vision Scoring")
    wait_for_gpu()

    code = run_experiment(
        "phase7_bon_sft_r1",
        "PYTHONUNBUFFERED=1 python -u scripts/phase7_bon_sft_textvqa.py "
        "--num-samples 500 --group-size 8 --sft-lr 1e-6 --sft-epochs 2 "
        "--output-dir checkpoints/phase7_bon_sft/r1",
        timeout_min=180,  # 3 hours
    )
    metrics = parse_results(PROJECT / "logs" / "phase7_bon_sft_r1.log")
    log_result("phase7_bon_sft_r1", metrics,
               "completed" if code == 0 else f"exit_{code}")

    # ── Experiment 3: Phase 7 Round 2 (if R1 improved) ──
    baseline_tvqa = 0.727  # Known baseline
    if metrics.get("textvqa", 0) > baseline_tvqa:
        print(f"\n[queue] Phase 7 R2: BoN+SFT from R1 checkpoint "
              f"(TextVQA {metrics['textvqa']:.1%} > {baseline_tvqa:.1%})")
        wait_for_gpu()

        code = run_experiment(
            "phase7_bon_sft_r2",
            "PYTHONUNBUFFERED=1 python -u scripts/phase7_bon_sft_textvqa.py "
            "--model-path checkpoints/phase7_bon_sft/r1/final "
            "--num-samples 500 --group-size 8 --sft-lr 5e-7 --sft-epochs 1 "
            "--seed 43 --output-dir checkpoints/phase7_bon_sft/r2",
            timeout_min=180,
        )
        metrics_r2 = parse_results(PROJECT / "logs" / "phase7_bon_sft_r2.log")
        log_result("phase7_bon_sft_r2", metrics_r2,
                   "completed" if code == 0 else f"exit_{code}")

        # Update best metrics
        if metrics_r2.get("textvqa", 0) > metrics.get("textvqa", 0):
            metrics = metrics_r2
            print(f"  [keep] R2 improved: TextVQA={metrics['textvqa']:.1%}")
        else:
            print(f"  [discard] R2 no improvement")
    else:
        print(f"\n[skip] Phase 7 R1 didn't improve TextVQA "
              f"({metrics.get('textvqa', 0):.1%} <= {baseline_tvqa:.1%})")

    # ── Experiment 4: Alpha ablation (no vision scoring) ──
    print("\n[queue] Phase 7 Ablation: BoN+SFT with correctness-only scoring")
    wait_for_gpu()

    code = run_experiment(
        "phase7_bon_sft_ablation",
        "PYTHONUNBUFFERED=1 python -u scripts/phase7_bon_sft_textvqa.py "
        "--num-samples 500 --group-size 8 --sft-lr 1e-6 --sft-epochs 2 "
        "--w-correct 1.0 --w-vision 0.0 "
        "--output-dir checkpoints/phase7_bon_sft/ablation_no_vision",
        timeout_min=120,
    )
    metrics_abl = parse_results(PROJECT / "logs" / "phase7_bon_sft_ablation.log")
    log_result("phase7_bon_ablation_no_vision", metrics_abl,
               "completed" if code == 0 else f"exit_{code}")

    # ── Summary ──
    print(f"\n{'#'*70}")
    print(f"  AUTO-RESEARCH COMPLETE")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Results in: {RESULTS_TSV}")
    print(f"{'#'*70}\n")

    # Print results table
    if RESULTS_TSV.exists():
        print("\nResults Summary:")
        print("-" * 80)
        with open(RESULTS_TSV) as f:
            print(f.read())


if __name__ == "__main__":
    main()
