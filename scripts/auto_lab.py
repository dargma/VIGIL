"""
VIGIL Auto Lab — state machine that drives the experiment loop.

Observe → Hypothesize → Experiment → Analyze → Iterate

Reads current state from lab/RESEARCH_JOURNAL.md and checkpoints,
decides what to run next, runs it, logs the outcome, loops.
"""

import sys
import os
import json
import time
import traceback
import subprocess
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent
JOURNAL_PATH = PROJECT_ROOT / "lab" / "RESEARCH_JOURNAL.md"
RESULTS_DIR = PROJECT_ROOT / "lab" / "results"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
STATE_FILE = PROJECT_ROOT / "lab" / "auto_lab_state.json"

# Ordered experiment pipeline
STAGES = [
    "smoke_test",
    "calibrate",
    "baseline_eval",
    "steering_eval",
    "blind_test_baseline",
    "grpo_train",
    "grpo_eval",
    "blind_test_grpo",
    "grpo_lightweight_train",
    "grpo_lightweight_eval",
    "blind_test_lightweight",
    "ablation",
]


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"completed": [], "current": None, "failures": {}, "last_updated": None}


def save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state["last_updated"] = datetime.now().isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def journal_append(entry: str):
    with open(JOURNAL_PATH, "a") as f:
        f.write(f"\n---\n\n## {datetime.now().strftime('%Y-%m-%d %H:%M')} — Auto Lab\n\n{entry}\n")


def determine_next(state: dict) -> str | None:
    for stage in STAGES:
        if stage not in state["completed"]:
            # Skip if failed too many times
            fail_count = state["failures"].get(stage, 0)
            if fail_count >= 3:
                print(f"[auto_lab] Skipping {stage} (failed {fail_count} times)")
                continue
            return stage
    return None


def run_stage(stage: str, model_key: str = "qwen3_vl_2b") -> dict:
    """Run a stage and return result dict."""
    print(f"\n{'='*60}")
    print(f"[auto_lab] Running: {stage}")
    print(f"{'='*60}\n")

    result = {"stage": stage, "status": "unknown", "detail": ""}

    try:
        if stage == "smoke_test":
            result = run_smoke_test(model_key)
        elif stage == "calibrate":
            result = run_calibrate(model_key)
        elif stage == "baseline_eval":
            result = run_baseline_eval(model_key)
        elif stage == "steering_eval":
            result = run_steering_eval(model_key)
        elif stage == "blind_test_baseline":
            result = run_blind_test(model_key, tag="baseline")
        elif stage == "grpo_train":
            result = run_grpo(model_key, reward_mode="full")
        elif stage == "grpo_eval":
            result = run_eval_post_training(model_key, tag="grpo_full")
        elif stage == "blind_test_grpo":
            result = run_blind_test(model_key, tag="grpo_full")
        elif stage == "grpo_lightweight_train":
            result = run_grpo(model_key, reward_mode="lightweight")
        elif stage == "grpo_lightweight_eval":
            result = run_eval_post_training(model_key, tag="grpo_lightweight")
        elif stage == "blind_test_lightweight":
            result = run_blind_test(model_key, tag="grpo_lightweight")
        elif stage == "ablation":
            result = run_ablation(model_key)
        else:
            result = {"stage": stage, "status": "error", "detail": f"Unknown stage: {stage}"}
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        result = {"stage": stage, "status": "oom", "detail": "GPU OOM — will retry with smaller batch"}
    except Exception as e:
        result = {"stage": stage, "status": "error", "detail": f"{type(e).__name__}: {str(e)[:200]}"}
        traceback.print_exc()

    return result


def run_smoke_test(model_key: str) -> dict:
    ret = subprocess.run(
        [sys.executable, "scripts/smoke_test.py", "--model", model_key],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=600,
    )
    passed = ret.returncode == 0
    return {
        "stage": "smoke_test", "status": "ok" if passed else "fail",
        "detail": ret.stdout[-500:] if ret.stdout else ret.stderr[-500:],
    }


def run_calibrate(model_key: str) -> dict:
    output_dir = str(CHECKPOINT_DIR / "calibration" / model_key)
    if (Path(output_dir) / "calibration_meta.json").exists():
        return {"stage": "calibrate", "status": "ok", "detail": "Already calibrated"}

    ret = subprocess.run(
        [sys.executable, "scripts/calibrate.py",
         "--model", model_key, "--output-dir", output_dir,
         "--max-samples", "1000", "--top-k", "20", "--limit-per-source", "500"],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=1800,
    )
    passed = ret.returncode == 0
    return {
        "stage": "calibrate", "status": "ok" if passed else "fail",
        "detail": ret.stdout[-500:] if ret.stdout else ret.stderr[-500:],
    }


def run_baseline_eval(model_key: str) -> dict:
    ret = subprocess.run(
        [sys.executable, "scripts/eval_benchmarks.py",
         "--model", model_key, "--benchmarks", "pope",
         "--mode", "greedy", "--output-dir", str(RESULTS_DIR / "baseline")],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=3600,
    )
    passed = ret.returncode == 0
    return {
        "stage": "baseline_eval", "status": "ok" if passed else "fail",
        "detail": ret.stdout[-500:] if ret.stdout else ret.stderr[-500:],
    }


def run_steering_eval(model_key: str) -> dict:
    calib_dir = str(CHECKPOINT_DIR / "calibration" / model_key)
    ret = subprocess.run(
        [sys.executable, "scripts/eval_benchmarks.py",
         "--model", model_key, "--benchmarks", "pope",
         "--mode", "steered", "--calibration-dir", calib_dir,
         "--output-dir", str(RESULTS_DIR / "steered")],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=3600,
    )
    passed = ret.returncode == 0
    return {
        "stage": "steering_eval", "status": "ok" if passed else "fail",
        "detail": ret.stdout[-500:] if ret.stdout else ret.stderr[-500:],
    }


def run_blind_test(model_key: str, tag: str = "baseline") -> dict:
    ret = subprocess.run(
        [sys.executable, "scripts/run_blind_test.py",
         "--model", model_key,
         "--output-dir", str(RESULTS_DIR / f"blind_test_{tag}")],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=3600,
    )
    passed = ret.returncode == 0
    return {
        "stage": f"blind_test_{tag}", "status": "ok" if passed else "fail",
        "detail": ret.stdout[-500:] if ret.stdout else ret.stderr[-500:],
    }


def run_grpo(model_key: str, reward_mode: str = "full") -> dict:
    ret = subprocess.run(
        [sys.executable, "scripts/train_grpo.py",
         "--model", model_key, "--reward-mode", reward_mode,
         "--output-dir", str(CHECKPOINT_DIR / f"grpo_{reward_mode}")],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=14400,
    )
    passed = ret.returncode == 0
    return {
        "stage": f"grpo_{reward_mode}_train", "status": "ok" if passed else "fail",
        "detail": ret.stdout[-500:] if ret.stdout else ret.stderr[-500:],
    }


def run_eval_post_training(model_key: str, tag: str = "grpo_full") -> dict:
    ret = subprocess.run(
        [sys.executable, "scripts/eval_benchmarks.py",
         "--model", model_key, "--benchmarks", "pope",
         "--mode", "greedy",
         "--checkpoint", str(CHECKPOINT_DIR / tag),
         "--output-dir", str(RESULTS_DIR / tag)],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=3600,
    )
    passed = ret.returncode == 0
    return {
        "stage": f"{tag}_eval", "status": "ok" if passed else "fail",
        "detail": ret.stdout[-500:] if ret.stdout else ret.stderr[-500:],
    }


def run_ablation(model_key: str) -> dict:
    return {"stage": "ablation", "status": "skip", "detail": "Ablation not yet implemented"}


def main():
    import torch  # late import for non-GPU environments

    model_key = sys.argv[1] if len(sys.argv) > 1 else "qwen3_vl_2b"
    max_iterations = int(sys.argv[2]) if len(sys.argv) > 2 else len(STAGES)

    state = load_state()
    print(f"[auto_lab] State: completed={state['completed']}, failures={state['failures']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for iteration in range(max_iterations):
        next_stage = determine_next(state)
        if next_stage is None:
            print("[auto_lab] All stages complete.")
            break

        state["current"] = next_stage
        save_state(state)

        result = run_stage(next_stage, model_key)
        print(f"\n[auto_lab] Result: {result['status']} — {result.get('detail', '')[:200]}")

        if result["status"] == "ok":
            state["completed"].append(next_stage)
            journal_append(
                f"**Stage**: `{next_stage}` — **PASSED**\n\n"
                f"**Detail**: {result.get('detail', 'No detail')[:500]}\n"
            )
        elif result["status"] == "oom":
            state["failures"][next_stage] = state["failures"].get(next_stage, 0) + 1
            journal_append(
                f"**Stage**: `{next_stage}` — **OOM** (attempt {state['failures'][next_stage]})\n\n"
                f"**Action**: Will retry with reduced batch size.\n"
            )
        elif result["status"] == "skip":
            state["completed"].append(next_stage)
            journal_append(f"**Stage**: `{next_stage}` — **SKIPPED**: {result.get('detail', '')}\n")
        else:
            state["failures"][next_stage] = state["failures"].get(next_stage, 0) + 1
            journal_append(
                f"**Stage**: `{next_stage}` — **FAILED** (attempt {state['failures'][next_stage]})\n\n"
                f"**Error**: {result.get('detail', 'Unknown')[:500]}\n\n"
                f"**Hypothesis**: Needs investigation.\n"
            )

        state["current"] = None
        save_state(state)

    print(f"\n[auto_lab] Done. Completed: {state['completed']}")


if __name__ == "__main__":
    main()
