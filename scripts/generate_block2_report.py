#!/usr/bin/env python3
"""Generate Block 2 experiment result plots.

Loads v2-A, v2-B (JSON), and v3-B (log parsing) results, then creates:
  1. POPE accuracy over training steps (3 runs overlaid)
  2. Blind test Gap over training steps (3 runs overlaid)
  3. Skip rate over training steps (comparison)
  4. Reward statistics over steps (mean reward + std for v2-B and v3-B)

Saves plots to lab/reports/.
"""

import json
import re
import os
import sys
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = "/content/drive/MyDrive/VIGIL"
V2A_PATH = os.path.join(BASE, "lab/results/block2_v2/block2v2_A_20260308_013027.json")
V2B_PATH = os.path.join(BASE, "lab/results/block2_v2/block2v2_B_20260308_015055.json")
V3B_LOG  = os.path.join(BASE, "logs/block2v3_settingB.log")
OUT_DIR  = os.path.join(BASE, "lab/reports")
os.makedirs(OUT_DIR, exist_ok=True)


# ── Load v2 JSON data ─────────────────────────────────────────────────────
def load_v2_json(path):
    with open(path) as f:
        data = json.load(f)
    # Eval history: step, pope.acc, blind.gap
    eval_steps = [e["step"] for e in data["eval_history"]]
    pope_acc   = [e["pope"]["acc"] for e in data["eval_history"]]
    blind_gap  = [e["blind"]["gap"] for e in data["eval_history"]]
    # Log history: per-step training stats
    log_steps     = [e["step"] for e in data["log_history"]]
    reward_mean   = [e["reward_mean"] for e in data["log_history"]]
    reward_std    = [e["reward_std"] for e in data["log_history"]]
    # Skip rate: skipped / (valid + skipped)
    skip_rate = []
    for e in data["log_history"]:
        total = e["valid"] + e["skipped"]
        skip_rate.append(e["skipped"] / total * 100 if total > 0 else 0)
    return {
        "eval_steps": eval_steps, "pope_acc": pope_acc, "blind_gap": blind_gap,
        "log_steps": log_steps, "reward_mean": reward_mean, "reward_std": reward_std,
        "skip_rate": skip_rate,
    }


# ── Parse v3 log ──────────────────────────────────────────────────────────
def parse_v3_log(path):
    eval_steps, pope_acc, blind_gap = [], [], []
    log_steps, reward_mean, reward_std, skip_rate = [], [], [], []

    with open(path) as f:
        lines = f.readlines()

    # First POPE/Blind before any "Eval at step" = step 0 baseline
    baseline_pope = None
    baseline_blind = None
    seen_eval = False

    for i, line in enumerate(lines):
        # Per-step training line
        m_step = re.search(
            r'Step\s+(\d+)\s+\|.*?R=([\d.]+)\+/-([\d.]+).*?skip=(\d+)%', line
        )
        if m_step:
            log_steps.append(int(m_step.group(1)))
            reward_mean.append(float(m_step.group(2)))
            reward_std.append(float(m_step.group(3)))
            skip_rate.append(float(m_step.group(4)))

        # Eval header
        m_eval = re.search(r'Eval at step (\d+)', line)
        if m_eval:
            seen_eval = True
            current_step = int(m_eval.group(1))

        # POPE line
        m_pope = re.search(r'POPE:\s+([\d.]+)%', line)
        if m_pope:
            acc = float(m_pope.group(1))
            if not seen_eval and baseline_pope is None:
                baseline_pope = acc
            elif seen_eval:
                eval_steps.append(current_step)
                pope_acc.append(acc)

        # Blind line
        m_blind = re.search(r'Gap=([\d.]+)pp', line)
        if m_blind:
            gap = float(m_blind.group(1))
            if not seen_eval and baseline_blind is None:
                baseline_blind = gap
            elif seen_eval:
                blind_gap.append(gap)

    # Prepend baseline as step 0
    if baseline_pope is not None:
        eval_steps.insert(0, 0)
        pope_acc.insert(0, baseline_pope)
    if baseline_blind is not None:
        blind_gap.insert(0, baseline_blind)

    return {
        "eval_steps": eval_steps, "pope_acc": pope_acc, "blind_gap": blind_gap,
        "log_steps": log_steps, "reward_mean": reward_mean, "reward_std": reward_std,
        "skip_rate": skip_rate,
    }


# ── Load all data ─────────────────────────────────────────────────────────
print("Loading v2-A ...")
v2a = load_v2_json(V2A_PATH)
print(f"  eval points: {len(v2a['eval_steps'])}, log steps: {len(v2a['log_steps'])}")

print("Loading v2-B ...")
v2b = load_v2_json(V2B_PATH)
print(f"  eval points: {len(v2b['eval_steps'])}, log steps: {len(v2b['log_steps'])}")

print("Parsing v3-B log ...")
v3b = parse_v3_log(V3B_LOG)
print(f"  eval points: {len(v3b['eval_steps'])}, log steps: {len(v3b['log_steps'])}")


# ── Plotting ──────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.size": 12})

COLORS = {"v2-A": "#2196F3", "v2-B": "#E91E63", "v3-B": "#4CAF50"}
LABELS = {
    "v2-A": "v2-A (no IIG, 50 steps)",
    "v2-B": "v2-B (+IIG, 50 steps)",
    "v3-B": "v3-B (+IIG, 100 steps)",
}


# ── Plot 1: POPE accuracy ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
for key, d in [("v2-A", v2a), ("v2-B", v2b), ("v3-B", v3b)]:
    ax.plot(d["eval_steps"], d["pope_acc"], "o-", color=COLORS[key],
            label=LABELS[key], linewidth=2, markersize=6)
ax.set_xlabel("Training Step")
ax.set_ylabel("POPE Accuracy (%)")
ax.set_title("Block 2: POPE Accuracy Over Training Steps")
ax.legend(loc="lower left")
ax.set_ylim(80, 88)
ax.axhline(y=84.5, color="gray", linestyle="--", alpha=0.5, label="Baseline (84.5%)")
fig.tight_layout()
out1 = os.path.join(OUT_DIR, "block2_pope_accuracy.png")
fig.savefig(out1, dpi=150)
plt.close(fig)
print(f"Saved: {out1}")


# ── Plot 2: Blind test Gap ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
for key, d in [("v2-A", v2a), ("v2-B", v2b), ("v3-B", v3b)]:
    # Ensure blind_gap and eval_steps are same length
    n = min(len(d["eval_steps"]), len(d["blind_gap"]))
    ax.plot(d["eval_steps"][:n], d["blind_gap"][:n], "s-", color=COLORS[key],
            label=LABELS[key], linewidth=2, markersize=6)
ax.set_xlabel("Training Step")
ax.set_ylabel("Blind Test Gap (pp)")
ax.set_title("Block 2: Blind Test Gap Over Training Steps")
ax.legend(loc="lower left")
ax.set_ylim(28, 40)
ax.axhline(y=35.0, color="gray", linestyle="--", alpha=0.5, label="Baseline (35.0pp)")
fig.tight_layout()
out2 = os.path.join(OUT_DIR, "block2_blind_gap.png")
fig.savefig(out2, dpi=150)
plt.close(fig)
print(f"Saved: {out2}")


# ── Plot 3: Skip rate ─────────────────────────────────────────────────────
def rolling_mean(arr, window=5):
    """Simple rolling mean for smoothing."""
    out = []
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        out.append(np.mean(arr[start:i+1]))
    return out

fig, ax = plt.subplots(figsize=(10, 6))
for key, d in [("v2-A", v2a), ("v2-B", v2b), ("v3-B", v3b)]:
    smoothed = rolling_mean(d["skip_rate"], window=5)
    ax.plot(d["log_steps"], smoothed, "-", color=COLORS[key],
            label=LABELS[key], linewidth=2, alpha=0.85)
    # Light raw data behind
    ax.plot(d["log_steps"], d["skip_rate"], ".", color=COLORS[key],
            alpha=0.15, markersize=3)
ax.set_xlabel("Training Step")
ax.set_ylabel("Skip Rate (%)")
ax.set_title("Block 2: Zero-Variance Group Skip Rate (5-step rolling avg)")
ax.legend(loc="upper right")
ax.set_ylim(0, 105)
fig.tight_layout()
out3 = os.path.join(OUT_DIR, "block2_skip_rate.png")
fig.savefig(out3, dpi=150)
plt.close(fig)
print(f"Saved: {out3}")


# ── Plot 4: Reward statistics (v2-B and v3-B) ─────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for ax, key, d, title in [
    (ax1, "v2-B", v2b, "v2-B (+IIG, 50 steps)"),
    (ax2, "v3-B", v3b, "v3-B (+IIG, 100 steps)"),
]:
    steps = np.array(d["log_steps"])
    rmean = np.array(d["reward_mean"])
    rstd  = np.array(d["reward_std"])

    ax.fill_between(steps, rmean - rstd, rmean + rstd,
                     alpha=0.2, color=COLORS[key])
    ax.plot(steps, rmean, "-", color=COLORS[key], linewidth=1.5, label="Mean reward")
    # Overlay smoothed mean
    sm = rolling_mean(rmean.tolist(), window=5)
    ax.plot(steps, sm, "-", color="black", linewidth=2, alpha=0.7, label="Smoothed mean (w=5)")
    ax.set_xlabel("Training Step")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=10)

ax1.set_ylabel("Reward")
fig.suptitle("Block 2: Reward Statistics Over Training Steps", fontsize=14, y=1.02)
fig.tight_layout()
out4 = os.path.join(OUT_DIR, "block2_reward_stats.png")
fig.savefig(out4, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out4}")

print("\nAll 4 plots generated successfully.")
