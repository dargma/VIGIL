#!/usr/bin/env python3
"""
Generate publication-quality plots for VIGIL Block 2 experiment progression.

Figures:
  1. POPE accuracy across all experiments
  2. Blind Gap progression across all experiments
  3. BoN+SFT candidate quality distribution
  4. Method comparison table (rendered as heatmap)

Usage:
  python lab/reports/generate_block2_plots.py
"""

import json
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTDIR = os.path.join(BASE, "lab", "reports")
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Colour palette  (colourblind-friendly)
# ---------------------------------------------------------------------------
C_COLLAPSED = "#d62728"   # red  — collapsed runs
C_GRPO_V2   = "#ff7f0e"   # orange — GRPO v2
C_GRPO_V3   = "#9467bd"   # purple — GRPO v3
C_BON_R1    = "#2ca02c"   # green  — BoN+SFT round 1
C_BON_R2    = "#1f77b4"   # blue   — BoN+SFT round 2 (placeholder)
C_BASELINE  = "#7f7f7f"   # grey   — baseline

# ---------------------------------------------------------------------------
# Data  (hard-coded from experiment JSONs — avoids runtime file dependency)
# ---------------------------------------------------------------------------

# Experiment labels and final results
experiments = [
    # label, pope_start, pope_end, gap_start, gap_end, collapsed, color
    ("Block 1 v1\n(TRL GRPO)",     76.0, 31.5, 26.0, 31.5, True,  C_COLLAPSED),
    ("Block 1 v2\n(+format rwd)",  76.0, 30.0, 26.0, 30.0, True,  C_COLLAPSED),
    ("Block 1 v3\n(ultra-cons.)",  77.0, 31.0, 27.0, 31.0, True,  C_COLLAPSED),
    ("Block 2 v2\n(Custom GRPO)",  84.5, 85.0, 35.0, 36.0, False, C_GRPO_V2),
    ("Block 2 v3\n(100 steps)",    84.5, 83.5, 35.0, 33.0, False, C_GRPO_V3),
    ("BoN+SFT\nRound 1",          83.0, 85.5, 32.0, 37.0, False, C_BON_R1),
    ("BoN+SFT\nRound 2",          85.5, None, 37.0, None, False, C_BON_R2),  # placeholder
]

# Block 2 v2 Setting B eval history (step -> pope, gap)
grpo_v2_eval = [
    (0,  84.5, 35.0),
    (10, 84.5, 35.0),
    (20, 85.0, 36.0),
    (30, 84.0, 34.0),
    (40, 84.0, 34.0),
    (50, 85.0, 36.0),
]

# Block 2 v3 Setting B eval history
grpo_v3_eval = [
    (0,   84.5, 35.0),
    (10,  83.5, 33.0),
    (20,  83.5, 33.0),
    (30,  83.5, 33.0),
    (40,  84.0, 34.0),
    (50,  84.5, 35.0),
    (60,  84.0, 34.0),
    (70,  84.0, 34.0),
    (80,  84.5, 35.0),
    (90,  84.0, 34.0),
    (100, 83.5, 33.0),
]


# ===================================================================
# Figure 1: POPE Accuracy Progression
# ===================================================================
def fig1_pope_progression():
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = []
    starts = []
    ends = []
    colors = []
    hatches = []

    for label, ps, pe, gs, ge, collapsed, color in experiments:
        if pe is None:
            continue  # skip placeholder
        labels.append(label)
        starts.append(ps)
        ends.append(pe)
        colors.append(color)
        hatches.append("///" if collapsed else "")

    x = np.arange(len(labels))
    width = 0.35

    bars_start = ax.bar(x - width / 2, starts, width, label="Before",
                        color=[c + "55" for c in colors], edgecolor=colors, linewidth=1.2)
    bars_end = ax.bar(x + width / 2, ends, width, label="After",
                      color=colors, edgecolor="black", linewidth=0.8)

    # Add hatch to collapsed bars
    for i, h in enumerate(hatches):
        if h:
            bars_end[i].set_hatch(h)
            bars_end[i].set_edgecolor("white")

    # Value annotations
    for i in range(len(labels)):
        delta = ends[i] - starts[i]
        sign = "+" if delta >= 0 else ""
        color_txt = C_COLLAPSED if hatches[i] else "black"
        ax.annotate(f"{sign}{delta:.1f}pp",
                    xy=(x[i] + width / 2, ends[i]),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold", color=color_txt)

    # Baseline reference line
    ax.axhline(y=76.0, color=C_BASELINE, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(len(labels) - 0.5, 76.5, "Baseline (76%)", fontsize=8,
            color=C_BASELINE, ha="right")

    # Collapse zone
    ax.axhspan(0, 50, alpha=0.05, color="red")
    ax.text(0.02, 0.08, "Collapse zone", transform=ax.transAxes,
            fontsize=9, color=C_COLLAPSED, alpha=0.6)

    ax.set_ylabel("POPE Accuracy (%)")
    ax.set_title("POPE Accuracy Progression Across Experiments")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(20, 95)
    ax.legend(loc="upper left")

    # Custom legend for collapsed
    collapse_patch = mpatches.Patch(facecolor=C_COLLAPSED, hatch="///",
                                     edgecolor="white", label="Collapsed")
    success_patch = mpatches.Patch(facecolor=C_BON_R1, label="Improved")
    ax.legend(handles=[
        mpatches.Patch(facecolor="#cccccc", edgecolor="black", label="Before"),
        mpatches.Patch(facecolor="#666666", edgecolor="black", label="After"),
        collapse_patch, success_patch,
    ], loc="upper left", fontsize=9)

    fig.tight_layout()
    path = os.path.join(OUTDIR, "fig1_pope_progression.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ===================================================================
# Figure 2: Blind Gap Progression
# ===================================================================
def fig2_blind_gap():
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = []
    starts = []
    ends = []
    colors = []

    for label, ps, pe, gs, ge, collapsed, color in experiments:
        if ge is None:
            continue
        labels.append(label)
        starts.append(gs)
        ends.append(ge)
        colors.append(color)

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width / 2, starts, width, label="Before",
           color=[c + "55" for c in colors], edgecolor=colors, linewidth=1.2)
    ax.bar(x + width / 2, ends, width, label="After",
           color=colors, edgecolor="black", linewidth=0.8)

    for i in range(len(labels)):
        delta = ends[i] - starts[i]
        sign = "+" if delta >= 0 else ""
        ax.annotate(f"{sign}{delta:.1f}pp",
                    xy=(x[i] + width / 2, ends[i]),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold")

    # Note: Block 1 gap "increases" are collapse artifacts
    for i in range(3):  # first 3 are collapsed
        ax.annotate("*", xy=(x[i] + width / 2, ends[i] + 2),
                    ha="center", fontsize=14, color=C_COLLAPSED, fontweight="bold")

    ax.set_ylabel("Blind Test Gap (pp)")
    ax.set_title("Blind Test Gap: Real Image Acc - Black Image Acc")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(20, 45)
    ax.legend(loc="upper left", fontsize=9)

    # Footnote
    ax.text(0.02, 0.02, "* Gap increase is collapse artifact (always-yes/no bias)",
            transform=ax.transAxes, fontsize=8, color=C_COLLAPSED, style="italic")

    fig.tight_layout()
    path = os.path.join(OUTDIR, "fig2_blind_gap_progression.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ===================================================================
# Figure 3: BoN+SFT Candidate Quality Distribution
# ===================================================================
def fig3_candidate_quality():
    # Load candidate data
    candidates_path = os.path.join(BASE, "data", "training", "bon_candidates.json")
    if not os.path.exists(candidates_path):
        print(f"Skipping fig3: {candidates_path} not found")
        return

    with open(candidates_path) as f:
        candidates = json.load(f)

    best_scores = [c["best_score"] for c in candidates]
    all_scores = []
    for c in candidates:
        all_scores.extend(c["all_scores"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: all candidate scores vs best-of-N scores
    ax = axes[0]
    ax.hist(all_scores, bins=50, alpha=0.5, color="#1f77b4", label="All candidates (N=8)", density=True)
    ax.hist(best_scores, bins=50, alpha=0.7, color=C_BON_R1, label="Best-of-N selected", density=True)
    ax.set_xlabel("Composite Score (R_correct + IIG)")
    ax.set_ylabel("Density")
    ax.set_title("Candidate Score Distribution")
    ax.legend(fontsize=9)
    ax.axvline(x=np.mean(best_scores), color=C_BON_R1, linestyle="--", linewidth=1)
    ax.text(np.mean(best_scores) + 0.02, ax.get_ylim()[1] * 0.9,
            f"mean={np.mean(best_scores):.3f}", fontsize=8, color=C_BON_R1)

    # Right: breakdown by source
    ax = axes[1]
    from collections import Counter
    sources = Counter(c["source"] for c in candidates)
    source_names = list(sources.keys())
    source_counts = [sources[s] for s in source_names]
    source_colors = ["#ff7f0e", "#2ca02c", "#1f77b4"]

    bars = ax.bar(source_names, source_counts, color=source_colors[:len(source_names)],
                  edgecolor="black", linewidth=0.8)
    for bar, count in zip(bars, source_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(count), ha="center", fontsize=10, fontweight="bold")

    ax.set_ylabel("Number of Curated Samples")
    ax.set_title(f"BoN+SFT Training Data (N={len(candidates)} total)")
    ax.set_ylim(0, max(source_counts) * 1.15)

    fig.suptitle("BoN+SFT Round 1: Candidate Selection Analysis", fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(OUTDIR, "fig3_bon_candidate_quality.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ===================================================================
# Figure 4: Method Comparison Table
# ===================================================================
def fig4_method_comparison():
    methods = [
        "TRL GRPO\n(Block 1 v1-v3)",
        "Custom GRPO\n(Block 2 v2)",
        "Custom GRPO\n(Block 2 v3, 100 steps)",
        "BoN+SFT\n(Round 1)",
    ]

    # Metrics: POPE delta, Gap delta, Collapsed?, Training stability
    pope_delta  = [-45.0, +0.5,  -1.0,  +2.5]
    gap_delta   = [+5.0,  +1.0,  -2.0,  +5.0]   # Block 1 gap is artifact
    collapsed   = [1,     0,     0,     0]
    stability   = [0,     0.7,   0.8,   1.0]  # 0=collapsed, 1=perfect

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")

    # Build table data
    col_labels = ["Method", "POPE\u0394 (pp)", "Gap\u0394 (pp)", "Collapsed?", "Stability"]
    cell_text = []
    cell_colors = []

    for i in range(len(methods)):
        row = [
            methods[i].replace("\n", " "),
            f"{pope_delta[i]:+.1f}" if pope_delta[i] != -45.0 else "-45.0",
            f"{gap_delta[i]:+.1f}" + (" *" if collapsed[i] else ""),
            "YES" if collapsed[i] else "No",
            ["Collapsed", "Stable", "Stable", "Stable"][i],
        ]
        cell_text.append(row)

        # Color coding
        if collapsed[i]:
            cell_colors.append(["#ffcccc"] * 5)
        elif pope_delta[i] > 1.0:
            cell_colors.append(["#ccffcc"] * 5)
        else:
            cell_colors.append(["#ffffcc"] * 5)

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     cellColours=cell_colors,
                     colColours=["#d9e2f3"] * 5,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)

    # Bold header
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_text_props(fontweight="bold")
        cell.set_edgecolor("#999999")

    ax.set_title("Method Comparison: GRPO vs BoN+SFT",
                 fontsize=14, fontweight="bold", pad=20)

    # Footnote
    fig.text(0.5, 0.02,
             "* Block 1 Gap increases are artifacts of mode collapse (always-yes/no).\n"
             "Green = improved POPE. Yellow = stable. Red = collapsed.",
             ha="center", fontsize=9, style="italic", color="#666666")

    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    path = os.path.join(OUTDIR, "fig4_method_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ===================================================================
# Bonus: GRPO training dynamics (v2 + v3 step-by-step)
# ===================================================================
def fig5_grpo_dynamics():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # POPE over steps
    steps_v2 = [s for s, _, _ in grpo_v2_eval]
    pope_v2  = [p for _, p, _ in grpo_v2_eval]
    steps_v3 = [s for s, _, _ in grpo_v3_eval]
    pope_v3  = [p for _, p, _ in grpo_v3_eval]

    ax1.plot(steps_v2, pope_v2, "o-", color=C_GRPO_V2, linewidth=2, markersize=6,
             label="GRPO v2 (50 steps)")
    ax1.plot(steps_v3, pope_v3, "s-", color=C_GRPO_V3, linewidth=2, markersize=6,
             label="GRPO v3 (100 steps)")

    # BoN+SFT result as horizontal band
    ax1.axhspan(85.0, 86.0, alpha=0.15, color=C_BON_R1)
    ax1.axhline(y=85.5, color=C_BON_R1, linestyle="--", linewidth=1.5, label="BoN+SFT R1")

    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("POPE Accuracy (%)")
    ax1.set_title("POPE During GRPO Training")
    ax1.legend(fontsize=9)
    ax1.set_ylim(82, 87)

    # Gap over steps
    gap_v2 = [g for _, _, g in grpo_v2_eval]
    gap_v3 = [g for _, _, g in grpo_v3_eval]

    ax2.plot(steps_v2, gap_v2, "o-", color=C_GRPO_V2, linewidth=2, markersize=6,
             label="GRPO v2 (50 steps)")
    ax2.plot(steps_v3, gap_v3, "s-", color=C_GRPO_V3, linewidth=2, markersize=6,
             label="GRPO v3 (100 steps)")

    ax2.axhline(y=37.0, color=C_BON_R1, linestyle="--", linewidth=1.5, label="BoN+SFT R1")

    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Blind Test Gap (pp)")
    ax2.set_title("Blind Gap During GRPO Training")
    ax2.legend(fontsize=9)
    ax2.set_ylim(30, 40)

    fig.suptitle("GRPO Oscillates While BoN+SFT Delivers Consistent Improvement",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    path = os.path.join(OUTDIR, "fig5_grpo_dynamics.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    print("Generating VIGIL Block 2 experiment plots...")
    print(f"Output directory: {OUTDIR}\n")
    fig1_pope_progression()
    fig2_blind_gap()
    fig3_candidate_quality()
    fig4_method_comparison()
    fig5_grpo_dynamics()
    print("\nAll plots generated.")
