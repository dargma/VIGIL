#!/usr/bin/env python3
"""Generate publication-quality Phase 2 visualization figures for VIGIL."""

import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

OUT_DIR = "/content/drive/MyDrive/VIGIL/lab/reports/phase2_visualizations"
LSR_PATH = "/content/drive/MyDrive/VIGIL/lab/reports/pope_thinking_steering/lsr_trajectories_20260310_113327.json"
HISTORY_DIR = "/content/drive/MyDrive/VIGIL/checkpoints/phase2_grpo_lsr"

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

ROUND_COLORS = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0']
ROUND_LABELS = ['Round 1', 'Round 2', 'Round 3', 'Round 4', 'Round 5']


def load_histories():
    """Load all round histories."""
    histories = []
    for r in range(1, 6):
        files = glob.glob(f"{HISTORY_DIR}/round{r}/history_*.json")
        if files:
            with open(sorted(files)[0]) as f:
                histories.append(json.load(f))
    return histories


# ============================================================
# Figure 1: LSR Heatmap
# ============================================================
def figure1_lsr_heatmap():
    with open(LSR_PATH) as f:
        trajectories = json.load(f)

    # Pad/truncate KL sequences to common length and build matrix
    max_tokens = max(len(t['kl_per_token']) for t in trajectories)
    # Use a reasonable display width — bin tokens into groups
    n_bins = min(max_tokens, 100)
    n_samples = len(trajectories)

    matrix = np.zeros((n_samples, n_bins))
    think_boundaries = []

    for i, t in enumerate(trajectories):
        kl = np.array(t['kl_per_token'], dtype=np.float64)
        kl = np.clip(kl, 0, None)  # remove any tiny negatives
        # Bin into n_bins
        bin_size = max(1, len(kl) / n_bins)
        for b in range(n_bins):
            start = int(b * bin_size)
            end = int((b + 1) * bin_size)
            end = min(end, len(kl))
            if start < len(kl):
                matrix[i, b] = np.mean(kl[start:end])
        # Think boundary as bin index
        think_tok = t.get('think_end_token', len(kl))
        think_boundaries.append(int(think_tok / bin_size))

    # Sort by think_end_token for cleaner visual
    sort_idx = np.argsort(think_boundaries)
    matrix = matrix[sort_idx]
    think_boundaries = [think_boundaries[i] for i in sort_idx]

    # Cap extreme values for better color range
    vmax = np.percentile(matrix[matrix > 0], 95) if np.any(matrix > 0) else 1.0

    fig, ax = plt.subplots(figsize=(14, 6))
    im = sns.heatmap(
        matrix, ax=ax, cmap='RdYlBu_r', vmin=0, vmax=vmax,
        cbar_kws={'label': 'KL Divergence'},
        xticklabels=False, yticklabels=False,
    )

    # Mark think/answer boundary
    for i, tb in enumerate(think_boundaries):
        if 0 < tb < n_bins:
            ax.plot(tb, i + 0.5, 'k|', markersize=4, markeredgewidth=0.8)

    # Add vertical line at median think boundary
    median_tb = int(np.median(think_boundaries))
    ax.axvline(x=median_tb, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(median_tb + 1, -0.8, 'Think/Answer\nboundary', fontsize=9,
            va='bottom', ha='left', style='italic')

    ax.set_xlabel('Token Position (binned)')
    ax.set_ylabel('Sample (sorted by thinking length)')
    ax.set_title('Per-Token Logit Shift (KL Divergence) in Thinking Chain')

    # Add a few x-tick labels
    tick_positions = np.linspace(0, n_bins, 6).astype(int)
    tick_labels = [str(int(p * max_tokens / n_bins)) for p in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    fig.tight_layout()
    path = f"{OUT_DIR}/fig1_lsr_heatmap.png"
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


# ============================================================
# Figure 2: Training Dynamics (4 subplots)
# ============================================================
def figure2_training_dynamics():
    histories = load_histories()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Collect cumulative step offset per round
    step_offset = 0
    for r_idx, h in enumerate(histories):
        steps = h['steps']
        color = ROUND_COLORS[r_idx]
        label = ROUND_LABELS[r_idx]

        x = [s.get('step', i+1) + step_offset for i, s in enumerate(steps)]
        loss = [s.get('loss', 0) for s in steps]
        lsr = [s.get('mean_lsr_raw', 0) for s in steps]
        correct = [s.get('mean_correct', 0) for s in steps]

        # Top-left: GRPO Loss
        axes[0, 0].plot(x, loss, color=color, label=label, linewidth=1.5, alpha=0.9)

        # Top-right: Mean LSR
        axes[0, 1].plot(x, lsr, color=color, label=label, linewidth=1.5, alpha=0.9)

        # Bottom-left: Training Correctness
        axes[1, 0].plot(x, correct, color=color, label=label, linewidth=1.5, alpha=0.9)

        # Bottom-right: POPE Accuracy at eval points
        evals = h.get('evals', [])
        if evals:
            ex = [e['step'] + step_offset for e in evals]
            ey = [e['pope']['acc'] * 100 for e in evals]
            axes[1, 1].plot(ex, ey, color=color, label=label, linewidth=1.5,
                            marker='*', markersize=10, alpha=0.9)

        # Add round boundary
        for ax in axes.flat:
            ax.axvline(x=step_offset + 0.5, color='gray', linestyle=':', linewidth=0.7, alpha=0.5)

        step_offset += len(steps)

    axes[0, 0].set_title('GRPO Loss')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_xlabel('Cumulative Step')
    axes[0, 0].legend(loc='best', fontsize=9)

    axes[0, 1].set_title('Mean LSR (KL Divergence)')
    axes[0, 1].set_ylabel('Mean LSR')
    axes[0, 1].set_xlabel('Cumulative Step')
    axes[0, 1].legend(loc='best', fontsize=9)

    axes[1, 0].set_title('Training Correctness')
    axes[1, 0].set_ylabel('Correctness')
    axes[1, 0].set_xlabel('Cumulative Step')
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].legend(loc='best', fontsize=9)

    axes[1, 1].set_title('POPE Accuracy Progression')
    axes[1, 1].set_ylabel('POPE Acc (%)')
    axes[1, 1].set_xlabel('Cumulative Step')
    axes[1, 1].set_ylim(85, 100)
    axes[1, 1].legend(loc='best', fontsize=9)

    fig.suptitle('Multi-Round GRPO-LSR Training Dynamics', fontsize=16, y=1.02)
    fig.tight_layout()
    path = f"{OUT_DIR}/fig2_training_dynamics.png"
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


# ============================================================
# Figure 3: Eval Progression (Dual Y-axis)
# ============================================================
def figure3_eval_progression():
    histories = load_histories()

    eval_points = []
    step_offset = 0
    for r_idx, h in enumerate(histories):
        n_steps = len(h['steps'])
        for e in h.get('evals', []):
            eval_points.append({
                'round': r_idx + 1,
                'cum_step': e['step'] + step_offset,
                'pope_acc': e['pope']['acc'] * 100,
                'gap': e['blind']['gap'] * 100,
            })
        step_offset += n_steps

    xs = [e['cum_step'] for e in eval_points]
    pope = [e['pope_acc'] for e in eval_points]
    gap = [e['gap'] for e in eval_points]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    ln1 = ax1.plot(xs, pope, 'o-', color='#2196F3', linewidth=2, markersize=7, label='POPE Acc (%)')
    ln2 = ax2.plot(xs, gap, 's--', color='#E91E63', linewidth=2, markersize=7, label='Blind Gap (pp)')

    # Find and annotate peak POPE
    peak_idx = int(np.argmax(pope))
    ax1.annotate(
        f'Peak: {pope[peak_idx]:.1f}%',
        xy=(xs[peak_idx], pope[peak_idx]),
        xytext=(xs[peak_idx] + 3, pope[peak_idx] + 1),
        fontsize=11, fontweight='bold', color='#2196F3',
        arrowprops=dict(arrowstyle='->', color='#2196F3', lw=1.5),
    )

    # Find and annotate peak Gap
    peak_gap_idx = int(np.argmax(gap))
    ax2.annotate(
        f'Peak: {gap[peak_gap_idx]:.1f}pp',
        xy=(xs[peak_gap_idx], gap[peak_gap_idx]),
        xytext=(xs[peak_gap_idx] + 3, gap[peak_gap_idx] + 1),
        fontsize=11, fontweight='bold', color='#E91E63',
        arrowprops=dict(arrowstyle='->', color='#E91E63', lw=1.5),
    )

    # Round boundaries
    step_offset = 0
    for r_idx, h in enumerate(histories):
        n_steps = len(h['steps'])
        if r_idx > 0:
            ax1.axvline(x=step_offset + 0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        mid = step_offset + n_steps / 2
        ax1.text(mid, ax1.get_ylim()[0] + 0.3, f'R{r_idx+1}', fontsize=9,
                 ha='center', color='gray', style='italic')
        step_offset += n_steps

    ax1.set_xlabel('Cumulative Training Step')
    ax1.set_ylabel('POPE Accuracy (%)', color='#2196F3')
    ax2.set_ylabel('Blind Gap (pp)', color='#E91E63')
    ax1.tick_params(axis='y', labelcolor='#2196F3')
    ax2.tick_params(axis='y', labelcolor='#E91E63')

    # Combined legend
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='lower right', fontsize=11)

    ax1.set_title('POPE Accuracy and Blind Gap Across Multi-Round GRPO-LSR Training')
    fig.tight_layout()
    path = f"{OUT_DIR}/fig3_eval_progression.png"
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


# ============================================================
# Figure 4: Reward Distribution
# ============================================================
def figure4_reward_distribution():
    histories = load_histories()

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    step_offset = 0
    for r_idx, h in enumerate(histories):
        steps = h['steps']
        color = ROUND_COLORS[r_idx]
        label = ROUND_LABELS[r_idx]

        x = np.array([s['step'] + step_offset for s in steps])
        mean_r = np.array([s['mean_reward'] for s in steps])
        std_r = np.array([s.get('reward_std', 0) for s in steps])
        lsr = np.array([s.get('mean_lsr_raw', 0) for s in steps])

        ax1.plot(x, mean_r, color=color, linewidth=1.5, label=label, alpha=0.9)
        ax1.fill_between(x, mean_r - std_r, mean_r + std_r, color=color, alpha=0.15)

        ax2.plot(x, lsr, color=color, linewidth=1.2, linestyle='--', alpha=0.6)

        # Round boundary
        if r_idx > 0:
            ax1.axvline(x=step_offset + 0.5, color='gray', linestyle=':', linewidth=0.7, alpha=0.5)

        step_offset += len(steps)

    ax1.set_xlabel('Cumulative Training Step')
    ax1.set_ylabel('Mean Reward (solid, +/- std shaded)', color='#333333')
    ax2.set_ylabel('Mean LSR Raw (dashed)', color='#888888')
    ax2.tick_params(axis='y', labelcolor='#888888')

    ax1.legend(loc='upper left', fontsize=10)
    ax1.set_title('Reward Distribution and LSR Signal Across Training')

    fig.tight_layout()
    path = f"{OUT_DIR}/fig4_reward_distribution.png"
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


# ============================================================
# Figure 5: Method Comparison Bar Chart
# ============================================================
def figure5_method_comparison():
    methods = [
        'Baseline\n(Thinking)',
        'Steered\n(\u03b1=5)',
        'BoN+SFT\n(Instruct)',
        'GRPO-LSR\nBest (Think)',
    ]
    pope_acc = [91.7, 89.7, 88.0, 95.0]
    blind_gap = [40.0, 38.0, 38.0, 44.0]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, pope_acc, width, label='POPE Acc (%)',
                   color='#2196F3', edgecolor='white', linewidth=0.8, zorder=3)
    bars2 = ax.bar(x + width/2, blind_gap, width, label='Blind Gap (pp)',
                   color='#E91E63', edgecolor='white', linewidth=0.8, zorder=3)

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Highlight best
    bars1[3].set_edgecolor('#0D47A1')
    bars1[3].set_linewidth(2.5)
    bars2[3].set_edgecolor('#880E4F')
    bars2[3].set_linewidth(2.5)

    ax.set_ylabel('Score')
    ax.set_title('VIGIL Method Comparison: POPE Accuracy and Blind Gap')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 105)
    ax.legend(loc='upper left', fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))

    # Add a horizontal baseline reference
    ax.axhline(y=91.7, color='#2196F3', linestyle=':', linewidth=0.8, alpha=0.4)
    ax.axhline(y=40.0, color='#E91E63', linestyle=':', linewidth=0.8, alpha=0.4)

    fig.tight_layout()
    path = f"{OUT_DIR}/fig5_method_comparison.png"
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    figure1_lsr_heatmap()
    figure2_training_dynamics()
    figure3_eval_progression()
    figure4_reward_distribution()
    figure5_method_comparison()
    print(f"\nAll figures saved to {OUT_DIR}/")
