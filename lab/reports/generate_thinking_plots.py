#!/usr/bin/env python3
"""Generate thinking mode drift analysis plots.

Reads from lab/reports/thinking_mode/results_*.json and produces:
  1. Vision head activation drift curve (baseline vs BoN R1 vs R2)
  2. POPE thinking mode comparison (bar chart)
  3. Thinking chain length histogram

Style: seaborn-whitegrid, font 12, figsize (10,6).
"""

import json
import os
import glob
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not installed. Skipping plot generation.")

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "thinking_mode")
OUTPUT_DIR = SCRIPT_DIR

# Style config
STYLE = "seaborn-v0_8-whitegrid"
FONT_SIZE = 12
FIGSIZE = (10, 6)
COLORS = {
    "baseline": "#1f77b4",
    "steered": "#ff7f0e",
    "bon_r1": "#2ca02c",
    "bon_r2": "#d62728",
    "bon_r1_steered": "#9467bd",
}
LABELS = {
    "baseline": "Baseline (greedy)",
    "steered": "Steered (alpha=1.0)",
    "bon_r1": "BoN+SFT Round 1",
    "bon_r2": "BoN+SFT Round 2",
    "bon_r1_steered": "BoN R1 + Steering",
}


def load_results():
    """Load all results_*.json files from thinking_mode directory."""
    results = {}
    if not os.path.isdir(RESULTS_DIR):
        print(f"Results directory not found: {RESULTS_DIR}")
        print("Creating directory and using placeholder data for plot layout.")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        return None
    pattern = os.path.join(RESULTS_DIR, "results_*.json")
    files = glob.glob(pattern)
    if not files:
        print(f"No results files found matching {pattern}")
        print("Using placeholder data for plot layout.")
        return None
    for fpath in sorted(files):
        key = os.path.basename(fpath).replace("results_", "").replace(".json", "")
        with open(fpath) as f:
            results[key] = json.load(f)
    return results


def get_placeholder_data():
    """Generate placeholder data for layout verification."""
    np.random.seed(42)
    n_tokens = 200

    # Drift curves: exponential decay for baseline, slower for steered/trained
    positions = np.arange(n_tokens)
    data = {
        "baseline": {
            "drift_curve": {
                "positions": positions.tolist(),
                "activations": (8.0 * np.exp(-0.015 * positions)
                                + np.random.normal(0, 0.3, n_tokens)).tolist(),
            },
            "pope_acc": 77.0,
            "chain_lengths": np.random.lognormal(4.0, 0.6, 50).astype(int).tolist(),
        },
        "steered": {
            "drift_curve": {
                "positions": positions.tolist(),
                "activations": (8.0 * np.exp(-0.005 * positions)
                                + np.random.normal(0, 0.3, n_tokens)).tolist(),
            },
            "pope_acc": 78.0,
            "chain_lengths": np.random.lognormal(4.1, 0.6, 50).astype(int).tolist(),
        },
        "bon_r1": {
            "drift_curve": {
                "positions": positions.tolist(),
                "activations": (8.5 * np.exp(-0.008 * positions)
                                + np.random.normal(0, 0.3, n_tokens)).tolist(),
            },
            "pope_acc": 85.5,
            "chain_lengths": np.random.lognormal(4.0, 0.5, 50).astype(int).tolist(),
        },
        "bon_r2": {
            "drift_curve": {
                "positions": positions.tolist(),
                "activations": (9.0 * np.exp(-0.006 * positions)
                                + np.random.normal(0, 0.3, n_tokens)).tolist(),
            },
            "pope_acc": 87.0,
            "chain_lengths": np.random.lognormal(3.9, 0.5, 50).astype(int).tolist(),
        },
        "bon_r1_steered": {
            "drift_curve": {
                "positions": positions.tolist(),
                "activations": (9.5 * np.exp(-0.003 * positions)
                                + np.random.normal(0, 0.3, n_tokens)).tolist(),
            },
            "pope_acc": 88.0,
            "chain_lengths": np.random.lognormal(4.2, 0.55, 50).astype(int).tolist(),
        },
    }
    return data


def plot_drift_curve(data, output_path):
    """Plot vision head activation vs token position (Figure 1 candidate).

    Shows how vision head activation decays over the thinking chain
    for different conditions. The core visualization of visual attention drift.
    """
    plt.style.use(STYLE)
    plt.rcParams.update({"font.size": FONT_SIZE})
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for key in ["baseline", "steered", "bon_r1", "bon_r2", "bon_r1_steered"]:
        if key not in data:
            continue
        curve = data[key].get("drift_curve", {})
        positions = curve.get("positions", [])
        activations = curve.get("activations", [])
        if not positions or not activations:
            continue

        positions = np.array(positions)
        activations = np.array(activations)

        # Smooth with rolling average for readability
        window = max(1, len(activations) // 20)
        if len(activations) > window:
            smoothed = np.convolve(activations, np.ones(window) / window, mode="valid")
            pos_smoothed = positions[:len(smoothed)]
        else:
            smoothed = activations
            pos_smoothed = positions

        color = COLORS.get(key, "#333333")
        label = LABELS.get(key, key)

        # Plot raw data as faint background
        ax.plot(positions, activations, color=color, alpha=0.15, linewidth=0.5)
        # Plot smoothed curve
        ax.plot(pos_smoothed, smoothed, color=color, alpha=0.9, linewidth=2.0, label=label)

    ax.set_xlabel("Token Position in Thinking Chain", fontsize=FONT_SIZE)
    ax.set_ylabel("Mean Vision Head Activation (Top-K)", fontsize=FONT_SIZE)
    ax.set_title("Visual Attention Drift in Thinking Mode", fontsize=FONT_SIZE + 2, fontweight="bold")
    ax.legend(fontsize=FONT_SIZE - 1, loc="upper right")

    # Add annotation for the drift region
    ax.axvline(x=50, color="gray", linestyle="--", alpha=0.3)
    ax.text(52, ax.get_ylim()[1] * 0.95, "<think> tokens", fontsize=9, color="gray", alpha=0.6)

    ax.set_xlim(0, None)
    ax.set_ylim(0, None)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_pope_thinking_comparison(data, output_path):
    """Bar chart comparing POPE accuracy across conditions in thinking mode."""
    plt.style.use(STYLE)
    plt.rcParams.update({"font.size": FONT_SIZE})
    fig, ax = plt.subplots(figsize=FIGSIZE)

    conditions = []
    accuracies = []
    colors = []

    for key in ["baseline", "steered", "bon_r1", "bon_r2", "bon_r1_steered"]:
        if key not in data:
            continue
        acc = data[key].get("pope_acc", 0)
        conditions.append(LABELS.get(key, key))
        accuracies.append(acc)
        colors.append(COLORS.get(key, "#333333"))

    if not conditions:
        print("No POPE accuracy data found. Skipping plot.")
        return

    x = np.arange(len(conditions))
    bars = ax.bar(x, accuracies, color=colors, width=0.6, edgecolor="white", linewidth=0.5)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=FONT_SIZE - 1, fontweight="bold")

    # Add baseline reference line
    if accuracies:
        ax.axhline(y=accuracies[0], color=COLORS["baseline"], linestyle="--", alpha=0.3, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=20, ha="right", fontsize=FONT_SIZE - 1)
    ax.set_ylabel("POPE Adversarial Accuracy (%)", fontsize=FONT_SIZE)
    ax.set_title("POPE Accuracy: Thinking Mode Comparison", fontsize=FONT_SIZE + 2, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_chain_length_histogram(data, output_path):
    """Histogram of thinking chain lengths across conditions."""
    plt.style.use(STYLE)
    plt.rcParams.update({"font.size": FONT_SIZE})
    fig, ax = plt.subplots(figsize=FIGSIZE)

    all_lengths = []
    plot_keys = []

    for key in ["baseline", "steered", "bon_r1", "bon_r2", "bon_r1_steered"]:
        if key not in data:
            continue
        lengths = data[key].get("chain_lengths", [])
        if not lengths:
            continue
        all_lengths.append(np.array(lengths))
        plot_keys.append(key)

    if not all_lengths:
        print("No chain length data found. Skipping plot.")
        return

    # Determine shared bin edges
    all_flat = np.concatenate(all_lengths)
    bins = np.linspace(0, np.percentile(all_flat, 98), 30)

    for lengths, key in zip(all_lengths, plot_keys):
        color = COLORS.get(key, "#333333")
        label = LABELS.get(key, key)
        ax.hist(lengths, bins=bins, alpha=0.4, color=color, label=label, edgecolor=color, linewidth=0.5)

    # Add mean lines
    for lengths, key in zip(all_lengths, plot_keys):
        color = COLORS.get(key, "#333333")
        mean_len = np.mean(lengths)
        ax.axvline(x=mean_len, color=color, linestyle="--", linewidth=1.5, alpha=0.7)
        ax.text(mean_len + 2, ax.get_ylim()[1] * 0.9, f"mean={mean_len:.0f}",
                color=color, fontsize=9, alpha=0.8)

    ax.set_xlabel("Thinking Chain Length (tokens)", fontsize=FONT_SIZE)
    ax.set_ylabel("Count", fontsize=FONT_SIZE)
    ax.set_title("Thinking Chain Length Distribution", fontsize=FONT_SIZE + 2, fontweight="bold")
    ax.legend(fontsize=FONT_SIZE - 1)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    if not HAS_MPL:
        print("Cannot generate plots without matplotlib. Install with: pip install matplotlib")
        return

    # Load real data or fall back to placeholder
    results = load_results()
    if results is None:
        print("Using placeholder data (replace with real results for publication).")
        results = get_placeholder_data()
        is_placeholder = True
    else:
        is_placeholder = False

    suffix = "_placeholder" if is_placeholder else ""

    # Generate all 3 plots
    plot_drift_curve(
        results,
        os.path.join(OUTPUT_DIR, f"fig_thinking_drift_curve{suffix}.png"),
    )
    plot_pope_thinking_comparison(
        results,
        os.path.join(OUTPUT_DIR, f"fig_thinking_pope_comparison{suffix}.png"),
    )
    plot_chain_length_histogram(
        results,
        os.path.join(OUTPUT_DIR, f"fig_thinking_chain_lengths{suffix}.png"),
    )

    print(f"\nAll thinking mode plots generated in {OUTPUT_DIR}")
    if is_placeholder:
        print("NOTE: These use placeholder data. Run thinking mode experiments and save")
        print(f"results to {RESULTS_DIR}/results_<condition>.json to get real plots.")
        print("Expected JSON format per condition:")
        print('  {"drift_curve": {"positions": [...], "activations": [...]},')
        print('   "pope_acc": 85.5,')
        print('   "chain_lengths": [102, 87, 156, ...]}')


if __name__ == "__main__":
    main()
