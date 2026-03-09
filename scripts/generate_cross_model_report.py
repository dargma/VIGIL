"""
VIGIL Cross-Model Comparison Report Generator.

Generates publication-quality figures comparing all models and methods.
"""

import os, sys, json
import numpy as np
from pathlib import Path
from datetime import datetime

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def generate_all():
    output_dir = Path("lab/reports/multimodel")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──
    # All results from experiments
    data = {
        "Qwen3-VL-2B": {
            "Baseline": {"acc": 87.4, "f1": 87.2, "precision": 88.7, "recall": 85.7, "gap": 37.4},
            "Steered α=5": {"acc": 88.0, "f1": 88.0, "precision": 88.0, "recall": 88.0, "gap": 38.0},
            "BoN+SFT": {"acc": 87.8, "f1": 87.4, "precision": 90.3, "recall": 84.7, "gap": 37.8},
            "DAPO": {"acc": 87.8, "f1": 87.4, "precision": 90.3, "recall": 84.6, "gap": 37.8},
            "BoN+SFT (POPE)": {"acc": 88.0, "f1": 87.7, "precision": 89.9, "recall": 85.6, "gap": 38.0},
            "BoN+SFT+Steer": {"acc": 87.4, "f1": 87.0, "precision": 90.0, "recall": 84.2, "gap": 37.4},
        },
        "InternVL3.5-1B": {
            "Baseline": {"acc": 78.2, "f1": 80.8, "precision": 72.1, "recall": 92.0, "gap": 28.2},
            "Steered α=1": {"acc": 78.4, "f1": 81.0, "precision": 72.3, "recall": 92.0, "gap": 28.4},
            "BoN+SFT R1": {"acc": 82.6, "f1": 83.7, "precision": 78.8, "recall": 89.2, "gap": 32.6},
            "BoN+SFT R2": {"acc": 83.4, "f1": 84.3, "precision": 79.9, "recall": 89.2, "gap": 33.4},
            "BoN+SFT (POPE)": {"acc": 82.4, "f1": 83.6, "precision": 78.3, "recall": 89.6, "gap": 32.4},
        },
    }

    # ── Figure 1: Accuracy comparison across models ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    plt.style.use("seaborn-v0_8-whitegrid")

    for idx, (model_name, methods) in enumerate(data.items()):
        ax = axes[idx]
        labels = list(methods.keys())
        accs = [methods[l]["acc"] for l in labels]
        gaps = [methods[l]["gap"] for l in labels]

        x = np.arange(len(labels))
        width = 0.35

        bars1 = ax.bar(x - width/2, accs, width, label="POPE Acc", color="#2196F3", alpha=0.85)
        bars2 = ax.bar(x + width/2, gaps, width, label="Blind Gap", color="#FF5722", alpha=0.85)

        ax.set_ylabel("Score (%)" if idx == 0 else "", fontsize=12)
        ax.set_title(model_name, fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 100)

        for bar in bars1:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
        for bar in bars2:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    fig.suptitle("VIGIL: Cross-Model POPE Evaluation", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig(output_dir / "cross_model_acc_gap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[1] cross_model_acc_gap.png")

    # ── Figure 2: Precision improvement (anti-hallucination) ──
    fig, ax = plt.subplots(figsize=(12, 6))

    models = ["Qwen3-VL-2B", "InternVL3.5-1B"]
    methods_order = ["Baseline", "Steered", "BoN+SFT (best)"]
    colors = {"Baseline": "#607D8B", "Steered": "#03A9F4", "BoN+SFT (best)": "#4CAF50"}

    x = np.arange(len(models))
    width = 0.25
    offsets = [-width, 0, width]

    for i, method in enumerate(methods_order):
        vals = []
        for model_name in models:
            if method == "Steered":
                key = "Steered α=5" if "5" in str(data[model_name].keys()) else "Steered α=1"
                vals.append(data[model_name].get(key, data[model_name].get("Steered α=1", {})).get("precision", 0))
            elif method == "BoN+SFT (best)":
                # Use best BoN+SFT variant per model
                bon_key = "BoN+SFT R2" if model_name == "InternVL3.5-1B" else "BoN+SFT"
                vals.append(data[model_name].get(bon_key, {}).get("precision", 0))
            else:
                vals.append(data[model_name].get(method, {}).get("precision", 0))

        bars = ax.bar(x + offsets[i], vals, width, label=method, color=colors[method], alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=10)

    ax.set_ylabel("Precision (%)", fontsize=12)
    ax.set_title("POPE Precision: Anti-Hallucination Improvement", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim(60, 100)

    # Add improvement annotations
    for i, model_name in enumerate(models):
        baseline_p = data[model_name]["Baseline"]["precision"]
        bon_key = "BoN+SFT R2" if model_name == "InternVL3.5-1B" else "BoN+SFT"
        bon_p = data[model_name][bon_key]["precision"]
        delta = bon_p - baseline_p
        ax.annotate(f"+{delta:.1f}pp",
                    xy=(i + width, bon_p), xytext=(i + width + 0.15, bon_p + 2),
                    fontsize=11, color="green", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="green", lw=1.5))

    plt.tight_layout()
    fig.savefig(output_dir / "precision_improvement.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[2] precision_improvement.png")

    # ── Figure 3: Blind Gap comparison ──
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, color in [("Qwen3-VL-2B", "#2196F3"), ("InternVL3.5-1B", "#FF9800")]:
        methods_sorted = sorted(data[model_name].items(), key=lambda x: x[1]["gap"])
        labels = [m for m, _ in methods_sorted]
        gaps = [v["gap"] for _, v in methods_sorted]

        ax.barh(labels, gaps, height=0.35, color=color, alpha=0.8, label=model_name)

        for j, (label, gap) in enumerate(zip(labels, gaps)):
            ax.annotate(f"{gap:.1f}pp", xy=(gap + 0.3, j), va="center", fontsize=10)

    ax.set_xlabel("Blind Gap (pp)", fontsize=12)
    ax.set_title("Image Dependence: Blind Gap Across Models & Methods", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    plt.tight_layout()
    fig.savefig(output_dir / "blind_gap_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[3] blind_gap_comparison.png")

    # ── Figure 4: Cohen's d comparison ──
    cal_qwen = Path("checkpoints/calibration/qwen3_vl_2b/calibration_meta.json")
    cal_intern = Path("checkpoints/calibration/internvl3_5_1b/calibration.json")

    if cal_qwen.exists() and cal_intern.exists():
        with open(cal_qwen) as f:
            qwen_cal = json.load(f)
        with open(cal_intern) as f:
            intern_cal = json.load(f)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Qwen3 top heads
        if "head_scores" in qwen_cal:
            hs = qwen_cal["head_scores"]
            # Handle both formats: direct float or dict with cohens_d
            qwen_ds = []
            for k, v in hs.items():
                d_val = v if isinstance(v, (int, float)) else v.get("cohens_d", 0)
                qwen_ds.append((k.replace("_", ","), d_val))
            qwen_ds.sort(key=lambda x: x[1], reverse=True)
            qwen_ds = qwen_ds[:20]
        else:
            qwen_ds = []

        intern_ds = sorted(
            [(k, v["cohens_d"]) for k, v in intern_cal.get("head_scores", {}).items()],
            key=lambda x: x[1], reverse=True
        )[:20]

        for ax, head_ds, title, color in [
            (axes[0], qwen_ds, "Qwen3-VL-2B", "#2196F3"),
            (axes[1], intern_ds, "InternVL3.5-1B", "#FF9800"),
        ]:
            if head_ds:
                labels = [f"L{k.split(',')[0]}H{k.split(',')[1]}" for k, _ in head_ds]
                ds = [d for _, d in head_ds]
                ax.barh(labels[::-1], ds[::-1], color=color, alpha=0.8)
                ax.set_xlabel("Cohen's d", fontsize=12)
                ax.set_title(f"{title}: Top 20 Vision Heads", fontsize=13, fontweight="bold")

        plt.tight_layout()
        fig.savefig(output_dir / "cohens_d_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[4] cohens_d_comparison.png")

    # ── Summary table ──
    print(f"\n{'='*90}")
    print(f"VIGIL Cross-Model Final Results")
    print(f"{'='*90}")
    print(f"{'Model':<18} {'Method':<20} {'Acc':>6} {'F1':>6} {'P':>6} {'R':>6} {'Gap':>7}")
    print("-" * 90)
    for model_name, methods in data.items():
        for method_name, m in methods.items():
            print(f"{model_name:<18} {method_name:<20} {m['acc']:>5.1f}% {m['f1']:>5.1f}% "
                  f"{m['precision']:>5.1f}% {m['recall']:>5.1f}% {m['gap']:>5.1f}pp")
        print("-" * 90)
    print("=" * 90)


if __name__ == "__main__":
    generate_all()
