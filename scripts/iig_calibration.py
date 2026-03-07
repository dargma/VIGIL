"""
VIGIL Block 0 — IIG Lambda Calibration.

Computes IIG on GQA calibration data, auto-determines lambda,
generates diagnostic plots. Gate: IIG positive ratio >= 60%.
"""
import sys, os, time, json
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

from src.model_registry import load_model, make_chat_prompt
from src.data_loader import load_gqa_balanced_val, load_pope
from src.iig import compute_iig, calibrate_lambda, BLACK_IMAGE

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 12, "figure.dpi": 150})

RESULTS_DIR = Path("lab/results/iig_calibration")
REPORTS_DIR = Path("lab/reports/iig_calibration")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    t0 = time.time()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"{'='*60}\nVIGIL Block 0: IIG Lambda Calibration\n{datetime.now().isoformat()}\n{'='*60}")

    # Load model
    print("\n--- Loading Qwen3-VL-2B-Instruct ---")
    model_info = load_model("qwen3_vl_2b", dtype=torch.float16, device="auto")

    # Load calibration data — need samples WITH images
    # GQA disk cache lacks images, so use POPE (has embedded images)
    # POPE is yes/no which is simpler, but IIG still measures log-prob differential
    print("\n--- Loading calibration data (POPE, has images) ---")
    calib_data = load_pope("adversarial", limit=500)
    calib_with_images = [s for s in calib_data if s.get("image") is not None]
    print(f"Samples with images: {len(calib_with_images)}/{len(calib_data)}")

    if len(calib_with_images) < 50:
        print("ERROR: Too few samples with images. Check data loader.")
        return

    # Run IIG calibration
    print(f"\n--- Computing IIG on {len(calib_with_images)} samples ---")
    lam, all_iig = calibrate_lambda(model_info, calib_with_images, max_samples=500)

    if not all_iig:
        print("FATAL: No IIG values computed.")
        return

    all_iig = np.array(all_iig)
    positives = all_iig[all_iig > 0]
    negatives = all_iig[all_iig <= 0]
    pos_ratio = len(positives) / len(all_iig) * 100

    # === GATE CHECK ===
    gate_pass = pos_ratio >= 60.0
    print(f"\n{'='*60}")
    print(f"GATE CHECK: IIG positive ratio = {pos_ratio:.1f}% (threshold: 60%)")
    print(f"Result: {'PASS' if gate_pass else 'FAIL'}")
    print(f"{'='*60}")

    # Save results
    results = {
        "timestamp": ts,
        "model": "qwen3_vl_2b",
        "n_samples": len(all_iig),
        "n_positive": int(len(positives)),
        "n_negative": int(len(negatives)),
        "pos_ratio": float(pos_ratio),
        "mean_all": float(np.mean(all_iig)),
        "std_all": float(np.std(all_iig)),
        "median_all": float(np.median(all_iig)),
        "mean_positive": float(np.mean(positives)) if len(positives) > 0 else 0.0,
        "std_positive": float(np.std(positives)) if len(positives) > 0 else 0.0,
        "mean_negative": float(np.mean(negatives)) if len(negatives) > 0 else 0.0,
        "lambda": float(lam),
        "gate_pass": gate_pass,
        "iig_values": all_iig.tolist(),
    }
    results_path = RESULTS_DIR / f"iig_calibration_{ts}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {results_path}")

    # === Plot 1: IIG Distribution Histogram ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(all_iig, bins=50, color="#5B8DB8", edgecolor="white", alpha=0.8)
    ax1.axvline(0, color="red", linestyle="--", linewidth=2, label="IIG=0")
    ax1.axvline(np.mean(all_iig), color="orange", linestyle="-", linewidth=2,
                label=f"Mean={np.mean(all_iig):.3f}")
    ax1.set_xlabel("IIG Value")
    ax1.set_ylabel("Count")
    ax1.set_title(f"IIG Distribution (N={len(all_iig)})\n"
                  f"Positive: {len(positives)} ({pos_ratio:.0f}%) | "
                  f"Negative: {len(negatives)} ({100-pos_ratio:.0f}%)")
    ax1.legend()
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Cumulative distribution
    sorted_iig = np.sort(all_iig)
    cdf = np.arange(1, len(sorted_iig) + 1) / len(sorted_iig)
    ax2.plot(sorted_iig, cdf, color="#E8834A", linewidth=2)
    ax2.axvline(0, color="red", linestyle="--", alpha=0.5)
    ax2.axhline(1 - pos_ratio/100, color="gray", linestyle=":", alpha=0.5,
                label=f"Negative fraction: {100-pos_ratio:.0f}%")
    ax2.set_xlabel("IIG Value")
    ax2.set_ylabel("CDF")
    ax2.set_title(f"IIG CDF | lambda={lam:.2f}")
    ax2.legend()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle(f"Block 0: IIG Calibration — {'PASS' if gate_pass else 'FAIL'}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    hist_path = REPORTS_DIR / f"iig_distribution_{ts}.png"
    fig.savefig(hist_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {hist_path}")

    # === Plot 2: IIG by answer length ===
    fig, ax = plt.subplots(figsize=(8, 5))
    # Bucket by answer length
    lengths = []
    iig_vals = []
    for s, v in zip(calib_with_images[:len(all_iig)], all_iig):
        lengths.append(len(s["answer"].split()))
        iig_vals.append(v)

    if lengths:
        lengths = np.array(lengths)
        iig_vals = np.array(iig_vals)
        # Bin into groups
        bins = [0, 1, 2, 3, 5, 100]
        labels = ["1 word", "2 words", "3 words", "4-5 words", "6+ words"]
        bin_means = []
        bin_labels_used = []
        for i in range(len(bins) - 1):
            mask = (lengths >= bins[i] + (1 if i > 0 else 1)) & (lengths <= bins[i + 1])
            if i == 0:
                mask = lengths == 1
            if mask.sum() > 0:
                bin_means.append(np.mean(iig_vals[mask]))
                bin_labels_used.append(f"{labels[i]}\n(n={mask.sum()})")

        if bin_means:
            ax.bar(range(len(bin_means)), bin_means, color="#6AAF6A", edgecolor="white")
            ax.set_xticks(range(len(bin_means)))
            ax.set_xticklabels(bin_labels_used)
            ax.axhline(0, color="red", linestyle="--", alpha=0.5)
            ax.set_ylabel("Mean IIG")
            ax.set_title("IIG by Answer Length")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.tight_layout()
    len_path = REPORTS_DIR / f"iig_by_length_{ts}.png"
    fig.savefig(len_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {len_path}")

    # === Summary ===
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"IIG Calibration Summary")
    print(f"{'='*60}")
    print(f"  Samples:       {len(all_iig)}")
    print(f"  Positive:      {len(positives)} ({pos_ratio:.1f}%)")
    print(f"  Mean (all):    {np.mean(all_iig):.4f}")
    print(f"  Mean (pos):    {np.mean(positives):.4f}" if len(positives) > 0 else "  Mean (pos):    N/A")
    print(f"  Std (pos):     {np.std(positives):.4f}" if len(positives) > 0 else "  Std (pos):     N/A")
    print(f"  Lambda:        {lam:.4f}")
    print(f"  Gate (>=60%):  {'PASS' if gate_pass else 'FAIL'}")
    print(f"  Time:          {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*60}")

    if gate_pass:
        print("\n>>> IIG signal confirmed. Proceed to Block 1 (Minimal GRPO).")
    else:
        print("\n>>> IIG signal too weak. Debug compute_iig or try different reference condition.")


if __name__ == "__main__":
    main()
