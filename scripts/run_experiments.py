"""
VIGIL Experiment Runner — Blocks 1-3 remaining experiments.

Runs: blind test, proportional steering, alpha/K/layer sweeps, DeepStack test,
per-sample analysis. All results saved to lab/results/ and lab/reports/.
"""

import sys
import os
os.environ["PYTHONUNBUFFERED"] = "1"
import json
import time
import gc
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_registry import load_model, make_chat_prompt
from src.calibrator import CalibrationResult
from src.steerer import ActivationSteerer, SteeringHook
from src.blind_test import run_blind_test, save_blind_test_results
from src.data_loader import load_pope

RESULTS_DIR = Path(__file__).parent.parent / "lab" / "results"
REPORTS_DIR = Path(__file__).parent.parent / "lab" / "reports"
CAL_DIR = Path(__file__).parent.parent / "checkpoints" / "calibration" / "qwen3_vl_2b"


def generate_fn(model_info, question, image):
    """Generate short answer from model."""
    inputs = make_chat_prompt(model_info, question, image)
    with torch.no_grad():
        output_ids = model_info["model"].generate(
            **inputs, max_new_tokens=20, do_sample=False,
        )
    input_len = inputs["input_ids"].shape[1]
    pred = model_info["tokenizer"].decode(output_ids[0][input_len:], skip_special_tokens=True)
    return pred.strip()


def check_correct(prediction, ground_truth, q_type="yesno"):
    pred = prediction.strip().lower()
    gt = ground_truth.strip().lower()
    if q_type == "yesno":
        has_yes = "yes" in pred
        has_no = "no" in pred
        if has_yes and has_no:
            pred_yn = "yes" if pred.index("yes") < pred.index("no") else "no"
        elif has_yes:
            pred_yn = "yes"
        elif has_no:
            pred_yn = "no"
        else:
            pred_yn = ""
        return pred_yn == gt
    return pred.startswith(gt[:5]) if gt else False


def eval_pope(model_info, samples, label="baseline"):
    """Evaluate on POPE samples, return accuracy and per-sample results."""
    correct = 0
    total = 0
    per_sample = []
    for i, s in enumerate(samples):
        try:
            pred = generate_fn(model_info, s["question"], s.get("image"))
            is_correct = check_correct(pred, s["answer"], s.get("type", "yesno"))
            per_sample.append({
                "idx": i, "pred": pred, "gt": s["answer"],
                "correct": is_correct, "question": s["question"][:80],
            })
            if is_correct:
                correct += 1
            total += 1
            if (i + 1) % 50 == 0:
                print(f"  [{label}] {i+1}/{len(samples)}: {correct/total*100:.1f}%")
        except Exception as e:
            print(f"  [{label}] Skip {i}: {e}")
    acc = correct / total * 100 if total > 0 else 0.0
    print(f"  [{label}] Final: {acc:.1f}% ({correct}/{total})")
    return acc, per_sample


def save_result(data, subdir, name):
    out = RESULTS_DIR / subdir
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = out / f"{name}_{ts}.json"
    with open(fname, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {fname}")
    return fname


def make_steerer_with_k(model_info, cal, top_k, steer_start=4):
    """Create steerer with only top-K heads."""
    # Create a modified calibration with fewer heads
    sub_cal = CalibrationResult(
        steering_vectors=cal.steering_vectors,
        head_scores=cal.head_scores,
        top_heads=cal.top_heads[:top_k],
        n_correct=cal.n_correct,
        n_incorrect=cal.n_incorrect,
    )
    return ActivationSteerer(model_info, sub_cal, steer_layers_start=steer_start)


def make_steerer_layer_range(model_info, cal, layer_start, layer_end):
    """Create steerer restricted to a layer range."""
    filtered_heads = [(li, hi) for li, hi in cal.top_heads if layer_start <= li <= layer_end]
    sub_cal = CalibrationResult(
        steering_vectors=cal.steering_vectors,
        head_scores=cal.head_scores,
        top_heads=filtered_heads,
        n_correct=cal.n_correct,
        n_incorrect=cal.n_incorrect,
    )
    return ActivationSteerer(model_info, sub_cal, steer_layers_start=layer_start)


# ============================================================
# EXPERIMENTS
# ============================================================

def exp_blind_test(model_info, pope_adv, cal=None, steered=False):
    """Blind test: baseline or steered. PV4 = compare both."""
    mode = "steered" if steered else "baseline"
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: Blind Test ({mode}, 500 POPE Adversarial)")
    print(f"{'='*60}")

    steerer = None
    if steered and cal is not None:
        steerer = ActivationSteerer(model_info, cal, steer_layers_start=4)
        steerer.steer(alpha=1.0)

    results = run_blind_test(model_info, pope_adv, generate_fn, max_samples=500)
    results["mode"] = mode
    save_blind_test_results(results, str(RESULTS_DIR / "blind_test"))

    if steerer:
        steerer.cleanup()

    return results


def exp_proportional_steering(model_info, cal, pope_adv_200):
    """Block 2A extension: proportional vs uniform steering."""
    print("\n" + "="*60)
    print("EXPERIMENT: Proportional vs Uniform Steering (200 POPE-A)")
    print("="*60)

    # Baseline (no steering)
    acc_base, ps_base = eval_pope(model_info, pope_adv_200, "baseline")

    # Uniform alpha=1.0
    steerer = ActivationSteerer(model_info, cal, steer_layers_start=4)
    steerer.steer(alpha=1.0)
    acc_uni, ps_uni = eval_pope(model_info, pope_adv_200, "uniform_1.0")
    steerer.cleanup()

    # Proportional alpha=1.0
    steerer = ActivationSteerer(model_info, cal, steer_layers_start=4)
    steerer.steer_proportional(global_alpha=1.0)
    acc_prop, ps_prop = eval_pope(model_info, pope_adv_200, "proportional_1.0")
    steerer.cleanup()

    # Proportional alpha=2.0
    steerer = ActivationSteerer(model_info, cal, steer_layers_start=4)
    steerer.steer_proportional(global_alpha=2.0)
    acc_prop2, ps_prop2 = eval_pope(model_info, pope_adv_200, "proportional_2.0")
    steerer.cleanup()

    results = {
        "baseline": acc_base,
        "uniform_1.0": acc_uni,
        "proportional_1.0": acc_prop,
        "proportional_2.0": acc_prop2,
        "delta_uni": acc_uni - acc_base,
        "delta_prop": acc_prop - acc_base,
        "delta_prop2": acc_prop2 - acc_base,
        "n_samples": len(pope_adv_200),
    }
    save_result(results, "proportional", "proportional_vs_uniform")

    print(f"\n  Summary:")
    print(f"    Baseline:          {acc_base:.1f}%")
    print(f"    Uniform α=1.0:     {acc_uni:.1f}% (Δ={acc_uni-acc_base:+.1f})")
    print(f"    Proportional α=1.0:{acc_prop:.1f}% (Δ={acc_prop-acc_base:+.1f})")
    print(f"    Proportional α=2.0:{acc_prop2:.1f}% (Δ={acc_prop2-acc_base:+.1f})")
    return results


def exp_alpha_sweep(model_info, cal, pope_adv_100):
    """Block 3: Alpha sweep."""
    print("\n" + "="*60)
    print("EXPERIMENT: Alpha Sweep [0.5, 1.0, 2.0, 3.0, 5.0] (100 POPE-A)")
    print("="*60)

    # Baseline
    acc_base, _ = eval_pope(model_info, pope_adv_100, "baseline")

    alphas = [0.5, 1.0, 2.0, 3.0, 5.0]
    results = {"baseline": acc_base, "alphas": {}}

    for alpha in alphas:
        steerer = ActivationSteerer(model_info, cal, steer_layers_start=4)
        steerer.steer(alpha=alpha)
        acc, _ = eval_pope(model_info, pope_adv_100, f"alpha={alpha}")
        steerer.cleanup()
        results["alphas"][str(alpha)] = acc
        print(f"    α={alpha}: {acc:.1f}% (Δ={acc-acc_base:+.1f})")

    results["n_samples"] = len(pope_adv_100)
    save_result(results, "sweeps", "alpha_sweep")
    return results


def exp_k_sweep(model_info, cal, pope_adv_100):
    """Block 3: K (number of heads) sweep."""
    print("\n" + "="*60)
    print("EXPERIMENT: K Sweep [1, 3, 5, 8, 16, 20] (100 POPE-A)")
    print("="*60)

    # Baseline
    acc_base, _ = eval_pope(model_info, pope_adv_100, "baseline")

    ks = [1, 3, 5, 8, 16, 20]
    results = {"baseline": acc_base, "ks": {}}

    for k in ks:
        steerer = make_steerer_with_k(model_info, cal, top_k=k, steer_start=4)
        steerer.steer(alpha=1.0)
        acc, _ = eval_pope(model_info, pope_adv_100, f"K={k}")
        steerer.cleanup()
        results["ks"][str(k)] = acc
        print(f"    K={k}: {acc:.1f}% (Δ={acc-acc_base:+.1f})")

    results["n_samples"] = len(pope_adv_100)
    save_result(results, "sweeps", "k_sweep")
    return results


def exp_deepstack_test(model_info, cal, pope_adv_100):
    """Block 3: DeepStack test — steer all layers vs 4+ vs 1-3 only."""
    print("\n" + "="*60)
    print("EXPERIMENT: DeepStack Test (100 POPE-A)")
    print("="*60)

    acc_base, _ = eval_pope(model_info, pope_adv_100, "baseline")

    configs = {
        "all_layers": (0, 27),
        "layers_4_plus": (4, 27),
        "layers_1_3_only": (0, 3),
        "layers_8_plus": (8, 27),
    }
    results = {"baseline": acc_base, "configs": {}}

    for name, (start, end) in configs.items():
        steerer = make_steerer_layer_range(model_info, cal, start, end)
        steerer.steer(alpha=1.0)
        n_hooks = len(steerer.hooks)
        acc, _ = eval_pope(model_info, pope_adv_100, name)
        steerer.cleanup()
        results["configs"][name] = {"acc": acc, "n_hooks": n_hooks}
        print(f"    {name} ({n_hooks} hooks): {acc:.1f}% (Δ={acc-acc_base:+.1f})")

    results["n_samples"] = len(pope_adv_100)
    save_result(results, "sweeps", "deepstack_test")
    return results


def exp_per_sample_analysis(model_info, cal, pope_adv_200):
    """Block 2C: Per-sample helped/hurt/neutral analysis."""
    print("\n" + "="*60)
    print("EXPERIMENT: Per-Sample Analysis (200 POPE-A)")
    print("="*60)

    # Baseline
    _, ps_base = eval_pope(model_info, pope_adv_200, "baseline")

    # Steered
    steerer = ActivationSteerer(model_info, cal, steer_layers_start=4)
    steerer.steer(alpha=1.0)
    _, ps_steer = eval_pope(model_info, pope_adv_200, "steered")
    steerer.cleanup()

    helped = []
    hurt = []
    neutral = []
    for b, s in zip(ps_base, ps_steer):
        if not b["correct"] and s["correct"]:
            helped.append({"idx": b["idx"], "question": b["question"],
                           "base_pred": b["pred"], "steer_pred": s["pred"], "gt": b["gt"]})
        elif b["correct"] and not s["correct"]:
            hurt.append({"idx": b["idx"], "question": b["question"],
                         "base_pred": b["pred"], "steer_pred": s["pred"], "gt": b["gt"]})
        else:
            neutral.append({"idx": b["idx"], "base_correct": b["correct"],
                            "steer_correct": s["correct"]})

    results = {
        "n_helped": len(helped),
        "n_hurt": len(hurt),
        "n_neutral": len(neutral),
        "n_total": len(ps_base),
        "net_effect": len(helped) - len(hurt),
        "helped_samples": helped[:20],
        "hurt_samples": hurt[:20],
        "gt_distribution": {
            "yes": sum(1 for s in pope_adv_200 if s["answer"] == "yes"),
            "no": sum(1 for s in pope_adv_200 if s["answer"] == "no"),
        },
    }

    print(f"\n  Helped (wrong→right): {len(helped)}")
    print(f"  Hurt (right→wrong):   {len(hurt)}")
    print(f"  Neutral:              {len(neutral)}")
    print(f"  Net effect:           {len(helped) - len(hurt):+d}")

    save_result(results, "analysis", "per_sample_analysis")
    return results


def generate_plots(all_results):
    """Generate summary plots from all experiments."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Alpha sweep plot
        if "alpha_sweep" in all_results:
            r = all_results["alpha_sweep"]
            alphas = sorted(r["alphas"].keys(), key=float)
            accs = [r["alphas"][a] for a in alphas]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot([float(a) for a in alphas], accs, "o-", color="steelblue", linewidth=2)
            ax.axhline(r["baseline"], color="gray", linestyle="--", label=f"Baseline: {r['baseline']:.1f}%")
            ax.set_xlabel("Alpha (steering strength)")
            ax.set_ylabel("POPE Adversarial Accuracy (%)")
            ax.set_title("Alpha Sweep: Steering Strength vs Accuracy")
            ax.legend()
            ax.grid(True, alpha=0.3)
            for a, acc in zip(alphas, accs):
                delta = acc - r["baseline"]
                ax.annotate(f"Δ={delta:+.1f}", (float(a), acc), textcoords="offset points",
                            xytext=(0, 10), ha="center", fontsize=9)
            fig.tight_layout()
            fig.savefig(REPORTS_DIR / f"alpha_sweep_{ts}.png", dpi=150)
            plt.close()
            print(f"Saved alpha sweep plot")

        # K sweep plot
        if "k_sweep" in all_results:
            r = all_results["k_sweep"]
            ks = sorted(r["ks"].keys(), key=int)
            accs = [r["ks"][k] for k in ks]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot([int(k) for k in ks], accs, "s-", color="darkorange", linewidth=2)
            ax.axhline(r["baseline"], color="gray", linestyle="--", label=f"Baseline: {r['baseline']:.1f}%")
            ax.set_xlabel("K (number of steered heads)")
            ax.set_ylabel("POPE Adversarial Accuracy (%)")
            ax.set_title("K Sweep: Number of Steered Heads vs Accuracy")
            ax.legend()
            ax.grid(True, alpha=0.3)
            for k, acc in zip(ks, accs):
                delta = acc - r["baseline"]
                ax.annotate(f"Δ={delta:+.1f}", (int(k), acc), textcoords="offset points",
                            xytext=(0, 10), ha="center", fontsize=9)
            fig.tight_layout()
            fig.savefig(REPORTS_DIR / f"k_sweep_{ts}.png", dpi=150)
            plt.close()
            print(f"Saved K sweep plot")

        # DeepStack bar chart
        if "deepstack" in all_results:
            r = all_results["deepstack"]
            names = list(r["configs"].keys())
            accs = [r["configs"][n]["acc"] for n in names]
            hooks = [r["configs"][n]["n_hooks"] for n in names]
            fig, ax = plt.subplots(figsize=(9, 5))
            bars = ax.bar(range(len(names)), accs, color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"])
            ax.axhline(r["baseline"], color="gray", linestyle="--", label=f"Baseline: {r['baseline']:.1f}%")
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels([f"{n}\n({h} hooks)" for n, h in zip(names, hooks)], fontsize=9)
            ax.set_ylabel("POPE Adversarial Accuracy (%)")
            ax.set_title("DeepStack Test: Which Layers to Steer?")
            ax.legend()
            for bar, acc in zip(bars, accs):
                delta = acc - r["baseline"]
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        f"Δ={delta:+.1f}", ha="center", fontsize=10)
            fig.tight_layout()
            fig.savefig(REPORTS_DIR / f"deepstack_{ts}.png", dpi=150)
            plt.close()
            print(f"Saved DeepStack plot")

        # Per-sample analysis pie chart
        if "per_sample" in all_results:
            r = all_results["per_sample"]
            fig, ax = plt.subplots(figsize=(7, 5))
            sizes = [r["n_helped"], r["n_hurt"],
                     r["n_neutral"] - sum(1 for _ in [] )]  # just n_neutral
            labels = [f"Helped ({r['n_helped']})", f"Hurt ({r['n_hurt']})",
                      f"Neutral ({r['n_neutral']})"]
            colors = ["#4CAF50", "#F44336", "#9E9E9E"]
            ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
            ax.set_title(f"Per-Sample Steering Effect (n={r['n_total']}, net={r['net_effect']:+d})")
            fig.tight_layout()
            fig.savefig(REPORTS_DIR / f"per_sample_{ts}.png", dpi=150)
            plt.close()
            print(f"Saved per-sample plot")

    except Exception as e:
        print(f"Plot generation failed: {e}")


def main():
    print("="*60)
    print("VIGIL Experiment Runner")
    print(f"Time: {datetime.now().isoformat()}")
    print("="*60)

    # Load model
    print("\n[1/7] Loading model...")
    model_info = load_model("qwen3_vl_2b", dtype=torch.float16, device="auto")

    # Load calibration
    print("\n[2/7] Loading calibration...")
    cal = CalibrationResult.load(str(CAL_DIR))

    # Load data
    print("\n[3/7] Loading POPE data...")
    pope_adv = load_pope("adversarial", limit=500)
    pope_adv_200 = pope_adv[:200]
    pope_adv_100 = pope_adv[:100]

    all_results = {}
    t0 = time.time()

    # --- PV4: Blind tests (baseline + steered) ---
    print("\n[EXP 1/6] Blind test BASELINE...")
    blind_base = exp_blind_test(model_info, pope_adv)
    all_results["blind_test_baseline"] = blind_base

    print("\n[EXP 2/6] Blind test STEERED...")
    blind_steer = exp_blind_test(model_info, pope_adv, cal=cal, steered=True)
    all_results["blind_test_steered"] = blind_steer

    gap_delta = blind_steer["gap"] - blind_base["gap"]
    print(f"\n  >>> PV4 RESULT: Gap baseline={blind_base['gap']:.1f}pp, steered={blind_steer['gap']:.1f}pp, Δ={gap_delta:+.1f}pp")
    print(f"  >>> {'PASS' if gap_delta >= 0 else 'FAIL'}: Steering {'increases' if gap_delta >= 0 else 'decreases'} image-dependence")

    # --- Block 2C: Per-sample analysis ---
    print("\n[EXP 3/6] Per-sample analysis (200 POPE-A)...")
    per_sample = exp_per_sample_analysis(model_info, cal, pope_adv_200)
    all_results["per_sample"] = per_sample

    # --- Block 3: Sweeps ---
    print("\n[EXP 4/6] Alpha sweep...")
    alpha = exp_alpha_sweep(model_info, cal, pope_adv_100)
    all_results["alpha_sweep"] = alpha

    print("\n[EXP 5/6] K sweep...")
    k = exp_k_sweep(model_info, cal, pope_adv_100)
    all_results["k_sweep"] = k

    print("\n[EXP 6/6] DeepStack test...")
    ds = exp_deepstack_test(model_info, cal, pope_adv_100)
    all_results["deepstack"] = ds

    # Generate plots
    print("\nGenerating plots...")
    generate_plots(all_results)

    # Save combined results
    save_result(all_results, "combined", "all_experiments")

    elapsed = time.time() - t0

    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)

    print(f"\n--- PV4: Blind Test Gap ---")
    print(f"  Baseline: real={blind_base['acc_real']:.1f}%, black={blind_base['acc_black']:.1f}%, gap={blind_base['gap']:.1f}pp")
    print(f"  Steered:  real={blind_steer['acc_real']:.1f}%, black={blind_steer['acc_black']:.1f}%, gap={blind_steer['gap']:.1f}pp")
    print(f"  Gap Δ = {gap_delta:+.1f}pp → {'PASS' if gap_delta >= 0 else 'FAIL'}")

    print(f"\n--- Per-Sample Analysis ---")
    print(f"  Helped: {per_sample['n_helped']}, Hurt: {per_sample['n_hurt']}, Net: {per_sample['net_effect']:+d}")

    print(f"\n--- Alpha Sweep ---")
    best_a = max(alpha["alphas"], key=lambda a: alpha["alphas"][a])
    print(f"  Best α={best_a}: {alpha['alphas'][best_a]:.1f}% (Δ={alpha['alphas'][best_a]-alpha['baseline']:+.1f})")
    for a in sorted(alpha["alphas"].keys(), key=float):
        print(f"    α={a}: {alpha['alphas'][a]:.1f}%")

    print(f"\n--- K Sweep ---")
    best_k = max(k["ks"], key=lambda x: k["ks"][x])
    print(f"  Best K={best_k}: {k['ks'][best_k]:.1f}% (Δ={k['ks'][best_k]-k['baseline']:+.1f})")
    for kk in sorted(k["ks"].keys(), key=int):
        print(f"    K={kk}: {k['ks'][kk]:.1f}%")

    print(f"\n--- DeepStack ---")
    for name, cfg in ds["configs"].items():
        print(f"  {name}: {cfg['acc']:.1f}% ({cfg['n_hooks']} hooks, Δ={cfg['acc']-ds['baseline']:+.1f})")

    print(f"\n--- Pre-Validation Summary ---")
    print(f"  PV1 (vision heads exist):     PASS (prior: mean Δ=6.1, max=66.2)")
    print(f"  PV2 (steering improves acc):   PASS (prior: +1.5-2pp on POPE)")
    print(f"  PV3 (thinking mode drift):     PENDING (needs Thinking model)")
    pv4 = "PASS" if gap_delta >= 0 else "FAIL"
    print(f"  PV4 (blind test gap up):       {pv4} (gap Δ={gap_delta:+.1f}pp)")

    print(f"\nTotal time: {elapsed/60:.1f} min")
    print("DONE.")


if __name__ == "__main__":
    main()
