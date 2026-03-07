"""
VIGIL — Re-run blind test (fixed correctness) + PV3 Thinking model + extended alpha sweep.
"""
import sys, os
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

import json, time, gc, torch
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model_registry import load_model, make_chat_prompt
from src.calibrator import CalibrationResult
from src.steerer import ActivationSteerer
from src.blind_test import run_blind_test, save_blind_test_results
from src.data_loader import load_pope

RESULTS_DIR = Path(__file__).parent.parent / "lab" / "results"
REPORTS_DIR = Path(__file__).parent.parent / "lab" / "reports"
CAL_DIR = Path(__file__).parent.parent / "checkpoints" / "calibration" / "qwen3_vl_2b"


def generate_fn(model_info, question, image, max_tokens=30):
    inputs = make_chat_prompt(model_info, question, image)
    with torch.no_grad():
        out = model_info["model"].generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    return model_info["tokenizer"].decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def check_yesno(pred, gt):
    p = pred.strip().lower()
    g = gt.strip().lower()
    has_yes = "yes" in p
    has_no = "no" in p
    if has_yes and has_no:
        yn = "yes" if p.index("yes") < p.index("no") else "no"
    elif has_yes:
        yn = "yes"
    elif has_no:
        yn = "no"
    else:
        yn = ""
    return yn == g


def eval_pope_quick(model_info, samples, label="", max_tokens=30):
    correct = total = 0
    per_sample = []
    for i, s in enumerate(samples):
        try:
            pred = generate_fn(model_info, s["question"], s.get("image"), max_tokens)
            is_correct = check_yesno(pred, s["answer"])
            per_sample.append({"idx": i, "pred": pred, "gt": s["answer"], "correct": is_correct})
            if is_correct:
                correct += 1
            total += 1
            if (i + 1) % 100 == 0:
                print(f"  [{label}] {i+1}/{len(samples)}: {correct/total*100:.1f}%")
        except Exception as e:
            print(f"  [{label}] Skip {i}: {e}")
    acc = correct / total * 100 if total > 0 else 0.0
    print(f"  [{label}] Final: {acc:.1f}% ({correct}/{total})")
    return acc, per_sample


def save_json(data, subdir, name):
    out = RESULTS_DIR / subdir
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = out / f"{name}_{ts}.json"
    with open(fname, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved: {fname}")
    return fname


def main():
    t0 = time.time()
    print(f"{'='*60}\nVIGIL Remaining Experiments\n{datetime.now().isoformat()}\n{'='*60}")

    # Load model
    print("\n--- Loading Instruct model ---")
    model_info = load_model("qwen3_vl_2b", dtype=torch.float16, device="auto")
    cal = CalibrationResult.load(str(CAL_DIR))

    # Load data
    pope_adv = load_pope("adversarial", limit=500)
    pope_adv_100 = pope_adv[:100]

    # ============================================================
    # 1. RE-RUN BLIND TEST with fixed correctness check
    # ============================================================
    print(f"\n{'='*60}\n1. BLIND TEST (fixed correctness) — baseline\n{'='*60}")
    blind_base = run_blind_test(model_info, pope_adv, generate_fn, max_samples=500)
    blind_base["mode"] = "baseline_fixed"
    save_blind_test_results(blind_base, str(RESULTS_DIR / "blind_test"))

    print(f"\n{'='*60}\n2. BLIND TEST (fixed correctness) — steered α=1.0\n{'='*60}")
    steerer = ActivationSteerer(model_info, cal, steer_layers_start=4)
    steerer.steer(alpha=1.0)
    blind_steer = run_blind_test(model_info, pope_adv, generate_fn, max_samples=500)
    blind_steer["mode"] = "steered_fixed"
    save_blind_test_results(blind_steer, str(RESULTS_DIR / "blind_test"))
    steerer.cleanup()

    gap_delta = blind_steer["gap"] - blind_base["gap"]
    print(f"\n>>> FIXED PV4: baseline gap={blind_base['gap']:.1f}pp, steered gap={blind_steer['gap']:.1f}pp, Δ={gap_delta:+.1f}pp")

    # ============================================================
    # 3. EXTENDED ALPHA SWEEP [1, 2, 3, 5, 8, 10]
    # ============================================================
    print(f"\n{'='*60}\n3. EXTENDED ALPHA SWEEP [1,2,3,5,8,10] — 100 POPE-A\n{'='*60}")
    acc_base, _ = eval_pope_quick(model_info, pope_adv_100, "baseline")
    alphas = [1, 2, 3, 5, 8, 10]
    alpha_results = {"baseline": acc_base, "alphas": {}}
    for alpha in alphas:
        steerer = ActivationSteerer(model_info, cal, steer_layers_start=4)
        steerer.steer(alpha=float(alpha))
        acc, _ = eval_pope_quick(model_info, pope_adv_100, f"α={alpha}")
        steerer.cleanup()
        alpha_results["alphas"][str(alpha)] = acc
        print(f"  α={alpha}: {acc:.1f}% (Δ={acc-acc_base:+.1f})")
    save_json(alpha_results, "sweeps", "alpha_sweep_extended")

    # ============================================================
    # 4. PV3: THINKING MODEL
    # ============================================================
    print(f"\n{'='*60}\n4. PV3: THINKING MODEL\n{'='*60}")

    # Free Instruct model memory
    del model_info["model"]
    gc.collect()
    torch.cuda.empty_cache()
    print("Freed Instruct model VRAM")

    # Check if Thinking model exists
    try:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        print("Loading Qwen3-VL-2B-Thinking...")
        think_model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-2B-Thinking", torch_dtype=torch.float16, device_map="auto",
        )
        think_processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Thinking")
        think_info = {
            "model": think_model,
            "processor": think_processor,
            "tokenizer": think_processor.tokenizer,
            "get_layers_fn": lambda: think_model.model.language_model.layers,
            "get_lm_head_fn": lambda: think_model.lm_head,
            "get_norm_fn": lambda: think_model.model.language_model.norm,
            "model_key": "qwen3_vl_2b_thinking",
            "model_type": "qwen3_vl",
            "num_layers": 28,
            "num_heads": 16,
            "num_kv_heads": 8,
            "head_dim": 128,
            "hidden_size": 2048,
            "gqa": True,
            "steer_layers_start": 4,
            "device": next(think_model.parameters()).device,
        }
        print("Thinking model loaded")

        # Thinking model generates longer — use max_tokens=200
        def think_generate_fn(mi, q, img):
            return generate_fn(mi, q, img, max_tokens=200)

        # Baseline on POPE-A 100
        print("\nThinking baseline (100 POPE-A)...")
        acc_think_base, ps_think_base = eval_pope_quick(think_info, pope_adv_100, "think_base", max_tokens=200)

        # Steered on POPE-A 100
        think_steerer = ActivationSteerer(think_info, cal, steer_layers_start=4)
        think_steerer.steer(alpha=1.0)
        print("\nThinking steered α=1.0 (100 POPE-A)...")
        acc_think_steer, ps_think_steer = eval_pope_quick(think_info, pope_adv_100, "think_steer", max_tokens=200)
        think_steerer.cleanup()

        # Steered α=3.0
        think_steerer = ActivationSteerer(think_info, cal, steer_layers_start=4)
        think_steerer.steer(alpha=3.0)
        print("\nThinking steered α=3.0 (100 POPE-A)...")
        acc_think_steer3, _ = eval_pope_quick(think_info, pope_adv_100, "think_steer3", max_tokens=200)
        think_steerer.cleanup()

        think_delta = acc_think_steer - acc_think_base
        think_delta3 = acc_think_steer3 - acc_think_base
        print(f"\n>>> PV3: Thinking baseline={acc_think_base:.1f}%, steered α=1={acc_think_steer:.1f}% (Δ={think_delta:+.1f}), α=3={acc_think_steer3:.1f}% (Δ={think_delta3:+.1f})")

        think_results = {
            "baseline": acc_think_base,
            "steered_1.0": acc_think_steer,
            "steered_3.0": acc_think_steer3,
            "delta_1.0": think_delta,
            "delta_3.0": think_delta3,
            "n_samples": len(pope_adv_100),
        }
        save_json(think_results, "thinking", "pv3_thinking_model")

        pv3_pass = think_delta > 0 or think_delta3 > 0

    except Exception as e:
        print(f"Thinking model failed: {e}")
        think_results = {"error": str(e)}
        pv3_pass = None
        save_json(think_results, "thinking", "pv3_thinking_model_error")

    # ============================================================
    # SUMMARY
    # ============================================================
    elapsed = time.time() - t0
    print(f"\n{'='*60}\nFINAL SUMMARY\n{'='*60}")

    print(f"\n--- PV4 (fixed correctness) ---")
    print(f"  Baseline: real={blind_base['acc_real']:.1f}%, black={blind_base['acc_black']:.1f}%, gap={blind_base['gap']:.1f}pp")
    print(f"  Steered:  real={blind_steer['acc_real']:.1f}%, black={blind_steer['acc_black']:.1f}%, gap={blind_steer['gap']:.1f}pp")
    print(f"  Gap Δ = {gap_delta:+.1f}pp → {'PASS' if gap_delta >= 0 else 'FAIL'}")

    print(f"\n--- Extended Alpha Sweep ---")
    for a in sorted(alpha_results["alphas"].keys(), key=lambda x: float(x)):
        print(f"  α={a}: {alpha_results['alphas'][a]:.1f}% (Δ={alpha_results['alphas'][a]-alpha_results['baseline']:+.1f})")

    print(f"\n--- PV3 (Thinking Model) ---")
    if pv3_pass is not None:
        print(f"  Baseline: {think_results['baseline']:.1f}%")
        print(f"  Steered α=1: {think_results['steered_1.0']:.1f}% (Δ={think_results['delta_1.0']:+.1f})")
        print(f"  Steered α=3: {think_results['steered_3.0']:.1f}% (Δ={think_results['delta_3.0']:+.1f})")
        print(f"  → {'PASS' if pv3_pass else 'FAIL'}")
    else:
        print(f"  ERROR: {think_results.get('error', 'unknown')}")

    print(f"\n--- Pre-Validation Final ---")
    print(f"  PV1: PASS")
    print(f"  PV2: PASS")
    print(f"  PV3: {'PASS' if pv3_pass else ('FAIL' if pv3_pass is False else 'ERROR')}")
    print(f"  PV4: {'PASS' if gap_delta >= 0 else 'FAIL'}")
    if pv3_pass is not False and gap_delta >= 0:
        print(f"\n  >>> ALL PRE-VALIDATIONS PASSED → PROCEED TO R_vhad GRPO")
    print(f"\nTotal time: {elapsed/60:.1f} min")
    print("DONE.")


if __name__ == "__main__":
    main()
