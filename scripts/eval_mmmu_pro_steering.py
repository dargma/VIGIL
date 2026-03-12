"""
VIGIL MMMU-Pro Steering Analysis — Qwen3-VL-2B-Thinking

Purpose: Evaluate if steering improves MMMU-Pro performance.
         If yes → justifies GRPO soft R_vhad steering reward.

Analysis axes:
  1. α sweep: accuracy vs steering strength
  2. Image dependency: standard-10 vs vision mode
  3. Thinking chain: does steering change reasoning length?
  4. Per-subject: which domains benefit from steering?

Official settings (Qwen3-VL paper §5):
  Dense thinking: temp=1.0, top_p=0.95, top_k=20
  Paper reference: Qwen3-VL-2B-Thinking MMMU-Pro = 42.5%

Usage:
    # Full pipeline: baseline + calibrate + sweep
    PYTHONUNBUFFERED=1 python -u scripts/eval_mmmu_pro_steering.py \
        --max-samples 100 --alphas 0,1,3,5,7,10 2>&1 | tee logs/mmmu_pro_steering.log
"""

import os, sys, json, re, argparse, gc, time
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.steerer import ActivationSteerer
from src.calibrator import CalibrationResult, SteeringCalibrator


# ─── Constants ───────────────────────────────────────────────────────────────

DATA_ROOT = Path("data")
PAPER_SCORE = 42.5

GEN_KWARGS = dict(
    max_new_tokens=4096,
    temperature=1.0,
    top_p=0.95,
    top_k=20,
    do_sample=True,
)


# ─── Thinking parser ────────────────────────────────────────────────────────

def split_thinking(text):
    """Parse thinking chain from model output."""
    # <think>...</think> present
    m = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    # Only </think> (because <think> is in the prompt)
    m = re.search(r'</think>', text)
    if m:
        return text[:m.start()].strip(), text[m.end():].strip()
    return "", text.strip()


def extract_mc_answer(raw_output, num_choices=10):
    """Extract letter answer from model output."""
    _, answer_text = split_thinking(raw_output)
    if not answer_text:
        answer_text = raw_output
    valid = [chr(65 + i) for i in range(num_choices)]
    # Patterns
    for pat in [
        r'(?:the\s+)?answer\s+is\s*[:\s]*\(?([A-J])\)?',
        r'(?:answer|option|choice)\s*[:\s]*\(?([A-J])\)?',
        r'\*\*([A-J])\*\*',
        r'^\s*\(?([A-J])\)?[\.\s]*$',
    ]:
        m = re.search(pat, answer_text, re.IGNORECASE | re.MULTILINE)
        if m and m.group(1).upper() in valid:
            return m.group(1).upper()
    # Last letter
    found = re.findall(r'\b([A-J])\b', answer_text)
    for letter in reversed(found):
        if letter in valid:
            return letter
    return None


# ─── Data ────────────────────────────────────────────────────────────────────

def load_mmmu_pro():
    from datasets import load_from_disk
    datasets = {}
    for name, path in [
        ("standard-10", "eval/mmmu_pro_standard10"),
        ("vision", "eval/mmmu_pro_vision"),
    ]:
        full = DATA_ROOT / path
        if full.exists():
            datasets[name] = load_from_disk(str(full))
            print(f"[data] {name}: {len(datasets[name])} samples")
    return datasets


def load_mmmu_cal():
    from datasets import load_from_disk
    path = DATA_ROOT / "mmmu_full"
    ds = load_from_disk(str(path))
    print(f"[data] MMMU calibration: {len(ds)} samples")
    return ds


def parse_options(row):
    opts = row.get("options", [])
    if isinstance(opts, str):
        try: opts = json.loads(opts)
        except: opts = eval(opts) if opts.startswith("[") else []
    return opts if isinstance(opts, list) else []


def get_image(row):
    for key in ["image", "image_1", "question_image"]:
        img = row.get(key)
        if isinstance(img, Image.Image):
            return img
    return None


# ─── Model ───────────────────────────────────────────────────────────────────

def load_model():
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    hf_id = "Qwen/Qwen3-VL-2B-Thinking"
    print(f"[model] Loading {hf_id}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(hf_id)
    model.eval()
    info = {
        "model": model, "processor": processor,
        "tokenizer": processor.tokenizer,
        "get_layers_fn": lambda: model.model.language_model.layers,
        "get_lm_head_fn": lambda: model.lm_head,
        "get_norm_fn": lambda: model.model.language_model.norm,
        "num_layers": 28, "num_heads": 16, "num_kv_heads": 8,
        "head_dim": 128, "hidden_size": 2048, "gqa": True,
        "steer_layers_start": 4,
        "device": next(model.parameters()).device,
    }
    print(f"[model] Loaded. 28 layers, 16Q/8KV")
    return info


def generate(model_info, image, prompt):
    from qwen_vl_utils import process_vision_info
    model = model_info["model"]
    processor = model_info["processor"]
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True)
    if image is not None:
        imgs, _, _ = process_vision_info(messages, return_video_kwargs=True)
    else:
        imgs = None
    inputs = processor(text=[text], images=imgs, return_tensors="pt", padding=True)
    inputs = {k: v.to(model_info["device"]) for k, v in inputs.items()}
    with torch.no_grad():
        gen = model.generate(**inputs, **GEN_KWARGS)
    out = gen[0][inputs["input_ids"].shape[1]:]
    raw = processor.tokenizer.decode(out, skip_special_tokens=False)
    for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
        raw = raw.replace(tok, "")
    return raw.strip()


# ─── Calibration ─────────────────────────────────────────────────────────────

def calibrate(model_info, max_samples=500):
    """Load calibration vectors.

    Strategy: Use existing Instruct model calibration from POPE experiments.
    Instruct and Thinking models share the same base weights and attention heads,
    so vision head identification transfers. The Thinking model just adds
    reasoning chains before answering.

    Fallback: confidence-based split on MMMU if no POPE calibration exists.
    """
    # Prefer existing POPE calibration (validated, 20 heads)
    pope_cal = Path("checkpoints/calibration/qwen3_vl_2b")
    if pope_cal.exists() and (pope_cal / "steering_vectors.pt").exists():
        print(f"[cal] Using POPE calibration from {pope_cal}")
        print(f"  (Instruct & Thinking share base weights → vision heads transfer)")
        result = CalibrationResult.load(str(pope_cal))
        return result

    # Fallback: MMMU confidence-based calibration
    from qwen_vl_utils import process_vision_info
    cal_dir = Path("checkpoints/calibration/qwen3_vl_2b_thinking_mmmu")
    if cal_dir.exists() and (cal_dir / "steering_vectors.pt").exists():
        result = CalibrationResult.load(str(cal_dir))
        if len(result.top_heads) > 0:
            print(f"[cal] Loading from {cal_dir}")
            return result

    print(f"[cal] Running confidence-based calibration on MMMU...")
    ds = load_mmmu_cal()
    samples = []
    for i, row in enumerate(ds):
        if len(samples) >= max_samples:
            break
        image = get_image(row)
        if image is None:
            continue
        opts = parse_options(row)
        samples.append({
            "question": row.get("question", ""),
            "answer": str(row.get("answer", "")).strip().upper(),
            "choices": opts, "image": image, "type": "mc",
        })

    print(f"[cal] {len(samples)} samples with images")

    def process_fn(mi, s):
        processor = mi["processor"]
        q = s["question"]
        c = s.get("choices", [])
        if c:
            opts_text = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(c))
            prompt = f"{q}\nOptions:\n{opts_text}\nAnswer with the letter only."
        else:
            prompt = f"{q}\nAnswer briefly."
        messages = [{"role": "user", "content": [
            {"type": "image", "image": s["image"]},
            {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True,
                                              enable_thinking=False)
        imgs, _, _ = process_vision_info(messages, return_video_kwargs=True)
        inputs = processor(text=[text], images=imgs, return_tensors="pt", padding=True)
        inputs = {k: v.to(mi["device"]) for k, v in inputs.items()}
        return inputs, s["answer"]

    # Use confidence_split_threshold=0 to force confidence-based split
    cal = SteeringCalibrator(model_info, top_k=20, confidence_split_threshold=0)
    result = cal.calibrate(samples, process_fn, max_samples=len(samples))
    result.save(str(cal_dir))
    return result


# ─── Eval loop ───────────────────────────────────────────────────────────────

def eval_config(model_info, dataset, config_name, max_samples, label=""):
    n = min(len(dataset), max_samples)
    is_vision = "vision" in config_name
    correct, total = 0, 0
    think_lens = []
    subject_scores = defaultdict(lambda: {"c": 0, "t": 0})
    records = []
    t0 = time.time()

    for i in range(n):
        row = dataset[i]
        try:
            image = get_image(row)
            question = row.get("question", "")
            gt = str(row.get("answer", "")).strip().upper()
            subject = row.get("subject", "unknown")
            options = parse_options(row)
            nc = len(options) if options else 10

            if is_vision:
                prompt = "Identify the problem and solve it. Think step by step before answering."
            else:
                opts_text = "\n".join(f"{chr(65+j)}. {o}" for j, o in enumerate(options))
                hint = row.get("hint", "") or ""
                hint_text = f"{hint}\n" if hint else ""
                prompt = f"{hint_text}{question}\nOptions:\n{opts_text}\nPlease select the correct answer from the options above."

            raw = generate(model_info, image, prompt)
            thinking, answer = split_thinking(raw)
            think_len = len(thinking.split()) if thinking else 0
            think_lens.append(think_len)
            pred = extract_mc_answer(raw, nc)
            ok = (pred == gt) if pred else False
            if ok: correct += 1
            total += 1
            subject_scores[subject]["t"] += 1
            if ok: subject_scores[subject]["c"] += 1

            records.append({
                "i": i, "gt": gt, "pred": pred, "ok": ok,
                "think_len": think_len, "subject": subject,
                "raw": raw[:200],
            })

            if (i + 1) % 20 == 0:
                acc = correct / total * 100
                rate = (i + 1) / (time.time() - t0)
                eta = (n - i - 1) / rate / 60
                print(f"  [{label}|{config_name}] {i+1}/{n} acc={acc:.1f}% "
                      f"think={np.mean(think_lens):.0f}w {rate:.2f}/s ETA={eta:.1f}m")

        except Exception as e:
            print(f"  [{i}] ERR: {e}")
            records.append({"i": i, "error": str(e), "ok": False})
            total += 1

    acc = correct / total * 100 if total > 0 else 0
    elapsed = (time.time() - t0) / 60
    avg_think = float(np.mean(think_lens)) if think_lens else 0

    print(f"  → [{label}|{config_name}] acc={acc:.1f}% think={avg_think:.0f}w ({elapsed:.1f}m)")

    per_subj = {}
    for s, sc in sorted(subject_scores.items()):
        sa = sc["c"] / sc["t"] * 100 if sc["t"] > 0 else 0
        per_subj[s] = {"acc": sa, "correct": sc["c"], "total": sc["t"]}

    return {
        "acc": acc, "correct": correct, "total": total,
        "avg_think_words": avg_think, "elapsed_min": elapsed,
        "per_subject": per_subj, "records": records,
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--cal-samples", type=int, default=500)
    parser.add_argument("--alphas", type=str, default="0,1,3,5,7,10")
    parser.add_argument("--config", type=str, default="standard-10",
                        choices=["standard-10", "vision", "both"])
    parser.add_argument("--output-dir", type=str, default="lab/reports/mmmu_pro_steering")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    alphas = [float(a) for a in args.alphas.split(",")]

    # Load data
    mmmu_pro = load_mmmu_pro()
    configs = list(mmmu_pro.keys()) if args.config == "both" else [args.config]

    # Load model
    model_info = load_model()

    # Calibrate
    print("\n" + "="*60)
    print("Calibrating steering vectors...")
    print("="*60)
    calibration = calibrate(model_info, args.cal_samples)
    print(f"Top-5 heads:")
    for i, (li, hi) in enumerate(calibration.top_heads[:5]):
        d = calibration.head_scores.get((li, hi), 0)
        print(f"  {i+1}. L{li}H{hi}: d={d:.3f}")

    # Sweep alphas
    all_results = {}
    for alpha in alphas:
        steerer = None
        label = f"α={alpha}"
        if alpha > 0:
            steerer = ActivationSteerer(model_info, calibration, steer_layers_start=4)
            steerer.steer(alpha)
            print(f"\n{'─'*40} Steering α={alpha} {'─'*40}")
        else:
            print(f"\n{'─'*40} Baseline (α=0) {'─'*40}")

        for config_name in configs:
            result = eval_config(
                model_info, mmmu_pro[config_name], config_name,
                args.max_samples, label,
            )
            key = f"a{alpha}_{config_name}"
            all_results[key] = result

            # Save per-condition
            safe = {k: v for k, v in result.items() if k != "records"}
            with open(output_dir / f"{key}_{ts}.json", "w") as f:
                json.dump(safe, f, indent=2)

        if steerer:
            steerer.cleanup()

    # ── Analysis Report ──
    print("\n" + "="*75)
    print("MMMU-Pro Steering Analysis — Qwen3-VL-2B-Thinking")
    print(f"Paper MMMU-Pro = {PAPER_SCORE}%, {args.max_samples} samples/config")
    print("="*75)

    for config_name in configs:
        print(f"\n--- {config_name} ---")
        print(f"{'Alpha':>7} {'Acc':>8} {'Δ':>8} {'Think':>8} {'ΔThink':>8}")
        print("-" * 45)

        baseline_acc = None
        baseline_think = None
        for alpha in alphas:
            key = f"a{alpha}_{config_name}"
            r = all_results.get(key)
            if r is None:
                continue
            if baseline_acc is None:
                baseline_acc = r["acc"]
                baseline_think = r["avg_think_words"]

            delta_acc = r["acc"] - baseline_acc
            delta_think = r["avg_think_words"] - baseline_think

            delta_str = f"{delta_acc:+.1f}pp" if alpha > 0 else "—"
            dt_str = f"{delta_think:+.0f}w" if alpha > 0 else "—"

            print(f"{alpha:>7.0f} {r['acc']:>7.1f}% {delta_str:>8} "
                  f"{r['avg_think_words']:>7.0f}w {dt_str:>8}")

    # Per-subject analysis (for standard-10 only)
    if "standard-10" in configs:
        print(f"\n--- Per-Subject: α=0 vs best α ---")
        base_key = f"a0_standard-10"
        if base_key in all_results:
            base_subj = all_results[base_key]["per_subject"]
            # Find best alpha
            best_alpha = max(alphas, key=lambda a: all_results.get(f"a{a}_standard-10", {}).get("acc", 0))
            best_key = f"a{best_alpha}_standard-10"
            best_subj = all_results.get(best_key, {}).get("per_subject", {})

            print(f"{'Subject':<35} {'Base':>6} {'α={:.0f}'.format(best_alpha):>6} {'Δ':>7}")
            print("-" * 58)
            for subj in sorted(set(list(base_subj.keys()) + list(best_subj.keys()))):
                ba = base_subj.get(subj, {}).get("acc", 0)
                sa = best_subj.get(subj, {}).get("acc", 0)
                delta = sa - ba
                marker = " ★" if delta > 5 else " ▼" if delta < -5 else ""
                print(f"{subj:<35} {ba:>5.0f}% {sa:>5.0f}% {delta:>+6.1f}pp{marker}")

    # GRPO feasibility assessment
    print(f"\n{'='*75}")
    print("GRPO Steering Reward Feasibility Assessment")
    print("="*75)

    if "standard-10" in configs:
        base = all_results.get("a0_standard-10", {}).get("acc", 0)
        best_acc = max(all_results.get(f"a{a}_standard-10", {}).get("acc", 0) for a in alphas)
        best_a = max(alphas, key=lambda a: all_results.get(f"a{a}_standard-10", {}).get("acc", 0))
        gain = best_acc - base

        print(f"  Baseline acc: {base:.1f}%")
        print(f"  Best steered: {best_acc:.1f}% (α={best_a:.0f})")
        print(f"  Gain: {gain:+.1f}pp")

        if gain > 2:
            print(f"\n  ✓ POSITIVE: Steering improves MMMU-Pro by {gain:.1f}pp")
            print(f"  → Vision head activation matters for reasoning")
            print(f"  → GRPO soft R_vhad reward is justified")
            print(f"  → Next: implement R_vhad as continuous GRPO reward signal")
        elif gain > 0:
            print(f"\n  ~ MARGINAL: Steering shows {gain:.1f}pp improvement")
            print(f"  → Vision heads contribute but effect is small")
            print(f"  → GRPO R_vhad worth trying but may not be primary reward")
        else:
            print(f"\n  ✗ NEGATIVE: Steering does not improve MMMU-Pro")
            print(f"  → Vision heads may not be the bottleneck for reasoning")
            print(f"  → Consider alternative reward signals")

    # Save full results
    safe_all = {}
    for k, v in all_results.items():
        safe_all[k] = {kk: vv for kk, vv in v.items() if kk != "records"}
    safe_all["meta"] = {
        "timestamp": ts, "max_samples": args.max_samples,
        "alphas": alphas, "configs": configs,
        "paper_score": PAPER_SCORE,
    }
    with open(output_dir / f"steering_analysis_{ts}.json", "w") as f:
        json.dump(safe_all, f, indent=2)
    print(f"\nResults: {output_dir}/steering_analysis_{ts}.json")


if __name__ == "__main__":
    main()
