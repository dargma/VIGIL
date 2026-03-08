"""
VIGIL Multi-Model Pipeline — Apply calibration + BoN+SFT + eval to multiple VLMs.

Supports: Qwen3-VL-2B, InternVL3.5-1B, DeepSeek-VL2-Tiny

Usage:
    python scripts/run_multimodel.py --model internvl3_5_1b --stage calibrate
    python scripts/run_multimodel.py --model internvl3_5_1b --stage bon_sft
    python scripts/run_multimodel.py --model internvl3_5_1b --stage eval
    python scripts/run_multimodel.py --model internvl3_5_1b --stage all
    python scripts/run_multimodel.py --model all --stage all  # Everything
"""

import os, sys, gc, json, re, argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from datetime import datetime
from datasets import load_from_disk

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_registry import load_model, load_model_specs, make_chat_prompt
from src.calibrator import CalibrationResult
from src.steerer import ActivationSteerer
from src.profiler import VisionHeadProfiler


# ─── VLMEvalKit standard ───────────────────────────────────────────────────

def process_punctuation(inText):
    outText = inText
    punct = [';', r'/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\',
             '_', '-', '>', '<', '@', '`', ',', '?', '!']
    commaStrip = re.compile(r'(\d)(,)(\d)')
    periodStrip = re.compile(r'(?<!\d)\.(?!\d)')
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (
                re.search(commaStrip, inText) is not None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub('', outText, re.UNICODE)
    return outText


def YOrN_Extraction(output):
    s = output.lower()
    words = process_punctuation(s).split()
    if 'yes' in words and 'no' not in words:
        return 'Yes'
    if 'yes' not in words and 'no' in words:
        return 'No'
    return 'Unknown'


POPE_PROMPT = "{question} Please answer yes or no."


def pope_metrics(records):
    import pandas as pd
    def cal_f1(yt, yp):
        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        return f1, p, r
    df = pd.DataFrame(records)
    results = {}
    for split_name, sub in [("Overall", df)] + [
        (cat, df[df["category"] == cat]) for cat in sorted(df["category"].unique())
    ]:
        yt = np.array([1 if a == "Yes" else 0 for a in sub["answer"]])
        yp = np.array([1 if a == "Yes" else 0 for a in sub["extracted"]])
        score = np.array([1 if a == e else 0 for a, e in zip(sub["answer"], sub["extracted"])])
        f1, p, r = cal_f1(yt, yp)
        results[split_name] = {
            "acc": float(np.mean(score) * 100), "f1": float(f1 * 100),
            "precision": float(p * 100), "recall": float(r * 100),
            "n": len(sub), "n_unknown": int(sum(1 for e in sub["extracted"] if e == "Unknown")),
        }
    return results


# ─── Model-agnostic generation ────────────────────────────────────────────

def generate_answer(model_info, image, question, blind=False, max_tokens=64):
    """Generate answer using any supported model."""
    model = model_info["model"]
    processor = model_info.get("processor")
    tokenizer = model_info["tokenizer"]
    model_type = model_info["model_type"]

    if blind:
        image = Image.new("RGB", image.size, (0, 0, 0))

    prompt = POPE_PROMPT.format(question=question)

    if model_type in ("qwen3_vl", "qwen2_vl"):
        from qwen_vl_utils import process_vision_info
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, _ = process_vision_info(messages, return_video_kwargs=True)
        inputs = processor(text=[text], images=images, videos=videos,
                           return_tensors="pt", padding=True)
        inputs = {k: v.to(model_info["device"]) for k, v in inputs.items()}

    elif model_type == "internvl3":
        # InternVL uses chat method
        pixel_values = _internvl_preprocess(image, model_info)
        inputs = {"pixel_values": pixel_values, "question": prompt}

    elif model_type == "deepseek_vl2":
        # DeepSeek-VL2 uses its own preprocessing
        inputs = _deepseek_preprocess(image, prompt, model_info)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    with torch.no_grad():
        if model_type == "internvl3":
            # InternVL3 uses model.chat()
            try:
                response = model.chat(
                    tokenizer, inputs["pixel_values"], inputs["question"],
                    generation_config={"max_new_tokens": max_tokens, "temperature": 0.01},
                )
                return response.strip()
            except Exception:
                # Fallback: standard generate
                chat_inputs = make_chat_prompt(model_info, prompt, image)
                gen = model.generate(**chat_inputs, max_new_tokens=max_tokens,
                                     temperature=0.01, top_p=0.8)
                out = gen[0][chat_inputs["input_ids"].shape[1]:]
                return tokenizer.decode(out, skip_special_tokens=True).strip()
        else:
            gen = model.generate(**inputs, max_new_tokens=max_tokens,
                                 temperature=0.01, top_p=0.8, top_k=20)
            out = gen[0][inputs["input_ids"].shape[1]:]
            return tokenizer.decode(out, skip_special_tokens=True).strip()


def _internvl_preprocess(image, model_info):
    """Preprocess image for InternVL3."""
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    transform = T.Compose([
        T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    pixel_values = transform(image).unsqueeze(0).to(
        device=model_info["device"], dtype=torch.bfloat16
    )
    return pixel_values


def _deepseek_preprocess(image, prompt, model_info):
    """Preprocess for DeepSeek-VL2."""
    tokenizer = model_info["tokenizer"]
    inputs = tokenizer(f"<image>\n{prompt}", return_tensors="pt")
    return {k: v.to(model_info["device"]) for k, v in inputs.items()}


# ─── Stage 1: Calibration ─────────────────────────────────────────────────

def run_calibration(model_key, args):
    """Run Cohen's d calibration on a model."""
    print(f"\n{'='*60}")
    print(f"CALIBRATE: {model_key}")
    print(f"{'='*60}")

    model_info = load_model(model_key)
    model = model_info["model"]
    model.eval()

    # Use POPE data for calibration (correct/incorrect split)
    dataset = load_from_disk("data/eval/pope")
    n_cal = min(len(dataset), args.cal_samples)

    profiler = VisionHeadProfiler(model_info)

    correct_acts = []
    incorrect_acts = []

    print(f"[cal] Profiling {n_cal} samples...")
    for i in range(n_cal):
        s = dataset[i]
        try:
            raw = generate_answer(model_info, s["image"], s["question"])
            ext = YOrN_Extraction(raw)
            gt = s["answer"].strip().capitalize()

            # Get per-head activations
            acts = profiler.profile_sample(s["image"], s["question"])

            if ext == gt:
                correct_acts.append(acts)
            else:
                incorrect_acts.append(acts)

        except Exception as e:
            continue

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{n_cal}] correct={len(correct_acts)}, incorrect={len(incorrect_acts)}")

    print(f"\n[cal] {len(correct_acts)} correct, {len(incorrect_acts)} incorrect")

    if len(incorrect_acts) < 10:
        print("[WARN] Too few incorrect samples for reliable calibration")

    # Compute Cohen's d and save
    calibration = profiler.compute_calibration(correct_acts, incorrect_acts, top_k=20)

    cal_dir = Path(f"checkpoints/calibration/{model_key}")
    cal_dir.mkdir(parents=True, exist_ok=True)
    calibration.save(str(cal_dir))
    print(f"[cal] Saved to {cal_dir}")

    # Print top heads
    print(f"\n[cal] Top 10 vision heads by Cohen's d:")
    for li, hi in calibration.top_heads[:10]:
        d = calibration.head_scores.get((li, hi), {}).get("cohens_d", 0)
        print(f"  Layer {li}, Head {hi}: d={d:.3f}")

    del model_info, model
    torch.cuda.empty_cache()
    return calibration


# ─── Stage 2: Evaluation ──────────────────────────────────────────────────

def run_eval(model_key, args):
    """Run POPE evaluation with baseline + steering."""
    print(f"\n{'='*60}")
    print(f"EVAL: {model_key}")
    print(f"{'='*60}")

    model_info = load_model(model_key)
    model = model_info["model"]
    model.eval()

    dataset = load_from_disk("data/eval/pope")
    n_eval = args.eval_samples

    results = {}

    # ── Baseline ──
    records = []
    print(f"\n[eval] Baseline — {n_eval} samples")
    for i in range(n_eval):
        s = dataset[i]
        try:
            raw = generate_answer(model_info, s["image"], s["question"])
        except Exception:
            raw = ""
        ext = YOrN_Extraction(raw)
        records.append({
            "index": i, "question": s["question"],
            "answer": s["answer"].strip().capitalize(),
            "prediction": raw, "extracted": ext,
            "category": s.get("category", "unknown"),
        })
        if (i + 1) % 200 == 0:
            acc = pope_metrics(records)["Overall"]["acc"]
            print(f"  [{i+1}/{n_eval}] acc={acc:.1f}%")

    met = pope_metrics(records)
    results["baseline"] = met
    print(f"  → baseline: acc={met['Overall']['acc']:.1f}%, F1={met['Overall']['f1']:.1f}%")

    # ── Blind baseline ──
    records_b = []
    print(f"\n[eval] Blind — {min(n_eval, 300)} samples")
    for i in range(min(n_eval, 300)):
        s = dataset[i]
        try:
            raw = generate_answer(model_info, s["image"], s["question"], blind=True)
        except Exception:
            raw = ""
        ext = YOrN_Extraction(raw)
        records_b.append({
            "index": i, "question": s["question"],
            "answer": s["answer"].strip().capitalize(),
            "prediction": raw, "extracted": ext,
            "category": s.get("category", "unknown"),
        })

    met_b = pope_metrics(records_b)
    results["blind"] = met_b
    gap = met["Overall"]["acc"] - met_b["Overall"]["acc"]
    print(f"  → blind: acc={met_b['Overall']['acc']:.1f}%, gap={gap:.1f}pp")

    # ── Steered (if calibration exists) ──
    cal_path = Path(f"checkpoints/calibration/{model_key}")
    if cal_path.exists():
        calibration = CalibrationResult.load(str(cal_path))
        steerer = ActivationSteerer(
            model_info, calibration,
            steer_layers_start=model_info.get("steer_layers_start", 0)
        )
        steerer.steer(alpha=3.0)
        print(f"\n[eval] Steered (α=3) — {n_eval} samples")

        records_s = []
        for i in range(n_eval):
            s = dataset[i]
            try:
                raw = generate_answer(model_info, s["image"], s["question"])
            except Exception:
                raw = ""
            ext = YOrN_Extraction(raw)
            records_s.append({
                "index": i, "question": s["question"],
                "answer": s["answer"].strip().capitalize(),
                "prediction": raw, "extracted": ext,
                "category": s.get("category", "unknown"),
            })
            if (i + 1) % 200 == 0:
                acc = pope_metrics(records_s)["Overall"]["acc"]
                print(f"  [{i+1}/{n_eval}] acc={acc:.1f}%")

        met_s = pope_metrics(records_s)
        results["steered"] = met_s
        print(f"  → steered: acc={met_s['Overall']['acc']:.1f}%")

        steerer.cleanup()

    # ── Summary ──
    out_dir = Path(f"lab/reports/multimodel/{model_key}")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {
        "model": model_key,
        "timestamp": ts,
        "n_samples": n_eval,
        "results": {k: v["Overall"] for k, v in results.items()},
        "gap": gap,
    }

    with open(out_dir / f"eval_{ts}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"EVAL SUMMARY: {model_key} ({n_eval} samples)")
    print(f"{'='*60}")
    print(f"{'Condition':<15} {'Acc':>7} {'F1':>7} {'P':>7} {'R':>7}")
    print("-" * 60)
    for label, m in results.items():
        o = m["Overall"]
        print(f"{label:<15} {o['acc']:>6.1f}% {o['f1']:>6.1f}% "
              f"{o['precision']:>6.1f}% {o['recall']:>6.1f}%")
    print(f"Blind Gap: {gap:.1f}pp")
    print("=" * 60)

    del model_info, model
    torch.cuda.empty_cache()
    return results


# ─── Main ─────────────────────────────────────────────────────────────────

TRAINABLE_MODELS = ["qwen3_vl_2b", "internvl3_5_1b", "deepseek_vl2_tiny"]

def main():
    parser = argparse.ArgumentParser(description="VIGIL Multi-Model Pipeline")
    parser.add_argument("--model", type=str, required=True,
                        help="Model key or 'all'")
    parser.add_argument("--stage", type=str, required=True,
                        choices=["calibrate", "eval", "bon_sft", "all"])
    parser.add_argument("--cal-samples", type=int, default=500)
    parser.add_argument("--eval-samples", type=int, default=500)
    args = parser.parse_args()

    if args.model == "all":
        models = TRAINABLE_MODELS
    else:
        models = [args.model]

    for model_key in models:
        print(f"\n{'#'*60}")
        print(f"# MODEL: {model_key}")
        print(f"{'#'*60}")

        if args.stage in ("calibrate", "all"):
            run_calibration(model_key, args)

        if args.stage in ("eval", "all"):
            run_eval(model_key, args)

        # BoN+SFT is model-specific, needs more work per model
        if args.stage in ("bon_sft", "all"):
            print(f"[TODO] BoN+SFT for {model_key} — requires model-specific prompt handling")


if __name__ == "__main__":
    main()
