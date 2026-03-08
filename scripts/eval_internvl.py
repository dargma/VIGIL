"""
VIGIL InternVL3.5-1B POPE Evaluation — baseline + blind + calibration.

Usage:
    python scripts/eval_internvl.py --max-samples 500
    python scripts/eval_internvl.py --max-samples 3000  # full adversarial
"""

import os, sys, json, re, gc, argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from datetime import datetime
from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model_registry import load_model
from src.profiler import VisionHeadProfiler
from src.calibrator import CalibrationResult

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


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

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TRANSFORM = T.Compose([
    T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


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


def generate_one(model_info, image, question, blind=False):
    """Generate answer using InternVL3.5 model.chat() API."""
    if blind:
        image = Image.new("RGB", image.size, (0, 0, 0))

    image_rgb = image.convert("RGB")
    pixel_values = TRANSFORM(image_rgb).unsqueeze(0).to(
        device=model_info["device"], dtype=torch.bfloat16
    )

    prompt = POPE_PROMPT.format(question=question)
    with torch.no_grad():
        response = model_info["model"].chat(
            model_info["tokenizer"], pixel_values, prompt,
            generation_config={"max_new_tokens": 64}
        )
    return response.strip()


def run_pope(model_info, dataset, label, blind=False, max_n=None):
    n = min(len(dataset), max_n or len(dataset))
    records = []
    print(f"\n[pope] {label} ({'blind' if blind else 'real'}) — {n} samples")

    for i in range(n):
        s = dataset[i]
        try:
            raw = generate_one(model_info, s["image"], s["question"], blind=blind)
        except Exception as e:
            print(f"  [{i}] ERR: {e}")
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
            print(f"  [{i+1}/{n}] acc={acc:.1f}%")

    metrics = pope_metrics(records)
    o = metrics["Overall"]
    print(f"  → {label}: acc={o['acc']:.1f}%, F1={o['f1']:.1f}%, P={o['precision']:.1f}%, R={o['recall']:.1f}%")
    return records, metrics


def run_calibration(model_info, dataset, n_cal=500):
    """Run vision head calibration using Cohen's d."""
    print(f"\n[cal] Profiling {n_cal} samples for calibration...")
    profiler = VisionHeadProfiler(model_info)

    correct_acts = []
    incorrect_acts = []

    for i in range(min(n_cal, len(dataset))):
        s = dataset[i]
        try:
            raw = generate_one(model_info, s["image"], s["question"])
            ext = YOrN_Extraction(raw)
            gt = s["answer"].strip().capitalize()

            acts = profiler.profile_sample(s["image"], s["question"])

            if ext == gt:
                correct_acts.append(acts)
            else:
                incorrect_acts.append(acts)
        except Exception:
            continue

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{n_cal}] correct={len(correct_acts)}, incorrect={len(incorrect_acts)}")

    print(f"\n[cal] {len(correct_acts)} correct, {len(incorrect_acts)} incorrect")

    if len(incorrect_acts) < 5:
        print("[WARN] Too few incorrect samples for reliable calibration")
        return None

    calibration = profiler.compute_calibration(correct_acts, incorrect_acts, top_k=20)

    cal_dir = Path("checkpoints/calibration/internvl3_5_1b")
    cal_dir.mkdir(parents=True, exist_ok=True)
    calibration.save(str(cal_dir))
    print(f"[cal] Saved to {cal_dir}")

    print(f"\n[cal] Top 10 vision heads by Cohen's d:")
    for li, hi in calibration.top_heads[:10]:
        d = calibration.head_scores.get((li, hi), {}).get("cohens_d", 0)
        print(f"  Layer {li}, Head {hi}: d={d:.3f}")

    return calibration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--cal-samples", type=int, default=500)
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--skip-blind", action="store_true")
    parser.add_argument("--output-dir", type=str, default="lab/reports/multimodel/internvl3_5_1b")
    args = parser.parse_args()

    dataset = load_from_disk("data/eval/pope")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load model
    print("=" * 60)
    print("Loading InternVL3.5-1B")
    print("=" * 60)
    model_info = load_model("internvl3_5_1b")

    all_results = {}

    # ── Baseline eval ──
    recs, met = run_pope(model_info, dataset, "baseline", max_n=args.max_samples)
    all_results["baseline"] = {"records": recs, "metrics": met}

    # ── Blind eval ──
    if not args.skip_blind:
        blind_n = min(args.max_samples, 500)
        recs_b, met_b = run_pope(model_info, dataset, "blind", blind=True, max_n=blind_n)
        all_results["blind"] = {"records": recs_b, "metrics": met_b}

    # ── Calibration ──
    if not args.skip_calibration:
        calibration = run_calibration(model_info, dataset, n_cal=args.cal_samples)
        if calibration is not None:
            all_results["calibration"] = {
                "top_heads": [(li, hi) for li, hi in calibration.top_heads[:20]],
                "n_correct": len([1 for _ in range(10)]),  # placeholder
            }

    # ── Summary ──
    summary = {
        "model": "internvl3_5_1b",
        "timestamp": ts,
        "n_samples": args.max_samples,
    }

    for label, data in all_results.items():
        if "metrics" in data:
            summary[label] = data["metrics"]["Overall"]
            with open(output_dir / f"{label}_{ts}.json", "w") as f:
                json.dump({"records": data.get("records", []), "metrics": data["metrics"]}, f, indent=2)

    # Compute blind gap
    if "baseline" in summary and "blind" in summary:
        summary["blind_gap"] = summary["baseline"]["acc"] - summary["blind"]["acc"]

    with open(output_dir / f"summary_{ts}.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print table
    print(f"\n{'='*70}")
    print(f"InternVL3.5-1B POPE Eval — {args.max_samples} samples")
    print(f"{'='*70}")
    print(f"{'Condition':<15} {'Acc':>7} {'F1':>7} {'P':>7} {'R':>7} {'Unk':>5}")
    print("-" * 70)
    for label in ["baseline", "blind"]:
        if label in summary and isinstance(summary[label], dict):
            m = summary[label]
            print(f"{label:<15} {m['acc']:>6.1f}% {m['f1']:>6.1f}% "
                  f"{m['precision']:>6.1f}% {m['recall']:>6.1f}% {m['n_unknown']:>5}")
    if "blind_gap" in summary:
        print(f"\nBlind Gap: {summary['blind_gap']:.1f}pp")
    print("=" * 70)

    del model_info
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
