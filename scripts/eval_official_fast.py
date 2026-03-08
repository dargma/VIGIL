"""
VIGIL Official Fast Evaluation — loads each model ONCE, runs all conditions.

Usage:
    python scripts/eval_official_fast.py --max-samples 500
    python scripts/eval_official_fast.py  # full 9000
"""

import os, sys, json, re, argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from datetime import datetime
from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.steerer import ActivationSteerer
from src.calibrator import CalibrationResult


# ─── VLMEvalKit functions (inlined, exact copies) ──────────────────────────

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
    """POPE_rating equivalent."""
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
            "acc": float(np.mean(score) * 100),
            "f1": float(f1 * 100),
            "precision": float(p * 100),
            "recall": float(r * 100),
            "n": len(sub),
            "n_unknown": int(sum(1 for e in sub["extracted"] if e == "Unknown")),
        }
    return results


def load_qwen3vl(model_path=None):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    hf_id = model_path or "Qwen/Qwen3-VL-2B-Instruct"
    print(f"[eval] Loading: {hf_id}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    model.eval()
    return model, processor


def generate_one(model, processor, image, question, blind=False):
    from qwen_vl_utils import process_vision_info
    if blind:
        image = Image.new("RGB", image.size, (0, 0, 0))
    prompt = POPE_PROMPT.format(question=question)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, _ = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(text=[text], images=images, videos=videos,
                       return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=64, temperature=0.01,
                             top_p=0.8, top_k=20, repetition_penalty=1.0)
    out = gen[0][inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(out, skip_special_tokens=True).strip()


def setup_steering(model, alpha=5.0):
    cal_path = Path("checkpoints/calibration/qwen3_vl_2b")
    if not cal_path.exists():
        print("[eval] No calibration found, skipping steering")
        return None
    calibration = CalibrationResult.load(str(cal_path))
    model_info = {
        "get_layers_fn": lambda: model.model.language_model.layers,
        "num_heads": 16, "head_dim": 128,
    }
    steerer = ActivationSteerer(model_info, calibration, steer_layers_start=4)
    steerer.steer(alpha)
    print(f"[eval] Steering active, alpha={alpha}")
    return steerer


def run_pope(model, processor, dataset, label, blind=False, max_n=None):
    n = min(len(dataset), max_n or len(dataset))
    records = []
    print(f"\n[pope] {label} ({'blind' if blind else 'real'}) — {n} samples")

    for i in range(n):
        s = dataset[i]
        try:
            raw = generate_one(model, processor, s["image"], s["question"], blind=blind)
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
    print(f"  → {label}: acc={metrics['Overall']['acc']:.1f}%, F1={metrics['Overall']['f1']:.1f}%")
    return records, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--output-dir", type=str, default="lab/reports/official_eval")
    args = parser.parse_args()

    dataset = load_from_disk("data/eval/pope")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {}

    # ── Group 1: Baseline model (real + blind + steered + steered_blind) ──
    print("\n" + "="*60)
    print("Loading baseline Qwen3-VL-2B-Instruct")
    print("="*60)
    model, processor = load_qwen3vl()

    for label, blind, steer in [
        ("baseline", False, False),
        ("baseline_blind", True, False),
        ("steered", False, True),
        ("steered_blind", True, True),
    ]:
        steerer = None
        if steer:
            steerer = setup_steering(model, args.alpha)

        recs, met = run_pope(model, processor, dataset, label,
                             blind=blind, max_n=args.max_samples)
        all_results[label] = {"records": recs, "metrics": met}

        if steerer:
            steerer.cleanup()

    del model
    torch.cuda.empty_cache()

    # ── Group 2: BoN+SFT checkpoints ──
    bon_paths = [
        ("bon_r1", "checkpoints/block2_bon/final"),
    ]
    for name, path in bon_paths:
        if not Path(path).exists():
            print(f"[skip] {path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Loading {name}: {path}")
        print("="*60)
        model, processor = load_qwen3vl(path)

        for label, blind in [(name, False), (f"{name}_blind", True)]:
            recs, met = run_pope(model, processor, dataset, label,
                                 blind=blind, max_n=args.max_samples)
            all_results[label] = {"records": recs, "metrics": met}

        del model
        torch.cuda.empty_cache()

    # ── Summary ──
    summary = {"timestamp": ts, "n_samples": args.max_samples or len(dataset),
               "conditions": {}}
    for label, data in all_results.items():
        summary["conditions"][label] = data["metrics"]["Overall"]
        # Save per-condition
        with open(output_dir / f"{label}_{ts}.json", "w") as f:
            json.dump({"records": data["records"], "metrics": data["metrics"]}, f, indent=2)

    # Compute blind gaps
    for base in ["baseline", "steered", "bon_r1"]:
        blind_key = f"{base}_blind"
        if base in summary["conditions"] and blind_key in summary["conditions"]:
            real = summary["conditions"][base]["acc"]
            blind = summary["conditions"][blind_key]["acc"]
            summary["conditions"][base]["blind_gap"] = real - blind

    with open(output_dir / f"summary_{ts}.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print table
    print(f"\n{'='*80}")
    print(f"POPE Official Eval (VLMEvalKit standard) — {summary['n_samples']} samples")
    print(f"{'='*80}")
    print(f"{'Condition':<20} {'Acc':>7} {'F1':>7} {'P':>7} {'R':>7} {'Gap':>8} {'Unk':>5}")
    print("-" * 80)
    for c, m in summary["conditions"].items():
        gap = f"{m.get('blind_gap', 0):.1f}pp" if "blind_gap" in m else "-"
        print(f"{c:<20} {m['acc']:>6.1f}% {m['f1']:>6.1f}% "
              f"{m['precision']:>6.1f}% {m['recall']:>6.1f}% {gap:>8} {m['n_unknown']:>5}")
    print("="*80)


if __name__ == "__main__":
    main()
