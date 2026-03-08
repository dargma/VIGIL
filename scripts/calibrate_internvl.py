"""
VIGIL Calibration for InternVL3.5-1B — identify vision heads via Cohen's d.

Usage:
    python scripts/calibrate_internvl.py --max-samples 300
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

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

os.environ["PYTHONUNBUFFERED"] = "1"

POPE_PROMPT = "{question} Please answer yes or no."
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TRANSFORM = T.Compose([
    T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


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


def internvl_process_fn(model_info, sample):
    """Process function for VisionHeadProfiler — InternVL3.5-specific."""
    image = sample["image"].convert("RGB")
    question = sample["question"]
    gt = sample["answer"].strip().capitalize()

    pixel_values = TRANSFORM(image).unsqueeze(0).to(
        device=model_info["device"], dtype=torch.bfloat16
    )

    prompt = POPE_PROMPT.format(question=question)
    tokenizer = model_info["tokenizer"]

    # InternVL3.5 uses a specific prompt format
    # Build inputs manually for the profiler (need input_ids for forward pass)
    query = f"<image>\n{prompt}"
    inputs = tokenizer(query, return_tensors="pt")
    inputs = {k: v.to(model_info["device"]) for k, v in inputs.items()}
    inputs["pixel_values"] = pixel_values

    return inputs, gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--output-dir", type=str, default="checkpoints/calibration/internvl3_5_1b")
    args = parser.parse_args()

    dataset = load_from_disk("data/eval/pope")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("InternVL3.5-1B Calibration")
    print("=" * 60)

    model_info = load_model("internvl3_5_1b")

    # Run calibration via profiler
    # BUT: VisionHeadProfiler.profile() needs model(**inputs) to work
    # InternVL3.5 might not support this directly (uses chat() API)
    # Let's do manual calibration instead

    profiler = VisionHeadProfiler(model_info)
    profiler._install_hooks()

    correct_acts = {}   # (li, hi) -> list of norms
    incorrect_acts = {}
    n_correct = n_incorrect = n_error = 0
    n_total = min(args.max_samples, len(dataset))

    print(f"\n[cal] Profiling {n_total} samples...")

    for i in range(n_total):
        s = dataset[i]
        try:
            image = s["image"].convert("RGB")
            pixel_values = TRANSFORM(image).unsqueeze(0).to(
                device=model_info["device"], dtype=torch.bfloat16
            )

            prompt = POPE_PROMPT.format(question=s["question"])
            model = model_info["model"]
            tokenizer = model_info["tokenizer"]

            # Use model.chat() which triggers the forward hooks
            response = model.chat(tokenizer, pixel_values, prompt,
                                   generation_config={"max_new_tokens": 64})
            ext = YOrN_Extraction(response.strip())
            gt = s["answer"].strip().capitalize()

            is_correct = (ext == gt)
            bucket = correct_acts if is_correct else incorrect_acts
            if is_correct:
                n_correct += 1
            else:
                n_incorrect += 1

            # Extract per-head activations from captured hooks
            for li in range(profiler.num_layers):
                head_acts = profiler._get_per_head_activations(li)
                if head_acts is None:
                    continue
                for hi in range(profiler.num_heads):
                    key = (li, hi)
                    norm = head_acts[0, hi, :].float().norm().item()
                    bucket.setdefault(key, []).append(norm)

            profiler._captured.clear()

        except Exception as e:
            n_error += 1
            if n_error <= 5:
                print(f"  [{i}] ERR: {e}")
            continue

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n_total}] correct={n_correct}, incorrect={n_incorrect}, errors={n_error}")

    profiler._remove_hooks()

    print(f"\n[cal] Final: {n_correct} correct, {n_incorrect} incorrect, {n_error} errors")

    if n_incorrect < 5:
        print("[WARN] Too few incorrect samples for reliable calibration")
        if n_correct > 0 and n_incorrect == 0:
            print("[INFO] Model got everything right — try more samples or harder data")
        return

    # Compute Cohen's d
    head_scores = {}
    for li in range(profiler.num_layers):
        for hi in range(profiler.num_heads):
            key = (li, hi)
            c_norms = correct_acts.get(key, [])
            w_norms = incorrect_acts.get(key, [])

            if len(c_norms) >= 5 and len(w_norms) >= 5:
                c_arr = np.array(c_norms)
                w_arr = np.array(w_norms)
                pooled_std = np.sqrt(
                    (c_arr.var() * (len(c_arr) - 1) + w_arr.var() * (len(w_arr) - 1))
                    / (len(c_arr) + len(w_arr) - 2)
                )
                if pooled_std > 1e-8:
                    d = abs(c_arr.mean() - w_arr.mean()) / pooled_std
                else:
                    d = 0.0
                head_scores[f"{li},{hi}"] = {
                    "cohens_d": float(d),
                    "correct_mean": float(c_arr.mean()),
                    "incorrect_mean": float(w_arr.mean()),
                    "n_correct": len(c_arr),
                    "n_incorrect": len(w_arr),
                }

    # Rank by Cohen's d
    ranked = sorted(head_scores.items(), key=lambda x: x[1]["cohens_d"], reverse=True)
    top_heads = [(int(k.split(",")[0]), int(k.split(",")[1])) for k, _ in ranked[:20]]

    print(f"\n[cal] Top 20 vision heads by Cohen's d:")
    for k, v in ranked[:20]:
        li, hi = k.split(",")
        print(f"  Layer {li}, Head {hi}: d={v['cohens_d']:.3f} "
              f"(correct={v['correct_mean']:.2f}, incorrect={v['incorrect_mean']:.2f})")

    # Save calibration results
    cal_data = {
        "model": "internvl3_5_1b",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "n_errors": n_error,
        "top_heads": top_heads,
        "head_scores": head_scores,
    }

    with open(output_dir / "calibration.json", "w") as f:
        json.dump(cal_data, f, indent=2)
    print(f"\n[cal] Saved to {output_dir / 'calibration.json'}")

    # Analyze head types
    decision_heads = [(li, hi) for li, hi in top_heads if li < 14]
    feature_heads = [(li, hi) for li, hi in top_heads if li >= 14]
    print(f"\n[cal] Head type analysis:")
    print(f"  Decision heads (L0-13): {len(decision_heads)} — {decision_heads[:5]}")
    print(f"  Feature heads (L14+):   {len(feature_heads)} — {feature_heads[:5]}")


if __name__ == "__main__":
    main()
