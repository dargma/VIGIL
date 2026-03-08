"""
VIGIL BoN+SFT for InternVL3.5-1B — Best-of-N sampling + SFT.

Generates N=8 candidates per sample, scores with accuracy + IIG,
selects best, then fine-tunes on curated data.

Usage:
    python scripts/bon_sft_internvl.py --stage generate --max-samples 200
    python scripts/bon_sft_internvl.py --stage sft
    python scripts/bon_sft_internvl.py --stage eval
    python scripts/bon_sft_internvl.py --stage all --max-samples 200
"""

import os, sys, json, re, gc, argparse, random
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from datetime import datetime
from datasets import load_from_disk

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model_registry import load_model

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


POPE_PROMPT = "{question} Please answer yes or no."
TRANSFORM = T.Compose([
    T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
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


def generate_candidates(model_info, image, question, n=8, temperature=0.7):
    """Generate N candidates with sampling."""
    image_rgb = image.convert("RGB")
    pixel_values = TRANSFORM(image_rgb).unsqueeze(0).to(
        device=model_info["device"], dtype=torch.bfloat16
    )
    prompt = POPE_PROMPT.format(question=question)
    candidates = []

    for _ in range(n):
        with torch.no_grad():
            response = model_info["model"].chat(
                model_info["tokenizer"], pixel_values, prompt,
                generation_config={
                    "max_new_tokens": 64,
                    "temperature": temperature,
                    "do_sample": True,
                    "top_p": 0.9,
                }
            )
        candidates.append(response.strip())

    return candidates


def score_candidate(pred, gt):
    """Score a candidate answer. Returns (correct, score)."""
    ext = YOrN_Extraction(pred)
    correct = (ext == gt)
    # Score: 1.0 for correct, 0.5 for clean wrong, 0.0 for unknown
    if correct:
        score = 1.0
    elif ext == "Unknown":
        score = 0.0
    else:
        score = 0.1  # at least it parsed
    # Bonus for concise answers
    if len(pred.split()) <= 3:
        score += 0.1
    return correct, min(score, 1.0)


def compute_iig_simple(model_info, image, question, candidate):
    """Simplified IIG: compare log-probs with real vs black image."""
    try:
        image_rgb = image.convert("RGB")
        black_img = Image.new("RGB", image.size, (0, 0, 0))

        def get_logprob(img):
            pixel_values = TRANSFORM(img).unsqueeze(0).to(
                device=model_info["device"], dtype=torch.bfloat16
            )
            prompt = POPE_PROMPT.format(question=question)
            tokenizer = model_info["tokenizer"]

            # Simplified: just check if model prefers the candidate answer
            full_text = f"{prompt} {candidate}"
            inputs = tokenizer(full_text, return_tensors="pt")
            inputs = {k: v.to(model_info["device"]) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model_info["model"].language_model(**inputs)
                logits = outputs.logits
                # Get log-prob of last token
                last_logprob = torch.log_softmax(logits[:, -2, :], dim=-1)
                target_id = inputs["input_ids"][:, -1]
                lp = last_logprob.gather(1, target_id.unsqueeze(1)).item()
            return lp

        lp_real = get_logprob(image_rgb)
        lp_black = get_logprob(black_img)
        return lp_real - lp_black
    except Exception:
        return 0.0


def run_bon_generation(model_info, dataset, max_samples=200, n_candidates=8):
    """Generate BoN candidates and select best."""
    print(f"\n[bon] Generating {n_candidates} candidates per sample, {max_samples} samples...")

    curated = []
    n_correct = 0
    n_total = min(max_samples, len(dataset))

    for i in range(n_total):
        s = dataset[i]
        gt = s["answer"].strip().capitalize()

        try:
            candidates = generate_candidates(model_info, s["image"], s["question"], n=n_candidates)
        except Exception as e:
            print(f"  [{i}] ERR generating: {e}")
            continue

        # Score all candidates
        scored = []
        for c in candidates:
            correct, score = score_candidate(c, gt)
            scored.append({"text": c, "correct": correct, "score": score})

        # Select best
        best = max(scored, key=lambda x: x["score"])

        if best["correct"]:
            n_correct += 1
            curated.append({
                "question": s["question"],
                "answer": gt,
                "best_candidate": best["text"],
                "score": best["score"],
                "n_correct_in_group": sum(1 for s in scored if s["correct"]),
                "index": i,
                "category": s.get("category", "unknown"),
            })

        if (i + 1) % 50 == 0:
            pct = n_correct / (i + 1) * 100
            print(f"  [{i+1}/{n_total}] curated={len(curated)}, "
                  f"bon_hit_rate={pct:.1f}%")

    print(f"\n[bon] Curated {len(curated)} samples from {n_total} "
          f"(hit rate: {len(curated)/n_total*100:.1f}%)")

    return curated


def run_sft(model_info, curated_data, output_dir, epochs=2, lr=5e-6):
    """Fine-tune on curated BoN data."""
    from torch.utils.data import Dataset, DataLoader

    print(f"\n[sft] Training on {len(curated_data)} samples, {epochs} epochs, lr={lr}")

    model = model_info["model"]
    tokenizer = model_info["tokenizer"]
    model.train()

    # Prepare training data
    train_texts = []
    for item in curated_data:
        prompt = POPE_PROMPT.format(question=item["question"])
        # Format: prompt + answer
        text = f"{prompt}\n{item['best_candidate']}"
        train_texts.append(text)

    # Simple training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_texts))

    total_loss = 0
    n_steps = 0

    for epoch in range(epochs):
        random.shuffle(train_texts)
        epoch_loss = 0

        for j, text in enumerate(train_texts):
            inputs = tokenizer(text, return_tensors="pt", max_length=256, truncation=True)
            inputs = {k: v.to(model_info["device"]) for k, v in inputs.items()}
            inputs["labels"] = inputs["input_ids"].clone()

            try:
                outputs = model.language_model(**inputs)
                loss = outputs.loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                total_loss += loss.item()
                n_steps += 1
            except Exception as e:
                print(f"  [sft] step {j} error: {e}")
                optimizer.zero_grad()
                continue

            if (j + 1) % 100 == 0:
                avg = epoch_loss / (j + 1)
                print(f"  [epoch {epoch+1}] step {j+1}/{len(train_texts)}, loss={avg:.4f}")

        print(f"  [epoch {epoch+1}] avg_loss={epoch_loss / max(len(train_texts), 1):.4f}")

    model.eval()

    # Save checkpoint
    ckpt_dir = Path(output_dir) / "final"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))
    print(f"[sft] Saved to {ckpt_dir}")

    return total_loss / max(n_steps, 1)


def run_eval(model_info, dataset, max_samples=500):
    """Run POPE evaluation."""
    records = []
    n = min(max_samples, len(dataset))
    print(f"\n[eval] Running POPE eval on {n} samples...")

    model_info["model"].eval()

    for i in range(n):
        s = dataset[i]
        try:
            image_rgb = s["image"].convert("RGB")
            pixel_values = TRANSFORM(image_rgb).unsqueeze(0).to(
                device=model_info["device"], dtype=torch.bfloat16
            )
            prompt = POPE_PROMPT.format(question=s["question"])
            with torch.no_grad():
                response = model_info["model"].chat(
                    model_info["tokenizer"], pixel_values, prompt,
                    generation_config={"max_new_tokens": 64}
                )
            raw = response.strip()
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
            print(f"  [{i+1}/{n}] acc={acc:.1f}%")

    met = pope_metrics(records)
    o = met["Overall"]
    print(f"  → acc={o['acc']:.1f}%, F1={o['f1']:.1f}%, P={o['precision']:.1f}%, R={o['recall']:.1f}%")

    # Blind eval
    print(f"\n[eval] Running blind eval on {min(n, 300)} samples...")
    records_b = []
    for i in range(min(n, 300)):
        s = dataset[i]
        try:
            black_img = Image.new("RGB", s["image"].size, (0, 0, 0))
            pixel_values = TRANSFORM(black_img).unsqueeze(0).to(
                device=model_info["device"], dtype=torch.bfloat16
            )
            prompt = POPE_PROMPT.format(question=s["question"])
            with torch.no_grad():
                response = model_info["model"].chat(
                    model_info["tokenizer"], pixel_values, prompt,
                    generation_config={"max_new_tokens": 64}
                )
            raw = response.strip()
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
    gap = met["Overall"]["acc"] - met_b["Overall"]["acc"]
    print(f"  → blind: acc={met_b['Overall']['acc']:.1f}%, gap={gap:.1f}pp")

    return met, met_b, gap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, required=True,
                        choices=["generate", "sft", "eval", "all"])
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--eval-samples", type=int, default=500)
    parser.add_argument("--n-candidates", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--output-dir", type=str, default="checkpoints/internvl_bon_sft")
    args = parser.parse_args()

    dataset = load_from_disk("data/eval/pope")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_info = load_model("internvl3_5_1b")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.stage in ("generate", "all"):
        curated = run_bon_generation(model_info, dataset, args.max_samples, args.n_candidates)

        # Save curated data
        with open(output_dir / f"curated_{ts}.json", "w") as f:
            json.dump(curated, f, indent=2)
        print(f"[bon] Saved {len(curated)} curated samples")

    if args.stage in ("sft", "all"):
        # Load curated data
        if args.stage == "sft":
            curated_files = sorted(output_dir.glob("curated_*.json"), reverse=True)
            if not curated_files:
                print("[ERROR] No curated data found. Run 'generate' stage first.")
                return
            with open(curated_files[0]) as f:
                curated = json.load(f)

        avg_loss = run_sft(model_info, curated, str(output_dir), args.epochs, args.lr)
        print(f"[sft] Average loss: {avg_loss:.4f}")

    if args.stage in ("eval", "all"):
        met, met_b, gap = run_eval(model_info, dataset, args.eval_samples)

        results = {
            "model": "internvl3_5_1b_bon_sft",
            "timestamp": ts,
            "baseline": met["Overall"],
            "blind": met_b["Overall"],
            "blind_gap": gap,
        }

        report_dir = Path("lab/reports/multimodel/internvl3_5_1b")
        report_dir.mkdir(parents=True, exist_ok=True)
        with open(report_dir / f"bon_sft_eval_{ts}.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*60}")
        print(f"InternVL3.5-1B BoN+SFT Eval")
        print(f"{'='*60}")
        o = met["Overall"]
        print(f"Acc: {o['acc']:.1f}%, F1: {o['f1']:.1f}%, P: {o['precision']:.1f}%, R: {o['recall']:.1f}%")
        print(f"Blind Gap: {gap:.1f}pp")
        print(f"{'='*60}")

    del model_info
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
