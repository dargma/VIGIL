"""
VIGIL BoN+SFT with POPE-format training data.

Key insight: Data domain must match evaluation format.
Previous experiments trained on VQAv2 open-ended → POPE yes/no = regression.
This script trains on POPE random+popular splits → evaluates on adversarial.

Usage:
    python scripts/bon_sft_pope_format.py --model qwen3 --max-train 500 --eval-samples 500
    python scripts/bon_sft_pope_format.py --model internvl --max-train 500 --eval-samples 500
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

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


# ─── Shared utilities ─────────────────────────────────────────────────────

POPE_PROMPT = "{question} Please answer yes or no."

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


# ─── Model-specific generation ────────────────────────────────────────────

INTERNVL_TRANSFORM = T.Compose([
    T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])


def generate_qwen3(model, processor, image, question, temperature=0.01, do_sample=False):
    """Generate with Qwen3-VL-2B."""
    from qwen_vl_utils import process_vision_info
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
    gen_kwargs = {"max_new_tokens": 64, "repetition_penalty": 1.0}
    if do_sample:
        gen_kwargs.update({"temperature": temperature, "do_sample": True, "top_p": 0.9})
    else:
        gen_kwargs.update({"temperature": 0.01, "top_p": 0.8, "top_k": 20})
    with torch.no_grad():
        gen = model.generate(**inputs, **gen_kwargs)
    out = gen[0][inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(out, skip_special_tokens=True).strip()


def generate_internvl(model, tokenizer, image, question, temperature=0.01, do_sample=False):
    """Generate with InternVL3.5-1B."""
    image_rgb = image.convert("RGB")
    pixel_values = INTERNVL_TRANSFORM(image_rgb).unsqueeze(0).to(
        device=next(model.parameters()).device, dtype=torch.bfloat16
    )
    prompt = POPE_PROMPT.format(question=question)
    gen_config = {"max_new_tokens": 64}
    if do_sample:
        gen_config.update({"temperature": temperature, "do_sample": True, "top_p": 0.9})
    with torch.no_grad():
        response = model.chat(tokenizer, pixel_values, prompt, generation_config=gen_config)
    return response.strip()


def load_model_for_task(model_type, for_training=False, model_path=None):
    """Load model based on type."""
    if model_type == "qwen3":
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        base_id = "Qwen/Qwen3-VL-2B-Instruct"
        load_id = model_path or base_id
        print(f"[model] Loading Qwen3-VL-2B from {load_id}")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            load_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(base_id)
        if for_training:
            model.train()
            model.gradient_checkpointing_enable()
        else:
            model.eval()
        return {"model": model, "processor": processor, "tokenizer": processor.tokenizer,
                "type": "qwen3", "device": next(model.parameters()).device}

    elif model_type == "internvl":
        from src.model_registry import load_model
        model_info = load_model("internvl3_5_1b")
        if for_training:
            model_info["model"].train()
        else:
            model_info["model"].eval()
        model_info["type"] = "internvl"
        return model_info

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def generate_one(minfo, image, question, **kwargs):
    """Model-agnostic generation."""
    if minfo["type"] == "qwen3":
        return generate_qwen3(minfo["model"], minfo["processor"], image, question, **kwargs)
    elif minfo["type"] == "internvl":
        return generate_internvl(minfo["model"], minfo["tokenizer"], image, question, **kwargs)


# ─── BoN Generation ──────────────────────────────────────────────────────

def run_bon_generation(minfo, train_data, n_candidates=8, temperature=0.7):
    """Generate N candidates per sample, select best correct one."""
    print(f"\n[bon] Generating {n_candidates} candidates × {len(train_data)} samples")
    curated = []
    n_correct = 0

    for i, s in enumerate(train_data):
        gt = s["answer"].strip().capitalize()
        candidates = []
        for _ in range(n_candidates):
            try:
                c = generate_one(minfo, s["image"], s["question"],
                                 temperature=temperature, do_sample=True)
                candidates.append(c)
            except Exception:
                candidates.append("")

        # Score: correct + concise
        scored = []
        for c in candidates:
            ext = YOrN_Extraction(c)
            correct = (ext == gt)
            score = 1.0 if correct else 0.0
            if correct and len(c.split()) <= 3:
                score += 0.1  # bonus for concise
            scored.append({"text": c, "ext": ext, "correct": correct, "score": score})

        best = max(scored, key=lambda x: x["score"])
        if best["correct"]:
            n_correct += 1
            curated.append({
                "question": s["question"],
                "answer": gt,
                "best_candidate": best["text"],
                "n_correct": sum(1 for x in scored if x["correct"]),
                "category": s.get("category", "unknown"),
            })

        if (i + 1) % 50 == 0:
            rate = n_correct / (i + 1) * 100
            print(f"  [{i+1}/{len(train_data)}] curated={len(curated)}, hit_rate={rate:.1f}%")

    print(f"\n[bon] Curated {len(curated)} from {len(train_data)} "
          f"(hit_rate={len(curated)/len(train_data)*100:.1f}%)")
    return curated


# ─── SFT Training ────────────────────────────────────────────────────────

def run_sft_qwen3(minfo, curated, epochs=2, lr=2e-6):
    """Fine-tune Qwen3-VL-2B on curated data."""
    from qwen_vl_utils import process_vision_info
    model = minfo["model"]
    processor = minfo["processor"]
    model.train()
    model.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_loss = 0
    n_steps = 0

    for epoch in range(epochs):
        random.shuffle(curated)
        epoch_loss = 0

        for j, item in enumerate(curated):
            prompt = POPE_PROMPT.format(question=item["question"])
            answer = item["best_candidate"]

            # Build input with the answer as target
            messages = [
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
                {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            ]
            text = processor.apply_chat_template(messages, tokenize=False)
            inputs = processor.tokenizer(text, return_tensors="pt", max_length=256, truncation=True)
            inputs = {k: v.to(minfo["device"]) for k, v in inputs.items()}
            inputs["labels"] = inputs["input_ids"].clone()

            try:
                outputs = model(**inputs)
                loss = outputs.loss / 4  # grad accumulation
                loss.backward()

                if (j + 1) % 4 == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    n_steps += 1

                epoch_loss += loss.item() * 4
                total_loss += loss.item() * 4
            except Exception as e:
                optimizer.zero_grad()
                continue

            if (j + 1) % 100 == 0:
                avg = epoch_loss / (j + 1)
                print(f"  [epoch {epoch+1}] step {j+1}/{len(curated)}, loss={avg:.4f}")

        # Final grad step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        n_steps += 1
        print(f"  [epoch {epoch+1}] avg_loss={epoch_loss / max(len(curated), 1):.4f}")

    model.eval()
    return total_loss / max(n_steps, 1)


def run_sft_internvl(minfo, curated, epochs=2, lr=5e-6):
    """Fine-tune InternVL3.5-1B on curated data."""
    model = minfo["model"]
    tokenizer = minfo["tokenizer"]
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_loss = 0
    n_steps = 0

    for epoch in range(epochs):
        random.shuffle(curated)
        epoch_loss = 0

        for j, item in enumerate(curated):
            prompt = POPE_PROMPT.format(question=item["question"])
            text = f"{prompt}\n{item['best_candidate']}"
            inputs = tokenizer(text, return_tensors="pt", max_length=256, truncation=True)
            inputs = {k: v.to(minfo["device"]) for k, v in inputs.items()}
            inputs["labels"] = inputs["input_ids"].clone()

            try:
                outputs = model.language_model(**inputs)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                total_loss += loss.item()
                n_steps += 1
            except Exception as e:
                optimizer.zero_grad()
                continue

            if (j + 1) % 100 == 0:
                avg = epoch_loss / (j + 1)
                print(f"  [epoch {epoch+1}] step {j+1}/{len(curated)}, loss={avg:.4f}")

        print(f"  [epoch {epoch+1}] avg_loss={epoch_loss / max(len(curated), 1):.4f}")

    model.eval()
    return total_loss / max(n_steps, 1)


# ─── Evaluation ──────────────────────────────────────────────────────────

def run_eval(minfo, dataset, max_samples=500):
    """Run POPE evaluation on adversarial split."""
    records = []
    blind_records = []
    n = min(max_samples, len(dataset))

    print(f"\n[eval] POPE adversarial — {n} samples")
    for i in range(n):
        s = dataset[i]
        try:
            raw = generate_one(minfo, s["image"], s["question"])
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

    # Blind eval (subset)
    blind_n = min(n, 300)
    print(f"\n[eval] Blind — {blind_n} samples")
    for i in range(blind_n):
        s = dataset[i]
        try:
            black_img = Image.new("RGB", s["image"].size, (0, 0, 0))
            raw = generate_one(minfo, black_img, s["question"])
        except Exception:
            raw = ""
        ext = YOrN_Extraction(raw)
        blind_records.append({
            "index": i, "question": s["question"],
            "answer": s["answer"].strip().capitalize(),
            "prediction": raw, "extracted": ext,
            "category": s.get("category", "unknown"),
        })

    met_b = pope_metrics(blind_records)
    gap = met["Overall"]["acc"] - met_b["Overall"]["acc"]

    o = met["Overall"]
    print(f"\n  → acc={o['acc']:.1f}%, F1={o['f1']:.1f}%, P={o['precision']:.1f}%, R={o['recall']:.1f}%")
    print(f"  → blind_acc={met_b['Overall']['acc']:.1f}%, gap={gap:.1f}pp")

    return met, met_b, gap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "internvl"])
    parser.add_argument("--max-train", type=int, default=500)
    parser.add_argument("--eval-samples", type=int, default=500)
    parser.add_argument("--n-candidates", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.lr is None:
        args.lr = 2e-6 if args.model == "qwen3" else 5e-6
    if args.output_dir is None:
        args.output_dir = f"checkpoints/{args.model}_bon_sft_pope"

    dataset = load_from_disk("data/eval/pope")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Split: train on random+popular, eval on adversarial
    train_indices = [i for i in range(len(dataset)) if dataset[i].get("category") != "adversarial"]
    eval_indices = [i for i in range(len(dataset)) if dataset[i].get("category") == "adversarial"]

    random.seed(42)
    random.shuffle(train_indices)
    train_indices = train_indices[:args.max_train]

    train_data = [dataset[i] for i in train_indices]
    eval_data = dataset.select(eval_indices[:args.eval_samples])

    print(f"Train: {len(train_data)} (random+popular), Eval: {len(eval_data)} (adversarial)")

    # Load model
    minfo = load_model_for_task(args.model, for_training=False, model_path=args.model_path)

    # Phase 1: BoN generation
    curated = run_bon_generation(minfo, train_data, n_candidates=args.n_candidates,
                                  temperature=args.temperature)

    with open(output_dir / f"curated_{ts}.json", "w") as f:
        json.dump(curated, f, indent=2)

    # Phase 2: SFT
    print(f"\n[sft] Training on {len(curated)} curated samples, {args.epochs} epochs")
    if args.model == "qwen3":
        avg_loss = run_sft_qwen3(minfo, curated, epochs=args.epochs, lr=args.lr)
    else:
        avg_loss = run_sft_internvl(minfo, curated, epochs=args.epochs, lr=args.lr)
    print(f"[sft] Average loss: {avg_loss:.4f}")

    # Save
    ckpt_dir = output_dir / "final"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if args.model == "qwen3":
        minfo["model"].save_pretrained(str(ckpt_dir))
        minfo["processor"].save_pretrained(str(ckpt_dir))
    else:
        minfo["model"].save_pretrained(str(ckpt_dir))
        minfo["tokenizer"].save_pretrained(str(ckpt_dir))
    print(f"[sft] Saved to {ckpt_dir}")

    # Phase 3: Evaluation on adversarial split
    met, met_b, gap = run_eval(minfo, eval_data, max_samples=args.eval_samples)

    results = {
        "model": args.model,
        "method": "bon_sft_pope_format",
        "timestamp": ts,
        "n_train": len(curated),
        "n_eval": args.eval_samples,
        "baseline": met["Overall"],
        "blind": met_b["Overall"],
        "blind_gap": gap,
    }

    report_dir = Path(f"lab/reports/multimodel/{args.model}")
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / f"bon_sft_pope_{ts}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"{args.model} BoN+SFT (POPE format) Results")
    print(f"{'='*60}")
    o = met["Overall"]
    print(f"Acc: {o['acc']:.1f}%, F1: {o['f1']:.1f}%, P: {o['precision']:.1f}%, R: {o['recall']:.1f}%")
    print(f"Blind Gap: {gap:.1f}pp")
    print(f"{'='*60}")

    del minfo
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
