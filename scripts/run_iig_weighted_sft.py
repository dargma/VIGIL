"""
VIGIL P2-04: IIG-Weighted SFT Loss (Axis D)

Token-level IIG weighting: visually-grounded tokens (high IIG)
get amplified in the SFT loss. Structural tokens (low IIG) get
down-weighted. The model learns that visually-grounded tokens matter more.

loss_i = CE_i * (1 + gamma * max(IIG_i, 0))

Usage:
    python scripts/run_iig_weighted_sft.py --gamma 1.0  # default
    python scripts/run_iig_weighted_sft.py --gamma 0.5  # mild
    python scripts/run_iig_weighted_sft.py --gamma 2.0  # strong
"""

import os, sys, gc, json, re, argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from datetime import datetime
from datasets import load_from_disk

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.calibrator import CalibrationResult


# ─── VLMEvalKit standard functions ─────────────────────────────────────────

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


# ─── Model + generation ────────────────────────────────────────────────────

def load_model(model_path=None):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    hf_id = model_path or "Qwen/Qwen3-VL-2B-Instruct"
    print(f"[model] Loading: {hf_id}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
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
                             top_p=0.8, top_k=20)
    out = gen[0][inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(out, skip_special_tokens=True).strip()


def eval_pope(model, processor, dataset, label, blind=False, max_n=None):
    n = min(len(dataset), max_n or len(dataset))
    records = []
    model.eval()
    print(f"\n[pope] {label} ({'blind' if blind else 'real'}) — {n} samples")
    for i in range(n):
        s = dataset[i]
        try:
            raw = generate_one(model, processor, s["image"], s["question"], blind=blind)
        except Exception as e:
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


# ─── Per-token IIG ──────────────────────────────────────────────────────────

BLACK_IMAGE = Image.new("RGB", (448, 448), (0, 0, 0))

@torch.no_grad()
def compute_per_token_iig(model, processor, image, question, answer_text):
    """Compute per-token IIG values for a given answer.

    Returns: tensor of shape (T,) with per-token IIG values.
    """
    from qwen_vl_utils import process_vision_info

    prompt = POPE_PROMPT.format(question=question)

    # Encode answer tokens
    answer_ids = processor.tokenizer(
        answer_text, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(model.device)
    T = answer_ids.shape[1]
    if T == 0:
        return torch.zeros(0, device=model.device)

    # Build inputs for real and black image
    def build_inputs(img):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, _ = process_vision_info(messages, return_video_kwargs=True)
        inputs = processor(text=[text], images=images, videos=videos,
                           return_tensors="pt", padding=True)
        return {k: v.to(model.device) for k, v in inputs.items()}

    real_inputs = build_inputs(image)
    black_inputs = build_inputs(BLACK_IMAGE)

    prompt_len_real = real_inputs["input_ids"].shape[1]
    prompt_len_black = black_inputs["input_ids"].shape[1]

    # Concatenate prompt + answer tokens
    ids_real = torch.cat([real_inputs["input_ids"], answer_ids], dim=1)
    ids_black = torch.cat([black_inputs["input_ids"], answer_ids], dim=1)

    # Extend attention masks
    def extend_inputs(inputs, extra_len):
        result = {}
        for k, v in inputs.items():
            if k == "input_ids":
                continue
            if k == "attention_mask" and v is not None:
                ext = torch.ones(v.shape[0], extra_len, dtype=v.dtype, device=v.device)
                result[k] = torch.cat([v, ext], dim=1)
            else:
                result[k] = v
        return result

    kw_real = extend_inputs(real_inputs, T)
    kw_black = extend_inputs(black_inputs, T)

    # Forward passes
    logits_real = model(input_ids=ids_real, **kw_real).logits
    logits_black = model(input_ids=ids_black, **kw_black).logits

    # Per-token log probs
    lp_real = F.log_softmax(logits_real[:, prompt_len_real-1:prompt_len_real-1+T, :], dim=-1)
    tok_lp_real = lp_real.gather(-1, answer_ids[:, :T].unsqueeze(-1)).squeeze(-1)

    lp_black = F.log_softmax(logits_black[:, prompt_len_black-1:prompt_len_black-1+T, :], dim=-1)
    tok_lp_black = lp_black.gather(-1, answer_ids[:, :T].unsqueeze(-1)).squeeze(-1)

    # Per-token IIG
    iig_per_token = (tok_lp_real - tok_lp_black).squeeze(0)  # (T,)
    return iig_per_token


# ─── IIG-Weighted SFT ──────────────────────────────────────────────────────

def iig_weighted_sft_step(model, processor, image, question, answer, gamma, device):
    """One SFT step with per-token IIG weighting.

    loss_i = CE_i * (1 + gamma * max(IIG_i, 0))
    """
    from qwen_vl_utils import process_vision_info

    prompt = POPE_PROMPT.format(question=question)
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": answer},
        ]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False)
    images, videos, _ = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(text=[text], images=images, videos=videos,
                       return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_ids = inputs["input_ids"]
    labels = input_ids.clone()

    # Mask prompt tokens
    prompt_only = processor.apply_chat_template(
        messages[:1], tokenize=False, add_generation_prompt=True
    )
    prompt_len = len(processor.tokenizer.encode(prompt_only))
    labels[0, :prompt_len] = -100

    # Forward pass to get per-token loss
    outputs = model(**inputs, labels=labels)

    # Get per-token CE loss (need to recompute manually)
    logits = outputs.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Per-token cross entropy
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    per_token_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    ).view(shift_labels.shape)

    # Mask for non-ignored tokens
    mask = (shift_labels != -100).float()
    n_tokens = mask.sum()

    if n_tokens == 0 or gamma == 0.0:
        # Standard SFT
        return (per_token_loss * mask).sum() / max(n_tokens, 1)

    # Compute per-token IIG weights
    # Extract answer tokens (non-masked positions)
    answer_start = prompt_len - 1  # shifted by 1
    answer_len = int(mask[0, answer_start:].sum().item())

    if answer_len > 0:
        try:
            iig_per_token = compute_per_token_iig(
                model, processor, image, question, answer
            )
            # Align IIG with loss positions
            weights = torch.ones_like(mask)
            iig_len = min(len(iig_per_token), answer_len)
            iig_clipped = torch.clamp(iig_per_token[:iig_len], min=0.0)
            weights[0, answer_start:answer_start+iig_len] = 1.0 + gamma * iig_clipped
        except Exception:
            weights = torch.ones_like(mask)
    else:
        weights = torch.ones_like(mask)

    # Weighted loss
    weighted_loss = (per_token_loss * mask * weights).sum() / max(n_tokens, 1)
    return weighted_loss


def main():
    parser = argparse.ArgumentParser(description="VIGIL IIG-Weighted SFT")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="IIG weighting strength (0=standard SFT, 1=moderate, 2=strong)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--eval-samples", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="checkpoints/phase2/iig_weighted_sft")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(f"P2-04: IIG-WEIGHTED SFT (gamma={args.gamma})")
    print("=" * 60)

    eval_data = load_from_disk("data/eval/pope")
    train_data = load_from_disk("data/training/vqav2_train")

    # Use VQAv2 training data directly — each sample has image + question + answer
    # Filter to short answers only (more suited for yes/no-adjacent training)
    bon_data = []
    for i in range(min(len(train_data), 1000)):
        s = train_data[i]
        answer = s.get("multiple_choice_answer", "").strip()
        if answer and len(answer.split()) <= 5:  # short answers only
            bon_data.append({
                "question": s["question"],
                "answer": answer,
                "image_idx": i,
            })
    print(f"[data] {len(bon_data)} short-answer VQAv2 samples for IIG-weighted SFT")

    # Start from BoN+SFT checkpoint
    bon_path = "checkpoints/block2_bon/final"
    if Path(bon_path).exists():
        print(f"[model] Starting from BoN+SFT: {bon_path}")
        model, processor = load_model(bon_path)
    else:
        model, processor = load_model()
    device = next(model.parameters()).device

    # Baseline eval
    recs_base, met_base = eval_pope(model, processor, eval_data,
                                     "baseline", max_n=args.eval_samples)
    recs_blind, met_blind = eval_pope(model, processor, eval_data,
                                       "baseline_blind", blind=True, max_n=200)
    base_gap = met_base["Overall"]["acc"] - met_blind["Overall"]["acc"]
    print(f"\n[baseline] acc={met_base['Overall']['acc']:.1f}%, gap={base_gap:.1f}pp")

    # SFT with IIG weighting
    model.train()
    model.gradient_checkpointing_enable()
    for p in model.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    out_dir = Path(args.output_dir) / f"gamma_{args.gamma}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        total_loss = 0
        n_steps = 0

        for ci, cdata in enumerate(bon_data):
            try:
                # Get image from training data
                img_idx = cdata.get("image_idx", ci % len(train_data))
                sample = train_data[img_idx]
                image = sample["image"]
                question = cdata["question"]
                answer = cdata["answer"]

                loss = iig_weighted_sft_step(
                    model, processor, image, question, answer,
                    gamma=args.gamma, device=device
                )
                loss = loss / args.grad_accum
                loss.backward()
                total_loss += loss.item() * args.grad_accum

                if (ci + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    n_steps += 1

            except Exception as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                continue

        avg_loss = total_loss / max(len(bon_data), 1)
        print(f"  Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f} ({n_steps} steps)")

    # Eval
    model.eval()
    recs, met = eval_pope(model, processor, eval_data,
                          f"iig_sft_g{args.gamma}", max_n=args.eval_samples)
    recs_b, met_b = eval_pope(model, processor, eval_data,
                               f"iig_sft_g{args.gamma}_blind", blind=True, max_n=200)
    gap = met["Overall"]["acc"] - met_b["Overall"]["acc"]

    print(f"\n[RESULT] gamma={args.gamma}: acc={met['Overall']['acc']:.1f}%, "
          f"gap={gap:.1f}pp, F1={met['Overall']['f1']:.1f}%")
    print(f"  vs baseline: {met['Overall']['acc'] - met_base['Overall']['acc']:+.1f}pp acc, "
          f"{gap - base_gap:+.1f}pp gap")

    # Save
    model.save_pretrained(str(out_dir))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(out_dir / f"results_{ts}.json", "w") as f:
        json.dump({
            "gamma": args.gamma,
            "baseline": met_base, "baseline_gap": base_gap,
            "result": met, "result_gap": gap,
            "delta_acc": met["Overall"]["acc"] - met_base["Overall"]["acc"],
            "delta_gap": gap - base_gap,
        }, f, indent=2)

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
