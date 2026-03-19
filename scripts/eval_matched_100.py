"""Matched 100-sample POPE eval: Baseline vs Exp10, identical settings.

Evaluates both Qwen3-VL-2B-Thinking (HF) and Exp10 checkpoint on the
exact same 100 POPE samples with identical generation and parsing.
"""
import torch, json, time, gc, os, sys
from pathlib import Path
from datasets import load_dataset
from PIL import Image

# ── Config ──────────────────────────────────────────────────────────
N_SAMPLES = 100  # per-split balanced: 34+33+33
MAX_NEW_TOKENS = 512
DO_SAMPLE = False
ENABLE_THINKING = True
EXP10_PATH = "checkpoints/exp10_sharp_soft/scaled_final/best"
OUT_DIR = Path("lab/reports/matched_eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

POPE_SPLITS = ["random", "popular", "adversarial"]


# ── YOrN Extraction (identical for both) ────────────────────────────
def extract_yes_no(raw: str) -> str:
    s = raw.lower().strip()
    # Strip thinking
    if '</think>' in s:
        s = s.split('</think>')[-1].strip()
    # Strip special tokens
    for tok in ['<|im_end|>', '<|endoftext|>', '<|im_start|>']:
        s = s.replace(tok, '')
    s = s.strip().rstrip('.!,')
    if s in ('yes', 'no'):
        return s.capitalize()
    words = s.split()
    for w in words[:5]:
        w = w.strip('.,!?')
        if w in ('yes', 'no'):
            return w.capitalize()
    return 'Unknown'


# ── Load POPE samples ──────────────────────────────────────────────
def load_pope_100():
    ds = load_dataset("lmms-lab/POPE", split="test")
    per_split = {s: [] for s in POPE_SPLITS}
    per_n = {POPE_SPLITS[0]: 34, POPE_SPLITS[1]: 33, POPE_SPLITS[2]: 33}
    for row in ds:
        cat = row.get("category", "unknown")
        if cat not in POPE_SPLITS:
            continue
        if len(per_split[cat]) >= per_n[cat]:
            if all(len(per_split[s]) >= per_n[s] for s in POPE_SPLITS):
                break
            continue
        per_split[cat].append(row)
    samples = []
    for s in POPE_SPLITS:
        samples.extend(per_split[s])
    print(f"Loaded {len(samples)} POPE samples "
          f"({', '.join(f'{s}={len(per_split[s])}' for s in POPE_SPLITS)})")
    return samples


# ── Evaluate one model ─────────────────────────────────────────────
def evaluate(model, processor, samples, label):
    from qwen_vl_utils import process_vision_info

    model.eval()
    records = []
    correct = total = unknown = 0
    t0 = time.time()

    for i, s in enumerate(samples):
        image = s["image"]
        if not isinstance(image, Image.Image):
            records.append({"index": i, "question": s["question"],
                           "answer": s["answer"], "category": s["category"],
                           "raw": "", "extracted": "Unknown"})
            total += 1
            unknown += 1
            continue

        q = s["question"] + " Please answer yes or no."
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": q}
        ]}]
        text = processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=ENABLE_THINKING)
        image_inputs, _ = process_vision_info(msgs)
        inputs = processor(
            text=[text], images=image_inputs,
            return_tensors="pt", padding=True
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
            )
        raw = processor.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=False)
        pred = extract_yes_no(raw)
        gt = s["answer"].strip().capitalize()
        if gt not in ("Yes", "No"):
            gt = "Yes" if "yes" in s["answer"].lower() else "No"

        is_correct = pred == gt
        if is_correct:
            correct += 1
        if pred == "Unknown":
            unknown += 1
        total += 1

        # Extract thinking length
        think_text = ""
        if '<think>' in raw and '</think>' in raw:
            think_text = raw.split('<think>')[1].split('</think>')[0]

        records.append({
            "index": i,
            "question": s["question"],
            "answer": gt,
            "category": s["category"],
            "raw": raw[:500],  # truncate for storage
            "thinking_words": len(think_text.split()) if think_text else 0,
            "extracted": pred,
            "correct": is_correct,
        })

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"  [{label}] {i+1}/{len(samples)} | "
                  f"acc={correct/total*100:.1f}% | unk={unknown} | "
                  f"{(i+1)/elapsed:.2f} sps")

    acc = correct / total * 100
    print(f"\n  [{label}] FINAL: {correct}/{total} = {acc:.1f}% "
          f"(unknown={unknown})")
    return {"label": label, "n": total, "correct": correct,
            "acc": acc, "unknown": unknown, "records": records}


# ── Main ───────────────────────────────────────────────────────────
def main():
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    samples = load_pope_100()

    results = {}

    # 1. Baseline (HF Thinking model)
    print("\n" + "="*60)
    print("  Evaluating BASELINE: Qwen3-VL-2B-Thinking (HF)")
    print("="*60)
    hf_id = "Qwen/Qwen3-VL-2B-Thinking"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        hf_id, torch_dtype=torch.float16, device_map="cuda")
    processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
    results["baseline"] = evaluate(model, processor, samples, "baseline")

    # Free memory
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Exp10 checkpoint
    print("\n" + "="*60)
    print(f"  Evaluating EXP10: {EXP10_PATH}")
    print("="*60)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        EXP10_PATH, torch_dtype=torch.float16, device_map="cuda")
    # Use same processor as baseline (tokenizer is the same)
    results["exp10"] = evaluate(model, processor, samples, "exp10")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Compare
    print("\n" + "="*60)
    print("  COMPARISON")
    print("="*60)
    bl = results["baseline"]
    ex = results["exp10"]
    print(f"  Baseline: {bl['acc']:.1f}% ({bl['correct']}/{bl['n']})")
    print(f"  Exp10:    {ex['acc']:.1f}% ({ex['correct']}/{ex['n']})")
    print(f"  Delta:    {ex['acc'] - bl['acc']:+.1f}pp")

    # Per-split
    for split in POPE_SPLITS:
        bl_split = [r for r in bl["records"] if r["category"] == split]
        ex_split = [r for r in ex["records"] if r["category"] == split]
        bl_acc = sum(r["correct"] for r in bl_split) / len(bl_split) * 100 if bl_split else 0
        ex_acc = sum(r["correct"] for r in ex_split) / len(ex_split) * 100 if ex_split else 0
        print(f"  {split:12s}: BL={bl_acc:.1f}% → Exp10={ex_acc:.1f}% ({ex_acc-bl_acc:+.1f}pp)")

    # Per-sample cross-tab
    both_correct = sum(1 for b, e in zip(bl["records"], ex["records"]) if b["correct"] and e["correct"])
    bl_only = sum(1 for b, e in zip(bl["records"], ex["records"]) if b["correct"] and not e["correct"])
    ex_only = sum(1 for b, e in zip(bl["records"], ex["records"]) if not b["correct"] and e["correct"])
    both_wrong = sum(1 for b, e in zip(bl["records"], ex["records"]) if not b["correct"] and not e["correct"])
    n = len(bl["records"])
    print(f"\n  Cross-tab ({n} samples):")
    print(f"    Both correct:  {both_correct} ({both_correct/n*100:.1f}%)")
    print(f"    BL only (reg): {bl_only} ({bl_only/n*100:.1f}%)")
    print(f"    Exp10 only:    {ex_only} ({ex_only/n*100:.1f}%)")
    print(f"    Both wrong:    {both_wrong} ({both_wrong/n*100:.1f}%)")

    # Save
    out_path = OUT_DIR / "matched_100_results.json"
    save_data = {
        "config": {
            "n_samples": N_SAMPLES,
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": DO_SAMPLE,
            "enable_thinking": ENABLE_THINKING,
            "exp10_path": EXP10_PATH,
        },
        "baseline": {k: v for k, v in bl.items() if k != "records"},
        "exp10": {k: v for k, v in ex.items() if k != "records"},
        "baseline_records": bl["records"],
        "exp10_records": ex["records"],
    }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
