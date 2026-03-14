"""
OCRBench evaluation: Thinking vs Short answer mode comparison.
100 samples, substring matching scoring.
"""
import os, sys, json, re, time, gc
import torch
from pathlib import Path
from PIL import Image
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from qwen_vl_utils import process_vision_info

HF_ID = "Qwen/Qwen3-VL-2B-Thinking"
OUT_DIR = Path("lab/reports/thinking_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# OCRBench category mapping (from VLMEvalKit)
CATEGORY_MAP = {
    "Regular Text Recognition": "text_rec",
    "Irregular Text Recognition": "text_rec",
    "Artistic Text Recognition": "text_rec",
    "Handwriting Recognition": "text_rec",
    "Digit String Recognition": "text_rec",
    "Non-Semantic Text Recognition": "text_rec",
    "Scene Text-centric VQA": "scene_vqa",
    "Doc-oriented VQA": "doc_vqa",
    "Key Information Extraction": "kie",
    "Handwritten Mathematical Expression Recognition": "math_expr",
}

def split_thinking(text):
    m = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if m: return m.group(1).strip(), text[m.end():].strip()
    m = re.search(r'</think>', text)
    if m: return text[:m.start()].strip(), text[m.end():].strip()
    return "", text.strip()

def score_ocrbench(pred, gt, category=""):
    """Score OCRBench prediction using substring matching."""
    if not pred or not gt:
        return 0
    # Clean prediction
    _, answer = split_thinking(pred)
    if not answer:
        answer = pred
    answer = answer.strip()

    # For math expressions, remove whitespace
    if "Mathematical" in category:
        answer_clean = re.sub(r'\s+', '', answer)
        gt_clean = re.sub(r'\s+', '', gt)
        return 1 if gt_clean.lower() in answer_clean.lower() else 0

    # Standard: case-insensitive substring match
    return 1 if gt.lower() in answer.lower() else 0

def load_ocrbench(max_samples=100):
    """Load OCRBench from HuggingFace."""
    from datasets import load_dataset
    print("Loading OCRBench from HuggingFace...")
    ds = load_dataset("echo840/OCRBench", split="test")
    print(f"  Total samples: {len(ds)}")

    # Sample evenly across categories
    per_cat = defaultdict(list)
    for row in ds:
        cat = row.get("question_type", row.get("category", row.get("type", "unknown")))
        per_cat[cat].append(row)

    samples = []
    n_cats = len(per_cat)
    per_cat_limit = max(max_samples // n_cats, 5)

    for cat, rows in per_cat.items():
        selected = rows[:per_cat_limit]
        for row in selected:
            # answer field is a list like ['CENTRE'] — extract first element
            raw_ans = row["answer"]
            if isinstance(raw_ans, list):
                ans = raw_ans[0] if raw_ans else ""
            else:
                ans = str(raw_ans)
            samples.append({
                "image": row["image"],
                "question": row["question"],
                "answer": ans,
                "category": cat,
            })
        if len(samples) >= max_samples:
            break

    samples = samples[:max_samples]
    print(f"  Selected {len(samples)} samples across {n_cats} categories")
    return samples

def load_model():
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    print(f"Loading {HF_ID}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        HF_ID, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()
    processor = AutoProcessor.from_pretrained(HF_ID, trust_remote_code=True)
    return model, processor

def prepare_inputs(processor, image, question, device, enable_thinking=True):
    # For short mode, add "Answer briefly." to reduce reasoning output
    q = question if enable_thinking else question + " Answer briefly."
    content = [{"type": "image", "image": image}, {"type": "text", "text": q}]
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=enable_thinking)
    imgs, _, _ = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(text=[text], images=imgs, return_tensors="pt", padding=True)
    return {k: v.to(device) for k, v in inputs.items()}

def run_comparison(model, processor, samples, device):
    """Run both thinking and non-thinking on OCRBench samples."""
    results = []
    t0 = time.time()

    for i, s in enumerate(samples):
        try:
            q = s["question"]
            gt = s["answer"]
            cat = s["category"]

            # --- Thinking mode ---
            inputs_t = prepare_inputs(processor, s["image"], q, device, enable_thinking=True)
            with torch.no_grad():
                out_t = model.generate(**inputs_t, max_new_tokens=1024, do_sample=False)
            raw_t = processor.tokenizer.decode(out_t[0][inputs_t["input_ids"].shape[1]:], skip_special_tokens=False)
            for tok in ["<|im_end|>","<|endoftext|>","<|im_start|>"]: raw_t = raw_t.replace(tok, "")
            score_t = score_ocrbench(raw_t, gt, cat)
            thinking_text, answer_t = split_thinking(raw_t)
            thinking_len = len(processor.tokenizer.encode(thinking_text)) if thinking_text else 0

            # --- Short (non-thinking) mode ---
            inputs_s = prepare_inputs(processor, s["image"], q, device, enable_thinking=False)
            with torch.no_grad():
                out_s = model.generate(**inputs_s, max_new_tokens=256, do_sample=False)
            raw_s = processor.tokenizer.decode(out_s[0][inputs_s["input_ids"].shape[1]:], skip_special_tokens=False)
            for tok in ["<|im_end|>","<|endoftext|>","<|im_start|>"]: raw_s = raw_s.replace(tok, "")
            score_s = score_ocrbench(raw_s, gt, cat)

            result = {
                "idx": i,
                "question": q[:200],
                "category": cat,
                "gt": gt[:200],
                "answer_thinking": answer_t[:200] if answer_t else raw_t[:200],
                "answer_short": raw_s.strip()[:200],
                "score_thinking": score_t,
                "score_short": score_s,
                "thinking_len_tokens": thinking_len,
                "thinking_text": thinking_text[:300] if thinking_text else "",
            }
            results.append(result)

            # Save images for disagreement cases
            if score_t != score_s:
                img_dir = OUT_DIR / "disagreement_images"
                img_dir.mkdir(parents=True, exist_ok=True)
                try:
                    s["image"].save(img_dir / f"sample_{i}.png")
                except Exception:
                    pass

            if (i+1) % 10 == 0:
                st = sum(r["score_thinking"] for r in results)
                ss = sum(r["score_short"] for r in results)
                n = len(results)
                print(f"  [{i+1}/{len(samples)}] thinking={st/n*100:.1f}% short={ss/n*100:.1f}% ({time.time()-t0:.0f}s)", flush=True)

        except Exception as e:
            print(f"  ERROR at {i}: {e}", flush=True)

    return results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=300)
    args = parser.parse_args()

    samples = load_ocrbench(args.max_samples)
    model, processor = load_model()
    device = next(model.parameters()).device

    print("\n=== OCRBench: Thinking vs Short ===")
    results = run_comparison(model, processor, samples, device)

    # Save
    with open(OUT_DIR / "ocrbench_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary
    total = len(results)
    st = sum(r["score_thinking"] for r in results)
    ss = sum(r["score_short"] for r in results)

    disagree = [r for r in results if r["score_thinking"] != r["score_short"]]
    short_wins = [d for d in disagree if d["score_short"] > d["score_thinking"]]
    think_wins = [d for d in disagree if d["score_thinking"] > d["score_short"]]

    print(f"\n{'='*60}")
    print(f"OCRBench RESULTS ({total} samples)")
    print(f"{'='*60}")
    print(f"Thinking: {st}/{total} = {st/total*100:.1f}%")
    print(f"Short:    {ss}/{total} = {ss/total*100:.1f}%")
    print(f"Delta:    {(st-ss)/total*100:+.1f}pp")
    print(f"Disagreements: {len(disagree)}")
    print(f"  Short wins: {len(short_wins)}")
    print(f"  Think wins: {len(think_wins)}")

    # Per-category
    cats = defaultdict(lambda: {"st":0, "ss":0, "n":0})
    for r in results:
        mapped = CATEGORY_MAP.get(r["category"], r["category"])
        c = cats[mapped]
        c["n"] += 1
        c["st"] += r["score_thinking"]
        c["ss"] += r["score_short"]

    print(f"\nPer-category:")
    for cat in sorted(cats.keys()):
        c = cats[cat]
        if c["n"] > 0:
            print(f"  {cat:15s}: think={c['st']/c['n']*100:5.1f}% short={c['ss']/c['n']*100:5.1f}% delta={(c['st']-c['ss'])/c['n']*100:+5.1f}pp (n={c['n']})")

    # Show disagreement examples
    if short_wins:
        print(f"\n=== SHORT WINS (OCRBench) ===")
        for d in short_wins[:5]:
            print(f"  Q: {d['question'][:60]}")
            print(f"    GT: {d['gt'][:60]}")
            print(f"    Short: {d['answer_short'][:60]}")
            print(f"    Think: {d['answer_thinking'][:60]} (len={d['thinking_len_tokens']})")
            print()

    if think_wins:
        print(f"\n=== THINK WINS (OCRBench) ===")
        for d in think_wins[:5]:
            print(f"  Q: {d['question'][:60]}")
            print(f"    GT: {d['gt'][:60]}")
            print(f"    Short: {d['answer_short'][:60]}")
            print(f"    Think: {d['answer_thinking'][:60]} (len={d['thinking_len_tokens']})")
            print()

    print(f"{'='*60}")
    print(f"Saved to {OUT_DIR / 'ocrbench_results.json'}")

if __name__ == "__main__":
    main()
