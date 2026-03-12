"""1K POPE eval: baseline + GRPO-LSR best"""
import os, sys, json, re, string, time, gc
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from qwen_vl_utils import process_vision_info

HF_ID = "Qwen/Qwen3-VL-2B-Thinking"
BEST_CKPT = "checkpoints/phase2_grpo_lsr/round4/best"
POPE_SPLITS = ["random", "popular", "adversarial"]

def split_thinking(text):
    m = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if m: return m.group(1).strip(), text[m.end():].strip()
    m = re.search(r'</think>', text)
    if m: return text[:m.start()].strip(), text[m.end():].strip()
    return "", text.strip()

def extract_yes_no(raw):
    _, answer = split_thinking(raw)
    if not answer: answer = raw
    text = answer.strip().lower()
    for p in string.punctuation: text = text.replace(p, " ")
    words = text.split()
    for w in words[:5]:
        if w in ("yes", "true"): return "yes"
        if w in ("no", "false"): return "no"
    if "yes" in words: return "yes"
    if "no" in words: return "no"
    return None

def load_pope(max_per_split=334):
    from datasets import load_dataset
    ds = load_dataset("lmms-lab/POPE", split="test", streaming=True)
    per_split = defaultdict(list)
    for row in ds:
        cat = row.get("category", "unknown")
        if cat not in POPE_SPLITS: continue
        if len(per_split[cat]) >= max_per_split:
            if all(len(per_split[s]) >= max_per_split for s in POPE_SPLITS): break
            continue
        per_split[cat].append({
            "image": row["image"], "question": row["question"],
            "answer": row["answer"].strip().lower(), "category": cat,
        })
    samples = []
    for s in POPE_SPLITS: samples.extend(per_split[s])
    print(f"Loaded {len(samples)} POPE samples")
    return samples

def load_model(path=None):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    p = path or HF_ID
    print(f"Loading {p}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        p, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()
    processor = AutoProcessor.from_pretrained(HF_ID, trust_remote_code=True)
    return model, processor

def prepare_inputs(processor, image, question, device):
    content = [{"type": "image", "image": image}, {"type": "text", "text": question}]
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=True)
    imgs, _, _ = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(text=[text], images=imgs, return_tensors="pt", padding=True)
    return {k: v.to(device) for k, v in inputs.items()}

def eval_pope(model, processor, samples, device, label=""):
    per_split = defaultdict(lambda: {"tp":0,"fp":0,"tn":0,"fn":0,"total":0})
    error_count = 0
    t0 = time.time()
    for i, s in enumerate(samples):
        try:
            q = s["question"] + " Please answer yes or no."
            inputs = prepare_inputs(processor, s["image"], q, device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            raw = processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
            for tok in ["<|im_end|>","<|endoftext|>","<|im_start|>"]: raw = raw.replace(tok, "")
            pred = extract_yes_no(raw)
            gt = s["answer"]; cat = s.get("category","unknown")
            d = per_split[cat]; d["total"] += 1
            if pred == "yes" and gt == "yes": d["tp"] += 1
            elif pred == "yes" and gt == "no": d["fp"] += 1
            elif pred == "no" and gt == "no": d["tn"] += 1
            elif pred == "no" and gt == "yes": d["fn"] += 1
            if (i+1) % 100 == 0:
                total_c = sum(d["tp"]+d["tn"] for d in per_split.values())
                total_n = sum(d["total"] for d in per_split.values())
                print(f"  [{label}] {i+1}/{len(samples)} acc={total_c/total_n:.1%} ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            error_count += 1
            if error_count <= 3:
                print(f"  [{label}] ERROR at sample {i} (#{error_count}): {type(e).__name__}: {e}", flush=True)
            if error_count == 11:
                print(f"  [{label}] WARNING: {error_count} errors so far — suppressing further error logs", flush=True)
            per_split[s.get("category","unknown")]["total"] += 1
    if error_count > 0:
        print(f"  [{label}] TOTAL ERRORS: {error_count}/{len(samples)} ({error_count/len(samples)*100:.1f}%)", flush=True)

    results = {}
    for cat in POPE_SPLITS + ["overall"]:
        if cat == "overall":
            tp = sum(d["tp"] for d in per_split.values())
            fp = sum(d["fp"] for d in per_split.values())
            tn = sum(d["tn"] for d in per_split.values())
            fn = sum(d["fn"] for d in per_split.values())
            n = sum(d["total"] for d in per_split.values())
        else:
            d = per_split[cat]; tp,fp,tn,fn,n = d["tp"],d["fp"],d["tn"],d["fn"],d["total"]
        acc = (tp+tn)/n if n > 0 else 0
        p = tp/(tp+fp) if (tp+fp) > 0 else 0
        r = tp/(tp+fn) if (tp+fn) > 0 else 0
        f1 = 2*p*r/(p+r) if (p+r) > 0 else 0
        results[cat] = {"acc":acc, "f1":f1, "precision":p, "recall":r, "total":n}
    return results

def eval_blind(model, processor, samples, device, n=200, label=""):
    real_c = blind_c = total = error_count = 0
    for i, s in enumerate(samples[:n]):
        try:
            q = s["question"] + " Please answer yes or no."
            gt = s["answer"]
            inputs = prepare_inputs(processor, s["image"], q, device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            raw = processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
            if extract_yes_no(raw) == gt: real_c += 1
            black = Image.new('RGB', s["image"].size, (0,0,0))
            inputs_b = prepare_inputs(processor, black, q, device)
            with torch.no_grad():
                out_b = model.generate(**inputs_b, max_new_tokens=512, do_sample=False)
            raw_b = processor.tokenizer.decode(out_b[0][inputs_b["input_ids"].shape[1]:], skip_special_tokens=False)
            if extract_yes_no(raw_b) == gt: blind_c += 1
            total += 1
            if (i+1) % 50 == 0:
                print(f"  [{label} blind] {i+1}/{n} real={real_c/total:.1%} blind={blind_c/total:.1%} gap={(real_c-blind_c)/total:.1%}", flush=True)
        except Exception as e:
            error_count += 1
            if error_count <= 3:
                print(f"  [{label} blind] ERROR at sample {i} (#{error_count}): {type(e).__name__}: {e}", flush=True)
            if error_count == 11:
                print(f"  [{label} blind] WARNING: {error_count} errors so far — suppressing further error logs", flush=True)
            total += 1
    if error_count > 0:
        print(f"  [{label} blind] TOTAL ERRORS: {error_count}/{n} ({error_count/n*100:.1f}%)", flush=True)
    ra = real_c/total if total > 0 else 0
    ba = blind_c/total if total > 0 else 0
    return {"real_acc": ra, "blind_acc": ba, "gap": ra-ba, "total": total}

def main():
    samples = load_pope(334)  # ~1000 total

    all_results = {}

    # Baseline
    print("\n=== BASELINE ===")
    model, processor = load_model()
    device = next(model.parameters()).device
    pope_b = eval_pope(model, processor, samples, device, "Baseline")
    blind_b = eval_blind(model, processor, samples, device, 200, "Baseline")
    all_results["Baseline"] = {"pope": pope_b, "blind": blind_b}
    print(f"\nBaseline: Acc={pope_b['overall']['acc']:.1%} F1={pope_b['overall']['f1']:.1%} Gap={blind_b['gap']:.1%}")
    del model; torch.cuda.empty_cache(); gc.collect()

    # GRPO-LSR
    print("\n=== GRPO-LSR R4-BEST ===")
    model, processor = load_model(BEST_CKPT)
    device = next(model.parameters()).device
    pope_g = eval_pope(model, processor, samples, device, "GRPO-LSR")
    blind_g = eval_blind(model, processor, samples, device, 200, "GRPO-LSR")
    all_results["GRPO-LSR"] = {"pope": pope_g, "blind": blind_g}
    print(f"\nGRPO-LSR: Acc={pope_g['overall']['acc']:.1%} F1={pope_g['overall']['f1']:.1%} Gap={blind_g['gap']:.1%}")
    del model; torch.cuda.empty_cache(); gc.collect()

    # Summary
    delta_acc = (pope_g['overall']['acc'] - pope_b['overall']['acc']) * 100
    delta_gap = (blind_g['gap'] - blind_b['gap']) * 100

    print("\n" + "="*70)
    print("FINAL RESULTS (1K POPE)")
    print("="*70)
    print(f"{'Cond':<15} {'Split':<12} {'Acc':>6} {'F1':>6} {'P':>6} {'R':>6}")
    print("-"*70)
    for cond in all_results:
        for split in POPE_SPLITS + ["overall"]:
            d = all_results[cond]["pope"].get(split, {})
            print(f"{cond:<15} {split:<12} {d['acc']*100:5.1f}% {d['f1']*100:5.1f}% {d['precision']*100:5.1f}% {d['recall']*100:5.1f}%")
    print("-"*70)
    print(f"Blind Gap: Baseline={blind_b['gap']*100:.1f}pp, GRPO-LSR={blind_g['gap']*100:.1f}pp (Δ={delta_gap:+.1f}pp)")
    print(f"POPE Delta: {delta_acc:+.1f}pp")
    print("="*70)

    # Save
    Path("lab/reports/full_pope_eval").mkdir(parents=True, exist_ok=True)
    serializable = {}
    for k, v in all_results.items():
        serializable[k] = {
            "pope": {sk: dict(sv) for sk, sv in v["pope"].items()},
            "blind": dict(v["blind"]),
        }
    with open("lab/reports/full_pope_eval/results_1k.json", "w") as f:
        json.dump(serializable, f, indent=2)
    print("Saved to lab/reports/full_pope_eval/results_1k.json")

if __name__ == "__main__":
    main()
