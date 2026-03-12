"""
Full 9K POPE evaluation — publication-quality numbers.
Evaluates baseline + best GRPO-LSR checkpoint on all 9000 POPE samples.
Reports per-split (random/popular/adversarial) accuracy, F1, precision, recall.
Also runs blind test (black image) for Gap metric.

Usage:
    python scripts/eval_full_pope.py 2>&1 | tee logs/eval_full_pope.log
"""

import os, sys, json, re, string, time, gc
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from qwen_vl_utils import process_vision_info

HF_ID = "Qwen/Qwen3-VL-2B-Thinking"
POPE_SPLITS = ["random", "popular", "adversarial"]
BEST_CKPT = "checkpoints/phase2_grpo_lsr/round4/best"
PROJECT_ROOT = Path(__file__).parent.parent


# ═══════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════

def split_thinking(text):
    m = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    m = re.search(r'</think>', text)
    if m:
        return text[:m.start()].strip(), text[m.end():].strip()
    return "", text.strip()


def extract_yes_no(raw):
    _, answer = split_thinking(raw)
    if not answer:
        answer = raw
    text = answer.strip().lower()
    for p in string.punctuation:
        text = text.replace(p, " ")
    words = text.split()
    for w in words[:5]:
        if w in ("yes", "true"): return "yes"
        if w in ("no", "false"): return "no"
    if "yes" in words: return "yes"
    if "no" in words: return "no"
    return None


# ═══════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════

def load_full_pope():
    """Load all 9000 POPE samples (3000 per split)."""
    from datasets import load_dataset
    ds = load_dataset("lmms-lab/POPE", split="test", streaming=True)
    per_split = defaultdict(list)

    for row in ds:
        cat = row.get("category", "unknown")
        if cat not in POPE_SPLITS:
            continue
        per_split[cat].append({
            "image": row["image"],
            "question": row["question"],
            "answer": row["answer"].strip().lower(),
            "category": cat,
        })

    samples = []
    for s in POPE_SPLITS:
        samples.extend(per_split[s])
        print(f"  {s}: {len(per_split[s])} samples")
    print(f"  Total: {len(samples)} samples")
    return samples


# ═══════════════════════════════════════════════════
#  Model Loading
# ═══════════════════════════════════════════════════

def load_model(model_path=None):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    path = model_path or HF_ID
    print(f"[model] Loading {path}...")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        HF_ID, trust_remote_code=True)

    return model, processor


def prepare_inputs(processor, image, question, device):
    content = [{"type": "image", "image": image},
               {"type": "text", "text": question}]
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True)
    imgs, _, _ = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(text=[text], images=imgs, return_tensors="pt",
                       padding=True)
    return {k: v.to(device) for k, v in inputs.items()}


# ═══════════════════════════════════════════════════
#  Evaluation
# ═══════════════════════════════════════════════════

def evaluate_pope(model, processor, samples, device, label=""):
    """Full POPE evaluation with per-split metrics."""
    per_split = defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "total": 0})
    think_lengths = []
    error_count = 0
    t0 = time.time()

    for i, s in enumerate(samples):
        try:
            q = s["question"] + " Please answer yes or no."
            inputs = prepare_inputs(processor, s["image"], q, device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            gen = out[0][inputs["input_ids"].shape[1]:]
            raw = processor.tokenizer.decode(gen, skip_special_tokens=False)
            for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
                raw = raw.replace(tok, "")

            thinking, _ = split_thinking(raw)
            think_lengths.append(len(thinking.split()) if thinking else 0)

            pred = extract_yes_no(raw)
            gt = s["answer"]
            cat = s.get("category", "unknown")

            d = per_split[cat]
            d["total"] += 1
            if pred == "yes" and gt == "yes": d["tp"] += 1
            elif pred == "yes" and gt == "no": d["fp"] += 1
            elif pred == "no" and gt == "no": d["tn"] += 1
            elif pred == "no" and gt == "yes": d["fn"] += 1

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                total_d = sum(d["tp"] + d["tn"] for d in per_split.values())
                total_n = sum(d["total"] for d in per_split.values())
                print(f"  [{label}] {i+1}/{len(samples)} "
                      f"acc={total_d/total_n:.1%} ({elapsed:.0f}s)", flush=True)

        except Exception as e:
            error_count += 1
            if error_count <= 3:
                print(f"  [{label}] ERROR at sample {i} (#{error_count}): {type(e).__name__}: {e}", flush=True)
            if error_count == 11:
                print(f"  [{label}] WARNING: {error_count} errors so far — suppressing further error logs", flush=True)
            cat = s.get("category", "unknown")
            per_split[cat]["total"] += 1

    if error_count > 0:
        print(f"  [{label}] TOTAL ERRORS: {error_count}/{len(samples)} ({error_count/len(samples)*100:.1f}%)", flush=True)

    # Compute metrics per split
    results = {}
    for cat in POPE_SPLITS:
        d = per_split[cat]
        n = d["total"]
        if n == 0:
            continue
        tp, fp, tn, fn = d["tp"], d["fp"], d["tn"], d["fn"]
        acc = (tp + tn) / n
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        yes_rate = (tp + fp) / n

        results[cat] = {
            "acc": acc, "f1": f1, "precision": precision,
            "recall": recall, "yes_rate": yes_rate, "total": n,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }

    # Overall
    total = sum(d["total"] for d in per_split.values())
    total_tp = sum(d["tp"] for d in per_split.values())
    total_fp = sum(d["fp"] for d in per_split.values())
    total_tn = sum(d["tn"] for d in per_split.values())
    total_fn = sum(d["fn"] for d in per_split.values())

    overall_acc = (total_tp + total_tn) / total if total > 0 else 0
    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0

    results["overall"] = {
        "acc": overall_acc, "f1": overall_f1, "precision": overall_p,
        "recall": overall_r, "yes_rate": (total_tp + total_fp) / total if total > 0 else 0,
        "total": total,
        "avg_think_words": float(np.mean(think_lengths)) if think_lengths else 0,
    }

    return results


def evaluate_blind(model, processor, samples, device, n=200, label=""):
    """Blind test: real vs black image accuracy."""
    real_correct = blind_correct = total = error_count = 0
    t0 = time.time()

    for i, s in enumerate(samples[:n]):
        try:
            q = s["question"] + " Please answer yes or no."
            gt = s["answer"]

            # Real image
            inputs = prepare_inputs(processor, s["image"], q, device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            raw = processor.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=False)
            if extract_yes_no(raw) == gt:
                real_correct += 1

            # Black image
            black = Image.new('RGB', s["image"].size, (0, 0, 0))
            inputs_b = prepare_inputs(processor, black, q, device)
            with torch.no_grad():
                out_b = model.generate(**inputs_b, max_new_tokens=512, do_sample=False)
            raw_b = processor.tokenizer.decode(
                out_b[0][inputs_b["input_ids"].shape[1]:],
                skip_special_tokens=False)
            if extract_yes_no(raw_b) == gt:
                blind_correct += 1

            total += 1

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                ra = real_correct / total
                ba = blind_correct / total
                print(f"  [{label} blind] {i+1}/{n} "
                      f"real={ra:.1%} blind={ba:.1%} gap={ra-ba:.1%} ({elapsed:.0f}s)",
                      flush=True)

        except Exception as e:
            error_count += 1
            if error_count <= 3:
                print(f"  [{label} blind] ERROR at sample {i} (#{error_count}): {type(e).__name__}: {e}", flush=True)
            if error_count == 11:
                print(f"  [{label} blind] WARNING: {error_count} errors so far — suppressing further error logs", flush=True)
            total += 1

    if error_count > 0:
        print(f"  [{label} blind] TOTAL ERRORS: {error_count}/{n} ({error_count/n*100:.1f}%)", flush=True)
    ra = real_correct / total if total > 0 else 0
    ba = blind_correct / total if total > 0 else 0
    return {"real_acc": ra, "blind_acc": ba, "gap": ra - ba, "total": total}


# ═══════════════════════════════════════════════════
#  Report Generation
# ═══════════════════════════════════════════════════

def generate_report(all_results, report_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid")

    report_dir.mkdir(parents=True, exist_ok=True)

    # Bar chart: per-split accuracy comparison
    conditions = list(all_results.keys())
    splits = POPE_SPLITS + ["overall"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(splits))
    width = 0.35
    colors = ['#4C72B0', '#DD8452']

    for i, cond in enumerate(conditions):
        accs = [all_results[cond]["pope"].get(s, {}).get("acc", 0) * 100
                for s in splits]
        ax.bar(x + i * width - width/2, accs, width, label=cond, color=colors[i])
        for j, v in enumerate(accs):
            ax.text(x[j] + i * width - width/2, v + 0.3, f"{v:.1f}",
                    ha='center', va='bottom', fontsize=9)

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("POPE Full 9K Evaluation: Baseline vs GRPO-LSR", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in splits], fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(70, 100)
    plt.tight_layout()
    fig.savefig(report_dir / "pope_per_split.png", dpi=150)
    plt.close(fig)

    # Blind Gap comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    conds = list(all_results.keys())
    gaps = [all_results[c].get("blind", {}).get("gap", 0) * 100 for c in conds]
    bars = ax.bar(conds, gaps, color=colors[:len(conds)], width=0.5)
    for bar, val in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.5, f"{val:.1f}pp",
                ha='center', fontsize=12, fontweight='bold')
    ax.set_ylabel("Blind Gap (pp)", fontsize=12)
    ax.set_title("Blind Test Gap: Real vs Black Image", fontsize=14)
    ax.set_ylim(0, max(gaps) * 1.3)
    plt.tight_layout()
    fig.savefig(report_dir / "blind_gap.png", dpi=150)
    plt.close(fig)

    # Markdown report
    md = f"""# Full POPE Evaluation Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Model**: Qwen3-VL-2B-Thinking
**Eval**: Full 9000 POPE samples (3000 per split)

## Per-Split Results

| Condition | Split | Acc | F1 | Precision | Recall | Yes Rate |
|-----------|-------|-----|-----|-----------|--------|----------|
"""
    for cond, data in all_results.items():
        for split in splits:
            d = data["pope"].get(split, {})
            if not d:
                continue
            md += (f"| {cond} | {split} | {d.get('acc',0)*100:.1f}% | "
                   f"{d.get('f1',0)*100:.1f}% | {d.get('precision',0)*100:.1f}% | "
                   f"{d.get('recall',0)*100:.1f}% | {d.get('yes_rate',0)*100:.1f}% |\n")

    md += "\n## Blind Test Gap\n\n"
    md += "| Condition | Real Acc | Blind Acc | Gap |\n"
    md += "|-----------|----------|-----------|-----|\n"
    for cond, data in all_results.items():
        b = data.get("blind", {})
        md += (f"| {cond} | {b.get('real_acc',0)*100:.1f}% | "
               f"{b.get('blind_acc',0)*100:.1f}% | {b.get('gap',0)*100:.1f}pp |\n")

    md += f"\n![Per-split](pope_per_split.png)\n![Gap](blind_gap.png)\n"

    with open(report_dir / "REPORT.md", "w") as f:
        f.write(md)

    print(f"[report] {report_dir}")


# ═══════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════

def main():
    report_dir = Path("lab/reports/full_pope_eval")
    report_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Full 9K POPE Evaluation")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading full POPE dataset...")
    samples = load_full_pope()

    # Subsample for blind test (200 per condition — balanced)
    blind_samples = []
    per_split_blind = defaultdict(list)
    for s in samples:
        cat = s["category"]
        if len(per_split_blind[cat]) < 67:  # ~200 total
            per_split_blind[cat].append(s)
    for cat in POPE_SPLITS:
        blind_samples.extend(per_split_blind[cat])

    all_results = {}

    # Baseline
    print("\n[2/5] Evaluating BASELINE...")
    model, processor = load_model(None)
    device = next(model.parameters()).device

    pope_baseline = evaluate_pope(model, processor, samples, device, "Baseline")
    blind_baseline = evaluate_blind(model, processor, blind_samples, device, 200, "Baseline")
    all_results["Baseline"] = {"pope": pope_baseline, "blind": blind_baseline}

    print(f"\n  Baseline: Acc={pope_baseline['overall']['acc']:.1%} "
          f"F1={pope_baseline['overall']['f1']:.1%} "
          f"Gap={blind_baseline['gap']:.1%}")

    del model
    torch.cuda.empty_cache(); gc.collect()

    # GRPO-LSR Best
    print(f"\n[3/5] Evaluating GRPO-LSR (R4 best)...")
    model, processor = load_model(BEST_CKPT)
    device = next(model.parameters()).device

    pope_grpo = evaluate_pope(model, processor, samples, device, "GRPO-LSR")
    blind_grpo = evaluate_blind(model, processor, blind_samples, device, 200, "GRPO-LSR")
    all_results["GRPO-LSR"] = {"pope": pope_grpo, "blind": blind_grpo}

    print(f"\n  GRPO-LSR: Acc={pope_grpo['overall']['acc']:.1%} "
          f"F1={pope_grpo['overall']['f1']:.1%} "
          f"Gap={blind_grpo['gap']:.1%}")

    del model
    torch.cuda.empty_cache(); gc.collect()

    # Delta
    delta_acc = (pope_grpo['overall']['acc'] - pope_baseline['overall']['acc']) * 100
    delta_gap = (blind_grpo['gap'] - blind_baseline['gap']) * 100
    print(f"\n  Delta: Acc {delta_acc:+.1f}pp, Gap {delta_gap:+.1f}pp")

    # Save results
    print("\n[4/5] Generating report...")
    # Convert to serializable
    serializable = {}
    for k, v in all_results.items():
        serializable[k] = {
            "pope": {sk: dict(sv) for sk, sv in v["pope"].items()},
            "blind": dict(v["blind"]),
        }
    with open(report_dir / "results.json", "w") as f:
        json.dump(serializable, f, indent=2)

    generate_report(all_results, report_dir)

    # Print final table
    print("\n[5/5] FINAL RESULTS")
    print("=" * 70)
    print(f"{'Condition':<15} {'Split':<12} {'Acc':>6} {'F1':>6} {'P':>6} {'R':>6}")
    print("-" * 70)
    for cond in all_results:
        for split in POPE_SPLITS + ["overall"]:
            d = all_results[cond]["pope"].get(split, {})
            if not d:
                continue
            print(f"{cond:<15} {split:<12} "
                  f"{d['acc']*100:5.1f}% {d.get('f1',0)*100:5.1f}% "
                  f"{d.get('precision',0)*100:5.1f}% {d.get('recall',0)*100:5.1f}%")
    print("-" * 70)
    print(f"Blind Gap: Baseline={blind_baseline['gap']*100:.1f}pp, "
          f"GRPO-LSR={blind_grpo['gap']*100:.1f}pp "
          f"(Δ={delta_gap:+.1f}pp)")
    print("=" * 70)


if __name__ == "__main__":
    main()
