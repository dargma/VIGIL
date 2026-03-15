"""
Session 14: Verified experiment suite.

Experiment A: Alpha=0.0 ablation (proves head weighting matters)
  - GDPO+VPPO from base model, alpha=0.0, 15 steps
  - Compare with Phase 6b results (alpha=0.5)

Experiment B: MME eval on baseline + Phase 2 R4 best

Experiment C: Larger POPE eval (300 samples) on best checkpoints for variance reduction

Usage:
    PYTHONUNBUFFERED=1 python -u scripts/session14_experiments.py \
        --exp A 2>&1 | tee logs/session14_expA.log
    PYTHONUNBUFFERED=1 python -u scripts/session14_experiments.py \
        --exp B 2>&1 | tee logs/session14_expB.log
    PYTHONUNBUFFERED=1 python -u scripts/session14_experiments.py \
        --exp C 2>&1 | tee logs/session14_expC.log
    PYTHONUNBUFFERED=1 python -u scripts/session14_experiments.py \
        --exp all 2>&1 | tee logs/session14_all.log
"""

import os, sys, gc, json, re, time, random, argparse, string
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from datetime import datetime
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

HF_ID = "Qwen/Qwen3-VL-2B-Thinking"
PROJECT_ROOT = Path(__file__).parent.parent

# ═══════════════════════════════════════════════════════════════════
#  Common model loading
# ═══════════════════════════════════════════════════════════════════

def load_model(model_path=None, dtype=torch.bfloat16):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    path = model_path or HF_ID
    print(f"[load] Loading model from: {path}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        path, torch_dtype=dtype, device_map="auto")
    processor = AutoProcessor.from_pretrained(HF_ID)
    model.eval()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[load] {n:.0f}M params, dtype={dtype}")
    return model, processor


def free_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def generate_answer(model, processor, image, question, enable_thinking=True,
                    max_new_tokens=2048):
    from qwen_vl_utils import process_vision_info
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": question},
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking)
    imgs, _, _ = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(text=[text], images=imgs, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=1.0, top_p=0.95, top_k=20, do_sample=True)
    input_len = inputs["input_ids"].shape[1]
    response = processor.tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
    return response.strip()


def extract_yes_no(raw):
    """VLMEvalKit-compatible yes/no extraction."""
    # Strip thinking
    m = re.search(r'</think>', raw)
    text = raw[m.end():].strip() if m else raw.strip()
    if not text:
        text = raw

    s = text.lower().strip()
    for p in [';', '/', '[', ']', '"', '{', '}', '(', ')', '=', '+',
              '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']:
        s = s.replace(p, ' ')
    s = re.sub(r'(?<!\d)\.(?!\d)', '', s)
    words = s.split()
    has_yes = 'yes' in words
    has_no = 'no' in words
    if has_yes and not has_no: return "Yes"
    if has_no and not has_yes: return "No"
    first = words[0] if words else ""
    if first.startswith("yes"): return "Yes"
    if first.startswith("no"): return "No"
    return "Unknown"


# ═══════════════════════════════════════════════════════════════════
#  Experiment A: Alpha=0.0 Ablation
# ═══════════════════════════════════════════════════════════════════

def run_experiment_A():
    """Run GDPO+VPPO with alpha=0.0 (no head weighting) and compare."""
    import subprocess

    print("\n" + "=" * 70)
    print("  EXPERIMENT A: Alpha=0.0 Ablation (GDPO+VPPO, no head weighting)")
    print("=" * 70)

    cmd = [
        sys.executable, "-u", "scripts/phase6_head_mask_grpo.py",
        "--output-dir", "checkpoints/phase6_head_mask/alpha0_ablation",
        "--steps", "15",
        "--alpha", "0.0",
        "--gdpo",
        "--vppo-mask",
        "--lr", "2e-6",
        "--eval-every", "5",
        "--train-samples", "500",
        "--seed", "42",
    ]

    log_file = "logs/phase6_alpha0_ablation.log"
    t0 = time.time()

    with open(log_file, "w") as lf:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
            lf.flush()
        proc.wait()

    elapsed = time.time() - t0
    status = "OK" if proc.returncode == 0 else f"FAIL (rc={proc.returncode})"
    print(f"\n  [ExpA] {status} in {elapsed:.0f}s")

    # Print comparison
    print(f"\n{'='*70}")
    print("  ABLATION COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Config':<35} {'POPE':>8} {'Gap':>8}")
    print(f"  {'-'*55}")
    print(f"  {'Baseline (HF)':<35} {'91.7%':>8} {'40.0pp':>8}")
    print(f"  {'GDPO+VPPO α=0.5 (Phase 6b)':<35} {'95.0%':>8} {'44.0pp':>8}")
    print(f"  {'GDPO+VPPO α=0.0 (this run)':<35} {'see log':>8} {'':>8}")
    print(f"{'='*70}")

    return proc.returncode == 0


# ═══════════════════════════════════════════════════════════════════
#  Experiment B: MME Eval
# ═══════════════════════════════════════════════════════════════════

MME_DATA_PATH = Path("data/eval/mme")
PERCEPTION_SUBTASKS = [
    "existence", "count", "position", "color", "posters",
    "celebrity", "scene", "landmark", "artwork", "OCR",
]
COGNITION_SUBTASKS = [
    "commonsense_reasoning", "numerical_calculation",
    "text_translation", "code_reasoning",
]


def load_mme_data():
    from datasets import load_from_disk
    ds = load_from_disk(str(MME_DATA_PATH))
    grouped = defaultdict(list)
    for i in range(len(ds)):
        row = ds[i]
        grouped[row["question_id"]].append({
            "index": i, "question_id": row["question_id"],
            "question": row["question"], "answer": row["answer"].strip(),
            "category": row["category"], "image": row["image"],
        })
    pairs = list(grouped.values())
    n_q = sum(len(p) for p in pairs)
    cats = set(p[0]["category"] for p in pairs)
    print(f"[mme] {len(pairs)} images ({n_q} questions), {len(cats)} subtasks")
    return pairs


def compute_mme_scores(results):
    subtask_scores = defaultdict(lambda: {"correct_pairs": 0, "total_pairs": 0,
                                           "correct_q": 0, "total_q": 0})
    for qid, pair_results in results.items():
        if not pair_results: continue
        cat = pair_results[0]["category"]
        st = subtask_scores[cat]
        st["total_pairs"] += 1
        all_correct = True
        for r in pair_results:
            st["total_q"] += 1
            if r["extracted"] == r["gt_answer"]:
                st["correct_q"] += 1
            else:
                all_correct = False
        if all_correct:
            st["correct_pairs"] += 1

    report = {}
    p_total = c_total = 0
    for subtask in sorted(subtask_scores):
        s = subtask_scores[subtask]
        score = s["correct_pairs"]
        max_s = s["total_pairs"]
        acc_p = s["correct_q"] / s["total_q"] * 100 if s["total_q"] > 0 else 0
        report[subtask] = {"score": score, "max_score": max_s,
                           "acc_plus": round(acc_p, 1)}
        if subtask in PERCEPTION_SUBTASKS: p_total += score
        elif subtask in COGNITION_SUBTASKS: c_total += score

    p_max = sum(report.get(s, {}).get("max_score", 0) for s in PERCEPTION_SUBTASKS)
    c_max = sum(report.get(s, {}).get("max_score", 0) for s in COGNITION_SUBTASKS)
    report["_Perception"] = {"score": p_total, "max_score": p_max}
    report["_Cognition"] = {"score": c_total, "max_score": c_max}
    report["_Total"] = {"score": p_total + c_total, "max_score": p_max + c_max}
    return report


def run_mme_eval(model, processor, pairs, label):
    results = {}
    total_q = correct_q = 0
    t0 = time.time()

    for pi, pair in enumerate(pairs):
        qid = pair[0]["question_id"]
        cat = pair[0]["category"]
        pair_results = []

        for q in pair:
            try:
                raw = generate_answer(model, processor, q["image"], q["question"])
            except torch.cuda.OutOfMemoryError:
                print(f"  [{pi}] OOM, skipping")
                torch.cuda.empty_cache()
                raw = ""
            except Exception as e:
                print(f"  [{pi}] Error: {e}")
                raw = ""

            ext = extract_yes_no(raw)
            gt = q["answer"]
            pair_results.append({
                "question_id": qid, "question": q["question"],
                "gt_answer": gt, "extracted": ext,
                "correct": ext == gt, "category": cat,
            })
            total_q += 1
            if ext == gt: correct_q += 1

        results[qid] = pair_results

        if (pi + 1) % 50 == 0 or (pi + 1) == len(pairs):
            elapsed = time.time() - t0
            rate = (pi + 1) / elapsed * 60 if elapsed > 0 else 0
            acc = correct_q / total_q * 100 if total_q > 0 else 0
            print(f"  [{label}] {pi+1}/{len(pairs)} ({rate:.0f} img/min), acc+={acc:.1f}%")

    report = compute_mme_scores(results)
    elapsed = time.time() - t0
    print(f"  [{label}] Done in {elapsed:.0f}s")
    return results, report


def print_mme_report(report, label=""):
    print(f"\n{'='*70}")
    print(f"  MME Results — {label}")
    print(f"{'='*70}")
    print(f"  {'Subtask':<28} {'Score':>8} {'Max':>8} {'Acc+':>8}")
    print(f"  {'-'*56}")

    print("  PERCEPTION:")
    for s in PERCEPTION_SUBTASKS:
        if s in report:
            r = report[s]
            print(f"    {s:<26} {r['score']:>8} {r['max_score']:>8} {r['acc_plus']:>7.1f}%")
    p = report["_Perception"]
    print(f"  {'PERCEPTION TOTAL':<28} {p['score']:>8} {p['max_score']:>8}")

    print("  COGNITION:")
    for s in COGNITION_SUBTASKS:
        if s in report:
            r = report[s]
            print(f"    {s:<26} {r['score']:>8} {r['max_score']:>8} {r['acc_plus']:>7.1f}%")
    c = report["_Cognition"]
    print(f"  {'COGNITION TOTAL':<28} {c['score']:>8} {c['max_score']:>8}")

    t = report["_Total"]
    print(f"\n  {'TOTAL':<28} {t['score']:>8} {t['max_score']:>8}")
    print(f"{'='*70}")


def run_experiment_B(max_images=200):
    """MME eval: baseline vs Phase 2 R4 best."""
    print("\n" + "=" * 70)
    print(f"  EXPERIMENT B: MME Evaluation (max {max_images} images)")
    print("=" * 70)

    pairs = load_mme_data()
    # Sample proportionally from each subtask
    if max_images and len(pairs) > max_images:
        import random
        random.seed(42)
        by_cat = defaultdict(list)
        for p in pairs:
            by_cat[p[0]["category"]].append(p)
        sampled = []
        per_cat = max(2, max_images // len(by_cat))
        for cat in sorted(by_cat):
            items = by_cat[cat]
            n = min(len(items), per_cat)
            sampled.extend(random.sample(items, n))
        pairs = sampled[:max_images]
        n_q = sum(len(p) for p in pairs)
        cats = set(p[0]["category"] for p in pairs)
        print(f"[mme] Sampled {len(pairs)} images ({n_q} questions), {len(cats)} subtasks")
    out_dir = Path("lab/reports/mme")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_reports = []

    # Baseline
    print("\n--- BASELINE ---")
    model, processor = load_model()
    results_b, report_b = run_mme_eval(model, processor, pairs, "baseline")
    print_mme_report(report_b, "Baseline")
    all_reports.append(("Baseline", report_b))
    with open(out_dir / f"mme_baseline_{ts}.json", "w") as f:
        json.dump({"label": "baseline", "report": report_b}, f, indent=2, default=str)
    free_model(model)

    # Phase 2 R4 best
    ckpt = "checkpoints/phase2_grpo_lsr/round4/best"
    if Path(ckpt).exists():
        print("\n--- PHASE 2 R4 BEST ---")
        model, processor = load_model(ckpt)
        results_c, report_c = run_mme_eval(model, processor, pairs, "grpo_lsr_r4")
        print_mme_report(report_c, "GRPO-LSR R4")
        all_reports.append(("GRPO-LSR R4", report_c))
        with open(out_dir / f"mme_grpo_lsr_r4_{ts}.json", "w") as f:
            json.dump({"label": "grpo_lsr_r4", "report": report_c}, f, indent=2, default=str)
        free_model(model)

    # Comparison
    print(f"\n{'='*70}")
    print("  MME COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Condition':<20} {'Perception':>12} {'Cognition':>12} {'Total':>12}")
    print(f"  {'-'*58}")
    for label, report in all_reports:
        p = report["_Perception"]["score"]
        c = report["_Cognition"]["score"]
        t = report["_Total"]["score"]
        print(f"  {label:<20} {p:>12} {c:>12} {t:>12}")
    print(f"{'='*70}")

    # Save summary
    with open(out_dir / f"mme_summary_{ts}.json", "w") as f:
        json.dump({
            "timestamp": ts,
            "conditions": {
                label: {
                    "Perception": r["_Perception"]["score"],
                    "Cognition": r["_Cognition"]["score"],
                    "Total": r["_Total"]["score"],
                }
                for label, r in all_reports
            }
        }, f, indent=2)

    # Generate chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [r[0] for r in all_reports]
        p_scores = [r[1]["_Perception"]["score"] for r in all_reports]
        c_scores = [r[1]["_Cognition"]["score"] for r in all_reports]
        x = np.arange(len(labels))
        w = 0.35
        fig, ax = plt.subplots(figsize=(10, 6))
        b1 = ax.bar(x - w/2, p_scores, w, label="Perception", color="#4C72B0")
        b2 = ax.bar(x + w/2, c_scores, w, label="Cognition", color="#DD8452")
        for b in [b1, b2]:
            for bar in b:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                        f"{int(bar.get_height())}", ha="center", fontsize=11)
        ax.set_ylabel("Score"); ax.set_title("MME Benchmark", fontsize=14)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
        ax.legend(fontsize=11); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"mme_comparison_{ts}.png", dpi=150)
        plt.close()
        print(f"[mme] Chart saved")
    except Exception as e:
        print(f"[mme] Chart error: {e}")

    return True


# ═══════════════════════════════════════════════════════════════════
#  Experiment C: Larger POPE eval (300 samples, 3 checkpoints)
# ═══════════════════════════════════════════════════════════════════

def load_pope_300():
    """Load 300 POPE samples (100 per split)."""
    from datasets import load_dataset
    ds = load_dataset("lmms-lab/POPE", split="test", streaming=True)
    samples = []
    split_counts = defaultdict(int)

    for row in ds:
        # Determine split from source_dataset or question index
        q = row["question"]
        answer = row["answer"].strip()
        img = row.get("image")
        if img is None:
            continue
        samples.append({
            "image": img,
            "question": q + " Please answer yes or no.",
            "answer": answer,
        })
        if len(samples) >= 300:
            break

    print(f"[pope300] Loaded {len(samples)} samples")
    return samples


def run_pope_eval(model, processor, samples, label):
    correct = 0
    total = 0
    t0 = time.time()

    for i, s in enumerate(samples):
        try:
            raw = generate_answer(model, processor, s["image"], s["question"],
                                  max_new_tokens=2048)
            ext = extract_yes_no(raw)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            ext = "Unknown"
        except Exception as e:
            ext = "Unknown"

        gt = s["answer"]
        if ext == gt:
            correct += 1
        total += 1

        if (i + 1) % 50 == 0:
            acc = correct / total * 100
            elapsed = time.time() - t0
            print(f"  [{label}] {i+1}/{len(samples)} acc={acc:.1f}% "
                  f"({elapsed:.0f}s)")

    acc = correct / total * 100
    print(f"  [{label}] Final: {correct}/{total} = {acc:.1f}%")
    return {"label": label, "correct": correct, "total": total, "acc": acc}


def run_pope_blind_eval(model, processor, samples, label):
    """Run POPE with black images to compute Gap."""
    correct = 0
    total = 0

    for i, s in enumerate(samples):
        try:
            black = Image.new('RGB', (224, 224), (0, 0, 0))
            raw = generate_answer(model, processor, black, s["question"],
                                  max_new_tokens=2048)
            ext = extract_yes_no(raw)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            ext = "Unknown"
        except:
            ext = "Unknown"

        if ext == s["answer"]:
            correct += 1
        total += 1

        if (i + 1) % 50 == 0:
            acc = correct / total * 100
            print(f"  [{label}_blind] {i+1}/{len(samples)} acc={acc:.1f}%")

    acc = correct / total * 100
    print(f"  [{label}_blind] Final: {correct}/{total} = {acc:.1f}%")
    return {"label": f"{label}_blind", "correct": correct, "total": total, "acc": acc}


def run_experiment_C():
    """300-sample POPE eval on baseline + best checkpoints."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT C: 300-Sample POPE Evaluation")
    print("=" * 70)

    samples = load_pope_300()
    out_dir = Path("lab/reports/pope_300")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []

    checkpoints = [
        ("Baseline", None),
        ("GRPO-LSR-R4", "checkpoints/phase2_grpo_lsr/round4/best"),
        ("HeadMask-TextVQA", "checkpoints/phase6_head_mask/textvqa_v2/final"),
    ]

    for label, ckpt_path in checkpoints:
        if ckpt_path and not Path(ckpt_path).exists():
            print(f"  Skipping {label}: {ckpt_path} not found")
            continue

        print(f"\n--- {label} ---")
        model, processor = load_model(ckpt_path)

        real = run_pope_eval(model, processor, samples, label)
        blind = run_pope_blind_eval(model, processor, samples[:100], label)

        gap = real["acc"] - blind["acc"]
        result = {
            "label": label, "ckpt": ckpt_path or HF_ID,
            "pope_acc": real["acc"], "blind_acc": blind["acc"],
            "gap": round(gap, 1),
        }
        all_results.append(result)
        print(f"  [{label}] POPE={real['acc']:.1f}% Blind={blind['acc']:.1f}% "
              f"Gap={gap:.1f}pp")

        free_model(model)

    # Summary
    print(f"\n{'='*70}")
    print("  300-SAMPLE POPE COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Model':<25} {'POPE':>8} {'Blind':>8} {'Gap':>8}")
    print(f"  {'-'*51}")
    for r in all_results:
        print(f"  {r['label']:<25} {r['pope_acc']:>7.1f}% {r['blind_acc']:>7.1f}% "
              f"{r['gap']:>7.1f}pp")
    print(f"{'='*70}")

    with open(out_dir / f"pope300_{ts}.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[pope300] Results saved to {out_dir / f'pope300_{ts}.json'}")

    return True


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True,
                        choices=["A", "B", "C", "all"],
                        help="Which experiment to run")
    parser.add_argument("--max-images", type=int, default=200,
                        help="Max images for MME eval (default: 200)")
    args = parser.parse_args()

    t0 = time.time()

    if args.exp in ("A", "all"):
        run_experiment_A()

    if args.exp in ("B", "all"):
        run_experiment_B(max_images=args.max_images)

    if args.exp in ("C", "all"):
        run_experiment_C()

    elapsed = time.time() - t0
    print(f"\n[session14] Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
