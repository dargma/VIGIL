"""
VIGIL MMMU-Pro Evaluation with Qwen3-VL-2B-Thinking + Steering

Official settings from Qwen3-VL Technical Report (arXiv:2511.21631):
  - Dense thinking: temperature=1.0, top_p=0.95, top_k=20
  - max_new_tokens=32768
  - Prompt: VLMEvalKit standard MCQ format
  - Answer parsing: split <think>...</think>, extract letter from post-think text
  - MMMU-Pro score = avg(standard-10, vision)
  - Paper reference: Qwen3-VL-2B-Thinking MMMU-Pro = 42.5

Pipeline:
  1. Download MMMU (dev+validation for calibration) + MMMU-Pro (for eval)
  2. Baseline eval on MMMU-Pro (standard-10 + vision configs)
  3. Calibrate steering vectors using MMMU dev+validation
  4. Steered eval on MMMU-Pro
  5. Compare with paper score (42.5)

Usage:
    python scripts/eval_mmmu_pro.py --phase all
    python scripts/eval_mmmu_pro.py --phase baseline --max-samples 100
    python scripts/eval_mmmu_pro.py --phase calibrate
    python scripts/eval_mmmu_pro.py --phase steered --alphas 1,3,5
"""

import os, sys, json, re, argparse, gc, time
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.steerer import ActivationSteerer
from src.calibrator import CalibrationResult, SteeringCalibrator


# ─── Constants ───────────────────────────────────────────────────────────────

DATA_ROOT = Path("data")
MMMU_SUBJECTS = [
    "Accounting", "Agriculture", "Architecture_and_Engineering",
    "Art", "Art_Theory", "Basic_Medical_Science", "Biology",
    "Chemistry", "Clinical_Medicine", "Computer_Science",
    "Design", "Diagnostics_and_Laboratory_Medicine", "Economics",
    "Electronics", "Energy_and_Power", "Finance", "Geography",
    "History", "Literature", "Manage", "Marketing",
    "Materials", "Math", "Mechanical_Engineering", "Music",
    "Pharmacy", "Physics", "Psychology", "Public_Health",
    "Sociology",
]

PAPER_SCORE = 42.5  # Qwen3-VL-2B-Thinking MMMU-Pro (arXiv:2511.21631, Table 4)

# Official VLMEvalKit MCQ prompt (ImageMCQDataset.build_prompt)
MCQ_PROMPT_TEMPLATE = """{hint}{question}
Options:
{options}
Please select the correct answer from the options above."""

# Official vision-only prompt (no text options, question embedded in image)
VISION_PROMPT = "Identify the problem and solve it. Think step by step before answering."

# Official generation params for dense thinking models (Qwen3-VL paper §5)
# Paper uses max_new_tokens=32768 but we cap at 4096 for practical latency
# (MMMU-Pro answers rarely exceed 2K thinking + answer tokens)
GEN_KWARGS_THINKING = dict(
    max_new_tokens=4096,
    temperature=1.0,
    top_p=0.95,
    top_k=20,
    do_sample=True,
    repetition_penalty=1.0,
)


# ─── Think Tag Parsing (VLMEvalKit SPLIT_THINK) ─────────────────────────────

def split_thinking(text):
    """Split model output into thinking and answer parts.
    Follows VLMEvalKit's split_thinking convention.

    Note: Qwen3-VL-Thinking puts <think> in the generation prompt,
    so model output may start directly with thinking content followed
    by </think> and then the answer.
    """
    # Case 1: Full <think>...</think> tags present
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        answer = text[think_match.end():].strip()
        return thinking, answer

    # Case 2: Only </think> present (thinking content before it, answer after)
    # This happens because <think> is in the prompt, not in generated output
    close_match = re.search(r'</think>', text)
    if close_match:
        thinking = text[:close_match.start()].strip()
        answer = text[close_match.end():].strip()
        return thinking, answer

    return "", text.strip()


def extract_mc_answer(raw_output, num_choices=10):
    """Extract multiple-choice answer letter from model output.
    Official VLMEvalKit approach: parse post-think text for letter answer."""
    # Split off thinking
    _, answer_text = split_thinking(raw_output)

    if not answer_text:
        answer_text = raw_output  # Fallback to full output

    valid_letters = [chr(65 + i) for i in range(num_choices)]

    # Pattern 1: "The answer is (A)" / "Answer: B"
    patterns = [
        r'(?:the\s+)?answer\s+is\s*[:\s]*\(?([A-J])\)?',
        r'(?:answer|option|choice)\s*[:\s]*\(?([A-J])\)?',
        r'\*\*([A-J])\*\*',  # bold letter
        r'^\s*\(?([A-J])\)?[\.\s]*$',  # just the letter on a line
    ]
    for pattern in patterns:
        m = re.search(pattern, answer_text, re.IGNORECASE | re.MULTILINE)
        if m:
            letter = m.group(1).upper()
            if letter in valid_letters:
                return letter

    # Pattern 2: Last standalone letter A-J in text
    found = re.findall(r'\b([A-J])\b', answer_text)
    for letter in reversed(found):
        if letter in valid_letters:
            return letter

    # Pattern 3: First letter A-J anywhere
    for letter in found:
        if letter in valid_letters:
            return letter

    return None


# ─── Data Download ───────────────────────────────────────────────────────────

def download_mmmu_splits():
    """Download MMMU dev + validation splits for calibration."""
    from datasets import load_dataset, Dataset

    cache_dir = DATA_ROOT / "mmmu_full"
    if cache_dir.exists() and (cache_dir / "dataset_info.json").exists():
        from datasets import load_from_disk
        ds = load_from_disk(str(cache_dir))
        print(f"[data] MMMU loaded from cache: {len(ds)} samples")
        return ds

    print("[data] Downloading MMMU (all subjects, dev+validation)...")
    all_samples = []
    for subject in MMMU_SUBJECTS:
        for split in ["dev", "validation"]:
            try:
                ds = load_dataset("MMMU/MMMU", subject, split=split)
                for row in ds:
                    row_dict = dict(row)
                    row_dict["subject"] = subject
                    row_dict["orig_split"] = split
                    all_samples.append(row_dict)
                print(f"  {subject}/{split}: {len(ds)} samples")
            except Exception as e:
                print(f"  {subject}/{split}: FAILED ({e})")

    print(f"[data] Total MMMU samples: {len(all_samples)}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    ds = Dataset.from_list(all_samples)
    ds.save_to_disk(str(cache_dir))
    print(f"[data] Saved to {cache_dir}")
    return ds


def download_mmmu_pro():
    """Download MMMU-Pro dataset (standard-10 + vision configs)."""
    from datasets import load_dataset, load_from_disk

    configs_to_download = {
        "standard (10 options)": "eval/mmmu_pro_standard10",
        "vision": "eval/mmmu_pro_vision",
    }

    datasets = {}
    for config_name, cache_path in configs_to_download.items():
        full_path = DATA_ROOT / cache_path
        if full_path.exists():
            try:
                ds = load_from_disk(str(full_path))
                print(f"[data] MMMU-Pro '{config_name}' loaded from cache: {len(ds)}")
                datasets[config_name] = ds
                continue
            except Exception:
                pass

        print(f"[data] Downloading MMMU-Pro '{config_name}'...")
        try:
            ds = load_dataset("MMMU/MMMU_Pro", config_name, split="test")
            full_path.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(full_path))
            print(f"  → {len(ds)} samples saved to {full_path}")
            datasets[config_name] = ds
        except Exception as e:
            print(f"  → FAILED: {e}")

    return datasets


# ─── Model Loading ───────────────────────────────────────────────────────────

def load_thinking_model():
    """Load Qwen3-VL-2B-Thinking model."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    hf_id = "Qwen/Qwen3-VL-2B-Thinking"
    print(f"[model] Loading {hf_id}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(hf_id)
    model.eval()

    model_info = {
        "model": model,
        "processor": processor,
        "tokenizer": processor.tokenizer,
        "get_layers_fn": lambda: model.model.language_model.layers,
        "get_lm_head_fn": lambda: model.lm_head,
        "get_norm_fn": lambda: model.model.language_model.norm,
        "num_layers": 28,
        "num_heads": 16,
        "num_kv_heads": 8,
        "head_dim": 128,
        "hidden_size": 2048,
        "gqa": True,
        "steer_layers_start": 4,
        "device": next(model.parameters()).device,
        "model_type": "qwen3_vl",
    }

    # Verify architecture
    cfg = model.config
    llm_cfg = getattr(cfg, "text_config", cfg)
    assert getattr(llm_cfg, "num_hidden_layers", None) == 28, "Layer count mismatch"
    print(f"[model] Loaded. Layers=28, Heads=16Q/8KV, hidden=2048")

    return model_info


# ─── Prompt Building (VLMEvalKit official format) ────────────────────────────

def build_mcq_prompt(question, options, hint=""):
    """Build VLMEvalKit standard MCQ prompt."""
    options_text = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))
    hint_text = f"{hint}\n" if hint else ""
    return MCQ_PROMPT_TEMPLATE.format(hint=hint_text, question=question, options=options_text)


def parse_mmmu_pro_options(row):
    """Parse options from MMMU-Pro row."""
    options = row.get("options", [])
    if isinstance(options, str):
        try:
            options = json.loads(options)
        except (json.JSONDecodeError, ValueError):
            try:
                options = eval(options)
            except:
                options = []
    return options if isinstance(options, list) else []


def get_mmmu_pro_image(row):
    """Extract image from MMMU-Pro row."""
    for key in ["image", "image_1", "question_image"]:
        img = row.get(key)
        if img is not None and isinstance(img, Image.Image):
            return img
    return None


# ─── Generation ──────────────────────────────────────────────────────────────

def generate_answer(model_info, image, prompt):
    """Generate answer using official Qwen3-VL-2B-Thinking settings."""
    from qwen_vl_utils import process_vision_info

    model = model_info["model"]
    processor = model_info["processor"]

    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]
    # Enable thinking mode for Thinking variant
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True,
    )

    if image is not None:
        img_inputs, _, _ = process_vision_info(messages, return_video_kwargs=True)
    else:
        img_inputs = None

    inputs = processor(
        text=[text], images=img_inputs,
        return_tensors="pt", padding=True,
    )
    inputs = {k: v.to(model_info["device"]) for k, v in inputs.items()}

    with torch.no_grad():
        gen = model.generate(**inputs, **GEN_KWARGS_THINKING)

    out_ids = gen[0][inputs["input_ids"].shape[1]:]
    # Keep think tags for parsing, then strip special tokens
    raw = processor.tokenizer.decode(out_ids, skip_special_tokens=False).strip()
    # Remove EOS and padding tokens but keep <think></think>
    for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
        raw = raw.replace(tok, "")
    return raw.strip()


# ─── Evaluation ──────────────────────────────────────────────────────────────

def eval_mmmu_pro_config(model_info, dataset, config_name, max_samples=None):
    """Evaluate one MMMU-Pro config (standard-10 or vision)."""
    n = min(len(dataset), max_samples or len(dataset))
    is_vision = "vision" in config_name.lower()

    correct = 0
    total = 0
    records = []
    subject_scores = defaultdict(lambda: {"correct": 0, "total": 0})
    think_lengths = []

    print(f"\n[eval] {config_name} — {n} samples")
    t0 = time.time()

    for i in range(n):
        row = dataset[i]
        try:
            image = get_mmmu_pro_image(row)
            question = row.get("question", "")
            answer_gt = str(row.get("answer", "")).strip().upper()
            subject = row.get("subject", "unknown")
            hint = row.get("hint", "") or ""

            options = parse_mmmu_pro_options(row)
            num_choices = len(options) if options else 10

            # Build prompt
            if is_vision:
                prompt = VISION_PROMPT
            else:
                prompt = build_mcq_prompt(question, options, hint)

            # Generate
            raw = generate_answer(model_info, image, prompt)

            # Parse
            thinking, answer_text = split_thinking(raw)
            think_lengths.append(len(thinking.split()) if thinking else 0)
            pred = extract_mc_answer(raw, num_choices)
            is_correct = (pred == answer_gt) if pred else False

            if is_correct:
                correct += 1
            total += 1

            subject_scores[subject]["total"] += 1
            if is_correct:
                subject_scores[subject]["correct"] += 1

            records.append({
                "index": i,
                "question": question[:100],
                "answer_gt": answer_gt,
                "prediction": pred,
                "raw_output": raw[:300],
                "thinking_len": len(thinking.split()) if thinking else 0,
                "correct": is_correct,
                "subject": subject,
            })

            if (i + 1) % 50 == 0:
                acc = correct / total * 100 if total > 0 else 0
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (n - i - 1) / rate if rate > 0 else 0
                avg_think = np.mean(think_lengths) if think_lengths else 0
                print(f"  [{i+1}/{n}] acc={acc:.1f}% ({correct}/{total}) "
                      f"avg_think={avg_think:.0f}w rate={rate:.1f}/s ETA={eta/60:.1f}m")

        except Exception as e:
            print(f"  [{i}] ERR: {e}")
            records.append({"index": i, "error": str(e), "correct": False})
            total += 1

    acc = correct / total * 100 if total > 0 else 0
    elapsed = time.time() - t0
    avg_think = np.mean(think_lengths) if think_lengths else 0
    print(f"  → {config_name}: {acc:.1f}% ({correct}/{total}) "
          f"avg_think={avg_think:.0f}w elapsed={elapsed/60:.1f}m")

    # Per-subject breakdown
    per_subject = {}
    for subj, sc in sorted(subject_scores.items()):
        sub_acc = sc["correct"] / sc["total"] * 100 if sc["total"] > 0 else 0
        per_subject[subj] = {"acc": sub_acc, **sc}

    return {
        "config": config_name,
        "acc": acc,
        "correct": correct,
        "total": total,
        "avg_thinking_words": float(avg_think),
        "elapsed_min": elapsed / 60,
        "records": records,
        "per_subject": per_subject,
    }


# ─── Calibration ─────────────────────────────────────────────────────────────

def calibrate_on_mmmu(model_info, mmmu_dataset, max_samples=500):
    """Calibrate steering vectors using MMMU dev+validation."""
    from qwen_vl_utils import process_vision_info

    cal_dir = Path("checkpoints/calibration/qwen3_vl_2b_thinking_mmmu")
    if cal_dir.exists() and (cal_dir / "steering_vectors.pt").exists():
        print(f"[calibration] Loading cached from {cal_dir}")
        return CalibrationResult.load(str(cal_dir))

    print(f"[calibration] Running on MMMU ({max_samples} samples)...")

    # Prepare samples with images
    samples = []
    for i, row in enumerate(mmmu_dataset):
        if i >= max_samples * 2:  # scan more to get enough with images
            break

        options_raw = row.get("options", [])
        if isinstance(options_raw, str):
            try:
                options_raw = json.loads(options_raw)
            except:
                try:
                    options_raw = eval(options_raw)
                except:
                    options_raw = []

        image = None
        for key in ["image_1", "image", "image_2"]:
            img = row.get(key)
            if img is not None and isinstance(img, Image.Image):
                image = img
                break

        if image is None:
            continue

        answer = str(row.get("answer", "")).strip().upper()
        samples.append({
            "question": row.get("question", ""),
            "answer": answer,
            "choices": options_raw if isinstance(options_raw, list) else [],
            "image": image,
            "type": "mc",
        })
        if len(samples) >= max_samples:
            break

    print(f"[calibration] {len(samples)} samples with images")

    def process_fn(model_info, sample):
        processor = model_info["processor"]
        question = sample["question"]
        choices = sample.get("choices", [])

        if choices:
            prompt = build_mcq_prompt(question, choices)
        else:
            prompt = f"{question}\nAnswer with the letter only."

        messages = [{"role": "user", "content": [
            {"type": "image", "image": sample["image"]},
            {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True)
        img_inputs, _, _ = process_vision_info(messages, return_video_kwargs=True)
        inputs = processor(text=[text], images=img_inputs,
                           return_tensors="pt", padding=True)
        inputs = {k: v.to(model_info["device"]) for k, v in inputs.items()}
        return inputs, sample["answer"]

    calibrator = SteeringCalibrator(model_info, top_k=20)
    result = calibrator.calibrate(samples, process_fn, max_samples=len(samples))
    result.save(str(cal_dir))
    return result


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "download", "baseline", "calibrate", "steered", "report"])
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit eval samples per config (None=all 1730)")
    parser.add_argument("--cal-samples", type=int, default=500,
                        help="Calibration samples from MMMU")
    parser.add_argument("--alphas", type=str, default="1,3,5,7",
                        help="Steering alpha values to test (comma-separated)")
    parser.add_argument("--output-dir", type=str, default="lab/reports/mmmu_pro")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    alphas = [float(a) for a in args.alphas.split(",")]

    phases = ["download", "baseline", "calibrate", "steered", "report"] if args.phase == "all" else [args.phase]

    # ── Phase 1: Download ──
    if "download" in phases:
        print("\n" + "="*60)
        print("Phase 1: Download datasets")
        print("="*60)
        mmmu_ds = download_mmmu_splits()
        mmmu_pro = download_mmmu_pro()
        print(f"\nMMU: {len(mmmu_ds)} samples")
        for k, v in mmmu_pro.items():
            print(f"MMMU-Pro '{k}': {len(v)} samples")

    # ── Phase 2: Baseline ──
    all_results = {}
    model_info = None

    if "baseline" in phases:
        print("\n" + "="*60)
        print("Phase 2: Baseline eval (Qwen3-VL-2B-Thinking)")
        print(f"  Official settings: temp=1.0, top_p=0.95, top_k=20, max_tokens=32768")
        print("="*60)
        model_info = load_thinking_model()
        mmmu_pro = download_mmmu_pro()

        for config_name, dataset in mmmu_pro.items():
            result = eval_mmmu_pro_config(
                model_info, dataset, config_name,
                max_samples=args.max_samples,
            )
            all_results[f"baseline_{config_name}"] = result

            # Save intermediate per-config
            safe_result = {k: v for k, v in result.items() if k != "records"}
            with open(output_dir / f"baseline_{config_name.replace(' ', '_')}_{ts}.json", "w") as f:
                json.dump(safe_result, f, indent=2)

        # Compute MMMU-Pro score (average of standard-10 + vision)
        s10 = [v for k, v in all_results.items() if "standard" in k and "baseline" in k]
        vis = [v for k, v in all_results.items() if "vision" in k and "baseline" in k]
        if s10 and vis:
            mmmu_pro_score = (s10[0]["acc"] + vis[0]["acc"]) / 2
            all_results["baseline_mmmu_pro_score"] = mmmu_pro_score
            delta = mmmu_pro_score - PAPER_SCORE
            print(f"\n>>> Baseline MMMU-Pro Score: {mmmu_pro_score:.1f}% "
                  f"(paper: {PAPER_SCORE}%, Δ={delta:+.1f}pp)")

    # ── Phase 3: Calibrate ──
    calibration = None

    if "calibrate" in phases:
        print("\n" + "="*60)
        print("Phase 3: Calibrate steering vectors (MMMU)")
        print("="*60)
        if model_info is None:
            model_info = load_thinking_model()

        mmmu_ds = download_mmmu_splits()
        calibration = calibrate_on_mmmu(model_info, mmmu_ds, max_samples=args.cal_samples)

        print(f"\n[calibration] Top-10 heads by Cohen's d:")
        for i, (li, hi) in enumerate(calibration.top_heads[:10]):
            d = calibration.head_scores.get((li, hi), 0)
            print(f"  {i+1}. Layer {li}, Head {hi}: d={d:.3f}")

    # ── Phase 4: Steered eval ──
    if "steered" in phases:
        print("\n" + "="*60)
        print("Phase 4: Steered eval on MMMU-Pro")
        print("="*60)
        if model_info is None:
            model_info = load_thinking_model()
        if calibration is None:
            cal_dir = Path("checkpoints/calibration/qwen3_vl_2b_thinking_mmmu")
            if cal_dir.exists():
                calibration = CalibrationResult.load(str(cal_dir))
            else:
                print("[ERROR] No calibration found. Run --phase calibrate first.")
                return

        mmmu_pro = download_mmmu_pro()

        for alpha in alphas:
            print(f"\n{'─'*40}")
            print(f"Steering α={alpha}")
            print(f"{'─'*40}")
            steerer = ActivationSteerer(model_info, calibration, steer_layers_start=4)
            steerer.steer(alpha)

            for config_name, dataset in mmmu_pro.items():
                result = eval_mmmu_pro_config(
                    model_info, dataset, config_name,
                    max_samples=args.max_samples,
                )
                all_results[f"steered_a{alpha}_{config_name}"] = result

                # Save intermediate
                safe_result = {k: v for k, v in result.items() if k != "records"}
                with open(output_dir / f"steered_a{alpha}_{config_name.replace(' ', '_')}_{ts}.json", "w") as f:
                    json.dump(safe_result, f, indent=2)

            steerer.cleanup()

            # Compute MMMU-Pro score
            s10 = [v for k, v in all_results.items() if "standard" in k and f"steered_a{alpha}" in k]
            vis = [v for k, v in all_results.items() if "vision" in k and f"steered_a{alpha}" in k]
            if s10 and vis:
                score = (s10[0]["acc"] + vis[0]["acc"]) / 2
                all_results[f"steered_a{alpha}_mmmu_pro_score"] = score
                print(f">>> Steered α={alpha} MMMU-Pro Score: {score:.1f}%")

    # ── Phase 5: Report ──
    if "report" in phases or args.phase == "all":
        print("\n" + "="*60)
        print("Phase 5: Summary Report")
        print("="*60)

        # Save full results (without records for size)
        safe_results = {}
        for k, v in all_results.items():
            if isinstance(v, dict) and "records" in v:
                safe_results[k] = {kk: vv for kk, vv in v.items() if kk != "records"}
            else:
                safe_results[k] = v

        with open(output_dir / f"full_results_{ts}.json", "w") as f:
            json.dump(safe_results, f, indent=2)

        # Print summary table
        print(f"\n{'='*75}")
        print(f"MMMU-Pro Evaluation — Qwen3-VL-2B-Thinking (Official Settings)")
        print(f"  temp=1.0, top_p=0.95, top_k=20, max_tokens=32768")
        print(f"{'='*75}")
        print(f"{'Condition':<30} {'Std-10':>10} {'Vision':>10} {'Score':>10} {'Δ Paper':>10}")
        print("-"*75)

        # Paper reference
        print(f"{'Paper (reference)':<30} {'—':>10} {'—':>10} {PAPER_SCORE:>9.1f}% {'—':>10}")

        # Collect results
        for prefix in ["baseline"] + [f"steered_a{a}" for a in alphas]:
            s10_key = [k for k in all_results if "standard" in k and k.startswith(prefix + "_")]
            vis_key = [k for k in all_results if "vision" in k and k.startswith(prefix + "_")]
            score_key = f"{prefix}_mmmu_pro_score"

            if not s10_key and not vis_key:
                continue

            s10_acc = all_results[s10_key[0]]["acc"] if s10_key else None
            vis_acc = all_results[vis_key[0]]["acc"] if vis_key else None
            score = all_results.get(score_key)

            s10_str = f"{s10_acc:.1f}%" if s10_acc is not None else "—"
            vis_str = f"{vis_acc:.1f}%" if vis_acc is not None else "—"
            score_str = f"{score:.1f}%" if score is not None else "—"
            delta_str = f"{score - PAPER_SCORE:+.1f}pp" if score is not None else "—"

            label = prefix.replace("_", " ").replace("steered a", "steered α=")
            print(f"{label:<30} {s10_str:>10} {vis_str:>10} {score_str:>10} {delta_str:>10}")

        print("="*75)
        print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
