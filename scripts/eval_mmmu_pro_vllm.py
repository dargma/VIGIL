"""
MMMU-Pro Steering Analysis — Qwen3-VL-2B-Thinking (vLLM + HF hybrid)

- vLLM for baseline (α=0): ~3-5x faster inference
- HF for steered conditions (α>0): needs forward hooks

Usage:
    PYTHONUNBUFFERED=1 python -u scripts/eval_mmmu_pro_vllm.py \
        --max-samples 50 --alphas 0,1,3,5,7,10 2>&1 | tee logs/mmmu_pro_vllm.log
"""

import os, sys, json, re, argparse, gc, time
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

# ─── Constants ───────────────────────────────────────────────────────────────

DATA_ROOT = Path("data")
PAPER_SCORE = 42.5
HF_ID = "Qwen/Qwen3-VL-2B-Thinking"

GEN_KWARGS_VLLM = dict(
    max_tokens=4096,
    temperature=1.0,
    top_p=0.95,
    top_k=20,
)

GEN_KWARGS_HF = dict(
    max_new_tokens=2048,  # reduced for speed (avg thinking ~777 words)
    temperature=1.0,
    top_p=0.95,
    top_k=20,
    do_sample=True,
)


# ─── Shared utilities ───────────────────────────────────────────────────────

def split_thinking(text):
    m = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    m = re.search(r'</think>', text)
    if m:
        return text[:m.start()].strip(), text[m.end():].strip()
    return "", text.strip()


def extract_mc_answer(raw_output, num_choices=10):
    _, answer_text = split_thinking(raw_output)
    if not answer_text:
        answer_text = raw_output
    valid = [chr(65 + i) for i in range(num_choices)]
    for pat in [
        r'(?:the\s+)?answer\s+is\s*[:\s]*\(?([A-J])\)?',
        r'(?:answer|option|choice)\s*[:\s]*\(?([A-J])\)?',
        r'\*\*([A-J])\*\*',
        r'^\s*\(?([A-J])\)?[\.\s]*$',
    ]:
        m = re.search(pat, answer_text, re.IGNORECASE | re.MULTILINE)
        if m and m.group(1).upper() in valid:
            return m.group(1).upper()
    found = re.findall(r'\b([A-J])\b', answer_text)
    for letter in reversed(found):
        if letter in valid:
            return letter
    return None


def parse_options(row):
    opts = row.get("options", [])
    if isinstance(opts, str):
        try: opts = json.loads(opts)
        except: opts = eval(opts) if opts.startswith("[") else []
    return opts if isinstance(opts, list) else []


def get_image(row):
    for key in ["image", "image_1", "question_image"]:
        img = row.get(key)
        if isinstance(img, Image.Image):
            return img
    return None


def build_prompt(row, is_vision=False):
    question = row.get("question", "")
    options = parse_options(row)
    if is_vision:
        return "Identify the problem and solve it. Think step by step before answering."
    opts_text = "\n".join(f"{chr(65+j)}. {o}" for j, o in enumerate(options))
    hint = row.get("hint", "") or ""
    hint_text = f"{hint}\n" if hint else ""
    return f"{hint_text}{question}\nOptions:\n{opts_text}\nPlease select the correct answer from the options above."


# ─── Data ────────────────────────────────────────────────────────────────────

def load_mmmu_pro():
    from datasets import load_from_disk
    datasets = {}
    for name, path in [
        ("standard-10", "eval/mmmu_pro_standard10"),
        ("vision", "eval/mmmu_pro_vision"),
    ]:
        full = DATA_ROOT / path
        if full.exists():
            datasets[name] = load_from_disk(str(full))
            print(f"[data] {name}: {len(datasets[name])} samples")
    return datasets


# ─── vLLM Engine ─────────────────────────────────────────────────────────────

def create_vllm_engine():
    """Create vLLM offline engine for Qwen3-VL-2B-Thinking."""
    from vllm import LLM, SamplingParams
    print(f"[vllm] Loading {HF_ID}...")
    llm = LLM(
        model=HF_ID,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
    )
    print(f"[vllm] Engine ready")
    return llm


def eval_vllm(llm, dataset, config_name, max_samples, label="baseline"):
    """Fast evaluation using vLLM offline batched inference."""
    from vllm import SamplingParams
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(HF_ID)
    sampling = SamplingParams(**GEN_KWARGS_VLLM)
    is_vision = "vision" in config_name
    n = min(len(dataset), max_samples)

    # Build all prompts
    prompts = []
    metas = []
    for i in range(n):
        row = dataset[i]
        image = get_image(row)
        gt = str(row.get("answer", "")).strip().upper()
        subject = row.get("subject", "unknown")
        options = parse_options(row)
        prompt_text = build_prompt(row, is_vision)

        # Build chat messages
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": content}]

        # Apply chat template with thinking enabled
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True)

        prompts.append({"prompt": text, "image": image})
        metas.append({"i": i, "gt": gt, "subject": subject,
                      "num_choices": len(options) if options else 10})

    # Process in batches for vLLM
    from vllm import TextPrompt
    from vllm.multimodal import MultiModalDataDict

    batch_size = 8
    all_outputs = []
    t0 = time.time()

    for b_start in range(0, len(prompts), batch_size):
        b_end = min(b_start + batch_size, len(prompts))
        batch_prompts = prompts[b_start:b_end]

        vllm_inputs = []
        for p in batch_prompts:
            inp = {"prompt": p["prompt"]}
            if p["image"] is not None:
                inp["multi_modal_data"] = {"image": p["image"]}
            vllm_inputs.append(inp)

        try:
            outputs = llm.generate(vllm_inputs, sampling_params=sampling)
            for out in outputs:
                raw = out.outputs[0].text
                all_outputs.append(raw)
        except Exception as e:
            print(f"  [vllm batch {b_start}-{b_end}] ERR: {e}")
            for _ in range(b_end - b_start):
                all_outputs.append("")

        done = min(b_end, n)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (n - done) / rate / 60 if rate > 0 else 999
        if done % 8 == 0 or done == n:
            # Compute running accuracy
            correct_so_far = 0
            for j in range(done):
                raw = all_outputs[j]
                pred = extract_mc_answer(raw, metas[j]["num_choices"])
                if pred == metas[j]["gt"]:
                    correct_so_far += 1
            acc = correct_so_far / done * 100
            print(f"  [{label}|{config_name}] {done}/{n} acc={acc:.1f}% "
                  f"{rate:.2f}/s ETA={eta:.1f}m", flush=True)

    # Score all
    correct, total = 0, 0
    think_lens = []
    subject_scores = defaultdict(lambda: {"c": 0, "t": 0})
    records = []

    for j in range(len(all_outputs)):
        raw = all_outputs[j]
        meta = metas[j]
        thinking, answer = split_thinking(raw)
        think_len = len(thinking.split()) if thinking else 0
        think_lens.append(think_len)
        pred = extract_mc_answer(raw, meta["num_choices"])
        ok = (pred == meta["gt"]) if pred else False
        if ok: correct += 1
        total += 1
        subject_scores[meta["subject"]]["t"] += 1
        if ok: subject_scores[meta["subject"]]["c"] += 1
        records.append({
            "i": meta["i"], "gt": meta["gt"], "pred": pred, "ok": ok,
            "think_len": think_len, "subject": meta["subject"],
            "raw": raw[:300],
        })

    acc = correct / total * 100 if total > 0 else 0
    elapsed = (time.time() - t0) / 60
    avg_think = float(np.mean(think_lens)) if think_lens else 0
    print(f"  → [{label}|{config_name}] acc={acc:.1f}% "
          f"think={avg_think:.0f}w ({elapsed:.1f}m)", flush=True)

    per_subj = {}
    for s, sc in sorted(subject_scores.items()):
        sa = sc["c"] / sc["t"] * 100 if sc["t"] > 0 else 0
        per_subj[s] = {"acc": sa, "correct": sc["c"], "total": sc["t"]}

    return {
        "acc": acc, "correct": correct, "total": total,
        "avg_think_words": avg_think, "elapsed_min": elapsed,
        "per_subject": per_subj, "records": records,
    }


# ─── HF Engine (for steering) ───────────────────────────────────────────────

def load_hf_model():
    """Load HF model for steered inference."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    print(f"[hf] Loading {HF_ID}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        HF_ID, torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(HF_ID)
    model.eval()
    info = {
        "model": model, "processor": processor,
        "tokenizer": processor.tokenizer,
        "get_layers_fn": lambda: model.model.language_model.layers,
        "get_lm_head_fn": lambda: model.lm_head,
        "get_norm_fn": lambda: model.model.language_model.norm,
        "num_layers": 28, "num_heads": 16, "num_kv_heads": 8,
        "head_dim": 128, "hidden_size": 2048, "gqa": True,
        "steer_layers_start": 4,
        "device": next(model.parameters()).device,
    }
    print(f"[hf] Loaded. 28 layers, 16Q/8KV")
    return info


def generate_hf(model_info, image, prompt):
    from qwen_vl_utils import process_vision_info
    model = model_info["model"]
    processor = model_info["processor"]
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True)
    if image is not None:
        imgs, _, _ = process_vision_info(messages, return_video_kwargs=True)
    else:
        imgs = None
    inputs = processor(text=[text], images=imgs, return_tensors="pt", padding=True)
    inputs = {k: v.to(model_info["device"]) for k, v in inputs.items()}
    with torch.no_grad():
        gen = model.generate(**inputs, **GEN_KWARGS_HF)
    out = gen[0][inputs["input_ids"].shape[1]:]
    raw = processor.tokenizer.decode(out, skip_special_tokens=False)
    for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
        raw = raw.replace(tok, "")
    return raw.strip()


def eval_hf_steered(model_info, dataset, config_name, max_samples, alpha, label=""):
    """Evaluate with HF model + steering hooks."""
    from src.steerer import ActivationSteerer
    from src.calibrator import CalibrationResult

    # Load calibration
    pope_cal = Path("checkpoints/calibration/qwen3_vl_2b")
    calibration = CalibrationResult.load(str(pope_cal))
    print(f"[cal] {len(calibration.top_heads)} heads from POPE calibration")

    steerer = ActivationSteerer(model_info, calibration, steer_layers_start=4)
    steerer.steer(alpha)
    print(f"[steer] α={alpha} active on {len(calibration.top_heads)} heads")

    is_vision = "vision" in config_name
    n = min(len(dataset), max_samples)
    correct, total = 0, 0
    think_lens = []
    subject_scores = defaultdict(lambda: {"c": 0, "t": 0})
    records = []
    t0 = time.time()

    for i in range(n):
        row = dataset[i]
        try:
            image = get_image(row)
            gt = str(row.get("answer", "")).strip().upper()
            subject = row.get("subject", "unknown")
            options = parse_options(row)
            nc = len(options) if options else 10
            prompt = build_prompt(row, is_vision)

            raw = generate_hf(model_info, image, prompt)
            thinking, answer = split_thinking(raw)
            think_len = len(thinking.split()) if thinking else 0
            think_lens.append(think_len)
            pred = extract_mc_answer(raw, nc)
            ok = (pred == gt) if pred else False
            if ok: correct += 1
            total += 1
            subject_scores[subject]["t"] += 1
            if ok: subject_scores[subject]["c"] += 1
            records.append({
                "i": i, "gt": gt, "pred": pred, "ok": ok,
                "think_len": think_len, "subject": subject,
                "raw": raw[:300],
            })
        except Exception as e:
            print(f"  [{i}] ERR: {e}")
            records.append({"i": i, "error": str(e), "ok": False})
            total += 1

        if (i + 1) % 10 == 0:
            acc = correct / total * 100
            rate = (i + 1) / (time.time() - t0)
            eta = (n - i - 1) / rate / 60
            print(f"  [{label}|{config_name}] {i+1}/{n} acc={acc:.1f}% "
                  f"think={np.mean(think_lens):.0f}w {rate:.3f}/s ETA={eta:.1f}m",
                  flush=True)

    steerer.cleanup()

    acc = correct / total * 100 if total > 0 else 0
    elapsed = (time.time() - t0) / 60
    avg_think = float(np.mean(think_lens)) if think_lens else 0
    print(f"  → [{label}|{config_name}] acc={acc:.1f}% "
          f"think={avg_think:.0f}w ({elapsed:.1f}m)", flush=True)

    per_subj = {}
    for s, sc in sorted(subject_scores.items()):
        sa = sc["c"] / sc["t"] * 100 if sc["t"] > 0 else 0
        per_subj[s] = {"acc": sa, "correct": sc["c"], "total": sc["t"]}

    return {
        "acc": acc, "correct": correct, "total": total,
        "avg_think_words": avg_think, "elapsed_min": elapsed,
        "per_subject": per_subj, "records": records,
    }


# ─── Analysis: Vision Drift ─────────────────────────────────────────────────

def analyze_drift_hf(model_info, calibration, dataset, max_samples=15, alphas=[0, 5]):
    """Record per-position vision head activation norms with/without steering."""
    from src.steerer import ActivationSteerer

    print("\n" + "="*60)
    print("Vision Drift Analysis (per-position activation norms)")
    print("="*60)

    layers = model_info["get_layers_fn"]()
    top_heads = calibration.top_heads[:10]
    num_heads = model_info["num_heads"]
    head_dim = model_info["head_dim"]

    drift_results = {}
    for alpha in alphas:
        steerer = None
        if alpha > 0:
            steerer = ActivationSteerer(model_info, calibration, steer_layers_start=4)
            steerer.steer(alpha)

        trajectories = []
        for idx in range(min(len(dataset), max_samples)):
            row = dataset[idx]
            image = get_image(row)
            if image is None:
                continue

            prompt = build_prompt(row, is_vision=False)
            act_norms = {}

            def make_hook(layer_idx):
                def hook_fn(module, args):
                    x = args[0] if isinstance(args, tuple) else args
                    if x.dim() == 3:
                        B, S, D = x.shape
                        x_heads = x.view(B, S, num_heads, head_dim)
                        for li, hi in top_heads:
                            if li == layer_idx:
                                norms = x_heads[0, :, hi, :].norm(dim=-1).detach().cpu().numpy()
                                act_norms[(li, hi)] = norms
                    return args
                return hook_fn

            hooks = []
            for li, hi in top_heads:
                h = layers[li].self_attn.o_proj.register_forward_pre_hook(make_hook(li))
                hooks.append(h)

            try:
                from qwen_vl_utils import process_vision_info
                processor = model_info["processor"]
                content = [{"type": "image", "image": image},
                          {"type": "text", "text": prompt}]
                messages = [{"role": "user", "content": content}]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=True)
                imgs, _, _ = process_vision_info(messages, return_video_kwargs=True)
                inputs = processor(text=[text], images=imgs, return_tensors="pt", padding=True)
                inputs = {k: v.to(model_info["device"]) for k, v in inputs.items()}
                with torch.no_grad():
                    _ = model_info["model"](**inputs)  # forward only, no generation (faster)
            except Exception as e:
                print(f"  [drift {idx}] ERR: {e}")
            finally:
                for h in hooks:
                    h.remove()

            if act_norms:
                all_n = list(act_norms.values())
                min_len = min(len(n) for n in all_n)
                stacked = np.stack([n[:min_len] for n in all_n], axis=0)
                avg_traj = stacked.mean(axis=0)
                trajectories.append({
                    "idx": idx, "trajectory": avg_traj.tolist(), "seq_len": min_len,
                })

            if (idx + 1) % 5 == 0:
                print(f"  [drift α={alpha}] {idx+1}/{max_samples}", flush=True)

        drift_results[alpha] = trajectories
        if steerer:
            steerer.cleanup()

    return drift_results


# ─── Analysis: Activation Heatmap ───────────────────────────────────────────

def analyze_heatmap_hf(model_info, calibration, dataset, max_samples=10, alpha=5):
    """Layer×Head activation heatmap: unsteered vs steered."""
    from src.steerer import ActivationSteerer

    print("\n" + "="*60)
    print("Activation Heatmap Analysis")
    print("="*60)

    layers = model_info["get_layers_fn"]()
    num_layers = model_info["num_layers"]
    num_heads = model_info["num_heads"]
    head_dim = model_info["head_dim"]

    heatmaps = {}
    for a in [0, alpha]:
        steerer = None
        if a > 0:
            steerer = ActivationSteerer(model_info, calibration, steer_layers_start=4)
            steerer.steer(a)

        accum = np.zeros((num_layers, num_heads))
        counts = np.zeros((num_layers, num_heads))

        for idx in range(min(len(dataset), max_samples)):
            row = dataset[idx]
            image = get_image(row)
            if image is None:
                continue

            prompt = build_prompt(row, is_vision=False)
            act_data = {}

            def make_hook(layer_idx):
                def hook_fn(module, args):
                    x = args[0] if isinstance(args, tuple) else args
                    if x.dim() == 3:
                        B, S, D = x.shape
                        x_heads = x.view(B, S, num_heads, head_dim)
                        for hi in range(num_heads):
                            norm = x_heads[0, :, hi, :].norm(dim=-1).mean().item()
                            act_data[(layer_idx, hi)] = norm
                    return args
                return hook_fn

            hooks = []
            for li in range(num_layers):
                h = layers[li].self_attn.o_proj.register_forward_pre_hook(make_hook(li))
                hooks.append(h)

            try:
                from qwen_vl_utils import process_vision_info
                processor = model_info["processor"]
                content = [{"type": "image", "image": image},
                          {"type": "text", "text": prompt}]
                messages = [{"role": "user", "content": content}]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=True)
                imgs, _, _ = process_vision_info(messages, return_video_kwargs=True)
                inputs = processor(text=[text], images=imgs, return_tensors="pt", padding=True)
                inputs = {k: v.to(model_info["device"]) for k, v in inputs.items()}
                with torch.no_grad():
                    _ = model_info["model"](**inputs)
            except Exception as e:
                print(f"  [heatmap {idx}] ERR: {e}")
            finally:
                for h in hooks:
                    h.remove()

            for (li, hi), norm in act_data.items():
                accum[li, hi] += norm
                counts[li, hi] += 1

            if (idx + 1) % 5 == 0:
                print(f"  [heatmap α={a}] {idx+1}/{max_samples}", flush=True)

        mask = counts > 0
        heatmaps[a] = np.divide(accum, counts, where=mask, out=np.zeros_like(accum))
        if steerer:
            steerer.cleanup()

    heatmaps["diff"] = heatmaps[alpha] - heatmaps[0]
    return heatmaps


# ─── Analysis: Vision-Only vs All Heads ──────────────────────────────────────

def analyze_head_types_hf(model_info, calibration, dataset, max_samples=25, alpha=5):
    """Compare vision-only heads vs all top-K vs random."""
    from src.steerer import ActivationSteerer
    from src.calibrator import CalibrationResult

    print("\n" + "="*60)
    print("Head Type Comparison (Vision-Only vs All vs Random)")
    print("="*60)

    vision_heads = calibration.top_heads[:20]
    all_scored = sorted(calibration.head_scores.items(), key=lambda x: abs(x[1]), reverse=True)
    all_heads = [h for h, s in all_scored[:20]]

    np.random.seed(42)
    random_heads = list(set(
        (int(np.random.randint(4, 28)), int(np.random.randint(0, 16))) for _ in range(30)
    ))[:20]

    conditions = [
        ("baseline", None),
        ("vision_only", vision_heads),
        ("all_top_k", all_heads),
        ("random", random_heads),
    ]

    results = {}
    for cond_name, head_list in conditions:
        steerer = None
        if head_list is not None:
            mod_cal = CalibrationResult.__new__(CalibrationResult)
            mod_cal.top_heads = head_list
            mod_cal.head_scores = calibration.head_scores
            mod_cal.steering_vectors = calibration.steering_vectors
            steerer = ActivationSteerer(model_info, mod_cal, steer_layers_start=4)
            steerer.steer(alpha)

        correct, total = 0, 0
        is_vision = False
        n = min(len(dataset), max_samples)
        t0 = time.time()

        for idx in range(n):
            row = dataset[idx]
            image = get_image(row)
            if image is None:
                continue
            options = parse_options(row)
            nc = len(options) if options else 10
            prompt = build_prompt(row, is_vision)

            try:
                raw = generate_hf(model_info, image, prompt)
                gt = str(row.get("answer", "")).strip().upper()
                pred = extract_mc_answer(raw, nc)
                if pred == gt:
                    correct += 1
                total += 1
            except Exception as e:
                print(f"  [{cond_name}][{idx}] ERR: {e}")
                total += 1

            if (idx + 1) % 10 == 0:
                acc = correct / total * 100 if total > 0 else 0
                print(f"  [{cond_name}] {idx+1}/{n} acc={acc:.1f}%", flush=True)

        if steerer:
            steerer.cleanup()

        acc = correct / total * 100 if total > 0 else 0
        results[cond_name] = {
            "acc": acc, "correct": correct, "total": total,
            "num_heads": len(head_list) if head_list else 0,
        }
        print(f"  → {cond_name}: {acc:.1f}% ({correct}/{total})", flush=True)

    return results


# ─── Plotting ────────────────────────────────────────────────────────────────

def generate_plots(all_results, drift_results, heatmaps, head_type_results, output_dir, alphas):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid")
    output_dir = Path(output_dir)

    # Fig 1: Alpha sweep
    config_accs = defaultdict(dict)
    for key, r in all_results.items():
        if key == "meta":
            continue
        parts = key.split("_", 1)
        alpha = float(parts[0][1:])
        config = parts[1] if len(parts) > 1 else "standard-10"
        config_accs[config][alpha] = r["acc"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for config, alpha_acc in config_accs.items():
        xs = sorted(alpha_acc.keys())
        ys = [alpha_acc[x] for x in xs]
        ax.plot(xs, ys, "o-", linewidth=2, markersize=8, label=config)
    ax.axhline(y=PAPER_SCORE, color="red", linestyle="--", alpha=0.5, label=f"Paper={PAPER_SCORE}%")
    ax.set_xlabel("Steering Strength (α)", fontsize=12)
    ax.set_ylabel("MMMU-Pro Accuracy (%)", fontsize=12)
    ax.set_title("Steering Alpha Sweep — Qwen3-VL-2B-Thinking", fontsize=14)
    ax.legend(fontsize=11)
    plt.tight_layout()
    fig.savefig(output_dir / "fig1_alpha_sweep.png", dpi=150)
    plt.close(fig)
    print(f"  Saved fig1_alpha_sweep.png")

    # Fig 2: Vision drift
    if drift_results:
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ["#4C72B0", "#C44E52", "#55A868", "#8172B2"]
        for i, (alpha, trajs) in enumerate(sorted(drift_results.items())):
            if not trajs:
                continue
            n_bins = 50
            binned = []
            for t in trajs:
                traj = np.array(t["trajectory"])
                if len(traj) < 2:
                    continue
                indices = np.linspace(0, len(traj)-1, n_bins).astype(int)
                binned.append(traj[indices])
            if not binned:
                continue
            binned = np.stack(binned)
            mean = binned.mean(axis=0)
            std = binned.std(axis=0)
            x = np.linspace(0, 100, n_bins)
            ax.plot(x, mean, label=f"α={alpha}", color=colors[i % len(colors)], linewidth=2)
            ax.fill_between(x, mean-std, mean+std, alpha=0.15, color=colors[i % len(colors)])
        ax.set_xlabel("Sequence Position (%)", fontsize=12)
        ax.set_ylabel("Mean Vision Head Activation Norm", fontsize=12)
        ax.set_title("Vision Attention Drift Along Reasoning Chain", fontsize=14)
        ax.legend(fontsize=11)
        plt.tight_layout()
        fig.savefig(output_dir / "fig2_vision_drift.png", dpi=150)
        plt.close(fig)
        print(f"  Saved fig2_vision_drift.png")

    # Fig 3: Heatmap
    if heatmaps:
        keys = [k for k in heatmaps if k != "diff"]
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        for ax, (key, title) in zip(axes, [
            (keys[0] if keys else 0, "Unsteered (α=0)"),
            (keys[1] if len(keys) > 1 else keys[0], "Steered"),
            ("diff", "Difference"),
        ]):
            if key not in heatmaps:
                continue
            data = heatmaps[key]
            if key == "diff":
                vmax = max(abs(data.min()), abs(data.max())) or 1
                im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            else:
                im = ax.imshow(data, aspect="auto", cmap="viridis")
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_xlabel("Head Index", fontsize=11)
            ax.set_ylabel("Layer", fontsize=11)
            ax.set_title(title, fontsize=12)
        plt.suptitle("Layer×Head Activation Heatmap (MMMU-Pro)", fontsize=14)
        plt.tight_layout()
        fig.savefig(output_dir / "fig3_activation_heatmap.png", dpi=150)
        plt.close(fig)
        print(f"  Saved fig3_activation_heatmap.png")

    # Fig 4: Head type comparison
    if head_type_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        names = list(head_type_results.keys())
        accs = [head_type_results[n]["acc"] for n in names]
        colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
        bars = ax.bar(names, accs, color=colors[:len(names)], width=0.6)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   f"{acc:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title("Vision-Only vs All-Heads Steering (MMMU-Pro, α=5)", fontsize=14)
        ax.set_ylim(0, max(accs) * 1.15 if accs else 50)
        plt.tight_layout()
        fig.savefig(output_dir / "fig4_head_type_comparison.png", dpi=150)
        plt.close(fig)
        print(f"  Saved fig4_head_type_comparison.png")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--alphas", type=str, default="0,1,3,5,7,10")
    parser.add_argument("--config", type=str, default="standard-10",
                        choices=["standard-10", "vision", "both"])
    parser.add_argument("--output-dir", type=str, default="lab/reports/mmmu_pro_steering")
    parser.add_argument("--skip-analysis", action="store_true",
                        help="Skip drift/heatmap/head-type analysis")
    parser.add_argument("--analysis-only", action="store_true",
                        help="Skip alpha sweep, run analyses only")
    parser.add_argument("--drift-samples", type=int, default=15)
    parser.add_argument("--heatmap-samples", type=int, default=10)
    parser.add_argument("--head-type-samples", type=int, default=25)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    alphas = [float(a) for a in args.alphas.split(",")]

    mmmu_pro = load_mmmu_pro()
    configs = list(mmmu_pro.keys()) if args.config == "both" else [args.config]

    all_results = {}
    drift_results = {}
    heatmaps = {}
    head_type_results = {}

    if not args.analysis_only:
        # ── Phase 1: Baseline with vLLM (fast) ──
        has_baseline = 0 in alphas
        steered_alphas = [a for a in alphas if a > 0]

        if has_baseline:
            print("\n" + "="*60)
            print("Phase 1: Baseline (α=0) with vLLM")
            print("="*60)
            llm = create_vllm_engine()
            for config_name in configs:
                result = eval_vllm(llm, mmmu_pro[config_name], config_name,
                                   args.max_samples, "α=0")
                key = f"a0.0_{config_name}"
                all_results[key] = result
                safe = {k: v for k, v in result.items() if k != "records"}
                with open(output_dir / f"{key}_{ts}.json", "w") as f:
                    json.dump(safe, f, indent=2)

            # Free vLLM GPU memory
            del llm
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(2)

        # ── Phase 2: Steered with HF ──
        if steered_alphas:
            print("\n" + "="*60)
            print(f"Phase 2: Steered ({steered_alphas}) with HF + hooks")
            print("="*60)
            model_info = load_hf_model()

            pope_cal = Path("checkpoints/calibration/qwen3_vl_2b")
            from src.calibrator import CalibrationResult
            calibration = CalibrationResult.load(str(pope_cal))
            print(f"[cal] {len(calibration.top_heads)} heads")
            for i, (li, hi) in enumerate(calibration.top_heads[:5]):
                d = calibration.head_scores.get((li, hi), 0)
                print(f"  {i+1}. L{li}H{hi}: d={d:.3f}")

            for alpha in steered_alphas:
                for config_name in configs:
                    result = eval_hf_steered(
                        model_info, mmmu_pro[config_name], config_name,
                        args.max_samples, alpha, f"α={alpha}")
                    key = f"a{alpha}_{config_name}"
                    all_results[key] = result
                    safe = {k: v for k, v in result.items() if k != "records"}
                    with open(output_dir / f"{key}_{ts}.json", "w") as f:
                        json.dump(safe, f, indent=2)
    else:
        # Load existing results
        model_info = load_hf_model()
        pope_cal = Path("checkpoints/calibration/qwen3_vl_2b")
        from src.calibrator import CalibrationResult
        calibration = CalibrationResult.load(str(pope_cal))

    # ── Phase 3: Deep Analysis ──
    if not args.skip_analysis:
        # Need HF model for analysis
        if 'model_info' not in dir() or model_info is None:
            model_info = load_hf_model()
        if 'calibration' not in dir() or calibration is None:
            pope_cal = Path("checkpoints/calibration/qwen3_vl_2b")
            from src.calibrator import CalibrationResult
            calibration = CalibrationResult.load(str(pope_cal))

        dataset = mmmu_pro.get("standard-10")

        # Vision drift
        drift_results = analyze_drift_hf(
            model_info, calibration, dataset,
            max_samples=args.drift_samples, alphas=[0, 5])
        with open(output_dir / f"drift_{ts}.json", "w") as f:
            json.dump({str(k): [{"idx": t["idx"], "seq_len": t["seq_len"],
                                  "trajectory": t["trajectory"]} for t in v]
                       for k, v in drift_results.items()}, f, indent=2)

        # Heatmap
        heatmaps = analyze_heatmap_hf(
            model_info, calibration, dataset,
            max_samples=args.heatmap_samples, alpha=5)
        np.savez(output_dir / f"heatmaps_{ts}.npz",
                 **{str(k): v for k, v in heatmaps.items()})

        # Head types
        head_type_results = analyze_head_types_hf(
            model_info, calibration, dataset,
            max_samples=args.head_type_samples, alpha=5)
        with open(output_dir / f"head_types_{ts}.json", "w") as f:
            json.dump(head_type_results, f, indent=2)

    # ── Generate plots ──
    print("\n" + "="*60)
    print("Generating Figures")
    print("="*60)
    generate_plots(all_results, drift_results, heatmaps, head_type_results,
                   output_dir, alphas)

    # ── Summary Report ──
    print("\n" + "="*75)
    print("MMMU-Pro Steering Analysis — Qwen3-VL-2B-Thinking")
    print(f"Paper MMMU-Pro = {PAPER_SCORE}%, {args.max_samples} samples/config")
    print("="*75)

    for config_name in configs:
        print(f"\n--- {config_name} ---")
        print(f"{'Alpha':>7} {'Acc':>8} {'Δ':>8} {'Think':>8} {'Time':>8}")
        print("-" * 45)

        baseline_acc = None
        for alpha in alphas:
            key = f"a{alpha}_{config_name}"
            r = all_results.get(key)
            if r is None:
                continue
            if baseline_acc is None:
                baseline_acc = r["acc"]
            delta = r["acc"] - baseline_acc
            delta_str = f"{delta:+.1f}pp" if alpha > 0 else "—"
            print(f"{alpha:>7.0f} {r['acc']:>7.1f}% {delta_str:>8} "
                  f"{r['avg_think_words']:>7.0f}w {r['elapsed_min']:>6.1f}m")

    if head_type_results:
        print(f"\n--- Head Type Comparison (α=5) ---")
        for name, r in head_type_results.items():
            print(f"  {name:<15}: {r['acc']:.1f}% ({r['correct']}/{r['total']})")

    # GRPO feasibility
    print(f"\n{'='*75}")
    print("GRPO Steering Reward Feasibility")
    print("="*75)
    for config_name in configs:
        base = all_results.get(f"a0.0_{config_name}", all_results.get(f"a0_{config_name}", {}))
        base_acc = base.get("acc", 0)
        best_acc = base_acc
        best_a = 0
        for alpha in alphas:
            key = f"a{alpha}_{config_name}"
            r = all_results.get(key, {})
            if r.get("acc", 0) > best_acc:
                best_acc = r["acc"]
                best_a = alpha
        gain = best_acc - base_acc
        print(f"  [{config_name}] Baseline={base_acc:.1f}%, Best={best_acc:.1f}% (α={best_a}), Gain={gain:+.1f}pp")
        if gain > 2:
            print(f"  → POSITIVE: Steering improves by {gain:.1f}pp → R_vhad justified")
        elif gain > 0:
            print(f"  → MARGINAL: {gain:.1f}pp improvement, R_vhad worth exploring")
        else:
            print(f"  → NEGATIVE: No improvement, consider alternative rewards")

    # Save full results
    safe_all = {}
    for k, v in all_results.items():
        safe_all[k] = {kk: vv for kk, vv in v.items() if kk != "records"}
    safe_all["meta"] = {
        "timestamp": ts, "max_samples": args.max_samples,
        "alphas": alphas, "configs": configs, "paper_score": PAPER_SCORE,
    }
    if head_type_results:
        safe_all["head_type_comparison"] = head_type_results
    with open(output_dir / f"full_analysis_{ts}.json", "w") as f:
        json.dump(safe_all, f, indent=2)
    print(f"\nResults: {output_dir}/full_analysis_{ts}.json")


if __name__ == "__main__":
    main()
