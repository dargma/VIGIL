"""
POPE Steering Analysis — Qwen3-VL-2B-Thinking

Test if steering improves thinking model performance on POPE.
Uses existing POPE calibration (Instruct model, transfers to Thinking).

Pipeline:
  1. vLLM baseline (α=0) — fast batched inference
  2. HF steered (α=1,3,5,7,10) — with o_proj hooks
  3. Vision drift analysis — per-position activation norms
  4. Heatmap — layer×head activation change
  5. Head-type comparison — vision-only vs all heads

Usage:
    PYTHONUNBUFFERED=1 python -u scripts/eval_pope_thinking_steering.py \
        --max-samples 500 --alphas 0,1,3,5,7,10 2>&1 | tee logs/pope_thinking_steering.log
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

HF_ID = "Qwen/Qwen3-VL-2B-Thinking"
POPE_SPLITS = ["random", "popular", "adversarial"]


# ─── POPE Parsing (VLMEvalKit standard) ──────────────────────────────────────

def process_punctuation(text):
    import string
    for p in string.punctuation:
        text = text.replace(p, " ")
    return " ".join(text.split())


def extract_yes_no(raw):
    """VLMEvalKit YOrN_Extraction."""
    _, answer = split_thinking(raw)
    if not answer:
        answer = raw
    answer = answer.strip().lower()
    answer = process_punctuation(answer)
    words = answer.split()
    # Check first few words
    for w in words[:5]:
        if w in ("yes", "true"):
            return "yes"
        if w in ("no", "false"):
            return "no"
    # Fallback: any yes/no in text
    if "yes" in words:
        return "yes"
    if "no" in words:
        return "no"
    return None


def split_thinking(text):
    m = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    m = re.search(r'</think>', text)
    if m:
        return text[:m.start()].strip(), text[m.end():].strip()
    return "", text.strip()


# ─── Data ────────────────────────────────────────────────────────────────────

def load_pope(max_samples=500):
    """Load POPE from HF with images."""
    from datasets import load_dataset
    print(f"[data] Loading POPE (streaming, max {max_samples}/split)...")
    # POPE has single 'test' split with 'category' field
    ds = load_dataset("lmms-lab/POPE", split="test", streaming=True)
    per_split = defaultdict(list)
    samples = []
    for row in ds:
        cat = row.get("category", "unknown")
        if cat not in POPE_SPLITS:
            continue
        if len(per_split[cat]) >= max_samples:
            # Check if all splits are full
            if all(len(per_split[s]) >= max_samples for s in POPE_SPLITS):
                break
            continue
        image = row.get("image")
        if image is None:
            continue
        question = row.get("question", "")
        answer = row.get("answer", "").strip().lower()
        s = {
            "image": image, "question": question, "answer": answer,
            "split": cat, "idx": len(per_split[cat]),
        }
        per_split[cat].append(s)
        samples.append(s)
    for split in POPE_SPLITS:
        print(f"  {split}: {len(per_split[split])} samples")
    print(f"[data] Total: {len(samples)} POPE samples")
    return samples


# ─── vLLM baseline ──────────────────────────────────────────────────────────

def eval_pope_vllm(samples, max_tokens=2048):
    """Fast POPE eval with vLLM."""
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    print(f"\n{'='*60}")
    print(f"POPE Baseline (α=0) with vLLM — {len(samples)} samples")
    print(f"{'='*60}")

    llm = LLM(
        model=HF_ID, dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
    )
    processor = AutoProcessor.from_pretrained(HF_ID)
    sampling = SamplingParams(
        max_tokens=max_tokens, temperature=1.0,
        top_p=0.95, top_k=20,
    )

    # Build prompts
    vllm_inputs = []
    for s in samples:
        prompt = f"{s['question']} Please answer yes or no."
        content = [
            {"type": "image", "image": s["image"]},
            {"type": "text", "text": prompt},
        ]
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True)
        vllm_inputs.append({
            "prompt": text,
            "multi_modal_data": {"image": s["image"]},
        })

    # Batched inference
    t0 = time.time()
    print(f"[vllm] Generating {len(vllm_inputs)} responses...", flush=True)
    outputs = llm.generate(vllm_inputs, sampling_params=sampling)
    elapsed = (time.time() - t0) / 60
    print(f"[vllm] Done in {elapsed:.1f}m ({len(samples)/elapsed:.0f} samples/min)", flush=True)

    # Score
    results = score_pope(samples, [o.outputs[0].text for o in outputs])

    # Free GPU
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)

    return results


def score_pope(samples, raw_outputs):
    """Score POPE results with per-split breakdown."""
    split_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "total": 0})
    think_lens = []
    records = []

    for s, raw in zip(samples, raw_outputs):
        thinking, answer = split_thinking(raw)
        think_len = len(thinking.split()) if thinking else 0
        think_lens.append(think_len)

        pred = extract_yes_no(raw)
        gt = s["answer"]
        split = s["split"]

        ok = (pred == gt) if pred else False
        st = split_stats[split]
        st["total"] += 1
        if gt == "yes" and pred == "yes": st["tp"] += 1
        elif gt == "no" and pred == "no": st["tn"] += 1
        elif gt == "no" and pred == "yes": st["fp"] += 1
        elif gt == "yes" and pred == "no": st["fn"] += 1

        records.append({
            "split": split, "gt": gt, "pred": pred, "ok": ok,
            "think_len": think_len, "raw": raw[:200],
        })

    # Compute metrics per split
    per_split = {}
    for split in POPE_SPLITS:
        st = split_stats[split]
        tp, fp, tn, fn = st["tp"], st["fp"], st["tn"], st["fn"]
        total = st["total"]
        acc = (tp + tn) / total * 100 if total > 0 else 0
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        per_split[split] = {
            "acc": acc, "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "total": total,
        }

    # Overall
    all_tp = sum(s["tp"] for s in split_stats.values())
    all_fp = sum(s["fp"] for s in split_stats.values())
    all_tn = sum(s["tn"] for s in split_stats.values())
    all_fn = sum(s["fn"] for s in split_stats.values())
    all_total = sum(s["total"] for s in split_stats.values())
    overall_acc = (all_tp + all_tn) / all_total * 100 if all_total > 0 else 0
    overall_p = all_tp / (all_tp + all_fp) * 100 if (all_tp + all_fp) > 0 else 0
    overall_r = all_tp / (all_tp + all_fn) * 100 if (all_tp + all_fn) > 0 else 0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0

    avg_think = float(np.mean(think_lens)) if think_lens else 0

    return {
        "overall": {
            "acc": overall_acc, "precision": overall_p, "recall": overall_r,
            "f1": overall_f1, "total": all_total,
        },
        "per_split": per_split,
        "avg_think_words": avg_think,
        "records": records,
    }


# ─── HF Steered Eval ────────────────────────────────────────────────────────

def load_hf_model():
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
        "num_layers": 28, "num_heads": 16, "num_kv_heads": 8,
        "head_dim": 128, "hidden_size": 2048, "gqa": True,
        "steer_layers_start": 4,
        "device": next(model.parameters()).device,
    }
    print(f"[hf] Loaded. 28L, 16Q/8KV")
    return info


def generate_hf(model_info, image, prompt, max_new_tokens=2048):
    from qwen_vl_utils import process_vision_info
    model = model_info["model"]
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
        gen = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=1.0, top_p=0.95, top_k=20, do_sample=True)
    out = gen[0][inputs["input_ids"].shape[1]:]
    raw = processor.tokenizer.decode(out, skip_special_tokens=False)
    for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
        raw = raw.replace(tok, "")
    return raw.strip()


def eval_pope_hf_steered(model_info, samples, alpha, label=""):
    """POPE eval with HF + steering hooks."""
    from src.steerer import ActivationSteerer
    from src.calibrator import CalibrationResult

    pope_cal = Path("checkpoints/calibration/qwen3_vl_2b")
    calibration = CalibrationResult.load(str(pope_cal))

    steerer = ActivationSteerer(model_info, calibration, steer_layers_start=4)
    steerer.steer(alpha)
    print(f"[steer] α={alpha} on {len(calibration.top_heads)} heads")

    raw_outputs = []
    t0 = time.time()
    for i, s in enumerate(samples):
        try:
            prompt = f"{s['question']} Please answer yes or no."
            raw = generate_hf(model_info, s["image"], prompt)
            raw_outputs.append(raw)
        except Exception as e:
            print(f"  [{i}] ERR: {e}")
            raw_outputs.append("")

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(samples) - i - 1) / rate / 60
            # Running accuracy
            partial = score_pope(samples[:i+1], raw_outputs[:i+1])
            acc = partial["overall"]["acc"]
            print(f"  [{label}] {i+1}/{len(samples)} acc={acc:.1f}% "
                  f"{rate:.2f}/s ETA={eta:.1f}m", flush=True)

    steerer.cleanup()
    results = score_pope(samples, raw_outputs)
    elapsed = (time.time() - t0) / 60
    print(f"  → [{label}] acc={results['overall']['acc']:.1f}% "
          f"P={results['overall']['precision']:.1f}% ({elapsed:.1f}m)", flush=True)
    return results


# ─── Vision Drift Analysis ──────────────────────────────────────────────────

def analyze_drift(model_info, samples, max_samples=20, alphas=[0, 5]):
    """Per-position vision head activation norms."""
    from src.steerer import ActivationSteerer
    from src.calibrator import CalibrationResult

    print(f"\n{'='*60}")
    print("Vision Drift Analysis")
    print(f"{'='*60}")

    pope_cal = Path("checkpoints/calibration/qwen3_vl_2b")
    calibration = CalibrationResult.load(str(pope_cal))
    top_heads = calibration.top_heads[:10]
    layers = model_info["get_layers_fn"]()
    num_heads = model_info["num_heads"]
    head_dim = model_info["head_dim"]

    drift_results = {}
    for alpha in alphas:
        steerer = None
        if alpha > 0:
            steerer = ActivationSteerer(model_info, calibration, steer_layers_start=4)
            steerer.steer(alpha)

        trajectories = []
        for idx in range(min(len(samples), max_samples)):
            s = samples[idx]
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
                prompt = f"{s['question']} Please answer yes or no."
                content = [{"type": "image", "image": s["image"]},
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
                print(f"  [drift {idx}] ERR: {e}")
            finally:
                for h in hooks:
                    h.remove()

            if act_norms:
                all_n = list(act_norms.values())
                min_len = min(len(n) for n in all_n)
                stacked = np.stack([n[:min_len] for n in all_n], axis=0)
                trajectories.append({
                    "idx": idx, "trajectory": stacked.mean(axis=0).tolist(),
                    "seq_len": min_len,
                })

            if (idx + 1) % 5 == 0:
                print(f"  [drift α={alpha}] {idx+1}/{max_samples}", flush=True)

        drift_results[alpha] = trajectories
        if steerer:
            steerer.cleanup()

    return drift_results


# ─── Heatmap ─────────────────────────────────────────────────────────────────

def analyze_heatmap(model_info, samples, max_samples=10, alpha=5):
    from src.steerer import ActivationSteerer
    from src.calibrator import CalibrationResult

    print(f"\n{'='*60}")
    print("Activation Heatmap (Layer×Head)")
    print(f"{'='*60}")

    pope_cal = Path("checkpoints/calibration/qwen3_vl_2b")
    calibration = CalibrationResult.load(str(pope_cal))
    layers = model_info["get_layers_fn"]()
    NL, NH, HD = model_info["num_layers"], model_info["num_heads"], model_info["head_dim"]

    heatmaps = {}
    for a in [0, alpha]:
        steerer = None
        if a > 0:
            steerer = ActivationSteerer(model_info, calibration, steer_layers_start=4)
            steerer.steer(a)

        accum = np.zeros((NL, NH))
        counts = np.zeros((NL, NH))

        for idx in range(min(len(samples), max_samples)):
            s = samples[idx]
            act_data = {}

            def make_hook(layer_idx):
                def hook_fn(module, args):
                    x = args[0] if isinstance(args, tuple) else args
                    if x.dim() == 3:
                        B, S, D = x.shape
                        x_heads = x.view(B, S, NH, HD)
                        for hi in range(NH):
                            act_data[(layer_idx, hi)] = x_heads[0, :, hi, :].norm(dim=-1).mean().item()
                    return args
                return hook_fn

            hooks = []
            for li in range(NL):
                h = layers[li].self_attn.o_proj.register_forward_pre_hook(make_hook(li))
                hooks.append(h)

            try:
                from qwen_vl_utils import process_vision_info
                processor = model_info["processor"]
                prompt = f"{s['question']} Please answer yes or no."
                content = [{"type": "image", "image": s["image"]},
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

        mask = counts > 0
        heatmaps[a] = np.divide(accum, counts, where=mask, out=np.zeros_like(accum))
        if steerer:
            steerer.cleanup()
        print(f"  [heatmap α={a}] done", flush=True)

    heatmaps["diff"] = heatmaps[alpha] - heatmaps[0]
    return heatmaps


# ─── Head Type Comparison ────────────────────────────────────────────────────

def analyze_head_types(model_info, samples, max_samples=100, alpha=5):
    from src.steerer import ActivationSteerer
    from src.calibrator import CalibrationResult

    print(f"\n{'='*60}")
    print("Head Type Comparison (vision-only vs all vs random)")
    print(f"{'='*60}")

    pope_cal = Path("checkpoints/calibration/qwen3_vl_2b")
    calibration = CalibrationResult.load(str(pope_cal))

    vision_heads = calibration.top_heads[:20]
    all_scored = sorted(calibration.head_scores.items(), key=lambda x: abs(x[1]), reverse=True)
    all_heads = [h for h, s in all_scored[:20]]

    np.random.seed(42)
    random_heads = list(set(
        (int(np.random.randint(4, 28)), int(np.random.randint(0, 16))) for _ in range(30)
    ))[:20]

    conditions = [
        ("baseline", None),
        ("vision_only_top20", vision_heads),
        ("all_top20", all_heads),
        ("random_20", random_heads),
    ]

    sub_samples = samples[:max_samples]
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

        raw_outputs = []
        t0 = time.time()
        for i, s in enumerate(sub_samples):
            try:
                prompt = f"{s['question']} Please answer yes or no."
                raw = generate_hf(model_info, s["image"], prompt)
                raw_outputs.append(raw)
            except Exception as e:
                raw_outputs.append("")

            if (i + 1) % 25 == 0:
                partial = score_pope(sub_samples[:i+1], raw_outputs[:i+1])
                print(f"  [{cond_name}] {i+1}/{len(sub_samples)} "
                      f"acc={partial['overall']['acc']:.1f}%", flush=True)

        if steerer:
            steerer.cleanup()

        r = score_pope(sub_samples, raw_outputs)
        elapsed = (time.time() - t0) / 60
        results[cond_name] = r
        print(f"  → {cond_name}: acc={r['overall']['acc']:.1f}% "
              f"P={r['overall']['precision']:.1f}% ({elapsed:.1f}m)", flush=True)

    return results


# ─── Plotting ────────────────────────────────────────────────────────────────

def generate_plots(all_results, drift_results, heatmaps, head_type_results, output_dir, alphas):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid")
    output_dir = Path(output_dir)

    # Fig 1: Alpha sweep
    if all_results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        xs = sorted(all_results.keys())
        accs = [all_results[a]["overall"]["acc"] for a in xs]
        precs = [all_results[a]["overall"]["precision"] for a in xs]
        thinks = [all_results[a].get("avg_think_words", 0) for a in xs]

        axes[0].plot(xs, accs, "o-", color="#4C72B0", linewidth=2, markersize=8, label="Accuracy")
        axes[0].plot(xs, precs, "s--", color="#55A868", linewidth=2, markersize=8, label="Precision")
        axes[0].axhline(y=accs[0], color="gray", linestyle=":", alpha=0.5)
        axes[0].set_xlabel("Steering α", fontsize=12)
        axes[0].set_ylabel("Score (%)", fontsize=12)
        axes[0].set_title("POPE Accuracy & Precision vs Steering", fontsize=13)
        axes[0].legend(fontsize=11)

        # Per-split
        for split in POPE_SPLITS:
            split_accs = [all_results[a]["per_split"].get(split, {}).get("acc", 0) for a in xs]
            axes[1].plot(xs, split_accs, "o-", linewidth=2, markersize=6, label=split)
        axes[1].set_xlabel("Steering α", fontsize=12)
        axes[1].set_ylabel("Accuracy (%)", fontsize=12)
        axes[1].set_title("Per-Split Accuracy", fontsize=13)
        axes[1].legend(fontsize=11)

        plt.suptitle("POPE Steering — Qwen3-VL-2B-Thinking", fontsize=14)
        plt.tight_layout()
        fig.savefig(output_dir / "fig1_pope_steering_sweep.png", dpi=150)
        plt.close(fig)
        print(f"  Saved fig1_pope_steering_sweep.png")

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
        ax.set_ylabel("Vision Head Activation Norm", fontsize=12)
        ax.set_title("Vision Drift: Thinking Chain (POPE)", fontsize=14)
        ax.legend(fontsize=11)
        plt.tight_layout()
        fig.savefig(output_dir / "fig2_vision_drift.png", dpi=150)
        plt.close(fig)
        print(f"  Saved fig2_vision_drift.png")

    # Fig 3: Heatmap
    if heatmaps:
        numeric_keys = sorted([k for k in heatmaps if k != "diff"])
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        plot_data = [
            (numeric_keys[0], f"Unsteered (α={numeric_keys[0]})"),
            (numeric_keys[-1], f"Steered (α={numeric_keys[-1]})"),
            ("diff", "Difference"),
        ]
        for ax, (key, title) in zip(axes, plot_data):
            data = heatmaps[key]
            if key == "diff":
                vmax = max(abs(data.min()), abs(data.max())) or 1
                im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            else:
                im = ax.imshow(data, aspect="auto", cmap="viridis")
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_xlabel("Head", fontsize=11)
            ax.set_ylabel("Layer", fontsize=11)
            ax.set_title(title, fontsize=12)
        plt.suptitle("Layer×Head Activation Heatmap (POPE Thinking)", fontsize=14)
        plt.tight_layout()
        fig.savefig(output_dir / "fig3_heatmap.png", dpi=150)
        plt.close(fig)
        print(f"  Saved fig3_heatmap.png")

    # Fig 4: Head type
    if head_type_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        names = list(head_type_results.keys())
        accs = [head_type_results[n]["overall"]["acc"] for n in names]
        precs = [head_type_results[n]["overall"]["precision"] for n in names]
        x_pos = np.arange(len(names))
        w = 0.35
        b1 = ax.bar(x_pos - w/2, accs, w, label="Accuracy", color="#4C72B0")
        b2 = ax.bar(x_pos + w/2, precs, w, label="Precision", color="#55A868")
        for bar, val in zip(b1, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   f"{val:.1f}", ha="center", fontsize=10)
        for bar, val in zip(b2, precs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   f"{val:.1f}", ha="center", fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, fontsize=10)
        ax.set_ylabel("Score (%)", fontsize=12)
        ax.set_title("Vision-Only vs All Heads Steering (POPE, α=5)", fontsize=13)
        ax.legend(fontsize=11)
        plt.tight_layout()
        fig.savefig(output_dir / "fig4_head_types.png", dpi=150)
        plt.close(fig)
        print(f"  Saved fig4_head_types.png")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Samples per POPE split for vLLM baseline (3 splits)")
    parser.add_argument("--steered-samples", type=int, default=100,
                        help="Samples per split for steered HF eval (slower)")
    parser.add_argument("--alphas", type=str, default="0,1,3,5,7,10")
    parser.add_argument("--output-dir", type=str, default="lab/reports/pope_thinking_steering")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip vLLM baseline (use existing)")
    parser.add_argument("--drift-samples", type=int, default=20)
    parser.add_argument("--heatmap-samples", type=int, default=10)
    parser.add_argument("--head-type-samples", type=int, default=100)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    alphas = [float(a) for a in args.alphas.split(",")]

    # Load POPE
    samples = load_pope(args.max_samples)

    all_alpha_results = {}

    # ── Phase 1: Baseline with vLLM ──
    if 0 in alphas and not args.skip_baseline:
        result = eval_pope_vllm(samples)
        all_alpha_results[0] = result
        safe = {k: v for k, v in result.items() if k != "records"}
        with open(output_dir / f"alpha0_{ts}.json", "w") as f:
            json.dump(safe, f, indent=2)
    elif 0 in alphas and args.skip_baseline:
        # Load existing baseline
        existing = sorted(output_dir.glob("alpha0_*.json"))
        if existing:
            with open(existing[-1]) as f:
                all_alpha_results[0] = json.load(f)
            print(f"[baseline] Loaded existing: {existing[-1].name}")

    # ── Phase 2: Steered with HF ──
    steered_alphas = [a for a in alphas if a > 0]
    if steered_alphas:
        # Use fewer samples for steered (HF is ~180x slower than vLLM)
        steered_n = args.steered_samples * 3  # 3 splits
        steered_samples = samples[:steered_n]
        print(f"[steered] Using {len(steered_samples)} samples "
              f"({args.steered_samples}/split) for HF eval")

        model_info = load_hf_model()
        for alpha in steered_alphas:
            result = eval_pope_hf_steered(
                model_info, steered_samples, alpha, f"α={alpha}")
            all_alpha_results[alpha] = result
            safe = {k: v for k, v in result.items() if k != "records"}
            with open(output_dir / f"alpha{alpha}_{ts}.json", "w") as f:
                json.dump(safe, f, indent=2)

    # ── Phase 3: Deep Analysis ──
    drift_results = {}
    heatmaps = {}
    head_type_results = {}

    if not args.skip_analysis:
        if 'model_info' not in locals():
            model_info = load_hf_model()

        drift_results = analyze_drift(
            model_info, samples, max_samples=args.drift_samples, alphas=[0, 5])
        with open(output_dir / f"drift_{ts}.json", "w") as f:
            json.dump({str(k): [{"idx": t["idx"], "seq_len": t["seq_len"],
                                  "trajectory": t["trajectory"]} for t in v]
                       for k, v in drift_results.items()}, f, indent=2)

        heatmaps = analyze_heatmap(
            model_info, samples, max_samples=args.heatmap_samples, alpha=5)
        np.savez(output_dir / f"heatmaps_{ts}.npz",
                 **{str(k): v for k, v in heatmaps.items()})

        head_type_results = analyze_head_types(
            model_info, samples, max_samples=args.head_type_samples, alpha=5)
        with open(output_dir / f"head_types_{ts}.json", "w") as f:
            json.dump({k: {kk: vv for kk, vv in v.items() if kk != "records"}
                       for k, v in head_type_results.items()}, f, indent=2)

    # ── Plots ──
    print(f"\n{'='*60}")
    print("Generating Figures")
    print(f"{'='*60}")
    generate_plots(all_alpha_results, drift_results, heatmaps, head_type_results,
                   output_dir, alphas)

    # ── Summary ──
    print(f"\n{'='*75}")
    print("POPE Steering Analysis — Qwen3-VL-2B-Thinking")
    print(f"{'='*75}")
    print(f"{'Alpha':>7} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Think':>8}")
    print("-" * 55)

    baseline_acc = None
    for alpha in sorted(all_alpha_results.keys()):
        r = all_alpha_results[alpha]
        o = r["overall"]
        if baseline_acc is None:
            baseline_acc = o["acc"]
        delta = o["acc"] - baseline_acc
        delta_str = f"{delta:+.1f}" if alpha > 0 else "—"
        print(f"{alpha:>7.0f} {o['acc']:>7.1f}% {o['precision']:>7.1f}% "
              f"{o['recall']:>7.1f}% {o['f1']:>7.1f}% "
              f"{r.get('avg_think_words', 0):>7.0f}w  {delta_str}")

    print(f"\nPer-split (adversarial):")
    for alpha in sorted(all_alpha_results.keys()):
        adv = all_alpha_results[alpha]["per_split"].get("adversarial", {})
        print(f"  α={alpha}: acc={adv.get('acc', 0):.1f}% prec={adv.get('precision', 0):.1f}%")

    # GRPO feasibility
    base_acc = all_alpha_results.get(0, {}).get("overall", {}).get("acc", 0)
    best_a = max(all_alpha_results, key=lambda a: all_alpha_results[a]["overall"]["acc"])
    best_acc = all_alpha_results[best_a]["overall"]["acc"]
    gain = best_acc - base_acc

    print(f"\n{'='*75}")
    print("GRPO R_vhad Feasibility")
    print(f"{'='*75}")
    print(f"  Baseline: {base_acc:.1f}%  Best: {best_acc:.1f}% (α={best_a})  Gain: {gain:+.1f}pp")
    if gain > 2:
        print(f"  ✓ POSITIVE → R_vhad reward justified")
    elif gain > 0:
        print(f"  ~ MARGINAL → R_vhad worth exploring")
    else:
        print(f"  ✗ NEGATIVE → consider alternatives")

    # Save consolidated
    safe = {}
    for a, r in all_alpha_results.items():
        safe[f"alpha_{a}"] = {k: v for k, v in r.items() if k != "records"}
    safe["meta"] = {"timestamp": ts, "max_samples": args.max_samples, "alphas": alphas}
    if head_type_results:
        safe["head_types"] = {k: {kk: vv for kk, vv in v.items() if kk != "records"}
                              for k, v in head_type_results.items()}
    with open(output_dir / f"full_analysis_{ts}.json", "w") as f:
        json.dump(safe, f, indent=2)
    print(f"\nResults: {output_dir}/full_analysis_{ts}.json")


if __name__ == "__main__":
    main()
