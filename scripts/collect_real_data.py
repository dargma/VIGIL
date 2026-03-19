#!/usr/bin/env python3
"""
Collect real GPU-based measurements for VIGIL analysis figures.

Two tasks:
  Task A: Vision drift data (10 POPE samples × 2 models × per-token hooks) → ~20 min
  Task B: Full 9K POPE eval with Exp10 checkpoint → ~2-3 hours

Usage:
    python scripts/collect_real_data.py --task drift     # Task A only
    python scripts/collect_real_data.py --task eval       # Task B only
    python scripts/collect_real_data.py --task all        # Both
"""

import os, sys, json, re, gc, argparse, time
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from datetime import datetime
from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

# ─── VLMEvalKit functions (inlined) ────────────────────────────────────────

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

# Top 12 vision heads from calibration (layer, head, cohen_d)
VISION_HEADS = [
    (5, 0, 9.795), (4, 6, 6.943), (23, 2, 6.602),
    (2, 9, 6.551), (5, 7, 6.353), (11, 2, 6.279),
    (2, 6, 5.440), (8, 3, 5.125), (2, 8, 5.022),
    (4, 1, 4.957), (10, 8, 4.932), (5, 10, 4.552),
]


# ─── Model Loading ─────────────────────────────────────────────────────────

def load_model(model_path=None):
    """Load Qwen3-VL-2B model + processor."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    hf_id = model_path or "Qwen/Qwen3-VL-2B-Thinking"
    print(f"[load] Loading: {hf_id}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Thinking")
    model.eval()
    return model, processor


def prepare_input(processor, image, question, blind=False):
    """Prepare model input for a single POPE sample."""
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
    return {k: v.to("cuda") for k, v in inputs.items()}


# ─── POPE Data Loading ─────────────────────────────────────────────────────

def load_pope_samples(max_samples=None):
    """Load POPE samples with images from disk cache."""
    pope_path = "data/eval/pope"
    ds = load_from_disk(pope_path)

    # Flat dataset with 'category' column
    samples = []
    for item in ds:
        answer = item["answer"]
        # Normalize answer to capitalized form
        if answer.lower().strip() in ("yes", "no"):
            answer = answer.strip().capitalize()
        samples.append({
            "question": item["question"],
            "answer": answer,
            "category": item["category"],
            "image": item["image"],
        })
    print(f"[data] Loaded {len(samples)} POPE samples")
    if max_samples and max_samples < len(samples):
        import random
        rng = random.Random(42)
        by_split = {}
        for s in samples:
            by_split.setdefault(s["category"], []).append(s)
        per_split = max(1, max_samples // len(by_split))
        samples = []
        for cat in sorted(by_split.keys()):
            subset = by_split[cat]
            rng.shuffle(subset)
            samples.extend(subset[:per_split])
        print(f"[data] Subsampled to {len(samples)} ({per_split} per split)")
    return samples


# ═══════════════════════════════════════════════════════════════════════════
#  TASK A: Vision Drift Data Collection
# ═══════════════════════════════════════════════════════════════════════════

class ActivationCollector:
    """Collect per-head activations at o_proj during model.generate().

    Hooks fire on every forward pass. During generation with KV cache,
    each generated token triggers one forward pass with seq_len=1.
    The first forward pass processes the entire prompt (seq_len >> 1),
    so we skip it and only record token-by-token generation steps.
    """

    def __init__(self, model, target_layers=None):
        self.model = model
        self.hooks = []
        self.activations = {}  # {(layer, step): np.array of shape [n_heads]}
        self.step_counter = 0
        self.collecting = False
        self._prompt_pass_done = False

        # Get language model layers
        if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
            layers = model.model.language_model.layers
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        else:
            raise ValueError("Cannot find transformer layers")

        if target_layers is None:
            target_layers = set(l for l, h, d in VISION_HEADS)

        for layer_idx in target_layers:
            if layer_idx < len(layers):
                o_proj = layers[layer_idx].self_attn.o_proj
                hook = o_proj.register_forward_pre_hook(
                    self._make_hook(layer_idx)
                )
                self.hooks.append(hook)

    def _make_hook(self, layer_idx):
        def hook_fn(module, args):
            if not self.collecting:
                return
            x = args[0]
            if x.dim() == 3:
                seq_len = x.shape[1]
                if seq_len > 1:
                    # Prompt processing pass — skip but mark as done
                    self._prompt_pass_done = True
                    return
                if not self._prompt_pass_done:
                    return
                # Generation step: [1, 1, num_heads * head_dim]
                hidden = x[0, 0]  # [num_heads * head_dim]
                n_heads = 16  # Qwen3-VL-2B has 16 Q heads
                head_dim = hidden.shape[0] // n_heads
                per_head = hidden.view(n_heads, head_dim)
                norms = per_head.float().norm(dim=-1).cpu().numpy()
                self.activations[(layer_idx, self.step_counter)] = norms
        return hook_fn

    def start(self):
        self.collecting = True
        self.step_counter = 0
        self.activations = {}
        self._prompt_pass_done = False

    def stop(self):
        self.collecting = False

    def increment_step(self):
        """Call after each layer processes the same token to advance step counter."""
        pass  # Step counting is done via unique (layer, step) keys

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


class StepCounter:
    """Callback to increment step counter between generation steps."""

    def __init__(self, collector):
        self.collector = collector
        self._last_layer_count = 0

    def __call__(self):
        # Count how many (layer, step) entries have the current step
        current_step = self.collector.step_counter
        entries_at_step = sum(1 for (l, s) in self.collector.activations
                              if s == current_step)
        if entries_at_step > 0 and entries_at_step >= len(set(
            l for l, h, d in VISION_HEADS)):
            # All target layers have reported for this step
            self.collector.step_counter += 1


def generate_with_hooks(model, processor, image, question, collector,
                        blind=False, max_tokens=100):
    """Generate response with model.generate() while collecting activations."""
    inputs = prepare_input(processor, image, question, blind=blind)
    prompt_len = inputs["input_ids"].shape[1]

    # We need to increment the step counter after each generated token.
    # Use a LogitsProcessor to detect when a new token is being generated.
    from transformers import LogitsProcessor, LogitsProcessorList

    class StepIncrementor(LogitsProcessor):
        def __init__(self, collector):
            self.collector = collector
            self.call_count = 0
        def __call__(self, input_ids, scores):
            if self.call_count > 0:
                # After the first call, each subsequent call = new generation step
                self.collector.step_counter += 1
            self.call_count += 1
            return scores

    step_inc = StepIncrementor(collector)
    collector.start()

    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.01,
            top_p=0.8,
            top_k=20,
            logits_processor=LogitsProcessorList([step_inc]),
        )

    collector.stop()

    out_ids = gen[0][prompt_len:]
    text = processor.tokenizer.decode(out_ids, skip_special_tokens=True).strip()
    return text, dict(collector.activations)


def compute_per_token_deltas(act_real, act_black):
    """Compute per-head activation delta (real - black) at each token position."""
    real_steps = sorted(set(s for (l, s) in act_real.keys()))
    black_steps = sorted(set(s for (l, s) in act_black.keys()))
    common_steps = sorted(set(real_steps) & set(black_steps))

    head_deltas = {}
    for layer_idx, head_idx, cohen_d in VISION_HEADS:
        key = f"L{layer_idx}H{head_idx}"
        deltas = []
        for step in common_steps:
            r = act_real.get((layer_idx, step))
            b = act_black.get((layer_idx, step))
            if r is not None and b is not None and head_idx < len(r):
                deltas.append(abs(float(r[head_idx]) - float(b[head_idx])))
            else:
                deltas.append(0.0)
        head_deltas[key] = deltas

    return head_deltas, common_steps


def collect_drift_data(n_samples=10, max_gen_tokens=100):
    """Collect real per-token activation data for drift analysis figures."""
    print("=" * 60)
    print("  TASK A: Collecting vision drift data")
    print("=" * 60)

    OUT_DIR = PROJECT_ROOT / "lab" / "reports" / "deep_drift_analysis"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    samples = load_pope_samples(max_samples=n_samples * 3)[:n_samples]
    results = {"baseline": [], "exp10": []}

    # Determine best Exp10 checkpoint
    exp10_path = "checkpoints/exp10_sharp_soft/scaled_final/best"
    if not os.path.exists(exp10_path):
        exp10_path = "checkpoints/exp10_sharp_soft/run1/final"
    print(f"[drift] Exp10 checkpoint: {exp10_path}")

    for model_label, model_path in [("baseline", None), ("exp10", exp10_path)]:
        print(f"\n{'─'*40}")
        print(f"  Model: {model_label}")
        print(f"{'─'*40}")

        model, processor = load_model(model_path)
        collector = ActivationCollector(model)

        for i, sample in enumerate(samples):
            t0 = time.time()
            question = sample["question"]
            image = sample["image"]
            if not isinstance(image, Image.Image):
                try:
                    image = Image.open(image["path"]) if isinstance(image, dict) else image
                except Exception:
                    print(f"  [skip] Sample {i}: bad image type {type(image)}")
                    continue

            print(f"  [{i+1}/{n_samples}] {question[:60]}...")

            # Forward with real image
            try:
                text_real, act_real = generate_with_hooks(
                    model, processor, image, question, collector,
                    blind=False, max_tokens=max_gen_tokens
                )
            except Exception as e:
                print(f"    [error] Real image: {e}")
                continue

            # Forward with black image
            try:
                text_black, act_black = generate_with_hooks(
                    model, processor, image, question, collector,
                    blind=True, max_tokens=max_gen_tokens
                )
            except Exception as e:
                print(f"    [error] Black image: {e}")
                continue

            # Compute per-head per-token deltas
            head_deltas, steps = compute_per_token_deltas(act_real, act_black)

            extracted_real = YOrN_Extraction(text_real)
            extracted_black = YOrN_Extraction(text_black)

            result = {
                "index": i,
                "question": question,
                "answer": sample["answer"],
                "category": sample["category"],
                "pred_real": extracted_real,
                "pred_black": extracted_black,
                "text_real": text_real[:200],
                "text_black": text_black[:200],
                "n_tokens": len(steps),
                "head_deltas": head_deltas,  # {head_name: [delta_per_token]}
            }
            results[model_label].append(result)

            elapsed = time.time() - t0
            print(f"    Real: {extracted_real} | Black: {extracted_black} | "
                  f"{len(steps)} tokens | {elapsed:.1f}s")

        # Cleanup model
        collector.remove()
        del model, processor, collector
        gc.collect()
        torch.cuda.empty_cache()

    # Save results
    out_path = OUT_DIR / "real_drift_data.json"
    # Convert numpy arrays to lists for JSON serialization
    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        return obj

    serializable = {}
    for model_label, model_results in results.items():
        serializable[model_label] = []
        for r in model_results:
            sr = dict(r)
            sr["head_deltas"] = {k: [to_serializable(v) for v in vals]
                                  for k, vals in r["head_deltas"].items()}
            serializable[model_label].append(sr)

    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, default=to_serializable)
    print(f"\n[drift] Saved {out_path} ({len(results['baseline'])} baseline, "
          f"{len(results['exp10'])} exp10 samples)")
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  TASK B: Full 9K POPE Eval with Exp10 Checkpoint
# ═══════════════════════════════════════════════════════════════════════════

def run_full_pope_eval(max_samples=None):
    """Run full POPE eval with Exp10 best checkpoint."""
    print("=" * 60)
    print("  TASK B: Full 9K POPE Eval (Exp10)")
    print("=" * 60)

    OUT_DIR = PROJECT_ROOT / "lab" / "reports" / "case_analysis"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine checkpoint
    exp10_path = "checkpoints/exp10_sharp_soft/scaled_final/best"
    if not os.path.exists(exp10_path):
        exp10_path = "checkpoints/exp10_sharp_soft/run1/final"
    print(f"[eval] Exp10 checkpoint: {exp10_path}")

    # Check for resume file
    resume_path = OUT_DIR / "exp10_eval_progress.json"
    records = []
    start_idx = 0
    if resume_path.exists():
        with open(resume_path) as f:
            progress = json.load(f)
        records = progress["records"]
        start_idx = len(records)
        print(f"[eval] Resuming from sample {start_idx}")

    # Load POPE data
    samples = load_pope_samples(max_samples=max_samples)

    # Load model
    model, processor = load_model(exp10_path)

    n_total = len(samples)
    t0 = time.time()
    save_every = 100

    for i in range(start_idx, n_total):
        sample = samples[i]
        question = sample["question"]
        image = sample["image"]

        if not isinstance(image, Image.Image):
            print(f"  [skip] Sample {i}: bad image type")
            records.append({
                "index": i, "question": question,
                "answer": sample["answer"], "category": sample["category"],
                "prediction": "", "extracted": "Unknown",
            })
            continue

        try:
            inputs = prepare_input(processor, image, question)
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=128,
                                     do_sample=False)
            out = gen[0][inputs["input_ids"].shape[1]:]
            prediction = processor.tokenizer.decode(out, skip_special_tokens=True).strip()
        except Exception as e:
            prediction = f"ERROR: {e}"

        extracted = YOrN_Extraction(prediction)
        records.append({
            "index": i,
            "question": question,
            "answer": sample["answer"],
            "category": sample["category"],
            "prediction": prediction,
            "extracted": extracted,
        })

        # Progress
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            speed = (i + 1 - start_idx) / elapsed
            n_correct = sum(1 for r in records if r["extracted"].lower() == r["answer"].lower())
            eta = (n_total - i - 1) / speed if speed > 0 else 0
            print(f"  [{i+1}/{n_total}] acc={n_correct/len(records)*100:.1f}% | "
                  f"{speed:.1f} sps | ETA {eta/60:.0f}m")

        # Save checkpoint
        if (i + 1) % save_every == 0:
            with open(resume_path, "w") as f:
                json.dump({"records": records}, f)

    # Final save
    n_correct = sum(1 for r in records if r["extracted"].lower() == r["answer"].lower())
    final_path = OUT_DIR / f"exp10_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    result = {
        "records": records,
        "metrics": {
            "accuracy": n_correct / len(records) * 100,
            "n_total": len(records),
            "n_correct": n_correct,
        }
    }
    with open(final_path, "w") as f:
        json.dump(result, f, indent=2)

    # Also save as the standard path that case_analysis.py looks for
    standard_path = OUT_DIR / "exp10_real_eval.json"
    with open(standard_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[eval] Final: {n_correct}/{len(records)} = {n_correct/len(records)*100:.1f}%")
    print(f"[eval] Saved: {final_path}")
    print(f"[eval] Saved: {standard_path}")

    # Cleanup resume file
    if resume_path.exists():
        resume_path.unlink()

    # Cleanup
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["drift", "eval", "all"], default="all")
    parser.add_argument("--drift-samples", type=int, default=10)
    parser.add_argument("--drift-tokens", type=int, default=100)
    parser.add_argument("--eval-samples", type=int, default=None,
                       help="Max POPE eval samples (None=all 9K)")
    args = parser.parse_args()

    # Disk safety
    usage = os.popen("df -h . | awk 'NR==2 {print $5}' | tr -d '%'").read().strip()
    if usage and int(usage) >= 95:
        print("DISK 95% — halted")
        return

    if args.task in ("drift", "all"):
        collect_drift_data(n_samples=args.drift_samples, max_gen_tokens=args.drift_tokens)

    if args.task in ("eval", "all"):
        run_full_pope_eval(max_samples=args.eval_samples)


if __name__ == "__main__":
    main()
