"""
Head-Level Activation Heatmap Analyzer

Generates attention heatmaps from vision heads overlaid on input images.
Shows where the model "looks" at each token position during thinking.

Usage:
    python scripts/analyze_head_heatmap.py \
        --model-path Qwen/Qwen3-VL-2B-Thinking \
        --num-samples 5
"""

import os, sys, json, re, gc, argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

HF_ID = "Qwen/Qwen3-VL-2B-Thinking"

# Top vision heads from calibration
VISION_HEADS = [
    (5, 0, 9.795), (4, 6, 6.943), (23, 2, 6.602),
    (2, 9, 6.551), (5, 7, 6.353), (11, 2, 6.279),
    (2, 6, 5.440), (8, 3, 5.125), (2, 8, 5.022),
    (4, 1, 4.957), (10, 8, 4.932), (5, 10, 4.552),
]


def split_thinking(text):
    m = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if m: return m.group(1).strip(), text[m.end():].strip()
    m = re.search(r'</think>', text)
    if m: return text[:m.start()].strip(), text[m.end():].strip()
    return "", text.strip()


class AttentionCapturer:
    """Capture attention weights from vision heads during forward pass."""

    def __init__(self, model, vision_heads, num_heads=16, head_dim=128):
        self.vision_heads = vision_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self._attn_weights = {}  # layer -> (batch, num_heads, seq, seq)
        self._o_proj_acts = {}   # layer -> (batch, seq, hidden)
        self._hooks = []

        layers_needed = sorted(set(l for l, h, d in vision_heads))
        layers = model.model.language_model.layers

        for li in layers_needed:
            layer = layers[li]
            o_proj = layer.self_attn.o_proj

            # Capture o_proj input (pre-projection per-head activations)
            def make_oproj_hook(layer_idx):
                def hook_fn(module, args):
                    self._o_proj_acts[layer_idx] = args[0].detach()
                return hook_fn
            handle = o_proj.register_forward_pre_hook(make_oproj_hook(li))
            self._hooks.append(handle)

    def get_per_token_norms(self, prompt_len, n_gen_tokens):
        """Get activation norms for each vision head at each generated token.

        Returns: dict (layer, head) -> (n_gen_tokens,) tensor of norms
        """
        result = {}
        for l, h, d in self.vision_heads:
            acts = self._o_proj_acts.get(l)
            if acts is None:
                continue
            # acts: (1, total_seq, hidden_size)
            reshaped = acts[0].view(-1, self.num_heads, self.head_dim)
            # Extract generated token positions
            gen_acts = reshaped[prompt_len:prompt_len + n_gen_tokens, h, :]
            norms = gen_acts.float().norm(dim=-1)  # (n_gen_tokens,)
            result[(l, h)] = norms
        return result

    def clear(self):
        self._attn_weights.clear()
        self._o_proj_acts.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self.clear()


def analyze_sample(model, processor, sample, capturer, device, out_dir):
    """Analyze one sample: generate, capture head activations, create heatmap."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from qwen_vl_utils import process_vision_info

    question = sample["question"]
    image = sample["image"]
    gt = sample.get("answer", "")
    idx = sample.get("idx", 0)

    # Prepare inputs
    content = [{"type": "image", "image": image},
               {"type": "text", "text": question}]
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True)
    imgs, _, _ = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(text=[text], images=imgs, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    # Generate
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    gen_ids = out[0][prompt_len:]
    raw = processor.tokenizer.decode(gen_ids, skip_special_tokens=False)
    for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
        raw = raw.replace(tok, "")

    thinking, answer = split_thinking(raw)

    # Teacher-force with REAL image to get per-token head activations
    full_ids = torch.cat([inputs["input_ids"], gen_ids.unsqueeze(0)], dim=1)
    fwd_inputs = {k: v for k, v in inputs.items()
                  if k not in ("input_ids", "attention_mask")}
    fwd_inputs["input_ids"] = full_ids
    fwd_inputs["attention_mask"] = torch.ones_like(full_ids)

    capturer.clear()
    with torch.no_grad():
        model(**fwd_inputs)
    real_norms = capturer.get_per_token_norms(prompt_len, len(gen_ids))

    # Teacher-force with BLACK image
    black_image = Image.new('RGB', image.size, (0, 0, 0))
    content_b = [{"type": "image", "image": black_image},
                 {"type": "text", "text": question}]
    messages_b = [{"role": "user", "content": content_b}]
    text_b = processor.apply_chat_template(
        messages_b, tokenize=False, add_generation_prompt=True,
        enable_thinking=True)
    imgs_b, _, _ = process_vision_info(messages_b, return_video_kwargs=True)
    inputs_b = processor(text=[text_b], images=imgs_b, return_tensors="pt", padding=True)
    inputs_b = {k: v.to(device) for k, v in inputs_b.items()}
    bpl = inputs_b["input_ids"].shape[1]

    bf = torch.cat([inputs_b["input_ids"], gen_ids.unsqueeze(0)], dim=1)
    fwd_b = {k: v for k, v in inputs_b.items()
             if k not in ("input_ids", "attention_mask")}
    fwd_b["input_ids"] = bf
    fwd_b["attention_mask"] = torch.ones_like(bf)

    capturer.clear()
    with torch.no_grad():
        model(**fwd_b)
    black_norms = capturer.get_per_token_norms(bpl, len(gen_ids))

    # Compute per-token vision score (activation difference)
    n_tokens = len(gen_ids)
    vision_scores = torch.zeros(n_tokens)
    n_heads = 0

    for key in real_norms:
        if key in black_norms:
            rn = real_norms[key].float().cpu()
            bn = black_norms[key].float().cpu()
            ml = min(rn.shape[0], bn.shape[0], n_tokens)
            diff = torch.abs(rn[:ml] - bn[:ml])
            vision_scores[:ml] += diff
            n_heads += 1

    if n_heads > 0:
        vision_scores /= n_heads

    # Get token texts for annotation
    token_texts = [processor.tokenizer.decode([tid]) for tid in gen_ids.cpu().tolist()]

    # Find think/answer boundary
    think_end = n_tokens
    for i, tt in enumerate(token_texts):
        if "</think>" in tt:
            think_end = i
            break

    # ─── Create visualization ───
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Input image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title(f"Input Image\nQ: {question[:60]}...")
    axes[0, 0].axis('off')

    # Plot 2: Vision score over token positions
    x = np.arange(n_tokens)
    colors = ['blue' if i < think_end else 'green' for i in range(n_tokens)]
    axes[0, 1].bar(x, vision_scores.numpy(), color=colors, alpha=0.7, width=1.0)
    axes[0, 1].axvline(x=think_end, color='red', linestyle='--', label='Think/Answer boundary')
    axes[0, 1].set_xlabel("Token Position")
    axes[0, 1].set_ylabel("Vision Head Δ (real - black)")
    axes[0, 1].set_title("Per-Token Vision Attention Score")
    axes[0, 1].legend()

    # Plot 3: Per-head heatmap
    head_data = []
    head_labels = []
    for l, h, d in VISION_HEADS:
        if (l, h) in real_norms and (l, h) in black_norms:
            rn = real_norms[(l, h)].float().cpu()
            bn = black_norms[(l, h)].float().cpu()
            ml = min(rn.shape[0], bn.shape[0], n_tokens)
            diff = torch.abs(rn[:ml] - bn[:ml]).numpy()
            # Pad to n_tokens if needed
            if len(diff) < n_tokens:
                diff = np.pad(diff, (0, n_tokens - len(diff)))
            head_data.append(diff[:n_tokens])
            head_labels.append(f"L{l}H{h} (d={d:.1f})")

    if head_data:
        heatmap = np.stack(head_data)
        im = axes[1, 0].imshow(heatmap, aspect='auto', cmap='hot',
                                interpolation='nearest')
        axes[1, 0].set_yticks(range(len(head_labels)))
        axes[1, 0].set_yticklabels(head_labels, fontsize=8)
        axes[1, 0].set_xlabel("Token Position")
        axes[1, 0].set_title("Per-Head Vision Δ Heatmap")
        axes[1, 0].axvline(x=think_end, color='cyan', linestyle='--', alpha=0.8)
        plt.colorbar(im, ax=axes[1, 0])

    # Plot 4: Answer + summary
    summary = (
        f"GT: {gt}\n"
        f"Answer: {answer[:100]}\n"
        f"Think length: {think_end} tokens\n"
        f"Mean vision Δ (think): {vision_scores[:think_end].mean():.3f}\n"
        f"Mean vision Δ (answer): {vision_scores[think_end:].mean():.3f}\n"
        f"Drift: {(vision_scores[:min(5,think_end)].mean() - vision_scores[max(0,think_end-5):think_end].mean()):.3f}"
    )
    axes[1, 1].text(0.05, 0.95, summary, transform=axes[1, 1].transAxes,
                     fontsize=10, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].set_title("Analysis Summary")
    axes[1, 1].axis('off')

    plt.suptitle(f"Head-Level Vision Attention Analysis — Sample {idx}", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / f"heatmap_sample_{idx}.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Sample {idx}: think={think_end} tokens, "
          f"mean_Δ_think={vision_scores[:think_end].mean():.3f}, "
          f"mean_Δ_answer={vision_scores[think_end:].mean():.3f}")

    return {
        "idx": idx, "question": question[:200], "gt": gt,
        "answer": answer[:200], "think_len": think_end,
        "mean_delta_think": vision_scores[:think_end].mean().item(),
        "mean_delta_answer": vision_scores[think_end:].mean().item(),
        "total_tokens": n_tokens,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=HF_ID)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--output-dir", type=str,
                        default="lab/reports/head_heatmaps")
    parser.add_argument("--dataset", type=str, default="textvqa",
                        choices=["textvqa", "pope", "ocrbench"])
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    print(f"Loading {args.model_path}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    model.eval()
    processor = AutoProcessor.from_pretrained(HF_ID, trust_remote_code=True)
    device = next(model.parameters()).device

    # Install hooks
    capturer = AttentionCapturer(model, VISION_HEADS)
    print(f"Hooks installed on {len(set(l for l,h,d in VISION_HEADS))} layers")

    # Load data
    from datasets import load_dataset
    if args.dataset == "textvqa":
        ds = load_dataset("lmms-lab/textvqa", split="validation", streaming=True)
        samples = []
        for i, row in enumerate(ds):
            if row.get("image") is None: continue
            answers = row.get("answers", [])
            gt = Counter(answers).most_common(1)[0][0] if answers else ""
            samples.append({
                "image": row["image"], "question": row["question"],
                "answer": gt, "idx": i,
            })
            if len(samples) >= args.num_samples: break
    elif args.dataset == "pope":
        ds = load_dataset("lmms-lab/POPE", split="test", streaming=True)
        samples = []
        for i, row in enumerate(ds):
            samples.append({
                "image": row["image"],
                "question": row["question"] + " Please answer yes or no.",
                "answer": row["answer"].strip().lower(), "idx": i,
            })
            if len(samples) >= args.num_samples: break

    print(f"\nAnalyzing {len(samples)} {args.dataset} samples...\n")

    results = []
    for s in samples:
        try:
            r = analyze_sample(model, processor, s, capturer, device, out_dir)
            results.append(r)
        except Exception as e:
            print(f"  Error on sample {s['idx']}: {e}")

    capturer.remove()

    # Save results
    with open(out_dir / "analysis_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary
    if results:
        mean_think = np.mean([r["mean_delta_think"] for r in results])
        mean_answer = np.mean([r["mean_delta_answer"] for r in results])
        print(f"\n{'='*60}")
        print(f"Mean vision Δ (think phase): {mean_think:.3f}")
        print(f"Mean vision Δ (answer phase): {mean_answer:.3f}")
        print(f"Ratio (answer/think): {mean_answer/mean_think:.2f}")
        print(f"Saved {len(results)} heatmaps to {out_dir}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
