"""
Vision Head Activation Overlay Visualization

Overlays vision head activation heatmaps on original images.
Generates multi-panel figures showing:
1. Per-layer activation overlays on the image
2. Sequence-position evolution (how attention shifts over generation)
3. Key "vision burst" moments identified automatically

Usage:
    python scripts/visualize_heatmap_overlay.py \
        --model-path Qwen/Qwen3-VL-2B-Thinking \
        --num-samples 5 \
        --output-dir lab/reports/heatmap_overlays
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

# Top vision heads from calibration (layer, head, cohen_d)
VISION_HEADS = [
    (5, 0, 9.795), (4, 6, 6.943), (23, 2, 6.602),
    (2, 9, 6.551), (5, 7, 6.353), (11, 2, 6.279),
    (2, 6, 5.440), (8, 3, 5.125), (2, 8, 5.022),
    (4, 1, 4.957), (10, 8, 4.932), (5, 10, 4.552),
]

# Group heads by layer for layer-level analysis
LAYER_GROUPS = {}
for l, h, d in VISION_HEADS:
    LAYER_GROUPS.setdefault(l, []).append((h, d))


def split_thinking(text):
    m = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if m: return m.group(1).strip(), text[m.end():].strip()
    m = re.search(r'</think>', text)
    if m: return text[:m.start()].strip(), text[m.end():].strip()
    return "", text.strip()


class AttentionCapturer:
    """Capture per-head activations + cross-attention to image tokens."""

    def __init__(self, model, vision_heads, num_heads=16, head_dim=128):
        self.vision_heads = vision_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self._o_proj_acts = {}
        self._hooks = []

        layers_needed = sorted(set(l for l, h, d in vision_heads))
        layers = model.model.language_model.layers

        for li in layers_needed:
            layer = layers[li]
            o_proj = layer.self_attn.o_proj

            def make_hook(layer_idx):
                def hook_fn(module, args):
                    self._o_proj_acts[layer_idx] = args[0].detach()
                return hook_fn
            handle = o_proj.register_forward_pre_hook(make_hook(li))
            self._hooks.append(handle)

    def get_per_token_norms(self, prompt_len, n_gen_tokens):
        result = {}
        for l, h, d in self.vision_heads:
            acts = self._o_proj_acts.get(l)
            if acts is None:
                continue
            reshaped = acts[0].view(-1, self.num_heads, self.head_dim)
            gen_acts = reshaped[prompt_len:prompt_len + n_gen_tokens, h, :]
            norms = gen_acts.float().norm(dim=-1)
            result[(l, h)] = norms
        return result

    def get_full_activations(self, prompt_len, n_gen_tokens):
        """Get full activation vectors (not just norms) for richer analysis."""
        result = {}
        for l, h, d in self.vision_heads:
            acts = self._o_proj_acts.get(l)
            if acts is None:
                continue
            reshaped = acts[0].view(-1, self.num_heads, self.head_dim)
            gen_acts = reshaped[prompt_len:prompt_len + n_gen_tokens, h, :]
            result[(l, h)] = gen_acts.float().cpu()
        return result

    def clear(self):
        self._o_proj_acts.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self.clear()


def create_spatial_heatmap(image, vision_delta, image_token_positions, image_grid_shape):
    """Create a spatial heatmap that maps back to image regions.

    For Qwen3-VL, image tokens are arranged in a grid after the vision encoder.
    We map the activation delta at each image token position to the corresponding
    spatial location in the original image.
    """
    h_patches, w_patches = image_grid_shape
    n_patches = h_patches * w_patches

    # Average vision delta across heads for each image token
    img_w, img_h = image.size
    heatmap = np.zeros((h_patches, w_patches))

    for i in range(min(n_patches, len(vision_delta))):
        r, c = i // w_patches, i % w_patches
        if r < h_patches and c < w_patches:
            heatmap[r, c] = vision_delta[i]

    # Resize to image dimensions
    from PIL import Image as PILImage
    hm_img = PILImage.fromarray(heatmap).resize(
        (img_w, img_h), PILImage.BILINEAR)
    return np.array(hm_img)


def find_vision_bursts(vision_scores, threshold_percentile=90):
    """Find token positions where vision attention spikes."""
    if len(vision_scores) == 0:
        return []
    threshold = np.percentile(vision_scores, threshold_percentile)
    bursts = []
    in_burst = False
    start = 0
    for i, v in enumerate(vision_scores):
        if v >= threshold and not in_burst:
            in_burst = True
            start = i
        elif v < threshold and in_burst:
            in_burst = False
            peak_idx = start + np.argmax(vision_scores[start:i])
            bursts.append({
                "start": start, "end": i, "peak": int(peak_idx),
                "peak_value": float(vision_scores[peak_idx]),
                "duration": i - start,
            })
    if in_burst:
        peak_idx = start + np.argmax(vision_scores[start:])
        bursts.append({
            "start": start, "end": len(vision_scores), "peak": int(peak_idx),
            "peak_value": float(vision_scores[peak_idx]),
            "duration": len(vision_scores) - start,
        })
    return sorted(bursts, key=lambda b: b["peak_value"], reverse=True)


def analyze_and_overlay(model, processor, sample, capturer, device, out_dir, idx):
    """Full analysis: generate, capture, overlay, multi-panel visualization."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from qwen_vl_utils import process_vision_info

    question = sample["question"]
    image = sample["image"]
    gt = sample.get("answer", "")

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

    # Find image token positions in the prompt
    input_ids = inputs["input_ids"][0].cpu().tolist()
    # Qwen3-VL uses special image tokens - find their range
    # Image pad token is typically 151655 for Qwen3-VL
    IMAGE_PAD_ID = getattr(processor.tokenizer, 'image_pad_token_id', None)
    if IMAGE_PAD_ID is None:
        # Try common IDs
        for tid in [151655, 151859]:
            if tid in input_ids:
                IMAGE_PAD_ID = tid
                break

    img_token_positions = [i for i, tid in enumerate(input_ids) if tid == IMAGE_PAD_ID] if IMAGE_PAD_ID else []
    n_img_tokens = len(img_token_positions)
    print(f"  Sample {idx}: {n_img_tokens} image tokens in prompt")

    # Generate
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    gen_ids = out[0][prompt_len:]
    raw = processor.tokenizer.decode(gen_ids, skip_special_tokens=False)
    for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
        raw = raw.replace(tok, "")
    thinking, answer = split_thinking(raw)

    # Teacher-force with REAL image
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

    # Compute per-token, per-head delta
    n_tokens = len(gen_ids)
    token_texts = [processor.tokenizer.decode([tid]) for tid in gen_ids.cpu().tolist()]

    think_end = n_tokens
    for i, tt in enumerate(token_texts):
        if "</think>" in tt:
            think_end = i
            break

    # Per-head delta matrix
    head_deltas = {}
    for l, h, d in VISION_HEADS:
        if (l, h) in real_norms and (l, h) in black_norms:
            rn = real_norms[(l, h)].float().cpu()
            bn = black_norms[(l, h)].float().cpu()
            ml = min(rn.shape[0], bn.shape[0], n_tokens)
            diff = torch.abs(rn[:ml] - bn[:ml]).numpy()
            if len(diff) < n_tokens:
                diff = np.pad(diff, (0, n_tokens - len(diff)))
            head_deltas[(l, h)] = diff[:n_tokens]

    # Aggregate vision scores
    vision_scores = np.zeros(n_tokens)
    for key, diff in head_deltas.items():
        vision_scores += diff
    if head_deltas:
        vision_scores /= len(head_deltas)

    # Per-layer aggregated scores
    layer_scores = {}
    for layer_idx, heads in LAYER_GROUPS.items():
        layer_data = []
        for h, d in heads:
            if (layer_idx, h) in head_deltas:
                layer_data.append(head_deltas[(layer_idx, h)])
        if layer_data:
            layer_scores[layer_idx] = np.mean(layer_data, axis=0)

    # Find vision bursts
    bursts = find_vision_bursts(vision_scores, threshold_percentile=85)

    # ─── FIGURE 1: Main overlay + per-head heatmap ───
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # (0,0): Original image
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(image)
    ax_img.set_title(f"Input Image", fontsize=11)
    ax_img.axis('off')

    # (0,1): Image with vision score color bar (average activation strength)
    ax_overlay = fig.add_subplot(gs[0, 1])
    ax_overlay.imshow(image, alpha=0.6)
    # Create a simple overlay showing overall vision engagement
    img_arr = np.array(image)
    h, w = img_arr.shape[:2]
    # Create gradient overlay based on mean vision score
    mean_vs = float(vision_scores.mean()) if len(vision_scores) > 0 else 0
    think_vs = float(vision_scores[:think_end].mean()) if think_end > 0 else 0
    ans_vs = float(vision_scores[think_end:].mean()) if think_end < n_tokens else 0
    overlay_color = plt.cm.hot(min(mean_vs / (vision_scores.max() + 1e-8), 1.0))[:3]
    overlay = np.ones((h, w, 4)) * np.array([*overlay_color, 0.3])
    ax_overlay.imshow(overlay)
    ax_overlay.set_title(f"Mean Vision Δ = {mean_vs:.2f}\n"
                         f"Think: {think_vs:.2f} | Answer: {ans_vs:.2f}", fontsize=10)
    ax_overlay.axis('off')

    # (0,2): Vision score bar chart with burst markers
    ax_bar = fig.add_subplot(gs[0, 2])
    x = np.arange(n_tokens)
    colors = ['#3498db' if i < think_end else '#2ecc71' for i in range(n_tokens)]
    ax_bar.bar(x, vision_scores, color=colors, alpha=0.7, width=1.0)
    ax_bar.axvline(x=think_end, color='red', linestyle='--', linewidth=2, label='Think→Answer')
    for b in bursts[:3]:
        ax_bar.axvspan(b["start"], b["end"], alpha=0.15, color='orange')
        ax_bar.annotate(f'Burst\n{b["peak_value"]:.1f}',
                       xy=(b["peak"], b["peak_value"]),
                       fontsize=7, ha='center', va='bottom', color='red')
    ax_bar.set_xlabel("Token Position")
    ax_bar.set_ylabel("Vision Δ")
    ax_bar.set_title("Per-Token Vision Score + Bursts")
    ax_bar.legend(fontsize=8)

    # (1, 0:3): Per-head heatmap — full width
    ax_heatmap = fig.add_subplot(gs[1, :])
    head_data = []
    head_labels = []
    for l, h, d in VISION_HEADS:
        if (l, h) in head_deltas:
            head_data.append(head_deltas[(l, h)])
            head_labels.append(f"L{l}H{h} (d={d:.1f})")
    if head_data:
        heatmap = np.stack(head_data)
        im = ax_heatmap.imshow(heatmap, aspect='auto', cmap='hot',
                                interpolation='nearest')
        ax_heatmap.set_yticks(range(len(head_labels)))
        ax_heatmap.set_yticklabels(head_labels, fontsize=8)
        ax_heatmap.set_xlabel("Token Position")
        ax_heatmap.set_title("Per-Head Vision Δ Heatmap (12 vision heads)")
        ax_heatmap.axvline(x=think_end, color='cyan', linestyle='--', linewidth=2)
        plt.colorbar(im, ax=ax_heatmap, shrink=0.6)

    # (2, 0): Per-layer aggregated
    ax_layer = fig.add_subplot(gs[2, 0])
    layer_data_list = []
    layer_labels = []
    for li in sorted(layer_scores.keys()):
        layer_data_list.append(layer_scores[li])
        n_heads_in = len(LAYER_GROUPS[li])
        layer_labels.append(f"Layer {li} ({n_heads_in}h)")
    if layer_data_list:
        lhm = np.stack(layer_data_list)
        im2 = ax_layer.imshow(lhm, aspect='auto', cmap='viridis', interpolation='nearest')
        ax_layer.set_yticks(range(len(layer_labels)))
        ax_layer.set_yticklabels(layer_labels, fontsize=9)
        ax_layer.set_xlabel("Token Position")
        ax_layer.set_title("Per-Layer Aggregated Δ")
        ax_layer.axvline(x=think_end, color='cyan', linestyle='--')
        plt.colorbar(im2, ax=ax_layer, shrink=0.8)

    # (2, 1): Think vs Answer phase comparison
    ax_phase = fig.add_subplot(gs[2, 1])
    head_think_means = []
    head_answer_means = []
    for l, h, d in VISION_HEADS:
        if (l, h) in head_deltas:
            hd = head_deltas[(l, h)]
            head_think_means.append(hd[:think_end].mean() if think_end > 0 else 0)
            head_answer_means.append(hd[think_end:].mean() if think_end < n_tokens else 0)
    x_heads = np.arange(len(head_think_means))
    width = 0.35
    ax_phase.bar(x_heads - width/2, head_think_means, width, label='Think', color='#3498db')
    ax_phase.bar(x_heads + width/2, head_answer_means, width, label='Answer', color='#2ecc71')
    ax_phase.set_xticks(x_heads)
    ax_phase.set_xticklabels([f"L{l}H{h}" for l, h, d in VISION_HEADS
                               if (l, h) in head_deltas], rotation=45, fontsize=7)
    ax_phase.set_ylabel("Mean Vision Δ")
    ax_phase.set_title("Think vs Answer Phase per Head")
    ax_phase.legend(fontsize=8)

    # (2, 2): Summary text
    ax_summary = fig.add_subplot(gs[2, 2])
    burst_text = "\n".join([f"  Burst {i+1}: pos {b['peak']}, Δ={b['peak_value']:.2f}"
                           for i, b in enumerate(bursts[:5])])
    summary = (
        f"Question: {question[:80]}\n"
        f"GT: {gt}\n"
        f"Answer: {answer[:80]}\n\n"
        f"Think tokens: {think_end}\n"
        f"Total tokens: {n_tokens}\n"
        f"Mean Δ (think): {think_vs:.3f}\n"
        f"Mean Δ (answer): {ans_vs:.3f}\n"
        f"Drift ratio: {ans_vs/(think_vs+1e-8):.2f}\n\n"
        f"Vision Bursts:\n{burst_text}"
    )
    ax_summary.text(0.05, 0.95, summary, transform=ax_summary.transAxes,
                     fontsize=8, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax_summary.set_title("Analysis Summary")
    ax_summary.axis('off')

    fig.suptitle(f"Vision Head Overlay Analysis — Sample {idx}", fontsize=14, fontweight='bold')
    plt.savefig(out_dir / f"overlay_sample_{idx}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ─── FIGURE 2: Sequence evolution — how attention shifts at key moments ───
    if bursts:
        n_moments = min(6, len(bursts) + 2)  # bursts + start + end of think
        fig2, axes2 = plt.subplots(2, 3, figsize=(18, 12))
        axes2 = axes2.flatten()

        # Select key moments: start, top bursts, think boundary, answer
        moments = []
        moments.append(("Start (t=0)", 0))
        for i, b in enumerate(bursts[:3]):
            moments.append((f"Burst {i+1} (t={b['peak']})", b["peak"]))
        if think_end < n_tokens:
            moments.append((f"Think→Ans (t={think_end})", think_end))
        if n_tokens > 5:
            moments.append((f"End (t={n_tokens-1})", n_tokens - 1))

        for ax_i, (label, t_pos) in enumerate(moments[:6]):
            ax = axes2[ax_i]
            ax.imshow(image, alpha=0.5)

            # Show which heads are active at this position
            head_vals = []
            for l, h, d in VISION_HEADS:
                if (l, h) in head_deltas and t_pos < len(head_deltas[(l, h)]):
                    head_vals.append((f"L{l}H{h}", head_deltas[(l, h)][t_pos]))
            head_vals.sort(key=lambda x: x[1], reverse=True)

            # Color intensity based on total vision delta at this position
            total_delta = vision_scores[t_pos] if t_pos < len(vision_scores) else 0
            intensity = min(total_delta / (vision_scores.max() + 1e-8), 1.0)
            overlay = np.zeros((*np.array(image).shape[:2], 4))
            overlay[:, :, 0] = intensity  # Red channel
            overlay[:, :, 3] = intensity * 0.4  # Alpha
            ax.imshow(overlay)

            # Annotate with top-3 active heads
            head_text = "\n".join([f"{n}: {v:.2f}" for n, v in head_vals[:3]])
            ax.text(0.02, 0.98, head_text, transform=ax.transAxes,
                    fontsize=8, va='top', color='white', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

            token_text = token_texts[t_pos] if t_pos < len(token_texts) else "?"
            ax.set_title(f"{label}\nToken: '{token_text[:20]}'\nΔ={total_delta:.2f}",
                        fontsize=9)
            ax.axis('off')

        # Hide unused axes
        for ax_i in range(len(moments), 6):
            axes2[ax_i].axis('off')

        fig2.suptitle(f"Sequence Evolution — Key Moments — Sample {idx}", fontsize=13)
        plt.tight_layout()
        plt.savefig(out_dir / f"evolution_sample_{idx}.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  Sample {idx}: think={think_end}tok, Δ_think={think_vs:.3f}, "
          f"Δ_answer={ans_vs:.3f}, {len(bursts)} bursts")

    return {
        "idx": idx, "question": question[:200], "gt": gt,
        "answer": answer[:200], "think_len": think_end,
        "total_tokens": n_tokens,
        "mean_delta_think": think_vs,
        "mean_delta_answer": ans_vs,
        "drift_ratio": ans_vs / (think_vs + 1e-8),
        "n_bursts": len(bursts),
        "top_bursts": bursts[:5],
        "per_head_think": {f"L{l}H{h}": float(head_deltas[(l,h)][:think_end].mean())
                          for l, h, d in VISION_HEADS if (l,h) in head_deltas and think_end > 0},
        "per_head_answer": {f"L{l}H{h}": float(head_deltas[(l,h)][think_end:].mean())
                           for l, h, d in VISION_HEADS if (l,h) in head_deltas and think_end < n_tokens},
    }


def create_cross_sample_summary(results, out_dir):
    """Create a summary figure comparing all samples."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if len(results) < 2:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1: Think vs Answer delta across samples
    ax = axes[0, 0]
    samples = [f"S{r['idx']}" for r in results]
    think_deltas = [r['mean_delta_think'] for r in results]
    answer_deltas = [r['mean_delta_answer'] for r in results]
    x = np.arange(len(samples))
    ax.bar(x - 0.2, think_deltas, 0.35, label='Think Phase', color='#3498db')
    ax.bar(x + 0.2, answer_deltas, 0.35, label='Answer Phase', color='#2ecc71')
    ax.set_xticks(x)
    ax.set_xticklabels(samples)
    ax.set_ylabel("Mean Vision Δ")
    ax.set_title("Vision Attention by Phase")
    ax.legend()

    # 2: Think length vs drift ratio
    ax = axes[0, 1]
    think_lens = [r['think_len'] for r in results]
    drift_ratios = [r['drift_ratio'] for r in results]
    ax.scatter(think_lens, drift_ratios, s=100, c='#e74c3c', zorder=5)
    for i, r in enumerate(results):
        ax.annotate(f"S{r['idx']}", (think_lens[i], drift_ratios[i]),
                   fontsize=8, ha='center', va='bottom')
    ax.set_xlabel("Think Length (tokens)")
    ax.set_ylabel("Drift Ratio (answer/think)")
    ax.set_title("Vision Drift vs Think Length")
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='No drift')
    ax.legend()

    # 3: Per-head consistency across samples
    ax = axes[1, 0]
    head_names = sorted(set().union(*[set(r.get('per_head_think', {}).keys()) for r in results]))
    head_means = []
    for hn in head_names:
        vals = [r['per_head_think'].get(hn, 0) for r in results if 'per_head_think' in r]
        head_means.append(np.mean(vals))
    if head_names:
        y_pos = np.arange(len(head_names))
        ax.barh(y_pos, head_means, color='#9b59b6')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(head_names, fontsize=8)
        ax.set_xlabel("Mean Vision Δ (Think Phase)")
        ax.set_title("Head Consistency Across Samples")

    # 4: Burst statistics
    ax = axes[1, 1]
    burst_counts = [r['n_bursts'] for r in results]
    ax.bar(x, burst_counts, color='#f39c12')
    ax.set_xticks(x)
    ax.set_xticklabels(samples)
    ax.set_ylabel("Number of Vision Bursts")
    ax.set_title("Vision Burst Frequency")

    plt.suptitle("Cross-Sample Vision Head Analysis Summary", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / "cross_sample_summary.png", dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=HF_ID)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--output-dir", type=str,
                        default="lab/reports/heatmap_overlays")
    parser.add_argument("--dataset", type=str, default="textvqa",
                        choices=["textvqa", "pope"])
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

    print(f"\nAnalyzing {len(samples)} {args.dataset} samples with overlay...\n")

    results = []
    for s in samples:
        try:
            r = analyze_and_overlay(model, processor, s, capturer, device, out_dir, s["idx"])
            results.append(r)
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            import traceback
            print(f"  Error on sample {s['idx']}: {e}")
            traceback.print_exc()

    capturer.remove()

    # Cross-sample summary
    if len(results) >= 2:
        create_cross_sample_summary(results, out_dir)
        print(f"\nCross-sample summary saved")

    # Save results JSON
    with open(out_dir / "overlay_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    if results:
        mean_think = np.mean([r["mean_delta_think"] for r in results])
        mean_answer = np.mean([r["mean_delta_answer"] for r in results])
        mean_drift = np.mean([r["drift_ratio"] for r in results])
        mean_bursts = np.mean([r["n_bursts"] for r in results])
        print(f"\n{'='*60}")
        print(f"  OVERLAY ANALYSIS COMPLETE")
        print(f"  Samples: {len(results)}")
        print(f"  Mean vision Δ (think): {mean_think:.3f}")
        print(f"  Mean vision Δ (answer): {mean_answer:.3f}")
        print(f"  Mean drift ratio: {mean_drift:.2f}")
        print(f"  Mean bursts per sample: {mean_bursts:.1f}")
        print(f"  Output: {out_dir}/")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
