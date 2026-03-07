"""
VIGIL Visual Analysis 1: Attention Heatmap (post-Block 1 sanity check).

Compares baseline vs IIG-trained model attention over image patches.
Generates 4-panel heatmaps for dramatic samples.
"""
import sys, os, gc, json, time
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from datetime import datetime

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

OUT_DIR = Path("lab/reports/visual_analysis/block1_sanity")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Qwen3-VL special tokens
VISION_START_ID = 151652  # <|vision_start|>
VISION_END_ID = 151653    # <|vision_end|>
IMAGE_PAD_ID = 151655     # <|image_pad|>


def find_visual_token_range(input_ids):
    """Find start/end indices of visual tokens in input_ids (1D tensor)."""
    ids = input_ids.cpu().tolist()

    # Find <|vision_start|> and <|vision_end|>
    start_idx = None
    end_idx = None
    for i, t in enumerate(ids):
        if t == VISION_START_ID:
            start_idx = i + 1  # tokens AFTER vision_start
        if t == VISION_END_ID and start_idx is not None:
            end_idx = i  # tokens BEFORE vision_end
            break

    if start_idx is None or end_idx is None:
        # Fallback: find contiguous image_pad tokens
        pad_indices = [i for i, t in enumerate(ids) if t == IMAGE_PAD_ID]
        if pad_indices:
            start_idx = pad_indices[0]
            end_idx = pad_indices[-1] + 1
        else:
            raise ValueError("No visual tokens found in input_ids")

    return start_idx, end_idx


def extract_visual_attention(model, processor, image, question,
                             max_new_tokens=30):
    """
    Extract attention from generated tokens to visual tokens.

    Strategy: generate first, then do a single forward pass with teacher-forced
    tokens and output_attentions=True to get the full attention matrix.

    Returns dict with attn_map, per_token_attn, generated_text, grid_size.
    """
    from src.model_registry import make_chat_prompt

    model_info = {
        "model": model, "processor": processor,
        "tokenizer": processor.tokenizer, "model_type": "qwen3_vl",
        "device": next(model.parameters()).device,
    }

    inputs = make_chat_prompt(model_info, question, image)
    input_ids = inputs["input_ids"]
    prefix_len = input_ids.shape[1]

    # Find visual token range
    vis_start, vis_end = find_visual_token_range(input_ids[0])
    num_vis = vis_end - vis_start

    # Compute grid dimensions that match num_vis tokens
    # Qwen3-VL uses image_grid_thw but actual visual token count may differ
    # due to merge_size (default 2). Find factors of num_vis close to reported grid.
    if "image_grid_thw" in inputs and inputs["image_grid_thw"] is not None:
        grid_info = inputs["image_grid_thw"][0]  # [T, H, W]
        reported_h = grid_info[1].item()
        reported_w = grid_info[2].item()
        # Try merge_size=2: actual tokens = (H/2) * (W/2)
        for merge in [2, 1, 4]:
            gh = reported_h // merge
            gw = reported_w // merge
            if gh * gw == num_vis:
                grid_h, grid_w = gh, gw
                break
        else:
            # Fallback: find closest factors
            grid_h = grid_w = int(np.ceil(np.sqrt(num_vis)))
    else:
        grid_h = grid_w = int(np.ceil(np.sqrt(num_vis)))

    # Step 1: Generate answer
    with torch.no_grad():
        gen_output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    gen_ids = gen_output[0, prefix_len:]
    gen_text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
    T = gen_ids.shape[0]

    if T == 0:
        return {
            "attn_map": np.zeros((grid_h, grid_w)),
            "per_token_attn": [],
            "generated_text": gen_text,
            "grid_size": (grid_h, grid_w),
            "num_vis_tokens": num_vis,
        }

    # Step 2: Forward pass with teacher-forced tokens to get attention
    full_ids = torch.cat([input_ids, gen_ids.unsqueeze(0)], dim=1)

    # Extend attention_mask
    fwd_kwargs = {}
    for k, v in inputs.items():
        if k == "input_ids":
            continue
        if k == "attention_mask" and v is not None:
            ext = torch.ones(v.shape[0], T, dtype=v.dtype, device=v.device)
            fwd_kwargs[k] = torch.cat([v, ext], dim=1)
        else:
            fwd_kwargs[k] = v

    with torch.no_grad():
        outputs = model(input_ids=full_ids, output_attentions=True, **fwd_kwargs)

    # outputs.attentions: tuple of [batch, heads, seq_len, seq_len] per layer
    attentions = outputs.attentions
    num_layers = len(attentions)
    layers_to_use = min(2, num_layers)  # Only last 2 layers to save memory

    per_token_attn = []
    gen_positions = range(prefix_len, prefix_len + T)

    for t_pos in gen_positions:
        token_vis_attn = []
        for layer_idx in range(num_layers - layers_to_use, num_layers):
            layer_attn = attentions[layer_idx]  # [1, heads, seq, seq]
            # Attention from token at t_pos to visual tokens
            attn_to_vis = layer_attn[0, :, t_pos, vis_start:vis_end]  # [heads, num_vis]
            attn_to_vis = attn_to_vis.float().mean(dim=0).cpu().numpy()  # [num_vis]
            token_vis_attn.append(attn_to_vis)

        avg_attn = np.mean(token_vis_attn, axis=0)
        # Reshape to grid
        if len(avg_attn) >= grid_h * grid_w:
            attn_grid = avg_attn[:grid_h * grid_w].reshape(grid_h, grid_w)
        else:
            attn_grid = np.zeros((grid_h, grid_w))
            attn_grid.flat[:len(avg_attn)] = avg_attn
        per_token_attn.append(attn_grid)

    attn_map = np.mean(per_token_attn, axis=0) if per_token_attn else np.zeros((grid_h, grid_w))

    # Free attention tensors (very large)
    del outputs, attentions
    torch.cuda.empty_cache()

    return {
        "attn_map": attn_map,
        "per_token_attn": per_token_attn,
        "generated_text": gen_text,
        "grid_size": (grid_h, grid_w),
        "num_vis_tokens": num_vis,
    }


def check_yesno(pred, gt):
    """Check yes/no answer match."""
    p = pred.strip().lower()
    g = gt.strip().lower()
    has_yes = "yes" in p
    has_no = "no" in p
    if has_yes and has_no:
        yn = "yes" if p.index("yes") < p.index("no") else "no"
    elif has_yes:
        yn = "yes"
    elif has_no:
        yn = "no"
    else:
        yn = ""
    return yn == g


def compute_attn_kl(attn_a, attn_b):
    """KL divergence between two attention maps (handles different sizes)."""
    a = attn_a.flatten().astype(np.float64)
    b = attn_b.flatten().astype(np.float64)
    # Resize to same length if needed
    min_len = min(len(a), len(b))
    if min_len == 0:
        return 0.0
    a = a[:min_len]
    b = b[:min_len]
    a = a / (a.sum() + 1e-10) + 1e-10
    b = b / (b.sum() + 1e-10) + 1e-10
    return float(np.sum(a * np.log(a / b)))


def plot_4panel_heatmap(image, result_base, result_trained, question, answer_gt,
                        save_path, label_trained="IIG-trained"):
    """4-panel: [Original] [Baseline Attn] [Trained Attn] [Difference]"""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    img_array = np.array(image)

    def resize_attn(attn):
        if attn.max() == 0:
            return np.zeros((img_array.shape[0], img_array.shape[1]))
        normed = (attn / attn.max() * 255).astype(np.uint8)
        return np.array(
            Image.fromarray(normed).resize(
                (img_array.shape[1], img_array.shape[0]), Image.BILINEAR
            )
        ) / 255.0

    # (a) Original
    axes[0].imshow(img_array)
    q_short = question[:60] + "..." if len(question) > 60 else question
    axes[0].set_title(f"Q: {q_short}\nGT: {answer_gt}", fontsize=9)
    axes[0].axis("off")

    # (b) Baseline attention
    attn_b = resize_attn(result_base["attn_map"])
    axes[1].imshow(img_array)
    axes[1].imshow(attn_b, alpha=0.5, cmap="jet")
    axes[1].set_title(f"Baseline\nans: {result_base['generated_text'][:30]}", fontsize=9)
    axes[1].axis("off")

    # (c) Trained attention
    attn_t = resize_attn(result_trained["attn_map"])
    axes[2].imshow(img_array)
    axes[2].imshow(attn_t, alpha=0.5, cmap="jet")
    axes[2].set_title(f"{label_trained}\nans: {result_trained['generated_text'][:30]}", fontsize=9)
    axes[2].axis("off")

    # (d) Difference — resize to same shape if needed
    if attn_t.shape != attn_b.shape:
        target_h, target_w = img_array.shape[0], img_array.shape[1]
        attn_b = np.array(Image.fromarray((attn_b * 255).astype(np.uint8)).resize(
            (target_w, target_h), Image.BILINEAR)) / 255.0
        attn_t = np.array(Image.fromarray((attn_t * 255).astype(np.uint8)).resize(
            (target_w, target_h), Image.BILINEAR)) / 255.0
    diff = attn_t - attn_b
    axes[3].imshow(img_array, alpha=0.3)
    vmax = max(abs(diff.min()), abs(diff.max()), 0.01)
    im = axes[3].imshow(diff, cmap="RdBu_r", alpha=0.7, vmin=-vmax, vmax=vmax)
    axes[3].set_title(f"Delta Attention\n(red={label_trained} up, blue=down)", fontsize=9)
    axes[3].axis("off")
    plt.colorbar(im, ax=axes[3], fraction=0.046)

    fig.suptitle("Visual Attention Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def find_dramatic_samples(model, processor, eval_data, top_n=5):
    """
    Run model on eval_data, find samples where both baseline/iig get correct
    but attention differs most. Since we only have one model at a time,
    this version compares real-image vs black-image attention on same model.
    """
    from src.iig import compute_iig

    model_info = {
        "model": model, "processor": processor,
        "tokenizer": processor.tokenizer, "model_type": "qwen3_vl",
        "device": next(model.parameters()).device,
    }

    candidates = []
    for i, s in enumerate(eval_data):
        if s.get("image") is None:
            continue
        try:
            result = extract_visual_attention(model, processor, s["image"], s["question"])
            correct = check_yesno(result["generated_text"], s["answer"])

            # Compute IIG
            iig_val = compute_iig(model_info, s["question"], s["image"],
                                  result["generated_text"])

            candidates.append({
                "idx": i,
                "sample": s,
                "result": result,
                "correct": correct,
                "iig": iig_val,
                "attn_variance": float(result["attn_map"].var()),
            })

            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(eval_data)}")
        except Exception as e:
            print(f"  Skip {i}: {e}")
            continue

        torch.cuda.empty_cache()

    # Sort by IIG (high IIG = image matters most = most dramatic)
    candidates.sort(key=lambda x: abs(x["iig"]), reverse=True)
    return candidates[:top_n]


def main():
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from src.data_loader import load_pope
    from src.iig import compute_iig

    t0 = time.time()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"{'='*60}\nVIGIL Analysis 1: Attention Heatmap (Block 1)\n{datetime.now().isoformat()}\n{'='*60}")

    # Load model with eager attention (required for output_attentions=True)
    print("\n--- Loading baseline model (eager attention) ---")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct", torch_dtype=torch.float16, device_map="auto",
        attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    device = next(model.parameters()).device

    # Load eval data
    print("\n--- Loading POPE eval data ---")
    pope = load_pope("adversarial", limit=50)
    pope_with_img = [s for s in pope if s.get("image") is not None]
    print(f"  {len(pope_with_img)} samples with images")

    # Find dramatic samples (high IIG = image matters)
    print("\n--- Finding dramatic samples (top 5 by IIG) ---")
    dramatic = find_dramatic_samples(model, processor, pope_with_img, top_n=5)

    if not dramatic:
        print("ERROR: No dramatic samples found.")
        return

    print(f"\n  Found {len(dramatic)} dramatic samples:")
    for d in dramatic:
        print(f"    idx={d['idx']}, IIG={d['iig']:.3f}, correct={d['correct']}, "
              f"Q={d['sample']['question'][:50]}...")

    # For each dramatic sample: compare real-image attention vs black-image attention
    # This shows "what attention looks like when image helps vs doesn't"
    print("\n--- Generating 4-panel heatmaps ---")
    black_img = Image.new("RGB", (448, 448), (0, 0, 0))

    metadata = []
    for rank, d in enumerate(dramatic):
        s = d["sample"]
        print(f"\n  Sample {rank+1}/{len(dramatic)}: Q={s['question'][:50]}...")

        # Real image attention (already computed)
        result_real = d["result"]

        # Black image attention
        try:
            result_black = extract_visual_attention(
                model, processor, black_img, s["question"]
            )
        except Exception as e:
            print(f"    Black image failed: {e}")
            continue

        save_path = OUT_DIR / f"heatmap_sample_{rank+1}.png"
        plot_4panel_heatmap(
            s["image"], result_black, result_real,
            s["question"], s["answer"], save_path,
            label_trained="Real Image"
        )

        # Compute attention KL
        kl = compute_attn_kl(result_real["attn_map"], result_black["attn_map"])

        metadata.append({
            "rank": rank + 1,
            "question": s["question"],
            "answer": s["answer"],
            "iig": d["iig"],
            "correct": d["correct"],
            "generated": result_real["generated_text"],
            "attn_kl": kl,
            "grid_size": result_real["grid_size"],
            "num_vis_tokens": result_real["num_vis_tokens"],
        })

        torch.cuda.empty_cache()

    # Save metadata
    meta_path = OUT_DIR / f"dramatic_samples_{ts}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"\nSaved metadata: {meta_path}")

    # Write summary
    summary_lines = [
        "# Analysis 1: Attention Heatmap — Block 1 Sanity Check",
        f"\nDate: {datetime.now().isoformat()}",
        f"\nSamples analyzed: {len(metadata)}",
        "\n## Results\n",
    ]
    for m in metadata:
        summary_lines.append(
            f"- **Sample {m['rank']}**: Q=\"{m['question'][:60]}...\" "
            f"| IIG={m['iig']:.3f} | KL={m['attn_kl']:.3f} | "
            f"Correct={m['correct']} | Gen=\"{m['generated'][:30]}...\""
        )

    avg_kl = np.mean([m["attn_kl"] for m in metadata]) if metadata else 0
    avg_iig = np.mean([m["iig"] for m in metadata]) if metadata else 0
    summary_lines.extend([
        f"\n## Summary Statistics",
        f"- Mean IIG: {avg_iig:.3f}",
        f"- Mean attention KL (real vs black): {avg_kl:.3f}",
        f"\n## Interpretation",
        f"- {'Attention shifts significantly between real and black images.' if avg_kl > 0.1 else 'Attention patterns are similar — model may not be using image.'}",
        f"- {'High IIG confirms image contributes to answer.' if avg_iig > 1.0 else 'Low IIG — model may be answering from priors.'}",
    ])

    summary_path = OUT_DIR / "summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"Saved summary: {summary_path}")

    elapsed = time.time() - t0
    print(f"\nAnalysis 1 complete in {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
