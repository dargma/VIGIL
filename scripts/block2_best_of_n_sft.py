"""
VIGIL Block 2 — Best-of-N + SFT (Rejection Sampling Fine-Tuning).

Instead of GRPO (which plateaus), this:
1. Generates N candidates per training sample
2. Scores each with composite reward (R_correct + lambda*IIG)
3. Selects the best candidate per sample
4. Fine-tunes (SFT) on the curated (prompt, best_answer) pairs

This is equivalent to a single step of implicit KL-regularized RL (ReST/RAFT)
but is far more stable than GRPO and immune to the zero-variance group problem.

Can be iterated: generate with model_k, SFT → model_{k+1}, repeat.
"""

import sys, os, gc, json, time, random, argparse
from pathlib import Path
from datetime import datetime

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description="VIGIL: Best-of-N + SFT")
    # Phase selection
    p.add_argument("--phase", choices=["generate", "sft", "both"], default="both")
    # Generation
    p.add_argument("--n-candidates", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.2)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--train-samples", type=int, default=1000)
    # Reward
    p.add_argument("--lambda-iig", type=float, default=0.0615)
    p.add_argument("--eps-iig", type=float, default=0.1)
    # SFT
    p.add_argument("--sft-epochs", type=int, default=2)
    p.add_argument("--sft-lr", type=float, default=2e-6)
    p.add_argument("--sft-batch-size", type=int, default=1)
    p.add_argument("--sft-grad-accum", type=int, default=8)
    # Eval
    p.add_argument("--eval-samples", type=int, default=200)
    p.add_argument("--blind-test-samples", type=int, default=100)
    # Paths
    p.add_argument("--output-dir", type=str, default="checkpoints/block2_bon")
    p.add_argument("--candidates-file", type=str, default="data/training/bon_candidates.json")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(for_training=False):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    print("[model] Loading Qwen3-VL-2B-Instruct...")
    dtype = torch.bfloat16 if for_training else torch.float16
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct", torch_dtype=dtype, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    if for_training:
        model.train()
        model.gradient_checkpointing_enable()
        for p in model.parameters():
            p.requires_grad = True
    else:
        model.eval()

    device = next(model.parameters()).device
    model_info = {
        "model": model, "processor": processor, "tokenizer": processor.tokenizer,
        "model_type": "qwen3_vl", "device": device,
        "num_layers": 28, "num_heads": 16, "num_kv_heads": 8,
        "head_dim": 128, "hidden_size": 2048,
    }
    return model, processor, model_info


def build_inputs(model_info, question, image):
    processor = model_info["processor"]
    messages = [{"role": "user", "content": []}]
    if image is not None:
        messages[0]["content"].append({"type": "image", "image": image})
    messages[0]["content"].append({"type": "text", "text": question})
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image] if image is not None else None,
                       return_tensors="pt", padding=True)
    return {k: v.to(model_info["device"]) for k, v in inputs.items()}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_training_data(limit, seed):
    from src.data_loader import (
        load_vqav2_train, load_aokvqa_train, load_textvqa_train,
        load_pope, check_image_overlap, remove_overlapping,
    )
    per_src = max(limit // 3, 200)
    vqav2 = load_vqav2_train(limit=per_src * 3)
    aokvqa = load_aokvqa_train(limit=per_src)
    textvqa = load_textvqa_train(limit=per_src)

    pope = load_pope("adversarial", limit=3000)
    overlap = check_image_overlap(aokvqa, pope)
    if overlap: aokvqa = remove_overlapping(aokvqa, overlap)
    overlap2 = check_image_overlap(vqav2, pope)
    if overlap2: vqav2 = remove_overlapping(vqav2, overlap2)

    # Filter: no binary, must have image
    vqav2 = [s for s in vqav2 if s["answer"].strip().lower() not in ("yes","no") and s.get("image")]
    aokvqa = [s for s in aokvqa if s.get("image")]
    textvqa = [s for s in textvqa if s.get("image")]

    n_t = min(int(limit * 0.4), len(textvqa))
    n_m = min(int(limit * 0.3), len(aokvqa))
    n_s = min(limit - n_t - n_m, len(vqav2))

    random.seed(seed)
    random.shuffle(vqav2); random.shuffle(aokvqa); random.shuffle(textvqa)
    combined = textvqa[:n_t] + aokvqa[:n_m] + vqav2[:n_s]
    random.shuffle(combined)

    for s in combined:
        if s["type"] == "mc" and "choices" in s:
            letters = "ABCDEFGH"
            choices_str = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(s["choices"]) if i < len(letters))
            s["prompt_text"] = f"{s['question']}\n{choices_str}\nAnswer with the letter only."
        elif s["type"] == "yesno":
            s["prompt_text"] = f"{s['question']}\nAnswer yes or no only."
        else:
            s["prompt_text"] = f"{s['question']}\nAnswer in a few words."

    print(f"[data] {len(combined)} training samples")
    return combined


# ---------------------------------------------------------------------------
# Phase 1: Generate N candidates per sample and score them
# ---------------------------------------------------------------------------
def phase_generate(args, model, model_info, train_data):
    print(f"\n{'='*60}")
    print(f"Phase 1: Generate {args.n_candidates} candidates per sample")
    print(f"{'='*60}")

    use_iig = args.lambda_iig > 0
    results = []
    total = len(train_data)

    for idx, sample in enumerate(train_data):
        question = sample.get("prompt_text", sample["question"])
        image = sample.get("image")
        gt = sample["answer"]
        qtype = sample["type"]

        if idx % 50 == 0:
            print(f"  [{idx}/{total}] Generating candidates...")

        inputs = build_inputs(model_info, question, image)
        prompt_len = inputs["input_ids"].shape[1]

        # Generate N candidates in one batch
        try:
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_return_sequences=args.n_candidates,
                )
        except torch.cuda.OutOfMemoryError:
            gc.collect(); torch.cuda.empty_cache()
            continue

        candidates = []
        for i in range(output_ids.shape[0]):
            text = model_info["tokenizer"].decode(
                output_ids[i, prompt_len:], skip_special_tokens=True
            )
            candidates.append(text)

        # Score each candidate
        from src.rewards import compute_r_correct
        scores = []
        for cand in candidates:
            r_correct = compute_r_correct(cand, gt, qtype)
            if use_iig and image is not None:
                try:
                    from src.iig import compute_iig, vigil_reward
                    iig = compute_iig(model_info, question, image, cand)
                    score = vigil_reward(r_correct, iig, args.lambda_iig, args.eps_iig)
                except Exception:
                    score = r_correct
            else:
                score = r_correct
            scores.append(score)

        # Select best
        best_idx = int(np.argmax(scores))
        best_score = scores[best_idx]
        best_text = candidates[best_idx]

        results.append({
            "question": question,
            "answer": gt,
            "type": qtype,
            "source": sample.get("source", ""),
            "best_candidate": best_text,
            "best_score": float(best_score),
            "all_scores": [float(s) for s in scores],
            "n_candidates": len(candidates),
        })

        # Free memory
        del output_ids, inputs
        if idx % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # Filter: only keep samples where best candidate scored > 0
    good = [r for r in results if r["best_score"] > 0]
    print(f"\n  Generated: {len(results)} samples")
    print(f"  With score > 0: {len(good)} ({len(good)/max(len(results),1)*100:.1f}%)")
    print(f"  Mean best score: {np.mean([r['best_score'] for r in good]):.3f}")

    # Save candidates
    cand_path = Path(args.candidates_file)
    cand_path.parent.mkdir(parents=True, exist_ok=True)
    # Save without images (just text)
    with open(cand_path, "w") as f:
        json.dump(good, f, indent=2)
    print(f"  Saved to {cand_path}")

    return good


# ---------------------------------------------------------------------------
# Phase 2: SFT on best candidates
# ---------------------------------------------------------------------------
def phase_sft(args, model, model_info, candidates_data, train_data_with_images):
    print(f"\n{'='*60}")
    print(f"Phase 2: SFT on {len(candidates_data)} best candidates")
    print(f"{'='*60}")

    # Build question→image mapping from training data
    q_to_image = {}
    for s in train_data_with_images:
        q = s.get("prompt_text", s["question"])
        q_to_image[q] = s.get("image")

    # Prepare SFT model
    model.train()
    model.gradient_checkpointing_enable()
    for p in model.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.sft_lr, weight_decay=0.01)
    processor = model_info["processor"]
    tokenizer = model_info["tokenizer"]
    device = model_info["device"]

    print(f"  lr={args.sft_lr}, epochs={args.sft_epochs}, batch={args.sft_batch_size}, accum={args.sft_grad_accum}")

    for epoch in range(1, args.sft_epochs + 1):
        random.shuffle(candidates_data)
        total_loss = 0
        n_batches = 0

        for i, entry in enumerate(candidates_data):
            question = entry["question"]
            answer = entry["best_candidate"]
            image = q_to_image.get(question)

            # Build full input with answer for teacher forcing
            messages = [{"role": "user", "content": []}]
            if image is not None:
                messages[0]["content"].append({"type": "image", "image": image})
            messages[0]["content"].append({"type": "text", "text": question})

            # Add assistant response
            messages.append({"role": "assistant", "content": answer})

            try:
                text = processor.apply_chat_template(messages, tokenize=False)
                inputs = processor(
                    text=[text],
                    images=[image] if image is not None else None,
                    return_tensors="pt", padding=True,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Create labels (mask the prompt, only compute loss on answer tokens)
                input_ids = inputs["input_ids"]
                labels = input_ids.clone()

                # Find where the assistant response starts
                # The answer tokens come after the prompt template
                # Simple approach: mask everything before "assistant\n"
                # More robust: use the prompt-only length
                prompt_messages = [messages[0]]
                prompt_text = processor.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True
                )
                prompt_inputs = processor(
                    text=[prompt_text],
                    images=[image] if image is not None else None,
                    return_tensors="pt", padding=True,
                )
                prompt_len = prompt_inputs["input_ids"].shape[1]
                labels[0, :prompt_len] = -100  # mask prompt

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=inputs.get("attention_mask"),
                    pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw"),
                    labels=labels,
                )
                loss = outputs.loss / args.sft_grad_accum
                loss.backward()
                total_loss += loss.item() * args.sft_grad_accum
                n_batches += 1

                if n_batches % args.sft_grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            except torch.cuda.OutOfMemoryError:
                gc.collect(); torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue
            except Exception as e:
                if i < 3:
                    print(f"    SFT error at sample {i}: {e}")
                continue

            if i % 100 == 0 and i > 0:
                avg = total_loss / n_batches
                print(f"    Epoch {epoch} [{i}/{len(candidates_data)}] loss={avg:.4f}")

            # Memory management
            del outputs, loss, inputs, labels
            if i % 20 == 0:
                gc.collect()

        # Flush remaining grads
        if n_batches % args.sft_grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch}: avg_loss={avg_loss:.4f}, batches={n_batches}")

    return model


# ---------------------------------------------------------------------------
# Evaluation (same as v2/v3)
# ---------------------------------------------------------------------------
def extract_yesno(text):
    t = text.strip().lower()
    if t.startswith("yes") or t.startswith("no"):
        return "yes" if t.startswith("yes") else "no"
    t50 = t[:50]
    has_yes = "yes" in t50
    has_no = "no" in t50
    if has_yes and has_no:
        return "yes" if t50.index("yes") < t50.index("no") else "no"
    if has_yes: return "yes"
    if has_no: return "no"
    return ""


def eval_pope(model, model_info, n_samples=200):
    from src.data_loader import load_pope
    model.eval()
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    samples = load_pope("adversarial", limit=n_samples)
    correct = total = yes_count = no_count = 0
    for s in samples:
        try:
            inputs = build_inputs(model_info, s["question"] + "\nAnswer yes or no only.", s.get("image"))
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
            pred = model_info["tokenizer"].decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            yn = extract_yesno(pred)
            if yn == "yes": yes_count += 1
            elif yn == "no": no_count += 1
            if yn == s["answer"].strip().lower(): correct += 1
            total += 1
        except Exception:
            continue
    acc = correct / max(total, 1) * 100
    model.train()
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    return {"acc": acc, "correct": correct, "total": total,
            "yes": yes_count, "no": no_count}


def eval_blind_test(model, model_info, n_samples=100):
    from src.data_loader import load_pope
    model.eval()
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    samples = load_pope("adversarial", limit=n_samples)
    black_img = Image.new("RGB", (448, 448), (0, 0, 0))
    correct_real = correct_blind = total = 0
    for s in samples:
        try:
            prompt = s["question"] + "\nAnswer yes or no only."
            inputs_r = build_inputs(model_info, prompt, s.get("image"))
            with torch.no_grad():
                out_r = model.generate(**inputs_r, max_new_tokens=20, do_sample=False)
            pred_r = model_info["tokenizer"].decode(out_r[0][inputs_r["input_ids"].shape[1]:], skip_special_tokens=True)
            inputs_b = build_inputs(model_info, prompt, black_img)
            with torch.no_grad():
                out_b = model.generate(**inputs_b, max_new_tokens=20, do_sample=False)
            pred_b = model_info["tokenizer"].decode(out_b[0][inputs_b["input_ids"].shape[1]:], skip_special_tokens=True)
            gt = s["answer"].strip().lower()
            if extract_yesno(pred_r) == gt: correct_real += 1
            if extract_yesno(pred_b) == gt: correct_blind += 1
            total += 1
        except Exception:
            continue
    acc_r = correct_real / max(total, 1) * 100
    acc_b = correct_blind / max(total, 1) * 100
    model.train()
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    return {"acc_real": acc_r, "acc_blind": acc_b, "gap": acc_r - acc_b, "total": total}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"VIGIL: Best-of-N + SFT")
    print(f"  phase={args.phase}, N={args.n_candidates}, samples={args.train_samples}")
    print(f"  lambda_iig={args.lambda_iig}, sft_lr={args.sft_lr}, epochs={args.sft_epochs}")
    print(f"{'='*60}")

    # Load data
    print("\n[1] Loading training data...")
    train_data = load_training_data(args.train_samples, args.seed)

    if args.phase in ("generate", "both"):
        # Phase 1: generate + score
        model, processor, model_info = load_model(for_training=False)

        # Baseline eval
        print("\n[2] Baseline evaluation...")
        pope_base = eval_pope(model, model_info, args.eval_samples)
        blind_base = eval_blind_test(model, model_info, args.blind_test_samples)
        print(f"  POPE: {pope_base['acc']:.1f}% | Blind Gap: {blind_base['gap']:.1f}pp")

        candidates_data = phase_generate(args, model, model_info, train_data)

        # Free generation model if doing SFT
        if args.phase == "both":
            del model
            gc.collect()
            torch.cuda.empty_cache()
    else:
        # Load pre-generated candidates
        with open(args.candidates_file) as f:
            candidates_data = json.load(f)
        print(f"  Loaded {len(candidates_data)} candidates from {args.candidates_file}")
        pope_base = blind_base = None

    if args.phase in ("sft", "both"):
        # Phase 2: SFT on best candidates
        model, processor, model_info = load_model(for_training=True)

        if pope_base is None:
            print("\n[2] Baseline evaluation...")
            pope_base = eval_pope(model, model_info, args.eval_samples)
            blind_base = eval_blind_test(model, model_info, args.blind_test_samples)
            print(f"  POPE: {pope_base['acc']:.1f}% | Blind Gap: {blind_base['gap']:.1f}pp")

        model = phase_sft(args, model, model_info, candidates_data, train_data)

        # Post-SFT eval
        print("\n[3] Post-SFT evaluation...")
        pope_post = eval_pope(model, model_info, args.eval_samples)
        blind_post = eval_blind_test(model, model_info, args.blind_test_samples)
        print(f"  POPE: {pope_post['acc']:.1f}% (delta={pope_post['acc']-pope_base['acc']:+.1f}pp)")
        print(f"  Blind: real={blind_post['acc_real']:.1f}%, blind={blind_post['acc_blind']:.1f}%, Gap={blind_post['gap']:.1f}pp (delta={blind_post['gap']-blind_base['gap']:+.1f}pp)")

        # Save model
        model.save_pretrained(str(output_dir / "final"))
        model_info["tokenizer"].save_pretrained(str(output_dir / "final"))
        print(f"  Saved to {output_dir / 'final'}")

        # Save results
        results = {
            "timestamp": ts,
            "config": vars(args),
            "baseline": {"pope": pope_base, "blind": blind_base},
            "post_sft": {"pope": pope_post, "blind": blind_post},
            "n_candidates_used": len(candidates_data),
        }
        results_path = output_dir / f"results_{ts}.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Results: {results_path}")

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
