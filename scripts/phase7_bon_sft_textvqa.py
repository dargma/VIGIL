"""
Phase 7: BoN+SFT on TextVQA with Head-Level Vision Scoring

Key insight from experiments:
  - GRPO is unstable for open-ended VQA (reward variance → oscillation)
  - BoN+SFT (ReST/RAFT approach) already proven best for VIGIL (+2.5pp POPE, +5pp Gap)
  - Head-level vision Δ provides stronger scoring signal than logit-level LSR
  - Vision-dependent curriculum filtering removes "blind-answerable" questions

Pipeline:
  1. Generate N=8 candidates per TextVQA sample
  2. Score each: composite = w_correct * R_correct + w_vision * R_head_vision
  3. Filter: keep only vision-dependent questions (blind test)
  4. Select top-1 per sample → SFT dataset
  5. Fine-tune 2 epochs on curated data

Usage:
    # Step 1: Generate + Score
    python scripts/phase7_bon_sft_textvqa.py --phase generate \
        --num-samples 500 --group-size 8

    # Step 2: SFT on curated data
    python scripts/phase7_bon_sft_textvqa.py --phase sft \
        --candidates-file checkpoints/phase7_bon/candidates.json

    # All-in-one:
    python scripts/phase7_bon_sft_textvqa.py --phase all \
        --num-samples 500 --group-size 8
"""

import os, sys, gc, json, re, time, random, argparse, string
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from datetime import datetime
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

HF_ID = "Qwen/Qwen3-VL-2B-Thinking"
PROJECT_ROOT = Path(__file__).parent.parent
POPE_SPLITS = ["random", "popular", "adversarial"]


# ══════════════════════════════════════════════════════════════════════
#  Shared utilities (from phase6)
# ══════════════════════════════════════════════════════════════════════

def split_thinking(text):
    m = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if m: return m.group(1).strip(), text[m.end():].strip()
    m = re.search(r'</think>', text)
    if m: return text[:m.start()].strip(), text[m.end():].strip()
    return "", text.strip()

def extract_answer(raw, qtype="short_answer"):
    _, answer = split_thinking(raw)
    if not answer: answer = raw
    text = answer.strip()
    if qtype == "yesno":
        return extract_yes_no(raw)
    return text.split("\n")[0].strip()[:100]

def extract_yes_no(raw):
    _, answer = split_thinking(raw)
    if not answer: answer = raw
    text = answer.strip().lower()
    for p in string.punctuation: text = text.replace(p, " ")
    words = text.split()
    for w in words[:5]:
        if w in ("yes", "true"): return "yes"
        if w in ("no", "false"): return "no"
    if "yes" in words: return "yes"
    if "no" in words: return "no"
    return None

def textvqa_accuracy(pred, answers_all):
    if not pred: return 0.0
    pred_clean = pred.strip().lower()
    for prefix in ["the answer is ", "it says ", "the text reads ",
                   "the brand is ", "it is ", "this is "]:
        if pred_clean.startswith(prefix):
            pred_clean = pred_clean[len(prefix):]
    match_count = sum(1 for a in answers_all if a.strip().lower() == pred_clean)
    if match_count > 0: return min(match_count / 3.0, 1.0)
    match_count = sum(1 for a in answers_all if a.strip().lower() in pred_clean)
    if match_count > 0: return min(match_count / 3.0, 1.0)
    match_count = sum(1 for a in answers_all if pred_clean in a.strip().lower())
    return min(match_count / 3.0, 1.0)


# ══════════════════════════════════════════════════════════════════════
#  Vision Head Hooks (lightweight, from phase6)
# ══════════════════════════════════════════════════════════════════════

DEFAULT_VISION_HEADS = [
    (5, 0, 9.795), (4, 6, 6.943), (23, 2, 6.602),
    (2, 9, 6.551), (5, 7, 6.353), (11, 2, 6.279),
    (2, 6, 5.440), (8, 3, 5.125), (2, 8, 5.022),
    (4, 1, 4.957), (10, 8, 4.932), (5, 10, 4.552),
]

class VisionHeadHooks:
    def __init__(self, model, vision_heads, num_heads=16, head_dim=128):
        self.vision_heads = vision_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self._captured = {}
        self._hooks = []
        self.layers_needed = sorted(set(l for l, h, d in vision_heads))
        layers = model.model.language_model.layers
        for li in self.layers_needed:
            o_proj = layers[li].self_attn.o_proj
            def make_hook(layer_idx):
                def hook_fn(module, args):
                    self._captured[layer_idx] = args[0].detach()
                return hook_fn
            self._hooks.append(o_proj.register_forward_pre_hook(make_hook(li)))

    def get_per_token_head_acts(self, prompt_len, seq_len):
        result = {}
        for l, h, d in self.vision_heads:
            inp = self._captured.get(l)
            if inp is None: continue
            reshaped = inp[0].view(-1, self.num_heads, self.head_dim)
            result[(l, h)] = reshaped[prompt_len:prompt_len + seq_len, h, :]
        return result

    def clear(self): self._captured.clear()
    def remove(self):
        for h in self._hooks: h.remove()
        self._hooks.clear(); self._captured.clear()


# ══════════════════════════════════════════════════════════════════════
#  Data Loading
# ══════════════════════════════════════════════════════════════════════

def load_textvqa_train(limit=1000, seed=42):
    from datasets import load_dataset
    rng = random.Random(seed)
    samples = []
    print("[data] Loading TextVQA train...")
    ds = load_dataset("lmms-lab/textvqa", split="train", streaming=True)
    for row in ds:
        img = row.get("image")
        if img is None: continue
        answers = row.get("answers", [])
        if not answers: continue
        ans = Counter(answers).most_common(1)[0][0]
        samples.append({
            "question": row["question"],
            "answer": ans, "image": img,
            "answers_all": answers,
        })
        if len(samples) >= limit * 2: break
    rng.shuffle(samples)
    samples = samples[:limit]
    print(f"[data] {len(samples)} TextVQA train samples")
    return samples

def load_textvqa_eval(max_samples=200):
    from datasets import load_dataset
    ds = load_dataset("lmms-lab/textvqa", split="validation", streaming=True)
    samples = []
    for row in ds:
        img = row.get("image")
        if img is None: continue
        answers = row.get("answers", [])
        if not answers: continue
        ans = Counter(answers).most_common(1)[0][0]
        samples.append({
            "image": img, "question": row["question"],
            "answer": ans, "answers_all": answers,
        })
        if len(samples) >= max_samples: break
    print(f"[data] {len(samples)} TextVQA eval samples")
    return samples

def load_pope_eval(max_samples=300):
    from datasets import load_dataset
    ds = load_dataset("lmms-lab/POPE", split="test", streaming=True)
    per_split = defaultdict(list)
    per_sample = max_samples // 3
    for row in ds:
        cat = row.get("category", "unknown")
        if cat not in POPE_SPLITS: continue
        if len(per_split[cat]) >= per_sample:
            if all(len(per_split[s]) >= per_sample for s in POPE_SPLITS): break
            continue
        per_split[cat].append({
            "image": row["image"], "question": row["question"],
            "answer": row["answer"].strip().lower(), "category": cat,
        })
    samples = []
    for s in POPE_SPLITS: samples.extend(per_split[s])
    return samples


# ══════════════════════════════════════════════════════════════════════
#  Model + Input Prep
# ══════════════════════════════════════════════════════════════════════

def load_model(model_path=None, for_training=False):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    path = model_path or HF_ID
    print(f"[model] Loading {path} (bfloat16)...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(HF_ID, trust_remote_code=True)
    if for_training:
        model.train()
        model.gradient_checkpointing_enable()
        for p in model.parameters(): p.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable: {trainable:,} params")
    else:
        model.eval()
    return model, processor

def prepare_inputs(processor, image, question, device):
    from qwen_vl_utils import process_vision_info
    content = [{"type": "image", "image": image},
               {"type": "text", "text": question + " Answer briefly."}]
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True)
    imgs, _, _ = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(text=[text], images=imgs, return_tensors="pt", padding=True)
    return {k: v.to(device) for k, v in inputs.items()}


# ══════════════════════════════════════════════════════════════════════
#  Vision Head Scoring
# ══════════════════════════════════════════════════════════════════════

def compute_vision_score(model, processor, sample, candidate_text,
                         candidate_ids, device, hooks):
    """Compute vision head activation Δ for a single candidate.
    Returns float: mean head activation difference (real vs black image)."""
    if candidate_ids.numel() == 0:
        return 0.0

    image = sample["image"]
    question = sample["question"] + " Answer briefly."
    n_cand = candidate_ids.numel()

    # Forward with real image
    real_inputs = prepare_inputs(processor, image, question.replace(" Answer briefly.", ""), device)
    rpl = real_inputs["input_ids"].shape[1]
    rf = torch.cat([real_inputs["input_ids"], candidate_ids.unsqueeze(0)], dim=1)
    real_inputs["input_ids"] = rf
    real_inputs["attention_mask"] = torch.ones_like(rf)
    hooks.clear()
    with torch.no_grad():
        model(**real_inputs)
    real_acts = hooks.get_per_token_head_acts(rpl, n_cand)

    # Forward with black image
    black = Image.new('RGB', image.size, (0, 0, 0))
    black_inputs = prepare_inputs(processor, black, question.replace(" Answer briefly.", ""), device)
    bpl = black_inputs["input_ids"].shape[1]
    bf = torch.cat([black_inputs["input_ids"], candidate_ids.unsqueeze(0)], dim=1)
    black_inputs["input_ids"] = bf
    black_inputs["attention_mask"] = torch.ones_like(bf)
    hooks.clear()
    with torch.no_grad():
        model(**black_inputs)
    black_acts = hooks.get_per_token_head_acts(bpl, n_cand)

    # Mean L2 diff across heads and tokens
    total_score = 0.0
    n_heads = 0
    for (l, h) in real_acts:
        if (l, h) not in black_acts: continue
        ra = real_acts[(l, h)].float()
        ba = black_acts[(l, h)].float()
        ml = min(ra.shape[0], ba.shape[0])
        if ml == 0: continue
        diff = (ra[:ml] - ba[:ml]).norm(dim=-1).mean().item()
        # Weight by Cohen's d
        cohen_d = 1.0
        for vl, vh, vd in hooks.vision_heads:
            if vl == l and vh == h: cohen_d = vd; break
        total_score += diff * cohen_d
        n_heads += 1

    hooks.clear()
    del real_inputs, black_inputs
    return total_score / max(n_heads, 1)


def is_vision_dependent(model, processor, sample, device):
    """Quick blind test: does the model need the image to answer correctly?"""
    question = sample["question"] + " Answer briefly."
    gt = sample["answer"]
    answers_all = sample.get("answers_all", [gt])

    # Answer with real image
    try:
        inputs = prepare_inputs(processor, sample["image"], question, device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        raw = processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                          skip_special_tokens=False)
        for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
            raw = raw.replace(tok, "")
        pred_real = extract_answer(raw)
        acc_real = textvqa_accuracy(pred_real, answers_all)
    except Exception:
        return True  # Assume vision-dependent if eval fails

    # Answer with black image
    try:
        black = Image.new('RGB', sample["image"].size, (0, 0, 0))
        inputs_b = prepare_inputs(processor, black, question, device)
        with torch.no_grad():
            out_b = model.generate(**inputs_b, max_new_tokens=512, do_sample=False)
        raw_b = processor.tokenizer.decode(out_b[0][inputs_b["input_ids"].shape[1]:],
                                            skip_special_tokens=False)
        for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
            raw_b = raw_b.replace(tok, "")
        pred_blind = extract_answer(raw_b)
        acc_blind = textvqa_accuracy(pred_blind, answers_all)
    except Exception:
        return True

    # Vision-dependent: correct with image, wrong without
    return acc_real > acc_blind


# ══════════════════════════════════════════════════════════════════════
#  Phase 1: Generate + Score Candidates
# ══════════════════════════════════════════════════════════════════════

def generate_and_score(model, processor, samples, cfg, hooks, device):
    """Generate N candidates per sample, score by correctness + vision."""
    results = []
    w_correct = cfg.get("w_correct", 0.6)
    w_vision = cfg.get("w_vision", 0.4)
    vision_scale = cfg.get("vision_scale", 10.0)

    total_samples = len(samples)
    for si, sample in enumerate(samples):
        t0 = time.time()
        question = sample["question"] + " Answer briefly."
        gt = sample["answer"]
        answers_all = sample.get("answers_all", [gt])

        inputs = prepare_inputs(processor, sample["image"], question, device)
        prompt_len = inputs["input_ids"].shape[1]

        candidates = []
        for ci in range(cfg["group_size"]):
            try:
                with torch.no_grad():
                    out = model.generate(
                        **inputs, max_new_tokens=cfg.get("max_new_tokens", 512),
                        temperature=cfg.get("temperature", 1.2),
                        top_p=cfg.get("top_p", 0.95), do_sample=True)
                gen_ids = out[0][prompt_len:].clone()
                text = processor.tokenizer.decode(gen_ids, skip_special_tokens=False)
                for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
                    text = text.replace(tok, "")
                text = text.strip()

                pred = extract_answer(text)
                r_correct = textvqa_accuracy(pred, answers_all)

                # Vision score (head-level Δ)
                try:
                    v_score = compute_vision_score(
                        model, processor, sample, text, gen_ids, device, hooks)
                    r_vision = min(v_score / vision_scale, 1.0)
                except Exception:
                    r_vision = 0.0

                composite = w_correct * r_correct + w_vision * r_vision

                candidates.append({
                    "text": text,
                    "prediction": pred,
                    "r_correct": r_correct,
                    "r_vision": r_vision,
                    "composite": composite,
                    "gen_ids": gen_ids.cpu().tolist(),
                })
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); gc.collect()
                continue
            except Exception as e:
                continue

        if not candidates:
            continue

        # Select best by composite score
        best = max(candidates, key=lambda c: c["composite"])
        elapsed = time.time() - t0

        entry = {
            "question": sample["question"],
            "answer": gt,
            "answers_all": answers_all,
            "best_text": best["text"],
            "best_correct": best["r_correct"],
            "best_vision": best["r_vision"],
            "best_composite": best["composite"],
            "n_candidates": len(candidates),
            "mean_correct": np.mean([c["r_correct"] for c in candidates]),
            "mean_vision": np.mean([c["r_vision"] for c in candidates]),
            "best_gen_ids": best["gen_ids"],
        }
        results.append(entry)

        if (si + 1) % 10 == 0 or si == 0:
            hit_rate = np.mean([r["best_correct"] > 0 for r in results])
            mean_comp = np.mean([r["best_composite"] for r in results])
            print(f"  [{si+1}/{total_samples}] hit={hit_rate:.1%} "
                  f"composite={mean_comp:.3f} "
                  f"correct={entry['best_correct']:.2f} "
                  f"vision={entry['best_vision']:.2f} ({elapsed:.1f}s)",
                  flush=True)

        # Periodic save
        if (si + 1) % 50 == 0:
            _save_candidates(results, cfg["output_dir"])

    return results


def _save_candidates(results, output_dir):
    """Save candidates without image data (not JSON serializable)."""
    save_data = []
    for r in results:
        entry = {k: v for k, v in r.items() if k != "best_gen_ids"}
        save_data.append(entry)
    out_path = Path(output_dir) / "candidates.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)


# ══════════════════════════════════════════════════════════════════════
#  Phase 2: Vision-Dependent Filtering
# ══════════════════════════════════════════════════════════════════════

def filter_vision_dependent(model, processor, samples, results, device,
                             max_filter=200):
    """Filter to keep only vision-dependent questions."""
    print(f"\n[filter] Testing vision dependency on {min(len(results), max_filter)} samples...")
    kept = []
    tested = 0
    for i, (sample, result) in enumerate(zip(samples, results)):
        if tested >= max_filter:
            # Keep remaining unfiltered
            kept.append(result)
            continue
        if result["best_correct"] <= 0:
            continue  # Skip already wrong samples

        try:
            dep = is_vision_dependent(model, processor, sample, device)
            if dep:
                kept.append(result)
            tested += 1
        except Exception:
            kept.append(result)  # Keep on error
            tested += 1

        if (tested) % 50 == 0:
            print(f"  [{tested}/{max_filter}] kept={len(kept)}/{tested}")

    print(f"[filter] Kept {len(kept)}/{len(results)} vision-dependent samples")
    return kept


# ══════════════════════════════════════════════════════════════════════
#  Phase 3: SFT on Curated Data
# ══════════════════════════════════════════════════════════════════════

def run_sft(model, processor, results, samples, cfg, device):
    """Fine-tune on curated BoN data."""
    # Filter: only samples where best candidate is correct
    sft_data = []
    for r, s in zip(results, samples):
        if r["best_correct"] > 0:
            sft_data.append({
                "question": r["question"],
                "answer_text": r["best_text"],
                "image": s["image"],
            })

    if not sft_data:
        print("[sft] No correct candidates to train on!")
        return

    print(f"\n[sft] Training on {len(sft_data)} curated samples "
          f"({len(sft_data)/len(results)*100:.0f}% hit rate)")

    model.train()
    model.gradient_checkpointing_enable()
    for p in model.parameters(): p.requires_grad = True

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.get("sft_lr", 1e-6), weight_decay=0.01)

    epochs = cfg.get("sft_epochs", 2)
    losses = []

    for epoch in range(epochs):
        random.shuffle(sft_data)
        epoch_loss = 0.0
        n_batches = 0

        for i, item in enumerate(sft_data):
            try:
                # Build input with response
                from qwen_vl_utils import process_vision_info
                content = [{"type": "image", "image": item["image"]},
                           {"type": "text", "text": item["question"] + " Answer briefly."}]
                messages = [
                    {"role": "user", "content": content},
                    {"role": "assistant", "content": item["answer_text"]},
                ]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                    enable_thinking=True)
                imgs, _, _ = process_vision_info(
                    [{"role": "user", "content": content}],
                    return_video_kwargs=True)
                inputs = processor(text=[text], images=imgs,
                                   return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Compute prompt length for label masking
                prompt_content = [{"type": "image", "image": item["image"]},
                                  {"type": "text", "text": item["question"] + " Answer briefly."}]
                prompt_msgs = [{"role": "user", "content": prompt_content}]
                prompt_text = processor.apply_chat_template(
                    prompt_msgs, tokenize=False, add_generation_prompt=True,
                    enable_thinking=True)
                prompt_ids = processor.tokenizer.encode(prompt_text, add_special_tokens=False)
                prompt_len = len(prompt_ids)

                # Forward
                labels = inputs["input_ids"].clone()
                labels[0, :prompt_len] = -100  # Mask prompt

                outputs = model(**inputs, labels=labels)
                loss = outputs.loss

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                loss.backward()

                if (i + 1) % cfg.get("sft_grad_accum", 4) == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item()
                n_batches += 1

            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad()
                torch.cuda.empty_cache(); gc.collect()
                continue
            except Exception:
                continue

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} "
              f"({n_batches} batches)")

    # Final optimizer step
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()

    return losses


# ══════════════════════════════════════════════════════════════════════
#  Evaluation
# ══════════════════════════════════════════════════════════════════════

def evaluate_textvqa(model, processor, samples, device, max_eval=100):
    was_training = model.training
    model.eval()
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    total_acc = 0.0; total = 0
    for s in samples[:max_eval]:
        try:
            inputs = prepare_inputs(processor, s["image"],
                                     s["question"] + " Answer briefly.", device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            raw = processor.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
            for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
                raw = raw.replace(tok, "")
            pred = extract_answer(raw)
            total_acc += textvqa_accuracy(pred, s.get("answers_all", [s["answer"]]))
            total += 1
        except Exception:
            total += 1
    if was_training:
        model.train()
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
    return {"acc": total_acc / total if total > 0 else 0, "total": total}

def evaluate_pope(model, processor, samples, device, max_eval=60):
    was_training = model.training
    model.eval()
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    correct = total = 0
    for s in samples[:max_eval]:
        try:
            q = s["question"] + " Please answer yes or no."
            inputs = prepare_inputs(processor, s["image"], q, device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            raw = processor.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
            for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
                raw = raw.replace(tok, "")
            pred = extract_yes_no(raw)
            if pred == s["answer"]: correct += 1
            total += 1
        except Exception:
            total += 1
    if was_training:
        model.train()
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
    return {"acc": correct / total if total > 0 else 0, "total": total}

def evaluate_blind(model, processor, samples, device, n=50):
    was_training = model.training
    model.eval()
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    real_c = blind_c = total = 0
    for s in samples[:n]:
        try:
            q = s["question"] + " Please answer yes or no."
            gt = s["answer"]
            inputs = prepare_inputs(processor, s["image"], q, device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            raw = processor.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
            if extract_yes_no(raw) == gt: real_c += 1
            black = Image.new('RGB', s["image"].size, (0, 0, 0))
            inputs_b = prepare_inputs(processor, black, q, device)
            with torch.no_grad():
                out_b = model.generate(**inputs_b, max_new_tokens=512, do_sample=False)
            raw_b = processor.tokenizer.decode(
                out_b[0][inputs_b["input_ids"].shape[1]:], skip_special_tokens=False)
            if extract_yes_no(raw_b) == gt: blind_c += 1
            total += 1
        except Exception:
            total += 1
    if was_training:
        model.train()
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
    ra = real_c / total if total > 0 else 0
    ba = blind_c / total if total > 0 else 0
    return {"real_acc": ra, "blind_acc": ba, "gap": ra - ba, "total": total}


# ══════════════════════════════════════════════════════════════════════
#  Report Generation
# ══════════════════════════════════════════════════════════════════════

def generate_report(history, report_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Candidate quality distribution
    if "generation" in history:
        gen = history["generation"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.hist([r["best_correct"] for r in gen], bins=20, alpha=0.7, label="Best Correct")
        ax1.hist([r["mean_correct"] for r in gen], bins=20, alpha=0.5, label="Mean Correct")
        ax1.set_xlabel("Correctness Score"); ax1.legend()
        ax1.set_title("Candidate Quality Distribution")

        ax2.hist([r["best_vision"] for r in gen], bins=20, alpha=0.7, color='green',
                label="Best Vision")
        ax2.hist([r["mean_vision"] for r in gen], bins=20, alpha=0.5, color='orange',
                label="Mean Vision")
        ax2.set_xlabel("Vision Score"); ax2.legend()
        ax2.set_title("Vision Head Activation Δ")
        plt.tight_layout()
        plt.savefig(report_dir / "fig1_candidate_quality.png", dpi=150)
        plt.close()

    # Plot 2: Before/After comparison
    if "pre_eval" in history and "post_eval" in history:
        pre = history["pre_eval"]
        post = history["post_eval"]
        metrics = ["TextVQA", "POPE", "Gap"]
        pre_vals = [pre.get("textvqa", {}).get("acc", 0) * 100,
                   pre.get("pope", {}).get("acc", 0) * 100,
                   pre.get("blind", {}).get("gap", 0) * 100]
        post_vals = [post.get("textvqa", {}).get("acc", 0) * 100,
                    post.get("pope", {}).get("acc", 0) * 100,
                    post.get("blind", {}).get("gap", 0) * 100]

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(metrics))
        w = 0.35
        ax.bar(x - w/2, pre_vals, w, label='Before', color='steelblue')
        ax.bar(x + w/2, post_vals, w, label='After BoN+SFT', color='coral')
        ax.set_ylabel('Score (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.set_title('Phase 7: BoN+SFT with Head-Level Vision Scoring')
        for i, (pv, ov) in enumerate(zip(pre_vals, post_vals)):
            delta = ov - pv
            ax.annotate(f'{delta:+.1f}pp', xy=(i + w/2, ov + 0.5),
                       ha='center', fontsize=10, fontweight='bold')
        plt.tight_layout()
        plt.savefig(report_dir / "fig2_before_after.png", dpi=150)
        plt.close()

    with open(report_dir / "history.json", "w") as f:
        json.dump({k: v for k, v in history.items()
                   if k not in ("generation",)}, f, indent=2)
    # Save generation stats separately (can be large)
    if "generation" in history:
        gen_stats = [{k: v for k, v in r.items() if k != "best_gen_ids"}
                     for r in history["generation"]]
        with open(report_dir / "generation_stats.json", "w") as f:
            json.dump(gen_stats, f, indent=2)
    print(f"  [report] Saved to {report_dir}")


# ══════════════════════════════════════════════════════════════════════
#  Main Pipeline
# ══════════════════════════════════════════════════════════════════════

def run_pipeline(cfg):
    output_dir = Path(cfg["output_dir"])
    report_dir = Path("lab/reports/phase7_bon_sft")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Phase 7: BoN+SFT on TextVQA with Head-Level Vision Scoring")
    print(f"  N={cfg['group_size']} | w_correct={cfg['w_correct']} | "
          f"w_vision={cfg['w_vision']} | samples={cfg['num_samples']}")
    print(f"  SFT: lr={cfg['sft_lr']} | epochs={cfg['sft_epochs']}")
    print(f"{'='*70}\n")

    # Load model
    model, processor = load_model(cfg.get("model_path"), for_training=False)
    device = next(model.parameters()).device

    # Install vision hooks
    vision_heads = list(DEFAULT_VISION_HEADS[:cfg.get("top_k_heads", 12)])
    hooks = VisionHeadHooks(model, vision_heads, num_heads=16, head_dim=128)
    print(f"[hooks] Installed on {len(hooks.layers_needed)} layers")

    # Load data
    train_data = load_textvqa_train(cfg["num_samples"], cfg.get("seed", 42))
    textvqa_eval = load_textvqa_eval(200)
    pope_eval = load_pope_eval(300)

    history = {"config": cfg}

    # Pre-eval
    print("\n[eval] Pre-training baseline...")
    pre_tvqa = evaluate_textvqa(model, processor, textvqa_eval, device, 75)
    pre_pope = evaluate_pope(model, processor, pope_eval, device, 60)
    pre_blind = evaluate_blind(model, processor, pope_eval, device, 50)
    print(f"  TextVQA={pre_tvqa['acc']:.1%} POPE={pre_pope['acc']:.1%} "
          f"Gap={pre_blind['gap']:.1%}")
    history["pre_eval"] = {"textvqa": pre_tvqa, "pope": pre_pope, "blind": pre_blind}

    # Phase 1: Generate + Score
    print(f"\n[gen] Generating {cfg['group_size']} candidates × "
          f"{len(train_data)} samples...")
    results = generate_and_score(model, processor, train_data, cfg, hooks, device)
    history["generation"] = results

    hit_rate = np.mean([r["best_correct"] > 0 for r in results]) if results else 0
    mean_composite = np.mean([r["best_composite"] for r in results]) if results else 0
    print(f"\n[gen] Done: {len(results)} samples, hit_rate={hit_rate:.1%}, "
          f"mean_composite={mean_composite:.3f}")

    # Save candidates
    _save_candidates(results, output_dir)

    # Remove hooks before SFT
    hooks.remove()

    # Phase 2: SFT
    print(f"\n[sft] Fine-tuning on {sum(1 for r in results if r['best_correct'] > 0)} "
          f"correct candidates...")
    sft_losses = run_sft(model, processor, results, train_data, cfg, device)
    history["sft_losses"] = sft_losses

    # Post-eval
    print("\n[eval] Post-training evaluation...")
    model.eval()
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    post_tvqa = evaluate_textvqa(model, processor, textvqa_eval, device, 75)
    post_pope = evaluate_pope(model, processor, pope_eval, device, 60)
    post_blind = evaluate_blind(model, processor, pope_eval, device, 50)
    print(f"  TextVQA={post_tvqa['acc']:.1%} POPE={post_pope['acc']:.1%} "
          f"Gap={post_blind['gap']:.1%}")
    history["post_eval"] = {"textvqa": post_tvqa, "pope": post_pope, "blind": post_blind}

    # Save model
    model.save_pretrained(output_dir / "final")
    processor.save_pretrained(output_dir / "final")

    # Summary
    d_tvqa = (post_tvqa["acc"] - pre_tvqa["acc"]) * 100
    d_pope = (post_pope["acc"] - pre_pope["acc"]) * 100
    d_gap = (post_blind["gap"] - pre_blind["gap"]) * 100

    print(f"\n{'='*70}")
    print(f"  Phase 7 COMPLETE: BoN+SFT with Head-Level Vision Scoring")
    print(f"  TextVQA: {pre_tvqa['acc']:.1%} → {post_tvqa['acc']:.1%} ({d_tvqa:+.1f}pp)")
    print(f"  POPE:    {pre_pope['acc']:.1%} → {post_pope['acc']:.1%} ({d_pope:+.1f}pp)")
    print(f"  Gap:     {pre_blind['gap']:.1%} → {post_blind['gap']:.1%} ({d_gap:+.1f}pp)")
    print(f"  Checkpoint: {output_dir / 'final'}")
    print(f"{'='*70}")

    # Generate report
    generate_report(history, report_dir)

    del model; torch.cuda.empty_cache(); gc.collect()
    return history


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 7: BoN+SFT TextVQA")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Model path (default: HF base)")
    parser.add_argument("--output-dir", type=str,
                        default="checkpoints/phase7_bon_sft")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--w-correct", type=float, default=0.6)
    parser.add_argument("--w-vision", type=float, default=0.4)
    parser.add_argument("--vision-scale", type=float, default=10.0)
    parser.add_argument("--sft-lr", type=float, default=1e-6)
    parser.add_argument("--sft-epochs", type=int, default=2)
    parser.add_argument("--sft-grad-accum", type=int, default=4)
    parser.add_argument("--top-k-heads", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = vars(args)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
