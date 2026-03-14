"""
Phase 6: Head-Level Mask LSR-Weighted GRPO

Key innovations:
  v1: Head-level activation Δ(real, black) as per-token GRPO weights
  v2: TextVQA training data (fixes distribution mismatch)
  v3 (6b): GDPO normalization + curriculum filtering + VPPO token masking

Three fixes for training instability:
  1. GDPO: Normalize R_correct and R_head_lsr independently (prevents reward dominance)
  2. Curriculum: Skip samples where all candidates agree (zero gradient)
  3. VPPO masking: Zero-out gradients on tokens with head Δ < mean (focused learning)

Usage:
    PYTHONUNBUFFERED=1 python -u scripts/phase6_head_mask_grpo.py \
        --steps 30 --alpha 0.5 --gdpo --vppo-mask \
        2>&1 | tee logs/phase6b_head_mask.log
"""

import os, sys, gc, json, re, time, random, argparse, string
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from datetime import datetime
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

HF_ID = "Qwen/Qwen3-VL-2B-Thinking"
POPE_SPLITS = ["random", "popular", "adversarial"]
PROJECT_ROOT = Path(__file__).parent.parent

# Default vision heads from calibration (top-K by Cohen's d)
# Format: (layer_idx, head_idx, cohen_d)
DEFAULT_VISION_HEADS = [
    (5, 0, 9.795), (4, 6, 6.943), (23, 2, 6.602),
    (2, 9, 6.551), (5, 7, 6.353), (11, 2, 6.279),
    (2, 6, 5.440), (8, 3, 5.125), (2, 8, 5.022),
    (4, 1, 4.957), (10, 8, 4.932), (5, 10, 4.552),
]


# ══════════════════════════════════════════════════════════════════════
#  Utilities (shared with phase5)
# ══════════════════════════════════════════════════════════════════════

def split_thinking(text):
    m = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if m: return m.group(1).strip(), text[m.end():].strip()
    m = re.search(r'</think>', text)
    if m: return text[:m.start()].strip(), text[m.end():].strip()
    return "", text.strip()

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

def extract_answer(raw, qtype="short_answer"):
    _, answer = split_thinking(raw)
    if not answer: answer = raw
    text = answer.strip()
    if qtype == "yesno": return extract_yes_no(raw)
    if qtype == "mc":
        for ch in text[:5]:
            if ch.upper() in "ABCDEFGH": return ch.upper()
        return text[:20]
    return text.split("\n")[0].strip()[:100]

def find_think_token_range(tokenizer, gen_ids):
    gen_list = gen_ids.tolist()
    think_end_ids = tokenizer.encode("</think>", add_special_tokens=False)
    think_start_ids = tokenizer.encode("<think>", add_special_tokens=False)
    start_idx = 0
    if think_start_ids:
        for i in range(len(gen_list)):
            if gen_list[i] == think_start_ids[0]:
                start_idx = i + 1; break
    end_idx = len(gen_list)
    if think_end_ids:
        for i in range(len(gen_list)):
            if gen_list[i] == think_end_ids[0]:
                end_idx = i; break
    return start_idx, end_idx


# ══════════════════════════════════════════════════════════════════════
#  Data Loading
# ══════════════════════════════════════════════════════════════════════

def load_training_data(limit=500, seed=42):
    """Load TextVQA train — open-ended, image-dependent, same format as eval."""
    from datasets import load_dataset
    rng = random.Random(seed)
    samples = []

    print("[data] Loading TextVQA train...")
    try:
        ds = load_dataset("lmms-lab/textvqa", split="train", streaming=True)
        count = 0
        for row in ds:
            img = row.get("image")
            if img is None: continue
            answers = row.get("answers", [])
            if not answers: continue
            # Use most common answer as ground truth
            ans = Counter(answers).most_common(1)[0][0]
            samples.append({
                "question": row["question"] + " Answer briefly.",
                "answer": ans, "image": img,
                "answers_all": answers,  # Keep all for soft scoring
                "type": "short_answer", "source": "textvqa",
            })
            count += 1
            if count >= limit * 2: break  # Load extra, shuffle, then trim
    except Exception as e:
        print(f"  TextVQA error: {e}")

    rng.shuffle(samples)
    samples = samples[:limit]
    src = Counter(s["source"] for s in samples)
    print(f"[data] {len(samples)} training samples "
          f"({', '.join(f'{k}={v}' for k, v in src.items())})")
    return samples

def load_textvqa_eval(max_samples=200):
    """Load TextVQA val for evaluation — same format as training."""
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
            "image": row["image"], "question": row["question"],
            "answer": ans, "answers_all": answers,
            "type": "short_answer",
        })
        if len(samples) >= max_samples: break
    print(f"[data] {len(samples)} TextVQA eval samples")
    return samples

def load_pope_eval(max_samples=300):
    """POPE eval for cross-benchmark generalization check."""
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
#  Model Loading
# ══════════════════════════════════════════════════════════════════════

def load_model(model_path=None, for_training=True):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    path = model_path or HF_ID
    print(f"[model] Loading {path} (full finetune, bfloat16)...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(HF_ID, trust_remote_code=True)
    if for_training:
        model.train()
        model.gradient_checkpointing_enable()
        for p in model.parameters(): p.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable: {trainable:,} params, gradient checkpointing ON")
    return model, processor, processor.tokenizer


# ══════════════════════════════════════════════════════════════════════
#  Input Preparation & Generation
# ══════════════════════════════════════════════════════════════════════

def prepare_inputs(processor, image, question, device):
    from qwen_vl_utils import process_vision_info
    content = [{"type": "image", "image": image},
               {"type": "text", "text": question}]
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True)
    imgs, _, _ = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(text=[text], images=imgs, return_tensors="pt", padding=True)
    return {k: v.to(device) for k, v in inputs.items()}

def generate_candidates(model, processor, sample, group_size, temperature,
                        top_p, max_new_tokens, min_think_tokens, device):
    question = sample["question"]
    image = sample["image"]
    inputs = prepare_inputs(processor, image, question, device)
    prompt_len = inputs["input_ids"].shape[1]
    candidates, candidate_ids_list, think_ranges = [], [], []
    for _ in range(group_size):
        try:
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    min_new_tokens=min_think_tokens,
                    temperature=temperature, top_p=top_p, do_sample=True)
            gen_ids = out[0][prompt_len:].clone()
            text = processor.tokenizer.decode(gen_ids, skip_special_tokens=False)
            for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
                text = text.replace(tok, "")
            candidates.append(text.strip())
            candidate_ids_list.append(gen_ids.detach())
            think_ranges.append(find_think_token_range(processor.tokenizer, gen_ids))
        except Exception:
            candidates.append("")
            candidate_ids_list.append(torch.tensor([], dtype=torch.long, device=device))
            think_ranges.append((0, 0))
    inputs = {k: v.clone() if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}
    return candidates, candidate_ids_list, prompt_len, inputs, think_ranges


# ══════════════════════════════════════════════════════════════════════
#  HEAD-LEVEL MASK LSR (KEY INNOVATION)
# ══════════════════════════════════════════════════════════════════════

class VisionHeadHooks:
    """Lightweight hook manager for capturing vision head activations
    at ALL token positions (not just last token like profiler.py)."""

    def __init__(self, model, vision_heads, num_heads=16, head_dim=128):
        self.vision_heads = vision_heads  # list of (layer, head, cohen_d)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self._captured = {}  # layer_idx -> (batch, seq, hidden_size)
        self._hooks = []

        # Get unique layers that need hooks
        self.layers_needed = sorted(set(l for l, h, d in vision_heads))

        # Install hooks
        layers = model.model.language_model.layers
        for li in self.layers_needed:
            layer = layers[li]
            o_proj = layer.self_attn.o_proj

            def make_hook(layer_idx):
                def hook_fn(module, args):
                    # o_proj input: (batch, seq, num_heads * head_dim)
                    self._captured[layer_idx] = args[0].detach()
                return hook_fn

            handle = o_proj.register_forward_pre_hook(make_hook(li))
            self._hooks.append(handle)

    def get_per_token_head_acts(self, prompt_len, seq_len):
        """Extract per-token activations for vision heads only.

        Returns: dict of (layer, head) -> (seq_len, head_dim) for candidate tokens
        """
        result = {}
        for l, h, d in self.vision_heads:
            inp = self._captured.get(l)
            if inp is None:
                continue
            # inp shape: (1, total_seq, hidden_size)
            # Reshape to (1, total_seq, num_heads, head_dim)
            reshaped = inp[0].view(-1, self.num_heads, self.head_dim)
            # Extract candidate token positions, head h
            cand_acts = reshaped[prompt_len:prompt_len + seq_len, h, :]  # (seq_len, head_dim)
            result[(l, h)] = cand_acts
        return result

    def clear(self):
        self._captured.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._captured.clear()


def compute_head_level_lsr(model, processor, sample, candidate_ids,
                           think_range, device, hooks):
    """Compute per-token vision score using HEAD-LEVEL activation differences.

    Instead of KL(logits_real || logits_black), computes:
        score(t) = mean_h(||act_real[h,t] - act_black[h,t]||_2)
    where h iterates over calibrated vision heads.

    Returns:
        head_scores: tensor of shape (think_len,) — per-token head activation Δ
        mean_score: float — sequence-level mean
        think_len: int
    """
    if candidate_ids.numel() == 0:
        return torch.zeros(0, device=device), 0.0, 0

    t_start, t_end = think_range
    if t_end <= t_start:
        return torch.zeros(0, device=device), 0.0, 0

    image = sample["image"]
    question = sample["question"]
    candidate_ids = candidate_ids.clone().detach()
    n_cand = candidate_ids.numel()

    # Forward with real image (teacher-forced)
    real_inputs = prepare_inputs(processor, image, question, device)
    rpl = real_inputs["input_ids"].shape[1]
    rf = torch.cat([real_inputs["input_ids"], candidate_ids.unsqueeze(0)], dim=1)
    real_inputs["input_ids"] = rf
    real_inputs["attention_mask"] = torch.ones_like(rf)
    hooks.clear()
    with torch.no_grad():
        model(**real_inputs)
    real_acts = hooks.get_per_token_head_acts(rpl, n_cand)

    # Forward with black image (teacher-forced)
    black_image = Image.new('RGB', image.size, (0, 0, 0))
    black_inputs = prepare_inputs(processor, black_image, question, device)
    bpl = black_inputs["input_ids"].shape[1]
    bf = torch.cat([black_inputs["input_ids"], candidate_ids.unsqueeze(0)], dim=1)
    black_inputs["input_ids"] = bf
    black_inputs["attention_mask"] = torch.ones_like(bf)
    hooks.clear()
    with torch.no_grad():
        model(**black_inputs)
    black_acts = hooks.get_per_token_head_acts(bpl, n_cand)

    # Compute per-token vision score = mean head activation L2 difference
    t_end_safe = min(t_end, n_cand)
    t_start_safe = min(t_start, t_end_safe)
    think_len = t_end_safe - t_start_safe

    if think_len <= 0:
        del real_inputs, black_inputs
        hooks.clear()
        return torch.zeros(0, device=device), 0.0, 0

    # Compute per-token score across all vision heads
    scores = torch.zeros(think_len, device=device)
    n_heads_found = 0

    for (l, h) in real_acts:
        if (l, h) not in black_acts:
            continue
        ra = real_acts[(l, h)]  # (n_cand, head_dim)
        ba = black_acts[(l, h)]  # (n_cand, head_dim)

        # Compute L2 norm of difference for thinking tokens
        min_len = min(ra.shape[0], ba.shape[0], t_end_safe)
        if min_len <= t_start_safe:
            continue

        diff = (ra[t_start_safe:min_len] - ba[t_start_safe:min_len]).float()
        head_scores = diff.norm(dim=-1)  # (think_len_effective,)

        # Weight by Cohen's d (stronger vision heads matter more)
        cohen_d = 1.0
        for vl, vh, vd in hooks.vision_heads:
            if vl == l and vh == h:
                cohen_d = vd
                break

        effective_len = min(head_scores.shape[0], think_len)
        scores[:effective_len] += head_scores[:effective_len] * cohen_d
        n_heads_found += 1

    if n_heads_found > 0:
        scores /= n_heads_found  # Average across heads

    mean_score = scores.mean().item()

    del real_inputs, black_inputs, real_acts, black_acts
    hooks.clear()
    return scores, mean_score, think_len


def normalize_head_scores(scores, eps=1e-6):
    """Normalize per-token head scores so mean=1.0."""
    if scores.numel() == 0:
        return scores
    mean_s = scores.mean() + eps
    return scores / mean_s


def compute_decay_penalty(scores, smooth_window=3):
    """Penalty for head activation decrease during thinking chain."""
    if scores.numel() < 2:
        return 0.0
    s = scores.float()
    if smooth_window > 1 and s.numel() >= smooth_window:
        kernel = torch.ones(1, 1, smooth_window, device=s.device) / smooth_window
        s_smooth = F.conv1d(
            s.unsqueeze(0).unsqueeze(0), kernel,
            padding=smooth_window // 2
        ).squeeze()[:s.numel()]
    else:
        s_smooth = s
    gradient = torch.diff(s_smooth)
    decay = torch.clamp(-gradient, min=0).sum()
    return decay.item() / max(s.numel() - 1, 1)


# ══════════════════════════════════════════════════════════════════════
#  Correctness Reward
# ══════════════════════════════════════════════════════════════════════

def compute_r_correct(prediction, ground_truth, qtype="short_answer",
                      answers_all=None):
    if not prediction: return 0.0
    pred = prediction.strip().lower()
    gt = ground_truth.strip().lower()
    if qtype == "yesno":
        return 1.0 if extract_yes_no(prediction) == gt else 0.0
    if qtype == "mc":
        return 1.0 if pred[:1] == gt[:1].lower() else 0.0
    # For TextVQA: use multi-answer VQA accuracy if available
    if answers_all:
        return textvqa_accuracy(prediction, answers_all)
    # Fallback: F1 token overlap
    pred_tokens = set(pred.split())
    gt_tokens = set(gt.split())
    if not gt_tokens: return 0.0
    overlap = pred_tokens & gt_tokens
    if not overlap:
        # Try substring match
        if gt in pred or pred in gt:
            return 0.5
        return 0.0
    p = len(overlap) / len(pred_tokens)
    r = len(overlap) / len(gt_tokens)
    return 2 * p * r / (p + r)


# ══════════════════════════════════════════════════════════════════════
#  Head-Level Weighted Rewards + Loss
# ══════════════════════════════════════════════════════════════════════

def gdpo_normalize(values, eps=1e-8):
    """GDPO: Normalize a reward component independently to zero mean, unit std."""
    arr = np.array(values)
    std = arr.std()
    if std < eps:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / (std + eps)


def compute_rewards_with_head_lsr(model, processor, sample, candidates,
                                   cand_ids_list, think_ranges, prompt_len,
                                   device, cfg, hooks):
    """Compute rewards AND per-token head-level weights for each candidate.

    v3 changes:
      - GDPO: normalize R_correct and R_head_lsr independently before combining
      - VPPO masking: zero-out token weights where head Δ < mean (focused learning)
    """
    r_correct_list, r_lsr_list = [], []
    head_scores_list, think_lens = [], []
    details, token_weights_list = [], []
    gt = sample["answer"]
    qtype = sample.get("type", "short_answer")
    answers_all = sample.get("answers_all")

    # Phase 1: Compute raw rewards for ALL candidates first (needed for GDPO)
    for cand, cand_ids, t_range in zip(candidates, cand_ids_list, think_ranges):
        pred = extract_answer(cand, qtype)
        r_correct = compute_r_correct(pred, gt, qtype, answers_all=answers_all)

        try:
            head_scores, mean_score, think_len = compute_head_level_lsr(
                model, processor, sample, cand_ids, t_range, device, hooks)
        except Exception:
            head_scores = torch.zeros(0, device=device)
            mean_score, think_len = 0.0, 0

        r_lsr = min(mean_score / cfg["lsr_scale"], 1.0)

        r_correct_list.append(r_correct)
        r_lsr_list.append(r_lsr)
        head_scores_list.append(head_scores)
        think_lens.append(think_len)

    # Phase 2: GDPO normalization — normalize each reward component independently
    use_gdpo = cfg.get("gdpo", False)
    use_vppo = cfg.get("vppo_mask", False)

    if use_gdpo and len(r_correct_list) > 1:
        # Independent normalization (GDPO arXiv:2601.05242)
        w_correct = cfg.get("gdpo_w_correct", 0.6)
        w_lsr = cfg.get("gdpo_w_lsr", 0.4)
        norm_correct = gdpo_normalize(r_correct_list)
        norm_lsr = gdpo_normalize(r_lsr_list)
        combined = w_correct * norm_correct + w_lsr * norm_lsr
        rewards = combined.tolist()
    else:
        # Legacy gated reward
        rewards = []
        for rc, rl in zip(r_correct_list, r_lsr_list):
            r_total = rc * 0.5 + rc * rl * 0.5
            rewards.append(r_total)

    # Phase 3: Build per-token weights + decay penalty
    for i, (cand_ids, t_range) in enumerate(zip(cand_ids_list, think_ranges)):
        head_scores = head_scores_list[i]
        decay_pen = compute_decay_penalty(head_scores) if head_scores.numel() >= 2 else 0.0
        rewards[i] -= cfg["beta_decay"] * decay_pen

        t_start, t_end = t_range
        n_tokens = cand_ids.numel()
        token_w = torch.ones(n_tokens, device=device)

        if head_scores.numel() >= 10 and cfg["alpha"] > 0:
            norm_scores = normalize_head_scores(head_scores)
            norm_scores = torch.clamp(norm_scores, 0.0, 5.0)
            t_end_safe = min(t_end, n_tokens)
            t_start_safe = min(t_start, t_end_safe)
            w_len = min(norm_scores.numel(), t_end_safe - t_start_safe)
            if w_len > 0:
                if use_vppo:
                    # VPPO masking: zero-out tokens with below-mean head Δ
                    # Only compute gradients on visually-grounded tokens
                    mask = (norm_scores[:w_len] >= 1.0).float()  # >= mean (normalized)
                    token_w[t_start_safe:t_start_safe + w_len] = (
                        mask * (1.0 + cfg["alpha"] * norm_scores[:w_len]))
                else:
                    token_w[t_start_safe:t_start_safe + w_len] = (
                        1.0 + cfg["alpha"] * norm_scores[:w_len])

        details.append({
            "correct": r_correct_list[i], "head_score_raw": r_lsr_list[i] * cfg["lsr_scale"],
            "decay_penalty": decay_pen,
            "token_weight_mean": token_w.mean().item(),
            "token_weight_max": token_w.max().item(),
            "think_len": think_lens[i],
        })
        token_weights_list.append(token_w)

    return rewards, details, token_weights_list


def compute_weighted_logprobs(model, inputs, candidate_ids, prompt_len,
                              token_weights=None):
    """Compute log-probs with optional per-token weighting."""
    full_ids = torch.cat([inputs["input_ids"][:, :prompt_len],
                          candidate_ids.unsqueeze(0)], dim=1)
    attn = torch.ones_like(full_ids)
    fwd = {k: v for k, v in inputs.items()
           if k not in ("input_ids", "attention_mask")}
    fwd["input_ids"] = full_ids
    fwd["attention_mask"] = attn

    out = model(**fwd)
    logits = out.logits[0, prompt_len - 1: prompt_len - 1 + len(candidate_ids)]
    log_probs = F.log_softmax(logits, dim=-1)
    token_lp = log_probs.gather(1, candidate_ids.unsqueeze(1)).squeeze(1)

    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()

    if token_weights is not None:
        w = token_weights[:len(token_lp)]
        if len(w) < len(token_lp):
            w = F.pad(w, (0, len(token_lp) - len(w)), value=1.0)
        weighted_lp = (token_lp * w).sum() / (w.sum() + 1e-8)
    else:
        weighted_lp = token_lp.mean()

    return weighted_lp, entropy


def compute_head_lsr_grpo_loss(model, inputs, cand_ids_list, prompt_len,
                                advantages, token_weights_list, cfg):
    """GRPO loss with per-token head-level weighting."""
    total_loss = torch.tensor(0.0, device=next(model.parameters()).device,
                              requires_grad=True)
    n_valid = 0
    stats = {"entropy": [], "token_weight_mean": []}

    for cand_ids, adv, tw in zip(cand_ids_list, advantages, token_weights_list):
        if cand_ids.numel() == 0 or abs(adv) < 1e-8:
            continue

        weighted_lp, entropy = compute_weighted_logprobs(
            model, inputs, cand_ids, prompt_len, tw)

        policy_loss = -weighted_lp * adv
        loss = policy_loss - cfg["beta_entropy"] * entropy

        total_loss = total_loss + loss
        n_valid += 1
        stats["entropy"].append(entropy.item())
        stats["token_weight_mean"].append(tw.mean().item())

    if n_valid > 0:
        total_loss = total_loss / n_valid
    return total_loss, stats


# ══════════════════════════════════════════════════════════════════════
#  Evaluation
# ══════════════════════════════════════════════════════════════════════

def textvqa_accuracy(pred, answers_all):
    """VQA accuracy with fuzzy matching for thinking model outputs.
    Checks both exact match AND substring containment."""
    if not pred:
        return 0.0
    pred_clean = pred.strip().lower()
    # Remove common prefixes from thinking model output
    for prefix in ["the answer is ", "it says ", "the text reads ",
                   "the brand is ", "it is ", "this is "]:
        if pred_clean.startswith(prefix):
            pred_clean = pred_clean[len(prefix):]

    # Try exact match first
    match_count = sum(1 for a in answers_all if a.strip().lower() == pred_clean)
    if match_count > 0:
        return min(match_count / 3.0, 1.0)

    # Try substring: GT contained in pred
    match_count = sum(1 for a in answers_all
                      if a.strip().lower() in pred_clean)
    if match_count > 0:
        return min(match_count / 3.0, 1.0)

    # Try substring: pred contained in GT
    match_count = sum(1 for a in answers_all
                      if pred_clean in a.strip().lower())
    return min(match_count / 3.0, 1.0)

def evaluate_textvqa(model, processor, samples, device, max_eval=100):
    """Evaluate on TextVQA using VQA accuracy metric."""
    was_training = model.training
    model.eval()
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    total_acc = 0.0
    total = 0
    for s in samples[:max_eval]:
        try:
            q = s["question"]
            inputs = prepare_inputs(processor, s["image"], q, device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            raw = processor.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
            for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
                raw = raw.replace(tok, "")
            pred = extract_answer(raw, "short_answer")
            acc = textvqa_accuracy(pred, s.get("answers_all", [s["answer"]]))
            total_acc += acc
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
    think_lengths = []
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
            gt = s["answer"]
            if pred == gt: correct += 1
            total += 1
            thinking, _ = split_thinking(raw)
            think_lengths.append(len(thinking.split()) if thinking else 0)
        except Exception:
            total += 1
    if was_training:
        model.train()
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
    acc = correct / total if total > 0 else 0
    return {"acc": acc, "total": total,
            "avg_think_words": np.mean(think_lengths) if think_lengths else 0}

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

    steps = [s for s in history["steps"] if not s.get("skipped")]
    if not steps:
        with open(report_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        return

    # Plot 1: Loss + Reward
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = [s["step"] for s in steps]
    ax1.plot(x, [s["loss"] for s in steps], 'b-o', label="Loss")
    ax1.set_xlabel("Step"); ax1.set_ylabel("Loss"); ax1.legend()
    ax1.set_title("Training Loss")

    ax2.plot(x, [s["mean_reward"] for s in steps], 'g-o', label="Reward")
    ax2.plot(x, [s.get("mean_decay_pen", 0) for s in steps], 'r--s',
             label="Decay Penalty", alpha=0.7)
    ax2.set_xlabel("Step"); ax2.set_ylabel("Value"); ax2.legend()
    ax2.set_title("Reward + Head Decay Penalty")
    plt.tight_layout()
    plt.savefig(report_dir / "fig1_training_curves.png", dpi=150)
    plt.close()

    # Plot 2: Token weight stats + head scores
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(x, [s.get("token_weight_mean", 1.0) for s in steps], 'b-o',
            label="Mean weight")
    ax1.plot(x, [s.get("token_weight_max", 1.0) for s in steps], 'r--s',
            label="Max weight")
    ax1.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel("Step"); ax1.set_ylabel("Token Weight")
    ax1.legend(); ax1.set_title("Head-Level Token Weights")

    ax2.plot(x, [s.get("mean_head_score", 0) for s in steps], 'purple',
             marker='o', label="Mean Head Δ")
    ax2.set_xlabel("Step"); ax2.set_ylabel("Head Score")
    ax2.legend(); ax2.set_title("Vision Head Activation Δ (real vs black)")
    plt.tight_layout()
    plt.savefig(report_dir / "fig2_head_weights.png", dpi=150)
    plt.close()

    # Plot 3: Eval progression (TextVQA + POPE + Blind Gap)
    evals = history.get("evals", [])
    if evals:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        ex = [e["step"] for e in evals]
        if evals[0].get("textvqa"):
            axes[0].plot(ex, [e["textvqa"]["acc"] * 100 for e in evals], 'r-o')
            axes[0].set_xlabel("Step"); axes[0].set_ylabel("TextVQA Acc (%)")
            axes[0].set_title("TextVQA Accuracy (primary)")
        axes[1].plot(ex, [e["pope"]["acc"] * 100 for e in evals], 'b-o')
        axes[1].set_xlabel("Step"); axes[1].set_ylabel("POPE Acc (%)")
        axes[1].set_title("POPE Accuracy (generalization)")
        axes[2].plot(ex, [e["blind"]["gap"] * 100 for e in evals], 'g-o')
        axes[2].set_xlabel("Step"); axes[2].set_ylabel("Blind Gap (pp)")
        axes[2].set_title("Blind Test Gap")
        plt.tight_layout()
        plt.savefig(report_dir / "fig3_eval_progression.png", dpi=150)
        plt.close()

    with open(report_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"  [report] Saved to {report_dir}")


# ══════════════════════════════════════════════════════════════════════
#  Main Training Loop
# ══════════════════════════════════════════════════════════════════════

def run_training(cfg, train_data, eval_data, model_path=None,
                 textvqa_eval_data=None):
    output_dir = Path(cfg["output_dir"])
    run_name = output_dir.name
    report_dir = Path("lab/reports/phase6_head_mask") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    gdpo_str = "ON" if cfg.get("gdpo") else "OFF"
    vppo_str = "ON" if cfg.get("vppo_mask") else "OFF"
    print(f"\n{'='*70}")
    print(f"  Phase 6b: Head-Level Mask LSR-Weighted GRPO")
    print(f"  GDPO={gdpo_str} | VPPO={vppo_str}")
    print(f"  alpha={cfg['alpha']} | beta_decay={cfg['beta_decay']} | "
          f"steps={cfg['num_steps']} | group={cfg['group_size']} | "
          f"T={cfg['temperature']} | lr={cfg['lr']}")
    if cfg.get("gdpo"):
        print(f"  GDPO weights: correct={cfg.get('gdpo_w_correct', 0.6)}, "
              f"lsr={cfg.get('gdpo_w_lsr', 0.4)}")
    print(f"  Vision heads: {len(cfg['vision_heads'])} heads across "
          f"{len(set(l for l,h,d in cfg['vision_heads']))} layers")
    print(f"{'='*70}\n")

    model, processor, tokenizer = load_model(model_path, for_training=True)
    device = next(model.parameters()).device

    # Install vision head hooks (persistent during training)
    hooks = VisionHeadHooks(model, cfg["vision_heads"],
                            num_heads=16, head_dim=128)
    print(f"[hooks] Installed on layers {hooks.layers_needed}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr"], weight_decay=0.01)

    # Pre-eval
    print("Pre-training eval...")
    pre_textvqa = evaluate_textvqa(model, processor, textvqa_eval_data or [], device, 50) \
        if textvqa_eval_data else {"acc": 0, "total": 0}
    pre_pope = evaluate_pope(model, processor, eval_data, device, 60)
    pre_blind = evaluate_blind(model, processor, eval_data, device, 50)
    print(f"  TextVQA: {pre_textvqa['acc']:.1%} | POPE: {pre_pope['acc']:.1%} | Gap: {pre_blind['gap']:.1%}")

    history = {
        "config": {k: v for k, v in cfg.items() if k != "vision_heads"},
        "vision_heads": [(l, h, d) for l, h, d in cfg["vision_heads"]],
        "pre_eval": {"textvqa": pre_textvqa, "pope": pre_pope, "blind": pre_blind},
        "steps": [], "evals": [],
    }

    model.train()
    optimizer.zero_grad()
    best_acc = pre_pope["acc"]

    for step in range(cfg["num_steps"]):
        step_t0 = time.time()
        sample = train_data[step % len(train_data)]

        # Generate candidates
        model.eval()
        if hasattr(model, 'gradient_checkpointing_disable'):
            model.gradient_checkpointing_disable()
        try:
            candidates, cand_ids_list, prompt_len, inputs, think_ranges = \
                generate_candidates(
                    model, processor, sample, cfg["group_size"],
                    cfg["temperature"], cfg["top_p"],
                    cfg["max_new_tokens"], cfg["min_think_tokens"], device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); gc.collect()
            print(f"  [step {step+1}] OOM gen, skip"); continue

        # Rewards + head-level token weights
        try:
            rewards, details, token_weights_list = compute_rewards_with_head_lsr(
                model, processor, sample, candidates, cand_ids_list,
                think_ranges, prompt_len, device, cfg, hooks)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); gc.collect()
            print(f"  [step {step+1}] OOM reward, skip"); continue

        rarr = np.array(rewards)
        rstd = rarr.std()

        # Curriculum filtering + Dynamic resampling (DAPO-style)
        # Skip when all candidates agree → zero gradient → wasted step
        if rstd < 1e-8:
            resample_tries = 0
            max_resample = 5  # Try harder (was 3)
            while rstd < 1e-8 and resample_tries < max_resample:
                resample_tries += 1
                alt_idx = (step * 7 + resample_tries * 13) % len(train_data)
                alt_sample = train_data[alt_idx]
                try:
                    candidates, cand_ids_list, prompt_len, inputs, think_ranges = \
                        generate_candidates(
                            model, processor, alt_sample, cfg["group_size"],
                            cfg["temperature"], cfg["top_p"],
                            cfg["max_new_tokens"], cfg["min_think_tokens"], device)
                    rewards, details, token_weights_list = compute_rewards_with_head_lsr(
                        model, processor, alt_sample, candidates, cand_ids_list,
                        think_ranges, prompt_len, device, cfg, hooks)
                    sample = alt_sample
                    rarr = np.array(rewards)
                    rstd = rarr.std()
                except Exception:
                    break

            if rstd < 1e-8:
                elapsed = time.time() - step_t0
                print(f"  [step {step+1}/{cfg['num_steps']}] "
                      f"SKIP r={rarr.mean():.3f} ({elapsed:.1f}s)")
                history["steps"].append({
                    "step": step + 1, "skipped": True,
                    "mean_reward": float(rarr.mean())})
                del candidates, cand_ids_list, inputs; continue

        if cfg.get("gdpo", False):
            # GDPO: rewards are already independently normalized + combined
            # Just use them directly as advantages (already zero-mean-ish)
            advantages = ((rarr - rarr.mean()) / (rstd + 1e-8)).tolist()
        else:
            advantages = ((rarr - rarr.mean()) / (rstd + 1e-8)).tolist()

        # Loss with head-level token weighting
        model.train()
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        try:
            loss, lstats = compute_head_lsr_grpo_loss(
                model, inputs, cand_ids_list, prompt_len,
                advantages, token_weights_list, cfg)
            (loss / cfg["grad_accum"]).backward()

            if (step + 1) % cfg["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.get("max_grad_norm", 1.0))
                optimizer.step()
                optimizer.zero_grad()
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad()
            torch.cuda.empty_cache(); gc.collect()
            print(f"  [step {step+1}] OOM loss, skip"); continue

        elapsed = time.time() - step_t0
        mc = np.mean([d["correct"] for d in details])
        mh = np.mean([d["head_score_raw"] for d in details])
        md = np.mean([d["decay_penalty"] for d in details])
        tw_mean = np.mean([d["token_weight_mean"] for d in details])
        tw_max = np.mean([d["token_weight_max"] for d in details])

        history["steps"].append({
            "step": step + 1, "loss": loss.item(),
            "mean_reward": float(rarr.mean()),
            "reward_std": float(rstd),
            "mean_correct": float(mc),
            "mean_head_score": float(mh),
            "mean_decay_pen": float(md),
            "token_weight_mean": float(tw_mean),
            "token_weight_max": float(tw_max),
            "mean_entropy": float(np.mean(lstats["entropy"])) if lstats["entropy"] else 0,
            "elapsed": elapsed,
        })

        print(f"  [step {step+1}/{cfg['num_steps']}] "
              f"loss={loss.item():.4f} r={rarr.mean():.3f}±{rstd:.3f} "
              f"correct={mc:.2f} headΔ={mh:.3f} decay={md:.3f} "
              f"tw={tw_mean:.2f}/{tw_max:.1f} ({elapsed:.1f}s)", flush=True)

        # Eval
        if (step + 1) % cfg["eval_every"] == 0 or step + 1 == cfg["num_steps"]:
            tvqa_res = evaluate_textvqa(model, processor,
                                         textvqa_eval_data or [], device, 50) \
                if textvqa_eval_data else {"acc": 0, "total": 0}
            pope_res = evaluate_pope(model, processor, eval_data, device, 60)
            blind_res = evaluate_blind(model, processor, eval_data, device, 50)
            print(f"  === Eval step {step+1}: "
                  f"TextVQA={tvqa_res['acc']:.1%} "
                  f"POPE={pope_res['acc']:.1%} "
                  f"Gap={blind_res['gap']:.1%} ===")
            history["evals"].append({
                "step": step + 1, "textvqa": tvqa_res,
                "pope": pope_res, "blind": blind_res})

            # Track best by TextVQA (primary metric)
            metric = tvqa_res["acc"] if tvqa_res["total"] > 0 else pope_res["acc"]
            if metric > best_acc:
                best_acc = metric
                model.save_pretrained(output_dir / "best")
                processor.save_pretrained(output_dir / "best")
                print(f"  ★ New best: {best_acc:.1%}")

        del candidates, cand_ids_list, inputs, token_weights_list

    # Cleanup hooks
    hooks.remove()

    # Final eval
    history["final_eval"] = history["evals"][-1] if history["evals"] else {}

    # Save final model
    model.save_pretrained(output_dir / "final")
    processor.save_pretrained(output_dir / "final")

    # Generate report
    generate_report(history, report_dir)

    # Summary
    pre_tvqa = pre_textvqa["acc"]
    pre_pope_acc = pre_pope["acc"]
    pre_gap = pre_blind["gap"]
    final = history.get("final_eval", {})
    f_tvqa = final.get("textvqa", {}).get("acc", 0)
    f_pope = final.get("pope", {}).get("acc", 0)
    f_gap = final.get("blind", {}).get("gap", 0)

    print(f"\n{'='*70}")
    print(f"  Phase 6 COMPLETE (Head-Level Mask)")
    print(f"  TextVQA: {pre_tvqa:.1%} → {f_tvqa:.1%} ({(f_tvqa-pre_tvqa)*100:+.1f}pp)")
    print(f"  POPE: {pre_pope_acc:.1%} → {f_pope:.1%} ({(f_pope-pre_pope_acc)*100:+.1f}pp)")
    print(f"  Gap:  {pre_gap:.1%} → {f_gap:.1%} ({(f_gap-pre_gap)*100:+.1f}pp)")
    print(f"  Best: {best_acc:.1%}")
    print(f"  Saved to {output_dir}")
    print(f"{'='*70}")

    del model; torch.cuda.empty_cache(); gc.collect()
    return history


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 6: Head-Level Mask LSR GRPO")
    parser.add_argument("--model-path", type=str,
                        default="Qwen/Qwen3-VL-2B-Thinking")
    parser.add_argument("--output-dir", type=str,
                        default="checkpoints/phase6_head_mask/alpha05")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=1.3)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Head-level weight scale (0=standard GRPO)")
    parser.add_argument("--beta-decay", type=float, default=0.1,
                        help="Decay penalty weight")
    parser.add_argument("--lsr-scale", type=float, default=10.0,
                        help="Head score normalization (head scores range 5-10)")
    parser.add_argument("--beta-entropy", type=float, default=0.01)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--min-think-tokens", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--train-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k-heads", type=int, default=12,
                        help="Number of top vision heads to use")
    parser.add_argument("--calibration-file", type=str,
                        default="checkpoints/calibration/qwen3_vl_2b/calibration_meta.json")
    # Phase 6b: GDPO + VPPO
    parser.add_argument("--gdpo", action="store_true",
                        help="GDPO: normalize rewards independently")
    parser.add_argument("--gdpo-w-correct", type=float, default=0.6,
                        help="GDPO weight for R_correct")
    parser.add_argument("--gdpo-w-lsr", type=float, default=0.4,
                        help="GDPO weight for R_head_lsr")
    parser.add_argument("--vppo-mask", action="store_true",
                        help="VPPO: zero gradient on low-Δ tokens")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load vision heads from calibration
    vision_heads = list(DEFAULT_VISION_HEADS[:args.top_k_heads])
    if os.path.exists(args.calibration_file):
        try:
            with open(args.calibration_file) as f:
                meta = json.load(f)
            scores = meta["head_scores"]
            sorted_heads = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            vision_heads = []
            for key, d in sorted_heads[:args.top_k_heads]:
                l, h = key.split("_")
                vision_heads.append((int(l), int(h), float(d)))
            print(f"[cal] Loaded {len(vision_heads)} vision heads from {args.calibration_file}")
        except Exception as e:
            print(f"[cal] Using default heads: {e}")

    cfg = vars(args)
    cfg["num_steps"] = cfg.pop("steps")
    cfg["vision_heads"] = vision_heads

    train_data = load_training_data(args.train_samples, args.seed)
    pope_eval_data = load_pope_eval(300)
    textvqa_eval_data = load_textvqa_eval(200)

    run_training(cfg, train_data, pope_eval_data, args.model_path,
                 textvqa_eval_data=textvqa_eval_data)


if __name__ == "__main__":
    main()
