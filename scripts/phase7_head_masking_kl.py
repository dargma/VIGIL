"""
Phase 7: Head Masking KL + Learned Head Importance

Exp4: Head Masking KL GRPO
  - Mask 12 calibrated vision heads → forward pass → get logits_masked
  - Normal forward pass → get logits_normal
  - R_headKL = KL(logits_normal || logits_masked) — higher = model relies on vision heads
  - GRPO optimizes: R_correct + R_headKL (GDPO-normalized)

Exp5: Learned Head Importance + Head Masking KL
  - Replace binary 12-head mask with trainable head_importance [28, 16]
  - Initialized from Cohen's d scores (softmax-normalized with temperature)
  - Soft masking: act *= (1 - head_importance[l, h]) during masked forward
  - head_importance is optimized by GRPO gradient alongside model weights
  - Combines learned importance with KL reward for end-to-end head selection

Exp6: Learned Head Importance + Gated Head-LSR
  - Extends Exp1 (Gated Head-LSR) with trainable head_importance [28, 16]
  - Uses real-vs-black image activation delta (like Exp1) but with soft head weights
  - token_weight(t) = 1.0 + alpha * sum(sigmoid(importance[l,h]) * delta[l,h,t])
  - head_importance is optimized by GRPO gradient alongside model weights
  - Gated: uses correctness-only when R_correct has variance, else importance-weighted LSR

Usage:
    # Exp4: Head Masking KL
    PYTHONUNBUFFERED=1 python -u scripts/phase7_head_masking_kl.py \
        --exp 4 --steps 15 --gdpo \
        2>&1 | tee logs/phase7_exp4_head_kl.log

    # Exp5: Learned Head Importance + KL
    PYTHONUNBUFFERED=1 python -u scripts/phase7_head_masking_kl.py \
        --exp 5 --steps 15 --gdpo --importance-lr 1e-3 --importance-temp 2.0 \
        2>&1 | tee logs/phase7_exp5_learned.log

    # Exp6: Learned Head Importance + Gated Head-LSR
    PYTHONUNBUFFERED=1 python -u scripts/phase7_head_masking_kl.py \
        --exp 6 --steps 15 --gdpo --importance-lr 1e-3 --importance-temp 2.0 \
        2>&1 | tee logs/phase7_exp6_learned_lsr.log
"""

import os, sys, gc, json, re, time, random, argparse, string
import numpy as np
import torch
import torch.nn as nn
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
DEFAULT_VISION_HEADS = [
    (5, 0, 9.795), (4, 6, 6.943), (23, 2, 6.602),
    (2, 9, 6.551), (5, 7, 6.353), (11, 2, 6.279),
    (2, 6, 5.440), (8, 3, 5.125), (2, 8, 5.022),
    (4, 1, 4.957), (10, 8, 4.932), (5, 10, 4.552),
]

# Full head d-scores for all 448 heads (28 layers × 16 heads)
# Will be populated from calibration file
ALL_HEAD_SCORES = {}


# ══════════════════════════════════════════════════════════════════════
#  Utilities (shared with phase6)
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
#  Data Loading
# ══════════════════════════════════════════════════════════════════════

def load_training_data(limit=500, seed=42):
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
            ans = Counter(answers).most_common(1)[0][0]
            samples.append({
                "question": row["question"] + " Answer briefly.",
                "answer": ans, "image": img,
                "answers_all": answers,
                "type": "short_answer", "source": "textvqa",
            })
            count += 1
            if count >= limit * 2: break
    except Exception as e:
        print(f"  TextVQA error: {e}")
    rng.shuffle(samples)
    samples = samples[:limit]
    print(f"[data] {len(samples)} training samples")
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
            "image": row["image"], "question": row["question"],
            "answer": ans, "answers_all": answers,
            "type": "short_answer",
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
    candidates, candidate_ids_list = [], []
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
        except Exception:
            candidates.append("")
            candidate_ids_list.append(torch.tensor([], dtype=torch.long, device=device))
    inputs = {k: v.clone() if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}
    return candidates, candidate_ids_list, prompt_len, inputs


# ══════════════════════════════════════════════════════════════════════
#  HEAD MASKING KL (EXP4 — CORE INNOVATION)
# ══════════════════════════════════════════════════════════════════════

class HeadMaskingHooks:
    """Hook manager that supports two modes:
    1. Normal mode: captures activations, no modification
    2. Masked mode: zeros out (or soft-masks) vision head activations at o_proj input

    For Exp4: binary masking (zero out selected heads)
    For Exp6: soft masking via learned head_importance weights
    """

    def __init__(self, model, vision_heads, num_heads=16, head_dim=128,
                 head_importance=None):
        """
        Args:
            vision_heads: list of (layer, head, cohen_d) for binary masking
            head_importance: optional Tensor[28, 16] for soft masking (Exp6)
        """
        self.vision_heads = vision_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.head_importance = head_importance  # Tensor[28, 16] or None
        self._hooks = []
        self._masking_active = False  # Toggle for masked forward pass

        # Build binary mask lookup: set of (layer, head) to mask
        self.mask_set = set((l, h) for l, h, d in vision_heads)
        self.layers_needed = sorted(set(l for l, h, d in vision_heads))

        # For Exp6: all layers if we have learned importance
        if head_importance is not None:
            self.layers_needed = list(range(head_importance.shape[0]))

        # Install hooks
        layers = model.model.language_model.layers
        for li in self.layers_needed:
            if li >= len(layers):
                continue
            layer = layers[li]
            o_proj = layer.self_attn.o_proj

            def make_hook(layer_idx):
                def hook_fn(module, args):
                    if not self._masking_active:
                        return  # Normal mode: pass through

                    # args[0] shape: (batch, seq, num_heads * head_dim)
                    x = args[0]
                    batch, seq, hidden = x.shape
                    x_heads = x.view(batch, seq, self.num_heads, self.head_dim)

                    if self.head_importance is not None:
                        # Exp6: Soft masking with learned importance
                        # importance[layer, head] in [0, 1] after sigmoid
                        imp = torch.sigmoid(self.head_importance[layer_idx])
                        # Mask = scale activations by (1 - importance)
                        # High importance → more masking → bigger KL
                        mask = (1.0 - imp).view(1, 1, self.num_heads, 1)
                        x_masked = x_heads * mask
                    else:
                        # Exp4: Binary masking
                        x_masked = x_heads.clone()
                        for h in range(self.num_heads):
                            if (layer_idx, h) in self.mask_set:
                                x_masked[:, :, h, :] = 0.0

                    return (x_masked.view(batch, seq, hidden),) + args[1:]
                return hook_fn

            handle = o_proj.register_forward_pre_hook(make_hook(li))
            self._hooks.append(handle)

    def set_masking(self, active: bool):
        """Toggle masking on/off."""
        self._masking_active = active

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def compute_head_masking_kl(model, inputs, candidate_ids, prompt_len,
                            hooks, device, kl_scale=1.0):
    """Compute KL divergence between normal and head-masked logits.

    KL(P_normal || P_masked) = sum(p_normal * log(p_normal / p_masked))

    Higher KL = model depends more on vision heads = higher reward.

    Returns:
        kl_value: float — mean KL divergence per token
        per_token_kl: Tensor — KL at each generated token position
    """
    if candidate_ids.numel() == 0:
        return 0.0, torch.zeros(0, device=device)

    n_cand = candidate_ids.numel()

    # Build teacher-forced input: prompt + candidate tokens
    full_ids = torch.cat([inputs["input_ids"][:, :prompt_len],
                          candidate_ids.unsqueeze(0)], dim=1)
    attn = torch.ones_like(full_ids)
    fwd_kwargs = {k: v for k, v in inputs.items()
                  if k not in ("input_ids", "attention_mask")}
    fwd_kwargs["input_ids"] = full_ids
    fwd_kwargs["attention_mask"] = attn

    # Forward 1: Normal (no masking)
    hooks.set_masking(False)
    with torch.no_grad():
        out_normal = model(**fwd_kwargs)
    # Logits at candidate token positions
    logits_normal = out_normal.logits[0, prompt_len - 1:prompt_len - 1 + n_cand].float()
    p_normal = F.softmax(logits_normal, dim=-1)
    log_p_normal = F.log_softmax(logits_normal, dim=-1)

    # Forward 2: Masked (vision heads zeroed/soft-masked)
    hooks.set_masking(True)
    with torch.no_grad():
        out_masked = model(**fwd_kwargs)
    logits_masked = out_masked.logits[0, prompt_len - 1:prompt_len - 1 + n_cand].float()
    log_p_masked = F.log_softmax(logits_masked, dim=-1)
    hooks.set_masking(False)

    # KL(P_normal || P_masked) per token
    # = sum_vocab p_normal * (log_p_normal - log_p_masked)
    per_token_kl = (p_normal * (log_p_normal - log_p_masked)).sum(dim=-1)
    # Clamp negative values (can occur from numerical issues)
    per_token_kl = torch.clamp(per_token_kl, min=0.0)

    kl_value = per_token_kl.mean().item() * kl_scale

    del out_normal, out_masked, logits_normal, logits_masked
    return kl_value, per_token_kl.detach()


# ══════════════════════════════════════════════════════════════════════
#  EXP6: LEARNED HEAD IMPORTANCE + GATED HEAD-LSR
# ══════════════════════════════════════════════════════════════════════

class VisionHeadHooksLSR:
    """Hook manager for Exp6: captures activations at ALL token positions
    for importance-weighted real-vs-black LSR computation.
    Hooks on ALL 28 layers (not just top-12) since importance is learned."""

    def __init__(self, model, head_importance, num_heads=16, head_dim=128):
        self.head_importance = head_importance  # nn.Parameter [28, 16]
        self.num_heads = num_heads
        self.head_dim = head_dim
        self._captured = {}
        self._hooks = []

        layers = model.model.language_model.layers
        num_layers = min(head_importance.shape[0], len(layers))
        self.layers_needed = list(range(num_layers))

        for li in self.layers_needed:
            o_proj = layers[li].self_attn.o_proj

            def make_hook(layer_idx):
                def hook_fn(module, args):
                    self._captured[layer_idx] = args[0].detach()
                return hook_fn

            handle = o_proj.register_forward_pre_hook(make_hook(li))
            self._hooks.append(handle)

    def get_per_token_head_acts(self, prompt_len, seq_len):
        """Extract per-token activations for ALL heads, all hooked layers.
        Returns: dict of (layer, head) -> (seq_len, head_dim)"""
        result = {}
        for li in self.layers_needed:
            inp = self._captured.get(li)
            if inp is None:
                continue
            reshaped = inp[0].view(-1, self.num_heads, self.head_dim)
            for h in range(self.num_heads):
                act = reshaped[prompt_len:prompt_len + seq_len, h, :]
                result[(li, h)] = act
        return result

    def clear(self):
        self._captured.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._captured.clear()


def compute_importance_weighted_lsr(model, processor, sample, candidate_ids,
                                     device, hooks_lsr, head_importance,
                                     lsr_scale=10.0):
    """Compute per-token vision score using importance-weighted head activation deltas.

    Like Phase 6 Head-LSR but:
    - Considers ALL 448 heads (not just top-12)
    - Weights each head's delta by sigmoid(head_importance[l,h])
    - Higher importance → head's delta counts more in the score

    score(t) = sum_lh(sigmoid(imp[l,h]) * ||act_real[l,h,t] - act_black[l,h,t]||_2)
             / sum_lh(sigmoid(imp[l,h]))

    Returns: head_scores (tensor), mean_score (float), n_tokens (int)
    """
    if candidate_ids.numel() == 0:
        return torch.zeros(0, device=device), 0.0, 0

    image = sample["image"]
    question = sample["question"]
    n_cand = candidate_ids.numel()

    # Forward with real image
    real_inputs = prepare_inputs(processor, image, question, device)
    rpl = real_inputs["input_ids"].shape[1]
    rf = torch.cat([real_inputs["input_ids"], candidate_ids.unsqueeze(0)], dim=1)
    real_inputs["input_ids"] = rf
    real_inputs["attention_mask"] = torch.ones_like(rf)
    hooks_lsr.clear()
    with torch.no_grad():
        model(**real_inputs)
    real_acts = hooks_lsr.get_per_token_head_acts(rpl, n_cand)

    # Forward with black image
    black_image = Image.new('RGB', image.size, (0, 0, 0))
    black_inputs = prepare_inputs(processor, black_image, question, device)
    bpl = black_inputs["input_ids"].shape[1]
    bf = torch.cat([black_inputs["input_ids"], candidate_ids.unsqueeze(0)], dim=1)
    black_inputs["input_ids"] = bf
    black_inputs["attention_mask"] = torch.ones_like(bf)
    hooks_lsr.clear()
    with torch.no_grad():
        model(**black_inputs)
    black_acts = hooks_lsr.get_per_token_head_acts(bpl, n_cand)

    # Importance-weighted delta scores
    imp_sigmoid = torch.sigmoid(head_importance).detach()  # [28, 16]
    scores = torch.zeros(n_cand, device=device)
    total_weight = 0.0

    for (l, h) in real_acts:
        if (l, h) not in black_acts:
            continue
        ra = real_acts[(l, h)]
        ba = black_acts[(l, h)]
        min_len = min(ra.shape[0], ba.shape[0], n_cand)
        if min_len == 0:
            continue

        diff = (ra[:min_len] - ba[:min_len]).float()
        head_delta = diff.norm(dim=-1)  # (min_len,)

        w = imp_sigmoid[l, h].item()
        scores[:min_len] += head_delta[:min_len] * w
        total_weight += w

    if total_weight > 0:
        scores /= total_weight

    mean_score = scores.mean().item()

    del real_inputs, black_inputs, real_acts, black_acts
    hooks_lsr.clear()
    return scores, mean_score, n_cand


def compute_rewards_learned_lsr(model, processor, sample, candidates,
                                 cand_ids_list, prompt_len, inputs,
                                 device, cfg, hooks_lsr, head_importance):
    """Compute R_correct + R_learned_lsr (Gated) for each candidate.

    Gated logic (same as Phase 6c Exp1):
    - If R_correct has variance → use correctness-only (targeted updates)
    - If zero variance → use importance-weighted head-LSR (grounding signal)
    """
    r_correct_list, r_lsr_list = [], []
    token_weights_list = []
    details = []
    gt = sample["answer"]
    qtype = sample.get("type", "short_answer")
    answers_all = sample.get("answers_all")

    for cand, cand_ids in zip(candidates, cand_ids_list):
        pred = extract_answer(cand, qtype)
        r_correct = compute_r_correct(pred, gt, qtype, answers_all=answers_all)

        try:
            head_scores, mean_score, n_tokens = compute_importance_weighted_lsr(
                model, processor, sample, cand_ids, device,
                hooks_lsr, head_importance, cfg.get("lsr_scale", 10.0))
        except Exception:
            head_scores = torch.zeros(0, device=device)
            mean_score, n_tokens = 0.0, 0

        r_lsr = min(mean_score / cfg.get("lsr_scale", 10.0), 1.0)
        r_correct_list.append(r_correct)
        r_lsr_list.append(r_lsr)

        # Per-token weights from head scores
        n_tok = cand_ids.numel()
        token_w = torch.ones(n_tok, device=device)
        if head_scores.numel() >= 5:
            ns = head_scores / (head_scores.mean() + 1e-6)
            ns = torch.clamp(ns, 0.0, 5.0)
            w_len = min(ns.numel(), n_tok)
            token_w[:w_len] = 1.0 + cfg.get("alpha", 0.5) * ns[:w_len]
        token_weights_list.append(token_w)

    # Gated GDPO normalization
    correct_has_variance = np.std(r_correct_list) > 1e-6

    if correct_has_variance:
        w_correct, w_lsr = 1.0, 0.0
        gate_mode = "correctness"
    else:
        w_correct, w_lsr = 0.0, 1.0
        gate_mode = "learned_lsr"

    if len(r_correct_list) > 1:
        norm_correct = gdpo_normalize(r_correct_list)
        norm_lsr = gdpo_normalize(r_lsr_list)
        combined = w_correct * norm_correct + w_lsr * norm_lsr
        rewards = combined.tolist()
    else:
        rewards = [r_correct_list[0]]

    # When gated to correctness, disable token weighting (standard GRPO)
    if correct_has_variance:
        token_weights_list = [torch.ones(c.numel(), device=device)
                              for c in cand_ids_list]

    for i in range(len(candidates)):
        details.append({
            "correct": r_correct_list[i],
            "learned_lsr": r_lsr_list[i] * cfg.get("lsr_scale", 10.0),
            "gate_mode": gate_mode,
            "token_weight_mean": token_weights_list[i].mean().item(),
        })

    return rewards, details, token_weights_list


# ══════════════════════════════════════════════════════════════════════
#  LEARNED HEAD IMPORTANCE INITIALIZATION
# ══════════════════════════════════════════════════════════════════════

def init_head_importance(num_layers=28, num_heads=16,
                         calibration_file=None, temperature=2.0):
    """Initialize head_importance tensor from Cohen's d scores.

    Returns: nn.Parameter of shape [28, 16] — raw logits (pre-sigmoid).
    Higher value → head is more important for vision → gets masked more in KL.

    Initialization: logit = inverse_sigmoid(softmax(d_score / temperature))
    This gives a smooth, differentiable starting point.
    """
    # Start with small random values (slight vision bias everywhere)
    importance = torch.zeros(num_layers, num_heads)

    # Load calibration scores if available
    scores = {}
    if calibration_file and os.path.exists(calibration_file):
        try:
            with open(calibration_file) as f:
                meta = json.load(f)
            scores = meta.get("head_scores", {})
            print(f"[importance] Loaded {len(scores)} head scores from calibration")
        except Exception as e:
            print(f"[importance] Calibration load failed: {e}")

    # Also use DEFAULT_VISION_HEADS as fallback
    if not scores:
        for l, h, d in DEFAULT_VISION_HEADS:
            scores[f"{l}_{h}"] = d

    # Fill importance matrix with d-scores
    d_values = []
    for key, d in scores.items():
        parts = key.split("_")
        l, h = int(parts[0]), int(parts[1])
        if l < num_layers and h < num_heads:
            d_values.append(d)

    if d_values:
        # Normalize: map d-scores to [0, 1] range via softmax-like scaling
        d_max = max(d_values)
        d_min = min(d_values) if len(d_values) > 1 else 0.0
        d_range = max(d_max - d_min, 1e-6)

        for key, d in scores.items():
            parts = key.split("_")
            l, h = int(parts[0]), int(parts[1])
            if l < num_layers and h < num_heads:
                # Scale to [0, 1] then to logit space
                normalized = (d - d_min) / d_range
                # Apply temperature: higher temp → softer distribution
                scaled = normalized / temperature
                # Convert to logit (inverse sigmoid): logit = log(p / (1-p))
                p = max(min(scaled, 0.95), 0.05)  # Clamp for numerical stability
                logit = np.log(p / (1 - p))
                importance[l, h] = logit

    param = nn.Parameter(importance)
    print(f"[importance] Initialized head_importance [{num_layers}, {num_heads}]")
    print(f"  Top-5 by importance (post-sigmoid):")
    imp_sigmoid = torch.sigmoid(importance)
    flat = imp_sigmoid.view(-1)
    topk = torch.topk(flat, min(5, flat.numel()))
    for val, idx in zip(topk.values, topk.indices):
        l = idx.item() // num_heads
        h = idx.item() % num_heads
        print(f"    L{l}H{h}: {val.item():.3f}")

    return param


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
    if answers_all:
        return textvqa_accuracy(prediction, answers_all)
    pred_tokens = set(pred.split())
    gt_tokens = set(gt.split())
    if not gt_tokens: return 0.0
    overlap = pred_tokens & gt_tokens
    if not overlap:
        if gt in pred or pred in gt: return 0.5
        return 0.0
    p = len(overlap) / len(pred_tokens)
    r = len(overlap) / len(gt_tokens)
    return 2 * p * r / (p + r)


# ══════════════════════════════════════════════════════════════════════
#  Rewards: Correctness + Head Masking KL
# ══════════════════════════════════════════════════════════════════════

def gdpo_normalize(values, eps=1e-8):
    arr = np.array(values)
    std = arr.std()
    if std < eps: return np.zeros_like(arr)
    return (arr - arr.mean()) / (std + eps)


def compute_rewards_headkl(model, processor, sample, candidates,
                           cand_ids_list, prompt_len, inputs,
                           device, cfg, hooks):
    """Compute R_correct + R_headKL for each candidate (GDPO-normalized).

    Exp4: Binary masking KL
    Exp6: Soft masking KL (learned head_importance)

    Returns: rewards, details, per_token_kl_list
    """
    r_correct_list, r_kl_list = [], []
    per_token_kl_list = []
    details = []
    gt = sample["answer"]
    qtype = sample.get("type", "short_answer")
    answers_all = sample.get("answers_all")

    for cand, cand_ids in zip(candidates, cand_ids_list):
        pred = extract_answer(cand, qtype)
        r_correct = compute_r_correct(pred, gt, qtype, answers_all=answers_all)

        try:
            kl_val, per_token_kl = compute_head_masking_kl(
                model, inputs, cand_ids, prompt_len, hooks, device,
                kl_scale=cfg.get("kl_scale", 1.0))
        except Exception as e:
            kl_val = 0.0
            per_token_kl = torch.zeros(0, device=device)

        r_correct_list.append(r_correct)
        r_kl_list.append(kl_val)
        per_token_kl_list.append(per_token_kl)

    # GDPO normalization
    use_gated = cfg.get("gated_head_kl", False)
    correct_has_variance = np.std(r_correct_list) > 1e-6
    kl_has_variance = np.std(r_kl_list) > 1e-6

    w_correct = cfg.get("gdpo_w_correct", 0.6)
    w_kl = cfg.get("gdpo_w_kl", 0.4)

    if use_gated:
        if correct_has_variance:
            w_correct, w_kl = 1.0, 0.0
        else:
            w_correct, w_kl = 0.0, 1.0

    if len(r_correct_list) > 1:
        norm_correct = gdpo_normalize(r_correct_list)
        norm_kl = gdpo_normalize(r_kl_list)
        combined = w_correct * norm_correct + w_kl * norm_kl
        rewards = combined.tolist()
    else:
        rewards = [r_correct_list[0]]

    gate_mode = "standard"
    if use_gated:
        gate_mode = "correctness" if correct_has_variance else "head_kl"

    for i in range(len(candidates)):
        details.append({
            "correct": r_correct_list[i],
            "head_kl": r_kl_list[i],
            "gate_mode": gate_mode,
            "kl_tokens": per_token_kl_list[i].numel(),
        })

    return rewards, details, per_token_kl_list


# ══════════════════════════════════════════════════════════════════════
#  GRPO Loss
# ══════════════════════════════════════════════════════════════════════

def compute_grpo_loss(model, inputs, cand_ids_list, prompt_len,
                      advantages, cfg):
    """Standard GRPO loss (no per-token weighting for simplicity)."""
    total_loss = torch.tensor(0.0, device=next(model.parameters()).device,
                              requires_grad=True)
    n_valid = 0
    stats = {"entropy": []}

    for cand_ids, adv in zip(cand_ids_list, advantages):
        if cand_ids.numel() == 0 or abs(adv) < 1e-8:
            continue

        full_ids = torch.cat([inputs["input_ids"][:, :prompt_len],
                              cand_ids.unsqueeze(0)], dim=1)
        attn = torch.ones_like(full_ids)
        fwd = {k: v for k, v in inputs.items()
               if k not in ("input_ids", "attention_mask")}
        fwd["input_ids"] = full_ids
        fwd["attention_mask"] = attn

        out = model(**fwd)
        logits = out.logits[0, prompt_len - 1:prompt_len - 1 + len(cand_ids)]
        log_probs = F.log_softmax(logits, dim=-1)
        token_lp = log_probs.gather(1, cand_ids.unsqueeze(1)).squeeze(1)

        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        weighted_lp = token_lp.mean()
        policy_loss = -weighted_lp * adv
        loss = policy_loss - cfg.get("beta_entropy", 0.01) * entropy

        total_loss = total_loss + loss
        n_valid += 1
        stats["entropy"].append(entropy.item())

    if n_valid > 0:
        total_loss = total_loss / n_valid
    return total_loss, stats


def compute_weighted_grpo_loss(model, inputs, cand_ids_list, prompt_len,
                                advantages, token_weights_list, cfg):
    """GRPO loss with per-token weighting (Exp6: importance-weighted LSR)."""
    total_loss = torch.tensor(0.0, device=next(model.parameters()).device,
                              requires_grad=True)
    n_valid = 0
    stats = {"entropy": []}

    for cand_ids, adv, tw in zip(cand_ids_list, advantages, token_weights_list):
        if cand_ids.numel() == 0 or abs(adv) < 1e-8:
            continue

        full_ids = torch.cat([inputs["input_ids"][:, :prompt_len],
                              cand_ids.unsqueeze(0)], dim=1)
        attn = torch.ones_like(full_ids)
        fwd = {k: v for k, v in inputs.items()
               if k not in ("input_ids", "attention_mask")}
        fwd["input_ids"] = full_ids
        fwd["attention_mask"] = attn

        out = model(**fwd)
        logits = out.logits[0, prompt_len - 1:prompt_len - 1 + len(cand_ids)]
        log_probs = F.log_softmax(logits, dim=-1)
        token_lp = log_probs.gather(1, cand_ids.unsqueeze(1)).squeeze(1)

        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        # Apply token weights
        w = tw[:len(token_lp)]
        if len(w) < len(token_lp):
            w = F.pad(w, (0, len(token_lp) - len(w)), value=1.0)
        weighted_lp = (token_lp * w).sum() / (w.sum() + 1e-8)

        policy_loss = -weighted_lp * adv
        loss = policy_loss - cfg.get("beta_entropy", 0.01) * entropy

        total_loss = total_loss + loss
        n_valid += 1
        stats["entropy"].append(entropy.item())

    if n_valid > 0:
        total_loss = total_loss / n_valid
    return total_loss, stats


# ══════════════════════════════════════════════════════════════════════
#  Evaluation
# ══════════════════════════════════════════════════════════════════════

def evaluate_textvqa(model, processor, samples, device, max_eval=50):
    was_training = model.training
    model.eval()
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    total_acc, total = 0.0, 0
    for s in samples[:max_eval]:
        try:
            inputs = prepare_inputs(processor, s["image"], s["question"], device)
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
    return {"acc": total_acc / max(total, 1), "total": total}

def evaluate_pope(model, processor, samples, device, max_eval=60):
    was_training = model.training
    model.eval()
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    correct, total = 0, 0
    for s in samples[:max_eval]:
        try:
            inputs = prepare_inputs(processor, s["image"], s["question"], device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            raw = processor.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
            pred = extract_yes_no(raw)
            if pred == s["answer"]: correct += 1
            total += 1
        except Exception:
            total += 1
    if was_training:
        model.train()
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
    return {"acc": correct / max(total, 1), "correct": correct, "total": total}

def evaluate_blind(model, processor, samples, device, n=50):
    was_training = model.training
    model.eval()
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    real_correct, blind_correct, total = 0, 0, 0
    for s in samples[:n]:
        try:
            # Real image
            inputs = prepare_inputs(processor, s["image"], s["question"], device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            raw = processor.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
            if extract_yes_no(raw) == s["answer"]: real_correct += 1

            # Black image
            black = Image.new('RGB', s["image"].size, (0, 0, 0))
            inputs_b = prepare_inputs(processor, black, s["question"], device)
            with torch.no_grad():
                out_b = model.generate(**inputs_b, max_new_tokens=256, do_sample=False)
            raw_b = processor.tokenizer.decode(
                out_b[0][inputs_b["input_ids"].shape[1]:], skip_special_tokens=False)
            if extract_yes_no(raw_b) == s["answer"]: blind_correct += 1
            total += 1
        except Exception:
            total += 1
    if was_training:
        model.train()
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
    real_acc = real_correct / max(total, 1)
    blind_acc = blind_correct / max(total, 1)
    return {"real_acc": real_acc, "blind_acc": blind_acc,
            "gap": real_acc - blind_acc, "total": total}


# ══════════════════════════════════════════════════════════════════════
#  Report Generation
# ══════════════════════════════════════════════════════════════════════

def generate_report(history, report_dir):
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    with open(report_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2, default=str)

    # Summary markdown
    cfg = history.get("config", {})
    exp_type = "Exp4 (Head Masking KL)" if not cfg.get("learned_importance") \
        else "Exp5 (Learned Head Importance + KL)"

    lines = [f"# Phase 7: {exp_type}", ""]
    lines.append(f"- Steps: {cfg.get('num_steps', '?')}")
    lines.append(f"- Group size: {cfg.get('group_size', '?')}")
    lines.append(f"- LR: {cfg.get('lr', '?')}")
    if cfg.get("learned_importance"):
        lines.append(f"- Importance LR: {cfg.get('importance_lr', '?')}")
        lines.append(f"- Importance temp: {cfg.get('importance_temp', '?')}")
    lines.append("")

    # Eval progression
    if history.get("evals"):
        lines.append("## Evaluation Progression")
        lines.append("| Step | TextVQA | POPE | Gap |")
        lines.append("|------|---------|------|-----|")
        pre = history.get("pre_eval", {})
        lines.append(f"| Pre | {pre.get('textvqa', {}).get('acc', 0):.1%} | "
                     f"{pre.get('pope', {}).get('acc', 0):.1%} | "
                     f"{pre.get('blind', {}).get('gap', 0):.1%} |")
        for e in history["evals"]:
            lines.append(f"| {e['step']} | {e['textvqa']['acc']:.1%} | "
                        f"{e['pope']['acc']:.1%} | {e['blind']['gap']:.1%} |")

    # Head KL stats
    if history.get("steps"):
        lines.append("")
        lines.append("## Head KL Statistics")
        kl_vals = [s.get("mean_head_kl", 0) for s in history["steps"]
                   if not s.get("skipped")]
        if kl_vals:
            lines.append(f"- Mean KL: {np.mean(kl_vals):.4f}")
            lines.append(f"- Max KL: {np.max(kl_vals):.4f}")
            lines.append(f"- Min KL: {np.min(kl_vals):.4f}")

    # Learned importance evolution (Exp6)
    if history.get("importance_evolution"):
        lines.append("")
        lines.append("## Learned Head Importance Evolution")
        for snap in history["importance_evolution"]:
            lines.append(f"\n### Step {snap['step']}")
            lines.append("Top-5 heads:")
            for l, h, v in snap["top5"]:
                lines.append(f"  - L{l}H{h}: {v:.3f}")

    with open(report_dir / "REPORT.md", "w") as f:
        f.write("\n".join(lines))
    print(f"[report] Saved to {report_dir}")


# ══════════════════════════════════════════════════════════════════════
#  Training Loop
# ══════════════════════════════════════════════════════════════════════

def run_training(cfg, train_data, eval_data, model_path=None,
                 textvqa_eval_data=None):
    output_dir = Path(cfg["output_dir"])
    exp_type = cfg.get("exp_type", "exp4")
    run_name = output_dir.name
    report_dir = Path("lab/reports/phase7") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"  Phase 7: Head Masking KL GRPO — {exp_type.upper()}")
    print(f"  steps={cfg['num_steps']} | group={cfg['group_size']} | "
          f"T={cfg['temperature']} | lr={cfg['lr']}")
    print(f"  GDPO weights: correct={cfg.get('gdpo_w_correct', 0.6)}, "
          f"kl={cfg.get('gdpo_w_kl', 0.4)}")
    if cfg.get("learned_importance"):
        print(f"  Learned importance: lr={cfg['importance_lr']}, "
              f"temp={cfg['importance_temp']}")
    print(f"  Vision heads: {len(cfg['vision_heads'])} heads")
    print(f"{'='*70}\n")

    model, processor, tokenizer = load_model(model_path, for_training=True)
    device = next(model.parameters()).device

    # Initialize head importance (Exp6)
    head_importance = None
    if cfg.get("learned_importance"):
        head_importance = init_head_importance(
            num_layers=28, num_heads=16,
            calibration_file=cfg.get("calibration_file"),
            temperature=cfg.get("importance_temp", 2.0))
        # Move to device without breaking leaf status
        head_importance.data = head_importance.data.to(device)

    # Install hooks
    is_exp6 = cfg.get("exp_type") == "exp6"
    hooks = None
    hooks_lsr = None

    if is_exp6:
        # Exp6: LSR hooks on all layers for importance-weighted delta
        hooks_lsr = VisionHeadHooksLSR(model, head_importance,
                                        num_heads=16, head_dim=128)
        print(f"[hooks] Exp6 LSR hooks on {len(hooks_lsr.layers_needed)} layers")
    else:
        # Exp4/5: Masking hooks for KL computation
        hooks = HeadMaskingHooks(model, cfg["vision_heads"],
                                 num_heads=16, head_dim=128,
                                 head_importance=head_importance)
        print(f"[hooks] Installed on {len(hooks.layers_needed)} layers, "
              f"masking={'soft' if head_importance is not None else 'binary'}")

    # Optimizer: model params + optional head_importance
    model_params = [p for p in model.parameters() if p.requires_grad and p.is_leaf]
    non_leaf = [p for p in model.parameters() if p.requires_grad and not p.is_leaf]
    if non_leaf:
        print(f"[optimizer] WARNING: {len(non_leaf)} non-leaf params excluded from optimizer")

    if head_importance is not None:
        print(f"[optimizer] head_importance: is_leaf={head_importance.is_leaf}, "
              f"requires_grad={head_importance.requires_grad}, "
              f"device={head_importance.device}, grad_fn={head_importance.grad_fn}")
        # Use separate optimizers to avoid param group issues
        optimizer = torch.optim.AdamW(model_params, lr=cfg["lr"], weight_decay=0.01)
        imp_optimizer = torch.optim.AdamW([head_importance],
                                           lr=cfg.get("importance_lr", 1e-3),
                                           weight_decay=0.0)
    else:
        optimizer = torch.optim.AdamW(model_params, lr=cfg["lr"], weight_decay=0.01)
        imp_optimizer = None

    # Pre-eval
    print("Pre-training eval...")
    pre_textvqa = evaluate_textvqa(model, processor, textvqa_eval_data or [], device, 50) \
        if textvqa_eval_data else {"acc": 0, "total": 0}
    pre_pope = evaluate_pope(model, processor, eval_data, device, 60)
    pre_blind = evaluate_blind(model, processor, eval_data, device, 50)
    print(f"  TextVQA: {pre_textvqa['acc']:.1%} | POPE: {pre_pope['acc']:.1%} | "
          f"Gap: {pre_blind['gap']:.1%}")

    history = {
        "config": {k: v for k, v in cfg.items() if k != "vision_heads"},
        "vision_heads": [(l, h, d) for l, h, d in cfg["vision_heads"]],
        "pre_eval": {"textvqa": pre_textvqa, "pope": pre_pope, "blind": pre_blind},
        "steps": [], "evals": [], "importance_evolution": [],
    }

    model.train()
    optimizer.zero_grad()
    if imp_optimizer is not None:
        imp_optimizer.zero_grad()
    best_pope = pre_pope["acc"]

    for step in range(cfg["num_steps"]):
        step_t0 = time.time()
        sample = train_data[step % len(train_data)]

        # Generate candidates
        model.eval()
        if hasattr(model, 'gradient_checkpointing_disable'):
            model.gradient_checkpointing_disable()
        try:
            candidates, cand_ids_list, prompt_len, inputs = \
                generate_candidates(
                    model, processor, sample, cfg["group_size"],
                    cfg["temperature"], cfg["top_p"],
                    cfg["max_new_tokens"], cfg.get("min_think_tokens", 32), device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); gc.collect()
            print(f"  [step {step+1}] OOM gen, skip"); continue

        # Rewards computation (Exp4/5: KL, Exp6: learned LSR)
        token_weights_list = None
        per_token_kl_list = None
        try:
            if is_exp6:
                rewards, details, token_weights_list = compute_rewards_learned_lsr(
                    model, processor, sample, candidates, cand_ids_list,
                    prompt_len, inputs, device, cfg, hooks_lsr, head_importance)
            else:
                rewards, details, per_token_kl_list = compute_rewards_headkl(
                    model, processor, sample, candidates, cand_ids_list,
                    prompt_len, inputs, device, cfg, hooks)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); gc.collect()
            print(f"  [step {step+1}] OOM reward, skip"); continue

        rarr = np.array(rewards)
        rstd = rarr.std()

        # Dynamic resampling on zero variance
        if rstd < 1e-8:
            resample_tries = 0
            while rstd < 1e-8 and resample_tries < 5:
                resample_tries += 1
                alt_idx = (step * 7 + resample_tries * 13) % len(train_data)
                alt_sample = train_data[alt_idx]
                try:
                    candidates, cand_ids_list, prompt_len, inputs = \
                        generate_candidates(
                            model, processor, alt_sample, cfg["group_size"],
                            cfg["temperature"], cfg["top_p"],
                            cfg["max_new_tokens"], cfg.get("min_think_tokens", 32),
                            device)
                    if is_exp6:
                        rewards, details, token_weights_list = compute_rewards_learned_lsr(
                            model, processor, alt_sample, candidates, cand_ids_list,
                            prompt_len, inputs, device, cfg, hooks_lsr, head_importance)
                    else:
                        rewards, details, per_token_kl_list = compute_rewards_headkl(
                            model, processor, alt_sample, candidates, cand_ids_list,
                            prompt_len, inputs, device, cfg, hooks)
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

        # Advantages
        advantages = ((rarr - rarr.mean()) / (rstd + 1e-8)).tolist()

        # GRPO loss (Exp6 uses token-weighted loss via compute_weighted_grpo_loss)
        model.train()
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        try:
            if is_exp6 and token_weights_list is not None:
                loss, lstats = compute_weighted_grpo_loss(
                    model, inputs, cand_ids_list, prompt_len,
                    advantages, token_weights_list, cfg)
            else:
                loss, lstats = compute_grpo_loss(
                    model, inputs, cand_ids_list, prompt_len, advantages, cfg)
            (loss / cfg.get("grad_accum", 2)).backward()

            if (step + 1) % cfg.get("grad_accum", 2) == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.get("max_grad_norm", 1.0))
                # Also clip head_importance gradient if exists
                if head_importance is not None and head_importance.grad is not None:
                    torch.nn.utils.clip_grad_norm_([head_importance], 1.0)
                optimizer.step()
                optimizer.zero_grad()
                if imp_optimizer is not None:
                    imp_optimizer.step()
                    imp_optimizer.zero_grad()
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad()
            if imp_optimizer is not None:
                imp_optimizer.zero_grad()
            torch.cuda.empty_cache(); gc.collect()
            print(f"  [step {step+1}] OOM loss, skip"); continue

        elapsed = time.time() - step_t0
        mc = np.mean([d["correct"] for d in details])
        gate = details[0].get("gate_mode", "standard") if details else "standard"

        # Exp6 logs LSR score, Exp4/5 logs KL
        if is_exp6:
            metric_val = np.mean([d.get("learned_lsr", 0) for d in details])
            metric_name = "learnedLSR"
        else:
            metric_val = np.mean([d.get("head_kl", 0) for d in details])
            metric_name = "headKL"

        step_info = {
            "step": step + 1, "loss": loss.item(),
            "mean_reward": float(rarr.mean()),
            "reward_std": float(rstd),
            "mean_correct": float(mc),
            "mean_head_kl": float(metric_val),
            "gate_mode": gate,
            "elapsed": elapsed,
        }

        # Log learned importance stats (Exp6)
        if head_importance is not None:
            imp_sig = torch.sigmoid(head_importance).detach()
            step_info["importance_mean"] = imp_sig.mean().item()
            step_info["importance_max"] = imp_sig.max().item()
            step_info["importance_std"] = imp_sig.std().item()

            # Track evolution every 5 steps
            if (step + 1) % 5 == 0:
                flat = imp_sig.view(-1)
                topk = torch.topk(flat, 5)
                top5 = []
                for val, idx in zip(topk.values, topk.indices):
                    l = idx.item() // 16
                    h = idx.item() % 16
                    top5.append((l, h, val.item()))
                history["importance_evolution"].append({
                    "step": step + 1,
                    "top5": top5,
                    "mean": imp_sig.mean().item(),
                    "std": imp_sig.std().item(),
                })

        history["steps"].append(step_info)

        imp_str = ""
        if head_importance is not None:
            imp_str = f" imp={step_info['importance_mean']:.3f}±{step_info['importance_std']:.3f}"

        print(f"  [step {step+1}/{cfg['num_steps']}] "
              f"loss={loss.item():.4f} r={rarr.mean():.3f}±{rstd:.3f} "
              f"correct={mc:.2f} {metric_name}={metric_val:.4f} "
              f"gate={gate}{imp_str} ({elapsed:.1f}s)", flush=True)

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

            if pope_res["acc"] >= best_pope:
                best_pope = pope_res["acc"]
                model.save_pretrained(output_dir / "best")
                processor.save_pretrained(output_dir / "best")
                if head_importance is not None:
                    torch.save(head_importance.data,
                              output_dir / "best" / "head_importance.pt")
                print(f"  ★ New best POPE: {best_pope:.1%}")

        del candidates, cand_ids_list, inputs
        if per_token_kl_list is not None:
            del per_token_kl_list
        if token_weights_list is not None:
            del token_weights_list

    # Cleanup
    if hooks is not None:
        hooks.remove()
    if hooks_lsr is not None:
        hooks_lsr.remove()

    # Final save
    model.save_pretrained(output_dir / "final")
    processor.save_pretrained(output_dir / "final")
    if head_importance is not None:
        torch.save(head_importance.data,
                  output_dir / "final" / "head_importance.pt")
        # Save readable importance matrix
        imp_sig = torch.sigmoid(head_importance).detach().cpu()
        imp_dict = {}
        for l in range(imp_sig.shape[0]):
            for h in range(imp_sig.shape[1]):
                v = imp_sig[l, h].item()
                if v > 0.1:  # Only save non-trivial
                    imp_dict[f"L{l}H{h}"] = round(v, 4)
        with open(output_dir / "final" / "head_importance_readable.json", "w") as f:
            json.dump(imp_dict, f, indent=2, sort_keys=True)

    history["final_eval"] = history["evals"][-1] if history["evals"] else {}
    generate_report(history, report_dir)

    # Summary
    pre = history["pre_eval"]
    final = history.get("final_eval", {})
    print(f"\n{'='*70}")
    print(f"  Phase 7 COMPLETE ({exp_type})")
    print(f"  TextVQA: {pre['textvqa']['acc']:.1%} → "
          f"{final.get('textvqa', {}).get('acc', 0):.1%}")
    print(f"  POPE: {pre['pope']['acc']:.1%} → "
          f"{final.get('pope', {}).get('acc', 0):.1%}")
    print(f"  Gap: {pre['blind']['gap']:.1%} → "
          f"{final.get('blind', {}).get('gap', 0):.1%}")
    print(f"  Best POPE: {best_pope:.1%}")
    print(f"  Saved to {output_dir}")
    print(f"{'='*70}")

    del model; torch.cuda.empty_cache(); gc.collect()
    return history


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Phase 7: Head Masking KL + Learned Head Importance")
    parser.add_argument("--exp", type=int, default=4, choices=[4, 5, 6],
                        help="Experiment: 4=Head Masking KL, 5=Learned Importance+KL, "
                             "6=Learned Importance+Gated LSR")
    parser.add_argument("--model-path", type=str,
                        default="Qwen/Qwen3-VL-2B-Thinking")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output dir (default: auto based on --exp)")
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--group-size", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=1.3)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--beta-entropy", type=float, default=0.01)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--min-think-tokens", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--train-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k-heads", type=int, default=12)
    parser.add_argument("--calibration-file", type=str,
                        default="checkpoints/calibration/qwen3_vl_2b/calibration_meta.json")

    # GDPO
    parser.add_argument("--gdpo", action="store_true",
                        help="GDPO: normalize rewards independently")
    parser.add_argument("--gdpo-w-correct", type=float, default=0.6,
                        help="GDPO weight for R_correct")
    parser.add_argument("--gdpo-w-kl", type=float, default=0.4,
                        help="GDPO weight for R_headKL")
    parser.add_argument("--gated-head-kl", action="store_true",
                        help="Gate: correctness-only when variance, else headKL")

    # KL scaling
    parser.add_argument("--kl-scale", type=float, default=1.0,
                        help="Scale factor for KL reward")

    # Exp5/6: Learned head importance
    parser.add_argument("--learned-importance", action="store_true",
                        help="Learn head importance weights (Exp5/6)")
    parser.add_argument("--importance-lr", type=float, default=1e-3,
                        help="Learning rate for head_importance")
    parser.add_argument("--importance-temp", type=float, default=2.0,
                        help="Temperature for importance initialization")

    # Exp6: LSR-specific params
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Head-level weight scale for token weighting (Exp6)")
    parser.add_argument("--lsr-scale", type=float, default=10.0,
                        help="Head score normalization scale (Exp6)")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Auto output dir
    if args.output_dir is None:
        args.output_dir = f"checkpoints/phase7/exp{args.exp}"

    # Force learned_importance for exp6
    if args.exp in (5, 6):
        args.learned_importance = True

    # Load vision heads
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
    cfg["exp_type"] = f"exp{args.exp}"

    train_data = load_training_data(args.train_samples, args.seed)
    pope_eval_data = load_pope_eval(300)
    textvqa_eval_data = load_textvqa_eval(200)

    run_training(cfg, train_data, pope_eval_data, args.model_path,
                 textvqa_eval_data=textvqa_eval_data)


if __name__ == "__main__":
    main()
