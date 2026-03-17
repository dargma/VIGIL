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

POPE_SPLITS = ["random", "popular", "adversarial"]
PROJECT_ROOT = Path(__file__).parent.parent

# ══════════════════════════════════════════════════════════════════════
#  Multi-Model Configuration
# ══════════════════════════════════════════════════════════════════════

MODEL_CONFIGS = {
    "qwen3_vl_2b": {
        "hf_id": "Qwen/Qwen3-VL-2B-Thinking",
        "model_class": "Qwen3VLForConditionalGeneration",
        "processor_class": "AutoProcessor",
        "num_layers": 28, "num_heads": 16, "num_kv_heads": 8, "head_dim": 128,
        "hidden_size": 2048, "gqa": True,
        "layer_path": "model.language_model.layers",
        "input_api": "qwen_vl",  # uses qwen_vl_utils
        "thinking": True,
        "trust_remote_code": True,
        "calibration_file": "checkpoints/calibration/qwen3_vl_2b/calibration_meta.json",
    },
    "internvl3_5_1b": {
        "hf_id": "OpenGVLab/InternVL3_5-1B",
        "model_class": "AutoModel",
        "processor_class": None,  # uses tokenizer + TRANSFORM
        "num_layers": 28, "num_heads": 16, "num_kv_heads": 8, "head_dim": 128,
        "hidden_size": 1024, "gqa": True,
        "layer_path": "language_model.model.layers",
        "input_api": "internvl",  # uses model.chat()
        "thinking": False,
        "trust_remote_code": True,
        "calibration_file": "checkpoints/calibration/internvl3_5_1b/calibration_meta.json",
    },
    "deepseek_vl2_tiny": {
        "hf_id": "deepseek-ai/deepseek-vl2-tiny",
        "model_class": "AutoModelForCausalLM",
        "processor_class": None,
        "num_layers": 12, "num_heads": 10, "num_kv_heads": 10, "head_dim": 256,
        "hidden_size": 2560, "gqa": False,
        "layer_path": "model.layers",
        "input_api": "deepseek_vl2",
        "thinking": False,
        "trust_remote_code": True,
        "max_temperature": 0.7,
        "calibration_file": "checkpoints/calibration/deepseek_vl2_tiny/calibration_meta.json",
    },
}

# Default (backwards compat)
HF_ID = MODEL_CONFIGS["qwen3_vl_2b"]["hf_id"]
ACTIVE_MODEL_KEY = "qwen3_vl_2b"
ACTIVE_MODEL_CFG = MODEL_CONFIGS["qwen3_vl_2b"]

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

def load_mme_train_data(max_samples=500, eval_reserve=200, seed=42):
    """Load MME data for training, excluding first eval_reserve question_ids (reserved for eval)."""
    from datasets import load_from_disk
    rng = random.Random(seed)
    mme_path = Path("data/eval/mme")
    if not mme_path.exists():
        print("[data] No MME data found, skipping MME train")
        return []

    ds = load_from_disk(str(mme_path))

    # Group by question_id (same as evaluate_mme)
    grouped = defaultdict(list)
    for i in range(len(ds)):
        row = ds[i]
        grouped[row["question_id"]].append(row)

    # Reserve first eval_reserve question_ids for eval (matches evaluate_mme ordering)
    all_qids = list(grouped.keys())
    eval_qids = set(all_qids[:eval_reserve])
    train_qids = [qid for qid in all_qids if qid not in eval_qids]

    samples = []
    for qid in train_qids:
        for row in grouped[qid]:
            img = row.get("image")
            if img is None:
                continue
            ans = row["answer"].strip()
            # Rewrite prompt: encourage reasoning about visual content
            # Original: "Is X? Please answer yes or no."
            # Updated: "Look at the image carefully. Is X? Think step by step, then answer yes or no."
            q = row["question"]
            q = q.replace("Please answer yes or no.",
                          "Think step by step about what you see in the image, then answer yes or no.")
            samples.append({
                "question": q,
                "answer": ans, "image": img,
                "answers_all": [ans],
                "type": "yesno", "source": "mme",
            })

    rng.shuffle(samples)
    samples = samples[:max_samples]
    print(f"[data] MME train: {len(samples)} samples "
          f"(excluded {len(eval_qids)} eval question_ids)")
    return samples


def load_training_data(limit=500, seed=42, include_mme=False, mme_ratio=0.3,
                       mme_eval_reserve=200):
    """Load TextVQA train + optional MME train data."""
    from datasets import load_dataset
    rng = random.Random(seed)
    samples = []

    # Compute per-source limits
    if include_mme:
        mme_limit = int(limit * mme_ratio)
        textvqa_limit = limit - mme_limit
    else:
        textvqa_limit = limit
        mme_limit = 0

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
            if count >= textvqa_limit * 2: break  # Load extra, shuffle, then trim
    except Exception as e:
        print(f"  TextVQA error: {e}")

    rng.shuffle(samples)
    samples = samples[:textvqa_limit]

    # Add MME train data
    if include_mme and mme_limit > 0:
        mme_samples = load_mme_train_data(mme_limit, mme_eval_reserve, seed)
        samples.extend(mme_samples)
        rng.shuffle(samples)

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

def load_model(model_path=None, for_training=True, model_key=None):
    global ACTIVE_MODEL_KEY, ACTIVE_MODEL_CFG, HF_ID
    mcfg = MODEL_CONFIGS.get(model_key or ACTIVE_MODEL_KEY)
    ACTIVE_MODEL_KEY = model_key or ACTIVE_MODEL_KEY
    ACTIVE_MODEL_CFG = mcfg
    HF_ID = mcfg["hf_id"]

    path = model_path or mcfg["hf_id"]
    api = mcfg["input_api"]
    print(f"[model] Loading {path} ({ACTIVE_MODEL_KEY}, full finetune, bfloat16)...")

    if api == "qwen_vl":
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(mcfg["hf_id"], trust_remote_code=True)
        tokenizer = processor.tokenizer
    elif api == "internvl":
        from transformers import AutoModel, AutoTokenizer
        model = AutoModel.from_pretrained(
            path, torch_dtype=torch.bfloat16,
            trust_remote_code=True).cuda().eval()
        tokenizer = AutoTokenizer.from_pretrained(mcfg["hf_id"], trust_remote_code=True)
        processor = None
    elif api == "deepseek_vl2":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(mcfg["hf_id"], trust_remote_code=True)
        processor = None
    else:
        raise ValueError(f"Unknown input_api: {api}")

    if for_training:
        model.train()
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        for p in model.parameters(): p.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable: {trainable:,} params, gradient checkpointing ON")
    return model, processor, tokenizer


# ══════════════════════════════════════════════════════════════════════
#  Input Preparation & Generation
# ══════════════════════════════════════════════════════════════════════

def prepare_inputs(processor, image, question, device, tokenizer=None):
    api = ACTIVE_MODEL_CFG["input_api"]

    if api == "qwen_vl":
        from qwen_vl_utils import process_vision_info
        content = [{"type": "image", "image": image},
                   {"type": "text", "text": question}]
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=ACTIVE_MODEL_CFG.get("thinking", False))
        imgs, _, _ = process_vision_info(messages, return_video_kwargs=True)
        inputs = processor(text=[text], images=imgs, return_tensors="pt", padding=True)
        return {k: v.to(device) for k, v in inputs.items()}

    elif api == "internvl":
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        pixel_values = transform(image).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        query = f"<image>\n{question}"
        text_inputs = tokenizer(query, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in text_inputs.items()}
        inputs["pixel_values"] = pixel_values
        return inputs

    elif api == "deepseek_vl2":
        # DeepSeek-VL2 uses its own preprocessing
        from deepseek_vl2.utils.io import load_pil_images
        conversation = [{"role": "user", "content": f"<image>\n{question}",
                         "images": [image]}]
        text_inputs = tokenizer(question, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in text_inputs.items()}
        # Note: DeepSeek image processing may need adaptation
        return inputs

    raise ValueError(f"Unknown input_api: {api}")

def generate_candidates(model, processor, sample, group_size, temperature,
                        top_p, max_new_tokens, min_think_tokens, device,
                        tokenizer=None):
    question = sample["question"]
    image = sample["image"]
    tok = tokenizer or (processor.tokenizer if processor else None)
    api = ACTIVE_MODEL_CFG["input_api"]

    # Clamp temperature for models with max_temperature constraint
    max_temp = ACTIVE_MODEL_CFG.get("max_temperature")
    if max_temp and temperature > max_temp:
        temperature = max_temp

    inputs = prepare_inputs(processor, image, question, device, tokenizer=tok)
    prompt_len = inputs["input_ids"].shape[1]
    candidates, candidate_ids_list, think_ranges = [], [], []

    for _ in range(group_size):
        try:
            if api == "internvl" and hasattr(model, 'chat'):
                # InternVL uses model.chat() for generation
                import torchvision.transforms as T
                from torchvision.transforms.functional import InterpolationMode
                IMAGENET_MEAN = (0.485, 0.456, 0.406)
                IMAGENET_STD = (0.229, 0.224, 0.225)
                transform = T.Compose([
                    T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                    T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ])
                pv = transform(image).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
                with torch.no_grad():
                    response = model.chat(tok, pv, question,
                                          generation_config={"max_new_tokens": max_new_tokens,
                                                             "temperature": temperature,
                                                             "do_sample": True})
                candidates.append(response.strip())
                # Encode response to get candidate_ids
                gen_ids = tok.encode(response, return_tensors="pt")[0].to(device)
                candidate_ids_list.append(gen_ids.detach())
                think_ranges.append((0, len(gen_ids)))  # No thinking tags
            else:
                with torch.no_grad():
                    gen_kwargs = dict(
                        **inputs, max_new_tokens=max_new_tokens,
                        temperature=temperature, top_p=top_p, do_sample=True)
                    if ACTIVE_MODEL_CFG.get("thinking", False):
                        gen_kwargs["min_new_tokens"] = min_think_tokens
                    out = model.generate(**gen_kwargs)
                gen_ids = out[0][prompt_len:].clone()
                text = tok.decode(gen_ids, skip_special_tokens=False)
                for special in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
                    text = text.replace(special, "")
                candidates.append(text.strip())
                candidate_ids_list.append(gen_ids.detach())
                if ACTIVE_MODEL_CFG.get("thinking", False):
                    think_ranges.append(find_think_token_range(tok, gen_ids))
                else:
                    think_ranges.append((0, len(gen_ids)))
        except Exception as e:
            candidates.append("")
            candidate_ids_list.append(torch.tensor([], dtype=torch.long, device=device))
            think_ranges.append((0, 0))
    inputs = {k: v.clone() if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}
    return candidates, candidate_ids_list, prompt_len, inputs, think_ranges


def _get_model_layers(model):
    """Navigate to model layers using ACTIVE_MODEL_CFG layer_path."""
    path = ACTIVE_MODEL_CFG["layer_path"]
    obj = model
    for attr in path.split("."):
        obj = getattr(obj, attr)
    return obj


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

        # Install hooks — navigate to layers via model config layer_path
        layers = _get_model_layers(model)
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


class AdaptiveVisionHeadHooks:
    """Hook ALL 28 layers to capture ALL 448 heads for per-rollout adaptive selection.

    Unlike VisionHeadHooks (fixed 12 heads on 7 layers), this captures everything
    so we can select top-K heads per sample based on real-vs-black activation delta.
    Zero extra cost: LSR already does real/black forward passes.
    """

    def __init__(self, model, num_layers=28, num_heads=16, head_dim=128):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self._captured = {}
        self._hooks = []

        layers = _get_model_layers(model)
        for li in range(num_layers):
            o_proj = layers[li].self_attn.o_proj

            def make_hook(layer_idx):
                def hook_fn(module, args):
                    self._captured[layer_idx] = args[0].detach()
                return hook_fn

            handle = o_proj.register_forward_pre_hook(make_hook(li))
            self._hooks.append(handle)

    def get_all_head_acts(self, prompt_len, seq_len):
        """Get activations for ALL heads across ALL layers.
        Returns: dict of (layer, head) -> (seq_len, head_dim)
        """
        result = {}
        for li in range(self.num_layers):
            inp = self._captured.get(li)
            if inp is None:
                continue
            reshaped = inp[0].view(-1, self.num_heads, self.head_dim)
            for hi in range(self.num_heads):
                cand_acts = reshaped[prompt_len:prompt_len + seq_len, hi, :]
                result[(li, hi)] = cand_acts
        return result

    # Also support the fixed-head interface for compatibility
    def get_per_token_head_acts(self, prompt_len, seq_len):
        return self.get_all_head_acts(prompt_len, seq_len)

    def clear(self):
        self._captured.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._captured.clear()


def compute_adaptive_head_lsr(model, processor, sample, candidate_ids,
                              think_range, device, hooks, top_k=12):
    """Per-rollout adaptive head selection + LSR scoring.

    Like compute_head_level_lsr but selects top-K heads PER SAMPLE based on
    real-vs-black activation delta across ALL 448 heads.

    Returns:
        head_scores: tensor of shape (think_len,) — per-token head activation Δ
        mean_score: float — sequence-level mean
        think_len: int
        selected_heads: list of (layer, head, mean_delta) — top-K heads for this sample
    """
    if candidate_ids.numel() == 0:
        return torch.zeros(0, device=device), 0.0, 0, []

    t_start, t_end = think_range
    if t_end <= t_start:
        return torch.zeros(0, device=device), 0.0, 0, []

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
    real_acts = hooks.get_all_head_acts(rpl, n_cand)

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
    black_acts = hooks.get_all_head_acts(bpl, n_cand)

    # Compute per-head mean delta across thinking token positions
    t_end_safe = min(t_end, n_cand)
    t_start_safe = min(t_start, t_end_safe)
    think_len = t_end_safe - t_start_safe

    if think_len <= 0:
        del real_inputs, black_inputs
        hooks.clear()
        return torch.zeros(0, device=device), 0.0, 0, []

    head_deltas = {}  # (layer, head) -> mean delta
    for (l, h) in real_acts:
        if (l, h) not in black_acts:
            continue
        ra = real_acts[(l, h)]
        ba = black_acts[(l, h)]
        min_len = min(ra.shape[0], ba.shape[0], t_end_safe)
        if min_len <= t_start_safe:
            continue
        diff = (ra[t_start_safe:min_len] - ba[t_start_safe:min_len]).float()
        mean_delta = diff.norm(dim=-1).mean().item()
        head_deltas[(l, h)] = mean_delta

    # Select top-K heads by mean delta for this sample
    sorted_heads = sorted(head_deltas.items(), key=lambda x: x[1], reverse=True)
    selected = sorted_heads[:top_k]
    selected_heads = [(l, h, d) for (l, h), d in selected]

    # Compute per-token scores using only selected heads (weighted by delta)
    scores = torch.zeros(think_len, device=device)
    n_heads_found = 0

    for (l, h), delta in selected:
        ra = real_acts[(l, h)]
        ba = black_acts[(l, h)]
        min_len = min(ra.shape[0], ba.shape[0], t_end_safe)
        if min_len <= t_start_safe:
            continue
        diff = (ra[t_start_safe:min_len] - ba[t_start_safe:min_len]).float()
        per_token_delta = diff.norm(dim=-1)

        effective_len = min(per_token_delta.shape[0], think_len)
        scores[:effective_len] += per_token_delta[:effective_len] * delta
        n_heads_found += 1

    if n_heads_found > 0:
        scores /= n_heads_found

    mean_score = scores.mean().item()

    del real_inputs, black_inputs, real_acts, black_acts
    hooks.clear()
    return scores, mean_score, think_len, selected_heads


def compute_soft_weighted_head_lsr(model, processor, sample, candidate_ids,
                                    think_range, device, hooks,
                                    temperature="auto", temp_scale=1.0,
                                    layer_aware=False, top_p_heads=0.0):
    """Exp9: Soft-weighted ALL-head LSR scoring with continuous sigmoid weights.

    Instead of top-K discrete selection (Exp8), ALL 448 heads contribute with
    sigmoid-based weights derived from their real-vs-black activation delta:
        w(l,h) = sigmoid((delta(l,h) - mean_delta) / T)

    This captures:
    - High-delta heads → weight ≈ 1.0 (strong vision signal)
    - Low-delta heads → weight ≈ 0.0 (text-only, negligible contribution)
    - Mixed heads (delta ≈ mean) → weight ≈ 0.5 (dual text+vision use)
    - Suppression heads (low delta but non-zero) → small but non-zero weight

    Temperature is adaptive: T = std(deltas), ensuring weights scale with the
    distribution of deltas per sample.

    Returns:
        scores: tensor (think_len,) — weighted per-token activation Δ
        mean_score: float — sequence-level mean
        think_len: int
        head_weights: dict of {(layer, head): weight} — continuous weights for all heads
    """
    if candidate_ids.numel() == 0:
        return torch.zeros(0, device=device), 0.0, 0, {}

    t_start, t_end = think_range
    if t_end <= t_start:
        return torch.zeros(0, device=device), 0.0, 0, {}

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
    real_acts = hooks.get_all_head_acts(rpl, n_cand)

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
    black_acts = hooks.get_all_head_acts(bpl, n_cand)

    # Compute per-head mean delta
    t_end_safe = min(t_end, n_cand)
    t_start_safe = min(t_start, t_end_safe)
    think_len = t_end_safe - t_start_safe

    if think_len <= 0:
        del real_inputs, black_inputs
        hooks.clear()
        return torch.zeros(0, device=device), 0.0, 0, {}

    head_deltas = {}
    for (l, h) in real_acts:
        if (l, h) not in black_acts:
            continue
        ra = real_acts[(l, h)]
        ba = black_acts[(l, h)]
        min_len = min(ra.shape[0], ba.shape[0], t_end_safe)
        if min_len <= t_start_safe:
            continue
        diff = (ra[t_start_safe:min_len] - ba[t_start_safe:min_len]).float()
        mean_delta = diff.norm(dim=-1).mean().item()
        head_deltas[(l, h)] = mean_delta

    if not head_deltas:
        del real_inputs, black_inputs, real_acts, black_acts
        hooks.clear()
        return torch.zeros(think_len, device=device), 0.0, think_len, {}

    # Compute sigmoid weights for ALL heads
    all_deltas = np.array(list(head_deltas.values()))
    mean_d = float(all_deltas.mean())
    std_d = float(all_deltas.std()) + 1e-6

    if temperature == "auto":
        T = std_d * temp_scale  # adaptive, scaled by temp_scale
    else:
        T = float(temperature) * temp_scale
    T = max(T, 1e-6)  # prevent division by zero

    # Layer-aware bonuses (Exp11)
    DECISION_LAYERS = {4, 5}
    FEATURE_LAYERS = {24, 25, 26, 27}

    head_weights = {}
    for (l, h), delta in head_deltas.items():
        w = 1.0 / (1.0 + np.exp(-(delta - mean_d) / T))
        # Exp11: layer-aware bonus
        if layer_aware:
            if l in DECISION_LAYERS:
                w *= 2.0  # decision heads get 2× weight
            elif l in FEATURE_LAYERS:
                w *= 1.5  # feature heads get 1.5× weight
        head_weights[(l, h)] = w

    # Exp12: Top-P head selection (soft pruning)
    if top_p_heads > 0:
        sorted_items = sorted(head_weights.items(), key=lambda x: x[1], reverse=True)
        total_w = sum(w for _, w in sorted_items)
        if total_w > 0:
            cumsum = 0.0
            threshold_idx = len(sorted_items)
            target = total_w * top_p_heads
            for idx, (key, w) in enumerate(sorted_items):
                cumsum += w
                if cumsum >= target:
                    threshold_idx = idx + 1
                    break
            # Zero-out heads below threshold
            kept_keys = set(k for k, _ in sorted_items[:threshold_idx])
            for key in list(head_weights.keys()):
                if key not in kept_keys:
                    head_weights[key] = 0.0

    # Per-token score: weighted sum over all heads with w > 0.01
    scores = torch.zeros(think_len, device=device)
    total_weight = 0.0
    n_active = 0

    for (l, h), w in head_weights.items():
        if w < 0.01:
            continue
        ra = real_acts[(l, h)]
        ba = black_acts[(l, h)]
        min_len = min(ra.shape[0], ba.shape[0], t_end_safe)
        if min_len <= t_start_safe:
            continue
        diff = (ra[t_start_safe:min_len] - ba[t_start_safe:min_len]).float()
        per_token_delta = diff.norm(dim=-1)

        effective_len = min(per_token_delta.shape[0], think_len)
        scores[:effective_len] += per_token_delta[:effective_len] * w
        total_weight += w
        n_active += 1

    if total_weight > 0:
        scores /= total_weight

    mean_score = scores.mean().item()

    del real_inputs, black_inputs, real_acts, black_acts
    hooks.clear()
    return scores, mean_score, think_len, head_weights


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
    use_gated = cfg.get("gated_head_lsr", False)

    # Gated Head-LSR: check if R_correct has variance
    correct_has_variance = np.std(r_correct_list) > 1e-6

    if use_gdpo and len(r_correct_list) > 1:
        # Independent normalization (GDPO arXiv:2601.05242)
        w_correct = cfg.get("gdpo_w_correct", 0.6)
        w_lsr = cfg.get("gdpo_w_lsr", 0.4)

        if use_gated:
            if correct_has_variance:
                # Correctness signal exists → use it alone (targeted updates)
                w_correct, w_lsr = 1.0, 0.0
            else:
                # Zero variance → fall back to head-LSR grounding signal
                w_correct, w_lsr = 0.0, 1.0

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

        # Gated Head-LSR: only use head weighting when correctness has no signal
        alpha_effective = cfg["alpha"]
        if use_gated:
            alpha_effective = 0.0 if correct_has_variance else cfg.get("gated_alpha", cfg["alpha"])

        if head_scores.numel() >= 10 and alpha_effective > 0:
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
                        mask * (1.0 + alpha_effective * norm_scores[:w_len]))
                else:
                    token_w[t_start_safe:t_start_safe + w_len] = (
                        1.0 + alpha_effective * norm_scores[:w_len])

        gate_mode = "standard"
        if use_gated:
            gate_mode = "correctness" if correct_has_variance else "head_lsr"
        details.append({
            "correct": r_correct_list[i], "head_score_raw": r_lsr_list[i] * cfg["lsr_scale"],
            "decay_penalty": decay_pen,
            "token_weight_mean": token_w.mean().item(),
            "token_weight_max": token_w.max().item(),
            "think_len": think_lens[i],
            "gate_mode": gate_mode,
        })
        token_weights_list.append(token_w)

    return rewards, details, token_weights_list


def compute_rewards_adaptive_head(model, processor, sample, candidates,
                                  cand_ids_list, think_ranges, prompt_len,
                                  device, cfg, hooks):
    """Rewards with per-rollout adaptive head selection (Exp8).

    Same as compute_rewards_with_head_lsr but uses compute_adaptive_head_lsr
    to dynamically select top-K heads per sample from ALL 448 heads.
    """
    r_correct_list, r_lsr_list = [], []
    head_scores_list, think_lens = [], []
    details, token_weights_list = [], []
    all_selected_heads = []
    gt = sample["answer"]
    qtype = sample.get("type", "short_answer")
    answers_all = sample.get("answers_all")
    adaptive_top_k = cfg.get("adaptive_top_k", 12)

    # Phase 1: Compute raw rewards for ALL candidates
    for cand, cand_ids, t_range in zip(candidates, cand_ids_list, think_ranges):
        pred = extract_answer(cand, qtype)
        r_correct = compute_r_correct(pred, gt, qtype, answers_all=answers_all)

        try:
            head_scores, mean_score, think_len, selected = compute_adaptive_head_lsr(
                model, processor, sample, cand_ids, t_range, device, hooks,
                top_k=adaptive_top_k)
        except Exception:
            head_scores = torch.zeros(0, device=device)
            mean_score, think_len = 0.0, 0
            selected = []

        r_lsr = min(mean_score / cfg["lsr_scale"], 1.0)

        r_correct_list.append(r_correct)
        r_lsr_list.append(r_lsr)
        head_scores_list.append(head_scores)
        think_lens.append(think_len)
        all_selected_heads.append(selected)

    # Phase 2: GDPO normalization (same as fixed-head version)
    use_gdpo = cfg.get("gdpo", False)
    use_vppo = cfg.get("vppo_mask", False)
    use_gated = cfg.get("gated_head_lsr", False)

    correct_has_variance = np.std(r_correct_list) > 1e-6

    if use_gdpo and len(r_correct_list) > 1:
        w_correct = cfg.get("gdpo_w_correct", 0.6)
        w_lsr = cfg.get("gdpo_w_lsr", 0.4)

        if use_gated:
            if correct_has_variance:
                w_correct, w_lsr = 1.0, 0.0
            else:
                w_correct, w_lsr = 0.0, 1.0

        norm_correct = gdpo_normalize(r_correct_list)
        norm_lsr = gdpo_normalize(r_lsr_list)
        combined = w_correct * norm_correct + w_lsr * norm_lsr
        rewards = combined.tolist()
    else:
        rewards = []
        for rc, rl in zip(r_correct_list, r_lsr_list):
            rewards.append(rc * 0.5 + rc * rl * 0.5)

    # Phase 3: Build per-token weights + decay penalty
    for i, (cand_ids, t_range) in enumerate(zip(cand_ids_list, think_ranges)):
        head_scores = head_scores_list[i]
        decay_pen = compute_decay_penalty(head_scores) if head_scores.numel() >= 2 else 0.0
        rewards[i] -= cfg["beta_decay"] * decay_pen

        t_start, t_end = t_range
        n_tokens = cand_ids.numel()
        token_w = torch.ones(n_tokens, device=device)

        alpha_effective = cfg["alpha"]
        if use_gated:
            alpha_effective = 0.0 if correct_has_variance else cfg.get("gated_alpha", cfg["alpha"])

        if head_scores.numel() >= 10 and alpha_effective > 0:
            norm_scores = normalize_head_scores(head_scores)
            norm_scores = torch.clamp(norm_scores, 0.0, 5.0)
            t_end_safe = min(t_end, n_tokens)
            t_start_safe = min(t_start, t_end_safe)
            w_len = min(norm_scores.numel(), t_end_safe - t_start_safe)
            if w_len > 0:
                if use_vppo:
                    mask = (norm_scores[:w_len] >= 1.0).float()
                    token_w[t_start_safe:t_start_safe + w_len] = (
                        mask * (1.0 + alpha_effective * norm_scores[:w_len]))
                else:
                    token_w[t_start_safe:t_start_safe + w_len] = (
                        1.0 + alpha_effective * norm_scores[:w_len])

        gate_mode = "standard"
        if use_gated:
            gate_mode = "correctness" if correct_has_variance else "adaptive_head_lsr"

        # Log selected heads info
        sel = all_selected_heads[i] if i < len(all_selected_heads) else []
        sel_summary = [(l, h, round(d, 2)) for l, h, d in sel[:5]]  # top-5 for logging

        details.append({
            "correct": r_correct_list[i],
            "head_score_raw": r_lsr_list[i] * cfg["lsr_scale"],
            "decay_penalty": decay_pen,
            "token_weight_mean": token_w.mean().item(),
            "token_weight_max": token_w.max().item(),
            "think_len": think_lens[i],
            "gate_mode": gate_mode,
            "selected_heads_top5": sel_summary,
            "n_adaptive_heads": len(sel),
        })
        token_weights_list.append(token_w)

    return rewards, details, token_weights_list


def compute_rewards_soft_weighted(model, processor, sample, candidates,
                                  cand_ids_list, think_ranges, prompt_len,
                                  device, cfg, hooks):
    """Rewards with soft-weighted all-head scoring (Exp9).

    Same reward structure as Exp8 but uses compute_soft_weighted_head_lsr
    with continuous sigmoid weights instead of discrete top-K selection.
    """
    r_correct_list, r_lsr_list = [], []
    head_scores_list, think_lens = [], []
    details, token_weights_list = [], []
    all_head_weights = []
    gt = sample["answer"]
    qtype = sample.get("type", "short_answer")
    answers_all = sample.get("answers_all")
    soft_temp = cfg.get("soft_temperature", "auto")
    soft_temp_scale = cfg.get("soft_temperature_scale", 1.0)
    soft_layer_aware = cfg.get("layer_aware", False)
    soft_top_p = cfg.get("top_p_heads", 0.0)

    # Phase 1: Compute raw rewards for ALL candidates
    for cand, cand_ids, t_range in zip(candidates, cand_ids_list, think_ranges):
        pred = extract_answer(cand, qtype)
        r_correct = compute_r_correct(pred, gt, qtype, answers_all=answers_all)

        try:
            head_scores, mean_score, think_len, hw = compute_soft_weighted_head_lsr(
                model, processor, sample, cand_ids, t_range, device, hooks,
                temperature=soft_temp, temp_scale=soft_temp_scale,
                layer_aware=soft_layer_aware, top_p_heads=soft_top_p)
        except Exception:
            head_scores = torch.zeros(0, device=device)
            mean_score, think_len = 0.0, 0
            hw = {}

        r_lsr = min(mean_score / cfg["lsr_scale"], 1.0)

        r_correct_list.append(r_correct)
        r_lsr_list.append(r_lsr)
        head_scores_list.append(head_scores)
        think_lens.append(think_len)
        all_head_weights.append(hw)

    # Phase 2: GDPO normalization
    use_gdpo = cfg.get("gdpo", False)
    use_vppo = cfg.get("vppo_mask", False)
    use_gated = cfg.get("gated_head_lsr", False)

    correct_has_variance = np.std(r_correct_list) > 1e-6

    if use_gdpo and len(r_correct_list) > 1:
        w_correct = cfg.get("gdpo_w_correct", 0.6)
        w_lsr = cfg.get("gdpo_w_lsr", 0.4)

        if use_gated:
            if correct_has_variance:
                w_correct, w_lsr = 1.0, 0.0
            else:
                w_correct, w_lsr = 0.0, 1.0

        norm_correct = gdpo_normalize(r_correct_list)
        norm_lsr = gdpo_normalize(r_lsr_list)
        combined = w_correct * norm_correct + w_lsr * norm_lsr
        rewards = combined.tolist()
    else:
        rewards = []
        for rc, rl in zip(r_correct_list, r_lsr_list):
            rewards.append(rc * 0.5 + rc * rl * 0.5)

    # Phase 3: Build per-token weights + decay penalty
    for i, (cand_ids, t_range) in enumerate(zip(cand_ids_list, think_ranges)):
        head_scores = head_scores_list[i]
        decay_pen = compute_decay_penalty(head_scores) if head_scores.numel() >= 2 else 0.0
        rewards[i] -= cfg["beta_decay"] * decay_pen

        t_start, t_end = t_range
        n_tokens = cand_ids.numel()
        token_w = torch.ones(n_tokens, device=device)

        alpha_effective = cfg["alpha"]
        if use_gated:
            alpha_effective = 0.0 if correct_has_variance else cfg.get("gated_alpha", cfg["alpha"])

        if head_scores.numel() >= 10 and alpha_effective > 0:
            norm_scores = normalize_head_scores(head_scores)
            norm_scores = torch.clamp(norm_scores, 0.0, 5.0)
            t_end_safe = min(t_end, n_tokens)
            t_start_safe = min(t_start, t_end_safe)
            w_len = min(norm_scores.numel(), t_end_safe - t_start_safe)
            if w_len > 0:
                if use_vppo:
                    mask = (norm_scores[:w_len] >= 1.0).float()
                    token_w[t_start_safe:t_start_safe + w_len] = (
                        mask * (1.0 + alpha_effective * norm_scores[:w_len]))
                else:
                    token_w[t_start_safe:t_start_safe + w_len] = (
                        1.0 + alpha_effective * norm_scores[:w_len])

        gate_mode = "standard"
        if use_gated:
            gate_mode = "correctness" if correct_has_variance else "soft_weighted_lsr"

        # Log head weight stats
        hw = all_head_weights[i] if i < len(all_head_weights) else {}
        if hw:
            w_vals = list(hw.values())
            n_active = sum(1 for w in w_vals if w >= 0.01)
            n_high = sum(1 for w in w_vals if w >= 0.8)
            n_mid = sum(1 for w in w_vals if 0.3 <= w < 0.8)
            n_low = sum(1 for w in w_vals if 0.01 <= w < 0.3)
            top5 = sorted(hw.items(), key=lambda x: x[1], reverse=True)[:5]
            top5_summary = [(l, h, round(w, 3)) for (l, h), w in top5]
        else:
            n_active, n_high, n_mid, n_low = 0, 0, 0, 0
            top5_summary = []

        details.append({
            "correct": r_correct_list[i],
            "head_score_raw": r_lsr_list[i] * cfg["lsr_scale"],
            "decay_penalty": decay_pen,
            "token_weight_mean": token_w.mean().item(),
            "token_weight_max": token_w.max().item(),
            "think_len": think_lens[i],
            "gate_mode": gate_mode,
            "n_active_heads": n_active,
            "n_high_weight": n_high,
            "n_mid_weight": n_mid,
            "n_low_weight": n_low,
            "top5_heads": top5_summary,
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
            q = s["question"] + " Answer briefly."
            inputs = prepare_inputs(processor, s["image"], q, device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            raw = (processor.tokenizer if processor else tokenizer).decode(
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
            raw = (processor.tokenizer if processor else tokenizer).decode(
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
            raw = (processor.tokenizer if processor else tokenizer).decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
            if extract_yes_no(raw) == gt: real_c += 1
            black = Image.new('RGB', s["image"].size, (0, 0, 0))
            inputs_b = prepare_inputs(processor, black, q, device)
            with torch.no_grad():
                out_b = model.generate(**inputs_b, max_new_tokens=512, do_sample=False)
            raw_b = (processor.tokenizer if processor else tokenizer).decode(
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


def evaluate_mme(model, processor, device, max_pairs=100):
    """Evaluate on MME benchmark (Perception + Cognition).

    MME scoring: each image has 2 questions (yes/no variant).
    Image scores 1 point only if BOTH are answered correctly.
    """
    from qwen_vl_utils import process_vision_info
    was_training = model.training
    model.eval()
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()

    # Load MME data
    mme_path = Path("data/eval/mme")
    if not mme_path.exists():
        print("[mme] No local data, skipping MME eval")
        if was_training:
            model.train()
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
        return {"perception": 0, "cognition": 0, "total": 0, "n_pairs": 0}

    from datasets import load_from_disk
    ds = load_from_disk(str(mme_path))

    PERCEPTION = ["existence", "count", "position", "color", "posters",
                  "celebrity", "scene", "landmark", "artwork", "OCR"]
    COGNITION = ["commonsense_reasoning", "numerical_calculation",
                 "text_translation", "code_reasoning"]

    # Group by question_id
    grouped = defaultdict(list)
    for i in range(len(ds)):
        row = ds[i]
        grouped[row["question_id"]].append({
            "question": row["question"],
            "answer": row["answer"].strip(),
            "category": row["category"],
            "image": row["image"],
        })

    pairs = list(grouped.values())[:max_pairs]

    subtask_correct = defaultdict(int)
    subtask_total = defaultdict(int)

    for pair in pairs:
        cat = pair[0]["category"]
        subtask_total[cat] += 1
        all_ok = True
        for q in pair:
            try:
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": q["image"]},
                    {"type": "text", "text": q["question"]},
                ]}]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=True)
                images, videos, _ = process_vision_info(messages, return_video_kwargs=True)
                inputs = processor(text=[text], images=images, videos=videos,
                                   return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
                raw = (processor.tokenizer if processor else tokenizer).decode(
                    out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
                for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
                    raw = raw.replace(tok, "")
                pred = extract_yes_no(raw)
                if pred != q["answer"]:
                    all_ok = False
            except Exception:
                all_ok = False
        if all_ok:
            subtask_correct[cat] += 1

    # Compute aggregate scores
    perc_score = sum(subtask_correct.get(s, 0) for s in PERCEPTION
                     if s in subtask_total)
    perc_total = sum(subtask_total.get(s, 0) for s in PERCEPTION
                     if s in subtask_total)
    cog_score = sum(subtask_correct.get(s, 0) for s in COGNITION
                    if s in subtask_total)
    cog_total = sum(subtask_total.get(s, 0) for s in COGNITION
                    if s in subtask_total)

    if was_training:
        model.train()
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()

    return {
        "perception": perc_score / perc_total if perc_total > 0 else 0,
        "cognition": cog_score / cog_total if cog_total > 0 else 0,
        "perc_score": perc_score, "perc_total": perc_total,
        "cog_score": cog_score, "cog_total": cog_total,
        "n_pairs": len(pairs),
        "subtasks": {k: {"correct": subtask_correct.get(k, 0),
                         "total": v} for k, v in subtask_total.items()},
    }


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
#  Curriculum by Think Length
# ══════════════════════════════════════════════════════════════════════

def prescreen_think_lengths(model, processor, samples, device,
                            temperature, top_p, max_new_tokens,
                            min_think_tokens, cache_file=None, max_screen=200):
    """Generate 1 candidate per sample to estimate think token count.

    Returns dict {sample_idx: think_token_count}.
    """
    if cache_file and Path(cache_file).exists():
        with open(cache_file) as f:
            cache = {int(k): v for k, v in json.load(f).items()}
        print(f"[curriculum] Loaded think lengths from cache ({len(cache)} entries)")
        return cache

    print(f"[curriculum] Pre-screening think lengths ({min(max_screen, len(samples))} samples)...")
    cache = {}
    model.eval()
    for i, sample in enumerate(samples[:max_screen]):
        try:
            cands, cand_ids, pl, inp, tranges = generate_candidates(
                model, processor, sample, 1,
                temperature, top_p, max_new_tokens,
                min_think_tokens, device, tokenizer=tokenizer)
            t_start, t_end = tranges[0]
            cache[i] = t_end - t_start
            del cands, cand_ids, inp
        except Exception:
            cache[i] = 999
        if (i + 1) % 50 == 0:
            print(f"  [curriculum] {i+1}/{min(max_screen, len(samples))} screened")

    # Fill unscreened with median
    median_len = int(np.median(list(cache.values()))) if cache else 100
    for i in range(len(samples)):
        if i not in cache:
            cache[i] = median_len

    if cache_file:
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(cache, f)
        print(f"[curriculum] Saved think length cache to {cache_file}")

    short = sum(1 for v in cache.values() if v <= 100)
    med = sum(1 for v in cache.values() if 100 < v <= 200)
    long = sum(1 for v in cache.values() if v > 200)
    print(f"[curriculum] Distribution: short(≤100)={short}, med(101-200)={med}, long(>200)={long}")
    return cache


def build_curriculum_bins(samples, think_cache, thresholds):
    """Build cumulative sample bins for curriculum phases.

    thresholds: [100, 200, 999] → phase 0 gets ≤100, phase 1 gets ≤200, phase 2 gets all
    Returns: list of lists, each containing (original_idx, sample) tuples.
    """
    bins = [[] for _ in range(len(thresholds))]
    for i, sample in enumerate(samples):
        tl = think_cache.get(i, 999)
        for phase_idx, thresh in enumerate(thresholds):
            if tl <= thresh:
                bins[phase_idx].append((i, sample))
                break

    # Make cumulative: each phase includes all previous
    cumulative = []
    running = []
    for b in bins:
        running = running + b
        cumulative.append(list(running))

    for pi, c in enumerate(cumulative):
        print(f"[curriculum] Phase {pi}: {len(c)} samples available")

    return cumulative


def get_curriculum_phase(step, boundaries):
    """Return 0-based phase index for current step."""
    for i, b in enumerate(boundaries):
        if step < b:
            return i
    return len(boundaries) - 1


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
    gated_str = "ON" if cfg.get("gated_head_lsr") else "OFF"
    curric_str = "ON" if cfg.get("curriculum") else "OFF"
    adapt_str = "ON" if cfg.get("adaptive_heads") else "OFF"
    print(f"\n{'='*70}")
    print(f"  Phase 6c: Head-Level Mask LSR-Weighted GRPO")
    print(f"  GDPO={gdpo_str} | VPPO={vppo_str} | Gated={gated_str} | "
          f"Curriculum={curric_str} | Adaptive={adapt_str}")
    print(f"  alpha={cfg['alpha']} | beta_decay={cfg['beta_decay']} | "
          f"steps={cfg['num_steps']} | group={cfg['group_size']} | "
          f"T={cfg['temperature']} | lr={cfg['lr']}")
    if cfg.get("gdpo"):
        print(f"  GDPO weights: correct={cfg.get('gdpo_w_correct', 0.6)}, "
              f"lsr={cfg.get('gdpo_w_lsr', 0.4)}")
    print(f"  Vision heads: {len(cfg['vision_heads'])} heads across "
          f"{len(set(l for l,h,d in cfg['vision_heads']))} layers")
    print(f"{'='*70}\n")

    model, processor, tokenizer = load_model(model_path, for_training=True,
                                              model_key=cfg.get("model_key"))
    device = next(model.parameters()).device

    # Install vision head hooks (persistent during training)
    use_adaptive = cfg.get("adaptive_heads", False)
    use_soft = cfg.get("soft_weighted_heads", False)
    mcfg = ACTIVE_MODEL_CFG
    n_layers = mcfg["num_layers"]
    n_heads = mcfg["num_heads"]
    h_dim = mcfg["head_dim"]
    total_heads = n_layers * n_heads

    if use_adaptive or use_soft:
        hooks = AdaptiveVisionHeadHooks(model, num_layers=n_layers,
                                         num_heads=n_heads, head_dim=h_dim)
        if use_soft:
            soft_t = cfg.get("soft_temperature", "auto")
            print(f"[hooks] Exp9 soft-weighted: ALL {n_layers} layers hooked ({total_heads} heads), "
                  f"sigmoid weights, temperature={soft_t}")
        else:
            print(f"[hooks] Exp8 adaptive: ALL {n_layers} layers hooked ({total_heads} heads), "
                  f"top-{cfg.get('adaptive_top_k', 12)} selected per sample")
    else:
        hooks = VisionHeadHooks(model, cfg["vision_heads"],
                                num_heads=n_heads, head_dim=h_dim)
        print(f"[hooks] Fixed mode: installed on layers {hooks.layers_needed}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr"], weight_decay=0.01)

    # Curriculum pre-screening
    curriculum_bins = None
    if cfg.get("curriculum"):
        cache_file = str(output_dir / "think_length_cache.json")
        think_cache = prescreen_think_lengths(
            model, processor, train_data, device,
            cfg["temperature"], cfg["top_p"],
            cfg["max_new_tokens"], cfg["min_think_tokens"],
            cache_file=cache_file, max_screen=min(200, len(train_data)))
        thresholds = cfg.get("curriculum_thresholds", [100, 200, 999])
        curriculum_bins = build_curriculum_bins(train_data, think_cache, thresholds)

    # Eval sample sizes (configurable)
    n_pope = cfg.get("eval_pope_samples", 60)
    n_blind = cfg.get("eval_blind_samples", 50)
    n_tvqa = cfg.get("eval_textvqa_samples", 50)
    n_mme = cfg.get("eval_mme_pairs", 0)

    # Pre-eval (fast: POPE-only, 60 samples)
    pre_n = min(60, n_pope)
    print(f"Pre-training eval (POPE={pre_n} quick)...")
    pre_pope = evaluate_pope(model, processor, eval_data, device, pre_n)
    pre_textvqa = {"acc": 0, "total": 0}
    pre_blind = {"gap": 0, "real_acc": 0, "blind_acc": 0, "total": 0}
    pre_mme = None
    print(f"  POPE: {pre_pope['acc']:.1%} (quick pre-eval)")
    eval_at = cfg.get('eval_steps_list', []) or f"every {cfg['eval_every']}"
    print(f"  Full eval (POPE={n_pope}, Blind={n_blind}, TextVQA={n_tvqa}, MME={n_mme}) at steps: {eval_at}")

    history = {
        "config": {k: v for k, v in cfg.items() if k != "vision_heads"},
        "vision_heads": [(l, h, d) for l, h, d in cfg["vision_heads"]],
        "pre_eval": {"textvqa": pre_textvqa, "pope": pre_pope, "blind": pre_blind,
                     **({"mme": pre_mme} if pre_mme else {})},
        "steps": [], "evals": [],
    }

    model.train()
    optimizer.zero_grad()
    best_acc = pre_pope["acc"]

    # Data iterator: deterministic order via seed, epoch-aware reshuffling
    samples_per_step = cfg.get("samples_per_step", 1)
    data_rng = random.Random(cfg.get("seed", 42))
    data_order = list(range(len(train_data)))
    data_rng.shuffle(data_order)
    data_cursor = 0
    data_epoch = 0
    total_samples_seen = 0
    n_accum = samples_per_step  # accumulate over this many samples

    print(f"[data] {len(train_data)} samples, {samples_per_step} per step, "
          f"{cfg['num_steps']} steps → {cfg['num_steps'] * samples_per_step} total "
          f"({cfg['num_steps'] * samples_per_step * 100 / len(train_data):.1f}% coverage)")

    for step in range(cfg["num_steps"]):
        step_t0 = time.time()
        step_losses = []
        step_details_all = []
        sub_ok = 0

        for sub in range(samples_per_step):
            # Deterministic sample selection with epoch-aware reshuffling
            if data_cursor >= len(data_order):
                data_epoch += 1
                data_rng.shuffle(data_order)
                data_cursor = 0
                print(f"  [data] Epoch {data_epoch} — reshuffled")

            if curriculum_bins is not None:
                boundaries = cfg.get("curriculum_phases", [10, 20, 30])
                global_idx = step * samples_per_step + sub
                phase_idx = get_curriculum_phase(global_idx, boundaries)
                available = curriculum_bins[min(phase_idx, len(curriculum_bins) - 1)]
                if not available:
                    available = [(i, s) for i, s in enumerate(train_data)]
                _, sample = available[global_idx % len(available)]
            else:
                sample_idx = data_order[data_cursor]
                sample = train_data[sample_idx]
                data_cursor += 1
                total_samples_seen += 1

            # Generate candidates
            model.eval()
            if hasattr(model, 'gradient_checkpointing_disable'):
                model.gradient_checkpointing_disable()
            try:
                candidates, cand_ids_list, prompt_len, inputs, think_ranges = \
                    generate_candidates(
                        model, processor, sample, cfg["group_size"],
                        cfg["temperature"], cfg["top_p"],
                        cfg["max_new_tokens"], cfg["min_think_tokens"], device,
                        tokenizer=tokenizer)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); gc.collect()
                print(f"  [step {step+1}.{sub+1}] OOM gen, skip"); continue

            # Rewards + head-level token weights
            try:
                if use_soft:
                    rewards, details, token_weights_list = compute_rewards_soft_weighted(
                        model, processor, sample, candidates, cand_ids_list,
                        think_ranges, prompt_len, device, cfg, hooks)
                elif use_adaptive:
                    rewards, details, token_weights_list = compute_rewards_adaptive_head(
                        model, processor, sample, candidates, cand_ids_list,
                        think_ranges, prompt_len, device, cfg, hooks)
                else:
                    rewards, details, token_weights_list = compute_rewards_with_head_lsr(
                        model, processor, sample, candidates, cand_ids_list,
                        think_ranges, prompt_len, device, cfg, hooks)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); gc.collect()
                print(f"  [step {step+1}.{sub+1}] OOM reward, skip"); continue

            rarr = np.array(rewards)
            rstd = rarr.std()

            # Dynamic resampling for zero-variance groups (DAPO-style)
            if rstd < 1e-8:
                resample_tries = 0
                max_resample = 5
                while rstd < 1e-8 and resample_tries < max_resample:
                    resample_tries += 1
                    alt_idx = (step * 7 + sub * 31 + resample_tries * 13) % len(train_data)
                    alt_sample = train_data[alt_idx]
                    try:
                        candidates, cand_ids_list, prompt_len, inputs, think_ranges = \
                            generate_candidates(
                                model, processor, alt_sample, cfg["group_size"],
                                cfg["temperature"], cfg["top_p"],
                                cfg["max_new_tokens"], cfg["min_think_tokens"], device,
                                tokenizer=tokenizer)
                        if use_soft:
                            rewards, details, token_weights_list = compute_rewards_soft_weighted(
                                model, processor, alt_sample, candidates, cand_ids_list,
                                think_ranges, prompt_len, device, cfg, hooks)
                        elif use_adaptive:
                            rewards, details, token_weights_list = compute_rewards_adaptive_head(
                                model, processor, alt_sample, candidates, cand_ids_list,
                                think_ranges, prompt_len, device, cfg, hooks)
                        else:
                            rewards, details, token_weights_list = compute_rewards_with_head_lsr(
                                model, processor, alt_sample, candidates, cand_ids_list,
                                think_ranges, prompt_len, device, cfg, hooks)
                        sample = alt_sample
                        rarr = np.array(rewards)
                        rstd = rarr.std()
                    except Exception:
                        break

                if rstd < 1e-8:
                    if samples_per_step == 1:
                        elapsed = time.time() - step_t0
                        print(f"  [step {step+1}/{cfg['num_steps']}] "
                              f"SKIP r={rarr.mean():.3f} ({elapsed:.1f}s)")
                        history["steps"].append({
                            "step": step + 1, "skipped": True,
                            "mean_reward": float(rarr.mean())})
                    del candidates, cand_ids_list, inputs; continue

            if cfg.get("gdpo", False):
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
                (loss / n_accum).backward()
                step_losses.append(loss.item())
                step_details_all.extend(details)
                sub_ok += 1
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); gc.collect()
                print(f"  [step {step+1}.{sub+1}] OOM loss, skip"); continue

        # Optimizer step after accumulating all sub-samples
        if sub_ok > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.get("max_grad_norm", 1.0))
            optimizer.step()
        optimizer.zero_grad()

        if sub_ok == 0:
            history["steps"].append({
                "step": step + 1, "skipped": True, "mean_reward": 0.0})
            continue

        # Use accumulated details for logging
        details = step_details_all
        loss_val = np.mean(step_losses)

        elapsed = time.time() - step_t0
        mc = np.mean([d["correct"] for d in details])
        mh = np.mean([d["head_score_raw"] for d in details])
        md = np.mean([d["decay_penalty"] for d in details])
        tw_mean = np.mean([d["token_weight_mean"] for d in details])
        tw_max = np.mean([d["token_weight_max"] for d in details])

        step_info = {
            "step": step + 1, "loss": float(loss_val),
            "samples_ok": sub_ok, "samples_seen": total_samples_seen,
            "mean_reward": float(rarr.mean()),
            "reward_std": float(rstd),
            "mean_correct": float(mc),
            "mean_head_score": float(mh),
            "mean_decay_pen": float(md),
            "token_weight_mean": float(tw_mean),
            "token_weight_max": float(tw_max),
            "mean_entropy": float(np.mean(lstats["entropy"])) if (sub_ok > 0 and lstats.get("entropy")) else 0,
            "elapsed": elapsed,
        }
        # Gated Head-LSR tracking
        if cfg.get("gated_head_lsr"):
            gate_modes = [d.get("gate_mode", "standard") for d in details]
            step_info["gate_mode"] = gate_modes[0] if gate_modes else "standard"
        # Adaptive head tracking
        if use_adaptive and not use_soft:
            top5_heads = details[0].get("selected_heads_top5", []) if details else []
            step_info["adaptive_top5"] = top5_heads
        # Soft-weighted head tracking
        if use_soft and details:
            d0 = details[0]
            step_info["soft_n_active"] = d0.get("n_active_heads", 0)
            step_info["soft_n_high"] = d0.get("n_high_weight", 0)
            step_info["soft_n_mid"] = d0.get("n_mid_weight", 0)
            step_info["soft_n_low"] = d0.get("n_low_weight", 0)
            step_info["soft_top5"] = d0.get("top5_heads", [])
        # Curriculum tracking
        if curriculum_bins is not None:
            step_info["curriculum_phase"] = get_curriculum_phase(
                step, cfg.get("curriculum_phases", [10, 20, 30]))
        history["steps"].append(step_info)

        sub_info = f" [{sub_ok}/{samples_per_step}]" if samples_per_step > 1 else ""
        step_msg = (f"  [step {step+1}/{cfg['num_steps']}] "
                    f"loss={loss_val:.4f} r={rarr.mean():.3f}±{rstd:.3f} "
                    f"correct={mc:.2f} headΔ={mh:.3f} decay={md:.3f} "
                    f"tw={tw_mean:.2f}/{tw_max:.1f} ({elapsed:.1f}s){sub_info}")
        if use_soft and details:
            d0 = details[0]
            step_msg += (f" [soft: {d0.get('n_active_heads',0)} active, "
                        f"{d0.get('n_high_weight',0)}H/{d0.get('n_mid_weight',0)}M/"
                        f"{d0.get('n_low_weight',0)}L]")
        print(step_msg, flush=True)

        # Eval
        eval_steps_list = cfg.get("eval_steps_list", [])
        should_eval = ((step + 1) in eval_steps_list) if eval_steps_list else \
                      ((step + 1) % cfg["eval_every"] == 0 or step + 1 == cfg["num_steps"])
        if should_eval:
            # Free training memory before eval to avoid OOM
            torch.cuda.empty_cache(); gc.collect()
            tvqa_res = evaluate_textvqa(model, processor,
                                         textvqa_eval_data or [], device, n_tvqa) \
                if textvqa_eval_data else {"acc": 0, "total": 0}
            pope_res = evaluate_pope(model, processor, eval_data, device, n_pope)
            blind_res = evaluate_blind(model, processor, eval_data, device, n_blind)
            mme_res = evaluate_mme(model, processor, device, n_mme) if n_mme > 0 else None
            eval_msg = (f"  === Eval step {step+1}: "
                        f"TextVQA={tvqa_res['acc']:.1%} "
                        f"POPE={pope_res['acc']:.1%} "
                        f"Gap={blind_res['gap']:.1%}")
            if mme_res:
                eval_msg += (f" MME P={mme_res['perception']:.1%} "
                            f"C={mme_res['cognition']:.1%}")
            eval_msg += " ==="
            print(eval_msg)
            eval_entry = {"step": step + 1, "textvqa": tvqa_res,
                          "pope": pope_res, "blind": blind_res}
            if mme_res:
                eval_entry["mme"] = mme_res
            history["evals"].append(eval_entry)

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
    parser.add_argument("--model-key", type=str, default="qwen3_vl_2b",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model to train (default: qwen3_vl_2b)")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Override HF model path (default: from model config)")
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
    parser.add_argument("--samples-per-step", type=int, default=1,
                        help="Number of samples per optimizer step (mini-batch). "
                             "Total coverage = steps × samples_per_step / train_samples")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--min-think-tokens", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-steps", type=str, default="",
                        help="Comma-separated list of steps to eval at (overrides --eval-every)")
    parser.add_argument("--eval-pope-samples", type=int, default=60,
                        help="POPE eval samples (default 60)")
    parser.add_argument("--eval-blind-samples", type=int, default=50,
                        help="Blind test eval samples (default 50)")
    parser.add_argument("--eval-textvqa-samples", type=int, default=50,
                        help="TextVQA eval samples (default 50)")
    parser.add_argument("--eval-mme-pairs", type=int, default=0,
                        help="MME eval pairs (0=disabled, default 0)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--train-samples", type=int, default=500)
    parser.add_argument("--include-mme-train", action="store_true",
                        help="Include MME data in training (excludes eval samples)")
    parser.add_argument("--mme-ratio", type=float, default=0.3,
                        help="Fraction of train-samples to fill with MME (default 0.3)")
    parser.add_argument("--mme-eval-reserve", type=int, default=200,
                        help="Number of MME question_ids reserved for eval (default 200)")
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
    # Gated Head-LSR
    parser.add_argument("--gated-head-lsr", action="store_true",
                        help="Gate: correctness-only when R_correct has variance, "
                             "head-LSR when zero variance")
    parser.add_argument("--gated-alpha", type=float, default=None,
                        help="Alpha for head-LSR branch (default: same as --alpha)")
    # Exp8: Adaptive per-rollout head selection
    parser.add_argument("--adaptive-heads", action="store_true",
                        help="Exp8: Select top-K heads per sample from ALL 448 "
                             "heads based on real-vs-black activation delta")
    parser.add_argument("--adaptive-top-k", type=int, default=12,
                        help="Number of heads to select per sample (default: 12)")
    # Exp9: Soft-weighted all-head LSR
    parser.add_argument("--soft-weighted-heads", action="store_true",
                        help="Exp9: Use continuous sigmoid weights for ALL 448 heads "
                             "instead of discrete top-K selection")
    parser.add_argument("--soft-temperature", type=str, default="auto",
                        help="Temperature for sigmoid weights ('auto'=std(deltas), "
                             "or float value)")
    parser.add_argument("--soft-temperature-scale", type=float, default=1.0,
                        help="Multiply temperature by this (Exp10: 0.33 for sharper)")
    parser.add_argument("--layer-aware", action="store_true",
                        help="Exp11: Layer-aware weighting (decision 2×, feature 1.5×)")
    parser.add_argument("--top-p-heads", type=float, default=0.0,
                        help="Exp12: Top-P head selection (0.9 keeps top 90%% weight)")
    # Curriculum by think length
    parser.add_argument("--curriculum", action="store_true",
                        help="Enable think-length curriculum (short→long)")
    parser.add_argument("--curriculum-phases", type=str, default="10,20,30",
                        help="Step boundaries for curriculum phases (comma-separated)")
    parser.add_argument("--curriculum-thresholds", type=str, default="100,200,999",
                        help="Max think tokens per phase (comma-separated)")
    args = parser.parse_args()

    # Set active model config
    global ACTIVE_MODEL_KEY, ACTIVE_MODEL_CFG, HF_ID
    ACTIVE_MODEL_KEY = args.model_key
    ACTIVE_MODEL_CFG = MODEL_CONFIGS[args.model_key]
    HF_ID = ACTIVE_MODEL_CFG["hf_id"]
    if args.model_path is None:
        args.model_path = ACTIVE_MODEL_CFG["hf_id"]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load vision heads from calibration
    cal_file = args.calibration_file
    if cal_file == "checkpoints/calibration/qwen3_vl_2b/calibration_meta.json" and args.model_key != "qwen3_vl_2b":
        cal_file = ACTIVE_MODEL_CFG.get("calibration_file", cal_file)
    vision_heads = list(DEFAULT_VISION_HEADS[:args.top_k_heads])
    args.calibration_file = cal_file
    if os.path.exists(cal_file):
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

    # Soft-weighted temperature parsing
    soft_t = cfg.get("soft_temperature", "auto")
    if soft_t != "auto":
        try:
            cfg["soft_temperature"] = float(soft_t)
        except ValueError:
            cfg["soft_temperature"] = "auto"

    # Gated Head-LSR
    cfg["gated_head_lsr"] = cfg.pop("gated_head_lsr", False)
    if cfg.get("gated_alpha") is None:
        cfg["gated_alpha"] = cfg["alpha"]

    # Eval steps
    es = cfg.pop("eval_steps", "")
    cfg["eval_steps_list"] = [int(x) for x in es.split(",") if x.strip()] if es else []

    # Curriculum
    cfg["curriculum"] = cfg.pop("curriculum", False)
    if cfg["curriculum"]:
        cfg["curriculum_phases"] = [int(x) for x in cfg.pop("curriculum_phases", "10,20,30").split(",")]
        cfg["curriculum_thresholds"] = [int(x) for x in cfg.pop("curriculum_thresholds", "100,200,999").split(",")]
    else:
        cfg.pop("curriculum_phases", None)
        cfg.pop("curriculum_thresholds", None)

    train_data = load_training_data(
        args.train_samples, args.seed,
        include_mme=args.include_mme_train,
        mme_ratio=args.mme_ratio,
        mme_eval_reserve=args.mme_eval_reserve)
    pope_eval_data = load_pope_eval(300)
    textvqa_eval_data = load_textvqa_eval(200)

    run_training(cfg, train_data, pope_eval_data, args.model_path,
                 textvqa_eval_data=textvqa_eval_data)


if __name__ == "__main__":
    main()
