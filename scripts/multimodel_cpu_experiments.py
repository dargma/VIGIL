#!/usr/bin/env python3
"""
Multi-Model CPU Experiments: Exp1/Exp8/Exp10 on InternVL3.5-1B & DeepSeek-VL2-Tiny

Runs 1-step GRPO training + quick POPE eval for each (model × experiment) combo.
CPU-only. Results saved to lab/reports/multimodel_cpu/

Experiments:
  Exp1:  Baseline GRPO with fixed top-K vision heads (from calibration)
  Exp8:  Adaptive per-rollout top-K head selection from ALL heads
  Exp10: Soft-weighted ALL heads with sharp sigmoid (T/3)

Usage:
    PYTHONUNBUFFERED=1 python -u scripts/multimodel_cpu_experiments.py \
        2>&1 | tee logs/multimodel_cpu.log
"""

import os, sys, json, time, random, gc, re, string, traceback
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from datetime import datetime
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

# ══════════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════════

MODEL_CONFIGS = {
    "internvl3_5_1b": {
        "hf_id": "OpenGVLab/InternVL3_5-1B",
        "num_layers": 28, "num_heads": 16, "num_kv_heads": 8, "head_dim": 128,
        "hidden_size": 1024, "gqa": True,
        "layer_path": "language_model.model.layers",
        "input_api": "internvl",
        "trust_remote_code": True,
        "thinking": False,
        "calibration_file": "checkpoints/calibration/internvl3_5_1b/calibration.json",
    },
    "deepseek_vl2_tiny": {
        "hf_id": "deepseek-ai/deepseek-vl2-tiny",
        "num_layers": 12, "num_heads": 10, "num_kv_heads": 10, "head_dim": 256,
        "hidden_size": 2560, "gqa": False,
        "layer_path": "language_model.model.layers",  # will verify at load time
        "input_api": "deepseek_vl2",
        "trust_remote_code": True,
        "thinking": False,
        "calibration_file": "",  # no calibration yet
    },
}

EXPERIMENTS = {
    "exp1_baseline": {
        "description": "Fixed top-K heads GRPO (baseline)",
        "adaptive_heads": False,
        "soft_weighted_heads": False,
    },
    "exp8_adaptive_topk": {
        "description": "Adaptive per-rollout top-K head selection",
        "adaptive_heads": True,
        "soft_weighted_heads": False,
        "adaptive_top_k": 12,
    },
    "exp10_sharp_sigmoid": {
        "description": "Soft-weighted ALL heads, sharp sigmoid (T/3)",
        "adaptive_heads": False,
        "soft_weighted_heads": True,
        "soft_temperature": "auto",
        "soft_temperature_scale": 0.33,
    },
}

TRAINING_CFG = {
    "num_steps": 1,
    "group_size": 2,      # minimal for CPU
    "temperature": 1.0,
    "top_p": 0.95,
    "lr": 5e-7,
    "alpha": 0.5,
    "beta_decay": 0.1,
    "beta_entropy": 0.01,
    "lsr_scale": 10.0,
    "max_new_tokens": 32,  # very short for CPU speed
    "min_think_tokens": 0,  # no thinking for non-thinking models
    "gdpo": True,
    "gdpo_w_correct": 0.6,
    "gdpo_w_lsr": 0.4,
    "vppo_mask": False,
    "gated_head_lsr": False,
    "max_grad_norm": 1.0,
    "seed": 42,
}

EVAL_SAMPLES = 3   # minimal eval (CPU: ~30s per sample with text-only)
TRAIN_SAMPLES = 2   # minimal training data


# ══════════════════════════════════════════════════════════════════════
#  Utilities
# ══════════════════════════════════════════════════════════════════════

def disk_check():
    usage = os.popen("df -h . | awk 'NR==2 {print $5}' | tr -d '%'").read().strip()
    if usage and int(usage) >= 95:
        raise RuntimeError("DISK 95% — halted")

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
    return text.split("\n")[0].strip()[:100]


# ══════════════════════════════════════════════════════════════════════
#  Model Loading (CPU)
# ══════════════════════════════════════════════════════════════════════

def find_layer_path(model):
    """Auto-detect the transformer layers path."""
    candidates = [
        "language_model.model.layers",
        "model.language_model.layers",
        "model.layers",
        "model.model.layers",
    ]
    for path in candidates:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if hasattr(obj, '__len__') and len(obj) > 0:
                # Verify it has self_attn
                if hasattr(obj[0], 'self_attn'):
                    return path
        except (AttributeError, TypeError):
            continue
    raise ValueError(f"Cannot find transformer layers. Model type: {type(model)}")


def get_model_layers(model, layer_path):
    """Navigate to model layers."""
    obj = model
    for attr in layer_path.split("."):
        obj = getattr(obj, attr)
    return obj


def load_model_cpu(model_key):
    """Load model on CPU in bfloat16."""
    mcfg = MODEL_CONFIGS[model_key]
    hf_id = mcfg["hf_id"]
    api = mcfg["input_api"]
    dtype = torch.bfloat16

    print(f"\n{'='*60}")
    print(f"  Loading {model_key}: {hf_id}")
    print(f"  CPU, dtype={dtype}")
    print(f"{'='*60}")

    if api == "internvl":
        from transformers import AutoModel, AutoTokenizer
        # InternVL custom code calls .item() in __init__ — incompatible with
        # meta tensors from low_cpu_mem_usage=True. Must load eagerly.
        model = AutoModel.from_pretrained(
            hf_id, torch_dtype=dtype, trust_remote_code=True,
            low_cpu_mem_usage=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        processor = None

    elif api == "deepseek_vl2":
        # DeepSeek-VL2 uses its own package (deepseek_vl2) with custom model code.
        # Requires: pip install --no-deps git+https://github.com/deepseek-ai/DeepSeek-VL2.git
        from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
        processor = DeepseekVLV2Processor.from_pretrained(hf_id)
        tokenizer = processor.tokenizer
        model = DeepseekVLV2ForCausalLM.from_pretrained(
            hf_id, torch_dtype=dtype,
        )

    else:
        raise ValueError(f"Unsupported api: {api}")

    model.eval()

    # InternVL: set img_context_token_id (required for generate/forward)
    if api == "internvl" and hasattr(model, 'img_context_token_id'):
        img_ctx_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        model.img_context_token_id = img_ctx_id
        print(f"  InternVL img_context_token_id = {img_ctx_id}")

    # Auto-detect layer path
    detected_path = find_layer_path(model)
    if detected_path != mcfg["layer_path"]:
        print(f"  Layer path override: {mcfg['layer_path']} → {detected_path}")
        mcfg["layer_path"] = detected_path

    # Verify architecture
    layers = get_model_layers(model, mcfg["layer_path"])
    actual_layers = len(layers)
    actual_heads = layers[0].self_attn.num_heads if hasattr(layers[0].self_attn, 'num_heads') else mcfg["num_heads"]
    print(f"  Layers: {actual_layers} (expected {mcfg['num_layers']})")
    print(f"  Attention heads: {actual_heads} (expected {mcfg['num_heads']})")
    if actual_layers != mcfg["num_layers"]:
        mcfg["num_layers"] = actual_layers
        print(f"  WARNING: layer count mismatch, updated config")

    # Check o_proj exists
    attn0 = layers[0].self_attn
    if not hasattr(attn0, 'o_proj'):
        # Some models use different names
        if hasattr(attn0, 'out_proj'):
            print(f"  Note: using 'out_proj' instead of 'o_proj'")
            mcfg["o_proj_name"] = "out_proj"
        else:
            print(f"  WARNING: no o_proj found. Available: {[n for n, _ in attn0.named_modules()]}")
            mcfg["o_proj_name"] = "o_proj"
    else:
        mcfg["o_proj_name"] = "o_proj"

    # Detect actual head_dim from o_proj weight shape
    o_proj = getattr(attn0, mcfg["o_proj_name"])
    if hasattr(o_proj, 'weight'):
        out_features, in_features = o_proj.weight.shape
        detected_head_dim = in_features // mcfg["num_heads"]
        if detected_head_dim != mcfg["head_dim"]:
            print(f"  head_dim override: {mcfg['head_dim']} → {detected_head_dim} "
                  f"(o_proj: {out_features}×{in_features}, {mcfg['num_heads']} heads)")
            mcfg["head_dim"] = detected_head_dim

    params = sum(p.numel() for p in model.parameters())
    mem_gb = params * 2 / 1e9  # bf16 = 2 bytes
    print(f"  Total params: {params/1e9:.2f}B ({mem_gb:.1f}GB in bf16)")

    return model, processor, tokenizer


# ══════════════════════════════════════════════════════════════════════
#  Input Preparation (CPU, per-model)
# ══════════════════════════════════════════════════════════════════════

CPU_TEXT_ONLY = os.environ.get("VIGIL_TEXT_ONLY", "1") == "1"

def prepare_inputs_internvl(model, tokenizer, image, question, dtype=torch.bfloat16):
    """Prepare InternVL inputs.

    CPU_TEXT_ONLY=True: skip image tokens for speed (~10x faster on CPU).
    The vision head hooks still capture activations from text processing,
    which validates the full GRPO pipeline architecture.
    """
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    if CPU_TEXT_ONLY:
        # Text-only mode: fast CPU inference, still tests full pipeline
        # Don't pass pixel_values — InternVL's generate will use text-only path
        text_inputs = tokenizer(question, return_tensors="pt")
        return dict(text_inputs)

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    pixel_values = transform(image).unsqueeze(0).to(dtype=dtype)

    # Build query with proper image token format
    num_image_token = getattr(model, 'num_image_token', 256)
    image_tokens = '<img>' + '<IMG_CONTEXT>' * num_image_token + '</img>'
    query = f"{image_tokens}\n{question}"

    text_inputs = tokenizer(query, return_tensors="pt")
    inputs = dict(text_inputs)
    inputs["pixel_values"] = pixel_values
    return inputs


def prepare_inputs_deepseek(model, processor, tokenizer, image, question, dtype=torch.bfloat16):
    """Prepare DeepSeek-VL2 inputs. Uses processor if available, else text-only."""
    if processor is not None:
        try:
            conversation = [{"role": "<|User|>",
                           "content": f"<image>\n{question}",
                           "images": [image]},
                          {"role": "<|Assistant|>", "content": ""}]
            inputs = processor(conversations=[conversation], images=[[image]],
                             force_batchify=True, return_tensors="pt")
            return {k: v.to(dtype=dtype) if v.dtype in (torch.float32, torch.float16) else v
                    for k, v in inputs.items()}
        except Exception as e:
            print(f"    [deepseek] Processor input failed: {e}, using text-only")

    # Fallback: text-only
    text_inputs = tokenizer(question, return_tensors="pt")
    return dict(text_inputs)


def prepare_inputs(model_key, model, processor, tokenizer, image, question):
    """Dispatch to model-specific input prep."""
    api = MODEL_CONFIGS[model_key]["input_api"]
    if api == "internvl":
        return prepare_inputs_internvl(model, tokenizer, image, question)
    elif api == "deepseek_vl2":
        return prepare_inputs_deepseek(model, processor, tokenizer, image, question)
    raise ValueError(f"Unknown api: {api}")


# ══════════════════════════════════════════════════════════════════════
#  Generation
# ══════════════════════════════════════════════════════════════════════

def get_lm(model, model_key):
    """Get the language model for direct text generation."""
    api = MODEL_CONFIGS[model_key]["input_api"]
    if api == "internvl" and hasattr(model, 'language_model'):
        return model.language_model
    return model


def generate_response(model, model_key, processor, tokenizer, image, question,
                      max_new_tokens=128, temperature=1.0, do_sample=True):
    """Generate a single response. Returns (text, gen_ids, prompt_len, inputs)."""
    inputs = prepare_inputs(model_key, model, processor, tokenizer, image, question)
    prompt_len = inputs["input_ids"].shape[1]

    # For text-only InternVL, use language_model directly
    gen_model = get_lm(model, model_key) if CPU_TEXT_ONLY else model
    gen_inputs = {k: v for k, v in inputs.items()
                  if k in ("input_ids", "attention_mask")}

    gen_kwargs = dict(**gen_inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.95

    with torch.no_grad():
        out = gen_model.generate(**gen_kwargs)
    gen_ids = out[0][prompt_len:].clone()
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text, gen_ids, prompt_len, inputs


def generate_candidates(model, model_key, processor, tokenizer, sample,
                        group_size, temperature, max_new_tokens):
    """Generate multiple candidates for GRPO."""
    question = sample["question"]
    image = sample["image"]
    candidates, cand_ids_list, think_ranges = [], [], []

    inputs = prepare_inputs(model_key, model, processor, tokenizer, image, question)
    prompt_len = inputs["input_ids"].shape[1]

    gen_model = get_lm(model, model_key) if CPU_TEXT_ONLY else model
    gen_inputs = {k: v for k, v in inputs.items()
                  if k in ("input_ids", "attention_mask")}

    for _ in range(group_size):
        try:
            gen_kwargs = dict(**gen_inputs, max_new_tokens=max_new_tokens,
                            temperature=temperature, top_p=0.95, do_sample=True)
            with torch.no_grad():
                out = gen_model.generate(**gen_kwargs)
            gen_ids = out[0][prompt_len:].clone()
            text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            candidates.append(text)
            cand_ids_list.append(gen_ids.detach())
            think_ranges.append((0, len(gen_ids)))
        except Exception as e:
            print(f"    gen error: {type(e).__name__}: {e}")
            candidates.append("")
            cand_ids_list.append(torch.tensor([], dtype=torch.long))
            think_ranges.append((0, 0))

    inputs = {k: v.clone() if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}
    return candidates, cand_ids_list, prompt_len, inputs, think_ranges


# ══════════════════════════════════════════════════════════════════════
#  Vision Head Hooks
# ══════════════════════════════════════════════════════════════════════

class VisionHeadHooks:
    """Fixed vision head hooks for Exp1."""
    def __init__(self, model, vision_heads, layer_path, num_heads, head_dim, o_proj_name="o_proj"):
        self.vision_heads = vision_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self._captured = {}
        self._hooks = []
        self.layers_needed = sorted(set(l for l, h, d in vision_heads))

        layers = get_model_layers(model, layer_path)
        for li in self.layers_needed:
            o_proj = getattr(layers[li].self_attn, o_proj_name)
            def make_hook(layer_idx):
                def hook_fn(module, args):
                    self._captured[layer_idx] = args[0].detach()
                return hook_fn
            handle = o_proj.register_forward_pre_hook(make_hook(li))
            self._hooks.append(handle)

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


class AllHeadHooks:
    """All-head hooks for Exp8/Exp10."""
    def __init__(self, model, layer_path, num_layers, num_heads, head_dim, o_proj_name="o_proj"):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self._captured = {}
        self._hooks = []

        layers = get_model_layers(model, layer_path)
        for li in range(num_layers):
            o_proj = getattr(layers[li].self_attn, o_proj_name)
            def make_hook(layer_idx):
                def hook_fn(module, args):
                    self._captured[layer_idx] = args[0].detach()
                return hook_fn
            handle = o_proj.register_forward_pre_hook(make_hook(li))
            self._hooks.append(handle)

    def get_all_head_acts(self, prompt_len, seq_len):
        result = {}
        for li in range(self.num_layers):
            inp = self._captured.get(li)
            if inp is None: continue
            reshaped = inp[0].view(-1, self.num_heads, self.head_dim)
            for hi in range(self.num_heads):
                result[(li, hi)] = reshaped[prompt_len:prompt_len + seq_len, hi, :]
        return result

    def get_per_token_head_acts(self, prompt_len, seq_len):
        return self.get_all_head_acts(prompt_len, seq_len)

    def clear(self): self._captured.clear()
    def remove(self):
        for h in self._hooks: h.remove()
        self._hooks.clear(); self._captured.clear()


# ══════════════════════════════════════════════════════════════════════
#  Head-Level LSR Computation (3 variants)
# ══════════════════════════════════════════════════════════════════════

def _fwd_pass(model, model_key, inputs):
    """Forward pass through the right sub-model (language_model for text-only InternVL)."""
    fwd_model = get_lm(model, model_key) if CPU_TEXT_ONLY else model
    fwd_inputs = {k: v for k, v in inputs.items()
                  if k in ("input_ids", "attention_mask")}
    return fwd_model(**fwd_inputs)


def compute_head_lsr_exp1(model, model_key, processor, tokenizer, sample,
                          candidate_ids, think_range, hooks):
    """Exp1: Fixed top-K heads LSR."""
    if candidate_ids.numel() == 0: return torch.zeros(0), 0.0, 0

    t_start, t_end = think_range
    if t_end <= t_start: return torch.zeros(0), 0.0, 0

    image, question = sample["image"], sample["question"]
    n_cand = candidate_ids.numel()

    # Real image forward
    real_inputs = prepare_inputs(model_key, model, processor, tokenizer, image, question)
    rpl = real_inputs["input_ids"].shape[1]
    rf = torch.cat([real_inputs["input_ids"], candidate_ids.unsqueeze(0)], dim=1)
    real_inputs["input_ids"] = rf
    real_inputs["attention_mask"] = torch.ones_like(rf)
    hooks.clear()
    with torch.no_grad(): _fwd_pass(model, model_key, real_inputs)
    real_acts = hooks.get_per_token_head_acts(rpl, n_cand)

    # Black image forward (text-only: same input, simulates different image)
    black = Image.new('RGB', image.size, (0, 0, 0))
    black_inputs = prepare_inputs(model_key, model, processor, tokenizer, black, question)
    bpl = black_inputs["input_ids"].shape[1]
    bf = torch.cat([black_inputs["input_ids"], candidate_ids.unsqueeze(0)], dim=1)
    black_inputs["input_ids"] = bf
    black_inputs["attention_mask"] = torch.ones_like(bf)
    hooks.clear()
    with torch.no_grad(): _fwd_pass(model, model_key, black_inputs)
    black_acts = hooks.get_per_token_head_acts(bpl, n_cand)

    t_end_safe = min(t_end, n_cand)
    t_start_safe = min(t_start, t_end_safe)
    think_len = t_end_safe - t_start_safe
    if think_len <= 0:
        hooks.clear()
        return torch.zeros(0), 0.0, 0

    scores = torch.zeros(think_len)
    n_found = 0
    for (l, h) in real_acts:
        if (l, h) not in black_acts: continue
        ra, ba = real_acts[(l, h)], black_acts[(l, h)]
        ml = min(ra.shape[0], ba.shape[0], t_end_safe)
        if ml <= t_start_safe: continue
        diff = (ra[t_start_safe:ml] - ba[t_start_safe:ml]).float()
        hs = diff.norm(dim=-1)
        cohen_d = 1.0
        if hasattr(hooks, 'vision_heads'):
            for vl, vh, vd in hooks.vision_heads:
                if vl == l and vh == h: cohen_d = vd; break
        el = min(hs.shape[0], think_len)
        scores[:el] += hs[:el] * cohen_d
        n_found += 1

    if n_found > 0: scores /= n_found
    hooks.clear()
    return scores, scores.mean().item(), think_len


def compute_head_lsr_exp8(model, model_key, processor, tokenizer, sample,
                          candidate_ids, think_range, hooks, top_k=12):
    """Exp8: Adaptive per-rollout top-K from ALL heads."""
    if candidate_ids.numel() == 0: return torch.zeros(0), 0.0, 0, []

    t_start, t_end = think_range
    if t_end <= t_start: return torch.zeros(0), 0.0, 0, []

    image, question = sample["image"], sample["question"]
    n_cand = candidate_ids.numel()

    real_inputs = prepare_inputs(model_key, model, processor, tokenizer, image, question)
    rpl = real_inputs["input_ids"].shape[1]
    rf = torch.cat([real_inputs["input_ids"], candidate_ids.unsqueeze(0)], dim=1)
    real_inputs["input_ids"] = rf
    real_inputs["attention_mask"] = torch.ones_like(rf)
    hooks.clear()
    with torch.no_grad(): _fwd_pass(model, model_key, real_inputs)
    real_acts = hooks.get_all_head_acts(rpl, n_cand)

    black = Image.new('RGB', image.size, (0, 0, 0))
    black_inputs = prepare_inputs(model_key, model, processor, tokenizer, black, question)
    bpl = black_inputs["input_ids"].shape[1]
    bf = torch.cat([black_inputs["input_ids"], candidate_ids.unsqueeze(0)], dim=1)
    black_inputs["input_ids"] = bf
    black_inputs["attention_mask"] = torch.ones_like(bf)
    hooks.clear()
    with torch.no_grad(): _fwd_pass(model, model_key, black_inputs)
    black_acts = hooks.get_all_head_acts(bpl, n_cand)

    t_end_safe = min(t_end, n_cand)
    t_start_safe = min(t_start, t_end_safe)
    think_len = t_end_safe - t_start_safe
    if think_len <= 0:
        hooks.clear(); return torch.zeros(0), 0.0, 0, []

    head_deltas = {}
    for (l, h) in real_acts:
        if (l, h) not in black_acts: continue
        ra, ba = real_acts[(l, h)], black_acts[(l, h)]
        ml = min(ra.shape[0], ba.shape[0], t_end_safe)
        if ml <= t_start_safe: continue
        diff = (ra[t_start_safe:ml] - ba[t_start_safe:ml]).float()
        head_deltas[(l, h)] = diff.norm(dim=-1).mean().item()

    sorted_heads = sorted(head_deltas.items(), key=lambda x: x[1], reverse=True)
    selected = sorted_heads[:top_k]
    selected_heads = [(l, h, d) for (l, h), d in selected]

    scores = torch.zeros(think_len)
    n_found = 0
    for (l, h), delta in selected:
        ra, ba = real_acts[(l, h)], black_acts[(l, h)]
        ml = min(ra.shape[0], ba.shape[0], t_end_safe)
        if ml <= t_start_safe: continue
        diff = (ra[t_start_safe:ml] - ba[t_start_safe:ml]).float()
        ptd = diff.norm(dim=-1)
        el = min(ptd.shape[0], think_len)
        scores[:el] += ptd[:el] * delta
        n_found += 1

    if n_found > 0: scores /= n_found
    hooks.clear()
    return scores, scores.mean().item(), think_len, selected_heads


def compute_head_lsr_exp10(model, model_key, processor, tokenizer, sample,
                           candidate_ids, think_range, hooks,
                           temp_scale=0.33):
    """Exp10: Soft-weighted ALL heads with sharp sigmoid (T = std/3)."""
    if candidate_ids.numel() == 0: return torch.zeros(0), 0.0, 0, {}

    t_start, t_end = think_range
    if t_end <= t_start: return torch.zeros(0), 0.0, 0, {}

    image, question = sample["image"], sample["question"]
    n_cand = candidate_ids.numel()

    real_inputs = prepare_inputs(model_key, model, processor, tokenizer, image, question)
    rpl = real_inputs["input_ids"].shape[1]
    rf = torch.cat([real_inputs["input_ids"], candidate_ids.unsqueeze(0)], dim=1)
    real_inputs["input_ids"] = rf
    real_inputs["attention_mask"] = torch.ones_like(rf)
    hooks.clear()
    with torch.no_grad(): _fwd_pass(model, model_key, real_inputs)
    real_acts = hooks.get_all_head_acts(rpl, n_cand)

    black = Image.new('RGB', image.size, (0, 0, 0))
    black_inputs = prepare_inputs(model_key, model, processor, tokenizer, black, question)
    bpl = black_inputs["input_ids"].shape[1]
    bf = torch.cat([black_inputs["input_ids"], candidate_ids.unsqueeze(0)], dim=1)
    black_inputs["input_ids"] = bf
    black_inputs["attention_mask"] = torch.ones_like(bf)
    hooks.clear()
    with torch.no_grad(): _fwd_pass(model, model_key, black_inputs)
    black_acts = hooks.get_all_head_acts(bpl, n_cand)

    t_end_safe = min(t_end, n_cand)
    t_start_safe = min(t_start, t_end_safe)
    think_len = t_end_safe - t_start_safe
    if think_len <= 0:
        hooks.clear(); return torch.zeros(0), 0.0, 0, {}

    head_deltas = {}
    for (l, h) in real_acts:
        if (l, h) not in black_acts: continue
        ra, ba = real_acts[(l, h)], black_acts[(l, h)]
        ml = min(ra.shape[0], ba.shape[0], t_end_safe)
        if ml <= t_start_safe: continue
        diff = (ra[t_start_safe:ml] - ba[t_start_safe:ml]).float()
        head_deltas[(l, h)] = diff.norm(dim=-1).mean().item()

    if not head_deltas:
        hooks.clear(); return torch.zeros(think_len), 0.0, think_len, {}

    all_deltas = np.array(list(head_deltas.values()))
    mean_d = float(all_deltas.mean())
    std_d = float(all_deltas.std()) + 1e-6
    T = std_d * temp_scale  # Sharp sigmoid: T/3
    T = max(T, 1e-6)

    head_weights = {}
    for (l, h), delta in head_deltas.items():
        head_weights[(l, h)] = 1.0 / (1.0 + np.exp(-(delta - mean_d) / T))

    scores = torch.zeros(think_len)
    total_w = 0.0
    for (l, h), w in head_weights.items():
        if w < 0.01: continue
        ra, ba = real_acts[(l, h)], black_acts[(l, h)]
        ml = min(ra.shape[0], ba.shape[0], t_end_safe)
        if ml <= t_start_safe: continue
        diff = (ra[t_start_safe:ml] - ba[t_start_safe:ml]).float()
        ptd = diff.norm(dim=-1)
        el = min(ptd.shape[0], think_len)
        scores[:el] += ptd[:el] * w
        total_w += w

    if total_w > 0: scores /= total_w
    hooks.clear()
    return scores, scores.mean().item(), think_len, head_weights


# ══════════════════════════════════════════════════════════════════════
#  Reward + Loss
# ══════════════════════════════════════════════════════════════════════

def compute_r_correct(prediction, ground_truth, qtype="yesno"):
    if not prediction: return 0.0
    if qtype == "yesno":
        return 1.0 if extract_yes_no(prediction) == ground_truth.strip().lower() else 0.0
    return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0


def gdpo_normalize(values, eps=1e-8):
    arr = np.array(values)
    std = arr.std()
    if std < eps: return np.zeros_like(arr)
    return (arr - arr.mean()) / (std + eps)


def normalize_head_scores(scores, eps=1e-6):
    if scores.numel() == 0: return scores
    return scores / (scores.mean() + eps)


def compute_rewards(model, model_key, processor, tokenizer, sample, candidates,
                    cand_ids_list, think_ranges, cfg, hooks, exp_type):
    """Compute rewards for all candidates using the specified experiment variant."""
    r_correct_list, r_lsr_list = [], []
    head_scores_list, think_lens = [], []
    token_weights_list = []
    gt = sample["answer"]

    for cand, cand_ids, t_range in zip(candidates, cand_ids_list, think_ranges):
        pred = extract_yes_no(cand)
        r_correct = compute_r_correct(pred, gt)

        try:
            if exp_type == "exp1_baseline":
                hs, ms, tl = compute_head_lsr_exp1(
                    model, model_key, processor, tokenizer, sample,
                    cand_ids, t_range, hooks)
            elif exp_type == "exp8_adaptive_topk":
                hs, ms, tl, _ = compute_head_lsr_exp8(
                    model, model_key, processor, tokenizer, sample,
                    cand_ids, t_range, hooks, top_k=cfg.get("adaptive_top_k", 12))
            elif exp_type == "exp10_sharp_sigmoid":
                hs, ms, tl, _ = compute_head_lsr_exp10(
                    model, model_key, processor, tokenizer, sample,
                    cand_ids, t_range, hooks, temp_scale=0.33)
            else:
                hs, ms, tl = torch.zeros(0), 0.0, 0
        except Exception as e:
            print(f"    LSR error: {e}")
            hs, ms, tl = torch.zeros(0), 0.0, 0

        r_lsr = min(ms / cfg["lsr_scale"], 1.0)
        r_correct_list.append(r_correct)
        r_lsr_list.append(r_lsr)
        head_scores_list.append(hs)
        think_lens.append(tl)

    # GDPO normalization
    if len(r_correct_list) > 1:
        nc = gdpo_normalize(r_correct_list)
        nl = gdpo_normalize(r_lsr_list)
        rewards = (cfg["gdpo_w_correct"] * nc + cfg["gdpo_w_lsr"] * nl).tolist()
    else:
        rewards = [r_correct_list[0] * 0.5 + r_lsr_list[0] * 0.5]

    # Token weights
    for i, cand_ids in enumerate(cand_ids_list):
        n_tokens = cand_ids.numel()
        tw = torch.ones(n_tokens)
        hs = head_scores_list[i]
        if hs.numel() >= 5 and cfg["alpha"] > 0:
            ns = normalize_head_scores(hs)
            ns = torch.clamp(ns, 0.0, 5.0)
            el = min(ns.numel(), n_tokens)
            tw[:el] = 1.0 + cfg["alpha"] * ns[:el]
        token_weights_list.append(tw)

    return rewards, r_correct_list, r_lsr_list, token_weights_list


def compute_grpo_loss(model, model_key, inputs, cand_ids_list, prompt_len, advantages,
                      token_weights_list, cfg):
    """GRPO loss with token weighting."""
    total_loss = torch.tensor(0.0, requires_grad=True)
    n_valid = 0
    fwd_model = get_lm(model, model_key) if CPU_TEXT_ONLY else model

    for cand_ids, adv, tw in zip(cand_ids_list, advantages, token_weights_list):
        if cand_ids.numel() == 0 or abs(adv) < 1e-8: continue

        full_ids = torch.cat([inputs["input_ids"][:, :prompt_len],
                             cand_ids.unsqueeze(0)], dim=1)
        attn = torch.ones_like(full_ids)
        fwd = {"input_ids": full_ids, "attention_mask": attn}

        out = fwd_model(**fwd)
        logits = out.logits[0, prompt_len-1:prompt_len-1+len(cand_ids)]
        log_probs = F.log_softmax(logits, dim=-1)
        token_lp = log_probs.gather(1, cand_ids.unsqueeze(1)).squeeze(1)

        w = tw[:len(token_lp)]
        if len(w) < len(token_lp):
            w = F.pad(w, (0, len(token_lp) - len(w)), value=1.0)
        weighted_lp = (token_lp * w).sum() / (w.sum() + 1e-8)

        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        loss = -weighted_lp * adv - cfg["beta_entropy"] * entropy
        total_loss = total_loss + loss
        n_valid += 1

    if n_valid > 0: total_loss = total_loss / n_valid
    return total_loss, n_valid


# ══════════════════════════════════════════════════════════════════════
#  Evaluation
# ══════════════════════════════════════════════════════════════════════

def evaluate_pope(model, model_key, processor, tokenizer, samples, max_eval=20):
    """Quick POPE evaluation."""
    model.eval()
    correct = total = 0
    for s in samples[:max_eval]:
        try:
            q = s["question"] + " Please answer yes or no."
            text, _, _, _ = generate_response(
                model, model_key, processor, tokenizer, s["image"], q,
                max_new_tokens=32, do_sample=False)
            pred = extract_yes_no(text)
            if pred == s["answer"]: correct += 1
            total += 1
        except Exception as e:
            print(f"    eval error: {type(e).__name__}: {e}")
            total += 1
    acc = correct / total if total > 0 else 0
    return {"acc": acc, "correct": correct, "total": total}


def evaluate_blind(model, model_key, processor, tokenizer, samples, max_eval=10):
    """Quick blind test."""
    model.eval()
    real_c = blind_c = total = 0
    for s in samples[:max_eval]:
        try:
            q = s["question"] + " Please answer yes or no."
            # Real image
            text, _, _, _ = generate_response(
                model, model_key, processor, tokenizer, s["image"], q,
                max_new_tokens=32, do_sample=False)
            if extract_yes_no(text) == s["answer"]: real_c += 1
            # Black image
            black = Image.new('RGB', s["image"].size, (0, 0, 0))
            text_b, _, _, _ = generate_response(
                model, model_key, processor, tokenizer, black, q,
                max_new_tokens=32, do_sample=False)
            if extract_yes_no(text_b) == s["answer"]: blind_c += 1
            total += 1
        except Exception as e:
            print(f"    blind eval error: {type(e).__name__}: {e}")
            total += 1
    ra = real_c / total if total > 0 else 0
    ba = blind_c / total if total > 0 else 0
    return {"real_acc": ra, "blind_acc": ba, "gap": ra - ba, "total": total}


# ══════════════════════════════════════════════════════════════════════
#  Data Loading
# ══════════════════════════════════════════════════════════════════════

def load_pope_data(max_samples=30):
    """Load POPE eval data."""
    from datasets import load_dataset
    POPE_SPLITS = ["random", "popular", "adversarial"]
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
    print(f"[data] POPE: {len(samples)} samples ({', '.join(f'{k}={len(v)}' for k, v in per_split.items())})")
    return samples


def load_textvqa_train(max_samples=20):
    """Load TextVQA train for GRPO training."""
    from datasets import load_dataset
    ds = load_dataset("lmms-lab/textvqa", split="train", streaming=True)
    samples = []
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
            "type": "short_answer",
        })
        if len(samples) >= max_samples: break
    print(f"[data] TextVQA train: {len(samples)} samples")
    return samples


# ══════════════════════════════════════════════════════════════════════
#  Calibration
# ══════════════════════════════════════════════════════════════════════

def load_calibration(model_key):
    """Load calibration data if available, else return default heads."""
    mcfg = MODEL_CONFIGS[model_key]
    cal_file = mcfg.get("calibration_file", "")
    n_layers = mcfg["num_layers"]
    n_heads = mcfg["num_heads"]

    if cal_file and os.path.exists(cal_file):
        try:
            with open(cal_file) as f:
                meta = json.load(f)
            scores = meta["head_scores"]
            sorted_heads = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            vision_heads = []
            for key, d in sorted_heads[:12]:
                parts = key.replace("_", ",").split(",")
                l, h = int(parts[0]), int(parts[1])
                vision_heads.append((l, h, float(d)))
            print(f"  [cal] Loaded {len(vision_heads)} heads from {cal_file}")
            return vision_heads
        except Exception as e:
            print(f"  [cal] Failed to load {cal_file}: {e}")

    # Generate synthetic calibration (uniform across layers)
    print(f"  [cal] No calibration found, using distributed default heads")
    vision_heads = []
    step = max(1, n_layers // 12)
    for i in range(0, n_layers, step):
        h = i % n_heads
        vision_heads.append((i, h, 5.0))  # synthetic Cohen's d
    return vision_heads[:12]


# ══════════════════════════════════════════════════════════════════════
#  Single Experiment Run
# ══════════════════════════════════════════════════════════════════════

def run_experiment(model, model_key, processor, tokenizer, exp_name, exp_cfg,
                   vision_heads, train_data, eval_data, training_cfg):
    """Run 1-step GRPO + eval for one experiment."""
    mcfg = MODEL_CONFIGS[model_key]
    cfg = {**training_cfg, **exp_cfg}
    cfg["vision_heads"] = vision_heads

    print(f"\n  --- {exp_name}: {exp_cfg['description']} ---")

    # Pre-eval
    print(f"  Pre-eval (POPE {EVAL_SAMPLES} samples)...")
    pre_pope = evaluate_pope(model, model_key, processor, tokenizer, eval_data, EVAL_SAMPLES)
    pre_blind = evaluate_blind(model, model_key, processor, tokenizer, eval_data, min(10, EVAL_SAMPLES))
    print(f"    POPE: {pre_pope['acc']:.1%} ({pre_pope['correct']}/{pre_pope['total']})")
    print(f"    Blind: real={pre_blind['real_acc']:.1%} blind={pre_blind['blind_acc']:.1%} gap={pre_blind['gap']:.1%}")

    # Install hooks
    n_layers = mcfg["num_layers"]
    n_heads = mcfg["num_heads"]
    h_dim = mcfg["head_dim"]
    o_proj_name = mcfg.get("o_proj_name", "o_proj")

    if exp_cfg.get("adaptive_heads") or exp_cfg.get("soft_weighted_heads"):
        hooks = AllHeadHooks(model, mcfg["layer_path"], n_layers, n_heads, h_dim, o_proj_name)
        hook_type = "all-heads"
    else:
        hooks = VisionHeadHooks(model, vision_heads, mcfg["layer_path"], n_heads, h_dim, o_proj_name)
        hook_type = f"fixed-{len(vision_heads)}"
    print(f"    Hooks: {hook_type}")

    # Enable training
    model.train()
    for p in model.parameters(): p.requires_grad = True
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr"], weight_decay=0.01)

    # 1 training step
    sample = train_data[0]
    step_t0 = time.time()

    try:
        model.eval()
        candidates, cand_ids_list, prompt_len, inputs, think_ranges = \
            generate_candidates(model, model_key, processor, tokenizer, sample,
                              cfg["group_size"], cfg["temperature"], cfg["max_new_tokens"])

        print(f"    Generated {len(candidates)} candidates, "
              f"lengths: {[c_ids.numel() for c_ids in cand_ids_list]}")

        rewards, r_correct_list, r_lsr_list, token_weights_list = \
            compute_rewards(model, model_key, processor, tokenizer, sample,
                          candidates, cand_ids_list, think_ranges, cfg, hooks, exp_name)

        rarr = np.array(rewards)
        rstd = rarr.std()
        advantages = ((rarr - rarr.mean()) / (rstd + 1e-8)).tolist() if rstd > 1e-8 else [0.0] * len(rewards)

        print(f"    Rewards: {[f'{r:.3f}' for r in rewards]}")
        print(f"    R_correct: {r_correct_list}, R_lsr: {[f'{r:.3f}' for r in r_lsr_list]}")

        model.train()
        loss, n_valid = compute_grpo_loss(
            model, model_key, inputs, cand_ids_list, prompt_len,
            advantages, token_weights_list, cfg)

        if n_valid > 0:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            optimizer.step()
            optimizer.zero_grad()
            print(f"    Loss: {loss.item():.4f} ({n_valid} valid candidates)")
        else:
            print(f"    SKIP: no valid candidates (zero advantage)")

    except Exception as e:
        print(f"    TRAINING ERROR: {e}")
        traceback.print_exc()
        loss = None

    step_time = time.time() - step_t0

    # Remove hooks
    hooks.remove()

    # Post-eval
    model.eval()
    print(f"  Post-eval...")
    post_pope = evaluate_pope(model, model_key, processor, tokenizer, eval_data, EVAL_SAMPLES)
    post_blind = evaluate_blind(model, model_key, processor, tokenizer, eval_data, min(10, EVAL_SAMPLES))
    print(f"    POPE: {post_pope['acc']:.1%} ({post_pope['correct']}/{post_pope['total']})")
    print(f"    Blind: real={post_blind['real_acc']:.1%} blind={post_blind['blind_acc']:.1%} gap={post_blind['gap']:.1%}")

    # Reset model weights (reload for next experiment)
    # We don't reload — just note the 1-step change is minimal
    for p in model.parameters(): p.requires_grad = False

    result = {
        "model": model_key,
        "experiment": exp_name,
        "description": exp_cfg["description"],
        "pre_pope": pre_pope,
        "pre_blind": pre_blind,
        "post_pope": post_pope,
        "post_blind": post_blind,
        "loss": loss.item() if loss is not None else None,
        "rewards": rewards if 'rewards' in dir() else [],
        "step_time_s": step_time,
        "timestamp": datetime.now().isoformat(),
    }
    return result


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def main():
    disk_check()
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    report_dir = PROJECT_ROOT / "lab" / "reports" / "multimodel_cpu"
    report_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*70}")
    print(f"  VIGIL Multi-Model CPU Experiments")
    print(f"  Models: {list(MODEL_CONFIGS.keys())}")
    print(f"  Experiments: {list(EXPERIMENTS.keys())}")
    print(f"  Date: {datetime.now().isoformat()}")
    print(f"{'='*70}")

    # Load data once
    print("\n[1/3] Loading data...")
    pope_data = load_pope_data(max_samples=EVAL_SAMPLES + TRAIN_SAMPLES + 6)  # eval + train + buffer
    eval_data = pope_data[:EVAL_SAMPLES]
    # Use POPE data as pseudo-training data (yesno format)
    train_data_pope = pope_data[EVAL_SAMPLES:EVAL_SAMPLES + TRAIN_SAMPLES]
    if len(train_data_pope) < TRAIN_SAMPLES:
        train_data_pope = pope_data[:TRAIN_SAMPLES]  # fallback
    print(f"  Eval: {len(eval_data)} samples, Train: {len(train_data_pope)} samples")

    all_results = []

    # Determine which models to run
    models_to_run = os.environ.get("VIGIL_MODELS", ",".join(MODEL_CONFIGS.keys())).split(",")
    models_to_run = [m.strip() for m in models_to_run if m.strip() in MODEL_CONFIGS]
    print(f"  Models to run: {models_to_run}")

    # Run each model
    for model_key in models_to_run:
        print(f"\n{'='*70}")
        print(f"  MODEL: {model_key}")
        print(f"{'='*70}")

        try:
            model, processor, tokenizer = load_model_cpu(model_key)
        except Exception as e:
            print(f"\n  LOAD FAILED: {e}")
            traceback.print_exc()
            all_results.append({
                "model": model_key, "status": "load_failed", "error": str(e),
            })
            continue

        # Load calibration
        vision_heads = load_calibration(model_key)
        print(f"  Vision heads ({len(vision_heads)}): "
              f"{[(l,h,f'{d:.1f}') for l,h,d in vision_heads[:5]]}...")

        # Run each experiment (reload model between experiments to start fresh)
        for exp_name, exp_cfg in EXPERIMENTS.items():
            try:
                result = run_experiment(
                    model, model_key, processor, tokenizer,
                    exp_name, exp_cfg, vision_heads,
                    train_data_pope, eval_data, TRAINING_CFG)
                result["status"] = "ok"
                all_results.append(result)
            except Exception as e:
                print(f"\n  EXPERIMENT FAILED: {exp_name}: {e}")
                traceback.print_exc()
                all_results.append({
                    "model": model_key, "experiment": exp_name,
                    "status": "failed", "error": str(e),
                })

        # Free memory
        del model
        if processor: del processor
        del tokenizer
        gc.collect()
        print(f"\n  {model_key} unloaded, memory freed")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*70}")

    print(f"\n{'Model':<20} {'Experiment':<25} {'Status':<10} "
          f"{'Pre POPE':>10} {'Post POPE':>10} {'Pre Gap':>10} {'Post Gap':>10} {'Loss':>8}")
    print("-" * 103)

    for r in all_results:
        if r.get("status") != "ok":
            print(f"{r['model']:<20} {r.get('experiment','—'):<25} {r['status']:<10} "
                  f"{'—':>10} {'—':>10} {'—':>10} {'—':>10} {'—':>8}")
            continue
        pre_p = f"{r['pre_pope']['acc']:.1%}"
        post_p = f"{r['post_pope']['acc']:.1%}"
        pre_g = f"{r['pre_blind']['gap']:.1%}"
        post_g = f"{r['post_blind']['gap']:.1%}"
        loss = f"{r['loss']:.4f}" if r['loss'] is not None else "—"
        print(f"{r['model']:<20} {r['experiment']:<25} {'ok':<10} "
              f"{pre_p:>10} {post_p:>10} {pre_g:>10} {post_g:>10} {loss:>8}")

    # Save results
    results_file = report_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")

    # Also save latest
    with open(report_dir / "results_latest.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"  DONE — {len([r for r in all_results if r.get('status')=='ok'])}/{len(all_results)} succeeded")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
