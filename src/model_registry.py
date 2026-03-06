"""
VIGIL Model Registry — unified loading and architecture verification for target VLMs.

Supports: Qwen3-VL-2B, InternVL3.5-1B, DeepSeek-VL2-Tiny, Qwen2.5-VL-7B
Returns standardized model_info dict with get_layers_fn, architecture metadata, etc.
"""

import yaml
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional

CONFIGS_PATH = Path(__file__).parent.parent / "configs" / "models.yaml"


@dataclass
class ModelSpec:
    name: str
    hf_id: str
    model_type: str
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    hidden_size: int
    gqa: bool
    steer_layers_start: int = 0
    trust_remote_code: bool = False
    is_moe: bool = False
    eval_only: bool = False


def load_model_specs() -> Dict[str, ModelSpec]:
    with open(CONFIGS_PATH) as f:
        cfg = yaml.safe_load(f)
    specs = {}
    for key, v in cfg["models"].items():
        specs[key] = ModelSpec(
            name=v["name"],
            hf_id=v["hf_id"],
            model_type=v["model_type"],
            num_hidden_layers=v["num_hidden_layers"],
            num_attention_heads=v["num_attention_heads"],
            num_key_value_heads=v["num_key_value_heads"],
            head_dim=v["head_dim"],
            hidden_size=v["hidden_size"],
            gqa=v.get("gqa", False),
            steer_layers_start=v.get("steer_layers_start", 0),
            trust_remote_code=v.get("trust_remote_code", False),
            is_moe=v.get("is_moe", False),
            eval_only=v.get("eval_only", False),
        )
    return specs


def _verify_architecture(model, spec: ModelSpec):
    """Abort if loaded model doesn't match expected architecture."""
    config = model.config
    # Try to find the LLM config (may be nested)
    llm_config = getattr(config, "llm_config", None) or getattr(config, "text_config", None) or config

    checks = {
        "num_hidden_layers": (getattr(llm_config, "num_hidden_layers", None), spec.num_hidden_layers),
        "num_attention_heads": (getattr(llm_config, "num_attention_heads", None), spec.num_attention_heads),
        "num_key_value_heads": (getattr(llm_config, "num_key_value_heads", None), spec.num_key_value_heads),
        "hidden_size": (getattr(llm_config, "hidden_size", None), spec.hidden_size),
    }
    head_dim_actual = getattr(llm_config, "head_dim", None)
    if head_dim_actual is None and checks["hidden_size"][0] and checks["num_attention_heads"][0]:
        head_dim_actual = checks["hidden_size"][0] // checks["num_attention_heads"][0]
    checks["head_dim"] = (head_dim_actual, spec.head_dim)

    print(f"\n[VIGIL] Architecture verification for {spec.name}:")
    all_ok = True
    for param, (actual, expected) in checks.items():
        status = "OK" if actual == expected else "MISMATCH"
        if status == "MISMATCH":
            all_ok = False
        print(f"  {param}: expected={expected}, actual={actual} [{status}]")

    if not all_ok:
        raise ValueError(f"Architecture mismatch for {spec.name}. Aborting.")
    print(f"  All checks passed.\n")


def _load_qwen3_vl(spec: ModelSpec, dtype, device) -> Dict[str, Any]:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        spec.hf_id, torch_dtype=dtype, device_map=device,
    )
    processor = AutoProcessor.from_pretrained(spec.hf_id)
    return {
        "model": model,
        "processor": processor,
        "tokenizer": processor.tokenizer,
        "get_layers_fn": lambda: model.model.layers,
        "get_lm_head_fn": lambda: model.lm_head,
        "get_norm_fn": lambda: model.model.norm,
    }


def _load_internvl3(spec: ModelSpec, dtype, device) -> Dict[str, Any]:
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained(
        spec.hf_id, torch_dtype=dtype, device_map=device,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, trust_remote_code=True)
    return {
        "model": model,
        "processor": None,
        "tokenizer": tokenizer,
        "get_layers_fn": lambda: model.language_model.model.layers,
        "get_lm_head_fn": lambda: model.language_model.lm_head,
        "get_norm_fn": lambda: model.language_model.model.norm,
    }


def _load_deepseek_vl2(spec: ModelSpec, dtype, device) -> Dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, torch_dtype=dtype, device_map=device,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, trust_remote_code=True)
    return {
        "model": model,
        "processor": None,
        "tokenizer": tokenizer,
        "get_layers_fn": lambda: model.model.layers,
        "get_lm_head_fn": lambda: model.lm_head,
        "get_norm_fn": lambda: model.model.norm,
    }


def _load_qwen2_vl(spec: ModelSpec, dtype, device) -> Dict[str, Any]:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        spec.hf_id, torch_dtype=dtype, device_map=device,
    )
    processor = AutoProcessor.from_pretrained(spec.hf_id)
    return {
        "model": model,
        "processor": processor,
        "tokenizer": processor.tokenizer,
        "get_layers_fn": lambda: model.model.layers,
        "get_lm_head_fn": lambda: model.lm_head,
        "get_norm_fn": lambda: model.model.norm,
    }


_LOADERS = {
    "qwen3_vl": _load_qwen3_vl,
    "internvl3": _load_internvl3,
    "deepseek_vl2": _load_deepseek_vl2,
    "qwen2_vl": _load_qwen2_vl,
}


def load_model(
    model_key: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "auto",
) -> Dict[str, Any]:
    """Load model by registry key. Returns standardized model_info dict."""
    specs = load_model_specs()
    if model_key not in specs:
        raise KeyError(f"Unknown model key '{model_key}'. Available: {list(specs.keys())}")

    spec = specs[model_key]
    loader = _LOADERS.get(spec.model_type)
    if loader is None:
        raise ValueError(f"No loader for model_type '{spec.model_type}'")

    print(f"[VIGIL] Loading {spec.name} from {spec.hf_id}...")
    info = loader(spec, dtype, device)
    _verify_architecture(info["model"], spec)

    info.update({
        "spec": spec,
        "model_key": model_key,
        "model_type": spec.model_type,
        "num_layers": spec.num_hidden_layers,
        "num_heads": spec.num_attention_heads,
        "num_kv_heads": spec.num_key_value_heads,
        "head_dim": spec.head_dim,
        "hidden_size": spec.hidden_size,
        "gqa": spec.gqa,
        "is_moe": spec.is_moe,
        "steer_layers_start": spec.steer_layers_start,
        "device": next(info["model"].parameters()).device,
    })
    return info


def make_chat_prompt(model_info: Dict[str, Any], question: str, image=None) -> dict:
    """Create model-specific chat input. Returns dict with input_ids, etc."""
    model_type = model_info["model_type"]
    processor = model_info.get("processor")
    tokenizer = model_info.get("tokenizer")

    if model_type in ("qwen3_vl", "qwen2_vl"):
        messages = [{"role": "user", "content": []}]
        if image is not None:
            messages[0]["content"].append({"type": "image", "image": image})
        messages[0]["content"].append({"type": "text", "text": question})
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image] if image is not None else None,
                           return_tensors="pt", padding=True)
        return {k: v.to(model_info["device"]) for k, v in inputs.items()}

    elif model_type == "internvl3":
        if image is not None:
            prompt = f"<image>\n{question}"
        else:
            prompt = question
        inputs = tokenizer(prompt, return_tensors="pt")
        return {k: v.to(model_info["device"]) for k, v in inputs.items()}

    elif model_type == "deepseek_vl2":
        if image is not None:
            prompt = f"<image>\n{question}"
        else:
            prompt = question
        inputs = tokenizer(prompt, return_tensors="pt")
        return {k: v.to(model_info["device"]) for k, v in inputs.items()}

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
