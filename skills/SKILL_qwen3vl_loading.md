# Skill: Loading Qwen3-VL-2B

## Problem
Qwen3-VL-2B uses a different class than Qwen2.5-VL models.

## Solution
```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct",
    torch_dtype=torch.float16,  # NOT bfloat16 — numpy can't convert bf16
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
```

## Layer Access Path
```python
# WRONG: model.model.layers (this is Qwen2.5-VL path)
# RIGHT:
layers = model.model.language_model.layers  # 28 layers
norm = model.model.language_model.norm
lm_head = model.lm_head
```

## Architecture
- 28 layers, GQA 16Q/8KV, head_dim=128, hidden=2048
- ~4.3GB VRAM in fp16
- o_proj input shape: (batch, seq, 2048) = (batch, seq, 16 * 128)

## Gotchas
- `Qwen2_5_VLForConditionalGeneration` CANNOT load Qwen3-VL weights
- bfloat16 tensors cannot be converted to numpy — use `.float()` before `.numpy()`
- Chat template requires structured content: `[{"type": "image", "image": img}, {"type": "text", "text": q}]`
