# Skill: o_proj Pre-Hooks for Head-Level Activation Capture

## Problem
Need to capture per-Q-head activations before o_proj merges them.

## Solution
```python
captured = {}

def make_hook(layer_idx):
    def hook_fn(module, args):
        # args[0] shape: (batch, seq, num_Q_heads * head_dim)
        captured[layer_idx] = args[0].detach()
    return hook_fn

layers = model.model.language_model.layers
for li, layer in enumerate(layers):
    layer.self_attn.o_proj.register_forward_pre_hook(make_hook(li))
```

## Extracting Per-Head Vectors
```python
# For last token position:
inp = captured[layer_idx]  # (batch, seq, 2048)
last = inp[0, -1, :]       # (2048,)
per_head = last.view(num_Q_heads, head_dim)  # (16, 128)
head_vec = per_head[head_idx]  # (128,)
```

## Key Facts
- Use `register_forward_pre_hook` (NOT `register_forward_hook`) — pre-hook sees input BEFORE o_proj projects
- For GQA: reshape by num_Q_heads (16), NOT num_KV_heads (8)
- Hooks fire during both forward() and generate()
- Always `.detach()` to avoid graph leaks
- Clear `captured` dict between samples to avoid memory growth
