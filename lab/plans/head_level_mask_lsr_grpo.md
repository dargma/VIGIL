# Head-Level Mask LSR-Weighted GRPO — Design Plan

## Concept

Current Token-LSR computes `KL(P_real || P_black)` at the **output logit level** per token.
Head-Level Mask LSR goes **deeper** — computing activation difference at **individual attention heads** per token position.

```
Token-LSR:     logits(real) vs logits(black) → per-token KL → token weight
Head-LSR:      head_act(real) vs head_act(black) → per-head-per-token Δ → token weight
```

## Why Head-Level is Better

1. **More targeted**: Vision heads (identified by Cohen's d calibration) directly measure image dependency
2. **Avoids logit noise**: Output logits mix all heads including non-vision ones — dilutes the signal
3. **Granular signal**: Can identify WHICH heads lose image attention at WHICH tokens
4. **Direct connection to drift**: O(1/L) drift is a HEAD-level phenomenon, not a logit-level one
5. **Publishable novelty**: Combines VIGIL's head-level profiling with VPPO's token-level approach

## Architecture

```
For each candidate in GRPO group:
  1. Forward pass with real image → capture vision head activations at ALL token positions
  2. Forward pass with black image → capture vision head activations at ALL token positions
  3. Per-token vision score: mean(|act_real[h,t] - act_black[h,t]|) for h in vision_heads
  4. Normalize: token_weight(t) = 1.0 + alpha * normalized_head_score(t)
  5. Use token_weight in GRPO loss (same as Token-LSR)
```

## Key Difference from Token-LSR

| Aspect | Token-LSR (current) | Head-Level Mask LSR |
|--------|---------------------|---------------------|
| Signal source | Output logits KL | Vision head activations |
| Compute cost | 2 forward passes per candidate | 2 forward passes + hook extraction |
| Granularity | Token-level only | Head × Token (can aggregate) |
| Vision specificity | Low (all heads mixed) | High (only vision heads) |
| Hook requirement | None (uses logits) | o_proj pre-hooks (already in profiler.py) |

## Implementation Plan

### Step 1: Head-Level Token Score Function
```python
def compute_head_level_lsr(model, processor, sample, candidate_ids,
                           think_range, device, vision_heads):
    """
    Returns per-token vision score using head-level activation differences.

    vision_heads: list of (layer_idx, head_idx) from calibration
    """
    # Install hooks on vision head layers only (efficient)
    hooks = install_vision_hooks(model, vision_heads)

    # Forward with real image
    real_inputs = prepare_teacher_forced(processor, sample["image"],
                                         sample["question"], candidate_ids, device)
    with torch.no_grad():
        model(**real_inputs)
    real_acts = extract_per_token_head_acts(hooks, vision_heads)  # {(l,h): (seq_len, head_dim)}

    # Forward with black image
    black_image = Image.new('RGB', sample["image"].size, (0, 0, 0))
    black_inputs = prepare_teacher_forced(processor, black_image,
                                          sample["question"], candidate_ids, device)
    with torch.no_grad():
        model(**black_inputs)
    black_acts = extract_per_token_head_acts(hooks, vision_heads)

    remove_hooks(hooks)

    # Per-token vision score = mean head activation difference
    t_start, t_end = think_range
    scores = []
    for t in range(t_start, t_end):
        head_diffs = []
        for (l, h) in vision_heads:
            diff = (real_acts[(l,h)][t] - black_acts[(l,h)][t]).norm()
            head_diffs.append(diff)
        scores.append(torch.stack(head_diffs).mean())

    return torch.stack(scores)  # (think_len,)
```

### Step 2: Hook Design (Efficient)

Unlike profiler which captures last-token only, we need ALL token positions:
```python
def install_vision_hooks(model, vision_heads):
    """Install hooks only on layers that contain vision heads."""
    layers_needed = set(l for l, h in vision_heads)
    hooks = {}
    for li in layers_needed:
        layer = model.model.language_model.layers[li]
        def make_hook(layer_idx):
            def hook_fn(module, args):
                # args[0]: (batch, seq, num_heads * head_dim)
                hooks[f"acts_{layer_idx}"] = args[0].detach()
            return hook_fn
        handle = layer.self_attn.o_proj.register_forward_pre_hook(make_hook(li))
        hooks[f"handle_{li}"] = handle
    return hooks
```

Key optimization: only hook layers containing vision heads (typically 3-5 layers out of 28).

### Step 3: Integration into GRPO

Replace `compute_token_lsr()` with `compute_head_level_lsr()` in the reward/weight computation.
Everything else (normalize, decay penalty, weighted loss) stays the same.

### Step 4: Calibration Data Requirement

Need vision head list from calibration. Currently stored in `skills/SKILL_calibration_results.md`:
- Top vision heads by Cohen's d: L4H13, L5H3, L24H8, L25H15, etc.
- Feature heads (high Δ): L24-27
- Decision heads (high Cohen's d): L4-5

Use **both** types for comprehensive coverage.

## Compute Budget

Per candidate: 2 forward passes (same as Token-LSR) + hook extraction (negligible).
Only difference: hook data is larger (all positions, not just logits).
Memory: ~extra 20MB per candidate for stored activations (manageable).

## Expected Improvements

1. **Sharper signal**: Vision heads directly measure image usage vs logit KL which is noisy
2. **Better decay detection**: Can see exact head × position where drift occurs
3. **Head-specific weight**: Can weight by head importance (Cohen's d × activation diff)
4. **Novel figure**: Head × Token heatmap showing drift (Figure 1 candidate for paper)

## Risks

1. Hook overhead might slow training ~10% (mitigated by hooking only vision head layers)
2. Vision heads identified on calibration data may not generalize (mitigated by using robust Cohen's d)
3. All-positions hook data may be too large for long sequences (mitigated by truncating to thinking tokens only)

## Execution Plan

1. **Phase A**: Implement `compute_head_level_lsr()` with existing profiler hooks
2. **Phase B**: Run 20-step comparison: Head-LSR vs Token-LSR vs standard GRPO
3. **Phase C**: Generate head × token heatmaps for visualization
4. **Phase D**: If Head-LSR wins, run full training (50 steps, multi-round)

## Priority: HIGH
- Novel contribution (no prior work combines head-level profiling with token-level GRPO weighting)
- Directly addresses the paper's core thesis (vision drift is a head-level phenomenon)
- Reuses existing VIGIL infrastructure (profiler, calibration, hooks)
- Minimal code changes (swap one function in the training loop)
