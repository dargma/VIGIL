# Exp9: Soft-Weighted All-Head LSR

## Motivation (from user feedback)

Exp8 selects top-K heads per sample — but this has problems:

1. **Top-K is discrete**: heads #12 and #13 might have delta 5.1 and 5.0, but one gets full weight and the other gets zero
2. **"Vision head should have high score" is wrong assumption**: Some heads do vision processing with LOWER activation — e.g., suppression heads that quiet non-visual noise
3. **Mixed heads ignored**: A head at 0.5 weight (text+vision dual-use) carries useful signal but gets discarded by top-K
4. **Model internals are continuous**: The latent space doesn't operate in discrete top-K; attention weights are soft distributions

## Core Idea

Replace discrete top-K selection with **continuous soft weights** derived from the real-vs-black activation delta:

```
weight(l, h) = softmax(delta(l, h) / temperature)   # across ALL 448 heads
score(t) = Σ_{all l,h} weight(l,h) × ||act_real[l,h,t] - act_black[l,h,t]||
```

This means:
- **ALL 448 heads contribute** — no hard cutoff
- High-delta heads get high weight naturally (softmax concentrates)
- Low-delta heads get low but non-zero weight (they still contribute)
- Temperature controls sharpness: low T → approaches top-K, high T → approaches uniform
- Mixed text/vision heads with moderate delta get moderate weight (~0.05-0.1)

## Key Design Decisions

### Temperature Selection
- `T = mean(deltas)` → adaptive temperature that scales with the delta distribution
- This ensures weights are neither too sharp (top-1 dominates) nor too flat (uniform)
- Alternative: `T = percentile_75(deltas)` — top quarter gets meaningful weight

### Weight Normalization
- Softmax ensures weights sum to 1.0 — comparable reward scale across samples
- But softmax over 448 heads is very flat. Better: **top-p style cutoff** after softmax
  - Compute softmax weights for all 448
  - Cumulative sum until reaching p=0.9
  - Zero-out heads below threshold
  - Renormalize remaining weights
  - This keeps the continuous nature while ignoring truly irrelevant heads

### Alternative: Sigmoid + Normalize
Instead of softmax (competitive), use sigmoid (independent):
```
weight(l,h) = sigmoid((delta(l,h) - mean_delta) / temperature)
```
- Each head gets weight independently based on whether its delta is above/below average
- Heads with delta >> mean → weight ≈ 1.0
- Heads with delta << mean → weight ≈ 0.0
- Heads with delta ≈ mean → weight ≈ 0.5 (the "mixed" heads!)
- No normalization needed since VPPO masking handles scale

**Recommendation: Use sigmoid approach** — it naturally captures the "some heads are 0.5 weight" intuition.

## Implementation

Based on Exp8's `compute_adaptive_head_lsr()`, modify:

```python
def compute_soft_weighted_head_lsr(model, processor, sample, candidate_ids,
                                    think_range, device, hooks, temperature="auto"):
    # ... same real/black forward passes as Exp8 ...

    # Compute per-head mean delta (same as Exp8)
    head_deltas = {}
    for (l, h) in real_acts:
        ...
        head_deltas[(l, h)] = mean_delta

    # NEW: Soft weights instead of top-K selection
    all_deltas = np.array(list(head_deltas.values()))
    mean_delta = all_deltas.mean()
    std_delta = all_deltas.std() + 1e-6

    if temperature == "auto":
        temperature = std_delta  # adaptive T

    # Sigmoid: each head independently weighted
    weights = {}
    for (l, h), delta in head_deltas.items():
        w = 1.0 / (1.0 + np.exp(-(delta - mean_delta) / temperature))
        weights[(l, h)] = w

    # Per-token score: weighted sum over ALL heads
    scores = torch.zeros(think_len, device=device)
    total_weight = 0.0

    for (l, h), w in weights.items():
        if w < 0.01:  # skip truly negligible heads for efficiency
            continue
        ra = real_acts[(l, h)]
        ba = black_acts[(l, h)]
        ...
        per_token_delta = diff.norm(dim=-1)
        scores[:effective_len] += per_token_delta[:effective_len] * w
        total_weight += w

    if total_weight > 0:
        scores /= total_weight

    return scores, mean_score, think_len, weights
```

## What This Captures That Exp8 Misses

1. **Suppression heads**: Heads with low delta but important for suppressing hallucination — get small but non-zero weight
2. **Mixed heads**: Heads that process both text and vision (delta ≈ mean) — get ~0.5 weight
3. **Smooth gradient landscape**: No discontinuity at the top-K boundary
4. **Per-sample adaptive temperature**: Different images have different delta distributions; T adapts

## Run Command

```bash
PYTHONUNBUFFERED=1 python -u scripts/phase6_head_mask_grpo.py \
    --steps 30 \
    --alpha 0.5 \
    --gdpo \
    --vppo-mask \
    --gated-head-lsr \
    --soft-weighted-heads \
    --soft-temperature auto \
    --eval-every 5 \
    --lr 2e-6 \
    --group-size 6 \
    --temperature 1.3 \
    --train-samples 1000 \
    --output-dir checkpoints/exp9_soft_weighted/run1 \
    2>&1 | tee logs/exp9_soft_weighted.log
```

## Expected vs Exp8

| Aspect | Exp8 | Exp9 |
|--------|------|------|
| Head selection | Top-12 (discrete) | All 448 (continuous weights) |
| Weight range | 0 or delta_value | sigmoid(0-1) |
| Mixed heads | Excluded | Included (~0.3-0.7 weight) |
| Low-delta heads | Excluded | Included (~0.01-0.1 weight) |
| Temperature | N/A (top-K cutoff) | Adaptive (std of deltas) |
| Efficiency | 12 heads scored | ~50-100 heads scored (w > 0.01) |
| Signal | Strong (top-K are strongest) | Broader (includes subtle signals) |
