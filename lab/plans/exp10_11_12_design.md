# Exp10, Exp11, Exp12: Iterating on Soft-Weighted Head LSR

## Exp9 Interim Results (step 5)

- POPE: 95.0% (matches Exp8)
- Gap: 44.0pp (matches Exp8)
- TextVQA: 72.7% (stable)
- **Problem**: 448 active heads, ~400 mid-weight (0.3-0.8), ~45 high-weight (>0.8), 0 low-weight
- **Diagnosis**: `T = std(deltas)` makes sigmoid too flat — most heads get ~0.5 weight
- Effectively degenerates to uniform weighting, which shouldn't work as well as top-K... but it does. Why?

### Why Exp9 works despite flat weights:
Even with ~uniform weights, the per-token delta computation still captures vision-relevant tokens correctly because high-delta heads have both high weight AND high per-token delta — so their contribution to the weighted sum dominates naturally via the delta magnitude, not the weight.

## Exp10: Sharper Temperature (T = std/3)

**Hypothesis**: Making the sigmoid sharper will concentrate weight on truly high-delta heads, similar to Exp8 but without the hard cutoff.

**Changes from Exp9**:
- Temperature: `T = std(deltas) / 3` — 3× sharper sigmoid
- Expected: ~50 high, ~100 mid, ~300 low weight heads
- This creates a "soft top-50" effect — much more concentrated than Exp9's near-uniform

**Implementation**: Add `--soft-temperature-scale` parameter (default 1.0, Exp10 uses 0.33)

```bash
python -u scripts/phase6_head_mask_grpo.py \
    --steps 30 --alpha 0.5 --gdpo --vppo-mask --gated-head-lsr \
    --soft-weighted-heads --soft-temperature auto --soft-temperature-scale 0.33 \
    --eval-every 5 --lr 2e-6 --group-size 6 --temperature 1.3 \
    --train-samples 1000 \
    --output-dir checkpoints/exp10_sharp_soft/run1
```

## Exp11: Layer-Aware Weighting

**Hypothesis**: Not all layers should contribute equally. Late feature layers (L24-27) and early decision layers (L4-5) are more important than mid-layers.

**Changes from Exp9**:
- Layer multiplier: `layer_weight(l) = 1.0 + bonus(l)` where:
  - Decision layers (L4-5): bonus = 1.0 (2× weight)
  - Feature layers (L24-27): bonus = 0.5 (1.5× weight)
  - Other layers: bonus = 0.0 (normal weight)
- Final weight: `w(l,h) = sigmoid(delta) × layer_weight(l)`

**Implementation**: Add `--layer-aware` flag + layer bonus config

```bash
python -u scripts/phase6_head_mask_grpo.py \
    --steps 30 --alpha 0.5 --gdpo --vppo-mask --gated-head-lsr \
    --soft-weighted-heads --soft-temperature auto --layer-aware \
    --eval-every 5 --lr 2e-6 --group-size 6 --temperature 1.3 \
    --train-samples 1000 \
    --output-dir checkpoints/exp11_layer_aware/run1
```

## Exp12: Top-P Soft Selection

**Hypothesis**: Instead of using all heads (like Exp9) or top-K (like Exp8), use **top-P** — keep adding heads by descending weight until cumulative weight reaches P=0.9, then zero out the rest.

**Changes from Exp9**:
- Compute sigmoid weights for all 448 heads
- Sort by weight (descending)
- Accumulate weights until cumsum reaches p × total_weight
- Zero-out remaining heads, renormalize kept heads
- Expected: ~30-80 heads kept (adaptive — varies per sample)
- This is the "soft top-K" that naturally adapts the number of heads

**Implementation**: Add `--top-p-heads` parameter (default 0.0 = disabled, Exp12 uses 0.9)

```bash
python -u scripts/phase6_head_mask_grpo.py \
    --steps 30 --alpha 0.5 --gdpo --vppo-mask --gated-head-lsr \
    --soft-weighted-heads --soft-temperature auto --top-p-heads 0.9 \
    --eval-every 5 --lr 2e-6 --group-size 6 --temperature 1.3 \
    --train-samples 1000 \
    --output-dir checkpoints/exp12_top_p_heads/run1
```

## Execution Plan

1. Wait for Exp9 to finish (step 30)
2. Implement Exp10/11/12 code changes (all in `compute_soft_weighted_head_lsr()`)
3. Run Exp10 → Exp11 → Exp12 sequentially (each ~10 min for 30 steps)
4. Compare all results (Exp8, 9, 10, 11, 12) in a comprehensive table
5. Select best for scaled training (50 steps, 2K samples)

## Decision Criteria

Best method is selected by:
1. POPE accuracy (primary)
2. Blind Gap (secondary — vision grounding)
3. TextVQA stability (should not degrade)
4. Efficiency (fewer active heads = faster, if accuracy is equal)
