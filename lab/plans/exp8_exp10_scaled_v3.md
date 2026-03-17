# Exp8 & Exp10 Scaled Training Plan (v3)

## Date: 2026-03-17

## Changes from v2
- **Fixed data coverage**: v2 used 1 sample/step → 50/2000 = 2.5% coverage
- **Added `--samples-per-step N`**: processes N samples per optimizer step (mini-batch)
- **Deterministic data iterator**: epoch-aware reshuffling with seed, identical order across experiments
- **Removed `--grad-accum`** dependency: replaced by `--samples-per-step` which is conceptually cleaner

## Data Coverage Analysis

| Config | Steps | Samples/Step | Total Samples | Coverage |
|--------|-------|-------------|---------------|----------|
| v2 (old) | 50 | 1 | 50 | 2.5% |
| v3 (conservative) | 50 | 4 | 200 | 10% |
| v3 (recommended) | 50 | 8 | 400 | 20% |
| v3 (full) | 50 | 40 | 2000 | 100% |

**Recommendation**: `--samples-per-step 4` — 4× more data, ~4× slower per step (~200s vs ~50s).
At ~200s/step × 50 steps ≈ 2.8 hours. Feasible on A100.

## Seed Guarantee

Both experiments use `--seed 42`:
1. `random.seed(42)` → Python random state
2. `np.random.seed(42)` → NumPy random state
3. `torch.manual_seed(42)` → PyTorch random state
4. `load_training_data(seed=42)` → data loading uses `random.Random(42)` for shuffle
5. `data_rng = random.Random(42)` → training data order uses separate RNG (deterministic)

**Result**: Exp8 and Exp10 with same `--seed 42 --train-samples 2000 --samples-per-step 4`
will process the exact same samples in the exact same order.

## Exp10 Scaled v6 (Priority 1)

**Method**: Sharp sigmoid (T/3)

```bash
python -u scripts/phase6_head_mask_grpo.py \
    --steps 50 --alpha 0.5 --gdpo --vppo-mask --gated-head-lsr \
    --soft-weighted-heads --soft-temperature auto --soft-temperature-scale 0.33 \
    --eval-every 10 --lr 2e-6 --group-size 6 --temperature 1.3 \
    --train-samples 2000 --samples-per-step 4 \
    --include-mme-train --mme-ratio 0.3 --mme-eval-reserve 200 \
    --eval-pope-samples 60 --eval-blind-samples 50 --eval-textvqa-samples 30 \
    --seed 42 \
    --output-dir checkpoints/exp10_sharp_soft/scaled_v6
```

**Data**: 2000 samples (1400 TextVQA + 600 MME), 4 per step, 200 total (10%)
**Expected**: POPE ≥ 93.3% by step 10, ≥ 95.0% by step 20

## Exp8 Scaled v6 (Priority 2)

**Method**: Adaptive top-K per sample

```bash
python -u scripts/phase6_head_mask_grpo.py \
    --steps 50 --alpha 0.5 --gdpo --vppo-mask --gated-head-lsr \
    --adaptive-heads \
    --eval-every 10 --lr 2e-6 --group-size 6 --temperature 1.3 \
    --train-samples 2000 --samples-per-step 4 \
    --include-mme-train --mme-ratio 0.3 --mme-eval-reserve 200 \
    --eval-pope-samples 60 --eval-blind-samples 50 --eval-textvqa-samples 30 \
    --seed 42 \
    --output-dir checkpoints/exp8_adaptive_head/scaled_v6
```

**Data**: Same as Exp10 (identical order, identical samples)
**Expected**: POPE ≥ 95.0%, TextVQA ≥ 72.7%

## Success Criteria
1. No OOM crashes during eval
2. POPE ≥ 95.0% sustained for ≥ 2 eval checkpoints
3. Blind Gap ≥ 44.0pp
4. TextVQA stable (no more than -2pp from baseline)
5. Complete 50 steps without interruption
6. **NEW**: Both experiments see identical training samples (verify via `samples_seen` log)

## Comparison Plan (after both complete)
- Compare identical-data results: Exp8 vs Exp10
- Update OPENREVIEW_REPORT.md with final scaled results
- Regenerate CAM figures with scaled data
- Git commit + push
