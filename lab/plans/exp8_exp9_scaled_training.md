# Exp8 & Exp9 Scaled Training Plan

## Goal

Run Exp8 and Exp9 with **scaled training dataset** for **50 steps**, evaluating every **10 steps** across 4 benchmarks with **200 samples** each.

## Training Configuration

### Shared Config
- **Steps**: 50
- **Eval every**: 10 steps (steps 10, 20, 30, 40, 50)
- **Learning rate**: 2e-6
- **Group size**: 6
- **Temperature**: 1.3
- **Top-p**: 0.95
- **Flags**: `--gdpo --vppo-mask --gated-head-lsr`
- **Alpha**: 0.5
- **Beta decay**: 0.1

### Training Dataset (Scaled)
- **TextVQA train**: 2000 samples (up from 1000)
- **Total**: 2000 samples
- **Selection**: `--train-samples 2000 --seed 42`

### Evaluation (200 samples each, every 10 steps)
| Benchmark | Samples | Metric |
|-----------|---------|--------|
| POPE (3 splits) | 200 | Accuracy, F1 |
| Blind Test (Gap) | 200 | Gap = acc(real) - acc(black) |
| TextVQA | 200 | Accuracy |
| MME | 200 | Perception + Cognition |

## Exp8: Per-Rollout Adaptive Head Gate

```bash
PYTHONUNBUFFERED=1 python -u scripts/phase6_head_mask_grpo.py \
    --steps 50 \
    --alpha 0.5 \
    --gdpo \
    --vppo-mask \
    --gated-head-lsr \
    --adaptive-heads \
    --adaptive-top-k 12 \
    --eval-every 10 \
    --lr 2e-6 \
    --group-size 6 \
    --temperature 1.3 \
    --train-samples 2000 \
    --output-dir checkpoints/exp8_scaled/run1 \
    2>&1 | tee logs/exp8_scaled.log
```

## Exp9: Soft-Weighted All-Head LSR

```bash
PYTHONUNBUFFERED=1 python -u scripts/phase6_head_mask_grpo.py \
    --steps 50 \
    --alpha 0.5 \
    --gdpo \
    --vppo-mask \
    --gated-head-lsr \
    --soft-weighted-heads \
    --soft-temperature auto \
    --eval-every 10 \
    --lr 2e-6 \
    --group-size 6 \
    --temperature 1.3 \
    --train-samples 2000 \
    --output-dir checkpoints/exp9_scaled/run1 \
    2>&1 | tee logs/exp9_scaled.log
```

## Evaluation Changes Required

The current eval functions in `phase6_head_mask_grpo.py` use:
- POPE: 60 samples → need 200
- Blind: 50 samples → need 200
- TextVQA: 50 samples → need 200
- MME: not yet integrated → need to add

### Changes needed:
1. Add `--eval-pope-samples`, `--eval-blind-samples`, `--eval-textvqa-samples` CLI args
2. Integrate MME evaluation (import from `scripts/eval_mme.py`)
3. Default all to 200

## Execution Order

1. Wait for current Exp9 (30 steps, 1K samples) to finish
2. Modify `phase6_head_mask_grpo.py` to support configurable eval samples + MME
3. Run Exp8 scaled (50 steps, 2K samples)
4. Run Exp9 scaled (50 steps, 2K samples)
5. Generate comparative report with full results table

## Expected Timeline
- Exp8: ~50 steps × ~3 min/step = ~2.5 hours
- Exp9: ~50 steps × ~4 min/step = ~3.3 hours (more heads to score)
- Total: ~6 hours sequential

## Report Requirements
- Training dataset type and count explicitly stated
- Per-step results table (steps 10, 20, 30, 40, 50)
- All 4 benchmarks reported
- Comparison with Exp1 baseline and prior Exp8/Exp9 results
