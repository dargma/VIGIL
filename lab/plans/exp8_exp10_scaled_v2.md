# Exp8 & Exp10 Scaled Training Plan (v2)

## Date: 2026-03-17

## Problem
- Previous scaled runs (v1-v4) all crashed during eval due to OOM on A100 40GB
- Root cause: no `torch.cuda.empty_cache()` before eval → memory fragmentation after training

## Fix Applied
- Added `torch.cuda.empty_cache(); gc.collect()` before eval block (line ~1956)
- Fixed MME train type: `"yes_no"` → `"yesno"`
- Fixed MME train prompt: reasoning-encouraging instruction
- Fixed TextVQA eval: added "Answer briefly." suffix (consistent with train)

## Exp10 Scaled (Priority 1)

**Method**: Sharp sigmoid (T/3) — most stable at 1K scale (4/6 evals at 95% POPE)

```bash
python -u scripts/phase6_head_mask_grpo.py \
    --steps 50 --alpha 0.5 --gdpo --vppo-mask --gated-head-lsr \
    --soft-weighted-heads --soft-temperature auto --soft-temperature-scale 0.33 \
    --eval-every 10 --lr 2e-6 --group-size 6 --temperature 1.3 \
    --train-samples 2000 --include-mme-train --mme-ratio 0.3 --mme-eval-reserve 200 \
    --eval-pope-samples 60 --eval-blind-samples 50 --eval-textvqa-samples 30 \
    --output-dir checkpoints/exp10_sharp_soft/scaled_v5
```

**Data**: 2000 samples (1400 TextVQA + 600 MME train)
**Eval**: POPE 60, Blind 50, TextVQA 30 (reduced for memory)
**Expected**: POPE ≥ 93.3% by step 10, ≥ 95.0% by step 15-20

## Exp8 Scaled (Priority 2, after Exp10)

**Method**: Adaptive top-K per sample — best TextVQA stability at 1K scale

```bash
python -u scripts/phase6_head_mask_grpo.py \
    --steps 50 --alpha 0.5 --gdpo --vppo-mask --gated-head-lsr \
    --adaptive-heads \
    --eval-every 10 --lr 2e-6 --group-size 6 --temperature 1.3 \
    --train-samples 2000 --include-mme-train --mme-ratio 0.3 --mme-eval-reserve 200 \
    --eval-pope-samples 60 --eval-blind-samples 50 --eval-textvqa-samples 30 \
    --output-dir checkpoints/exp8_adaptive_head/scaled_v5
```

**Data**: Same as Exp10
**Expected**: POPE ≥ 95.0%, TextVQA ≥ 72.7% (no degradation)

## Success Criteria
1. No OOM crashes during eval
2. POPE ≥ 95.0% sustained for ≥ 2 eval checkpoints
3. Blind Gap ≥ 44.0pp
4. TextVQA stable (no more than -2pp from baseline)
5. Complete 50 steps without interruption

## Comparison Plan (after both complete)
- Update OPENREVIEW_REPORT.md with final scaled results
- Regenerate CAM figures with scaled data
- Git commit + push
