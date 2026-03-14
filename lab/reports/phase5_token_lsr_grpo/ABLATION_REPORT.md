# Phase 5: Token-Level LSR-Weighted GRPO — Ablation Results

**Date**: 2026-03-14
**Base model**: `checkpoints/phase2_grpo_lsr/round4/best` (POPE 95.0%, Gap 44.0pp)

---

## Ablation Summary

| Run | alpha | beta_decay | lr | Pre-POPE | Step 5 | Step 10 | Step 15 | Delta |
|-----|-------|------------|------|----------|--------|---------|---------|-------|
| R1 (original) | 0.5 | 0.1 | 1e-6 | 95.0% | 93.3% | 95.0% | 91.7% | -3.3pp |
| R2 (stronger) | 1.0 | 0.1 | 1e-6 | 95.0% | 90.0% | 90.0% | 91.7% | -3.3pp |
| Ablation (no LSR) | 0.0 | 0.0 | 1e-6 | 95.0% | 93.3% | 93.3% | 91.7% | -3.3pp |

### Token Weight Statistics

| Run | alpha | tw_mean | tw_max |
|-----|-------|---------|--------|
| R1 (α=0.5) | 0.5 | 1.34 | 3.5 |
| R2 (α=1.0) | 1.0 | 1.67 | 6.0 |
| Ablation (α=0.0) | 0.0 | 1.00 | 1.0 |

## Key Finding

**All three runs show identical final degradation: -3.3pp POPE.** This means:

1. **Token-level LSR weighting is NOT the cause of degradation** — α=0.0 (uniform weights) degrades identically
2. **The degradation is from GRPO training itself** on a converged model with out-of-distribution data (A-OKVQA vs POPE)
3. **α=0.5 showed transient benefit** — step 10 recovered to 95.0% before falling at step 15
4. **α=1.0 is too aggressive** — degraded faster (90% at step 5) but converged to same final point

## Diagnosis: Why GRPO Degrades a Converged Model

The Phase 2 GRPO-LSR model (95% POPE) was trained for 5 rounds specifically on POPE-style binary VQA. Training it further with:
- **Different data** (A-OKVQA short-answer, not binary VQA)
- **Different random seed** (seed=43 vs 42)
- **Only 15 steps** (not enough to reconverge)

...causes "catastrophic re-adaptation" — the model starts adapting to A-OKVQA patterns and loses some POPE specialization.

## Implications

1. **Token-level LSR weighting is orthogonal to the convergence issue** — it doesn't help or hurt
2. **To test Token-LSR fairly, need to start from the HF base model** (not a converged checkpoint)
3. **Or use POPE-specific training data** (as Phase 2 did)
4. **The 95% POPE ceiling appears to be the model's capacity limit at 2B scale**

## Next Steps

1. **Phase 5b**: Token-LSR GRPO from HF base model (same as Phase 2 Round 1 starting point)
2. **Phase 5c**: Longer training (50+ steps) with curriculum filtering
3. **Alternative**: Focus on different benchmarks (OCRBench, MME) where ceiling isn't hit
