# Autoresearch Phase 7: Post-Phase 6c Improvement Loop

## Status: ACTIVE
## Created: 2026-03-15
## Goal: Push POPE beyond 95.0% ceiling and stabilize improvements

---

## Current Best Results (Phase 6c)

| Experiment | Best Step | POPE | Gap | TextVQA | Notes |
|-----------|-----------|------|-----|---------|-------|
| Exp1: Gated Head-LSR | 10 | 95.0% | 44.0pp | **74.7%** | Best TextVQA; degrades at 15 |
| Exp2: Curriculum | 10-15 | 95.0% | 44.0pp | 72.7% | Oscillates; crashes at 30 |
| Exp3: Gated+Curriculum | TBD | TBD | TBD | TBD | Running |
| Phase 2 GRPO-LSR R4 | 10 | 95.0% | 44.0pp | N/A | Historical best |

## Key Observations

1. **POPE 95.0% ceiling**: 4 different methods all peak at 95.0%. This may be a dataset/eval ceiling.
2. **Step 10 is the sweet spot**: All methods peak around step 10, then degrade.
3. **Gated LSR gives best TextVQA**: 74.7% vs 72.7% baseline — head weighting helps open-ended QA.
4. **Curriculum alone is insufficient**: Same peak but oscillates more (unstable).
5. **Overtraining is the enemy**: All methods degrade after 15-20 steps.

## Proposed Experiments (Priority Order)

### Exp A: Larger Eval Set (300-sample POPE)
**Hypothesis**: 60-sample POPE may have noise. 300 samples gives more reliable signal.
**What**: Run 300-sample POPE on baseline + best checkpoints (Exp1 step 10, Phase 2 R4).
**Why**: Distinguish real 95.0% from sampling noise.
**Cost**: ~15 min per eval, no training.

### Exp B: Early Stopping + Higher LR
**Hypothesis**: If step 10 is optimal, a higher learning rate might reach the peak faster.
**What**: lr=5e-6 (2.5x current), 10 steps, Gated Head-LSR. Eval every 2 steps.
**Why**: Faster convergence might prevent the overfitting we see at step 15+.
**Cost**: ~30 min training.

### Exp C: Mixed Training Data (TextVQA + POPE-format)
**Hypothesis**: Training on TextVQA only limits POPE gains. Adding POPE-format data (yes/no questions from GQA-balanced) could directly target POPE.
**What**: 250 TextVQA + 250 GQA-balanced-val (yes/no subset) = 500 mixed samples.
**Why**: Phase 5 showed training format matters — A-OKVQA degraded POPE.
**Cost**: ~45 min (data prep + 10-step training).

### Exp D: Reward Shaping — Continuous Correctness
**Hypothesis**: Binary correct/incorrect reward has zero gradient when all candidates agree. Partial credit (fuzzy match, F1-based) gives continuous signal.
**What**: Replace binary R_correct with F1-based reward for TextVQA, keep binary for yes/no.
**Why**: Steps with correct=1.00 currently get no gradient from correctness — only head-LSR.
**Cost**: ~30 min (reward modification + 10-step training).

### Exp E: KL Penalty for Stability
**Hypothesis**: Models degrade at step 15+ because they drift too far from base model.
**What**: Add KL penalty (beta=0.01) to GRPO loss. Keep Gated Head-LSR.
**Why**: Standard GRPO/DAPO tradeoff — KL prevents catastrophic forgetting.
**Cost**: ~45 min training.

### Exp F: Ensemble Best-of-N Checkpoints
**Hypothesis**: Different checkpoints have complementary strengths.
**What**: Generate with step 5, 10, 15 checkpoints. Majority vote or best-of-3.
**Why**: If step 5 and step 15 have different failure modes, ensemble improves.
**Cost**: ~20 min (3 eval passes).

### Exp G: Temperature Tuning for Group Diversity
**Hypothesis**: T=1.3 may be suboptimal. Higher T = more diverse groups = more gradient signal, but too high = garbage.
**What**: Test T=1.1, 1.3, 1.5 with Gated Head-LSR, 10 steps each.
**Why**: Temperature directly controls exploration-exploitation tradeoff.
**Cost**: ~90 min (3 runs).

### Exp H: Head Subset Ablation (Feature vs Decision heads)
**Hypothesis**: Feature heads (L23-27) and decision heads (L4-5) may contribute differently to training.
**What**: Train with only feature heads (L23H2) vs only decision heads (L5H0, L4H6).
**Why**: If one type is more important, we can focus computational budget.
**Cost**: ~60 min (2 runs).

## Execution Order

1. **Exp A** (fast, informative): Is 95.0% real or noise?
2. **Exp B** (fast, actionable): Higher LR + early stop
3. **Exp D** (novel): Continuous reward shaping
4. **Exp C** (data-driven): Mixed training data
5. **Exp E** (stability): KL penalty
6. **Exp G** (tuning): Temperature sweep
7. **Exp H** (ablation): Head type contribution
8. **Exp F** (ensemble): Test-time compute

## Autoresearch Loop Protocol

1. Run experiment N
2. Record result in `lab/reports/autoresearch/results.tsv`
3. If improvement: checkpoint as new best, update this plan
4. If no improvement: note finding, move to next experiment
5. Git commit + push after each experiment
6. Update MEMORY.md with key findings
7. Repeat until time budget exhausted or breakthrough
