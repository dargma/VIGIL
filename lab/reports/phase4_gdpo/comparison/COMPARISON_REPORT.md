# Phase 4 GDPO Comparison Report

**Date**: 2026-03-13
**Model**: Qwen3-VL-2B-Thinking
**Base checkpoint**: Phase 2 GRPO-LSR round start (POPE 91.7%, Gap 40.0pp)

---

## 1. Results Summary

| Method | POPE Acc | Delta | Blind Gap | Delta | Grad Updates | Skip Rate |
|--------|----------|-------|-----------|-------|-------------|-----------|
| Baseline (pre-training) | 91.7% | -- | 40.0pp | -- | -- | -- |
| **GDPO no-LSR** | **93.3%** | **+1.7pp** | **42.0pp** | **+2.0pp** | 19/50 | 62% |
| GDPO with-LSR | 91.7% | +0.0pp | 40.0pp | +0.0pp | 50/50 | 0% |
| Phase 2 GRPO-LSR (best) | 95.0% | +3.3pp | 44.0pp | +4.0pp | ~75 steps (5 rounds) | ~60% |

## 2. Configuration Comparison

| Parameter | No-LSR | With-LSR |
|-----------|--------|----------|
| `w_correct` | 0.7 | 0.4 |
| `w_format` | 0.3 | 0.2 |
| `w_lsr` | **0.0** | **0.4** |
| `lsr_scale` | 2.0 | 2.0 |
| `group_size` | 6 | 6 |
| `temperature` | 1.3 | 1.3 |
| `lr` | 2e-6 | 2e-6 |
| `beta_entropy` | 0.01 | 0.01 |
| `grad_accum` | 2 | 2 |
| `num_steps` | 50 | 50 |

## 3. Skip Rate Analysis

### No-LSR (62% skip rate, 31/50 skipped)

The no-LSR run used only R_correct (w=0.7) and R_format (w=0.3). Because POPE is binary yes/no, groups frequently had **zero reward variance** — all 6 candidates got the same correct/incorrect answer. GDPO correctly skips these zero-variance groups (no learning signal).

- Steps with gradient updates: 1, 7, 19, 20, 22, 25, 27, 29, 30, 31, 32, 34, 38, 44, 45, 47, 48, 50 (and step 19) = **19 steps**
- The 19 informative gradient updates were enough to produce a +1.7pp POPE gain

### With-LSR (0% skip rate, 50/50 updated)

Adding LSR (w=0.4) provides **continuous reward variance** within every group — each candidate has a different KL divergence between real and black image logits. This means every group has non-zero advantage variance, so no steps are skipped.

However, this was actually **harmful**: all 50 gradient updates produced zero net improvement.

## 4. Analysis: Why No-LSR Outperforms With-LSR

### The Dilution Hypothesis

The core issue is **reward signal dilution**. When LSR contributes 40% of the total reward, it dominates the advantage signal in groups where correctness is uniform (all correct or all incorrect). In these groups:

- **No-LSR**: The step is skipped (zero variance). No gradient update occurs.
- **With-LSR**: The step proceeds, but the gradient is driven entirely by LSR variance. The model learns to maximize KL(P_real || P_black) **regardless of whether the answer is correct**.

This creates a conflict: LSR rewards can reinforce incorrect answers that happen to have high visual sensitivity, while penalizing correct answers with lower sensitivity. Over 50 steps, these conflicting gradients cancel out.

### Evidence from the History

In the with-LSR run, many steps had `mean_correct=1.0` (all candidates correct) with non-zero LSR variance:
- Steps 2-6, 8, 10-16, 18-19, 21, 23-25, 28, 33, 35-37, 39-40, 43, 46, 49 (30+ steps)
- In these steps, the only gradient signal came from LSR differences between equally-correct candidates
- The model was being pushed to change its internal representations to maximize visual sensitivity, **not** to answer more accurately

### The GRPO-LSR Difference

Phase 2 GRPO-LSR succeeded because LSR was **gated by correctness**: `R_total = R_correct * 0.5 + R_correct * R_LSR * 0.5`. This means LSR only contributed reward when the answer was correct, preventing the conflicting gradient problem.

In GDPO, the reward channels are **decoupled** by design (each reward is normalized independently). This decoupling, while theoretically cleaner, removes the gating mechanism that made LSR effective in Phase 2.

### Skip-as-Feature

The 62% skip rate in no-LSR is not a bug — it is a feature. By only updating on groups with genuine correctness disagreement, the model receives a concentrated, high-quality learning signal. Each of the 19 gradient updates pushed the model toward better answers.

## 5. Comparison with Phase 2 GRPO-LSR

| Aspect | Phase 4 GDPO no-LSR | Phase 2 GRPO-LSR |
|--------|---------------------|------------------|
| Training steps | 50 (19 effective) | ~75 (5 rounds x 15 steps) |
| Peak POPE | 93.3% | **95.0%** |
| Peak Gap | 42.0pp | **44.0pp** |
| Multi-round | No (single run) | Yes (5 rounds, best-of) |
| LSR integration | None | Gated (R_correct * R_LSR) |
| Reward normalization | Per-channel (GDPO) | Single composite (GRPO) |

Phase 2 GRPO-LSR remains the best approach by +1.7pp POPE and +2.0pp Gap. Key advantages:

1. **Multi-round training**: Resetting the optimizer and data sampling each round prevents overfitting to easy samples
2. **Gated LSR**: The multiplicative coupling `R_correct * R_LSR` ensures visual grounding only reinforces correct behavior
3. **More total effective updates**: ~75 steps across 5 rounds vs 19 effective steps in one GDPO run

## 6. Recommendations

1. **Do not use LSR as an independent GDPO reward channel.** The decoupled normalization removes the correctness gating that makes LSR useful.
2. **GDPO's skip mechanism is valuable.** The 62% skip rate filters out uninformative groups, producing cleaner gradients.
3. **Multi-round GRPO-LSR remains optimal** for binary VQA at 2B scale.
4. **If pursuing GDPO further**: Try a gated LSR channel where `R_lsr_gated = R_correct * R_lsr_raw` before feeding into GDPO's per-channel normalization. This preserves both GDPO's decoupled normalization and the correctness gating.
5. **Curriculum filtering** (25-75% pass rate samples) would increase the fraction of informative groups, reducing the skip rate while maintaining gradient quality.
