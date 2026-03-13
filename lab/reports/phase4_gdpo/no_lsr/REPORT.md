# Phase 4 GDPO (LSR OFF)

**Date**: 2026-03-13 16:41
**Model**: Qwen3-VL-2B-Thinking (full fine-tune)
**Base**: Qwen/Qwen3-VL-2B-Thinking
**Method**: GDPO (arXiv:2601.05242) -- decoupled per-reward normalization

## Config
| Param | Value |
|-------|-------|
| Steps | 50 |
| Group | 6 |
| T | 1.3 |
| LR | 2e-06 |
| Rewards | correct=0.7, format=0.3, LSR=0.0 |

## GDPO vs GRPO
| Aspect | GRPO | GDPO |
|--------|------|------|
| Normalization | combine rewards -> normalize | normalize each -> combine -> normalize |
| Signal preservation | LSR washed out by R_correct | Each reward preserved independently |

## Results
| Metric | Pre | Post | Delta |
|--------|:---:|:----:|:-----:|
| POPE | 91.7% | 93.3% | +1.7pp |
| Gap | 40.0pp | 42.0pp | +2.0pp |
| Think | 40w | 40w | -- |
| Skip rate | 32/50 (64%) | -- | -- |

![Training](training_dynamics.png)
![Reward Variance](reward_variance.png)
![Eval](eval_progression.png)
