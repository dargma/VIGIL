# Phase 2 GRPO-LSR Round 1

**Date**: 2026-03-10 17:31
**Model**: Qwen3-VL-2B-Thinking (Unsloth full fine-tune, NO LoRA)

## Config
| Param | Value |
|-------|-------|
| Steps | 15 |
| Group | 6 |
| T | 1.3 |
| LR | 2e-06 |
| Reward | R_correct*0.5 + R_correct*R_LSR*0.5 (gated) |

## Results
| Metric | Pre | Post | Δ |
|--------|:---:|:----:|:-:|
| POPE | 91.7% | 93.3% | +1.7pp |
| Gap | 40.0pp | 42.0pp | +2.0pp |
| Think | 40w | 40w | — |
| Skip Rate | 1/15 (7%) | — | — |

![Training](training_dynamics.png)
![Eval](eval_progression.png)
