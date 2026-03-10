# Phase 2 GRPO-LSR Round 3

**Date**: 2026-03-10 19:11
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
| POPE | 91.7% | 91.7% | +0.0pp |
| Gap | 40.0pp | 40.0pp | +0.0pp |
| Think | 38w | 37w | — |
| Skip Rate | 1/14 (7%) | — | — |

![Training](training_dynamics.png)
![Eval](eval_progression.png)
