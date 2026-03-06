# Skill: Vision Activation Delta — Qwen3-VL-2B

## Finding (2026-03-06 Smoke Test)

Real vs black image activation difference is **strongly non-trivial**.

## Numbers
- mean_Δ = 6.11 (across all 28×16 = 448 heads)
- max_Δ = 66.22 at layer 26, head 9

## Layer Distribution
Activation delta increases with depth:
- Layers 0-3: Δ ~ 0.2-0.4 (early/DeepStack layers)
- Layers 4-8: Δ ~ 0.5-0.8
- Layers 9-15: Δ ~ 1.2-3.0
- Layers 16-23: Δ ~ 5-15
- Layers 24-27: Δ ~ 15-30 (late layers dominate)

## Top-5 Heads by Δ
| Layer | Head | Δ |
|-------|------|---|
| 26 | 9 | 66.22 |
| 24 | 0 | 63.12 |
| 25 | 0 | 47.22 |
| 27 | 5 | 44.05 |
| 26 | 14 | 42.82 |

## Implications
1. R_vhad reward signal is strong and viable — won't be zero/noisy
2. Late layers (24-27) are most vision-sensitive — these should be priority for steering
3. DeepStack exclusion (layers 0-3) is correct — they have minimal vision signal anyway
4. The calibration top heads (layer 5 head 0 by Cohen's d) vs smoke test top heads (layer 26 head 9 by Δ) differ — Cohen's d measures correct/incorrect separation, Δ measures image sensitivity. Both metrics are useful, they capture different aspects.
