# Skill: Calibration Results — Qwen3-VL-2B

## Run Date: 2026-03-06

## Setup
- 1000 samples (500 GQA-balanced-val + 500 TextVQA-val)
- Correctness-based split: 43 correct, 957 incorrect
- Top-K = 20 heads selected by Cohen's d

## Top Vision Heads (by Cohen's d)
| Rank | Layer | Head | Cohen's d |
|------|-------|------|-----------|
| 1 | 5 | 0 | 9.795 |
| 2 | 4 | 6 | 6.943 |
| 3 | 23 | 2 | 6.602 |

## Observations
- Early-mid layers (4-5) have the strongest vision specialization
- Layer 23 (late) also has a strong vision head — interesting for long-range grounding
- DeepStack layers 0-3 recommended for exclusion — but layer 4 has top heads, so steer_layers_start=4 is correct (just barely)
- Low correct count (43/1000 = 4.3%) — model struggles with short-answer VQA in single-token prediction. May need to revisit correctness criterion.

## Checkpoint
`checkpoints/calibration/qwen3_vl_2b/steering_vectors.pt`
`checkpoints/calibration/qwen3_vl_2b/calibration_meta.json`
