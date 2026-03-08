---
name: vigil-grounding
description: VIGIL-specific VLM visual grounding domain knowledge — models, rewards, evaluation, pitfalls
---

# VIGIL Domain Knowledge

## Model Details
- **Primary**: Qwen3-VL-2B-Instruct (GQA 16Q/8KV, 28 layers, head_dim=128, hidden=2048)
- **Thinking**: Qwen3-VL-2B-Thinking (same arch, extended reasoning)
- **Layer path**: `model.model.language_model.layers`
- **Steering hook**: o_proj pre-hook, per-Q-head activation modification
- **Steer layers 4+ only** (DeepStack on L1-3)

## Calibration (20 Vision Heads)
- Top 3: (5,0) d=9.8, (4,6) d=6.9, (23,2) d=6.6
- Two types: Decision heads (L0-13, high Cohen's d) → affect yes/no accuracy
              Feature heads (L14+, high activation Δ) → affect visual grounding
- Stored: `checkpoints/calibration/qwen3_vl_2b/`

## IIG (Image Information Gain)
- `IIG = mean(log P(y_t | real_img) - log P(y_t | black_img))`
- λ = 0.0615 (auto-calibrated), 99.4% positive rate
- Used in BoN scoring: `score = R_correct + λ * IIG`
- Per-token IIG available for weighted SFT loss

## Evaluation (VLMEvalKit Standard)
- POPE prompt: `"{question} Please answer yes or no."`
- Parsing: `YOrN_Extraction()` — `process_punctuation()` then word-level yes/no
- 9000 samples: adversarial (0-2999), popular (3000-5999), random (6000-8999)
- **Always run blind test** (black image) → track Gap metric
- Precision matters more than accuracy for paper story

## Training Methods (Ranked)
1. **BoN+SFT** (WORKS): Generate N=8 → score → SFT on best. POPE +2.5pp, Gap +5pp
2. **DAPO custom** (TESTING): Asymmetric clipping, soft rewards, KL=0.02
3. **TRL GRPO** (BROKEN): Collapses on binary VQA. Do NOT use.
4. **DPO** (UNTESTED): IIG-ranked preference pairs, immune to binary collapse

## Key Results
| Condition | POPE Acc | Blind Gap |
|-----------|----------|-----------|
| Baseline (official) | 87.4% | 37.4pp |
| BoN+SFT r1 | 85.5%* | 37.0pp* |
| Steered α=3 | 88.0% | 38.0pp |
*BoN+SFT numbers from custom eval; official eval shows 89.5%/37.1pp

## Critical Pitfalls
- Steering α>5 with official prompts → no benefit (already saturated)
- POPE dataset ordered: adversarial first → partial evals only cover hardest split
- Thinking model on POPE: lower raw accuracy due to <think> tokens in parsing
- Two model copies for DAPO: ~8GB VRAM, reduce group_size to 4 if tight
- `enable_thinking=False` in tokenizer when training non-thinking model
- COCO val2014 images shared between 2014/2017 → check A-OKVQA overlap with POPE
