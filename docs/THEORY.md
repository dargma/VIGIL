# VIGIL Theoretical Framework

## Core Problem: Visual Attention Drift
In small VLMs (1-3B parameters), attention to visual tokens decays as O(1/L_total) during generation. As the output sequence grows longer, the model progressively ignores the image, eventually becoming a "blind reasoner" that generates plausible-sounding but visually ungrounded answers.

## Thesis
Head-level activation steering + RL training with visually-grounded reward produces small VLMs that habitually use visual information, preventing blind reasoner degeneration.

## Two-Stage Approach

### Stage A: Inference-time Steering (Surgical, Reversible)
1. **Calibrate**: Compute per-head activation statistics using correct vs incorrect responses
2. **Identify**: Rank heads by Cohen's d → select top-K vision-specialized heads (<7% of all heads)
3. **Steer**: Inject steering vectors at o_proj pre-hook, gated by layer agreement

### Stage B: RL Training (Permanent Weight Changes)
Train with composite reward that emphasizes visual grounding over raw correctness:
```
R_total = 0.3 × R_correct + 0.5 × R_visual_grounding + 0.2 × R_fluency
```

## Key Insight: Two Types of Vision Heads
Discovered during calibration of Qwen3-VL-2B:
- **Decision heads** (early-mid layers L0-13): High Cohen's d, separate correct/incorrect responses
- **Feature heads** (late layers L14+): High activation delta, encode raw visual information

No prior work distinguishes these. This is a novel contribution.

## Image Information Gain (IIG)
```
IIG(y; I | Q) = (1/T) Σ_t [log P(y_t | I, Q, y_<t) - log P(y_t | I_black, Q, y_<t)]
```
Measures per-token how much the image helped generation. Positive IIG = image contributed. Used as reward signal in BoN selection and DAPO training.

## Blind Test (Killer Experiment)
Replace all test images with black images. A well-grounded model should perform much worse without images (high Gap). A blind reasoner performs similarly (low Gap).
- Baseline gap: ~37pp on POPE
- Target: increase gap through training (model becomes MORE image-dependent)

## Competitive Positioning
| Method | Type | Permanence | Mechanism |
|--------|------|-----------|-----------|
| VISTA | Inference steering | Transient | Global attention head amplification |
| DVRP | Training | Permanent | External image perturbation |
| DMAS | Inference | Transient | Semantic-level discrimination |
| **VIGIL** | Both | Both | Internal head-level + IIG reward + agreement gating |

## Key References
- VISTA (ICML 2025): Transient steering, no training
- DVRP (2026): External perturbation-based training
- DMAS (2025): Training-free semantic discrimination
- DeepSeek-Math GRPO: Group relative policy optimization
- DAPO (ByteDance): Decoupled alignment, asymmetric clipping
