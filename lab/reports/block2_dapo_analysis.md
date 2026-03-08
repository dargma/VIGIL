# DAPO for Visual Grounding: Detailed Analysis

**Date**: 2026-03-08
**Context**: Block 1 v1-v3 (TRL GRPO) collapsed on binary VQA. Block 2 v2 (custom GRPO, full unfreeze, mixed data) achieved first stable run but POPE oscillated within +/-1.5pp of baseline. Block 2 v3 with KL fix + higher LR still could not push above 85% POPE. Best-of-N + SFT is running as a parallel approach.

---

## A. Why DAPO Might Work Better Than GRPO for VIGIL

### A.1 Asymmetric Clipping Directly Addresses the Reward Variance Problem

GRPO clips the importance ratio symmetrically: both upward and downward updates are bounded by epsilon=0.2. In VIGIL's setting, this is actively harmful for two reasons:

1. **Reward signal is sparse and asymmetric**. Most GRPO groups produce similar outputs (62% skip rate in v2, 44% in v3). When a group finally contains a genuinely better candidate (higher R_correct + IIG), the symmetric clip limits how much the model can learn from that rare event. DAPO's asymmetric clipping (epsilon_low=0.2, epsilon_high=0.28) allows 40% more probability mass to flow toward good candidates than away from bad ones. Over 100 steps, this compounds: the model can climb toward good behavior faster than it slides toward bad behavior.

2. **Binary VQA has a "gravity well" problem**. The model has two attractors: always-yes and always-no. GRPO's symmetric clip means the model drifts toward whichever attractor it starts closer to, because the clip prevents large enough corrections to escape. DAPO's asymmetric clip means that whenever a group contains both correct-yes and correct-no examples, the upward update for the correct answer is larger than the downward update for the wrong answer. This breaks the symmetry that leads to mode collapse.

**Quantitative estimate**: In Block 2 v3, the mean absolute advantage was ~0.12 (very small). With symmetric clip at 0.2, the effective learning rate is lr * min(ratio, 1+0.2) * advantage. With DAPO's 0.28 upper clip, the effective upward learning rate increases by 40% for positive-advantage samples. Over 100 steps, this could be the difference between oscillating at baseline and achieving +2-3pp improvement.

### A.2 No-KL is Safe When Combined With Dynamic Sampling

The standard concern with removing KL penalty is unconstrained policy drift. In Block 1, we saw that even beta=0.1 (10x the TRL default) could not prevent collapse. The failure mode was not "too much drift" but "drift in a degenerate direction." KL penalty anchors the model to the reference distribution but does not care about the direction of drift -- it penalizes moving toward a better policy just as much as moving toward collapse.

DAPO replaces KL with three alternative stability mechanisms:

1. **Dynamic sampling**: Resample groups where all members produce identical reward. This is precisely what VIGIL needs. In Block 2 v2, 62% of groups were skipped (zero variance). DAPO would resample those groups at higher temperature or with different random seeds until reward variance appears. This converts wasted compute into useful gradient signal.

2. **Overlong reward shaping**: Applies a linear penalty as generation approaches max_new_tokens. For VQA, this prevents the model from generating verbose hedging answers ("Based on the image, I believe the answer might be...") that dilute the correctness signal. It pushes toward concise, committal answers that are easier to reward accurately.

3. **Token-level loss normalization**: See A.3 below.

The combination means DAPO is self-stabilizing through the quality of its gradient signal rather than through an explicit anchor to the reference policy. This is more appropriate for VIGIL because the reference policy (base Qwen3-VL-2B) already has the visual attention drift problem we are trying to fix. Anchoring to it via KL actively prevents the model from learning to use vision heads more.

**Critical caveat from Alpha-Triton**: CLAUDE.md Section 4.6 notes "DAPO at 0.6B without KL: -41pp compile (catastrophic). Small models need KL regularization." However, that was for 0.6B. At 2B with mixed non-binary data and dynamic sampling, the risk is substantially lower. The 0.6B collapse was on binary code compilation (compile/no-compile), which has the same structure as binary VQA. The key difference is that VIGIL's mixed data pipeline (TextVQA + A-OKVQA MC + VQAv2 short-answer) provides much higher output entropy than binary VQA. Recommendation: start with beta=0.005 (very small KL, not zero) and remove it after 20 stable steps.

### A.3 Token-Level Loss Handles Variable-Length VQA Answers Better

GRPO computes loss per sequence: `mean(loss_per_token)` for each sequence, then `mean(loss_per_sequence)` for the batch. This means a one-token "yes" answer gets the same weight as a 50-token open-ended answer. In VIGIL's mixed data pipeline:

- Binary yes/no: ~1-2 tokens
- Short answer (VQAv2): ~3-10 tokens
- MC explanation (A-OKVQA): ~10-30 tokens
- TextVQA: ~5-20 tokens

With per-sequence loss, binary answers dominate the gradient despite having the least reward information. This partly explains why Block 2 v2 oscillated: the model was being pulled most strongly by the noisiest reward signal.

DAPO's token-level normalization computes `sum(loss * mask) / sum(mask)` across the entire batch. Each token contributes equally. Longer, more informative answers contribute proportionally more to the gradient. This naturally upweights the samples where the reward signal is most informative (open-ended answers with continuous partial credit) and downweights the samples where it is noisiest (binary yes/no).

For R_vhad specifically, token-level loss is even more important because vision head activation varies across token positions. The first few tokens (typically "The" or "Yes") have the highest vision head activation. Later tokens that elaborate on visual content have decreasing activation (the drift problem). Token-level loss means the model gets credit for maintaining vision head activation at every token position, not just averaged per sequence.

---

## B. Concrete DAPO Experiment Plan for VIGIL

### B.1 Configuration for Qwen3-VL-2B

```yaml
dapo_vigil:
  # Model
  model: Qwen/Qwen3-VL-2B-Instruct
  precision: bf16
  full_unfreeze: true  # no LoRA — Block 2 showed full unfreeze is necessary
  gradient_checkpointing: true

  # DAPO core
  epsilon_low: 0.2
  epsilon_high: 0.28
  beta_kl: 0.005  # near-zero KL initially, remove after 20 steps
  remove_kl_after_step: 20

  # Dynamic sampling
  dynamic_sampling: true
  max_resample_attempts: 3
  resample_temperature_boost: 0.3  # temp += 0.3 on each resample

  # Overlong shaping
  overlong_shaping: true
  overlong_start: 0.7  # start penalty at 70% of max_new_tokens
  overlong_coefficient: 0.5

  # Token-level loss
  token_level_loss: true  # sum(loss*mask)/sum(mask)

  # Generation
  group_size: 16  # larger than GRPO (exploit A100 headroom)
  max_new_tokens: 96  # shorter than GRPO — VQA answers are brief
  temperature: 1.4  # match Block 2 v2 (proven stable)
  top_p: 0.95

  # Optimization
  learning_rate: 5e-6  # 2.5x Block 2 v3's 2e-6 — DAPO allows more aggressive LR without KL
  max_grad_norm: 1.0
  grad_accum: 4
  warmup_steps: 5

  # Schedule
  num_steps: 100
  eval_every: 10
  eval_samples: 200
  save_every: 20

  # Data
  data: mixed_nonbinary  # same as Block 2 (TextVQA + A-OKVQA + VQAv2)
  train_samples: 2000
  seed: 42
```

**Rationale for key differences from Block 2 v3 GRPO**:
- `group_size=16` (vs 8): DAPO's dynamic sampling means zero-variance groups are resampled, not skipped. Larger groups give even more diversity per sample.
- `lr=5e-6` (vs 2e-6): Without KL penalty, the model can tolerate larger updates. Block 2 v3 with KL at 2e-6 oscillated, suggesting the learning rate was too low to make meaningful progress but KL was preventing raising it.
- `max_new_tokens=96` (vs 128): Overlong shaping starts at 67 tokens (0.7 * 96). This pushes the model to answer in under 67 tokens for full reward, which is appropriate for VQA.
- `beta_kl=0.005 -> 0.0 after step 20`: Safety ramp. If POPE stays within 5pp of baseline through step 20, KL is safe to remove entirely.

### B.2 Integration with IIG Reward

IIG integrates naturally with DAPO's composite reward. The DAPO reward becomes:

```
R_dapo = R_correct + lambda_iig * R_iig + overlong_penalty
```

Where `lambda_iig = 0.0615` (calibrated in Block 0).

**Key interaction with dynamic sampling**: IIG provides continuous reward even when R_correct is binary. This means fewer groups will have truly zero variance, reducing the need for resampling. Track the resample rate with and without IIG to quantify this.

**Implementation detail**: IIG requires a forward pass with and without image. This is the same computation as R_vhad. Compute both in a single pair of forward passes (real image + black image), extract IIG from logprobs and R_vhad from hooks simultaneously.

### B.3 Integration with R_vhad

R_vhad is VIGIL's core novelty and integrates with DAPO better than GRPO for structural reasons:

**Composite DAPO reward**:
```
R_total = 0.3 * R_correct + 0.3 * R_vhad + 0.2 * R_iig + 0.2 * R_fluency + overlong_penalty
```

Note the weight redistribution vs the original VIGIL design (which had R_vhad inside R_visual_grounding). For DAPO, flattening the reward hierarchy is better because:
1. Each component contributes independently to reward variance
2. Token-level loss naturally weights R_vhad's per-token contribution
3. Overlong penalty is additive, not multiplicative

**R_vhad computation during DAPO training** (per group of 16 candidates):
1. For each candidate, run forward pass with real image (hooks on vision heads)
2. Run forward pass with black image (hooks on vision heads)
3. R_vhad = mean(|act_real[h] - act_black[h]|) for top-K vision heads, normalized to [0, 1]

This is 2 forward passes per candidate = 32 forward passes per group. On A100 40GB with Qwen3-VL-2B (4.3GB), this fits comfortably. Estimate: ~2 seconds per group for R_vhad computation.

**Interaction with dynamic sampling**: R_vhad is inherently continuous (activation magnitudes vary smoothly). Even when all 16 candidates produce the same text answer, their R_vhad values will differ because different token sequences activate vision heads differently. This makes R_vhad a natural anti-zero-variance mechanism that complements DAPO's dynamic sampling.

### B.4 Expected Results vs GRPO

| Metric | GRPO Block 2 v3 | DAPO (predicted) | Basis for prediction |
|--------|-----------------|-------------------|---------------------|
| POPE Accuracy | 83.5-85.0% (oscillating) | 86-88% | Asymmetric clip + higher LR allows escape from oscillation basin |
| Blind Test Gap | 33-36pp (oscillating) | 37-40pp | R_vhad directly optimizes gap; token-level loss preserves per-token grounding |
| Skip/resample rate | 44% skipped | <15% resampled | Dynamic sampling + R_vhad continuous reward |
| Collapse events | 0 (stable but flat) | 0 | Dynamic sampling + gradual KL removal |
| Training steps to peak | N/A (no peak, oscillation) | 40-60 steps | Higher effective LR from asymmetric clip |

**Conservative prediction**: +2-3pp POPE, +3-5pp Gap over GRPO baseline. This would be the first measurable improvement from RL training in the VIGIL pipeline.

**Aggressive prediction**: +5pp POPE, +7pp Gap. This would match the steering-only result at alpha=5 but with permanent weight changes.

---

## C. Novel DAPO Extensions for Visual Grounding

### C.1 Vision-Aware Dynamic Sampling

**Standard DAPO**: Resample if all group members have identical reward (zero variance).

**VIGIL extension**: Also resample if vision heads are inactive, even when reward variance is nonzero.

```python
def should_resample(rewards, vhad_scores, threshold_var=1e-6, threshold_vhad=0.1):
    # Standard DAPO: zero reward variance
    if np.var(rewards) < threshold_var:
        return True
    # VIGIL extension: low vision head activation across all candidates
    if np.mean(vhad_scores) < threshold_vhad:
        return True
    # VIGIL extension: zero variance in vhad (all candidates equally blind)
    if np.var(vhad_scores) < threshold_var:
        return True
    return False
```

**Why**: A group where all candidates score R_correct=1.0 but R_vhad=0.01 (model answered correctly without using the image) has nonzero reward variance from noise, but the gradient points nowhere useful. Resampling at higher temperature forces the model to generate more diverse responses, some of which may actually attend to the image.

**Risk**: Excessive resampling wastes compute. Cap at 3 attempts, then use the group as-is with a warning flag.

**Novelty**: No prior DAPO work conditions resampling on modality-specific activation patterns. This is a direct contribution.

### C.2 Modality-Specific Clipping

**Standard DAPO**: Single epsilon_low / epsilon_high pair for all tokens.

**VIGIL extension**: Different clipping bounds for tokens at visual vs linguistic positions.

The key insight is that VIGIL wants to encourage upward movement on vision-grounded tokens more aggressively than on linguistic tokens. The model already generates fluent language; it needs to learn to use visual information.

```python
# For each token position t:
if token_is_vision_grounded[t]:  # determined by per-token IIG or vhad
    eps_low, eps_high = 0.15, 0.35  # more asymmetric — strongly favor upward
else:
    eps_low, eps_high = 0.2, 0.25   # nearly symmetric — preserve language quality
```

**How to determine vision-grounded tokens**: Per-token IIG is already computed (see Block 0 calibration: "yes" token IIG=16.3, "person" token IIG=12.5, non-visual tokens near 0). Threshold at IIG > 1.0 to classify vision-grounded tokens.

**Risk**: Adds per-token computation overhead and complexity. May interact badly with token-level loss normalization if vision tokens are rare (gradient dominated by language tokens with tight clip).

**Mitigation**: Only apply in the second half of training (steps 50-100) after the model has stabilized.

**Novelty**: Per-token adaptive clipping based on modality-specific activation is entirely novel. No DAPO/GRPO paper has proposed this.

### C.3 DAPO + Steering-Augmented Generation

**Concept**: During DAPO's generation phase, produce candidates under two regimes:
- 8 candidates: unsteered (natural model behavior)
- 8 candidates: steered at alpha=3 (amplified vision head activation)

This creates systematic within-group reward structure:
- Steered candidates will have higher R_vhad (by construction)
- If steered candidates also have higher R_correct, the advantage signal teaches the model to internalize steering behavior
- If steered candidates have lower R_correct (over-steering), the advantage signal correctly learns to avoid that

**DAPO-specific advantage**: DAPO's asymmetric clipping means the model learns more from the steered-and-correct candidates (upward update at epsilon_high=0.28) than it unlearns from the steered-and-wrong candidates (downward update at epsilon_low=0.2). Over time, this biases the model toward internalizing steering.

**Annealing schedule**: Reduce steering alpha over training to prevent the model from depending on external steering.
```
alpha(step) = max(0, alpha_init * (1 - step / anneal_steps))
# alpha_init=5, anneal_steps=80 → alpha=0 by step 80, last 20 steps fully autonomous
```

**Implementation**: During generation, apply `ActivationSteerer` hooks for half the group, remove hooks for the other half. The gradient computation operates on the unsteered policy (no hooks during backprop), so the steered candidates serve only as reward-rich training signal.

**Risk**: The model might learn "steered responses are rewarded" as a spurious correlation rather than learning to activate vision heads internally. Mitigation: the annealing schedule forces autonomous behavior in the final training phase, and evaluation always uses the unsteered model.

**Novelty**: Using inference-time steering to bootstrap RL training is the core VIGIL thesis. DAPO's asymmetric clipping makes this particularly effective because the learning asymmetry matches the intended direction of improvement.

### C.4 Entropy-Aware Overlong Shaping

**Standard DAPO**: Linear penalty starting at a fixed fraction of max_new_tokens.

**VIGIL extension**: Vary the penalty based on answer type:
- For binary questions (detected by prompt format): penalty starts at 10 tokens (force "yes"/"no")
- For MC questions: penalty starts at 20 tokens (allow "B) reason")
- For open-ended: penalty starts at 70% of max_new_tokens (standard)

This exploits the mixed data pipeline to provide appropriate length pressure per question type. Binary questions currently generate verbose hedging ("Based on my analysis of the image, I would say that yes, there appears to be...") which dilutes both R_correct and R_vhad signal.

---

## D. DAPO vs Best-of-N + SFT for Visual Grounding

### D.1 Structural Comparison

| Dimension | DAPO | Best-of-N + SFT |
|-----------|------|-----------------|
| Learning signal | Per-step advantage (online) | Best candidate selection (offline) |
| Exploration | Dynamic sampling + temperature | Fixed generation with N candidates |
| Stability | Dynamic sampling prevents collapse | SFT is inherently stable |
| Reward utilization | All candidates contribute to gradient | Only best candidate is used |
| Grounding permanence | Direct optimization of R_vhad | Indirect (best candidate happens to be grounded) |
| Compute per step | 1 generation + 1 backprop | N generations + 1 SFT epoch |
| Iterability | Continuous (100 steps) | Discrete rounds (2-3 iterations) |
| Reward horizon | Single-step advantage | Cumulative over SFT epoch |

### D.2 Why DAPO is Better for Visual Grounding (Theoretical)

**Argument 1: Direct R_vhad optimization.** DAPO's advantage function directly incorporates R_vhad, meaning the gradient explicitly points toward higher vision head activation. Best-of-N + SFT only uses R_vhad for ranking candidates; the SFT loss is maximum likelihood on the selected text, with no explicit vision grounding objective. The model might learn to produce the same text without using vision heads (the blind reasoner problem resurfaces).

**Argument 2: Continuous exploration.** DAPO with dynamic sampling continuously generates diverse candidates, some of which may discover novel strategies for visual grounding. Best-of-N generates candidates once per round, then commits to the best ones. If the base model's generation distribution does not contain sufficiently grounded candidates, Best-of-N has no way to discover them. DAPO's temperature and resampling can explore further.

**Argument 3: Per-token credit assignment.** DAPO's token-level loss with R_vhad means the model learns which specific tokens benefit from visual grounding. SFT loss treats all tokens equally (cross-entropy on the entire sequence), so the model cannot distinguish which parts of the answer are vision-dependent.

### D.3 Why Best-of-N + SFT is Better (Practical)

**Argument 1: Proven stability.** SFT cannot collapse. It minimizes cross-entropy on fixed targets. Every training step moves the model closer to the selected best answers. DAPO, despite its improvements over GRPO, still uses policy gradient methods that can diverge. The Alpha-Triton codebase explicitly warns that DAPO at small scale (0.6B) collapsed catastrophically without KL. At 2B, the risk is lower but nonzero.

**Argument 2: Higher effective data utilization.** With N=16 candidates per sample and 2000 samples, Best-of-N generates 32,000 candidates and selects the best 2,000. This is equivalent to 16x more training data than a single DAPO group. The selected dataset is enriched in correct, grounded answers by construction.

**Argument 3: Composability with steering.** Best-of-N can generate half the candidates with steering and half without. The ranking step naturally selects steered-and-correct answers as "best," creating an SFT dataset that implicitly encodes steering behavior. This is simpler and more reliable than the DAPO + steering-augmented generation approach.

**Argument 4: Faster iteration.** One round of Best-of-N + SFT takes ~5-6 hours (generation + training). DAPO takes ~8-10 hours for 100 steps. If Best-of-N works, you can iterate 2-3 rounds in the same GPU time as one DAPO run.

### D.4 Recommendation: Run Both, Combine the Winner

**Phase 1 (parallel)**:
- Run DAPO with R_vhad + IIG (Section B config): 100 steps, ~8 hours
- Run Best-of-N + SFT round 1 (using same composite reward): ~5 hours
- Evaluate both on POPE + Blind Test Gap

**Phase 2 (sequential)**:
- If DAPO wins: Run 2 more rounds of DAPO from the checkpoint
- If Best-of-N wins: Run round 2 of Best-of-N starting from round 1 SFT checkpoint
- If tied: Use Best-of-N checkpoint as the starting point for DAPO (SFT warmstart + DAPO refinement). This combination is theoretically optimal: SFT provides a better initialization, DAPO provides direct R_vhad optimization from that better starting point.

**Phase 3 (synthesis)**:
- Take the best checkpoint from Phase 2
- Apply steering at inference time (alpha=3)
- Evaluate on full Tier 1 benchmarks (POPE, MMBench, MME, MMMU)
- Run blind test to measure final Gap

### D.5 The Real Answer: They Solve Different Problems

Best-of-N + SFT is better at **finding and memorizing good behavior that already exists** in the model's generation distribution. It works when the base model can already produce grounded answers at some temperature, and the task is to make those answers the default.

DAPO is better at **discovering and reinforcing new behavior** that the model does not naturally produce. It works when the model needs to learn a new strategy (e.g., attending to vision heads more strongly) that is not well-represented in its generation distribution.

For VIGIL, both are needed:
- **Short-term (next 2 sessions)**: Best-of-N + SFT to harvest existing grounded behavior. This gets +2-3pp POPE with minimal risk.
- **Medium-term (sessions 3-5)**: DAPO from the Best-of-N checkpoint to push beyond what the model can naturally produce. This is where R_vhad and steering-augmented generation become essential.

The failure mode to avoid: running DAPO from the base model (which already struggled in Block 2 v3). DAPO from a Best-of-N-improved checkpoint starts from a better policy, where more groups will have nonzero reward variance, and the gradient signal is more informative.

---

## Summary Table

| Aspect | GRPO (Block 2) | DAPO (proposed) | Best-of-N+SFT |
|--------|----------------|-----------------|----------------|
| Collapse risk | Low (v2 stable) | Very low (dynamic sampling) | None |
| Improvement over baseline | +0.5pp POPE (noise) | +2-5pp POPE (predicted) | +2-3pp POPE (predicted) |
| R_vhad integration | Not tested | Direct (per-token) | Indirect (ranking) |
| Blind Test Gap | Flat | +3-7pp (predicted) | +2-4pp (predicted) |
| Compute cost (A100) | 4 hours / 100 steps | 8 hours / 100 steps | 5 hours / round |
| Implementation complexity | Done | Medium (new script) | Medium (new script) |
| Risk | Low | Medium (small-model KL concern) | Low |

**Bottom line**: DAPO is the theoretically superior approach for VIGIL because it directly optimizes R_vhad with token-level credit assignment and asymmetric clipping in the direction of visual grounding. Best-of-N + SFT is the safer path that should run first or in parallel. The optimal strategy is Best-of-N first, then DAPO refinement from that checkpoint.
