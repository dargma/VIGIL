# Block 2 Ideation: Making GRPO Actually Improve Performance

**Date**: 2026-03-08
**Context**: Block 2 Setting A (R_correct only, custom GRPO) completed without collapse for the first time. POPE 84.5% -> 83.5% (-1pp), Gap 35 -> 33pp (-2pp). Setting B (R_correct + IIG) is running. The winning recipe was full model unfreeze (no LoRA), mixed non-binary data, group_size=8, temp=1.4. The challenge now shifts from preventing collapse to achieving actual improvement.

---

## Diagnosis: Why Setting A Did Not Improve

1. **High skip rate**: Zero-variance groups are skipped entirely, wasting compute. With binary/short answers and group_size=8, many groups produce identical outputs.
2. **Coarse reward**: R_correct is 0/1 for binary, 0/0.5/1 for open-ended. Not enough gradient resolution.
3. **KL penalty dominates**: beta_kl=0.05 on full-model unfreeze is conservative. The model barely moves from the reference policy.
4. **No grounding pressure**: Setting A has zero visual grounding signal. The model can improve accuracy by memorizing answer distributions.

---

## Idea 1: Soft Reward via Token-Level F1 (Reduce Skip Rate)

**What**: Replace binary R_correct with token-level F1 between the generated answer and the ground truth. For yes/no questions, use a calibrated soft score based on log-probability of the correct token (e.g., sigmoid(logp_yes - logp_no) for yes-answer questions).

**Why it might work**: The core problem is that 0/1 rewards produce zero-variance groups when all candidates get the same binary score. Token F1 and logprob-based soft scores create continuous reward signal even when all candidates are "correct" or all are "wrong" -- there are degrees of correctness. This directly attacks the skip-rate bottleneck.

**Risk/difficulty**: Low. Token F1 is standard. Logprob soft scoring needs one extra forward pass but is cheap.
**GPU cost**: Negligible overhead (one extra softmax per candidate).
**Priority**: **P0** -- this is the single highest-leverage fix. Without reward variance, GRPO cannot learn.

---

## Idea 2: R_vhad as Training Reward (Vision Head Activation Differential)

**What**: During GRPO, for each candidate answer, compute R_vhad = normalized activation difference at top-K vision heads between forward passes with the real image vs. a black image. Add to composite reward: R = w1*R_correct + w2*R_vhad.

**Why it might work**: R_vhad is inherently continuous (activation magnitudes vary smoothly across candidates) so it provides non-zero variance in almost every group. It directly encodes "how much did the model use the image to produce this answer" which is the thesis target. The pre-validation showed strong vision-head activation signal (mean delta=6.1, max=66.2). Unlike IIG, R_vhad does not require a separate generation pass -- just two forward passes with hooks.

**Risk/difficulty**: Medium. Requires maintaining hooks during training, two forward passes per candidate (real + black), and care with gradient flow (hooks should be no_grad). Memory cost is ~2x forward passes.
**GPU cost**: ~2x forward cost per candidate. On A100 40GB, feasible with group_size=4 and grad_accum.
**Priority**: **P0** -- this is the core VIGIL contribution. Must be tested.

---

## Idea 3: Best-of-N Rejection Sampling Then SFT (Bypass GRPO)

**What**: Generate N=16 candidates per training sample. Score with composite reward (R_correct + R_vhad + IIG). Select the best candidate per sample. Fine-tune (SFT) on this curated dataset for 1-2 epochs. Repeat for K iterations (iterative best-of-N).

**Why it might work**: GRPO's advantage computation is noisy and prone to collapse. Best-of-N + SFT is provably equivalent to a single step of implicit KL-regularized RL (see ReST, RAFT papers) but is far more stable. It sidesteps the zero-variance group problem entirely because you just pick the best answer regardless of variance. Iterating 2-3 rounds gives compounding improvement.

**Risk/difficulty**: Low. SFT is well-understood. The only question is whether the reward model ranks candidates well enough.
**GPU cost**: Phase 1 (generation): ~4 hours for 2000 samples x 16 candidates on A100. Phase 2 (SFT): ~1 hour for 1-2 epochs. Total ~5-6 hours per iteration.
**Priority**: **P0** -- strongest fallback if GRPO continues to plateau. Known to work in practice (used by DeepSeek, Llama teams).

---

## Idea 4: KTO (Kahneman-Tversky Optimization) with IIG Labels

**What**: Use KTO instead of GRPO/DPO. KTO requires only binary labels (desirable/undesirable) per sample, not paired preferences or group-relative advantage. Label samples as desirable if R_correct > 0.5 AND IIG > median_IIG, undesirable otherwise.

**Why it might work**: KTO's loss function `L = sigma(beta * (logp_policy - logp_ref)) * w` with asymmetric weighting (lambda_D for desirable, lambda_U for undesirable) is fundamentally immune to the zero-variance problem. It does not compute group advantage at all. It also handles the binary VQA case gracefully because each sample is independently labeled. The IIG criterion ensures that "desirable" means both correct AND visually grounded.

**Risk/difficulty**: Medium. KTO is newer, less battle-tested than DPO. Need to implement or find a TRL-compatible KTO trainer. Labeling threshold matters.
**GPU cost**: Similar to DPO (~3-4 hours for 50 steps on A100).
**Priority**: **P1** -- strong alternative if both GRPO and best-of-N underperform. Theoretically cleaner for our use case.

---

## Idea 5: Curriculum from Easy to Hard (High Vision Salience First)

**What**: Sort training samples by vision salience (measured by pre-computed IIG or activation delta on the base model). Train first on samples where the image is most important (high IIG, large activation delta), then gradually include harder samples where visual grounding is subtler.

**Why it might work**: High-salience samples produce the strongest IIG/R_vhad reward signal, giving GRPO clear gradients to learn from. Starting there builds the "habit" of image-dependent reasoning before facing ambiguous cases. This is analogous to curriculum learning in RL where easy environments are solved first. It also reduces skip rate because high-salience samples are more likely to produce diverse candidates (the image provides more distinguishing information).

**Risk/difficulty**: Low. Just requires pre-computing salience scores and sorting. No algorithm changes.
**GPU cost**: ~1 hour for salience scoring (500 samples, 2 forward passes each). Training cost unchanged.
**Priority**: **P1** -- easy to implement, good expected payoff, composable with any training method.

---

## Idea 6: Steering-Augmented Generation (Test-Time + Training Synergy)

**What**: During GRPO generation, apply steering (alpha=3-5) to half the candidates in each group, and generate the other half unsteered. This creates systematic within-group diversity: steered candidates tend to be more visually grounded, unsteered tend to follow text priors.

**Why it might work**: The alpha sweep showed +5pp at alpha=5. Steered candidates will systematically score higher on R_vhad/IIG, creating guaranteed reward variance within each group. The GRPO advantage will consistently point "toward" the steered behavior, teaching the model to internalize steering permanently. This is the VIGIL thesis in action: inference-time steering bootstraps RL training.

**Risk/difficulty**: Medium. Need to interleave steered/unsteered generation cleanly. Risk: model might learn to depend on steering hooks rather than internalizing the behavior. Mitigation: reduce steering alpha over training (anneal from 5 to 0).

**GPU cost**: Negligible extra (steering hooks add <5% overhead). Same generation cost.
**Priority**: **P1** -- novel, directly tests the core thesis, and solves the reward variance problem simultaneously.

---

## Idea 7: SimPO (Simple Preference Optimization) -- Reference-Free

**What**: Use SimPO instead of DPO. SimPO removes the reference model entirely: the reward is just the average log-probability of the response under the policy, with a length-normalized margin. Loss: `L = -log(sigma(beta/|y_w| * sum(logp(y_w)) - beta/|y_r| * sum(logp(y_r)) - gamma))`.

**Why it might work**: SimPO eliminates the reference model, saving ~4GB VRAM and removing the KL anchor that may be preventing the model from moving. On A100 40GB, this frees capacity for group_size=8 with full model or even batch_size=2. SimPO's length normalization also handles the variable-length answer problem (binary vs. open-ended in mixed data) more gracefully than DPO. Published results show SimPO matches or beats DPO on multiple benchmarks with less compute.

**Risk/difficulty**: Low-Medium. SimPO is well-documented. Need preference pairs (generate from best-of-N or IIG ranking). Main risk: without reference model, the policy could drift further, but SimPO's margin term gamma controls this.
**GPU cost**: Lower than DPO (no reference model forward pass). ~2-3 hours for 50 steps.
**Priority**: **P2** -- good if DPO is tried and VRAM is tight. Lower priority because DPO/KTO are more established.

---

## Idea 8: Exploit A100 Capacity -- Group Size 16 + Batch Accumulation

**What**: The current setup uses group_size=8 on what was an L4 (23GB). On A100 40GB with fp16 (model ~4.3GB), there is ~35GB headroom. Use group_size=16 with gradient checkpointing and sequential generation. Alternatively, accumulate 4 samples x group_size=4 per gradient step for better batch diversity.

**Why it might work**: Larger groups have lower skip rates (more likely to contain at least one different answer). With 16 candidates, the probability of all-same-reward drops exponentially. Multi-sample batches also average out per-sample noise in the advantage estimate. The Alpha-Triton GRPO (same codebase family) uses group_size=8 with 2048-token generations on A100 -- our 128-token VQA generations are far cheaper.

**Risk/difficulty**: Low. Just configuration changes + gradient checkpointing.
**GPU cost**: Same wall-clock (sequential generation), better gradient quality.
**Priority**: **P1** -- free improvement from hardware upgrade. Should be the default for all A100 runs.

---

## Priority Summary

| Priority | Idea | Category | Key Benefit |
|----------|------|----------|-------------|
| **P0** | 1. Soft reward / token F1 | Reward design | Eliminates zero-variance groups |
| **P0** | 2. R_vhad as training reward | Reward design | Core VIGIL contribution, continuous signal |
| **P0** | 3. Best-of-N + iterative SFT | Alternative training | Stable, proven, bypasses GRPO fragility |
| **P1** | 4. KTO with IIG labels | Alternative training | Immune to group variance problem |
| **P1** | 5. Curriculum by vision salience | Curriculum | Easy, composable, better early gradients |
| **P1** | 6. Steering-augmented generation | Training + test-time | Guaranteed group diversity, tests thesis |
| **P2** | 7. SimPO (reference-free) | Alternative training | Saves VRAM, removes KL anchor |
| **P1** | 8. Group size 16 on A100 | Compute efficiency | Free improvement from hardware |

---

## Recommended Execution Order

1. **Immediate** (next GPU session): Implement Ideas 1+2+8 together -- soft reward + R_vhad + group_size=16. Run Setting C: R_soft + R_vhad, 50 steps. This is the strongest possible GRPO configuration.
2. **If Setting C plateaus**: Run Idea 3 (best-of-N + SFT) using the composite reward from step 1 for ranking. This is the stable fallback.
3. **If best-of-N works**: Add Idea 6 (steering-augmented generation) to the best-of-N pipeline for round 2. Steered candidates as chosen, unsteered as rejected.
4. **Parallel**: Pre-compute salience scores (Idea 5) during generation phase of step 2. Use for curriculum in subsequent rounds.
5. **If all GRPO/best-of-N approaches plateau below +2pp**: Switch to Idea 4 (KTO) as the fundamentally different approach.
