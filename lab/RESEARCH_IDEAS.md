# VIGIL Research Ideas — Aggressive Ideation Log

> Observe → Hypothesize → Experiment → Analyze → Iterate

---

## Idea 1: Two Types of Vision Heads (HIGH PRIORITY)

### Observation
Calibration (Cohen's d) and smoke test (activation Δ) identify DIFFERENT top heads:
- **Cohen's d top**: L5H0 (d=9.80), L4H6 (d=6.94), L23H2 (d=6.60) — early-mid layers
- **Activation Δ top**: L26H9 (Δ=66.2), L24H0 (Δ=63.1), L25H0 (Δ=47.2) — late layers

### Hypothesis
There are two functionally distinct types of vision heads:
1. **Feature heads** (late layers, high Δ): encode raw visual information. These "see" the image.
2. **Decision heads** (early-mid layers, high Cohen's d): use visual info for answer selection. These "act on" the image.

### Experiment
- Run steering with ONLY feature heads (late, high Δ)
- Run steering with ONLY decision heads (early-mid, high Cohen's d)
- Run steering with BOTH (current default)
- Run steering with weighted combination: α_decision × decision_vectors + α_feature × feature_vectors
- Compare POPE accuracy across all 4 conditions

### Why This Matters
No prior work distinguishes these two types. VISTA/DMAS steer uniformly. If decision heads matter more for accuracy but feature heads matter more for preventing blind reasoning, VIGIL can do both targeted.

### Paper Contribution
"We identify two functionally distinct types of vision-specialized attention heads..." → Table + heatmap showing the two clusters.

---

## Idea 2: Adaptive Reward Weights During Training (MEDIUM)

### Observation
Fixed w_correct=0.3, w_visual=0.5, w_fluency=0.2 assumes visual grounding is always more important. But early in training, the model needs to learn basic correctness first.

### Hypothesis
A curriculum approach: start with w_correct=0.7, w_visual=0.2, w_fluency=0.1, then shift to w_correct=0.3, w_visual=0.5, w_fluency=0.2 over training.

### Experiment
- Phase 1 (steps 0-50): correctness-heavy (w_correct=0.7)
- Phase 2 (steps 50-100): balanced (w_correct=0.5, w_visual=0.3)
- Phase 3 (steps 100+): grounding-heavy (w_correct=0.3, w_visual=0.5)
- Track blind test Gap at each phase transition
- Compare vs static weights

### Why This Matters
DVRP shows correctness-only GRPO creates blind reasoners. But grounding-only from the start might prevent learning correct answers at all. The curriculum avoids both failure modes.

---

## Idea 3: Vision Drift Curve as Training Signal (HIGH)

### Observation
Smoke test shows activation Δ increases monotonically with layer depth. During generation, if this curve FLATTENS or INVERTS, the model is drifting.

### Hypothesis
Add a "drift penalty" to the reward: measure the slope of vision head activation over generated token positions. Negative slope (decay) → penalty. Flat/positive slope → bonus.

### Experiment
- Implement drift tracking in InSituVisionReward
- R_drift = sigmoid(slope * scaling_factor)
- Add to composite reward: R_total = w1*R_correct + w2*R_visual + w3*R_fluency + w4*R_drift
- Compare with and without R_drift

### Why This Matters
This is the DIRECT solution to the visual attention drift problem described in the thesis. No other method penalizes drift explicitly during training.

---

## Idea 4: Agreement Gating Threshold Ablation (LOW-MEDIUM)

### Observation
Current agreement gating uses threshold=0.7. We don't know if this is optimal.

### Hypothesis
The optimal threshold depends on the model's calibration. Too low = steer everything (noisy). Too high = never steer (useless).

### Experiment
- Sweep threshold: [0.3, 0.5, 0.7, 0.9, 1.0 (always steer)]
- Eval POPE accuracy for each
- Also track: what fraction of tokens get steered at each threshold?

---

## Idea 5: Cross-Modal Steering Transfer (NOVEL, HIGH RISK)

### Observation
Both Qwen3-VL-2B and InternVL3.5-1B have similar architectures (GQA 16Q/8KV, 28 layers). Could steering vectors learned on one transfer to the other?

### Hypothesis
If vision heads are architecturally determined (by position), steering vectors should partially transfer. If they're learned, they won't.

### Experiment
- Calibrate on Qwen3-VL-2B → apply vectors to InternVL3.5-1B (mapped by layer/head index)
- Calibrate on InternVL3.5-1B → apply vectors to Qwen3-VL-2B
- Compare transferred vs native calibration

### Why This Matters
If transfer works, it suggests vision head specialization is an architectural universal, not model-specific. This would be a significant finding about VLM attention structure.

---

## Idea 6: Steering Strength Proportional to Δ (SIMPLE, DO FIRST)

### Observation
Different heads have vastly different activation Δ (0.2 to 66.2). Currently we steer all top-K heads with the same alpha.

### Hypothesis
Heads with higher Δ are more vision-sensitive and should be steered more aggressively. Heads with lower Δ should be steered gently to avoid disruption.

### Experiment
- α_per_head = normalize(Δ_head) × global_alpha
- Compare uniform-alpha vs proportional-alpha on POPE

---

## Idea 7: Thinking Mode Analysis (KILLER EXPERIMENT)

### Observation
Qwen3-VL-2B has a Thinking variant that generates long reasoning chains. This is exactly where visual attention drift is worst (the thesis problem).

### Hypothesis
In thinking mode:
1. Vision head activations will show clear decay over token position
2. Steering will prevent this decay
3. GRPO-trained model will have inherently less decay

### Experiment
- Load Qwen3-VL-2B-Thinking
- Generate 50 thinking-chain responses on POPE
- Plot vision head activation (mean across top-K) vs token position
- 4 conditions: baseline, steered, GRPO-trained, GRPO-trained + steered
- The resulting plot IS Figure 1 of the paper

---

## Idea 8: Compact Steering (EFFICIENCY)

### Observation
Calibration produces 20 steering vectors × 128 dimensions = 2,560 parameters. This is tiny.

### Hypothesis
Steering vectors can be compressed even further using PCA or low-rank approximation without losing effectiveness.

### Experiment
- PCA on the 20 steering vectors
- Keep top-K principal components (K = 1, 3, 5, 10, 20)
- Eval POPE at each compression level
- Report: parameters vs accuracy tradeoff

### Why This Matters
If K=3 components match K=20 full vectors, it proves steering is even more efficient than claimed. Good for the "practical deployment" angle.

---

## Priority Order

1. **Idea 6**: Proportional steering (simple, implement now)
2. **Idea 1**: Two types of vision heads (run after baseline)
3. **Idea 7**: Thinking mode analysis (need Thinking model)
4. **Idea 3**: Vision drift as training signal (implement in reward)
5. **Idea 2**: Adaptive reward weights (for GRPO training)
6. **Idea 4**: Agreement threshold ablation (quick sweep)
7. **Idea 5**: Cross-modal transfer (need InternVL loaded)
8. **Idea 8**: Compact steering (post-results optimization)

---

## Active Experiments Status

| Experiment | Status | Result |
|-----------|--------|--------|
| Smoke test | DONE | 9/9 pass, Δ=6.1 mean |
| Calibration | DONE | 20 heads, top L5H0 d=9.80 |
| POPE baseline | DONE | 79.0% random, 76.5% popular, 78.5% adversarial |
| POPE steered | DONE | +1.5-2.0pp across all splits (alpha=1.0) |
| Blind test | DONE | Gap 25.4pp baseline, 28.4pp steered (+3.0pp) |
| Alpha sweep | DONE | Monotonic to alpha=10 (+9pp), no saturation |
| IIG calibration | DONE | lambda=0.0615, 99.4% positive |
| Block 1 GRPO v1-v3 | DONE | All collapsed (TRL not viable for binary VQA) |
| Block 2 v2 GRPO | DONE | Stable but flat (+0.5pp POPE max) |
| Block 2 v3 GRPO | DONE | Oscillated, no lasting improvement |
| Block 2 BoN+SFT R1 | DONE | POPE +2.5pp, Gap +5.0pp (BREAKTHROUGH) |
| Block 2 BoN+SFT R2 | IN PROGRESS | Multi-round iteration |
| Two-head-types ablation | PLANNED | After BoN rounds |
| Proportional steering | PLANNED | After BoN rounds |

---

## Idea 9: R_vhad + BoN Scoring (HIGH PRIORITY)

### Observation
Current BoN scoring uses R_correct + IIG only. R_vhad (vision head activation differential) provides a complementary signal measuring how much the model's internal activations differ between real and black images.

### Hypothesis
Adding R_vhad to the BoN scoring function will select candidates that are not just correct and IIG-grounded, but also show strong internal visual processing. This should produce a curated dataset that teaches even deeper visual grounding than IIG alone.

### Experiment
- Score = w1*R_correct + w2*IIG + w3*R_vhad
- Sweep w3: [0.0, 0.3, 0.5, 0.7]
- Compare curated dataset quality (mean vision head activation in selected candidates)
- Train SFT on each curated set, compare POPE + Blind Gap

### Why This Matters
IIG measures token-level information gain (output space). R_vhad measures head-level activation (internal space). The combination selects for candidates that are correct, output-grounded, AND internally-grounded. Triple filtering should produce the highest-quality curated data.

---

## Idea 10: Steering-Augmented Candidate Generation (HIGH)

### Observation
BoN generates N=8 candidates from the base model. Steering increases vision head activation and improves accuracy (+2pp at alpha=1, +9pp at alpha=10). Steered candidates should be systematically better.

### Hypothesis
Generate half the candidates unsteered and half steered (at varying alpha). The steered candidates will have higher R_correct and R_vhad scores, biasing the "best" selection toward visually-grounded answers.

### Experiment
- Generate 4 unsteered + 4 steered (alpha=3.0) candidates per sample
- Compare best candidate quality vs 8 unsteered
- Ablation: all-steered vs mixed vs all-unsteered
- Track whether SFT on steered-selected data produces models that need steering at inference

### Why This Matters
If training on steered candidates teaches the model to internally replicate steering behavior, this collapses Stage A (inference-time steering) into Stage B (training). The model becomes self-steering.

---

## Idea 11: Vision Drift Penalty in BoN Scoring (MEDIUM-HIGH)

### Observation
Visual attention drift (O(1/L_total) decay) is the core problem VIGIL addresses. Current BoN scoring ignores token-position dynamics entirely.

### Hypothesis
Add a drift penalty to BoN scoring: measure vision head activation slope over the generated token positions. Candidates with flat or positive slopes (sustained visual attention) get rewarded; candidates with steep negative slopes (rapid drift) get penalized.

### Experiment
- R_drift = sigmoid(slope_of_vision_activation * scale)
- Score = R_correct + lambda*IIG + mu*R_drift
- Compare selected candidates: do drift-penalized selections produce longer, more visually-grounded answers?
- Particularly relevant for open-ended (TextVQA, A-OKVQA) answers where token count > 10

### Why This Matters
Directly operationalizes the thesis problem as a training signal. Unique to VIGIL -- no prior work penalizes drift during candidate selection.

---

## Idea 12: DAPO + Dynamic Sampling with IIG (MEDIUM)

### Observation
GRPO collapsed on binary VQA due to zero variance in groups. DAPO's dynamic sampling resamples zero-variance groups. IIG adds continuous variance. The combination may fix both failure modes.

### Hypothesis
DAPO with IIG reward will avoid collapse AND provide useful gradient, because:
1. Dynamic sampling skips degenerate groups (DAPO fix for zero variance)
2. IIG adds continuous reward to break ties (VIGIL fix for binary reward)
3. Asymmetric clipping allows larger upward updates (helps when signal is weak)
4. No KL penalty avoids premature convergence (but may be risky at 2B -- see Block 1 failure)

### Experiment
- Implement custom DAPO loop (TRL GRPOTrainer failed, need manual)
- Compare: DAPO alone, DAPO+IIG, GRPO+IIG, BoN+SFT
- Use mixed non-binary data (learned from Block 2 that binary VQA is unsuitable for RL)
- Monitor entropy, yes/no balance, skip rate at every step

### Risk
DAPO without KL penalty collapsed catastrophically at 0.6B in Alpha-Triton (-41pp). At 2B it might be stable, but must add entropy floor as safety net.

---

## Idea 13: Cross-Model Steering Transfer (Qwen3-VL to InternVL3) (MEDIUM-HIGH)

### Observation
Both Qwen3-VL-2B and InternVL3.5-1B share GQA 16Q/8KV architecture with 28 layers. The alpha sweep shows monotonic benefit, suggesting steering is not model-specific but architecture-specific.

### Hypothesis
Steering vectors calibrated on Qwen3-VL-2B will partially transfer to InternVL3.5-1B if vision head specialization is determined by architecture (layer depth, head position) rather than learned weights.

### Experiment
- Calibrate steering vectors on Qwen3-VL-2B (already done)
- Apply directly to InternVL3.5-1B (same layer/head indices)
- Also calibrate natively on InternVL3.5-1B for comparison
- Measure: transfer accuracy / native accuracy ratio
- If >80%: architecturally universal. If <50%: model-specific.

### Why This Matters
Universality of vision heads is a novel finding. If confirmed, steering becomes a model-agnostic tool -- calibrate once on a cheap model, deploy on expensive ones.

---

## Idea 14: Thinking Mode Drift Curve (FIGURE 1 CANDIDATE) (HIGH)

### Observation
Qwen3-VL-2B-Thinking generates long reasoning chains (100-500 tokens). The thesis predicts vision head activation decays as O(1/L_total) during generation. This decay IS the visual attention drift problem.

### Hypothesis
Plotting vision head activation vs token position will show:
1. Baseline: clear exponential decay in thinking chain
2. Steered: sustained activation (flat or slower decay)
3. BoN-trained: intermediate (partial internalization of visual attention)
4. BoN-trained + steered: best of both (permanent + augmented)

### Experiment
- Load Qwen3-VL-2B-Thinking
- Generate 50 thinking-chain responses on POPE (with hooks recording per-token activation)
- Plot: x=token position, y=mean vision head activation (top-K heads)
- 4 curves: baseline, steered, BoN R1, BoN R1 + steered
- Annotate: think token boundaries, answer token

### Why This Matters
This plot IS the paper's Figure 1. It visually demonstrates the problem (drift) and the solution (VIGIL). No prior work has published this type of analysis for VLMs.

---

## Idea 15: Token-Level IIG as Process Reward (HIGH)

### Observation
IIG calibration showed per-token IIG varies dramatically: "yes" token IIG=16.3, "person" IIG=12.5, structural tokens IIG~0. Currently we use mean IIG across all tokens.

### Hypothesis
Using token-level IIG as a process reward (weight each token's contribution to the loss by its IIG) would teach the model that visually-grounded tokens matter more than structural ones.

### Experiment
- During SFT: loss_token_i = CE_i * (1 + gamma * IIG_i) where gamma scales IIG influence
- Compare: uniform SFT loss vs IIG-weighted loss
- Evaluate: does the model learn to produce more visually-grounded tokens?
- Measure: mean IIG of generated tokens before/after training

### Why This Matters
Token-level process rewards are the cutting edge of RL for LLMs (DeepSeek-R1, OpenAI o1). Applying this concept to visual grounding is novel. IIG provides a natural per-token reward signal without needing a separate reward model.

---

## Updated Priority Order (2026-03-12)

### Tier 0: Must-Do (Eval + Paper Essentials)
1. **1K POPE eval** on best checkpoint (script ready, needs GPU)
2. **MME eval** on best checkpoint (Perception vs Cognition, script ready)
3. **Drift curve figure** (Idea 14) — hero Figure 1 for paper
4. **Full method comparison table** — all conditions on same eval scale

### Tier 1: High-Impact Improvements (from literature review)
5. **Idea 16: GDPO** — Decoupled reward normalization (NVIDIA, 2026). Drop-in fix: normalize R_correct and R_LSR independently before combining. Currently LSR gets washed out when R_correct dominates. **Effort: LOW, Impact: HIGH.**
6. **Idea 17: VPPO-style token masking** — Only compute GRPO gradients on high-LSR tokens (ICLR 2026, arXiv:2510.09285). Validated on same problem. **Effort: MEDIUM, Impact: HIGH. Must cite + differentiate.**
7. **Idea 18: Curriculum filtering** — Train only on samples with 25-75% pass rate. Easy samples give zero variance, hard samples give uninformative negative rewards. This should break the step-10 plateau. **Effort: LOW, Impact: HIGH.**
8. **Idea 19: DPO with LSR-ranked pairs** — Generate N candidates, rank by R_correct + R_LSR, construct preference pairs. DPO is immune to binary collapse. Alternative path when GRPO plateaus. **Effort: MEDIUM, Impact: MEDIUM-HIGH.**

### Tier 2: Existing Ideas (re-ranked)
9. **Idea 15**: Token-level IIG as process reward (novel, ties to DeepSeek-R1)
10. **Idea 9**: R_vhad + BoN scoring (extends best pipeline)
11. **Idea 1**: Two types of vision heads (paper contribution, already observed)
12. **Idea 3/11**: Vision drift as training signal / drift penalty in scoring
13. **Idea 13**: Cross-model transfer (novel finding about architecture)
14. **Idea 10**: Steering-augmented generation
15. **Idea 12**: DAPO + dynamic sampling + IIG
16. **Idea 6**: Proportional steering (simple ablation)
17. **Idea 2**: Adaptive reward weights
18. **Idea 4**: Agreement threshold ablation
19. **Idea 8**: Compact steering via PCA

---

## Key Papers to Cite (from 2026-03-12 literature review)

| Paper | Venue | Relation to VIGIL |
|-------|-------|-------------------|
| **Qwen-LookAgain** (arXiv:2505.23558) | 2025 | Same problem (O(1/L) decay), architectural fix. VIGIL = mechanistic fix. |
| **VPPO** (arXiv:2510.09285) | ICLR 2026 | Closest method. Per-token visual dependency for GRPO reweighting. |
| **VISTA** (ICML 2025) | ICML 2025 | Inference-only steering. VIGIL adds RL permanence. |
| **DMAS** (arXiv:2602.21704) | 2026 | Closest competitor. Head-level steering + semantic DB. |
| **GDPO** (arXiv:2601.05242) | NVIDIA 2026 | Fixes multi-reward normalization for GRPO. |
| **Vision-SR1** (arXiv:2508.19652) | 2025 | Self-rewarding VLM, behavioral grounding. |
| **TLDR** (ICLR 2025) | ICLR 2025 | Token-level reward model prior art. |
| **Perception-R1** (arXiv:2506.07218) | 2025 | Visual perception reward for CoT. |
| **ASD** (ACL 2025) | ACL 2025 | Activation steering baseline. |
| **SteerVLM** (EMNLP 2025) | EMNLP 2025 | Learned steering module alternative. |
| **VEGAS** (arXiv:2512.12089) | 2025 | Encoder-side steering, complementary. |

### VIGIL's Unique Position
No existing paper combines: (1) head-level mechanistic steering + (2) RL for permanent weight changes + (3) internal activation-based reward (LSR). Qwen-LA = architectural, DMAS/VISTA = inference-only, Vision-SR1 = behavioral, VPPO = closest but no steering component.
