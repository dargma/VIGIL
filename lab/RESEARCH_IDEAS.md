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
| POPE baseline | RUNNING | Waiting... |
| POPE steered | PENDING | After baseline |
| Blind test | PENDING | After steered |
| Two-head-types ablation | PLANNED | After baseline |
| Proportional steering | PLANNED | Implement next |
