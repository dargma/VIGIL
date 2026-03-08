# VIGIL Phase 2: Experiment Axes

> **Date**: 2026-03-08
> **Objective**: 3-5 novel, complementary experiment axes that evolve Phase 1 findings into paper-ready results.
> **Constraints**: Each axis implementable + testable in a single GPU session (4-8 hours on A100/L4).

---

## Design Rationale

Phase 1 established three facts that constrain Phase 2 design:

1. **BoN+SFT is the viable training method** (GRPO collapsed 3x on binary VQA, GRPO v2/v3 oscillated). All Phase 2 training axes use BoN+SFT as the backbone.
2. **IIG reward works** (99.4% positive, lambda=0.0615 auto-calibrated). But IIG measures output-space grounding only. Internal activation grounding (R_vhad, drift slope) is untapped.
3. **Two head types exist** (feature heads L24-27 high delta, decision heads L4-5 high Cohen's d). No axis has exploited this distinction yet.

The 15 ideas from Phase 1 reduce to 4 orthogonal axes after crossover/selection:

| Axis | Parents (from idea list) | Core mutation |
|------|-------------------------|---------------|
| A: Steered Distillation | Ideas 10, 14 | Steer during BoN generation to create self-steering model |
| B: Drift-Penalized Selection | Ideas 3, 11, 15 | Token-level vision drift as BoN scoring signal |
| C: Dual-Head Steering | Ideas 1, 2, 6 | Feature vs decision heads steered independently |
| D: IIG-Weighted SFT | Idea 15 | Token-level IIG weights the SFT loss |

---

## Axis A: Steered Distillation (Self-Steering BoN)

### Core Idea

Generate BoN candidates **with steering active** (alpha=5), score them, SFT on the best. The model learns to internalize the behavior that steering produces externally. After training, the model should perform well **without** steering at inference -- it has been "distilled" from the steered version of itself.

### Why Novel

- VISTA is transient (steering off = back to baseline). Steered Distillation makes steering permanent via training data curation.
- DVRP uses external image perturbation. This uses internal activation amplification as the generation-time intervention.
- No prior work combines inference-time head steering with BoN+SFT rejection sampling. The steered model is the "expert" policy; SFT distills that expert into the base weights.

### Expected Effect

| Metric | Phase 1 Best | Expected | Reasoning |
|--------|-------------|----------|-----------|
| POPE | 85.5% | 87-89% | alpha=5 steering gives +10pp at inference; distilling ~50% of that gain is conservative |
| Blind Gap | 37.0pp | 40-43pp | Steered candidates are more visually grounded by construction |

### Implementation

```
1. Load calibration (existing: 20 heads, steering vectors)
2. Install steerer with alpha=5 (existing: ActivationSteerer.steer())
3. Generate N=8 candidates PER SAMPLE with steering active
4. Score with R_correct + IIG (existing vigil_reward)
5. SFT on best candidates (existing phase_sft)
6. Eval WITHOUT steering to measure internalization
7. Eval WITH steering (alpha=1) to measure stacking
```

Key modification to `block2_best_of_n_sft.py`: install steering hooks before `model.generate()` in `phase_generate()`, deactivate before SFT phase. Approximately 30 lines of code.

### Risk

- Steered outputs may have artifacts (repetition, hallucinated visual details) at high alpha. Mitigate: cap alpha at 5, use IIG to filter out garbage candidates.
- Diminishing returns if base model already near ceiling on POPE. Mitigate: also eval on TextVQA/MMBench where headroom is larger.
- Model may learn to mimic steered style without actually grounding. Mitigate: blind test Gap is the ground truth -- if Gap does not increase, internalization failed.

### GPU Time Estimate

- Generation (1000 samples x 8 candidates, steered): ~90 min
- IIG scoring (1000 x 8): ~60 min
- SFT (2 epochs x ~700 samples): ~30 min
- Eval (POPE 200 + blind 100, unsteered + steered): ~20 min
- **Total: ~3.5 hours**

---

## Axis B: Drift-Penalized Selection

### Core Idea

Add a **vision drift penalty** to the BoN scoring function. For each candidate, measure the slope of vision head activation across generated tokens using `InSituVisionReward`. Candidates where vision activation decays steeply (drift) are penalized; candidates with sustained or increasing activation are rewarded. The composite score becomes:

```
Score = R_correct + lambda*IIG + mu*R_drift
```

where `R_drift = sigmoid(slope * scale)` (existing in `InSituVisionReward.compute()`).

### Why Novel

- Directly operationalizes the thesis problem (O(1/L_total) attention decay) as a training signal.
- IIG measures what the image contributes to the output. Drift measures whether the model sustains internal visual processing throughout generation. These are complementary: a model can have high IIG on the first token but zero visual attention by token 10.
- No prior work (VISTA, DVRP, DMAS) uses activation trajectory shape as a selection criterion.

### Expected Effect

| Metric | Phase 1 Best | Expected | Reasoning |
|--------|-------------|----------|-----------|
| POPE | 85.5% | 86-87% | Modest -- POPE answers are short (1-2 tokens), drift is less relevant |
| Blind Gap | 37.0pp | 39-41pp | Drift penalty selects for internally-grounded candidates |
| TextVQA | untested | measurable | Longer answers = more drift = larger effect |

### Implementation

```
1. In phase_generate(), install InSituVisionReward hooks before model.generate()
2. After each candidate generation, call insitu_reward.compute() to get R_drift
3. Composite score = R_correct + lambda*IIG + mu*R_drift
4. Sweep mu in [0.0, 0.3, 0.5, 1.0] to find optimal weight
5. SFT on best candidates, eval as usual
```

Key addition: ~40 lines in `phase_generate()` to install/remove InSituVisionReward hooks and collect drift scores per candidate. Requires sequential generation (not batch) to track per-candidate trajectories, which is slower.

### Risk

- For short answers (POPE yes/no), the trajectory is 1-3 tokens -- too short for meaningful slope estimation. InSituVisionReward already handles this (returns magnitude-only for len<2). The effect will be strongest on TextVQA/A-OKVQA.
- Hook overhead: InSituVisionReward installs one pre-hook + one post-hook per target layer per generation step. With 20 heads across ~10 layers, this is ~10 hooks. Overhead is <5% per forward pass.

### GPU Time Estimate

- Generation with drift tracking (sequential, not batch): ~150 min (1.5x slower)
- IIG scoring: ~60 min
- SFT + eval: ~50 min
- **Total: ~4.5 hours**

---

## Axis C: Dual-Head Ablation and Targeted Steering

### Core Idea

Exploit the discovered two-type head structure. Split the 20 calibrated heads into **Feature heads** (top by activation delta, late layers L20+) and **Decision heads** (top by Cohen's d, early-mid layers L4-10). Run BoN+SFT with three steering configurations:

1. **Feature-only**: Steer only Feature heads (amplify visual encoding)
2. **Decision-only**: Steer only Decision heads (amplify visual decision-making)
3. **Proportional**: Steer all heads with alpha proportional to their respective metric (delta for Feature, Cohen's d for Decision), using `steer_proportional()`

### Why Novel

- First work to identify and separately manipulate Feature vs Decision vision heads in VLMs.
- VISTA steers uniformly across all identified heads. DMAS uses semantic-level (not head-level) discrimination.
- The proportional steering (Idea 6) is a natural consequence of the two-type discovery: each head type should be scaled by its own relevance metric.

### Expected Effect

| Metric | Phase 1 Best | Expected | Reasoning |
|--------|-------------|----------|-----------|
| POPE | 85.5% | 86-88% | Decision heads directly affect yes/no accuracy |
| Blind Gap | 37.0pp | 38-41pp | Feature heads affect visual grounding |

Prediction: Decision-only steering helps POPE more. Feature-only steering helps Blind Gap more. Proportional combines both. This is the key paper finding.

### Implementation

```
1. From calibration results, partition heads:
   - Feature heads: top 10 by activation_delta, layer >= 20
   - Decision heads: top 10 by cohens_d, layer < 20
2. Create 3 ActivationSteerer instances (Feature, Decision, Proportional)
3. Run BoN+SFT 3 times (or 1 combined run with 3 steering configs during generation)
4. Compare: which config improves POPE? Blind Gap? Both?
5. Generate heatmap visualization: head (y-axis) vs layer (x-axis), colored by type
```

Key changes: partition logic in calibration loading (~20 lines), 3 BoN generation runs with different steerer configs. Can reuse SFT pipeline unchanged.

### Risk

- Head partition may not be clean -- some heads may be both high-delta AND high-Cohen's-d. Mitigate: allow overlap, report the overlap fraction.
- Small K per type (10 Feature + 10 Decision) may reduce individual steering effectiveness. Mitigate: compare against full 20-head steering as control.

### GPU Time Estimate

- 3 BoN generation runs x 90 min = 270 min (but can share IIG computation)
- With sharing: ~180 min generation + 60 min IIG + 90 min SFT (3x30) + 60 min eval
- **Total: ~6.5 hours** (can be split across 2 sessions)
- **Minimal version (no SFT, just steered inference comparison)**: ~1 hour

---

## Axis D: IIG-Weighted SFT Loss

### Core Idea

During the SFT phase, weight each token's cross-entropy loss by its per-token IIG value. Tokens that are visually grounded (high IIG, e.g. "person", "yes") get amplified in the loss; structural tokens (low IIG, e.g. punctuation, articles) get down-weighted. The model learns that visually-grounded tokens matter more.

```
loss_i = CE_i * (1 + gamma * max(IIG_i, 0))
```

where `gamma` controls the strength of IIG weighting.

### Why Novel

- Token-level process rewards are cutting-edge in LLM RL (DeepSeek-R1, OpenAI o1). Applying per-token reward weighting to visual grounding SFT is novel.
- All prior VLM training treats all output tokens equally. VIGIL's IIG provides a natural per-token signal without needing a separate reward model.
- This is complementary to BoN selection (Axis A/B select WHICH candidates to train on; Axis D changes HOW the loss is computed within each candidate).

### Expected Effect

| Metric | Phase 1 Best | Expected | Reasoning |
|--------|-------------|----------|-----------|
| POPE | 85.5% | 86-87% | Amplified weight on "yes"/"no" token where IIG is strongest |
| Blind Gap | 37.0pp | 39-42pp | Model learns to weight visually-grounded tokens more heavily |

### Implementation

```
1. Pre-compute per-token IIG for all curated candidates:
   - For each candidate, run compute_iig but return per-token values instead of mean
   - Store as iig_per_token array alongside candidate text
2. Modify phase_sft() to apply token-level loss weighting:
   - After computing labels and CE loss per token, multiply by (1 + gamma * IIG_i)
   - Sweep gamma in [0.0, 0.5, 1.0, 2.0]
3. Eval as usual
```

Key changes:
- Add `compute_iig_per_token()` function to `src/iig.py` (~15 lines, return array instead of mean)
- Modify `phase_sft()` to accept and apply per-token weights (~25 lines)

### Risk

- IIG per-token computation doubles the IIG scoring time (need to store full arrays, not just means).
- For very short answers (1-2 tokens), the weighting is trivial. Effect is strongest on TextVQA/A-OKVQA open-ended answers.
- Gamma too high could overfit to visual tokens and degrade fluency. Mitigate: gamma sweep + R_fluency monitoring.

### GPU Time Estimate

- Per-token IIG computation (1000 x best candidates): ~90 min
- SFT with weighted loss (x4 gamma values): ~120 min
- Eval (x4 configs): ~80 min
- **Total: ~5 hours**

---

## Experiment Matrix

The 4 axes are independently testable and stackable. The matrix below defines the experiment order:

### Phase 2a: Quick Wins (1 session, ~4 hours)

| Exp | Axes | Config | Expected POPE | Expected Gap | Priority |
|-----|------|--------|---------------|-------------|----------|
| P2-01 | C (minimal) | Steered inference only, no SFT, 3 configs | baseline | baseline+1-3pp | 1 (fastest) |
| P2-02 | A | Steered BoN+SFT, alpha=5 | 87-89% | 40-43pp | 2 (highest expected gain) |

### Phase 2b: Deepening (1 session, ~5 hours)

| Exp | Axes | Config | Expected POPE | Expected Gap | Priority |
|-----|------|--------|---------------|-------------|----------|
| P2-03 | A+B | Steered BoN + drift penalty in scoring | 87-89% | 41-44pp | 3 |
| P2-04 | D | IIG-weighted SFT on P2-02 candidates | 87-89% | 40-43pp | 4 |

### Phase 2c: Combinations (1 session, ~6 hours)

| Exp | Axes | Config | Expected POPE | Expected Gap | Priority |
|-----|------|--------|---------------|-------------|----------|
| P2-05 | A+B+D | Steered BoN + drift + IIG-weighted SFT | best so far | best so far | 5 |
| P2-06 | C+A | Dual-head proportional steering during BoN+SFT | ablation | ablation | 6 |

### Phase 2d: Paper Figures (1 session, ~3 hours)

| Exp | Axes | Config | Output | Priority |
|-----|------|--------|--------|----------|
| P2-07 | Analysis | Thinking mode drift curve (4 conditions) | Figure 1 | 7 |
| P2-08 | Analysis | Head type heatmap (feature vs decision) | Figure 2 | 7 |
| P2-09 | Analysis | IIG per-token distribution before/after training | Figure 3 | 8 |

---

## Execution Order

```
Session 1 (4h):
  P2-01 → P2-02
  Quick validation of Axis C (inference-only), then Axis A (main training run)

Session 2 (5h):
  P2-03 → P2-04
  Stack Axis B onto best from Session 1, then test Axis D independently

Session 3 (6h):
  P2-05 → P2-06
  Full combination run, dual-head ablation

Session 4 (3h):
  P2-07 → P2-08 → P2-09
  Paper figure generation (no training, just analysis)
```

### Decision Gates

After Session 1:
- If P2-02 POPE < 86%: increase alpha to 8, add more training samples (2000 instead of 1000)
- If P2-02 Gap < 38pp: the steered candidates are not visually grounded enough. Add R_vhad to scoring.
- If P2-02 POPE > 88%: skip P2-03 drift penalty (marginal gains not worth complexity). Jump to P2-07 figures.

After Session 2:
- If P2-03 does not improve over P2-02: drift penalty is ineffective for short answers. Drop Axis B from combinations.
- If P2-04 shows clear gamma optimum: use that gamma in all subsequent SFT runs.
- If P2-04 shows no effect: IIG-weighted loss is too subtle. Drop Axis D.

---

## Summary Table

| Axis | Name | Core Mechanism | Key Metric | Novel Contribution |
|------|------|---------------|------------|-------------------|
| A | Steered Distillation | Steer during BoN generation, SFT to internalize | POPE +2-4pp, Gap +3-6pp | Permanent internalization of transient steering |
| B | Drift-Penalized Selection | Vision head slope in BoN scoring | Gap +2-4pp | Activation trajectory as selection criterion |
| C | Dual-Head Steering | Feature vs Decision head independent control | Ablation table | Two functionally distinct vision head types |
| D | IIG-Weighted SFT | Per-token loss weighting by visual grounding | Gap +2-5pp | Token-level process reward for VLM grounding |

### Key Properties

- **Independent**: Each axis can be tested alone. Results are interpretable without the others.
- **Stackable**: A+B, A+D, A+B+D are natural combinations. C provides the ablation analysis.
- **Novel**: No axis replicates VISTA/DVRP/DMAS. Each has a clear differentiator.
- **Measurable**: Two primary metrics (POPE accuracy, Blind Gap) with clear baselines.
- **Implementable**: Each requires 20-40 lines of code changes to existing scripts.
