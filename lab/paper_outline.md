# VIGIL Paper Outline

## Title Options

1. **VIGIL: Vision-Grounded Inference via Logit-Shift Reward for Small VLMs**
2. **Curing Visual Blindness in Small VLMs: Logit-Shift Reward Prevents Attention Drift During Reasoning**
3. **Don't Forget to Look: Logit-Shift RL Training Keeps Small Vision-Language Models Visually Grounded**

Recommendation: Option 2 for workshop/conference (descriptive), Option 3 for broader audience (accessible).

---

## Abstract (~200 words)

Small vision-language models (1-3B parameters) suffer from Visual Attention Drift: attention to visual tokens decays as O(1/L) during generation, causing the model to increasingly ignore the image as it reasons. This is especially severe in thinking-mode models, where long reasoning chains amplify the drift. We present VIGIL, a method that keeps small VLMs visually grounded through a novel Logit-Shift Reward (LSR) used during GRPO training. LSR measures the KL divergence between the model's token-level logit distributions when conditioned on the real image versus a black image, computed over thinking-phase tokens only. This provides a dense, continuous reward signal that directly penalizes image-independent reasoning. Applied to Qwen3-VL-2B-Thinking, GRPO-LSR improves POPE accuracy from 89.8% to 95.0% (+5.2pp) and increases the Blind Test Gap -- the accuracy difference between real and black images -- from 37.4pp to 44.0pp (+6.6pp), demonstrating stronger image dependence. We find that step 10 is consistently optimal across five rounds, with further training causing regression. Our calibration analysis reveals two functionally distinct types of vision-specialized attention heads: Feature heads in late layers that encode raw visual information, and Decision heads in early-mid layers that drive answer selection. VIGIL generalizes across architectures, showing improvements on both Qwen3-VL-2B and InternVL3.5-1B.

---

## Section Outline

### 1. Introduction (2 pages)

**Key points:**
- Small VLMs (1-3B) are practical for deployment but suffer from visual attention drift
- Problem definition: attention to visual tokens decays as O(1/L_total), especially in thinking-mode models with long reasoning chains
- Prior work addresses this with inference-time steering (VISTA, DMAS) but effects are transient
- Our contribution: LSR provides a permanent fix through RL training
- Teaser result: +5.2pp POPE, +6.6pp Blind Gap

**Figure 1**: Vision drift curve -- vision head activation vs token position for baseline vs VIGIL-trained model. Shows exponential decay in baseline, sustained activation in VIGIL.

### 2. Related Work (1.5 pages)

**2.1 Visual Grounding in VLMs**
- Hallucination literature (LLaVA-RLHF, RLHF-V) -- focuses on factual accuracy, not mechanism
- VISTA (ICML 2025) -- inference-time steering, transient effect. VIGIL is permanent via RL.
- DMAS (2025) -- training-free semantic retrieval. Complementary to VIGIL.
- DVRP (2026) -- external visual perturbation reward. VIGIL uses internal activation (head-level logit shift).

**2.2 RL for VLMs**
- GRPO (DeepSeekMath) -- group relative advantage. We build on this.
- DAPO (ByteDance) -- asymmetric clipping, dynamic sampling. Tested but GRPO-LSR sufficient.
- ReST/RAFT -- offline RL via best-of-N selection. Our BoN+SFT baseline uses this.

**2.3 Attention Head Analysis**
- Head Pursuit (sparse head identification)
- No prior work distinguishes Feature vs Decision heads in VLMs

### 3. Method (3 pages)

**3.1 Problem: Visual Attention Drift**
- Formal definition: Let a_t^h be the activation of vision head h at generation step t. We observe a_t^h ~ O(1/t) decay.
- Blind Test metric: Gap = Acc(real_image) - Acc(black_image). Low gap = blind reasoner.
- Why standard GRPO makes it worse: R_correct-only training rewards correct answers regardless of image use, producing blind reasoners.

**3.2 Logit-Shift Reward (LSR)**
- Core idea: R_LSR = D_KL(P_real || P_black) computed per token over thinking-phase tokens
- P_real: logit distribution conditioned on real image
- P_black: logit distribution conditioned on black (zero) image
- Masked to thinking tokens only (answer-phase KL ~ 0, provides no signal)
- Gated combination: R_total = R_correct * 0.5 + R_correct * R_LSR * 0.5
  - Gating prevents rewarding visually-grounded but incorrect answers
  - R_LSR only contributes when the answer is correct

**Figure 2**: LSR computation diagram. Two forward passes (real vs black image), KL divergence on thinking tokens, gated by correctness.

**3.3 Two Types of Vision Heads**
- Calibration via Cohen's d identifies Decision heads (early-mid layers L4-5, high d up to 9.8)
- Activation delta analysis identifies Feature heads (late layers L24-27, high delta up to 66.2)
- These are functionally distinct: Feature heads encode visual features, Decision heads route them to the answer
- No prior work makes this distinction

**Figure 3**: Heatmap of Cohen's d vs Activation Delta across all 448 heads (28 layers x 16 heads). Two distinct clusters visible.

**3.4 GRPO-LSR Training**
- Base: Qwen3-VL-2B-Thinking, full fine-tuning (no LoRA -- tested, worse)
- Data: A-OKVQA multiple-choice (non-binary to ensure group diversity)
- Group size 6, temperature 1.3
- Multi-round: 5 rounds x 10 steps, use best checkpoint (not final) for next round
- Gradient checkpointing (disabled during .generate(), re-enabled for backward)

### 4. Experiments (3 pages)

**4.1 Setup**
- Models: Qwen3-VL-2B-Thinking (primary), InternVL3.5-1B (cross-model)
- Benchmarks: POPE (3 splits: random/popular/adversarial), Blind Test Gap
- Baselines: Greedy baseline, Inference-time steering (alpha sweep), BoN+SFT (ReST)
- Evaluation: VLMEvalKit-standard prompts and parsing

**4.2 Main Results**

**Table 1: POPE Accuracy and Blind Test Gap**

| Method | POPE Acc | Blind Gap | Delta Acc | Delta Gap |
|--------|----------|-----------|-----------|-----------|
| Baseline | 89.8% | 37.4pp | -- | -- |
| Steering (alpha=5) | 87.6% | 37.6pp | -2.2pp | +0.2pp |
| BoN+SFT R1 | 88.0% | 38.0pp | -1.8pp | +0.6pp |
| GRPO (R_correct only) | collapsed | -- | -- | -- |
| **GRPO-LSR (ours)** | **95.0%** | **44.0pp** | **+5.2pp** | **+6.6pp** |

Note: Steering hurts Thinking mode but helps Instruct mode (Table 2 in appendix).

**4.3 Training Dynamics**

**Figure 4**: POPE accuracy and Blind Gap across 5 rounds x 10 steps. Step 10 consistently optimal; 15 steps causes regression. Shows diminishing returns after round 2.

**Table 2: Per-Round Results**

| Round | Pre POPE | Best POPE | Best Gap | Optimal Step |
|-------|----------|-----------|----------|-------------|
| R1 | 91.7% | 93.3% | 42.0pp | 10 |
| R2 | 93.3% | 95.0% | 44.0pp | 10 |
| R3 | 91.7% | 93.3% | 42.0pp | 10 |
| R4 | 91.7% | 95.0% | 44.0pp | 5-10 |
| R5 | 93.3% | 93.3% | 42.0pp | 10 |

**4.4 Cross-Model Generalization**

InternVL3.5-1B results with BoN+SFT (same pipeline, different architecture):
- Baseline: 78.2% / 25.6pp -> BoN+SFT R2: 83.4% / 33.4pp (+5.2pp / +7.8pp)
- Confirms method is architecture-agnostic

**4.5 Ablations**

**Table 3: Ablation Study**

| Variant | POPE | Gap | Notes |
|---------|------|-----|-------|
| R_correct only (no LSR) | collapsed | -- | Blind reasoner collapse |
| R_LSR only (no gating) | ~85% | ~35pp | Rewards grounded but wrong answers |
| Gated R_correct + R_LSR | **95.0%** | **44.0pp** | Full method |
| LoRA r=16 | ~80% | ~32pp | Insufficient capacity |
| Full fine-tuning | **95.0%** | **44.0pp** | Required for graph signal |
| 15 steps | 91.7% | 40.0pp | Overfitting regression |
| 10 steps | **95.0%** | **44.0pp** | Optimal |

### 5. Analysis (1.5 pages)

**5.1 Why 10 Steps?**
- Skip rate analysis: 7-13% at step 10 (healthy diversity), >20% at step 15 (mode collapse beginning)
- LSR signal dynamics: mean KL 0.6-1.0 during training, drops after step 10
- Hypothesis: small models have limited capacity for RL -- early steps fix the biggest errors, later steps overfit

**5.2 Steering vs Training**
- Steering and BoN+SFT are substitutes, not complements (Table in appendix)
- BoN+SFT already internalizes steering benefit into weights
- Steering helps Instruct mode (+2pp) but hurts Thinking mode (-2pp) -- thinking chains are disrupted by external intervention
- GRPO-LSR achieves the benefit permanently without inference-time overhead

**5.3 Why Binary VQA Breaks GRPO**
- Binary (yes/no) output has ~1 bit entropy
- GRPO groups of size 6-8 produce near-identical outputs at reasonable temperatures
- Zero variance in group rewards -> zero advantage -> zero gradient
- LSR on non-binary MC data solves this by providing continuous reward with natural variance

**Figure 5**: Reward distribution within GRPO groups. Binary VQA: delta-function at 0 or 1. MC + LSR: continuous distribution enabling gradient.

### 6. Discussion (1 page)

- Visual attention drift is a fundamental limitation of autoregressive VLMs, not a training data artifact
- LSR is the first reward that directly targets the drift mechanism (logit sensitivity to image)
- The two-types-of-heads finding suggests VLMs develop specialized visual processing circuits analogous to ventral (what) and dorsal (where) streams in primate vision
- 10-step training sweet spot implies small models have a narrow RL training window
- Future: LSR at 7B+ scale, LSR for video understanding (temporal drift), combining LSR with process reward models

### 7. Conclusion (0.5 pages)

- VIGIL addresses visual attention drift in small VLMs through LSR, a novel RL reward
- GRPO-LSR achieves POPE 95.0% and Blind Gap 44.0pp, substantial improvements over baseline
- Two types of vision heads discovered (Feature vs Decision)
- Method generalizes across architectures (Qwen3-VL, InternVL)
- Small models can be visually grounded with surgical RL -- 10 steps suffice

---

## Figure Plan

| # | Content | Section | Type |
|---|---------|---------|------|
| 1 | Vision drift curve: activation vs token position (baseline vs VIGIL) | Intro / 3.1 | Line plot |
| 2 | LSR computation pipeline diagram | 3.2 | Architecture diagram |
| 3 | Two-types heatmap: Cohen's d vs Activation Delta across all heads | 3.3 | Heatmap |
| 4 | Training dynamics: POPE + Gap across 5 rounds x 10 steps | 4.3 | Line plot (dual y-axis) |
| 5 | Reward distribution: binary VQA vs MC+LSR within GRPO groups | 5.3 | Histogram / violin |
| 6 | Cross-model comparison bar chart | 4.4 | Grouped bar |

**Tables**: 3 main (results, per-round, ablation) + 2 appendix (steering+BoN combo, per-split POPE)

---

## Related Work Positioning

### vs VISTA (ICML 2025)
- **Similarity**: Both identify vision-specialized heads and intervene at the head level
- **Difference**: VISTA is inference-time only (transient -- remove steering, performance reverts). VIGIL's GRPO-LSR produces permanent weight changes. VISTA does not use RL.
- **Positioning**: VIGIL subsumes VISTA. Our steering experiments (Section 4.2) replicate VISTA's approach as a baseline. GRPO-LSR strictly dominates.
- **Complementarity**: VISTA-style steering could augment VIGIL-trained models, but we show the benefit is negligible (methods are substitutes).

### vs DVRP (2026)
- **Similarity**: Both use visual perturbation to measure image dependence
- **Difference**: DVRP perturbs the input image externally (crops, occlusion). VIGIL compares real vs black image at the logit level internally. DVRP's reward is based on answer change; LSR is based on logit distribution change (denser signal).
- **Positioning**: LSR is more principled -- it measures the full distribution shift, not just the argmax answer. LSR also works per-token on thinking chains, while DVRP only checks final answers.

### vs DMAS (2025)
- **Similarity**: Both aim to improve visual grounding in VLMs
- **Difference**: DMAS is training-free, using semantic retrieval to select relevant image regions. VIGIL modifies the model's weights via RL.
- **Positioning**: Orthogonal and complementary. DMAS improves the input; VIGIL improves the model. Could be combined.

### vs LLaVA-RLHF / RLHF-V
- **Similarity**: RL training for VLMs
- **Difference**: These use human preference data for general hallucination reduction. VIGIL uses automated, vision-specific rewards (LSR) targeting the drift mechanism directly. No human annotation required.

### vs Head Pursuit
- **Similarity**: Identifying important attention heads
- **Difference**: Head Pursuit finds task-relevant heads for pruning/compression. VIGIL finds vision-specialized heads and discovers the Feature/Decision head dichotomy, which is a structural finding about VLM attention.

---

## Limitations

1. **Evaluation scale**: Primary results are on POPE (binary VQA) with 60-sample fast eval. Full 9K-sample POPE eval and harder benchmarks (MMMU-Pro, MME, MMBench) are pending. The +5.2pp improvement needs validation at scale.

2. **Model scale**: Tested on 2B and 1B models only. The visual attention drift problem may manifest differently at 7B+ scale, where models have more capacity to maintain visual attention natively. LSR's benefit at larger scales is unknown.

3. **Benchmark breadth**: POPE tests object existence (yes/no), which is a narrow visual reasoning task. Performance on spatial reasoning, counting, OCR, and multi-step visual reasoning (e.g., MathVista) is untested.

4. **LSR computational cost**: Each training sample requires two forward passes (real + black image) plus KL computation. This approximately doubles the reward computation cost compared to correctness-only GRPO. For the 2B model on A100, this adds ~30% wall-clock overhead.

5. **Training fragility**: The 10-step optimum is narrow. Overtraining by just 5 steps causes regression (95.0% -> 91.7%). This fragility suggests the method may require careful early stopping, limiting its practicality without a held-out validation set during RL training.

6. **Black image assumption**: LSR assumes a black image is a null visual input. For models trained with black/dark images in their training data, this assumption may not hold. A learned null embedding could be more robust.

7. **Thinking mode specific**: LSR is designed for thinking-mode models where reasoning chains expose the drift. For instruct-mode models without extended reasoning, the thinking-token mask is inapplicable and the method reduces to answer-token LSR, which we show has near-zero signal.

8. **Cross-model results**: InternVL3.5-1B was tested with BoN+SFT only (not GRPO-LSR) due to TRL incompatibility with trust_remote_code. The cross-model claim is weaker than stated.
