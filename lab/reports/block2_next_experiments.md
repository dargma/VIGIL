# Block 2 Next Experiments: Paper-Strength Results

**Date**: 2026-03-08
**Context**: GRPO is fundamentally limited for this task (oscillates +/-1.5pp around baseline). BoN+SFT is running. Time to think bigger about what produces strong paper figures and tables.

**Core insight**: Inference-time steering already gives +2pp (alpha=1) to +9pp (alpha=10) on POPE. The paper story should be: (1) discover vision heads, (2) steer them at inference to prove the mechanism, (3) use steering to bootstrap training that *internalizes* the behavior, (4) show the trained model is more visually grounded (Blind Test Gap). The experiments below are ordered by expected paper impact.

---

## Experiment 1: Steered Best-of-N + SFT (Primary Training Result)

### Hypothesis
Applying steering (alpha=5) during Best-of-N candidate generation produces higher-quality candidates that the base model cannot generate on its own. SFT on these "steered best" candidates teaches the model to *internalize* steering behavior, permanently improving visual grounding without requiring steering at inference time.

### Protocol
1. Load base Qwen3-VL-2B-Instruct
2. Install steering hooks (alpha=5, top-20 heads from calibration)
3. For each of 2000 mixed training samples: generate N=16 candidates WITH steering
4. Score candidates: R = 0.4 * R_correct + 0.4 * R_vhad + 0.2 * R_fluency
   - R_vhad computed on the BASE model (no steering) to measure inherent grounding
   - This is key: generate with steering, but score grounding WITHOUT steering
5. Select best candidate per sample
6. SFT on (prompt, best_answer) pairs for 2 epochs, full unfreeze, lr=2e-6
7. Evaluate: POPE (3 splits), Blind Test Gap, MME
8. Compare: base vs SFT-only (BoN without steering) vs Steered-BoN-SFT

### Expected Result
- POPE: +2-4pp over baseline (85 -> 87-89%), because steered generation accesses the model's latent visual capacity (proven by alpha sweep: +9pp at alpha=10) and SFT distills this
- Blind Test Gap: +3-5pp (35 -> 38-40pp), because R_vhad scoring selects visually-grounded answers
- The gap between "BoN without steering" and "Steered BoN" isolates the contribution of steering to training data quality

### GPU Time (A100 40GB)
- Generation: 2000 samples x 16 candidates x ~0.5s = ~4.5 hours
- R_vhad scoring: 2000 x 16 x 2 forward passes x ~0.3s = ~5.3 hours
- SFT: ~1 hour (2 epochs, 2000 samples)
- Eval: ~0.5 hours
- **Total: ~11 hours** (can be split across sessions; generation results are saved to disk)

### Paper Outputs
- **Table 1**: Main results table (base / steered-inference / BoN-SFT / steered-BoN-SFT) x (POPE-R, POPE-P, POPE-A, Blind Gap, MME-P, MME-C)
- **Figure**: Bar chart comparing all 4 conditions on POPE-Adversarial + Blind Test Gap

---

## Experiment 2: R_vhad as Explicit Training Reward (Core Contribution)

### Hypothesis
R_vhad (vision head activation differential: real image vs black image) provides a continuous, per-candidate training signal that directly measures visual grounding. When used as the dominant reward in BoN+SFT candidate selection, it produces models that are measurably more image-dependent than R_correct-only training.

### Protocol
1. Use the same BoN generation from Experiment 1 (no re-generation needed)
2. Score candidates 3 ways:
   - (A) R_correct only
   - (B) R_correct + R_vhad (the VIGIL reward: 0.4 correct + 0.4 vhad + 0.2 fluency)
   - (C) R_vhad only (extreme ablation: grounding without correctness)
3. For each scoring, select best candidate per sample, SFT separately
4. Evaluate all 3 on POPE + Blind Test Gap + MME

### Expected Result
- (A) R_correct only: POPE improves (+1-2pp) but Blind Test Gap is flat or drops (blind reasoner)
- (B) Full VIGIL reward: POPE improves (+2-3pp) AND Blind Test Gap improves (+3-5pp)
- (C) R_vhad only: POPE drops slightly (-1pp) but Blind Test Gap is highest (+5-7pp)
- The ordering B > A on Gap and B > C on POPE demonstrates that the composite reward is necessary

### GPU Time
- No extra generation needed (reuse Experiment 1 candidates)
- R_vhad scoring: already computed in Experiment 1
- 3 SFT runs: ~3 hours
- 3 eval runs: ~1.5 hours
- **Total: ~4.5 hours** (incremental over Experiment 1)

### Paper Outputs
- **Table 2**: Reward ablation (R_correct / R_vhad / R_full) x (POPE, Gap, MME-P, MME-C)
- **Figure**: Radar plot showing each reward variant's profile across metrics
- This is the core ablation that proves R_vhad matters. It is the single most important table for the paper.

---

## Experiment 3: Multi-Round Iterative BoN+SFT (Compounding Improvement)

### Hypothesis
A single round of BoN+SFT is limited by the quality of candidates the current model can generate. Iterating (generate from model_k, SFT -> model_{k+1}, generate from model_{k+1}, ...) allows compounding improvement because each successive model generates better candidates than the last. 2-3 rounds should produce measurable gains beyond round 1.

### Protocol
1. Round 1: Base model -> Steered BoN (N=16, alpha=5) -> score with R_full -> SFT -> Model_1
2. Round 2: Model_1 -> Steered BoN (N=16, alpha=3, reduced because model is better) -> score -> SFT -> Model_2
3. Round 3: Model_2 -> Steered BoN (N=16, alpha=1) -> score -> SFT -> Model_3
4. Evaluate all 3 rounds + baseline on POPE + Blind Test Gap
5. Key detail: decrease steering alpha across rounds (5 -> 3 -> 1) to wean the model off steering

### Expected Result
- Round 1: +3pp POPE, +4pp Gap (from Experiment 1)
- Round 2: +1-2pp additional POPE, +2pp additional Gap
- Round 3: diminishing returns, +0.5pp POPE, +1pp Gap
- Total after 3 rounds: POPE 85 -> 90-91%, Gap 35 -> 42-44pp
- The alpha annealing curve shows the model progressively internalizing steering

### GPU Time
- 3 rounds x (generation ~4.5h + scoring ~5h + SFT ~1h + eval ~0.5h) = ~33 hours total
- But rounds 2-3 can use smaller N=8 (model is better, less diversity needed): ~22 hours total
- **Can be run as 3 separate sessions across days**

### Paper Outputs
- **Figure (candidate for Figure 2)**: Line plot of POPE accuracy and Blind Test Gap across rounds, with steering alpha annotated. Shows compounding improvement and alpha annealing.
- **Table**: Per-round results with delta from previous round

---

## Experiment 4: Vision Attention Drift in Thinking Chains (Figure 1 Candidate)

### Hypothesis
In Qwen3-VL-2B-Thinking's extended reasoning chains, vision head activation decays as a function of token position (the "visual attention drift" that is VIGIL's core problem statement). Steering prevents this decay, and a model trained with R_vhad has inherently less decay than the base model.

### Protocol
1. Load Qwen3-VL-2B-Thinking (the `enable_thinking=True` variant)
2. Generate 50 POPE responses with full thinking chains (max_new_tokens=2048)
3. During generation, record per-head activation norms at EVERY token position for top-20 vision heads
4. Plot: x = token position (0 to end of generation), y = mean vision head activation norm
5. 4 conditions on the same plot:
   - (a) Base Thinking model (expect: clear decay curve)
   - (b) Base + steering alpha=3 (expect: flat or slower decay)
   - (c) Steered-BoN-SFT model, no steering at inference (expect: less decay than (a))
   - (d) Steered-BoN-SFT model + steering (expect: flattest curve)
6. Also compute: mean accuracy for each condition to show drift correlates with performance

### Expected Result
- Base Thinking model shows O(1/L) decay in vision activation across the chain
- Steered model maintains activation (flat curve)
- SFT-trained model shows intermediate decay (partially internalized)
- SFT + steering shows the flattest curve
- This directly visualizes the core thesis: visual attention drifts, and VIGIL fixes it

### GPU Time
- 4 conditions x 50 samples x ~5s per thinking chain = ~17 minutes generation
- Activation recording adds ~20% overhead = ~20 minutes total
- **Total: < 1 hour** (this is cheap and high-impact)

### Paper Outputs
- **Figure 1 (the paper's hook figure)**: 4-line plot of vision head activation vs token position. This is the most important visualization -- it demonstrates the problem (drift) and the solution (steering + training) in a single plot.
- Could also generate a heatmap variant: x = token position, y = head index, color = activation magnitude

---

## Experiment 5: Cross-Model Generality (InternVL3.5-1B)

### Hypothesis
Vision head specialization is not Qwen-specific. InternVL3.5-1B (different architecture family, different pretraining) also has vision-specialized heads, and VIGIL's calibrate-steer-train pipeline transfers to it. This proves VIGIL is a general method, not a model-specific trick.

### Protocol
1. Load InternVL3.5-1B (trust_remote_code, fp16)
2. Run VIGIL calibration: profiler (real vs black activation delta), Cohen's d on POPE subset
3. Verify: do feature heads and decision heads also separate? (Two types hypothesis)
4. Run steering sweep: alpha = [0, 1, 3, 5] on POPE-Adversarial (200 samples)
5. Run Blind Test baseline + steered
6. If steering helps: run 1 round of Steered-BoN-SFT with R_vhad scoring
7. Compare all results to Qwen3-VL-2B in a single table

### Expected Result
- InternVL3.5-1B will also show vision head specialization (high activation delta in late layers)
- Steering will improve POPE by +1-3pp (smaller effect due to 1B scale)
- Blind Test Gap will increase with steering
- The two-types-of-heads pattern (feature vs decision) will replicate
- Even if gains are smaller, the replication across architectures is the key finding

### GPU Time
- Model loading + calibration: ~1 hour
- Steering sweep (4 alphas x 200 samples): ~30 minutes
- Blind test: ~15 minutes
- Optional BoN+SFT: ~8 hours (smaller model, faster generation)
- **Calibration + steering only: ~2 hours. Full pipeline: ~10 hours.**

### Paper Outputs
- **Table 3**: Cross-model comparison (Qwen3-VL-2B / InternVL3.5-1B) x (vision heads found, steering delta, blind gap delta, BoN-SFT delta)
- **Figure**: Side-by-side heatmaps of vision head activation by layer for both models
- Addresses the generality question that reviewers will ask

---

## Execution Plan

| Order | Experiment | GPU Hours | Dependencies | Paper Section |
|-------|-----------|-----------|--------------|---------------|
| 1 | Exp 4: Vision Drift in Thinking | <1h | None (base model only) | Figure 1, Introduction |
| 2 | Exp 1: Steered BoN+SFT | ~11h | Calibration (done) | Section 4: Main Results |
| 3 | Exp 2: R_vhad Reward Ablation | ~4.5h | Exp 1 candidates | Section 5: Ablations |
| 4 | Exp 3: Multi-Round Iteration | ~22h | Exp 1 model | Section 4: Main Results |
| 5 | Exp 5: InternVL Generality | ~2-10h | None | Section 6: Generality |

**Total**: ~40-50 GPU hours on A100 40GB (3-4 sessions).

**Critical path**: Exp 4 -> Exp 1 -> Exp 2 can run in sequence in a single long session (~16 hours). Exp 4 is cheap and produces the paper's hook figure. Exp 1+2 together produce the main result table and core ablation. Exp 3 and 5 are independent extensions that strengthen the paper but are not strictly necessary for a first submission.

### Minimum Viable Paper (if GPU budget is tight)
Run only Exp 4 + Exp 1 + Exp 2 (~16 hours). This gives:
- Figure 1: Vision drift visualization (the problem)
- Table 1: Steered-BoN-SFT improves POPE + Gap (the solution)
- Table 2: R_vhad ablation proves the reward design matters (the mechanism)

That is sufficient for a workshop paper. Add Exp 3 + Exp 5 for a full conference submission.
