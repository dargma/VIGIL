# Block 2 Best-of-N + SFT: Pre-Execution Analysis

**Date**: 2026-03-08
**Context**: GRPO v2 Setting B achieved +0.5pp POPE, +1pp Gap in 50 steps. GRPO v3 (100 steps, all fixes) ended at -1pp POPE, -2pp Gap. GRPO has plateaued. This analysis evaluates the Best-of-N + SFT approach before running it.

---

## 1. Theoretical Best-of-N Accuracy

### Formula

For independent samples with base accuracy p, the probability that at least one of N candidates is correct:

```
P(at least 1 correct in N) = 1 - (1 - p)^N
```

### Expected Accuracy by N

Using baseline POPE accuracy as a proxy for per-sample correctness probability (p = 0.845 from POPE-Adv baseline):

| N | P(best correct) | Improvement over base |
|---|----------------|-----------------------|
| 1 | 84.5% | -- |
| 2 | 97.6% | +13.1pp |
| 4 | 99.9% | +15.4pp |
| 8 | ~100% | +15.5pp |
| 16 | ~100% | +15.5pp |

This is misleading for two reasons:

1. **POPE is binary** (yes/no). For any sample, p is either high (model is confident and correct) or low (model is uncertain or wrong). The 31 wrong samples out of 200 likely have p < 0.5, where best-of-8 gives:
   - p=0.3: 1-(0.7)^8 = 94.2% (huge gain)
   - p=0.1: 1-(0.9)^8 = 56.9% (moderate gain)
   - p=0.0: 0% (no gain possible -- model cannot answer this)

2. **Training data is NOT POPE**. The script uses mixed non-binary data (TextVQA, A-OKVQA MC, VQAv2 short-answer). For these tasks, baseline accuracy is lower and the answer space is larger, so:
   - Open-ended VQA: p ~ 0.3-0.5 typically
   - MC (4 choices): p ~ 0.4-0.7
   - Best-of-8 at p=0.4: 1-(0.6)^8 = 98.3%

### Realistic Estimate

For the mixed training data with ~60% average base accuracy:
- Best-of-8 correct rate: 1-(0.4)^8 = 99.93%
- But this assumes perfect reward ranking. With noisy R_correct (0/0.5/1 with substring matching), effective best-of-N is lower.
- **Realistic estimate: ~85-90% of training samples will have a "correct" best candidate** (up from ~60% baseline).

This means SFT will train on a dataset where ~85-90% of responses are correct, versus the base model's ~60% correctness. This is the key advantage.

---

## 2. Will SFT on Mixed Data Improve POPE?

### Arguments FOR transfer

1. **General reasoning improvement**: SFT on high-quality VQA answers should improve the model's general visual reasoning, which transfers to POPE. The model learns to look at images more carefully, produce more grounded answers.
2. **IIG-scored candidates are more visually grounded**: With lambda_iig=0.0615, the best candidate is not just correct but also image-dependent. SFT on these teaches the model to habitually use visual information, which directly helps POPE (where image content determines the answer).
3. **Precedent**: GRPO v2 Setting B (with IIG) showed +0.5pp POPE despite training on the same mixed non-binary data. SFT should provide a stronger and more stable training signal.
4. **Format diversity helps generalization**: Training on TextVQA + A-OKVQA + VQAv2 exposes the model to diverse visual reasoning patterns. POPE is a simpler task, so general improvement should transfer.

### Arguments AGAINST transfer

1. **Domain gap**: Training data is open-ended/MC, POPE is binary yes/no. The model may learn to produce longer, more detailed answers that don't transfer to single-token POPE responses.
2. **POPE is already saturated**: At 84.5%, the remaining errors are adversarial negatives that are genuinely hard. SFT on unrelated data cannot fix "Is there a fire hydrant?" when the model doesn't recognize fire hydrants.
3. **Short answer training**: With max_new_tokens=64 and "answer in a few words" prompts, SFT candidates are short. This may not build robust reasoning patterns.
4. **Catastrophic forgetting risk**: Full unfreeze SFT on 1000 samples for 2 epochs may degrade POPE performance. The model could overfit to the training distribution (open-ended VQA) and lose binary classification ability.

### Prediction

**POPE delta: +0 to +2pp** (83.5-86.5%). The transfer is indirect and POPE is near saturation. The best outcome is matching the v2 Setting B peak (85.0%) consistently rather than the oscillating 83.5-85.0% range we saw in GRPO.

The more likely win is **stability**: SFT produces a fixed model that either helps or doesn't, without the oscillation and eventual regression that GRPO showed over 100 steps.

---

## 3. Blind Test Gap: Will IIG Scoring Help?

### How IIG influences candidate selection

When lambda_iig > 0, the reward becomes:
```
score = R_correct + lambda * max(IIG - eps, 0)
```

With lambda=0.0615, eps=0.1, and typical IIG values of 0.5-10.0 for correct answers:
- IIG contribution: 0.0615 * (IIG - 0.1) = 0.025 to 0.61
- R_correct contribution: 0 or 1

For correct candidates (R_correct=1.0), IIG acts as a tiebreaker: among several correct answers, the one with highest IIG is selected. This candidate is the one where the model's logprobs were most different with vs without the image -- i.e., the most image-dependent correct answer.

### Impact on Blind Test Gap

SFT on IIG-ranked candidates teaches the model to produce answers that are both correct AND image-dependent. This should:

1. **Increase real-image accuracy**: The selected candidates are the "best" correct answers -- ones most grounded in visual content.
2. **Not change blind accuracy much**: Black-image accuracy (50%) is determined by text priors. SFT on visually-grounded answers doesn't help text-only performance.
3. **Net effect: Gap increases**.

### Prediction

**Gap delta: +1 to +3pp** (36-38pp). The IIG tiebreaker effect is meaningful because it selects qualitatively different answers (image-grounded vs text-prior). But the effect size is limited because:
- Many samples have only 1 correct candidate (no tiebreaker needed)
- IIG's contribution to the composite score is small relative to R_correct (0.06 vs 1.0)
- 2 SFT epochs on 1000 samples is a light training signal

To maximize Gap improvement, lambda_iig should be increased (e.g., 0.2-0.5) so IIG has more influence on candidate ranking. But this risks selecting incorrect-but-grounded candidates over correct-but-ungrounded ones.

---

## 4. How Many Iterations?

### Iterative Best-of-N (ReST/RAFT framework)

Each iteration:
1. Generate N=8 candidates per sample using model_k
2. Score and select best
3. SFT on best candidates to produce model_{k+1}

Theoretical analysis:
- **Iteration 1**: Biggest jump. Model goes from ~60% to ~85-90% correct training data. SFT on this should give the largest improvement.
- **Iteration 2**: model_1 is already better, so candidates are higher quality. But diminishing returns -- the easy wins are captured in iteration 1.
- **Iteration 3+**: Risk of overfitting increases. Each iteration narrows the training distribution. After 3+ rounds, the model converges to a fixed point.

### Practical constraints

- **GPU time per iteration**: ~4 hours generation + ~1 hour SFT = ~5 hours on A100
- **Diminishing returns**: DeepSeek-R1's ReST used 2-3 iterations. Llama's RAFT used 1-2.
- **Monitoring**: Must eval POPE + Gap after each iteration. Stop if metrics regress.

### Recommendation

**Plan for 2 iterations, evaluate after each, stop if iteration 2 regresses.**

- Iteration 1: baseline model -> generate -> SFT -> model_1. Eval.
- Iteration 2: model_1 -> generate -> SFT -> model_2. Eval.
- Iteration 3: Only if both iterations 1-2 showed improvement AND Gap is still increasing.

Budget: ~10-12 hours total GPU time.

---

## 5. Risks

### 5.1 Catastrophic Forgetting

**Risk level**: HIGH. Full unfreeze SFT on 1000 samples for 2 epochs = 2000 gradient updates (with grad_accum=8, that's 250 optimizer steps). At lr=2e-6, this is 4x more aggressive than the GRPO v2 that already showed oscillation.

**Mitigation**:
- Reduce to 1 epoch first, evaluate, only do epoch 2 if no regression
- Use lr=1e-6 (half the default) for safety
- Keep a frozen copy of the base model for comparison

### 5.2 Overfitting to Short Answers

**Risk level**: MEDIUM. With max_new_tokens=64 and "answer in a few words" prompts, the model learns to produce very short answers. This could:
- Improve POPE (which wants single-token yes/no) -- actually beneficial
- Hurt longer-form benchmarks (MMBench, MME) -- must evaluate
- Make the model less useful as a general VLM

**Mitigation**: Include some TextVQA samples which require longer answers (descriptions, OCR text). The 40/30/30 mix already handles this.

### 5.3 Reward Model Quality

**Risk level**: MEDIUM. The composite reward (R_correct + lambda*IIG) has known issues:
- R_correct uses substring matching with partial credit (0.5 for substring match). The block2 analysis flagged this as a bug ("car" in "carnival" = 0.5 credit).
- IIG is noisy on short answers (single-token responses have high variance)
- Together, the reward may select confidently-wrong-but-grounded candidates

**Mitigation**: Fix the substring matching bug before running. Use exact match for binary questions, token-F1 for open-ended.

### 5.4 Distribution Narrowing Across Iterations

**Risk level**: LOW for 2 iterations, HIGH for 3+. Each iteration trains on the previous model's best outputs, which are a subset of the model's capability. After multiple rounds:
- The model converges to a narrow "dialect" of answers
- Diversity drops, making future best-of-N less effective
- The model may lose ability to generate creative/diverse answers

This is the known "mode collapse of iterative distillation." Two iterations are safe.

---

## 6. Comparison with GRPO Results

### GRPO v2 Setting B (50 steps): The Best GRPO Run

| Metric | Start | End | Delta |
|--------|-------|-----|-------|
| POPE | 84.5% | 85.0% | +0.5pp |
| Blind Gap | 35.0pp | 36.0pp | +1.0pp |
| Training time | -- | ~3 min | -- |
| Collapse | -- | No | -- |

### GRPO v3 Setting B (100 steps): The Full Run

| Metric | Start | Peak | End | Delta (end) |
|--------|-------|------|-----|-------------|
| POPE | 84.5% | 85.0% (step 50,80) | 83.5% | -1.0pp |
| Blind Gap | 35.0pp | 35.0pp (step 50,80) | 33.0pp | -2.0pp |
| Skip rate | -- | -- | 44% | -- |
| Training time | -- | -- | ~20 min | -- |

Key observation from v3: **GRPO oscillates and eventually regresses**. The model peaks at step 50-80 then degrades. Steps 10-70 show POPE bouncing between 83.5% and 85.0%, and Gap between 33.0pp and 35.0pp. At step 100, it ended below baseline on both metrics. This pattern -- initial mild improvement followed by regression -- is characteristic of GRPO with insufficient reward signal. The advantage estimates are too noisy to consistently push the model in the right direction.

### Expected Best-of-N + SFT Comparison

| Metric | GRPO v2 (best) | GRPO v3 (end) | BoN+SFT (predicted) |
|--------|---------------|---------------|---------------------|
| POPE delta | +0.5pp | -1.0pp | +0 to +2pp |
| Gap delta | +1.0pp | -2.0pp | +1 to +3pp |
| Stability | oscillating | regressed | fixed (no drift) |
| GPU time | ~3 min | ~20 min | ~5 hours |
| Collapse risk | low | low | very low |

### Why BoN+SFT Should Beat GRPO

1. **No noise accumulation**: GRPO takes many small noisy gradient steps that can cancel out or drift. BoN+SFT takes one large clean step (SFT on curated data). No oscillation.

2. **Immune to zero-variance problem**: GRPO v3 had 44% skip rate (zero-variance groups). BoN+SFT doesn't need within-group variance -- it just picks the best of N regardless.

3. **Stronger signal per sample**: In GRPO, a sample contributes a tiny advantage-weighted gradient. In SFT, a sample contributes a full cross-entropy loss toward the target answer. Orders of magnitude stronger learning signal per sample.

4. **IIG acts as tiebreaker, not noise**: In GRPO, IIG adds ~0.06 to a 0/1 reward, creating tiny advantage differences that get swamped by noise. In BoN+SFT, IIG deterministically selects the most grounded correct answer. Cleaner selection.

### Why BoN+SFT Might NOT Beat GRPO

1. **GRPO explores, SFT exploits**: GRPO can discover novel answer strategies through sampling. SFT only reinforces what the base model already generates. If the model can't generate a good answer in 8 tries, BoN+SFT can't help.

2. **SFT may overfit**: 1000 samples for 2 epochs with full unfreeze is aggressive. GRPO's weak learning signal is also its safety net -- it can't overfit because it barely learns. SFT can.

3. **Transfer gap is the same**: Both approaches train on mixed non-binary data and evaluate on POPE. The domain gap doesn't change.

---

## 7. Recommended Execution Plan

### Phase 1: Generate candidates (iteration 1)
- N=8 candidates per sample (not 16 -- diminishing returns above 8 at p~0.6)
- 1000 training samples (same as GRPO v2/v3)
- lambda_iig=0.0615 (calibrated value)
- **FIX**: substring matching bug in R_correct before running
- Temperature=1.2, top_p=0.95 (match GRPO v2/v3 settings)
- **Estimated time**: 2-3 hours on A100

### Phase 2: SFT (iteration 1)
- 1 epoch first (not 2), evaluate, then decide on epoch 2
- lr=1e-6 (conservative -- half the script default)
- grad_accum=8, batch_size=1
- Full unfreeze with gradient checkpointing
- **Estimated time**: 30-45 min

### Phase 3: Evaluate
- POPE-Adv (200 samples)
- Blind Test (100 samples)
- Check yes/no balance (collapse indicator)
- **Success criteria**: POPE >= 84.5% (no regression) AND Gap >= 35.0pp (no regression)
- **Stretch goal**: POPE >= 86% OR Gap >= 37pp

### Phase 4: Decide on iteration 2
- If Phase 3 shows improvement: repeat with model_1 as base
- If Phase 3 shows regression: reduce lr to 5e-7, try 1 epoch only
- If Phase 3 shows no change: increase lambda_iig to 0.2 for stronger grounding pressure

---

## 8. Summary

Best-of-N + SFT is the right next step after GRPO plateaued. The theoretical improvement ceiling is high (best-of-8 at p=0.6 gives ~98% correct training data), the approach is stable and well-understood, and it directly addresses GRPO's core weakness (zero-variance groups producing noisy gradients).

The realistic expected improvement is modest: +0 to +2pp POPE, +1 to +3pp Gap. This is comparable to GRPO v2's peak (+0.5pp POPE, +1pp Gap) but more stable -- it won't oscillate and regress like v3 did over 100 steps.

The key risks are catastrophic forgetting (mitigate with conservative lr and single epoch) and reward model quality (fix substring matching). Two iterations should be the plan, with evaluation gating each step.

The most important metric to watch is **Blind Test Gap**, not POPE accuracy. POPE is near saturation and the remaining errors are hard. Gap directly measures visual grounding, which is VIGIL's core thesis. If BoN+SFT increases Gap by 2-3pp while maintaining POPE, that is a clear win for the paper.
