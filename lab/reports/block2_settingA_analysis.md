# Block 2 Setting A Analysis: First Non-Collapsing GRPO Run

**Date**: 2026-03-08
**Run**: `block2v2_A_20260308_013027.json`
**Model**: Qwen3-VL-2B-Instruct (full unfreeze, bf16)

---

## 1. Summary of Results

| Metric | Step 0 | Step 10 | Step 20 | Step 30 | Step 40 | Step 50 | Delta |
|--------|--------|---------|---------|---------|---------|---------|-------|
| POPE Acc | 84.5% | 84.5% | 84.0% | 84.5% | 84.5% | 83.5% | -1.0pp |
| Blind Gap | 35.0pp | 35.0pp | 34.0pp | 35.0pp | 35.0pp | 33.0pp | -2.0pp |
| Yes/No ratio | 49.5/50.5 | 48.5/51.5 | 49/51 | 48.5/51.5 | 49.5/50.5 | 49.5/50.5 | stable |

The yes/no balance stayed near 50/50 throughout. This is the key success: **no mode collapse**. All three Block 1 runs collapsed to all-yes or all-no within 5-20 steps.

## 2. What Prevented Collapse

Three changes from Block 1 worked together:

1. **Full unfreeze (no LoRA)**: Block 1 v1-v3 all used LoRA (r=4 to r=16). LoRA's low-rank update space may be too constrained for the model to find a gradient direction that improves reward without degenerate shortcuts. Full unfreeze gives the optimizer access to the complete loss surface.

2. **Mixed non-binary data**: Training on VQAv2 short-answer (734), A-OKVQA MC (600), and TextVQA open-ended (666) instead of POPE binary yes/no. Non-binary answers have much higher output entropy, so group members naturally diverge.

3. **Zero-variance group skipping + entropy bonus**: Groups where all 8 members agree get skipped instead of producing noise gradients. The entropy bonus (beta_entropy=0.01) provides a small pressure against entropy collapse.

## 3. Detailed Training Dynamics

### Effective utilization was very low

- **50 nominal steps, but only 32 produced gradients** (64%)
- **62 of 100 groups skipped** (62% skip rate)
- Only 6 steps had both samples produce valid (non-zero-variance) groups
- Total wall time: 2.5 minutes (most steps skipped in <1s)
- Mean loss across valid steps: -0.0077 (essentially zero net gradient)

### The model barely learned anything

The near-zero mean loss explains the flat POPE curve. With samples_per_step=2 and group_size=8, each step processes 16 generations, but 62% of groups are discarded. On a valid step, at most 8 generations contribute to the gradient. Over 50 steps, the model saw roughly 38 valid groups x 8 generations = 304 effective samples. At lr=5e-7, this is an extremely weak training signal.

### KL was always zero — a critical bug

KL=0.0000 at every step means the ref model and policy model have identical logprobs. This happens because `beta_kl=0.02` is configured but the KL computation uses the same model weights for both policy and reference (no frozen copy, no LoRA toggling in v2). Without KL regularization, the only thing preventing drift is the weak learning rate. This must be fixed.

## 4. Why POPE Barely Moved

### 4a. Insufficient effective training signal

With only 304 effective samples across 50 steps, and a learning rate of 5e-7, the total parameter update magnitude is negligible. For comparison, Block 1 v1 (which collapsed) used lr=5e-6 with all samples contributing. The model weights barely changed.

### 4b. Evaluation is on POPE, training is on mixed non-binary data

The training data (VQAv2 short-answer, A-OKVQA MC, TextVQA) has zero overlap with POPE in terms of task format. POPE is binary yes/no object presence. Transfer from open-ended VQA to binary classification is indirect. We should expect small POPE changes even with strong training signal.

### 4c. 50 steps is not enough

Even if all groups were valid, 50 steps x 2 samples x 8 generations = 800 total generations. Standard GRPO runs use 1000-10000 steps. The current run is 10-100x too short.

### 4d. Blind test Gap decline (-2pp) is within noise

The blind test uses only 100 samples. A 2pp change is 2 samples, well within binomial noise (95% CI: +/-6pp at n=100). The Gap is essentially stable.

## 5. Diagnosing the 62% Skip Rate

### Why groups have zero variance

Group size=8 with max_new_tokens=64 at temperature=1.4. For many VQA questions, even with high temperature, 8 samples converge to the same correct answer (or the same wrong answer). This is especially true for:

- **Short factual answers** (VQAv2: "blue", "3", "dog") — limited output space
- **MC questions** (A-OKVQA) — only 4 choices, so P(all 8 agree) = sum(p_i^8) which is high when one option dominates

### Potential fixes (ranked by expected impact)

1. **Increase group_size to 16 or 32**: P(all agree) = p^N drops exponentially with N. Going from 8 to 16 roughly squares the probability of getting variance. This is the single most impactful change.

2. **Increase temperature to 1.6-2.0**: More randomness in sampling. Risk: incoherent outputs. Mitigate by filtering invalid responses.

3. **Increase max_new_tokens to 128-256**: Longer outputs have more tokens to disagree on, creating token-level reward variance even when final answers agree. Also allows chain-of-thought reasoning.

4. **Use token-level reward instead of sequence-level**: If the reward is binary (correct/incorrect), all tokens in a sequence get the same reward. Token-level partial credit (e.g., reward for each correct reasoning step) would provide variance even within agreeing groups.

5. **Sample harder questions preferentially**: Easy questions (where the model is confident) produce zero-variance groups. Hard questions (model accuracy 30-70%) maximize expected variance. Curriculum sampling by model confidence.

## 6. The KL Bug and How to Fix It

The v2 script uses full unfreezing with no separate reference model. The KL term is computed as `kl = (policy_logprobs - ref_logprobs)` but both come from the same forward pass (same weights). Result: KL is identically zero, providing no regularization.

### Fix options

1. **Snapshot reference logprobs at step 0**: Before training begins, compute and cache ref_logprobs for each sample. Compare against current policy logprobs at each step. This is memory-cheap but only works if the training data is fixed (no dynamic sampling).

2. **Keep a frozen model copy**: Load two copies of the model. Use one as ref. At 2B bf16, this costs ~4GB extra VRAM. On L4 (23GB), this may be feasible if generation is done sequentially.

3. **Periodic ref model sync**: Every N steps, copy current weights to ref model. Cheaper than continuous KL but still provides drift protection.

Option 1 is simplest and most memory-efficient. For 2000 training samples with 64 tokens each, caching logprobs requires ~2000 x 64 x 4 bytes = 0.5MB. Trivial.

## 7. Learning Rate Analysis

At lr=5e-7 with only 38 valid gradient steps, the effective total learning is approximately:

```
Total param shift ~ lr * num_steps * mean_grad_magnitude
                  ~ 5e-7 * 38 * O(1)
                  ~ 2e-5 (relative to initial weights)
```

This is extremely conservative. Block 1 collapsed at lr=5e-6 (10x higher), but Block 1 used LoRA on binary data. With full unfreeze on mixed data:

- **Try lr=2e-6** (4x increase): This was the v1 default. With zero-variance skipping and entropy bonus, collapse risk is lower.
- **Try lr=1e-6** (2x increase): More conservative middle ground.
- **Do NOT go above 5e-6**: Block 1 showed this is dangerous territory.

The learning rate increase should be combined with the KL fix. Without KL, higher LR will drift faster with no safety net.

## 8. Recommended Setting B Configuration

Based on this analysis, the next run (Setting B: R_correct + IIG) should change these parameters:

| Parameter | Setting A | Recommended B | Rationale |
|-----------|-----------|---------------|-----------|
| group_size | 8 | 16 | Reduce skip rate from 62% to ~35% |
| lr | 5e-7 | 2e-6 | 4x more learning per step |
| max_new_tokens | 64 | 128 | More output diversity |
| num_steps | 50 | 100 | Double training duration |
| lambda_iig | 0.0 | 0.0615 | Enable IIG reward |
| KL | broken (=0) | Fix with cached ref logprobs | Safety net for drift |
| samples_per_step | 2 | 4 | More data per optimizer step |

Expected outcome with these changes:
- Skip rate: ~35% (down from 62%)
- Effective gradient steps: ~65 of 100 (up from 32 of 50)
- Effective generations: ~65 x 4 x 16 = 4160 (up from 304)
- **This is a 14x increase in effective training signal**

## 9. Theoretical Maximum Improvement

### What GRPO can and cannot do here

GRPO optimizes towards higher reward within the model's existing capability distribution. It cannot teach the model new knowledge — only shift probability mass toward already-possible good responses.

For POPE at 84.5% baseline:
- The model already gets 169/200 correct. The 31 errors are likely genuinely hard cases (adversarial negatives).
- GRPO might fix 5-10 of these by shifting probability slightly, giving a ceiling of ~89-92%.
- But we are training on non-POPE data, so transfer is indirect. Realistic ceiling: 86-88% (+2-4pp).

For Blind Test Gap at 35pp:
- Adding IIG reward (Setting B) directly rewards image-grounding behavior.
- If IIG works as designed, Gap should increase by 2-5pp to 37-40pp.
- This is the more important metric for VIGIL's thesis.

### The real question

The real question is not "can GRPO improve POPE by 5pp" but "can GRPO with IIG reward produce a model that is measurably more visually grounded (higher Gap) without sacrificing accuracy." Setting A establishes that the training loop is stable. Setting B tests whether IIG reward achieves its purpose.

## 10. Key Takeaways

1. **Block 2 v2 solved the collapse problem.** Full unfreeze + mixed data + entropy bonus = stable training. This is the first successful GRPO run in the project.

2. **But the model barely learned.** 62% skip rate + lr=5e-7 + 50 steps = negligible parameter updates. The stability came partly from the training being too weak to move anything.

3. **The KL bug means there is no safety net.** If we increase LR or steps, we need working KL regularization first.

4. **Group size is the primary lever for skip rate.** Going from 8 to 16 is the single most impactful change.

5. **Setting B should be run with the recommended configuration changes.** The current Setting A is a proof of stability, not a proof of learning. The next run needs 10-14x more effective training signal to see meaningful metric movement.
