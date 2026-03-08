# Block 2 v2 Custom GRPO — Code Review Bug Report

**File**: `/content/drive/MyDrive/VIGIL/scripts/block2_custom_grpo_v2.py`
**Date**: 2026-03-08
**Reviewer**: Claude (automated)

---

## Bug 1: KL Divergence Is Always Zero (CRITICAL)

**Location**: Lines 663-672 (ref logprob computation) + lines 410-415 (KL formula in `compute_grpo_loss`)

**Problem**: Reference log-probs are computed from the *current* model immediately before the GRPO loss forward pass within the same step. Since no gradient step has occurred between computing `ref_lps` (line 669) and the policy forward pass inside `compute_grpo_loss` (line 396-403), the model weights are identical. Therefore `cur_lp ~= ref_lp` and KL is always approximately zero.

This explains the `KL=0.0000` observed in all training logs. The `beta_kl=0.02` KL penalty term is doing nothing, meaning there is no constraint preventing the policy from drifting arbitrarily far from the reference.

**Impact**: HIGH. Without a working KL penalty, the only regularization is gradient clipping (`max_grad_norm=1.0`) and the PPO clip (`epsilon_clip=0.2`). The PPO clip alone is insufficient because ratios are also ~1.0 when ref and current are the same model, so the clipping never activates either. This likely contributes to the collapse patterns seen in Block 1.

**Fix options (ranked)**:
1. **Frozen ref model copy (recommended)**: At initialization, create `ref_model = deepcopy(model)` with `requires_grad=False` and `ref_model.eval()`. Use `ref_model` for all reference logprob computation. Cost: ~4.3GB extra VRAM for Qwen3-VL-2B in bf16. This is the standard GRPO/PPO approach.
2. **EMA reference model**: Maintain an exponential moving average of model weights. More memory-efficient than a full copy if using shared buffers, but adds complexity. Update EMA every N steps.
3. **Save initial logprobs at step 0**: Pre-compute reference logprobs for all training samples before training begins. Avoids extra VRAM but requires storing logprobs for all samples (disk I/O cost), and becomes stale as training progresses (which may actually be desired for a fixed reference).

Note: `from copy import deepcopy` is already imported on line 29 but never used. This suggests option 1 was originally intended.

---

## Bug 2: num_return_sequences Without Input Expansion (MODERATE)

**Location**: Lines 217-224

**Problem**: `model.generate()` is called with `num_return_sequences=group_size` (default 8) on a single input (batch_size=1). For HuggingFace `generate()`, `num_return_sequences > 1` with `do_sample=True` internally expands the input batch by repeating it `num_return_sequences` times. This is handled correctly by the generate method itself — no manual input expansion is needed.

**Verdict**: This is CORRECT for Qwen3-VL's generate. The HF `GenerationMixin` handles the expansion internally via `_expand_inputs_for_generation`. However, there is a **memory concern**: with group_size=8, this creates an effective batch of 8 multimodal inputs (including 8 copies of pixel_values). For a 448x448 image with Qwen3-VL's vision encoder, this could spike VRAM significantly during generation.

The OOM fallback on line 225-238 partially addresses this, but the fallback halves group_size to 4, which may produce insufficient reward diversity for GRPO advantage computation.

---

## Bug 3: Gradient Checkpointing Toggle (MINOR)

**Location**: Lines 626-633 (generation), lines 450-451/475-476 (eval_pope), lines 485-486/517-518 (eval_blind_test)

**Pattern**:
```python
model.eval()
model.gradient_checkpointing_disable()
# ... generate or eval ...
model.train()
model.gradient_checkpointing_enable()
```

**Verdict**: This is CORRECT. Gradient checkpointing must be disabled during `generate()` because generation uses `use_cache=True` (KV caching), which is incompatible with gradient checkpointing. The toggle pattern is standard.

**Minor concern**: In `eval_pope` (line 735) and `eval_blind_test` (line 736), the function internally calls `model.eval()` and `model.gradient_checkpointing_disable()`, then restores `model.train()` and `model.gradient_checkpointing_enable()`. But the caller at line 734 also calls `model.eval()`, and at line 737 calls `model.train()`. This double-toggle is harmless but redundant. More importantly, `gradient_checkpointing_enable()` is called inside the eval functions but the caller does NOT explicitly re-enable it — it relies on the eval function to do so. This coupling is fragile but not buggy.

---

## Bug 4: Partial Credit Reward Breaks Advantage Distribution (MODERATE)

**Location**: Lines 310-312

```python
if qtype == "short_answer" and r_correct == 0:
    if gt.lower() in cand.lower():
        r_correct = 0.5  # partial credit for containing answer
```

**Problem**: This partial credit creates a three-level reward for short_answer: {0.0, 0.5, F1_score}. The 0.5 value is applied when the F1-based `compute_r_correct` returns 0 (no word overlap) but the ground truth string appears as a substring in the candidate.

This is subtly wrong: `compute_r_correct` for `short_answer` type uses word-level F1. If F1 is 0 (no common words), the ground truth cannot appear as complete words in the candidate. The only way `gt.lower() in cand.lower()` triggers after F1=0 is when the GT appears as a substring of a larger word. Example: GT="car", candidate="carnival" gives F1=0 but substring match=True, awarding 0.5 partial credit for a wrong answer.

**Impact**: MODERATE. This inflates rewards for some wrong answers, which reduces the effective variance in the GRPO group (more candidates cluster at 0.5 instead of a clean 0.0/1.0 split). This is counterproductive since reward variance is already the bottleneck.

**Additionally**: When `use_iig=True` (Setting B), the partial credit is fed into `vigil_reward()` which combines it with IIG. The 0.5 partial credit may mask the IIG signal since `r_correct` and IIG are on similar scales.

---

## Bug 5: Memory Leak in Checkpoint Saving (MINOR)

**Location**: Lines 750-755

```python
torch.save({
    "step": step,
    "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
    "optimizer_state_dict": optimizer.state_dict(),
    "eval_history": eval_history,
}, str(ckpt / "checkpoint.pt"))
```

**Problem**: `model.state_dict()` creates a full copy of all parameters. The dict comprehension `{k: v.cpu() ...}` creates CPU copies. Both the GPU state_dict and CPU copies exist simultaneously in memory before the save completes. For a 2B model in bf16, this is ~4.3GB GPU + ~4.3GB CPU transiently.

**No persistent leak**, but this transient spike could cause OOM during checkpointing, especially since the optimizer state (AdamW with momentum and variance) is another ~8.6GB.

**Other memory observations**:
- Line 694: `del loss, inputs, cand_ids, candidates, ref_lps` is good practice.
- The `gc.collect()` on line 695 is appropriate.
- `torch.cuda.empty_cache()` every 3 steps (line 704) is reasonable.
- `eval_history` list grows unboundedly but is tiny (dicts of scalars).
- `log_history` similarly grows but is small.

**No persistent memory leak found.**

---

## Bug 6: Loss Scaling Is Correct But Interacts Poorly With Variable step_valid (MINOR)

**Location**: Line 683

```python
(loss / args.samples_per_step).backward()
```

And line 436:
```python
return total_loss / n_valid, ...
```

**Analysis**: The loss is first averaged over `n_valid` candidates inside `compute_grpo_loss`, then divided by `samples_per_step` before `.backward()`. This implements manual gradient accumulation across the `samples_per_step` samples: each sample contributes `1/samples_per_step` of the total gradient.

**This is correct** for the intended gradient accumulation pattern. The optimizer step happens once per outer step (line 699), and the accumulated gradient is the mean over samples.

**Edge case**: When `step_valid < samples_per_step` (some samples skipped due to zero-variance rewards or OOM), the effective learning rate increases because the gradient is still divided by `samples_per_step` but fewer terms contribute. Example: if `samples_per_step=2` but only 1 sample has a valid loss, the gradient is halved compared to what it would be without the division. This is conservative (under-steps), which is actually safer than over-stepping.

---

## Summary

| # | Issue | Severity | Impact on Training |
|---|-------|----------|-------------------|
| 1 | KL always zero (ref == current model) | CRITICAL | No policy constraint, contributes to collapse |
| 2 | num_return_sequences memory spike | MODERATE | OOM risk with large images, fallback reduces diversity |
| 3 | Gradient checkpointing toggle | OK | Correct, minor redundancy |
| 4 | Partial credit 0.5 for substring match | MODERATE | Rewards wrong answers, reduces reward variance |
| 5 | Transient memory spike during checkpoint | MINOR | Possible OOM during save, no persistent leak |
| 6 | Loss scaling with variable step_valid | MINOR | Conservative (under-steps), not harmful |

**Priority fix order**: Bug 1 (KL) >> Bug 4 (partial credit) > Bug 2 (memory) > rest.

Bug 1 alone likely explains why KL=0.0000 appears in all logs and why the model drifts unconstrained. Fixing this should be the first action before any further training runs.
