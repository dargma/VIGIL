# Block 2 Best-of-N + SFT Code Review

**File**: `scripts/block2_best_of_n_sft.py`
**Date**: 2026-03-08

---

## Bug 1 (HIGH): IIG computed with model in wrong state during generation phase

**Location**: `phase_generate()`, lines 196-199

When `compute_iig` is called inside the generation loop, it calls `model_info["model"]` which is the same model loaded with `for_training=False` (line 470). This model is in `eval()` mode and fp16, which is fine. However, `compute_iig` calls `_build_inputs` -> `make_chat_prompt` from `model_registry.py`, NOT the local `build_inputs` function defined at line 85. The two functions are nearly identical but are **separate code paths** -- any future drift between them would cause silent divergence. Not a crash bug, but a maintenance hazard.

**Severity**: Low (currently functionally equivalent, but fragile).

---

## Bug 2 (HIGH): IIG is not batched -- N forward passes per candidate per sample

**Location**: `phase_generate()`, lines 193-203

For each of the N candidates, `compute_iig` does **two full forward passes** (real image + black image). For N=8 candidates and 1000 samples, that is `8 * 2 * 1000 = 16,000` forward passes just for IIG scoring, on top of the 1000 generation calls. The `compute_iig_batch_candidates` function exists in `src/iig.py` (line 114) but is itself just a loop wrapper -- it does not actually batch. This is a major performance issue but not a correctness bug.

**Impact**: Generation phase will be extremely slow (~16x slower than necessary). The real and black image prompt encodings could be computed once per sample and reused across all N candidates.

---

## Bug 3 (HIGH): Prompt length mismatch in SFT label masking

**Location**: `phase_sft()`, lines 308-318

The label masking strategy tokenizes the prompt separately (lines 308-316) to get `prompt_len`, then uses it to mask `labels[0, :prompt_len] = -100`. This is **unreliable** for two reasons:

1. **Chat template difference**: The full-conversation template (line 292, `apply_chat_template(messages)` with user+assistant) and the prompt-only template (line 309-311, `apply_chat_template(prompt_messages, add_generation_prompt=True)`) may produce different tokenizations around the boundary. The full template includes the assistant turn header, while the prompt-only version ends at the generation prompt marker. Depending on the template, these may or may not align -- e.g., some templates add `<|im_start|>assistant\n` as the generation prompt, but the full conversation template may format the assistant header differently.

2. **Image token expansion is non-deterministic with padding**: Both `processor()` calls (lines 293-297 and 312-316) process the image independently. For Qwen3-VL, the processor converts images to `pixel_values` and inserts image placeholder tokens into the text. If the two calls produce the same image token count, the masking works. But if any processor state or rounding differs, `prompt_len` will be wrong, causing either (a) some answer tokens to be masked (under-training) or (b) some prompt tokens to be unmasked (training on prompt, mild noise).

**Fix direction**: Instead of re-tokenizing, search for a sentinel token (e.g., `<|im_start|>assistant`) in the full `input_ids` and mask everything before it.

---

## Bug 4 (MEDIUM): Gradient checkpointing is re-enabled after eval inside training loop

**Location**: `eval_pope()` line 407, `eval_blind_test()` line 441

Both eval functions call `model.gradient_checkpointing_enable()` at the end to restore training state. This is correct in principle. However, if `eval_pope` or `eval_blind_test` is called on a **generation-phase model** (loaded with `for_training=False`), it will **turn ON gradient checkpointing** on a model that never had it. This happens at lines 474-475 where baseline eval runs on the generation model.

With gradient checkpointing enabled on a model in `eval()` mode, subsequent `model.generate()` calls may break because gradient checkpointing is incompatible with KV cache (it re-computes activations, which conflicts with the caching mechanism).

**Impact**: After baseline eval at line 474-475, the model has gradient_checkpointing ON. The subsequent `phase_generate()` call at line 478 uses `model.generate()` -- this may fail or produce incorrect outputs on some model/transformers versions.

**Fix direction**: The eval functions should save and restore the original gradient_checkpointing state, or `phase_generate` should explicitly ensure gradient_checkpointing is OFF before generation.

---

## Bug 5 (MEDIUM): Image mapping in SFT phase uses question text as key

**Location**: `phase_sft()`, lines 254-257

The `q_to_image` mapping uses `s.get("prompt_text", s["question"])` as the dictionary key, and the lookup uses `entry["question"]` (line 280) where `entry` comes from `candidates_data`. The candidates were saved with `"question": question` where `question = sample.get("prompt_text", sample["question"])` (line 157). So the key should match.

However, `q_to_image` is built from `train_data_with_images` which was shuffled (line 128), and some questions may not be unique across datasets (e.g., "What color is the car?" could appear in both VQAv2 and TextVQA with different images). In that case, only the **last** image wins in the dict, and some SFT samples will train with the **wrong image**.

**Impact**: Silent correctness issue -- wrong image paired with answer. Frequency depends on question duplication across datasets.

**Fix direction**: Use a compound key (source + question) or (sample_id) instead of question text alone.

---

## Bug 6 (MEDIUM): Model re-loading in "both" phase loses generation model's processor

**Location**: `main()`, lines 481-494

When phase="both", after generation the model is deleted (line 482) but `processor` and `model_info` from the first `load_model` call are still in scope. Then a new model is loaded at line 494 with `for_training=True`. The new `model_info` correctly comes from the new `load_model` call. This is actually fine -- no bug here. The old references go out of scope.

**Status**: No bug. The re-loading is correct.

---

## Bug 7 (LOW): `candidates_data` is mutated by `random.shuffle` in SFT

**Location**: `phase_sft()`, line 273

`random.shuffle(candidates_data)` mutates the list in-place. This means the order stored in `args.candidates_file` (from Phase 1) no longer matches what was trained. Not a bug per se, but makes debugging harder if you want to correlate training order with the saved file.

---

## Bug 8 (LOW): No `add_generation_prompt=False` on full conversation template

**Location**: `phase_sft()`, line 292

`processor.apply_chat_template(messages, tokenize=False)` is called without `add_generation_prompt=False`. When the messages list ends with an assistant role, some chat templates will still append a generation prompt marker after the assistant content. This could add extra tokens at the end that are trained on but serve no purpose. The default behavior varies by template -- for Qwen3-VL it likely does the right thing (no extra prompt after assistant), but this is template-dependent and should be explicit.

---

## Summary

| # | Severity | Category | Description |
|---|----------|----------|-------------|
| 1 | Low | Maintenance | Two separate `build_inputs` code paths (local vs model_registry) |
| 2 | High | Performance | IIG does 2N forward passes per sample; not batched |
| 3 | High | Correctness | Prompt length masking via separate tokenization is unreliable |
| 4 | Medium | Correctness | Gradient checkpointing enabled on generation model after baseline eval |
| 5 | Medium | Correctness | Question-text dict key for image mapping can collide |
| 6 | -- | OK | Model re-loading is correct |
| 7 | Low | Debug | In-place shuffle of candidates list |
| 8 | Low | Robustness | Missing explicit `add_generation_prompt=False` on full conversation |

**Recommended fix priority**: Bug 4 > Bug 3 > Bug 5 > Bug 2 > Bug 8 > Bug 1 > Bug 7

Bug 4 can cause generation to silently break. Bug 3 can cause SFT to train on wrong tokens. Bug 5 causes wrong images. Bug 2 is a major perf issue but not a correctness problem.
