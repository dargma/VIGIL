# Multi-Model Expansion Plan: InternVL3.5-1B + DeepSeek-VL2-Tiny

## Date: 2026-03-17

## Goal
Replicate the Exp10 (sharp sigmoid Head-LSR GRPO) result on two additional architectures:
- **InternVL3.5-1B**: GQA architecture (like Qwen3), smaller hidden size
- **DeepSeek-VL2-Tiny**: MHA + MoE architecture (fundamentally different)

This validates that Head-LSR is architecture-agnostic, not Qwen3-specific.

## Model Specs

| Property | Qwen3-VL-2B (done) | InternVL3.5-1B | DeepSeek-VL2-Tiny |
|----------|-------------------|----------------|-------------------|
| HF ID | `Qwen/Qwen3-VL-2B-Thinking` | `OpenGVLab/InternVL3_5-1B` | `deepseek-ai/deepseek-vl2-tiny` |
| Layers | 28 | 28 | 12 |
| Q Heads | 16 | 16 | 10 |
| KV Heads | 8 (GQA) | 8 (GQA) | 10 (MHA) |
| Head Dim | 128 | 128 | 256 |
| Hidden | 2048 | 1024 | 2560 |
| Total Heads | 448 | 448 | 120 |
| MoE | No | No | Yes (64 experts, top-6) |
| Layer Path | `model.language_model.layers` | `language_model.model.layers` | `model.layers` |
| Input API | `qwen_vl_utils` + processor | `model.chat()` + TRANSFORM | custom preprocessor |
| Thinking | Yes (`enable_thinking`) | No | No |
| Trust Remote | No | Yes | Yes |

## Previous Results (from MULTIMODEL_REPORT.md)

### InternVL3.5-1B
- POPE baseline: **78.2%**, Gap: **28.2pp**
- BoN+SFT R2: **83.4%** (+5.2pp), Gap: **33.4pp** (+5.2pp)
- Max Cohen's d: **0.774** (vs Qwen3's 9.8 — 12× lower)
- Steering alpha=1: +0.2pp only
- Calibration: `checkpoints/calibration/internvl3_5_1b/`
- Checkpoints: `checkpoints/internvl_bon_sft/round2/`

### DeepSeek-VL2-Tiny
- No baseline results yet
- MoE routing tracker: `src/moe_routing.py`
- TRL incompatible → custom training loop (same as our phase6 script)

## Architecture Adaptation Required

### 1. Model Loading (`load_model()`)
Current: Hardcoded `Qwen3VLForConditionalGeneration`
Change: Use `model_registry.load_model(key)` which returns standardized dict

### 2. Input Preparation (`prepare_inputs()`)
Current: `qwen_vl_utils.process_vision_info` + `processor.apply_chat_template(enable_thinking=True)`
Change: Model-specific input prep:
- InternVL: `TRANSFORM(image)` + `tokenizer(query)` + `pixel_values`
- DeepSeek: custom preprocessor from `deepseek_vl2` package

### 3. Hook Installation (`VisionHeadHooks`, `AdaptiveVisionHeadHooks`)
Current: Hardcoded `model.model.language_model.layers`, `num_heads=16, head_dim=128`
Change: Parameterize via model spec:
- InternVL: `model.language_model.model.layers`, 16 heads, 128 dim
- DeepSeek: `model.model.layers`, 10 heads, 256 dim, 12 layers

### 4. Generation
Current: `model.generate(**inputs, ...)`
Change:
- InternVL: `model.chat(tokenizer, pixel_values, prompt, generation_config=...)`
- DeepSeek: `model.generate()` but with temp ≤ 0.7

### 5. Answer Extraction
Current: Parse `<think>...</think>` tags + yes/no extraction
Change:
- InternVL/DeepSeek: No thinking tags, direct answer extraction

## Implementation Plan

### Phase A: Refactor phase6 script for multi-model (no new experiments yet)

1. Add `--model-key` argument (default: `qwen3_vl_2b_thinking`)
2. Create model-specific `prepare_inputs()` dispatcher
3. Create model-specific `generate_candidates()` dispatcher
4. Parameterize hook classes with model spec
5. Add model-specific eval functions (or adapt existing)

### Phase B: Smoke test each model

1. Load model → verify architecture
2. Run 5-sample POPE eval → verify answer parsing
3. Install hooks → verify activation capture
4. Generate 1 sample with 6 candidates → verify generation

### Phase C: Run Exp10 on each model

Same config as Qwen3 Exp10:
- 50 steps, 2 samples/step, seed=42
- Sharp sigmoid (T/3), GDPO, VPPO, gated head-LSR
- Eval at steps 10, 25, 50

**InternVL command**:
```bash
python -u scripts/phase6_head_mask_grpo.py \
    --model-key internvl3_5_1b \
    --steps 50 --alpha 0.5 --gdpo --vppo-mask --gated-head-lsr \
    --soft-weighted-heads --soft-temperature auto --soft-temperature-scale 0.33 \
    --lr 2e-6 --group-size 6 --temperature 1.3 \
    --max-new-tokens 512 --train-samples 2000 --samples-per-step 2 \
    --eval-steps 10,25,50 --eval-pope-samples 60 --eval-blind-samples 50 \
    --seed 42 --output-dir checkpoints/internvl/exp10_scaled
```

**DeepSeek command**:
```bash
python -u scripts/phase6_head_mask_grpo.py \
    --model-key deepseek_vl2_tiny \
    --steps 50 --alpha 0.5 --gdpo --vppo-mask --gated-head-lsr \
    --soft-weighted-heads --soft-temperature auto --soft-temperature-scale 0.33 \
    --lr 2e-6 --group-size 6 --temperature 0.7 \
    --max-new-tokens 512 --train-samples 2000 --samples-per-step 2 \
    --eval-steps 10,25,50 --eval-pope-samples 60 --eval-blind-samples 50 \
    --seed 42 --output-dir checkpoints/deepseek/exp10_scaled
```

## Risks

1. **InternVL Cohen's d is 12× lower** → Head-LSR signal may be too weak
   - Mitigation: higher alpha, or use activation delta instead of Cohen's d for head selection
2. **DeepSeek MoE** → Expert routing may dilute per-head signal
   - Mitigation: Track expert routing shift as additional metric
3. **Memory**: DeepSeek-VL2-Tiny may be larger than expected with MoE
   - Mitigation: Reduce group_size to 4

## Success Criteria

1. Both models complete 50 steps without crash
2. POPE improves ≥ 1.0pp from baseline on at least one model
3. Blind Gap improves ≥ 1.0pp from baseline on at least one model
4. Results are comparable or better than BoN+SFT (existing best for InternVL)
