# VIGIL Research Journal

> Append-only experiment log. Newest entries at the bottom. Never delete old entries.

---

## 2026-03-06 — Project Bootstrap

**Status**: Project initialized.

**What was done**:
- Created project scaffold: `src/`, `configs/`, `scripts/`, `lab/`, `data/`, `logs/`
- Wrote `CLAUDE.md` (project instructions), `README.md` (overview), this journal
- Defined target models, reward design, evaluation protocol

**Key decisions**:
- Reward weights: w1=0.3 (correct), w2=0.5 (visual grounding), w3=0.2 (fluency)
- R_vhad weight α=0.6 within visual grounding component
- Agreement-gated steering with 1-step lag
- Qwen3-VL-2B DeepStack: steer layer 4+ only (layers 1-3 excluded)

**Next steps**:
1. ~~Check environment (GPU, disk, packages)~~ DONE
2. ~~Read V-LENS codebase for reusable patterns~~ DONE
3. ~~Implement all core modules~~ DONE
4. Install TRL, get GPU runtime
5. Run first calibration on Qwen3-VL-2B

---

## 2026-03-06 — Scaffold Implementation Complete

**Environment**: CPU-only torch 2.10, transformers 5.0, no TRL. 195GB Drive free.

**V-LENS patterns extracted**:
- Model registry: model_info dict + get_layers_fn lambda closures
- Hook site: `self_attn.o_proj` pre-hook → (batch, seq, num_heads * head_dim)
- Steering: pre-hook modifies input, alpha scaling, agreement gating
- Rewards: composite V2 (accuracy + steering alignment + format + length)
- GRPO: TRL GRPOTrainer with custom reward_fn factory
- GQA reshape: view(batch, num_Q_heads, head_dim) at o_proj input

**12 modules implemented** (all syntax verified):
- model_registry.py, profiler.py, calibrator.py, steerer.py
- agreement_gate.py, rewards.py, blind_test.py, vision_drift.py
- moe_routing.py, data_loader.py, trainer.py, __init__.py

**4 entry scripts**: calibrate.py, train_grpo.py, eval_benchmarks.py, run_blind_test.py

**3 configs**: models.yaml, training.yaml, eval.yaml

**Next steps**:
1. ~~Download datasets~~ DONE (see below)
2. Install GPU runtime + TRL
3. Load Qwen3-VL-2B, verify architecture
4. Run calibration (GQA + TextVQA val, ~2K samples)
5. POPE baseline eval, blind test baseline
6. Steering eval, GRPO training

---

## 2026-03-06 — Datasets Downloaded

**All 9 datasets downloaded to Drive** (18GB total, persistent):

| Dataset | Path | Rows | Purpose |
|---------|------|------|---------|
| GQA balanced val | data/calibration/gqa_balanced_val | 12,578 | Calibration |
| TextVQA val | data/calibration/textvqa_val | 5,000 | Calibration |
| VQAv2 train | data/training/vqav2_train | 21,435 | GRPO/DAPO |
| A-OKVQA train | data/training/aokvqa_train | 17,056 | GRPO/DAPO |
| TextVQA train | data/training/textvqa_train | 34,602 | GRPO/DAPO |
| POPE | data/eval/pope | 9,000 | Eval (3 splits) |
| MMBench | data/eval/mmbench | 4,329 | Eval |
| MME | data/eval/mme | 2,374 | Eval |
| MMMU | data/eval/mmmu | 900 | Eval (30 subjects) |

**Download gotchas fixed**:
- GQA requires config: `lmms-lab/GQA`, `testdev_balanced_instructions`, split=`testdev`
- TextVQA: use `lmms-lab/textvqa` (original `textvqa` uses deprecated scripts)
- VQAv2: use `merve/vqav2-small` (HuggingFaceM4/VQAv2 uses deprecated scripts)
- MMMU: requires per-subject download, 30 subjects concatenated
- MMMU options field: not always JSON-parseable, needs type checking

**data_loader.py updated**: loads from disk first via `load_from_disk()`, all 9 loaders verified

**Next**: GPU runtime + TRL → calibration → eval

---

## 2026-03-06 — GPU Session: Smoke Test + Calibration

**Environment**: NVIDIA L4 23GB, torch 2.x, TRL 0.29.0, transformers 5.x

### Bug Fixes Before Running
1. **Qwen3VL class**: `model_registry.py` used `Qwen2_5_VLForConditionalGeneration` — wrong class. Fixed to `Qwen3VLForConditionalGeneration`.
2. **Layer path**: `model.model.layers` → `model.model.language_model.layers` for Qwen3-VL.
3. **BFloat16 → numpy**: `torch.stack(list).norm().numpy()` crashes on bf16. Fixed with `.float().cpu().numpy()`.
4. **Dataset images missing**: Saved Arrow datasets don't include images (only text metadata). Smoke test uses synthetic image instead.

### Smoke Test Results (9/9 PASS)
- Model loads in fp16: 4.3GB VRAM on L4
- Architecture verified: 28 layers, GQA 16Q/8KV, head_dim=128, hidden=2048
- o_proj hooks work in both forward() and generate()
- **Key finding: Real vs black image activation Δ is strongly non-trivial**
  - mean_Δ = 6.11, max_Δ = 66.22 (layer 26 head 9)
  - Δ increases monotonically with depth: layers 0-3 ~0.3, layers 24-27 ~20+
  - **Conclusion: R_vhad will produce strong, usable reward signal**

### Calibration Results (1000 samples, Cohen's d)
- 43 correct / 957 incorrect (4.3% accuracy — single-token VQA is hard)
- Top-3 vision heads by Cohen's d: L5H0 (d=9.80), L4H6 (d=6.94), L23H2 (d=6.60)
- 20 heads selected for steering
- **Observation**: Cohen's d top heads (L4-5) differ from activation-Δ top heads (L24-27). This is expected — Cohen's d measures correct/incorrect behavioral separation, Δ measures raw image sensitivity. Both are useful for different purposes.

### Decision Log
- **Why fp16 not bf16?** L4 supports both, but bf16 causes numpy conversion issues. fp16 is fine for inference and saves debugging time.
- **Why synthetic image for smoke test?** Saved datasets lack images (Arrow format issue). Real calibration needs HF streaming or re-download with images.
- **Why only 4.3% accuracy?** Single next-token prediction matching first 3 chars of GT. This is an underestimate — model often generates multi-token answers. The correctness split is still valid because we have enough samples in both buckets.

### Created
- `scripts/smoke_test.py` — 9 checks, validates pipeline end-to-end
- `scripts/auto_lab.py` — state machine for automated experiment loop
- `skills/` directory with 4 skill files (qwen3vl loading, o_proj hooks, calibration results, vision activation delta)
- Lightweight reward (`InSituVisionReward`) updated in `rewards.py`

### Next
1. Fix dataset images (need images for baseline eval and GRPO)
2. POPE baseline eval (greedy, no steering)
3. POPE steered eval
4. Blind test baseline (measure initial Gap)
5. GRPO training with VIGIL reward

---

## 2026-03-07 — POPE Baseline Complete + Steered Eval Bug Fix

**Environment**: NVIDIA L4 23GB, same as previous session.

### POPE Baseline Results (200 samples each, greedy, no steering)
| Split | Accuracy | Correct/Total |
|-------|----------|---------------|
| Random | **79.0%** | 158/200 |
| Popular | **76.5%** | 153/200 |
| Adversarial | **78.5%** | 157/200 |

Saved: `lab/results/baseline/eval_qwen3_vl_2b_greedy_baseline_20260307_000338.json`

### Steered Eval — Bug Found and Fixed
- **Bug**: `SteeringHook._pre_hook_fn` had tensor memory aliasing error. `modified[:, -1, :]` returns a view, and `view()` on that creates another view sharing memory. The in-place `+=` then corrupts the original tensor.
- **Error**: "unsupported operation: some elements of the input tensor and the written-to tensor refer to a single memory location"
- **Result**: All 200 samples skipped on all 3 POPE splits (0/0 accuracy).
- **Fix**: Added `.clone()` before `view()` in `steerer.py:42` to break memory aliasing.
- **Status**: Fix committed and pushed. Re-run started but GPU session ending before completion.

### Proportional Steering (from last session)
- `steer_proportional()` method added to `ActivationSteerer` — scales per-head alpha by normalized Cohen's d score.
- Not yet evaluated (steered eval was blocked by the aliasing bug).

### Key Decision
- Steered eval needs to be re-run with the fixed steerer.py in next GPU session.

### Next (for next GPU session)
1. Re-run steered POPE eval (uniform alpha=1.0) with fixed steerer
2. Run steered POPE eval (proportional alpha) — compare uniform vs proportional
3. Blind test baseline (Gap metric)
4. Two-head-types ablation (feature heads vs decision heads)
5. GRPO training with VIGIL reward
