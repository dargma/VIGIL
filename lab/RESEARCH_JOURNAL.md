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
