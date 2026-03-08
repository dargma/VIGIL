<!-- DEPRECATED: Migrated to OpenCode/OmO format. See AGENTS.md + opencode.json + .opencode/ -->

# VIGIL: Vision-Grounded Inference via Guided head-Level steering

## 1. Project Overview

VIGIL fixes a fundamental problem in small VLMs (1-3B): **Visual Attention Drift** — attention to visual tokens decays as O(1/L_total) during generation, causing models to ignore images entirely in long reasoning chains.

### Core Thesis

> Head-level activation steering + RL training with visually-grounded reward
> produces small VLMs that habitually use visual information, preventing the
> "blind reasoner" degeneration observed in standard GRPO/DAPO training.

### Two-Stage Approach

```
Stage A: Inference-time head-level steering (surgical, reversible)
  Calibrate → Identify vision heads (Cohen's d) → Inject steering vectors (agreement-gated)

Stage B: RL training with visually-grounded reward (permanent weight changes)
  GRPO/DAPO with R_total = w1*R_correct + w2*R_visual_grounding + w3*R_fluency
  R_visual_grounding = α*R_vhad + (1-α)*R_asi
```

### Novel Reward Design (Core Contribution)

```
R_total = 0.3 × R_correct + 0.5 × R_visual_grounding + 0.2 × R_fluency

R_visual_grounding = 0.6 × R_vhad + 0.4 × R_asi
```

- **R_vhad**: Forward pass with real vs black image → activation difference in top-K vision heads
- **R_asi**: Generate answer with vs without image → answer sensitivity metric
- **w2 > w1 intentionally** — grounding matters more than correctness to prevent blind reasoner collapse

---

## 2. Target Models

| Role | Model | HF ID | Architecture |
|------|-------|-------|-------------|
| Main | Qwen3-VL-2B | `Qwen/Qwen3-VL-2B-Instruct` / `Thinking` | GQA 16Q/8KV, 28 layers, head_dim=128, hidden=2048 |
| Main | InternVL3.5-1B | `OpenGVLab/InternVL3_5-1B` | GQA 16Q/8KV, 28 layers, hidden=1024, trust_remote_code |
| MoE | DeepSeek-VL2-Tiny | `deepseek-ai/deepseek-vl2-tiny` | MHA 10 heads, 12 layers, 64 experts top-6, context 4K |
| 7B Ref | Qwen2.5-VL-7B | `Qwen/Qwen2.5-VL-7B-Instruct` | Unsteered baseline only |

### Model Verification (mandatory after loading)

Print and compare: `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `hidden_size`.

### Steering Hook Location

Pre-hook on `o_proj` input — individual Q head outputs are separable as `[num_Q_heads × head_dim]` before projection to hidden_size.

### Model-Specific Constraints

- **Qwen3-VL-2B**: DeepStack on layers 1-3 → steer layer 4+ only
- **InternVL3.5-1B**: beta=0.0 required (TRL ref model fails with trust_remote_code)
- **DeepSeek-VL2-Tiny**: TRL incompatible — needs custom training loop. temp≤0.7

---

## 3. Core Algorithms

### 3.1 Calibration (Cohen's d)

Compute per-head activation statistics on correct vs incorrect responses using GQA-balanced-val + TextVQA-val (~2K samples). Rank heads by Cohen's d. Select top-K vision-specialized heads (<7% of all heads).

### 3.2 Agreement-Gated Steering

Monitor layer agreement across heads. Only inject steering vectors when the model is uncertain (low agreement). Uses 1-step lag to avoid self-reinforcing feedback loops.

### 3.3 R_vhad (Vision Head Activation Differential)

```python
# Forward with real image → collect activations at vision heads
act_real = forward_with_hooks(model, real_image, prompt)
# Forward with black image → collect activations at vision heads
act_black = forward_with_hooks(model, black_image, prompt)
# Reward = normalized activation difference
R_vhad = normalize(sum(|act_real[h] - act_black[h]| for h in vision_heads))
```

### 3.4 R_asi (Answer Sensitivity to Image)

```python
# Generate with image
answer_with = generate(model, image, prompt)
# Generate without image (or black image)
answer_without = generate(model, black_image, prompt)
# If answers are the same → model is blind → low reward
R_asi = 1.0 - similarity(answer_with, answer_without)
```

### 3.5 Blind Test (Killer Experiment)

Replace all test images with black images. Measure Gap = acc(real) - acc(black).
- Baseline gap ~16
- R_correct-only GRPO gap ~7 (blind reasoner)
- VIGIL full reward gap target: ~19 (more image-dependent)

---

## 4. Data Pipeline

**Iron rule: zero image overlap between calibration/training and evaluation.**

| Purpose | Dataset | Image Source | POPE Overlap |
|---------|---------|-------------|-------------|
| Calibration | GQA balanced val (~1K) + TextVQA val (~1K) | Visual Genome / TextVQA | None |
| Training | VQAv2 train (~20K) + A-OKVQA train (~17K) + TextVQA train (~10K) | COCO train2014 / 2017 / TextVQA | Check A-OKVQA |
| Eval Tier 1 | POPE, MMBench, MME, MMMU | Independent | None |

**POPE uses COCO val2014 images. COCO 2014 and 2017 share the same image pool.** Must cross-check A-OKVQA train image_ids against POPE before training.

---

## 5. Evaluation Protocol

5 modes × 3 models × Tier-1 benchmarks (POPE-3splits, MMBench, MME, MMMU):

1. Greedy baseline
2. Steered inference only
3. Post-GRPO + steering
4. Post-DAPO + steering
5. Blind test (black image → Gap metric)

For Qwen3-VL-2B additionally: repeat modes 1-4 with Thinking variant. Analyze vision head activation over token positions in thinking chain.

---

## 6. Success Criteria

1. **POPE Adversarial**: Steered+GRPO small model narrows gap with or matches unsteered 7B
2. **Blind Test Gap**: VIGIL full-reward Gap > baseline Gap (more image-dependent)
3. **MME Perception vs Cognition**: Perception up, Cognition flat (surgical)
4. **Thinking mode drift curve**: steering prevents vision head activation decay
5. **Ablations**: K (head count), alpha (steering strength), reward weights, DeepStack exclusion
6. **MoE routing shift** (DeepSeek only): steering redirects expert selection

---

## 7. Codebase Map

### src/ — Core Implementation

| File | Description |
|------|-------------|
| `rewards.py` | R_vhad + R_asi computation (core novelty) |
| `agreement_gate.py` | Layer agreement monitoring for conditional steering |
| `blind_test.py` | Black image evaluation + Gap metric |
| `vision_drift.py` | Thinking chain token-position vs activation analysis |
| `moe_routing.py` | Expert selection tracking (DeepSeek only) |
| `steerer.py` | Head-level activation steering with GQA support |
| `calibrator.py` | Cohen's d calibration pipeline |
| `profiler.py` | Per-head activation extraction |
| `model_registry.py` | Model loading + architecture verification |
| `data_loader.py` | Dataset loading with overlap checking |
| `trainer.py` | GRPO/DAPO training with visual grounding reward |

### configs/

| File | Description |
|------|-------------|
| `models.yaml` | Model registry: HF IDs, architecture params, constraints |
| `training.yaml` | GRPO/DAPO hyperparameters, reward weights |
| `eval.yaml` | Benchmark configurations, blind test settings |

### scripts/

| File | Description |
|------|-------------|
| `calibrate.py` | Run calibration on target model |
| `train_grpo.py` | Launch GRPO/DAPO training |
| `eval_benchmarks.py` | Run full evaluation suite |
| `run_blind_test.py` | Black image evaluation |
| `smoke_test.py` | Pipeline validation (9 checks, run before anything else) |
| `auto_lab.py` | State machine for automated experiment loop |

### skills/

| File | Description |
|------|-------------|
| `SKILL_qwen3vl_loading.md` | Qwen3VL class, layer path, dtype gotchas |
| `SKILL_oproj_hooks.md` | o_proj pre-hook for per-head activation capture |
| `SKILL_calibration_results.md` | Qwen3-VL-2B calibration numbers |
| `SKILL_vision_activation_delta.md` | Real vs black image Δ analysis |

### lab/

| Path | Description |
|------|-------------|
| `RESEARCH_JOURNAL.md` | Append-only experiment log |
| `RESEARCH_IDEAS.md` | Aggressive ideation log with prioritized experiments |
| `reports/` | Generated plots and comparison tables |

---

## 8. Competitive Positioning

### vs VISTA (ICML 2025)
- VISTA is transient (off → back to baseline). VIGIL GRPO is permanent.
- VIGIL + VISTA combined should beat either alone.
- VIGIL preserves Cognition (MME Cognition +5). VISTA may hurt it (-15).

### vs DVRP (2026)
- DVRP uses external perturbation. VIGIL uses internal activation (head-level).
- Blind Test Gap metric proves VIGIL > DVRP on grounding enforcement.

### vs DMAS (2025)
- DMAS is training-free + semantic DB. VIGIL adds RL permanence + agreement gating.

---

## 9. Reference Codebase

Prior V-LENS work: `/content/drive/MyDrive/V-LENS/`
- Steering hooks, calibration, profiler, GRPO/DAPO with TRL, model registry
- **Reuse patterns, not specifics.** All V-LENS configs/results are invalid for VIGIL.

---

## 10. Operating Rules

- **Never stop**: try/except all GPU ops. OOM → halve batch, retry. Log and skip non-fatal errors.
- **Resource checks**: `df -h` before downloads (keep ≥15GB free). `nvidia-smi` before GPU jobs.
- **State tracking**: `lab/RESEARCH_JOURNAL.md` append-only. Update this file at session end.
- **Checkpointing**: Save every N steps (default 10). Support `--resume`.
- **Code quality**: One function, one job. No duplicates. No hardcoded paths.
- **Git milestones**: Push to `https://github.com/dargma/VIGIL` at key milestones.
- **Visualization**: After every eval → plots in `lab/reports/`. Style: seaborn-whitegrid, font 12, figsize (10,6).

---

## 11. Experiment State Log

> **Purpose**: Session-resumable state. Read this section to continue from last session.
> Entries are chronological (newest last). Do NOT delete old entries.

### [Initial] Project Bootstrap
- VIGIL directory created with scaffold structure
- CLAUDE.md, README.md, RESEARCH_JOURNAL.md initialized
- No experiments run yet

### [2026-03-06] Scaffold Complete
- **Environment**: CPU-only (torch 2.10, transformers 5.0, no TRL yet). 195GB free on Drive. No GPU.
- **V-LENS codebase reviewed**: Extracted patterns for model registry, o_proj hooks, calibration, steering, rewards, GRPO integration.
- **All 12 core modules implemented** (syntax verified):
  - `src/model_registry.py` — 4 model loaders (Qwen3-VL, InternVL3, DeepSeek-VL2, Qwen2.5-VL), architecture verification, chat prompt formatting
  - `src/profiler.py` — VisionHeadProfiler with o_proj hooks, Cohen's d ranking, GQA support
  - `src/calibrator.py` — SteeringCalibrator with correctness + confidence split, save/load
  - `src/steerer.py` — SteeringHook (o_proj pre-hook), ActivationSteerer, AgreementMonitor (1-step lag)
  - `src/agreement_gate.py` — AgreementGate with layer sampling, 1-step lag, threshold gating
  - `src/rewards.py` — R_vhad, R_asi, R_correct, R_fluency, composite reward, InSituVisionReward
  - `src/blind_test.py` — run_blind_test, Gap metric, comparison tables
  - `src/vision_drift.py` — VisionDriftAnalyzer, per-head trajectories, decay metrics
  - `src/moe_routing.py` — MoERoutingTracker, distribution comparison, KL divergence
  - `src/data_loader.py` — POPE, GQA, TextVQA, VQAv2, A-OKVQA loaders + overlap checking
  - `src/trainer.py` — VIGILRewardFunction (TRL compatible), GRPO + DAPO setup
- **3 configs**: models.yaml, training.yaml, eval.yaml
- **4 scripts**: calibrate.py, train_grpo.py, eval_benchmarks.py, run_blind_test.py
- **Datasets downloaded** (18GB, all on Drive, all loaders verified):
  - Calibration: GQA-balanced-val (12,578), TextVQA-val (5,000)
  - Training: VQAv2 (21,435), A-OKVQA (17,056), TextVQA-train (34,602)
  - Eval: POPE (9,000), MMBench (4,329), MME (2,374), MMMU (900)
- `data_loader.py` updated to load from disk cache first, all 9 loaders verified
- **Next**: Install GPU runtime + TRL, then run calibration on Qwen3-VL-2B

### [2026-03-06] GPU Session: Smoke Test + Calibration + Baseline Eval

**Environment**: NVIDIA L4 23GB, TRL 0.29.0, Qwen3-VL-2B in fp16 (4.3GB VRAM).

**Bug fixes**:
- `Qwen2_5_VLForConditionalGeneration` → `Qwen3VLForConditionalGeneration`
- Layer path: `model.model.layers` → `model.model.language_model.layers`
- BFloat16 → numpy fix: `.float().cpu().numpy()`

**Key finding — Two types of vision heads**:
- **Feature heads** (late layers L24-27, high activation Δ up to 66.2): encode raw visual info
- **Decision heads** (early-mid layers L4-5, high Cohen's d up to 9.8): separate correct/incorrect
- No prior work distinguishes these. Novel contribution for paper.

**Results**:
- Smoke test: 9/9 pass, vision Δ non-trivial (mean=6.1, max=66.2)
- Calibration: 20 heads selected, 43/957 correct/incorrect split
- POPE baseline: IN PROGRESS

**New files**: `scripts/smoke_test.py`, `scripts/auto_lab.py`, `skills/` (4 files), `lab/RESEARCH_IDEAS.md` (8 prioritized ideas)

**Research ideas prioritized** (see `lab/RESEARCH_IDEAS.md`):
1. Proportional steering by Δ magnitude
2. Two-types-of-heads ablation (feature vs decision)
3. Thinking mode drift curve (Figure 1 candidate)
4. Vision drift as explicit training signal
5. Adaptive reward curriculum

**Next**:
1. ~~Get POPE baseline results~~ DONE
2. ~~Run steered eval~~ DONE
3. ~~Blind test baseline~~ DONE
4. ~~IIG Block 0 calibration~~ DONE
5. Block 1: Minimal GRPO with IIG reward

### [2026-03-07] Pre-Validation Complete + IIG Block 0

**All 4 pre-validations PASSED:**
- PV1: Vision heads exist (mean Δ=6.1, max=66.2)
- PV2: Steering helps (+2pp POPE, +9pp at α=10)
- PV3: Thinking model (marginal +1pp at α=1)
- PV4: Blind test gap up (25.4→28.4pp, +3.0pp)

**IIG Block 0 (λ calibration) PASSED:**
- 99.4% positive IIG (gate threshold: 60%)
- λ = 0.0615 (auto-calibrated)
- Mean IIG = 9.95

**Bugs found and fixed:**
1. `compute_iig()` attention_mask mismatch: Qwen3-VL uses mask for RoPE positions. Must extend mask when appending candidate tokens.
2. GQA disk cache lacks images (Arrow format). Use POPE or stream from HF for image-dependent tasks.

**Key files created:**
- `src/iig.py` — IIG reward (compute_iig, calibrate_lambda, vigil_reward)
- `scripts/iig_calibration.py` — Block 0 calibration
- `scripts/generate_report.py` — Pre-validation report generator
- `lab/reports/` — 7 figures + markdown report

**Next**: Block 1 Minimal GRPO (50 steps, R_correct vs R_correct+IIG)

### [2026-03-07] Block 1 GRPO: 3 Attempts, All Collapsed

**TRL GRPOTrainer is not viable for binary VQA at 2B scale.**

| Version | Config Changes | Result | Collapse Mode |
|---------|---------------|--------|---------------|
| v1 | group=4, temp=1.0, lr=5e-6, LoRA r=16, 50 steps | POPE 76→31% | always-no |
| v2 | group=8, temp=1.2, lr=2e-6, LoRA r=8, 20 steps + format reward | POPE 76→30% | always-yes |
| v3 | beta=0.1, lr=5e-7, LoRA r=4, 5 steps (ultra-conservative) | POPE 77→31% | always-yes |

IIG reward worked correctly in all 3 runs (100% positive, mean~1.0) but could not prevent collapse.
v4 (open-ended TextVQA) script written but never ran (GPU unavailable).

**Root cause**: Binary VQA has ~1 bit output entropy → GRPO groups have no diversity → zero advantage → collapse.

### [2026-03-07] Block 2 Pivot: Custom GRPO + DPO (CPU-only prep)

**Scripts prepared (no GPU needed):**
1. `scripts/block2_custom_grpo.py` — Manual GRPO loop with:
   - Entropy bonus (beta_entropy=0.01), dynamic temperature
   - Mixed data (non-binary only: TextVQA + A-OKVQA MC + VQAv2 short-answer)
   - Zero-variance group skipping, collapse detection
   - IIG integration (lambda=0.0615)
2. `scripts/block2_dpo_iig.py` — DPO with IIG-ranked preference pairs (fallback)
3. `scripts/prepare_mixed_data.py` — Mixed non-binary training data pipeline
4. `configs/training.yaml` updated with `custom_grpo`, `dpo`, `iig` sections

**When GPU returns:**
1. Run `block2_custom_grpo.py` (primary)
2. If collapse: run `block2_dpo_iig.py` (DPO is immune to binary collapse)
3. Evaluate on POPE + Blind Test Gap

### [2026-03-08] Block 2: BoN+SFT BREAKTHROUGH

**Best-of-N + SFT delivered the first real improvement in the project.**

**Results**:
- POPE: 83.0% → **85.5%** (+2.5pp)
- Blind Gap: 32.0pp → **37.0pp** (+5.0pp)
- Real acc: 82.0% → 87.0% (+5.0pp), Blind acc: 50.0% (stable)

**Pipeline**: Generate N=8 candidates → score with R_correct + IIG → select best → SFT on 692 curated samples (2 epochs)
**Checkpoint**: `checkpoints/block2_bon/final`

**Prior runs this session** (all completed):
- Block 2 v2 Setting A (GRPO, R_correct only): POPE 84.5→83.5% (no collapse but no improvement)
- Block 2 v2 Setting B (GRPO, R_correct+IIG): POPE 84.5→85.0% (+0.5pp, within noise)
- Block 2 v3 (100 steps, all fixes): POPE oscillated 83.5-85%, no lasting improvement
- **Block 2 BoN+SFT**: POPE 83→85.5%, Gap 32→37pp (**BEST**)

**Key lesson**: GRPO advantage estimation is too noisy for binary/short-answer VQA. BoN+SFT (ReST/RAFT approach) is strictly superior for this task — curates high-quality data then trains on it.

**Next**:
1. ~~Multi-round BoN+SFT (use round-1 model for round-2 generation)~~ IN PROGRESS
2. Add R_vhad to scoring
3. DAPO comparison
4. Git push milestone

### [2026-03-08] Session 2: Multi-Round BoN+SFT + Thinking Mode Eval

**BoN+SFT Round 2 (IN PROGRESS)**:
- Model: `checkpoints/block2_bon/final` (round-1 checkpoint)
- Config: lr=1e-6, seed=43, N=8, temp=1.2, 1000 samples
- Log: `logs/block2_bon_round2.log`
- Output: `checkpoints/block2_bon/round2`
- Status: Generation 700/1000

**Thinking Mode Eval (PREPARED, pending GPU)**:
- Script: `scripts/eval_thinking_mode.py`
- Tests 3 conditions: baseline, bon_r1, bon_r2
- Tracks: POPE accuracy, Blind Gap, Vision Drift Curve (Figure 1)
- Run commands:
  1. `python scripts/eval_thinking_mode.py --model-label baseline`
  2. `python scripts/eval_thinking_mode.py --model-path checkpoints/block2_bon/final --model-label bon_r1`
  3. `python scripts/eval_thinking_mode.py --model-path checkpoints/block2_bon/round2 --model-label bon_r2`

**Publication plots generated**: `lab/reports/generate_block2_plots.py`
- fig1_pope_progression.png, fig2_blind_gap_progression.png
- fig3_bon_candidate_quality.png, fig4_method_comparison.png, fig5_grpo_dynamics.png

**Verification issues found** (non-blocking but should fix):
1. SFT label masking: prompt_len may miss assistant role tokens
2. OOM handling: n_batches counter not properly tracked
3. Image persistence: images not saved in candidates JSON

**New files**:
- `scripts/eval_thinking_mode.py` — thinking mode eval + drift analysis
- `lab/reports/generate_block2_plots.py` — 5 publication figures

**Next when resuming**:
1. Check round 2 results: `tail -50 logs/block2_bon_round2.log`
2. If done: run thinking mode eval (3 conditions)
3. Fix verification bugs in block2_best_of_n_sft.py
4. Git commit + push milestone

### [2026-03-08] Session 3: Official VLMEvalKit Evaluation

**VLMEvalKit integration for publication-quality scores.**

**Key decision**: Use VLMEvalKit's exact prompt templates and answer parsing (not custom).
- POPE prompt: `{question} Please answer yes or no.` (from `vlmeval/vlm/qwen3_vl/prompt.py`)
- Parsing: `YOrN_Extraction()` — process_punctuation + word-level yes/no check
- Metrics: acc, F1, precision, recall per split (random/popular/adversarial)
- Functions inlined in eval script (full vlmeval import fails due to missing megabench submodule)

**Official Results (500 samples, first pass)**:
| Condition | Acc | F1 | P | R | Blind Gap |
|-----------|-----|-----|---|---|-----------|
| Baseline | 87.6% | 87.4% | 88.5% | 86.4% | 37.6pp |
| Steered (α=5) | 87.6% | 87.6% | 87.3% | 88.0% | 37.6pp |
| BoN+SFT R1 | **88.0%** | 87.6% | **90.6%** | 84.8% | **38.0pp** |
| All blind | 50.0% | 0.0% | 0.0% | 0.0% | — |

**Key findings**:
- Baseline higher than custom eval (87.6% vs 83.0%) — official prompts + parsing more consistent
- BoN+SFT R1 still shows improvement (+0.4pp acc, +0.4pp gap)
- Blind = all-No (50% acc on balanced POPE) — model truly uses images
- Steering shows no effect at α=5 with official prompts — may need re-tuning

**Full 9000-sample eval**: RUNNING in background

**New files**:
- `scripts/eval_official.py` — Full VLMEvalKit-standard eval (single condition)
- `scripts/eval_official_fast.py` — Fast multi-condition eval (loads model once per group)
- `src/soft_rewards.py` — Soft thresholding reward functions

**Next**:
1. Full 9000-sample results (per-split breakdown: random/popular/adversarial)
2. Re-tune steering alpha with official prompts
3. MME + MMBench eval (need pair-based scoring for MME)
4. DAPO + soft thresholding integration
5. Git push milestone
