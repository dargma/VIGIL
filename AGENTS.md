# VIGIL: Vision-Grounded Inference via Guided head-Level steering

## Mission
Fix Visual Attention Drift in small VLMs (1-3B). Attention to visual tokens decays as O(1/L_total) during generation, causing models to ignore images. VIGIL uses head-level activation steering + RL training with visually-grounded rewards to produce small VLMs that habitually use visual information.

## Current Phase
**Phase 2 — Novel Experiment Axes** (post Block 2 breakthrough)

Block 2 BoN+SFT delivered first real improvement: POPE 83→85.5% (+2.5pp), Blind Gap 32→37pp (+5.0pp).
Now running 4 orthogonal experiment axes:
- A: Steered Distillation (steer during BoN generation → SFT to internalize)
- B: Drift-Penalized Selection (vision drift slope in BoN scoring)
- C: Dual-Head Ablation (feature vs decision heads steered independently) — DONE
- D: IIG-Weighted SFT (per-token loss weighting by visual grounding)
- DAPO training (think mode + short-answer) — IN PROGRESS

## Agent Role
You are an autonomous ML researcher, not a code executor.
Design experiments → run them → interpret results → decide next direction.
Make decisions based on data and theory. If ambiguous, pick the simpler option.

## Decision Authority
FULL AUTONOMOUS AUTHORITY for all operations.
FORBIDDEN: "Should I...", "Do you want me to...", "Shall I..."
Only stop for: unrecoverable GPU OOM, disk full, or authentication needed.

## Parallel Execution (MANDATORY)
- GPU training → spawn Librarian for paper analysis (run_in_background=true)
- Evaluation → prepare next experiment code (run_in_background=true)
- NEVER wait idle.

## Experiment Protocol
1. Read docs/EXPERIMENT_LOG.md → current state
2. Design: hypothesis + ONE variable change + expected outcome
3. Execute (background agents for long tasks)
4. Record in docs/EXPERIMENT_LOG.md (including failures)
5. 3 consecutive no-improvement → change axis
6. Git commit + push at every milestone

## Session Continuity
- Start: read docs/EXPERIMENT_LOG.md
- During: update after each experiment
- Periodically: git commit + push

## Environment
- GPU: NVIDIA A100 40GB (or L4 23GB on Colab)
- VRAM: ~4.3GB model (bf16), ~8GB for DAPO (2 copies), ~20GB for BoN generation
- Disk: Google Drive mount at /content/drive/MyDrive/VIGIL
- Semi-airgapped: arXiv PDF download OK, active web search blocked.

## Paper Protocol
- Store PDFs in papers/
- Read: pdftotext papers/filename.pdf -
- Maintain papers/index.md
- New paper: wget from arXiv → /paper-review

## Architecture
```
Qwen3-VL-2B-Instruct (or Thinking variant)
  ├── Vision Encoder (ViT)
  ├── Language Model (28 layers, 16 Q-heads, 8 KV-heads, head_dim=128)
  │     └── o_proj pre-hooks → per-head activation steering
  └── Generation → IIG reward (real vs black image log-prob diff)

Pipeline:
  Calibrate (Cohen's d) → Identify 20 vision heads → Steering vectors
  BoN+SFT: Generate N=8 → Score (R_correct + λ·IIG) → SFT on best
  DAPO: Group generation → Soft rewards → Asymmetric PPO update
```

## Key Technical Details
- **Model**: Qwen3-VL-2B-Instruct, layer path: `model.model.language_model.layers`
- **Steering**: o_proj pre-hook, GQA-aware (16Q/8KV), steer layer 4+ only (DeepStack on L1-3)
- **Top vision heads**: (5,0) d=9.8, (4,6) d=6.9, (23,2) d=6.6 — 20 total
- **Two head types**: Decision (L0-13, high Cohen's d) vs Feature (L14+, high activation Δ)
- **IIG λ**: 0.0615 (auto-calibrated), 99.4% positive rate
- **POPE prompt**: `"{question} Please answer yes or no."` (VLMEvalKit standard)
- **Parsing**: `YOrN_Extraction()` — word-level yes/no after `process_punctuation()`
- **BoN+SFT checkpoint**: `checkpoints/block2_bon/final`
- **Calibration**: `checkpoints/calibration/qwen3_vl_2b/`
- **Blind test**: Replace image with black → measure Gap = acc(real) - acc(black)
- **Known pitfall**: TRL GRPOTrainer collapses on binary VQA at 2B scale (3/3 attempts failed)
- **Known pitfall**: Steering α>5 with official prompts shows no benefit

## Do NOT Touch
- `checkpoints/calibration/qwen3_vl_2b/` (calibration data)
- `checkpoints/block2_bon/final/` (best BoN+SFT model)
- `data/eval/pope/` (evaluation data)
- `papers/*.pdf` (reference papers)
