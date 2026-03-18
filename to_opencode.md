# VIGIL Project Handoff Guide — Complete Reproduction Instructions

This document provides step-by-step instructions to reproduce the entire VIGIL project from scratch, including background, model/data setup, training, and evaluation.

---

## Table of Contents

1. [Project Background](#1-project-background)
2. [Algorithm Overview](#2-algorithm-overview)
3. [Environment Setup](#3-environment-setup)
4. [Model Acquisition](#4-model-acquisition)
5. [Dataset Acquisition & Preprocessing](#5-dataset-acquisition--preprocessing)
6. [Calibration (Pre-requisite)](#6-calibration)
7. [Training — Baseline](#7-training--baseline)
8. [Training — Exp1: Gated Head-LSR](#8-training--exp1-gated-head-lsr)
9. [Training — Exp4: Head Masking KL](#9-training--exp4-head-masking-kl)
10. [Training — Exp5: Learned Head Importance + KL](#10-training--exp5-learned-head-importance--kl)
11. [Training — Exp6: Learned Head Importance + Gated LSR](#11-training--exp6-learned-head-importance--gated-lsr)
12. [Training — Exp7: Dynamic Head Selection (FAILED)](#12-training--exp7-dynamic-head-selection-failed)
13. [Training — Exp8: Per-Rollout Adaptive Head Gate](#13-training--exp8-per-rollout-adaptive-head-gate)
14. [Training — Exp9: Soft-Weighted All-Head LSR](#14-training--exp9-soft-weighted-all-head-lsr)
15. [Training — Exp10: Sharp Sigmoid Head-LSR](#15-training--exp10-sharp-sigmoid-head-lsr)
16. [Training — Exp11: Layer-Aware Head Selection](#16-training--exp11-layer-aware-head-selection)
17. [Training — Exp12: Top-P Adaptive Selection](#17-training--exp12-top-p-adaptive-selection)
18. [Scaled Training (2K Samples)](#18-scaled-training-2k-samples)
19. [Multi-Model Expansion](#19-multi-model-expansion)
20. [Evaluation](#20-evaluation)
21. [Expected Results](#21-expected-results)
22. [Troubleshooting](#22-troubleshooting)
23. [**TOP 3 RECOMMENDED: Reproduce & Expand Exp1, 8, 10**](#23-top-3-recommended-reproduce--expand-exp1-8-10)
24. [Exp13: Target-Calibrated Reward (Negative Result)](#24-exp13-target-calibrated-reward-negative-result)

---

## 1. Project Background

### Problem: Visual Attention Drift

Small VLMs (1-3B parameters) suffer from **Visual Attention Drift** — attention to visual tokens decays as O(1/L_total) during generation. This causes the model to ignore the image entirely in long reasoning chains, becoming a "Blind Reasoner."

**Evidence**: In Qwen3-VL-2B-Thinking:
- Thinking phase: mean vision head activation Δ = 0.44
- Answer phase: mean Δ = 0.23 (~48% decay)
- Longer chains = worse decay (338-token sample: Δ=0.14 vs 83-token: Δ=0.27)

### Solution: VIGIL

VIGIL (Vision-Grounded Inference via Guided head-Level steering) uses head-level analysis and RL training with visually-grounded rewards:

1. **Identify vision heads**: Use Cohen's d to find attention heads specialized for visual processing
2. **Head-level reward signal**: Measure how much vision heads contribute to the model's output
3. **Gated GRPO training**: Switch between correctness reward (when gradient exists) and vision reward (when all candidates agree)

### Key Results (Final, 60-sample POPE eval)

| Experiment | POPE | Gap | TextVQA | Stability | Notes |
|-----------|------|-----|---------|-----------|-------|
| Baseline (HF) | 91.7% | 40.0pp | 72.7% | — | No training |
| **Exp1: Gated Head-LSR** | **95.0%** | **44.0pp** | **74.7%** | 2/3 at 95% | **Best TextVQA, most novel** |
| **Exp8: Adaptive Top-K** | **95.0%** | **44.0pp** | 72.7% | 3/4 at 95% | **Zero TextVQA cost** |
| **Exp10: Sharp Sigmoid** | **95.0%** | **44.0pp** | 70.7% | **4/6 at 95%** | **Most stable** |
| Exp9: Soft All-Heads | 95.0% | 44.0pp | 68.7% | 1/6 at 95% | Too diluted, -4pp TextVQA |
| Exp11: Layer-Aware | 95.0% | 44.0pp | 70.7% | 2/6 at 95% | No advantage over Exp10 |
| Exp13: Target-Calibrated | 95.0% | 44.0pp | — | Degrades by step 10 | Negative result |
| Exp4: Head Masking KL | 90.0% | 38.0pp | 72.7% | — | KL signal too weak |
| Exp5: Learned Imp + KL | 90.0% | 38.0pp | 72.7% | — | FAILED (collapsed) |
| Exp6: Learned Imp + LSR | 91.7% | 40.0pp | 72.7% | — | FAILED (detach bug) |
| Exp7: Dynamic Selection | 91.7% | 40.0pp | 70.7% | — | FAILED (wrong signal) |

### Critical Note on Evaluation Scale

60-sample POPE: each sample = 1.67pp. The 91.7%→95.0% jump = **2 more correct answers**.
1K POPE eval (Phase 2 only): baseline 89.9%, GRPO-LSR 90.4% (+0.5pp).
**1K POPE eval on Exp1/8/10 is the #1 priority — not yet done.**

---

## 2. Algorithm Overview

### 2.1 Vision Head Discovery (Calibration)

Run the model on ~1000 samples, split into correct/incorrect responses. For each of the 448 attention heads (28 layers × 16 heads), compute:

```
Cohen's d = (mean_correct - mean_incorrect) / pooled_std
```

Select top-K heads by |d|. These are "vision heads" — heads that activate differently for correct vs incorrect answers.

**Result**: 12 heads selected, mostly in layers 2-5 ("Decision heads") with one in layer 23 ("Feature head").

### 2.2 Reward Signals

#### R_correct (binary)
```
R_correct = 1.0 if answer matches ground_truth, else 0.0
```

#### R_head_lsr (Head-Level LSR — Exp1)
For each generated token t:
```
head_score(t) = Σ_{h in vision_heads} ||act_real_h(t) - act_black_h(t)||₂
```
Where `act_real` = activation with real image, `act_black` = activation with black image.
Higher score = model uses vision more at token t.

#### R_headKL (Head Masking KL — Exp4)
```
logits_normal = forward(model, input)
logits_masked = forward(model, input, vision_heads_zeroed=True)
R_headKL = mean_over_tokens(KL(softmax(logits_normal) || softmax(logits_masked)))
```
Higher KL = model depends more on vision heads (good).

### 2.3 GDPO (Decoupled Normalization)

```
R_total = w1 × Z(R_correct) + w2 × Z(R_visual)
Z(x) = (x - mean(x)) / (std(x) + eps)
```

Each reward component is normalized independently before combining. Critical for mixing binary rewards (0/1) with continuous rewards (0.01-0.05).

### 2.4 Gating Mechanism

```python
if variance(R_correct across group) > 0:
    # Some candidates correct, some wrong → use correctness gradient
    use R_correct only, uniform token weights
else:
    # All correct or all wrong → zero gradient from R_correct
    use R_visual (head_lsr or headKL), vision-weighted tokens
```

This prevents the vision signal from diluting the correctness signal when it matters.

### 2.5 Learned Head Importance (Exp5/6)

Instead of fixed 12 heads, learn a 28×16 importance map:

```python
importance = nn.Parameter(torch.zeros(28, 16))  # initialized from Cohen's d
# Soft masking for Exp5:
masked_act = act * (1 - sigmoid(importance[layer, head]))
# Weighted LSR for Exp6:
score(t) = Σ sigmoid(imp[l,h]) × ||act_real[l,h,t] - act_black[l,h,t]|| / Σ sigmoid(imp)
```

---

## 3. Environment Setup

### Hardware Requirements

- **GPU**: NVIDIA A100 40GB+ or L4 23GB (L4 is tight but works with gradient checkpointing)
- **RAM**: 16GB+
- **Disk**: ~50GB for datasets + checkpoints

### Software

```bash
# Clone the repo
git clone https://github.com/dargma/VIGIL.git
cd VIGIL

# Install dependencies
pip install torch torchvision torchaudio  # CUDA 12.x
pip install transformers>=4.45.0 accelerate
pip install qwen-vl-utils  # CRITICAL: needed for Qwen3-VL image processing
pip install pillow numpy tqdm matplotlib seaborn
pip install datasets  # HuggingFace datasets

# Optional but recommended
pip install flash-attn --no-build-isolation  # faster attention
```

### Verify Setup

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name()}')"
python -c "from transformers import Qwen3VLForConditionalGeneration; print('OK')"
python scripts/smoke_test.py  # Should pass 9/9 checks
```

---

## 4. Model Acquisition

### Primary Model

```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

model_id = "Qwen/Qwen3-VL-2B-Thinking"  # or "Qwen/Qwen3-VL-2B-Instruct"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)
```

### Architecture Details (verify after loading)

```python
config = model.config
print(f"Layers: {config.num_hidden_layers}")          # 28
print(f"Attention heads: {config.num_attention_heads}")  # 16
print(f"KV heads: {config.num_key_value_heads}")         # 8 (GQA)
print(f"Head dim: {config.head_dim}")                    # 128
print(f"Hidden: {config.hidden_size}")                   # 2048
```

### Critical Notes

- **Layer path**: `model.model.language_model.layers[i].self_attn.o_proj` — NOT `model.model.layers`
- **Thinking mode**: Set `enable_thinking=True` in processor for Thinking variant
- **bfloat16**: Required. float16 causes NaN. Convert to float for numpy: `.float().cpu().numpy()`

---

## 5. Dataset Acquisition & Preprocessing

### 5.1 Data Directory Structure

```
data/
├── calibration/
│   ├── gqa_balanced_val/          # GQA balanced validation (12,578 samples)
│   └── textvqa_val/               # TextVQA validation (5,000 samples)
├── training/
│   ├── textvqa_train/             # TextVQA training (34,602 samples)
│   ├── vqav2_train/               # VQAv2 training (21,435 samples)
│   └── aokvqa_train/              # A-OKVQA training (17,056 samples)
└── eval/
    ├── pope/                      # POPE eval (9,000: 3×3,000 splits)
    ├── mme/                       # MME eval (2,374 pairs)
    ├── mmbench/                   # MMBench (4,329 samples)
    └── mmmu/                      # MMMU (900 samples)
```

### 5.2 Download Scripts

The project uses `src/data_loader.py` with cached download:

```python
from src.data_loader import load_pope_data, load_textvqa_data, load_gqa_data

# Download and cache all datasets
pope_data = load_pope_data(split="adversarial")      # For evaluation
textvqa_train = load_textvqa_data(split="train")     # For training
gqa_val = load_gqa_data(split="balanced_val")        # For calibration
```

**Manual download** (if needed):

```bash
# POPE (COCO val2014 images + POPE annotations)
python -c "
from datasets import load_dataset
ds = load_dataset('lmms-lab/POPE', split='test')
ds.save_to_disk('data/eval/pope')
"

# TextVQA
python -c "
from datasets import load_dataset
ds = load_dataset('facebook/textvqa', split='train')
ds.save_to_disk('data/training/textvqa_train')
ds = load_dataset('facebook/textvqa', split='validation')
ds.save_to_disk('data/calibration/textvqa_val')
"

# GQA balanced val
# Download from https://cs.stanford.edu/people/dorarad/gqa/download.html
# Place in data/calibration/gqa_balanced_val/
```

### 5.3 Data Loading for Training

The training scripts load data internally:

```python
# In scripts/phase6_head_mask_grpo.py
# Default: TextVQA train only
textvqa = load_dataset("facebook/textvqa", split="train")
train_data = textvqa.select(range(args.train_samples))  # --train-samples N

# With MME train data (recommended for scaled runs):
# --include-mme-train --mme-ratio 0.3 --mme-eval-reserve 200
# Adds 600 MME yes/no samples (30% of 2000), reserves 200 for eval
# Result: 1400 TextVQA + 600 MME = 2000 training samples
```

### 5.4 POPE Evaluation Data

POPE uses COCO val2014 images. The POPE annotation files define 3 splits:
- `random`: random negative objects
- `popular`: popular (frequent) negative objects
- `adversarial`: co-occurring negative objects (hardest)

Each split has 3,000 yes/no questions. Our quick eval uses 60 samples (20 per split).

### 5.5 Image Overlap Check

**CRITICAL**: POPE uses COCO val2014 images. Training data must NOT overlap:
- TextVQA: Uses TextVQA-specific images (no overlap ✓)
- VQAv2: Uses COCO train2014/2017 (minimal overlap, but check IDs)
- A-OKVQA: Uses COCO images — **must filter A-OKVQA IDs against POPE image IDs**

---

## 6. Calibration

Calibration identifies which attention heads are "vision heads" using Cohen's d.

### 6.1 Run Calibration

```bash
python scripts/calibrate.py \
    --model-name Qwen/Qwen3-VL-2B-Thinking \
    --cal-data gqa_balanced_val textvqa_val \
    --num-samples 1000 \
    --output-dir checkpoints/calibration/qwen3_vl_2b \
    --top-k 12
```

**What it does**:
1. Loads model + 1000 calibration samples
2. Runs each sample through the model
3. Records per-head activation norms (via o_proj pre-hooks)
4. Splits into correct/incorrect responses
5. Computes Cohen's d for each of 448 heads
6. Saves top-K heads + steering vectors

### 6.2 Output

```
checkpoints/calibration/qwen3_vl_2b/
├── calibration_meta.json    # Head rankings, Cohen's d scores, model config
└── steering_vectors.pt      # Mean correct/incorrect activation differences per head
```

### 6.3 Pre-computed Results (skip calibration)

The calibration results are already in the repo. The 12 selected vision heads:

```python
VISION_HEADS = [
    (5, 0, 9.795),   # Layer 5, Head 0, Cohen's d = 9.795 — strongest Decision head
    (4, 6, 6.943),   # Layer 4, Head 6
    (23, 2, 6.602),  # Layer 23, Head 2 — only Feature head (late layer)
    (2, 9, 6.551),   # Layer 2, Head 9
    (5, 7, 6.353),   # Layer 5, Head 7
    (11, 2, 6.279),  # Layer 11, Head 2
    (2, 6, 5.440),   # Layer 2, Head 6
    (8, 3, 5.125),   # Layer 8, Head 3
    (2, 8, 5.022),   # Layer 2, Head 8
    (4, 1, 4.957),   # Layer 4, Head 1
    (10, 8, 4.932),  # Layer 10, Head 8
    (5, 10, 4.552),  # Layer 5, Head 10
]
```

File: `checkpoints/calibration/qwen3_vl_2b/calibration_meta.json`

---

## 7. Training — Baseline

The baseline is the unmodified Qwen3-VL-2B-Thinking model with no training. We just evaluate it.

```bash
# Baseline POPE evaluation (60 samples)
python scripts/eval_official.py \
    --model-name Qwen/Qwen3-VL-2B-Thinking \
    --benchmark pope \
    --max-samples 60 \
    --output-dir lab/reports/baseline
```

**Expected results**: POPE 91.7%, Gap 40.0pp, TextVQA 72.7%

---

## 8. Training — Exp1: Gated Head-LSR

**Script**: `scripts/phase6_head_mask_grpo.py`

This is the best-performing experiment. It uses real-vs-black image activation delta as a vision reward signal, with gating between correctness and vision reward.

### 8.1 How It Works

1. For each training step:
   - Sample a question + image from TextVQA train
   - Generate 6 candidate answers (group_size=6, temperature=1.3)
   - Compute R_correct for each candidate
   - For each candidate, run TWO forward passes:
     - Normal forward (real image) → capture vision head activations
     - Black image forward → capture vision head activations
   - Compute head_score(t) = Σ ||act_real - act_black|| per token
   - **Gating**: If R_correct has variance → use R_correct only. Else → use R_head_lsr with token weights.
   - Compute GRPO loss with GDPO normalization
   - Update model weights

### 8.2 Run Command

```bash
PYTHONUNBUFFERED=1 python -u scripts/phase6_head_mask_grpo.py \
    --steps 15 \
    --alpha 0.5 \
    --gdpo \
    --vppo-mask \
    --gated-head-lsr \
    --eval-every 5 \
    --lr 2e-6 \
    --group-size 6 \
    --temperature 1.3 \
    --train-samples 500 \
    --output-dir checkpoints/phase6c/gated_only \
    2>&1 | tee logs/exp1_gated_head_lsr.log
```

### 8.3 Key Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `--steps` | 15 | Training steps (best results at step 10) |
| `--alpha` | 0.5 | Token weight scaling: tw = 1.0 + alpha * norm(head_score) |
| `--gdpo` | flag | Enable GDPO (decoupled reward normalization) |
| `--vppo-mask` | flag | Zero-out negative advantages |
| `--gated-head-lsr` | flag | Enable gating between R_correct and R_head_lsr |
| `--eval-every` | 5 | Evaluate every N steps |
| `--lr` | 2e-6 | Learning rate |
| `--group-size` | 6 | Number of candidates per sample |
| `--temperature` | 1.3 | Sampling temperature (needs >1.0 for diversity) |
| `--train-samples` | 500 | Number of TextVQA training samples |
| `--gdpo-w-correct` | 0.6 | GDPO weight for R_correct (default) |
| `--gdpo-w-lsr` | 0.4 | GDPO weight for R_head_lsr (default) |

### 8.4 Output

```
checkpoints/phase6c/gated_only/
├── step_5/           # Checkpoint at step 5
├── step_10/          # Checkpoint at step 10 (BEST)
├── step_15/          # Checkpoint at step 15
└── final/            # Final checkpoint

lab/reports/phase6_head_mask/gated_only/
├── history.json      # Full training log (steps + evals)
├── fig1_training_curves.png
├── fig2_head_weights.png
└── fig3_eval_progression.png
```

### 8.5 Expected Results

| Step | POPE | Gap | TextVQA |
|------|------|-----|---------|
| Pre | 91.7% | 40.0pp | 72.7% |
| 5 | 95.0% | 44.0pp | 70.7% |
| 10 | **95.0%** | **44.0pp** | **74.7%** |
| 15 | 93.3% | 42.0pp | 70.7% |

**Best checkpoint**: step 10

---

## 9. Training — Exp4: Head Masking KL

**Script**: `scripts/phase7_head_masking_kl.py --exp 4`

Uses causal head masking to measure vision head contribution via KL divergence.

### 9.1 How It Works

1. For each candidate answer:
   - **Normal forward pass** → get logits P_normal
   - **Masked forward pass** (vision heads zeroed at o_proj) → get logits P_masked
   - R_headKL = mean_t(KL(P_normal(t) || P_masked(t)))
2. Gating: R_correct if variance > 0, else R_headKL
3. GDPO normalization: w_correct=0.6, w_kl=0.4

### 9.2 Run Command

```bash
PYTHONUNBUFFERED=1 python -u scripts/phase7_head_masking_kl.py \
    --exp 4 \
    --steps 15 \
    --gdpo \
    --gated-head-kl \
    --eval-every 5 \
    --lr 5e-7 \
    --kl-scale 100.0 \
    2>&1 | tee logs/exp4_head_masking_kl.log
```

### 9.3 Key Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `--exp` | 4 | Experiment number |
| `--gdpo` | flag | GDPO normalization |
| `--gated-head-kl` | flag | Gate between R_correct and R_headKL |
| `--kl-scale` | 100.0 | Scale factor for headKL (raw KL is ~0.02-0.05) |
| `--lr` | 5e-7 | Lower LR than Exp1 (more conservative) |

### 9.4 Key Difference from Exp1

- Exp1: Two forward passes with **different inputs** (real vs black image) → measures activation difference
- Exp4: Two forward passes with **different model states** (normal vs masked heads) → measures causal contribution

---

## 10. Training — Exp5: Learned Head Importance + KL

**Script**: `scripts/phase7_head_masking_kl.py --exp 5`

Replaces fixed 12-head binary masking with a trainable 28×16 importance map.

### 10.1 How It Works

1. Initialize `importance[28, 16]` from Cohen's d scores via `inverse_sigmoid(softmax(d/temperature))`
2. During masked forward: `act *= (1 - sigmoid(importance[l, h]))` (soft masking)
3. `importance` is in the optimizer and updated by GRPO gradient
4. Over training, the model discovers which heads matter most

### 10.2 Run Command

```bash
PYTHONUNBUFFERED=1 python -u scripts/phase7_head_masking_kl.py \
    --exp 5 \
    --steps 15 \
    --gdpo \
    --learned-importance \
    --importance-lr 1e-3 \
    --importance-temp 2.0 \
    --eval-every 5 \
    --lr 5e-7 \
    2>&1 | tee logs/exp5_learned_importance_kl.log
```

### 10.3 Key Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `--exp` | 5 | Experiment number |
| `--learned-importance` | flag | Enable trainable importance map |
| `--importance-lr` | 1e-3 | Learning rate for importance parameters (separate from model LR) |
| `--importance-temp` | 2.0 | Temperature for Cohen's d → importance initialization |

---

## 11. Training — Exp6: Learned Head Importance + Gated LSR

**Script**: `scripts/phase7_head_masking_kl.py --exp 6`

Combines the learned importance map (Exp5) with the real-vs-black activation delta (Exp1).

### 11.1 How It Works

1. Initialize `importance[28, 16]` from Cohen's d (same as Exp5)
2. Hooks on ALL 28 layers (not just 12 heads) to capture activations
3. For each token:
   ```
   score(t) = Σ_{all heads} sigmoid(imp[l,h]) × ||act_real[l,h,t] - act_black[l,h,t]||
            / Σ sigmoid(imp)
   ```
4. Gating: R_correct if variance > 0, else importance-weighted LSR
5. Token weights = 1.0 + alpha × normalized(score)

### 11.2 Run Command

```bash
PYTHONUNBUFFERED=1 python -u scripts/phase7_head_masking_kl.py \
    --exp 6 \
    --steps 15 \
    --gdpo \
    --learned-importance \
    --importance-lr 1e-3 \
    --importance-temp 2.0 \
    --eval-every 5 \
    --lr 5e-7 \
    --alpha 0.5 \
    --lsr-scale 10.0 \
    2>&1 | tee logs/exp6_learned_importance_lsr.log
```

### 11.3 Key Differences from Exp1

| Aspect | Exp1 | Exp6 |
|--------|------|------|
| Head count | 12 (fixed) | 448 (all, soft-weighted) |
| Head weights | Uniform | Learned sigmoid(importance) |
| Head selection | Static (calibration) | Dynamic (gradient-optimized) |
| Hooks | 7 layers (where vision heads are) | All 28 layers |

---

## 12. Training — Exp7: Dynamic Head Selection (FAILED)

**Script**: `scripts/phase7_head_masking_kl.py --exp 7`

**Status**: FAILED — no improvement over baseline.

### 12.1 How It Works

Instead of fixed calibrated heads, Exp7 selects heads per-input by examining attention scores to image tokens. For each training sample:
1. Forward pass with `output_attentions=True`
2. Find image token range via `<|vision_start|>` / `<|vision_end|>` markers
3. Compute per-Q-head attention to image region: `score[l,h] = mean(attn[h, :, image_range])`
4. Select top-K heads by attention score
5. Use those heads for real-vs-black LSR

### 12.2 Results (1000 samples, 25 steps)

| Step | POPE | Gap | TextVQA | dynLSR |
|------|------|-----|---------|--------|
| Pre  | 91.7% | 40.0pp | 72.7% | — |
| 5    | 91.7% | 40.0pp | 70.7% | 1.5 |
| 10   | 91.7% | 40.0pp | 72.7% | 1.8 |
| 15   | 91.7% | 40.0pp | 70.7% | 1.2 |
| 20   | 90.0% | 38.0pp | 70.7% | 1.5 |

### 12.3 Why It Failed

**Root cause**: Attention-to-image ≠ vision-discriminative.
- Exp7 selects heads that **attend** to image tokens (dynLSR signal: 1-2)
- Exp1 uses heads that **discriminate** between real and black images (headΔ: 7-10)
- These are fundamentally different properties. A head can attend heavily to image tokens while producing identical activations for real vs black images.

### 12.4 Lesson Learned

The signal quality matters more than input-adaptivity. Exp1's fixed 12 calibrated heads (selected by Cohen's d from real/black activation difference) produce 5-10x stronger signal than Exp7's dynamically selected heads.

This motivates Exp8: use the real-vs-black delta (Exp1's strong signal) for BOTH head selection AND scoring.

---

## 13. Training — Exp8: Per-Rollout Adaptive Head Gate

**Script**: `scripts/phase6_head_mask_grpo.py --adaptive-heads`

**Status**: NEW — addresses Exp7's failure by using real-vs-black activation delta (Exp1's proven signal) for per-sample head selection.

### 13.1 Core Idea

Combine Exp1's proven signal strength with input-adaptivity:
- Hook ALL 28 layers (448 heads total)
- During LSR computation, compute real-vs-black activation delta for ALL heads
- Select top-K heads by mean delta FOR THIS SPECIFIC SAMPLE
- Compute per-token LSR scores using only those K heads

**Zero extra cost**: LSR already performs real and black image forward passes. Head selection reuses those same activations.

### 13.2 How It Works

```python
# 1. Hook all 28 layers (all 448 heads captured)
hooks = AdaptiveVisionHeadHooks(model, num_layers=28, num_heads=16, head_dim=128)

# 2. For each candidate, compute LSR:
#    a) Forward with real image → capture ALL head activations
#    b) Forward with black image → capture ALL head activations
#    c) Compute per-head mean delta: delta[l,h] = mean_t(||act_real[l,h,t] - act_black[l,h,t]||)
#    d) Select top-K heads by delta
#    e) Per-token score: score(t) = Σ_{selected} delta[l,h] × ||act_real[l,h,t] - act_black[l,h,t]||

# 3. Gating (same as Exp1):
#    If R_correct has variance → use R_correct only
#    Else → use adaptive head-LSR with per-token weights
```

### 13.3 Key Differences from Exp1

| Aspect | Exp1 (Fixed) | Exp8 (Adaptive) |
|--------|-------------|-----------------|
| Head count | 12 (fixed from calibration) | top-K per sample (from all 448) |
| Selection criterion | Cohen's d (offline, from calibration) | Real-vs-black delta (online, per sample) |
| Hooks | 7 layers (where calibrated heads are) | All 28 layers |
| Signal source | Same: real-vs-black activation delta | Same: real-vs-black activation delta |
| Extra cost | None | None (reuses LSR forward passes) |

### 13.4 Why This Should Work

1. **Strong signal**: Uses the same real-vs-black delta as Exp1 (headΔ 5-12), not attention scores (Exp7's weak dynLSR 1-2)
2. **Input-adaptive**: Different images activate different vision heads. A face image vs a text image may use different heads.
3. **Broader coverage**: Can discover important heads outside the fixed top-12 that are relevant for specific inputs
4. **Self-consistent**: The heads used for scoring are the ones most responsive to THIS image

### 13.5 Run Command

```bash
PYTHONUNBUFFERED=1 python -u scripts/phase6_head_mask_grpo.py \
    --steps 30 \
    --alpha 0.5 \
    --gdpo \
    --vppo-mask \
    --gated-head-lsr \
    --adaptive-heads \
    --adaptive-top-k 12 \
    --eval-every 5 \
    --lr 2e-6 \
    --group-size 6 \
    --temperature 1.3 \
    --train-samples 1000 \
    --output-dir checkpoints/exp8_adaptive_head/run1 \
    2>&1 | tee logs/exp8_adaptive_head.log
```

### 13.6 Key Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `--adaptive-heads` | flag | Enable per-rollout adaptive head selection |
| `--adaptive-top-k` | 12 | Number of heads to select per sample |
| `--gated-head-lsr` | flag | Gate between R_correct and adaptive head-LSR |
| Other params | Same as Exp1 | GDPO, VPPO, alpha, lr, etc. |

### 13.7 Expected Behavior

- `headΔ` should be comparable to Exp1 (5-12 range) since we use the same signal
- Selected heads may vary across samples (the whole point of adaptivity)
- POPE should match or exceed Exp1's 93.3% (at 1000 samples scale)
- Log shows `gate_mode=adaptive_head_lsr` when correctness has zero variance

---

## 14. Training — Exp9: Soft-Weighted All-Head LSR

### Motivation

Exp8 selects top-K heads per sample — but this is a discrete cutoff. Problems:

1. **Top-K is discrete**: Heads #12 and #13 might have delta 5.1 and 5.0, but one gets full weight and the other gets zero
2. **Not all vision heads should have high score**: Some heads do vision processing with LOWER activation (suppression heads that quiet non-visual noise)
3. **Mixed text/vision heads ignored**: A head at 0.5 weight (dual-use) carries useful signal but gets discarded by top-K
4. **Model internals are continuous**: Attention weights are soft distributions, not discrete top-K

### Core Idea

Replace discrete top-K selection with **continuous sigmoid weights** derived from the activation delta:

```
weight(l, h) = sigmoid((delta(l,h) - mean_delta) / temperature)
score(t) = Σ_{all l,h} weight(l,h) × ||act_real[l,h,t] - act_black[l,h,t]|| / Σ weight
```

- **ALL 448 heads contribute** — no hard cutoff
- High-delta heads → weight ≈ 1.0 (strong vision signal)
- Low-delta heads → weight ≈ 0.0 (text-only, negligible but non-zero)
- Mixed heads (delta ≈ mean) → weight ≈ 0.5 (dual text+vision use)
- Temperature is adaptive: `T = std(deltas)` — scales with each sample's delta distribution
- Heads with w < 0.01 are skipped for efficiency (~50-100 heads active vs 12 in Exp8)

### Implementation

Added to `scripts/phase6_head_mask_grpo.py`:
- `compute_soft_weighted_head_lsr()` — sigmoid-based continuous weighting
- `compute_rewards_soft_weighted()` — reward computation with soft weights
- CLI flags: `--soft-weighted-heads`, `--soft-temperature`

### Run Command

```bash
PYTHONUNBUFFERED=1 python -u scripts/phase6_head_mask_grpo.py \
    --steps 30 \
    --alpha 0.5 \
    --gdpo \
    --vppo-mask \
    --gated-head-lsr \
    --soft-weighted-heads \
    --soft-temperature auto \
    --eval-every 5 \
    --lr 2e-6 \
    --group-size 6 \
    --temperature 1.3 \
    --train-samples 1000 \
    --output-dir checkpoints/exp9_soft_weighted/run1 \
    2>&1 | tee logs/exp9_soft_weighted.log
```

### Key Differences from Exp8

| Aspect | Exp8 | Exp9 |
|--------|------|------|
| Head selection | Top-12 (discrete) | All 448 (continuous sigmoid weights) |
| Weight range | 0 or delta_value | sigmoid(0-1) |
| Mixed heads | Excluded | Included (~0.3-0.7 weight) |
| Low-delta heads | Excluded | Included (~0.01-0.1 weight) |
| Temperature | N/A (top-K cutoff) | Adaptive (std of deltas) |
| Efficiency | 12 heads scored | ~50-100 heads scored (w > 0.01) |

### Expected Behavior

- Log shows `[soft: N active, XH/YM/ZL]` — count of high/mid/low weight heads
- Expect ~50-100 active heads per sample (w > 0.01)
- headΔ may be lower than Exp8 (diluted by many low-weight heads)
- But per-token discrimination should be finer (captures subtle vision signals)
- POPE target: match or exceed Exp8's 95.0%

---

## 15. Training — Exp10: Sharp Sigmoid Head-LSR

**Script**: `scripts/phase6_head_mask_grpo.py --soft-weighted-heads --soft-temperature auto --soft-temperature-scale 0.33`

**Status**: BEST RESULT — POPE 95.0% at 4 of 6 eval checkpoints (1K scale).

### 15.1 Core Idea

Exp9 uses `T = std(deltas)` for sigmoid temperature, which produces a smooth sigmoid that treats many heads as medium-importance. Exp10 sharpens the sigmoid by dividing the temperature by 3:

```
T_sharp = std(deltas) / 3
weight(l, h) = sigmoid((delta(l,h) - mean_delta) / T_sharp)
```

This creates a near-binary selection: high-delta heads get weight ≈ 1.0, low-delta heads get weight ≈ 0.0. It approximates Exp8's discrete top-K but with smooth gradients.

### 15.2 Run Command (1K Samples)

```bash
PYTHONUNBUFFERED=1 python -u scripts/phase6_head_mask_grpo.py \
    --steps 50 --alpha 0.5 --gdpo --vppo-mask --gated-head-lsr \
    --soft-weighted-heads --soft-temperature auto --soft-temperature-scale 0.33 \
    --eval-every 5 --lr 2e-6 --group-size 6 --temperature 1.3 \
    --train-samples 1000 \
    --output-dir checkpoints/exp10_sharp_soft/run1 \
    2>&1 | tee logs/exp10_sharp_sigmoid.log
```

### 15.3 Results (1K Samples, 50 Steps)

| Step | POPE | Gap | TextVQA |
|------|------|-----|---------|
| Pre  | 91.7% | 40.0pp | 72.7% |
| 5    | **95.0%** | **44.0pp** | 72.7% |
| 10   | **95.0%** | **44.0pp** | 70.7% |
| 15   | **95.0%** | **44.0pp** | 70.7% |
| 20   | **95.0%** | **44.0pp** | 72.7% |
| 25   | 93.3% | 42.0pp | 70.7% |
| 30   | **95.0%** | **44.0pp** | 70.7% |

4 of 6 eval checkpoints hit POPE 95.0% — the most stable result across all experiments.

### 15.4 Key Parameter

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `--soft-temperature-scale` | 0.33 | Divides auto temperature by 3 → sharper sigmoid |

---

## 16. Training — Exp11: Layer-Aware Head Selection

**Script**: `scripts/phase6_head_mask_grpo.py --layer-aware-heads`

**Status**: Implemented but not yet run at scale.

### 16.1 Core Idea

Weight vision heads differently based on their layer position:
- **Decision heads** (layers 2-5): High Cohen's d, process visual info for decision-making
- **Feature heads** (layers 23-27): High activation Δ, extract raw visual features

Exp11 applies layer-dependent scaling to the sigmoid weights from Exp10.

### 16.2 Run Command

```bash
PYTHONUNBUFFERED=1 python -u scripts/phase6_head_mask_grpo.py \
    --steps 50 --alpha 0.5 --gdpo --vppo-mask --gated-head-lsr \
    --soft-weighted-heads --soft-temperature auto --soft-temperature-scale 0.33 \
    --layer-aware-heads \
    --eval-every 5 --lr 2e-6 --group-size 6 --temperature 1.3 \
    --train-samples 1000 \
    --output-dir checkpoints/exp11_layer_aware/run1 \
    2>&1 | tee logs/exp11_layer_aware.log
```

---

## 17. Training — Exp12: Top-P Adaptive Selection

**Script**: `scripts/phase6_head_mask_grpo.py --top-p-heads --top-p-threshold 0.9`

**Status**: Implemented but not yet run at scale.

### 17.1 Core Idea

Instead of fixed top-K or temperature-based sigmoid, select heads by cumulative mass:
1. Sort heads by activation delta (descending)
2. Normalize deltas to a probability distribution
3. Include heads until cumulative probability reaches threshold P (default 0.9)

This auto-adapts the number of selected heads per sample based on delta distribution.

### 17.2 Run Command

```bash
PYTHONUNBUFFERED=1 python -u scripts/phase6_head_mask_grpo.py \
    --steps 50 --alpha 0.5 --gdpo --vppo-mask --gated-head-lsr \
    --top-p-heads --top-p-threshold 0.9 \
    --eval-every 5 --lr 2e-6 --group-size 6 --temperature 1.3 \
    --train-samples 1000 \
    --output-dir checkpoints/exp12_top_p/run1 \
    2>&1 | tee logs/exp12_top_p.log
```

---

## 18. Scaled Training (2K Samples)

### 18.1 Problem: Low Data Coverage

The initial experiments used `--train-samples 500-1000` with 1 sample per step:

| Config | Steps | Samples/Step | Total | Coverage |
|--------|-------|-------------|-------|----------|
| Original | 50 | 1 | 50 | 2.5-5% |
| Scaled (recommended) | 50 | 4 | 200 | 10% |

At 2.5% coverage, the model only sees 50 unique samples. This is insufficient for stable training.

### 18.2 New Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `--samples-per-step N` | 1 | Process N samples per optimizer step (mini-batch) |
| `--eval-steps 10,25,50` | (none) | Only eval at specific steps (skip intermediate) |
| `--seed 42` | 42 | Deterministic data ordering across experiments |

### 18.3 Deterministic Data Iterator

The training loop uses a separate `random.Random(seed)` for data ordering:

```python
data_rng = random.Random(cfg["seed"])  # Deterministic, separate from torch RNG
indices = list(range(len(train_data)))
data_rng.shuffle(indices)  # Epoch-aware reshuffling
```

Two experiments with the same `--seed 42 --train-samples 2000 --samples-per-step 4` will process identical samples in identical order.

### 18.4 MME Training Data

```bash
# Include MME train data (yes/no format, complementary to TextVQA)
--include-mme-train --mme-ratio 0.3 --mme-eval-reserve 200
```

This adds 600 MME train samples (30% of 2000) alongside 1400 TextVQA samples. The 200 reserved MME questions are excluded from training for eval.

### 18.5 Scaled Run Commands

**Exp10 Scaled (Priority 1)**:
```bash
PYTHONUNBUFFERED=1 python -u scripts/phase6_head_mask_grpo.py \
    --steps 50 --alpha 0.5 --gdpo --vppo-mask --gated-head-lsr \
    --soft-weighted-heads --soft-temperature auto --soft-temperature-scale 0.33 \
    --lr 2e-6 --group-size 6 --temperature 1.3 \
    --train-samples 2000 --samples-per-step 4 \
    --include-mme-train --mme-ratio 0.3 --mme-eval-reserve 200 \
    --eval-steps 10,25,50 \
    --eval-pope-samples 60 --eval-blind-samples 50 --eval-textvqa-samples 30 \
    --seed 42 \
    --output-dir checkpoints/exp10_sharp_soft/scaled_v6
```

**Exp8 Scaled (Priority 2, same data order)**:
```bash
PYTHONUNBUFFERED=1 python -u scripts/phase6_head_mask_grpo.py \
    --steps 50 --alpha 0.5 --gdpo --vppo-mask --gated-head-lsr \
    --adaptive-heads \
    --lr 2e-6 --group-size 6 --temperature 1.3 \
    --train-samples 2000 --samples-per-step 4 \
    --include-mme-train --mme-ratio 0.3 --mme-eval-reserve 200 \
    --eval-steps 10,25,50 \
    --eval-pope-samples 60 --eval-blind-samples 50 --eval-textvqa-samples 30 \
    --seed 42 \
    --output-dir checkpoints/exp8_adaptive_head/scaled_v6
```

---

## 19. Multi-Model Expansion

### 19.1 Goal

Validate that Head-LSR GRPO is architecture-agnostic by replicating Exp10 on two additional models.

### 19.2 Supported Models

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
| Thinking | Yes | No | No |

### 19.3 CLI Usage

Use the `--model-key` parameter to select a model:

```bash
# Qwen3-VL-2B (default)
python scripts/phase6_head_mask_grpo.py --model-key qwen3_vl_2b ...

# InternVL3.5-1B
python scripts/phase6_head_mask_grpo.py --model-key internvl3_5_1b ...

# DeepSeek-VL2-Tiny
python scripts/phase6_head_mask_grpo.py --model-key deepseek_vl2_tiny ...
```

### 19.4 Architecture-Specific Adaptations

The script auto-detects and adapts based on model key:

| Component | Qwen3-VL | InternVL3.5 | DeepSeek-VL2 |
|-----------|----------|-------------|--------------|
| Layer path | `model.language_model.layers` | `language_model.model.layers` | `model.layers` |
| Input API | `qwen_vl_utils` + processor | `model.chat()` + torchvision | custom preprocessor |
| Generation | `model.generate()` + think tags | `model.chat()` | `model.generate()` |
| Max temp | 1.3 | 1.3 | 0.7 |
| trust_remote_code | No | Yes | Yes |

### 19.5 InternVL3.5-1B Baseline

Prior results (from BoN+SFT experiments):
- POPE baseline: 78.2%, Gap: 28.2pp
- BoN+SFT R2: 83.4% (+5.2pp), Gap: 33.4pp (+5.2pp)
- Max Cohen's d: 0.774 (vs Qwen3's 9.8 — 12× lower signal)

### 19.6 InternVL Exp10 Command

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

### 19.7 DeepSeek-VL2-Tiny Exp10 Command

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

### 19.8 Calibration (Required Per Model)

Each model needs its own calibration. The calibration files are stored at:
- Qwen3: `checkpoints/calibration/qwen3_vl_2b/calibration_meta.json` (done)
- InternVL: `checkpoints/calibration/internvl3_5_1b/calibration_meta.json` (done)
- DeepSeek: `checkpoints/calibration/deepseek_vl2_tiny/calibration_meta.json` (needed)

### 19.9 Risks

1. **InternVL Cohen's d is 12× lower** → Head-LSR signal may be too weak. Mitigation: higher alpha or activation delta for head selection.
2. **DeepSeek MoE** → Expert routing may dilute per-head signal. Mitigation: Track routing shift.
3. **Memory**: DeepSeek MoE may exceed expectations. Mitigation: Reduce group_size to 4.

---

## 20. Evaluation

### 20.1 POPE Evaluation

```bash
# Quick eval (60 samples, used during training)
python scripts/eval_official.py \
    --model-path checkpoints/phase6c/gated_only/step_10 \
    --benchmark pope \
    --max-samples 60 \
    --output-dir lab/reports/eval_results

# Full eval (9000 samples, for publication)
python scripts/eval_official.py \
    --model-path checkpoints/phase6c/gated_only/step_10 \
    --benchmark pope \
    --max-samples 9000 \
    --output-dir lab/reports/eval_full
```

### 20.2 TextVQA Evaluation

```bash
python scripts/eval_official.py \
    --model-path checkpoints/phase6c/gated_only/step_10 \
    --benchmark textvqa \
    --max-samples 200 \
    --output-dir lab/reports/eval_textvqa
```

### 20.3 Blind Test (Gap Metric)

The blind test replaces all images with black images and measures the accuracy drop:

```bash
python scripts/run_blind_test.py \
    --model-path checkpoints/phase6c/gated_only/step_10 \
    --num-samples 50 \
    --output-dir lab/reports/blind_test
```

**Gap = acc(real_images) - acc(black_images)**
- Higher gap = model depends more on images (good)
- Baseline gap: 40.0pp
- Best gap: 44.0pp (Exp1 step 10)

### 20.4 MME Evaluation

```bash
python scripts/eval_mme.py \
    --model-path checkpoints/phase6c/gated_only/step_10 \
    --max-samples 200 \
    --output-dir lab/reports/mme
```

MME uses pair-based scoring: each sample has a "positive" and "negative" variant. Score = 1.0 only if BOTH answered correctly, 0.5 if one correct, 0.0 if both wrong.

### 20.5 Complete Evaluation Pipeline

After training, evaluate all conditions:

```bash
# 1. Baseline
python scripts/eval_official.py --benchmark pope --max-samples 60
python scripts/eval_official.py --benchmark textvqa --max-samples 200

# 2. Exp1 best (step 10)
python scripts/eval_official.py --model-path checkpoints/phase6c/gated_only/step_10 --benchmark pope --max-samples 60
python scripts/eval_official.py --model-path checkpoints/phase6c/gated_only/step_10 --benchmark textvqa --max-samples 200

# 3. Exp4/5/6 (after training)
python scripts/eval_official.py --model-path checkpoints/phase7/exp4/best --benchmark pope --max-samples 60
# ... repeat for exp5, exp6

# 4. Blind test for each
python scripts/run_blind_test.py --model-path checkpoints/phase6c/gated_only/step_10 --num-samples 50
```

---

## 21. Expected Results

### Final Results Table (Qwen3-VL-2B-Thinking, 60-sample POPE)

```
┌─────────────────────────────────────────────────────┬───────┬────────┬─────────┬──────────┐
│              Experiment                             │ POPE  │  Gap   │ TextVQA │  Status  │
├─────────────────────────────────────────────────────┼───────┼────────┼─────────┼──────────┤
│ Baseline (HF Thinking)                              │ 91.7% │ 40.0pp │ 72.7%   │ Done     │
│ ★ Exp1: Gated Head-LSR (500 samples, step 10)      │ 95.0% │ 44.0pp │ 74.7%   │ BEST TVQ │
│ Exp1: Gated Head-LSR (1K samples, step 10)          │ 93.3% │ 42.0pp │ 72.7%   │ Done     │
│ ★ Exp8: Adaptive Top-K (1K, step 5-20)             │ 95.0% │ 44.0pp │ 72.7%   │ 0 DAMAGE │
│ ★ Exp10: Sharp Sigmoid (1K, steps 5-30)            │ 95.0% │ 44.0pp │ 70.7%   │ STABLE   │
│ Exp10: Sharp Sigmoid (2K, 4sps, step 10)            │ 95.0% │ 44.0pp │   —     │ Scaled   │
│ Exp9: Soft All-Heads (1K, step 5)                   │ 95.0% │ 44.0pp │ 68.7%   │ Unstable │
│ Exp11: Layer-Aware (1K, step 5)                     │ 95.0% │ 44.0pp │ 70.7%   │ No gain  │
│ Exp13_1: Gaussian Target (2K, step 5)               │ 95.0% │ 44.0pp │   —     │ Neg res  │
│ Exp13_2: Linear Target (2K)                         │ 91.7% │ 40.0pp │   —     │ FAILED   │
│ Exp13_3: Asymmetric Target (2K, step 5)             │ 95.0% │ 44.0pp │   —     │ Neg res  │
│ Exp13_4: Clipped Target (2K, step 5)                │ 95.0% │ 44.0pp │   —     │ Neg res  │
│ Exp4: Head Masking KL (step 15)                     │ 90.0% │ 38.0pp │ 72.7%   │ No gain  │
│ Exp5: Learned Importance + KL (step 15)             │ 90.0% │ 38.0pp │ 72.7%   │ FAILED   │
│ Exp6: Learned Importance + LSR (step 15)            │ 90.0% │ 38.0pp │ 70.7%   │ FAILED   │
│ Exp7: Dynamic Head Selection (1K, step 25)          │ 91.7% │ 40.0pp │ 70.7%   │ FAILED   │
├─────────────────────────────────────────────────────┼───────┼────────┼─────────┼──────────┤
│ InternVL3.5-1B Baseline                             │ 78.2% │ 28.2pp │   —     │ Done     │
│ InternVL3.5-1B BoN+SFT R2                          │ 83.4% │ 33.4pp │   —     │ Done     │
│ InternVL3.5-1B Exp10 (planned)                      │  TBD  │  TBD   │  TBD    │ Blocked  │
│ DeepSeek-VL2-Tiny Exp10 (planned)                   │  TBD  │  TBD   │  TBD    │ Planned  │
└─────────────────────────────────────────────────────┴───────┴────────┴─────────┴──────────┘
```

### Top 3 Recommended Experiments (★)

| Rank | Exp | Why Best | Unique Strength | Weakness |
|------|-----|----------|-----------------|----------|
| 1 | **Exp1** | Only one that improved TextVQA (+2pp) | Gating mechanism (novel contribution) | Less stable than Exp10 |
| 2 | **Exp8** | Zero collateral damage to TextVQA | Per-sample targeting (most surgical) | Fewer eval points at 95% |
| 3 | **Exp10** | Most stable (4/6 at 95%) | Data-driven temperature (simplest) | -2pp TextVQA |

See [Section 23](#23-top-3-recommended-reproduce--expand-exp1-8-10) for full reproduction and expansion guide.

### Key Metrics

- **POPE**: Binary VQA accuracy on adversarial split (60 samples). Measures hallucination resistance.
- **Gap**: acc(real) - acc(blind). Measures visual grounding (how much model depends on image).
- **TextVQA**: Open-ended VQA accuracy (50 samples). Measures OCR + reasoning.
- **MME**: Pair-based scoring across 14 subtasks (Perception + Cognition).

---

## 22. Troubleshooting

### Common Issues

#### 1. Model loading OOM
```
torch.cuda.OutOfMemoryError
```
**Fix**: Use bfloat16, enable gradient checkpointing:
```python
model.gradient_checkpointing_enable()
```

#### 2. qwen-vl-utils not installed
```
ImportError: No module named 'qwen_vl_utils'
```
**Fix**: `pip install qwen-vl-utils`

#### 3. Layer path wrong
```
AttributeError: 'Qwen3VLForConditionalGeneration' has no attribute 'layers'
```
**Fix**: Correct path is `model.model.language_model.layers[i]`, NOT `model.model.layers[i]`

#### 4. BFloat16 → numpy conversion fails
```
TypeError: Got unsupported ScalarType BFloat16
```
**Fix**: Always `.float().cpu().numpy()` before numpy conversion

#### 5. All candidates have same answer (zero gradient)
This is expected with binary VQA. The gating mechanism handles it by switching to vision reward. If you see `gate=head_lsr` in logs, the gating is working correctly.

#### 6. POPE evaluation gives 50%
If POPE accuracy is exactly 50%, the model is answering all-Yes or all-No. This means training has collapsed. Stop and restart from last good checkpoint.

#### 7. GPU memory during eval
Eval needs less memory than training but may spike with long Thinking outputs. Set `max_new_tokens=512` for eval.

### File Reference

| Purpose | File |
|---------|------|
| All experiments (Exp1, Exp8-12) | `scripts/phase6_head_mask_grpo.py` |
| Exp4/5/6/7 training | `scripts/phase7_head_masking_kl.py` |
| Calibration | `scripts/calibrate.py` |
| POPE/TextVQA eval | `scripts/eval_official.py` |
| Blind test | `scripts/run_blind_test.py` |
| MME eval | `scripts/eval_mme.py` |
| Smoke test | `scripts/smoke_test.py` |
| Reward functions | `src/rewards.py` |
| Data loading | `src/data_loader.py` |
| Model registry | `src/model_registry.py` |
| Training config | `configs/training.yaml` |
| Model config | `configs/models.yaml` |
| Multi-model plan | `lab/plans/multimodel_expansion.md` |
| Scaled training plan | `lab/plans/exp8_exp10_scaled_v3.md` |
| Qwen3 calibration | `checkpoints/calibration/qwen3_vl_2b/calibration_meta.json` |
| InternVL calibration | `checkpoints/calibration/internvl3_5_1b/calibration_meta.json` |
| Results tracker | `lab/reports/autoresearch/results.tsv` |
| OpenReview report | `lab/reports/OPENREVIEW_REPORT.md` |

### Log Files

All training logs go to `logs/`. Key format:
```
logs/exp1_gated_head_lsr.log
logs/exp4_head_masking_kl.log
logs/exp5_learned_importance_kl.log
logs/exp6_learned_importance_lsr.log
```

### Results Storage

After each experiment, results should be appended to `lab/reports/autoresearch/results.tsv`:
```
exp     method     steps   pope    gap     textvqa     notes   timestamp
```

---

## 23. TOP 3 RECOMMENDED: Reproduce & Expand Exp1, 8, 10

This section is the **priority guide** for continuing experiments on a new machine. It covers exact reproduction commands, expected outputs, and expansion directions for the three best experiments.

### 23.1 Quick Start (New Machine)

```bash
# 1. Clone and setup
git clone https://github.com/dargma/VIGIL.git && cd VIGIL
pip install torch torchvision torchaudio
pip install transformers>=4.45.0 accelerate qwen-vl-utils pillow numpy tqdm matplotlib seaborn datasets

# 2. Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name()}')"

# 3. Verify model loads
python -c "
from transformers import Qwen3VLForConditionalGeneration
m = Qwen3VLForConditionalGeneration.from_pretrained('Qwen/Qwen3-VL-2B-Thinking', torch_dtype='auto', device_map='auto')
print(f'Layers={m.config.num_hidden_layers}, Heads={m.config.num_attention_heads}, KV={m.config.num_key_value_heads}')
# Expected: Layers=28, Heads=16, KV=8
"

# 4. Run baseline eval (should get ~91.7% POPE)
PYTHONUNBUFFERED=1 python -u scripts/phase6_head_mask_grpo.py \
    --steps 0 --alpha 0.5 --gdpo --vppo-mask --gated-head-lsr \
    --eval-pope-samples 60 --eval-blind-samples 50 --eval-textvqa-samples 30 \
    --output-dir checkpoints/baseline_verify
```

### 23.2 Reproduce Exp1: Gated Head-LSR (Recommended First)

**Why Exp1 first**: It's the most novel contribution (gating mechanism), has the best TextVQA result, and uses the simplest head selection (fixed 12 calibrated heads = fastest per-step).

#### Algorithm Summary

```
For each training step:
  1. Sample question+image from TextVQA train
  2. Generate 6 candidates (group=6, T=1.3)
  3. Compute R_correct for each candidate
  4. For each candidate:
     - Forward with real image → capture 12 vision head activations
     - Forward with black image → capture 12 vision head activations
     - head_score(t) = Σ_{12 heads} ||act_real(t) - act_black(t)||₂
  5. GATING DECISION:
     - If variance(R_correct) > 0 → use R_correct ONLY (correctness gradient)
     - If variance(R_correct) = 0 → use head_lsr with token weights (vision gradient)
  6. Token weight: tw(t) = 1.0 + alpha × normalized(head_score(t))
  7. GDPO: advantage = 0.6×Z(R_correct) + 0.4×Z(R_visual)
  8. GRPO loss with VPPO (clip negative advantages to 0)
```

#### Exact Reproduction Command (500 samples, 15 steps)

```bash
PYTHONUNBUFFERED=1 python -u scripts/phase6_head_mask_grpo.py \
    --steps 15 \
    --alpha 0.5 \
    --gdpo \
    --vppo-mask \
    --gated-head-lsr \
    --lr 2e-6 \
    --group-size 6 \
    --temperature 1.3 \
    --max-new-tokens 512 \
    --train-samples 500 \
    --eval-steps 5,10,15 \
    --eval-pope-samples 60 \
    --eval-blind-samples 50 \
    --eval-textvqa-samples 30 \
    --seed 42 \
    --output-dir checkpoints/exp1_reproduce \
    2>&1 | tee logs/exp1_reproduce.log
```

#### Expected Output

```
Pre-training eval: POPE=91.7%, Gap=40.0pp, TextVQA=72.7%
[step 1/15] loss=... correct=... headΔ=7-10 gate=head_lsr ...
...
=== Eval step 5:  POPE=95.0% Gap=44.0pp TextVQA=70.7% ===
=== Eval step 10: POPE=95.0% Gap=44.0pp TextVQA=74.7% ===  ← BEST
=== Eval step 15: POPE=93.3% Gap=42.0pp TextVQA=70.7% ===
```

**Key log indicators**:
- `gate=head_lsr` appears ~70% of steps (when all candidates correct)
- `gate=correctness` appears ~30% (when candidates disagree)
- `headΔ` should be 7-12 range
- `tw` (token weight) ranges from 1.0 to ~1.5

#### Checkpoints

```
checkpoints/exp1_reproduce/
├── best/           # Auto-saved when POPE improves (likely step 5 or 10)
├── final/          # After step 15
└── config.json     # Full config for reproducibility
```

#### Expansion: Exp1 at Scale

```bash
# Exp1 with 2K samples, 4 samples/step, 10 steps (step 10 sweet spot)
PYTHONUNBUFFERED=1 python -u scripts/phase6_head_mask_grpo.py \
    --steps 10 \
    --alpha 0.5 \
    --gdpo \
    --vppo-mask \
    --gated-head-lsr \
    --lr 2e-6 \
    --group-size 6 \
    --temperature 1.3 \
    --max-new-tokens 512 \
    --train-samples 2000 \
    --samples-per-step 4 \
    --include-mme-train --mme-ratio 0.3 --mme-eval-reserve 200 \
    --eval-steps 5,10 \
    --eval-pope-samples 60 --eval-blind-samples 50 --eval-textvqa-samples 30 \
    --seed 42 \
    --output-dir checkpoints/exp1_scaled \
    2>&1 | tee logs/exp1_scaled.log
```

#### Expansion: Exp1 with 1K POPE Eval (CRITICAL for publication)

```bash
# After training, run 1K POPE eval on best checkpoint
python scripts/eval_1k_pope.py \
    --model-path checkpoints/exp1_reproduce/best \
    --num-samples 1000 \
    --output-dir lab/reports/exp1_1k_pope

# Compare with baseline 1K POPE (89.9%)
# If Exp1 > 90.5% at 1K, the improvement is real (not noise)
```

---

### 23.3 Reproduce Exp8: Adaptive Top-K

**Why Exp8**: Zero TextVQA cost. Per-sample head selection is the most targeted approach.

#### Algorithm Summary

```
Same as Exp1, except step 4:
  4. For each candidate:
     - Hook ALL 28 layers (448 heads, not just 12)
     - Forward with real image → capture ALL head activations
     - Forward with black image → capture ALL head activations
     - Compute per-head mean delta: Δ[l,h] = mean_t(||act_real[l,h,t] - act_black[l,h,t]||)
     - Select top-12 heads by Δ FOR THIS SPECIFIC SAMPLE
     - head_score(t) = Σ_{selected 12} Δ[l,h] × ||act_real[l,h,t] - act_black[l,h,t]||
```

**Key difference from Exp1**: Different images may select different heads. A face image activates different heads than a text/OCR image.

#### Exact Reproduction Command (1K samples, 30 steps)

```bash
PYTHONUNBUFFERED=1 python -u scripts/phase6_head_mask_grpo.py \
    --steps 30 \
    --alpha 0.5 \
    --gdpo \
    --vppo-mask \
    --gated-head-lsr \
    --adaptive-heads \
    --adaptive-top-k 12 \
    --lr 2e-6 \
    --group-size 6 \
    --temperature 1.3 \
    --max-new-tokens 512 \
    --train-samples 1000 \
    --eval-steps 5,10,15,20 \
    --eval-pope-samples 60 --eval-blind-samples 50 --eval-textvqa-samples 30 \
    --seed 42 \
    --output-dir checkpoints/exp8_reproduce \
    2>&1 | tee logs/exp8_reproduce.log
```

#### Expected Output

```
=== Eval step 5:  POPE=95.0% Gap=44.0pp TextVQA=72.7% ===
=== Eval step 10: POPE=93.3% Gap=42.0pp TextVQA=70.7% ===
=== Eval step 15: POPE=95.0% Gap=44.0pp TextVQA=72.7% ===
=== Eval step 20: POPE=95.0% Gap=44.0pp TextVQA=72.7% ===
```

TextVQA stays at 72.7% (baseline) — zero collateral damage.

#### Key CLI Flags

| Flag | Purpose |
|------|---------|
| `--adaptive-heads` | Enable per-sample head selection from ALL 448 heads |
| `--adaptive-top-k 12` | Select top-12 heads per sample (matches Exp1's count) |

#### Expansion: Vary K

```bash
# Try K=8 (more selective) and K=20 (broader)
for K in 8 12 16 20; do
    python -u scripts/phase6_head_mask_grpo.py \
        --steps 10 --alpha 0.5 --gdpo --vppo-mask --gated-head-lsr \
        --adaptive-heads --adaptive-top-k $K \
        --lr 2e-6 --group-size 6 --temperature 1.3 --max-new-tokens 512 \
        --train-samples 1000 \
        --eval-steps 5,10 \
        --eval-pope-samples 60 --eval-blind-samples 50 --eval-textvqa-samples 30 \
        --seed 42 \
        --output-dir checkpoints/exp8_topk_${K} \
        2>&1 | tee logs/exp8_topk_${K}.log
done
```

#### Expansion: Exp8 at Scale

```bash
PYTHONUNBUFFERED=1 python -u scripts/phase6_head_mask_grpo.py \
    --steps 10 \
    --alpha 0.5 \
    --gdpo \
    --vppo-mask \
    --gated-head-lsr \
    --adaptive-heads \
    --adaptive-top-k 12 \
    --lr 2e-6 \
    --group-size 6 \
    --temperature 1.3 \
    --max-new-tokens 512 \
    --train-samples 2000 \
    --samples-per-step 4 \
    --include-mme-train --mme-ratio 0.3 --mme-eval-reserve 200 \
    --eval-steps 5,10 \
    --eval-pope-samples 60 --eval-blind-samples 50 --eval-textvqa-samples 30 \
    --seed 42 \
    --output-dir checkpoints/exp8_scaled \
    2>&1 | tee logs/exp8_scaled.log
```

---

### 23.4 Reproduce Exp10: Sharp Sigmoid

**Why Exp10**: Most stable result. Simplest to explain in a paper (one formula, data-driven temperature).

#### Algorithm Summary

```
Same as Exp1, except step 4 uses ALL 448 heads with soft sigmoid weights:
  4. For each candidate:
     - Hook ALL 28 layers (448 heads)
     - Forward with real/black → compute per-head Δ[l,h]
     - Compute adaptive temperature: T = std(Δ) / 3
     - Sigmoid weight: w[l,h] = σ((Δ[l,h] - mean(Δ)) / T)
     - Result: ~111 high-weight heads (>0.8), ~86 mid (0.3-0.8), ~251 low (<0.3)
     - head_score(t) = Σ_{all} w[l,h] × ||act_real[l,h,t] - act_black[l,h,t]|| / Σ w
```

**Key insight**: `T = std(Δ)/3` creates near-binary weights (approximates top-K) while preserving smooth gradients. The `/3` was found empirically — it concentrates ~25% of heads as "active".

#### Exact Reproduction Command (1K samples, 30 steps)

```bash
PYTHONUNBUFFERED=1 python -u scripts/phase6_head_mask_grpo.py \
    --steps 30 \
    --alpha 0.5 \
    --gdpo \
    --vppo-mask \
    --gated-head-lsr \
    --soft-weighted-heads \
    --soft-temperature auto \
    --soft-temperature-scale 0.33 \
    --lr 2e-6 \
    --group-size 6 \
    --temperature 1.3 \
    --max-new-tokens 512 \
    --train-samples 1000 \
    --eval-steps 5,10,15,20,25,30 \
    --eval-pope-samples 60 --eval-blind-samples 50 --eval-textvqa-samples 30 \
    --seed 42 \
    --output-dir checkpoints/exp10_reproduce \
    2>&1 | tee logs/exp10_reproduce.log
```

#### Expected Output

```
=== Eval step 5:  POPE=95.0% Gap=44.0pp TextVQA=72.7% ===
=== Eval step 10: POPE=95.0% Gap=44.0pp TextVQA=70.7% ===
=== Eval step 15: POPE=95.0% Gap=44.0pp TextVQA=70.7% ===
=== Eval step 20: POPE=95.0% Gap=44.0pp TextVQA=72.7% ===  ← 4/6 at 95%
=== Eval step 25: POPE=93.3% Gap=42.0pp TextVQA=70.7% ===
=== Eval step 30: POPE=95.0% Gap=44.0pp TextVQA=70.7% ===
```

Log shows `[soft: 448 active, ~111H/~86M/~251L]` — high/mid/low weight head distribution.

#### Key CLI Flags

| Flag | Purpose |
|------|---------|
| `--soft-weighted-heads` | Enable continuous sigmoid weighting for all 448 heads |
| `--soft-temperature auto` | Temperature = std(deltas) per sample (adaptive) |
| `--soft-temperature-scale 0.33` | Divide temperature by 3 → sharper sigmoid |

#### Expansion: Temperature Scale Ablation

```bash
# The 0.33 scale was found empirically. Try others:
for SCALE in 0.2 0.25 0.33 0.5 1.0; do
    python -u scripts/phase6_head_mask_grpo.py \
        --steps 10 --alpha 0.5 --gdpo --vppo-mask --gated-head-lsr \
        --soft-weighted-heads --soft-temperature auto --soft-temperature-scale $SCALE \
        --lr 2e-6 --group-size 6 --temperature 1.3 --max-new-tokens 512 \
        --train-samples 1000 \
        --eval-steps 5,10 \
        --eval-pope-samples 60 --eval-blind-samples 50 --eval-textvqa-samples 30 \
        --seed 42 \
        --output-dir checkpoints/exp10_scale_${SCALE} \
        2>&1 | tee logs/exp10_scale_${SCALE}.log
done
```

#### Expansion: Exp10 at Scale (Already Validated)

```bash
# This was already run and confirmed: 95.0% at step 10 with 4 sps
PYTHONUNBUFFERED=1 python -u scripts/phase6_head_mask_grpo.py \
    --steps 50 --alpha 0.5 --gdpo --vppo-mask --gated-head-lsr \
    --soft-weighted-heads --soft-temperature auto --soft-temperature-scale 0.33 \
    --lr 2e-6 --group-size 6 --temperature 1.3 --max-new-tokens 512 \
    --train-samples 2000 --samples-per-step 4 \
    --include-mme-train --mme-ratio 0.3 --mme-eval-reserve 200 \
    --eval-steps 10,25,50 \
    --eval-pope-samples 60 --eval-blind-samples 50 --eval-textvqa-samples 30 \
    --seed 42 \
    --output-dir checkpoints/exp10_scaled \
    2>&1 | tee logs/exp10_scaled.log
```

---

### 23.5 Head-to-Head Comparison Protocol

Run all 3 experiments with identical settings for fair comparison:

```bash
COMMON="--steps 10 --alpha 0.5 --gdpo --vppo-mask --gated-head-lsr \
    --lr 2e-6 --group-size 6 --temperature 1.3 --max-new-tokens 512 \
    --train-samples 1000 --eval-steps 5,10 \
    --eval-pope-samples 60 --eval-blind-samples 50 --eval-textvqa-samples 30 \
    --seed 42"

# Exp1: Fixed 12 heads (no soft/adaptive flags)
python -u scripts/phase6_head_mask_grpo.py $COMMON \
    --output-dir checkpoints/compare/exp1 2>&1 | tee logs/compare_exp1.log

# Exp8: Adaptive top-12
python -u scripts/phase6_head_mask_grpo.py $COMMON \
    --adaptive-heads --adaptive-top-k 12 \
    --output-dir checkpoints/compare/exp8 2>&1 | tee logs/compare_exp8.log

# Exp10: Sharp sigmoid
python -u scripts/phase6_head_mask_grpo.py $COMMON \
    --soft-weighted-heads --soft-temperature auto --soft-temperature-scale 0.33 \
    --output-dir checkpoints/compare/exp10 2>&1 | tee logs/compare_exp10.log

# Extract results
for exp in exp1 exp8 exp10; do
    echo "$exp:" && grep "Eval step" logs/compare_${exp}.log
done
```

### 23.6 Priority Actions for Publication

1. **1K POPE eval on all 3 best checkpoints** (highest priority — validates at scale)
   ```bash
   for ckpt in checkpoints/compare/exp1/best checkpoints/compare/exp8/best checkpoints/compare/exp10/best; do
       python scripts/eval_1k_pope.py --model-path $ckpt --num-samples 1000
   done
   ```

2. **MME eval** (shows Perception up, Cognition flat = surgical improvement)
   ```bash
   python scripts/eval_mme.py --model-path checkpoints/compare/exp1/best --max-samples 200
   ```

3. **Scaled runs** (2K samples, 4 sps, 10 steps — for "scales to more data" claim)

4. **Multi-model** (InternVL3.5-1B — blocked by transformers 5.0, try `pip install transformers==4.45.2`)

5. **Ablations** (for paper):
   - `--alpha 0` (no token weighting) — proves head weighting matters
   - Remove `--gated-head-lsr` (no gating) — proves gating matters
   - `--group-size 4` vs `8` — GRPO group sensitivity

### 23.7 Universal Training Tips

1. **Step 10 is the sweet spot**: All methods plateau by step 10. Beyond step 10, catastrophic forgetting degrades results. Train exactly 10 steps.

2. **Use `--seed 42`**: Ensures identical data ordering across experiments for fair comparison.

3. **Monitor `headΔ`**: Should be 7-12. If it saturates at 10.0 for many steps, the model has maximized vision head activation and further training is overfitting.

4. **Monitor gate mode**: `gate=head_lsr` ~70% means gating is working. If 100%, all samples are already correct (too easy data). If 0%, data is too hard (all wrong).

5. **OOM handling**: The script automatically retries with fewer sub-samples on OOM. `[3/4]` in logs means 1 sub-sample was skipped. This is fine — graceful degradation.

6. **Log buffering**: When piping through `tee`, output buffers. Use `PYTHONUNBUFFERED=1` and `python -u` to minimize. Alternatively, check checkpoints directly:
   ```bash
   ls checkpoints/exp1_reproduce/best/  # If exists, step 5 eval found new best
   ```

7. **Eval sample counts**: 60 POPE = 1.67pp per sample. For publication, use 1K+ samples.

---

## 24. Exp13: Target-Calibrated Reward (Negative Result)

**Status**: COMPLETE — negative result. Included for completeness and to prevent re-attempting.

### 24.1 Hypothesis

Standard monotonic headΔ reward (more activation = better) may overshoot, pushing heads beyond their natural operating range. A target-centered reward penalizing deviation from a calibrated target should improve stability.

### 24.2 Four Sub-Variants

| Variant | Mode | Formula | Result |
|---------|------|---------|--------|
| 13_1 | Gaussian | `exp(-(Δ-μ)²/2σ²)` | 95.0% step 5, 93.3% step 10 |
| 13_2 | Linear | `clamp(1-|Δ-μ|/μ, 0)` | **FAILED** (91.7%, no improvement) |
| 13_3 | Asymmetric | Gaussian + 2× blind penalty | 95.0% step 5, 93.3% step 10 |
| 13_4 | Clipped | Plateau ±σ + Gaussian tails | 95.0% step 5, 93.3% step 10 |

Target μ = 8.0 (auto-calibrated from baseline correct responses).

### 24.3 Run Commands

```bash
COMMON="--steps 10 --alpha 0.5 --gdpo --vppo-mask --gated-head-lsr \
    --soft-weighted-heads --soft-temperature auto --soft-temperature-scale 0.33 \
    --lr 2e-6 --group-size 6 --temperature 1.3 --max-new-tokens 512 \
    --train-samples 2000 --samples-per-step 2 \
    --include-mme-train --mme-ratio 0.3 --mme-eval-reserve 200 \
    --eval-steps 5,10 --eval-pope-samples 60 --eval-blind-samples 50 \
    --seed 42 --target-calibrated"

# 13_1: Gaussian
python -u scripts/phase6_head_mask_grpo.py $COMMON \
    --target-mode gaussian --target-sigma 2.0 \
    --output-dir checkpoints/exp13_1_gaussian 2>&1 | tee logs/exp13_1.log

# 13_2: Linear (WILL FAIL — included for negative result)
python -u scripts/phase6_head_mask_grpo.py $COMMON \
    --target-mode linear --target-sigma 2.0 \
    --output-dir checkpoints/exp13_2_linear 2>&1 | tee logs/exp13_2.log

# 13_3: Asymmetric
python -u scripts/phase6_head_mask_grpo.py $COMMON \
    --target-mode asymmetric --target-sigma 2.0 \
    --output-dir checkpoints/exp13_3_asymmetric 2>&1 | tee logs/exp13_3.log

# 13_4: Clipped
python -u scripts/phase6_head_mask_grpo.py $COMMON \
    --target-mode clipped --target-sigma 2.0 \
    --output-dir checkpoints/exp13_4_clipped 2>&1 | tee logs/exp13_4.log
```

### 24.4 CLI Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `--target-calibrated` | flag | Enable target-centered reward |
| `--target-delta` | 0.0 | Target headΔ (0 = auto-calibrate from 10 baseline samples) |
| `--target-sigma` | 2.0 | Width of reward bell curve |
| `--target-mode` | gaussian | One of: gaussian, linear, asymmetric, clipped |

### 24.5 Why It Failed

1. **The hypothesis was wrong**: Monotonic headΔ is not overshooting. The sharp sigmoid already provides implicit targeting via its temperature.
2. **Linear mode zeroes too many heads**: Δ > 2×target gets zero reward, excluding most active heads.
3. **The real bottleneck is catastrophic forgetting** after step 10, not reward function shape. No reward modification can fix this.
4. **All 3 working variants match Exp10 at step 5 but can't sustain it** — same pattern as all other experiments.

### 24.6 Lesson

Do not attempt further reward function modifications (bell curves, asymmetric penalties, etc.). The reward signal is already strong enough. Future work should focus on:
- Larger-scale evaluation (1K+ POPE)
- Catastrophic forgetting mitigation (replay buffers, EWC, smaller LR schedules)
- Multi-model validation
- Additional benchmarks (MME, MMMU-Pro)
