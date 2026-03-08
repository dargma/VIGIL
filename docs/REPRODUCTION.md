# VIGIL Reproduction Guide

## Environment Setup

### Hardware
- GPU: NVIDIA A100 40GB or L4 23GB (minimum)
- Disk: 50GB+ free space
- RAM: 32GB+

### Software
```bash
pip install torch torchvision transformers accelerate
pip install datasets pillow numpy pandas
pip install qwen-vl-utils  # for Qwen3-VL image processing
pip install trl peft  # optional, for LoRA experiments
```

### Data Preparation
Datasets are cached on Google Drive at `data/`:
```
data/
├── eval/pope/          # 9000 POPE samples (Arrow format)
├── training/
│   ├── vqav2_train/    # 21,435 VQAv2 samples
│   ├── textvqa_train/  # 34,602 TextVQA samples
│   └── aokvqa_train/   # 17,056 A-OKVQA samples
└── calibration/        # GQA + TextVQA val for calibration
```

To download from scratch:
```python
from datasets import load_dataset
# POPE
pope = load_dataset("lmms-lab/POPE", split="test")
pope.save_to_disk("data/eval/pope")
```

## Running Experiments

### Step 1: Calibration
```bash
python scripts/calibrate.py --model qwen3_vl_2b --samples 2000
```
Output: `checkpoints/calibration/qwen3_vl_2b/`

### Step 2: Baseline Evaluation
```bash
python scripts/eval_official_fast.py --max-samples 500
```

### Step 3: BoN+SFT Training
```bash
python scripts/block2_best_of_n_sft.py \
  --n-candidates 8 --train-samples 1000 \
  --sft-epochs 2 --sft-lr 2e-6
```
Output: `checkpoints/block2_bon/final/`

### Step 4: DAPO Training
```bash
# Think mode first
python scripts/run_dapo.py --phase think --dapo-steps 30 --group-size 4

# Short-answer mode
python scripts/run_dapo.py --phase short --dapo-steps 50 --group-size 8
```

### Step 5: Phase 2 Experiments
```bash
# Dual-head ablation (inference only)
python scripts/phase2_experiments.py --exp p2_01 --eval-samples 500

# Steered distillation
python scripts/phase2_experiments.py --exp p2_02 --train-samples 1000
```

## Expected Results

| Condition | POPE Acc | Blind Gap | Notes |
|-----------|----------|-----------|-------|
| Baseline | 87.4% | 37.4pp | VLMEvalKit standard, 500 samples |
| BoN+SFT | 85.5%* | 37.0pp* | *Custom eval; official ~89.5% |
| Steered α=3 | 88.0% | 38.0pp | All heads, inference only |
| DAPO (TBD) | TBD | TBD | In progress |

## Key Files
- `scripts/eval_official_fast.py` — Official VLMEvalKit-standard evaluation
- `scripts/block2_best_of_n_sft.py` — BoN+SFT training pipeline
- `scripts/run_dapo.py` — DAPO training (think + short modes)
- `scripts/phase2_experiments.py` — Phase 2 experiment axes
- `src/iig.py` — Image Information Gain computation
- `src/steerer.py` — Head-level activation steering
- `src/calibrator.py` — Cohen's d calibration pipeline
