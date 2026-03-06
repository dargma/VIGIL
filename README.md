# VIGIL: Vision-Grounded Inference via Guided head-Level steering

Small VLMs (1-3B) progressively lose visual attention during generation — attention to visual tokens decays as O(1/L_total). In thinking/reasoning mode, this creates "long-wrong" trajectories where models ignore images entirely. Standard GRPO/DAPO training makes this worse, producing **blind reasoners** that perform better *without* images.

VIGIL solves this with two stages:

1. **Inference-time head-level steering** — identify vision-specialized attention heads via calibration (Cohen's d), inject steering vectors when the model is uncertain (agreement-gated)
2. **RL training with visually-grounded reward** — GRPO/DAPO where reward explicitly measures whether the model uses visual information, not just answer correctness

## Key Innovation: Visually-Grounded Reward

```
R_total = 0.3 × R_correct + 0.5 × R_visual_grounding + 0.2 × R_fluency
```

- **R_vhad**: Activation differential in vision heads between real vs black image forward passes
- **R_asi**: Answer sensitivity — do answers change when the image is removed?
- **w2 > w1 by design** — grounding matters more than correctness to prevent blind reasoner collapse

## The Killer Experiment: Blind Test

After training, replace all test images with black images. Measure Gap = acc(real) - acc(black).

| Condition | Expected Gap |
|-----------|-------------|
| Baseline (no training) | ~16 |
| Standard GRPO (R_correct only) | ~7 (blind reasoner) |
| VIGIL (full reward) | ~19 (more image-dependent) |

## Target Models

| Model | Params | Role |
|-------|--------|------|
| Qwen3-VL-2B | 2B | Main target (+ Thinking variant) |
| InternVL3.5-1B | 1B | Main target |
| DeepSeek-VL2-Tiny | MoE | MoE expert routing analysis |
| Qwen2.5-VL-7B | 7B | Unsteered reference baseline |

## Benchmarks

POPE (3 splits), MMBench, MME, MMMU — evaluated across 5 modes: greedy baseline, steered only, post-GRPO+steering, post-DAPO+steering, blind test.

## Project Structure

```
src/                  # Core implementation
  rewards.py          # R_vhad + R_asi (core novelty)
  agreement_gate.py   # Conditional steering
  blind_test.py       # Black image evaluation
  vision_drift.py     # Thinking chain analysis
  steerer.py          # Head-level activation steering
  calibrator.py       # Cohen's d calibration
  profiler.py         # Per-head activation extraction
  model_registry.py   # Model loading + verification
  trainer.py          # GRPO/DAPO with visual reward
configs/              # Model, training, eval configs
scripts/              # Entry point scripts
lab/                  # Experiment journal + reports
data/                 # Datasets (not tracked)
logs/                 # Training logs
```

## Related Work

- **Qwen-LookAgain** (2505.23558): Mathematical proof of visual attention decay
- **When to Think and When to Look** (2511.15613): Long-wrong trajectories in thinking VLMs
- **Bi et al.** (CVPR 2025): Vision-specialized heads are sparse (<7%)
- **DVRP** (2601.06801): Standard GRPO/DAPO creates blind reasoners
- **VISTA** (ICML 2025): Training-free attention steering (~40% hallucination reduction)

## Setup

```bash
git clone https://github.com/dargma/VIGIL.git
cd VIGIL
pip install -r requirements.txt
```

## License

MIT
