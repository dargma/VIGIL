# Phase 2 GRPO-LSR Visualization Figures

Generated: 2026-03-10

All figures use seaborn-v0_8-whitegrid style, 150 DPI, font size 12.

---

## Figure 1: LSR Heatmap
**File**: `fig1_lsr_heatmap.png`

Per-token KL divergence heatmap showing how the Logit Shift Reward (LSR) signal varies across token positions in the thinking chain. Samples are sorted by thinking length (shortest at top). The dashed vertical line marks the median think/answer boundary.

**Key observation**: LSR signal is concentrated in the early-to-mid thinking phase (tokens 10-80), with high-KL hotspots (red, >5.0) appearing where the model's reasoning diverges most from the unsteered baseline. Signal drops sharply after the think/answer boundary, confirming that LSR primarily captures reasoning-phase logit shifts.

**Data source**: `lab/reports/pope_thinking_steering/lsr_trajectories_20260310_113327.json` (30 samples)

---

## Figure 2: Multi-Round Training Dynamics
**File**: `fig2_training_dynamics.png`

Four-panel plot showing training metrics across 5 rounds of GRPO-LSR (15 steps each, 73 total steps):

- **Top-left**: GRPO loss — generally negative (reward-maximizing), with some oscillation in later rounds.
- **Top-right**: Mean LSR (KL divergence) — decreases across rounds, indicating the model internalizes steering signal.
- **Bottom-left**: Training correctness — improves from ~0.67 (R1) to near 1.0 (R5).
- **Bottom-right**: POPE accuracy at eval points — peaks at 95.0% in Round 2 and Round 4.

Each round is color-coded. Dotted vertical lines mark round boundaries. Stars mark eval points.

**Data source**: `checkpoints/phase2_grpo_lsr/round{1-5}/history_*.json`

---

## Figure 3: Eval Progression
**File**: `fig3_eval_progression.png`

Dual y-axis line plot tracking POPE accuracy (left axis, blue circles) and Blind Gap (right axis, pink squares) at each eval checkpoint (steps 5, 10, 15 per round).

- **Peak POPE**: 95.0% at cumulative step 25 (Round 2, step 10) and step 46 (Round 4, steps 5 and 10)
- **Peak Gap**: 44.0pp at the same points
- **Trend**: Accuracy improves in R1-R2, plateaus in R3, recovers in R4, stabilizes in R5

Round boundaries annotated at bottom.

**Data source**: Same as Figure 2.

---

## Figure 4: Reward Distribution
**File**: `fig4_reward_distribution.png`

Mean reward with shaded standard deviation band (solid lines) and mean LSR raw signal (dashed lines, secondary axis) across all training steps.

- Reward mean increases from ~0.41 (R1 start) to ~0.55 (R5), with narrowing std indicating convergence.
- LSR raw signal decreases over training, consistent with the model learning to produce outputs that the LSR reward encourages — the model internalizes visually-grounded reasoning patterns.

**Data source**: Same as Figure 2.

---

## Figure 5: Method Comparison
**File**: `fig5_method_comparison.png`

Grouped bar chart comparing all VIGIL methods on POPE Accuracy and Blind Gap:

| Method | POPE Acc (%) | Blind Gap (pp) |
|--------|-------------|----------------|
| Baseline (Thinking) | 91.7 | 40.0 |
| Steered (alpha=5) | 89.7 | 38.0 |
| BoN+SFT (Instruct) | 88.0 | 38.0 |
| **GRPO-LSR Best (Thinking)** | **95.0** | **44.0** |

GRPO-LSR achieves the best results on both metrics (+3.3pp POPE, +4.0pp Gap over baseline). Steering alone and BoN+SFT both underperform the Thinking baseline on these 60-sample evaluations. The GRPO-LSR best bar is highlighted with a bold border.

**Note**: Baseline and GRPO-LSR use Thinking mode (60-sample POPE eval). BoN+SFT uses Instruct mode. Sample sizes differ — full 9K eval pending.

---

## Reproduction

```bash
python lab/reports/phase2_visualizations/generate_figures.py
```
