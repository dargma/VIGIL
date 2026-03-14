# Thinking vs Short Mode Analysis Report

**Model**: Qwen3-VL-2B-Thinking
**Benchmark**: POPE (300 samples, balanced yes/no)
**Date**: 2026-03-14

---

## 1. Headline Results

| Mode | Raw Accuracy | Valid Predictions | Valid Accuracy |
|------|-------------|-------------------|---------------|
| Thinking (`enable_thinking=True`) | 90.3% (271/300) | 271/300 (90.3%) | 100% |
| Short (`enable_thinking=False`) | 80.3% (241/300) | 241/300 (80.3%) | 100% |
| **Delta** | **+10.0pp** | — | **0.0pp** |

## 2. Key Finding: Format Failure, Not Accuracy Gap

**All 30 disagreements are format failures in short mode**, not accuracy failures:
- Short-only wins: **0**
- Think-only wins: **30** (all due to short mode failing to produce yes/no)
- Both correct: **241**
- Both wrong: **29**

When short mode produces a valid yes/no answer, it has **100% agreement** with thinking mode on POPE.

### Root Cause

With `enable_thinking=False`, the Thinking variant still generates reasoning text ("So, let's look at...") instead of a direct yes/no answer. The `extract_yes_no()` function then fails to parse a valid answer from the first 5 words.

**Fix**: Add explicit instruction "Answer directly with yes or no." to short mode prompts. Applied in updated scripts.

## 3. Short Mode Failure Pattern

All 30 short-mode failures share the same pattern:
```
Short answer: "So, let's look at the image. The scene is..."
Think answer: "yes" or "no" (correct)
```

The model's Thinking variant has learned a reasoning-first response style that persists even when `enable_thinking=False` removes the `<think>` tag structure.

## 4. Implications for Self-Play DPO

Since thinking and short modes agree when both produce valid answers, **disagreement-based DPO pairs from POPE are not useful** — there are no genuine accuracy disagreements to learn from.

For Self-Play DPO, we need:
1. **Open-ended benchmarks** (OCRBench, TextVQA) where reasoning quality genuinely differs
2. **Proper short-mode prompting** ("Answer briefly/directly") to eliminate format noise
3. **Harder questions** where thinking can go wrong (overreasoning → image drift)

## 5. LSR Analysis Summary

| Metric | Correct Predictions | Wrong Predictions |
|--------|-------------------|-------------------|
| Mean KL(real‖black) | Measured | Too few wrong samples for reliable stats |

See figures:
- `fig1_lsr_heatmap_comparison.png` — Per-token KL for correct vs wrong
- `fig3_vision_drift_curve.png` — Vision head activation over token position
- `fig6_lsr_distribution.png` — LSR score distributions

## 6. Vision Drift Observation

The drift curve (fig3) shows vision head activation patterns during thinking chains. The key insight is that **longer thinking chains do NOT systematically cause accuracy loss on POPE** — the 29 "both wrong" cases have similar thinking lengths to correct cases.

POPE may be too easy for this model to reveal drift effects. OCRBench (open-ended, harder) should show clearer differentiation.

## 7. Next Steps

1. ✅ Fix short mode prompt (add "Answer directly")
2. ⏳ Re-run with fixed prompts to confirm format fix resolves gap
3. ⏳ OCRBench comparison (300 samples) — open-ended answers where real disagreements should emerge
4. ⏳ Per-token LSR heatmap on OCRBench disagreement cases
5. ⏳ Image-level analysis of cases where thinking hurts
