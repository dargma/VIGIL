# VIGIL Experiment Log

## exp_001: Pre-Validation (PV1-PV4)
- Date: 2026-03-06
- Hypothesis: Vision heads exist and steering helps in Qwen3-VL-2B
- Results:
  - PV1: Vision heads exist (mean Δ=6.1, max=66.2) ✓
  - PV2: Steering helps (+2pp POPE, +9pp at α=10) ✓
  - PV3: Thinking model marginal (+1pp at α=1) ✓
  - PV4: Blind gap increases with steering (25.4→28.4pp, +3.0pp) ✓
- Analysis: All pre-validations passed. Two head types discovered (decision vs feature).
- Next: IIG calibration → BoN+SFT training

## exp_002: IIG Block 0 Calibration
- Date: 2026-03-07
- Hypothesis: IIG reward can distinguish visually grounded vs ungrounded responses
- Results: 99.4% positive IIG, λ=0.0615 (auto-calibrated), mean IIG=9.95
- Analysis: IIG works as intended. Gate threshold 60% easily met.
- Next: Block 1 GRPO with IIG

## exp_003: Block 1 GRPO v1 (TRL)
- Date: 2026-03-07
- Hypothesis: TRL GRPOTrainer + IIG reward improves POPE
- Changes: group=4, temp=1.0, lr=5e-6, LoRA r=16, 50 steps
- Results: POPE 76→31% (COLLAPSE: always-no)
- Analysis: Binary VQA has ~1 bit entropy → zero diversity in groups → collapse
- Next: Try v2 with format reward

## exp_004: Block 1 GRPO v2
- Date: 2026-03-07
- Changes: group=8, temp=1.2, lr=2e-6, LoRA r=8, + format reward
- Results: POPE 76→30% (COLLAPSE: always-yes)
- Analysis: Same root cause. TRL GRPO not viable for binary VQA.

## exp_005: Block 1 GRPO v3 (ultra-conservative)
- Date: 2026-03-07
- Changes: beta=0.1, lr=5e-7, LoRA r=4, 5 steps only
- Results: POPE 77→31% (COLLAPSE: always-yes)
- Analysis: 3/3 TRL GRPO attempts collapsed. Pivot to BoN+SFT.

## exp_006: Block 2 GRPO Custom (R_correct only)
- Date: 2026-03-08
- Hypothesis: Custom GRPO loop with entropy bonus avoids collapse
- Results: POPE 84.5→83.5% (no collapse but no improvement)
- Analysis: Stable but no signal for improvement with correctness-only reward.

## exp_007: Block 2 GRPO Custom (R_correct + IIG)
- Date: 2026-03-08
- Results: POPE 84.5→85.0% (+0.5pp, within noise)
- Analysis: IIG adds marginal signal but GRPO advantage too noisy.

## exp_008: Block 2 BoN+SFT ★ BREAKTHROUGH
- Date: 2026-03-08
- Hypothesis: BoN selection + SFT on curated data beats GRPO
- Changes: N=8 candidates, score=R_correct + λ·IIG, SFT 2 epochs on 692 samples
- Results:
  - POPE: 83.0→85.5% (+2.5pp)
  - Blind Gap: 32.0→37.0pp (+5.0pp)
  - Real acc: 82→87%, Blind acc: 50% (stable)
- Analysis: BoN+SFT is strictly superior to GRPO for binary/short VQA. Curates high-quality data then trains on it.
- Checkpoint: `checkpoints/block2_bon/final`
- Next: Phase 2 experiment axes

## exp_009: Official VLMEvalKit Evaluation
- Date: 2026-03-08
- Results (full 9K POPE, VLMEvalKit standard):
  | Condition | Acc | F1 | Precision | Recall | Gap |
  |-----------|-----|-----|-----------|--------|-----|
  | baseline | 89.6% | 89.5% | 92.8% | 86.4% | 39.6pp |
  | bon_r1 | 89.5% | 89.3% | 93.8% | 85.2% | 39.5pp |
  | steered α=5 | 87.4% | 87.5% | 86.8% | 88.2% | 37.4pp |
- Analysis: Official eval shows higher baselines. BoN+SFT gains precision (+1.0pp) not raw accuracy. Steering hurts with official prompts.

## exp_010: P2-01 Dual-Head Ablation
- Date: 2026-03-08
- Hypothesis: Decision heads and feature heads have different effects on POPE vs Gap
- Changes: 7 steering conditions (baseline, all α=3/5, decision α=3/5, feature α=5/10)
- Results (500 samples, VLMEvalKit standard):
  | Condition | Acc | Gap |
  |-----------|-----|-----|
  | baseline | 87.4% | 37.4pp |
  | all_heads α=3 | 88.0% | 38.0pp |
  | all_heads α=5 | 87.8% | 37.8pp |
  | decision α=3 | 87.8% | 37.8pp |
  | decision α=5 | 87.6% | 37.6pp |
  | feature α=5 | 87.8% | 37.8pp |
  | feature α=10 | 87.8% | 37.8pp |
- Analysis: All steering within ~0.6pp. No strong separation between head types at inference. α=3 slightly better than α=5. Feature heads insensitive to α increase (5→10 no change).
- Next: DAPO training, steered distillation

## exp_011: DAPO Think Mode (IN PROGRESS)
- Date: 2026-03-08
- Hypothesis: DAPO with thinking model improves visual grounding via extended reasoning
- Changes: Qwen3-VL-2B-Thinking, TextVQA training data, group=4, 30 steps, soft rewards
- Interim results:
  - Step 10: POPE acc=76.3%, gap=26.3pp, reward=0.420±0.099
  - Step 20: reward=0.717±0.035 (improving)
- Status: RUNNING

## Next Experiments (Priority Order)
1. Complete DAPO think mode → eval
2. DAPO short-answer mode (start from BoN+SFT checkpoint)
3. P2-02: Steered distillation BoN+SFT
4. P2-03: Drift-penalized selection (Axis B)
5. P2-04: IIG-weighted SFT loss (Axis D)
