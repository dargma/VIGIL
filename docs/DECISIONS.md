# VIGIL Design Decisions

## D1: BoN+SFT over TRL GRPO
We chose Best-of-N + SFT over TRL GRPOTrainer because GRPO collapsed 3/3 times on binary VQA at 2B scale. Binary answers have ~1 bit output entropy → GRPO groups have no diversity → zero advantage → always-yes/always-no collapse. BoN+SFT curates high-quality data first, then does standard SFT, which is immune to this failure mode.

## D2: IIG over R_vhad/R_asi for BoN scoring
We chose Image Information Gain (log-prob difference with/without image) as the primary visual grounding reward because it requires only 2 forward passes (not generation), works on any VLM without architecture modification, and showed 99.4% positive rate at calibration. R_vhad requires hook installation and is slower; R_asi requires two generations.

## D3: o_proj pre-hook over attention output hook
We steer at the o_proj input (before the output projection) because individual Q-head outputs are separable there as [num_Q_heads × head_dim]. After o_proj, heads are mixed and cannot be steered independently.

## D4: Cohen's d over activation delta for head selection
Cohen's d measures effect size (mean difference / pooled std), accounting for variance. Raw activation delta doesn't account for natural variance in activations. Top-Cohen's-d heads are the most reliably discriminative, not just the most active.

## D5: Steer layer 4+ only
Layers 1-3 are used by Qwen3-VL's DeepStack mechanism for cross-modal fusion. Steering these layers disrupts the fusion and hurts performance. Layer 4+ is safe for steering.

## D6: KL coef = 0.02 for DAPO (not 0.0)
The original DAPO paper uses kl_coef=0.0, but that's for large models (7B+). At 2B scale, removing KL regularization entirely causes catastrophic forgetting. 0.02 is small enough to allow exploration while preventing collapse.

## D7: VLMEvalKit standard prompts for evaluation
We use the exact POPE prompt format from VLMEvalKit (`"{question} Please answer yes or no."`) and its `YOrN_Extraction()` parser for publication-quality results. Custom prompts showed inflated/deflated numbers depending on format.

## D8: Mixed (non-binary) data for RL training
Binary VQA (yes/no) provides insufficient reward variance for RL. TextVQA and A-OKVQA open-ended questions provide diverse answer spaces where RL can find meaningful gradients.

## D9: Soft thresholding rewards over hard gates
Hard reward thresholds create cliff edges where small changes in model behavior cause large reward jumps → noisy gradients. Sigmoid-based soft thresholds provide smooth reward landscapes with defined gradients everywhere. Temperature annealing (soft→hard) serves as curriculum.
