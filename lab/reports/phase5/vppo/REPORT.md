# Phase 5: VPPO Token Masking

**Date**: 2026-03-14 03:10
**Model**: Qwen/Qwen3-VL-2B-Thinking

## Config

| Parameter | Value |
|-----------|-------|
| experiment | vppo |
| num_steps | 50 |
| group_size | 6 |
| temperature | 1.3 |
| top_p | 0.95 |
| max_new_tokens | 512 |
| min_think_tokens | 32 |
| lr | 2e-06 |
| beta_entropy | 0.01 |
| grad_accum | 2 |
| max_grad_norm | 1.0 |
| lsr_scale | 2.0 |
| eval_every | 999 |
| output_dir | checkpoints/phase5/vppo |
| vppo_low_weight | 0.1 |
| dpo_beta | 0.1 |
| dpo_lsr_weight | 0.3 |
| dpo_format_weight | 0.1 |
| dpo_use_lsr | True |
| w_correct | 0.5 |
| w_format | 0.2 |
| w_lsr | 0.3 |

## Results

| Metric | Pre | Best | Delta |
|--------|-----|------|-------|
| POPE | 91.7% | 93.3% | +1.7% |
| Gap | 40.0% | — | — |

**Steps**: 21 effective / 50 total (29 skipped, 58%)

**Loss**: mean=-0.0698, min=-0.2089, max=0.0659
