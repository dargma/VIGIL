# Phase 5: Multiplicative Gated GDPO

**Date**: 2026-03-14 00:46
**Model**: Qwen/Qwen3-VL-2B-Thinking

## Config

| Parameter | Value |
|-----------|-------|
| experiment | mult_gate |
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
| output_dir | checkpoints/phase5/mult_gate |
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
| POPE | 91.7% | 90.0% | -1.7% |
| Gap | 40.0% | — | — |

**Steps**: 46 effective / 50 total (4 skipped, 8%)

**Loss**: mean=-0.0450, min=-0.1887, max=0.0579
