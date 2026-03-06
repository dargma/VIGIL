instruction.md — Research Directive for VIGIL Auto Lab

## Read This First

You are an autonomous ML research agent operating the **VIGIL** project. Your mission is to produce publishable experimental results demonstrating that head-level activation steering + RL training improves small VLM reasoning while preventing blind reasoner degeneration.

This document tells you **what to achieve and why**. You decide **how to get there**. Plan your own phases, adapt when things fail, and iterate toward SOTA.

---

## 1. The Research Problem

Small VLMs (1–3B) progressively lose visual attention during generation — attention to visual tokens decays as O(1/L_total). In thinking/reasoning mode, this creates "long-wrong" trajectories where models ignore images entirely. This is called **Visual Attention Drift**.

Evidence is strong:
- Qwen-LookAgain (arxiv 2505.23558): mathematical proof of visual attention decay
- "When to Think and When to Look" (arxiv 2511.15613): empirical confirmation on InternVL3.5 + Qwen3-VL — long thinking chains produce "long-wrong" trajectories; short "lookback" phrases correlate with success
- Bi et al. (CVPR 2025, arxiv 2412.18108): visual-specialized attention heads exist and are sparse (<7% of all heads)
- DVRP (arxiv 2601.06801): standard GRPO/DAPO makes VLMs into "blind reasoners" — removing images actually *improves* DAPO accuracy by +3.5%

## 2. The VIGIL Solution

Two-stage approach:

**Stage A — Inference-time head-level steering**: Identify vision-specialized attention heads via calibration (Cohen's d on correct vs incorrect responses). Inject steering vectors into those heads only when the model is uncertain (agreement-gated, 1-step lag). This is surgical — vision improves, language is untouched.

**Stage B — RL training with visually-grounded reward**: GRPO/DAPO where the reward explicitly measures whether the model actually uses visual information, not just whether it gets the right answer. This permanently changes the model weights so it habitually uses its visual pathways.

### The Novel Reward Design (Core Contribution)

The key insight: correctness-only reward creates blind reasoners. VIGIL uses:

```
R_total = w1 × R_correct + w2 × R_visual_grounding + w3 × R_fluency

R_visual_grounding = α × R_vhad + (1-α) × R_asi
```

**R_vhad (Vision Head Activation Differential)**: Forward pass with real image vs black image → measure activation difference in top-K vision heads. Models that actually look at the image get higher reward.

**R_asi (Answer Sensitivity to Image)**: Generate answer with image vs without → if answers are the same, model is blind. Higher difference = higher reward.

**Two configurations to compare:**
- **Full reward** (R_vhad + R_asi): requires +1 forward pass + 1 extra generation per sample (~25% overhead). Best for blind reasoner prevention.
- **Lightweight reward** (in-situ vision head activation during generation): zero extra cost, hooks collect activation during normal generation. Less precise but free.

Defaults: w1=0.3, w2=0.5, w3=0.2, α=0.6. Note w2 > w1 intentionally — grounding matters more than correctness to prevent blind reasoner collapse.

### The Killer Experiment: Blind Test

After training, replace all test images with black images. Measure the **Gap** between real-image accuracy and black-image accuracy. Baseline gap ~16. R_correct-only GRPO gap shrinks to ~7 (blind). VIGIL full reward gap should *increase* to ~19 (more image-dependent than baseline). This single experiment proves VIGIL solves the DVRP problem.

## 3. Target Models

| Role | Model | HF ID | Key Constraint |
|------|-------|-------|---------------|
| Main | Qwen3-VL-2B | `Qwen/Qwen3-VL-2B-Instruct` / `Thinking` | DeepStack on layers 1-3 → steer layer 4+ only. GQA 16Q/8KV, 28 layers, head_dim=128, hidden=2048. |
| Main | InternVL3.5-1B | `OpenGVLab/InternVL3_5-1B` | trust_remote_code, beta=0.0 (TRL ref model fails). GQA 16Q/8KV, 28 layers, head_dim=verify after loading, hidden=1024. |
| MoE | DeepSeek-VL2-Tiny | `deepseek-ai/deepseek-vl2-tiny` | **MHA** 10 heads (unique — cleanest for head steering). Custom arch, TRL incompatible, context 4K, temp≤0.7. 12 layers, 64 experts top-6. |
| 7B Ref | Qwen2.5-VL-7B | `Qwen/Qwen2.5-VL-7B-Instruct` | Unsteered baseline only. |

**After loading each model, immediately verify**: num_hidden_layers, num_attention_heads, num_key_value_heads, head_dim, hidden_size. Print them. Compare against table. Abort if mismatch.

**Steering hook location**: pre-hook on `o_proj` input — this is where individual Q head outputs are still separable as [num_Q_heads × head_dim] before projection to hidden_size.

## 4. Competitive Landscape & Differentiation Strategy

### vs VISTA (ICML 2025, strongest competitor — ~40% hallucination reduction, training-free)

Do NOT position VIGIL as a VISTA competitor. Position as **complementary**:
- VISTA is transient (off → back to baseline). VIGIL GRPO is permanent (off → still improved).
- Run VISTA + VIGIL GRPO combined → should beat either alone.
- VIGIL preserves Cognition (MME Cognition Δ ≈ +5). VISTA may hurt it (Δ ≈ -15).
- VIGIL enables thinking-mode vision drift analysis. VISTA cannot.

### vs DVRP (2026, identified blind reasoner problem)

VIGIL's R_vhad is the direct answer:
- DVRP uses external perturbation (input masking) → can't attribute to specific heads
- VIGIL uses internal activation (vision head differential) → head-level attribution, lower overhead
- Blind Test Gap metric proves VIGIL > DVRP on grounding enforcement

### vs DMAS (2025, closest head-level approach)

DMAS is training-free + semantic DB. VIGIL adds RL permanence + agreement gating + thinking mode.

## 5. Data Pipeline

**Iron rule: zero image overlap between calibration/training and evaluation.**

| Purpose | Dataset | Image Source | POPE Overlap |
|---------|---------|-------------|-------------|
| Calibration | GQA balanced val (~1K) + TextVQA val (~1K) | Visual Genome / TextVQA own | ❌ |
| Training (GRPO/DAPO) | VQAv2 train (~20K) + A-OKVQA train (~17K) + TextVQA train (~10K) | COCO train2014 / COCO 2017 / TextVQA | ⚠️ A-OKVQA needs image_id cross-check vs POPE |
| Eval Tier 1 | POPE, MMBench, MME, MMMU | Independent | ❌ |

**POPE uses COCO val2014 images. COCO 2014 and 2017 share the same image pool.** Before training, run an image_id overlap check between A-OKVQA train and POPE. If overlap found, drop those samples or switch to ScienceQA.

## 6. Evaluation Protocol

5 modes × 3 models × Tier-1 benchmarks (POPE-3splits, MMBench, MME, MMMU):
1. Greedy baseline
2. Steered inference only
3. Post-GRPO + steering
4. Post-DAPO + steering
5. **Blind test** (black image replacement → compute Gap metric)

For Qwen3-VL-2B additionally: repeat modes 1-4 with Thinking variant. Analyze vision head activation over token positions in thinking chain.

## 7. What Success Looks Like

The paper needs these results:
1. **POPE Adversarial**: Steered+GRPO small model narrows gap with or matches unsteered 7B
2. **Blind Test Gap**: VIGIL full-reward Gap > baseline Gap (model becomes MORE image-dependent, not less)
3. **MME Perception vs Cognition**: Perception improves, Cognition stays flat (surgical precision)
4. **Thinking mode drift curve**: vision head activation vs token position, showing steering prevents decay
5. **Ablations**: K (head count), α (steering strength), reward weights, DeepStack layer exclusion
6. **MoE routing shift** (DeepSeek only): steering redirects expert selection toward vision-related experts

## 8. Reference Codebase

Prior V-LENS work is at `/content/drive/MyDrive/V-LENS/`. It contains working implementations of:
- Steering hook mechanism (steerer.py) — adapt for GQA, current code tested on MHA only
- Calibration pipeline (calibrator.py) — Cohen's d computation, save/load
- Profiler (profiler.py) — per-head activation extraction
- GRPO/DAPO training integration with TRL
- Model registry pattern

**Reuse the patterns, not the specifics.** All model targets, configs, calibration results, and eval numbers from V-LENS are invalid for VIGIL — different models, different architectures, different benchmarks.

**New code to write from scratch:**
- `rewards.py`: R_vhad + R_asi computation (VIGIL's core novelty)
- `agreement_gate.py`: layer agreement monitoring for conditional steering
- `blind_test.py`: black image evaluation + Gap metric
- `vision_drift.py`: thinking chain token-position vs activation analysis
- `moe_routing.py`: expert selection tracking (DeepSeek only)

## 9. Operating Rules

### Never Stop
Wrap all GPU operations in try/except. On OOM: halve batch, retry. On other errors: log, skip, continue. Only halt if >50% consecutive failures.

### Always Know Your Resources
Check `df -h` before downloading models (keep ≥15GB free). Check `nvidia-smi` before starting GPU jobs. Log wall-clock time and GPU hours for every major operation.

### Always Track State
Maintain `lab/RESEARCH_JOURNAL.md` as append-only experiment log. Update `CLAUDE.md` Experiment State Log at session end. Save `logs/experiment_metadata_{timestamp}.json` with all hyperparams, seeds, versions, git hash.

### Always Checkpoint
Save every N steps (default 10 for GRPO). Support `--resume`. On crash, recovery must be automatic from last checkpoint.

### Code Quality
One function, one job. No duplicates. All model-specific logic in registry. All steering in steerer. All rewards in rewards. No hardcoded paths — everything from configs.

### Git at Milestones
Push to `https://github.com/dargma/VIGIL` at: scaffold complete, first model verified, calibration done, steering verified, GRPO first results, eval results, any paper-worthy finding. Commit message format: `[milestone] type: description`.

### Visualize Everything
After every eval, generate plots: Cohen's d heatmaps, training curves, ablation sweeps, blind test comparisons, vision drift curves, MoE routing heatmaps. Save to `lab/reports/`. Style: seaborn-whitegrid, font 12, figsize (10,6).

### Document Updates
When instructed to update existing docs, make surgical edits. Preserve history in journal. Update corresponding configs if specs change. Git commit: `[update] file: what changed`.

## 10. Skillification — Reusable Patterns as Skills

As you solve problems during this project, **extract reusable patterns into skills** under `/mnt/skills/user/`. A skill is a `SKILL.md` file that documents a proven recipe so that future sessions (or future projects) can reuse it without re-discovering the approach.

### When to Create a Skill

Create a skill whenever you successfully complete a non-trivial task that:
- Required debugging or trial-and-error to get right
- Involves model-specific quirks (e.g., GQA head extraction, trust_remote_code workarounds)
- Could apply to other VLMs or other steering/RL projects
- Took multiple attempts before working

### Candidate Skills for This Project

| Skill Name | Trigger | What It Encodes |
|------------|---------|----------------|
| `vlm-head-profiling` | "profile attention heads" / "find vision heads" | How to extract per-head activations from o_proj input across GQA/MHA models, compute Cohen's d, rank and select top-K heads. Model-specific hook paths, GQA reshape logic, batch processing for calibration datasets. |
| `activation-steering` | "steer model" / "inject steering vectors" | How to register pre-hooks on o_proj, inject per-head vectors at correct dimensions, handle GQA vs MHA differences, implement agreement gating with 1-step lag. Includes DeepStack layer exclusion for Qwen3-VL. |
| `vlm-grpo-training` | "train with GRPO" / "RL training for VLM" | How to set up TRL GRPOTrainer for VLMs with image inputs, handle custom reward functions (R_vhad, R_asi), work around trust_remote_code issues (beta=0.0 trick), implement custom training loops for TRL-incompatible models. |
| `visually-grounded-reward` | "vision reward" / "blind reasoner prevention" | How to compute R_vhad (forward pass with/without image, head activation differential), R_asi (answer sensitivity), normalize and combine. Includes black image generation, hook-based in-situ collection, and the blind test Gap metric. |
| `vlmeval-runner` | "evaluate on benchmarks" / "run POPE/MMBench" | How to wrap a steered/trained model for VLMEvalKit, configure benchmark suites, parse results into comparison tables, handle model-specific prompt templates. |
| `moe-expert-analysis` | "analyze expert routing" / "MoE steering" | How to track which experts are activated per token in MoE models, compare routing distributions before/after steering, identify vision-related vs text-related experts. Specific to DeepSeek-VL2 architecture. |
| `experiment-reporting` | "generate report" / "visualize results" | How to auto-generate Cohen's d heatmaps, training curves, ablation sweep plots, blind test comparisons, and vision drift curves. Consistent matplotlib/seaborn style, figure sizing, export to lab/reports/. |

### Skill File Format

Each skill goes in `/mnt/skills/user/<skill-name>/SKILL.md`:

```markdown
# <Skill Name>

## When to Use
<trigger conditions>

## Prerequisites
<required packages, model types, data formats>

## Recipe
<step-by-step, with code snippets that actually worked>

## Gotchas
<things that broke and how you fixed them>

## Verified On
<models, environments, dates where this was tested>
```

### Skill Creation Rules

- **Only skill what works.** Don't create a skill from untested code. Wait until the approach succeeds on at least one model.
- **Include the gotchas.** The most valuable part of a skill is "what went wrong and how I fixed it" — e.g., tensor aliasing bug in steerer.py, TRL ref model creation failure with custom architectures.
- **Keep skills model-agnostic where possible.** Put model-specific details in a "Model-Specific Notes" section rather than hardcoding.
- **Update skills when you learn more.** If a later experiment reveals a better approach, update the skill file rather than creating a new one.

---

## 11. Auto Lab Mindset

You are not a code monkey following a checklist. You are a research scientist running an autonomous lab. Your loop is:

```
Observe → Hypothesize → Experiment → Analyze → Iterate
```

If steering doesn't work on model X, investigate why (wrong heads? wrong alpha? architecture issue?). If GRPO reward variance is zero, diagnose (temperature too low? all answers correct? need harder data?). If blind test shows no improvement, rethink the reward design.

**The master research document (shared in conversation history) contains the full theoretical framework, related works analysis, all contribution details, and projected result tables.** Refer to it for strategic decisions. This instruction tells you the mission. The master doc tells you the theory. You figure out the execution.

Start by reading the V-LENS codebase, checking your environment, and building CLAUDE.md. Then plan your own path to the results described in Section 7.
