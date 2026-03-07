VGIL — Lab Instruction (Unified)

> This is the single instruction file for VIGIL. It replaces all previous instruction files (instruction.md, instruction2–7).
> It contains both the research context (R_vhad GRPO theory) and the experiment plan (pre-validation phase).
> Philosophy: mission + theory + direction. You read the codebase, you figure out execution.

---

## Scope

**One model, two modes, minimal data.**

- **Model**: `Qwen/Qwen3-VL-2B-Instruct` (short answer) and `Qwen/Qwen3-VL-2B-Thinking` (reasoning)
- All other models (InternVL, DeepSeek, 7B) are deferred to a future phase. Do not load them.

| Purpose | Dataset | Size |
|---------|---------|------|
| Calibration | GQA balanced val | 500 |
| Eval (main) | POPE Adversarial | 500 |
| Eval (sanity) | POPE Random | 100 |
| Eval (reasoning) | MMMU subset | 200 |

This fits in one Colab session. If GPU is slow (T4), halve the sweep sample counts.

---

## Mission

**Pre-validate the R_vhad GRPO hypothesis through steering experiments.**

The end goal is not steering itself — it's proving that **R_vhad (vision head activation difference) works as a GRPO reward signal** that teaches the model to use its own visual pathways. But running GRPO blindly without knowing if the underlying mechanism works is a waste of GPU hours.

This phase answers: **"If we manually force vision head activation up, does performance go up?"** If yes, then using activation as reward (R_vhad) will point the model in the right direction. If no, R_vhad is a dead end regardless of how we train.

The logic chain:
```
Steering proves: activation ↑ → accuracy ↑ (causal link)
    ↓ therefore
R_vhad reward: incentivize activation ↑ via GRPO → model learns it permanently
    ↓ and then
Post-training analysis: RL-strengthened heads = steering-targeted heads? (mechanistic validation)
```

---

## Background: R_vhad GRPO — What We're Building Toward

### The Problem: Small VLMs Go Blind During Reasoning

Small VLMs (1–3B) suffer from **Visual Attention Drift**: as autoregressive generation progresses, attention to visual tokens decays at O(1/L_total) rate (Qwen-LookAgain, 2505.23558). The model progressively relies on text prior instead of the image, leading to hallucination. In thinking mode, long reasoning chains make this worse — the model produces "long-wrong" trajectories that ignore the image entirely ("When to Think and When to Look", 2511.15613).

Large models (7B+) have enough redundant capacity to compensate. Small models don't. This is why a steered 2B model could potentially match an unsteered 7B — the 2B has the capacity, it just doesn't use its visual pathways efficiently.

### The DVRP Problem: Standard GRPO Makes It Worse

DVRP (2601.06801) showed that training small VLMs with standard GRPO using only correctness reward (R_correct) creates a **blind reasoner**: the model learns to answer correctly by exploiting text shortcuts, and its dependence on the actual image *decreases* after training. The Blind Test Gap (accuracy with real image − accuracy with black image) drops from ~16 to ~7, meaning the model barely cares whether it sees the image or not.

This happens because R_correct doesn't distinguish *how* the model arrived at the answer. A correct answer from text prior gets the same reward as a correct answer from actually looking at the image.

### The Solution: R_vhad — Reward the Model for Looking

R_vhad (Vision Head Activation Difference) directly measures whether the model is using its visual pathways:

```
R_vhad = mean(|activation(real_image) - activation(black_image)|) across vision heads
```

High R_vhad = the model's internal representations change significantly when it sees the image = it's actually looking. Low R_vhad = the model gives similar activations regardless of image = it's ignoring visual information.

By incorporating R_vhad into the GRPO reward:

```
R_total = 0.3 × R_correct + 0.5 × R_vhad + 0.2 × R_fluency
```

We incentivize the model to **be correct AND use the image**. The w2 > w1 weighting is intentional: we'd rather have a model that looks at the image and sometimes gets it wrong than a blind model that gets it right by guessing. The DVRP finding motivates this exact design choice.

### Implementation: How R_vhad Works in GRPO

During each GRPO step:
1. Generate N candidate answers (group_size=8) for each VQA sample
2. For each candidate, run one additional forward pass with the image replaced by black
3. Compare head activations: real vs black, across all heads (or top-K from calibration)
4. The activation difference IS the R_vhad signal — no external model, no human annotation
5. Combine with R_correct and R_fluency → group-relative advantage → policy gradient update

**Overhead**: one extra forward pass per candidate (~25% compute increase). A lightweight variant hooks into the generation forward pass directly (zero overhead but noisier signal).

**Key insight**: R_vhad doesn't require pre-selecting "vision heads". All heads contribute, but vision-irrelevant heads naturally produce near-zero difference (real ≈ black) and thus near-zero reward signal. The reward is self-selecting.

### Why This Is Novel

| Approach | Reward Signal | Knows About Vision? | Prevents Blind Reasoner? |
|----------|--------------|---------------------|--------------------------|
| Standard GRPO (VLM-R1, etc.) | R_correct only | No | No (DVRP problem) |
| DVRP (2601.06801) | R_correct + length | No | Identified problem, no fix |
| VIGIL (ours) | R_correct + **R_vhad** + R_fluency | **Yes — directly measures visual pathway usage** | **Yes — by design** |

No existing work uses internal vision head activation as a GRPO reward signal. DVRP diagnosed the blind reasoner problem; VIGIL's R_vhad is the direct solution.

### The Two-Paper Structure

This instruction covers **Part 1 only**: proving the mechanism works (steering experiments). The full story is:

**Part 1 (this phase)**: Steering proves activation ↑ → accuracy ↑. This validates R_vhad's direction.

**Part 2 (next phase, separate instruction)**: R_vhad GRPO trains the model to self-strengthen visual pathways. Post-training analysis shows RL-discovered vision heads match steering-targeted heads. Blind Test Gap increases (opposite of DVRP's finding). This is the main contribution.

Part 1 alone is workshop-publishable. Part 1 + Part 2 is a full paper.

---

## What You Must Answer (4 Pre-Validations for GRPO)

By the end of this phase, you need clear empirical answers to these 4 questions. All 4 must be positive to proceed to R_vhad GRPO.

1. **Do vision heads exist?** — Real vs black activation difference must be significant across a subset of heads. If negligible, R_vhad will produce zero reward signal. → *Validates: R_vhad can produce non-zero signal.*

2. **Does forcing activation up improve accuracy?** — Steering must improve POPE accuracy by Δ ≥ 0.5. If manually pushing activation up doesn't help, training the model to do it won't help either. → *Validates: R_vhad points in the right direction.*

3. **Is the effect stronger in thinking mode?** — If steering prevents vision drift during long reasoning chains, R_vhad will be especially valuable for thinking models where drift is worst. → *Validates: R_vhad addresses the hardest case.*

4. **Does the Blind Test Gap increase with steering?** — Gap = acc(real) - acc(black). If steering makes the model more image-dependent (larger gap), R_vhad will push against blind reasoner collapse. → *Validates: R_vhad prevents the DVRP problem.*

Additionally, optimize K, α, and layer range — these inform the design of R_vhad (which heads to weight, how strongly).

---

## Experiment Blocks

### Block 1 — Head Profiling (Calibration)

Load model. Hook every `o_proj` (28 layers × 16 Q heads = 448 heads). Run 500 GQA samples with real image and with black image. From one forward pass per condition, extract:

**Architecture note**: Qwen3-VL-2B uses GQA (16 Q heads, 2 KV heads, head_dim=128). Profiling and steering operate on **Q head output** — reshape o_proj output from `[batch, seq, 2048]` to `[batch, seq, 16, 128]` to isolate individual heads.

- **1A**: Activation difference heatmap (real − black) → identifies vision heads
- **1B**: Vision attention ratio per head (attention to visual tokens / total) → cross-validates 1A
- **1C**: Cohen's d per head (correct vs incorrect split) → identifies discriminative heads

Key question: Do the three maps agree? If vision-active heads ≠ discriminative heads, figure out why before proceeding.

**Blind Test (from the same data)**: Since you already run real vs black forward passes, also compute accuracy on both. Report **Blind Test Gap = acc(real) - acc(black)**. If the gap is small, the model barely uses vision even without steering — that's useful baseline data. This metric carries forward into GRPO phase as the key measure of whether training prevents "blind reasoner" collapse (DVRP problem).

### Block 2 — Steering Effect → R_vhad Feasibility

Using top-K heads from Block 1 (start with K=5, α=2.0):

- **2A**: Baseline vs steered on POPE Adversarial 500 (Instruct model, short answer)
- **2B**: Baseline vs steered on POPE Adversarial 500 (Thinking model, reasoning mode)
- **2C**: Per-sample analysis — categorize into helped/hurt/neutral, analyze what predicts each
- **2D**: Blind Test Gap comparison — steered vs unsteered (does steering increase image-dependence?)

The **vision drift curve** from 2B (token position vs vision head activation, steered vs unsteered) is a candidate paper figure. If steering prevents activation decay during long thinking chains, that's the headline result.

**How each result predicts GRPO success:**
- 2A Δ > 0 → R_vhad reward direction is correct (activation ↑ = good)
- 2B Δ > 2A Δ → R_vhad is most valuable for thinking models (stronger paper story)
- 2C "helped" samples correlate with low baseline activation → R_vhad will have highest gradient where it matters most
- 2D Gap increases → R_vhad will actively prevent blind reasoner collapse

If Block 2 shows Δ < 0.5 on POPE: do NOT proceed to Block 3. Debug first — verify hooks are modifying activations, verify steering vector magnitude, verify calibration found real vision heads. **If steering can't improve accuracy by manual intervention, R_vhad GRPO cannot succeed.**

### Block 3 — Optimization (conditional on Block 2 success)

- **α sweep**: [1, 2, 3, 5] on 100 POPE-A samples
- **K sweep**: [1, 3, 5, 8, 16] on 100 POPE-A samples  
- **Layer sweep**: steer one layer at a time, 100 POPE-A samples per layer
- **DeepStack test**: steer all layers vs layers 4+ vs layers 1–3 only, 100 samples each

Use 100 samples for sweeps (speed), then re-eval the winner on full 500 for reliability.

### Block 4 — Visualization (from data already collected)

No extra GPU time needed. Generate from Block 1–3 data:

- Attention heatmaps overlaid on images (samples where steering flipped wrong→right)
- Activation trajectory comparison (helped vs hurt vs neutral)
- UMAP/t-SNE of final-layer embeddings (steered vs unsteered)
- Combined head importance ranking

---

## Execution — The Iteration Loop

**The core problem this solves**: tendency to plan/document/research instead of running experiments. The loop below must actually execute as code, not exist as a document.

### Per-Iteration Protocol

```
Experiment → Measure (2 min) → Compare (3 min) → Diagnose (5 min) → Decide (2 min) → Execute next
```

Total overhead between experiments: **max 12 minutes**. If you're spending 30 minutes analyzing between 10-minute experiments, the ratio is wrong.

**Measure**: Print raw numbers. No interpretation yet.

**Compare**: Delta vs baseline, delta vs previous best. Simple subtraction.

**Diagnose**: If below expectation → 3 ranked hypotheses, each with a testable prediction. If above → note what worked, push harder. If mixed → identify the trade-off.

**Decide**: Exactly one of:
- **Iterate** — same setup, changed hyperparameter
- **Pivot** — different approach (hypothesis was wrong)
- **Lock** — good enough, move to next block
- **Abort** — fundamentally broken, debug before continuing

**Execute**: Run the next experiment NOW. Not a plan. Not a document. Code.

### Anti-Patterns

- "Let me research optimal hyperparameters first" → Run default, measure, adjust.
- "I should write a comprehensive analysis" → Print 5 numbers, write 3 sentences, run next.
- "I need a more sophisticated approach before testing" → Test simple first.
- Writing TODO.md instead of running code → Delete it, run experiment.

---

## Reporting & Git

### Every iteration produces:

```
lab/reports/iter_{N}_{name}_{timestamp}/
├── results.json
├── comparison.md       # vs baseline, vs best (3 lines each)
├── plot_main.png       # ONE self-explanatory plot
└── diagnosis.md        # hypotheses or push-harder plan
```

Plots must be self-explanatory: title says what was tested, gray dashed baseline, Δ annotation, legend.

### Every iteration gets a git commit:

```
git commit -m "iter-{N}: {experiment} — {one_line_result}"
```

Push every 5 iterations or at any milestone.

### Journal entries follow this structure:

```markdown
## YYYY-MM-DD — Iter {N}: {Title That States the Finding}
**Hypothesis**: ...
**Setup**: model, config, data, hyperparams
**Results**: numbers with Δ vs baseline
**Interpretation**: what the numbers mean (not restating them)
**Verdict**: paper-ready / promising / negative / inconclusive
**Next**: what will be run next and why
```

The "6 months later" test: can you reconstruct the full story from the journal alone?

---

## Autonomous Execution with Ralph Loop

Ralph Loop (`ralph-loop@claude-plugins-official`) is already installed. It intercepts session exits via a stop hook and re-feeds your prompt while preserving all file modifications and git history between iterations — creating autonomous improvement cycles where Claude refines its work based on previous attempts.

Ref: https://claude.com/plugins/ralph-loop

### Commands

```bash
# Start a loop (always set --max-iterations)
/ralph-loop "<prompt>" --max-iterations 20 --completion-promise "DONE"

# Cancel an active loop
/cancel-ralph
```

### Core Principle: Iterate Until Satisfied

**Do not accept first-pass results.** Use `/ralph-loop` for every experiment block. The loop should keep running until:
- Results meet the success criteria defined in the prompt
- The completion promise is emitted
- OR max iterations are reached (indicating the approach needs rethinking)

If a phase completes but results are unsatisfactory (e.g., POPE Δ < 1.0 when you expected 2.0+), **re-run the phase** with an adjusted prompt that incorporates what was learned. Ralph Loop's power is that each iteration sees its own previous work — use this to converge on good results, not to stop at the first number you get.

The workflow is: **run phase → inspect results → if not good enough, adjust prompt → run again**. Repeat until the result is paper-worthy or you've confirmed the approach doesn't work.

### Phase Structure

Run each phase as a Ralph loop.

**Phase 1 — Setup & Calibration**:
```
/ralph-loop "
Read the VIGIL codebase. Run Block 1 (head profiling) on Qwen3-VL-2B with 500 GQA samples.
Produce activation difference heatmap, vision attention ratio heatmap, and Cohen's d heatmap.
Also compute Blind Test Gap (acc with real image - acc with black image).
Save to lab/reports/. Commit to git.
If vision heads are found (>5 heads with Cohen's d > 0.5), output <promise>CALIBRATION_DONE</promise>.
If no clear vision heads found, document findings and output <promise>CALIBRATION_DONE</promise> anyway.
If blocked for 3+ iterations, document what's blocking and output <promise>BLOCKED</promise>.
" --max-iterations 10 --completion-promise "CALIBRATION_DONE"
```

**Phase 2 — Steering Experiments**:
```
/ralph-loop "
Read calibration results from lab/reports/. Run Block 2 (steering effect):
- 2A: baseline vs steered on POPE Adversarial 500 (Instruct)
- 2B: baseline vs steered on POPE Adversarial 500 (Thinking)
- 2C: per-sample helped/hurt/neutral analysis
Follow the iteration protocol: measure → compare → diagnose → decide → execute.
Generate vision drift curve for thinking mode.
Git commit every iteration. Save reports to lab/reports/.
Output <promise>STEERING_DONE</promise> when both 2A and 2B have clear results (positive or negative).
If blocked for 3+ iterations, output <promise>BLOCKED</promise> with explanation.
" --max-iterations 15 --completion-promise "STEERING_DONE"
```

**Phase 3 — Optimization** (only if Phase 2 showed effect):
```
/ralph-loop "
Read steering results from lab/reports/. Run Block 3 sweeps:
- α sweep, K sweep, layer sweep, DeepStack test
Use 100 samples for sweeps. Re-eval winner on 500 samples.
Follow iteration protocol. Git commit every iteration.
Output <promise>OPTIMIZATION_DONE</promise> when optimal K, α, and layer range are determined with evidence.
If blocked for 3+ iterations, output <promise>BLOCKED</promise> with explanation.
" --max-iterations 20 --completion-promise "OPTIMIZATION_DONE"
```

**Phase 4 — Visualization & Summary**:
```
/ralph-loop "
Generate all Block 4 visualizations from existing data.
Write comprehensive summary in RESEARCH_JOURNAL.md.
Create a one-page findings summary for the paper.
Git push all results.
Output <promise>COMPLETE</promise> when done.
" --max-iterations 5 --completion-promise "COMPLETE"
```

### Best Practices

- **Always set `--max-iterations` AND `--completion-promise`**. Without max-iterations, the loop runs indefinitely. Without completion-promise, it won't know when to stop.
- **Watch the first 2–3 iterations** before going AFK. Cancel with `/cancel-ralph` if behavior looks wrong.
- **Completion promise uses exact string matching**. `<promise>DONE</promise>` must appear literally in Claude's output.
- **Each iteration sees modified files and git history**. The loop is self-correcting — Claude reads its own past work to inform improvements.
- **If stuck for >3 iterations on the same issue**: cancel, adjust the prompt, re-run. Don't waste iterations on a bad prompt.
- **Write prompts with explicit completion criteria and success metrics.** The clearer the "done" condition, the better Ralph performs.
- **Cost awareness**: each iteration consumes tokens. Monitor usage, especially on Max plan.
- **Re-run phases if results aren't satisfactory.** Ralph Loop is cheap compared to your time. A second run with a refined prompt often produces dramatically better results than accepting a mediocre first run.

### Fallback: Manual Bash Loop

If the plugin stops working mid-session (e.g., Colab restart loses plugin state):

```bash
#!/bin/bash
MAX=15
for i in $(seq 1 $MAX); do
    echo "=== Iteration $i / $MAX ==="
    claude -p "$(cat .ralph/PROMPT.md)" 2>&1 | tee /tmp/iter_$i.txt
    grep -q "DONE\|BLOCKED\|COMPLETE" /tmp/iter_$i.txt && break
done
```

---

## GPU Survival (Colab)

- Run a background keepalive thread (dummy tensor ops every 120s) to prevent GPU reclaim.
- Never let GPU idle between experiments — overlap CPU analysis with GPU execution.
- If no experiment is queued, run a lightweight profiling task.

---

## Decision Tree — GRPO Go / No-Go

```
Pre-Validation 1: Vision heads exist?
├── No (real ≈ black across all heads) → STOP. R_vhad will produce zero signal.
└── Yes → continue

Pre-Validation 2: Steering improves accuracy? (Δ on POPE)
├── Δ < 0.5 → DEBUG. Manual intervention doesn't help → R_vhad can't help.
└── Δ ≥ 0.5 → continue
    ├── Thinking Δ > Short Δ → R_vhad most valuable for thinking models ★
    ├── Thinking Δ ≈ Short Δ → R_vhad works, weaker thinking-mode story
    └── Thinking Δ < Short Δ → R_vhad works, drop thinking-mode narrative

Pre-Validation 3: Vision drift curve shows decay?
├── No decay in thinking mode → drift isn't a real problem, reconsider motivation
└── Decay visible, steering prevents it → core paper figure confirmed

Pre-Validation 4: Blind Test Gap increases with steering?
├── Gap decreases → steering makes model MORE blind (bad sign for R_vhad)
└── Gap maintained or increases → R_vhad will prevent blind reasoner collapse

All 4 passed → PROCEED TO R_vhad GRPO
  → Design: R_total = 0.3×R_correct + 0.5×R_vhad + 0.2×R_fluency
  → R_vhad: all heads, no pre-selection needed (validation shows which heads matter)
  → Train on Qwen3-VL-2B-Thinking, then extend to other models
  → Post-training: compare RL-strengthened heads vs steering-targeted heads

Any failed → FIX before GRPO, or PIVOT approach
```

---

## What This Instruction Does NOT Specify

- Exact hook implementation details (you have the codebase — read it)
- Specific function signatures or class structures (figure it out from existing code)
- Config file formats (follow whatever convention the project already uses)
- Exact plotting code (use matplotlib/seaborn, make it readable)

The codebase already has modules for profiling, calibration, steering, rewards, etc. Read them. Use them. Fix them if they're broken. The instruction tells you **what to find out**, not **how to write the code**.

**What comes after this instruction**: If all 4 pre-validations pass, a separate GRPO instruction will be written covering R_vhad reward implementation, training pipeline, and post-training mechanistic analysis. That instruction depends on the results from this one.
