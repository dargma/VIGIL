instruction2.md — Supplementary Directive

> Read after instruction.md. This addresses gaps in the current implementation.

---

## The Core Problem

You wrote Auto Lab as philosophy. It needs to be **runnable code**. You wrote skills as a section in a document. They need to be **actual files**. You wrote smoke tests as a plan. They need to be **a script that runs**.

Documents that describe what should happen are not the same as code that makes it happen.

---

## 1. Auto Lab Must Be a Real Loop

The `Observe → Hypothesize → Experiment → Analyze → Iterate` loop is currently text in CLAUDE.md. Build `scripts/auto_lab.py` that actually does this:

- Reads the last entry in `lab/RESEARCH_JOURNAL.md` to understand current state
- Reads `lab/results/` to find the most recent experiment outputs
- Decides what to run next based on a priority queue (calibration → baseline → steering → GRPO → ablation)
- Runs it
- Logs the outcome
- Loops

This doesn't need to be AGI. A simple state machine is fine:
```
STATE_ORDER = [calibrate, baseline_eval, steering_eval, grpo_train, grpo_eval, blind_test, ablation, ...]
current_state = read_from_journal()
next_state = determine_next(current_state, last_results)
run(next_state)
```

When something fails, the loop should **diagnose and adapt**, not just skip. If OOM → reduce batch. If reward variance is zero → log the hypothesis for why and try a different temperature or dataset subset.

---

## 2. Smoke Test Is the First Thing That Runs on GPU

Before anything else on GPU, `scripts/smoke_test.py` must execute and pass. It validates the entire pipeline foundation in under 5 minutes:

What it checks:
- Model loads, config matches expected architecture (layers, heads, head_dim, hidden_size)
- o_proj hook registers successfully, activation shape is [batch, seq, num_Q_heads × head_dim]
- Single image forward pass produces coherent output
- **Real image vs black image: vision head activation difference is non-trivial** (if Δ ≈ 0, R_vhad will be useless and the entire reward design needs rethinking)
- Hook-based activation collection works during generation (for lightweight reward)

If any check fails, the script prints exactly what went wrong and what to investigate. No silent failures. No proceeding to calibration with a broken foundation.

---

## 3. Skills Are Files, Not Plans

Create `/content/drive/MyDrive/VIGIL/skills/` now. Not after success — the directory structure should exist from the start.

Then: every time you solve a non-trivial problem (model loading quirk, GQA reshape that works, TRL workaround), immediately write the skill file. Don't batch them. Don't plan to do it later.

The test: if you had to redo this project from scratch in a new environment, could you read the skills and get to the same point in half the time? If not, you haven't captured enough.

---

## 4. Two Reward Configurations Must Both Be Implemented

The current code only has the full reward path. Implement both:

**Full reward** (R_vhad + R_asi): +1 forward pass with black image, +1 generation without image. ~25% overhead. Maximum blind-reasoner prevention.

**Lightweight reward** (in-situ only): Hooks collect vision head activation magnitude during normal GRPO generation. Zero extra forward passes. Less precise but free.

Run GRPO with each. Compare: accuracy, blind test Gap, compute time. This comparison is itself a paper contribution — "when is the full reward worth the cost?"

---

## 5. VISTA Combination Is an Experiment, Not Just a Talking Point

The competitive strategy says "VIGIL + VISTA combined should beat either alone." This claim requires an actual experiment:

1. Reproduce VISTA's inference-time intervention on Qwen3-VL-2B (their code is public)
2. Run VISTA alone on POPE
3. Run VIGIL GRPO model (steering off) on POPE
4. Run VIGIL GRPO model + VISTA on POPE
5. Report all four numbers

If you can't reproduce VISTA in reasonable time, at minimum run the "permanent vs transient" experiment: VIGIL GRPO with steering ON vs OFF. This proves the RL training created lasting improvement without any inference overhead.

---

## 6. The Blind Test Is Not Optional

The Blind Test (Section 3.5 of CLAUDE.md) is the **single most important experiment** in the paper. It's the one result that no competing method can claim. Implement it early, not as an afterthought.

Specifically: after each GRPO training run, immediately run the blind test. Don't wait until "final evaluation." The Gap metric should be tracked alongside training reward as a health indicator — if Gap is shrinking during training, the model is becoming a blind reasoner and you need to stop and fix the reward.

---

## 7. What's Missing from the Journal

The RESEARCH_JOURNAL currently logs what was built. It should also log **decisions and reasoning**:

- Why did you choose these hyperparameters?
- What did you expect to happen vs what actually happened?
- When something failed, what was your hypothesis for why?

This is what makes the journal useful for writing the paper later. "Module implemented" is not a research finding. "Real vs black image activation Δ was 0.3 on layer 20 head 8, confirming vision specialization" is.

---

## Summary

| Gap | What exists now | What should exist |
|-----|----------------|-------------------|
| Auto Lab | Text in CLAUDE.md | `scripts/auto_lab.py` that runs the loop |
| Smoke Test | A plan in someone's head | `scripts/smoke_test.py` that validates on GPU |
| Skills | A section in instruction.md | Actual SKILL.md files in `skills/` |
| Full vs Lightweight reward | Only full reward in rewards.py | Both implemented, compared |
| VISTA combination | A talking point | An experiment with numbers |
| Blind Test | Implemented but not integrated into training loop | Run after every GRPO training, track Gap |
| Journal quality | Logs actions | Logs reasoning and findings |
