# Skill: Autonomous Research Loop (from Karpathy's autoresearch)

## Core Pattern: Keep/Discard Hill Climbing

```
LOOP FOREVER:
  1. Read current state (results.tsv, git log)
  2. Propose experiment (modify hyperparams, reward design, data mix)
  3. git commit the change
  4. Run experiment (FIXED time budget, e.g., 15 min)
  5. Parse single scalar metric
  6. If improved → KEEP (advance)
  7. If worse → DISCARD (git reset)
  8. Log to results.tsv
  9. NEVER STOP — think harder if stuck
```

## Key Design Principles

### 1. Fixed Time Budget
Every experiment gets exactly the same wall clock time (e.g., 15 min = ~20 GRPO steps).
Makes results directly comparable regardless of what was changed.

### 2. Single Scalar Metric
Pick ONE metric for keep/discard. For VIGIL:
```
score = 0.6 * TextVQA_acc + 0.4 * Blind_Gap_normalized
```
Multiple metrics are tracked but ONE decides.

### 3. Separation of Concerns
- `research_program.md` — human's instructions (what to try, what to avoid)
- `train.py` — agent's editable code
- `eval.py` — fixed, untouchable evaluation
- `results.tsv` — machine-parseable log

### 4. Crash Resilience
- Crashes logged as status="crash", experiment discarded
- OOM → halve batch, retry once
- Timeout → kill, log, move on
- NEVER wait for human intervention

### 5. Simplicity Criterion
- 0.001 improvement + 20 lines of hacky code → NOT worth it
- Deleting code for equal results → KEEP
- Prevent complexity accumulation over 100+ experiments

## results.tsv Format

```tsv
commit	metric	pope_acc	blind_gap	textvqa_acc	status	description	timestamp
abc1234	0.523	0.917	0.400	0.312	keep	baseline from HF	2026-03-14T12:00
def5678	0.531	0.933	0.420	0.325	keep	head-mask alpha=0.5, lr=2e-6	2026-03-14T12:20
ghi9012	0.518	0.900	0.380	0.310	discard	head-mask alpha=1.0 (too aggressive)	2026-03-14T12:40
```

## VIGIL-Specific Application

### Experiment Dimensions to Search
1. `alpha` (head-level weight): [0.0, 0.3, 0.5, 0.7, 1.0]
2. `lr`: [5e-7, 1e-6, 2e-6, 5e-6]
3. `top_k_heads`: [6, 12, 20]
4. `beta_decay`: [0.0, 0.1, 0.3]
5. `lsr_scale`: [5.0, 10.0, 20.0]
6. `temperature`: [1.0, 1.3, 1.5]

### Keep/Discard Rules
- Keep: composite_score improves by ≥ 0.005
- Discard: composite_score decreases or improves by < 0.005
- Crash: log and move on, don't retry same config

### Integration with auto_lab.py
The existing `scripts/auto_lab.py` state machine can be extended with:
1. `results.tsv` logging
2. Git-based checkpointing (commit before, reset on discard)
3. Configurable experiment queue from `research_program.md`
