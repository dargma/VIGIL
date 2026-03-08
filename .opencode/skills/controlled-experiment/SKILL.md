---
name: controlled-experiment
description: Controlled experiment protocol — one variable, systematic recording, decision rules for axis switching
---

# Controlled Experiment Protocol

## Rules
1. Change ONE variable at a time
2. Every experiment: hypothesis + expected outcome + actual outcome
3. Record ALL experiments (including failures) in docs/EXPERIMENT_LOG.md
4. 3 consecutive no-improvement → change axis (Architecture → Loss → Hyperparameter)
5. Quick eval (<5min) before full eval when possible

## Record Format
experiment_id: exp_NNN
hypothesis: [what and why]
changes: [exactly what changed]
config: [reference]
results: [metrics]
analysis: [why this happened]
next: [proposal]

## Decision Rules
- NaN → rollback + LR/10
- Collapse (>90% same answer) → check if binary VQA data, switch to mixed/open-ended
- OOM → reduce group_size first, then batch_size, then max_length
- Disk < 5GB → purge caches
- 3× no improvement → switch axis
- TRL GRPOTrainer → DO NOT USE for binary VQA at 2B scale (collapses 3/3 times)
