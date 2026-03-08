---
description: Run POPE evaluation on current or specified checkpoint
agent: sisyphus
---
Evaluate checkpoint on POPE:
1. Load model (default: checkpoints/block2_bon/final, or specify path)
2. Run `python scripts/eval_official_fast.py --max-samples 500`
3. Compare against previous results in docs/EXPERIMENT_LOG.md
4. Report: accuracy, F1, precision, recall, blind gap
5. Update docs/EXPERIMENT_LOG.md with results
