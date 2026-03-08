---
description: Execute next experiment from the experiment log
agent: sisyphus
---
Execute the next experiment:
1. Read docs/EXPERIMENT_LOG.md — find last experiment and "Next" entry
2. Implement code changes
3. Check GPU memory before launching (`nvidia-smi`)
4. Run training/eval
5. Record results in docs/EXPERIMENT_LOG.md
6. Analyze and propose next experiment
7. Git commit + push
