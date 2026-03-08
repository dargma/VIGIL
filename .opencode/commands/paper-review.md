---
description: Analyze a paper PDF and update the paper index
subtask: true
---
Analyze: papers/$1.pdf
1. pdftotext papers/$1.pdf -
2. Extract: motivation, method, key results, limitations
3. Relevance to VIGIL (reference docs/THEORY.md)
4. If applicable → propose experiment
5. Update papers/index.md
