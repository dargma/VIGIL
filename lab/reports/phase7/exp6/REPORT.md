# Phase 7: Exp5 (Learned Head Importance + KL)

- Steps: 15
- Group size: 6
- LR: 5e-07
- Importance LR: 0.001
- Importance temp: 2.0

## Evaluation Progression
| Step | TextVQA | POPE | Gap |
|------|---------|------|-----|
| Pre | 72.7% | 90.0% | 38.0% |
| 5 | 72.7% | 91.7% | 40.0% |
| 10 | 70.7% | 90.0% | 38.0% |
| 15 | 70.7% | 90.0% | 38.0% |

## Head KL Statistics
- Mean KL: 5.4897
- Max KL: 6.6044
- Min KL: 3.6557

## Learned Head Importance Evolution

### Step 5
Top-5 heads:
  - L5H0: 0.500
  - L4H6: 0.350
  - L23H2: 0.332
  - L2H9: 0.329
  - L5H7: 0.319

### Step 10
Top-5 heads:
  - L5H0: 0.500
  - L4H6: 0.350
  - L23H2: 0.332
  - L2H9: 0.329
  - L5H7: 0.319

### Step 15
Top-5 heads:
  - L5H0: 0.500
  - L4H6: 0.350
  - L23H2: 0.332
  - L2H9: 0.329
  - L5H7: 0.319