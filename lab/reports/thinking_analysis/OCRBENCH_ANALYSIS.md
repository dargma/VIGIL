# OCRBench Thinking vs Short: 4-Quadrant Analysis

**Model**: Qwen3-VL-2B-Thinking
**Benchmark**: OCRBench (300 samples, 10 categories)
**Date**: 2026-03-14
**Prompts**: Thinking: `enable_thinking=True`, Short: `enable_thinking=False` + "Answer briefly."

---

## 1. Final Results

| Metric | Thinking | Short | Delta |
|--------|----------|-------|-------|
| Overall | **76.0%** (228/300) | 73.7% (221/300) | **+2.3pp** |
| text_rec | 83.9% | 81.7% | +2.2pp |
| doc_vqa | **86.7%** | 76.7% | **+10.0pp** |
| kie | **86.7%** | 80.0% | **+6.7pp** |
| scene_vqa | 50.0% | **53.3%** | **-3.3pp** |
| math_expr | 33.3% | **36.7%** | **-3.3pp** |

## 2. 4-Quadrant Matrix

|  | Reasoning ✓ | Reasoning ✗ |
|---|---|---|
| **Short ✓** | 216 (72.0%) | **5 (1.7%)** ← 핵심 분석 대상 |
| **Short ✗** | 12 (4.0%) | 67 (22.3%) |

- **Disagreements**: 17 total (5.7%)
- **Net advantage**: thinking wins 12, short wins 5 → net +7 for thinking

## 3. Priority 1: Short Wins (5 cases)

### Case 1 (idx=100): Handwriting — "Drugs" → "Dogs"
- **Image**: Cursive handwriting, ambiguous 'u'/'o'
- **Failure**: Thinking chain commits to "Dogs" early, then reinforces error
- **Thinking text**: "The letters seem to be 'Dogs' but wait, maybe it's 'Dogs' with a typo?"
- **Root cause**: **Self-reinforcing error** — model can't recover from initial misread
- **Tokens**: 114

### Case 2 (idx=102): Handwriting — "Lord" → "hord"
- **Image**: Cursive 'L' misread as 'h'
- **Failure**: Same pattern — early letter-by-letter analysis locks in wrong character
- **Tokens**: 123

### Case 3 (idx=195): Scene VQA — "Microsoft Windows XP"
- **Failure**: Think answer includes ® symbols, GT doesn't → substring mismatch
- **Root cause**: **Scoring artifact**, not genuine reasoning failure
- **Note**: Think answer "Microsoft® Windows® XP" is actually MORE accurate than GT

### Case 4 (idx=205): Scene VQA — "www.shutterstock.com 30031780"
- **Failure**: Think answer has extra ". " between URL and number
- **Root cause**: **Scoring artifact** (extra punctuation/space)

### Case 5 (idx=287): Math Expression — LaTeX formula
- **Failure**: Thinking chain = 0 tokens (couldn't start thinking)
- **Root cause**: **Truncation/generation failure**, not reasoning error

### Summary of Short-Win Patterns

| Pattern | Count | Mechanism |
|---------|-------|-----------|
| Overthinking corrupts perception | 2 | Self-reinforcing visual error |
| Scoring artifact | 2 | Special chars / formatting mismatch |
| Generation failure | 1 | Zero-length thinking chain |

**Only 2 of 5 short-wins are genuine reasoning failures.** Both are handwriting recognition where the thinking chain commits to an incorrect initial perception.

## 4. Priority 2: Think Wins (12 cases)

| Category | Count | Mechanism |
|----------|-------|-----------|
| Digit String Recognition | 3 | Systematic digit-by-digit verification |
| Doc VQA | 3 | Structured information extraction |
| Handwriting Recognition | 2 | Letter-by-letter analysis (successful) |
| Key Information Extraction | 2 | Multi-field document parsing |
| Scene VQA | 1 | Context-aware interpretation |
| Artistic Text | 1 | Style-aware character recognition |

Think wins when task requires **systematic visual inspection** (examining digits, fields, characters one by one). Short mode makes hasty errors (e.g., "52869" instead of "52868").

## 5. Category Analysis

- **doc_vqa** (+10.0pp): Strongest think advantage. Requires multi-step information extraction.
- **kie** (+6.7pp): Key information extraction benefits from structured thinking.
- **scene_vqa** (-3.3pp): Short slightly better. Scene text is more about quick recognition.
- **math_expr** (-3.3pp): Short slightly better. LaTeX generation is format-sensitive.

## 6. Implications

### For LSR Training
- The 2 genuine short-win cases show **micro-level visual attention drift**: early token misperception is reinforced through the thinking chain
- Per-token LSR on these cases would show: high image attention at chain start, then **committed to wrong interpretation** without re-checking image
- LSR training signal: upweight early-chain tokens where model should be most carefully attending to image

### For Self-Play DPO
- 5 short-win + 12 think-win = 17 natural preference pairs from 300 samples
- Scaling to full OCRBench (1000): ~57 pairs expected
- Combined with POPE: enough pairs for meaningful DPO training

### Key Insight
**Thinking helps most when tasks need systematic inspection (doc VQA, digits, handwriting). Thinking hurts when initial perception is wrong and the chain reinforces the error (handwriting edge cases). This is the micro version of O(1/L) visual attention drift.**
