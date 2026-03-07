# Analysis 1: Attention Heatmap — Block 1 Sanity Check

Date: 2026-03-07T14:27:21.818467

Samples analyzed: 5

## Results

- **Sample 1**: Q="Is there a person in the image?..." | IIG=2.465 | KL=0.930 | Correct=True | Gen="Based on a careful examination..."
- **Sample 2**: Q="Is there a car in the image?..." | IIG=2.287 | KL=1.073 | Correct=False | Gen="Based on a careful examination..."
- **Sample 3**: Q="Is there a person in the image?..." | IIG=2.283 | KL=0.978 | Correct=True | Gen="Based on a thorough examinatio..."
- **Sample 4**: Q="Is there a truck in the image?..." | IIG=2.207 | KL=0.988 | Correct=False | Gen="Based on the image provided, t..."
- **Sample 5**: Q="Is there a book in the image?..." | IIG=2.201 | KL=1.008 | Correct=True | Gen="Based on a careful examination..."

## Summary Statistics
- Mean IIG: 2.289
- Mean attention KL (real vs black): 0.995

## Interpretation
- Attention shifts significantly between real and black images.
- High IIG confirms image contributes to answer.