# Final Results — post-fix, clean re-run

All numbers are mean episode return ± std across seeds (higher = better; Simple Spread returns are negative by construction). Stds are ddof=1 (unbiased sample std) to match the methodology note in §4.

## Simple Spread (N=3)

| Condition          | Return  | ±    | seeds | Notes |
|--------------------|---------|------|-------|-------|
| no-comm            | -21.99   | 1.73 | 3     |  |
| fc-comm            | -18.70   | 0.81 | 3     |  |
| attention-comm     | -19.29   | 1.04 | 3     |  |
| bandwidth (bw8)    | -18.44   | 0.47 | 3     |  |
| info-bottleneck    | -18.59   | 0.93 | 3     |  |
| noise σ=0.1        | -19.76   | 1.00 | 3     |  |
| noise σ=0.25       | -18.55   | 0.74 | 3     |  |
| noise σ=0.5        | -19.10   | 0.44 | 3     |  |
| GNN-comm           | -19.43   | 0.08 | 3     |  |

## Speaker-Listener (N=2)

| Condition          | Return  | ±    | seeds | Notes |
|--------------------|---------|------|-------|-------|
| sl-none            | -13.97   | 0.35 | 3     |  |
| sl-fc              | -14.94   | 0.43 | 3     |  |
| sl-attn            | -14.91   | 0.34 | 3     |  |

## What the numbers say

_(Interpretation paragraphs to be re-added by hand from the previous version of this file.)_

