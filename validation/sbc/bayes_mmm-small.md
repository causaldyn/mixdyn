# Simulation-based calibration: bayes_mmm, world small

Command: `python -m validation.sbc --engine bayes_mmm --world small --sims 200 --seed 0 --workers 5`, run 2026-11-10. Protocol: ADR 0059.

200 simulations, 200 fitted, 0 failed; ranks among 63 thinned draws of 4 chains of 1000 after 1000 of warm-up.

## Uniformity

Pearson's chi-square over 32 bins of the 64 ranks; a block's p-value is the least of its coordinates' times their number. The pass rule, at 200 simulations, is p > 0.01 for every block.

| block | coordinates | p |
|---|---|---|
| `log_roi` | 2 | 0.162 |
| `logit_retention` | 2 | 0.48 |
| `log_half_saturation` | 2 | 0.415 |
| `log_slope` | 2 | 0.152 |
| `level` | 1 | 0.89 |
| `log_sigma` | 1 | 0.755 |

## Rank histograms

Each bin's count in quarters of the 6.25 a uniform rank expects: `.` none, `:` a quarter, `-` a half, `=` three quarters, `+` as expected, and on to `@` at twice.

```
log_roi[tv]                  ++=*+:=#=-*#:=**-=***=:=+*-*::*-  p 0.596  [7, 7, 5, 8, 7, 2, 6, 10, 5, 4, 8, 10, 3, 6, 8, 9, 4, 5, 9, 9, 8, 6, 3, 5, 7, 8, 4, 9, 2, 3, 9, 4]
log_roi[search]              :%=+=-+*-=*%*+==+#.==:=@:=:*=-==  p 0.081  [3, 11, 6, 7, 6, 4, 7, 9, 4, 6, 8, 12, 8, 7, 5, 6, 7, 10, 1, 5, 6, 2, 5, 15, 3, 5, 3, 8, 5, 4, 6, 6]
logit_retention[tv]          =+=*+-=:=+#-%=-#+:**+--::%-+-#*=  p 0.240  [5, 7, 5, 9, 7, 4, 6, 3, 5, 7, 10, 4, 12, 6, 4, 10, 7, 2, 8, 8, 7, 4, 4, 2, 3, 12, 4, 7, 4, 10, 8, 6]
logit_retention[search]      *#*=*==***=+:==+=+=#:-=-:=-==+=+  p 0.981  [8, 10, 8, 6, 9, 6, 6, 8, 8, 8, 5, 7, 3, 6, 6, 7, 6, 7, 6, 10, 3, 4, 5, 4, 3, 6, 4, 5, 6, 7, 6, 7]
log_half_saturation[tv]      *+-==-*:-*=#=***==-=**--#=-=:=#*  p 0.811  [8, 7, 4, 5, 6, 4, 9, 3, 4, 8, 5, 10, 6, 9, 8, 8, 6, 5, 4, 5, 9, 8, 4, 4, 10, 6, 4, 5, 3, 5, 10, 8]
log_half_saturation[search]  :=+=*%=%#%+=+==---=-%==-=+:%==:=  p 0.208  [2, 6, 7, 5, 8, 12, 5, 12, 10, 11, 7, 5, 7, 5, 5, 4, 4, 4, 6, 4, 11, 5, 6, 4, 5, 7, 3, 11, 6, 5, 3, 5]
log_slope[tv]                -+=:-+=--#+======:==*-#=*##*-*+*  p 0.811  [4, 7, 6, 2, 4, 7, 5, 4, 4, 10, 7, 6, 6, 6, 6, 5, 6, 3, 5, 5, 9, 4, 10, 5, 8, 10, 10, 9, 4, 8, 7, 8]
log_slope[search]            %:=-%=:+=*:*+=:=-=*=*=@=*:.++**=  p 0.076  [11, 3, 6, 4, 11, 5, 3, 7, 5, 8, 3, 9, 7, 6, 2, 5, 4, 5, 8, 5, 8, 5, 15, 5, 9, 3, 1, 7, 7, 8, 9, 6]
level                        =*==-+*:+:===%*:=+#-++-*+++=+==:  p 0.890  [5, 8, 6, 6, 4, 7, 9, 3, 7, 3, 6, 6, 5, 12, 8, 3, 5, 7, 10, 4, 7, 7, 4, 9, 7, 7, 7, 6, 7, 6, 6, 3]
log_sigma                    +=#-++#==*.==+=***=%*:=-+++===-=  p 0.755  [7, 5, 10, 4, 7, 7, 10, 6, 5, 8, 0, 6, 6, 7, 5, 8, 8, 9, 6, 11, 8, 2, 5, 4, 7, 7, 7, 5, 5, 5, 4, 6]
```

## Divergences and health

Fits with a divergent transition: 1% (2 of 200); 4 divergent transitions in all, a fit's median 0 and most 3.
Fits with an R-hat above 1.01: 0; with a bulk ESS below 400: 1.

## Where the divergences fall

Share of fits that diverged among simulations whose true value is at or below the coordinate's median over the run, and above it; Spearman's correlation of a fit's divergent transitions with the true value; and the mean of the draws at the divergent transitions less the true value, over the fits that diverged.

| coordinate | diverged, true at or below median | above | Spearman | divergent less true |
|---|---|---|---|---|
| `log_roi[tv]` | 1% | 1% | -0.02 | +0.31 |
| `log_roi[search]` | 2% | 0% | -0.09 | -1.24 |
| `logit_retention[tv]` | 1% | 1% | +0.06 | -0.60 |
| `logit_retention[search]` | 2% | 0% | -0.04 | +0.91 |
| `log_half_saturation[tv]` | 0% | 2% | +0.11 | -0.87 |
| `log_half_saturation[search]` | 1% | 1% | -0.04 | -0.51 |
| `log_slope[tv]` | 0% | 2% | +0.11 | -0.60 |
| `log_slope[search]` | 1% | 1% | -0.06 | -0.41 |
| `level` | 1% | 1% | -0.04 | +0.17 |
| `log_sigma` | 0% | 2% | +0.11 | -0.05 |
