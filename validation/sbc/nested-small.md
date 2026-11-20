# Simulation-based calibration: nested, world small

Command: `python -m validation.sbc --engine nested --world small --sims 200 --seed 0 --workers 5`, run 2026-11-10. Protocol: ADR 0059.

200 simulations, 200 fitted, 0 failed; ranks among 63 thinned draws of 4 chains of 1000 after 1000 of warm-up.

## Uniformity

Pearson's chi-square over 32 bins of the 64 ranks; a block's p-value is the least of its coordinates' times their number. The pass rule, at 200 simulations, is p > 0.01 for every block.

| block | coordinates | p |
|---|---|---|
| `one.log_roi` | 1 | 0.662 |
| `one.logit_retention` | 1 | 0.932 |
| `one.log_half_saturation` | 1 | 0.188 |
| `one.log_slope` | 1 | 0.264 |
| `one.level` | 1 | 0.371 |
| `one.log_sigma` | 1 | 0.497 |
| `two.log_roi` | 1 | 0.53 |
| `two.logit_retention` | 1 | 0.161 |
| `two.log_half_saturation` | 1 | 0.563 |
| `two.log_slope` | 1 | 0.917 |
| `two.level` | 1 | 0.546 |
| `two.control` | 1 | 0.74 |
| `two.log_sigma` | 1 | 0.432 |

## Rank histograms

Each bin's count in quarters of the 6.25 a uniform rank expects: `.` none, `:` a quarter, `-` a half, `=` three quarters, `+` as expected, and on to `@` at twice.

```
one.log_roi[tv]                  =::=%*=+***====+=%-:**=-=-=+**:=  p 0.662  [6, 3, 3, 5, 11, 8, 6, 7, 8, 8, 8, 6, 5, 6, 5, 7, 6, 12, 4, 3, 9, 8, 5, 4, 5, 4, 5, 7, 9, 9, 2, 6]
one.logit_retention[tv]          =+-*-#==--*=-:++=+**==-#=*=:+#*=  p 0.932  [6, 7, 4, 8, 4, 10, 6, 6, 4, 4, 8, 5, 4, 3, 7, 7, 5, 7, 9, 8, 6, 6, 4, 10, 6, 8, 5, 3, 7, 10, 8, 5]
one.log_half_saturation[tv]      *+-*%*%=--*-@#===-+==:=:=-=+:+-=  p 0.188  [9, 7, 4, 9, 11, 9, 11, 6, 4, 4, 8, 4, 14, 10, 5, 6, 5, 4, 7, 6, 6, 2, 5, 3, 6, 4, 5, 7, 2, 7, 4, 6]
one.log_slope[tv]                =-===-:===*%++==:=-==%*:=%:%**==  p 0.264  [6, 4, 5, 6, 5, 4, 3, 5, 5, 5, 9, 12, 7, 7, 6, 6, 3, 5, 4, 5, 6, 11, 9, 2, 6, 11, 2, 12, 8, 9, 6, 6]
one.level                        *==*-*=:-===*=**+-=%=:%:+**+:*::  p 0.371  [8, 6, 5, 8, 4, 8, 5, 3, 4, 5, 6, 6, 9, 6, 9, 8, 7, 4, 6, 11, 6, 2, 12, 3, 7, 9, 9, 7, 3, 9, 2, 3]
one.log_sigma                    +*+=*=@=======:*%*-**=:=*:=+:-+=  p 0.497  [7, 9, 7, 6, 8, 5, 13, 6, 5, 5, 6, 5, 5, 5, 3, 9, 11, 9, 4, 9, 8, 5, 3, 6, 8, 2, 5, 7, 2, 4, 7, 6]
two.log_roi[search]              **=%=%::=*==+*-*+*:-#==:++-=++-=  p 0.530  [9, 9, 5, 12, 5, 11, 3, 3, 5, 8, 5, 5, 7, 9, 4, 8, 7, 8, 3, 4, 10, 6, 5, 2, 7, 7, 4, 5, 7, 7, 4, 6]
two.logit_retention[search]      =**=*=:**+*+::-=@==::=*+%*%:+=:=  p 0.161  [5, 9, 8, 5, 8, 6, 3, 8, 8, 7, 8, 7, 3, 2, 4, 5, 13, 6, 5, 2, 3, 6, 8, 7, 11, 8, 12, 2, 7, 5, 3, 6]
two.log_half_saturation[search]  +=====:-:*-=*+=+=*:=*+@*==+=:=+*  p 0.563  [7, 6, 5, 5, 5, 6, 3, 4, 3, 8, 4, 6, 8, 7, 5, 7, 5, 8, 3, 6, 8, 7, 16, 9, 6, 6, 7, 6, 3, 6, 7, 8]
two.log_slope[search]            =**==-+=*---=*=+*%+:-**=-+*=====  p 0.917  [5, 8, 8, 5, 5, 4, 7, 5, 8, 4, 4, 4, 6, 9, 6, 7, 9, 12, 7, 3, 4, 9, 8, 6, 4, 7, 9, 5, 6, 5, 6, 5]
two.level                        *=+:=-+===#=:==:+@==#-+*=+=%*::=  p 0.546  [8, 6, 7, 3, 6, 4, 7, 6, 6, 5, 10, 5, 3, 6, 5, 3, 7, 13, 6, 6, 10, 4, 7, 9, 5, 7, 5, 11, 8, 3, 3, 6]
two.control[brand]               *+%=++%=:*=--+===+=+:-#*-:+==+=*  p 0.740  [9, 7, 11, 6, 7, 7, 11, 5, 3, 9, 5, 4, 4, 7, 5, 5, 6, 7, 5, 7, 3, 4, 10, 9, 4, 2, 7, 6, 5, 7, 5, 8]
two.log_sigma                    +-+--#+---=:=*-@+*+:=*+#=:-#=-*+  p 0.432  [7, 4, 7, 4, 4, 10, 7, 4, 4, 4, 6, 3, 6, 9, 4, 13, 7, 8, 7, 3, 6, 9, 7, 10, 6, 2, 4, 10, 6, 4, 8, 7]
```

## Divergences and health

Fits with a divergent transition: 6% (13 of 200); 30 divergent transitions in all, a fit's median 0 and most 14.
Fits with an R-hat above 1.01: 2; with a bulk ESS below 400: 1.

## Where the divergences fall

Share of fits that diverged among simulations whose true value is at or below the coordinate's median over the run, and above it; Spearman's correlation of a fit's divergent transitions with the true value; and the mean of the draws at the divergent transitions less the true value, over the fits that diverged.

| coordinate | diverged, true at or below median | above | Spearman | divergent less true |
|---|---|---|---|---|
| `one.log_roi[tv]` | 7% | 6% | -0.04 | +0.14 |
| `one.logit_retention[tv]` | 8% | 5% | -0.03 | +0.81 |
| `one.log_half_saturation[tv]` | 6% | 7% | +0.05 | -0.35 |
| `one.log_slope[tv]` | 6% | 7% | +0.04 | +0.34 |
| `one.level` | 6% | 7% | -0.04 | -0.01 |
| `one.log_sigma` | 6% | 7% | +0.03 | -0.03 |
| `two.log_roi[search]` | 9% | 4% | -0.14 | +0.03 |
| `two.logit_retention[search]` | 5% | 8% | +0.07 | +0.04 |
| `two.log_half_saturation[search]` | 7% | 6% | -0.12 | +0.37 |
| `two.log_slope[search]` | 5% | 8% | +0.02 | +0.02 |
| `two.level` | 8% | 5% | -0.10 | -0.05 |
| `two.control[brand]` | 7% | 6% | +0.00 | -0.03 |
| `two.log_sigma` | 7% | 6% | -0.03 | +0.00 |
