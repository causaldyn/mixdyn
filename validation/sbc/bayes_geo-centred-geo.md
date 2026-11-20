# Simulation-based calibration: bayes_geo, world centred-geo

Command: `python -m validation.sbc --engine bayes_geo --world centred-geo --sims 200 --seed 0 --workers 5`, run 2026-11-10. Protocol: ADR 0059.

200 simulations, 200 fitted, 0 failed; ranks among 63 thinned draws of 4 chains of 1000 after 1000 of warm-up.

## Uniformity

Pearson's chi-square over 32 bins of the 64 ranks; a block's p-value is the least of its coordinates' times their number. The pass rule, at 200 simulations, is p > 0.01 for every block.

| block | coordinates | p |
|---|---|---|
| `log_roi` | 2 | 0.527 |
| `logit_retention` | 2 | 0.143 |
| `log_half_saturation` | 2 | 1 |
| `log_slope` | 2 | 0.685 |
| `log_spread` | 2 | 0.0235 |
| `deviation` | 6 | 0.819 |
| `log_base` | 4 | 1 |
| `log_sigma` | 1 | 0.302 |

## Rank histograms

Each bin's count in quarters of the 6.25 a uniform rank expects: `.` none, `:` a quarter, `-` a half, `=` three quarters, `+` as expected, and on to `@` at twice.

```
log_roi[tv]                  ==#%-=:=-.=-=*-++#%*=*:=--*=+=*#  p 0.448  [6, 5, 10, 11, 4, 5, 3, 5, 4, 1, 6, 4, 5, 8, 4, 7, 7, 10, 11, 8, 6, 9, 3, 5, 4, 4, 8, 6, 7, 6, 8, 10]
log_roi[search]              =+=:+%*==---*=-+==-@:*-:=:+#+@*=  p 0.264  [6, 7, 6, 3, 7, 11, 8, 5, 5, 4, 4, 4, 8, 6, 4, 7, 6, 5, 4, 13, 3, 8, 4, 3, 6, 3, 7, 10, 7, 13, 8, 5]
logit_retention[tv]          =#-=@*-=+**====*:+::=**==#::@-:=  p 0.071  [5, 10, 4, 6, 13, 9, 4, 6, 7, 8, 9, 5, 5, 5, 5, 8, 3, 7, 3, 2, 6, 9, 8, 5, 6, 10, 2, 3, 14, 4, 3, 6]
logit_retention[search]      -*=.=*=@-*:=:=+==*-*+%=:+#+=-%=-  p 0.115  [4, 9, 6, 1, 6, 9, 5, 13, 4, 9, 3, 5, 2, 5, 7, 6, 5, 8, 4, 8, 7, 12, 5, 2, 7, 10, 7, 6, 4, 11, 6, 4]
log_half_saturation[tv]      %-=#=:*==+=*+*=-**==+++:#==+:--=  p 0.755  [11, 4, 6, 10, 5, 2, 8, 5, 5, 7, 6, 9, 7, 9, 6, 4, 8, 9, 6, 5, 7, 7, 7, 2, 10, 5, 6, 7, 3, 4, 4, 6]
log_half_saturation[search]  =**===+*=+=*:-#**-*=*::=:*===*:+  p 0.836  [6, 8, 9, 6, 6, 6, 7, 8, 6, 7, 6, 8, 3, 4, 10, 9, 8, 4, 9, 6, 8, 3, 2, 6, 3, 8, 5, 5, 6, 9, 2, 7]
log_slope[tv]                ++=+=+=+=#-==:===*=:++*=+**=*===  p 0.999  [7, 7, 6, 7, 6, 7, 5, 7, 5, 10, 4, 5, 5, 3, 6, 6, 5, 9, 6, 3, 7, 7, 8, 6, 7, 8, 8, 5, 8, 5, 6, 6]
log_slope[search]            :*:::*=#=+*=+:=*===:+:=%*+%#*-+=  p 0.343  [3, 8, 3, 3, 2, 8, 6, 10, 5, 7, 8, 5, 7, 3, 5, 8, 5, 6, 6, 3, 7, 3, 6, 12, 8, 7, 12, 10, 8, 4, 7, 5]
log_spread[tv]               @@.=+:*+=*==++-*=:+:=*+-*.+*=*:=  p 0.012  [15, 14, 0, 6, 7, 3, 9, 7, 5, 9, 6, 6, 7, 7, 4, 8, 6, 2, 7, 2, 5, 8, 7, 4, 9, 1, 7, 8, 5, 8, 2, 6]
log_spread[search]           **=:-+*:#+*::**=#=+===#*+**-:=::  p 0.464  [9, 8, 6, 3, 4, 7, 8, 2, 10, 7, 8, 3, 3, 8, 8, 6, 10, 5, 7, 6, 5, 5, 10, 9, 7, 9, 9, 4, 3, 6, 3, 2]
deviation[tv,1]              =-#=:+=-=:*-%**=##+--::*=:*#:-%=  p 0.188  [6, 4, 10, 5, 3, 7, 5, 4, 6, 3, 8, 4, 11, 9, 9, 6, 10, 10, 7, 4, 4, 3, 3, 8, 6, 2, 9, 10, 3, 4, 11, 6]
deviation[tv,2]              =:#-+-+-+=**--.**#*#:*:=++**=:-=  p 0.464  [5, 3, 10, 4, 7, 4, 7, 4, 7, 6, 9, 9, 4, 4, 1, 9, 8, 10, 8, 10, 3, 9, 3, 6, 7, 7, 9, 8, 6, 3, 4, 6]
deviation[tv,3]              *++:=+*-*-+*%:%*=*==:-*-**%-:::=  p 0.218  [8, 7, 7, 3, 5, 7, 8, 4, 9, 4, 7, 8, 12, 3, 12, 9, 6, 8, 5, 5, 3, 4, 8, 4, 8, 8, 11, 4, 3, 2, 3, 5]
deviation[search,1]          -+*%*#=+=:==*-#%.:+=+-=#+=:=+=+.  p 0.136  [4, 7, 9, 12, 9, 10, 6, 7, 5, 2, 5, 6, 8, 4, 10, 12, 1, 3, 7, 6, 7, 4, 5, 10, 7, 6, 3, 5, 7, 5, 7, 1]
deviation[search,2]          =+-=::#:+*:=-:+#+--=***=-%*#+==*  p 0.357  [6, 7, 4, 6, 2, 3, 10, 3, 7, 9, 2, 5, 4, 3, 7, 10, 7, 4, 4, 5, 8, 9, 8, 6, 4, 12, 9, 10, 7, 6, 5, 8]
deviation[search,3]          =:+:===-=*-.*=%*##:*=+=*-%=-=-**  p 0.252  [5, 3, 7, 3, 6, 6, 5, 4, 5, 9, 4, 0, 8, 5, 11, 9, 10, 10, 3, 8, 6, 7, 5, 8, 4, 12, 6, 4, 6, 4, 8, 9]
log_base[geo0]               =-=+=+#+=:-==*.+:**+*#*#*.--==**  p 0.480  [6, 4, 5, 7, 6, 7, 10, 7, 6, 3, 4, 5, 5, 8, 1, 7, 3, 9, 8, 7, 8, 10, 9, 10, 9, 1, 4, 4, 5, 5, 8, 9]
log_base[geo1]               =*:=-=-+*+*+==*#+:#==@*=-:=+=.*=  p 0.464  [5, 8, 3, 5, 4, 5, 4, 7, 8, 7, 8, 7, 5, 6, 9, 10, 7, 3, 10, 6, 6, 13, 9, 6, 4, 2, 6, 7, 6, 1, 8, 5]
log_base[geo2]               *+==-*==++=+=*#*=+:==:=---#%-*-*  p 0.823  [8, 7, 5, 6, 4, 8, 6, 6, 7, 7, 6, 7, 5, 8, 10, 9, 5, 7, 3, 6, 5, 2, 5, 4, 4, 4, 10, 11, 4, 9, 4, 8]
log_base[geo3]               +===-==*=%#-:*=-*-::-+=%.*-==#**  p 0.276  [7, 5, 6, 6, 4, 6, 6, 8, 6, 11, 10, 4, 3, 9, 6, 4, 9, 4, 3, 3, 4, 7, 6, 12, 0, 8, 4, 6, 6, 10, 8, 9]
log_sigma                    ++--=*+*--:#+-:+#*+=@*:+*=%=:-:+  p 0.302  [7, 7, 4, 4, 5, 9, 7, 8, 4, 4, 3, 10, 7, 4, 3, 7, 10, 8, 7, 5, 13, 8, 3, 7, 8, 6, 11, 5, 2, 4, 3, 7]
```

## Divergences and health

Fits with a divergent transition: 100% (200 of 200); 57649 divergent transitions in all, a fit's median 250.5 and most 1036.
Fits with an R-hat above 1.01: 158; with a bulk ESS below 400: 135.

## Where the divergences fall

Share of fits that diverged among simulations whose true value is at or below the coordinate's median over the run, and above it; Spearman's correlation of a fit's divergent transitions with the true value; and the mean of the draws at the divergent transitions less the true value, over the fits that diverged.

| coordinate | diverged, true at or below median | above | Spearman | divergent less true |
|---|---|---|---|---|
| `log_roi[tv]` | 100% | 100% | +0.05 | -0.05 |
| `log_roi[search]` | 100% | 100% | +0.04 | -0.06 |
| `logit_retention[tv]` | 100% | 100% | -0.16 | +0.10 |
| `logit_retention[search]` | 100% | 100% | +0.03 | -0.08 |
| `log_half_saturation[tv]` | 100% | 100% | -0.02 | +0.09 |
| `log_half_saturation[search]` | 100% | 100% | -0.06 | +0.08 |
| `log_slope[tv]` | 100% | 100% | +0.14 | -0.01 |
| `log_slope[search]` | 100% | 100% | -0.08 | -0.08 |
| `log_spread[tv]` | 100% | 100% | -0.10 | -0.47 |
| `log_spread[search]` | 100% | 100% | -0.10 | -0.57 |
| `deviation[tv,1]` | 100% | 100% | +0.04 | -0.06 |
| `deviation[tv,2]` | 100% | 100% | -0.04 | -0.01 |
| `deviation[tv,3]` | 100% | 100% | -0.05 | +0.15 |
| `deviation[search,1]` | 100% | 100% | -0.05 | +0.15 |
| `deviation[search,2]` | 100% | 100% | -0.06 | -0.11 |
| `deviation[search,3]` | 100% | 100% | +0.05 | -0.15 |
| `log_base[geo0]` | 100% | 100% | -0.05 | -0.01 |
| `log_base[geo1]` | 100% | 100% | -0.15 | -0.01 |
| `log_base[geo2]` | 100% | 100% | +0.07 | -0.02 |
| `log_base[geo3]` | 100% | 100% | -0.04 | -0.00 |
| `log_sigma` | 100% | 100% | +0.05 | -0.00 |
