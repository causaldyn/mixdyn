# Simulation-based calibration: bayes_geo, world centred-geo, centred = 'search'

Command: `python -m validation.sbc --engine bayes_geo --world centred-geo --knob centred=search --sims 200 --seed 0 --workers 5`, run 2026-11-10. Protocol: ADR 0059.

200 simulations, 200 fitted, 0 failed; ranks among 63 thinned draws of 4 chains of 1000 after 1000 of warm-up.

## Uniformity

Pearson's chi-square over 32 bins of the 64 ranks; a block's p-value is the least of its coordinates' times their number. The pass rule, at 200 simulations, is p > 0.01 for every block.

| block | coordinates | p |
|---|---|---|
| `log_roi` | 2 | 1 |
| `logit_retention` | 2 | 1 |
| `log_half_saturation` | 2 | 0.48 |
| `log_slope` | 2 | 0.503 |
| `log_spread` | 2 | 0.0526 |
| `deviation` | 6 | 0.119 |
| `log_base` | 4 | 0.546 |
| `log_sigma` | 1 | 0.848 |

## Rank histograms

Each bin's count in quarters of the 6.25 a uniform rank expects: `.` none, `:` a quarter, `-` a half, `=` three quarters, `+` as expected, and on to `@` at twice.

```
log_roi[tv]                  =#=-=*==:+-==+-%++*===+=:+::+=#%  p 0.613  [6, 10, 6, 4, 5, 8, 5, 6, 3, 7, 4, 6, 6, 7, 4, 12, 7, 7, 9, 5, 6, 6, 7, 6, 3, 7, 2, 2, 7, 6, 10, 11]
log_roi[search]              =*--+%*==*:=:*==.=++==+*+:===%+#  p 0.546  [6, 8, 4, 4, 7, 11, 9, 5, 5, 9, 3, 6, 2, 8, 5, 5, 1, 5, 7, 7, 5, 6, 7, 8, 7, 3, 6, 6, 6, 12, 7, 10]
logit_retention[tv]          =+*==%==*#=*-+=-*:==*===#=:-+*=-  p 0.797  [6, 7, 8, 5, 5, 11, 5, 5, 8, 10, 6, 9, 4, 7, 6, 4, 8, 2, 5, 6, 9, 5, 6, 6, 10, 5, 2, 4, 7, 9, 6, 4]
logit_retention[search]      =*:+*=*#====-=++==+*=++*=-*:-**=  p 0.988  [5, 8, 2, 7, 8, 5, 8, 10, 5, 5, 5, 6, 4, 6, 7, 7, 6, 5, 7, 8, 6, 7, 7, 8, 6, 4, 9, 3, 4, 8, 8, 6]
log_half_saturation[tv]      #===*==*::=%*==%:==%=%=:=*-+-=:*  p 0.240  [10, 6, 5, 5, 9, 5, 5, 8, 2, 3, 6, 11, 9, 6, 6, 11, 3, 5, 5, 11, 5, 11, 5, 2, 6, 9, 4, 7, 4, 5, 3, 8]
log_half_saturation[search]  ==%==:=*%=:==%#-#+*+-=::%=-==-==  p 0.264  [5, 6, 12, 6, 6, 3, 5, 9, 11, 5, 3, 5, 5, 11, 10, 4, 10, 7, 9, 7, 4, 5, 3, 3, 11, 5, 4, 6, 5, 4, 6, 5]
log_slope[tv]                #:=#-=:=*=+*=::-==%=*:%++-=#*+-+  p 0.464  [10, 3, 5, 10, 4, 6, 3, 5, 8, 5, 7, 9, 6, 3, 3, 4, 5, 5, 11, 5, 9, 3, 11, 7, 7, 4, 6, 10, 8, 7, 4, 7]
log_slope[search]            -=:=:-=#+==+++=++-:.-%++#*@-#-++  p 0.252  [4, 6, 3, 5, 3, 4, 5, 10, 7, 6, 5, 7, 7, 7, 6, 7, 7, 4, 3, 1, 4, 12, 7, 7, 10, 8, 13, 4, 10, 4, 7, 7]
log_spread[tv]               ==%:*%=:=*=%:-**-=:-%:=-=+**=*:=  p 0.169  [5, 5, 11, 3, 9, 11, 5, 3, 6, 8, 5, 11, 3, 4, 8, 9, 4, 6, 3, 4, 12, 3, 6, 4, 6, 7, 9, 9, 5, 8, 2, 6]
log_spread[search]           @==.==+#--+:=-+:**%=+#=:*+=*:+=.  p 0.026  [16, 6, 5, 1, 6, 5, 7, 10, 4, 4, 7, 2, 6, 4, 7, 3, 8, 9, 11, 6, 7, 10, 6, 3, 9, 7, 6, 9, 3, 7, 5, 1]
deviation[tv,1]              =-**-=*=+-=*+*@==-:=*:-=**+-*:*#  p 0.357  [5, 4, 9, 8, 4, 6, 8, 5, 7, 4, 5, 8, 7, 8, 15, 5, 5, 4, 3, 5, 8, 2, 4, 5, 8, 8, 7, 4, 8, 3, 8, 10]
deviation[tv,2]              *.**+:@:=:+-==@--*=+*:*=+-%---#:  p 0.020  [9, 1, 9, 9, 7, 3, 13, 3, 6, 3, 7, 4, 6, 5, 13, 4, 4, 8, 5, 7, 9, 3, 9, 6, 7, 4, 12, 4, 4, 4, 10, 2]
deviation[tv,3]              +*%:%===+=::*+*++-%*:=*:=--===-+  p 0.546  [7, 8, 12, 3, 11, 5, 6, 5, 7, 6, 3, 3, 9, 7, 9, 7, 7, 4, 11, 8, 3, 6, 8, 3, 5, 4, 4, 6, 6, 6, 4, 7]
deviation[search,1]          +***+=*=-*#++:-*-==--*%+==+=-+=.  p 0.755  [7, 9, 9, 8, 7, 5, 8, 6, 4, 8, 10, 7, 7, 2, 4, 9, 4, 5, 6, 4, 4, 8, 11, 7, 5, 6, 7, 5, 4, 7, 6, 1]
deviation[search,2]          =:==-+=*=-+-==:=*#=--*+=*-%*+#*=  p 0.870  [6, 3, 6, 5, 4, 7, 5, 9, 6, 4, 7, 4, 5, 5, 3, 6, 9, 10, 5, 4, 4, 8, 7, 6, 8, 4, 11, 8, 7, 10, 8, 6]
deviation[search,3]          =:-%-=-:=:*:*=@#=-=+-**:+=**:*=%  p 0.086  [5, 3, 4, 11, 4, 5, 4, 2, 6, 3, 9, 3, 8, 5, 14, 10, 5, 4, 6, 7, 4, 8, 9, 3, 7, 6, 9, 9, 3, 8, 5, 11]
log_base[geo0]               =+-*===*=+::+%=:#=-=-*##=+::=%*-  p 0.343  [6, 7, 4, 9, 6, 6, 5, 9, 6, 7, 2, 2, 7, 11, 5, 3, 10, 6, 4, 6, 4, 9, 10, 10, 5, 7, 3, 3, 5, 11, 8, 4]
log_base[geo1]               -*:.=-+-=@=**+*#++*-%=-====*:-:*  p 0.136  [4, 8, 3, 1, 5, 4, 7, 4, 5, 15, 5, 8, 9, 7, 9, 10, 7, 7, 8, 4, 11, 6, 4, 6, 5, 5, 6, 8, 3, 4, 3, 9]
log_base[geo2]               +-%-=#==-==**%:*+-==::+:#*.===#+  p 0.329  [7, 4, 11, 4, 6, 10, 6, 5, 4, 6, 6, 8, 9, 11, 2, 9, 7, 4, 5, 6, 3, 3, 7, 3, 10, 8, 1, 6, 6, 6, 10, 7]
log_base[geo3]               +==-*#.*=%*++==:=#::=*=:===+#+%=  p 0.252  [7, 5, 5, 4, 8, 10, 0, 8, 5, 11, 9, 7, 7, 5, 6, 3, 5, 10, 3, 2, 6, 9, 6, 2, 6, 5, 5, 7, 10, 7, 11, 6]
log_sigma                    ==+==+:*-+=*==+#++**=+==-=*+=-.#  p 0.848  [5, 5, 7, 5, 5, 7, 2, 9, 4, 7, 5, 9, 6, 5, 7, 10, 7, 7, 9, 9, 6, 7, 5, 6, 4, 5, 9, 7, 6, 4, 1, 10]
```

## Divergences and health

Fits with a divergent transition: 100% (200 of 200); 54730 divergent transitions in all, a fit's median 211 and most 1385.
Fits with an R-hat above 1.01: 119; with a bulk ESS below 400: 110.

## Where the divergences fall

Share of fits that diverged among simulations whose true value is at or below the coordinate's median over the run, and above it; Spearman's correlation of a fit's divergent transitions with the true value; and the mean of the draws at the divergent transitions less the true value, over the fits that diverged.

| coordinate | diverged, true at or below median | above | Spearman | divergent less true |
|---|---|---|---|---|
| `log_roi[tv]` | 100% | 100% | +0.12 | -0.07 |
| `log_roi[search]` | 100% | 100% | +0.04 | -0.11 |
| `logit_retention[tv]` | 100% | 100% | +0.15 | +0.10 |
| `logit_retention[search]` | 100% | 100% | +0.03 | -0.07 |
| `log_half_saturation[tv]` | 100% | 100% | +0.08 | +0.08 |
| `log_half_saturation[search]` | 100% | 100% | +0.15 | +0.04 |
| `log_slope[tv]` | 100% | 100% | +0.13 | -0.01 |
| `log_slope[search]` | 100% | 100% | -0.10 | -0.06 |
| `log_spread[tv]` | 100% | 100% | +0.03 | -0.01 |
| `log_spread[search]` | 100% | 100% | -0.09 | -1.13 |
| `deviation[tv,1]` | 100% | 100% | -0.01 | -0.07 |
| `deviation[tv,2]` | 100% | 100% | +0.00 | +0.12 |
| `deviation[tv,3]` | 100% | 100% | -0.01 | +0.11 |
| `deviation[search,1]` | 100% | 100% | -0.03 | +0.14 |
| `deviation[search,2]` | 100% | 100% | -0.03 | -0.07 |
| `deviation[search,3]` | 100% | 100% | +0.01 | -0.14 |
| `log_base[geo0]` | 100% | 100% | -0.06 | -0.00 |
| `log_base[geo1]` | 100% | 100% | -0.18 | -0.01 |
| `log_base[geo2]` | 100% | 100% | -0.10 | -0.03 |
| `log_base[geo3]` | 100% | 100% | -0.07 | -0.00 |
| `log_sigma` | 100% | 100% | -0.19 | +0.00 |
