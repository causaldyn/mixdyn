# Simulation-based calibration: bayes_geo, world centred-geo, centred = 'tv'

Command: `python -m validation.sbc --engine bayes_geo --world centred-geo --knob centred=tv --sims 200 --seed 0 --workers 5`, run 2026-11-10. Protocol: ADR 0059.

200 simulations, 200 fitted, 0 failed; ranks among 63 thinned draws of 4 chains of 1000 after 1000 of warm-up.

## Uniformity

Pearson's chi-square over 32 bins of the 64 ranks; a block's p-value is the least of its coordinates' times their number. The pass rule, at 200 simulations, is p > 0.01 for every block.

| block | coordinates | p |
|---|---|---|
| `log_roi` | 2 | 0.23 |
| `logit_retention` | 2 | 0.182 |
| `log_half_saturation` | 2 | 0.772 |
| `log_slope` | 2 | 0.961 |
| `log_spread` | 2 | 0.00258 |
| `deviation` | 6 | 0.223 |
| `log_base` | 4 | 0.872 |
| `log_sigma` | 1 | 0.997 |

## Rank histograms

Each bin's count in quarters of the 6.25 a uniform rank expects: `.` none, `:` a quarter, `-` a half, `=` three quarters, `+` as expected, and on to `@` at twice.

```
log_roi[tv]                  ***=:--=*=:+===++@=**-#::=:==+@*  p 0.115  [8, 8, 8, 6, 3, 4, 4, 6, 9, 5, 2, 7, 5, 6, 5, 7, 7, 14, 5, 8, 8, 4, 10, 2, 3, 5, 2, 5, 5, 7, 13, 9]
log_roi[search]              -+*-*+*-+++=.=*:*==*:+.=*-@=**:*  p 0.178  [4, 7, 9, 4, 9, 7, 8, 4, 7, 7, 7, 5, 1, 5, 9, 2, 8, 5, 5, 9, 3, 7, 1, 5, 9, 4, 13, 6, 9, 9, 3, 9]
logit_retention[tv]          =+%=:*=#+%-=#:-*::#===%:*=*=:=:=  p 0.091  [5, 7, 12, 6, 3, 8, 5, 10, 7, 11, 4, 6, 10, 3, 4, 9, 2, 2, 10, 6, 5, 6, 12, 2, 8, 6, 8, 5, 3, 6, 3, 6]
logit_retention[search]      ===+++***====::+:==@+=**+*==-**-  p 0.755  [5, 6, 5, 7, 7, 7, 8, 9, 9, 5, 5, 5, 6, 2, 2, 7, 3, 5, 6, 13, 7, 5, 8, 8, 7, 8, 5, 5, 4, 9, 8, 4]
log_half_saturation[tv]      #===:%+*=-=+==+=*=*+%=-=:#.#::+=  p 0.386  [10, 5, 6, 6, 3, 11, 7, 8, 6, 4, 5, 7, 6, 5, 7, 6, 9, 6, 9, 7, 11, 6, 4, 5, 3, 10, 1, 10, 3, 2, 7, 5]
log_half_saturation[search]  +-%-=%-=-*:=%==**+=*=+*:---=*=+=  p 0.580  [7, 4, 12, 4, 6, 12, 4, 6, 4, 8, 3, 6, 11, 6, 6, 8, 8, 7, 5, 8, 6, 7, 8, 2, 4, 4, 4, 5, 8, 5, 7, 5]
log_slope[tv]                ==**+=-*:=*:*-:=%-=-*:**%*+-==+=  p 0.480  [6, 6, 9, 9, 7, 5, 4, 8, 3, 5, 8, 2, 9, 4, 2, 5, 11, 4, 5, 4, 8, 3, 9, 8, 12, 8, 7, 4, 6, 6, 7, 6]
log_slope[search]            =:*=-:=-#=*=:+*+=+=:-=#*+##**==+  p 0.513  [6, 2, 8, 5, 4, 2, 5, 4, 10, 6, 8, 5, 2, 7, 9, 7, 6, 7, 5, 2, 4, 5, 10, 9, 7, 10, 10, 8, 9, 6, 5, 7]
log_spread[tv]               @=+-==:+=*:#=++--:#+=:*=*=-+=:==  p 0.001  [22, 5, 7, 4, 6, 6, 3, 7, 6, 8, 3, 10, 5, 7, 7, 4, 4, 2, 10, 7, 6, 3, 8, 6, 8, 5, 4, 7, 6, 3, 5, 6]
log_spread[search]           ==*=-==.#-*=*++*-*.-@*-+#*=**:-=  p 0.152  [5, 6, 8, 5, 4, 6, 5, 0, 10, 4, 9, 5, 9, 7, 7, 8, 4, 8, 1, 4, 13, 9, 4, 7, 10, 9, 6, 8, 8, 2, 4, 5]
deviation[tv,1]              =-+:+*==*--++=*#%=*-*:=:*===-*=*  p 0.823  [6, 4, 7, 2, 7, 8, 6, 5, 8, 4, 4, 7, 7, 6, 8, 10, 12, 6, 8, 4, 8, 3, 6, 3, 8, 5, 6, 6, 4, 8, 5, 9]
deviation[tv,2]              :*=-+*-+*=+#:#==+++=+*-*==*#=:=:  p 0.694  [3, 9, 5, 4, 7, 9, 4, 7, 8, 5, 7, 10, 3, 10, 6, 5, 7, 7, 7, 5, 7, 9, 4, 9, 6, 5, 8, 10, 5, 2, 5, 2]
deviation[tv,3]              *==:***=-=%:*:%**==-+=*:#-==-*.=  p 0.252  [9, 6, 5, 3, 8, 8, 8, 5, 4, 6, 11, 3, 9, 3, 12, 9, 9, 6, 6, 4, 7, 6, 8, 3, 10, 4, 6, 5, 4, 8, 0, 5]
deviation[search,1]          #*=:#*+**=%*:-:+=.+==@*=-:+=*:==  p 0.109  [10, 8, 6, 3, 10, 8, 7, 8, 9, 5, 11, 9, 2, 4, 2, 7, 6, 1, 7, 5, 6, 13, 8, 6, 4, 2, 7, 6, 8, 2, 5, 5]
deviation[search,2]          :==.+=*=#===*=.*=-.+*%--==+%:##%  p 0.037  [3, 6, 5, 1, 7, 5, 8, 5, 10, 6, 5, 6, 8, 6, 1, 9, 5, 4, 1, 7, 8, 12, 4, 4, 5, 5, 7, 12, 3, 10, 10, 12]
deviation[search,3]          =.+-:*-=:::#%#*+:==+#+@==*==+-=*  p 0.161  [6, 1, 7, 4, 3, 8, 4, 6, 3, 3, 3, 10, 11, 10, 8, 7, 3, 5, 5, 7, 10, 7, 14, 5, 5, 8, 6, 6, 7, 4, 6, 8]
log_base[geo0]               *:=*+:***+==.--===*=+**+=:=**=%:  p 0.448  [9, 2, 5, 9, 7, 2, 9, 9, 8, 7, 6, 5, 1, 4, 4, 6, 6, 5, 8, 6, 7, 9, 8, 7, 6, 3, 5, 9, 8, 5, 12, 3]
log_base[geo1]               -+=:===*#===-%%=++#=#=#:==-+::=#  p 0.218  [4, 7, 5, 2, 5, 6, 5, 8, 10, 5, 6, 6, 4, 12, 11, 5, 7, 7, 10, 6, 10, 5, 10, 2, 5, 6, 4, 7, 2, 2, 6, 10]
log_base[geo2]               #-*:=%=:=#++**#=+.*=:-:-+*===*++  p 0.276  [10, 4, 9, 2, 5, 11, 5, 2, 6, 10, 7, 7, 9, 8, 10, 5, 7, 1, 9, 6, 3, 4, 2, 4, 7, 9, 6, 5, 5, 8, 7, 7]
log_base[geo3]               ==-*====#=*===*--**==-+.*+:%=+*=  p 0.870  [6, 6, 4, 8, 6, 6, 6, 5, 10, 5, 8, 6, 6, 6, 9, 4, 4, 8, 8, 5, 5, 4, 7, 1, 8, 7, 3, 12, 6, 7, 8, 6]
log_sigma                    #=-=-=+=+=+--+++***=*-=*+=*==-==  p 0.997  [10, 5, 4, 6, 4, 6, 7, 5, 7, 6, 7, 4, 4, 7, 7, 7, 9, 9, 8, 5, 8, 4, 6, 8, 7, 5, 8, 6, 5, 4, 6, 6]
```

## Divergences and health

Fits with a divergent transition: 100% (200 of 200); 55700 divergent transitions in all, a fit's median 224.5 and most 1116.
Fits with an R-hat above 1.01: 123; with a bulk ESS below 400: 109.

## Where the divergences fall

Share of fits that diverged among simulations whose true value is at or below the coordinate's median over the run, and above it; Spearman's correlation of a fit's divergent transitions with the true value; and the mean of the draws at the divergent transitions less the true value, over the fits that diverged.

| coordinate | diverged, true at or below median | above | Spearman | divergent less true |
|---|---|---|---|---|
| `log_roi[tv]` | 100% | 100% | +0.14 | -0.05 |
| `log_roi[search]` | 100% | 100% | +0.14 | -0.08 |
| `logit_retention[tv]` | 100% | 100% | +0.20 | +0.14 |
| `logit_retention[search]` | 100% | 100% | +0.06 | -0.06 |
| `log_half_saturation[tv]` | 100% | 100% | -0.07 | +0.13 |
| `log_half_saturation[search]` | 100% | 100% | -0.01 | +0.06 |
| `log_slope[tv]` | 100% | 100% | -0.04 | -0.01 |
| `log_slope[search]` | 100% | 100% | -0.04 | -0.08 |
| `log_spread[tv]` | 100% | 100% | -0.05 | -1.04 |
| `log_spread[search]` | 100% | 100% | +0.06 | -0.03 |
| `deviation[tv,1]` | 100% | 100% | -0.13 | -0.03 |
| `deviation[tv,2]` | 100% | 100% | -0.11 | +0.03 |
| `deviation[tv,3]` | 100% | 100% | +0.02 | +0.11 |
| `deviation[search,1]` | 100% | 100% | -0.00 | +0.16 |
| `deviation[search,2]` | 100% | 100% | -0.02 | -0.15 |
| `deviation[search,3]` | 100% | 100% | -0.05 | -0.12 |
| `log_base[geo0]` | 100% | 100% | +0.02 | -0.01 |
| `log_base[geo1]` | 100% | 100% | -0.08 | -0.01 |
| `log_base[geo2]` | 100% | 100% | -0.06 | -0.03 |
| `log_base[geo3]` | 100% | 100% | +0.00 | -0.01 |
| `log_sigma` | 100% | 100% | +0.02 | +0.00 |
