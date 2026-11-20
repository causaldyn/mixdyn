# Simulation-based calibration: bayes_geo, world centred-geo, slope = 'fixed'

Command: `python -m validation.sbc --engine bayes_geo --world centred-geo --knob slope=fixed --sims 200 --seed 0 --workers 5`, run 2026-11-10. Protocol: ADR 0059.

200 simulations, 200 fitted, 0 failed; ranks among 63 thinned draws of 4 chains of 1000 after 1000 of warm-up.

## Uniformity

Pearson's chi-square over 32 bins of the 64 ranks; a block's p-value is the least of its coordinates' times their number. The pass rule, at 200 simulations, is p > 0.01 for every block.

| block | coordinates | p |
|---|---|---|
| `log_roi` | 2 | 0.194 |
| `logit_retention` | 2 | 1 |
| `log_half_saturation` | 2 | 0.896 |
| `log_spread` | 2 | 0.00606 |
| `deviation` | 6 | 0.616 |
| `log_base` | 4 | 1 |
| `log_sigma` | 1 | 0.0398 |

## Rank histograms

Each bin's count in quarters of the 6.25 a uniform rank expects: `.` none, `:` a quarter, `-` a half, `=` three quarters, `+` as expected, and on to `@` at twice.

```
log_roi[tv]                  =**+=-*:#:*-===*#*=:+-+*=:-=:###  p 0.417  [5, 8, 8, 7, 5, 4, 8, 2, 10, 3, 8, 4, 5, 6, 6, 8, 10, 9, 5, 3, 7, 4, 7, 9, 5, 2, 4, 5, 3, 10, 10, 10]
log_roi[search]              -*+++++=--+*::@=::%:=:+-=**%*=%=  p 0.097  [4, 8, 7, 7, 7, 7, 7, 5, 4, 4, 7, 8, 2, 2, 13, 6, 2, 3, 11, 2, 5, 3, 7, 4, 6, 9, 8, 11, 9, 6, 11, 5]
logit_retention[tv]          +=#==#=+***=+=:+=:=:=#=*=*=*:-==  p 0.811  [7, 5, 10, 5, 6, 10, 5, 7, 8, 9, 9, 6, 7, 5, 2, 7, 5, 3, 5, 3, 5, 10, 6, 9, 5, 8, 6, 9, 3, 4, 5, 6]
logit_retention[search]      :*==+*=*:===+*-:--+@*=+=*-===+=*  p 0.823  [3, 9, 6, 6, 7, 8, 5, 8, 2, 6, 6, 6, 7, 8, 4, 3, 4, 4, 7, 13, 8, 5, 7, 6, 9, 4, 6, 6, 5, 7, 6, 9]
log_half_saturation[tv]      #===*+:=#**:*+==+=*=*+%--===:-+=  p 0.823  [10, 6, 5, 5, 8, 7, 2, 6, 10, 9, 8, 3, 8, 7, 6, 5, 7, 5, 8, 5, 9, 7, 11, 4, 4, 6, 5, 5, 3, 4, 7, 5]
log_half_saturation[search]  **===#=+=:--#=@:=+=+***:=-=*=+=:  p 0.448  [8, 9, 5, 5, 6, 10, 6, 7, 6, 2, 4, 4, 10, 6, 13, 3, 5, 7, 6, 7, 8, 9, 8, 2, 5, 4, 6, 9, 6, 7, 5, 2]
log_spread[tv]               @%-=:=%==---=%::+*:+==*-*=*==:-=  p 0.003  [19, 11, 4, 6, 3, 6, 11, 5, 5, 4, 4, 4, 6, 12, 3, 3, 7, 8, 2, 7, 6, 5, 9, 4, 8, 6, 8, 5, 6, 3, 4, 6]
log_spread[search]           @=.-==+-**=:-*=-*+-+=++@*-+*-*-:  p 0.023  [17, 6, 1, 4, 6, 5, 7, 4, 8, 9, 5, 2, 4, 9, 5, 4, 8, 7, 4, 7, 6, 7, 7, 13, 8, 4, 7, 8, 4, 8, 4, 2]
deviation[tv,1]              =*=*::==*-*=*=#%*===:=:#-:#==*:*  p 0.629  [6, 8, 5, 8, 3, 3, 6, 5, 8, 4, 8, 6, 8, 5, 10, 11, 8, 6, 6, 6, 3, 6, 3, 10, 4, 3, 10, 5, 6, 9, 3, 8]
deviation[tv,2]              -=*====**-=*==-#:#@#====:=*+*-=:  p 0.302  [4, 5, 8, 6, 5, 5, 5, 8, 9, 4, 6, 9, 6, 5, 4, 10, 3, 10, 14, 10, 5, 5, 6, 5, 2, 5, 8, 7, 9, 4, 6, 2]
deviation[tv,3]              *====-=@*=*=:-++%@:==:+-+*==:+--  p 0.103  [8, 5, 6, 6, 5, 4, 6, 13, 8, 5, 9, 5, 3, 4, 7, 7, 12, 15, 3, 6, 6, 2, 7, 4, 7, 8, 6, 5, 3, 7, 4, 4]
deviation[search,1]          =%=%***+--+:*+=--*-===:+=-%=*:*:  p 0.401  [5, 11, 5, 11, 9, 9, 8, 7, 4, 4, 7, 3, 8, 7, 6, 4, 4, 9, 4, 6, 6, 5, 3, 7, 6, 4, 12, 5, 8, 3, 8, 2]
deviation[search,2]          +:#===.=+=*#-:+==-++=***=+=+#+=+  p 0.870  [7, 3, 10, 5, 5, 6, 1, 5, 7, 5, 8, 10, 4, 3, 7, 5, 5, 4, 7, 7, 5, 9, 9, 8, 5, 7, 6, 7, 10, 7, 6, 7]
deviation[search,3]          :*-==:%*==+.-#==+=+=*%%-=-*:-*=%  p 0.109  [2, 8, 4, 5, 5, 2, 11, 8, 5, 6, 7, 1, 4, 10, 6, 6, 7, 5, 7, 5, 8, 12, 12, 4, 6, 4, 9, 3, 4, 8, 5, 11]
log_base[geo0]               +=#--*=+*:%===:=+=+%=+:=*-=***--  p 0.678  [7, 6, 10, 4, 4, 9, 6, 7, 9, 2, 11, 5, 5, 5, 3, 6, 7, 6, 7, 11, 5, 7, 3, 5, 9, 4, 5, 8, 8, 8, 4, 4]
log_base[geo1]               =%:=+=:=::#==%****:+*+#:=*===:+=  p 0.371  [5, 11, 2, 6, 7, 5, 3, 5, 3, 2, 10, 6, 5, 11, 8, 8, 9, 9, 3, 7, 8, 7, 10, 3, 5, 9, 6, 6, 6, 3, 7, 5]
log_base[geo2]               #--*+-*#=-==@.+**==----:-*:+****  p 0.329  [10, 4, 4, 8, 7, 4, 8, 10, 6, 4, 6, 6, 14, 1, 7, 8, 8, 6, 6, 4, 4, 4, 4, 3, 4, 8, 3, 7, 8, 8, 8, 8]
log_base[geo3]               :==++*+*====+=::%*+=+=======:*%+  p 0.823  [3, 6, 5, 7, 7, 9, 7, 8, 5, 5, 6, 6, 7, 5, 2, 3, 11, 9, 7, 6, 7, 5, 5, 6, 5, 5, 6, 6, 3, 9, 12, 7]
log_sigma                    =#::-++-%-*==+.#@-*+=*=**#-::=+=  p 0.040  [6, 10, 2, 2, 4, 7, 7, 4, 12, 4, 8, 6, 5, 7, 1, 10, 14, 4, 8, 7, 5, 8, 5, 8, 9, 10, 4, 2, 2, 6, 7, 6]
```

## Divergences and health

Fits with a divergent transition: 100% (200 of 200); 59927 divergent transitions in all, a fit's median 250.5 and most 1161.
Fits with an R-hat above 1.01: 161; with a bulk ESS below 400: 144.

## Where the divergences fall

Share of fits that diverged among simulations whose true value is at or below the coordinate's median over the run, and above it; Spearman's correlation of a fit's divergent transitions with the true value; and the mean of the draws at the divergent transitions less the true value, over the fits that diverged.

| coordinate | diverged, true at or below median | above | Spearman | divergent less true |
|---|---|---|---|---|
| `log_roi[tv]` | 100% | 100% | +0.01 | -0.06 |
| `log_roi[search]` | 100% | 100% | -0.15 | -0.12 |
| `logit_retention[tv]` | 100% | 100% | +0.01 | +0.08 |
| `logit_retention[search]` | 100% | 100% | -0.03 | -0.12 |
| `log_half_saturation[tv]` | 100% | 100% | +0.07 | +0.14 |
| `log_half_saturation[search]` | 100% | 100% | +0.04 | +0.11 |
| `log_spread[tv]` | 100% | 100% | +0.04 | -0.48 |
| `log_spread[search]` | 100% | 100% | +0.03 | -0.60 |
| `deviation[tv,1]` | 100% | 100% | -0.01 | -0.04 |
| `deviation[tv,2]` | 100% | 100% | -0.01 | +0.01 |
| `deviation[tv,3]` | 100% | 100% | -0.11 | +0.12 |
| `deviation[search,1]` | 100% | 100% | +0.07 | +0.14 |
| `deviation[search,2]` | 100% | 100% | -0.08 | -0.08 |
| `deviation[search,3]` | 100% | 100% | +0.04 | -0.11 |
| `log_base[geo0]` | 100% | 100% | -0.02 | +0.01 |
| `log_base[geo1]` | 100% | 100% | +0.08 | -0.01 |
| `log_base[geo2]` | 100% | 100% | +0.06 | -0.01 |
| `log_base[geo3]` | 100% | 100% | -0.09 | -0.02 |
| `log_sigma` | 100% | 100% | +0.10 | +0.00 |
