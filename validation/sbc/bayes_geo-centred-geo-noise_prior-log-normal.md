# Simulation-based calibration: bayes_geo, world centred-geo, noise_prior = 'log-normal'

Command: `python -m validation.sbc --engine bayes_geo --world centred-geo --knob noise_prior=log-normal --sims 200 --seed 0 --workers 5`, run 2026-11-10. Protocol: ADR 0059.

200 simulations, 200 fitted, 0 failed; ranks among 63 thinned draws of 4 chains of 1000 after 1000 of warm-up.

## Uniformity

Pearson's chi-square over 32 bins of the 64 ranks; a block's p-value is the least of its coordinates' times their number. The pass rule, at 200 simulations, is p > 0.01 for every block.

| block | coordinates | p |
|---|---|---|
| `log_roi` | 2 | 0.258 |
| `logit_retention` | 2 | 0.375 |
| `log_half_saturation` | 2 | 0.993 |
| `log_slope` | 2 | 0.657 |
| `log_spread` | 2 | 0.104 |
| `deviation` | 6 | 0.963 |
| `log_base` | 4 | 1 |
| `log_sigma` | 1 | 0.755 |

## Rank histograms

Each bin's count in quarters of the 6.25 a uniform rank expects: `.` none, `:` a quarter, `-` a half, `=` three quarters, `+` as expected, and on to `@` at twice.

```
log_roi[tv]                  =++@:--=--====+=@==#+=--:--*-#**  p 0.129  [6, 7, 7, 13, 3, 4, 4, 6, 4, 4, 5, 6, 6, 5, 7, 6, 15, 6, 5, 10, 7, 6, 4, 4, 2, 4, 4, 9, 4, 10, 8, 9]
log_roi[search]              =*.*+%+=:*=+:-+:**=-:-**-==+%+%+  p 0.218  [6, 8, 1, 8, 7, 11, 7, 6, 2, 9, 6, 7, 2, 4, 7, 2, 8, 8, 5, 4, 2, 4, 8, 8, 4, 6, 6, 7, 12, 7, 11, 7]
logit_retention[tv]          =*-%+:=#@+=-#::*=-+=-%+:*==:*-==  p 0.188  [5, 8, 4, 11, 7, 3, 5, 10, 13, 7, 5, 4, 10, 3, 3, 8, 5, 4, 7, 6, 4, 12, 7, 3, 8, 5, 5, 3, 9, 4, 6, 6]
logit_retention[search]      +:+==#%=+:===.-*===***%+==:++++=  p 0.678  [7, 2, 7, 5, 5, 10, 11, 6, 7, 3, 5, 6, 5, 1, 4, 8, 6, 6, 5, 8, 8, 8, 12, 7, 5, 6, 3, 7, 7, 7, 7, 6]
log_half_saturation[tv]      *=:+=*=#*==:+==*==**-@+*=+.===+:  p 0.613  [9, 5, 3, 7, 6, 9, 6, 10, 8, 5, 5, 3, 7, 5, 5, 8, 5, 5, 8, 9, 4, 13, 7, 8, 5, 7, 1, 6, 6, 5, 7, 3]
log_half_saturation[search]  +=#+*=:.#=-+*#=+*+*=*-=:--:-*==*  p 0.497  [7, 6, 10, 7, 9, 6, 3, 0, 10, 6, 4, 7, 9, 10, 5, 7, 9, 7, 8, 6, 9, 4, 6, 3, 4, 4, 3, 4, 9, 5, 5, 8]
log_slope[tv]                #*:*:*=:-+*-=*===*=*--*+-+=*+=*-  p 0.909  [10, 9, 3, 8, 2, 8, 5, 3, 4, 7, 8, 4, 6, 8, 6, 6, 6, 9, 6, 8, 4, 4, 8, 7, 4, 7, 6, 9, 7, 6, 8, 4]
log_slope[search]            --===:=*++==*=+-=+-*--:+*@%==+++  p 0.329  [4, 4, 5, 5, 5, 3, 5, 8, 7, 7, 6, 5, 8, 6, 7, 4, 6, 7, 4, 8, 4, 4, 2, 7, 8, 16, 12, 6, 6, 7, 7, 7]
log_spread[tv]               @=#==+=+=*:=%==:=-=-==-***:=-=:*  p 0.052  [18, 5, 10, 5, 5, 7, 6, 7, 5, 8, 3, 6, 11, 6, 6, 2, 6, 4, 6, 4, 6, 5, 4, 9, 8, 9, 3, 5, 4, 6, 3, 8]
log_spread[search]           @=:+=.+--=+:##*+:=#=*==**=#:*=::  p 0.059  [15, 6, 2, 7, 6, 1, 7, 4, 4, 6, 7, 3, 10, 10, 8, 7, 2, 5, 10, 6, 8, 6, 5, 8, 9, 6, 10, 3, 8, 5, 3, 3]
deviation[tv,1]              ===*+:::*==+*+%+==*=-====-**-==#  p 0.909  [6, 5, 6, 9, 7, 3, 3, 3, 8, 5, 6, 7, 9, 7, 11, 7, 5, 6, 9, 6, 4, 5, 6, 6, 5, 4, 8, 9, 4, 5, 6, 10]
deviation[tv,2]              :=*====**-.++=-%++#=#+=**:-+%=::  p 0.161  [2, 6, 9, 6, 6, 6, 5, 8, 9, 4, 1, 7, 7, 5, 4, 12, 7, 7, 10, 5, 10, 7, 5, 8, 9, 2, 4, 7, 12, 5, 3, 2]
deviation[tv,3]              =%:=**==++=%:**=#====:*:*+===:=-  p 0.580  [6, 11, 3, 6, 9, 8, 5, 5, 7, 7, 6, 11, 2, 8, 9, 5, 10, 6, 6, 6, 5, 3, 9, 3, 9, 7, 6, 5, 5, 2, 6, 4]
deviation[search,1]          =***+**+**+-*:==:*=*--=++=+*-==:  p 0.939  [6, 9, 9, 8, 7, 8, 8, 7, 9, 8, 7, 4, 8, 3, 5, 5, 3, 9, 5, 8, 4, 4, 5, 7, 7, 5, 7, 8, 4, 5, 6, 2]
deviation[search,2]          =*=:-+=++=::-+:++++-=*=*%*=%-=@:  p 0.357  [5, 8, 6, 3, 4, 7, 6, 7, 7, 6, 3, 3, 4, 7, 3, 7, 7, 7, 7, 4, 5, 8, 5, 9, 11, 8, 6, 12, 4, 6, 13, 2]
deviation[search,3]          :=-:*+:+%.=:#=-+==#==#==+*%++:=#  p 0.208  [3, 5, 4, 3, 9, 7, 3, 7, 11, 1, 5, 2, 10, 6, 4, 7, 6, 6, 10, 5, 6, 10, 5, 6, 7, 8, 12, 7, 7, 3, 5, 10]
log_base[geo0]               =+=**--**==::*===*-%*-#=%=:=*=*:  p 0.252  [6, 7, 5, 9, 8, 4, 4, 9, 9, 5, 5, 2, 2, 8, 6, 5, 5, 9, 4, 12, 9, 4, 10, 5, 11, 5, 2, 5, 8, 5, 9, 3]
log_base[geo1]               =*-:--=*===-@=-*=%%*+=**=-+-::+=  p 0.386  [5, 8, 4, 3, 4, 4, 6, 9, 6, 6, 5, 4, 13, 6, 4, 9, 5, 11, 11, 8, 7, 6, 8, 9, 6, 4, 7, 4, 3, 2, 7, 6]
log_base[geo2]               *#===#:-%:-+=+%:+===:*-=++-=*==#  p 0.513  [9, 10, 6, 5, 5, 10, 3, 4, 11, 3, 4, 7, 5, 7, 11, 3, 7, 6, 5, 6, 3, 9, 4, 5, 7, 7, 4, 5, 9, 5, 5, 10]
log_base[geo3]               +=*+:*:+--**-*::***=*=+*::==*%=*  p 0.315  [7, 5, 9, 7, 3, 8, 3, 7, 4, 4, 8, 8, 4, 8, 3, 2, 9, 9, 9, 5, 9, 5, 7, 9, 2, 2, 5, 5, 9, 12, 5, 8]
log_sigma                    =+:-*-==*=+:=+=*%-*=*+=**==+:=-*  p 0.755  [6, 7, 3, 4, 8, 4, 6, 6, 9, 5, 7, 2, 6, 7, 6, 8, 12, 4, 9, 6, 9, 7, 5, 9, 9, 5, 5, 7, 2, 5, 4, 8]
```

## Divergences and health

Fits with a divergent transition: 100% (200 of 200); 68883 divergent transitions in all, a fit's median 257 and most 1483.
Fits with an R-hat above 1.01: 156; with a bulk ESS below 400: 140.

## Where the divergences fall

Share of fits that diverged among simulations whose true value is at or below the coordinate's median over the run, and above it; Spearman's correlation of a fit's divergent transitions with the true value; and the mean of the draws at the divergent transitions less the true value, over the fits that diverged.

| coordinate | diverged, true at or below median | above | Spearman | divergent less true |
|---|---|---|---|---|
| `log_roi[tv]` | 100% | 100% | +0.05 | -0.06 |
| `log_roi[search]` | 100% | 100% | -0.01 | -0.05 |
| `logit_retention[tv]` | 100% | 100% | +0.03 | +0.09 |
| `logit_retention[search]` | 100% | 100% | -0.00 | -0.07 |
| `log_half_saturation[tv]` | 100% | 100% | +0.09 | +0.07 |
| `log_half_saturation[search]` | 100% | 100% | -0.03 | +0.08 |
| `log_slope[tv]` | 100% | 100% | +0.10 | +0.03 |
| `log_slope[search]` | 100% | 100% | -0.06 | -0.06 |
| `log_spread[tv]` | 100% | 100% | +0.02 | -0.48 |
| `log_spread[search]` | 100% | 100% | -0.00 | -0.61 |
| `deviation[tv,1]` | 100% | 100% | +0.04 | -0.05 |
| `deviation[tv,2]` | 100% | 100% | +0.02 | +0.02 |
| `deviation[tv,3]` | 100% | 100% | -0.00 | +0.11 |
| `deviation[search,1]` | 100% | 100% | -0.01 | +0.15 |
| `deviation[search,2]` | 100% | 100% | +0.15 | -0.08 |
| `deviation[search,3]` | 100% | 100% | +0.09 | -0.14 |
| `log_base[geo0]` | 100% | 100% | +0.11 | +0.00 |
| `log_base[geo1]` | 100% | 100% | -0.05 | -0.01 |
| `log_base[geo2]` | 100% | 100% | +0.09 | -0.03 |
| `log_base[geo3]` | 100% | 100% | -0.04 | -0.00 |
| `log_sigma` | 100% | 100% | -0.06 | +0.00 |
