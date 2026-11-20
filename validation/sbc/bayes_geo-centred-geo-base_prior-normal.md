# Simulation-based calibration: bayes_geo, world centred-geo, base_prior = 'normal'

Command: `python -m validation.sbc --engine bayes_geo --world centred-geo --knob base_prior=normal --sims 200 --seed 0 --workers 5`, run 2026-11-10. Protocol: ADR 0059.

200 simulations, 200 fitted, 0 failed; ranks among 63 thinned draws of 4 chains of 1000 after 1000 of warm-up.

## Uniformity

Pearson's chi-square over 32 bins of the 64 ranks; a block's p-value is the least of its coordinates' times their number. The pass rule, at 200 simulations, is p > 0.01 for every block.

| block | coordinates | p |
|---|---|---|
| `log_roi` | 2 | 0.143 |
| `logit_retention` | 2 | 1 |
| `log_half_saturation` | 2 | 0.63 |
| `log_slope` | 2 | 0.833 |
| `log_spread` | 2 | 0.111 |
| `deviation` | 6 | 0.147 |
| `base` | 4 | 1 |
| `log_sigma` | 1 | 0.264 |

## Rank histograms

Each bin's count in quarters of the 6.25 a uniform rank expects: `.` none, `:` a quarter, `-` a half, `=` three quarters, `+` as expected, and on to `@` at twice.

```
log_roi[tv]                  *=*=#=:-:=-=--=*%#*=+=*=:==-.*%*  p 0.218  [8, 5, 8, 6, 10, 5, 2, 4, 3, 6, 4, 5, 4, 4, 5, 9, 12, 10, 9, 6, 7, 6, 8, 6, 3, 5, 6, 4, 1, 9, 12, 8]
log_roi[search]              +*-+:%-*==:+%::*=-*:====:#-*=@*=  p 0.071  [7, 8, 4, 7, 3, 11, 4, 8, 6, 5, 2, 7, 12, 2, 3, 9, 6, 4, 8, 3, 5, 5, 6, 6, 2, 10, 4, 8, 6, 14, 9, 6]
logit_retention[tv]          *-==*%===**+-++:+:-%+=*+=+::**=:  p 0.546  [9, 4, 6, 5, 8, 12, 6, 6, 5, 8, 8, 7, 4, 7, 7, 3, 7, 3, 4, 12, 7, 5, 8, 7, 5, 7, 3, 2, 8, 9, 5, 3]
logit_retention[search]      ==**-#++=====:---=*=#**%==-=+#=+  p 0.870  [5, 5, 8, 8, 4, 10, 7, 7, 5, 5, 6, 5, 6, 2, 4, 4, 4, 5, 8, 6, 10, 8, 8, 11, 5, 5, 4, 5, 7, 10, 6, 7]
log_half_saturation[tv]      #=++=:#==.+%=+**====**:=*=-:==+=  p 0.662  [10, 5, 7, 7, 6, 3, 10, 5, 6, 1, 7, 12, 5, 7, 9, 9, 6, 5, 6, 5, 8, 9, 3, 6, 8, 6, 4, 3, 5, 5, 7, 5]
log_half_saturation[search]  ==#%++=:-*-=@=:+-***==*::==+==-+  p 0.315  [6, 5, 10, 11, 7, 7, 5, 3, 4, 9, 4, 6, 14, 6, 3, 7, 4, 8, 8, 9, 6, 5, 9, 2, 2, 5, 5, 7, 6, 6, 4, 7]
log_slope[tv]                =*-=+=+:*#:::*+*:.#:**+*=***:=*+  p 0.417  [6, 9, 4, 5, 7, 6, 7, 3, 9, 10, 3, 3, 3, 8, 7, 8, 2, 1, 10, 3, 8, 9, 7, 8, 6, 8, 9, 8, 3, 5, 8, 7]
log_slope[search]            ::=*::=+#+--#*+-+====:++*%=+%=*:  p 0.464  [2, 3, 6, 8, 3, 3, 5, 7, 10, 7, 4, 4, 10, 8, 7, 4, 7, 6, 5, 6, 6, 3, 7, 7, 9, 12, 6, 7, 11, 6, 8, 3]
log_spread[tv]               @=*=:.+***=+-**+:=.-*==+==-*+=.*  p 0.055  [16, 5, 8, 6, 2, 1, 7, 9, 9, 9, 6, 7, 4, 9, 8, 7, 3, 5, 1, 4, 9, 6, 6, 7, 6, 6, 4, 8, 7, 6, 1, 8]
log_spread[search]           %*=-=--*-=-%====+*=+#:===*=+*==:  p 0.613  [12, 8, 6, 4, 6, 4, 4, 8, 4, 6, 4, 12, 5, 6, 5, 6, 7, 9, 5, 7, 10, 3, 5, 6, 5, 9, 5, 7, 9, 6, 5, 2]
deviation[tv,1]              =*-:=+=+-%=.-@%%*:====:-:=+*+=+*  p 0.025  [5, 8, 4, 3, 5, 7, 5, 7, 4, 12, 5, 0, 4, 15, 12, 12, 8, 3, 6, 5, 6, 5, 3, 4, 3, 6, 7, 8, 7, 6, 7, 8]
deviation[tv,2]              :++*-=+**=++:+=-+#*==--*+===@==:  p 0.725  [3, 7, 7, 8, 4, 5, 7, 9, 8, 5, 7, 7, 3, 7, 5, 4, 7, 10, 9, 6, 6, 4, 4, 9, 7, 5, 6, 5, 13, 6, 5, 2]
deviation[tv,3]              **=::*@:=*==++#%=:*:==*-=*+-:=-=  p 0.229  [9, 9, 5, 3, 3, 8, 13, 3, 6, 9, 6, 5, 7, 7, 10, 12, 5, 3, 8, 2, 5, 6, 9, 4, 6, 8, 7, 4, 3, 6, 4, 5]
deviation[search,1]          =#+*%+#-*=+-:*+:=*=+*-*=++--#:=.  p 0.357  [5, 10, 7, 8, 12, 7, 10, 4, 8, 5, 7, 4, 2, 8, 7, 2, 5, 8, 5, 7, 8, 4, 9, 6, 7, 7, 4, 4, 10, 3, 6, 1]
deviation[search,2]          =*==:+-**:.:**:*=+==+-*=###+*-=*  p 0.530  [5, 8, 6, 6, 2, 7, 4, 9, 8, 2, 1, 3, 8, 8, 3, 8, 6, 7, 6, 5, 7, 4, 8, 6, 10, 10, 10, 7, 9, 4, 5, 8]
deviation[search,3]          ==+:-=*=+:+-==-##+*===*##-++=-+*  p 0.859  [6, 5, 7, 2, 4, 5, 8, 5, 7, 2, 7, 4, 5, 6, 4, 10, 10, 7, 8, 5, 6, 6, 8, 10, 10, 4, 7, 7, 6, 4, 7, 8]
base[geo0]                   +:+*=*=+==:=+:-=+*:*%=+*#:*-=-=#  p 0.629  [7, 2, 7, 8, 6, 8, 6, 7, 6, 6, 2, 6, 7, 3, 4, 6, 7, 8, 3, 9, 12, 6, 7, 8, 10, 3, 8, 4, 5, 4, 5, 10]
base[geo1]                   =+:-==*+:**=*+**+=-***#+:=:+=#-=  p 0.811  [5, 7, 3, 4, 5, 5, 8, 7, 2, 8, 8, 6, 9, 7, 8, 8, 7, 5, 4, 8, 9, 8, 10, 7, 3, 5, 2, 7, 5, 10, 4, 6]
base[geo2]                   *==**+-=*-*+++:=:*+=*:=-=-**-==%  p 0.740  [8, 5, 5, 9, 9, 7, 4, 6, 9, 4, 8, 7, 7, 7, 3, 6, 2, 8, 7, 6, 9, 2, 5, 4, 5, 4, 9, 9, 4, 6, 5, 11]
base[geo3]                   =#-==:**=+%=+:#::=+:-+=*:=*+=+%+  p 0.480  [6, 10, 4, 6, 5, 3, 8, 8, 6, 7, 12, 5, 7, 2, 10, 3, 3, 6, 7, 3, 4, 7, 5, 9, 3, 5, 8, 7, 6, 7, 11, 7]
log_sigma                    -=***--==:++=**====@%=*===:+.%-=  p 0.264  [4, 5, 8, 8, 8, 4, 4, 6, 5, 3, 7, 7, 6, 8, 9, 5, 5, 6, 5, 14, 12, 5, 8, 6, 5, 6, 3, 7, 1, 11, 4, 5]
```

## Divergences and health

Fits with a divergent transition: 100% (200 of 200); 69466 divergent transitions in all, a fit's median 283 and most 1183.
Fits with an R-hat above 1.01: 168; with a bulk ESS below 400: 146.

## Where the divergences fall

Share of fits that diverged among simulations whose true value is at or below the coordinate's median over the run, and above it; Spearman's correlation of a fit's divergent transitions with the true value; and the mean of the draws at the divergent transitions less the true value, over the fits that diverged.

| coordinate | diverged, true at or below median | above | Spearman | divergent less true |
|---|---|---|---|---|
| `log_roi[tv]` | 100% | 100% | +0.06 | -0.04 |
| `log_roi[search]` | 100% | 100% | +0.03 | -0.10 |
| `logit_retention[tv]` | 100% | 100% | -0.05 | +0.07 |
| `logit_retention[search]` | 100% | 100% | -0.13 | -0.09 |
| `log_half_saturation[tv]` | 100% | 100% | -0.01 | +0.08 |
| `log_half_saturation[search]` | 100% | 100% | +0.06 | +0.14 |
| `log_slope[tv]` | 100% | 100% | -0.01 | +0.01 |
| `log_slope[search]` | 100% | 100% | -0.01 | -0.07 |
| `log_spread[tv]` | 100% | 100% | -0.06 | -0.60 |
| `log_spread[search]` | 100% | 100% | +0.05 | -0.53 |
| `deviation[tv,1]` | 100% | 100% | +0.01 | -0.02 |
| `deviation[tv,2]` | 100% | 100% | -0.02 | +0.03 |
| `deviation[tv,3]` | 100% | 100% | +0.01 | +0.12 |
| `deviation[search,1]` | 100% | 100% | -0.05 | +0.10 |
| `deviation[search,2]` | 100% | 100% | +0.05 | -0.09 |
| `deviation[search,3]` | 100% | 100% | -0.01 | -0.15 |
| `base[geo0]` | 100% | 100% | +0.02 | -0.02 |
| `base[geo1]` | 100% | 100% | +0.05 | -0.00 |
| `base[geo2]` | 100% | 100% | +0.03 | -0.01 |
| `base[geo3]` | 100% | 100% | -0.14 | -0.01 |
| `log_sigma` | 100% | 100% | -0.01 | +0.00 |
