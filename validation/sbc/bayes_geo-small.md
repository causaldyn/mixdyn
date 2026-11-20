# Simulation-based calibration: bayes_geo, world small

Command: `python -m validation.sbc --engine bayes_geo --world small --sims 200 --seed 0 --workers 5`, run 2026-11-10. Protocol: ADR 0059.

200 simulations, 200 fitted, 0 failed; ranks among 63 thinned draws of 4 chains of 1000 after 1000 of warm-up.

## Uniformity

Pearson's chi-square over 32 bins of the 64 ranks; a block's p-value is the least of its coordinates' times their number. The pass rule, at 200 simulations, is p > 0.01 for every block.

| block | coordinates | p |
|---|---|---|
| `log_roi` | 2 | 0.172 |
| `logit_retention` | 2 | 1 |
| `log_half_saturation` | 2 | 0.0606 |
| `log_slope` | 2 | 0.152 |
| `log_spread` | 2 | 1 |
| `deviation` | 6 | 0.158 |
| `log_base` | 4 | 1 |
| `log_sigma` | 1 | 0.917 |

## Rank histograms

Each bin's count in quarters of the 6.25 a uniform rank expects: `.` none, `:` a quarter, `-` a half, `=` three quarters, `+` as expected, and on to `@` at twice.

```
log_roi[tv]                  ===*%-===:*:-+@-:%*-+=:*+:+:-#%=  p 0.086  [6, 6, 5, 8, 11, 4, 5, 5, 5, 3, 8, 3, 4, 7, 13, 4, 3, 11, 9, 4, 7, 6, 2, 9, 7, 3, 7, 3, 4, 10, 12, 6]
log_roi[search]              =*-:**+=+=+=-:*-#*:.-====*#=-*%=  p 0.530  [5, 8, 4, 3, 9, 9, 7, 6, 7, 6, 7, 6, 4, 3, 8, 4, 10, 9, 3, 1, 4, 5, 6, 5, 6, 9, 10, 6, 4, 9, 11, 6]
logit_retention[tv]          ==*:*%=-*==#+===:=:=+*#:*===*=:=  p 0.629  [6, 6, 8, 3, 9, 12, 5, 4, 9, 5, 5, 10, 7, 6, 6, 6, 3, 6, 3, 6, 7, 9, 10, 2, 9, 5, 5, 5, 8, 6, 3, 6]
logit_retention[search]      +-=====##-=:=+====-*%=+*++:-#+=+  p 0.909  [7, 4, 6, 5, 6, 6, 6, 10, 10, 4, 6, 3, 6, 7, 5, 5, 5, 5, 4, 9, 11, 5, 7, 8, 7, 7, 3, 4, 10, 7, 5, 7]
log_half_saturation[tv]      #=:**=-:===*:%*.**==%@=--:*+--:*  p 0.030  [10, 5, 2, 9, 9, 6, 4, 3, 6, 6, 5, 9, 3, 12, 8, 1, 9, 8, 5, 6, 12, 13, 5, 4, 4, 2, 8, 7, 4, 4, 3, 8]
log_half_saturation[search]  *==#*=====%*+=*==*+**:=:.-=-*+=+  p 0.694  [8, 5, 5, 10, 8, 5, 5, 5, 5, 6, 11, 8, 7, 6, 8, 6, 6, 9, 7, 8, 9, 2, 6, 3, 0, 4, 5, 4, 9, 7, 6, 7]
log_slope[tv]                =**-=*-=--:-%#--:+:+*#*==%=*==*-  p 0.513  [6, 8, 9, 4, 6, 8, 4, 6, 4, 4, 3, 4, 11, 10, 4, 4, 3, 7, 3, 7, 8, 10, 9, 5, 5, 11, 5, 9, 5, 6, 8, 4]
log_slope[search]            ==:-:=.%*+=*==++::=-++*=%*%=@--=  p 0.076  [6, 6, 3, 4, 3, 5, 1, 11, 9, 7, 6, 9, 5, 6, 7, 7, 2, 2, 5, 4, 7, 7, 8, 5, 11, 9, 11, 6, 14, 4, 4, 6]
log_spread[tv]               -#***:+=*===+-++*=+:=-#=:-***=-=  p 0.836  [4, 10, 9, 8, 9, 3, 7, 6, 9, 5, 5, 6, 7, 4, 7, 7, 8, 5, 7, 2, 6, 4, 10, 5, 3, 4, 9, 8, 8, 6, 4, 5]
log_spread[search]           :=+-*+--++===-*==*=**%=%#:*+=-=:  p 0.811  [3, 5, 7, 4, 8, 7, 4, 4, 7, 7, 6, 6, 5, 4, 8, 5, 5, 8, 6, 9, 8, 11, 5, 11, 10, 3, 8, 7, 6, 4, 6, 3]
deviation[tv,1]              +:=+:**+=+=#==:%=:=*=-++-=+==***  p 0.890  [7, 3, 5, 7, 3, 9, 8, 7, 6, 7, 6, 10, 5, 6, 3, 11, 6, 2, 5, 8, 5, 4, 7, 7, 4, 6, 7, 5, 6, 9, 8, 8]
deviation[tv,2]              =+==*+-*+==-*===*-*#-===**:=:=#:  p 0.900  [6, 7, 5, 5, 8, 7, 4, 9, 7, 6, 6, 4, 8, 6, 6, 5, 9, 4, 8, 10, 4, 6, 6, 6, 9, 9, 3, 6, 2, 6, 10, 3]
deviation[tv,3]              *+%==-*=+=:=#*-*+#==*=-::::*+=+=  p 0.694  [9, 7, 11, 6, 6, 4, 8, 6, 7, 6, 3, 5, 10, 8, 4, 8, 7, 10, 5, 6, 9, 6, 4, 2, 3, 3, 3, 9, 7, 5, 7, 6]
deviation[search,1]          =*@=%+*=*-+*==-*:*-:=-*=+=#==-=:  p 0.401  [5, 8, 13, 5, 11, 7, 8, 5, 9, 4, 7, 9, 5, 6, 4, 9, 3, 9, 4, 2, 5, 4, 8, 5, 7, 5, 10, 5, 6, 4, 5, 3]
deviation[search,2]          +::=:+*%-*-=:**=+:-*==++=%+-*#%=  p 0.197  [7, 2, 2, 6, 2, 7, 8, 12, 4, 9, 4, 5, 2, 8, 8, 5, 7, 3, 4, 8, 5, 5, 7, 7, 5, 11, 7, 4, 9, 10, 11, 6]
deviation[search,3]          -:=:*:=:=+-=*@+-*+-=#%=*-%:.-+@=  p 0.026  [4, 3, 5, 3, 9, 3, 6, 3, 6, 7, 4, 6, 8, 13, 7, 4, 9, 7, 4, 5, 10, 12, 6, 8, 4, 11, 3, 1, 4, 7, 13, 5]
log_base[geo0]               =-=*=#=+#:-*-=-:=%*==#-=*#::=+*-  p 0.417  [6, 4, 5, 9, 6, 10, 6, 7, 10, 2, 4, 8, 4, 5, 4, 3, 5, 11, 9, 6, 5, 10, 4, 6, 9, 10, 3, 3, 6, 7, 9, 4]
log_base[geo1]               ===--=++==+#@=-=%*===*=-+--=++:=  p 0.694  [5, 5, 5, 4, 4, 6, 7, 7, 5, 6, 7, 10, 14, 6, 4, 6, 12, 8, 5, 6, 6, 8, 6, 4, 7, 4, 4, 6, 7, 7, 3, 6]
log_base[geo2]               +==-=-+#=**#-%==:===-:+*=**:*=*=  p 0.710  [7, 5, 6, 4, 6, 4, 7, 10, 5, 8, 9, 10, 4, 11, 5, 6, 2, 6, 6, 6, 4, 3, 7, 8, 5, 8, 8, 2, 9, 5, 9, 5]
log_base[geo3]               +-%::=+*=*#=++=++--:+=*:=-+-*@==  p 0.464  [7, 4, 11, 3, 3, 5, 7, 8, 5, 9, 10, 5, 7, 7, 5, 7, 7, 4, 4, 2, 7, 6, 9, 3, 6, 4, 7, 4, 9, 13, 6, 6]
log_sigma                    *=:*===:++*+*=*:#=+==#*+**--=-==  p 0.917  [8, 5, 2, 8, 6, 5, 5, 2, 7, 7, 8, 7, 9, 6, 8, 3, 10, 5, 7, 6, 6, 10, 8, 7, 8, 8, 4, 4, 6, 4, 5, 6]
```

## Divergences and health

Fits with a divergent transition: 3% (6 of 200); 6 divergent transitions in all, a fit's median 0 and most 1.
Fits with an R-hat above 1.01: 0; with a bulk ESS below 400: 0.

## Where the divergences fall

Share of fits that diverged among simulations whose true value is at or below the coordinate's median over the run, and above it; Spearman's correlation of a fit's divergent transitions with the true value; and the mean of the draws at the divergent transitions less the true value, over the fits that diverged.

| coordinate | diverged, true at or below median | above | Spearman | divergent less true |
|---|---|---|---|---|
| `log_roi[tv]` | 4% | 2% | -0.10 | +0.37 |
| `log_roi[search]` | 3% | 3% | +0.02 | +0.11 |
| `logit_retention[tv]` | 6% | 0% | -0.20 | +1.78 |
| `logit_retention[search]` | 3% | 3% | +0.01 | -0.28 |
| `log_half_saturation[tv]` | 2% | 4% | +0.07 | -0.55 |
| `log_half_saturation[search]` | 3% | 3% | -0.01 | +0.39 |
| `log_slope[tv]` | 3% | 3% | -0.01 | +0.25 |
| `log_slope[search]` | 5% | 1% | -0.14 | +0.54 |
| `log_spread[tv]` | 4% | 2% | -0.11 | -0.71 |
| `log_spread[search]` | 5% | 1% | -0.06 | -1.29 |
| `deviation[tv,1]` | 4% | 2% | -0.03 | +0.06 |
| `deviation[tv,2]` | 1% | 5% | +0.10 | -0.09 |
| `deviation[tv,3]` | 3% | 3% | +0.00 | +0.76 |
| `deviation[search,1]` | 3% | 3% | -0.05 | +0.33 |
| `deviation[search,2]` | 4% | 2% | -0.02 | +0.33 |
| `deviation[search,3]` | 2% | 4% | +0.06 | +0.31 |
| `log_base[geo0]` | 2% | 4% | +0.08 | -0.09 |
| `log_base[geo1]` | 3% | 3% | +0.00 | +0.03 |
| `log_base[geo2]` | 4% | 2% | -0.07 | +0.09 |
| `log_base[geo3]` | 4% | 2% | -0.10 | +0.02 |
| `log_sigma` | 3% | 3% | +0.01 | -0.00 |
