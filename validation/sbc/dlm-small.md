# Simulation-based calibration: dlm, world small

Command: `python -m validation.sbc --engine dlm --world small --sims 200 --seed 0 --workers 5`, run 2026-11-10. Protocol: ADR 0059.

200 simulations, 200 fitted, 0 failed; ranks among 63 thinned draws of 4 chains of 1000 after 1000 of warm-up.

## Uniformity

Pearson's chi-square over 32 bins of the 64 ranks; a block's p-value is the least of its coordinates' times their number. The pass rule, at 200 simulations, is p > 0.01 for every block.

| block | coordinates | p |
|---|---|---|
| `level` | 1 | 0.58 |
| `coefficient` | 2 | 1 |
| `log_variance` | 1 | 0.357 |

## Rank histograms

Each bin's count in quarters of the 6.25 a uniform rank expects: `.` none, `:` a quarter, `-` a half, `=` three quarters, `+` as expected, and on to `@` at twice.

```
level                ++-*==+:=%#-==-==+=-:%+:-++#++*+  p 0.580  [7, 7, 4, 9, 5, 6, 7, 2, 6, 11, 10, 4, 5, 5, 4, 6, 5, 7, 5, 4, 2, 12, 7, 3, 4, 7, 7, 10, 7, 7, 8, 7]
coefficient[tv]      *=--=**-+==**:=-=+*#+%*.+==+-=#:  p 0.546  [9, 5, 4, 4, 6, 9, 8, 4, 7, 6, 5, 8, 8, 3, 5, 4, 5, 7, 8, 10, 7, 12, 8, 1, 7, 6, 5, 7, 4, 6, 10, 2]
coefficient[search]  ==@**=*==*=*=*-:=*=#=:===+:=:=#=  p 0.530  [6, 5, 13, 8, 9, 6, 8, 5, 5, 8, 5, 9, 5, 9, 4, 2, 5, 8, 6, 10, 6, 2, 5, 5, 5, 7, 3, 6, 3, 6, 10, 6]
log_variance         %==###:=#=:-:=*#=+===:**=+=+-:*=  p 0.357  [12, 5, 6, 10, 10, 10, 2, 6, 10, 6, 3, 4, 2, 5, 8, 10, 5, 7, 5, 5, 5, 3, 8, 8, 5, 7, 5, 7, 4, 3, 8, 6]
```

## Divergences and health

The sampler has no divergent transitions (Gibbs chains).
Fits with an R-hat above 1.01: 0; with a bulk ESS below 400: 0.
