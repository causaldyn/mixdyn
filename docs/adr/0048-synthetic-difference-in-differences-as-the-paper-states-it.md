# 0048. Synthetic difference in differences as the paper states it

Status: accepted, 2026-10-28; adds `mixdyn.experiments.sdid`, a geo test's reader that hands on a
`mixdyn.calibration.LiftEstimate`; amended 2026-10-28: random placebos drawn from the child stream
`SeedSequence(seed).spawn(1)[0]`, the `LiftEstimate`'s error widened to the t as ADR 0047's readers
widen theirs, the t's count floored at `(N_co - 1) / 2` (the table below), controls named by
`control=`, and the Maxima enumeration kept as `validation/sdid_simplex_weights.mac`; audited on
null panels by ADR 0058 (`validation/aa_audit.md`): the placebo test holds its level but at a trend
over 10 geos, where it is conservative, and design-aware unit placebos are opt-in by
`variance="unit_placebo"`; amended 2026-11-06: the audit's cells, reader by reader, under
Consequences

## Context

A geo test's analysis hands on a lift and its standard error, a `LiftEstimate`. Synthetic difference in
differences (Arkhangelsky, Athey, Hirshberg, Imbens and Wager 2021, American Economic Review
111(12), 4088-4118) is the reader practitioners now ask for: it weights the control geos so that
their pre-period tracks the treated geos', as a synthetic control does, weights the pre-periods so
that they track the post-period, and reads the effect as a weighted difference in differences.
Read as the paper states it, its estimate is the same in any of the KPI's units and at any geo's
level. A reader that departs from it loses that:

- **Without the intercepts** `w0` and `l0` the unit weights match the treated geos' level as well
  as their path, and the estimate moves with each geo's level. On the tests' panel it read 1.6134
  where the paper's reads 1.6413.
- **A box `[0, 1]` and a renormalisation** in place of the simplex is not the optimum of either
  problem: renormalised, its weights are not the problem's, and from an inexact solver they move
  with the units.
- **A ridge `zeta` where the loss has `zeta^2`**: the fit is in the KPI's units squared and the
  penalty in its units, so their balance, and the weights, move with the units. On the same panel
  the effect per unit of scale read 1.6376 at x1 and 1.6203 at x1000; the paper's reads 1.6413 at
  both.
- **A jackknife over the treated geos alone** reads the variance from `N_tr` values: with two
  treated geos it has one degree of freedom, and against the normal it rejects far above its level.

## Decision

- **The estimator is Algorithm 1.** `sigma^2` is the variance of the controls' first differences
  over the pre-period, about their grand mean, over `N_co (T_pre - 1)`; `zeta = (N_tr T_post)^(1/4)
  sigma`; the unit weights minimise `sum_t (w0 + sum_i w_i Y_it - mean_tr Y_t)^2 + zeta^2 T_pre
  ||w||^2` and the time weights `sum_i (l0 + sum_t l_t Y_it - mean_post Y_i)^2 + zeta_l^2 N_co
  ||l||^2`, both over the simplex, at `zeta_l = 1e-6 sigma`, the authors' package's ridge. The
  effect is the weighted two-way fixed-effects regression, which is the weighted double
  difference; a test checks that the two agree to `1e-10`. A week's shock that is not a common
  trend moves `sigma`, as the paper defines it, and so `zeta`: the estimate is free of each geo's
  level and of a trend all geos share, not of every week's shock.
- **An exact solver.** Each problem's intercept is eliminated by centring, its ridge is a block of
  rows, and the least squares over the simplex is one non-negative least squares, Lawson and
  Hanson's active set (`scipy.optimize.nnls`), by the reduction `chc.scm` documents for a synthetic
  control. The data are scaled by `sigma` first. Stationarity on the support and the multipliers'
  signs off it are checked to `1e-10` of the problem's scale, and weights that miss raise
  `RuntimeError`. Over 1500 panels like the size test's the residual was at most `4.5e-16`. Maxima
  enumerated every support of both problems on a panel of five geos and five weeks in exact
  arithmetic; one point meets the conditions in each, the solver's to `1e-12`, and a test repeats
  the enumeration with `fractions.Fraction` against Maxima's rationals. The enumeration is
  `validation/sdid_simplex_weights.mac` (amended 2026-10-28), each step ending in a PASS or FAIL
  line; it also finds that without the intercept the unit weights' optimum is another point, and a
  test runs it where `maxima` is installed.
- **The placebo variance (Algorithm 4) by default.** `N_tr` controls drawn as placebo treated
  geos, the estimator run whole on the controls alone, the variance the placebo effects'. It holds
  with one treated geo. Where there are no more assignments than `replications`, every one is read
  once, so one treated geo among 20 controls reads its 20 placebos, not draws of them. Draws come
  from `SeedSequence(seed).spawn(1)[0]`, never from `default_rng(seed)`, the stream a caller's
  own data may use (amended 2026-10-28); on the tests' panel of three treated geos among 20
  controls the placebo error at seed 0 moved from 4.3610 to 4.9315, a draw's own error.
- **The jackknife (Algorithm 3) on request**, leaving out every geo, controls included, with the
  weights held and renormalised over the geos left, its variance about the full estimate. It
  refuses fewer than two treated geos.
- **A t's interval and p-value, not the normal's.** The paper reads either variance against the
  normal. Over 2000 panels of 20 controls, 10 weeks before and 5 after, a level of sd 2 a geo, a
  shock a week and unit noise, with no effect, the normal test rejected 0.0735, 0.0825, 0.067 and
  0.063 of the time with 1, 2, 3 and 5 treated geos, above the 0.0597 that 2000 panels allow a
  5 % test. The placebo variance is unbiased there (its mean over the panels within 1 % of the
  estimate's variance with one treated geo) but is read with the error of a sample variance of some
  12 degrees of freedom, not the 19 its 20 controls suggest: with every time weight held equal the
  count was 18.7, with the time weights fit 11.5. The time weights are fit on the controls, `|S|`
  of them non-zero, so the controls' gaps keep `N_co - |S|` degrees of freedom, as a regression's
  residuals keep their count less its parameters; on that panel `|S|` averaged 7.2. The test is a
  t's at `nu = N_co - |S|`, at least 1; `B` random placebo draws add a sample variance's own error,
  so `1 / nu = 1 / (N_co - |S|) + 1 / (B - 1)`, Satterthwaite's count. It rejected 0.0505, 0.0545,
  0.040 and 0.042. The jackknife's variance with few treated geos rests on their `N_tr`
  leave-one-out terms, so its count is the smaller of `N_co - |S|` and `N_tr - 1`: it rejected
  0.0015, 0.0165 and 0.0285 with 2, 3 and 5 treated geos, against 0.1855, 0.1295 and 0.100 at the
  normal. `quantile="normal"` reads the paper's normal.
- **The count floored at `(N_co - 1) / 2`** (amended 2026-10-28): `nu_0 = max(N_co - |S|,
  (N_co - 1) / 2)` in place of `N_co - |S|`, in the placebo's count and the jackknife's. With 5 to
  10 controls the time weights' support nears or passes `N_co` in every panel, so the count fell
  to 1 and the t rejected almost nothing (below); the placebo variance's own count, read from its
  spread over the panels, was 2 to 8 there. `N_co / 2` rejected 0.056 with seven controls and one
  treated geo over 22 000 panels; `(N_co - 1) / 2`, half a sample variance's count, 0.048. At 20
  controls `N_co - |S|` is mostly above the floor and the rejections above hold.
- `read_sdid(panel, treated, start, end, *, spend, planned, ...)` reads the dates explicitly, every
  geo not treated a control unless `control=` names the controls (amended 2026-10-28: a geo in
  neither group is not read, so its missing weeks do not refuse the test, as with a `GeoTest`), and
  every period before `start` the pre-period, refuses a KPI missing over them, and returns the
  `LiftEstimate` (the lift summed over the treated geos and the readout) with an `SdidReading`: the
  interval, the p-value, a verdict on the interval's sign, the weights.
- **The `LiftEstimate`'s error widened to the t** (amended 2026-10-28): at `quantile="t"` it is
  `se t_nu(0.975) / z(0.975)`, as ADR 0047's readers hand on theirs, so that the normal 95 %
  interval a model reads of it is the reading's t interval; at `quantile="normal"` it is `se`.
  Handed on unwidened, a test of 3 treated geos and 7 controls over 52 weeks read at the normal
  rejected 0.1275 of 400 panels with no effect.

## The t's count with few controls (amended 2026-10-28)

2000 panels a cell with no effect, read at 5 %; "power" at an effect of 1.5 a geo-week. Panel A
is the size test's: 10 weeks before and 5 after, a level of sd 2 a geo, a shock a week and unit
noise. Panel B: 48 weeks before and 4 after, the same level and shock and an autoregression of 0.5
in place of the unit noise. Each cell reads the rejection rate / the power; 2000 panels allow
0.0403 to 0.0597 about a 5 % test.

| Controls, treated | `\|S\|` A | placebo at `N_co - \|S\|` A | at the floor A | normal A | `\|S\|` B | at `N_co - \|S\|` B | at the floor B | normal B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5, 1 | 5.1 | 0.0025 / 0.02 | 0.0255 / 0.20 | 0.1845 | 17.4 | 0.0000 / 0.00 | 0.0140 / 0.07 | 0.1425 |
| 5, 3 | 5.3 | 0.0000 / 0.03 | 0.0085 / 0.31 | 0.1360 | 17.3 | 0.0000 / 0.00 | 0.0060 / 0.08 | 0.0760 |
| 6, 1 | 4.8 | 0.0095 / 0.06 | 0.0410 / 0.24 | 0.1505 | 13.0 | 0.0005 / 0.00 | 0.0195 / 0.13 | 0.1380 |
| 7, 1 | 4.9 | 0.0120 / 0.13 | 0.0415 / 0.27 | 0.1240 | 9.8 | 0.0010 / 0.01 | 0.0240 / 0.15 | 0.1200 |
| 7, 2 | 5.0 | 0.0060 / 0.17 | 0.0320 / 0.41 | 0.1235 | 10.2 | 0.0000 / 0.01 | 0.0175 / 0.23 | 0.1030 |
| 7, 3 | 4.9 | 0.0020 / 0.20 | 0.0215 / 0.47 | 0.0910 | 10.1 | 0.0000 / 0.01 | 0.0195 / 0.26 | 0.0920 |
| 8, 1 | 5.1 | 0.0265 / 0.20 | 0.0455 / 0.31 | 0.1200 | 8.4 | 0.0040 / 0.03 | 0.0425 / 0.18 | 0.1260 |
| 10, 1 | 5.6 | 0.0430 / 0.34 | 0.0505 / 0.39 | 0.1085 | 7.3 | 0.0125 / 0.09 | 0.0335 / 0.20 | 0.0885 |
| 10, 3 | 5.6 | 0.0200 / 0.63 | 0.0245 / 0.69 | 0.0675 | 7.4 | 0.0050 / 0.17 | 0.0315 / 0.39 | 0.0855 |
| 20, 1 | 7.1 | 0.0525 / 0.53 | 0.0525 / 0.53 | 0.0765 | 9.4 | 0.0355 / 0.29 | 0.0365 / 0.29 | 0.0675 |
| 20, 3 | 7.1 | 0.0425 / 0.94 | 0.0425 / 0.94 | 0.0610 | 9.3 | 0.0425 / 0.61 | 0.0440 / 0.61 | 0.0645 |

Over the 18 cells of each panel, 5, 6, 7, 8, 10 and 20 controls by 1 to 3 treated geos, the floor
rejected at most 0.0525 with the placebo (at 20 controls, where it seldom binds) and 0.0510 with
the jackknife. The jackknife's rejections with three treated geos and 5 to 8 controls rose from
0.0095-0.0325 to 0.0300-0.0510 in panel A and from 0.0070-0.0260 to 0.0405-0.0490 in panel B, its
power at 5 controls from 0.10 to 0.40 and from 0.05 to 0.27; with two treated geos its count is
`N_tr - 1` either way. `N_co / 2`, the next floor up, rejected up to 0.0530 and `0.6 N_co` up
to 0.0655. The normal over-rejects in every cell. A test with seven controls and one treated geo
holds its rejection within the binomial's three standard deviations of 0.05 either side.

## The variance by the number of treated geos

| Treated geos | Placebo (Algorithm 4) | Jackknife (Algorithm 3) | Bootstrap (Algorithm 2) |
| --- | --- | --- | --- |
| 1 | the only one that reads; every placebo once | refused: no treated geo left out | undefined: draws with no treated geo |
| 2 to 5 | the default; holds its level at the t | conservative at the t, far above its level at the normal | not offered |
| many | valid where treated and controls are alike | valid, and needs no refit | the paper's choice; not offered |

The placebo assumes the treated geos' noise is the controls'; where the treated geos were chosen
for being unlike the rest, its variance is the controls', not theirs. The jackknife needs no refit
and reads the treated geos' own noise, which with few of them is little.

## Consequences

- A test with few controls reads a wider interval than the paper's: at 20 controls and some 7 time
  weights, a t at 12 degrees of freedom, 2.18 against 1.96. The standard error handed on in the
  `LiftEstimate` is the paper's widened by that ratio (amended 2026-10-28; it was the paper's).
- Each placebo refits both weight problems, so the placebo costs `replications` fits.
- The panel must be balanced over the periods read.
- *On null panels* (amended 2026-11-06). ADR 0058's audit (`validation/aa_audit.md`: 200 null
  panels of 104 weeks a cell, the false-positive rate at a nominal 0.05, the band [0.03, 0.08])
  reads `read_sdid` at its placebo variance under AR(1) noise, a trend and a shift, at 10 and 30
  geos: 0.055 and 0.050, 0.020 and 0.035, 0.030 and 0.030, in the band in five of six cells and
  below it under a trend over 10 geos, where it is conservative; at `variance="unit_placebo"`
  0.040 and 0.030, 0.045 and 0.045, 0.040 and 0.050, in the band in all six.

## Alternatives

- **Frank-Wolfe**, as the authors' package solves: rejected. It stops at a tolerance on the
  objective's decrease, not on the conditions of optimality, and its weights depend on that
  tolerance; the active set ends at the optimum.
- **A box then renormalised**: rejected, above.
- **The normal quantile**, the paper's: kept as `quantile="normal"`, not the default, for the
  rejection above.
- **The placebo distribution's own quantile, a rank test**: rejected. The placebos are drawn from
  the controls alone, so the treated estimate is not exchangeable with them; it rejected 0.058 to
  0.0755 with 1 to 3 treated geos.
- **A t at `N_co - 1`**: rejected; it rejected 0.0615 with one treated geo, the time weights'
  degrees of freedom not counted.
- **The bootstrap (Algorithm 2)**: not offered. The paper recommends it with many treated geos,
  where the placebo and the jackknife also hold; with one it is undefined, and with few its draws
  hold no treated geo or one.
