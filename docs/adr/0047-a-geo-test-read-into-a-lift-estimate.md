# 0047. A geo test read into a lift estimate

Status: accepted, 2026-10-28; reads the `LiftEstimate` of ADR 0015 from a panel, so that a test
designed (ADR 0030, 0031) and run is read, calibrates the model and re-plans inside mixdyn; amended
2026-10-28: a readout whose blank and readout periods have too many arrangements for
`blank_period_inference` to enumerate (at 8 blank periods, over 11 weeks) is read without its
permutation p-value, and the placebo readers refuse a pre-period with room for fewer than two
windows after the warmup; audited on null panels by ADR 0058 (`validation/aa_audit.md`): `read_did`
and `read_synthetic_control` hold their level, `read_tbr` is conservative, and design-aware unit
placebos and `read_pairs`'s sign flip are opt-in; amended 2026-11-06: the audit's cells, reader by
reader, under Consequences

## Context

A model reads a lift test as a `LiftEstimate`: the lift over the readout and its standard error.
Until now the analysis that turns a geo test into those two numbers ran outside mixdyn. Three
choices decide whether that reading can be trusted:

- **Where the test is.** A reading that infers the test's start from the data reads a lift that is
  not the test's. With 10 geos over 52 weeks and +10 a geo-week over the last 4 weeks in 5 geos,
  a split at the median date reads 200 spread over 26 weeks and 5 geos, 1.5385 a geo-week where
  the effect is 10.
- **The standard error.** A geo's KPI noise carries over from week to week, and there are few
  geos. A standard error that reads the periods as independent is too small (Bertrand, Duflo and
  Mullainathan 2004, QJE 119, for differences in differences). One clustered by geo rests on many
  clusters and over-rejects with ten (Cameron, Gelbach and Miller 2008, REStat 90).
- **The verdict.** A test whose KPI fell significantly is a finding, not a success, and a model
  must read it as a lift below 0.

## Decision

- **`GeoTest`** names the test (named apart from ADR 0049's `GeoDesign`, the design before a test
  runs): the treated and the control geos, the channel, the test's first
  and last period as dates, the treated group's spend as run and as planned (the estimate's
  `spend` and `planned`) and the readout's length, counted from the test's first period so that a
  cooldown is read with it. The pre-period is every period before the first. Nothing is inferred.
- **Three readers** sum the treated geos' KPI `T` and the control geos' `C`:
  - `read_did`: the gap `g_t = T_t - b C_t`, `b` the ratio of the pre-period means, so the gap
    averages 0 before the test whatever the groups' sizes; the lift sums the readout's gaps less
    the mean pre-period gap. A ratio rather than a regression keeps the reading a difference in
    differences of one scale; the regression is TBR's.
  - `read_tbr`: Kerman, Wang and Vaver (2017, "Estimating ad effectiveness using geo experiments
    in a time-based regression framework"): `T` on `C` by ordinary least squares over the
    pre-period, and the lift the readout's `T` less the fit's prediction, summed.
  - `read_synthetic_control`: `chc.scm.blank_period_inference(serial="ar1")` on the treated geos'
    mean as one unit and each control geo as a donor, every unit less its mean over the fitting
    periods (the demeaned synthetic control of Ferman and Pinto 2021, Quantitative Economics 12),
    so that a convex combination of single geos of other levels can reach a group's mean; the
    lift is the treated geos' count times the gaps summed. Fewer than 3 donors are refused.
- **Standard errors that read the dependence over time.**
  - The difference in differences and TBR read theirs from **walk-forward placebo windows**:
    every window of the readout's length in the pre-period with at least `warmup = 13` periods
    before it is read as the test is, its ratio or regression refitted on the periods before it.
    Each placebo lift carries the estimator's own error, its parameters' included, and the noise's
    dependence over a window. The error is their root mean square; the reading is a Student t at
    the windows' effective count, ADR 0045's `nu`, now shared through a private helper so that
    `PlaceboNoise`'s outputs stay bit-identical (a test checks them by exact equality).
  - `read_tbr(variance="iid")` keeps Kerman, Wang and Vaver's own predictive variance,
    `s^2 (R + R^2 / n + (X_R - R xbar)^2 / S_xx)` at `n - 2` degrees of freedom, for errors that
    are independent.
  - The synthetic control reads its gaps as a first-order autoregression at the coefficient `rho`
    that `blank_period_inference` estimates from the donors. The readout's gaps are measured from
    the fitting periods' mean gap, so their sum is the contrast `c` (+1 on the readout, `-R /
    n_fit` on the fitting periods) of the gaps' path, of variance `-sum c_i c_j gamma(|i - j|)`
    with the semivariogram `gamma(d) = sigma^2 (1 + rho + ... + rho^(d-1)) / (1 + rho)`, which
    stays finite at `rho = 1`. `sigma^2` is the mean square of the gaps' innovations over the 8
    blank periods the weights never saw, and the reading a t at their count. Its per-period
    conformal intervals are not an interval of the sum. Its permutation p-value is kept beside the
    reading as `permutation_p`.
- **One level, a sign-keyed verdict.** `GeoReading` holds the lift, the error and its degrees of
  freedom, the two-sided interval at `alpha`, the p-value of the same t and the verdict: "up" when
  the interval lies above 0, "down" below, "inconclusive" when it holds 0. So `p < alpha` exactly
  where the verdict is not "inconclusive", and a significant fall reads "down".
- **The estimate.** `LiftEstimate(spend, planned, readout, lift, se t_df(0.975) / z(0.975), geos=
  treated, end=the readout's last date)`, its share 1 of the treated geos: a lift at or below 0 is
  kept, and the error is widened as ADR 0045's is, so that the normal 95 % interval the likelihood
  reads is the reading's t interval.

Over 10 000 panels (10 seeds of 1000) of 10 geos and 52 weeks, 5 treated over the last 4, geo
levels of standard deviation 1 and noise of 2 at an autoregression of 0.8, with no effect, a 5 %
test rejected:

| reading | rejects | reads "up" |
| --- | --- | --- |
| difference in differences | 0.0455 | 0.0227 |
| TBR | 0.0442 | 0.0212 |
| synthetic control | 0.0431 | 0.0207 |
| TBR, `variance="iid"` (Kerman's own) | 0.3205 | 0.157 |

The tests run one of those seeds for each reader and hold each rate within 2 Monte Carlo standard
errors of 0.05, and the "up" rate of 0.025.

## Consequences

- Design, run, read, calibrate and re-plan all run inside mixdyn: a reader's estimate goes to the
  models as any other `LiftEstimate`.
- The walk-forward placebos need the readout's length and 13 periods more in the pre-period; a
  short pre-period gives few windows, a small effective count and a wide interval. That is the
  price of reading the dependence from the history alone.
- The placebos read the pre-period as exchangeable in time: a noise level that moves over the
  history is read at its mean over it (`PlaceboNoise`'s studentised windows read the level before
  the test; see the alternatives).
- Each reader assumes the control geos' KPI moves as the treated geos' would have without the
  test: the difference in differences in proportion (a constant ratio of the groups' sums), TBR
  along a line fitted on the pre-period, the synthetic control as a weighted mean of single
  control geos up to a level. No spillover from the treated geos reaches the controls.
- The synthetic control's error assumes the gaps follow one first-order autoregression and that
  the donors' gaps persist as the treated unit's do; its 8 blank periods leave the fit, and a
  long readout with many blank periods can exceed the arrangements `blank_period_inference`
  enumerates, which it refuses.
- Synthetic difference in differences is a separate module and decision (ADR 0048).
- *On null panels* (amended 2026-11-06). ADR 0058's audit (`validation/aa_audit.md`: 200 null
  panels of 104 weeks a cell, the false-positive rate at a nominal 0.05, the band [0.03, 0.08])
  reads each reader at its default and each opt-in. Under AR(1) noise, a trend and a shift, at
  10 and 30 geos: `read_did` 0.025 and 0.055, 0.065 and 0.065, 0.065 and 0.045, in the band in
  five of six cells, the sixth 0.036 over 1000 draws; `read_tbr` 0.010 and 0.015, 0.010 and
  0.020, 0.015 and 0.025, below the band in all six, conservative; `read_synthetic_control`
  0.035 and 0.050, 0.045 and 0.055, 0.040 and 0.030, in the band in all six. Opt-in:
  `read_did` at `variance="unit_placebo"` 0.030 and 0.070, 0.030 and 0.045, 0.035 and 0.040,
  in the band in all six; `read_tbr` at `variance="unit_placebo"` 0.015 and 0.085, 0.030 and
  0.050, 0.045 and 0.045, out of the band under AR(1) noise (0.029 and 0.055 over 1000 draws);
  `read_pairs` refuses 5 pairs, so every 10-geo cell, and reads 0.070, 0.030 and 0.045 at 30
  geos.

## Alternatives

- **A standard error clustered by geo**: rejected. With 5 treated and 5 control geos its
  asymptotics in the count of clusters do not hold; it over-rejects with so few clusters.
- **`PlaceboNoise`'s studentised windows on the pre-period gaps** (ADR 0045): rejected for the
  readers. In a prototype over this world it rejected 0.073 with 26 periods studentising each
  window, and 0.053 to 0.076 over three seeds with 13: a scale read over a short stretch at an
  autoregression of 0.8 is itself noisy, and the effective count does not see that noise.
- **Placebo windows each read against the rest of the pre-period** rather than the periods before
  them: rejected; 0.058 to 0.072 over three seeds in a prototype. The real test's baseline lies
  wholly before it, and a window between two stretches of its baseline is read too tightly.
- **Kerman's predictive variance generalised to autoregressive errors**, the coefficient read from
  the residuals and corrected for its bias, at `n - 2` degrees of freedom or a Satterthwaite count
  including the coefficient's error: rejected; 0.056 to 0.111 in prototypes. The coefficient is
  read too low from 48 periods, and the variance is steep in it near 1. An effective count of
  `(n - 2) (1 - rho) / (1 + rho)` came closest, 0.046 to 0.067 over six seeds, but has no
  derivation behind it.
- **The synthetic control's error as a stationary autoregression about 0**, without the level
  read on the fitting periods: rejected; 0.053 to 0.096 over 8, 12 and 16 blank periods in a
  prototype. The fitting periods' mean is read with error, and that error moves every gap of the
  readout together.
- **The synthetic control's permutation p-value as the reading's**: kept beside it, not used. Its
  test has no interval of the sum, so the verdict and the p-value would rest on two levels.
