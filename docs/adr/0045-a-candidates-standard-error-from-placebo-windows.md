# 0045. A candidate's standard error from placebo windows

Status: accepted, 2026-10-13; adds to ADR 0030's `GroupNoise` a second noise model for `designs`;
audited on null panels by ADR 0058 (`validation/aa_audit.md`), through the geo readers that share
its windows' effective count: `read_did` within the band in five of six cells, `read_tbr`
conservative in every cell; amended 2026-11-06: the audit's cells, reader by reader, under
Consequences

## Context

ADR 0030 sets a candidate test's standard error before it runs by `GroupNoise(market_sd)`:
`market_sd sqrt(R s / (1 - s))` over a readout of `R` periods in a group of share `s`. That is the
error where the periods' noises are independent. A geo's noise often carries over from one week to
the next. Under a first-order autoregression of coefficient `phi`, a sum over `R` periods has
`f(R, phi) = 1 + 2 sum_{k<R} (1 - k/R) phi^k` times the variance of `R` independent periods
(Maxima; the closed form is `(1 + phi)/(1 - phi) - 2 phi (1 - phi^R) / (R (1 - phi)^2)`). At a
12-week readout `f` is 10923/4096 at `phi = 1/2` and 287877519/48828125 at `phi = 4/5`, so
`GroupNoise` understates the error 1.63 and 2.43 times. Over 40 000 simulated markets of 104 weeks,
its 95 % interval held the lift 0.950, 0.770 and 0.582 of the time at `phi` 0, 0.5 and 0.8.

## Decision

- **`PlaceboNoise(residuals, pre=26)`** reads the error from the market's own noise over the
  history before the test: the residuals a model of the KPI leaves. `se(readout, share)` has the
  signature of `GroupNoise.se`, and `designs` takes either (`Noise = GroupNoise | PlaceboNoise`).
  `GroupNoise` stays the default the callers pass; nothing changes for them.
- **Placebo windows in time.** Every window of `readout` periods that has `pre` periods before it
  is a placebo test. Its sum holds whatever dependence the noise has inside a window, so no model
  of that dependence is fitted.
- **Studentised windows.** Each window's sum is divided by the root mean square of the noise over
  its own `pre` periods before it, and the test's error is read in units of the `pre` periods
  before the test. This is the role of Chung and Romano's (2013, Ann. Statist. 41, Theorem 2.2)
  studentised permutation test: units that are not exchangeable, there two samples of unequal
  variance, here windows at different noise levels, give a placebo distribution of the right
  limit only once each statistic is studentised by its own scale. Their theorem is stated for
  independent samples; here it is the reason for the design, not a proof. With the noise's size
  trebling over the history, the studentised interval held the lift 0.960 of the time over 40 000
  markets, an unstudentised prototype's 0.823.
- **A t's quantile at the windows' effective count.** The windows overlap and the noise carries
  over, so their mean square is far noisier than their count suggests. The error is widened so
  that its normal 95 % interval is a t's at `nu = 2 n^2 / ((kappa - 1) sum rho^2)`, `rho` the
  studentised sums' autocorrelation at every lag and `kappa` their kurtosis, at least 3. Over
  40 000 markets of 104 weeks the interval held the lift 0.951, 0.952 and 0.948 of the time at
  `phi` 0, 0.5 and 0.8, at a median width 1.17, 1.22 and 1.31 times the error's spread.

## Consequences

- An interval read from 104 weeks is wider than the true error's: few effective windows are the
  price of reading the dependence from the history alone. A longer history narrows it.
- The residuals are the caller's: the model that leaves them, and whether its fit absorbed some
  of the noise, are outside this class.
- A test inside the history (`Within`) needs residuals that end where the test starts; the class
  reads the periods after its residuals as the test's.
- *On null panels* (amended 2026-11-06). ADR 0058's audit (`validation/aa_audit.md`: 200 null
  panels of 104 weeks a cell, the false-positive rate at a nominal 0.05, the band [0.03, 0.08])
  reads the geo readers that share these windows' effective count. Under AR(1) noise, a trend
  and a shift, at 10 and 30 geos: `read_did` 0.025 and 0.055, 0.065 and 0.065, 0.065 and 0.045,
  in the band in five of six cells, the sixth 0.036 over 1000 draws; `read_tbr` 0.010 and 0.015,
  0.010 and 0.020, 0.015 and 0.025, below the band in all six, conservative.

## Alternatives

- **Fit `phi` and scale `GroupNoise` by `sqrt(f(R, phi))`**: rejected. It is right only where the
  noise is a first-order autoregression; the windows need no such model.
- **The variance-matched widening `sqrt(nu / (nu - 2))`**, as the bench's geo placebo uses: held
  0.937 at `phi = 0.8` in a prototype, its count read to twice the readout without the kurtosis;
  rejected for the 95 % quantile, since the candidate's error feeds a 95 % reading.
- **The permutation quantile itself, `before` times the 95th percentile of the windows' absolute
  studentised sums**: held 0.904 to 0.926 at 104 weeks. A percentile of some 70 overlapping
  windows is too noisy.
- **A Satterthwaite count adding the test's own scale's noise**, read from the residuals'
  autocorrelation: unstable at `phi = 0.8`, where the count fell to 2 or below.
