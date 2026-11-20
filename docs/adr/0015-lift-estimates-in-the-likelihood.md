# 0015. Lift estimates in the likelihood

Status: accepted, 2026-10-03; extended 2026-10-03 by ADR 0018: an estimate names the geos its test
ran in, and the model over geos and the nested model read estimates; amended 2026-11-06: estimates
read from PyMC-Marketing's lift table and from a geo experiment's record, and written back

## Context

A geo test's analysis, a difference in differences, a synthetic control or a time-based
regression, usually hands on one number and its standard error: the lift over the readout. ADR
0010's model reads a test period by period, each group's spend and outcome (`chc.lift.LiftTest`),
but most tests reach a modeller as the summary, not the series. Calibrating on those summaries,
with their errors, is the practitioners' first ask of the engine. Read as a prior on each
channel's coefficient instead, they are easy to misread: a lift at or below 0 dropped, a standard
error ignored, one channel's prior resetting another's, and a test's return per unit of spend
compared with a coefficient per unit of adstock.

## Decision

- **`mixdyn.calibration.LiftEstimate(spend, planned, readout, lift, se, share=1.0)`**: the tested
  group's spend on the channel as the test ran it and as planned, over the periods before the
  readout and the readout; the lift over the last `readout` periods and its standard error, in the
  KPI's units; the group's share of the market. It refuses a spend not finite and at or above 0, a
  test that changed no spend, a readout outside the spend, a lift not finite, an error not
  positive and finite, and a share outside `(0, 1]`. A lift at or below 0 is kept.
- **A term of the likelihood**, through `estimates`, each channel's, in `bayes_model` and
  `fit_bayes`:
  `lift / share ~ N(beta_m sum_readout [s_m(adstock_m(spend / share)) - s_m(adstock_m(planned / share))], se / share)`,
  on the model's scales, both series carried over and saturated as the channel's own, zero before
  them. No parameter is added: the error is the analysis's.
- **Each estimate is one observation** of the pointwise log-likelihood, after the KPI's periods
  and the geo tests' readout periods.

## Consequences

- *Three tests of one channel* (`tests/test_calibration.py`), each 10 of spend, lifts 10, -8 and
  2. The model reads them as one test at their precision-weighted mean lift with their summed
  precision, at every point to `2e-16` of the log density: a lift of 4/3 with equal errors, 6.00
  with errors 1, 100 and 1. Dropping the negative lift instead centres the evidence at 6, 4.5
  times as far under equal errors. Television's estimates add their own terms and nothing else,
  so search's gradient does not move.
- *A go-dark test on a world of this model's family whose decay is 0.5*, in a tenth of the market,
  read as an estimate at the world's return, half-saturation point and slope: its log density over
  the decays from 0 to 0.9 is largest at 0.5, a normal's at its mean.
- *The oracle*: Stan's reading of the model on its three worlds, each now with three lift
  estimates beside its three geo tests, a go-dark and a heavy-up in sub-markets and a national
  heavy-up. On each the log density differs from Stan's by one constant to `2e-15` of its size and
  the gradient is Stan's to `2e-15` of its largest entry; at 25 of Stan's draws each of the 119
  observations' log-likelihoods, the estimates' among them, is Stan's to `7e-14`. Four chains of
  1000 draws, from uniform starts on every world and from Pathfinder's on the normal one, put each
  parameter's posterior mean and SD and each observation's log-likelihood's within 4.5 joint Monte
  Carlo errors of Stan's 40 000 draws, with no divergent transition, as Stan had none, and every
  R-hat below 1.01.
- *Mutations*: each of 30 fails a test, 29 of them one that fits no posterior: the estimates'
  terms dropped, the error read as 1 or as a variance, another channel's coefficient, the gap's
  sign or its planned arm dropped, each share and unit dropped, the readout a period long or
  short, every row read as the first estimate, a mean for a sum, the kernels' length blind to the
  estimates, estimates of an unknown or the wrong channel or at the wrong spend unit, and each of
  `LiftEstimate`'s refusals and its read-only copies dropped. The 30th, `fit_bayes` dropping the
  estimates, fails the test that reads a short fit's log-likelihood at each draw.
- The model reads an estimate's error as exact. An analysis that understates it, a difference in
  differences with errors that ignore the serial correlation, outweighs the KPI, and the model
  cannot tell.
- The readout is the periods the lift sums over. An analysis that reports a lift past the readout,
  or a return rather than a lift, is not an estimate of this form until restated.
- An estimate's spend is zero before its series, as a test's is (ADR 0010): a pre-period shorter
  than the carryover understates the adstock at the readout's start.

### Amended 2026-11-06: tests read from other tools' tables

A buyer's tests are already written in another tool's form. `mixdyn.calibration.tables` reads two:

- **`read_lift_table(panel, table, *, periods, cooldown=0)`**: PyMC-Marketing's lift table, a row
  a test, `channel`, `x`, `delta_x`, `delta_y`, `sigma`, a `geo` column and a `date` where given.
  The row is a lift a period at a steady spend, and says nothing of how long the test ran, so the
  caller states it: the spend held at `x + delta_x` for `periods` periods, the lift read over those
  and `cooldown` more at `x`, `delta_y` times `periods` the lift and `sigma` times `periods` its
  error. That inverts the reduction the bench makes of a go-dark test for PyMC-Marketing's
  likelihood, the lift over the dark weeks and the cooldown per dark week: the test's lift and
  error read back, its spend at the dark weeks' mean level. A date is the first changed period, on
  the panel's calendar.
- **`read_experiments(panel, records, *, per_unit=False)`**: a geo experiment's record, as GeoX
  reports one and Meridian's calibration takes it: `estimate`, `standard_error`, the window
  `start` to `end` on the panel's calendar, the group's `spend` and `planned` spend a period over
  it, and where given `cooldown`, `geo` and `share`. The estimate is the lift, or, `per_unit`, a
  return a currency unit of spend changed (an iROAS, Meridian's ROI), times the spend the window
  changed.

Both read any column table `read_panel` reads, a mapping of columns or a frame, and return
`LiftEstimate`s by channel, so the estimand each names, a lift against zero spend or a marginal
change, is `mixdyn.api.estimands`'. `lift_table` and `experiment_table` write estimates back.
Read and written back, a record's lift and error are its own to the bit; a lift-table row's are its
`delta_y` and `sigma` times the period count, to the bit where that is a power of two and within
one rounding otherwise (`tests/test_lift_tables.py`, on tables written by hand in each format's
documented columns). An estimate a row cannot hold, a ramp, more than one geo or a share of a
market in a lift table, is refused at writing, not rounded into one.

## Alternatives considered

- **A prior on the return set from the tests**, Meridian's calibration.
  Rejected: a test measures the lift of a change of spend over its readout, not the return on the
  window's spend; under a curve and a carryover the two differ, and several tests of a channel
  have to be pooled by hand before the fit.
- **A lift on the saturation curve at a steady spend**, PyMC-Marketing's
  `MMM.add_lift_test_measurements` (1.2.0): `s(x + dx) - s(x)` against `|dy|` under a Gamma.
  Rejected: it reads no carryover, refuses a test whose lift has the sign opposite to its change of
  spend (`NonMonotonicError`), and gives a lift of 0 no density.
- **A calibration error among the search's objectives**, Robyn's `MAPE.LIFT`. Rejected: the tests
  choose among fits by their relative error rather than enter a posterior.
- **One noise scale shared with the geo tests**, the model's `test_sigma`. Rejected: a summary
  carries its own error, and one scale across estimates of different precision weighs them alike.
