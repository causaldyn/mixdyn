# 0046. A lift test read period by period

Status: accepted, 2026-10-13; adds to ADR 0015's estimates a second form of a test in the
Bayesian model's likelihood, its covariance read by ADR 0045's noise models; audited 2026-10-31
(`tests/test_lift_staleness.py`): no path of `calibration/lift.py`, `models/dlm_lift.py`, the
static model's estimate and series terms or `decisions/returns.py` turns a test's lift into a
return-level quantity, and the return prior (ADR 0053) reads no test. Each reads
`beta [g(spend) - g(planned)]` at the test's own spend. A heavy-up at ten times the model's spend,
the tanh curve held at a point, moves the static model's return at its own spend to the return its
contrast implies, to `1e-5`, where the plug-in, the lift over the spend added, reads 0.118 of it;
the DLM's weights are the transform's contrast to `1e-12`. A mutant of either reading linearised
at the plan fails its test; amended 2026-11-06: the test's estimand is stated in one line by
`mixdyn.api.documents.estimands` (the `"mixdyn.lift_tests"` document): the weeks it changed, the
spend against the plan, the readout, and whether its lift is read against zero spend or as a
marginal change between the two levels (`tests/test_documents.py`), the contrast this audit reads;
`tests/test_lift_staleness.py` holds the audit's two heavy-up readings

## Context

ADR 0015 reads a test as its analysis reports it: one lift over the readout and its standard
error. One number reads one direction of a channel's parameters. On the return and the decay its
Fisher matrix is the outer product of one gradient, rank 1, whatever the error. Worse, for a
go-dark test whose readout holds the whole carryover and a curve that is straight over the test's
spend, the lift does not move with the decay at all. Under a normalised geometric kernel of
length 6, a test dark for 2 periods and read over 8, the summed lift is `-2 beta` and its
derivative in the decay is 0 (Maxima). The decay is then read from the KPI's history alone, where
spend that moves slowly leaves it weakly identified, and from the prior.

The gaps the test reads period by period hold the decay in their shape: the gap rises while the
spend is off and falls back as the carryover runs out. On the same design their Fisher matrix on
the coefficient and the decay has the determinant `beta^2 p(r) / ((r + 1)^4 (r^2 - r + 1)^4
(r^2 + r + 1)^4)`, `p` a polynomial of degree 16 whose coefficients are all positive (Maxima), so
it is rank 2 at every decay; under a first-order
autoregressive noise at `phi = 1/2` it is rank 2 again, and at a Hill curve's point (coefficient
2, decay 1/2, half-saturation 0.7, slope 1.5) both eigenvalues are positive where the summed
lift's determinant is 0. A return is the coefficient times a positive function of the decay, so
the rank on the return and the decay is the same. `chc.lift.fit_lift` already reads tests this
way, by least squares under independent noise of one scale.

## Decision

- **`mixdyn.calibration.LiftSeries(spend, planned, gaps, covariance, share=1.0, geos=None)`**:
  `LiftEstimate`'s test with the gap at each period of the readout, `gaps`, and the covariance of
  their errors. The readout is the gaps' length. It refuses what `LiftEstimate` refuses of the
  spend, the share and the geos; gaps that are not a non-empty finite vector or are longer than
  the spend; and a covariance that is not finite, of the gaps' size, symmetric to `1e-12` of its
  largest entry and positive definite. `summed` is the same test as one `LiftEstimate`: the gaps'
  sum at the error of that sum.
- **A term of the likelihood**, through `series=`, each channel's, in `bayes_model` and
  `fit_bayes`:
  `gaps / share ~ MVN(beta_m [s_m(adstock_m(spend / share)) - s_m(adstock_m(planned / share))]_t, covariance / share^2)`
  over the readout's periods `t`, on the model's scales, as an estimate's gap is computed. No
  parameter is added: the covariance is the analysis'. The density is computed through the
  inverse of the covariance's Cholesky factor, once, when the model is built.
- **Each series is one observation** of the pointwise log-likelihood, after the estimates: its
  periods are correlated, so leave-one-out leaves out the test. In the panel's units its density
  is over the KPI's unit to the power of its readout.
- **The covariance from the noise models**: `GroupNoise.covariance(readout, share)` holds each
  period's variance on the diagonal. `PlaceboNoise.covariance(readout, share)` is the residuals'
  uncentred autocorrelation at each lag below the readout as a Toeplitz matrix, positive definite
  as a sample autocorrelation's is, scaled so that its sum is `PlaceboNoise.se(readout, share)`
  squared. The gaps' sum then has exactly the summed test's error, so a series and its sum read
  the same total at the same precision and differ only in the shape.
- **The default is unchanged.** `series` is `None` unless given. The model over geos, the nested
  model and the DLM do not read series.

## Consequences

- *The oracle*, `chc.lift.fit_lift`. Three tests of one channel, a go-dark and two heavy-ups,
  read period by period under independent noise of one size, on a world of the model's family
  with a tanh curve: the series' terms alone, maximised over the channel's return, decay and
  half-saturation point, peak at `fit_lift`'s least-squares fit to `3.4e-10` of each parameter
  (the test holds it to `1e-8`), and there their sum is the normal density of `fit_lift`'s least
  cost to `1e-12`.
- *Each series is a joint normal* about the gaps `chc.response` computes for the draw's channel,
  at the series' covariance over the share squared, to `1e-11`, for a Hill and a tanh, on either
  channel, and with a series longer than the panel, which sets the kernels' length. A series of
  independent periods is, value and gradient, the sum of its one-period estimates; a series adds
  its own term and moves nothing else.
- *The ridge*, on a noiseless test of a Hill world: at every decay from 0.1 to 0.9 the summed
  lift is fitted exactly by moving the return, its density at its top; the gaps are fitted exactly
  at the world's decay alone and fall short by more than 5 in the log elsewhere.
- *Recovery* (`scripts/study_lift_series.py`): 100 worlds of one channel at each decay, 0.3 and
  0.7, and each geo noise's carry-over, `phi` 0 and 0.5, their history's spend moving slowly; one
  go-dark test in half the market read as its sum and as its series, both at `PlaceboNoise`'s
  error, fitted on the same world and seed. The decay's posterior mean less the world's, its root
  mean square, and the central 90 % interval's coverage and width, Monte Carlo standard errors in
  brackets:

  | decay | phi | read | bias | RMSE | coverage | width |
  |---|---|---|---|---|---|---|
  | 0.3 | 0 | sum | -0.050 (0.010) | 0.107 | 0.91 (0.03) | 0.399 (0.006) |
  | 0.3 | 0 | series | -0.019 (0.006) | 0.060 | 0.86 (0.03) | 0.193 (0.004) |
  | 0.3 | 0.5 | sum | -0.050 (0.010) | 0.107 | 0.90 (0.03) | 0.399 (0.006) |
  | 0.3 | 0.5 | series | -0.021 (0.006) | 0.062 | 0.90 (0.03) | 0.220 (0.005) |
  | 0.7 | 0 | sum | -0.084 (0.010) | 0.127 | 0.86 (0.03) | 0.307 (0.011) |
  | 0.7 | 0 | series | -0.012 (0.003) | 0.035 | 0.89 (0.03) | 0.112 (0.003) |
  | 0.7 | 0.5 | sum | -0.086 (0.010) | 0.129 | 0.88 (0.03) | 0.314 (0.012) |
  | 0.7 | 0.5 | series | -0.022 (0.004) | 0.047 | 0.87 (0.03) | 0.147 (0.004) |

  The series narrows the interval 1.8 to 2.7 times and cuts the RMSE 1.7 to 3.7 times, most at
  the high decay, where the sum leaves the decay to the prior, Beta(1, 3), and is pulled down by
  0.08. Every coverage is
  within 1.33 standard errors of 0.90. The series keeps a bias of 0.01 to 0.02 toward the prior.
  Carry-over in the noise widens the series' interval, read through its covariance, and leaves
  the sum's, whose error already holds it. Divergent transitions: 28 of 200 000 in the series'
  fits at 0.7 and no carry-over, 17 in the sum's, at most 4 in every other cell.
- *Mutations*: each of 19 fails a test that fits no posterior: the summed error from the
  covariance's trace; the positive-definite check, the symmetric part, the series' channel and
  geo checks each dropped; the share dropped from the scale, the covariance scaled by the scale
  rather than its square, whitened by the wrong triangle, the normal constant dropped; the readout
  a period early, the first channel's coefficient for every series, the planned arm dropped, a
  series counted as one unit of the KPI, the series dropped from the likelihood or by
  `bayes_model`, the kernels blind to a series' length; `PlaceboNoise.covariance` unscaled or its
  autocorrelation centred; `GroupNoise.covariance` not over the readout.
- `PlaceboNoise.covariance`'s shape is the residuals' sample autocorrelation, noisy at the long
  lags on a short history; its total is the placebo error, widened as ADR 0045 widens it, so an
  interval read from it is conservative where that one is.
- The gaps of a geo test read as a difference in differences share the pre-period's means, so
  their errors are correlated even where the noise is not; a covariance that ignores it is too
  narrow on the gaps' level.

## Alternatives

- **The gaps under independent noise of a sampled scale**, as ADR 0010 reads `chc.lift.LiftTest`'s
  arms. Rejected: a geo's noise carries over from one period to the next, and an analysis reports
  its error; one scale across periods misweighs both.
- **A first-order autoregression's `phi^|i - j|` with `phi` fitted from the residuals**. Rejected
  as ADR 0045 rejects it for the sum: right only for that process, where the autocorrelation needs
  no model.
- **The sample covariance of the placebo windows' gap vectors**. Rejected: 78 entries for 12
  periods from some 70 overlapping windows is near-singular; the Toeplitz form needs 12 numbers.
- **Series in `estimates=`, as a union of the two forms**. Rejected: the model over geos, the
  nested model and the DLM read `estimates` and would each have to refuse a series.
