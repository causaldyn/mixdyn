# 0038. A lift test read by the DLM

Status: accepted, 2026-10-11; measured 2026-10-31 on a stale test (`tests/test_lift_staleness.py`):
on 20 worlds whose return per euro is 2 through a go-dark test at weeks 60 to 63 and falls
linearly by 40 % to 1.2 at week 104, the 90 % interval of the week-104 return covers it in 17 of
20 for the DLM given the test, its discounts chosen by `choose_dlm` (16 of 20 without it), and in
0 of 20 for the static Bayesian model given it (ADR 0015), whose median there is 1.99 (0 of 20
without it too: its one coefficient, not the test, loses the drift). The DLM reads the test at its
weeks and the KPI after them at theirs; amended 2026-11-06: the test's estimand is stated in one
line by `mixdyn.api.documents.estimands` (the `"mixdyn.lift_tests"` document): the weeks it changed,
the spend against the plan, the readout, and whether its lift is read against zero spend or as a
marginal change (`tests/test_documents.py`); `tests/test_lift_staleness.py` holds the stale test's
coverage above and the heavy-up read through the transform at its own spend

## Context

ADR 0015 reads a lift test into the static model's likelihood: the lift over the readout, at the
standard error the analysis reported, against the model's response to the spend the test changed.
The dynamic model over media (ADR 0020) read none. Its coefficient moves from period to period, so
a test reads it over the readout's periods, and `LiftEstimate` held no place in the calendar
(ADR 0037).

The drift run read the cost. Its worlds' channel effects drift by five laws and stay still by a
sixth, and its ladder's top rung hands every arm four go-dark tests. There every arm that read the
tests planned with less regret per euro than the DLM, which read none, by the largest margin in
the run. PyMC-Marketing's static model's regret fell with the tests; the DLM's plan at four tests
was its plan at none. Without tests the DLM's move from the status quo was as often away from the
oracle's as toward it, and the direction is what a go-dark test reads.

A test of a channel reads, at the market's scale,

    lift / share = sum over the readout of b_t [g(spend / share)_t - g(planned / share)_t] + e,

`b_t` the channel's coefficient at period `t`, `g` the effect's column, the spend carried over and
saturated by its transform, and `e ~ N(0, (se / share)^2)`: the static model's reading with the
coefficient at each period in place of one coefficient.

## Decision

- **A test is placed by its readout's end.** `LiftEstimate.end` is the date of the readout's last
  period in the panel's calendar. The DLM refuses an estimate without it, one whose readout runs
  outside the panel's calendar, one in another geo, and a test of a control; the static models pass
  it by. This revises ADR 0037's "`LiftEstimate` gains no place in the calendar": the place says
  which coefficients a test reads. The cells the readout reads still stay in the likelihood.
- **Nothing is linearised.** Given a spec the transform is fixed, so the reading is linear in the
  state path. The transform's nonlinearity is read where the spec is chosen: each spec's
  likelihood holds its tests, below.
- **The tests are observations of the path, read exactly** (`mixdyn.models.dlm_lift`). The
  discounts set each step's evolution variance through the KPI's filter, in units of the variance
  `V` and from the design alone. With them held, the model is a Gaussian prior over paths given
  `V`, as `chc.dlm.constrained_sample` reads it, and a test is a linear observation of it. The
  posterior given the KPI and the tests is the KPI's smoothed path conditioned on the tests: each
  state's covariance with a test is carried from the readout's periods by the smoother's gains,
  back from those after the state and forward from those before it. It is the posterior a
  cumulator gives, a state summing the test's weights times the coefficient over the readout and
  observed once at its end, in a filter whose evolution variances are the KPI filter's. The tests
  do not change those variances.
- **The variance is plugged in.** A test's error is the analysis's, in the KPI's units, and `V` is
  learned. A test enters at `se^2 / S_T` times `V`, `S_T` the KPI filter's estimate at the last
  period: its own variance where `V = S_T`. The model stays Normal-Gamma, so the tests add to the
  degrees of freedom and every reading of a fit, its Student t at the last period, its draws and
  its forecasts, is the conjugate one. With the variance known the reading is exact.
- **Tests that overlap are each an observation**, their errors apart: two analyses of one readout
  are two tests. A test outside the panel's calendar is refused, not truncated: the part outside
  reads coefficients the fit does not hold. A variance discount below 1 is refused: no one variance
  states a test's error.
- **Each spec's likelihood holds its tests.** A spec's log-likelihood is its KPI filter's plus its
  tests' log density given the KPI, a multivariate Student t: the joint density of all it read.
  `choose_dlm` and `search_media` take `estimates`, keep the specs within `chi2_p(level) / 2` of
  the best on that likelihood (`within`, the rule `chc.dlm.confidence_set` applies to the KPI's),
  and fit the kept specs given the tests. `DlmChoice.lifts` holds each spec's tests' log density.
  The readings over the kept specs move with the tests through the specs kept and through each
  fit's posterior.
- **A signed fit draws given the tests** (`signed_sample`). It is CHC's sampler with one block
  changed: the signed coefficients' distinct values given `V` have the KPI's Gaussian conditioned
  on the tests, truncated to the signs by CHC's exact Hamiltonian trajectory; the trajectory's
  velocity is CHC's draw moved by the tests it would have read; and `V` given the values counts the
  tests' errors. A test reads only the values, so the other states given them are drawn as CHC
  draws them. A test of a channel not held to a sign in a signed fit is refused. With no test the
  sampler is CHC's, draw for draw.
- **Every reading of a fit is the posterior given the tests.** The decomposition's draws move the
  KPI's backward draws by the tests they would have read (Matheron's rule), at a variance drawn
  from its posterior given the tests. A fit's channels are offered at the last period only, as a
  signed fit's are. A forecast starts from the state at the last period given the tests and
  evolves as the KPI's filter does, each step's evolution variance the filter's in units of the
  variance. The filter itself stays the KPI's: its one-step errors are not the tests'.
- **The scorecard's `dlm` arm reads the tests**, each lift row as the static arm reads it
  (`scorecard.arms.panel.estimates`), placed at its readout's last week, one fit a rung. ADR 0021's
  `dlm` read none. `dlm-untested` is the same arm handed none: its regret less `dlm`'s is what
  reading the tests is worth.

## Consequences

- The oracles. Maxima holds the closed form on one channel over three weeks, a test over the last
  two, with the variance known and learned: the smoothed means, variances and lag-one covariances
  given the test, the degrees of freedom and the variance's estimate are the batch posterior's that
  holds the test as one more observation, as exact rationals, and the log-likelihood is the batch's
  log density to `1e-40` at 60 digits. Fourteen checks hold and six controls fail
  (`tests/fixtures/oracles/dlm_lift_maxima.json`). On 24 weeks of a level, a season, two
  transformed channels and a control, a week of the KPI missing, a channel idle three weeks in six
  and three tests, two overlapping and one in half the market, the fit is a NumPy batch posterior's
  to `1e-9`, known and learned. The signed sampler given a test is rejection sampling's from the
  posterior given it, within 4.5 Monte Carlo errors. Twenty mutants of the reading, the
  conditioning, the variance, the draws, the sampler, the forecast, the spec's likelihood and the
  arm each fail a test.
- The plug-in, measured. The exact reading holds a test's error at its own variance whatever `V`:
  the posterior is a mixture over `V`, computed by quadrature over `log V`, which matched Maxima's
  `quad_qagi` on the three-week instance to `1e-11`. There, the KPI read at 6 degrees of freedom,
  the plug-in's last coefficient is 0.121 of the exact SD above the exact mean, its SD 1.099 times
  the exact, and the test's log density 0.067 lower. On 18 of the drift family's pilot worlds,
  three a law, at four tests, the KPI read at 157 degrees of freedom and the arm's starting spec
  without its signs, over 54 tested coefficients: at the last week the plug-in's mean lies within
  0.077 exact SDs of the exact and its SD at 0.998 to 1.014 times; over the readouts' weeks within
  0.124 SDs and at 0.994 to 1.047 times; the tests' log density differs by -0.016 to +0.180,
  against a likelihood-ratio margin of at least 1.92.
- `signed_sample` imports CHC's private helpers (`chc.dlm._structure`, `_transitions`,
  `_unit_filter`, `_path_covariance`, `_reflected_orbit`, `_convergence`): a CHC release that moves
  them breaks it, and a test that compares it with `constrained_sample` draw for draw says so. Its
  natural home is CHC's sampler taking linear observations of the path.
- A fit read with tests holds `DlmFit.lifted`; `DlmFit.log_likelihood` and `DlmFit.last()` read
  it. `DlmFit.filtered` and `DlmChoice.filtered` stay the KPI's filters.

## Alternatives considered

- **The tests as a prior on the initial state.** Rejected: a test reads the coefficient over its
  readout, not at the first period. As a prior it would be read before the KPI, dated at the start
  and carried forward by the discounts, which forget it at their rate; a test in the readout's own
  weeks would come back as a statement about week 1. Moving it back to week 1 exactly is the
  smoothing this decision does, and placing it there leaves it out of the spec's likelihood.
- **The tests as a penalty** on the smoothed path's lift. Rejected: a penalty moves the mean and
  leaves the uncertainty, the draws and the forecasts as the KPI's, and its weight is a tuning
  constant that the spec's likelihood does not read. At the weight `1 / se^2`, with the covariances
  conditioned and the density in the likelihood, it is this decision.
- **A cumulator inside the discount filter.** Rejected: the discount would then set each step's
  evolution variance from the posterior given the test, so a precise test would change how fast
  the coefficient drifts after it, and the filter, smoother and sampler would leave CHC's. Held at
  the KPI filter's variances, a cumulator gives this decision's posterior.
- **The variance read exactly**, a test's error at its own variance whatever `V`. Rejected: the
  posterior is then a mixture over `V`, not Normal-Gamma, and every reading, the Student t, the
  draws, the sampler's `V` block and the forecast, loses its conjugate form. What the plug-in costs
  is measured above.
- **Reweighting the KPI's draws** by the tests' likelihood. Rejected: precise tests leave a few
  draws with all the weight, and the spec's likelihood would be a Monte Carlo estimate.
