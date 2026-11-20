# 0053. A return prior that counts the carry-out

Status: accepted, 2026-10-28; adds to ADR 0010's prior an opt-in on what its return counts, and
to ADR 0036's level the coordinate that follows it

## Context

The one-geo Bayesian model (ADR 0010) does not sample a channel's coefficient. It sets it from the
channel's return, `roi`, and states the prior `LogNormal(0.2, 0.9)` on that return:

    beta = roi * window_spend / sum_{t in window} s(adstock(x))_t

So `roi` is the window's contribution over the window's spend. Spend near the window's end drops
the carryover it causes after the window, `r^(T - t)` of each unit spent at `t` on a linear curve.
For a constant spend over 52 weeks at a retention of 0.8 this is 7.7 % of the return (ADR 0016,
the last bullet of its Consequences). mixdyn's one definition of a return,
`mixdyn.decisions.returns` (ADR 0016), counts that carry-out. So the prior was stated on a
different quantity from the posterior return reported for the same channel.

Meridian's ROI counts the same quantity as the model's parameter. Its documentation says the ROI
numerator is the incremental outcome over the selected time window. That outcome includes lagged
effects of media run before the window and excludes effects after the window of media run during
it (Google, *Meridian documentation: Incremental outcome, ROI, mROI and response curves*, and the
`InputData` reference, 2024–2025). The documentation adds that the gap is small over a year and
larger over shorter windows. The default centre, 0.2, is Meridian's prior for a revenue KPI. It
is stated on Meridian's quantity, which is the window's.

## Decision

- **`RoiPrior(mu, sigma, counts="window")`**. `counts` names what the return counts: `"window"`
  is the window's contribution over its spend, as before, and `"carried"` is
  `mixdyn.decisions.returns`' `roi`. The field is named `counts` because the option changes what
  the number counts, not its distribution or its units. The prior's form and its parameters read
  the same under either value.
- **Under `"carried"`**, the coefficient comes from the window spend's whole incremental return:

      beta = roi * window_spend / D,
      D = sum_t [ s(adstock(x))_t - s(adstock(x with the window's spend removed))_t ]

  The series is the panel's spend (padded before it as the model pads it), run on for `L - 1`
  periods with nothing spent, where `L` is the kernel's length. The sum runs from the window's
  first period to `L - 1` past its last; at every other period both terms are equal. This is
  `chc.response.roi`, which `returns` calls: the carry-out is counted, spend after the window stays
  in both terms, and carry-in from spend before the window is not credited to the window. On a
  linear curve `D` is the window's spend in the model's units, because a normalised kernel's
  weights sum to 1. The window's contribution is then `roi * window_spend * S / D`, with `S` the
  window's saturated sum. That share depends on the decay and the curve.
- **The level (ADR 0036) follows the identity.** The level is the intercept plus the channels'
  contribution averaged over the window:

      level = intercept + (1 / periods) sum_m beta_m S_m

  Under `"carried"` this sum depends on `(log_roi, logit_retention, log_half_saturation,
  log_slope)` and on nothing else, and the intercept never enters it. The map from `(those, level,
  rest)` to `(those, intercept, rest)` is still a shear, `intercept = level - f(those)`. Its
  Jacobian matrix is the identity with one row changed below the diagonal, so it is triangular
  with a unit diagonal and its determinant is 1. The density gains no term. (Maxima: the
  determinant of the Jacobian of `(a, r, k, n, c) -> (a, r, k, n, c + f(a, r, k, n))` is 1, for
  `f` a function of the first four coordinates.)
- **Per channel.** One model can hold carried and window returns side by side. `BayesModel.tails`
  holds each carried channel's two lag matrices. A model with no carried channel computes exactly
  what it did before. The default code path is unchanged: `_media_level` keeps its closed form
  and `_terms` its window sum.
- **The model over geos and the nested model refuse `"carried"`.** `BayesGeoSpec` and
  `NestedSpec` raise on a carried prior, naming the channels.
- **The default stays `"window"`**, and `mu = 0.2` stays the default under both values. A carried
  return is the window's return plus the carry-out, so under `"carried"` the same prior centre
  puts slightly less mass on large window contributions. A user who restates Meridian's prior on
  the carried quantity can raise `mu` by `-log(1 - carry-out share)`: 0.08 at the 7.7 % above.

## Consequences

- *Tests* (`tests/test_roi_carry_out.py`):
  - **The default is bit-identical.** The digests of the log density and gradient at Stan's 25
    points of each of the four worlds, and of a short seeded fit's draws and log likelihood in
    two worlds, were recorded before the change and match after it. Stan's oracle tests
    (`tests/test_bayes_mmm.py`) pass untouched.
  - **A fit's `roi` is the reported return.** In a carried fit over the panel's first 80 of 104
    weeks, the draw's `roi` equals `returns(channel_draws[s][c], spend, window).roi` to `1e-12`
    at each of its 200 draws, for each channel. The window's contribution falls short of it at
    every draw.
  - **The density is its formula.** With two channels carried and one not, over weeks 24–79, the
    density equals a numpy re-implementation of the formula to `1e-12`, on Hill and on `tanh`.
  - **The level keeps its meaning.** The level is the intercept plus the mean of the model's own
    channel terms, and `_with_level` inverts the map, one point or many.
  - **No carryover, no difference.** At a retention of 0 the two counts give the same
    coefficients, and at a logit decay of −40 the same density to `1e-12`.
  - **ADR 0016's figure.** At 100 a week for 52 weeks, a retention of 0.8 and a curve linear at
    the spend, the window's coefficient over the carried one gives a carry-out of 0.076914. That
    matches the exact rational to `1e-9` and rounds to 7.7 %.
  - **The refusals** of the model over geos and the nested model.
- *Symbolic checks* (Maxima 5.46, run outside the tree; 38 checks pass, and a control, an
  unnormalised kernel, fails as it should):
  - a normalised kernel of length `L ≤ 8` sums to 1;
  - for every spend vector, every window `[0, T)` with `T ≤ 5` and every `L` from `T` to 6, a
    linear channel's carried increment equals its spend;
  - at `T = L ≤ 8`, and at `T = 52, r = 4/5`, the window's share of a constant spend's return has
    the closed form `(T - r (1 - r^T) / (1 - r)) / (T (1 - r^T))`, so the carry-out is 0.0769139;
  - the shear's determinant is 1.

  The test file repeats the first three checks in exact `fractions` arithmetic at rational points.
- *Mutations*, each run on the tree with the change reverted after:
  - dropping the carry-out from the coefficient;
  - leaving the level on the window's identity;
  - reading the increment without removing the window's spend;
  - cutting the run-on at the window's end;
  - computing the default's level through the channel terms;
  - dropping each of the three refusals.

  Each fails at least one test.
- A carried prior costs a second pair of saturated series per channel, over `window + L - 1`
  rows, at each density evaluation.
- *What extending the other models needs.* The model over geos sets each geo's coefficient from
  the national return divided by the reach and scaled by its deviation factor (ADR 0013). A carried
  return there needs each geo's increment, `D_g`, under its own deviation, and a national return
  that is the population-weighted sum of the geos'. The coefficient then no longer factors through
  one national denominator, and `channel_draws`' identity in that model has to be stated afresh.
  The nested model (ADR 0014) reads its stages as `BayesModel`s. Stage one's return is in the mediator's
  units, and the carry-out into stage two runs through the mediator's own path, so "the window's
  spend's whole return" on sales needs both stages' kernels together. Each extension needs its own
  oracle, and neither is in scope here.
- The default's digests are recorded for two x86-64 CPU classes: XLA's code for the host's vector
  extensions moves the last bits.

## Alternatives considered

- **Restating the default as carried.** Rejected. Stan's recorded densities and posteriors
  (`tests/fixtures/oracles/`) are of the window identity and would have to be recorded again. The
  recorded runs that read the default (the scorecard's static arm, ADR 0036's divergence tables)
  would no longer describe the model. A prior centre taken from Meridian, which is stated on the
  window's quantity, would also move silently. An opt-in leaves every record true.
- **Keeping the prior on the window and correcting the reported return afterwards.** This is
  already the case: `returns` reads the channels, not the parameter. It does not let a user state a
  belief, such as an experiment's or a previous model's return, on the quantity mixdyn reports.
- **Counting only the carry-out, but crediting carry-in to the window** (the window's
  contribution plus its tail). Rejected. It is not `returns`' definition. When the window starts
  after the panel's first period, it credits the window with returns earned by spend before it.
- **A boolean, `carried=True`.** Rejected in favour of a named count. A third reading, such as a
  steady-state return, would fit the same field.
