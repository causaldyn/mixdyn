# 0010. The Bayesian engine: returns as parameters, geo tests in the likelihood

Status: accepted, 2026-10-02; amended 2026-10-06 by ADR 0036: the intercept's coordinate is the
level, so a point of the Stan program is the engine's once `_with_level` moves it there

## Context

Meridian and PyMC-Marketing fit the static media-mix model in Bayes' form: carryover and saturation
inside the model, priors on what a practitioner knows, and a posterior that hands every decision
its uncertainty. The engine needs that model from a `MediaPanel`, sampled by its own NUTS (ADR
0009), with an oracle for its density, and with the geo tests read the way ADR 0008's ridge reads
them: a test's gap as the channel's response to the spend each group saw, not as a summary of it.

## Decision

- **The model** (`mixdyn.models.bayes_mmm`), over one geo's window:
  `y_t = intercept + calendar_t . delta + controls_t . gamma + sum_m beta_m hill_m(adstock_m(x_m))_t + e_t`,
  with `e_t` normal or Student-t, its degrees of freedom sampled.
- **One set of priors for every panel**, on scales the model sets itself: the KPI over its mean on
  the window's observed periods, each channel's spend over its mean positive spend.
  - Each channel's spend carried over by a normalised geometric kernel as long as the longest
    series the model reads, so no carryover is cut inside the data, then saturated by a Hill,
    `chc.response`'s kernel and curve, so that the planner reads the model's own curves. The
    spend at each period the model reads and every period before it is laid out once, so the
    kernel's weights carry it over as one matrix product, not a convolution at every step.
  - **A channel's coefficient is not a parameter.** Its return on the window's spend is, as
    Meridian parametrises it: the coefficient is whatever makes the channel's contribution over the
    window its return times its spend, so the prior on the return holds whatever the carryover and
    the curve. `log roi ~ N(0.2, 0.9)` by default, Meridian's for a revenue KPI.
  - The decay `Beta(1, 3)`, PyMC-Marketing's; the half-saturation point `LogNormal(0, 1)` in units
    of the mean positive spend and the Hill's slope `LogNormal(0, 0.5)`; the intercept `N(1, 1)`;
    the trend's columns over the window's length, so a coefficient is a change over the window, and
    every calendar and standardised control coefficient `N(0, 1)`, a control held to its sign as
    its spec says; the noise scale `HalfNormal(1)`, Student-t's degrees of freedom `Gamma(2, 0.1)`
    (Juárez and Steel 2010).
- **Geo tests in the likelihood.** A test's gap between its groups, at the market's scale, is the
  channel's coefficient times the gap in its saturated spend, each group's spend divided by its
  share of the market and carried over and saturated as the channel's own, read on the test's
  readout periods, plus normal noise of one scale across all tests, `HalfNormal(1)`. A go-dark test
  is read through the carryover it cost, not reduced to a steady-state lift.
- **Sampled on Stan's transforms, in a Stan program's parameter order**: the logit of a decay, the
  logarithm of a scale, a control held to a sign as the logarithm of its size. A point of the
  sampling coordinates is a point of the Stan program written beside it, so the two densities can
  be compared point for point.
- **`fit_bayes`** samples four chains of 1000 draws after 1000 of warm-up by default, each started
  uniform in `[-2, 2]` on the sampling coordinates as Stan starts its chains, with the compiled
  transition. It returns the draws in the panel's units, the posterior mean of each term's
  contribution, each parameter's R-hat and effective sizes, the divergences and each chain's
  E-BFMI. `channels()` hands the planner each channel at its parameters' posterior medians.

## Consequences

- The oracle (`tests/fixtures/oracles/bayes_mmm_stan.json`): the model written a second time, as
  a Stan program, and CmdStan's readings of two worlds of 104 weeks, three channels, two
  controls, a trend, a season and three geo tests, one with normal noise and one with Student-t:
  Stan's log density and gradient at 25 points uniform in
  `[-2, 2]` on the sampling coordinates, and its posterior from four chains of 10 000 draws after
  1000 of warm-up (CmdStan 2.38). The model's log density differs from Stan's by one constant to
  `1e-15` of its size and its gradient is Stan's to `1e-15` of its largest entry. Four chains of
  1000 draws, seed 1 as the test was written, put every parameter's posterior mean and standard
  deviation within 4.5 joint Monte Carlo standard errors of Stan's on both worlds, each error
  from ten batch means a chain; Stan's chains decorrelate within three draws, so a batch of 100
  spans them. Of the 106 standardised differences, the largest was 2.8 at seed 1 and 1.9 at
  seed 2.
- Mutations: each of 75 of the model fails a test that draws no posterior: its priors and
  Jacobians, the return's identity, the carryover and the curve, the tests' shares, units,
  readout and sign, the units its draws are reported in, the planner's channels and every
  refusal. The intercept's shift by the controls' means was caught only by Stan's posterior until
  a control moved by a constant was tested.
- The planner reads a plug-in, each channel at its medians, not the posterior's expected return;
  a plan that averages over the posterior is a decision of its own, and is measured on Track M v2
  against this one before either is the default.

## Alternatives considered

- **A prior on the coefficient**, PyMC-Marketing's default. Rejected: as the carryover and the curve
  move, a fixed prior on the coefficient is a different prior on the return, which is what a
  practitioner knows and states.
- **A go-dark test as a steady-state lift**, PyMC-Marketing's lift measurement: the spend's change
  and the KPI's over the test. Rejected: it reads a test that went dark as if the carryover it cost
  had settled, and it drops a test whose KPI did not fall.
- **A kernel cut at a fixed number of lags** (Meridian's `max_lag`, PyMC-Marketing's `l_max`).
  Rejected: the cut is a choice no data informs, and with a kernel normalised over its lags it
  moves every weight.
