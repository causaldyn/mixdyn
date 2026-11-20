# 0017. Log-normal noise about the mean

Status: accepted, 2026-10-03

## Context

ADR 0010's model reads the KPI's noise as normal or Student-t: additive, of one scale over the
window. A KPI that counts sales is above 0, and its noise grows with its level, a peak week's
spread in euros wider than a trough's. Practitioners reach for a log-normal likelihood there, and a
log-normal can be read three ways: about the mean, about the median, or through a log link, where
the KPI's log is the linear predictor and the channels multiply. Only the first keeps the model
ADR 0010 states.

## Decision

- **`LogNormal()`**, a third `BayesSpec.likelihood`:
  `y_t ~ LogNormal(log mu_t - sigma^2 / 2, sigma)`, so that `E[y_t] = mu_t`, the mean ADR 0010
  writes. Every identity the model rests on holds in the mean: a channel's contribution is still
  `beta_m s_m(adstock_m(spend_m))`, its return over the window is still the parameter, the
  decomposition adds up and the planner reads the same curves. Only the noise changes: its
  standard deviation `mu_t sqrt(e^(sigma^2) - 1)`, proportional to the level.
- **`sigma` is the log's scale**, under the same `HalfNormal(1)`, and `BayesFit.sigma` reports it
  as it is: a log's scale has no unit.
- **The support.** A point whose mean is not above 0 at some observed period has no density, as in
  Stan: the log density is `-inf` there and its gradient finite, so a proposal there is rejected
  as a divergent one is. `bayes_model` refuses a KPI at or below 0 at an observed period.
- **Starts are moved into the support.** A start whose least mean is not above 0 has its
  intercept raised until that mean is the least KPI observed, which raises every period's mean
  alike and leaves the other coordinates as drawn. Uniform starts and Pathfinder's paths start
  there.
- **The model over geos (ADR 0013) refuses it**: its noise per head is weighted by population,
  and a log-normal's weighting is another model.

## Consequences

- *The oracle*: a fourth world of the Stan oracle's, its KPI drawn with this
  noise, beside the normal, Student-t and `tanh` worlds, each world's points drawn from a stream of
  its own. At 25 points in the support the log density is Stan's
  `lognormal_lpdf(y | log(mu) - square(sigma) / 2, sigma)` up to a constant, to `6.6e-16` of the
  largest, and its gradient Stan's to `1.2e-15` of the largest entry; each of the 119
  observations' log-likelihoods at 25 of Stan's draws is Stan's to `3.2e-14`. Four chains of
  1000 draws, from uniform starts moved into the support, put each parameter's posterior mean
  and SD within 2.64 joint Monte Carlo errors of Stan's 40 000 draws and each observation's
  log-likelihood's within 2.91, where the posterior tests' bound is 5.91, with no divergent
  transition and the largest R-hat 1.0039.
- *The prior* the posterior reads is ADR 0010's truncated to the support. On the oracle's
  log-normal world 1 122 of 20 000 draws from it, 5.6 %, put every observed week's mean above 0.
  A simulation-based calibration draws from the truncated prior, and a prior predictive check
  shows it, not the stated one.
- How a plan reads the noises, one by its leave-one-out density or several by stacking weights, is
  measured on the bench before any is the default, as ADR 0011's families were. The pointwise
  log-likelihood is the KPI's density in the panel's units under every noise, so leave-one-out
  compares them on one scale.
- *Mutations*: each of 16 mutants fails a test that draws no posterior: the median read for the
  mean, the correction's sign or its square dropped, no support, the noise read as normal, starts
  not lifted or lifted under every noise, to a mean of 0 or the mean KPI or along another
  coordinate, Pathfinder from unlifted starts, a KPI at or below 0 accepted, `sigma` in the KPI's
  unit, and the model over geos accepting the noise. The test that a start already in the support
  is kept was added for one of them before they ran.

## Alternatives considered

- **The median, `LogNormal(log mu_t, sigma)`.** Rejected: its mean is `mu_t e^(sigma^2 / 2)`, so
  every contribution and return read off `mu` is low by `1 - e^(-sigma^2 / 2)`: 0.5 % at a sigma
  of 0.1, 11.8 % at 0.5.
- **A log link, `log y_t = ... + sum_m beta_m s_m(...)`.** Rejected: the channels multiply, so a
  contribution needs a counterfactual to be defined, the return is no longer the parameter and
  the planner's curves no longer add. It is another model, a multiplicative one read in
  elasticities, not another noise.
- **Stan's start, uniform and drawn again at most 100 times until the density is finite.**
  Rejected: on the oracle's log-normal world 599 of 200 000 uniform points, 0.30 %, are in the
  support, so 100 draws start a chain one time in four and four chains about one time in 220.
