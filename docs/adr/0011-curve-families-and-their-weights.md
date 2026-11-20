# 0011. Two curve families, and the data's weights on them

Status: accepted, 2026-10-02

## Context

ADR 0010's model saturates every channel by a Hill, as Meridian and Robyn do. PyMC-Marketing's
default is its logistic saturation, `(1 - e^(-lam x)) / (1 + e^(-lam x)) = tanh(lam x / 2)`,
concave from no spend, and the worlds of the bench's Track M v2 are of that family. A model of one
family fitted to a world of the other is misspecified where a plan reads it hardest: past the spend
the history saw, where a doubled-spend test reads. Which family a panel is of is not known in
advance, and choosing it for a bench's worlds is fitting the bench.

## Decision

- **`BayesSpec.saturation`** is `HillCurve()`, the default and ADR 0010's, or `TanhCurve()`:
  `tanh(atanh(1/2) a / K)`, `K` the adstock at half the ceiling, as the Hill's half-saturation
  point is, so that `K ~ LogNormal(0, 1)` states the same belief in both families and the planner
  reads the same parameter. A `tanh` curve has no slope.
- **Each observation's log density at each draw** is part of the fit, `BayesFit.log_likelihood`:
  the KPI at each fitted period, then each test's gap at its readout periods, in the panel's units,
  so that two engines' fits of one panel are scored on one scale.
- **`mixdyn.inference.loo`**, the engine's own: PSIS-LOO (Vehtari, Gelman and Gabry 2017; Vehtari,
  Simpson, Gelman, Yao and Gabry 2024), the generalised Pareto fitted by Zhang and Stephens'
  profile empirical Bayes estimate with the shape shrunk toward 1/2 as by ten prior observations,
  the tail `ceil(min(S / 5, 3 sqrt(S / r_eff)))` draws, `r_eff` read off the likelihood over the
  split chains; and stacking (Yao, Vehtari, Simpson and Gelman 2018) by Cover's iteration for a
  log-optimal portfolio, stopped by its certificate.
- How a plan reads the two families (one family, the one with the better leave-one-out density,
  or both by their stacking weights) is measured on Track M v2 before any is the default, as ADR
  0010's planners were.

## Consequences

- The oracle gains a third world, `tanh` curves with normal noise; the Stan program takes the
  family as data and generates each observation's log-likelihood. On all three worlds the log
  density differs from Stan's by one constant to `1e-15` of its size and the gradient is Stan's to
  `1e-15` of its largest entry; at 25 of Stan's draws each of the 116 observations'
  log-likelihoods is Stan's to `7e-14`. Four chains of 1000 draws, seed 1, put every parameter's
  posterior mean and standard deviation, and every observation's log-likelihood's, within 4.5
  joint Monte Carlo standard errors of Stan's 40 000 draws. The largest standardised difference
  among the parameters' (52, 54 and 46 on the normal, Student-t and `tanh` worlds) is 2.84, and
  among the log-likelihoods' (232 on each) 3.71.
- PSIS is arviz-stats' (1.3.3, run in the bench's pinned PyMC-Marketing environment) to `1e-14`
  in shape and log predictive density on six cases, shapes from −0.14 to 1.33, tails of a fifth
  of the draws and of `3 sqrt(S / r_eff)`; on the normal mean model it is exact leave-one-out to
  Monte Carlo error, and it flags the observation the posterior leans on.
- The tail's length is rounded up; arviz-stats rounds it down. The oracle's cases sit where the
  length is a whole number, so no test separates the two conventions.
- ADR 0010's two planners, measured on Track M v2's 60 pilot worlds with the Hill: the posterior's
  expected return ties the plug-in on the 40 drawn worlds, +0.005 [−0.002, +0.012] regret per
  euro, and loses to it on the 20 reference worlds, +0.028 [+0.004, +0.052]. The bench keeps the
  expected return by the rule it stated before the pilot ran, which read the drawn worlds alone;
  `channels()` still hands the planner the medians.
- Mutations: each of 87 of the model fails a test that draws no posterior, ADR 0010's and twelve
  more: the `tanh` curve's half point in the model and the planner, the slope's absence from the
  coordinates, the prior and the draws, the family's dispatch in both, the log-likelihood's units,
  order and shape, and the tests' noise scale. 34 of the module's 35 fail one; the 35th rounds the
  tail's length down, arviz-stats' convention.

## Alternatives considered

- **`tanh`'s own scale, `lam`, as PyMC-Marketing parametrises it.** Rejected: a prior on `lam` is a
  prior on `1 / K` of another shape than the Hill's, so the two families would hold different
  beliefs about the same panel before seeing it.
- **WAIC.** Rejected: PSIS-LOO's shape diagnoses its own failure; WAIC's estimate fails without
  a sign (Vehtari, Gelman and Gabry 2017).
- **Leave-future-out cross-validation** (Bürkner, Gabry and Vehtari 2020), the honest measure of a
  forecast. Not now: each fold is a refit by NUTS. Leave-one-out measures the curve's fit to the
  weeks and the tests the model reads, which is what the family decides.
- **A third-party implementation** (ArviZ, R's `loo`): the engine's dependency policy (ADR 0001);
  arviz-stats is its oracle instead.
