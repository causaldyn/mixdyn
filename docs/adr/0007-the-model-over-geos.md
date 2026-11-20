# 0007. The model over geos

Status: accepted, 2026-10-02

## Context

A geo's effect read from its own series is noisy, and read from the nation's it hides how the geos
differ. CHC fits the hierarchy between the two (`chc.dlm.GeoDLM`): national coefficients, each
geo's static deviation from them, and the deviations' spread chosen by the exact marginal
likelihood, with draws from its posterior (`fit_geo_spread`, CHC's ADR 0047). The engine needs it
on a `MediaPanel`, with a default prior, the per-capita scaling ADR 0006 left to the model, and a
decomposition and effects for each geo.

## Decision

- **`GeoDlmSpec(effects, trend, season, form, variance_discount)`** has the one-geo `DlmSpec`'s
  fields. An effect's discount is its national coefficient's, a geo's deviation is static, and the
  trend and season are each geo's own. A signed effect is refused: a posterior truncated to a sign
  is not drawn over geos.
- **A geo's noise variance is in proportion to its population**, its share of the mean, and the
  same in every geo without one. That is the per-capita scaling: in a model linear in its columns,
  a fit per head is the fit in totals with those weights.
- **`vague_geo_prior`** sets each standard deviation to ten times its coordinate's scale, as
  `vague_prior` does, on each geo's own KPI scale. The vaguest spread is the national coefficient's
  prior variance. The variance's guess is each geo's least-squares residual variance over its
  weight, averaged, with one degree of freedom. A prior's covariance is stated at that guess. A
  guess at the KPI's whole variance would tighten every prior by the noise's share of it once the
  variance is learned, the vaguest spread too; in CHC's bench that put the hierarchy's own spread
  outside the range searched.
- **`fit_geo_dlm(panel, spec, prior=None, *, draws=64, seed=0, level=0.9)`** needs two geos or
  more. `level` is that of the spread's intervals.
- **The smoothed states are mixed over the spread's draws**, and the decomposition and the effects
  are read off the mixture. Each step's mean and covariance are the mixture's, by the law of total
  covariance, and so is the covariance between adjacent steps. The degrees of freedom are the
  draws' mean. Every draw has the same `n_T` at the last step, and with a variance discount of 1
  at every step; below 1 the earlier steps' follow each draw's own variance. The marginals are
  mixtures of Student t, so the result is not `exact`. By linearity, a geo's contribution over its
  column is its effect's mean.
- **`geo_effects(fit, level)`** gives each geo's coefficients at the last period, the national one
  plus its deviation. It reads them as a Student t with the last step's degrees of freedom and the
  mixture's variance, so the t's scale is the variance times `(n - 2) / n`.

## Consequences

- On the tests' hierarchy the maximum-likelihood spread sits at its least in 7 of 8
  channel-worlds. That hierarchy has 8 geos, 52 weeks, two channels whose effects spread by 0.4
  and 0.3, and a noise at which one geo's effect alone is about as uncertain as the spread; 4
  worlds were drawn.
  - At that spread, the decomposition would pool every geo's effect into the nation's.
  - The effects at the ML spread miss by an RMSE of 0.367, those mixed over the draws by 0.285, and
    a model per geo by 0.455.
  - The mixed intervals at 90 % cover 58 of 64 effects.
- Cost: a filter and a smoother a draw, on top of the spread's search. `geo_effects` runs neither.
- `tests/test_geo_dlm.py` holds:
  - the decomposition against each geo's smoothed forecast;
  - each part against its own truth, which a part read off another's coordinates misses by more
    than its truth's spread;
  - recovery against a model per geo, and coverage, on that hierarchy;
  - the prior: the variance's guess near the noise's at the mean population, every standard
    deviation ten times its scale, and the deviations' variances equal to the national ones;
  - a national coefficient that moves beside deviations that do not;
  - the effects' interval as the mixture of the coefficient's quadratic form over the filters;
  - the smoothed states as the moments of their mixture, cross covariances and degrees of freedom
    included, both through `GeoSpread.mixture`;
  - a contribution over its spend equal to the effect's mean;
  - those four with the variance fixed and with it discounted;
  - a channel bought nationally, its effect in each geo within twice its 90 % interval's
    half-width;
  - seed and level reaching the spread, a filled step with a control, population weights, and the
    refusals.
- Of 38 mutations tried, 35 fail a test. The three that pass were predicted to before the run:
  - the variance's guess averaged without dividing by each geo's weight, unbiased either way since
    the weights average 1;
  - the variance's prior at 100 degrees of freedom instead of 1, no worse when the guess is close;
  - the spend's square also taken over steps without a KPI, which moves a vague scale by their
    share.

  Reading the effects' degrees of freedom off the first step instead of the last passes with the
  variance fixed and fails with it discounted, and so does the first draw's degrees of freedom
  taken for the mixture's. The discounted fixture is there for those two.

## Alternatives considered

- **The decomposition at the ML spread**, the empirical-Bayes plug-in. It needs one smoother
  instead of one a draw, and it disagrees with the effects: at a spread at its least, the
  contributions say every geo is the nation and the effects do not.
- **The effects at the ML spread** read the spread as known. Above, they miss by 29 % more.
- **A model per geo** pools nothing, and **complete pooling** hides the differences.
- **The mixture's variance read as a t's scale**, as CHC's bench reads it. It differs from the
  matched t only in the draws' spread, by `2 / n`, under 1 % here, but the matched t is what the
  docstring states.
- **The variance per head in the panel**: ADR 0006.
