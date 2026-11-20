# 0013. The Bayesian model over geos: effects pooled per head, returns over the nation

Status: accepted, 2026-10-02; amended 2026-10-04 by ADR 0024, on the Student-t world the
R-hat ceiling is 1.1, and by ADR 0025, a run passes on divergences up to the most of Stan's own
twelve runs

## Context

ADR 0010's Bayesian model reads one geo. A geo panel read one geo at a time pools nothing, and read
as the nation it hides how the geos differ; the geo DLM (ADR 0007) pools them, but linearly, with no
carryover or curve inside the model. Meridian fits the geos jointly, each channel's effect a geo's
own draw about a national one, its carryover and curve shared. The engine needs that model from a
`MediaPanel`, with the per-capita scaling ADR 0006 left to the model, priors a practitioner can
state for the nation, and an oracle for its density.

## Decision

- **The model** (`mixdyn.models.bayes_geo`), per head of each geo over the observed cells of the
  window:
  `y_gt = a_g (1 + calendar_t . delta) + controls_gt . gamma + sum_m beta_gm s_m(adstock_m(x_gm))_t + e_gt`.
  A geo's KPI and spend are divided by its population; `e_gt` is normal or Student-t of scale
  `sigma / sqrt(w_g)`, `w_g` the geo's population over the mean, so that a geo's noise variance in
  totals is in proportion to its population, ADR 0007's weighting. A panel without populations has
  one head a geo.
- **The scales are ADR 0010's, pooled over the geos**: the KPI over its mean per head on the
  observed cells, each channel's spend over its mean positive spend per head over the calendar and
  the geos. One set of priors serves every panel, and a national channel, split by population, has
  the same spend per head in every geo.
- **Carryover and curve are each channel's, shared by the geos**: ADR 0010's normalised geometric
  kernel as long as the calendar, then a Hill or `tanh` (ADR 0011), with ADR 0010's priors.
- **A channel's effect is pooled over the geos**: `beta_gm = b_m exp(eta_m z_gm)`,
  `eta_m ~ HalfNormal(spread)`, 1 by default, Meridian's prior on its log-normal geo effects. `b_m`
  is not sampled: it is set by the channel's return over the window and every geo, `roi_m`, on
  which the prior is stated, `log roi_m ~ N(0.2, 0.9)` by default. The contribution over the window
  and the geos is `roi_m` times the spend over them, whatever the carryover, the curve and the
  geos' factors.
- **A channel's `z` sum to 0 over the geos**: `z_m = H u_m`, `H` Helmert's orthonormal contrasts,
  `u_m ~ N(0, I)` over `G - 1` coordinates. Under the return's identity a common shift of a
  channel's `z` moves only `b_m`, which the likelihood does not see; with `z_gm ~ N(0, 1)` that
  direction is held by the prior alone. Integrating the shift out of `z_gm ~ N(0, 1)` gives exactly
  the zero-sum normal, so every quantity the data see has the same posterior, and `b_m` is the
  geometric mean of the channel's geo effects.
- **A channel's deviations are sampled non-centred unless the spec centres it**
  (`BayesGeoSpec.centred`): non-centred, the coordinates are `u_m`; centred, they are `eta_m u_m`,
  `N(0, eta_m^2 I)`. Both are the same model: the centred density is the non-centred one less
  `(G - 1) log eta_m`, the change of coordinates' Jacobian, and every parameter is the same
  function of the point. Non-centred suits a channel whose effect in a geo the data say little
  about, the usual case; centred suits one whose geo effects the data pin, where the non-centred
  coordinates narrow as `eta_m` grows. The default is non-centred, and the analyst names the
  channels to centre.
- **A geo's base per head multiplies the shared calendar**: `a_g ~ LogNormal(0, 1)`, independent
  over geos as Meridian's geo intercepts are, and the trend's and season's coefficients
  `delta ~ N(0, 1)` are shares of the base, so every geo moves with the season in proportion to
  its base. A base is positive.
- **Controls are shared**, standardised over the window's cells that were not filled, each held to
  its sign as its spec says; a filled cell, which has no KPI, reads each control at its mean.
- **`fit_bayes_geo(panel, BayesGeoSpec(...), seed=...)`** fits every geo of the panel, two or
  more, with ADR 0009's sampler and ADR 0012's starts. It returns the draws in the panel's units,
  per head where a quantity is (a geo's coefficients, its base, the noise's scale at the mean
  population); each fitted cell's log-likelihood in totals, for ADR 0011's leave-one-out; and the
  posterior mean of each term's contribution in each geo, in totals, summed a batch of draws at a
  time. `channels(geo)` hands the planner a geo's channels in totals at the posterior medians.
- **Sampled on Stan's transforms, in a Stan program's parameter order**, as ADR 0010's: each
  channel's log return, logit decay, log half-saturation point, log slope and log spread, each
  channel's `G - 1` deviations channel by channel, on its spread's scale where it is centred,
  each geo's log base, the calendar's and controls' coefficients, the log noise scale and the log
  degrees of freedom.

## Consequences

- The oracle (`tests/fixtures/oracles/bayes_geo_stan.json`): the model written a second time, as
  a Stan program, and CmdStan's readings of three worlds of six geos and 104 weeks, populations
  log-normal, three channels, one bought nationally and split by population, two controls, a trend,
  a season, three cells without a KPI, and the KPI drawn from the model at known parameters: Hill
  curves with normal noise, with Student-t noise, and `tanh` curves.

- *The geometry, measured on the oracle's normal world* (one CmdStan chain of 1000 draws, its
  unconstrained draws' correlations): with `z_gm ~ N(0, 1)` a channel's `z` moved together,
  correlations up to 0.97, the correlation matrix's condition number 750, trees 7.9 deep on average;
  with the zero-sum `z`, 441 and 7.5. What remains is the non-centred form's own: where a geo's
  data pin `eta_m u_m`, the spread and the deviations trade off along `u_m ~ 1 / eta_m`, a
  correlation of 0.98 for the channel whose geo effects the data read best.
- *The divergences, and what centring buys* (CmdStan 2.38, `adapt_delta` 0.8; divergent
  transitions in four chains of 10 000 draws, in four of 2 500 with every channel centred;
  leapfrog steps a draw in brackets):

  | world | non-centred | `search` centred | every channel centred |
  |---|---|---|---|
  | normal | 202 (251) | 1 (70) | 0 (64) |
  | Student-t | 0 (128) | | 17 (64) |
  | `tanh` | 8 (128) | 3 (68) | 76 (52) |

  On the normal world every divergence lay where `search`'s spread ran large, the inverted funnel
  of the trade-off above, and the engine's own sampler diverged at Stan's rate, 18 times in 4 000
  draws. Centring `search` leaves 1 of the 202 and cuts the work a draw by a factor of 3.6;
  centring every channel moves the divergences to the channels the data pin least, where the
  spread runs small. The normal world's two posteriors, non-centred and `search` centred, agree
  within 2.5 joint Monte Carlo errors on every mean and SD: the divergences cost work, not a bias
  the oracle could see. The oracle records the normal and `tanh` worlds with `search` centred and
  the Student-t world non-centred, so that it checks both forms.
- *The engine against the oracle:* four chains of 1000 draws hold every posterior mean and SD
  within 4.5 joint Monte Carlo errors of Stan's 40 000 draws on each world, every R-hat below 1.01
  and every chain's E-BFMI above 0.3. Since Stan's own chains diverge on two worlds, the test asks
  of the engine's divergences only that they be no more frequent than Stan's: its share of the two
  samplers' divergences not above its share of their draws, by a one-sided binomial test at 0.001.
  On `tanh` the engine diverged once in 4 000, where Stan did three times in 40 000.
- Mutations: 91 of 92 fail a test, the comparison with Stan's posterior left out; one of them only
  once the planner's test held the kernel's length to the calendar's. The 92nd reads a negative
  lag's weight where the model then sets it to 0, the same model.

## Alternatives considered

- **An additive calendar shared by the geos**, Meridian's time effects. Per head it gives every geo
  the same swing whatever its base, so a geo with twice the base per head has half the seasonality
  in proportion. A shared calendar times the base keeps the proportion, the usual shape of a sales
  season, and is what `chc.geo_world` draws.
- **Each geo its own calendar**, pooled. More coordinates than the panel's geos usually hold, for
  a pooling the shared proportion already gives.
- **A prior on each geo's return.** The return a practitioner states is the nation's; a geo's
  follows from its share of the channel's reach, and its factor is what the data move.
- **The national effect sampled, `log b_m ~ N(.)`**, with the return a derived quantity. As ADR
  0010 found for one geo, a fixed prior on an effect is a different prior on the return whenever
  the carryover, the curve or the geos' factors move.
- **Every channel centred**, `log beta_gm ~ N(log b_m, eta_m)`. Where a geo's data say little
  about its effect, the centred form has the funnel between `eta_m` and the effects that NUTS
  cannot cross, which is the usual case for a channel's effect in one geo: the table's last column.
- **A higher target acceptance, every channel non-centred.** On the normal world the engine's
  sampler at 0.9 and 0.95 still diverged 3 times in 4 000 draws each, its trees 300 and 346
  leapfrog steps a draw: a smaller step reaches further into the funnel's narrow end, not to its
  end. With `search` centred, at 0.8, it did not diverge in 4 000 draws, at 70 steps a draw.
- **A rule that centres a channel by itself**, from how much a pilot's posterior narrows its geo
  effects against `eta_m`. Not built until a bench measures it against the analyst's choice.
- **`z_gm ~ N(0, 1)` with the shift left in**, as the return's identity first gave it: the same
  posterior for everything the data see, and a direction the sampler crosses on the prior alone.
- **Carryover and curve per geo, pooled.** Not built yet: the curve's half-saturation point and the
  decay are weakly identified in one geo, and the effect's factor carries what differs in scale.
