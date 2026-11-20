# 0018. Lift estimates over geos and in the nested model

Status: accepted, 2026-10-03; amended 2026-10-05 by ADR 0033: a lift on the KPI of a channel
of stage one is read, both stages drawn as one model under it, rather than refused.

## Context

ADR 0015 reads a lift test's summary, a lift over its readout and its standard error, as a term of
the one-geo Bayesian model's likelihood. A geo test runs somewhere: in a sub-market of a few geos,
in one, or over the whole market. ADR 0013's model over geos reads each geo per head, with an
effect of its own, `beta_gm = b_m exp(eta_m z_gm)`, so a test's lift there depends on which geos
it ran in, how many people they hold and how the spend was spread over them. ADR 0014's nested
model draws its two stages apart because their joint posterior factorises; a lift a stage's
parameters set alone keeps that, and a lift that reads both does not.

## Decision

- **`LiftEstimate.geos`**: the geos the test ran in, a `frozenset` of the panel's geos, or `None`,
  the default, for the whole market the model reads: every geo of the model over geos, the one
  geo of a model of one. `share` is the tested group's share of those geos' market. A set without
  a geo is refused.
- **Over geos** (`bayes_geo_model` and `fit_bayes_geo`, through `estimates`, each channel's): the
  group's people are `share` times the population of its geos `S`, the same share of each. Its
  spend is spread over them, the same per head in each geo, as a national channel's is (ADR 0013),
  and carried over and saturated as the channel's own. Its lift in totals is the sum over its geos
  of its people there times the geo's effect per head times the gap per head, so on the model's
  scales
  `lift / share ~ N(sum_{g in S} w_g beta_gm sum_readout [s_m(adstock_m(x)) - s_m(adstock_m(x_planned))], se / share)`,
  `x = spend / (share sum_{g in S} pop_g)` over the channel's spend unit, `w_g` the geo's
  population over the mean, the lift and its error over the KPI's unit times the mean population.
  A panel without populations has one head a geo. No parameter is added.
- **Each estimate is one observation** of the pointwise log-likelihood, after the cells, in the
  panel's units: on the model's scales less the log of the KPI's unit times the mean population.
- **The kernels are as long as the longest series the model reads**, the calendar or an
  estimate's, as in ADR 0015, and a fit's `length` is the model's.
- **An estimate in a geo the model does not read is refused**: by the model over geos, a geo the
  panel does not have; by a model of one geo, any geo but its own, which it reads as `None`.
- **The nested model** (`fit_nested`, through `mediator_estimates` and `estimates`): a channel's
  lift on the mediator, in the mediator's units, enters stage one; a lift on the KPI of a channel
  only stage two has enters stage two. A lift on the KPI of a channel stage one has, whether or
  not stage two has it too, is `gamma` times its lift on the mediator plus its direct lift: it
  reads both stages, so the posterior would no longer factorise, and `fit_nested` refuses it
  before it draws either stage, naming the lift on the mediator as what stage one reads.

## Consequences

- *The oracle*: the Stan program for the model over geos
  (`tests/fixtures/oracles/bayes_geo_stan.json`) now reads three lift estimates on each of its
  three worlds: a go-dark of `social` in half the market of two geos, a national heavy-up of `tv` and a heavy-up
  of `search` in one geo, each lift drawn about the model's at the world's parameters with the
  estimate's own error. The harness lays the estimates out itself, the group's spend over its
  share spread over its geos' people and each geo weighted by its population, rather than through
  mixdyn's code. The worlds' panels are the ones the fixture held. At 25 points on each world the
  log density differs from Stan's by one constant to `1.4e-15` of the largest and its gradient is
  Stan's to `3.1e-15` of its largest entry; at 25 of Stan's draws each of the 624 observations'
  log-likelihoods is Stan's to `5.3e-14`, the estimates' to `1.3e-14`. Four chains of 1000 draws,
  from uniform starts as before, put each parameter's posterior mean and SD within 2.87 joint
  Monte Carlo errors of Stan's 40 000 draws and each observation's log-likelihood's within 4.33,
  where the test's bound is 6.23, with every R-hat below 1.005 and every chain's E-BFMI above 0.7.
  The engine did not diverge; Stan's chains diverged 3, 0 and 12 times on the normal, Student-t
  and `tanh` worlds, where without the estimates they had diverged 1, 0 and 3 times.
- *What the tests hold besides*: a lift one standard error above the one the parameters in the
  panel's units give, computed by hand geo by geo, has a normal's log density there; an
  estimate's term leaves every other term and every coordinate but its channel's as it was; two
  estimates of one channel are two terms, each the one it is alone; a group half its share, with
  half its spend, lift and error, moves nothing; the geos in another order, the panel's units ten
  times as large or its populations ten times as large move only what ADR 0013 says they move; a
  panel without populations reads one head a geo; an estimate longer than the calendar runs the
  kernels over its length; and in the nested model each stage's likelihood holds its own lift at
  each draw, computed by hand from the draws.
- *Mutations*: each of 43 fails a test, 36 of them one that fits no posterior. Of the model over
  geos, 35: the estimates' terms dropped, the error read as 1 or as a variance, another channel's
  effects, kernel or curve, the gap's sign or its planned arm dropped, a mean for a sum, every geo
  tested, the geos unweighted or weighted over the population's sum, the named geos ignored, the
  group's people over every geo, without its share or one a geo, each share and unit dropped, the
  readout a period long or short, every row read as the first estimate, the kernels' length blind
  to the estimates or the calendar's, estimates of the wrong channel, at the wrong spend unit or in
  the order given, each refusal dropped, and the estimates' log-likelihoods in the KPI's unit
  alone; a fit's length the calendar's and `fit_bayes_geo` dropping the estimates fail the tests
  that read a short fit. Of the other 8, a model of one geo accepting another geo or refusing its
  own and `LiftEstimate` accepting no geo fail tests that fit nothing; the nested model's coupled
  lift accepted, its guard reading stage two, either stage's lifts dropped and the stages swapped
  fail the tests that fit it. Every row read as the first estimate survived the first run, as each
  Stan world has one estimate a channel; the test of two estimates of one channel was added for it.
- A group is read as the same share of each of its geos' people, at the same spend per head in
  each. A test in the whole of one geo and part of another, analysed as one lift, is read at its
  mean; one analysed geo by geo is one estimate a geo.
- An estimate in a few geos moves those geos' effects first and the national one through the
  pooling; a national estimate reads every geo's effect, weighted by its population as the
  return's identity weights it.
- A lift on the KPI of a channel that moves the mediator has no place in the nested model as it is.
  A test that also read the mediator, branded searches over its readout, enters stage one through
  that; otherwise the lift is left out, and the refusal says so.

## Alternatives considered

- **Every estimate read off the national effect `b_m`**, wherever the test ran. Rejected: `b_m` is
  the geometric mean of the geos' effects, and the point of the model over geos is that they
  differ; a test in one geo measures that geo's.
- **`share` as the group's share of the whole market** rather than of its geos'. Rejected: a test's
  design states its group against the geos it ran in, and a share of the nation would move with
  the population of geos the test never touched.
- **The spend spread over the group's geos by each geo's own spend on the channel** rather than
  by its people. Not built: a test's design sets the spend per head of its group, and spreading by
  people is how the model already reads a national channel. A group whose spend per head differed
  across its geos is one estimate per geo if its analysis reported them so, and is read at its
  mean otherwise.
- **For a lift on the KPI that reads both stages of the nested model**, four ways, none built:
  - *Both stages sampled as one model*, the coupling term in its likelihood: exact. It needs a
    model of both stages' coordinates, its own map to the panel's units and its own Stan program
    as an oracle; the sampler adapts to both stages' scales at once and to the correlation the
    term puts between `gamma` and stage one's effect, every gradient pays for both stages, and
    stage one's draws then read the KPI through the lift, so it no longer stands as the mediator's
    model alone. ADR 0014 rejected this form only because, without such a term, it buys nothing.
  - *A cut* (modular Bayes): stage one drawn as now; stage two drawn given each of stage one's
    draws, the term reading that draw. One stage-two chain per stage-one draw, thousands of fits,
    or an approximation of the cut; the lift never moves stage one, and the result is the
    posterior of no joint model, so its intervals carry no calibration of their own. Stan cannot
    express a cut, so the oracle would be built by hand.
  - *A moment-matched term in stage two*: stage one's posterior of its lift on the mediator
    summarised by its mean `mu_1` and SD `sigma_1`, and the KPI's lift read as
    `N(gamma mu_1 + direct lift, sqrt(se^2 + gamma^2 sigma_1^2))`. One cheap term, but it reads
    as normal a posterior a curve skews, and the lift moves stage one no more than under a cut.
  - *The factorised draws reweighted* by the term and resampled, the weights Pareto-smoothed as
    ADR 0011's leave-one-out smooths its own: no new sampler, and exact as the draws grow; but the
    weights degenerate, Pareto `k` above 0.7, as the lift grows informative against the stages'
    posterior, which is when it matters, and a resampled draw repeats.
