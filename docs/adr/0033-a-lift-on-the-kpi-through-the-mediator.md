# 0033. A lift on the KPI through the mediator: both stages drawn as one model

Status: accepted, 2026-10-05

## Context

ADR 0014 draws the nested model's stages apart: they share no parameter and the mediator is
observed, so their joint posterior is the product of the two. ADR 0018 lets a stage's likelihood
hold the lifts its parameters alone set, and refuses a lift on the KPI of a channel that moves the
mediator: its lift is `gamma` times its lift on the mediator, plus its direct lift where stage two
has the channel, so it reads both stages and the posterior no longer factorises. That is the
common case of a geo test of an upper-funnel channel, whose analysis reads sales rather than
branded searches. ADR 0018 laid out four ways to admit it: both stages sampled as one model, a
cut, a moment-matched term, and the factorised draws reweighted by Pareto-smoothed importance
sampling.

## Decision

- **`fit_nested` reads such a lift** through `estimates`, beside the lifts on the KPI of channels
  stage two alone has. A lift on the KPI of a channel of stage one, whether or not stage two has
  it too, is, at the market's scale,
  `lift / share ~ N(gamma l_1 + l_2, se / share)`:
  `l_1` the channel's coefficient in stage one times the gap in its curve there between the spend
  as the test ran and as planned, summed over the readout, in the mediator's units; `l_2` the same
  of stage two, in the KPI's, where stage two has the channel, and 0 where it does not; `gamma` the
  mediator's coefficient in stage two, per unit of the mediator.
- **Both stages are drawn as one model where there is such a lift** (`nested_model`,
  `NestedModel`): its log density is the stages' plus each lift's term, on stage one's coordinates
  and then stage two's, each in its model's order; NUTS draws it from Pathfinder's starts, from a
  third seed spawned from the caller's; each draw is split into the stages' coordinates, and each
  stage's fit is summarised from its own as `fit_bayes` summarises a fit, both holding the one
  sampler's divergences and E-BFMI. Without such a lift the stages are drawn apart as before, from
  the same two seeds, bit for bit.
- **Each stage's kernels are as long as the longest series its parameters read**, as ADR 0015's
  are: stage one's run over every such lift's spend, stage two's over those of the channels it
  has.
- **`NestedFit.mediated_log_likelihood`** is each draw's log density of each such lift in the
  KPI's units, `(chains, draws, lifts)`, channel by channel in stage one's order and each
  channel's in the order given; a fit drawn apart holds none. With each stage's `log_likelihood`
  it is every observation's, in the order the oracle records them: stage one's, stage two's, then
  these.
- **Refused**, before anything is drawn: such a channel given no lift, and such a lift in a geo
  other than the model's, as each stage refuses its own.

## Consequences

- *The oracle* (`evidence/stan_nested.py`;
  `tests/fixtures/oracles/nested_stan.json`): the model written a second time, as one Stan
  program, each stage's block the one-geo program's, on two worlds of 104 weeks where television
  moves branded searches and searches and search ads move sales: `mediated`, television in stage
  one alone, both curves Hill's, its lifts on sales a go-dark and a heavy-up whose spend is
  recorded from ten weeks before the panel, so that stage one's kernels run past the calendar;
  `both`, television in both stages, stage one's curve `tanh`, its lifts on sales a go-dark and a
  national heavy-up. Each world has a lift on the searches and one on sales of the search ads
  besides. At 25 points on each world the log density differs from Stan's by one constant to
  `3.4e-15` of the largest and its gradient is Stan's to `2.8e-15` of its largest entry; at 25 of
  Stan's draws each of the 212 observations' log-likelihoods is Stan's to `6.7e-14`. Four chains
  of 1000 draws from Pathfinder's starts put each parameter's posterior mean and SD within 3.23
  joint Monte Carlo errors of Stan's 40 000 draws and each observation's log-likelihood's within
  2.87, where the test's bound is 5.77, with every R-hat below 1.005 and every chain's E-BFMI
  above 0.89. Stan's chains diverged 84 and 32 times in their 40 000 draws on `mediated` and
  `both`, and the engine's 7 and 6 times in its 4 000. The test reads them by ADR 0025's
  binomial test, which passes up to 20 and 11 at Stan's counts, rather than holding the engine
  to none; Stan's own runs of the test's length are not recorded on these worlds, so that ADR's
  second leg, the most divergent of them, is not read.
- *By hand*: at points of both stages' coordinates the joint log density less the stages' is each
  lift's normal log density, computed in the panel's units with its own kernels and curves, plus
  the log of the KPI's unit, to `1e-13` of the density, on a world with television in stage one
  alone and on one with both channels in both stages in the other order in each; at each draw of a
  fit drawn as one, each lift's log-likelihood is its normal density by hand from the stages'
  draws, to `1e-12`, and each stage's own lift's is as drawn apart.
- *Reweighting, measured*: on the tests' world, the go-dark of television read on sales with a
  standard error of 30, the factorised draws' importance ratios fit a Pareto shape of -1.26, as a
  normal likelihood's bounded ratios do (`mixdyn.inference.psis`), at an effective sample of 304
  of 400. The lift moves no posterior mean by more than 0.01 of its SD, and the joint fit's means
  of `gamma`, television's returns and its carryover are the reweighted draws' within 0.8 joint
  Monte Carlo errors. At the fits' own length, four chains of 1000 draws, three factorised fits
  (seeds 5, 6 and 7) were reweighted for standard errors of 30 and 10 and set against joint fits
  at NUTS target acceptances of 0.8 and 0.95: shapes from -1.80 to -1.46 at effective samples of
  2520 to 3344 of 4000, and the joint means of `gamma`, television's return on the searches and on
  sales, its carryover and the search ads' return within 2.41 joint Monte Carlo errors of the
  reweighted draws' in all 60 comparisons, and within 0.09 of a posterior SD. Seed 5's fit, shared
  by its four comparisons, put `gamma` below in all four, by 1.1 to 2.4 errors; seeds 6 and 7 put
  it on both sides, within 1.5. The short runs' one disagreement, the search ads' return 2.9 to 4.4
  errors apart at every standard error, was their chains'. The joint fits diverged nowhere, the
  factorised ones once in their 24 000 draws a stage.
- *Mutations*, each applied alone to `nested.py` or `bayes_mmm.py`, with the tests that fit at
  most a small posterior and not the Stan posterior tests: 21, of the lift's term (6: `gamma`'s
  scale and unit, the direct lift, `gamma` itself, the error, the term in the density), the
  log-likelihood's units (2), the kernels' lengths (3), the coordinates' layout and the mediator's
  place (3), the lifts' order and the stage that reads them (2), the refusals (2), and the joint
  draw and what each stage's fit holds of it (3). All 21 failed a test.
- Under such a lift stage one's draws read the KPI through it: stage one is no longer the
  mediator's model alone, which is the posterior the lift implies. Its fit's diagnostics read the
  joint chains.
- Each gradient of the joint density costs the two stages' together, and one step size and one
  tree serve both, so the joint chains move at the pace the harder stage's geometry allows; a lift
  that pins `gamma l_1` adds a ridge between `gamma` and stage one's coefficient. No timing is
  claimed here.

## Alternatives considered

- **The factorised draws reweighted by PSIS, gated by Pareto `k` at most 0.7, and both stages as
  one model where `k` fails**, the first plan. Rejected on three counts. The stages drawn apart
  cost what the joint draw costs, the same two densities' gradients, so reweighting saves no
  sampling. The shape is the wrong gate: a normal likelihood's ratios are bounded, and their tail
  fits a negative shape while the effective sample falls, as ADR 0030 found and the measurement
  above shows. And a reweighted fit carries weights that `roi`, `channel_draws`, the planner and
  leave-one-out would each have to read, or resampled draws that repeat and form no chains for
  R-hat or leave-one-out's relative efficiency. The reweighting is kept as a cross-check, above.
- **A cut** and **a moment-matched term**, as ADR 0018 rejected them: neither lets the lift move
  stage one, and the cut's result is the posterior of no joint model.
- **Every pair of the stages' draws, not only those of one index**, as importance draws from the
  factorised posterior: `S²` pairs where the lift constrains `gamma l_1`. Not built, with the
  reweighting.
- **A higher target acceptance for the joint draw**, to hold the oracle's worlds to no divergence:
  rejected. Stan's own chains diverge on both at its default, so the engine would be held to a
  standard Stan does not meet; the test reads the engine's against Stan's (ADR 0025). On the
  tests' world the joint means at 0.95 are those at 0.8 within 2.53 joint Monte Carlo errors and
  0.09 of a posterior SD, and each target's are the reweighted draws' (above).
