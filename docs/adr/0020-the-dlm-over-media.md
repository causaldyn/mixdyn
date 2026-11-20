# 0020. The DLM over media

Status: accepted, 2026-10-04

## Context

The dynamic linear model (`fit_dlm`) multiplied each effect's random-walk coefficient by its raw
column. A channel's effect had no carryover and no curve, so the fit could not hand a planner what
`chc.allocation` plans on, a `chc.response.Channel`: ADR 0004 records that a ridge fit hands its
channels over and a dynamic fit does not yet. The static engines estimate a carryover and a curve
per channel; the dynamic model is the one whose coefficients follow an effect that drifts.

The model is conjugate because it is linear in its states given the columns: the filter, the
smoother, the likelihood of a discount spec and the projection over the likelihood ratio's set
(`choose_dlm`) all rest on that. A channel's retention, half-saturation point and curve family are
not states of such a model. Whatever chooses them has to fix the columns before the filter runs,
and carry the choice's uncertainty the way the discounts' is carried.

## Decision

- **A transform is a value.** `mixdyn.transforms.Transform(retention, half_saturation, shape,
  length)`: the Bayesian model's family (ADR 0010, ADR 0011), a normalised geometric kernel of
  `length` periods, nothing spent before the series, then `tanh(atanh(1/2) a / K)` (`TanhShape()`)
  or `a^n / (a^n + K^n)` (`HillShape(n)`), `K` the adstock at half the ceiling in the spend's
  currency. It holds numbers, not `chc.response` modules, so a spec that carries it compares and
  hashes; its kernel, curve and channel are built on demand from `chc.response`, in double
  precision only.
- **An effect may carry one.** `Effect(column, discount, transform=...)` on a channel: the column
  the coefficient multiplies is the channel's spend carried over and saturated from the panel's
  first period, `DlmFit.x` holds it, and the decomposition is unchanged, the coefficient times the
  column. A transform on a control, in the model over geos and in `forecast_dlm` is refused: a
  control carries no spend, the geo model reads its columns raw, and a forecast's future column is
  the plan's spend carried over from the panel's, which a fit does not hold. ADR 0027: a fit now
  holds that spend, and `forecast_dlm` takes a transformed effect's plan as its spend.
- **The fit hands its channels over.** `DlmFit.channels(at=-1)`: each transformed effect as
  `Channel(kernel, curve, m_at)`, its coefficient the filter's mean at `at`, the posterior given
  the periods up to it, so the channel run over the spend up to `at` returns there the coefficient
  times the column. With a signed effect the coefficient is the mean of the truncated posterior's
  draws, which read every period, so only the last period is offered.
- **Two stages.** `transforms_from_bayes(fit)` reads each channel's retention, half-saturation
  point and slope at their posterior medians, and the fit's kernel length, as `BayesFit.channels`
  does; `choose_dlm` then picks the discounts with the transforms held.
  `DlmChoice.coefficient_bounds(level)` is each coefficient's interval at the last period, the
  union of the kept fits' Student t intervals (projection), and is refused where the kept specs'
  transforms differ, since each coefficient then multiplies another column.
- **Readings.** `DlmChoice.readings(draws, seed)` returns a tuple of `draws` readings, each a tuple
  of `chc.response.Channel`, one per transformed effect in the order of `channels()`: what
  `chc.allocation.allocate` takes one at a time and `chc.allocation.minimax_allocate` takes whole.
  A plain tuple, not a type of this package's, since a planner over draws reads a sequence of
  channel sequences and nothing more. The draws are spread evenly over the kept specs, a flat
  mixture in which no spec the likelihood ratio keeps outranks another; a spec's coefficients are
  drawn from its filter's multivariate Student t at the last period, `n_T` degrees of freedom
  about `m_T` with scale `C_T`, or with a signed effect from its truncated posterior's draws
  without replacement, each spec from a stream of its own; the readings come shuffled from `seed`.
  Their quantiles are the mixture's, narrower than the union `coefficient_bounds` gives.
- **The search.** `search_media(panel, geo, spec, grid, discounts, refinements=1, level=0.95)`
  chooses each transformed channel's retention, half-saturation point and curve family on a
  `TransformGrid`, with the discounts, by the one-step log-likelihood, and returns a `DlmChoice`
  of the specs it keeps:
  - *A grid*, because the likelihood ratio's set is a set of filtered specs and projection is a
    union over it: a grid gives that set, a continuous optimiser's path does not, and the curve
    family is discrete. Half-saturation points are multiples of the channel's mean positive spend
    on the geo, the Bayesian model's unit for them.
  - *Blocks.* The discounts over the whole discount grid are one block, as `choose_dlm` searches
    them; each channel's retentions, half-saturation points and shapes are another, since a
    channel's three parameters reach the likelihood only through its own column. On the grid a
    channel's block is its whole product, so the coarse stage assumes nothing of the likelihood's
    shape within a channel. A pass moves the discounts' block, then each channel's, each to its
    best spec with the rest held; passes repeat until one moves nothing.
  - *Refinement.* `refinements` times, every interval of the grid is halved, the retentions by
    distance and the half-saturation points by ratio, and the passes repeat with each channel's
    block its point and the neighbours either side in retention and in half-saturation, at every
    shape. The fractions are dyadic, so a coarser point is the same number at every finer level
    and a spec is filtered once whichever stage reaches it. This is the one place the search
    assumes a shape: that the coarse grid resolves the likelihood's mode, so that neighbours that
    each raise the likelihood lead from the coarse best to the finer one.
  - *Ending.* A block keeps the current spec wherever it ties the block's best, so every move
    raises the likelihood, and a level's grid is finite: each stage ends, at a spec no block
    improves. That is a block-wise maximum, not by construction the product's; the oracles below
    check that it is the product's on one channel and on two.
  - *The kept set* is every spec the search filtered within `chi2_p(level) / 2` of the best, as
    `chc.dlm.confidence_set` keeps discount specs, each then fitted. `p` counts once a channel
    each axis the grid varies, and once each discount family the discount grid varies: three
    for a channel searched over several retentions, half-saturation points and shapes, so seven
    for two such channels and the effects' discounts. Refinement adds no parameter.
  - It starts at the grid point nearest each channel's transform, the retention by distance, the
    half-saturation point by ratio and the transform's shape where the grid has it, so a Bayesian
    fit's medians (`transforms_from_bayes`) are its natural start. Each move is logged at debug
    level with the log-likelihoods it goes between, each stage's end and the kept set at info.

## Consequences

- *The oracles*, each a test that fails when its implementation is reverted:
  - at discount 1 the filter's state at the last period, its scale matrix, variance estimate and
    degrees of freedom are the batch normal-gamma regression's under the same prior, the
    channels' columns computed by their definition in NumPy: the state's largest relative error
    `1.4e-12`, the variance estimate's `6.9e-14`, the scale matrix's largest `4.4e-13` in
    correlation units, the degrees of freedom equal at 104, against a tolerance of `1e-9`;
  - each channel run over the spend up to a period returns the filter's coefficient there times
    the column, at every week to `7.8e-16` of it, against `1e-12`;
  - on each channel alone over the grid halved once, 70 transforms at three discounts, and on two
    channels over nine transforms each at three discounts, the search ends at the spec that
    filtering all of them ranks first, its log-likelihood the same to `1e-12`; the kept specs are
    exactly those of the last pass's blocks inside the margin, at four parameters and at five.
- *Coverage: the pre-registered check failed.* On 200 worlds of 156 weeks, two channels whose
  coefficient is `beta e^x_t`, `x_t` a random walk of `sigma_c ~ logU[0.01, 0.04]` a week, with
  each channel's true transform given to the model and `choose_dlm` on effects discounts of 0.97,
  0.99 and 1.0, the 90% projected interval at the last week held `beta e^x_T` in 0.860
  [0.804, 0.905] of the worlds for television and 0.855 [0.798, 0.901] for search (Clopper-Pearson
  95%), against the 0.88 set before the run (`evidence/dlm_media_coverage.py`, its rule in its
  docstring; its record `evidence/results/dlm_media_coverage.json`). The likelihood's pick alone
  held 0.700 and 0.730, so projection recovers part of the loss and not enough. Coverage falls as
  the drift grows, 0.955, 0.879 and 0.746 across television's terciles of `sigma_c`, and the
  effects' best discount was the grid's lowest, 0.97, in 117 of the 200 worlds: the grid stops
  above the evolution the faster walks need. An exploration after the failure, on worlds of
  another stream and with no rule, put a grid reaching 0.90 at 0.895 and 0.920 where this grid held
  0.795 and 0.855 on the same worlds, its intervals 1.17 and 1.19 times as wide. The grid is not
  changed on that: a lower floor is a hypothesis for a check of its own, set before it runs.
- *One discount for every channel's effect.* The effects' family moves together, so two channels
  that drift at different rates share one evolution, which the coverage worlds draw and this
  model cannot follow. ADR 0026 gives each effect a discount of its own (`PerEffect`).
- *Saturation that the spend does not reach is not learned.* On the tests' world, where
  television's adstock stays below its half-saturation point and search's below one and a half
  times its own, the likelihood rises as the half-saturation point grows towards a straight line,
  and the best half-saturation point is the grid's highest for both channels; the true curves'
  log-likelihoods lie 1.4 and 1.0 below the straight line's, inside the margin, and the set is
  wide, 13 of the 58 specs television's search filtered and 30 of search's 38. A channel whose
  best sits at the grid's end has found a limit, not a point; ADR 0026 names each such end.
- *The search's kept set is a subset of the likelihood ratio's set over the whole product*: a spec
  it never filtered is not kept however near the best. A projection over it can be narrower than
  the full set's, and its coverage with searched transforms is not measured here.
- *The curve family is a discrete choice*: Wilks's theorem does not cover it, nor a retention of
  0 on its boundary, and counting the family as one degree of freedom a channel is a convention,
  which widens the set over counting it as none.
- *Readings under-cover the union.* The flat mixture's quantiles sit inside the union of the kept
  fits' intervals; CHC's ADR 0041 records the set's draws pooled covering 0.79 and 0.81 on its
  discount benchmark's pilot, and the union 0.938 and 0.944 on the scored run that followed. A
  planner that wants the union's hedge plans over each kept fit's channels
  (`DlmChoice.kept[k].channels()`).
- *Cost.* A pass filters the discount grid once and each channel's block once, on the coarse grid
  `|retentions| x |half-saturations| x |shapes|` a channel, not their product over channels; the
  filters within the margin are held until the end.
- `coefficient_bounds` refuses a search's set whose transforms differ; `DlmChoice.bounds` reads the
  contributions, which compare across transforms.
- What ADR 0004 records a dynamic fit does not yet do, it now does: a dynamic fit whose effects
  carry transforms hands its channels over, keyed by name as the ridge fit's are.
- *Mutations*: each of 32 mutants of the transform, the channels, the refusals, the two stages
  and the readings fails a test, and 32 of 33 of the search. The one that passes does not repeat
  a pass that moved only the discounts, and ends at the same spec, since that pass filtered each
  channel's block at the new discounts.

## Alternatives considered

- **The transforms as states**, a nonlinear filter or a particle filter over retention and
  half-saturation. Rejected: the model is no longer conjugate, so the filter's closed forms, the
  likelihood's exactness and the oracles above go, and a sampler would be needed where none is now.
- **One joint posterior over transforms and states by MCMC.** Rejected for the same reason; it is
  the Bayesian model's ground, not the dynamic model's.
- **Every spec of the product.** Rejected: ten retentions, eight half-saturation points and three
  shapes are 240 transforms a channel, `240^3` specs for three channels before the discounts.
- **A continuous optimiser** over the logit of the retention and the log of the half-saturation
  point, a family at a time. Rejected: the family is discrete, and the likelihood ratio's set needs
  a set of filtered specs to project over; an optimiser's path is not one.
- **One axis at a time within a channel.** Rejected: it stops wherever no single axis improves, a
  weaker end than no block improving, and a channel's coarse product is small enough to search
  whole.
- **Readings weighted by likelihood**, West and Harrison's class I mixture over the kept specs.
  Rejected: its weights favour the specs the selection favours, and on CHC's benchmark the class I
  mixture over the whole discount grid covered 0.830 and 0.826 where the union over the kept set
  covered 0.938 and 0.944 (CHC's ADR 0041).
- **One reading a kept spec, at its coefficients' mean**: the set's members, the hedge a minimax
  planner over the set would take. Not the readings: a planner over a posterior's draws needs
  draws, and the coefficients' own uncertainty within a spec would go. Each kept fit's
  `channels()` gives it.
- **The channel's coefficient from the smoother**, the posterior given every period, at a period
  before the last. Rejected: a plan made at a period reads what was known then; at the last period
  the two agree.
