# 0043. A geo world planned by the engine's arms

Status: accepted, 2026-10-12; amends ADR 0021's families and ADR 0041's refusal of a geo export

## Context

The bench's geo family (family 8) draws markets of ten geos over 104 weeks. Each geo buys two
channels itself, search and social, and the market buys TV once for every geo, split by
population. The export carries a geo axis (`causaldyn_bench.scorecard.observe.Observation`):

- `population` holds each geo's share of the market, and `national` names TV.
- The plan's cells are each geo's own channels, geo by geo, then each national channel once, its
  spend summed over the geos: 21 cells at ten geos.
- Each lift row is one go-dark test in the geos it names, every other geo of the ring, read by
  difference in differences against the other geos (`causaldyn_bench.scorecard.ladder.geo_rung`).
- The bench scores a plan of every cell, or a plan of one weekly spend a channel. It allots a
  channel's plan to the cells once, for every arm: a per-geo channel's across the geos as the
  status quo splits it, and a national channel's to its one cell.

The scorecard read no geo export. The panel every engine arm reads refused it, so the static model,
the DLM and the ridge arm refused it, and the CHC arm refused it by name (ADR 0041). The engine has
every part a geo arm needs: a geo panel with a national channel (ADR 0006), a lift estimate in its
geos or in a share of a market (ADRs 0018 and 0037), the Bayesian model over geos (ADR 0013), and
a plan of every geo's cells with a national channel planned as one spend (ADR 0042).

Two readings were open. A national arm can fit the market's sums and plan one spend a channel.
Then it must read a test that ran in some of the geos. A geo arm can fit every geo and plan every
cell. Then it must plan TV as the buyer buys it.

## Decision

### The reader (`scorecard.arms.panel`)

- **A geo export is a geo panel.** Each geo of the axis is the panel's geo `geo-<g>`, under one
  root, `mixdyn.NATIONAL`, its population the export's share. Each geo's own channels are spent as
  the export spends them. Each national channel is recorded as `allot_national` records it: its
  spend summed over the geos, split by the shares.
- **The export's own split is held to the panel's check** of a national channel, each geo's spend
  its share of the period's to `1e-9` of it, or the export is refused. On the family's first ten
  pilot worlds of each environment, the export's TV spend in a geo lies at most `6.1e-16` of
  itself from its share of the national spend. The check found nothing to refuse.
- **A geo export with controls is refused.** Its controls are the market's, one series a week, and
  no rule reads them geo by geo. The family exports none.
- **A geo test is a `LiftEstimate` in its geos.** Its spend is the named geos' spend summed, with
  the dark weeks at 0; its planned spend the history's own; its readout the dark weeks and the
  cooldown; its lift `delta_y` times the dark weeks, and its standard error `sigma` times them; its
  geos the named ones; its share 1, as the group is the whole of its geos' market. The row's `x`
  must be the named geos' mean spend over the dark weeks, to `1e-12` of it, as for a national row.
  The model over geos spreads the group's spend over its geos' people at one spend per head (ADR
  0018). On TV that is each geo's own spend, which a test holds to `1e-12`.

### The national arms on a geo export: the market

- **An arm declares whether it plans every cell** (`cells`). The harness hands an arm that does
  not, `bayes`, `dlm`, `ridge` and `chc`, the export's market (`scorecard.arms.panel.market`), and
  the arm runs on it unchanged.
- **The market is the geos' sums**, by `mixdyn.aggregate` to the root: one weekly sales series and
  one weekly spend a channel. A channel's status quo is its cells' summed. Its box is the spends
  whose allotment keeps each of its cells in the cell's box, the intersection of the cells' boxes
  over the status quo's split. On the family's first ten pilot worlds of each environment each
  cell's box is half and twice its status quo, exactly, so the market's box is half and twice its
  channel's. A cell whose status quo is 0 is refused, as the allotment gives it nothing; so is a
  channel no one spend of which keeps each of its cells in its box.
- **Every record of a geo export says which it planned**, in `plan_level`: `market` or `cells`.
  The record's digest is the geo export's, so the bench scores the market's plan on the world.
- **A test of the market is the geo test in its geos' share of the market.** The estimate keeps
  its spend and lift, drops its geos, and takes as `share` the named geos' share of the
  population. The model of the market then reads the group's spend and lift over its share, as if
  the whole market had gone dark at the group's spend per head. On TV, which spends the same per
  head in every geo, that reading is exact but for the group's effect per head against the
  market's. On a per-geo channel it reads the group at its mean spend per head, as the model over
  geos does.
- **A national arm handed a geo export itself refuses it.** The CHC arm's own fit names its geos.
  The static model, the DLM and the ridge model refuse it as the library's national fits refuse a
  panel with no geo `national`. The harness never hands one a geo export.

### The model over geos as an arm (`scorecard.arms.bayes_geo`, `bayes-geo`)

- **The fit** is `fit_bayes_geo` on the geo panel, every geo test in its likelihood as an estimate
  in its geos. Its trend, season, noise, prior on each return and sampler are the static arm's,
  from a child of the export's seed. Where the model's ADRs name a default, the arm takes it: a
  Hill (ADR 0011); every channel's deviations over the geos non-centred, as ADR 0013 names no rule
  that centres a channel; and chains started uniformly, as ADR 0012 keeps them on the model over
  geos until a pilot on that model decides. The settings record the start and the channels
  centred, and the flags the divergences, under the static arm's names.
- **`bayes geo plug-in`** is `plan_geos` on each geo's channels at the posterior's medians
  (`BayesGeoFit.channels`), TV planned as one national spend in its one cell's box (ADR 0042).
- **`bayes geo cvar 0.1`** is the static arm's `cvar 0.1` over the posterior's readings, each
  reading every cell's channel at one draw, by `chc.allocation.cvar_allocate`.
- **A reading's TV cell is one channel of the national spend.** Geo `g` reads TV as
  `p_g b_g h(a(s_g X) / (p_g K))` with `s_g = p_g / P`, `P` the market's people and `a` the linear
  adstock. That is `p_g b_g h(a(X) / (P K))`, so the geos' sum is one channel with its curve's
  scale `P K` and its coefficient `sum_g p_g b_g`. Maxima holds the identity for the Hill and the
  `tanh`, and a test holds the one channel to the geos' sum to `1e-12` on a fit's readings.
- **Each plan is written in the export's cell layout**, TV's cell its national spend, and claims
  its gain over the status quo: the mean and the central 90 % interval over the readings.
- **No forecast and no mapped response.** The library draws no forecast of the model over geos. The
  bench maps a response of the market alone, and a per-geo channel's market is a sum of curves,
  one a geo. The record says why under `n/a` and `response.unmapped`.
- **A smoke run** samples 2 chains of 100 draws after 100 of warm-up, as the static arm's does,
  and reads 40 of them: `cvar_allocate` searches every cell of every reading at once.

### Registration

- `scorecard.run.ARMS` holds `bayes-geo`, and `scorecard.layout.FAMILIES` holds family 8. A run
  declares the arm's limits as it declares every arm's (ADR 0039).

## Consequences

- Family 8 runs end to end: `declare`, `export`, `one`, `score` and `report`. The tests run its
  first pilot world at rungs 0 and 4 with the geo arm and the CHC arm, and check each record's
  fields, the cell layout, a market's plan of one spend a channel that the bench allots inside
  the box, the same record twice at a seed, and the refusals.
- A national arm's plan on a geo world cannot move spend between geos. Its regret holds that
  restriction beside its model's error. The record's `plan_level` names it, so a comparison of a
  market's plan with a plan of the cells is not silent.
- A geo export without its tests names no geos, so the export a test-free arm reads at any rung is
  the export at rung 0, digest and all. Before, its rows carried no geo field, which the export's
  own check refuses where the export has a geo axis, so a test-free arm could not read a geo
  export.
- An arm that plans every cell must declare `cells = True`; the harness reads the attribute on
  every geo export.

## Alternatives considered

- **A test read on the market's sums**: the market's spend less the group's against the market's.
  Rejected. On a concave curve that reads the market's last euros, where the test took the
  group's first ones. It was measured on 240 TV tests of the family's first 30 pilot worlds of
  each environment, without noise, on the worlds' own curves, the group a share of 0.134 to 0.866
  of the market. The share reading equalled the true lift times the market's mean effect per head
  over the group's, to `8.9e-16` of it; the log of its ratio to the truth has mean `+0.0015` and
  standard deviation `0.193`. The sums reading came to 0.952 of the share reading on average,
  from 0.827 to 0.996; the log of its ratio to the truth has mean `-0.048` and standard deviation
  `0.196`.
- **A national arm refusing a geo test.** Rejected: the share reading is exact on TV but for the
  effects, and a test-reading arm on a geo world would lose every test.
- **The market as a flag on the export, or as a second argument to the arms.** Rejected: the
  market is a national export, so the arms read it unchanged. Its type carries the geo export,
  which the reader needs to place each test in its share.
- **The posterior's means for the plug-in.** `BayesGeoFit.channels` reads each parameter at its
  median, the library's one plug-in reading of the fit. A mean would need a reading the library
  does not make.
- **The static arm's `stacked`**: the expected return of a Hill's fit and a `tanh`'s, stacked.
  Rejected: here it would be two fits of the model over geos.
- **TV's cells planned one by one.** Rejected by ADR 0042: the buyer cannot place a split other
  than the population's.
