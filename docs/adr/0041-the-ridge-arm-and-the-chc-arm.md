# 0041. The ridge arm and the CHC arm

Status: accepted, 2026-10-11; amends ADR 0021's arms; amended 2026-10-12 by ADR 0043: on a geo
export both arms plan its market, one weekly spend a channel; amended 2026-10-12 by ADR 0044: the
CHC arm also plans the worst tenth of its readings against the equal split, a second variant;
amended 2026-10-14: a history too short for 4 folds after 26 weeks is searched by fewer; amended
2026-11-07: why the CHC arm's returns read several times the truth, and its `tanh` scales are not
the DLM arm's half-saturation points; the arm does not change

## Context

The scorecard's engine arms were the static Bayesian model (`bayes`) and the DLM over media
(`dlm`). Two more planners belong beside them:

- **The engine's ridge model** (ADR 0008). The library ran its search on Track M v2 in a script of
  its own, outside the harness. It never wrote the record that every arm writes, so its plans
  were not scored beside the other arms on the same exports.
- **CHC's own planner over curves fitted with CHC alone.** The engine is built on CHC. An arm that
  uses CHC plainly shows what the engine adds over the library under it.

Two facts shaped both arms:

- **The export reduces each lift test to one lift and its standard error.** Each row holds the
  test's mean weekly change over its readout and that change's standard error
  (`causaldyn_bench.budget_regret.lift_rows`). The ridge model reads a geo test as
  `chc.lift.LiftTest`: the gap between the test's groups, period by period. It reads the tests'
  noise off the readout periods before the groups' spend parts (ADR 0008). CHC's one fit of a
  curve, `chc.lift.fit_lift`, reads the same gaps. The export holds neither.
- **CHC fits no curve to a sales history.** `chc.lift.fit_lift` fits a channel to a test's gaps.
  `chc.dlm` fits a sales series on columns, and its filter's log-likelihood is the criterion it
  names for choosing a model. Nothing in CHC chooses a kernel or a curve from a history.

## Decision

### The ridge arm (`scorecard.arms.ridge`)

- **The library's search and its fit, unchanged.** `search_ridge` on the export's panel, over the
  space that the library's scored run on Track M v2 used. Each channel's geometric `theta` is in
  [0, 0.9], its Hill's `alpha` in [0.5, 3] and its `gamma` in [0.3, 1]. The penalty is in
  [1e-4, 1e2]. A promotion's coefficient is at or above 0, and every other control is free. The
  trend has 3 knots and the yearly season has 3 harmonics. The search scores 256 Sobol starts by
  the validation NRMSE of 4 folds of 13 weeks, and it polishes the best 8. A child of the export's
  seed scrambles the starts.
- **A short history is searched by fewer folds** (`Folds.within`, amended 2026-10-14). The first
  fold fits at least 26 weeks, and the earliest folds are dropped until it does. 26 weeks is the
  shortest prefix the 4 folds leave on a history of 78 weeks. So a history of 78 weeks or more is
  searched by the 4 folds, as before, and one of 52 weeks by the latest 2. 4 folds on 52 weeks
  cover the whole history: the first fold fits nothing, and `Folds.masks` refuses it. The record's
  settings name the folds the search used.
- **No second fit.** `RidgeSearch.fit` is `fit_ridge` of the chosen model over the whole history.
  The library composes the two that way, and its own tests hold the two fits equal. The arm reads
  that fit. A test holds the arm's fit to `fit_ridge` of its choice.
- **The plan** is `chc.allocation.allocate` on the fit's channels, against the export's box and
  budget, with the history carried in. The same planner makes the DLM arm's plug-in plan.
- **The claim** is the plan's gain over the status quo on the fit's channels. It is a point:
  the fit has no draws, so the record names no interval, and `n/a` says why.
- **The response** is the fit's channels as one draw, beside each channel's contribution to the
  fit in each week of the history. The kernels are unnormalised geometric kernels as long as the
  history, and the curves are Hills, as `RidgeFit.channels` gives them.
- **No forecast.** The library has no forecast of a ridge fit, and `n/a` says so.
- **No lift test.** The model has no way to read the export's rows. So the arm declares that it
  reads no test (`tests = False`). The harness then fits each world once for every rung, and a
  rung's plan with tests is its plan at k = 0. Every record says so: `n/a` names the lift tests
  and why, and the settings give `lift_rows_read: 0`.
- **Flags**: the search's score and evaluations, the fit's NRMSE, the chosen penalty and each
  channel's `theta`, `alpha` and `gamma`, whether each of these sits at an end of its interval,
  and whether each channel's coefficient sits on its bound at 0.

### The CHC arm (`scorecard.arms.chc_arm`)

- **The fit is `chc.dlm`'s filter of a static model.** Every discount is 1, so the filter is the
  conjugate regression over every week at once. The weekly sales are regressed on a level and a
  slope, a yearly season of 3 harmonics, each channel's curve at a coefficient of 1, and the
  controls. A channel's curve is a normalised geometric kernel of the export's length, then a
  `tanh`.
- **The prior is the engine's DLM default, written out in the arm.** Each prior standard
  deviation is ten times its coordinate's scale: the sales' standard deviation `s` for the level
  and the season, `s / T` for the slope, and `s / rms(x)` for a column. The variance is learned
  from one degree of freedom at `s²`. A column's prior so scales with the column. A column times
  `c` leaves the log-likelihood as it was, and a test holds that to `1e-9`. So the curve's scale
  moves the likelihood only through the fit.
- **The curves are chosen on a grid by the filter's log-likelihood.** Each channel's retention is
  0, 0.1, ..., 0.9, and its `tanh` scale is 0.25, 0.5, 1, 2 or 4 times the channel's mean positive
  spend: the DLM arm's range of half-saturation points.
- **The grid is searched from 32 starts, and the 4 most likely are polished.** This is the shape
  of the engine's ridge search: score a sample, then polish the best. The first start is the
  grid's middle for every channel, and a child of the export's seed draws the other 31. A polish
  moves one channel at a time to the most likely point of its grid while the others are held. It
  takes a move only where the likelihood rises, and it stops when a sweep moves nothing, or after
  10 sweeps. The choice is the most likely polished point.
- **Every channel's coefficient is held at or above 0**, since `allocate` refuses a channel that
  returns less as it is spent. The chosen model's coefficients are drawn from its posterior
  truncated to those signs (`chc.dlm.constrained_sample`): 4 chains of 1000 draws after 1000 of
  warm-up, from a second child of the export's seed. A channel's coefficient is the mean of its
  draws, not a clipped estimate.
- **The plan** is `allocate` on the curves at those means. **The claim** is the mean gain over
  the status quo, with its central 90 % interval, over 400 readings: the chosen curves at the
  coefficients of 400 draws, evenly spaced over the pooled draws.
- **The response** is the readings, beside each channel's part of each draw's mean as the filter
  reads it (`chc.dlm.decompose`).
- **No forecast.** `chc.dlm.forecast` reads the posterior without the signs, not the draws that
  the plan reads, and `n/a` says so.
- **No lift test**, as for the ridge arm: `fit_lift` needs gaps that the export does not hold.
  Every record says so in the same two fields.
- **A national export alone.** An export with a geo axis is refused with an error that names the
  geos and what is missing: the arm fits no channel for each geo, which
  `chc.allocation.allocate_geos` would plan on.
- **Flags**: each channel's retention and scale and whether each sits at an end of its grid; the
  chosen model's log-likelihood; the models the search filtered; the sweeps of the chosen polish
  and whether it ended by moving nothing; whether the draws did not mix by CHC's check (`fitted
  not mixed`, the DLM arm's flag); and their least bulk effective sample size.
- **A flat column** (amended 2026-11-07). Each channel's record also holds its chosen column's
  coefficient of variation, `<channel> column cv`, and `<channel> flat`, true where it is below
  `FLAT`, 0.05. `scorecard.flags` reads `flat` as an alarm beside `at an end`; the number is
  recorded and not read. The variation is read over the periods from the first whose carryover
  holds 0.99 of its kernel's weight (`FULL`): before it the column rises from the nothing the
  kernel assumes spent before the history. Over those periods a spend that never moves varies its
  column by at most 1 % of it, so such a channel is always flagged; a curve saturated where the
  spend is never 0 is flagged where the search chooses it.
  - *Why a column's variation.* The fit reads a coefficient from how its column moves; a return
    counts the column's whole height, down to no spend. A column whose standard deviation is under
    a twentieth of its mean leaves most of that height on the ridge where the coefficient trades
    against the level.
  - *Why 0.05.* It is the line the refresh cell's pilot drew, as this ADR's amendment of
    2026-11-07 on why the arm's returns read several times the truth records it
    (`evidence/bench/chc_arm_returns.py`): 41 of its 84 channel-fits chose a column below it, and
    all 41 read their return above 3 times the truth. Read over this flag's periods instead of the
    script's, after the kernel's first `kernel_length` weeks, the same 41 fall below it. The flag
    is conservative there: the next five, at 0.055 to 0.058, read 3.0 to 7.0 times the truth, and
    a line at the gap above them, 0.1, would be fitted to the same pilot. The line is fixed before
    any run reads the flag as a classifier.
  - *No reading changes.* The flag is computed from the chosen curves after the fit; a smoke record
    of Track M v2's drawn world 905, with and without the flag, differs in the new keys and the
    compute record alone. A record written before it holds no `flat` key, and the flags read it as
    before.
  - *Tests* (`tests/scorecard/test_ridge_and_chc.py`, `tests/scorecard/test_flags.py`): a channel
    whose spend never moves is flagged at a variation under a tenth of `FLAT`, its record read as
    an alarm; at a one-point grid a never-0 swinging spend is flat at the most saturated scale and
    not at the least, the spend that never moves flat at both; `flat` reads 1 and 0 and `column cv`
    is not read; records with and without the flag are read together, the flag over those that
    hold it.

### Both arms

- `scorecard.run.ARMS` holds them as `ridge` and `chc`, each with one variant of the same name. A
  run declares their limits as it declares every arm's (ADR 0039), and `declare` refuses either
  arm without them.
- `report.past` reads the CHC arm's chains by `fitted not mixed`, as it reads the DLM's. The
  ridge arm's flags name no chain.
- A smoke run reduces each search: 16 ridge starts with 2 polished; a CHC grid of 3 retentions
  and 3 scales, 8 starts with 1 polished, and 2 chains of 50 draws after 50 of warm-up.

## Consequences

- A comparison at a rung with tests reads both arms without the tests that the other arms read.
  The record says so, so the reading is not silent.
- Both arms are this engine's arms. So the claim (ADR 0040) reads them as the engine's own
  alternatives to the default procedure, not as rivals.
- The tests run one drawn world of Track M v2 end to end through `declare`, `export`, `one`,
  `score` and `report` at rungs 0 and 4. They also check: each record's fields; a plan in the box
  that spends the budget; the same record twice at a seed, from a child of the export's seed;
  each response held by the bench's mapping check to the fit's own parts; each plan as
  `allocate`'s on the fit's channels, and its claim recomputed; the refusals of a geo export and
  of a channel that never spends; and the prior's form. On a world made with known curves on the
  grid, the CHC search reaches a model at least as likely as the world's own curves. Held to 0,
  a channel that returns nothing draws no negative coefficient.
- A CHC curve can sit at an end of its grid. The flags say so, as the DLM arm's do.
- A geo family's worlds are refused by both arms. The ridge arm refuses them through the panel
  that every engine arm reads (`scorecard.arms.panel`).

## Alternatives considered

- **Shorter folds on a short history**, 4 blocks of 6 weeks on 52 weeks. Rejected: a block of 13
  weeks is the quarter that every plan is made for, so the search scores the model on the
  horizon it plans. Fewer blocks of a quarter keep that.
- **A first fold of 13 weeks**, 3 folds on 52 weeks. Rejected: a channel's carryover at `theta`
  0.9, the end of the searched interval, still holds 25 % of a week's spend 13 weeks later and
  6.5 % 26 weeks later, so a fold of 13 weeks fits a carryover it has hardly seen decay.

- **LiftTests built from the export's rows**, for the ridge model and for `fit_lift`. Rejected:
  a row holds one lift and its standard error. A test's gaps period by period, and the quiet
  periods that the ridge model reads the tests' noise off, would have to be invented.
- **Fixed curves**, the DLM arm's start (a retention of 0.5 and a `tanh` at the mean positive
  spend), with only the coefficients fitted. This is the cheapest fit, but it fits no curve, and
  the plan would read curves chosen by a rule rather than by the data.
- **The whole product grid**: the most likely of all 50 points of every channel's grid. Rejected
  for its cost: 50 to the power of the channels, 125 000 filters for 3 channels.
- **One polish from the grid's middle.** Rejected: the likelihood over the curves has more than
  one peak. On a world made with known curves, one polish from the middle stopped 35.4 nats below
  the world's own curves, and a test holds it more than 10 below.
- **Discounts below 1**, coefficients that walk. That is the engine's DLM arm, not the simplest
  fit.
- **`chc.irf.local_projection_irf`.** It fits a linear impulse response. It has no curve, and its
  kernel is in none of `chc.response`'s families.
- **`chc.research.mmm` and `prescribe`.** They plan a continuous plant whose carryover rates are
  known. A plan for a box and a budget over `chc.response` channels is `allocate`'s (CHC's ADR
  0034).
- **A negative coefficient clipped at 0**, or a choice restricted to models whose coefficients
  are all positive. Rejected: a clipped estimate moves no other coefficient. The restricted choice
  fails a world in which no grid point gives every channel a positive mean.

## Amended 2026-11-07: why the CHC arm's returns read several times the truth

- **The reading.** On the refresh cell's pilot (ADR 0079), the CHC arm read returns several times
  the truth. The worlds are Track M v2's drawn and reference worlds 900 and the drift family's
  static and walk worlds. Their true returns on the reading's window are 1.65 to 5.18 a euro.
  `evidence/bench/chc_arm_returns.py` separates the causes on each world's first cut. Its record
  is `evidence/bench/results/chc_arm_returns.json`, at mixdyn `b2c7cf0` and the bench `95a61f9`.
- **The arithmetic is right.** Give each channel the world's own kernel and curve, and give the
  model the world's baseline and promotional uplift as its controls. The arm's model then reads
  each return at 0.87 to 1.01 of the truth. So the arm maps a fitted coefficient to a return a
  euro correctly: the kernel's normalisation, the curve's scale in spend, the KPI's units and the
  national sum.
- **CHC's fit is right as the arm calls it.** The filter's posterior mean equals the conjugate
  regression's closed form to `3.5e-12`, relative. The means of the sign-held draws equal those
  of rejection sampling from the same truncated posterior, within 2.7 Monte Carlo standard errors
  on all 36 comparisons.
- **Cause 1: flat curves.** The grid's most saturated curves are almost constant where the spend
  is never 0. Such a column trades against the level along a ridge that the likelihood hardly
  sees. A return counts the curve down to no spend, so it is read along that ridge.
  - The pilot holds 84 channel-fits: 4 worlds, 7 cuts and 3 channels. 41 of them chose a column
    whose coefficient of variation over the history is below 0.05. All 41 read their return above
    3 times the truth, at a median of 10.3 times. The flags name 35 of the 41 at an end of the
    grid.
  - The sign hold adds to it: a wide posterior that lies mostly below 0 is read as a large
    positive return. On the reference world, television reads -41 unsigned and +14 held, against
    a truth of 1.65.
  - Confounding is not needed for it. With the world's baseline and uplift as the controls, the
    search still chose a flat column for meta on 2 of 4 worlds. Its unsigned returns were -24 and
    -25, against 2.9 and 2.5.
- **Cause 2: spend set by the sales.** Track M v2's spend follows the sales before media: the
  budget, the bidding rule, and spend raised ahead of promotions that the export observes at 70 %
  recall. A sentiment that no column holds moves the baseline. With the world's own curves and the
  exported controls, paid shopping reads 1.7 to 2.2 times its truth, and television reads below 0
  on all four worlds. The 43 channel-fits whose column varies read a median of 2.7 times the
  truth, and 19 of them read above 3 times.
- **The grid's unit.** The Decision calls the arm's `tanh` scales "the DLM arm's range of
  half-saturation points". The curves are not the same: `tanh(x / s)` is half its ceiling at
  `atanh(1/2) s`, about `0.55 s`. So the arm's half-saturation points are 0.14 to 2.2 times the
  mean positive spend, and its most saturated curve is flatter than the DLM arm's. With the grid's
  numbers read as half-saturation points, scales of 0.46 to 7.3, the search still chose a flat
  column on 3 of the 4 first cuts, and meta read 53 and 56 on the drawn and reference worlds. So
  the factor is not the cause.
- **Decision: the arm does not change.** It reads what its history and its model identify. The
  history does not identify the return of a channel whose curve is flat over it, and the flags
  name most such fits. The return of a channel that is always on needs a lift test or a prior on
  the return, and the arm reads neither by design. The engine's Bayesian arm reads both. A scored
  arm's records do not move.
