# 0078. The decision, reader, switchback and geo DLM arms of the scorecard

Status: accepted, 2026-11-07; adds the arms `decisions`, `read-tbr`, `read-sc`, `read-sdid`,
`switchback` and `geo-dlm` to ADR 0021's harness, beside ADR 0077's `voi`; no existing arm, record
or output changes; amended 2026-11-07: the readers and the switchback read the bench's version-4
export, the adapters are removed, and the switchback's reading and `flighting`'s schedule are
scored on the truth

## Context

The engine ships decision tools and readers no arm scored: the flighted planner (ADR 0052), the
scenarios read as differences with their carry-out (ADR 0051), the plan table (ADR 0050), the geo
readers (ADR 0047, SDID by ADR 0048), the switchback reading (ADR 0056) and the model over geos as
a DLM (ADR 0007). Each is a function of a posterior or of a test's panel. The scorecard scores a
plan, one weekly spend a cell, by its regret on the world's truth (`causaldyn_bench.scorecard.
scoring`), so each needs an arm that turns its output into a plan, or into a reading a plan is
made from.

Three facts of the bench shape them. It scores no schedule: a plan is one spend a week a cell. A
geo export keeps each test's reading by its difference in differences, not the panel the test ran
in, since each channel's tests run in a universe of their own (`causaldyn_bench.scorecard.ladder.
geo_rung`). And no family gives its worlds a switchback test, nor does a test of the bench draw
one.

## Decision

- **`decisions`** (`scorecard.arms.decisions`), on the static arm's fit (`scorecard.arms.bayes`),
  at each rung, on family 0 (any national export, or a geo export's market):
  - `flighting`: `plan_flighting` on the posterior of the family of the larger stacking weight
    (the Hill curve's on a tie), at its weighted median. CHOICE: the planner reads one curve's
    median or one draw, and the stacked posterior holds both curves, which have no median
    together; the leading family's median is the planner's own reading of the posterior the
    stacking leans on, where one draw would be one sample of it. CHOICE: the record's plan is its
    best rate a week, which the bench scores; the schedule, its recommendation, the family and its
    edge over the rate over the family's draws are kept beside it, and a record whose schedule is
    recommended says its regret is the rate's.
  - `scenarios`: `compare_scenarios` over the status quo, the bench's equal split, the static
    arm's `stacked` and `cvar 0.1`, each family's fit at its share of the static arm's readings,
    the differences pooled; the plan is the scenario of the largest mean difference, the claim
    its mean and central 90 % interval. An export that gives controls but not the quarter's
    fails it.
  - `plan table`: `plan_table` of `stacked` against the status quo; the plan is `stacked`'s, the
    claim the table's total change in the KPI, and the record keeps the rows.
- **`read-tbr`, `read-sc`, `read-sdid`** (`scorecard.arms.readers`), one variant each, `read tbr`,
  `read sc`, `read sdid`, on family 8 at each rung: each geo test read again in its universe by
  `read_tbr`, `read_synthetic_control` or `read_sdid`, its tested geos treated and every other geo
  a control, its four dark weeks its change and its sixteen weeks of readout, and the reading,
  per dark week, in the row's place; the static arm's `stacked` plan of the market on those rows.
  SDID's error is its jackknife's: the ladder tests half the geos, so no placebo is left. A test
  the reader refuses keeps the bench's reading, and the record counts them: the time-based
  regression's placebo windows need 30 weeks before a test, which the tests from week 20 lack.
- **`switchback`** (`scorecard.arms.switchback`), one variant, on family 8 at each rung: a
  switchback test of TV, the national channel, over the history's last 12 blocks of 8 weeks, 6
  on, the first 4 weeks after a switch left out, read by randomization on the KPI summed over the
  geos; its estimate, of the whole market, read beside the rung's tests by the static arm's fit
  of the market, and that fit's `stacked` plan. CHOICE: no family or fixture of the bench has a
  switchback, so the arm draws its own on family 8's worlds.
- **The universes are drawn again by an adapter** (`scorecard.arms.universes`), until the bench
  exports them: the world from family 8's own `world`, each channel's dark weeks and its noise
  from the ladder's stream, as `geo_rung` draws them; the switchback's from a further index of the
  family's stream of tests that the ladder does not draw. A test holds the adapter to the export:
  the ladder's reading of each of the 12 tests of family 8's first pilot world at rung 4, read
  again on the universe drawn here, is the row's to the last bit. CHOICE: an arm reads one export
  and nothing else; the adapter reads the world it was drawn from, and hands on only what a test's
  analyst observes, the universe's KPI and spend.
- **`geo-dlm`** (`scorecard.arms.geo_dlm`), one variant, `geo dlm`, on family 8, fitted once for
  every rung (the model reads no test): `fit_geo_dlm`, additive, each channel an effect on its
  raw spend, the discounts the dynamic arm's starting spec's (`scorecard.arms.dlm`), 0.99 for
  each effect's national coefficient and for the level and 1 for the season of 3 harmonics; 64
  draws of the geos' spread (16 in a smoke run) from a child of the export's seed. Each cell's
  coefficient is the last week's (`geo_effects`' means), a national channel's its geos' by
  population share. Its plan is the exact optimum on those lines (`geo_dlm.fill`): every cell at
  its least, then the rest of the week's budget to the cells of the largest coefficient first,
  each up to its most; a test holds it to HiGHS' solution of the linear program. It claims that
  plan's gain over the status quo with no interval. CHOICE: the model over geos refuses a
  transform or a sign, so its channels are linear with no carryover, and a coefficient may be
  below 0; the arm plans what the shipped model says. `chc.allocation.allocate` refuses a
  coefficient below 0, and the bench scores a plan at the whole budget, so the fill is the
  arm's own.
- **The static arm's fit is reused** through `scorecard.arms.bayes.family_fits` and `fitted`.
  `family_fits` takes the estimates to read in place of the export's rows, which the switchback
  arm hands it; given none, it reads the rows as before.

## Consequences

- The arms' records name their variants: `flighting`, `scenarios`, `plan table`, `read tbr`,
  `read sc`, `read sdid`, `switchback`, `geo dlm`. Each runs one smoke world of its family end to
  end under `scorecard/run.py` and is scored and reported.
- A reading arm's regret less the static arm's on the same market at the same rung is what
  reading the rung's tests by that reader rather than the bench's difference in differences moves
  the plan by; the switchback arm's less the static arm's is what the switchback adds.
- `flighting`'s regret is the rate's. A schedule's regret needs the bench to score a schedule.
- `geo dlm`'s regret is that of a linear model's corner plan: it measures the shipped model over
  geos as a planner, not a planner the engine recommends.
- Bench changes that would retire the adapters and the substitutes: each geo rung's universes in
  the export; a family whose worlds carry a switchback test; a schedule scored as a schedule.

## Alternatives considered

- **Flighting's schedule scored by its mean rate.** Rejected: that is a third plan, neither the
  schedule nor the rate the planner chose.
- **A reader's refused test dropped.** Rejected: the rung's information would then differ from
  the static arm's, and the comparison would read the count of tests as well as the reader.
- **SDID by placebo.** Refused by the reader: the ladder's controls are no more than its treated
  geos.
- **The switchback in place of TV's geo tests.** Rejected: the rung's tests are the rung's; the
  switchback is read beside them, so its arm at rung 0 reads it alone.
- **A gain interval for `geo dlm` from the coefficients' marginals.** Rejected: the gain is a sum
  over cells whose coefficients are correlated, which the marginals do not hold.

## Amended 2026-11-07: the bench's universes, its switchback test and its score of a schedule

The bench now exports what the adapters and the substitutes stood in for (`causaldyn_bench.
scorecard.observe`, version 4): each geo test's universe, and a switchback test of the national
channel on every world of the geo families 8, 9 and 10. It also scores a schedule
(`causaldyn_bench.scorecard.schedule.score_schedule`).

- **The readers read the export's universes.** `scorecard.arms.readers.reread` reads each test's
  KPI in every geo from `Observation.universes`, the channel's universe at the rung, `sales[i].T`
  on the panel's axes. These are the adapter's draws: a test held the export's arrays to the
  adapter's, and each reader's rows on the one to its rows on the other, to the last bit, on family
  8's first pilot world at rung 4, before the adapter was removed. The readers now read families 9
  and 10 as well. A test the reader cannot read keeps the bench's reading, counted, as the
  time-based regression's early tests already did: the synthetic control needs 3 control geos and
  SDID's jackknife 2 treated and 2 control geos, which the tests in a world of 3 geos (family 10's
  `geos-3`) lack; family 8's tests have 5 of each, so its readings do not move. `read_sdid`'s
  jackknife now refuses a single control geo with a `ValueError`; before, it divided by zero. An
  export of an earlier version holds no universes, and the arm's fit refuses it.
- **The switchback arm reads the bench's test.** Its design is the export's
  (`Observation.switchback`): weeks 9 to 104 in 12 blocks of 8, 6 on in the bench's order, the
  universe the bench drew. The arm builds `SwitchbackDesign` from the given blocks, with a washout
  of `block // 2`, and no longer draws its own. Its seed, the same child of the export's seed, now
  only names the stream a randomization reading would draw arrangements from, past 100 000; 924
  are enumerated. The record's settings add the reading's interval, `lower` and `upper`.
- **The switchback's reading is scored on the truth**, beside the bench's score of the plan
  (`scorecard.arms.switchback.scored`, under `switchback score` in the score's record): its lift a
  week on, the reading's lift over the 48 weeks on, against `Truth.switchback_steady`, the bench's
  lift a week on as a reader of that washout reads it; the relative error; whether the interval,
  a week on, holds it; and `Truth.switchback`, the test's whole lift with its carry-out, beside
  them. The bench states the gap between its steady truth and the difference in means a reader
  reads, 0.956 to 0.996 of the truth on its 35 pilot worlds: the kept weeks off still carry the
  lags of the blocks on before them.
- **`flighting` is scored on its schedule where the bench scores one**
  (`scorecard.arms.decisions.scored`, under `schedule score`): the bench's score of the record's
  schedule, its regret against the best schedule, on the realised path too, the best schedule's
  edge over the best plan, its move into the box, and the rate's regret beside them. Where the
  schedule is recommended, the variant's regret is the schedule's: what it returns short of the
  best plan, per euro, the schedule's regret less the edge, so that it is read on the axis every
  other arm is read on, and may be below 0. Where the rate is recommended, its regret stays the
  rate's. Where the bench refuses the schedule (cells not concave, a line, or cells without each
  week's reach: twins, siMMMulator, S-curves, Robyn's genre, outside's cliff), the regret is the
  rate's and the score holds the bench's reason. The claimed gain stays the rate's. The record no
  longer names the schedule absent.
- **The adapters are removed**: `scorecard.arms.universes`, with `geo_universes`,
  `switchback_universe` and `world`. Nothing else read them.

Records: the reading arms' records are unchanged in form, and on family 8 in value (the same
draws); the switchback arm's flags move with the bench's design and universe, and its settings
add `lower` and `upper`; `flighting`'s record drops `absent.schedule`. Scores add `switchback
score` and `schedule score` to those variants' records, and `flighting`'s regret, realised regret
and move are the schedule's where it is recommended. Old records and old score lines still load:
the new keys are read where present. A run exported before version 4 is read by every arm but the
readers and the switchback, which refuse it.

Consequences for a pre-registration that names these arms: `flighting`'s regret is the
recommended plan's, a schedule or a rate; `switchback`'s reading has its own error against a
stated truth.
