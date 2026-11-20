# 0044. A no-test plan read against the equal split

Status: accepted, 2026-10-12; amends ADR 0021's arms and ADR 0041's CHC arm; with ADR 0043, a
geo export's market reads the bench's equal split of its cells summed by channel; amended
2026-11-07: a plan of the posterior's mean names each channel it moves whose spend never moved;
amended again 2026-11-07: so do the plan for the worst share at every level, the flighted plan,
the plan of the geos and the decision page

## Context

Most panels have no lift test. The engine's rule for a plan without one is that it moves only as
far as its evidence does: with no test, it must not do worse than the equal split in the
posterior's own bad cases.

The scorecard's no-test plans read the status quo as their reference. `cvar 0.1` (ADR 0029) and
`dlm cvar 0.1` plan, by `chc.allocation.cvar_allocate`, the split with the most mean gain over the
status quo in the worst tenth of their readings. The reference gains 0 under every reading, so
the split never does worse than the status quo there. It can do worse than the equal split. On
the drift run's scored worlds at k = 0, pooled over its laws, the equal split's mean regret per
euro was below every engine arm's, and below every tool's but Meridian's (the run's report).

The bench scores the equal split on every world without a record
(`causaldyn_bench.scorecard.scoring.BUILT_IN`): an equal share of the budget a channel, moved into
the box at the budget by one shift (`causaldyn_bench.mmm_decision.Quarter.equal_split`). The
export holds the budget and the box. It does not hold the split.

## Decision

- **`scorecard.arms.panel.equal_split(observation)` rebuilds the bench's equal split.** It makes
  the bench's `Quarter` from the export's budget, box, status quo and history, and calls the
  bench's own `equal_split`. The rule is called, not restated. An export with a geo axis is
  refused: the bench splits a geo world's budget over its channels, and their cells by
  population. A geo export's market (ADR 0043), which the national arms plan, reads the bench's
  own `GeoQuarter.equal_split` of the geo export's cells summed by channel: each channel's total
  as the bench spends it, which the market's allotment then splits over the geos as the status
  quo does. Those totals are not the market's own equal shares: on family 8's first three pilot
  worlds at policy 0 they lie up to 10 % a channel off them, as the bench moves each cell into
  its box.
- **A variant for each arm that holds more than one reading of its fit.** Each plans by
  `cvar_allocate` at level 0.1 over the arm's readings, against the equal split, within the
  export's box and budget, with the history carried in:
  - `cvar 0.1 vs equal`: the static arm's readings of the stacked posterior (ADR 0021);
  - `dlm cvar 0.1 vs equal` and `dlm untested cvar 0.1 vs equal`: the DLM's readings over its
    fitted specs, and its ablation's, so that what the tests are worth to this plan is read as
    for the DLM's other plans (ADR 0038);
  - `chc cvar 0.1 vs equal`: the CHC arm's readings, its chosen curves at the coefficients of
    evenly spaced sign-held draws (ADR 0041).
- **The ridge arm gets none.** Its fit is a point: one reading. The worst tenth of one reading is
  that reading, so the plan would be the split that reading returns most on, whatever the
  reference: `ridge`'s own plan. The search's polished starts are local optima of a validation
  score, not draws of a posterior, so they are not read as readings.
- **The variant runs at every rung.** The static arm and the DLM read the tests, so at each rung
  the variant plans on that rung's posterior, and it moves as the tests move it. The CHC arm and
  the DLM's ablation read none, so their plan at every rung is k = 0's.
- **The record is the other CVaR variants' record.** It holds the plan; the claim of its gain over
  the status quo, the mean and the central 90 % interval over the readings, which the bench
  scores; the fit's flags and settings; and the variant that carries the fit's forecast and
  response. Every CVaR variant's settings now name its reference, `"against": "equal split"` or
  `"status quo"`.
- **The existing variants keep their names and their reference.**

## Consequences

- The split's worst tenth gains at least 0 over the equal split **on the readings**, not on the
  world. A posterior that is wrong in all its readings carries the error into its worst tenth, so
  the scored regret can still exceed the equal split's.
- Where the readings disagree about every move, the plan is the equal split. Where they coincide,
  it is the plan for that one reading. Between the two, it moves from the equal split only as far
  as its worst readings agree it gains.
- The CHC arm's readings hold the search's chosen curves and vary only the coefficients. Its worst
  tenth so reads less of the fit's uncertainty than the static arm's or the DLM's.
- Each world costs each of these arms one more `cvar_allocate` over its 400 readings, and one
  equal split. The fit is not repeated: every variant of an arm plans from the same fit.
- **A run declared before this decision** names its records by `scorecard.run.names`, which now
  holds the new variants. Relaunching `arm` on such a run would fit every world again, since the
  new records are missing, and would rewrite the world's other records with the new fit; `score`
  waits for the new records. Such a run is read as it stands, and not relaunched on this code. The
  drift run's pre-registration (`scorecard/drift.py`) does not change.
- **The tests.**
  - The equal split rebuilt is the bench's, to the bit, on a drawn and a reference world of Track
    M v2 and a world of the drift family's walk, and a record that plays it is scored at the
    equal split's regret. An equal share below a channel's floor is moved into the box, and a geo
    export is refused; its market's equal split is the bench's split of its cells summed by
    channel, to 1e-12 of it.
  - Each variant hands `cvar_allocate` the fit's readings, level 0.1, the equal split, the box, the
    budget and the history. The existing CVaR variants still hand it the status quo.
  - On each arm's fit of a drawn world, the plan's certificate holds when it is read again with
    the harness's own worth: the worst tenth's mean gain over the equal split is at least 0, and
    it is `cvar_allocate`'s to 1e-9 of the largest return.
  - Twenty readings of three like channels, each channel the most profitable in at least a tenth
    of them, give the equal split to 1e-9, though their mean's plan puts more than 70 of the 150
    a week on the first channel. Twenty readings alike give `allocate`'s plan on that reading, to
    1e-4 of each spend and 1e-9 of its return. Maxima checks the construction: each reading's
    first-order gain of a move `d` is proportional to `2 d1`, `d2` or `-d1 - d2`, and no move but
    0 makes all three at least 0, so every move loses under at least a tenth of the readings.
  - The record's fields; the same plan twice from one fit; the CHC arm's records the same twice
    at a seed, and scored at both rungs beside the status quo and the equal split.
  - Fifteen mutants of the equal split, each variant's reference, level and readings, the
    variants' names and the record each fail a test.
- A smoke run of the harness, on three drawn worlds of Track M v2 at k = 0, fitted the four arms,
  recorded each new variant and scored it beside the equal split and the status quo, none failed.
  It checks the path from the export to the score, not what the variants are worth.

## Alternatives considered

- **The equal split's rule restated in the engine.** Rejected: two copies of one rule can drift,
  and the bench is already on the harness's path. A test holds the split to the bench's own, to
  the bit.
- **The equal split in the export.** It would be the bench's change and a new export version, and
  the bench's scorer would then read one split from two places. Not taken.
- **The existing `cvar 0.1` variants moved onto the equal split.** Rejected: the drift run's
  records and its pre-registration name them, and one name would then mean two plans.
- **Both references at once**: the split with the most mean gain over the equal split in the
  worst tenth, held to gain over the status quo there too. `cvar_allocate` reads one reference,
  so this waits for CHC.
- **A ridge variant.** Rejected: it would record `ridge`'s plan twice.
- **The certificate in the record.** The record's claim is the gain over the status quo, which
  the bench scores, and the certificate is a property of the plan that the tests hold. A record
  that carried it would hold a second gain, read against another reference. Not taken.

## Amended 2026-11-07: a channel whose spend never moved, in a plan of the posterior's mean

- **The world.** `tests/test_flat_spend.py`: 104 weeks; retail media spends 80 every week, and
  before the panel too, so `read_panel` flags it `IssueCode.CONSTANT`, "its effect cannot be
  identified". The Bayesian model's 90 % interval of retail's return keeps 0.92 of the prior's
  width on the log scale. The plan this ADR's clause makes without a test, `plan_cvar` at 0.1
  against the plan in place, holds retail at 80. `plan_posterior`, the plan of the posterior's
  mean, cuts it to the floor of its range, 40, over 13 weeks within half to twice each channel's
  spend: its worth there is the prior's, and the plan said nothing.
- **Decision: the plan names it; no number moves.** `PosteriorPlan.unidentified` and
  `RobustPlan.unidentified` (`mixdyn.decisions`) name each channel the plan moves off its
  reference by more than `1e-12` of its range's most, the current spend for `plan_posterior` and
  `against` for `plan_robust`, whose spend the panel flags `CONSTANT` in the plan's geo or in
  every geo. It is always on, and every other field of either plan is what it was, to the bit; the
  field defaults to empty, so a plan saved before it loads.
- **Not an opt-in hold.** A range that is a point at the current spend already holds a channel in
  every planner (`Range(80, 80)`), and the plan without a test holds it by this ADR's clause. A
  second switch would say the same thing twice. Holding it by default would move the plan of the
  mean, which is not the plan made without a test, so it is not done.
- **Read from the panel's flag**, as `read_panel` raised it, by the `CONSTANT` issue's `channel`
  (amended 2026-11-07; it read the issue's `column`, so a channel read from a column of another
  name, `spend={"retail": "retail_eur"}`, was never named, and a constant control whose column bore
  a channel's name named that channel; ADR 0063, amended 2026-11-07). A panel built field by field
  carries no flag, and its plan names nothing; nor does a panel loaded from a file saved before
  issues named their channel.
- **Left out.** `plan_cvar` above 0.1 can move such a channel (retail by 1 % at 0.25 on the world
  above), and `plan_flighting` and `plan_geos` read one point of the posterior, its median; none
  names it yet. The decision document (`mixdyn.api.decision`) does not carry the field: it is a
  versioned schema (ADR 0063), and a new field there is a new version.
- **Tests** (`tests/test_flat_spend.py`): on the world above, `plan_posterior` names retail and
  only it, and its spend, current spend, budget, gains and grid equal, to the bit, those of the
  same plan on the panel without the flag, which names nothing; with retail's range the point 80,
  retail is held and not named; `plan_robust` cuts retail to 40 and names it at a factor of 1.1,
  holds it at 80 and names nothing at 1.5, and names nothing where its range holds it. A plan
  naming a channel is saved and loaded (`tests/test_roundtrip.py`).

## Amended again 2026-11-07: the flag in every plan and on the decision page

- **What the amendment above left out**: `plan_cvar` above 0.1, `plan_flighting`, `plan_geos` and
  the decision document. Each now names the channels, or the cells, it moves whose spend never
  moved, by the rule above: moved off its reference by more than `1e-12` of its range's most, its
  spend flagged `CONSTANT` in the geo or in every geo. No number of any plan changes.
- **`plan_cvar`, at every level.** "Above 0.1" is where the plan moved the channel on the
  flat-spend world, not a rule of the level. At the test's fit the plan keeps retail at 80 to the
  bit at 0.1 and 0.15, and moves it by 0.21 at 0.2, 0.82 at 0.25, 21.7 at 0.5 and 40 at 1.
  `CvarPlan.unidentified` reads the move against `against` at any level, so at 0.1 it names
  nothing there because the plan moves nothing. Rejected: a flag gated on the level. It would stay
  silent at 0.1 on a world where that plan did move such a channel.
- **`plan_flighting`.** `FlightPlan.unidentified` names a channel that the schedule, in any period,
  or the continuous plan moves off its current spend. The plan holds both; the recommendation
  picks one, and either may be the one a planner acts on.
- **`plan_geos`.** `GeoPlan.unidentified` names cells, `(geo, channel)` in the panel's order. A
  channel flagged in one geo is named in that geo only; one flagged at one value in every geo
  (an issue with no geo) is named in each geo whose cell the plan moves. A national channel's cells
  have no range of their own; the national range's most stands in for theirs. The plan reads the
  caller's responses, not a posterior's draws, but the flag is the panel's: whoever fitted those
  responses read no movement in that cell's spend.
- **The decision page.** `Decision.unidentified` holds the plan's channels by name, the last
  field, written on the page as `unidentified: none` or as a list. The schema stays
  `"mixdyn.decision"` version 1, under ADR 0063's clause that a schema changes in place before
  the first release: no released engine wrote a decision document. A document written without the
  field is refused, the field named. ADR 0072 is amended to say so.
- **Saved files.** Each new field defaults to empty, so a plan or a decision saved before it loads
  naming nothing (ADR 0062); `FORMAT_VERSION` stays 1.
- **A channel read from a column of another name** is named too: an issue about a channel's spend
  carries the channel, and the flag reads it (ADR 0063, amended 2026-11-07).
- **Tests.**
  - `tests/test_flat_spend.py`: `plan_cvar` keeps retail at 0.1 and names nothing; at 0.25 it
    names retail and only it, and every other field equals, to the bit, the plan on the panel
    without the flag; a point range holds retail and names nothing. The flighted plan at the
    posterior's median cuts retail to 40 in every week and names it, its numbers equal to the
    unflagged plan's, to the bit; held, it names nothing. `plan_geos` on the world's own curves
    cuts retail to 40 and names its cell; held, none.
  - `tests/test_geo_plan.py`: TV flat in one of two geos, read by `read_panel`, is named in that
    geo alone though the plan moves it in both, every number equal to the unflagged plan's; held
    at a point it is not named; flat at one value in both geos, both cells are named. A national
    channel flagged in every geo is named in both geos' cells, and not where its national range
    holds it at its current spend.
  - `tests/test_factsheet.py`: the decision carries the plan's names and its page lists them; the
    tiny world's page reads `unidentified: none` (the golden page gains that line); a document
    without the field is refused. A fifth report, a plan naming search, joins the reports whose
    every field is on its page and in its document.
  - `tests/test_roundtrip.py`: each of the six classes naming a channel by hand is saved and
    loaded the same; saved with the field removed from the file, each loads naming nothing.
