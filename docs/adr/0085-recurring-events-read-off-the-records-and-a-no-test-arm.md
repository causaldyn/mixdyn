# 0085. Recurring events read off the records, and a no-test arm

Status: proposed, 2026-11-16; adds `mixdyn.data.recurring_events`, `PerEffect.held`,
`search_media`'s `half_saturation_prior` and the scorecard's `dlm-events` arm; changes no default
and no other arm's plan. A screen on M2's worlds found no lower regret at k = 0 (Consequences)

## Context

M2's no-test plan, `dlm plug-in` at k = 0, lost to Meridian on every confirmatory family and to the
equal split on most. The planner was not at fault: on the worlds' true channels it plans with
regret 0. The estimate was. Two errors compounded, each measured by a one-change refit on M2's
archived exports:

- **The promotion control.** The bench's export records its promotions as the reference paper's
  measurement layer does: a flag that is on in a promoted week with probability 0.7 and in any
  other week with probability 0.15, and a price that drops by the promotion's discount. The DLM read
  the flag as one control, one coefficient. The true promotions lift sales by +52 % to +129 % by
  event, and the cyber week sits just before television's Christmas burst. With the worlds' own
  curves, one coefficient on the flag left television's coefficient at e^-1.28 of its effect; one
  coefficient on an exact flag at e^-1.09; a dummy for each promoted week of the year at e^+0.09,
  and regret then fell from 0.228 to 0.131 a euro on 122 worlds. A rule that reads the weeks off the
  flag alone (on in a majority of years) missed 16-29 % of them and added 2.6 false weeks a world,
  and did not lower regret.
- **The curve without tests.** Without lift tests the one-step likelihood is flat in a channel's
  half-saturation point: more than half the specs the search filters are kept, and the kept set
  reaches both ends of the grid in 95-100 % of worlds. Its best put television's half-saturation
  at about e^-2 of the truth's, a curve saturated at the status quo's spend, so every plan cut
  television. A wider grid lets the argmax run further on the same flat surface.

The promotions recur: the same weeks of every year. The flag and the price, read across the years,
say which weeks those are far better than either says it of one week. And the engine's static
Bayesian model already holds a prior on the half-saturation point (`K_m ~ LogNormal(0, 1)` about
the channel's mean positive spend, `mixdyn.models.bayes_mmm`), which the DLM's search did not.

## Decision

- **`mixdyn.data.recurring_events(panel, indicators)` reads the weeks of the year an event recurs
  in.** Each week of the year (ISO 8601's, its 53rd read as its 52nd) is one of two classes, the
  same in every year. The named controls, each turned by its direction so that an event raises
  it, are independent given the class: a control of two values, a flag, is a Bernoulli of a rate a
  class, drawn towards one half by one week of each value; any other is a normal of a mean and a
  variance a class. Expectation maximisation fits the two classes from the split of the weeks by
  their mean indicator; the two are read only where they beat one class by the Bayesian
  information criterion. A week of the year whose posterior is above one half is an event's, and
  each run of consecutive event weeks, the year's last next to its first, is one event, named by
  its weeks. `RecurringEvents.days` gives each event's days as `MediaPanel.events` holds them (ADR
  0069), and `columns` its column over any weekly dates, the calendar's or a forecast's past it.
- **CHOICE: the KPI is no indicator.** A channel's bursts recur too, in the bench at weeks 48-52 and
  23-25; read off the KPI they would be events, and the event's coefficient would take the
  channel's burst.
- **CHOICE: a flag is a Bernoulli, not a normal.** A normal of the flag in a class whose weeks
  all hold one value has no width; the flag alone then decides each week whatever the price says,
  and on a year of weeks (family 7's `L-52`) the false positives became events.
- **`PerEffect(values, held=names)`** keeps the named effects' discounts as the spec gives them: no
  parameter, no block of the search, no end. An event recurs once a year, so its coefficient is
  static (ADR 0069), and searching its discount would multiply the search's pairs.
- **`search_media(..., half_saturation_prior=s)`** adds to every spec's score the log density of a
  log-normal prior of scale `s` about the channel's mean positive spend, the grid's unit:
  `-(log m)^2 / (2 s^2)` for the multiple `m`. The search's best, its moves and its kept set read
  the sum; where the likelihood is flat the prior decides. The grid is even in `log m`, so no
  Jacobian enters. Without the argument the search is as it was.
- **The scorecard's `dlm-events` arm** (`scorecard.arms.dlm.DlmEvents`): the M2 arm's search on the
  export's panel with the events `recurring_events` reads off the flag (+1) and the price (-1) in
  place of the flag, each event a static effect held out of the discount search, the price a
  control as before, the half-saturation prior at the static model's scale, 1. Where no event is
  read, the flag stays. Its four variants plan as the M2 arm's (`dlm events plug-in`, `mixture`,
  `cvar 0.1`, `cvar 0.1 vs equal`). It reads no test (`tests = False`): the harness fits it once
  for every rung, and its plan at a rung with tests is k = 0's.
- **Its claim is calibrated to its planner's level.** A plan claims as its gain the mean of the
  worst tenth of its readings' gains (`claimed`), the lower tail `cvar_allocate` plans on at 0.1,
  not their mean: M2's plug-in claimed 1.76 a euro more than it realised. The interval stays the
  readings' central 90 %.
- **Defaults are unchanged.** The M2 arm searches, plans and claims as it did; a test holds what
  it hands the search.

## Consequences

- *The event terms' criterion*: on a world with a yearly cycle and three events, the level-only
  model's interval of a channel's return misses the truth and the model with the event terms holds
  it in at least 90 % of seeds. Measured (`scripts/study_recurring_events.py`, worlds 0 to 199):
  156 weeks of a KPI with a yearly cycle, one channel returning 0.8, three events that recur in
  the same weeks (a clearance in weeks 2-4 lifting 250, a spring sale in week 16 lifting 400, a
  cyber week in weeks 46-47 lifting 600), the channel's spend rising from 100 to 250 in them,
  recorded by a flag of recall 0.7 and false positives 0.15 and a price 1 less the discount (0.4,
  0.3, 0.4) with noise of SD 0.08. The events read are the world's, every week of each and no
  other, in 198 of 200.

  | model | 90 % holds 0.8 | 95 % holds 0.8 | mean 90 % interval |
  |---|---|---|---|
  | the level, the season, the flag and the price | 0 of 200 | 0 of 200 | [2.180, 2.756] |
  | the events read in place of the flag | 185 of 200, 0.925 | 192 of 200, 0.960 | [0.617, 1.018] |

  The Monte Carlo standard error of a coverage of 0.9 over 200 worlds is 0.021.
- *On the bench's exports* (200 of M2's scored worlds of families 0, 2, 3, 5 and 7 at k = 0),
  the weeks read from the flag and the price are the promoted weeks of the year in 1 777 of 1 800
  promoted week slots over 200 worlds, with 10 false weeks, 8 of them on family 7's one-year and
  one-and-a-half-year panels.
- *A flag alone* of recall 0.7 over three years is weak evidence: with nine promoted weeks the
  information criterion keeps one class in most draws. The price is what makes the events
  readable; a clean flag alone reads them.
- *The screen against M2's recorded no-test plans* (20 of M2's scored worlds at k = 0, four of
  each of families 0, 2, 3, 5 and 7 over their environments, chosen and the rule written before
  any refit): the arm was refitted on each, and each plan scored by the bench's scorer, which
  reproduces M2's recorded regret of `dlm plug-in` exactly; M2's arm refitted on one of the worlds
  reproduces its four recorded plans. The arm does not lower regret. Mean regret a euro:

  | planner | `dlm-events` | M2's `dlm` | paired difference, 95 % |
  |---|---|---|---|
  | plug-in | 0.307 | 0.215 | +0.093 [-0.022, +0.207] |
  | mixture | 0.284 | 0.213 | +0.071 [-0.040, +0.182] |
  | cvar 0.1 | 0.196 | 0.187 | +0.009 [-0.019, +0.036] |
  | cvar 0.1 vs equal | 0.307 | 0.222 | +0.085 [+0.012, +0.158] |

  Meridian's plans on the same worlds: 0.177; the status quo's 0.203; the equal split's 0.328.
- *Why, as far as measured*: television's curve left the grid's low end, as intended (its
  half-saturation point a median 1.18 times its mean positive spend), and the plug-in cut
  television in 7 of the 20 worlds where M2's had cut it in all 20. But the best plan, the bench's
  oracle, cuts television in 16 of the 20: a flat curve is not a strong one. The plug-in moved 23
  of the 60 channels in the oracle's direction, M2's 34. `cvar 0.1` held the status quo in 15 of
  20 worlds (M2's in 6). Refitted on the same worlds with each change alone, the events carry the
  loss and the prior lowers nothing: less M2's arm, the events alone +0.083 (plug-in), +0.049
  (mixture), +0.009 (`cvar 0.1`) and +0.076 [+0.003, +0.149] (`cvar 0.1 vs equal`); the prior
  alone +0.053, +0.020, +0.001 and +0.009. The events read the promotions, but with curves no
  test pins they change which wrong curve the search keeps; that is a hypothesis, not measured.
- *The claim*, less the realised gain a euro, on the same worlds: the mean of the worst tenth of
  the readings' gains claims -0.54 (plug-in), -0.40 (mixture), +0.015 (`cvar 0.1`, 15 of whose 20
  plans are the status quo's, which claims and realises 0) and -0.41 (`cvar 0.1 vs equal`); the
  readings' mean of the same fits +0.40, +0.42, +0.07 and +0.15; M2's recorded claims +1.46
  (plug-in), +0.52 (`cvar 0.1`) and +0.71 (`cvar 0.1 vs equal`). The claim at the CVaR's level
  suits the planner at that level and under-claims the others. Over M2's 1 073 recorded k = 0
  plans of those families, the low end of the 90 % interval claims -1.07 (plug-in) and +0.19
  (`cvar 0.1`) a euro, the mean +1.78 and +0.62: no one quantile of the readings is calibrated
  across the planners.
- *The arm's cost*: on a shared laptop, a fit took 890 CPU-s on average over the 20 worlds (median
  784, at most 1 424; 498 on family 7's), its peak 5.9 GiB. On the one world both were fitted
  there, M2's arm took 1 274 CPU-s and this arm 1 424, under different loads; per world, this arm's
  laptop fit took a median 1.32 times M2's arm's on M2's box, and M2's arm took 1.35 times on the
  laptop what it took on the box. A fit costs about what M2's arm's does.
- *The tests* (`tests/test_recurring_events.py`, `tests/scorecard/test_dlm_events.py`, and
  additions to `tests/test_dlm_choice.py` and `tests/test_dlm_media.py`): ISO weeks over
  1999-2030; the events, their runs across the year's end and their columns past the calendar;
  records that do not recur read no event on ten draws; refusals; the event terms' criterion on
  worlds 0 to 39; a held effect's grid; the search with a prior ending at the dense grid's best
  posterior and a tight prior holding the point at the mean positive spend; what each arm hands
  the search; the claim at the CVaR's level; the forecast's event columns over the quarter.

## Alternatives considered

- **A dummy for every week of the year, shrunk.** Television's bursts recur in fixed weeks, so
  free week dummies take the bursts; a normal shrinkage prior learns the promotions' large lifts
  and then shrinks the bursts' smaller ones too little. Not tried on the bench for that reason.
- **The weeks a majority of years flag.** Measured above: it misses a fifth of the promoted weeks
  and adds false ones, some on television's bursts.
- **The flag's error rates as known.** They are the bench's, not an analyst's; the mixture learns
  them.
- **A wider grid, or a fixed half-saturation point.** Measured on M2: neither helps; the likelihood
  is flat.
- **West and Harrison's intervention monitoring.** It would stop missed promotion weeks from
  leaning on the level, but a week it discounts carries no information about the media either, and
  it does not read event-specific lifts. Not built.
