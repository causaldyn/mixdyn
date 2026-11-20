# 0051. Scenarios read as differences with their carry-out

Status: accepted, 2026-10-28; builds on ADR 0028's posterior predictive and reads a plan as
ADR 0005's `GeoPlan.gain` does

## Context

A planner asks what another spend path would do. `forecast_bayes` (ADR 0028) draws the static
model's KPI under one planned path, and nothing compares two. Comparing their forecasts over the
horizon misreads them in two ways. The spend of the horizon's last periods carries over past it:
under a normalised kernel a unit spent in the last period returns its whole effect only over that
period and the kernel's `length - 1` after it, so a horizon's totals undercount a late spend and
favour an early one. And two forecasts drawn with their own noise differ by the noise of both: at
a noise scale `sigma` a period over `H` periods, the gap of two totals has a standard deviation of
`sigma sqrt(2H)` whatever the plans, which buries a small change of spend.

Three defects of a scenario reader seen in use, stated here as properties, set the tests: a
channel's coefficient applied twice, so zeroing it took away twice its contribution; the plan
ignored, so a scenario that changed spend read the baseline's forecast; and a move between
regions made by multipliers on the two cells, so moving 20 from a cell of 100 to one of 1000 moved
200 into the second and created 180.

## Decision

- **`mixdyn.decisions.scenarios.Scenario(name, spend, controls={})`**: a named path over the
  horizon, each channel's spend and the controls it changes; a control it does not name is held at
  the baseline's.
- **`compare_scenarios(fit, scenarios, *, baseline, horizon, draws, seed, level=0.9)`** runs
  `forecast_bayes` for each scenario over the horizon and the `fit.length - 1` periods after it, in
  which every scenario spends nothing and holds the baseline's controls at their last value. Each
  scenario's **total** is its KPI drawn over the horizon alone, its mean and central `level`
  interval over the draws. Each scenario's **difference** is its mean less the baseline's at each
  draw, summed over the horizon and the carry-out. The two are read over different windows, and the
  docstring says so: a total is what the horizon shows, a difference what the spend changes.
- **Common random numbers**: every scenario reads the same thinned draws and the same noise, one
  `seed`. A difference is read from the means, so no noise enters it at all, and a scenario equal
  to the baseline differs from it by exactly 0 at every draw. Under log-normal noise the mean is the
  KPI's expectation (ADR 0017), so a difference of means is the difference of expected KPIs there
  as under normal and Student-t noise.
- **`media_difference(channel_draws, history, plan_a, plan_b, periods)`**: the same difference read
  through the channels alone, each draw's `chc.response.Channel` run over the history, the plan and
  the kernel's reach, for any fit that hands `channel_draws`: the static model, one geo of the
  model over geos (with that geo's history), the nested model. The non-media terms are additive in
  the mean in every one of these models, so between two plans under the same controls they cancel.
  This is `GeoPlan.gain`'s accounting, a plan read against another over its periods and the
  carryover after them.
- **`move_between_geos(plan, channel, from_geo, to_geo, amount)`**: a budget-neutral move on two
  cells of a plan by geo and channel. The larger cell after the move is rounded to its last bit and
  the smaller is the cells' sum less it, computed in exact rational arithmetic, so what leaves one
  is what the other gains and the two cells hold the same real sum. A move larger than the source
  holds is refused. Where the sum's last bit is finer than the smaller cell can hold (moving 5000
  from 9000.37 to 12.34 leaves about 4000.37 and 5012.34, neither of which has room for 12.34's last
  bits), no two floats near the move hold that real number; the two cells' double-precision sum
  is then kept to the bit instead, the real sum within 1.5 units in the smaller cell's last place.
- **`response_curves(worths, readings, *, level, current=None)`**: each channel's mean worth over
  the draws at `Readings.weights`, and its central `level` band by the weighted inverse of the
  empirical distribution, which at equal weights is `numpy.quantile`'s `inverted_cdf` (Hyndman and
  Fan 1996, definition 1); a current spend is marked at the mean, linear between the grid's points.

## Consequences

- *Zeroing a channel* at the history's own spend run on differs from the baseline by that
  channel's contribution over the plan and its carry-out less what the history alone carries into
  them, `chc.response.contribution` at each draw, to `1e-12`; with no history, by its whole
  contribution. Doubling a channel's spend moves the difference by what `chc.response` gives at
  each draw, never 0 at a coefficient that is not.
- *Two linear geos*, spends 100 and 1000 returning 2 a unit: moving 20 changes the KPI by exactly 0
  and the total not at all (multipliers read +360). Spends 100 and 300 returning 2 and 1: moving 20
  from the first to the second changes the KPI by exactly -20 (multipliers read +20).
- *Through a fit*: on a short `fit_bayes` fit with a trend, a season and two controls, a scenario's
  difference is `media_difference` at the draws `BayesFit.channel_draws` reads, to `1e-12` of the
  KPI's mean over the horizon, under normal and log-normal noise; raising a control by 1 over the
  horizon moves it by the control's coefficient times the horizon's length; and a total less its
  mean is the same draw of noise in every scenario.
- A hand mutation pass of the module, 17 mutants (the coefficient applied twice, the plan ignored,
  the carry-out dropped in either reader, independent noise per scenario, differences from the
  drawn KPI, a move by multipliers, unweighted bands among them), is killed by the tests.
- Scenarios are read on the static model's forecast only: the model over geos and the nested model
  have no forecast yet, and are compared through `media_difference`, which holds their non-media
  terms fixed.

## Alternatives considered

- **Scenario totals over the horizon alone, their gap the comparison.** Rejected: it drops the
  carry-out, so a path that spends late reads worse than one that spends the same early, by the
  share of its effect that lands after the horizon.
- **Independent noise per scenario.** Rejected: the gap of two totals then carries the noise of
  both, `sigma sqrt(2H)` at normal noise, and a difference worth reading is buried in it; the draws
  of the posterior must be shared too, or the gap mixes two posteriors' spread with the plans'.
- **Differences from the drawn KPI under common random numbers.** Under additive noise it is the
  difference of means up to rounding, but under log-normal noise the noise multiplies the mean and
  the gap of two draws is the means' gap times it. The means' gap is the expected one in every
  model. Not taken.
- **Moves by multipliers on the cells**, the share of the source applied to the target. Rejected:
  it moves the source's share of the target's spend, not the amount, and makes or loses money
  wherever the two cells differ.
- **Moves in decimal arithmetic.** It would keep a total of decimal amounts, but the plan's cells
  are binary floats read by `chc.response`, and a decimal total does not survive their conversion.
  Not taken.
