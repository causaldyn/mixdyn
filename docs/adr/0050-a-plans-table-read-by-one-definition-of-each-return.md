# 0050. A plan's table, read by one definition of each return

Status: accepted, 2026-10-28; amended 2026-10-28: the weighted quantile is one helper,
`mixdyn.decisions._quantile`, that `response_curves` and `plan_flighting`'s interval ends read too,
a cumulative weight reaching a share within its sums' rounding;
amended 2026-11-07: the diminishing-returns point reads its means in closed form, at every draw and
spend at once

## Context

The engine plans a posterior's budget (`plan_posterior`, ADR 0019; `plan_cvar`, ADR 0029;
`plan_robust`) and reads how far a plan's effects may be off before its gain is gone
(`plan_breakdown`). Each hands back spend vectors and totals. A planner reads a plan as a table,
one row a channel: what it spends now and in the plan, what the change does to the KPI and to
revenue, what a further unit returns, and where returns stop paying. Meridian and PyMC-Marketing
both report such a table; a planner expects one.

A table is where returns are most easily misread. ADR 0016 lists the misreadings: a coefficient
reported as a return, a net return beside a gross one, a marginal return that drops the carryover.
A table that computes its own returns can bring any of them back. A table that reads breakeven at
a gross return of 1 reads it on revenue, and says nothing about profit: at a gross margin `m` a
gross return pays back at `1 / m`.

## Decision

- **`plan_table(panel, geo, readings, spend, periods, *, against, revenue_per_kpi, margin, level,
  limits)`**: one row a channel, in the readings' order, of a plan of one geo's channels a period
  against `against`, or else the current spend `plan_breakdown` reads, each channel's mean over the
  geo's last `periods` observed periods. `PlanTable.rows()` hands the rows on by column name.
  - **Current and planned spend a period, and the change in percent.** Where the current spend is
    0 the change is NaN, or 0 where the plan spends nothing either: a percentage of nothing is not
    a number, and a held zero is no change.
  - **The KPI change** over the plan's periods and the carryover after them, the history's carried
    in: each draw's is `Breakdown.gains`, so the rows' means sum to the plan's expected gain over
    the reference, and a channel the plan holds changes by exactly 0 at every draw.
  - **The revenue change**: the KPI change times the channel's revenue per KPI, one rate for every
    channel or one a channel, at each draw, in the mean and in the interval's ends.
  - **The marginal return a currency unit at the current and at the planned spend**: at each draw,
    `returns(...).marginal_roi` of the draw's channel, `chc.response`'s, on the history's spend
    followed by the plan's periods at that rate, the series the planner reads, the window those
    periods. At a rate of 0 the window spends nothing and `returns` reads no return: the table
    holds NaN there.
  - **The diminishing-returns point**: the spend a period, within the channel's range, past which
    the posterior-mean marginal return stays below breakeven. It is read at 64 spends evenly across
    the range and bisected between the last at or above breakeven and the first of the run below it
    that reaches the range's most, to `1e-9` of the range. It is None where the mean is at or above
    breakeven at the most, or below it from the first spend that spends something. The range is
    `limits`'s, or else from 0 to the most the channel spent a period in the history, the plan or
    the reference: where its response was seen or is asked for, as ADR 0019 reads a response
    nowhere past its range.
  - **Breakeven is 1 on revenue, and `1 / margin` on profit** where a gross margin in `(0, 1]` is
    given. The returns stay gross: the margin moves the line they are read against, not them.
  - **The table's level**: the KPI and revenue changes summed at each draw, the total's mean the
    plan's expected gain; the plan's breakdown factor `Breakdown.factor`, and the interval of the
    draws' own factors.
- **Every return is `returns` at a draw.** The module holds no return formula of its own. A
  coefficient is never a return: on a Hill it is the curve's ceiling, not what a unit earns.
- **Each quantity is a `Spread`**: each draw's value, their mean by the draws' weights, and the
  `level` interval, 0.9 by default. **An interval's ends are weighted quantiles**: at
  `(1 - level) / 2` and `(1 + level) / 2`, each the least draw whose cumulative weight, the draws
  sorted, reaches that share of the total, the inverse of the weighted empirical distribution,
  Hyndman and Fan's (1996, *The American Statistician* 50(4)) first definition with each draw at
  its weight. An end is always a draw's own value; a draw of weight 0, a stacked fit of weight 0's,
  is never read. A quantity NaN at any draw is NaN in its mean and its interval.
- **One weighted quantile for every decision module.** `mixdyn.decisions._quantile` reads the
  ends of `plan_table`'s intervals, of `response_curves`' bands (ADR 0051) and of
  `plan_flighting`'s intervals (ADR 0052). A cumulative weight reaches a share where it is within
  a relative `1e-12` of `share * total`: the two are rounded by different sums, and at 20 draws of
  weight 0.05 the first five's 0.25 is a unit in the last place short of `0.25 * total`, the total
  being a unit above 1, so a plain comparison read the sixth draw as the quartile. It did so at
  20, 40, 100, 200 and 1 000 draws of equal weight: in 16 of 30 cases over 10 to 1 000 draws and
  levels 0.5 to 0.98.
  `compare_scenarios` keeps `numpy.quantile`'s default interpolation for its totals and
  differences: its draws weigh alike and are as many as the caller asks for, so the argument
  against interpolating below, a few draws of uneven weight, does not reach it, and its outputs,
  recorded under normal and Student-t noise, stay as they were.
- **`ranges_from_current(current, *, least_share, most_multiple, floors)`** states a plan's ranges
  a period as a planner does: each channel's least `least_share` of its current spend, its most
  `most_multiple` times it, and its least `floors[channel]` where a floor in currency is named. A
  share outside `[0, 1]`, a multiple below 1 or not finite, and a floor above the most, below 0 or
  for a channel with no current spend are refused.

## Consequences

- *The tests* (`tests/test_plan_table.py`), on hand-made readings of three draws, two fits
  weighted 0.25 and 0.75, Hill and `tanh`, no posterior fit:
  - each draw's marginal return at the current and the planned spend is `returns`'s on a channel
    built in the test from the draw's parameters, to `1e-12`, at each channel's revenue per KPI,
    and differs from the draw's coefficient, and from it at that revenue;
  - with a margin of 0.4 the diminishing-returns point has the mean marginal return at or above
    2.5 just before it and below just after, within the bisection's tolerance, and at 1 without a
    margin, the two points apart; it is None on a range too narrow to reach breakeven and on one
    already below it;
  - a channel the plan holds changes the KPI by exactly 0 at every draw, against the current
    spend and against a named reference;
  - the rows' KPI means sum to `plan_breakdown`'s expected gain to `1e-12`, the draws are its
    gains, and the revenue column, its mean and its interval's ends are the KPI's times the rate;
  - every interval's ends are a brute force's weighted quantiles at levels 0.5, 0.9 and 0.99,
    and at 20 and 100 draws of equal weight and levels 0.5 and 0.9 the draws exact arithmetic
    reads from the level as written (`tests/test_decision_quantiles.py`), and
    a fit of weight 0 is never an end;
  - the change in percent, NaN from no current spend and 0 for a held zero; the refusals.
- *Mutations*: a hand pass of `plan_table.py` (ten or more mutants, each killed), among them the
  coefficient read as a return, breakeven fixed at 1, a held channel credited with its worth over
  no spend, unweighted quantiles and the bisection's direction.
- Each draw's marginal return is one `returns` call, through JAX, and the diminishing-returns point
  reads the mean at the scan's spends and the bisection's: a table of many draws costs that many
  calls a spend. The table reads a posterior's draws as they are given; thinning them is the
  caller's, as `readings(fit, draws=...)` does.

## Alternatives considered

- **A table at the posterior median only**, or at the posterior mean's parameters. A plan's
  consequences are uncertain, and the reader needs how much: a median row hides an interval that
  crosses 0. The return at the mean parameters is not the mean return on a curved response, and
  the rows of a median plan need not sum to anything the plan reports. Each column is read at
  every draw instead, and summarised by its weighted mean and interval.
- **A net return as the default.** ADR 0016 rejects reporting a net return beside a gross one:
  every return in the engine is gross. A planner who reads profit gives the margin, and the table
  moves breakeven to `1 / margin`; the numbers it compares stay the ones every other report shows.
- **Interpolated weighted quantiles.** With a few draws of uneven weight, interpolating between
  draws reports values no draw holds, and its definition differs between packages. The inverse of
  the weighted empirical distribution is the one definition that needs no choice.
- **A breakeven fixed at 1.** It reads every return on revenue, and a channel whose gross marginal
  return is 1.2 at a margin of 0.4 loses 0.52 a unit in profit while the table says it pays.

## Amended 2026-11-07: the diminishing-returns point read at every draw and spend at once

- **Context.** The point read each mean marginal return by one `returns` call a draw, through
  JAX, at each of the scan's spends and the bisection's: on three draws, three channels and a
  range from 0 to 80, 783 calls, beside the 18 of the two marginal-return columns. A posterior of
  200 draws makes that about 52 000 a table, and the scorecard's decision arm reads one table a
  world.
- **Decision.** The point's means are read in NumPy, at every draw of weight and every spend of
  the scan at once, and at every draw at each bisection step: the derivative of the draw's curve
  in closed form, `tanh`'s and the Hill's, along the adstock, which is linear in the spend. A
  draw's marginal return at `x` a period is `b sum_t h'(a_t + x r_t) r_t / periods`, `a` the
  adstock the history leaves and `r` that of one currency unit a period over the plan's periods,
  each over those periods and the kernel's length after them: `returns`'s jvp with the spend
  divided out. The scan, the bisection and the weighted `math.fsum` over the draws are unchanged.
- **The one formula outside `returns`.** The rule above, "every return is `returns` at a draw",
  still holds for every return the table reports: the marginal-return columns call `returns`. The
  point's means are not reported; a test holds them to `returns`'s.
- **Consequences.**
  - Each mean is the eager one to `1e-12` of it, its sums run in another order, on Hill, `tanh`
    and mixed draws, a fit of weight 0 among them, at 41 spends from 0.05 to 200 and two revenues
    per KPI. The point is the eager scan's to the bit on 15 cases of three channels, 15 points and
    30 Nones among them, and on the default range: the bisection reads only on which side of
    breakeven a mean falls. A mean within `1e-12` of breakeven could fall on the other side and
    move the point by up to a bisection step. Over every mean those cases read, the largest gap
    to the eager mean was `7.7e-16` of it, and the least distance from breakeven `4.3e-10` of it,
    at the bisection's last steps.
  - Counted on the same three draws: `returns` is called 18 times, none for the point, and the
    batched means 25 times a channel, once at the scan's 63 spends above 0 and once at each of
    the bisection's 24 halvings.
- **Alternatives.** A batched `marginal_roi` in `chc.response` would keep the formula in one
  place; its channel types check their parameters when they are built, so they do not take a
  draw's array of them, and that is a change to causal-hybrid-control. JAX's `vmap` over a
  rebuilt curve runs the same arithmetic as here through another kernel, with no tighter claim.
