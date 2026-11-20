# 0070. A conversions KPI's cost per incremental KPI

Status: accepted, 2026-11-06; adds `mixdyn.decisions.costs`, read off ADR 0016's `returns`, and
changes no return

## Context

A panel's KPI may count events, orders, installs or sign-ups (`Count(unit)`, ADR 0002). ADR 0016's
returns are then in the KPI's units per currency unit, and in revenue through `revenue_per_kpi`.
A planner of a conversions KPI reads the other way up: what an incremental order costs, CPIK, the
cost per incremental KPI Meridian reports for a KPI that is not revenue, and what one more costs
at the margin. Inverting a return by hand brings back what ADR 0016 removed, a CPA read off the
conversions a decomposition attributes to the window's periods, which drops the carryover past
the window's end, and an infinity or a NaN where the spend earned nothing.

## Decision

- **`costs(channel, spend, window) -> Costs`**, each cost the inverse of a return `returns` reads
  in the KPI's units, a revenue per KPI of 1:
  - `incremental`, the KPI the window's spend earns wherever it falls, its carryover counted:
    `roi` times the spend;
  - `cost_per_kpi`, CPIK, the window's spend over that KPI, `1 / roi`; it is the CPA where the KPI
    counts acquisitions;
  - `marginal_cost_per_kpi`, the cost of one more incremental KPI at the window's spend,
    `1 / marginal_roi`, one more currency unit spread over the window as its spend is.
- **No cost formula of its own.** Every cost is a `returns` reading inverted, so the carryover,
  the tail past the series' end and the window are ADR 0016's.
- **Revenue per KPI moves the return, not the cost.** `returns(..., revenue_per_kpi=v).roi` is
  `v / cost_per_kpi`; the costs take no rate.
- **A cost with no value is `Undefined(reason)`**, never an infinity or a NaN: a window that
  spends nothing has neither cost; a spend that earns no incremental KPI, or loses some, has no
  cost per KPI; a unit more that earns none, or loses some, has no marginal cost. The reason
  names which.

## Consequences

- *The conversions world* (`tests/test_costs.py`): 52 weeks of one channel's spend into orders, a
  normalised geometric kernel of 30 weeks at a retention of 0.6 and a Hill of half-saturation 50
  and slope 1.5 under a ceiling of 400 orders. The generator reads its own definitions with NumPy
  alone: the orders the window's spend adds, the whole series' less the series with the window
  emptied, run on for the kernel's length; and the derivative along the window's spend from the
  Hill's slope in closed form. Over the whole series, weeks 10 to 20 and the last four weeks, the
  cost per KPI and the marginal cost match it to `1e-9`, the marginal cost above the average on
  each. At 35 an order the return is `35 / cost_per_kpi` to `1e-12`, and the marginal return
  `35 / marginal_cost_per_kpi`.
- *The undefined cases*: a dark window reads a spend and an incremental KPI of 0 and two
  `Undefined`s that say it spent 0; a coefficient of 0 reads no incremental KPI, and a negative one
  a loss, each named.
- *Mutations*: the cost per KPI read off the window's contribution, the marginal cost off the
  steady-state marginal return, and an infinity returned for a zero return each fail a test.
- `Costs` and `Undefined` are saved and loaded (ADR 0062).

## Alternatives considered

- **A CPA as the spend over the conversions the decomposition attributes to the window.**
  Rejected for ADR 0016's reason: a window ending at the panel's last period loses the carryover
  its spend caused, so two windows' CPAs do not compare.
- **NaN or infinity for an undefined cost.** Rejected: a NaN sorts and averages silently, and an
  infinity reads as a very bad channel where the channel may have spent nothing.
- **Revenue per KPI on the panel**, one rate a geo and period as Meridian takes it. Not now: the
  returns, the plan's table and the scenarios take the rate as an argument, one a channel where
  needed, and no engine reads it from the data.
