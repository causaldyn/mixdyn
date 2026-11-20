# 0071. A posterior's plan for a target return

Status: accepted, 2026-11-06; amends ADR 0019 and ADR 0034: `plan_posterior` takes a target
marginal return or a target return on spend in the budget's place, and its search plans a total
within a stretch less a charge a unit, the fixed budget its case of a point and no charge

## Context

Every rival's optimiser has a mode in which the budget is not given and a target return sets it:
Meridian's flexible budget, to a target total ROI or a target marginal ROI, with per-channel
bounds of 30 % either side of the historical share by default for a fixed budget and 100 % for a
flexible one; Robyn's target efficiency; PyMC-Marketing's target response. A buyer asks for it
first after the fixed budget.

`plan_geos` (ADR 0005) takes CHC's three goals, `ReturnTarget`, `MarginalReturnTarget` and
`ReturnOnSpendTarget`, for one response a cell: the plug-in plan, at the posterior's medians.
The plans a posterior makes, for its expected return (ADR 0019, 0034), its worst share (ADR 0029)
and its least gain over effects off by a factor, took a budget only. No test held any of them
against a rival's optimiser.

## Decision

- **`plan_posterior(panel, geo, readings, budget, periods, limits)` takes
  `MarginalReturnTarget(per_unit)` or `ReturnOnSpendTarget(per_unit)` in the budget's place** and
  returns the `PosteriorPlan` of the budget that meets it, which `budget` then holds. A
  `ReturnTarget` is refused with a `TypeError`: `plan_geos` meets it, and no rival's posterior
  plan asks for it. A goal's value must be finite.
- **The marginal target** is the plan of the most expected worth less `per_unit` a currency unit
  spent, the budget free. That is separable in the channels and linear between a grid's points, so
  each channel's plan is the point of its grid where its mean less the charge is largest, the
  least on a tie: where the slope of its mean's interpolant crosses `per_unit`, or an end of its
  range. It is exact on the interpolants whatever their shape.
- **The target return on spend** is the most budget whose plan's expected gain is `per_unit` a
  currency unit or more, the gain against spending nothing in the plan, as `PosteriorPlan.gain`
  reads it. Where the ranges' most meets it, that is the plan; where the plan of the most gain less
  `per_unit` a unit does not, no budget does, and the plan is refused, naming that peak and its
  budget. Otherwise the stretch from the peak's total to the ranges' most is halved: its upper half
  is searched for the plan of the most worth less the charge with its total in the half; where
  that plan meets the target the stretch starts at its total, else it is the lower half; the
  search ends when the stretch is under `1e-12` of the ranges' most. Every total past the stretch
  was shown short, so on S-shaped means, whose gain less the charge can cross 0 more than once,
  the budget found is the last crossing, as CHC's `budget_for` finds it.
- **One search for both.** ADR 0034's branch and bound plans a total within a stretch
  `[least, most]` less a charge a unit of the total: a box's bound is the fill down its envelopes
  at the box's own best total, where their slopes fall below the charge, clipped into the stretch
  and the box's totals, less the charge on it, the most any split in the box gains less the charge,
  since the envelopes' worth is concave in the total. A fixed budget is the stretch of one point
  and no charge, and its plan is the same, bit for bit: on 3 000 cases of two to four channels,
  grids of 5 to 129 points and Hill means of slope 0.5 to 5, the plans before and after this
  change are equal in every bit.
- **Constraints** are each channel's range, a `Range` a period: an inequality, or an equality where
  the range is a point, which the plan holds under either goal. `plan_geos` keeps its own: each
  cell's range, and totals over a geo or a channel, an equality where a total's range is a point.
- `plan_cvar` and `plan_robust` plan a fixed budget, as before.

## Consequences

- *Against Meridian's optimiser on Meridian's own curves.* Family 12 of the bench draws media in
  Meridian 2.1's form at its default priors. On its first eight pilot worlds
  (`tests/fixtures/oracles/family12_curves.json`, written by `scripts/family12_curves.py`), each
  channel bounded 30 % either side of its share of the quarter's budget, Meridian's default for a
  fixed budget, three scenarios: the quarter's budget; the median channel's marginal return at the
  status quo as the target marginal return; and the best plan's return on spend at 1.15 times the
  budget as the target return on spend. Meridian is not installed: its rule is reproduced in the
  test from its documentation (bounds, its grid's step from `gtol`, the hill-climbing search and
  its three stops), on the same channels. Measured, both `plan_geos` and `plan_posterior` (the
  world's curves as its one draw, a grid of 2048) are within `5e-4` of Meridian's plan in gain and
  in budget in every case; the test holds them to 1 %. Meridian's own plan is within `5e-4` of the
  exact one. Against an exact solver of the KKT conditions written in the test, `plan_geos` is the
  exact plan to `1e-13`; `plan_posterior`'s budget plan is within `3.2e-9` of the best gain and its
  return on spend's budget within `2.8e-9` of the exact one, held to `1e-8`; its marginal plan is
  on a point of each grid, within one step of the exact spend.
- *The target marginal return's plan stops where each channel's marginal return crosses the
  target*: on the same worlds, every channel inside its range has a marginal return at the target,
  `plan_geos` within `1e-9` of it and `plan_posterior` within the change in the channel's marginal
  return over one grid step either side; a channel at its least returns no more than the target
  there, one at its most no less; and some channel is inside its range on every world.
- *On S-shaped means*, 300 cases each of two or three channels of Hill means of slope 0.5 to 5:
  the target return on spend's total is a brute force's over the interpolants to `1e-9` of the
  ranges' most, and a charged box's bound holds every split in the box, the search's plan within
  `2e-9` of a brute force's best.
- The plan's gain is read on the means, not their interpolants. Where a mean is concave the
  interpolant is below it and the plan's return on spend is at least the target; where a mean is
  convex near the plan, the return read may fall short of the target by the interpolant's error.
- The search for a return on spend runs about 40 plans, each one fill a box on concave means.

## Alternatives considered

- **CHC's `budget_for` on the medians' channels**: `plan_geos` does that already; it is the
  plug-in, not the decision for the posterior's expected return (ADR 0019).
- **Bisection on the budget with the fixed-budget planner**: exact where the gain less the charge
  crosses 0 once past its peak, as on concave means; on S-shaped means it can stop at a crossing
  before a later one. Rejected for the halving search, which reads every budget past the stretch.
- **Meridian's hill climb on a grid**: the reference in the test; its plan is a grid step from the
  exact one on concave curves, and a greedy step can stop short where a curve is S-shaped.
- **A target return for `plan_cvar`**: the conditional value at risk of a plan's gain a currency
  unit is a new objective, not CHC's `cvar_allocate`'s. Left until a buyer asks for it.
