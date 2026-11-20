# 0005. Geo plans: CHC's joint program over every geo's channels, by name

Status: accepted, 2026-10-02; amended 2026-10-11 by ADR 0042: a channel the panel bought
nationally is planned as one national spend split by population, its range in
`GeoLimits.national`

## Context

A common way to plan geos and channels is in two steps: split the budget over the regions by their
average return on spend, each capped at a multiple of its current spend, then scale the shares back
to the budget; within a region, split its share over the channels by their average return. Each
step reads averages, not slopes, and the scaling can undo the cap.

`chc.allocation` plans every geo's channels at once (CHC's ADR 0043 and 0044): a budget, or one of
three goals in its place, with each cell's box and totals over geos and over channels. It returns
a price for the budget and for each total, and on concave curves the plan is exact. Its inputs are
arrays in an order the caller keeps.

## Decision

- **`mixdyn.decisions.plan_geos(panel, responses, target, periods, limits)`** hands
  `chc.allocation` one cell for each geo's channel, in the panel's order, and returns a `GeoPlan`
  keyed by name. `target` is a budget over the plan or a goal: `ReturnTarget`,
  `MarginalReturnTarget` or `ReturnOnSpendTarget`, CHC's own types, re-exported.
- **Limits are named.** `GeoLimits.cells` gives each `(geo, channel)` a `Range` a period;
  `GeoLimits.geos` and `GeoLimits.channels` bound what a geo or a channel spends over the plan.
- **Every cell needs a finite most.** A response is read where its spend was seen; a plan past it
  is the curve's extrapolation, and a caller who wants it states how far.
- **The current spend is each cell's mean over the panel's last `periods` observed periods.** A
  filled step is not a week of zero spend. The current spend's gain is computed as the plan's is,
  run through the same history against spending nothing.
- **Double precision is required**, and responses with single-precision parameters are refused. In
  single, CHC's Newton step does not close, and a response built in single keeps its rounded
  parameters once JAX runs in double.
- **CHC's refusals are re-raised** with the geos and channels in the order its message counts them.

## Consequences

- A two-level plan is one plan on the grid, so it never returns more than the joint plan of its
  budget; `tests/test_geo_plan.py` holds a plan made geo by geo to that on drawn worlds.
- What CHC does not do, `plan_geos` does not do: no plan across geos with spillover, and on an
  S-shaped curve the plan is the envelope's, its `shortfall` the bound on the gap.
- The engine's own responses come from its models; `plan_geos` takes any `chc.response.Channel`, so
  a plan can be checked on a world's own curves.

## Alternatives considered

- **The two-level plan.** Rejected: its scaling can break its own cap, and it splits by averages
  where the best plan meets one slope, so it is the best plan of its budget only by coincidence.
- **Arrays in the public API, as CHC takes them.** Rejected: the order of geos and channels is a
  place for two callers to disagree (ADR 0002).
- **Plan past the data by default.** Rejected: a curve fitted on the spend seen says little past it,
  and the plan would put the budget where it says most.
