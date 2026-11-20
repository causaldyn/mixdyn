# 0042. A channel bought nationally planned as one spend

Status: accepted, 2026-10-11; amends ADR 0005's limits

## Context

A panel records the channels bought for the whole country (`MediaPanel.national`): each period,
a geo's spend on one is its population share of the national spend (`allot_national`). TV is the
common case.

`plan_geos` planned every geo's channel as a cell of its own, the national channel's cells too.
Its plan could put more TV in one geo and less in another than their population shares. A buyer
cannot place that plan: the channel is bought once, and the market splits it.

CHC 0.16.0 plans such a channel as it is bought (`chc.allocation.National`). The plan chooses one
national spend a period within its own box. Each geo's cell spends its share and reads it on its
own response. The national spend counts whole in the budget and in its channel's total, and by
each geo's share in that geo's total.

## Decision

- **`plan_geos` plans each channel the panel bought nationally as one national spend**, split by
  the panel's population shares, the split `allot_national` and the panel's own check use.
- **Its range is the national spend's, a period, in `GeoLimits.national`.** It needs a finite
  most, as a cell's range does (ADR 0005). Its cells take no range of their own, and a range given
  for one is refused rather than read.
- **Every channel bought nationally needs its range, and only those take one.** A range for a
  channel the panel bought per geo is refused, so the panel stays the one record of how a channel
  is bought.
- **`GeoPlan.spend` keeps one column a channel.** A national channel's column holds each geo's
  share of the national spend, which the column sums to. Its channel price is its national spend's
  over the plan, as CHC reports it.
- **The engine requires `causal-hybrid-control>=0.16.0`**, the first release with `National`.

## Consequences

- A plan for a panel with a national channel is one the buyer can place. Its gain never exceeds
  the plan of the same cells planned apart, each boxed at its share of the national range, because
  the national plan is one of those plans.
- With one geo, a national channel is its cell: the plan equals the cell's plan within its range.
- A caller who planned a national channel's cells one by one now gets an error that names them,
  not a different plan.

## Alternatives considered

- **Keep the cells, each boxed at its share of the national range.** Rejected: the boxes allow
  shares other than the population's, so the plan stays one the buyer cannot place.
- **Shares given in `GeoLimits` rather than read off the panel.** Rejected: the panel already
  records how the channel was split, and two records of one split can disagree.
- **A national channel as a geo of its own.** Rejected: its lift is read on each geo's own response
  and history, which a separate geo does not have.
