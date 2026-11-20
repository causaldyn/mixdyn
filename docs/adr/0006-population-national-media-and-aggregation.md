# 0006. Population, media bought nationally, and the panel at a cut of its hierarchy

Status: accepted, 2026-10-02; amended 2026-10-07: what a population split leaves to identify a
national channel (Consequences).

## Context

ADR 0002 left three things out of the panel until an engine read them: population, the flag that
a channel is bought nationally, and exposure. A model over geos needs the first two:
- a geo's noise and its media scale with its people;
- a national channel's spend in a geo is a share of one national purchase, so its split is an
  assumption that every effect per geo inherits.

A country is also read at several levels, DMAs, regions and the whole, and the levels must add
up: a region's KPI and spend are its DMAs'.

## Decision

- **`MediaPanel.population`**, float64 (geo,), finite and positive, or None. It is one number a
  geo, not a series: populations move over years, and an MMM's window is two or three of them.
  `read_panel(population=...)` maps names to numbers. A geo without one is an error; a name not in
  the data is dropped with a warning, as the hierarchy's are.
- **`MediaPanel.national`** maps each channel bought nationally to its `Allotment`. The panel holds
  the split as an invariant: every period, each geo's spend on the channel is its share of the
  national spend, to `1e-9` of it. A panel that claims a split its spend does not follow is
  refused. A result per geo can state the split, because the panel cannot carry a wrong one.
- **`allot_national(panel, channel, spend, by=Allotment.POPULATION)`** adds a channel from its
  national spend a period. It refuses national spend in a period some geo has no row for: the
  panel holds such a step at zero spend (ADR 0002), and the national purchase reached that geo
  regardless. `Allotment` has one member, `POPULATION`. Exposure arrives with an exposure field.
- **`aggregate(panel, nodes, controls)`** reads the panel at a cut of its hierarchy: every geo sits
  at or below exactly one node.
  - A node's KPI, spend and population are its geos' sums. Its KPI is missing in a period any of
    its geos misses, since a partial sum is not the node's KPI. Its step is filled where all of
    theirs are.
  - Controls combine by a rule each, `Combine.SUM` or `Combine.POPULATION_MEAN`, with no default:
    a temperature does not add and a count does not average. Controls cannot be combined in a
    period where only some of a node's geos have a row, and that is refused.
  - National channels stay national, since a sum of population shares is the node's share. The
    hierarchy keeps the part above the cut.
- **Per-capita scaling is the model's, not the panel's.** A model over geos divides by population
  and reports in totals. A panel per head would need a unit per head and a currency per head, and
  everything downstream would have to tell the two apart.

## Consequences

- `tests/test_geos.py` holds:
  - a national split against each geo's share, and its sum against the national spend;
  - a node's KPI, spend, population and controls against its geos';
  - a geo as its own node, and the root as the whole country with no hierarchy left;
  - a missing KPI and a filled step at the node;
  - a national channel still split by population at the regions;
  - a property over 60 panels with random filled steps: two steps up the hierarchy are one, to
    `1e-13`, and the root's spend is every geo's;
  - the refusals, and populations read in the panel's order with a stray dropped.
- A document holds a panel's population and its national split (`tests/test_documents.py`).
- **A population split adds no information across geos** (amended 2026-10-07). Split by population,
  a national channel's spend per head is the same in every geo, period by period. The geos then
  hold no contrast in its spend:
  - its effect is read from its movement over the periods, as the one-geo model reads it, and a
    geo's own factor (ADR 0013) from how that geo's KPI follows the same series;
  - across the geos in one period, what the split leaves of the channel is each geo's factor
    times the same saturated spend, beside the geo's base and its noise. Nothing in its spend tells
    its effect from anything else that moves every geo with the national purchase: a season, a
    trend, or whatever moved the buyer.

  So the model over geos identifies a national channel's effect no better than the model of the
  whole country does. What it adds is the effect's spread over the geos, under the prior's
  pooling. An allotment that varies over the geos, by exposure, would add the contrast; it stays out
  until a user has exposures.
- Each of 19 mutations of the population, the split and the aggregation fails one of those tests:
  a node's KPI summed over the geos that report it, a mean not weighted by population, nested
  nodes, a split even across geos, a split checked loosely, a population read in the mapping's
  order, and the rest.

## Alternatives considered

- **A population series a period.** It is more general, and no engine reads one; a series can
  replace the vector when a source of one exists.
- **National spend as a row of its own, a pseudo-geo.** Every engine would then have to know that
  one geo's spend belongs to the others, and the KPI of that row would mean nothing.
- **A default rule for controls.** Either default is wrong for half of them, silently.
- **Sum a node's KPI over the geos that report it.** That is a partial sum labelled as the node's.
