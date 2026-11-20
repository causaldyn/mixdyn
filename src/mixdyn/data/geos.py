"""Media bought nationally, and the panel at the nodes of its geo hierarchy.

`allot_national` adds a channel bought for the whole country, its spend each period split over the
geos by population, and the panel records the split so that a result per geo can state it.
`aggregate` reads the panel at any cut of its hierarchy, a region or the country: each node's KPI,
spend and population are the sums of the geos below it, so the levels add up by construction.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import replace
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from mixdyn.data.panel import Allotment, Channel, Geo, MediaPanel

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy.typing as npt

log = logging.getLogger(__name__)


class Combine(Enum):
    """How the controls of the geos below a node make the node's."""

    SUM = "sum"
    """A count or an amount, which adds up."""
    POPULATION_MEAN = "population_mean"
    """A rate or a level, such as a price or a temperature: the mean weighted by population."""


def allot_national(
    panel: MediaPanel,
    channel: str,
    spend: npt.ArrayLike,
    by: Allotment = Allotment.POPULATION,
) -> MediaPanel:
    """The panel with `channel` added, bought nationally: each period's national `spend` split
    over the geos `by` their population.

    Raises:
        ValueError: on a channel the panel has, national spend that is not one finite,
            non-negative value a period, a panel with no population, or national spend in a period
            some geo has no row for, which the panel holds at zero spend.
    """
    if channel in panel.channels:
        msg = f"the panel already has a channel {channel!r}"
        raise ValueError(msg)
    total = np.asarray(spend, dtype=np.float64)
    if total.shape != panel.dates.shape:
        msg = f"national spend on {channel!r} has shape {total.shape}, expected {panel.dates.shape}"
        raise ValueError(msg)
    if not (np.isfinite(total) & (total >= 0)).all():
        msg = f"national spend on {channel!r} must be finite and non-negative in every period"
        raise ValueError(msg)
    if panel.population is None:
        msg = f"{channel!r} is split by population, and the panel has none"
        raise ValueError(msg)
    unread = panel.filled & (total > 0)[None, :]
    if unread.any():
        g, t = np.nonzero(unread)
        where = ", ".join(
            f"{panel.geos[i]} {panel.dates[j]}" for i, j in zip(g[:3], t[:3], strict=True)
        )
        msg = (
            f"national spend on {channel!r} falls in {int(unread.sum())} periods a geo has no row "
            f"for, from {where}: the panel holds those at zero spend"
        )
        raise ValueError(msg)
    share = panel.population / panel.population.sum()
    added = share[:, None] * total[None, :]
    return replace(
        panel,
        channels=(*panel.channels, Channel(channel)),
        spend=np.concatenate([panel.spend, added[:, :, None]], axis=2),
        national={**panel.national, Channel(channel): by},
    )


def aggregate(
    panel: MediaPanel,
    nodes: Sequence[str],
    controls: Mapping[str, Combine] | None = None,
) -> MediaPanel:
    """The panel at `nodes` of its hierarchy, each node a geo of the result.

    `nodes` must cut the hierarchy: every geo sits at or below exactly one of them. A node's KPI,
    spend and population are the sums of its geos'; its KPI is missing in a period any of them
    misses, and its step is filled where all of theirs are. Each control combines as `controls`
    says. Channels bought nationally stay so, split as before. The hierarchy keeps the nodes above
    the cut.

    Raises:
        ValueError: on nodes that do not cut the hierarchy, a control with no rule or a rule for
            no control, or a population mean without a population.
    """
    rules = dict(controls or {})
    unruled = [c for c in panel.controls if c not in rules]
    extra = sorted(set(rules) - set(panel.controls))
    if unruled or extra:
        msg = f"controls need one rule each: none for {unruled}, and rules for no control {extra}"
        raise ValueError(msg)
    means = [k for k, c in enumerate(panel.controls) if rules[c] is Combine.POPULATION_MEAN]
    if means and panel.population is None:
        msg = "a population mean of the controls needs the panel's population"
        raise ValueError(msg)
    members = _cut(panel, [Geo(str(n)) for n in nodes])

    kpi, spend, controls_out, filled, population = [], [], [], [], []
    for node, below in members.items():
        rows = [panel.geos.index(g) for g in below]
        some = panel.filled[rows].any(axis=0) & ~panel.filled[rows].all(axis=0)
        if panel.controls and some.any():
            listed = ", ".join(str(d) for d in panel.dates[some][:3])
            msg = (
                f"node {node!r}: some of its geos have no row in {int(some.sum())} periods, from "
                f"{listed}, so its controls there are not known"
            )
            raise ValueError(msg)
        kpi.append(panel.kpi[rows].sum(axis=0))
        spend.append(panel.spend[rows].sum(axis=0))
        filled.append(panel.filled[rows].all(axis=0))
        combined = panel.control_values[rows].sum(axis=0)
        if panel.population is not None:
            weights = panel.population[rows]
            population.append(weights.sum())
            mean = np.einsum("g,gtk->tk", weights, panel.control_values[rows]) / weights.sum()
            combined[:, means] = mean[:, means]
        controls_out.append(combined)
        lost = int((np.isnan(kpi[-1]) & ~np.isnan(panel.kpi[rows]).all(axis=0)).sum())
        if lost:
            log.warning(
                "node %r: its KPI is missing in %d periods that only some of its %d geos miss",
                node,
                lost,
                len(rows),
            )
    filled_grid = np.array(filled)
    control_grid = np.array(controls_out)
    return replace(
        panel,
        geos=tuple(members),
        kpi=np.array(kpi),
        spend=np.array(spend),
        control_values=control_grid,
        filled=filled_grid,
        hierarchy=_above(panel.hierarchy, tuple(members)),
        population=None if panel.population is None else np.array(population),
    )


def _cut(panel: MediaPanel, nodes: list[Geo]) -> dict[Geo, list[Geo]]:
    """Each node's geos, in the panel's order, when every geo sits under exactly one node."""
    if len(set(nodes)) < len(nodes) or not nodes:
        msg = f"nodes must be distinct and at least one, got {nodes}"
        raise ValueError(msg)
    known = set(panel.geos) | set(panel.hierarchy) | set(panel.hierarchy.values())
    unknown = [n for n in nodes if n not in known]
    if unknown:
        msg = f"{unknown} are neither geos of the panel nor nodes of its hierarchy"
        raise ValueError(msg)
    members: defaultdict[Geo, list[Geo]] = defaultdict(list)
    for geo in panel.geos:
        path = [geo]
        while path[-1] in panel.hierarchy:
            path.append(panel.hierarchy[path[-1]])
        above = [n for n in path if n in nodes]
        if len(above) != 1:
            msg = (
                f"nodes must cut the hierarchy, and geo {geo!r} sits under {above or 'none'} of "
                f"them, along {' -> '.join(path)}"
            )
            raise ValueError(msg)
        members[above[0]].append(geo)
    return {n: members[n] for n in nodes}


def _above(hierarchy: Mapping[Geo, Geo], nodes: tuple[Geo, ...]) -> dict[Geo, Geo]:
    """The hierarchy's part from `nodes` up, or none when one node is the whole country."""
    kept: dict[Geo, Geo] = {}
    for node in nodes:
        while node in hierarchy and node not in kept:
            kept[node] = hierarchy[node]
            node = hierarchy[node]
    return kept
