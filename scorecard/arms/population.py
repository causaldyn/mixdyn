"""The dynamic arm's market plan split over a geo export's cells by population (ADR 0083).

`DlmByPopulation`, declared as `dlm-by-population`, one variant, `dlm plug-in, split by
population`: the dynamic arm (`scorecard.arms.dlm.Dlm`) fitted on the geo export's market
(`scorecard.arms.panel.market`), each lift test read in its geos' share of the market, at each
rung as that arm is; its plug-in plan of one weekly spend a channel; then each per-geo channel's
spend split over its geos by population, and a national channel's spend its one cell's (`split`).
Where a geo's share of the channel's spend lies outside its cell's box, the channel's cells are
moved into their boxes by one shift, so the channel's total is still the market plan's.

CHOICE: the market plan says how much each channel spends, not where. The bench allots a market
plan over the geos as the status quo splits each channel (`GeoQuarter.allot`), which keeps the
history's geo split whatever it returns. A curve of the geos' summed spend is the sum's response
exactly where every geo answers through one curve of its spend per head and spends the same per
head: `x_g = p_g x / P` gives `sum_g p_g h(x_g / p_g) = P h(x / P)`, `x` the channel's total and
`P` the market's people. That is the population's split, at which the model over geos's response
of the sum is one channel of the sum (`scorecard.arms.bayes_geo`). On a concave curve that split
is also the one that returns most at the channel's total, and it is the split the bench's equal
split gives a channel (`GeoQuarter.equal_split`). The arm plans the cells as the market model's
reading holds, rather than by the history's split.

The record claims the market plan's gain over the status quo as the dynamic arm reads it: that
model reads a plan through its channels' totals, which the split keeps. Its flags, settings and
effective sample size are the dynamic arm's plug-in's, the settings with the split named. It
carries no forecast and no response: the market's are the dynamic arm's `dlm mixture`'s, and the
bench maps a response of the market alone.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import brentq

from scorecard.arms import panel as reading
from scorecard.arms.dlm import RESPONDS, Dlm

if TYPE_CHECKING:
    from causaldyn_bench.scorecard.observe import Observation

    from scorecard.arms.base import Planned
    from scorecard.arms.dlm import Fitted

PLUG_IN = "dlm plug-in"
VARIANT = "dlm plug-in, split by population"
ROUNDING = 1e-12
"""How far a channel's total may lie outside its cells' boxes summed, relative to the bound:
rounding, as `scorecard.arms.panel.market` divides each cell's bound by its share of the status
quo."""


def into_box(target: np.ndarray, lower: np.ndarray, upper: np.ndarray, total: float) -> np.ndarray:
    """The spends nearest `target` that sum to `total` with each within its box: `target` less one
    shift, clipped to the box, the shift found by bisection.

    Raises:
        ValueError: on a total the boxes cannot hold, beyond rounding.
    """
    low, high = float(np.sum(lower)), float(np.sum(upper))
    slack = ROUNDING * max(abs(low), abs(high))
    if not low - slack <= total <= high + slack:
        msg = f"cells whose boxes sum to [{low}, {high}] cannot spend {total}"
        raise ValueError(msg)
    total = min(max(total, low), high)

    def excess(shift: float) -> float:
        return float(np.clip(target - shift, lower, upper).sum() - total)

    start, end = float(np.min(target - upper)), float(np.max(target - lower))
    if start == end:
        return np.clip(target - start, lower, upper)
    shift = brentq(
        excess, start, end, xtol=1e-12 * max(abs(total), 1.0), rtol=4 * np.finfo(float).eps
    )
    return np.clip(target - shift, lower, upper)


def split(observation: Observation, weekly: np.ndarray) -> np.ndarray:
    """A plan of `observation`'s channels, `weekly`, one spend a channel, as a plan of its cells
    in `reading.cells`' order: a per-geo channel's spend over its geos by population, each moved
    into its box at the channel's spend (`into_box`); a national channel's its one cell's.

    Raises:
        ValueError: on an export with no geo axis, or a channel's spend its cells cannot hold.
    """
    if observation.population is None:
        msg = "a plan is split over a geo export's cells, and this export has no geo axis"
        raise ValueError(msg)
    people = np.asarray(observation.population, dtype=float)
    lower = np.asarray(observation.lower, dtype=float)
    upper = np.asarray(observation.upper, dtype=float)
    cells = reading.cells(observation)
    plan = np.empty(len(cells))
    for j, name in enumerate(observation.channels):
        at = np.array([i for i, (_, c) in enumerate(cells) if c == name])
        geos = [cells[i][0] for i in at]
        if geos == [None]:
            plan[at] = float(weekly[j])
            continue
        share = people[geos] / people[geos].sum()
        plan[at] = into_box(float(weekly[j]) * share, lower[at], upper[at], float(weekly[j]))
    return plan


class DlmByPopulation:
    """The arm; see the module."""

    variants = (VARIANT,)
    tests = True
    cells = True

    def fit(self, observation: Observation, *, smoke: bool) -> Fitted:
        if observation.population is None:
            msg = "the arm splits a plan over a geo export's cells, and this export has no geo axis"
            raise ValueError(msg)
        return Dlm().fit(reading.market(observation), smoke=smoke)

    def plan(self, fit: Fitted, variant: str, observation: Observation) -> Planned:
        planned = Dlm().plan(fit, PLUG_IN, reading.market(observation))
        elsewhere = f"the market's is the dynamic arm's {RESPONDS!r}"
        return dataclasses.replace(
            planned,
            weekly=split(observation, np.asarray(planned.weekly, dtype=float)),
            settings=dict(planned.settings) | {"split": "population", "market plan": PLUG_IN},
            absent={"forecast": elsewhere, "response": elsewhere},
        )
