"""Recurring events read off a weekly panel's own records of them: the weeks of the year an event
falls in every year, where the panel's controls mark it with noise.

A retailer's promotions recur: the same clearance, the same cyber week, each year. The panel may
record them only roughly, as a flag that misses some promoted weeks and marks some others, or as a
price that drops in them. One coefficient on such a flag fits one average lift to events whose
lifts differ, and a channel that spends into an event takes the rest of its lift. An event of its
own beside the level (ADR 0069) needs its days; `recurring_events` reads them off the panel.

Each week of the year is one of two classes, an event's or an ordinary week's, the same in every
year. The controls named `indicators`, each turned by its direction so that an event raises it,
are independent given the class: a flag, a control of two values, is a Bernoulli of a rate a
class, and any other a normal of a mean and a variance a class; the weeks of a week of the year
are independent given its class. Expectation maximisation fits the two classes, and a week of the
year whose posterior is above one half is an event's. CHOICE: a flag is read as a Bernoulli and
not a normal: a class whose weeks all hold the flag's one value would have a normal of no width,
and the flag alone would then decide each week whatever the other indicators say. The two
classes are read only where they beat one class by the Bayesian information criterion; one class
reads no event. Each run of consecutive event weeks, the year's last week next to its first, is
one event. CHOICE: the KPI is not an indicator, so that a channel's own bursts, which recur too,
are never read as events.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from mixdyn.data.panel import NATIONAL, Cadence

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mixdyn.data.panel import Geo, MediaPanel

WEEKS = 52
"""The weeks of a year a recurring event is read on: ISO 8601's, its 53rd week read as its 52nd."""
FLOOR = 1e-2
"""The least variance of a class's normal indicator, in the indicator's standard units: a class
whose weeks hold one value keeps a density."""
ITERATIONS = 1000
TOLERANCE = 1e-10


@dataclass(frozen=True, slots=True)
class RecurringEvents:
    """The events `recurring_events` reads: each one's `weeks` of the year, a run of consecutive
    weeks from 1 to 52, named in `names`; each week of the year's `posterior` probability of the
    event class, `(52,)`, the prior's where the panel holds no week of it; and `separation`, the
    one class's Bayesian information criterion less the two classes', above 0 where two read
    better and the events are read."""

    names: tuple[str, ...]
    weeks: tuple[tuple[int, ...], ...]
    posterior: npt.NDArray[np.float64]
    separation: float

    def columns(self, dates: npt.ArrayLike) -> dict[str, npt.NDArray[np.float64]]:
        """Each event's column over the weekly `dates`, `(dates,)`: 1 in each week whose week of
        the year is one of the event's, and 0 elsewhere; a calendar's weeks or those past it."""
        of_year = week_of_year(dates)
        return {
            name: np.isin(of_year, weeks).astype(float)
            for name, weeks in zip(self.names, self.weeks, strict=True)
        }

    def days(self, dates: npt.ArrayLike) -> dict[str, npt.NDArray[np.datetime64]]:
        """Each event's days over the weekly `dates`, as `MediaPanel.events` holds them: the label
        of each week of the event."""
        labels = np.asarray(dates, dtype="datetime64[D]")
        return {name: labels[column > 0.0] for name, column in self.columns(labels).items()}


def week_of_year(dates: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Each date's ISO 8601 week, from 1 to 52, its 53rd week read as its 52nd."""
    days = np.asarray(dates, dtype="datetime64[D]").astype(np.int64)
    # 1970-01-01 was a Thursday: the ISO week's Thursday decides its year
    thursday = days - (days + 3) % 7 + 3
    year_start = (
        (thursday.astype("datetime64[D]").astype("datetime64[Y]")).astype("datetime64[D]")
    ).astype(np.int64)
    return np.minimum((thursday - year_start) // 7 + 1, WEEKS)


def recurring_events(
    panel: MediaPanel,
    indicators: Mapping[str, int],
    *,
    geo: Geo = NATIONAL,
    prefix: str = "recurring",
) -> RecurringEvents:
    """The events that recur in `panel`'s weeks of the year, read off its controls `indicators`,
    each with its direction, 1 where an event raises it and -1 where it lowers it; see the module.
    An event is named `prefix` and its weeks, as `"recurring weeks 2-4"`.

    Raises:
        ValueError: on a panel that is not weekly, a geo it does not have, no indicator, a name that
            is not one of its controls, a direction that is not 1 or -1, an indicator that does not
            vary over the weeks observed, or fewer observed weeks than a year's.
    """
    problems = []
    if panel.cadence is not Cadence.WEEK:
        problems.append(f"recurring events are read on a weekly panel, not a {panel.cadence.value}")
    if geo not in panel.geos:
        problems.append(f"the panel has no geo {geo!r}; it has {list(panel.geos)}")
    if not indicators:
        problems.append("recurring events need at least one indicator")
    problems += [f"{n!r} is not one of the controls" for n in indicators if n not in panel.controls]
    problems += [
        f"indicator {n!r} has direction {d!r}, not 1 or -1"
        for n, d in indicators.items()
        if isinstance(d, bool) or d not in (1, -1)
    ]
    if problems:
        raise ValueError("; ".join(problems))
    g = panel.geos.index(geo)
    observed = ~panel.filled[g]
    names = list(indicators)
    raw = panel.control_values[g][observed][:, [panel.controls.index(n) for n in names]]
    directed = raw * np.array([indicators[n] for n in names], dtype=float)
    spread = directed.std(axis=0)
    if observed.sum() < WEEKS or not np.all(spread > 0.0):
        spreads = dict(zip(names, spread.tolist(), strict=True))
        msg = (
            f"recurring events need a year of observed weeks and indicators that vary, got "
            f"{int(observed.sum())} weeks and spreads {spreads}"
        )
        raise ValueError(msg)
    binary = np.array([np.unique(column).size == 2 for column in directed.T])
    x = np.where(
        binary,
        directed == directed.max(axis=0),
        (directed - directed.mean(axis=0)) / spread,
    ).astype(float)
    slot = week_of_year(panel.dates[observed]) - 1
    posterior, two = _two_classes(x, slot, binary)
    one = float(_density(x, _fitted(x, np.ones(len(x)), binary), binary).sum())
    count = len(x)
    free = int(np.sum(np.where(binary, 1, 2)))
    separation = (-2.0 * one + free * math.log(count)) - (
        -2.0 * two + (2 * free + 1) * math.log(count)
    )
    runs = _runs(np.flatnonzero(posterior > 0.5) + 1) if separation > 0.0 else []
    return RecurringEvents(
        names=tuple(
            f"{prefix} week {r[0]}" if len(r) == 1 else f"{prefix} weeks {r[0]}-{r[-1]}"
            for r in runs
        ),
        weeks=tuple(runs),
        posterior=posterior,
        separation=float(separation),
    )


type _Fitted = tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]


def _fitted(
    x: npt.NDArray[np.float64], share: npt.NDArray[np.float64], binary: npt.NDArray
) -> _Fitted:
    """A class's parameters from its weeks' `share`: each indicator's mean and variance, a binary
    one's rate of its higher value drawn towards one half by one week of each value (Laplace's
    rule), so that a class whose weeks all hold one value keeps both possible."""
    total = max(float(share.sum()), 1e-12)
    mean = share @ x / total
    rate = (share @ x + 1.0) / (total + 2.0)
    variance = np.maximum(share @ (x - mean) ** 2 / total, FLOOR)
    return np.where(binary, rate, mean), variance


def _density(x: npt.NDArray[np.float64], fitted: _Fitted, binary: npt.NDArray) -> npt.NDArray:
    """Each week's log density given a class, `(weeks,)`: a Bernoulli's for a binary indicator, a
    normal's for the others, independent across indicators."""
    mean, variance = fitted
    normal = -0.5 * (np.log(2.0 * math.pi * variance) + (x - mean) ** 2 / variance)
    rate = np.clip(mean, 1e-12, 1.0 - 1e-12)
    bernoulli = x * np.log(rate) + (1.0 - x) * np.log1p(-rate)
    return np.sum(np.where(binary, bernoulli, normal), axis=1)


def _two_classes(
    x: npt.NDArray[np.float64], slot: npt.NDArray[np.int64], binary: npt.NDArray
) -> tuple[npt.NDArray[np.float64], float]:
    """Each week of the year's posterior probability of the event class, `(52,)`, and the
    mixture's log-likelihood, fitted by expectation maximisation from the split of the weeks of
    the year by their mean indicator at the midpoint of its range. The event class is the one of
    the larger summed mean."""
    weeks = np.bincount(slot, minlength=WEEKS)
    held = weeks > 0
    score = np.bincount(slot, weights=x.sum(axis=1), minlength=WEEKS)[held] / weeks[held]
    r = np.zeros(WEEKS)
    r[held] = (score > (score.min() + score.max()) / 2.0).astype(float)
    loglik = -math.inf
    for _ in range(ITERATIONS):
        weight = np.clip(r[held].mean(), 1e-6, 1.0 - 1e-6)
        w = r[slot]
        classes = [_fitted(x, share, binary) for share in (1.0 - w, w)]
        per_slot = np.stack(
            [np.bincount(slot, weights=_density(x, c, binary), minlength=WEEKS) for c in classes],
            axis=1,
        ) + np.log([1.0 - weight, weight])
        top = per_slot.max(axis=1, keepdims=True)
        joint = top[:, 0] + np.log(np.exp(per_slot - top).sum(axis=1))
        updated = np.exp(per_slot[:, 1] - joint)
        total = float(joint[held].sum())
        done = np.max(np.abs(updated - r)) < TOLERANCE and total - loglik < TOLERANCE
        r, loglik = updated, total
        if done:
            break
    if classes[0][0].sum() > classes[1][0].sum():
        r = 1.0 - r
    return r, loglik


def _runs(weeks: npt.NDArray[np.int64]) -> list[tuple[int, ...]]:
    """`weeks` of the year, ascending, in runs of consecutive weeks, the year's last next to its
    first."""
    runs: list[list[int]] = []
    for week in weeks.tolist():
        if runs and week == runs[-1][-1] + 1:
            runs[-1].append(week)
        else:
            runs.append([week])
    if len(runs) > 1 and runs[0][0] == 1 and runs[-1][-1] == WEEKS:
        runs[0] = runs.pop() + runs[0]
    return [tuple(r) for r in runs]
