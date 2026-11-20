"""The media panel: a KPI and media spend for every geo and period of one regular calendar.

The calendar is the panel's invariant. `dates` holds every period from the first to the last,
evenly spaced at the panel's cadence, and position `t` of every array belongs to `dates[t]` in
every geo. A period a geo has no row for is a step marked `filled`, never a row left out, so a
period's index is fixed by its date: the same in every geo, and the same in a panel read from data
with a week missing. Trend and season are read off that index, never off a count of rows.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import NewType

import numpy as np
import numpy.typing as npt

from mixdyn.data.issues import Issue

Geo = NewType("Geo", str)
Channel = NewType("Channel", str)

NATIONAL = Geo("national")
"""The geo of a panel read without a geo column."""

_CURRENCY = re.compile(r"[A-Z]{3}")


class Cadence(Enum):
    """The spacing of consecutive periods. A monthly period is labelled by its month's first day."""

    DAY = "day"
    WEEK = "week"
    MONTH = "month"


@dataclass(frozen=True, slots=True)
class Revenue:
    """The KPI is revenue in the panel's currency: a return on spend is a ratio of one currency."""


@dataclass(frozen=True, slots=True)
class Count:
    """The KPI counts events (orders, installs, sign-ups) named by `unit`.

    A return on spend is then a cost per event, not a ratio.
    """

    unit: str


type KpiUnit = Revenue | Count


class Allotment(Enum):
    """How a channel bought nationally is split over the geos, period by period."""

    POPULATION = "population"
    """In proportion to each geo's population."""


_ALLOTTED = 1e-9
"""How far, relative to a period's national spend, a geo's share may sit from its allotment."""


class LeverKind(Enum):
    """What a lever is: a column of the panel, not media spend, that moves the KPI."""

    PRICE = "price"
    """A price, or a discount off it."""
    PROMOTION = "promotion"
    """A promotion's depth or intensity, or a flag that one ran."""
    ORGANIC = "organic"
    """Activity bought with no media spend: email sends, organic posts, unpaid search."""


class Baseline(Enum):
    """A lever's counterfactual level read off the panel: the least or the most the lever was
    observed at, over every geo's periods that were not filled."""

    MIN = "min"
    MAX = "max"


@dataclass(frozen=True, slots=True)
class Lever:
    """A control the panel declares a lever: its `kind`, and the level its contribution is
    measured against, `baseline`, the lever's observed least or most or a level given in the
    control's own units.

    A lever enters a model as its control does, with no adstock or curve. Its contribution at a
    period is its coefficient times its departure from the baseline, so the KPI at the baseline is
    the base, and a planner never allocates it: it has no spend.

    Raises:
        ValueError: on a kind that is not a `LeverKind`, or a baseline neither a `Baseline` nor a
            finite number.
    """

    kind: LeverKind
    baseline: Baseline | float = Baseline.MIN

    def __post_init__(self) -> None:
        if not isinstance(self.kind, LeverKind):
            msg = f"a lever's kind is a LeverKind, got {self.kind!r}"
            raise ValueError(msg)
        level = self.baseline
        number = isinstance(level, int | float) and not isinstance(level, bool)
        if not (isinstance(level, Baseline) or (number and math.isfinite(level))):
            msg = f"a lever's baseline is a Baseline or a finite level, got {level!r}"
            raise ValueError(msg)


def calendar(start: np.datetime64, periods: int, cadence: Cadence) -> npt.NDArray[np.datetime64]:
    """The labels of `periods` consecutive periods, the first labelled `start`."""
    steps = np.arange(periods)
    if cadence is Cadence.MONTH:
        return (np.datetime64(start, "M") + np.timedelta64(1, "M") * steps).astype("datetime64[D]")
    spacing = np.timedelta64(7 if cadence is Cadence.WEEK else 1, "D")
    return np.datetime64(start, "D") + spacing * steps


def unit_problems(kpi_unit: KpiUnit, currency: str) -> list[str]:
    """Why the KPI's unit and the spend's currency cannot price a return on spend."""
    problems = []
    if not isinstance(currency, str) or not _CURRENCY.fullmatch(currency):
        problems.append(f"currency {currency!r} is not a three-letter ISO 4217 code such as 'USD'")
    if not isinstance(kpi_unit, Revenue | Count):
        problems.append(f"kpi_unit must be Revenue() or Count(unit), got {kpi_unit!r}")
    elif isinstance(kpi_unit, Count) and not kpi_unit.unit.strip():
        problems.append("a Count KPI needs the name of what it counts, such as Count('orders')")
    return problems


def hierarchy_problems(geos: tuple[Geo, ...], parent: Mapping[Geo, Geo]) -> list[str]:
    """Why `parent` is not one tree whose leaves are exactly `geos`. An empty mapping is no tree."""
    if not parent:
        return []
    problems = [f"geo {g!r} has no parent in the hierarchy" for g in geos if g not in parent]
    children: defaultdict[str, list[str]] = defaultdict(list)
    for child, up in parent.items():
        children[up].append(child)
    problems += [
        f"geo {g!r} is in the data and is the parent of {sorted(children[g])}: its rows would "
        "count theirs twice"
        for g in geos
        if g in children
    ]
    members = set(geos)
    problems += [
        f"the hierarchy names {node!r}, which is neither a geo in the data nor anyone's parent"
        for node in parent
        if node not in members and node not in children
    ]
    cycles: set[frozenset[str]] = set()
    for start in parent:
        path = [start]
        node = start
        while node in parent:
            node = parent[node]
            if node in path:
                loop = path[path.index(node) :]
                if frozenset(loop) not in cycles:
                    cycles.add(frozenset(loop))
                    problems.append(f"the hierarchy has a cycle: {' -> '.join([*loop, node])}")
                break
            path.append(node)
    roots = sorted({up for up in parent.values() if up not in parent})
    if len(roots) > 1:
        problems.append(f"the hierarchy has {len(roots)} roots, {roots}: give them one parent")
    return problems


def _read_only[A: np.ndarray](array: A) -> A:
    array.flags.writeable = False
    return array


@dataclass(frozen=True, eq=False, slots=True)
class MediaPanel:
    """A KPI, media spend and controls for every geo and period of a regular calendar.

    Arrays are read-only, indexed by geo, then period, then channel or control:

    - `kpi`, float64 (geo, period): NaN where a period has no observation.
    - `spend`, float64 (geo, period, channel): finite, non-negative, in `currency`.
    - `control_values`, float64 (geo, period, control): finite except on filled steps.
    - `filled`, bool (geo, period): True on a step the input had no row for. Such a step has no
      KPI observation, zero spend in every channel and NaN controls.

    `hierarchy` maps each geo to its parent and, when not empty, is one tree whose leaves are
    exactly `geos`. `population`, float64 (geo,), is each geo's, finite and positive, or None when
    not known. `national` names the channels bought nationally and how each was split over the
    geos: every period, a geo's spend on one is its share of the period's national spend.
    `issues` holds the warnings raised while reading.

    `levers` declares some controls levers (`Lever`): price, promotion or organic activity whose
    contribution is measured against a counterfactual level, `counterfactual`. `events` names
    calendar events, holidays among them, each by its days, every one inside the calendar: an
    event's column (`event_column`) is 1 in each period one of its days falls in, the same in every
    geo, and 0 elsewhere, and a model reads it by the event's name as it reads a control's
    (`regressors`).
    """

    dates: npt.NDArray[np.datetime64]
    cadence: Cadence
    geos: tuple[Geo, ...]
    channels: tuple[Channel, ...]
    controls: tuple[str, ...]
    kpi: npt.NDArray[np.float64]
    kpi_unit: KpiUnit
    spend: npt.NDArray[np.float64]
    currency: str
    control_values: npt.NDArray[np.float64]
    filled: npt.NDArray[np.bool_]
    hierarchy: Mapping[Geo, Geo] = field(default_factory=lambda: MappingProxyType({}))
    population: npt.NDArray[np.float64] | None = None
    national: Mapping[Channel, Allotment] = field(default_factory=lambda: MappingProxyType({}))
    issues: tuple[Issue, ...] = ()
    levers: Mapping[str, Lever] = field(default_factory=lambda: MappingProxyType({}))
    events: Mapping[str, npt.NDArray[np.datetime64]] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        dates = np.array(self.dates, dtype="datetime64[D]")
        kpi = np.array(self.kpi, dtype=np.float64)
        spend = np.array(self.spend, dtype=np.float64)
        controls = np.array(self.control_values, dtype=np.float64)
        filled = np.array(self.filled, dtype=np.bool_)
        population = None if self.population is None else np.array(self.population, np.float64)
        n_geo, n_channel, n_control = len(self.geos), len(self.channels), len(self.controls)
        problems: list[str] = []
        if dates.ndim != 1 or dates.size == 0:
            problems.append("dates must be a non-empty 1-D array")
        elif not np.array_equal(dates, calendar(dates[0], dates.size, self.cadence)):
            problems.append(f"dates are not one regular {self.cadence.value} calendar")
        n_period = dates.size
        if n_geo == 0:
            problems.append("a panel needs at least one geo")
        for kind, names in (
            ("geo", self.geos),
            ("channel", self.channels),
            ("control", self.controls),
        ):
            if len(set(names)) != len(names):
                problems.append(f"{kind} names repeat: {list(names)}")
            if not all(isinstance(name, str) and name for name in names):
                problems.append(f"{kind} names must be non-empty strings: {list(names)}")
        for name, array, shape in (
            ("kpi", kpi, (n_geo, n_period)),
            ("spend", spend, (n_geo, n_period, n_channel)),
            ("control_values", controls, (n_geo, n_period, n_control)),
            ("filled", filled, (n_geo, n_period)),
        ):
            if array.shape != shape:
                problems.append(f"{name} has shape {array.shape}, expected {shape}")
        if not problems:
            if np.isinf(kpi).any():
                problems.append("kpi is infinite somewhere")
            if not np.isfinite(spend).all() or (spend < 0).any():
                problems.append("spend must be finite and non-negative everywhere")
            if not np.isfinite(controls[~filled]).all():
                problems.append("control_values must be finite on every step that was not filled")
            if not (
                np.isnan(kpi[filled]).all()
                and (spend[filled] == 0).all()
                and np.isnan(controls[filled]).all()
            ):
                problems.append("a filled step must have NaN kpi, zero spend and NaN controls")
        problems += unit_problems(self.kpi_unit, self.currency)
        problems += hierarchy_problems(self.geos, self.hierarchy)
        if population is not None and population.shape != (n_geo,):
            problems.append(f"population has shape {population.shape}, expected ({n_geo},)")
        elif population is not None and not (np.isfinite(population) & (population > 0)).all():
            problems.append("population must be finite and positive in every geo")
        if not problems:
            problems += _national_problems(self, spend, population, dates)
        problems += _lever_problems(self.levers, self.controls)
        taken = (*map(str, self.channels), *self.controls)
        events, event_problems = read_events(self.events, taken, dates, self.cadence)
        problems += event_problems
        if problems:
            raise ValueError("invalid MediaPanel:\n- " + "\n- ".join(problems))
        object.__setattr__(self, "dates", _read_only(dates))
        object.__setattr__(self, "kpi", _read_only(kpi))
        object.__setattr__(self, "spend", _read_only(spend))
        object.__setattr__(self, "control_values", _read_only(controls))
        object.__setattr__(self, "filled", _read_only(filled))
        object.__setattr__(self, "hierarchy", MappingProxyType(dict(self.hierarchy)))
        if population is not None:
            object.__setattr__(self, "population", _read_only(population))
        object.__setattr__(self, "national", MappingProxyType(dict(self.national)))
        object.__setattr__(self, "levers", MappingProxyType(dict(self.levers)))
        object.__setattr__(self, "events", MappingProxyType(events))

    def counterfactual(self, lever: str) -> float:
        """The level `lever`'s contribution is measured against: its `Lever.baseline`, or the least
        or the most of its values over every geo's periods that were not filled.

        Raises:
            ValueError: on a name the panel does not declare a lever.
        """
        if lever not in self.levers:
            msg = f"{lever!r} is not one of the panel's levers, {sorted(self.levers)}"
            raise ValueError(msg)
        level = self.levers[lever].baseline
        if not isinstance(level, Baseline):
            return float(level)
        observed = self.control_values[:, :, self.controls.index(lever)][~self.filled]
        return float(observed.min() if level is Baseline.MIN else observed.max())

    def event_column(self, event: str) -> npt.NDArray[np.float64]:
        """`event`'s column over the calendar, `(period,)`: 1 in each period one of its days falls
        in, the last period labelled on or before the day, and 0 elsewhere.

        Raises:
            ValueError: on a name that is not one of the panel's events.
        """
        if event not in self.events:
            msg = f"{event!r} is not one of the panel's events, {sorted(self.events)}"
            raise ValueError(msg)
        column = np.zeros(self.dates.size)
        column[np.searchsorted(self.dates, self.events[event], side="right") - 1] = 1.0
        return column

    def regressors(
        self,
        names: Sequence[str],
        geo: Geo,
        periods: npt.NDArray[np.bool_] | None = None,
    ) -> npt.NDArray[np.float64]:
        """The values of the controls and events `names` in `geo` at the periods `periods` marks,
        every period by default, `(periods, names)`: a control's as the panel holds them, an
        event's its column, the same in every geo.

        Raises:
            ValueError: on a geo the panel does not have, or a name that is neither a control nor
                an event of the panel.
        """
        if geo not in self.geos:
            msg = f"the panel has no geo {geo!r}; it has {list(self.geos)}"
            raise ValueError(msg)
        unknown = [n for n in names if n not in self.controls and n not in self.events]
        if unknown:
            msg = f"the panel has no control or event named {unknown}"
            raise ValueError(msg)
        rows = slice(None) if periods is None else periods
        values = self.control_values[self.geos.index(geo)][rows]
        if all(n in self.controls for n in names):
            # indexed as the engines indexed their controls before events, so that the columns'
            # memory order, and the bits of every mean and spread over them, stay theirs
            return values[:, [self.controls.index(n) for n in names]]
        return np.column_stack(
            [
                values[:, self.controls.index(n)]
                if n in self.controls
                else self.event_column(n)[rows]
                for n in names
            ]
        )

    def counterfactuals(self, controls: Sequence[str]) -> npt.NDArray[np.float64]:
        """Each of `controls`' counterfactual level, `(controls,)`: a lever's `counterfactual`, and
        0 for a control that is not a lever, whose contribution is measured against nothing."""
        return np.array([self.counterfactual(c) if c in self.levers else 0.0 for c in controls])


def refuse_levers(panel: MediaPanel, columns: Sequence[str], engine: str) -> None:
    """Refuse a model that would read some of the panel's levers against 0.

    Raises:
        ValueError: where some of `columns` are levers of `panel`.
    """
    levers = [c for c in columns if c in panel.levers]
    if levers:
        msg = (
            f"{engine} reads a control against 0, and the panel measures {levers} against a "
            "counterfactual level: fit them with the ridge, the one-geo Bayesian model or the DLM, "
            "or read the panel without declaring them levers"
        )
        raise ValueError(msg)


def _lever_problems(levers: Mapping[str, Lever], controls: tuple[str, ...]) -> list[str]:
    """Why `levers` do not each declare one of the panel's controls a lever."""
    problems = [
        f"lever {name!r} is not one of the panel's controls, {list(controls)}"
        for name in levers
        if name not in controls
    ]
    problems += [
        f"lever {name!r}: {lever!r} is not a Lever"
        for name, lever in levers.items()
        if not isinstance(lever, Lever)
    ]
    return problems


def read_events(
    events: Mapping[str, npt.ArrayLike],
    taken: Sequence[str],
    dates: npt.NDArray[np.datetime64],
    cadence: Cadence,
) -> tuple[dict[str, npt.NDArray[np.datetime64]], list[str]]:
    """Each event's days, sorted and distinct, and why some event is not one a panel of `dates` can
    hold: a name that is empty or one of `taken`, the channels' and the controls', or days that
    are not days inside the calendar, from its first label to the day before the period after its
    last."""
    out: dict[str, npt.NDArray[np.datetime64]] = {}
    problems: list[str] = []
    for name, given in events.items():
        if not isinstance(name, str) or not name or name in taken:
            problems.append(f"event {name!r} needs a name of its own, no channel's or control's")
            continue
        try:
            days = np.unique(np.asarray(given, dtype="datetime64[D]").reshape(-1))
        except (TypeError, ValueError):
            problems.append(f"event {name!r}: {given!r} are not dates")
            continue
        if days.size == 0 or np.isnat(days).any():
            problems.append(f"event {name!r} needs at least one day, and no missing one")
            continue
        if dates.ndim != 1 or dates.size == 0:
            continue
        end = calendar(dates[-1], 2, cadence)[1]
        outside = days[(days < dates[0]) | (days >= end)]
        if outside.size:
            listed = ", ".join(str(d) for d in outside[:3])
            problems.append(
                f"event {name!r} has {outside.size} days outside the calendar, {dates[0]} to the "
                f"day before {end}: {listed}"
            )
            continue
        out[name] = _read_only(days)
    return out, problems


def _national_problems(
    panel: MediaPanel,
    spend: npt.NDArray[np.float64],
    population: npt.NDArray[np.float64] | None,
    dates: npt.NDArray[np.datetime64],
) -> list[str]:
    """Why a channel the panel says was bought nationally is not split as it says."""
    problems = []
    for channel, allotment in panel.national.items():
        if channel not in panel.channels:
            problems.append(f"national channel {channel!r} is not one of {list(panel.channels)}")
            continue
        if not isinstance(allotment, Allotment):
            problems.append(f"national channel {channel!r}: {allotment!r} is not an Allotment")
            continue
        if population is None:
            problems.append(
                f"national channel {channel!r} is split by a population the panel lacks"
            )
            continue
        parts = spend[:, :, panel.channels.index(channel)]
        total = parts.sum(axis=0)
        share = population / population.sum()
        off = np.abs(parts - share[:, None] * total).max(axis=0) > _ALLOTTED * total
        if off.any():
            listed = ", ".join(str(d) for d in dates[off][:3])
            problems.append(
                f"national channel {channel!r} is not split by population in {int(off.sum())} "
                f"periods, from {listed}"
            )
    return problems
