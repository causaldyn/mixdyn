"""A budget, or a goal in its place, planned over every geo's channels at once, by name.

`plan_geos` hands `chc.allocation` one cell for each geo's channel: its response a
`chc.response.Channel` (carryover, curve and coefficient), its history the panel's spend, and the
limits named by geo and channel, each cell's range a period and what each geo and each channel may
spend over the plan. Every cell's spend is chosen at once, so a plan made in two steps, the budget
split over the geos first and each geo's share over its channels after, never returns more. On
concave curves the plan is exact; on an S-shaped one it is the envelope's, and
`GeoPlan.shortfall` bounds how far it is from the best.

A channel the panel bought nationally (`MediaPanel.national`) is planned as it is bought: one spend
a period, within its own range, split over the geos by population as the panel's was, each geo's
cell reading its share on its own response.

A plan is read against the current spend: each cell's mean a period over the panel's last
`periods` observed periods, run over the plan's periods from the same history. It names in
`unidentified` each cell it moves off its current spend whose channel the panel flags
`IssueCode.CONSTANT` in that geo or in every geo, as `mixdyn.decisions.posterior`'s plans name a
channel: the responses' reading of it rests on no movement in its spend. No number changes.

`chc.allocation` is exact in double precision only. In single, JAX's default, its Newton step does
not close and its plan is exact only to its cutting planes' gap, so a plan is refused unless JAX
runs in double.

`plan_mixture` plans cells whose expected response is a mixture of curves, one a grid point of a
posterior (`mixdyn.models.geo_eb.CellResponse`), which no one `chc.response.Channel` holds. It
water-fills: each cell's slope is read on `GRID_POINTS` spends evenly over its box and made
non-increasing, each point's the least of those at or below its spend; each cell spends where its
slope meets one price, or at a bound, the price bisected `BISECTIONS` times to the budget, and the
two plans that bracket the budget are joined at it, so a cell whose slope is flat at the price
takes the rest. It is NumPy alone, and the grid's arithmetic is the one the model over geos' rule
was frozen with (ADR 0090).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from chc.allocation import (
    MarginalReturnTarget,
    National,
    ReturnOnSpendTarget,
    ReturnTarget,
    Totals,
    allocate_geos,
    budget_for_geos,
)

from mixdyn.decisions.posterior import _unidentified

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    from chc.response import Channel as Response

    from mixdyn.data import Channel, Geo, MediaPanel
    from mixdyn.models.geo_eb import CellResponse

type _Array = npt.NDArray[np.float64]
type Goal = ReturnTarget | MarginalReturnTarget | ReturnOnSpendTarget
"""A goal `plan_geos` meets in place of a budget, one of `chc.allocation`'s."""

GRID_POINTS = 401
"""The spends `plan_mixture` reads each cell's slope at, evenly over its box."""
BISECTIONS = 200
"""The halvings of `plan_mixture`'s price interval."""
FEASIBLE = 1e-12
"""How far a budget a period may lie outside the box's floors' and ceilings' sums, relative to it:
rounding."""


@dataclass(frozen=True, slots=True)
class Range:
    """At least `least` and at most `most`, in the panel's currency; `most` may be infinite.

    Raises:
        ValueError: unless `0 <= least <= most`, with `least` finite.
    """

    least: float = 0.0
    most: float = math.inf

    def __post_init__(self) -> None:
        if not (math.isfinite(self.least) and 0.0 <= self.least <= self.most):
            msg = (
                "a range needs a finite least of 0 or more and a most no smaller: "
                f"got [{self.least}, {self.most}]"
            )
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class GeoLimits:
    """What a plan may spend, by name.

    `cells` holds each geo's channel's range a period, and every cell needs one with a finite most:
    a response is read where it was seen, not past it by default. A channel the panel bought
    nationally has no cell ranges: `national` holds its national spend's range a period, with a
    finite most. `geos` bounds what a geo spends over the plan, its channels together, a national
    channel's share among them, and `channels` what a channel spends over it, its geos together;
    a geo or a channel with no entry is bounded by its cells alone.
    """

    cells: Mapping[tuple[Geo, Channel], Range]
    geos: Mapping[Geo, Range] = field(default_factory=lambda: MappingProxyType({}))
    channels: Mapping[Channel, Range] = field(default_factory=lambda: MappingProxyType({}))
    national: Mapping[Channel, Range] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True, slots=True, eq=False)
class GeoPlan:
    """A plan of every geo's channels, and the current spend it changes.

    `spend` and `current` are `(geo, channel)` in the panel's order, in spend a period: the plan's,
    and each cell's mean over the panel's last `periods` observed periods. A national channel's
    column holds each geo's share of its national spend, which the column sums to.

    `gain` is what the plan's spend adds to the KPI over its periods and the carryover after them,
    and `current_gain` what the current spend adds run the same way, each against spending nothing
    in the plan, so the history's carryover counts in neither. `shortfall` bounds how far `gain` is
    below the best plan's within the limits, 0 on concave curves. `price` is what one more currency
    unit of budget returns; `geo_prices` and `channel_prices` are what one more unit of room in a
    geo's or a channel's total returns, positive where it binds at its most, negative at its least,
    0 where it does not bind. A national channel's price is its national spend's over the plan.
    `unidentified` names each cell, `(geo, channel)` in the panel's order, the plan moves off its
    current spend whose spend the panel flags as never moving in that geo or in every geo
    (`IssueCode.CONSTANT`).
    """

    geos: tuple[Geo, ...]
    channels: tuple[Channel, ...]
    periods: int
    spend: _Array
    current: _Array
    budget: float
    gain: float
    current_gain: float
    shortfall: float
    price: float
    geo_prices: Mapping[Geo, float]
    channel_prices: Mapping[Channel, float]
    unidentified: tuple[tuple[Geo, Channel], ...] = ()

    @property
    def return_on_spend(self) -> float:
        """The gain a currency unit of the budget returns: a ROAS where the KPI is revenue, NaN
        on a budget of nothing."""
        return self.gain / self.budget if self.budget else math.nan


def plan_geos(
    panel: MediaPanel,
    responses: Mapping[Geo, Mapping[Channel, Response]],
    target: float | Goal,
    periods: int,
    limits: GeoLimits,
) -> GeoPlan:
    """The joint plan of every geo's channels over `periods` periods of the panel's cadence that
    spends `target`, a budget over the plan, or meets it, a goal in its place.

    A goal is `chc.allocation`'s: `ReturnTarget(amount)`, the least budget whose plan gains
    `amount`; `MarginalReturnTarget(per_unit)`, the plan at which one more currency unit returns
    `per_unit`; `ReturnOnSpendTarget(per_unit)`, the most budget whose plan returns `per_unit` a
    currency unit on average.

    Raises:
        ValueError: on a response or a cell's range missing for a cell of the panel, or named for
            one it does not have; a cell's range with no finite most; a range for a cell of a
            channel bought nationally; a national range missing for a channel the panel bought
            nationally, named for one it did not, or with no finite most; a limit for a geo or a
            channel the panel does not have; a geo with no observed period among the last
            `periods`; JAX in single precision; and what `chc.allocation` refuses, its geos and
            channels named in the panel's order.
    """
    if not jax.config.x64_enabled:
        msg = (
            "a plan needs JAX in double precision (JAX_ENABLE_X64=1): in single, chc.allocation's "
            "Newton step does not close and its plan is exact only to its cutting planes' gap"
        )
        raise ValueError(msg)
    geos, channels = panel.geos, panel.channels
    cells = _cells(responses, geos, channels)
    lower, upper = _boxes(limits.cells, geos, channels, panel.national.keys())
    history = np.transpose(panel.spend, (1, 0, 2))
    current = _current(panel, periods)
    geo_totals = _totals(limits.geos, geos, "geo")
    channel_totals = _totals(limits.channels, channels, "channel")
    national = _national(panel, limits.national)
    try:
        match target:
            case ReturnTarget() | MarginalReturnTarget() | ReturnOnSpendTarget():
                plan = budget_for_geos(
                    cells,
                    target,
                    periods,
                    lower=lower,
                    upper=upper,
                    geo_totals=geo_totals,
                    channel_totals=channel_totals,
                    national=national,
                    history=history,
                )
            case _:
                plan = allocate_geos(
                    cells,
                    float(target),
                    periods,
                    lower=lower,
                    upper=upper,
                    geo_totals=geo_totals,
                    channel_totals=channel_totals,
                    national=national,
                    history=history,
                )
    except ValueError as error:
        msg = f"{error} (geos in order {list(geos)}; channels {list(channels)})"
        raise ValueError(msg) from error
    return GeoPlan(
        geos=geos,
        channels=channels,
        periods=periods,
        spend=plan.spend,
        current=current,
        budget=plan.budget,
        gain=plan.gain,
        current_gain=_gain(cells, history, current, periods),
        shortfall=plan.bound - plan.worth,
        price=plan.price,
        geo_prices=MappingProxyType(dict(zip(geos, map(float, plan.geo_prices), strict=True))),
        channel_prices=MappingProxyType(
            dict(zip(channels, map(float, plan.channel_prices), strict=True))
        ),
        unidentified=tuple(
            (geo, c)
            for g, geo in enumerate(geos)
            for c in _unidentified(
                panel,
                geo,
                channels,
                np.asarray(plan.spend, dtype=np.float64)[g],
                current[g],
                _cell_most(upper[g], channels, limits.national),
            )
        ),
    )


@dataclass(frozen=True, slots=True, eq=False)
class MixturePlan:
    """`plan_mixture`'s plan: each cell's spend a period, `weekly`, and `price`, the slope each
    cell inside its box spends at."""

    weekly: _Array
    price: float


def plan_mixture(
    cells: Sequence[CellResponse],
    lower: npt.ArrayLike,
    upper: npt.ArrayLike,
    budget: float,
    periods: int,
) -> MixturePlan:
    """The plan of `cells` over `periods` periods that spends `budget` over them, each cell's spend
    a period within `[lower, upper]`, by water-filling on each cell's expected worth (the module).

    Where every cell's expected worth is concave over its box, the plan is the best one, to the
    grid's reading of a slope. Where a cell's is not, as one whose mixture keeps a point with a
    coefficient below 0, the plan still spends the budget within the box, and it is not
    necessarily the best: the cell's slope is read as the least at or below each spend, so the
    plan does not cross a stretch where the slope rises.

    Raises:
        ValueError: on bounds that are not one finite `(cells,)` array each, a floor above its
            ceiling, a budget that is not finite or that no plan in the box spends, or `periods`
            not a whole number of periods, at least 1.
    """
    low = np.asarray(lower, dtype=float)
    high = np.asarray(upper, dtype=float)
    if low.shape != (len(cells),) or high.shape != low.shape:
        msg = f"the bounds are ({len(cells)},) each, a cell's, not {low.shape} and {high.shape}"
        raise ValueError(msg)
    if isinstance(periods, bool) or not isinstance(periods, int) or periods < 1:
        msg = f"periods must be a whole number of periods, at least 1, got {periods!r}"
        raise ValueError(msg)
    rate = float(budget) / float(periods)
    slack = FEASIBLE * abs(rate)
    if (
        not all(np.all(np.isfinite(a)) for a in (low, high))
        or not math.isfinite(rate)
        or np.any(low > high)
        or not low.sum() - slack <= rate <= high.sum() + slack
    ):
        msg = f"no plan in the box [{low.tolist()}, {high.tolist()}] spends {rate} a period"
        raise ValueError(msg)
    grid = low[:, None] + (high - low)[:, None] * np.linspace(0.0, 1.0, GRID_POINTS)[None, :]
    slopes = np.zeros((len(cells), GRID_POINTS))
    for i, cell in enumerate(cells):
        slopes[i] = cell.slope(grid[i], periods)
    slopes = np.minimum.accumulate(slopes, axis=1)

    def spend_at(price: float) -> _Array:
        out = np.empty(len(cells))
        for i, s in enumerate(slopes):
            if s[0] <= price:
                out[i] = grid[i, 0]
            elif s[-1] >= price:
                out[i] = grid[i, -1]
            else:
                out[i] = np.interp(price, s[::-1], grid[i, ::-1])
        return out

    lo, hi = slopes.min() - 1.0, slopes.max() + 1.0
    for _ in range(BISECTIONS):
        mid = 0.5 * (lo + hi)
        if spend_at(mid).sum() > rate:
            lo = mid
        else:
            hi = mid
    above, below = spend_at(lo), spend_at(hi)
    span = above.sum() - below.sum()
    share = 0.0 if span <= 0.0 else (rate - below.sum()) / span
    return MixturePlan(weekly=below + share * (above - below), price=float(0.5 * (lo + hi)))


def _cells(
    responses: Mapping[Geo, Mapping[Channel, Response]],
    geos: Sequence[Geo],
    channels: Sequence[Channel],
) -> tuple[tuple[Response, ...], ...]:
    grid = _grid(geos, channels)
    named = [(g, c) for g, row in responses.items() for c in row]
    missing = sorted(set(grid) - set(named))
    unknown = sorted(set(named) - set(grid))
    if missing or unknown:
        msg = f"the responses miss cells {missing} and name cells the panel does not have {unknown}"
        raise ValueError(msg)
    # a response built in single precision keeps its rounded parameters in double
    single = [
        (g, c)
        for g, c in grid
        if any(
            getattr(leaf, "dtype", None) == jnp.float32 for leaf in jax.tree.leaves(responses[g][c])
        )
    ]
    if single:
        msg = f"the responses of {single} hold single-precision parameters: build them in double"
        raise ValueError(msg)
    return tuple(tuple(responses[g][c] for c in channels) for g in geos)


def _boxes(
    ranges: Mapping[tuple[Geo, Channel], Range],
    geos: Sequence[Geo],
    channels: Sequence[Channel],
    national: Collection[Channel],
) -> tuple[_Array, _Array]:
    """Each cell's least and most a period, `nan` in a national channel's column, whose cells
    `chc.allocation` gives no box of their own."""
    grid = _grid(geos, channels)
    own = [cell for cell in grid if cell[1] not in national]
    missing = [cell for cell in own if cell not in ranges]
    unknown = [cell for cell in ranges if cell not in set(grid)]
    open_ended = [cell for cell in own if cell in ranges and not math.isfinite(ranges[cell].most)]
    if missing or unknown or open_ended:
        msg = (
            f"every cell needs a range a period with a finite most: missing {missing}, no such "
            f"cell {unknown}, no finite most {open_ended}"
        )
        raise ValueError(msg)
    allotted = [cell for cell in grid if cell[1] in national and cell in ranges]
    if allotted:
        msg = (
            f"the cells {allotted} are of a channel bought nationally, whose spend is one range a "
            "period in GeoLimits.national"
        )
        raise ValueError(msg)
    shape = (len(geos), len(channels))
    lower = np.array([ranges[c].least if c in ranges else math.nan for c in grid]).reshape(shape)
    upper = np.array([ranges[c].most if c in ranges else math.nan for c in grid]).reshape(shape)
    return lower, upper


def _national(panel: MediaPanel, ranges: Mapping[Channel, Range]) -> dict[int, National]:
    """Each channel the panel bought nationally, by its column, as `chc.allocation` plans it: one
    spend a period within its range, split by the panel's population shares."""
    bought = panel.national
    missing = [c for c in bought if c not in ranges]
    unknown = [c for c in ranges if c not in bought]
    open_ended = [c for c in bought if c in ranges and not math.isfinite(ranges[c].most)]
    if missing or unknown or open_ended:
        msg = (
            "every channel bought nationally needs a range of its national spend a period with a "
            f"finite most: missing {missing}, not bought nationally {unknown}, no finite most "
            f"{open_ended}"
        )
        raise ValueError(msg)
    population = panel.population
    if population is None:
        # a panel splits a national channel by its population, so one without buys none
        return {}
    shares = population / population.sum()
    return {
        panel.channels.index(c): National(shares, ranges[c].least, ranges[c].most) for c in bought
    }


def _totals[N: str](named: Mapping[N, Range], names: Sequence[N], kind: str) -> Totals | None:
    unknown = [name for name in named if name not in names]
    if unknown:
        msg = f"{kind} totals name {unknown}, which the panel does not have: {list(names)}"
        raise ValueError(msg)
    if not named:
        return None
    ranges = [named.get(name, Range()) for name in names]
    return Totals(
        least=np.array([r.least for r in ranges]), most=np.array([r.most for r in ranges])
    )


def _cell_most(
    upper: _Array, channels: Sequence[Channel], national: Mapping[Channel, Range]
) -> _Array:
    """A geo's cells' most a period, a national channel's its national spend's most, which bounds
    the geo's share of it."""
    return np.array(
        [
            national[c].most if c in national else most
            for c, most in zip(channels, upper, strict=True)
        ]
    )


def _grid(geos: Sequence[Geo], channels: Sequence[Channel]) -> list[tuple[Geo, Channel]]:
    return [(g, c) for g in geos for c in channels]


def _current(panel: MediaPanel, periods: int) -> _Array:
    """Each cell's mean spend a period over the panel's last `periods` observed periods, or over
    all of them where it has fewer."""
    if isinstance(periods, bool) or not isinstance(periods, int) or periods < 1:
        msg = f"periods must be a whole number of periods, at least 1, got {periods!r}"
        raise ValueError(msg)
    observed = ~panel.filled[:, -periods:]
    counts = observed.sum(axis=1)
    if not counts.all():
        silent = [g for g, n in zip(panel.geos, counts, strict=True) if not n]
        msg = f"geos {silent} have no observed period among the panel's last {periods}"
        raise ValueError(msg)
    # a filled step's spend is 0, so the sum over the window is the observed periods' sum
    return panel.spend[:, -periods:, :].sum(axis=1) / counts[:, None]


def _gain(
    cells: tuple[tuple[Response, ...], ...], history: _Array, rates: _Array, periods: int
) -> float:
    """What spending `rates` a period adds over the plan's periods and each kernel's tail after,
    against spending nothing in them, each cell run over its history, the plan and the tail as one
    spend series: the gain `chc.allocation` reads."""
    before = history.shape[0]
    total = 0.0
    for g, row in enumerate(cells):
        for c, cell in enumerate(row):
            tail = np.zeros(cell.kernel.length - 1)
            for rate, sign in ((rates[g, c], 1.0), (0.0, -1.0)):
                series = np.concatenate([history[:, g, c], np.full(periods, rate), tail])
                total += sign * float(jnp.sum(cell(jnp.asarray(series))[before:]))
    return total
