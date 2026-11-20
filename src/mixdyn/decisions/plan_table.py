"""A plan of one geo's channels, read against the current spend channel by channel.

`plan_table` reads a plan as a planner reads it: each channel's spend a period now and in the plan,
the change in percent, and what the change does to the KPI and to revenue over the plan's periods
and the carryover after them, each a posterior mean and an interval over the draws. Each draw's
KPI change is `plan_breakdown`'s, so the rows sum to the plan's expected gain and a channel the plan
holds is credited with nothing at every draw.

Every return in the table is `mixdyn.decisions.returns` at a draw, as ADR 0016 defines it: the
marginal return a currency unit at the current and at the planned spend is `returns`'s
`marginal_roi` of the draw's channel, `chc.response`'s, on the history's spend followed by the
plan's periods at that rate, the series the planner reads, and the window those periods. It is
gross and in revenue at the channel's revenue per KPI. A draw's coefficient is never a return: on a
Hill it is the curve's ceiling, not what a unit earns.

The diminishing-returns point is the spend a period, within the channel's range, past which the
posterior-mean marginal return stays below breakeven: 1 on revenue, a gross return of 1 paying the
spend back, and `1 / margin` on profit at a gross margin `margin`. It is read on `_SCAN` spends
evenly across the range and bisected between the last spend at or above breakeven and the first of
the run below it that reaches the range's most, to `_TOLERANCE` of the range. It is None where the
mean is at or above breakeven at the most, or below it from the range's first spend that spends
something. Each mean there is read at every draw and every spend of the scan at once, in NumPy, by
the derivative of the draw's curve in closed form: the adstock is linear in the spend, so a draw's
marginal return at a rate `x` is `b sum_t h'(a_t + x r_t) r_t / periods`, `a` the adstock the
history leaves and `r` that of one currency unit a period over the plan's periods, each over those
periods and the kernel's length after them. It is `returns`'s `marginal_roi` to `1e-12` of it, its
sums run in another order.

An interval is the weighted quantiles of the draws at `(1 - level) / 2` and `(1 + level) / 2`, each
the least draw whose cumulative weight, the draws sorted, reaches that share of the total: the
inverse of the weighted empirical distribution, a draw's own value, with no interpolation between
draws, so a draw of weight 0 is never read. A cumulative weight reaches a share within its sums'
rounding, as `mixdyn.decisions._quantile` reads it for every decision module.

`ranges_from_current` states a plan's ranges a period as a planner does: each channel's least a
share of its current spend and its most a multiple of it, a floor in currency where one is named.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from mixdyn.decisions._quantile import weighted_quantiles
from mixdyn.decisions.geo_plan import Range
from mixdyn.decisions.posterior import (
    _BLOCK,
    _carried,
    _history,
    _inputs,
    _response,
    plan_breakdown,
)
from mixdyn.decisions.returns import returns
from mixdyn.models.bayes_mmm import _TANH_HALF

if TYPE_CHECKING:
    from mixdyn.data import Channel, Geo, MediaPanel
    from mixdyn.decisions.posterior import Readings

type _Array = npt.NDArray[np.float64]

_SCAN = 64
"""How many spends evenly across a channel's range its mean marginal return is read at."""
_TOLERANCE = 1e-9
"""The share of a channel's range within which the diminishing-returns point is bisected."""


@dataclass(frozen=True, slots=True, eq=False)
class Spread:
    """A quantity over a posterior's draws: `draws`, `(draws, ...)`, each draw's value; `mean`
    their mean by the draws' weights; `low` and `high` their weighted quantiles at
    `(1 - level) / 2` and `(1 + level) / 2`, each `(...)`. A quantity NaN at any draw is NaN in
    each."""

    draws: _Array
    mean: _Array
    low: _Array
    high: _Array


@dataclass(frozen=True, slots=True, eq=False)
class PlanTable:
    """A plan of one geo's channels against the current spend, one row a channel.

    `current` and `spend` are `(channels,)` in the readings' order, in spend a period: the
    reference's and the plan's. `change` is the plan's change in percent of the current spend, 0
    where both are 0 and NaN where only the current is. `kpi` is each channel's change in the KPI
    from the current spend to the plan's over its periods and the carryover after them, the
    history's carried in, and `revenue` that change times the channel's `revenue_per_kpi`.
    `marginal_current` and `marginal_planned` are each draw's `returns(...).marginal_roi` at the
    current and at the planned spend, in revenue a currency unit, NaN at a spend of 0, where the
    window spends nothing. `diminishing` is each channel's spend a period at which the
    posterior-mean marginal return falls to `breakeven` within its range, or None.

    `total_kpi` and `total_revenue` are the channels' changes summed at each draw; `total_kpi`'s
    mean is the plan's expected gain over the reference. `factor` is the plan's breakdown factor
    for the expected gain, and `factor_low` and `factor_high` the quantiles of the draws' own.
    """

    channels: tuple[Channel, ...]
    periods: int
    level: float
    margin: float | None
    breakeven: float
    current: _Array
    spend: _Array
    change: _Array
    revenue_per_kpi: _Array
    kpi: Spread
    revenue: Spread
    marginal_current: Spread
    marginal_planned: Spread
    diminishing: tuple[float | None, ...]
    total_kpi: Spread
    total_revenue: Spread
    factor: float
    factor_low: float
    factor_high: float

    def rows(self) -> list[dict[str, object]]:
        """The table's rows, one a channel in its order, each column by name."""
        return [
            {
                "channel": channel,
                "current": float(self.current[c]),
                "planned": float(self.spend[c]),
                "change_percent": float(self.change[c]),
                "kpi_change": float(self.kpi.mean[c]),
                "kpi_change_low": float(self.kpi.low[c]),
                "kpi_change_high": float(self.kpi.high[c]),
                "revenue_change": float(self.revenue.mean[c]),
                "revenue_change_low": float(self.revenue.low[c]),
                "revenue_change_high": float(self.revenue.high[c]),
                "marginal_return_current": float(self.marginal_current.mean[c]),
                "marginal_return_planned": float(self.marginal_planned.mean[c]),
                "diminishing_returns_point": self.diminishing[c],
            }
            for c, channel in enumerate(self.channels)
        ]


def plan_table(
    panel: MediaPanel,
    geo: Geo,
    readings: Readings,
    spend: npt.ArrayLike,
    periods: int,
    *,
    against: npt.ArrayLike | None = None,
    revenue_per_kpi: float | Mapping[Channel, float] = 1.0,
    margin: float | None = None,
    level: float = 0.9,
    limits: Mapping[Channel, Range] | None = None,
) -> PlanTable:
    """The table of `spend`, a plan of `geo`'s channels a period over `periods` periods of the
    panel's cadence, against `against`, or else each channel's mean spend over the geo's last
    `periods` observed periods, as `plan_breakdown` reads the current spend.

    `revenue_per_kpi` is one rate for every channel or a rate by channel. `margin`, a gross
    margin in `(0, 1]`, sets breakeven at `1 / margin`, and at 1 on revenue where it is None.
    `limits` holds each channel's range a period for its diminishing-returns point, each with a
    finite most; where it is None a channel's range runs from 0 to the most it spent a period in
    the history, the plan or the reference, where its response was seen or is asked for.

    Raises:
        ValueError: as `plan_breakdown`; a `revenue_per_kpi` not positive and finite or not one a
            channel; a margin outside `(0, 1]`; a level outside `(0, 1)`; and limits as `worths`'s.
    """
    reading = plan_breakdown(panel, geo, readings, spend, periods, against=against)
    rates = _revenue_rates(revenue_per_kpi, readings.channels)
    if margin is not None and not (math.isfinite(margin) and 0.0 < margin <= 1.0):
        msg = f"a margin must be in (0, 1], got {margin}"
        raise ValueError(msg)
    if not (math.isfinite(level) and 0.0 < level < 1.0):
        msg = f"a level must be in (0, 1), got {level}"
        raise ValueError(msg)
    if limits is None:
        history = _history(panel, geo, readings)
        lower = np.zeros(len(readings.channels))
        upper = np.max(np.vstack([history, reading.spend, reading.against]), axis=0)
    else:
        history, lower, upper = _inputs(panel, geo, readings, periods, limits)
    breakeven = 1.0 if margin is None else 1.0 / margin
    weights = readings.weights
    shares = ((1.0 - level) / 2.0, (1.0 + level) / 2.0)

    def spread(draws: _Array, mean: _Array | None = None) -> Spread:
        low, high = weighted_quantiles(draws, weights, shares)
        value = weights @ draws if mean is None else mean
        for array in (draws, value, low, high):
            array.setflags(write=False)
        return Spread(draws=draws, mean=value, low=low, high=high)

    marginal = np.empty((2, readings.fit.size, len(readings.channels)))
    for c in range(len(readings.channels)):
        for k, rate in enumerate((reading.against[c], reading.spend[c])):
            for s in range(readings.fit.size):
                marginal[k, s, c] = _marginal(
                    readings, s, c, history[:, c], rate, periods, rates[c]
                )
    kpi = np.array(reading.gains)
    revenue = kpi * rates
    with np.errstate(divide="ignore", invalid="ignore"):
        change = np.where(
            reading.against > 0.0,
            100.0 * (reading.spend - reading.against) / reading.against,
            np.where(reading.spend > 0.0, np.nan, 0.0),
        )
    factor_low, factor_high = weighted_quantiles(reading.draws[:, None], weights, shares)
    for value in (change, rates):
        value.setflags(write=False)
    return PlanTable(
        channels=readings.channels,
        periods=periods,
        level=level,
        margin=margin,
        breakeven=breakeven,
        current=reading.against,
        spend=reading.spend,
        change=change,
        revenue_per_kpi=rates,
        kpi=spread(kpi, np.array(reading.expected)),
        revenue=spread(revenue, reading.expected * rates),
        marginal_current=spread(marginal[0]),
        marginal_planned=spread(marginal[1]),
        diminishing=tuple(
            _diminishing(
                readings, c, history[:, c], periods, rates[c], breakeven, lower[c], upper[c]
            )
            for c in range(len(readings.channels))
        ),
        total_kpi=spread(kpi.sum(axis=1), np.array(reading.expected.sum())),
        total_revenue=spread(revenue.sum(axis=1), np.array((reading.expected * rates).sum())),
        factor=reading.factor,
        factor_low=float(factor_low[0]),
        factor_high=float(factor_high[0]),
    )


def ranges_from_current(
    current: Mapping[Channel, float],
    *,
    least_share: float,
    most_multiple: float,
    floors: Mapping[Channel, float] | None = None,
) -> dict[Channel, Range]:
    """Each channel's range a period as a planner states it: at least `least_share` of its
    `current` spend a period, or its floor in `floors` where it has one, and at most
    `most_multiple` times its current spend.

    Raises:
        ValueError: on a current spend not finite and at or above 0; a share outside `[0, 1]`; a
            multiple below 1 or not finite; a floor for a channel `current` does not have, not
            finite or below 0, or above the channel's most.
    """
    if not (math.isfinite(least_share) and 0.0 <= least_share <= 1.0):
        msg = f"the least share of the current spend must be in [0, 1], got {least_share}"
        raise ValueError(msg)
    if not (math.isfinite(most_multiple) and most_multiple >= 1.0):
        msg = f"the most multiple of the current spend must be finite, 1 or more: {most_multiple}"
        raise ValueError(msg)
    floors = {} if floors is None else dict(floors)
    unknown = [c for c in floors if c not in current]
    if unknown:
        msg = f"floors name channels with no current spend: {unknown}"
        raise ValueError(msg)
    ranges = {}
    for channel, spend in current.items():
        spend = float(spend)
        if not (math.isfinite(spend) and spend >= 0.0):
            msg = f"{channel}'s current spend must be finite and at or above 0, got {spend}"
            raise ValueError(msg)
        most = most_multiple * spend
        least = least_share * spend
        if channel in floors:
            floor = float(floors[channel])
            if not (math.isfinite(floor) and 0.0 <= floor <= most):
                msg = f"{channel}'s floor {floor} is not in [0, its most {most}]"
                raise ValueError(msg)
            least = floor
        ranges[channel] = Range(least, most)
    return ranges


def _revenue_rates(
    revenue_per_kpi: float | Mapping[Channel, float], channels: tuple[Channel, ...]
) -> _Array:
    """Each channel's revenue per KPI, `(channels,)`."""
    if isinstance(revenue_per_kpi, Mapping):
        missing = [c for c in channels if c not in revenue_per_kpi]
        extra = [c for c in revenue_per_kpi if c not in channels]
        if missing or extra:
            msg = f"revenue per KPI needs one rate a channel: missing {missing}, no such {extra}"
            raise ValueError(msg)
        rates = np.array([float(revenue_per_kpi[c]) for c in channels])
    else:
        rates = np.full(len(channels), float(revenue_per_kpi))
    if not np.all(np.isfinite(rates) & (rates > 0.0)):
        msg = f"revenue per KPI must be positive and finite, got {rates}"
        raise ValueError(msg)
    return rates


def _marginal(
    readings: Readings,
    s: int,
    c: int,
    history: _Array,
    rate: float,
    periods: int,
    revenue_per_kpi: float,
) -> float:
    """Draw `s`'s marginal return on channel `c` at `rate` a period over the plan's periods, after
    the history: `returns`'s, NaN at a rate of 0, where the window spends nothing."""
    if not rate > 0.0:
        return math.nan
    series = np.concatenate([history, np.full(periods, rate)])
    window = slice(history.size, history.size + periods)
    return returns(
        _response(readings, s, c), series, window, revenue_per_kpi=revenue_per_kpi
    ).marginal_roi


def _mean_marginals(
    readings: Readings,
    c: int,
    history: _Array,
    rates: _Array,
    periods: int,
    revenue_per_kpi: float,
) -> _Array:
    """The posterior-mean marginal return on channel `c` at each of `rates`, each above 0, over
    the draws of weight: `(rates,)`, each draw's `_marginal` read in closed form."""
    weights = readings.weights
    kept = np.flatnonzero(weights > 0.0)
    retention = readings.retention[kept, c]
    length = readings.length
    steps = np.concatenate(
        [np.ones((kept.size, 1)), np.repeat(retention[:, None], length - 1, 1)], 1
    )
    kernel = np.cumprod(steps, axis=1)
    kernel /= kernel.sum(axis=1, keepdims=True)
    before = history.size
    carry = _carried(np.concatenate([history, np.zeros(periods + length - 1)]), kernel, before)
    unit = np.concatenate([np.zeros(before), np.ones(periods), np.zeros(length - 1)])
    reach = _carried(unit, kernel, before)
    tanh = readings.tanh[kept]
    half = readings.half_saturation[kept, c][:, None]
    slope = readings.slope[kept, c][:, None]
    coefficient = readings.coefficient[kept, c]
    means = np.empty(rates.size)
    rows = max(1, _BLOCK // carry.size)
    for start in range(0, rates.size, rows):
        block = rates[start : start + rows]
        adstock = carry[None] + block[:, None, None] * reach[None]
        bend = np.empty_like(adstock)
        # tanh(a / s), s = half / atanh(1/2): its slope 4 e / (1 + e)^2 / s with e = exp(-2 a / s)
        scale = half[tanh] / _TANH_HALF
        e = np.exp(-2.0 * adstock[:, tanh] / scale)
        bend[:, tanh] = 4.0 * e / (1.0 + e) ** 2 / scale
        # the Hill a^n / (a^n + k^n): its slope n u / (a (1 + u)^2), u = (a / k)^n or its inverse,
        # whichever is at most 1, so no power overflows; read only where a unit's adstock reaches,
        # where the adstock is above 0
        hill = ~tanh
        a, k, n = adstock[:, hill], half[hill], slope[hill]
        reached = np.broadcast_to(reach[hill] > 0.0, a.shape)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            u = np.where(a <= k, (a / k) ** n, (k / a) ** n)
            bend[:, hill] = np.where(reached, n * u / (a * (1.0 + u) ** 2), 0.0)
        marginals = revenue_per_kpi * coefficient * (bend * reach[None]).sum(axis=2) / periods
        means[start : start + block.size] = [math.fsum(weights[kept] * row) for row in marginals]
    return means


def _diminishing(
    readings: Readings,
    c: int,
    history: _Array,
    periods: int,
    revenue_per_kpi: float,
    breakeven: float,
    least: float,
    most: float,
) -> float | None:
    """The spend a period in `[least, most]` past which channel `c`'s mean marginal return stays
    below `breakeven` on the scan's spends, bisected, or None."""
    spends = np.linspace(least, most, _SCAN)
    spends = spends[spends > 0.0]
    if spends.size == 0:
        return None

    def mean(rate: float) -> float:
        return float(
            _mean_marginals(readings, c, history, np.array([rate]), periods, revenue_per_kpi)[0]
        )

    below = _mean_marginals(readings, c, history, spends, periods, revenue_per_kpi) < breakeven
    if not below[-1]:
        return None
    if below.all():
        return None
    first = int(np.flatnonzero(~below)[-1]) + 1
    low, high = float(spends[first - 1]), float(spends[first])
    while high - low > _TOLERANCE * (most - least):
        middle = 0.5 * (low + high)
        if middle <= low or middle >= high:
            break
        if mean(middle) < breakeven:
            high = middle
        else:
            low = middle
    return 0.5 * (low + high)
