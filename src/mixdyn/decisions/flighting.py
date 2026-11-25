"""A flighted plan for a posterior: when to spend over the periods, scored at every draw.

`plan_flighting` reads one set of channels off the posterior, each parameter at its weighted median
over the draws (or one draw's), and plans them twice by `chc.allocation.allocate`: a spend of its
own in each period, the schedule, and one rate a period, the continuous plan. It then scores the
schedule, the continuous plan and the status quo, each channel's current spend in every period, at
every draw: what each adds to the KPI over the plan's periods and the kernel's length after them,
against spending nothing in the plan, the history carried into all three alike. A plan that keeps
the status quo therefore gains exactly what the status quo does, at every draw.

The schedule is called flighted only where its gain over the continuous plan has a weighted
central `level` interval over the draws whose lower end is above the schedule's search tolerance;
else the plan is continuous. On concave curves through one-period kernels the best schedule is the
best rate a period, so a pulse is never recommended there.

Where a kernel is longer than one period the schedule is chc's best of local ascents, its bound the
curves' concave envelopes': chc then reads `limit` as `"local"` and logs a warning
(`chc_event="allocation_local"`), which reaches the caller as logged. The planner spends one budget
over the periods; per-period budgets, pacing, are refused.

A plan names in `unidentified` each channel the schedule, in any period, or the continuous plan
moves off its current spend that the panel flags `IssueCode.CONSTANT` in the geo, as
`mixdyn.decisions.posterior`'s plans name it: its worth there is the prior's. No number changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
from chc.allocation import SearchLimit, SearchStatus, allocate

from mixdyn.decisions._quantile import weighted_quantiles
from mixdyn.decisions.posterior import (
    Readings,
    _carried,
    _current,
    _inputs,
    _response,
    _spendable,
    _unidentified,
)
from mixdyn.models.bayes_mmm import _TANH_HALF

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mixdyn.data import Channel, Geo, MediaPanel
    from mixdyn.decisions.geo_plan import Range

type _Array = npt.NDArray[np.float64]

_HALF = 1e-12
"""How near a weighted quantile's cumulative weight is to its level for the two values beside it
to be averaged, as `np.median` averages an even count's middle two."""

Recommendation = Literal["flighted", "continuous"]
"""What a `FlightPlan` recommends: its schedule, `"flighted"`, or the best rate in every period,
`"continuous"`."""


@dataclass(frozen=True, slots=True, eq=False)
class FlightPlan:
    """A schedule of one geo's channels over periods, the best rate a period beside it, and how
    each and the status quo score over a posterior's draws.

    `schedule` is `(periods, channels)` spend in each period, `continuous` and `current` are
    `(channels,)` spend a period, the channels in the readings' order: the schedule's, the best
    rate a period's, both planned at `reading`, `"median"` or a draw's index, and each channel's
    mean over the panel's last `periods` observed periods. `gains` is `(3, draws)`, each draw's
    gain of the schedule, the continuous plan and the status quo in that order, each what its spend
    adds to the KPI over the periods and the kernel's length after them against spending nothing
    in the plan, the history carried into all three. `expected` is their weighted means, `(3,)`,
    and `intervals` their weighted central `level` intervals, `(3, 2)`. `edge` is each draw's gain
    of the schedule over the continuous plan, `(draws,)`, and `edge_interval` its interval.
    `worth`, `bound`, `stopped`, `limit` and `tolerance` are the schedule's search's at the
    reading, as `chc.allocation.Allocation` reads them; where the search's schedule returns less
    than the best rate a period at the reading, the schedule is that rate in every period and
    `worth` its worth. `recommendation` is `"flighted"` where the lower end of `edge_interval` is
    above `tolerance`, else `"continuous"`. An interval's ends are weighted quantiles, each the
    least draw whose cumulative weight reaches its share; a draw of weight 0 enters neither them
    nor the median. `unidentified` names each channel the schedule, in any period, or the
    continuous plan moves off its current spend whose spend the panel flags as never moving in the
    geo (`IssueCode.CONSTANT`), its worth there read from its prior alone.
    """

    channels: tuple[Channel, ...]
    periods: int
    budget: float
    reading: str | int
    level: float
    schedule: _Array
    continuous: _Array
    current: _Array
    gains: _Array
    expected: _Array
    intervals: _Array
    edge: _Array
    edge_interval: _Array
    worth: float
    bound: float
    stopped: SearchStatus
    limit: SearchLimit | None
    tolerance: float
    recommendation: Recommendation
    unidentified: tuple[Channel, ...] = ()


def plan_flighting(
    panel: MediaPanel,
    geo: Geo,
    readings: Readings,
    budget: float,
    periods: int,
    limits: Mapping[Channel, Range],
    *,
    reading: str | int = "median",
    level: float = 0.9,
) -> FlightPlan:
    """The schedule of `geo`'s channels over `periods` periods of the panel's cadence that spends
    `budget` over them within `limits`, each channel's range in each period, planned on the
    readings' channels at `reading`, and the best rate a period beside it; each scored, with the
    status quo, at every draw.

    `reading` is `"median"`, each parameter at its weighted median over the draws of weight above
    0, which must all be of one curve, or the index of one draw.

    Raises:
        ValueError: on per-period budgets, which the planner does not pace; what `plan_posterior`
            refuses; a `reading` neither `"median"` nor a draw's index, or `"median"` over draws
            of both curves; and a `level` outside `(0, 1)`.
    """
    if np.ndim(budget) != 0:
        msg = (
            "per-period budgets are not planned: chc's planner spends one budget over the "
            "periods, each period's spend free within its range; pass the total"
        )
        raise ValueError(msg)
    history, lower, upper = _inputs(panel, geo, readings, periods, limits)
    budget = _spendable(budget, periods, lower, upper)
    level = float(level)
    if not 0.0 < level < 1.0:
        msg = f"level={level!r} is not a share in (0, 1)"
        raise ValueError(msg)
    current = _current(panel, geo, readings.channels, periods)
    planned, s = _planned(readings, reading)
    channels = [_response(planned, s, c) for c in range(len(readings.channels))]
    scheduled, continuous = (
        allocate(
            channels, budget, periods, lower=lower, upper=upper, history=history, schedule=flighted
        )
        for flighted in (True, False)
    )
    rate = np.asarray(continuous.spend, dtype=np.float64)
    worth = float(scheduled.worth)
    if worth >= continuous.worth:
        schedule = np.asarray(scheduled.schedule, dtype=np.float64)
    else:
        # the boxes' search closes its gap on the cells' interpolants, and its schedule can fall
        # short of the best rate a period on the curves by more than its tolerance
        schedule, worth = np.tile(rate, (periods, 1)), float(continuous.worth)
    grids = (schedule, np.tile(rate, (periods, 1)), np.tile(current, (periods, 1)))
    nothing = _scored(history, readings, np.zeros_like(schedule))
    gains = np.stack([_scored(history, readings, grid) - nothing for grid in grids])
    weights = readings.weights
    edge = gains[0] - gains[1]
    edge_interval = _interval(edge, weights, level)
    tolerance = float(scheduled.tolerance)
    for value in (schedule, rate, current, gains, edge, edge_interval):
        value.setflags(write=False)
    return FlightPlan(
        channels=readings.channels,
        periods=periods,
        budget=budget,
        reading=reading,
        level=level,
        schedule=schedule,
        continuous=rate,
        current=current,
        gains=gains,
        expected=gains @ weights,
        intervals=np.stack([_interval(g, weights, level) for g in gains]),
        edge=edge,
        edge_interval=edge_interval,
        worth=worth,
        bound=max(float(scheduled.bound), worth),
        stopped=scheduled.stopped,
        limit=scheduled.limit,
        tolerance=tolerance,
        recommendation="flighted" if edge_interval[0] > tolerance else "continuous",
        unidentified=_unidentified(
            panel, geo, readings.channels, np.vstack([schedule, rate]), current, upper
        ),
    )


def _planned(readings: Readings, reading: str | int) -> tuple[Readings, int]:
    """The readings and the draw the planner reads: the draws' weighted medians as one draw, or
    the draw `reading` of the readings."""
    draws = readings.fit.size
    if reading == "median":
        tanh = np.unique(readings.tanh[readings.weights > 0.0])
        if tanh.size > 1:
            msg = "the draws hold both curves, which have no median: plan at a draw's index"
            raise ValueError(msg)
        weights = readings.weights
        channels = len(readings.channels)

        def median(values: _Array) -> _Array:
            return np.array([[_quantile(values[:, c], weights, 0.5) for c in range(channels)]])

        one = Readings(
            channels=readings.channels,
            retention=median(readings.retention),
            half_saturation=median(readings.half_saturation),
            slope=np.full((1, channels), np.nan) if tanh[0] else median(readings.slope),
            coefficient=median(readings.coefficient),
            tanh=tanh[:1],
            fit=np.zeros(1, dtype=np.intp),
            chain=np.zeros(1, dtype=np.intp),
            stacking=np.ones(1),
            pointwise=None,
            length=readings.length,
        )
        return one, 0
    if isinstance(reading, bool) or not isinstance(reading, int) or not 0 <= reading < draws:
        msg = f"reading={reading!r} is neither 'median' nor a draw's index in [0, {draws})"
        raise ValueError(msg)
    return readings, reading


def _quantile(values: _Array, weights: _Array, share: float) -> float:
    """The weighted `share` quantile of `values` over the draws of weight above 0: the least value
    whose cumulative weight reaches the share, averaged with the next where it reaches it exactly,
    so that alike weights read `np.median`'s median; a draw of weight 0 is never read."""
    kept = weights > 0.0
    values, weights = values[kept], weights[kept]
    order = np.argsort(values, kind="stable")
    ordered, cumulative = values[order], np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    i = min(int(np.searchsorted(cumulative, share - _HALF)), ordered.size - 1)
    if abs(cumulative[i] - share) <= _HALF and i + 1 < ordered.size:
        return float((ordered[i] + ordered[i + 1]) / 2.0)
    return float(ordered[i])


def _interval(values: _Array, weights: _Array, level: float) -> _Array:
    """The weighted central `level` interval of `values`, `(2,)`: each end a draw's own value,
    as `mixdyn.decisions._quantile` reads every decision module's."""
    low, high = weighted_quantiles(values, weights, ((1.0 - level) / 2.0, (1.0 + level) / 2.0))
    return np.array([float(low), float(high)])


def _scored(history: _Array, readings: Readings, grid: _Array) -> _Array:
    """Each draw's worth of the schedule `grid`, `(periods, channels)`, over its periods and the
    kernel's length after them, the history's carryover in, summed over the channels:
    `(draws,)`, each draw at its own parameters."""
    length = readings.length
    before = history.shape[0]
    total = np.zeros(readings.fit.size)
    for c in range(grid.shape[1]):
        retention = readings.retention[:, c]
        steps = np.concatenate(
            [np.ones((retention.size, 1)), np.repeat(retention[:, None], length - 1, 1)], 1
        )
        weights = np.cumprod(steps, axis=1)
        weights /= weights.sum(axis=1, keepdims=True)
        series = np.concatenate([history[:, c], grid[:, c], np.zeros(length - 1)])
        adstock = _carried(series, weights, before)
        half = readings.half_saturation[:, c, None]
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            hill = np.where(
                adstock > 0.0, 1.0 / (1.0 + (half / adstock) ** readings.slope[:, c, None]), 0.0
            )
        curve = np.where(readings.tanh[:, None], np.tanh(_TANH_HALF * adstock / half), hill)
        total += readings.coefficient[:, c] * curve.sum(axis=1)
    return total
