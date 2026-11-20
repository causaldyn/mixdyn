"""Every return mixdyn reports for a channel, one definition each, read off `chc.response`.

Over a window of a spend series:

* `contribution`: the channel's revenue in the window's periods, carried over from all of its
  spend before them, what a decomposition attributes to it there;
* `roi`: the revenue the window's spend earns, per currency unit of it, all of its carryover
  counted: the series runs on past its end for the kernel's length with nothing spent. It is gross,
  so a return of 1 pays the spend back, and the net return is `roi - 1`;
* `marginal_roi`: the revenue of one more currency unit, spread over the window as its spend is,
  carryover counted as in `roi`;
* `steady_state_marginal_roi`: the revenue of one more currency unit a period, once the window's
  mean spend a period has been held for longer than the kernel remembers;
* `carryover`: the share of a unit of spend's adstock that lands after the period it is spent in,
  `1 - w_0 / sum(w)`, which is the share of its return on a linear curve.

Revenue is the KPI times `revenue_per_kpi`; at 1 the KPI is revenue. A portfolio's return is its
channels' revenue over their spend, `portfolio_roi`, each channel's revenue at its own
`revenue_per_kpi`. A Bayesian fit's returns at each of its draws are `returns` of the channels its
`channel_draws` hands on, and a return's interval is their quantiles.

Where the KPI counts events, conversions, orders or sign-ups, a return in the KPI's units is
incremental KPI per currency unit, and `costs` reads its inverses, each off `returns` at a revenue
per KPI of 1:

* `cost_per_kpi`, CPIK: the window's spend over the incremental KPI it earns, its carryover
  counted, `1 / roi`; a CPA, where the KPI counts acquisitions;
* `marginal_cost_per_kpi`: the cost of one more incremental KPI at the window's spend,
  `1 / marginal_roi`.

A cost with no value, a window that spends nothing or a spend that earns no KPI, is `Undefined`,
with its reason, never an infinity or a NaN.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from chc import response


@dataclass(frozen=True, slots=True)
class Returns:
    """One channel's returns over a window: `spend` in currency, `contribution` in revenue, the
    returns in revenue per currency unit and `carryover` a share."""

    spend: float
    contribution: float
    roi: float
    marginal_roi: float
    steady_state_marginal_roi: float
    carryover: float


def returns(
    channel: response.Channel,
    spend: npt.ArrayLike,
    window: slice = slice(None),
    *,
    revenue_per_kpi: float = 1.0,
) -> Returns:
    """`channel`'s returns over `window` of its `(T,)` spend series, nothing spent before it.

    Raises:
        ValueError: on a spend series that is not a vector of finite values at or above 0, a
            window that spends nothing, and a `revenue_per_kpi` not positive and finite.
    """
    series = np.asarray(spend, dtype=float)
    if series.ndim != 1 or series.size == 0:
        msg = f"spend must be a non-empty vector, got shape {series.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(series) & (series >= 0.0)):
        msg = "spend must be finite and at or above 0"
        raise ValueError(msg)
    if not (np.isfinite(revenue_per_kpi) and revenue_per_kpi > 0.0):
        msg = f"revenue per KPI must be positive and finite, got {revenue_per_kpi}"
        raise ValueError(msg)
    periods = slice(*window.indices(series.size))
    weights = np.asarray(channel.kernel.weights())
    extended = np.r_[series, np.zeros(weights.size)]
    inside = series[periods]
    return Returns(
        spend=float(inside.sum()),
        contribution=response.contribution(
            channel, series, periods, revenue_per_kpi=revenue_per_kpi
        ),
        roi=response.roi(channel, extended, periods, revenue_per_kpi=revenue_per_kpi),
        marginal_roi=response.marginal_roi(
            channel, extended, periods, revenue_per_kpi=revenue_per_kpi
        ),
        steady_state_marginal_roi=response.steady_state_marginal_roi(
            channel, float(inside.mean()), revenue_per_kpi=revenue_per_kpi
        ),
        carryover=float(1.0 - weights[0] / weights.sum()),
    )


@dataclass(frozen=True, slots=True)
class Undefined:
    """A cost with no value, and why."""

    reason: str


@dataclass(frozen=True, slots=True)
class Costs:
    """One channel's costs per incremental KPI over a window: `spend` in currency; `incremental`,
    the KPI the window's spend earns wherever it falls, its carryover counted, `roi` in the KPI's
    units times the spend; `cost_per_kpi`, CPIK, the spend over that KPI; and
    `marginal_cost_per_kpi`, the cost of one more incremental KPI at the window's spend. Each cost
    is in currency per KPI, or `Undefined`."""

    spend: float
    incremental: float
    cost_per_kpi: float | Undefined
    marginal_cost_per_kpi: float | Undefined


def costs(channel: response.Channel, spend: npt.ArrayLike, window: slice = slice(None)) -> Costs:
    """`channel`'s costs per incremental KPI over `window` of its `(T,)` spend series, nothing spent
    before it: one over `returns`' `roi` and `marginal_roi` in the KPI's units.

    A window that spends nothing has neither cost; a spend that earns no incremental KPI, or loses
    some, has no cost per KPI, and a unit more that earns none, or loses some, no marginal cost.
    Each is `Undefined`, with its reason.

    Raises:
        ValueError: on a spend series that is not a vector of finite values at or above 0.
    """
    series = np.asarray(spend, dtype=float)
    if series.ndim != 1 or series.size == 0:
        msg = f"spend must be a non-empty vector, got shape {series.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(series) & (series >= 0.0)):
        msg = "spend must be finite and at or above 0"
        raise ValueError(msg)
    spent = float(series[window].sum())
    if not spent > 0.0:
        none = Undefined(f"the window spends {spent}, so nothing was paid for any KPI")
        return Costs(spent, 0.0, none, none)
    read = returns(channel, series, window)
    return Costs(
        spend=read.spend,
        incremental=read.roi * read.spend,
        cost_per_kpi=_inverse(read.roi, "the window's spend"),
        marginal_cost_per_kpi=_inverse(read.marginal_roi, "one more currency unit"),
    )


def _inverse(kpi_per_unit: float, what: str) -> float | Undefined:
    """The currency a KPI costs where `what` earns `kpi_per_unit` KPI a currency unit."""
    if kpi_per_unit > 0.0:
        return 1.0 / kpi_per_unit
    if kpi_per_unit == 0.0:
        return Undefined(f"{what} earns no incremental KPI")
    if kpi_per_unit < 0.0:
        return Undefined(f"{what} loses {-kpi_per_unit} KPI a currency unit")
    return Undefined(f"{what} earns {kpi_per_unit} KPI a currency unit")


def portfolio_roi(returns: Iterable[Returns]) -> float:
    """Several channels' revenue over their spend: each channel's `roi` weighted by its spend.

    Raises:
        ValueError: on channels that spend nothing between them.
    """
    spend = revenue = 0.0
    for r in returns:
        spend += r.spend
        revenue += r.roi * r.spend
    if not spend > 0.0:
        msg = f"the channels spend {spend}; their return per unit is undefined"
        raise ValueError(msg)
    return revenue / spend
