"""Costs per incremental KPI (ADR 0070): on a conversions world the cost per incremental KPI and
the marginal cost `costs` reads match the generator's own definitions to 1e-9; a revenue per KPI
moves the return and not the costs; and a cost with no value says why, never an infinity or a
NaN."""

from __future__ import annotations

import math

import numpy as np
import pytest
from chc import response

from mixdyn.decisions import Costs, Undefined, costs, returns

WEEKS, LENGTH = 52, 30
RETENTION, HALF, SLOPE, COEFFICIENT = 0.6, 50.0, 1.5, 400.0
"""The conversions world: orders a week from one channel, carried over by the normalised geometric
kernel of 30 weeks at a retention of 0.6 and saturated by a Hill of half-saturation 50 and slope
1.5, under a ceiling of 400 orders."""
VALUE = 35.0
"""Revenue an order brings."""


def spend() -> np.ndarray:
    rng = np.random.default_rng(20261106)
    return rng.gamma(2.0, 30.0, WEEKS) * (rng.random(WEEKS) < 0.8)


def channel(coefficient: float = COEFFICIENT) -> response.Channel:
    return response.Channel(
        response.GeometricAdstock(RETENTION, length=LENGTH, normalized=True),
        response.Hill(HALF, SLOPE),
        coefficient,
    )


def adstock(series: np.ndarray) -> np.ndarray:
    """The generator's carryover: the series run on for the kernel's length less one with nothing
    spent, each period the weighted sum of its last 30 periods' spend."""
    weights = RETENTION ** np.arange(LENGTH)
    weights /= weights.sum()
    run = np.r_[series, np.zeros(LENGTH - 1)]
    return np.array(
        [weights[: t + 1][::-1] @ run[max(0, t + 1 - LENGTH) : t + 1] for t in range(run.size)]
    )


def orders(series: np.ndarray) -> np.ndarray:
    a = adstock(series)
    return COEFFICIENT * a**SLOPE / (a**SLOPE + HALF**SLOPE)


def generator(series: np.ndarray, window: slice) -> tuple[float, float]:
    """The world's cost per incremental order and its marginal cost: the window's spend over the
    orders it adds wherever they fall, and over the derivative of every period's orders along it,
    the Hill's slope `n K^n a^(n - 1) / (a^n + K^n)^2` at each period's adstock."""
    inside = np.zeros(series.size)
    inside[window] = 1.0
    spent = float(series[window].sum())
    added = float(np.sum(orders(series) - orders(series * (1.0 - inside))))
    a, along = adstock(series), adstock(series * inside)
    slope = SLOPE * HALF**SLOPE * a ** (SLOPE - 1.0) / (a**SLOPE + HALF**SLOPE) ** 2
    return spent / added, spent / float(np.sum(COEFFICIENT * slope * along))


@pytest.mark.parametrize("window", [slice(None), slice(10, 21), slice(48, None)])
def test_the_costs_are_the_generator_s(window: slice) -> None:
    series = spend()
    read = costs(channel(), series, window)
    per_kpi, marginal = generator(series, window)
    assert isinstance(read.cost_per_kpi, float)
    assert isinstance(read.marginal_cost_per_kpi, float)
    assert read.spend == pytest.approx(float(series[window].sum()), rel=1e-12, abs=0.0)
    assert read.cost_per_kpi == pytest.approx(per_kpi, rel=1e-9, abs=0.0)
    assert read.marginal_cost_per_kpi == pytest.approx(marginal, rel=1e-9, abs=0.0)
    assert read.incremental == pytest.approx(read.spend / per_kpi, rel=1e-9, abs=0.0)
    assert read.marginal_cost_per_kpi > read.cost_per_kpi


def test_revenue_per_kpi_moves_the_return_and_not_the_costs() -> None:
    series, window = spend(), slice(10, 21)
    read = costs(channel(), series, window)
    valued = returns(channel(), series, window, revenue_per_kpi=VALUE)
    assert isinstance(read.cost_per_kpi, float)
    assert isinstance(read.marginal_cost_per_kpi, float)
    assert valued.roi == pytest.approx(VALUE / read.cost_per_kpi, rel=1e-12, abs=0.0)
    assert valued.marginal_roi == pytest.approx(
        VALUE / read.marginal_cost_per_kpi, rel=1e-12, abs=0.0
    )


def test_a_cost_with_no_value_says_why() -> None:
    series = spend()
    series[20:30] = 0.0
    dark = costs(channel(), series, slice(20, 30))
    assert dark == Costs(0.0, 0.0, dark.cost_per_kpi, dark.marginal_cost_per_kpi)
    for cost in (dark.cost_per_kpi, dark.marginal_cost_per_kpi):
        assert isinstance(cost, Undefined)
        assert "spends 0.0, so nothing was paid" in cost.reason
    idle = costs(channel(0.0), spend(), slice(None))
    assert idle.incremental == 0.0
    assert idle.cost_per_kpi == Undefined("the window's spend earns no incremental KPI")
    assert idle.marginal_cost_per_kpi == Undefined(
        "one more currency unit earns no incremental KPI"
    )
    harmful = costs(channel(-COEFFICIENT), spend(), slice(None))
    assert harmful.incremental < 0.0
    assert isinstance(harmful.cost_per_kpi, Undefined)
    assert harmful.cost_per_kpi.reason.startswith("the window's spend loses ")
    assert isinstance(harmful.marginal_cost_per_kpi, Undefined)
    assert harmful.marginal_cost_per_kpi.reason.startswith("one more currency unit loses ")
    for read in (dark, idle, harmful):
        assert all(math.isfinite(v) for v in (read.spend, read.incremental))


@pytest.mark.parametrize(
    ("series", "match"),
    [
        (np.ones((2, 3)), "non-empty vector"),
        (np.array([]), "non-empty vector"),
        (np.array([1.0, np.inf]), "finite and at or above 0"),
        (np.array([1.0, -1.0]), "finite and at or above 0"),
    ],
)
def test_what_is_not_a_spend_series_is_refused(series: np.ndarray, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        costs(channel(), series)
