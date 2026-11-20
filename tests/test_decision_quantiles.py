"""mixdyn.decisions._quantile: every decision module's interval ends are one weighted quantile.

Each end is the least draw whose cumulative weight reaches its share, Hyndman and Fan's first
definition, read here in exact arithmetic from the level as written and each draw's weight as a
share of the draws: at 20 and 100 draws of equal weight, levels 0.5 and 0.9, `plan_table`'s,
`response_curves`' and `plan_flighting`'s ends are those draws, though the float sums of the
weights and the shares land a unit in the last place either side of their exact values. A draw of
weight 0 is never an end, and a quantity NaN at any draw is NaN.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from mixdyn import NATIONAL, Cadence, Channel, MediaPanel, Revenue
from mixdyn.decisions import (
    Range,
    Readings,
    Worths,
    plan_breakdown,
    plan_flighting,
    plan_table,
    response_curves,
)
from mixdyn.decisions._quantile import weighted_quantiles

TV = Channel("tv")
CASES = [(draws, level) for draws in (20, 100) for level in ("0.5", "0.9")]


def _exact(values: np.ndarray, weights: list[Fraction], share: Fraction) -> float:
    """The least value whose cumulative weight, in exact arithmetic, reaches `share` of the
    total."""
    total = sum(weights, Fraction(0))
    cumulative = Fraction(0)
    for i in np.argsort(values, kind="stable"):
        cumulative += weights[i]
        if weights[i] > 0 and cumulative >= share * total:
            return float(values[i])
    raise AssertionError


def _ends(values: np.ndarray, level: str) -> tuple[float, float]:
    """The exact ends of the central `level` interval of `values` at equal weights."""
    weights = [Fraction(1, values.size)] * values.size
    exact = Fraction(level)
    return (
        _exact(values, weights, (1 - exact) / 2),
        _exact(values, weights, (1 + exact) / 2),
    )


def _readings(draws: int, seed: int) -> Readings:
    """`draws` Hill draws of one channel, each weighted alike."""
    rng = np.random.default_rng(seed)
    return Readings(
        channels=(TV,),
        retention=rng.uniform(0.0, 0.6, (draws, 1)),
        half_saturation=rng.uniform(5.0, 15.0, (draws, 1)),
        slope=rng.uniform(0.6, 2.0, (draws, 1)),
        coefficient=rng.uniform(20.0, 60.0, (draws, 1)),
        tanh=np.zeros(draws, dtype=bool),
        fit=np.zeros(draws, dtype=np.intp),
        chain=np.zeros(draws, dtype=np.intp),
        stacking=np.ones(1),
        pointwise=None,
        length=3,
    )


def _panel(weeks: int = 8) -> MediaPanel:
    return MediaPanel(
        dates=np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(weeks),
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=(TV,),
        controls=(),
        kpi=np.ones((1, weeks)),
        kpi_unit=Revenue(),
        spend=np.random.default_rng(3).uniform(2.0, 12.0, (1, weeks, 1)),
        currency="EUR",
        control_values=np.zeros((1, weeks, 0)),
        filled=np.zeros((1, weeks), dtype=bool),
    )


@pytest.mark.parametrize(("draws", "level"), CASES)
def test_a_plan_table_s_ends_are_the_exact_weighted_quantiles(draws: int, level: str) -> None:
    """The KPI change's, the marginal returns' and the breakdown factor's ends, the range a point
    so that no diminishing-returns point is read."""
    readings, limits = _readings(draws, draws), {TV: Range(0.0, 0.0)}
    table = plan_table(_panel(), NATIONAL, readings, [9.0], 4, level=float(level), limits=limits)
    for spread in (table.kpi, table.marginal_current, table.marginal_planned, table.total_kpi):
        column = spread.draws.reshape(draws, -1)[:, 0]
        ends = (float(np.ravel(spread.low)[0]), float(np.ravel(spread.high)[0]))
        assert ends == _ends(column, level)
    factors = plan_breakdown(_panel(), NATIONAL, readings, [9.0], 4).draws
    assert (table.factor_low, table.factor_high) == _ends(factors, level)


@pytest.mark.parametrize(("draws", "level"), CASES)
def test_a_response_curve_s_band_is_the_exact_weighted_quantiles(draws: int, level: str) -> None:
    rng = np.random.default_rng(draws)
    worths = Worths(np.linspace(0.0, 20.0, 5)[None], rng.uniform(0.0, 50.0, (draws, 1, 5)))
    curves = response_curves(worths, _readings(draws, draws), level=float(level))
    for g in range(5):
        column = worths.table[:, 0, g]
        assert (float(curves.low[0, g]), float(curves.high[0, g])) == _ends(column, level)


@pytest.mark.parametrize(("draws", "level"), CASES)
def test_a_flighted_plan_s_intervals_are_the_exact_weighted_quantiles(
    draws: int, level: str
) -> None:
    limits = {TV: Range(0.0, 12.0)}
    readings = _readings(draws, draws)
    plan = plan_flighting(_panel(), NATIONAL, readings, 30.0, 4, limits, level=float(level))
    for k in range(3):
        assert tuple(plan.intervals[k]) == _ends(plan.gains[k], level)
    assert tuple(plan.edge_interval) == _ends(plan.edge, level)


def test_a_draw_of_weight_0_is_never_an_end_and_nan_reads_nan() -> None:
    values = np.array([[1.0, 2.0], [5.0, np.nan], [100.0, 3.0]])
    weights = np.array([0.5, 0.0, 0.5])
    low, high = weighted_quantiles(values, weights, (0.5, 0.75))
    np.testing.assert_array_equal(low, [1.0, np.nan])
    np.testing.assert_array_equal(high, [100.0, np.nan])
