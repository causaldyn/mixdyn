"""mixdyn.decisions.plan_table: a plan read channel by channel, each return `returns` at a draw.

Each draw's marginal return is `returns`'s on a channel built here from the draw's parameters, on
the history followed by the plan's periods at the rate, never the draw's coefficient. The
diminishing-returns point is where the mean of those returns crosses breakeven, `1 / margin` on
profit and 1 on revenue. A channel the plan holds changes the KPI by 0 at every draw; the KPI
changes sum to `plan_breakdown`'s expected gain and the revenue column is the KPI column times its
rate. The intervals are the weighted quantiles a brute force reads off the draws' weights.

The point's means are read at every draw and spend at once, in closed form: each is the weighted
`math.fsum` of `returns`' marginal returns to `1e-12` of it, and the point is the one the eager
scan read, one `returns` call a draw and a spend, to the bit.
"""

from __future__ import annotations

import importlib
import math

import numpy as np
import pytest
from chc.response import Channel as Response
from chc.response import GeometricAdstock, Hill, Tanh

from mixdyn import NATIONAL, Cadence, Channel, MediaPanel, Revenue
from mixdyn.decisions import Range, Readings, plan_breakdown, returns
from mixdyn.decisions.plan_table import (
    _SCAN,
    _TOLERANCE,
    _mean_marginals,
    plan_table,
    ranges_from_current,
)

TANH_HALF = math.atanh(0.5)
table_module = importlib.import_module("mixdyn.decisions.plan_table")
TV, SEARCH, SOCIAL = Channel("tv"), Channel("search"), Channel("social")
CHANNELS = (TV, SEARCH, SOCIAL)
PERIODS = 8


def _panel(history: np.ndarray) -> MediaPanel:
    weeks = history.shape[0]
    return MediaPanel(
        dates=np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(weeks),
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=CHANNELS[: history.shape[1]],
        controls=(),
        kpi=np.ones((1, weeks)),
        kpi_unit=Revenue(),
        spend=history[None],
        currency="EUR",
        control_values=np.zeros((1, weeks, 0)),
        filled=np.zeros((1, weeks), dtype=bool),
    )


def _history(weeks: int = 16) -> np.ndarray:
    return np.random.default_rng(7).uniform(2.0, 12.0, (weeks, 3))


def _hill() -> Readings:
    """Three Hill draws of two fits, weighted 0.25 and 0.75: the draws weigh 0.125, 0.125, 0.75."""
    rng = np.random.default_rng(20261028)
    return Readings(
        channels=CHANNELS,
        retention=rng.uniform(0.2, 0.7, (3, 3)),
        half_saturation=rng.uniform(5.0, 15.0, (3, 3)),
        slope=rng.uniform(0.6, 1.0, (3, 3)),
        coefficient=rng.uniform(20.0, 60.0, (3, 3)),
        tanh=np.zeros(3, dtype=bool),
        fit=np.array([0, 0, 1]),
        chain=np.array([0, 1, 0]),
        stacking=np.array([0.25, 0.75]),
        pointwise=None,
        length=6,
    )


def _mixed() -> Readings:
    """Two Hill draws of one fit and a `tanh` draw of another, weighted 0.25 and 0.75."""
    rng = np.random.default_rng(20261013)
    return Readings(
        channels=CHANNELS,
        retention=rng.uniform(0.2, 0.7, (3, 3)),
        half_saturation=rng.uniform(5.0, 15.0, (3, 3)),
        slope=np.vstack([rng.uniform(0.8, 2.5, (2, 3)), np.full((1, 3), np.nan)]),
        coefficient=rng.uniform(20.0, 60.0, (3, 3)),
        tanh=np.array([False, False, True]),
        fit=np.array([0, 0, 1]),
        chain=np.array([0, 1, 0]),
        stacking=np.array([0.25, 0.75]),
        pointwise=None,
        length=6,
    )


def _response(readings: Readings, s: int, c: int) -> Response:
    kernel = GeometricAdstock(readings.retention[s, c], length=readings.length, normalized=True)
    half = readings.half_saturation[s, c]
    curve = Tanh(half / TANH_HALF) if readings.tanh[s] else Hill(half, readings.slope[s, c])
    return Response(kernel, curve, readings.coefficient[s, c])


def _marginal(readings: Readings, s: int, c: int, rate: float, revenue_per_kpi: float) -> float:
    history = _history()[:, c]
    series = np.concatenate([history, np.full(PERIODS, rate)])
    window = slice(history.size, history.size + PERIODS)
    reading = returns(_response(readings, s, c), series, window, revenue_per_kpi=revenue_per_kpi)
    return reading.marginal_roi


def _mean_marginal(readings: Readings, c: int, rate: float, revenue_per_kpi: float = 1.0) -> float:
    weights = readings.weights
    return sum(
        weights[s] * _marginal(readings, s, c, rate, revenue_per_kpi) for s in range(weights.size)
    )


def _quantile(values: np.ndarray, weights: np.ndarray, share: float) -> float:
    """The least value whose weight, with every value's at or below it, reaches `share`."""
    held = values[weights > 0.0]
    return min(v for v in held if weights[values <= v].sum() >= share * weights.sum())


SPEND = np.array([9.0, 7.0, 2.0])


def test_each_marginal_return_is_returns_at_the_draw_and_not_its_coefficient() -> None:
    """On Hill channels each draw's marginal return at the current and the planned spend is
    `returns(...).marginal_roi` of the draw's channel to 1e-12, at each channel's revenue per KPI,
    and is not the draw's coefficient, nor the coefficient at that revenue."""
    readings = _hill()
    rates = {TV: 1.0, SEARCH: 2.5, SOCIAL: 0.4}
    panel = _panel(_history())
    table = plan_table(panel, NATIONAL, readings, SPEND, PERIODS, revenue_per_kpi=rates)
    for c, channel in enumerate(CHANNELS):
        for spread, rate in (
            (table.marginal_current, table.current[c]),
            (table.marginal_planned, table.spend[c]),
        ):
            for s in range(3):
                expected = _marginal(readings, s, c, rate, rates[channel])
                assert spread.draws[s, c] == pytest.approx(expected, rel=1e-12, abs=0.0)
                coefficient = readings.coefficient[s, c]
                assert abs(spread.draws[s, c] - coefficient) > 1.0
                assert abs(spread.draws[s, c] - coefficient * rates[channel]) > 1.0
            assert spread.mean[c] == pytest.approx(
                readings.weights @ spread.draws[:, c], rel=1e-15, abs=0.0
            )


def test_a_marginal_return_at_no_spend_is_not_read() -> None:
    readings = _hill()
    panel = _panel(_history())
    table = plan_table(panel, NATIONAL, readings, [9.0, 0.0, 2.0], PERIODS)
    assert np.all(np.isnan(table.marginal_planned.draws[:, 1]))
    assert math.isnan(table.marginal_planned.mean[1])
    assert math.isnan(table.marginal_planned.low[1])
    assert np.all(np.isfinite(table.marginal_planned.draws[:, [0, 2]]))


@pytest.mark.parametrize("margin", [None, 0.4])
def test_the_diminishing_returns_point_is_where_the_mean_marginal_return_meets_breakeven(
    margin: float | None,
) -> None:
    """With a margin `m` the point is where the mean marginal return falls to `1 / m`, and to 1
    without one: above it just before the point and below it just after, within the bisection's
    tolerance of the range. The two points differ."""
    readings = _hill()
    panel = _panel(_history())
    limits = {c: Range(0.0, 80.0) for c in CHANNELS}
    table = plan_table(panel, NATIONAL, readings, SPEND, PERIODS, margin=margin, limits=limits)
    breakeven = 1.0 if margin is None else 1.0 / margin
    assert table.breakeven == breakeven
    step = _TOLERANCE * 80.0
    for c in range(3):
        point = table.diminishing[c]
        assert point is not None
        assert _mean_marginal(readings, c, point - step) >= breakeven
        assert _mean_marginal(readings, c, point + step) < breakeven
        assert _mean_marginal(readings, c, point) == pytest.approx(breakeven, rel=1e-6, abs=0.0)
    other = plan_table(
        panel, NATIONAL, readings, SPEND, PERIODS, margin=None if margin else 0.4, limits=limits
    )
    for theirs, ours in zip(other.diminishing, table.diminishing, strict=True):
        assert theirs is not None
        assert ours is not None
        assert abs(theirs - ours) > 1.0


def _eager_mean(
    readings: Readings, c: int, history: np.ndarray, rate: float, revenue_per_kpi: float
) -> float:
    """The posterior-mean marginal return as the table read it before its means were batched: the
    weighted `math.fsum` of `returns`' marginal return at each draw of weight, a call a draw."""
    weights = readings.weights
    return math.fsum(
        weights[s] * table_module._marginal(readings, s, c, history, rate, PERIODS, revenue_per_kpi)
        for s in np.flatnonzero(weights > 0.0).tolist()
    )


def _eager_point(
    readings: Readings,
    c: int,
    history: np.ndarray,
    revenue_per_kpi: float,
    breakeven: float,
    least: float,
    most: float,
) -> float | None:
    """The diminishing-returns point as the table read it before its means were batched: the same
    scan and bisection on `_eager_mean`."""
    spends = np.linspace(least, most, _SCAN)
    spends = spends[spends > 0.0]
    if spends.size == 0:
        return None
    below = np.array(
        [_eager_mean(readings, c, history, float(x), revenue_per_kpi) < breakeven for x in spends]
    )
    if not below[-1] or below.all():
        return None
    first = int(np.flatnonzero(~below)[-1]) + 1
    low, high = float(spends[first - 1]), float(spends[first])
    while high - low > _TOLERANCE * (most - least):
        middle = 0.5 * (low + high)
        if middle <= low or middle >= high:
            break
        if _eager_mean(readings, c, history, middle, revenue_per_kpi) < breakeven:
            high = middle
        else:
            low = middle
    return 0.5 * (low + high)


def _zero_weighted() -> Readings:
    """`_mixed`'s draws with the first fit weighted 0: its two Hill draws are never read."""
    mixed = _mixed()
    return Readings(
        channels=mixed.channels,
        retention=mixed.retention,
        half_saturation=mixed.half_saturation,
        slope=mixed.slope,
        coefficient=mixed.coefficient,
        tanh=mixed.tanh,
        fit=mixed.fit,
        chain=mixed.chain,
        stacking=np.array([0.0, 1.0]),
        pointwise=None,
        length=mixed.length,
    )


@pytest.mark.parametrize(
    "readings", [_hill(), _mixed(), _zero_weighted()], ids=["hill", "mixed", "zero-weighted"]
)
@pytest.mark.parametrize("revenue_per_kpi", [1.0, 2.5])
def test_the_batched_mean_marginal_return_is_returns_s_to_1e_12(
    readings: Readings, revenue_per_kpi: float
) -> None:
    """At 41 spends a period from 0.05 to 200, past each channel's half-saturation point and the
    history's most, each batched mean is the eager weighted `math.fsum` of `returns`' marginal
    returns to `1e-12` of it: the closed form's sums run in another order than JAX's jvp."""
    history = _history()
    rates = np.geomspace(0.05, 200.0, 41)
    for c in range(3):
        batched = _mean_marginals(readings, c, history[:, c], rates, PERIODS, revenue_per_kpi)
        eager = [_eager_mean(readings, c, history[:, c], float(x), revenue_per_kpi) for x in rates]
        assert batched == pytest.approx(eager, rel=1e-12, abs=0.0)


@pytest.mark.parametrize(
    "readings", [_hill(), _mixed(), _zero_weighted()], ids=["hill", "mixed", "zero-weighted"]
)
@pytest.mark.parametrize(
    ("margin", "least", "most"),
    [(None, 0.0, 80.0), (0.4, 0.0, 80.0), (0.25, 1.0, 20.0), (0.01, 1.0, 20.0), (None, 1.0, 2.0)],
)
def test_the_point_is_the_eager_scan_s_to_the_bit(
    readings: Readings, margin: float | None, least: float, most: float
) -> None:
    """The point read on the batched means is the one the eager scan and bisection read, to the
    bit, at each breakeven and range here: the bisection reads only which side of breakeven each
    mean falls, and no mean here is within `1e-12` of it. Each of the first three rows holds a
    point of at least one channel, 15 over the three readings; the last two hold none."""
    history = _history()
    panel = _panel(history)
    rates = {TV: 1.0, SEARCH: 2.5, SOCIAL: 0.4}
    limits = {c: Range(least, most) for c in CHANNELS}
    table = plan_table(
        panel,
        NATIONAL,
        readings,
        SPEND,
        PERIODS,
        revenue_per_kpi=rates,
        margin=margin,
        limits=limits,
    )
    breakeven = 1.0 if margin is None else 1.0 / margin
    eager = tuple(
        _eager_point(readings, c, history[:, c], rates[channel], breakeven, least, most)
        for c, channel in enumerate(CHANNELS)
    )
    assert table.diminishing == eager
    if (margin, most) in ((0.01, 20.0), (None, 2.0)):
        assert eager == (None, None, None)
    else:
        assert any(point is not None for point in eager)


def test_the_default_range_s_point_is_the_eager_scan_s_to_the_bit() -> None:
    readings = _mixed()
    history = _history()
    panel = _panel(history)
    plan, against = np.array([9.0, 3.0, 2.0]), np.array([9.0, 4.0, 2.0])
    table = plan_table(panel, NATIONAL, readings, plan, PERIODS, against=against)
    most = np.max(np.vstack([history, plan, against]), axis=0)
    eager = tuple(
        _eager_point(readings, c, history[:, c], 1.0, 1.0, 0.0, float(most[c])) for c in range(3)
    )
    assert table.diminishing == eager


def test_the_point_calls_returns_at_no_draw(monkeypatch: pytest.MonkeyPatch) -> None:
    """Counted on `_hill`'s 3 draws, 3 channels and the range 0 to 80: the table calls `returns`
    18 times, at each draw and channel at the current and the planned spend, and none for the
    diminishing-returns point. Its means are read 25 times a channel, once at the scan's 63
    spends above 0 and once at each of the bisection's 24 halvings, from `80 / 63` to `1e-9` of
    the range. The eager point called `returns` once a draw at each of those 87 spends, 783
    times over the 3 channels and 801 in all."""
    calls = {"returns": 0, "means": 0, "rates": 0}
    eager_returns, batched = table_module.returns, table_module._mean_marginals

    def counted_returns(*args: object, **kwargs: object) -> object:
        calls["returns"] += 1
        return eager_returns(*args, **kwargs)  # ty: ignore[invalid-argument-type]

    def counted_means(
        readings: Readings,
        c: int,
        history: np.ndarray,
        rates: np.ndarray,
        periods: int,
        revenue_per_kpi: float,
    ) -> np.ndarray:
        calls["means"] += 1
        calls["rates"] += rates.size
        return batched(readings, c, history, rates, periods, revenue_per_kpi)

    monkeypatch.setattr(table_module, "returns", counted_returns)
    monkeypatch.setattr(table_module, "_mean_marginals", counted_means)
    limits = {c: Range(0.0, 80.0) for c in CHANNELS}
    table = plan_table(_panel(_history()), NATIONAL, _hill(), SPEND, PERIODS, limits=limits)
    assert all(point is not None for point in table.diminishing)
    assert calls == {"returns": 2 * 3 * 3, "means": 3 * (1 + 24), "rates": 3 * (63 + 24)}


def test_the_point_is_none_where_the_mean_never_falls_to_breakeven_in_the_range() -> None:
    readings = _hill()
    panel = _panel(_history())
    narrow = plan_table(
        panel, NATIONAL, readings, SPEND, PERIODS, limits={c: Range(1.0, 2.0) for c in CHANNELS}
    )
    assert narrow.diminishing == (None, None, None)
    assert all(_mean_marginal(readings, c, 2.0) >= 1.0 for c in range(3))
    beyond = plan_table(
        panel,
        NATIONAL,
        readings,
        SPEND,
        PERIODS,
        margin=0.01,
        limits={c: Range(1.0, 20.0) for c in CHANNELS},
    )
    assert beyond.diminishing == (None, None, None)
    assert all(_mean_marginal(readings, c, 1.0) < 100.0 for c in range(3))


def test_the_default_range_reaches_the_most_the_history_spent() -> None:
    """Search planned at 3 a period against 4, where the history spent up to 11.96: its point,
    7.07, lies past both, so a range that stopped at the plan and the reference would hold none.
    Every channel's point is the one read on the range from 0 to that most."""
    readings = _hill()
    history = _history()
    panel = _panel(history)
    plan, against = np.array([9.0, 3.0, 2.0]), np.array([9.0, 4.0, 2.0])
    table = plan_table(panel, NATIONAL, readings, plan, PERIODS, against=against)
    most = np.max(np.vstack([history, plan, against]), axis=0)
    assert most[1] == history[:, 1].max() > 7.0
    limits = {c: Range(0.0, float(m)) for c, m in zip(CHANNELS, most, strict=True)}
    ranged = plan_table(panel, NATIONAL, readings, plan, PERIODS, against=against, limits=limits)
    assert table.diminishing == ranged.diminishing
    point = table.diminishing[1]
    assert point is not None
    assert 4.0 < point < most[1]


@pytest.mark.parametrize("readings", [_hill(), _mixed()], ids=["hill", "mixed"])
def test_a_channel_the_plan_holds_changes_the_kpi_by_nothing_at_any_draw(
    readings: Readings,
) -> None:
    panel = _panel(_history())
    current = plan_table(panel, NATIONAL, readings, SPEND, PERIODS).current
    held = np.array([SPEND[0], current[1], SPEND[2]])
    table = plan_table(panel, NATIONAL, readings, held, PERIODS)
    np.testing.assert_array_equal(table.kpi.draws[:, 1], 0.0)
    np.testing.assert_array_equal(table.revenue.draws[:, 1], 0.0)
    assert table.kpi.mean[1] == 0.0
    assert table.change[1] == 0.0
    against = np.array([4.0, 5.0, 6.0])
    named = plan_table(panel, NATIONAL, readings, [8.0, 5.0, 1.0], PERIODS, against=against)
    np.testing.assert_array_equal(named.kpi.draws[:, 1], 0.0)
    assert np.all(named.kpi.draws[:, [0, 2]] != 0.0)


@pytest.mark.parametrize("readings", [_hill(), _mixed()], ids=["hill", "mixed"])
def test_the_kpi_changes_sum_to_the_plans_expected_gain_and_revenue_is_kpi_at_its_rate(
    readings: Readings,
) -> None:
    panel = _panel(_history())
    rates = {TV: 1.5, SEARCH: 0.8, SOCIAL: 3.0}
    table = plan_table(panel, NATIONAL, readings, SPEND, PERIODS, revenue_per_kpi=rates)
    reading = plan_breakdown(panel, NATIONAL, readings, SPEND, PERIODS)
    gain = float(reading.expected.sum())
    assert float(table.kpi.mean.sum()) == pytest.approx(gain, rel=1e-12, abs=1e-12)
    assert float(table.total_kpi.mean) == pytest.approx(gain, rel=1e-12, abs=1e-12)
    np.testing.assert_array_equal(table.kpi.draws, reading.gains)
    column = np.array([rates[c] for c in CHANNELS])
    np.testing.assert_array_equal(table.revenue_per_kpi, column)
    np.testing.assert_array_equal(table.revenue.draws, table.kpi.draws * column)
    np.testing.assert_array_equal(table.revenue.mean, table.kpi.mean * column)
    np.testing.assert_array_equal(table.revenue.low, table.kpi.low * column)
    np.testing.assert_array_equal(table.revenue.high, table.kpi.high * column)
    np.testing.assert_array_equal(table.total_revenue.draws, (table.kpi.draws * column).sum(1))
    np.testing.assert_array_equal(table.current, reading.against)
    assert table.factor == reading.factor


@pytest.mark.parametrize("level", [0.5, 0.9, 0.99])
def test_the_intervals_are_the_draws_weighted_quantiles(level: float) -> None:
    """Each end is the least draw whose cumulative weight reaches its share, a brute force over
    the draws; a draw of weight 0 is never an end."""
    readings = _mixed()
    panel = _panel(_history())
    table = plan_table(panel, NATIONAL, readings, SPEND, PERIODS, level=level)
    weights = readings.weights
    low, high = (1.0 - level) / 2.0, (1.0 + level) / 2.0
    for spread in (table.kpi, table.revenue, table.marginal_current, table.marginal_planned):
        for c in range(3):
            column = spread.draws[:, c]
            assert spread.low[c] == _quantile(column, weights, low)
            assert spread.high[c] == _quantile(column, weights, high)
    factors = plan_breakdown(panel, NATIONAL, readings, SPEND, PERIODS).draws
    assert table.factor_low == _quantile(factors, weights, low)
    assert table.factor_high == _quantile(factors, weights, high)
    unweighted = Readings(
        channels=readings.channels,
        retention=readings.retention,
        half_saturation=readings.half_saturation,
        slope=readings.slope,
        coefficient=readings.coefficient,
        tanh=readings.tanh,
        fit=readings.fit,
        chain=readings.chain,
        stacking=np.array([0.0, 1.0]),
        pointwise=None,
        length=readings.length,
    )
    alone = plan_table(panel, NATIONAL, unweighted, SPEND, PERIODS, level=level)
    np.testing.assert_array_equal(alone.kpi.low, alone.kpi.draws[2])
    np.testing.assert_array_equal(alone.kpi.high, alone.kpi.draws[2])


def test_a_change_from_no_current_spend_is_not_a_percentage() -> None:
    history = _history()
    history[-PERIODS:, 1:] = 0.0
    panel = _panel(history)
    table = plan_table(panel, NATIONAL, _hill(), [9.0, 3.0, 0.0], PERIODS)
    assert table.current[1] == table.current[2] == 0.0
    assert math.isnan(table.change[1])
    assert table.change[2] == 0.0
    assert table.change[0] == pytest.approx(
        100.0 * (9.0 - table.current[0]) / table.current[0], rel=1e-15, abs=0.0
    )
    row = table.rows()[0]
    assert row["channel"] == TV
    assert row["kpi_change"] == table.kpi.mean[0]


@pytest.mark.parametrize(
    ("keywords", "match"),
    [
        ({"margin": 0.0}, "margin"),
        ({"margin": 1.5}, "margin"),
        ({"level": 0.0}, "level"),
        ({"level": 1.0}, "level"),
        ({"revenue_per_kpi": 0.0}, "revenue per KPI"),
        ({"revenue_per_kpi": {TV: 1.0, SEARCH: 1.0}}, "one rate a channel"),
        ({"limits": {TV: Range(0.0, 5.0)}}, "range"),
    ],
)
def test_the_table_refuses(keywords: dict[str, object], match: str) -> None:
    panel = _panel(_history())
    with pytest.raises(ValueError, match=match):
        plan_table(panel, NATIONAL, _hill(), SPEND, PERIODS, **keywords)  # ty: ignore[invalid-argument-type]


def test_ranges_from_current_state_a_share_and_a_multiple_of_the_current_spend() -> None:
    ranges = ranges_from_current(
        {TV: 10.0, SEARCH: 4.0, SOCIAL: 0.0},
        least_share=0.5,
        most_multiple=2.0,
        floors={SEARCH: 3.0},
    )
    assert ranges == {TV: Range(5.0, 20.0), SEARCH: Range(3.0, 8.0), SOCIAL: Range(0.0, 0.0)}


@pytest.mark.parametrize(
    ("keywords", "match"),
    [
        ({"least_share": -0.1, "most_multiple": 2.0}, "share"),
        ({"least_share": 1.1, "most_multiple": 2.0}, "share"),
        ({"least_share": 0.5, "most_multiple": 0.9}, "multiple"),
        ({"least_share": 0.5, "most_multiple": math.inf}, "multiple"),
        ({"least_share": 0.5, "most_multiple": 2.0, "floors": {TV: 21.0}}, "floor"),
        ({"least_share": 0.5, "most_multiple": 2.0, "floors": {SOCIAL: 1.0}}, "no current"),
    ],
)
def test_ranges_from_current_refuse(keywords: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ranges_from_current({TV: 10.0, SEARCH: 4.0}, **keywords)  # ty: ignore[invalid-argument-type]
