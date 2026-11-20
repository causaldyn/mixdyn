"""mixdyn.decisions.flighting: a schedule planned at the posterior's median, scored at every draw.

An unchanged plan gains exactly nothing over the status quo at every draw, the history carried
into both; on a concave curve through a one-period kernel the schedule is never called flighted
and returns at least the best rate a period; the schedule spends the whole budget; each draw's
score reads that draw's retention, and the schedule the median's; chc's warning that a schedule
is the best of local ascents reaches the caller. Where pulsing pays, on an S-shaped curve, the
schedule is called flighted.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from chc.allocation import allocate
from chc.response import Channel as Response
from chc.response import GeometricAdstock, Hill, Tanh
from hypothesis import given, settings
from hypothesis import strategies as st

from mixdyn import NATIONAL, Cadence, Channel, MediaPanel, Revenue
from mixdyn.decisions import Range, Readings
from mixdyn.decisions.flighting import _interval, _quantile, plan_flighting

TANH_HALF = math.atanh(0.5)
TV, SEARCH = Channel("tv"), Channel("search")


def _panel(history: np.ndarray) -> MediaPanel:
    """One national geo whose spend is `history`, `(weeks, channels)`, on tv and search."""
    weeks, channels = history.shape
    return MediaPanel(
        dates=np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(weeks),
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=(TV, SEARCH)[:channels],
        controls=(),
        kpi=np.ones((1, weeks)),
        kpi_unit=Revenue(),
        spend=history[None].astype(np.float64),
        currency="EUR",
        control_values=np.zeros((1, weeks, 0)),
        filled=np.zeros((1, weeks), dtype=bool),
    )


def _readings(length: int, **parameters: Any) -> Readings:
    """One fit's draws, each weighted alike, of Hill curves unless `tanh` says otherwise."""
    retention = np.array(parameters["retention"], dtype=np.float64)
    draws, channels = retention.shape
    tanh = bool(parameters.get("tanh", False))
    return Readings(
        channels=(TV, SEARCH)[:channels],
        retention=retention,
        half_saturation=np.array(parameters["half"], dtype=np.float64),
        slope=np.full((draws, channels), np.nan) if tanh else np.array(parameters["slope"]),
        coefficient=np.array(parameters["coefficient"], dtype=np.float64),
        tanh=np.full(draws, tanh),
        fit=np.zeros(draws, dtype=np.intp),
        chain=np.zeros(draws, dtype=np.intp),
        stacking=np.ones(1),
        pointwise=None,
        length=length,
    )


def _column(values: list[float]) -> np.ndarray:
    return np.array(values, dtype=np.float64)[:, None]


def _response(readings: Readings, s: int, c: int) -> Response:
    """Draw `s`'s channel `c` as `chc.response` reads it."""
    kernel = GeometricAdstock(readings.retention[s, c], length=readings.length, normalized=True)
    half = readings.half_saturation[s, c]
    curve = Tanh(half / TANH_HALF) if readings.tanh[s] else Hill(half, readings.slope[s, c])
    return Response(kernel, curve, readings.coefficient[s, c])


def _worth(channel: Response, history: np.ndarray, plan: np.ndarray) -> float:
    """`channel`'s return over `plan`'s periods and the kernel's length after them, `history`
    carried in, by `chc.response`."""
    after = np.zeros(channel.kernel.length - 1)
    series = jnp.asarray(np.concatenate([history, plan, after]))
    adstock = channel.kernel(series)[history.size :]
    return float(channel.coefficient * jnp.sum(channel.curve(adstock)))


def test_an_unchanged_plan_gains_exactly_nothing_over_the_status_quo_at_every_draw() -> None:
    # every period at the current 100, the range a point: the schedule, the one rate and the
    # status quo are one plan, so each draw scores them alike, the history carried into all three
    reading = _readings(
        4,
        retention=_column([0.3, 0.4, 0.5, 0.6, 0.7]),
        half=np.full((5, 1), 100.0),
        slope=np.ones((5, 1)),
        coefficient=_column([1.0, 2.0, 3.0, 4.0, 5.0]),
    )
    plan = plan_flighting(
        _panel(np.full((12, 1), 100.0)), NATIONAL, reading, 400.0, 4, {TV: Range(100.0, 100.0)}
    )
    np.testing.assert_array_equal(plan.schedule, np.full((4, 1), 100.0))
    np.testing.assert_array_equal(plan.current, [100.0])
    np.testing.assert_array_equal(plan.gains[0] - plan.gains[2], np.zeros(5))
    np.testing.assert_array_equal(plan.gains[1] - plan.gains[2], np.zeros(5))
    np.testing.assert_array_equal(plan.edge, np.zeros(5))
    assert plan.recommendation == "continuous"
    # what the status quo's spend adds is what it returns less what the carry-in alone returns
    history = np.full(12, 100.0)
    for s in range(5):
        channel = _response(reading, s, 0)
        alone = _worth(channel, history, np.zeros(4))
        added = _worth(channel, history, np.full(4, 100.0)) - alone
        assert plan.gains[2, s] == pytest.approx(added, rel=1e-12, abs=0.0)
        assert plan.gains[2, s] > 0.0


def _root(coefficient: list[float]) -> Readings:
    """`c sqrt(x)` read by a Hill of slope 1/2 whose half-saturation, 1e12, is far past the spend:
    `c sqrt(x) / (1 + sqrt(x) / 1e6)`; no carryover."""
    draws = len(coefficient)
    return _readings(
        1,
        retention=np.zeros((draws, 1)),
        half=np.full((draws, 1), 1e12),
        slope=np.full((draws, 1), 0.5),
        coefficient=_column(coefficient),
    )


def test_a_concave_curve_is_never_pulsed_and_its_schedule_returns_the_continuous_plans() -> None:
    def root(x: float) -> float:
        return 1e7 * math.sqrt(x) / (math.sqrt(x) + 1e6)

    continuous, pulse = 8 * root(55.0), 4 * root(10.0) + 4 * root(100.0)
    # 10 sqrt(x): 8 x 10 sqrt(55) = 593.3 continuous, 4 x 10 sqrt(10) + 4 x 10 sqrt(100) = 526.5
    assert round(8 * 10 * math.sqrt(55.0), 1) == 593.3
    assert round(40 * math.sqrt(10.0) + 400.0, 1) == 526.5
    assert (round(continuous, 1), round(pulse, 1)) == (593.3, 526.5)
    reading = _root([0.8e7, 1e7, 1.2e7])
    plan = plan_flighting(
        _panel(np.zeros((10, 1))), NATIONAL, reading, 440.0, 8, {TV: Range(10.0, 100.0)}
    )
    assert plan.recommendation == "continuous"
    assert plan.worth >= continuous * (1.0 - 1e-9)
    np.testing.assert_allclose(plan.continuous, [55.0], rtol=1e-12)
    pulsed = np.repeat([10.0, 100.0], 4)
    for s, scale in enumerate((0.8, 1.0, 1.2)):
        assert plan.gains[0, s] >= scale * continuous * (1.0 - 1e-9)
        assert plan.gains[0, s] > _worth(_response(reading, s, 0), np.zeros(10), pulsed)


def _s_shaped(length: int) -> Readings:
    """An S-shaped Hill, slope 3 and half-saturation 1, at three draws."""
    return _readings(
        length,
        retention=np.full((3, 1), 0.5),
        half=_column([0.9, 1.0, 1.1]),
        slope=np.full((3, 1), 3.0),
        coefficient=_column([0.9, 1.0, 1.1]),
    )


def test_a_pulse_is_flighted_where_every_draw_says_it_pays() -> None:
    # chc's own case: 1.2 over four periods returns 0.105 at one rate a period, 0.633 in one
    plan = plan_flighting(
        _panel(np.zeros((6, 1))), NATIONAL, _s_shaped(1), 1.2, 4, {TV: Range(0.0, 2.0)}
    )
    assert plan.recommendation == "flighted"
    assert plan.edge_interval[0] > 0.1
    assert np.count_nonzero(plan.schedule > 1e-9) < 4
    assert plan.expected[0] > plan.expected[1] > 0.0
    assert plan.limit is None


def test_the_schedule_spends_the_whole_budget_the_box_allows() -> None:
    reading = _readings(
        2,
        retention=np.full((2, 2), 0.3),
        half=np.full((2, 2), 50.0),
        slope=np.full((2, 2), 0.8),
        coefficient=np.array([[2.0, 1.0], [3.0, 1.5]]),
    )
    limits = {TV: Range(0.0, 100.0), SEARCH: Range(0.0, 80.0)}
    history = np.full((8, 2), 20.0)
    for budget in (500.0, 1000.0, 1080.0):
        plan = plan_flighting(_panel(history), NATIONAL, reading, budget, 6, limits)
        assert plan.schedule.shape == (6, 2)
        assert plan.schedule.sum() == pytest.approx(budget, rel=1e-12, abs=0.0)
        assert 6 * plan.continuous.sum() == pytest.approx(budget, rel=1e-12, abs=0.0)
        assert plan.budget == budget
        assert np.all(plan.schedule <= [100.0, 80.0])
        assert np.all(plan.schedule >= 0.0)


def _decaying(*, tanh: bool = False) -> Readings:
    """A channel whose draws keep 0.2, 0.8 and 0.9 a period; the median keeps 0.8."""
    return _readings(
        4,
        retention=_column([0.2, 0.8, 0.9]),
        half=np.full((3, 1), 40.0),
        slope=np.full((3, 1), 1.0),
        coefficient=_column([5.0, 5.0, 5.0]),
        tanh=tanh,
    )


@pytest.mark.parametrize("tanh", [False, True])
def test_each_draws_score_reads_its_own_retention(tanh: bool) -> None:
    reading = _decaying(tanh=tanh)
    history = np.array([0.0, 10.0, 80.0, 30.0, 60.0, 20.0])
    plan = plan_flighting(
        _panel(history[:, None]), NATIONAL, reading, 160.0, 4, {TV: Range(0.0, 100.0)}
    )
    grids = (plan.schedule[:, 0], np.repeat(plan.continuous, 4), np.repeat(plan.current, 4))
    for s in range(3):
        channel = _response(reading, s, 0)
        alone = _worth(channel, history, np.zeros(4))
        for k, grid in enumerate(grids):
            added = _worth(channel, history, grid) - alone
            assert plan.gains[k, s] == pytest.approx(added, rel=1e-12, abs=1e-12)


def test_the_schedule_is_chcs_at_the_medians_retention(caplog: pytest.LogCaptureFixture) -> None:
    reading = _decaying()
    history = np.array([0.0, 10.0, 80.0, 30.0, 60.0, 20.0])
    median = Response(GeometricAdstock(0.8, length=4, normalized=True), Hill(40.0, 1.0), 5.0)
    with caplog.at_level(logging.WARNING):
        expected = allocate(
            [median],
            160.0,
            4,
            lower=np.zeros(1),
            upper=np.full(1, 100.0),
            history=history[:, None],
            schedule=True,
        )
        plan = plan_flighting(
            _panel(history[:, None]), NATIONAL, reading, 160.0, 4, {TV: Range(0.0, 100.0)}
        )
    np.testing.assert_array_equal(plan.schedule, expected.schedule)
    assert plan.worth == expected.worth


def test_chcs_warning_that_the_schedule_is_local_reaches_the_caller(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        plan = plan_flighting(
            _panel(np.full((6, 1), 0.2)), NATIONAL, _s_shaped(3), 1.2, 4, {TV: Range(0.0, 2.0)}
        )
    local = [r for r in caplog.records if getattr(r, "chc_event", None) == "allocation_local"]
    assert len(local) == 1
    assert local[0].levelno == logging.WARNING
    assert (plan.limit, plan.stopped) == ("local", "cap")
    assert plan.bound > plan.worth


def test_the_median_reading_is_each_parameters_median_over_the_draws() -> None:
    # the middle draw holds every parameter's median; the means lie elsewhere
    reading = _readings(
        3,
        retention=np.array([[0.1, 0.2], [0.5, 0.6], [0.55, 0.9]]),
        half=np.array([[10.0, 20.0], [30.0, 40.0], [300.0, 400.0]]),
        slope=np.array([[0.5, 0.6], [0.8, 0.9], [1.0, 1.0]]),
        coefficient=np.array([[1.0, 2.0], [3.0, 4.0], [30.0, 40.0]]),
    )
    panel = _panel(np.full((6, 2), 10.0))
    limits = {TV: Range(0.0, 50.0), SEARCH: Range(0.0, 50.0)}
    median = plan_flighting(panel, NATIONAL, reading, 200.0, 4, limits)
    middle = plan_flighting(panel, NATIONAL, reading, 200.0, 4, limits, reading=1)
    np.testing.assert_array_equal(median.schedule, middle.schedule)
    np.testing.assert_array_equal(median.continuous, middle.continuous)
    np.testing.assert_array_equal(median.gains, middle.gains)
    assert (median.reading, middle.reading) == ("median", 1)
    first = plan_flighting(panel, NATIONAL, reading, 200.0, 4, limits, reading=0)
    assert not np.array_equal(first.continuous, median.continuous)


def test_the_means_and_intervals_are_the_draws_weighted_ones() -> None:
    reading = _readings(
        1,
        retention=np.zeros((4, 1)),
        half=np.full((4, 1), 50.0),
        slope=np.full((4, 1), 1.0),
        coefficient=_column([4.0, 1.0, 3.0, 2.0]),
    )
    plan = plan_flighting(
        _panel(np.full((6, 1), 30.0)),
        NATIONAL,
        reading,
        100.0,
        2,
        {TV: Range(0.0, 100.0)},
        level=0.5,
    )
    np.testing.assert_allclose(plan.expected, plan.gains.mean(axis=1), rtol=1e-15)
    # an end is a draw's own value, never the average of the two beside an exact hit
    quartiles = np.quantile(plan.gains, [0.25, 0.75], axis=1, method="inverted_cdf")
    np.testing.assert_array_equal(plan.intervals, quartiles.T)
    np.testing.assert_array_equal(
        plan.edge_interval, np.quantile(plan.edge, [0.25, 0.75], method="inverted_cdf")
    )


@settings(max_examples=200, deadline=None, derandomize=True)
@given(
    st.lists(st.floats(-5.0, 5.0), min_size=1, max_size=9),
    st.lists(st.floats(0.01, 1.0), min_size=9, max_size=9),
    st.floats(0.05, 0.95),
)
def test_a_weighted_quantile_is_the_inverted_cdfs(
    values: list[float], weights: list[float], share: float
) -> None:
    x, w = np.array(values), np.array(weights[: len(values)])
    w /= w.sum()
    cumulative = np.cumsum(w[np.argsort(x, kind="stable")])
    if np.any(np.abs(cumulative - share) <= 1e-9):
        return  # at a tie the quantile averages the two values beside it
    expected = np.quantile(x, share, method="inverted_cdf", weights=w)
    assert _quantile(x, w, share) == expected


def test_a_draw_of_weight_0_enters_neither_the_median_nor_an_interval() -> None:
    # the weight-0 draw, 5, sits between the two that weigh: the median is theirs, 50.5
    values, weights = np.array([1.0, 5.0, 100.0]), np.array([0.5, 0.0, 0.5])
    assert _quantile(values, weights, 0.5) == 50.5
    np.testing.assert_array_equal(_interval(values, weights, 0.5), [1.0, 100.0])


def _two_fits(
    stacking: list[float], half: list[float], fit: list[int], tanh: list[bool]
) -> Readings:
    """Draws of one-period Hill or `tanh` channels of slope 3, a fit's index each."""
    draws = len(half)
    return Readings(
        channels=(TV,),
        retention=np.zeros((draws, 1)),
        half_saturation=_column(half),
        slope=np.where(np.array(tanh)[:, None], np.nan, 3.0),
        coefficient=np.ones((draws, 1)),
        tanh=np.array(tanh),
        fit=np.array(fit),
        chain=np.zeros(draws, dtype=np.intp),
        stacking=np.array(stacking),
        pointwise=None,
        length=1,
    )


@pytest.mark.parametrize("tanh", [False, True], ids=["hill", "tanh"])
def test_a_fit_of_weight_0_leaves_the_plan_as_it_was(tanh: bool) -> None:
    """A fit of weight 1 with half-saturation points 0.5 and 4, and one of weight 0 at 1, a Hill
    or a `tanh`: the median, the schedule, the means and the intervals are those of the first fit
    alone. Read with the weight-0 draw, the median was 0.75, and the schedule a rate of 1 in each
    period where it pulses 2, 2, 0, 0."""
    panel, limits = _panel(np.ones((8, 1))), {TV: Range(0.0, 2.0)}
    alone = plan_flighting(
        panel, NATIONAL, _two_fits([1.0], [0.5, 4.0], [0, 0], [False] * 2), 4.0, 4, limits
    )
    stacked = _two_fits([1.0, 0.0], [0.5, 4.0, 1.0], [0, 0, 1], [False, False, tanh])
    plan = plan_flighting(panel, NATIONAL, stacked, 4.0, 4, limits)
    np.testing.assert_array_equal(alone.schedule, [[2.0], [2.0], [0.0], [0.0]])
    np.testing.assert_array_equal(plan.schedule, alone.schedule)
    np.testing.assert_array_equal(plan.continuous, alone.continuous)
    np.testing.assert_array_equal(plan.gains[:, :2], alone.gains)
    np.testing.assert_array_equal(plan.expected, alone.expected)
    np.testing.assert_array_equal(plan.intervals, alone.intervals)
    np.testing.assert_array_equal(plan.edge_interval, alone.edge_interval)
    assert plan.recommendation == alone.recommendation


def test_alike_weights_read_numpys_median() -> None:
    for values in ([3.0, 1.0, 2.0], [4.0, 1.0, 3.0, 2.0]):
        x = np.array(values)
        assert _quantile(x, np.full(x.size, 1.0 / x.size), 0.5) == np.median(x)


@st.composite
def _concave(draw: st.DrawFn) -> tuple[Readings, float, list[Range]]:
    """Concave channels through one-period kernels, their draws, a box and a budget it spends."""
    channels = draw(st.integers(1, 2))
    draws = draw(st.integers(1, 4))
    tanh = draw(st.booleans())

    def table(low: float, high: float) -> np.ndarray:
        cells = st.floats(low, high, allow_nan=False)
        flat = draw(st.lists(cells, min_size=draws * channels, max_size=draws * channels))
        return np.array(flat).reshape(draws, channels)

    reading = _readings(
        1,
        retention=table(0.0, 1.0),
        half=table(5.0, 200.0),
        slope=table(0.3, 1.0),
        coefficient=table(0.5, 10.0),
        tanh=tanh,
    )
    limits = [Range(0.0, draw(st.floats(20.0, 150.0))) for _ in range(channels)]
    share = draw(st.floats(0.05, 0.95))
    return reading, share * sum(r.most for r in limits), limits


@settings(max_examples=12, deadline=None, derandomize=True)
@given(_concave())
def test_on_concave_one_period_channels_the_schedule_returns_at_least_the_continuous_plans(
    case: tuple[Readings, float, list[Range]],
) -> None:
    reading, rate, ranges = case
    periods = 3
    channels = reading.channels
    limits = dict(zip(channels, ranges, strict=True))
    plan = plan_flighting(
        _panel(np.full((4, len(channels)), 10.0)),
        NATIONAL,
        reading,
        periods * rate,
        periods,
        limits,
    )
    one = _readings(
        1,
        retention=np.median(reading.retention, axis=0)[None],
        half=np.median(reading.half_saturation, axis=0)[None],
        slope=np.median(reading.slope, axis=0)[None],
        coefficient=np.median(reading.coefficient, axis=0)[None],
        tanh=bool(reading.tanh[0]),
    )
    continuous = sum(
        periods * _worth(_response(one, 0, c), np.zeros(0), plan.continuous[c : c + 1])
        for c in range(len(channels))
    )
    assert plan.worth >= continuous - 1e-9 * abs(continuous)
    assert plan.recommendation == "continuous"
    assert plan.limit is None


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"budget": [100.0, 100.0, 100.0, 100.0]}, "per-period budgets"),
        ({"budget": np.full(4, 100.0)}, "per-period budgets"),
        ({"budget": 1e6}, "outside what the ranges spend"),
        ({"reading": "mean"}, "neither 'median' nor a draw's index"),
        ({"reading": 3}, "neither 'median' nor a draw's index"),
        ({"reading": -1}, "neither 'median' nor a draw's index"),
        ({"reading": True}, "neither 'median' nor a draw's index"),
        ({"level": 0.0}, "not a share in"),
        ({"level": 1.0}, "not a share in"),
        ({"periods": 0}, "periods must be a whole number"),
    ],
)
def test_a_flighted_plan_refuses_what_it_cannot_plan(change: dict[str, Any], match: str) -> None:
    arguments: dict[str, Any] = {
        "panel": _panel(np.full((6, 1), 100.0)),
        "geo": NATIONAL,
        "readings": _decaying(),
        "budget": 400.0,
        "periods": 4,
        "limits": {TV: Range(0.0, 200.0)},
    } | change
    with pytest.raises(ValueError, match=match):
        plan_flighting(**arguments)


def test_the_median_of_draws_of_both_curves_is_refused() -> None:
    reading = Readings(
        channels=(TV,),
        retention=np.full((2, 1), 0.5),
        half_saturation=np.full((2, 1), 10.0),
        slope=np.array([[1.0], [np.nan]]),
        coefficient=np.ones((2, 1)),
        tanh=np.array([False, True]),
        fit=np.array([0, 1]),
        chain=np.zeros(2, dtype=np.intp),
        stacking=np.array([0.5, 0.5]),
        pointwise=None,
        length=2,
    )
    panel, limits = _panel(np.full((6, 1), 10.0)), {TV: Range(0.0, 50.0)}
    with pytest.raises(ValueError, match="both curves"):
        plan_flighting(panel, NATIONAL, reading, 100.0, 4, limits)
    plan = plan_flighting(panel, NATIONAL, reading, 100.0, 4, limits, reading=1)
    assert plan.gains.shape == (3, 2)


def test_the_median_weighs_each_draw_by_its_fits_weight() -> None:
    # one draw of a fit weighted 0.8 against three of a fit weighted 0.2: the heavy draw is the
    # weighted median of every parameter, where the draws' plain median is the light fit's
    light = [0.6, 0.7, 0.8]
    reading = Readings(
        channels=(TV,),
        retention=_column([0.2, *light]),
        half_saturation=_column([20.0, 60.0, 70.0, 80.0]),
        slope=_column([0.7, 1.0, 1.0, 1.0]),
        coefficient=_column([2.0, 6.0, 7.0, 8.0]),
        tanh=np.zeros(4, dtype=bool),
        fit=np.array([0, 1, 1, 1]),
        chain=np.zeros(4, dtype=np.intp),
        stacking=np.array([0.8, 0.2]),
        pointwise=None,
        length=2,
    )
    panel, limits = _panel(np.full((6, 1), 10.0)), {TV: Range(0.0, 50.0)}
    median = plan_flighting(panel, NATIONAL, reading, 100.0, 4, limits)
    heavy = plan_flighting(panel, NATIONAL, reading, 100.0, 4, limits, reading=0)
    np.testing.assert_array_equal(median.schedule, heavy.schedule)
    assert median.worth == heavy.worth
    np.testing.assert_allclose(median.expected, median.gains @ reading.weights, rtol=1e-15)


def test_a_schedule_short_of_the_best_rate_a_period_is_that_rate() -> None:
    # chc's search of the boxes over the periods closes its gap on the cells' interpolants; on a
    # Hill this steep at 0 its schedule, 10.0024 in the first period, returned 3.5 times its
    # tolerance less than 10 in every period
    reading = _readings(
        1,
        retention=np.zeros((1, 1)),
        half=np.full((1, 1), 5.0),
        slope=np.full((1, 1), 0.3046875),
        coefficient=np.ones((1, 1)),
    )
    plan = plan_flighting(
        _panel(np.full((4, 1), 10.0)), NATIONAL, reading, 30.0, 3, {TV: Range(0.0, 20.0)}
    )
    continuous = 3 * _worth(_response(reading, 0, 0), np.zeros(0), np.array([10.0]))
    assert plan.worth >= continuous * (1.0 - 1e-12)
    np.testing.assert_array_equal(plan.schedule, np.tile(plan.continuous, (3, 1)))
    np.testing.assert_array_equal(plan.edge, [0.0])
    assert plan.recommendation == "continuous"
