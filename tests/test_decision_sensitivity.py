"""mixdyn.decisions.posterior: how far a plan's effects may be off before its gain is gone.

`breakdown`'s closed form, `sqrt(sum g^+ / sum g^-)` (Maxima: the least of `sum_c b_c g_c` over
`b` in `[1/r, r]^C` is at the corner `gp / r - r gm`, which falls in `r` and is 0 there), is the
root a bisection finds of the corners' least gain, on random gains. `plan_breakdown`'s gains are
`chc.response`'s worths' differences. `plan_robust` at a factor of 1 is `plan_posterior`'s plan; at
a larger one its least gain is the best a brute force finds over every split on aligned grids, each
split's least gain a minimum over the corners of the factors; on two `tanh` channels it moves what
Maxima's search moves and keeps the reference from the square root of the ratio of the channels'
marginal worths there.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from chc.response import Channel as Response
from chc.response import GeometricAdstock, Hill, Tanh
from hypothesis import given, settings
from hypothesis import strategies as st

from mixdyn import NATIONAL, Cadence, Channel, MediaPanel, Revenue
from mixdyn.decisions import (
    Range,
    Readings,
    breakdown,
    plan_breakdown,
    plan_posterior,
    plan_robust,
    worths,
)

TANH_HALF = math.atanh(0.5)
TV, SEARCH, SOCIAL = Channel("tv"), Channel("search"), Channel("social")


def _panel(channels: tuple[Channel, ...], history: np.ndarray) -> MediaPanel:
    weeks = history.shape[0]
    return MediaPanel(
        dates=np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(weeks),
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=channels,
        controls=(),
        kpi=np.ones((1, weeks)),
        kpi_unit=Revenue(),
        spend=history[None],
        currency="EUR",
        control_values=np.zeros((1, weeks, 0)),
        filled=np.zeros((1, weeks), dtype=bool),
    )


def _mixed(channels: int) -> Readings:
    """Two Hill draws of one fit and a `tanh` draw of another, weighted 0.25 and 0.75."""
    rng = np.random.default_rng(20261013)
    return Readings(
        channels=(TV, SEARCH, SOCIAL)[:channels],
        retention=rng.uniform(0.2, 0.7, (3, channels)),
        half_saturation=rng.uniform(5.0, 15.0, (3, channels)),
        slope=np.vstack([rng.uniform(0.8, 2.5, (2, channels)), np.full((1, channels), np.nan)]),
        coefficient=rng.uniform(20.0, 60.0, (3, channels)),
        tanh=np.array([False, False, True]),
        fit=np.array([0, 0, 1]),
        chain=np.array([0, 1, 0]),
        stacking=np.array([0.25, 0.75]),
        pointwise=None,
        length=6,
    )


def _history(channels: int, weeks: int = 16) -> np.ndarray:
    return np.random.default_rng(7).uniform(2.0, 12.0, (weeks, channels))


def _response(readings: Readings, s: int, c: int) -> Response:
    kernel = GeometricAdstock(readings.retention[s, c], length=readings.length, normalized=True)
    half = readings.half_saturation[s, c]
    curve = Tanh(half / TANH_HALF) if readings.tanh[s] else Hill(half, readings.slope[s, c])
    return Response(kernel, curve, readings.coefficient[s, c])


def _worth(response: Response, history: np.ndarray, rate: float, periods: int) -> float:
    tail = np.zeros(response.kernel.length - 1)
    series = np.concatenate([history, np.full(periods, rate), tail])
    return float(jnp.sum(response(jnp.asarray(series))[history.size :]))


def _least(gains: np.ndarray, factor: float) -> float:
    """The least of `sum_c b_c g_c` over every corner of `b` in `[1/factor, factor]^C`."""
    return min(
        float(np.dot(corner, gains))
        for corner in itertools.product((1.0 / factor, factor), repeat=gains.size)
    )


def _counted(gains: np.ndarray, factor: float) -> float:
    """`sum_c b_c g_c` with `b_c` `1/factor` where `g_c` gains and `factor` where it loses."""
    return float(np.sum(np.where(gains > 0.0, gains / factor, gains * factor)))


def test_the_factor_is_the_square_root_of_what_the_plan_grows_over_what_it_cuts() -> None:
    """Maxima's instance, `[3, 1, -2, -1/2]`: `r* = 2^(3/2) / sqrt(5)`; a plan whose net gain is
    5 % of what its cuts give up: `sqrt(1.05)`, 2.5 % off; each row of a table read alike."""
    assert float(breakdown([3.0, 1.0, -2.0, -0.5])) == pytest.approx(
        2.0**1.5 / math.sqrt(5.0), rel=1e-15, abs=0.0
    )
    assert float(breakdown([1.05, -1.0])) == pytest.approx(math.sqrt(1.05), rel=1e-15, abs=0.0)
    assert float(breakdown([1.05, -1.0])) == pytest.approx(1.0247, abs=5e-5)
    np.testing.assert_array_equal(
        breakdown([[4.0, -1.0], [1.0, -4.0], [2.0, 0.0], [0.0, -2.0], [0.0, 0.0]]),
        [2.0, 0.5, np.inf, 0.0, 1.0],
    )
    assert breakdown(np.ones((2, 3, 4))).shape == (2, 3)


@pytest.mark.parametrize("gains", [[], [[]], 3.0, [1.0, np.nan], [np.inf, -1.0]])
def test_the_factor_refuses_gains_it_cannot_read(gains: Any) -> None:
    with pytest.raises(ValueError, match="gains must be finite"):
        breakdown(gains)


@settings(max_examples=200, deadline=None)
@given(
    st.lists(st.floats(0.01, 10.0), min_size=1, max_size=3),
    st.lists(st.floats(-10.0, -0.01), min_size=1, max_size=3),
    st.randoms(use_true_random=False),
)
def test_the_factor_is_where_the_corners_least_gain_crosses_0(
    up: list[float], down: list[float], order: Any
) -> None:
    """A bisection on the least gain over the corners finds the closed form's root; below it the
    least gain is above 0, above it below."""
    gains = np.array(up + down)
    order.shuffle(gains)
    factor = float(breakdown(gains))

    def least(r: float) -> float:
        return _least(gains, r) if r >= 1.0 else -_least(-gains, 1.0 / r)

    low, high = 1e-3, 1e3
    for _ in range(80):
        middle = math.sqrt(low * high)
        low, high = (middle, high) if least(middle) > 0.0 else (low, middle)
    assert factor == pytest.approx(low, rel=1e-12, abs=0.0)
    if factor >= 1.0:
        assert _least(gains, factor) == pytest.approx(0.0, abs=1e-12 * np.abs(gains).sum())
        assert _counted(gains, factor) == pytest.approx(_least(gains, factor), abs=1e-12)


def test_a_plans_gains_are_each_draws_change_in_chc_responses_worths() -> None:
    reading = _mixed(3)
    history = _history(3)
    panel = _panel(reading.channels, history)
    periods, spend, against = 5, np.array([4.0, 11.0, 6.5]), np.array([7.0, 6.0, 8.5])
    result = plan_breakdown(panel, NATIONAL, reading, spend, periods, against=against)
    for s in range(3):
        for c in range(3):
            response = _response(reading, s, c)
            expected = _worth(response, history[:, c], spend[c], periods) - _worth(
                response, history[:, c], against[c], periods
            )
            assert result.gains[s, c] == pytest.approx(expected, rel=1e-10, abs=1e-12)
    np.testing.assert_allclose(result.expected, reading.weights @ result.gains, rtol=1e-15)
    assert result.factor == float(breakdown(result.expected))
    np.testing.assert_array_equal(result.draws, breakdown(result.gains))
    assert result.channels == reading.channels
    np.testing.assert_array_equal(result.spend, spend)
    np.testing.assert_array_equal(result.against, against)
    current = plan_breakdown(panel, NATIONAL, reading, spend, periods)
    np.testing.assert_array_equal(current.against, history[-periods:].mean(axis=0))
    with pytest.raises(ValueError, match="read-only"):
        result.gains[0, 0] = 0.0


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"spend": [1.0, 2.0]}, r"spend must be one finite spend at or above 0 a channel, \(3,\)"),
        ({"spend": [1.0, -2.0, 3.0]}, "spend must be one finite spend"),
        ({"against": [1.0, np.nan, 3.0]}, "against must be one finite spend"),
        ({"periods": 0}, "periods must be a whole number"),
        ({"geo": "north"}, "the panel has no geo"),
    ],
)
def test_a_breakdown_refuses_what_it_cannot_read(change: dict[str, Any], match: str) -> None:
    reading = _mixed(3)
    arguments: dict[str, Any] = {
        "panel": _panel(reading.channels, _history(3)),
        "geo": NATIONAL,
        "readings": reading,
        "spend": [1.0, 2.0, 3.0],
        "periods": 4,
        "against": [2.0, 2.0, 2.0],
    }
    with pytest.raises(ValueError, match=match):
        plan_breakdown(**(arguments | change))


def test_at_a_factor_of_1_the_robust_plan_is_the_posteriors_plan() -> None:
    reading = _mixed(3)
    panel = _panel(reading.channels, _history(3))
    limits = {c: Range(1.0, 20.0) for c in reading.channels}
    plan = plan_posterior(panel, NATIONAL, reading, 6 * 24.0, 6, limits, grid=301)
    robust = plan_robust(panel, NATIONAL, reading, 6 * 24.0, 6, limits, scale=1.0, grid=301)
    np.testing.assert_array_equal(robust.spend, plan.spend)
    np.testing.assert_array_equal(robust.against, plan.current)
    assert robust.gain == pytest.approx(plan.gain - plan.current_gain, rel=1e-12, abs=0.0)
    assert robust.worst == robust.gain


@pytest.mark.parametrize("scale", [1.0, 1.05, 1.2, 1.5, 3.0])
def test_the_robust_plan_is_the_best_least_gain_a_brute_force_finds(scale: float) -> None:
    """Three channels, each range `[0, B]` on the same grid, so every split of `B` on the grids
    spends a point of each; the reference is one of them. Each split's least gain is the minimum
    over the factors' corners of the channels' expected changes in worth."""
    points, budget, periods = 61, 24.0, 4
    reading = _mixed(3)
    panel = _panel(reading.channels, _history(3))
    limits = {c: Range(0.0, budget) for c in reading.channels}
    table = worths(panel, NATIONAL, reading, periods, limits, grid=points)
    means = np.tensordot(reading.weights, table.table, axes=1)
    spends = table.rates[0]
    first, second = 12, 30
    third = points - 1 - first - second
    against = np.array([spends[first], spends[second], spends[third]])
    base = means[[0, 1, 2], [first, second, third]]
    best, argument = -np.inf, (first, second, third)
    for i in range(points):
        for j in range(points - i):
            k = points - 1 - i - j
            change = means[[0, 1, 2], [i, j, k]] - base
            least = _least(change, scale)
            if least > best:
                best, argument = least, (i, j, k)
    robust = plan_robust(
        panel,
        NATIONAL,
        reading,
        periods * budget,
        periods,
        limits,
        scale=scale,
        against=against,
        grid=points,
    )
    np.testing.assert_array_equal(robust.against, against)
    assert robust.worst == pytest.approx(best, rel=1e-9, abs=1e-12)
    np.testing.assert_allclose(robust.spend, spends[list(argument)], rtol=0.0, atol=1e-9)
    gains = plan_breakdown(panel, NATIONAL, reading, robust.spend, periods, against=against)
    assert robust.gain == pytest.approx(gains.expected.sum(), rel=1e-10, abs=1e-12)
    assert robust.worst == pytest.approx(_least(gains.expected, scale), rel=1e-10, abs=1e-12)
    if robust.worst > 0.0:
        assert gains.factor > scale


def _two_tanh() -> tuple[MediaPanel, Readings, dict[Channel, Range]]:
    """Maxima's two channels, one period with no carryover: `W_i(s) = tanh(c s)` and
    `W_j(s) = 1.6 tanh(c s / 1.5)`, `c = atanh(1/2)`, each spend in `[0, 2]`."""
    reading = Readings(
        channels=(TV, SEARCH),
        retention=np.array([[0.5, 0.5]]),
        half_saturation=np.array([[1.0, 1.5]]),
        slope=np.full((1, 2), np.nan),
        coefficient=np.array([[1.0, 1.6]]),
        tanh=np.array([True]),
        fit=np.array([0]),
        chain=np.array([0]),
        stacking=np.array([1.0]),
        pointwise=None,
        length=1,
    )
    return (
        _panel(reading.channels, np.ones((4, 2))),
        reading,
        {TV: Range(0.0, 2.0)} | {SEARCH: Range(0.0, 2.0)},
    )


def _marginal_ratio() -> float:
    def slope(top: float, half: float, s: float) -> float:
        return top * TANH_HALF / half / math.cosh(TANH_HALF * s / half) ** 2

    return slope(1.6, 1.5, 0.8) / slope(1.0, 1.0, 1.2)


def test_two_tanh_channels_move_what_maxima_moves() -> None:
    """At a factor of 1.1 Maxima's search over moves of 0.001 moves 0.238 from the first channel
    to the second, for a least gain of 0.0103944."""
    panel, reading, limits = _two_tanh()
    against = np.array([1.2, 0.8])
    robust = plan_robust(panel, NATIONAL, reading, 2.0, 1, limits, scale=1.1, against=against)
    move = robust.spend[1] - against[1]
    assert move == pytest.approx(0.238, abs=2e-3)
    assert robust.spend.sum() == pytest.approx(2.0, rel=1e-12)
    assert robust.worst == pytest.approx(0.0103944, rel=1e-4)
    assert _marginal_ratio() == pytest.approx(1.47134, rel=1e-5)


def test_the_reference_is_kept_from_the_square_root_of_the_marginal_ratio() -> None:
    """The least factor at which the plan keeps the reference is `sqrt(W_j' / W_i')` there, 1.213,
    to the grid's step: below it the plan moves toward the steeper channel, above it it keeps the
    reference's spend as given, whose breakdown factor is 1, no gain to erase."""
    panel, reading, limits = _two_tanh()
    against = np.array([1.2, 0.8])
    threshold = math.sqrt(_marginal_ratio())

    def moves(scale: float) -> bool:
        plan = plan_robust(panel, NATIONAL, reading, 2.0, 1, limits, scale=scale, against=against)
        return bool(np.max(np.abs(plan.spend - against)) > 1e-9)

    assert moves(0.995 * threshold)
    assert not moves(1.005 * threshold)
    low, high = 1.0, 2.0
    for _ in range(30):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if moves(middle) else (low, middle)
    assert high == pytest.approx(threshold, rel=2e-3)
    kept = plan_robust(panel, NATIONAL, reading, 2.0, 1, limits, scale=1.5, against=against)
    np.testing.assert_array_equal(kept.spend, against)
    assert kept.worst == kept.gain == 0.0
    kept_breakdown = plan_breakdown(panel, NATIONAL, reading, kept.spend, 1, against=against)
    assert kept_breakdown.factor == 1.0


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"scale": 0.9}, "not a finite factor of at least 1"),
        ({"scale": math.nan}, "not a finite factor of at least 1"),
        ({"scale": math.inf}, "not a finite factor of at least 1"),
        ({"against": [1.0]}, r"against must be one finite spend at or above 0 a channel, \(2,\)"),
        ({"budget": 5.0}, "outside what the ranges spend"),
        ({"grid": 1}, "grid must be a whole number, at least 2"),
    ],
)
def test_a_robust_plan_refuses_what_it_cannot_read(change: dict[str, Any], match: str) -> None:
    panel, reading, limits = _two_tanh()
    arguments: dict[str, Any] = {
        "panel": panel,
        "geo": NATIONAL,
        "readings": reading,
        "budget": 2.0,
        "periods": 1,
        "limits": limits,
        "scale": 1.2,
        "against": [1.2, 0.8],
    }
    with pytest.raises(ValueError, match=match):
        plan_robust(**(arguments | change))
