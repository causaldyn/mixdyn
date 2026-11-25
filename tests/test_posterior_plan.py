"""mixdyn.decisions.posterior: a budget planned for a posterior's expected worth over its draws.

The fixture tests/fixtures/oracles/posterior_plan_bench.json fitted three of Track M v2's pilot
worlds without geo tests, each by a Hill and by a `tanh`, kept 400 draws of each fit evenly
over its chains, and recorded the bench's gated planner's plans of each fit's expected worth and of
the two stacked by Cover's weights on their leave-one-out densities, with each of those means at
every 64th spend of its grid: `readings`, `stacked` and `plan_posterior` make the bench's plans to
1e-12, and `worths` its means. Each draw's worth is `chc.response`'s; a plan of one draw is within
the grid's bound of `chc.allocation.allocate`'s exact plan; two fits' stacking by bisection meets
Cover's certificate and reaches Cover's objective; the plan's gains are the draws' weighted mean
gains against spending nothing in the plan. The plan for the draws' worst share is
`chc.allocation.cvar_allocate`'s on the readings the scorecard's `cvar 0.1` arm reads, each fit's
share of the draws its weight's by largest remainders, and its CVaR the mean of its worst gains.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import logging
import math
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
import pytest
from chc.allocation import allocate, cvar_allocate
from chc.response import Channel as Response
from chc.response import GeometricAdstock, Hill, Tanh
from hypothesis import given, settings
from hypothesis import strategies as st

from mixdyn import NATIONAL, Cadence, Channel, MediaPanel, Revenue
from mixdyn.decisions import (
    Range,
    Readings,
    pick_by_lift_tests,
    plan_cvar,
    plan_posterior,
    posterior,
    readings,
    stacked_readings,
    worths,
)
from mixdyn.decisions.posterior import _alike, _allocated, _fill, _rates
from mixdyn.decisions.posterior import _envelope as _vertices
from mixdyn.inference import loo, stacking
from mixdyn.inference.loo import _stacking_of_two
from mixdyn.models.bayes_mmm import BayesFit, BayesMedia, BayesSpec, HillCurve, TanhCurve

if TYPE_CHECKING:
    from collections.abc import Sequence

BENCH = json.loads(
    (Path(__file__).parent / "fixtures" / "oracles" / "posterior_plan_bench.json").read_text()
)
WORLDS = BENCH["worlds"]
FAMILIES = tuple(BENCH["families"])
PERIODS, GRID, EVERY = BENCH["periods"], BENCH["grid"], BENCH["every"]
TANH_HALF = math.atanh(0.5)


def _panel(world: dict[str, Any]) -> MediaPanel:
    history = np.array(world["history"])
    weeks = history.shape[0]
    return MediaPanel(
        dates=np.datetime64(world["start"]) + np.timedelta64(7, "D") * np.arange(weeks),
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=tuple(Channel(c) for c in world["channels"]),
        controls=(),
        kpi=np.array(world["sales"])[None],
        kpi_unit=Revenue(),
        spend=history[None],
        currency="EUR",
        control_values=np.zeros((1, weeks, 0)),
        filled=np.zeros((1, weeks), dtype=bool),
    )


def _limits(world: dict[str, Any]) -> dict[Channel, Range]:
    return {
        Channel(c): Range(low, high)
        for c, low, high in zip(world["channels"], world["lower"], world["upper"], strict=True)
    }


def _fit(world: dict[str, Any], family: str, log_likelihood: np.ndarray | None = None) -> BayesFit:
    """A fit holding the recorded draws, `(chains, draws)` as they were kept: each chain's in
    order, so that every draw of it is the bench's thinned draws in the bench's order."""
    record = world["fits"][family]
    chains = record["chains"]
    assert np.bincount(record["chain"]).tolist() == [len(record["chain"]) // chains] * chains
    names = ("retention", "half_saturation", "coefficient", *(() if record["tanh"] else ("slope",)))
    draws = {
        name: np.array(record[name]).reshape(chains, -1, len(world["channels"])) for name in names
    }
    weeks = len(world["sales"])
    per_chain = draws["retention"].shape[1]
    return BayesFit(
        spec=BayesSpec(
            media=tuple(BayesMedia(c) for c in world["channels"]),
            saturation=TanhCurve() if record["tanh"] else HillCurve(),
        ),
        dates=np.datetime64(world["start"]) + np.timedelta64(7, "D") * np.arange(weeks),
        y=np.array(world["sales"]),
        fitted=np.ones(weeks, dtype=bool),
        draws=MappingProxyType(draws),
        log_likelihood=(
            np.zeros((chains, per_chain, weeks)) if log_likelihood is None else log_likelihood
        ),
        contributions=MappingProxyType({}),
        estimated=np.zeros(weeks),
        diagnostics=MappingProxyType({}),
        divergent=np.zeros(chains, dtype=np.int64),
        e_bfmi=np.ones(chains),
        length=record["length"],
        spend=MappingProxyType(
            {c: np.array(world["history"])[:, j] for j, c in enumerate(world["channels"])}
        ),
        cadence=Cadence.WEEK,
    )


def _bench_readings(world: dict[str, Any], arm: str) -> Readings:
    if arm == "stacked":
        weights = [world["stacking"][f] for f in FAMILIES]
        return stacked_readings([_fit(world, f) for f in FAMILIES], weights=weights)
    return readings(_fit(world, arm))


def _response(readings: Readings, s: int, c: int) -> Response:
    """Draw `s`'s channel `c` as `chc.response` reads it, as `BayesFit.channel_draws` builds it."""
    kernel = GeometricAdstock(readings.retention[s, c], length=readings.length, normalized=True)
    half = readings.half_saturation[s, c]
    curve = Tanh(half / TANH_HALF) if readings.tanh[s] else Hill(half, readings.slope[s, c])
    return Response(kernel, curve, readings.coefficient[s, c])


def _worth(response: Response, history: np.ndarray, rate: float) -> float:
    """What `response` returns over the plan's periods and the kernel's tail at `rate` a period,
    its history carried in: `chc.allocation`'s worth."""
    tail = np.zeros(response.kernel.length - 1)
    series = np.concatenate([history, np.full(PERIODS, rate), tail])
    return float(jnp.sum(response(jnp.asarray(series))[history.size :]))


@pytest.mark.parametrize("world", WORLDS, ids=lambda w: f"{w['environment']}-{w['seed']}")
@pytest.mark.parametrize("arm", [*FAMILIES, "stacked"])
def test_the_plan_is_the_bench_planners(world: dict[str, Any], arm: str) -> None:
    plan = plan_posterior(
        _panel(world),
        NATIONAL,
        _bench_readings(world, arm),
        world["budget"],
        PERIODS,
        _limits(world),
        grid=GRID,
    )
    np.testing.assert_allclose(plan.spend, world["plans"][arm], rtol=0.0, atol=1e-12)


def test_each_fits_mean_worth_is_the_bench_planners() -> None:
    """On the drawn world whose fits' weights are nearest even: each fit's mean worth over its
    draws, and the two stacked by their weights, at every 64th spend of the bench's grid."""
    world = WORLDS[1]
    both = _bench_readings(world, "stacked")
    table = worths(_panel(world), NATIONAL, both, PERIODS, _limits(world), grid=GRID)
    means = {f: table.table[both.fit == k].mean(axis=0) for k, f in enumerate(FAMILIES)}
    means["stacked"] = sum(both.stacking[k] * means[f] for k, f in enumerate(FAMILIES))
    for arm, mean in means.items():
        np.testing.assert_allclose(mean[:, ::EVERY], world["mean_returns"][arm], rtol=1e-12)


@pytest.mark.parametrize("world", WORLDS, ids=lambda w: f"{w['environment']}-{w['seed']}")
def test_each_draws_worth_is_chc_responses(world: dict[str, Any]) -> None:
    both = _bench_readings(world, "stacked")
    history = np.array(world["history"])
    table = worths(_panel(world), NATIONAL, both, PERIODS, _limits(world), grid=3)
    np.testing.assert_array_equal(table.rates[:, 0], world["lower"])
    np.testing.assert_array_equal(table.rates[:, -1], world["upper"])
    for s in (0, 1, 217, 399, 400, 401, 613, 799):
        for c in range(len(world["channels"])):
            response = _response(both, s, c)
            for g, rate in enumerate(table.rates[c]):
                expected = _worth(response, history[:, c], rate)
                assert table.table[s, c, g] == pytest.approx(expected, rel=1e-12)


def test_a_kernel_longer_than_the_history_carries_the_historys_first_period() -> None:
    """Kernels longer than the panel's history: each draw's worth is still `chc.response`'s, the
    history's first period carried into the plan's."""
    weeks = 5
    longer: dict[str, Any] = {"length": 12}
    reading = Readings(**(_valid() | longer))
    spend = np.zeros((1, weeks, 2))
    spend[0, 0] = [40.0, 30.0]
    spend[0, -1] = [10.0, 20.0]
    panel = MediaPanel(
        dates=np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(weeks),
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=(Channel("tv"), Channel("search")),
        controls=(),
        kpi=np.ones((1, weeks)),
        kpi_unit=Revenue(),
        spend=spend,
        currency="EUR",
        control_values=np.zeros((1, weeks, 0)),
        filled=np.zeros((1, weeks), dtype=bool),
    )
    limits = {Channel("tv"): Range(5.0, 30.0), Channel("search"): Range(5.0, 30.0)}
    table = worths(panel, NATIONAL, reading, PERIODS, limits, grid=3)
    for s in range(reading.fit.size):
        for c in range(2):
            for g, rate in enumerate(table.rates[c]):
                expected = _worth(_response(reading, s, c), spend[0, :, c], rate)
                assert table.table[s, c, g] == pytest.approx(expected, rel=1e-12)


def test_a_one_draw_plan_is_within_the_grids_bound_of_chc_allocations() -> None:
    """A draw of the `tanh` fit: each channel's worth is concave in its spend, so the plan on the
    grid's envelope falls short of the exact plan's worth by at most `sum_c h_c^2 M_c / 8`, `h_c`
    the grid's step and `M_c` a bound on the worth's curvature, `coefficient kappa^2 max|tanh''|
    sum_t reach_t^2`, `kappa` the curve's 1 / scale."""
    world = WORLDS[1]
    tanh = _fit(world, "tanh")
    draw = readings(tanh, draws=1)
    channels = [_response(draw, 0, c) for c in range(len(world["channels"]))]
    history = np.array(world["history"])
    lower, upper = np.array(world["lower"]), np.array(world["upper"])
    plan = plan_posterior(
        _panel(world), NATIONAL, draw, world["budget"], PERIODS, _limits(world), grid=GRID
    )
    exact = allocate(channels, world["budget"], PERIODS, lower=lower, upper=upper, history=history)

    def worth(spend: np.ndarray) -> float:
        return sum(_worth(r, history[:, c], spend[c]) for c, r in enumerate(channels))

    step = (upper - lower) / (GRID - 1)
    unit = np.concatenate([np.zeros(history.shape[0]), np.ones(PERIODS), np.zeros(draw.length - 1)])
    curvature = [
        float(r.coefficient)
        / float(r.curve.scale) ** 2
        * 4.0
        / (3.0 * math.sqrt(3.0))
        * float(jnp.sum(r.kernel(jnp.asarray(unit))[history.shape[0] :] ** 2))
        for r in channels
    ]
    bound = float(np.sum(step**2 * np.array(curvature))) / 8.0
    shortfall = worth(exact.spend) - worth(plan.spend)
    assert -1e-9 <= shortfall <= bound
    assert bound <= 1e-6 * plan.gain
    assert PERIODS * plan.spend.sum() == pytest.approx(world["budget"], rel=1e-12)


def _on_interpolants(rates: np.ndarray, means: np.ndarray, spend: np.ndarray) -> float:
    return float(sum(np.interp(spend[c], rates[c], means[c]) for c in range(spend.size)))


def _best_split(
    rates: np.ndarray | Sequence[np.ndarray],
    means: np.ndarray | Sequence[np.ndarray],
    target: float,
) -> float:
    """The most worth on the means' interpolants of a split of `target` within the grids, a grid of
    each channel's spends. In each cell of the grids the worth is linear, so a best split has every
    channel but one on a point of its grid, and that one takes the rest."""
    best = -np.inf
    channels = len(rates)
    for free in range(channels):
        others = [c for c in range(channels) if c != free]
        for points in itertools.product(*(range(len(rates[c])) for c in others)):
            rest = target - sum(rates[c][i] for c, i in zip(others, points, strict=True))
            slack = 1e-12 * max(1.0, abs(target))
            if not rates[free][0] - slack <= rest <= rates[free][-1] + slack:
                continue
            rest = min(max(rest, rates[free][0]), rates[free][-1])
            worth = np.interp(rest, rates[free], means[free]) + sum(
                means[c][i] for c, i in zip(others, points, strict=True)
            )
            best = max(best, float(worth))
    return best


def _one_draw_panel(channels: tuple[Channel, ...], weeks: int) -> MediaPanel:
    return MediaPanel(
        dates=np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(weeks),
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=channels,
        controls=(),
        kpi=np.ones((1, weeks)),
        kpi_unit=Revenue(),
        spend=np.zeros((1, weeks, len(channels))),
        currency="EUR",
        control_values=np.zeros((1, weeks, 0)),
        filled=np.zeros((1, weeks), dtype=bool),
    )


def _one_draw(channels: tuple[Channel, ...], **parameters: list[float]) -> Readings:
    return Readings(
        channels=channels,
        retention=np.array([parameters["retention"]]),
        half_saturation=np.array([parameters["half"]]),
        slope=np.array([parameters["slope"]]),
        coefficient=np.array([parameters["coefficient"]]),
        tanh=np.array([False]),
        fit=np.array([0]),
        chain=np.array([0]),
        stacking=np.array([1.0]),
        pointwise=None,
        length=int(parameters["length"][0]),
    )


def test_an_s_shaped_mean_beside_a_steeper_one_gets_nothing() -> None:
    """A Hill of slope 4 on television, S-shaped across its range, beside a nearly linear search:
    the steepest stretch of television's mean is steeper than search's, but no split that spends on
    it beats spending all on search, which is the best split a brute force finds."""
    target, tv, search = 20.0, Channel("tv"), Channel("search")
    panel = _one_draw_panel((tv, search), 20)
    reading = _one_draw(
        (tv, search),
        retention=[0.3, 0.3],
        half=[30.0, 400.0],
        slope=[4.0, 1.0],
        coefficient=[100.0, 900.0],
        length=[4],
    )
    limits = {tv: Range(0.0, 60.0), search: Range(0.0, 60.0)}
    plan = plan_posterior(panel, NATIONAL, reading, 13 * target, 13, limits, grid=61)
    table = worths(panel, NATIONAL, reading, 13, limits, grid=61)
    rates, means = table.rates, table.table[0]
    assert np.max(np.diff(means[0])) > np.max(np.diff(means[1]))
    best = _best_split(rates, means, target)
    assert _on_interpolants(rates, means, plan.spend) == pytest.approx(best, rel=1e-12, abs=0.0)
    np.testing.assert_allclose(plan.spend, [0.0, target], rtol=0.0, atol=1e-12)


def test_an_s_shaped_mean_is_planned_where_its_envelope_falls_short(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Two Hill means of slope 3 at half-saturation points 1 and 1.01, each channel's range 0 to
    1.6 a period, and 1.6 a period to spend. Down the envelopes the budget splits about 1.27 / 0.33,
    the second channel inside its chord from nothing, for about 0.705 a period of the curves; all
    on the first returns 1.6^3 / (1.6^3 + 1). The search returns that, and stopped at its first box
    it returns the envelopes' plan, with a warning."""
    target, tv, search = 1.6, Channel("tv"), Channel("search")
    panel = _one_draw_panel((tv, search), 20)
    reading = _one_draw(
        (tv, search),
        retention=[0.5, 0.5],
        half=[1.0, 1.01],
        slope=[3.0, 3.0],
        coefficient=[1.0, 1.0],
        length=[1],
    )
    limits = {tv: Range(0.0, target), search: Range(0.0, target)}
    plan = plan_posterior(panel, NATIONAL, reading, 13 * target, 13, limits)
    np.testing.assert_allclose(plan.spend, [target, 0.0], rtol=0.0, atol=1e-12)
    assert plan.gain == pytest.approx(13 * target**3 / (target**3 + 1.0), rel=1e-12, abs=0.0)
    monkeypatch.setattr(posterior, "_BOXES", 1)
    with caplog.at_level(logging.WARNING, logger=posterior.__name__):
        first = plan_posterior(panel, NATIONAL, reading, 13 * target, 13, limits)
    assert "the plan stopped at 1 boxes" in caplog.text
    assert first.spend[1] > 0.3
    assert first.gain < 0.9 * plan.gain
    assert first.gain == pytest.approx(13 * 0.705, rel=2e-3, abs=0.0)


@st.composite
def _s_shaped(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray, float]:
    """Two or three channels, each mean a Hill curve of slope 0.5 to 5 tilted by a line, on a grid
    small enough to search by brute force, and a budget anywhere the ranges can spend."""
    channels = draw(st.integers(2, 3))
    grid = 41 if channels == 2 else 13
    unit = st.floats(0.0, 1.0)
    lower = np.array([draw(unit) for _ in range(channels)])
    upper = lower + np.array([draw(st.floats(0.5, 3.0)) for _ in range(channels)])
    rates = _rates(lower, upper, grid)
    means = np.empty_like(rates)
    for c in range(channels):
        half, slope = draw(st.floats(0.3, 2.5)), draw(st.floats(0.5, 5.0))
        top, tilt = draw(st.floats(0.5, 3.0)), draw(st.floats(-0.05, 0.05))
        means[c] = top * rates[c] ** slope / (rates[c] ** slope + half**slope) + tilt * rates[c]
    target = float(lower.sum() + draw(unit) * (upper.sum() - lower.sum()))
    return rates, means, target


@settings(max_examples=300, deadline=None)
@given(_s_shaped())
def test_the_plan_is_the_best_split_a_brute_force_finds(
    case: tuple[np.ndarray, np.ndarray, float],
) -> None:
    rates, means, target = case
    spend = _allocated(rates, means, target)
    best = _best_split(rates, means, target)
    assert abs(spend.sum() - target) <= 1e-12 * max(1.0, target)
    assert np.all(spend >= rates[:, 0] - 1e-12)
    assert np.all(spend <= rates[:, -1] + 1e-12)
    worth = _on_interpolants(rates, means, spend)
    assert best - 2e-9 * abs(best) <= worth <= best + 1e-12 * abs(best)


@settings(max_examples=300, deadline=None)
@given(_s_shaped(), st.data())
def test_a_boxs_fill_bounds_every_split_in_the_box(
    case: tuple[np.ndarray, np.ndarray, float], data: st.DataObject
) -> None:
    """A box of each channel's grid points and a budget the box can spend: the fill down the
    envelopes of the box's points spends it inside the box, and its worth on the envelopes is at
    least every split's in the box on the interpolants, a brute force's best."""
    rates, means, _ = case
    channels, grid = rates.shape
    box = []
    for _ in range(channels):
        lo = data.draw(st.integers(0, grid - 2))
        box.append((lo, data.draw(st.integers(lo + 1, grid - 1))))
    least = sum(rates[c, lo] for c, (lo, _) in enumerate(box))
    most = sum(rates[c, hi] for c, (_, hi) in enumerate(box))
    target = float(least + data.draw(st.floats(0.0, 1.0)) * (most - least))
    spend, top, _ = _fill(rates, means, tuple(box), target, {})
    inside = [slice(lo, hi + 1) for lo, hi in box]
    best = _best_split(
        [rates[c, cells] for c, cells in enumerate(inside)],
        [means[c, cells] for c, cells in enumerate(inside)],
        target,
    )
    assert abs(spend.sum() - target) <= 1e-12 * max(1.0, target)
    for c, (lo, hi) in enumerate(box):
        assert rates[c, lo] - 1e-12 <= spend[c] <= rates[c, hi] + 1e-12
    assert top >= best - 1e-12 * max(1.0, abs(best))


def test_the_gains_are_the_draws_mean_gains_against_nothing_in_the_plan() -> None:
    world = WORLDS[1]
    both = stacked_readings([_fit(world, f) for f in FAMILIES], weights=[0.4, 0.6], draws=7)
    history = np.array(world["history"])
    plan = plan_posterior(
        _panel(world), NATIONAL, both, world["budget"], PERIODS, _limits(world), grid=256
    )
    np.testing.assert_allclose(plan.current, history[-PERIODS:].mean(axis=0), rtol=1e-15)
    for spend, gain in ((plan.spend, plan.gain), (plan.current, plan.current_gain)):
        gains = [
            sum(
                _worth(_response(both, s, c), history[:, c], spend[c])
                - _worth(_response(both, s, c), history[:, c], 0.0)
                for c in range(len(world["channels"]))
            )
            for s in range(both.fit.size)
        ]
        assert gain == pytest.approx(float(np.dot(both.weights, gains)), rel=1e-10)
    assert plan.gain > plan.current_gain
    assert np.all(plan.spend >= np.array(world["lower"]))
    assert np.all(plan.spend <= np.array(world["upper"]))


def _certificate(weights: np.ndarray, pointwise: np.ndarray) -> float:
    density = np.exp(pointwise - pointwise.max(axis=0))
    return float((density / (weights @ density)).mean(axis=1).max() - 1.0)


def _objective(weights: np.ndarray, pointwise: np.ndarray) -> float:
    density = np.exp(pointwise - pointwise.max(axis=0))
    return float(np.mean(np.log(weights @ density)))


def _pointwise_cases() -> list[np.ndarray]:
    rng = np.random.default_rng(20261004)
    recorded = [np.array([w["fits"][f]["pointwise"] for f in FAMILIES]) for w in WORLDS]
    drawn = [rng.normal(-1.0, scale, size=(2, 156)) for scale in (0.05, 0.5, 3.0)]
    shifted = rng.normal(-1.0, 0.5, size=156)
    return [*recorded, *drawn, np.stack([shifted + 0.2, shifted]), np.stack([shifted, shifted])]


@pytest.mark.parametrize("pointwise", _pointwise_cases())
def test_two_fits_are_stacked_by_bisection_where_covers_certificate_holds(
    pointwise: np.ndarray,
) -> None:
    two = _stacking_of_two(pointwise)
    cover = stacking(pointwise)
    assert two.sum() == pytest.approx(1.0, abs=1e-15)
    assert np.all(two >= 0.0)
    assert _certificate(two, pointwise) <= 1e-12
    assert _objective(two, pointwise) >= _objective(cover, pointwise) - 1e-12


def _slope(weight: float, pointwise: np.ndarray) -> float:
    """The stacking objective's slope in the first fit's weight."""
    a, b = np.exp(pointwise - pointwise.max(axis=0))
    with np.errstate(divide="ignore"):
        return float(np.mean((a - b) / (weight * a + (1.0 - weight) * b)))


def _flat_at_an_end() -> list[np.ndarray]:
    """Two observations of three favour the first fit by a factor of 2 and one the second by 2:
    the objective's slope is 0 at the first fit's end, exactly where `exp` rounds `log(1/2)` to
    1/2, and above 0 below it. And the same with the fits swapped."""
    half = math.log(0.5)
    pointwise = np.array([[0.0, 0.0, half], [half, half, 0.0]])
    return [pointwise, pointwise[::-1]]


@pytest.mark.parametrize("pointwise", [*_pointwise_cases(), *_flat_at_an_end()])
def test_two_fits_weight_is_an_end_whose_slope_does_not_point_inward_or_else_the_slopes_root(
    pointwise: np.ndarray,
) -> None:
    """Where neither end's slope points inward the densities are equal at every observation and
    the fits share the weight evenly; inside, the weight's slope is above 0 and the next float's
    is not."""
    weight = _stacking_of_two(pointwise)[0]
    top, bottom = _slope(1.0, pointwise), _slope(0.0, pointwise)
    if top >= 0.0 and bottom <= 0.0:
        assert weight == 0.5
    elif top >= 0.0:
        assert weight == 1.0
    elif bottom <= 0.0:
        assert weight == 0.0
    else:
        assert _slope(weight, pointwise) > 0.0 >= _slope(np.nextafter(weight, 1.0), pointwise)


def test_a_fit_better_at_every_observation_takes_all_the_weight() -> None:
    shifted = np.random.default_rng(3).normal(-1.0, 0.5, size=40)
    np.testing.assert_array_equal(_stacking_of_two(np.stack([shifted + 0.2, shifted])), [1, 0])
    np.testing.assert_array_equal(_stacking_of_two(np.stack([shifted, shifted + 0.2])), [0, 1])


def test_fits_equal_at_every_observation_share_the_weight_as_covers_iteration_leaves_it() -> None:
    shifted = np.random.default_rng(3).normal(-1.0, 0.5, size=40)
    both = np.stack([shifted, shifted])
    np.testing.assert_array_equal(_stacking_of_two(both), [0.5, 0.5])
    np.testing.assert_array_equal(_stacking_of_two(both), stacking(both))


def test_two_fits_weights_ignore_what_each_observations_densities_share() -> None:
    """Each observation's densities moved by one constant, down to where `exp` underflows."""
    pointwise = np.array([WORLDS[0]["fits"][f]["pointwise"] for f in FAMILIES])
    offsets = np.random.default_rng(5).uniform(-900.0, 0.0, size=pointwise.shape[1])
    np.testing.assert_allclose(
        _stacking_of_two(pointwise + offsets), _stacking_of_two(pointwise), rtol=0.0, atol=1e-9
    )


def _normal_fit(world: dict[str, Any], family: str, centre: float, seed: int) -> BayesFit:
    """A fit of the recorded draws whose log-likelihood is a normal mean model's at 30
    observations, its mean's draws about `centre`."""
    rng = np.random.default_rng(seed)
    y = np.random.default_rng(0).normal(0.3, 1.0, size=30)
    mu = rng.normal(centre, 0.2, size=(4, 100))
    log_likelihood = -0.5 * (y - mu[..., None]) ** 2 - 0.5 * math.log(2.0 * math.pi)
    return _fit(world, family, log_likelihood)


def test_stacked_solves_the_weights_from_the_fits_leave_one_out_densities() -> None:
    world = WORLDS[0]
    fits = [
        _normal_fit(world, family, centre, seed)
        for family, centre, seed in (("hill", 0.0, 1), ("tanh", 0.6, 2), ("hill", 0.3, 3))
    ]
    pointwise = np.stack([loo(f.log_likelihood).pointwise for f in fits])
    one, two, three = stacked_readings(fits[:1]), stacked_readings(fits[:2]), stacked_readings(fits)
    np.testing.assert_array_equal(one.stacking, [1.0])
    np.testing.assert_array_equal(two.stacking, _stacking_of_two(pointwise[:2]))
    np.testing.assert_array_equal(three.stacking, stacking(pointwise))
    assert 0.0 < two.stacking[0] < 1.0
    np.testing.assert_array_equal(two.pointwise, pointwise[:2])
    assert stacked_readings(fits[:2], weights=[0.5, 0.5]).pointwise is None
    assert readings(fits[0]).pointwise is None


def test_readings_keep_each_fits_draws_evenly_over_its_chains_and_share_its_weight() -> None:
    world = WORLDS[1]
    hill, tanh = (_fit(world, f) for f in FAMILIES)
    both = stacked_readings([hill, tanh], weights=[0.25, 0.75], draws=10)
    keep = np.linspace(0, 399, 10).round().astype(np.intp)
    np.testing.assert_array_equal(both.fit, np.repeat([0, 1], 10))
    np.testing.assert_array_equal(both.chain, np.tile(keep // 100, 2))
    np.testing.assert_array_equal(
        both.coefficient[10:], tanh.draws["coefficient"].reshape(400, -1)[keep]
    )
    np.testing.assert_array_equal(both.tanh, np.repeat([False, True], 10))
    assert np.all(np.isnan(both.slope[10:]))
    np.testing.assert_array_equal(both.slope[:10], hill.draws["slope"].reshape(400, -1)[keep])
    np.testing.assert_allclose(both.weights, np.repeat([0.025, 0.075], 10), rtol=1e-15)
    whole = readings(tanh)
    assert whole.fit.size == 400
    np.testing.assert_array_equal(whole.weights, np.full(400, 1 / 400))
    assert whole.channels == tuple(world["channels"])
    with pytest.raises(ValueError, match="count must be from 1"):
        readings(tanh, draws=401)


def _valid() -> dict[str, Any]:
    return {
        "channels": (Channel("tv"), Channel("search")),
        "retention": np.full((3, 2), 0.5),
        "half_saturation": np.full((3, 2), 10.0),
        "slope": np.array([[1.5, 2.0], [1.5, 2.0], [np.nan, np.nan]]),
        "coefficient": np.full((3, 2), 4.0),
        "tanh": np.array([False, False, True]),
        "fit": np.array([0, 0, 1]),
        "chain": np.array([0, 1, 0]),
        "stacking": np.array([0.25, 0.75]),
        "pointwise": None,
        "length": 8,
    }


def test_a_valid_reading_is_read_only_and_weighs_each_fits_draws_alike() -> None:
    reading = Readings(**_valid())
    np.testing.assert_array_equal(reading.weights, [0.125, 0.125, 0.75])
    with pytest.raises(ValueError, match="read-only"):
        reading.retention[0, 0] = 0.1


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"channels": ()}, "each named once"),
        ({"channels": (Channel("tv"), Channel("tv"))}, "each named once"),
        ({"retention": np.full((3, 3), 0.5)}, r"retention must be \(draws >= 1, 2 channels\)"),
        ({"coefficient": np.full(3, 4.0)}, r"coefficient must be \(draws >= 1, 2 channels\)"),
        ({"half_saturation": np.full((2, 2), 10.0)}, "one row a draw"),
        ({"tanh": np.array([0, 0, 1])}, "tanh must hold one boolean a draw"),
        ({"fit": np.array([0.0, 0.0, 1.0])}, "fit must hold one integer a draw"),
        ({"chain": np.array([0, 1])}, "chain must hold one integer a draw"),
        ({"stacking": np.array([0.3, 0.75])}, "sum to 1"),
        ({"stacking": np.array([-0.25, 1.25])}, "at or above 0"),
        ({"stacking": np.array([np.nan, 1.0])}, "at or above 0"),
        ({"stacking": np.array([[0.25, 0.75]])}, "at or above 0"),
        ({"fit": np.array([0, 0, 2])}, "outside the 2 weighted"),
        ({"fit": np.array([0, 0, -1])}, "outside the 2 weighted"),
        ({"chain": np.array([0, -1, 0])}, "negative chain"),
        ({"stacking": np.array([0.25, 0.25, 0.5])}, r"fits \[2\] have none"),
        (
            {
                "tanh": np.array([False, True, True]),
                "slope": np.array([[1.5, 2.0], *[[np.nan] * 2] * 2]),
            },
            r"\[0\] both",
        ),
        ({"retention": np.full((3, 2), 1.5)}, "retention must be in"),
        ({"retention": np.full((3, 2), -0.1)}, "retention must be in"),
        ({"half_saturation": np.full((3, 2), 0.0)}, "above 0"),
        ({"half_saturation": np.full((3, 2), np.inf)}, "must be finite"),
        ({"coefficient": np.full((3, 2), np.nan)}, "must be finite"),
        ({"slope": np.array([[1.5, 2.0], [1.5, 2.0], [1.0, np.nan]])}, "NaN where"),
        ({"slope": np.array([[1.5, 0.0], [1.5, 2.0], [np.nan, np.nan]])}, "NaN where"),
        ({"slope": np.array([[1.5, np.nan], [1.5, 2.0], [np.nan, np.nan]])}, "NaN where"),
        ({"pointwise": np.zeros((3, 4))}, r"pointwise must be \(2 fits"),
        ({"pointwise": np.zeros((2, 0))}, r"pointwise must be \(2 fits"),
        ({"pointwise": np.full((2, 4), -np.inf)}, "must be finite"),
        ({"length": 0}, "whole number of periods"),
        ({"length": True}, "whole number of periods"),
        ({"length": 8.0}, "whole number of periods"),
    ],
)
def test_readings_refuse_what_no_posterior_holds(change: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Readings(**(_valid() | change))


def test_stacked_refuses_fits_of_different_panels_and_weights_not_one_a_fit() -> None:
    world = WORLDS[0]
    hill = _fit(world, "hill")
    other = _fit(WORLDS[1], "tanh")
    with pytest.raises(ValueError, match="at least one fit"):
        stacked_readings([])
    with pytest.raises(ValueError, match=r"fits \[1\] differ"):
        stacked_readings([hill, other], weights=[0.5, 0.5])
    shorter = _fit(world, "tanh")
    object.__setattr__(shorter, "length", hill.length - 1)
    with pytest.raises(ValueError, match=r"fits \[1\] differ"):
        stacked_readings([hill, shorter], weights=[0.5, 0.5])
    observed = _fit(world, "tanh", np.zeros((4, 100, 3)))
    with pytest.raises(ValueError, match=r"fits \[1\] differ"):
        stacked_readings([hill, observed], weights=[0.5, 0.5])
    later = _fit(world, "tanh")
    object.__setattr__(later, "dates", later.dates + np.timedelta64(7, "D"))
    with pytest.raises(ValueError, match=r"fits \[1\] differ"):
        stacked_readings([hill, later], weights=[0.5, 0.5])
    unfitted = _fit(world, "tanh")
    object.__setattr__(unfitted, "fitted", np.concatenate([[False], unfitted.fitted[1:]]))
    with pytest.raises(ValueError, match=r"fits \[1\] differ"):
        stacked_readings([hill, unfitted], weights=[0.5, 0.5])
    with pytest.raises(ValueError, match=r"weights must be one a fit, \(2,\)"):
        stacked_readings([hill, _fit(world, "tanh")], weights=[1.0])


def test_the_current_spend_is_the_mean_over_the_windows_observed_periods() -> None:
    weeks = 20
    filled = np.zeros((1, weeks), dtype=bool)
    filled[0, [-1, -4, -9]] = True
    spend = np.where(filled[..., None], 0.0, 1.0 + np.arange(weeks * 2.0).reshape(1, weeks, 2))
    panel = MediaPanel(
        dates=np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(weeks),
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=(Channel("tv"), Channel("search")),
        controls=(),
        kpi=np.where(filled, np.nan, 1.0),
        kpi_unit=Revenue(),
        spend=spend,
        currency="EUR",
        control_values=np.zeros((1, weeks, 0)),
        filled=filled,
    )
    limits = {Channel("tv"): Range(5.0, 60.0), Channel("search"): Range(5.0, 60.0)}
    plan = plan_posterior(panel, NATIONAL, Readings(**_valid()), 13 * 40.0, 13, limits, grid=64)
    window = spend[0, -13:][~filled[0, -13:]]
    assert window.shape == (10, 2)
    np.testing.assert_allclose(plan.current, window.mean(axis=0), rtol=1e-15)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"geo": "nowhere"}, "no geo 'nowhere'"),
        ({"periods": 0}, "periods must be a whole number, at least 1"),
        ({"periods": True}, "periods must be a whole number, at least 1"),
        ({"grid": 1}, "grid must be a whole number, at least 2"),
        ({"grid": 64.0}, "grid must be a whole number, at least 2"),
        ({"limits": "missing"}, r"missing \['tv'\]"),
        ({"limits": "extra"}, r"no such channel \['radio'\]"),
        ({"limits": "open"}, r"no finite most \['tv'\]"),
        ({"budget": 13 * 9.0}, "outside what the ranges spend"),
        ({"budget": 13 * 61.0}, "outside what the ranges spend"),
        ({"budget": math.nan}, "outside what the ranges spend"),
        ({"panel": "other channels"}, r"no channels \['search'\]"),
        ({"panel": "filled"}, "no observed period among the panel's last 13"),
    ],
)
def test_a_plan_refuses_what_it_cannot_read(change: dict[str, Any], match: str) -> None:
    reading = Readings(**_valid())
    weeks = 20
    channels = (Channel("tv"), Channel("search"))
    filled = np.zeros((1, weeks), dtype=bool)
    if change.get("panel") == "filled":
        filled[0, -13:] = True
    if change.get("panel") == "other channels":
        channels = (Channel("tv"), Channel("radio"))
    spend = np.where(filled[..., None], 0.0, 10.0) * np.ones((1, weeks, 2))
    panel = MediaPanel(
        dates=np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(weeks),
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=channels,
        controls=(),
        kpi=np.where(filled, np.nan, 1.0),
        kpi_unit=Revenue(),
        spend=spend,
        currency="EUR",
        control_values=np.zeros((1, weeks, 0)),
        filled=filled,
    )
    limits = {Channel("tv"): Range(5.0, 30.0), Channel("search"): Range(5.0, 30.0)}
    match change.get("limits"):
        case "missing":
            del limits[Channel("tv")]
        case "extra":
            limits[Channel("radio")] = Range(0.0, 1.0)
        case "open":
            limits[Channel("tv")] = Range(5.0)
    arguments: dict[str, Any] = {
        "geo": NATIONAL,
        "periods": 13,
        "grid": 64,
        "budget": 13 * 20.0,
    } | {k: v for k, v in change.items() if k in ("geo", "periods", "grid", "budget")}
    with pytest.raises(ValueError, match=match):
        plan_posterior(
            panel,
            arguments["geo"],
            reading,
            arguments["budget"],
            arguments["periods"],
            limits,
            grid=arguments["grid"],
        )
    if "budget" not in change and change.get("panel") != "filled":
        with pytest.raises(ValueError, match=match):
            worths(
                panel,
                arguments["geo"],
                reading,
                arguments["periods"],
                limits,
                grid=arguments["grid"],
            )


@pytest.mark.parametrize(("per_period", "end"), [(10.0, [5.0, 5.0]), (60.0, [30.0, 30.0])])
def test_a_budget_at_either_end_of_what_the_ranges_spend_is_planned_at_that_end(
    per_period: float, end: list[float]
) -> None:
    weeks = 20
    panel = MediaPanel(
        dates=np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(weeks),
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=(Channel("tv"), Channel("search")),
        controls=(),
        kpi=np.ones((1, weeks)),
        kpi_unit=Revenue(),
        spend=np.full((1, weeks, 2), 10.0),
        currency="EUR",
        control_values=np.zeros((1, weeks, 0)),
        filled=np.zeros((1, weeks), dtype=bool),
    )
    limits = {Channel("tv"): Range(5.0, 30.0), Channel("search"): Range(5.0, 30.0)}
    reading = Readings(**_valid())
    plan = plan_posterior(panel, NATIONAL, reading, 13 * per_period, 13, limits, grid=64)
    np.testing.assert_allclose(plan.spend, end, rtol=0.0, atol=1e-12)


def _scan(x: np.ndarray, y: np.ndarray) -> list[int]:
    """The upper concave hull's vertices by Andrew's monotone chain: the hull's last vertex is
    popped while it lies on or below the chord from the one before it to the next point."""
    hull: list[int] = []
    for i in range(x.size):
        while len(hull) >= 2 and (y[hull[-1]] - y[hull[-2]]) * (x[i] - x[hull[-2]]) <= (
            y[i] - y[hull[-2]]
        ) * (x[hull[-1]] - x[hull[-2]]):
            hull.pop()
        hull.append(i)
    return hull


@settings(max_examples=300, deadline=None)
@given(st.lists(st.integers(-4, 4), min_size=1, max_size=12), st.booleans())
def test_the_envelopes_vertices_are_the_scans_and_no_point_on_a_chord_is_one(
    rises: list[int], concave: bool
) -> None:
    """Integer rises, sorted to fall where `concave`: a run of equal rises puts points on their
    neighbours' chords, on a curve the one pass would otherwise read whole."""
    steps = sorted(rises, reverse=True) if concave else rises
    y = np.concatenate([[0.0], np.cumsum(steps, dtype=np.float64)])
    x = np.arange(y.size, dtype=np.float64)
    assert _vertices(x, y).tolist() == _scan(x, y)


def test_on_a_tie_an_earlier_channels_stretch_is_spent_first() -> None:
    """Two channels of one strictly concave mean, each stretch of the first tied with the second's:
    at every budget the first channel spends at least the second's, by at most a stretch."""
    rates = np.tile(np.linspace(0.0, 1.0, 64), (2, 1))
    means = np.sqrt(rates)
    stretch = float(np.max(np.diff(rates[0])))
    for target in np.linspace(0.0, 2.0, 1001):
        spend = _allocated(rates, means, float(target))
        assert 0.0 <= spend[0] - spend[1] <= stretch


CVAR_DRAWS = 60


def _arm_readings(world: dict[str, Any], draws: int) -> list[tuple[Response, ...]]:
    """The readings the scorecard's `cvar 0.1` arm plans over: the Hill fit's share of `draws` its
    weight's, rounded, and the `tanh` fit's the rest, each evenly over its fit's draws by
    `BayesFit.channel_draws`."""
    hill = round(world["stacking"]["hill"] * draws)
    out: list[tuple[Response, ...]] = []
    for family, count in (("hill", hill), ("tanh", draws - hill)):
        if count:
            fit = _fit(world, family)
            roi = {"roi": np.zeros(fit.draws["retention"].shape)}
            fit = dataclasses.replace(fit, draws=MappingProxyType(dict(fit.draws) | roi))
            out += [tuple(d[c] for c in world["channels"]) for d in fit.channel_draws(count)]
    return out


@pytest.mark.parametrize("world", WORLDS, ids=lambda w: f"{w['environment']}-{w['seed']}")
def test_the_cvar_plan_is_cvar_allocate_on_the_scorecard_arms_readings(
    world: dict[str, Any],
) -> None:
    """The stacked posterior's plan for its worst tenth against the status quo is the scorecard's
    `cvar 0.1` arm's; on the reference world the Hill fit's weight, 3e-10, gives it no draw."""
    plan = plan_cvar(
        _panel(world),
        NATIONAL,
        _bench_readings(world, "stacked"),
        world["budget"],
        PERIODS,
        _limits(world),
        against=world["status_quo"],
        draws=CVAR_DRAWS,
    )
    direct = cvar_allocate(
        _arm_readings(world, CVAR_DRAWS),
        world["budget"],
        PERIODS,
        level=0.1,
        against=np.array(world["status_quo"]),
        lower=np.array(world["lower"]),
        upper=np.array(world["upper"]),
        history=np.array(world["history"]),
    )
    np.testing.assert_array_equal(plan.spend, direct.spend)
    np.testing.assert_array_equal(plan.gain, direct.gain)
    assert (plan.cvar, plan.bound) == (direct.cvar, direct.bound)
    np.testing.assert_array_equal(plan.against, world["status_quo"])
    assert (plan.level, plan.budget, plan.periods) == (0.1, world["budget"], PERIODS)


@pytest.mark.parametrize("level", [0.2, 1.0])
def test_the_plans_cvar_is_the_mean_of_its_worst_gains_and_never_below_the_references(
    level: float,
) -> None:
    """Over the 400 draws read by default, the plan's CVaR is the mean of its worst 80 gains over
    the reference at 0.2, and of all of them at 1; never below the reference's own 0, nor above the
    bound on every split's."""
    world = WORLDS[1]
    plan = plan_cvar(
        _panel(world),
        NATIONAL,
        _bench_readings(world, "stacked"),
        world["budget"],
        PERIODS,
        _limits(world),
        against=world["status_quo"],
        level=level,
    )
    assert plan.gain.shape == (400,)
    worst = np.sort(plan.gain)[: round(level * 400)]
    assert plan.cvar == pytest.approx(float(worst.mean()), rel=1e-12, abs=0.0)
    assert 0.0 <= plan.cvar <= plan.bound


def _fits_of_ten(stacking: list[float]) -> Readings:
    """Three `tanh` fits of ten draws each, their draws in order, weighted by `stacking`."""
    draws = 30
    return Readings(
        channels=(Channel("tv"), Channel("search")),
        retention=np.full((draws, 2), 0.5),
        half_saturation=np.full((draws, 2), 10.0),
        slope=np.full((draws, 2), np.nan),
        coefficient=np.full((draws, 2), 4.0),
        tanh=np.ones(draws, dtype=bool),
        fit=np.repeat(np.arange(3), 10),
        chain=np.zeros(draws, dtype=np.intp),
        stacking=np.array(stacking),
        pointwise=None,
        length=8,
    )


@pytest.mark.parametrize(
    ("stacking", "draws", "expected"),
    [
        # shares 3.5, 2.1 and 1.4: the floors take 6, and the largest remainder the seventh
        ([0.5, 0.3, 0.2], 7, [0, 3, 6, 9, 10, 19, 20]),
        # 1.5 and 1.5 tie, and the earlier fit takes the third; a fit of weight 0 gives none
        ([0.5, 0.5, 0.0], 3, [0, 9, 10]),
        # whole shares, each spread from its fit's first draw to its last: 4.5 rounds to 4
        ([0.2, 0.3, 0.5], 10, [0, 9, 10, 14, 19, 20, 22, 24, 27, 29]),
    ],
)
def test_each_fit_gives_its_weights_share_of_the_draws_by_largest_remainders(
    stacking: list[float], draws: int, expected: list[int]
) -> None:
    assert _alike(_fits_of_ten(stacking), draws).tolist() == expected


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"draws": 0}, "draws must be a whole number, at least 1"),
        ({"draws": True}, "draws must be a whole number, at least 1"),
        ({"draws": 60.0}, "draws must be a whole number, at least 1"),
        (
            {"draws": 2000},
            r"fits \[0, 1\] hold \[400, 400\] draws, fewer than their shares of 2000",
        ),
        ({"level": 0.0}, r"level=0.0 is not a share of the readings in \(0, 1\]"),
        ({"against": "more"}, "the reference spends .* a period and the budget"),
        ({"geo": "nowhere"}, "no geo 'nowhere'"),
    ],
)
def test_a_cvar_plan_refuses_what_it_cannot_read(change: dict[str, Any], match: str) -> None:
    world = WORLDS[1]
    against = np.array(world["status_quo"]) * (1.1 if change.get("against") == "more" else 1.0)
    with pytest.raises(ValueError, match=match):
        plan_cvar(
            _panel(world),
            change.get("geo", NATIONAL),
            _bench_readings(world, "stacked"),
            world["budget"],
            PERIODS,
            _limits(world),
            against=against,
            level=change.get("level", 0.1),
            draws=change.get("draws", CVAR_DRAWS),
        )


LIFT_ROWS = 12


def _lifted_fit(family: str, kpi: float, lift: float, seed: int, rows: int = LIFT_ROWS) -> BayesFit:
    """`family`'s fit of the first world whose log-likelihood is a normal mean model's at its 156
    weeks and `rows` lift rows after them, the mean's draws about `kpi` at the weeks and about
    `lift` at the rows: the nearer 0.3, the better the fit reads them."""
    world = WORLDS[0]
    weeks = len(world["sales"])
    rng = np.random.default_rng(seed)
    y = np.random.default_rng(0).normal(0.3, 1.0, size=weeks + rows)
    mu = np.concatenate(
        [
            np.repeat(rng.normal(kpi, 0.2, size=(4, 100, 1)), weeks, axis=2),
            np.repeat(rng.normal(lift, 0.2, size=(4, 100, 1)), rows, axis=2),
        ],
        axis=2,
    )
    return _fit(world, family, -0.5 * (y - mu) ** 2 - 0.5 * math.log(2.0 * math.pi))


def test_the_lift_rows_pick_the_family_that_reads_them_best_whatever_the_weeks_say() -> None:
    """The Hill reads the weeks better and the lift rows worse: stacking leans to it, the pick
    takes the `tanh`, and the other way round."""
    hill, tanh = _lifted_fit("hill", 0.3, 1.5, 1), _lifted_fit("tanh", 1.0, 0.3, 2)
    assert stacked_readings([hill, tanh], draws=1).stacking[0] > 0.5
    np.testing.assert_array_equal(pick_by_lift_tests([hill, tanh]), [0.0, 1.0])
    hill, tanh = _lifted_fit("hill", 1.0, 0.3, 1), _lifted_fit("tanh", 0.3, 1.5, 2)
    assert stacked_readings([hill, tanh], draws=1).stacking[1] > 0.5
    np.testing.assert_array_equal(pick_by_lift_tests([hill, tanh]), [1.0, 0.0])


def test_the_pick_is_the_lift_rows_leave_one_out_density_and_the_first_on_a_tie() -> None:
    fits = [_lifted_fit("hill", 0.3, 0.6, 1), _lifted_fit("tanh", 0.3, 0.5, 2)]
    weeks = len(WORLDS[0]["sales"])
    elpd = [loo(f.log_likelihood[:, :, weeks:]).elpd for f in fits]
    np.testing.assert_array_equal(pick_by_lift_tests(fits), np.eye(2)[int(np.argmax(elpd))])
    np.testing.assert_array_equal(pick_by_lift_tests([fits[1], fits[1]]), [1.0, 0.0])


def test_fits_that_read_no_lift_row_are_left_to_stacking() -> None:
    fits = [_lifted_fit(f, 0.3, 0.3, k, rows=0) for k, f in enumerate(FAMILIES)]
    assert pick_by_lift_tests(fits) is None


@pytest.mark.parametrize(
    ("fits", "match"),
    [
        (list, "at least one fit"),
        (
            lambda: [_lifted_fit("hill", 0.3, 0.3, 1), _lifted_fit("tanh", 0.3, 0.3, 2, rows=4)],
            r"read different observations, \(KPI's, all\) \[\(156, 160\), \(156, 168\)\]",
        ),
    ],
)
def test_a_pick_refuses_what_it_cannot_read(fits, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        pick_by_lift_tests(fits())
