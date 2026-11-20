"""The DLM over media: each channel's spend carried over and saturated before its random-walk
coefficient multiplies it, and what a planner reads off the fit.

The world: 104 weeks of television, on three weeks in six, and search, always on, beside a price
about 10, with a KPI of `1000 + 300 m_t tv~ + 150 search~ - 20 (price - 10) + N(0, 10²)`, each `~`
the channel's spend through its transform and television's effect drifting, `log m` a random walk
of 0.03 a week from 0. One week's KPI is missing.

The oracles: a transform's column against its definition in plain NumPy; at discount 1 the filter
against the batch conjugate regression under the same prior; each channel, run over the spend up to
a period, against the filter's coefficient there times the column there; and the search's best
against filtering every spec of a grid small enough to filter whole, on one channel and on two.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import logging
import math
from collections import Counter
from types import MappingProxyType

import jax
import numpy as np
import pytest
from chc import response
from chc.allocation import allocate, allocate_geos, minimax_allocate
from scipy import stats

from mixdyn import NATIONAL, Cadence, Channel, MediaPanel, Revenue
from mixdyn.decisions import GeoLimits, Range, plan_geos
from mixdyn.models import (
    DiscountGrid,
    DlmChoice,
    DlmFit,
    DlmSpec,
    Edge,
    Effect,
    GeoDlmSpec,
    PerEffect,
    TransformGrid,
    Trend,
    choose_dlm,
    fit_dlm,
    forecast_dlm,
    search_media,
    transforms_from_bayes,
    vague_prior,
)
from mixdyn.models.bayes_mmm import (
    BayesFit,
    BayesMedia,
    BayesSpec,
    HillCurve,
    Saturation,
    TanhCurve,
)
from mixdyn.models.dlm import _carried, _planned
from mixdyn.models.dlm_choice import _coefficient_draws
from mixdyn.models.dlm_media import _block, _nearest, _Point, _Search
from mixdyn.transforms import HillShape, TanhShape, Transform

WEEKS, LENGTH, MISSING = 104, 8, 30
TV = Transform(0.6, 120.0, TanhShape(), LENGTH)
SEARCH = Transform(0.2, 40.0, HillShape(1.5), LENGTH)
TRANSFORMS = {"tv": TV, "search": SEARCH}
GRID = DiscountGrid(trend=(0.99, 1.0), effects=(0.97, 0.99, 0.995, 1.0))
"""Fine enough near 1 that the likelihood ratio keeps several specs on the drifting world."""


def column(spend: np.ndarray, transform: Transform) -> np.ndarray:
    """The saturated adstock by its definition: the geometric weights over the transform's lags
    divided by their sum, nothing spent before the series, then the curve that is half its
    ceiling at the half-saturation point."""
    weights = transform.retention ** np.arange(transform.length)
    weights = weights / weights.sum()
    adstock = np.array(
        [weights[: t + 1] @ spend[t::-1][: transform.length] for t in range(spend.size)]
    )
    z = adstock / transform.half_saturation
    if isinstance(transform.shape, HillShape):
        return z**transform.shape.slope / (1.0 + z**transform.shape.slope)
    return np.tanh(math.atanh(0.5) * z)


def make_panel(weeks: int, seed: int, follow: float = 0.0) -> MediaPanel:
    """The world over `weeks`; search's spend moves with television's by `follow` a euro."""
    rng = np.random.default_rng(seed)
    week = np.arange(weeks)
    tv = np.where(week % 6 < 3, rng.uniform(80.0, 160.0, weeks), 0.0)
    search = rng.uniform(20.0, 60.0, weeks) + follow * tv
    price = 10.0 + 0.5 * rng.standard_normal(weeks)
    drift = np.exp(np.cumsum(0.03 * rng.standard_normal(weeks)))
    kpi = (
        1000.0
        + 300.0 * drift * column(tv, TV)
        + 150.0 * column(search, SEARCH)
        - 20.0 * (price - 10.0)
        + 10.0 * rng.standard_normal(weeks)
    )
    if weeks > MISSING:
        kpi[MISSING] = np.nan
    return MediaPanel(
        dates=np.datetime64("2024-01-01") + np.timedelta64(7, "D") * week,
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=(Channel("tv"), Channel("search")),
        controls=("price",),
        kpi=kpi[None],
        kpi_unit=Revenue(),
        spend=np.column_stack([tv, search])[None],
        currency="EUR",
        control_values=price[None, :, None],
        filled=np.zeros((1, weeks), dtype=bool),
    )


@functools.cache
def panel() -> MediaPanel:
    return make_panel(WEEKS, 20261003)


def spend(name: str) -> np.ndarray:
    return panel().spend[0, :, [str(c) for c in panel().channels].index(name)]


def media_spec(discount: float = 0.99, trend: float = 0.98) -> DlmSpec:
    """A level, the two channels through their transforms and the price as it is."""
    effects = tuple(Effect(c, discount, transform=t) for c, t in TRANSFORMS.items())
    return DlmSpec((*effects, Effect("price", discount)), trend=Trend(1, trend))


@functools.cache
def fitted() -> DlmFit:
    return fit_dlm(panel(), NATIONAL, media_spec())


@functools.cache
def signed() -> DlmFit:
    effects = tuple(Effect(c, 0.99, sign="positive", transform=t) for c, t in TRANSFORMS.items())
    return fit_dlm(panel(), NATIONAL, DlmSpec(effects, trend=Trend(1, 0.98)), draws=100, seed=3)


@functools.cache
def chosen() -> DlmChoice:
    return choose_dlm(panel(), NATIONAL, media_spec(), GRID)


def alone(fit: DlmFit) -> DlmChoice:
    """A choice that kept one spec, `fit`'s."""
    return DlmChoice((fit.spec,), (fit.filtered,), 0, MappingProxyType({0: fit}))


def coefficients(readings: tuple[tuple[response.Channel, ...], ...]) -> np.ndarray:
    return np.array([[float(c.coefficient) for c in reading] for reading in readings])


def retention(channel: response.Channel) -> float:
    assert isinstance(channel.kernel, response.GeometricAdstock)
    return float(channel.kernel.retention)


@pytest.mark.parametrize(
    "transform",
    [TV, SEARCH, Transform(0.0, 50.0, HillShape(3.0), 4), Transform(1.0, 200.0, TanhShape(), 13)],
)
def test_a_transform_s_column_is_its_kernel_and_curve_by_definition(transform: Transform) -> None:
    for name in TRANSFORMS:
        np.testing.assert_allclose(
            transform.column(spend(name)), column(spend(name), transform), rtol=1e-13, atol=0.0
        )


@pytest.mark.parametrize("shape", [TanhShape(), HillShape(0.7), HillShape(2.5)])
def test_each_curve_is_half_its_ceiling_at_the_half_saturation_point(shape) -> None:
    curve = Transform(0.5, 80.0, shape, 4).curve()
    assert float(curve(80.0)) == pytest.approx(0.5, rel=1e-14, abs=0.0)
    assert float(curve(8e13)) == pytest.approx(1.0, rel=1e-6)


def test_a_transform_needs_double_precision() -> None:
    with jax.enable_x64(False), pytest.raises(ValueError, match="double precision"):
        TV.column(spend("tv"))


def test_the_fit_reads_a_transformed_effect_s_column_and_a_control_as_it_is() -> None:
    fit = fitted()
    np.testing.assert_array_equal(fit.x[:, 0], TV.column(spend("tv")))
    np.testing.assert_array_equal(fit.x[:, 1], SEARCH.column(spend("search")))
    np.testing.assert_array_equal(fit.x[:, 2], panel().control_values[0, :, 0])
    contribution = fit.smoothed.mean[:, 1] * TV.column(spend("tv"))
    np.testing.assert_allclose(fit.contributions["tv"], contribution, rtol=1e-15)


def test_at_discount_one_the_filter_is_the_batch_conjugate_regression() -> None:
    """Every discount 1: the state does not move, and the filter at the last week is the
    normal-gamma regression of the observed weeks on a constant and the effects' columns, each
    channel's by its definition, under the prior the filter started from."""
    spec = media_spec(discount=1.0, trend=1.0)
    prior = vague_prior(panel(), NATIONAL, spec)
    filtered = fit_dlm(panel(), NATIONAL, spec, prior).filtered
    y = panel().kpi[0]
    seen = np.isfinite(y)
    design = np.column_stack(
        [
            np.ones(WEEKS),
            column(spend("tv"), TV),
            column(spend("search"), SEARCH),
            panel().control_values[0, :, 0],
        ]
    )[seen]
    unit = prior.covariance / prior.scale
    precision = np.linalg.inv(unit) + design.T @ design
    mean = np.linalg.solve(precision, np.linalg.solve(unit, prior.mean) + design.T @ y[seen])
    errors, shift = y[seen] - design @ mean, mean - prior.mean
    dof = prior.dof + seen.sum()
    scale = (prior.dof * prior.scale + errors @ errors + shift @ np.linalg.solve(unit, shift)) / dof
    covariance = np.linalg.inv(precision) * scale

    np.testing.assert_allclose(filtered.mean[-1], mean, rtol=1e-9)
    assert filtered.dof[-1] == dof
    assert filtered.scale[-1] == pytest.approx(scale, rel=1e-9)
    norm = 1.0 / np.sqrt(np.diag(covariance))
    np.testing.assert_allclose(
        filtered.covariance[-1] * np.outer(norm, norm),
        covariance * np.outer(norm, norm),
        rtol=0.0,
        atol=1e-9,
    )


@pytest.mark.parametrize("at", [-1, 60, MISSING])
def test_each_channel_returns_the_filter_s_coefficient_times_the_column_at_its_period(at) -> None:
    """Run over the spend up to the period, a channel returns there the coefficient given the
    weeks up to it times the column, computed by its definition."""
    fit = fitted()
    period = at % WEEKS
    channels = fit.channels(at)
    assert list(channels) == list(TRANSFORMS)
    for j, (name, transform) in enumerate(TRANSFORMS.items()):
        expected = fit.filtered.mean[period, 1 + j] * column(spend(name), transform)[period]
        ran = float(channels[name](spend(name)[: period + 1])[-1])
        assert ran == pytest.approx(expected, rel=1e-12)


def test_a_signed_fit_s_channels_read_its_truncated_draws_at_the_last_period_only() -> None:
    fit = signed()
    assert fit.signed is not None
    states = fit.signed.pooled.states[:, -1]
    for at in (-1, WEEKS - 1):
        channels = fit.channels(at)
        for j, name in enumerate(TRANSFORMS):
            assert float(channels[name].coefficient) == pytest.approx(
                states[:, 1 + j].mean(), rel=1e-14, abs=0.0
            )
    with pytest.raises(ValueError, match="read at the last, not at 50"):
        fit.channels(50)


def test_a_mixdyn_plan_and_chc_s_allocation_plan_on_the_channels() -> None:
    """One national geo: `plan_geos` is chc's `allocate_geos`, which plans the search channel's
    S-curve on its envelope, and `allocate` searches it for the best split. Read on the channels,
    the searched plan gains at least as much, and no more than the shortfall `plan_geos` reports."""
    channels = fitted().channels()
    names = [Channel(n) for n in channels]
    budget, periods = 4 * 150.0, 4
    plan = plan_geos(
        panel(),
        {NATIONAL: dict(zip(names, channels.values(), strict=True))},
        budget,
        periods,
        GeoLimits(cells={(NATIONAL, n): Range(0.0, 200.0) for n in names}),
    )
    history = panel().spend[0]
    geo = allocate_geos(
        (tuple(channels.values()),),
        budget,
        periods,
        lower=np.zeros((1, 2)),
        upper=np.full((1, 2), 200.0),
        history=history[:, None, :],
    )
    alone = allocate(
        list(channels.values()),
        budget,
        periods,
        lower=np.zeros(2),
        upper=np.full(2, 200.0),
        history=history,
    )
    assert periods * plan.spend.sum() == pytest.approx(budget, rel=1e-12)
    np.testing.assert_allclose(plan.spend, geo.spend, rtol=1e-9)
    slack = 1e-12 * alone.worth
    assert plan.gain <= alone.gain + (alone.bound - alone.worth) + slack
    assert alone.gain - plan.gain <= plan.shortfall + slack


def test_the_discounts_are_chosen_on_the_transformed_columns_with_the_transforms_held() -> None:
    choice = chosen()
    assert len(choice.specs) == 8
    assert {tuple(e.transform for e in s.effects) for s in choice.specs} == {(TV, SEARCH, None)}
    for filtered in choice.filtered:
        np.testing.assert_array_equal(filtered.design[:, 1], TV.column(spend("tv")))
        np.testing.assert_array_equal(filtered.design[:, 2], SEARCH.column(spend("search")))


def test_readings_are_tuples_of_the_channels_planned_by_allocate_and_hedged_whole() -> None:
    """Each reading planned alone by `allocate`, and all nine at once by `minimax_allocate`."""
    choice = chosen()
    readings = choice.readings(9, seed=4)
    assert len(readings) == 9
    names = list(choice.channels())
    for reading in readings:
        assert isinstance(reading, tuple)
        assert len(reading) == len(names)
        for channel, name in zip(reading, names, strict=True):
            assert isinstance(channel, response.Channel)
            assert retention(channel) == TRANSFORMS[name].retention
            assert channel.kernel.length == LENGTH
    plan = allocate(
        readings[0],
        600.0,
        4,
        lower=np.zeros(2),
        upper=np.full(2, 200.0),
        history=panel().spend[0],
    )
    assert plan.spend.sum() == pytest.approx(150.0, rel=1e-12)
    hedged = minimax_allocate(
        readings,
        600.0,
        4,
        lower=np.zeros(2),
        upper=np.full(2, 200.0),
        history=panel().spend[0],
    )
    assert hedged.spend.sum() == pytest.approx(150.0, rel=1e-9)
    assert hedged.regret.shape == (9,)
    again = coefficients(choice.readings(9, seed=4))
    np.testing.assert_array_equal(coefficients(readings), again)
    assert not np.array_equal(coefficients(choice.readings(9, seed=5)), again)


def test_readings_spread_alike_over_the_kept_specs_each_with_its_own_transforms() -> None:
    """Two kept specs whose television retains 0.6 and 0.3: each takes half the draws, the odd one
    goes to one of them, and the readings come shuffled."""
    other = fit_dlm(
        panel(),
        NATIONAL,
        DlmSpec(
            (
                Effect("tv", 0.99, transform=dataclasses.replace(TV, retention=0.3)),
                Effect("search", 0.99, transform=SEARCH),
                Effect("price", 0.99),
            ),
            trend=Trend(1, 0.98),
        ),
    )
    first = fitted()
    choice = DlmChoice(
        (first.spec, other.spec),
        (first.filtered, other.filtered),
        0,
        MappingProxyType({0: first, 1: other}),
    )
    for draws, counts in ((8, [4, 4]), (7, [3, 4])):
        readings = choice.readings(draws, seed=2)
        retentions = [retention(r[0]) for r in readings]
        assert sorted([retentions.count(0.6), retentions.count(0.3)]) == counts
        assert retentions != sorted(retentions, reverse=True)
    with pytest.raises(ValueError, match="transforms differ"):
        choice.coefficient_bounds()


def test_each_kept_spec_draws_its_readings_from_a_stream_of_its_own() -> None:
    """Two kept specs alike: drawn from one stream, each reading would come twice."""
    fit = fitted()
    twice = DlmChoice(
        (fit.spec, fit.spec), (fit.filtered, fit.filtered), 0, MappingProxyType({0: fit, 1: fit})
    )
    drawn = coefficients(twice.readings(40, seed=6))
    assert len({tuple(row) for row in drawn}) == 40


def test_no_kept_spec_draws_its_readings_from_the_stream_that_spreads_them() -> None:
    """numpy pads a seed with zeros, so a first spec drawn from `(seed, 0)` would read the bits
    that spread the draws over the specs and order them."""
    fit = fitted()
    drawn = coefficients(alone(fit).readings(6, seed=8))
    spreading = _coefficient_draws(fit, [1, 2], 6, np.random.default_rng(8))
    assert not np.isin(drawn, spreading).any()


QUARTER = {"tv": np.zeros(4), "search": np.full(4, 40.0), "price": np.full(4, 10.0)}
"""Four weeks with television dark, so its column is the history's carryover alone."""


def test_a_transformed_effect_s_plan_is_its_spend_carried_over_from_the_fit_s() -> None:
    """Television's column over the four dark weeks is the history's spend carried over and
    saturated, by the transform's definition; without the history it would be 0."""
    fit = fitted()
    ahead = forecast_dlm(fit, 4, QUARTER)
    coefficient = fit.filtered.mean[-1, 1]
    carried = column(np.concatenate([spend("tv"), QUARTER["tv"]]), TV)[-4:]
    np.testing.assert_allclose(ahead.contributions["tv"], coefficient * carried, rtol=1e-12)
    assert (carried > 0.0).all()
    assert np.array_equal(column(QUARTER["tv"], TV), np.zeros(4))


def test_each_kept_spec_draws_its_forecast_from_a_stream_of_its_own() -> None:
    """Two kept specs alike: drawn from one stream, each draw would come twice."""
    fit = fitted()
    twice = DlmChoice(
        (fit.spec, fit.spec), (fit.filtered, fit.filtered), 0, MappingProxyType({0: fit, 1: fit})
    )
    drawn = twice.forecast(4, QUARTER, 40, seed=6)
    assert drawn.kpi.shape == (40, 4)
    assert len({tuple(row) for row in drawn.kpi}) == 40
    np.testing.assert_array_equal(twice.forecast(4, QUARTER, 40, seed=6).kpi, drawn.kpi)


def test_no_spec_draws_its_forecast_from_a_stream_its_readings_or_their_spread_take() -> None:
    """numpy pads a seed with zeros: a spec's forecast drawn from `(seed, k + 1)` would read its
    readings' bits, and one from `(seed, 0)` the bits that spread them."""
    fit = fitted()
    drawn = alone(fit).forecast(4, QUARTER, 6, seed=8).kpi
    x = _planned(fit, 4, QUARTER)
    for stream in ((8, 1), 8):
        taken = _carried(fit, x, 6, np.random.default_rng(stream))[1]
        assert not np.isin(drawn, taken).any()


def test_a_signed_choice_s_forecast_draws_each_spec_s_signed_posterior() -> None:
    fit = signed()
    choice = alone(fit)
    drawn = choice.forecast(13, {"tv": np.full(13, 120.0), "search": np.full(13, 40.0)}, 50, 4)
    assert drawn.kpi.shape == (50, 13)
    assert np.isfinite(drawn.kpi).all()
    assert 0.0 < drawn.acceptance <= 1.0
    with pytest.raises(ValueError, match="draws must be a positive integer"):
        choice.forecast(13, {"tv": np.full(13, 120.0), "search": np.full(13, 40.0)}, 0, 4)


def test_a_spec_s_coefficients_are_drawn_from_its_filter_s_student_t_at_the_last_period() -> None:
    """Twelve weeks, so the variance is learnt from 13 degrees of freedom and the Student t's
    covariance is 13/11 of its scale's: a normal draw would be 15 % narrow. Search's spend follows
    television's, so their coefficients are correlated."""
    short = make_panel(12, 7, follow=0.3)
    fit = fit_dlm(short, NATIONAL, media_spec())
    count = 8000
    draws = coefficients(alone(fit).readings(count, seed=7))
    filtered = fit.filtered
    dof = filtered.dof[-1]
    assert dof == 13.0
    mean = filtered.mean[-1, 1:3]
    covariance = filtered.covariance[-1][1:3, 1:3] * dof / (dof - 2.0)
    sd = np.sqrt(np.diag(covariance))
    assert np.all(np.abs(draws.mean(axis=0) - mean) < 4.0 * sd / math.sqrt(count))
    sample = np.cov(draws.T)
    np.testing.assert_allclose(np.diag(sample), np.diag(covariance), rtol=0.08)
    correlation = covariance[0, 1] / (sd[0] * sd[1])
    assert abs(correlation) > 0.2
    assert sample[0, 1] / np.sqrt(sample[0, 0] * sample[1, 1]) == pytest.approx(
        correlation, abs=0.05
    )


def test_a_signed_spec_s_readings_are_its_own_draws_taken_without_replacement() -> None:
    fit = signed()
    assert fit.signed is not None
    pool = fit.signed.pooled.states[:, -1][:, 1:3]
    drawn = coefficients(alone(fit).readings(50, seed=1))
    rows = [np.flatnonzero((pool == row).all(axis=1)) for row in drawn]
    assert all(r.size == 1 for r in rows)
    assert len({int(r[0]) for r in rows}) == 50
    with pytest.raises(ValueError, match="above the 400"):
        alone(fit).readings(401, seed=1)


def test_a_coefficient_s_interval_is_the_union_of_the_kept_fits_intervals_at_the_last_week() -> (
    None
):
    choice = chosen()
    assert len(choice.kept) > 1
    lows, highs = [], []
    for fit in choice.kept.values():
        filtered = fit.filtered
        half = stats.t.ppf(0.95, filtered.dof[-1]) * np.sqrt(np.diag(filtered.covariance[-1]))
        lows.append(filtered.mean[-1] - half)
        highs.append(filtered.mean[-1] + half)
    bounds = choice.coefficient_bounds(0.9)
    for j, name in enumerate(("tv", "search", "price")):
        assert bounds[name].lower == pytest.approx(np.min(lows, axis=0)[1 + j], rel=1e-14, abs=0.0)
        assert bounds[name].upper == pytest.approx(np.max(highs, axis=0)[1 + j], rel=1e-14, abs=0.0)
    own = alone(choice.fit).coefficient_bounds(0.9)
    assert any(bounds[n].upper - bounds[n].lower > own[n].upper - own[n].lower for n in bounds)


def test_a_signed_fit_s_coefficient_interval_is_its_draws_quantiles() -> None:
    fit = signed()
    assert fit.signed is not None
    draws = fit.signed.pooled.states[:, -1]
    level = 0.8
    bounds = alone(fit).coefficient_bounds(level)
    for j, name in enumerate(TRANSFORMS):
        # 0.5 - 0.8 / 2 is the float below 0.1, and between two draws far enough apart the
        # interpolation keeps that last bit
        low, high = np.quantile(draws[:, 1 + j], [0.5 - level / 2.0, 0.5 + level / 2.0])
        assert (bounds[name].lower, bounds[name].upper) == (low, high)


def bayes_fit(saturation: Saturation) -> BayesFit:
    """A Bayesian fit's posterior as `fit_bayes` holds it, two chains of five draws, drawn at
    random: only `BayesFit.channels` reads it."""
    rng = np.random.default_rng(5)
    shape = (2, 5, 2)
    draws = {
        "roi": rng.uniform(0.5, 3.0, shape),
        "retention": rng.uniform(0.1, 0.9, shape),
        "half_saturation": rng.uniform(20.0, 200.0, shape),
        "coefficient": rng.uniform(100.0, 400.0, shape),
    }
    if isinstance(saturation, HillCurve):
        draws["slope"] = rng.uniform(0.5, 3.0, shape)
    return BayesFit(
        spec=BayesSpec((BayesMedia("tv"), BayesMedia("search")), saturation=saturation),
        dates=panel().dates,
        y=panel().kpi[0],
        fitted=np.ones(WEEKS, dtype=bool),
        draws=MappingProxyType(draws),
        log_likelihood=np.zeros((2, 5, WEEKS)),
        contributions=MappingProxyType({}),
        estimated=np.zeros(WEEKS),
        diagnostics=MappingProxyType({}),
        divergent=np.zeros(2, dtype=np.int64),
        e_bfmi=np.ones(2),
        length=LENGTH,
        spend=MappingProxyType({c: spend(c) for c in ("tv", "search")}),
        cadence=Cadence.WEEK,
    )


@pytest.mark.parametrize("saturation", [HillCurve(), TanhCurve()])
def test_the_first_stage_reads_the_bayesian_fit_s_curves_at_its_medians(saturation) -> None:
    """The transforms are the Bayesian fit's channels at its medians: the same retention, kernel
    length, half-saturation point and slope, so the same columns."""
    fit = bayes_fit(saturation)
    transforms = transforms_from_bayes(fit)
    assert list(transforms) == ["tv", "search"]
    for j, (name, channel) in enumerate(fit.channels().items()):
        transform = transforms[name]
        assert transform.retention == np.median(fit.draws["retention"][..., j])
        assert transform.half_saturation == np.median(fit.draws["half_saturation"][..., j])
        assert transform.length == LENGTH
        if isinstance(saturation, HillCurve):
            assert transform.shape == HillShape(float(np.median(fit.draws["slope"][..., j])))
        else:
            assert transform.shape == TanhShape()
        np.testing.assert_allclose(
            transform.column(spend(name)),
            np.asarray(channel.curve(channel.kernel(spend(name)))),
            rtol=1e-15,
            atol=0.0,
        )


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda: Transform(-0.1, 1.0, TanhShape(), 4), "retention must be in"),
        (lambda: Transform(1.1, 1.0, TanhShape(), 4), "retention must be in"),
        (lambda: Transform(0.5, 0.0, TanhShape(), 4), "half-saturation point must be positive"),
        (lambda: Transform(0.5, math.inf, TanhShape(), 4), "half-saturation point must be"),
        (lambda: Transform(0.5, 1.0, TanhShape(), 0), "length must be a whole number"),
        (lambda: Transform(0.5, 1.0, TanhShape(), True), "length must be a whole number"),
        (
            lambda: Transform(0.5, 1.0, "tanh", 4),  # ty: ignore[invalid-argument-type]
            "shape must be TanhShape",
        ),
        (lambda: HillShape(0.0), "slope must be positive"),
        (lambda: TV.column(np.array([1.0, -1.0])), "nonnegative"),
        (
            lambda: Effect("tv", 0.99, transform=TanhShape()),  # ty: ignore[invalid-argument-type]
            "a transform is a Transform",
        ),
        (
            lambda: fit_dlm(
                panel(), NATIONAL, DlmSpec((Effect("price", 0.99, transform=TV),), Trend(1, 0.98))
            ),
            r"\['price'\] are the panel's controls",
        ),
        (lambda: GeoDlmSpec((Effect("tv", 0.99, transform=TV),)), "not fitted over geos"),
        (
            lambda: forecast_dlm(
                fitted(), 4, {"tv": [1.0, -1.0, 1.0, 1.0], "search": [1.0] * 4, "price": [1.0] * 4}
            ),
            r"the plan's 'tv' spends below 0 at steps \[1\]",
        ),
        (
            lambda: fit_dlm(panel(), NATIONAL, DlmSpec((Effect("tv", 0.99),))).channels(),
            "no effect carries a transform",
        ),
        (lambda: fitted().channels(WEEKS), "at must be a period"),
        (lambda: fitted().channels(-WEEKS - 1), "at must be a period"),
        (lambda: fitted().channels(True), "at must be a period"),
        (lambda: chosen().readings(0, seed=1), "draws must be a positive integer"),
        (lambda: chosen().coefficient_bounds(1.0), r"level must lie in \(0, 1\)"),
    ],
)
def test_what_is_not_a_transform_or_a_planner_s_reading_is_refused(make, message) -> None:
    with pytest.raises(ValueError, match=message):
        make()


COARSE = TransformGrid((0.0, 0.4, 0.8), (0.5, 1.0, 2.0, 4.0), (TanhShape(), HillShape(2.0)))
"""Television's best retention, 0.6, is a point of the grid halved and not of the grid, so the
refined stage has to move to it."""
SMALL = TransformGrid((0.0, 0.4, 0.8), (0.5, 1.0, 2.0), (TanhShape(),))
EFFECT_DISCOUNTS = (0.97, 0.99, 1.0)
EFFECTS = DiscountGrid(effects=EFFECT_DISCOUNTS)
START = Transform(0.3, 100.0, TanhShape(), LENGTH)


def positive_mean(name: str) -> float:
    series = spend(name)
    return float(series[series > 0.0].mean())


def placed(spec: DlmSpec, transforms: dict[str, Transform]) -> DlmSpec:
    """`spec` with the named effects carrying the given transforms."""
    effects = tuple(
        dataclasses.replace(e, transform=transforms[e.column]) if e.column in transforms else e
        for e in spec.effects
    )
    return dataclasses.replace(spec, effects=effects)


def on_grid(grid: TransformGrid, name: str) -> list[Transform]:
    """Every transform of `grid` for a channel, its half-saturation multiples of the channel's mean
    positive spend."""
    scale = positive_mean(name)
    return [
        Transform(r, h * scale, shape, LENGTH)
        for r, h, shape in itertools.product(grid.retentions, grid.half_saturations, grid.shapes)
    ]


def loglik(spec: DlmSpec) -> float:
    return fit_dlm(panel(), NATIONAL, spec).filtered.log_likelihood


def dense(spec: DlmSpec, grid: TransformGrid, names: tuple[str, ...]) -> tuple[DlmSpec, float]:
    """The likelihood's best of every spec of the grid's product over the named channels, at
    every discount of `EFFECTS`, each filtered."""
    best, top = spec, -math.inf
    for transforms in itertools.product(*(on_grid(grid, n) for n in names)):
        for candidate in EFFECTS.specs(placed(spec, dict(zip(names, transforms, strict=True)))):
            value = loglik(candidate)
            if value > top:
                best, top = candidate, value
    return best, top


def one_channel(name: str) -> DlmSpec:
    """The named channel searched from a retention of 0.3 and a half-saturation point of 100; the
    other channel and the price read as they are."""
    effects = tuple(Effect(c, 0.99, transform=START if c == name else None) for c in TRANSFORMS)
    return DlmSpec((*effects, Effect("price", 0.99)), trend=Trend(1, 0.99))


def two_channels() -> DlmSpec:
    effects = tuple(Effect(c, 0.99, transform=START) for c in TRANSFORMS)
    return DlmSpec((*effects, Effect("price", 0.99)), trend=Trend(1, 0.99))


@functools.cache
def searched(name: str) -> DlmChoice:
    return search_media(panel(), NATIONAL, one_channel(name), COARSE, EFFECTS, refinements=1)


def transform_of(spec: DlmSpec, name: str) -> Transform:
    (transform,) = (e.transform for e in spec.effects if e.column == name)
    assert transform is not None
    return transform


def assert_same(found: DlmSpec, expected: DlmSpec) -> None:
    assert found.trend == expected.trend
    for ours, theirs in zip(found.effects, expected.effects, strict=True):
        assert (ours.column, ours.discount) == (theirs.column, theirs.discount)
        if theirs.transform is None:
            assert ours.transform is None
            continue
        assert ours.transform is not None
        assert ours.transform.retention == theirs.transform.retention
        assert ours.transform.half_saturation == pytest.approx(
            theirs.transform.half_saturation, rel=1e-14, abs=0.0
        )
        assert (ours.transform.shape, ours.transform.length) == (
            theirs.transform.shape,
            theirs.transform.length,
        )


def test_a_grid_halved_keeps_its_points_and_puts_new_ones_between_them() -> None:
    """Retentions halved by distance, half-saturation multiples by ratio, at every level the
    coarser points exactly where they were."""
    fine = COARSE.refined(1)
    assert fine.retentions == pytest.approx((0.0, 0.2, 0.4, 0.6, 0.8), rel=1e-15, abs=0.0)
    expected = (0.5, math.sqrt(0.5), 1.0, math.sqrt(2.0), 2.0, math.sqrt(8.0), 4.0)
    assert fine.half_saturations == pytest.approx(expected, rel=1e-15, abs=0.0)
    assert fine.shapes == COARSE.shapes
    assert COARSE.refined(0) == COARSE
    for level in range(4):
        finer = COARSE.refined(level)
        assert len(finer.retentions) == 2 * 2**level + 1
        assert len(finer.half_saturations) == 3 * 2**level + 1
        assert finer.retentions[:: 2**level] == COARSE.retentions
        assert finer.half_saturations[:: 2**level] == COARSE.half_saturations
        assert COARSE.refined(level + 1).half_saturations[::2] == finer.half_saturations


def test_a_grid_counts_each_axis_it_varies_as_a_channel_s_parameter() -> None:
    assert COARSE.parameters == 3
    assert SMALL.parameters == 2
    assert TransformGrid((0.5,), (1.0, 2.0), (TanhShape(),)).parameters == 1


@pytest.mark.parametrize("name", ["tv", "search"])
def test_the_search_s_best_is_the_dense_grid_s_on_one_channel(name: str) -> None:
    """Each channel over the grid halved once, 70 transforms at three discounts: the coarse passes
    and the local passes after them end at the spec the likelihood ranks first of all 210."""
    found = searched(name)
    best, top = dense(one_channel(name), COARSE.refined(1), (name,))
    assert_same(found.fit.spec, best)
    assert found.filtered[found.best].log_likelihood == pytest.approx(top, rel=1e-12)
    assert transform_of(found.fit.spec, name) != START


@pytest.mark.parametrize("name", ["tv", "search"])
def test_a_prior_s_search_ends_at_the_dense_grid_s_best_posterior(name: str) -> None:
    """With a prior on the half-saturation point (ADR 0085), a spec's score is its log-likelihood
    plus the prior's log density of the point's multiple of the channel's mean positive spend: the
    search ends at the dense grid's best of that sum, keeps by it, and a prior tight about the mean
    positive spend holds the point there."""
    spread = 0.5

    def posterior(spec: DlmSpec) -> float:
        multiple = transform_of(spec, name).half_saturation / positive_mean(name)
        return loglik(spec) - 0.5 * (math.log(multiple) / spread) ** 2

    found = search_media(
        panel(),
        NATIONAL,
        one_channel(name),
        COARSE,
        EFFECTS,
        refinements=1,
        half_saturation_prior=spread,
    )
    candidates = [
        candidate
        for transform in on_grid(COARSE.refined(1), name)
        for candidate in EFFECTS.specs(placed(one_channel(name), {name: transform}))
    ]
    best = max(candidates, key=posterior)
    assert_same(found.fit.spec, best)
    top = posterior(found.fit.spec)
    margin = stats.chi2.ppf(0.95, COARSE.parameters + EFFECTS.parameters(one_channel(name))) / 2.0
    assert all(posterior(s) >= top - margin for s in found.specs)
    tight = search_media(
        panel(),
        NATIONAL,
        one_channel(name),
        COARSE,
        EFFECTS,
        refinements=1,
        half_saturation_prior=1e-3,
    )
    held = transform_of(tight.fit.spec, name).half_saturation
    assert held == pytest.approx(positive_mean(name), rel=1e-12, abs=0.0)
    with pytest.raises(ValueError, match="positive and finite"):
        search_media(
            panel(), NATIONAL, one_channel(name), COARSE, EFFECTS, half_saturation_prior=0.0
        )


def logged_search(
    caplog: pytest.LogCaptureFixture, name: str
) -> tuple[DlmChoice, list[dict[str, object]]]:
    """The named channel's search over the grid halved once, and each move it logged."""
    with caplog.at_level(logging.DEBUG, logger="mixdyn.models.dlm_media"):
        found = search_media(panel(), NATIONAL, one_channel(name), COARSE, EFFECTS, refinements=1)
    return found, [r.__dict__ for r in caplog.records if "moved_to" in r.__dict__]


def test_every_move_of_a_search_raises_the_likelihood(caplog: pytest.LogCaptureFixture) -> None:
    """A block moves the search only to a spec of a higher one-step log-likelihood than the
    current one's, so each stage ends. The discounts' moves are logged as the channels' are, the
    trend's discount then each effect's, the trend's held where the grid does not vary it. The
    refined stage moves a channel only to a neighbour of its point on the grid halved:
    television's retention from a point of the grid to one of the grid halved."""
    found, moves = logged_search(caplog, "tv")
    assert {m["stage"] for m in moves} == {0, 1}
    for m in moves:
        before, after = m["before"], m["after"]
        assert isinstance(before, float)
        assert isinstance(after, float)
        assert after > before
    discounts = [m for m in moves if m["block"] == "the discounts"]
    assert discounts
    for m in discounts:
        a, b = m["moved_from"], m["moved_to"]
        assert isinstance(a, tuple)
        assert isinstance(b, tuple)
        assert (a[0], b[0], len(a), len(b)) == (0.99, 0.99, 4, 4)
        assert len(set(b[1:])) == 1
        assert b[1] in EFFECT_DISCOUNTS
    fine, scale = COARSE.refined(1), positive_mean("tv")
    halves = [h * scale for h in fine.half_saturations]
    refined = [m for m in moves if m["stage"] == 1 and m["block"] == "channel 'tv'"]
    assert refined
    for m in refined:
        a, b = m["moved_from"], m["moved_to"]
        assert isinstance(a, Transform)
        assert isinstance(b, Transform)
        assert abs(fine.retentions.index(a.retention) - fine.retentions.index(b.retention)) <= 1
        assert abs(halves.index(a.half_saturation) - halves.index(b.half_saturation)) <= 1
    assert transform_of(found.fit.spec, "tv").retention not in COARSE.retentions


def test_a_search_s_refined_stage_starts_at_the_coarse_stage_s_spec(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Search's coarse best is the best of the grid halved too, so the refined stage, which starts
    there, has no move to make."""
    found, moves = logged_search(caplog, "search")
    best = transform_of(found.fit.spec, "search")
    scale = positive_mean("search")
    assert best.retention in COARSE.retentions
    assert best.half_saturation in [h * scale for h in COARSE.half_saturations]
    assert moves
    assert all(m["stage"] == 0 for m in moves)


def test_a_block_is_the_coarse_grid_whole_and_a_finer_point_s_neighbours_at_every_shape() -> None:
    """On the coarse grid a channel's block is the grid's product, in its order; on a finer one
    the retentions and half-saturation points either side of the channel's point at every shape,
    cut at the grid's ends."""
    fine = COARSE.refined(1)
    whole = [_Point(*p) for p in itertools.product(range(3), range(4), range(2))]
    assert _block(_Point(1, 2, 0), COARSE, coarse=True) == whole
    inner = _block(_Point(2, 3, 1), fine, coarse=False)
    assert set(inner) == {_Point(*p) for p in itertools.product((1, 2, 3), (2, 3, 4), (0, 1))}
    corner = _block(_Point(4, 0, 0), fine, coarse=False)
    assert set(corner) == {_Point(*p) for p in itertools.product((3, 4), (0, 1), (0, 1))}
    assert len(inner) == len(set(inner))


def assert_kept(found: DlmChoice, candidates: list[DlmSpec], parameters: int) -> None:
    """Every kept spec within `chi2_p(0.95) / 2` of the best, the best first by likelihood, and
    of `candidates`, specs the search filtered, exactly those inside the margin kept, more than
    one of them."""
    margin = stats.chi2.ppf(0.95, parameters) / 2.0
    values = np.array([f.log_likelihood for f in found.filtered])
    top = values.max()
    assert found.best == int(np.argmax(values))
    assert np.all(values >= top - margin)
    inside = 0
    for candidate in candidates:
        held = loglik(candidate) >= top - margin
        inside += held
        assert (candidate in found.specs) == held
    assert inside > 1


def test_the_search_s_best_is_the_dense_grid_s_on_two_channels() -> None:
    """Both channels over nine transforms each, 81 pairs at three discounts. The margin counts
    five parameters, two axes a channel and the effects' discount; the last pass moved nothing,
    so it filtered the three discounts and each channel's nine transforms with the other held,
    each kept exactly when inside the margin."""
    found = search_media(panel(), NATIONAL, two_channels(), SMALL, EFFECTS, refinements=0)
    best, top = dense(two_channels(), SMALL, ("tv", "search"))
    assert_same(found.fit.spec, best)
    assert found.filtered[found.best].log_likelihood == pytest.approx(top, rel=1e-12)
    chosen = found.fit.spec
    held = {e.column: e.transform for e in chosen.effects if e.transform is not None}
    candidates = list(EFFECTS.specs(chosen))
    for name in ("tv", "search"):
        candidates += [placed(chosen, held | {name: t}) for t in on_grid(SMALL, name)]
    assert_kept(found, candidates, 5)


@pytest.mark.parametrize("name", ["tv", "search"])
def test_the_search_keeps_every_spec_it_filtered_within_the_likelihood_ratio_s_margin(
    name: str,
) -> None:
    """At `chi2_p(0.95) / 2` with p four, three axes of the channel's and the effects' discount.
    The search ends on a pass in which neither block moved, so it filtered the best transform at
    every discount and the best's neighbours at every shape at the best discount: each of those
    is kept exactly when it is inside the margin. Every kept spec is on the grid halved, the
    start's spec included only were it a point of it."""
    found = searched(name)
    best = found.fit.spec
    transform = transform_of(best, name)
    fine, scale = COARSE.refined(1), positive_mean(name)
    halves = [h * scale for h in fine.half_saturations]
    for spec in found.specs:
        kept = transform_of(spec, name)
        assert kept.retention in fine.retentions
        assert kept.half_saturation in halves
        assert (kept.shape in fine.shapes, kept.length) == (True, LENGTH)
    r, h = fine.retentions.index(transform.retention), halves.index(transform.half_saturation)
    block = [
        placed(best, {name: Transform(fine.retentions[i], halves[k], s, LENGTH)})
        for i in range(max(r - 1, 0), min(r + 2, len(fine.retentions)))
        for k in range(max(h - 1, 0), min(h + 2, len(halves)))
        for s in fine.shapes
    ]
    assert_kept(found, [*EFFECTS.specs(best), *block], 4)


def test_a_search_names_each_end_of_its_grids_its_set_reaches(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """On the tests' world the likelihood rises as television's half-saturation point grows
    towards a straight line, so its best sits at the grid's highest, 4 times the mean positive
    spend, an end where the set may stop short of the data's. Search's set reaches both ends of
    its half-saturation points and the effects' lowest discount, and holds a retention of 0, the
    parameter's own bound and no end. A search logs its ends with the geo."""
    assert searched("tv").edges == (Edge("tv half-saturation", 4.0, best=True),)
    found = searched("search")
    assert found.edges == (
        Edge("effects discount", 0.97, best=False),
        Edge("search half-saturation", 0.5, best=False),
        Edge("search half-saturation", 4.0, best=True),
    )
    assert 0.0 in {transform_of(s, "search").retention for s in found.specs}
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="mixdyn.models.dlm_media"):
        small = search_media(panel(), NATIONAL, one_channel("tv"), SMALL, EFFECTS, refinements=0)
    (record,) = [r for r in caplog.records if "edges" in r.__dict__]
    assert small.edges
    assert record.__dict__["edges"] == [(e.parameter, e.end, e.best) for e in small.edges]
    assert record.__dict__["geo"] == str(NATIONAL)


EACH = DiscountGrid(trend=(0.99, 1.0), effects=PerEffect((0.95, 0.97, 0.99, 1.0)))


@pytest.mark.parametrize("name", ["tv", "search"])
def test_a_search_s_pairs_of_discounts_end_at_the_product_s_best(name: str) -> None:
    """Each effect's discount on its own and the channel's transform held at its truth: the
    search moves the discounts a pair at a time and ends at the spec that filtering all 128 of
    the product ranks first, its set a part of the product's. One discount at a time stops 1.24
    below it on television, where search's and the price's discounts trade off. The last pass
    moved nothing, so it filtered every pair about the best: each is kept exactly when it is
    inside the margin at four parameters, the trend's discount and each effect's."""
    truth = TRANSFORMS[name]
    point = TransformGrid(
        (truth.retention,), (truth.half_saturation / positive_mean(name),), (truth.shape,)
    )
    found = search_media(panel(), NATIONAL, one_channel(name), point, EACH, refinements=0)
    held = placed(one_channel(name), {name: transform_of(found.fit.spec, name)})
    product = choose_dlm(panel(), NATIONAL, held, EACH)
    assert len(product.specs) == 128
    assert found.fit.spec == product.fit.spec
    top = found.filtered[found.best].log_likelihood
    assert top == product.filtered[product.best].log_likelihood
    assert set(found.specs) <= {product.specs[k] for k in product.kept}
    pairs = dict.fromkeys(itertools.chain.from_iterable(EACH.blocks(found.fit.spec)))
    assert_kept(found, list(pairs), 4)


def test_a_search_moves_each_effect_s_discount_a_pair_at_a_time(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Every move raises the likelihood, and a move of the discounts sets at most two of them, a
    block built about the spec the last block moved to."""
    with caplog.at_level(logging.DEBUG, logger="mixdyn.models.dlm_media"):
        search_media(panel(), NATIONAL, one_channel("tv"), SMALL, EACH, refinements=0)
    moves = [r.__dict__ for r in caplog.records if "moved_to" in r.__dict__]
    discounts = [m for m in moves if m["block"] == "the discounts"]
    assert len(discounts) > 1
    for m in moves:
        before, after = m["before"], m["after"]
        assert isinstance(before, float)
        assert isinstance(after, float)
        assert after > before
    for m in discounts:
        a, b = m["moved_from"], m["moved_to"]
        assert isinstance(a, tuple)
        assert isinstance(b, tuple)
        assert 1 <= sum(x != y for x, y in zip(a, b, strict=True)) <= 2


def test_a_search_moves_a_block_only_to_a_better_spec_and_breaks_other_ties_by_order() -> None:
    """A block's best is the current spec wherever it ties the block's top, so every move raises
    the likelihood and a pass that moves nothing ends the stage; otherwise the first option at
    the top."""
    low, high, other = (media_spec(d) for d in (0.97, 0.99, 1.0))
    search = _Search(panel(), NATIONAL, 1.0, scores={low: 1.0, high: 2.0, other: 2.0}, near={})
    assert search.best(other, [low, high, other]) == other
    assert search.best(low, [low, high, other]) == high
    assert search.best(low, [low, other, high]) == other


def test_a_search_keeps_each_channel_s_kernel_length() -> None:
    start = Transform(0.3, 100.0, TanhShape(), 3)
    spec = DlmSpec(
        (Effect("tv", 0.99, transform=start), Effect("price", 0.99)), trend=Trend(1, 0.99)
    )
    grid = TransformGrid((0.0, 0.5), (1.0, 2.0), (TanhShape(),))
    found = search_media(panel(), NATIONAL, spec, grid, EFFECTS, refinements=0)
    lengths = set()
    for kept in found.specs:
        transform = kept.effects[0].transform
        assert transform is not None
        lengths.add(transform.length)
    assert lengths == {3}


def curve_key(channel: response.Channel) -> tuple[float, str, float]:
    """A channel's carryover and curve: its retention, its curve's family and scale."""
    curve = channel.curve
    assert isinstance(curve, response.Tanh | response.Hill)
    return retention(channel), type(curve).__name__, float(curve.scale)


def test_a_search_holds_only_its_kept_specs_each_fitted_and_read_by_a_planner() -> None:
    """Twice as many readings as kept specs: each spec's transforms come twice."""
    found = searched("tv")
    assert sorted(found.kept) == list(range(len(found.specs)))
    for k, spec in enumerate(found.specs):
        assert found.kept[k].spec == spec
        assert found.kept[k].filtered.log_likelihood == found.filtered[k].log_likelihood
    assert list(found.channels()) == ["tv"]
    kept = []
    for spec in found.specs:
        kept += 2 * [curve_key(transform_of(spec, "tv").channel(1.0))]
    readings = found.readings(2 * len(found.specs), seed=3)
    assert Counter(curve_key(r[0]) for r in readings) == Counter(kept)


def test_a_search_fits_its_kept_specs_at_evenly_spaced_ranks_of_their_likelihood() -> None:
    """Six of the kept specs, at ranks `floor(i (N - 1) / 5)` of their log-likelihood from the
    best: the best first and the last kept last, and ranks that are not their own mirror image, so
    that the direction of the ranking shows. The choice holds every kept spec and its filter, as
    the search that fits them all does, and its readings come from the six alone; a cap above the
    kept count fits them all."""
    every = searched("tv")
    count, fitted = len(every.specs), 6
    picks = [i * (count - 1) // (fitted - 1) for i in range(fitted)]
    assert sorted(picks) != sorted(count - 1 - p for p in picks)
    found = search_media(
        panel(), NATIONAL, one_channel("tv"), COARSE, EFFECTS, refinements=1, fitted=fitted
    )
    assert found.specs == every.specs
    likelihoods = [f.log_likelihood for f in every.filtered]
    assert [f.log_likelihood for f in found.filtered] == likelihoods
    ranks = sorted(range(count), key=lambda k: -likelihoods[k])
    assert sorted(found.kept) == sorted(ranks[p] for p in picks)
    assert {ranks[0], ranks[-1]} <= set(found.kept)
    assert (found.best, found.fit.spec) == (every.best, every.fit.spec)
    for k, fit in found.kept.items():
        assert fit.spec == found.specs[k]
    drawn = [curve_key(transform_of(found.specs[k], "tv").channel(1.0)) for k in found.kept]
    readings = found.readings(2 * fitted, seed=3)
    assert Counter(curve_key(r[0]) for r in readings) == Counter(2 * drawn)
    whole = search_media(
        panel(), NATIONAL, one_channel("tv"), COARSE, EFFECTS, refinements=1, fitted=count + 1
    )
    assert sorted(whole.kept) == list(range(count))


def test_a_search_starts_at_the_grid_point_nearest_each_channel_s_transform() -> None:
    """The nearest retention by distance, the nearest half-saturation point by ratio, and the
    transform's own shape where the grid has it, else the grid's first."""
    scale = positive_mean("tv")
    near = Transform(0.5, 2.9 * scale, HillShape(2.0), LENGTH)
    assert _nearest(COARSE, near, scale) == _Point(1, 3, 1)
    other = Transform(0.7, 0.6 * scale, HillShape(3.0), LENGTH)
    assert _nearest(COARSE, other, scale) == _Point(2, 0, 0)


def silent_panel() -> MediaPanel:
    quiet = panel().spend.copy()
    quiet[0, :, 0] = 0.0
    return dataclasses.replace(panel(), spend=quiet)


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda: TransformGrid((), (1.0,), (TanhShape(),)), "retentions must ascend"),
        (
            lambda: TransformGrid((0.5, 0.3), (1.0,), (TanhShape(),)),
            "retentions must ascend",
        ),
        (
            lambda: TransformGrid((0.5, 1.2), (1.0,), (TanhShape(),)),
            "retentions must ascend",
        ),
        (
            lambda: TransformGrid((0.3, 0.3), (1.0,), (TanhShape(),)),
            "retentions must ascend",
        ),
        (
            lambda: TransformGrid((0.5,), (0.0, 1.0), (TanhShape(),)),
            "multiples must ascend",
        ),
        (
            lambda: TransformGrid((0.5,), (1.0, math.inf), (TanhShape(),)),
            "multiples must ascend",
        ),
        (
            lambda: TransformGrid((0.5,), (2.0, 1.0), (TanhShape(),)),
            "multiples must ascend",
        ),
        (
            lambda: TransformGrid((0.5,), (1.0, 1.0), (TanhShape(),)),
            "multiples must ascend",
        ),
        (lambda: TransformGrid((0.5,), (1.0,), ()), "shapes must name at least one"),
        (
            lambda: TransformGrid((0.5,), (1.0,), (TanhShape(), TanhShape())),
            "none twice",
        ),
        (
            lambda: TransformGrid((0.5,), (1.0,), ("tanh",)),  # ty: ignore[invalid-argument-type]
            "a shape is TanhShape",
        ),
        (lambda: COARSE.refined(-1), "level must be a whole number"),
        (lambda: COARSE.refined(True), "level must be a whole number"),
        (
            lambda: search_media(
                panel(), NATIONAL, one_channel("tv"), COARSE, EFFECTS, refinements=-1
            ),
            "refinements must be a whole number",
        ),
        (
            lambda: search_media(panel(), NATIONAL, one_channel("tv"), COARSE, EFFECTS, level=1.0),
            r"level must lie in \(0, 1\)",
        ),
        (
            lambda: search_media(panel(), NATIONAL, one_channel("tv"), COARSE, EFFECTS, fitted=0),
            "fitted must be a whole number at or above 1",
        ),
        (
            lambda: search_media(
                panel(), NATIONAL, one_channel("tv"), COARSE, EFFECTS, fitted=True
            ),
            "fitted must be a whole number at or above 1",
        ),
        (
            lambda: search_media(
                panel(),
                NATIONAL,
                one_channel("tv"),
                COARSE,
                EFFECTS,
                fitted=2.0,  # ty: ignore[invalid-argument-type]
            ),
            "fitted must be a whole number at or above 1",
        ),
        (
            lambda: search_media(
                panel(), NATIONAL, DlmSpec((Effect("tv", 0.99),)), COARSE, EFFECTS
            ),
            "no effect carries a transform",
        ),
        (
            lambda: search_media(silent_panel(), NATIONAL, one_channel("tv"), COARSE, EFFECTS),
            "spends nothing",
        ),
        (
            lambda: search_media(
                panel(),
                NATIONAL,
                DlmSpec((Effect("price", 0.99, transform=START),)),
                COARSE,
                EFFECTS,
            ),
            "'price' is not a channel",
        ),
        (
            lambda: search_media(
                panel(), NATIONAL, one_channel("tv"), COARSE, DiscountGrid(season=(0.99,))
            ),
            "the spec has no season",
        ),
    ],
)
def test_what_is_not_a_search_is_refused(make, message) -> None:
    with pytest.raises(ValueError, match=message):
        make()
