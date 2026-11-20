"""The engine's model over geos: its decomposition is each geo's smoothed forecast, and on panels
drawn from a hierarchy of its own it recovers each geo's effects better than a model per geo,
with intervals that cover them."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest
from chc.dlm import smooth
from scipy import stats

from mixdyn import Cadence, Channel, Geo, MediaPanel, Revenue, allot_national
from mixdyn.models import (
    Effect,
    GeoDlmFit,
    GeoDlmSpec,
    Season,
    Trend,
    fit_dlm,
    fit_geo_dlm,
    geo_effects,
    vague_geo_prior,
)

NATIONAL_EFFECT = np.array([2.0, 1.0])
SPREAD = np.array([0.4, 0.3])
NOISE = 30.0
NOISY = 200.0
"""A noise at which a geo's effect alone is about as uncertain as the geos' spread."""
SPEC = GeoDlmSpec(
    (Effect("search", 1.0), Effect("social", 1.0)), Trend(1, 1.0), Season(52.0, 1, 1.0)
)
REGIONAL = 5
"""A geo's coordinates under `SPEC`: its level, two deviations and a harmonic's pair."""


def deviation(geo: int, effect: int) -> int:
    """Where a geo's deviation on an effect sits in `SPEC`'s stacked state."""
    return 2 + REGIONAL * geo + 1 + effect


def hierarchy(
    seed: int, geos: int = 8, weeks: int = 104, *, noise: float = NOISE, population: bool = True
) -> tuple[MediaPanel, np.ndarray]:
    """A KPI of each geo's base, moved by a season, plus its spend times its own effects, each the
    national one plus a normal deviation; the noise's variance in proportion to the population."""
    rng = np.random.default_rng(seed)
    people = 1e5 * np.exp(0.6 * rng.standard_normal(geos))
    spend = people[:, None, None] * rng.gamma(4.0, 0.125, (geos, weeks, 2))
    effect = NATIONAL_EFFECT + SPREAD * rng.standard_normal((geos, 2))
    season = 1.0 + 0.2 * np.sin(2.0 * np.pi * np.arange(weeks) / 52.0)
    error = noise * np.sqrt(people)[:, None] * rng.standard_normal((geos, weeks))
    kpi = people[:, None] * season + np.einsum("gtc,gc->gt", spend, effect) + error
    panel = MediaPanel(
        dates=np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(weeks),
        cadence=Cadence.WEEK,
        geos=tuple(Geo(f"g{g}") for g in range(geos)),
        channels=(Channel("search"), Channel("social")),
        controls=(),
        kpi=kpi,
        kpi_unit=Revenue(),
        spend=spend,
        currency="EUR",
        control_values=np.zeros((geos, weeks, 0)),
        filled=np.zeros((geos, weeks), dtype=bool),
        population=people if population else None,
    )
    return panel, effect


@pytest.fixture(scope="module", params=[1.0, 0.98], ids=["fixed variance", "moving variance"])
def small(request: pytest.FixtureRequest) -> tuple[MediaPanel, GeoDlmFit]:
    """A fit with the variance fixed and one with it discounted, whose smoothed degrees of freedom
    differ from step to step."""
    panel, _ = hierarchy(7, geos=3, weeks=40)
    spec = dataclasses.replace(SPEC, variance_discount=request.param)
    return panel, fit_geo_dlm(panel, spec, draws=8)


def test_the_decomposition_is_each_geo_s_smoothed_forecast(
    small: tuple[MediaPanel, GeoDlmFit],
) -> None:
    _, fit = small
    forecast = np.einsum("tgp,tp->gt", fit.spread.fit.design, fit.smoothed.mean)
    np.testing.assert_allclose(
        fit.estimated, forecast, rtol=1e-12, atol=1e-9 * np.abs(forecast).max()
    )


@pytest.mark.parametrize("seed", [1, 2])
def test_each_part_of_the_decomposition_is_its_own_truth(seed: int) -> None:
    """Each part's error is a small share of its truth's spread, which a part read off another
    part's coordinates is not: the parts sum to the forecast however they are labelled."""
    panel, effect = hierarchy(seed, geos=4, weeks=60)
    fit = fit_geo_dlm(panel, SPEC, draws=8)
    assert panel.population is not None
    assert fit.base is not None
    assert fit.seasonality is not None
    people = np.repeat(panel.population[:, None], 60, axis=1)
    parts = {
        "base": (fit.base, people),
        "season": (fit.seasonality, people * 0.2 * np.sin(2.0 * np.pi * np.arange(60) / 52.0)),
        "search": (fit.contributions["search"], panel.spend[:, :, 0] * effect[:, [0]]),
        "social": (fit.contributions["social"], panel.spend[:, :, 1] * effect[:, [1]]),
    }
    for name, (got, truth) in parts.items():
        error = np.sqrt(np.mean(np.square(got - truth))) / truth.std()
        assert error < 0.25, name


def test_each_geo_s_effect_is_recovered_better_than_by_a_model_a_geo_and_covered() -> None:
    covered, pooled_error, alone_error = [], [], []
    for seed in range(4):
        panel, effect = hierarchy(seed, weeks=52, noise=NOISY)
        out = geo_effects(fit_geo_dlm(panel, SPEC, draws=32, seed=seed), level=0.9)
        covered.append((out.lower <= effect) & (effect <= out.upper))
        pooled_error.append(out.mean - effect)
        for g, geo in enumerate(panel.geos):
            one = fit_dlm(panel, geo, SPEC.one_geo())
            alone_error.append(one.smoothed.mean[-1, 1:3] - effect[g])
    assert np.mean(covered) >= 0.8
    pooled, alone = (np.sqrt(np.mean(np.square(e))) for e in (pooled_error, alone_error))
    assert pooled < 0.8 * alone


def test_the_prior_s_guess_of_the_variance_is_the_noise_s_at_the_mean_population() -> None:
    panel, _ = hierarchy(2)
    prior = vague_geo_prior(panel, SPEC)
    assert panel.population is not None
    truth = NOISE**2 * panel.population.mean()
    assert 0.8 * truth < prior.scale < 1.25 * truth
    variances = np.diag(prior.covariance)
    for g in range(len(panel.geos)):
        np.testing.assert_array_equal(variances[[deviation(g, 0), deviation(g, 1)]], variances[:2])


def test_each_prior_standard_deviation_is_ten_times_its_coordinate_s_scale() -> None:
    panel, _ = hierarchy(2)
    prior = vague_geo_prior(panel, SPEC)
    sd = np.sqrt(np.diag(prior.covariance))
    scale = np.std(panel.kpi, axis=1, ddof=1)
    spend = np.sqrt(np.mean(np.square(panel.spend), axis=(0, 1)))
    np.testing.assert_allclose(sd[:2] * spend, 10.0 * np.sqrt(np.mean(scale**2)), rtol=1e-12)
    levels = 2 + REGIONAL * np.arange(len(panel.geos))
    np.testing.assert_allclose(prior.mean[levels], panel.kpi.mean(axis=1), rtol=1e-12)
    for at in (levels, levels + 3, levels + 4):
        np.testing.assert_allclose(sd[at], 10.0 * scale, rtol=1e-12)


def test_an_effect_s_discount_moves_its_national_coefficient_and_never_a_geo_s_deviation() -> None:
    panel, _ = hierarchy(9, geos=3, weeks=40)
    spec = GeoDlmSpec((Effect("search", 0.9), Effect("social", 1.0)), SPEC.trend, SPEC.season)
    paths = fit_geo_dlm(panel, spec, draws=4).smoothed.mean
    assert np.ptp(paths[:, 0]) > 1e-3 * np.abs(paths[:, 0]).mean()
    static = [1, *(deviation(g, c) for g in range(3) for c in range(2))]
    np.testing.assert_allclose(paths[:, static], np.tile(paths[-1, static], (40, 1)), rtol=1e-9)


def test_a_geo_s_effect_interval_is_the_mixture_of_its_coefficient_s_quadratic_form(
    small: tuple[MediaPanel, GeoDlmFit],
) -> None:
    """A filter's last covariance is a Student t's scale; the mixture's variance is matched by a t
    of the same degrees of freedom."""
    _, fit = small
    contrast = np.zeros((3, 2, fit.smoothed.mean.shape[1]))
    for g in range(3):
        for c in range(2):
            contrast[g, c, [c, deviation(g, c)]] = 1.0
    dof = fit.spread.fit.dof[-1]

    def form(filtered: Any) -> tuple[np.ndarray, np.ndarray]:
        m, scale = filtered.mean[-1], filtered.covariance[-1]
        variance = np.einsum("gcp,pq,gcq->gc", contrast, scale, contrast) * dof / (dof - 2.0)
        return contrast @ m, variance

    mean, variance = fit.spread.mixture(form)
    half = stats.t.ppf(0.9, dof) * np.sqrt(variance * (dof - 2.0) / dof)
    out = geo_effects(fit, level=0.8)
    np.testing.assert_allclose(out.mean, mean, rtol=1e-12)
    np.testing.assert_allclose(out.upper - out.mean, half, rtol=1e-9)
    np.testing.assert_allclose(out.mean - out.lower, half, rtol=1e-9)


def test_the_smoothed_states_are_the_moments_of_their_mixture_over_the_draws(
    small: tuple[MediaPanel, GeoDlmFit],
) -> None:
    """Each coordinate summed over two adjacent steps: its mixture's mean and variance, draw by
    draw, are those the mixed means, variances and cross covariances give; the degrees of freedom
    are the draws' mean."""
    _, fit = small

    def pairs(states: Any) -> tuple[np.ndarray, np.ndarray]:
        variance = np.diagonal(states.covariance, axis1=1, axis2=2)
        cross = np.diagonal(states.cross_covariance, axis1=1, axis2=2)
        return states.mean[:-1] + states.mean[1:], variance[:-1] + variance[1:] + 2.0 * cross

    mean, variance = fit.spread.mixture(lambda filtered: pairs(smooth(filtered)))
    got_mean, got_variance = pairs(fit.smoothed)
    np.testing.assert_allclose(got_mean, mean, rtol=1e-12, atol=1e-12 * np.abs(mean).max())
    np.testing.assert_allclose(got_variance, variance, rtol=1e-9)
    dof, _ = fit.spread.mixture(lambda filtered: (smooth(filtered).dof, np.zeros(40)))
    np.testing.assert_allclose(fit.smoothed.dof, dof, rtol=1e-12)


def test_a_geo_s_contribution_over_its_spend_is_its_effect_s_mean(
    small: tuple[MediaPanel, GeoDlmFit],
) -> None:
    panel, fit = small
    out = geo_effects(fit)
    for c, name in enumerate(out.effects):
        ratio = fit.contributions[name] / panel.spend[:, :, c]
        np.testing.assert_allclose(ratio, np.tile(out.mean[:, [c]], (1, 40)), rtol=1e-9)


def test_the_spread_s_draws_follow_the_seed_and_its_intervals_the_level() -> None:
    panel, _ = hierarchy(8, geos=3, weeks=40)
    narrow = fit_geo_dlm(panel, SPEC, draws=4, seed=1, level=0.5).spread
    wide = fit_geo_dlm(panel, SPEC, draws=4, seed=2, level=0.95).spread
    assert not np.array_equal(narrow.draws, wide.draws)
    assert (wide.lower <= narrow.lower).all()
    assert (narrow.upper <= wide.upper).all()
    assert (wide.upper - wide.lower > narrow.upper - narrow.lower).any()


def test_a_filled_step_contributes_nothing_and_leaves_the_fit_finite() -> None:
    panel, _ = hierarchy(6, geos=3, weeks=40)
    kpi, spend, filled = panel.kpi.copy(), panel.spend.copy(), panel.filled.copy()
    controls = np.random.default_rng(6).standard_normal((3, 40, 1))
    kpi[1, 10], spend[1, 10], filled[1, 10], controls[1, 10] = np.nan, 0.0, True, np.nan
    panel = dataclasses.replace(
        panel,
        kpi=kpi,
        spend=spend,
        filled=filled,
        controls=("temperature",),
        control_values=controls,
    )
    spec = GeoDlmSpec((*SPEC.effects, Effect("temperature", 1.0)), SPEC.trend, SPEC.season)
    fit = fit_geo_dlm(panel, spec, draws=4)
    assert np.isfinite(fit.estimated).all()
    assert fit.contributions["temperature"][1, 10] == 0.0


@pytest.mark.parametrize("population", [True, False])
def test_a_geo_s_noise_follows_its_population_when_the_panel_has_one(population: bool) -> None:
    panel, _ = hierarchy(3, geos=3, weeks=40, population=population)
    fit = fit_geo_dlm(panel, SPEC, draws=4)
    weights = np.asarray(fit.spread.fit.model.relative_variance, dtype=np.float64)
    expected = panel.population / panel.population.mean() if panel.population is not None else 1.0
    np.testing.assert_allclose(weights, expected * np.ones(3), rtol=1e-15)


def test_a_channel_bought_nationally_is_recovered_as_any_other() -> None:
    """Its spend in a geo is the geo's share of one national series, from a stream of its own."""
    panel, _ = hierarchy(4, geos=4, weeks=60)
    rng = np.random.default_rng((4, 1))
    tv = allot_national(panel, "tv", rng.gamma(4.0, 1e5, 60))
    effect = 1.5 + 0.2 * rng.standard_normal(4)
    tv = dataclasses.replace(tv, kpi=tv.kpi + tv.spend[:, :, 2] * effect[:, None])
    spec = GeoDlmSpec((*SPEC.effects, Effect("tv", 1.0)), SPEC.trend, SPEC.season)
    out = geo_effects(fit_geo_dlm(tv, spec, draws=8))
    assert out.effects == ("search", "social", "tv")
    half = out.upper[:, 2] - out.mean[:, 2]
    assert (np.abs(out.mean[:, 2] - effect) < 2.0 * half).all()


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: (hierarchy(5, geos=1)[0], SPEC), "needs two geos or more"),
        (
            lambda: (hierarchy(5, geos=2)[0], GeoDlmSpec((Effect("radio", 1.0),), Trend(1, 1.0))),
            r"no channel or control named \['radio'\]",
        ),
    ],
)
def test_what_it_cannot_fit_is_refused(make: Any, match: str) -> None:
    panel, spec = make()
    with pytest.raises(ValueError, match=match):
        fit_geo_dlm(panel, spec)


def test_a_signed_effect_is_refused_over_geos() -> None:
    with pytest.raises(ValueError, match=r"signed effects \['search'\] are not fitted over geos"):
        GeoDlmSpec((Effect("search", 1.0, sign="positive"),), Trend(1, 1.0))
