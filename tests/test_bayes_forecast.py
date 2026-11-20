"""The static Bayesian model's forecast: its mean is the fitted mean's formula, written again in
numpy, run on past the window at each draw `BayesFit.channel_draws` reads; its KPI is drawn about
that mean by the model's noise, normal, Student-t or log-normal; and what it cannot forecast is
refused."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from types import MappingProxyType
from typing import Any

import jax
import numpy as np
import pytest
from scipy import stats

from mixdyn import MediaPanel
from mixdyn.models.bayes_mmm import (
    BayesFit,
    LogNormal,
    Normal,
    StudentT,
    TanhCurve,
    fit_bayes,
    forecast_bayes,
)
from test_bayes_mmm import NATIONAL, stan_world

START, HORIZON, DRAWS = 24, 13, 200
"""The window runs from the panel's 25th week to the 14th before its end, and the forecast over
the 13 after it; two chains of 100 draws."""


@pytest.fixture(scope="module", params=["normal", "tanh"])
def fitted(request: pytest.FixtureRequest) -> tuple[MediaPanel, BayesFit]:
    panel, tests, estimates, spec = stan_world(request.param)
    return panel, fit_bayes(
        panel,
        NATIONAL,
        spec,
        seed=3,
        window=(panel.dates[START], panel.dates[-HORIZON - 1]),
        tests=tests,
        estimates=estimates,
        chains=2,
        warmup=150,
        draws=100,
    )


def quarter(panel: MediaPanel, fit: BayesFit) -> dict[str, np.ndarray]:
    """The panel's own spend and controls over the 13 weeks after the window."""
    channels = [str(c) for c in panel.channels]
    plan = {m.channel: panel.spend[0, -HORIZON:, channels.index(m.channel)] for m in fit.spec.media}
    controls = list(panel.controls)
    return plan | {
        c.column: panel.control_values[0, -HORIZON:, controls.index(c.column)]
        for c in fit.spec.controls
    }


def formula(panel: MediaPanel, fit: BayesFit, keep: np.ndarray) -> np.ndarray:
    """The fitted mean at each draw `keep` picks of the chains' draws in chain order, over the
    window and the 13 weeks after it, `(draws, weeks)`: the intercept; the line bent at knots
    spaced evenly inside the window, in units of its length, and the season's harmonics, both
    counted from the window's first week; each control's coefficient times its value; and each
    channel's coefficient times its curve of its adstock, its kernel's weights `r^l` over the fit's
    length, normalised, read over the spend from the panel's first week."""
    at = {name: every.reshape(-1, *every.shape[2:])[keep] for name, every in fit.draws.items()}
    assert fit.spec.trend is not None
    assert fit.spec.season is not None
    window = fit.dates.size
    t = np.arange(window + HORIZON, dtype=np.float64)
    knots = window * np.arange(1, fit.spec.trend.knots + 1) / (fit.spec.trend.knots + 1)
    trend = np.column_stack([t, *(np.maximum(0.0, t - k) for k in knots)]) / window
    angle = 2.0 * math.pi * t / fit.spec.season.period
    season = [
        f(h * angle) for h in range(1, fit.spec.season.harmonics + 1) for f in (np.cos, np.sin)
    ]
    controls = panel.control_values[0, START:][
        :, [list(panel.controls).index(c.column) for c in fit.spec.controls]
    ]
    mean = (
        at["intercept"][:, None]
        + at["calendar"] @ np.column_stack([trend, *season]).T
        + at["control"] @ controls.T
    )
    channels = [str(c) for c in panel.channels]
    for j, m in enumerate(fit.spec.media):
        spend = panel.spend[0, :, channels.index(m.channel)]
        for i in range(keep.size):
            weights = at["retention"][i, j] ** np.arange(fit.length)
            adstock = np.convolve(spend, weights / weights.sum())[: spend.size][START:]
            half = at["half_saturation"][i, j]
            if isinstance(fit.spec.saturation, TanhCurve):
                curve = np.tanh(math.atanh(0.5) * adstock / half)
            else:
                slope = at["slope"][i, j]
                curve = adstock**slope / (adstock**slope + half**slope)
            mean[i] += at["coefficient"][i, j] * curve
    return mean


def test_the_mean_is_the_fitted_mean_run_on_past_the_window(
    fitted: tuple[MediaPanel, BayesFit],
) -> None:
    """At the window's weeks the formula, averaged over every draw, is the fit's posterior mean of
    its fitted mean, so the formula is the model's; past them the forecast's mean is the formula at
    each of 37 draws evenly spaced over the chains' from the first to the last, to 1e-12: the
    trend run on from its last bend, the season on its period, and the spend before the window and
    over it carried into the quarter's."""
    panel, fit = fitted
    window = fit.dates.size
    assert window == panel.dates.size - START - HORIZON
    every = formula(panel, fit, np.arange(DRAWS))
    np.testing.assert_allclose(every[:, :window].mean(axis=0), fit.estimated, rtol=1e-12)
    ahead = forecast_bayes(fit, HORIZON, quarter(panel, fit), 37, seed=5)
    keep = np.linspace(0, DRAWS - 1, 37).round().astype(np.intp)
    np.testing.assert_allclose(ahead.mean, formula(panel, fit, keep)[:, window:], rtol=1e-12)
    np.testing.assert_array_equal(ahead.dates, panel.dates[-HORIZON:])


def test_a_fit_holds_the_spend_from_the_panel_s_first_week_to_the_window_s_last(
    fitted: tuple[MediaPanel, BayesFit],
) -> None:
    panel, fit = fitted
    channels = [str(c) for c in panel.channels]
    assert list(fit.spend) == [m.channel for m in fit.spec.media]
    for name, spend in fit.spend.items():
        np.testing.assert_array_equal(spend, panel.spend[0, :-HORIZON, channels.index(name)])
    assert fit.cadence == panel.cadence


def test_a_model_of_its_channels_alone_forecasts_its_intercept_and_its_channels() -> None:
    """Each draw's channels as `BayesFit.channel_draws` reads them, run over the spend and the
    plan's: a model with no trend, season or control has no other term to run on."""
    panel, _, _, spec = stan_world("normal")
    alone = dataclasses.replace(spec, controls=(), trend=None, season=None)
    fit = fit_bayes(panel, NATIONAL, alone, seed=4, chains=2, warmup=50, draws=20)
    plan = {c: np.full(HORIZON, 30.0 * (j + 1)) for j, c in enumerate(fit.spend)}
    ahead = forecast_bayes(fit, HORIZON, plan, 40, seed=2)
    intercept = fit.draws["intercept"].reshape(-1)
    expected = [
        intercept[i]
        + sum(
            np.asarray(channel(np.concatenate([fit.spend[name], plan[name]])))[-HORIZON:]
            for name, channel in channels.items()
        )
        for i, channels in zip(range(40), fit.channel_draws(40), strict=True)
    ]
    np.testing.assert_allclose(ahead.mean, np.array(expected), rtol=1e-12)


def _noisy(fit: BayesFit, likelihood: str) -> BayesFit:
    """The fit with its noise `likelihood`'s, each draw's scale spread from 1/20 to 4 times its
    median, a log-normal's log scale from 0.02 to 0.4, and a Student-t's degrees of freedom 1.5 and
    60 in turn, so that a draw's KPI read at another draw's noise is seen."""
    rng = np.random.default_rng(17)
    shape = fit.draws["sigma"].shape
    if likelihood == "lognormal":
        scale = rng.uniform(0.02, 0.4, shape)
    else:
        scale = float(np.median(fit.draws["sigma"])) * rng.uniform(0.05, 4.0, shape)
    draws = dict(fit.draws) | {"sigma": scale}
    if likelihood == "student":
        draws["nu"] = np.where(np.arange(scale.size).reshape(shape) % 2 == 0, 1.5, 60.0)
    noise = {"normal": Normal(), "student": StudentT(), "lognormal": LogNormal()}[likelihood]
    return dataclasses.replace(
        fit,
        spec=dataclasses.replace(fit.spec, likelihood=noise),
        draws=MappingProxyType(draws),
    )


@pytest.mark.parametrize("likelihood", ["normal", "student", "lognormal"])
def test_the_kpi_is_drawn_about_each_draw_s_mean_at_its_noise(
    fitted: tuple[MediaPanel, BayesFit], likelihood: str
) -> None:
    """Each KPI drawn, standardised at its own draw's noise, is the noise's standard law by a
    Kolmogorov-Smirnov test over the 200 draws' 13 weeks at 0.001: normal; Student-t at the draw's
    degrees of freedom, read over the draws at each value apart, since a law drawn at one value
    between the two is too light-tailed for one group and too heavy for the other, and over both
    together the two errors cancel; or the log's normal, whose mean is the draw's mean. The noise
    moves the KPI alone."""
    panel, fit = fitted
    noisy = _noisy(fit, likelihood)
    ahead = forecast_bayes(noisy, HORIZON, quarter(panel, fit), DRAWS, seed=11)
    np.testing.assert_array_equal(
        ahead.mean, forecast_bayes(fit, HORIZON, quarter(panel, fit), DRAWS, seed=12).mean
    )
    scale = noisy.draws["sigma"].reshape(-1)[:, None]
    if likelihood == "lognormal":
        z = (np.log(ahead.kpi) - np.log(ahead.mean) + scale**2 / 2.0) / scale
        p = stats.kstest(z.ravel(), "norm").pvalue
    elif likelihood == "student":
        nu = noisy.draws["nu"].reshape(-1)
        u = stats.t.cdf((ahead.kpi - ahead.mean) / scale, nu[:, None])
        p = min(stats.kstest(u[nu == v].ravel(), "uniform").pvalue for v in (1.5, 60.0))
    else:
        p = stats.kstest(((ahead.kpi - ahead.mean) / scale).ravel(), "norm").pvalue
    assert p > 0.001


def _refused(change: Callable[[dict[str, Any]], None]) -> Callable[[MediaPanel, BayesFit], Any]:
    def call(panel: MediaPanel, fit: BayesFit) -> Any:
        given: dict[str, Any] = {
            "horizon": HORIZON,
            "plan": quarter(panel, fit),
            "draws": 10,
            "seed": 1,
        }
        change(given)
        return forecast_bayes(fit, **given)

    return call


def _plan(edit: Callable[[dict[str, np.ndarray]], object]) -> Callable[[dict[str, Any]], None]:
    def change(given: dict[str, Any]) -> None:
        edit(given["plan"])

    return change


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda g: g.update(horizon=0), "horizon must be a positive integer, got 0"),
        (lambda g: g.update(horizon=True), "horizon must be a positive integer, got True"),
        (lambda g: g.update(draws=0), "draws must be a positive integer, got 0"),
        (lambda g: g.update(draws=True), "draws must be a positive integer, got True"),
        (lambda g: g.update(draws=201), "count must be from 1 to the fit's 200 draws, got 201"),
        (_plan(lambda p: p.pop("tv")), r"it lacks \['tv'\] and adds \[\]"),
        (_plan(lambda p: p.update(radio=np.zeros(HORIZON))), r"lacks \[\] and adds \['radio'\]"),
        (
            _plan(lambda p: p.update(search=np.zeros(HORIZON - 1))),
            r"the plan's 'search' has shape \(12,\), not \(13,\)",
        ),
        (
            _plan(lambda p: p.update(price=np.where(np.arange(HORIZON) == 3, np.nan, 1.0))),
            r"the plan's 'price' is not finite at steps \[3\]",
        ),
        (
            _plan(lambda p: p.update(tv=np.where(np.arange(HORIZON) == 2, -1.0, 5.0))),
            r"the plan's 'tv' spends below 0 at steps \[2\]",
        ),
    ],
)
def test_what_cannot_be_forecast_is_refused(
    fitted: tuple[MediaPanel, BayesFit], change: Callable[[dict[str, Any]], None], match: str
) -> None:
    panel, fit = fitted
    with pytest.raises(ValueError, match=match):
        _refused(change)(panel, fit)


def test_a_control_below_zero_is_no_spend_and_is_forecast(
    fitted: tuple[MediaPanel, BayesFit],
) -> None:
    panel, fit = fitted
    plan = quarter(panel, fit) | {"price": np.full(HORIZON, -2.0)}
    assert np.isfinite(forecast_bayes(fit, HORIZON, plan, 10, seed=1).kpi).all()


def test_a_log_normal_mean_not_above_zero_is_refused(
    fitted: tuple[MediaPanel, BayesFit],
) -> None:
    """Where a draw's mean is not above 0 the model has no KPI: an intercept far below 0 at every
    draw takes every week of every draw there."""
    panel, fit = fitted
    noisy = _noisy(fit, "lognormal")
    sunk = dataclasses.replace(
        noisy,
        draws=MappingProxyType(
            dict(noisy.draws) | {"intercept": np.full(noisy.draws["intercept"].shape, -1e9)}
        ),
    )
    with pytest.raises(
        ValueError,
        match=(
            r"under log-normal noise the mean is not above 0 at steps \[0, 1, 2, 3, 4, 5, 6, 7, 8,"
            r" 9, 10, 11, 12\] in 10 of the 10 draws, where the model has no KPI"
        ),
    ):
        forecast_bayes(sunk, HORIZON, quarter(panel, fit), 10, seed=1)


def test_single_precision_is_refused(fitted: tuple[MediaPanel, BayesFit]) -> None:
    panel, fit = fitted
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(ValueError, match="double precision"):
            forecast_bayes(fit, HORIZON, quarter(panel, fit), 10, seed=1)
    finally:
        jax.config.update("jax_enable_x64", True)
