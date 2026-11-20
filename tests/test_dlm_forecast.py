"""The engine's forecast of a world with an exact 52-week season (`seasonal_world`), whose
season, contributions and mean are known over the weeks forecast."""

from __future__ import annotations

import functools

import numpy as np
import pytest
from scipy import stats

from mixdyn import NATIONAL
from mixdyn.models import (
    DlmForecast,
    DlmSpec,
    Effect,
    Season,
    Trend,
    fit_dlm,
    forecast_dlm,
    forecast_draws,
)
from seasonal_world import CONTRIBUTION, DATES, HISTORY, KPI, MEAN, NOISE, SEASON, X, history

SPEC = DlmSpec((Effect("x", 0.95),), trend=Trend(1, 0.95), season=Season(52.0, 2, 0.98))
"""A level, `x` and two harmonics of 52 weeks, the level and `x` at 0.95."""


@functools.cache
def engine() -> DlmForecast:
    plan = {"x": X[HISTORY:]}
    return forecast_dlm(fit_dlm(history(), NATIONAL, SPEC), plan["x"].size, plan)


def rmse(forecast: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((forecast - truth) ** 2)))


def coverage(lower: np.ndarray, upper: np.ndarray) -> float:
    y = KPI[HISTORY:]
    return float(np.mean((lower <= y) & (y <= upper)))


def test_the_forecast_carries_the_season() -> None:
    """The seasonal forecast misses the season by 0.29, against its SD of 7.62, which a forecast
    that dropped it would miss by, and noise of 1."""
    ours = engine().seasonality
    assert ours is not None
    assert rmse(ours, SEASON[HISTORY:]) < NOISE


def test_the_forecast_carries_the_planned_spend() -> None:
    """Given the plan, the forecast's contribution misses the true one by 0.41."""
    assert rmse(engine().contributions["x"], CONTRIBUTION[HISTORY:]) < NOISE


def test_the_forecast_misses_the_kpi_mean_by_less_than_the_noise() -> None:
    """By 0.43."""
    assert rmse(engine().mean, MEAN[HISTORY:]) < NOISE


def test_the_80_percent_interval_covers_at_least_80_percent() -> None:
    """It covers 0.94 of the weeks: discounting at 0.95 lets the state drift, and this world's
    state does not."""
    assert coverage(*engine().interval(0.8)) >= 0.8


def test_the_forecast_variance_grows_linearly_with_the_horizon() -> None:
    """A level alone: each step ahead adds the first step's evolution variance. Compounding the
    discount instead would grow the variance as `discount ** -horizon`."""
    level = DlmSpec((), trend=Trend(1, 0.95))
    ahead = forecast_dlm(fit_dlm(history(), NATIONAL, level), 52, {})
    steps = np.diff(ahead.scale)
    assert np.abs(np.diff(steps)).max() <= 1e-12 * steps[0]
    assert steps[0] > 0.0


def test_the_interval_holds_its_level_of_each_periods_student_t() -> None:
    ahead = engine()
    lower, upper = ahead.interval(0.8)
    scale = np.sqrt(ahead.scale)
    held = stats.t.cdf((upper - ahead.mean) / scale, ahead.dof) - stats.t.cdf(
        (lower - ahead.mean) / scale, ahead.dof
    )
    assert np.abs(held - 0.8).max() <= 1e-12


def test_the_forecast_splits_into_its_parts() -> None:
    ahead = engine()
    assert ahead.base is not None
    assert ahead.seasonality is not None
    parts = ahead.base + ahead.seasonality + ahead.contributions["x"]
    assert np.abs(parts - ahead.mean).max() <= 1e-12 * np.abs(ahead.mean).max()


def test_the_forecast_dates_continue_the_calendar() -> None:
    assert np.array_equal(engine().dates, DATES[HISTORY:])


def test_a_plan_must_name_every_effect_and_nothing_else() -> None:
    fit = fit_dlm(history(), NATIONAL, SPEC)
    with pytest.raises(ValueError, match=r"lacks \['x'\] and adds \['spend_tv'\]"):
        forecast_dlm(fit, 2, {"spend_tv": [1.0, 2.0]})


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([1.0, 2.0, 3.0], r"'x' has shape \(3,\), not \(2,\)"),
        ([1.0, float("nan")], r"'x' is not finite at steps \[1\]"),
    ],
)
def test_a_planned_column_must_be_one_finite_value_a_period(
    values: list[float], message: str
) -> None:
    fit = fit_dlm(history(), NATIONAL, SPEC)
    with pytest.raises(ValueError, match=message):
        forecast_dlm(fit, 2, {"x": values})


@pytest.mark.parametrize("horizon", [0, -1, True])
def test_a_horizon_below_one_is_refused(horizon: int) -> None:
    fit = fit_dlm(history(), NATIONAL, SPEC)
    with pytest.raises(ValueError, match="horizon must be a positive integer"):
        forecast_dlm(fit, horizon, {"x": []})


@pytest.mark.parametrize("level", [0.0, 1.0, 1.5])
def test_an_interval_outside_zero_and_one_is_refused(level: float) -> None:
    with pytest.raises(ValueError, match=r"level must lie in \(0, 1\)"):
        engine().interval(level)


def test_draws_of_a_fit_without_signs_follow_its_student_t() -> None:
    """Each of 13 weeks' 20 000 draws against `forecast_dlm`'s Student t, a Kolmogorov-Smirnov test
    at 0.001 over the weeks: a draw of the filter's normal-gamma posterior at the last week,
    carried through the evolution `forecast_dlm` holds, is a draw of that Student t."""
    fit = fit_dlm(history(), NATIONAL, SPEC)
    plan = {"x": X[HISTORY : HISTORY + 13]}
    exact = forecast_dlm(fit, 13, plan)
    drawn = forecast_draws(fit, 13, plan, 20_000, seed=11)
    assert drawn.kpi.shape == (20_000, 13)
    assert drawn.acceptance == 1.0
    assert np.array_equal(drawn.dates, exact.dates)
    p = [
        stats.kstest(
            drawn.kpi[:, k], "t", args=(exact.dof[k], exact.mean[k], np.sqrt(exact.scale[k]))
        ).pvalue
        for k in range(13)
    ]
    assert min(p) > 0.001 / 13


@pytest.mark.parametrize("draws", [0, -1, True])
def test_a_forecast_drawn_below_one_draw_is_refused(draws: int) -> None:
    fit = fit_dlm(history(), NATIONAL, SPEC)
    with pytest.raises(ValueError, match="draws must be a positive integer"):
        forecast_draws(fit, 2, {"x": [1.0, 2.0]}, draws, seed=0)


def test_a_forecast_drawn_from_a_variance_that_moves_is_refused() -> None:
    """Below a variance discount of 1 the variance moves over the horizon, where a draw holds the
    last week's."""
    moving = DlmSpec(SPEC.effects, trend=SPEC.trend, season=SPEC.season, variance_discount=0.99)
    fit = fit_dlm(history(), NATIONAL, moving)
    with pytest.raises(ValueError, match="needs a variance discount of 1"):
        forecast_draws(fit, 2, {"x": [1.0, 2.0]}, 10, seed=0)
