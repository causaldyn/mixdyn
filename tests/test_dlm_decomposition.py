"""The decomposition's draws against the smoothed decomposition, and its windows by hand.

The world is `seasonal_world`'s history: 156 weeks of `y = 100 + s + 0.5 x + N(0, 1)`, `s` two
harmonics of 52 weeks, so every component's truth is known.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest

from mixdyn import NATIONAL
from mixdyn.models import (
    DlmDecomposition,
    DlmFit,
    DlmSpec,
    Effect,
    Season,
    Trend,
    decompose_dlm,
    fit_dlm,
)
from seasonal_world import CONTRIBUTION, HISTORY, SEASON, history

SPEC = DlmSpec((Effect("x", 0.95),), trend=Trend(1, 0.95), season=Season(52.0, 2, 0.98))
DRAWS = 4000


@functools.cache
def fit(spec: DlmSpec = SPEC) -> DlmFit:
    return fit_dlm(history(), NATIONAL, spec)


@functools.cache
def decomposition(spec: DlmSpec = SPEC) -> DlmDecomposition:
    return decompose_dlm(fit(spec), draws=DRAWS, seed=20261001)


def within_monte_carlo(draws: np.ndarray, expected: np.ndarray) -> float:
    """The largest standardised gap between the draws' mean and `expected`, period by period."""
    error = draws.std(axis=0) / np.sqrt(draws.shape[0])
    return float(np.max(np.abs(draws.mean(axis=0) - expected) / error))


def test_the_draws_average_to_the_smoothed_decomposition() -> None:
    smoothed, draws = fit(), decomposition().draws
    assert smoothed.base is not None
    assert draws.base is not None
    assert smoothed.seasonality is not None
    assert draws.seasonality is not None
    assert draws.base.shape == (DRAWS, HISTORY)
    assert within_monte_carlo(draws.base, smoothed.base) < 4.5
    assert within_monte_carlo(draws.seasonality, smoothed.seasonality) < 4.5
    assert within_monte_carlo(draws.contributions["x"], smoothed.contributions["x"]) < 4.5
    assert within_monte_carlo(draws.total(), smoothed.estimated) < 4.5


def test_a_window_totals_each_component_over_its_periods_both_ends_included() -> None:
    whole = decomposition()
    dates = whole.dates
    totals = whole.window(dates[10], dates[19])
    assert totals.base is not None
    assert whole.draws.base is not None
    np.testing.assert_allclose(totals.base, whole.draws.base[:, 10:20].sum(axis=1), rtol=1e-12)
    expected = whole.draws.contributions["x"][:, 10:20].sum(axis=1)
    np.testing.assert_allclose(totals.contributions["x"], expected, rtol=1e-12)
    np.testing.assert_allclose(
        totals.total(), whole.draws.total()[:, 10:20].sum(axis=1), rtol=1e-12
    )


def test_a_change_is_the_difference_of_the_windows_means_per_period() -> None:
    whole = decomposition()
    dates = whole.dates
    moved = whole.change((dates[0], dates[25]), (dates[26], dates[38]))
    x = whole.draws.contributions["x"]
    expected = x[:, 26:39].mean(axis=1) - x[:, :26].mean(axis=1)
    scale = float(np.abs(x).max())
    np.testing.assert_allclose(moved.contributions["x"], expected, rtol=0.0, atol=1e-12 * scale)


def test_the_spend_s_and_the_season_s_window_intervals_cover_their_truths() -> None:
    """Over the 156 weeks, x contributes `0.5 Σ x` and the season, three whole cycles, nothing."""
    whole = decomposition()
    totals = whole.window(whole.dates[0], whole.dates[-1])
    spend = float(np.sum(CONTRIBUTION[:HISTORY]))
    lower, upper = np.quantile(totals.contributions["x"], [0.05, 0.95])
    assert lower <= spend <= upper
    assert totals.seasonality is not None
    lower, upper = np.quantile(totals.seasonality, [0.05, 0.95])
    truth = float(np.sum(SEASON[:HISTORY]))
    assert lower <= truth <= upper


def test_a_component_the_spec_lacks_has_no_draws() -> None:
    effects_only = DlmSpec((Effect("x", 0.95),))
    draws = decomposition(effects_only).draws
    assert draws.base is None
    assert draws.seasonality is None
    np.testing.assert_allclose(draws.total(), draws.contributions["x"])


def test_a_window_outside_the_fit_is_refused_with_its_span() -> None:
    whole = decomposition()
    with pytest.raises(ValueError, match="no period of the fit lies from 2030-01-01"):
        whole.window(np.datetime64("2030-01-01"), np.datetime64("2030-12-31"))
