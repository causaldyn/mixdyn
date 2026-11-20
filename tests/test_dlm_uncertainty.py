"""The engine's variance, prior and intervals where the level drifts.

Five worlds of 156 weeks of `y = L + 0.5 x + N(0, 2^2)`, `L` a random walk from 100 of `N(0, 1)`
steps and `x = max(0, N(50, 15))`. The engine learns `V`, starts from a vague prior, and gives
every contribution an interval.
"""

from __future__ import annotations

import functools

import numpy as np

from mixdyn import NATIONAL, Cadence, Count, MediaPanel, read_panel
from mixdyn.models import DiscountGrid, DlmFit, DlmSpec, Effect, Trend, choose_dlm, fit_dlm

DATES = np.datetime64("2023-01-02") + np.timedelta64(7, "D") * np.arange(156)
SEEDS = range(20261003, 20261008)
EFFECT, NOISE = 0.5, 2.0
SPEC = DlmSpec((Effect("x", 0.95),), trend=Trend(1, 0.95))
"""A level and x, each at 0.95."""


@functools.cache
def world(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The world's KPI and x."""
    rng = np.random.default_rng(seed)
    x = np.maximum(0.0, rng.normal(50.0, 15.0, DATES.size))
    level = 100.0 + np.cumsum(rng.normal(0.0, 1.0, DATES.size))
    return level + EFFECT * x + rng.normal(0.0, NOISE, DATES.size), x


def panel(seed: int) -> MediaPanel:
    kpi, x = world(seed)
    return read_panel(
        {"date": DATES, "kpi": kpi, "x": x},
        date="date",
        kpi="kpi",
        spend={"x": "x"},
        cadence=Cadence.WEEK,
        kpi_unit=Count("units"),
        currency="USD",
    )


@functools.cache
def engine(seed: int) -> DlmFit:
    return fit_dlm(panel(seed), NATIONAL, SPEC)


def test_a_forecast_before_any_data_is_less_sure_than_one_after_all_of_it() -> None:
    """The first one-step variance is over 1300 times the last. A prior fitted to the whole sample
    would make the first forecast as sure as the last."""
    for seed in SEEDS:
        ours = engine(seed).filtered.one_step_scale
        assert ours[0] > 100.0 * ours[-1]


def test_the_variance_is_the_noise_s_within_a_factor_of_two() -> None:
    """At these discounts the engine learns `V` at 1.27 to 1.35 times the noise's variance of 4,
    where a regression's residual variance on the whole sample would carry the drift as well."""
    for seed in SEEDS:
        assert engine(seed).filtered.scale[-1] < 2.0 * NOISE**2


def test_a_quarter_s_contribution_comes_with_an_interval_that_covers_its_truth() -> None:
    """Twelve quarters a world, with the discounts chosen by likelihood and the intervals by
    projection: the 90 % intervals of x's contribution cover 56 of the 60 truths. A contribution
    reported as a point is an interval of no width, which covers none."""
    grid = DiscountGrid(trend=(0.7, 0.8, 0.9, 0.95), effects=(0.95, 0.99, 1.0))
    ours = quarters = 0
    for seed in SEEDS:
        choice = choose_dlm(panel(seed), NATIONAL, SPEC, grid)
        x = world(seed)[1]
        for q in range(DATES.size // 13):
            weeks = slice(13 * q, 13 * q + 13)
            truth = EFFECT * x[weeks].sum()
            start, end = DATES[weeks][[0, -1]]
            bounds = choice.bounds(lambda d, s=start, e=end: d.window(s, e), draws=2000, seed=1)
            ours += bounds.contributions["x"].lower <= truth <= bounds.contributions["x"].upper
            quarters += 1
    assert quarters == 60
    assert ours >= 0.85 * quarters
