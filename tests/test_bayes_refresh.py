"""The static model's holdout and its refresh: a blocked holdout by default, since an interleaved
one leaks the autocorrelated noise its neighbours share; a refresh that is the full refit at weight
0 and stays near it at the default; and refits on four more weeks at a time whose returns stay
inside the posterior's 80 % before them on a world that holds still."""

from __future__ import annotations

import dataclasses
import inspect
import itertools
import math
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np
import pytest

from mixdyn import Cadence, Channel, Geo, MediaPanel, Revenue
from mixdyn.models.bayes_mmm import (
    WEIGHT,
    BayesMedia,
    BayesSpec,
    HalfNormalRoi,
    RoiPrior,
    TanhCurve,
    _crps,
    _window_means,
    fit_bayes,
    holdout_bayes,
    refresh_bayes,
    returns_on,
    tempered_spec,
)
from mixdyn.models.ridge_mmm import LinearTrend

NATIONAL = Geo("national")
CHAINS, WARMUP, DRAWS = 2, 200, 200
"""Every fit's chains, and the draws of each after its warm-up."""


def world(seed: int, *, weeks: int = 104, rho: float = 0.0, sd: float = 6.0) -> MediaPanel:
    """Two channels' spend, `max(0, N(50, 15))` a week, each carried over at retention 0.5 and
    saturated by `tanh` at returns 30 and 18 a week at most, on a level of 200, and noise of sd
    `sd` that is AR(1) at `rho` from its stationary law."""
    rng = np.random.default_rng(seed)
    spend = np.maximum(0.0, rng.normal(50.0, 15.0, (weeks, 2)))
    adstock = np.zeros_like(spend)
    for t in range(weeks):
        adstock[t] = spend[t] + (0.5 * adstock[t - 1] if t else 0.0)
    media = 30.0 * np.tanh(adstock / 80.0) @ np.array([1.0, 0.6])
    shocks = rng.normal(0.0, sd * math.sqrt(1.0 - rho**2), weeks)
    noise = np.empty(weeks)
    noise[0] = rng.normal(0.0, sd)
    for t in range(1, weeks):
        noise[t] = rho * noise[t - 1] + shocks[t]
    return MediaPanel(
        dates=np.datetime64("2023-01-02") + np.timedelta64(7, "D") * np.arange(weeks),
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=(Channel("a"), Channel("b")),
        controls=(),
        kpi=(200.0 + media + noise)[None, :],
        kpi_unit=Revenue(),
        spend=spend[None],
        currency="EUR",
        control_values=np.zeros((1, weeks, 0)),
        filled=np.zeros((1, weeks), dtype=bool),
    )


def spec(knots: int = 0, *, counts: Literal["window", "carried"] = "window") -> BayesSpec:
    media = tuple(BayesMedia(c, RoiPrior(counts=counts)) for c in ("a", "b"))
    return BayesSpec(media, trend=LinearTrend(knots), saturation=TanhCurve())


def test_an_interleaved_holdout_reads_the_noise_its_neighbours_share() -> None:
    """On noise AR(1) at 0.95, an interleaved holdout of 13 of 104 weeks scores below half the
    blocked one's CRPS, and below 0.6 of its own on the same world with noise of the same sd and no
    autocorrelation: its neighbours hand a trend bent at 12 knots the slow swings of the noise
    that no forecast past the window reads. Measured on worlds 1 to 4 at 2 chains of 200 draws,
    the first ratio was 0.07 to 0.28 and the second 0.19 to 0.42 (ADR 0079); world 1 reads 0.07
    and 0.19, so a seed's sampler noise would not move either across its bound."""
    model = spec(12)
    autocorrelated, independent = world(1, rho=0.95), world(1)
    scored = {
        (name, kind): holdout_bayes(
            panel,
            NATIONAL,
            model,
            periods=13,
            seed=1,
            kind=kind,
            chains=CHAINS,
            warmup=WARMUP,
            draws=DRAWS,
        ).crps
        for name, panel in (("ar", autocorrelated), ("iid", independent))
        for kind in ("blocked", "interleaved")
        if (name, kind) != ("iid", "blocked")
    }
    assert scored["ar", "interleaved"] < 0.5 * scored["ar", "blocked"]
    assert scored["ar", "interleaved"] < 0.6 * scored["iid", "interleaved"]


def test_a_holdout_is_blocked_by_default_and_keeps_its_periods_out_of_the_fit() -> None:
    """The default holds out the window's last periods; an interleaved one a seeded draw; either's
    draws are one a posterior draw, and its score is the CRPS of them over the KPI's mean."""
    assert inspect.signature(holdout_bayes).parameters["kind"].default == "blocked"
    panel = world(2)
    blocked = holdout_bayes(
        panel, NATIONAL, spec(), periods=5, seed=3, chains=CHAINS, warmup=WARMUP, draws=DRAWS
    )
    assert blocked.kind == "blocked"
    np.testing.assert_array_equal(blocked.dates, panel.dates[-5:])
    np.testing.assert_array_equal(blocked.kpi, panel.kpi[0, -5:])
    assert blocked.draws.shape == (400, 5)
    mean = float(np.mean(panel.kpi))
    assert blocked.crps == pytest.approx(_crps(blocked.draws, blocked.kpi) / mean, rel=1e-12)
    picked = holdout_bayes(
        panel,
        NATIONAL,
        spec(),
        periods=5,
        seed=3,
        kind="interleaved",
        chains=CHAINS,
        warmup=WARMUP,
        draws=DRAWS,
    )
    again = holdout_bayes(
        panel,
        NATIONAL,
        spec(),
        periods=5,
        seed=3,
        kind="interleaved",
        chains=CHAINS,
        warmup=WARMUP,
        draws=DRAWS,
    )
    np.testing.assert_array_equal(picked.dates, again.dates)
    assert picked.dates.size == 5
    assert not np.array_equal(picked.dates, panel.dates[-5:])


def test_a_holdout_refuses_what_it_cannot_score() -> None:
    panel = world(2)
    for periods in (0, 104, 2.0, True):
        with pytest.raises(ValueError, match="observed periods"):
            holdout_bayes(panel, NATIONAL, spec(), periods=periods, seed=1)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="blocked"):
        holdout_bayes(panel, NATIONAL, spec(), periods=5, seed=1, kind="random")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="no geo"):
        holdout_bayes(panel, Geo("north"), spec(), periods=5, seed=1)


def test_crps_is_its_pairwise_definition() -> None:
    """`E|X - y| - E|X - X'| / 2` over every pair of draws, at each period, averaged."""
    rng = np.random.default_rng(4)
    draws, target = rng.gamma(2.0, 3.0, (60, 7)), rng.normal(6.0, 2.0, 7)
    pairs = np.abs(draws[:, None, :] - draws[None, :, :]).mean(axis=(0, 1))
    expected = np.mean(np.abs(draws - target).mean(axis=0) - pairs / 2.0)
    assert _crps(draws, target) == pytest.approx(expected, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("counts", ["window", "carried"])
def test_a_fit_s_returns_and_means_are_its_own_draws(counts: Literal["window", "carried"]) -> None:
    """Read on the fit's own window, `returns_on` is its `roi` draws as its prior counts them, and
    the window's means averaged over the draws are its `estimated`, each to 1e-12."""
    panel = world(5, weeks=80)
    fit = fit_bayes(
        panel, NATIONAL, spec(1, counts=counts), seed=2, chains=CHAINS, warmup=WARMUP, draws=DRAWS
    )
    np.testing.assert_allclose(
        returns_on(fit, panel, NATIONAL, fit.dates),
        fit.draws["roi"].reshape(-1, 2),
        rtol=1e-12,
    )
    means = _window_means(fit, np.zeros((fit.dates.size, 0)))
    np.testing.assert_allclose(means.mean(axis=0), fit.estimated, rtol=1e-12)
    with pytest.raises(ValueError, match="consecutive"):
        returns_on(fit, panel, NATIONAL, fit.dates[::2])


def test_a_tempered_prior_is_the_power_prior_s_normal() -> None:
    """Each channel's prior times the previous periods' evidence on its log return to the power
    `a`, the closed form validation/refresh_prior.mac proves; a channel the data did not narrow
    keeps its prior; at 0 the spec is the spec."""
    rng = np.random.default_rng(6)
    log_a, log_b = rng.normal(0.5, 0.3, (2, 500)), rng.normal(0.1, 1.2, (2, 500))
    previous = SimpleNamespace(spec=spec(), draws={"roi": np.exp(np.stack([log_a, log_b], -1))})
    assert tempered_spec(spec(), previous, 0.0) == spec()  # ty: ignore[invalid-argument-type]
    base = dataclasses.replace(
        spec(), media=(BayesMedia("a", RoiPrior(0.0, 2.0)), BayesMedia("b", RoiPrior(1.0, 0.5)))
    )
    out = tempered_spec(base, previous, 0.25)  # ty: ignore[invalid-argument-type]
    m, q = float(np.mean(log_a)), 1.0 / float(np.var(log_a, ddof=1))
    pp = 1.0 / 0.9**2
    precision = 1.0 / 2.0**2 + 0.25 * (q - pp)
    centre = (0.0 / 2.0**2 + 0.25 * (q * m - pp * 0.2)) / precision
    tempered = out.media[0].roi
    assert isinstance(tempered, RoiPrior)
    assert tempered.mu == pytest.approx(centre, rel=1e-12, abs=0.0)
    assert tempered.sigma == pytest.approx(1.0 / math.sqrt(precision), rel=1e-12, abs=0.0)
    assert tempered.counts == "window"
    assert out.media[1] == base.media[1]


def test_a_tempered_prior_refuses_what_it_cannot_temper() -> None:
    previous = SimpleNamespace(spec=spec(), draws={"roi": np.ones((1, 4, 2))})
    for weight in (-0.1, 1.5):
        with pytest.raises(ValueError, match="from 0 to 1"):
            tempered_spec(spec(), previous, weight)  # ty: ignore[invalid-argument-type]
    half = dataclasses.replace(spec(), media=(BayesMedia("a", HalfNormalRoi()), BayesMedia("b")))
    with pytest.raises(ValueError, match="log-normal"):
        tempered_spec(half, previous, WEIGHT)  # ty: ignore[invalid-argument-type]
    assert tempered_spec(half, previous, 0.0) == half  # ty: ignore[invalid-argument-type]
    other = dataclasses.replace(spec(), media=(BayesMedia("a"), BayesMedia("c")))
    with pytest.raises(ValueError, match="channels"):
        tempered_spec(other, previous, WEIGHT)  # ty: ignore[invalid-argument-type]


def test_a_refresh_at_weight_0_is_the_full_refit_and_reports_its_drift() -> None:
    """At weight 0 the refresh is `fit_bayes` on the whole panel, draw for draw. The drift reads
    each return on the previous window's spend: before, the previous posterior's mean and 80 %;
    after, the refreshed one's mean of the same return; it leaves where outside."""
    panel = world(7, weeks=92)
    previous = fit_bayes(
        panel,
        NATIONAL,
        spec(),
        seed=1,
        window=(panel.dates[0], panel.dates[79]),
        chains=CHAINS,
        warmup=WARMUP,
        draws=DRAWS,
    )
    refreshed = refresh_bayes(
        panel, NATIONAL, previous, seed=2, weight=0.0, chains=CHAINS, warmup=WARMUP, draws=DRAWS
    )
    full = fit_bayes(panel, NATIONAL, spec(), seed=2, chains=CHAINS, warmup=WARMUP, draws=DRAWS)
    assert refreshed.weight == 0.0
    for name, drawn in full.draws.items():
        np.testing.assert_array_equal(refreshed.fit.draws[name], drawn)
    after = returns_on(full, panel, NATIONAL, previous.dates)
    for j, drift in enumerate(refreshed.drift):
        old = previous.draws["roi"][..., j].reshape(-1)
        assert drift.channel == ("a", "b")[j]
        assert drift.before == pytest.approx(float(np.mean(old)), rel=1e-12)
        np.testing.assert_allclose((drift.low, drift.high), np.quantile(old, [0.1, 0.9]))
        assert drift.after == pytest.approx(float(np.mean(after[:, j])), rel=1e-12)
        assert drift.change == drift.after - drift.before
        assert drift.leaves == (not drift.low <= drift.after <= drift.high)
    with pytest.raises(ValueError, match="holds the previous"):
        refresh_bayes(panel, NATIONAL, previous, seed=2, window=(panel.dates[10], panel.dates[-1]))
    with pytest.raises(ValueError, match="level"):
        refresh_bayes(panel, NATIONAL, previous, seed=2, level=1.0)


def test_refits_on_four_more_weeks_stay_inside_the_80_percent_before_them() -> None:
    """On a world that holds still, the model fitted on its first 80 weeks and refitted six times
    on four more each time: each channel's return on the year that ends at week 80, read at each
    fit, has its mean inside the central 80 % of the fit before it, every time. Its posterior
    moves by about its spread times the square root of the share of weeks added, a tenth of the
    80 % interval's half-width, so the test fails only on a refit that jumps."""
    panel = world(8)
    common = panel.dates[28:80]
    readings = []
    for weeks in range(80, 105, 4):
        fit = fit_bayes(
            panel,
            NATIONAL,
            spec(1),
            seed=weeks,
            window=(panel.dates[0], panel.dates[weeks - 1]),
            chains=CHAINS,
            warmup=WARMUP,
            draws=DRAWS,
        )
        readings.append(returns_on(fit, panel, NATIONAL, common))
    for before, after in itertools.pairwise(readings):
        low, high = np.quantile(before, [0.1, 0.9], axis=0)
        mean = after.mean(axis=0)
        assert np.all((low <= mean) & (mean <= high)), (low, mean, high)


@pytest.mark.skipif(shutil.which("maxima") is None, reason="maxima is not on PATH")
def test_maxima_proves_the_tempered_prior_and_its_bounds_at_every_step() -> None:
    script = Path(__file__).parents[1] / "validation" / "refresh_prior.mac"
    out = subprocess.run(
        ["maxima", "--very-quiet", "-b", str(script)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    ).stdout
    assert "Lisp error" not in out
    verdicts = [line.strip() for line in out.splitlines() if line.startswith("STEP ")]
    assert len(verdicts) == 6
    assert all(line.endswith("(PASS)") for line in verdicts), verdicts
