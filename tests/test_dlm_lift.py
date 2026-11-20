"""Lift tests read by the DLM: each an observation of its channel's coefficient path over its
readout (`mixdyn.models.dlm_lift`).

The oracles:

* Maxima's (`fixtures/oracles/dlm_lift_maxima.json`): one channel over three weeks, a test read over
  the last two, the KPI's smoother conditioned on it against the batch posterior that holds the test
  as one more observation, exactly, with the variance known and learned, and the log-likelihood
  against the batch's log density;
* a batch posterior in NumPy, over a level, a season and two channels through their transforms,
  with a week of the KPI missing, a channel idle three weeks in six and three tests, two of them
  overlapping, one in half the market and one ending at the last week;
* CHC's sampler, draw for draw, where no test is read;
* rejection draws of the posterior given a test, kept where the sign holds, against the signed
  sampler given it.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from chc.dlm import Prior, Sign, confidence_set, constrained_sample, forecast
from scipy import stats

from mixdyn import NATIONAL, Cadence, Channel, Geo, MediaPanel, Revenue
from mixdyn.calibration import LiftEstimate
from mixdyn.models import (
    DiscountGrid,
    DlmFit,
    DlmSpec,
    Effect,
    Season,
    TransformGrid,
    Trend,
    choose_dlm,
    decompose_dlm,
    fit_dlm,
    forecast_dlm,
    forecast_draws,
    search_media,
    vague_prior,
)
from mixdyn.models.dlm import _planned, _transition
from mixdyn.models.dlm_choice import within
from mixdyn.models.dlm_lift import lift_reading, lifted, lifted_draws, signed_sample
from mixdyn.transforms import HillShape, TanhShape, Transform

ORACLE = json.loads(
    (Path(__file__).parent / "fixtures" / "oracles" / "dlm_lift_maxima.json").read_text()
)


def weekly(kpi: np.ndarray, spend: dict[str, np.ndarray], controls: dict[str, np.ndarray]):
    weeks = kpi.size
    return MediaPanel(
        dates=np.datetime64("2025-01-06") + np.timedelta64(7, "D") * np.arange(weeks),
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=tuple(Channel(c) for c in spend),
        controls=tuple(controls),
        kpi=kpi[None],
        kpi_unit=Revenue(),
        spend=np.column_stack(list(spend.values()))[None],
        currency="EUR",
        control_values=(
            np.column_stack(list(controls.values())) if controls else np.zeros((weeks, 0))
        )[None],
        filled=np.zeros((1, weeks), dtype=bool),
    )


# ---------------------------------------------------------------------------------- Maxima's case


def exact(record: dict[str, Any]) -> np.ndarray:
    return np.array([float(Fraction(v)) for v in record["exact"]])


def three_weeks(case: str) -> DlmFit:
    instance = ORACLE["instance"]
    x = np.array(instance["x"], dtype=float)
    data = weekly(np.array(instance["y"], dtype=float), {"x": x}, {})
    dof = math.inf if case == "known" else float(instance["n0"])
    prior = Prior(np.array([float(instance["m0"])]), np.array([[float(instance["C0"])]]), 2.0, dof)
    estimate = LiftEstimate(
        spend=np.array(instance["spend"], dtype=float),
        planned=x,
        readout=2,
        lift=float(instance["lift"]),
        se=math.sqrt(instance["se2"]),
        end=data.dates[-1],
    )
    delta = float(Fraction(instance["delta"]["exact"]))
    spec = DlmSpec((Effect("x", delta),))
    return fit_dlm(data, NATIONAL, spec, prior, estimates={"x": (estimate,)})


def test_every_maxima_check_held_every_control_failed_and_its_rationals_are_its_floats() -> None:
    assert len(ORACLE["checks"]) == 14
    assert all(check["holds"] for check in ORACLE["checks"])
    assert len(ORACLE["controls"]) == 6
    assert all(control["fails"] for control in ORACLE["controls"])
    for case in ("known", "learned"):
        for key in ("mean", "variance in units of V", "lag-one covariance in units of V"):
            record = ORACLE[case][key]
            assert exact(record).tolist() == record["value"]


@pytest.mark.parametrize("case", ["known", "learned"])
def test_a_test_read_by_the_smoother_is_the_batch_posterior_that_holds_it(case: str) -> None:
    """The smoothed means, variances and lag-one covariances given the test, the state at the last
    week, the variance's estimate and the log-likelihood are Maxima's batch ones, to rounding."""
    fit = three_weeks(case)
    record = ORACLE[case]
    expected = float(Fraction(record["expected variance"]["exact"]))
    scale = float(Fraction(record["scale"]["exact"]))
    variance = exact(record["variance in units of V"])
    np.testing.assert_allclose(fit.smoothed.mean[:, 0], exact(record["mean"]), rtol=1e-13, atol=0)
    np.testing.assert_allclose(
        fit.smoothed.covariance[:, 0, 0], expected * variance, rtol=1e-13, atol=0
    )
    np.testing.assert_allclose(
        fit.smoothed.cross_covariance[:, 0, 0],
        expected * exact(record["lag-one covariance in units of V"]),
        rtol=1e-13,
        atol=0,
    )
    mean, covariance, dof, estimate = fit.last()
    assert mean[0] == pytest.approx(exact(record["mean"])[-1], rel=1e-13, abs=0.0)
    assert covariance[0, 0] == pytest.approx(scale * variance[-1], rel=1e-13, abs=0.0)
    assert dof == (math.inf if case == "known" else record["dof"])
    assert estimate == pytest.approx(scale, rel=1e-13, abs=0.0)
    kpi = record["kpi log-likelihood"]["value"]
    assert fit.filtered.log_likelihood == pytest.approx(kpi, rel=1e-13, abs=0.0)
    assert fit.log_likelihood == pytest.approx(
        record["log-likelihood"]["value"], rel=1e-13, abs=0.0
    )
    assert fit.lifted is not None
    assert fit.lifted.reading.weights.tolist() == [ORACLE["instance"]["weights"]]


# ---------------------------------------------------------------------------- a batch in NumPy

WEEKS = 24
TV = Transform(0.5, 100.0, TanhShape(), 4)
SEARCH = Transform(0.2, 30.0, HillShape(1.5), 3)


@functools.cache
def media() -> MediaPanel:
    rng = np.random.default_rng(20261011)
    week = np.arange(WEEKS)
    tv = np.where(week % 6 < 3, rng.uniform(60.0, 140.0, WEEKS), 0.0)
    search = rng.uniform(20.0, 50.0, WEEKS)
    price = 10.0 + rng.standard_normal(WEEKS)
    drift = 1.0 + 0.3 * np.sin(week / 4.0)
    kpi = (
        500.0
        + 40.0 * np.sin(2.0 * math.pi * week / 6.0)
        + 200.0 * drift * TV.column(tv)
        + 80.0 * SEARCH.column(search)
        - 5.0 * (price - 10.0)
        + 6.0 * rng.standard_normal(WEEKS)
    )
    kpi[7] = np.nan
    return weekly(kpi, {"tv": tv, "search": search}, {"price": price})


def media_spec(tv: Sign | None = None, search: Sign | None = None) -> DlmSpec:
    return DlmSpec(
        (
            Effect("tv", 0.9, sign=tv, transform=TV),
            Effect("search", 0.95, sign=search, transform=SEARCH),
            Effect("price", 1.0),
        ),
        trend=Trend(1, 0.95),
        season=Season(6.0, 1, 0.97),
    )


def lifts_of(data: MediaPanel) -> dict[str, tuple[LiftEstimate, ...]]:
    """Two go-dark tests of television, readouts weeks 8-13 and 11-16, one series starting four
    weeks before the panel; one heavy-up of search in half the market, readout weeks 18-23."""
    tv = data.spend[0, :, 0]
    search = data.spend[0, :, 1]
    early = np.concatenate([np.full(4, 90.0), tv[:14]])
    dark = early.copy()
    dark[12:16] = 0.0
    late = tv[:17].copy()
    off = late.copy()
    off[11:14] = 0.0
    up = 0.5 * search.copy()
    raised = up.copy()
    raised[18:21] *= 2.0
    return {
        "tv": (
            LiftEstimate(dark, early, 6, -150.0, 25.0, end=data.dates[13]),
            LiftEstimate(off, late, 6, -90.0, 40.0, end=data.dates[16]),
        ),
        "search": (LiftEstimate(raised, up, 6, 30.0, 12.0, share=0.5, end=data.dates[23]),),
    }


def batch(fit: DlmFit, estimates: dict[str, tuple[LiftEstimate, ...]]) -> dict[str, Any]:
    """The posterior of the whole path given the KPI and the tests at once, from the filter's
    evolution variances in units of the variance, and the joint log density of what it read."""
    filtered = fit.filtered
    horizon, p = filtered.mean.shape
    g = _transition(filtered)
    model = filtered.model
    covariance = np.zeros((horizon, p, horizon, p))
    covariance[0, :, 0] = filtered.prior_covariance[0] / model.prior.scale
    for t in range(1, horizon):
        w = (filtered.prior_covariance[t] - g @ filtered.covariance[t - 1] @ g.T) / filtered.scale[
            t - 1
        ]
        for s in range(t):
            covariance[t, :, s] = g @ covariance[t - 1, :, s]
            covariance[s, :, t] = covariance[t, :, s].T
        covariance[t, :, t] = g @ covariance[t - 1, :, t - 1] @ g.T + w
    big = covariance.reshape(horizon * p, horizon * p)
    mean = np.empty((horizon, p))
    mean[0] = filtered.prior_mean[0]
    for t in range(1, horizon):
        mean[t] = g @ mean[t - 1]
    observed = np.flatnonzero(~np.isnan(filtered.y))
    rows, values, units = [], [], []
    for t in observed:
        row = np.zeros((horizon, p))
        row[t] = filtered.design[t]
        rows.append(row.ravel())
        values.append(filtered.y[t])
        units.append(1.0)
    reading = lift_reading(media(), NATIONAL, fit.spec, estimates)
    for j in range(reading.lift.size):
        row = np.zeros((horizon, p))
        row[:, reading.coordinates[j]] = reading.weights[j]
        rows.append(row.ravel())
        values.append(reading.lift[j])
        units.append(reading.se[j] ** 2 / filtered.scale[-1])
    h = np.array(rows)
    m = h @ big @ h.T + np.diag(units)
    r = np.array(values) - h @ mean.ravel()
    gain = big @ h.T @ np.linalg.inv(m)
    posterior = (mean.ravel() + gain @ r).reshape(horizon, p)
    spread = (big - gain @ h @ big).reshape(horizon, p, horizon, p)
    n0, s0 = model.prior.dof, model.prior.scale
    if math.isinf(n0):
        scale, expected = s0, s0
        density = stats.multivariate_normal(np.zeros(r.size), s0 * m).logpdf(r)
    else:
        n = n0 + r.size
        scale = (n0 * s0 + r @ np.linalg.solve(m, r)) / n
        expected = scale * n / (n - 2.0)
        density = stats.multivariate_t(np.zeros(r.size), s0 * m, df=n0).logpdf(r)
    return {
        "mean": posterior,
        "spread": spread,
        "scale": scale,
        "expected": expected,
        "density": float(density),
    }


@pytest.mark.parametrize("known", [True, False])
def test_tests_read_by_the_smoother_are_the_batch_posterior_that_holds_them(known: bool) -> None:
    data = media()
    prior = vague_prior(data, NATIONAL, media_spec())
    if known:
        prior = dataclasses.replace(prior, scale=36.0, dof=math.inf)
    estimates = lifts_of(data)
    fit = fit_dlm(data, NATIONAL, media_spec(), prior, estimates=estimates)
    truth = batch(fit, estimates)
    horizon = WEEKS
    spread = truth["spread"]
    marginal = np.stack([spread[t, :, t] for t in range(horizon)])
    lagged = np.stack([spread[t, :, t + 1] for t in range(horizon - 1)])
    scale = np.abs(marginal).max()
    np.testing.assert_allclose(fit.smoothed.mean, truth["mean"], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(
        fit.smoothed.covariance, truth["expected"] * marginal, rtol=0, atol=1e-9 * scale
    )
    np.testing.assert_allclose(
        fit.smoothed.cross_covariance, truth["expected"] * lagged, rtol=0, atol=1e-9 * scale
    )
    mean, covariance, _, estimate = fit.last()
    np.testing.assert_allclose(mean, truth["mean"][-1], rtol=1e-9, atol=1e-9)
    assert estimate == pytest.approx(truth["scale"], rel=1e-10, abs=0.0)
    np.testing.assert_allclose(
        covariance, truth["scale"] * marginal[-1], rtol=0, atol=1e-9 * truth["scale"] * scale
    )
    assert fit.log_likelihood == pytest.approx(truth["density"], rel=1e-10, abs=0.0)


def test_the_tests_move_the_path_they_read_and_add_their_density_to_the_kpi_s() -> None:
    data = media()
    estimates = lifts_of(data)
    fit = fit_dlm(data, NATIONAL, media_spec(), estimates=estimates)
    blind = fit_dlm(data, NATIONAL, media_spec())
    assert fit.lifted is not None
    assert fit.filtered.log_likelihood == blind.filtered.log_likelihood
    assert fit.log_likelihood == blind.log_likelihood + fit.lifted.log_density
    # television's coefficient over the first readout moves; the KPI's filter does not
    assert np.abs(fit.smoothed.mean[8:14, 1] - blind.smoothed.mean[8:14, 1]).max() > 1.0
    np.testing.assert_array_equal(fit.filtered.mean, blind.filtered.mean)


def test_a_test_s_weights_are_its_channel_s_column_gap_over_its_readout_at_the_market_s_scale():
    data = media()
    reading = lift_reading(data, NATIONAL, media_spec(), lifts_of(data))
    assert reading.channels == ("tv", "tv", "search")
    assert reading.coordinates.tolist() == [1, 1, 2]
    (first, second), (heavy,) = lifts_of(data)["tv"], lifts_of(data)["search"]
    gap = TV.column(first.spend) - TV.column(first.planned)
    np.testing.assert_array_equal(reading.weights[0, 8:14], gap[-6:])
    assert not reading.weights[0, :8].any()
    assert not reading.weights[0, 14:].any()
    later = TV.column(second.spend) - TV.column(second.planned)
    np.testing.assert_array_equal(reading.weights[1, 11:17], later[-6:])
    assert not reading.weights[1, :11].any()
    assert not reading.weights[1, 17:].any()
    half = SEARCH.column(heavy.spend / 0.5) - SEARCH.column(heavy.planned / 0.5)
    np.testing.assert_array_equal(reading.weights[2, 18:], half[-6:])
    assert reading.lift.tolist() == [-150.0, -90.0, 60.0]
    assert reading.se.tolist() == [25.0, 40.0, 24.0]


# ----------------------------------------------------------------------------- what it refuses


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"end": None}, "has no end"),
        ({"end": np.datetime64("2025-01-07")}, "outside the panel's calendar"),
        ({"end": np.datetime64("2025-01-13")}, "reads 6 periods ending 2025-01-13, outside"),
        ({"end": np.datetime64("2026-01-05")}, "outside the panel's calendar"),
        ({"geos": frozenset({Geo("north")})}, r"ran in \['north'\], not in 'national'"),
    ],
)
def test_an_estimate_the_dlm_cannot_place_in_its_calendar_is_refused(change, match) -> None:
    data = media()
    estimates = lifts_of(data)
    moved = {"tv": (dataclasses.replace(estimates["tv"][0], **change),)}
    with pytest.raises(ValueError, match=match):
        fit_dlm(data, NATIONAL, media_spec(), estimates=moved)


def test_a_test_of_a_control_an_unknown_channel_or_none_is_refused() -> None:
    data = media()
    (estimate, _) = lifts_of(data)["tv"]
    for estimates in ({"price": (estimate,)}, {"radio": (estimate,)}, {"tv": ()}):
        with pytest.raises(ValueError, match="estimates name channels"):
            fit_dlm(data, NATIONAL, media_spec(), estimates=estimates)


def test_a_signed_fit_refuses_a_test_of_a_channel_not_held_to_its_sign() -> None:
    data = media()
    with pytest.raises(ValueError, match="not held to a sign"):
        fit_dlm(data, NATIONAL, media_spec(search="positive"), draws=20, estimates=lifts_of(data))


def test_a_variance_that_moves_along_the_path_is_refused() -> None:
    data = media()
    spec = dataclasses.replace(media_spec(), variance_discount=0.99)
    with pytest.raises(ValueError, match=r"variance discount of 0\.99"):
        fit_dlm(data, NATIONAL, spec, estimates=lifts_of(data))


def test_an_end_that_is_not_a_date_is_refused() -> None:
    estimate = lifts_of(media())["tv"][0]
    for end in ("2025-03-31", np.datetime64("NaT", "D")):
        with pytest.raises(ValueError, match="must be a date"):
            dataclasses.replace(estimate, end=end)


def test_a_fit_read_with_tests_hands_its_channels_at_the_last_period_only() -> None:
    data = media()
    fit = fit_dlm(data, NATIONAL, media_spec(), estimates=lifts_of(data))
    assert fit.lifted is not None
    channels = fit.channels()
    assert float(channels["tv"].coefficient) == fit.lifted.mean[1]
    with pytest.raises(ValueError, match="read at the last, not at 10"):
        fit.channels(10)


# ------------------------------------------------------------------- draws, forecasts, the choice


def test_the_draws_given_the_tests_are_the_posterior_given_them() -> None:
    """Matheron's rule on the KPI's backward draws: 40 000 draws' means within 4.5 Monte Carlo
    errors of the smoothed means at every week, their variances at the last week within 3 % of
    the smoothed ones, and the variance's draws averaging its posterior's mean."""
    data = media()
    fit = fit_dlm(data, NATIONAL, media_spec(), estimates=lifts_of(data))
    assert fit.lifted is not None
    draws = lifted_draws(fit.filtered, fit.lifted, 40_000, seed=11)
    sd = np.sqrt(np.einsum("tii->ti", fit.smoothed.covariance))
    z = (draws.states.mean(axis=0) - fit.smoothed.mean) / (sd / math.sqrt(40_000))
    assert np.abs(z).max() < 4.5
    np.testing.assert_allclose(
        np.var(draws.states[:, -1], axis=0), sd[-1] ** 2, rtol=0.03, atol=0.0
    )
    expected = fit.lifted.scale * fit.lifted.dof / (fit.lifted.dof - 2.0)
    assert draws.variance[:, 0].mean() == pytest.approx(expected, rel=0.02, abs=0.0)
    decomposition = decompose_dlm(fit, draws=2000, seed=4)
    np.testing.assert_allclose(
        decomposition.draws.contributions["tv"].mean(axis=0),
        fit.contributions["tv"],
        rtol=0.0,
        atol=6.0 * np.abs(fit.contributions["tv"]).std() / math.sqrt(2000),
    )


def test_a_forecast_given_the_tests_starts_from_their_state_and_evolves_as_the_kpi_s_filter():
    """Unrolled, the forecast's state covariance `k` steps on is `G^k (C' - r C_T) G'^k + r A_k`,
    `C'` the last state's given the tests, `C_T` the filter's, `A_k` the KPI's forecast's and `r`
    the variance's estimate given the tests over the filter's; its draws agree with it."""
    data = media()
    fit = fit_dlm(data, NATIONAL, media_spec(), estimates=lifts_of(data))
    plan = {"tv": np.full(3, 100.0), "search": np.full(3, 30.0), "price": np.full(3, 10.0)}
    ahead = forecast_dlm(fit, 3, plan)
    filtered = fit.filtered
    g = _transition(filtered)
    x = _planned(fit, 3, plan)
    kpi = forecast(filtered, 3, x, evolution="constant")
    mean, covariance, dof, scale = fit.last()
    r = scale / filtered.scale[-1]
    power = np.eye(g.shape[0])
    for k in range(3):
        power = g @ power
        design = filtered.design[-1].copy()
        design[1:4] = x[k]
        state = (
            power @ (covariance - r * filtered.covariance[-1]) @ power.T + r * kpi.state_scale[k]
        )
        assert ahead.mean[k] == pytest.approx(design @ power @ mean, rel=1e-12, abs=0.0)
        assert ahead.scale[k] == pytest.approx(design @ state @ design + scale, rel=1e-10, abs=0.0)
    assert ahead.dof.tolist() == [dof] * 3
    draws = forecast_draws(fit, 3, plan, 40_000, seed=5)
    z = (draws.kpi.mean(axis=0) - ahead.mean) / np.sqrt(ahead.scale * dof / (dof - 2) / 40_000)
    assert np.abs(z).max() < 4.5
    np.testing.assert_allclose(
        draws.kpi.var(axis=0), ahead.scale * dof / (dof - 2), rtol=0.05, atol=0.0
    )


def test_a_forecast_given_a_test_that_reads_nothing_is_the_kpi_s() -> None:
    """With the variance known, a test whose standard error is 10^8 leaves the KPI's forecast."""
    data = media()
    prior = dataclasses.replace(vague_prior(data, NATIONAL, media_spec()), scale=36.0, dof=math.inf)
    loose = {"tv": (dataclasses.replace(lifts_of(data)["tv"][0], se=1e8),)}
    plan = {"tv": np.full(3, 100.0), "search": np.full(3, 30.0), "price": np.full(3, 10.0)}
    told = forecast_dlm(fit_dlm(data, NATIONAL, media_spec(), prior, estimates=loose), 3, plan)
    blind = forecast_dlm(fit_dlm(data, NATIONAL, media_spec(), prior), 3, plan)
    np.testing.assert_allclose(told.mean, blind.mean, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(told.scale, blind.scale, rtol=1e-9, atol=0.0)
    assert told.dof.tolist() == blind.dof.tolist()


def drifting(weeks: int = 104) -> tuple[MediaPanel, dict[str, tuple[LiftEstimate, ...]]]:
    """A channel whose coefficient steps from 2 to 4 at week 52 under a KPI noisy enough that
    the KPI alone prefers it held still, and two go-dark tests that read it before and after
    the step."""
    rng = np.random.default_rng(8)
    x = rng.uniform(5.0, 15.0, weeks)
    beta = np.where(np.arange(weeks) < 52, 2.0, 4.0)
    data = weekly(100.0 + beta * x + 60.0 * rng.standard_normal(weeks), {"x": x}, {})
    tests = []
    for end in (30, 90):
        dark = x[: end + 1].copy()
        dark[end - 3 :] = 0.0
        lift = -float((beta[end - 3 : end + 1] * x[end - 3 : end + 1]).sum())
        tests.append(LiftEstimate(dark, x[: end + 1], 4, lift, 2.0, end=data.dates[end]))
    return data, {"x": tuple(tests)}


def test_the_tests_enter_each_spec_s_likelihood_and_move_the_kept_set() -> None:
    """Under the KPI alone a coefficient held still is the one spec kept; the tests, which read
    it at 2 and at 4, leave it out for one that moves, and the choice's log-likelihoods are the
    filters' and the tests' together."""
    data, estimates = drifting()
    spec = DlmSpec((Effect("x", 1.0),), trend=Trend(1, 1.0))
    grid = DiscountGrid(effects=(0.9, 0.97, 1.0))
    blind = choose_dlm(data, NATIONAL, spec, grid)
    told = choose_dlm(data, NATIONAL, spec, grid, estimates=estimates)
    assert (blind.best, tuple(blind.kept)) == (2, (2,))
    assert told.best == 1
    assert 2 not in told.kept
    assert len(told.lifts) == 3
    assert blind.lifts == ()
    for k, fit in told.kept.items():
        assert told.log_likelihoods[k] == fit.log_likelihood
        assert fit.lifted is not None
        assert told.lifts[k] == fit.lifted.log_density


def test_without_tests_the_kept_set_is_chc_s_confidence_set() -> None:
    data, _ = drifting()
    spec = DlmSpec((Effect("x", 1.0),), trend=Trend(1, 0.99))
    grid = DiscountGrid(trend=(0.95, 0.99, 1.0), effects=(0.9, 0.97, 0.99, 1.0))
    choice = choose_dlm(data, NATIONAL, spec, grid, level=0.5)
    assert len(choice.kept) > 1
    assert tuple(choice.kept) == confidence_set(list(choice.filtered), 2, 0.5)
    loglik = [f.log_likelihood for f in choice.filtered]
    assert within(loglik, 2, 0.5) == confidence_set(list(choice.filtered), 2, 0.5)


def test_the_media_search_reads_the_tests_in_every_spec_s_likelihood() -> None:
    data = media()
    estimates = lifts_of(data)
    grid = TransformGrid((0.0, 0.5), (1.0, 2.0), (TanhShape(),))
    discounts = DiscountGrid(effects=(0.9, 1.0))
    told = search_media(
        data, NATIONAL, media_spec(), grid, discounts, refinements=0, estimates=estimates
    )
    assert len(told.lifts) == len(told.specs) > 1
    for spec, filtered, lift in zip(told.specs, told.filtered, told.lifts, strict=True):
        assert lift == lifted(filtered, lift_reading(data, NATIONAL, spec, estimates)).log_density
    assert told.best == int(np.argmax(told.log_likelihoods))
    assert told.kept
    for k, fit in told.kept.items():
        assert fit.lifted is not None
        assert fit.log_likelihood == told.log_likelihoods[k]
    blind = search_media(data, NATIONAL, media_spec(), grid, discounts, refinements=0)
    assert blind.lifts == ()
    assert all(fit.lifted is None for fit in blind.kept.values())


# --------------------------------------------------------------------------- the signed sampler


@functools.cache
def small() -> MediaPanel:
    """30 weeks of a channel whose effect is small beside the noise, so that its sign binds."""
    rng = np.random.default_rng(30)
    x = rng.uniform(5.0, 15.0, 30)
    return weekly(50.0 + 0.3 * x + 4.0 * rng.standard_normal(30), {"x": x}, {})


def small_spec() -> DlmSpec:
    return DlmSpec((Effect("x", 0.9, sign="positive"),), trend=Trend(1, 1.0))


def small_lifts(data: MediaPanel) -> dict[str, tuple[LiftEstimate, ...]]:
    x = data.spend[0, :, 0]
    dark = x[:20].copy()
    dark[16:] = 0.0
    return {"x": (LiftEstimate(dark, x[:20], 4, -6.0, 6.0, end=data.dates[19]),)}


def test_with_no_test_the_signed_sampler_is_chc_s_draw_for_draw() -> None:
    data = small()
    fit = fit_dlm(data, NATIONAL, small_spec(), draws=50, seed=2)
    ours = signed_sample(fit.filtered, {0: "positive"}, None, 50, seed=8)
    theirs = constrained_sample(fit.filtered, {0: "positive"}, 50, seed=8)
    np.testing.assert_array_equal(ours.states, theirs.states)
    np.testing.assert_array_equal(ours.variance, theirs.variance)
    assert (ours.rhat, ours.ess) == (theirs.rhat, theirs.ess)


def test_the_signed_sampler_given_a_test_is_the_posterior_given_it_truncated_to_the_sign() -> None:
    """Rejection: draws of the posterior given the test, kept where the coefficient is at or
    above 0 at every week, against the signed sampler's chains, each mean within 4.5 joint Monte
    Carlo errors, the sampler's from its effective sample size; the sign binds, and the test moves
    the signed posterior."""
    data = small()
    estimates = small_lifts(data)
    unsigned = dataclasses.replace(small_spec(), effects=(Effect("x", 0.9),))
    free = fit_dlm(data, NATIONAL, unsigned, estimates=estimates)
    assert free.lifted is not None
    pool = lifted_draws(free.filtered, free.lifted, 400_000, seed=9)
    held = np.all(pool.states[:, :, 1] >= 0.0, axis=1)
    assert 0.05 < held.mean() < 0.9
    accepted = pool.states[held]
    signed = fit_dlm(data, NATIONAL, small_spec(), draws=4000, seed=1, estimates=estimates)
    assert signed.signed is not None
    assert signed.signed.mixed
    draws = signed.signed.pooled
    for ours, theirs in (
        (draws.states[:, :, 1], accepted[:, :, 1]),
        (draws.states[:, :, 0], accepted[:, :, 0]),
        (draws.variance[:, :1], pool.variance[held][:, :1]),
    ):
        error = np.sqrt(ours.var(axis=0) / signed.signed.ess + theirs.var(axis=0) / held.sum())
        assert np.abs(ours.mean(axis=0) - theirs.mean(axis=0)).max() < 4.5 * error.max()
    blind = fit_dlm(data, NATIONAL, small_spec(), draws=4000, seed=1)
    assert blind.signed is not None
    moved = blind.signed.pooled.states[:, 16:20, 1].mean() - draws.states[:, 16:20, 1].mean()
    assert abs(moved) > 0.05
