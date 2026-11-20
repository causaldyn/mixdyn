"""The ridge media-mix model: Robyn's models reached from the raw panel at Robyn's hyperparameters,
the truth recovered from a world without noise, the window, the gaps and the units handled as the
model states, and a fit calibrated by geo tests the joint least squares of the KPI and the tests."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
from chc.lift import GeoArm, LiftTest

from mixdyn import Cadence, Channel, Geo, MediaPanel, Revenue
from mixdyn.models.ridge_mmm import (
    Control,
    FourierSeason,
    LinearTrend,
    Media,
    RidgeFit,
    RidgeSpec,
    fit_ridge,
)
from mixdyn.transforms import Geometric, Hill, carry_over, saturate
from ridge_worlds import NATIONAL, panel_of, world, world_and_tests

ROBYN = json.loads(
    (Path(__file__).parent / "fixtures" / "oracles" / "robyn_ridge.json").read_text()
)


def _robyn(run: dict) -> tuple[RidgeFit, np.ndarray, np.ndarray]:
    """The run's model fitted from the raw panel, its window's KPI and design."""
    media = ROBYN["media"]
    controls = {c: np.array(run["columns"][c]) for c in run["controls"]}
    panel = MediaPanel(
        dates=np.array(ROBYN["dates"], dtype="datetime64[D]"),
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=tuple(Channel(m) for m in media),
        controls=tuple(controls),
        kpi=np.array(ROBYN["revenue"])[None, :],
        kpi_unit=Revenue(),
        spend=np.column_stack([ROBYN["spend"][m] for m in media])[None, :, :],
        currency="EUR",
        control_values=np.column_stack(list(controls.values()))[None, :, :],
        filled=np.zeros((1, len(ROBYN["dates"])), dtype=bool),
    )
    first, last = run["window"][0] - 1, run["window"][1] - 1
    inside = slice(first, last + 1)
    y = np.array(ROBYN["revenue"])[inside]
    hyper = ROBYN["hyperparameters"]
    spec = RidgeSpec(
        media=tuple(
            Media(m, Geometric(hyper["thetas"][i]), Hill(hyper["alphas"][i], hyper["gammas"][i]))
            for i, m in enumerate(media)
        ),
        # Robyn holds prophet's components to [0, 1], its trend to [-1, 0] where it sums below 0
        controls=tuple(
            Control(c, *((-1.0, 0.0) if c == "trend" and controls[c][inside].sum() < 0 else (0, 1)))
            if c in ("trend", "season", "holiday")
            else Control(c)
            for c in controls
        ),
        penalty=run["lambda"] / np.sqrt(np.mean((y - y.mean()) ** 2)),
    )
    fit = fit_ridge(panel, NATIONAL, spec, (panel.dates[first], panel.dates[last]))
    return fit, y, fit.design


@pytest.mark.parametrize("run", ROBYN["runs"], ids=lambda r: r["name"])
def test_robyn_s_model_is_reached_from_the_rawpanel_of(run: dict) -> None:
    """The oracle is glmnet run to a threshold of 1e-14 on Robyn's design, which the harness
    rebuilt and checked returns Robyn's own coefficients at glmnet's default. Nearly
    unpenalised (capped) even that stops 1.4e-6 of y's spread short, so the coefficients agree
    to 1e-5 of it in standardised units, and neither glmnet's point lies below this solver's
    objective by more than the objective's rounding. glmnet returns a coefficient it clamps as
    the limit itself, so the coefficients on their bounds are Robyn's: in the capped run the
    holiday at Robyn's cap of 1 on a prophet component."""
    fit, y, design = _robyn(run)
    np.testing.assert_allclose(
        [fit.half_saturation[m] for m in ROBYN["media"]], run["inflexion"], rtol=1e-15
    )
    spread = np.sqrt(np.mean((y - y.mean()) ** 2))
    terms = fit.spec.terms()
    scale = np.sqrt(np.mean((design - design.mean(axis=0)) ** 2, axis=0))
    ours = np.array([fit.coefficients[t] for t in terms])
    converged = dict(zip(run["terms"], run["converged"], strict=True))
    robyn = dict(zip(run["terms"], run["coefficients"], strict=True))
    oracle = np.array([converged[t] for t in terms])
    assert np.abs((ours - oracle) * scale).max() < 1e-5 * spread
    assert abs(fit.intercept - converged["(Intercept)"]) < 1e-5 * spread

    def objective(point: dict[str, float]) -> float:
        b = np.array([point[t] for t in terms])
        residual = y - point["(Intercept)"] - design @ b
        return np.mean(residual**2) / 2 + fit.spec.penalty / 2 * np.sum((scale * b) ** 2)

    lowest = objective(dict(fit.coefficients) | {"(Intercept)": fit.intercept})
    assert lowest <= min(objective(converged), objective(robyn)) * (1 + 1e-13)
    limits = {m.channel: (0.0, np.inf) for m in fit.spec.media}
    limits |= {c.column: (c.lower, c.upper) for c in fit.spec.controls}
    assert fit.bound == {t: robyn[t] in limits[t] for t in terms}
    assert fit.bound["holiday"] == (run["name"] == "capped")


@pytest.mark.parametrize(
    "run", [r for r in ROBYN["runs"] if r["name"] != "capped"], ids=lambda r: r["name"]
)
def test_robyn_s_scores_are_reached_where_its_glmnet_converges(run: dict) -> None:
    """Penalised, Robyn's glmnet stops within 1e-10 of the minimum's objective, and its fit and
    scores are this one's to 1e-5. In the capped run they carry its glmnet's error instead."""
    fit, y, _ = _robyn(run)
    spread = np.sqrt(np.mean((y - y.mean()) ** 2))
    assert np.abs(fit.estimated - np.array(run["fitted"])).max() < 1e-5 * spread
    assert fit.nrmse() == pytest.approx(run["nrmse_train"], rel=1e-5)
    assert fit.decomp_rssd() == pytest.approx(run["decomp_rssd"], rel=1e-5)


def test_the_truth_is_recovered_from_a_world_without_noise() -> None:
    spend, media, beta = world()
    columns = []
    for j, m in enumerate(media):
        adstocked = carry_over(spend[:, j], m.adstock)
        columns.append(saturate(adstocked, m.hill, m.hill.gamma * adstocked.max()))
    trend = np.arange(spend.shape[0], dtype=float)
    kpi = 1000.0 + np.column_stack(columns) @ beta + 2.0 * trend
    fit = fit_ridge(
        panel_of(kpi, spend, {"trend": trend}),
        NATIONAL,
        RidgeSpec(tuple(media), (Control("trend"),), penalty=1e-12),
    )
    np.testing.assert_allclose([fit.coefficients[m.channel] for m in media], beta, rtol=1e-8)
    assert fit.coefficients["trend"] == pytest.approx(2.0, rel=1e-8)
    assert fit.intercept == pytest.approx(1000.0, rel=1e-8)
    assert fit.nrmse() < 1e-9


def test_a_bent_trend_and_a_season_are_recovered_from_a_world_without_noise() -> None:
    spend, media, beta = world()
    columns = [
        saturate(a := carry_over(spend[:, j], m.adstock), m.hill, m.hill.gamma * a.max())
        for j, m in enumerate(media)
    ]
    t = np.arange(120.0)
    angle = 2.0 * np.pi * t / 52.18
    calendar = 2.0 * t + 3.0 * np.maximum(0.0, t - 60.0) + 50.0 * np.cos(angle)
    kpi = 1000.0 + np.column_stack(columns) @ beta + calendar - 30.0 * np.sin(2.0 * angle)
    spec = RidgeSpec(tuple(media), (), 1e-12, LinearTrend(1), FourierSeason(52.18, 2))
    fit = fit_ridge(panel_of(kpi, spend), NATIONAL, spec)
    assert spec.terms()[3:] == (
        "trend",
        "trend_bend_1",
        "season_cos_1",
        "season_sin_1",
        "season_cos_2",
        "season_sin_2",
    )
    np.testing.assert_allclose([fit.coefficients[m.channel] for m in media], beta, rtol=1e-8)
    np.testing.assert_allclose(
        [fit.coefficients[term] for term in spec.terms()[3:]],
        [2.0, 3.0, 50.0, 0.0, 0.0, -30.0],
        rtol=1e-8,
        atol=1e-8,
    )
    assert not any(fit.bound[term] for term in spec.terms()[3:])


def test_the_trend_bends_at_knots_spaced_evenly_and_the_season_counts_from_the_window() -> None:
    t = np.arange(120.0)
    np.testing.assert_array_equal(
        LinearTrend(3).columns(120),
        np.column_stack([t, *(np.maximum(0.0, t - k) for k in (30.0, 60.0, 90.0))]),
    )
    angle = 2.0 * np.pi * np.arange(10.0) / 13.0
    np.testing.assert_allclose(
        FourierSeason(13.0, 2).columns(10),
        np.column_stack([np.cos(angle), np.sin(angle), np.cos(2 * angle), np.sin(2 * angle)]),
        rtol=1e-15,
        atol=1e-15,
    )
    spend, media, _ = world()
    kpi = 1000.0 + spend @ np.array([1.0, 0.5, 0.2]) + np.random.default_rng(9).normal(0, 5, 120)
    panel = panel_of(kpi, spend)
    spec = RidgeSpec(tuple(media), (), 0.1, LinearTrend(1), FourierSeason(13.0, 2))
    fit = fit_ridge(panel, NATIONAL, spec, (panel.dates[40], panel.dates[-1]))
    np.testing.assert_array_equal(fit.design[:, 3:], spec.calendar(80))


def test_the_trend_runs_on_past_the_window_from_its_last_bend() -> None:
    """Ten periods past a window of 120, the knots still the window's."""
    t = np.arange(130.0)
    ahead = LinearTrend(3).columns(120, ahead=10)
    np.testing.assert_array_equal(ahead[:120], LinearTrend(3).columns(120))
    np.testing.assert_array_equal(
        ahead, np.column_stack([t, *(np.maximum(0.0, t - k) for k in (30.0, 60.0, 90.0))])
    )


def test_spend_before_the_window_carries_into_it_and_a_missing_week_is_carried_through() -> None:
    spend, media, _ = world()
    spend[38:40] = 0.0
    spend[35:38, 0] = 500.0
    spend[60, 1] = 2000.0
    kpi = 1000.0 + spend @ np.array([1.0, 0.5, 0.2]) + np.random.default_rng(1).normal(0, 5, 120)
    kpi[60] = np.nan
    panel = panel_of(kpi, spend)
    fit = fit_ridge(
        panel, NATIONAL, RidgeSpec(tuple(media), (), 0.1), (panel.dates[38], panel.dates[-1])
    )
    for j, m in enumerate(media):
        adstocked = carry_over(spend[:, j], m.adstock)[38:]
        np.testing.assert_allclose(
            fit.design[:, j],
            saturate(adstocked, m.hill, fit.half_saturation[m.channel]),
            rtol=1e-14,
        )
        assert fit.half_saturation[m.channel] == m.hill.gamma * adstocked.max()
    assert fit.design[0, 0] > 0.0
    assert not fit.fitted[60 - 38]
    assert fit.fitted.sum() == 120 - 38 - 1
    effect = np.array([np.sum(fit.contributions[m.channel]) for m in media])
    share = spend[38:].sum(axis=0)
    distance = np.sqrt(np.sum((effect / effect.sum() - share / share.sum()) ** 2))
    assert fit.decomp_rssd() == pytest.approx(distance, rel=1e-14, abs=0.0)


def test_the_planner_s_channels_return_the_fit_s_contributions() -> None:
    spend, media, _ = world()
    kpi = 1000.0 + spend @ np.array([1.0, 0.5, 0.2]) + np.random.default_rng(2).normal(0, 5, 120)
    panel = panel_of(kpi, spend)
    fit = fit_ridge(
        panel, NATIONAL, RidgeSpec(tuple(media), (), 0.05), (panel.dates[20], panel.dates[-1])
    )
    for j, (name, planned) in enumerate(fit.channels().items()):
        np.testing.assert_allclose(
            np.asarray(planned(spend[:, j]))[20:], fit.contributions[name], rtol=1e-13, atol=1e-12
        )


def test_what_pulls_against_its_bound_sits_on_it_and_a_dead_channel_raises_decomp_rssd() -> None:
    spend, media, _ = world()
    columns = [
        saturate(a := carry_over(spend[:, j], m.adstock), m.hill, m.hill.gamma * a.max())
        for j, m in enumerate(media)
    ]
    rng = np.random.default_rng(3)
    promo = rng.random(120)
    noise = rng.normal(0, 1.0, 120)
    kpi = 1000.0 + 300.0 * columns[0] + 200.0 * columns[1] - 150.0 * columns[2] - 80 * promo + noise
    spec = RidgeSpec(tuple(media), (Control("promo", lower=0.0),), 1e-4)
    fit = fit_ridge(panel_of(kpi, spend, {"promo": promo}), NATIONAL, spec)
    assert fit.bound == {"c0": False, "c1": False, "c2": True, "promo": True}
    assert fit.coefficients["c2"] == fit.coefficients["promo"] == 0.0
    effect = np.array([fit.contributions[m.channel].sum() for m in media])
    share = np.array([fit.spend[m.channel] for m in media])
    distance = np.sqrt(np.sum((effect / effect.sum() - share / share.sum()) ** 2))
    assert fit.decomp_rssd() == pytest.approx(distance * (1 + 1 / 3), rel=1e-14, abs=0.0)
    # a share of a thousandth is small, not nothing: Robyn rounds at four decimals
    small = 2e-3 * fit.contributions["c0"]
    alive = dataclasses.replace(fit, contributions=dict(fit.contributions) | {"c2": small})
    effect[2] = small.sum()
    assert 1e-4 < effect[2] / effect.sum() < 5e-3
    distance = np.sqrt(np.sum((effect / effect.sum() - share / share.sum()) ** 2))
    assert alive.decomp_rssd() == pytest.approx(distance, rel=1e-14, abs=0.0)


def test_units_move_only_what_they_should() -> None:
    """A channel's spend in other units moves nothing but its spend share, the KPI's scales the
    coefficients and the intercept, and a control's moves its own coefficient alone. DECOMP.RSSD
    compares spend across channels, so only a currency every channel shares leaves it alone."""
    spend, media, _ = world()
    rng = np.random.default_rng(4)
    price = rng.normal(10.0, 1.0, 120)
    kpi = 1000.0 + spend @ np.array([1.0, 0.5, 0.2]) - 20.0 * price + rng.normal(0, 5, 120)
    spec = RidgeSpec(tuple(media), (Control("price", upper=0.0),), 0.02)
    base = fit_ridge(panel_of(kpi, spend, {"price": price}), NATIONAL, spec)
    moved = fit_ridge(
        panel_of(3.0 * kpi, spend * np.array([1e-3, 1.0, 40.0]), {"price": 100.0 * price}),
        NATIONAL,
        spec,
    )
    for m in media:
        assert moved.coefficients[m.channel] == pytest.approx(
            3.0 * base.coefficients[m.channel], rel=1e-10
        )
    assert moved.coefficients["price"] == pytest.approx(
        0.03 * base.coefficients["price"], rel=1e-10
    )
    assert moved.intercept == pytest.approx(3.0 * base.intercept, rel=1e-10)
    assert moved.nrmse() == pytest.approx(base.nrmse(), rel=1e-10)
    shared = fit_ridge(panel_of(kpi, 7.0 * spend, {"price": price}), NATIONAL, spec)
    assert shared.decomp_rssd() == pytest.approx(base.decomp_rssd(), rel=1e-10)


def test_each_geo_is_fitted_to_its_own_data() -> None:
    spend, media, _ = world()
    rng = np.random.default_rng(8)
    spends = (spend, spend[::-1].copy())
    kpis = [1000.0 + s @ np.array([1.0, 0.5, 0.2]) + rng.normal(0, 5, 120) for s in spends]
    prices = [rng.normal(10.0, 1.0, 120) for _ in spends]
    one = panel_of(kpis[0], spends[0], {"price": prices[0]})
    two = dataclasses.replace(
        one,
        geos=(Geo("north"), Geo("south")),
        kpi=np.stack(kpis),
        spend=np.stack(spends),
        control_values=np.stack(prices)[:, :, None],
        filled=np.zeros((2, 120), dtype=bool),
    )
    spec = RidgeSpec(tuple(media), (Control("price"),), 0.1)
    for geo, kpi, s, price in zip(two.geos, kpis, spends, prices, strict=True):
        alone = fit_ridge(panel_of(kpi, s, {"price": price}), NATIONAL, spec)
        fit = fit_ridge(two, geo, spec)
        assert fit.coefficients == alone.coefficients
        assert fit.spend == alone.spend


TV = RidgeSpec((Media("tv", Geometric(0.1), Hill(1.0, 0.5)),), (), 1.0)


@pytest.mark.parametrize(
    ("geo", "spec", "window", "match"),
    [
        (Geo("north"), None, None, "no geo"),
        (NATIONAL, TV, None, "no channel or control"),
        (NATIONAL, None, ("2024-01-08", "2023-01-01"), "run forward"),
        (NATIONAL, None, ("2024-01-09", "2024-06-03"), "run forward"),
    ],
)
def test_what_it_cannot_fit_is_refused(
    geo: Geo, spec: RidgeSpec | None, window: tuple[str, str] | None, match: str
) -> None:
    spend, media, _ = world()
    kpi = 1000.0 + spend @ np.array([1.0, 0.5, 0.2])
    dates = None if window is None else (np.datetime64(window[0]), np.datetime64(window[1]))
    with pytest.raises(ValueError, match=match):
        fit_ridge(panel_of(kpi, spend), geo, spec or RidgeSpec(tuple(media), (), 1.0), dates)


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: Control("price", lower=0.5), "lower <= 0 <= upper"),
        (lambda: RidgeSpec((), (), 1.0), "at least one channel"),
        (
            lambda: RidgeSpec(
                (Media("c0", Geometric(0.1), Hill(1.0, 0.5)),), (Control("c0"),), 1.0
            ),
            "more than once",
        ),
        (
            lambda: RidgeSpec((Media("c0", Geometric(0.1), Hill(1.0, 0.5)),), (), 0.0),
            "positive and finite",
        ),
        (
            lambda: RidgeSpec(
                (Media("c0", Geometric(0.1), Hill(1.0, 0.5)),),
                (Control("trend"),),
                1.0,
                LinearTrend(),
            ),
            r"\['trend'\] more than once",
        ),
        (lambda: LinearTrend(-1), "at least 0"),
        (lambda: FourierSeason(2.0, 1), "period above 2"),
        (lambda: FourierSeason(math.inf, 1), "period above 2"),
        (lambda: FourierSeason(52.18, 0), "harmonics from 1"),
        (lambda: FourierSeason(52.0, 26), "below half of it"),
    ],
)
def test_a_model_that_cannot_be_is_refused(make, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        make()


def test_a_fit_calibrated_by_geo_tests_is_the_joint_least_squares_of_the_kpi_and_the_tests() -> (
    None
):
    panel, media, _, tests, reads = world_and_tests(test_noise=10.0)
    spec = RidgeSpec(tuple(media), (), 0.05)
    plain = fit_ridge(panel, NATIONAL, spec)
    fit = fit_ridge(panel, NATIONAL, spec, tests=tests)
    assert plain.calibration is None
    assert fit.calibration is not None
    np.testing.assert_array_equal(fit.design, plain.design)
    # in the columns' own units: the KPI's rows, the ridge's, and each test period's at the tests'
    # weight, all over the fitted periods' number
    x, y = fit.design[fit.fitted], fit.y[fit.fitted]
    n, p = x.shape
    share = math.sqrt(fit.calibration.weight / n)
    rows = [
        np.column_stack([np.ones(n), x]) / math.sqrt(n),
        np.column_stack([np.zeros(p), math.sqrt(spec.penalty) * np.diag(x.std(axis=0))]),
    ]
    targets = [y / math.sqrt(n), np.zeros(p)]
    for j, m in enumerate(media):
        for gap, observed in reads.get(m.channel, []):
            block = np.zeros((gap.size, p + 1))
            block[:, 1 + j] = gap
            rows.append(share * block)
            targets.append(share * observed)
    joint = np.linalg.lstsq(np.vstack(rows), np.concatenate(targets), rcond=None)[0]
    assert (joint[1:] > 0.0).all()
    assert not any(fit.bound.values())
    assert fit.intercept == pytest.approx(joint[0], rel=1e-10)
    np.testing.assert_allclose([fit.coefficients[m.channel] for m in media], joint[1:], rtol=1e-10)


def test_a_test_period_weighs_as_the_kpi_s_noise_variance_over_the_tests() -> None:
    panel, media, _, tests, reads = world_and_tests(test_noise=10.0)
    spec = RidgeSpec(tuple(media), (), 0.05)
    plain = fit_ridge(panel, NATIONAL, spec)
    fit = fit_ridge(panel, NATIONAL, spec, tests=tests)
    assert fit.calibration is not None
    # the KPI's: the fit without tests, its residuals' squares over the fitted periods less the
    # intercept and the trace of the ridge's hat matrix on the standardised design
    x = plain.design[plain.fitted]
    z = (x - x.mean(axis=0)) / x.std(axis=0)
    n = z.shape[0]
    hat = z @ np.linalg.solve(z.T @ z + n * spec.penalty * np.eye(z.shape[1]), z.T)
    residual = (plain.y - plain.estimated)[plain.fitted]
    kpi = residual @ residual / (n - 1.0 - np.trace(hat))
    # the tests': each tested channel's coefficient by least squares on its gaps, a degree of
    # freedom off for each
    estimates, squares, periods = {}, 0.0, 0
    for channel, read in reads.items():
        gap, observed = (np.concatenate(part) for part in zip(*read, strict=True))
        estimates[channel] = np.linalg.lstsq(gap[:, None], observed, rcond=None)[0][0]
        squares += np.sum((observed - estimates[channel] * gap) ** 2)
        periods += gap.size
    assert fit.calibration.estimates == pytest.approx(estimates, rel=1e-12)
    tests_noise = squares / (periods - len(reads))
    assert fit.calibration.weight == pytest.approx(kpi / tests_noise, rel=1e-10)


def test_tests_of_little_noise_hold_their_channels_at_the_truth_a_ridge_shrinks() -> None:
    panel, media, beta, tests, _ = world_and_tests(test_noise=1e-6)
    spec = RidgeSpec(tuple(media), (), 0.5)
    plain = fit_ridge(panel, NATIONAL, spec)
    fit = fit_ridge(panel, NATIONAL, spec, tests=tests)
    assert fit.calibration is not None
    assert set(fit.calibration.estimates) == {"c0", "c2"}
    for j in (0, 2):
        channel = media[j].channel
        assert abs(plain.coefficients[channel] / beta[j] - 1.0) > 0.05
        assert fit.calibration.estimates[channel] == pytest.approx(beta[j], rel=1e-7)
        assert fit.coefficients[channel] == pytest.approx(beta[j], rel=1e-7)


def test_a_calibrated_fit_moves_with_the_kpi_s_unit_and_not_with_the_currency() -> None:
    panel, media, _, tests, _ = world_and_tests(test_noise=10.0)
    spec = RidgeSpec(tuple(media), (), 0.05)
    base = fit_ridge(panel, NATIONAL, spec, tests=tests)
    moved = fit_ridge(
        dataclasses.replace(panel, kpi=3.0 * panel.kpi, spend=1e-3 * panel.spend),
        NATIONAL,
        spec,
        tests={
            channel: [
                LiftTest(
                    *(
                        GeoArm(1e-3 * arm.spend, 3.0 * arm.outcome, arm.share)
                        for arm in (test.treated, test.control)
                    )
                )
                for test in channel_tests
            ]
            for channel, channel_tests in tests.items()
        },
    )
    assert base.calibration is not None
    assert moved.calibration is not None
    for m in media:
        assert moved.coefficients[m.channel] == pytest.approx(
            3.0 * base.coefficients[m.channel], rel=1e-10
        )
    assert moved.intercept == pytest.approx(3.0 * base.intercept, rel=1e-10)
    assert moved.calibration.weight == pytest.approx(base.calibration.weight, rel=1e-10)
    assert moved.calibration.estimates == pytest.approx(
        {channel: 3.0 * value for channel, value in base.calibration.estimates.items()}, rel=1e-10
    )


def _one_period(test: LiftTest) -> LiftTest:
    """The test read in its last period alone, 8 weeks after its groups' spend parted."""
    return LiftTest(
        *(GeoArm(arm.spend, arm.outcome[-1:], arm.share) for arm in (test.treated, test.control))
    )


def _no_gap(test: LiftTest) -> LiftTest:
    """The test with its treated group's outcome the control's: its gaps' estimate is 0, exactly."""
    treated, control = test.treated, test.control
    return LiftTest(GeoArm(treated.spend, control.outcome, control.share), control)


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda tests: {"c9": tests["c0"]}, r"does not have, \['c9'\]"),
        (lambda tests: {"c0": []}, r"or none, \['c0'\]"),
        (
            lambda tests: {"c0": [LiftTest(tests["c0"][1].control, tests["c0"][1].control)]},
            "move no saturated spend",
        ),
        (lambda tests: {"c0": [_one_period(tests["c0"][0])]}, "leave none past"),
        (lambda tests: {"c0": [_no_gap(tests["c0"][1])]}, "fit their estimates exactly"),
    ],
)
def test_tests_it_cannot_read_are_refused(edit, match: str) -> None:
    panel, media, _, tests, _ = world_and_tests(test_noise=10.0)
    with pytest.raises(ValueError, match=match):
        fit_ridge(panel, NATIONAL, RidgeSpec(tuple(media), (), 0.05), tests=edit(tests))


def test_a_kpi_its_model_interpolates_leaves_no_noise_to_weigh_tests_by() -> None:
    panel, media, _, tests, _ = world_and_tests(test_noise=10.0)
    window = (panel.dates[-4], panel.dates[-1])
    with pytest.raises(ValueError, match="no degree of freedom"):
        fit_ridge(panel, NATIONAL, RidgeSpec(tuple(media), (), 1e-30), window, tests)
