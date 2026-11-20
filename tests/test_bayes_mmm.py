"""The Bayesian media-mix model: its log density is Stan's up to a constant, and its posterior
Stan's within Monte Carlo error, on the worlds CmdStan 2.38 recorded with their geo tests and
lift estimates; a channel's contributions over the window are its return times its spend; the
planner's curves are the model's; units move only what they should; and what it cannot fit is
refused."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from chc import response
from chc.lift import GeoArm, LiftTest
from scipy import stats

from mixdyn import Cadence, Channel, Geo, MediaPanel, Revenue
from mixdyn.calibration import LiftEstimate
from mixdyn.models.bayes_mmm import (
    BayesFit,
    BayesMedia,
    BayesSpec,
    HillCurve,
    LogNormal,
    Normal,
    PathfinderStart,
    QRCoordinates,
    RoiPrior,
    Start,
    StudentT,
    TanhCurve,
    UniformStart,
    _initial,
    _Mapped,
    _summary,
    bayes_model,
    fit_bayes,
)
from mixdyn.models.ridge_mmm import Control, FourierSeason, LinearTrend

STAN = json.loads(
    (Path(__file__).parent / "fixtures" / "oracles" / "bayes_mmm_stan.json").read_text()
)
NATIONAL = Geo("national")
CHANNELS = ("tv", "search", "social")


def stan_world(
    name: str,
) -> tuple[MediaPanel, dict[str, list[LiftTest]], dict[str, list[LiftEstimate]], BayesSpec]:
    """The harness's world: its panel, geo tests and lift estimates as the fixture records them,
    its model as the harness states it."""
    case = STAN["cases"][name]
    kpi = np.array(case["panel"]["kpi"])
    panel = MediaPanel(
        dates=np.datetime64("2023-01-02") + np.timedelta64(7, "D") * np.arange(kpi.size),
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=tuple(Channel(c) for c in CHANNELS),
        controls=("promotion", "price"),
        kpi=kpi[None, :],
        kpi_unit=Revenue(),
        spend=np.array(case["panel"]["spend"])[None],
        currency="EUR",
        control_values=np.array(case["panel"]["controls"])[None],
        filled=np.zeros((1, kpi.size), dtype=bool),
    )
    tests = {
        channel: [
            LiftTest(
                *(
                    GeoArm(np.array(t[arm]["spend"]), np.array(t[arm]["outcome"]), t[arm]["share"])
                    for arm in ("treated", "control")
                )
            )
            for t in recorded
        ]
        for channel, recorded in case["tests"].items()
    }
    estimates = {
        channel: [
            LiftEstimate(e["spend"], e["planned"], e["readout"], e["lift"], e["se"], e["share"])
            for e in recorded
        ]
        for channel, recorded in case["estimates"].items()
    }
    spec = BayesSpec(
        tuple(BayesMedia(c) for c in CHANNELS),
        (Control("promotion", lower=0.0), Control("price")),
        LinearTrend(1),
        FourierSeason(52.18, 2),
        {"Normal": Normal(), "StudentT": StudentT(), "LogNormal": LogNormal()}[case["likelihood"]],
        {"HillCurve": HillCurve(), "TanhCurve": TanhCurve()}[case["saturation"]],
    )
    return panel, tests, estimates, spec


def summary(x: np.ndarray, batches: int) -> tuple[np.ndarray, ...]:
    """``x`` is ``(chains, draws, quantities)``: each quantity's mean and standard deviation, and
    their standard errors from ``batches`` batch means a chain, as the harness reads Stan's."""
    chains, draws, k = x.shape
    size = draws // batches
    mean, sd = x.mean(axis=(0, 1)), x.std(axis=(0, 1))
    cut = x[:, : size * batches].reshape(chains * batches, size, k)
    n = math.sqrt(chains * batches)
    squares = ((cut - mean) ** 2).mean(axis=1)
    return (
        mean,
        sd,
        cut.mean(axis=1).std(axis=0, ddof=1) / n,
        squares.std(axis=0, ddof=1) / n / (2.0 * sd),
    )


def standardised(
    x: np.ndarray, stan: dict[str, Any], provenance: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """``x`` is ``(chains, draws, quantities)``: each quantity's mean and standard deviation less
    Stan's, over the two's joint Monte Carlo error. Each error is from batch means, and ours is
    never below Stan's at its draws scaled to ours: a run that misses a heavy tail understates its
    standard deviation and that deviation's error together, so its own error is no measure of how
    far off it is."""
    chains, draws, _ = x.shape
    mean, sd, mean_se, sd_se = summary(x, provenance["batches"])
    scale = math.sqrt(provenance["chains"] * provenance["draws"] / (chains * draws))
    mean_se = np.maximum(mean_se, scale * np.asarray(stan["mean_se"]))
    sd_se = np.maximum(sd_se, scale * np.asarray(stan["sd_se"]))
    return (
        (mean - stan["mean"]) / np.hypot(mean_se, stan["mean_se"]),
        (sd - stan["sd"]) / np.hypot(sd_se, stan["sd_se"]),
    )


def bound(cases: list[dict[str, Any]], provenance: dict[str, Any]) -> float:
    """The largest standardised difference a posterior test allows over ``cases``, each a mean and
    a standard deviation of every quantity: Bonferroni over them all, each difference a t with the
    batch means' degrees of freedom, so that a sampler drawing Stan's posterior fails the test once
    in a thousand runs at most."""
    compared = sum(
        2 * np.size(moments["mean"])
        for case in cases
        for moments in [*case["posterior"].values(), case["log_lik"]]
    )
    batches = provenance["chains"] * provenance["batches"]
    return float(stats.t.isf(0.001 / (2 * compared), batches - 1))


@pytest.mark.parametrize("name", sorted(STAN["cases"]))
def test_the_log_density_is_stan_s_up_to_a_constant(name: str) -> None:
    """At 25 points uniform in ``[-2, 2]`` on Stan's sampling coordinates, where Stan starts its
    chains, each read with the intercept's coordinate moved to the level, a map whose Jacobian is
    1: the log density, with every transform's Jacobian, differs from Stan's by one constant, the
    normalising terms Stan's sampling statements drop, and its gradient by the chain rule is
    Stan's."""
    case = STAN["cases"][name]
    panel, tests, estimates, spec = stan_world(name)
    model = bayes_model(panel, NATIONAL, spec, tests=tests, estimates=estimates)
    points = np.array(case["points"])
    assert len(model.coordinates) == model.dimension == points.shape[1]
    stan_s = jax.jit(jax.value_and_grad(lambda t: model._log_density(model._with_level(t))))
    ours = [(float(v), np.asarray(g)) for v, g in (stan_s(jnp.asarray(p)) for p in points)]
    offset = np.array([value for value, _ in ours]) - np.array(case["log_prob"])
    assert np.ptp(offset) <= 1e-12 * np.abs(case["log_prob"]).max()
    np.testing.assert_allclose(
        np.array([gradient for _, gradient in ours]),
        case["gradient"],
        rtol=1e-11,
        atol=1e-12 * np.abs(case["gradient"]).max(),
    )


@pytest.mark.parametrize("name", sorted(STAN["cases"]))
def test_the_level_is_the_intercept_and_the_channels_mean_contribution(name: str) -> None:
    """At Stan's 25 points, read as the engine's: the level coordinate is the intercept plus the
    channels' contribution averaged over the window's periods, as the model's own terms sum it."""
    panel, tests, estimates, spec = stan_world(name)
    model = bayes_model(panel, NATIONAL, spec, tests=tests, estimates=estimates)
    points = np.array(STAN["cases"][name]["points"])
    parts = []
    for point in points:
        p, _ = model._constrained(jnp.asarray(point))
        saturated, beta = model._terms(p)
        parts.append((float(p["intercept"][0]), float(jnp.mean(saturated @ beta))))
    intercept, media = np.array(parts).T
    assert media.min() > 0.0
    np.testing.assert_allclose(
        intercept + media,
        points[:, model.coordinates.index("level")],
        rtol=0.0,
        atol=1e-12 * np.max(np.abs(intercept) + media),
    )


POSTERIOR = [
    *((name, UniformStart()) for name in sorted(STAN["cases"])),
    ("normal", PathfinderStart()),
]
BOUND = bound([STAN["cases"][name] for name, _ in POSTERIOR], STAN["provenance"])


# Named, not numbered: CI runs each by its node id, which a world added must not shift.
@pytest.mark.parametrize(
    ("name", "start"),
    POSTERIOR,
    ids=[
        f"{name}-{type(start).__name__.removesuffix('Start').lower()}" for name, start in POSTERIOR
    ],
)
def test_the_posterior_is_stan_s(name: str, start: Start) -> None:
    """Four chains of 1000 draws after 1000 of warm-up, from uniform starts on every world and
    from Pathfinder's on the normal one: each parameter's posterior mean and standard deviation, in
    the panel's units, and each observation's log-likelihood's, within ``BOUND`` joint Monte Carlo
    standard errors of those of Stan's 40 000 draws, as `standardised` reads them, each error from
    batch means of ten batches a chain; no divergent transition, every R-hat below 1.01 and every
    chain's E-BFMI above 0.3. On the tanh world the third channel's coefficient has a heavy tail:
    Stan's draws know its standard deviation, 115, only to 10."""
    case = STAN["cases"][name]
    panel, tests, estimates, spec = stan_world(name)
    fit = fit_bayes(panel, NATIONAL, spec, seed=1, tests=tests, estimates=estimates, start=start)
    assert set(fit.draws) == set(case["posterior"])
    unit = bayes_model(panel, NATIONAL, spec, tests=tests, estimates=estimates).kpi_unit
    ours = {**fit.draws, "log_lik": fit.log_likelihood + math.log(unit)}
    for key, stan in {**case["posterior"], "log_lik": case["log_lik"]}.items():
        mean, sd = standardised(ours[key].reshape(4, 1000, -1), stan, STAN["provenance"])
        assert np.all(np.abs(mean) <= BOUND), key
        assert np.all(np.abs(sd) <= BOUND), key
    for key in case["posterior"]:
        assert np.all(fit.diagnostics[key].rhat < 1.01), key
    assert not fit.divergent.any()
    assert np.all(fit.e_bfmi > 0.3)


def test_on_qr_coordinates_the_posterior_is_stan_s() -> None:
    """On the normal world from Pathfinder's starts, the sampler on the linear block's QR
    coordinates draws Stan's posterior as `test_the_posterior_is_stan_s` reads it, its own
    comparisons Bonferroni's at one failure in a thousand runs, back on the model's coordinates."""
    case = STAN["cases"]["normal"]
    panel, tests, estimates, spec = stan_world("normal")
    fit = fit_bayes(
        panel, NATIONAL, spec, seed=1, tests=tests, estimates=estimates, coordinates=QRCoordinates()
    )
    unit = bayes_model(panel, NATIONAL, spec, tests=tests, estimates=estimates).kpi_unit
    ours = {**fit.draws, "log_lik": fit.log_likelihood + math.log(unit)}
    limit = bound([case], STAN["provenance"])
    for key, stan in {**case["posterior"], "log_lik": case["log_lik"]}.items():
        mean, sd = standardised(ours[key].reshape(4, 1000, -1), stan, STAN["provenance"])
        assert np.all(np.abs(mean) <= limit), key
        assert np.all(np.abs(sd) <= limit), key
    for key in case["posterior"]:
        assert np.all(fit.diagnostics[key].rhat < 1.01), key
    assert not fit.divergent.any()
    assert np.all(fit.e_bfmi > 0.3)


@pytest.mark.parametrize("name", sorted(STAN["cases"]))
def test_qr_coordinates_whiten_the_linear_block_and_leave_the_rest(name: str) -> None:
    """The map from QR coordinates to the model's is `R*^-1` on the level, the calendar and the
    free controls, which turns their columns over the observed periods into orthogonal ones of
    squared norm ``n - 1``, and the identity elsewhere, a control held to its sign among them; the
    density on QR coordinates is the model's at the mapped point."""
    panel, tests, estimates, spec = stan_world(name)
    held = (Control(spec.controls[0].column, lower=0.0), *spec.controls[1:])
    spec = dataclasses.replace(spec, controls=held)
    model = bayes_model(panel, NATIONAL, spec, tests=tests, estimates=estimates)
    matrix = model.qr_map()
    names = model.coordinates
    level = names.index("level")
    calendar = [i for i, n in enumerate(names) if n.startswith("calendar[")]
    controls = [i for i, n in enumerate(names) if n.startswith("control[")]
    block = [level, *calendar, *controls[1:]]
    rest = [i for i in range(model.dimension) if i not in block]
    assert calendar
    assert len(controls) > 1
    np.testing.assert_array_equal(matrix[np.ix_(rest, rest)], np.eye(len(rest)))
    np.testing.assert_array_equal(matrix[np.ix_(rest, block)], 0.0)
    np.testing.assert_array_equal(matrix[np.ix_(block, rest)], 0.0)
    observed = model.observed
    columns = np.column_stack(
        [np.ones(int(observed.sum())), model.calendar[observed], model.controls[observed][:, 1:]]
    )
    whitened = columns @ matrix[np.ix_(block, block)]
    n = columns.shape[0]
    np.testing.assert_allclose(whitened.T @ whitened, (n - 1) * np.eye(len(block)), atol=1e-9 * n)
    mapped = _Mapped(model, matrix)
    points = np.random.default_rng(3).uniform(-2.0, 2.0, (5, model.dimension))
    for u in points:
        assert float(mapped._log_density(jnp.asarray(u))) == float(
            model._log_density(jnp.asarray(matrix @ u))
        )


def test_pathfinder_starts_each_chain_in_the_posterior_s_bulk() -> None:
    """On the normal world every parameter of each of four starts, in the panel's units, lies
    within 4 posterior standard deviations of Stan's posterior mean; uniform starts lie over 100
    away."""
    case = STAN["cases"]["normal"]
    panel, tests, estimates, spec = stan_world("normal")
    model = bayes_model(panel, NATIONAL, spec, tests=tests, estimates=estimates)
    for start, low, high in ((PathfinderStart(), 0.0, 4.0), (UniformStart(), 100.0, math.inf)):
        raw, _, _ = _summary(model, _initial(model, start, 4, 1))
        worst = max(
            np.max(np.abs(raw[key].reshape(4, -1) - np.ravel(stan["mean"])) / np.ravel(stan["sd"]))
            for key, stan in case["posterior"].items()
        )
        assert low <= worst < high, start


@pytest.mark.parametrize("name", sorted(STAN["cases"]))
def test_each_observation_s_log_likelihood_is_stan_s(name: str) -> None:
    """At 25 of Stan's draws, each of the KPI's weeks, then each test's readout periods, then each
    lift estimate, every normalising term in, as Stan's generated quantities compute it on the
    model's scales: in the panel's units, less the log of the KPI's unit."""
    recorded = STAN["cases"][name]["log_lik_draws"]
    panel, tests, estimates, spec = stan_world(name)
    model = bayes_model(panel, NATIONAL, spec, tests=tests, estimates=estimates)
    theta = np.asarray(model._with_level(jnp.asarray(recorded["theta"])))
    np.testing.assert_allclose(
        model.log_likelihood(theta) + math.log(model.kpi_unit),
        recorded["log_lik"],
        rtol=1e-11,
        atol=1e-11,
    )


@pytest.fixture(scope="module", params=["normal", "tanh"])
def short_fit(request: pytest.FixtureRequest) -> tuple[MediaPanel, BayesFit]:
    """Over the panel's last 80 weeks, its kernels reading the spend before them."""
    panel, tests, estimates, spec = stan_world(request.param)
    window = (panel.dates[24], panel.dates[-1])
    return panel, fit_bayes(
        panel,
        NATIONAL,
        spec,
        seed=3,
        window=window,
        tests=tests,
        estimates=estimates,
        chains=2,
        warmup=150,
        draws=100,
    )


def test_a_channel_s_contributions_over_the_window_are_its_return_times_its_spend(
    short_fit: tuple[MediaPanel, BayesFit],
) -> None:
    """Whatever its carryover and curve, and the intercept the rest of the fitted mean."""
    panel, fit = short_fit
    inside = np.isin(panel.dates, fit.dates)
    assert inside.sum() == 80
    for j, media in enumerate(fit.spec.media):
        assert fit.contributions[media.channel].sum() == pytest.approx(
            fit.draws["roi"][..., j].mean() * panel.spend[0, inside, j].sum(), rel=1e-12
        )
    rest = fit.estimated - sum(fit.contributions.values())
    np.testing.assert_allclose(rest, fit.draws["intercept"].mean(), rtol=1e-12)


def test_the_log_likelihood_reads_each_observation_at_each_draw(
    short_fit: tuple[MediaPanel, BayesFit],
) -> None:
    """Each of the two chains' 100 draws, at the window's 80 weeks, the tests' 12 readout periods
    and the 3 lift estimates."""
    _, fit = short_fit
    assert fit.log_likelihood.shape == (2, 100, 80 + 12 + 3)
    assert np.all(np.isfinite(fit.log_likelihood))


def test_a_model_of_its_channels_alone_has_no_calendar_or_control_draws() -> None:
    """As a `tanh` has no slope's; and its fitted mean is its intercept and its channels."""
    panel, _, _, spec = stan_world("normal")
    alone = dataclasses.replace(spec, controls=(), trend=None, season=None)
    fit = fit_bayes(panel, NATIONAL, alone, seed=4, chains=2, warmup=50, draws=20)
    assert "calendar" not in fit.draws
    assert "control" not in fit.draws
    assert set(fit.diagnostics) == set(fit.draws)
    rest = fit.estimated - sum(fit.contributions.values())
    np.testing.assert_allclose(rest, fit.draws["intercept"].mean(), rtol=1e-12)


def test_the_planner_s_channels_are_the_model_s_curves_at_the_posterior_medians(
    short_fit: tuple[MediaPanel, BayesFit],
) -> None:
    """A Hill at its median slope, or `tanh` at half its ceiling where the adstock is the
    half-saturation point, which then has no slope to draw."""
    panel, fit = short_fit
    tanh = isinstance(fit.spec.saturation, TanhCurve)
    assert ("slope" in fit.draws) is not tanh
    for j, (name, planned) in enumerate(fit.channels().items()):
        retention, half, coefficient = (
            float(np.median(fit.draws[key][..., j]))
            for key in ("retention", "half_saturation", "coefficient")
        )
        spend = panel.spend[0, :, j]
        weights = retention ** np.arange(fit.length)
        adstock = np.convolve(spend, weights / weights.sum())[: spend.size]
        if tanh:
            curve = np.tanh(math.atanh(0.5) * adstock / half)
        else:
            slope = float(np.median(fit.draws["slope"][..., j]))
            curve = adstock**slope / (adstock**slope + half**slope)
        np.testing.assert_allclose(
            np.asarray(planned(spend)), coefficient * curve, rtol=1e-12, err_msg=name
        )


def test_each_draw_s_channels_contribute_its_return_times_the_window_s_spend(
    short_fit: tuple[MediaPanel, BayesFit],
) -> None:
    """The identity the model is parametrised by, at every draw rather than at the medians: each
    channel as `chc.response` reads it there, carried in from the spend before the window,
    contributes over the window the draw's return times the window's spend, to `1e-12`."""
    panel, fit = short_fit
    roi = fit.draws["roi"].reshape(-1, len(fit.spec.media))
    window = slice(int(np.isin(panel.dates, fit.dates).argmax()), None)
    for i, channels in enumerate(fit.channel_draws(roi.shape[0])):
        for j, (name, channel) in enumerate(channels.items()):
            spend = panel.spend[0, :, j]
            assert response.contribution(channel, spend, window) == pytest.approx(
                roi[i, j] * spend[window].sum(), rel=1e-12
            ), name


def test_a_posterior_s_channels_are_read_at_draws_evenly_spaced_from_the_first_to_the_last(
    short_fit: tuple[MediaPanel, BayesFit],
) -> None:
    """Seven of two chains' 200 draws, in chain order, each channel at that draw's decay; a
    count of no draws or of more than the fit has is refused."""
    _, fit = short_fit
    retention = fit.draws["retention"].reshape(-1, len(fit.spec.media))
    assert retention.shape[0] == 200
    for channels, row in zip(fit.channel_draws(7), (0, 33, 66, 100, 133, 166, 199), strict=True):
        for j, channel in enumerate(channels.values()):
            weights = np.asarray(channel.kernel.weights())
            assert weights[1] / weights[0] == pytest.approx(retention[row, j], rel=1e-12, abs=0.0)
    for count in (0, 201):
        with pytest.raises(ValueError, match="count must be from 1 to the fit's 200 draws"):
            fit.channel_draws(count)


def _scaled(tests: dict[str, list[LiftTest]], factor: float) -> dict[str, list[LiftTest]]:
    return {
        c: [
            LiftTest(
                *(
                    GeoArm(arm.spend * factor, arm.outcome * factor, arm.share)
                    for arm in (t.treated, t.control)
                )
            )
            for t in ts
        ]
        for c, ts in tests.items()
    }


def _rescaled(
    estimates: dict[str, list[LiftEstimate]], factor: float, share: float = 1.0
) -> dict[str, list[LiftEstimate]]:
    """Each estimate's spend, planned spend, lift and error `factor` times as large, its share
    `share` times."""
    return {
        c: [
            dataclasses.replace(
                e,
                spend=e.spend * factor,
                planned=e.planned * factor,
                lift=e.lift * factor,
                se=e.se * factor,
                share=e.share * share,
            )
            for e in es
        ]
        for c, es in estimates.items()
    }


def test_units_move_only_what_they_should() -> None:
    """The KPI and every spend ten times as large, the tests' and the lift estimates' too: the
    model reads the same data,
    so a point has the same log density, and the same return, carryover and slope, while what is in
    the KPI's or the currency's units moves ten times, and each observation's log density in them
    by the log of a tenth. A control ten times as large moves only its
    own coefficient, a tenth as large; one moved by a constant moves only the intercept, by the
    control's coefficient times the move."""
    panel, tests, estimates, spec = stan_world("student")
    model = bayes_model(panel, NATIONAL, spec, tests=tests, estimates=estimates)
    larger = bayes_model(
        dataclasses.replace(panel, kpi=panel.kpi * 10.0, spend=panel.spend * 10.0),
        NATIONAL,
        spec,
        tests=_scaled(tests, 10.0),
        estimates=_rescaled(estimates, 10.0),
    )
    theta = np.random.default_rng(4).uniform(-2.0, 2.0, (3, model.dimension))
    for point in theta:
        value, gradient = model.log_density(point)
        assert larger.log_density(point)[0] == pytest.approx(value, rel=1e-12)
        np.testing.assert_allclose(larger.log_density(point)[1], gradient, rtol=1e-10)
    ours, theirs = model.parameters(theta), larger.parameters(theta)
    for key in ("roi", "retention", "slope", "nu"):
        np.testing.assert_allclose(theirs[key], ours[key], rtol=1e-12, err_msg=key)
    for key in ("half_saturation", "coefficient", "intercept", "calendar", "control", "sigma"):
        np.testing.assert_allclose(theirs[key], 10.0 * ours[key], rtol=1e-12, err_msg=key)
    np.testing.assert_allclose(theirs["test_sigma"], 10.0 * ours["test_sigma"], rtol=1e-12)
    np.testing.assert_allclose(
        larger.log_likelihood(theta), model.log_likelihood(theta) - math.log(10.0), rtol=1e-12
    )
    values = np.array(panel.control_values)
    values[..., 1] *= 10.0
    priced = bayes_model(
        dataclasses.replace(panel, control_values=values), NATIONAL, spec, tests=tests
    ).parameters(theta)
    np.testing.assert_allclose(priced["control"][:, 1], ours["control"][:, 1] / 10.0, rtol=1e-12)
    np.testing.assert_allclose(priced["control"][:, 0], ours["control"][:, 0], rtol=1e-12)
    for key in ("intercept", "coefficient", "sigma"):
        np.testing.assert_allclose(priced[key], ours[key], rtol=1e-12, err_msg=key)
    values = np.array(panel.control_values)
    values[..., 1] += 100.0
    moved = bayes_model(
        dataclasses.replace(panel, control_values=values), NATIONAL, spec, tests=tests
    ).parameters(theta)
    np.testing.assert_allclose(
        moved["intercept"], ours["intercept"] - 100.0 * ours["control"][:, 1], rtol=1e-10
    )
    for key in ("control", "coefficient", "sigma"):
        np.testing.assert_allclose(moved[key], ours[key], rtol=1e-10, err_msg=key)


def test_a_group_s_share_moves_nothing_the_market_sees() -> None:
    """Each test's control group three times the share of the market, with three times its spend
    and outcome, and each lift estimate's group half its share, with half its spend, lift and
    error: the market sees the same tests, so a point has the same log density."""
    panel, tests, estimates, spec = stan_world("normal")
    larger = {
        c: [
            LiftTest(t.treated, GeoArm(3.0 * t.control.spend, 3.0 * t.control.outcome, 0.3))
            for t in ts
        ]
        for c, ts in tests.items()
    }
    assert all(t.control.share == 0.1 for ts in tests.values() for t in ts)
    model = bayes_model(panel, NATIONAL, spec, tests=tests, estimates=estimates)
    other = bayes_model(
        panel, NATIONAL, spec, tests=larger, estimates=_rescaled(estimates, 0.5, share=0.5)
    )
    for point in np.random.default_rng(8).uniform(-2.0, 2.0, (3, model.dimension)):
        assert other.log_density(point)[0] == pytest.approx(model.log_density(point)[0], rel=1e-12)


def test_an_estimate_in_the_model_s_geo_is_one_in_its_whole_market() -> None:
    """Naming the geo the model reads names the market the model reads: the same log density as
    naming none."""
    panel, tests, estimates, spec = stan_world("normal")
    named = {
        c: [dataclasses.replace(e, geos=frozenset({NATIONAL})) for e in es]
        for c, es in estimates.items()
    }
    model = bayes_model(panel, NATIONAL, spec, tests=tests, estimates=estimates)
    other = bayes_model(panel, NATIONAL, spec, tests=tests, estimates=named)
    for point in np.random.default_rng(10).uniform(-2.0, 2.0, (3, model.dimension)):
        assert other.log_density(point)[0] == model.log_density(point)[0]


def test_a_week_without_a_kpi_is_not_fitted() -> None:
    panel, tests, _, spec = stan_world("normal")
    kpi = np.array(panel.kpi)
    kpi[0, [3, 40]] = np.nan
    model = bayes_model(dataclasses.replace(panel, kpi=kpi), NATIONAL, spec, tests=tests)
    assert np.flatnonzero(~model.observed).tolist() == [3, 40]
    np.testing.assert_allclose(
        model.y * model.kpi_unit, np.delete(panel.kpi[0], [3, 40]), rtol=1e-15
    )
    assert np.isfinite(model.log_density(np.zeros(model.dimension))[0])


def test_a_control_held_to_a_sign_keeps_it() -> None:
    panel, tests, _, spec = stan_world("normal")
    held = (Control("promotion", lower=0.0), Control("price", upper=0.0))
    model = bayes_model(panel, NATIONAL, dataclasses.replace(spec, controls=held), tests=tests)
    control = model.parameters(np.random.default_rng(6).uniform(-2.0, 2.0, (50, model.dimension)))[
        "control"
    ]
    assert (control[:, 0] > 0.0).all()
    assert (control[:, 1] < 0.0).all()


def test_a_test_longer_than_the_panel_runs_the_kernels_over_its_length() -> None:
    """The panel's spend is preceded by none, and the window stays the panel's."""
    panel, tests, _, spec = stan_world("normal")
    test = tests["tv"][0]
    longer = LiftTest(
        *(
            GeoArm(np.r_[np.full(120, arm.spend[0]), arm.spend], arm.outcome, arm.share)
            for arm in (test.treated, test.control)
        )
    )
    model = bayes_model(panel, NATIONAL, spec, tests={"tv": [longer]})
    spend = panel.spend[0] / model.spend_unit
    assert model.lagged.shape == (3, panel.dates.size, 120 + test.treated.spend.size)
    for t in (0, 50, panel.dates.size - 1):
        np.testing.assert_array_equal(model.lagged[:, t, : t + 1], spend[t::-1].T)
        assert not model.lagged[:, t, t + 1 :].any()


def test_a_hill_s_slope_prior_is_the_one_its_curve_states() -> None:
    """Moving the prior moves the log density by the two priors' difference at each log slope,
    and nothing else: the default is LogNormal(0, 0.5)."""
    panel, tests, estimates, spec = stan_world("normal")
    assert HillCurve() == HillCurve(0.0, 0.5)
    model = bayes_model(panel, NATIONAL, spec, tests=tests, estimates=estimates)
    leaning = bayes_model(
        panel,
        NATIONAL,
        dataclasses.replace(spec, saturation=HillCurve(-0.3, 0.4)),
        tests=tests,
        estimates=estimates,
    )
    slopes = [i for i, c in enumerate(model.coordinates) if c.startswith("log_slope[")]
    assert len(slopes) == len(CHANNELS)
    for theta in np.random.default_rng(4).uniform(-2.0, 2.0, (5, model.dimension)):
        log_slope = theta[slopes]
        expected = stats.norm.logpdf(log_slope, -0.3, 0.4) - stats.norm.logpdf(log_slope, 0.0, 0.5)
        moved = leaning.log_density(theta)[0] - model.log_density(theta)[0]
        assert moved == pytest.approx(expected.sum(), rel=1e-9, abs=1e-9)


def test_a_kernel_s_stated_length_cuts_every_series_lags_and_normalises_over_them() -> None:
    """Each row of lags keeps its first 6, lag 0 first, the tests' and the estimates' too; a
    channel's adstock is its spend under a geometric kernel of 6 weeks normalised over them; the
    default kernel runs over the panel."""
    panel, tests, estimates, spec = stan_world("normal")
    whole = bayes_model(panel, NATIONAL, spec, tests=tests, estimates=estimates)
    model = bayes_model(
        panel, NATIONAL, dataclasses.replace(spec, kernel=6), tests=tests, estimates=estimates
    )
    assert whole.lagged.shape == (3, panel.dates.size, panel.dates.size)
    assert model.lagged.shape == (3, panel.dates.size, 6)
    np.testing.assert_array_equal(model.lagged, whole.lagged[..., :6])
    for cut, kept in zip(model.arms, whole.arms, strict=True):
        np.testing.assert_array_equal(cut.treated, kept.treated[:, :6])
        np.testing.assert_array_equal(cut.control, kept.control[:, :6])
    for cut, kept in zip(model.estimates, whole.estimates, strict=True):
        np.testing.assert_array_equal(cut.spend, kept.spend[:, :6])
        np.testing.assert_array_equal(cut.planned, kept.planned[:, :6])
    p, _ = model._constrained(jnp.zeros(model.dimension))
    retention = float(p["retention"][0])
    weights = retention ** np.arange(6)
    spend = panel.spend[0, :, 0] / model.spend_unit[0]
    adstock = np.convolve(spend, weights / weights.sum())[: spend.size]
    half, slope = float(p["half"][0]), float(p["slope"][0])
    np.testing.assert_allclose(
        np.asarray(model._saturated(p, 0, model.lagged[0])),
        adstock**slope / (adstock**slope + half**slope),
        rtol=1e-12,
    )


def test_a_kernel_longer_than_every_series_carries_no_spend_from_before_the_panel() -> None:
    panel, _, _, spec = stan_world("normal")
    model = bayes_model(panel, NATIONAL, dataclasses.replace(spec, kernel=panel.dates.size + 20))
    whole = bayes_model(panel, NATIONAL, spec)
    assert model.lagged.shape == (3, panel.dates.size, panel.dates.size + 20)
    np.testing.assert_array_equal(model.lagged[..., : panel.dates.size], whole.lagged)
    assert not model.lagged[..., panel.dates.size :].any()


def test_a_fit_s_channels_carry_the_kernel_s_stated_length() -> None:
    panel, tests, estimates, spec = stan_world("tanh")
    fit = fit_bayes(
        panel,
        NATIONAL,
        dataclasses.replace(spec, kernel=6),
        seed=3,
        tests=tests,
        estimates=estimates,
        chains=1,
        warmup=20,
        draws=10,
    )
    assert fit.length == 6
    assert {c.kernel.length for c in fit.channels().values()} == {6}


def test_a_lift_estimate_longer_than_the_panel_runs_the_kernels_over_its_length() -> None:
    panel, _, estimates, spec = stan_world("normal")
    e = estimates["tv"][0]
    longer = dataclasses.replace(
        e,
        spend=np.r_[np.full(120, e.spend[0]), e.spend],
        planned=np.r_[np.full(120, e.planned[0]), e.planned],
    )
    model = bayes_model(panel, NATIONAL, spec, estimates={"tv": [longer]})
    assert model.lagged.shape == (3, panel.dates.size, 120 + e.spend.size)


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: RoiPrior(sigma=0.0), "positive, finite sigma"),
        (lambda: RoiPrior(mu=math.inf), "finite mu"),
        (lambda: BayesSpec(()), "at least one channel"),
        (lambda: BayesSpec((BayesMedia("tv"),), (Control("tv"),)), r"\['tv'\] more than once"),
        (
            lambda: BayesSpec((BayesMedia("tv"),), (Control("price", lower=-1.0),)),
            r"\['price'\] are held to bounds other than a sign",
        ),
        (
            lambda: BayesSpec((BayesMedia("tv"),), (Control("price", lower=0.0, upper=0.0),)),
            "bounds other than a sign",
        ),
        (lambda: HillCurve(slope_scale=0.0), "finite scale above 0, not 0.0 and 0.0"),
        (lambda: HillCurve(slope_location=math.nan), "finite location"),
        (lambda: HillCurve(slope_scale=math.inf), "finite scale above 0"),
        (lambda: BayesSpec((BayesMedia("tv"),), kernel=0), "kernel's length .* not 0"),
        (lambda: BayesSpec((BayesMedia("tv"),), kernel=True), "kernel's length .* not True"),
        (
            lambda: BayesSpec((BayesMedia("tv"),), kernel=6.0),  # ty: ignore[invalid-argument-type]
            "kernel's length .* not 6.0",
        ),
    ],
)
def test_a_model_that_cannot_be_is_refused(make, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        make()


def _dark(panel: MediaPanel) -> MediaPanel:
    spend = np.array(panel.spend)
    spend[0, :10, 2] = 0.0
    return dataclasses.replace(panel, spend=spend)


def _kpi(panel: MediaPanel, kpi: np.ndarray) -> MediaPanel:
    return dataclasses.replace(panel, kpi=np.broadcast_to(kpi, panel.kpi.shape))


def _flat(panel: MediaPanel) -> MediaPanel:
    values = np.array(panel.control_values)
    values[0, :, 1] = 10.0
    return dataclasses.replace(panel, control_values=values)


FIRST_TEN = (np.datetime64("2023-01-02"), np.datetime64("2023-03-06"))


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda c: {"geo": Geo("north")}, "no geo"),
        (
            lambda c: {"spec": dataclasses.replace(c["spec"], media=(BayesMedia("radio"),))},
            r"no channel or control named \['radio'\]",
        ),
        (
            lambda c: {"spec": dataclasses.replace(c["spec"], controls=(Control("weather"),))},
            r"no channel or control named \['weather'\]",
        ),
        (lambda c: {"tests": {"radio": c["tests"]["tv"]}}, r"does not have, \['radio'\]"),
        (lambda c: {"tests": {"tv": []}}, r"or none, \['tv'\]"),
        (
            lambda c: {"estimates": {"radio": c["estimates"]["tv"]}},
            r"estimates name channels the model does not have, \['radio'\]",
        ),
        (lambda c: {"estimates": {"tv": []}}, r"estimates name .* or none, \['tv'\]"),
        (
            lambda c: {
                "estimates": {
                    "tv": [
                        dataclasses.replace(
                            c["estimates"]["tv"][0], geos=frozenset({NATIONAL, Geo("north")})
                        )
                    ]
                }
            },
            r"estimates name geos \['north'\] that the model of 'national' does not read",
        ),
        (lambda c: {"window": FIRST_TEN[::-1]}, "must run forward"),
        (
            lambda c: {"panel": _dark(c["panel"]), "window": FIRST_TEN},
            r"\['social'\] spend nothing",
        ),
        (
            lambda c: {"panel": _kpi(c["panel"], np.r_[1.0, np.full(103, np.nan)])},
            "has 1 observed periods",
        ),
        (lambda c: {"panel": _kpi(c["panel"], -np.ones(104))}, "of mean KPI -1.0"),
        (lambda c: {"panel": _flat(c["panel"])}, r"\['price'\] do not vary"),
        (
            lambda c: {
                "panel": _kpi(c["panel"], np.r_[1.0, 0.0, np.full(102, 2.0)]),
                "spec": dataclasses.replace(c["spec"], likelihood=LogNormal()),
            },
            r"log-normal noise needs a KPI above 0, and the window's periods \[1\] are not",
        ),
    ],
)
def test_what_it_cannot_fit_is_refused(edit, match: str) -> None:
    panel, tests, estimates, spec = stan_world("normal")
    call: dict[str, Any] = {
        "panel": panel,
        "geo": NATIONAL,
        "spec": spec,
        "window": None,
        "tests": tests,
        "estimates": estimates,
    }
    call |= edit(call)
    with pytest.raises(ValueError, match=match):
        bayes_model(**call)


def test_single_precision_is_refused() -> None:
    panel, tests, _, spec = stan_world("normal")
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(ValueError, match="double precision"):
            bayes_model(panel, NATIONAL, spec, tests=tests)
    finally:
        jax.config.update("jax_enable_x64", True)


def _log_normal_world() -> tuple[MediaPanel, dict[str, list[LiftTest]], BayesSpec, np.ndarray]:
    """The normal world under log-normal noise, and five uniform starts in its support."""
    panel, tests, _, spec = stan_world("normal")
    spec = dataclasses.replace(spec, likelihood=LogNormal())
    model = bayes_model(panel, NATIONAL, spec, tests=tests)
    return panel, tests, spec, _initial(model, UniformStart(), 5, 9)


def test_log_normal_noise_is_read_about_the_mean() -> None:
    """Each week's log density in the panel's units is scipy's log-normal whose mean is the
    model's, `lognorm(s=sigma, scale=mu exp(-sigma^2 / 2))`, `sigma` the log's scale, reported in
    no unit."""
    panel, tests, spec, points = _log_normal_world()
    model = bayes_model(panel, NATIONAL, spec, tests=tests)
    assert len(points) == 5
    for point in points:
        raw, parts, pointwise = _summary(model, point[None])
        mean = sum(parts.values(), np.zeros(panel.dates.size))
        sigma = float(raw["sigma"][0])
        scale = mean * math.exp(-(sigma**2) / 2.0)
        np.testing.assert_allclose(stats.lognorm.mean(sigma, scale=scale), mean, rtol=1e-12)
        np.testing.assert_allclose(
            pointwise[0, : panel.dates.size],
            stats.lognorm.logpdf(panel.kpi[0], sigma, scale=scale),
            rtol=1e-11,
        )


def test_a_point_whose_mean_is_not_above_zero_has_no_density() -> None:
    """Under log-normal noise, as Stan rejects it, with a finite gradient, so that the sampler
    reads it as a divergence."""
    panel, tests, spec, points = _log_normal_world()
    model = bayes_model(panel, NATIONAL, spec, tests=tests)
    point = points[0].copy()
    point[model.coordinates.index("level")] = -50.0
    value, gradient = model.log_density(point)
    assert value == -math.inf
    assert np.all(np.isfinite(gradient))


def test_log_normal_noise_moves_with_no_unit() -> None:
    """The KPI and every spend ten times as large: the same log density at each point, the same
    `sigma`, and each week's log density in the panel's units moved by the log of a tenth."""
    panel, tests, spec, points = _log_normal_world()
    model = bayes_model(panel, NATIONAL, spec, tests=tests)
    larger = bayes_model(
        dataclasses.replace(panel, kpi=panel.kpi * 10.0, spend=panel.spend * 10.0),
        NATIONAL,
        spec,
        tests=_scaled(tests, 10.0),
    )
    for point in points:
        assert larger.log_density(point)[0] == pytest.approx(model.log_density(point)[0], rel=1e-12)
    np.testing.assert_allclose(
        larger.parameters(points)["sigma"], model.parameters(points)["sigma"], rtol=1e-12
    )
    np.testing.assert_allclose(
        larger.log_likelihood(points), model.log_likelihood(points) - math.log(10.0), rtol=1e-12
    )


def test_a_start_outside_the_support_has_its_intercept_raised() -> None:
    """Under log-normal noise few points uniform in `[-2, 2]` put every week's mean above 0; a
    start that does not has its intercept raised until the least mean is the least KPI observed,
    uniform or Pathfinder's, and every other coordinate kept; a start in the support is kept as
    drawn. Where the noise bounds no mean, the starts are the plain uniform draws."""
    panel, tests, _, spec = stan_world("normal")
    normal = bayes_model(panel, NATIONAL, spec, tests=tests)
    np.testing.assert_array_equal(
        _initial(normal, UniformStart(), 20, 7),
        np.random.default_rng(7).uniform(-2.0, 2.0, (20, normal.dimension)),
    )
    model = bayes_model(
        panel, NATIONAL, dataclasses.replace(spec, likelihood=LogNormal()), tests=tests
    )
    plain = np.random.default_rng(7).uniform(-2.0, 2.0, (20, model.dimension))
    starts = _initial(model, UniformStart(), 20, 7)
    level = model.coordinates.index("level")
    raised = starts[:, level] != plain[:, level]
    assert raised.sum() >= 15
    np.testing.assert_array_equal(np.delete(starts, level, axis=1), np.delete(plain, level, axis=1))
    np.testing.assert_array_equal(starts[~raised], plain[~raised])
    inside = starts[raised][0].copy()
    inside[level] += 1.0
    np.testing.assert_array_equal(model._lift(inside), inside)
    for point in starts[raised]:
        _, parts, _ = _summary(model, point[None])
        least = sum(parts.values(), np.zeros(panel.dates.size))[model.observed].min()
        assert least == pytest.approx(panel.kpi[0, model.observed].min(), rel=1e-9)
    for point in (*starts, *_initial(model, PathfinderStart(), 4, 7)):
        assert math.isfinite(model.log_density(point)[0])
