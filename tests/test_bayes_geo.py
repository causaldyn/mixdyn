"""The Bayesian media-mix model over geos: its log density is Stan's up to a constant, and its
posterior Stan's within Monte Carlo error, on the worlds CmdStan 2.38 recorded with their lift
estimates; a channel's contributions over the window and the geos are its
return times its spend; the planner's curves in a geo are the model's; a channel's geo deviations
sum to 0, so its national effect is its geo effects' geometric mean; a lift estimate is read in
the geos its test ran in, at the spend per head their people saw; units and the geos' order move
only what they should; and what it cannot fit is refused."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest
from chc import response
from scipy import stats
from scipy.stats import binomtest

from mixdyn import Cadence, Channel, Geo, MediaPanel, Revenue
from mixdyn.calibration import LiftEstimate
from mixdyn.models.bayes_geo import (
    BayesGeoFit,
    BayesGeoModel,
    BayesGeoSpec,
    _helmert,
    bayes_geo_model,
    fit_bayes_geo,
)
from mixdyn.models.bayes_mmm import BayesMedia, HillCurve, LogNormal, Normal, StudentT, TanhCurve
from mixdyn.models.ridge_mmm import Control, FourierSeason, LinearTrend
from test_bayes_mmm import _rescaled, bound, standardised

STAN = json.loads(
    (Path(__file__).parent / "fixtures" / "oracles" / "bayes_geo_stan.json").read_text()
)
CHANNELS = ("tv", "search", "social")


def stan_world(name: str) -> tuple[MediaPanel, BayesGeoSpec]:
    """The harness's world: its panel as the fixture records it, its model as the harness states
    it."""
    case = STAN["cases"][name]
    kpi = np.array(case["panel"]["kpi"], dtype=float)
    panel = MediaPanel(
        dates=np.datetime64("2023-01-02") + np.timedelta64(7, "D") * np.arange(kpi.shape[1]),
        cadence=Cadence.WEEK,
        geos=tuple(Geo(f"geo{g}") for g in range(kpi.shape[0])),
        channels=tuple(Channel(c) for c in CHANNELS),
        controls=("promotion", "price"),
        kpi=kpi,
        kpi_unit=Revenue(),
        spend=np.array(case["panel"]["spend"]),
        currency="EUR",
        control_values=np.array(case["panel"]["controls"]),
        filled=np.zeros(kpi.shape, dtype=bool),
        population=np.array(case["panel"]["population"]),
    )
    spec = BayesGeoSpec(
        tuple(BayesMedia(c) for c in CHANNELS),
        (Control("promotion", lower=0.0), Control("price")),
        LinearTrend(1),
        FourierSeason(52.18, 2),
        {"Normal": Normal(), "StudentT": StudentT()}[case["likelihood"]],
        {"HillCurve": HillCurve(), "TanhCurve": TanhCurve()}[case["saturation"]],
        centred=frozenset(case["centred"]),
    )
    return panel, spec


def stan_estimates(name: str) -> dict[str, list[LiftEstimate]]:
    """The harness's lift estimates on its world, as the fixture records them."""
    return {
        channel: [
            LiftEstimate(
                np.array(e["spend"]),
                np.array(e["planned"]),
                e["readout"],
                e["lift"],
                e["se"],
                e["share"],
                None if e["geos"] is None else frozenset(Geo(g) for g in e["geos"]),
            )
            for e in recorded
        ]
        for channel, recorded in STAN["cases"][name]["estimates"].items()
    }


def _log_units(panel: MediaPanel, model: BayesGeoModel) -> np.ndarray:
    """The log of each fitted cell's population times the KPI's unit, geo by geo, then of the mean
    population times the KPI's unit for each lift estimate."""
    seen = np.isfinite(panel.kpi)
    people = np.asarray(panel.population)
    cells = np.broadcast_to(people[:, None], seen.shape)[seen]
    lifts = np.full(sum(e.lift.size for e in model.estimates), people.mean())
    return np.log(np.concatenate([cells, lifts]) * model.kpi_unit)


@pytest.mark.parametrize("name", sorted(STAN["cases"]))
def test_the_log_density_is_stan_s_up_to_a_constant(name: str) -> None:
    """At 25 points uniform in ``[-2, 2]`` on the sampling coordinates, where Stan starts its
    chains: the log density, with every transform's Jacobian, differs from Stan's by one constant,
    and its gradient is Stan's."""
    case = STAN["cases"][name]
    panel, spec = stan_world(name)
    model = bayes_geo_model(panel, spec, estimates=stan_estimates(name))
    points = np.array(case["points"])
    assert len(model.coordinates) == model.dimension == points.shape[1]
    ours = [model.log_density(point) for point in points]
    offset = np.array([value for value, _ in ours]) - np.array(case["log_prob"])
    assert np.ptp(offset) <= 1e-12 * np.abs(case["log_prob"]).max()
    np.testing.assert_allclose(
        np.array([gradient for _, gradient in ours]),
        case["gradient"],
        rtol=1e-11,
        atol=1e-12 * np.abs(case["gradient"]).max(),
    )


BOUND = bound(list(STAN["cases"].values()), STAN["provenance"])
# Stan's own chains cross 1.01 on the Student-t world about one run in ten (ADR 0024)
RHAT_CEILING = {"normal": 1.01, "student": 1.1, "tanh": 1.01}
# the most divergent transitions in any of Stan's own twelve runs of four chains of 1000 draws
# (ADR 0025): its divergences come in bursts, which a binomial test reads as a defect
STAN_MOST_DIVERGENT = {"normal": 1, "student": 2, "tanh": 27}


@pytest.mark.parametrize("name", sorted(STAN["cases"]))
def test_the_posterior_is_stan_s(name: str) -> None:
    """Four chains of 1000 draws after 1000 of warm-up, the world's lift estimates in the
    likelihood: each parameter's posterior mean and standard deviation, in the panel's units, and
    each observation's log-likelihood's, within ``BOUND`` joint Monte Carlo standard errors of
    those of Stan's 40 000 draws, as `test_bayes_mmm.standardised` reads them, each error from
    batch means of ten batches a chain; every R-hat below its world's ceiling, 1.01 or, on the
    Student-t world, 1.1 (ADR 0024), and every chain's E-BFMI above 0.3;
    and divergent transitions no more frequent than Stan's, whose own chains diverge on some of
    the worlds: the engine's share of the two samplers' divergences is not above its share of their
    draws, by a one-sided binomial test at 0.001, or no more of them than in the most divergent of
    Stan's own twelve runs on the world (ADR 0025)."""
    case = STAN["cases"][name]
    panel, spec = stan_world(name)
    estimates = stan_estimates(name)
    fit = fit_bayes_geo(panel, spec, seed=1, estimates=estimates)
    assert set(fit.draws) == set(case["posterior"])
    model = bayes_geo_model(panel, spec, estimates=estimates)
    ours = {**fit.draws, "log_lik": fit.log_likelihood + _log_units(panel, model)}
    for key, stan in {**case["posterior"], "log_lik": case["log_lik"]}.items():
        mean, sd = standardised(ours[key].reshape(4, 1000, -1), stan, STAN["provenance"])
        assert np.all(np.abs(mean) <= BOUND), key
        assert np.all(np.abs(sd) <= BOUND), key
    for key in case["posterior"]:
        assert np.all(fit.diagnostics[key].rhat < RHAT_CEILING[name]), key
    ours, theirs = int(fit.divergent.sum()), case["divergent"]
    share = 4000 / (4000 + STAN["provenance"]["chains"] * STAN["provenance"]["draws"])
    assert (
        ours <= STAN_MOST_DIVERGENT[name]
        or binomtest(ours, ours + theirs, share, alternative="greater").pvalue > 1e-3
    ), ours
    assert np.all(fit.e_bfmi > 0.3)


@pytest.mark.parametrize("name", sorted(STAN["cases"]))
def test_each_observation_s_log_likelihood_is_stan_s(name: str) -> None:
    """At 25 of Stan's draws, each fitted cell geo by geo and then each lift estimate, every
    normalising term in, as Stan's generated quantities compute it on the model's scales: in the
    panel's units, less the log of the cell's population, or of the mean population for an
    estimate, times the KPI's unit."""
    recorded = STAN["cases"][name]["log_lik_draws"]
    panel, spec = stan_world(name)
    model = bayes_geo_model(panel, spec, estimates=stan_estimates(name))
    np.testing.assert_allclose(
        model.log_likelihood(np.array(recorded["theta"])) + _log_units(panel, model),
        recorded["log_lik"],
        rtol=1e-11,
        atol=1e-11,
    )


@pytest.fixture(scope="module", params=["normal", "tanh"])
def short_fit(request: pytest.FixtureRequest) -> tuple[MediaPanel, BayesGeoFit]:
    """Over the panel's last 80 weeks, its kernels reading the spend before them, the world's lift
    estimates in the likelihood."""
    panel, spec = stan_world(request.param)
    window = (panel.dates[24], panel.dates[-1])
    return panel, fit_bayes_geo(
        panel,
        spec,
        seed=3,
        window=window,
        estimates=stan_estimates(request.param),
        chains=2,
        warmup=150,
        draws=100,
    )


def test_a_model_of_its_channels_alone_has_no_calendar_or_control_draws() -> None:
    """And its fitted mean is each geo's base times its population, and its channels."""
    panel, spec = stan_world("normal")
    alone = dataclasses.replace(spec, controls=(), trend=None, season=None)
    fit = fit_bayes_geo(panel, alone, seed=4, chains=2, warmup=50, draws=20)
    assert "calendar" not in fit.draws
    assert "control" not in fit.draws
    assert set(fit.diagnostics) == set(fit.draws)
    rest = fit.estimated - sum(fit.contributions.values())
    base = fit.population * fit.draws["base"].mean(axis=(0, 1))
    np.testing.assert_allclose(rest, np.broadcast_to(base[:, None], rest.shape), rtol=1e-12)


def test_a_channel_s_contributions_over_the_window_and_the_geos_are_its_return_times_its_spend(
    short_fit: tuple[MediaPanel, BayesGeoFit],
) -> None:
    """Whatever its carryover, curve and geo deviations; the base's, the rest of the fitted mean,
    is each geo's base less its controls' coefficients times their means, times its population."""
    panel, fit = short_fit
    inside = np.isin(panel.dates, fit.dates)
    assert inside.sum() == 80
    for j, media in enumerate(fit.spec.media):
        assert fit.contributions[media.channel].shape == (6, 80)
        assert fit.contributions[media.channel].sum() == pytest.approx(
            fit.draws["roi"][..., j].mean() * panel.spend[:, inside, j].sum(), rel=1e-12
        )
    means = panel.control_values[:, inside].reshape(-1, 2).mean(axis=0)
    base = fit.draws["base"] - (fit.draws["control"] @ means)[..., None]
    rest = fit.estimated - sum(fit.contributions.values())
    np.testing.assert_allclose(
        rest,
        np.broadcast_to((base.mean(axis=(0, 1)) * fit.population)[:, None], rest.shape),
        rtol=1e-12,
    )


def test_each_draw_s_channels_contribute_its_return_times_the_spend_over_the_geos(
    short_fit: tuple[MediaPanel, BayesGeoFit],
) -> None:
    """The identity at every draw: each geo's channels as `chc.response` reads them there, in the
    panel's totals, contribute over the window and the geos the draw's return times their spend,
    to `1e-12`; a geo the fit does not have is refused."""
    panel, fit = short_fit
    roi = fit.draws["roi"].reshape(-1, len(fit.spec.media))
    window = slice(int(np.isin(panel.dates, fit.dates).argmax()), None)
    total = np.zeros_like(roi)
    for g, geo in enumerate(fit.geos):
        for i, channels in enumerate(fit.channel_draws(geo, roi.shape[0])):
            for j, channel in enumerate(channels.values()):
                total[i, j] += response.contribution(channel, panel.spend[g, :, j], window)
    np.testing.assert_allclose(total, roi * panel.spend[:, window].sum(axis=(0, 1)), rtol=1e-12)
    with pytest.raises(ValueError, match="the fit has no geo"):
        fit.channel_draws(Geo("nowhere"), 1)


def test_the_log_likelihood_reads_each_observation_at_each_draw(
    short_fit: tuple[MediaPanel, BayesGeoFit],
) -> None:
    """Each of the two chains' 100 draws, at the window's 80 weeks of six geos less the two
    without a KPI and at the 3 lift estimates; the KPI and the chains' divergences and E-BFMI
    beside it."""
    panel, fit = short_fit
    np.testing.assert_array_equal(fit.y, panel.kpi[:, 24:])
    np.testing.assert_array_equal(fit.fitted, np.isfinite(panel.kpi[:, 24:]))
    assert fit.log_likelihood.shape == (2, 100, 6 * 80 - 2 + 3)
    assert np.all(np.isfinite(fit.log_likelihood))
    assert fit.divergent.shape == fit.e_bfmi.shape == (2,)


def test_the_planner_s_channels_in_a_geo_are_the_model_s_curves_at_the_posterior_medians(
    short_fit: tuple[MediaPanel, BayesGeoFit],
) -> None:
    """In the panel's totals: the curve of the spend per head, a Hill at its median slope or
    `tanh` at half its ceiling where the adstock is the half-saturation point, times the geo's
    median coefficient per head and its population; the kernel as long as the calendar the model
    carries spend over, not the window."""
    panel, fit = short_fit
    tanh = isinstance(fit.spec.saturation, TanhCurve)
    assert ("slope" in fit.draws) is not tanh
    assert fit.length == panel.dates.size
    for g, geo in enumerate(fit.geos):
        people = float(np.asarray(panel.population)[g])
        for j, (name, planned) in enumerate(fit.channels(geo).items()):
            retention, half = (
                float(np.median(fit.draws[key][..., j])) for key in ("retention", "half_saturation")
            )
            coefficient = float(np.median(fit.draws["coefficient"][..., g, j]))
            spend = panel.spend[g, :, j]
            weights = retention ** np.arange(panel.dates.size)
            adstock = np.convolve(spend / people, weights / weights.sum())[: spend.size]
            if tanh:
                curve = np.tanh(math.atanh(0.5) * adstock / half)
            else:
                slope = float(np.median(fit.draws["slope"][..., j]))
                curve = adstock**slope / (adstock**slope + half**slope)
            np.testing.assert_allclose(
                np.asarray(planned(spend)), people * coefficient * curve, rtol=1e-12, err_msg=name
            )
    with pytest.raises(ValueError, match="no geo"):
        fit.channels(Geo("north"))


def test_units_move_only_what_they_should() -> None:
    """The KPI and every spend ten times as large, the lift estimates' too: the model reads the
    same data, so a point has the same log density, and the same return, carryover, slope, spread
    and calendar, while what is in the KPI's or the currency's units moves ten times, and each
    observation's log density in them by the log of a tenth. Every population ten times as large:
    what is per head moves a tenth, and the rest, each observation's log density in totals too,
    does not. A control ten times as large moves only its own coefficient, a tenth as large; one
    moved by a constant moves nothing."""
    panel, spec = stan_world("student")
    estimates = stan_estimates("student")
    model = bayes_geo_model(panel, spec, estimates=estimates)
    theta = np.random.default_rng(4).uniform(-2.0, 2.0, (3, model.dimension))
    ours = model.parameters(theta)
    fixed = ("roi", "retention", "slope", "spread", "calendar", "nu")
    per_head = ("half_saturation", "coefficient", "base", "control", "sigma")
    for edit, given, factor, shift in (
        (
            {"kpi": panel.kpi * 10.0, "spend": panel.spend * 10.0},
            _rescaled(estimates, 10.0),
            10.0,
            -math.log(10.0),
        ),
        ({"population": np.asarray(panel.population) * 10.0}, estimates, 0.1, 0.0),
    ):
        other = bayes_geo_model(dataclasses.replace(panel, **edit), spec, estimates=given)
        for point in theta:
            value, gradient = model.log_density(point)
            assert other.log_density(point)[0] == pytest.approx(value, rel=1e-12)
            np.testing.assert_allclose(other.log_density(point)[1], gradient, rtol=1e-10)
        theirs = other.parameters(theta)
        for key in fixed:
            np.testing.assert_allclose(theirs[key], ours[key], rtol=1e-12, err_msg=key)
        for key in per_head:
            np.testing.assert_allclose(theirs[key], factor * ours[key], rtol=1e-12, err_msg=key)
        np.testing.assert_allclose(
            other.log_likelihood(theta), model.log_likelihood(theta) + shift, rtol=1e-12
        )
    for move, factor in ((lambda v: v * 10.0, 0.1), (lambda v: v + 100.0, 1.0)):
        values = np.array(panel.control_values)
        values[..., 1] = move(values[..., 1])
        moved = bayes_geo_model(dataclasses.replace(panel, control_values=values), spec)
        theirs = moved.parameters(theta)
        np.testing.assert_allclose(theirs["control"][:, 1], factor * ours["control"][:, 1])
        np.testing.assert_allclose(theirs["control"][:, 0], ours["control"][:, 0], rtol=1e-12)
        for key in ("base", "coefficient", "sigma"):
            np.testing.assert_allclose(theirs[key], ours[key], rtol=1e-12, err_msg=key)


def test_the_geos_order_moves_nothing() -> None:
    """The panel's geos in another order, and a point's coordinates with them, each channel's
    deviations through Helmert's contrasts: the same log density, the lift estimates' geos read by
    their names."""
    panel, spec = stan_world("normal")
    estimates = stan_estimates("normal")
    order = [3, 0, 5, 1, 4, 2]
    shuffled = dataclasses.replace(
        panel,
        geos=tuple(panel.geos[g] for g in order),
        kpi=panel.kpi[order],
        spend=panel.spend[order],
        control_values=panel.control_values[order],
        filled=panel.filled[order],
        population=np.asarray(panel.population)[order],
    )
    model = bayes_geo_model(panel, spec, estimates=estimates)
    other = bayes_geo_model(shuffled, spec, estimates=estimates)
    assert sorted(other.coordinates) == sorted(model.coordinates)
    helmert = _helmert(6)
    for point in np.random.default_rng(5).uniform(-2.0, 2.0, (3, model.dimension)):
        mapped = np.array([point[model.coordinates.index(c)] for c in other.coordinates])
        for c in CHANNELS:
            rows = [i for i, n in enumerate(model.coordinates) if n.startswith(f"deviation[{c},")]
            mapped[rows] = helmert.T @ (helmert @ point[rows])[order]
        assert other.log_density(mapped)[0] == pytest.approx(model.log_density(point)[0], rel=1e-12)


@pytest.mark.parametrize("geos", [2, 3, 6, 9])
def test_helmert_s_contrasts_are_orthonormal_and_sum_to_zero(geos: int) -> None:
    basis = _helmert(geos)
    np.testing.assert_allclose(basis.T @ basis, np.eye(geos - 1), atol=1e-15)
    np.testing.assert_allclose(basis.sum(axis=0), 0.0, atol=1e-15)


@pytest.mark.parametrize("centred", [frozenset(), frozenset({"search"})])
def test_a_channel_s_national_effect_is_its_geo_effects_geometric_mean(
    centred: frozenset[str],
) -> None:
    """A channel's log factors sum to 0 over the geos, centred or not, so the log of its
    coefficient in each geo averages to the log of the effect its return sets over every geo."""
    panel, spec = stan_world("normal")
    model = bayes_geo_model(panel, dataclasses.replace(spec, centred=centred))
    for point in np.random.default_rng(7).uniform(-2.0, 2.0, (3, model.dimension)):
        p, _ = model._constrained(point)
        saturated, beta = (np.asarray(a) for a in model._terms(p))
        effect = np.asarray(p["effect"])
        np.testing.assert_allclose(effect.sum(axis=0), 0.0, atol=1e-12)
        reach = np.einsum("g,gm,gtm->m", model.weight, np.exp(effect), saturated)
        national = np.exp(np.asarray(p["log_roi"])) * model.window_spend / reach
        np.testing.assert_allclose(np.exp(np.log(beta).mean(axis=0)), national, rtol=1e-12)


def test_a_centred_channel_is_the_same_model_on_other_coordinates() -> None:
    """Its deviations are the non-centred ones times its spread: the log density moves by that
    map's log Jacobian, `G - 1` times the log spread, and no parameter or cell's log-likelihood
    moves at all."""
    panel, spec = stan_world("normal")
    free = bayes_geo_model(panel, dataclasses.replace(spec, centred=frozenset()))
    centred = bayes_geo_model(panel, dataclasses.replace(spec, centred=frozenset({"search"})))
    spread = free.coordinates.index("log_spread[search]")
    rows = [i for i, c in enumerate(free.coordinates) if c.startswith("deviation[search,")]
    assert len(rows) == len(panel.geos) - 1
    for point in np.random.default_rng(11).uniform(-2.0, 2.0, (5, free.dimension)):
        moved = point.copy()
        moved[rows] *= math.exp(point[spread])
        assert centred.log_density(moved)[0] == pytest.approx(
            free.log_density(point)[0] - len(rows) * point[spread], rel=1e-12
        )
        theirs = centred.parameters(moved)
        for key, value in free.parameters(point).items():
            np.testing.assert_allclose(theirs[key], value, rtol=1e-12, err_msg=key)
        np.testing.assert_allclose(
            centred.log_likelihood(moved), free.log_likelihood(point), rtol=1e-12
        )


def test_the_spread_s_prior_has_the_spec_s_scale() -> None:
    """Halving the half-normal's scale moves the log density by the prior's change alone."""
    panel, spec = stan_world("normal")
    model = bayes_geo_model(panel, spec)
    narrow = bayes_geo_model(panel, dataclasses.replace(spec, spread=0.5))
    columns = [i for i, c in enumerate(model.coordinates) if c.startswith("log_spread[")]
    for point in np.random.default_rng(10).uniform(-2.0, 2.0, (3, model.dimension)):
        eta = np.exp(point[columns])
        change = np.sum(-0.5 * (eta / 0.5) ** 2 - math.log(0.5) + 0.5 * eta**2)
        assert narrow.log_density(point)[0] - model.log_density(point)[0] == pytest.approx(
            change, rel=1e-9
        )


def test_a_hill_s_slope_prior_is_the_one_its_curve_states() -> None:
    """LogNormal(-0.3, 0.4) in place of LogNormal(0, 0.5) moves the log density by the two
    priors' difference at each log slope alone."""
    panel, spec = stan_world("normal")
    assert isinstance(spec.saturation, HillCurve)
    model = bayes_geo_model(panel, spec)
    leaning = bayes_geo_model(panel, dataclasses.replace(spec, saturation=HillCurve(-0.3, 0.4)))
    columns = [i for i, c in enumerate(model.coordinates) if c.startswith("log_slope[")]
    assert columns
    for point in np.random.default_rng(14).uniform(-2.0, 2.0, (3, model.dimension)):
        u = point[columns]
        change = np.sum(stats.norm.logpdf(u, -0.3, 0.4) - stats.norm.logpdf(u, 0.0, 0.5))
        assert leaning.log_density(point)[0] - model.log_density(point)[0] == pytest.approx(
            change, rel=1e-9
        )


def test_a_log_normal_noise_prior_moves_the_density_by_the_prior_s_change_alone() -> None:
    """`sigma ~ LogNormal(log 0.6745, 0.5)` in place of `HalfNormal(1)`, on the same coordinate:
    the log density moves by the difference of the two priors at `sigma`, the half-normal's
    written as the engine writes it, without its constant `log 2`, and nothing else."""
    panel, spec = stan_world("normal")
    model = bayes_geo_model(panel, spec)
    other = bayes_geo_model(panel, dataclasses.replace(spec, noise_prior="log-normal"))
    assert other.coordinates == model.coordinates
    column = model.coordinates.index("log_sigma")
    for point in np.random.default_rng(12).uniform(-2.0, 2.0, (3, model.dimension)):
        sigma = math.exp(point[column])
        change = stats.lognorm.logpdf(sigma, 0.5, scale=0.6745) - stats.norm.logpdf(sigma)
        assert other.log_density(point)[0] - model.log_density(point)[0] == pytest.approx(
            change, rel=1e-9
        )


def test_a_normal_base_prior_samples_the_base_itself() -> None:
    """`a_g ~ N(1, 1)` sampled as `a_g`: at a positive base the log density is the log-normal
    model's less its prior and Jacobian there plus the normal prior, the parameters are the same,
    and a base below 0 has a finite density."""
    panel, spec = stan_world("normal")
    model = bayes_geo_model(panel, spec)
    other = bayes_geo_model(panel, dataclasses.replace(spec, base_prior="normal"))
    rows = [i for i, c in enumerate(model.coordinates) if c.startswith("log_base[")]
    assert [other.coordinates[i] for i in rows] == [f"base[{g}]" for g in panel.geos]
    for point in np.random.default_rng(13).uniform(-2.0, 2.0, (3, model.dimension)):
        moved = point.copy()
        base = np.exp(point[rows])
        moved[rows] = base
        change = np.sum(
            stats.norm.logpdf(base, 1.0, 1.0) - stats.lognorm.logpdf(base, 1.0) - point[rows]
        )
        assert other.log_density(moved)[0] - model.log_density(point)[0] == pytest.approx(
            change, rel=1e-9
        )
        theirs = other.parameters(moved)
        for key, value in model.parameters(point).items():
            np.testing.assert_allclose(theirs[key], value, rtol=1e-12, err_msg=key)
        moved[rows] = -base
        assert np.isfinite(other.log_density(moved)[0])


def test_a_fixed_slope_is_one_and_not_sampled() -> None:
    """Each Hill's slope fixed at 1: the slopes' coordinates are gone, the log density is the
    sampled model's at log slope 0 less the slope's prior there, and the parameters hold no
    slope."""
    panel, spec = stan_world("normal")
    model = bayes_geo_model(panel, spec)
    fixed = bayes_geo_model(panel, dataclasses.replace(spec, slope="fixed"))
    rows = [i for i, c in enumerate(model.coordinates) if c.startswith("log_slope[")]
    assert len(rows) == len(CHANNELS)
    assert fixed.coordinates == tuple(np.delete(model.coordinates, rows))
    for point in np.random.default_rng(14).uniform(-2.0, 2.0, (3, fixed.dimension)):
        full = np.insert(point, rows[0], np.zeros(len(rows)))
        prior = len(rows) * stats.norm.logpdf(0.0, 0.0, 0.5)
        assert fixed.log_density(point)[0] == pytest.approx(
            model.log_density(full)[0] - prior, rel=1e-12
        )
        theirs = fixed.parameters(point)
        assert "slope" not in theirs
        for key, value in theirs.items():
            np.testing.assert_allclose(value, model.parameters(full)[key], rtol=1e-12, err_msg=key)


def test_a_fit_with_a_fixed_slope_hands_on_hills_of_slope_one() -> None:
    panel, spec = stan_world("normal")
    fit = fit_bayes_geo(
        panel, dataclasses.replace(spec, slope="fixed"), seed=3, chains=2, warmup=20, draws=10
    )
    assert "slope" not in fit.draws
    for channel in fit.channels(panel.geos[0]).values():
        assert isinstance(channel.curve, response.Hill)
        assert float(channel.curve.slope) == 1.0


def test_the_scales_are_pooled_over_the_geos() -> None:
    """Over a window: the KPI's unit is its total over the observed cells over their heads, and a
    channel's its mean positive spend per head over the panel's whole calendar and every geo."""
    panel, spec = stan_world("normal")
    window = (panel.dates[24], panel.dates[-1])
    model = bayes_geo_model(panel, spec, window=window)
    people = np.asarray(panel.population)
    kpi = panel.kpi[:, 24:]
    seen = np.isfinite(kpi)
    assert model.kpi_unit == pytest.approx(
        kpi[seen].sum() / np.broadcast_to(people[:, None], seen.shape)[seen].sum(),
        rel=1e-14,
        abs=0.0,
    )
    head = panel.spend / people[:, None, None]
    np.testing.assert_allclose(
        model.spend_unit, [h[h > 0.0].mean() for h in head.reshape(-1, 3).T], rtol=1e-14
    )


def test_a_window_s_spend_carries_over_from_before_it() -> None:
    """Each geo's saturated spend over a window is the whole calendar's over the window's
    periods."""
    panel, spec = stan_world("tanh")
    whole = bayes_geo_model(panel, spec)
    windowed = bayes_geo_model(panel, spec, window=(panel.dates[24], panel.dates[-1]))
    for point in np.random.default_rng(11).uniform(-2.0, 2.0, (2, whole.dimension)):
        p, _ = whole._constrained(point)
        for j in range(3):
            np.testing.assert_allclose(
                windowed._saturated(p, j), whole._saturated(p, j)[:, 24:], rtol=1e-12
            )


def test_a_panel_without_populations_has_one_head_a_geo() -> None:
    """The same log density as every population 1, or every population 7, the lift estimates'
    groups' spend spread over their geos' heads."""
    panel, spec = stan_world("tanh")
    models = [
        bayes_geo_model(
            dataclasses.replace(panel, population=population),
            spec,
            estimates=stan_estimates("tanh"),
        )
        for population in (None, np.ones(6), np.full(6, 7.0))
    ]
    for point in np.random.default_rng(9).uniform(-2.0, 2.0, (3, models[0].dimension)):
        values = [m.log_density(point)[0] for m in models]
        assert values == pytest.approx([values[0]] * 3, rel=1e-12)


def _expected_lift(
    model: BayesGeoModel, panel: MediaPanel, point: np.ndarray, channel: str, e: LiftEstimate
) -> float:
    """The lift a test shows at `point`, in the KPI's units, from the parameters in the panel's
    units: in each geo it ran in, the group's share of the geo's people times the geo's coefficient
    per head, times the gap in the channel's curve at the spend per head the group's people saw,
    summed over the readout."""
    raw = model.parameters(point[None])
    j = CHANNELS.index(channel)
    people = np.asarray(panel.population)
    where = np.array([e.geos is None or g in e.geos for g in panel.geos])
    heads = e.share * people[where].sum()
    weights = raw["retention"][0, j] ** np.arange(model.length)
    half = raw["half_saturation"][0, j]

    def curve(series: np.ndarray) -> np.ndarray:
        padded = np.r_[np.zeros(model.length - series.size), series / heads]
        adstock = np.convolve(padded, weights / weights.sum())[: model.length]
        if isinstance(model.spec.saturation, TanhCurve):
            return np.tanh(math.atanh(0.5) * adstock / half)
        slope = raw["slope"][0, j]
        return adstock**slope / (adstock**slope + half**slope)

    gap = (curve(e.spend) - curve(e.planned))[-e.readout :].sum()
    return e.share * float(people[where] @ raw["coefficient"][0, where, j]) * gap


@pytest.mark.parametrize("name", ["normal", "tanh"])
def test_an_estimate_is_read_in_the_geos_its_test_ran_in(name: str) -> None:
    """A go-dark in a sub-market of two geos, a heavy-up over every geo and one in a single geo:
    each estimate's lift is normal about the group's share of its geos' people times each geo's
    coefficient per head, times the gap in the channel's curve at the spend per head the group's
    people saw, summed over the readout. One standard error above that, its log density, the
    lift's over its share, is a normal's there."""
    panel, spec = stan_world(name)
    model = bayes_geo_model(panel, spec)
    estimates = stan_estimates(name)
    assert sorted((c, e.geos and sorted(e.geos)) for c, es in estimates.items() for e in es) == [
        ("search", ["geo3"]),
        ("social", ["geo1", "geo4"]),
        ("tv", None),
    ]
    for point in np.random.default_rng(12).uniform(-2.0, 2.0, (3, model.dimension)):
        for channel, recorded in estimates.items():
            for e in recorded:
                lift = _expected_lift(model, panel, point, channel, e) + e.se
                read = bayes_geo_model(
                    panel, spec, estimates={channel: [dataclasses.replace(e, lift=lift)]}
                )
                assert read.log_likelihood(point)[0, -1] == pytest.approx(
                    -math.log(e.se / e.share) - 0.5 * math.log(2.0 * math.pi) - 0.5, rel=1e-12
                ), channel


def test_a_channel_s_estimates_add_their_own_terms_and_read_its_parameters_alone() -> None:
    """Television's national estimate adds one term to the log density at every point and leaves
    the rest, every prior and every cell's term, as it was: the gradient in every coordinate but
    television's does not move."""
    panel, spec = stan_world("normal")
    tested = bayes_geo_model(panel, spec, estimates={"tv": stan_estimates("normal")["tv"]})
    untested = bayes_geo_model(panel, spec)
    theta = np.random.default_rng(13).uniform(-2.0, 2.0, (4, tested.dimension))
    terms = tested.log_likelihood(theta)
    np.testing.assert_allclose(terms[:, :-1], untested.log_likelihood(theta), rtol=1e-14)
    others = [k for k, name in enumerate(tested.coordinates) if "[tv" not in name]
    unit = math.log(tested.kpi_unit * np.asarray(panel.population).mean())
    for k in range(theta.shape[0]):
        (value, gradient), (base, base_gradient) = (
            tested.log_density(theta[k]),
            untested.log_density(theta[k]),
        )
        # the densities, near 3e3, cancel to a difference near 0.1 that keeps their rounding,
        # 9e-14 here, so it is read to 1e-12 absolute where that is wider than its 1e-12 relative
        assert value - base == pytest.approx(terms[k, -1] + unit, rel=1e-12, abs=1e-12)
        np.testing.assert_allclose(gradient[others], base_gradient[others], rtol=1e-13)


def test_each_of_a_channel_s_estimates_is_a_term_of_its_own() -> None:
    """Television's national estimate and a shorter one in a single geo, read together: each
    estimate's log density is the one it has read alone."""
    panel, spec = stan_world("tanh")
    national = stan_estimates("tanh")["tv"][0]
    local = dataclasses.replace(
        national,
        readout=national.readout - 2,
        lift=0.2 * national.lift,
        geos=frozenset({panel.geos[2]}),
    )
    both = bayes_geo_model(panel, spec, estimates={"tv": [national, local]})
    theta = np.random.default_rng(17).uniform(-2.0, 2.0, (3, both.dimension))
    terms = both.log_likelihood(theta)[:, -2:]
    for k, e in enumerate((national, local)):
        alone = bayes_geo_model(panel, spec, estimates={"tv": [e]})
        np.testing.assert_allclose(terms[:, k], alone.log_likelihood(theta)[:, -1], rtol=1e-13)


def test_a_group_s_share_moves_nothing_its_geos_see() -> None:
    """Each lift estimate's group half its share of its geos' market, with half its spend, lift
    and error: the geos see the same test, so a point has the same log density."""
    panel, spec = stan_world("normal")
    estimates = stan_estimates("normal")
    model = bayes_geo_model(panel, spec, estimates=estimates)
    other = bayes_geo_model(panel, spec, estimates=_rescaled(estimates, 0.5, share=0.5))
    for point in np.random.default_rng(14).uniform(-2.0, 2.0, (3, model.dimension)):
        assert other.log_density(point)[0] == pytest.approx(model.log_density(point)[0], rel=1e-12)


def test_a_lift_estimate_longer_than_the_panel_runs_the_kernels_over_its_length() -> None:
    """The panel's spend is preceded by none: each geo's saturated spend is carried over by the
    kernel normalised over the estimate's length, and a fit hands the planner that kernel."""
    panel, spec = stan_world("tanh")
    e = stan_estimates("tanh")["tv"][0]
    longer = dataclasses.replace(
        e,
        spend=np.r_[np.full(120, e.spend[0]), e.spend],
        planned=np.r_[np.full(120, e.planned[0]), e.planned],
    )
    model = bayes_geo_model(panel, spec, estimates={"tv": [longer]})
    assert model.length == 120 + e.spend.size > panel.dates.size
    head = panel.spend / np.asarray(panel.population)[:, None, None]
    for point in np.random.default_rng(15).uniform(-2.0, 2.0, (2, model.dimension)):
        p, _ = model._constrained(point)
        raw = model.parameters(point[None])
        for j in range(3):
            weights = raw["retention"][0, j] ** np.arange(model.length)
            adstock = np.array(
                [np.convolve(h, weights / weights.sum())[: h.size] for h in head[:, :, j]]
            )
            np.testing.assert_allclose(
                model._saturated(p, j),
                np.tanh(math.atanh(0.5) * adstock / raw["half_saturation"][0, j]),
                rtol=1e-12,
            )
    fit = fit_bayes_geo(
        panel, spec, seed=16, estimates={"tv": [longer]}, chains=2, warmup=20, draws=10
    )
    assert fit.length == model.length


def test_a_cell_without_a_kpi_is_not_fitted() -> None:
    panel, spec = stan_world("normal")
    model = bayes_geo_model(panel, spec)
    assert np.argwhere(~model.observed).tolist() == [[2, 5], [2, 60], [4, 30]]
    seen = np.isfinite(panel.kpi)
    heads = np.broadcast_to(np.asarray(panel.population)[:, None], seen.shape)[seen]
    np.testing.assert_allclose(model.y * model.kpi_unit * heads, panel.kpi[seen], rtol=1e-15)


def test_a_filled_cell_reads_each_control_at_its_mean() -> None:
    """A step the input had no row for has no KPI, no spend and no controls: the controls are
    standardised over the other cells, and the filled one reads 0."""
    panel, spec = stan_world("normal")
    kpi, spend, values, filled = (
        np.array(a) for a in (panel.kpi, panel.spend, panel.control_values, panel.filled)
    )
    kpi[1, 10], spend[1, 10], values[1, 10], filled[1, 10] = np.nan, 0.0, np.nan, True
    model = bayes_geo_model(
        dataclasses.replace(panel, kpi=kpi, spend=spend, control_values=values, filled=filled),
        spec,
    )
    others = np.delete(panel.control_values.reshape(-1, 2), 1 * 104 + 10, axis=0)
    np.testing.assert_allclose(model.control_mean, others.mean(axis=0), rtol=1e-14)
    np.testing.assert_allclose(model.control_sd, others.std(axis=0), rtol=1e-14)
    assert not model.controls[1, 10].any()
    assert np.isfinite(model.log_density(np.zeros(model.dimension))[0])


def test_a_control_held_to_a_sign_keeps_it() -> None:
    panel, spec = stan_world("normal")
    held = (Control("promotion", lower=0.0), Control("price", upper=0.0))
    model = bayes_geo_model(panel, dataclasses.replace(spec, controls=held))
    control = model.parameters(np.random.default_rng(6).uniform(-2.0, 2.0, (50, model.dimension)))[
        "control"
    ]
    assert (control[:, 0] > 0.0).all()
    assert (control[:, 1] < 0.0).all()


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: BayesGeoSpec((BayesMedia("tv"),), spread=0.0), "positive, finite spread"),
        (lambda: BayesGeoSpec((BayesMedia("tv"),), spread=math.inf), "positive, finite spread"),
        (lambda: BayesGeoSpec(()), "at least one channel"),
        (
            lambda: BayesGeoSpec((BayesMedia("tv"),), centred=frozenset({"radio"})),
            r"centred channels \['radio'\] are not among the spec's media",
        ),
        (
            lambda: BayesGeoSpec((BayesMedia("tv"),), (Control("tv"),)),
            r"\['tv'\] more than once",
        ),
        (
            lambda: BayesGeoSpec((BayesMedia("tv"),), likelihood=LogNormal()),  # ty: ignore[invalid-argument-type]
            "reads normal or Student-t noise, not LogNormal",
        ),
        (
            lambda: BayesGeoSpec((BayesMedia("tv"),), noise_prior="gamma"),  # ty: ignore[invalid-argument-type]
            r"a noise prior is one of \['half-normal', 'log-normal'\], got 'gamma'",
        ),
        (
            lambda: BayesGeoSpec((BayesMedia("tv"),), base_prior="flat"),  # ty: ignore[invalid-argument-type]
            r"a base prior is one of \['log-normal', 'normal'\], got 'flat'",
        ),
        (
            lambda: BayesGeoSpec((BayesMedia("tv"),), slope="free"),  # ty: ignore[invalid-argument-type]
            r"a slope is one of \['sampled', 'fixed'\], got 'free'",
        ),
        (
            lambda: BayesGeoSpec((BayesMedia("tv"),), saturation=TanhCurve(), slope="fixed"),
            "a tanh curve has no slope to fix",
        ),
    ],
)
def test_a_model_that_cannot_be_is_refused(make, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        make()


def _one_geo(panel: MediaPanel) -> MediaPanel:
    return dataclasses.replace(
        panel,
        geos=panel.geos[:1],
        kpi=panel.kpi[:1],
        spend=panel.spend[:1],
        control_values=panel.control_values[:1],
        filled=panel.filled[:1],
        population=np.asarray(panel.population)[:1],
    )


def _dark(panel: MediaPanel) -> MediaPanel:
    spend = np.array(panel.spend)
    spend[:, :10, 2] = 0.0
    return dataclasses.replace(panel, spend=spend)


def _blind(panel: MediaPanel) -> MediaPanel:
    kpi = np.array(panel.kpi)
    kpi[3, :10] = np.nan
    return dataclasses.replace(panel, kpi=kpi)


def _flat(panel: MediaPanel) -> MediaPanel:
    values = np.array(panel.control_values)
    values[:, :, 1] = 10.0
    return dataclasses.replace(panel, control_values=values)


FIRST_TEN = (np.datetime64("2023-01-02"), np.datetime64("2023-03-06"))


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda c: {"panel": _one_geo(c["panel"])}, r"two geos or more; the panel has \['geo0'\]"),
        (
            lambda c: {
                "spec": dataclasses.replace(
                    c["spec"], media=(BayesMedia("radio"),), centred=frozenset()
                )
            },
            r"no channel or control named \['radio'\]",
        ),
        (
            lambda c: {"spec": dataclasses.replace(c["spec"], controls=(Control("weather"),))},
            r"no channel or control named \['weather'\]",
        ),
        (lambda c: {"window": FIRST_TEN[::-1]}, "must run forward"),
        (
            lambda c: {"panel": _dark(c["panel"]), "window": FIRST_TEN},
            r"\['social'\] spend nothing",
        ),
        (
            lambda c: {"panel": _blind(c["panel"]), "window": FIRST_TEN},
            r"geos \['geo3'\] have no KPI",
        ),
        (
            lambda c: {"panel": dataclasses.replace(c["panel"], kpi=-c["panel"].kpi)},
            "mean KPI per head is -",
        ),
        (lambda c: {"panel": _flat(c["panel"])}, r"\['price'\] do not vary"),
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
                            c["estimates"]["tv"][0], geos=frozenset({Geo("geo1"), Geo("north")})
                        )
                    ]
                }
            },
            r"estimates name geos \['north'\] that the panel does not have",
        ),
    ],
)
def test_what_it_cannot_fit_is_refused(edit, match: str) -> None:
    panel, spec = stan_world("normal")
    call: dict[str, Any] = {
        "panel": panel,
        "spec": spec,
        "window": None,
        "estimates": stan_estimates("normal"),
    }
    call |= edit(call)
    with pytest.raises(ValueError, match=match):
        bayes_geo_model(**call)


def test_single_precision_is_refused() -> None:
    panel, spec = stan_world("normal")
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(ValueError, match="double precision"):
            bayes_geo_model(panel, spec)
    finally:
        jax.config.update("jax_enable_x64", True)
