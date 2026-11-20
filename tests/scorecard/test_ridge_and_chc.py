"""The ridge arm and the CHC arm: each runs a world end to end and is scored beside the built-in
arms, writes the record every arm writes, repeats itself at a seed, reads no lift test and says so,
and refuses an export it cannot read. The ridge arm's fit is the library's search and fit; the CHC
arm's search finds the curves a world was made with, and its draws hold every channel at or above
0."""

import dataclasses
import inspect
import json
import math

import numpy as np
import pytest

pytest.importorskip("causaldyn_bench")

from causaldyn_bench.scorecard.geo import GEO
from causaldyn_bench.scorecard.mapping import TOLERANCE, drawn, residual
from causaldyn_bench.scorecard.returns import drawn_returns
from causaldyn_bench.scorecard.short import SHORT
from causaldyn_bench.scorecard.track_m2 import TRACK_M2
from chc.allocation import allocate
from chc.dlm import constrained_sample, forward_filter

from mixdyn import NATIONAL
from mixdyn.models.ridge_mmm import Control, FourierSeason, LinearTrend, fit_ridge
from mixdyn.models.ridge_search import (
    Folds,
    GeometricSpace,
    Interval,
    MediaSpace,
    RidgeSpace,
    search_ridge,
)
from scorecard import flags as f
from scorecard import report, run
from scorecard.arms import chc_arm, ridge
from scorecard.arms import panel as reading
from scorecard.arms.base import records
from scorecard.layout import Declaration, Layout, Limits

SEED = 905
LIMITS = Limits(memory="4G", timeout=600.0, slots=1)
COMMON = {"version", "seed", "digest", "family", "environment", "k", "python", "versions"}
RECORDED = COMMON | {"smoke", "weekly", "flags", "settings", "error", "gain", "n/a", "response"}


@pytest.fixture(scope="module")
def observation():
    return TRACK_M2.observe(TRACK_M2.world("drawn", SEED), 0)


@pytest.fixture(scope="module")
def ridge_fitted(observation):
    return ridge.Ridge().fit(observation, smoke=True)


@pytest.fixture(scope="module")
def chc_fitted(observation):
    return chc_arm.Chc().fit(observation, smoke=True)


@pytest.fixture(scope="module")
def made(observation):
    """Each arm's records of the world, twice over."""
    return {
        name: [records(arm, observation, smoke=True) for _ in range(2)]
        for name, arm in (("ridge", ridge.Ridge()), ("chc", chc_arm.Chc()))
    }


def _child(seed):
    return int(np.random.SeedSequence(seed).spawn(1)[0].generate_state(1)[0])


def test_both_arms_run_a_world_end_to_end_and_are_scored_beside_the_built_in_arms(tmp_path, capsys):
    """Declared with their limits, fitted once for both rungs, scored and reported: each rung's
    record is the k = 0 plan, which says it read no test."""
    layout = Layout(tmp_path / "run")
    declared = Declaration(
        family=0,
        environments=("drawn",),
        worlds=1,
        pilot=True,
        rungs=(0, 4),
        arms=("ridge", "chc"),
        smoke=True,
        limits={"ridge": LIMITS, "chc": LIMITS},
    )
    with pytest.raises(ValueError, match="no limits for"):
        run.declare(layout, dataclasses.replace(declared, limits={"ridge": LIMITS}))
    run.declare(layout, declared)
    assert layout.read().limits == {"ridge": LIMITS, "chc": LIMITS}
    run.export_all(layout)
    (seed,) = layout.read().seeds("drawn")
    for arm in ("ridge", "chc"):
        run.one(layout, arm, "drawn", seed, (0, 4))
        first, second = (
            json.loads(layout.record(arm, 0, "drawn", k, seed).read_text()) for k in (0, 4)
        )
        assert first["error"] is None
        assert (first["k"], second["k"]) == (0, 4)
        assert second["weekly"] == first["weekly"]
        assert second["cost"] == {"fitted with k": 0}
        assert "lift tests" in second["n/a"]
        assert second["settings"]["lift_rows_read"] == 0
    run.score(layout)
    lines = [json.loads(line) for line in layout.scores.read_text().splitlines()]
    assert sorted(line["k"] for line in lines) == [0, 4]
    for line in lines:
        for arm in ("ridge", "chc"):
            assert line["arms"][arm]["failure"] is None
            assert math.isfinite(line["arms"][arm]["regret"])
    report.main(layout, ["ridge", "chc"])
    shown = capsys.readouterr().out.splitlines()
    for arm in ("ridge", "chc"):
        assert any(line.split()[:2] == [arm, "regret"] for line in shown)


@pytest.mark.parametrize("name", ["ridge", "chc"])
def test_each_record_holds_a_plan_in_the_box_on_the_budget_and_what_it_did_not_read(
    made, observation, name
):
    record = made[name][0][name]
    expected = RECORDED | ({"gain_interval"} if name == "chc" else set()) | {"cost"}
    assert set(record) == expected
    assert record["error"] is None
    weekly = np.asarray(record["weekly"])
    assert np.all(weekly >= observation.lower * (1.0 - 1e-9))
    assert np.all(weekly <= observation.upper * (1.0 + 1e-9))
    assert weekly.sum() * observation.planned == pytest.approx(
        observation.budget, rel=1e-9, abs=0.0
    )
    assert record["n/a"]["lift tests"].endswith("every rung's plan is the plan of k = 0")
    assert "forecast" in record["n/a"]
    assert record["settings"]["lift_rows_read"] == 0
    assert record["settings"]["planner"] == "allocate"
    smoke = {"ridge": ridge.SMOKE_STARTS, "chc": dataclasses.asdict(chc_arm.SMOKE_SEARCH)}[name]
    assert record["settings"]["starts" if name == "ridge" else "search"] == smoke
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize("name", ["ridge", "chc"])
def test_each_arm_repeats_itself_at_a_seed_from_a_child_of_the_exports(made, observation, name):
    first, second = (records[name] for records in made[name])
    assert {k: v for k, v in first.items() if k != "cost"} == {
        k: v for k, v in second.items() if k != "cost"
    }
    assert first["settings"]["seed"] == _child(observation.seed) != observation.seed
    if name == "chc":
        second_child = np.random.SeedSequence(observation.seed).spawn(2)[1]
        assert first["settings"]["search_seed"] == int(second_child.generate_state(1)[0])


@pytest.mark.parametrize("arm", [ridge.Ridge(), chc_arm.Chc()])
def test_neither_arm_reads_the_tests(arm):
    assert arm.tests is False


@pytest.fixture(scope="module")
def geo_export():
    """Family 8's first pilot world, exported with no test."""
    return GEO.observe(GEO.world("policy-0", GEO.pilots["policy-0"][0]), 0)


@pytest.mark.parametrize("arm", [ridge.Ridge(), chc_arm.Chc()])
def test_neither_arm_plans_a_geo_exports_cells(arm):
    """The harness hands each a geo export's market (`tests/scorecard/test_geo_arms.py`)."""
    assert arm.cells is False


def test_the_chc_arm_refuses_a_geo_export_and_names_what_it_lacks(geo_export):
    with pytest.raises(ValueError, match="geo axis of 10 geos: the arm fits no channel for each"):
        chc_arm.Chc().fit(geo_export, smoke=True)


def test_the_ridge_arm_refuses_a_geo_export(geo_export):
    with pytest.raises(ValueError, match="the panel has no geo 'national'"):
        ridge.Ridge().fit(geo_export, smoke=True)


def test_the_ridge_arm_searches_the_space_of_the_librarys_scored_run(observation, monkeypatch):
    """Each channel's geometric carryover and Hill, the penalty, the controls, the trend and the
    season and the folds as the library's scored run searched them, from a child of the export's
    seed; a full run's starts and polishes are the library's own."""
    seen = []

    def search(*args, **kwargs):
        seen.append((args, kwargs))
        return search_ridge(*args, **kwargs)

    monkeypatch.setattr(ridge, "search_ridge", search)
    ridge.Ridge().fit(observation, smoke=True)
    ((args, kwargs),) = seen
    media = tuple(
        MediaSpace(c, GeometricSpace(Interval(0.0, 0.9)), Interval(0.5, 3.0), Interval(0.3, 1.0))
        for c in observation.channels
    )
    space = RidgeSpace(
        media=media,
        controls=(Control("promotion", lower=0.0), Control("price")),
        penalty=Interval(1e-4, 1e2),
        trend=LinearTrend(3),
        season=FourierSeason(365.25 / 7.0, 3),
    )
    assert args[1:] == (NATIONAL, space, Folds(4, 13))
    assert kwargs == {"seed": _child(observation.seed), "starts": 16, "polish": 2}
    defaults = inspect.signature(search_ridge).parameters
    assert (defaults["starts"].default, defaults["polish"].default) == (ridge.STARTS, ridge.POLISH)


@pytest.mark.parametrize(("environment", "count"), [("L-52", 2), ("L-78", 4)])
def test_the_ridge_arm_searches_a_short_history_by_the_folds_it_holds(environment, count):
    """4 folds of 13 weeks would leave a history of 52 weeks nothing to fit: it is searched by 2,
    the first fitting 26 weeks, as the first of 78 weeks' 4 does."""
    observation = SHORT.observe(SHORT.world(environment, SHORT.pilots[environment][0]), 0)
    assert ridge.Ridge().fit(observation, smoke=True).settings["folds"] == [count, 13]


def test_the_ridge_arms_fit_is_fit_ridges_of_the_searchs_choice(ridge_fitted, observation):
    """The library composes the two: the search's fit is `fit_ridge` of the model it chose over
    the whole history, its promotion held at or above 0 and its price free."""
    fit = ridge_fitted.search.fit
    assert fit.spec.controls == (Control("promotion", lower=0.0), Control("price"))
    again = fit_ridge(reading.panel(observation), NATIONAL, fit.spec)
    assert again.coefficients == fit.coefficients
    assert again.intercept == fit.intercept


def test_the_ridge_plan_is_allocates_on_the_fits_channels_and_claims_its_gain(
    ridge_fitted, observation
):
    channels = ridge_fitted.search.fit.channels()
    read = [channels[c] for c in observation.channels]
    expected = allocate(
        read,
        observation.budget,
        observation.planned,
        lower=observation.lower,
        upper=observation.upper,
        history=observation.spend,
    ).spend
    planned = ridge.Ridge().plan(ridge_fitted, "ridge", observation)
    np.testing.assert_array_equal(planned.weekly, np.asarray(expected))
    worth = sum(
        reading.worth(c, observation.spend[:, j], float(planned.weekly[j]), observation.planned)
        - reading.worth(
            c, observation.spend[:, j], float(observation.status_quo[j]), observation.planned
        )
        for j, c in enumerate(read)
    )
    assert planned.gain == pytest.approx(worth, rel=1e-12, abs=0.0)
    assert planned.gain_interval is None
    assert "gain_interval" in planned.absent


def test_the_ridge_flags_read_the_searchs_choice(ridge_fitted, observation):
    flags, fit = ridge_fitted.flags, ridge_fitted.search.fit
    media = {m.channel: m for m in fit.spec.media}
    assert flags["penalty"] == fit.spec.penalty
    assert flags["score"] == ridge_fitted.search.score
    assert flags["nrmse"] == fit.nrmse()
    # R² and NRMSE read one residual: its sum of squares is n (NRMSE x the KPI's range)^2
    y = fit.y[fit.fitted]
    unexplained = y.size * (fit.nrmse() * (y.max() - y.min())) ** 2
    assert flags["r2"] == pytest.approx(
        1.0 - unexplained / np.sum((y - y.mean()) ** 2), rel=1e-12, abs=0.0
    )
    assert flags["penalty at an end"] is ridge._at_an_end(fit.spec.penalty, ridge.PENALTY)
    for name in observation.channels:
        m = media[name]
        for parameter, value, interval in (
            ("retention", m.adstock.theta, ridge.THETA),
            ("alpha", m.hill.alpha, ridge.ALPHA),
            ("gamma", m.hill.gamma, ridge.GAMMA),
        ):
            assert flags[f"{name} {parameter}"] == value
            assert flags[f"{name} {parameter} at an end"] is ridge._at_an_end(value, interval)
        assert flags[f"{name} on its bound"] == fit.bound[name]
    assert "fitted not mixed" not in flags
    assert not any("rhat" in key for key in flags)


@pytest.mark.parametrize(
    ("ends", "end"),
    [
        ((ridge.PENALTY.low, ridge.THETA.low, ridge.ALPHA.low, ridge.GAMMA.low), True),
        ((ridge.PENALTY.high, ridge.THETA.high, ridge.ALPHA.high, ridge.GAMMA.high), True),
        ((1.0, 0.45, 1.5, 0.6), False),
    ],
)
def test_the_ridge_flags_say_which_of_the_choice_sits_at_an_end(
    ridge_fitted, observation, ends, end
):
    """The search's choice moved to an end of each interval, the penalty and the first channel's
    carryover and Hill, is flagged at that end; moved inside, it is not."""
    penalty, theta, alpha, gamma = ends
    search, first = ridge_fitted.search, observation.channels[0]
    media = tuple(
        dataclasses.replace(
            m,
            adstock=dataclasses.replace(m.adstock, theta=theta),
            hill=dataclasses.replace(m.hill, alpha=alpha, gamma=gamma),
        )
        if m.channel == first
        else m
        for m in search.fit.spec.media
    )
    spec = dataclasses.replace(search.fit.spec, penalty=penalty, media=media)
    moved = dataclasses.replace(search, fit=dataclasses.replace(search.fit, spec=spec))
    flags = ridge._flags(moved, observation.channels)
    assert flags["penalty at an end"] is end
    for parameter in ("retention", "alpha", "gamma"):
        assert flags[f"{first} {parameter} at an end"] is end


@pytest.mark.parametrize(
    ("value", "interval", "end"),
    [
        (0.0, ridge.THETA, True),
        (0.9, ridge.THETA, True),
        (0.9 * (1.0 - 1e-12), ridge.THETA, True),
        (0.45, ridge.THETA, False),
        (1e-4, ridge.PENALTY, True),
        (1e2, ridge.PENALTY, True),
        (1.0, ridge.PENALTY, False),
        (0.3 * (1.0 + 1e-6), ridge.GAMMA, False),
    ],
)
def test_a_parameter_is_at_an_end_of_its_interval_but_for_rounding(value, interval, end):
    assert ridge._at_an_end(value, interval) is end


@pytest.mark.parametrize("name", ["ridge", "chc"])
def test_each_response_is_held_by_the_bench_to_the_fits_own_parts(made, observation, name):
    response = made[name][0][name]["response"]
    assert residual(response, observation) <= TOLERANCE
    returns = drawn_returns(response, observation)
    assert np.all(np.isfinite(returns.roi))
    assert np.all(np.isfinite(returns.marginal))


def test_the_ridge_response_is_one_draw_of_hill_curves_on_unnormalised_kernels(made, observation):
    response = made["ridge"][0]["ridge"]["response"]
    parsed = drawn(response, observation.channels)
    assert (response["length"], response["normalized"]) == (observation.sales.size, False)
    for name in observation.channels:
        assert parsed[name].curves == ("Hill",)


def test_the_chc_response_is_its_readings_tanh_curves_on_the_exports_kernel(
    chc_fitted, observation
):
    response = chc_fitted.response
    parsed = drawn(response, observation.channels)
    assert (response["length"], response["normalized"]) == (observation.kernel_length, True)
    for j, name in enumerate(observation.channels):
        assert parsed[name].curves == ("Tanh",) * len(chc_fitted.readings)
        assert parsed[name].coefficient.tolist() == [
            float(r[j].coefficient) for r in chc_fitted.readings
        ]


def test_the_chc_plug_in_is_the_mean_of_its_readings_each_at_or_above_0(chc_fitted):
    """A smoke run draws 100, fewer than the readings' 400, so every draw is a reading, once."""
    sampler = chc_arm.SMOKE_SAMPLER
    assert len(chc_fitted.readings) == sampler.chains * sampler.draws
    drawn_ = np.array([[float(c.coefficient) for c in r] for r in chc_fitted.readings])
    assert len({tuple(row) for row in drawn_}) == len(drawn_)
    assert np.all(drawn_ >= 0.0)
    plug_in = [float(c.coefficient) for c in chc_fitted.channels]
    np.testing.assert_allclose(plug_in, drawn_.mean(axis=0), rtol=1e-12, atol=0.0)


def test_the_chc_claim_is_the_mean_and_central_90_of_its_readings_gains(chc_fitted, observation):
    planned = chc_arm.Chc().plan(chc_fitted, "chc", observation)
    gains = reading.gains(chc_fitted.readings, observation, planned.weekly)
    assert planned.gain == pytest.approx(float(np.mean(gains)), rel=1e-12, abs=0.0)
    assert planned.gain_interval == pytest.approx(
        tuple(np.quantile(gains, [0.05, 0.95])), rel=1e-12, abs=0.0
    )
    expected = allocate(
        list(chc_fitted.channels),
        observation.budget,
        observation.planned,
        lower=observation.lower,
        upper=observation.upper,
        history=observation.spend,
    ).spend
    np.testing.assert_array_equal(planned.weekly, np.asarray(expected))
    assert planned.ess == chc_fitted.ess == chc_fitted.flags["min ess bulk"]


@pytest.mark.parametrize("name", ["ridge", "chc"])
def test_each_arm_hands_allocate_the_exports_budget_box_and_history(
    request, observation, monkeypatch, name
):
    """On this world the box and the budget pin both plans, two channels at their floors, so the
    history moves neither past rounding: each arm is held to handing it on."""
    module, arm = {"ridge": (ridge, ridge.Ridge()), "chc": (chc_arm, chc_arm.Chc())}[name]
    fitted = request.getfixturevalue(f"{name}_fitted")
    seen = []

    def spy(channels, budget, periods, **kwargs):
        seen.append((budget, periods, kwargs))
        return allocate(channels, budget, periods, **kwargs)

    monkeypatch.setattr(module, "allocate", spy)
    arm.plan(fitted, name, observation)
    ((budget, periods, kwargs),) = seen
    assert (budget, periods) == (observation.budget, observation.planned)
    assert kwargs.keys() == {"lower", "upper", "history"}
    assert kwargs["lower"] is observation.lower
    assert kwargs["upper"] is observation.upper
    assert kwargs["history"] is observation.spend


def test_the_chc_draws_are_sign_held_by_its_sampler_and_flagged_as_chc_reads_them(
    observation, monkeypatch
):
    """Every channel's coefficient held at or above 0, the sampler's chains and draws at the
    record's seed, and the flags what CHC's check says of those draws."""
    seen = []

    def sample(fit, signs, draws, seed, **kwargs):
        seen.append(
            (signs, draws, seed, kwargs, constrained_sample(fit, signs, draws, seed, **kwargs))
        )
        return seen[-1][-1]

    monkeypatch.setattr(chc_arm, "constrained_sample", sample)
    fitted = chc_arm.Chc().fit(observation, smoke=True)
    ((signs, draws, seed, kwargs, signed),) = seen
    sampler = chc_arm.SMOKE_SAMPLER
    assert signs == dict.fromkeys(range(len(observation.channels)), "positive")
    assert (draws, kwargs) == (sampler.draws, {"chains": sampler.chains, "warmup": sampler.warmup})
    assert seed == fitted.settings["seed"]
    assert fitted.flags["fitted not mixed"] == int(not signed.mixed)
    assert fitted.flags["min ess bulk"] == fitted.ess == signed.ess


def test_the_report_reads_the_chc_arms_chains_by_chcs_check_and_the_ridge_as_no_chain(made):
    chc_record, ridge_record = made["chc"][0]["chc"], made["ridge"][0]["ridge"]
    assert report.past(chc_record) is (chc_record["flags"]["fitted not mixed"] != 0)
    assert report.past(ridge_record) is None


TRUTH = ((0.3, 0.5), (0.7, 2.0), (0.1, 1.0))
"""Each channel's retention and scale multiple in the made world, on the arm's grid."""


def _world(observation, coefficients, noise=1.0):
    """`observation` with its sales made by `TRUTH`'s channels at `coefficients`, a level and
    normal noise of `noise`."""
    means = chc_arm._means(observation)
    sales = np.full(observation.sales.shape, 1000.0)
    for j, ((retention, multiple), coefficient) in enumerate(zip(TRUTH, coefficients, strict=True)):
        made = chc_arm._channel(retention, multiple * means[j], 1.0, observation.kernel_length)
        sales += coefficient * np.asarray(made(observation.spend[:, j]))
    sales += np.random.default_rng(0).normal(0.0, noise, sales.shape)
    return dataclasses.replace(observation, sales=sales)


def _likelihood(world, transforms):
    """The static model's log-likelihood of `world` with each channel's curve at `transforms`."""
    means = chc_arm._means(world)
    columns = [
        np.asarray(chc_arm._channel(r, m * means[j], 1.0, world.kernel_length)(world.spend[:, j]))
        for j, (r, m) in enumerate(transforms)
    ]
    x = np.column_stack(columns + [np.asarray(world.controls[c]) for c in world.controls])
    sales = np.asarray(world.sales, dtype=float)
    return forward_filter(chc_arm._model(sales, x, len(transforms)), sales, x).log_likelihood


def test_the_chc_search_finds_a_model_as_likely_as_the_one_a_world_was_made_with(observation):
    """From the grid's middle alone a polish stops on a lower peak; the starts and polishes reach
    one at least as likely as the world's own curves, every polish ending by moving none."""
    world = _world(observation, (300.0, 500.0, 200.0))
    means, truth = chc_arm._means(world), _likelihood(world, TRUTH)
    alone = chc_arm.choose(world, chc_arm.GRID, means, chc_arm.Search(starts=1, polish=1), 0)
    assert _likelihood(world, alone.transforms) < truth - 10.0
    choice = chc_arm.choose(world, chc_arm.GRID, means, chc_arm.SEARCH, 0)
    assert choice.fit.log_likelihood == _likelihood(world, choice.transforms) >= truth
    assert choice.converged
    assert 2 <= choice.sweeps < chc_arm.SWEEPS
    assert choice.evaluations > chc_arm.SEARCH.starts


def test_the_chc_search_polishes_its_most_likely_starts(observation, monkeypatch):
    """With a polish that moves nothing, every start is handed to it most likely first, a search
    that polishes 2 polishes the first 2 of them, and the choice is the most likely start."""
    world = _world(observation, (300.0, 500.0, 200.0))
    means = chc_arm._means(world)
    seen = []

    def still(start, points, likelihood):
        seen.append((start, likelihood(start)))
        return start, 1, True

    monkeypatch.setattr(chc_arm, "_polish", still)
    every = chc_arm.choose(world, chc_arm.GRID, means, chc_arm.Search(starts=8, polish=8), 0)
    handed = seen.copy()
    seen.clear()
    two = chc_arm.choose(world, chc_arm.GRID, means, chc_arm.Search(starts=8, polish=2), 0)
    likelihoods = [likelihood for _, likelihood in handed]
    assert likelihoods == sorted(likelihoods, reverse=True)
    assert likelihoods[0] > likelihoods[-1]
    assert seen == handed[:2]
    assert every.transforms == two.transforms == handed[0][0]


def test_the_chc_draws_hold_a_channel_that_returns_nothing_at_or_above_0(observation):
    """The third channel returns nothing, so its unsigned posterior sits about 0: held, its draws
    are at or above 0, and not all at it."""
    world = _world(observation, (300.0, 500.0, 0.0), noise=20.0)
    fitted = chc_arm.Chc().fit(world, smoke=True)
    third = np.array([float(r[2].coefficient) for r in fitted.readings])
    assert np.all(third >= 0.0)
    assert np.any(third > 0.0)


def test_the_chc_prior_scales_with_each_column_so_a_curves_scale_moves_the_fit_alone(observation):
    """The level starts at the sales' mean; each standard deviation is ten times its coordinate's
    scale, `s` for the level and the season, `s / T` for the slope and `s / rms(x)` for a column;
    so a column times `c` leaves the filter's log-likelihood as it was."""
    sales = np.asarray(observation.sales, dtype=float)
    x = np.column_stack(
        [np.asarray(observation.spend, dtype=float)[:, :2], observation.controls["price"]]
    )
    model = chc_arm._model(sales, x, 2)
    s2, weeks = float(np.var(sales, ddof=1)), sales.size
    scales = [s2, s2 / weeks**2, *[s2] * 2 * chc_arm.HARMONICS, *(s2 / np.mean(x**2, axis=0))]
    np.testing.assert_allclose(
        np.diag(model.prior.covariance), chc_arm.SPREAD * np.array(scales), rtol=1e-12
    )
    assert np.count_nonzero(model.prior.covariance - np.diag(np.diag(model.prior.covariance))) == 0
    assert model.prior.mean.tolist() == [float(np.mean(sales))] + [0.0] * (len(scales) - 1)
    assert (model.prior.scale, model.prior.dof) == (s2, chc_arm.PRIOR_DOF)
    scaled = x * np.array([7.0, 0.01, 3.0])
    once = forward_filter(model, sales, x).log_likelihood
    again = forward_filter(chc_arm._model(sales, scaled, 2), sales, scaled).log_likelihood
    assert again == pytest.approx(once, rel=1e-9, abs=0.0)


def test_the_chc_flags_name_each_channels_choice_and_its_grids_ends(chc_fitted, observation):
    flags, grid = chc_fitted.flags, chc_arm.SMOKE_GRID
    for name in observation.channels:
        retention, scale = flags[f"{name} retention"], flags[f"{name} scale"]
        assert retention in grid.retentions
        assert scale in grid.scales
        assert flags[f"{name} retention at an end"] is (retention != grid.retentions[1])
        assert flags[f"{name} scale at an end"] is (scale != grid.scales[1])
    assert flags["fitted not mixed"] in (0, 1)


def _flat_world(observation):
    """`observation` with pla's spend its mean in every week and meta's swinging half its mean
    either side of it over 13 weeks, never 0; tv's as drawn; the sales made by `TRUTH`."""
    spend = np.asarray(observation.spend, dtype=float).copy()
    weeks = np.arange(spend.shape[0])
    spend[:, 0] = spend[:, 0].mean()
    spend[:, 1] = spend[:, 1].mean() * (1.0 + 0.5 * np.sin(2.0 * np.pi * weeks / 13.0))
    return _world(dataclasses.replace(observation, spend=spend), (300.0, 500.0, 200.0))


def test_the_chc_flags_name_a_channel_whose_spend_never_moves_flat(observation):
    """Whatever curve the search chooses for pla, whose spend never moves, its column varies by at
    most 1 % of itself over the periods read, so its record flags it `flat` beside its grid's ends,
    and the flags read it as an alarm."""
    world = _flat_world(observation)
    record = records(chc_arm.Chc(), world, smoke=True)["chc"]
    flags = record["flags"]
    assert flags["pla column cv"] < 0.1 * chc_arm.FLAT
    assert flags["pla flat"] is True
    assert {"pla retention at an end", "pla scale at an end"} <= set(flags)
    assert f.numbers(record)["pla flat"] == 1.0


@pytest.mark.parametrize(("scale", "flat"), [(0.25, True), (4.0, False)])
def test_the_chc_flat_flag_reads_the_chosen_column_not_the_spend(
    observation, monkeypatch, scale, flat
):
    """meta's spend swings half its mean either side of it and is never 0: at the grid's most
    saturated scale its column varies by under 0.007 of its mean and is flat, at its least by
    0.29 and is not. pla's spend never moves, so its column is flat at either."""
    monkeypatch.setattr(chc_arm, "SMOKE_GRID", chc_arm.Grid((0.5,), (scale,)))
    flags = chc_arm.Chc().fit(_flat_world(observation), smoke=True).flags
    assert flags["meta scale"] == scale
    assert flags["meta flat"] is flat
    assert (flags["meta column cv"] < chc_arm.FLAT) is flat
    assert flags["pla flat"] is True


def test_a_grid_starts_at_its_middles():
    assert chc_arm.Grid((0.1, 0.2, 0.3), (1.0, 2.0, 4.0, 8.0)).start == (0.2, 4.0)
    assert chc_arm.GRID.start == (0.5, 1.0)


def test_the_chc_arm_refuses_a_channel_that_never_spends(observation):
    spend = np.asarray(observation.spend, dtype=float).copy()
    spend[:, 1] = 0.0
    with pytest.raises(ValueError, match=r"\['meta'\] spend nothing"):
        chc_arm._means(dataclasses.replace(observation, spend=spend))
