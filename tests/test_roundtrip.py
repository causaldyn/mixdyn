"""mixdyn.api.serialize: every engine's fit, and every result read off one, saved and loaded back
the same to the bit.

Each engine is fitted on a tiny world, saved and loaded. The loaded object equals the fitted one
field by field, arrays exactly, their dtypes too, NaN where NaN; and what is read off it, each
channel's returns, each plan, each interval's ends and each forecast, equals what is read off the
fitted one exactly, not approximately. A file of another format version, or naming a class the
format does not hold, fails to load with a message that says so.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import inspect
import io
import json
import math
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mixdyn import NATIONAL, Baseline, Cadence, Channel, Geo, Lever, LeverKind, MediaPanel, Revenue
from mixdyn.api import Decision, Estimand, Schema, Spend, Unfilled, estimands
from mixdyn.api.serialize import FORMAT_VERSION, REGISTRY, FormatError, load, save
from mixdyn.calibration import LiftEstimate
from mixdyn.calibration.anchors import Anchored, ChannelTests, Curves
from mixdyn.data import Combine, Gap, Issue, IssueCode, RecurringEvents, Severity, read_panel
from mixdyn.data.feedback import SpendFeedback
from mixdyn.decisions import (
    Costs,
    CvarPlan,
    FlightPlan,
    GeoLimits,
    GeoPlan,
    MarginalReturnTarget,
    PosteriorPlan,
    Range,
    ReturnOnSpendTarget,
    Returns,
    ReturnTarget,
    RobustPlan,
    Scenario,
    Undefined,
    compare_scenarios,
    plan_breakdown,
    plan_cvar,
    plan_flighting,
    plan_geos,
    plan_posterior,
    plan_robust,
    plan_table,
    readings,
    response_curves,
    returns,
    worths,
)
from mixdyn.decisions.geo_plan import MixturePlan
from mixdyn.decisions.margin_plan import Shrunk
from mixdyn.experiments import (
    After,
    Candidate,
    Gaussian,
    GeoTest,
    Gross,
    GroupNoise,
    Net,
    PlaceboNoise,
    Refit,
    Reweight,
    Within,
    assign,
    audit_power,
    balance,
    carryover_order,
    choose_tests,
    designs,
    pair_geos,
    power,
    read_did,
    read_sdid,
    read_switchback,
    stratified_design,
    switchback_design,
    synthetic_did,
    value_of_tests,
)
from mixdyn.inference import loo, nuts, pathfinder, psis, sample
from mixdyn.inference.diagnostics import health, prior_predictive
from mixdyn.inference.sensitivity import Benchmark, msm_sensitivity, ridge_sensitivity
from mixdyn.models import (
    DiscountGrid,
    TransformGrid,
    choose_dlm,
    decompose_dlm,
    diagnose_dlm,
    fit_dlm,
    fit_geo_dlm,
    forecast_dlm,
    forecast_draws,
    geo_effects,
)
from mixdyn.models.bayes_geo import BayesGeoModel, BayesGeoSpec, fit_bayes_geo
from mixdyn.models.bayes_mmm import (
    BayesMedia,
    BayesModel,
    BayesSpec,
    HalfNormalRoi,
    RoiPrior,
    UniformStart,
    fit_bayes,
    forecast_bayes,
    holdout_bayes,
    refresh_bayes,
)
from mixdyn.models.demand import DemandProxy
from mixdyn.models.geo_eb import CellResponse, GeoEbFit, Mixture
from mixdyn.models.margin_mmm import Curve, CurvedFit, MarginFit, NoiseMixture, Projection
from mixdyn.models.nested import NestedModel, fit_nested
from mixdyn.models.ridge import Anchor, RidgePath
from mixdyn.models.ridge_mmm import RidgeSpec, fit_ridge
from mixdyn.models.ridge_search import (
    Folds,
    GeometricSpace,
    Interval,
    MediaSpace,
    RidgeSpace,
    search_ridge,
)
from mixdyn.transforms import HillShape, TanhShape
from ridge_worlds import world_and_tests
from test_dlm_lift import lifts_of, media, media_spec
from test_geo_dlm import SPEC as GEO_DLM_SPEC
from test_geo_dlm import hierarchy
from test_nested import SPEC as NESTED_SPEC
from test_nested import TV_ON_SEARCHES, mediation_world
from test_public_names import modules

SHORT: dict[str, Any] = {"chains": 2, "warmup": 60, "draws": 40, "start": UniformStart()}
"""A posterior too short to read, long enough to have every array a long one has."""
_STEP = "a working step of the sampler or the solver, which no fit holds and no caller reads"
_MODEL = "a compiled density of private parts, which a fit is drawn from and does not hold"
_ARGUMENT = "an argument of a reader of data, which no field holds"
_RULE = (
    "a part of a planning rule the scorecard's arms run (ADRs 0086 to 0093), whose records the"
    " harness writes; no file needs it yet, and registering it later moves no version"
)
NOT_SAVED: Mapping[type, str] = MappingProxyType(
    {
        Refit: "holds `fit`, a function, which a file does not carry",
        Schema: "holds `data`, a type, which a file does not carry",
        nuts.Point: _STEP,
        nuts.Noise: _STEP,
        nuts.Transition: _STEP,
        pathfinder.Path: _STEP,
        pathfinder.Resampled: _STEP,
        Anchor: _STEP,
        RidgePath: _STEP,
        BayesModel: _MODEL,
        BayesGeoModel: _MODEL,
        NestedModel: _MODEL,
        Gap: _ARGUMENT,
        Combine: _ARGUMENT,
        Anchored: _RULE,
        ChannelTests: _RULE,
        Curves: _RULE,
        MixturePlan: _RULE,
        Shrunk: _RULE,
        CellResponse: _RULE,
        GeoEbFit: _RULE,
        Mixture: _RULE,
        Curve: _RULE,
        CurvedFit: _RULE,
        MarginFit: _RULE,
        NoiseMixture: _RULE,
        Projection: _RULE,
    }
)
"""Every public class `REGISTRY` leaves out, and why. Adding one to the registry later moves no
format version; taking one out does, so what no file needs stays out."""


def identical(a: Any, b: Any, where: str = "obj") -> list[str]:
    """Where `a` and `b` differ: in type, a field, an item, an array's dtype or shape, or any bit
    of a value."""
    if type(a) is not type(b) and not (isinstance(a, jax.Array) and isinstance(b, jax.Array)):
        return [f"{where}: {type(a).__name__} against {type(b).__name__}"]
    if isinstance(a, jax.Array | np.ndarray | np.generic):
        a, b = np.asarray(a), np.asarray(b)
        if a.dtype != b.dtype or a.shape != b.shape:
            return [f"{where}: {a.dtype}{a.shape} against {b.dtype}{b.shape}"]
        return [] if a.tobytes() == b.tobytes() else [f"{where}: the values differ"]
    if isinstance(a, float):
        return [] if a.hex() == b.hex() else [f"{where}: {a!r} against {b!r}"]
    if isinstance(a, tuple | list):
        if len(a) != len(b):
            return [f"{where}: {len(a)} items against {len(b)}"]
        return [
            d
            for i, (x, y) in enumerate(zip(a, b, strict=True))
            for d in identical(x, y, f"{where}[{i}]")
        ]
    if isinstance(a, Mapping):
        if list(a) != list(b):
            return [f"{where}: keys {list(a)} against {list(b)}"]
        return [d for k in a for d in identical(a[k], b[k], f"{where}[{k!r}]")]
    if dataclasses.is_dataclass(a):
        return [
            d
            for f in dataclasses.fields(a)
            for d in identical(getattr(a, f.name), getattr(b, f.name), f"{where}.{f.name}")
        ]
    if isinstance(a, enum.Enum | str | int | bool | frozenset | dt.date) or a is None:
        return [] if a == b else [f"{where}: {a!r} against {b!r}"]
    return [f"{where}: no comparison for {type(a).__name__}"]


def roundtrip(obj: Any, tmp_path: Path) -> Any:
    path = tmp_path / "saved.mixdyn"
    save(obj, path)
    loaded = load(path)
    assert identical(obj, loaded) == []
    return loaded


def spend_of(panel: MediaPanel, g: int) -> dict[str, np.ndarray]:
    """Each channel's spend series in geo `g`, by its name."""
    return {str(c): panel.spend[g, :, j] for j, c in enumerate(panel.channels)}


def channel_returns(channels: Mapping[str, Any], spend: Mapping[str, np.ndarray]) -> dict:
    """Each channel's `returns` over its spend series."""
    return {name: returns(channel, spend[name]) for name, channel in channels.items()}


# --------------------------------------------------------------------------------------- worlds


def geo_panel(geos: int = 4, weeks: int = 40) -> MediaPanel:
    """Geos of a KPI moved by two channels' spend in proportion, a population each."""
    rng = np.random.default_rng(20261031)
    people = 1e4 * np.exp(0.4 * rng.standard_normal(geos))
    spend = people[:, None, None] * rng.gamma(3.0, 0.002, (geos, weeks, 2))
    kpi = people[:, None] * (1.0 + 0.1 * rng.standard_normal((geos, weeks)))
    kpi += spend @ np.array([1.5, 0.8])
    return MediaPanel(
        dates=np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(weeks),
        cadence=Cadence.WEEK,
        geos=tuple(Geo(f"g{g}") for g in range(geos)),
        channels=(Channel("tv"), Channel("search")),
        controls=(),
        kpi=kpi,
        kpi_unit=Revenue(),
        spend=spend,
        currency="EUR",
        control_values=np.zeros((geos, weeks, 0)),
        filled=np.zeros((geos, weeks), dtype=bool),
        population=people,
    )


@pytest.fixture(scope="module")
def ridge() -> tuple[MediaPanel, Any]:
    """The ridge engine calibrated by geo tests: a fit that holds what its tests read."""
    panel, media_, _, tests, _ = world_and_tests(10.0)
    return panel, fit_ridge(panel, NATIONAL, RidgeSpec(tuple(media_), (), 1e-3), tests=tests)


@pytest.fixture(scope="module")
def bayes() -> tuple[MediaPanel, Any]:
    """The Bayesian model, its likelihood holding a lift estimate."""
    panel = mediation_world()
    spec = BayesSpec((BayesMedia("tv", RoiPrior(0.0, 1.0)), BayesMedia("search")))
    fit = fit_bayes(panel, NATIONAL, spec, seed=3, estimates={"tv": [TV_ON_SEARCHES]}, **SHORT)
    return panel, fit


# ---------------------------------------------------------------------------------------- tests


def test_the_ridge_fit_with_its_tests_returns_the_same(
    ridge: tuple[MediaPanel, Any], tmp_path: Path
) -> None:
    panel, fit = ridge
    loaded = roundtrip(fit, tmp_path)
    assert fit.calibration is not None
    spend = spend_of(panel, 0)
    assert (
        identical(channel_returns(fit.channels(), spend), channel_returns(loaded.channels(), spend))
        == []
    )
    assert (loaded.nrmse(), loaded.decomp_rssd()) == (fit.nrmse(), fit.decomp_rssd())


def test_the_ridge_search_returns_the_same(ridge: tuple[MediaPanel, Any], tmp_path: Path) -> None:
    panel, _ = ridge
    space = RidgeSpace(
        media=tuple(
            MediaSpace(
                c, GeometricSpace(Interval(0.0, 0.9)), Interval(1.0, 3.0), Interval(0.3, 1.0)
            )
            for c in panel.channels
        ),
        controls=(),
        penalty=Interval(1e-6, 1e-2),
    )
    search = search_ridge(panel, NATIONAL, space, Folds(2, 8), seed=3, starts=8, polish=1)
    loaded = roundtrip(search, tmp_path)
    spend = spend_of(panel, 0)
    assert (
        identical(
            channel_returns(search.fit.channels(), spend),
            channel_returns(loaded.fit.channels(), spend),
        )
        == []
    )


def test_a_plan_over_geos_is_the_same_planned_off_the_loaded_fit(
    ridge: tuple[MediaPanel, Any], tmp_path: Path
) -> None:
    panel, fit = ridge

    def plan(f: Any) -> Any:
        current = panel.spend[0, -8:].mean(axis=0)
        cells = {
            (NATIONAL, c): Range(0.5 * float(x), 2.0 * float(x))
            for c, x in zip(panel.channels, current, strict=True)
        }
        responses = {NATIONAL: f.channels()}
        return plan_geos(panel, responses, 8 * current.sum(), 8, GeoLimits(cells=cells))

    planned = plan(fit)
    assert identical(planned, plan(roundtrip(fit, tmp_path))) == []
    assert identical(planned, roundtrip(planned, tmp_path)) == []


def test_the_bayesian_fit_with_its_estimate_returns_the_same(
    bayes: tuple[MediaPanel, Any], tmp_path: Path
) -> None:
    panel, fit = bayes
    loaded = roundtrip(fit, tmp_path)
    spend = spend_of(panel, 0)
    assert (
        identical(channel_returns(fit.channels(), spend), channel_returns(loaded.channels(), spend))
        == []
    )
    assert (
        identical(
            [channel_returns(d, spend) for d in fit.channel_draws(5)],
            [channel_returns(d, spend) for d in loaded.channel_draws(5)],
        )
        == []
    )


def test_the_posterior_s_plan_and_its_table_are_the_same(
    bayes: tuple[MediaPanel, Any], tmp_path: Path
) -> None:
    panel, fit = bayes
    loaded = roundtrip(fit, tmp_path)
    current = panel.spend[0, -8:].mean(axis=0)
    limits = {
        c: Range(0.5 * float(x), 2.0 * float(x))
        for c, x in zip(panel.channels, current, strict=True)
    }

    def plan(f: Any) -> tuple[Any, Any]:
        read = readings(f, draws=20)
        planned = plan_posterior(panel, NATIONAL, read, 8 * current.sum(), 8, limits, grid=64)
        return planned, plan_table(panel, NATIONAL, read, planned.spend, 8, limits=limits)

    (planned, table), (planned_again, table_again) = plan(fit), plan(loaded)
    assert identical(planned, planned_again) == []
    assert identical(table, table_again) == []
    assert identical(table, roundtrip(table, tmp_path)) == []
    assert identical(planned, roundtrip(planned, tmp_path)) == []
    assert identical(readings(fit), roundtrip(readings(fit), tmp_path)) == []


def _off_the_posterior(panel: MediaPanel, fit: Any) -> dict[str, Any]:
    """Every result a posterior's decisions read off one geo's fit."""
    current = panel.spend[0, -4:].mean(axis=0)
    limits = {
        c: Range(0.5 * float(x), 2.0 * float(x))
        for c, x in zip(panel.channels, current, strict=True)
    }
    read = readings(fit, draws=10)
    plan = {"tv": np.full(4, 50.0), "search": np.full(4, 30.0)}
    worth = worths(panel, NATIONAL, read, 4, limits, grid=16)
    return {
        "forecast": forecast_bayes(fit, 4, plan, draws=10, seed=1),
        "scenarios": compare_scenarios(
            fit,
            [Scenario("now", plan), Scenario("more", {k: 2.0 * v for k, v in plan.items()})],
            baseline="now",
            horizon=4,
            draws=10,
            seed=1,
        ),
        "worths": worth,
        "curves": response_curves(worth, read, level=0.8),
        "breakdown": plan_breakdown(panel, NATIONAL, read, 1.1 * current, 4),
        "cvar": plan_cvar(
            panel, NATIONAL, read, 4 * current.sum(), 4, limits, against=current, draws=10
        ),
        "robust": plan_robust(
            panel, NATIONAL, read, 4 * current.sum(), 4, limits, scale=1.1, grid=32
        ),
        "flighting": plan_flighting(panel, NATIONAL, read, 4 * current.sum(), 4, limits),
    }


def test_each_decision_read_off_the_loaded_posterior_is_the_same(
    bayes: tuple[MediaPanel, Any], tmp_path: Path
) -> None:
    panel, fit = bayes
    results = _off_the_posterior(panel, fit)
    again = _off_the_posterior(panel, roundtrip(fit, tmp_path))
    for name, result in results.items():
        assert identical(result, again[name], name) == []
        assert identical(result, roundtrip(result, tmp_path), name) == []


def test_a_holdout_and_a_refresh_are_the_same(
    bayes: tuple[MediaPanel, Any], tmp_path: Path
) -> None:
    panel, fit = bayes
    held = holdout_bayes(panel, NATIONAL, fit.spec, periods=6, seed=5, **SHORT)
    roundtrip(held, tmp_path)
    refreshed = refresh_bayes(panel, NATIONAL, fit, seed=4, **SHORT)
    loaded = roundtrip(refreshed, tmp_path)
    assert [d.leaves for d in loaded.drift] == [d.leaves for d in refreshed.drift]


def test_the_bayesian_fit_over_geos_returns_the_same(tmp_path: Path) -> None:
    panel = geo_panel()
    spec = BayesGeoSpec((BayesMedia("tv"), BayesMedia("search")))
    fit = fit_bayes_geo(panel, spec, seed=4, **SHORT)
    loaded = roundtrip(fit, tmp_path)
    for g, geo in enumerate(panel.geos):
        spend = spend_of(panel, g)
        assert (
            identical(
                channel_returns(fit.channels(geo), spend),
                channel_returns(loaded.channels(geo), spend),
            )
            == []
        )


def test_the_nested_fit_returns_the_same(tmp_path: Path) -> None:
    panel = mediation_world()
    fit = fit_nested(panel, NATIONAL, NESTED_SPEC, seed=5, **SHORT)
    loaded = roundtrip(fit, tmp_path)
    assert identical(fit.roi(), loaded.roi()) == []
    spend = spend_of(panel, 0)
    assert (
        identical(channel_returns(fit.channels(), spend), channel_returns(loaded.channels(), spend))
        == []
    )


@pytest.mark.parametrize("signed", [False, True], ids=["unsigned", "signed"])
def test_the_dlm_with_its_lift_tests_returns_and_forecasts_the_same(
    signed: bool, tmp_path: Path
) -> None:
    data = media()
    spec = media_spec("positive", "positive") if signed else media_spec()
    fit = fit_dlm(data, NATIONAL, spec, estimates=lifts_of(data), draws=60, seed=2)
    assert fit.lifted is not None
    assert (fit.signed is not None) is signed
    loaded = roundtrip(fit, tmp_path)
    assert loaded.log_likelihood == fit.log_likelihood
    assert identical(fit.last(), loaded.last()) == []
    spend = spend_of(data, 0)
    assert (
        identical(channel_returns(fit.channels(), spend), channel_returns(loaded.channels(), spend))
        == []
    )
    plan = {e.column: np.full(4, 30.0) for e in spec.effects}
    if signed:
        forecast = forecast_draws(fit, 4, plan, draws=20, seed=1)
        assert identical(forecast, forecast_draws(loaded, 4, plan, draws=20, seed=1)) == []
    else:
        forecast = forecast_dlm(fit, 4, plan)
        assert identical(forecast, forecast_dlm(loaded, 4, plan)) == []
    assert identical(forecast, roundtrip(forecast, tmp_path)) == []


def test_what_is_read_off_a_loaded_dlm_is_the_same(tmp_path: Path) -> None:
    data = media()
    fit = fit_dlm(data, NATIONAL, media_spec())
    loaded = roundtrip(fit, tmp_path)
    for result, again in [
        (decompose_dlm(fit, draws=20, seed=3), decompose_dlm(loaded, draws=20, seed=3)),
        (diagnose_dlm(fit), diagnose_dlm(loaded)),
    ]:
        assert identical(result, again) == []
        assert identical(result, roundtrip(result, tmp_path)) == []
    choice = choose_dlm(data, NATIONAL, media_spec(), DiscountGrid(trend=(0.95, 1.0)))
    assert identical(choice, roundtrip(choice, tmp_path)) == []


def test_the_dlm_over_geos_returns_the_same(tmp_path: Path) -> None:
    panel, _ = hierarchy(7, geos=3, weeks=40)
    fit = fit_geo_dlm(panel, GEO_DLM_SPEC, draws=8)
    effects = geo_effects(fit)
    assert identical(effects, geo_effects(roundtrip(fit, tmp_path))) == []
    assert identical(effects, roundtrip(effects, tmp_path)) == []


def test_a_geo_test_s_reading_and_its_estimate_are_the_same(tmp_path: Path) -> None:
    panel = geo_panel(geos=6, weeks=40)
    planned = np.full(8, 50.0)
    design = GeoTest(
        Channel("tv"),
        panel.geos[0::2],
        panel.geos[1::2],
        panel.dates[36],
        panel.dates[39],
        np.r_[planned[:4], 2.0 * planned[4:]],
        planned,
        4,
    )
    estimate, reading = read_did(panel, design, warmup=8)
    assert identical(reading, roundtrip(reading, tmp_path)) == []
    assert identical(estimate, roundtrip(estimate, tmp_path)) == []
    assert identical(design, roundtrip(design, tmp_path)) == []
    assert identical(panel, roundtrip(panel, tmp_path)) == []


def test_what_is_read_off_a_posterior_before_a_test_runs_is_the_same(
    bayes: tuple[MediaPanel, Any], tmp_path: Path
) -> None:
    panel, fit = bayes
    read = readings(fit, draws=20)
    current = panel.spend[0, -4:].mean(axis=0)
    limits = {
        c: Range(0.5 * float(x), 2.0 * float(x))
        for c, x in zip(panel.channels, current, strict=True)
    }
    candidates = designs(
        panel,
        NATIONAL,
        panel.channels[:1],
        GroupNoise(5.0),
        when=Within(panel.dates.size - 8),
        multipliers=(0.0, 2.0),
        durations=(2,),
        cooldown=2,
        shares=(0.2,),
    )
    plan: tuple[Any, ...] = (panel, NATIONAL, read, candidates, 4 * current.sum(), 4, limits)
    values = value_of_tests(*plan, seed=1, draws=8, grid=16)
    for result in (
        health(fit, panel, NATIONAL, seed=1, prior_draws=50),
        prior_predictive(panel, NATIONAL, fit.spec, draws=20, seed=1),
        msm_sensitivity(fit, "tv"),
        values,
        values.values[0],
        values.values[0].candidates[0],
        choose_tests(*plan, test_budget=1e9, seed=1, draws=8, grid=16),
    ):
        roundtrip(result, tmp_path)


def test_a_ridge_fit_s_sensitivity_and_its_bound_are_the_same(
    ridge: tuple[MediaPanel, Any], tmp_path: Path
) -> None:
    panel, fit = ridge
    sensitivity = ridge_sensitivity(fit, str(panel.channels[0]))
    roundtrip(sensitivity, tmp_path)
    roundtrip(sensitivity.bound(0.1, 0.1, label="a confounder"), tmp_path)


def test_a_geo_test_s_design_its_power_and_its_synthetic_reading_are_the_same(
    tmp_path: Path,
) -> None:
    panel = geo_panel(geos=6, weeks=40)
    blocks = pair_geos(np.arange(6.0))
    design = assign(blocks, 1)
    planned = np.full(8, 50.0)
    estimate, reading = read_sdid(
        panel,
        panel.geos[:2],
        panel.dates[32],
        panel.dates[39],
        spend=np.r_[planned[:4], 2.0 * planned[4:]],
        planned=planned,
        replications=5,
    )
    (estimand,) = estimands(panel, {"tv": [estimate]})
    assert isinstance(estimand.end, dt.date)
    for result in (
        blocks,
        design,
        balance(design, {"size": np.arange(6.0)}),
        power(panel.kpi, design, 4, np.array([0.0, 0.1]), pre=24, reps=5, seed=1),
        audit_power(panel.kpi, design, 2, np.array([0.1, 0.5]), pre=16),
        stratified_design(panel.kpi, 2, pre=16, seed=1),
        synthetic_did(panel.kpi, np.arange(6) < 2, 8),
        reading,
        estimand,
    ):
        roundtrip(result, tmp_path)


@pytest.mark.parametrize("method", ["randomization", "hac", "jackknife"])
def test_a_switchback_test_s_design_and_reading_are_the_same(method: Any, tmp_path: Path) -> None:
    panel = geo_panel(geos=2, weeks=40)
    design = switchback_design(panel.dates, 4, 1, seed=1)
    spend = np.where(np.repeat(design.on, 4), 100.0, 0.0)
    _, reading = read_switchback(panel.kpi, spend, design, method=method)
    roundtrip(design, tmp_path)
    roundtrip(reading, tmp_path)
    roundtrip(carryover_order(panel.kpi, design), tmp_path)


def test_what_the_sampler_and_the_comparison_return_is_the_same(tmp_path: Path) -> None:
    rng = np.random.default_rng(20261101)

    def standard_normal(x: np.ndarray) -> tuple[float, np.ndarray]:
        return float(-0.5 * x @ x), -x

    roundtrip(sample(standard_normal, np.full((2, 2), 0.1), seed=1, warmup=20, draws=10), tmp_path)
    roundtrip(loo(rng.normal(-1.0, 0.1, (2, 50, 5))), tmp_path)
    roundtrip(psis(rng.normal(size=200)), tmp_path)


def test_a_panel_saved_before_its_issues_named_a_channel_loads_naming_none(
    tmp_path: Path,
) -> None:
    """A file written before an issue named its channel has no `channel` field on its issues: the
    panel loads, each issue naming no channel, every other field as saved."""
    weeks = np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(30)
    panel = read_panel(
        {"week": weeks, "sales": 100.0 + np.arange(30.0), "tv_eur": np.full(30, 10.0)},
        date="week",
        kpi="sales",
        spend={"tv": "tv_eur"},
        cadence=Cadence.WEEK,
        kpi_unit=Revenue(),
        currency="EUR",
    )
    (flag,) = (i for i in panel.issues if i.code is IssueCode.CONSTANT)
    assert (flag.column, flag.channel) == ("tv_eur", "tv")

    def strip(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("$type") == "mixdyn.data.issues:Issue":
                node["fields"].pop("channel")
            for value in node.values():
                strip(value)
        elif isinstance(node, list):
            for value in node:
                strip(value)

    path = tmp_path / "before.mixdyn"
    save(panel, path)
    _rewritten(path, strip)
    loaded = load(path)
    unnamed = tuple(dataclasses.replace(i, channel=None) for i in panel.issues)
    assert identical(dataclasses.replace(panel, issues=unnamed), loaded) == []


def test_a_panel_read_with_a_warning_is_the_same(tmp_path: Path) -> None:
    """A panel holds what reading it found, each finding with the dates it is about."""
    dates = np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.delete(np.arange(30), 12)
    panel = read_panel(
        {"week": dates, "sales": 100.0 + np.arange(29.0), "tv": 10.0 + np.arange(29.0) % 3},
        date="week",
        kpi="sales",
        spend={"tv": "tv"},
        cadence=Cadence.WEEK,
        kpi_unit=Revenue(),
        currency="EUR",
        gaps=Gap.STEP,
    )
    assert any(issue.dates for issue in panel.issues)
    assert roundtrip(panel, tmp_path).issues == panel.issues


def _naming(channels: tuple[Channel, ...]) -> list[Any]:
    """Each plan that names the channels it moves on their prior alone, and the decision document,
    built by hand naming `channels`; a geo plan names them as the national geo's cells."""
    spend, current = np.array([60.0, 40.0]), np.array([50.0, 80.0])
    both = (Channel("tv"), Channel("retail"))
    unfilled = Unfilled("not recorded by hand")
    return [
        PosteriorPlan(both, 4, spend, current, 400.0, 12.5, 10.0, 64, channels),
        RobustPlan(both, 4, spend, current, 400.0, 1.1, 2.5, 1.0, 64, channels),
        CvarPlan(both, 4, spend, current, 400.0, 0.25, np.array([1.0, -0.5]), 0.1, 0.2, channels),
        FlightPlan(
            channels=both,
            periods=4,
            budget=400.0,
            reading="median",
            level=0.9,
            schedule=np.tile(spend, (4, 1)),
            continuous=spend,
            current=current,
            gains=np.ones((3, 2)),
            expected=np.ones(3),
            intervals=np.ones((3, 2)),
            edge=np.zeros(2),
            edge_interval=np.zeros(2),
            worth=12.5,
            bound=12.5,
            stopped="closed",
            limit=None,
            tolerance=1e-9,
            recommendation="continuous",
            unidentified=channels,
        ),
        GeoPlan(
            geos=(NATIONAL,),
            channels=both,
            periods=4,
            spend=spend[None],
            current=current[None],
            budget=400.0,
            gain=12.5,
            current_gain=10.0,
            shortfall=0.0,
            price=0.5,
            geo_prices=MappingProxyType({NATIONAL: 0.0}),
            channel_prices=MappingProxyType(dict.fromkeys(both, 0.0)),
            unidentified=tuple((NATIONAL, c) for c in channels),
        ),
        Decision(
            title="by hand",
            geo=str(NATIONAL),
            currency="EUR",
            periods=4,
            budget=400.0,
            gain=12.5,
            current_gain=10.0,
            spend=tuple(
                Spend(str(c), float(a), float(b))
                for c, a, b in zip(both, current, spend, strict=True)
            ),
            health=unfilled,
            reconciliation=unfilled,
            next_test=unfilled,
            events=(),
            compute=unfilled,
            unidentified=tuple(str(c) for c in channels),
        ),
    ]


@pytest.mark.parametrize(
    "held",
    [
        ReturnTarget(1500.0),
        MarginalReturnTarget(1.25),
        ReturnOnSpendTarget(2.0),
        Issue(IssueCode.BAD_SPEC, Severity.ERROR, "no rows"),
        Issue(IssueCode.OFF_GRID, Severity.WARNING, "off", "g0", "tv", (dt.date(2024, 1, 8),)),
        Issue(IssueCode.CONSTANT, Severity.WARNING, "flat", None, "retail_eur", (), "retail"),
        Estimand(
            "tv", (), 0.2, "added", "marginal change", 4, 8, 1.5, 1.0, None, None, None, "a test"
        ),
        After({Channel("tv"): 50.0}),
        Within(12),
        GroupNoise(5.0),
        PlaceboNoise(np.linspace(-1.0, 1.0, 40), pre=8),
        Candidate(Channel("tv"), np.array([0.0, 0.0]), np.array([1.0, 1.0]), 2, 0.5, 0.2, "dark"),
        Gross(),
        Net(0.5),
        Reweight(0.25),
        Gaussian(),
        Benchmark("price", 0.1, 0.2),
        SpendFeedback("national", "pla", 0.31, 0.05, 0.21, 0.41, 0.95, 155, 4),
        RecurringEvents(("weeks 47-48",), ((47, 48),), np.linspace(0.0, 1.0, 52), 3.5),
        DemandProxy("queries", "confounder", ("search",)),
        TransformGrid((0.0, 0.5), (1.0, 2.0), (TanhShape(), HillShape(1.5))),
        Lever(LeverKind.PRICE, Baseline.MAX),
        Lever(LeverKind.ORGANIC, 0.0),
        Costs(120.0, 40.0, 3.0, Undefined("one more currency unit earns no incremental KPI")),
        BayesMedia("tv", HalfNormalRoi(2.0, counts="carried")),
        PosteriorPlan(
            (Channel("tv"), Channel("retail")),
            4,
            np.array([60.0, 40.0]),
            np.array([50.0, 80.0]),
            400.0,
            12.5,
            10.0,
            64,
            (Channel("retail"),),
        ),
        *(_naming((Channel("retail"),))),
    ],
    ids=lambda held: type(held).__name__,
)
def test_what_a_caller_gives_by_hand_is_the_same(held: Any, tmp_path: Path) -> None:
    roundtrip(held, tmp_path)


@pytest.mark.parametrize(
    "held", _naming((Channel("retail"),)), ids=lambda held: type(held).__name__
)
def test_a_plan_saved_before_it_named_unidentified_channels_loads_naming_none(
    held: Any, tmp_path: Path
) -> None:
    """A file written before a plan named the channels it moves on their prior alone has no
    `unidentified` field: it loads, naming none, every other field as saved."""
    path = tmp_path / "before.mixdyn"
    save(held, path)
    _rewritten(path, lambda manifest: manifest["root"]["fields"].pop("unidentified"))
    loaded = load(path)
    assert loaded.unidentified == ()
    assert identical(dataclasses.replace(held, unidentified=()), loaded) == []


# ----------------------------------------------------------------------------- what it refuses


def _rewritten(path: Path, edit: Callable[[dict[str, Any]], None]) -> None:
    with zipfile.ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(entries["manifest.json"])
    edit(manifest)
    entries["manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)


def _saved(tmp_path: Path) -> Path:
    path = tmp_path / "estimate.mixdyn"
    save(LiftEstimate(np.zeros(4), np.ones(4), 2, -1.0, 0.5), path)
    return path


@pytest.mark.parametrize(
    ("version", "match"),
    [
        (FORMAT_VERSION + 1, "format version 2, written by mixdyn .*reads format version 1 only"),
        (0, "format version 0, .*with an earlier mixdyn"),
        ("1", "no valid format version, '1'"),
        (None, "no valid format version, None"),
        (True, "no valid format version, True"),
    ],
)
def test_a_corrupted_version_fails_to_load(version: Any, match: str, tmp_path: Path) -> None:
    path = _saved(tmp_path)
    _rewritten(path, lambda manifest: manifest.update(format_version=version))
    with pytest.raises(FormatError, match=match):
        load(path)


def test_a_file_naming_an_unknown_class_fails_to_load(tmp_path: Path) -> None:
    path = _saved(tmp_path)
    _rewritten(path, lambda manifest: manifest["root"].update({"$type": "os:system"}))
    with pytest.raises(FormatError, match="names a class the format does not hold, 'os:system'"):
        load(path)


def test_what_is_not_the_format_fails_to_load(tmp_path: Path) -> None:
    path = tmp_path / "not.mixdyn"
    path.write_text("{}")
    with pytest.raises(FormatError, match="not a zip archive"):
        load(path)
    path = _saved(tmp_path)
    _rewritten(path, lambda manifest: manifest.update(format="pickle"))
    with pytest.raises(FormatError, match="names no format 'mixdyn'"):
        load(path)


def test_a_subclass_of_a_json_value_keeps_its_class(tmp_path: Path) -> None:
    held = Returns(np.str_("tv"), np.float64(1.5), np.int64(2), math.inf, -0.0, math.nan)  # ty: ignore[invalid-argument-type]
    loaded = roundtrip(held, tmp_path)
    assert [type(getattr(loaded, f)) for f in ("spend", "contribution", "roi")] == [
        np.str_,
        np.float64,
        np.int64,
    ]


def test_a_damaged_archive_fails_to_load(tmp_path: Path) -> None:
    path = _saved(tmp_path)
    _rewritten(path, lambda manifest: manifest.pop("root"))
    with pytest.raises(FormatError, match="its manifest holds no object"):
        load(path)
    path = _saved(tmp_path)
    with zipfile.ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data[:20] if name == "arrays/0.npy" else data)
    with pytest.raises(FormatError, match=r"obj\.spend: array 0 is damaged"):
        load(path)


def _replaced(path: Path, name: str, data: bytes) -> None:
    """`path`'s archive with its member `name` holding `data`, each member stored as it is."""
    with zipfile.ZipFile(path) as archive:
        entries = {n: archive.read(n) for n in archive.namelist()}
    entries[name] = data
    with zipfile.ZipFile(path, "w") as archive:
        for n, d in entries.items():
            archive.writestr(n, d)


def test_a_file_whose_values_its_class_refuses_fails_to_load(tmp_path: Path) -> None:
    """A class's checks refuse a value out of its range with `ValueError`, and a value of another
    type than its field's with whatever error their code meets first: either fails to load."""
    path = _saved(tmp_path)
    _rewritten(path, lambda manifest: manifest["root"]["fields"].update(se=-1.0))
    with pytest.raises(FormatError, match=r"obj: .*LiftEstimate cannot be rebuilt: ValueError"):
        load(path)
    path = tmp_path / "panel.mixdyn"
    save(geo_panel(2, 4), path)
    _rewritten(path, lambda manifest: manifest["root"]["fields"].update(hierarchy="g0"))
    with pytest.raises(FormatError, match=r"obj: .*MediaPanel cannot be rebuilt: AttributeError"):
        load(path)


@pytest.mark.parametrize(
    ("name", "match"),
    [
        ("manifest.json", "it has no readable manifest.json"),
        ("arrays/0.npy", r"obj\.spend: arrays/0\.npy in the archive does not read: Bad CRC-32"),
    ],
)
def test_a_member_whose_bytes_do_not_read_fails_to_load(
    name: str, match: str, tmp_path: Path
) -> None:
    """A member whose bytes no longer match its checksum fails to load, as does any other the
    zip module or its decompressor does not read."""
    path = _saved(tmp_path)
    with zipfile.ZipFile(path) as archive:
        data = archive.read(name)
    _replaced(path, name, data)  # stored, so its last byte is a byte of the file
    damaged = bytearray(path.read_bytes())
    damaged[damaged.index(data) + len(data) - 1] ^= 1
    path.write_bytes(bytes(damaged))
    with pytest.raises(FormatError, match=match):
        load(path)


def test_an_archive_of_a_zip_version_python_does_not_read_fails_to_load(tmp_path: Path) -> None:
    path = _saved(tmp_path)
    data = bytearray(path.read_bytes())
    at = data.index(b"PK\x01\x02") + 6  # the version its first member needs to be read
    data[at : at + 2] = (99).to_bytes(2, "little")
    path.write_bytes(bytes(data))
    with pytest.raises(FormatError, match="not a zip archive that Python reads"):
        load(path)


@pytest.mark.parametrize(
    ("depth", "match"),
    [
        # from CPython 3.14 on, the parser's depth is the C stack's, so it may read this one and
        # leave it to the decoder
        (100_000, "it has no readable manifest.json|nests its object deeper"),
        (600, "nests its object deeper than Python's recursion limit"),
    ],
)
def test_a_manifest_nested_past_python_s_recursion_fails_to_load(
    depth: int, match: str, tmp_path: Path
) -> None:
    """An object nested deeper than `json` reads fails to load, as does one `json` reads but the
    decoder would recurse through past Python's limit."""
    path = _saved(tmp_path)
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    manifest["root"] = "root"
    nested = '{"$tuple": [' * depth + "1" + "]}" * depth
    _replaced(
        path, "manifest.json", json.dumps(manifest).replace('"root"}', f"{nested}}}").encode()
    )
    with pytest.raises(FormatError, match=match):
        load(path)


def test_a_manifest_the_parser_refuses_for_its_depth_fails_to_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parser refuses a manifest nested past its stack with `RecursionError`, at a depth the C
    stack sets from CPython 3.14 on: the refusal is made here."""
    path = _saved(tmp_path)

    def parse(_: bytes) -> object:
        raise RecursionError("maximum recursion depth exceeded while decoding a JSON object")

    monkeypatch.setattr(json, "loads", parse)
    with pytest.raises(FormatError, match=r"it has no readable manifest\.json"):
        load(path)


def test_a_version_nested_past_what_repr_writes_is_named_in_short(tmp_path: Path) -> None:
    """`json` reads a value nested deeper than `repr` writes; the refusal names it cut short. On
    Python 3.12 and 3.13 the parser stops first, and the manifest does not read."""
    path = _saved(tmp_path)
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    manifest["format_version"] = "version"
    nested = '{"a": [' * 30_000 + "1" + "]}" * 30_000
    _replaced(path, "manifest.json", json.dumps(manifest).replace('"version"', nested).encode())
    with pytest.raises(FormatError, match=r"no valid format version|no readable") as error:
        load(path)
    assert len(str(error.value)) < 400


@pytest.mark.parametrize(
    ("field", "node", "match"),
    [
        ("lift", {"$float": {}}, r"obj\.lift: \$float holds \{\}, where the format writes a str"),
        ("lift", {"$float": "x"}, r"obj\.lift: could not convert string to float: 'x'"),
        ("end", {"$date": 5}, r"obj\.end: \$date holds 5, where the format writes a str"),
        ("end", {"$date": "soon"}, r"obj\.end: Invalid isoformat string: 'soon'"),
        ("geos", {"$tuple": 5}, r"obj\.geos: \$tuple holds 5, where the format writes a list"),
        ("geos", {"$frozenset": [{"$list": []}]}, r"obj\.geos: .*unhashable type: 'list'"),
        ("geos", {"$dict": [[1]]}, r"obj\.geos: \$dict holds an item that is not a \[key, value\]"),
        ("geos", {"$dict": [[{"$list": []}, 1]]}, r"obj\.geos: .*unhashable type: 'list'"),
        (
            "geos",
            {"$enum": "mixdyn.data.panel:Cadence", "member": []},
            r"obj\.geos: mixdyn\.data\.panel:Cadence has no member \[\]",
        ),
        (
            "geos",
            {"$type": "mixdyn.decisions.returns:Undefined", "fields": []},
            r"obj\.geos: fields holds \[\], where the format writes a dict",
        ),
    ],
)
def test_a_node_that_holds_what_save_never_writes_there_fails_to_load(
    field: str, node: dict[str, Any], match: str, tmp_path: Path
) -> None:
    """A node's payload is the JSON `save` writes there, a set's members and a mapping's keys are
    values a set holds, and an enum's member is a name: anything else fails to load."""
    path = _saved(tmp_path)
    _rewritten(path, lambda manifest: manifest["root"]["fields"].update({field: node}))
    with pytest.raises(FormatError, match=match):
        load(path)


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _npz_bytes() -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, spend=np.zeros(4))
    return buffer.getvalue()


def _header_of(shape: tuple[int, ...]) -> bytes:
    """A `.npy` file of float64 whose header declares `shape`, and no data."""
    buffer = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        buffer, {"descr": "<f8", "fortran_order": False, "shape": shape}
    )
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("data", "match"),
    [
        (b"", "array 0 is damaged: EOF"),
        (_npz_bytes(), "array 0 is damaged: the magic string is not correct"),
        (_header_of((2**40,)), r"array 0 is damaged: its header declares \(1099511627776,\) of"),
        (_header_of((10**30,)), r"array 0 is damaged: its header declares \(10{30},\) of"),
    ],
    ids=["empty", "an npz archive", "8 TiB declared", "past int64"],
)
def test_an_array_member_that_np_save_never_writes_fails_to_load(
    data: bytes, match: str, tmp_path: Path
) -> None:
    """An array's member is an `.npy` file whose header declares no more data than follows it:
    not empty, not an `.npz` archive, and not a header that would allocate terabytes."""
    path = _saved(tmp_path)
    _replaced(path, "arrays/0.npy", data)
    with pytest.raises(FormatError, match=rf"obj\.spend: {match}"):
        load(path)


@pytest.mark.parametrize(
    "array", [np.array(["abc"]), np.array([1.0, 2.0], dtype=">f8")], ids=["text", "big-endian"]
)
def test_a_jax_array_of_a_dtype_jax_does_not_hold_fails_to_load(
    array: np.ndarray, tmp_path: Path
) -> None:
    path = tmp_path / "held.mixdyn"
    save(Returns(jnp.arange(3.0), 0.0, 0.0, 0.0, 0.0, 0.0), path)  # ty: ignore[invalid-argument-type]
    _replaced(path, "arrays/0.npy", _npy_bytes(array))
    with pytest.raises(
        FormatError, match=rf"obj\.spend is a JAX array of {array.dtype}, which JAX"
    ):
        load(path)


def test_a_value_of_a_class_the_format_does_not_hold_is_not_saved(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="is not a class the format saves"):
        save({"a": 1}, tmp_path / "x")

    @dataclasses.dataclass(frozen=True)
    class Stranger:
        value: float

    with pytest.raises(TypeError, match=r"obj\.spend is a .*Stranger, which the format does not"):
        save(Returns(Stranger(1.0), 0.0, 0.0, 0.0, 0.0, 0.0), tmp_path / "x")  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match=r"obj\.spend is an array of Python objects"):
        save(Returns(np.array([None]), 0.0, 0.0, 0.0, 0.0, 0.0), tmp_path / "x")  # ty: ignore[invalid-argument-type]


def test_a_jax_array_loads_as_one_and_never_into_single_precision(tmp_path: Path) -> None:
    """No fit holds a JAX array today; one that does keeps its kind and its precision."""
    held = Returns(jnp.arange(3.0), 0.0, 0.0, 0.0, 0.0, 0.0)  # ty: ignore[invalid-argument-type]
    loaded = roundtrip(held, tmp_path)
    assert isinstance(loaded.spend, jax.Array)
    path = tmp_path / "held.mixdyn"
    save(held, path)
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(FormatError, match="enable `jax_enable_x64` before loading"):
            load(path)
    finally:
        jax.config.update("jax_enable_x64", True)


def test_the_manifest_names_its_versions_and_holds_no_pickle(tmp_path: Path) -> None:
    path = _saved(tmp_path)
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        names = archive.namelist()
    assert manifest["format"] == "mixdyn"
    assert manifest["format_version"] == FORMAT_VERSION
    assert {"mixdyn", "causal-hybrid-control"} <= set(manifest)
    assert all(
        n == "manifest.json" or (n.startswith("arrays/") and n.endswith(".npy")) for n in names
    )


def test_the_registry_holds_dataclasses_and_enums_alone() -> None:
    assert all(dataclasses.is_dataclass(c) or issubclass(c, enum.Enum) for c in REGISTRY.values())
    assert isinstance(REGISTRY, MappingProxyType)


def _public_classes() -> set[type]:
    """Every dataclass and enum an `__all__` lists, and those a module with no `__all__` defines
    under a public name, as `mixdyn.inference.diagnostics` does its health card."""
    found = set()
    for module in modules():
        names = getattr(module, "__all__", None)
        if names is None:
            names = [
                name
                for name, value in vars(module).items()
                if not name.startswith("_") and getattr(value, "__module__", "") == module.__name__
            ]
        for name in names:
            value = getattr(module, name)
            if inspect.isclass(value) and (
                dataclasses.is_dataclass(value) or issubclass(value, enum.Enum)
            ):
                found.add(value)
    return found


def test_every_public_class_is_registered_or_left_out_with_its_reason() -> None:
    public, registered = _public_classes(), set(REGISTRY.values())
    missing = public - registered - NOT_SAVED.keys()
    assert sorted(f"{c.__module__}:{c.__qualname__}" for c in missing) == []
    assert NOT_SAVED.keys() <= public
    assert not NOT_SAVED.keys() & registered


@pytest.mark.parametrize(
    "held",
    [
        Refit(2, lambda estimates: estimates),  # ty: ignore[invalid-argument-type]
        Schema("a document", 1, tuple[int, ...]),
    ],
    ids=lambda held: type(held).__name__,
)
def test_what_a_file_does_not_carry_is_not_saved(held: Any, tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="is not a class the format saves"):
        save(held, tmp_path / "x")
