"""The model over geos as a DLM arm on family 8: the additive spec of raw spend at the dynamic arm's
starting discounts; each cell's coefficient its geo's, a national channel's its geos' by
population share; a plan of linear cells, the linear program's optimum within each cell's box at
the budget, and its gain on those lines. The arm runs a world end to end under the harness and is
scored.

`fit_geo_dlm` is a stand-in here, so that a world runs in seconds: its coefficients are given."""

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import linprog

pytest.importorskip("causaldyn_bench")

from causaldyn_bench.scorecard.geo import GEO
from causaldyn_bench.scorecard.track_m2 import TRACK_M2

from mixdyn.models import Effect, Season, Trend
from scorecard import report, run
from scorecard.arms import geo_dlm
from scorecard.arms import panel as reading
from scorecard.arms.base import withheld
from scorecard.layout import Declaration, Layout, Limits

ENVIRONMENT = "policy-0"
LIMITS = Limits(memory="4G", timeout=600.0, slots=1)


@pytest.fixture(scope="module")
def observed():
    return withheld(GEO.observe(GEO.world(ENVIRONMENT, GEO.pilots[ENVIRONMENT][0]), 4))


def _given(mean):
    """`fit_geo_dlm` and `geo_effects`' stand-ins: each call's panel and spec kept, `mean` the
    coefficients."""
    calls = []

    def fit(panel, spec, prior=None, *, draws, seed):
        calls.append(SimpleNamespace(panel=panel, spec=spec, draws=draws, seed=seed))
        return "the fit"

    def effects(fit):
        assert fit == "the fit"
        return SimpleNamespace(mean=mean)

    return fit, effects, calls


def _linear_program(coefficients, lower, upper, budget, weeks):
    """The best plan of linear cells as HiGHS solves the linear program: the week's budget spent
    whole, each cell within its box."""
    solved = linprog(
        -np.asarray(coefficients),
        A_eq=np.ones((1, len(coefficients))),
        b_eq=[budget / weeks],
        bounds=list(zip(lower, upper, strict=True)),
        method="highs",
    )
    assert solved.status == 0, solved.message
    return solved.x


def test_the_spec_is_additive_raw_spend_at_the_dynamic_arms_starting_discounts(observed):
    spec = geo_dlm.spec(observed)
    assert spec.form == "additive"
    assert spec.effects == tuple(Effect(c, 0.99) for c in observed.channels)
    assert spec.trend == Trend(1, 0.99)
    assert spec.season == Season(52.18, 3, 1.0)


def test_a_cells_coefficient_is_its_geos_and_a_national_channels_by_population(observed):
    rng = np.random.default_rng(20261107)
    mean = rng.normal(size=(observed.population.size, len(observed.channels)))
    held = geo_dlm.coefficients(observed, mean)
    shares = observed.population / observed.population.sum()
    for k, (g, c) in enumerate(reading.cells(observed)):
        j = observed.channels.index(c)
        expected = float(shares @ mean[:, j]) if g is None else mean[g, j]
        assert held[k] == pytest.approx(expected, rel=1e-12, abs=0.0)
    assert len(held) == len(observed.status_quo)


def test_the_plan_fills_the_largest_coefficient_first_and_claims_its_gain(observed, monkeypatch):
    """Coefficients drawn so that some are below 0 (which `chc.allocation.allocate` refuses):
    the plan is the linear program's optimum, to HiGHS' tolerance, and its gain the lines' worth of
    it less the status quo's over the quarter."""
    rng = np.random.default_rng(7)
    mean = rng.normal(0.5, 1.0, size=(observed.population.size, len(observed.channels)))
    fit, effects, calls = _given(mean)
    monkeypatch.setattr(geo_dlm, "fit_geo_dlm", fit)
    monkeypatch.setattr(geo_dlm, "geo_effects", effects)
    arm = geo_dlm.GeoDlm()
    fitted = arm.fit(observed, smoke=True)
    (call,) = calls
    assert call.draws == geo_dlm.SMOKE_DRAWS
    assert call.spec == geo_dlm.spec(observed)
    assert fitted.flags["negative coefficients"] == int(np.sum(fitted.coefficients < 0.0))
    planned = arm.plan(fitted, geo_dlm.VARIANT, observed)
    assert fitted.flags["negative coefficients"] > 0
    best = _linear_program(
        fitted.coefficients, observed.lower, observed.upper, observed.budget, observed.planned
    )
    scale = float(np.max(observed.upper))
    np.testing.assert_allclose(planned.weekly, best, rtol=0.0, atol=1e-6 * scale)
    gain = observed.planned * float(fitted.coefficients @ (planned.weekly - observed.status_quo))
    assert planned.gain == pytest.approx(gain, rel=1e-9, abs=0.0)
    assert planned.gain_interval is None


def test_the_arm_refuses_a_national_export():
    national = TRACK_M2.observe(TRACK_M2.world("drawn", TRACK_M2.pilots["drawn"][0]), 0)
    with pytest.raises(ValueError, match="no geo axis"):
        geo_dlm.GeoDlm().fit(national, smoke=True)


def test_the_arm_runs_a_world_end_to_end_and_is_scored_and_reported(
    tmp_path, observed, monkeypatch
):
    mean = np.full((observed.population.size, len(observed.channels)), 0.3)
    mean[:, 0] = 0.6
    fit, effects, _ = _given(mean)
    monkeypatch.setattr(geo_dlm, "fit_geo_dlm", fit)
    monkeypatch.setattr(geo_dlm, "geo_effects", effects)
    layout = Layout(tmp_path / "run")
    run.declare(
        layout,
        Declaration(
            family=8,
            environments=(ENVIRONMENT,),
            worlds=1,
            pilot=True,
            rungs=(0, 4),
            arms=("geo-dlm",),
            smoke=True,
            limits={"geo-dlm": LIMITS},
        ),
    )
    run.export_all(layout)
    run.one(layout, "geo-dlm", ENVIRONMENT, observed.seed, (0, 4))
    run.score(layout)
    shown = []
    monkeypatch.setattr("builtins.print", lambda *a, **_: shown.append(" ".join(map(str, a))))
    report.main(layout, [geo_dlm.VARIANT])
    for k in (0, 4):
        record = json.loads(
            layout.record(geo_dlm.VARIANT, 8, ENVIRONMENT, k, observed.seed).read_text()
        )
        assert record["error"] is None, record.get("traceback")
        assert record["plan_level"] == "cells"
    lines = [json.loads(x) for x in layout.scores.read_text().splitlines()]
    assert sorted(line["k"] for line in lines) == [0, 4]
    for line in lines:
        assert math.isfinite(line["arms"][geo_dlm.VARIANT]["regret"])
    assert f"  {geo_dlm.VARIANT:16s} regret" in "\n".join(shown)
