"""The dynamic arm's market plan split over a geo export's cells by population: the projection into
the cells' boxes at a channel's spend, the split of families 8 and 10, and the arm, which plans the
market as the dynamic arm does and runs a world end to end under the harness.

The dynamic arm is a stand-in in the arm's tests, so that a world runs in seconds: its plan is
given."""

import json
import math
from typing import ClassVar

import numpy as np
import pytest

pytest.importorskip("causaldyn_bench")

from causaldyn_bench.scorecard.few_geos import FEW_GEOS
from causaldyn_bench.scorecard.geo import GEO
from causaldyn_bench.scorecard.track_m2 import TRACK_M2

from scorecard import report, run
from scorecard.arms import panel as reading
from scorecard.arms import population
from scorecard.arms.base import Planned
from scorecard.layout import Declaration, Layout, Limits

ENVIRONMENT = "policy-0"
LIMITS = Limits(memory="4G", timeout=600.0, slots=1)


@pytest.fixture(scope="module", params=[(GEO, "policy-0"), (FEW_GEOS, "geos-8")], ids=["f8", "f10"])
def observed(request):
    family, environment = request.param
    return family.observe(family.world(environment, family.pilots[environment][0]), 4)


def _check_projection(target, lower, upper, total, got):
    """`got` sums to `total`, lies in the box, and is `target` less one shift where it is inside
    the box, the shift no more than a bound's room where it sits on that bound: the nearest point
    of the box at the total."""
    assert got.sum() == pytest.approx(total, rel=1e-12, abs=0.0)
    assert np.all(got >= lower)
    assert np.all(got <= upper)
    inside = (got > lower) & (got < upper)
    assert inside.any()
    shifts = (target - got)[inside]
    shift = float(shifts[0])
    np.testing.assert_allclose(shifts, shift, rtol=0.0, atol=1e-9 * total)
    assert np.all(target[got == upper] - upper[got == upper] >= shift - 1e-9 * total)
    assert np.all(target[got == lower] - lower[got == lower] <= shift + 1e-9 * total)


def test_a_target_in_the_box_at_the_total_is_kept():
    target = np.array([1.0, 2.0, 3.0])
    got = population.into_box(target, np.zeros(3), np.full(3, 5.0), 6.0)
    np.testing.assert_allclose(got, target, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("seed", range(5))
def test_a_target_is_moved_to_the_nearest_point_of_the_box_at_the_total(seed):
    rng = np.random.default_rng(seed)
    lower = rng.uniform(0.0, 1.0, 8)
    upper = lower + rng.uniform(0.2, 2.0, 8)
    total = float(rng.uniform(lower.sum() + 0.1, upper.sum() - 0.1))
    target = rng.uniform(0.0, 3.0, 8)
    got = population.into_box(target, lower, upper, total)
    _check_projection(target, lower, upper, total, got)


def test_a_total_the_boxes_cannot_hold_is_refused():
    with pytest.raises(ValueError, match="cannot spend"):
        population.into_box(np.ones(2), np.zeros(2), np.ones(2), 2.5)


def test_a_channels_spend_is_split_over_its_geos_by_population_into_their_boxes(observed):
    """The market's status quo, each channel's total, split: each per-geo channel's cells hold its
    total, each in its box, the population's shares less one shift; a national channel's one cell
    its spend."""
    whole = reading.market(observed)
    weekly = np.asarray(whole.status_quo, dtype=float)
    plan = population.split(observed, weekly)
    cells = reading.cells(observed)
    people = observed.population / observed.population.sum()
    lower, upper = np.asarray(observed.lower), np.asarray(observed.upper)
    assert plan.sum() == pytest.approx(weekly.sum(), rel=1e-12, abs=0.0)
    for j, name in enumerate(observed.channels):
        at = np.array([i for i, (_, c) in enumerate(cells) if c == name])
        if name in observed.national:
            assert plan[at].tolist() == [weekly[j]]
            continue
        target = weekly[j] * people[[cells[i][0] for i in at]]
        _check_projection(target, lower[at], upper[at], weekly[j], plan[at])


def test_a_plan_is_split_only_over_a_geo_export():
    national = TRACK_M2.observe(TRACK_M2.world("drawn", TRACK_M2.pilots["drawn"][0]), 0)
    with pytest.raises(ValueError, match="no geo axis"):
        population.split(national, np.asarray(national.status_quo))
    with pytest.raises(ValueError, match="no geo axis"):
        population.DlmByPopulation().fit(national, smoke=True)


class _Dynamic:
    """The dynamic arm's stand-in: its fit keeps what it was handed, its plug-in plan is the
    market's status quo with the first channel's spend moved to the second within the box."""

    handed: ClassVar[list] = []

    def fit(self, observation, *, smoke):
        self.handed.append(observation)
        return "the fit"

    def plan(self, fit, variant, observation):
        assert fit == "the fit"
        assert variant == population.PLUG_IN
        assert isinstance(observation, reading.Market)
        weekly = np.asarray(observation.status_quo, dtype=float).copy()
        step = min(weekly[0] - observation.lower[0], observation.upper[1] - weekly[1]) / 2.0
        weekly[0] -= step
        weekly[1] += step
        return Planned(
            weekly=weekly,
            gain=1.5,
            gain_interval=(1.0, 2.0),
            flags={"kept": 3},
            settings={"planner": "allocate"},
            absent={"forecast": "recorded under 'dlm mixture'"},
            ess=400.0,
        )


def test_the_arm_plans_the_cells_from_the_dynamic_arms_market_plan(observed, monkeypatch):
    monkeypatch.setattr(population, "Dlm", _Dynamic)
    _Dynamic.handed = []
    arm = population.DlmByPopulation()
    fitted = arm.fit(observed, smoke=True)
    (handed,) = _Dynamic.handed
    assert isinstance(handed, reading.Market)
    assert handed.geo is observed
    planned = arm.plan(fitted, population.VARIANT, observed)
    market = _Dynamic().plan("the fit", population.PLUG_IN, reading.market(observed))
    np.testing.assert_array_equal(planned.weekly, population.split(observed, market.weekly))
    assert (planned.gain, planned.gain_interval, planned.ess) == (1.5, (1.0, 2.0), 400.0)
    assert planned.settings == {
        "planner": "allocate",
        "split": "population",
        "market plan": population.PLUG_IN,
    }
    assert set(planned.absent) == {"forecast", "response"}


def test_the_arm_runs_a_world_end_to_end_and_is_scored_and_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(population, "Dlm", _Dynamic)
    seed = GEO.pilots[ENVIRONMENT][0]
    layout = Layout(tmp_path / "run")
    run.declare(
        layout,
        Declaration(
            family=8,
            environments=(ENVIRONMENT,),
            worlds=1,
            pilot=True,
            rungs=(0, 4),
            arms=("dlm-by-population",),
            smoke=True,
            limits={"dlm-by-population": LIMITS},
        ),
    )
    run.export_all(layout)
    for k in (0, 4):
        run.one(layout, "dlm-by-population", ENVIRONMENT, seed, (k,))
    run.score(layout)
    shown = []
    monkeypatch.setattr("builtins.print", lambda *a, **_: shown.append(" ".join(map(str, a))))
    report.main(layout, [population.VARIANT])
    for k in (0, 4):
        record = json.loads(layout.record(population.VARIANT, 8, ENVIRONMENT, k, seed).read_text())
        assert record["error"] is None, record.get("traceback")
        assert record["plan_level"] == "cells"
        assert record["settings"]["split"] == "population"
    lines = [json.loads(x) for x in layout.scores.read_text().splitlines()]
    assert sorted(line["k"] for line in lines) == [0, 4]
    for line in lines:
        assert math.isfinite(line["arms"][population.VARIANT]["regret"])
    assert population.VARIANT in "\n".join(shown)
