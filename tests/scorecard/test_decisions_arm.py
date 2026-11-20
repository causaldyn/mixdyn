"""The flighted planner, the scenarios and the plan table as an arm on the static arm's posterior:
`flighting` records the best rate a week the flighted planner plans beside its schedule, the
schedule and its recommendation kept, and the harness scores the schedule on the truth, its
regret the variant's where the schedule is recommended; `scenarios` plans the scenario of the
largest mean difference from the status quo, its draws pooled over the two families at their
shares; `plan table` plans `stacked` and claims the table's total. The arm runs a world end to end
under the harness and is scored.

The static arm's two fits are a stand-in here (`Fake`, its first fit's draws on the Hill curve
and its second's on tanh, as the stacked posterior holds both), and the scenarios' forecasts and
one fit's readings too, which read a `BayesFit`, so that a world runs in seconds; the planners
are the engine's own, but for the plan table's diminishing-returns point, which bisects the
posterior's mean marginal return draw by draw for minutes and which no record of the arm reads
but as a row's column: it is None here."""

import dataclasses
import importlib
import json
import math
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

pytest.importorskip("causaldyn_bench")

from causaldyn_bench.mmm_decision import PLANNED
from causaldyn_bench.scorecard.drift import DRIFT
from causaldyn_bench.scorecard.observe import FIRST
from causaldyn_bench.scorecard.schedule import score_schedule, worth
from causaldyn_bench.scorecard.scoring import built_in
from causaldyn_bench.scorecard.track_m2 import TRACK_M2
from fake_posterior import Fake

from mixdyn import NATIONAL
from mixdyn.decisions import Readings, plan_flighting, plan_table
from scorecard import report, run
from scorecard.arms import bayes, decisions
from scorecard.arms import panel as reading
from scorecard.layout import Declaration, Layout, Limits

LIMITS = Limits(memory="4G", timeout=600.0, slots=1)
# the package names the function as the module is named
TABLE_MODULE = importlib.import_module("mixdyn.decisions.plan_table")
OFFSETS = {"status quo": 0.0, "equal split": 5.0, "stacked": 3.0, "cvar 0.1": 7.0}
"""The stand-in's mean difference of each scenario from the status quo."""


class Comparing:
    """`compare_scenarios`' stand-in: each call kept, each scenario's differences its offset
    plus a draw's own spread, 0 for the baseline."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, fit, scenarios, *, baseline, horizon, draws, seed, level):
        self.calls.append(
            {
                "fit": fit,
                "scenarios": scenarios,
                "baseline": baseline,
                "horizon": horizon,
                "draws": draws,
                "seed": seed,
                "level": level,
            }
        )
        spread = np.linspace(-1.0, 1.0, draws)
        rows = [OFFSETS[s.name] + (0.0 if s.name == baseline else spread) for s in scenarios]
        return SimpleNamespace(differences=np.array([np.broadcast_to(r, draws) for r in rows]))


@dataclasses.dataclass(frozen=True)
class Family:
    """A family's fit as the stand-ins read it: its name and its draws of the stacked posterior."""

    name: str
    draws: Readings


def _both_curves(posterior):
    """The stand-in's posterior with its first fit's draws on the Hill curve of slope 1."""
    hill = posterior.fit == 0
    return dataclasses.replace(
        posterior, slope=np.where(hill[:, None], 1.0, posterior.slope), tanh=~hill
    )


def _draws_of(posterior, f):
    """The posterior's draws of fit `f` alone, as `mixdyn.decisions.readings` reads one fit."""
    keep = posterior.fit == f
    return dataclasses.replace(
        posterior,
        retention=posterior.retention[keep],
        half_saturation=posterior.half_saturation[keep],
        slope=posterior.slope[keep],
        coefficient=posterior.coefficient[keep],
        tanh=posterior.tanh[keep],
        fit=np.zeros(int(keep.sum()), dtype=np.intp),
        chain=posterior.chain[keep],
        stacking=np.ones(1),
    )


def _read_one(fit, *, draws):
    """`mixdyn.decisions.readings`' stand-in: the family's draws, every one of them asked."""
    assert draws == fit.draws.fit.size
    return fit.draws


class Faked(decisions.Decisions):
    """The arm on the stand-in's posterior; the fits named by family."""

    def fit(self, observation, *, smoke):
        fitted = Fake(draws=10)(observation, bayes.SMOKE).fitted
        posterior = _both_curves(fitted.posterior)
        fitted = dataclasses.replace(fitted, posterior=posterior, settings={"readings": 40})
        static = bayes.Bayes()
        stacked, cvar = (
            np.asarray(static.plan(fitted, v, observation).weekly, dtype=float)
            for v in (bayes.RESPONDS, "cvar 0.1")
        )
        fits = cast(
            "Any",
            {
                family: Family(f"{family}'s fit", _draws_of(posterior, f))
                for f, family in enumerate(bayes.FAMILIES)
            },
        )
        return decisions.Fitted(fitted, fits, stacked, cvar)


@pytest.fixture(autouse=True)
def stand_ins(monkeypatch):
    monkeypatch.setattr(TABLE_MODULE, "_diminishing", lambda *_: None)
    monkeypatch.setattr(decisions, "readings", _read_one)


@pytest.fixture(scope="module")
def observation():
    return TRACK_M2.observe(TRACK_M2.world("drawn", TRACK_M2.pilots["drawn"][0]), 1)


@pytest.fixture(scope="module")
def fitted(observation):
    return Faked().fit(observation, smoke=True)


def test_flighting_records_the_rate_and_keeps_the_schedule(fitted, observation):
    """The flighted planner takes no median of the stacked posterior's two curves; the arm plans
    on the draws of the family stacking leans on, here the first, of weight 0.6."""
    posterior = fitted.fitted.posterior
    plan = (
        reading.panel(observation),
        NATIONAL,
        posterior,
        observation.budget,
        observation.planned,
        decisions.limits(observation),
    )
    with pytest.raises(ValueError, match="both curves"):
        plan_flighting(*plan, level=bayes.GAIN_LEVEL)
    planned = decisions.Decisions().plan(fitted, decisions.FLIGHTING, observation)
    one = fitted.fits["hill"].draws
    assert posterior.stacking[0] > posterior.stacking[1]
    flight = plan_flighting(*plan[:2], one, *plan[3:], level=bayes.GAIN_LEVEL)
    assert planned.settings["family"] == "hill"
    assert planned.flags["edge"] == pytest.approx(float(np.mean(flight.edge)), rel=1e-12, abs=0.0)
    np.testing.assert_array_equal(planned.weekly, flight.continuous)
    assert planned.weekly.sum() * observation.planned == pytest.approx(
        observation.budget, rel=1e-9, abs=0.0
    )
    assert np.all(planned.weekly >= observation.lower - 1e-9)
    assert np.all(planned.weekly <= observation.upper + 1e-9)
    np.testing.assert_array_equal(planned.settings["schedule"], flight.schedule)
    assert planned.settings["recommendation"] == flight.recommendation
    assert planned.flags["flighted"] == (flight.recommendation == "flighted")
    assert "schedule" not in planned.absent
    gain = reading.gains(fitted.fitted.readings, observation, planned.weekly)
    assert planned.gain == pytest.approx(float(np.mean(gain)), rel=1e-12, abs=0.0)
    assert planned.gain is not None
    assert planned.gain_interval is not None
    low, high = planned.gain_interval
    assert low <= planned.gain <= high


def _held(base, schedule, recommendation):
    """`base`, a score of the bench's, its record a flighting record's schedule and
    recommendation."""
    settings = {"schedule": np.asarray(schedule).tolist(), "recommendation": recommendation}
    return dataclasses.replace(base, record={"settings": settings})


@pytest.fixture(scope="module")
def world_truth():
    world = TRACK_M2.world("drawn", TRACK_M2.pilots["drawn"][0])
    return world, TRACK_M2.truth(world)


def test_a_recommended_schedule_is_scored_against_the_best_plan_as_every_arm_is(
    world_truth, observation
):
    """The schedule's regret less the best schedule's edge over the best plan is what it returns
    short of the best plan, per euro; a rate held flat scores the bench's regret of the rate."""
    _, truth = world_truth
    quarter = truth.quarter
    base = built_in(truth)["status quo"]
    shifted = np.tile(quarter.status_quo, (PLANNED, 1))
    shifted[:6] *= 1.2
    shifted[6:] *= (quarter.budget - shifted[:6].sum()) / shifted[6:].sum()
    scored = decisions.scored(_held(base, shifted, "flighted"), truth, observation)
    expected = (truth.best.worth - worth(truth.cells, shifted)) / quarter.budget
    assert scored.regret == pytest.approx(expected, rel=1e-9, abs=0.0)
    assert scored.regret != base.regret
    held = scored.record[decisions.SCORE]
    schedule = score_schedule(truth, shifted)
    assert held == dataclasses.asdict(schedule) | {
        "rate regret": base.regret,
        "scored": "the schedule",
    }
    assert held["edge"] > 0.0
    flat = np.tile(quarter.status_quo, (PLANNED, 1))
    still = decisions.scored(_held(base, flat, "flighted"), truth, observation)
    assert still.regret == pytest.approx(base.regret, rel=1e-9, abs=0.0)


def test_a_schedule_not_recommended_or_refused_leaves_the_rates_regret(world_truth, observation):
    """The rate the record recommends keeps its regret, its schedule scored beside it; a schedule
    on cells that do not state their reach week by week is refused, and the score says why; a
    failed fit's score is the bench's."""
    _, truth = world_truth
    quarter = truth.quarter
    base = built_in(truth)["equal split"]
    flat = np.tile(quarter.status_quo, (PLANNED, 1))
    scored = decisions.scored(_held(base, flat, "continuous"), truth, observation)
    assert (scored.regret, scored.realised, scored.moved) == (base.regret, None, None)
    held = scored.record[decisions.SCORE]
    assert "scored" not in held
    assert held["rate regret"] == base.regret
    assert held["regret"] == pytest.approx(
        built_in(truth)["status quo"].regret + held["edge"], rel=1e-9, abs=0.0
    )
    blind = dataclasses.replace(
        truth, cells=tuple(dataclasses.replace(c, unit=None) for c in truth.cells)
    )
    refused = decisions.scored(_held(base, flat, "flighted"), blind, observation)
    assert refused.regret == base.regret
    assert "each quarter week's reach" in refused.record[decisions.SCORE]["refused"]
    failed = dataclasses.replace(_held(base, flat, "flighted"), failure="RuntimeError: x")
    assert decisions.scored(failed, truth, observation) is failed


def test_a_flat_schedule_on_a_drifting_world_scores_the_rates_realised_regret_too():
    """On a world whose quarter realised another path, a recommended schedule's realised regret
    is read against the best plan in hindsight, as every arm's: a rate held flat scores the
    bench's realised regret of the rate."""
    environment = next(iter(DRIFT.pilots))
    world = DRIFT.world(environment, DRIFT.pilots[environment][0])
    truth = DRIFT.truth(world)
    assert truth.realised is not None
    base = built_in(truth)["status quo"]
    assert base.realised is not None
    flat = np.tile(truth.quarter.status_quo, (PLANNED, 1))
    scored = decisions.scored(_held(base, flat, "flighted"), truth, DRIFT.observe(world, 0))
    assert scored.regret == pytest.approx(base.regret, rel=1e-9, abs=0.0)
    assert scored.realised == pytest.approx(base.realised, rel=1e-9, abs=0.0)


def test_the_scenarios_plan_the_largest_mean_difference_pooled_over_the_families(
    fitted, observation, monkeypatch
):
    comparing = Comparing()
    monkeypatch.setattr(decisions, "compare_scenarios", comparing)
    planned = decisions.Decisions().plan(fitted, decisions.SCENARIOS, observation)
    np.testing.assert_array_equal(planned.weekly, fitted.cvar)
    assert planned.settings["chosen"] == "cvar 0.1"
    counts = bayes.shares(float(fitted.fitted.posterior.stacking[0]), 40)
    assert [c["fit"].name for c in comparing.calls] == [
        f"{family}'s fit" for family, count in counts.items() if count
    ]
    for call, count in zip(comparing.calls, [c for c in counts.values() if c], strict=True):
        assert call["draws"] == count
        assert call["baseline"] == "status quo"
        assert call["horizon"] == observation.planned
        assert [s.name for s in call["scenarios"]] == list(OFFSETS)
        for scenario in call["scenarios"]:
            for name in observation.controls:
                np.testing.assert_array_equal(
                    scenario.controls[name], observation.future_controls[name]
                )
    pooled = np.concatenate([np.linspace(-1.0, 1.0, c) for c in counts.values() if c]) + 7.0
    assert planned.gain == pytest.approx(float(pooled.mean()), rel=1e-12, abs=0.0)
    low, high = np.quantile(pooled, [0.05, 0.95])
    assert planned.gain_interval == pytest.approx((low, high), rel=1e-12, abs=0.0)
    assert planned.flags["equal split difference"] == pytest.approx(5.0, rel=1e-12, abs=0.0)


def test_the_scenarios_refuse_an_export_without_the_quarters_controls(fitted, observation):
    blind = dataclasses.replace(observation, version=FIRST, future_controls=None)
    with pytest.raises(ValueError, match="the quarter's controls"):
        decisions.Decisions().plan(fitted, decisions.SCENARIOS, blind)


def test_the_plan_table_plans_stacked_and_claims_its_total(fitted, observation):
    planned = decisions.Decisions().plan(fitted, decisions.TABLE, observation)
    np.testing.assert_array_equal(planned.weekly, fitted.stacked)
    table = plan_table(
        reading.panel(observation),
        NATIONAL,
        fitted.fitted.posterior,
        fitted.stacked,
        observation.planned,
        against=observation.status_quo,
        level=bayes.GAIN_LEVEL,
        limits=decisions.limits(observation),
    )
    assert planned.gain == float(table.total_kpi.mean)
    assert planned.gain_interval == (float(table.total_kpi.low), float(table.total_kpi.high))
    rows = planned.settings["rows"]
    assert [r["channel"] for r in rows] == list(observation.channels)
    total = sum(r["kpi_change"] for r in rows)
    assert total == pytest.approx(planned.gain, rel=1e-9, abs=0.0)


def test_the_arm_runs_a_world_end_to_end_and_is_scored_and_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(decisions, "compare_scenarios", Comparing())
    monkeypatch.setitem(run.ARMS, "decisions", Faked)
    layout = Layout(tmp_path / "run")
    run.declare(
        layout,
        Declaration(
            family=0,
            environments=("drawn",),
            worlds=1,
            pilot=True,
            rungs=(1,),
            arms=("decisions",),
            smoke=True,
            limits={"decisions": LIMITS},
        ),
    )
    run.export_all(layout)
    (seed,) = layout.read().seeds("drawn")
    run.one(layout, "decisions", "drawn", seed, (1,))
    run.score(layout)
    shown = []
    monkeypatch.setattr("builtins.print", lambda *a, **_: shown.append(" ".join(map(str, a))))
    report.main(layout, list(decisions.Decisions.variants))
    (line,) = [json.loads(x) for x in layout.scores.read_text().splitlines()]
    held = line["arms"][decisions.FLIGHTING]["record"][decisions.SCORE]
    assert math.isfinite(held["regret"])
    assert "schedule scored 1" in "\n".join(shown)
    for variant in (decisions.SCENARIOS, decisions.TABLE):
        assert decisions.SCORE not in line["arms"][variant]["record"]
    for variant in decisions.Decisions.variants:
        record = json.loads(layout.record(variant, 0, "drawn", 1, seed).read_text())
        assert record["error"] is None, (variant, record.get("traceback"))
        assert line["arms"][variant]["failure"] is None
        assert math.isfinite(line["arms"][variant]["regret"])
        assert any(x.startswith(f"  {variant:16s} regret") for x in "\n".join(shown).splitlines())
