"""The tested rung's candidates: each fit they read drawn once, from the seed the static arm draws
its own from; each candidate's two fits weighed by its rule, the lift rows' pick or stacking, and
planned as the static arm's `stacked`; its settings name the candidate and its fits' CPU seconds;
the runner holds each candidate in one arm, both concave candidates in one that draws their two
fits once."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("causaldyn_bench")

from causaldyn_bench.scorecard.track_m2 import TRACK_M2

from mixdyn.models.bayes_mmm import HillCurve, TanhCurve
from scorecard.arms import bayes, tested
from scorecard.arms.base import Planned
from scorecard.layout import Declaration, Limits
from scorecard.run import ARMS, names

OBSERVATION = TRACK_M2.observe(TRACK_M2.world("drawn", 905), 4)


class Planning(Exception):
    """What a stand-in for the static arm's posterior raises once it has recorded its call."""


def _fits(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, object, int | None]]:
    """`bayes.family_fits` replaced by one that records each fit asked for and returns a stand-in
    named by it."""
    calls = []

    def family_fits(seen, sampler, estimates=None, *, curves, kernel):
        assert seen is OBSERVATION
        assert sampler == bayes.SAMPLER
        assert estimates is None
        ((family, curve),) = curves.items()
        calls.append((family, curve, kernel))
        return {family: SimpleNamespace(drawn=(family, curve, kernel))}

    monkeypatch.setattr(bayes, "family_fits", family_fits)
    return calls


def test_each_fit_the_candidates_read_is_drawn_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fits(monkeypatch)
    six = tested.Candidate("six", rule="lift rows", kernel=6)
    arm = tested.Tested((*tested.CANDIDATES, tested.Candidate("default"), six))
    fitted = arm.fit(OBSERVATION, smoke=False)
    assert calls == [
        ("hill", HillCurve(), None),
        ("tanh", TanhCurve(), None),
        ("hill", HillCurve(-0.3, 0.4), None),
        ("hill", HillCurve(), 6),
        ("tanh", TanhCurve(), 6),
    ]
    assert set(fitted.fits) == set(fitted.cpu)
    assert all(seconds >= 0.0 for seconds in fitted.cpu.values())
    assert fitted.sampler == bayes.SAMPLER


@pytest.mark.parametrize(
    ("candidate", "weighed"),
    [
        (tested.LIFT_PICK, "lift rows"),
        (tested.CONCAVE_PICK, "lift rows"),
        (tested.CONCAVE_HILL, "stacking"),
        (tested.Candidate("stacking"), "stacking"),
    ],
)
def test_a_candidate_weighs_its_two_fits_by_its_rule(
    candidate: tested.Candidate, weighed: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lift rows' pick reaches the posterior as its weights; stacking's rule hands none, so
    the posterior is stacked over every observation. The pick here goes against the stacking
    weight the module's stand-in gives, so a candidate that skipped its rule would be read."""
    _fits(monkeypatch)
    picked, handed = np.array([0.0, 1.0]), {}

    def pick_by_lift_tests(fits):
        handed["picked from"] = [f.drawn for f in fits]
        return picked

    def fitted(observation, fits, sampler, *, given=None):
        handed.update(fits=fits, sampler=sampler, weights=given)
        raise Planning

    monkeypatch.setattr(tested, "pick_by_lift_tests", pick_by_lift_tests)
    monkeypatch.setattr(bayes, "fitted", fitted)
    arm = tested.Tested((candidate,))
    with pytest.raises(Planning):
        arm.plan(arm.fit(OBSERVATION, smoke=False), candidate.name, OBSERVATION)
    hill = ("hill", HillCurve(*candidate.slope), None)
    tanh = ("tanh", TanhCurve(), None)
    assert [f.drawn for f in handed["fits"].values()] == [hill, tanh]
    assert list(handed["fits"]) == list(bayes.FAMILIES)
    assert handed["sampler"] == bayes.SAMPLER
    if weighed == "lift rows":
        assert handed["picked from"] == [hill, tanh]
        assert handed["weights"] is picked
    else:
        assert "picked from" not in handed
        assert handed["weights"] is None


def test_a_candidate_plans_as_stacked_and_records_its_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fits(monkeypatch)
    posterior = SimpleNamespace(stacking=np.array([0.7, 0.3]))
    made = SimpleNamespace()
    planned = Planned(
        weekly=np.ones(3), flags={"hill weight": 0.0}, settings={"planner": "plan_posterior"}
    )

    def plan(self, fit, variant, observation):
        assert fit is made
        assert variant == bayes.RESPONDS == "stacked"
        assert observation is OBSERVATION
        return planned

    monkeypatch.setattr(tested, "pick_by_lift_tests", lambda fits: np.array([0.0, 1.0]))
    monkeypatch.setattr(tested, "stacked_readings", lambda fits, *, draws: posterior)
    monkeypatch.setattr(tested, "_lift_rows", lambda fits: {"hill": -91.0, "tanh": -89.5})
    monkeypatch.setattr(bayes, "fitted", lambda *args, **kwargs: made)
    monkeypatch.setattr(bayes.Bayes, "plan", plan)
    arm = tested.Tested((tested.LIFT_PICK,))
    out = arm.plan(arm.fit(OBSERVATION, smoke=False), tested.LIFT_PICK.name, OBSERVATION)
    np.testing.assert_array_equal(out.weekly, planned.weekly)
    assert out.flags == {
        "hill weight": 0.0,
        "stacking hill weight": 0.7,
        "hill lift-row elpd": -91.0,
        "tanh lift-row elpd": -89.5,
    }
    assert out.settings["planner"] == "plan_posterior"
    assert out.settings["candidate"] == {
        "rule": "lift rows",
        "slope prior": [0.0, 0.5],
        "kernel": "the panel's",
    }
    assert out.settings["rule read"] == "lift rows"
    assert [v["family"] for v in out.settings["fits"].values()] == ["hill", "tanh"]
    assert list(out.settings["fits"]) == [
        "hill LogNormal(0, 0.5), kernel the panel's",
        "tanh, kernel the panel's",
    ]
    assert all(v["cpu_seconds"] >= 0.0 for v in out.settings["fits"].values())


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (
            lambda: tested.Candidate("x", rule="whole"),  # ty: ignore[invalid-argument-type]
            "one of .*'stacking', 'lift rows'",
        ),
        (lambda: tested.Candidate("x", slope=(0.0, 0.0)), "finite scale above 0"),
        (lambda: tested.Candidate("x", kernel=0), "whole number of weeks, at least 1, not 0"),
        (lambda: tested.Tested(()), "one candidate or more"),
        (
            lambda: tested.Tested((tested.LIFT_PICK, tested.LIFT_PICK)),
            "each named once",
        ),
    ],
)
def test_what_cannot_be_a_candidate_is_refused(make, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        make()


def test_the_concave_arm_plans_both_its_variants_from_one_draw_of_its_two_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stacking's plan and the lift rows' pick read the same concave Hill and `tanh`, drawn once,
    and each hands the posterior its own rule's weights."""
    calls = _fits(monkeypatch)
    picked, handed = np.array([0.0, 1.0]), []
    monkeypatch.setattr(tested, "pick_by_lift_tests", lambda fits: picked)

    def fitted(observation, fits, sampler, *, given=None):
        handed.append(([f.drawn for f in fits.values()], given))
        raise Planning

    monkeypatch.setattr(bayes, "fitted", fitted)
    arm = ARMS["bayes-concave"]()
    drawn = arm.fit(OBSERVATION, smoke=False)
    pair = [("hill", HillCurve(-0.3, 0.4), None), ("tanh", TanhCurve(), None)]
    assert calls == pair
    assert arm.variants == (tested.CONCAVE_HILL.name, tested.CONCAVE_PICK.name)
    for variant in arm.variants:
        with pytest.raises(Planning):
            arm.plan(drawn, variant, OBSERVATION)
    assert [fits for fits, _ in handed] == [pair, pair]
    assert handed[0][1] is None
    assert handed[1][1] is picked


def test_the_runner_holds_each_candidate_in_one_arm() -> None:
    held = {
        "bayes-lift-pick": (tested.LIFT_PICK,),
        "bayes-concave": (tested.CONCAVE_HILL, tested.CONCAVE_PICK),
    }
    for arm, candidates in held.items():
        made = ARMS[arm]()
        assert made.variants == tuple(c.name for c in candidates)
        assert made.tests
        assert not made.cells
    assert "bayes-concave-lift-pick" not in ARMS
    own = {c.name: c for candidates in held.values() for c in candidates}
    assert set(own) == {c.name for c in tested.CANDIDATES}
    arms = ("bayes", "dlm", *held, "dlm-two-stage-lift-pick")
    declared = Declaration(
        family=0,
        environments=("drawn",),
        worlds=2,
        pilot=True,
        rungs=(0, 4),
        arms=arms,
        limits={a: Limits("4G", 3600.0, 1) for a in arms},
    )
    assert set(names(declared)) >= {"stacked", "dlm two-stage, lift-row pick", *own}
