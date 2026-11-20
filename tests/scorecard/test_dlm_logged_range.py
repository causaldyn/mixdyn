"""The two-stage arm's secondary variant, `dlm two-stage, logged range`: the same fit planned within
each channel's logged spend range. Where the box lies inside that range, or the two-stage plan
does, it is the two-stage plan; where the range binds, it holds the plan inside it and still spends
the budget. It runs only where a run declares it."""

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("causaldyn_bench")

from causaldyn_bench.scorecard.track_m2 import TRACK_M2

from mixdyn.transforms import HillShape, Transform
from scorecard.arms import dlm
from scorecard.layout import Declaration
from scorecard.run import ARMS, names

OBSERVATION = TRACK_M2.observe(TRACK_M2.world("drawn", 905), 0)


@pytest.fixture(autouse=True)
def _no_response(monkeypatch):
    """The record's response reads the fit's posterior draws, which the fit here has none of."""
    monkeypatch.setattr(dlm, "_response", lambda choice, observation: {})


def _fit(observation, coefficients=(3.0, 1.0, 0.5)):
    """A fit whose best spec's channels are concave Hill curves at each channel's mean spend, as
    the two-stage arm plans from: no sampler runs."""
    spend = np.asarray(observation.spend, dtype=float)
    channels = {
        name: Transform(
            0.3, float(spend[:, j].mean()), HillShape(1.0), observation.kernel_length
        ).channel(coefficients[j])
        for j, name in enumerate(observation.channels)
    }
    reading = tuple(channels[c] for c in observation.channels)
    return dlm.Fitted(
        choice=SimpleNamespace(channels=lambda: channels),  # ty: ignore[invalid-argument-type]
        readings=(reading,) * 4,
        forecast=None,
        flags={},
        settings={},
        absent={},
        ess=1.0,
        best_ess=1.0,
    )


def _with_history(observation, spend):
    return dataclasses.replace(observation, spend=np.asarray(spend, dtype=float))


def _plans(observation, fit):
    arm = dlm.DlmTwoStageLogged()
    two = arm.plan(fit, dlm.TWO_STAGE, observation)
    logged = arm.plan(fit, dlm.LOGGED, observation)
    return two, logged


def test_where_the_box_lies_inside_the_logged_range_the_variant_is_the_two_stage_plan():
    spend = np.array(OBSERVATION.spend, dtype=float)
    spend[0] = 0.9 * np.asarray(OBSERVATION.lower)
    spend[1] = 1.1 * np.asarray(OBSERVATION.upper)
    observation = _with_history(OBSERVATION, spend)
    cut = dlm.logged_range(observation)
    assert cut is not None
    np.testing.assert_array_equal(cut[0], observation.lower)
    np.testing.assert_array_equal(cut[1], observation.upper)
    two, logged = _plans(observation, _fit(observation))
    np.testing.assert_array_equal(logged.weekly, two.weekly)
    assert (logged.gain, logged.gain_interval) == (two.gain, two.gain_interval)
    assert logged.settings == two.settings | {"spend range": "logged"}


def test_where_the_two_stage_plan_lies_inside_the_logged_range_the_variant_plans_it_too():
    """The range cut below the box's upper bound but above the two-stage plan's spend: the
    curves are concave, so the plan inside the smaller box is the same plan."""
    fit = _fit(OBSERVATION)
    two, _ = _plans(OBSERVATION, fit)
    spend = np.array(OBSERVATION.spend, dtype=float)
    upper = np.asarray(OBSERVATION.upper, dtype=float)
    ceiling = np.minimum(two.weekly + 0.5 * (upper - two.weekly), spend.max(axis=0))
    spend = np.minimum(spend, ceiling)
    spend[0] = np.minimum(spend[0], np.asarray(OBSERVATION.lower))
    observation = _with_history(OBSERVATION, spend)
    cut = dlm.logged_range(observation)
    assert cut is not None
    assert np.all(cut[1] < upper)
    assert np.all(two.weekly < cut[1])
    two, logged = _plans(observation, fit)
    assert np.all(two.weekly < cut[1])
    np.testing.assert_allclose(logged.weekly, two.weekly, rtol=1e-6, atol=0.0)


def test_where_the_logged_range_binds_the_plan_stays_inside_it_and_spends_the_budget():
    fit = _fit(OBSERVATION)
    two, _ = _plans(OBSERVATION, fit)
    heavy = int(np.argmax(two.weekly / np.asarray(OBSERVATION.status_quo)))
    spend = np.array(OBSERVATION.spend, dtype=float)
    cap = float(np.asarray(OBSERVATION.status_quo)[heavy])
    assert two.weekly[heavy] > cap
    spend[:, heavy] = np.minimum(spend[:, heavy], cap)
    observation = _with_history(OBSERVATION, spend)
    cut = dlm.logged_range(observation)
    assert cut is not None
    assert cut[1][heavy] == cap
    two, logged = _plans(observation, fit)
    assert two.weekly[heavy] > cap
    assert logged.weekly[heavy] <= cap * (1 + 1e-12)
    assert np.all(logged.weekly >= cut[0] * (1 - 1e-12))
    assert np.all(logged.weekly <= cut[1] * (1 + 1e-12))
    assert logged.weekly.sum() * observation.planned == pytest.approx(
        observation.budget, rel=1e-9, abs=0.0
    )
    assert logged.gain <= two.gain


def test_where_the_logged_range_cannot_spend_the_budget_the_variant_plans_in_the_box():
    spend = np.full_like(np.asarray(OBSERVATION.spend, dtype=float), 0.0)
    spend[:, :] = 0.5 * np.asarray(OBSERVATION.lower)
    observation = _with_history(OBSERVATION, spend)
    assert dlm.logged_range(observation) is None
    two, logged = _plans(observation, _fit(observation))
    np.testing.assert_array_equal(logged.weekly, two.weekly)
    assert logged.settings["spend range"].startswith("the box")


def test_the_variant_runs_only_where_a_run_declares_it():
    assert dlm.DlmTwoStage().variants == (dlm.TWO_STAGE,)
    assert ARMS["dlm-two-stage"]().variants == (dlm.TWO_STAGE,)
    assert ARMS["dlm-two-stage-logged"]().variants == (dlm.LOGGED,)
    declared = Declaration(
        family=0,
        environments=("drawn",),
        worlds=2,
        pilot=True,
        rungs=(4,),
        arms=("dlm-two-stage", "dlm-two-stage-logged"),
    )
    assert names(declared) == [dlm.TWO_STAGE, dlm.LOGGED]
    two = dlm.DlmTwoStage().plan(_fit(OBSERVATION), dlm.TWO_STAGE, OBSERVATION)
    assert "spend range" not in two.settings
