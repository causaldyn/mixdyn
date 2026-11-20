"""The two-stage DLM arm: the static arm's fits stacked, the higher-weighted family's medians as
every channel's transform, every effect's discount 1, the trend's chosen with the tests read; and
its one plan, the plug-in's, spends the budget in the box."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("causaldyn_bench")

from causaldyn_bench.scorecard.track_m2 import TRACK_M2

from mixdyn.transforms import HillShape, TanhShape, Transform
from scorecard.arms import bayes, dlm
from scorecard.arms import panel as reading
from scorecard.arms.base import records
from scorecard.run import ARMS


class Chosen(Exception):
    """What a discount choice that only records its arguments raises, before any fit."""


@pytest.mark.parametrize(
    ("weights", "family"),
    [((0.3, 0.7), "tanh"), ((0.8, 0.2), "hill"), ((0.5, 0.5), "hill")],
)
@pytest.mark.parametrize("k", [0, 4])
def test_the_second_stage_reads_the_higher_weighted_family_s_curves_at_discount_1(
    weights, family, k, monkeypatch
) -> None:
    observation = TRACK_M2.observe(TRACK_M2.world("drawn", 905), k)
    fits = {name: object() for name in bayes.FAMILIES}
    transforms = {
        c: Transform(0.1 * (j + 2), 100.0 * (j + 1), HillShape(1.5), observation.kernel_length)
        for j, c in enumerate(observation.channels)
    }
    handed = {}

    def family_fits(seen, sampler):
        assert seen is observation
        assert sampler == bayes.SAMPLER
        return fits

    def stacked(given, *, draws):
        assert given == list(fits.values())
        assert draws == bayes.DRAWS
        return SimpleNamespace(stacking=np.array(weights))

    def from_bayes(fit):
        handed["fit"] = fit
        return transforms

    def choose(panel, geo, spec, grid, *, level, estimates):
        handed.update(spec=spec, grid=grid, level=level, estimates=estimates)
        raise Chosen

    monkeypatch.setattr(bayes, "family_fits", family_fits)
    monkeypatch.setattr(dlm, "stacked", stacked)
    monkeypatch.setattr(dlm, "transforms_from_bayes", from_bayes)
    monkeypatch.setattr(dlm, "choose_dlm", choose)
    with pytest.raises(Chosen):
        dlm.DlmTwoStage().fit(observation, smoke=False)

    assert handed["fit"] is fits[family]
    spec = handed["spec"]
    assert [e.column for e in spec.effects] == [*observation.channels, *observation.controls]
    assert all(e.discount == 1.0 for e in spec.effects)
    media = spec.effects[: len(observation.channels)]
    assert [e.transform for e in media] == [transforms[c] for c in observation.channels]
    assert all(e.sign == "positive" for e in media)
    assert all(e.transform is None for e in spec.effects[len(media) :])
    assert handed["grid"] == dlm.TWO_STAGE_DISCOUNTS
    assert handed["level"] == dlm.LEVEL
    expected = reading.estimates(observation)
    if k == 0:
        assert handed["estimates"] is None
        return
    assert sum(len(rows) for rows in expected.values()) > 0
    assert list(handed["estimates"]) == list(expected)
    for name, rows in expected.items():
        assert [(r.lift, r.se, r.end, r.readout) for r in handed["estimates"][name]] == [
            (r.lift, r.se, r.end, r.readout) for r in rows
        ]


@pytest.mark.parametrize(("picked", "family"), [((0.0, 1.0), "tanh"), (None, "hill")])
def test_the_lift_row_pick_s_first_stage_reads_the_family_the_lift_rows_pick(
    picked, family, monkeypatch
) -> None:
    """Stacking leans to the Hill; the lift rows pick the `tanh`, whose curves the second stage
    reads; where the export holds no test the pick is None and the stacking weight's family is
    read."""
    observation = TRACK_M2.observe(TRACK_M2.world("drawn", 905), 4)
    fits = {name: object() for name in bayes.FAMILIES}
    handed = {}

    def lift_row_pick(given):
        assert given == list(fits.values())
        return None if picked is None else np.array(picked)

    def from_bayes(fit):
        handed["fit"] = fit
        raise Chosen

    monkeypatch.setattr(bayes, "family_fits", lambda seen, sampler: fits)
    monkeypatch.setattr(
        dlm, "stacked", lambda given, *, draws: SimpleNamespace(stacking=np.array([0.8, 0.2]))
    )
    monkeypatch.setattr(dlm, "lift_row_pick", lift_row_pick)
    monkeypatch.setattr(dlm, "transforms_from_bayes", from_bayes)
    with pytest.raises(Chosen):
        dlm.DlmTwoStageLiftPick().fit(observation, smoke=False)
    assert handed["fit"] is fits[family]
    with pytest.raises(Chosen):
        dlm.DlmTwoStage().fit(observation, smoke=False)
    assert handed["fit"] is fits["hill"]


def test_the_two_stage_arm_is_declared_beside_the_search_s() -> None:
    assert ARMS["dlm-two-stage"]().variants == ("dlm two-stage",)
    assert ARMS["dlm-two-stage-lift-pick"]().variants == ("dlm two-stage, lift-row pick",)
    assert "dlm plug-in" in ARMS["dlm"]().variants
    assert dlm.DlmTwoStage.tests is True


def test_the_two_stage_plan_spends_the_budget_in_the_box_and_records_its_first_stage() -> None:
    observation = TRACK_M2.observe(TRACK_M2.world("drawn", 905), 4)
    (record,) = records(dlm.DlmTwoStage(), observation, smoke=True).values()
    assert record["error"] is None
    weekly = np.asarray(record["weekly"], dtype=float)
    assert np.all(weekly >= np.asarray(observation.lower) - 1e-9)
    assert np.all(weekly <= np.asarray(observation.upper) + 1e-9)
    assert weekly.sum() * observation.planned == pytest.approx(
        observation.budget, rel=1e-9, abs=0.0
    )
    settings = record["settings"]
    assert settings["first stage"] in bayes.FAMILIES
    assert settings["effects discount"] == 1.0
    assert settings["lift_rows_read"] > 0
    shapes = {t["shape"] for t in settings["transforms"].values()}
    assert shapes <= {repr(TanhShape())} or all(s.startswith("HillShape") for s in shapes)
    assert record["flags"]["trend discount"] in dlm.TWO_STAGE_DISCOUNTS.trend
    assert record["forecast"] is not None
