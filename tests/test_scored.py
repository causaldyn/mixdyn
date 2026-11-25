"""The scored rules' public planners (`mixdyn.decisions.scored`) on the arms' fixture worlds, read
as a caller holds them: a `MediaPanel`, the channels' `Range`s and the lift tests as
`LiftEstimate`s. What each plans is the arm's to the last bit (`tests/scorecard/
test_public_planners.py`, which reads the bench); here, what each refuses, what it names, and that
the plan spends the budget in the box."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from gs_worlds import WORLDS as TESTED
from margin_worlds import PARITY
from mixdyn import NATIONAL, Cadence, Channel, Geo, MediaPanel, Revenue
from mixdyn.calibration import LiftEstimate
from mixdyn.decisions import (
    GeoLimits,
    MarginPlan,
    Range,
    plan_anchored_margins,
    plan_margins,
    plan_pooled_geos,
    scored,
)

FIRST_WEEK = np.datetime64("2024-01-01")
DARK = 4
"""The weeks each fixture world's tests went dark."""
COOLDOWN = 8
"""The weeks each fixture world's tests read after their dark weeks."""


def national(world: dict[str, Any]) -> MediaPanel:
    """The world's history as a weekly panel of one national geo, its controls by name."""
    sales = np.asarray(world["sales"], dtype=float)
    names = tuple(world["controls"])
    values = np.column_stack([np.asarray(world["controls"][n], dtype=float) for n in names])
    return MediaPanel(
        dates=FIRST_WEEK + np.timedelta64(7, "D") * np.arange(sales.size),
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=tuple(Channel(c) for c in world["channels"]),
        controls=names,
        kpi=sales[None],
        kpi_unit=Revenue(),
        spend=np.asarray(world["spend"], dtype=float)[None],
        currency="EUR",
        control_values=values[None],
        filled=np.zeros((1, sales.size), dtype=bool),
    )


def go_darks(world: dict[str, Any]) -> dict[str, tuple[LiftEstimate, ...]]:
    """The world's lift rows as the go-dark tests they read, by channel: each the history's spend
    to its readout's end, 0 over its dark weeks, its lift and standard error over them."""
    spend = np.asarray(world["spend"], dtype=float)
    lift = world["lift"]
    found: dict[str, list[LiftEstimate]] = {}
    for name, start, delta_y, sigma in zip(
        lift["channel"], lift["start"], lift["delta_y"], lift["sigma"], strict=True
    ):
        first = start - 1
        end = first + DARK + COOLDOWN
        planned = spend[:end, world["channels"].index(name)].copy()
        tested = planned.copy()
        tested[first : first + DARK] = 0.0
        found.setdefault(name, []).append(
            LiftEstimate(
                spend=tested,
                planned=planned,
                readout=DARK + COOLDOWN,
                lift=delta_y * DARK,
                se=sigma * DARK,
                end=FIRST_WEEK + np.timedelta64(7, "D") * (end - 1),
            )
        )
    return {name: tuple(tests) for name, tests in found.items()}


def ranges(world: dict[str, Any]) -> dict[Channel, Range]:
    """Each channel's box a week, the bench's."""
    return {
        Channel(c): Range(float(lo), float(hi))
        for c, lo, hi in zip(world["channels"], world["lower"], world["upper"], strict=True)
    }


@pytest.fixture(scope="module")
def margin_plan() -> MarginPlan:
    world = PARITY[0]
    return plan_margins(
        national(world), None, world["planned"], ranges(world), kernel=world["kernel_length"]
    )


@pytest.fixture(scope="module")
def anchored_plan() -> MarginPlan:
    world = TESTED[0]
    return plan_anchored_margins(
        national(world),
        go_darks(world),
        None,
        world["planned"],
        ranges(world),
        kernel=world["kernel_length"],
    )


@pytest.mark.parametrize("plan", ["margin_plan", "anchored_plan"])
def test_a_margin_plan_spends_the_status_quos_weekly_total_in_the_box(
    plan: str, request: pytest.FixtureRequest
) -> None:
    made: MarginPlan = request.getfixturevalue(plan)
    world = PARITY[0] if plan == "margin_plan" else TESTED[0]
    np.testing.assert_array_equal(made.current, np.asarray(world["status_quo"]))
    assert made.budget == world["planned"] * float(made.current.sum())
    assert float(made.spend.sum()) == pytest.approx(float(made.current.sum()), rel=1e-12, abs=0.0)
    assert np.all(np.asarray(world["lower"]) <= made.spend)
    assert np.all(made.spend <= np.asarray(world["upper"]))
    assert made.margins.shape == (3,)
    assert made.cov.shape == (3, 3)
    assert made.interval[0] <= made.gain <= made.interval[1]


def test_each_plan_names_the_arm_whose_record_it_reproduces(
    margin_plan: MarginPlan, anchored_plan: MarginPlan
) -> None:
    assert margin_plan.record == "margin held"
    assert anchored_plan.record == "anchored floored"
    world = TESTED[0]
    args = (national(world), go_darks(world), None, world["planned"], ranges(world))

    def record(spend_floor: float | None) -> str | None:
        planned = plan_anchored_margins(
            *args, kernel=world["kernel_length"], spend_floor=spend_floor
        )
        return planned.record

    assert record(None) == "anchored held"
    assert record(0.8) is None


def test_another_budget_is_planned_and_spent() -> None:
    world = PARITY[0]
    quo = np.asarray(world["status_quo"])
    budget = 0.9 * world["planned"] * float(quo.sum())
    made = plan_margins(
        national(world), budget, world["planned"], ranges(world), kernel=world["kernel_length"]
    )
    assert made.budget == budget
    assert float(made.spend.sum()) == pytest.approx(budget / world["planned"], rel=1e-12, abs=0.0)


def _geo_panel() -> MediaPanel:
    world = PARITY[0]
    made = national(world)
    return dataclasses.replace(
        made,
        geos=(Geo("north"), Geo("south")),
        kpi=np.repeat(made.kpi, 2, axis=0),
        spend=np.repeat(made.spend, 2, axis=0),
        control_values=np.repeat(made.control_values, 2, axis=0),
        filled=np.zeros((2, made.dates.size), dtype=bool),
    )


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"cadence": Cadence.DAY}, "weekly history"),
        ({"controls": ("promotion", "weather")}, "the controls"),
    ],
)
def test_a_national_rule_refuses_a_panel_its_record_does_not_cover(
    change: dict[str, Any], match: str
) -> None:
    world = PARITY[0]
    made = national(world)
    if change.get("cadence") is Cadence.DAY:
        made = dataclasses.replace(
            made, dates=FIRST_WEEK + np.timedelta64(1, "D") * np.arange(made.dates.size), **change
        )
    else:
        made = dataclasses.replace(made, **change)
    with pytest.raises(ValueError, match=match):
        plan_margins(made, None, 13, ranges(world), kernel=world["kernel_length"])


def test_a_national_rule_refuses_a_geo_panel_and_names_the_geo_planner() -> None:
    with pytest.raises(ValueError, match="plan_pooled_geos"):
        plan_margins(_geo_panel(), None, 13, ranges(PARITY[0]), kernel=8)


@pytest.mark.parametrize(
    ("limits", "match"),
    [
        (lambda r: {c: x for c, x in r.items() if c != "tv"}, r"missing \['tv'\]"),
        (lambda r: {**r, Channel("radio"): Range(0.0, 1.0)}, r"no such channel \['radio'\]"),
        (lambda r: {**r, Channel("tv"): Range(0.0)}, "finite most"),
    ],
)
def test_every_channel_needs_its_own_finite_range(limits: Any, match: str) -> None:
    world = PARITY[0]
    with pytest.raises(ValueError, match=match):
        plan_margins(national(world), None, 13, limits(ranges(world)), kernel=8)


@pytest.mark.parametrize(("periods", "budget"), [(0, None), (13.0, None), (13, float("nan"))])
def test_the_periods_are_whole_weeks_and_the_budget_is_finite(periods: Any, budget: Any) -> None:
    world = PARITY[0]
    with pytest.raises(ValueError, match=r"whole number of weeks|finite"):
        plan_margins(national(world), budget, periods, ranges(world), kernel=8)


def test_the_anchored_rule_refuses_a_channel_without_a_test() -> None:
    """Its record planned every channel tested; with one untested the rule is not the one scored."""
    world = TESTED[0]
    tests = {c: t for c, t in go_darks(world).items() if c != "tv"}
    with pytest.raises(ValueError, match=r"no test of \['tv'\]"):
        plan_anchored_margins(national(world), tests, None, 13, ranges(world), kernel=8)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"share": 0.5}, "share of the market"),
        ({"end": np.datetime64("2030-01-07")}, "no date of the panel's"),
        ({"readout": 4}, "go-dark whose dark weeks open its readout"),
    ],
)
def test_the_anchored_rule_refuses_a_test_it_does_not_read(
    change: dict[str, Any], match: str
) -> None:
    world = TESTED[0]
    tests = go_darks(world)
    tests["tv"] = (dataclasses.replace(tests["tv"][0], **change), *tests["tv"][1:])
    with pytest.raises(ValueError, match=match):
        plan_anchored_margins(national(world), tests, None, 13, ranges(world), kernel=8)


@pytest.mark.parametrize("spend_floor", [0.0, 1.5, float("nan")])
def test_a_spend_floor_is_a_share(spend_floor: float) -> None:
    world = TESTED[0]
    with pytest.raises(ValueError, match="spend floor"):
        plan_anchored_margins(
            national(world),
            go_darks(world),
            None,
            13,
            ranges(world),
            kernel=8,
            spend_floor=spend_floor,
        )


def test_the_tests_are_read_as_the_bench_wrote_them() -> None:
    """The reduction of a go-dark test to what the rule reads: its planned spend's mean over the
    dark weeks, its lift and standard error a dark week, the week it went dark."""
    world = TESTED[0]
    rows = scored._rows(national(world), go_darks(world))
    lift = world["lift"]
    assert rows.channel == tuple(lift["channel"])
    np.testing.assert_array_equal(rows.x, np.asarray(lift["x"]))
    np.testing.assert_array_equal(rows.delta_y, np.asarray(lift["delta_y"]))
    np.testing.assert_array_equal(rows.sigma, np.asarray(lift["sigma"]))
    np.testing.assert_array_equal(rows.start, np.asarray(lift["start"]))
    assert (rows.dark, rows.readout) == (DARK, DARK + COOLDOWN)


def test_the_geo_rule_refuses_totals_it_does_not_read() -> None:
    made = _geo_panel()
    cells = {(g, c): Range(0.0, 1.0) for g in made.geos for c in made.channels}
    limits = GeoLimits(cells=cells, geos={Geo("north"): Range(0.0, 1.0)})
    with pytest.raises(ValueError, match="reads no geo's or channel's total"):
        plan_pooled_geos(made, {}, None, 13, limits, kernel=8)
