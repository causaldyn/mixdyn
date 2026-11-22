"""The anchored arm (`scorecard.arms.anchored`) and the composite `gs` (`scorecard.composite`): on
eight of M2's worlds the arm's route, its anchored margins, their shrunk means and its plan are the
research code's, which found the rule; a world whose tests rise with the spend routes by the slope,
one whose kernel runs past the readout routes by the kernel, and tests at one spend read no slope;
a regression that raises leaves the tests alone, as the research code's did; at k = 0 the plan is
the margin arm's; an export the rule does not read, a geo export and its market among them, is a
failed record. The composite plays
`stacked`'s recorded plan where the arm routes and the arm's own elsewhere: scored by the bench, its
regret on each of the eight worlds is the rule's as the research code scored it, and the
scorecard's readers, the claim's and the selection's, assemble it. The variant `anchored curved`
and the composite `gs curved`: on the eight worlds each channel's weight on the line, the plan and
its regret are the curved rule's research code's; with the line's prior at 1 the plan is
`anchored`'s corner; where the regression raises or the export holds no test the plan is
`anchored`'s; and the composite plays `stacked` where the variant routes. The variant
`anchored held` and the composite `gs held`: where no channel's weekly spend follows the week's
number the record is `anchored curved`'s to the last bit and the composites play alike; a channel
that follows it is held at its status quo, the others planned at the budget in their box, or the
status quo where fewer than two are free, where their box holds no plan, and where the fit's
readings route the world, which is then not routed; a spend that never moves is not held, and one
just below the threshold is not. The variant `anchored floored`: where no floor is raised the record
is `anchored held`'s to the last bit but never routed; where the held plan lies below the raised
floor the plan lies on it, the water-filling's in the box so raised; a held channel stays at its
status quo; floors raised past the budget leave the guard's box; and a floor above its ceiling is
kept, failing as the held variant fails."""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest

from gs_worlds import CURVED, ROUTED, WORLDS, name

pytest.importorskip("causaldyn_bench")

from causaldyn_bench.budget_regret import LiftRows, lift_rows
from causaldyn_bench.scorecard.geo import GEO
from causaldyn_bench.scorecard.observe import Observation
from causaldyn_bench.scorecard.scoring import ArmScore, WorldRecord, score_world
from test_development import _write

from mixdyn.decisions.margin_plan import plan_corner, water_fill_tabulated
from mixdyn.models.margin_mmm import DegenerateFitError
from scorecard import claim, development, run
from scorecard.arms import anchored, margin
from scorecard.arms import panel as reading
from scorecard.arms.base import records
from scorecard.composite import GS, GS_CURVED, GS_HELD, compose
from scorecard.layout import FAMILIES, Declaration

RELATIVE = 1e-9
KEPT = [w for w in WORLDS if not w["research"]["routed"]]
HELD = "anchored held"
FLOORED = "anchored floored"
RAMP = 0.97
"""The correlation with the week's number of a channel the tests have the guard hold."""
REASONS = ("routed by slope", "routed by kernel")


def observation(world, **change):
    """The world's tested-rung export as the arm reads it, rebuilt from the fixture."""

    def series(values):
        return np.asarray(values, dtype=float)

    lift = world["lift"]
    seen = Observation(
        family=world["family"],
        environment=world["environment"],
        seed=world["seed"],
        k=world["k"],
        channels=tuple(world["channels"]),
        sales=series(world["sales"]),
        spend=series(world["spend"]),
        controls={k: series(v) for k, v in world["controls"].items()},
        future_controls={k: series(v) for k, v in world["future_controls"].items()},
        kernel_length=world["kernel_length"],
        planned=world["planned"],
        budget=world["budget"],
        lower=series(world["lower"]),
        upper=series(world["upper"]),
        status_quo=series(world["status_quo"]),
        roi_window=tuple(world["roi_window"]),
        lift=LiftRows(
            channel=tuple(lift["channel"]),
            start=tuple(lift["start"]),
            x=series(lift["x"]),
            delta_x=series(lift["delta_x"]),
            delta_y=series(lift["delta_y"]),
            sigma=series(lift["sigma"]),
            dropped=lift["dropped"],
        ),
    )
    return dataclasses.replace(seen, **change)


def _relifted(seen, **rows):
    return dataclasses.replace(seen, lift=dataclasses.replace(seen.lift, **rows))


def test_the_arm_is_declared_as_anchored_reads_the_tests_and_plans_the_market():
    assert run.ARMS["anchored"] is anchored.Anchored
    arm = anchored.Anchored()
    assert (arm.variants, arm.tests, arm.cells) == (
        ("anchored", "anchored curved", "anchored held", "anchored floored"),
        True,
        False,
    )
    assert (GS.name, GS.router, GS.routed) == ("gs", "anchored", "stacked")
    assert (GS_CURVED.name, GS_CURVED.router, GS_CURVED.routed) == (
        "gs curved",
        "anchored curved",
        "stacked",
    )
    assert (GS_HELD.name, GS_HELD.router, GS_HELD.routed) == ("gs held", "anchored held", "stacked")
    assert anchored.READOUT == 12


@pytest.mark.parametrize("world", WORLDS, ids=name)
def test_the_rebuilt_export_is_the_archived_one(world):
    assert observation(world).digest() == world["digest"] == world["stacked"]["record"]["digest"]


@pytest.mark.parametrize("world", WORLDS, ids=name)
def test_the_route_the_anchored_margins_and_the_plan_are_the_research_codes(world):
    seen = observation(world)
    arm = anchored.Anchored()
    fitted = arm.fit(seen, smoke=False)
    research, flags = world["research"], fitted.flags
    assert flags["routed"] is research["routed"]
    assert (flags["routed by slope"], flags["routed by kernel"]) == (research["routed"], False)
    assert flags["max roi slope t"] == pytest.approx(max(research["t"]), rel=RELATIVE, abs=0.0)
    assert flags["tau_b"] == pytest.approx(research["tau_b"], rel=RELATIVE, abs=0.0)
    for c, channel in enumerate(seen.channels):
        for flag, key in (
            ("margin", "mean"),
            ("test roi", "roi"),
            ("roi slope t", "t"),
            ("anchored margin", "anchored"),
            ("shrunk margin", "shrunk"),
        ):
            assert flags[f"{channel} {flag}"] == pytest.approx(
                research[key][c], rel=RELATIVE, abs=0.0
            ), flag
    assert fitted.shrunk is not None
    np.testing.assert_allclose(fitted.shrunk.mean, research["shrunk"], rtol=RELATIVE, atol=0.0)
    np.testing.assert_array_equal(arm.plan(fitted, "anchored", seen).weekly, research["plan"])


def test_the_record_holds_the_plan_the_route_and_its_reasons():
    (world, *_) = ROUTED
    seen = observation(world)
    made = records(anchored.Anchored(), seen, smoke=False)["anchored"]
    assert made["error"] is None
    weekly = np.asarray(made["weekly"])
    assert np.all(weekly >= seen.lower)
    assert np.all(weekly <= seen.upper)
    assert weekly.sum() * seen.planned == pytest.approx(seen.budget, rel=1e-12, abs=0.0)
    flags = made["flags"]
    assert (flags["routed"], flags["routed by slope"], flags["routed by kernel"]) == (
        True,
        True,
        False,
    )
    assert (flags["lift rows"], flags["regression raised"]) == (len(world["lift"]["x"]), False)
    settings = made["settings"]
    assert (settings["slope_z"], settings["routes_to"], settings["lift_rows_read"]) == (
        1.645,
        "stacked",
        len(world["lift"]["x"]),
    )
    assert settings["readout_weeks"] == {"dark": 4, "cooldown": 8}
    assert set(made["n/a"]) == {"forecast", "response"}
    json.dumps(made, allow_nan=False)


def _rising(seen):
    """`seen` with its first channel's tests reading a return per euro that rises with the
    spend, precisely."""
    lift = seen.lift
    first = np.array([c == seen.channels[0] for c in lift.channel])
    x = np.asarray(lift.x)
    rho = 1.0 + 0.5 * np.log(x / seen.status_quo[0])
    delta_y = np.where(first, -rho * x, lift.delta_y)
    sigma = np.where(first, 0.01 * x, lift.sigma)
    return _relifted(seen, delta_y=delta_y, sigma=sigma)


def test_tests_whose_return_rises_with_the_spend_route_by_the_slope():
    seen = observation(KEPT[0])
    flags = anchored.Anchored().fit(_rising(seen), smoke=False).flags
    assert flags[f"{seen.channels[0]} roi slope t"] > anchored.SLOPE_Z
    assert (flags["routed"], flags["routed by slope"], flags["routed by kernel"]) == (
        True,
        True,
        False,
    )


@pytest.mark.parametrize(("kernel", "routed"), [(12, False), (13, True)])
def test_a_kernel_past_the_tests_readout_routes_by_the_kernel(kernel, routed):
    seen = observation(KEPT[0], kernel_length=kernel)
    flags = anchored.Anchored().fit(seen, smoke=False).flags
    assert (flags["routed"], flags["routed by slope"], flags["routed by kernel"]) == (
        routed,
        False,
        routed,
    )


def test_tests_at_one_spend_read_no_slope_and_route_nothing():
    seen = observation(KEPT[0])
    lift = seen.lift
    x = np.array([seen.status_quo[seen.channels.index(c)] for c in lift.channel])
    flat = _relifted(seen, x=x, delta_x=-x)
    flags = anchored.Anchored().fit(flat, smoke=False).flags
    assert [flags[f"{c} roi slope t"] for c in seen.channels] == [0.0] * len(seen.channels)
    assert (flags["max roi slope t"], flags["routed"]) == (0.0, False)


@pytest.mark.parametrize("world", WORLDS, ids=name)
def test_a_regression_that_raises_leaves_the_tests_alone_as_the_research_code(world, monkeypatch):
    def degenerate(*_, **__):
        msg = "no residual degree of freedom"
        raise DegenerateFitError(msg)

    monkeypatch.setattr(anchored, "fit_margins", degenerate)
    seen = observation(world)
    made = records(anchored.Anchored(), seen, smoke=False)["anchored"]
    assert made["error"] is None
    np.testing.assert_array_equal(made["weekly"], world["research"]["tests_alone"]["plan"])
    flags = made["flags"]
    shrunk = [flags[f"{c} shrunk margin"] for c in seen.channels]
    np.testing.assert_allclose(
        shrunk, world["research"]["tests_alone"]["shrunk"], rtol=RELATIVE, atol=0.0
    )
    assert (flags["regression raised"], flags["tau_b"], flags["routed"]) == (
        True,
        None,
        world["research"]["routed"],
    )
    assert made["n/a"]["margins"].startswith("the regression raised, DegenerateFitError: ")


def test_at_k_0_the_plan_is_the_margin_arms_and_no_world_is_routed():
    seen = observation(ROUTED[0], k=0, lift=lift_rows({}))
    made = records(anchored.Anchored(), seen, smoke=False)["anchored"]
    plain = records(margin.Margin(), seen, smoke=False)["margin"]
    assert made["error"] is None
    assert made["weekly"] == plain["weekly"]
    assert made["gain"] == plain["gain"]
    assert (made["flags"]["routed"], made["flags"]["lift rows"]) == (False, 0)
    assert made["n/a"]["lift tests"].startswith("the export holds no lift row")


def test_at_k_0_a_regression_that_raises_is_the_fits_failure(monkeypatch):
    def degenerate(*_, **__):
        msg = "a residual whose lagged sum of squares is 0"
        raise DegenerateFitError(msg)

    monkeypatch.setattr(anchored, "fit_margins", degenerate)
    seen = observation(KEPT[0], k=0, lift=lift_rows({}))
    made = records(anchored.Anchored(), seen, smoke=False)["anchored"]
    assert made["error"].startswith("DegenerateFitError: ")
    assert made["failure_kind"] == "fit"


def test_an_export_the_rule_does_not_read_is_a_failed_record():
    seen = observation(KEPT[0])
    lift = seen.lift
    kept = [c != seen.channels[0] for c in lift.channel]
    partial = _relifted(
        seen,
        channel=tuple(c for c, k in zip(lift.channel, kept, strict=True) if k),
        start=tuple(s for s, k in zip(lift.start, kept, strict=True) if k),
        x=lift.x[kept],
        delta_x=lift.delta_x[kept],
        delta_y=lift.delta_y[kept],
        sigma=lift.sigma[kept],
    )
    for refused, says, kind in (
        (partial, "ValueError: every row's channel is one of", "fit"),
        (
            dataclasses.replace(seen, budget=seen.budget * 1.01),
            "ValueError: the rule spends",
            "plan",
        ),
    ):
        made = records(anchored.Anchored(), refused, smoke=False)["anchored"]
        assert made["error"].startswith(says)
        assert made["failure_kind"] == kind


@pytest.mark.parametrize("k", [0, 4])
def test_a_geo_export_and_its_market_are_refused(k):
    """The harness hands the arm a geo export's market (`scorecard.arms.base.records`), which
    holds no control and, past k = 0, the geos' tests: their spends are a geo's, not the
    market's, and the rule would read them against the market's status quo."""
    seen = GEO.observe(GEO.world("policy-0", GEO.pilots["policy-0"][0]), k)
    assert (len(seen.lift.channel) > 0) == (k > 0)
    made = records(anchored.Anchored(), seen, smoke=False)
    for variant in ("anchored", "anchored curved"):
        assert made[variant]["error"] == f"ValueError: {margin.GEO}"
        assert made[variant]["failure_kind"] == "fit"
    for read in (seen, reading.market(seen)):
        with pytest.raises(ValueError, match="a geo export is refused"):
            anchored.Anchored().fit(read, smoke=False)


@pytest.mark.parametrize("k", [0, 4])
def test_an_export_without_controls_fits_without_them_as_the_margin_arm_does(k):
    """Some families' exports hold no promotion and no price. Both variants plan there; without
    tests their plan is the margin arm's on the same export, and with them the curves are read."""
    held = observation(KEPT[0]) if k else observation(KEPT[0], k=0, lift=lift_rows({}))
    seen = dataclasses.replace(held, controls={}, future_controls={})
    made = records(anchored.Anchored(), seen, smoke=False)
    for variant in ("anchored", "anchored curved"):
        assert made[variant]["error"] is None
        assert made[variant]["settings"]["controls"] == []
    if k == 0:
        plain = records(margin.Margin(), seen, smoke=False)["margin"]
        assert made["anchored"]["weekly"] == plain["weekly"]
        assert made["anchored curved"]["weekly"] == plain["weekly"]
    else:
        assert made["anchored curved"]["settings"]["planner"] == "water-filling on a table"


def _arm(regret, *, routed=None, failure=None, cpu=1.0):
    record = {"error": failure, "cost": {"cpu_seconds": cpu, "wall_seconds": 2.0 * cpu}}
    if routed is not None:
        record["flags"] = {"routed": routed}
    return ArmScore(
        regret=regret,
        failure=failure,
        moved=None,
        bounds=0,
        realised=None,
        uplift=None,
        covered=None,
        crps=None,
        record=record,
    )


def _place(arms, *, seed=1, k=4, index=0):
    return WorldRecord(
        family=3,
        environment="r-0.9",
        seed=seed,
        index=index,
        pilot=False,
        k=k,
        budget=1000.0,
        best=2000.0,
        channels=3,
        oracle_bounds=0,
        covariates={},
        arms={"status quo": _arm(0.3), "equal split": _arm(0.25), **arms},
    )


def test_the_composite_plays_stacked_where_the_arm_routes_and_the_arm_elsewhere():
    stacked = _arm(0.12, cpu=300.0)
    routed = compose(_place({"anchored": _arm(0.05, routed=True), "stacked": stacked}))
    gs = routed.arms["gs"]
    assert dataclasses.replace(gs, record=stacked.record) == stacked
    assert gs.record["composite"] == {"router": "anchored", "played": "stacked", "routed": True}
    assert gs.record["cost"] == {"cpu_seconds": 301.0, "wall_seconds": 602.0}
    kept = compose(_place({"anchored": _arm(0.05, routed=False), "stacked": stacked}))
    assert (kept.arms["gs"].regret, kept.arms["gs"].record["cost"]["cpu_seconds"]) == (0.05, 1.0)
    assert kept.arms["gs"].record["composite"]["played"] == "anchored"
    assert compose(kept) == kept


def test_a_router_that_failed_fails_the_composite_and_a_world_without_both_arms_holds_none():
    failed = _arm(0.3, failure="ValueError: the export holds no control")
    held = compose(_place({"anchored": failed, "stacked": _arm(0.12)})).arms["gs"]
    assert (held.failure, held.regret, held.record["composite"]["played"]) == (
        failed.failure,
        0.3,
        "anchored",
    )
    assert "gs" not in compose(_place({"anchored": _arm(0.05, routed=True)})).arms
    assert "gs" not in compose(_place({"stacked": _arm(0.12)})).arms


def test_a_record_cannot_take_a_composites_name():
    held = Declaration(
        family=0,
        environments=("drawn",),
        worlds=2,
        pilot=True,
        rungs=(0,),
        arms=("margin",),
        external=("gs",),
    )
    with pytest.raises(ValueError, match="names of their own"):
        run.names(held)


@pytest.mark.parametrize("world", WORLDS, ids=name)
def test_scored_by_the_bench_the_composite_is_the_rules_regret(world):
    """The arm's record and M2's recorded `stacked` plan scored on the world's truth: the arm's
    regret is the research code's plan's, and the composite's is the rule's, `stacked`'s where it
    routes, within 1e-9 a euro."""
    seen = observation(world)
    made = records(anchored.Anchored(), seen, smoke=False)["anchored"]
    scored = score_world(
        FAMILIES[world["family"]],
        world["environment"],
        world["seed"],
        world["k"],
        {"anchored": made, "stacked": world["stacked"]["record"]},
        archived=seen,
    )
    research = world["research"]
    assert scored.arms["anchored"].regret == pytest.approx(research["regret"], rel=0, abs=1e-9)
    assert scored.arms["stacked"].regret == pytest.approx(
        world["stacked"]["regret"], rel=0, abs=1e-9
    )
    gs = compose(scored).arms["gs"]
    expected = world["stacked"]["regret"] if research["routed"] else research["regret"]
    assert gs.regret == pytest.approx(expected, rel=0, abs=1e-9)
    played = scored.arms["stacked" if research["routed"] else "anchored"]
    assert dataclasses.replace(gs, record=played.record) == played


def test_the_selections_reading_assembles_the_composite_of_a_pilots_arm_and_a_stored_one(
    tmp_path, monkeypatch
):
    """Each export's digest is the text written there in place of an archive."""
    monkeypatch.setattr(development, "_digest", lambda path: path.read_text())
    stored_declaration = Declaration(
        family=3,
        environments=("r-0.9",),
        worlds=2,
        pilot=False,
        bench="316fada",
        rungs=(4,),
        arms=("bayes",),
        external=("orbit",),
        default="stacked",
        margin=0.05,
    )
    seeds = list(stored_declaration.seeds("r-0.9"))
    stored = [
        _place({"stacked": _arm(0.12 + 0.01 * i), "orbit": _arm(0.2)}, seed=s, index=i)
        for i, s in enumerate(seeds)
    ]
    _write(tmp_path / "f3", stored_declaration, stored)
    pilot = [
        dataclasses.replace(
            r,
            arms={
                **_place({}).arms,
                "anchored": _arm(0.05, routed=bool(i)),
                HELD: _arm(0.03, routed=False),
            },
        )
        for i, r in enumerate(stored)
    ]
    pilot_declaration = dataclasses.replace(
        stored_declaration, arms=("anchored",), default=None, margin=None
    )
    _write(tmp_path / "p3", pilot_declaration, pilot)
    held = development.joined([tmp_path / "f3"], [tmp_path / "p3"])
    first, second = (held[3, "r-0.9", s, 4].arms["gs"] for s in seeds)
    assert (first.regret, second.regret) == (0.05, 0.13)
    assert [g.record["composite"]["played"] for g in (first, second)] == ["anchored", "stacked"]
    assert [held[3, "r-0.9", s, 4].arms["gs held"].regret for s in seeds] == [0.03, 0.03]


def test_the_claims_reading_assembles_the_composite_as_the_default_above_0(tmp_path):
    """A run that plans both arms and declares `gs` its default above 0: the claim reads the
    composite against each rival."""
    d = Declaration(
        family=3,
        environments=("r-0.9",),
        worlds=4,
        pilot=False,
        bench="316fada",
        rungs=(0, 4),
        arms=("margin", "bayes", "anchored"),
        external=("orbit",),
        default="margin",
        margin=0.05,
        default_above_0="gs",
    )
    seeds = list(d.seeds("r-0.9"))
    out = []
    for i, s in enumerate(seeds):
        out.append(_place({"margin": _arm(0.1), "orbit": _arm(0.2)}, seed=s, k=0, index=i))
        arms = {
            "anchored": _arm(0.05 + 0.01 * i, routed=i % 2 == 1),
            HELD: _arm(0.03, routed=i == 0),
            "stacked": _arm(0.15 + 0.01 * i),
            "orbit": _arm(0.2 + 0.02 * i),
        }
        out.append(_place(arms, seed=s, index=i))
    _write(tmp_path / "f3", d, out)
    ((root, _, read),) = claim.joined_runs([tmp_path / "f3"])[0]
    assert root == tmp_path / "f3"
    assert read == claim.read_run(tmp_path / "f3")[1]
    at_4 = [r for r in read if r.k == 4]
    expected = [0.05, 0.16, 0.07, 0.18]
    assert [r.arms["gs"].regret for r in sorted(at_4, key=lambda r: r.index)] == pytest.approx(
        expected, rel=0, abs=1e-15
    )
    assert [r.arms["gs held"].regret for r in sorted(at_4, key=lambda r: r.index)] == [
        0.15,
        0.03,
        0.03,
        0.03,
    ]
    (comparison,) = [c for c in claim.compare(read, d) if (c.k, c.rival) == (4, "orbit")]
    assert comparison.default == "gs"
    mean = float(np.mean(np.array(expected) - np.array([0.2, 0.22, 0.24, 0.26])))
    assert comparison.mean == pytest.approx(mean, rel=1e-12, abs=0.0)


def test_the_claims_reading_assembles_the_composite_of_an_arm_beside_the_run(tmp_path):
    """A run that plans `stacked` and a root beside it that plans `anchored`, every variant of it:
    the run read with the root beside it holds the composites, the run read alone does not."""
    d = Declaration(
        family=3,
        environments=("r-0.9",),
        worlds=2,
        pilot=False,
        bench="316fada",
        rungs=(0, 4),
        arms=("margin", "bayes"),
        external=("orbit",),
        default="margin",
        margin=0.05,
        default_above_0="gs",
    )
    seeds = list(d.seeds("r-0.9"))
    run_records, side_records = [], []
    for i, s in enumerate(seeds):
        for k in d.rungs:
            arms = {"margin": _arm(0.1)} if k == 0 else {"stacked": _arm(0.15 + 0.01 * i)}
            run_records.append(_place(arms, seed=s, k=k, index=i))
            routed = k > 0 and i == 1
            planned = {
                "anchored": _arm(0.05 + 0.01 * i, routed=routed),
                "anchored curved": _arm(0.04 + 0.01 * i, routed=routed),
                HELD: _arm(0.03 + 0.01 * i, routed=False),
                FLOORED: _arm(0.02 + 0.01 * i, routed=False),
            }
            side_records.append(_place(planned, seed=s, k=k, index=i))
    _write(tmp_path / "f3", d, run_records)
    side = dataclasses.replace(
        d, arms=("anchored",), external=(), default=None, margin=None, default_above_0=None
    )
    _write(tmp_path / "f3-anchored", side, side_records)
    ((_, _, read),), _ = claim.joined_runs([tmp_path / "f3", tmp_path / "f3-anchored"])
    at_4 = sorted((r for r in read if r.k == 4), key=lambda r: r.index)
    assert [r.arms["gs"].regret for r in at_4] == [0.05, 0.16]
    assert [r.arms["gs"].record["composite"]["played"] for r in at_4] == ["anchored", "stacked"]
    assert [r.arms["gs curved"].regret for r in at_4] == [0.04, 0.16]
    assert [r.arms["gs held"].regret for r in at_4] == [0.03, 0.04]
    assert [r.arms["gs held"].record["composite"]["played"] for r in at_4] == [HELD, HELD]
    alone = claim.read_run(tmp_path / "f3")[1]
    assert all(not {"gs", "gs curved", "gs held"} & set(r.arms) for r in alone)


def test_the_claims_reading_assembles_the_composite_of_the_joined_roots_two_arms(tmp_path):
    """A run that plans neither arm and declares `gs curved` its default above 0 from its joined
    root, which plans `anchored` and `bayes` at the tested rung alone: the run read on its own
    joins the router's and the routed arm's records and reads the composite against each rival. A
    joined root that plans one of the two is refused."""
    d = Declaration(
        family=3,
        environments=("r-0.9",),
        worlds=2,
        pilot=False,
        bench="316fada",
        rungs=(0, 4),
        arms=("margin",),
        external=("orbit",),
        default="margin",
        margin=0.05,
        default_above_0="gs curved",
        joined="-ts",
    )
    seeds = list(d.seeds("r-0.9"))
    run_records, joined_records = [], []
    for i, s in enumerate(seeds):
        run_records.append(_place({"margin": _arm(0.1), "orbit": _arm(0.2)}, seed=s, k=0, index=i))
        run_records.append(_place({"orbit": _arm(0.2 + 0.02 * i)}, seed=s, index=i))
        planned = {
            "anchored": _arm(0.05 + 0.01 * i, routed=i == 1),
            "anchored curved": _arm(0.04 + 0.01 * i, routed=i == 1),
            "stacked": _arm(0.15 + 0.01 * i),
        }
        joined_records.append(_place(planned, seed=s, index=i))
    _write(tmp_path / "f3", d, run_records)
    joined = dataclasses.replace(
        d,
        rungs=(4,),
        arms=("anchored", "bayes"),
        external=(),
        default=None,
        margin=None,
        default_above_0=None,
        joined=None,
    )
    _write(tmp_path / "f3-ts", joined, joined_records)
    declared, read = claim.read_run(tmp_path / "f3")
    at_4 = sorted((r for r in read if r.k == 4), key=lambda r: r.index)
    assert [r.arms["gs curved"].regret for r in at_4] == [0.04, 0.16]
    assert [r.arms["gs curved"].record["composite"]["played"] for r in at_4] == [
        "anchored curved",
        "stacked",
    ]
    assert all("gs curved" not in r.arms for r in read if r.k == 0)
    (comparison,) = [c for c in claim.compare(read, declared) if (c.k, c.rival) == (4, "orbit")]
    assert comparison.default == "gs curved"
    mean = float(np.mean(np.array([0.04, 0.16]) - np.array([0.2, 0.22])))
    assert comparison.mean == pytest.approx(mean, rel=1e-12, abs=0.0)
    ((root, _, through),), _ = claim.joined_runs([tmp_path / "f3", tmp_path / "f3-ts"])
    assert (root, through) == (tmp_path / "f3", read)
    for one in ("anchored", "bayes"):
        _write(tmp_path / f"only-{one}", d, run_records)
        _write(tmp_path / f"only-{one}-ts", dataclasses.replace(joined, arms=(one,)), [])
        with pytest.raises(ValueError, match="does not hold 'gs curved'"):
            claim.read_run(tmp_path / f"only-{one}")


@pytest.mark.parametrize("world", WORLDS, ids=name)
def test_the_curved_plan_its_line_weights_and_its_regret_are_the_research_codes(world):
    """The variant's record and M2's recorded `stacked` plan scored on the world's truth: the
    plan, each channel's posterior weight on the line and the regret are the curved rule's
    research code's, and the composite's regret is the rule's, `stacked`'s where it routes."""
    seen = observation(world)
    research = CURVED[world["seed"]]
    made = records(anchored.Anchored(), seen, smoke=False)
    curved = made["anchored curved"]
    assert curved["error"] is None
    np.testing.assert_allclose(curved["weekly"], research["plan"], rtol=RELATIVE, atol=0.0)
    np.testing.assert_allclose(
        [curved["flags"][f"{c} line weight"] for c in seen.channels],
        research["line_weight"],
        rtol=RELATIVE,
        atol=0.0,
    )
    assert curved["flags"]["routed"] is world["research"]["routed"]
    scored = score_world(
        FAMILIES[world["family"]],
        world["environment"],
        world["seed"],
        world["k"],
        {"anchored curved": curved, "stacked": world["stacked"]["record"]},
        archived=seen,
    )
    assert scored.arms["anchored curved"].regret == pytest.approx(
        research["regret"], rel=0, abs=1e-9
    )
    composed = compose(scored).arms["gs curved"]
    routed = world["research"]["routed"]
    assert composed.regret == pytest.approx(
        world["stacked"]["regret"] if routed else research["regret"], rel=0, abs=1e-9
    )
    assert composed.record["composite"] == {
        "router": "anchored curved",
        "played": "stacked" if routed else "anchored curved",
        "routed": routed,
    }


@pytest.mark.parametrize("world", WORLDS, ids=name)
def test_with_the_lines_prior_at_1_the_curved_plan_is_the_anchored_corner(world, monkeypatch):
    monkeypatch.setattr(anchored, "LINE_PRIOR", 1.0)
    seen = observation(world)
    made = records(anchored.Anchored(), seen, smoke=False)
    corner, curved = made["anchored"], made["anchored curved"]
    np.testing.assert_allclose(curved["weekly"], corner["weekly"], rtol=1e-12, atol=0.0)
    assert curved["gain"] == pytest.approx(corner["gain"], rel=1e-9, abs=0.0)
    np.testing.assert_allclose(
        curved["gain_interval"], corner["gain_interval"], rtol=1e-9, atol=0.0
    )
    assert [curved["flags"][f"{c} line weight"] for c in seen.channels] == [1.0] * len(
        seen.channels
    )


def test_the_curved_record_holds_the_route_the_curves_and_a_plan_apart_from_the_corner():
    world = next(w for w in KEPT if CURVED[w["seed"]]["plan"] != w["research"]["plan"])
    seen = observation(world)
    made = records(anchored.Anchored(), seen, smoke=False)
    corner, curved = made["anchored"], made["anchored curved"]
    weekly = np.asarray(curved["weekly"])
    assert np.all(weekly >= seen.lower)
    assert np.all(weekly <= seen.upper)
    assert weekly.sum() * seen.planned == pytest.approx(seen.budget, rel=1e-12, abs=0.0)
    assert curved["weekly"] != corner["weekly"]
    low, high = curved["gain_interval"]
    assert low < curved["gain"] < high
    flags, settings = curved["flags"], curved["settings"]
    for channel in seen.channels:
        assert 0.0 < flags[f"{channel} line weight"] < 1.0
        assert 0.0 < flags[f"{channel} elasticity"] < 1.0
    assert {k: v for k, v in flags.items() if k in corner["flags"]} == corner["flags"]
    assert (settings["line_prior"], settings["planner"], settings["table_points"]) == (
        0.7,
        "water-filling on a table",
        4001,
    )
    assert len(settings["elasticities"]) == 41
    assert (corner["settings"]["planner"], "line_prior" in corner["settings"]) == ("corner", False)
    assert set(curved["n/a"]) == {"forecast", "response"}
    json.dumps(curved, allow_nan=False)


def test_where_the_regression_raises_the_curved_plan_is_the_tests_alone_at_the_corner(
    monkeypatch,
):
    def degenerate(*_, **__):
        msg = "no residual degree of freedom"
        raise DegenerateFitError(msg)

    monkeypatch.setattr(anchored, "fit_margins", degenerate)
    (world, *_) = KEPT
    seen = observation(world)
    curved = records(anchored.Anchored(), seen, smoke=False)["anchored curved"]
    assert curved["error"] is None
    np.testing.assert_array_equal(curved["weekly"], world["research"]["tests_alone"]["plan"])
    assert [curved["flags"][f"{c} line weight"] for c in seen.channels] == [None] * 3
    assert curved["settings"]["planner"] == "corner"


def test_at_k_0_the_curved_plan_is_the_margin_arms():
    seen = observation(KEPT[0], k=0, lift=lift_rows({}))
    made = records(anchored.Anchored(), seen, smoke=False)
    plain = records(margin.Margin(), seen, smoke=False)["margin"]
    curved = made["anchored curved"]
    assert curved["error"] is None
    assert (curved["weekly"], curved["gain"]) == (plain["weekly"], plain["gain"])
    assert (curved["flags"]["routed"], curved["settings"]["planner"]) == (False, "corner")


def test_the_curved_composite_plays_stacked_where_the_curved_variant_routes():
    stacked = _arm(0.12, cpu=300.0)
    arms = {
        "anchored": _arm(0.05, routed=True),
        "anchored curved": _arm(0.04, routed=True, cpu=2.0),
        "stacked": stacked,
    }
    routed = compose(_place(arms)).arms
    assert (routed["gs"].regret, routed["gs curved"].regret) == (0.12, 0.12)
    assert routed["gs curved"].record["cost"] == {"cpu_seconds": 302.0, "wall_seconds": 604.0}
    kept = compose(_place(arms | {"anchored curved": _arm(0.04, routed=False)})).arms
    assert (kept["gs"].regret, kept["gs curved"].regret) == (0.12, 0.04)
    assert kept["gs curved"].record["composite"] == {
        "router": "anchored curved",
        "played": "anchored curved",
        "routed": False,
    }
    assert GS_CURVED.name not in compose(_place({"anchored": arms["anchored"]})).arms
    held = Declaration(
        family=0,
        environments=("drawn",),
        worlds=2,
        pilot=True,
        rungs=(0,),
        arms=("margin",),
        external=("gs curved",),
    )
    with pytest.raises(ValueError, match="names of their own"):
        run.names(held)


def _trended(seen, channel, correlation):
    """`seen` with one channel's weekly spend at the Pearson `correlation` with the week's
    number: the week's number and the part of the channel's own history at right angles to it,
    mixed, about the history's mean at a fifth of it in standard deviation."""
    c = seen.channels.index(channel)
    own = seen.spend[:, c]
    week = np.arange(own.size, dtype=float)
    week = (week - week.mean()) / week.std()
    rest = own - own.mean()
    rest = rest - (rest @ week) / (week @ week) * week
    mixed = correlation * week + math.sqrt(1.0 - correlation**2) * rest / rest.std()
    spend = seen.spend.copy()
    spend[:, c] = own.mean() * (1.0 + 0.2 * mixed)
    return dataclasses.replace(seen, spend=spend)


def _without(seen, channel):
    """`seen` without one channel: its history, its box, its status quo and its tests, and the
    budget the other channels' status quo spends."""
    kept = np.array([c != channel for c in seen.channels])
    lift = seen.lift
    rows = np.array([c != channel for c in lift.channel])
    quo = seen.status_quo[kept]
    return dataclasses.replace(
        seen,
        channels=tuple(c for c in seen.channels if c != channel),
        spend=seen.spend[:, kept],
        lower=seen.lower[kept],
        upper=seen.upper[kept],
        status_quo=quo,
        budget=seen.planned * float(quo.sum()),
        lift=dataclasses.replace(
            lift,
            channel=tuple(c for c in lift.channel if c != channel),
            start=tuple(s for s, k in zip(lift.start, rows, strict=True) if k),
            x=lift.x[rows],
            delta_x=lift.delta_x[rows],
            delta_y=lift.delta_y[rows],
            sigma=lift.sigma[rows],
        ),
    )


def _with_radio(seen):
    """`seen` with a fourth channel, `radio`: `meta`'s history backwards, and its box, its status
    quo and its tests, and the budget the four channels' status quo spends."""
    m = seen.channels.index("meta")
    lift = seen.lift
    rows = [i for i, c in enumerate(lift.channel) if c == "meta"]
    quo = np.append(seen.status_quo, seen.status_quo[m])
    return dataclasses.replace(
        seen,
        channels=(*seen.channels, "radio"),
        spend=np.column_stack([seen.spend, seen.spend[::-1, m]]),
        lower=np.append(seen.lower, seen.lower[m]),
        upper=np.append(seen.upper, seen.upper[m]),
        status_quo=quo,
        budget=seen.planned * float(quo.sum()),
        lift=dataclasses.replace(
            lift,
            channel=(*lift.channel, *("radio" for _ in rows)),
            start=(*lift.start, *(lift.start[i] for i in rows)),
            x=np.append(lift.x, lift.x[rows]),
            delta_x=np.append(lift.delta_x, lift.delta_x[rows]),
            delta_y=np.append(lift.delta_y, lift.delta_y[rows]),
            sigma=np.append(lift.sigma, lift.sigma[rows]),
        ),
    )


def _still(planned, seen):
    """Whether a plan is the status quo to the last bit, claiming nothing."""
    return (
        np.array_equal(planned.weekly, seen.status_quo)
        and planned.gain == 0.0
        and tuple(planned.gain_interval) == (0.0, 0.0)
    )


@pytest.mark.parametrize("world", WORLDS, ids=name)
def test_without_a_trend_the_held_record_is_the_curved_ones_and_the_composites_play_alike(world):
    """No channel of the eight worlds follows the week's number: the held variant's plan, its
    claim, its route and every flag and setting it shares with the curved variant are that
    variant's to the last bit, and scored by the bench the composite `gs held` is `gs curved`'s
    regret, on `stacked` where the world is routed."""
    seen = observation(world)
    made = records(anchored.Anchored(), seen, smoke=False)
    curved, held = made["anchored curved"], made[HELD]
    assert held["error"] is None
    for field in ("weekly", "gain", "gain_interval"):
        assert held[field] == curved[field], field
    flags, settings = held["flags"], held["settings"]
    assert {k: v for k, v in flags.items() if k in curved["flags"]} == curved["flags"]
    assert {k: v for k, v in settings.items() if k in curved["settings"]} == curved["settings"]
    correlations = [f"{c} time correlation" for c in seen.channels]
    assert set(flags) - set(curved["flags"]) == {*correlations, "time held", "route withheld"}
    assert max(flags[c] for c in correlations) < 0.25
    assert (flags["time held"], flags["route withheld"]) == (False, False)
    assert set(settings) - set(curved["settings"]) == {"time_held", "held at the status quo"}
    assert (settings["time_held"], settings["held at the status quo"]) == (0.8, {})
    json.dumps(held, allow_nan=False)
    scored = score_world(
        FAMILIES[world["family"]],
        world["environment"],
        world["seed"],
        world["k"],
        {"anchored curved": curved, HELD: held, "stacked": world["stacked"]["record"]},
        archived=seen,
    )
    composed = compose(scored).arms
    routed = world["research"]["routed"]
    assert composed["gs held"].regret == composed["gs curved"].regret
    assert composed["gs held"].record["composite"] == {
        "router": HELD,
        "played": "stacked" if routed else HELD,
        "routed": routed,
    }


def test_of_two_channels_one_that_follows_the_week_is_held_and_the_plan_is_the_status_quo():
    seen = _trended(_without(observation(KEPT[0]), "meta"), "tv", RAMP)
    made = records(anchored.Anchored(), seen, smoke=False)
    curved, held = made["anchored curved"], made[HELD]
    assert (curved["error"], held["error"]) == (None, None)
    assert curved["weekly"] != seen.status_quo.tolist()
    assert held["weekly"] == seen.status_quo.tolist()
    assert (held["gain"], held["gain_interval"]) == (0.0, [0.0, 0.0])
    flags, settings = held["flags"], held["settings"]
    assert flags["tv time correlation"] == pytest.approx(RAMP, rel=1e-12, abs=0.0)
    assert flags["pla time correlation"] < 0.1
    assert (flags["time held"], flags["routed"], flags["route withheld"]) == (True, False, False)
    assert settings["time_held"] == 0.8
    assert settings["held at the status quo"] == {
        "tv": "its weekly spend's correlation with the week's number is 0.970 in absolute value,"
        " above 0.8"
    }
    assert "held at the status quo" not in curved["settings"]
    json.dumps(held, allow_nan=False)


@pytest.mark.parametrize("reason", REASONS)
def test_a_world_with_a_held_channel_is_never_routed_and_plays_the_status_quo_where_the_fit_routes(
    reason,
):
    """A fit whose tests read a convex curve, or whose kernel runs past the readout, routes the
    curved variant; with a channel held the held variant keeps both reasons as the fit read them,
    says the route was withheld, is not routed and plays the status quo, though two channels are
    free. The composite `gs held` then plays the variant, where `gs curved` plays `stacked`."""
    seen = observation(KEPT[0])
    if reason == "routed by slope":
        seen = _rising(seen)
    else:
        seen = dataclasses.replace(seen, kernel_length=anchored.READOUT + 1)
    seen = _trended(seen, "tv", RAMP)
    arm = anchored.Anchored()
    fitted = arm.fit(seen, smoke=False)
    curved, held = (arm.plan(fitted, v, seen) for v in ("anchored curved", HELD))
    assert (curved.flags["routed"], curved.flags[reason]) == (True, True)
    assert not np.array_equal(curved.weekly, seen.status_quo)
    assert {r: held.flags[r] for r in REASONS} == {r: curved.flags[r] for r in REASONS}
    assert (held.flags["routed"], held.flags["route withheld"], held.flags["time held"]) == (
        False,
        True,
        True,
    )
    assert _still(held, seen)
    assert "route withheld" not in curved.flags
    arms = {
        "anchored curved": _arm(0.04, routed=curved.flags["routed"]),
        HELD: _arm(0.03, routed=held.flags["routed"], cpu=2.0),
        "stacked": _arm(0.12, cpu=300.0),
    }
    composed = compose(_place(arms)).arms
    assert (composed["gs curved"].regret, composed["gs held"].regret) == (0.12, 0.03)
    assert composed["gs curved"].record["composite"]["played"] == "stacked"
    assert composed["gs held"].record["composite"] == {
        "router": HELD,
        "played": HELD,
        "routed": False,
    }
    assert composed["gs held"].record["cost"] == {"cpu_seconds": 2.0, "wall_seconds": 4.0}


@pytest.mark.parametrize(("channel", "raised"), [("pla", True), ("tv", False)])
def test_with_one_channel_of_three_held_the_other_two_are_planned_at_the_budget_in_their_box(
    channel, raised
):
    """The curved variant raises the first channel and cuts the last. Held, each stays at its
    status quo to the last bit, and the plan is the water-filling's on the same curves in the box
    whose floor and ceiling for that channel are its status quo; the claim is that plan's."""
    seen = _trended(observation(KEPT[0]), channel, RAMP)
    arm = anchored.Anchored()
    fitted = arm.fit(seen, smoke=False)
    curved, held = (arm.plan(fitted, v, seen) for v in ("anchored curved", HELD))
    c = seen.channels.index(channel)
    quo = seen.status_quo
    assert bool(curved.weekly[c] > quo[c]) is raised
    assert curved.weekly[c] != quo[c]
    assert held.weekly[c] == quo[c]
    free = np.arange(quo.size) != c
    assert np.all(held.weekly[free] >= seen.lower[free])
    assert np.all(held.weekly[free] <= seen.upper[free])
    assert np.all(held.weekly[free] != quo[free])
    assert held.weekly.sum() == pytest.approx(quo.sum(), rel=1e-12, abs=0.0)
    curves = anchored._curves(fitted, seen)
    assert curves is not None
    assert fitted.shrunk is not None
    level = fitted.shrunk.mean
    expected = water_fill_tabulated(
        [curves.margin(i, float(level[i])) for i in range(quo.size)],
        np.where(free, seen.lower, quo),
        np.where(free, seen.upper, quo),
        float(quo.sum()),
    )
    np.testing.assert_array_equal(held.weekly, expected)
    gain = seen.planned * float(level @ curves.rise(held.weekly, level))
    assert held.gain == pytest.approx(gain, rel=1e-12, abs=0.0)
    assert held.gain is not None
    assert held.gain_interval is not None
    assert curved.gain is not None
    low, high = held.gain_interval
    assert low < held.gain < high
    assert 0.0 < held.gain < curved.gain
    assert (held.flags["time held"], held.flags["routed"], held.flags["route withheld"]) == (
        True,
        False,
        False,
    )
    assert list(held.settings["held at the status quo"]) == [channel]
    assert held.settings["planner"] == "water-filling on a table"


@pytest.mark.parametrize("second", ["held", "boxed to a point"])
def test_with_one_channel_of_three_free_the_plan_is_the_status_quo_and_none_is_searched(
    second, monkeypatch
):
    """Two channels held, one of them falling with the week; or one held and one whose box is a
    point: one channel is free, the budget leaves it where it is, and the planner is not asked."""
    seen = _trended(observation(KEPT[0]), "pla", RAMP)
    if second == "held":
        seen = _trended(seen, "tv", -RAMP)
    else:
        point = np.array([c == "tv" for c in seen.channels])
        seen = dataclasses.replace(
            seen,
            lower=np.where(point, seen.status_quo, seen.lower),
            upper=np.where(point, seen.status_quo, seen.upper),
        )
    arm = anchored.Anchored()
    fitted = arm.fit(seen, smoke=False)
    assert not np.array_equal(arm.plan(fitted, "anchored curved", seen).weekly, seen.status_quo)

    def searched(*_, **__):
        msg = "one free channel has no plan to search"
        raise AssertionError(msg)

    monkeypatch.setattr(anchored, "water_fill_tabulated", searched)
    held = arm.plan(fitted, HELD, seen)
    assert _still(held, seen)
    named = ["pla", "tv"] if second == "held" else ["pla"]
    assert list(held.settings["held at the status quo"]) == named
    for channel in named:
        assert held.flags[f"{channel} time correlation"] == pytest.approx(RAMP, rel=1e-12, abs=0.0)
    assert (held.flags["time held"], held.flags["routed"]) == (True, False)


@pytest.mark.parametrize("box", ["ceilings short of the budget", "a floor above its ceiling"])
def test_a_free_box_that_holds_no_plan_at_the_budget_is_the_status_quo(box):
    """Television is held. In the first box, three channels', the ceilings reach the budget through
    television alone, so that with it held they fall short: the curved variant plans there. In the
    second, four channels', the fourth's floor lies above its ceiling while the floors and the
    ceilings still sum about the budget, and two channels are free: the curved variant's plan
    fails. The held variant plays the status quo in both."""
    seen = _trended(observation(KEPT[0]), "tv", RAMP)
    if box == "ceilings short of the budget":
        quo = seen.status_quo
        seen = dataclasses.replace(seen, upper=np.array([0.9, 1.0, 2.0]) * quo)
    else:
        seen = _with_radio(seen)
        quo = seen.status_quo
        seen = dataclasses.replace(
            seen,
            lower=np.array([0.5, 0.5, 0.5, 1.2]) * quo,
            upper=np.array([2.0, 2.0, 2.0, 0.9]) * quo,
        )
        assert seen.lower.sum() < quo.sum() < seen.upper.sum()
    made = records(anchored.Anchored(), seen, smoke=False)
    curved = made["anchored curved"]
    if box == "ceilings short of the budget":
        assert curved["error"] is None
        assert curved["weekly"][2] > quo[2]
    else:
        assert curved["error"].startswith("ValueError: no plan in the box")
    held = made[HELD]
    assert held["error"] is None
    assert held["weekly"] == quo.tolist()
    assert (held["gain"], held["gain_interval"]) == (0.0, [0.0, 0.0])
    assert list(held["settings"]["held at the status quo"]) == ["tv"]


def test_with_one_channel_of_four_held_the_other_three_are_planned():
    seen = _trended(_with_radio(observation(KEPT[0])), "tv", RAMP)
    made = records(anchored.Anchored(), seen, smoke=False)
    curved, held = made["anchored curved"], made[HELD]
    assert (curved["error"], held["error"]) == (None, None)
    weekly, quo = np.asarray(held["weekly"]), seen.status_quo
    assert curved["weekly"][2] != quo[2]
    assert weekly[2] == quo[2]
    free = np.arange(quo.size) != 2
    assert np.all(weekly[free] >= seen.lower[free])
    assert np.all(weekly[free] <= seen.upper[free])
    assert np.all(weekly[free] != quo[free])
    assert weekly.sum() == pytest.approx(quo.sum(), rel=1e-12, abs=0.0)
    assert held["gain"] > 0.0
    assert list(held["settings"]["held at the status quo"]) == ["tv"]


@pytest.mark.parametrize("drift", [0.0, 1e-12])
def test_a_channel_of_constant_spend_is_not_time_held(drift):
    """A spend that never moves, and one that moves in its last bits alone, along a straight line
    of the week's number: neither is read as a trend, and the plan is the curved variant's."""
    seen = observation(KEPT[0])
    spend = seen.spend.copy()
    spend[:, 2] = spend[:, 2].mean() * (1.0 + drift * np.arange(spend.shape[0]))
    seen = dataclasses.replace(seen, spend=spend)
    made = records(anchored.Anchored(), seen, smoke=False)
    curved, held = made["anchored curved"], made[HELD]
    assert (curved["error"], held["error"]) == (None, None)
    assert held["flags"]["tv time correlation"] == 0.0
    assert (held["flags"]["time held"], held["settings"]["held at the status quo"]) == (False, {})
    for field in ("weekly", "gain", "gain_interval"):
        assert held[field] == curved[field], field


@pytest.mark.parametrize(("correlation", "named"), [(0.7999, False), (0.8001, True)])
def test_a_channel_is_held_just_above_the_threshold_and_not_just_below(correlation, named):
    seen = _trended(observation(KEPT[0]), "tv", correlation)
    made = records(anchored.Anchored(), seen, smoke=False)
    curved, held = made["anchored curved"], made[HELD]
    assert (curved["error"], held["error"]) == (None, None)
    assert held["flags"]["tv time correlation"] == pytest.approx(correlation, rel=1e-12, abs=0.0)
    assert held["flags"]["time held"] is named
    assert list(held["settings"]["held at the status quo"]) == (["tv"] if named else [])
    assert curved["weekly"][2] != seen.status_quo[2]
    assert (held["weekly"][2] == float(seen.status_quo[2])) is named
    assert (held["weekly"] == curved["weekly"]) is not named


def test_at_k_0_a_held_channel_stays_and_the_others_take_the_corner():
    """An export without tests reads no curve: the held variant plans the corner of the shrunk
    margins in the box whose floor and ceiling for the held channel are its status quo."""
    seen = _trended(observation(KEPT[0], k=0, lift=lift_rows({})), "pla", RAMP)
    arm = anchored.Anchored()
    fitted = arm.fit(seen, smoke=False)
    corner, held = (arm.plan(fitted, v, seen) for v in ("anchored", HELD))
    quo = seen.status_quo
    assert corner.weekly[0] != quo[0]
    assert held.weekly[0] == quo[0]
    assert fitted.shrunk is not None
    first = np.arange(quo.size) == 0
    expected = plan_corner(
        fitted.shrunk.mean,
        np.where(first, quo, seen.lower),
        np.where(first, quo, seen.upper),
        float(quo.sum()),
    )
    np.testing.assert_array_equal(held.weekly, expected)
    assert not np.array_equal(held.weekly, quo)
    assert held.gain == pytest.approx(
        seen.planned * float(fitted.shrunk.mean @ (held.weekly - quo)), rel=1e-12, abs=0.0
    )
    assert (held.settings["planner"], held.flags["routed"], held.flags["time held"]) == (
        "corner",
        False,
        True,
    )


def test_the_held_composite_plays_stacked_where_the_held_variant_routes_and_keeps_its_name():
    stacked = _arm(0.12, cpu=300.0)
    arms = {HELD: _arm(0.03, routed=True, cpu=2.0), "stacked": stacked}
    routed = compose(_place(arms)).arms["gs held"]
    assert (routed.regret, routed.record["composite"]["played"]) == (0.12, "stacked")
    assert routed.record["cost"] == {"cpu_seconds": 302.0, "wall_seconds": 604.0}
    assert "gs curved" not in compose(_place(arms)).arms
    assert GS_HELD.name not in compose(_place({"anchored curved": arms[HELD]})).arms
    held = Declaration(
        family=0,
        environments=("drawn",),
        worlds=2,
        pilot=True,
        rungs=(0,),
        arms=("margin",),
        external=("gs held",),
    )
    with pytest.raises(ValueError, match="names of their own"):
        run.names(held)


def _least(spend):
    """`SPEND_FLOOR` times each channel's least weekly spend over a history."""
    return anchored.SPEND_FLOOR * np.asarray(spend, dtype=float).min(axis=0)


UNRAISED = [w for w in WORLDS if np.all(_least(w["spend"]) <= np.asarray(w["lower"]))]
"""The worlds whose every floor lies at or above the floor the floored variant raises it to."""


def test_the_worlds_hold_floors_raised_and_not_routed_and_not():
    raised = [w for w in WORLDS if w not in UNRAISED]
    assert (len(UNRAISED), len(raised)) == (4, 4)
    for group in (UNRAISED, raised):
        assert {w["research"]["routed"] for w in group} == {True, False}


@pytest.mark.parametrize("world", UNRAISED, ids=name)
def test_where_no_floor_is_raised_the_floored_record_is_the_held_ones_but_never_routed(world):
    """Every channel's floor lies at or above `SPEND_FLOOR` times its least weekly spend: the
    floored variant's plan, its claim and its interval are the held variant's to the last bit, and
    so is every flag and setting the two share but the route, false on a routed world too, its
    reasons kept as the fit read them."""
    seen = observation(world)
    made = records(anchored.Anchored(), seen, smoke=False)
    held, floored = made[HELD], made[FLOORED]
    assert (held["error"], floored["error"]) == (None, None)
    for field in ("weekly", "gain", "gain_interval"):
        assert floored[field] == held[field], field
    assert held["flags"]["routed"] is world["research"]["routed"]
    assert floored["flags"] == held["flags"] | {"routed": False, "floor applied": False}
    assert floored["settings"] == held["settings"] | {"spend_floor": 0.9}
    json.dumps(floored, allow_nan=False)


def test_where_the_held_plan_lies_below_the_floor_the_floored_plan_lies_on_it():
    """The held variant plans meta at its box's floor, half its status quo, below 0.9 times its
    least weekly spend. The floored variant plans it at that raised floor to the last bit: the plan
    is the water-filling's on the same curves in the box so raised, at the budget, and claims that
    plan's gain, less than the held plan's in its wider box."""
    seen = observation(KEPT[0])
    arm = anchored.Anchored()
    fitted = arm.fit(seen, smoke=False)
    held, floored = (arm.plan(fitted, v, seen) for v in (HELD, FLOORED))
    m = seen.channels.index("meta")
    least, quo = _least(seen.spend), seen.status_quo
    assert held.weekly[m] == seen.lower[m] < least[m]
    assert floored.weekly[m] == least[m]
    curves = anchored._curves(fitted, seen)
    assert curves is not None
    assert fitted.shrunk is not None
    level = fitted.shrunk.mean
    expected = water_fill_tabulated(
        [curves.margin(c, float(level[c])) for c in range(quo.size)],
        np.maximum(seen.lower, np.minimum(least, seen.upper)),
        seen.upper,
        float(quo.sum()),
    )
    np.testing.assert_array_equal(floored.weekly, expected)
    assert floored.weekly.sum() == pytest.approx(quo.sum(), rel=1e-12, abs=0.0)
    gain = seen.planned * float(level @ curves.rise(floored.weekly, level))
    assert floored.gain == pytest.approx(gain, rel=1e-12, abs=0.0)
    assert floored.gain is not None
    assert held.gain is not None
    assert 0.0 < floored.gain < held.gain
    assert (floored.flags["floor applied"], floored.flags["routed"]) == (True, False)
    assert floored.settings["spend_floor"] == 0.9


def test_a_held_channel_stays_at_its_status_quo_and_the_free_channels_floors_are_raised():
    """Television follows the week and its history never spent as little as its status quo: it
    stays there to the last bit, and the other two channels are planned at the budget in the box
    whose floors are raised as the variant raises them."""
    seen = _trended(observation(KEPT[0]), "tv", RAMP)
    t = seen.channels.index("tv")
    spend = seen.spend.copy()
    spend[:, t] *= 2.0 * seen.status_quo[t] / spend[:, t].min()
    seen = dataclasses.replace(seen, spend=spend)
    arm = anchored.Anchored()
    fitted = arm.fit(seen, smoke=False)
    floored = arm.plan(fitted, FLOORED, seen)
    least, quo = _least(seen.spend), seen.status_quo
    assert least[t] > quo[t]
    assert floored.weekly[t] == quo[t]
    held = np.arange(quo.size) == t
    curves = anchored._curves(fitted, seen)
    assert curves is not None
    assert fitted.shrunk is not None
    level = fitted.shrunk.mean
    expected = water_fill_tabulated(
        [curves.margin(c, float(level[c])) for c in range(quo.size)],
        np.where(held, quo, np.maximum(seen.lower, np.minimum(least, seen.upper))),
        np.where(held, quo, seen.upper),
        float(quo.sum()),
    )
    np.testing.assert_array_equal(floored.weekly, expected)
    assert (floored.flags["time held"], floored.flags["floor applied"]) == (True, True)
    assert list(floored.settings["held at the status quo"]) == ["tv"]


def test_floors_raised_past_the_budget_leave_the_guards_box():
    """The history spent two and a half times as much each week: its least weekly spends, so
    raised, sum past the budget, and the floored record is the held one's but for its route."""
    seen = observation(KEPT[0])
    seen = dataclasses.replace(seen, spend=2.5 * seen.spend)
    raised = np.maximum(seen.lower, np.minimum(_least(seen.spend), seen.upper))
    assert raised.sum() > seen.status_quo.sum()
    made = records(anchored.Anchored(), seen, smoke=False)
    held, floored = made[HELD], made[FLOORED]
    assert (held["error"], floored["error"]) == (None, None)
    for field in ("weekly", "gain", "gain_interval"):
        assert floored[field] == held[field], field
    assert floored["flags"]["floor applied"] is False
    assert floored["weekly"] != seen.status_quo.tolist()


def test_a_floor_above_its_ceiling_is_kept_and_the_floored_plan_fails_as_the_held_one():
    """Radio's floor lies above its ceiling, and its least weekly spend below both: the variant
    never lowers a floor, so the box holds no plan, as in the held variant."""
    seen = _with_radio(observation(KEPT[0]))
    quo = seen.status_quo
    seen = dataclasses.replace(
        seen,
        lower=np.array([0.5, 0.5, 0.5, 1.2]) * quo,
        upper=np.array([2.0, 2.0, 2.0, 0.9]) * quo,
    )
    assert _least(seen.spend)[3] < seen.upper[3] < seen.lower[3]
    made = records(anchored.Anchored(), seen, smoke=False)
    held, floored = made[HELD], made[FLOORED]
    assert held["error"].startswith("ValueError: no plan in the box")
    assert floored["error"] == held["error"]
