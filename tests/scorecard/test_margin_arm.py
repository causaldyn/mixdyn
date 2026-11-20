"""The margin arm (`scorecard.arms.margin`): declared as `margin`, it runs a world end to end beside
the built-in arms; on six of M2's worlds its margins, its shrunk means and its plan are the research
code's, which found the rule; its record holds a plan in the box on the budget, the claimed gain
the scorecard's reading gives on straight-line channels with its interval, and what it did not read
or give; no positive margin plays the status quo; family 14's world fits; an export without the
promotion or the price fits without them, family 17's among them; a geo export, a budget the
rule does not spend or a fit the data leave degenerate is a failed record, never a plan of NaN;
`margin guarded` holds a channel the prior sets the margin of at its status quo, family 19's flat
channel among them, plans the others by the corner on their own shrinkage, and with no channel
held plans as `margin`; and `margin ranged` plans the corner in the box cut to the spend the
history logged, never outside it, and an export whose cut box holds no plan is its failed plan
alone; `margin curved` water-fills the curved regression's shrunk margins inside that box cut to
each channel's adstock, claims the curves' rise, plans as `margin ranged` where its curves are
lines and its box the observed range, fails alone where its fit is degenerate, and plays the status
quo where no margin is positive; and `margin curved noise` is that variant on the margins of the
regression under its mixture of error models, its shrunk margins and its plan the research code's
on the seven worlds, its record naming the mixture, while `margin curved`'s record reads none of
it; it holds a channel whose columns the nuisance columns span, family 19's flat channel, at its
status quo, and its plan is the same whichever sign rounding would leave that channel's margin;
and `margin held` is that variant with the time guard: it plays the status quo on family 17's
twins and says which channel it held and at what correlation, plans as `margin curved noise` where
no spend follows the week, holds a channel just above the guard's threshold and none just below,
plans the other channels at the budget where one of three is held, its margin still in the
shrinkage, plays the status quo where one channel is left free or the free box holds no plan,
and holds a flat channel as `margin curved noise` does, not by the guard; and an export whose
price never moves, whose price is a straight line in the week or whose promotion flag is its
promoted weeks' dummies' sum is a failed record of the two variants on the mixture alone, which
the bench scores at the status quo."""

from __future__ import annotations

import dataclasses
import json
import math
import re
from statistics import NormalDist

import numpy as np
import pytest

from margin_worlds import LONG, NOISE, PARITY, name

pytest.importorskip("causaldyn_bench")

from causaldyn_bench.budget_regret import lift_rows
from causaldyn_bench.scorecard.flat import FLAT
from causaldyn_bench.scorecard.geo import GEO
from causaldyn_bench.scorecard.observe import Observation
from causaldyn_bench.scorecard.scoring import score_world
from causaldyn_bench.scorecard.track_m2 import TRACK_M2
from causaldyn_bench.scorecard.twins import TWINS
from chc import response

from mixdyn.decisions.margin_plan import (
    adstock_range,
    observed_range,
    plan_corner,
    shrink_margins,
    time_held,
    water_fill,
)
from mixdyn.models.margin_mmm import (
    DegenerateFitError,
    fit_curved_margins,
    fit_margins,
    promoted_phases,
)
from scorecard import report, run
from scorecard.arms import margin
from scorecard.arms import panel as reading
from scorecard.arms.base import records
from scorecard.layout import Declaration, Layout, Limits

RELATIVE = 1e-9
LIMITS = Limits(memory="4G", timeout=600.0, slots=1)
COMMON = {"version", "seed", "digest", "family", "environment", "k", "python", "versions"}


def observation(world):
    """The world's k = 0 export as the arm reads it, rebuilt from the fixture."""

    def series(values):
        return np.asarray(values, dtype=float)

    return Observation(
        family=world["family"],
        environment=world["environment"],
        seed=world["seed"],
        k=0,
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
        lift=lift_rows({}),
    )


@pytest.fixture(scope="module")
def made():
    """The arm's records of the first parity world."""
    return records(margin.Margin(), observation(PARITY[0]), smoke=False)["margin"]


def test_the_arm_is_declared_as_margin_reads_no_test_and_plans_the_market():
    assert run.ARMS["margin"] is margin.Margin
    arm = margin.Margin()
    assert arm.variants == (
        "margin",
        "margin guarded",
        "margin ranged",
        "margin curved",
        "margin curved noise",
        "margin held",
    )
    assert (arm.tests, arm.cells) == (False, False)


@pytest.mark.parametrize("world", PARITY, ids=name)
def test_the_arms_margins_shrunk_means_and_plan_are_the_research_codes(world):
    seen = observation(world)
    arm = margin.Margin()
    fitted = arm.fit(seen, smoke=False)
    research = world["research"]
    np.testing.assert_allclose(fitted.margins.mean, research["mean"], rtol=RELATIVE, atol=0.0)
    np.testing.assert_allclose(fitted.margins.cov, research["cov"], rtol=RELATIVE, atol=0.0)
    assert fitted.shrunk is not None
    np.testing.assert_allclose(fitted.shrunk.mean, research["shrunk"], rtol=RELATIVE, atol=0.0)
    np.testing.assert_array_equal(arm.plan(fitted, "margin", seen).weekly, research["plan"])


def test_the_record_holds_a_plan_in_the_box_on_the_budget_and_what_it_did_not_read(made):
    seen = observation(PARITY[0])
    assert set(made) == COMMON | {"weekly", "flags", "settings", "error", "gain"} | {
        "gain_interval",
        "n/a",
        "cost",
    }
    assert made["error"] is None
    weekly = np.asarray(made["weekly"])
    assert np.all(weekly >= seen.lower)
    assert np.all(weekly <= seen.upper)
    assert weekly.sum() * seen.planned == pytest.approx(seen.budget, rel=1e-12, abs=0.0)
    absent = made["n/a"]
    assert absent["lift tests"].endswith("every rung's plan is the plan of k = 0")
    assert {"forecast", "response"} <= set(absent)
    settings = made["settings"]
    assert (settings["lift_rows_read"], settings["planner"]) == (0, "corner")
    assert settings["retentions"] == [0.1, 0.3, 0.5, 0.7, 0.9]
    assert settings["kernel_length"] == seen.kernel_length
    flags = made["flags"]
    research = PARITY[0]["research"]
    for c, channel in enumerate(seen.channels):
        assert flags[f"{channel} margin"] == pytest.approx(research["mean"][c], rel=RELATIVE)
        assert flags[f"{channel} margin sd"] == pytest.approx(
            math.sqrt(research["cov"][c][c]), rel=RELATIVE
        )
        assert flags[f"{channel} shrunk margin"] == pytest.approx(
            research["shrunk"][c], rel=RELATIVE
        )
        assert flags[f"{channel} retention"] == pytest.approx(
            research["alpha_mean"][c], rel=RELATIVE
        )
    assert (flags["rows"], flags["burn-in rows"], flags["events"]) == (
        research["n"],
        PARITY[0]["kernel_length"] - 1,
        research["events"],
    )
    json.dumps(made, allow_nan=False)


def test_the_claim_is_the_scorecards_reading_on_straight_line_channels(made):
    """A channel whose curve is a line through 0 at the shrunk margin, read by
    `scorecard.arms.panel.gains` over the quarter and the kernel's tail, the history's carryover
    in: the claim; its interval the normal one of the shrunk posterior at 90 %."""
    seen = observation(PARITY[0])
    fitted = margin.Margin().fit(seen, smoke=False)
    assert fitted.shrunk is not None
    lines = [
        response.Channel(
            response.GeometricAdstock(0.5, length=seen.kernel_length, normalized=True),
            response.Power(scale=1.0, exponent=1.0),
            float(m),
        )
        for m in fitted.shrunk.mean
    ]
    weekly = np.asarray(made["weekly"])
    (read,) = reading.gains([lines], seen, weekly)
    assert made["gain"] == pytest.approx(float(read), rel=1e-9, abs=0.0)
    move = weekly - seen.status_quo
    spread = seen.planned * math.sqrt(float(move @ fitted.shrunk.cov @ move))
    z = NormalDist().inv_cdf(0.95)
    low, high = made["gain_interval"]
    assert (low, high) == pytest.approx(
        (made["gain"] - z * spread, made["gain"] + z * spread), rel=1e-12
    )


def test_no_positive_margin_plays_the_status_quo_and_claims_nothing():
    seen = observation(PARITY[0])
    arm = margin.Margin()
    fitted = dataclasses.replace(arm.fit(seen, smoke=False), shrunk=None)
    planned = arm.plan(fitted, "margin", seen)
    np.testing.assert_array_equal(planned.weekly, seen.status_quo)
    assert (planned.gain, planned.gain_interval) == (0.0, (0.0, 0.0))


def test_a_kernel_as_long_as_the_history_fits_through_the_arm():
    """Family 14's world, on which the research code raised."""
    seen = observation(LONG)
    record = records(margin.Margin(), seen, smoke=False)["margin"]
    assert record["error"] is None
    assert (record["flags"]["burn-in rows"], record["flags"]["rows"]) == (13, 143)
    assert np.all(np.isfinite(record["weekly"]))


def test_a_fit_the_data_leave_degenerate_is_a_failed_record_not_a_nan_plan():
    """The first 20 weeks of a world leave the design no residual degree of freedom."""
    seen = observation(PARITY[0])
    short = dataclasses.replace(
        seen,
        sales=seen.sales[:20],
        spend=seen.spend[:20],
        controls={k: v[:20] for k, v in seen.controls.items()},
        roi_window=(1, 20),
    )
    record = records(margin.Margin(), short, smoke=False)["margin"]
    assert record["error"].startswith("DegenerateFitError: ")
    assert record["failure_kind"] == "fit"
    assert "weekly" not in record


@pytest.mark.parametrize("kept", [(), ("promotion",), ("price",)], ids=str)
def test_an_export_without_a_control_fits_without_it_and_names_the_controls_read(kept):
    seen = observation(PARITY[0])
    bare = dataclasses.replace(
        seen,
        controls={k: seen.controls[k] for k in kept},
        future_controls={k: seen.future_controls[k] for k in kept},
    )
    record = records(margin.Margin(), bare, smoke=False)["margin"]
    assert record["error"] is None
    assert record["settings"]["controls"] == list(kept)
    fit = fit_margins(
        seen.sales,
        seen.spend,
        seen.controls["promotion"] if "promotion" in kept else None,
        seen.controls["price"] if "price" in kept else None,
        seen.kernel_length,
    )
    for c, channel in enumerate(seen.channels):
        assert record["flags"][f"{channel} margin"] == fit.mean[c]
    assert record["flags"]["events"] == (PARITY[0]["research"]["events"] if "price" in kept else 0)
    weekly = np.asarray(record["weekly"])
    assert np.all((weekly >= seen.lower) & (weekly <= seen.upper))
    assert weekly.sum() * seen.planned == pytest.approx(seen.budget, rel=1e-12, abs=0.0)
    json.dumps(record, allow_nan=False)


def test_family_17s_export_without_controls_fits():
    """The twins set their status quo apart from the history, not as its last year's mean; the
    ranged variant's cut box still holds a plan on their budget."""
    environment = next(iter(TWINS.pilots))
    seen = TWINS.observe(TWINS.world(environment, TWINS.pilots[environment][0]), 0)
    assert dict(seen.controls) == {}
    made = records(margin.Margin(), seen, smoke=False)
    for record in made.values():
        assert record["error"] is None
        assert record["settings"]["controls"] == []
        weekly = np.asarray(record["weekly"])
        assert np.all((weekly >= seen.lower) & (weekly <= seen.upper))


def test_a_geo_export_and_its_market_are_refused():
    """The harness hands the arm a geo export's market (`scorecard.arms.base.records`)."""
    seen = GEO.observe(GEO.world("policy-0", GEO.pilots["policy-0"][0]), 0)
    record = records(margin.Margin(), seen, smoke=False)["margin"]
    assert record["error"] == f"ValueError: {margin.GEO}"
    assert record["failure_kind"] == "fit"
    for read in (seen, reading.market(seen)):
        with pytest.raises(ValueError, match="a geo export is refused"):
            margin.Margin().fit(read, smoke=False)


@pytest.mark.parametrize("world", PARITY, ids=name)
def test_with_no_channel_held_the_guarded_plan_and_claim_are_the_margin_variants(world):
    made = records(margin.Margin(), observation(world), smoke=False)
    plain, guarded = made["margin"], made["margin guarded"]
    assert not any(guarded["flags"][f"{c} held"] for c in world["channels"])
    assert guarded["settings"]["held at the status quo"] == {}
    assert guarded["weekly"] == plain["weekly"]
    assert (guarded["gain"], guarded["gain_interval"]) == (plain["gain"], plain["gain_interval"])


@pytest.mark.parametrize("environment", ["flat-pla", "flat-meta", "flat-tv"])
def test_the_guarded_variant_holds_family_19s_flat_channel_at_its_status_quo(environment):
    """The flat channel's column is the trend's: `margin` filled it on the environment's first
    pilot world, and `margin guarded` holds it and says why."""
    seen = FLAT.observe(FLAT.world(environment, FLAT.pilots[environment][0]), 0)
    flat = seen.channels.index(environment.removeprefix("flat-"))
    made = records(margin.Margin(), seen, smoke=False)
    guarded = made["margin guarded"]
    assert guarded["error"] is None
    held = [guarded["flags"][f"{c} held"] for c in seen.channels]
    assert held == [c == flat for c in range(len(seen.channels))]
    assert guarded["flags"][f"{seen.channels[flat]} prior share"] == pytest.approx(
        1.0, rel=0.0, abs=1e-12
    )
    (reason,) = guarded["settings"]["held at the status quo"].values()
    assert reason.startswith("the prior's precision on its margin exceeds the data's")
    assert guarded["flags"][f"{seen.channels[flat]} shrunk margin"] is None
    weekly = np.asarray(guarded["weekly"])
    assert weekly[flat] == seen.status_quo[flat]
    assert weekly.sum() * seen.planned == pytest.approx(seen.budget, rel=1e-12, abs=0.0)
    assert made["margin"]["weekly"][flat] > seen.status_quo[flat]


def test_a_held_channel_stays_at_its_status_quo_and_the_others_are_shrunk_and_cornered_alone():
    seen = observation(PARITY[0])
    arm = margin.Margin()
    fitted = arm.fit(seen, smoke=False)
    held = np.array([True, False, False])
    guarded = margin._guarded(fitted.margins, fitted.shrunk, held)
    assert guarded is not None
    alone = shrink_margins(fitted.margins.mean[1:], fitted.margins.cov[1:, 1:])
    assert alone is not None
    np.testing.assert_array_equal(guarded.mean, alone.mean)
    planned = arm.plan(
        dataclasses.replace(fitted, held=held, guarded=guarded), "margin guarded", seen
    )
    quo = seen.status_quo
    assert planned.weekly[0] == quo[0]
    np.testing.assert_allclose(
        planned.weekly[1:],
        plan_corner(alone.mean, seen.lower[1:], seen.upper[1:], quo.sum() - quo[0]),
        rtol=1e-12,
        atol=0.0,
    )
    move = (planned.weekly - quo)[1:]
    assert planned.gain == pytest.approx(seen.planned * float(alone.mean @ move), rel=1e-12)


def test_with_every_channel_held_the_guarded_plan_is_the_status_quo():
    seen = observation(PARITY[0])
    arm = margin.Margin()
    fitted = arm.fit(seen, smoke=False)
    held = np.ones(len(seen.channels), dtype=bool)
    assert margin._guarded(fitted.margins, fitted.shrunk, held) is None
    planned = arm.plan(dataclasses.replace(fitted, held=held, guarded=None), "margin guarded", seen)
    np.testing.assert_array_equal(planned.weekly, seen.status_quo)
    assert (planned.gain, planned.gain_interval) == (0.0, (0.0, 0.0))
    assert set(planned.settings["held at the status quo"]) == set(seen.channels)


@pytest.mark.parametrize("world", PARITY, ids=name)
def test_the_ranged_plan_is_the_corner_in_the_box_cut_to_the_spend_the_history_logged(world):
    """`margin`'s plan stays the research code's; `margin ranged`'s is the corner on the same
    shrunk means in the cut box, no channel outside its history's range."""
    seen = observation(world)
    made = records(margin.Margin(), seen, smoke=False)
    np.testing.assert_array_equal(made["margin"]["weekly"], world["research"]["plan"])
    ranged = made["margin ranged"]
    assert ranged["error"] is None
    fitted = margin.Margin().fit(seen, smoke=False)
    assert fitted.shrunk is not None
    budget = float(seen.status_quo.sum())
    floor, ceiling = observed_range(seen.spend, seen.lower, seen.upper, budget)
    np.testing.assert_array_equal(
        ranged["weekly"], plan_corner(fitted.shrunk.mean, floor, ceiling, budget)
    )
    assert ranged["flags"]["out of range"] == pytest.approx(0.0, rel=0.0, abs=1e-12)
    assert ranged["settings"]["planning range"].startswith("each channel's box cut to")
    move = np.asarray(ranged["weekly"]) - seen.status_quo
    assert ranged["gain"] == pytest.approx(
        seen.planned * float(fitted.shrunk.mean @ move), rel=1e-12
    )


def test_an_export_whose_cut_box_holds_no_plan_is_a_failed_ranged_plan_alone():
    """A history scaled up until every channel's least lies above its status quo, so the floors
    sum past the budget."""
    seen = observation(PARITY[0])
    scale = 2.0 * float(np.max(seen.status_quo / seen.spend.min(axis=0)))
    raised = dataclasses.replace(seen, spend=scale * seen.spend)
    assert np.all(raised.spend.min(axis=0) > seen.status_quo)
    made = records(margin.Margin(), raised, smoke=False)
    assert made["margin ranged"]["error"].startswith("ObservedRangeError: the box cut to")
    assert made["margin ranged"]["failure_kind"] == "plan"
    assert made["margin"]["error"] is None
    assert made["margin guarded"]["error"] is None


def test_a_budget_that_is_not_the_status_quos_spend_is_refused():
    seen = observation(PARITY[0])
    other = dataclasses.replace(seen, budget=seen.budget * 1.01)
    record = records(margin.Margin(), other, smoke=False)["margin"]
    assert record["error"].startswith("ValueError: the rule spends the status quo's weekly total")
    assert record["failure_kind"] == "plan"


def test_the_arm_runs_a_world_end_to_end_and_is_scored_beside_the_built_in_arms(tmp_path, capsys):
    """Declared with its limits, fitted once for both rungs, scored and reported: each rung's
    record is the k = 0 plan, which says it read no test."""
    layout = Layout(tmp_path / "run")
    run.declare(
        layout,
        Declaration(
            family=0,
            environments=("drawn",),
            worlds=1,
            pilot=True,
            rungs=(0, 4),
            arms=("margin",),
            smoke=True,
            limits={"margin": LIMITS},
        ),
    )
    run.export_all(layout)
    (seed,) = layout.read().seeds("drawn")
    run.one(layout, "margin", "drawn", seed, (0, 4))
    first, second = (
        json.loads(layout.record("margin", 0, "drawn", k, seed).read_text()) for k in (0, 4)
    )
    assert first["error"] is None
    assert (first["k"], second["k"]) == (0, 4)
    assert second["weekly"] == first["weekly"]
    assert second["cost"] == {"fitted with k": 0}
    run.score(layout)
    lines = [json.loads(line) for line in layout.scores.read_text().splitlines()]
    assert sorted(line["k"] for line in lines) == [0, 4]
    for line in lines:
        assert line["arms"]["margin"]["failure"] is None
        assert line["arms"]["margin"]["moved"] is None
        assert math.isfinite(line["arms"]["margin"]["regret"])
    report.main(layout, ["margin"])
    assert any(
        line.split()[:2] == ["margin", "regret"] for line in capsys.readouterr().out.splitlines()
    )


@pytest.mark.parametrize("world", [*PARITY, LONG], ids=name)
def test_the_curved_plan_water_fills_the_curved_margins_inside_the_adstock_range(world):
    """`margin curved`'s record: the shrunk margins at the status quo of the curved regression,
    water-filled along each channel's curve inside the box cut to its history's spend and then to
    its adstock, on the budget; the claim the curves' rise over the quarter."""
    seen = observation(world)
    record = records(margin.Margin(), seen, smoke=False)["margin curved"]
    assert record["error"] is None
    quo = seen.status_quo
    budget = float(quo.sum())
    curved = fit_curved_margins(
        seen.sales,
        seen.spend,
        seen.controls["promotion"],
        seen.controls["price"],
        seen.kernel_length,
        quo,
        seen.planned,
    )
    post = shrink_margins(curved.margins.mean, curved.margins.cov)
    assert post is not None
    lower, upper = adstock_range(
        seen.spend, curved.planning_adstock(), seen.lower, seen.upper, budget
    )
    weekly = np.asarray(record["weekly"])
    np.testing.assert_array_equal(
        weekly, water_fill(post.mean, curved.curve.slope, curved.shaped, lower, upper, budget)
    )
    floor, ceiling = observed_range(seen.spend, seen.lower, seen.upper, budget)
    assert np.all((weekly >= lower) & (weekly <= upper))
    assert np.all((lower >= floor) & (upper <= ceiling))
    assert weekly.sum() * seen.planned == pytest.approx(seen.budget, rel=1e-12, abs=0.0)
    rise = curved.curve.rise(weekly, quo)
    assert record["gain"] == pytest.approx(seen.planned * float(post.mean @ rise), rel=1e-12)
    spread = seen.planned * math.sqrt(float(rise @ post.cov @ rise))
    z = NormalDist().inv_cdf(0.95)
    low, high = record["gain_interval"]
    assert (low, high) == pytest.approx(
        (record["gain"] - z * spread, record["gain"] + z * spread), rel=1e-12
    )
    flags = record["flags"]
    for c, channel in enumerate(seen.channels):
        assert flags[f"{channel} half-saturation"] == 16.0 * quo[c]
        assert flags[f"{channel} margin"] == curved.margins.mean[c]
        assert (flags[f"{channel} planning floor"], flags[f"{channel} planning ceiling"]) == (
            lower[c],
            upper[c],
        )
    assert record["settings"]["planner"] == "water-filling"
    assert record["settings"]["half_saturation"] == 16.0
    assert record["n/a"]["lift tests"].endswith("every rung's plan is the plan of k = 0")
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize("world", PARITY, ids=name)
def test_with_lines_for_curves_in_the_ranged_box_the_curved_plan_is_the_ranged_plan(
    world, monkeypatch
):
    """At an infinite half-saturation the curves are lines and their margins `margin`'s, so in
    `margin ranged`'s box, the observed range, the plan and its claim are `margin ranged`'s."""
    monkeypatch.setattr(margin, "HALF_SATURATION", math.inf)
    monkeypatch.setattr(
        margin,
        "adstock_range",
        lambda spend, _adstock, lower, upper, budget: observed_range(spend, lower, upper, budget),
    )
    made = records(margin.Margin(), observation(world), smoke=False)
    curved, ranged = made["margin curved"], made["margin ranged"]
    assert curved["error"] is None
    np.testing.assert_allclose(curved["weekly"], ranged["weekly"], rtol=1e-12, atol=0.0)
    assert curved["gain"] == pytest.approx(ranged["gain"], rel=1e-9, abs=0.0)


def test_a_curved_fit_the_data_leave_degenerate_fails_the_curved_variants_alone(monkeypatch):
    def degenerate(*_args, noise):
        raise DegenerateFitError("the curved fit at retentions (0.1, 0.1, 0.1) leaves nothing")

    monkeypatch.setattr(margin, "fit_curved_margins", degenerate)
    made = records(margin.Margin(), observation(PARITY[0]), smoke=False)
    for variant in ("margin curved", "margin curved noise", "margin held"):
        assert made[variant]["error"].startswith("DegenerateFitError: the curved fit")
        assert made[variant]["failure_kind"] == "plan"
    for variant in ("margin", "margin guarded", "margin ranged"):
        assert made[variant]["error"] is None


def test_a_fit_the_mixture_leaves_degenerate_fails_the_noise_variant_alone(monkeypatch):
    """`margin curved`'s regression has no mixture: where the mixture's alone is degenerate, as
    under nuisance columns the weeks do not tell apart, `margin curved` still plans."""
    fit = margin.fit_curved_margins

    def degenerate(*args, noise):
        if noise is not None:
            raise DegenerateFitError("the weeks read do not tell the nuisance columns apart")
        return fit(*args, noise=noise)

    monkeypatch.setattr(margin, "fit_curved_margins", degenerate)
    made = records(margin.Margin(), observation(PARITY[0]), smoke=False)
    for variant in ("margin curved noise", "margin held"):
        assert made[variant]["error"].startswith("DegenerateFitError: the weeks read")
        assert made[variant]["failure_kind"] == "plan"
    for variant in ("margin", "margin guarded", "margin ranged", "margin curved"):
        assert made[variant]["error"] is None


def test_no_positive_curved_margin_plays_the_status_quo_and_claims_nothing(monkeypatch):
    seen = observation(PARITY[0])
    arm = margin.Margin()
    fitted = arm.fit(seen, smoke=False)
    monkeypatch.setattr(margin, "shrink_margins", lambda _mean, _cov: None)
    planned = arm.plan(fitted, "margin curved", seen)
    np.testing.assert_array_equal(planned.weekly, seen.status_quo)
    assert (planned.gain, planned.gain_interval) == (0.0, (0.0, 0.0))


@pytest.mark.parametrize("world", [*PARITY, LONG], ids=name)
def test_the_noise_variants_shrunk_margins_and_plan_are_the_research_codes(world):
    """The frozen research rule on the seven worlds, family 14's among them: the margins at the
    status quo of the curved regression under the mixture, their shrunk means, and the plan."""
    seen = observation(world)
    record = records(margin.Margin(), seen, smoke=False)["margin curved noise"]
    assert record["error"] is None
    research = NOISE[world["seed"]]
    np.testing.assert_allclose(record["weekly"], research["plan"], rtol=RELATIVE, atol=0.0)
    flags = record["flags"]
    for c, channel in enumerate(seen.channels):
        assert flags[f"{channel} margin"] == pytest.approx(research["mean"][c], rel=RELATIVE)
        assert flags[f"{channel} margin sd"] == pytest.approx(
            math.sqrt(research["cov"][c][c]), rel=RELATIVE
        )
        assert flags[f"{channel} shrunk margin"] == pytest.approx(
            research["shrunk"][c], rel=RELATIVE
        )
    assert flags["phi"] == pytest.approx(research["rho"], rel=RELATIVE, abs=0.0)
    assert flags["ar share"] == pytest.approx(research["omega"], rel=RELATIVE, abs=0.0)
    assert flags["promoted variance"] == pytest.approx(research["h"], rel=RELATIVE, abs=0.0)


@pytest.mark.parametrize("world", [PARITY[0], PARITY[4]], ids=name)
def test_the_noise_variant_plans_as_the_curved_variant_on_the_mixtures_margins(world):
    """`margin curved noise`'s record: `margin curved`'s shrinkage, curves, box and water-filling
    on the curved regression under the arm's mixture of error models; its settings name the
    mixture, and `margin curved`'s record, of the regression without one, reads none of it."""
    seen = observation(world)
    made = records(margin.Margin(), seen, smoke=False)
    record, plain = made["margin curved noise"], made["margin curved"]
    assert record["error"] is None
    quo = seen.status_quo
    budget = float(quo.sum())
    curved = fit_curved_margins(
        seen.sales,
        seen.spend,
        seen.controls["promotion"],
        seen.controls["price"],
        seen.kernel_length,
        quo,
        seen.planned,
        noise=margin.NOISE_MIXTURE,
    )
    post = shrink_margins(curved.margins.mean, curved.margins.cov)
    assert post is not None
    lower, upper = adstock_range(
        seen.spend, curved.planning_adstock(), seen.lower, seen.upper, budget
    )
    weekly = np.asarray(record["weekly"])
    np.testing.assert_array_equal(
        weekly, water_fill(post.mean, curved.curve.slope, curved.shaped, lower, upper, budget)
    )
    assert weekly.sum() * seen.planned == pytest.approx(seen.budget, rel=1e-12, abs=0.0)
    rise = curved.curve.rise(weekly, quo)
    assert record["gain"] == pytest.approx(seen.planned * float(post.mean @ rise), rel=1e-12)
    flags, settings = record["flags"], record["settings"]
    for c, channel in enumerate(seen.channels):
        assert flags[f"{channel} half-saturation"] == 16.0 * quo[c]
        assert flags[f"{channel} margin"] == curved.margins.mean[c]
        assert (flags[f"{channel} planning floor"], flags[f"{channel} planning ceiling"]) == (
            lower[c],
            upper[c],
        )
    assert (flags["phi"], flags["ar share"], flags["promoted variance"]) == (
        curved.margins.phi,
        curved.margins.share,
        curved.margins.promoted,
    )
    assert flags["rows"] == plain["flags"]["rows"]
    assert (settings["planner"], settings["half_saturation"]) == ("water-filling", 16.0)
    assert settings["trend"] == {"degree": 3, "knot_every": 52}
    assert settings["nuisance_prior"] == "flat"
    assert settings["ar"] == [0.3, 0.6, 0.8, 0.9, 0.95, 0.98]
    assert settings["ar_share"] == [0.5, 0.7, 0.85, 0.95, 1.0]
    assert settings["promoted_variance"] == [1.0, 2.0, 4.0, 8.0]
    assert settings["white_errors"] is True
    assert record["n/a"]["lift tests"].endswith("every rung's plan is the plan of k = 0")
    json.dumps(record, allow_nan=False)
    assert plain["settings"]["trend"] == {"degree": 3, "knot_every": 26}
    assert plain["settings"]["ar"] == [0.0, 0.95]
    assert settings["spanned"] == 1e-8
    assert settings["held at the status quo"] == {}
    assert not any(flags[f"{channel} held"] for channel in seen.channels)
    mixture = {
        "nuisance_prior",
        "ar_share",
        "promoted_variance",
        "white_errors",
        "spanned",
        "held at the status quo",
    }
    assert not mixture & set(plain["settings"])
    assert not {"ar share", "promoted variance"} & set(plain["flags"])
    assert set(settings) - set(plain["settings"]) == mixture
    assert set(flags) - set(plain["flags"]) == {"ar share", "promoted variance"}
    assert plain["flags"][f"{seen.channels[0]} margin"] != flags[f"{seen.channels[0]} margin"]


REPRODUCED = (
    "a price that never moves",
    "a price that is a straight line in the week",
    "a promotion flag that is its promoted weeks' dummies' sum",
)


def _reproduced(seen, control):
    """The export with a control the mixture's other nuisance columns reproduce on every week."""
    week = np.arange(seen.sales.size)
    ahead = seen.sales.size + np.arange(seen.planned)
    controls, future = dict(seen.controls), dict(seen.future_controls)
    if control == REPRODUCED[0]:
        controls["price"], future["price"] = np.ones(week.size), np.ones(ahead.size)
    elif control == REPRODUCED[1]:
        controls["price"], future["price"] = 1.0 + 0.0005 * week, 1.0 + 0.0005 * ahead
    else:
        phases = promoted_phases(seen.controls["price"])
        assert phases
        controls["promotion"] = np.isin(week % 52, phases).astype(float)
        future["promotion"] = np.isin(ahead % 52, phases).astype(float)
    return dataclasses.replace(seen, controls=controls, future_controls=future)


@pytest.mark.parametrize("control", REPRODUCED)
def test_a_control_the_nuisance_columns_reproduce_fails_the_mixtures_variants_alone(control):
    """Under the mixture's flat nuisance prior such a control leaves the fit improper: the
    variants on the mixture are failed records that say which columns the weeks do not tell
    apart, the bench scores them at the status quo, and the variants on the ridge plan."""
    seed = TRACK_M2.pilots["drawn"][0]
    seen = _reproduced(TRACK_M2.observe(TRACK_M2.world("drawn", seed), 0), control)
    made = records(margin.Margin(), seen, smoke=False)
    mixture = ("margin curved noise", "margin held")
    for variant in mixture:
        record = made[variant]
        assert re.fullmatch(
            r"DegenerateFitError: the \d+ weeks read do not tell the design's \d+ nuisance"
            r" columns apart, and their flat prior leaves the fit improper",
            record["error"],
        )
        assert record["failure_kind"] == "plan"
        assert "weekly" not in record
    for variant in ("margin", "margin guarded", "margin ranged", "margin curved"):
        assert made[variant]["error"] is None
        assert np.all(np.isfinite(made[variant]["weekly"]))
    scored = score_world(
        TRACK_M2, "drawn", seed, 0, {v: made[v] for v in mixture}, pilot=True, archived=seen
    )
    for variant in mixture:
        assert scored.arms[variant].failure == made[variant]["error"]
        assert scored.arms[variant].regret == scored.arms["status quo"].regret


def _moved(seen, channel, move):
    """The export with `channel`'s spend moved by `move` of itself along a fixed pattern."""
    spend = np.array(seen.spend, dtype=float)
    pattern = np.random.default_rng(19).standard_normal(spend.shape[0])
    spend[:, channel] *= 1.0 + move * pattern / np.abs(pattern).max()
    return dataclasses.replace(seen, spend=spend)


def test_the_noise_variant_holds_a_spanned_channel_whatever_sign_rounding_leaves_its_margin():
    """Family 19's flat channel is a multiple of the trend's columns' sum: under the mixture's flat
    nuisance prior its margin is its prior's 0, and what floating point computed in its place was
    a residue 1e-12 of the other margins, of either sign. A positive residue entered the mean of
    the positive margins, the shrinkage's scale: on `flat-pla`'s pilot world 40 the two signs'
    plans lay 1.7 % of the weekly budget apart. Moving the flat spend by 1e-9 of itself along a
    pattern, and against it, gives the computed margin one sign each way. The variant reads the
    margin as exactly 0, holds the channel at its status quo and says why, and the three plans
    agree; `margin curved`, whose ridge sets that margin, holds nothing."""
    seen = FLAT.observe(FLAT.world("flat-pla", FLAT.pilots["flat-pla"][40]), 0)
    flat = seen.channels.index("pla")
    assert np.ptp(seen.spend[:, flat]) == 0.0
    plans = []
    for move in (0.0, 1e-9, -1e-9):
        made = records(margin.Margin(), _moved(seen, flat, move), smoke=False)
        record = made["margin curved noise"]
        assert record["error"] is None
        flags = record["flags"]
        assert [flags[f"{c} held"] for c in seen.channels] == [
            c == flat for c in range(len(seen.channels))
        ]
        assert flags["pla margin"] == 0.0
        assert set(record["settings"]["held at the status quo"]) == {"pla"}
        assert record["settings"]["held at the status quo"]["pla"].startswith(
            "the nuisance columns span its columns"
        )
        weekly = np.asarray(record["weekly"])
        assert weekly[flat] == seen.status_quo[flat]
        assert (flags["pla planning floor"], flags["pla planning ceiling"]) == (
            seen.status_quo[flat],
            seen.status_quo[flat],
        )
        assert weekly.sum() * seen.planned == pytest.approx(seen.budget, rel=1e-12, abs=0.0)
        assert not any(made["margin curved"]["flags"][f"{c} held"] for c in seen.channels)
        assert "held at the status quo" not in made["margin curved"]["settings"]
        plans.append(weekly)
    budget = float(np.sum(seen.status_quo))
    assert np.abs(plans[0] - seen.status_quo).max() > 0.01 * budget
    for moved in plans[1:]:
        np.testing.assert_allclose(moved, plans[0], rtol=0.0, atol=1e-8 * budget)


def test_a_channel_the_fit_names_spanned_is_held_at_its_status_quo_whatever_its_box(monkeypatch):
    """A flat channel's box is a point already. The hold does not lean on that: a channel the fit
    names spanned stays at its status quo, the others are planned at the budget inside their
    boxes, and its margin, in the shrinkage as the fit read it, moves no euro."""
    seen = observation(PARITY[0])
    fit = margin.fit_curved_margins

    def spanned(*args, noise):
        curved = fit(*args, noise=noise)
        named = np.array([True, False, False])
        return dataclasses.replace(
            curved, margins=dataclasses.replace(curved.margins, spanned=named)
        )

    free = records(margin.Margin(), seen, smoke=False)["margin curved noise"]
    monkeypatch.setattr(margin, "fit_curved_margins", spanned)
    record = records(margin.Margin(), seen, smoke=False)["margin curved noise"]
    assert record["error"] is None
    first, *others = seen.channels
    weekly = np.asarray(record["weekly"])
    assert free["weekly"][0] != seen.status_quo[0]
    assert weekly[0] == seen.status_quo[0]
    assert set(record["settings"]["held at the status quo"]) == {first}
    flags = record["flags"]
    assert flags[f"{first} held"] is True
    assert flags[f"{first} planning floor"] == flags[f"{first} planning ceiling"] == weekly[0]
    assert flags[f"{first} shrunk margin"] == free["flags"][f"{first} shrunk margin"]
    for c, channel in enumerate(others, start=1):
        assert flags[f"{channel} held"] is False
        assert (
            flags[f"{channel} planning floor"] <= weekly[c] <= flags[f"{channel} planning ceiling"]
        )
    assert weekly.sum() * seen.planned == pytest.approx(seen.budget, rel=1e-12, abs=0.0)
    assert np.abs(weekly[1:] - np.asarray(free["weekly"])[1:]).max() > 0.0


GUARD = {"time_held", "free_least"}
"""The settings a `margin held` record adds to a `margin curved noise` record's."""


def _following(seen, correlations):
    """The export with each channel `c` of `correlations` given a weekly spend of its own mean and
    spread whose correlation with the week's number is `correlations[c]`: the centred weeks and
    the channel's own spend made orthogonal to them, each of length 1, mixed."""
    spend = np.array(seen.spend, dtype=float)
    week = np.arange(spend.shape[0], dtype=float)
    along = week - week.mean()
    along /= np.linalg.norm(along)
    for c, r in correlations.items():
        across = spend[:, c] - spend[:, c].mean()
        scale = np.linalg.norm(across)
        across -= (across @ along) * along
        across /= np.linalg.norm(across)
        spend[:, c] = spend[:, c].mean() + scale * (r * along + math.sqrt(1.0 - r * r) * across)
    assert np.all(spend > 0.0)
    return dataclasses.replace(seen, spend=spend)


def _guarded_flags(seen):
    return {f"{c} time correlation" for c in seen.channels} | {"free channels"}


@pytest.mark.parametrize("environment", ["curve", "line"])
def test_on_the_twins_the_held_variant_plays_the_status_quo_and_names_the_ramp(environment):
    """Family 17: television's spend rises over three quarters of the history, and a response
    that curves and one that drifts give that history alike. `margin curved noise` moves
    television on its reading of it; `margin held` holds television, which leaves search alone
    free, and plays the status quo, claims nothing and says why."""
    seen = TWINS.observe(TWINS.world(environment, TWINS.pilots[environment][30]), 0)
    made = records(margin.Margin(), seen, smoke=False)
    record, free = made["margin held"], made["margin curved noise"]
    assert record["error"] is None
    assert record["weekly"] == list(seen.status_quo)
    assert (record["gain"], record["gain_interval"]) == (0.0, [0.0, 0.0])
    assert np.abs(np.asarray(free["weekly"]) - seen.status_quo).max() > 1.0
    flags, settings = record["flags"], record["settings"]
    assert flags["tv time correlation"] > 0.95
    assert flags["search time correlation"] < 0.5
    assert (flags["tv held"], flags["search held"], flags["free channels"]) == (True, False, 1)
    (reason,) = settings["held at the status quo"].values()
    assert set(settings["held at the status quo"]) == {"tv"}
    assert reason.startswith(
        "its weekly spend's absolute correlation with the week's number is"
        f" {flags['tv time correlation']:.3f}, above 0.8"
    )
    assert (settings["time_held"], settings["free_least"]) == (0.8, 2)
    assert flags["tv shrunk margin"] == free["flags"]["tv shrunk margin"]
    assert free["settings"]["held at the status quo"] == {}
    assert not GUARD & set(free["settings"])
    assert not _guarded_flags(seen) & set(free["flags"])
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize("world", [PARITY[0], PARITY[4], LONG], ids=name)
def test_where_no_spend_follows_the_week_the_held_plan_is_the_noise_variants(world):
    seen = observation(world)
    made = records(margin.Margin(), seen, smoke=False)
    record, free = made["margin held"], made["margin curved noise"]
    assert record["error"] is None
    assert record["weekly"] == free["weekly"]
    assert (record["gain"], record["gain_interval"]) == (free["gain"], free["gain_interval"])
    assert record["settings"]["held at the status quo"] == {}
    assert set(record["settings"]) - set(free["settings"]) == GUARD
    assert {k: v for k, v in record["settings"].items() if k not in GUARD} == free["settings"]
    added = _guarded_flags(seen)
    assert set(record["flags"]) - set(free["flags"]) == added
    assert {k: v for k, v in record["flags"].items() if k not in added} == free["flags"]
    correlation, _ = time_held(seen.spend)
    for c, channel in enumerate(seen.channels):
        assert record["flags"][f"{channel} time correlation"] == correlation[c] < 0.8
    assert record["flags"]["free channels"] == len(seen.channels)


def test_one_of_three_channels_held_stays_at_its_status_quo_and_the_others_are_planned():
    """A spend at a correlation of 0.8001 with the week is held and one at 0.7999 is not. The
    held channel stays at its status quo, its margin in the shrinkage as `margin curved noise`
    reads it on the same export; the other two are water-filled at the budget inside their boxes,
    and the claim is the returned plan's."""
    below = _following(observation(PARITY[0]), {0: 0.7999})
    made = records(margin.Margin(), below, smoke=False)
    assert made["margin held"]["flags"][f"{below.channels[0]} time correlation"] == pytest.approx(
        0.7999, rel=0.0, abs=1e-9
    )
    assert made["margin held"]["settings"]["held at the status quo"] == {}
    assert made["margin held"]["weekly"] == made["margin curved noise"]["weekly"]
    seen = _following(observation(PARITY[0]), {0: 0.8001})
    first, *others = seen.channels
    made = records(margin.Margin(), seen, smoke=False)
    record, free = made["margin held"], made["margin curved noise"]
    assert record["error"] is None
    flags = record["flags"]
    assert flags[f"{first} time correlation"] == pytest.approx(0.8001, rel=0.0, abs=1e-9)
    assert set(record["settings"]["held at the status quo"]) == {first}
    assert [flags[f"{c} held"] for c in seen.channels] == [True, False, False]
    assert flags["free channels"] == 2
    weekly = np.asarray(record["weekly"])
    assert weekly[0] == seen.status_quo[0]
    assert free["weekly"][0] != seen.status_quo[0]
    assert flags[f"{first} planning floor"] == flags[f"{first} planning ceiling"] == weekly[0]
    for c, channel in enumerate(others, start=1):
        assert (
            flags[f"{channel} planning floor"] <= weekly[c] <= flags[f"{channel} planning ceiling"]
        )
        assert (
            flags[f"{channel} planning floor"],
            flags[f"{channel} planning ceiling"],
        ) == (
            free["flags"][f"{channel} planning floor"],
            free["flags"][f"{channel} planning ceiling"],
        )
    assert weekly.sum() * seen.planned == pytest.approx(seen.budget, rel=1e-12, abs=0.0)
    assert np.abs(weekly[1:] - seen.status_quo[1:]).max() > 0.0
    for channel in seen.channels:
        for key in ("margin", "shrunk margin", "shrunk margin sd"):
            assert flags[f"{channel} {key}"] == free["flags"][f"{channel} {key}"]
    assert flags["tau_rel"] == free["flags"]["tau_rel"]
    quo = np.asarray(seen.status_quo, dtype=float)
    curved = fit_curved_margins(
        seen.sales,
        seen.spend,
        seen.controls["promotion"],
        seen.controls["price"],
        seen.kernel_length,
        quo,
        seen.planned,
        noise=margin.NOISE_MIXTURE,
    )
    post = shrink_margins(curved.margins.mean, curved.margins.cov)
    assert post is not None
    rise = curved.curve.rise(weekly, quo)
    assert rise[0] == 0.0
    assert record["gain"] == pytest.approx(seen.planned * float(post.mean @ rise), rel=1e-12)
    assert record["gain"] != free["gain"]


def test_with_one_channel_left_free_the_held_plan_is_the_status_quo():
    seen = _following(observation(PARITY[0]), {0: 0.9, 1: -0.9})
    made = records(margin.Margin(), seen, smoke=False)
    record = made["margin held"]
    assert record["error"] is None
    assert record["weekly"] == list(seen.status_quo)
    assert (record["gain"], record["gain_interval"]) == (0.0, [0.0, 0.0])
    assert set(record["settings"]["held at the status quo"]) == set(seen.channels[:2])
    assert record["flags"]["free channels"] == 1
    assert made["margin curved noise"]["weekly"] != list(seen.status_quo)


def test_a_free_box_that_holds_no_plan_at_the_budget_is_the_status_quo(monkeypatch):
    """The status quo need not lie in the box the history cuts: with a channel held there, the
    other channels' box may hold no plan at the budget. The plan is then the status quo, not a
    failure; without a channel held by the guard the box is the noise variant's own matter."""
    seen = _following(observation(PARITY[0]), {0: 0.9})
    quo = np.asarray(seen.status_quo, dtype=float)

    def above(*_args):
        return 1.05 * quo, 1.5 * quo

    monkeypatch.setattr(margin, "adstock_range", above)
    made = records(margin.Margin(), seen, smoke=False)
    record = made["margin held"]
    assert record["error"] is None
    assert record["weekly"] == list(seen.status_quo)
    assert (record["gain"], record["gain_interval"]) == (0.0, [0.0, 0.0])
    assert record["flags"]["free channels"] == 2
    assert made["margin curved noise"]["error"].startswith("ValueError")


def test_a_channel_whose_box_is_a_point_is_not_free(monkeypatch):
    """The free channels are counted by the holds and by the box, never by a margin: with one
    channel held by the guard and another's box a point, one channel is free, and the plan is the
    status quo."""
    seen = _following(observation(PARITY[0]), {0: 0.9})
    quo = np.asarray(seen.status_quo, dtype=float)

    def pointed(*_args):
        return quo * [0.5, 1.0, 0.5], quo * [2.0, 1.0, 2.0]

    monkeypatch.setattr(margin, "adstock_range", pointed)
    record = records(margin.Margin(), seen, smoke=False)["margin held"]
    assert record["error"] is None
    assert record["flags"]["free channels"] == 1
    assert [record["flags"][f"{c} held"] for c in seen.channels] == [True, False, False]
    assert record["weekly"] == list(seen.status_quo)
    assert (record["gain"], record["gain_interval"]) == (0.0, [0.0, 0.0])


def test_a_flat_channel_is_held_as_the_noise_variant_holds_it_not_by_the_guard(monkeypatch):
    """Family 19's flat channel has no correlation with the week: the guard reads 0 and the
    channel is held as spanned, the plan `margin curved noise`'s. A channel the fit names spanned
    whose spend follows the week is held as spanned too, and the guard counts no channel."""
    seen = FLAT.observe(FLAT.world("flat-pla", FLAT.pilots["flat-pla"][40]), 0)
    made = records(margin.Margin(), seen, smoke=False)
    record, free = made["margin held"], made["margin curved noise"]
    assert record["flags"]["pla time correlation"] == 0.0
    assert record["weekly"] == free["weekly"]
    assert (
        record["settings"]["held at the status quo"] == free["settings"]["held at the status quo"]
    )
    assert set(record["settings"]["held at the status quo"]) == {"pla"}
    assert record["flags"]["free channels"] == 2
    ramp = _following(observation(PARITY[0]), {0: 0.9})
    fit = margin.fit_curved_margins

    def spanned(*args, noise):
        curved = fit(*args, noise=noise)
        named = np.array([True, False, False])
        return dataclasses.replace(
            curved, margins=dataclasses.replace(curved.margins, spanned=named)
        )

    monkeypatch.setattr(margin, "fit_curved_margins", spanned)
    made = records(margin.Margin(), ramp, smoke=False)
    record, free = made["margin held"], made["margin curved noise"]
    (reason,) = record["settings"]["held at the status quo"].values()
    assert reason.startswith("the nuisance columns span its columns")
    assert record["weekly"] == free["weekly"]
    assert record["weekly"] != list(ramp.status_quo)
