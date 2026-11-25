"""Each rule the benchmark scored, planned through its public planner, is its arm's plan to the
last bit: `plan_margins` is `margin held`'s, `plan_anchored_margins` is `anchored floored`'s and,
with no floor, `anchored held`'s, and `plan_pooled_geos` is `geo eb`'s, on the arms' fixture worlds
and on family 8's pilot world at both rungs. The planners read the export as a caller holds it:
its panel, its boxes as `Range`s and its lift rows as the tests' `LiftEstimate`s
(`scorecard.arms.panel`), which the anchored planner reduces to the arm's lift rows."""

import numpy as np
import pytest

from gs_worlds import WORLDS as TESTED
from gs_worlds import name as gs_name
from margin_worlds import PARITY
from margin_worlds import name as margin_name

pytest.importorskip("causaldyn_bench")

from causaldyn_bench.scorecard.geo import GEO
from test_anchored_arm import observation as gs_export
from test_margin_arm import observation as margin_export

from mixdyn import Channel
from mixdyn.decisions import (
    GeoLimits,
    Range,
    plan_anchored_margins,
    plan_margins,
    plan_pooled_geos,
    scored,
)
from scorecard.arms import anchored, geo_eb, margin
from scorecard.arms import panel as reading
from scorecard.arms.base import withheld


def _ranges(seen):
    return {
        Channel(c): Range(float(lo), float(hi))
        for c, lo, hi in zip(seen.channels, seen.lower, seen.upper, strict=True)
    }


def _same(planned, made):
    """The arm's plan and claim, and the planner's, bit for bit."""
    assert made.spend.tobytes() == np.asarray(planned.weekly, dtype=float).tobytes()
    assert made.gain.hex() == float(planned.gain).hex()
    assert tuple(x.hex() for x in made.interval) == tuple(
        float(x).hex() for x in planned.gain_interval
    )


@pytest.mark.parametrize("world", PARITY, ids=margin_name)
def test_plan_margins_is_the_margin_held_arms_plan(world):
    seen = margin_export(world)
    arm = margin.Margin()
    planned = arm.plan(arm.fit(seen, smoke=False), margin.HELD, seen)
    made = plan_margins(
        reading.panel(seen), None, seen.planned, _ranges(seen), kernel=seen.kernel_length
    )
    _same(planned, made)
    assert made.current.tobytes() == seen.status_quo.tobytes()
    assert made.budget == seen.budget
    assert made.record == margin.HELD


@pytest.mark.parametrize("world", TESTED, ids=gs_name)
def test_the_tests_as_estimates_reduce_to_the_exports_lift_rows(world):
    seen = gs_export(world)
    rows = scored._rows(reading.panel(seen), reading.estimates(seen))
    lift = seen.lift
    assert rows.channel == tuple(lift.channel)
    for got, want in ((rows.x, lift.x), (rows.delta_y, lift.delta_y), (rows.sigma, lift.sigma)):
        assert got.tobytes() == np.asarray(want, dtype=float).tobytes()
    assert rows.start.tolist() == list(lift.start)
    assert (rows.dark, rows.readout) == (anchored.TEST, anchored.READOUT)


@pytest.mark.parametrize(
    ("variant", "spend_floor"), [(anchored.FLOORED, 0.9), (anchored.HELD, None)]
)
@pytest.mark.parametrize("world", TESTED, ids=gs_name)
def test_plan_anchored_margins_is_the_anchored_arms_plan(world, variant, spend_floor):
    seen = gs_export(world)
    arm = anchored.Anchored()
    planned = arm.plan(arm.fit(seen, smoke=False), variant, seen)
    made = plan_anchored_margins(
        reading.panel(seen),
        reading.estimates(seen),
        None,
        seen.planned,
        _ranges(seen),
        kernel=seen.kernel_length,
        spend_floor=spend_floor,
    )
    _same(planned, made)
    assert made.current.tobytes() == seen.status_quo.tobytes()
    assert made.record == variant


@pytest.mark.parametrize("tested", [False, True], ids=["untested", "tested"])
def test_plan_pooled_geos_is_the_geo_eb_arms_plan(tested):
    environment = "policy-0"
    seen = GEO.observe(GEO.world(environment, GEO.pilots[environment][0]), 4)
    if not tested:
        seen = withheld(seen)
    arm = geo_eb.GeoEb()
    planned = arm.plan(arm.fit(seen, smoke=False), geo_eb.VARIANT, seen)
    data = reading.panel(seen)
    boxes = [
        (g, Channel(c), Range(float(lo), float(hi)))
        for (g, c), lo, hi in zip(reading.cells(seen), seen.lower, seen.upper, strict=True)
    ]
    limits = GeoLimits(
        cells={(data.geos[g], c): r for g, c, r in boxes if g is not None},
        national={c: r for g, c, r in boxes if g is None},
    )
    made = plan_pooled_geos(
        data, reading.estimates(seen), seen.budget, seen.planned, limits, kernel=seen.kernel_length
    )
    assert made.spend.tobytes() == np.asarray(planned.weekly, dtype=float).tobytes()
    assert [(None if g is None else data.geos.index(g), str(c)) for g, c in made.cells] == list(
        reading.cells(seen)
    )
    assert made.record == geo_eb.VARIANT
