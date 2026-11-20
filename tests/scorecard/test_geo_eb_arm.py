"""The closed-form model over geos as an arm on family 8: one fit of the export's panel with its
geo tests, one plan of every cell at the budget in the box, no gain claimed, and a world run end to
end under the harness and scored. The arm reads the export alone."""

import dataclasses
import json
import math

import numpy as np
import pytest

pytest.importorskip("causaldyn_bench")

from causaldyn_bench.scorecard.geo import GEO
from causaldyn_bench.scorecard.observe import UNIVERSES, SwitchbackTest, Universes
from causaldyn_bench.scorecard.track_m2 import TRACK_M2

from mixdyn import Channel
from mixdyn.decisions.geo_plan import plan_mixture
from mixdyn.models.geo_eb import POINTS, fit_geo_eb
from scorecard import report, run
from scorecard.arms import geo_eb
from scorecard.arms import panel as reading
from scorecard.arms.base import records, withheld
from scorecard.layout import Declaration, Layout, Limits

ENVIRONMENT = "policy-0"
LIMITS = Limits(memory="4G", timeout=600.0, slots=1)


@pytest.fixture(scope="module")
def observed():
    return GEO.observe(GEO.world(ENVIRONMENT, GEO.pilots[ENVIRONMENT][0]), 4)


@pytest.fixture(scope="module")
def fitted(observed):
    return geo_eb.GeoEb().fit(observed, smoke=False)


def test_the_arm_is_registered_beside_the_bayes_geo_arms_and_reads_the_tests():
    arm = run.ARMS["geo-eb"]()
    assert isinstance(arm, geo_eb.GeoEb)
    assert arm.variants == ("geo eb",)
    assert arm.tests
    assert arm.cells


def test_the_fit_is_the_librarys_on_the_exports_panel_and_tests(observed, fitted):
    data = reading.panel(observed)
    tests = reading.estimates(observed)
    fit = fit_geo_eb(data, tests, observed.kernel_length)
    assert fitted.fit.channels == tuple(observed.channels)
    assert fitted.fit.selected == fit.selected
    assert fitted.flags["lift rows"] == len(observed.lift.channel) == 12
    assert fitted.settings["lift_rows"] == 12
    for c, j in zip(observed.channels, fit.selected, strict=True):
        assert fitted.settings["selected"][c]["retention"] == POINTS[j][0]
        assert fitted.settings["selected"][c]["curvature"] == str(POINTS[j][1])
    cells = reading.cells(observed)
    assert len(fitted.cells) == len(cells) == observed.status_quo.size
    for held, (g, c) in zip(fitted.cells, cells, strict=True):
        made = fit.cell(None if g is None else data.geos[g], Channel(c))
        np.testing.assert_array_equal(held.coefficients, made.coefficients)
        np.testing.assert_array_equal(held.history, made.history)
    spent = reading.cell_spend(observed)
    for j, held in enumerate(fitted.cells):
        np.testing.assert_allclose(held.history, spent[:, j], rtol=1e-9, atol=0.0)
    json.dumps(fitted.settings, allow_nan=False)


def test_the_untested_rung_reads_no_lift_row(observed):
    fit = geo_eb.GeoEb().fit(withheld(observed), smoke=False)
    assert fit.flags["lift rows"] == 0


def test_the_plan_spends_the_budget_in_the_box_and_claims_no_gain(observed, fitted):
    planned = geo_eb.GeoEb().plan(fitted, geo_eb.VARIANT, observed)
    weekly = planned.weekly
    rate = observed.budget / observed.planned
    assert weekly.sum() == pytest.approx(rate, rel=1e-12, abs=0.0)
    assert np.all(weekly >= observed.lower)
    assert np.all(weekly <= observed.upper)
    direct = plan_mixture(
        fitted.cells, observed.lower, observed.upper, observed.budget, observed.planned
    )
    np.testing.assert_array_equal(weekly, direct.weekly)
    assert planned.gain is None
    assert planned.gain_interval is None
    assert planned.forecast is None
    assert set(planned.absent) == {"forecast", "gain"}
    assert planned.response == {"unmapped": geo_eb.UNMAPPED}
    assert planned.settings["planner"] == "plan_mixture"


def test_the_record_holds_no_gain_and_names_it_absent(observed):
    """The plan's worth at the posterior means over-states the gain the plan realises, so the
    record claims none and no interval of one, and says why under `n/a`."""
    record = records(geo_eb.GeoEb(), observed, smoke=False)[geo_eb.VARIANT]
    assert record["error"] is None, record.get("traceback")
    assert "gain" not in record
    assert "gain_interval" not in record
    assert set(record["n/a"]) == {"forecast", "gain"}
    assert "over-states" in record["n/a"]["gain"]
    assert "no interval" in record["n/a"]["gain"]


def test_the_flags_count_the_kept_points_and_the_margins_below_zero(observed, fitted):
    """The fewest and the most grid points a channel's posterior kept, and the cells whose
    expected margin is below 0: none on this world, and every cell on its mirror, the KPI negated
    with no test read, where each coefficient's posterior mean changes its sign."""
    kept = [int(m.points.size) for m in fitted.fit.mixtures]
    assert min(kept) < max(kept)
    assert fitted.flags["fewest kept points"] == min(kept)
    assert fitted.flags["most kept points"] == max(kept)
    assert all(float(cell.weights @ cell.coefficients) > 0.0 for cell in fitted.cells)
    assert fitted.flags["negative margins"] == 0
    mirror = dataclasses.replace(withheld(observed), sales=-observed.sales)
    mirrored = geo_eb.GeoEb().fit(mirror, smoke=False)
    assert all(float(cell.weights @ cell.coefficients) < 0.0 for cell in mirrored.cells)
    assert mirrored.flags["negative margins"] == len(mirrored.cells) == observed.status_quo.size


def test_the_settings_are_the_models_constants_and_each_channels_kept_points(observed, fitted):
    settings = geo_eb.GeoEb().plan(fitted, geo_eb.VARIANT, observed).settings
    assert settings["retentions"] == [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
    assert " ".join(settings["curvatures"]) == "0.25 0.5 1.0 2.0 4.0 8.0 16.0 32.0 inf"
    assert settings["spread"] == 0.4
    assert settings["prior_width"] == 10.0
    assert settings["season_harmonics"] == 2
    assert settings["sweeps"] == 3
    assert settings["kept"] == 1e-4
    assert settings["grid_points"] == 401
    assert settings["bisections"] == 200
    for c, mixture in zip(observed.channels, fitted.fit.mixtures, strict=True):
        assert settings["selected"][c]["kept_points"] == mixture.points.size


def test_the_arm_reads_the_export_alone(observed):
    """What the arm is not meant to read cannot move the plan: the record's plan, flags and
    settings are the same from the export and from one with the world's labels, its rung's
    number, its returns' window, its rows' weekly gaps and dropped count each replaced, and from
    a version-4 export that holds a universe for each tested channel and a switchback test."""
    first = records(geo_eb.GeoEb(), observed, smoke=False)[geo_eb.VARIANT]
    assert first["error"] is None, first.get("traceback")
    assert first["plan_level"] == "cells"
    assert first["digest"] == observed.digest()
    assert observed.version != UNIVERSES
    assert observed.universes is None
    assert observed.switchback is None
    tested = tuple(c for c in observed.channels if c in set(observed.lift.channel))
    universes = Universes(
        channel=tested,
        sales=np.repeat(0.5 * observed.sales[None], len(tested), axis=0),
        spend=np.repeat(2.0 * observed.spend[None], len(tested), axis=0),
    )
    switchback = SwitchbackTest(
        channel=observed.national[0],
        first=1,
        block=2,
        on=(True, False),
        sales=3.0 * observed.sales,
        spend=0.0 * observed.spend,
    )
    lift = dataclasses.replace(observed.lift, gaps=1.0 + observed.lift.gaps, dropped=3)
    changes = {
        "seed": {"seed": observed.seed + 1},
        "family": {"family": 99},
        "environment": {"environment": "another"},
        "k": {"k": 1},
        "roi_window": {"roi_window": (1, 2)},
        "gaps and dropped": {"lift": lift},
        "universes": {"version": UNIVERSES, "universes": universes},
        "switchback": {"version": UNIVERSES, "universes": universes, "switchback": switchback},
    }
    for name, change in changes.items():
        other = dataclasses.replace(observed, **change)
        assert other.digest() != first["digest"], name
        second = records(geo_eb.GeoEb(), other, smoke=False)[geo_eb.VARIANT]
        assert second["error"] is None, (name, second.get("traceback"))
        assert second["weekly"] == first["weekly"], name
        assert second["flags"] == first["flags"], name
        assert second["settings"] == first["settings"], name


def test_the_arm_refuses_a_national_export():
    national = TRACK_M2.observe(TRACK_M2.world("drawn", TRACK_M2.pilots["drawn"][0]), 0)
    with pytest.raises(ValueError, match="no geo axis"):
        geo_eb.GeoEb().fit(national, smoke=True)


def test_the_arm_runs_a_world_end_to_end_and_is_scored_and_reported(
    tmp_path, observed, monkeypatch
):
    layout = Layout(tmp_path / "run")
    run.declare(
        layout,
        Declaration(
            family=8,
            environments=(ENVIRONMENT,),
            worlds=1,
            pilot=True,
            rungs=(0, 4),
            arms=("geo-eb",),
            smoke=True,
            limits={"geo-eb": LIMITS},
        ),
    )
    run.export_all(layout)
    for k in (0, 4):
        run.one(layout, "geo-eb", ENVIRONMENT, observed.seed, (k,))
    run.score(layout)
    shown = []
    monkeypatch.setattr("builtins.print", lambda *a, **_: shown.append(" ".join(map(str, a))))
    report.main(layout, [geo_eb.VARIANT])
    rows = {}
    for k in (0, 4):
        record = json.loads(
            layout.record(geo_eb.VARIANT, 8, ENVIRONMENT, k, observed.seed).read_text()
        )
        assert record["error"] is None, record.get("traceback")
        assert record["plan_level"] == "cells"
        rows[k] = record["flags"]["lift rows"]
    assert rows == {0: 0, 4: 12}
    lines = [json.loads(x) for x in layout.scores.read_text().splitlines()]
    assert sorted(line["k"] for line in lines) == [0, 4]
    for line in lines:
        arm = line["arms"][geo_eb.VARIANT]
        assert math.isfinite(arm["regret"])
        assert arm["failure"] is None
        assert arm["regret"] < line["arms"]["status quo"]["regret"]
        assert arm["uplift"] is None
        assert arm["covered"] is None
    printed = [
        x for x in "\n".join(shown).splitlines() if x.startswith(f"  {geo_eb.VARIANT:16s} regret")
    ]
    assert printed
    assert not any("claimed less realised" in x for x in printed)
