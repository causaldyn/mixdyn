"""Family 8's geo tests read again by the engine's readers, and a switchback test of its national
channel, each as an arm: each test is read in the universe the bench's version-4 export holds,
the ladder's own reading of it there the row's to the last bit; each reader's reading of a test
takes the row's place in the market the static fit reads, a test the reader refuses keeps the
bench's, and families 9 and 10 are read as family 8; the switchback is read in the universe of
the bench's switchback test, TV off in every geo in the blocks its design turns off, and its
estimate is read beside the rung's tests. Each arm runs a world end to end under the harness and
is scored.

The static arm's two fits are a stand-in here (`Fake`), so that a world runs in seconds; the
readings and the switchback are the engine's own."""

import dataclasses
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("causaldyn_bench")

from causaldyn_bench.lift_calibration import COOLDOWN, TEST
from causaldyn_bench.scorecard.few_geos import FEW_GEOS
from causaldyn_bench.scorecard.geo import GEO
from causaldyn_bench.scorecard.ladder import _geo_reading, _slope
from causaldyn_bench.scorecard.observe import UNIVERSES, VERSION
from causaldyn_bench.scorecard.scoring import ArmScore
from causaldyn_bench.scorecard.spillover import SPILLOVER
from fake_posterior import Fake

from mixdyn import Channel
from mixdyn.experiments import (
    GeoTest,
    SwitchbackDesign,
    read_sdid,
    read_switchback,
    read_synthetic_control,
)
from scorecard import report, run
from scorecard.arms import bayes, readers, switchback
from scorecard.arms import panel as reading
from scorecard.layout import Declaration, Layout, Limits

ENVIRONMENT = "policy-0"
LIMITS = Limits(memory="4G", timeout=600.0, slots=1)
ARMS = {"read-tbr": readers.TBR, "read-sc": readers.SC, "read-sdid": readers.SDID}


@pytest.fixture(scope="module")
def world():
    return GEO.world(ENVIRONMENT, GEO.pilots[ENVIRONMENT][0])


@pytest.fixture(scope="module")
def observed(world):
    return GEO.observe(world, 4, version=UNIVERSES)


def _universe(observation, name):
    """`name`'s universe in `observation`'s export: its KPI, `(geos, weeks)`, and its spend,
    `(geos, weeks, channels)`."""
    held = observation.universes
    i = held.channel.index(name)
    return held.sales[i].T, held.spend[i].transpose(1, 0, 2)


class Kept:
    """The static fit's stand-in: each call's market and estimates kept, the fake's fit made."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, dict]] = []

    def fits(self, observation, sampler, estimates=None):
        held = reading.estimates(observation) if estimates is None else estimates
        self.calls.append((observation, held))
        return {"hill": None, "tanh": None}

    def fitted(self, observation, fits, sampler):
        made = Fake()(observation, sampler).fitted
        return dataclasses.replace(made, settings={"readings": 40, "lift_rows": 12})


@pytest.fixture
def kept(monkeypatch):
    stand_in = Kept()
    monkeypatch.setattr(bayes, "family_fits", stand_in.fits)
    monkeypatch.setattr(bayes, "fitted", stand_in.fitted)
    return stand_in


def test_the_exported_universes_read_the_exports_rows_to_the_last_bit(world, observed):
    """The ladder's own reading, its slope and its gaps, of each test on the universe the export
    holds, read on the axes the arm reads it on, is the row's delta_y exactly; the universe's spend
    is the history's with the channel dark in each test's geos and weeks."""
    lift = observed.lift
    assert observed.universes.channel == tuple(c for c in observed.channels if c in lift.channel)
    for i, name in enumerate(lift.channel):
        kpi, spend = _universe(observed, name)
        c = observed.channels.index(name)
        treated = list(lift.geos[i])
        others = [g for g in range(kpi.shape[0]) if g not in treated]
        first = lift.start[i] - 1
        ours, theirs = kpi[treated].sum(axis=0), kpi[others].sum(axis=0)
        slope = _slope(ours, theirs, first, world.history.mix.length)
        assert _geo_reading(ours, theirs, first, slope) == lift.delta_y[i]
        expected = world.history.spend.copy()
        for r, other in enumerate(lift.channel):
            if other == name:
                start = lift.start[r] - 1
                expected[list(lift.geos[r]), start : start + TEST, c] = 0.0
        np.testing.assert_array_equal(spend, expected)
    assert GEO.observe(world, 0, version=UNIVERSES).universes.channel == ()


def test_a_reading_needs_the_universes_of_a_version_4_export(world):
    with pytest.raises(ValueError, match="version 4 alone, not 2"):
        readers.reread(GEO.observe(world, 4), readers.SC)


@pytest.mark.parametrize(
    ("family", "environment"),
    [(SPILLOVER, "spillover-0.2"), (FEW_GEOS, "geos-3"), (FEW_GEOS, "geos-8")],
)
def test_families_9_and_10_are_read_in_their_exported_universes(family, environment):
    """Each test of a pilot world of families 9 and 10 read in the universe its export holds:
    by the synthetic control where 3 geos are its controls, by SDID's jackknife where 2 are
    treated and 2 controls, and the bench's reading kept elsewhere and counted, as in every test
    of 3 geos; the export read as version 2."""
    world = family.world(environment, family.pilots[environment][0])
    observation = family.observe(world, 4, version=UNIVERSES)
    geos = observation.population.size
    for variant, reads in (
        (readers.SC, lambda treated: geos - treated >= 3),
        (readers.SDID, lambda treated: min(treated, geos - treated) >= 2),
    ):
        moved, rows = readers.reread(observation, variant)
        assert moved.version == VERSION
        read = [reads(len(group)) for group in observation.lift.geos]
        assert [row["reader"] is not None for row in rows] == read, variant
        assert any(read) == (environment != "geos-3")
        for i, row in enumerate(rows):
            if row["reader"] is None:
                bench = (observation.lift.delta_y[i], observation.lift.sigma[i])
                assert (moved.lift.delta_y[i], moved.lift.sigma[i]) == bench
            else:
                assert moved.lift.sigma[i] > 0.0


@pytest.mark.parametrize("variant", [readers.TBR, readers.SC, readers.SDID])
def test_each_row_is_its_readers_reading_per_dark_week(observed, variant):
    """Every column of a row but its reading and its gaps is the bench's; the reading is the
    reader's lift and error over the readout over the dark weeks; the time-based regression keeps
    the bench's reading of the tests from week 20, whose placebo windows do not fit before it."""
    moved, rows = readers.reread(observed, variant)
    lift, read = observed.lift, moved.lift
    assert read.gaps is None
    for name in ("channel", "start", "geos"):
        assert getattr(read, name) == getattr(lift, name)
    np.testing.assert_array_equal(read.x, lift.x)
    np.testing.assert_array_equal(read.delta_x, lift.delta_x)
    for i, row in enumerate(rows):
        assert row["bench"] == [lift.delta_y[i], lift.sigma[i]]
        if variant == readers.TBR and lift.start[i] == 20:
            assert row["reader"] is None
            assert (read.delta_y[i], read.sigma[i]) == (lift.delta_y[i], lift.sigma[i])
        else:
            assert row["reader"] == [read.delta_y[i], read.sigma[i]]
            assert read.sigma[i] > 0.0
    kept = sum(row["reader"] is None for row in rows)
    assert kept == (3 if variant == readers.TBR else 0)


def test_a_reading_is_the_reader_called_on_the_tests_own_universe(observed):
    """One row each of the synthetic control and SDID, read again here on the universe's KPI by
    the reader called directly: the test's geos treated, the rest controls, its dark weeks its
    change and 16 weeks of readout."""
    i = 5
    name, first = observed.lift.channel[i], observed.lift.start[i] - 1
    data = reading.panel(observed)
    universe = dataclasses.replace(data, kpi=_universe(observed, name)[0])
    named = reading.geos(observed)
    treated = tuple(named[g] for g in observed.lift.geos[i])
    control = tuple(g for g in named if g not in treated)
    (estimate,) = [
        e
        for e, s in zip(reading.estimates(observed)[name], [20, 44, 68, 92], strict=True)
        if s == first + 1
    ]
    dates = universe.dates
    design = GeoTest(
        Channel(name),
        treated,
        control,
        dates[first],
        dates[first + TEST - 1],
        estimate.spend,
        estimate.planned,
        TEST + COOLDOWN,
    )
    sc, _ = read_synthetic_control(universe, design)
    sdid, _ = read_sdid(
        universe,
        treated,
        dates[first],
        dates[first + TEST + COOLDOWN - 1],
        spend=estimate.spend,
        planned=estimate.planned,
        variance="jackknife",
    )
    for variant, expected in ((readers.SC, sc), (readers.SDID, sdid)):
        moved, _ = readers.reread(observed, variant)
        assert moved.lift.delta_y[i] == expected.lift / TEST
        assert moved.lift.sigma[i] == expected.se / TEST


def test_a_reading_arm_fits_the_market_on_its_readings(observed, kept):
    """The static fit reads the market whose tests are the reader's, each in its geos' share;
    the record says how many rows the reader read and kept, beside the static arm's own."""
    market = reading.market(observed)
    made = readers.Reader(readers.SDID)
    fit = made.fit(market, smoke=True)
    ((_, estimates),) = kept.calls
    moved, _ = readers.reread(observed, readers.SDID)
    expected = reading.estimates(reading.market(moved))
    for name, rows in expected.items():
        assert [r.lift for r in estimates[name]] == [r.lift for r in rows]
        assert [r.share for r in estimates[name]] == [r.share for r in rows]
    assert fit.flags == {"rows read": 12, "rows kept": 0}
    assert fit.settings["reader"] == readers.SDID
    planned = made.plan(fit, readers.SDID, market)
    assert planned.flags["rows read"] == 12
    static = bayes.Bayes().plan(fit.fitted, bayes.RESPONDS, market)
    for key, value in static.settings.items():
        assert planned.settings[key] == value, key
    assert len(planned.settings["reader rows"]) == 12
    with pytest.raises(ValueError, match="reads a geo export's tests"):
        made.fit(observed, smoke=True)
    with pytest.raises(ValueError, match="the readers are"):
        readers.Reader("read did")


def test_the_switchback_is_read_beside_the_rungs_tests(observed, kept):
    """TV's tests at the rung, each in its geos' share, and one more: the switchback the export
    holds, over the last 12 blocks of 8 weeks, 6 on in the bench's order, its first 4 weeks after
    a switch left out, read by randomization on its universe's KPI summed over the geos, of share
    1."""
    test = observed.switchback
    assert (test.channel, test.first, test.block, len(test.on)) == ("tv", 9, 8, 12)
    market = reading.market(observed)
    fit = switchback.Switchback().fit(market, smoke=True)
    ((_, estimates),) = kept.calls
    rung = reading.estimates(market)
    for name in ("search", "social"):
        assert [r.lift for r in estimates[name]] == [r.lift for r in rung[name]]
    *tv, added = estimates["tv"]
    assert [r.lift for r in tv] == [r.lift for r in rung["tv"]]
    held = fit.settings["switchback"]
    assert (held["block"], held["washout"], held["weeks"]) == (8, 4, [9, 104])
    assert held["on"] == list(test.on)
    assert sum(held["on"]) == 6
    kpi, spend = test.sales.T, test.spend.transpose(1, 0, 2)
    design = SwitchbackDesign(
        reading.panel(market).dates[8:], 8, 4, tuple(held["on"]), 0.5, held["seed"]
    )
    estimate, read = read_switchback(kpi[:, 8:], spend[:, 8:, 2], design)
    assert added.share == 1.0
    assert added.readout == 96
    assert added.lift == estimate.lift
    assert added.se == estimate.se
    assert fit.flags["switchback lift"] == read.lift
    assert (held["lower"], held["upper"]) == (read.lower, read.upper)
    assert held["arrangements"] == math.comb(12, 6)
    with pytest.raises(ValueError, match="national channel"):
        switchback.Switchback().fit(observed, smoke=True)
    with pytest.raises(ValueError, match="version 4 alone, not 2"):
        switchback.Switchback().fit(reading.market(reading.second(observed)), smoke=True)


@pytest.fixture(scope="module")
def truth(world):
    return GEO.truth(world)


def _scored(observed, lift, *, on=None, lower=-1.0, upper=1.0, failure=None):
    """A stand-in of the bench's score of the arm's plan, its record holding a reading of `lift`
    with the interval `[lower, upper]`, the export's design unless `on` names another."""
    held = {
        "block": 8,
        "on": list(observed.switchback.on) if on is None else on,
        "lower": lower,
        "upper": upper,
    }
    record = {"flags": {"switchback lift": lift}, "settings": {"switchback": held}}
    return ArmScore(0.1, failure, None, 0, None, None, None, None, record)


def test_the_switchback_reading_is_scored_a_week_on_against_the_steady_truth(observed, truth):
    """The reading's lift over its 48 weeks on, against the bench's steady truth a week on, its
    interval a week on too, and the test's whole lift beside them; the plan's score is the
    bench's."""
    steady = truth.switchback_steady
    assert steady > 0.0
    lift = 48.0 * 1.25 * steady
    score = _scored(observed, lift, lower=48.0 * 0.5 * steady, upper=48.0 * 1.5 * steady)
    scored = switchback.scored(score, truth, observed)
    read = scored.record[switchback.SCORE]
    assert read["a week"] == lift / 48.0
    assert read["steady"] == steady
    assert read["error"] == pytest.approx(0.25, rel=1e-12, abs=0.0)
    assert read["held"] is True
    assert read["total"] == truth.switchback
    assert dataclasses.replace(scored, record=score.record) == score
    above = _scored(observed, lift, lower=48.0 * 1.1 * steady, upper=48.0 * 1.5 * steady)
    assert switchback.scored(above, truth, observed).record[switchback.SCORE]["held"] is False


def test_a_switchback_reading_of_another_test_is_not_scored(world, observed, truth):
    """A reading of a design other than the export's, or on an export without the test, says so;
    a failed fit's score is the bench's alone."""
    other = [not b for b in observed.switchback.on]
    for score, export in (
        (_scored(observed, 1.0, on=other), observed),
        (_scored(observed, 1.0), GEO.observe(world, 4)),
    ):
        read = switchback.scored(score, truth, export).record[switchback.SCORE]
        assert set(read) == {"unscored"}
    failed = _scored(observed, 1.0, failure="RuntimeError: the fit broke")
    assert switchback.scored(failed, truth, observed) is failed


@pytest.fixture(scope="module")
def ran(tmp_path_factory, world):
    """A smoke run of family 8's first pilot world at rung 4 by each reading arm and the
    switchback arm, the static fit the stand-in, scored and reported."""
    layout = Layout(tmp_path_factory.mktemp("readers") / "run")
    names = [*ARMS, "switchback"]
    run.declare(
        layout,
        Declaration(
            family=8,
            environments=(ENVIRONMENT,),
            worlds=1,
            pilot=True,
            rungs=(4,),
            arms=tuple(names),
            smoke=True,
            limits=dict.fromkeys(names, LIMITS),
        ),
    )
    run.export_all(layout)
    stand_in = Kept()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(bayes, "family_fits", stand_in.fits)
        patch.setattr(bayes, "fitted", stand_in.fitted)
        for name in names:
            run.one(layout, name, ENVIRONMENT, world.seed, (4,))
    run.score(layout)
    variants = [*ARMS.values(), switchback.VARIANT]
    with pytest.MonkeyPatch.context() as patch:
        shown = []
        patch.setattr("builtins.print", lambda *a, **_: shown.append(" ".join(map(str, a))))
        report.main(layout, variants)
    records = {
        v: json.loads(layout.record(v, 8, ENVIRONMENT, 4, world.seed).read_text()) for v in variants
    }
    scores = [json.loads(line) for line in layout.scores.read_text().splitlines()]
    return SimpleNamespace(records=records, scores=scores, shown="\n".join(shown))


def test_each_arm_runs_a_world_end_to_end_and_is_scored_and_reported(ran):
    for variant, record in ran.records.items():
        assert record["error"] is None, (variant, record.get("traceback"))
        assert record["plan_level"] == "market"
        assert f"  {variant:16s} regret" in ran.shown
    (line,) = ran.scores
    for variant in ran.records:
        assert line["arms"][variant]["failure"] is None
        assert math.isfinite(line["arms"][variant]["regret"])
    assert ran.records[readers.TBR]["flags"]["rows kept"] == 3


def test_the_switchback_reading_is_scored_on_the_truth_beside_the_plan(ran, truth):
    (line,) = ran.scores
    record = ran.records[switchback.VARIANT]
    read = line["arms"][switchback.VARIANT]["record"][switchback.SCORE]
    assert read["a week"] == record["flags"]["switchback lift"] / 48.0
    assert (read["steady"], read["total"]) == (truth.switchback_steady, truth.switchback)
    assert "switchback a week on less its truth" in ran.shown
    for variant in ARMS.values():
        assert switchback.SCORE not in line["arms"][variant]["record"]


def test_the_static_fits_read_the_estimates_given_in_place_of_the_rows(observed, monkeypatch):
    """`family_fits` hands each family's fit the estimates it is given, and the export's rows
    where it is given none."""
    seen = []

    def fit(*args, estimates, **kwargs):
        seen.append(estimates)
        return "a fit"

    monkeypatch.setattr(bayes, "fit_bayes", fit)
    market = reading.market(observed)
    rows = reading.estimates(market)
    given = {"tv": rows["tv"][:1]}
    assert bayes.family_fits(market, bayes.SMOKE, estimates=given) == dict.fromkeys(
        bayes.FAMILIES, "a fit"
    )
    assert seen == [given, given]
    seen.clear()
    bayes.family_fits(market, bayes.SMOKE)
    assert [sorted(e) for e in seen] == [sorted(rows)] * 2
    assert [[r.lift for r in e["tv"]] for e in seen] == [[r.lift for r in rows["tv"]]] * 2
