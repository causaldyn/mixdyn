"""Family 8's geo worlds read by this engine's arms: a geo export read as a geo panel and its tests
as estimates in their geos; the market a national arm plans, one weekly spend a channel, its tests
in their geos' share; and the model over geos as an arm that plans every cell."""

import dataclasses
import json
import math

import numpy as np
import pytest

pytest.importorskip("causaldyn_bench")

from causaldyn_bench.lift_calibration import COOLDOWN, TEST
from causaldyn_bench.scorecard.few_geos import FEW_GEOS
from causaldyn_bench.scorecard.geo import GEO
from causaldyn_bench.scorecard.observe import UNIVERSES, VERSION, read
from causaldyn_bench.scorecard.scoring import score_world
from causaldyn_bench.scorecard.track_m2 import TRACK_M2
from chc.allocation import cvar_allocate

from mixdyn import NATIONAL, Allotment, Channel, aggregate
from mixdyn.decisions import GeoLimits, Range, plan_geos
from scorecard import report, run
from scorecard.arms import bayes, bayes_geo, chc_arm, dlm, ridge
from scorecard.arms import panel as reading
from scorecard.arms.base import records, withheld
from scorecard.layout import Declaration, Layout, Limits

ENVIRONMENT = "policy-0"
LIMITS = Limits(memory="8G", timeout=3600.0, slots=1)
COMMON = {"version", "seed", "digest", "family", "environment", "k", "python", "versions"}
GEO_RECORD = COMMON | {
    "smoke",
    "plan_level",
    "weekly",
    "flags",
    "settings",
    "error",
    "gain",
    "gain_interval",
    "response",
    "n/a",
    "cost",
}
VARIANTS = bayes_geo.BayesGeo.variants


@pytest.fixture(scope="module")
def world():
    return GEO.world(ENVIRONMENT, GEO.pilots[ENVIRONMENT][0])


@pytest.fixture(scope="module")
def observed(world):
    return GEO.observe(world, 4)


@pytest.fixture(scope="module")
def quarter(world):
    return GEO.truth(world).quarter


@pytest.fixture(scope="module")
def made(observed):
    """The geo arm's records of the world at rung 4 from a smoke fit, and the fit they planned
    from."""
    kept = []

    class Keeping(bayes_geo.BayesGeo):
        def fit(self, observation, *, smoke):
            kept.append(super().fit(observation, smoke=smoke))
            return kept[-1]

    return records(Keeping(), observed, smoke=True), kept[0]


@pytest.fixture(scope="module")
def ran(tmp_path_factory, world):
    """A smoke run of family 8's first pilot world: the geo arm at rung 4, the CHC arm fitted
    once for rungs 0 and 4, both scored."""
    layout = Layout(tmp_path_factory.mktemp("geo") / "run")
    run.declare(
        layout,
        Declaration(
            family=8,
            environments=(ENVIRONMENT,),
            worlds=1,
            pilot=True,
            rungs=(0, 4),
            arms=("bayes-geo", "chc"),
            smoke=True,
            limits={"bayes-geo": LIMITS, "chc": LIMITS},
        ),
    )
    run.export_all(layout)
    assert layout.read().seeds(ENVIRONMENT) == range(world.seed, world.seed + 1)
    run.one(layout, "bayes-geo", ENVIRONMENT, world.seed, (4,))
    run.one(layout, "chc", ENVIRONMENT, world.seed, (0, 4))
    run.score(layout)
    return layout


def _numbers(geos):
    return sorted(int(g.removeprefix("geo-")) for g in geos)


def _same(row, other, *, but=()):
    """Two estimates hold the same fields, but those named."""
    for field in dataclasses.fields(row):
        if field.name not in but:
            np.testing.assert_array_equal(getattr(row, field.name), getattr(other, field.name))


def test_a_geo_export_reads_as_a_geo_panel_with_its_national_channel_allotted(observed):
    data = reading.panel(observed)
    count = observed.population.size
    assert data.geos == tuple(f"geo-{g}" for g in range(count))
    assert dict(data.hierarchy) == dict.fromkeys(data.geos, NATIONAL)
    np.testing.assert_array_equal(data.population, observed.population)
    np.testing.assert_array_equal(data.kpi, observed.sales.T)
    assert data.channels == ("search", "social", "tv")
    assert dict(data.national) == {"tv": Allotment.POPULATION}
    np.testing.assert_array_equal(data.spend[:, :, :2], observed.spend[:, :, :2].transpose(1, 0, 2))
    national = observed.spend[:, :, 2].sum(axis=1)
    share = observed.population / observed.population.sum()
    np.testing.assert_array_equal(data.spend[:, :, 2], share[:, None] * national[None, :])
    np.testing.assert_allclose(data.spend[:, :, 2], observed.spend[:, :, 2].T, rtol=1e-12, atol=0)
    whole = aggregate(data, (NATIONAL,))
    np.testing.assert_allclose(whole.kpi[0], observed.sales.sum(axis=1), rtol=1e-12, atol=0)


def test_an_export_whose_national_split_is_not_its_population_is_refused(observed):
    spend = observed.spend.copy()
    spend[10, 3, 2] *= 1.0 + 1e-12
    reading.panel(dataclasses.replace(observed, spend=spend))
    spend[10, 3, 2] *= 1.0 + 1e-6
    with pytest.raises(ValueError, match="'tv' is not split by population in 1 periods"):
        reading.panel(dataclasses.replace(observed, spend=spend))


def test_a_geo_export_with_controls_is_refused(observed):
    weeks = observed.sales.shape[0]
    with_controls = dataclasses.replace(
        observed,
        controls={"price": np.ones(weeks)},
        future_controls={"price": np.ones(observed.planned)},
    )
    with pytest.raises(ValueError, match=r"controls \['price'\] are the market's"):
        reading.panel(with_controls)


def test_each_geo_test_is_its_geos_spend_dark_over_its_weeks_read_over_the_readout(observed):
    lift = observed.lift
    read_back = reading.estimates(observed)
    assert sum(len(rows) for rows in read_back.values()) == len(lift.channel) == 12
    for name, rows in read_back.items():
        j = observed.channels.index(name)
        index = [i for i, c in enumerate(lift.channel) if c == name]
        for i, row in zip(index, rows, strict=True):
            group = list(lift.geos[i])
            first, end = lift.start[i] - 1, lift.start[i] - 1 + TEST + COOLDOWN
            dark = range(first, first + TEST)
            planned = observed.spend[:end, group, j].sum(axis=1)
            np.testing.assert_array_equal(row.planned, planned)
            np.testing.assert_array_equal(row.spend[first : first + TEST], 0.0)
            np.testing.assert_array_equal(np.delete(row.spend, dark), np.delete(planned, dark))
            assert _numbers(row.geos) == group
            assert (row.share, row.readout) == (1.0, TEST + COOLDOWN)
            assert row.lift == lift.delta_y[i] * TEST
            assert row.se == lift.sigma[i] * TEST
            assert row.end == reading.FIRST_WEEK + np.timedelta64(7 * (end - 1), "D")


def test_a_national_channels_test_spread_over_its_geos_people_is_each_geos_own_spend(
    observed, world
):
    """A group's spend at one spend per head, as the model over geos reads it, is each tested
    geo's own on a channel bought nationally, dark weeks and all."""
    people = world.history.population
    for row in reading.estimates(observed)["tv"]:
        group = _numbers(row.geos)
        weeks = row.planned.size
        for g in group:
            share = people[g] / people[group].sum()
            np.testing.assert_allclose(
                share * row.planned, world.history.spend[g, :weeks, 2], rtol=1e-12, atol=0
            )


def test_a_geo_row_whose_spend_is_not_its_geos_is_refused(observed):
    lift = dataclasses.replace(observed.lift, x=observed.lift.x * (1.0 + 1e-9))
    with pytest.raises(ValueError, match="where the history spent"):
        reading.estimates(dataclasses.replace(observed, lift=lift))


def test_a_plans_cells_are_each_geos_own_channels_then_the_national_one(observed, quarter):
    cells = reading.cells(observed)
    assert len(cells) == observed.status_quo.size == 21
    assert cells[:3] == ((0, "search"), (0, "social"), (1, "search"))
    assert cells[-1] == (None, "tv")
    assert [observed.channels[c] for c in quarter.channel] == [c for _, c in cells]
    np.testing.assert_allclose(reading.cell_spend(observed), quarter.history, rtol=1e-12, atol=0)


def test_the_status_quo_gains_nothing_and_a_plan_gains_what_the_worlds_cells_return(
    observed, world, quarter
):
    truth = GEO.truth(world)
    rows = world.history.cells()
    tv = bayes_geo.national([row[2] for row in rows], world.history.population)
    cells = tuple(
        tv if g is None else rows[g][observed.channels.index(c)] for g, c in reading.cells(observed)
    )
    assert reading.gains([cells], observed, observed.status_quo)[0] == 0.0
    weekly = truth.best.weekly
    gain = truth.best.worth - sum(
        cell.worth(float(w)) for cell, w in zip(truth.cells, quarter.status_quo, strict=True)
    )
    assert reading.gains([cells], observed, weekly)[0] == pytest.approx(gain, rel=1e-9, abs=0.0)


def test_the_market_is_the_geos_sums_and_its_box_keeps_every_cell_in_its_own(observed, quarter):
    whole = reading.market(observed)
    np.testing.assert_allclose(whole.sales, observed.sales.sum(axis=1), rtol=1e-12, atol=0)
    np.testing.assert_allclose(whole.spend, observed.market_spend, rtol=1e-12, atol=0)
    assert (whole.population, whole.national, whole.lift.geos) == (None, (), None)
    assert (whole.budget, whole.planned, whole.k) == (observed.budget, observed.planned, 4)
    np.testing.assert_allclose(
        whole.status_quo,
        np.bincount(np.asarray(quarter.channel), weights=observed.status_quo),
        rtol=1e-12,
        atol=0,
    )
    for edge in (whole.lower, whole.upper):
        allotted = quarter.allot(edge)
        assert np.all(allotted >= observed.lower * (1 - 1e-12))
        assert np.all(allotted <= observed.upper * (1 + 1e-12))
    assert np.any(quarter.allot(whole.lower * (1 - 1e-9)) < observed.lower)
    assert np.any(quarter.allot(whole.upper * (1 + 1e-9)) > observed.upper)


def test_the_markets_box_is_the_spends_whose_allotment_keeps_each_cell_in_its_box(
    observed, quarter
):
    """A cell whose box is not its share of its channel's: the channel's box shrinks to it."""
    quo, cell = observed.status_quo, np.arange(observed.status_quo.size)
    lower = np.where(cell == 3, 0.8 * quo, observed.lower)
    upper = np.where(cell == 4, 1.1 * quo, observed.upper)
    whole = reading.market(dataclasses.replace(observed, lower=lower, upper=upper))
    totals = np.bincount(np.asarray(quarter.channel), weights=quo)
    assert whole.lower[1] == pytest.approx(0.8 * totals[1], rel=1e-12, abs=0.0)
    assert whole.upper[0] == pytest.approx(1.1 * totals[0], rel=1e-12, abs=0.0)
    assert whole.lower[0] == pytest.approx(0.5 * totals[0], rel=1e-12, abs=0.0)
    shut = dataclasses.replace(observed, lower=np.where(cell == 3, 2.5 * quo, observed.lower))
    with pytest.raises(ValueError, match=r"no one spend of \['social'\]"):
        reading.market(shut)
    empty = dataclasses.replace(observed, status_quo=np.where(cell == 5, 0.0, quo))
    with pytest.raises(ValueError, match=r"cells \[5\] hold a status quo of 0"):
        reading.market(empty)


def test_a_markets_equal_split_is_the_benchs_equal_split_of_its_cells_summed_by_channel(
    observed, quarter
):
    """Not the market's own equal shares: the bench moves each cell into its box, which moves the
    channels' totals off equal."""
    whole = reading.market(observed)
    bench = np.bincount(np.asarray(quarter.channel), weights=quarter.equal_split())
    np.testing.assert_allclose(reading.equal_split(whole), bench, rtol=1e-12, atol=0)
    assert np.ptp(bench) > 1e-3 * bench.mean()
    with pytest.raises(ValueError, match="made over its cells; read its market"):
        reading.equal_split(observed)


@pytest.mark.parametrize("seed", [400022, 400024])
def test_a_markets_equal_split_whose_cells_sit_at_their_bounds_is_in_the_markets_box(seed):
    """Family 10's geos-3 worlds whose second channel's cells all sit at their upper bounds: the
    cells' sum came 1 ulp above the market's bound, and `cvar_allocate` refused it as the
    reference (the w30c pilot's 16 failed records)."""
    whole = reading.market(FEW_GEOS.observe(FEW_GEOS.world("geos-3", seed), 0))
    split = reading.equal_split(whole)
    assert np.all(split >= whole.lower)
    assert np.all(split <= whole.upper)
    assert split[1] == whole.upper[1]
    assert split.sum() == pytest.approx(whole.budget / whole.planned, rel=1e-12, abs=0.0)


def test_a_markets_tests_are_its_geo_tests_in_their_geos_share(observed):
    population = observed.population
    by_geo = reading.estimates(observed)
    in_share = reading.estimates(reading.market(observed))
    assert sum(len(rows) for rows in in_share.values()) == 12
    for name, rows in in_share.items():
        for row, geo in zip(rows, by_geo[name], strict=True):
            group = _numbers(geo.geos)
            assert row.geos is None
            assert row.share == pytest.approx(
                population[group].sum() / population.sum(), rel=1e-12, abs=0.0
            )
            assert 0.0 < row.share < 1.0
            _same(row, geo, but=("geos", "share"))
            assert geo.share == 1.0


def test_a_national_export_has_no_market_and_the_geo_arm_refuses_it():
    national = TRACK_M2.observe(TRACK_M2.world("drawn", 905), 0)
    with pytest.raises(ValueError, match="its own market"):
        reading.market(national)
    with pytest.raises(ValueError, match="no geo axis"):
        bayes_geo.BayesGeo().fit(national, smoke=True)
    for record in records(bayes_geo.BayesGeo(), national, smoke=True).values():
        assert record["failure_kind"] == "fit"
        assert "plan_level" not in record


def test_a_geo_export_without_its_tests_names_no_geos_and_is_one_at_every_rung(world, observed):
    bare = withheld(observed)
    assert (bare.k, bare.lift.geos, bare.lift.channel) == (0, (), ())
    assert bare.digest() == withheld(GEO.observe(world, 2)).digest()
    assert bare.digest() == GEO.observe(world, 0).digest()


def test_a_version_4_export_is_read_as_version_2_without_its_tests_and_as_its_market(
    world, observed
):
    """The universes and the switchback test are tests: without its tests a version-4 export is
    version 2's rung 0 at every rung, and its market is version 2's, its export kept beside it."""
    full = GEO.observe(world, 4, version=UNIVERSES)
    assert full.universes is not None
    assert full.switchback is not None
    bare = withheld(full)
    assert (bare.version, bare.universes, bare.switchback) == (VERSION, None, None)
    assert bare.digest() == GEO.observe(world, 0).digest()
    assert bare.digest() == withheld(GEO.observe(world, 2, version=UNIVERSES)).digest()
    market = reading.market(full)
    assert market.digest() == reading.market(observed).digest()
    assert market.geo is full
    plain = reading.second(full)
    assert plain.digest() == full.digest(VERSION) == observed.digest()
    assert reading.second(observed) is observed


@pytest.mark.parametrize("arm", [ridge.Ridge(), chc_arm.Chc()])
def test_a_national_arm_plans_a_geo_exports_market_one_weekly_spend_a_channel(
    arm, world, observed, quarter
):
    made = records(arm, observed, smoke=True)
    assert tuple(made) == arm.variants
    for record in made.values():
        assert record["error"] is None
        assert record["plan_level"] == "market"
        assert record["digest"] == observed.digest()
        weekly = np.asarray(record["weekly"])
        assert weekly.shape == (3,)
        assert quarter.feasible(quarter.allot(weekly))
    scored = score_world(GEO, ENVIRONMENT, world.seed, 4, made, pilot=True, archived=observed)
    for name in arm.variants:
        assert scored.arms[name].moved is None
        assert math.isfinite(scored.arms[name].regret)
    assert scored.arms[arm.variants[0]].unread is None


@pytest.mark.parametrize(
    ("module", "fitter", "arm"),
    [(bayes, "fit_bayes", bayes.Bayes()), (dlm, "search_media", dlm.Dlm())],
)
def test_a_national_arm_reads_a_geo_test_in_its_geos_share_of_the_market(
    observed, monkeypatch, module, fitter, arm
):
    seen = []

    def held(*args, estimates=None, **kwargs):
        seen.append(estimates)
        raise RuntimeError("held")

    monkeypatch.setattr(module, fitter, held)
    for record in records(arm, observed, smoke=True).values():
        assert (record["failure_kind"], record["plan_level"]) == ("fit", "market")
    expected = reading.estimates(reading.market(observed))
    assert seen[0].keys() == expected.keys()
    for name, rows in expected.items():
        for row, other in zip(rows, seen[0][name], strict=True):
            _same(row, other)
    assert sum(len(rows) for rows in seen[0].values()) == 12


def test_the_geo_arm_runs_a_world_end_to_end_and_is_scored_beside_the_built_in_arms(
    ran, world, capsys
):
    (line,) = [json.loads(line) for line in ran.scores.read_text().splitlines()]
    assert line["k"] == 4
    assert line["channels"] == 21
    for name in (*VARIANTS, "chc"):
        assert line["arms"][name]["failure"] is None
        assert math.isfinite(line["arms"][name]["regret"])
        assert line["arms"][name]["moved"] is None
    assert line["arms"][bayes_geo.PLUG_IN]["record"]["plan_level"] == "cells"
    assert line["arms"]["chc"]["record"]["plan_level"] == "market"
    assert line["arms"][bayes_geo.PLUG_IN]["unread"].startswith("unmapped: ")
    with pytest.raises(ValueError, match="reads the tests"):
        run.one(ran, "bayes-geo", ENVIRONMENT, world.seed, (0, 4))
    report.main(ran, [*VARIANTS, "chc"])
    shown = capsys.readouterr().out.splitlines()
    assert any(line.split()[:4] == ["bayes", "geo", "plug-in", "regret"] for line in shown)


def test_each_geo_record_plans_every_cell_in_its_box_on_the_budget(made, observed):
    for name, record in made[0].items():
        assert set(record) == GEO_RECORD, name
        assert record["error"] is None
        assert record["plan_level"] == "cells"
        weekly = np.asarray(record["weekly"])
        assert weekly.shape == (21,)
        assert np.all(weekly >= observed.lower * (1.0 - 1e-9))
        assert np.all(weekly <= observed.upper * (1.0 + 1e-9))
        assert weekly.sum() * observed.planned == pytest.approx(observed.budget, rel=1e-9, abs=0.0)
        assert record["response"] == {"unmapped": bayes_geo.UNMAPPED}
        assert set(record["n/a"]) == {"forecast"}
        low, high = record["gain_interval"]
        assert low <= record["gain"] <= high
        settings = record["settings"]
        assert (settings["families"], settings["start"], settings["centred"]) == (
            ["hill"],
            "uniform",
            [],
        )
        assert settings["lift_rows"] == 12
        assert settings["sampler"] == dataclasses.asdict(bayes.SMOKE)
        assert settings["readings"] == bayes_geo.SMOKE_READINGS
        assert record["flags"]["hill weight"] == 1.0
        assert {"hill divergences", "hill max rhat", "hill min ess bulk"} <= set(record["flags"])
        assert isinstance(report.past(record), bool)
        json.dumps(record, allow_nan=False)


def test_the_geo_arm_repeats_itself_at_a_seed_from_a_child_of_the_exports(made, ran, observed):
    """The run's records are the arm's on version 2's arrays, each naming the version-4 export the
    run held."""
    exported = read(ran.export(8, ENVIRONMENT, 4, observed.seed))
    named = {"cost", "provenance", "version", "digest"}
    for name, record in made[0].items():
        again = json.loads(ran.record(name, 8, ENVIRONMENT, 4, observed.seed).read_text())
        assert (again["version"], again["digest"]) == (UNIVERSES, exported.digest())
        assert {k: v for k, v in record.items() if k not in named} == {
            k: v for k, v in again.items() if k not in named
        }
    child = np.random.SeedSequence(observed.seed).spawn(1)[0]
    assert made[0][bayes_geo.PLUG_IN]["settings"]["seed"] == int(child.generate_state(1)[0])


def test_a_national_channels_geos_are_one_channel_of_its_national_spend(made, observed):
    """On a reading, the one channel's return on the national spend is the sum of its geos' own
    on their allotted spend, week by week."""
    fitted = made[1]
    fit, count = fitted.fit, len(fitted.readings)
    national = observed.spend[:, :, 2].sum(axis=1)
    share = fit.population / fit.population.sum()
    by_geo = [fit.channel_draws(g, count) for g in fit.geos]
    for i in (0, count - 1):
        summed = sum(
            np.asarray(drawn[i]["tv"](s * national)) for drawn, s in zip(by_geo, share, strict=True)
        )
        one = fitted.readings[i][-1]
        np.testing.assert_allclose(np.asarray(one(national)), summed, rtol=1e-12, atol=0)
        assert fitted.readings[i][:2] == (by_geo[0][i]["search"], by_geo[0][i]["social"])


def test_the_plug_in_is_plan_geos_on_the_medians_with_tv_one_national_spend(made, observed):
    fit = made[1].fit
    data = reading.panel(observed)
    ranges = [
        Range(float(a), float(b)) for a, b in zip(observed.lower, observed.upper, strict=True)
    ]
    cells = {
        (data.geos[g], Channel(c)): r
        for (g, c), r in zip(reading.cells(observed), ranges, strict=True)
        if g is not None
    }
    plan = plan_geos(
        data,
        {g: {Channel(c): ch for c, ch in fit.channels(g).items()} for g in data.geos},
        observed.budget,
        observed.planned,
        GeoLimits(cells=cells, national={Channel("tv"): ranges[-1]}),
    )
    weekly = np.asarray(made[0][bayes_geo.PLUG_IN]["weekly"])
    np.testing.assert_array_equal(weekly[:-1], plan.spend[:, :2].reshape(-1))
    assert weekly[-1] == plan.spend[:, 2].sum()
    assert data.population is not None
    share = data.population / data.population.sum()
    np.testing.assert_allclose(plan.spend[:, 2] / weekly[-1], share, rtol=1e-12, atol=0)


def test_the_cvar_plan_is_cvar_allocates_over_the_readings_of_every_cell(made, observed):
    fitted = made[1]
    plan = cvar_allocate(
        fitted.readings,
        observed.budget,
        observed.planned,
        level=bayes.LEVEL,
        against=observed.status_quo,
        lower=observed.lower,
        upper=observed.upper,
        history=reading.cell_spend(observed),
    )
    record = made[0][bayes_geo.CVAR]
    np.testing.assert_array_equal(record["weekly"], plan.spend)
    gains = reading.gains(fitted.readings, observed, np.asarray(plan.spend))
    assert record["gain"] == pytest.approx(float(np.mean(gains)), rel=1e-12, abs=0.0)


def test_the_plug_in_arm_records_the_geo_arms_plug_in_alone(made, observed, monkeypatch):
    """`bayes-geo-plug-in` on the geo arm's fit: one record, the geo arm's plug-in record but for
    its cost, and no CVaR plan."""
    full, fitted = made
    monkeypatch.setattr(bayes_geo.BayesGeo, "fit", lambda self, observation, *, smoke: fitted)
    planned = []
    plan = bayes_geo.BayesGeo.plan

    def counted(self, fit, variant, observation):
        planned.append(variant)
        return plan(self, fit, variant, observation)

    monkeypatch.setattr(bayes_geo.BayesGeo, "plan", counted)
    alone = records(bayes_geo.BayesGeoPlugIn(), observed, smoke=True)
    assert planned == [bayes_geo.PLUG_IN]
    assert list(alone) == [bayes_geo.PLUG_IN]
    kept = {k: v for k, v in alone[bayes_geo.PLUG_IN].items() if k != "cost"}
    assert kept == {k: v for k, v in full[bayes_geo.PLUG_IN].items() if k != "cost"}


def test_a_run_declares_the_geo_arm_or_its_plug_in_alone_not_both():
    declared = Declaration(
        family=8,
        environments=(ENVIRONMENT,),
        worlds=1,
        pilot=True,
        rungs=(0,),
        arms=("bayes-geo", "bayes-geo-plug-in"),
        smoke=True,
        limits={"bayes-geo": LIMITS, "bayes-geo-plug-in": LIMITS},
    )
    with pytest.raises(ValueError, match="names of their own"):
        run.names(declared)


def test_family_8_is_a_runs_family_and_its_exports_read_back(ran, world):
    exported = read(ran.export(8, ENVIRONMENT, 4, world.seed))
    assert exported.population is not None
    assert (exported.family, exported.population.size, exported.national) == (8, 10, ("tv",))
    assert exported.version == UNIVERSES
    assert exported.digest() == GEO.observe(world, 4, version=UNIVERSES).digest()
