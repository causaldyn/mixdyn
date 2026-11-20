"""The bench's families 9, 12 and 13 in a run: 9, family 8's geo worlds whose geos' own media lift
their ring neighbours, 12, Meridian's home ground, and 13, PyMC-Marketing's. A pilot world of each
is declared, exported, planned by the CHC arm, by the oracle and by a tool's script, scored and
reported as family 8's and the national families' are; a run of each that declares the claim's
margin is read by the claim, though no prediction names them. The truth family 9 is scored against
counts what a geo's own media spill into its neighbours. And the harness holds every family the
bench declares."""

import importlib
import inspect
import json
import math
import pkgutil
import sys
from dataclasses import fields, replace
from functools import partial
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("causaldyn_bench")

import causaldyn_bench.scorecard
from causaldyn_bench.budget_regret import TIE
from causaldyn_bench.mmm_decision import PLANNED
from causaldyn_bench.scorecard.observe import UNIVERSES, VERSION, read
from causaldyn_bench.scorecard.scoring import BUILT_IN, WorldRecord
from causaldyn_bench.scorecard.seeds import TRACK_M2
from causaldyn_bench.scorecard.truth import GAP, geo_cells, geo_parts, geo_quarter

from scorecard import claim, report, run
from scorecard.arms.base import Planned
from scorecard.layout import FAMILIES, Declaration, Layout, Limits
from scorecard.tools import TOOLS, Tool

FAKE = Path(__file__).with_name("fake_tool.py")
COMMON = {"version", "seed", "digest", "family", "environment", "k", "python", "versions"}
CHC_RECORD = COMMON | {
    "smoke",
    "weekly",
    "flags",
    "settings",
    "error",
    "gain",
    "gain_interval",
    "n/a",
    "response",
    "cost",
    "provenance",
}
"""The CHC arm's record as `run.one` writes it of a national export; of a geo export it names the
level it planned at too (`plan_level`)."""
NEW = [(9, "spillover-0.2", True), (12, "priors", False), (13, "priors", False)]
"""Each new family, the environment its pilot world is drawn from, and whether it has geos."""


class Oracle:
    """This engine's arm, which plays the world's best plan: its truth's, read from the bench."""

    variants = ("oracle",)
    tests = False
    cells = True

    def fit(self, observation, *, smoke):
        family = FAMILIES[observation.family]
        assert not isinstance(family, type)  # a family that draws its worlds
        world = family.world(observation.environment, observation.seed)
        return family.truth(world).best.weekly

    def plan(self, fit, variant, observation):
        return Planned(weekly=np.asarray(fit))


class Still:
    """This engine's arm, which plays the status quo."""

    variants = ("still",)
    tests = True
    cells = False

    def fit(self, observation, *, smoke):
        return observation.status_quo

    def plan(self, fit, variant, observation):
        return Planned(weekly=fit)


def declared_numbers():
    """Every family the bench declares, by number: each class of `causaldyn_bench.scorecard`'s
    modules that names its own stream and observes and scores its worlds."""
    package = causaldyn_bench.scorecard
    numbers = set()
    for module in pkgutil.iter_modules(package.__path__):
        held = importlib.import_module(f"{package.__name__}.{module.name}")
        for value in vars(held).values():
            if (
                inspect.isclass(value)
                and value.__module__ == held.__name__
                and isinstance(vars(value).get("stream"), int)
                and callable(getattr(value, "observe", None))
                and callable(getattr(value, "truth", None))
            ):
                numbers.add(value.stream - TRACK_M2)
    return numbers


def test_the_harness_holds_every_family_the_bench_declares():
    assert declared_numbers() >= {0, 8, 9, 12, 13, 14, 15}
    assert set(FAMILIES) == declared_numbers()


@pytest.mark.parametrize(("family", "environment", "geos"), NEW)
def test_a_pilot_world_runs_end_to_end_and_is_scored(
    tmp_path, monkeypatch, capsys, family, environment, geos
):
    """Declared, exported, planned by the CHC arm at its smoke settings, the oracle and the fake
    tool, scored and reported: the export round-trips its digest, every record names it, the
    oracle's regret is 0 and the status quo's is above it. A geo family is exported at version 4,
    and the tool is handed it as version 2, whose digest its record names."""
    modes = tmp_path / "modes.json"
    monkeypatch.setitem(run.ARMS, "oracle", Oracle)
    monkeypatch.setitem(TOOLS, "fake", Tool((sys.executable, str(FAKE), "--modes", str(modes))))
    monkeypatch.setattr(run, "_scoped", lambda limits, command: command)  # no systemd here
    declared = Declaration(
        family=family,
        environments=(environment,),
        worlds=1,
        pilot=True,
        rungs=(0,),
        arms=("chc", "oracle"),
        external=("fake",),
        smoke=True,
        limits={
            "chc": Limits(memory="4G", timeout=600.0, slots=1),
            "oracle": Limits(memory="1G", timeout=60.0, slots=1),
            "fake": Limits(memory="256M", timeout=60.0, slots=1),
        },
        reserve="0",
        settle=0.0,
    )
    out = Layout(tmp_path / "run")
    run.declare(out, declared)
    assert out.read() == declared
    run.export_all(out)
    (seed,) = declared.seeds(environment)
    path = out.export(family, environment, 0, seed)
    exported = read(path)
    with np.load(path) as saved:
        digest = str(saved["digest"])
    assert exported.digest() == digest
    source = declared.source()
    version = UNIVERSES if geos else VERSION
    assert exported.version == version
    observe = source.observe
    if geos:
        observe = partial(source.observe, version=UNIVERSES)
    assert observe(source.world(environment, seed), 0).digest() == digest
    assert (exported.family, exported.environment, exported.seed) == (family, environment, seed)
    assert (exported.population is not None) == geos

    modes.write_text(json.dumps({str(seed): "record"}))
    run.one(out, "chc", environment, seed, (0,))
    run.one(out, "oracle", environment, seed, (0,))
    run.run_arm(out, "fake")
    made = {
        arm: json.loads(out.record(arm, family, environment, 0, seed).read_text())
        for arm in ("chc", "chc cvar 0.1 vs equal", "oracle", "fake")
    }
    for arm, record in made.items():
        assert record["error"] is None, arm
        assert record["digest"] == exported.digest(record["version"]), arm
        assert record["version"] == (VERSION if arm == "fake" else version), arm
    handed = read(out.inputs("fake", family, environment, 0, seed) / f"world_{seed}.npz")
    assert (handed.version, handed.digest()) == (VERSION, exported.digest(VERSION))
    assert set(made["chc"]) == CHC_RECORD | ({"plan_level"} if geos else set())
    if geos:
        assert made["chc"]["plan_level"] == "market"

    run.score(out)
    (line,) = out.scores.read_text().splitlines()
    world = json.loads(line)
    assert set(world) == {f.name for f in fields(WorldRecord)}
    assert (world["family"], world["environment"], world["seed"]) == (family, environment, seed)
    arms = world["arms"]
    assert set(arms) == {*BUILT_IN, "chc", "chc cvar 0.1 vs equal", "oracle", "fake"}
    for arm, scored in arms.items():
        assert scored["failure"] is None, arm
        assert math.isfinite(scored["regret"]), arm
    # the best plan scored against itself: 0 but for rounding, below the oracle's certified gap
    assert arms["oracle"]["regret"] == pytest.approx(0.0, abs=GAP)
    assert arms["status quo"]["regret"] > arms["oracle"]["regret"] + TIE
    assert arms["fake"]["regret"] == arms["status quo"]["regret"]  # the fake tool plays it

    report.main(out, ["chc", "chc cvar 0.1 vs equal", "oracle"])
    shown = capsys.readouterr().out
    assert f": {source.name}, smoke, not a result;" in shown
    for label in source.labels:
        assert f"by construction: {label}" in shown
    assert f"k = 0, {environment}:" in shown
    for arm in (*BUILT_IN, "chc", "oracle", "fake"):
        assert any(
            row.split()[: len(arm.split()) + 1] == [*arm.split(), "regret"]
            for row in shown.splitlines()
        )


@pytest.mark.parametrize(("family", "environment", "geos"), NEW)
def test_a_run_that_declares_the_margin_is_read_by_the_claim(
    tmp_path, monkeypatch, capsys, family, environment, geos
):
    """No pre-registered prediction names the new families: a run of one that declares the
    claim's margin is still compared rival by rival and reported with its claim."""
    monkeypatch.setitem(run.ARMS, "still", Still)
    declared = Declaration(
        family=family,
        environments=(environment,),
        worlds=2,
        pilot=True,
        rungs=(0,),
        arms=("still",),
        limits={"still": Limits(memory="1G", timeout=60.0, slots=1)},
        default="still",
        margin=0.05,
    )
    out = Layout(tmp_path / "run")
    run.declare(out, declared)
    run.export_all(out)
    for seed in declared.seeds(environment):
        run.one(out, "still", environment, seed, (0,))
    run.score(out)
    records = [
        WorldRecord.from_json(json.loads(line)) for line in out.scores.read_text().splitlines()
    ]
    compared = claim.compare(records, declared)
    assert [(c.family, c.k, c.rival) for c in compared] == [
        (family, 0, rival) for rival in claim.rivals(declared)
    ]
    report.main(out, ["still"])
    shown = capsys.readouterr().out
    assert "the claim on this run alone, still less every rival:" in shown
    assert f"family {family}, k = 0, less" in shown


@pytest.mark.parametrize(("environment", "spill"), [("spillover-0.2", 0.2), ("spillover-0.5", 0.5)])
def test_a_spilled_geos_cell_is_worth_what_it_lifts_in_every_geo(environment, spill):
    """In a pilot world of family 9, one geo's social and the national TV spend more over the
    quarter than at the status quo: what the world lifts, summed over every geo, the quarter and
    the kernel's tail, is what the truth's cell gains, and for the geo's own channel that is
    `1 + s` times what it lifts at home; TV does not spill."""
    family = FAMILIES[9]
    assert not isinstance(family, type)
    history = family.world(environment, family.pilots[environment][0]).history
    assert history.mix.spillover == spill
    quarter = geo_quarter(history)
    cells = geo_cells(history, quarter)
    parts = geo_parts(history)
    geos, weeks, channels = history.spend.shape
    horizon = weeks + PLANNED + history.mix.length - 1
    local = sum(not channel.national for channel in history.mix.channels)
    social = next(i for i, c in enumerate(history.mix.channels) if c.name == "social")
    tv = next(i for i, c in enumerate(history.mix.channels) if c.national)
    geo = 1
    for cell, spread in ((geo * local + social, None), (len(parts) - 1, history.shares)):
        part = parts[cell]
        assert part.channel == (social if spread is None else tv)
        lifts = []
        for weekly in (quarter.status_quo[cell], 1.7 * quarter.status_quo[cell]):
            spend = np.zeros((geos, horizon, channels))
            spend[:, :weeks] = history.spend
            if spread is None:
                spend[geo, weeks : weeks + PLANNED, part.channel] = weekly
            else:
                spend[:, weeks : weeks + PLANNED, part.channel] = spread[:, None] * weekly
            world = replace(history, spend=spend, drift=np.ones((horizon, channels)))
            market = world.lift(spend)[:, weeks:].sum()
            home = replace(world, mix=replace(world.mix, spillover=0.0)).lift(spend)[:, weeks:]
            lifts.append((market, home.sum(), cells[cell].worth(weekly)))
        (market0, home0, worth0), (market1, home1, worth1) = lifts
        assert worth1 - worth0 == pytest.approx(market1 - market0, rel=1e-9, abs=0.0)
        ratio = 1.0 + spill if spread is None else 1.0
        assert market1 - market0 == pytest.approx(ratio * (home1 - home0), rel=1e-9, abs=0.0)
