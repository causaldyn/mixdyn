"""The bench's families a run may hold beside Track M v2, drift and geo: 2 to 7, each a Track M v2
world with one mechanism added, 10, the geo family at other counts of geos, 14, Robyn's home
ground, and 15, siMMMulator's own worlds read from the directory a run declares: a pilot world of
each is exported, recorded and scored."""

import json

import pytest

pytest.importorskip("causaldyn_bench")

import causaldyn_bench
from causaldyn_bench.scorecard.observe import read

from scorecard import run
from scorecard.arms.base import Planned
from scorecard.layout import Declaration, Layout, Limits
from scorecard.provenance import tree

RECORDED = tree(causaldyn_bench) / "tests" / "fixtures" / "simmmulator"
"""The bench's own record of siMMMulator's first pilot world."""


class Still:
    """This engine's arm, which plays the status quo."""

    variants = ("still",)
    tests = True
    cells = False

    def fit(self, observation, *, smoke):
        return observation.status_quo

    def plan(self, fit, variant, observation):
        return Planned(weekly=fit)


@pytest.mark.parametrize(
    ("family", "environment", "directory"),
    [
        (2, "kappa-1", None),
        (3, "r-0.97", None),
        (4, "logistic-4", None),
        (5, "rho-0.6", None),
        (6, "both", None),
        (7, "L-52", None),
        (10, "geos-3", None),
        (14, "genre", None),
        (15, "demo", str(RECORDED)),
    ],
)
def test_a_pilot_world_of_the_family_is_exported_and_scored(
    tmp_path, monkeypatch, family, environment, directory
):
    monkeypatch.setitem(run.ARMS, "still", Still)
    declared = Declaration(
        family=family,
        environments=(environment,),
        worlds=1,
        pilot=True,
        rungs=(0,),
        arms=("still",),
        limits={"still": Limits(memory="1G", timeout=60.0, slots=1)},
        world_directory=directory,
    )
    out = Layout(tmp_path / "run")
    run.declare(out, declared)
    assert out.read() == declared
    run.export_all(out)
    (seed,) = declared.seeds(environment)
    exported = read(out.export(family, environment, 0, seed))
    assert (exported.family, exported.environment, exported.seed) == (family, environment, seed)
    run.one(out, "still", environment, seed, (0,))
    run.score(out)
    (line,) = out.scores.read_text().splitlines()
    world = json.loads(line)
    assert (world["family"], world["environment"], world["seed"]) == (family, environment, seed)
    assert world["arms"]["still"]["failure"] is None
    assert world["arms"]["still"]["regret"] == world["arms"]["status quo"]["regret"]
    assert world["arms"]["equal split"]["regret"] >= 0.0
