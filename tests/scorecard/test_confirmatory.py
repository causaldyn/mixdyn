"""A confirmatory run: each environment's worlds no scored run read, a prefix of their own seeded
order, declared as a scored run is and written as before where a run is not one."""

import dataclasses
import json

import pytest

pytest.importorskip("causaldyn_bench")

from causaldyn_bench.scorecard.scoring import score_world
from causaldyn_bench.scorecard.track_m2 import TRACK_M2
from test_claim_roots import _m2

from scorecard import claim
from scorecard.layout import Declaration, Layout

SCORED = Declaration(
    family=2,
    environments=("kappa-0.5", "kappa-1"),
    worlds=3,
    pilot=False,
    rungs=(0,),
    arms=("ridge",),
    bench="abc1234",
)
CONFIRMATORY = dataclasses.replace(SCORED, confirmatory=True)


def test_a_confirmatory_run_draws_the_block_past_every_scored_world():
    assert SCORED.seeds("kappa-1") == range(242_000, 242_003)
    assert CONFIRMATORY.seeds("kappa-0.5") == range(250_000, 250_003)
    assert CONFIRMATORY.seeds("kappa-1") == range(251_000, 251_003)
    family_0 = Declaration(
        family=0,
        environments=("drawn", "reference"),
        worlds=2,
        pilot=False,
        rungs=(0,),
        arms=("ridge",),
        bench="abc1234",
        confirmatory=True,
    )
    assert family_0.seeds("drawn") == range(12_000, 12_002)
    assert family_0.seeds("reference") == range(22_000, 22_002)
    assert CONFIRMATORY.scored


def test_a_confirmatory_run_is_written_and_read_back_and_a_scored_one_as_before(tmp_path):
    Layout(tmp_path / "c").declare(CONFIRMATORY)
    assert Layout(tmp_path / "c").read() == CONFIRMATORY
    assert json.loads((tmp_path / "c" / "run.json").read_text())["confirmatory"] is True
    Layout(tmp_path / "s").declare(SCORED)
    assert "confirmatory" not in json.loads((tmp_path / "s" / "run.json").read_text())


def test_a_run_of_pilots_is_not_confirmatory():
    with pytest.raises(ValueError, match="pilots or confirmatory"):
        dataclasses.replace(SCORED, pilot=True, bench=None, confirmatory=True)


def test_a_confirmatory_world_is_scored_at_its_place_in_its_own_order():
    scored = score_world(TRACK_M2, "drawn", 12_001, 0, {})
    assert (scored.seed, scored.index, scored.pilot) == (12_001, 1, False)


@pytest.mark.parametrize("flipped", ["f0", "f10", "f10-ts", "f10-robyn"])
def test_a_run_s_roots_draw_the_same_worlds_or_are_refused(tmp_path, flipped):
    """A run's roots, its pool's, its joined root and the roots beside it, all confirmatory or
    none: a root that draws other worlds than its run's is refused, not joined on worlds it does
    not hold."""
    roots = _m2(tmp_path)
    layout = Layout(tmp_path / flipped)
    d = layout.read()
    layout.declaration.unlink()
    layout.declare(dataclasses.replace(d, confirmatory=True))
    with pytest.raises(ValueError, match=r"does not hold|not a root beside|apart"):
        claim.read_runs(roots)


SECOND = dataclasses.replace(CONFIRMATORY, second_confirmatory=True)


def test_a_second_confirmatory_run_draws_the_blocks_past_the_first_run_s_worlds():
    assert SECOND.seeds("kappa-0.5") == range(250_300, 250_303)
    assert SECOND.seeds("kappa-1") == range(251_300, 251_303)
    family_0 = Declaration(
        family=0,
        environments=("drawn", "reference"),
        worlds=2,
        pilot=False,
        rungs=(0,),
        arms=("ridge",),
        bench="abc1234",
        confirmatory=True,
        second_confirmatory=True,
    )
    assert family_0.seeds("drawn") == range(12_300, 12_302)
    assert family_0.seeds("reference") == range(22_300, 22_302)
    assert dataclasses.replace(SECOND, worlds=520).seeds("kappa-1")[-1] == 251_819


def test_a_first_confirmatory_run_stays_below_the_second_s_worlds():
    assert dataclasses.replace(CONFIRMATORY, worlds=300).seeds("kappa-1")[-1] == 251_299
    with pytest.raises(ValueError, match="at most"):
        dataclasses.replace(CONFIRMATORY, worlds=301)


def test_a_second_confirmatory_run_is_written_and_read_back_and_a_first_as_before(tmp_path):
    Layout(tmp_path / "second").declare(SECOND)
    assert Layout(tmp_path / "second").read() == SECOND
    assert json.loads((tmp_path / "second" / "run.json").read_text())["second_confirmatory"]
    Layout(tmp_path / "first").declare(CONFIRMATORY)
    assert "second_confirmatory" not in json.loads((tmp_path / "first" / "run.json").read_text())


def test_a_second_confirmatory_run_is_a_confirmatory_run():
    with pytest.raises(ValueError, match="declare it confirmatory"):
        dataclasses.replace(SCORED, second_confirmatory=True)


def test_a_second_confirmatory_world_is_scored_at_its_place_in_its_own_order():
    scored = score_world(TRACK_M2, "drawn", 12_301, 0, {})
    assert (scored.seed, scored.index, scored.pilot) == (12_301, 1, False)


@pytest.mark.parametrize("flipped", ["f0", "f10-ts", "f10-robyn"])
def test_a_run_s_roots_are_all_a_first_or_all_a_second_confirmatory_run(tmp_path, flipped):
    roots = _m2(tmp_path)
    for root in tmp_path.iterdir():
        layout = Layout(root)
        d = layout.read()
        layout.declaration.unlink()
        second = root.name == flipped
        layout.declare(dataclasses.replace(d, confirmatory=True, second_confirmatory=second))
    with pytest.raises(ValueError, match=r"does not hold|not a root beside|apart"):
        claim.read_runs(roots)
