"""The claim read over a family's roots as M2 lays them out: a pool whose runs test at different
rungs above 0, each environment read at its own rung, and the roots beside a run, whose tools are
read with the run on their own environments and caps, Robyn's on a prefix of each environment's
worlds and Meridian's on two of family 10's environments; `claim read` handed every root."""

import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import t as student

pytest.importorskip("causaldyn_bench")

from causaldyn_bench.scorecard.scoring import WorldRecord
from test_claim import _score

from scorecard import claim
from scorecard.layout import Declaration, Layout, stamp

DEFAULT = "dlm plug-in"
TWO_STAGE = "dlm two-stage"
BUILT_IN = ("status quo", "equal split")
CHC = ("chc", "chc cvar 0.1 vs equal")
DLM = (DEFAULT, "dlm mixture", "dlm cvar 0.1", "dlm cvar 0.1 vs equal")
BAYES = ("stacked", "cvar 0.1", "cvar 0.1 vs equal")
MAIN_ARMS = (DEFAULT, *CHC, "meridian", "orbit")
GEO = ("geos-3", "geos-8")
BAYES_GEO = ("bayes geo plug-in", "bayes geo cvar 0.1")
POOL = ("f7a", "f7b")
TESTED = (("L-52", 1), ("L-78", 2), ("L-104", 2))
"""Family 7's tested rungs, as M2 runs them: 1 on L-52, 2 on L-78 and L-104."""
TESTED_LINE = "the tested rung: k = 1 on L-52, 2 on L-78 and L-104"


def _regret(world, arm):
    """An arm's regret on one world at one rung, drawn from the world's place and the arm's name;
    the default's and the two-stage arm's 0.1 below the rest's."""
    place = f"{world.family}/{world.environment}/{world.index}/{world.k}/{arm}".encode()
    seed = int.from_bytes(hashlib.sha256(place).digest()[:8], "little")
    drawn = 0.2 + 0.1 * float(np.random.default_rng(seed).random())
    return drawn - 0.1 if arm in (DEFAULT, TWO_STAGE) else drawn


def _worlds(declaration, arms):
    """Every world of the run at each of its rungs, the built-in arms and each of `arms` scored."""
    d = declaration
    out = []
    for environment in d.environments:
        for index, seed in enumerate(d.seeds(environment)):
            for k in d.rungs:
                world = WorldRecord(
                    family=d.family,
                    environment=environment,
                    seed=seed,
                    index=index,
                    pilot=d.pilot,
                    k=k,
                    budget=1000.0,
                    best=2000.0,
                    channels=3,
                    oracle_bounds=0,
                    covariates={},
                    arms={},
                )
                scores = {arm: _score(_regret(world, arm)) for arm in (*BUILT_IN, *arms)}
                out.append(dataclasses.replace(world, arms=scores))
    return out


def _side(records, side, arms):
    """The worlds the root `side` holds of the run's `records`, their built-in arms the run's and
    each of `arms` scored; `chc` drawn apart from the run's, so that joining it would be refused."""
    out = []
    for r in records:
        if r.environment in side.environments and r.index < side.worlds and r.k in side.rungs:
            scores = {name: r.arms[name] for name in BUILT_IN}
            scores |= {a: _score(_regret(r, a) + (7.0 if a in CHC else 0.0)) for a in arms}
            out.append(dataclasses.replace(r, arms=scores))
    return out


def _run(family, environments, rungs, **change):
    """A main root as M2 declares one: the claim's default and margin, Hall's reader."""
    held = Declaration(
        family=family,
        environments=environments,
        worlds=8,
        pilot=False,
        bench="316fada",
        rungs=rungs,
        arms=("dlm", "chc"),
        external=("meridian", "orbit"),
        default=DEFAULT,
        margin=0.05,
        reader="hall",
    )
    return dataclasses.replace(held, **change)


def _beside(run, **change):
    """A root beside `run` as M2 declares Robyn's: `chc` and Robyn, capped and exploratory, on the
    first 4 worlds of each environment."""
    held = Declaration(
        family=run.family,
        environments=run.environments,
        worlds=4,
        pilot=run.pilot,
        bench=run.bench,
        rungs=run.rungs,
        arms=("chc",),
        external=("robyn",),
        caps={"robyn": 4},
        exploratory_caps=True,
    )
    return dataclasses.replace(held, **change)


def _write(root, declaration, records):
    layout = Layout(root)
    layout.declare(declaration)
    layout.scores.write_text("".join(json.dumps(r.as_json()) + "\n" for r in records))
    return root


def _with(records, tool, scores):
    """`records` with `tool` scored from `scores`, by environment, seed and rung, and 9 on every
    other world, which a cap never reads."""
    return [
        dataclasses.replace(
            r, arms={**r.arms, tool: scores.get((r.environment, r.seed, r.k), _score(9.0))}
        )
        for r in records
    ]


def _bare(comparisons):
    """The comparisons without the environments and tested rungs they name."""
    return [dataclasses.replace(c, environments=(), tested=()) for c in comparisons]


# ------------------------------------------------------------- a pool whose runs test apart


def _family_7():
    """Family 7 as one run over its three environments at rungs 0 and 1, and as M2's pool of two
    runs, L-52 at rungs 0 and 1 and L-78 and L-104 at 0 and 2, holding the same scores."""
    whole = _run(7, ("L-52", "L-78", "L-104"), (0, 1))
    records = _worlds(whole, MAIN_ARMS)
    a = _run(7, ("L-52",), (0, 1), pool=POOL)
    b = _run(7, ("L-78", "L-104"), (0, 2), pool=POOL)
    held_a = [r for r in records if r.environment == "L-52"]
    held_b = [
        dataclasses.replace(r, k=2) if r.k == 1 else r for r in records if r.environment != "L-52"
    ]
    return (whole, records), [(Path("f7a"), a, held_a), (Path("f7b"), b, held_b)]


def test_a_pool_whose_runs_test_at_different_rungs_reads_each_environment_at_its_own():
    (whole, records), runs = _family_7()
    ((declared, joined),) = claim.pooled(runs)
    assert (declared.environments, declared.rungs) == (("L-52", "L-78", "L-104"), (0, 1))
    assert claim.tested_rungs([d for _, d, _ in runs]) == TESTED
    comparisons = claim.compare(joined, declared, tested=TESTED)
    # each environment's own rung read as one comparison over all three, once a rival
    assert _bare(comparisons) == claim.compare(records, whole)
    assert [(c.k, c.tested) for c in comparisons] == [(0, ())] * 4 + [(1, TESTED)] * 4
    printed = claim.lines(comparisons, 0.05)
    assert [line.startswith("  family 7, k = 1,") for line in printed if TESTED_LINE in line] == [
        True
    ] * 4
    assert printed[-2] == "the claim holds: 8 of 8 comparisons within 0.05"
    assert printed[-1].endswith("over 8 comparisons: none")
    # runs that test apart otherwise, two rungs above 0 or none at 0, are refused as before
    first, (root, b, held) = runs
    for rungs in ((0, 2, 4), (2,)):
        with pytest.raises(ValueError, match="differ in \\['rungs'\\]"):
            claim.pooled([first, (root, dataclasses.replace(b, rungs=rungs), held)])


def test_a_pool_whose_runs_test_alike_reads_their_records_as_they_are():
    _, ((root, a, held), (other, b, theirs)) = _family_7()
    alike = dataclasses.replace(b, rungs=(0, 1))
    at_1 = [dataclasses.replace(r, k=1) if r.k == 2 else r for r in theirs]
    assert claim.tested_rungs([a, alike]) == ()
    ((_, joined),) = claim.pooled([(root, a, held), (other, alike, at_1)])
    assert joined == held + at_1


# ------------------------------------------------------------------------ roots beside a run


def test_robyn_beside_a_run_is_read_as_a_capped_tool_of_the_run_is(tmp_path):
    run = _run(1, ("walk", "trend"), (0, 4))
    records = _worlds(run, MAIN_ARMS)
    side = _beside(run)
    theirs = _side(records, side, (*CHC, "robyn"))
    _write(tmp_path / "f1", run, records)
    _write(tmp_path / "f1-robyn", side, theirs)
    (read,) = claim.read_runs([tmp_path / "f1", tmp_path / "f1-robyn"])
    # one run holding Robyn as its own tool, capped at 4 and exploratory
    capped = dataclasses.replace(
        run, external=(*run.external, "robyn"), caps={"robyn": 4}, exploratory_caps=True
    )
    robyn = {(r.environment, r.seed, r.k): r.arms["robyn"] for r in theirs}
    comparisons = read.comparisons()
    assert comparisons == claim.compare(_with(records, "robyn", robyn), capped)
    assert [(c.rival, c.cap, c.pairs) for c in comparisons if c.exploratory] == [
        ("robyn", 4, 8)
    ] * 2
    # the arm both roots plan is the run's own
    assert [r.arms["chc"] for r in read.records] == [r.arms["chc"] for r in records]
    printed = claim.lines(comparisons, 0.05)
    assert sum("less robyn" in line and "on its cap, 4 worlds" in line for line in printed) == 2
    assert printed[-2] == "the claim holds: 8 of 8 comparisons within 0.05"


def test_meridian_beside_family_10_is_read_on_its_root_s_environments_alone(tmp_path):
    run = _run(10, (*GEO, "geos-50"), (0, 4), external=("pymc-marketing", "orbit"))
    records = _worlds(run, (DEFAULT, *CHC, "pymc-marketing", "orbit"))
    side = _beside(run, environments=GEO, worlds=8, external=("meridian",), caps={"meridian": 8})
    theirs = _side(records, side, (*CHC, "meridian"))
    failed = theirs[0]
    assert (failed.environment, failed.index, failed.k) == ("geos-3", 0, 0)
    raised = dataclasses.replace(failed.arms["meridian"], failure="raised")
    theirs[0] = dataclasses.replace(failed, arms={**failed.arms, "meridian": raised})
    _write(tmp_path / "f10", run, records)
    _write(tmp_path / "f10-meridian", side, theirs)
    (read,) = claim.read_runs([tmp_path / "f10", tmp_path / "f10-meridian"])
    comparisons = read.comparisons()
    assert [c for c in comparisons if c.rival != "meridian"] == claim.compare(records, run)
    # Meridian on geos-3 and geos-8 as a run of those two holding it would read it
    apart = dataclasses.replace(
        run,
        environments=GEO,
        external=(*run.external, "meridian"),
        caps={"meridian": 8},
        exploratory_caps=True,
    )
    meridian = {(r.environment, r.seed, r.k): r.arms["meridian"] for r in theirs}
    alone = claim.compare(_with(records, "meridian", meridian), apart)
    beside = [c for c in comparisons if c.rival == "meridian"]
    assert _bare(beside) == [c for c in alone if c.rival == "meridian"]
    assert [(c.environments, c.exploratory) for c in beside] == [(GEO, True)] * 2
    # geos-50, where Meridian's root plans nothing, holds none of its failures
    assert [(c.failures, c.pairs, c.left_out) for c in beside] == [(1, 15, ()), (0, 16, ())]
    printed = "\n".join(claim.lines(comparisons, 0.05))
    assert printed.count("; on geos-3 and geos-8 only; on its cap, 8 worlds an environment") == 2
    assert "the claim holds: 8 of 8 comparisons within 0.05" in printed


def test_bayes_geo_beside_family_10_is_joined_and_compared_with_nothing(tmp_path):
    run = _run(10, (*GEO, "geos-50"), (0, 4), external=("pymc-marketing", "orbit"))
    records = _worlds(run, (DEFAULT, *CHC, "pymc-marketing", "orbit"))
    side = dataclasses.replace(
        _beside(run, environments=GEO, worlds=8, caps=None, exploratory_caps=False),
        arms=("bayes-geo",),
        external=(),
    )
    _write(tmp_path / "f10", run, records)
    _write(tmp_path / "f10-bayes-geo", side, _side(records, side, BAYES_GEO))
    (read,) = claim.read_runs([tmp_path / "f10", tmp_path / "f10-bayes-geo"])
    held = [(r.environment, set(BAYES_GEO) <= r.arms.keys()) for r in read.records]
    assert held == [(r.environment, r.environment in GEO) for r in records]
    assert read.comparisons() == claim.compare(records, run)


def test_a_world_beside_the_run_not_held_yet_leaves_its_comparison_unread(tmp_path):
    run = _run(1, ("walk", "trend"), (0, 4))
    records = _worlds(run, MAIN_ARMS)
    side = _beside(run)
    theirs = _side(records, side, (*CHC, "robyn"))
    _write(tmp_path / "f1", run, records)
    _write(tmp_path / "f1-robyn", side, [r for r in theirs if (r.index, r.k) != (3, 4)])
    (read,) = claim.read_runs([tmp_path / "f1", tmp_path / "f1-robyn"])
    robyn = [c for c in read.comparisons() if c.rival == "robyn"]
    assert [(c.k, c.within is not None) for c in robyn] == [(0, True), (4, False)]
    assert "unread: the run does not hold its worlds whole" in claim.lines(robyn, 0.05)[1]


def _roots(tmp_path, design):
    """A run and a root beside it, declared as `_beside` declares one but for `design`."""
    run = _run(1, ("walk", "trend"), (0, 4))
    _write(tmp_path / "f1", run, _worlds(run, MAIN_ARMS))
    _write(tmp_path / "f1-robyn", _beside(run, **design), [])
    return [tmp_path / "f1", tmp_path / "f1-robyn"]


@pytest.mark.parametrize(
    "design",
    [
        {"rungs": (0,)},
        {"bench": "0000000"},
        {"environments": ("walk", "step")},
        {"worlds": 3, "caps": None, "exploratory_caps": False},
        {
            "worlds": 3,
            "arms": ("bayes-geo",),
            "external": (),
            "caps": None,
            "exploratory_caps": False,
        },
        {"external": (), "caps": None, "exploratory_caps": False},
    ],
    ids=[
        "a rung short",
        "another bench",
        "another environment",
        "short uncapped",
        "an arm of its own short",
        "no arm",
    ],
)
def test_a_root_beside_a_run_declared_otherwise_is_refused(tmp_path, design):
    with pytest.raises(ValueError, match="is not a root beside"):
        claim.read_runs(_roots(tmp_path, design))


def test_a_root_beside_a_run_on_fewer_worlds_reads_its_tools_on_their_caps(tmp_path):
    design = {"worlds": 3, "caps": {"robyn": 3}, "exploratory_caps": False}
    (read,) = claim.read_runs(_roots(tmp_path, design))
    robyn = [c for c in read.comparisons() if c.rival == "robyn"]
    assert [(c.cap, c.exploratory) for c in robyn] == [(3, False), (3, False)]


def test_a_root_beside_a_run_under_another_pre_registration_is_refused_naming_both(tmp_path):
    first, second = tmp_path / "first.md", tmp_path / "second.md"
    first.write_text("one design\n")
    second.write_text("another design\n")
    one, other = stamp(first), stamp(second)
    run = _run(1, ("walk", "trend"), (0, 4), preregistration=one)
    _write(tmp_path / "f1", run, _worlds(run, MAIN_ARMS))
    _write(tmp_path / "f1-robyn", _beside(run, preregistration=other), [])
    with pytest.raises(ValueError, match="runs under the pre-registration") as refused:
        claim.read_runs([tmp_path / "f1", tmp_path / "f1-robyn"])
    assert f"{tmp_path / 'f1-robyn'} runs under the pre-registration {other.path}" in str(
        refused.value
    )
    assert f"{tmp_path / 'f1'} under {one.path}" in str(refused.value)


def test_a_root_without_a_margin_beside_none_of_the_runs_handed_is_refused(tmp_path):
    run = _run(1, ("walk", "trend"), (0, 4))
    _write(tmp_path / "f1", run, _worlds(run, MAIN_ARMS))
    _write(tmp_path / "f10-robyn", _beside(run), [])
    with pytest.raises(ValueError, match="beside none of the runs handed"):
        claim.read_runs([tmp_path / "f1", tmp_path / "f10-robyn"])


def test_the_roots_beside_a_pool_s_runs_are_read_with_it_and_handed_whole(tmp_path):
    (whole, records), runs = _family_7()
    robyn = {}
    for root, d, held in runs:
        _write(tmp_path / root, d, held)
        side = _beside(d)
        theirs = _side(held, side, (*CHC, "robyn"))
        _write(tmp_path / f"{root}-robyn", side, theirs)
        # the pool reads L-78's and L-104's rung 2 as L-52's rung 1
        robyn |= {(r.environment, r.seed, min(r.k, 1)): r.arms["robyn"] for r in theirs}
    roots = [tmp_path / name for name in ("f7a", "f7b", "f7a-robyn", "f7b-robyn")]
    (read,) = claim.read_runs(roots)
    assert read.tested == TESTED
    capped = dataclasses.replace(
        whole, external=(*whole.external, "robyn"), caps={"robyn": 4}, exploratory_caps=True
    )
    comparisons = read.comparisons()
    assert _bare(comparisons) == claim.compare(_with(records, "robyn", robyn), capped)
    assert [c.tested for c in comparisons if c.rival == "robyn"] == [(), TESTED]
    with pytest.raises(ValueError, match="hand the claim \\['f7b-robyn'\\] too"):
        claim.read_runs(roots[:3])


def test_roots_of_one_name_beside_a_pool_s_runs_that_plan_apart_are_refused(tmp_path):
    _, runs = _family_7()
    for (root, d, held), arms in zip(runs, (("chc",), ("chc", "ridge")), strict=True):
        _write(tmp_path / root, d, held)
        _write(tmp_path / f"{root}-robyn", _beside(d, arms=arms, worlds=8), [])
    roots = [tmp_path / name for name in ("f7a", "f7b", "f7a-robyn", "f7b-robyn")]
    with pytest.raises(ValueError, match="they differ in \\['arms'\\]"):
        claim.read_runs(roots)


def test_the_runs_of_a_pool_that_read_their_default_against_apart_are_refused(tmp_path):
    _, runs = _family_7()
    for (root, d, held), against in zip(runs, (("dlm mixture",), ()), strict=True):
        _write(tmp_path / root, dataclasses.replace(d, default_less=against), held)
    with pytest.raises(ValueError, match="they differ in \\['default_less'\\]"):
        claim.read_runs([tmp_path / "f7a", tmp_path / "f7b"])


# ------------------------------------------------------------------------ reported regardless


def _by_hand(records, arm, rival, k, environments, worlds):
    """The worlds `arm` and `rival` both planned at rung `k`, the first `worlds` of each of
    `environments`: their count, and the mean of `arm`'s regret less `rival`'s with its 95 %
    Student-t interval."""
    x = np.array(
        [
            r.arms[arm].regret - r.arms[rival].regret
            for r in records
            if r.k == k
            and r.environment in environments
            and r.index < worlds
            and r.arms[arm].failure is None
            and r.arms[rival].failure is None
        ]
    )
    half = float(student.ppf(0.975, x.size - 1)) * x.std(ddof=1) / np.sqrt(x.size)
    mean = float(x.mean())
    return (
        x.size,
        pytest.approx(mean, rel=1e-12, abs=0.0),
        pytest.approx((mean - half, mean + half), rel=1e-12, abs=0.0),
    )


def test_every_recorded_arm_is_read_less_every_rival_outside_the_claim(tmp_path):
    run = _run(1, ("walk", "trend"), (0, 4))
    records = _worlds(run, (*DLM, *CHC, "meridian", "orbit"))
    failed = records[0]
    assert (failed.environment, failed.index, failed.k) == ("walk", 0, 0)
    raised = dataclasses.replace(failed.arms["dlm mixture"], failure="raised")
    records[0] = dataclasses.replace(failed, arms={**failed.arms, "dlm mixture": raised})
    side = _beside(run)
    _write(tmp_path / "f1", run, records)
    _write(tmp_path / "f1-robyn", side, _side(records, side, (*CHC, "robyn")))
    (read,) = claim.read_runs([tmp_path / "f1", tmp_path / "f1-robyn"])
    readings = claim.recorded(read)
    # each of the engine's arms but the default, at each rung, less the run's rivals and Robyn
    arms = DLM[1:] + CHC
    rivals = (*BUILT_IN, "meridian", "orbit", "robyn")
    assert [(r.k, r.arm, r.rival) for r in readings] == [
        (k, arm, rival) for k in (0, 4) for arm in arms for rival in rivals
    ]
    mixture = {(r.k, r.rival): r for r in readings if r.arm == "dlm mixture"}
    one, capped = mixture[0, "meridian"], mixture[0, "robyn"]
    assert (one.pairs, one.mean, one.interval) == _by_hand(
        records, "dlm mixture", "meridian", 0, run.environments, 8
    )
    assert (capped.pairs, capped.mean, capped.interval) == _by_hand(
        read.records, "dlm mixture", "robyn", 0, run.environments, 4
    )
    assert [(r.pairs, r.arm_failures, r.failures, r.cap) for r in (one, capped)] == [
        (15, 1, 0, None),
        (7, 1, 0, 4),
    ]
    assert all(r.pairs == 16 for r in readings if r.k == 4 and r.rival != "robyn")
    printed = claim.recorded_lines(readings)
    assert printed[0].startswith("reported regardless, outside the claim and Holm's list:")
    assert one.interval is not None
    low, high = one.interval
    assert (
        f"  family 1, k = 0, {'dlm mixture less meridian':46s} {one.mean:+.4f} "
        f"[{low:+.4f}, {high:+.4f}]; {one.reading}, unadjusted; 15 complete cases, the rival "
        "failed 0, the arm 1"
    ) in printed
    assert sum(line.endswith("; on its cap, 4 worlds an environment") for line in printed) == 10
    # the claim reads its default as it did
    assert read.comparisons() == claim.compare(read.records, read.declaration, beside=read.beside)


def test_bayes_geo_beside_family_10_is_read_less_every_rival_on_its_root_s_environments(tmp_path):
    run = _run(10, (*GEO, "geos-50"), (0, 4), external=("pymc-marketing", "orbit"))
    records = _worlds(run, (*DLM, *CHC, "pymc-marketing", "orbit"))
    meridian = _beside(
        run, environments=GEO, worlds=8, external=("meridian",), caps={"meridian": 8}
    )
    geo = dataclasses.replace(
        _beside(run, environments=GEO, worlds=8, caps=None, exploratory_caps=False),
        arms=("bayes-geo",),
        external=(),
    )
    _write(tmp_path / "f10", run, records)
    _write(tmp_path / "f10-meridian", meridian, _side(records, meridian, (*CHC, "meridian")))
    _write(tmp_path / "f10-bayes-geo", geo, _side(records, geo, BAYES_GEO))
    roots = [tmp_path / name for name in ("f10", "f10-meridian", "f10-bayes-geo")]
    (read,) = claim.read_runs(roots)
    readings = claim.recorded(read)
    rivals = (*BUILT_IN, "pymc-marketing", "orbit", "meridian")
    plug_in = [r for r in readings if r.arm == "bayes geo plug-in"]
    assert [(r.k, r.rival, r.environments, r.pairs) for r in plug_in] == [
        (k, rival, GEO, 16) for k in (0, 4) for rival in rivals
    ]
    one = plug_in[-1]
    assert (one.k, one.rival, one.cap) == (4, "meridian", 8)
    assert (one.pairs, one.mean, one.interval) == _by_hand(
        read.records, "bayes geo plug-in", "meridian", 4, GEO, 8
    )
    # the run's own arms less Meridian on its root's environments, less the rest on all three
    chc = [(r.rival, r.environments, r.pairs) for r in readings if (r.k, r.arm) == (4, "chc")]
    assert chc == [*((rival, (), 24) for rival in rivals[:-1]), ("meridian", GEO, 16)]
    printed = claim.recorded_lines(readings)
    on = "; on geos-3 and geos-8 only; on its cap, 8 worlds an environment"
    assert sum("bayes geo plug-in less meridian" in line and on in line for line in printed) == 2


def test_the_default_above_0_is_read_less_the_static_arm_on_the_joined_records(tmp_path):
    tiers = {"default_above_0": TWO_STAGE, "joined": "-ts"}
    run = _run(1, ("walk", "trend"), (0, 4), arms=("bayes", "dlm"), **tiers)
    records = _worlds(run, (*BAYES, *DLM, "meridian", "orbit"))
    ts = dataclasses.replace(
        run,
        rungs=(4,),
        arms=("dlm-two-stage",),
        external=(),
        default=None,
        margin=None,
        reader="student",
        default_above_0=None,
        joined=None,
    )
    theirs = _side(records, ts, (TWO_STAGE,))
    _write(tmp_path / "f1", run, records)
    _write(tmp_path / "f1-ts", ts, theirs)
    (read,) = claim.read_runs([tmp_path / "f1", tmp_path / "f1-ts"])
    readings = claim.recorded(read)
    (second,) = [r for r in readings if r.arm == TWO_STAGE]
    assert (second.k, second.rival) == (4, "stacked")
    # `dlm two-stage` from the joined root, `stacked` from the run's
    two_stage = {(r.environment, r.seed): r.arms[TWO_STAGE].regret for r in theirs}
    x = np.array(
        [two_stage[r.environment, r.seed] - r.arms["stacked"].regret for r in records if r.k == 4]
    )
    half = float(student.ppf(0.975, x.size - 1)) * x.std(ddof=1) / np.sqrt(x.size)
    assert (second.pairs, second.mean) == (16, pytest.approx(float(x.mean()), rel=1e-12, abs=0.0))
    assert second.interval == pytest.approx((x.mean() - half, x.mean() + half), rel=1e-12, abs=0.0)
    # `dlm plug-in` is a recorded arm at the tested rung, not at 0
    assert {r.arm for r in readings if r.k == 0} == {*BAYES, *DLM} - {DEFAULT}
    assert {r.arm for r in readings if r.k == 4} == {*BAYES, *DLM, TWO_STAGE}
    # a world the joined root does not hold yet leaves that reading unread, and no other
    _write(tmp_path / "f1-ts", ts, theirs[:-1])
    (read,) = claim.read_runs([tmp_path / "f1", tmp_path / "f1-ts"])
    readings = claim.recorded(read)
    assert [r.pairs for r in readings if r.arm == TWO_STAGE] == [None]
    assert all(r.pairs == 16 for r in readings if r.arm != TWO_STAGE)
    last = claim.recorded_lines(readings)[-1]
    assert last.startswith("  family 1, k = 4, dlm two-stage less stacked")
    assert last.endswith(" unread: the run does not hold its worlds whole")


# -------------------------------------------------------------- claim read over M2's roots


def _m2(tmp_path):
    """Roots laid out as M2's: family 0 as `f0` and `f0ref`, family 7 as `f7a` and `f7b`, each
    pair one pool, and family 10; beside each main root a `-ts` root and a `-robyn` root, and
    beside family 10 Meridian's and `bayes-geo`'s roots on `geos-3` and `geos-8`. Each main root
    plans `bayes`, `dlm` and `chc`, every variant scored. Every root but the main ones declares no
    margin; Robyn's and Meridian's declare their caps exploratory."""
    held = {"arms": ("bayes", "dlm", "chc"), "default_above_0": TWO_STAGE, "joined": "-ts"}
    mains = {
        "f0": _run(0, ("drawn",), (0, 4), pool=("f0", "f0ref"), **held),
        "f0ref": _run(0, ("reference",), (0, 4), pool=("f0", "f0ref"), **held),
        "f7a": _run(7, ("L-52",), (0, 1), pool=POOL, **held),
        "f7b": _run(7, ("L-78", "L-104"), (0, 2), pool=POOL, **held),
        "f10": _run(10, (*GEO, "geos-50"), (0, 4), external=("pymc-marketing", "orbit"), **held),
    }
    roots = []
    for name, d in mains.items():
        records = _worlds(d, (*BAYES, *DLM, *CHC, *d.external))
        roots.append(_write(tmp_path / name, d, records))
        ts = dataclasses.replace(
            d,
            rungs=tuple(k for k in d.rungs if k > 0),
            arms=("dlm-two-stage",),
            external=(),
            default=None,
            margin=None,
            reader="student",
            default_above_0=None,
            joined=None,
            pool=(),
        )
        roots.append(_write(tmp_path / f"{name}-ts", ts, _side(records, ts, (TWO_STAGE,))))
        robyn = _beside(d)
        roots.append(
            _write(tmp_path / f"{name}-robyn", robyn, _side(records, robyn, (*CHC, "robyn")))
        )
        if name == "f10":
            meridian = _beside(
                d, environments=GEO, worlds=8, external=("meridian",), caps={"meridian": 8}
            )
            geo = dataclasses.replace(
                _beside(d, environments=GEO, worlds=8, caps=None, exploratory_caps=False),
                arms=("bayes-geo",),
                external=(),
            )
            theirs = _side(records, meridian, (*CHC, "meridian"))
            roots.append(_write(tmp_path / "f10-meridian", meridian, theirs))
            roots.append(_write(tmp_path / "f10-bayes-geo", geo, _side(records, geo, BAYES_GEO)))
    return roots


def _read(roots, capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["claim", "read", *map(str, roots)])
    claim.main()
    return capsys.readouterr().out.splitlines()


def test_claim_read_over_every_root_of_a_run_laid_out_as_m2_s(tmp_path, capsys, monkeypatch):
    roots = _m2(tmp_path)
    out = _read(roots, capsys, monkeypatch)
    family_7 = [line for line in out if line.startswith("  family 7, k = 1,")]
    assert len(family_7) == 5
    assert all(f"; {TESTED_LINE}" in line for line in family_7)
    robyn = [line for line in out if "less robyn" in line]
    assert len(robyn) == 6
    assert all("; on its cap, 4 worlds an environment" in line for line in robyn)
    assert all(line.endswith("; exploratory") for line in robyn)
    meridian = [line for line in out if line.startswith("  family 10,") and "less meridian" in line]
    assert len(meridian) == 2
    assert all("; on geos-3 and geos-8 only; on its cap, 8 worlds" in line for line in meridian)
    assert not any("bayes geo" in line for line in out)
    # three families' claims at two rungs against the four rivals of their main roots
    assert out[-2] == "the claim holds: 24 of 24 comparisons within 0.05"
    assert out[-1] == "losses by Holm's step-down at familywise 0.025 over 24 comparisons: none"
    # the main roots alone read as they did; the roots beside them add their own comparisons
    mains = [r for r in roots if r.name in ("f0", "f0ref", "f7a", "f7b", "f10")]
    beside = {*robyn, *meridian}
    assert _read(mains, capsys, monkeypatch) == [line for line in out if line not in beside]


def test_claim_read_recorded_prints_what_the_claim_does_not_read_after_it(
    tmp_path, capsys, monkeypatch
):
    roots = _m2(tmp_path)
    claimed = _read(roots, capsys, monkeypatch)
    out = _read(["--recorded", *roots], capsys, monkeypatch)
    assert out[: len(claimed) + 1] == [*claimed, ""]
    head, *readings = out[len(claimed) + 1 :]
    assert head.startswith("reported regardless, outside the claim and Holm's list:")
    # f10: 10 arms at k = 0 and 11 at k = 4 less 6 rivals; f0 and f7: 8 and 9 arms less 5; and
    # the default above 0 less `stacked` once a family
    assert len(readings) == 10 * 6 + 11 * 6 + 1 + 2 * (8 * 5 + 9 * 5 + 1)
    assert not any("unread" in line for line in readings)
    family_7 = [line for line in readings if line.startswith("  family 7, k = 1,")]
    assert len(family_7) == 9 * 5 + 1
    assert all(line.endswith(f"; {TESTED_LINE}") for line in family_7)
    geo = [line for line in readings if "bayes geo plug-in less meridian" in line]
    on = "; on geos-3 and geos-8 only; on its cap, 8 worlds an environment"
    assert [line[:19] for line in geo] == ["  family 10, k = 0,", "  family 10, k = 4,"]
    assert all(line.endswith(on) for line in geo)
    second = [line for line in readings if "dlm two-stage less stacked" in line]
    assert [line.split(", dlm two-stage")[0] for line in second] == [
        "  family 10, k = 4",
        "  family 0, k = 4",
        "  family 7, k = 1",
    ]
