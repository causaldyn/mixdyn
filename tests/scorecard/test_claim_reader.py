"""The claim as its pre-registration states it: read by Student's t or by Hall's transform, its
default procedure per rung with the two-stage arm's records joined from their own root, and one
family read over several runs, pooled. A declaration that names none of these reads, writes and
prints as it did before they existed, to the byte."""

import dataclasses
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("causaldyn_bench")

from causaldyn_bench.scorecard.gate import ALPHA
from scipy.stats import norm
from scipy.stats import t as t_dist
from test_claim import (
    DEFAULT,
    RIVALS,
    SCORED,
    _differences,
    _failing,
    _known,
    _records,
)

from scorecard import claim, report
from scorecard.layout import Layout, stamp

OLD_FIELDS = (
    "family",
    "k",
    "rival",
    "exploratory",
    "mean",
    "interval",
    "reading",
    "within",
    "level",
    "cap",
    "pairs",
    "failures",
    "default_failures",
    "left_out",
    "compared",
    "fallback_mean",
    "fallback_interval",
    "fallback_within",
)
"""A comparison's fields before the reader, the per-rung default and the pool."""


def _cases():
    """Runs whose readings cover each path of `compare`: known verdicts, Holm, failures, an
    environment and a rival left out, caps, an exploratory run and a run not yet whole."""
    means, spreads = _known()
    known = _records(_differences(means, spreads))
    within = _records(_differences(dict.fromkeys(RIVALS, -0.1), dict.fromkeys(RIVALS, 0.1)))
    failed = _failing(within, "orbit", lambda r: r.environment == "walk" and r.index <= 10)
    failed = _failing(failed, DEFAULT, lambda r: (r.environment, r.index) == ("trend", 2))
    everywhere = _failing(within, "meridian", lambda r: r.index < 15)
    capped = dataclasses.replace(SCORED, caps={"orbit": 8}, exploratory_caps=True)
    short = [r for r in within if not (r.environment == "trend" and r.index == 19)]
    return [
        (known, SCORED),
        (within, SCORED),
        (failed, SCORED),
        (everywhere, SCORED),
        (known, capped),
        (known, dataclasses.replace(SCORED, caps={"meridian": 5})),
        (known, dataclasses.replace(SCORED, exploratory=True)),
        (short, SCORED),
    ]


def _digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _readings(cases):
    comparisons = [c for records, d in cases for c in claim.compare(records, d)]
    fields = [[repr(getattr(c, name)) for name in OLD_FIELDS] for c in comparisons]
    return json.dumps(fields), "\n".join(claim.lines(comparisons, 0.05))


DIGESTS = {
    "fields": "2c8b139f8d19b255badcf39ce41b038d7399e1156be98ab1455cb43b2d1cb9f7",
    "lines": "cf831019e43c072a3498954f19ad987a2d6df78fa258ecd5ffc5ef3b3160bcd3",
    "report": "37ddaf6e5729b5325cc2fa89fcd0f51cee50de4b10f22bb89125971edb5707ec",
    "declarations": "b3f0619cd2903b8ad126f1b94b791f8a4f0198cf5a040b9fec879953374d899b",
}
"""The readings of `_cases` by the code before the reader, the per-rung default and the pool. The
report prints the bench's notes on the family, so its digest moves with them: family 1's note on
its cpm environment moved it."""


def test_a_run_that_declares_none_of_the_new_fields_reads_as_it_did_to_the_byte(tmp_path, capsys):
    fields, lines = _readings(_cases())
    means, spreads = _known()
    layout = Layout(tmp_path / "run")
    layout.declare(SCORED)
    records = _records(_differences(means, spreads))
    layout.scores.write_text("".join(json.dumps(r.as_json()) + "\n" for r in records))
    report.main(layout, ["dlm plug-in", "dlm mixture"])
    printed = capsys.readouterr().out.replace(str(tmp_path), "<tmp>")
    written = []
    for _, d in _cases():
        held = Layout(tmp_path / f"declared-{len(written)}")
        held.declare(d)
        written.append(held.declaration.read_text())
    assert {
        "fields": _digest(fields),
        "lines": _digest(lines),
        "report": _digest(printed),
        "declarations": _digest("\n".join(written)),
    } == DIGESTS


# ------------------------------------------------------------------------------ Hall's reader

MAXIMA_HALL = (
    (
        0.05,
        (-0.03, -0.01, 0.0, 0.02, 0.07, 0.01),
        2.141029229725695030442153012066828961928,
        0.9115458350499376962129045191793567815071,
    ),
    (
        0.1,
        (0.1, 0.1, 0.1, 0.1, -0.9, 0.2, 0.0, -0.1),
        1.822782234668904436290619379188351798147,
        -0.5889002444942942232233476769269137986556,
    ),
    (
        0.02,
        (-0.04, 0.0, 0.02, 0.02, 0.06, 0.18, -0.08, 0.0, 0.04, -0.1),
        0.3748977938795043529627313869232047814629,
        0.4750978966780437953743967406757503920029,
    ),
)
"""Each sample's margin, its paired differences D, the default less the rival, and Hall's g(T)
of the margin less D and of D, exactly and to 40 digits by Maxima 5.46
(validation/hall_reader.mac, STEP 4)."""


def _sample(differences, **change):
    """Every rival's differences `differences`, half in each law, on a run read by `change`."""
    worlds = len(differences) // 2
    declared = dataclasses.replace(SCORED, worlds=worlds, **change)
    return _records(dict.fromkeys(RIVALS, differences), worlds=worlds), declared


@pytest.mark.parametrize(("margin", "differences", "within", "loss"), MAXIMA_HALL)
def test_hall_s_reader_reads_maxima_s_cubic_and_its_losses_at_its_tail(
    margin, differences, within, loss
):
    records, declared = _sample(differences, margin=margin, reader="hall")
    comparisons = claim.compare(records, declared)
    assert [c.rival for c in comparisons] == list(RIVALS)
    for c in comparisons:
        assert c.reader == "hall"
        assert c.statistic == pytest.approx(within, rel=1e-12, abs=0.0)
        assert c.within is bool(within >= norm.isf(ALPHA))
        tail = float(norm.sf(loss))
        assert c.level == (1.0 if tail >= 0.5 else pytest.approx(tail, rel=1e-9, abs=0.0))
    # Student's reading of the same worlds is another statistic
    (student,) = (
        c for c in claim.compare(*_sample(differences, margin=margin)) if c.rival == "orbit"
    )
    n = len(differences)
    se = float(np.std(differences, ddof=1)) / math.sqrt(n)
    assert student.statistic == pytest.approx(
        (margin - float(np.mean(differences))) / se, rel=1e-12, abs=0.0
    )
    assert student.statistic != pytest.approx(within, rel=1e-3, abs=0.0)


def test_hall_s_reader_and_student_s_part_on_a_skewed_comparison():
    """One world of 20 where the default is 1 a euro worse, the margin less each difference
    skewed to the left: Student's t reads 2.2 and shows the claim, Hall's g(T) does not."""
    differences = (0.0,) * 19 + (1.0,)
    margin = 0.05 + 2.2 * float(np.std(differences, ddof=1)) / math.sqrt(20)
    student = claim.compare(*_sample(differences, margin=margin))[0]
    hall = claim.compare(*_sample(differences, margin=margin, reader="hall"))[0]
    assert student.statistic is not None
    assert hall.statistic is not None
    assert student.statistic == pytest.approx(2.2, rel=1e-12, abs=0.0)
    assert student.statistic >= float(t_dist.isf(ALPHA, len(differences) - 1))
    assert hall.statistic < float(norm.isf(ALPHA))
    assert (student.within, hall.within) == (True, False)


def test_hall_s_reader_is_printed_beside_each_comparison_and_in_the_report(tmp_path, capsys):
    records, declared = _sample(MAXIMA_HALL[0][1], margin=0.05, reader="hall")
    printed = "\n".join(claim.lines(claim.compare(records, declared), 0.05))
    assert printed.count("by Hall's reader, the interval Student's") == len(RIVALS)
    layout = Layout(tmp_path)
    layout.declare(declared)
    layout.scores.write_text("".join(json.dumps(r.as_json()) + "\n" for r in records))
    report.main(layout, [DEFAULT])
    assert "dlm plug-in less every rival, by Hall's reader:" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"reader": "bootstrap"}, "reads a look by one of"),
        ({"reader": "hall", "default": None, "margin": None}, "declare its margin"),
    ],
)
def test_a_reader_the_claim_cannot_read_is_refused(change, match):
    with pytest.raises(ValueError, match=match):
        dataclasses.replace(SCORED, **change)


def test_the_declared_reader_is_written_and_read_back(tmp_path):
    hall = dataclasses.replace(SCORED, reader="hall")
    Layout(tmp_path / "a").declare(hall)
    assert json.loads((tmp_path / "a" / "run.json").read_text())["reader"] == "hall"
    assert Layout(tmp_path / "a").read() == hall
    Layout(tmp_path / "b").declare(SCORED)
    assert "reader" not in json.loads((tmp_path / "b" / "run.json").read_text())
    assert Layout(tmp_path / "b").read().reader == "student"


@pytest.mark.skipif(shutil.which("maxima") is None, reason="maxima is not on PATH")
def test_maxima_s_check_of_hall_s_reader_passes_every_step():
    script = Path(__file__).parents[2] / "validation" / "hall_reader.mac"
    out = subprocess.run(
        ["maxima", "--very-quiet", "-Q", "-b", str(script)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    ).stdout
    assert "Lisp error" not in out
    assert "(FAIL)" not in out
    assert [f"STEP {k} (PASS)" in out for k in range(1, 6)] == [True] * 5


# ------------------------------------------------------------- the default per rung, joined

TWO_STAGE = "dlm two-stage"
TIERED = dataclasses.replace(
    SCORED, rungs=(0, 4), default_above_0=TWO_STAGE, joined="-ts", external=()
)
JOINED = dataclasses.replace(
    SCORED,
    rungs=(4,),
    arms=("dlm-two-stage",),
    external=(),
    default=None,
    margin=None,
)
"""A run whose claim reads `dlm plug-in` at k = 0 and `dlm two-stage` above it from the root
beside it, and that root: the two-stage arm alone at the rung above 0."""
LOCAL = ("status quo", "equal split")


def _tiered(within, losing):
    """The main root's records at k = 0 and 4, the default within the margin of every rival, and
    the joined root's at k = 4, the two-stage arm 0.2 worse than every rival."""
    spread = dict.fromkeys(LOCAL, 0.1)
    main = [
        *_records(_differences(dict.fromkeys(LOCAL, within), spread), k=0),
        *_records(_differences(dict.fromkeys(LOCAL, within), spread), k=4),
    ]
    two = _differences(dict.fromkeys(LOCAL, losing), spread, seed=8)
    joined = []
    for r in _records(two, k=4):
        # the joined root scores the same worlds: its built-in arms are the main root's
        twin = next(m for m in main if (m.environment, m.seed, m.k) == (r.environment, r.seed, 4))
        arms = {name: twin.arms[name] for name in LOCAL}
        arms[TWO_STAGE] = dataclasses.replace(
            r.arms[DEFAULT], regret=twin.arms["status quo"].regret + losing
        )
        joined.append(dataclasses.replace(r, arms=arms))
    return main, joined


def _write(layout, declaration, records):
    layout.declare(declaration)
    layout.scores.write_text("".join(json.dumps(r.as_json()) + "\n" for r in records))


def test_the_default_above_0_is_read_at_the_rungs_above_0_from_the_joined_root(tmp_path):
    main, joined = _tiered(-0.1, 0.2)
    _write(Layout(tmp_path / "f1"), TIERED, main)
    _write(Layout(tmp_path / "f1-ts"), JOINED, joined)
    declared, records = claim.read_run(tmp_path / "f1")
    assert declared == TIERED
    comparisons = claim.compare(records, declared)
    by = {(c.k, c.rival): c for c in comparisons}
    assert {key: c.default for key, c in by.items()} == {
        (0, "status quo"): DEFAULT,
        (0, "equal split"): DEFAULT,
        (4, "status quo"): TWO_STAGE,
        (4, "equal split"): TWO_STAGE,
    }
    assert all(by[0, rival].within for rival in LOCAL)
    for rival in LOCAL:
        assert by[4, rival].mean == pytest.approx(0.2, rel=0.0, abs=1e-12)
        assert not by[4, rival].within
    assert not claim.holds(comparisons)
    printed = "\n".join(claim.lines(comparisons, 0.05))
    assert "family 1, k = 4, dlm two-stage less status quo" in printed
    assert "family 1, k = 0, dlm plug-in less status quo" in printed


def test_a_world_the_joined_root_does_not_hold_yet_leaves_the_rung_unread(tmp_path):
    main, joined = _tiered(-0.1, -0.1)
    _write(Layout(tmp_path / "f1"), TIERED, main)
    _write(Layout(tmp_path / "f1-ts"), JOINED, joined[:-1])
    declared, records = claim.read_run(tmp_path / "f1")
    comparisons = claim.compare(records, declared)
    assert [(c.k, c.within) for c in comparisons] == [(0, True), (0, True), (4, None), (4, None)]
    _write(Layout(tmp_path / "f1-ts"), JOINED, joined)
    declared, records = claim.read_run(tmp_path / "f1")
    comparisons = claim.compare(records, declared)
    assert claim.holds(comparisons)


def test_a_joined_root_under_another_preregistration_is_refused_naming_both(tmp_path):
    main, joined = _tiered(-0.1, 0.2)
    first, second = tmp_path / "first.md", tmp_path / "second.md"
    first.write_text("one design\n")
    second.write_text("another design\n")
    one, other = stamp(first), stamp(second)
    for name, theirs in (("a", other), ("b", None)):
        _write(Layout(tmp_path / name), dataclasses.replace(TIERED, preregistration=one), main)
        _write(
            Layout(tmp_path / f"{name}-ts"),
            dataclasses.replace(JOINED, preregistration=theirs),
            joined,
        )
        with pytest.raises(ValueError, match="runs under the pre-registration") as refused:
            claim.read_run(tmp_path / name)
        named = "none" if theirs is None else f"{theirs.path}, SHA-256 {theirs.sha256}"
        assert f"{tmp_path / name}-ts runs under the pre-registration {named}" in str(refused.value)
        assert f"{tmp_path / name} under {one.path}, SHA-256 {one.sha256}" in str(refused.value)
    _write(Layout(tmp_path / "c"), dataclasses.replace(TIERED, preregistration=one), main)
    _write(Layout(tmp_path / "c-ts"), dataclasses.replace(JOINED, preregistration=one), joined)
    declared, records = claim.read_run(tmp_path / "c")
    assert declared.preregistration == one
    assert len(records) == len(main)


def test_the_command_line_and_the_report_join_the_root_beside_the_run(
    tmp_path, capsys, monkeypatch
):
    main, joined = _tiered(-0.1, 0.2)
    _write(Layout(tmp_path / "f1"), TIERED, main)
    _write(Layout(tmp_path / "f1-ts"), JOINED, joined)
    monkeypatch.setattr(sys, "argv", ["claim", "read", str(tmp_path / "f1")])
    claim.main()
    out = capsys.readouterr().out
    assert "the claim does not hold: 2 of 4 not within" in out
    assert "family 1 k 4 status quo, family 1 k 4 equal split" in out
    report.main(Layout(tmp_path / "f1"), [DEFAULT])
    out = capsys.readouterr().out
    assert "the claim on this run alone, dlm plug-in at k = 0 and dlm two-stage above" in out
    assert "the claim does not hold: 2 of 4 not within" in out


@pytest.mark.parametrize(
    ("joined_change", "match"),
    [
        ({"environments": ("walk",)}, "does not hold 'dlm two-stage'"),
        ({"rungs": (8,)}, "does not hold 'dlm two-stage'"),
        ({"arms": ("dlm",)}, "does not hold 'dlm two-stage'"),
        ({"bench": "0000000"}, "does not hold 'dlm two-stage'"),
    ],
)
def test_a_joined_root_declared_otherwise_than_the_run_is_refused(tmp_path, joined_change, match):
    main, joined = _tiered(-0.1, 0.2)
    _write(Layout(tmp_path / "f1"), TIERED, main)
    _write(Layout(tmp_path / "f1-ts"), dataclasses.replace(JOINED, **joined_change), joined)
    with pytest.raises(ValueError, match=match):
        claim.read_run(tmp_path / "f1")


def test_a_joined_world_scored_apart_or_held_twice_is_refused():
    main, joined = _tiered(-0.1, 0.2)
    shifted = dataclasses.replace(
        joined[0],
        arms={
            **joined[0].arms,
            "status quo": dataclasses.replace(joined[0].arms["status quo"], regret=9.0),
        },
    )
    with pytest.raises(ValueError, match="is not the joined root's"):
        claim.join(main, [shifted, *joined[1:]], TWO_STAGE)
    with pytest.raises(ValueError, match="twice"):
        claim.join(main, [joined[0], *joined], TWO_STAGE)
    with pytest.raises(ValueError, match="scored in both roots"):
        claim.join(claim.join(main, joined, TWO_STAGE), joined, TWO_STAGE)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"default_above_0": TWO_STAGE, "default": None, "margin": None}, "default above 0"),
        ({"default_above_0": "meridian"}, "default above 0"),
        ({"joined": "-ts"}, "joined root"),
        ({"default_above_0": TWO_STAGE, "joined": "ts"}, "joined root"),
        ({"default_above_0": TWO_STAGE, "joined": "-t/s"}, "joined root"),
    ],
)
def test_a_default_above_0_the_claim_cannot_read_is_refused(change, match):
    with pytest.raises(ValueError, match=match):
        dataclasses.replace(SCORED, **change)


def test_the_default_at_a_rung_is_the_one_declared_there(tmp_path):
    assert [TIERED.default_at(k) for k in (0, 1, 4)] == [DEFAULT, TWO_STAGE, TWO_STAGE]
    assert [SCORED.default_at(k) for k in (0, 4)] == [DEFAULT, DEFAULT]
    Layout(tmp_path).declare(TIERED)
    assert Layout(tmp_path).read() == TIERED


# --------------------------------------------------------------------- a family over runs, pooled

POOL = ("f1a", "f1b")


def _split(records, change=None):
    """`records` as two runs of the pool, a law each."""
    out = []
    for law in ("trend", "walk"):
        d = dataclasses.replace(SCORED, environments=(law,), pool=POOL, **(change or {}))
        out.append((d, [r for r in records if r.environment == law]))
    return out


def test_a_pool_reads_its_runs_as_one_run_over_their_environments(tmp_path, capsys, monkeypatch):
    means, spreads = _known()
    records = _records(_differences(means, spreads))
    whole = claim.compare(records, SCORED)
    for name, (d, held) in zip(POOL, _split(records), strict=True):
        _write(Layout(tmp_path / name), d, held)
    runs = [(tmp_path / name, *claim.read_run(tmp_path / name)) for name in POOL]
    ((declared, joined),) = claim.pooled(runs)
    assert declared.environments == ("trend", "walk")
    pooled = claim.compare(joined, declared)
    assert [[getattr(c, f) for f in OLD_FIELDS] for c in pooled] == [
        [getattr(c, f) for f in OLD_FIELDS] for c in whole
    ]
    apart = [c for _, d, held in runs for c in claim.compare(held, dataclasses.replace(d, pool=()))]
    assert len(apart) == 2 * len(whole)
    monkeypatch.setattr(sys, "argv", ["claim", "read", *(str(tmp_path / n) for n in POOL)])
    claim.main()
    out = capsys.readouterr().out
    assert "the claim does not hold: 2 of 4 not within" in out


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"margin": 0.04}, "differ in \\['margin'\\]"),
        ({"reader": "hall"}, "differ in \\['reader'\\]"),
        ({"worlds": 19}, "differ in \\['worlds'\\]"),
        ({"arms": ("dlm", "chc")}, "differ in \\['arms'\\]"),
    ],
)
def test_runs_of_a_pool_that_declare_apart_what_the_claim_reads_are_refused(change, match):
    (first, held), second = _split(_records(_differences(*_known())))
    other = (dataclasses.replace(second[0], **change), second[1])
    with pytest.raises(ValueError, match=match):
        claim.pooled([(Path(POOL[0]), first, held), (Path(POOL[1]), *other)])


def test_runs_of_a_pool_under_two_pre_registrations_are_refused(tmp_path):
    first, second = tmp_path / "first.md", tmp_path / "second.md"
    first.write_text("one design\n")
    second.write_text("another design\n")
    one, other = stamp(first), stamp(second)
    (a, held), (b, theirs) = _split(_records(_differences(*_known())))
    run = (Path(POOL[0]), dataclasses.replace(a, preregistration=one), held)
    for apart in (other, None):
        declared = dataclasses.replace(b, preregistration=apart)
        with pytest.raises(ValueError, match="differ in \\['preregistration'\\]"):
            claim.pooled([run, (Path(POOL[1]), declared, theirs)])
    alike = dataclasses.replace(b, preregistration=one)
    ((declared, records),) = claim.pooled([run, (Path(POOL[1]), alike, theirs)])
    assert (declared.preregistration, records) == (one, held + theirs)


def test_a_pool_is_read_whole_and_each_environment_once():
    (first, held), (second, theirs) = _split(_records(_differences(*_known())))
    with pytest.raises(ValueError, match="read whole"):
        claim.pooled([(Path(POOL[0]), first, held)])
    with pytest.raises(ValueError, match="not one of the runs its pool names"):
        claim.pooled([(Path("f1c"), first, held)])
    twice = dataclasses.replace(second, environments=first.environments)
    with pytest.raises(ValueError, match="each its own"):
        claim.pooled([(Path(POOL[0]), first, held), (Path(POOL[1]), twice, theirs)])


@pytest.mark.parametrize(
    ("pool", "change"),
    [
        (("f1a",), {}),
        (("f1a", "f1a"), {}),
        (("f1a", "a/b"), {}),
        (POOL, {"default": None, "margin": None}),
    ],
)
def test_a_pool_the_claim_cannot_read_is_refused(pool, change):
    with pytest.raises(ValueError, match="a pool names 2 runs or more"):
        dataclasses.replace(SCORED, pool=pool, **change)
