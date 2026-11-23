"""The claim by family and rung (ADR 0081): one intersection-union test a rung of a family over the
rivals it claims, the verdicts tested at 0.025 in the fixed sequence the run declares, the
comparisons predicted to fail read and left out, Hall's word printed where Hall's reader reads the
look, the headline only where every family and rung is claimed whole and holds, and the secondary
readings beside it; a run that does not declare it reads as before."""

import dataclasses
import json
import math

import numpy as np
import pytest

pytest.importorskip("causaldyn_bench")

from causaldyn_bench.scorecard.gate import holm
from scipy.stats import t as student
from test_claim import DEFAULT, RIVALS, SCORED, _differences, _records
from test_claim_roots import _m2, _read

from scorecard import report
from scorecard.claim import ALPHA, Comparison, compare, lines, losses, verdict_lines, verdicts
from scorecard.layout import Layout

BY_RUNG = dataclasses.replace(SCORED, reader="hall", per_rung=True, order=((1, 0),))
M2_ORDER = ((0, 0), (0, 4), (7, 0), (7, 1), (10, 0), (10, 4))
"""The verdicts of the roots laid out as M2's (`_m2`), in a declared order."""


def _comparison(family, k, rival, level, *, claimed=True):
    return Comparison(
        family,
        k,
        rival,
        False,
        mean=0.0,
        interval=(-0.01, 0.01),
        reading="ties",
        within=level <= ALPHA,
        within_level=level,
        claimed=claimed,
    )


COMPARISONS = [
    _comparison(2, 0, "status quo", 1e-4),
    _comparison(2, 0, "meridian", 0.004),
    _comparison(2, 4, "pymc-marketing", 0.02),
    _comparison(3, 0, "orbit", 0.001),
    _comparison(3, 0, "meridian", 0.9, claimed=False),
    _comparison(5, 0, "meridian", 0.3, claimed=False),
]
ORDER = ((3, 0), (2, 0), (2, 4))


def test_a_verdict_is_read_at_its_worst_claimed_level_and_tested_in_the_declared_order():
    """Three verdicts at 0.001, 0.004 and 0.02, each at most 0.025, hold in the order declared; a
    rival predicted to fail does not raise its verdict's level, and a verdict that claims nothing is
    left out of the order, read last and never holds."""
    held = verdicts(COMPARISONS, ORDER)
    assert [(v.family, v.k, v.level, v.holds, v.stopped) for v in held] == [
        (3, 0, 0.001, True, None),
        (2, 0, 0.004, True, None),
        (2, 4, 0.02, True, None),
        (5, 0, 1.0, False, None),
    ]
    assert held[0].rivals == ("orbit",)
    assert held[0].predicted == ("meridian",)
    assert held[3].rivals == ()


def test_the_first_verdict_that_fails_ends_the_sequence_however_strong_the_ones_after_it():
    """A verdict at 0.03 second in the order fails and stops the sequence: the one after it, at
    0.004, is read and not confirmed. In another order the same verdicts hold up to the failure."""
    raised = [*COMPARISONS[:2], _comparison(2, 4, "pymc-marketing", 0.03), *COMPARISONS[3:]]
    held = verdicts(raised, ((3, 0), (2, 4), (2, 0)))
    assert [(v.family, v.k, v.holds, v.stopped) for v in held[:3]] == [
        (3, 0, True, None),
        (2, 4, False, None),
        (2, 0, False, (2, 4)),
    ]
    assert held[2].level == 0.004
    assert held[2].within
    lines = verdict_lines(raised, 0.05, ((3, 0), (2, 4), (2, 0)))
    assert lines[2].startswith("  family 2, k = 4: does not hold (level 0.03;")
    assert lines[2].endswith("; the sequence stops here")
    assert lines[3].startswith(
        "  family 2, k = 0: not confirmed: the sequence stopped at family 2 rung 4 (level 0.004;"
    )
    assert lines[-1].startswith("no headline")
    later = verdicts(raised, ORDER)
    assert [v.holds for v in later[:3]] == [True, True, False]


def test_a_fixed_sequence_holds_verdicts_holm_would_not():
    """Three verdicts at 0.02 each: in sequence each is tested at 0.025 and all hold, where Holm's
    first step, 0.025 / 3, would show none. Every verdict holding at 0.025 says the headline."""
    comparisons = [_comparison(f, 0, "status quo", 0.02) for f in (0, 2, 3)]
    order = ((0, 0), (2, 0), (3, 0))
    held = verdicts(comparisons, order)
    assert all(v.within and v.holds for v in held)
    assert not holm({(v.family, v.k): v.level for v in held})
    assert verdict_lines(comparisons, 0.05, order)[-1].startswith("the headline holds")


def test_a_verdict_the_order_does_not_name_is_refused_and_one_it_names_unread_stops_it():
    with pytest.raises(ValueError, match="does not name family 2 rung 4, which the claim reads"):
        verdicts(COMPARISONS, ((3, 0), (2, 0)))
    held = verdicts(COMPARISONS, ((3, 0), (4, 0), (2, 0), (2, 4)))
    assert [(v.family, v.k, v.holds, v.stopped) for v in held[:4]] == [
        (3, 0, True, None),
        (4, 0, False, None),
        (2, 0, False, (4, 0)),
        (2, 4, False, (4, 0)),
    ]
    lines = verdict_lines(COMPARISONS, 0.05, ((3, 0), (4, 0), (2, 0), (2, 4)))
    assert lines[2] == "  family 4, k = 0: unread; the sequence stops here"


def _skewed():
    """The default less Meridian: 36 worlds at -0.01 and 4 at +0.38, skewed to the right: by
    Student's t the interval holds 0, a tie; by Hall's reader the rival beats the default."""
    differences = _differences(dict.fromkeys(RIVALS, -0.2), dict.fromkeys(RIVALS, 0.05))
    differences["meridian"] = np.array([-0.01] * 18 + [0.38] * 2 + [-0.01] * 18 + [0.38] * 2)
    return _records(differences)


def test_by_rung_the_printed_word_is_hall_s_where_hall_s_reader_reads_the_look():
    records = _skewed()
    hall_m2 = {c.rival: c for c in compare(records, dataclasses.replace(SCORED, reader="hall"))}
    by_rung = {c.rival: c for c in compare(records, BY_RUNG)}
    x = np.array([-0.01] * 36 + [0.38] * 4)
    low = x.mean() - float(student.ppf(0.975, 39)) * x.std(ddof=1) / math.sqrt(40)
    assert low < 0.0
    # M2's line printed Student's word beside Hall's level; by rung both are Hall's
    assert hall_m2["meridian"].reading == "ties"
    assert hall_m2["meridian"].level <= ALPHA
    assert by_rung["meridian"].reading == "loses to"
    assert by_rung["status quo"].reading == "beats"
    for c in by_rung.values():
        assert (c.reading == "loses to") == (c.level <= ALPHA)
        assert c.within == (c.within_level <= ALPHA)


def test_a_comparison_predicted_to_fail_is_printed_and_leaves_its_verdict():
    d = dataclasses.replace(BY_RUNG, predicted=((0, "meridian"),))
    by = {c.rival: c for c in compare(_skewed(), d)}
    assert not by["meridian"].claimed
    assert all(by[r].claimed for r in RIVALS if r != "meridian")
    (v,) = verdicts(list(by.values()), d.order)
    assert v.predicted == ("meridian",)
    assert "meridian" not in v.rivals


def test_a_declaration_by_rung_is_written_and_read_back_and_m2_s_is_unchanged(tmp_path):
    d = dataclasses.replace(
        BY_RUNG, predicted=((0, "meridian"), (4, "pymc-marketing")), order=((1, 0), (2, 4))
    )
    Layout(tmp_path / "a").declare(d)
    assert Layout(tmp_path / "a").read() == d
    assert json.loads((tmp_path / "a" / "run.json").read_text())["order"] == [[1, 0], [2, 4]]
    Layout(tmp_path / "a").declare(d)  # declared again: the same
    Layout(tmp_path / "b").declare(SCORED)
    held = json.loads((tmp_path / "b" / "run.json").read_text())
    assert "per_rung" not in held
    assert "predicted" not in held
    assert "order" not in held


ORDERED = {"per_rung": True, "order": ((1, 0),)}


@pytest.mark.parametrize(
    "change",
    [
        {**ORDERED, "default": None, "margin": None},
        {"predicted": ((0, "meridian"),)},
        {**ORDERED, "predicted": ((-1, "meridian"),)},
        {**ORDERED, "predicted": ((0, "meridian"), (0, "meridian"))},
        {"per_rung": True},
        {"order": ((1, 0),)},
        {"per_rung": True, "order": ((1, 0), (1, 0))},
        {"per_rung": True, "order": ((11, 0),)},
        {"per_rung": True, "order": ((1, -1),)},
        {"per_rung": True, "order": ((1, True),)},
        {"per_rung": True, "order": ((1, 0, 4),)},
    ],
)
def test_a_claim_by_rung_without_a_claim_or_a_malformed_prediction_or_order_is_refused(change):
    with pytest.raises(ValueError, match=r"claim by rung|predicted failure|order of verdicts"):
        dataclasses.replace(SCORED, **change)


EVERY_RIVAL = tuple((0, rival) for rival in RIVALS)
TWO_RUNGS = dataclasses.replace(BY_RUNG, rungs=(0, 4), order=((1, 4),))
"""A run that claims its rung 4 alone: its rung 0 is either exploratory or predicted to fail."""


def _two_rungs():
    """Rung 0 with a loss beyond the margin to Orbit, rung 4 plainly within everywhere."""
    spread = dict.fromkeys(RIVALS, 0.1)
    at_0 = {"status quo": -0.2, "equal split": -0.1, "meridian": -0.1, "orbit": 0.3}
    return [
        *_records(_differences(at_0, spread), k=0),
        *_records(_differences(dict.fromkeys(RIVALS, -0.2), spread), k=4),
    ]


def test_an_exploratory_rung_is_read_printed_and_left_out_of_the_claim_holm_and_headline():
    """Rung 0 declared exploratory is read and printed as exploratory, never "predicted to fail",
    and leaves the verdicts, Holm's list and the headline. Declared by naming every rival predicted
    to fail, as M4's roots kept their rungs past the cap out, it would enter Holm's list."""
    d = dataclasses.replace(TWO_RUNGS, exploratory_rungs=(0,))
    comparisons = compare(_two_rungs(), d)
    assert {c.k for c in comparisons if c.exploratory} == {0}
    assert all(c.claimed for c in comparisons)
    printed = "\n".join(lines(comparisons, 0.05, order=d.order))
    assert "family 1, k = 0, less orbit" in printed
    assert "predicted to fail" not in printed
    assert "  family 1, k = 0: exploratory, read and not claimed" in printed
    assert "family 1, k = 4: holds" in printed
    assert printed.endswith("over 4 comparisons: none")
    assert losses(comparisons) == frozenset()
    assert "no headline" in printed
    (v,) = verdicts(comparisons, d.order)
    assert (v.family, v.k, v.holds) == (1, 4, True)

    predicted = dataclasses.replace(TWO_RUNGS, predicted=EVERY_RIVAL)
    as_predicted = compare(_two_rungs(), predicted)
    assert not any(c.exploratory for c in as_predicted)
    assert (1, 0, "orbit") in losses(as_predicted)
    assert "not claimed, predicted to fail" in "\n".join(
        verdict_lines(as_predicted, 0.05, predicted.order)
    )


def test_an_exploratory_run_leaves_no_headline_and_a_capped_exploratory_tool_does_not_stop_it():
    """A family read only as exploratory, its run declared so, is printed and stops the headline;
    a tool read on its cap and exploratory is read beside a verdict that holds, as M2's Robyn and
    Meridian were, and the headline is said."""
    spread = dict.fromkeys(RIVALS, 0.1)
    records = _records(_differences(dict.fromkeys(RIVALS, -0.2), spread))
    capped = dataclasses.replace(BY_RUNG, caps={"orbit": 8}, exploratory_caps=True)
    printed = verdict_lines(compare(records, capped), 0.05, capped.order)
    assert printed[-1].startswith("the headline holds")
    explored = dataclasses.replace(SCORED, exploratory=True)
    # another family's exploratory run: its comparisons, read as family 2's
    other = [dataclasses.replace(c, family=2) for c in compare(records, explored)]
    both = [*compare(records, BY_RUNG), *other]
    printed = verdict_lines(both, 0.05, BY_RUNG.order)
    assert "  family 2, k = 0: exploratory, read and not claimed" in printed
    assert printed[-1] == "no headline: some family or rung is not claimed whole or does not hold"


def test_exploratory_rungs_are_written_and_read_back_and_left_out_where_none(tmp_path):
    d = dataclasses.replace(TWO_RUNGS, exploratory_rungs=(0,))
    Layout(tmp_path / "a").declare(d)
    assert Layout(tmp_path / "a").read() == d
    assert json.loads((tmp_path / "a" / "run.json").read_text())["exploratory_rungs"] == [0]
    Layout(tmp_path / "b").declare(TWO_RUNGS)
    assert "exploratory_rungs" not in json.loads((tmp_path / "b" / "run.json").read_text())
    # a pool's rung above 0, named as its first run names it
    pooled = dataclasses.replace(
        d, rungs=(0, 2), pool=("a", "b"), exploratory_rungs=(1,), order=((1, 0),)
    )
    assert pooled.reads(0)
    assert not pooled.reads(1)


@pytest.mark.parametrize(
    "change",
    [
        {"exploratory_rungs": (2,)},
        {"exploratory_rungs": (0, 0)},
        {"exploratory_rungs": (True,)},
        {
            "exploratory_rungs": (0,),
            "reader": "student",
            "default": None,
            "margin": None,
            "per_rung": False,
            "order": (),
        },
    ],
)
def test_an_exploratory_rung_the_run_does_not_hold_or_cannot_claim_is_refused(change):
    with pytest.raises(ValueError, match="an exploratory rung"):
        dataclasses.replace(TWO_RUNGS, **change)


@pytest.mark.parametrize(
    "change",
    [
        {"exploratory": True},
        {"exploratory_rungs": (0,)},
        {"order": ((1, 4),)},
        {"predicted": EVERY_RIVAL},
        {"predicted": EVERY_RIVAL[:-1], "caps": {"orbit": 5}, "exploratory_caps": True},
    ],
)
def test_an_order_that_names_a_verdict_of_the_run_the_claim_does_not_read_is_refused(change):
    """The run's own family at a rung the claim does not read of it: an exploratory run, a rung
    the run does not hold, or every rival predicted to fail or capped exploratory."""
    with pytest.raises(ValueError, match="which the claim does not read of this run"):
        dataclasses.replace(BY_RUNG, **change)


def test_an_order_names_what_the_claim_reads_and_other_families_unchecked():
    held = dataclasses.replace(BY_RUNG, predicted=EVERY_RIVAL[:-1], order=((1, 0), (2, 4), (3, 0)))
    assert held.reads(0)
    assert not held.reads(4)
    capped = dataclasses.replace(held, caps={"orbit": 5})
    assert capped.reads(0)
    pooled = dataclasses.replace(
        held, rungs=(0, 2), pool=("a", "b"), order=((1, 0), (1, 1)), predicted=()
    )
    assert pooled.reads(1)


def _by_rung(roots, **change):
    """M2's layout with every run declaring the claim by rung, its verdicts in `M2_ORDER`."""
    for root in roots:
        layout = Layout(root)
        d = layout.read()
        if d.margin is not None:
            layout.declaration.unlink()
            layout.declare(
                dataclasses.replace(d, **{"per_rung": True, "order": M2_ORDER, **change})
            )
    return roots


def test_claim_read_by_rung_prints_verdicts_the_headline_and_the_secondary_readings(
    tmp_path, capsys, monkeypatch
):
    roots = _by_rung(_m2(tmp_path))
    out = _read(roots, capsys, monkeypatch)
    start = out.index(next(line for line in out if line.startswith("the claim by family")))
    held = out[start + 1 : start + 7]
    assert len(held) == 6
    assert [line.split(":")[0] for line in held] == [f"  family {f}, k = {k}" for f, k in M2_ORDER]
    assert all(line.split(": ")[1].startswith("holds (level") for line in held)
    assert all(line.endswith("4 rivals, each within unadjusted)") for line in held)
    assert out[start + 7].startswith("the headline holds")
    assert not any(line.startswith("the claim holds") for line in out)
    secondary = out[
        out.index("secondary readings, reported beside the claim and never read by it:") :
    ]
    assert secondary[1].startswith("  every stage of every fit past the bounds at the status quo:")
    assert secondary[2].startswith("  at the margin 0.01:")
    assert len([line for line in secondary if ", k = 0: the best plan less" in line]) == 3


def test_a_prediction_keeps_the_headline_from_being_said(tmp_path, capsys, monkeypatch):
    roots = _by_rung(_m2(tmp_path), predicted=((0, "orbit"),))
    out = _read(roots, capsys, monkeypatch)
    assert any(line.endswith("; not claimed: predicted to fail") for line in out)
    assert any("not claimed, predicted to fail: orbit" in line for line in out)
    assert any(line.startswith("no headline") for line in out)


def test_runs_that_declare_the_claim_by_rung_apart_are_refused(tmp_path, capsys, monkeypatch):
    roots = _m2(tmp_path)
    _by_rung([r for r in roots if r.name == "f10"])
    with pytest.raises(SystemExit):
        _read(roots, capsys, monkeypatch)
    assert "every run declares the claim by rung, or none" in capsys.readouterr().err


def test_runs_that_declare_their_verdicts_in_other_orders_are_refused(
    tmp_path, capsys, monkeypatch
):
    roots = _by_rung(_m2(tmp_path))
    f10 = Layout(next(r for r in roots if r.name == "f10"))
    d = f10.read()
    f10.declaration.unlink()
    f10.declare(dataclasses.replace(d, order=M2_ORDER[::-1]))
    with pytest.raises(SystemExit):
        _read(roots, capsys, monkeypatch)
    assert "every run declares the same order of verdicts" in capsys.readouterr().err


def test_an_order_that_leaves_out_a_verdict_the_claim_reads_is_refused(
    tmp_path, capsys, monkeypatch
):
    roots = _by_rung(_m2(tmp_path), order=M2_ORDER[:-1])
    with pytest.raises(SystemExit):
        _read(roots, capsys, monkeypatch)
    assert "does not name family 10 rung 4" in capsys.readouterr().err


def _two_stage(rhat):
    return {
        "flags": {
            "fitted not mixed": 0,
            "hill weight": 1.0,
            "hill max rhat": rhat,
            "hill divergences": 0,
            "tanh max rhat": 1.5,
            "tanh divergences": 4000,
        },
        "settings": {
            "first stage": "hill",
            "sampler": {"chains": 4, "warmup": 1000, "draws": 1000},
            "readings": 400,
        },
    }


def test_the_convergence_rule_reads_the_two_stage_arm_s_first_stage():
    """The first stage's Hill fit, the one its readings hold, past R-hat 1.01 makes the record
    past, though the DLM's own chains mixed; the tanh fit it reads none of is not read."""
    assert report.past(_two_stage(1.005)) is False
    assert report.past(_two_stage(1.02)) is True
    plain = _two_stage(1.02)
    del plain["settings"]["first stage"]
    assert report.past(plain) is False


def test_the_secondary_reading_s_gain_is_the_status_quo_s_regret_negated(tmp_path):
    """A run by rung of two worlds an environment: the best plan less the status quo is minus the
    status quo's mean regret, and Meridian less the status quo is read on Meridian's worlds."""
    from scorecard.claim import Run, secondary_lines

    d = dataclasses.replace(BY_RUNG, worlds=2)
    records = _records(
        _differences(dict.fromkeys(RIVALS, -0.1), dict.fromkeys(RIVALS, 0.05), n=4), worlds=2
    )
    lines = secondary_lines([Run(d, tuple(records))])
    regrets = [r.arms["status quo"].regret for r in records]
    gains = [r.arms["meridian"].regret - r.arms["status quo"].regret for r in records]
    assert f"the best plan less the status quo {-np.mean(regrets):+.4f}" in lines[-1]
    assert f"meridian less the status quo {np.mean(gains):+.4f}" in lines[-1]


AGAINST = dataclasses.replace(BY_RUNG, default_less=("dlm mixture",))
"""A run by rung that reads its default at rung 0 against its arm's other variant `dlm mixture`."""


@pytest.mark.parametrize(
    "change",
    [
        {"default": None, "margin": None, "reader": "student", "per_rung": False, "order": ()},
        {"rungs": (4,), "order": ((1, 4),)},
        {"default_less": ("dlm mixture", "dlm mixture")},
        {"default_less": ("dlm plug-in",)},
        {"default_less": ("meridian",)},
    ],
)
def test_a_variant_read_against_the_default_off_a_claimed_rung_0_or_named_badly_is_refused(change):
    with pytest.raises(ValueError, match="read against variants"):
        dataclasses.replace(AGAINST, **change)


def test_a_declaration_naming_variants_is_written_with_them_and_one_naming_none_without(tmp_path):
    Layout(tmp_path / "a").declare(AGAINST)
    assert Layout(tmp_path / "a").read() == AGAINST
    assert json.loads((tmp_path / "a" / "run.json").read_text())["default_less"] == ["dlm mixture"]
    Layout(tmp_path / "b").declare(BY_RUNG)
    assert "default_less" not in json.loads((tmp_path / "b" / "run.json").read_text())


def _shifted(records, shift, failed=()):
    """`records` with `dlm mixture`'s regret raised by `shift`, and its plan failed on the worlds
    of indices `failed` of the first environment."""
    out = []
    for r in records:
        score = r.arms["dlm mixture"]
        score = dataclasses.replace(score, regret=score.regret + shift)
        if r.environment == records[0].environment and r.index in failed:
            score = dataclasses.replace(score, failure="raised")
        out.append(dataclasses.replace(r, arms={**r.arms, "dlm mixture": score}))
    return out


def test_the_default_is_read_against_a_declared_variant_run_by_run_and_pooled_by_stratum():
    """Two confirmatory runs and one on scored worlds: each run's line is the paired mean and
    Student-t interval of its complete cases (a failed plan of the variant leaves its world out and
    is counted); the pooled line reads the confirmatory runs alone, every complete case one world,
    the variance of the mean stratified by run, sum (n_f / N)^2 s_f^2 / n_f, and names a family
    two of its runs share once (family 7's two roots are one family)."""
    from scorecard.claim import Run, secondary_lines

    first = dataclasses.replace(AGAINST, confirmatory=True)
    second = dataclasses.replace(AGAINST, confirmatory=True, worlds=14)
    scored = dataclasses.replace(AGAINST, worlds=10)
    differences = _differences(dict.fromkeys(RIVALS, -0.1), dict.fromkeys(RIVALS, 0.05))
    runs = [
        Run(first, tuple(_shifted(_records(differences, worlds=20), 0.0, failed=(3,)))),
        Run(second, tuple(_shifted(_records(differences, worlds=14), 0.25))),
        Run(scored, tuple(_shifted(_records(differences, worlds=10), -0.5))),
    ]
    lines = secondary_lines(runs)
    held = [line for line in lines if "dlm plug-in less dlm mixture" in line]
    assert len(held) == 4
    gaps = []
    for run, line in zip(runs, held, strict=False):
        kept = [r for r in run.records if r.arms["dlm mixture"].failure is None]
        gap = np.array([r.arms[DEFAULT].regret - r.arms["dlm mixture"].regret for r in kept])
        half = student.ppf(0.975, len(gap) - 1) * gap.std(ddof=1) / math.sqrt(len(gap))
        assert line == (
            f"  family 1, k = 0, dlm plug-in less dlm mixture: {gap.mean():+.4f} "
            f"[{gap.mean() - half:+.4f}, {gap.mean() + half:+.4f}] on {len(gap)} complete cases; "
            f"dlm mixture failed {len(run.records) - len(kept)}, dlm plug-in 0"
        )
        gaps.append(gap)
    confirmatory = gaps[:2]
    total = sum(len(g) for g in confirmatory)
    mean = np.concatenate(confirmatory).mean()
    se = math.sqrt(sum((len(g) / total) ** 2 * g.var(ddof=1) / len(g) for g in confirmatory))
    z = 1.959963984540054
    assert held[3] == (
        "  pooled over the confirmatory runs at k = 0 (families 1), dlm plug-in less dlm "
        f"mixture: {mean:+.4f} [{mean - z * se:+.4f}, {mean + z * se:+.4f}] on {total} complete "
        "cases, the variance stratified by run"
    )
    assert total == 2 * 20 - 1 + 2 * 14
