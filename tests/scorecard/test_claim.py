"""The claim: its constants against Maxima's, the worlds a family needs and the power they give, the
gate's verdicts and Holm's list of losses on scores whose answers are known, the
intersection-union reading over runs, and the declaration that carries the margin. A comparison
reads its complete cases, the failures counted beside it, an environment where the rival failed on
most worlds left out, and the status quo standing in for a failed plan as a secondary readout."""

import dataclasses
import json
import math
import sys

import numpy as np
import pytest

pytest.importorskip("causaldyn_bench")

from causaldyn_bench.scorecard.scoring import ArmScore, WorldRecord
from scipy.stats import norm
from scipy.stats import t as student

from scorecard import claim, report, run
from scorecard.claim import CAP, Size, Z, compare, holds, losses, needed, size
from scorecard.layout import Declaration, Layout, stamp

MAXIMA_Z = 2.801585218112968440703300551883799628581
"""z_0.975 + z_0.8, each solved from erf at 40 digits by bisection in Maxima 5.50."""
LAWS = ("walk", "trend")
DEFAULT = "dlm plug-in"
RIVALS = ("status quo", "equal split", "meridian", "orbit")
SIBLING = (-1.0, 0.5, 2.0, -1.5, 0.0, 1.0, -1.0)
SCORED = Declaration(
    family=1,
    environments=LAWS,
    worlds=20,
    pilot=False,
    bench="1f4161c",
    rungs=(0,),
    arms=("dlm",),
    external=("meridian", "orbit"),
    default=DEFAULT,
    margin=0.05,
)


def _score(regret):
    return ArmScore(
        regret=float(regret),
        failure=None,
        moved=None,
        bounds=0,
        realised=None,
        uplift=None,
        covered=None,
        crps=None,
        record={"error": None},
    )


def _standard(n, seed):
    """`n` values of mean 0 and SD 1 exactly."""
    z = np.random.default_rng(seed).standard_normal(n)
    return (z - z.mean()) / z.std(ddof=1)


def _records(differences, *, worlds=20, k=0, pilot=False):
    """Every law's `worlds` worlds at rung `k`, the default's regret 1 and each rival's 1 less its
    row of `differences`, which runs over the laws' worlds in order."""
    records = []
    for place, law in enumerate(LAWS):
        for i in range(worlds):
            row = place * worlds + i
            # the default's sibling spreads widely: a rival, it would bind every size
            arms = {DEFAULT: _score(1.0), "dlm mixture": _score(1.0 + 5.0 * SIBLING[row % 7])}
            arms |= {rival: _score(1.0 - differences[rival][row]) for rival in differences}
            records.append(
                WorldRecord(
                    family=1,
                    environment=law,
                    seed=221_000 + 1000 * place + i,
                    index=i,
                    pilot=pilot,
                    k=k,
                    budget=1000.0,
                    best=2000.0,
                    channels=3,
                    oracle_bounds=0,
                    covariates={},
                    arms=arms,
                )
            )
    return records


def _differences(means, spreads, n=40, seed=3):
    z = _standard(n, seed)
    return {rival: means[rival] + spreads[rival] * z for rival in means}


def test_z_is_maxima_s_sum_of_the_two_quantiles():
    assert abs(Z - MAXIMA_Z) <= 1e-12
    assert round(Z, 4) == 2.8016


def test_the_cap_is_the_design_s_worlds_for_a_tie_at_its_paired_sd():
    """(2.8016 * 0.081 / 0.01)^2 is 514.96 in Maxima: 520 in tens, and not past the cap."""
    assert needed(0.081, 0.01) == CAP == 520
    assert not Size(0, 1, "a", 0, 0.081, needed(0.081, 0.01)).exploratory
    assert Size(0, 1, "a", 0, 0.1, 530).exploratory


def test_n_is_the_fewest_tens_that_reach_the_power():
    """An SD whose exact need is 41.2 worlds needs 50; one whose need is below 10 needs 10."""
    sd = 0.05 * math.sqrt(41.2) / Z
    assert needed(sd, 0.05) == 50
    assert needed(1e-6, 0.05) == 10
    assert needed(0.0, 0.05) == 10


@pytest.mark.parametrize("sd", [0.08, 0.21, 0.3, 0.33])
def test_the_needed_worlds_reach_the_power_by_the_bench_s_gate_and_ten_fewer_do_not(sd):
    n = needed(sd, 0.05)
    assert n <= CAP
    s = Size(1, 1, "a", 0, sd, n)
    closed = float(norm.cdf(0.05 * math.sqrt(n) / sd - norm.isf(0.025)))
    assert s.power(0.05) == pytest.approx(closed, rel=0.0, abs=1e-9)
    assert s.power(0.05) >= 0.8
    assert Size(1, 1, "a", 0, sd, n - 10).power(0.05) < 0.8


def test_size_reads_the_largest_sd_over_the_rivals_and_the_rungs():
    """At k = 4 Meridian's paired SD is 0.3, the largest: (2.8016 * 0.3 / 0.05)^2 is 282.6."""
    pilot = dataclasses.replace(
        SCORED, worlds=10, pilot=True, bench=None, rungs=(0, 4), default=None, margin=None
    )
    spreads = dict(zip(RIVALS, (0.1, 0.2, 0.25, 0.05), strict=True))
    records = _records(
        _differences(dict.fromkeys(RIVALS, 0.02), spreads, 20), worlds=10, pilot=True
    )
    spreads["meridian"] = 0.3
    at4 = _differences(dict.fromkeys(RIVALS, -0.1), spreads, 20, seed=5)
    records += _records(at4, worlds=10, k=4, pilot=True)
    s = size(records, pilot, DEFAULT, 0.05)
    assert (s.rival, s.k, s.needed, s.worlds) == ("meridian", 4, 290, 145)
    assert s.sd == pytest.approx(0.3, rel=1e-12, abs=0.0)
    assert not s.exploratory
    wide = size(records, pilot, DEFAULT, 0.03)
    assert (wide.needed, wide.exploratory, wide.worlds) == (790, True, 260)


def test_size_refuses_a_pilot_that_misses_an_environment():
    pilot = dataclasses.replace(SCORED, worlds=10, pilot=True, default=None, margin=None)
    records = _records(_differences(dict.fromkeys(RIVALS, 0.0), dict.fromkeys(RIVALS, 0.1)))
    with pytest.raises(ValueError, match=r"no world of \['trend'\]"):
        size([r for r in records if r.environment == "walk"], pilot, DEFAULT, 0.05)


def _known():
    """The default less: the status quo, -0.2 at SD 0.1, plainly within; the equal split, +0.04 at
    SD 0.01, within the margin and yet a loss; Meridian, +0.03 at SD 0.2, a tie the margin does not
    hold; Orbit, +0.2 at SD 0.2, a loss outside it."""
    means = {"status quo": -0.2, "equal split": 0.04, "meridian": 0.03, "orbit": 0.2}
    spreads = {"status quo": 0.1, "equal split": 0.01, "meridian": 0.2, "orbit": 0.2}
    return means, spreads


def test_the_gate_reads_each_rival_against_the_margin_and_the_rule_against_nought():
    means, spreads = _known()
    comparisons = compare(_records(_differences(means, spreads)), SCORED)
    by = {c.rival: c for c in comparisons}
    assert list(by) == list(RIVALS)
    assert {r: by[r].within for r in RIVALS} == {
        "status quo": True,
        "equal split": True,
        "meridian": False,
        "orbit": False,
    }
    assert {r: by[r].reading for r in RIVALS} == {
        "status quo": "beats",
        "equal split": "loses to",
        "meridian": "ties",
        "orbit": "loses to",
    }
    half = float(student.ppf(0.975, 39)) / math.sqrt(40)
    for rival in RIVALS:
        assert by[rival].mean == pytest.approx(means[rival], rel=0.0, abs=1e-12)
        interval = by[rival].interval
        assert interval is not None
        low, high = interval
        assert high == pytest.approx(means[rival] + half * spreads[rival], rel=1e-12, abs=1e-15)
        assert low == pytest.approx(means[rival] - half * spreads[rival], rel=1e-12, abs=1e-15)
        statistic = means[rival] / (spreads[rival] / math.sqrt(40))
        expected = float(student.sf(statistic, 39))
        assert by[rival].level == pytest.approx(max(expected, 1e-12), rel=1e-6, abs=0.0)
    assert not holds(comparisons)
    assert losses(comparisons) == {(1, 0, "equal split"), (1, 0, "orbit")}


def test_holm_leaves_out_a_loss_the_unadjusted_rule_reads():
    """Meridian's loss has a p-value of 0.015, under 0.025 alone but not under 0.025 / 4."""
    spread = 0.1
    mean = float(student.isf(0.015, 39)) * spread / math.sqrt(40)
    means = {"status quo": -0.2, "equal split": -0.2, "meridian": mean, "orbit": -0.2}
    differences = _differences(means, dict.fromkeys(RIVALS, spread))
    comparisons = compare(_records(differences), SCORED)
    (meridian,) = (c for c in comparisons if c.rival == "meridian")
    assert meridian.reading == "loses to"
    assert meridian.level == pytest.approx(0.015, rel=1e-6, abs=0.0)
    assert losses(comparisons) == frozenset()


def test_the_claim_needs_every_comparison_and_leaves_an_exploratory_run_out():
    within = dict.fromkeys(RIVALS, -0.1)
    passing = compare(_records(_differences(within, dict.fromkeys(RIVALS, 0.1))), SCORED)
    assert holds(passing)
    means, spreads = _known()
    exploratory = dataclasses.replace(SCORED, exploratory=True)
    lost = compare(_records(_differences(means, spreads)), exploratory)
    assert all(c.exploratory for c in lost)
    assert holds(passing + lost)
    assert losses(passing + lost) == frozenset()
    assert not holds(lost)
    one_out = dataclasses.replace(passing[2], within=False)
    assert not holds([*passing[:2], one_out, *passing[3:]])


def test_a_run_that_does_not_hold_its_worlds_whole_is_unread_and_fails_the_claim():
    records = _records(_differences(dict.fromkeys(RIVALS, -0.1), dict.fromkeys(RIVALS, 0.1)))
    short = [r for r in records if not (r.environment == "trend" and r.index == 19)]
    alone = [r for r in records if r.environment == "walk"]
    for kept in (short, alone, []):
        comparisons = compare(kept, SCORED)
        assert [c.within for c in comparisons] == [None] * len(RIVALS)
        assert [c.level for c in comparisons] == [1.0] * len(RIVALS)
        assert not holds(comparisons)
        assert any("unread" in line for line in claim.lines(comparisons, 0.05))


def test_compare_refuses_a_run_that_declares_no_margin():
    bare = dataclasses.replace(SCORED, default=None, margin=None)
    with pytest.raises(ValueError, match="no default procedure and margin"):
        compare([], bare)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"margin": None}, "declared together"),
        ({"default": None}, "declared together"),
        ({"margin": 0.0}, "above 0"),
        ({"margin": -0.05}, "above 0"),
        ({"margin": math.nan}, "above 0"),
        ({"margin": math.inf}, "above 0"),
        ({"default": "meridian"}, "this engine's arms"),
        ({"worlds": 1}, "at least 2 worlds"),
        ({"default": None, "margin": None, "exploratory": True}, "exploratory"),
    ],
)
def test_a_claim_the_run_cannot_make_is_refused(change, match):
    with pytest.raises(ValueError, match=match):
        dataclasses.replace(SCORED, **change)


def _design(tmp_path):
    """A pre-registration's file, which the runner declares a scored run only with."""
    path = tmp_path / "design.md"
    path.write_text("the design, fixed before the first scored world\n")
    return path


def test_the_default_procedure_is_one_of_this_engine_s_variants(tmp_path):
    declared = dataclasses.replace(
        SCORED,
        external=(),
        limits={"dlm": run.Limits("1G", 60.0, 1)},
        preregistration=stamp(_design(tmp_path)),
    )
    with pytest.raises(ValueError, match="default procedure is one of"):
        run.declare(Layout(tmp_path / "a"), dataclasses.replace(declared, default="stacked"))
    run.declare(Layout(tmp_path / "b"), declared)
    assert Layout(tmp_path / "b").read() == declared


def test_a_declaration_written_before_the_claim_reads_without_one(tmp_path):
    layout = Layout(tmp_path)
    held = dataclasses.asdict(dataclasses.replace(SCORED, default=None, margin=None))
    for name in ("default", "margin", "exploratory"):
        del held[name]
    layout.declaration.write_text(json.dumps(held))
    assert (layout.read().margin, layout.read().default, layout.read().exploratory) == (
        None,
        None,
        False,
    )


def _write(layout, declaration, records):
    layout.declare(declaration)
    layout.scores.write_text("".join(json.dumps(r.as_json()) + "\n" for r in records))


def test_the_report_ends_with_the_claim_on_its_run(tmp_path, capsys):
    means, spreads = _known()
    layout = Layout(tmp_path)
    _write(layout, SCORED, _records(_differences(means, spreads)))
    report.main(layout, ["dlm plug-in", "dlm mixture", "dlm cvar 0.1"])
    out = capsys.readouterr().out
    assert "the claim on this run alone, dlm plug-in less every rival:" in out
    assert "less orbit" in out
    assert (
        "the claim does not hold: 2 of 4 not within: family 1 k 0 meridian, family 1 k 0 orbit"
        in out
    )
    assert "over 4 comparisons: family 1 k 0 equal split, family 1 k 0 orbit" in out


def test_the_command_line_sizes_pilots_and_reads_the_claim_over_runs(tmp_path, capsys, monkeypatch):
    pilot = dataclasses.replace(SCORED, worlds=20, pilot=True, default=None, margin=None)
    spreads = dict(zip(RIVALS, (0.1, 0.2, 0.25, 0.05), strict=True))
    _write(
        Layout(tmp_path / "pilot"),
        pilot,
        _records(_differences(dict.fromkeys(RIVALS, 0.0), spreads), pilot=True),
    )
    command = ["claim", "size", "--default", DEFAULT, "--margin", "0.05", str(tmp_path / "pilot")]
    monkeypatch.setattr(sys, "argv", command)
    claim.main()
    line = capsys.readouterr().out.strip()
    assert line.startswith(
        "family 1: N 200; 100 worlds in each of 2 environments; binding meridian"
    )
    within = dict.fromkeys(RIVALS, -0.1)
    _write(Layout(tmp_path / "a"), SCORED, _records(_differences(within, spreads)))
    _write(Layout(tmp_path / "b"), SCORED, _records(_differences(within, spreads, seed=9)))
    monkeypatch.setattr(sys, "argv", ["claim", "read", str(tmp_path / "a"), str(tmp_path / "b")])
    claim.main()
    out = capsys.readouterr().out
    assert "the claim holds: 8 of 8 comparisons within 0.05" in out


MAXIMA_ONE = 0.8041312305315330763726352881896923749616
MAXIMA_FOUR = 0.4181265235777157620854725286767616429850
"""Phi(0.2 sqrt(200) - t_0.975,199) and its fourth power: the chance of one comparison of SD 1 at
a true tie against a margin of 0.2 on 200 worlds, and of four independent ones together, by Maxima
5.50 at 40 digits (the t quantile by bisection on cdf_student_t)."""


def _independent(columns, worlds=2000, seed=7):
    """Two environments' `worlds` worlds of `columns` independent comparisons, each column of
    mean 0 and SD 1 exactly in each environment."""
    rng = np.random.default_rng(seed)
    strata = []
    for _ in range(2):
        z = rng.standard_normal((worlds, columns))
        strata.append((z - z.mean(axis=0)) / z.std(axis=0, ddof=1))
    return claim.Differences(tuple((f"r{i}", 0) for i in range(columns)), tuple(strata))


def test_one_comparison_needs_what_the_binding_rule_gives_it():
    """SD 0.3 against 0.05: the rule's 290 worlds give 0.808 by the gate, 280 give 0.794."""
    pilot = _independent(1)
    scaled = claim.Differences(pilot.comparisons, tuple(0.3 * s for s in pilot.strata))
    n, (chance,), joint = claim.joint_needed(scaled, 0.05)
    assert n == needed(0.3, 0.05) == 290
    assert chance == joint
    se = math.sqrt(0.8 * 0.2 / claim.RESAMPLES)
    assert joint == pytest.approx(Size(1, 2, "r0", 0, 0.3, n).power(0.05), rel=0.0, abs=4 * se)


def test_two_identical_comparisons_need_what_one_does():
    pilot = _independent(1)
    twice = claim.Differences(
        (("r0", 0), ("r0", 4)), tuple(np.column_stack([s, s]) for s in pilot.strata)
    )
    one = claim.joint_needed(pilot, 0.2)
    n, (first, second), joint = claim.joint_needed(twice, 0.2)
    assert (n, joint) == (one[0], one[2])
    assert first == second == one[1][0]


def test_independent_comparisons_together_reach_the_product_of_their_chances():
    pilot = _independent(4)
    each, joint = claim.chances(pilot, 0.2, 100)
    one = math.sqrt(MAXIMA_ONE * (1 - MAXIMA_ONE) / claim.RESAMPLES)
    four = math.sqrt(MAXIMA_FOUR * (1 - MAXIMA_FOUR) / claim.RESAMPLES)
    assert each == pytest.approx([MAXIMA_ONE] * 4, rel=0.0, abs=4 * one)
    assert joint == pytest.approx(MAXIMA_FOUR, rel=0.0, abs=4 * four)


def test_the_joint_n_is_not_the_binding_comparison_s():
    """Four comparisons of one SD: the binding rule's 200 worlds give them 0.42 together; the joint
    N is the fewest tens whose chance reaches 0.8, and ten fewer do not."""
    pilot = _independent(4)
    rule = needed(1.0, 0.2)
    n, each, joint = claim.joint_needed(pilot, 0.2)
    assert rule == 200
    assert n >= 300
    assert joint >= claim.POWER
    assert min(each) > claim.POWER
    assert claim.chances(pilot, 0.2, (n - 10) // 2)[1] < claim.POWER
    assert claim.chances(pilot, 0.2, rule // 2)[1] < 0.45


def test_the_bootstrap_is_a_fact_of_its_seed():
    pilot = _independent(3, worlds=200)
    assert claim.chances(pilot, 0.2, 60, seed=5) == claim.chances(pilot, 0.2, 60, seed=5)
    assert claim.chances(pilot, 0.2, 60, seed=5) != claim.chances(pilot, 0.2, 60, seed=6)


def test_joint_size_reads_every_rival_at_every_rung_of_a_pilot():
    pilot = dataclasses.replace(
        SCORED, worlds=10, pilot=True, bench=None, rungs=(0, 4), default=None, margin=None
    )
    spreads = dict(zip(RIVALS, (0.1, 0.2, 0.25, 0.05), strict=True))
    records = _records(
        _differences(dict.fromkeys(RIVALS, 0.02), spreads, 20), worlds=10, pilot=True
    )
    records += _records(
        _differences(dict.fromkeys(RIVALS, -0.1), spreads, 20, seed=5), worlds=10, k=4, pilot=True
    )
    j = claim.joint_size(records, pilot, DEFAULT, 0.05)
    assert j.comparisons == tuple((rival, k) for k in (0, 4) for rival in RIVALS)
    assert j.environments == 2
    assert j.needed >= size(records, pilot, DEFAULT, 0.05).needed
    assert j.joint >= claim.POWER
    assert len(j.chances) == 8
    with pytest.raises(ValueError, match=r"no world of \['trend'\]"):
        claim.joint_size([r for r in records if r.environment == "walk"], pilot, DEFAULT, 0.05)
    with pytest.raises(ValueError, match="regret is not scored"):
        claim.differences(records, dataclasses.replace(pilot, external=("robyn",)), DEFAULT)


def test_the_command_line_prints_the_joint_size_beside_the_binding_one(
    tmp_path, capsys, monkeypatch
):
    pilot = dataclasses.replace(SCORED, worlds=20, pilot=True, default=None, margin=None)
    spreads = dict(zip(RIVALS, (0.1, 0.2, 0.25, 0.05), strict=True))
    _write(
        Layout(tmp_path / "pilot"),
        pilot,
        _records(_differences(dict.fromkeys(RIVALS, 0.0), spreads), pilot=True),
    )
    command = ["claim", "size", "--default", DEFAULT, "--margin", "0.05", "--joint"]
    monkeypatch.setattr(sys, "argv", [*command, "--resamples", "2000", str(tmp_path / "pilot")])
    claim.main()
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith("family 1: N 200; 100 worlds in each of 2 environments")
    assert lines[1].startswith("family 1: joint N ")
    assert "(2000 resamples, seed 1982); at the binding rule's N 200 (200 worlds), " in lines[1]
    assert [line.split(":")[0].split() for line in lines[2:-1]] == [
        ["less", *rival.split(), "k", "=", "0"] for rival in RIVALS
    ]
    joint = lines[1].split("with chance ")[1][:5]
    assert lines[-1].startswith(
        f"the launch claim over the families sized here (1), at a true tie: chance {joint} at "
        "their joint N"
    )


@pytest.mark.parametrize(
    ("worlds", "confidence", "ratio"),
    [
        (180, 0.8, 1.048211794771896063739707461186745432274),
        (30, 0.8, 1.135922354981720114641358096730123317794),
        (180, 0.9, 1.07387108953435366475669999057569479912),
    ],
)
def test_an_sd_s_upper_limit_is_maxima_s(worlds, confidence, ratio):
    """sqrt((m - 1) / chi2_(1 - confidence, m - 1)), the quantile by bisection on cdf_chi2 at 40
    digits in Maxima 5.50."""
    assert claim.upper_sd(0.3, worlds, confidence) == pytest.approx(0.3 * ratio, rel=1e-12, abs=0.0)


def test_an_sd_s_upper_limit_covers_the_true_sd_at_its_confidence_on_normal_values():
    replicates, worlds = 100_000, 30
    rng = np.random.default_rng(11)
    sds = rng.normal(0.0, 2.0, size=(replicates, worlds)).std(axis=1, ddof=1)
    covered = np.mean(claim.upper_sd(1.0, worlds) * sds >= 2.0)
    assert covered == pytest.approx(0.8, rel=0.0, abs=3 * math.sqrt(0.8 * 0.2 / replicates))


@pytest.mark.parametrize(("worlds", "confidence"), [(1, 0.8), (30, 0.0), (30, 1.0)])
def test_an_sd_s_upper_limit_refuses_what_it_cannot_read(worlds, confidence):
    with pytest.raises(ValueError, match="upper limit needs"):
        claim.upper_sd(1.0, worlds, confidence)


def test_the_joint_size_on_the_upper_limits_raises_every_difference_by_one_ratio():
    pilot = dataclasses.replace(SCORED, worlds=20, pilot=True, default=None, margin=None)
    spreads = dict(zip(RIVALS, (0.1, 0.2, 0.25, 0.05), strict=True))
    records = _records(_differences(dict.fromkeys(RIVALS, 0.0), spreads), pilot=True)
    plain = claim.joint_size(records, pilot, DEFAULT, 0.05, resamples=2000)
    raised = claim.joint_size(records, pilot, DEFAULT, 0.05, resamples=2000, confidence=0.8)
    assert plain.scale == 1.0
    assert raised.scale == claim.upper_sd(1.0, 40, 0.8)
    assert raised.needed > plain.needed
    scaled = claim.differences(records, pilot, DEFAULT)
    scaled = claim.Differences(scaled.comparisons, tuple(raised.scale * s for s in scaled.strata))
    assert claim.joint_needed(scaled, 0.05, resamples=2000)[0] == raised.needed


def test_the_command_line_reads_a_confidence_only_with_the_joint_size(
    tmp_path, capsys, monkeypatch
):
    pilot = dataclasses.replace(SCORED, worlds=20, pilot=True, default=None, margin=None)
    spreads = dict(zip(RIVALS, (0.1, 0.2, 0.25, 0.05), strict=True))
    _write(
        Layout(tmp_path / "pilot"),
        pilot,
        _records(_differences(dict.fromkeys(RIVALS, 0.0), spreads), pilot=True),
    )
    command = ["claim", "size", "--default", DEFAULT, "--margin", "0.05", "--confidence", "0.8"]
    monkeypatch.setattr(sys, "argv", [*command, str(tmp_path / "pilot")])
    with pytest.raises(SystemExit):
        claim.main()
    assert "it needs --joint" in capsys.readouterr().err
    monkeypatch.setattr(
        sys, "argv", [*command, "--joint", "--resamples", "2000", str(tmp_path / "pilot")]
    )
    claim.main()
    line = capsys.readouterr().out.splitlines()[1]
    assert line.endswith(
        f"; the pilot's SDs raised by {claim.upper_sd(1.0, 40, 0.8):.4f}, to their 0.8 upper limits"
    )


MAXIMA_TEN = 0.1073741824
MAXIMA_EACH_OF_TEN = 0.9779327685429285101017642069566420499628
"""0.8^10 and 0.8^(1/10), by Maxima 5.50 at 40 digits."""


def test_independent_families_at_the_power_together_reach_its_power_to_their_count():
    assert claim.overall([claim.POWER] * 10) == pytest.approx(MAXIMA_TEN, rel=1e-14, abs=0.0)
    assert claim.overall([0.9, 0.5]) == pytest.approx(0.45, rel=1e-15, abs=0.0)
    assert claim.overall([]) == 1.0


def test_each_family_sized_to_the_power_s_root_brings_the_launch_claim_to_the_power():
    assert claim.each_family(1) == claim.POWER
    assert claim.each_family(10) == pytest.approx(MAXIMA_EACH_OF_TEN, rel=1e-14, abs=0.0)
    for families in range(1, 15):
        each = claim.each_family(families)
        assert claim.overall([each] * families) == pytest.approx(claim.POWER, rel=1e-14, abs=0.0)


@pytest.mark.parametrize(("families", "power"), [(0, 0.8), (3, 1.0), (3, 0.0)])
def test_each_family_refuses_what_it_cannot_read(families, power):
    with pytest.raises(ValueError, match="needs a family and a power"):
        claim.each_family(families, power)


def test_the_joint_n_for_one_family_is_today_s_and_for_ten_reaches_the_tenth_root():
    pilot = dataclasses.replace(SCORED, worlds=20, pilot=True, default=None, margin=None)
    spreads = dict(zip(RIVALS, (0.1, 0.2, 0.25, 0.05), strict=True))
    records = _records(_differences(dict.fromkeys(RIVALS, 0.0), spreads), pilot=True)
    today = claim.joint_size(records, pilot, DEFAULT, 0.05, resamples=2000)
    one = claim.joint_size(
        records, pilot, DEFAULT, 0.05, resamples=2000, power=claim.each_family(1)
    )
    ten = claim.joint_size(
        records, pilot, DEFAULT, 0.05, resamples=2000, power=claim.each_family(10)
    )
    assert one == today
    assert ten.needed > today.needed
    assert ten.joint >= claim.each_family(10)


def test_the_command_line_sizes_each_family_for_the_launch_claim_when_asked(
    tmp_path, capsys, monkeypatch
):
    pilot = dataclasses.replace(SCORED, worlds=20, pilot=True, default=None, margin=None)
    spreads = dict(zip(RIVALS, (0.1, 0.2, 0.25, 0.05), strict=True))
    _write(
        Layout(tmp_path / "pilot"),
        pilot,
        _records(_differences(dict.fromkeys(RIVALS, 0.0), spreads), pilot=True),
    )
    command = ["claim", "size", "--default", DEFAULT, "--margin", "0.05", "--resamples", "2000"]
    monkeypatch.setattr(sys, "argv", [*command, "--families", "3", str(tmp_path / "pilot")])
    with pytest.raises(SystemExit):
        claim.main()
    assert "--families sizes the joint N: it needs --joint" in capsys.readouterr().err
    outs = []
    for families in ([], ["--families", "1"], ["--families", "10"]):
        argv = [*command, "--joint", *families, str(tmp_path / "pilot")]
        monkeypatch.setattr(sys, "argv", argv)
        claim.main()
        outs.append(capsys.readouterr().out.splitlines())
    today, one, ten = outs
    assert one == today
    assert ten[1].endswith("; sized to 0.9779, 0.8 over 10 families")
    assert int(ten[1].split("joint N ")[1].split(";")[0].split(",")[0]) > int(
        today[1].split("joint N ")[1].split(";")[0].split(",")[0]
    )
    monkeypatch.setattr(sys, "argv", [*command, "--joint", "--families", "0", str(tmp_path)])
    with pytest.raises(SystemExit):
        claim.main()
    assert "1 or more: 0" in capsys.readouterr().err


MAXIMA_CAPPED = 0.5062962639199799813451914904728452724766
MAXIMA_THREE_AND_CAPPED = 0.2632603842451378363406783532319194218951
"""Phi(0.2 sqrt(100) - t_0.975,99), one comparison of SD 1 at a true tie against a margin of 0.2 on
100 worlds, and its product with MAXIMA_ONE cubed, by Maxima 5.50 at 40 digits."""


def test_a_capped_comparison_passes_with_its_cap_s_chance_and_bounds_the_joint_one():
    pilot = _independent(4)
    each, joint = claim.chances(pilot, 0.2, 100, capped={"r0": 50})
    one = math.sqrt(MAXIMA_ONE * (1 - MAXIMA_ONE) / claim.RESAMPLES)
    capped = math.sqrt(MAXIMA_CAPPED * (1 - MAXIMA_CAPPED) / claim.RESAMPLES)
    both = math.sqrt(MAXIMA_THREE_AND_CAPPED * (1 - MAXIMA_THREE_AND_CAPPED) / claim.RESAMPLES)
    assert each[0] == pytest.approx(MAXIMA_CAPPED, rel=0.0, abs=4 * capped)
    assert each[1:] == pytest.approx([MAXIMA_ONE] * 3, rel=0.0, abs=4 * one)
    assert joint == pytest.approx(MAXIMA_THREE_AND_CAPPED, rel=0.0, abs=4 * both)
    assert (
        claim.chances(pilot, 0.2, 100, capped={"r0": 50})[0][0]
        == (claim.chances(pilot, 0.2, 50)[0][0])
    )
    assert claim.chances(pilot, 0.2, 100, capped={"r0": 100}) == claim.chances(pilot, 0.2, 100)
    assert claim.chances(pilot, 0.2, 100, capped={}) == claim.chances(pilot, 0.2, 100)


def test_an_exploratory_capped_comparison_leaves_the_joint_chance_and_the_joint_n():
    pilot = _independent(4)
    rest = claim.Differences(pilot.comparisons[1:], tuple(s[:, 1:] for s in pilot.strata))
    assert claim.joint_needed(pilot, 0.2, capped={"r0": 50}, exploratory=True) == (
        claim.joint_needed(rest, 0.2)[0],
        (claim.chances(pilot, 0.2, 50)[0][0], *claim.joint_needed(rest, 0.2)[1]),
        claim.joint_needed(rest, 0.2)[2],
    )
    with pytest.raises(ValueError, match=r"capped comparisons together pass with chance 0\.5"):
        claim.joint_needed(pilot, 0.2, capped={"r0": 50})
    n, each, joint = claim.joint_needed(pilot, 0.2, capped={"r0": 150})
    assert n > claim.joint_needed(pilot, 0.2)[0]
    assert joint >= claim.POWER
    assert each[0] == claim.chances(pilot, 0.2, 150)[0][0]


@pytest.mark.parametrize(
    ("capped", "match"),
    [({"r9": 20}, r"\['r9'\]"), ({"r0": 1}, "'r0': 1")],
)
def test_a_cap_is_refused_on_a_rival_the_pilot_does_not_read_or_below_two_worlds(capped, match):
    with pytest.raises(ValueError, match=match):
        claim.chances(_independent(2, worlds=50), 0.2, 20, capped=capped)


def test_every_comparison_capped_and_exploratory_leaves_no_claim():
    pilot = _independent(1, worlds=50)
    with pytest.raises(ValueError, match="the claim reads none"):
        claim.chances(pilot, 0.2, 20, capped={"r0": 10}, exploratory=True)


def test_joint_size_states_what_a_capped_tool_costs_the_family():
    pilot = dataclasses.replace(SCORED, worlds=20, pilot=True, default=None, margin=None)
    spreads = dict(zip(RIVALS, (0.1, 0.2, 0.25, 0.05), strict=True))
    records = _records(_differences(dict.fromkeys(RIVALS, 0.0), spreads), pilot=True)
    today = claim.joint_size(records, pilot, DEFAULT, 0.05, resamples=2000)
    assert (today.capped, today.exploratory_caps, today.whole) == ((), False, today.joint)
    capped = claim.joint_size(
        records, pilot, DEFAULT, 0.05, resamples=2000, capped={"meridian": 20}, exploratory=True
    )
    assert capped.capped == (("meridian", 20),)
    assert capped.exploratory_caps
    assert capped.needed <= today.needed
    assert capped.whole is not None
    assert capped.whole < capped.joint
    count = math.ceil(capped.needed / 2)
    differences = claim.differences(records, pilot, DEFAULT)
    assert (
        capped.whole
        == claim.chances(differences, 0.05, count, resamples=2000, capped={"meridian": 20})[1]
    )
    assert capped.chances[2] == claim.chances(differences, 0.05, 20, resamples=2000)[0][2]


def test_the_command_line_reads_caps_only_with_the_joint_size_and_marks_them(
    tmp_path, capsys, monkeypatch
):
    pilot = dataclasses.replace(SCORED, worlds=20, pilot=True, default=None, margin=None)
    spreads = dict(zip(RIVALS, (0.1, 0.2, 0.25, 0.05), strict=True))
    _write(
        Layout(tmp_path / "pilot"),
        pilot,
        _records(_differences(dict.fromkeys(RIVALS, 0.0), spreads), pilot=True),
    )
    command = ["claim", "size", "--default", DEFAULT, "--margin", "0.05", "--resamples", "2000"]
    for extra, message in (
        (["--cap", "meridian=20"], "they need --joint"),
        (["--joint", "--exploratory-caps"], "name them by --cap"),
        (["--joint", "--cap", "meridian"], "a cap is TOOL=WORLDS"),
    ):
        monkeypatch.setattr(sys, "argv", [*command, *extra, str(tmp_path / "pilot")])
        with pytest.raises(SystemExit):
            claim.main()
        assert message in capsys.readouterr().err
    extra = ["--joint", "--cap", "meridian=20", "--exploratory-caps"]
    monkeypatch.setattr(sys, "argv", [*command, *extra, str(tmp_path / "pilot")])
    claim.main()
    lines = capsys.readouterr().out.splitlines()
    assert (
        "; meridian on 20 worlds an environment, exploratory: with them every comparison"
        in (lines[1])
    )
    assert lines[4].startswith("  less meridian")
    assert lines[4].endswith(" on 20 worlds an environment; exploratory")
    assert not lines[3].endswith("exploratory")


EXISTING = (
    '{"family": 0, "environments": ["drawn"], "worlds": 3, "pilot": true, "rungs": [0], "arms": '
    '["bayes", "dlm", "dlm-untested", "chc"], "external": [], "smoke": true, "bench": null, '
    '"limits": {"bayes": {"memory": "4G", "timeout": 3600.0, "slots": 2}, "dlm": {"memory": "6G", '
    '"timeout": 3600.0, "slots": 2}, "dlm-untested": {"memory": "6G", "timeout": 3600.0, "slots": '
    '2}, "chc": {"memory": "3G", "timeout": 1800.0, "slots": 2}}, "reserve": "2G", "settle": '
    '20.0, "attempts": 2, "r_library": null, "nevergrad": null, "world_directory": null, '
    '"default": null, "margin": null, "exploratory": false}'
)
"""A run's declaration as the harness wrote it before caps, byte for byte."""


def test_a_declaration_written_before_caps_reads_and_is_declared_again_unchanged(tmp_path):
    layout = Layout(tmp_path)
    layout.declaration.write_text(EXISTING)
    declared = layout.read()
    assert (declared.caps, declared.exploratory_caps) == (None, False)
    assert layout.declare(declared) == declared
    again = Layout(tmp_path / "again")
    again.declare(declared)
    assert again.declaration.read_text() == layout.declaration.read_text() == EXISTING


def test_a_declaration_s_caps_are_written_and_read_back(tmp_path):
    capped = dataclasses.replace(SCORED, caps={"meridian": 10}, exploratory_caps=True)
    layout = Layout(tmp_path)
    layout.declare(capped)
    stored = json.loads(layout.declaration.read_text())
    assert (stored["caps"], stored["exploratory_caps"]) == ({"meridian": 10}, True)
    assert layout.read() == capped
    assert layout.declare(capped) == capped
    with pytest.raises(RuntimeError, match="is declared as"):
        layout.declare(dataclasses.replace(capped, caps={"meridian": 12}))
    with pytest.raises(RuntimeError, match="is declared as"):
        layout.declare(SCORED)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"caps": {"dlm plug-in": 10}}, "one of the run's tools"),
        ({"caps": {"status quo": 10}}, "one of the run's tools"),
        ({"caps": {"meridian": 1}}, "2 worlds or more"),
        ({"caps": {"meridian": True}}, "2 worlds or more"),
        ({"caps": {"meridian": 10.0}}, "2 worlds or more"),
        ({"exploratory_caps": True}, "declare their caps"),
        ({"caps": {}, "exploratory_caps": True}, "declare their caps"),
    ],
)
def test_a_cap_the_run_cannot_read_is_refused(change, match):
    with pytest.raises(ValueError, match=match):
        dataclasses.replace(SCORED, **change)


def test_a_capped_tool_is_read_on_its_cap_s_prefix_and_the_others_on_every_world():
    means, spreads = _known()
    records = _records(_differences(means, spreads))
    capped = dataclasses.replace(SCORED, caps={"meridian": 10, "orbit": 30})
    prefix = [r for r in records if r.index < 10]
    by = {c.rival: c for c in compare(records, capped)}
    whole = {c.rival: c for c in compare(records, SCORED)}
    alone = {c.rival: c for c in compare(prefix, dataclasses.replace(SCORED, worlds=10))}
    assert by["meridian"] == dataclasses.replace(alone["meridian"], cap=10)
    assert by["meridian"] != whole["meridian"]
    assert by["orbit"] == dataclasses.replace(whole["orbit"], cap=20)
    for rival in ("status quo", "equal split"):
        assert by[rival] == whole[rival]
    assert not any(c.exploratory for c in by.values())
    line = next(x for x in claim.lines(list(by.values()), 0.05) if "less meridian" in x)
    assert line.endswith("; on its cap, 10 worlds an environment")


def test_a_capped_tool_is_read_once_its_prefix_is_whole():
    records = _records(_differences(dict.fromkeys(RIVALS, -0.1), dict.fromkeys(RIVALS, 0.1)))
    prefix = [r for r in records if r.index < 10]
    by = {c.rival: c for c in compare(prefix, dataclasses.replace(SCORED, caps={"meridian": 10}))}
    assert by["meridian"].within
    assert [by[r].within for r in ("status quo", "equal split", "orbit")] == [None] * 3
    assert by["meridian"].cap == 10
    lines = claim.lines(list(by.values()), 0.05)
    unread = [i for i, line in enumerate(lines) if "unread: the run does not hold" in line]
    assert unread == [0, 1, 3]
    assert lines[2].endswith("; on its cap, 10 worlds an environment")


def test_exploratory_caps_leave_the_capped_comparisons_out_of_the_claim_and_holm_s_list():
    means, spreads = _known()
    records = _records(_differences(means, spreads))
    capped = dataclasses.replace(SCORED, caps={"orbit": 20}, exploratory_caps=True)
    comparisons = compare(records, capped)
    by = {c.rival: c for c in comparisons}
    assert [by[r].exploratory for r in RIVALS] == [False, False, False, True]
    assert losses(comparisons) == {(1, 0, "equal split")}
    assert losses(compare(records, SCORED)) == {(1, 0, "equal split"), (1, 0, "orbit")}
    kept = dataclasses.replace(by["meridian"], within=True)
    assert holds([by["status quo"], by["equal split"], kept, by["orbit"]])
    assert not holds([by["status quo"], by["equal split"], by["meridian"], by["orbit"]])
    claimed = dataclasses.replace(capped, exploratory_caps=False)
    assert not next(c for c in compare(records, claimed) if c.rival == "orbit").exploratory


def test_the_joint_sizing_reads_a_pilot_s_declared_caps_as_it_reads_the_command_line_s(
    tmp_path, capsys, monkeypatch
):
    pilot = dataclasses.replace(SCORED, worlds=20, pilot=True, default=None, margin=None)
    spreads = dict(zip(RIVALS, (0.1, 0.2, 0.25, 0.05), strict=True))
    records = _records(_differences(dict.fromkeys(RIVALS, 0.0), spreads), pilot=True)
    _write(Layout(tmp_path / "bare"), pilot, records)
    declared = dataclasses.replace(pilot, caps={"meridian": 20}, exploratory_caps=True)
    _write(Layout(tmp_path / "capped"), declared, records)
    command = ["claim", "size", "--default", DEFAULT, "--margin", "0.05", "--resamples", "2000"]
    named = ["--joint", "--cap", "meridian=20", "--exploratory-caps", str(tmp_path / "bare")]
    monkeypatch.setattr(sys, "argv", [*command, *named])
    claim.main()
    by_flags = capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", [*command, "--joint", str(tmp_path / "capped")])
    claim.main()
    by_declaration = capsys.readouterr().out
    assert by_declaration == by_flags
    assert "meridian on 20 worlds an environment, exploratory" in by_declaration
    in_claim = dataclasses.replace(declared, exploratory_caps=False)
    _write(Layout(tmp_path / "in-claim"), in_claim, records)
    replaced = ["--joint", "--cap", "meridian=40", "--exploratory-caps", str(tmp_path / "in-claim")]
    monkeypatch.setattr(sys, "argv", [*command, *replaced])
    claim.main()
    assert "meridian on 40 worlds an environment, exploratory" in capsys.readouterr().out


def test_the_runner_declares_a_tool_s_cap(tmp_path, monkeypatch):
    root = tmp_path / "run"
    command = [
        "run", "declare", "--root", str(root), "--family", "1", "--environments", *LAWS,
        "--worlds", "20", "--rungs", "0", "--arms", "dlm", "--external", "meridian", "orbit",
        "--bench", "1f4161c", "--default", DEFAULT, "--margin", "0.05",
        "--limit", "dlm", "1G", "60", "1", "--limit", "meridian", "1G", "60", "1",
        "--limit", "orbit", "1G", "60", "1",
        "--cap", "meridian=10", "--exploratory-caps",
        "--preregistration", str(_design(tmp_path)),
    ]  # fmt: skip
    monkeypatch.setattr(sys, "argv", command)
    run.main()
    declared = Layout(root).read()
    assert (declared.caps, declared.exploratory_caps) == ({"meridian": 10}, True)
    assert declared.worlds_of("meridian") == 10
    assert declared.worlds_of("orbit") == 20


def _failing(records, arm, where):
    """`records` with `arm`'s plan failed on the worlds `where` picks, scored as the status quo."""
    out = []
    for r in records:
        if where(r):
            score = dataclasses.replace(r.arms[arm], regret=1.5, failure="RuntimeError: no fit")
            r = dataclasses.replace(r, arms={**r.arms, arm: score})
        out.append(r)
    return out


def _paired_t(differences):
    d = np.asarray(differences, dtype=float)
    half = float(student.ppf(0.975, d.size - 1)) * d.std(ddof=1) / math.sqrt(d.size)
    return d.mean(), (d.mean() - half, d.mean() + half)


def test_a_comparison_reads_its_complete_cases_and_the_fallback_beside_them():
    differences = _differences(dict.fromkeys(RIVALS, -0.1), dict.fromkeys(RIVALS, 0.1))
    whole = _records(differences)
    failed = {(LAWS[0], 3), (LAWS[0], 11), (LAWS[1], 7)}
    records = _failing(whole, "orbit", lambda r: (r.environment, r.index) in failed)
    records = _failing(records, DEFAULT, lambda r: (r.environment, r.index) == (LAWS[1], 2))
    by = {c.rival: c for c in compare(records, SCORED)}
    orbit = by["orbit"]
    kept = [row for row in range(40) if (LAWS[row // 20], row % 20) not in failed | {(LAWS[1], 2)}]
    mean, interval = _paired_t([differences["orbit"][row] for row in kept])
    assert (orbit.pairs, orbit.failures, orbit.default_failures) == (36, 3, 1)
    assert orbit.left_out == ()
    assert orbit.compared
    assert orbit.mean == pytest.approx(mean, rel=0.0, abs=1e-12)
    assert orbit.interval == pytest.approx(interval, rel=0.0, abs=1e-12)
    fallback = [
        (1.5 if (LAWS[row // 20], row % 20) == (LAWS[1], 2) else 1.0)
        - (1.5 if (LAWS[row // 20], row % 20) in failed else 1.0 - differences["orbit"][row])
        for row in range(40)
    ]
    mean, interval = _paired_t(fallback)
    assert orbit.fallback_mean == pytest.approx(mean, rel=0.0, abs=1e-12)
    assert orbit.fallback_interval == pytest.approx(interval, rel=0.0, abs=1e-12)
    assert by["meridian"].pairs == 39
    assert "36 complete cases, the rival failed 3, the default 1" in "\n".join(
        claim.lines(list(by.values()), 0.05)
    )


def test_an_environment_where_the_rival_fails_on_most_worlds_is_not_compared():
    differences = _differences(dict.fromkeys(RIVALS, -0.1), dict.fromkeys(RIVALS, 0.1))
    records = _failing(
        _records(differences), "orbit", lambda r: r.environment == LAWS[0] and r.index <= 10
    )
    (orbit,) = (c for c in compare(records, SCORED) if c.rival == "orbit")
    assert orbit.left_out == (LAWS[0],)
    assert orbit.compared
    assert (orbit.pairs, orbit.failures) == (20, 11)
    mean, interval = _paired_t(differences["orbit"][20:])
    assert orbit.mean == pytest.approx(mean, rel=0.0, abs=1e-12)
    assert orbit.interval == pytest.approx(interval, rel=0.0, abs=1e-12)
    half = _failing(
        _records(differences), "orbit", lambda r: r.environment == LAWS[0] and r.index < 10
    )
    (orbit,) = (c for c in compare(half, SCORED) if c.rival == "orbit")
    assert orbit.left_out == ()
    assert orbit.pairs == 30


def test_a_rival_that_fails_on_most_worlds_everywhere_leaves_the_claim():
    within = dict.fromkeys(RIVALS, -0.1)
    records = _failing(
        _records(_differences(within, dict.fromkeys(RIVALS, 0.1))), "orbit", lambda r: r.index < 15
    )
    comparisons = compare(records, SCORED)
    (orbit,) = (c for c in comparisons if c.rival == "orbit")
    assert not orbit.compared
    assert orbit.left_out == tuple(sorted(LAWS))
    assert orbit.pairs == 0
    assert orbit.within is None
    assert holds(comparisons)
    assert any("not compared" in line for line in claim.lines(comparisons, 0.05))
