"""The pre-registration a run runs under: declared with the SHA-256 of its file, required of a
scored run, held to its file whenever the run is reported, and printed as the bench's stamp; a run
that declares none reports as it did before runs declared one. Its export and its arms do not
start once the file moved. And the bench's report of the run's family, asked for: its JSON and its
markdown carry the stamp a scored run declares."""

import dataclasses
import hashlib
import json
import sys

import pytest

pytest.importorskip("causaldyn_bench")

from causaldyn_bench.scorecard import report as bench_report
from causaldyn_bench.scorecard.preregistration import Stamp, verify
from test_claim import SCORED, _differences, _known, _records

from scorecard import provenance, report, run
from scorecard.layout import TREE, Declaration, Layout, Limits, stamp

OURS = ["dlm plug-in", "dlm mixture"]
PILOT = dataclasses.replace(SCORED, worlds=10, pilot=True, bench=None)
PILOT_REPORT = "226455fec357450f15deccdcaa9bc34784af355184265b2f65ff8a4bc832ec55"
"""The SHA-256 of `PILOT`'s report, its root written `<run>`, printed by the code before runs
declared a pre-registration. It prints the bench's notes on the family, so the digest moves with
them: family 1's note on its cpm environment moved it."""


def _design(tmp_path, text="the design, fixed before the first scored world\n"):
    path = tmp_path / "design.md"
    path.write_text(text)
    return path


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(root, declaration, *, pilot=False):
    """`declaration` under `root` beside its scores, each rival of `_known` on its worlds."""
    layout = Layout(root)
    layout.declare(declaration)
    records = _records(
        _differences(*_known(), n=2 * declaration.worlds), worlds=declaration.worlds, pilot=pilot
    )
    layout.scores.write_text("".join(json.dumps(r.as_json()) + "\n" for r in records))
    return layout, records


class _Held:
    variants = ("held",)
    tests = True
    cells = False


def _printed(capsys, layout):
    return capsys.readouterr().out.replace(str(layout.root), "<run>")


def test_a_stamp_is_the_files_sha256_named_as_the_engines_tree_names_it(tmp_path, monkeypatch):
    path = _design(tmp_path)
    assert stamp(path) == Stamp(str(path), _sha256(path))
    monkeypatch.chdir(tmp_path)
    assert stamp("design.md") == Stamp(str(path), _sha256(path))  # read from where it is declared
    kept = TREE / "scorecard" / "preregistration" / "drift.md"
    held = stamp(kept)
    assert held == Stamp("scorecard/preregistration/drift.md", _sha256(kept))
    monkeypatch.chdir(TREE / "scorecard")
    assert dataclasses.replace(PILOT, preregistration=held).preregistered() == held
    with pytest.raises(FileNotFoundError):
        stamp(tmp_path / "absent.md")


def test_a_scored_run_is_declared_only_with_its_preregistration(tmp_path, monkeypatch):
    monkeypatch.setitem(run.ARMS, "held", _Held)
    path = _design(tmp_path)
    root = tmp_path / "run"
    command = ["run", "declare", "--root", str(root), "--family", "0", "--environments", "drawn"]
    command += ["--worlds", "2", "--rungs", "0", "--arms", "held", "--limit", "held", "1G", "60"]
    command += ["1"]
    monkeypatch.setattr(sys, "argv", [*command, "--bench", "abc1234"])
    with pytest.raises(ValueError, match="a scored run names the pre-registration it runs under"):
        run.main()
    assert not Layout(root).declaration.exists()
    scored = [*command, "--bench", "abc1234", "--preregistration", str(path)]
    monkeypatch.setattr(sys, "argv", scored)
    run.main()
    held = json.loads(Layout(root).declaration.read_text())["preregistration"]
    assert held == {"path": str(path), "sha256": _sha256(path)}
    assert Layout(root).read().preregistration == Stamp(str(path), _sha256(path))
    run.main()  # declared again under the same file: the run already holds that declaration
    path.write_text("a design that moved after the run was declared\n")
    with pytest.raises(RuntimeError, match="is declared as"):
        run.main()
    monkeypatch.setattr(sys, "argv", [*command, "--pilot", "--root", str(tmp_path / "pilot")])
    run.main()
    assert "preregistration" not in json.loads(Layout(tmp_path / "pilot").declaration.read_text())
    stale = dataclasses.replace(Layout(root).read(), preregistration=Stamp(str(path), "0" * 64))
    with pytest.raises(ValueError, match="the file there is"):
        run.declare(Layout(tmp_path / "stale"), stale)


def test_a_run_that_declares_no_preregistration_reports_as_it_did_to_the_byte(tmp_path, capsys):
    layout, _ = _run(tmp_path / "run", PILOT, pilot=True)
    report.main(layout, OURS)
    printed = _printed(capsys, layout)
    assert "pre-registration" not in printed
    assert hashlib.sha256(printed.encode()).hexdigest() == PILOT_REPORT


def test_the_export_and_the_arms_do_not_start_once_the_file_moved(tmp_path, monkeypatch):
    monkeypatch.setitem(run.ARMS, "held", _Held)
    path = _design(tmp_path)
    layout = Layout(tmp_path / "run")
    declared = Declaration(
        family=0,
        environments=("drawn",),
        worlds=1,
        pilot=True,
        rungs=(0,),
        arms=("held",),
        limits={"held": Limits(memory="1G", timeout=60.0, slots=1)},
        settle=0.0,
        preregistration=stamp(path),
    )
    run.declare(layout, declared)
    provenance.require(bench=None, preregistration=declared.preregistration)  # as declared
    path.write_text("a design that moved after the run was declared\n")
    for step in (run.export_all, lambda out: run.run_arm(out, "held")):
        with pytest.raises(RuntimeError, match="the pre-registration is not the one declared"):
            step(layout)
    path.unlink()
    with pytest.raises(RuntimeError, match="No such file"):
        run.export_all(layout)
    assert not (layout.root / "exports").exists()
    assert not (layout.root / "records").exists()


def test_a_declared_preregistration_is_printed_below_the_first_line(tmp_path, capsys):
    path = _design(tmp_path)
    bare, _ = _run(tmp_path / "bare", PILOT, pilot=True)
    report.main(bare, OURS)
    without = _printed(capsys, bare).splitlines()
    held = stamp(path)
    layout, _ = _run(
        tmp_path / "held", dataclasses.replace(PILOT, preregistration=held), pilot=True
    )
    report.main(layout, OURS)
    printed = _printed(capsys, layout)
    assert Stamp.from_line(printed) == held
    lines = printed.splitlines()
    assert lines[1] == f"pre-registration {path}, SHA-256 {_sha256(path)}"
    assert lines[:1] + lines[2:] == without


def test_a_scored_runs_report_is_refused_where_its_file_moved_or_is_gone(tmp_path, capsys):
    path = _design(tmp_path)
    declared = dataclasses.replace(SCORED, preregistration=stamp(path))
    layout, _ = _run(tmp_path / "run", declared)
    report.main(layout, OURS)
    assert Stamp.from_line(capsys.readouterr().out) == declared.preregistration
    path.write_text("a design that moved after the run was declared\n")
    for columns in (False, True):
        with pytest.raises(ValueError, match="the file there is"):
            report.main(layout, OURS, columns=columns)
        assert capsys.readouterr().out == ""  # refused before a line is printed
    path.unlink()
    with pytest.raises(FileNotFoundError):
        report.main(layout, OURS)
    assert capsys.readouterr().out == ""


def test_the_bench_s_report_of_a_scored_run_carries_its_stamp(tmp_path, capsys, monkeypatch):
    path = _design(tmp_path)
    declared = dataclasses.replace(SCORED, preregistration=stamp(path))
    layout, records = _run(tmp_path / "run", declared)
    report.main(layout, OURS)
    alone = capsys.readouterr().out
    saved = tmp_path / "report" / "family-1.json"
    report.main(layout, OURS, columns=True, saved=saved)
    printed = capsys.readouterr().out
    assert printed.startswith(alone)
    data = json.loads(saved.read_text())
    written = saved.with_suffix(".md").read_text()
    assert printed == alone + "\n" + written
    assert data["kind"] == "scored"
    assert Stamp.from_json(data["preregistration"]) == declared.preregistration
    assert Stamp.from_line(written) == declared.preregistration
    verify(Stamp.from_line(written), TREE)
    own = bench_report.report(declared.source(), records, preregistration=path)
    assert data == json.loads(json.dumps(own))
    assert written == bench_report.markdown(own)
    with pytest.raises(ValueError, match=r"written to a \.json path"):
        report.main(layout, OURS, saved=tmp_path / "family-1.md")
    kept = TREE / "scorecard" / "preregistration" / "drift.md"
    inside = dataclasses.replace(SCORED, preregistration=stamp(kept))
    layout, _ = _run(tmp_path / "inside", inside)
    monkeypatch.chdir(tmp_path)  # the tree's file, named as the tree names it, read from anywhere
    report.main(layout, OURS, saved=tmp_path / "inside.json")
    held = json.loads((tmp_path / "inside.json").read_text())["preregistration"]
    assert held == {"path": "scorecard/preregistration/drift.md", "sha256": _sha256(kept)}
    assert Stamp.from_line((tmp_path / "inside.md").read_text()) == inside.preregistration


def test_the_bench_refuses_the_family_report_of_a_scored_run_that_declares_no_file(
    tmp_path, capsys
):
    layout, _ = _run(tmp_path / "run", SCORED)
    report.main(layout, OURS)
    alone = capsys.readouterr().out
    with pytest.raises(ValueError, match="prints the pre-registration the run declares"):
        report.main(layout, OURS, columns=True)
    assert capsys.readouterr().out == alone
