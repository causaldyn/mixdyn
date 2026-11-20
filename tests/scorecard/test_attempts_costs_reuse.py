"""What a run keeps of its processes and takes from another run: every attempt written down as it
starts and as it ends, its log appended to, each process's cost appended as it ends, the CPUs a
tool's process reserves declared with its limits, and a tool's records taken from another run's
root where the world is the observation the tool read and the tool is pinned alike."""

import dataclasses
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("causaldyn_bench")

import causaldyn_bench
from causaldyn_bench.scorecard.observe import read
from test_runner import FAKE, Still, scopes

from scorecard import run
from scorecard.layout import Declaration, Layout, Limits
from scorecard.provenance import tree
from scorecard.tools import RESERVED, TOOLS, Tool, fingerprint

MODES = ("record", "exit", "half")


def _declared(**change):
    held = Declaration(
        family=0,
        environments=("drawn",),
        worlds=len(MODES),
        pilot=True,
        rungs=(0,),
        arms=("still",),
        external=("fake",),
        limits={
            "still": Limits(memory="1G", timeout=60.0, slots=1),
            "fake": Limits(memory="256M", timeout=30.0, slots=3),
        },
        settle=0.0,
    )
    return dataclasses.replace(held, **change)


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """The tool `fake` over a world for each of `MODES`; the run and each mode's seed."""
    modes = tmp_path / "modes.json"
    monkeypatch.setitem(run.ARMS, "still", Still)
    monkeypatch.setitem(TOOLS, "fake", Tool((sys.executable, str(FAKE), "--modes", str(modes))))
    out = Layout(tmp_path / "run")
    run.declare(out, _declared())
    run.export_all(out)
    seeds = dict(zip(MODES, out.read().seeds("drawn"), strict=True))
    modes.write_text(json.dumps({str(seed): mode for mode, seed in seeds.items()}))
    return out, seeds


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@scopes
def test_each_attempt_is_written_down_as_it_starts_and_ends_and_its_log_appended_to(fake):
    out, seeds = fake
    log = out.root / "logs" / "fake" / f"drawn-k0-{seeds['exit']}.log"
    log.parent.mkdir(parents=True)
    log.write_text("a line before the run\n")
    run.run_arm(out, "fake")
    attempts = _lines(out.attempts)
    for mode, seed in seeds.items():
        mine = [a for a in attempts if a["seed"] == seed]
        starts = [a for a in mine if a["event"] == "start"]
        ends = [a for a in mine if a["event"] == "end"]
        tried = 1 if mode == "record" else 2
        assert (
            [a["attempt"] for a in starts] == [a["attempt"] for a in ends] == [*range(1, tried + 1)]
        )
        assert all(
            a["host"] and a["began"] <= e["ended"] for a, e in zip(starts, ends, strict=True)
        )
        assert all(a["log"] == str(log.with_name(f"drawn-k0-{seed}.log")) for a in mine)
    exited = [a["outcome"] for a in attempts if a["seed"] == seeds["exit"] and a["event"] == "end"]
    assert exited == ["exit code 3", "exit code 3"]
    text = log.read_text()
    assert text.startswith("a line before the run\n--- attempt 1 of 2, began ")
    assert text.index("--- attempt 1 of 2") < text.index("--- attempt 2 of 2")

    costs = _lines(out.costs)
    assert sorted((c["tag"]["seed"], c["tag"]["attempt"]) for c in costs) == sorted(
        (a["seed"], a["attempt"]) for a in attempts if a["event"] == "start"
    )
    for c in costs:
        if c["tag"]["seed"] == seeds["exit"]:
            assert c["exit"] == 3
        assert c["began"] <= c["ended"]
        assert c["cpu_usec"] is None or c["cpu_usec"] >= 0
        assert c["peak_bytes"] is None or c["peak_bytes"] > 0
    # the half-written world: killed by SIGKILL in its first attempt, recorded in its second
    half = sorted(
        (c["tag"]["attempt"], c["exit"]) for c in costs if c["tag"]["seed"] == seeds["half"]
    )
    assert half == [(1, -9), (2, 0)]


def test_the_cost_wrapper_appends_its_process_s_line_and_keeps_its_exit(tmp_path):
    costs = tmp_path / "costs.jsonl"
    tag = json.dumps({"arm": "fake", "seed": 7})
    command = [sys.executable, "-I", "-c", run.COST, str(costs), tag, "/bin/sh", "-c", "exit 7"]
    assert subprocess.run(command, check=False).returncode == 7
    command[-1] = "exit 0"
    assert subprocess.run(command, check=False).returncode == 0
    held = _lines(costs)
    assert [(c["tag"], c["exit"]) for c in held] == [
        ({"arm": "fake", "seed": 7}, e) for e in (7, 0)
    ]
    assert set(held[0]) == {"tag", "exit", "began", "ended", "cpu_usec", "peak_bytes"}
    # a process killed by a signal: the wrapper ends by the same one, as `wait` read it before
    command[-3:] = [sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGSEGV)"]
    assert subprocess.run(command, check=False).returncode == -11
    assert _lines(costs)[-1]["exit"] == -11


def test_a_scope_runs_its_command_by_the_cost_wrapper_and_keeps_m2_s_shape(tmp_path):
    """The scope's argv is M2's, `_scoped(limits, command)` as a stage's driver wraps it, and the
    command it runs is the cost wrapper's, which runs the process."""
    limits = Limits(memory="1G", timeout=60.0, slots=1)
    out = Layout(tmp_path / "run")
    plain = run._scoped(limits, ["true"])
    costed = run._scoped(limits, run._costed(["true"], out, {"seed": 1}))
    assert plain[6:8] == costed[6:8] == ["-p", "MemorySwapMax=0"]
    assert costed[8:12] == [sys.executable, "-I", "-c", run.COST]
    assert costed[12:14] == [str(out.costs), '{"seed": 1}']
    assert costed[14:] == plain[8:] == ["true"]


def test_a_tool_s_cpus_are_declared_with_its_limits_and_one_is_not_written(tmp_path, monkeypatch):
    root = tmp_path / "run"
    command = [
        "run", "declare", "--root", str(root), "--family", "1", "--environments", "walk",
        "--worlds", "2", "--rungs", "0", "--pilot", "--arms", "ridge",
        "--external", "meridian", "orbit", "robyn", "--r-library", "/lib", "--nevergrad", "/ng",
        "--limit", "ridge", "1G", "60", "1", "--limit", "meridian", "1G", "60", "1",
        "--limit", "orbit", "1G", "60", "1", "--limit", "robyn", "1G", "60", "1",
        "--cpus", "orbit", "2",
    ]  # fmt: skip
    monkeypatch.setattr(sys, "argv", command)
    run.main()
    limits = Layout(root).read().limits
    assert {arm: held.cpus for arm, held in limits.items()} == {
        "ridge": 1.0,
        "meridian": RESERVED["meridian"],
        "orbit": 2.0,
        "robyn": RESERVED["robyn"],
    }
    written = json.loads(Layout(root).declaration.read_text())["limits"]
    assert written["meridian"]["cpus"] == 1.2
    assert "cpus" not in written["ridge"]
    monkeypatch.setitem(run.ARMS, "still", Still)
    Layout(tmp_path / "one").declare(_declared())
    assert (
        "cpus" not in json.loads(Layout(tmp_path / "one").declaration.read_text())["limits"]["fake"]
    )
    with pytest.raises(ValueError, match=r"CPUs|cpus"):
        Limits("1G", 60.0, 1, cpus=0.0)


def test_a_tool_s_fingerprint_follows_the_bench_s_scripts_its_files_import():
    files = {
        "scripts/a.py": b"import b, numpy\nfrom c import x\nfrom .d import y\n",
        "scripts/a.txt": b"pkg==1\n",
        "scripts/b.py": b"",
        "scripts/c.py": b"from b import z\n",
        "scripts/d.py": b"",
    }
    held = fingerprint(Tool(("a",), pins=("scripts/a.py", "scripts/a.txt")), files.get)
    assert set(held) == {"scripts/a.py", "scripts/a.txt", "scripts/b.py", "scripts/c.py"}
    moved = fingerprint(Tool(("a",), pins=("scripts/a.py",)), {**files, "scripts/b.py": b"#"}.get)
    assert moved["scripts/b.py"] != held["scripts/b.py"]
    assert fingerprint(Tool(("a",), pins=("scripts/gone.R",)), files.get) == {
        "scripts/gone.R": "absent"
    }


def test_every_tool_s_pinned_files_are_in_the_bench():
    show = run._show(tree(causaldyn_bench), None)
    held = {tool: fingerprint(t, show) for tool, t in TOOLS.items()}
    assert all("absent" not in f.values() for f in held.values())
    common = {"scripts/arm_machine.py", "scripts/geo_export.py"}
    assert set(held["meridian"]) == {
        "scripts/meridian_arm.py", "scripts/meridian_arm.txt", "scripts/meridian_roi.py", *common
    }  # fmt: skip
    assert set(held["pymc-marketing"]) == {
        "scripts/pymc_marketing_arm.py", "scripts/pymc_marketing_arm.txt",
        "scripts/pymc_marketing_lift.py", *common,
    }  # fmt: skip
    assert set(held["orbit"]) == {"scripts/orbit_arm.py", "scripts/orbit_arm.txt", *common}
    assert set(held["robyn"]) == {
        "scripts/robyn_arm.R", "scripts/robyn_arm.lock.txt", "scripts/geo_export.py"
    }  # fmt: skip


@pytest.fixture
def two(tmp_path, monkeypatch):
    """A source run whose `fake` records name its exports' digests but the last world's, and a
    run of the same worlds that holds none."""
    monkeypatch.setitem(run.ARMS, "still", Still)
    monkeypatch.setitem(TOOLS, "fake", Tool(("true",)))
    source, target = Layout(tmp_path / "source"), Layout(tmp_path / "target")
    for layout in (source, target):
        run.declare(layout, _declared())
    run.export_all(source)
    run.export_all(target)
    seeds = list(source.read().seeds("drawn"))
    for i, seed in enumerate(seeds):
        digest = read(source.export(0, "drawn", 0, seed)).digest()
        record = {"version": 2, "seed": seed, "digest": digest if i < 2 else "0", "weekly": [1.0]}
        record["error"] = "RuntimeError: x" if i == 1 else None
        path = source.record("fake", 0, "drawn", 0, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1))
    return source, target, seeds


def test_a_tool_s_records_are_taken_where_the_world_is_the_observation_it_read(two):
    source, target, seeds = two
    held = run.reuse(target, source, ["fake"])
    assert held == {"fake": 2, "fake left": 1}
    for seed in seeds[:2]:
        taken = target.record("fake", 0, "drawn", 0, seed).read_bytes()
        assert taken == source.record("fake", 0, "drawn", 0, seed).read_bytes()
    assert not target.record("fake", 0, "drawn", 0, seeds[2]).exists()
    notes = _lines(target.reused)
    assert [(n["seed"], n["failed"]) for n in notes] == [(seeds[0], False), (seeds[1], True)]
    assert run.reuse(target, source, ["fake"]) == {"fake": 0, "fake left": 1}
    assert len(_lines(target.reused)) == 2


def test_a_tool_pinned_apart_or_another_run_s_worlds_are_refused(two, tmp_path, monkeypatch):
    source, target, _ = two
    monkeypatch.setitem(TOOLS, "fake", Tool(("true",), pins=("scripts/fake.py",)))
    monkeypatch.setattr(
        run, "_show", lambda bench, commit: lambda path: b"x" if commit is None else b"y"
    )
    moved = Layout(tmp_path / "moved")
    run.declare(moved, _declared(bench="abc1234"))
    shutil.copytree(source.root / "records", moved.root / "records")
    with pytest.raises(ValueError, match="pinned apart"):
        run.reuse(target, moved, ["fake"])
    assert not target.reused.exists()
    # a file changed outside the tool's default setting, accepted by name
    assert run.reuse(target, moved, ["fake"], accepted=["scripts/fake.py"])["fake"] == 2
    assert [n["accepted"] for n in _lines(target.reused)] == [["scripts/fake.py"]] * 2
    assert run.reuse(target, source, ["fake"]) == {"fake": 0, "fake left": 1}  # pinned alike
    with pytest.raises(ValueError, match="both runs plan"):
        run.reuse(target, source, ["orbit"])
    unexported = Layout(tmp_path / "unexported")
    run.declare(unexported, _declared())
    with pytest.raises(ValueError, match="has not exported"):
        run.reuse(unexported, source, ["fake"])
    family = Layout(tmp_path / "family")
    run.declare(family, _declared(family=1, environments=("walk",)))
    with pytest.raises(ValueError, match="another family"):
        run.reuse(family, source, ["fake"])
