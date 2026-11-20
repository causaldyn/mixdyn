"""The one runner: a tool's arm runs by the engine's rule. Each world is a process of its own, in a
scope capped at the arm's declared memory, admitted by the memory gate, stopped at its timeout and
tried twice, then recorded as a failure of the kind its ending gives; a record a killed process
left half-written is set aside and refitted, never scored; a relaunch runs only what is missing;
and the report reads every world."""

import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("causaldyn_bench")

import causaldyn_bench

from scorecard import report, run
from scorecard.arms.base import Planned
from scorecard.layout import Declaration, Layout, Limits, write_json
from scorecard.provenance import tree
from scorecard.tools import TOOLS, Tool

FAKE = Path(__file__).with_name("fake_tool.py")
TIMEOUT = 10.0  # a fake world's: past a start under load, and short of the test's patience
MODES = ("record", "raise", "exit", "memory", "segfault", "half", "sleep")
GIB = 2**30


def _scopes() -> bool:
    if shutil.which("systemd-run") is None:
        return False
    command = ["systemd-run", "--user", "--scope", "-q", "true"]
    return subprocess.run(command, capture_output=True, check=False).returncode == 0


scopes = pytest.mark.skipif(not _scopes(), reason="no user systemd to run a scope")


class Still:
    """This engine's arm, which plays the status quo."""

    variants = ("still",)
    tests = True
    cells = False

    def fit(self, observation, *, smoke):
        return observation.status_quo

    def plan(self, fit, variant, observation):
        return Planned(weekly=fit)


@pytest.fixture
def tooled(tmp_path, monkeypatch):
    """A run of the engine's `still` and the tool `fake` over a world for each of `MODES`, each
    world's `still` records in; the run and each mode's seed."""
    modes = tmp_path / "modes.json"
    monkeypatch.setitem(run.ARMS, "still", Still)
    monkeypatch.setitem(TOOLS, "fake", Tool((sys.executable, str(FAKE), "--modes", str(modes))))
    declared = Declaration(
        family=0,
        environments=("drawn",),
        worlds=len(MODES),
        pilot=True,
        rungs=(0,),
        arms=("still",),
        external=("fake",),
        limits={
            "still": Limits(memory="1G", timeout=60.0, slots=1),
            "fake": Limits(memory="256M", timeout=TIMEOUT, slots=4),
        },
        settle=0.0,
    )
    out = Layout(tmp_path / "run")
    run.declare(out, declared)
    run.export_all(out)
    seeds = dict(zip(MODES, declared.seeds("drawn"), strict=False))
    modes.write_text(json.dumps({str(seed): mode for mode, seed in seeds.items()}))
    for seed in seeds.values():
        run.one(out, "still", "drawn", seed, (0,))
    return out, seeds


def _starts(out: Layout, seed: int) -> int:
    counted = out.root.parent / "modes.starts" / str(seed)
    return len(counted.read_text().splitlines()) if counted.exists() else 0


def _fake(out: Layout, seed: int) -> dict:
    return json.loads(out.record("fake", 0, "drawn", 0, seed).read_text())


@scopes
def test_each_world_of_a_tool_ends_as_a_record_or_a_failure_of_its_kind(tooled, capsys):
    out, seeds = tooled
    run.run_arm(out, "fake")
    ended = {}
    for mode, seed in seeds.items():
        record = _fake(out, seed)
        ended[mode] = record.get("failure_kind", record["error"])
    assert ended == {
        "record": None,
        "raise": "RuntimeError: the fit broke",
        "exit": "exit",
        "memory": "signal",
        "segfault": "signal",
        "half": None,
        "sleep": "timeout",
    }
    assert {mode: _starts(out, seed) for mode, seed in seeds.items()} == {
        "record": 1,
        "raise": 1,
        "exit": 2,
        "memory": 2,
        "segfault": 2,
        "half": 2,
        "sleep": 2,
    }
    assert _fake(out, seeds["exit"])["error"].endswith("exit code 3; exit code 3")
    assert _fake(out, seeds["segfault"])["error"].endswith("exit code -11; exit code -11")
    assert _fake(out, seeds["sleep"])["error"].endswith(f"stopped after {TIMEOUT:g} s")
    assert _fake(out, seeds["sleep"])["provenance"]["bench"]["path"] == str(tree(causaldyn_bench))
    (aside,) = (out.root / "logs" / "fake").glob("*.unread-*.json")
    assert aside.name.startswith(f"drawn-k0-{seeds['half']}.unread-")
    assert not list((out.root / "inputs").rglob("*.json"))  # each record moved into the run

    run.score(out)
    assert len(out.scores.read_text().splitlines()) == len(MODES)
    report.main(out, ["still"])
    shown = capsys.readouterr().out
    assert f"{len(MODES)} worlds, the first {len(MODES)} of each environment" in shown
    assert "failed 5 (exit 1, raised 1, signal 2, timeout 1)" in shown


@scopes
def test_a_relaunch_runs_only_what_is_missing_and_refits_a_half_written_record(tooled, monkeypatch):
    out, seeds = tooled
    (out.root.parent / "modes.json").write_text(
        json.dumps({str(seed): "record" for seed in seeds.values()})
    )
    first, second, third, *rest = seeds.values()
    record = {"version": 2, "seed": first, "digest": "0", "error": None, "weekly": [1.0]}
    write_json(out.record("fake", 0, "drawn", 0, first), record)  # recorded before the stop
    made = out.inputs("fake", 0, "drawn", 0, second) / f"world_{second}.json"
    made.parent.mkdir(parents=True)
    made.write_text(json.dumps(record | {"seed": second}))  # made, but not yet moved, at the stop
    half = out.record("fake", 0, "drawn", 0, third)
    half.write_text(json.dumps(record | {"seed": third})[:20])  # as a script writing into the run
    started = []
    real = run.wait

    def watched(command, *args, **kwargs):
        started.append(int(Path(command[command.index("--worlds") + 1]).name.split("_")[1]))
        return real(command, *args, **kwargs)

    monkeypatch.setattr(run, "wait", watched)
    run.run_arm(out, "fake")
    assert sorted(started) == sorted([third, *rest])
    assert _fake(out, first) == record
    assert _fake(out, second)["seed"] == second
    assert _fake(out, third)["digest"] != "0"  # refitted
    assert len(list((out.root / "logs" / "fake").glob("*.unread-*.json"))) == 1
    started.clear()
    run.run_arm(out, "fake")
    assert started == []


def test_a_record_that_does_not_parse_is_not_scored_and_waits_for_its_arm(tooled, caplog):
    out, seeds = tooled
    for mode, seed in seeds.items():
        digest = json.loads(out.record("still", 0, "drawn", 0, seed).read_text())["digest"]
        text = json.dumps({"version": 2, "digest": digest, "error": "RuntimeError: x"})
        path = out.record("fake", 0, "drawn", 0, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text[:30] if mode == "half" else text)  # as a killed process leaves it
    run.score(out)
    scored = [json.loads(line)["seed"] for line in out.scores.read_text().splitlines()]
    assert sorted(scored) == sorted(seed for mode, seed in seeds.items() if mode != "half")
    assert "1 records do not parse" in caplog.text


def test_the_gate_admits_a_process_once_its_cap_and_the_reserve_are_available(
    tmp_path, monkeypatch
):
    readings = iter([3 * GIB, 7 * GIB, 8 * GIB, 1 * GIB])
    seen = []

    def available():
        seen.append(next(readings))
        return seen[-1]

    monkeypatch.setattr(run, "available", available)
    monkeypatch.setattr(run, "POLL", 0.0)
    with run.Gate(tmp_path / "gate.lock", reserve=2 * GIB, settle=0.0).admit(6 * GIB):
        assert seen == [3 * GIB, 7 * GIB, 8 * GIB]


def test_memory_available_is_the_kernels_reckoning():
    meminfo = Path("/proc/meminfo").read_text()
    stated = next(int(line.split()[1]) for line in meminfo.splitlines() if "MemAvailable" in line)
    assert abs(run.available() - stated * 1024) < 256 * 2**20


def test_a_start_holds_the_gate_until_it_settles_or_its_process_ends(tmp_path):
    gate = run.Gate(tmp_path / "gate.lock", reserve=0, settle=1.5)
    stamp = [sys.executable, "-c", "import time; print(time.time()); time.sleep(2)"]

    def start(name):
        run.wait(stamp, tmp_path / name, 30.0, gate=gate)

    threads = [threading.Thread(target=start, args=(name,)) for name in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    a, b = (float((tmp_path / name).read_text()) for name in ("a", "b"))
    assert abs(a - b) >= 1.4  # the second was admitted once the first had settled
    began = time.monotonic()
    quick = [sys.executable, "-c", "pass"]
    gate = run.Gate(tmp_path / "gate.lock", reserve=0, settle=600.0)
    assert run.wait(quick, tmp_path / "c", 30.0, gate=gate) == "exit code 0"
    assert run.wait(quick, tmp_path / "d", 30.0, gate=gate) == "exit code 0"
    assert time.monotonic() - began < 30.0  # a process that ends lets the next in


def test_a_tools_timeout_counts_from_its_start_not_from_the_gate(tmp_path):
    gate = run.Gate(tmp_path / "gate.lock", reserve=0, settle=20.0)
    began = time.monotonic()
    sleeper = [sys.executable, "-c", "import time; time.sleep(60)"]
    assert run.wait(sleeper, tmp_path / "l", 1.0, gate=gate) == "stopped after 1 s"
    assert time.monotonic() - began < 1.8  # counted from the start, the settle within it


@pytest.mark.skipif(shutil.which("uv") is None, reason="no uv to run a tool")
def test_a_python_tool_killed_under_uv_fails_by_a_signal(tmp_path):
    dies = (
        "import os, resource, signal; resource.setrlimit(resource.RLIMIT_CORE, (0, 0)); "
        "os.kill(os.getpid(), signal.SIGSEGV)"
    )
    command = ["uv", "run", "--no-project", "--python", sys.executable, "python", "-c", dies]
    ended = run.wait(command, tmp_path / "l", 120.0)
    assert ended == "exit code 139"  # uv's account of its child killed by SIGSEGV
    assert run.ending(ended) == "signal"


def test_each_tool_runs_as_the_bench_runs_it():
    uv = ("uv", "run", "--no-project", "--python", "3.12")
    pymc = ("--with-requirements", "scripts/pymc_marketing_arm.txt", "python")
    assert TOOLS["pymc-marketing shared"].command == (
        *uv,
        *pymc,
        "scripts/pymc_marketing_arm.py",
        "--setting",
        "shared",
    )
    assert TOOLS["orbit"].command == (
        *uv,
        "--extra-index-url",
        "https://download.pytorch.org/whl/cpu",
        "--index-strategy",
        "unsafe-best-match",
        "--with-requirements",
        "scripts/orbit_arm.txt",
        "python",
        "scripts/orbit_arm.py",
    )
    assert TOOLS["robyn"].command == (
        "Rscript",
        "--vanilla",
        "scripts/robyn_arm.R",
        "--ranges",
        "genre",
    )
    bench = tree(causaldyn_bench)
    for tool in TOOLS.values():
        for part in tool.command:
            if part.startswith("scripts/"):
                assert (bench / part).is_file(), part
    assert run._scoped(Limits(memory="8G", timeout=1.0, slots=1), ["x"]) == [
        "systemd-run",
        "--user",
        "--scope",
        "-q",
        "-p",
        "MemoryMax=8G",
        "-p",
        "MemorySwapMax=0",
        "x",
    ]
    handed = run.Handed(TOOLS["meridian"], Path("/w/world_1"), Path("/r/world_1.json"))
    assert handed.command(smoke=True)[-5:] == [
        "--worlds",
        "/w/world_1",
        "--out",
        "/w/world_1",
        "--smoke",
    ]
    assert handed.made == Path("/w/world_1/world_1.json")


def test_robyn_runs_on_the_r_library_and_nevergrad_the_run_declares(tmp_path):
    declared = Declaration(
        family=0,
        environments=("drawn",),
        worlds=1,
        pilot=True,
        rungs=(0,),
        arms=("bayes",),
        external=("robyn",),
        r_library="/lib",
        nevergrad="/ng",
    )
    assert TOOLS["robyn"].environment(declared) == {
        "R_LIBS": "/lib",
        "R_LIBS_USER": "/lib",
        "RETICULATE_PYTHON": "/ng/bin/python",
    }
    assert TOOLS["meridian"].environment(declared) == {}
