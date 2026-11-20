"""The harness's plumbing: a run declared once, each world's records written beside its export, a
failure that plays the status quo and names its kind, a process stopped at its timeout, the compute
a record holds and where its process ran, and the scores and the report read off the records, with
the bench's report of the family where it is asked for."""

import dataclasses
import json
import math
import resource
import sys
import time
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("causaldyn_bench")

from causaldyn_bench.scorecard.observe import read
from causaldyn_bench.scorecard.track_m2 import TRACK_M2

from scorecard import report, run
from scorecard.arms.base import Planned, records
from scorecard.layout import Declaration, Layout, Limits, size, slug, write_json
from scorecard.provenance import ONE_THREAD


class Held:
    """An arm that plays the status quo under one name and fails to plan under the other."""

    variants = ("held", "unplanned")
    tests = True
    cells = False

    def fit(self, observation, *, smoke):
        return observation.status_quo

    def plan(self, fit, variant, observation):
        if variant == "unplanned":
            raise RuntimeError("no plan")
        return Planned(weekly=fit, gain=0.0, gain_interval=(-1.0, 1.0), flags={"odd": math.nan})


class Broken:
    variants = ("broken",)
    tests = True
    cells = False

    def fit(self, observation, *, smoke):
        raise ValueError("no fit")

    def plan(self, fit, variant, observation):
        raise AssertionError("never planned")


LIMITS = Limits(memory="1G", timeout=60.0, slots=1)
DECLARED = Declaration(
    family=0,
    environments=("drawn",),
    worlds=2,
    pilot=True,
    rungs=(0,),
    arms=("held",),
    limits={"held": LIMITS},
    settle=0.0,
)


@pytest.fixture
def layout(tmp_path, monkeypatch):
    monkeypatch.setitem(run.ARMS, "held", Held)
    monkeypatch.setitem(run.ARMS, "broken", Broken)
    out = Layout(tmp_path / "run")
    run.declare(out, DECLARED)
    run.export_all(out)
    return out


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"family": 99}, "family 99"),
        ({"environments": ("nowhere",)}, "draws"),
        ({"worlds": 0}, "at least 1 world"),
        ({"rungs": (-1,)}, "at or above 0"),
        ({"arms": ("held", "held")}, "once each"),
        ({"pilot": False}, "pre-registered"),
        ({"limits": {"held": LIMITS, "other": LIMITS}}, "limits for"),
        ({"reserve": "2GB"}, "a size is"),
        ({"settle": -1.0}, "settles"),
        ({"attempts": 0}, "an attempt"),
        ({"r_library": "/lib"}, "declared together"),
        ({"world_directory": "/worlds"}, "reads none from a directory"),
        ({"family": 15, "environments": ("demo",)}, "declare their directory"),
    ],
)
def test_a_declaration_the_run_cannot_hold_is_refused(change, match):
    with pytest.raises(ValueError, match=match):
        dataclasses.replace(DECLARED, **change)


@pytest.mark.parametrize(
    "change",
    [
        {"memory": "1.5G"},
        {"memory": "8GB"},
        {"memory": "0"},
        {"timeout": 0.0},
        {"slots": 0},
    ],
)
def test_limits_no_world_can_run_by_are_refused(change):
    with pytest.raises(ValueError, match=r"a size is|some memory"):
        dataclasses.replace(LIMITS, **change)


def test_a_size_is_read_as_systemd_reads_memory_max():
    assert [size(text) for text in ("4096", "64K", "256M", "8G", "1T")] == [
        4096,
        64 * 2**10,
        256 * 2**20,
        8 * 2**30,
        2**40,
    ]
    assert LIMITS.cap == 2**30


def test_a_run_declares_every_arms_limits_and_robyns_paths(layout, tmp_path):
    out = Layout(tmp_path / "other")
    with pytest.raises(ValueError, match="no limits for"):
        run.declare(out, dataclasses.replace(DECLARED, limits={}))
    robyn = dataclasses.replace(
        DECLARED, external=("robyn",), limits={"held": LIMITS, "robyn": LIMITS}
    )
    with pytest.raises(ValueError, match="nevergrad"):
        run.declare(out, robyn)
    with pytest.raises(ValueError, match="runs no tool"):
        run.names(dataclasses.replace(DECLARED, external=("excel",)))
    assert not out.declaration.exists()
    run.declare(out, dataclasses.replace(robyn, r_library="/lib", nevergrad="/ng"))
    assert out.read().limits == {"held": LIMITS, "robyn": LIMITS}


def test_a_run_is_declared_once(layout):
    assert layout.read() == DECLARED
    with pytest.raises(RuntimeError, match="is declared as"):
        layout.declare(dataclasses.replace(DECLARED, worlds=3))


def test_records_need_names_of_their_own(layout):
    with pytest.raises(ValueError, match="once each"):
        dataclasses.replace(DECLARED, external=("held",))
    for clash in ("status quo", "Held"):  # a built-in arm's name; one kept in held's directory
        with pytest.raises(ValueError, match="names of their own"):
            run.names(dataclasses.replace(DECLARED, external=(clash,)))
    with pytest.raises(ValueError, match="no arms"):
        run.names(dataclasses.replace(DECLARED, arms=("nothing",), limits={}))
    assert slug("cvar 0.1") == "cvar-0-1"


def test_the_dlm_and_its_ablation_keep_their_records_apart_and_only_the_dlm_reads_the_tests():
    held = run.names(dataclasses.replace(DECLARED, arms=("dlm", "dlm-untested"), limits={}))
    assert held == [*run.ARMS["dlm"]().variants, *run.ARMS["dlm-untested"]().variants]
    assert len({slug(name) for name in held}) == 8
    assert (run.ARMS["dlm"]().tests, run.ARMS["dlm-untested"]().tests) == (True, False)


def test_a_failed_fit_records_its_error_for_every_variant_and_plays_the_status_quo(layout):
    observation = _observation(layout, TRACK_M2.pilots["drawn"][0])
    made = records(Broken(), observation, smoke=True)
    assert made["broken"]["error"] == "ValueError: no fit"
    assert made["broken"]["failure_kind"] == "fit"
    assert "Traceback" in made["broken"]["traceback"]
    assert made["broken"]["smoke"] == "smoke, not a result"
    assert made["broken"]["digest"] == observation.digest()
    assert "plan_wall_seconds" not in made["broken"]["cost"]


def test_one_variant_failing_leaves_the_others_and_a_nan_flag_is_null(layout):
    observation = _observation(layout, TRACK_M2.pilots["drawn"][0])
    made = records(Held(), observation, smoke=False)
    assert made["held"]["error"] is None
    assert "failure_kind" not in made["held"]
    assert made["held"]["flags"]["odd"] is None
    assert set(made["held"]["flags"]) == {"odd", "out of range"} | {
        f"{c} out of range" for c in observation.channels
    }
    assert made["unplanned"]["error"] == "RuntimeError: no plan"
    assert made["unplanned"]["failure_kind"] == "plan"
    json.dumps(made, allow_nan=False)


def test_a_malformed_plan_is_recorded_as_an_error(layout):
    observation = _observation(layout, TRACK_M2.pilots["drawn"][0])

    class Short(Held):
        def plan(self, fit, variant, observation):
            return Planned(weekly=np.append(fit, 1.0))

    made = records(Short(), observation, smoke=False)["held"]
    assert "a plan is a finite weekly spend" in made["error"]
    assert made["failure_kind"] == "record"


def _compile():
    """Compile and run a function no earlier call compiled."""
    scale = float(np.random.default_rng().random())
    jax.jit(lambda a: jnp.sin(a) * scale)(jnp.ones(3)).block_until_ready()


class Sampled(Held):
    """An arm whose fit compiles a function of its own, and whose plans name their chains' least
    bulk effective sample size: 400 for one; for the other, which compiles one more, not a
    number."""

    def fit(self, observation, *, smoke):
        _compile()
        return observation.status_quo

    def plan(self, fit, variant, observation):
        if variant == "held":
            return Planned(weekly=fit, ess=400.0)
        _compile()
        return Planned(weekly=fit, ess=math.nan)


def test_a_record_holds_its_steps_seconds_compiling_and_the_worlds_peak_resident_set(layout):
    observation = _observation(layout, TRACK_M2.pilots["drawn"][0])
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    made = records(Sampled(), observation, smoke=False)
    status = Path("/proc/self/status").read_text()
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    held, unplanned = made["held"]["cost"], made["unplanned"]["cost"]
    assert held["fit_compile_seconds"] > 0.0
    assert held["plan_compile_seconds"] == 0.0
    assert unplanned["plan_compile_seconds"] > 0.0
    for cost in (held, unplanned):
        for step in ("wall", "cpu", "compile"):
            assert (
                cost[f"{step}_seconds"]
                == cost[f"fit_{step}_seconds"] + cost[f"plan_{step}_seconds"]
            )
    # the process's peak in bytes, as getrusage reads it in kB from per-CPU counters it does not
    # fold, so the two may differ by those counters' batches
    assert held["fit_peak_rss_bytes"] <= held["peak_rss_bytes"]
    assert held["peak_rss_bytes"] == pytest.approx(after, rel=0.05, abs=0.0)
    assert after >= before
    # the status's own line, in KiB, read once the record was made
    hwm = next(int(line.split()[1]) for line in status.splitlines() if line.startswith("VmHWM:"))
    assert 0 <= hwm * 1024 - held["peak_rss_bytes"] < 2**20
    assert {k: v for k, v in held.items() if "peak" in k} == {
        k: v for k, v in unplanned.items() if "peak" in k
    }
    assert held["ess_bulk_per_fit_cpu_second"] == 400.0 / held["fit_cpu_seconds"]
    assert "ess_bulk_per_fit_cpu_second" not in unplanned
    assert made["unplanned"]["n/a"]["ess"] == "the least bulk ESS is not finite: nan"
    json.dumps(made, allow_nan=False)


def test_a_tools_compute_is_its_process_s_cpu_where_its_record_does_not_split_the_fit():
    """A tool's record (the bench's process wrapper) holds peak RSS and the whole process's CPU,
    not the fit's compiling: the w30c pilot's every report stopped on its missing key."""
    gib = 2.0**30
    tool = {"peak_rss_bytes": 2 * gib, "cpu_seconds": 30.0, "wall_seconds": 10.0}
    engine = tool | {"fit_compile_seconds": 4.0, "fit_cpu_seconds": 20.0}
    tools = [types.SimpleNamespace(record={"cost": tool | {"cpu_seconds": c}}) for c in (30, 50)]
    engines = [types.SimpleNamespace(record={"cost": engine})] * 2
    assert report._compute(tools) == (
        "compute of 2: peak RSS median 2.00 GiB, most 2.00; the process's CPU median 40.0 s"
    )
    assert report._compute(engines) == (
        "compute of 2: peak RSS median 2.00 GiB, most 2.00; the fit compiling median 4.0 s, its "
        "CPU median 20.0 s"
    )


def test_records_score_beside_the_built_in_arms_and_report(layout, capsys):
    d = layout.read()
    for seed in d.seeds("drawn"):
        run.one(layout, "held", "drawn", seed, (0,))
    run.score(layout)
    run.score(layout)  # a world already scored is not scored again
    lines = layout.scores.read_text().splitlines()
    assert len(lines) == 2
    for line in lines:
        arms = json.loads(line)["arms"]
        assert arms["held"]["regret"] == arms["status quo"]["regret"]
        assert arms["held"]["uplift"] is not None
        assert arms["unplanned"]["failure"] == "RuntimeError: no plan"
        assert arms["unplanned"]["regret"] == arms["status quo"]["regret"]
    report.main(layout, ["held", "unplanned"])
    shown = capsys.readouterr().out
    assert "pilot" in shown
    assert "held less, regret per euro" in shown
    assert "failed 2 (plan 2)" in shown
    assert "compute of 2: peak RSS median" in shown
    assert "past the bounds" not in shown  # nothing here samples
    assert "flags" not in shown
    report.main(layout, ["held", "unplanned"], tau=0.25)
    flagged = capsys.readouterr().out
    assert flagged.startswith(shown)
    assert "k = 0, flags over the first 2 worlds of every environment:" in flagged
    assert "ROI error above 0.25 (return)" in flagged


def test_a_steps_compiling_is_jax_s_tracing_lowering_and_compiling_within_it(layout):
    observation = _observation(layout, TRACK_M2.pilots["drawn"][0])
    events = (
        "/jax/core/compile/jaxpr_trace_duration",
        "/jax/core/compile/jaxpr_to_mlir_module_duration",
        "/jax/core/compile/backend_compile_duration",
    )
    heard = []

    def listen(event, duration_secs, **_):
        if event in events:
            heard.append(duration_secs)

    jax.monitoring.register_event_duration_secs_listener(listen)
    try:
        made = records(Sampled(), observation, smoke=False)
    finally:
        jax.monitoring.unregister_event_duration_listener(listen)
    held, unplanned = made["held"]["cost"], made["unplanned"]["cost"]
    steps = held["fit_compile_seconds"] + held["plan_compile_seconds"]
    assert steps + unplanned["plan_compile_seconds"] == pytest.approx(math.fsum(heard), rel=1e-9)
    assert len(heard) >= 2 * len(events)  # the fit's function and the plan's, each in three steps


class Heavy(Held):
    """An arm whose plan holds 200 MB at once, and whose fit nothing much."""

    variants = ("held",)

    def plan(self, fit, variant, observation):
        held = np.ones(25_000_000)
        return Planned(weekly=fit + 0.0 * held[-1])


def test_the_fits_peak_is_read_before_the_plans_and_the_worlds_after_them(layout):
    observation = _observation(layout, TRACK_M2.pilots["drawn"][0])
    Path("/proc/self/clear_refs").write_text("5")  # the peak is the resident set now, from here on
    cost = records(Heavy(), observation, smoke=False)["held"]["cost"]
    assert cost["peak_rss_bytes"] - cost["fit_peak_rss_bytes"] >= 150 * 2**20


def _cpu_model():
    """The CPU's model, the first that `/proc/cpuinfo` names."""
    lines = Path("/proc/cpuinfo").read_text().splitlines()
    return next(line.split(":", 1)[1].strip() for line in lines if line.startswith("model name"))


def test_a_record_names_where_its_process_ran_as_the_bench_s_compute_table_reads_it(
    layout, monkeypatch
):
    observation = _observation(layout, TRACK_M2.pilots["drawn"][0])
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.setenv("PJRT_NPROC", "2")
    made = records(Held(), observation, smoke=False)
    for variant in Held.variants:
        assert made[variant]["cost"]["cpu_model"] == _cpu_model()
        assert made[variant]["cost"]["threads"] == "3"
        assert made[variant]["cost"]["pjrt_nproc"] == "2"
    monkeypatch.delenv("OMP_NUM_THREADS")
    monkeypatch.delenv("PJRT_NPROC")
    cost = records(Broken(), observation, smoke=False)["broken"]["cost"]
    assert (cost["threads"], cost["pjrt_nproc"]) == (None, None)


def test_the_report_asked_for_its_columns_ends_with_the_bench_s_report_of_the_family(
    layout, capsys, monkeypatch
):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    d = layout.read()
    for seed in d.seeds("drawn"):
        run.one(layout, "held", "drawn", seed, (0,))
    run.score(layout)
    report.main(layout, ["held", "unplanned"])
    alone = capsys.readouterr().out
    report.main(layout, ["held", "unplanned"], columns=True)
    printed = capsys.readouterr().out
    assert printed.startswith(alone)
    family = printed.removeprefix(alone)
    title = family.splitlines()[1]
    assert family.startswith("\n# Family 0: ")
    assert title.endswith(", pilot")
    assert "pre-registration" not in family
    assert (
        "| arm | regret per euro | failed | moved | CRPS | MAPE, % | coverage 80/90/95 "
        "| effect against spend share |"
    ) in family
    for arm in ("held", "unplanned"):
        assert f"| {arm} | 2 | {_cpu_model()} | 1 | " in family
    nothing = " | ".join(["not recorded"] * 5)
    assert f"| status quo | 2 | {nothing} |" in family


def test_a_world_whose_process_leaves_no_records_is_recorded_as_a_failure(layout, monkeypatch):
    calls = []
    endings = iter(["stopped after 1 s", "exit code -9"] * 2)
    monkeypatch.setattr(run, "wait", lambda *a, **k: calls.append(a) or next(endings))
    run.run_arm(layout, "held")
    assert len(calls) == 2 * 2  # two worlds, two attempts each
    d = layout.read()
    for seed in d.seeds("drawn"):
        for variant in Held.variants:
            record = json.loads(layout.record(variant, 0, "drawn", 0, seed).read_text())
            assert record["error"].startswith("the world's process ended without its records")
            assert record["error"].endswith("stopped after 1 s; exit code -9")
            assert record["failure_kind"] == "signal"


@pytest.mark.parametrize(
    ("reason", "kind"),
    [
        ("stopped after 3600 s", "timeout"),
        ("exit code -9", "signal"),
        ("exit code 0", "exit"),
        ("exit code 1", "exit"),
        ("exit code 137", "signal"),  # uv's account of its child killed by SIGKILL
        ("exit code 139", "signal"),  # and by SIGSEGV
        ("exit code 128", "exit"),
        ("exit code 192", "signal"),
        ("exit code 193", "exit"),
    ],
)
def test_a_process_that_leaves_no_records_fails_by_how_it_ended(reason, kind):
    assert run.ending(reason) == kind


@pytest.fixture
def blind(tmp_path, monkeypatch):
    """A run of an arm that reads no test, one world at two rungs, and what the arm was handed."""
    seen = []

    class Blind(Held):
        tests = False

        def fit(self, observation, *, smoke):
            seen.append(observation)
            return super().fit(observation, smoke=smoke)

    monkeypatch.setitem(run.ARMS, "blind", Blind)
    out = Layout(tmp_path / "blind")
    declared = dataclasses.replace(
        DECLARED, worlds=1, rungs=(0, 2), arms=("blind",), limits={"blind": LIMITS}, attempts=1
    )
    run.declare(out, declared)
    run.export_all(out)
    return out, out.read().seeds("drawn")[0], seen


def test_an_arm_that_reads_no_test_fits_a_world_once_for_every_rung(blind):
    out, seed, seen = blind
    assert read(out.export(0, "drawn", 2, seed)).lift.channel  # the second rung holds tests
    run.one(out, "blind", "drawn", seed, (2, 0))  # fitted on the rung that holds tests
    assert len(seen) == 1
    assert (seen[0].k, seen[0].lift.channel) == (0, ())
    for k in (0, 2):
        record = json.loads(out.record("held", 0, "drawn", k, seed).read_text())
        assert (record["k"], record["digest"]) == (k, _observation(out, seed, k).digest())


@pytest.mark.parametrize(("tests", "processes"), [(False, [["0", "2"]]), (True, [["0"], ["2"]])])
def test_a_world_is_one_process_unless_its_arm_reads_the_tests(
    blind, monkeypatch, tests, processes
):
    out, seed, _ = blind
    monkeypatch.setattr(run.ARMS["blind"], "tests", tests)
    calls = []
    monkeypatch.setattr(
        run, "wait", lambda command, *a, **k: calls.append(command) or "exit code 0"
    )
    run.run_arm(out, "blind")
    assert [c[c.index("--k") + 1 :] for c in calls] == processes
    for k in (0, 2):
        assert json.loads(out.record("held", 0, "drawn", k, seed).read_text())["k"] == k


def test_an_arm_that_reads_the_tests_records_one_rung_a_process(blind, monkeypatch):
    out, seed, seen = blind
    monkeypatch.setattr(run.ARMS["blind"], "tests", True)
    with pytest.raises(ValueError, match="runs on its own"):
        run.one(out, "blind", "drawn", seed, (0, 2))
    assert not seen


def test_rungs_whose_exports_differ_beyond_their_tests_are_refused(blind, monkeypatch):
    out, seed, seen = blind
    exported = run.read

    def drifted(path):
        o = exported(path)
        return dataclasses.replace(o, sales=o.sales * (1.0 + 1e-12)) if o.k == 2 else o

    monkeypatch.setattr(run, "read", drifted)
    with pytest.raises(ValueError, match="differ beyond their tests"):
        run.one(out, "blind", "drawn", seed, (0, 2))
    assert not seen


def test_a_worlds_process_computes_on_one_thread(tmp_path, monkeypatch):
    for name in ONE_THREAD:
        monkeypatch.delenv(name, raising=False)
    probe = "\n".join(
        [
            "import os",
            "os.environ['JAX_PLATFORMS'] = 'cpu'",
            "import jax, jax.numpy as jnp",
            "jax.jit(lambda a: a @ a)(jnp.ones((64, 64))).block_until_ready()",
            "tasks = os.listdir('/proc/self/task')",
            "print([open(f'/proc/self/task/{t}/comm').read().strip() for t in tasks]"
            ".count('tf_XLAEigen'))",
        ]
    )
    log = tmp_path / "log"
    assert run.wait([sys.executable, "-c", probe], log, 120) == "exit code 0"
    assert log.read_text().split()[-1] == "1"  # XLA's compute pool, a thread a CPU without the cap


def test_a_process_past_its_timeout_is_stopped(tmp_path):
    began = time.perf_counter()
    ended = run.wait([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path / "l", 0.5)
    assert ended == "stopped after 0.5 s"
    assert time.perf_counter() - began < 10.0
    assert run.wait([sys.executable, "-c", "raise SystemExit(3)"], tmp_path / "l", 30) == (
        "exit code 3"
    )


def _gone(pid, within=10.0):
    """Whether process `pid` ends, or is left a zombie, within `within` seconds."""
    stat = Path(f"/proc/{pid}/stat")
    ends = time.monotonic() + within
    while time.monotonic() < ends:
        if not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] == "Z":
            return True
        time.sleep(0.05)
    return False


@pytest.mark.parametrize(
    ("end", "ended"),
    [("pass", "exit code 0"), ("os.kill(os.getpid(), signal.SIGSEGV)", "exit code -11")],
)
def test_what_a_process_forked_ends_with_it(tmp_path, end, ended):
    forks = (
        "import os, resource, signal, time\n"
        "resource.setrlimit(resource.RLIMIT_CORE, (0, 0))\n"
        "worker = os.fork()\n"
        "if worker == 0:\n"
        "    time.sleep(60)\n"
        "print(worker, flush=True)\n"
        f"{end}\n"
    )
    log = tmp_path / "log"
    assert run.wait([sys.executable, "-c", forks], log, 60.0) == ended
    worker = int(log.read_text().split()[0])
    assert _gone(worker)  # as the workers R forks outlive R's own segfault


def test_write_json_is_strict(tmp_path):
    with pytest.raises(ValueError, match="Out of range float values"):
        write_json(tmp_path / "x.json", {"x": math.nan})
    assert not (tmp_path / "x.json").exists()


def _observation(layout, seed, k=0):
    return read(layout.export(0, "drawn", k, seed))
