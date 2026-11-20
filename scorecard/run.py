"""A scorecard run: declare it, export its worlds, run every arm on them, this engine's and the
tools', score every record, and report.

    PYTHONPATH=<bench>/src JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \\
        OPENBLAS_NUM_THREADS=1 PJRT_NPROC=1 \\
        uv run --no-sync python -m scorecard.run <command> --root <run> ...

* `declare`: write the run's declaration (`scorecard.layout.Declaration`), or check that the run
  already holds that one; every arm's limits are declared with it (`--limit`, and the CPUs a
  process of it reserves, `--cpus`), and a tool's cap, the worlds an environment its comparisons
  read (`--cap`, `--exploratory-caps`). A confirmatory run draws the worlds no scored run read
  (`--confirmatory`). A capped tool
  still plans every world: the cap is how the claim reads it. The pre-registration the run runs
  under is declared with its SHA-256 (`--preregistration`, `scorecard.layout.stamp`), and a scored
  run is declared only with one;
* `export`: export every world of the run at every rung, as the bench archives it; a geo family's
  at the bench's version 4 (`causaldyn_bench.scorecard.observe.UNIVERSES`), which holds the
  universe each of the rung's tests ran in and the world's switchback test, the panels the
  readers and the switchback arm read. A tool is handed a version-4 export as version 2, the
  export its script reads, and the record it writes is scored on the archive by version 2's
  digest (`_hand`);
* `arm --arm NAME`: run one of the run's arms, this engine's or a tool's (`scorecard.tools`), on
  every world it has not recorded, by one rule. Each world runs in a process of its own on one
  thread (`scorecard.provenance.ONE_THREAD`), in a systemd scope capped at the arm's declared
  memory with no swap, admitted by the run's memory gate (`Gate`), the arm's declared slots at a
  time, each stopped after its declared timeout. An engine's arm that reads the tests runs a world
  at each rung in a process; one that does not fits a world once for all its rungs (`one`). A
  tool reads the tests, so it runs a world at each rung: its script is handed a directory holding
  that world's export alone and writes its record there, and the harness moves the record into
  the run once it parses, so a record a killed process left half-written is never taken for one.
  A world is done when its records are in, whatever its process's exit code, since a process
  killed for memory can exit 0; one that ends without them runs again, up to the declared
  attempts in all, and is then recorded as a failure, which plays the status quo, of the kind its
  last attempt's ending gives (`ending`). Each attempt is written down as it starts and as it
  ends, in `logs/attempts.jsonl` (`Layout.attempts`), each line flushed to disk before the process
  starts, so that work a machine's stop cut short is read from the run; its log gets a line that
  says which attempt follows, appended as every attempt's output is. Inside its scope each process
  appends its cost to `logs/costs.jsonl` (`Layout.costs`) as it ends: its scope's CPU-seconds and
  peak memory, its wall and its exit, the stage's costs summed from those lines. An arm's limits
  name the CPUs a process of it takes (`Limits.cpus`), which a stage's admission reserves:
  measured, a tool takes more than its one thread (`scorecard.tools.RESERVED`);
* `reuse --from ROOT --arm TOOL [--arm TOOL ...] [--accept PATH ...]`: take a tool's records of
  the run's worlds from another run's root instead of fitting them again, each where the world's
  export is the same observation, by its digest as the tool read it, and the tool's pinned files
  are the same at both runs' commits of the bench, but the files accepted as changed outside the
  tool's default setting (`reuse`); each record taken is written down in `logs/reused.jsonl`;
* `score`: score every world whose records are all in and parse against its archived export, and
  append its score to `scores.jsonl`; a world already there is not scored again, and one whose
  record does not parse waits for its arm to run again. A variant whose record holds what the
  bench's scorer of a world does not read, the flighted schedule and the switchback's reading, is
  scored on the world's truth beside it (`SCORED`);
* `report`: the report (`scorecard.report`); with `--flags TAU`, each rung's flags read as
  classifiers, a channel's return wrong past `TAU` (`scorecard.flags`); with `--columns`, the
  bench's report of the run's family after it, and with `--json PATH` that report written to
  `PATH` and its markdown beside it.

A run's export and its arms start only where JAX runs in double precision and the pre-registration
the run declares is still the file declared; a scored run's, only where the bench is also at the
commit the run is declared against and both trees are clean (`scorecard.provenance.require`).
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import select
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import causaldyn_bench
from causaldyn_bench.scorecard.geo import Geo
from causaldyn_bench.scorecard.observe import UNIVERSES, export, read
from causaldyn_bench.scorecard.scoring import BUILT_IN, score_world

from scorecard import claim, report
from scorecard.arms import decisions, switchback
from scorecard.arms.anchored import Anchored
from scorecard.arms.base import records, withheld
from scorecard.arms.bayes import Bayes
from scorecard.arms.bayes_geo import BayesGeo, BayesGeoPlugIn
from scorecard.arms.chc_arm import Chc
from scorecard.arms.decisions import Decisions
from scorecard.arms.dlm import (
    Dlm,
    DlmEvents,
    DlmTwoStage,
    DlmTwoStageLiftPick,
    DlmTwoStageLogged,
    DlmUntested,
)
from scorecard.arms.geo_dlm import GeoDlm
from scorecard.arms.geo_eb import GeoEb
from scorecard.arms.margin import Margin
from scorecard.arms.panel import second
from scorecard.arms.population import DlmByPopulation
from scorecard.arms.readers import SC, SDID, TBR, Reader
from scorecard.arms.ridge import Ridge
from scorecard.arms.switchback import Switchback
from scorecard.arms.tested import CONCAVE_HILL, CONCAVE_PICK, LIFT_PICK, Tested
from scorecard.arms.voi import Voi
from scorecard.composite import COMPOSITES
from scorecard.layout import Declaration, Layout, Limits, size, slug, stamp, write_json
from scorecard.provenance import ONE_THREAD, provenance, require, tree
from scorecard.tools import RESERVED, TOOLS, Tool, fingerprint

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping, Sequence

    from causaldyn_bench.scorecard.family import Truth
    from causaldyn_bench.scorecard.observe import Observation
    from causaldyn_bench.scorecard.scoring import ArmScore

    from scorecard.arms.base import Arm

log = logging.getLogger("scorecard")

ARMS: dict[str, Callable[[], Arm[Any]]] = {
    "bayes": Bayes,
    "bayes-geo": BayesGeo,
    "bayes-geo-plug-in": BayesGeoPlugIn,
    "geo-eb": GeoEb,
    "dlm": Dlm,
    "dlm-untested": DlmUntested,
    "dlm-events": DlmEvents,
    "dlm-two-stage": DlmTwoStage,
    "dlm-two-stage-logged": DlmTwoStageLogged,
    "dlm-two-stage-lift-pick": DlmTwoStageLiftPick,
    "bayes-lift-pick": partial(Tested, (LIFT_PICK,)),
    "bayes-concave": partial(Tested, (CONCAVE_HILL, CONCAVE_PICK)),
    "dlm-by-population": DlmByPopulation,
    "ridge": Ridge,
    "margin": Margin,
    "anchored": Anchored,
    "chc": Chc,
    "voi": Voi,
    "decisions": Decisions,
    "switchback": Switchback,
    "read-tbr": partial(Reader, TBR),
    "read-sc": partial(Reader, SC),
    "read-sdid": partial(Reader, SDID),
    "geo-dlm": GeoDlm,
}
"""This engine's arms, by the name a declaration gives them."""
SCORED: dict[str, Callable[[ArmScore, Truth, Observation], ArmScore]] = {
    decisions.FLIGHTING: decisions.scored,
    switchback.VARIANT: switchback.scored,
}
"""The variants whose records hold what the bench's scorer of a world does not read, each with
what scores it on the world's truth and its archived export, beside the bench's score."""
ROOT = Path(__file__).resolve().parents[1]
POLL = 10.0
"""The seconds between the memory gate's readings of the memory available."""


def names(declaration: Declaration) -> list[str]:
    """The names the run's records are kept under: each arm's variants, then the tools' arms.

    Raises:
        ValueError: on an arm the harness does not have, a name held twice, a built-in arm's or
            a composite's (`scorecard.composite`), or two names kept in one directory, or a
            tool the harness does not run.
    """
    unknown = [a for a in declaration.arms if a not in ARMS]
    if unknown:
        msg = f"this engine has no arms {unknown}; it has {sorted(ARMS)}"
        raise ValueError(msg)
    held = [v for a in declaration.arms for v in ARMS[a]().variants] + list(declaration.external)
    slugs = [slug(n) for n in held]
    taken = (*BUILT_IN, *(c.name for c in COMPOSITES))
    if len(set(slugs)) != len(held) or set(held) & set(taken):
        msg = f"the run's records need names of their own, apart from {taken}: {held}"
        raise ValueError(msg)
    unrun = [t for t in declaration.external if t not in TOOLS]
    if unrun:
        msg = f"the harness runs no tool {unrun}; it runs {sorted(TOOLS)}"
        raise ValueError(msg)
    return held


def declare(layout: Layout, declaration: Declaration) -> None:
    """Declare the run, once every arm it names has its limits and every tool what it runs on.

    Raises:
        ValueError: on a scored run that names no pre-registration, what `names` refuses, an arm
            without limits, Robyn without its paths, or a default procedure that is not one of
            this engine's arms' variants; and as `Declaration.preregistered` does.
    """
    if declaration.scored and declaration.preregistration is None:
        msg = "a scored run names the pre-registration it runs under: declare its file"
        raise ValueError(msg)
    declaration.preregistered()
    held = names(declaration)
    ours = held[: len(held) - len(declaration.external)]
    if declaration.default is not None and declaration.default not in ours:
        msg = f"the default procedure is one of {ours}, not {declaration.default!r}"
        raise ValueError(msg)
    bare = [a for a in (*declaration.arms, *declaration.external) if a not in declaration.limits]
    if bare:
        msg = f"the run declares no limits for {bare}"
        raise ValueError(msg)
    for tool in declaration.external:
        TOOLS[tool].environment(declaration)
    layout.declare(declaration)
    log.info("declared %s: %d world-rungs", layout.root, len(declaration.jobs()))


def export_all(layout: Layout) -> None:
    d = layout.read()
    require(bench=d.bench if d.scored else None, preregistration=d.preregistration)
    family = d.source()
    observe = family.observe
    if isinstance(family, Geo):  # the other families' observe takes no version
        observe = partial(family.observe, version=UNIVERSES)
    for environment in d.environments:
        for seed in d.seeds(environment):
            world = None
            for k in d.rungs:
                if layout.export(d.family, environment, k, seed).exists():
                    continue
                world = world or family.world(environment, seed)
                export(observe(world, k), layout.exports(d.family, environment, k))
        log.info("exported %s", environment)


def one(layout: Layout, arm: str, environment: str, seed: int, rungs: Sequence[int]) -> None:
    """One world's records of one arm at each of `rungs`, each written beside the export it is
    scored against.

    An arm that reads the tests runs at one rung. One that does not is handed each export without
    its tests (`scorecard.arms.base.withheld`), one observation at every rung of a world, so it
    fits once, and the first rung's records are each rung's, naming that rung's export.

    Raises:
        ValueError: on several rungs for an arm that reads the tests, or rungs whose exports
            differ beyond their tests.
    """
    d = layout.read()
    made = ARMS[arm]()
    observations = [read(layout.export(d.family, environment, k, seed)) for k in rungs]
    if len(observations) > 1:
        if made.tests:
            msg = f"{arm} reads the tests, so each of the rungs {list(rungs)} runs on its own"
            raise ValueError(msg)
        if len({withheld(o).digest() for o in observations}) > 1:
            msg = f"world {seed}'s exports at the rungs {list(rungs)} differ beyond their tests"
            raise ValueError(msg)
    source = provenance()
    made_records = records(made, observations[0], smoke=d.smoke)
    for i, (k, observation) in enumerate(zip(rungs, observations, strict=True)):
        shared = {"k": k, "digest": observation.digest(), "cost": {"fitted with k": rungs[0]}}
        for variant, record in made_records.items():
            write_json(
                layout.record(variant, d.family, environment, k, seed),
                record | (shared if i else {}) | {"provenance": source},
            )


Job = tuple[str, int, tuple[int, ...]]
"""One world's process: the environment, the seed and the rungs it records."""


def run_arm(layout: Layout, arm: str) -> None:
    """Every world of the run that `arm`, this engine's or a tool's, has not recorded, each in a
    process of its own by the arm's declared limits: at each rung for an arm that reads the
    tests, at every rung at once for one that does not.

    Raises:
        ValueError: on an arm the run does not name or declares no limits for, or a tool in R
            without its paths.
    """
    d = layout.read()
    require(bench=d.bench if d.scored else None, preregistration=d.preregistration)
    limits = d.limits.get(arm)
    if limits is None:
        msg = f"the run declares no limits for {arm}"
        raise ValueError(msg)
    if arm in d.arms:
        made = ARMS[arm]()
        variants, tests, tool = made.variants, made.tests, None
    elif arm in d.external:
        variants, tests, tool = (arm,), True, TOOLS[arm]
        tool.environment(d)
    else:
        msg = f"the run names no arm {arm}: {[*d.arms, *d.external]}"
        raise ValueError(msg)
    gate = Gate(layout.gate, size(d.reserve), d.settle)

    def recorded(environment: str, seed: int, k: int) -> bool:
        """Whether the world's records are in: written aside and moved into place by this engine,
        so in once they exist; a tool's, once they parse."""
        paths = [layout.record(v, d.family, environment, k, seed) for v in variants]
        return all(p.exists() if tool is None else parsed(p) is not None for p in paths)

    def world(job: Job) -> str:
        environment, seed, rungs = job
        began, reasons = time.perf_counter(), []
        at = "-".join(map(str, rungs))
        log_path = layout.root / "logs" / slug(arm) / f"{environment}-k{at}-{seed}.log"
        handed = None if tool is None else _hand(layout, tool, arm, job, log_path)
        for attempt in range(1, d.attempts + 1):
            tag = {
                "arm": arm,
                "environment": environment,
                "seed": seed,
                "rungs": list(rungs),
                "attempt": attempt,
                "attempts": d.attempts,
                "log": str(log_path),
            }
            if handed is None:
                command, cwd, extra = _command(layout, arm, job, limits, tag), ROOT, {}
            else:
                handed.take(log_path)
                if all(recorded(environment, seed, k) for k in rungs):
                    return "recorded"
                command = _scoped(limits, _costed(handed.command(smoke=d.smoke), layout, tag))
                cwd, extra = tree(causaldyn_bench), handed.tool.environment(d)
            attempted(layout, tag, log_path)
            ended = wait(
                command,
                log_path,
                limits.timeout,
                cwd=cwd,
                environment=extra,
                gate=gate,
                cap=limits.cap,
            )
            note(layout.attempts, tag | {"event": "end", "ended": time.time(), "outcome": ended})
            reasons.append(ended)
            if handed is not None:
                handed.take(log_path)
            if all(recorded(environment, seed, k) for k in rungs):
                return "recorded"
        _fail(layout, variants, job, reasons, began)
        return "failed: " + "; ".join(reasons)

    pending = [job for job in d.jobs() if not recorded(*job)]
    if tests:
        jobs: list[Job] = [(environment, seed, (k,)) for environment, seed, k in pending]
    else:
        grouped: dict[tuple[str, int], list[int]] = {}
        for environment, seed, k in pending:
            grouped.setdefault((environment, seed), []).append(k)
        jobs = [(environment, seed, tuple(rungs)) for (environment, seed), rungs in grouped.items()]
    log.info(
        "%s: %d of %d world-rungs to run, in %d processes, %d at a time",
        arm,
        len(pending),
        len(d.jobs()),
        len(jobs),
        limits.slots,
    )
    with ThreadPoolExecutor(limits.slots) as pool:
        for (environment, seed, rungs), outcome in zip(jobs, pool.map(world, jobs), strict=True):
            log.info("%s %s world %d at k = %s: %s", arm, environment, seed, list(rungs), outcome)


def note(path: Path, line: Mapping[str, Any]) -> None:
    """Append `line` to the JSON lines at `path`, flushed to disk before it returns: a line short
    of a pipe's buffer is written whole by one append, so processes that note at once do not mix
    their lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as sink:
        sink.write(json.dumps(line, allow_nan=False) + "\n")
        sink.flush()
        os.fsync(sink.fileno())


def attempted(layout: Layout, tag: Mapping[str, Any], log_path: Path) -> None:
    """Write an attempt down as it starts: its line in the run's attempts, and a line in its log
    that says which attempt the output after it is."""
    now = time.time()
    note(layout.attempts, {**tag, "event": "start", "began": now, "host": os.uname().nodename})
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as sink:
        when = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now))
        sink.write(f"--- attempt {tag['attempt']} of {tag['attempts']}, began {when}\n")


def _fail(
    layout: Layout, variants: Sequence[str], job: Job, reasons: Sequence[str], began: float
) -> None:
    """Record a world whose process ended without its records, in every attempt, as a failure of
    each variant that holds no record, of the kind the last attempt's ending gives."""
    d = layout.read()
    environment, seed, rungs = job
    for k in rungs:
        observation = read(layout.export(d.family, environment, k, seed))
        failure = {
            "version": observation.version,
            "seed": seed,
            "digest": observation.digest(),
            "family": d.family,
            "environment": environment,
            "k": k,
            "error": f"the world's process ended without its records: {'; '.join(reasons)}",
            "traceback": None,
            "failure_kind": ending(reasons[-1]),
            "cost": {"wall_seconds": time.perf_counter() - began},
            "provenance": provenance(),
        }
        if d.smoke:
            failure["smoke"] = "smoke, not a result"
        for v in variants:
            path = layout.record(v, d.family, environment, k, seed)
            if not path.exists():
                write_json(path, failure)


def _show(bench: Path, commit: str | None) -> Callable[[str], bytes | None]:
    """A reader of the bench's files at `commit`, its working tree's where None, None for a file
    it does not hold."""

    def show(path: str) -> bytes | None:
        if commit is None:
            held = bench / path
            return held.read_bytes() if held.is_file() else None
        held = subprocess.run(
            ["git", "-C", str(bench), "show", f"{commit}:{path}"], capture_output=True, check=False
        )
        return held.stdout if held.returncode == 0 else None

    return show


def reuse(
    layout: Layout, source: Layout, tools: Sequence[str], *, accepted: Sequence[str] = ()
) -> dict[str, int]:
    """Take each of `tools`' records of the run's worlds that `source`, another run's root, holds
    and the run does not, instead of fitting them again: each only where the run's export of the
    world is the observation the tool read, by the digest its record names (a version-4 export's
    by version 2's, as `_hand` hands it), and only for a tool whose pinned files
    (`scorecard.tools.fingerprint`), its script, the lock of its environment and what they read,
    are the same at the bench's commits both runs declare (the working tree's where a run declares
    none), but the files `accepted` names: a file changed only where its tool's default setting
    does not run, as an option it adds, which the pre-registration names with the evidence that
    the default's records are unchanged. Each record, a failure as well as a plan, is written as
    `source` holds it and noted in `Layout.reused`, with the accepted files that moved. Returns
    the records taken and the ones left to fit, whose export is another observation, by tool.

    Raises:
        ValueError: on another family or directory of worlds, a tool both runs do not plan, pinned
            files that differ and are not accepted, or a world of the run not exported yet; and as
            `scorecard.provenance.require` does.
    """
    d, o = layout.read(), source.read()
    require(bench=d.bench if d.scored else None, preregistration=d.preregistration)
    if (d.family, d.world_directory) != (o.family, o.world_directory):
        msg = f"{source.root} holds another family's worlds than {layout.root}"
        raise ValueError(msg)
    bench = tree(causaldyn_bench)
    out: dict[str, int] = {}
    for tool in tools:
        if tool not in d.external or tool not in o.external:
            msg = f"both runs plan the tool {tool!r}"
            raise ValueError(msg)
        mine = fingerprint(TOOLS[tool], _show(bench, d.bench))
        theirs = fingerprint(TOOLS[tool], _show(bench, o.bench))
        moved = sorted(
            path for path in mine.keys() | theirs.keys() if mine.get(path) != theirs.get(path)
        )
        refused = [path for path in moved if path not in accepted]
        if refused:
            msg = f"{tool} is pinned apart at the two runs: {refused}"
            raise ValueError(msg)
        taken = left = 0
        for environment, seed, k in d.jobs():
            target = layout.record(tool, d.family, environment, k, seed)
            found = source.record(tool, o.family, environment, k, seed)
            record = parsed(found)
            if target.exists() or record is None:
                continue
            export = layout.export(d.family, environment, k, seed)
            if not export.exists():
                msg = f"{layout.root} has not exported {environment} world {seed} at k = {k}"
                raise ValueError(msg)
            observation = read(export)
            if observation.version == UNIVERSES:
                observation = second(observation)
            if record.get("digest") != observation.digest():
                left += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            aside = target.with_name(f".{target.name}.{os.getpid()}")
            aside.write_bytes(found.read_bytes())
            aside.replace(target)
            note(
                layout.reused,
                {
                    "arm": tool,
                    "environment": environment,
                    "seed": seed,
                    "k": k,
                    "from": str(found),
                    "digest": record["digest"],
                    "failed": record.get("error") is not None,
                    "benches": [o.bench, d.bench],
                    "accepted": moved,
                },
            )
            taken += 1
        out[tool] = taken
        out[f"{tool} left"] = left
        log.info("%s: took %d records from %s, %d left to fit", tool, taken, source.root, left)
    return out


def parsed(path: Path) -> dict[str, Any] | None:
    """The record at `path`, or None where there is none or it is not one JSON object whole, as a
    process killed while writing it leaves it."""
    try:
        record = json.loads(path.read_bytes())
    except (FileNotFoundError, ValueError):
        return None
    return record if isinstance(record, dict) else None


@dataclass(frozen=True, slots=True)
class Handed:
    """One world at one rung handed to `tool`'s script: `inputs`, the directory that holds the
    world's export alone, where the script writes its record, and `record`, where the run keeps
    it."""

    tool: Tool
    inputs: Path
    record: Path

    @property
    def made(self) -> Path:
        return self.inputs / self.record.name

    def command(self, *, smoke: bool) -> list[str]:
        """The script's argv, pointed at the world's directory for its worlds and its records."""
        handed = ["--worlds", str(self.inputs), "--out", str(self.inputs)]
        return [*self.tool.command, *handed, *(["--smoke"] if smoke else [])]

    def take(self, log_path: Path) -> None:
        """Move the record the script made into the run once it parses; set one that does not
        parse aside, so that the world runs again."""
        if not self.made.exists():
            return
        if parsed(self.made) is None:
            _aside(self.made, log_path)
            return
        self.record.parent.mkdir(parents=True, exist_ok=True)
        self.made.replace(self.record)


def _hand(layout: Layout, tool: Tool, arm: str, job: Job, log_path: Path) -> Handed:
    """The world of `job` handed to `tool`: its export hard-linked into a directory of its own, a
    version-4 export written there as version 2 (`scorecard.arms.panel.second`), and a record the
    run holds of it that does not parse set aside. The tools' scripts read versions 1 and 2 and
    record version 2 beside the digest of the file they read, so the record is scored on the
    archive by version 2's digest, which a version-4 observation gives."""
    d = layout.read()
    environment, seed, (k,) = job
    handed = Handed(
        tool,
        layout.inputs(arm, d.family, environment, k, seed),
        layout.record(arm, d.family, environment, k, seed),
    )
    if handed.record.exists() and parsed(handed.record) is None:
        _aside(handed.record, log_path)
    handed.inputs.mkdir(parents=True, exist_ok=True)
    linked = handed.inputs / f"world_{seed}.npz"
    if not linked.exists():
        archived = layout.export(d.family, environment, k, seed)
        observation = read(archived)
        if observation.version == UNIVERSES:
            export(second(observation), handed.inputs)
        else:
            linked.hardlink_to(archived)
    return handed


def _aside(path: Path, log_path: Path) -> None:
    """Set a record that does not parse aside beside the world's log, and say so there."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    aside = log_path.with_name(f"{log_path.stem}.unread-{time.time_ns()}.json")
    path.replace(aside)
    with log_path.open("a") as sink:
        sink.write(f"{path} does not parse; set aside as {aside.name}\n")


COST = r"""import json, os, resource, signal, subprocess, sys, time
out, tag, command = sys.argv[1], sys.argv[2], sys.argv[3:]
began = time.time()
code = subprocess.call(command)
with open("/proc/self/cgroup") as held:
    group = "/sys/fs/cgroup" + next(l[3:] for l in held.read().splitlines() if l.startswith("0::"))
def read(name, key=None):
    try:
        with open(os.path.join(group, name)) as held:
            text = held.read()
    except OSError:
        return None
    if key is None:
        return int(text)
    return next((int(l.split()[1]) for l in text.splitlines() if l.split()[0] == key), None)
line = {"tag": json.loads(tag), "exit": code, "began": began, "ended": time.time(),
        "cpu_usec": read("cpu.stat", "usage_usec"), "peak_bytes": read("memory.peak")}
with open(out, "a") as sink:
    sink.write(json.dumps(line) + "\n")
if code < 0:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if -code not in (signal.SIGKILL, signal.SIGSTOP):
        signal.signal(-code, signal.SIG_DFL)
    os.kill(os.getpid(), -code)
sys.exit(code)
"""
"""What runs a process inside its scope and appends its cost as it ends: the scope's CPU-seconds,
its processes' and their children's, and its peak memory, read from its cgroup, beside its wall and
its exit. It ends as its process ended, by the same signal or with the same code, so `wait` reads
the process's own ending; a shell would turn a signal into a code. A process stopped at its
timeout or by a machine's stop is killed with it and appends none: its attempt's lines say so."""


def _costed(command: list[str], layout: Layout, tag: Mapping[str, Any]) -> list[str]:
    """`command` run by `COST`, which appends its cost to the run's costs (`Layout.costs`) under
    `tag` as it ends; inside `_scoped`'s scope, the scope's cost."""
    line = json.dumps(dict(tag), allow_nan=False)
    return [sys.executable, "-I", "-c", COST, str(layout.costs), line, *command]


def _scoped(limits: Limits, command: list[str]) -> list[str]:
    """`command` in a systemd scope of its own, capped at the arm's memory with no swap."""
    scope = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={limits.memory}"]
    return [*scope, "-p", "MemorySwapMax=0", *command]


def _command(
    layout: Layout, arm: str, job: Job, limits: Limits, tag: Mapping[str, Any] | None = None
) -> list[str]:
    """The argv of one world's process of this engine's `arm`; with `tag`, its cost appended
    under it (`_costed`)."""
    environment, seed, rungs = job
    command = [sys.executable, "-m", "scorecard.run", "one", "--root", str(layout.root)]
    command += ["--arm", arm, "--environment", environment, "--seed", str(seed)]
    command += ["--k", *map(str, rungs)]
    return _scoped(limits, command if tag is None else _costed(command, layout, tag))


def available() -> int:
    """The memory the kernel reckons a new process can take without swapping, in bytes."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    msg = "/proc/meminfo names no MemAvailable"
    raise RuntimeError(msg)


@dataclass(frozen=True, slots=True)
class Gate:
    """The run's admission of a world's process, whichever arm and whichever runner starts it: one
    start at a time, each once the memory available holds the process's cap and `reserve` bytes,
    the gate then held `settle` seconds or to the process's end, so that its memory shows before
    the next is admitted. The gate is a lock on a file of the run's (`Layout.gate`)."""

    path: Path
    reserve: int
    settle: float

    @contextmanager
    def admit(self, cap: int) -> Generator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as held:
            fcntl.flock(held, fcntl.LOCK_EX)
            while available() < cap + self.reserve:
                time.sleep(POLL)
            yield


def wait(
    command: list[str],
    log_path: Path,
    timeout: float,
    *,
    cwd: Path = ROOT,
    environment: Mapping[str, str] | None = None,
    gate: Gate | None = None,
    cap: int = 0,
) -> str:
    """Run `command` from `cwd` in a session of its own and on one thread (`ONE_THREAD`), with
    `environment` beside the harness's own, its output appended to `log_path`, to its end or
    `timeout` seconds from its start; how it ended. With `gate`, it starts once the gate admits
    `cap` bytes.

    At its end or its timeout its whole session is killed: at its end, what it forked and left
    behind, as the workers R forks outlive R's own segfault. The process is waited for on a file
    descriptor that does not reap it, so its session is still its own when it is killed."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as sink:
        with gate.admit(cap) if gate is not None else nullcontext():
            child = subprocess.Popen(
                command,
                cwd=cwd,
                env=os.environ | ONE_THREAD | dict(environment or {}),
                stdout=sink,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            began = time.monotonic()
            ended = os.pidfd_open(child.pid)
            if gate is not None:
                select.select([ended], [], [], min(gate.settle, timeout))
        try:
            left = max(0.0, timeout - (time.monotonic() - began))
            over = not select.select([ended], [], [], left)[0]
        finally:
            os.close(ended)
        with suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
        code = child.wait()
    return f"stopped after {timeout:g} s" if over else f"exit code {code}"


def ending(reason: str) -> str:
    """The kind of failure a world's process that left no records is, by `wait`'s account of how
    it ended: stopped at its timeout, killed by a signal, as by the memory cap or a segfault, or
    exited. `uv`, which runs every Python tool, ends with exit code `128 + n` when its child is
    killed by signal `n`, as a shell does, so such a code is a signal's."""
    if reason.startswith("stopped after "):
        return "timeout"
    code = int(reason.removeprefix("exit code "))
    return "signal" if code < 0 or 128 < code < 128 + signal.NSIG else "exit"


def score(layout: Layout) -> None:
    """Score every world whose records are all in, each parsed whole, and that `scores.jsonl` does
    not hold. A world with a record that does not parse, which a process killed while writing it
    leaves, is not scored: its arm runs it again."""
    d = layout.read()
    family = d.source()
    held = names(d)
    done = set()
    if layout.scores.exists():
        for line in layout.scores.read_text().splitlines():
            world = json.loads(line)
            done.add((world["environment"], world["seed"], world["k"]))
    waiting, unread = 0, []
    with layout.scores.open("a") as sink:
        for environment, seed, k in d.jobs():
            if (environment, seed, k) in done:
                continue
            paths = {n: layout.record(n, d.family, environment, k, seed) for n in held}
            if not all(p.exists() for p in paths.values()):
                waiting += 1
                continue
            records_in = {n: parsed(p) for n, p in paths.items()}
            whole = {n: r for n, r in records_in.items() if r is not None}
            if len(whole) < len(paths):
                unread += [str(paths[n]) for n in paths if n not in whole]
                continue
            archived = read(layout.export(d.family, environment, k, seed))
            scored = score_world(
                family, environment, seed, k, whole, pilot=d.pilot, archived=archived
            )
            beside = [n for n in whole if n in SCORED]
            if beside:
                truth = family.truth(family.world(environment, seed))
                arms = {n: SCORED[n](scored.arms[n], truth, archived) for n in beside}
                scored = replace(scored, arms=scored.arms | arms)
            sink.write(json.dumps(scored.as_json(), allow_nan=False) + "\n")
            sink.flush()
    log.info("scored; %d world-rungs wait on records", waiting)
    if unread:
        log.warning("%d records do not parse; their arms run them again: %s", len(unread), unread)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "command", choices=("declare", "export", "arm", "one", "score", "report", "reuse")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--family", type=int)
    parser.add_argument("--environments", nargs="+", default=[])
    parser.add_argument("--worlds", type=int)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument(
        "--confirmatory", action="store_true", help="draw the worlds no scored run read"
    )
    parser.add_argument("--rungs", type=int, nargs="+", default=[])
    parser.add_argument("--arms", nargs="+", default=[], help="this engine's arms, for declare")
    parser.add_argument("--external", nargs="+", default=[], help="the tools' arms, by name")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--bench", help="the bench's commit a scored run is declared against")
    parser.add_argument(
        "--preregistration",
        type=Path,
        help="the pre-registration the run runs under, declared with its SHA-256; a scored run's",
    )
    parser.add_argument(
        "--limit",
        nargs=4,
        action="append",
        default=[],
        metavar=("ARM", "MEMORY", "TIMEOUT", "SLOTS"),
        help="an arm's memory cap as systemd reads it (8G), its timeout in seconds and how many "
        "of its worlds run at once; one for every arm, for declare",
    )
    parser.add_argument(
        "--cpus",
        nargs=2,
        action="append",
        default=[],
        metavar=("ARM", "CPUS"),
        help="the CPUs a stage reserves for one process of ARM, for declare; a tool's default is "
        "its measured share (scorecard.tools.RESERVED), every other arm's 1",
    )
    parser.add_argument("--from", dest="source", type=Path, help="for reuse: the run to take from")
    parser.add_argument(
        "--accept",
        nargs="+",
        default=[],
        metavar="PATH",
        help="for reuse: the bench's pinned files whose change leaves the tool's default setting "
        "as it ran, by their paths in the bench",
    )
    parser.add_argument("--reserve", default="2G", help="the memory the gate keeps free")
    parser.add_argument(
        "--settle", type=float, default=60.0, help="the seconds a start holds the memory gate"
    )
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--r-library", type=Path, help="Robyn's R library")
    parser.add_argument("--nevergrad", type=Path, help="nevergrad's Python environment, for Robyn")
    parser.add_argument("--world-directory", type=Path, help="where family 15's worlds are")
    parser.add_argument("--default", help="the engine's default procedure, which the claim reads")
    parser.add_argument("--margin", type=float, help="the claim's margin, regret per euro")
    parser.add_argument(
        "--exploratory", action="store_true", help="read the run, and leave it out of the claim"
    )
    parser.add_argument(
        "--cap",
        action="append",
        type=claim.cap_argument,
        default=[],
        metavar="TOOL=WORLDS",
        help="a tool read on its first WORLDS worlds an environment; again for another",
    )
    parser.add_argument(
        "--exploratory-caps",
        action="store_true",
        help="read the capped tools' comparisons, and leave them out of the claim",
    )
    parser.add_argument(
        "--arm", action="append", help="the arm to run, for arm and one; the tools, for reuse"
    )
    parser.add_argument("--environment")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--k", type=int, nargs="+", help="the rungs a world's process records")
    parser.add_argument(
        "--flags",
        type=float,
        metavar="TAU",
        help="for report: read every flag as a classifier, a channel's ROI wrong past TAU",
    )
    parser.add_argument(
        "--columns",
        action="store_true",
        help="for report: the bench's report of the run's family after it",
    )
    parser.add_argument(
        "--json",
        type=Path,
        metavar="PATH",
        help="for report: write the bench's report of the family to PATH, its markdown beside",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stderr)
    layout = Layout(args.root)
    match args.command:
        case "declare":

            def resolved(path: Path | None) -> str | None:
                return None if path is None else str(path.resolve())

            cpus = RESERVED | {arm: float(n) for arm, n in args.cpus}

            declaration = Declaration(
                family=args.family,
                environments=tuple(args.environments),
                worlds=args.worlds,
                pilot=args.pilot,
                confirmatory=args.confirmatory,
                rungs=tuple(args.rungs),
                arms=tuple(args.arms),
                external=tuple(args.external),
                smoke=args.smoke,
                bench=args.bench,
                limits={
                    arm: Limits(
                        memory=memory,
                        timeout=float(timeout),
                        slots=int(slots),
                        cpus=cpus.get(arm, 1.0),
                    )
                    for arm, memory, timeout, slots in args.limit
                },
                reserve=args.reserve,
                settle=args.settle,
                attempts=args.attempts,
                r_library=resolved(args.r_library),
                nevergrad=resolved(args.nevergrad),
                world_directory=resolved(args.world_directory),
                default=args.default,
                margin=args.margin,
                exploratory=args.exploratory,
                caps=dict(args.cap) or None,
                exploratory_caps=args.exploratory_caps,
                preregistration=None
                if args.preregistration is None
                else stamp(args.preregistration),
            )
            declare(layout, declaration)
        case "export":
            export_all(layout)
        case "arm":
            run_arm(layout, _one(parser, args.arm))
        case "one":
            one(layout, _one(parser, args.arm), args.environment, args.seed, args.k)
        case "reuse":
            if args.source is None or not args.arm:
                parser.error("reuse takes --from ROOT and the tools by --arm")
            reuse(layout, Layout(args.source), args.arm, accepted=args.accept)
        case "score":
            score(layout)
        case "report":
            d = layout.read()
            report.main(
                layout,
                [v for a in d.arms for v in ARMS[a]().variants],
                tau=args.flags,
                columns=args.columns,
                saved=args.json,
            )


def _one(parser: argparse.ArgumentParser, arms: Sequence[str] | None) -> str:
    """The one arm `--arm` names, for arm and one."""
    if not arms or len(arms) != 1:
        parser.error("arm and one run one arm: name it once by --arm")
    return arms[0]


if __name__ == "__main__":
    main()
