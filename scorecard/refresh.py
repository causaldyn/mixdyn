"""The refresh cell: this engine's arms fitted again as weeks arrive, read by the bench's refresh
swing and rolling origin (`causaldyn_bench.scorecard.refresh`).

    PYTHONPATH=<bench>/src JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \\
        OPENBLAS_NUM_THREADS=1 PJRT_NPROC=1 \\
        python -m scorecard.refresh <command> --root <dir> ...

* `run --kind KIND --family F --environment E --world S --arm NAME [--k K] [--cut I] [--smoke]`:
  fit one of this engine's arms (`scorecard.run.ARMS`) on every cut of one world in turn, or on its
  cut `I` alone, in one process, and keep each cut's records, one a variant, under
  `<root>/records/`: a `swing` cuts the world's export and plans its one quarter
  (`refresh.windows`), a `rolling` origin cuts the world itself (`refresh.origins`). A cut whose
  records are all in is not fitted again, so a run that stopped resumes. An arm that reads no lift
  test is handed each cut without them, as `scorecard.run` hands it an export
  (`scorecard.arms.base.records`); a fit that raises plays the status quo. A tool's arm
  (`scorecard.tools.TOOLS`) is fitted on each cut as `scorecard.run` runs a tool's world: the cut
  exported as the bench archives it into a directory of its own under `<root>/inputs/`, the bench's
  script run on it from the bench's tree in a scope capped at `--memory` with no swap, stopped at
  `--timeout`, tried twice, and a cut whose process left no record kept as a failure, which plays
  the status quo; Robyn on `--r-library` and `--nevergrad`.
* `cell --plan PLAN.json [--slots N] [--reserve SIZE] [--smoke]`: every job of a cell, `--slots`
  jobs at once, each job one world's arm at one rung and each of its cuts a `run --cut` in a
  process of its own: in a scope capped at the arm's memory with no swap, admitted by a memory
  gate, stopped at the arm's timeout. `PLAN.json` holds `jobs`, each `[kind, family, environment,
  world, k, arm]`; `limits`, each arm's `[memory, timeout]` for one cut's process; and `r_library`
  and `nevergrad` where Robyn is an arm. An engine arm's cut is tried twice, as a tool's is in its
  `run`, and a cut that left no record is kept as a failure of each variant, which plays the status
  quo; a cut whose records are all in starts nothing. CHOICE: a process a cut, not a world, as the
  runner runs a world at a rung: a process holds one fit, so an arm's cap is its one fit's, and a
  process that dies costs one cut;
* `report`: every world whose cuts are all recorded, each variant read by the bench
  (`refresh.swing`, `refresh.rolling`), written to `<root>/refresh.json`, with the provenance of
  the trees and the packages, and its markdown beside it (`refresh.markdown`).

The cell runs on pilot worlds: neither command reads a run's declaration or a pre-registration,
and its numbers state what moved between refits, not a claim.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import causaldyn_bench
from causaldyn_bench.scorecard import refresh
from causaldyn_bench.scorecard.observe import export

from scorecard.arms.base import records
from scorecard.layout import FAMILIES, Limits, size, slug, write_json
from scorecard.provenance import provenance, require, tree
from scorecard.run import ARMS, Gate, _scoped, ending, parsed, wait
from scorecard.tools import TOOLS

if TYPE_CHECKING:
    from causaldyn_bench.scorecard.family import Family
    from causaldyn_bench.scorecard.observe import Observation

    from scorecard.arms.base import Arm

log = logging.getLogger("scorecard.refresh")

KINDS = ("swing", "rolling")
ATTEMPTS = 2
"""The processes a tool's cut runs in before it is kept as a failure, as `scorecard.run` tries a
world."""
EXPORT_SECONDS = 600.0
"""The seconds a tool's cut's `run` takes beyond its tool's processes, its export and its start."""
SETTLE = 5.0
"""The seconds a cell's start holds the memory gate (`Declaration.settle`): a few, as the box's
runs declare it, the starts bounded by `--slots`."""


def cuts(kind: str, family: Family[Any], world: Any, k: int) -> tuple[Observation, ...]:
    """What an arm reads at each of `kind`'s cuts of `world` at rung `k`, first to last.

    Raises:
        ValueError: on a kind that is neither of `KINDS`, and what the bench's cuts refuse.
    """
    if kind == "swing":
        return refresh.windows(family.observe(world, k))
    if kind == "rolling":
        return tuple(family.observe(c, k) for c in refresh.origins(world))
    msg = f"a refresh is one of {KINDS}, not {kind!r}"
    raise ValueError(msg)


def _chosen(
    observations: tuple[Observation, ...], only: int | None
) -> list[tuple[int, Observation]]:
    """Each cut with its index, or cut `only` alone."""
    if only is None:
        return list(enumerate(observations))
    if not 0 <= only < len(observations):
        msg = f"the world has cuts 0 to {len(observations) - 1}, not {only}"
        raise ValueError(msg)
    return [(only, observations[only])]


def directory(root: Path, kind: str, number: int, environment: str, seed: int, k: int) -> Path:
    """Where one world's records of `kind` at rung `k` are kept, a directory a variant."""
    return root / "records" / kind / f"family-{number}" / slug(environment) / f"k{k}" / str(seed)


def fit(
    root: Path,
    kind: str,
    number: int,
    environment: str,
    seed: int,
    k: int,
    arm: Arm[Any],
    *,
    smoke: bool,
    only: int | None = None,
) -> None:
    """`arm`'s records of every cut of family `number`'s world `seed` of `environment` at rung
    `k`, or of its cut `only`, each kept as `<variant>/<cut>.json` under `directory`; a cut whose
    records are all in is left as it is.

    Raises:
        ValueError: on a cut `only` the world does not have.
    """
    family = FAMILIES[number]
    observations = cuts(kind, family, family.world(environment, seed), k)
    place = directory(root, kind, number, environment, seed, k)
    for i, observation in _chosen(observations, only):
        paths = {v: place / slug(v) / f"{i}.json" for v in arm.variants}
        if all(p.exists() for p in paths.values()):
            continue
        made = records(arm, observation, smoke=smoke)
        for variant, record in made.items():
            cut = {"variant": variant, "cut": i, "weeks": observation.sales.shape[0]}
            write_json(paths[variant], record | cut)
        log.info("cut %d of %d recorded", i + 1, len(observations))


def fit_tool(
    root: Path,
    kind: str,
    number: int,
    environment: str,
    seed: int,
    k: int,
    name: str,
    limits: Limits,
    *,
    smoke: bool,
    paths: dict[str, str] | None = None,
    only: int | None = None,
) -> None:
    """The tool `name`'s record of every cut of family `number`'s world `seed` of `environment` at
    rung `k`, or of its cut `only`, each kept as `<name>/<cut>.json` under `directory`, its
    process's own beside it under `<root>/inputs/` and `<root>/logs/`; a cut whose record is in is
    left as it is. `paths` is what the tool's process reads beyond the harness's environment
    (`scorecard.tools.Tool.paths`).

    Raises:
        ValueError: on a cut `only` the world does not have.
    """
    tool = TOOLS[name]
    family = FAMILIES[number]
    observations = cuts(kind, family, family.world(environment, seed), k)
    place = directory(root, kind, number, environment, seed, k)
    handed = place.relative_to(root / "records") / slug(name)
    bench = tree(causaldyn_bench)
    for i, observation in _chosen(observations, only):
        target = place / slug(name) / f"{i}.json"
        if target.exists():
            continue
        inputs = root / "inputs" / handed / str(i)
        log_path = root / "logs" / handed / f"{i}.log"
        export(observation, inputs)  # once: a cut already exported is checked and kept
        made = inputs / f"world_{seed}.json"
        command = [*tool.command, "--worlds", str(inputs), "--out", str(inputs)]
        reasons = []
        record = None
        for _ in range(ATTEMPTS):
            made.unlink(missing_ok=True)
            reasons.append(
                wait(
                    _scoped(limits, [*command, *(["--smoke"] if smoke else [])]),
                    log_path,
                    limits.timeout,
                    cwd=bench,
                    environment=paths,
                )
            )
            record = parsed(made)
            if record is not None:
                break
        if record is None:
            record = {
                "version": observation.version,
                "seed": seed,
                "digest": observation.digest(),
                "error": f"the cut's process ended without its record: {'; '.join(reasons)}",
                "traceback": None,
                "failure_kind": ending(reasons[-1]),
            }
        cut = {"variant": name, "cut": i, "weeks": observation.sales.shape[0]}
        write_json(target, record | cut)
        log.info("cut %d of %d recorded", i + 1, len(observations))


def cell(
    root: Path, plan: dict[str, Any], slots: int, reserve: str = "2G", *, smoke: bool = False
) -> list[str]:
    """Every job of `plan` (see the module), `slots` at once; how many cuts each job recorded, in
    the plan's order.

    Raises:
        ValueError: on a job whose arm the plan gives no limits, or an arm the harness lacks.
    """
    limits = {
        arm: Limits(memory, float(timeout), 1) for arm, (memory, timeout) in plan["limits"].items()
    }
    unknown = sorted({job[5] for job in plan["jobs"]} - limits.keys())
    if unknown:
        msg = f"the plan gives no limits for {unknown}"
        raise ValueError(msg)
    lacking = sorted(limits.keys() - ARMS.keys() - TOOLS.keys())
    if lacking:
        msg = f"the harness runs no arm {lacking}"
        raise ValueError(msg)
    gate = Gate(root / "logs" / "gate.lock", size(reserve), SETTLE)

    def one(job: list[Any]) -> str:
        kind, number, environment, seed, k, arm = job
        held, tool = limits[arm], arm in TOOLS
        command = [sys.executable, "-m", "scorecard.refresh", "run", "--root", str(root)]
        command += ["--kind", kind, "--family", str(number), "--environment", environment]
        command += ["--world", str(seed), "--k", str(k), "--arm", arm]
        command += ["--smoke"] if smoke else []
        # a tool's run tries the tool's process itself: one start, the time of every try
        timeout, attempts = held.timeout, ATTEMPTS
        if tool:
            command += ["--memory", held.memory, "--timeout", str(held.timeout)]
            if TOOLS[arm].r:
                command += ["--r-library", plan["r_library"], "--nevergrad", plan["nevergrad"]]
            timeout, attempts = held.timeout * ATTEMPTS + EXPORT_SECONDS, 1
        family = FAMILIES[number]
        observations = cuts(kind, family, family.world(environment, seed), k)
        place = directory(root, kind, number, environment, seed, k)
        variants = (arm,) if tool else ARMS[arm]().variants
        log_path = root / "logs" / "cell" / kind / f"family-{number}" / slug(environment)
        log_path = log_path / f"k{k}" / f"{seed}-{slug(arm)}.log"
        failed = 0
        for i, observation in enumerate(observations):
            paths = {v: place / slug(v) / f"{i}.json" for v in variants}
            reasons = []
            for _ in range(attempts):
                if all(p.exists() for p in paths.values()):
                    break
                scoped = _scoped(held, [*command, "--cut", str(i)])
                reasons.append(wait(scoped, log_path, timeout, gate=gate, cap=held.cap))
            if all(p.exists() for p in paths.values()):
                continue
            failed += 1
            failure = {
                "version": observation.version,
                "seed": seed,
                "digest": observation.digest(),
                "error": f"the cut's process ended without its records: {'; '.join(reasons)}",
                "traceback": None,
                "failure_kind": ending(reasons[-1]),
                "cut": i,
                "weeks": observation.sales.shape[0],
            }
            for variant, path in paths.items():
                if not path.exists():
                    write_json(path, failure | {"variant": variant})
        return f"{len(observations) - failed} of {len(observations)} cuts recorded"

    with ThreadPoolExecutor(slots) as pool:
        ended = list(pool.map(one, plan["jobs"]))
    for job, outcome in zip(plan["jobs"], ended, strict=True):
        log.info("%s: %s", " ".join(map(str, job)), outcome)
    return ended


def read(root: Path) -> list[dict[str, Any]]:
    """Every world's reading by each variant whose cuts are all recorded under `root`, one row a
    world and a variant; a variant still short of a cut is left out."""
    rows = []
    for place in sorted((root / "records").glob("*/family-*/*/k*/*")):
        kind, number = place.parts[-5], int(place.parts[-4].removeprefix("family-"))
        k, seed = int(place.parts[-2].removeprefix("k")), int(place.parts[-1])
        family = FAMILIES[number]
        environment = next(e for e in family.pilots if slug(e) == place.parts[-3])
        world = family.world(environment, seed)
        count = len(cuts(kind, family, world, k))
        for variant in sorted(p for p in place.iterdir() if p.is_dir()):
            held = [variant / f"{i}.json" for i in range(count)]
            if not all(p.exists() for p in held):
                continue
            made = [json.loads(p.read_text()) for p in held]
            reader = refresh.swing if kind == "swing" else refresh.rolling
            rows.append(
                {
                    "kind": kind,
                    "family": number,
                    "environment": environment,
                    "seed": seed,
                    "k": k,
                    "arm": made[0]["variant"],
                    "smoke": any("smoke" in r for r in made),
                    "reading": reader(family, world, k, made),
                }
            )
    return rows


def report(root: Path) -> dict[str, Any]:
    """The cell's report of every reading under `root`, written to `refresh.json` and its markdown
    to `refresh.md`."""
    data = refresh.report(read(root)) | {"provenance": provenance()}
    write_json(root / "refresh.json", data)
    (root / "refresh.md").write_text(refresh.markdown(data))
    return data


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m scorecard.refresh")
    parser.add_argument("command", choices=("run", "cell", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--kind", choices=KINDS)
    parser.add_argument("--family", type=int)
    parser.add_argument("--environment")
    parser.add_argument("--world", type=int)
    parser.add_argument("--arm", choices=sorted(ARMS.keys() | TOOLS.keys()))
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--cut", type=int, help="for run: that cut alone")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--memory", default="8G", help="a tool's cut's memory cap")
    parser.add_argument("--timeout", type=float, default=7200.0, help="a tool's cut's seconds")
    parser.add_argument("--r-library", help="Robyn's R library")
    parser.add_argument("--nevergrad", help="nevergrad's Python environment, for Robyn")
    parser.add_argument("--plan", type=Path, help="for cell: the cell's jobs and limits")
    parser.add_argument("--slots", type=int, default=1, help="for cell: jobs at once")
    parser.add_argument("--reserve", default="2G", help="for cell: the memory the gate keeps free")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    require(bench=None)
    if args.command == "report":
        report(args.root)
        return
    if args.command == "cell":
        if args.plan is None:
            parser.error("cell needs --plan")
        plan = json.loads(args.plan.read_text())
        cell(args.root, plan, args.slots, args.reserve, smoke=args.smoke)
        return
    needed = ("kind", "family", "environment", "world", "arm")
    missing = [f"--{name}" for name in needed if getattr(args, name) is None]
    if missing:
        parser.error(f"run needs {' '.join(missing)}")
    if args.arm in TOOLS:
        fit_tool(
            args.root,
            args.kind,
            args.family,
            args.environment,
            args.world,
            args.k,
            args.arm,
            Limits(memory=args.memory, timeout=args.timeout, slots=1),
            smoke=args.smoke,
            paths=TOOLS[args.arm].paths(args.r_library, args.nevergrad),
            only=args.cut,
        )
        return
    fit(
        args.root,
        args.kind,
        args.family,
        args.environment,
        args.world,
        args.k,
        ARMS[args.arm](),
        smoke=args.smoke,
        only=args.cut,
    )


if __name__ == "__main__":
    main()
