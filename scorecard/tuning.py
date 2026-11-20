"""Each tuned rival's configuration, chosen on the development worlds by the rival's own criterion,
before any scored world: the tuning's fits and the choice (`scorecard.tools` reads it as the
rival's tuned arm).

    PYTHONPATH=<bench>/src JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \\
        OPENBLAS_NUM_THREADS=1 PJRT_NPROC=1 \\
        python -m scorecard.tuning <command> --rival meridian|pymc-marketing --root <dir> ...

* `jobs --development DIR [--per-environment N]`: the worlds the tuning fits, printed and counted;
* `run --development DIR [--per-environment N] [--memory 8G] [--timeout S] [--slots N]`: every
  candidate of the rival's grid (`scorecard.tools.GRIDS`) on every world `jobs` names, each world
  a process of its own, as `scorecard.run` runs a tool: in a scope capped at `--memory` with no
  swap, admitted by a memory gate, stopped at `--timeout`, tried twice, then recorded as a failure.
  A world whose record is in is not fitted again, so a run that stopped resumes;
* `select [--write]`: the rule below over every record under `<root>`, written to
  `<root>/tuning-<rival>.json` and its markdown beside it; with `--write`, the choice into
  `scorecard/tuned.json`, which the tuned arm reads.

**The worlds.** The development set is M2's scored worlds, the records' root `DIR` of M2's roots.
Within each family and environment, M2's scored seeds sorted ascending, a world whose rank mod 3 is
2 is in the check split, and so is every world of a `-robyn` root; every other world is in the fit
split (`split`). The tuning reads the fit split alone, the first `N` worlds of it in each
environment of M2's national main roots (`ROOTS`), at rung 0, the rung without tests: a
configuration is a choice of model, made before any test is read, and a tuned arm reads each
rung's tests as its rival's arm does. The geo families are not tuned on: their fits cost 3 to 5
national ones, and the tuned arm runs on them at the configuration the national worlds chose.

**Meridian** (`meridian`): every candidate fitted at the arm's sampler with the history's last
tenth of weeks held out of the likelihood (`--holdout 0.1`, a forecast, as the cross-validation
below reads one, since weeks held at random leak through autocorrelation; automatic knot selection
runs on the fitted weeks alone), and read by Meridian's own health card, the model reviewer's:

1. each candidate's pass share, its fits whose overall status is not `FAIL`, and its mean health
   score, a fit that raised counting as a failure of score 0;
2. the candidates within `PASS_SHARE` of the best pass share and `HEALTH` of the best mean health
   score are admissible, the card's own verdict first;
3. among them, the card's out-of-sample fit, the goodness-of-fit check's `wmape_test` on the held
   weeks, on the worlds where every admissible candidate has one: the lowest mean wins, and every
   candidate whose paired mean difference from it is within one paired standard error ties with it;
4. a tie goes to the recommended configuration (`scorecard.tools.RECOMMENDED`), then to the
   candidate that moves fewer settings, then to the lower mean.

**PyMC-Marketing** (`pymc-marketing`): every candidate's time-slice cross-validation by the
library's own `TimeSliceCrossValidator` (the bench's `scripts/pymc_marketing_cv.py`: four quarterly
folds on a 156-week history), read by its mean CRPS on each fold's test weeks, in units of the
history's mean weekly sales:

1. a candidate whose cross-validation failed on more than `FAILED` of the worlds is not admissible;
2. the sampler's mixing gates the rest, as Meridian's card gates Meridian: a world mixed where its
   cross-validation ran and every fold's largest R-hat is at most `RHAT` and its divergent draws at
   most `DIVERGENT` of its kept draws (the folds' flags, as the arm reads its fit's); a candidate
   whose share of mixed worlds lies more than `MIXED` below the best candidate's is not admissible;
3. among the rest, the mean over the worlds where every admissible candidate has one: the lowest
   wins, and every candidate within one paired standard error ties with it;
4. a tie goes as Meridian's does.

**Limitations.** Every candidate of a grid fits within `scorecard.tools.COST_CAP` times its
rival's recommended settings' CPU a fit, alike for every rival, so the tuning searches only
configurations a scored run can afford (PyMC-Marketing's Hill saturation with a varying intercept
is out). The tuning reads national worlds at rung 0 and its choice holds at every rung and on the
geo families. The mixing gate is relative, as the card's is: 5.6 % of M2's fits of the
recommended settings on its national worlds at rung 0 are past R-hat 1.1, at the arm's full
draws, and every one reports divergent draws, so a gate on a fixed share of 5 % would hold out the
recommended settings themselves.

The rule reads no record of the bench's scoring and no world's truth: it chooses by what a
practitioner of each tool sees. Its fits are never scored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import causaldyn_bench

from scorecard.layout import Limits, size, slug, write_json
from scorecard.provenance import provenance, require, tree
from scorecard.run import Gate, _scoped, ending, parsed, wait
from scorecard.tools import GRIDS, RECOMMENDED, TUNED, Tool, pinned

log = logging.getLogger("scorecard.tuning")

ROOTS = ("f0", "f0ref", "f1", "f2", "f3", "f4", "f5", "f6", "f7a", "f7b", "f14", "f15")
"""M2's national main roots, whose fit split the tuning reads."""
RIVALS = tuple(GRIDS)
HOLDOUT = "0.1"
"""The share of the history's last weeks a Meridian tuning fit leaves out of the likelihood."""
PASS_SHARE = 0.05
HEALTH = 1.0
FAILED = 0.05
RHAT = 1.1
DIVERGENT = 0.1
"""The most divergent draws a fold mixes with, as a share of its kept draws. The arm states no
bound, and PyMC warns at any divergence, which every one of M2's fits of the recommended settings
reports (a median 3.5 % of its draws, its 95th percentile 11.9 %, 7.3 % of them past a tenth):
the bound holds out a sampler that broke, not the recommended settings' usual divergences."""
MIXED = 0.05
ATTEMPTS = 2
PER_ENVIRONMENT = 5
"""The fit split's worlds an environment the tuning reads by default: 140 worlds over M2's 28
national environments."""


def command(rival: str, candidate: str) -> Tool:
    """The tuning's fit of `candidate` of `rival`'s grid: Meridian's arm with its holdout, or
    PyMC-Marketing's cross-validation in the arm's environment.

    Raises:
        KeyError: on a rival without a grid or a candidate outside it.
    """
    options = GRIDS[rival][candidate]
    if rival == "meridian":
        return pinned("meridian_arm", *options, "--holdout", HOLDOUT)
    validation = pinned("pymc_marketing_arm").command
    script = validation.index("scripts/pymc_marketing_arm.py")
    return Tool((*validation[:script], "scripts/pymc_marketing_cv.py", *options))


def _seeds(directory: Path) -> list[int]:
    return sorted(int(p.stem.removeprefix("world_")) for p in directory.glob("world_*.npz"))


def environments(development: Path, root: str) -> dict[tuple[int, str], list[int]]:
    """Each family and environment of one of M2's roots, with the seeds it exported at rung 0."""
    out = {}
    for place in sorted((development / root / "exports").glob("family-*/*/k0")):
        family = int(place.parts[-3].removeprefix("family-"))
        out[family, place.parts[-2]] = _seeds(place)
    return out


def split(development: Path) -> dict[tuple[int, str, int], str]:
    """Every scored world of M2 in the development set, by family, environment and seed: `fit` or
    `check`, by the rule of the module. A world held by a `-robyn` root is `check`."""
    scored: dict[tuple[int, str], set[int]] = {}
    robyn: set[tuple[int, str, int]] = set()
    for root in sorted(p.name for p in development.iterdir() if (p / "exports").is_dir()):
        for (family, environment), seeds in environments(development, root).items():
            if root.endswith("-robyn"):
                robyn |= {(family, environment, s) for s in seeds}
            scored.setdefault((family, environment), set()).update(seeds)
    out = {}
    for (family, environment), seeds in scored.items():
        for rank, seed in enumerate(sorted(seeds)):
            world = (family, environment, seed)
            out[world] = "check" if rank % 3 == 2 or world in robyn else "fit"
    return out


@dataclass(frozen=True, slots=True)
class Job:
    """One world the tuning fits: its root in the development set, family, environment, seed."""

    root: str
    family: int
    environment: str
    seed: int

    def export(self, development: Path) -> Path:
        place = development / self.root / "exports" / f"family-{self.family}" / self.environment
        return place / "k0" / f"world_{self.seed}.npz"


def jobs(development: Path, per_environment: int = PER_ENVIRONMENT) -> list[Job]:
    """The first `per_environment` worlds of the fit split in each environment of `ROOTS`, seeds
    ascending.

    Raises:
        ValueError: on an environment whose fit split holds fewer worlds.
    """
    splits = split(development)
    out = []
    for root in ROOTS:
        for (family, environment), seeds in environments(development, root).items():
            fit = [s for s in seeds if splits[family, environment, s] == "fit"]
            if len(fit) < per_environment:
                msg = f"{root} {environment} holds {len(fit)} fit worlds, not {per_environment}"
                raise ValueError(msg)
            out += [Job(root, family, environment, s) for s in fit[:per_environment]]
    return out


def record_path(root: Path, rival: str, candidate: str, job: Job) -> Path:
    place = root / "records" / slug(rival) / slug(candidate) / f"family-{job.family}"
    return place / job.environment / f"world_{job.seed}.json"


def fit(
    root: Path,
    rival: str,
    development: Path,
    limits: Limits,
    *,
    per_environment: int = PER_ENVIRONMENT,
    reserve: str = "2G",
    smoke: bool = False,
) -> None:
    """Every candidate of `rival` on every world of `jobs` whose record is not in, each world a
    process of its own by `limits`, recorded as a failure of its kind after `ATTEMPTS`."""
    require(bench=None)
    gate = Gate(root / "logs" / "gate.lock", size(reserve), 5.0)
    bench = tree(causaldyn_bench)
    pending = [
        (candidate, job)
        for candidate in GRIDS[rival]
        for job in jobs(development, per_environment)
        if parsed(record_path(root, rival, candidate, job)) is None
    ]
    log.info("%s: %d fits to run, %d at a time", rival, len(pending), limits.slots)

    def one(item: tuple[str, Job]) -> str:
        candidate, job = item
        target = record_path(root, rival, candidate, job)
        inputs = root / "inputs" / target.relative_to(root / "records").with_suffix("")
        log_path = root / "logs" / target.relative_to(root / "records").with_suffix(".log")
        inputs.mkdir(parents=True, exist_ok=True)
        handed = inputs / f"world_{job.seed}.npz"
        if not handed.exists():  # a copy: the development set's records are never linked to
            shutil.copyfile(job.export(development), handed)
        made = inputs / f"world_{job.seed}.json"
        argv = [*command(rival, candidate).command, "--worlds", str(inputs), "--out", str(inputs)]
        reasons, began = [], time.perf_counter()
        for _ in range(ATTEMPTS):
            made.unlink(missing_ok=True)
            ended = wait(
                _scoped(limits, [*argv, *(["--smoke"] if smoke else [])]),
                log_path,
                limits.timeout,
                cwd=bench,
                gate=gate,
                cap=limits.cap,
            )
            reasons.append(ended)
            if parsed(made) is not None:
                target.parent.mkdir(parents=True, exist_ok=True)
                made.replace(target)
                return "recorded"
        write_json(
            target,
            {
                "seed": job.seed,
                "error": f"the world's process ended without its record: {'; '.join(reasons)}",
                "failure_kind": ending(reasons[-1]),
                "cost": {"wall_seconds": time.perf_counter() - began},
            },
        )
        return "failed: " + "; ".join(reasons)

    with ThreadPoolExecutor(limits.slots) as pool:
        for item, outcome in zip(pending, pool.map(one, pending), strict=True):
            log.info("%s %s %s %d: %s", rival, item[0], item[1].environment, item[1].seed, outcome)


def _criterion(rival: str, record: dict[str, Any]) -> dict[str, Any]:
    """What the rule reads of one tuning record: Meridian's card, PyMC-Marketing's CRPS."""
    failed = record.get("error") is not None
    if rival == "meridian":
        if failed:
            return {"failed": True, "pass": False, "health": 0.0, "out": None}
        review = record["review"]
        fitness = review["checks"].get("GoodnessOfFitCheck", {}).get("details", {})
        out = fitness.get("wmape_test") if isinstance(fitness, dict) else None
        return {
            "failed": False,
            "pass": review["overall"] != "FAIL",
            "health": float(review["health_score"] or 0.0),
            "out": None if out is None else float(out),
        }
    if failed:
        return {"failed": True, "mixed": False, "out": None}
    flags = [fold["flags"] for fold in record["folds"]]
    mixed = all(
        f["max_rhat"] <= RHAT and f["divergences"] <= DIVERGENT * f["kept_draws"] for f in flags
    )
    return {"failed": False, "mixed": mixed, "out": float(record["mean_test_crps_scaled"])}


def read(root: Path, rival: str) -> dict[str, dict[tuple[int, str, int], dict[str, Any]]]:
    """Every candidate's tuning records under `root`, read by `_criterion`, by world."""
    out: dict[str, dict[tuple[int, str, int], dict[str, Any]]] = {}
    for candidate in GRIDS[rival]:
        place = root / "records" / slug(rival) / slug(candidate)
        held = {}
        for path in sorted(place.glob("family-*/*/world_*.json")):
            record = parsed(path)
            if record is None:
                continue
            family = int(path.parts[-3].removeprefix("family-"))
            world = (family, path.parts[-2], int(path.stem.removeprefix("world_")))
            held[world] = _criterion(rival, record)
        out[candidate] = held
    return out


def _moved(rival: str, candidate: str) -> int:
    return len(GRIDS[rival][candidate]) // 2


def choose(rival: str, read_records: dict[str, dict[Any, dict[str, Any]]]) -> dict[str, Any]:
    """The rule of the module over each candidate's read records: the candidate chosen, each
    candidate's summary, and why each candidate that lost lost.

    Raises:
        ValueError: on a candidate with no record, or no admissible candidate.
    """
    empty = [c for c, held in read_records.items() if not held]
    if empty:
        msg = f"the tuning holds no record of {empty}"
        raise ValueError(msg)
    summary: dict[str, dict[str, Any]] = {}
    for candidate, held in read_records.items():
        rows = list(held.values())
        failed = sum(r["failed"] for r in rows) / len(rows)
        summary[candidate] = {"worlds": len(rows), "failed": failed}
        if rival == "meridian":
            summary[candidate] |= {
                "pass": sum(r["pass"] for r in rows) / len(rows),
                "health": sum(r["health"] for r in rows) / len(rows),
            }
    if rival == "meridian":
        best_pass = max(s["pass"] for s in summary.values())
        best_health = max(s["health"] for s in summary.values())
        admissible = [
            c
            for c, s in summary.items()
            if s["pass"] >= best_pass - PASS_SHARE and s["health"] >= best_health - HEALTH
        ]
        lost = {c: "the health card" for c in summary if c not in admissible}
    else:
        for candidate, held in read_records.items():
            summary[candidate]["mixed"] = sum(r["mixed"] for r in held.values()) / len(held)
        why = f"failed on more than {FAILED:.0%} of the worlds"
        lost = {c: why for c, s in summary.items() if s["failed"] > FAILED}
        best_mixed = max((s["mixed"] for c, s in summary.items() if c not in lost), default=0.0)
        for c, s in summary.items():
            if c not in lost and s["mixed"] < best_mixed - MIXED:
                lost[c] = "the sampler's mixing"
        admissible = [c for c in summary if c not in lost]
    if not admissible:
        msg = f"no candidate of {rival} is admissible: {summary}"
        raise ValueError(msg)
    worlds = sorted(
        set.intersection(
            *({w for w, r in read_records[c].items() if r["out"] is not None} for c in admissible)
        )
    )
    if len(worlds) < 2:
        msg = f"{rival}'s admissible candidates share {len(worlds)} worlds out of sample"
        raise ValueError(msg)
    values = {c: [read_records[c][w]["out"] for w in worlds] for c in admissible}
    means = {c: sum(v) / len(v) for c, v in values.items()}
    best = min(admissible, key=lambda c: (means[c], _moved(rival, c)))
    tied = []
    for c in admissible:
        differences = [a - b for a, b in zip(values[c], values[best], strict=True)]
        mean = sum(differences) / len(differences)
        spread = sum((d - mean) ** 2 for d in differences) / (len(differences) - 1)
        error = math.sqrt(spread / len(differences))
        summary[c] |= {"out": means[c], "difference": mean, "error": error}
        if mean <= error:
            tied.append(c)
        else:
            lost[c] = "out of sample, past one standard error of the best"
    recommended = RECOMMENDED[rival]
    chosen = (
        recommended
        if recommended in tied
        else min(tied, key=lambda c: (_moved(rival, c), means[c]))
    )
    for c in tied:
        if c != chosen:
            lost[c] = "tied, and the tie went to another"
    return {
        "rival": rival,
        "candidate": chosen,
        "options": list(GRIDS[rival][chosen]),
        "best": best,
        "tied": tied,
        "worlds out of sample": len(worlds),
        "summary": summary,
        "lost": lost,
    }


def fingerprint(root: Path, rival: str) -> str:
    """The SHA-256 of every tuning record of `rival` under `root`, by path and bytes, so that the
    choice names what it read."""
    sha = hashlib.sha256()
    for path in sorted((root / "records" / slug(rival)).rglob("world_*.json")):
        sha.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes() + b"\0")
    return sha.hexdigest()


def markdown(chosen: dict[str, Any]) -> str:
    """The choice as a table, a candidate a row."""
    rival = chosen["rival"]
    meridian = rival == "meridian"
    gate = "| pass | health |" if meridian else "| mixed |"
    lines = [
        f"## {rival}: {chosen['candidate']}",
        "",
        f"| candidate | worlds | failed {gate} out of sample | vs best | 1 SE | |",
        "|---|---|---|---|---|---|---|---|---|"
        if meridian
        else "|---|---|---|---|---|---|---|---|",
    ]
    for c, s in chosen["summary"].items():
        card = f" {s['pass']:.3f} | {s['health']:.2f} |" if meridian else f" {s['mixed']:.3f} |"
        fitted = (
            f" {s['out']:.4f} | {s['difference']:+.4f} | {s['error']:.4f} |"
            if "out" in s
            else " n/a | n/a | n/a |"
        )
        verdict = "chosen" if c == chosen["candidate"] else chosen["lost"].get(c, "")
        lines.append(f"| {c} | {s['worlds']} | {s['failed']:.3f} |{card}{fitted} {verdict} |")
    return "\n".join(lines) + "\n"


def select(root: Path, rival: str, *, write: bool = False) -> dict[str, Any]:
    """The rule over `rival`'s records under `root`, kept as `tuning-<rival>.json` and `.md`; with
    `write`, the choice kept in `scorecard/tuned.json` too."""
    chosen = choose(rival, read(root, rival)) | {"records": fingerprint(root, rival)}
    write_json(root / f"tuning-{rival}.json", chosen | {"provenance": provenance()})
    (root / f"tuning-{rival}.md").write_text(markdown(chosen))
    if write:
        held = json.loads(TUNED.read_text()) if TUNED.exists() else {}
        TUNED.write_text(json.dumps(held | {rival: chosen}, indent=1, sort_keys=True) + "\n")
    return chosen


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m scorecard.tuning")
    parser.add_argument("command", choices=("jobs", "run", "select"))
    parser.add_argument("--rival", choices=RIVALS, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--development", type=Path, help="M2's records' root of roots")
    parser.add_argument("--per-environment", type=int, default=PER_ENVIRONMENT)
    parser.add_argument("--memory", default="8G")
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument("--slots", type=int, default=1)
    parser.add_argument("--reserve", default="2G")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--write", action="store_true", help="for select: into tuned.json")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if args.command == "select":
        chosen = select(args.root, args.rival, write=args.write)
        sys.stdout.write(markdown(chosen))
        return
    if args.development is None:
        parser.error(f"{args.command} needs --development")
    if args.command == "jobs":
        held = jobs(args.development, args.per_environment)
        for job in held:
            sys.stdout.write(f"{job.root} {job.family} {job.environment} {job.seed}\n")
        fits = len(held) * len(GRIDS[args.rival])
        sys.stdout.write(f"{len(held)} worlds, {fits} fits of {args.rival}\n")
        return
    limits = Limits(memory=args.memory, timeout=args.timeout, slots=args.slots)
    fit(
        args.root,
        args.rival,
        args.development,
        limits,
        per_environment=args.per_environment,
        reserve=args.reserve,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
