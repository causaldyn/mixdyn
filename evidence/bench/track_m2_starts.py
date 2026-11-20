"""Where the Bayesian engine's chains start, the pilot that decides ``fit_bayes``' default
start (mixdyn's ADR 0012).

Every world of the design's pilot seeds, both environments, is fitted by ``fit_bayes`` as
``track_m2_bayes.py``'s ``untested`` arm fits it: the world's raw panel without its geo tests, each
channel's curve a Hill, four chains of 1000 draws after 1000 of warm-up, seeded by the world's
seed; once from each start:

* ``uniform``: ``UniformStart()``, each chain uniform in ``[-2, 2]`` on the sampling scale, as the
  first pilot's fits started (``results/track_m2_bayes_score_pilot.json``);
* ``pathfinder``: ``PathfinderStart()``, each chain at a draw of four Pathfinder paths.

**The rule**, stated before the pilot ran: ``PathfinderStart`` becomes ``fit_bayes``' default if,
over the 60 worlds, fewer of its fits than of the uniform starts' have a parameter whose R-hat is
above ``RHAT``, a fit that fails counting as one, and its fits' divergent transitions, summed, are
not more than the uniform starts'. Otherwise the default stays uniform.

Recorded beside and read by no rule: each fit's sampler health as ``track_m2_bayes.fitted``
records it, with its wall time, a loaded machine's and not a benchmark; and its plan by the bench's
planner at the posterior medians, scored as the ``untested`` arm is.

Each world's record is appended to ``<out>.jsonl`` as it lands, and a run restarted with the same
``--out`` fits only the worlds not there. A world whose oracle is not the budgets run's stops the
run.

    PYTHONPATH=<bench>/src JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \\
        uv run --project <mixdyn> --no-sync python evidence/bench/track_m2_starts.py \\
        score --out <json> [--workers 3]
    PYTHONPATH=<bench>/src uv run --project <mixdyn> --no-sync \\
        python evidence/bench/track_m2_starts.py report <json>
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from causaldyn_bench.budget_regret import PILOTS, TIE, Comparison, Mean, planned
from causaldyn_bench.external_arms import score_arm
from causaldyn_bench.mmm_decision import Quarter, oracle, regret
from track_m2_bayes import fitted, spec_of
from track_m2_ridge import BUDGETS, NATIONAL, committed, panel_of, provenance

from mixdyn.models.bayes_mmm import BayesFit, PathfinderStart, UniformStart, fit_bayes

STARTS = {"uniform": UniformStart(), "pathfinder": PathfinderStart()}
RHAT = 1.01


def started_world(job: tuple[str, int]) -> dict[str, Any]:
    """Pilot environment ``job[0]``'s world from seed ``job[1]``, fitted from each start.

    Raises:
        RuntimeError: the oracle is not the budgets run's, so the regrets would be scored apart
            from the bench's; or a plan beat the oracle.
    """
    name, seed = job
    environment = next(e for e in PILOTS if e.name == name)
    world = environment.generator(seed).simulate(seed)
    quarter = Quarter.after(world)
    best = oracle(world, quarter)
    status_quo = regret(world, quarter, quarter.status_quo, best) / quarter.budget
    theirs = committed(BUDGETS[True])[name][seed]
    if (
        abs(best.worth - theirs["best"]) > TIE * quarter.budget
        or abs(status_quo - theirs["regret"]["status quo"]) > TIE
    ):
        msg = f"{name} world {seed}: the oracle is not the budgets run's"
        raise RuntimeError(msg)
    panel, spec = panel_of(world), spec_of(world)
    out: dict[str, Any] = {"environment": name, "seed": seed}
    for label, start in STARTS.items():
        began = time.perf_counter()
        fit: BayesFit | None = None
        try:
            fit = fit_bayes(panel, NATIONAL, spec, seed=seed, start=start)
            health = fitted(fit, time.perf_counter() - began)
        except (RuntimeError, ValueError) as error:
            health = {"error": str(error)[:200]}
        if fit is None:
            record: dict[str, Any] = {"error": health["error"]}
        else:
            try:
                record = {"error": None, "weekly": planned(list(fit.channels().values()), quarter)}
            except (RuntimeError, ValueError) as error:
                record = {"error": str(error)[:200]}
        score = score_arm(world, quarter, best, record)
        out[label] = {
            "regret": score.regret,
            "failure": score.failure,
            "moved": score.moved,
            "bounds": score.bounds,
            "weekly": None if record["error"] else np.asarray(record["weekly"]).tolist(),
        } | health
    print(
        json.dumps(
            {"environment": name, "seed": seed}
            | {label: [out[label].get("rhat"), out[label].get("divergent")] for label in STARTS}
        ),
        flush=True,
    )
    return out


def score(out: Path, workers: int) -> None:
    """Fit every pilot world not yet in ``out``'s ``.jsonl`` from each start, then write ``out``."""
    source = provenance()
    partial = out.with_suffix(".jsonl")
    done = {}
    if partial.exists():
        for line in partial.read_text().splitlines():
            world = json.loads(line)
            done[world["environment"], world["seed"]] = world
    jobs = [(e.name, s) for e in PILOTS for s in e.seeds if (e.name, s) not in done]
    with ProcessPoolExecutor(workers) as pool, partial.open("a") as sink:
        for future in as_completed([pool.submit(started_world, job) for job in jobs]):
            world = future.result()
            sink.write(json.dumps(world) + "\n")
            sink.flush()
            done[world["environment"], world["seed"]] = world
    out.write_text(
        json.dumps(
            source
            | {
                "pilot": True,
                "starts": list(STARTS),
                "rhat": RHAT,
                "environments": [
                    {"name": e.name, "first": e.first, "worlds": e.worlds} for e in PILOTS
                ],
                "worlds": sorted(done.values(), key=lambda w: (w["environment"], w["seed"])),
            },
            indent=1,
        )
    )


def unmixed(fit: dict[str, Any]) -> bool:
    return "rhat" not in fit or fit["rhat"] > RHAT


def report(record: dict[str, Any]) -> None:
    worlds = record["worlds"]
    print(f"mixdyn {record['mixdyn']['commit'][:7]}; {len(worlds)} pilot worlds")
    count, divergent = {}, {}
    for label in STARTS:
        fits = [w[label] for w in worlds]
        healthy = [f for f in fits if "rhat" in f]
        count[label] = sum(unmixed(f) for f in fits)
        divergent[label] = sum(f["divergent"] for f in healthy)
        print(
            f"\n{label}: {count[label]} fits with an R-hat above {RHAT} or failed"
            f" ({len(fits) - len(healthy)} failed): "
            f"{[(w['environment'][:9], w['seed']) for w in worlds if unmixed(w[label])]}"
        )
        print(
            f"  {divergent[label]} divergent transitions in"
            f" {sum(f['divergent'] > 0 for f in healthy)} fits; largest R-hat"
            f" {max(f['rhat'] for f in healthy):.4f}, least bulk ESS"
            f" {min(f['ess_bulk'] for f in healthy):.0f}, least E-BFMI"
            f" {min(f['e_bfmi'] for f in healthy):.2f}, median"
            f" {np.median([f['seconds'] for f in healthy]):.0f} s"
        )
    adopted = (
        count["pathfinder"] < count["uniform"] and divergent["pathfinder"] <= divergent["uniform"]
    )
    print(
        "\nrule: PathfinderStart becomes the default"
        if adopted
        else "\nrule: the default stays uniform"
    )
    for environment in record["environments"]:
        name = environment["name"]
        ours = {w["seed"]: w for w in worlds if w["environment"] == name}
        print(f"\n{name}: {len(ours)} worlds of {environment['worlds']} (read by no rule)")
        columns = {label: np.array([w[label]["regret"] for w in ours.values()]) for label in STARTS}
        for label, regrets in columns.items():
            mean = Mean.of(regrets)
            print(
                f"  {label}: {mean.mean:+.4f} [{mean.interval[0]:+.4f}, {mean.interval[1]:+.4f}],"
                f" median {mean.median:.4f}"
            )
        comparison = Comparison.of("uniform", columns["pathfinder"], columns["uniform"])
        low, high = comparison.difference.interval
        print(
            f"  pathfinder less uniform: {comparison.difference.mean:+.4f} [{low:+.4f},"
            f" {high:+.4f}]; lower {comparison.lower}, tied {comparison.tied}, higher"
            f" {comparison.higher}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("score", help="every pilot world fitted from each start")
    plan.add_argument("--out", type=Path, required=True)
    plan.add_argument("--workers", type=int, default=3)
    show = sub.add_parser("report")
    show.add_argument("record", type=Path)
    args = parser.parse_args()
    if args.command == "report":
        report(json.loads(args.record.read_text()))
        return
    if jnp.zeros(()).dtype != jnp.float64:
        raise SystemExit("JAX_ENABLE_X64=1 is required: the bench scored at float64")
    score(args.out, args.workers)


if __name__ == "__main__":
    main()
