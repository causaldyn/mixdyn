"""mixdyn's Bayesian engine planning Track M v2's quarter: the pilot of its arms, and its
scored run.

Every arm fits the world's raw panel with ``fit_bayes`` (mixdyn's ADR 0010): each channel's return,
carryover and Hill, the promotion's coefficient held at or above 0 and the price's free, a trend of
3 knots and a yearly season of 3 harmonics, as the ridge engine's arms read the panel; normal noise;
Meridian's default prior on each channel's return; four chains of 1000 draws after 1000 of
warm-up, seeded by the world's seed.

* ``untested``: the fit without the geo tests, planned by the bench's planner (``planned``) on each
  channel at its parameters' posterior medians (``BayesFit.channels``).
* ``plug-in``: the fit with every test in its likelihood, planned the same way.
* ``expected``: the same fit, planned for the posterior's expected return (``expected``): the
  weekly spend in the quarter's box at its budget that maximises the mean, over ``DRAWS`` draws
  thinned evenly from the chains, of the return each draw's channels make over the quarter and the
  carryover after it, the history's carryover included. The return is separable, so each
  channel's mean return is read on a grid of ``GRID`` weekly spends across its box and the
  budget is spent down the concave envelopes' slopes, steepest first.

**The rule**, stated before the pilot ran: the gated arm is ``expected`` unless the drawn pilot's
paired 95 % interval of ``expected`` less ``plug-in`` lies above 0, and then ``plug-in``. The
posterior's expected return is the decision a Bayesian model is fitted for; the plug-in is the
shortcut, kept only if the pilot shows the expectation costs regret.

``score --pilot`` runs the three arms on the design's pilot seeds, both environments, beside the
bench's pilots; ``score`` on the budgets run's scored worlds, beside its CHC, PyMC-Marketing,
observational model and status quo and the bench's Meridian and Robyn, as
``track_m2_ridge.py``'s scored run reads them. Each world's score is appended to ``<out>.jsonl``
as it lands, and a run restarted with the same ``--out`` plans only the worlds not there. A world
whose oracle is not the budgets run's stops the run.

    PYTHONPATH=<bench>/src JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \\
        uv run --project <mixdyn checkout> --no-sync python evidence/bench/track_m2_bayes.py \\
        score [--pilot] --out <json> [--workers 3]
    PYTHONPATH=<bench>/src uv run --project <mixdyn checkout> --no-sync \\
        python evidence/bench/track_m2_bayes.py report <json>
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
from causaldyn_bench.budget_regret import (
    ENVIRONMENTS,
    PILOTS,
    TIE,
    Comparison,
    Mean,
    experiments,
    planned,
)
from causaldyn_bench.external_arms import score_arm, verdict
from causaldyn_bench.mmm_decision import PLANNED, Quarter, oracle, regret
from track_m2_ridge import (
    BENCH,
    BUDGETS,
    EXTERNAL,
    EXTERNALS,
    GATED,
    KNOTS,
    NATIONAL,
    READ,
    committed,
    panel_of,
    provenance,
)

from mixdyn import Channel, MediaPanel
from mixdyn.decisions import Range, Readings, plan_posterior, readings
from mixdyn.models.bayes_mmm import BayesFit, BayesMedia, BayesSpec, fit_bayes
from mixdyn.models.ridge_mmm import Control, FourierSeason, LinearTrend

ARMS = ("untested", "plug-in", "expected")
DRAWS = 400
"""The posterior draws ``expected`` averages the return over, thinned evenly from the chains."""
GRID = 2048
"""The weekly spends across a channel's box at which ``expected`` reads its mean return."""


def spec_of(world) -> BayesSpec:
    return BayesSpec(
        media=tuple(BayesMedia(c) for c in world.channels),
        controls=(Control("promotion", lower=0.0), Control("price")),
        trend=LinearTrend(KNOTS),
        season=FourierSeason(365.25 / 7.0, 3),
    )


def expected(posterior: Readings, panel: MediaPanel, quarter: Quarter) -> np.ndarray:
    """The weekly spend in the quarter's box, at its budget, that maximises the mean return over
    ``posterior``'s draws: mixdyn's ``decisions.plan_posterior`` on a grid of ``GRID`` weekly spends
    across each channel's box, the planner this harness held until mixdyn took it over."""
    limits = {
        Channel(c): Range(float(low), float(high))
        for c, low, high in zip(posterior.channels, quarter.lower, quarter.upper, strict=True)
    }
    return plan_posterior(
        panel, NATIONAL, posterior, quarter.budget, PLANNED, limits, grid=GRID
    ).spend


def fitted(fit: BayesFit, seconds: float) -> dict[str, Any]:
    """What a score keeps of a fit: its sampler's health and its wall time."""
    return {
        "seconds": seconds,
        "divergent": int(fit.divergent.sum()),
        "rhat": max(float(np.max(d.rhat)) for d in fit.diagnostics.values()),
        "ess_bulk": min(float(np.min(d.ess_bulk)) for d in fit.diagnostics.values()),
        "e_bfmi": float(np.min(fit.e_bfmi)),
    }


def scored_world(job: tuple[str, int, bool]) -> dict[str, Any]:
    """Environment ``job[0]``'s world from seed ``job[1]``, a pilot's if ``job[2]``, planned by the
    three arms and scored as the bench scores its external arms.

    Raises:
        RuntimeError: the oracle is not the budgets run's, so the two sides of a comparison would
            be scored apart; or a plan beat the oracle.
    """
    name, seed, pilot = job
    environment = next(e for e in (PILOTS if pilot else ENVIRONMENTS) if e.name == name)
    world = environment.generator(seed).simulate(seed)
    tests = experiments(world, seed)
    quarter = Quarter.after(world)
    best = oracle(world, quarter)
    status_quo = regret(world, quarter, quarter.status_quo, best) / quarter.budget
    theirs = committed(BUDGETS[pilot])[name][seed]
    if (
        abs(best.worth - theirs["best"]) > TIE * quarter.budget
        or abs(status_quo - theirs["regret"]["status quo"]) > TIE
    ):
        msg = f"{name} world {seed}: the oracle is not the budgets run's"
        raise RuntimeError(msg)
    panel, spec = panel_of(world), spec_of(world)
    out: dict[str, Any] = {"environment": name, "seed": seed}
    fits: dict[bool, tuple[BayesFit | None, dict[str, Any]]] = {}
    for tested in (False, True):
        start = time.perf_counter()
        try:
            fit = fit_bayes(panel, NATIONAL, spec, seed=seed, tests=tests if tested else None)
            fits[tested] = fit, fitted(fit, time.perf_counter() - start)
        except (RuntimeError, ValueError) as error:
            fits[tested] = None, {"error": str(error)[:200]}
    for arm in ARMS:
        fit, health = fits[arm != "untested"]
        if fit is None:
            record: dict[str, Any] = {"error": health["error"]}
        else:
            try:
                plan = (
                    expected(readings(fit, draws=DRAWS), panel, quarter)
                    if arm == "expected"
                    else planned(list(fit.channels().values()), quarter)
                )
                record = {"error": None, "weekly": plan}
            except (RuntimeError, ValueError) as error:
                record = {"error": str(error)[:200]}
        score = score_arm(world, quarter, best, record)
        out[arm] = {
            "regret": score.regret,
            "failure": score.failure,
            "moved": score.moved,
            "bounds": score.bounds,
            "weekly": None if record["error"] else np.asarray(record["weekly"]).tolist(),
        } | health
    print(
        json.dumps({"environment": name, "seed": seed} | {a: out[a]["regret"] for a in ARMS}),
        flush=True,
    )
    return out


def score(out: Path, workers: int, *, pilot: bool) -> None:
    """Plan and score every world of the run not yet in ``out``'s ``.jsonl``, then write ``out``."""
    source = provenance()
    partial = out.with_suffix(".jsonl")
    done = {}
    if partial.exists():
        for line in partial.read_text().splitlines():
            world = json.loads(line)
            done[world["environment"], world["seed"]] = world
    jobs = [
        (e.name, s, pilot)
        for e in (PILOTS if pilot else ENVIRONMENTS)
        for s in e.seeds
        if (e.name, s) not in done
    ]
    with ProcessPoolExecutor(workers) as pool, partial.open("a") as sink:
        for future in as_completed([pool.submit(scored_world, job) for job in jobs]):
            world = future.result()
            sink.write(json.dumps(world) + "\n")
            sink.flush()
            done[world["environment"], world["seed"]] = world
    out.write_text(
        json.dumps(
            source
            | {
                "pilot": pilot,
                "draws": DRAWS,
                "grid": GRID,
                "environments": [
                    {"name": e.name, "first": e.first, "worlds": e.worlds}
                    for e in (PILOTS if pilot else ENVIRONMENTS)
                ],
                "worlds": sorted(done.values(), key=lambda w: (w["environment"], w["seed"])),
            },
            indent=1,
        )
    )


def report(record: dict[str, Any]) -> None:
    pilot = record["pilot"]
    budgets = committed(BUDGETS[pilot])
    external = BENCH / EXTERNALS[pilot]
    print(f"mixdyn {record['mixdyn']['commit'][:7]}; {'pilot' if pilot else 'scored'} worlds")
    for environment in record["environments"]:
        name = environment["name"]
        ours = {w["seed"]: w for w in record["worlds"] if w["environment"] == name}
        columns = {arm: {s: w[arm]["regret"] for s, w in ours.items()} for arm in ARMS}
        columns |= {arm: {s: budgets[name][s]["regret"][arm] for s in ours} for arm in READ}
        planned_by = committed(EXTERNALS[pilot]).get(name, {}) if external.exists() else {}
        for arm in EXTERNAL:
            columns[arm] = {
                s: w["arms"][arm]["regret"]
                for s, w in planned_by.items()
                if s in ours and arm in w["arms"]
            }
        print(f"\n{name}: {len(ours)} worlds of {environment['worlds']}")
        for arm in ARMS:
            failed = [s for s, w in ours.items() if w[arm]["failure"]]
            moved = [s for s, w in ours.items() if w[arm]["moved"]]
            print(f"  {arm} failed on {failed}, moved on {moved}")
        for tested in ("untested", "plug-in"):
            fits = [w[tested] for w in ours.values() if "seconds" in w[tested]]
            print(
                f"  {tested} fits: {sum(f['divergent'] > 0 for f in fits)} with divergences,"
                f" largest R-hat {max(f['rhat'] for f in fits):.4f}, least bulk ESS"
                f" {min(f['ess_bulk'] for f in fits):.0f}, least E-BFMI"
                f" {min(f['e_bfmi'] for f in fits):.2f}, median"
                f" {np.median([f['seconds'] for f in fits]):.0f} s"
            )
        for arm, regrets in columns.items():
            if regrets:
                mean = Mean.of(np.array(list(regrets.values())))
                print(
                    f"  {arm} ({len(regrets)}): {mean.mean:+.4f} [{mean.interval[0]:+.4f},"
                    f" {mean.interval[1]:+.4f}], median {mean.median:.4f}"
                )
        for first in ("expected", "plug-in"):
            print(f"  {first} less:")
            verdicts = {}
            for arm, regrets in columns.items():
                common = sorted(set(regrets) & set(columns[first]))
                if arm == first or len(common) < 2:
                    continue
                ours_ = np.array([columns[first][s] for s in common])
                comparison = Comparison.of(arm, ours_, np.array([regrets[s] for s in common]))
                verdicts[arm] = verdict(comparison)
                low, high = comparison.difference.interval
                print(
                    f"    {arm} ({len(common)}): {comparison.difference.mean:+.4f} [{low:+.4f},"
                    f" {high:+.4f}]; lower {comparison.lower}, tied {comparison.tied}, higher"
                    f" {comparison.higher}; lower share [{comparison.lower_share[0]:.2f},"
                    f" {comparison.lower_share[1]:.2f}]; {verdicts[arm]}"
                )
            read = [arm for arm in GATED if arm in verdicts]
            beaten_by = [arm for arm in read if verdicts[arm] == "loses to"]
            gate = "met" if not beaten_by else f"not met, beaten by {beaten_by}"
            print(f"    gate over {read}: {gate}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser(
        "score", help="the three arms, scored as the bench scores its external arms"
    )
    plan.add_argument("--pilot", action="store_true", help="the design's seeds, not the scored")
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
    score(args.out, args.workers, pilot=args.pilot)


if __name__ == "__main__":
    main()
