"""mixdyn's ridge engine planning Track M v2's quarter: the pilot that chose how the geo
tests calibrate it, and the engine's scored run.

Pilot seeds only: the design's 40 drawn worlds from seed 900 (``PILOTS[0]``), which no reading rule
of the scored run reads. Each arm's fit hands its channels to the bench's planner (``planned``), and
the plan is scored as the bench scores every arm's: regret per euro against the world's oracle; a
fit or plan that raises plays the status quo, its error recorded; a plan off the budget or the box
by more than ``OUT_OF_PLAN`` is moved into the box, and the move recorded.

Every arm fits the world's raw panel over the same space and folds: each channel's geometric decay
in [0, 0.9], Hill shape in [0.5, 3] and inflexion in [0.3, 1], the penalty in [1e-4, 1e2], the
promotion's coefficient at or above 0, a trend with 3 knots, a yearly season of 3 harmonics, 4 folds
of 13 weeks, the Sobol sample scrambled by the world's seed.

* ``v0``: ``search_ridge`` without the tests.
* ``v1``: the tests in each fold's fit alone, the score the KPI's. Each tested channel's coefficient
  is anchored at its tests' least-squares estimate, weighted by the KPI's noise variance over the
  tests', the KPI's read once off the whole window.
* ``v2``: ``v1``'s folds, the score the joint least squares of the folds' errors and the tests'
  errors under the whole window's calibrated fit, each source over a noise variance fixed for every
  candidate: the KPI's ``v0``'s score in the KPI's units, the tests' their gaps' mean square over
  the ``PRE`` weeks before they go dark.
* ``library``: ``search_ridge(tests=...)``, ``v2`` as the library carries it: each fold's KPI noise
  read off its own training periods, the tests' off every readout period before their groups' spend
  parts.

``v1`` and ``v2`` are written here on ``mixdyn``'s internals, as the pilot ran them. The rule,
stated before the pilot ran: ``v2`` only if its paired 95 % interval against ``v1`` lies below 0.

The other arms are read, not refitted: CHC's, PyMC-Marketing's, the observational model's and the
status quo's regrets from the bench's pilot (``results/track_m2_budgets_pilot.json``), and Robyn's
from its genre pilot (``results/track_m2_external_pilot_genre.json``) on the 10 worlds it planned.
Each world first reproduces the bench pilot's status-quo regret to ``TIE`` per euro, so both sides
of a comparison are scored against one oracle.

**The scored run** (``score``) plans with three arms: the library's ``ridge`` (``v0``) and
``calibrated`` (``library``), and ``anchored`` (``v1``), so that the pilot's ordering of the two
calibrations is read on a second stream of worlds. It scores them as the bench scores its
external arms (``causaldyn_bench.external_arms``): each plan by ``score_arm``, each world first
reproducing the budgets run's oracle worth and status-quo regret. ``score --pilot`` runs them on
the design's pilot seeds, both environments, beside the bench's pilots; ``score`` on the budgets
run's scored worlds, the 200 drawn from seed 10 000 and the 100 reference from 20 000, beside its
CHC, PyMC-Marketing, observational model and status quo (``results/track_m2_budgets.json``) and
the bench's Meridian and Robyn (``results/track_m2_external.json``), Robyn on the 100 drawn worlds
it planned. Each world's score
is appended to ``<out>.jsonl`` as it lands, and a run restarted with the same ``--out`` plans only
the worlds not there.

The analysis is the bench's: each arm's mean regret per euro with its 95 % Student-t interval, and
its median; paired differences, ``calibrated``'s regret less each other arm's over the worlds both
planned, with the worlds where it is the lower, tied within ``TIE`` or higher, a Clopper-Pearson
interval on the lower share, and the bench's verdict by where the interval lies against 0.

**The scored run's pre-registration**, written 10-02 from ``score --pilot``'s record
(``results/track_m2_ridge_score_pilot.json``: mixdyn ``39a0a72``, the bench ``e747207``, CHC
``0ab2cd7``) before any scored world ran:

* The run is this file's ``score`` at its defaults, on the same mixdyn and CHC, the bench's code
  ``e747207``'s; a world that raises is a run that stops, not a world dropped.
* The gate, the engine's gate read on Track M v2: on each environment no arm of ``GATED`` beats
  ``calibrated`` by the bench's verdict; Robyn is read on the 100 drawn worlds it planned.
* The predictions, each the pilot's verdict carried to the scored worlds. The pilot's drawn worlds
  reproduce the first pilot's ``v0``, ``v1`` and ``library`` to the last bit.

  - Drawn: ``calibrated`` beats ``ridge``, ``anchored`` (the pilot's ordering of the two
    calibrations, -0.064 [-0.106, -0.022], read on a second stream), the observational model, the
    status quo and Meridian (-0.056 [-0.086, -0.025]); ties CHC and PyMC-Marketing; beats Robyn, the
    least certain of these (-0.086 [-0.183, +0.011] on its 10 pilot worlds). The gate is met.
  - Reference: ``calibrated`` beats ``ridge``, the observational model and the status quo; ties
    ``anchored`` (+0.028 [-0.079, +0.135], higher on 11 of 20 worlds), CHC (+0.046
    [-0.014, +0.105]) and Meridian (+0.006 [-0.076, +0.088]). Against PyMC-Marketing, +0.061
    [-0.014, +0.135] with higher on 9 of 20, a loss is more likely than not: about three in four
    over 100 worlds, if the pilot's mean and spread hold. So the gate is predicted not met there.
    The reference's quarters are one design with 20 noise draws, and a plan there is close to a
    corner: a world's regret is near 0 or near 0.6, so a few worlds where the plan picks the wrong
    corner decide each comparison.

    PYTHONPATH=<bench>/src JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \\
        uv run --project <mixdyn checkout> --no-sync python evidence/bench/track_m2_ridge.py \\
        run --out <json> [--workers 3]          # the pilot of the four arms
        score [--pilot] --out <json> [--workers 3]  # the three arms, scored
    PYTHONPATH=<bench>/src uv run --project <mixdyn checkout> --no-sync \\
        python evidence/bench/track_m2_ridge.py report <json>
"""

from __future__ import annotations

import argparse
import functools
import importlib.metadata
import json
import math
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import causaldyn_bench
import chc
import jax.numpy as jnp
import numpy as np
from causaldyn_bench.budget_regret import (
    ENVIRONMENTS,
    OUT_OF_PLAN,
    PILOTS,
    TIE,
    Comparison,
    Mean,
    experiments,
    planned,
)
from causaldyn_bench.external_arms import score_arm, verdict
from causaldyn_bench.mmm_decision import Quarter, oracle, regret
from scipy import optimize
from scipy.stats import qmc
from scipy.stats import t as student

import mixdyn
from mixdyn import Cadence, Channel, Geo, MediaPanel, Revenue
from mixdyn.models.ridge_mmm import Control, FourierSeason, LinearTrend, _bounds, _design, _read
from mixdyn.models.ridge_search import (
    Folds,
    GeometricSpace,
    Interval,
    MediaSpace,
    RidgeSpace,
    search_ridge,
)
from mixdyn.transforms import carry_over, channel, saturate

NATIONAL = Geo("national")
PRE = 4
"""The weeks each of the bench's geo tests reads before its treated group goes dark."""
KNOTS = 3
FOLDS = Folds(4, 13)
ARMS = ("v0", "v1", "v2", "library")
READ = ("CHC", "PyMC-Marketing", "observational", "status quo")
SCORED = ("ridge", "anchored", "calibrated")
EXTERNAL = ("Meridian", "Robyn")
GATED = ("PyMC-Marketing", *EXTERNAL)
"""The external arms the gate reads: none may beat ``calibrated`` on an environment."""
BUDGETS = {True: "track_m2_budgets_pilot.json", False: "track_m2_budgets.json"}
EXTERNALS = {True: "track_m2_external_pilot_genre.json", False: "track_m2_external.json"}
BENCH = Path(causaldyn_bench.__file__).resolve().parents[2] / "results"


def panel_of(world) -> MediaPanel:
    weeks = world.sales.size
    return MediaPanel(
        dates=np.datetime64("2023-01-02") + np.timedelta64(7, "D") * np.arange(weeks),
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=tuple(Channel(c) for c in world.channels),
        controls=("promotion", "price"),
        kpi=world.sales[None, :],
        kpi_unit=Revenue(),
        spend=world.spend[None, :, :],
        currency="EUR",
        control_values=np.column_stack(
            [world.observed_promotion.astype(float), world.observed_price]
        )[None, :, :],
        filled=np.zeros((1, weeks), dtype=bool),
    )


def space_of(world) -> RidgeSpace:
    return RidgeSpace(
        media=tuple(
            MediaSpace(
                c, GeometricSpace(Interval(0.0, 0.9)), Interval(0.5, 3.0), Interval(0.3, 1.0)
            )
            for c in world.channels
        ),
        controls=(Control("promotion", lower=0.0), Control("price")),
        penalty=Interval(1e-4, 1e2),
        trend=LinearTrend(KNOTS),
        season=FourierSeason(365.25 / 7.0, 3),
    )


def anchored(x, y, lam, lower, upper, center, precision):
    """Glmnet's ridge plus 1/2 sum precision_j (b_j - center_j)^2, bounded: BVLS on the augmented
    system, in the standardised coordinates; a column that does not vary gets 0."""
    n, p = x.shape
    mx, my = x.mean(axis=0), y.mean()
    s = np.sqrt(np.mean((x - mx) ** 2, axis=0))
    live = np.flatnonzero(s > 0)
    k = live.size
    sl = s[live]
    z = (x[:, live] - mx[live]) / sl
    yc = y - my
    q = precision[live] / sl**2
    system = np.vstack([z / math.sqrt(n), math.sqrt(lam) * np.eye(k), np.diag(np.sqrt(q))])
    rhs = np.concatenate([yc / math.sqrt(n), np.zeros(k), np.sqrt(q) * center[live] * sl])
    lo, hi = lower[live] * sl, upper[live] * sl
    c = np.linalg.lstsq(system, rhs, rcond=None)[0]
    if (c < lo).any() or (c > hi).any():
        c = optimize.lsq_linear(system, rhs, bounds=(lo, hi), method="bvls", tol=1e-14).x
    b = np.zeros(p)
    b[live] = c / sl
    return float(my - mx @ b), b


def ridge_df(x, lam):
    """The free ridge's effective degrees of freedom, sum d^2 / (d^2 + n lam), on the live
    columns."""
    s = x.std(axis=0)
    live = s > 0
    z = (x[:, live] - x[:, live].mean(axis=0)) / s[live]
    d = np.linalg.svd(z, compute_uv=False)
    return float(np.sum(d**2 / (d**2 + x.shape[0] * lam)))


def gaps(spec, halves, tests):
    """Per channel: each test period's saturated gap (treated less control) and observed gap."""
    out = {}
    for m in spec.media:
        delta, observed = [], []
        for test in tests[m.channel]:
            k = halves[m.channel]
            arms = []
            for arm in (test.treated, test.control):
                adstocked = carry_over(arm.spend / arm.share, m.adstock)
                arms.append(saturate(adstocked, m.hill, k)[arm.spend.size - arm.outcome.size :])
            delta.append(arms[0] - arms[1])
            observed.append(test.difference)
        out[m.channel] = (np.concatenate(delta), np.concatenate(observed))
    return out


def pre_noise(tests) -> float:
    """The tests' gap variance over the weeks before they went dark, where the arms spend alike."""
    pre = np.concatenate([t.difference[:PRE] for ts in tests.values() for t in ts])
    return float(np.mean(pre**2))


def calibrated_objective(data, space, masks, span, tests, mode, sigma0=None, tau0=None):
    count = [0]

    def score(point):
        count[0] += 1
        spec = space.spec(point)
        design, halves = _design(data, spec)
        lower, upper = _bounds(spec)
        fitted = data.fitted
        y = data.y
        g = gaps(spec, halves, tests)
        p = design.shape[1]
        names = spec.terms()
        mu = np.zeros(p)
        a = np.zeros(p)
        residual_test = 0.0
        n_test = 0
        for m in spec.media:
            j = names.index(m.channel)
            d, obs = g[m.channel]
            a[j] = float(d @ d)
            mu[j] = float(obs @ d) / a[j] if a[j] > 0 else 0.0
            residual_test += float(np.sum((obs - mu[j] * d) ** 2))
            n_test += obs.size
        tau2 = residual_test / (n_test - len(spec.media))
        b0, b = anchored(design[fitted], y[fitted], spec.penalty, lower, upper, mu, np.zeros(p))
        r = y[fitted] - b0 - design[fitted] @ b
        # the history's noise variance with the ridge's degrees of freedom taken off, as the tests'
        # is with their coefficients'
        sigma2 = float(r @ r) / (r.size - 1.0 - ridge_df(design[fitted], spec.penalty))
        w = sigma2 / tau2
        squares, total = 0.0, 0
        for train, held in masks:
            rho = w * a / train.sum()
            c0, c = anchored(design[train], y[train], spec.penalty, lower, upper, mu, rho)
            res = y[held] - c0 - design[held] @ c
            squares += float(res @ res)
            total += res.size
        if mode == "v1":
            return math.sqrt(squares / total) / span
        # v2: the joint predictive score, the history's held-out errors and the tests' gaps, each
        # in its own fixed noise unit
        rho = w * a / fitted.sum()
        c0, c = anchored(design[fitted], y[fitted], spec.penalty, lower, upper, mu, rho)
        test_sse = 0.0
        for m in spec.media:
            j = names.index(m.channel)
            d, obs = g[m.channel]
            test_sse += float(np.sum((obs - c[j] * d) ** 2))
        return squares / sigma0 + test_sse / tau0

    return score, count


def calibrated_fit(data, spec, tests):
    design, halves = _design(data, spec)
    lower, upper = _bounds(spec)
    y, fitted = data.y, data.fitted
    g = gaps(spec, halves, tests)
    names = spec.terms()
    p = design.shape[1]
    mu, a = np.zeros(p), np.zeros(p)
    residual_test, n_test = 0.0, 0
    for m in spec.media:
        j = names.index(m.channel)
        d, obs = g[m.channel]
        a[j] = float(d @ d)
        mu[j] = float(obs @ d) / a[j] if a[j] > 0 else 0.0
        residual_test += float(np.sum((obs - mu[j] * d) ** 2))
        n_test += obs.size
    tau2 = residual_test / (n_test - len(spec.media))
    b0, b = anchored(design[fitted], y[fitted], spec.penalty, lower, upper, mu, np.zeros(p))
    r = y[fitted] - b0 - design[fitted] @ b
    w = (float(r @ r) / (r.size - 1.0 - ridge_df(design[fitted], spec.penalty))) / tau2
    _, c = anchored(design[fitted], y[fitted], spec.penalty, lower, upper, mu, w * a / fitted.sum())
    return dict(zip(names, c.tolist(), strict=True)), halves, w


def search_calibrated(
    data, space, folds, tests, seed, mode, starts=256, polish=8, sigma0=None, tau0=None
):
    masks = folds.masks(data.fitted)
    observed = data.y[data.fitted]
    span = float(observed.max() - observed.min())
    score, count = calibrated_objective(data, space, masks, span, tests, mode, sigma0, tau0)
    d = space.dimension
    points = qmc.Sobol(d, scramble=True, seed=seed).random(starts)
    scores = np.array([score(p) for p in points])
    optima = []
    for i in np.argsort(scores, kind="stable")[:polish]:
        r = optimize.minimize(
            lambda u: math.log(score(u)), points[i], method="L-BFGS-B", bounds=[(0.0, 1.0)] * d
        )
        optima.append((math.exp(r.fun), r.x))
    optima.sort(key=lambda o: o[0])
    return space.spec(optima[0][1]), optima[0][0], count[0]


def scored(world, quarter, best, make) -> dict[str, Any]:
    """Regret per euro of the plan ``make`` returns, with the error that stood a fit or plan in for
    the status quo and the move that brought a plan into the box, each as the bench records it."""
    try:
        plan = make()
    except (RuntimeError, ValueError) as error:
        return {
            "regret": regret(world, quarter, quarter.status_quo, best) / quarter.budget,
            "failure": str(error)[:200],
            "moved": None,
        }
    moved = None
    if not quarter.feasible(plan, tolerance=OUT_OF_PLAN):
        projected = quarter.project(plan)
        moved = float(np.abs(projected - plan).sum() / quarter.budget)
        plan = projected
    return {
        "regret": regret(world, quarter, plan, best) / quarter.budget,
        "failure": None,
        "moved": moved,
    }


@functools.cache
def committed(name: str) -> dict[str, dict[int, dict[str, Any]]]:
    """A committed bench result's worlds, by environment and seed."""
    environments = json.loads((BENCH / name).read_text())["environments"]
    return {e["environment"]["name"]: {w["seed"]: w for w in e["worlds"]} for e in environments}


def run(seed: int) -> dict[str, Any]:
    world = PILOTS[0].generator(seed).simulate(seed)
    tests = experiments(world, seed)
    quarter = Quarter.after(world)
    best = oracle(world, quarter)
    status_quo = regret(world, quarter, quarter.status_quo, best) / quarter.budget
    theirs = committed("track_m2_budgets_pilot.json")["drawn"][seed]["regret"]["status quo"]
    if abs(status_quo - theirs) > TIE:
        msg = f"world {seed}'s status quo regrets {status_quo} per euro here, {theirs} in the pilot"
        raise RuntimeError(msg)
    panel, space = panel_of(world), space_of(world)
    out: dict[str, Any] = {"seed": seed, "status quo": status_quo}
    base = search_ridge(panel, NATIONAL, space, FOLDS, seed=seed)
    out["v0"] = scored(
        world, quarter, best, lambda: planned(list(base.fit.channels().values()), quarter)
    )
    out["v0"] |= {"penalty": base.fit.spec.penalty, "evaluations": base.evaluations}
    data = _read(panel, NATIONAL, space.spec(np.full(space.dimension, 0.5)), None)
    for mode in ("v1", "v2"):
        kwargs = {}
        if mode == "v2":
            kwargs = {"sigma0": (base.score * float(np.ptp(data.y))) ** 2, "tau0": pre_noise(tests)}
        spec, _, evaluations = search_calibrated(data, space, FOLDS, tests, seed, mode, **kwargs)
        coefficients, halves, w = calibrated_fit(data, spec, tests)
        channels = [
            channel(m.adstock, m.hill, halves[m.channel], coefficients[m.channel], data.dates.size)
            for m in spec.media
        ]
        out[mode] = scored(
            world, quarter, best, lambda channels=channels: planned(channels, quarter)
        )
        out[mode] |= {"penalty": spec.penalty, "evaluations": evaluations, "weight": w}
    search = search_ridge(panel, NATIONAL, space, FOLDS, seed=seed, tests=tests)
    fit = search.fit
    out["library"] = scored(
        world, quarter, best, lambda: planned(list(fit.channels().values()), quarter)
    )
    out["library"] |= {
        "penalty": fit.spec.penalty,
        "evaluations": search.evaluations,
        "weight": fit.calibration.weight,
    }
    print(json.dumps({"seed": seed} | {a: round(out[a]["regret"], 4) for a in ARMS}), flush=True)
    return out


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
    panel, space = panel_of(world), space_of(world)
    out: dict[str, Any] = {"environment": name, "seed": seed}
    for arm in SCORED:
        fitted: dict[str, Any] = {}
        try:
            if arm == "anchored":
                data = _read(panel, NATIONAL, space.spec(np.full(space.dimension, 0.5)), None)
                spec, _, evaluations = search_calibrated(data, space, FOLDS, tests, seed, "v1")
                coefficients, halves, w = calibrated_fit(data, spec, tests)
                length = data.dates.size
                channels = [
                    channel(m.adstock, m.hill, halves[m.channel], coefficients[m.channel], length)
                    for m in spec.media
                ]
                fitted = {"penalty": spec.penalty, "evaluations": evaluations, "weight": w}
            else:
                given = tests if arm == "calibrated" else None
                search = search_ridge(panel, NATIONAL, space, FOLDS, seed=seed, tests=given)
                channels = list(search.fit.channels().values())
                calibration = search.fit.calibration
                fitted = {
                    "penalty": search.fit.spec.penalty,
                    "evaluations": search.evaluations,
                    "weight": calibration.weight if calibration else None,
                }
            record: dict[str, Any] = {"error": None, "weekly": planned(channels, quarter)}
        except (RuntimeError, ValueError) as error:
            record = {"error": str(error)[:200]}
        score = score_arm(world, quarter, best, record)
        out[arm] = {
            "regret": score.regret,
            "failure": score.failure,
            "moved": score.moved,
            "bounds": score.bounds,
        } | fitted
    print(
        json.dumps({"environment": name, "seed": seed} | {a: out[a]["regret"] for a in SCORED}),
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
                "environments": [
                    {"name": e.name, "first": e.first, "worlds": e.worlds}
                    for e in (PILOTS if pilot else ENVIRONMENTS)
                ],
                "worlds": sorted(done.values(), key=lambda w: (w["environment"], w["seed"])),
            },
            indent=1,
        )
    )


def report_scored(record: dict[str, Any]) -> None:
    pilot = record["pilot"]
    budgets = committed(BUDGETS[pilot])
    external = BENCH / EXTERNALS[pilot]
    print(f"mixdyn {record['mixdyn']['commit'][:7]}; {'pilot' if pilot else 'scored'} worlds")
    if not external.exists():
        print(f"{external.name} is not committed yet: Meridian and Robyn are not read")
    for environment in record["environments"]:
        name = environment["name"]
        ours = {w["seed"]: w for w in record["worlds"] if w["environment"] == name}
        columns = {arm: {s: w[arm]["regret"] for s, w in ours.items()} for arm in SCORED}
        columns |= {arm: {s: budgets[name][s]["regret"][arm] for s in ours} for arm in READ}
        if external.exists():
            planned_by = committed(EXTERNALS[pilot]).get(name, {})
            for arm in EXTERNAL:
                columns[arm] = {
                    s: w["arms"][arm]["regret"]
                    for s, w in planned_by.items()
                    if s in ours and arm in w["arms"]
                }
        print(f"\n{name}: {len(ours)} worlds of {environment['worlds']}")
        for arm in SCORED:
            failed = [s for s, w in ours.items() if w[arm]["failure"]]
            moved = [s for s, w in ours.items() if w[arm]["moved"]]
            print(f"  {arm} failed on {failed}, moved on {moved}")
        for arm, regrets in columns.items():
            if regrets:
                mean = Mean.of(np.array(list(regrets.values())))
                print(
                    f"  {arm} ({len(regrets)}): {mean.mean:+.4f} [{mean.interval[0]:+.4f},"
                    f" {mean.interval[1]:+.4f}], median {mean.median:.4f}"
                )
        print("  calibrated less:")
        verdicts = {}
        for arm, regrets in columns.items():
            common = sorted(set(regrets) & set(columns["calibrated"]))
            if arm == "calibrated" or len(common) < 2:
                continue
            first = np.array([columns["calibrated"][s] for s in common])
            comparison = Comparison.of(arm, first, np.array([regrets[s] for s in common]))
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
        print(
            f"  gate over {read}: {'met' if not beaten_by else f'not met, beaten by {beaten_by}'}"
        )


def provenance() -> dict[str, Any]:
    distribution = importlib.metadata.distribution("causal-hybrid-control")
    return {
        "mixdyn": commit(mixdyn),
        "bench": commit(causaldyn_bench),
        "chc": {
            "module": chc.__file__,
            "installed": json.loads(distribution.read_text("direct_url.json") or "null"),
        },
    }


def commit(module) -> dict[str, Any]:
    root = Path(module.__file__).resolve().parents[2]

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    return {
        "path": str(root),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def interval(values) -> str:
    x = np.asarray(values, dtype=float)
    m = x.mean()
    h = student.ppf(0.975, x.size - 1) * x.std(ddof=1) / math.sqrt(x.size)
    return f"{m:+.4f} [{m - h:+.4f}, {m + h:+.4f}]"


def paired(ours, theirs) -> str:
    diff = np.asarray(ours, dtype=float) - np.asarray(theirs, dtype=float)
    lower, higher = int(np.sum(diff < -TIE)), int(np.sum(diff > TIE))
    return f"{interval(diff)}  lower {lower}, tied {diff.size - lower - higher}, higher {higher}"


def report(path: Path) -> None:
    record = json.loads(path.read_text())
    worlds = {w["seed"]: w for w in record["worlds"]}
    seeds = sorted(worlds)
    pilot = committed("track_m2_budgets_pilot.json")["drawn"]
    robyn = {
        s: w["arms"]["Robyn"]["regret"]
        for s, w in committed("track_m2_external_pilot_genre.json")["drawn"].items()
        if s in worlds and "Robyn" in w["arms"]
    }
    installed = record["chc"]["installed"]
    print(
        f"{len(seeds)} worlds; mixdyn {record['mixdyn']['commit'][:7]}, chc from"
        f" {record['chc']['module']} ({installed and installed['vcs_info']['commit_id'][:7]}"
        " installed)"
    )
    for arm in ARMS:
        regrets = [worlds[s][arm]["regret"] for s in seeds]
        failed = [s for s in seeds if worlds[s][arm]["failure"]]
        moved = [s for s in seeds if worlds[s][arm]["moved"]]
        print(
            f"{arm}: mean {interval(regrets)}, median {np.median(regrets):.4f};"
            f" failed {failed}, moved {moved}"
        )
    for arm in READ:
        print(f"{arm}: mean {interval([pilot[s]['regret'][arm] for s in seeds])}")
    print(f"Robyn ({len(robyn)}): mean {interval(list(robyn.values()))}")
    print("paired, the first arm's regret less the second's:")
    for first, second in (("v2", "v1"), ("v1", "v0"), ("v2", "v0"), ("library", "v2")):
        print(
            f"  {first} - {second}: "
            + paired(
                [worlds[s][first]["regret"] for s in seeds],
                [worlds[s][second]["regret"] for s in seeds],
            )
        )
    for arm in ARMS:
        for other in READ:
            ours = [worlds[s][arm]["regret"] for s in seeds]
            print(f"  {arm} - {other}: " + paired(ours, [pilot[s]["regret"][other] for s in seeds]))
        common = sorted(robyn)
        ours = [worlds[s][arm]["regret"] for s in common]
        print(f"  {arm} - Robyn ({len(common)}): " + paired(ours, [robyn[s] for s in common]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    go = sub.add_parser("run", help="the pilot of the four arms")
    go.add_argument("--out", type=Path, required=True)
    go.add_argument("--workers", type=int, default=3)
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
        record = json.loads(args.record.read_text())
        if "pilot" in record:
            report_scored(record)
        else:
            report(args.record)
        return
    if jnp.zeros(()).dtype != jnp.float64:
        raise SystemExit("JAX_ENABLE_X64=1 is required: the bench scored at float64")
    if args.command == "score":
        score(args.out, args.workers, pilot=args.pilot)
        return
    environment = PILOTS[0]
    source = provenance()
    with ProcessPoolExecutor(args.workers) as pool:
        worlds = list(pool.map(run, environment.seeds))
    args.out.write_text(
        json.dumps(
            {
                "environment": {
                    "name": environment.name,
                    "first": environment.first,
                    "worlds": environment.worlds,
                }
            }
            | source
            | {"worlds": worlds},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
