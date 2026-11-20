"""Does a lift test count twice when the panel holds its readout? Posteriors, coverage and regret
only, no wall time.

The worlds are Track M v2's drawn worlds (causaldyn-bench), at the ladder's top rung: four go-dark
tests of paid shopping, from weeks 20, 55, 100 and 140, each dark for 4 weeks and read over those
and the 8 after, 48 of the history's 156 weeks. The bench's overlapping rung
(`causaldyn_bench.scorecard.ladder.overlap`) runs the same tests in the market whose history the
model reads, in a share of it:

    R    the tests ran in universes of their own, and the panel is the history as planned; every
         cell and every lift in the likelihood
    B-a  the whole market ran them (share 1): each lift's error holds the panel's noise over its
         readout; every cell and every lift
    B-b  as B-a, the readouts' cells out of the likelihood: the KPI missing there, the spend kept
    B-c  as B-a, no lift
    A-a  half the market ran them (share 1/2), read against the other half: the panel holds the
         tests' spend and half their response, and none of their noise; every cell and every lift
    A-b  as A-a, the readouts' cells out as in B-b
    A-c  as A-a, no lift
    R-b  as R, the readouts' cells out as in B-b: what taking those weeks out moves with no test
         run in them
    B-i  as B-a, each lift read from the same tests with the analysis' noise drawn apart, from
         another seed, so that no cell shares its noise: B-a without the double count

Each test is read as the bench reads it (`lift_rows`), which drops a test whose sales did not fall,
as PyMC-Marketing's likelihood needs; `--every` reads every test, a lift at or above 0 among them,
as a `LiftEstimate` does.

Every world is fitted by the one-geo Bayesian model with the world's curve (tanh), the export's
controls, a linear trend of 3 bends and 3 harmonics of the year, 4 chains from a seed spawned from
the world's, and planned from 400 of its draws at the quarter's budget, as the scorecard plans:
regret per euro against the oracle's plan. Each job writes `<out>/<variant>/<seed>.json`; a job
already written is skipped. `--summary` reads them back: each channel's return against the world's
on the whole history, the 80 % and 90 % intervals' coverage, and the regret, each with its Monte
Carlo standard error over the worlds, and each variant's paired against the first one named. A lift
counted twice shows in the pairing as a mean squared z, `(mean - truth) / sd`, above the first's
at the same SD, and as a lower coverage.

ADR 0037's measurement, worlds 900 to 929, is these three runs and their summaries:

    PYTHONPATH=<causaldyn-bench>/src:. uv run python scripts/study_lift_overlap.py \\
        --noise 0.01 --out out/n0.01 R B-a B-b B-c B-i R-b
    PYTHONPATH=<causaldyn-bench>/src:. uv run python scripts/study_lift_overlap.py \\
        --noise 0.1 --out out/n0.1 R B-a B-b B-c B-i
    PYTHONPATH=<causaldyn-bench>/src:. uv run python scripts/study_lift_overlap.py \\
        --noise 0.1 --every --out out/n0.1-every R B-a B-b B-i R-b A-a A-b
    PYTHONPATH=<causaldyn-bench>/src:. uv run python scripts/study_lift_overlap.py \\
        --out out/n0.1-every --summary B-i B-a B-b
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

os.environ.update(
    {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PJRT_NPROC": "1",
        "JAX_ENABLE_X64": "1",
        "JAX_PLATFORMS": "cpu",
    }
)

CHANNEL = "pla"
CHANNELS = ("pla", "meta", "tv")
RUNG = 4
READINGS = 400
QUANTILES = (0.05, 0.10, 0.50, 0.90, 0.95)
VARIANTS = ("R", "B-a", "B-b", "B-c", "A-a", "A-b", "A-c", "R-b", "B-i")
APART = 100_000
"""Added to a world's seed to draw its tests' analysis noise apart from the history's (B-i)."""
BOOTSTRAPS, BOOTSTRAP_SEED = 20_000, 20261007
"""The resamples of the worlds behind a paired ratio's interval, and their seed."""


def every_test(tests: dict[str, Any]) -> Any:
    """The bench's `lift_rows` without its drop of a test whose sales did not fall: every test, a
    lift at or above 0 among them, as a `LiftEstimate` reads one."""
    from types import SimpleNamespace

    from causaldyn_bench.lift_calibration import COOLDOWN, PRE, TEST

    columns: dict[str, list[Any]] = {
        "channel": [],
        "start": [],
        "x": [],
        "delta_y": [],
        "sigma": [],
    }
    after = slice(PRE, PRE + TEST + COOLDOWN)
    for name, channel_tests in tests.items():
        noise = math.sqrt(
            float(np.mean([np.var(t.difference[:PRE], ddof=1) for t in channel_tests]))
        )
        for test in channel_tests:
            readout = test.control.spend[test.control.history :]
            columns["channel"].append(name)
            columns["start"].append(test.control.history + PRE + 1)
            columns["x"].append(float(np.mean(readout[PRE : PRE + TEST])))
            columns["delta_y"].append(float(np.sum(test.difference[after])) / TEST)
            columns["sigma"].append(noise * math.sqrt(TEST + COOLDOWN) / TEST)
    return SimpleNamespace(
        channel=tuple(columns["channel"]),
        start=tuple(columns["start"]),
        x=np.array(columns["x"]),
        delta_y=np.array(columns["delta_y"]),
        sigma=np.array(columns["sigma"]),
    )


def job(
    seed: int, variant: str, noise: float, warmup: int, draws: int, every: bool
) -> dict[str, Any]:
    from causaldyn_bench.budget_regret import lift_rows
    from causaldyn_bench.lift_calibration import COOLDOWN, TEST
    from causaldyn_bench.mmm_decision import PLANNED
    from causaldyn_bench.scorecard import ladder
    from causaldyn_bench.scorecard.track_m2 import TRACK_M2
    from causaldyn_bench.scorecard.truth import worth

    from mixdyn import NATIONAL, Channel
    from mixdyn.calibration import LiftEstimate
    from mixdyn.decisions import Range, plan_posterior, readings
    from mixdyn.models.bayes_mmm import BayesMedia, BayesSpec, TanhCurve, fit_bayes
    from mixdyn.models.ridge_mmm import Control, FourierSeason, LinearTrend
    from scorecard.arms import panel as reading

    world = TRACK_M2.world("drawn", seed)
    base = TRACK_M2.observe(world, RUNG)
    truth = TRACK_M2.truth(world)
    share = 0.5 if variant.startswith("A") else 1.0
    read = every_test if every else lift_rows
    held = ladder.overlap(world.history, seed, RUNG, CHANNEL, share, noise)
    rows = read(held.tests)
    if variant == "B-i":
        apart = ladder.overlap(world.history, seed + APART, RUNG, CHANNEL, share, noise)
        if not np.array_equal(apart.planned, held.planned):
            msg = f"world {seed}'s tests drawn apart planned another spend"
            raise RuntimeError(msg)
        rows = read(apart.tests)
    history = world.history if variant.startswith("R") else held.history
    panel = reading.panel(dataclasses.replace(base, sales=history.sales, spend=history.spend))
    estimates, readout = [], np.zeros(history.week.size, dtype=bool)
    for i in range(len(rows.channel)):
        first = rows.start[i] - 1
        end = first + TEST + COOLDOWN
        tested = held.planned[:end].copy()
        tested[first : first + TEST] = 0.0
        estimates.append(
            LiftEstimate(
                spend=share * tested,
                planned=share * held.planned[:end],
                readout=TEST + COOLDOWN,
                lift=share * float(rows.delta_y[i]) * TEST,
                se=share * float(rows.sigma[i]) * TEST,
                share=share,
            )
        )
        readout[first:end] = True
    if variant.endswith("-b"):
        kpi = np.array(panel.kpi)
        kpi[0, readout] = np.nan
        panel = dataclasses.replace(panel, kpi=kpi)
    spec = BayesSpec(
        media=tuple(BayesMedia(name) for name in base.channels),
        controls=(Control("promotion", lower=0.0), Control("price")),
        trend=LinearTrend(3),
        season=FourierSeason(365.25 / 7.0, 3),
        saturation=TanhCurve(),
    )
    fit = fit_bayes(
        panel,
        NATIONAL,
        spec,
        seed=int(np.random.SeedSequence(seed).spawn(1)[0].generate_state(1)[0]),
        estimates=None if variant.endswith("-c") or not estimates else {CHANNEL: tuple(estimates)},
        warmup=warmup,
        draws=draws,
    )
    limits = {
        Channel(name): Range(float(low), float(high))
        for name, low, high in zip(base.channels, base.lower, base.upper, strict=True)
    }
    plan = plan_posterior(
        panel,
        NATIONAL,
        readings(fit, draws=READINGS),
        base.budget,
        PLANNED,
        limits,
        grid=2048,
    )
    weekly = np.asarray(plan.spend, dtype=float)
    if not truth.quarter.feasible(weekly, tolerance=1e-6):
        weekly = truth.quarter.project(weekly)
    out: dict[str, Any] = {
        "seed": seed,
        "variant": variant,
        "noise_share": noise,
        "fitted": int(fit.fitted.sum()),
        "regret": (truth.best.worth - worth(truth.cells, weekly)) / truth.quarter.budget,
        "divergences": int(fit.divergent.sum()),
        "max_rhat": max(float(np.max(d.rhat)) for d in fit.diagnostics.values()),
        "truth": {"roi": (history.media.sum(axis=0) / history.spend.sum(axis=0)).tolist()},
        "posterior": {},
    }
    roi = fit.draws["roi"].reshape(-1, len(base.channels))
    out["posterior"]["roi"] = {
        "mean": roi.mean(axis=0).tolist(),
        "sd": roi.std(axis=0, ddof=1).tolist(),
        "quantiles": np.quantile(roi, QUANTILES, axis=0).tolist(),
    }
    return out


def run(seed: int, variant: str, args: argparse.Namespace) -> str:
    path = Path(args.out) / variant / f"{seed}.json"
    if path.exists():
        return f"{seed} {variant}: kept"
    record = job(seed, variant, args.noise, args.warmup, args.draws, args.every)
    path.parent.mkdir(parents=True, exist_ok=True)
    aside = path.with_suffix(f".{os.getpid()}.tmp")
    aside.write_text(json.dumps(record))
    aside.replace(path)
    return f"{seed} {variant}: written"


def mean_se(values: np.ndarray, digits: int = 3) -> str:
    error = values.std(ddof=1) / math.sqrt(values.size)
    return f"{values.mean():+.{digits}f} ± {error:.{digits}f}"


def hits(records: list[dict[str, Any]], c: int, low: int, high: int) -> np.ndarray:
    """1 where the interval between the quantiles `low` and `high` holds channel `c`'s return."""
    return np.array(
        [
            r["posterior"]["roi"]["quantiles"][low][c]
            <= r["truth"]["roi"][c]
            <= r["posterior"]["roi"]["quantiles"][high][c]
            for r in records
        ],
        dtype=float,
    )


def covered(records: list[dict[str, Any]], c: int, low: int, high: int) -> str:
    p = float(hits(records, c, low, high).mean())
    return f"{p:.2f} ± {math.sqrt(max(p * (1.0 - p), 1e-12) / len(records)):.2f}"


def paid_shopping(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Paid shopping's posterior mean and SD of its return, and the world's, world by world."""
    mean = np.array([r["posterior"]["roi"]["mean"][0] for r in records])
    sd = np.array([r["posterior"]["roi"]["sd"][0] for r in records])
    truth = np.array([r["truth"]["roi"][0] for r in records])
    return mean, sd, truth


def summary(out: Path, variants: list[str]) -> None:
    """Each variant's posterior of each channel's return against the world's, and its regret; then
    each paired against the first, on paid shopping's return: the SD as a ratio (geometric mean,
    two standard errors), the mean squared z as a ratio (a percentile bootstrap over the worlds,
    95 %), the 80 % coverage's difference, the mean's shift in the first's SDs, the absolute
    error's difference over the world's return, and the regret's difference."""
    records = {
        v: {int(p.stem): json.loads(p.read_text()) for p in (out / v).glob("*.json")}
        for v in variants
    }
    seeds = sorted(set.intersection(*(set(r) for r in records.values())))
    print(f"{len(seeds)} worlds in every variant, {seeds[0]} to {seeds[-1]}")
    for c, channel in enumerate(CHANNELS):
        print(f"\n{channel}: (mean - truth) / truth, sd / truth, 80 % and 90 % coverage")
        for v in variants:
            rs = [records[v][s] for s in seeds]
            truth = np.array([r["truth"]["roi"][c] for r in rs])
            mean = np.array([r["posterior"]["roi"]["mean"][c] for r in rs])
            sd = np.array([r["posterior"]["roi"]["sd"][c] for r in rs])
            print(
                f"  {v:4} {mean_se((mean - truth) / truth)}  {mean_se(sd / truth)}"
                f"  {covered(rs, c, 1, 3)}  {covered(rs, c, 0, 4)}"
            )
    print("\nregret per euro; divergences; largest R-hat")
    for v in variants:
        rs = [records[v][s] for s in seeds]
        print(
            f"  {v:4} {mean_se(np.array([r['regret'] for r in rs]), 4)}"
            f"  {sum(r['divergences'] for r in rs)}  {max(r['max_rhat'] for r in rs):.3f}"
        )
    first = variants[0]
    print(
        f"\npaired against {first}, {CHANNEL}'s return: sd ratio; mean squared z ratio; 80 % "
        "coverage; mean's shift; absolute error; regret"
    )
    b = [records[first][s] for s in seeds]
    mean_b, sd_b, truth_b = paid_shopping(b)
    z_b = (mean_b - truth_b) / sd_b
    picks = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(seeds), (BOOTSTRAPS, len(seeds)))
    for v in variants[1:]:
        a = [records[v][s] for s in seeds]
        mean_a, sd_a, truth_a = paid_shopping(a)
        z_a = (mean_a - truth_a) / sd_a
        ratio = np.log(sd_a / sd_b)
        m, s = ratio.mean(), ratio.std(ddof=1) / math.sqrt(ratio.size)
        squared = np.mean(z_a[picks] ** 2, axis=1) / np.mean(z_b[picks] ** 2, axis=1)
        low, high = np.quantile(squared, (0.025, 0.975))
        error = (np.abs(mean_a - truth_a) - np.abs(mean_b - truth_b)) / truth_b
        regret = np.array([x["regret"] - y["regret"] for x, y in zip(a, b, strict=True)])
        print(
            f"  {v:4} {math.exp(m):.3f} [{math.exp(m - 2 * s):.3f}, {math.exp(m + 2 * s):.3f}]"
            f"  {np.mean(z_a**2) / np.mean(z_b**2):.2f} [{low:.2f}, {high:.2f}]"
            f"  {mean_se(hits(a, 0, 1, 3) - hits(b, 0, 1, 3))}  {mean_se((mean_a - mean_b) / sd_b)}"
            f"  {mean_se(error)}  {mean_se(regret, 4)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variants", nargs="*", default=list(VARIANTS))
    parser.add_argument("--first", type=int, default=900)
    parser.add_argument("--worlds", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--out", required=True)
    parser.add_argument("--every", action="store_true", help="read every test, as mixdyn does")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    unknown = sorted(set(args.variants) - set(VARIANTS))
    if unknown:
        parser.error(f"no variants {unknown}; the variants are {VARIANTS}")
    if args.summary:
        summary(Path(args.out), args.variants)
        return
    jobs = [(s, v) for s in range(args.first, args.first + args.worlds) for v in args.variants]
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn")) as pool:
        for done in as_completed([pool.submit(run, s, v, args) for s, v in jobs]):
            print(done.result(), flush=True)


if __name__ == "__main__":
    main()
