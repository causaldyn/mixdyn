"""Does a test read period by period recover the decay a summed test cannot? Posteriors and coverage
only, no wall time.

Each world is 104 weeks of one channel, its spend a first-order autoregression about 100 a week at
0.9 with shocks of SD 10, carried over by the normalised geometric kernel at `--decays` and
saturated by a tanh at 1.2 times its mean positive spend, its return on the history's spend 2,
under a KPI of 1000 plus normal noise of SD 50: spend that moves slowly, so the history reads the
decay weakly. One go-dark test runs in half the market after the history: the group's planned
spend of 50 a week over 30 weeks, dark over the first 4 of the last 12, the readout. The group's gap
is the world's plus a geo noise that carries over, a first-order autoregression at `--phis` of SD
8 a week, running on from the 104 weeks before the test that `PlaceboNoise` reads as its residuals.
The same test is read two ways, each with the error `PlaceboNoise` reads from those residuals:

    sum     one `LiftEstimate`, the gaps summed over the readout at `PlaceboNoise.se`
    series  one `LiftSeries`, the gaps at `PlaceboNoise.covariance`

The covariance sums to the summed error squared, so the two read the same total at the same error;
only the shape differs. Every world is fitted by the one-geo Bayesian model with the world's curve,
4 chains of 500 draws after 500 of warm-up started by Pathfinder, the sum and the series on the
same world and seed. Each job writes `<out>/<decay>-<phi>/<seed>.json`; a job already written is
skipped. `--summary` reads them back: for each read, the decay's posterior mean less the world's
(bias), the root mean square of that, the central 90 % interval's coverage of the world's decay and
its mean width, each with its Monte Carlo standard error over the worlds, and the divergences.

ADR 0046's measurement, worlds 0 to 99 of each cell:

    uv run python scripts/study_lift_series.py --worlds 100 --out out/lift-series
    uv run python scripts/study_lift_series.py --out out/lift-series --summary
"""

from __future__ import annotations

import argparse
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

WEEKS, SPAN, READOUT, DARK, SHARE = 104, 30, 12, 4, 0.5
ROI, HALF, LEVEL = 2.0, 1.2, 0.9
KPI_SD, GEO_SD, PERSIST, JITTER = 50.0, 8.0, 0.9, 10.0


def world(decay: float, phi: float, seed: int) -> dict[str, Any]:
    """The world's panel and its test read both ways; see the module."""
    from chc import response

    from mixdyn import NATIONAL, Cadence, Channel, MediaPanel, Revenue
    from mixdyn.calibration import LiftSeries
    from mixdyn.experiments import PlaceboNoise
    from mixdyn.models.bayes_mmm import _TANH_HALF

    rng = np.random.default_rng([20261013, seed])
    spend = np.empty(WEEKS)
    spend[0] = 100.0
    for t in range(1, WEEKS):
        spend[t] = 100.0 + PERSIST * (spend[t - 1] - 100.0) + rng.normal(0.0, JITTER)
    spend = np.maximum(spend, 0.0)
    kernel = response.GeometricAdstock(decay, length=WEEKS, normalized=True)
    curve = response.Tanh(HALF * float(spend[spend > 0.0].mean()) / _TANH_HALF)
    saturated = np.asarray(curve(kernel(spend)))
    channel = response.Channel(kernel, curve, ROI * spend.sum() / saturated.sum())
    kpi = 1000.0 + np.asarray(channel(spend)) + rng.normal(0.0, KPI_SD, WEEKS)
    panel = MediaPanel(
        dates=np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(WEEKS),
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=(Channel("tv"),),
        controls=(),
        kpi=kpi[None],
        kpi_unit=Revenue(),
        spend=spend[None, :, None],
        currency="EUR",
        control_values=np.zeros((1, WEEKS, 0)),
        filled=np.zeros((1, WEEKS), dtype=bool),
    )
    planned = SHARE * np.full(SPAN, 100.0)
    run = planned.copy()
    run[SPAN - READOUT : SPAN - READOUT + DARK] = 0.0
    gap = channel(run / SHARE) - channel(planned / SHARE)
    shocks = rng.normal(0.0, GEO_SD * math.sqrt(1.0 - phi**2), WEEKS + READOUT)
    noise = np.empty(WEEKS + READOUT)
    noise[0] = rng.normal(0.0, GEO_SD)
    for t in range(1, noise.size):
        noise[t] = phi * noise[t - 1] + shocks[t]
    placebo = PlaceboNoise(noise[:WEEKS])
    scale = math.sqrt(SHARE / (1.0 - SHARE))
    gaps = SHARE * np.asarray(gap)[-READOUT:] + scale * noise[WEEKS:]
    series = LiftSeries(run, planned, gaps, placebo.covariance(READOUT, SHARE), SHARE)
    return {"panel": panel, "series": series}


def job(decay: float, phi: float, seed: int, out: Path) -> str:
    target = out / f"{decay:g}-{phi:g}" / f"{seed}.json"
    if target.exists():
        return f"{target} kept"
    from mixdyn import NATIONAL
    from mixdyn.models.bayes_mmm import BayesMedia, BayesSpec, TanhCurve, fit_bayes

    w = world(decay, phi, seed)
    spec = BayesSpec((BayesMedia("tv"),), saturation=TanhCurve())
    record: dict[str, Any] = {"decay": decay, "phi": phi, "seed": seed}
    series = w["series"]
    for name in ("sum", "series"):
        fit = fit_bayes(
            w["panel"],
            NATIONAL,
            spec,
            seed=seed,
            warmup=500,
            draws=500,
            estimates={"tv": [series.summed]} if name == "sum" else None,
            series={"tv": [series]} if name == "series" else None,
        )
        draws = fit.draws["retention"][..., 0].reshape(-1)
        roi = fit.draws["roi"][..., 0].reshape(-1)
        low, high = np.quantile(draws, [(1.0 - LEVEL) / 2.0, (1.0 + LEVEL) / 2.0])
        record[name] = {
            "mean": float(draws.mean()),
            "median": float(np.median(draws)),
            "low": float(low),
            "high": float(high),
            "roi_mean": float(roi.mean()),
            "divergent": int(fit.divergent.sum()),
            "rhat": float(np.max(fit.diagnostics["retention"].rhat)),
        }
    target.parent.mkdir(parents=True, exist_ok=True)
    aside = target.with_suffix(f".{os.getpid()}.tmp")
    aside.write_text(json.dumps(record))
    aside.replace(target)
    return f"{target} written"


def summary(out: Path) -> None:
    for cell in sorted(p for p in out.iterdir() if p.is_dir()):
        records = [json.loads(p.read_text()) for p in sorted(cell.glob("*.json"))]
        if not records:
            continue
        truth = records[0]["decay"]
        n = len(records)
        print(f"decay {truth:g}, phi {records[0]['phi']:g}: {n} worlds")
        for name in ("sum", "series"):
            reads = [r[name] for r in records]
            error = np.array([r["mean"] for r in reads]) - truth
            covered = np.array([r["low"] <= truth <= r["high"] for r in reads], dtype=float)
            width = np.array([r["high"] - r["low"] for r in reads])
            divergent = sum(r["divergent"] for r in reads)
            rhat = max(r["rhat"] for r in reads)
            cover = covered.mean()
            nominal_se = math.sqrt(LEVEL * (1.0 - LEVEL) / n)
            print(
                f"  {name:6s} bias {error.mean():+.4f} ({_se(error):.4f})"
                f"  rmse {math.sqrt(np.mean(error**2)):.4f}"
                f"  cover {cover:.3f} ({_se(covered):.3f}; {(cover - LEVEL) / nominal_se:+.2f} SE"
                f" of {LEVEL:g})  width {width.mean():.4f} ({_se(width):.4f})"
                f"  divergent {divergent}  max R-hat {rhat:.3f}"
            )


def _se(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1) / math.sqrt(x.size))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--decays", type=float, nargs="+", default=[0.3, 0.7])
    parser.add_argument("--phis", type=float, nargs="+", default=[0.0, 0.5])
    parser.add_argument("--worlds", type=int, default=120)
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.summary:
        summary(args.out)
        return
    jobs = [
        (decay, phi, seed, args.out)
        for seed in range(args.first, args.first + args.worlds)
        for decay in args.decays
        for phi in args.phis
    ]
    context = get_context("spawn")
    with ProcessPoolExecutor(args.processes, mp_context=context) as pool:
        futures = [pool.submit(job, decay, phi, seed, out) for decay, phi, seed, out in jobs]
        for done in as_completed(futures):
            print(done.result(), flush=True)


if __name__ == "__main__":
    main()
