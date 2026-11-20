"""Does a price lever's contribution interval cover the world's, and does the plan made around it
stay near the best plan? Posteriors, coverage and regret only, no wall time.

Each world is 104 weeks of two channels and a price. TV's spend is a gamma of mean 100 a week on
70 % of the weeks and 0 on the rest; search's a random walk about 100, at least 10. Each channel is
carried over by the normalised geometric kernel over the panel's 104 weeks, TV's at a retention of
0.6 and search's at 0.2, and saturated by a tanh that reaches half its ceiling at 80 and 120, under
ceilings of 300 and 250. The price is 10 a week, and 8 in a promotion week, each with noise of SD
0.2. In the `independent` world a promotion runs in a quarter of the weeks, drawn apart from the
spend; in the `confounded` one it runs in half the weeks TV spends above 120 and in a tenth of the
rest, so the price moves with TV. The KPI is 1000 plus the channels' contributions, minus 40 a unit
of price, plus normal noise of SD `--noise`.

The panel declares the price a lever measured against its observed most
(`Lever(LeverKind.PRICE, Baseline.MAX)`): the world's lever contribution is
`-40 * sum_t (price_t - max price)`, what the weeks below the highest price added. Each world is
fitted by the one-geo Bayesian model with the world's curve family, 4 chains of 500 draws after 500
of warm-up started by Pathfinder, as each of `--arms` says:

    lever    the price a control, free in sign, read as the panel's lever
    omitted  no price in the model

Each arm plans 13 weeks for its posterior's expected return (`plan_posterior`), the budget the
history's mean weekly spend over both channels times 13, each channel's range a week 0 to three
times its mean over the history. The best plan is `plan_posterior`'s on the world's own channels, a
posterior of one draw. A plan's regret per euro is the best plan's worth less its own, both read on
the world's channels, over the budget: the bench scorecard's unit, whose claim margin is 0.01.

Each job writes `<out>/<seed>.json`; a job already written is skipped. `--summary` reads them back:
the share of worlds whose central 90 % and 95 % intervals of the lever's contribution hold the
world's, with its exact (Clopper-Pearson) 95 % interval, the lever's bias, and each arm's regret
per euro, its mean with its 95 % Student-t interval, and its most; with both arms, the paired
difference of their regrets, the lever arm's less the omitted one's, world by world. It also
refits each world by least squares on the world's own media columns and the price, an estimator
that knows every transform, and reads its 90 % and 95 % Student-t intervals of the lever's
contribution the same way: what the worlds themselves allow.

Where the regret comes from. A plan of two channels at a budget is one number, TV's spend a week,
search taking the rest. Each job also writes, for each arm's plan:

    expected   the posterior's own expected regret per euro of its plan: under each draw, the
               draw's best split of the week's budget less the plan's, read on a grid of `SPLIT`
               splits, over the budget, averaged over the draws
    pit        the share of the draws under which the plan's regret is below the world's: about
               uniform over the worlds where the posterior's spread of the regret is the world's
    tv, search the regret of the plan made with that channel's draws set to the world's values,
               read on the same splits: what the other channel's error alone costs. The worth is
               separable in the channels, so that plan's expected worth is the channel's world
               worth plus the other's posterior mean
    spend, best  TV's spend a week in the plan and in the best plan, on the same splits
    seen, held the regret of the plan made within each channel's range of what the history spent
               a week, its least to its most, against the best plan within them; and what holding
               those ranges costs the best plan

and `--summary` reads each of them over the worlds, with two plans beside: the history's mean
split kept, and the plan of least squares on the world's own media columns and the price, which
knows every carryover and curve and estimates the two ceilings; a plan's worth is linear in the
ceilings, so that is also the plan of their Gaussian posterior.

ADR 0068's measurements, worlds 0 to 19 of each:

    python scripts/study_levers.py --world independent --noise 30 --out out/independent-30
    python scripts/study_levers.py --world independent --noise 15 --out out/independent-15
    python scripts/study_levers.py --world confounded --noise 30 --arms lever omitted \
        --out out/confounded-30
    python scripts/study_levers.py --out out/independent-30 --summary

and, both arms on the same worlds, as many as the compute allowed:

    python scripts/study_levers.py --world independent --noise 30 --arms lever omitted \
        --worlds 100 --out out/paired-independent-30

and the lever arm's regret decomposed on worlds 0 to 142, and on worlds 0 to 19 at a tenth of
the noise:

    python scripts/study_levers.py --world independent --noise 30 --worlds 143 \
        --out out/decomposed-independent-30
    python scripts/study_levers.py --world independent --noise 3 --out out/independent-3
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

WEEKS, PLAN = 104, 13
RETENTION, HALF, CEILING = (0.6, 0.2), (80.0, 120.0), (300.0, 250.0)
PRICE, PROMOTED, PRICE_SD, PRICE_EFFECT = 10.0, 8.0, 0.2, -40.0
BASE = 1000.0
PROMOTED_SHARE = {"independent": (0.25, 0.25), "confounded": (0.5, 0.1)}
"""A week's chance of a promotion where TV spends above 120, and where it does not."""
DELTA = 0.01
"""The scorecard's claim margin, regret per euro."""
SPLIT = 1001
"""The splits of a week's budget between the channels the decomposition reads each draw at."""


def world(seed: int, kind: str = "independent", noise: float = 30.0) -> dict[str, Any]:
    """The world's panel, its lever contribution and its channels; see the module."""
    from chc import response

    from mixdyn import NATIONAL, Baseline, Cadence, Channel, Lever, LeverKind, MediaPanel, Revenue
    from mixdyn.models.bayes_mmm import _TANH_HALF

    rng = np.random.default_rng([20261106, seed])
    tv = rng.gamma(2.0, 50.0, WEEKS) * (rng.random(WEEKS) < 0.7)
    search = np.maximum(10.0, 100.0 + np.cumsum(rng.normal(0.0, 5.0, WEEKS)))
    high, low = PROMOTED_SHARE[kind]
    promoted = rng.random(WEEKS) < np.where(tv > 120.0, high, low)
    price = np.where(promoted, PROMOTED, PRICE) + rng.normal(0.0, PRICE_SD, WEEKS)
    spend = np.column_stack([tv, search])
    channels = [
        response.Channel(
            response.GeometricAdstock(r, length=WEEKS, normalized=True),
            response.Tanh(k / _TANH_HALF),
            c,
        )
        for r, k, c in zip(RETENTION, HALF, CEILING, strict=True)
    ]
    media = sum(np.asarray(ch(spend[:, j])) for j, ch in enumerate(channels))
    kpi = BASE + media + PRICE_EFFECT * price + rng.normal(0.0, noise, WEEKS)
    panel = MediaPanel(
        dates=np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(WEEKS),
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=(Channel("tv"), Channel("search")),
        controls=("price",),
        kpi=kpi[None],
        kpi_unit=Revenue(),
        spend=spend[None],
        currency="EUR",
        control_values=price[None, :, None],
        filled=np.zeros((1, WEEKS), dtype=bool),
        levers={"price": Lever(LeverKind.PRICE, Baseline.MAX)},
    )
    return {"panel": panel, "lever": PRICE_EFFECT * float(np.sum(price - price.max()))}


def truth(panel: Any) -> Any:
    """The world's channels as a posterior of one draw."""
    from mixdyn.decisions import Readings

    return Readings(
        channels=panel.channels,
        retention=np.array([RETENTION]),
        half_saturation=np.array([HALF]),
        slope=np.full((1, 2), np.nan),
        coefficient=np.array([CEILING]),
        tanh=np.array([True]),
        fit=np.array([0]),
        chain=np.array([0]),
        stacking=np.array([1.0]),
        pointwise=None,
        length=WEEKS,
    )


def budget(panel: Any) -> tuple[float, dict[Any, Any]]:
    """The plan's budget and each channel's range a week; see the module."""
    from mixdyn.decisions import Range

    spend = panel.spend[0]
    limits = {c: Range(0.0, 3.0 * float(spend[:, j].mean())) for j, c in enumerate(panel.channels)}
    return PLAN * float(spend.mean(axis=0).sum()), limits


def regret(panel: Any, reads: Any) -> float:
    """The regret per euro of `reads`' plan, read on the world's channels."""
    from mixdyn import NATIONAL
    from mixdyn.decisions import plan_posterior

    total, limits = budget(panel)
    return regret_at(panel, plan_posterior(panel, NATIONAL, reads, total, PLAN, limits).spend)


def regret_at(panel: Any, spend: np.ndarray) -> float:
    """The regret per euro of spending `spend` a week on the channels, read on the world's."""
    from mixdyn import NATIONAL
    from mixdyn.decisions import plan_posterior
    from mixdyn.decisions.posterior import _history, _mean_worths

    total, limits = budget(panel)
    world_reads = truth(panel)
    best = plan_posterior(panel, NATIONAL, world_reads, total, PLAN, limits)
    at = np.column_stack([best.spend, spend])
    worth = _mean_worths(_history(panel, NATIONAL, world_reads), world_reads, PLAN, at).sum(axis=0)
    return float((worth[0] - worth[1]) / total)


def decomposed(panel: Any, reads: Any, lost: float) -> dict[str, float]:
    """Where the regret per euro `lost` of `reads`' plan comes from; see the module."""
    from mixdyn import NATIONAL
    from mixdyn.decisions import Range, worths

    total, limits = budget(panel)
    rate = total / PLAN
    tv, search = panel.channels
    low = max(limits[tv].least, rate - limits[search].most)
    high = min(limits[tv].most, rate - limits[search].least)
    split = {tv: Range(low, high), search: Range(rate - high, rate - low)}
    read = worths(panel, NATIONAL, reads, PLAN, split, grid=SPLIT)
    world = worths(panel, NATIONAL, truth(panel), PLAN, split, grid=SPLIT).table[0]
    # search's spends run from the week's rest down, so reversed they pair with TV's
    drawn = read.table[:, 0] + read.table[:, 1, ::-1]
    weights = reads.weights
    mean = weights @ read.table[:, 0], weights @ read.table[:, 1, ::-1]
    true = world[0] + world[1, ::-1]
    plan = int(np.argmax(mean[0] + mean[1]))
    lost_at = (drawn.max(axis=1) - drawn[:, plan]) / total
    spent = panel.spend[0]
    seen = (read.rates[0] >= max(spent[:, 0].min(), rate - spent[:, 1].max())) & (
        read.rates[0] <= min(spent[:, 0].max(), rate - spent[:, 1].min())
    )
    within = np.where(seen, mean[0] + mean[1], -np.inf), np.where(seen, true, -np.inf)

    def regret_of(curve: np.ndarray) -> float:
        return float((true.max() - true[np.argmax(curve)]) / total)

    return {
        "expected": float(weights @ lost_at),
        "pit": float(weights @ (lost_at < lost)),
        "tv": regret_of(world[0] + mean[1]),
        "search": regret_of(mean[0] + world[1, ::-1]),
        "spend": float(read.rates[0, plan]),
        "best": float(read.rates[0, np.argmax(true)]),
        "seen": float((within[1].max() - true[np.argmax(within[0])]) / total),
        "held": float((true.max() - within[1].max()) / total),
    }


def job(seed: int, kind: str, noise: float, arms: tuple[str, ...], out: Path) -> str:
    target = out / f"{seed}.json"
    if target.exists():
        return f"{target} kept"
    from mixdyn import NATIONAL
    from mixdyn.decisions import readings
    from mixdyn.models.bayes_mmm import (
        BayesMedia,
        BayesSpec,
        TanhCurve,
        fit_bayes,
        lever_contributions,
    )
    from mixdyn.models.ridge_mmm import Control

    w = world(seed, kind, noise)
    panel = w["panel"]
    media = (BayesMedia("tv"), BayesMedia("search"))
    record: dict[str, Any] = {"seed": seed, "world": kind, "noise": noise, "truth": w["lever"]}
    for name in arms:
        controls = (Control("price"),) if name == "lever" else ()
        spec = BayesSpec(media, controls, saturation=TanhCurve())
        fit = fit_bayes(panel, NATIONAL, spec, seed=seed, warmup=500, draws=500)
        reads = readings(fit)
        lost = regret(panel, reads)
        read: dict[str, Any] = {
            "regret": lost,
            **decomposed(panel, reads, lost),
            "divergent": int(fit.divergent.sum()),
            "rhat": float(max(np.max(d.rhat) for d in fit.diagnostics.values())),
        }
        if controls:
            draws = lever_contributions(fit, panel, NATIONAL)["price"]
            read["mean"] = float(draws.mean())
            for level in (0.9, 0.95):
                low, high = np.quantile(draws, [(1.0 - level) / 2.0, (1.0 + level) / 2.0])
                read[f"{level:g}"] = [float(low), float(high)]
        record[name] = read
    target.parent.mkdir(parents=True, exist_ok=True)
    aside = target.with_suffix(f".{os.getpid()}.tmp")
    aside.write_text(json.dumps(record))
    aside.replace(target)
    return f"{target} written"


def summary(out: Path) -> None:
    records = [json.loads(p.read_text()) for p in sorted(out.glob("*.json"))]
    n = len(records)
    kind, noise = records[0]["world"], records[0]["noise"]
    print(f"{n} {kind} worlds at noise {noise:g}")
    lever = np.array([r["truth"] for r in records])
    error = np.array([r["lever"]["mean"] for r in records]) - lever
    print(f"  lever bias {error.mean():+.1f} ({_se(error):.1f}), truth mean {lever.mean():.1f}")
    for level in ("0.9", "0.95"):
        bounds = np.array([r["lever"][level] for r in records])
        _coverage(f"{level} interval", (bounds[:, 0] <= lever) & (lever <= bounds[:, 1]), level)
    known = np.array([_least_squares(r["seed"], kind, noise) for r in records])
    estimate, se, dof = known[:, :3].T
    error = estimate - lever
    print(f"  least squares bias {error.mean():+.1f} ({_se(error):.1f})")
    for level in ("0.9", "0.95"):
        from scipy import stats

        width = stats.t.ppf((1.0 + float(level)) / 2.0, dof) * se
        _coverage(f"least squares {level} interval", np.abs(error) <= width, level)
    arms = [a for a in ("lever", "omitted") if all(a in r for r in records)]
    for name in arms:
        x = np.array([r[name]["regret"] for r in records])
        low, high = _interval(x)
        divergent = sum(r[name]["divergent"] for r in records)
        rhat = max(r[name]["rhat"] for r in records)
        print(
            f"  {name:8s} regret per euro mean {x.mean():.5f}, 95 % interval [{low:.5f},"
            f" {high:.5f}] ({'within' if high < DELTA else 'upper end above'} {DELTA:g}), most"
            f" {x.max():.5f}, above {DELTA:g} in {int((x > DELTA).sum())}; divergent"
            f" {divergent}, max R-hat {rhat:.3f}"
        )
        if all("expected" in r[name] for r in records):
            _decomposition(name, records)
    for j, name in enumerate(("the history's split kept", "least squares, curves known")):
        x = known[:, 3 + j]
        low, high = _interval(x)
        print(
            f"  {name}: regret per euro mean {x.mean():.5f}, 95 % interval [{low:.5f},"
            f" {high:.5f}], median {np.median(x):.5f}, above {DELTA:g} in {int((x > DELTA).sum())}"
        )
    for name in arms:
        x = np.array([r[name]["regret"] for r in records]) - known[:, 4]
        low, high = _interval(x)
        print(
            f"  {name} less least squares with the curves known, paired: mean {x.mean():+.5f},"
            f" 95 % interval [{low:+.5f}, {high:+.5f}]"
        )
    if len(arms) == 2:
        x = np.array([r["lever"]["regret"] - r["omitted"]["regret"] for r in records])
        low, high = _interval(x)
        print(
            f"  lever less omitted, paired: mean {x.mean():+.5f}, 95 % interval [{low:+.5f},"
            f" {high:+.5f}]; the lever's plan the better in {int((x < 0).sum())} of {n}"
        )


def _decomposition(name: str, records: list[dict[str, Any]]) -> None:
    """Print where an arm's regret comes from over the worlds; see the module."""
    from scipy import stats

    arm = [r[name] for r in records]
    n = len(arm)
    for key, label in (
        ("expected", "the posterior's own expected regret"),
        ("tv", "with TV's draws the world's"),
        ("search", "with search's draws the world's"),
        ("seen", "within the history's ranges, against the best plan within them"),
        ("held", "what those ranges cost the best plan"),
    ):
        x = np.array([a[key] for a in arm])
        low, high = _interval(x)
        print(f"    {label}: mean {x.mean():.5f}, 95 % interval [{low:.5f}, {high:.5f}]")
    pit = np.array([a["pit"] for a in arm])
    tail = int((pit >= 0.9).sum())
    print(
        f"    the world's regret among the draws': mean share below {pit.mean():.3f}, at or above"
        f" 0.9 in {tail} of {n} (0.1 of them where calibrated; exact binomial p"
        f" {stats.binomtest(tail, n, 0.1).pvalue:.2g})"
    )
    print(f"    the best plan leaves those ranges in {sum(a['held'] > 0.0 for a in arm)} of {n}")
    under = int(sum(a["spend"] < a["best"] for a in arm))
    gap = np.array([a["spend"] - a["best"] for a in arm])
    print(
        f"    TV a week in the plan less in the best plan: mean {gap.mean():+.1f}"
        f" ({_se(gap):.1f}), below it in {under} of {n}"
    )


def _coverage(name: str, held: np.ndarray, level: str) -> None:
    """Print a coverage count with its exact 95 % interval and where the nominal level falls."""
    from scipy import stats

    n, k, nominal = held.size, int(held.sum()), float(level)
    low = float(stats.beta.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0
    high = float(stats.beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0
    where = "inside" if low <= nominal <= high else "below" if nominal < low else "above"
    print(
        f"  {name} holds the world's in {k} of {n}: {k / n:.3f} (exact 95 % interval"
        f" [{low:.3f}, {high:.3f}], the nominal {nominal:g} {where} it)"
    )


def _interval(x: np.ndarray) -> tuple[float, float]:
    half = _t975(x.size - 1) * _se(x)
    return float(x.mean() - half), float(x.mean() + half)


def _least_squares(seed: int, kind: str, noise: float) -> tuple[float, float, int, float, float]:
    """The lever's contribution by least squares on the world's own media columns and the price:
    its estimate, standard error and degrees of freedom; and the regret per euro of the history's
    split kept and of the plan of least squares' ceilings on the world's curves."""
    from chc import response

    from mixdyn.models.bayes_mmm import _TANH_HALF

    panel = world(seed, kind, noise)["panel"]
    spend, price = panel.spend[0], panel.control_values[0, :, 0]
    columns = [
        np.asarray(
            response.Channel(
                response.GeometricAdstock(r, length=WEEKS, normalized=True),
                response.Tanh(k / _TANH_HALF),
                1.0,
            )(spend[:, j])
        )
        for j, (r, k) in enumerate(zip(RETENTION, HALF, strict=True))
    ]
    design = np.column_stack([np.ones(WEEKS), *columns, price])
    beta, *_ = np.linalg.lstsq(design, panel.kpi[0], rcond=None)
    residual = panel.kpi[0] - design @ beta
    dof = WEEKS - design.shape[1]
    variance = residual @ residual / dof * np.linalg.inv(design.T @ design)[-1, -1]
    departure = float(np.sum(price - price.max()))
    known = dataclasses.replace(truth(panel), coefficient=beta[None, 1:3])
    return (
        float(beta[-1] * departure),
        float(np.sqrt(variance) * abs(departure)),
        dof,
        regret_at(panel, spend.mean(axis=0)),
        regret(panel, known),
    )


def _se(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1) / math.sqrt(x.size))


def _t975(dof: int) -> float:
    from scipy import stats

    return float(stats.t.ppf(0.975, dof))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=20)
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--processes", type=int, default=2)
    parser.add_argument("--world", choices=sorted(PROMOTED_SHARE), default="independent")
    parser.add_argument("--noise", type=float, default=30.0)
    parser.add_argument("--arms", nargs="+", choices=["lever", "omitted"], default=["lever"])
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.summary:
        summary(args.out)
        return
    seeds = range(args.first, args.first + args.worlds)
    context = get_context("spawn")
    with ProcessPoolExecutor(args.processes, mp_context=context) as pool:
        futures = [
            pool.submit(job, seed, args.world, args.noise, tuple(args.arms), args.out)
            for seed in seeds
        ]
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
