"""Validation of mixdyn's nested media-mix model on worlds where television moves branded searches
and searches move sales: does its interval for television's return on sales, through the searches
and directly, hold the world's at its nominal rate, and what does reading the mediator buy over a
one-stage model that leaves it out?

**The worlds**, ``WORLDS`` of them, world ``i`` from ``SEED + i``: ``WEEKS`` weeks; television and
search ads, each on in a week with probability 0.7 and 0.8, its spend then gamma(2, 50) and
gamma(2, 30) euros; a yearly season; a price about 10 euros.

* The searches: 500 + 50 season + television's contribution + N(0, 15²). Television's return on
  them, ``mediated ~ U(0.2, 1.0)`` searches a euro, runs through a normalised geometric carryover of
  decay ``U(0.3, 0.8)`` and a Hill of slope 1, its half-saturation point ``U(0.5, 1.5)`` mean
  positive spends.
* The sales: 1000 + 40 season + ``gamma`` searches - 20 (price - 10) + the search ads' contribution
  + television's direct contribution + N(0, 20²). ``gamma ~ U(1, 3)`` euros a search; the search
  ads' return ``U(0.5, 2.0)``, decay 0.1, half-saturation point 0.8; television's direct return 0
  in the even worlds and ``U(0.2, 1.0)`` in the odd, decay 0.3, half-saturation point 1.
* Each contribution over the weeks is its return times its spend, as the model's return is, so
  television's return on sales is ``gamma * mediated`` plus its direct return.

**Three fits of each world**, each ``CHAINS`` chains of ``DRAWS`` draws after ``WARMUP``, started
uniform; every stage reads a yearly season of one harmonic, and every channel the default return
prior but television's on the searches, ``log mediated ~ N(-0.5, 1)``:

* ``nested``: stage one television on the searches; stage two search ads and television, the price
  held at or below 0 and the searches at or above 0. Television is in both stages, as an analyst
  who does not know whether it also acts directly would put it.
* ``total``: one stage without the searches, so television's return is its whole effect.
* ``blocked``: one stage with the searches a control, so television's return is its direct path.

**Pre-registered, 10-02, before any world was fitted.** The validation passes if ``nested``'s 90 %
interval for television's return on sales, its draws' 5th to 95th percentiles, holds the world's
in a number of worlds between the 2.5th and 97.5th percentiles of a binomial of ``WORLDS`` trials
at 0.9. Reported beside it, read by no rule: each arm's coverage over all worlds, the even and the
odd; the search ads'; the mean log ratio of ``nested``'s interval width to ``total``'s, with its
95 % t interval; each arm's mean error of the median, likewise; and the fits' largest R-hat and
divergences.

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 uv run --project <mixdyn checkout> \\
        --no-sync python evidence/nested_validation.py run --out <json> [--workers 3]
    uv run --project <mixdyn checkout> --no-sync python evidence/nested_validation.py \\
        report <json>
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import jax
import numpy as np
from sbc_bayes_mmm import checkout
from scipy.stats import binom
from scipy.stats import t as student

from mixdyn import Cadence, Channel, Geo, MediaPanel, Revenue
from mixdyn.models.bayes_mmm import BayesFit, BayesMedia, BayesSpec, RoiPrior, fit_bayes
from mixdyn.models.nested import NestedSpec, fit_nested
from mixdyn.models.ridge_mmm import Control, FourierSeason

jax.config.update("jax_enable_x64", True)

SEED, WORLDS, WEEKS, CHAINS, WARMUP, DRAWS = 20261005, 100, 104, 4, 500, 500
NATIONAL = Geo("national")
SEASON = FourierSeason(52.18, 1)
PRICE, SEARCHES = Control("price", upper=0.0), Control("searches", lower=0.0)
NESTED = NestedSpec(
    "searches",
    BayesSpec((BayesMedia("tv", RoiPrior(-0.5, 1.0)),), season=SEASON),
    BayesSpec((BayesMedia("search"), BayesMedia("tv")), (PRICE, SEARCHES), season=SEASON),
)
TOTAL = BayesSpec((BayesMedia("search"), BayesMedia("tv")), (PRICE,), season=SEASON)
BLOCKED = dataclasses.replace(TOTAL, controls=(PRICE, SEARCHES))
ARMS = ("nested", "total", "blocked")


def contribution(spend: np.ndarray, roi: float, retention: float, half: float) -> np.ndarray:
    """``roi`` times the spend over the weeks, spread as the carried-over, saturated spend is."""
    weights = retention ** np.arange(spend.size)
    adstock = np.convolve(spend / spend[spend > 0].mean(), weights / weights.sum())[: spend.size]
    saturated = adstock / (adstock + half)
    return roi * spend.sum() * saturated / saturated.sum()


def world(i: int) -> tuple[MediaPanel, dict[str, float]]:
    rng = np.random.default_rng(SEED + i)
    tv = rng.gamma(2.0, 50.0, WEEKS) * (rng.random(WEEKS) < 0.7)
    search = rng.gamma(2.0, 30.0, WEEKS) * (rng.random(WEEKS) < 0.8)
    truth = {
        name: float(rng.uniform(low, high))
        for name, low, high in (
            ("mediated", 0.2, 1.0),
            ("retention", 0.3, 0.8),
            ("half", 0.5, 1.5),
            ("gamma", 1.0, 3.0),
            ("search", 0.5, 2.0),
            ("direct", 0.2, 1.0),
        )
    }
    if i % 2 == 0:
        truth["direct"] = 0.0
    season = np.sin(2.0 * np.pi * np.arange(WEEKS) / 52.18)
    searches = (
        500.0
        + 50.0 * season
        + contribution(tv, truth["mediated"], truth["retention"], truth["half"])
        + 15.0 * rng.standard_normal(WEEKS)
    )
    price = 10.0 + 0.5 * rng.standard_normal(WEEKS)
    sales = (
        1000.0
        + 40.0 * season
        + truth["gamma"] * searches
        - 20.0 * (price - 10.0)
        + contribution(search, truth["search"], 0.1, 0.8)
        + contribution(tv, truth["direct"], 0.3, 1.0)
        + 20.0 * rng.standard_normal(WEEKS)
    )
    truth["tv"] = truth["gamma"] * truth["mediated"] + truth["direct"]
    panel = MediaPanel(
        dates=np.datetime64("2023-01-02") + np.timedelta64(7, "D") * np.arange(WEEKS),
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=(Channel("tv"), Channel("search")),
        controls=("price", "searches"),
        kpi=sales[None, :],
        kpi_unit=Revenue(),
        spend=np.column_stack([tv, search])[None],
        currency="EUR",
        control_values=np.column_stack([price, searches])[None],
        filled=np.zeros((1, WEEKS), dtype=bool),
    )
    return panel, truth


def interval(draws: np.ndarray) -> list[float]:
    """The 5th percentile, the median and the 95th."""
    return [float(q) for q in np.quantile(draws, [0.05, 0.5, 0.95])]


def health(*fits: BayesFit) -> dict[str, float]:
    return {
        "rhat": max(float(np.max(d.rhat)) for f in fits for d in f.diagnostics.values()),
        "divergent": int(sum(int(f.divergent.sum()) for f in fits)),
    }


def validated(i: int) -> dict[str, Any]:
    panel, truth = world(i)
    seeds = [int(s) for s in np.random.SeedSequence(SEED + i).generate_state(3)]
    options = {"chains": CHAINS, "warmup": WARMUP, "draws": DRAWS}
    nested = fit_nested(panel, NATIONAL, NESTED, seed=seeds[0], **options)
    total = fit_bayes(panel, NATIONAL, TOTAL, seed=seeds[1], **options)
    blocked = fit_bayes(panel, NATIONAL, BLOCKED, seed=seeds[2], **options)
    roi = nested.roi()
    return {
        "world": i,
        "truth": truth,
        "nested": {
            "tv": interval(roi["tv"]),
            "search": interval(roi["search"]),
            "gamma": interval(nested.mediator_coefficient()),
            "mediated": interval(nested.stage_one.draws["roi"][..., 0]),
            "direct": interval(nested.stage_two.draws["roi"][..., 1]),
            **health(nested.stage_one, nested.stage_two),
        },
        **{
            name: {
                "tv": interval(fit.draws["roi"][..., 1]),
                "search": interval(fit.draws["roi"][..., 0]),
                **health(fit),
            }
            for name, fit in (("total", total), ("blocked", blocked))
        },
    }


def run(out: Path, workers: int) -> None:
    """Every world not yet in ``out``'s ``.jsonl``, then ``out``."""
    source = checkout()
    partial = out.with_suffix(".jsonl")
    done = {}
    if partial.exists():
        for line in partial.read_text().splitlines():
            record = json.loads(line)
            done[record["world"]] = record
    jobs = [i for i in range(WORLDS) if i not in done]
    with ProcessPoolExecutor(workers) as pool, partial.open("a") as sink:
        for future in as_completed([pool.submit(validated, i) for i in jobs]):
            record = future.result()
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            done[record["world"]] = record
    out.write_text(
        json.dumps(
            {
                "mixdyn": source,
                "seed": SEED,
                "chains": CHAINS,
                "warmup": WARMUP,
                "draws": DRAWS,
                "worlds": [done[i] for i in sorted(done)],
            },
            indent=1,
        )
    )


def mean_interval(x: np.ndarray) -> str:
    half = student.ppf(0.975, x.size - 1) * x.std(ddof=1) / np.sqrt(x.size)
    return f"{x.mean():+.4f} [{x.mean() - half:+.4f}, {x.mean() + half:+.4f}]"


def report(record: dict[str, Any]) -> None:
    worlds = record["worlds"]
    n = len(worlds)
    odd = np.array([w["world"] % 2 == 1 for w in worlds])
    low, high = binom.ppf(0.025, n, 0.9), binom.ppf(0.975, n, 0.9)
    print(f"{n} worlds on mixdyn {record['mixdyn']['commit'][:7]}; 90 % intervals")
    for channel in ("tv", "search"):
        truth = np.array([w["truth"][channel] for w in worlds])
        for arm in ARMS:
            q = np.array([w[arm][channel] for w in worlds])
            held = (q[:, 0] <= truth) & (truth <= q[:, 2])
            print(
                f"  {channel:6s} {arm:8s} holds {held.sum():3d} of {n} (even {held[~odd].sum()},"
                f" odd {held[odd].sum()}); median less truth {mean_interval(q[:, 1] - truth)}"
            )
            if channel == "tv" and arm == "nested":
                verdict = "passes" if low <= held.sum() <= high else "FAILS"
                print(f"    the validation {verdict}: [{low:.0f}, {high:.0f}] at 0.9")
    width = {arm: np.array([w[arm]["tv"][2] - w[arm]["tv"][0] for w in worlds]) for arm in ARMS}
    ratio = np.log(width["nested"] / width["total"])
    print(f"  log(nested width / total width), television: {mean_interval(ratio)}")
    for arm in ARMS:
        rhat = max(w[arm]["rhat"] for w in worlds)
        divergent = [w[arm]["divergent"] for w in worlds]
        print(
            f"  {arm:8s} largest R-hat {rhat:.4f}; divergent in {sum(d > 0 for d in divergent)}"
            f" worlds, {sum(divergent)} in all"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    go = sub.add_parser("run")
    go.add_argument("--out", type=Path, required=True)
    go.add_argument("--workers", type=int, default=3)
    show = sub.add_parser("report")
    show.add_argument("record", type=Path)
    args = parser.parse_args()
    if args.command == "report":
        report(json.loads(args.record.read_text()))
        return
    run(args.out, args.workers)


if __name__ == "__main__":
    main()
