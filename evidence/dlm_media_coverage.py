"""Coverage of the DLM over media: on worlds whose media effects drift by a random walk, does the
interval mixdyn's dynamic model gives for each channel's coefficient at the last week hold the
world's, when the model is told each channel's true carryover and curve?

**The worlds**, ``WORLDS`` of them, world ``i`` from ``SEED + i``, written here and not taken from
mixdyn or the bench: ``WEEKS`` weeks of two channels, television and search.

* Spend: television on in a week with probability 0.7 and search with 0.9, each then gamma(2, 50)
  and gamma(2, 20) euros.
* Each channel's true transform, as mixdyn's `Transform` states it: a normalised geometric
  carryover of ``LENGTH`` weeks, retention ``U(0.3, 0.8)`` for television and ``U(0, 0.4)`` for
  search; a curve half its ceiling at ``U(0.5, 2)`` times the channel's mean positive spend, a
  ``tanh`` or, with probability 1/2, a Hill of slope ``U(0.8, 2.5)``. The column ``s_c,t`` is the
  spend carried over and saturated, computed here by its definition in NumPy.
* The drift, the walk law: ``x_c,t = x_c,t-1 + sigma_c e_c,t`` with ``x_c,-1 = 0``,
  ``e ~ N(0, 1)``, ``sigma_c ~ logU[0.01, 0.04]`` a week, and ``m_c = e^x_c`` multiplying the
  channel's effect.
* The KPI: ``1000 + 50 sin(2 pi t / 52.18 + phi) + sum_c beta_c m_c,t s_c,t + N(0, 25^2)``,
  ``phi ~ U(0, 2 pi)``, ``beta ~ U(200, 600)`` for television and ``U(100, 300)`` for search.
* The truth read is ``beta_c m_c,T``, each channel's coefficient on its column at the last week.

**The fit of each world**: `mixdyn.models.choose_dlm` on one geo, the spec a level, each channel an
`Effect` carrying its true transform, and a season of period 52.18 and one harmonic at discount 1;
the discounts chosen on the grid trend (0.98, 0.99, 1.0) by effects (0.97, 0.99, 1.0), two
parameters, the likelihood ratio's set at 0.95. The interval is
`DlmChoice.coefficient_bounds(0.9)`: the union over the kept specs of each one's central 90 %
Student t interval for the coefficient at the last week, given every week (projection).

**Pre-registered, 10-03, before any world of this stream was fitted**, on mixdyn `5f9a413`; two
worlds of another stream were run first to check that the script runs, and only their kept sets'
sizes and columns were read. The check passes if, for each channel, the interval holds
``beta_c m_c,T`` in at least 0.88 of the ``WORLDS`` worlds; each channel's count is reported with
its Clopper-Pearson 95 % interval. A miss is reported as a failure and nothing is tuned after the
run. Reported beside it, read by no rule: the best spec's own 90 % interval's coverage; the median
ratio of the projected width to the best spec's; the kept set's size; how often each effects'
discount is the best; the coverage by curve family and by tercile of ``sigma_c``; and how far the
model's columns are from the world's.

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 uv run --project <mixdyn checkout> \\
        --no-sync python evidence/dlm_media_coverage.py run --out <json>
    uv run --project <mixdyn checkout> --no-sync python evidence/dlm_media_coverage.py \\
        report <json>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import jax
import numpy as np
from sbc_bayes_mmm import checkout
from scipy.stats import beta as beta_distribution
from scipy.stats import t as student

from mixdyn import NATIONAL, Cadence, Channel, MediaPanel, Revenue
from mixdyn.models import DiscountGrid, DlmSpec, Effect, Season, Trend, choose_dlm
from mixdyn.transforms import HillShape, TanhShape, Transform

jax.config.update("jax_enable_x64", True)

SEED, WORLDS, WEEKS, LENGTH, LEVEL, GATE = 20261010, 200, 156, 8, 0.9, 0.88
CHANNELS = ("tv", "search")
GRID = DiscountGrid(trend=(0.98, 0.99, 1.0), effects=(0.97, 0.99, 1.0))
SEASON = Season(52.18, 1, 1.0)


def column(spend: np.ndarray, truth: dict[str, Any]) -> np.ndarray:
    """The spend carried over by the normalised geometric kernel and saturated, by definition."""
    weights = truth["retention"] ** np.arange(LENGTH)
    weights = weights / weights.sum()
    adstock = np.convolve(spend, weights)[: spend.size]
    z = adstock / truth["half"]
    if truth["family"] == "hill":
        return z ** truth["slope"] / (1.0 + z ** truth["slope"])
    return np.tanh(math.atanh(0.5) * z)


def world(i: int) -> tuple[MediaPanel, dict[str, dict[str, Any]]]:
    rng = np.random.default_rng(SEED + i)
    on = {"tv": 0.7, "search": 0.9}
    scale = {"tv": 50.0, "search": 20.0}
    retention = {"tv": (0.3, 0.8), "search": (0.0, 0.4)}
    effect = {"tv": (200.0, 600.0), "search": (100.0, 300.0)}
    spend = {c: rng.gamma(2.0, scale[c], WEEKS) * (rng.random(WEEKS) < on[c]) for c in CHANNELS}
    truth: dict[str, dict[str, Any]] = {}
    for c in CHANNELS:
        hill = bool(rng.random() < 0.5)
        truth[c] = {
            "retention": float(rng.uniform(*retention[c])),
            "half": float(rng.uniform(0.5, 2.0) * spend[c][spend[c] > 0].mean()),
            "family": "hill" if hill else "tanh",
            "slope": float(rng.uniform(0.8, 2.5)) if hill else None,
            "sigma": float(math.exp(rng.uniform(math.log(0.01), math.log(0.04)))),
            "beta": float(rng.uniform(*effect[c])),
        }
    phase = rng.uniform(0.0, 2.0 * math.pi)
    kpi = 1000.0 + 50.0 * np.sin(2.0 * math.pi * np.arange(WEEKS) / 52.18 + phase)
    for c in CHANNELS:
        drift = np.exp(np.cumsum(truth[c]["sigma"] * rng.standard_normal(WEEKS)))
        kpi = kpi + truth[c]["beta"] * drift * column(spend[c], truth[c])
        truth[c]["at_last"] = float(truth[c]["beta"] * drift[-1])
    kpi = kpi + 25.0 * rng.standard_normal(WEEKS)
    panel = MediaPanel(
        dates=np.datetime64("2023-01-02") + np.timedelta64(7, "D") * np.arange(WEEKS),
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=tuple(Channel(c) for c in CHANNELS),
        controls=(),
        kpi=kpi[None, :],
        kpi_unit=Revenue(),
        spend=np.column_stack([spend[c] for c in CHANNELS])[None],
        currency="EUR",
        control_values=np.zeros((1, WEEKS, 0)),
        filled=np.zeros((1, WEEKS), dtype=bool),
    )
    return panel, truth


def transform(truth: dict[str, Any]) -> Transform:
    shape = HillShape(truth["slope"]) if truth["family"] == "hill" else TanhShape()
    return Transform(truth["retention"], truth["half"], shape, LENGTH)


def validated(i: int) -> dict[str, Any]:
    panel, truth = world(i)
    effects = tuple(Effect(c, 0.99, transform=transform(truth[c])) for c in CHANNELS)
    choice = choose_dlm(panel, NATIONAL, DlmSpec(effects, Trend(1, 0.99), SEASON), GRID)
    projected = choice.coefficient_bounds(LEVEL)
    best = choice.fit.filtered
    half = student.ppf(0.5 + LEVEL / 2.0, best.dof[-1]) * np.sqrt(np.diag(best.covariance[-1]))
    kept = [choice.specs[k] for k in choice.kept]
    record: dict[str, Any] = {
        "world": i,
        "truth": truth,
        "kept": len(kept),
        "best": {
            "trend": choice.fit.spec.trend.discount if choice.fit.spec.trend else None,
            "effects": choice.fit.spec.effects[0].discount,
        },
    }
    for j, c in enumerate(CHANNELS):
        mean = float(best.mean[-1, 1 + j])
        ours = column(panel.spend[0, :, j], truth[c])
        record[c] = {
            "projected": [projected[c].lower, projected[c].upper],
            "pick": [mean - float(half[1 + j]), mean + float(half[1 + j])],
            "column_error": float(np.max(np.abs(choice.fit.x[:, j] - ours)) / np.max(ours)),
        }
    return record


def run(out: Path) -> None:
    """Every world not yet in ``out``'s ``.jsonl``, then ``out``."""
    source = checkout()
    partial = out.with_suffix(".jsonl")
    done = {}
    if partial.exists():
        for line in partial.read_text().splitlines():
            record = json.loads(line)
            done[record["world"]] = record
    with partial.open("a") as sink:
        for i in range(WORLDS):
            if i in done:
                continue
            record = validated(i)
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            done[i] = record
    out.write_text(
        json.dumps(
            {
                "mixdyn": source,
                "seed": SEED,
                "weeks": WEEKS,
                "level": LEVEL,
                "worlds": [done[i] for i in sorted(done)],
            },
            indent=1,
        )
    )


def clopper_pearson(held: int, n: int) -> tuple[float, float]:
    low = beta_distribution.ppf(0.025, held, n - held + 1) if held > 0 else 0.0
    high = beta_distribution.ppf(0.975, held + 1, n - held) if held < n else 1.0
    return float(low), float(high)


def holds(worlds: list[dict[str, Any]], channel: str, which: str) -> np.ndarray:
    truth = np.array([w["truth"][channel]["at_last"] for w in worlds])
    ends = np.array([w[channel][which] for w in worlds])
    return (ends[:, 0] <= truth) & (truth <= ends[:, 1])


def report(record: dict[str, Any]) -> None:
    worlds = record["worlds"]
    n = len(worlds)
    print(f"{n} worlds on mixdyn {record['mixdyn']['commit'][:7]}; {record['level']:.0%} intervals")
    verdicts = []
    for c in CHANNELS:
        held = holds(worlds, c, "projected")
        low, high = clopper_pearson(int(held.sum()), n)
        verdicts.append(held.mean() >= GATE)
        pick = holds(worlds, c, "pick")
        width = np.array([w[c]["projected"][1] - w[c]["projected"][0] for w in worlds])
        own = np.array([w[c]["pick"][1] - w[c]["pick"][0] for w in worlds])
        print(
            f"  {c:6s} projected holds {held.sum():3d} of {n}, {held.mean():.3f} "
            f"[{low:.3f}, {high:.3f}]; the best spec's own {pick.mean():.3f}; "
            f"width over the best's, median {np.median(width / own):.3f}"
        )
        family = np.array([w["truth"][c]["family"] for w in worlds])
        for name in ("tanh", "hill"):
            print(f"    {name}: {held[family == name].mean():.3f} of {(family == name).sum()}")
        sigma = np.array([w["truth"][c]["sigma"] for w in worlds])
        cuts = np.quantile(sigma, [1 / 3, 2 / 3])
        tercile = np.searchsorted(cuts, sigma)
        shares = ", ".join(f"{held[tercile == k].mean():.3f}" for k in range(3))
        print(f"    by tercile of sigma, low to high: {shares}")
    error = max(w[c]["column_error"] for w in worlds for c in CHANNELS)
    print(f"  the model's columns against the world's, largest error over the largest: {error:.1e}")
    kept = np.array([w["kept"] for w in worlds])
    print(f"  kept specs: median {np.median(kept):.0f}, from {kept.min()} to {kept.max()} of 9")
    effects = [w["best"]["effects"] for w in worlds]
    print(
        "  best effects' discount: "
        + ", ".join(f"{d}: {effects.count(d)}" for d in (0.97, 0.99, 1.0))
    )
    verdict = "passes" if all(verdicts) else "FAILS"
    print(f"  the check {verdict}: each channel at least {GATE}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    go = sub.add_parser("run")
    go.add_argument("--out", type=Path, required=True)
    show = sub.add_parser("report")
    show.add_argument("record", type=Path)
    args = parser.parse_args()
    if args.command == "report":
        report(json.loads(args.record.read_text()))
        return
    run(args.out)


if __name__ == "__main__":
    main()
