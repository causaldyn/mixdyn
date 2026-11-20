"""Coverage of the DLM over media with each effect's discount on its own and a lower floor: on the
walk worlds of ``dlm_media_coverage.py``, drawn from a fresh stream, does the interval mixdyn's
dynamic model gives for each channel's coefficient at the last week hold the world's, when the
model is told each channel's true carryover and curve?

That check failed on its pre-registered rule: 0.860 and 0.855 against 0.88, the effects' best
discount at the grid's floor, 0.97, in 117 of 200 worlds. An exploration after it, on worlds of
another stream (``SEED + 100000``) and with no rule, put a shared grid reaching 0.90 at 0.895 and
0.920. This check sets its own rule before any world of its stream is fitted.

**The worlds**: ``dlm_media_coverage.py``'s law, unchanged, world ``i`` from ``SEED + i`` with
``SEED`` this file's. `world` here is a copy of that file's, the base seed made an argument; `run`
first checks the copy against that file's own worlds, bit for bit.

**The fit of each world**: `mixdyn.models.choose_dlm` on one geo, the spec a level, each channel an
`Effect` carrying its true transform, and a season of period 52.18 and one harmonic at discount 1,
as before; the discounts chosen on ``GRID``: the trend's at 0.98, 0.99 and 1 and each channel's own
(`mixdyn.models.PerEffect`) at 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99 and 1, three parameters,
the likelihood ratio's set at 0.95. The interval is `DlmChoice.coefficient_bounds(0.9)`, the union
over the kept specs of each one's central 90 % Student t interval for the coefficient at the last
week (projection).

**Pre-registered, 10-04, before any world of this stream was fitted**, on the mixdyn commit
``MIXDYN`` names. Two worlds of another stream (``SMOKE``) were run first to check that the script
runs, and only their kept sets' sizes, their columns' error and the run's time were read. The check
passes if, for each channel, the interval holds ``beta_c m_c,T`` in at least 0.88 of the ``WORLDS``
worlds; each channel's count is reported with its Clopper-Pearson 95 % interval. A miss is
reported as a failure and nothing is tuned after the run. On a pass, the scorecard's DLM arm takes
``GRID``'s discounts, each effect's on its own; on a failure it keeps its grid.

Reported beside it, read by no rule:

* the best spec's own 90 % interval's coverage, and the median ratio of the projected width to it;
* on the same worlds, the projection over the failed check's grid (``SHARED_097``) and over one
  discount shared by both channels at this grid's values (``SHARED``), their coverage and the
  median ratio of this grid's width to theirs: what the floor bought, and what each channel's own
  discount bought beside it;
* the kept set's size; how often each channel's discount and the trend's is the best;
* the ends of the grid the kept set reaches (`DlmChoice.edges`): how often each is named, how
  often the best sits there, and the coverage of the worlds that name one against those that do
  not;
* the coverage by curve family and by tercile of ``sigma_c``; and how far the model's columns are
  from the world's.

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 uv run --project <mixdyn checkout> \\
        --no-sync python evidence/dlm_media_coverage_per_effect.py run --out <json>
    uv run --project <mixdyn checkout> --no-sync python \\
        evidence/dlm_media_coverage_per_effect.py report <json>
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import dlm_media_coverage as failed
import jax
import numpy as np
from sbc_bayes_mmm import checkout
from scipy.stats import beta as beta_distribution
from scipy.stats import t as student

from mixdyn import NATIONAL, Cadence, Channel, MediaPanel, Revenue
from mixdyn.models import DiscountGrid, DlmSpec, Effect, PerEffect, Season, Trend, choose_dlm
from mixdyn.transforms import HillShape, TanhShape, Transform

jax.config.update("jax_enable_x64", True)

SEED, SMOKE, WORLDS, WEEKS, LENGTH, LEVEL, GATE = 2026100500, 2026100900, 200, 156, 8, 0.9, 0.88
MIXDYN = "0b349c6a55f45ddad4b8b0be0c2f8702a25f7caa"
CHANNELS = ("tv", "search")
VALUES = (0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99, 1.0)
TRENDS = (0.98, 0.99, 1.0)
GRID = DiscountGrid(trend=TRENDS, effects=PerEffect(VALUES))
SHARED_097 = DiscountGrid(trend=TRENDS, effects=(0.97, 0.99, 1.0))
SHARED = DiscountGrid(trend=TRENDS, effects=VALUES)
GRIDS = {"each": GRID, "shared 0.97": SHARED_097, "shared": SHARED}
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


def world(i: int, base: int = SEED) -> tuple[MediaPanel, dict[str, dict[str, Any]]]:
    """``dlm_media_coverage.world``, from the stream ``base``."""
    rng = np.random.default_rng(base + i)
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


def same_law() -> None:
    """The copy's worlds are the failed check's, bit for bit, on its stream."""
    for i in (0, 1, WORLDS - 1):
        ours, theirs = world(i, failed.SEED), failed.world(i)
        assert np.array_equal(ours[0].kpi, theirs[0].kpi)
        assert np.array_equal(ours[0].spend, theirs[0].spend)
        assert ours[1] == theirs[1]
    assert (WEEKS, LENGTH, LEVEL, GATE, CHANNELS) == (
        failed.WEEKS,
        failed.LENGTH,
        failed.LEVEL,
        failed.GATE,
        failed.CHANNELS,
    )


def transform(truth: dict[str, Any]) -> Transform:
    shape = HillShape(truth["slope"]) if truth["family"] == "hill" else TanhShape()
    return Transform(truth["retention"], truth["half"], shape, LENGTH)


def fitted(
    panel: MediaPanel, truth: dict[str, dict[str, Any]], spec: DlmSpec, grid: DiscountGrid
) -> dict[str, Any]:
    """One grid's choice on the world: each channel's projected and best-spec intervals, the kept
    set's size, the best discounts, the ends named, and the columns' largest error over their
    largest value."""
    choice = choose_dlm(panel, NATIONAL, spec, grid)
    projected = choice.coefficient_bounds(LEVEL)
    best = choice.fit.filtered
    half = student.ppf(0.5 + LEVEL / 2.0, best.dof[-1]) * np.sqrt(np.diag(best.covariance[-1]))
    record: dict[str, Any] = {
        "kept": len(choice.kept),
        "best": {
            "trend": choice.fit.spec.trend.discount if choice.fit.spec.trend else None,
            **{e.column: e.discount for e in choice.fit.spec.effects},
        },
        "edges": [[e.parameter, e.end, e.best] for e in choice.edges],
    }
    errors = []
    for j, c in enumerate(CHANNELS):
        mean = float(best.mean[-1, 1 + j])
        record[c] = {
            "projected": [projected[c].lower, projected[c].upper],
            "pick": [mean - float(half[1 + j]), mean + float(half[1 + j])],
        }
        ours = column(panel.spend[0, :, j], truth[c])
        errors.append(float(np.max(np.abs(choice.fit.x[:, j] - ours)) / np.max(ours)))
    record["column_error"] = max(errors)
    return record


def validated(i: int, base: int = SEED) -> dict[str, Any]:
    panel, truth = world(i, base)
    effects = tuple(Effect(c, 0.99, transform=transform(truth[c])) for c in CHANNELS)
    spec = DlmSpec(effects, Trend(1, 0.99), SEASON)
    record: dict[str, Any] = {"world": i, "truth": truth}
    for name, grid in GRIDS.items():
        record[name] = fitted(panel, truth, spec, grid)
    record["column_error"] = max(record[name]["column_error"] for name in GRIDS)
    return record


def smoke() -> None:
    """Two worlds of the smoke stream: the kept sets' sizes, the columns' error and the time."""
    for i in range(2):
        start = time.perf_counter()
        record = validated(i, SMOKE)
        kept = {name: record[name]["kept"] for name in GRIDS}
        print(
            f"smoke world {i}: kept {kept}, columns' error {record['column_error']:.1e}, "
            f"{time.perf_counter() - start:.1f} s"
        )


def run(out: Path) -> None:
    """Every world not yet in ``out``'s ``.jsonl``, then ``out``; on a clean checkout of
    ``MIXDYN`` alone."""
    same_law()
    source = checkout()
    if source["commit"] != MIXDYN or source["dirty"]:
        msg = f"the check runs on a clean checkout of mixdyn {MIXDYN}, not {source}"
        raise SystemExit(msg)
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


def holds(worlds: list[dict[str, Any]], grid: str, channel: str, which: str) -> np.ndarray:
    truth = np.array([w["truth"][channel]["at_last"] for w in worlds])
    ends = np.array([w[grid][channel][which] for w in worlds])
    return (ends[:, 0] <= truth) & (truth <= ends[:, 1])


def width(worlds: list[dict[str, Any]], grid: str, channel: str, which: str) -> np.ndarray:
    return np.array([w[grid][channel][which][1] - w[grid][channel][which][0] for w in worlds])


def report(record: dict[str, Any]) -> None:
    worlds = record["worlds"]
    n = len(worlds)
    commit = record["mixdyn"]["commit"][:7]
    dirty = ", DIRTY" if record["mixdyn"]["dirty"] else ""
    print(f"{n} worlds on mixdyn {commit}{dirty}; {record['level']:.0%} intervals")
    verdicts = []
    for c in CHANNELS:
        held = holds(worlds, "each", c, "projected")
        low, high = clopper_pearson(int(held.sum()), n)
        verdicts.append(held.mean() >= GATE)
        pick = holds(worlds, "each", c, "pick")
        ours = width(worlds, "each", c, "projected")
        own = width(worlds, "each", c, "pick")
        print(
            f"  {c:6s} projected holds {held.sum():3d} of {n}, {held.mean():.3f} "
            f"[{low:.3f}, {high:.3f}]; the best spec's own {pick.mean():.3f}; "
            f"width over the best's, median {np.median(ours / own):.3f}"
        )
        for other in ("shared 0.97", "shared"):
            theirs = holds(worlds, other, c, "projected")
            ratio = np.median(ours / width(worlds, other, c, "projected"))
            print(
                f"    beside {other}: it holds {theirs.mean():.3f}; this grid's width over its, "
                f"median {ratio:.3f}"
            )
        family = np.array([w["truth"][c]["family"] for w in worlds])
        for name in ("tanh", "hill"):
            print(f"    {name}: {held[family == name].mean():.3f} of {(family == name).sum()}")
        sigma = np.array([w["truth"][c]["sigma"] for w in worlds])
        tercile = np.searchsorted(np.quantile(sigma, [1 / 3, 2 / 3]), sigma)
        shares = ", ".join(f"{held[tercile == k].mean():.3f}" for k in range(3))
        print(f"    by tercile of sigma, low to high: {shares}")
        best = [w["each"]["best"][c] for w in worlds]
        print(
            "    best discount: " + ", ".join(f"{d}: {best.count(d)}" for d in VALUES if d in best)
        )
        at_end = np.array(
            [any(e[0] == f"{c} discount" for e in w["each"]["edges"]) for w in worlds]
        )
        if at_end.any():
            print(
                f"    its discount's end named in {at_end.sum()} worlds, holding "
                f"{held[at_end].mean():.3f}; {held[~at_end].mean():.3f} in the rest"
            )
    trend = [w["each"]["best"]["trend"] for w in worlds]
    print("  best trend discount: " + ", ".join(f"{d}: {trend.count(d)}" for d in TRENDS))
    named: dict[tuple[str, float], list[int]] = {}
    for w in worlds:
        for parameter, end, best in w["each"]["edges"]:
            named.setdefault((parameter, end), [0, 0])
            named[parameter, end][0] += 1
            named[parameter, end][1] += bool(best)
    for (parameter, end), (count, best) in sorted(named.items()):
        print(f"  end named: {parameter} at {end:g} in {count} worlds, the best there in {best}")
    if not named:
        print("  no end of the grid named in any world")
    print(
        f"  the model's columns against the world's, largest error over the largest: "
        f"{max(w['column_error'] for w in worlds):.1e}"
    )
    kept = np.array([w["each"]["kept"] for w in worlds])
    size = len(GRID.specs(DlmSpec(tuple(Effect(c, 1.0) for c in CHANNELS), Trend(1, 1.0))))
    print(
        f"  kept specs: median {np.median(kept):.0f}, from {kept.min()} to {kept.max()} of {size}"
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
    sub.add_parser("smoke")
    args = parser.parse_args()
    if args.command == "report":
        report(json.loads(args.record.read_text()))
        return
    if args.command == "smoke":
        smoke()
        return
    run(args.out)


if __name__ == "__main__":
    main()
