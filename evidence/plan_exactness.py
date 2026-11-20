"""How far short of the best split the envelope's plan falls on a pilot's posteriors, and what the
search that replaced it costs (mixdyn ADR 0034).

On the scorecard's pilot of the drift family, the stacked arm's records: each record's readings
rebuilt from the draws it kept, each channel's mean worth on the arm's 2048-point grid, and three
plans of the budget a period: the envelope's, the first box's fill, which `plan_posterior` made
before ADR 0034; the search's, `_allocated`, counting the boxes it fills; and the best split of the
means' interpolants by brute force, every channel but one on a point of its grid and that one
taking the rest, which is exact on the interpolants because the worth is linear in each cell of the
grids. Each plan's expected gain is read on the means exactly, off the grid.

    cd <mixdyn checkout> && PYTHONPATH=<bench>/src uv run --no-sync python \\
        <this> <pilot run directory> <out.jsonl>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from causaldyn_bench.scorecard.observe import read

from mixdyn import Channel
from mixdyn.decisions import posterior
from mixdyn.decisions.posterior import Readings, _envelope, _mean_worths, _rates
from mixdyn.models.bayes_mmm import _TANH_HALF

GRID = 2048


def brute(rates: np.ndarray, means: np.ndarray, target: float) -> float:
    """The most worth on the interpolants of a split of `target`, for two or three channels."""
    channels = rates.shape[0]
    best = -np.inf
    for free in range(channels):
        others = [c for c in range(channels) if c != free]
        spent = sum(np.ix_(*[rates[c] for c in others])[k] for k in range(len(others)))
        worth = sum(np.ix_(*[means[c] for c in others])[k] for k in range(len(others)))
        rest = target - spent
        inside = (rest >= rates[free, 0]) & (rest <= rates[free, -1])
        if inside.any():
            value = np.interp(rest[inside], rates[free], means[free]) + worth[inside]
            best = max(best, float(value.max()))
    return best


def on_grid(rates: np.ndarray, means: np.ndarray, spend: np.ndarray) -> float:
    return float(sum(np.interp(spend[c], rates[c], means[c]) for c in range(spend.size)))


def sag(rates: np.ndarray, means: np.ndarray) -> float:
    """How far a channel's envelope stands above its mean at the most: 0 on a concave mean."""
    hull = _envelope(rates, means)
    return float(np.max(np.interp(rates, rates[hull], means[hull]) - means))


def readings_of(record: dict, channels: tuple[str, ...]) -> Readings:
    """The readings the record's response holds: each draw's parameters, the Hill fit's draws
    first, each fit weighted by its share of the draws, as the arm's stacking kept them."""
    parameters, kinds = record["response"]["parameters"], record["response"]["curve"]
    tanh = np.array([kind == "Tanh" for kind in kinds[channels[0]]])
    for name in channels:
        assert np.array_equal(tanh, [kind == "Tanh" for kind in kinds[name]])
    order = np.argsort(tanh, kind="stable")
    tanh = tanh[order]

    def column(key: str) -> np.ndarray:
        return np.column_stack([np.array(parameters[c][key], dtype=float)[order] for c in channels])

    scale = column("scale")
    slope = (
        column("slope")
        if all("slope" in parameters[c] for c in channels)
        else np.full_like(scale, np.nan)
    )
    slope[tanh] = np.nan
    half = np.where(tanh[:, None], scale * _TANH_HALF, scale)
    fit = tanh.astype(np.intp)
    counts = np.bincount(fit, minlength=2)
    kept = counts > 0
    fit = np.cumsum(kept)[fit] - 1
    return Readings(
        channels=tuple(Channel(c) for c in channels),
        retention=column("retention"),
        half_saturation=half,
        slope=slope,
        coefficient=column("coefficient"),
        tanh=tanh,
        fit=fit,
        chain=np.zeros(tanh.size, dtype=np.intp),
        stacking=counts[kept] / tanh.size,
        pointwise=None,
        length=int(record["response"]["length"]),
    )


def main(run: Path, out: Path) -> None:
    fill = posterior._fill
    boxes = 0

    def counted(*arguments: object) -> object:
        nonlocal boxes
        boxes += 1
        return fill(*arguments)

    posterior._fill = counted
    done = set()
    if out.exists():
        done = {json.loads(line)["record"] for line in out.read_text().splitlines()}
    for path in sorted((run / "records" / "stacked").rglob("*.json")):
        name = str(path.relative_to(run))
        record = json.loads(path.read_text())
        if name in done or record.get("error") or not record.get("response"):
            continue
        family, law, rung = path.parts[-4], path.parts[-3], path.parts[-2]
        observation = read(
            run / "exports" / family / law / rung / path.name.replace(".json", ".npz")
        )
        readings = readings_of(record, observation.channels)
        history = np.asarray(observation.spend, dtype=float)
        lower = np.asarray(observation.lower, dtype=float)
        upper = np.asarray(observation.upper, dtype=float)
        periods, target = observation.planned, observation.budget / observation.planned
        rates = _rates(lower, upper, GRID)
        means = _mean_worths(history, readings, periods, rates)
        first = ((0, GRID - 1),) * rates.shape[0]
        envelope = fill(rates, means, first, target, {})[0]
        boxes = 0
        searched = posterior._allocated(rates, means, target)
        at = np.column_stack([envelope, searched, np.zeros(rates.shape[0])])
        exact = _mean_worths(history, readings, periods, at)
        row = {
            "record": name,
            "envelope_gain": float(np.sum(exact[:, 0] - exact[:, 2])),
            "searched_gain": float(np.sum(exact[:, 1] - exact[:, 2])),
            "envelope_on_grid": on_grid(rates, means, envelope),
            "searched_on_grid": on_grid(rates, means, searched),
            "best_on_grid": brute(rates, means, target) if rates.shape[0] <= 3 else None,
            "boxes": boxes,
            "sag": [sag(rates[c], means[c]) for c in range(rates.shape[0])],
            "envelope_spend": envelope.tolist(),
            "searched_spend": searched.tolist(),
        }
        with out.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps({k: row[k] for k in ("record", "boxes")}), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
