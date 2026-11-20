"""Why the scorecard's chc arm reads returns several times the truth on Track M v2's worlds and the
drift family's (mixdyn ADR 0041, amended 2026-11-07).

Each world's export is cut as the bench's refresh swing cuts it for its first fit
(`causaldyn_bench.scorecard.refresh.windows`), and its returns are read on the year that ends at
the cut. The arm's model (`scorecard.arms.chc_arm`) is fitted five ways, each read at the filter's
posterior mean, unsigned, and where the arm itself fits, also at the mean of its sign-held draws:

* `arm`: as the arm fits the export, its curves chosen by its search over its grid;
* `own curves`: each channel at the world's own kernel and curve, the exported controls;
* `own curves, true baseline`: the same, the world's baseline and promotional uplift the
  controls; only the noise is left out, so this reads the arm's arithmetic from a fit to a return;
* `arm, true baseline`: the arm's own fit with those controls;
* `arm, half-saturation grid`: the arm's own fit, the exported controls, its grid's scales read
  as the DLM arm reads its numbers, as half-saturation points: each divided by `atanh(1/2)`.

Beside each, how flat each channel's column is over the history: its coefficient of variation
after the kernel's first weeks. And two checks of CHC as the arm calls it: the filter's posterior
mean against the closed form of the conjugate regression, and the sign-held draws' mean against
rejection sampling of the same truncated posterior.

`scan` reads a refresh cell's records of the chc arm (`scorecard.refresh`, `--root`'s `records/`):
each channel's column's flatness against its mean return over the truth.

    PYTHONPATH=<bench>/src:. JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \\
        uv run --project <mixdyn> --no-sync python evidence/bench/chc_arm_returns.py \\
        run --out <json> [--scan <refresh root>]
    PYTHONPATH=<bench>/src:. uv run --project <mixdyn> --no-sync \\
        python evidence/bench/chc_arm_returns.py report <json>
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from causaldyn_bench.scorecard import refresh
from causaldyn_bench.scorecard.returns import drawn_returns, true_returns
from chc.dlm import _structure, forward_filter

from scorecard.arms import chc_arm
from scorecard.layout import FAMILIES, slug
from scorecard.provenance import provenance

WORLDS = ("0:drawn:900", "0:reference:900", "1:static:220500", "1:walk:220000")
FLAT = 0.05
"""A column's coefficient of variation below which the scan calls it flat."""
PROPOSALS, BATCH = 40_000_000, 200_000


def _columns(observation: Any, transforms: tuple[tuple[float, float], ...]) -> list[np.ndarray]:
    """Each channel's column at `transforms`, each a retention and a scale in spend."""
    spend = np.asarray(observation.spend, dtype=float)
    return [
        np.asarray(chc_arm._channel(r, s, 1.0, observation.kernel_length)(spend[:, j]))
        for j, (r, s) in enumerate(transforms)
    ]


def _flatness(observation: Any, transforms: tuple[tuple[float, float], ...]) -> list[float]:
    return [
        float(np.std(c[observation.kernel_length :]) / np.mean(c[observation.kernel_length :]))
        for c in _columns(observation, transforms)
    ]


def _returns(
    observation: Any, transforms: tuple[tuple[float, float], ...], coefficients: np.ndarray
) -> np.ndarray:
    """Each channel's ROI on the observation's window at one coefficient a channel."""
    response = {
        "kernel": "GeometricAdstock",
        "length": observation.kernel_length,
        "normalized": True,
        "curve": "Tanh",
        "parameters": {
            name: {"retention": [r], "scale": [s], "coefficient": [float(coefficients[j])]}
            for j, (name, (r, s)) in enumerate(zip(observation.channels, transforms, strict=True))
        },
        "decomposition": {},
    }
    return drawn_returns(response, observation).roi[0]


def _unsigned(observation: Any, transforms: tuple[tuple[float, float], ...]) -> dict[str, Any]:
    """The arm's model at `transforms` with the observation's controls, read at the filter's
    posterior mean; the closed form's largest gap over the regression's coordinates, relative."""
    sales = np.asarray(observation.sales, dtype=float)
    controls = [np.asarray(observation.controls[c], dtype=float) for c in observation.controls]
    x = np.column_stack(_columns(observation, transforms) + controls)
    count = len(transforms)
    model = chc_arm._model(sales, x, count)
    fit = forward_filter(model, sales, x)
    first = sum(b.size for b in model.blocks[: chc_arm.LEADING])
    coefficients = fit.mean[-1, first : first + count]
    # the filter evolves before each step: y_t = F_t' G^(t+1) theta_0, theta_0's prior at V / S
    structure = _structure(model)
    evolution, regression = structure.evolution, structure.regression
    rows, power = np.empty((sales.size, model.size)), evolution.copy()
    for t in range(sales.size):
        f = structure.design.copy()
        f[regression] = x[t]
        rows[t], power = f @ power, evolution @ power
    prior = model.prior.covariance / model.prior.scale
    theta = np.linalg.solve(
        rows.T @ rows + np.linalg.inv(prior),
        rows.T @ sales + np.linalg.solve(prior, model.prior.mean),
    )
    closed = (np.linalg.matrix_power(evolution, sales.size) @ theta)[regression]
    gap = float(np.max(np.abs(fit.mean[-1, regression] - closed)) / np.max(np.abs(closed)))
    return {
        "roi": _returns(observation, transforms, coefficients).tolist(),
        "coefficient": coefficients.tolist(),
        "closed form gap": gap,
        "fit": fit,
        "first": first,
    }


def _rejection(fit: Any, first: int, count: int, seed: int) -> dict[str, Any]:
    """The channels' coefficients' mean under the filter's posterior truncated to at or above 0,
    by rejection: the variance from its inverse gamma, the state from its normal given it."""
    m, c, n, s = fit.mean[-1], fit.covariance[-1], fit.dof[-1], fit.scale[-1]
    rng, root = np.random.default_rng(seed), np.linalg.cholesky(c / s)
    kept, proposed = [], 0
    while sum(k.shape[0] for k in kept) < BATCH and proposed < PROPOSALS:
        variance = 1.0 / rng.gamma(n / 2.0, 2.0 / (n * s), size=BATCH)
        states = m + (rng.standard_normal((BATCH, m.size)) @ root.T) * np.sqrt(variance)[:, None]
        held = states[:, first : first + count]
        kept.append(held[np.all(held >= 0.0, axis=1)])
        proposed += BATCH
    drawn = np.concatenate(kept)
    return {
        "mean": drawn.mean(axis=0).tolist(),
        "sd": drawn.std(axis=0).tolist(),
        "accepted": int(drawn.shape[0]),
        "proposed": proposed,
    }


def _arm(observation: Any, grid: chc_arm.Grid = chc_arm.GRID) -> dict[str, Any]:
    """The arm's own fit on `grid`: its choice, its columns' flatness, its returns unsigned and
    signed, and its sign-held coefficients against rejection sampling."""
    kept, chc_arm.GRID = chc_arm.GRID, grid
    try:
        fitted = chc_arm.Chc().fit(observation, smoke=False)
    finally:
        chc_arm.GRID = kept
    means = chc_arm._means(observation)
    multiples = tuple(
        (fitted.flags[f"{name} retention"], fitted.flags[f"{name} scale"])
        for name in observation.channels
    )
    transforms = tuple((r, m * means[j]) for j, (r, m) in enumerate(multiples))
    unsigned = _unsigned(observation, transforms)
    held = np.array([[float(c.coefficient) for c in reading] for reading in fitted.readings])
    check = _rejection(unsigned.pop("fit"), unsigned.pop("first"), len(transforms), 20261107)
    # the readings are evenly spaced over the pooled draws: no more independent than either count
    se = np.asarray(check["sd"]) / np.sqrt(min(fitted.ess, held.shape[0]))
    return {
        "choice": [list(m) for m in multiples],
        "flatness": _flatness(observation, transforms),
        "unsigned": unsigned,
        "signed roi": drawn_returns(fitted.response, observation).roi.mean(axis=0).tolist(),
        "signed coefficient": held.mean(axis=0).tolist(),
        "rejection": check,
        "signed gap over se": ((held.mean(axis=0) - np.asarray(check["mean"])) / se).tolist(),
        "ess": fitted.ess,
    }


def measured(family: int, environment: str, seed: int) -> dict[str, Any]:
    """One world's first cut, read the four ways."""
    world = FAMILIES[family].world(environment, seed)
    observation = refresh.windows(FAMILIES[family].observe(world, 0))[0]
    history, weeks = world.history, observation.sales.shape[0]
    truth = true_returns(history, observation.roi_window)
    own = tuple((history.retention[c], 2.0 / history.saturation[c]) for c in range(3))
    blank = np.zeros(observation.planned)
    oracle = dataclasses.replace(
        observation,
        controls={
            "uplift": np.asarray(history.promotion[:weeks], dtype=float),
            "baseline": np.asarray(history.baseline[:weeks], dtype=float),
        },
        future_controls={"uplift": blank, "baseline": blank},
    )
    readings = {}
    for name, read in (("own curves", observation), ("own curves, true baseline", oracle)):
        made = _unsigned(read, own)
        made.pop("fit")
        made.pop("first")
        readings[name] = {"flatness": _flatness(read, own), "unsigned": made}
    readings["arm"] = _arm(observation)
    readings["arm, true baseline"] = _arm(oracle)
    halves = tuple(s / math.atanh(0.5) for s in chc_arm.GRID.scales)
    readings["arm, half-saturation grid"] = _arm(
        observation, chc_arm.Grid(chc_arm.GRID.retentions, halves)
    )
    means = chc_arm._means(observation)
    return {
        "family": family,
        "environment": environment,
        "seed": seed,
        "weeks": weeks,
        "window": list(observation.roi_window),
        "channels": list(observation.channels),
        "truth": truth.roi.tolist(),
        "own curves": [[r, s / means[j]] for j, (r, s) in enumerate(own)],
        "readings": readings,
    }


def scan(root: Path) -> list[dict[str, Any]]:
    """Every swing world's chc records under `root`: each channel's flatness and return."""
    rows = []
    for place in sorted((root / "records" / "swing").glob("family-*/*/k0/*")):
        family = int(place.parts[-4].removeprefix("family-"))
        number, seed = FAMILIES[family], int(place.parts[-1])
        environment = next(e for e in number.pilots if slug(e) == place.parts[-3])
        world = number.world(environment, seed)
        cuts = refresh.windows(number.observe(world, 0))
        truth = true_returns(world.history, cuts[0].roi_window).roi
        for i, observation in enumerate(cuts):
            record = json.loads((place / "chc" / f"{i}.json").read_text())
            means = chc_arm._means(observation)
            transforms = tuple(
                (record["flags"][f"{n} retention"], record["flags"][f"{n} scale"] * means[j])
                for j, n in enumerate(observation.channels)
            )
            roi = drawn_returns(record["response"], observation).roi.mean(axis=0)
            for j, name in enumerate(observation.channels):
                rows.append(
                    {
                        "world": f"{family}/{environment}/{seed}",
                        "cut": i,
                        "channel": name,
                        "flatness": _flatness(observation, transforms)[j],
                        "at an end": bool(
                            record["flags"][f"{name} retention at an end"]
                            or record["flags"][f"{name} scale at an end"]
                        ),
                        "over truth": float(roi[j] / truth[j]),
                    }
                )
    return rows


def run(out: Path, worlds: list[str], root: Path | None) -> None:
    """Each world's reading and the scan, written to `out` with the provenance."""
    data: dict[str, Any] = {"provenance": provenance(), "flat": FLAT, "worlds": []}
    for world in worlds:
        family, environment, seed = world.split(":")
        data["worlds"].append(measured(int(family), environment, int(seed)))
        out.write_text(json.dumps(data, indent=1))
    if root is not None:
        data["scan"] = scan(root)
        out.write_text(json.dumps(data, indent=1))


def _row(values: list[float]) -> str:
    return ", ".join(f"{v:.2f}" for v in values)


def report(path: Path) -> None:
    """The numbers ADR 0041's amendment quotes."""
    data = json.loads(path.read_text())
    for w in data["worlds"]:
        print(f"{w['family']}/{w['environment']}/{w['seed']}: truth {_row(w['truth'])}")
        for name, r in w["readings"].items():
            line = f"  {name}: unsigned {_row(r['unsigned']['roi'])}"
            if "signed roi" in r:
                line += (
                    f"; signed {_row(r['signed roi'])}; choice {r['choice']}; signed against"
                    f" rejection {_row(r['signed gap over se'])} se"
                )
            line += f"; flatness {_row(r['flatness'])}"
            line += f"; closed form {r['unsigned']['closed form gap']:.1e}"
            print(line)
    rows = data.get("scan", [])
    if rows:
        flat = [r for r in rows if r["flatness"] < data["flat"]]
        rest = [r for r in rows if r["flatness"] >= data["flat"]]
        high = [r for r in rows if r["over truth"] > 3.0]
        print(
            f"scan: {len(rows)} channel-fits; flat {len(flat)}, every one above 3x:"
            f" {all(r['over truth'] > 3.0 for r in flat)}, median"
            f" {np.median([r['over truth'] for r in flat]):.2f}x, at an end"
            f" {sum(r['at an end'] for r in flat)}; the rest median"
            f" {np.median([r['over truth'] for r in rest]):.2f}x, above 3x"
            f" {sum(r['over truth'] > 3.0 for r in rest)}; above 3x in all {len(high)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("path", nargs="?", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--world", action="append", default=None)
    parser.add_argument("--scan", type=Path, default=None)
    args = parser.parse_args()
    if args.command == "run":
        run(args.out, args.world or list(WORLDS), args.scan)
    else:
        report(args.path)


if __name__ == "__main__":
    main()
