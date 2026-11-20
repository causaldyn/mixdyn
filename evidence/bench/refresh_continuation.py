"""A refresh against the full refit, on Track M v2's worlds continued by their quarter (mixdyn ADR
0079).

Each world's export at rung 0 is continued by its shadow quarter: the quarter's sales, its spend at
the status quo, and its promotion and price as observed. The one-geo model is fitted as
`scorecard.arms.bayes` fits its tanh family, without the lift tests, four chains of 1000 draws
after 1000 of warm-up, four times: `previous` on the history alone; `full` on the history and the
quarter; `again`, `full` from another seed; and `refresh_bayes(previous)` on the history and the
quarter at the default weight. Read for each channel: the refresh's mean log return less the full
refit's, over the full refit's SD, and the same of `again`, which is the sampler's own spread; the
ratio of the two fits' SDs of the log return; the same gap of the return on the previous window's
spend; the tempered prior; and the refresh's drift.

    PYTHONPATH=<bench>/src:. JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \\
        uv run --project <mixdyn> --no-sync python evidence/bench/refresh_continuation.py \\
        run --out <json> [--world drawn:900 --world reference:900]
    PYTHONPATH=<bench>/src:. uv run --project <mixdyn> --no-sync \\
        python evidence/bench/refresh_continuation.py report <json>
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
from causaldyn_bench.scorecard.track_m2 import TRACK_M2

from mixdyn import NATIONAL
from mixdyn.models.bayes_mmm import (
    WEIGHT,
    BayesMedia,
    BayesSpec,
    TanhCurve,
    fit_bayes,
    refresh_bayes,
    returns_on,
)
from mixdyn.models.ridge_mmm import Control, FourierSeason, LinearTrend
from scorecard.arms import panel as reading
from scorecard.arms.bayes import HARMONICS, KNOTS, POSITIVE, YEAR
from scorecard.provenance import provenance

SAMPLER = {"chains": 4, "warmup": 1000, "draws": 1000}
SEEDS = {"previous": 11, "full": 12, "again": 13, "refresh": 12}


def continued(environment: str, seed: int) -> tuple[Any, int]:
    """The world's export at rung 0 with its quarter after it, as a panel, and the history's
    weeks."""
    world = TRACK_M2.world(environment, seed)
    observation = TRACK_M2.observe(world, 0)
    shadow = world.shadow
    observed = {
        "promotion": shadow.observed_promotion.astype(float),
        "price": shadow.observed_price,
    }
    if set(observation.controls) != set(observed):
        msg = f"the export's controls are {sorted(observation.controls)}, not {sorted(observed)}"
        raise ValueError(msg)
    quarter = dataclasses.replace(
        observation,
        sales=np.concatenate([observation.sales, shadow.sales]),
        spend=np.concatenate(
            [observation.spend, np.tile(observation.status_quo, (observation.planned, 1))]
        ),
        controls={
            name: np.concatenate([series, observed[name]])
            for name, series in observation.controls.items()
        },
    )
    return reading.panel(quarter), observation.sales.shape[0]


def measured(environment: str, seed: int) -> dict[str, Any]:
    """One world's fits and the reading of each channel."""
    panel, weeks = continued(environment, seed)
    spec = BayesSpec(
        media=tuple(BayesMedia(c) for c in panel.channels),
        controls=tuple(
            Control(c, lower=0.0) if c in POSITIVE else Control(c) for c in panel.controls
        ),
        trend=LinearTrend(KNOTS),
        season=FourierSeason(YEAR, HARMONICS),
        saturation=TanhCurve(),
    )
    window = (panel.dates[0], panel.dates[weeks - 1])
    previous = fit_bayes(panel, NATIONAL, spec, seed=SEEDS["previous"], window=window, **SAMPLER)
    full = fit_bayes(panel, NATIONAL, spec, seed=SEEDS["full"], **SAMPLER)
    again = fit_bayes(panel, NATIONAL, spec, seed=SEEDS["again"], **SAMPLER)
    fresh = refresh_bayes(panel, NATIONAL, previous, seed=SEEDS["refresh"], **SAMPLER)
    on_full = returns_on(full, panel, NATIONAL, previous.dates)
    on_fresh = returns_on(fresh.fit, panel, NATIONAL, previous.dates)
    channels = {}
    for j, media in enumerate(fresh.fit.spec.media):
        log_full = np.log(full.draws["roi"][..., j]).ravel()
        log_fresh = np.log(fresh.fit.draws["roi"][..., j]).ravel()
        log_again = np.log(again.draws["roi"][..., j]).ravel()
        sd = float(np.std(log_full))
        drift = fresh.drift[j]
        channels[media.channel] = {
            "tempered prior": [media.roi.mu, media.roi.sigma],
            "full mean return": float(np.mean(full.draws["roi"][..., j])),
            "refresh mean return": float(np.mean(fresh.fit.draws["roi"][..., j])),
            "gap": float((np.mean(log_fresh) - np.mean(log_full)) / sd),
            "seed to seed": float((np.mean(log_again) - np.mean(log_full)) / sd),
            "sd ratio": float(np.std(log_fresh) / sd),
            "gap on the previous window": float(
                (np.mean(on_fresh[:, j]) - np.mean(on_full[:, j])) / np.std(on_full[:, j])
            ),
            "drift": dataclasses.asdict(drift) | {"leaves": drift.leaves},
        }
    worst = max(
        float(np.max(fit.diagnostics["roi"].rhat)) for fit in (previous, full, again, fresh.fit)
    )
    return {"environment": environment, "seed": seed, "weeks": weeks, "most rhat": worst} | {
        "channels": channels
    }


def run(out: Path, worlds: list[str]) -> None:
    """Each world's reading, written to `out` with the provenance."""
    source = provenance()
    rows = []
    for world in worlds:
        environment, seed = world.split(":")
        rows.append(measured(environment, int(seed)))
        out.write_text(
            json.dumps({"provenance": source, "weight": WEIGHT, "worlds": rows}, indent=1)
        )


def report(path: Path) -> None:
    """The numbers ADR 0079 quotes."""
    data = json.loads(path.read_text())
    print(f"weight {data['weight']}")
    for row in data["worlds"]:
        print(f"{row['environment']}/{row['seed']}: most R-hat {row['most rhat']:.3f}")
        for name, c in row["channels"].items():
            print(
                f"  {name}: gap {c['gap']:+.2f} sd, seed to seed {c['seed to seed']:+.2f},"
                f" sd ratio {c['sd ratio']:.2f}, on the previous window"
                f" {c['gap on the previous window']:+.2f}, leaves {c['drift']['leaves']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("path", nargs="?", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--world", action="append", default=None)
    args = parser.parse_args()
    if args.command == "run":
        run(args.out, args.world or ["drawn:900", "reference:900"])
    else:
        report(args.path)


if __name__ == "__main__":
    main()
