"""Does a channel's return interval hold the world's once its events are terms of their own beside
the level? Intervals and coverage only, no wall time.

Each world is 156 weekly periods from 2023-01-02 of

    y = 1000 + s(w) + 0.8 x + 250 easter + 400 black_friday + 300 christmas + N(0, 25^2),

`s(w) = 100 sin(2 pi w / 52) + 40 cos(4 pi w / 52)` for the week `w` counted from 1, a yearly
cycle. Each event is 1 in the week its day falls in, Easter Sunday, Black Friday and Christmas Day
of 2023 to 2025, and 0 elsewhere. The channel's spend `x` is 100 a week plus normal noise of SD
20, and 250 plus the same noise in each event's week: spend that rises with the events, as a
planner's does. Its return on any window's spend is 0.8, its coefficient: the channel has no
carryover and no curve.

Each world is fitted by the DLM, a level of discount 0.98, a season of period 52 with 2 harmonics
held static, and the channel's coefficient static, twice:

    level   the level and the season alone carry the events
    events  the three events are effects of their own, each a static coefficient

The return over the 156 weeks is read off 1000 draws of the fit's decomposition (`decompose_dlm`),
the channel's contribution over the spend, and its central 90 % and 95 % intervals are read. The
coverage is the share of worlds whose interval holds 0.8, with its Monte Carlo standard error.

ADR 0069's measurement, worlds 0 to 199:

    python scripts/study_events.py --worlds 200
"""

from __future__ import annotations

import argparse
import math
import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mixdyn import MediaPanel

os.environ.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "JAX_PLATFORMS": "cpu"})

WEEKS, START = 156, np.datetime64("2023-01-02")
RETURN, SPEND, SPEND_SD, EVENT_SPEND, KPI_SD = 0.8, 100.0, 20.0, 250.0, 25.0
EVENTS = {
    "easter": (["2023-04-09", "2024-03-31", "2025-04-20"], 250.0),
    "black_friday": (["2023-11-24", "2024-11-29", "2025-11-28"], 400.0),
    "christmas": (["2023-12-25", "2024-12-25", "2025-12-25"], 300.0),
}
LEVELS = (0.9, 0.95)


def world(seed: int) -> MediaPanel:
    """The world's panel; see the module."""
    from mixdyn import NATIONAL, Cadence, Channel, MediaPanel, Revenue

    rng = np.random.default_rng([20261106, 33, seed])
    dates = START + np.timedelta64(7, "D") * np.arange(WEEKS)
    events = {name: np.array(days, dtype="datetime64[D]") for name, (days, _) in EVENTS.items()}
    week = np.arange(1, WEEKS + 1)
    season = 100.0 * np.sin(2 * np.pi * week / 52) + 40.0 * np.cos(4 * np.pi * week / 52)
    lift = np.zeros(WEEKS)
    held = np.zeros(WEEKS, dtype=bool)
    for name, (_, size) in EVENTS.items():
        at = np.searchsorted(dates, events[name], side="right") - 1
        lift[at] += size
        held[at] = True
    spend = np.where(held, EVENT_SPEND, SPEND) + rng.normal(0.0, SPEND_SD, WEEKS)
    kpi = 1000.0 + season + RETURN * spend + lift + rng.normal(0.0, KPI_SD, WEEKS)
    return MediaPanel(
        dates=dates,
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
        events=events,
    )


def intervals(panel: MediaPanel, seed: int) -> dict[str, list[tuple[float, float]]]:
    """Each model's intervals of the channel's return, one a level of `LEVELS`."""
    from mixdyn import NATIONAL
    from mixdyn.models import DlmSpec, Effect, Season, Trend, decompose_dlm, fit_dlm

    out = {}
    for name, events in (("level", ()), ("events", tuple(EVENTS))):
        effects = (Effect("tv", 1.0), *(Effect(e, 1.0) for e in events))
        spec = DlmSpec(effects, Trend(1, 0.98), Season(52.0, 2, 1.0))
        fit = fit_dlm(panel, NATIONAL, spec)
        draws = decompose_dlm(fit, draws=1000, seed=seed).draws.contributions["tv"]
        returns = draws.sum(axis=1) / float(panel.spend[0, :, 0].sum())
        ends = [np.quantile(returns, [(1 - q) / 2, (1 + q) / 2]) for q in LEVELS]
        out[name] = [(float(low), float(high)) for low, high in ends]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worlds", type=int, default=200)
    parser.add_argument("--first", type=int, default=0)
    args = parser.parse_args()
    seeds = range(args.first, args.first + args.worlds)
    reads = [intervals(world(seed), seed) for seed in seeds]
    n = len(reads)
    for name in ("level", "events"):
        for k, level in enumerate(LEVELS):
            ends = np.array([r[name][k] for r in reads])
            held = (ends[:, 0] <= RETURN) & (ends[:, 1] >= RETURN)
            share = float(held.mean())
            print(
                f"{name:6s} {level:g} interval holds {RETURN:g} in {int(held.sum())} of {n}:"
                f" {share:.3f} (Monte Carlo SE {math.sqrt(share * (1 - share) / n):.3f}; at the"
                f" nominal, {math.sqrt(level * (1 - level) / n):.3f}); mean ends"
                f" [{ends[:, 0].mean():.3f}, {ends[:, 1].mean():.3f}]"
            )


if __name__ == "__main__":
    main()
