"""Does a channel's return interval hold the world's once its recurring events are read off the
panel's noisy records of them, and not given? Intervals and coverage only, no wall time.

Each world is 156 weekly periods from 2023-01-02 of

    y = 1000 + s(w) + 0.8 x + 250 clearance + 400 spring + 600 cyber + N(0, 25^2),

`s(w) = 100 sin(2 pi w / 52) + 40 cos(4 pi w / 52)` for the week `w` counted from 1, a yearly
cycle. Each event recurs in the same weeks of every year: the clearance in weeks 2 to 4, the spring
sale in week 16 and the cyber week in weeks 46 and 47, as a retailer's promotions do. The channel's
spend `x` is 100 a week plus normal noise of SD 20, and 250 plus the same noise in each event's
week: spend that rises with the events, as a planner's does. Its return on any window's spend is
0.8, its coefficient: the channel has no carryover and no curve.

The panel records the events as a retailer's data does, with noise (the bench's measurement
layer): a promotion flag that is on in an event's week with probability 0.7 and in any other week
with probability 0.15, and a price that is 1 less the event's discount, 0.4 in the clearance and
the cyber week and 0.3 in the spring sale, plus normal noise of SD 0.08. The cyber week lifts the
KPI most at the clearance's discount, so neither the flag's one coefficient nor the price's one
slope fits every event.

Each world is fitted by the DLM, a level of discount 0.98, a season of period 52 with 2 harmonics
held static, and the channel's coefficient static, twice:

    level    the level and the season carry the events, the flag and the price two controls
    learned  the events `recurring_events` reads off the flag and the price, each a static
             coefficient, in place of the flag; the price a control

The return over the 156 weeks is read off 1000 draws of the fit's decomposition (`decompose_dlm`),
the channel's contribution over the spend, and its central 90 % and 95 % intervals are read. The
coverage is the share of worlds whose interval holds 0.8, with its Monte Carlo standard error.
Beside it, how often the events read are the world's, every week of each and no other.

ADR 0085's measurement, worlds 0 to 199:

    python scripts/study_recurring_events.py --worlds 200
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
    "clearance": ((2, 3, 4), 250.0, 0.4),
    "spring": ((16,), 400.0, 0.3),
    "cyber": ((46, 47), 600.0, 0.4),
}
RECALL, FALSE_POSITIVE, PRICE_SD = 0.7, 0.15, 0.08
LEVELS = (0.9, 0.95)


def world(seed: int) -> MediaPanel:
    """The world's panel; see the module."""
    from mixdyn import NATIONAL, Cadence, Channel, MediaPanel, Revenue

    rng = np.random.default_rng([20261116, 81, seed])
    dates = START + np.timedelta64(7, "D") * np.arange(WEEKS)
    week = np.arange(1, WEEKS + 1)
    of_year = (week - 1) % 52 + 1
    season = 100.0 * np.sin(2 * np.pi * week / 52) + 40.0 * np.cos(4 * np.pi * week / 52)
    lift, discount = np.zeros(WEEKS), np.zeros(WEEKS)
    for weeks, size, cut in EVENTS.values():
        inside = np.isin(of_year, weeks)
        lift[inside] = size
        discount[inside] = cut
    held = discount > 0.0
    spend = np.where(held, EVENT_SPEND, SPEND) + rng.normal(0.0, SPEND_SD, WEEKS)
    kpi = 1000.0 + season + RETURN * spend + lift + rng.normal(0.0, KPI_SD, WEEKS)
    caught = rng.random(WEEKS)
    flag = np.where(held, caught < RECALL, caught < FALSE_POSITIVE).astype(float)
    price = 1.0 - discount + rng.normal(0.0, PRICE_SD, WEEKS)
    return MediaPanel(
        dates=dates,
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=(Channel("tv"),),
        controls=("promotion", "price"),
        kpi=kpi[None],
        kpi_unit=Revenue(),
        spend=spend[None, :, None],
        currency="EUR",
        control_values=np.column_stack([flag, price])[None],
        filled=np.zeros((1, WEEKS), dtype=bool),
    )


def read(panel: MediaPanel, seed: int) -> tuple[bool, dict[str, list[tuple[float, float]]]]:
    """Whether the events read are the world's, and each model's intervals of the channel's
    return, one a level of `LEVELS`."""
    from dataclasses import replace

    from mixdyn import NATIONAL
    from mixdyn.data import recurring_events
    from mixdyn.models import DlmSpec, Effect, Season, Trend, decompose_dlm, fit_dlm

    found = recurring_events(panel, {"promotion": 1, "price": -1})
    learned = replace(panel, events=found.days(panel.dates))
    exact = sorted(found.weeks) == sorted(weeks for weeks, _, _ in EVENTS.values())
    out: dict[str, list[tuple[float, float]]] = {}
    for name, data, columns in (
        ("level", panel, ("promotion", "price")),
        ("learned", learned, ("price", *found.names)),
    ):
        effects = (Effect("tv", 1.0), *(Effect(c, 1.0) for c in columns))
        spec = DlmSpec(effects, Trend(1, 0.98), Season(52.0, 2, 1.0))
        fit = fit_dlm(data, NATIONAL, spec)
        draws = decompose_dlm(fit, draws=1000, seed=seed).draws.contributions["tv"]
        returns = draws.sum(axis=1) / float(panel.spend[0, :, 0].sum())
        ends = [np.quantile(returns, [(1 - q) / 2, (1 + q) / 2]) for q in LEVELS]
        out[name] = [(float(low), float(high)) for low, high in ends]
    return exact, out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worlds", type=int, default=200)
    parser.add_argument("--first", type=int, default=0)
    args = parser.parse_args()
    seeds = range(args.first, args.first + args.worlds)
    reads = [read(world(seed), seed) for seed in seeds]
    n = len(reads)
    exact = sum(found for found, _ in reads)
    print(f"events read exactly, every week of each and no other: {exact} of {n}")
    for name in ("level", "learned"):
        for k, level in enumerate(LEVELS):
            ends = np.array([r[name][k] for _, r in reads])
            held = (ends[:, 0] <= RETURN) & (ends[:, 1] >= RETURN)
            share = float(held.mean())
            print(
                f"{name:7s} {level:g} interval holds {RETURN:g} in {int(held.sum())} of {n}:"
                f" {share:.3f} (Monte Carlo SE {math.sqrt(share * (1 - share) / n):.3f}; at the"
                f" nominal, {math.sqrt(level * (1 - level) / n):.3f}); mean ends"
                f" [{ends[:, 0].mean():.3f}, {ends[:, 1].mean():.3f}]"
            )


if __name__ == "__main__":
    main()
