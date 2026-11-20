"""The twin conflation worlds (ADR 0073): two worlds of one history that a static fit cannot tell
apart, and whose best plans differ.

104 weeks from 2023-01-02. Television's spend rises from 30 to 100 a week over the first 78
weeks, `ramp(t)`, and holds there, each week's spend that level times a log-normal of log-scale
`SPREAD`, 0.05, and mean 1; search's is drawn about 50 a week, a log-normal of log-scale 0.5. The
KPI is `BASE` plus television's response, search's, `search(x) = 252 tanh(x / 50)`, and a normal
noise of `NOISE`, 40 a week, the same draw in both twins:

- `"curve"`: television's response is fixed on an S-shaped curve, a Hill, `curve(x) = 350 x^3 /
  (x^3 + 60^3)`;
- `"line"`: television's response is a line in its spend whose slope drifts over the weeks,
  `drift(t) x`, `drift(t) = curve(ramp(t)) / ramp(t)`, the curve's average return at the level of
  the week: 1.30 at the first week, 3.09 at most where the ramp passes 76, 2.88 from week 77 on.

Along the ramp the two return the same, and they differ only by how far each week's spend is off
its level, which `SPREAD` keeps small: the conflation Dew, Padilla and Shchetkina (arXiv
2408.07678) measure. Neither world carries spend over, so a plan's worth is its weeks' responses.

A plan spends `BUDGET`, 150 a week, over `PERIODS`, 13 weeks after the history, television
within `TV_RANGE` and search the rest, within `SEARCH_RANGE`. Its worth in the line's world reads
the drift where the history left it, `drift(WEEKS - 1)`, as the line's own forecast does. Each
world's best plan is a one-dimensional concave maximum, found to `1e-9` of a euro: television at
110.81 a week in the line's world and 94.69 in the curve's; the line's best plan costs the
curve's world 0.095 a euro of the budget, and the curve's costs the line's 0.061.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
from scipy import optimize

from mixdyn import Cadence, MediaPanel, Revenue, read_panel

type World = Literal["curve", "line"]

WEEKS = 104
RISE = 78
"""The weeks television's spend rises over; it holds at its top for the rest."""
SPREAD = 0.05
NOISE = 40.0
BASE = 1000.0
BUDGET = 150.0
PERIODS = 13
TV_RANGE = (50.0, 140.0)
SEARCH_RANGE = (10.0, 100.0)
HEIGHT, HALF, SLOPE = 350.0, 60.0, 3.0
"""Television's curve: its ceiling a week, its half-saturation point and its Hill slope."""
SEARCH_HEIGHT, SEARCH_SCALE = 252.0, 50.0
DATES = np.datetime64("2023-01-02") + np.timedelta64(7, "D") * np.arange(WEEKS)


def ramp(t: np.ndarray) -> np.ndarray:
    """Television's level a week at the weeks `t`, counted from 0."""
    return 30.0 + 70.0 * np.minimum(np.asarray(t, dtype=float) / (RISE - 1), 1.0)


def curve(x: np.ndarray | float) -> np.ndarray:
    """Television's response in the curve's world, a week."""
    x = np.asarray(x, dtype=float)
    return HEIGHT * x**SLOPE / (x**SLOPE + HALF**SLOPE)


def drift(t: np.ndarray | int) -> np.ndarray:
    """The line's slope at the weeks `t`: the curve's average return at the week's level."""
    level = ramp(np.asarray(t))
    return curve(level) / level


def search(x: np.ndarray | float) -> np.ndarray:
    """Search's response in both worlds, a week."""
    return SEARCH_HEIGHT * np.tanh(np.asarray(x, dtype=float) / SEARCH_SCALE)


def spend(seed: int) -> tuple[np.ndarray, np.ndarray, np.random.Generator]:
    """Television's and search's spend over the history, and the generator the KPI's noise is drawn
    from after them."""
    rng = np.random.default_rng(seed)
    level = ramp(np.arange(WEEKS))
    tv = level * np.exp(SPREAD * rng.standard_normal(WEEKS) - SPREAD**2 / 2.0)
    paid = 50.0 * np.exp(0.5 * rng.standard_normal(WEEKS) - 0.125)
    return tv, paid, rng


def twins(seed: int) -> dict[World, MediaPanel]:
    """Both worlds' histories at `seed`: the same spend, the same noise."""
    tv, paid, rng = spend(seed)
    noise = NOISE * rng.standard_normal(WEEKS)
    responses: dict[World, np.ndarray] = {
        "curve": curve(tv),
        "line": drift(np.arange(WEEKS)) * tv,
    }
    return {
        world: read_panel(
            {
                "date": DATES,
                "sales": BASE + tv_response + search(paid) + noise,
                "tv": tv,
                "search": paid,
            },
            date="date",
            kpi="sales",
            spend={"tv": "tv", "search": "search"},
            cadence=Cadence.WEEK,
            kpi_unit=Revenue(),
            currency="EUR",
        )
        for world, tv_response in responses.items()
    }


def worth(world: World, tv: float) -> float:
    """A plan's worth over its weeks in `world`, television at `tv` a week and search the rest."""
    response = curve(tv) if world == "curve" else drift(WEEKS - 1) * tv
    return float(PERIODS * (response + search(BUDGET - tv)))


def best_split(worth: Callable[[float], float]) -> float:
    """Television's spend a week that makes `worth`, a concave function of it, most, search
    spending the rest of the budget, each within its range."""
    low = max(TV_RANGE[0], BUDGET - SEARCH_RANGE[1])
    high = min(TV_RANGE[1], BUDGET - SEARCH_RANGE[0])
    found = optimize.minimize_scalar(
        lambda tv: -worth(tv), bounds=(low, high), method="bounded", options={"xatol": 1e-9}
    )
    return float(found.x)


def best(world: World) -> float:
    """Television's spend a week in `world`'s best plan."""
    return best_split(lambda tv: worth(world, tv))


def regret(world: World, tv: float) -> float:
    """What a plan, television at `tv` a week, leaves of `world`'s best, a euro of its budget."""
    return (worth(world, best(world)) - worth(world, tv)) / (BUDGET * PERIODS)
