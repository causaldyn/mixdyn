"""A world whose paid search buys against demand, which a query count measures (ADR 0074).

Weekly, from 2023-01-02, in the Bayesian model's own form: each channel's spend carried over by a
normalised geometric kernel over the whole history and saturated by `tanh(atanh(1/2) a / K)`.

- The organic demand is `d0_t = exp(0.15 sin(2 pi t / 52) + z_t)`, `z` a stationary AR(1) of
  coefficient 0.8 and innovations of sd 0.08, which no column holds.
- Television is flighted: in 60 % of the weeks a Gamma(4, 25) spend, else none. It raises the
  demand, `d_t = d0_t (1 + 0.1 s_tv,t)`, and the KPI directly.
- The queries are `100 000 d_t` times a log-normal error of sd 0.02, the panel's control
  `queries`.
- Paid search's spend is `60 d_t` times a log-normal of sd 0.25: an ad shows where someone
  searches, so its spend follows the queries, television's among them.
- The KPI is `1000 d_t + 200 s_search,t + 150 s_tv,t` and normal noise of sd 20, with retention
  0.2 and half-saturation 60 for search, 0.5 and 100 for television.

So the queries confound search: its spend and the KPI both follow the demand. And they mediate
television: it raises the queries, which raise the KPI. Each channel's true return over the history
is known three ways: search's own effect over its spend; television's `direct` effect, `150
s_tv`, over its spend; and its `total`, the direct effect and the organic KPI its demand brings,
`1000 d0 0.1 s_tv`, over its spend, the search spend as it ran. The search spend television's
demand brings is credited to search here, as a model holding search reads it.

The bench's family 2 sets every channel's spend by the drawn world's unobserved sentiment; a world
of this kind for it is described in ADR 0074.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from mixdyn import NATIONAL, Cadence, Channel, MediaPanel, Revenue

START = np.datetime64("2023-01-02")
CHANNELS = ("search", "tv")
_TANH_HALF = math.atanh(0.5)
_RETENTION = {"search": 0.2, "tv": 0.5}
_HALF = {"search": 60.0, "tv": 100.0}
_BETA = {"search": 200.0, "tv": 150.0}
_ORGANIC = 1000.0
_LIFT = 0.1  # the demand television raises at its whole saturation
_SEARCH = 60.0  # paid search's weekly spend at a demand of 1
_QUERIES = 100_000.0


@dataclass(frozen=True)
class SearchDemandWorld:
    """A history, its panel with the queries as the control `queries`, and the true returns:
    search's, and television's direct and total."""

    panel: MediaPanel
    search: float
    tv_direct: float
    tv_total: float


def _saturated(spend: np.ndarray, channel: str) -> np.ndarray:
    weeks = spend.size
    weights = _RETENTION[channel] ** np.arange(weeks)
    stock = np.convolve(spend, weights / weights.sum())[:weeks]
    return np.tanh(_TANH_HALF * stock / _HALF[channel])


def world(seed: int, weeks: int = 156) -> SearchDemandWorld:
    """The world drawn from `seed`."""
    rng = np.random.default_rng(seed)
    t = np.arange(1, weeks + 1)
    z = np.empty(weeks)
    z[0] = rng.normal(0.0, 0.08 / math.sqrt(1.0 - 0.8**2))
    for k in range(1, weeks):
        z[k] = 0.8 * z[k - 1] + rng.normal(0.0, 0.08)
    organic = np.exp(0.15 * np.sin(2.0 * math.pi * t / 52.0) + z)
    tv = rng.gamma(4.0, 25.0, weeks) * (rng.random(weeks) < 0.6)
    tv_effect = _saturated(tv, "tv")
    demand = organic * (1.0 + _LIFT * tv_effect)
    queries = _QUERIES * demand * np.exp(rng.normal(0.0, 0.02, weeks))
    search = _SEARCH * demand * np.exp(rng.normal(0.0, 0.25, weeks))
    search_effect = _BETA["search"] * _saturated(search, "search")
    direct = _BETA["tv"] * tv_effect
    kpi = _ORGANIC * demand + search_effect + direct + rng.normal(0.0, 20.0, weeks)
    panel = MediaPanel(
        dates=START + np.timedelta64(7, "D") * np.arange(weeks),
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=tuple(Channel(c) for c in CHANNELS),
        controls=("queries",),
        kpi=kpi[None],
        kpi_unit=Revenue(),
        spend=np.column_stack([search, tv])[None],
        currency="EUR",
        control_values=queries[None, :, None],
        filled=np.zeros((1, weeks), dtype=bool),
    )
    mediated = _ORGANIC * organic * _LIFT * tv_effect
    return SearchDemandWorld(
        panel=panel,
        search=float(search_effect.sum() / search.sum()),
        tv_direct=float(direct.sum() / tv.sum()),
        tv_total=float((direct + mediated).sum() / tv.sum()),
    )
