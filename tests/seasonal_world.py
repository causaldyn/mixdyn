"""The seasonal world the dynamic linear model's forecast, choice and decomposition tests read.

208 weeks from 2023-01-02 of `y = 100 + s(w) + 0.5 x + N(0, 1)`, `s(w) = 10 sin(2 pi w / 52) +
4 cos(4 pi w / 52)` for the week `w` counted from 1, and `x = max(0, N(50, 15))`. The first 156
weeks are the history a model is fitted to, the last 52 the weeks it forecasts, so every
component's truth is known in both.
"""

from __future__ import annotations

import numpy as np

from mixdyn import Cadence, Count, MediaPanel, read_panel

SEED = 20261001
WEEKS, HISTORY = 208, 156
NOISE = 1.0
DATES = np.datetime64("2023-01-02") + np.timedelta64(7, "D") * np.arange(WEEKS)
_WEEK = np.arange(1, WEEKS + 1)
SEASON = 10.0 * np.sin(2.0 * np.pi * _WEEK / 52.0) + 4.0 * np.cos(4.0 * np.pi * _WEEK / 52.0)
_RNG = np.random.default_rng(SEED)
X = np.maximum(0.0, _RNG.normal(50.0, 15.0, WEEKS))
CONTRIBUTION = 0.5 * X
MEAN = 100.0 + SEASON + CONTRIBUTION
KPI = MEAN + _RNG.normal(0.0, NOISE, WEEKS)


def history() -> MediaPanel:
    """The history as a panel of one channel, `x`."""
    return read_panel(
        {"date": DATES[:HISTORY], "kpi": KPI[:HISTORY], "x": X[:HISTORY]},
        date="date",
        kpi="kpi",
        spend={"x": "x"},
        cadence=Cadence.WEEK,
        kpi_unit=Count("units"),
        currency="USD",
    )
