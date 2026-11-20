"""The engine's discount choice on static worlds.

Five worlds of 156 weeks of `y = 100 + 0.8 x + N(0, 8^2)` with `x = max(0, N(50, 20))`. A static
world's true discount is 1, so of a grid below 1 the largest is the nearest.
"""

from __future__ import annotations

import numpy as np

from mixdyn import NATIONAL, Cadence, Count, read_panel
from mixdyn.models import DlmSpec, Effect, Trend, fit_dlm

DATES = np.datetime64("2023-01-02") + np.timedelta64(7, "D") * np.arange(156)
GRID = tuple(np.linspace(0.80, 0.995, 8))


def test_the_likelihood_chooses_the_static_truth() -> None:
    """The one-step likelihood over the eight discounts from 0.80 to 0.995, shared by the level and
    x, rises with the discount and picks 0.995 in every world."""
    for seed in range(11, 16):
        rng = np.random.default_rng(seed)
        x = np.maximum(0.0, rng.normal(50.0, 20.0, DATES.size))
        kpi = 100.0 + 0.8 * x + rng.normal(0.0, 8.0, DATES.size)
        panel = read_panel(
            {"date": DATES, "kpi": kpi, "x": x},
            date="date",
            kpi="kpi",
            spend={"x": "x"},
            cadence=Cadence.WEEK,
            kpi_unit=Count("units"),
            currency="USD",
        )
        likelihood = [
            fit_dlm(
                panel, NATIONAL, DlmSpec((Effect("x", d),), trend=Trend(1, d))
            ).filtered.log_likelihood
            for d in GRID
        ]
        assert np.all(np.diff(likelihood) > 0.0)
