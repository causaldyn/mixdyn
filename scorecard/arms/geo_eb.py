"""The engine's closed-form model over geos as an arm: one empirical-Bayes fit of a geo export's
panel, every geo test a row of its likelihood, and one plan of every cell of the export (ADR 0090).

The fit is `mixdyn.models.fit_pooled_geos` on the export's geo panel and its tests as estimates
in the geos they name (`scorecard.arms.panel`), its kernel the export's length: each channel's
curve of spend per head shared by the geos, its margin in a geo pooled about the channel's, each
geo its own intercept, season and trend, and each channel's posterior over a grid of its
carryover and curvature. It is NumPy alone and draws nothing, so it has no seed, and a smoke run
is the run.

The one variant, `geo eb`, plans every cell in the export's layout (`scorecard.arms.panel.cells`)
against the export's boxes and budget by `mixdyn.decisions.geo_plan.plan_mixture`: water-filling on
each cell's expected worth under its channel's posterior, the history carried in, a channel bought
nationally one cell of its national spend. It claims no gain (ADR 0090): the plan's expected worth
less the status quo's is read on the plan chosen to maximise it, at the posterior means, and that
reading over-states the gain the plan realises; the fit keeps each coefficient's posterior mean
under each grid point, not the joint posterior a correction or an interval needs.

Its flags are the lift rows it read, the fewest and the most grid points a channel's posterior
kept, and the count of cells whose expected margin is below 0. Its settings are the model's
constants, each channel's grid point at the end of the coordinate ascent and its kept points, and
the planner's.

The record carries no forecast and no response the bench reads, as the static model over geos's
(`scorecard.arms.bayes_geo`): a per-geo channel's market is a sum of curves, one a geo and a grid
point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mixdyn import Channel
from mixdyn.decisions.geo_plan import BISECTIONS, GRID_POINTS, plan_mixture
from mixdyn.models import geo_eb
from scorecard.arms import panel as reading
from scorecard.arms.base import Planned

if TYPE_CHECKING:
    from causaldyn_bench.scorecard.observe import Observation

VARIANT = "geo eb"
UNMAPPED = (
    "the model's channels are each geo's, and the bench maps a response of the market alone: a"
    " per-geo channel's market is a sum of curves, one a geo and a grid point"
)
"""Why a record carries no response the bench reads."""


@dataclass(frozen=True, slots=True, eq=False)
class Fitted:
    """The fit, each cell's expected response in the export's layout, and what a record keeps."""

    fit: geo_eb.PooledGeoFit
    cells: tuple[geo_eb.CellResponse, ...]
    flags: dict[str, float | int | bool]
    settings: dict[str, Any]


class GeoEb:
    """The arm; see the module."""

    variants = (VARIANT,)
    tests = True
    cells = True

    def fit(self, observation: Observation, *, smoke: bool) -> Fitted:
        if observation.population is None:
            msg = "the geo eb arm plans a geo export's cells, and this export has no geo axis"
            raise ValueError(msg)
        data = reading.panel(observation)
        tests = reading.estimates(observation)
        fit = geo_eb.fit_pooled_geos(data, tests, int(observation.kernel_length))
        cells = tuple(
            fit.cell(None if g is None else data.geos[g], Channel(c))
            for g, c in reading.cells(observation)
        )
        kept = [int(m.points.size) for m in fit.mixtures]
        margins = [float(cell.weights @ cell.coefficients) for cell in cells]
        flags: dict[str, float | int | bool] = {
            "lift rows": fit.tests,
            "fewest kept points": min(kept),
            "most kept points": max(kept),
            "negative margins": sum(m < 0.0 for m in margins),
        }
        settings = {
            "retentions": list(geo_eb.RETENTIONS),
            "curvatures": [str(kappa) for kappa in geo_eb.CURVATURES],
            "spread": geo_eb.SPREAD,
            "prior_width": geo_eb.PRIOR_WIDTH,
            "season_harmonics": geo_eb.HARMONICS,
            "sweeps": geo_eb.SWEEPS,
            "kept": geo_eb.KEPT,
            "selected": {
                str(c): {
                    "retention": geo_eb.POINTS[j][0],
                    "curvature": str(geo_eb.POINTS[j][1]),
                    "kept_points": n,
                }
                for c, j, n in zip(fit.channels, fit.selected, kept, strict=True)
            },
            "lift_rows": fit.tests,
        }
        return Fitted(fit, cells, flags, settings)

    def plan(self, fit: Fitted, variant: str, observation: Observation) -> Planned:
        periods = int(observation.planned)
        plan = plan_mixture(
            fit.cells, observation.lower, observation.upper, observation.budget, periods
        )
        return Planned(
            weekly=plan.weekly,
            gain=None,
            response={"unmapped": UNMAPPED},
            flags=fit.flags,
            settings=fit.settings
            | {"planner": "plan_mixture", "grid_points": GRID_POINTS, "bisections": BISECTIONS},
            absent={
                "forecast": "the library draws no forecast of the model over geos",
                "gain": (
                    "the plan's worth at the posterior means, read on the plan chosen to maximise"
                    " it, over-states the gain it realises, and the means give no interval of one"
                ),
            },
        )
