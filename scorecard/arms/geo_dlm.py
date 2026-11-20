"""The engine's dynamic model over geos as an arm: `mixdyn.models.fit_geo_dlm` on a geo export's
panel, and a plan of every cell from its coefficients (ADR 0078).

The model (`mixdyn.models.GeoDlmSpec`) is additive: each geo a level of its own and a yearly season
of 3 harmonics, and each channel an effect on its raw spend in the geo, its national coefficient
plus the geo's static deviation, the geos' spread chosen by the marginal likelihood and its draws
mixed. The discounts are the dynamic arm's starting spec's (`scorecard.arms.dlm`): 0.99 for each
effect's national coefficient and for the level, 1 for the season. CHOICE: the model over geos
reads its columns raw, refusing a transform or a sign, so a channel's response in a geo is
linear in its spend that week: no carryover, no saturation, and a coefficient that may be below
0. The arm fits that model as the library ships it and plans what it says; the dynamic arm's
search over transforms and discounts has no counterpart over geos.

The fit reads no lift test (the model over geos takes none), so the harness fits it once for every
rung. Each cell's coefficient is the last week's (`mixdyn.models.geo_effects`' means): a geo's own
channel its geo's coefficient on the raw spend, and a channel bought nationally its geos'
coefficients weighed by their shares of the population, since the panel splits its national spend
by them. On lines the best plan at the budget is exact (`fill`): every cell at its least, then the
rest of the week's budget to the cells of the largest coefficient first, each up to its most.
`chc.allocation.allocate` is not used: it refuses a coefficient below 0, whose curve it reads as
convex, and the model over geos may give one; the bench scores a plan at the whole budget, so a
cell below 0 still takes budget when the others are full. The arm claims the plan's gain over the
status quo on those lines, one number with no interval: `geo_effects` reads each coefficient's
marginal, not the joint posterior a gain's interval needs. Its flags are the spread's
draws and the count of cells whose coefficient is below 0; its settings the spec, each cell's
coefficient and the seed.

The record carries no forecast and no response the bench reads, as the static model over geos's
(`scorecard.arms.bayes_geo`). A smoke run draws `SMOKE_DRAWS` of the spread.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from chc import response

from mixdyn.models import Effect, GeoDlmSpec, Season, Trend, fit_geo_dlm, geo_effects
from scorecard.arms import panel as reading
from scorecard.arms.base import Planned
from scorecard.arms.dlm import HARMONICS, YEAR

if TYPE_CHECKING:
    from causaldyn_bench.scorecard.observe import Observation

    from mixdyn.models import GeoDlmFit

VARIANT = "geo dlm"
EFFECT, LEVEL, SEASON = 0.99, 0.99, 1.0
"""The discounts: each effect's national coefficient's, the level's and the season's."""
DRAWS, SMOKE_DRAWS = 64, 16
UNMAPPED = (
    "the model's channels are each geo's, and the bench maps a response of the market alone: a"
    " per-geo channel's market is a sum of lines, one a geo"
)


@dataclass(frozen=True, slots=True, eq=False)
class Fitted:
    """The fit, each cell's coefficient in the export's layout, and what a record keeps."""

    fit: GeoDlmFit
    coefficients: np.ndarray
    flags: dict[str, float | int | bool]
    settings: dict[str, Any]


def spec(observation: Observation) -> GeoDlmSpec:
    """The model the arm fits; see the module."""
    return GeoDlmSpec(
        effects=tuple(Effect(c, EFFECT) for c in observation.channels),
        trend=Trend(1, LEVEL),
        season=Season(YEAR, HARMONICS, SEASON),
        form="additive",
    )


def coefficients(observation: Observation, mean: np.ndarray) -> np.ndarray:
    """Each cell's coefficient in `reading.cells`' order from each geo's on each channel, `mean`,
    `(geos, channels)`: a geo's own its own, a national channel's its geos' by population share."""
    shares = np.asarray(observation.population, dtype=float)
    shares = shares / shares.sum()
    return np.array(
        [
            float(shares @ mean[:, observation.channels.index(c)])
            if g is None
            else float(mean[g, observation.channels.index(c)])
            for g, c in reading.cells(observation)
        ]
    )


def linear(coefficient: float) -> response.Channel:
    """A channel whose response a week is `coefficient` times its spend that week."""
    return response.Channel(
        response.GeometricAdstock(0.0, length=1, normalized=True),
        response.Power(1.0, 1.0),
        coefficient,
    )


def fill(
    coefficients: np.ndarray, lower: np.ndarray, upper: np.ndarray, weekly: float
) -> np.ndarray:
    """The plan of linear cells that returns most at `weekly` a week within each cell's box: every
    cell at its least, then the rest to the cells of the largest coefficient first."""
    plan = np.array(lower, dtype=float)
    left = weekly - float(plan.sum())
    slack = 1e-12 * weekly
    if not -slack <= left <= float(np.sum(upper)) - float(plan.sum()) + slack:
        msg = f"the cells' boxes hold no plan of {weekly} a week"
        raise ValueError(msg)
    for c in np.argsort(-np.asarray(coefficients), kind="stable"):
        step = min(float(upper[c]) - plan[c], max(left, 0.0))
        plan[c] += step
        left -= step
    return plan


class GeoDlm:
    """The arm; see the module."""

    variants = (VARIANT,)
    tests = False
    cells = True

    def fit(self, observation: Observation, *, smoke: bool) -> Fitted:
        if observation.population is None:
            msg = "the geo dlm arm plans a geo export's cells, and this export has no geo axis"
            raise ValueError(msg)
        seed = int(np.random.SeedSequence(observation.seed).spawn(1)[0].generate_state(1)[0])
        draws = SMOKE_DRAWS if smoke else DRAWS
        model = spec(observation)
        fit = fit_geo_dlm(reading.panel(observation), model, draws=draws, seed=seed)
        effects = geo_effects(fit)
        held = coefficients(observation, effects.mean)
        flags: dict[str, float | int | bool] = {
            "spread draws": draws,
            "negative coefficients": int(np.sum(held < 0.0)),
        }
        settings = {
            "form": model.form,
            "discounts": {"effects": EFFECT, "level": LEVEL, "season": SEASON},
            "season_harmonics": HARMONICS,
            "coefficients": held.tolist(),
            "seed": seed,
        }
        return Fitted(fit, held, flags, settings)

    def plan(self, fit: Fitted, variant: str, observation: Observation) -> Planned:
        channels = [linear(float(b)) for b in fit.coefficients]
        weekly = fill(
            fit.coefficients,
            np.asarray(observation.lower, dtype=float),
            np.asarray(observation.upper, dtype=float),
            observation.budget / observation.planned,
        )
        (gain,) = reading.gains((tuple(channels),), observation, weekly)
        return Planned(
            weekly=weekly,
            gain=float(gain),
            response={"unmapped": UNMAPPED},
            flags=fit.flags,
            settings=fit.settings | {"planner": "largest coefficient first"},
            absent={
                "forecast": "the library draws no forecast of the model over geos",
                "gain_interval": "the coefficients' marginals give no joint interval of a gain",
            },
        )
