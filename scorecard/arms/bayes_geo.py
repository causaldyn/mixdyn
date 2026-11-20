"""The engine's static model over geos as an arm: one fit of a geo export's panel, every geo test in
its likelihood, and plans of every cell of the export.

The fit is `mixdyn.models.bayes_geo.fit_bayes_geo` on the export's geo panel
(`scorecard.arms.panel`), each geo test in its likelihood as an estimate in the geos it names: each
channel's return, carryover and curve shared by the geos and its effect in a geo pooled about the
national one; a trend of 3 knots and a yearly season of 3 harmonics, as the static arm's; normal
noise; Meridian's default prior on each channel's return; four chains of 1000 draws after 1000 of
warm-up, from a child of the export's seed. Where the model's ADRs name a default, the arm takes
it: a Hill (ADR 0011), every channel's deviations over the geos non-centred (ADR 0013: the analyst
names the channels to centre, and no rule centres one by itself), and chains started uniformly as
Stan's are (ADR 0012: the model over geos keeps them until a pilot on its own model decides).

The variants, each a plan of every cell in the export's layout (`scorecard.arms.panel.cells`)
against the export's boxes and budget, the history carried in:

* `bayes geo plug-in`: `mixdyn.decisions.plan_geos` on each geo's channels at the posterior's
  medians (`BayesGeoFit.channels`), a channel bought nationally planned as one national spend
  within its one cell's box (ADR 0042);
* `bayes geo cvar 0.1`: the static arm's `cvar 0.1` on the posterior over geos, the split with the
  most mean gain over the status quo in the worst tenth of `DRAWS` readings, by
  `chc.allocation.cvar_allocate`, each reading every cell's channel at one draw.

`BayesGeoPlugIn`, declared as `bayes-geo-plug-in`, is the same arm planning `bayes geo plug-in`
alone: the same fit and the same plan, its record kept under the same name, so a run declares one
of the two (ADR 0083). CHOICE: the CVaR plan searches every cell of every reading at once, and its
seconds grow faster with the geos than the fit's; a run that claims the plug-in need not pay them.

The static arm's `stacked` is not here: it plans the expected return of two fits, a Hill's and a
`tanh`'s, stacked, which here would be two fits of the model over geos.

A channel bought nationally reaches a geo at the geo's share of the national spend, so its geos'
cells are one (`national`). The model reads geo `g`'s at its people `p_g` times a curve of the spend
per head, `p_g b_g h(a(s_g X) / (p_g K))` with `s_g = p_g / P`, `P` the market's people, `X` the
national spend and `a` the adstock; `a` is linear, so that is `p_g b_g h(a(X) / (P K))`: one channel
of the national spend, its curve's scale `P K` and its coefficient `sum_g p_g b_g`, for the Hill and
the `tanh` alike (Maxima). Each reading's cell of such a channel is that one channel.

Each variant claims its plan's gain over the status quo, the mean and the central 90 % interval of
the gain over the readings (`scorecard.arms.panel.gains`). Its flags are the fit's sampler health
and leave-one-out reading under the static arm's names: the Hill's (`hill ...`), and the Hill's
weight, 1, as the one family whose draws the readings hold, so that a report reads the fit's
chains as it reads the static arm's (`scorecard.report.past`). Its settings name the family, the
start, the channels centred and the spread's prior, and its effective sample size is the fit's
least bulk one.

Neither variant forecasts: the library draws no forecast of the model over geos. Neither carries a
response the bench reads: the bench maps a response of the market alone
(`causaldyn_bench.scorecard.mapping`), where each of the model's channels is one geo's, and a
per-geo channel's market is a sum of curves, one a geo, and no one channel; the record says why
under `response.unmapped`.

A smoke run samples 2 chains of 100 draws after 100 of warm-up and reads `SMOKE_READINGS` of
them: `cvar_allocate` searches every cell of every reading at once, 21 cells on family 8.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from chc import response
from chc.allocation import cvar_allocate

from mixdyn import Channel
from mixdyn.decisions import GeoLimits, Range, plan_geos
from mixdyn.models.bayes_geo import BayesGeoSpec, fit_bayes_geo
from mixdyn.models.bayes_mmm import BayesMedia, UniformStart
from mixdyn.models.ridge_mmm import FourierSeason, LinearTrend
from scorecard.arms import panel as reading
from scorecard.arms.base import Planned
from scorecard.arms.bayes import (
    DRAWS,
    GAIN_LEVEL,
    HARMONICS,
    KNOTS,
    LEVEL,
    SAMPLER,
    SMOKE,
    YEAR,
    _health,
    least_ess,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from causaldyn_bench.scorecard.observe import Observation

    from mixdyn.models.bayes_geo import BayesGeoFit

FAMILY = "hill"
"""The one curve family the fit reads, by the static arm's name for it."""
START = UniformStart()
SMOKE_READINGS = 40
PLUG_IN, CVAR = "bayes geo plug-in", "bayes geo cvar 0.1"
UNMAPPED = (
    "the model's channels are each geo's, and the bench maps a response of the market alone: a"
    " per-geo channel's market is a sum of curves, one a geo"
)
"""Why a record carries no response the bench reads."""


@dataclass(frozen=True, slots=True, eq=False)
class Fitted:
    """The fit, its readings in the export's cell layout, and what a record keeps of it."""

    fit: BayesGeoFit
    readings: tuple[tuple[response.Channel, ...], ...]
    flags: dict[str, float | int | bool]
    settings: dict[str, Any]
    ess: float


class BayesGeo:
    """The arm; see the module."""

    variants = (PLUG_IN, CVAR)
    tests = True
    cells = True

    def fit(self, observation: Observation, *, smoke: bool) -> Fitted:
        if observation.population is None:
            msg = "the bayes geo arm plans a geo export's cells, and this export has no geo axis"
            raise ValueError(msg)
        data = reading.panel(observation)
        tests = reading.estimates(observation)
        spec = BayesGeoSpec(
            media=tuple(BayesMedia(c) for c in data.channels),
            trend=LinearTrend(KNOTS),
            season=FourierSeason(YEAR, HARMONICS),
        )
        sampler = SMOKE if smoke else SAMPLER
        seed = int(np.random.SeedSequence(observation.seed).spawn(1)[0].generate_state(1)[0])
        fit = fit_bayes_geo(
            data,
            spec,
            seed=seed,
            estimates=tests or None,
            chains=sampler.chains,
            warmup=sampler.warmup,
            draws=sampler.draws,
            start=START,
        )
        count = min(SMOKE_READINGS if smoke else DRAWS, sampler.chains * sampler.draws)
        by_geo = [fit.channel_draws(g, count) for g in data.geos]
        readings = tuple(
            _cells(observation, [drawn[i] for drawn in by_geo], fit.population)
            for i in range(count)
        )
        flags: dict[str, float | int | bool] = {f"{FAMILY} weight": 1.0}
        flags |= {f"{FAMILY} {name}": value for name, value in _health(fit).items()}
        settings = {
            "families": [FAMILY],
            "start": "uniform",
            "centred": sorted(spec.centred),
            "spread": spec.spread,
            "trend_knots": KNOTS,
            "season_harmonics": HARMONICS,
            "lift_rows": sum(len(rows) for rows in tests.values()),
            "readings": count,
            "gain_level": GAIN_LEVEL,
            "sampler": dataclasses.asdict(sampler),
            "seed": seed,
        }
        return Fitted(fit, readings, flags, settings, least_ess(flags, {FAMILY: count}))

    def plan(self, fit: Fitted, variant: str, observation: Observation) -> Planned:
        if variant == PLUG_IN:
            weekly = _plug_in(fit.fit, observation)
            settings: dict[str, Any] = {"planner": "plan_geos"}
        else:
            weekly = cvar_allocate(
                fit.readings,
                observation.budget,
                observation.planned,
                level=LEVEL,
                against=observation.status_quo,
                lower=observation.lower,
                upper=observation.upper,
                history=reading.cell_spend(observation),
            ).spend
            settings = {"planner": "cvar_allocate", "level": LEVEL}
        weekly = np.asarray(weekly, dtype=float)
        gain = reading.gains(fit.readings, observation, weekly)
        low, high = np.quantile(gain, [(1.0 - GAIN_LEVEL) / 2.0, (1.0 + GAIN_LEVEL) / 2.0])
        return Planned(
            weekly=weekly,
            gain=float(np.mean(gain)),
            gain_interval=(float(low), float(high)),
            response={"unmapped": UNMAPPED},
            flags=fit.flags,
            settings=fit.settings | settings,
            absent={"forecast": "the library draws no forecast of the model over geos"},
            ess=fit.ess,
        )


class BayesGeoPlugIn(BayesGeo):
    """The arm planning its plug-in alone; see the module."""

    variants = (PLUG_IN,)


def national(channels: Sequence[response.Channel], people: np.ndarray) -> response.Channel:
    """A channel bought nationally, `channels` its geos', as one channel of its national spend;
    see the module. `people` is each geo's population, in the fit's units.

    Raises:
        ValueError: on a curve other than a Hill or a `tanh`.
    """
    first = channels[0]
    curve = first.curve
    scale = float(curve.scale) * float(people.sum()) / float(people[0])
    if isinstance(curve, response.Hill):
        whole: response.Hill | response.Tanh = response.Hill(scale, float(curve.slope))
    elif isinstance(curve, response.Tanh):
        whole = response.Tanh(scale)
    else:
        msg = f"a national channel's geos are one channel on a Hill or a tanh, not {curve!r}"
        raise ValueError(msg)
    return response.Channel(first.kernel, whole, float(sum(float(c.coefficient) for c in channels)))


def _cells(
    observation: Observation, by_geo: Sequence[Mapping[str, response.Channel]], people: np.ndarray
) -> tuple[response.Channel, ...]:
    """One reading's channel of each cell, in the export's layout: a geo's own channel its geo's,
    a national channel its geos' as one (`national`)."""
    return tuple(
        national([geo[c] for geo in by_geo], people) if g is None else by_geo[g][c]
        for g, c in reading.cells(observation)
    )


def _plug_in(fit: BayesGeoFit, observation: Observation) -> np.ndarray:
    """`plan_geos`' plan on each geo's channels at the posterior's medians, in the export's cell
    layout: a national channel's cell its national spend, its geos' column summed."""
    data = reading.panel(observation)
    cells = reading.cells(observation)
    ranges = [
        Range(float(low), float(high))
        for low, high in zip(observation.lower, observation.upper, strict=True)
    ]
    limits = GeoLimits(
        cells={
            (data.geos[g], Channel(c)): r
            for (g, c), r in zip(cells, ranges, strict=True)
            if g is not None
        },
        national={Channel(c): r for (g, c), r in zip(cells, ranges, strict=True) if g is None},
    )
    plan = plan_geos(
        data,
        {g: {Channel(c): ch for c, ch in fit.channels(g).items()} for g in data.geos},
        observation.budget,
        observation.planned,
        limits,
    )
    column = {str(c): j for j, c in enumerate(data.channels)}
    return np.array(
        [
            plan.spend[:, column[c]].sum() if g is None else plan.spend[g, column[c]]
            for g, c in cells
        ]
    )
