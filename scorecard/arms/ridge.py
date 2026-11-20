"""The engine's ridge model as an arm: the search of the export's panel for each channel's carryover
and Hill and the penalty, the chosen model fitted to the whole history, and the plan
`chc.allocation.allocate` makes on its channels.

The model is `mixdyn.models.ridge_mmm`'s (ADR 0008): each channel's spend through a geometric
adstock and a Hill, held at or above 0, the controls, a trend of 3 knots and a yearly season of 3
harmonics, by bounded ridge. `search_ridge` chooses each channel's `theta` in [0, 0.9], its Hill's
`alpha` in [0.5, 3] and `gamma` in [0.3, 1], and the penalty in [1e-4, 1e2], by the validation
NRMSE of 4 folds of 13 weeks: 256 Sobol starts, scrambled by a child of the export's seed, and the
8 best polished. These are the library's search as its scored run on Track M v2 ran it (ADR 0008).
A history too short for 4 folds after 26 weeks drops its earliest folds (`Folds.within`): the first
fold always fits at least 26 weeks, the shortest prefix 4 folds leave on 78 weeks, so 52 weeks are
searched by 2 folds and a history of 78 weeks or more by the 4 (ADR 0041).
The search's fit is `fit_ridge`'s of the chosen model over the whole history: the library composes
the two so (`RidgeSearch.fit`), and the arm reads that fit. A promotion's coefficient is held at or
above 0 and every other control is free, as the static arm holds them.

The model reads a geo test as `chc.lift.LiftTest`: the gap between its groups, period by period,
with the readout periods before the groups' spend parts, off which the fit reads the tests' noise.
The export reduces each test to one lift over its readout and its standard error
(`causaldyn_bench.budget_regret.lift_rows`), which the model has no way to read. So the arm reads no
lift test (`tests`): the harness hands it each export without them and fits a world once for every
rung, a rung's plan with tests is the plan without them, and every record says so, under `n/a`
(`UNREAD`) and in its settings (`lift_rows_read`). On a geo export the harness hands the arm the
export's market (`scorecard.arms.panel.market`, ADR 0043): it fits the geos' sums and plans one
weekly spend a channel.

The variant `ridge` plans with `allocate` on the fit's channels, against the export's box and
budget, the history carried in. It claims its plan's gain over the status quo on those channels: a
point, since the fit has no draws to read an interval off. Its response
(`scorecard.arms.mapped`) is the fit's channels as one draw, beside each channel's contribution to
the fit over the history's weeks. It forecasts nothing: the library has no forecast of a ridge fit.
Its flags are the search's score and evaluations, the fit's NRMSE and R² in sample, the chosen
penalty and each channel's `theta`, `alpha` and `gamma`, whether each of these sits at an end of
its interval, and whether each channel's coefficient sits on its bound at 0.

A smoke run scores 16 starts and polishes 2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from chc.allocation import allocate

from mixdyn import NATIONAL
from mixdyn.models.ridge_mmm import Control, FourierSeason, LinearTrend
from mixdyn.models.ridge_search import (
    Folds,
    GeometricSpace,
    Interval,
    MediaSpace,
    RidgeSpace,
    search_ridge,
)
from mixdyn.transforms import Geometric
from scorecard.arms import panel as reading
from scorecard.arms.base import Planned, r2
from scorecard.arms.mapped import mapped

if TYPE_CHECKING:
    from causaldyn_bench.scorecard.observe import Observation

    from mixdyn.models.ridge_search import RidgeSearch

THETA = Interval(0.0, 0.9)
ALPHA = Interval(0.5, 3.0)
GAMMA = Interval(0.3, 1.0)
PENALTY = Interval(1e-4, 1e2)
POSITIVE = ("promotion",)
"""Controls held at or above 0."""
KNOTS, HARMONICS, YEAR = 3, 3, 365.25 / 7.0
FOLDS = Folds(4, 13)
"""The most folds; a short history holds fewer."""
LEAST = 26
"""The weeks the first fold fits at least."""
STARTS, POLISH = 256, 8
SMOKE_STARTS, SMOKE_POLISH = 16, 2
UNREAD = (
    "the ridge model reads a geo test's gap period by period (chc.lift.LiftTest), and the export"
    " reduces each test to one lift and its standard error: every rung's plan is the plan of k = 0"
)
"""What a record says of the lift tests it did not read."""


@dataclass(frozen=True, slots=True, eq=False)
class Fitted:
    """The search, and what a record keeps of it."""

    search: RidgeSearch
    flags: dict[str, float | int | bool]
    settings: dict[str, Any]


class Ridge:
    """The arm; see the module."""

    variants = ("ridge",)
    tests = False
    cells = False

    def fit(self, observation: Observation, *, smoke: bool) -> Fitted:
        space = RidgeSpace(
            media=tuple(
                MediaSpace(c, GeometricSpace(THETA), ALPHA, GAMMA) for c in observation.channels
            ),
            controls=tuple(
                Control(c, lower=0.0) if c in POSITIVE else Control(c) for c in observation.controls
            ),
            penalty=PENALTY,
            trend=LinearTrend(KNOTS),
            season=FourierSeason(YEAR, HARMONICS),
        )
        starts, polish = (SMOKE_STARTS, SMOKE_POLISH) if smoke else (STARTS, POLISH)
        seed = int(np.random.SeedSequence(observation.seed).spawn(1)[0].generate_state(1)[0])
        folds = Folds.within(
            observation.sales.shape[0], most=FOLDS.count, horizon=FOLDS.horizon, least=LEAST
        )
        search = search_ridge(
            reading.panel(observation),
            NATIONAL,
            space,
            folds,
            seed=seed,
            starts=starts,
            polish=polish,
        )
        settings = {
            "theta": [THETA.low, THETA.high],
            "alpha": [ALPHA.low, ALPHA.high],
            "gamma": [GAMMA.low, GAMMA.high],
            "penalty": [PENALTY.low, PENALTY.high],
            "controls": {
                c: "at or above 0" if c in POSITIVE else "free" for c in observation.controls
            },
            "trend_knots": KNOTS,
            "season_harmonics": HARMONICS,
            "folds": [folds.count, folds.horizon],
            "starts": starts,
            "polish": polish,
            "seed": seed,
            "lift_rows_read": 0,
        }
        return Fitted(search, _flags(search, observation.channels), settings)

    def plan(self, fit: Fitted, variant: str, observation: Observation) -> Planned:
        channels = fit.search.fit.channels()
        read = tuple(channels[c] for c in observation.channels)
        weekly = allocate(
            list(read),
            observation.budget,
            observation.planned,
            lower=observation.lower,
            upper=observation.upper,
            history=observation.spend,
        ).spend
        return Planned(
            weekly=np.asarray(weekly, dtype=float),
            gain=float(reading.gains([read], observation, weekly)[0]),
            response=_response(fit.search, observation),
            flags=fit.flags,
            settings=fit.settings | {"planner": "allocate"},
            absent={
                "gain_interval": "the ridge fit is a point, with no draws to read an interval off",
                "forecast": "the library has no forecast of a ridge fit",
                "lift tests": UNREAD,
            },
        )


def _response(search: RidgeSearch, observation: Observation) -> dict[str, Any]:
    """The record's response: the fit's channels as one draw, beside each channel's contribution
    to the fit in each of the history's weeks."""
    fit, names = search.fit, observation.channels
    channels = fit.channels()
    weekly = np.column_stack([fit.contributions[c] for c in names])
    return mapped(
        names, [tuple(channels[c] for c in names)], weekly.sum(axis=0)[None], weekly[None]
    )


def _flags(search: RidgeSearch, channels: tuple[str, ...]) -> dict[str, float | int | bool]:
    """What a record keeps of the search; see the module."""
    fit = search.fit
    spec = fit.spec
    flags: dict[str, float | int | bool] = {
        "score": search.score,
        "evaluations": search.evaluations,
        "nrmse": fit.nrmse(),
        "r2": r2(fit.y[fit.fitted], fit.estimated[fit.fitted]),
        "penalty": spec.penalty,
        "penalty at an end": _at_an_end(spec.penalty, PENALTY),
    }
    media = {m.channel: m for m in spec.media}
    for name in channels:
        m = media[name]
        assert isinstance(m.adstock, Geometric)  # the space searches a geometric adstock alone
        for parameter, value, interval in (
            ("retention", m.adstock.theta, THETA),
            ("alpha", m.hill.alpha, ALPHA),
            ("gamma", m.hill.gamma, GAMMA),
        ):
            flags[f"{name} {parameter}"] = value
            flags[f"{name} {parameter} at an end"] = _at_an_end(value, interval)
        flags[f"{name} on its bound"] = fit.bound[name]
    return flags


def _at_an_end(value: float, interval: Interval) -> bool:
    """Whether `value` sits at either end of `interval`, but for rounding."""
    return any(
        math.isclose(value, end, rel_tol=1e-9, abs_tol=1e-12)
        for end in (interval.low, interval.high)
    )
