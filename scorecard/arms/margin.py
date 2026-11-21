"""The margin rule as an arm, `margin`: each channel's margin, the KPI a unit of weekly spend
returns, read off the export's history by regression (`mixdyn.models.margin_mmm`) without its
noise mixture, `noise=None`, as the rule was found, the margins shrunk towards their common level
by empirical Bayes, and the plan a linear response makes of the shrunk means, every channel at its
floor and the rest of the budget to the highest mean, then the next
(`mixdyn.decisions.margin_plan`). ADR 0086 records the rule and what it was measured at.

The arm reads the export alone: the sales, the spend, the controls `promotion` and `price` where
it holds them, the kernel's length, the status quo, the box and the budget. An export without one
fits without its columns, and without the price without the promoted weeks too, which are read off
it; its settings name the controls read. A geo export is refused, its market too: the rule reads
one national history, and M3 plans the geo families with its geo candidates. It reads no lift test
(`tests`): the harness hands it each export without them and fits a world once for every rung, a
rung's plan with tests is the plan without them, and every record says so, under `n/a` (`UNREAD`)
and in its settings (`lift_rows_read`). The fit is a regression and its evidence, no sampler: a
smoke run is the full run.

Six variants. `margin` is the rule as it was found. `margin guarded` holds each channel whose
margin the data do not identify at its status quo, its floor and its ceiling in the corner both
its status quo, and leaves its margin out of the shrinkage, which pulls the other channels'
towards their own common level; the rest of the budget goes to them by the corner. A channel is
not identified where the prior, not the data, sets its margin: where the retention combinations
in which the prior's precision on its coefficient exceeds the data's hold more than `HELD_SHARE`
of the evidence weight (`mixdyn.models.margin_mmm.MarginFit.prior_share`), as a channel whose
spend never moves does. With no channel held its plan is `margin`'s; with every channel held, or
no positive margin among the others, the status quo. `margin ranged` is `margin` with each
channel's box cut to the range of weekly spend its history logged
(`mixdyn.decisions.margin_plan.observed_range`): the margins are read inside that range alone, and
a plan outside it extrapolates them. An export whose cut box holds no plan on the budget is a
failed plan (`ObservedRangeError`). `margin curved` (ADR 0087) reads the margins off the curved
regression (`mixdyn.models.margin_mmm.fit_curved_margins`), each channel's column a
Michaelis-Menten curve of its adstock, half-saturated at 16 times its status quo: its margins at
the status quo are shrunk as the others', and the plan water-fills them along each channel's curve
(`mixdyn.decisions.margin_plan.water_fill`) inside `margin ranged`'s box cut further to the range
of each channel's adstock at its retention of the most evidence weight
(`mixdyn.decisions.margin_plan.adstock_range`). It runs its own regression in its plan, so its
cost and its failure are its own: a curved fit the data leave degenerate fails this variant alone.
`margin curved noise` (ADR 0091) is `margin curved` with its regression under a mixture of error
models (`mixdyn.models.margin_mmm.NoiseMixture`): the trend's knots every 52 weeks, a flat prior on
the nuisance columns, and errors that are an AR(1) plus white noise, of a larger variance on the
promoted weeks, every error model weighted by its evidence. Its shrinkage, its curves, its box and
its water-filling are `margin curved`'s. A channel whose columns the nuisance columns span
(`MarginFit.spanned`), as one whose spend never moves, has a margin of exactly 0 there, its
prior's, and no rounding residue in its place: it is not among the positive margins that set the
shrinkage's scale, and the variant holds it at its status quo. `margin held` (ADR 0092) is
`margin curved noise` with the time guard (`mixdyn.decisions.margin_plan.time_held`): a channel
whose weekly spend correlates with the week's number above `TIME_HELD` in absolute value is held
at its status quo, its floor and its ceiling both its status quo, because a response that curves
and one that drifts give such a history alike. Its margin stays in the shrinkage, as the others'
are read; a channel the fit names spanned is held as in `margin curved noise`, whatever its
correlation. The other channels are water-filled in their box at the budget; where a channel is
held by the guard and fewer than two channels are free, neither held nor of a box that is a
point, or the box holds no plan at the budget, the plan is the status quo. With no channel held
by the guard its plan is `margin curved noise`'s.

Each variant's weekly budget is the status quo's weekly total, as the rule was found with: the
bench's budget is the quarter's weeks times it in every family, and an export whose budget is
another is refused. Where no margin is positive the plan is the status quo. The record claims
the plan's gain over the status quo on the shrunk means, `planned sum_c m_c (p_c - s_c)` over the
quarter and the kernel's tail, the history's carryover the same under both plans, with the normal
interval of the posterior's covariance at `GAIN_LEVEL`; `margin curved`, `margin curved noise`
and `margin held` claim it along their curves, `planned sum_c m_c (H_c(p_c) - H_c(s_c))`, the
interval read alike, and nothing where the guard plays the status quo.
It forecasts nothing, and it records no response: its channels are straight lines, or mixtures
over retentions of curves, and a record's response maps one geometric kernel through a `tanh` or
a Hill (`scorecard.arms.mapped`); the bench scores the plan's regret and its claim without one.
Its flags: each channel's margin and its standard deviation before and after the variant's
shrinkage, NaN after it where nothing was shrunk or the channel was held, its prior share,
whether the variant held it, the shrinkage's posterior mean of `tau` over the mean positive
margin (`tau_rel`), the AR(1) coefficient's mean over the retention combinations (`phi`), each
channel's posterior mean retention, the promoted phases read (`events`), the evidence weight of
the most likely combination (`top weight`), the weeks the regression read (`rows`) and the
burn-in before them (`burn-in rows`), each read off the variant's own regression; a
`margin curved`, a `margin curved noise` and a `margin held` record add each channel's
half-saturation, the retention its planning range reads and that range. A `margin curved noise`
and a `margin held` record's `phi` is the AR(1) coefficient's posterior mean over the error
models, and they add those of the AR(1)'s share of the errors' variance (`ar share`) and of the
promoted weeks' variance over the other weeks' (`promoted variance`); a `margin held` record adds
each channel's absolute correlation with the week's number (`time correlation`) and the channels
left free (`free channels`). A `margin guarded`, a `margin curved noise` and a `margin held`
record's settings say why each channel it held was held; a `margin ranged`, a `margin curved`, a
`margin curved noise` and a `margin held` record's name its planning range; a
`margin curved noise` and a `margin held` record's name its trend's knots, its nuisance prior and
its error models' grids; and a `margin held` record's the guard's threshold.

A fit the data leave degenerate raises `mixdyn.models.margin_mmm.DegenerateFitError`, which the
harness records as the fit's failure, and the bench scores as the status quo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from mixdyn.decisions.margin_plan import (
    BISECTIONS,
    HALF_NORMAL_SCALE,
    PRICE_SPAN,
    TAU_GRID,
    TIME_HELD,
    Shrunk,
    adstock_range,
    observed_range,
    plan_corner,
    shrink_margins,
    time_held,
    water_fill,
)
from mixdyn.models.margin_mmm import (
    AR_MOST,
    BURN_IN,
    DEGREE,
    HALF_SATURATION,
    HARMONICS,
    KNOT_EVERY,
    PRESAMPLE,
    PRICE_DROP,
    PRIOR,
    RETENTIONS,
    SPANNED,
    YEAR,
    MarginFit,
    NoiseMixture,
    fit_curved_margins,
    fit_margins,
)
from scorecard.arms.base import Planned
from scorecard.arms.panel import Market

if TYPE_CHECKING:
    from causaldyn_bench.scorecard.observe import Observation

VARIANT = "margin"
GUARDED = "margin guarded"
RANGED = "margin ranged"
CURVED = "margin curved"
NOISE = "margin curved noise"
HELD = "margin held"
NOISE_MIXTURE = NoiseMixture()
"""The mixture of error models `NOISE`'s and `HELD`'s regression is fitted under: the regression's
own grids."""
FREE_LEAST = 2
"""The channels `HELD` needs free, neither held nor of a box that is a point, to plan where its
guard holds one: with fewer the budget leaves nothing to move."""
HELD_SHARE = 0.5
"""The evidence weight past which a channel's margin is the prior's, not the data's, and
`GUARDED` holds the channel at its status quo (`MarginFit.prior_share`)."""
CONTROLS = ("promotion", "price")
"""The controls the regression reads where an export holds them, by the names it gives them."""
GEO = (
    "a geo export is refused, its market too: the margin rule reads one national history, and M3"
    " plans the geo families with its geo candidates"
)
"""Why a geo export's record is a failed fit."""
GAIN_LEVEL = 0.9
"""The central share of the claimed gain's normal interval."""
SPENT = 1e-12
"""How far the quarter's budget may lie from its weeks times the status quo's weekly total,
relative to it: rounding."""
UNREAD = (
    "the margin regression reads the history alone, no lift row: every rung's plan is the plan of"
    " k = 0"
)
"""What a record says of the lift tests it did not read."""


@dataclass(frozen=True, slots=True, eq=False)
class Fitted:
    """The regression's margins and their shrunk posterior, None where no margin is positive; the
    channels `GUARDED` holds, and its posterior of the others' margins, None where every channel
    is held or none of theirs is positive; and the settings a record keeps."""

    margins: MarginFit
    shrunk: Shrunk | None
    held: npt.NDArray[np.bool_]
    guarded: Shrunk | None
    settings: dict[str, Any]


class Margin:
    """The arm; see the module."""

    variants = (VARIANT, GUARDED, RANGED, CURVED, NOISE, HELD)
    tests = False
    cells = False

    def fit(self, observation: Observation, *, smoke: bool) -> Fitted:
        """The margins, their shrinkage, the channels held and the others' shrinkage; `smoke`
        changes nothing.

        Raises:
            ValueError: on a geo export or its market (`GEO`), and as `fit_margins` raises,
                `DegenerateFitError` included.
        """
        if isinstance(observation, Market) or observation.population is not None:
            raise ValueError(GEO)
        margins = fit_margins(
            observation.sales,
            observation.spend,
            observation.controls.get("promotion"),
            observation.controls.get("price"),
            observation.kernel_length,
            noise=None,
        )
        shrunk = shrink_margins(margins.mean, margins.cov)
        settings = {
            "retentions": list(RETENTIONS),
            "prior": PRIOR,
            "trend": {"degree": DEGREE, "knot_every": KNOT_EVERY},
            "season": {"year": YEAR, "harmonics": HARMONICS},
            "price_drop": PRICE_DROP,
            "ar": [0.0, AR_MOST],
            "burn_in_most": BURN_IN,
            "presample_weeks": PRESAMPLE,
            "kernel_length": observation.kernel_length,
            "tau_rel": [float(TAU_GRID[0]), float(TAU_GRID[-1]), int(TAU_GRID.size)],
            "half_normal_scale": HALF_NORMAL_SCALE,
            "gain_level": GAIN_LEVEL,
            "held_share": HELD_SHARE,
            "controls": [c for c in CONTROLS if c in observation.controls],
            "lift_rows_read": 0,
        }
        held = margins.prior_share > HELD_SHARE
        return Fitted(margins, shrunk, held, _guarded(margins, shrunk, held), settings)

    def plan(self, fit: Fitted, variant: str, observation: Observation) -> Planned:
        """The corner on the variant's shrunk means in its box, the channels it holds at their
        status quo, or the status quo where it has no posterior; in `CURVED`, `NOISE` and
        `HELD`, the curved regression's plan (`_curved`).

        Raises:
            ValueError: on a budget that is not the quarter's weeks times the status quo's weekly
                total, and as `plan_corner` and, in `RANGED`, `observed_range` raise, and in
                `CURVED`, `NOISE` and `HELD` as `_curved` raises.
        """
        quo = np.asarray(observation.status_quo, dtype=float)
        weekly_budget = float(quo.sum())
        if abs(observation.planned * weekly_budget - observation.budget) > SPENT * abs(
            observation.budget
        ):
            msg = (
                f"the rule spends the status quo's weekly total, {weekly_budget} over"
                f" {observation.planned} weeks, and the quarter's budget is {observation.budget}"
            )
            raise ValueError(msg)
        if variant in (CURVED, NOISE, HELD):
            noise = None if variant == CURVED else NOISE_MIXTURE
            return _curved(fit, observation, quo, weekly_budget, noise, guarded=variant == HELD)
        if variant == GUARDED:
            post, held = fit.guarded, fit.held
        else:
            post, held = fit.shrunk, np.zeros(quo.size, dtype=bool)
        if variant == RANGED:
            lower, upper = observed_range(
                observation.spend, observation.lower, observation.upper, weekly_budget
            )
        else:
            lower, upper = observation.lower, observation.upper
        kept = ~held
        if post is None:
            weekly, gain, interval = quo.copy(), 0.0, (0.0, 0.0)
        else:
            means = np.zeros(quo.size)
            means[kept] = post.mean
            low = np.where(kept, lower, quo)
            high = np.where(kept, upper, quo)
            weekly = plan_corner(means, low, high, weekly_budget)
            move = (weekly - quo)[kept]
            gain = observation.planned * float(post.mean @ move)
            spread = observation.planned * math.sqrt(float(move @ post.cov @ move))
            z = NormalDist().inv_cdf((1.0 + GAIN_LEVEL) / 2.0)
            interval = (gain - z * spread, gain + z * spread)
        settings = fit.settings | {
            "planner": "corner",
            "weekly budget": "the status quo's weekly total",
        }
        if variant == GUARDED:
            settings["held at the status quo"] = {
                name: (
                    "the prior's precision on its margin exceeds the data's in retention"
                    f" combinations holding {share:.3f} of the evidence weight"
                )
                for name, share, hold in zip(
                    observation.channels, fit.margins.prior_share, held, strict=True
                )
                if hold
            }
        if variant == RANGED:
            settings["planning range"] = (
                "each channel's box cut to the least and the most weekly spend its history logged"
            )
        return Planned(
            weekly=weekly,
            gain=gain,
            gain_interval=interval,
            flags=_flags(fit.margins, post, held, observation.channels),
            settings=settings,
            absent={
                "forecast": "the margin regression forecasts nothing",
                "response": (
                    "the margins are straight lines, and a response maps a geometric kernel"
                    " through a tanh or a Hill; the plan's regret and claim read none"
                ),
                "lift tests": UNREAD,
            },
        )


def _guarded(
    margins: MarginFit, shrunk: Shrunk | None, held: npt.NDArray[np.bool_]
) -> Shrunk | None:
    """The margins of the channels not `held` shrunk towards their own common level: `shrunk`
    where none is held, None where every one is or none of theirs is positive."""
    kept = ~held
    if kept.all():
        return shrunk
    if not kept.any():
        return None
    return shrink_margins(margins.mean[kept], margins.cov[np.ix_(kept, kept)])


def _curved(
    fit: Fitted,
    observation: Observation,
    quo: npt.NDArray[np.float64],
    weekly_budget: float,
    noise: NoiseMixture | None,
    *,
    guarded: bool,
) -> Planned:
    """`CURVED`'s plan, and `NOISE`'s and `HELD`'s under the mixture of error models `noise`: the
    curved regression's margins at the status quo shrunk and water-filled along each channel's
    curve inside the box cut to its history's spend and adstock, a channel the fit names spanned
    at its status quo, or the status quo where no margin is positive; `guarded`, `HELD`'s, a
    channel the time guard names at its status quo too, and the status quo where that leaves fewer
    than two channels free or no plan at the budget; see the module.

    Raises:
        ValueError: as `fit_curved_margins`, `adstock_range`, `time_held` and `water_fill` raise,
            `DegenerateFitError` and `ObservedRangeError` among them.
    """
    curved = fit_curved_margins(
        observation.sales,
        observation.spend,
        observation.controls.get("promotion"),
        observation.controls.get("price"),
        observation.kernel_length,
        quo,
        observation.planned,
        HALF_SATURATION,
        noise=noise,
    )
    spanned = curved.margins.spanned
    correlation = np.zeros(quo.size)
    timed = np.zeros(quo.size, dtype=bool)
    if guarded:
        correlation, named = time_held(observation.spend)
        timed = named & ~spanned
    held = spanned | timed
    post = shrink_margins(curved.margins.mean, curved.margins.cov)
    lower = upper = np.full(quo.size, math.nan)
    free = ~held
    weekly, gain, interval = quo.copy(), 0.0, (0.0, 0.0)
    if post is not None:
        lower, upper = adstock_range(
            observation.spend,
            curved.planning_adstock(),
            observation.lower,
            observation.upper,
            weekly_budget,
        )
        lower, upper = np.where(held, quo, lower), np.where(held, quo, upper)
        free = lower < upper
        if not timed.any() or (
            free.sum() >= FREE_LEAST and lower.sum() <= weekly_budget <= upper.sum()
        ):
            weekly = water_fill(
                post.mean, curved.curve.slope, curved.shaped, lower, upper, weekly_budget
            )
            rise = curved.curve.rise(weekly, quo)
            gain = observation.planned * float(post.mean @ rise)
            spread = observation.planned * math.sqrt(float(rise @ post.cov @ rise))
            z = NormalDist().inv_cdf((1.0 + GAIN_LEVEL) / 2.0)
            interval = (gain - z * spread, gain + z * spread)
    flags = _flags(curved.margins, post, np.zeros(quo.size, dtype=bool), observation.channels)
    top = np.argmax(curved.weights, axis=1)
    for c, name in enumerate(observation.channels):
        flags[f"{name} held"] = bool(held[c])
        flags[f"{name} half-saturation"] = float(curved.half_saturation[c])
        flags[f"{name} planning retention"] = RETENTIONS[int(top[c])]
        flags[f"{name} planning floor"] = float(lower[c])
        flags[f"{name} planning ceiling"] = float(upper[c])
    settings = fit.settings | {
        "planner": "water-filling",
        "weekly budget": "the status quo's weekly total",
        "half_saturation": HALF_SATURATION,
        "planned_weeks": observation.planned,
        "bisections": BISECTIONS,
        "price_span": PRICE_SPAN,
        "planning range": (
            "each channel's box cut to the least and the most weekly spend its history"
            " logged, then to the least and the most of its adstock at its retention of the"
            " most evidence weight; the first where the second holds no plan"
        ),
    }
    reasons = {
        name: (
            "the nuisance columns span its columns: under their flat prior no data identify its"
            " margin, which is 0"
        )
        for name, hold in zip(observation.channels, spanned, strict=True)
        if hold
    } | {
        name: (
            "its weekly spend's absolute correlation with the week's number is"
            f" {correlation[c]:.3f}, above {TIME_HELD}: a response that curves and one that"
            " drifts give its history alike"
        )
        for c, name in enumerate(observation.channels)
        if timed[c]
    }
    if noise is not None:
        flags["ar share"] = curved.margins.share
        flags["promoted variance"] = curved.margins.promoted
        settings |= {
            "trend": {"degree": DEGREE, "knot_every": noise.knot_every},
            "nuisance_prior": "flat",
            "ar": list(noise.persistence),
            "ar_share": list(noise.share),
            "promoted_variance": list(noise.promoted),
            "white_errors": noise.white,
            "spanned": SPANNED,
            "held at the status quo": reasons,
        }
    if guarded:
        for c, name in enumerate(observation.channels):
            flags[f"{name} time correlation"] = float(correlation[c])
        flags["free channels"] = int(free.sum())
        settings |= {"time_held": TIME_HELD, "free_least": FREE_LEAST}
    return Planned(
        weekly=weekly,
        gain=gain,
        gain_interval=interval,
        flags=flags,
        settings=settings,
        absent={
            "forecast": "the margin regression forecasts nothing",
            "response": (
                "each channel's curve mixes Michaelis-Menten curves of its adstock over its"
                " retentions, and a response maps one geometric kernel through a tanh or a Hill;"
                " the plan's regret and claim read none"
            ),
            "lift tests": UNREAD,
        },
    )


def _flags(
    margins: MarginFit,
    post: Shrunk | None,
    held: npt.NDArray[np.bool_],
    channels: tuple[str, ...],
) -> dict[str, float | int | bool]:
    """What a record keeps of the fit and of the variant's shrinkage `post` of the channels it
    did not hold; see the module."""
    sd = np.sqrt(np.diag(margins.cov))
    after = np.full(len(channels), math.nan)
    after_sd = np.full(len(channels), math.nan)
    if post is not None:
        after[~held] = post.mean
        after_sd[~held] = np.sqrt(np.diag(post.cov))
    flags: dict[str, float | int | bool] = {}
    for c, name in enumerate(channels):
        flags[f"{name} margin"] = float(margins.mean[c])
        flags[f"{name} margin sd"] = float(sd[c])
        flags[f"{name} shrunk margin"] = float(after[c])
        flags[f"{name} shrunk margin sd"] = float(after_sd[c])
        flags[f"{name} retention"] = float(margins.retention[c])
        flags[f"{name} prior share"] = float(margins.prior_share[c])
        flags[f"{name} held"] = bool(held[c])
    flags |= {
        "tau_rel": math.nan if post is None else post.tau,
        "phi": margins.phi,
        "events": len(margins.phases),
        "top weight": margins.top_weight,
        "rows": margins.rows,
        "burn-in rows": margins.burn_in,
    }
    return flags
