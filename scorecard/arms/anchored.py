"""The tested-rung rule GS(1.645) as an arm, `anchored`, and the route the scorecard's composite
`gs` reads (`scorecard.composite`). ADR 0088 records the rule and what it was measured at.

**The plan.** Each channel's margin, the KPI a unit of weekly spend returns, is read off the
export's history by the margin regression without its noise mixture (`mixdyn.models.margin_mmm`,
`noise=None`), as the `margin` arm reads it (`scorecard.arms.margin`); the export's lift tests are
read as each channel's return per euro and anchor the margins, the regression's bias read by
moments (`mixdyn.calibration.anchors`); the anchored margins are shrunk towards their common level
by empirical Bayes, and the plan is the corner a straight-line response makes of the shrunk means
(`mixdyn.decisions.margin_plan`). Where the regression raises
`mixdyn.models.margin_mmm.DegenerateFitError`, the tests alone are shrunk and planned, each
channel's return per euro of variance its standard error squared. At k = 0 the export holds no lift
row: the plan is the regression's alone, the `margin` arm's, and no world is routed. An export
where some channels have tests and others none is refused, and so is a geo export, its market too:
the rule reads one national history, and a geo export's tests are its geos'.

**The route.** The rule plays `stacked`'s plan, the engine's static Bayesian model
(`scorecard.arms.bayes`), where a straight line is not to be trusted: where some channel's tests
read a return per euro that rises with the spend, its slope statistic above `SLOPE_Z` (the convex
part of an S-curve), or where the export's kernel runs longer than the tests' readout, `READOUT`,
the dark weeks and the cooldown after them (`causaldyn_bench.lift_calibration.TEST` and `COOLDOWN`,
which every export of the bench states as `lift_weeks` and `lift_cooldown`): a test reads none of
the carryover past its readout, so its return per euro reads low. The arm never fits `stacked`. It
records the route among its flags, `routed`, with each reason beside it, `routed by slope` and
`routed by kernel`, and the composite `gs` plays `stacked`'s recorded plan where a world is routed
and this arm's everywhere else. The arm's own plan is the anchored corner in every world, routed or
not, so read alone it is the rule without its route.

The arm reads the export alone: its history, the controls `promotion` and `price`, the kernel's
length, the lift rows, the status quo, the box and the budget; an export without a control fits
without it, as the `margin` arm does. Its weekly budget is the status quo's weekly total, as the
rule was found with, and an export whose budget is another is refused. Where no margin is positive
the plan is the status quo. The record claims the plan's gain over the status quo on the shrunk
means with the normal interval of their covariance, as the `margin` arm's does, forecasts nothing
and records no response. Its flags: each channel's regression margin, its tests' return per euro
and the slope statistic, its anchored margin and its shrunk margin, each with its standard
deviation, NaN where it was not read, and its posterior mean retention; the regression's bias
`tau_b`, the largest slope statistic, the lift rows read, the route and its reasons, whether the
regression raised, and the regression's diagnostics as the `margin` arm flags them. Its settings
name every constant. The fit is a regression and its evidence, no sampler: a smoke run is the full
run.

**The curved variant**, `anchored curved` (ADR 0089), the rule GS-ML(0.7) without its route: the
same regression, tests, anchoring, shrinkage and route, and in the corner's place a plan
water-filled along each channel's margin curve. The curve is the shrunk margin at the status quo
times the posterior mixture of the curve columns the channel's tests and its regression margin
read (`mixdyn.calibration.anchors.read_curves`): Michaelis-Menten and logistic curves on a grid of
elasticities, and the straight line, of prior mass `LINE_PRIOR`. The regression is read as a linear
map at its retention combination of the most evidence (`mixdyn.models.margin_mmm.project`), and
each test by the exact lift a curve gives its `TEST` dark weeks over its `READOUT` weeks. The plan
is the tabulated water-filling's (`mixdyn.decisions.margin_plan.water_fill_tabulated`), the planner
the rule was found with. Where the export holds no lift row or the regression raised there is no
curve to read, and the plan is `anchored`'s corner. The curves are read in the variant's plan, so
their cost is its own. Its record claims the plan's gain along the curves, `planned sum_c m_c
(H_c(p_c) - H_c(s_c))`, `H_c` the integral of the channel's margin over its level, with the normal
interval of the shrunk margins' covariance; its flags add each channel's posterior weight on the
line and its posterior mean elasticity at the status quo, NaN where no curve was read; its settings
add the line's prior, the grid and the planner. The composite `gs curved` reads its route.

**The held variant**, `anchored held` (ADR 0093), `anchored curved` under the time guard
(`mixdyn.decisions.margin_plan.time_held`): a channel whose weekly spend correlates with the
week's number above `TIME_HELD` in absolute value, over the export's whole history, is held at its
status quo. A go-dark test reads such a channel's whole contribution; the margin at the status quo
needs its curve, and where the spend is a function of the time a curved response and a drifting
one give the same history. With no channel held the record is `anchored curved`'s: the plan, the
claim and the route. With a channel held, its floor and its ceiling are its status quo, and the
other channels are planned at the budget in their box, by the water-filling, or the corner where
no curve is read; the claim is that plan's. With a channel held the plan is the status quo in
three cases: where fewer than `FREE` channels are free, a free channel being neither held nor
boxed to a point; where the box so cut holds no plan at the budget; and where the fit's readings
route the world. `stacked`'s plan holds no channel, so a world with a held channel is never
routed: its record's `routed` is false. The guard reads the export's spend alone, no margin. Its
flags add each channel's correlation with the week's number (`time correlation`), whether any
channel is held (`time held`) and whether a route was withheld (`route withheld`), the route's two
reasons as the fit read them; its settings add the threshold and, for each channel held, its
correlation. The composite `gs held` reads its route.

**The floored variant**, `anchored floored` (ADR 0099), `anchored held` with no route and a floor
under each free channel (`_floored`): a channel the time guard leaves free gets the larger of its
floor and `SPEND_FLOOR` times the least weekly spend of the export's whole history, at most its
ceiling; where the floors so raised do not fit the budget, the guard's box stands. Below the
history's range the export holds no week to read a channel's response from, and a response may
fall off there. Everything else is `anchored held`'s: where no floor is raised the plan, its claim
and its interval are that variant's to the last bit, and the plan is the status quo in the guard's
three cases. No composite reads it: its record's `routed` is false on every world, the route's
reasons kept as the fit read them. Its flags add whether a raised floor cut the box the plan was
searched in (`floor applied`); its settings add the constant (`spend_floor`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import TYPE_CHECKING, Any

import numpy as np
from causaldyn_bench.lift_calibration import COOLDOWN, TEST

from mixdyn.calibration import anchors
from mixdyn.decisions.margin_plan import (
    HALF_NORMAL_SCALE,
    POINTS,
    PRICE_BISECTIONS,
    TAU_GRID,
    TIME_HELD,
    Shrunk,
    plan_corner,
    shrink_margins,
    time_held,
    water_fill_tabulated,
)
from mixdyn.models.margin_mmm import (
    AR_MOST,
    BURN_IN,
    DEGREE,
    HARMONICS,
    KNOT_EVERY,
    PRESAMPLE,
    PRICE_DROP,
    PRIOR,
    RETENTIONS,
    YEAR,
    DegenerateFitError,
    MarginFit,
    fit_margins,
    project,
)
from scorecard.arms.base import Planned
from scorecard.arms.margin import CONTROLS, GAIN_LEVEL, GEO, SPENT
from scorecard.arms.panel import Market
from scorecard.composite import GS, GS_CURVED, GS_HELD, ROUTED

if TYPE_CHECKING:
    from causaldyn_bench.scorecard.observe import Observation

VARIANT = GS.router
CURVED = GS_CURVED.router
HELD = GS_HELD.router
FLOORED = "anchored floored"
SPEND_FLOOR = 0.9
"""The share of a free channel's least weekly spend over the export's history that `FLOORED`
plans it at, at least: the plan stays where the history saw spend, since a channel's response may
fall off below that range and no week of the history reads it there."""
LINE_PRIOR = anchors.LINE_PRIOR
"""The prior's mass on the straight line among a channel's curve columns, in `CURVED`, `HELD` and
`FLOORED`."""
FREE = 2
"""The fewest free channels `HELD` plans: with one, the budget leaves it at its status quo."""
SLOPE_Z = 1.645
"""The slope statistic past which a channel's tests read a convex curve: one-sided 5 %."""
READOUT = TEST + COOLDOWN
"""The weeks a test's readout counts, its dark weeks and its cooldown: a kernel past it routes."""
NOT_READ = math.nan


@dataclass(frozen=True, slots=True, eq=False)
class Fitted:
    """What the plan reads: the shrunk posterior, None where no margin is positive, the
    regression's margins, None where it raised, and what a record keeps of the fit, the route
    among its flags."""

    shrunk: Shrunk | None
    margins: MarginFit | None
    flags: dict[str, float | int | bool]
    settings: dict[str, Any]
    absent: dict[str, str]


@dataclass(frozen=True, slots=True, eq=False)
class Guard:
    """What the time guard reads of an export (`_guard`): each channel's absolute `correlation`
    of its weekly spend with the week's number and the channels `held`; the box `HELD` plans in,
    `lower` and `upper`, the export's with each held channel at its status quo; whether the plan
    is the status quo, `still`; and whether a route the fit read was `withheld`."""

    correlation: np.ndarray
    held: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    still: bool
    withheld: bool


class Anchored:
    """The arm; see the module."""

    variants = (VARIANT, CURVED, HELD, FLOORED)
    tests = True
    cells = False

    def fit(self, observation: Observation, *, smoke: bool) -> Fitted:
        """The regression, the tests, their anchoring, the shrinkage and the route; `smoke`
        changes nothing.

        Raises:
            ValueError: on a geo export or its market (`scorecard.arms.margin.GEO`), as
                `mixdyn.calibration.anchors.read_tests` raises, and as `fit_margins` raises but
                for its `DegenerateFitError` where the export holds tests.
        """
        if isinstance(observation, Market) or observation.population is not None:
            raise ValueError(GEO)
        lift = observation.lift
        rows = len(lift.channel)
        tests = (
            anchors.read_tests(
                observation.channels,
                observation.status_quo,
                tuple(lift.channel),
                lift.x,
                lift.delta_y,
                lift.sigma,
            )
            if rows
            else None
        )
        absent = {}
        try:
            margins: MarginFit | None = fit_margins(
                observation.sales,
                observation.spend,
                observation.controls.get("promotion"),
                observation.controls.get("price"),
                observation.kernel_length,
                noise=None,
            )
        except DegenerateFitError as error:
            if tests is None:
                raise
            margins = None
            absent["margins"] = f"the regression raised, DegenerateFitError: {error}"
        anchored = None
        if tests is None:
            assert margins is not None  # without tests a raise is the fit's
            shrunk = shrink_margins(margins.mean, margins.cov)
            absent["lift tests"] = (
                "the export holds no lift row: the plan is the regression's alone, the margin"
                " arm's, and no world is routed"
            )
        elif margins is None:
            shrunk = shrink_margins(tests.roi, np.diag(tests.roi_se**2))
        else:
            anchored = anchors.anchor(margins.mean, margins.cov, tests.roi, tests.roi_se)
            shrunk = shrink_margins(anchored.mean, anchored.cov)
        by_slope = tests is not None and float(tests.slope_t.max()) > SLOPE_Z
        by_kernel = tests is not None and observation.kernel_length > READOUT
        flags = _flags(observation.channels, margins, tests, anchored, shrunk)
        flags |= {
            "lift rows": rows,
            ROUTED: by_slope or by_kernel,
            "routed by slope": by_slope,
            "routed by kernel": by_kernel,
            "regression raised": margins is None,
        }
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
            "controls": [c for c in CONTROLS if c in observation.controls],
            "lift_rows_read": rows,
            "slope_z": SLOPE_Z,
            "flat_spread": anchors.FLAT,
            "readout_weeks": {"dark": TEST, "cooldown": COOLDOWN},
            "routes_to": GS.routed,
        }
        return Fitted(shrunk, margins, flags, settings, absent)

    def plan(self, fit: Fitted, variant: str, observation: Observation) -> Planned:
        """The corner on the shrunk means, or the status quo where no margin is positive; in
        `CURVED`, the plan water-filled along the curves the tests read, the corner where there is
        none to read; in `HELD`, `CURVED`'s plan in the box the time guard leaves (`_guard`), or
        the status quo where it leaves none; in `FLOORED`, `HELD`'s plan with the free channels'
        floors raised (`_floored`), never routed.

        Raises:
            ValueError: on a budget that is not the quarter's weeks times the status quo's weekly
                total, as `plan_corner` raises, in `CURVED`, `HELD` and `FLOORED` as `_curves` and
                `water_fill_tabulated` raise, and in `HELD` and `FLOORED` as `time_held` raises.
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
        curves = _curves(fit, observation) if variant in (CURVED, HELD, FLOORED) else None
        guard = _guard(fit, observation, quo, weekly_budget) if variant in (HELD, FLOORED) else None
        lower, upper = observation.lower, observation.upper
        if guard is not None:
            lower, upper = guard.lower, guard.upper
        applied = False
        if fit.shrunk is None or (guard is not None and guard.still):
            weekly, gain, interval = quo.copy(), 0.0, (0.0, 0.0)
        else:
            if variant == FLOORED and guard is not None:
                lower = _floored(guard, observation.spend, weekly_budget)
                applied = bool(np.any(lower != guard.lower))
            level = fit.shrunk.mean
            if curves is None:
                weekly = plan_corner(level, lower, upper, weekly_budget)
                move = weekly - quo
            else:
                weekly = water_fill_tabulated(
                    [curves.margin(c, float(level[c])) for c in range(quo.size)],
                    lower,
                    upper,
                    weekly_budget,
                )
                move = curves.rise(weekly, level)
            gain = observation.planned * float(level @ move)
            spread = observation.planned * math.sqrt(float(move @ fit.shrunk.cov @ move))
            z = NormalDist().inv_cdf((1.0 + GAIN_LEVEL) / 2.0)
            interval = (gain - z * spread, gain + z * spread)
        flags, settings = fit.flags, fit.settings
        planner, response = "corner", "the margins are straight lines"
        if variant in (CURVED, HELD, FLOORED):
            weight = np.full(quo.size, NOT_READ) if curves is None else curves.weights[:, -1]
            elasticity = np.full(quo.size, NOT_READ) if curves is None else curves.elasticity()
            flags = flags | {
                f"{name} {label}": float(values[c])
                for c, name in enumerate(observation.channels)
                for label, values in (("line weight", weight), ("elasticity", elasticity))
            }
            settings = settings | {
                "line_prior": LINE_PRIOR,
                "elasticities": [float(e) for e in anchors.ELASTICITIES],
                "curve families": ["michaelis-menten", "logistic"],
                "table_points": POINTS,
                "price_bisections": PRICE_BISECTIONS,
            }
            if curves is not None:
                planner = "water-filling on a table"
                response = (
                    "each channel's margin is a mixture of Michaelis-Menten and logistic curves"
                    " and the line"
                )
        if guard is not None:
            named = list(zip(observation.channels, guard.correlation, guard.held, strict=True))
            flags = flags | {f"{name} time correlation": float(r) for name, r, _ in named}
            flags = flags | {
                "time held": bool(guard.held.any()),
                ROUTED: bool(fit.flags[ROUTED]) and not guard.withheld,
                "route withheld": guard.withheld,
            }
            settings = settings | {
                "time_held": TIME_HELD,
                "held at the status quo": {
                    name: (
                        "its weekly spend's correlation with the week's number is"
                        f" {r:.3f} in absolute value, above {TIME_HELD}"
                    )
                    for name, r, hold in named
                    if hold
                },
            }
        if variant == FLOORED:
            flags = flags | {ROUTED: False, "floor applied": applied}
            settings = settings | {"spend_floor": SPEND_FLOOR}
        return Planned(
            weekly=weekly,
            gain=gain,
            gain_interval=interval,
            flags=flags,
            settings=settings
            | {"planner": planner, "weekly budget": "the status quo's weekly total"},
            absent={
                "forecast": "the margin regression forecasts nothing",
                "response": (
                    f"{response}, and a response maps a geometric kernel"
                    " through a tanh or a Hill; the plan's regret and claim read none"
                ),
            }
            | fit.absent,
        )


def _guard(fit: Fitted, observation: Observation, quo: np.ndarray, weekly_budget: float) -> Guard:
    """The time guard's reading of an export for `HELD`; see the module. It reads the export's
    spend, its box and its status quo, and of the fit its route alone, no margin.

    Raises:
        ValueError: as `mixdyn.decisions.margin_plan.time_held` raises.
    """
    correlation, held = time_held(observation.spend)
    lower, upper = observation.lower, observation.upper
    if not held.any():
        return Guard(correlation, held, lower, upper, still=False, withheld=False)
    lower, upper = np.where(held, quo, lower), np.where(held, quo, upper)
    withheld = bool(fit.flags[ROUTED])
    free = int(np.count_nonzero(upper > lower))
    spends = bool(np.all(lower <= upper)) and bool(lower.sum() <= weekly_budget <= upper.sum())
    still = withheld or free < FREE or not spends
    return Guard(correlation, held, lower, upper, still=still, withheld=withheld)


def _floored(guard: Guard, spend: np.ndarray, weekly_budget: float) -> np.ndarray:
    """`FLOORED`'s floors: each channel the guard leaves free at the larger of its floor and
    `SPEND_FLOOR` times its least weekly spend over the export's whole history `spend`, at most
    its ceiling; a held channel at its status quo. Where the floors so raised do not fit
    `weekly_budget` in the guard's ceilings, the guard's floors. A floor is never lowered: a free
    channel whose floor lies above its ceiling keeps it, and the box holds no plan, as in `HELD`."""
    least = SPEND_FLOOR * np.asarray(spend, dtype=float).min(axis=0)
    lower, upper = guard.lower, guard.upper
    raised = np.where(guard.held, lower, np.maximum(lower, np.minimum(least, upper)))
    fits = bool(np.all(raised <= upper)) and bool(raised.sum() <= weekly_budget <= upper.sum())
    return raised if fits else lower


def _curves(fit: Fitted, observation: Observation) -> anchors.Curves | None:
    """Each channel's posterior over its curve columns, read off the export's tests and the
    regression's margins; None where the export holds no lift row, the regression raised or no
    margin is positive, so that no curve is read or planned on.

    Raises:
        ValueError: as `mixdyn.models.margin_mmm.project` and
            `mixdyn.calibration.anchors.read_curves` raise.
    """
    lift = observation.lift
    margins = fit.margins
    if not len(lift.channel) or margins is None or fit.shrunk is None:
        return None
    regression = project(
        observation.sales,
        observation.spend,
        observation.controls.get("promotion"),
        observation.controls.get("price"),
        observation.kernel_length,
        margins.top,
    )
    return anchors.read_curves(
        observation.channels,
        observation.status_quo,
        tuple(lift.channel),
        lift.x,
        lift.delta_y,
        lift.sigma,
        lift.start,
        margins.mean,
        margins.cov,
        observation.spend,
        regression.slope,
        retention=regression.retention,
        kernel=observation.kernel_length,
        dark=TEST,
        readout=READOUT,
        line_prior=LINE_PRIOR,
    )


def _flags(
    channels: tuple[str, ...],
    margins: MarginFit | None,
    tests: anchors.ChannelTests | None,
    anchored: anchors.Anchored | None,
    shrunk: Shrunk | None,
) -> dict[str, float | int | bool]:
    """What a record keeps of the readings, NaN where one was not read; see the module."""
    count = len(channels)
    unread = np.full(count, NOT_READ)

    def sd(cov: np.ndarray | None) -> np.ndarray:
        return unread if cov is None else np.sqrt(np.diag(cov))

    columns = {
        "margin": (unread, None) if margins is None else (margins.mean, margins.cov),
        "test roi": (unread, None) if tests is None else (tests.roi, np.diag(tests.roi_se**2)),
        "anchored margin": (unread, None) if anchored is None else (anchored.mean, anchored.cov),
        "shrunk margin": (unread, None) if shrunk is None else (shrunk.mean, shrunk.cov),
    }
    slope = unread if tests is None else tests.slope_t
    retention = unread if margins is None else margins.retention
    flags: dict[str, float | int | bool] = {}
    for c, name in enumerate(channels):
        for label, (mean, cov) in columns.items():
            flags[f"{name} {label}"] = float(mean[c])
            flags[f"{name} {label} sd"] = float(sd(cov)[c])
        flags[f"{name} roi slope t"] = float(slope[c])
        flags[f"{name} retention"] = float(retention[c])
    flags |= {
        "tau_b": NOT_READ if anchored is None else anchored.tau,
        "max roi slope t": NOT_READ if tests is None else float(tests.slope_t.max()),
        "tau_rel": NOT_READ if shrunk is None else shrunk.tau,
    }
    if margins is not None:
        flags |= {
            "phi": margins.phi,
            "events": len(margins.phases),
            "top weight": margins.top_weight,
            "rows": margins.rows,
            "burn-in rows": margins.burn_in,
        }
    return flags
