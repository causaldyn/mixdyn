"""The rules the benchmark scored, each a planner of one call: `plan_margins`,
`plan_anchored_margins` and `plan_pooled_geos`.

Each planner fits and plans as the scorecard's arm whose records a pre-registered run read, and
returns what the arm planned: on the arms' worlds the plan is the arm's to the last bit. The arms
read the bench's export; a planner reads a `MediaPanel`, the channels' `Range`s and, where the rule
reads them, `LiftEstimate`s. Each docstring names the arm its record was made by and states the
record, which `docs/benchmark.md` holds whole: every comparison, its interval and every loss.

**What the record covers.** The runs planned a quarter of 13 weeks after a weekly history, each
channel between half and twice its status quo, at a budget equal to the status quo's, and read
`promotion` and `price` as the controls. A planner refuses what its rule does not read: a panel that
is not weekly, and, for the two national rules, a panel of more than one geo, a control by another
name, a filled week or an event. It accepts another budget and another box, which no run scored.

**The status quo** is each channel's mean weekly spend over the panel's last `STATUS_QUO_WEEKS`
weeks, as the bench's is. A `budget` of None spends its weekly total each week: the budget the
record covers.

**A lift test** of `plan_anchored_margins` is a go-dark of one channel over the whole market: its
`spend` is 0 over its dark weeks, which open its readout, and its `planned` elsewhere; its `end` is
a date of the panel's. It is read as the rule reads a test: the planned spend's mean over the dark
weeks, the lift and its standard error over the dark weeks, and the week it went dark.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from mixdyn.calibration import anchors
from mixdyn.data import Cadence
from mixdyn.decisions.geo_plan import plan_mixture
from mixdyn.decisions.margin_plan import (
    adstock_range,
    plan_corner,
    shrink_margins,
    time_held,
    water_fill,
    water_fill_tabulated,
)
from mixdyn.models.geo_eb import fit_pooled_geos
from mixdyn.models.margin_mmm import (
    HALF_SATURATION,
    DegenerateFitError,
    NoiseMixture,
    fit_curved_margins,
    fit_margins,
    project,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mixdyn.calibration import LiftEstimate
    from mixdyn.data import Channel, Geo, MediaPanel
    from mixdyn.decisions.geo_plan import GeoLimits, Range
    from mixdyn.decisions.margin_plan import Shrunk

type _Array = npt.NDArray[np.float64]

STATUS_QUO_WEEKS = 52
"""The weeks at the end of the history whose mean weekly spend is a channel's status quo."""
SPEND_FLOOR = 0.9
"""`plan_anchored_margins`' default floor, a share of a free channel's least weekly spend over the
history: the setting the record was scored with."""
GAIN_LEVEL = 0.9
"""The central share of a claimed gain's normal interval."""
FREE_LEAST = 2
"""The fewest channels a planner moves where the time guard holds one: with fewer, the budget
leaves nothing to move, and the plan is the status quo."""
SLOPE_Z = 1.645
"""The slope statistic past which a channel's tests read a convex curve, one-sided 5 %
(`mixdyn.calibration.anchors.read_tests`)."""
CONTROLS = ("promotion", "price")
"""The controls the margin regression reads, by name."""
MARGIN_HELD = "margin held"
"""The arm whose records `plan_margins` reproduces."""
ANCHORED_FLOORED = "anchored floored"
"""The arm whose records `plan_anchored_margins` reproduces at its default floor."""
ANCHORED_HELD = "anchored held"
"""The arm whose records `plan_anchored_margins` reproduces with no floor."""
GEO_EB = "geo eb"
"""The arm whose records `plan_pooled_geos` reproduces."""


@dataclass(frozen=True, slots=True, eq=False)
class MarginPlan:
    """A plan of one national history's channels by a margin rule (`plan_margins`,
    `plan_anchored_margins`), and the status quo it moves.

    `spend` and `current` are `(channels,)` in the panel's order, in spend a week: the plan's, and
    the status quo. `budget` is what the plan spends over its `periods` weeks. `margins` are the
    margins the plan ranks, each the KPI a unit of weekly spend returns at the status quo, and
    `cov` their covariance: shrunk towards their common level where one is positive, and the fit's
    own where none is, when the plan is the status quo. `gain` is what the rule claims the plan
    adds to the KPI over the status quo's, over the periods and the kernel's tail, at the margins'
    means, and `interval` its central `GAIN_LEVEL` normal interval; both are 0 where the plan is
    the status quo. `held` names the channels held at their status quo. `record` names the arm of
    the benchmark's records the plan reproduces, None for a setting no run recorded.
    """

    channels: tuple[Channel, ...]
    periods: int
    spend: _Array
    current: _Array
    budget: float
    margins: _Array
    cov: _Array
    gain: float
    interval: tuple[float, float]
    held: tuple[Channel, ...]
    record: str | None


@dataclass(frozen=True, slots=True, eq=False)
class PooledGeoPlan:
    """A plan of a geo panel's cells by the model over geos (`plan_pooled_geos`), and the status
    quo it moves.

    `cells` are each geo's own channels, geo by geo in the panel's order, then each channel bought
    nationally once, in no geo (None). `spend` and `current` are `(cells,)`, in spend a week: the
    plan's, and the status quo, a national cell's summed over the geos. `budget` is what the plan
    spends over its `periods` weeks, and `price` the slope of expected worth at which the cells
    spend it. The rule claims no gain (ADR 0090). `record` names the arm of the benchmark's records
    the plan reproduces.
    """

    cells: tuple[tuple[Geo | None, Channel], ...]
    periods: int
    spend: _Array
    current: _Array
    budget: float
    price: float
    record: str | None


@dataclass(frozen=True, slots=True)
class _National:
    """What a margin rule reads of a panel of one geo: the sales, the spend `(weeks, channels)`,
    the controls it reads or None, the status quo, and the box."""

    sales: _Array
    spend: _Array
    promotion: _Array | None
    price: _Array | None
    status_quo: _Array
    lower: _Array
    upper: _Array


@dataclass(frozen=True, slots=True)
class _Rows:
    """The lift tests as the rule reads them, one row a test: its channel, its planned spend's mean
    over the dark weeks `x`, its lift `delta_y` and standard error `sigma` over the dark weeks, and
    the week it went dark, counted from 1; the dark weeks and the readout every test shares."""

    channel: tuple[str, ...]
    x: _Array
    delta_y: _Array
    sigma: _Array
    start: _Array
    dark: int
    readout: int


def plan_margins(
    panel: MediaPanel,
    budget: float | None,
    periods: int,
    limits: Mapping[Channel, Range],
    *,
    kernel: int,
) -> MarginPlan:
    """The plan of the panel's channels over `periods` weeks that spends `budget` over them within
    `limits`, each channel's range a week, by the margin rule with no lift test.

    It reads one national weekly history, its sales, its spend and the controls `promotion` and
    `price` where the panel holds them, and no lift test. Each channel's margin at the status quo
    is read off the curved margin regression under its mixture of error models
    (`mixdyn.models.fit_margins` reads the straight one), each column a Michaelis-Menten curve of
    the channel's adstock through a kernel of `kernel` weeks, half-saturated at 16 times its status
    quo. The margins are shrunk towards their common level, and the budget is water-filled along
    each channel's curve, each box cut to the range of spend and adstock the history logged. A
    channel whose weekly spend follows the week's number, or whose columns the regression's
    nuisance spans, is held at its status quo (`mixdyn.decisions.margin_plan`). A `budget` of None
    spends the status quo's weekly total each week.

    Its record, as `margin held`: within δ = 0.05 of every rival on 15 of 16 family-rungs (M4 11,
    M5 4); not within δ on family 4 at k = 0, against PyMC-Marketing and Orbit. Every rung it
    planned was a national one without lift tests, at the status quo's budget; on family 4,
    S-curves, the data do not identify a curve past the range its spend was logged in.

    Raises:
        ValueError: on a panel that is not weekly, of more than one geo, with a control other than
            `promotion` and `price`, a filled week or an event; on a range missing for a channel,
            named for one the panel does not have or with no finite most; on `periods` not a whole
            number of weeks, at least 1, or a budget that is not finite; and as the regression and
            the planner raise, `DegenerateFitError` and a box that holds no plan at the budget
            among them.
        MemoryError: as `fit_margins` raises.
    """
    data = _national(panel, limits, periods)
    weekly = _weekly(budget, periods, data.status_quo)
    quo = data.status_quo
    curved = fit_curved_margins(
        data.sales,
        data.spend,
        data.promotion,
        data.price,
        kernel,
        quo,
        periods,
        HALF_SATURATION,
        noise=NoiseMixture(),
    )
    spanned = curved.margins.spanned
    timed = time_held(data.spend)[1] & ~spanned
    held = spanned | timed
    post = shrink_margins(curved.margins.mean, curved.margins.cov)
    plan, gain, interval = quo.copy(), 0.0, (0.0, 0.0)
    if post is not None:
        lower, upper = adstock_range(
            data.spend, curved.planning_adstock(), data.lower, data.upper, weekly
        )
        lower, upper = np.where(held, quo, lower), np.where(held, quo, upper)
        free = lower < upper
        if not timed.any() or (free.sum() >= FREE_LEAST and lower.sum() <= weekly <= upper.sum()):
            plan = water_fill(post.mean, curved.curve.slope, curved.shaped, lower, upper, weekly)
            gain, interval = _claim(post, curved.curve.rise(plan, quo), periods)
    margins = curved.margins if post is None else post
    return MarginPlan(
        channels=panel.channels,
        periods=periods,
        spend=plan,
        current=quo,
        budget=_total(budget, periods, weekly),
        margins=margins.mean,
        cov=margins.cov,
        gain=gain,
        interval=interval,
        held=tuple(c for c, h in zip(panel.channels, held, strict=True) if h),
        record=MARGIN_HELD,
    )


def plan_anchored_margins(
    panel: MediaPanel,
    estimates: Mapping[str, Sequence[LiftEstimate]],
    budget: float | None,
    periods: int,
    limits: Mapping[Channel, Range],
    *,
    kernel: int,
    spend_floor: float | None = SPEND_FLOOR,
) -> MarginPlan:
    """The plan of the panel's channels over `periods` weeks that spends `budget` over them within
    `limits`, each channel's range a week, by the margin rule anchored on each channel's lift
    tests, `estimates` by channel.

    It reads one national weekly history, as `plan_margins` does, and at least one go-dark test of
    every channel (the module). Each channel's margin is read off the straight margin regression
    without its noise mixture, anchored on the tests' returns per euro, its bias read by moments,
    and shrunk towards the channels' common level (`mixdyn.calibration.anchors`); where the data
    leave the regression degenerate, the tests alone are shrunk. The plan is water-filled along each
    channel's margin curve, a mixture of Michaelis-Menten and logistic curves and the line that the
    tests and the regression read, or the corner on the margins where no curve is read. A channel
    whose weekly spend follows the week's number is held at its status quo. With one held, the
    plan is the status quo where fewer than `FREE_LEAST` channels are free, where the box holds no
    plan, and where the tests read a convex curve or a kernel past their readout. Each free
    channel's floor is raised to `spend_floor` times its least weekly spend over the history, at
    most its ceiling, where the raised floors leave a plan; None raises none. A `budget` of None
    spends the status quo's weekly total each week.

    Its record, as `anchored floored`: within δ = 0.05 of every rival on 16 of 16 family-rungs
    (M4 15, M5 1). Every rung it planned was a national one with every channel tested, at the
    status quo's budget, with the floor at `SPEND_FLOOR`; only that share was scored, and a share of
    1 failed its pre-registered study's bar. With `spend_floor=None` the plan is `anchored held`'s,
    which no run claimed.

    Raises:
        ValueError: as `plan_margins` raises on the panel, the ranges, the periods and the budget;
            on a channel without a test or a test of a channel the panel does not have; on a test
            that is not a go-dark of the whole market whose dark weeks open its readout, ends on
            no date of the panel's or opens before it; on tests that went dark or read for
            different weeks; on a floor outside (0, 1]; and as the regression, the anchoring and
            the planners raise.
    """
    data = _national(panel, limits, periods)
    if spend_floor is not None and not 0.0 < spend_floor <= 1.0:
        msg = f"a spend floor is a share in (0, 1] of the least weekly spend, not {spend_floor}"
        raise ValueError(msg)
    rows = _rows(panel, estimates)
    weekly = _weekly(budget, periods, data.status_quo)
    quo = data.status_quo
    names = tuple(str(c) for c in panel.channels)
    tests = anchors.read_tests(names, quo, rows.channel, rows.x, rows.delta_y, rows.sigma)
    try:
        margins = fit_margins(
            data.sales, data.spend, data.promotion, data.price, kernel, noise=None
        )
    except DegenerateFitError:
        margins = None
    if margins is None:
        fitted = (tests.roi, np.diag(tests.roi_se**2))
    else:
        anchored = anchors.anchor(margins.mean, margins.cov, tests.roi, tests.roi_se)
        fitted = (anchored.mean, anchored.cov)
    shrunk = shrink_margins(*fitted)
    routed = float(tests.slope_t.max()) > SLOPE_Z or kernel > rows.readout
    curves = None
    if margins is not None and shrunk is not None:
        regression = project(
            data.sales, data.spend, data.promotion, data.price, kernel, margins.top
        )
        curves = anchors.read_curves(
            names,
            quo,
            rows.channel,
            rows.x,
            rows.delta_y,
            rows.sigma,
            rows.start,
            margins.mean,
            margins.cov,
            data.spend,
            regression.slope,
            retention=regression.retention,
            kernel=kernel,
            dark=rows.dark,
            readout=rows.readout,
            line_prior=anchors.LINE_PRIOR,
        )
    held = time_held(data.spend)[1]
    lower, upper, still = data.lower, data.upper, False
    if held.any():
        lower, upper = np.where(held, quo, lower), np.where(held, quo, upper)
        free = int(np.count_nonzero(upper > lower))
        spends = bool(np.all(lower <= upper)) and bool(lower.sum() <= weekly <= upper.sum())
        still = routed or free < FREE_LEAST or not spends
    plan, gain, interval = quo.copy(), 0.0, (0.0, 0.0)
    if shrunk is not None and not still:
        if spend_floor is not None:
            lower = _floored(spend_floor, held, lower, upper, data.spend, weekly)
        level = shrunk.mean
        if curves is None:
            plan = plan_corner(level, lower, upper, weekly)
            move = plan - quo
        else:
            plan = water_fill_tabulated(
                [curves.margin(c, float(level[c])) for c in range(quo.size)], lower, upper, weekly
            )
            move = curves.rise(plan, level)
        gain, interval = _claim(shrunk, move, periods)
    mean, cov = fitted if shrunk is None else (shrunk.mean, shrunk.cov)
    record = {SPEND_FLOOR: ANCHORED_FLOORED, None: ANCHORED_HELD}.get(spend_floor)
    return MarginPlan(
        channels=panel.channels,
        periods=periods,
        spend=plan,
        current=quo,
        budget=_total(budget, periods, weekly),
        margins=mean,
        cov=cov,
        gain=gain,
        interval=interval,
        held=tuple(c for c, h in zip(panel.channels, held, strict=True) if h),
        record=record,
    )


def plan_pooled_geos(
    panel: MediaPanel,
    estimates: Mapping[str, Sequence[LiftEstimate]],
    budget: float | None,
    periods: int,
    limits: GeoLimits,
    *,
    kernel: int,
) -> PooledGeoPlan:
    """The plan of the panel's cells over `periods` weeks that spends `budget` over them, each
    cell within its range a week in `limits`, by the closed-form model over geos fitted with the
    geo tests `estimates`, by channel, if any.

    It reads a weekly geo panel with each geo's population and no control, and any geo tests
    (`mixdyn.models.fit_pooled_geos`): each channel's curve of spend per head shared by the geos,
    its margin in each geo pooled about the channel's, each channel's posterior over a grid of its
    carryover and curvature, a kernel of `kernel` weeks. Each cell's expected worth under the
    posterior is water-filled (`mixdyn.decisions.geo_plan.plan_mixture`). A geo's own channel is
    one cell of each geo, ranged in `limits.cells`; a channel bought nationally one cell of its
    national spend, ranged in `limits.national`. A `budget` of None spends the cells' status quo's
    weekly total each week.

    Its record, as `geo eb`: within δ = 0.05 of every rival on 6 of 6 family-rungs (M3 6). Every
    rung it planned was a geo family's, with and without geo tests, at the status quo's budget, on
    M3's worlds alone.

    Raises:
        ValueError: on a panel that is not weekly; on limits that bound a geo's or a channel's
            total, which the rule does not read, or that miss a cell's range, name one the panel
            does not have or have no finite most; on `periods` not a whole number of weeks, at
            least 1, or a budget that is not finite; and as `fit_pooled_geos` and the planner
            raise.
    """
    if panel.cadence is not Cadence.WEEK:
        msg = f"the rule reads a weekly history, not one of cadence {panel.cadence.value}"
        raise ValueError(msg)
    _whole(periods)
    if limits.geos or limits.channels:
        msg = "the rule plans each cell within its own range; it reads no geo's or channel's total"
        raise ValueError(msg)
    own = [c for c in panel.channels if c not in panel.national]
    national = [c for c in panel.channels if c in panel.national]
    cells: list[tuple[Geo | None, Channel]] = [(g, c) for g in panel.geos for c in own]
    cells += [(None, c) for c in national]
    ranges = [limits.national.get(c) if g is None else limits.cells.get((g, c)) for g, c in cells]
    missing = [cell for cell, r in zip(cells, ranges, strict=True) if r is None]
    extra = [
        *(k for k in limits.cells if k not in cells),
        *(c for c in limits.national if c not in national),
    ]
    if missing or extra:
        msg = f"every cell needs a range a week: missing {missing}, no such cell {extra}"
        raise ValueError(msg)
    lower = np.array([r.least for r in ranges if r is not None])
    upper = np.array([r.most for r in ranges if r is not None])
    if not np.all(np.isfinite(upper)):
        msg = "every cell's range needs a finite most"
        raise ValueError(msg)
    history = np.column_stack(
        [
            panel.spend[:, :, panel.channels.index(c)].sum(axis=0)
            if g is None
            else panel.spend[panel.geos.index(g), :, panel.channels.index(c)]
            for g, c in cells
        ]
    )
    current = history[-STATUS_QUO_WEEKS:].mean(axis=0)
    total = periods * float(current.sum()) if budget is None else _finite(budget)
    fit = fit_pooled_geos(panel, estimates, kernel)
    plan = plan_mixture([fit.cell(g, c) for g, c in cells], lower, upper, total, periods)
    return PooledGeoPlan(
        cells=tuple(cells),
        periods=periods,
        spend=plan.weekly,
        current=current,
        budget=total,
        price=plan.price,
        record=GEO_EB,
    )


def _national(panel: MediaPanel, limits: Mapping[Channel, Range], periods: int) -> _National:
    """What a margin rule reads of `panel`, checked to be what its record covers (the module)."""
    if panel.cadence is not Cadence.WEEK:
        msg = f"the rule reads a weekly history, not one of cadence {panel.cadence.value}"
        raise ValueError(msg)
    if len(panel.geos) != 1:
        msg = (
            f"the rule reads one national history, not {len(panel.geos)} geos: plan a geo panel"
            " with plan_pooled_geos"
        )
        raise ValueError(msg)
    other = [c for c in panel.controls if c not in CONTROLS]
    if other or bool(panel.filled.any()) or panel.events:
        msg = (
            f"the rule reads the controls {list(CONTROLS)} alone, every week observed and no event:"
            f" not the controls {other}, {int(panel.filled.sum())} filled weeks and the events"
            f" {list(panel.events)}"
        )
        raise ValueError(msg)
    _whole(periods)
    missing = [c for c in panel.channels if c not in limits]
    extra = [c for c in limits if c not in panel.channels]
    if missing or extra:
        msg = f"every channel needs a range a week: missing {missing}, no such channel {extra}"
        raise ValueError(msg)
    lower = np.array([limits[c].least for c in panel.channels])
    upper = np.array([limits[c].most for c in panel.channels])
    if not np.all(np.isfinite(upper)):
        msg = "every channel's range needs a finite most"
        raise ValueError(msg)
    spend = np.array(panel.spend[0], dtype=float)

    def control(name: str) -> _Array | None:
        if name not in panel.controls:
            return None
        return np.array(panel.control_values[0, :, panel.controls.index(name)], dtype=float)

    return _National(
        sales=np.array(panel.kpi[0], dtype=float),
        spend=spend,
        promotion=control("promotion"),
        price=control("price"),
        status_quo=spend[-STATUS_QUO_WEEKS:].mean(axis=0),
        lower=lower,
        upper=upper,
    )


def _rows(panel: MediaPanel, estimates: Mapping[str, Sequence[LiftEstimate]]) -> _Rows:
    """Each test of `estimates` as the rule reads it, channel by channel in the panel's order
    (the module)."""
    names = [str(c) for c in panel.channels]
    stray = sorted(str(c) for c in estimates if str(c) not in names)
    untested = [c for c in names if not estimates.get(c)]
    if stray or untested:
        msg = (
            "the rule reads a lift test of every channel and of no other: no test of"
            f" {untested}, no such channel {stray}"
        )
        raise ValueError(msg)
    found: list[tuple[str, float, float, float, int, int, int]] = []
    for name in names:
        for estimate in estimates[name]:
            if estimate.geos is not None or estimate.share != 1.0:
                msg = (
                    f"a test of {name} in a share of the market: the rule reads the whole market's"
                )
                raise ValueError(msg)
            weeks = np.flatnonzero(panel.dates == estimate.end) if estimate.end is not None else ()
            if not len(weeks):
                msg = f"a test of {name} ends on {estimate.end}, no date of the panel's"
                raise ValueError(msg)
            readout = estimate.readout
            first = estimate.spend.size - readout
            changed = np.flatnonzero(estimate.spend != estimate.planned)
            dark = int(changed.size)
            opened = int(weeks[0]) - readout + 1
            if (
                not dark
                or int(changed[0]) != first
                or int(changed[-1]) != first + dark - 1
                or not np.all(estimate.spend[changed] == 0.0)
                or opened < 0
            ):
                msg = (
                    f"a test of {name} is a go-dark whose dark weeks open its readout, within the"
                    " panel's weeks"
                )
                raise ValueError(msg)
            x = float(np.mean(estimate.planned[first : first + dark]))
            found.append(
                (name, x, estimate.lift / dark, estimate.se / dark, opened + 1, dark, readout)
            )
    if len({row[5] for row in found}) > 1 or len({row[6] for row in found}) > 1:
        msg = "the rule reads tests that went dark for the same weeks and read for the same weeks"
        raise ValueError(msg)
    return _Rows(
        channel=tuple(row[0] for row in found),
        x=np.array([row[1] for row in found]),
        delta_y=np.array([row[2] for row in found]),
        sigma=np.array([row[3] for row in found]),
        start=np.array([row[4] for row in found]),
        dark=found[0][5],
        readout=found[0][6],
    )


def _floored(
    share: float,
    held: npt.NDArray[np.bool_],
    lower: _Array,
    upper: _Array,
    spend: _Array,
    weekly: float,
) -> _Array:
    """The floors with each free channel's raised to `share` times its least weekly spend over the
    history, at most its ceiling and never lowered; `lower` where the raised floors leave no plan
    of `weekly` a week."""
    least = share * spend.min(axis=0)
    raised = np.where(held, lower, np.maximum(lower, np.minimum(least, upper)))
    fits = bool(np.all(raised <= upper)) and bool(raised.sum() <= weekly <= upper.sum())
    return raised if fits else lower


def _claim(post: Shrunk, move: _Array, periods: int) -> tuple[float, tuple[float, float]]:
    """The gain `move` claims over `periods` weeks at the shrunk margins' means, and its central
    `GAIN_LEVEL` normal interval."""
    gain = periods * float(post.mean @ move)
    spread = periods * math.sqrt(float(move @ post.cov @ move))
    z = NormalDist().inv_cdf((1.0 + GAIN_LEVEL) / 2.0)
    return gain, (gain - z * spread, gain + z * spread)


def _weekly(budget: float | None, periods: int, status_quo: _Array) -> float:
    """The weekly budget: the status quo's weekly total where `budget` is None."""
    return float(status_quo.sum()) if budget is None else _finite(budget) / periods


def _total(budget: float | None, periods: int, weekly: float) -> float:
    """What a plan spends over its `periods` weeks: `budget`, or the weekly budget's `periods`
    times where it is None."""
    return periods * weekly if budget is None else float(budget)


def _finite(budget: float) -> float:
    value = float(budget)
    if not math.isfinite(value):
        msg = f"a budget is finite, not {budget}"
        raise ValueError(msg)
    return value


def _whole(periods: int) -> None:
    if isinstance(periods, bool) or not isinstance(periods, int) or periods < 1:
        msg = f"periods must be a whole number of weeks, at least 1, got {periods!r}"
        raise ValueError(msg)
