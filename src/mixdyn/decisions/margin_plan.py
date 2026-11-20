"""A budget planned on linear margins: each channel's margin, the KPI a unit of weekly spend returns
(`mixdyn.models.margin_mmm`), shrunk towards the channels' common level by empirical Bayes, and the
plan that a linear response makes of the shrunk means.

**The shrinkage** (`shrink_margins`) reads the margins' estimate as Gaussian in their own units,
`b ~ N(beta, V)`, each channel's margin its common level and its own departure,
`beta = mu 1 + delta`, `delta ~ N(0, tau^2 I)`, `mu` flat. Given `tau`, `mu` integrated out, the
margins' posterior has precision `(I - J / C) / tau^2 + V^-1` and mean that precision's inverse
times `V^-1 b`, `J` the matrix of ones over the `C` channels: a noisy channel's estimate is pulled
in further than a precise one's, which takes the winner's curse off the channel the plan ranks
first. `tau` is `tau_rel` times the mean of the positive margins, `tau_rel` on 60 points spaced
evenly in its logarithm over `[0.01, 3]`, each weighed by its restricted likelihood (REML,
`mu` integrated out) times a half-normal prior of scale 0.5 and the log grid's Jacobian, `tau_rel`
itself; the posterior is the mixture over the grid, its mean and covariance. Where no margin is
positive there is nothing to shrink towards: no posterior, and no move.

**The plan** (`plan_corner`). On a response linear in spend, a plan's expected gain is linear in
the margins, so the plan of the most expected gain is the linear program's corner on their
posterior means: every channel at its floor, then the rest of the budget to the channel of the
highest mean up to its ceiling, then to the next. Channels of equal means take it in their order.

**The observed range** (`observed_range`). The margins are read off the weekly spend the history
logged, and inside its range alone the data say anything of them: a plan past a channel's least
or most logged spend extrapolates a straight line no week of the history supports. The observed
range cuts each channel's box to the range its history logged, the floor the larger of the box's
and the history's least, the ceiling the smaller of the box's and the history's most. A status
quo that is a mean of the history's weeks lies inside both, so the floors sum below its budget
and the ceilings above it; an export where they do not raises `ObservedRangeError`.

**The time guard** (`time_held`). Where a channel's weekly spend is a function of the time, a ramp
or a step, a response that curves along the spend and one that drifts along the time give the
same history, and no reading of that history tells them apart: a plan that moves the channel
moves it on a guess. The guard names such a channel by its spend alone: the absolute Pearson
correlation of its weekly spend with the week's number, over every week of the history, above
`TIME_HELD`. A planner holds the channels it names at their status quo. Without noise, a spend
that steps once is named where the step lies between 31 % and 69 % of the history, and one that
rises in a straight line and then stays level where it rises over 36 % of the history or more
(`validation/time_guard.mac`). A spend that never moves has no correlation, and reads 0: it is
one whose range is at most `CONSTANT` of its mean's size, so the last bits of a level spend are
never read as a trend.

The arithmetic follows the research code that found the rule, step for step, so its means and its
plans are reproduced (`tests/fixtures/margin_worlds.json`); the posterior's covariance is this
package's, read as each `tau`'s covariance and the spread of its mean about the mixture's.

**The adstock range** (`adstock_range`). A curve's margins are read off the history's adstock, not
its weekly spend: the observed range is cut further to each channel's range of its adstock at the
retention its curve weighs most, inside which the history read the curve; where that box holds no
plan on the budget, the observed range stands. An adstock is a weighted mean of the channel's
spends, so it lies in their range, and it is held there: its rounding steps a last bit outside,
and on a channel whose spend never moved that bit would empty the cut box and decide the plan.

**Water-filling** (`water_fill`) plans on concave curves: the plan of the most gain
`sum_c m_c H_c(p_c)` in the box at the budget, `m_c` each channel's margin at the status quo and
`H_c` its curve, whose slope `H'_c` is 1 at the status quo and falls with the spend. At a price
`lam` each channel whose margin is positive spends where its margin meets the price,
`m_c H'_c(p_c) = lam`, held to its box, solved by bisection in `p_c`; a channel whose curve is a
line, or whose margin is not positive, is at its ceiling where its margin exceeds the price and at
its floor elsewhere, as the corner plans it. The price is bisected in its logarithm until the
channels spend at most the budget, and what is left goes to a line at the price, then to the
channels in the order of their margins up to their ceilings. Where the positive channels at their
ceilings leave budget over, the rest goes by the corner. Where no curve is curved the plan is the
corner's.

**Water-filling on a table** (`water_fill_tabulated`) plans on falling margin curves given as
functions of the spend, a mixture of curve families among them, without a slope to bisect: each
channel's margin is tabulated at `POINTS` spends across its box and held falling by a running
minimum; at a price each channel spends the most tabulated spend whose margin is at least the
price, its floor where none is; the price is bisected `PRICE_BISECTIONS` times between below the
least margin and above the most, and what the plan at the upper price leaves of the budget is
shared by the channels the lower price moves, in proportion to their moves. Its plan lies within
the table's steps of `water_fill`'s on the same curves; a curve that never falls is planned at the
corner. It is the planner the tested-rung rule was found with, kept so that its plans are that
rule's; `water_fill` is the exact one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

type _Array = npt.NDArray[np.float64]

TAU_GRID = np.exp(np.linspace(math.log(0.01), math.log(3.0), 60))
"""`tau` over the mean of the positive margins, the grid the posterior mixes over."""
HALF_NORMAL_SCALE = 0.5
"""The scale of the half-normal prior on `tau` over the mean of the positive margins."""
BISECTIONS = 56
"""The bisections of a channel's spend at a price; the price's take `BISECTIONS + 12`."""
PRICE_SPAN = 80.0
"""The span in the price's logarithm the bisection searches below twice the highest margin."""
POINTS = 4001
"""The spends across each channel's box at which `water_fill_tabulated` tabulates its margin."""
PRICE_BISECTIONS = 200
"""The bisections of the price in `water_fill_tabulated`."""
TIME_HELD = 0.8
"""The absolute correlation of a channel's weekly spend with the week's number above which
`time_held` names the channel."""
CONSTANT = 1e-9
"""The most a channel's weekly spend may range over, relative to its mean's size, to be a
constant to `time_held`."""


class ObservedRangeError(ValueError):
    """The box cut to the history's range holds no plan that spends the budget; the message says
    how."""


@dataclass(frozen=True, slots=True, eq=False)
class Shrunk:
    """The margins' posterior after the shrinkage: `mean` and `cov`, mixed over the grid, and
    `tau`, the posterior mean of `tau` over the mean of the positive margins."""

    mean: _Array
    cov: _Array
    tau: float


def shrink_margins(mean: npt.ArrayLike, cov: npt.ArrayLike) -> Shrunk | None:
    """The margins `mean`, `(channels,)`, of covariance `cov` shrunk towards their common level;
    None where no margin is positive. See the module."""
    b = np.asarray(mean, dtype=float)
    v = np.asarray(cov, dtype=float)
    positive = b[b > 0]
    if positive.size == 0:
        return None
    scale = float(positive.mean())
    count = b.size
    vi = np.linalg.inv(v)
    one = np.ones(count)
    centre = np.eye(count) - np.ones((count, count)) / count
    logw, means, covs = [], [], []
    for t_rel in TAU_GRID:
        t = t_rel * scale
        sigma = v + t * t * np.eye(count)
        si = np.linalg.inv(sigma)
        oso = one @ si @ one
        quad = b @ si @ b - (one @ si @ b) ** 2 / oso
        _, logdet = np.linalg.slogdet(sigma)
        reml = -0.5 * logdet - 0.5 * math.log(oso) - 0.5 * quad
        prior = -0.5 * (t_rel / HALF_NORMAL_SCALE) ** 2 + math.log(t_rel)
        logw.append(reml + prior)
        precision = centre / (t * t) + vi
        means.append(np.linalg.solve(precision, vi @ b))
        covs.append(np.linalg.inv(precision))
    logs = np.array(logw)
    w = np.exp(logs - logs.max())
    w /= w.sum()
    stacked = np.array(means)
    shrunk = np.einsum("k,kc->c", w, stacked)
    apart = stacked - shrunk
    spread = np.einsum("k,kij->ij", w, np.array(covs) + np.einsum("ki,kj->kij", apart, apart))
    return Shrunk(shrunk, 0.5 * (spread + spread.T), float(w @ TAU_GRID))


def plan_corner(
    margins: npt.ArrayLike, lower: npt.ArrayLike, upper: npt.ArrayLike, budget: float
) -> _Array:
    """The weekly plan of the most gain on linear `margins`, `(channels,)`, in the box
    `[lower, upper]` at `budget` a week: every channel at its floor, the rest of the budget to the
    highest margin up to its ceiling, then the next; equal margins in the channels' order.

    Raises:
        ValueError: on arrays of other shapes, a margin or a bound that is not finite, a floor
            above its ceiling, or floors that sum past the budget or ceilings that sum short of it.
    """
    g, low, high = _box(margins, lower, upper, budget)
    plan = low.copy()
    left = budget - plan.sum()
    for c in np.argsort(-g, kind="stable"):
        add = min(high[c] - plan[c], left)
        plan[c] += add
        left -= add
    return plan


def observed_range(
    spend: npt.ArrayLike, lower: npt.ArrayLike, upper: npt.ArrayLike, budget: float
) -> tuple[_Array, _Array]:
    """The box `[lower, upper]`, `(channels,)`, cut to each channel's range of weekly spend over
    the history `spend`, `(weeks, channels)`: the floors and the ceilings a plan of `budget` a
    week keeps to (the module).

    Raises:
        ValueError: on arrays of other shapes, or values that are not finite.
        ObservedRangeError: where a channel's box misses its range, or the floors sum past the
            budget or the ceilings short of it.
    """
    x = np.asarray(spend, dtype=float)
    low = np.asarray(lower, dtype=float)
    high = np.asarray(upper, dtype=float)
    if x.ndim != 2 or x.shape[0] < 1 or low.shape != (x.shape[1],) or high.shape != low.shape:
        msg = (
            f"the history is (weeks, channels) and the bounds (channels,), not {x.shape},"
            f" {low.shape} and {high.shape}"
        )
        raise ValueError(msg)
    if not all(np.all(np.isfinite(a)) for a in (x, low, high)) or not math.isfinite(budget):
        msg = "the history, the bounds and the budget are finite"
        raise ValueError(msg)
    floor = np.maximum(low, x.min(axis=0))
    ceiling = np.minimum(high, x.max(axis=0))
    if np.any(floor > ceiling) or not floor.sum() <= budget <= ceiling.sum():
        msg = (
            f"the box cut to the history's range, [{floor.tolist()}, {ceiling.tolist()}], holds no"
            f" plan of {budget} a week"
        )
        raise ObservedRangeError(msg)
    return floor, ceiling


def time_held(spend: npt.ArrayLike) -> tuple[_Array, npt.NDArray[np.bool_]]:
    """Each channel's absolute Pearson correlation of its weekly spend with the week's number
    over the history `spend`, `(weeks, channels)`, 0 for a channel whose spend ranges over at most
    `CONSTANT` of its mean's size; and the channels where it lies above `TIME_HELD` (the module).

    Raises:
        ValueError: on a history of another shape or of one week, or a value that is not finite.
    """
    x = np.asarray(spend, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        msg = f"the history is (weeks, channels) over 2 weeks or more, not {x.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(x)):
        msg = "the history is finite"
        raise ValueError(msg)
    week = np.arange(x.shape[0], dtype=float)
    moves = np.ptp(x, axis=0) > CONSTANT * np.abs(x.mean(axis=0))
    correlation = np.zeros(x.shape[1])
    for c in np.flatnonzero(moves):
        correlation[c] = abs(np.corrcoef(x[:, c], week)[0, 1])
    return correlation, correlation > TIME_HELD


def adstock_range(
    spend: npt.ArrayLike,
    adstock: npt.ArrayLike,
    lower: npt.ArrayLike,
    upper: npt.ArrayLike,
    budget: float,
) -> tuple[_Array, _Array]:
    """The observed range of the history `spend`, `(weeks, channels)`, in the box
    `[lower, upper]`, cut further to each channel's range of `adstock`, `(weeks, channels)`; the
    observed range where the cut box holds no plan of `budget` a week (the module).

    Raises:
        ValueError: as `observed_range` raises, and on an adstock of another shape or not finite.
        ObservedRangeError: as `observed_range` raises.
    """
    floor, ceiling = observed_range(spend, lower, upper, budget)
    a = np.asarray(adstock, dtype=float)
    if a.shape != np.shape(spend) or not np.all(np.isfinite(a)):
        msg = f"the adstock is the history's shape, {np.shape(spend)}, and finite, not {a.shape}"
        raise ValueError(msg)
    x = np.asarray(spend, dtype=float)
    a = np.clip(a, x.min(axis=0), x.max(axis=0))
    low = np.maximum(floor, a.min(axis=0))
    high = np.minimum(ceiling, a.max(axis=0))
    if np.any(low > high) or not low.sum() <= budget <= high.sum():
        return floor, ceiling
    return low, high


def water_fill(
    margins: npt.ArrayLike,
    slope: Callable[[_Array], _Array],
    shaped: npt.ArrayLike,
    lower: npt.ArrayLike,
    upper: npt.ArrayLike,
    budget: float,
) -> _Array:
    """The weekly plan of the most gain on concave curves through the `margins` at the status
    quo, `(channels,)`, in the box `[lower, upper]` at `budget` a week: `slope(p)` is each
    channel's curve's slope `H'_c(p_c)`, `(channels,)`, falling in `p_c`, and `shaped` marks the
    channels whose curve is not a line; see the module.

    Raises:
        ValueError: as `plan_corner` raises.
    """
    m, low, high = _box(margins, lower, upper, budget)
    curved = np.asarray(shaped, dtype=bool) & (m > 0)
    if not curved.any():
        return plan_corner(m, low, high, budget)
    at_low, at_high = slope(low), slope(high)

    def demand(price: float) -> _Array:
        """Each channel's spend at `price`."""
        corner = np.where(m > price, high, low).astype(float)
        target = price / np.where(m > 0, m, 1.0)
        a, b = low.copy(), high.copy()
        for _ in range(BISECTIONS):
            mid = 0.5 * (a + b)
            above = slope(mid) > target
            a = np.where(above, mid, a)
            b = np.where(above, b, mid)
        inner = np.where(at_low <= target, low, np.where(at_high >= target, high, 0.5 * (a + b)))
        return np.where(curved, inner, corner)

    top = float(np.max(np.where(m > 0, m * at_low, 0.0))) * 2.0 + 1e-300
    if float(demand(1e-300 * top).sum()) < budget:
        plan = np.where(m > 0, high, low).astype(float)
        rest = m <= 0
        plan[rest] = plan_corner(m[rest], low[rest], high[rest], budget - plan[~rest].sum())
        return plan
    a, b = math.log(top) - PRICE_SPAN, math.log(top)
    for _ in range(BISECTIONS + 12):
        mid = 0.5 * (a + b)
        if demand(math.exp(mid)).sum() > budget:
            a = mid
        else:
            b = mid
    price = math.exp(b)
    plan = demand(price)
    short = budget - plan.sum()
    order = np.argsort(-m, kind="stable")
    for c in order:
        if short <= 0:
            break
        # a line whose margin is the price, to rounding, takes what is left first
        if not curved[c] and m[c] >= price * (1 - 1e-9):
            add = min(high[c] - plan[c], short)
            plan[c] += add
            short -= add
    for c in order:
        if short <= 0:
            break
        add = min(high[c] - plan[c], short)
        plan[c] += add
        short -= add
    return plan


def water_fill_tabulated(
    curves: Sequence[Callable[[_Array], _Array]],
    lower: npt.ArrayLike,
    upper: npt.ArrayLike,
    budget: float,
) -> _Array:
    """The weekly plan of the most gain on falling margin curves in the box `[lower, upper]` at
    `budget` a week, `curves[c](p)` channel `c`'s margin at the weekly spends `p`, `(points,)`;
    see the module.

    Raises:
        ValueError: as `plan_corner` raises on the box and the budget.
    """
    _, low, high = _box(np.zeros(len(curves)), lower, upper, budget)
    grids = [np.linspace(low[c], high[c], POINTS) for c in range(low.size)]
    margins = [
        np.minimum.accumulate(curve(grid)) for curve, grid in zip(curves, grids, strict=True)
    ]

    def spend(price: float) -> _Array:
        """Each channel's most tabulated spend whose margin is at least `price`."""
        out = np.empty(low.size)
        for c in range(low.size):
            k = int(np.searchsorted(-margins[c], -price, side="right"))
            out[c] = grids[c][k - 1] if k > 0 else low[c]
        return out

    a = min(float(m.min()) for m in margins) - 1.0
    b = max(float(m.max()) for m in margins) + 1.0
    for _ in range(PRICE_BISECTIONS):
        mid = 0.5 * (a + b)
        if spend(mid).sum() > budget:
            a = mid
        else:
            b = mid
    above, plan = spend(a), spend(b)
    gap = budget - plan.sum()
    room = above - plan
    if gap > 0 and room.sum() > 0:
        plan = plan + gap * room / room.sum()
    return plan


def _box(
    margins: npt.ArrayLike, lower: npt.ArrayLike, upper: npt.ArrayLike, budget: float
) -> tuple[_Array, _Array, _Array]:
    """The margins and the box as arrays, checked to hold a plan of `budget` a week.

    Raises:
        ValueError: on arrays of other shapes, a margin or a bound that is not finite, a floor
            above its ceiling, or floors that sum past the budget or ceilings that sum short of it.
    """
    g = np.asarray(margins, dtype=float)
    low = np.asarray(lower, dtype=float)
    high = np.asarray(upper, dtype=float)
    if g.ndim != 1 or low.shape != g.shape or high.shape != g.shape:
        msg = f"margins and bounds are (channels,), not {g.shape}, {low.shape} and {high.shape}"
        raise ValueError(msg)
    if not all(np.all(np.isfinite(a)) for a in (g, low, high)) or not math.isfinite(budget):
        msg = "the margins, the bounds and the budget are finite"
        raise ValueError(msg)
    if np.any(low > high) or not low.sum() <= budget <= high.sum():
        msg = f"no plan in the box [{low.tolist()}, {high.tolist()}] spends {budget} a week"
        raise ValueError(msg)
    return g, low, high
