"""A budget planned for a Bayesian posterior's expected return, over its draws.

A posterior reaches the planner as `Readings`: each channel's retention, half-saturation point,
Hill slope and coefficient at each draw, as `BayesFit.draws` holds them, and each draw's weight.
`readings` reads one fit. `stacked_readings` reads several fits of one panel's geo as the mixture
of their predictive distributions that stacking (Yao, Vehtari, Simpson and Gelman 2018) weights by
their leave-one-out log predictive densities: each fit's weight spread evenly over its draws. Or it
reads weights given, as `pick_by_lift_tests` gives them: the whole weight on the fit that predicts
the lift rows best, left out one at a time.

A channel's worth at a draw and a spend a period is what it returns over the plan's periods and the
kernel's length after them, the history's carryover counted in, as `chc.allocation` reads a
channel: the coefficient times the curve of the adstock the history and the plan's spend leave in
each of those periods, summed. `worths` tabulates each draw's worth of each channel on a grid of
spends across its range. The worth is separable in the channels, so a plan's expected worth is the
sum of each channel's mean over the draws: `plan_posterior` reads each channel's mean on a grid
across its range and plans the best split of the means' interpolants. It spends the budget, each
channel from its least, down the upper concave envelopes of the means, the steepest stretch first;
where that stops inside a stretch that passes over points of a grid, as an S-shaped mean's chord
does, it searches boxes of the grids' points by branch and bound (Udell and Boyd 2016), each box
bounded by the same fill down the envelopes of its own points. An interpolant is within
`h^2 max|m''| / 8` of its mean for a grid's step `h`, from below where the mean is concave, so the
plan's expected worth is within the channels' sum of those of the best plan's on concave means,
and twice that on S-shaped ones.

`plan_posterior` takes a goal in the budget's place, as `plan_geos` does, and finds the budget.
At `MarginalReturnTarget(per_unit)` each channel is spent until its mean's slope crosses
`per_unit` a currency unit, or to an end of its range: the plan of the most expected worth less
`per_unit` a unit spent, the budget free, which is separable in the channels, so each channel's
best point of its grid, the least on a tie, exact on the interpolants whatever their shape. At
`ReturnOnSpendTarget(per_unit)` the budget is the most whose plan's expected gain is `per_unit`
a currency unit or more. On concave means the best plan's gain less `per_unit` a unit peaks at the
marginal target's budget and falls past it, so the budget is where it falls through 0; on S-shaped
means it may cross 0 more than once, and the stretch from the peak to the ranges' most is halved,
its upper half searched in boxes for the plan of the most gain less the charge with its total in
the half, as a budget's plan is searched: where that plan meets the target the stretch starts at
its budget, and otherwise it is the lower half. The budget found is the last crossing, to a share
`1e-12` of the most the ranges spend.

`plan_cvar` plans for the draws' worst share instead: the split with the most mean gain over a
reference, the plan in place, in the worst `level` share of the draws, by
`chc.allocation.cvar_allocate` on each draw's channels as `chc.response` reads them. Its draws weigh
alike, so each fit gives its weight's share of them, by largest remainders, evenly spaced over its
own as `BayesFit.channel_draws` spaces a fit's. The reference gains 0 under every draw, so the plan
never does worse in that share than keeping it.

`breakdown` and `plan_robust` ask how wrong the effects may be before a plan stops paying. Each
channel's effect may be off by a factor in `[1/r, r]`, the same `r` for every channel, its
direction each channel's own. A plan's gain over a reference, `sum_c g_c`, `g_c` channel `c`'s
change in worth, is least at `sum g_c^+ / r - r sum g_c^-`, effects overstated where the plan grows
a channel and understated where it cuts one, and that is 0 at `r* = sqrt(sum g_c^+ / sum g_c^-)`,
the plan's breakdown factor: a plan whose net gain is 5 % of what its cuts give up has
`r* = sqrt(1.05) = 1.025`, erased by effects 2.5 % off. `plan_robust` plans for the least gain at a
given `r`: each channel's worth over the reference's, `x`, counted as `x / r` above it and `r x`
below, a concave function of `x`, so a concave mean worth stays concave and the same search plans
it. It keeps the reference until some channel's marginal worth there is more than `r^2` times
another's. Runge and Pauwels (2026) ask the packages for sensitivity analysis of how much
unobserved confounding would change recommendations; this reads it at the recommendation. The
factor is on the effects' size, and is not `chc`'s marginal sensitivity `Gamma`, a bound on the
odds of exposure: `chc.lift.ObservationalCheck`'s factor, what lift tests read over what a channel
predicts, is one reading of the first, and its `least_gamma` the translation to the second.

A plan, `plan_posterior`'s, `plan_robust`'s or `plan_cvar`'s, names in `unidentified` each channel
it moves off its reference that the panel flags `IssueCode.CONSTANT` in the geo: a spend that never
moved over the history, so the posterior's worth there is the prior's and the move rests on nothing
the data read. No number of the plan changes. The plan read without a lift test, `plan_cvar` at
0.1, left such a channel where it is on the flat-spend world (ADR 0044), and names nothing there; at
a level above it the plan can move the channel, and names it. A range that is a point holds it in
any plan, and a lift test of it (`mixdyn.experiments.choose_tests`) reads it.

Two fits' stacking is a one-dimensional concave problem, solved by bisection on its slope; three
or more fits' by Cover's iteration, `mixdyn.inference.stacking`, which can fail to converge.

Each draw's kernel is a running product of its retention, normalised to sum 1, and its curve a
`tanh` of `atanh(1/2)` times the adstock over the half-saturation point, the Bayesian model's, or a
Hill. The arithmetic is NumPy's, in double precision whatever JAX's, but for `plan_cvar`'s, which is
`chc.allocation`'s.
"""

from __future__ import annotations

import heapq
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from chc import response
from chc.allocation import (
    MarginalReturnTarget,
    ReturnOnSpendTarget,
    ReturnTarget,
    cvar_allocate,
)

from mixdyn.data import Channel, IssueCode
from mixdyn.inference.loo import _stacking_of_two, loo, stacking
from mixdyn.models.bayes_mmm import _TANH_HALF, TanhCurve, _thinned

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from mixdyn.data import Geo, MediaPanel
    from mixdyn.decisions.geo_plan import Range
    from mixdyn.models.bayes_mmm import BayesFit

log = logging.getLogger(__name__)

type _Array = npt.NDArray[np.float64]
type _Stretches = tuple[_Array, _Array, npt.NDArray[np.intp]]
"""A box's envelope of a channel's points: each stretch's slope and run, and its ends' indices."""

_BLOCK = 1 << 21
"""The most elements a block of spends' adstocks holds, `(spends, draws, periods)`."""
_TOTAL = 1e-12
"""How far the fits' weights may sum from 1."""
_GAP = 1e-9
"""The share of the first box's bound within which a box's bound is the best plan's worth."""
_BOXES = 500
"""The most boxes a plan's search bounds."""
_KEPT = 1e-12
"""How near a robust plan's spend, over its range's most, is to the reference's that it keeps."""
_SPAN = 1e-12
"""The share of the most the ranges spend to which a return-on-spend target's budget is found."""


@dataclass(frozen=True, slots=True, eq=False)
class Readings:
    """A posterior's draws as a planner reads them, each draw weighted.

    `retention`, `half_saturation`, `slope` and `coefficient` are `(draws, channels)`, the channels
    in `channels`' order, each in the panel's units as `BayesFit.draws` holds it; `slope` is NaN at
    a draw whose curve is a `tanh`, as `tanh` marks it. `fit` is the index of the fit each draw
    comes from and `chain` its chain in that fit. `stacking` is each fit's weight, which its draws
    share evenly, `weights`. `pointwise` holds each fit's leave-one-out log predictive density of
    each observation, `(fits, observations)`, where the weights were solved from them, so that they
    can be solved again, and is None where they were given. `length` is the kernels' length.

    Raises:
        ValueError: on no channels or a channel named twice; parameters of other shapes, a
            retention outside `[0, 1]`, a half-saturation point not positive and finite, a slope
            not positive and finite where the curve is a Hill or not NaN where it is a `tanh`, a
            coefficient not finite; draws' marks of other shapes or kinds, a negative chain, a fit
            with no draw or with draws of both curves; weights not one a fit, at or above 0 and
            summing to 1; leave-one-out densities of another shape or not finite; and a length that
            is not a whole number of periods, at least 1.
    """

    channels: tuple[Channel, ...]
    retention: _Array
    half_saturation: _Array
    slope: _Array
    coefficient: _Array
    tanh: npt.NDArray[np.bool_]
    fit: npt.NDArray[np.intp]
    chain: npt.NDArray[np.intp]
    stacking: _Array
    pointwise: _Array | None
    length: int

    def __post_init__(self) -> None:
        channels = tuple(self.channels)
        if not channels or len(set(channels)) != len(channels):
            msg = f"readings need channels, each named once, got {list(channels)}"
            raise ValueError(msg)
        retention, half, slope, coefficient = (
            _parameter(getattr(self, name), name, len(channels))
            for name in ("retention", "half_saturation", "slope", "coefficient")
        )
        draws = retention.shape[0]
        if not half.shape[0] == slope.shape[0] == coefficient.shape[0] == draws:
            msg = (
                "the parameters need one row a draw: got "
                f"{[a.shape[0] for a in (retention, half, slope, coefficient)]}"
            )
            raise ValueError(msg)
        tanh = _mark(self.tanh, "tanh", draws, "b").astype(np.bool_)
        fit = _mark(self.fit, "fit", draws, "iu").astype(np.intp)
        chain = _mark(self.chain, "chain", draws, "iu").astype(np.intp)
        weights = np.array(self.stacking, dtype=np.float64)
        if not (
            weights.ndim == 1
            and weights.size >= 1
            and np.all(np.isfinite(weights))
            and np.all(weights >= 0.0)
            and abs(float(weights.sum()) - 1.0) <= _TOTAL
        ):
            msg = f"the fits' weights must be at or above 0 and sum to 1, got {weights}"
            raise ValueError(msg)
        if np.any(fit < 0) or np.any(fit >= weights.size) or np.any(chain < 0):
            msg = f"draws name fits outside the {weights.size} weighted or a negative chain"
            raise ValueError(msg)
        counts = np.bincount(fit, minlength=weights.size)
        mixed = [k for k in range(weights.size) if np.unique(tanh[fit == k]).size > 1]
        if not counts.all() or mixed:
            empty = np.flatnonzero(counts == 0).tolist()
            msg = f"every fit needs draws, all of one curve: fits {empty} have none, {mixed} both"
            raise ValueError(msg)
        if not (np.all((retention >= 0.0) & (retention <= 1.0)) and np.all(half > 0.0)):
            msg = "each retention must be in [0, 1] and each half-saturation point above 0"
            raise ValueError(msg)
        if not (np.all(np.isfinite(half)) and np.all(np.isfinite(coefficient))):
            msg = "each half-saturation point and coefficient must be finite"
            raise ValueError(msg)
        hill = slope[~tanh]
        if not (np.all(np.isnan(slope[tanh])) and np.all(np.isfinite(hill) & (hill > 0.0))):
            msg = "a slope must be NaN where the curve is a tanh and positive and finite elsewhere"
            raise ValueError(msg)
        pointwise = None if self.pointwise is None else _densities(self.pointwise, weights.size)
        if isinstance(self.length, bool) or not isinstance(self.length, int) or self.length < 1:
            msg = f"length={self.length!r} is not a whole number of periods, at least 1"
            raise ValueError(msg)
        object.__setattr__(self, "channels", channels)
        for name, value in (
            ("retention", retention),
            ("half_saturation", half),
            ("slope", slope),
            ("coefficient", coefficient),
            ("tanh", tanh),
            ("fit", fit),
            ("chain", chain),
            ("stacking", weights),
            ("pointwise", pointwise),
        ):
            if value is not None:
                value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def weights(self) -> _Array:
        """Each draw's weight, `(draws,)`: its fit's weight over the fit's number of draws."""
        counts = np.bincount(self.fit, minlength=self.stacking.size)
        return self.stacking[self.fit] / counts[self.fit]


@dataclass(frozen=True, slots=True, eq=False)
class Worths:
    """Each draw's worth of each channel at spends a period across its range: `rates`,
    `(channels, grid)`, the spends, and `table`, `(draws, channels, grid)`, the worths, the
    channels in the readings' order."""

    rates: _Array
    table: _Array


@dataclass(frozen=True, slots=True, eq=False)
class PosteriorPlan:
    """A plan of one geo's channels for a posterior's expected return, and the current spend it
    changes.

    `spend` and `current` are `(channels,)` in the readings' order, in spend a period: the plan's,
    and each channel's mean over the panel's last `periods` observed periods. `gain` is what the
    plan's spend adds to the KPI over its periods and the carryover after them, and `current_gain`
    what the current spend adds run the same way, each the posterior's expected value against
    spending nothing in the plan, so the history's carryover counts in neither. `grid` is how many
    spends across each channel's range its mean worth was read at. `unidentified` names each
    channel the plan moves off its current spend whose spend the panel flags as never moving in
    the geo (`IssueCode.CONSTANT`), its worth there read from its prior alone.
    """

    channels: tuple[Channel, ...]
    periods: int
    spend: _Array
    current: _Array
    budget: float
    gain: float
    current_gain: float
    grid: int
    unidentified: tuple[Channel, ...] = ()


@dataclass(frozen=True, slots=True, eq=False)
class CvarPlan:
    """A plan of one geo's channels for the most mean gain over a reference in the worst `level`
    share of a posterior's draws, and that reference.

    `spend` and `against` are `(channels,)` in the readings' order, in spend a period: the plan's
    and the reference's. `gain` is each draw's gain of the plan over the reference, `(draws,)`:
    what the plan's spend adds to the KPI over its periods and the carryover after them less what
    the reference's adds, both on the draw's concave envelopes. `cvar` is the mean of the worst
    `level` share of `gain`, never below 0, the reference's own; no split within the ranges that
    spends the budget has a `cvar` above `bound`. `unidentified` names each channel the plan moves
    off the reference whose spend the panel flags as never moving in the geo
    (`IssueCode.CONSTANT`), its worth there read from its prior alone.
    """

    channels: tuple[Channel, ...]
    periods: int
    spend: _Array
    against: _Array
    budget: float
    level: float
    gain: _Array
    cvar: float
    bound: float
    unidentified: tuple[Channel, ...] = ()


@dataclass(frozen=True, slots=True, eq=False)
class Breakdown:
    """How far a plan's effects may be off before its gain over a reference is gone.

    `spend` and `against` are `(channels,)` in the readings' order, in spend a period: the plan's
    and the reference's. `gains` is each draw's change in each channel's worth from the reference's
    spend to the plan's, `(draws, channels)`, over the plan's periods and the carryover after them,
    and `expected` its mean over the draws by their weights, `(channels,)`. `factor` is `breakdown`
    of `expected`, the plan's breakdown factor for the posterior's expected gain, and `draws`
    `breakdown` of each draw's gains, `(draws,)`.
    """

    channels: tuple[Channel, ...]
    periods: int
    spend: _Array
    against: _Array
    gains: _Array
    expected: _Array
    factor: float
    draws: _Array


@dataclass(frozen=True, slots=True, eq=False)
class RobustPlan:
    """A plan of one geo's channels for the least expected gain over a reference when each
    channel's effect may be off by a factor in `[1/scale, scale]`, and that reference.

    `spend` and `against` are `(channels,)` in the readings' order, in spend a period: the plan's
    and the reference's. `gain` is the plan's expected gain over the reference as the posterior
    reads it, and `worst` its least over the factors, each channel's expected change in worth `x`
    counted as `x / scale` where it is positive and `scale x` where not, both on the means'
    interpolants at `grid` spends across each channel's range, the reference's spend one of them.
    `unidentified` names each channel the plan moves off the reference whose spend the panel flags
    as never moving in the geo (`IssueCode.CONSTANT`), its worth there read from its prior alone.
    """

    channels: tuple[Channel, ...]
    periods: int
    spend: _Array
    against: _Array
    budget: float
    scale: float
    gain: float
    worst: float
    grid: int
    unidentified: tuple[Channel, ...] = ()


def breakdown(gains: npt.ArrayLike) -> _Array:
    """The breakdown factor of each row of channels' `gains` over a reference, `(..., channels)`:
    the common factor `r*` by which effects must be off, overstated where a channel gains and
    understated where it loses, for the gains to sum to 0, `sqrt(sum g^+ / sum g^-)`.

    The least gain at a factor `r`, `sum g^+ / r - r sum g^-`, falls as `r` grows, so at every
    factor below `r*` the gains still sum above 0, and at every one above it below 0. `r*` is below
    1 where the gains sum below 0 as read, `inf` where no channel loses and one gains, `0` where
    none gains and one loses, and 1 where every gain is 0. Returns `(...)`, a 0-d array for one row.

    Raises:
        ValueError: on no channels or a gain that is not finite.
    """
    array = np.asarray(gains, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] < 1 or not np.all(np.isfinite(array)):
        msg = f"gains must be finite, (..., channels >= 1), got shape {array.shape}"
        raise ValueError(msg)
    up = np.maximum(array, 0.0).sum(axis=-1)
    down = np.maximum(-array, 0.0).sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.sqrt(up / down)
    return np.where((up == 0.0) & (down == 0.0), 1.0, factor)


def readings(fit: BayesFit, *, draws: int | None = None) -> Readings:
    """`fit`'s posterior as a planner reads it: `draws` of its draws, evenly spaced over its
    chains' draws in order from the first to the last, or every draw, each weighted alike.

    Raises:
        ValueError: on a `draws` not from 1 to the fit's draws.
    """
    return _read((fit,), np.ones(1), None, draws)


def stacked_readings(
    fits: Sequence[BayesFit],
    *,
    weights: npt.ArrayLike | None = None,
    draws: int | None = None,
) -> Readings:
    """The fits' stacked posterior as a planner reads it: each fit's weight, `weights` or else
    stacking's on the fits' leave-one-out log predictive densities of the observations their
    likelihoods read, spread evenly over `draws` of its draws, evenly spaced over its chains, or
    over every draw.

    The fits are of one geo of one panel: of the same channels, calendar, KPI and observations, and
    kernels of the same length.

    Raises:
        ValueError: on no fits; fits of different channels, panels, observations or kernel
            lengths; weights not one a fit, at or above 0 and summing to 1; and a `draws` not from
            1 to a fit's draws.
        RuntimeError: three fits or more whose stacking does not converge.
    """
    fits = tuple(fits)
    if not fits:
        msg = "stacking needs at least one fit"
        raise ValueError(msg)
    first = fits[0]
    apart = [
        i
        for i, fit in enumerate(fits)
        if _channels(fit) != _channels(first)
        or fit.length != first.length
        or not np.array_equal(fit.dates, first.dates)
        or not np.array_equal(fit.y, first.y, equal_nan=True)
        or not np.array_equal(fit.fitted, first.fitted)
        or fit.log_likelihood.shape[2] != first.log_likelihood.shape[2]
    ]
    if apart:
        msg = (
            f"fits {apart} differ from the first in their channels, panel, observations or "
            "kernels' length; stacking reads fits of one geo of one panel"
        )
        raise ValueError(msg)
    if weights is not None:
        given = np.asarray(weights, dtype=np.float64)
        if given.shape != (len(fits),):
            msg = f"weights must be one a fit, ({len(fits)},), got shape {given.shape}"
            raise ValueError(msg)
        return _read(fits, given, None, draws)
    pointwise = np.stack([loo(fit.log_likelihood).pointwise for fit in fits])
    match len(fits):
        case 1:
            solved = np.ones(1)
        case 2:
            solved = _stacking_of_two(pointwise)
        case _:
            solved = stacking(pointwise)
    return _read(fits, solved, pointwise, draws)


def pick_by_lift_tests(fits: Sequence[BayesFit]) -> _Array | None:
    """Each fit's weight when the lift rows alone choose among `fits`: 1 on the fit whose
    leave-one-out log predictive density, by PSIS-LOO, summed over the observations past the
    KPI's, the geo tests' gaps, the lift estimates and the lift series, is the largest, the first
    on a tie, and 0 on the others; None where the fits read no such observation.
    `stacked_readings` reads the weights as given.

    Stacking reads every observation, and over a long panel the KPI's periods outnumber the lift
    rows tenfold. The tests pin each channel's chord from no spend to the spend tested, and a
    family that reads the KPI's periods a little better can turn that chord into a margin far
    from the truth's; the lift rows read the curve where the tests ran it (ADR 0082).

    Raises:
        ValueError: on no fits, or fits that read different observations.
    """
    fits = tuple(fits)
    if not fits:
        msg = "a pick needs at least one fit"
        raise ValueError(msg)
    shapes = {(int(f.fitted.sum()), f.log_likelihood.shape[2]) for f in fits}
    if len(shapes) > 1:
        msg = f"the fits read different observations, (KPI's, all) {sorted(shapes)}"
        raise ValueError(msg)
    ((kpi, every),) = shapes
    if every == kpi:
        return None
    elpd = [loo(f.log_likelihood[:, :, kpi:]).elpd for f in fits]
    weights = np.zeros(len(fits))
    weights[int(np.argmax(elpd))] = 1.0
    return weights


def worths(
    panel: MediaPanel,
    geo: Geo,
    readings: Readings,
    periods: int,
    limits: Mapping[Channel, Range],
    *,
    grid: int = 512,
) -> Worths:
    """Each draw's worth of each channel of `readings` over `periods` periods of the panel's
    cadence and its kernel's length after them, at `grid` spends a period evenly across the
    channel's range in `limits`, the geo's spend in the panel carried in.

    Raises:
        ValueError: on a geo or a channel the panel does not have; a range missing for a channel,
            named for one the readings do not have or with no finite most; a `periods` or `grid`
            not a whole number, at least 1 and 2.
    """
    history, lower, upper = _inputs(panel, geo, readings, periods, limits)
    _whole(grid, "grid", 2)
    rates = _rates(lower, upper, grid)
    table = np.empty((readings.fit.size, len(readings.channels), grid))
    for draws in _fits(readings):
        for c in range(len(readings.channels)):
            for block, worth in _worth(history[:, c], readings, draws, c, periods, rates[c]):
                table[draws, c, block] = worth.T
    return Worths(rates=rates, table=table)


def plan_posterior(
    panel: MediaPanel,
    geo: Geo,
    readings: Readings,
    budget: float | MarginalReturnTarget | ReturnOnSpendTarget,
    periods: int,
    limits: Mapping[Channel, Range],
    *,
    grid: int = 2048,
) -> PosteriorPlan:
    """The plan of `geo`'s channels over `periods` periods of the panel's cadence that spends
    `budget` over them within `limits`, each channel's range a period, for the most expected worth
    over the readings' draws, each channel's mean worth read at `grid` spends evenly across its
    range.

    `budget` may be a goal in its place, and the plan's budget is then the one that meets it:
    `MarginalReturnTarget(per_unit)`, the plan at which one more currency unit returns `per_unit`
    in expectation, each channel spent until its mean's slope crosses it or to an end of its
    range; `ReturnOnSpendTarget(per_unit)`, the most budget whose plan's expected gain is
    `per_unit` a currency unit or more. A channel whose range is a point is held there.

    Raises:
        TypeError: on a `ReturnTarget`, which `plan_geos` meets.
        ValueError: on a geo or a channel the panel does not have; a range missing for a channel,
            named for one the readings do not have or with no finite most; a `periods` or `grid`
            not a whole number, at least 1 and 2; a budget the ranges cannot spend over the
            periods; a goal whose value is not finite; a return on spend no budget within the
            ranges reaches; and a geo with no observed period among the panel's last `periods`.
    """
    history, lower, upper = _inputs(panel, geo, readings, periods, limits)
    _whole(grid, "grid", 2)
    match budget:
        case MarginalReturnTarget(per_unit=per_unit) | ReturnOnSpendTarget(per_unit=per_unit):
            per_unit = float(per_unit)
            if not math.isfinite(per_unit):
                msg = f"the goal's value is {per_unit}; it needs a finite one"
                raise ValueError(msg)
        case ReturnTarget():
            msg = "plan_posterior plans a budget, a MarginalReturnTarget or a ReturnOnSpendTarget"
            raise TypeError(msg)
        case _:
            budget = _spendable(budget, periods, lower, upper)
    current = _current(panel, geo, readings.channels, periods)
    rates = _rates(lower, upper, grid)
    means = _mean_worths(history, readings, periods, rates)
    match budget:
        case MarginalReturnTarget():
            spend = _marginal(rates, means, periods * per_unit)
            budget = periods * float(spend.sum())
        case ReturnOnSpendTarget():
            idle = float(
                _mean_worths(history, readings, periods, np.zeros((rates.shape[0], 1))).sum()
            )
            spend = _on_spend(rates, means, idle, periods * per_unit)
            if spend is None:
                peak = _searched(
                    rates, means, rates[:, 0].sum(), rates[:, -1].sum(), periods * per_unit, {}
                )
                msg = (
                    f"no budget within the ranges returns {per_unit} a currency unit it "
                    f"spends: the expected gain less {per_unit} a unit peaks at "
                    f"{peak[1] - idle:.6g}, at a budget of {periods * float(peak[0].sum()):.6g}"
                )
                raise ValueError(msg)
            budget = periods * float(spend.sum())
        case _:
            spend = _allocated(rates, means, budget / periods)
    at = np.column_stack([spend, current, np.zeros(spend.size)])
    worth = _mean_worths(history, readings, periods, at)
    return PosteriorPlan(
        channels=readings.channels,
        periods=periods,
        spend=spend,
        current=current,
        budget=budget,
        gain=float(np.sum(worth[:, 0] - worth[:, 2])),
        current_gain=float(np.sum(worth[:, 1] - worth[:, 2])),
        grid=grid,
        unidentified=_unidentified(panel, geo, readings.channels, spend, current, upper),
    )


def plan_cvar(
    panel: MediaPanel,
    geo: Geo,
    readings: Readings,
    budget: float,
    periods: int,
    limits: Mapping[Channel, Range],
    *,
    against: npt.ArrayLike,
    level: float = 0.1,
    draws: int = 400,
) -> CvarPlan:
    """The plan of `geo`'s channels over `periods` periods of the panel's cadence that spends
    `budget` over them within `limits`, each channel's range a period, for the most mean gain over
    `against`, the plan in place at this budget, in the worst `level` share of `draws` of the
    readings' draws: the gain's conditional value at risk, the history's carryover counted in. The
    draws weigh alike, each fit giving its weight's share of them. The default level, 0.1, is the
    one mixdyn plans at where no lift test has been run. `unidentified` names each channel the plan
    moves off `against` that the panel flags as never moving in the geo.

    Raises:
        ValueError: on a geo or a channel the panel does not have; a range missing for a channel,
            named for one the readings do not have or with no finite most; a `periods` or `draws`
            not a whole number, at least 1; a fit holding fewer draws than its share; and what
            `chc.allocation.cvar_allocate` refuses: a level outside `(0, 1]`, a budget the ranges
            cannot spend over the periods, and a reference of the wrong shape, outside the ranges
            or not spending the budget.
    """
    history, lower, upper = _inputs(panel, geo, readings, periods, limits)
    channels = range(len(readings.channels))
    reference = np.array(against, dtype=np.float64)
    allocation = cvar_allocate(
        [tuple(_response(readings, s, c) for c in channels) for s in _alike(readings, draws)],
        budget,
        periods,
        level=level,
        against=reference,
        lower=lower,
        upper=upper,
        history=history,
    )
    spend = np.asarray(allocation.spend, dtype=np.float64)
    return CvarPlan(
        channels=readings.channels,
        periods=periods,
        spend=spend,
        against=reference,
        budget=float(budget),
        level=float(level),
        gain=np.asarray(allocation.gain, dtype=np.float64),
        cvar=float(allocation.cvar),
        bound=float(allocation.bound),
        unidentified=_unidentified(panel, geo, readings.channels, spend, reference, upper),
    )


def plan_breakdown(
    panel: MediaPanel,
    geo: Geo,
    readings: Readings,
    spend: npt.ArrayLike,
    periods: int,
    *,
    against: npt.ArrayLike | None = None,
) -> Breakdown:
    """The breakdown factor of `spend`, a plan of `geo`'s channels a period over `periods` periods
    of the panel's cadence, over `against`, or else each channel's mean spend over the geo's last
    `periods` observed periods as `plan_posterior` reads the current spend: for the posterior's
    expected gain and for each draw's, the history's carryover counted in.

    Raises:
        ValueError: on a geo or a channel the panel does not have; a `periods` not a whole number,
            at least 1; spends not one a channel, finite and at or above 0; and a geo with no
            observed period among the panel's last `periods` where `against` is not given.
    """
    history = _history(panel, geo, readings)
    _whole(periods, "periods", 1)
    plan = _spends(spend, "spend", len(readings.channels))
    reference = (
        _current(panel, geo, readings.channels, periods)
        if against is None
        else _spends(against, "against", len(readings.channels))
    )
    gains = np.empty((readings.fit.size, len(readings.channels)))
    at = np.column_stack([plan, reference])
    for draws in _fits(readings):
        for c in range(len(readings.channels)):
            for _, worth in _worth(history[:, c], readings, draws, c, periods, at[c]):
                gains[draws, c] = worth[0] - worth[1]
    expected = readings.weights @ gains
    for value in (plan, reference, gains, expected):
        value.setflags(write=False)
    return Breakdown(
        channels=readings.channels,
        periods=periods,
        spend=plan,
        against=reference,
        gains=gains,
        expected=expected,
        factor=float(breakdown(expected)),
        draws=breakdown(gains),
    )


def plan_robust(
    panel: MediaPanel,
    geo: Geo,
    readings: Readings,
    budget: float,
    periods: int,
    limits: Mapping[Channel, Range],
    *,
    scale: float,
    against: npt.ArrayLike | None = None,
    grid: int = 2048,
) -> RobustPlan:
    """The plan of `geo`'s channels over `periods` periods of the panel's cadence that spends
    `budget` over them within `limits` for the most expected gain over `against` when each
    channel's effect may be off by a factor in `[1/scale, scale]`, its least over the factors:
    `plan_posterior`'s search on each channel's mean change in worth over the reference's, `x`,
    counted as `x / scale` above 0 and `scale x` below. `against` is the plan in place, or else
    each channel's mean spend over the geo's last `periods` observed periods.

    The count is concave in `x` and kinks at the reference's spend, which joins each channel's
    grid of `grid` spends where it is inside the range, in place of the grid's nearest point
    inside it. The plan keeps the reference's spend, where that spends the budget, while no
    channel's marginal expected worth there is above `scale^2` times another's, and returns it
    as given. At a `scale` of 1
    it is `plan_posterior`'s plan, on `plan_posterior`'s grid.

    Raises:
        ValueError: on what `plan_posterior` refuses; a `scale` not finite and at least 1; and
            reference spends not one a channel, finite and at or above 0.
    """
    history, lower, upper = _inputs(panel, geo, readings, periods, limits)
    _whole(grid, "grid", 2)
    budget = _spendable(budget, periods, lower, upper)
    scale = float(scale)
    if not (math.isfinite(scale) and scale >= 1.0):
        msg = f"scale={scale!r} is not a finite factor of at least 1"
        raise ValueError(msg)
    reference = (
        _current(panel, geo, readings.channels, periods)
        if against is None
        else _spends(against, "against", len(readings.channels))
    )
    rates = _rates(lower, upper, grid)
    if scale > 1.0:
        _place(rates, reference)
    means = _mean_worths(history, readings, periods, rates)
    base = _mean_worths(history, readings, periods, reference[:, None])
    change = means - base
    counted = means if scale == 1.0 else np.where(change >= 0.0, change / scale, change * scale)
    spend = _allocated(rates, counted, budget / periods)
    if np.all(np.abs(spend - reference) <= _KEPT * np.maximum(upper, 1.0)):
        # the fill adds its runs up from each channel's least, off the reference's spend by a
        # rounding that would read as a gain no channel's loss offsets
        spend = reference.copy()
    gained = _mean_worths(history, readings, periods, spend[:, None]) - base
    for value in (spend, reference):
        value.setflags(write=False)
    return RobustPlan(
        channels=readings.channels,
        periods=periods,
        spend=spend,
        against=reference,
        budget=budget,
        scale=scale,
        gain=float(gained.sum()),
        worst=float(np.where(gained >= 0.0, gained / scale, gained * scale).sum()),
        grid=grid,
        unidentified=_unidentified(panel, geo, readings.channels, spend, reference, upper),
    )


def _parameter(value: npt.ArrayLike, name: str, channels: int) -> _Array:
    array = np.array(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != channels:
        msg = f"{name} must be (draws >= 1, {channels} channels), got shape {array.shape}"
        raise ValueError(msg)
    return array


def _mark(value: npt.ArrayLike, name: str, draws: int, kinds: str) -> npt.NDArray[np.generic]:
    array = np.array(value)
    if array.shape != (draws,) or array.dtype.kind not in kinds:
        kind = "boolean" if kinds == "b" else "integer"
        msg = f"{name} must hold one {kind} a draw, ({draws},), got {array.dtype} {array.shape}"
        raise ValueError(msg)
    return array


def _densities(value: npt.ArrayLike, fits: int) -> _Array:
    array = np.array(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != fits or array.shape[1] < 1:
        msg = f"pointwise must be ({fits} fits, observations >= 1), got shape {array.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = "the leave-one-out log densities must be finite"
        raise ValueError(msg)
    return array


def _channels(fit: BayesFit) -> tuple[Channel, ...]:
    return tuple(Channel(m.channel) for m in fit.spec.media)


def _read(
    fits: Sequence[BayesFit], weights: _Array, pointwise: _Array | None, draws: int | None
) -> Readings:
    """Each fit's `draws` draws, evenly spaced over its chains, or every draw, weighted by
    `weights` over the fits."""
    parameters: dict[str, list[_Array]] = {
        name: [] for name in ("retention", "half_saturation", "slope", "coefficient")
    }
    tanh: list[npt.NDArray[np.bool_]] = []
    fit_of: list[npt.NDArray[np.intp]] = []
    chain: list[npt.NDArray[np.intp]] = []
    for k, fit in enumerate(fits):
        chains, per_chain = fit.draws["retention"].shape[:2]
        total = chains * per_chain
        keep = np.arange(total) if draws is None else _thinned((chains, per_chain), draws)
        for name in ("retention", "half_saturation", "coefficient"):
            parameters[name].append(fit.draws[name].reshape(total, -1)[keep])
        curve = isinstance(fit.spec.saturation, TanhCurve)
        parameters["slope"].append(
            np.full_like(parameters["retention"][-1], np.nan)
            if curve
            else fit.draws["slope"].reshape(total, -1)[keep]
        )
        tanh.append(np.full(keep.size, curve))
        fit_of.append(np.full(keep.size, k, dtype=np.intp))
        chain.append(keep // per_chain)
    joined = {name: np.concatenate(parts) for name, parts in parameters.items()}
    return Readings(
        channels=_channels(fits[0]),
        retention=joined["retention"],
        half_saturation=joined["half_saturation"],
        slope=joined["slope"],
        coefficient=joined["coefficient"],
        tanh=np.concatenate(tanh),
        fit=np.concatenate(fit_of),
        chain=np.concatenate(chain),
        stacking=weights,
        pointwise=pointwise,
        length=fits[0].length,
    )


def _inputs(
    panel: MediaPanel,
    geo: Geo,
    readings: Readings,
    periods: int,
    limits: Mapping[Channel, Range],
) -> tuple[_Array, _Array, _Array]:
    """The geo's spend on the readings' channels, `(periods, channels)`, and their ranges' ends."""
    history = _history(panel, geo, readings)
    missing = [c for c in readings.channels if c not in limits]
    extra = [c for c in limits if c not in readings.channels]
    open_ended = [c for c in readings.channels if c in limits and not math.isfinite(limits[c].most)]
    if missing or extra or open_ended:
        msg = (
            f"every channel needs a range a period with a finite most: missing {missing}, no such "
            f"channel {extra}, no finite most {open_ended}"
        )
        raise ValueError(msg)
    _whole(periods, "periods", 1)
    lower = np.array([limits[c].least for c in readings.channels])
    upper = np.array([limits[c].most for c in readings.channels])
    return history, lower, upper


def _history(panel: MediaPanel, geo: Geo, readings: Readings) -> _Array:
    """The geo's spend on the readings' channels, `(periods, channels)`."""
    if geo not in panel.geos:
        msg = f"the panel has no geo {geo!r}: it has {list(panel.geos)}"
        raise ValueError(msg)
    unknown = [c for c in readings.channels if c not in panel.channels]
    if unknown:
        msg = f"the panel has no channels {unknown}: it has {list(panel.channels)}"
        raise ValueError(msg)
    columns = [panel.channels.index(c) for c in readings.channels]
    return panel.spend[panel.geos.index(geo)][:, columns]


def _spendable(budget: float, periods: int, lower: _Array, upper: _Array) -> float:
    """`budget` as a float, where ranges `lower` to `upper` a period spend it over `periods`."""
    budget = float(budget)
    least, most = periods * float(lower.sum()), periods * float(upper.sum())
    if not (math.isfinite(budget) and least <= budget <= most):
        msg = (
            f"a budget of {budget} is outside what the ranges spend over {periods} periods, "
            f"[{least}, {most}]"
        )
        raise ValueError(msg)
    return budget


def _whole(value: int, name: str, least: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < least:
        msg = f"{name} must be a whole number, at least {least}, got {value!r}"
        raise ValueError(msg)


def _alike(readings: Readings, draws: int) -> npt.NDArray[np.intp]:
    """`draws` of the readings' draws that weigh alike: each fit's share of them its weight's, by
    largest remainders, an earlier fit first on a tie, evenly spaced over its own draws in order."""
    _whole(draws, "draws", 1)
    quota = readings.stacking * draws
    counts = np.floor(quota).astype(np.intp)
    counts[np.argsort(counts - quota, kind="stable")[: draws - int(counts.sum())]] += 1
    own = list(_fits(readings))
    short = [k for k, held in enumerate(own) if counts[k] > held.size]
    if short:
        msg = (
            f"fits {short} hold {[own[k].size for k in short]} draws, fewer than their shares of "
            f"{draws}, {counts[short].tolist()}"
        )
        raise ValueError(msg)
    return np.concatenate(
        [held[_thinned((1, held.size), int(n))] for held, n in zip(own, counts, strict=True) if n]
    )


def _response(readings: Readings, s: int, c: int) -> response.Channel:
    """Draw `s`'s channel `c` as `chc.response` reads it, as `BayesFit.channel_draws` builds it."""
    kernel = response.GeometricAdstock(
        float(readings.retention[s, c]), length=readings.length, normalized=True
    )
    half = float(readings.half_saturation[s, c])
    curve = (
        response.Tanh(half / _TANH_HALF)
        if readings.tanh[s]
        else response.Hill(half, float(readings.slope[s, c]))
    )
    return response.Channel(kernel, curve, float(readings.coefficient[s, c]))


def _current(panel: MediaPanel, geo: Geo, channels: Sequence[Channel], periods: int) -> _Array:
    """Each channel's mean spend a period over the geo's last `periods` observed periods, or over
    all of them where it has fewer."""
    g = panel.geos.index(geo)
    observed = int(np.sum(~panel.filled[g, -periods:]))
    if not observed:
        msg = f"geo {geo!r} has no observed period among the panel's last {periods}"
        raise ValueError(msg)
    columns = [panel.channels.index(c) for c in channels]
    # a filled step's spend is 0, so the sum over the window is the observed periods' sum
    return panel.spend[g, -periods:][:, columns].sum(axis=0) / observed


def _unidentified(
    panel: MediaPanel,
    geo: Geo,
    channels: Sequence[Channel],
    spend: _Array,
    reference: _Array,
    upper: _Array,
) -> tuple[Channel, ...]:
    """The channels `spend`, `(..., channels)`, moves off `reference` in any of its rows by more
    than `_KEPT` of their range's most whose spend the panel flags as constant, over every geo or
    in `geo`."""
    flat = {
        issue.channel
        for issue in panel.issues
        if issue.code is IssueCode.CONSTANT and issue.geo in (None, str(geo))
    }
    moved = np.abs(spend - reference) > _KEPT * np.maximum(upper, 1.0)
    moved = moved.reshape(-1, len(channels)).any(axis=0)
    return tuple(c for j, c in enumerate(channels) if str(c) in flat and moved[j])


def _rates(lower: _Array, upper: _Array, grid: int) -> _Array:
    # each channel's own linspace: one with array ends steps every channel alike where one's
    # range is a point, and rounds the others' spends otherwise than the bench
    return np.array([np.linspace(low, high, grid) for low, high in zip(lower, upper, strict=True)])


def _spends(value: npt.ArrayLike, name: str, channels: int) -> _Array:
    array = np.array(value, dtype=np.float64)
    if array.shape != (channels,) or not np.all(np.isfinite(array) & (array >= 0.0)):
        msg = f"{name} must be one finite spend at or above 0 a channel, ({channels},), got {array}"
        raise ValueError(msg)
    return array


def _place(rates: _Array, points: _Array) -> None:
    """Put each channel's point among its spends `rates`, in place of the nearest spend strictly
    inside the range, where the point is strictly inside it: within half a step of the spend it
    replaces, so the spends stay increasing."""
    grid = rates.shape[1]
    for c, point in enumerate(points):
        if grid >= 3 and rates[c, 0] < point < rates[c, -1]:
            nearest = int(np.argmin(np.abs(rates[c] - point)))
            rates[c, min(max(nearest, 1), grid - 2)] = point


def _fits(readings: Readings) -> Iterator[npt.NDArray[np.intp]]:
    """Each fit's draws, in order."""
    for k in range(readings.stacking.size):
        yield np.flatnonzero(readings.fit == k)


def _carried(series: _Array, weights: _Array, start: int) -> _Array:
    """`series` carried over by each row of `weights`, `(draws, L)`, at its periods from `start`
    on: `(draws, periods)`, nothing spent before the series."""
    length = weights.shape[1]
    lag = np.arange(start, series.size)[:, None] - np.arange(length)
    lagged = np.where(lag >= 0, series[np.maximum(lag, 0)], 0.0)
    return weights @ lagged.T


def _worth(
    history: _Array,
    readings: Readings,
    draws: npt.NDArray[np.intp],
    c: int,
    periods: int,
    rates: _Array,
) -> Iterator[tuple[slice, _Array]]:
    """Channel `c`'s worth at each of a fit's `draws` and each spend a period of `rates`, the
    history's carryover in: each block of the rates, and the worths at it, `(rates, draws)`."""
    retention = readings.retention[draws, c]
    half = readings.half_saturation[draws, c]
    coefficient = readings.coefficient[draws, c]
    slope = None if readings.tanh[draws[0]] else readings.slope[draws, c]
    length = readings.length
    steps = np.concatenate(
        [np.ones((draws.size, 1)), np.repeat(retention[:, None], length - 1, 1)], 1
    )
    weights = np.cumprod(steps, axis=1)
    weights /= weights.sum(axis=1, keepdims=True)
    before = history.size
    idle = np.concatenate([history, np.zeros(periods + length - 1)])
    unit = np.concatenate([np.zeros(before), np.ones(periods), np.zeros(length - 1)])
    carry, reach = _carried(idle, weights, before), _carried(unit, weights, before)
    rows = max(1, _BLOCK // carry.size)
    for start in range(0, rates.size, rows):
        block = slice(start, start + rows)
        adstock = carry[None] + rates[block, None, None] * reach[None]
        if slope is None:
            curve = np.tanh(_TANH_HALF * adstock / half[None, :, None])
        else:
            with np.errstate(divide="ignore", over="ignore"):
                curve = np.where(
                    adstock > 0.0,
                    1.0 / (1.0 + (half[None, :, None] / adstock) ** slope[None, :, None]),
                    0.0,
                )
        yield block, coefficient * curve.sum(axis=2)


def _mean_worths(history: _Array, readings: Readings, periods: int, rates: _Array) -> _Array:
    """Each channel's expected worth over the draws at each of its spends of `rates`,
    `(channels, spends)`: each fit's mean over its draws, weighted by the fits' weights."""
    out = np.empty(rates.shape)
    for c in range(rates.shape[0]):
        means = []
        for draws in _fits(readings):
            mean = np.empty(rates.shape[1])
            for block, worth in _worth(history[:, c], readings, draws, c, periods, rates[c]):
                mean[block] = worth.mean(axis=1)
            means.append(mean)
        out[c] = sum(w * mean for w, mean in zip(readings.stacking, means, strict=True))
    return out


def _envelope(x: _Array, y: _Array) -> npt.NDArray[np.intp]:
    """The indices of the upper concave hull of the points `(x, y)`, `x` increasing.

    Where no point lies on or above the chord of its two neighbours, as on a strictly concave
    curve, the scan pops none and the hull is every point, read in one pass; else the scan runs."""
    first, middle, last = slice(None, -2), slice(1, -1), slice(2, None)
    rise, span = y[middle] - y[first], x[last] - x[first]
    if not np.any(rise * span <= (y[last] - y[first]) * (x[middle] - x[first])):
        return np.arange(x.size)
    hull: list[int] = []
    for i in range(x.size):
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            if (y[b] - y[a]) * (x[i] - x[a]) <= (y[i] - y[a]) * (x[b] - x[a]):
                hull.pop()
            else:
                break
        hull.append(i)
    return np.array(hull)


def _allocated(rates: _Array, means: _Array, target: float) -> _Array:
    """The spend a period, each channel's on its grid `rates`, that spends `target` for the most
    worth on the interpolants of the channels' `means` at `rates`, to a share `_GAP` of the bound:
    `_searched` with the total held at `target` and nothing charged."""
    return _searched(rates, means, target, target, 0.0, {})[0]


def _marginal(rates: _Array, means: _Array, charge: float) -> _Array:
    """Each channel's spend a period on its grid `rates` of the most worth on its `means` less
    `charge` a unit, the least on a tie: where the interpolant's slope crosses `charge`, or an end
    of the grid. The worth less the charge is separable in the channels and linear between points,
    so this is the best plan with the total free."""
    best = np.argmax(means - charge * rates, axis=1)
    return rates[np.arange(rates.shape[0]), best]


def _on_spend(rates: _Array, means: _Array, idle: float, charge: float) -> _Array | None:
    """The spend a period of the most total whose best plan's worth on the interpolants less
    `charge` a unit of the total is at least `idle`, the worth of spending nothing, to a share
    `_SPAN` of the grids' most total; None where no total within the grids reaches it.

    Past the plan of the most worth less the charge, the stretch to the grids' most total is
    halved: its upper half is searched for the plan of the most worth less the charge with its
    total in the half; where that meets `idle` the stretch starts at its total, else it is the
    lower half. Every total past the stretch was shown short, so its start is the last crossing."""
    hulls: dict[tuple[int, int, int], _Stretches] = {}
    most = float(rates[:, -1].sum())
    spend, worth = _searched(rates, means, most, most, charge, hulls)
    if worth >= idle:
        return spend
    spend, worth = _searched(rates, means, float(rates[:, 0].sum()), most, charge, hulls)
    if worth < idle:
        return None
    low, high = float(spend.sum()), most
    while high - low > _SPAN * most:
        middle = (low + high) / 2.0
        found, worth = _searched(rates, means, middle, high, charge, hulls)
        if worth >= idle:
            spend, low = found, float(found.sum())
        else:
            high = middle
    return spend


def _searched(
    rates: _Array,
    means: _Array,
    least: float,
    most: float,
    charge: float,
    hulls: dict[tuple[int, int, int], _Stretches],
) -> tuple[_Array, float]:
    """The spend a period, each channel's on its grid `rates`, whose total is within `[least,
    most]`, for the most worth on the interpolants of the channels' `means` at `rates` less
    `charge` a unit of the total, to a share `_GAP` of the bound; and that worth less the charge.

    A box of each channel's grid points is bounded by `_charged`'s worth down the envelopes of its
    points, and the fill's spend, read on the interpolants, is a plan. The box of the largest bound
    is cut first, in the channel whose stretch the fill stops inside, at the first point at or past
    the fill's spend; the search ends when no box's bound is above the best plan's worth by more
    than `_GAP` of the first box's, or at `_BOXES` boxes. Where the first box's fill stops on its
    envelopes' points, as on concave means, its plan is returned and nothing is cut."""
    channels, grid = rates.shape
    first = ((0, grid - 1),) * channels
    spend, top, stop = _charged(rates, means, first, least, most, charge, hulls)
    best, worth = spend, _interpolated(rates, means, spend) - charge * spend.sum()
    tolerance = _GAP * abs(top)
    heap = [(-top, 0, first, spend, stop)]
    boxes = 1
    while heap and -heap[0][0] - worth > tolerance:
        if boxes >= _BOXES:
            log.warning(
                "the plan stopped at %d boxes, its worth %.17g on the grid's interpolants below "
                "their bound %.17g",
                boxes,
                worth,
                -heap[0][0],
                extra={"boxes": boxes, "worth": worth, "bound": -heap[0][0]},
            )
            break
        _, _, box, spend, stop = heapq.heappop(heap)
        if stop is None:  # its bound is its plan's worth but for rounding
            continue
        c, a, b = stop
        cut = int(np.clip(np.searchsorted(rates[c], spend[c]), a + 1, b - 1))
        for part in ((box[c][0], cut), (cut, box[c][1])):
            child = (*box[:c], part, *box[c + 1 :])
            low = sum(rates[j, lo] for j, (lo, _) in enumerate(child))
            high = sum(rates[j, hi] for j, (_, hi) in enumerate(child))
            if not (low <= most and least <= high):
                continue
            spend, top, stop = _charged(rates, means, child, least, most, charge, hulls)
            boxes += 1
            value = _interpolated(rates, means, spend) - charge * spend.sum()
            if value > worth:
                best, worth = spend, value
            if top - worth > tolerance:
                heapq.heappush(heap, (-top, boxes, child, spend, stop))
    return best, worth


def _charged(
    rates: _Array,
    means: _Array,
    boxes: tuple[tuple[int, int], ...],
    least: float,
    most: float,
    charge: float,
    hulls: dict[tuple[int, int, int], _Stretches],
) -> tuple[_Array, float, tuple[int, int, int] | None]:
    """`_fill` of the boxes at the total within `[least, most]` and the boxes' own where their
    envelopes' slopes fall below `charge`, its worth less the charge on that total: the most any
    split in the boxes with its total within `[least, most]` gains less the charge, since the
    envelopes' worth less the charge is concave in the total."""
    if least == most:
        target = least
    else:
        low = sum(rates[c, lo] for c, (lo, _) in enumerate(boxes))
        high = sum(rates[c, hi] for c, (_, hi) in enumerate(boxes))
        steep = 0.0
        for c, (lo, hi) in enumerate(boxes):
            slope, run, _ = _stretches(rates, means, c, lo, hi, hulls)
            steep += float(run[slope >= charge].sum())
        target = min(max(low + steep, least, low), most, high)
    spend, top, stop = _fill(rates, means, boxes, target, hulls)
    return spend, top - charge * target, stop


def _fill(
    rates: _Array,
    means: _Array,
    boxes: tuple[tuple[int, int], ...],
    target: float,
    hulls: dict[tuple[int, int, int], _Stretches],
) -> tuple[_Array, float, tuple[int, int, int] | None]:
    """Spend `target` from each channel's first point in its box `(lo, hi)` of `rates`' indices
    down the upper concave envelope of the box's points, the steepest stretch first, an earlier
    channel's and an earlier stretch first on a tie: the spend, its worth on the envelopes, which
    no split in the boxes passes on the interpolants, and the channel and the ends of the stretch
    the spend stops inside where that stretch passes over a point, else None."""
    slopes, runs, channels, ends = [], [], [], []
    for c, (lo, hi) in enumerate(boxes):
        slope, run, pair = _stretches(rates, means, c, lo, hi, hulls)
        slopes.append(slope)
        runs.append(run)
        channels.append(np.full(run.size, c))
        ends.append(pair)
    order = np.argsort(-np.concatenate(slopes), kind="stable")
    spend = np.array([rates[c, lo] for c, (lo, _) in enumerate(boxes)])
    top = float(sum(means[c, lo] for c, (lo, _) in enumerate(boxes)))
    left = target - spend.sum()
    stop = None
    for c, run, slope, (a, b) in zip(
        np.concatenate(channels)[order].tolist(),
        np.concatenate(runs)[order].tolist(),
        np.concatenate(slopes)[order].tolist(),
        np.concatenate(ends)[order].tolist(),
        strict=True,
    ):
        if left <= 0.0:
            break
        take = min(run, left)
        spend[c] += take
        top += slope * take
        left -= take
        if take < run and b - a > 1:
            stop = (c, a, b)
    return spend, top, stop


def _stretches(
    rates: _Array,
    means: _Array,
    c: int,
    lo: int,
    hi: int,
    hulls: dict[tuple[int, int, int], _Stretches],
) -> _Stretches:
    """Channel `c`'s envelope of its points `lo` to `hi`, kept in `hulls` for the boxes that share
    it: each stretch's slope and run, and its ends' indices, a stretch of no length left out."""
    if (c, lo, hi) not in hulls:
        hull = lo + _envelope(rates[c, lo : hi + 1], means[c, lo : hi + 1])
        run = rates[c, hull[1:]] - rates[c, hull[:-1]]
        kept = run > 0.0
        hulls[c, lo, hi] = (
            (means[c, hull[1:]] - means[c, hull[:-1]])[kept] / run[kept],
            run[kept],
            np.column_stack([hull[:-1], hull[1:]])[kept],
        )
    return hulls[c, lo, hi]


def _interpolated(rates: _Array, means: _Array, spend: _Array) -> float:
    """The worth of `spend` on the interpolants of the channels' `means` at `rates`."""
    return float(sum(np.interp(spend[c], rates[c], means[c]) for c in range(spend.size)))
