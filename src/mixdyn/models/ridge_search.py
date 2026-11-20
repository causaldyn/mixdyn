"""A ridge media-mix model's hyperparameters chosen by time-series cross-validation.

Every channel's adstock and Hill, and the ridge penalty, are searched over an interval each; an
interval whose ends are equal fixes its value. The objective is the validation NRMSE over `Folds`:
blocks of periods closing the window, each predicted in turn by the model fitted to the periods
before it, the squared errors pooled over every block and the root of their mean over the KPI's
range on the window's fitted periods. Robyn's DECOMP.RSSD, on the model fitted to the whole window,
can be added at a weight, for workflows that ask the effect shares to keep near the spend shares.

Each fold's design is the window's: the adstock runs from the panel's first period and the Hill's
half-saturation point is read off the whole window, both from spend, which a plan sets, so a fold
reads no KPI after its training periods.

Geo tests calibrate every fit of the search as they calibrate `fit_ridge`'s, each fold's included:
the tests' estimates and noise under each candidate's transforms, and the KPI's noise read off the
periods the fit reads, so a calibrated fold still reads no KPI after its training periods. They
also enter the score, since what a plan reads of a channel, its curve, the coefficient alone does
not carry, and a test's gap rises and falls as the channel's transforms say it should. The score is
the joint least squares of the folds' errors and the tests' errors under the whole window's fit,
each over its noise variance, read once and for every candidate alike: the KPI's as the validation
mean square error of the search without the tests, run first, and the tests' as their observed
gaps' mean square over the readout periods before their groups' spend parts, where no transform
puts a gap. A test period's squared error counts as the ratio of the two, in the KPI's units:

    score = sqrt((sum of the folds' squared errors + ratio * the tests' squared errors) / count)
            / the KPI's range

which without tests is the validation NRMSE. ADR 0008 records why the tests enter the score.

The search runs in the unit cube of the free intervals, every coordinate linear in its interval
but the penalty's, linear in its logarithm. A scrambled Sobol sample of `starts` points is scored;
L-BFGS-B, its gradient by finite differences, polishes the `polish` best of them; the choice is
the best polished point, refitted to the whole window. The polish runs on the score's logarithm:
L-BFGS-B stops on a change below a share of the larger of the score and 1, which on a small score,
as a world without noise gives, stopped it short of the minimum. The objective is smooth wherever
the set of coefficients on their bounds and the period of each channel's most adstocked spend stay
put, and continuous across, so the polish converges locally; the starts carry the search between
basins.
Robyn's search (Nevergrad over in-sample NRMSE and DECOMP.RSSD, a model chosen from the Pareto
front's clusters) is not reproduced.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy import optimize
from scipy.stats import qmc

from mixdyn.models.ridge import ridge_path
from mixdyn.models.ridge_mmm import (
    Control,
    FourierSeason,
    LinearTrend,
    Media,
    RidgeFit,
    RidgeSpec,
    _bounds,
    _check_tests,
    _Data,
    _design,
    _fit,
    _pull,
    _read,
)
from mixdyn.transforms import Geometric, Hill, WeibullCdf

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from chc.lift import LiftTest

    from mixdyn.data import Geo, MediaPanel
    from mixdyn.transforms import Adstock

log = logging.getLogger(__name__)

type _Array = npt.NDArray[np.float64]
type _Mask = npt.NDArray[np.bool_]

_QUIET = 1e-12
"""How near, as a share of the control group's, the treated group's spend at the market's scale is
to it in a period where the two have not parted: rounding apart."""


@dataclass(frozen=True, slots=True)
class Interval:
    """The values from `low` to `high`; equal ends fix the value."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.low) and math.isfinite(self.high) and self.low <= self.high):
            msg = f"an interval needs finite ends, low <= high, got {self.low!r}, {self.high!r}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class GeometricSpace:
    """A geometric adstock, its `theta` searched within `[0, 1)`."""

    theta: Interval

    def __post_init__(self) -> None:
        if not (self.theta.low >= 0.0 and self.theta.high < 1.0):
            msg = f"theta must be searched within [0, 1), got {self.theta}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class WeibullSpace:
    """A Weibull CDF adstock, its `shape` and its `scale` in periods searched over positive
    values."""

    shape: Interval
    scale: Interval

    def __post_init__(self) -> None:
        if not (self.shape.low > 0.0 and self.scale.low > 0.0):
            msg = f"shape and scale must be searched over positive values, got {self}"
            raise ValueError(msg)


type AdstockSpace = GeometricSpace | WeibullSpace


@dataclass(frozen=True, slots=True)
class MediaSpace:
    """A channel of the panel, its adstock and its Hill's `alpha > 0` and `gamma` in `(0, 1]`
    searched."""

    channel: str
    adstock: AdstockSpace
    alpha: Interval
    gamma: Interval

    def __post_init__(self) -> None:
        if not (self.alpha.low > 0.0 and self.gamma.low > 0.0 and self.gamma.high <= 1.0):
            msg = f"{self.channel!r}: need alpha > 0 and gamma in (0, 1], got {self}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class RidgeSpace:
    """The models a search may choose: its channels' spaces, its controls, its trend and season
    if any, and the penalty's interval, by default from nearly unpenalised to shrunk a hundredfold
    the KPI's spread.

    Raises:
        ValueError: on a penalty interval that is not positive, or what `RidgeSpec` refuses.
    """

    media: tuple[MediaSpace, ...]
    controls: tuple[Control, ...] = ()
    penalty: Interval = field(default_factory=lambda: Interval(1e-4, 1e2))
    trend: LinearTrend | None = None
    season: FourierSeason | None = None

    def __post_init__(self) -> None:
        if not self.penalty.low > 0.0:
            msg = f"the penalty must be searched over positive values, got {self.penalty}"
            raise ValueError(msg)
        self.spec(np.full(self.dimension, 0.5))

    def _intervals(self) -> list[Interval]:
        intervals = []
        for m in self.media:
            match m.adstock:
                case GeometricSpace(theta):
                    intervals.append(theta)
                case WeibullSpace(shape, scale):
                    intervals += [shape, scale]
            intervals += [m.alpha, m.gamma]
        return [*intervals, self.penalty]

    @property
    def dimension(self) -> int:
        """How many intervals are not fixed: the search's dimension."""
        return sum(i.low < i.high for i in self._intervals())

    def spec(self, point: npt.ArrayLike) -> RidgeSpec:
        """The model at `point` of the unit cube, a coordinate for each interval not fixed in
        order: each channel's adstock's, then its alpha and gamma, then the penalty.

        Raises:
            ValueError: on a point of the wrong dimension or off the unit cube.
        """
        unit = np.asarray(point, dtype=np.float64)
        if unit.shape != (self.dimension,) or not ((unit >= 0.0) & (unit <= 1.0)).all():
            msg = f"a point of this space is {self.dimension} coordinates in [0, 1], got {unit}"
            raise ValueError(msg)
        coordinates = iter(unit.tolist())
        intervals = self._intervals()
        values = [
            i.low if i.low == i.high else i.low + next(coordinates) * (i.high - i.low)
            for i in intervals[:-1]
        ]
        penalty = self.penalty
        if penalty.low < penalty.high:
            log_low, log_high = math.log(penalty.low), math.log(penalty.high)
            values.append(math.exp(log_low + next(coordinates) * (log_high - log_low)))
        else:
            values.append(penalty.low)
        read = iter(values)
        media = []
        for m in self.media:
            adstock: Adstock
            match m.adstock:
                case GeometricSpace():
                    adstock = Geometric(next(read))
                case WeibullSpace():
                    adstock = WeibullCdf(next(read), next(read))
            media.append(Media(m.channel, adstock, Hill(next(read), next(read))))
        return RidgeSpec(tuple(media), self.controls, next(read), self.trend, self.season)


@dataclass(frozen=True, slots=True)
class Folds:
    """`count` consecutive blocks of `horizon` periods that close the window, each predicted by the
    model fitted to the window's periods before it.

    Raises:
        ValueError: on a count or a horizon below 1.
    """

    count: int
    horizon: int

    def __post_init__(self) -> None:
        if not (self.count >= 1 and self.horizon >= 1):
            msg = (
                f"folds need a count and a horizon of at least 1, got {self.count}, {self.horizon}"
            )
            raise ValueError(msg)

    @classmethod
    def within(cls, periods: int, *, most: int, horizon: int, least: int) -> Folds:
        """At most `most` blocks of `horizon` periods closing a window of `periods`, as many as
        leave the first fold at least `least` periods to fit: on a short window the earliest
        blocks are dropped, so every fold keeps a training prefix and the latest blocks are still
        predicted.

        Raises:
            ValueError: on a `most`, `horizon` or `least` below 1, or a window too short for one
                block after `least` periods.
        """
        if not (most >= 1 and horizon >= 1 and least >= 1):
            msg = (
                "folds need a most, a horizon and a least of at least 1,"
                f" got {most}, {horizon}, {least}"
            )
            raise ValueError(msg)
        count = min(most, (periods - least) // horizon)
        if count < 1:
            msg = (
                f"a window of {periods} periods holds no block of {horizon} after the {least}"
                " a fold fits at least"
            )
            raise ValueError(msg)
        return cls(count, horizon)

    def masks(self, fitted: npt.ArrayLike) -> tuple[tuple[_Mask, _Mask], ...]:
        """Each fold's training and held-out periods among the window's `fitted` ones.

        Raises:
            ValueError: on blocks that cover the window, or a fold with fewer than two periods to
                fit or none to predict.
        """
        observed = np.asarray(fitted, dtype=bool)
        periods = np.arange(observed.size)
        first = observed.size - self.count * self.horizon
        if first < 1:
            msg = (
                f"{self.count} blocks of {self.horizon} periods leave none of the window's"
                f" {observed.size} periods to fit"
            )
            raise ValueError(msg)
        masks = []
        for k in range(self.count):
            start = first + k * self.horizon
            train = observed & (periods < start)
            held = observed & (periods >= start) & (periods < start + self.horizon)
            if train.sum() < 2 or not held.any():
                msg = (
                    f"fold {k + 1} of {self.count}, periods {start + 1} to {start + self.horizon}"
                    f" of the window's {observed.size}, fits {train.sum()} observed periods and"
                    f" predicts {held.sum()}; it needs at least 2 and 1"
                )
                raise ValueError(msg)
            masks.append((train, held))
        return tuple(masks)


@dataclass(frozen=True, slots=True)
class Optimum:
    """A polished point's model and its score."""

    spec: RidgeSpec
    score: float


@dataclass(frozen=True, slots=True, eq=False)
class RidgeSearch:
    """The chosen model fitted to the whole window, and the search behind it.

    `optima` holds each polished point's model and score, the best first: the choice is `fit.spec`,
    `optima[0]`. Optima far apart at near scores mark hyperparameters the data hardly tell apart.
    `evaluations` counts the objective's evaluations, those of the search without tests included.
    """

    fit: RidgeFit
    score: float
    optima: tuple[Optimum, ...]
    evaluations: int


@dataclass(frozen=True, slots=True, eq=False)
class _Calibrated:
    """The geo tests a search reads, and the KPI's noise variance over theirs, the weight of a test
    period's squared error in the score."""

    tests: Mapping[str, Sequence[LiftTest]]
    ratio: float


def _quiet(tests: Mapping[str, Sequence[LiftTest]]) -> float:
    """The tests' noise variance: their observed gaps' mean square over the readout periods before
    their groups' spend parts, at the market's scale, where no adstock or Hill puts a gap.

    Raises:
        ValueError: on tests with no such period, or gaps of 0 throughout them.
    """
    quiet = []
    for test in (test for channel in tests.values() for test in channel):
        treated = test.treated.spend / test.treated.share
        control = test.control.spend / test.control.share
        parted = np.cumsum(~np.isclose(treated, control, rtol=_QUIET, atol=0.0)) > 0
        quiet.append(test.difference[~parted[test.treated.history :]])
    gaps = np.concatenate(quiet)
    if not np.any(gaps):
        msg = (
            f"the tests' {gaps.size} readout periods before their groups' spend parts hold no gap,"
            " and the search's score reads the tests' noise off them"
        )
        raise ValueError(msg)
    return float(np.mean(gaps**2))


def _objective(
    data: _Data,
    spec: RidgeSpec,
    masks: tuple[tuple[_Mask, _Mask], ...],
    span: float,
    rssd_weight: float,
    calibrated: _Calibrated | None,
) -> float:
    design, halves = _design(data, spec)
    lower, upper = _bounds(spec)
    tests = calibrated.tests if calibrated else None
    pull = _pull(spec, design.shape[1], halves, tests) if tests else None
    squares, count = 0.0, 0
    for train, held in masks:
        x, y = design[train], data.y[train]
        anchor = pull.anchor(x, y, spec)[0] if pull else None
        path = ridge_path(x, y, [spec.penalty], lower, upper, anchor)
        residuals = data.y[held] - path.intercepts[0] - design[held] @ path.coefficients[0]
        squares += float(residuals @ residuals)
        count += residuals.size
    if calibrated and pull:
        x, y = design[data.fitted], data.y[data.fitted]
        whole = ridge_path(x, y, [spec.penalty], lower, upper, pull.anchor(x, y, spec)[0])
        squares += calibrated.ratio * pull.misses(whole.coefficients[0])
    score = math.sqrt(squares / count) / span
    if rssd_weight > 0.0:
        score += rssd_weight * _fit(data, spec, tests).decomp_rssd()
    return score


def _optima(
    objective: Callable[[_Array], float], d: int, seed: int, starts: int, polish: int
) -> list[tuple[float, _Array]]:
    """Each polished point's score and coordinates, the best first."""
    if d == 0:
        return [(objective(np.zeros(0)), np.zeros(0))]
    points = qmc.Sobol(d, scramble=True, seed=seed).random(starts)
    scores = np.array([objective(p) for p in points])
    optima = []
    for i in np.argsort(scores, kind="stable")[:polish]:
        result = optimize.minimize(
            lambda point: math.log(objective(point)),
            points[i],
            method="L-BFGS-B",
            bounds=[(0.0, 1.0)] * d,
        )
        optima.append((math.exp(result.fun), result.x))
        log.debug("ridge search polished start %d from %.6g to %.6g", i, scores[i], optima[-1][0])
    optima.sort(key=lambda o: o[0])
    return optima


def search_ridge(
    panel: MediaPanel,
    geo: Geo,
    space: RidgeSpace,
    folds: Folds,
    *,
    seed: int,
    window: tuple[np.datetime64, np.datetime64] | None = None,
    rssd_weight: float = 0.0,
    starts: int = 256,
    polish: int = 8,
    tests: Mapping[str, Sequence[LiftTest]] | None = None,
) -> RidgeSearch:
    """Choose the model of `space` that predicts one geo's `folds` best, and fit it to `window`;
    given `tests`, each channel's geo tests, the one that predicts the folds and the tests best,
    each fit calibrated by them.

    `seed` scrambles the Sobol sample; `starts` points are scored and the `polish` best polished,
    by the search without the tests first where there are tests.

    Raises:
        ValueError: on what `fit_ridge` or `Folds.masks` refuses, a channel with no spend in the
            window, a KPI constant over it, a weight that is negative or not finite, `polish`
            outside `1..starts`, or tests with no gap before their groups' spend parts.
    """
    if not 0.0 <= rssd_weight < math.inf:
        msg = f"the DECOMP.RSSD weight must be nonnegative and finite, got {rssd_weight!r}"
        raise ValueError(msg)
    if not 1 <= polish <= starts:
        msg = f"need 1 <= polish <= starts, got {polish} and {starts}"
        raise ValueError(msg)
    middle = space.spec(np.full(space.dimension, 0.5))
    if tests:
        _check_tests(middle, tests)
    data = _read(panel, geo, middle, window)
    dark = [
        m.channel
        for m, s in zip(space.media, data.spend[data.inside].T, strict=True)
        if not s.any()
    ]
    if dark:
        msg = f"the channels {dark} spend nothing in the window"
        raise ValueError(msg)
    masks = folds.masks(data.fitted)
    observed = data.y[data.fitted]
    span = float(observed.max() - observed.min())
    if span == 0.0:
        msg = "the KPI is constant over the window, so its NRMSE has no range to read"
        raise ValueError(msg)
    evaluations = 0

    def scored(rssd: float, calibrated: _Calibrated | None) -> Callable[[_Array], float]:
        def objective(point: _Array) -> float:
            nonlocal evaluations
            evaluations += 1
            return _objective(data, space.spec(point), masks, span, rssd, calibrated)

        return objective

    d = space.dimension
    calibrated = None
    if tests:
        noise = _quiet(tests)
        plain = _optima(scored(0.0, None), d, seed, starts, polish)[0][0]
        calibrated = _Calibrated(tests, (plain * span) ** 2 / noise)
        log.info(
            "ridge search calibrated for geo %s: a test period weighs %.6g of a KPI period",
            geo,
            calibrated.ratio,
            extra={"geo": str(geo), "ratio": calibrated.ratio},
        )
    optima = _optima(scored(rssd_weight, calibrated), d, seed, starts, polish)
    best = space.spec(optima[0][1])
    log.info(
        "ridge hyperparameters chosen for geo %s: %d coordinates, score %.6g in %d evaluations",
        geo,
        d,
        optima[0][0],
        evaluations,
        extra={"geo": str(geo), "dimension": d, "score": optima[0][0], "evaluations": evaluations},
    )
    return RidgeSearch(
        fit=_fit(data, best, tests),
        score=optima[0][0],
        optima=tuple(Optimum(space.spec(x), score) for score, x in optima),
        evaluations=evaluations,
    )
