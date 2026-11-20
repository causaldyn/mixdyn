"""Lift tests not yet run, as a plan reads them: each a candidate design, and the lift a posterior
expects it to read.

A `Candidate` is a lift test of one channel before it runs, a `mixdyn.calibration.LiftEstimate`
without its lift: the tested group's spend on the channel as the test would run it and as planned,
over the periods before the readout and the readout, the readout's length, the group's share of
the market and the standard error its analysis would report. `Candidate.estimate` is the estimate
it reports once its lift is read.

`GroupNoise` sets that error from the market's noise. A group of share `s` is read against the rest
of the market scaled to its size, and a period's noise has a variance in proportion to the share of
the market it is summed over: over a readout of `R` periods the lift's standard error is
`market_sd sqrt(R s / (1 - s))`, `market_sd` the standard deviation of the market's noise a period.
That holds where the periods' noises are independent. Where they are not, as a geo's noise that
carries over from one period to the next, the readout's sum is wider: under a first-order
autoregression of coefficient `phi` its variance is `f(R, phi) = 1 + 2 sum_{k<R} (1 - k/R) phi^k`
times `R` periods', `(1 + phi) / (1 - phi)` for a long readout, so at 12 periods `GroupNoise`'s
error is `1.63` times too small at `phi = 1/2` and `2.43` times at `phi = 4/5`.

`PlaceboNoise` reads the error from the market's own noise over the history instead, as a test's
analysis would read it on placebo windows of the pre-period. Each window of `R` periods sums the
noise, studentised by the root mean square of the noise over the `pre` periods before it; the
error is the root mean square of those studentised sums times the same root mean square over the
`pre` periods before the test, times `sqrt(s / (1 - s))` as `GroupNoise`'s, widened so that its
normal 95 % interval is a t's at the windows' effective count `nu`, by `t_nu(0.975) / z(0.975)`.
`nu` is the count of independent normal squares whose mean is as noisy as the windows' mean square,
`nu = 2 n^2 / ((kappa - 1) sum_{f, f'} rho(f - f')^2)` over the `n` windows: they overlap and the
noise may carry over, `rho` is the studentised sums' own autocorrelation, uncentred, at every lag,
and `kappa` their kurtosis, at least a normal's 3, as a sum over a scale read with error has.
Whatever the noise's dependence within a window, the sums hold it, so carried-over noise is read as
it is.
Studentising each window by its own scale is Chung and Romano's (2013, Theorem 2.2) correction of a
permutation test whose units are not exchangeable: an unstudentised placebo of a noise whose level
moves over the history reads the history's mean level, a studentised one the level before the test.

A test read period by period (`mixdyn.calibration.LiftSeries`) needs the covariance of its gaps'
errors, `covariance`. `GroupNoise`'s holds each period's variance on the diagonal. `PlaceboNoise`'s
is the residuals' uncentred autocorrelation at each lag below the readout as a Toeplitz matrix,
positive definite as a sample autocorrelation's is, scaled so that its sum is the square of the
error `se` reads for the sum: the gaps' sum has the summed test's error, and the dependence's shape
over the readout is the history's.

`designs` lays out each channel's candidates: its spend multiplied by each multiplier over each
duration, then a cooldown at the spend planned, the two read together, in a group of each share.
The test runs after the panel at the plan in place (`After`), or inside its history from one of its
periods (`Within`).

`seesaw` lays out a test that moves the spend above and below the plan in alternate blocks of
the same weeks, which separates two readings that a history whose spend moved slowly cannot tell
apart (Dew, Padilla and Shchetkina 2024): a channel whose effect a currency unit drifts over time,
its response a line in the spend, and one whose effect is fixed on an S-shaped curve. Where the
history's spend rose along the curve, the line's slope at each period is the curve's average
return at that period's spend, so at the spend planned the two return the same, and at no spend
both return nothing. A test that goes dark compares those two points, and reads what the two
readings share. They differ above and below the plan: a line's response to a change is its slope
times the change, the same up as down, a curve's its rise over the change, steeper on one side
than on the other. Under a line the gaps of a block above the plan and of one as far below are
equal and opposite, whatever the slope, so a seesaw whose high and low blocks move the spend by as
much sums to no lift over a readout that holds the cooldown's carryover; under a curve they are
not, and read period by period (`mixdyn.calibration.LiftSeries`) the high blocks and the low ones
read the curve's rise on each side of the plan. Alternating keeps both levels within weeks that a
slowly drifting slope moves alike, which a heavy-up and a cut in separate months would not.

`predicted_lift` is the lift each of a posterior's draws expects a candidate to read at the
market's scale: the channel's coefficient times the gap in its curve over the readout between the
spend as the test runs and as planned, each over the share and carried over by the draw's kernel,
as the Bayesian model's likelihood reads an estimate (`mixdyn.models.bayes_mmm`). A group's
reported lift is normal about the share times it, with the candidate's standard error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy import linalg, stats

from mixdyn.calibration import LiftEstimate
from mixdyn.decisions.posterior import _carried, _whole
from mixdyn.models.bayes_mmm import _TANH_HALF

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mixdyn.data import Channel, Geo, MediaPanel
    from mixdyn.decisions import Readings

type _Array = npt.NDArray[np.float64]

_UPPER = 0.975  # the quantile a placebo standard error's 95 % interval is widened at


@dataclass(frozen=True, slots=True, eq=False)
class Candidate:
    """A lift test of `channel` before it runs: a `LiftEstimate` without its lift, and a `label`
    to name it by.

    Raises:
        ValueError: on what `LiftEstimate` refuses of its spend, readout, standard error and share.
    """

    channel: Channel
    spend: _Array
    planned: _Array
    readout: int
    se: float
    share: float
    label: str = ""

    def __post_init__(self) -> None:
        checked = self.estimate(0.0)
        object.__setattr__(self, "spend", checked.spend)
        object.__setattr__(self, "planned", checked.planned)

    def estimate(self, lift: float, se: float | None = None) -> LiftEstimate:
        """The estimate the test reports once it has run: `lift` over the readout, at the market's
        share the group is, with the candidate's standard error or `se` where the analysis read
        another."""
        return LiftEstimate(
            self.spend,
            self.planned,
            self.readout,
            lift,
            self.se if se is None else se,
            self.share,
        )


@dataclass(frozen=True, slots=True)
class GroupNoise:
    """The standard deviation of the market's KPI noise a period, as it reaches a group's lift.

    Raises:
        ValueError: on a standard deviation not positive and finite.
    """

    market_sd: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.market_sd) and self.market_sd > 0.0):
            msg = f"market_sd={self.market_sd!r} is not positive and finite"
            raise ValueError(msg)

    def se(self, readout: int, share: float) -> float:
        """The standard error of the lift of a group of `share` of the market over `readout`
        periods, read against the rest scaled to its size.

        Raises:
            ValueError: on a readout not a whole number, at least 1, and a share outside `(0, 1)`.
        """
        _whole(readout, "readout", 1)
        if not (math.isfinite(share) and 0.0 < share < 1.0):
            msg = f"share={share!r} is not in (0, 1): a group needs the rest of the market"
            raise ValueError(msg)
        return self.market_sd * math.sqrt(readout * share / (1.0 - share))

    def covariance(self, readout: int, share: float) -> _Array:
        """The covariance of the lift's errors period by period over `readout` periods, `(readout,
        readout)`: each period's variance `market_sd^2 s / (1 - s)`, the periods independent.

        Raises:
            ValueError: on what `se` refuses.
        """
        return np.eye(readout) * self.se(readout, share) ** 2 / readout


@dataclass(frozen=True, slots=True, eq=False)
class PlaceboNoise:
    """The market's KPI noise a period over the history before a test, `residuals`, the error a
    model of the KPI leaves in each period, and the `pre` periods each placebo window and the test
    are studentised over; see the module.

    Raises:
        ValueError: on residuals not a one-dimensional series of finite numbers, and `pre` not a
            whole number, at least 2.
    """

    residuals: _Array
    pre: int = 26

    def __post_init__(self) -> None:
        residuals = np.array(self.residuals, dtype=np.float64)
        if residuals.ndim != 1 or not np.isfinite(residuals).all():
            msg = f"residuals must be a one-dimensional finite series, got shape {residuals.shape}"
            raise ValueError(msg)
        _whole(self.pre, "pre", 2)
        residuals.setflags(write=False)
        object.__setattr__(self, "residuals", residuals)

    def se(self, readout: int, share: float) -> float:
        """The standard error of the lift of a group of `share` of the market over `readout`
        periods after the residuals, read against the rest scaled to its size, from the placebo
        windows of `readout` periods with `pre` periods before them.

        Raises:
            ValueError: on a readout not a whole number, at least 1, and a share outside `(0, 1)`;
                residuals too short for a placebo window; residuals 0 over some window's `pre`
                periods before it, or the test's, which nothing studentises; and windows that all
                sum to 0.
        """
        _whole(readout, "readout", 1)
        if not (math.isfinite(share) and 0.0 < share < 1.0):
            msg = f"share={share!r} is not in (0, 1): a group needs the rest of the market"
            raise ValueError(msg)
        e, pre = self.residuals, self.pre
        windows = e.size - pre - readout + 1
        if windows < 1:
            msg = (
                f"{e.size} periods of residuals hold no placebo window of {readout} periods after "
                f"{pre} before it"
            )
            raise ValueError(msg)
        sums = np.concatenate([[0.0], np.cumsum(e)])
        squares = np.concatenate([[0.0], np.cumsum(e * e)])
        firsts = np.arange(pre, pre + windows)
        ends = np.append(firsts, e.size)
        scales = np.sqrt((squares[ends] - squares[ends - pre]) / pre)
        if not (scales > 0.0).all():
            msg = f"the residuals are 0 over {pre} periods, which studentise nothing"
            raise ValueError(msg)
        scale, before = scales[:-1], float(scales[-1])
        t = (sums[firsts + readout] - sums[firsts]) / scale
        square = float(t @ t)
        if not square > 0.0:
            msg = f"every placebo window of {readout} periods sums to 0"
            raise ValueError(msg)
        nu = _effective_count(t, square)
        widened = float(stats.t.ppf(_UPPER, nu) / stats.norm.ppf(_UPPER))
        return before * math.sqrt(square / windows) * widened * math.sqrt(share / (1.0 - share))

    def covariance(self, readout: int, share: float) -> _Array:
        """The covariance of the lift's errors period by period over `readout` periods after the
        residuals, `(readout, readout)`: the residuals' own autocorrelation, uncentred, at each lag
        below `readout`, as a Toeplitz matrix, scaled so that its sum, the variance of the lift
        summed over the readout, is `se(readout, share)` squared.

        Raises:
            ValueError: on what `se` refuses.
        """
        total = self.se(readout, share)
        e = self.residuals
        rho = np.array([e[: e.size - k] @ e[k:] for k in range(readout)]) / float(e @ e)
        shape = linalg.toeplitz(rho)
        return total**2 * shape / shape.sum()


def _effective_count(t: _Array, square: float) -> float:
    """The count of independent normal squares whose mean is as noisy as the mean square of the
    placebo windows' sums `t`, `square` their sum of squares, above 0: `2 n^2 / ((kappa - 1)
    sum_{f, f'} rho(f - f')^2)`, `rho` the sums' uncentred autocorrelation and `kappa` their
    kurtosis, at least 3; see the module."""
    windows = t.size
    spectrum = np.fft.rfft(t, 2 * windows)
    rho = np.fft.irfft(spectrum.real**2 + spectrum.imag**2, 2 * windows)[:windows] / square
    pairs = windows + 2.0 * float(np.arange(windows - 1, 0, -1) @ rho[1:] ** 2)
    kurtosis = max(float(np.mean(t**4)) * windows**2 / square**2, 3.0)
    return 2.0 * windows**2 / ((kurtosis - 1.0) * pairs)


type Noise = GroupNoise | PlaceboNoise


@dataclass(frozen=True, slots=True)
class After:
    """A test run after the panel, its periods at `current`, each channel's spend a period under
    the plan in place, but for the test's change."""

    current: Mapping[Channel, float]


@dataclass(frozen=True, slots=True)
class Within:
    """A test run inside the panel's history from its period `start`, counted from 0, at the
    market's spend there but for the test's change."""

    start: int


type When = After | Within


def designs(
    panel: MediaPanel,
    geo: Geo,
    channels: Sequence[Channel],
    noise: Noise,
    *,
    when: When,
    multipliers: Sequence[float] = (0.0, 0.5, 1.5, 2.0),
    durations: Sequence[int] = (2, 4, 8),
    cooldown: int = 8,
    shares: Sequence[float] = (0.05, 0.1, 0.2, 0.3, 0.5),
) -> tuple[Candidate, ...]:
    """Each of `channels`' candidates on `geo` of `panel`, in the order of the channels, the
    multipliers, the durations and the shares: the channel's spend times the multiplier over the
    duration, then `cooldown` periods at the spend planned, both read, in a group of the share,
    its standard error `noise`'s, run as `when` says. The group's spend is the share's of the
    market's, the panel's own over its history.

    Raises:
        ValueError: on a geo or a channel the panel does not have; no channels or a channel named
            twice; a multiplier not finite and at or above 0, or 1, which changes nothing; a
            duration not a whole number, at least 1; a cooldown not a whole number, at least 0; a
            share outside `(0, 1)`; a plan in place that misses a channel or spends one below 0 or
            not finitely; a start from which a test and its cooldown do not end within the history;
            and a channel that spends nothing over a test's periods, which no multiplier changes.
    """
    if geo not in panel.geos:
        msg = f"the panel has no geo {geo!r}: it has {list(panel.geos)}"
        raise ValueError(msg)
    unknown = [c for c in channels if c not in panel.channels]
    if not channels or unknown or len(set(channels)) != len(channels):
        msg = f"designs need channels of the panel, each once: got {list(channels)}"
        raise ValueError(msg)
    if not all(math.isfinite(m) and m >= 0.0 and m != 1.0 for m in multipliers):
        msg = f"multipliers must be finite, at or above 0 and not 1, got {list(multipliers)}"
        raise ValueError(msg)
    for duration in durations:
        _whole(duration, "a duration", 1)
    _whole(cooldown, "cooldown", 0)
    for share in shares:
        noise.se(1, share)
    history = panel.spend[panel.geos.index(geo)]
    out = []
    for channel in channels:
        market = history[:, panel.channels.index(channel)]
        for multiplier in multipliers:
            for duration in durations:
                planned, first = _planned(market, channel, when, duration + cooldown)
                tested = planned.copy()
                tested[first : first + duration] *= multiplier
                if np.array_equal(tested, planned):
                    msg = (
                        f"channel {channel!r} spends nothing over the {duration} periods from "
                        f"{first}, which no multiplier changes"
                    )
                    raise ValueError(msg)
                out += [
                    Candidate(
                        channel,
                        share * tested,
                        share * planned,
                        duration + cooldown,
                        noise.se(duration + cooldown, share),
                        share,
                        f"{channel} x{multiplier:g} for {duration} at {share:g}",
                    )
                    for share in shares
                ]
    return tuple(out)


def seesaw(
    panel: MediaPanel,
    geo: Geo,
    channels: Sequence[Channel],
    noise: Noise,
    *,
    when: When,
    high: float = 1.5,
    low: float = 0.5,
    block: int = 2,
    blocks: int = 8,
    cooldown: int = 8,
    shares: Sequence[float] = (0.05, 0.1, 0.2, 0.3, 0.5),
) -> tuple[Candidate, ...]:
    """Each of `channels`' seesaw candidates on `geo` of `panel`, in the order of the channels and
    the shares: the channel's spend times `high` and `low` in turn, `block` periods each, over
    `blocks` blocks from a high one, then `cooldown` periods at the spend planned, all read, in a
    group of the share, its standard error `noise`'s, run as `when` says; see the module. The
    group's spend is the share's of the market's, the panel's own over its history. At a spend
    planned alike over the test, multipliers as far above 1 as below it change no spend in all.

    Raises:
        ValueError: on a geo or a channel the panel does not have; no channels or a channel named
            twice; a `high` not finite and above 1, or a `low` not finite, at or above 0 and below
            1; a block not a whole number, at least 1; blocks not an even whole number, at least
            2; a cooldown not a whole number, at least 0; a share outside `(0, 1)`; what `designs`
            refuses of `when`; and a channel that spends nothing over the test's periods.
    """
    if geo not in panel.geos:
        msg = f"the panel has no geo {geo!r}: it has {list(panel.geos)}"
        raise ValueError(msg)
    unknown = [c for c in channels if c not in panel.channels]
    if not channels or unknown or len(set(channels)) != len(channels):
        msg = f"designs need channels of the panel, each once: got {list(channels)}"
        raise ValueError(msg)
    if not (math.isfinite(high) and high > 1.0 and math.isfinite(low) and 0.0 <= low < 1.0):
        msg = f"a seesaw needs a finite high above 1 and a low in [0, 1), got {high!r}, {low!r}"
        raise ValueError(msg)
    _whole(block, "block", 1)
    _whole(blocks, "blocks", 2)
    if blocks % 2:
        msg = f"blocks must be even, each high block with a low one, got {blocks}"
        raise ValueError(msg)
    _whole(cooldown, "cooldown", 0)
    for share in shares:
        noise.se(1, share)
    pattern = np.tile(np.repeat([high, low], block), blocks // 2)
    history = panel.spend[panel.geos.index(geo)]
    out = []
    for channel in channels:
        market = history[:, panel.channels.index(channel)]
        planned, first = _planned(market, channel, when, pattern.size + cooldown)
        tested = planned.copy()
        tested[first : first + pattern.size] *= pattern
        if np.array_equal(tested, planned):
            msg = (
                f"channel {channel!r} spends nothing over the {pattern.size} periods from "
                f"{first}, which no multiplier changes"
            )
            raise ValueError(msg)
        out += [
            Candidate(
                channel,
                share * tested,
                share * planned,
                pattern.size + cooldown,
                noise.se(pattern.size + cooldown, share),
                share,
                f"{channel} seesaw x{high:g}/x{low:g} by {block} over {blocks} at {share:g}",
            )
            for share in shares
        ]
    return tuple(out)


def predicted_lift(readings: Readings, candidate: Candidate) -> _Array:
    """Each draw's lift of `candidate` at the market's scale, `(draws,)`, in the KPI's units:
    the channel's coefficient times the gap in its curve over the readout between the spend as the
    test runs and as planned, each over the share and carried over by the draw's kernel, nothing
    spent before the candidate's periods. The kernel is the longer of the readings' and the
    candidate's periods, as the model that reads the estimate has it.

    Raises:
        ValueError: on a candidate of a channel the readings do not have.
    """
    if candidate.channel not in readings.channels:
        msg = f"the readings have no channel {candidate.channel!r}: they have {readings.channels}"
        raise ValueError(msg)
    c = readings.channels.index(candidate.channel)
    length = max(readings.length, candidate.spend.size)
    weights = readings.retention[:, c, None] ** np.arange(length)
    weights /= weights.sum(axis=1, keepdims=True)
    start = candidate.spend.size - candidate.readout
    half = readings.half_saturation[:, c, None]
    gap = np.zeros(readings.retention.shape[0])
    for series, sign in ((candidate.spend, 1.0), (candidate.planned, -1.0)):
        adstock = _carried(series / candidate.share, weights, start)
        with np.errstate(divide="ignore", over="ignore"):
            curve = np.where(
                readings.tanh[:, None],
                np.tanh(_TANH_HALF * adstock / half),
                np.where(
                    adstock > 0.0,
                    1.0 / (1.0 + (half / adstock) ** readings.slope[:, c, None]),
                    0.0,
                ),
            )
        gap += sign * curve.sum(axis=1)
    return readings.coefficient[:, c] * gap


def _planned(market: _Array, channel: Channel, when: When, periods: int) -> tuple[_Array, int]:
    """The market's spend on `channel` as planned, through a test of `periods` periods and its
    cooldown, and the test's first period."""
    match when:
        case After(current=current):
            if channel not in current:
                msg = f"the plan in place has no spend for channel {channel!r}"
                raise ValueError(msg)
            rate = float(current[channel])
            if not (math.isfinite(rate) and rate >= 0.0):
                msg = f"the plan in place spends {rate!r} a period on {channel!r}"
                raise ValueError(msg)
            return np.r_[market, np.full(periods, rate)], market.size
        case Within(start=start):
            _whole(start, "start", 0)
            if start + periods > market.size:
                msg = (
                    f"a test of {periods} periods with its cooldown from period {start} ends past "
                    f"the history's {market.size}"
                )
                raise ValueError(msg)
            return market[: start + periods].copy(), start
