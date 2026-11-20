"""Lift tests as anchors for a regression's margins: each channel's tests read as a return per euro
and as a slope of that return in the spend, and the tests combined with a regression's margins whose
bias the tests reveal.

**A channel's tests** (`read_tests`). A go-dark test of a channel reports its weekly spend `x` over
the dark weeks, which went to 0, and the KPI's gap `delta_y`, summed over the dark weeks and the
cooldown after them and divided by the dark weeks, with its standard error `sigma`: its return per
euro is `rho = -delta_y / x`, of standard error `sigma / x`, the carryover the readout holds counted
in. The channel's return per euro `roi` is the precision-weighted mean of its tests', of standard
error `1 / sqrt(sum_j 1 / se_j^2)`, every test read whatever its spend, so on a straight-line
response it is the line's slope. Its slope statistic `slope_t` is the t-statistic of the weighted
least-squares slope of its tests' returns on `ln(x / s)`, `s` the channel's status quo, weights
`1 / se^2` and the variances as the tests state them: on a curve through 0, the return per euro's
slope in `ln x` has the sign of the elasticity less 1, so a return that rises with the spend
significantly marks the convex part of an S-curve, which a straight line misreads. Tests that sit
at one spend, `ln(x / s)` within `FLAT` of one another, read no slope: 0.

**The anchor** (`anchor`). A regression's margins `b`, of covariance `V`, are the true margins `m`
plus a bias `delta` the history's confounding leaves and their noise; the tests' returns `r`, of
standard errors `se`, are `m` plus noise alone, as the tests are randomised. With
`delta ~ N(0, tau^2 I)` and a flat prior on `m`, `m`'s posterior has precision
`P = (V + tau^2 I)^-1 + diag(1 / se^2)` and mean `P^-1 ((V + tau^2 I)^-1 b + diag(1 / se^2) r)`:
the regression alone where the tests are vague, the tests alone as `tau` grows. Without bias,
`E[(b_c - r_c)^2] = V_cc + se_c^2`, so `tau^2` is read by moments,
`max(0, mean_c((b_c - r_c)^2 - V_cc - se_c^2))`, where the tests and the regression disagree by
more than their stated errors allow.

**The curves** (`read_curves`). A straight line misreads a concave response: a test far above the
status quo reads a chord deep in the saturation, and the regression reads the curve's mean slope
over the history. Each channel carries a posterior over curve columns through its margin at the
status quo `s`: Michaelis-Menten curves `beta a / (K + a)` of the adstock `a`, `K = s e / (1 - e)`,
and logistic curves `beta tanh(k a)`, `2 k s / sinh(2 k s) = e` (`logistic_rate`), each at every
elasticity at the status quo `e` of `ELASTICITIES`, and the straight line, of elasticity 1. The
prior puts `LINE_PRIOR` on the line and splits the rest evenly by the two families and uniformly
over the elasticities. Given a column, one weighted least squares reads the channel's margin at the
status quo `m` off the regression's margin and each test's return per euro, each the margin times
a factor the column fixes:

- the regression's margin `b_c = m B / B_line + noise` of variance `V_cc + tau^2`, `tau^2` the
  anchor's: `B` is the coefficient the regression reads off the column's curve of the channel's
  adstock, per unit of the curve's margin at `s` (its `slope`, the regression as a linear map at
  its retention combination of the most evidence), and `B_line` the same off the adstock itself;
- each test's return `rho_j = m R_j / R_line_j + noise` of variance `se_j^2`: `R_j` is the exact
  lift the curve gives the test, the curve at the history's adstock less the curve at the adstock
  with the test's dark weeks at no spend, summed over the readout, over the dark weeks' spend, per
  unit of the curve's margin at `s`, and `R_line_j` the line's, the share of the dark spend's
  carryover the readout holds.

A column's weight is its prior times its marginal likelihood under a flat prior on `m`,
`exp(-rss / 2) / sqrt(information)`, and its `ratio` is its least-squares margin over the line's.
The channel's margin curve is its level, the anchored and shrunk margin, times the columns' shapes,
each its margin at a spend over its margin at `s`, `((K + s) / (K + p))^2`,
`(cosh(k s) / cosh(k p))^2` and 1, weighed by the posterior times the ratio held at 0 or above
(`Curves.scale`); with the line's prior at 1 the curve is the level, a straight line. The tests'
adstock reads the history before its first week as no spend, the kernel at the regression's
retention of the most evidence for the channel.

The arithmetic follows the research code that found the rules step for step, so its readings are
reproduced to rounding (`tests/fixtures/gs_worlds.json`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Callable

type _Array = npt.NDArray[np.float64]

FLAT = 1e-9
"""How far apart in `ln(x / s)` a channel's tests must lie to read a slope."""
ELASTICITIES = np.r_[np.linspace(0.05, 0.95, 37), 0.97, 0.98, 0.99, 0.995]
"""The elasticities at the status quo each curve family's columns sit at (`read_curves`)."""
LINE_PRIOR = 0.7
"""The prior's mass on the straight line among a channel's curve columns."""
RATE_BISECTIONS = 200
"""The bisections of a logistic column's rate (`logistic_rate`)."""
RATE_MOST = 60.0
"""The most `k s` a logistic column's rate is searched to."""


@dataclass(frozen=True, slots=True, eq=False)
class ChannelTests:
    """Each channel's tests, in the channels' order: `roi`, the precision-weighted mean of the
    tests' returns per euro, and `roi_se` its standard error; `slope_t`, the slope statistic of the
    tests' returns in `ln(x / s)`, 0 where they sit at one spend; and `count`, the channel's tests.
    """

    roi: _Array
    roi_se: _Array
    slope_t: _Array
    count: tuple[int, ...]


@dataclass(frozen=True, slots=True, eq=False)
class Anchored:
    """The margins' posterior with the tests as anchors: `mean` and `cov`, and `tau`, the
    regression's bias read by moments, the square root of `tau^2`."""

    mean: _Array
    cov: _Array
    tau: float


@dataclass(frozen=True, slots=True, eq=False)
class Curves:
    """Each channel's posterior over its curve columns (`read_curves`): `status_quo`, each
    channel's weekly spend `s_c`; `half`, its Michaelis-Menten columns' half-saturations `K`, and
    `rate`, its logistic columns' `k`, `(channels, elasticities)`; `weights`, the posterior over
    the columns, the Michaelis-Menten ones, the logistic ones and the line last,
    `(channels, columns)`; and `ratio`, each column's margin at the status quo over the line's."""

    status_quo: _Array
    half: _Array
    rate: _Array
    weights: _Array
    ratio: _Array

    def scale(self) -> _Array:
        """Each column's weight in a channel's margin curve, its posterior weight times its
        ratio held at 0 or above, `(channels, columns)`."""
        return self.weights * np.clip(self.ratio, 0.0, None)

    def shapes(self, channel: int, spend: npt.ArrayLike) -> _Array:
        """Each column's margin at the weekly spends `spend`, `(points,)`, over its margin at the
        status quo, `(points, columns)`."""
        p = np.asarray(spend, dtype=float)
        s = float(self.status_quo[channel])
        cols = [((half + s) / (half + p)) ** 2 for half in self.half[channel]]
        cols += [(np.cosh(k * s) / np.cosh(k * p)) ** 2 for k in self.rate[channel]]
        cols.append(np.ones_like(p))
        return np.column_stack(cols)

    def margin(self, channel: int, level: float) -> Callable[[_Array], _Array]:
        """The channel's margin at weekly spends, `level` times its columns' shapes weighed by
        `scale`; where `level` is not positive, a line: its margin at the status quo everywhere."""
        w = self.scale()[channel]
        if level <= 0:
            flat = level * w.sum()
            return lambda p: np.full(np.shape(p), flat)
        return lambda p: level * (self.shapes(channel, p) @ w)

    def rise(self, spend: npt.ArrayLike, levels: npt.ArrayLike) -> _Array:
        """What each channel's margin curve gains from the status quo to the weekly spends
        `spend`, per unit of its level, `(channels,)`: the integral of `margin` over its level,
        each column's `(K + s) (p - s) / (K + p)`, `cosh(k s)^2 (tanh(k p) - tanh(k s)) / k` and
        `p - s`, weighed by `scale`; a line's where the channel's level in `levels` is not
        positive."""
        p = np.asarray(spend, dtype=float)
        level = np.asarray(levels, dtype=float)
        s = self.status_quo
        w = self.scale()
        out = np.empty(s.size)
        for c in range(s.size):
            move = p[c] - s[c]
            if level[c] <= 0:
                out[c] = w[c].sum() * move
                continue
            half, k = self.half[c], self.rate[c]
            each = np.r_[
                (half + s[c]) * move / (half + p[c]),
                np.cosh(k * s[c]) ** 2 * (np.tanh(k * p[c]) - np.tanh(k * s[c])) / k,
                move,
            ]
            out[c] = float(each @ w[c])
        return out

    def elasticity(self) -> _Array:
        """Each channel's posterior mean elasticity at the status quo, the line's 1."""
        at = np.r_[np.tile(ELASTICITIES, 2), 1.0]
        return (self.weights * at[None, :]).sum(axis=1)


def read_tests(
    channels: tuple[str, ...],
    status_quo: npt.ArrayLike,
    channel: tuple[str, ...],
    x: npt.ArrayLike,
    delta_y: npt.ArrayLike,
    sigma: npt.ArrayLike,
) -> ChannelTests:
    """Each of `channels`' tests read off the lift rows, one a test: the test's `channel`, its
    weekly spend `x` over the dark weeks, the gap `delta_y` and its standard error `sigma`, as the
    module states; `status_quo`, each channel's weekly spend, sets the slope's origin.

    Raises:
        ValueError: on rows of unequal lengths, a row of a channel not among `channels`, a channel
            with no row, a spend, a standard error or a status quo not positive and finite, or a
            gap not finite.
    """
    s = np.asarray(status_quo, dtype=float)
    spend = np.asarray(x, dtype=float)
    gap = np.asarray(delta_y, dtype=float)
    error = np.asarray(sigma, dtype=float)
    names = np.array(channel, dtype=str)
    if s.shape != (len(channels),) or not names.shape == spend.shape == gap.shape == error.shape:
        msg = (
            f"one status quo a channel and the lift rows of equal lengths, not {s.shape} for"
            f" {len(channels)} channels and {names.shape}, {spend.shape}, {gap.shape},"
            f" {error.shape}"
        )
        raise ValueError(msg)
    stray = sorted(set(channel) - set(channels))
    untested = [c for c in channels if c not in set(channel)]
    if stray or untested:
        msg = (
            f"every row's channel is one of {list(channels)} and each of them has a row, not"
            f" {stray or untested}"
        )
        raise ValueError(msg)
    if not (
        all(np.all(np.isfinite(a)) and np.all(a > 0.0) for a in (s, spend, error))
        and np.all(np.isfinite(gap))
    ):
        msg = "the spends, the standard errors and the status quo are positive and finite, the gaps"
        msg += " finite"
        raise ValueError(msg)
    rho = -gap / spend
    se = error / spend
    roi, roi_se, slope_t, count = [], [], [], []
    for c, name in enumerate(channels):
        sel = names == name
        weight = 1.0 / se[sel] ** 2
        roi.append(float(np.sum(weight * rho[sel]) / np.sum(weight)))
        roi_se.append(float(1.0 / np.sqrt(np.sum(weight))))
        slope_t.append(_slope_t(np.log(spend[sel] / float(s[c])), rho[sel], se[sel]))
        count.append(int(sel.sum()))
    return ChannelTests(np.array(roi), np.array(roi_se), np.array(slope_t), tuple(count))


def _slope_t(u: _Array, rho: _Array, se: _Array) -> float:
    """The weighted least-squares slope of `rho` on `u` over its standard error, the variances as
    `se` states them; 0 where `u` spans less than `FLAT`."""
    if np.ptp(u) < FLAT:
        return 0.0
    design = np.column_stack([np.ones(u.size), u])
    weights = np.diag(1.0 / se**2)
    cov = np.linalg.inv(design.T @ weights @ design)
    beta = cov @ design.T @ weights @ rho
    return float(beta[1] / np.sqrt(cov[1, 1]))


def anchor(
    mean: npt.ArrayLike, cov: npt.ArrayLike, roi: npt.ArrayLike, roi_se: npt.ArrayLike
) -> Anchored:
    """The regression's margins `mean`, `(channels,)`, of covariance `cov`, anchored by the tests'
    returns per euro `roi` of standard errors `roi_se`, the regression's bias `tau` read by
    moments; see the module.

    Raises:
        ValueError: on arrays of other shapes, a value that is not finite, or a test's standard
            error not positive.
    """
    b, v, r, se = _checked(mean, cov, roi, roi_se)
    tau2 = _bias(b, v, r, se)
    biased = v + tau2 * np.eye(b.size)
    tests = np.diag(1.0 / se**2)
    precision = np.linalg.inv(biased) + tests
    posterior = np.linalg.inv(precision)
    return Anchored(
        posterior @ (np.linalg.solve(biased, b) + tests @ r), posterior, math.sqrt(tau2)
    )


def _checked(
    mean: npt.ArrayLike, cov: npt.ArrayLike, roi: npt.ArrayLike, roi_se: npt.ArrayLike
) -> tuple[_Array, _Array, _Array, _Array]:
    """The margins, their covariance, the returns and their standard errors as arrays.

    Raises:
        ValueError: as `anchor` raises.
    """
    b = np.asarray(mean, dtype=float)
    v = np.asarray(cov, dtype=float)
    r = np.asarray(roi, dtype=float)
    se = np.asarray(roi_se, dtype=float)
    if b.ndim != 1 or v.shape != (b.size, b.size) or r.shape != b.shape or se.shape != b.shape:
        msg = (
            f"margins and returns are (channels,), the covariance (channels, channels), not"
            f" {b.shape}, {v.shape}, {r.shape} and {se.shape}"
        )
        raise ValueError(msg)
    if not all(np.all(np.isfinite(a)) for a in (b, v, r, se)) or not np.all(se > 0.0):
        msg = "the margins, their covariance and the returns are finite, the standard errors > 0"
        raise ValueError(msg)
    return b, v, r, se


def _bias(b: _Array, v: _Array, r: _Array, se: _Array) -> float:
    """The regression's bias `tau^2` read by moments (the module)."""
    d = b - r
    return max(0.0, float(np.mean(d**2 - np.diag(v) - se**2)))


def logistic_rate(elasticity: float) -> float:
    """`u = k s` of the logistic `tanh(k a)` whose elasticity at `s` is `elasticity`,
    `2 u / sinh(2 u) = elasticity`, by `RATE_BISECTIONS` bisections over `(0, RATE_MOST]`; 0 at
    an elasticity of 1 or above, the line."""
    if elasticity >= 1.0:
        return 0.0
    low, high = 1e-12, RATE_MOST
    for _ in range(RATE_BISECTIONS):
        mid = 0.5 * (low + high)
        if 2.0 * mid / math.sinh(2.0 * mid) > elasticity:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def read_curves(
    channels: tuple[str, ...],
    status_quo: npt.ArrayLike,
    channel: tuple[str, ...],
    x: npt.ArrayLike,
    delta_y: npt.ArrayLike,
    sigma: npt.ArrayLike,
    start: npt.ArrayLike,
    mean: npt.ArrayLike,
    cov: npt.ArrayLike,
    spend: npt.ArrayLike,
    slope: Callable[[int, _Array], float],
    *,
    retention: tuple[float, ...],
    kernel: int,
    dark: int,
    readout: int,
    line_prior: float = LINE_PRIOR,
) -> Curves:
    """Each channel's posterior over its curve columns, read off the lift rows as `read_tests`
    reads them, each test's first dark week `start` counted from 1, and off the regression's
    margins `mean` of covariance `cov`. `spend` is the history, `(weeks, channels)`; `slope(c, y)`
    the coefficient the regression reads on channel `c` off the series `y`, `(weeks,)`, in the
    sales' place; `retention` each channel's retention and `kernel` the kernel's length; a test
    went dark `dark` weeks and read `readout` weeks from its first. See the module.

    Raises:
        ValueError: as `read_tests` and `anchor` raise, on a history or retentions of another
            shape, a start that is not a whole week whose dark weeks lie in the history, or a
            line prior outside `[0, 1]`.
    """
    tests = read_tests(channels, status_quo, channel, x, delta_y, sigma)
    b, v, *_ = _checked(mean, cov, tests.roi, tests.roi_se)
    tau2 = _bias(b, v, tests.roi, tests.roi_se)
    history = np.asarray(spend, dtype=float)
    first = np.asarray(start, dtype=float) - 1.0
    if history.ndim != 2 or history.shape[1] != len(channels) or len(retention) != len(channels):
        msg = (
            f"the history is (weeks, channels) and each channel has a retention, not"
            f" {history.shape} and {len(retention)} for {len(channels)} channels"
        )
        raise ValueError(msg)
    weeks = history.shape[0]
    if (
        first.shape != np.shape(x)
        or not np.all(first == np.round(first))
        or not np.all(first >= 0.0)
        or not np.all(first + dark <= weeks)
    ):
        msg = f"each test's {dark} dark weeks lie in the history's {weeks}, not from {list(first)}"
        raise ValueError(msg)
    if not 0.0 <= line_prior <= 1.0:
        msg = f"the line's prior is a probability, not {line_prior}"
        raise ValueError(msg)
    s = np.asarray(status_quo, dtype=float)
    names = np.array(channel, dtype=str)
    rho = -np.asarray(delta_y, dtype=float) / np.asarray(x, dtype=float)
    se = np.asarray(sigma, dtype=float) / np.asarray(x, dtype=float)
    count = ELASTICITIES.size
    half = np.empty((s.size, count))
    rate = np.empty((s.size, count))
    logs, ratios = [], []
    for c, name in enumerate(channels):
        sel = names == name
        carried = _carried(history[:, c], retention[c], kernel)
        designs = []
        for f in first[sel].astype(int):
            darkened = history[:, c].copy()
            darkened[f : f + dark] = 0.0
            window = slice(f, f + readout)
            a = carried[window]
            dimmed = _carried(darkened, retention[c], kernel)[window]
            designs.append((a, a - dimmed, float(history[f : f + dark, c].mean())))
        line_b = slope(c, carried)
        line_r = np.array([d.sum() / (dark * xbar) for _, d, xbar in designs])
        mm_b, lg_b = np.ones(count), np.ones(count)
        mm_r, lg_r = np.ones((len(designs), count)), np.ones((len(designs), count))
        for i, e in enumerate(ELASTICITIES):
            k_half = s[c] * e / (1.0 - e)
            unit = k_half / (k_half + s[c]) ** 2
            mm_b[i] = slope(c, carried / (k_half + carried)) / unit
            lifts = [np.sum(a / (k_half + a) - (a - d) / (k_half + a - d)) for a, d, _ in designs]
            mm_r[:, i] = _returns(lifts, designs, dark) / unit
            k = logistic_rate(float(e)) / s[c]
            unit = k / np.cosh(k * s[c]) ** 2
            lg_b[i] = slope(c, np.tanh(k * carried)) / unit
            lifts = [np.sum(np.tanh(k * a) - np.tanh(k * (a - d))) for a, d, _ in designs]
            lg_r[:, i] = _returns(lifts, designs, dark) / unit
            half[c, i], rate[c, i] = k_half, k
        h = np.hstack(
            [
                np.vstack([(mm_b / line_b)[None, :], mm_r / line_r[:, None]]),
                np.vstack([(lg_b / line_b)[None, :], lg_r / line_r[:, None]]),
                np.ones((1 + len(designs), 1)),
            ]
        )
        y = np.r_[b[c], rho[sel]]
        w = np.r_[1.0 / (v[c, c] + tau2), 1.0 / se[sel] ** 2]
        info = (w[:, None] * h * h).sum(axis=0)
        m = (w[:, None] * h * y[:, None]).sum(axis=0) / info
        rss = (w[:, None] * (y[:, None] - h * m[None, :]) ** 2).sum(axis=0)
        logs.append(-0.5 * rss - 0.5 * np.log(info))
        ratios.append(m / m[-1] if m[-1] != 0 else np.ones_like(m))
    prior = np.r_[np.full(2 * count, (1.0 - line_prior) / (2 * count)), line_prior]
    with np.errstate(divide="ignore"):
        lp = np.array(logs) + np.log(prior)[None, :]
    post = np.exp(lp - lp.max(axis=1, keepdims=True))
    post /= post.sum(axis=1, keepdims=True)
    return Curves(s, half, rate, post, np.array(ratios))


def _carried(spend: _Array, retention: float, length: int) -> _Array:
    """`spend` through the geometric kernel of `length` weeks, its weights summing to 1, the
    history before its first week read as no spend."""
    w = retention ** np.arange(length)
    return np.convolve(spend, w / w.sum())[: spend.size]


def _returns(
    lifts: list[np.floating], designs: list[tuple[_Array, _Array, float]], dark: int
) -> _Array:
    """Each test's lift over its dark weeks' spend."""
    return np.array(
        [lift / (dark * xbar) for lift, (_, _, xbar) in zip(lifts, designs, strict=True)]
    )
