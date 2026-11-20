"""A geo test read: from a panel and the test's design, the lift the test moved and its standard
error, as a `mixdyn.calibration.LiftEstimate` a model reads, with the reading behind it.

A `GeoTest` names the test: the treated geos and the control geos, the channel, the first and the
last period the test changed the spend in, as dates, the treated group's spend on the channel as
the test ran it and as planned, and the readout's length, counted from the test's first period, so
that a cooldown after the test is read with it. The periods before the test's first are the
pre-period. Nothing is inferred from the data: a test read from a date the data suggest reads a
lift that is not the test's.

Three readers turn the design into a lift over the readout. Each sums the treated geos' KPI and
reads it against the control geos':

- `read_did`, a difference in differences: the gap `g_t = T_t - b C_t` between the treated sum `T`
  and the control sum `C`, `b` the ratio of their pre-period means, so that the gap is 0 on average
  before the test whatever the groups' sizes; the lift is the readout's gaps less the mean
  pre-period gap, summed.
- `read_tbr`, Kerman, Wang and Vaver's (2017) time-based regression: `T` on `C` by ordinary least
  squares over the pre-period, the counterfactual the fit predicts over the readout, and the lift
  the readout's `T` less it, summed.
- `read_synthetic_control`, a synthetic control through `chc.scm.blank_period_inference`: the
  treated geos' mean as one unit, each control geo a donor, every unit less its mean over the
  periods the weights are fitted on (Ferman and Pinto's 2021 demeaned synthetic control, so that a
  convex combination of geos of other levels can reach it); the lift is the treated geos' count
  times the gaps summed over the readout.

The standard errors read the noise's dependence over time rather than assume it away: a geo's KPI
noise carries over from week to week, and an error that treats the periods as independent is far
too small: over 10 000 panels of 10 geos and 52 weeks, 5 of them treated over the last 4, their
noise a first-order autoregression of 0.8, a 5 % test under Kerman's independent errors rejected a
true null 0.32 of the time, and each reader here 0.043 to 0.046 (ADR 0047).

- The difference in differences and the time-based regression read it from placebo windows in time
  (cf. `PlaceboNoise`). Every window of the readout's length in the pre-period with at least
  `warmup` periods before it is read as the test is, with the periods before it as its pre-period:
  the ratio or the regression refitted on them, the window's gaps summed; a pre-period with room
  for fewer than two windows is refused. Each placebo lift mimics the estimator, its parameters'
  error and the noise's dependence over the window included. The standard error is their root mean
  square, and the reading a Student t's at the windows' effective count, the count of independent
  normal squares whose mean is as noisy as theirs (`PlaceboNoise` reads the same count).
- `read_tbr(..., variance="iid")` is Kerman, Wang and Vaver's own predictive variance instead:
  `s^2 (R + R^2 / n + (X_R - R xbar)^2 / S_xx)`, `s^2` the fit's residual variance over `n - 2`
  degrees of freedom, `X_R` the control sum over the `R` readout periods, `xbar` and `S_xx` the
  control sum's pre-period mean and sum of squared deviations, read as a Student t at `n - 2`.
  It holds where the errors are independent.
- The synthetic control's gaps are read as a first-order autoregression, at the coefficient
  `blank_period_inference(serial="ar1")` estimates from the donors, `rho`. The readout's gaps are
  measured from the fitting periods' mean gap, 0 by construction, so their sum is a contrast `c`
  of the gaps' path, `+1` on the readout and `-R / n_fit` on the `n_fit` fitting periods, and its
  variance is `-sum_{i, j} c_i c_j gamma(|i - j|)`, `gamma(d) = sigma^2 (1 + rho + ... + rho^(d -
  1)) / (1 + rho)` the autoregression's semivariogram, which holds at `rho = 1` too. `sigma^2` is
  the mean square of the innovations `u_t - rho u_{t-1}` over the blank periods, which the weights
  never saw, and the reading a Student t at their count. `blank_period_inference`'s per-period
  intervals are not one for the sum; its permutation p-value is kept beside the reading, where
  its arrangements are few enough to enumerate (at 8 blank periods, a readout of 11 periods at
  most): none of the rest needs them.

Two readings, opt-in, read a test designed within matched pairs (`mixdyn.experiments.assign` over
`pair_geos`' pairs) by its own randomisation instead, the pairs named by `pairs=`, each a treated
geo and its control (ADR 0058):

- `read_did(..., variance="unit_placebo")` and `read_tbr(..., variance="unit_placebo")`: design-
  aware unit placebos. Every assignment the design could have drawn, each pair's geos swapped or
  not, is read as the test is, the observed one among them, enumerated where they are at most
  100 000 and otherwise the observed one and 10 000 drawn from `SeedSequence(seed).spawn(1)[0]`.
  Each assignment's lift is studentised by its pre-period out-of-sample error: the estimator
  fitted on the pre-period less its last readout's length of periods, the root mean square of its
  gaps over those held-out periods. The p-value is the share of assignments whose studentised
  lift is at least as far from 0 as the observed one, and the interval the lift plus or minus the
  observed error times the `ceil(alpha A)`-th largest of the `A` studentised lifts in absolute
  value, so that it holds 0 exactly where the p-value is at least `alpha`. With no lift the test
  is exact whatever the noise: the KPI is held fixed and only the assignment is random. Under a
  lift the assignments that swap a pair read part of it, so the interval is wider than the lift's
  own spread.
- `read_pairs`: the pairs' own difference in differences, read by sign-flip randomisation
  inference. Each pair's gap, its treated geo's KPI less its control's, less its mean over the
  pre-period, summed over the readout, is the pair's contribution `c_p`, and the lift their sum.
  Swapping a pair's geos flips its contribution's sign exactly, so the lift's distribution over
  the design's assignments is that of `sum_p s_p c_p` over the signs `s`, enumerated or drawn as
  above. The interval inverts the same test under a lift `L` shared alike by the pairs, `c_p - L /
  P`: each assignment is as far from 0 as the observed one on one interval of `L` about the
  estimate, so its ends are exact. The signs and their reverse read alike, so at most `2 / 2^P`
  of the assignments can reject: fewer than 6 pairs are refused, as no test at 5 % is possible.

Each reader returns the `LiftEstimate` and a `GeoReading`: the lift, the standard error and its
degrees of freedom, a two-sided interval at `alpha`, the p-value of the same t, and a verdict keyed
on the interval's sign: "up" when it lies above 0, "down" when below, "inconclusive" when it holds
0. One level serves the interval, the p-value and the verdict, so `p_value < alpha` exactly where
the verdict is not "inconclusive", and a significant fall reads "down". The estimate keeps a lift
at or below 0, as `LiftEstimate` does, and its standard error is widened, as `PlaceboNoise`'s is,
so that its normal 95 % interval is the reading's t interval: `se t_df(0.975) / z(0.975)`.
A randomisation reading's error is the spread of its lifts over the assignments, read at an
infinite count, and the estimate's the farther of its 95 % interval's ends from the lift over
`z(0.975)`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
from chc.scm import blank_period_inference
from scipy import stats

from mixdyn.calibration import LiftEstimate
from mixdyn.decisions.posterior import _whole
from mixdyn.experiments.candidates import _UPPER, _effective_count

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from mixdyn.data import Channel, Geo, MediaPanel

type _Array = npt.NDArray[np.float64]
type Method = Literal["did", "tbr", "synthetic_control", "pairs"]
type Verdict = Literal["up", "down", "inconclusive"]
type _Gaps = Callable[[_Array, _Array, int, int], _Array]

_ENUMERATED = 100_000  # the most assignments a randomisation reading enumerates
_DRAWN = 10_000  # the assignments it draws beside the observed one past that


@dataclass(frozen=True, slots=True, eq=False)
class GeoTest:
    """A geo test of `channel` as it ran: in the `treated` geos against the `control` geos, the
    spend changed from the period dated `start` to the one dated `end`, both included, and read
    over the `readout` periods from `start`. `spend` and `planned` are the treated group's spend
    on the channel as the test ran it and as planned, over the periods before the readout and the
    readout, as `LiftEstimate` holds them.

    Raises:
        ValueError: on no treated or no control geos, a geo named twice or in both groups, dates
            that are not dates or an end before the start, and what `LiftEstimate` refuses of the
            spend and the readout.
    """

    channel: Channel
    treated: tuple[Geo, ...]
    control: tuple[Geo, ...]
    start: np.datetime64
    end: np.datetime64
    spend: _Array
    planned: _Array
    readout: int

    def __post_init__(self) -> None:
        treated, control = tuple(self.treated), tuple(self.control)
        if not treated or not control:
            msg = "a geo test needs at least one treated and one control geo"
            raise ValueError(msg)
        if len(set(treated + control)) != len(treated) + len(control):
            msg = f"a geo is named twice in treated {list(treated)} and control {list(control)}"
            raise ValueError(msg)
        for name in ("start", "end"):
            date = getattr(self, name)
            if not isinstance(date, np.datetime64) or np.isnat(date):
                msg = f"the test's {name} must be a date, a numpy datetime64, got {date!r}"
                raise ValueError(msg)
        if self.end < self.start:
            msg = f"the test ends at {self.end} before it starts at {self.start}"
            raise ValueError(msg)
        checked = LiftEstimate(self.spend, self.planned, self.readout, 0.0, 1.0)
        object.__setattr__(self, "treated", treated)
        object.__setattr__(self, "control", control)
        object.__setattr__(self, "spend", checked.spend)
        object.__setattr__(self, "planned", checked.planned)


@dataclass(frozen=True, slots=True)
class GeoReading:
    """A geo test's reading by `method`: the `lift` over the readout and its standard error `se`,
    read as a Student t of `df` degrees of freedom; the two-sided interval `[lower, upper]` at
    level `alpha`; the `p_value` of the same t, two-sided, against no lift; and the `verdict`,
    "up" when the interval lies above 0, "down" when below and "inconclusive" when it holds 0.
    `permutation_p` is the synthetic control's permutation p-value (`chc.scm.BlankPeriodInference.
    p_value`), `None` for the other readers and where its arrangements of the blank and readout
    periods are too many to enumerate.
    """

    method: Method
    lift: float
    se: float
    df: float
    alpha: float
    lower: float
    upper: float
    p_value: float
    verdict: Verdict
    permutation_p: float | None = None


def read_did(
    panel: MediaPanel,
    design: GeoTest,
    *,
    alpha: float = 0.05,
    warmup: int = 13,
    variance: Literal["placebo", "unit_placebo"] = "placebo",
    pairs: Sequence[tuple[Geo, Geo]] | None = None,
    seed: int = 0,
) -> tuple[LiftEstimate, GeoReading]:
    """`design`'s test on `panel` read by a difference in differences, its standard error from
    placebo windows with at least `warmup` periods before each, or, at `variance="unit_placebo"`,
    read by unit placebos over the design's matched `pairs`, drawn by `seed` where they are too
    many to enumerate; see the module.

    Raises:
        ValueError: on what `_series` refuses; `alpha` outside `(0, 1)`; `warmup` not a whole
            number, at least 1; `variance` neither reading; at "placebo", fewer than `warmup`
            plus the readout plus 1 periods before the test, which leave fewer than two placebo
            windows, a control sum not above 0 over the periods before some window, and placebo
            windows that all read 0; and at "unit_placebo", what `_unit_placebo` refuses, a
            control sum not above 0 over the periods some assignment is fitted on among it.
    """
    _level(alpha)
    _whole(warmup, "warmup", 1)
    if variance == "unit_placebo":
        return _unit_placebo("did", panel, design, pairs, seed, alpha, 2, _did_gaps)
    if variance != "placebo":
        msg = f"variance must be 'placebo' or 'unit_placebo', got {variance!r}"
        raise ValueError(msg)
    treated, control, n, r = _series(panel, design, warmup + 1)
    _windows(n, r, warmup)
    if not (np.cumsum(control[:n])[warmup - 1 :] > 0.0).all():
        msg = "the control geos' KPI must sum above 0 over the periods before every window"
        raise ValueError(msg)

    def lift(first: int) -> float:
        ratio = treated[:first].mean() / control[:first].mean()
        gaps = treated[: first + r] - ratio * control[: first + r]
        return float(np.sum(gaps[first:] - gaps[:first].mean()))

    se, df = _placebo([lift(f) for f in range(warmup, n - r + 1)], r)
    return _read("did", design, panel, n, lift(n), se, df, alpha)


def read_tbr(
    panel: MediaPanel,
    design: GeoTest,
    *,
    alpha: float = 0.05,
    warmup: int = 13,
    variance: Literal["placebo", "iid", "unit_placebo"] = "placebo",
    pairs: Sequence[tuple[Geo, Geo]] | None = None,
    seed: int = 0,
) -> tuple[LiftEstimate, GeoReading]:
    """`design`'s test on `panel` read by Kerman, Wang and Vaver's (2017) time-based regression,
    its standard error from placebo windows with at least `warmup` periods before each, at
    `variance="iid"` their predictive variance under independent errors, or, at
    `variance="unit_placebo"`, read by unit placebos over the design's matched `pairs`, drawn by
    `seed` where they are too many to enumerate; see the module.

    Raises:
        ValueError: on what `_series` refuses; `alpha` outside `(0, 1)`; `warmup` not a whole
            number, at least 3; `variance` none of the readings; at "placebo", fewer than `warmup`
            plus the readout plus 1 periods before the test, which leave fewer than two placebo
            windows; a control sum constant over the periods before some window, which fits no
            regression; placebo windows that all read 0; at "iid", a fit with no residual; and at
            "unit_placebo", what `_unit_placebo` refuses, a control sum constant over the periods
            some assignment is fitted on among it.
    """
    _level(alpha)
    _whole(warmup, "warmup", 3)
    if variance == "unit_placebo":
        return _unit_placebo("tbr", panel, design, pairs, seed, alpha, 3, _tbr_gaps)
    if variance not in ("placebo", "iid"):
        msg = f"variance must be 'placebo' or 'iid', or 'unit_placebo', got {variance!r}"
        raise ValueError(msg)
    treated, control, n, r = _series(panel, design, 3 if variance == "iid" else warmup + 1)
    if variance == "placebo":
        _windows(n, r, warmup)

    def fit(first: int) -> tuple[float, float, float, float]:
        """The slope, the control sum's deviations over the readout from its pre-period mean,
        summed, the sum of its squared deviations, and the lift, fitted on `first` periods."""
        x, y = control[:first], treated[:first]
        dx = x - x.mean()
        spread = float(dx @ dx)
        if not spread > 0.0:
            msg = f"the control geos' KPI sum is constant over {first} periods: no regression"
            raise ValueError(msg)
        slope = float(dx @ (y - y.mean())) / spread
        ahead = float(np.sum(control[first : first + r] - x.mean()))
        lift = float(np.sum(treated[first : first + r] - y.mean())) - slope * ahead
        return slope, ahead, spread, lift

    slope, ahead, spread, lift = fit(n)
    if variance == "iid":
        x, y = control[:n], treated[:n]
        residuals = y - y.mean() - slope * (x - x.mean())
        s2 = float(residuals @ residuals) / (n - 2)
        if not s2 > 0.0:
            msg = "the regression fits the pre-period exactly: it has no residual to read"
            raise ValueError(msg)
        se = math.sqrt(s2 * (r + r * r / n + ahead * ahead / spread))
        df = float(n - 2)
    else:
        se, df = _placebo([fit(f)[3] for f in range(warmup, n - r + 1)], r)
    return _read("tbr", design, panel, n, lift, se, df, alpha)


def read_synthetic_control(
    panel: MediaPanel, design: GeoTest, *, alpha: float = 0.05, blank: int = 8
) -> tuple[LiftEstimate, GeoReading]:
    """`design`'s test on `panel` read by a synthetic control of the treated geos' mean, its
    weights fitted on the pre-period less its last `blank` periods, its standard error from the
    gaps' innovations over those `blank` periods; see the module.

    Raises:
        ValueError: on what `_series` refuses; `alpha` outside `(0, 1)`; `blank` not a whole
            number, at least 2; fewer than 3 control geos; what `chc.scm.blank_period_inference`
            refuses but too many arrangements of the blank and readout periods to enumerate,
            where the reading keeps no permutation p-value; an autoregression's coefficient of
            -1, which reads no variance; and gaps whose innovations over the blank periods are
            all 0.
    """
    _level(alpha)
    _whole(blank, "blank", 2)
    if len(design.control) < 3:
        msg = f"a synthetic control needs at least 3 donors, got {len(design.control)}"
        raise ValueError(msg)
    _, _, n, r = _series(panel, design, blank + 2)
    treated = panel.kpi[_rows(panel, design.treated)].mean(axis=0)
    units = np.vstack([treated, panel.kpi[_rows(panel, design.control)]])
    fitted = n - blank
    units = units[:, : n + r] - units[:, :fitted].mean(axis=1, keepdims=True)
    # The weights, the donors' coefficient and the gaps depend on the fitting periods and the
    # periods through the readout alone, not on where the blank periods end: one post period
    # reads them all with as many arrangements as periods after the fit.
    inference = blank_period_inference(units, 0, n + r - 1, n + r - 1 - fitted, serial="ar1")
    rho = inference.ar_coefficient
    if rho is None or rho <= -1.0:
        msg = f"an autoregression of coefficient {rho} reads no variance of the gaps' sum"
        raise ValueError(msg)
    weights = inference.estimate.weights
    after = np.concatenate([inference.blank_gaps, inference.estimate.att])
    last = float(units[0, fitted - 1] - units[1:, fitted - 1] @ weights)
    path = np.concatenate([[last], after[:blank]])
    innovations = path[1:] - rho * path[:-1]
    sigma2 = float(innovations @ innovations) / blank
    contrast = np.zeros(n + r)
    contrast[:fitted] = -r / fitted
    contrast[n:] = 1.0
    lags = np.abs(np.subtract.outer(np.arange(n + r), np.arange(n + r)))
    semivariogram = np.concatenate([[0.0], np.cumsum(rho ** np.arange(n + r - 1))]) / (1.0 + rho)
    shape = -float(contrast @ semivariogram[lags] @ contrast)
    if not sigma2 * shape > 0.0:
        msg = "the gaps' innovations over the blank periods are all 0: nothing reads their noise"
        raise ValueError(msg)
    count = len(design.treated)
    lift = count * float(after[blank:].sum())
    se = count * math.sqrt(sigma2 * shape)
    try:
        permutation_p = blank_period_inference(
            units, 0, n, blank, alpha=alpha, serial="ar1"
        ).p_value
    except ValueError as error:
        if "arrangements of" not in str(error) or "exceed" not in str(error):
            raise
        permutation_p = None
    return _read(
        "synthetic_control", design, panel, n, lift, se, float(blank), alpha, permutation_p
    )


def read_pairs(
    panel: MediaPanel,
    design: GeoTest,
    pairs: Sequence[tuple[Geo, Geo]],
    *,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[LiftEstimate, GeoReading]:
    """`design`'s test on `panel` read by its matched `pairs`' own difference in differences and
    sign-flip randomisation inference, the signs drawn by `seed` where they are too many to
    enumerate; see the module.

    Raises:
        ValueError: on what `_series` refuses; `alpha` outside `(0, 1)`; pairs that are not each
            one treated and one control geo of the design, every one of them; a seed not a whole
            number at least 0; and assignments too few for the test to reject at `alpha` or at
            95 %, the estimate's, which leave its interval unbounded: fewer than 6 pairs.
    """
    _level(alpha)
    treated, control, n, r = _pair_series(panel, design, pairs, 1)
    gaps = treated - control
    contributions = gaps[:, n:].sum(axis=1) - r * gaps[:, :n].mean(axis=1)
    signs = np.where(_swaps(contributions.size, seed), -1.0, 1.0)
    lifts = signs @ contributions
    lift = float(lifts[0])
    p_value = float(np.mean(np.abs(lifts) >= abs(lift)))
    # Under a lift L shared alike, `c_p - L / P`, an assignment's lift less its share of L,
    # `a - b L` with `b` its mean sign, is as far from 0 as the observed `lift - L` while
    # `|a - b lift - b u| >= u`, `u = L - lift`: on `[0, |a'| / (1 + sign(a') b)]`, `a' = a - b
    # lift`, and for ever at `|b| = 1`, the observed signs and their reverse.
    shares = signs.mean(axis=1)
    centred = lifts - shares * lift
    whole = np.abs(shares) == 1.0
    ends = []
    for side in (1.0, -1.0):
        denominator = np.where(whole, 1.0, 1.0 + side * np.sign(centred) * shares)
        ends.append(np.where(whole, math.inf, np.abs(centred) / denominator))
    above, below = ends
    reaches = [(_largest(below, a), _largest(above, a)) for a in (alpha, 2.0 * (1.0 - _UPPER))]
    if not all(math.isfinite(u) for pair in reaches for u in pair):
        msg = (
            f"{lifts.size} assignments of {contributions.size} pairs are too few for the test to "
            f"reject at a level of {1.0 - alpha} or of 0.95, the estimate's: it needs 6 pairs"
        )
        raise ValueError(msg)
    (low, high), (low95, high95) = reaches
    widened = max(low95, high95) / float(stats.norm.ppf(_UPPER))
    randomised = (lift - low, lift + high, p_value, widened)
    return _read(
        "pairs", design, panel, n, lift, float(lifts.std()), math.inf, alpha, None, randomised
    )


def _unit_placebo(
    method: Method,
    panel: MediaPanel,
    design: GeoTest,
    pairs: Sequence[tuple[Geo, Geo]] | None,
    seed: int,
    alpha: float,
    fitted: int,
    gaps: _Gaps,
) -> tuple[LiftEstimate, GeoReading]:
    """`design`'s test read by `gaps`' estimator over every assignment of its matched `pairs`,
    each lift studentised by its pre-period out-of-sample error; see the module.

    Raises:
        ValueError: on what `_pair_series` refuses, the pairs missing among it; fewer than
            `fitted` periods before the held-out ones; a seed not a whole number at least 0; an
            assignment whose gaps over the held-out periods are all 0; and assignments too few
            for the test to reject at `alpha` or at 95 %, fewer than 20 or than `1 / alpha`.
    """
    if pairs is None:
        msg = "variance='unit_placebo' reads the design's matched pairs: name them by pairs="
        raise ValueError(msg)
    yt, yc, n, r = _pair_series(panel, design, pairs, design.readout + fitted)
    swaps = _swaps(yt.shape[0], seed)
    kept, swapped = (~swaps).astype(np.float64), swaps.astype(np.float64)
    treated, control = kept @ yt + swapped @ yc, swapped @ yt + kept @ yc
    lifts = gaps(treated, control, n, r).sum(axis=1)
    held = gaps(treated[:, :n], control[:, :n], n - r, r)
    errors = np.sqrt(np.mean(held * held, axis=1))
    if not (errors > 0.0).all():
        msg = f"an assignment's gaps over the {r} held-out periods are all 0: nothing reads them"
        raise ValueError(msg)
    studentised = np.abs(lifts / errors)
    count = studentised.size
    if min(alpha, 2.0 * (1.0 - _UPPER)) * count <= 1.0:
        msg = (
            f"{count} assignments of {yt.shape[0]} pairs are too few for the test to reject at a "
            f"level of {1.0 - alpha} or of 0.95, the estimate's"
        )
        raise ValueError(msg)
    lift, error = float(lifts[0]), float(errors[0])
    p_value = float(np.mean(studentised >= studentised[0]))
    half = _half(studentised, error, lift, alpha)
    widened = _half(studentised, error, lift, 2.0 * (1.0 - _UPPER)) / float(stats.norm.ppf(_UPPER))
    randomised = (lift - half, lift + half, p_value, widened)
    return _read(
        method, design, panel, n, lift, float(lifts.std()), math.inf, alpha, None, randomised
    )


def _did_gaps(treated: _Array, control: _Array, first: int, length: int) -> _Array:
    """Each row's difference-in-differences gaps over the `length` periods from `first`, its
    ratio fitted on the periods before."""
    if not (control[:, :first].sum(axis=1) > 0.0).all():
        msg = f"the control geos' KPI must sum above 0 over the {first} periods a ratio is fit on"
        raise ValueError(msg)
    ratio = treated[:, :first].mean(axis=1) / control[:, :first].mean(axis=1)
    gaps = treated[:, : first + length] - ratio[:, None] * control[:, : first + length]
    return gaps[:, first:] - gaps[:, :first].mean(axis=1, keepdims=True)


def _tbr_gaps(treated: _Array, control: _Array, first: int, length: int) -> _Array:
    """Each row's time-based regression gaps over the `length` periods from `first`, its
    regression fitted on the periods before."""
    x, y = control[:, :first], treated[:, :first]
    xbar, ybar = x.mean(axis=1, keepdims=True), y.mean(axis=1, keepdims=True)
    dx = x - xbar
    spread = np.sum(dx * dx, axis=1)
    if not (spread > 0.0).all():
        msg = f"the control geos' KPI sum is constant over {first} periods: no regression"
        raise ValueError(msg)
    slope = np.sum(dx * (y - ybar), axis=1, keepdims=True) / spread[:, None]
    ahead = slice(first, first + length)
    return treated[:, ahead] - ybar - slope * (control[:, ahead] - xbar)


def _half(studentised: _Array, error: float, lift: float, alpha: float) -> float:
    """The half-width of a studentised randomisation interval at `alpha`: `error` times the
    `ceil(alpha A)`-th largest of the `A` absolute studentised lifts, the observed one first. It
    holds 0 exactly where the test does not reject, even where the observed one is that value and
    the product rounds below `|lift|`."""
    critical = _largest(studentised, alpha)
    half = critical * error
    if studentised[0] <= critical:
        return max(half, abs(lift))
    return min(half, math.nextafter(abs(lift), 0.0))


def _largest(values: _Array, alpha: float) -> float:
    """The `ceil(alpha n)`-th largest of the `n` values: a test that rejects past it rejects
    where fewer than `alpha` of them are at least as large."""
    rank = math.ceil(alpha * values.size - 1e-9)
    return float(np.sort(values)[values.size - rank])


def _swaps(count: int, seed: int) -> npt.NDArray[np.bool_]:
    """The assignments a randomisation reading reads, a row each, true where a pair's geos are
    swapped: every one where they are at most `_ENUMERATED`, the observed one first, and otherwise
    the observed one and `_DRAWN` drawn from `SeedSequence(seed).spawn(1)[0]`."""
    _whole(seed, "seed", 0)
    if 2**count <= _ENUMERATED:
        return ((np.arange(2**count)[:, None] >> np.arange(count)) & 1).astype(np.bool_)
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    return np.vstack([np.zeros((1, count), dtype=np.bool_), rng.random((_DRAWN, count)) < 0.5])


def _pair_series(
    panel: MediaPanel, design: GeoTest, pairs: Sequence[tuple[Geo, Geo]], least: int
) -> tuple[_Array, _Array, int, int]:
    """Each pair's treated and control geo's KPI through the readout, a row a pair, the
    pre-period's length and the readout's.

    Raises:
        ValueError: on what `_series` refuses, and pairs that are not each one treated and one
            control geo of the design, every one of them.
    """
    _, _, n, r = _series(panel, design, least)
    named = [g for pair in pairs for g in pair]
    if any(len(pair) != 2 for pair in pairs) or sorted(named) != sorted(
        design.treated + design.control
    ):
        msg = f"pairs must hold every geo of the test once, two to a pair, got {list(pairs)}"
        raise ValueError(msg)
    treated = set(design.treated)
    if any((a in treated) == (b in treated) for a, b in pairs):
        msg = "each pair must hold one treated and one control geo"
        raise ValueError(msg)
    first = [a if a in treated else b for a, b in pairs]
    second = [b if a in treated else a for a, b in pairs]
    kpi = panel.kpi[:, : n + r]
    return kpi[_rows(panel, tuple(first))], kpi[_rows(panel, tuple(second))], n, r


def _level(alpha: float) -> None:
    if not (math.isfinite(alpha) and 0.0 < alpha < 1.0):
        msg = f"alpha={alpha!r} is not a level in (0, 1)"
        raise ValueError(msg)


def _rows(panel: MediaPanel, geos: tuple[Geo, ...]) -> list[int]:
    return [panel.geos.index(g) for g in geos]


def _series(panel: MediaPanel, design: GeoTest, least: int) -> tuple[_Array, _Array, int, int]:
    """The treated and the control geos' KPI summed each period, through the readout, the
    pre-period's length and the readout's.

    Raises:
        ValueError: on a geo or a channel the panel does not have; a start or an end not one of
            the panel's dates; a test that does not end within the readout or a readout that
            does not end within the panel; fewer than `least` periods before the test; and a KPI
            missing in a geo of the test through the readout.
    """
    missing = [g for g in design.treated + design.control if g not in panel.geos]
    if missing:
        msg = f"the panel has no geos {missing}: it has {list(panel.geos)}"
        raise ValueError(msg)
    if design.channel not in panel.channels:
        msg = f"the panel has no channel {design.channel!r}: it has {list(panel.channels)}"
        raise ValueError(msg)
    first, last = (np.flatnonzero(panel.dates == d) for d in (design.start, design.end))
    if first.size != 1 or last.size != 1:
        msg = f"the test's dates {design.start} and {design.end} must be periods of the panel"
        raise ValueError(msg)
    n, r = int(first[0]), design.readout
    if int(last[0]) >= n + r:
        msg = f"the test runs past its readout of {r} periods from {design.start}"
        raise ValueError(msg)
    if n + r > panel.dates.size:
        msg = f"a readout of {r} periods from {design.start} ends past the panel"
        raise ValueError(msg)
    if n < least:
        msg = f"the test starts {n} periods into the panel; this reading needs {least} before it"
        raise ValueError(msg)
    kpi = panel.kpi[:, : n + r]
    treated, control = kpi[_rows(panel, design.treated)], kpi[_rows(panel, design.control)]
    if not (np.isfinite(treated).all() and np.isfinite(control).all()):
        msg = "a geo of the test has no KPI in some period through the readout"
        raise ValueError(msg)
    return treated.sum(axis=0), control.sum(axis=0), n, r


def _windows(n: int, readout: int, warmup: int) -> None:
    """Refuses a pre-period of `n` periods with fewer than two placebo windows of `readout`
    periods, each with at least `warmup` periods before it."""
    if n < warmup + readout + 1:
        windows = max(n - warmup - readout + 1, 0)
        msg = (
            f"the test starts {n} periods into the panel, room for {windows} placebo window(s) "
            f"of {readout} periods after a warmup of {warmup}: two need {warmup + readout + 1}"
        )
        raise ValueError(msg)


def _placebo(lifts: list[float], readout: int) -> tuple[float, float]:
    """The placebo lifts' root mean square and their effective count."""
    t = np.array(lifts)
    square = float(t @ t)
    if not square > 0.0:
        msg = f"every placebo window of {readout} periods reads a lift of 0"
        raise ValueError(msg)
    return math.sqrt(square / t.size), _effective_count(t, square)


def _read(
    method: Method,
    design: GeoTest,
    panel: MediaPanel,
    n: int,
    lift: float,
    se: float,
    df: float,
    alpha: float,
    permutation_p: float | None = None,
    randomised: tuple[float, float, float, float] | None = None,
) -> tuple[LiftEstimate, GeoReading]:
    """The estimate and the reading of `lift` and its standard error `se`, a t of `df` degrees of
    freedom, for the test starting at period `n` of `panel`; or, a randomisation reading's, its
    `randomised` interval's ends, p-value and the estimate's widened error."""
    if randomised is None:
        t = stats.t(df)
        half = float(t.ppf(1.0 - alpha / 2.0)) * se
        lower, upper = lift - half, lift + half
        p_value = 2.0 * float(t.sf(abs(lift) / se))
        widened = se * float(t.ppf(_UPPER) / stats.norm.ppf(_UPPER))
    else:
        lower, upper, p_value, widened = randomised
    verdict: Verdict = "up" if lower > 0.0 else "down" if upper < 0.0 else "inconclusive"
    reading = GeoReading(method, lift, se, df, alpha, lower, upper, p_value, verdict, permutation_p)
    estimate = LiftEstimate(
        design.spend,
        design.planned,
        design.readout,
        lift,
        widened,
        geos=frozenset(design.treated),
        end=panel.dates[n + design.readout - 1],
    )
    return estimate, reading
