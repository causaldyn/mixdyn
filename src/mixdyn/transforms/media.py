"""A channel's spend carried over, then saturated, in Robyn's convention (Runge et al. 2024), on
`chc.response`'s kernels and curve.

    Geometric(theta)          a_t = s_t + theta a_{t-1}, with nothing spent before the series;
    WeibullCdf(shape, scale)  a_t = sum_{l < T} w_l s_{t-l}, w_0 = 1, w_l = prod_{j<=l} (1 - F(j)),
                              F the Weibull CDF of `shape` and `scale` periods;
    Hill(alpha, gamma)        x^alpha / (x^alpha + K^alpha), with K = gamma max x over the
                              adstocked values of the window a model is fitted on (Robyn 3.12;
                              earlier releases put K gamma of the way from their least to their
                              most).

`channel` hands a fit to the decisions as `chc.response`'s `GeometricAdstock` or `WeibullAdstock`,
unnormalised and as long as the series, and its `Hill` at scale `K`. `carry_over` and `saturate`
are the same transforms in NumPy, for a fit's inner loop.

Robyn reads a Weibull scale as a share of the series' length `T`: `round(quantile(1:T, scale))`
periods, R's type-7 quantile rounded half to even, and a shape or a scale of 0 as no carryover.
`robyn_weibull` maps it onto these. A scale kept in periods leaves every earlier value where it was
when the series grows; a share moves them all. Robyn's Weibull PDF kernel, scaled to `[0, 1]`, is
not offered: the scaling sets its smallest weight to 0, and which weight that is depends on `T`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from chc import response
from scipy import signal

type _Array = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Geometric:
    """Each period keeps `theta` of the carryover it received, `0 <= theta < 1`."""

    theta: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.theta < 1.0:
            msg = f"theta must be in [0, 1), got {self.theta!r}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class WeibullCdf:
    """The carryover that survives each lag, a Weibull survival of `shape` and `scale` periods."""

    shape: float
    scale: float

    def __post_init__(self) -> None:
        if not (0.0 < self.shape < math.inf and 0.0 < self.scale < math.inf):
            msg = f"shape and scale must be positive and finite, got {self.shape!r}, {self.scale!r}"
            raise ValueError(msg)


type Adstock = Geometric | WeibullCdf
"""A carryover kernel: geometric (`Geometric`) or a Weibull survival (`WeibullCdf`)."""


@dataclass(frozen=True, slots=True)
class Hill:
    """A Hill curve of slope `alpha > 0`, its half-saturation point `gamma` times the most of the
    window's adstocked values, `0 < gamma <= 1`."""

    alpha: float
    gamma: float

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < math.inf and 0.0 < self.gamma <= 1.0):
            msg = f"alpha must be positive and gamma in (0, 1], got {self.alpha!r}, {self.gamma!r}"
            raise ValueError(msg)


def robyn_weibull(shape: float, scale: float, length: int) -> Adstock:
    """Robyn's CDF Weibull adstock of `shape >= 0` and `scale` in `[0, 1]`, a share of a series of
    `length` periods."""
    if not (0.0 <= shape < math.inf and 0.0 <= scale <= 1.0 and length >= 1):
        msg = f"need shape >= 0, scale in [0, 1] and length >= 1, got {shape}, {scale}, {length}"
        raise ValueError(msg)
    if shape == 0.0 or scale == 0.0:
        return Geometric(0.0)
    index = 1.0 + (length - 1) * scale
    low = math.floor(index)
    share = index - low
    # R interpolates between the two neighbours rather than adding the share
    periods = (1.0 - share) * low + share * (low + 1) if share > 0.0 else float(low)
    return WeibullCdf(shape, float(round(periods)))


def _spend(spend: npt.ArrayLike) -> _Array:
    series = np.asarray(spend, dtype=np.float64)
    if series.ndim != 1 or series.size == 0 or not (np.isfinite(series) & (series >= 0.0)).all():
        msg = f"spend must be a nonempty (T,) series, finite and nonnegative, not {series.shape}"
        raise ValueError(msg)
    return series


def carry_over(spend: npt.ArrayLike, adstock: Adstock) -> _Array:
    """The adstock of a `(T,)` spend series, its kernel as long as the series."""
    series = _spend(spend)
    match adstock:
        case Geometric(theta):
            return signal.lfilter([1.0], [1.0, -theta], series)
        case WeibullCdf(shape, scale):
            lags = np.arange(1, series.size, dtype=np.float64)
            hazard = np.cumsum((lags / scale) ** shape)
            weights = np.exp(-np.concatenate([np.zeros(1), hazard]))
            return np.convolve(series, weights)[: series.size]


def half_saturation(hill: Hill, window: npt.ArrayLike) -> float:
    """`K`, `gamma` times the most of a window's adstocked values.

    Raises:
        ValueError: when the window is dark throughout, so `K` is 0.
    """
    most = float(_spend(window).max())
    if most == 0.0:
        msg = "the window's adstocked spend is 0 throughout, so its half-saturation point is 0"
        raise ValueError(msg)
    return most * hill.gamma


def saturate(adstocked: npt.ArrayLike, hill: Hill, half: float) -> _Array:
    """The Hill curve of slope `hill.alpha` and half-saturation point `half` at adstocked values."""
    rising = (_spend(adstocked) / half) ** hill.alpha
    return rising / (1.0 + rising)


def channel(
    adstock: Adstock, hill: Hill, half: float, coefficient: float, length: int
) -> response.Channel:
    """The channel `coefficient * saturate(carry_over(spend))` in `chc.response`'s terms, its kernel
    `length` periods long."""
    match adstock:
        case Geometric(theta):
            kernel = response.GeometricAdstock(theta, length=length, normalized=False)
        case WeibullCdf(shape, scale):
            kernel = response.WeibullAdstock(shape, scale, length=length, normalized=False)
    return response.Channel(kernel, response.Hill(half, hill.alpha), coefficient)
