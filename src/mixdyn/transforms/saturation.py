"""A channel's spend carried over by a normalised geometric kernel and saturated by a curve whose
half-saturation point is its scale, on `chc.response`: the Bayesian model's transform
(`mixdyn.models.bayes_mmm`), held as a value a dynamic model's effect can carry.

    a_t = sum_{l < L} w_l s_{t-l},  w_l = r^l / sum_{j < L} r^j,  nothing spent before the series;
    tanh:  g(a) = tanh(atanh(1/2) a / K),  PyMC-Marketing's logistic saturation;
    Hill:  g(a) = a^n / (a^n + K^n).

`K` is the adstock at half the ceiling in both shapes, in the spend's currency, as the Bayesian
model reads it. The column a model reads is `g(a)` computed by `chc.response`'s kernel and curve,
and the channel a planner reads is `chc.response.Channel` of the same kernel and curve, so a
coefficient fitted on the column is the channel's.

A transform holds numbers, not `chc.response`'s modules, so it compares and hashes as a value; the
modules are built when they are asked for, and only in double precision: in single, JAX's default,
a curve keeps its rounded parameters and every column moves by a part in 1e7.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import numpy as np
import numpy.typing as npt
from chc import response

type _Array = npt.NDArray[np.float64]

_TANH_HALF = 0.5493061443340549
"""Where `tanh` reaches half its ceiling, `atanh(1/2) = ln(3)/2` correctly rounded. A literal
rather than `math.atanh(0.5)`, whose rounding is the platform libm's: glibc 2.39's is one ulp
low, and every number of a tanh curve moved with it there."""


@dataclass(frozen=True, slots=True)
class TanhShape:
    """`tanh(atanh(1/2) a / K)`: concave from no spend, half its ceiling at `K`."""


@dataclass(frozen=True, slots=True)
class HillShape:
    """`a^n / (a^n + K^n)`: half its ceiling at `K`, S-shaped where its `slope` `n` passes 1.

    Raises:
        ValueError: on a slope that is not positive and finite.
    """

    slope: float

    def __post_init__(self) -> None:
        if not 0.0 < self.slope < math.inf:
            msg = f"a Hill's slope must be positive and finite, got {self.slope!r}"
            raise ValueError(msg)


type Shape = TanhShape | HillShape


@dataclass(frozen=True, slots=True)
class Transform:
    """A channel's spend carried over by a normalised geometric kernel of `retention` and `length`
    periods, then saturated by a curve of `shape` that is half its ceiling where the adstock is
    `half_saturation`, in the spend's currency.

    Raises:
        ValueError: on a retention outside `[0, 1]`, a half-saturation point that is not positive
            and finite, a length that is not a whole number of periods, at least 1, or a shape
            that is neither.
    """

    retention: float
    half_saturation: float
    shape: Shape
    length: int

    def __post_init__(self) -> None:
        problems = []
        if not 0.0 <= self.retention <= 1.0:
            problems.append(f"retention must be in [0, 1], got {self.retention!r}")
        if not 0.0 < self.half_saturation < math.inf:
            half = self.half_saturation
            problems.append(f"the half-saturation point must be positive and finite, got {half!r}")
        if isinstance(self.length, bool) or not isinstance(self.length, int) or self.length < 1:
            problems.append(
                f"length must be a whole number of periods, at least 1, got {self.length!r}"
            )
        if not isinstance(self.shape, TanhShape | HillShape):
            problems.append(f"shape must be TanhShape() or HillShape(slope), got {self.shape!r}")
        if problems:
            raise ValueError("; ".join(problems))

    def kernel(self) -> response.GeometricAdstock:
        """The carryover as `chc.response` reads it.

        Raises:
            ValueError: on JAX in single precision.
        """
        _double()
        return response.GeometricAdstock(self.retention, length=self.length, normalized=True)

    def curve(self) -> response.Tanh | response.Hill:
        """The saturation as `chc.response` reads it, its scale the half-saturation point's.

        Raises:
            ValueError: on JAX in single precision.
        """
        _double()
        match self.shape:
            case TanhShape():
                return response.Tanh(self.half_saturation / _TANH_HALF)
            case HillShape(slope):
                return response.Hill(self.half_saturation, slope)

    def channel(self, coefficient: float) -> response.Channel:
        """The channel `coefficient * curve(kernel(spend))`, as `chc.allocation` plans on it.

        Raises:
            ValueError: on JAX in single precision.
        """
        return response.Channel(self.kernel(), self.curve(), coefficient)

    def column(self, spend: npt.ArrayLike) -> _Array:
        """The saturated adstock of a `(T,)` spend series, nothing spent before it: the column a
        model's coefficient multiplies.

        Raises:
            ValueError: on a spend series that is not a nonempty vector of finite values at or
                above 0, or JAX in single precision.
        """
        series = np.asarray(spend, dtype=np.float64)
        if (
            series.ndim != 1
            or series.size == 0
            or not (np.isfinite(series) & (series >= 0.0)).all()
        ):
            msg = (
                f"spend must be a nonempty (T,) series, finite and nonnegative, not {series.shape}"
            )
            raise ValueError(msg)
        return np.asarray(self.curve()(self.kernel()(series)), dtype=np.float64)


def _double() -> None:
    if not jax.config.x64_enabled:
        msg = (
            "a transform needs JAX in double precision (JAX_ENABLE_X64=1): in single, its curve "
            "keeps rounded parameters and its column moves by a part in 1e7"
        )
        raise ValueError(msg)
