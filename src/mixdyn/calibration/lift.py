"""A lift test as its analysis reported it: the lift over the readout and the lift's standard error,
with the spend the test changed and the geos it ran in, so that a model reads it through the
carryover the change cost rather than as a lift at a steady spend.

A geo test's analysis, a synthetic control or a difference in differences, usually hands on one
number and its standard error, not the gap period by period that `chc.lift.LiftTest` holds. The
standard error enters the likelihood as it is, and a lift at or below 0 is kept: a test whose KPI
did not move says the channel's return over the spend it changed is small.

`LiftSeries` is the same test read period by period: the gap in each period of the readout, with
the covariance of the gaps' errors. One number over the readout reads one direction of a channel's
parameters: where the curve is straight over the test's spend and the readout holds the whole
carryover, the total is the same under every decay. The gaps' shape over the readout reads the
decay apart from the return.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from mixdyn.data import Geo

type _Array = npt.NDArray[np.float64]


def _spend(value: npt.ArrayLike, name: str) -> _Array:
    array = np.array(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        msg = f"{name} must be a non-empty vector, got shape {array.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        msg = f"{name} must be finite and at or above 0"
        raise ValueError(msg)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class LiftEstimate:
    """A lift test of one channel as its analysis reported it.

    `spend` is the tested group's spend on the channel as the test ran, over the periods before
    the readout and the readout, and `planned` what the group would have spent without the test,
    over the same periods; the last `readout` periods are the readout. `lift` is the group's KPI
    over the readout less what it would have been at `planned`, as the analysis estimated it, in
    the KPI's units, and `se` that estimate's standard error. `geos` are the geos the test ran in,
    and `None`, the default, the whole market the model reads: every geo of the model over geos,
    the one geo of a model of one. `share` is the group's share of those geos' market: the group's
    spend and lift over it are theirs.

    `end` is the date of the readout's last period in the panel's calendar, so the series' last
    period: a model whose coefficient moves over time (`mixdyn.models.fit_dlm`) reads the test at
    those periods and refuses an estimate without it. A model whose coefficient is one number over
    the window reads no calendar and passes it by.

    Raises:
        ValueError: on a spend not finite and at or above 0, `spend` and `planned` of different
            lengths or the same, a readout not within them, a lift not finite, a standard error not
            positive and finite, a share outside `(0, 1]`, no geo named, or an end that is not a
            date.
    """

    spend: _Array
    planned: _Array
    readout: int
    lift: float
    se: float
    share: float = 1.0
    geos: frozenset[Geo] | None = None
    end: np.datetime64 | None = None

    def __post_init__(self) -> None:
        spend, planned = _spend(self.spend, "spend"), _spend(self.planned, "planned")
        if spend.size != planned.size:
            msg = f"spend runs {spend.size} periods and planned {planned.size}"
            raise ValueError(msg)
        if np.array_equal(spend, planned):
            msg = "the test changed no spend, so it reads nothing of the channel"
            raise ValueError(msg)
        if not 1 <= self.readout <= spend.size:
            msg = f"a readout of {self.readout} periods is not within the {spend.size} of spend"
            raise ValueError(msg)
        if not math.isfinite(self.lift):
            msg = f"the lift must be finite, got {self.lift}"
            raise ValueError(msg)
        if not (math.isfinite(self.se) and self.se > 0.0):
            msg = f"the standard error must be positive and finite, got {self.se}"
            raise ValueError(msg)
        if not (math.isfinite(self.share) and 0.0 < self.share <= 1.0):
            msg = f"share={self.share!r} is not a share of the market in (0, 1]"
            raise ValueError(msg)
        if self.geos is not None and not self.geos:
            msg = "a test runs in at least one geo; geos=None reads the whole market"
            raise ValueError(msg)
        if self.end is not None and (not isinstance(self.end, np.datetime64) or np.isnat(self.end)):
            msg = f"the readout's end must be a date, a numpy datetime64, got {self.end!r}"
            raise ValueError(msg)
        object.__setattr__(self, "spend", spend)
        object.__setattr__(self, "planned", planned)


@dataclass(frozen=True, slots=True)
class LiftSeries:
    """A lift test of one channel read period by period: `LiftEstimate`'s test with its lift
    over each period of the readout rather than summed.

    `spend`, `planned`, `share` and `geos` are `LiftEstimate`'s. `gaps` is the group's KPI in each
    of the readout's periods less what it would have been at `planned`, so the readout is its last
    `gaps.size` periods, and `covariance` the covariance of those gaps' errors, in the KPI's units
    squared: a geo's noise that carries over from one period to the next correlates them
    (`mixdyn.experiments.PlaceboNoise.covariance` reads it from the history).

    `summed` is the test as one estimate: the gaps' sum with the error of that sum.

    Raises:
        ValueError: on what `LiftEstimate` refuses of the spend, the share and the geos; gaps not a
            non-empty vector of finite numbers or longer than the spend; and a covariance not of
            the gaps' size, not finite, not symmetric to `1e-12` of its largest entry, or not
            positive definite. A covariance within that is held as its symmetric part.
    """

    spend: _Array
    planned: _Array
    gaps: _Array
    covariance: _Array
    share: float = 1.0
    geos: frozenset[Geo] | None = None

    def __post_init__(self) -> None:
        gaps = np.array(self.gaps, dtype=np.float64)
        if gaps.ndim != 1 or gaps.size == 0 or not np.all(np.isfinite(gaps)):
            msg = f"gaps must be a non-empty vector of finite numbers, got shape {gaps.shape}"
            raise ValueError(msg)
        covariance = np.array(self.covariance, dtype=np.float64)
        if covariance.shape != (gaps.size, gaps.size) or not np.all(np.isfinite(covariance)):
            msg = f"the covariance must be finite and {gaps.size} square, got {covariance.shape}"
            raise ValueError(msg)
        if not np.abs(covariance - covariance.T).max() <= 1e-12 * np.abs(covariance).max():
            msg = "the covariance must be symmetric"
            raise ValueError(msg)
        covariance = (covariance + covariance.T) / 2.0
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError:
            msg = "the covariance must be positive definite: some combination of gaps has no error"
            raise ValueError(msg) from None
        checked = LiftEstimate(self.spend, self.planned, gaps.size, 0.0, 1.0, self.share, self.geos)
        gaps.setflags(write=False)
        covariance.setflags(write=False)
        object.__setattr__(self, "spend", checked.spend)
        object.__setattr__(self, "planned", checked.planned)
        object.__setattr__(self, "gaps", gaps)
        object.__setattr__(self, "covariance", covariance)

    @property
    def readout(self) -> int:
        """The periods the gaps cover, the spend's last."""
        return self.gaps.size

    @property
    def summed(self) -> LiftEstimate:
        """The test as one estimate: the lift summed over the readout, its standard error the
        sum's under the covariance."""
        return LiftEstimate(
            self.spend,
            self.planned,
            self.readout,
            float(self.gaps.sum()),
            math.sqrt(float(self.covariance.sum())),
            self.share,
            self.geos,
        )
