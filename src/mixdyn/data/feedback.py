"""Whether a channel's spend reads the KPI: the endogenous-spend flag (ADR 0074).

A business that sets a quarter's budget from the last quarters' sales, or a bidding rule that
raises paid search after a strong week, spends on what the KPI did. Where the KPI's unobserved part
persists, as demand does, last period's KPI carries this period's, and a model that reads the spend
as set apart from the KPI credits the channel with the demand. Heusch's world (2026, "A Synthetic
Benchmark Dataset with Endogenous Marketing Spend for Validating Marketing Mix Models",
arXiv:2608.21130) is built so: its budget reads the last two quarters' sales before media, its
bidding rule last week's over the ten before, and its baseline carries a sentiment nobody observes.

`spend_feedback` writes each channel's spend as the business's budget model and its surprise,
`spend_t = m_t + v_t`, with

    m_t = a + phi spend_{t-1} + gamma kpi_{t-1} + b t / n + sum_h (c_h cos + s_h sin)(2 pi h t / P)

over the geo's periods `t` observed with the one before, `n` the panel's periods and `P` a year at
its cadence, and reads `gamma`, the spend per unit of the KPI's last period, by least squares. The
spend's own last period holds a flighted or slowly planned spend's persistence, so the KPI's last
period, which the media's carryover ties to the spend's past, is read beyond it; the trend and the
season hold what the calendar sets for both. The standard error is Newey and West's (1987,
Econometrica 55(3)): Bartlett weights `1 - j / M` to `M - 1 = floor(4 (n_read / 100)^(2/9))` lags,
so the residuals' autocorrelation and unequal variance are counted, each residual scaled by
`1 / (1 - h_t)`, `h_t` its period's leverage, as HC3 scales it (MacKinnon and White 1985): a
promotion's week holds a KPI far from the rest, and the unscaled error read it too small. The
interval's half-width is the error times Kiefer and Vogelsang's (2005, Econometric Theory 21(6))
fixed-b critical value for the Bartlett kernel at `b = M / n_read`, their cubic in `b`. Over
1000 seeds of Track M v2's world with its four mechanisms off, Student's interval on the unscaled
error put 7.3 % to 8.0 % of a channel's 0.95 intervals off 0; this one puts 5.3 % to 5.7 %
(ADR 0074).

Where the interval excludes 0 the spend reads the KPI, and `SpendFeedback.issue` is an
`IssueCode.ENDOGENOUS_SPEND` warning. Its `gamma` may be of either sign: a spend raised ahead of a
promotion and held back in it reads a strong week's KPI with a fall. It is a flag, not a
correction: it says the spend reads the outcome, which is what lets a persistent unobserved demand
bias a return, and nothing of how far a return is biased; `mixdyn.inference.sensitivity` bounds
that. A spend that reads the demand of the same period, as paid search's auction volume does,
shows here only as far as demand persists; a demand proxy in the model is its remedy
(`mixdyn.models.demand`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from mixdyn.data.issues import Issue, IssueCode, Severity
from mixdyn.data.panel import Cadence

if TYPE_CHECKING:
    from mixdyn.data.panel import Geo, MediaPanel

_YEAR = {Cadence.DAY: 365.25, Cadence.WEEK: 365.25 / 7.0, Cadence.MONTH: 12.0}
"""A year's periods at each cadence."""
_FIXED_B = {
    0.8: (1.2816, 1.3040, 0.5135, -0.3386),
    0.9: (1.6449, 2.1859, 0.3142, -0.3427),
    0.95: (1.9600, 2.9694, 0.4160, -0.5324),
    0.98: (2.3263, 4.1618, 0.5368, -0.9620),
}
"""Kiefer and Vogelsang's (2005) cubic in `b` for the Bartlett kernel's fixed-b critical value of a
two-sided interval at each level. A simulation of the statistic at `b` from 0.02 to 0.2 matches
each within 2 %."""


@dataclass(frozen=True, slots=True)
class SpendFeedback:
    """How far `channel`'s spend in `geo` reads the KPI's last period.

    `gamma` is the spend per unit of the KPI's last period, the spend's own last period and the
    calendar held, in the panel's currency per unit of the KPI; `se` its Newey-West standard error
    and `lower` and `upper` its interval at `level`. `periods` counts the periods read, each
    observed with the one before it, and `lags` the Newey-West lags, `M - 1`.
    """

    geo: str
    channel: str
    gamma: float
    se: float
    lower: float
    upper: float
    level: float
    periods: int
    lags: int

    @property
    def follows(self) -> bool:
        """Whether the interval excludes 0: the spend reads the KPI."""
        return self.lower > 0.0 or self.upper < 0.0

    def issue(self) -> Issue | None:
        """The `ENDOGENOUS_SPEND` warning where the spend reads the KPI, None where not."""
        if not self.follows:
            return None
        message = (
            f"geo {self.geo!r}: {self.channel!r}'s spend reads the KPI's last period, "
            f"{self.gamma:.4g} a unit of it, its {self.level:.0%} interval [{self.lower:.4g}, "
            f"{self.upper:.4g}] beside its own last period and the calendar; a model that reads "
            "the spend as set apart from demand credits the channel with the demand"
        )
        return Issue(
            IssueCode.ENDOGENOUS_SPEND,
            Severity.WARNING,
            message,
            geo=self.geo,
            column=self.channel,
            channel=self.channel,
        )


def spend_feedback(
    panel: MediaPanel, geo: Geo, level: float = 0.95, harmonics: int = 3
) -> tuple[SpendFeedback, ...]:
    """Each of `geo`'s channels whose spend varies, read as the module states: `gamma` and its
    interval at `level`, 0.8, 0.9, 0.95 or 0.98, the levels whose fixed-b critical values are
    tabled, and the season `harmonics` harmonics of a year at the panel's cadence.

    Raises:
        ValueError: on a geo the panel does not have, a level not tabled, harmonics not from 1
            to below half a year's periods, or too few periods observed with the one before for
            the regression's columns and two degrees of freedom.
    """
    if geo not in panel.geos:
        msg = f"the panel has no geo {geo!r}; it has {list(panel.geos)}"
        raise ValueError(msg)
    if level not in _FIXED_B:
        msg = f"level must be one of {sorted(_FIXED_B)}, got {level}"
        raise ValueError(msg)
    year = _YEAR[panel.cadence]
    if not 1 <= harmonics < year / 2.0:
        msg = f"need harmonics from 1 to below half of {year:.6g} periods, got {harmonics}"
        raise ValueError(msg)
    g = panel.geos.index(geo)
    kpi, spend = panel.kpi[g], panel.spend[g]
    observed = ~panel.filled[g] & np.isfinite(kpi)
    rows = np.flatnonzero(observed[1:] & observed[:-1]) + 1
    periods = panel.dates.size
    t = np.arange(periods, dtype=np.float64)
    angles = [2.0 * math.pi * h * t / year for h in range(1, harmonics + 1)]
    calendar = np.column_stack(
        [np.ones(periods), t / periods, *(f(a) for a in angles for f in (np.cos, np.sin))]
    )
    width = calendar.shape[1] + 2
    if rows.size - width < 2:
        msg = (
            f"geo {geo!r} has {rows.size} periods observed with the one before; the regression's"
            f" {width} columns and two degrees of freedom need {width + 2}"
        )
        raise ValueError(msg)
    out = []
    for c, channel in enumerate(panel.channels):
        y = spend[rows, c]
        if np.ptp(y) == 0.0:
            continue
        x = np.column_stack([calendar[rows], spend[rows - 1, c], kpi[rows - 1]])
        gamma, se, lags = _newey_west_last(x, y, rows, periods)
        b = (lags + 1) / rows.size
        half = sum(c * b**j for j, c in enumerate(_FIXED_B[level])) * se
        out.append(
            SpendFeedback(
                geo=str(geo),
                channel=str(channel),
                gamma=gamma,
                se=se,
                lower=gamma - half,
                upper=gamma + half,
                level=level,
                periods=int(rows.size),
                lags=lags,
            )
        )
    return tuple(out)


def _newey_west_last(
    x: np.ndarray, y: np.ndarray, rows: np.ndarray, periods: int
) -> tuple[float, float, int]:
    """The least squares of `y` on `x`'s columns, read at `rows` of a calendar of `periods`: the
    last column's coefficient, its Newey-West standard error with each residual scaled by one less
    its leverage and the lags counted on the calendar, so a gap holds no pair, and the lags.

    Raises:
        ValueError: on a design not of full rank.
    """
    n, k = x.shape
    if int(np.linalg.matrix_rank(x)) < k:
        msg = (
            "the spend's regression on its last period, the KPI's and the calendar is not of full"
            " rank"
        )
        raise ValueError(msg)
    bread = np.linalg.inv(x.T @ x)
    coefficients = bread @ (x.T @ y)
    leverage = np.einsum("ij,jk,ik->i", x, bread, x)
    scores = np.zeros((periods, k))
    scores[rows] = x * ((y - x @ coefficients) / (1.0 - leverage))[:, None]
    lags = int(4.0 * (n / 100.0) ** (2.0 / 9.0))
    meat = scores.T @ scores
    for lag in range(1, lags + 1):
        pairs = scores[lag:].T @ scores[:-lag]
        meat += (1.0 - lag / (lags + 1.0)) * (pairs + pairs.T)
    covariance = bread @ meat @ bread
    return float(coefficients[-1]), math.sqrt(float(covariance[-1, -1])), lags


__all__ = ["SpendFeedback", "spend_feedback"]
