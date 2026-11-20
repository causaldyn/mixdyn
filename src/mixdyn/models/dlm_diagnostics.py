"""What a fitted DLM's one-step errors and decomposition say about it.

The tests read the standardised one-step errors, not the smoothed residuals. Under the model each
one-step forecast is Student t given the past, so its probability integral transform is uniform
and independent of the past, and the normal scores `Φ⁻¹(PIT)` are independent standard normals.
Smoothed residuals obey no such law: each has seen the whole series, and the lower the discount
the closer the smoother follows the data, so a fit read off them looks better the more it
over-fits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from chc.dlm import monitor_evalues
from chc.gate import DriftAlarm
from scipy import stats

if TYPE_CHECKING:
    from mixdyn.models.dlm import DlmFit

log = logging.getLogger(__name__)

type _Array = npt.NDArray[np.float64]
type _Dates = npt.NDArray[np.datetime64]

_LAGS = 10
_LISTED = 3


@dataclass(frozen=True, slots=True)
class Statistic:
    """A test's statistic and its p-value under the model."""

    value: float
    p_value: float


class CheckCode(Enum):
    NEGATIVE_BASE = "negative_base"
    """The base is below zero."""
    BASE_ABOVE_KPI = "base_above_kpi"
    """The base is above the observed KPI, so the effects sum below zero there."""
    EFFECT_ABOVE_TWICE_MAX = "effect_above_twice_max"
    """An effect's contribution is above twice the KPI's largest observed value."""
    UNMIXED = "unmixed"
    """The chains that drew the posterior truncated to the signs have not mixed, so no period's
    decomposition can be read."""


@dataclass(frozen=True, slots=True, eq=False)
class Finding:
    """A plausibility check the decomposition fails: on which effect, if one, and on which dates.
    `message` lists only the first few dates."""

    code: CheckCode
    message: str
    effect: str | None
    dates: _Dates


@dataclass(frozen=True, slots=True, eq=False)
class DlmDiagnostics:
    """A fit's checks.

    `dates` are the observed periods and `scores` their one-step errors as normal scores.
    `ljung_box` tests the scores for autocorrelation over `lags` lags, pairing periods by the
    calendar, so a missing period breaks a pair rather than closing up the gap; `normality` tests
    them for the standard normal law (Cramér-von Mises), which a forecast that is too narrow, too
    wide or biased fails as surely as a skewed one; `durbin_watson` is their lag-one statistic,
    near 2 when they are uncorrelated.

    `log_likelihood` sums the one-step log predictive densities: the criterion to compare models
    by. `drift` holds the dates where a Shiryaev-Roberts alarm sounded, against one-step errors
    shifted by two scales either way or of twice the scale, with a false alarm every `arl` periods
    on average. `findings` holds the decomposition's failed plausibility checks.
    """

    dates: _Dates
    scores: _Array
    lags: int
    ljung_box: Statistic
    normality: Statistic
    durbin_watson: float
    log_likelihood: float
    arl: float
    drift: _Dates
    findings: tuple[Finding, ...]


def diagnose_dlm(fit: DlmFit, arl: float = 1000.0) -> DlmDiagnostics:
    """Test `fit`'s one-step errors, watch them for drift, and check its decomposition.

    Ljung-Box runs over `min(10, n // 5)` lags for `n` observed periods.

    Raises:
        ValueError: on fewer than ten observed periods, or an `arl` not above 1.
    """
    filtered = fit.filtered
    observed = np.isfinite(fit.y)
    n = int(observed.sum())
    if n < 10:
        msg = f"diagnostics need ten observed periods; the fit has {n}"
        raise ValueError(msg)
    alarm = DriftAlarm(arl)
    u = filtered.errors / np.sqrt(filtered.one_step_scale)
    calendar = np.sign(u) * stats.norm.isf(stats.t.sf(np.abs(u), filtered.one_step_dof))
    scores = calendar[observed]
    lags = min(_LAGS, n // 5)
    normality = stats.cramervonmises(scores, "norm")
    pairs = np.diff(calendar)
    evalues = monitor_evalues(filtered)
    sounded = [i for i in range(evalues.shape[0]) if alarm.update(evalues[i : i + 1])]
    return DlmDiagnostics(
        dates=fit.dates[observed],
        scores=scores,
        lags=lags,
        ljung_box=_ljung_box(calendar, lags),
        normality=Statistic(float(normality.statistic), float(normality.pvalue)),
        durbin_watson=float(np.nansum(pairs**2) / np.sum(scores**2)),
        log_likelihood=filtered.log_likelihood,
        arl=arl,
        drift=fit.dates[sounded],
        findings=_findings(fit),
    )


def _ljung_box(z: _Array, lags: int) -> Statistic:
    """Ljung-Box over a calendar with NaN on its missing periods: lag `k` pairs the periods `k`
    apart that are both observed, and with no period missing the statistic is the textbook one."""
    centred = z - np.nanmean(z)
    n = int(np.isfinite(z).sum())
    total = float(np.nansum(centred**2))
    q = 0.0
    for k in range(1, lags + 1):
        products = centred[:-k] * centred[k:]
        q += (np.nansum(products) / total) ** 2 / np.isfinite(products).sum()
    q *= n * (n + 2)
    return Statistic(float(q), float(stats.chi2.sf(q, lags)))


def _findings(fit: DlmFit) -> tuple[Finding, ...]:
    """The base below zero or above the observed KPI, any effect above twice the KPI's largest
    observed value, and, with signed effects, chains that have not mixed."""
    periods = fit.dates.size
    top = float(np.nanmax(fit.y))
    failed: list[tuple[CheckCode, str | None, npt.NDArray[np.bool_], str]] = []
    if fit.signed is not None and not fit.signed.mixed:
        what = (
            f"the signed posterior's chains have not mixed (R-hat {fit.signed.rhat:.3f}, bulk ESS "
            f"{fit.signed.ess:.0f}), so the decomposition is not to be read"
        )
        failed.append((CheckCode.UNMIXED, None, np.ones(periods, dtype=bool), what))
    if fit.base is not None:
        failed.append((CheckCode.NEGATIVE_BASE, None, fit.base < 0.0, "the base is below zero"))
        above = fit.base > np.where(np.isfinite(fit.y), fit.y, np.inf)
        failed.append((CheckCode.BASE_ABOVE_KPI, None, above, "the base is above the KPI"))
    for effect, contribution in fit.contributions.items():
        high = contribution > 2.0 * top
        what = f"{effect!r}'s contribution is above twice the KPI's largest value, {top:.6g},"
        failed.append((CheckCode.EFFECT_ABOVE_TWICE_MAX, effect, high, what))
    findings = []
    for code, effect, where, what in failed:
        dates = fit.dates[where]
        if dates.size:
            listed = ", ".join(str(d) for d in dates[:_LISTED])
            more = ", ..." if dates.size > _LISTED else ""
            message = f"{what} in {dates.size} of {periods} periods: {listed}{more}"
            log.warning("%s", message)
            findings.append(Finding(code, message, effect, dates))
    return tuple(findings)
