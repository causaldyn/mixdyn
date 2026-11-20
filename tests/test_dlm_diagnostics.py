"""The engine's diagnostics: tests on the one-step errors, drift, and the decomposition's checks.

The world is static, 156 weeks of `y = 100 + 0.8 x + N(0, 8^2)` with `x = max(0, N(50, 20))`, so a
discount of 1 is the true model and every lower one over-fits. The diagnostics read the one-step
forecasts.
"""

from __future__ import annotations

import dataclasses
import functools

import numpy as np
import pytest
from scipy import stats

from mixdyn import NATIONAL, Cadence, Count, MediaPanel, read_panel
from mixdyn.models import (
    CheckCode,
    DlmFit,
    DlmSpec,
    Effect,
    Trend,
    diagnose_dlm,
    fit_dlm,
)

DATES = np.datetime64("2023-01-02") + np.timedelta64(7, "D") * np.arange(156)
_RNG = np.random.default_rng(11)
X = np.maximum(0.0, _RNG.normal(50.0, 20.0, DATES.size))
KPI = 100.0 + 0.8 * X + _RNG.normal(0.0, 8.0, DATES.size)
DISCOUNTS = [0.8, 0.9, 0.95, 0.99, 1.0]


def static(kpi: np.ndarray | None = None) -> MediaPanel:
    return read_panel(
        {"date": DATES, "kpi": KPI if kpi is None else kpi, "x": X},
        date="date",
        kpi="kpi",
        spend={"x": "x"},
        cadence=Cadence.WEEK,
        kpi_unit=Count("units"),
        currency="USD",
    )


def spec(discount: float) -> DlmSpec:
    """A level and x, one discount for both."""
    return DlmSpec((Effect("x", discount),), trend=Trend(1, discount))


@functools.cache
def fit(discount: float = 1.0) -> DlmFit:
    return fit_dlm(static(), NATIONAL, spec(discount))


def test_the_one_step_likelihood_ranks_the_static_truth_first() -> None:
    """A discount of 1 is the true model. The log-likelihood rises from -578.3 at 0.80 to -547.5
    at 1."""
    ours = [diagnose_dlm(fit(d)).log_likelihood for d in DISCOUNTS]
    assert np.all(np.diff(ours) > 0.0)


def test_an_over_fitted_discount_leaves_its_one_step_errors_autocorrelated() -> None:
    """The state chases the noise, so each error partly undoes the last: at 0.80 Ljung-Box rejects
    at 1 % and Durbin-Watson reads 2.63; at the true discount neither moves."""
    over, true = diagnose_dlm(fit(0.80)), diagnose_dlm(fit(1.0))
    assert over.ljung_box.p_value < 0.01 < true.ljung_box.p_value
    assert over.durbin_watson > 2.4
    assert abs(true.durbin_watson - 2.0) < 0.2


def test_the_scores_are_the_one_step_errors_normal_scores() -> None:
    filtered = fit().filtered
    u = filtered.errors / np.sqrt(filtered.one_step_scale)
    expected = stats.norm.ppf(stats.t.cdf(u, filtered.one_step_dof))
    assert np.abs(diagnose_dlm(fit()).scores - expected).max() <= 1e-9


def textbook_ljung_box(z: np.ndarray, lags: int) -> float:
    n, centred = z.size, z - z.mean()
    acf = [centred[:-k] @ centred[k:] / (centred @ centred) for k in range(1, lags + 1)]
    return n * (n + 2) * sum(r**2 / (n - k) for k, r in enumerate(acf, start=1))


def test_ljung_box_is_the_textbook_statistic_with_no_period_missing() -> None:
    found = diagnose_dlm(fit())
    assert found.lags == 10
    assert found.ljung_box.value == pytest.approx(textbook_ljung_box(found.scores, 10), rel=1e-12)
    assert found.ljung_box.p_value == pytest.approx(stats.chi2.sf(found.ljung_box.value, 10))


def test_a_missing_period_breaks_the_pairs_across_it_rather_than_closing_the_gap() -> None:
    kpi = KPI.copy()
    kpi[80] = np.nan
    gapped = fit_dlm(static(kpi), NATIONAL, spec(1.0))
    found = diagnose_dlm(gapped)
    assert found.dates.size == 155
    assert gapped.dates[80] not in found.dates
    calendar = np.insert(found.scores, 80, np.nan)
    centred = calendar - np.nanmean(calendar)
    total = np.nansum(centred**2)
    q = 0.0
    for k in range(1, 11):
        products = centred[:-k] * centred[k:]
        q += (np.nansum(products) / total) ** 2 / np.isfinite(products).sum()
    assert found.ljung_box.value == pytest.approx(155 * 157 * q, rel=1e-12)
    closed = textbook_ljung_box(found.scores, 10)
    assert found.ljung_box.value != pytest.approx(closed, rel=1e-6)
    pairs = np.diff(calendar)
    expected = np.nansum(pairs**2) / np.sum(found.scores**2)
    assert found.durbin_watson == pytest.approx(expected, rel=1e-12)


def test_ljung_box_runs_over_a_fifth_of_a_short_series() -> None:
    kpi = KPI.copy()
    kpi[40:] = np.nan
    assert diagnose_dlm(fit_dlm(static(kpi), NATIONAL, spec(1.0))).lags == 8


def test_a_forecast_too_narrow_fails_normality_though_its_shape_is_normal() -> None:
    """Halving every one-step scale doubles the scores: Shapiro-Wilk, which reads only the shape,
    cannot see it."""
    base = fit()
    narrow = dataclasses.replace(
        base,
        filtered=dataclasses.replace(
            base.filtered, one_step_scale=base.filtered.one_step_scale / 4
        ),
    )
    assert diagnose_dlm(base).normality.p_value > 0.05
    assert diagnose_dlm(narrow).normality.p_value < 0.01
    assert stats.shapiro(diagnose_dlm(narrow).scores).pvalue > 0.05


def test_durbin_watson_reads_consecutive_periods() -> None:
    z = diagnose_dlm(fit()).scores
    assert diagnose_dlm(fit()).durbin_watson == pytest.approx(
        np.sum(np.diff(z) ** 2) / np.sum(z**2), rel=1e-12
    )


def test_a_level_shift_sounds_the_drift_alarm_on_its_first_week() -> None:
    """Five noise SDs from week 100 on, with week 50 missing. Without the shift the alarm never
    sounds."""
    kpi = KPI.copy()
    kpi[100:] += 40.0
    kpi[50] = np.nan
    shifted = diagnose_dlm(fit_dlm(static(kpi), NATIONAL, spec(1.0)))
    assert shifted.drift[0] == DATES[100]
    assert diagnose_dlm(fit()).drift.size == 0


def test_an_average_run_length_not_above_one_is_refused() -> None:
    with pytest.raises(ValueError, match="arl must exceed 1"):
        diagnose_dlm(fit(), arl=1.0)


def test_a_negative_effect_puts_the_base_above_the_kpi_and_the_finding_counts_its_weeks() -> None:
    """A world whose effect is -0.5, fitted without a sign."""
    noise = np.random.default_rng(5).normal(0.0, 5.0, X.size)
    found = diagnose_dlm(fit_dlm(static(100.0 - 0.5 * X + noise), NATIONAL, spec(1.0)))
    (finding,) = [f for f in found.findings if f.code is CheckCode.BASE_ABOVE_KPI]
    assert finding.dates.size >= 140
    assert f"in {finding.dates.size} of 156 periods" in finding.message


def test_a_negative_base_and_an_outsized_effect_are_found_on_their_dates() -> None:
    base = fit()
    assert base.base is not None
    top = float(np.nanmax(base.y))
    negative = base.base.copy()
    negative[[3, 7]] = -1.0
    outsized = base.contributions["x"].copy()
    outsized[[5, 9, 11, 13]] = 2.5 * top
    found = diagnose_dlm(
        dataclasses.replace(base, base=negative, contributions={"x": outsized})
    ).findings
    by_code = {f.code: f for f in found}
    assert np.array_equal(by_code[CheckCode.NEGATIVE_BASE].dates, base.dates[[3, 7]])
    effect = by_code[CheckCode.EFFECT_ABOVE_TWICE_MAX]
    assert effect.effect == "x"
    assert np.array_equal(effect.dates, base.dates[[5, 9, 11, 13]])
    assert effect.message.endswith(f"{base.dates[11]}, ...")


def test_diagnostics_need_ten_observed_periods() -> None:
    kpi = np.full(KPI.size, np.nan)
    kpi[:9] = KPI[:9]
    sparse = fit_dlm(static(kpi), NATIONAL, spec(1.0))
    with pytest.raises(ValueError, match="need ten observed periods; the fit has 9"):
        diagnose_dlm(sparse)
