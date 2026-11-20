"""The engine's dynamic linear model: its steady state against Maxima, its default prior's
equivariances, an idle effect held, and what it refuses.

The default prior's tests read a world of 104 weeks: a trend with a slope, a 52-week season, two
channels and two controls, every effect constant.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from chc.dlm import Prior

from mixdyn import NATIONAL, Cadence, Count, MediaPanel, read_panel
from mixdyn.models import DlmFit, DlmSpec, Effect, Season, Trend, fit_dlm, vague_prior


def weekly() -> MediaPanel:
    """104 weeks of `y = 2000 + 5 t + 300 sin(2 pi t / 52) + 0.4 search + 0.6 social + 150 events
    + 400 launches + N(0, 100^2)`."""
    rng = np.random.default_rng(20261004)
    weeks = np.arange(104)
    search = rng.gamma(2.0, 500.0, weeks.size)
    social = rng.gamma(2.0, 300.0, weeks.size) * (rng.random(weeks.size) < 0.8)
    events = rng.poisson(1.0, weeks.size).astype(float)
    launches = (rng.random(weeks.size) < 0.05).astype(float)
    kpi = (
        2000.0
        + 5.0 * weeks
        + 300.0 * np.sin(2.0 * np.pi * weeks / 52.0)
        + 0.4 * search
        + 0.6 * social
        + 150.0 * events
        + 400.0 * launches
        + rng.normal(0.0, 100.0, weeks.size)
    )
    return read_panel(
        {
            "date": np.datetime64("2022-01-03") + np.timedelta64(7, "D") * weeks,
            "kpi": kpi,
            "spend_search": search,
            "spend_social": social,
            "events": events,
            "launches": launches,
        },
        date="date",
        kpi="kpi",
        spend={"spend_search": "spend_search", "spend_social": "spend_social"},
        controls=["events", "launches"],
        cadence=Cadence.WEEK,
        kpi_unit=Count("units"),
        currency="USD",
    )


def relative_gap(ours: np.ndarray, theirs: np.ndarray) -> float:
    """The largest difference over `theirs`'s largest magnitude."""
    return float(np.max(np.abs(ours - theirs)) / np.max(np.abs(theirs)))


def test_an_idle_effect_is_held_while_its_column_is_zero() -> None:
    """A holiday that first comes on the 326th day of a daily series. Held while idle, its
    coefficient's variance stays where it was over the 325 days before; discounted at every step,
    it grows by `0.95 ** -324`, 1.65e7-fold."""
    rng = np.random.default_rng(3)
    days = 365
    holiday = np.zeros(days)
    holiday[[325, 326, 358, 359]] = 1.0
    spend = rng.gamma(2.0, 100.0, days)
    data = read_panel(
        {
            "date": np.datetime64("2024-01-01") + np.arange(days),
            "kpi": 500.0 + 0.5 * spend + 80.0 * holiday + rng.normal(0.0, 20.0, days),
            "spend": spend,
            "holiday": holiday,
        },
        date="date",
        kpi="kpi",
        spend={"spend": "spend"},
        controls=["holiday"],
        cadence=Cadence.DAY,
        kpi_unit=Count("units"),
        currency="USD",
    )
    dark = 325

    def variance(hold: bool) -> np.ndarray:
        """In units of the observational variance's estimate, which the data move either way."""
        spec = DlmSpec(
            (Effect("spend", 0.99), Effect("holiday", 0.95, hold_when_idle=hold)),
            trend=Trend(1, 0.99),
        )
        filtered = fit_dlm(data, NATIONAL, spec).filtered
        return filtered.covariance[:dark, 2, 2] / filtered.scale[:dark]

    held = variance(True)
    assert (np.diff(held) <= 1e-12 * held[:-1]).all()
    grown = variance(False)
    assert grown[-1] / grown[0] > 1e6


def test_a_discounted_level_settles_at_its_maxima_steady_state() -> None:
    """Textbook mode against CHC's `validation/discount_dlm.mac`, STEP 4: with `V` known, a level
    discounted at `delta` settles at `C = V (1 - delta)`, its gain at `1 - delta`, and its
    discount at the fixed evolution variance `W / V = (1 - delta)^2 / delta`."""
    delta, v = 0.8, 2.0
    rng = np.random.default_rng(2)
    data = read_panel(
        {
            "date": np.datetime64("2020-01-06") + 7 * np.arange(400),
            "kpi": rng.standard_normal(400),
            "x": rng.gamma(2.0, 1.0, 400),
        },
        date="date",
        kpi="kpi",
        spend={"x": "x"},
        cadence=Cadence.WEEK,
        kpi_unit=Count("units"),
        currency="USD",
    )
    prior = Prior(np.zeros(1), np.eye(1), v, math.inf)
    fit = fit_dlm(data, NATIONAL, DlmSpec((), trend=Trend(1, delta)), prior).filtered
    assert fit.covariance[-1, 0, 0] == pytest.approx(v * (1.0 - delta), rel=1e-12, abs=0.0)
    assert fit.prior_covariance[-1, 0, 0] / fit.one_step_scale[-1] == pytest.approx(
        1.0 - delta, rel=1e-12, abs=0.0
    )
    w = fit.prior_covariance[-1, 0, 0] - fit.covariance[-2, 0, 0]
    assert w / v == pytest.approx((1.0 - delta) ** 2 / delta, rel=1e-12, abs=0.0)


FULL = DlmSpec(
    effects=tuple(Effect(c, 0.95) for c in ("spend_search", "spend_social", "events", "launches")),
    trend=Trend(2, 0.95),
    season=Season(52.0, 2, 0.98),
)
EQUIVARIANT = 1e-11
"""The default fit is equivariant to 4.9e-14 under a KPI unit, a KPI shift and a column's unit."""


def parts(fit: DlmFit) -> dict[str, np.ndarray]:
    found = {"base": fit.base, "seasonality": fit.seasonality, "estimated": fit.estimated}
    return {k: v for k, v in (found | dict(fit.contributions)).items() if v is not None}


def test_the_default_prior_makes_the_fit_equivariant_to_the_kpi_unit() -> None:
    data = weekly()
    ours = parts(fit_dlm(data, NATIONAL, FULL))
    scaled = parts(fit_dlm(dataclasses.replace(data, kpi=data.kpi * 1000.0), NATIONAL, FULL))
    gaps = {k: relative_gap(scaled[k], 1000.0 * ours[k]) for k in ours}
    assert max(gaps.values()) <= EQUIVARIANT, gaps


def test_the_default_prior_moves_only_the_base_when_the_kpi_shifts() -> None:
    """The level starts at the KPI's mean, so a constant added to the KPI lands in the base."""
    data = weekly()
    ours = parts(fit_dlm(data, NATIONAL, FULL))
    shift = 100.0 * float(np.nanstd(data.kpi))
    shifted = parts(fit_dlm(dataclasses.replace(data, kpi=data.kpi + shift), NATIONAL, FULL))
    moved = {"base", "estimated"}
    gaps = {
        k: float(np.max(np.abs(shifted[k] - ours[k] - (shift if k in moved else 0.0))))
        / float(np.max(np.abs(ours[k])))
        for k in ours
    }
    assert max(gaps.values()) <= EQUIVARIANT, gaps


def test_the_default_prior_leaves_contributions_alone_when_a_column_is_rescaled() -> None:
    data = weekly()
    spend = data.spend.copy()
    spend[:, :, data.channels.index("spend_search")] *= 1000.0
    ours = parts(fit_dlm(data, NATIONAL, FULL))
    rescaled = parts(fit_dlm(dataclasses.replace(data, spend=spend), NATIONAL, FULL))
    gaps = {k: relative_gap(rescaled[k], ours[k]) for k in ours}
    assert max(gaps.values()) <= EQUIVARIANT, gaps


def test_the_default_prior_lets_the_slope_move_the_level_as_far_as_the_level_spreads() -> None:
    data = weekly()
    prior = vague_prior(data, NATIONAL, FULL)
    level, slope = prior.covariance[0, 0], prior.covariance[1, 1]
    assert slope * data.dates.size**2 == pytest.approx(level, rel=1e-12)


def test_a_kpi_that_does_not_vary_sets_no_prior() -> None:
    data = weekly()
    flat = dataclasses.replace(data, kpi=np.full_like(data.kpi, 7.0))
    with pytest.raises(ValueError, match="does not vary"):
        vague_prior(flat, NATIONAL, FULL)


def test_an_effect_named_twice_is_refused() -> None:
    with pytest.raises(ValueError, match=r"\['spend_meta'\] more than once"):
        DlmSpec((Effect("spend_meta", 0.99), Effect("spend_meta", 0.95)))


def test_an_effect_the_panel_lacks_is_refused_by_name() -> None:
    spec = DlmSpec((Effect("spend_radio", 0.99),))
    prior = Prior(np.zeros(1), np.eye(1), 1.0, math.inf)
    with pytest.raises(ValueError, match=r"no channel or control named \['spend_radio'\]"):
        fit_dlm(weekly(), NATIONAL, spec, prior)
