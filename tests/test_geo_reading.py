"""A geo test read into a lift estimate: difference in differences, time-based regression and a
synthetic control, each returning a `LiftEstimate` and the reading behind it."""

from __future__ import annotations

import functools
import math
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from chc.scm import blank_period_inference
from scipy import stats

from mixdyn import Cadence, Channel, Geo, MediaPanel, Revenue
from mixdyn.experiments import PlaceboNoise
from mixdyn.experiments.candidates import _effective_count
from mixdyn.experiments.reading import (
    GeoTest,
    read_did,
    read_pairs,
    read_synthetic_control,
    read_tbr,
)


def test_a_placebo_standard_error_is_bit_identical_on_a_fixed_series() -> None:
    """`PlaceboNoise` shares its windows' effective count with the geo readers; its outputs are
    the ones it gave before the count moved into a helper, to the last bit."""
    rng = np.random.default_rng(45)
    e = np.empty(80)
    e[0] = rng.normal()
    for t in range(1, 80):
        e[t] = 0.6 * e[t - 1] + rng.normal()
    noise = PlaceboNoise(e, pre=20)
    assert noise.se(6, 0.2) == 2.5293034025579884
    assert noise.se(1, 0.5) == 1.0794452888243633
    # The covariance's lagged dot products take their last bits from the CPU's SIMD kernels, and
    # the helper is not on its path: it is pinned to a few ulps, not to the bit.
    assert float(noise.covariance(4, 0.3).sum()) == pytest.approx(
        5.615726092717138, rel=1e-14, abs=0.0
    )
    np.testing.assert_allclose(
        noise.covariance(3, 0.1),
        [
            [0.1470237782815897, 0.09223067048349372, 0.04616589985374243],
            [0.09223067048349372, 0.1470237782815897, 0.09223067048349372],
            [0.04616589985374243, 0.09223067048349372, 0.1470237782815897],
        ],
        rtol=1e-14,
        atol=0.0,
    )


GEOS = tuple(Geo(f"g{i}") for i in range(10))
TREATED, CONTROL = GEOS[0::2], GEOS[1::2]
WEEKS, START, READOUT = 52, 48, 4
DATES = np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(WEEKS)
TV = Channel("tv")
READERS = {"did": read_did, "tbr": read_tbr, "synthetic_control": read_synthetic_control}
ALPHA, REPLICATIONS = 0.05, 1000


def _panel(kpi: np.ndarray) -> MediaPanel:
    geos, weeks = kpi.shape
    return MediaPanel(
        dates=np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(weeks),
        cadence=Cadence.WEEK,
        geos=GEOS[:geos],
        channels=(TV,),
        controls=(),
        kpi=kpi,
        kpi_unit=Revenue(),
        spend=np.zeros((geos, weeks, 1)),
        currency="EUR",
        control_values=np.zeros((geos, weeks, 0)),
        filled=np.zeros((geos, weeks), dtype=bool),
        population=None,
    )


def _design(**change: Any) -> GeoTest:
    planned = np.full(8, 50.0)
    fields: dict[str, Any] = {
        "channel": TV,
        "treated": TREATED,
        "control": CONTROL,
        "start": DATES[START],
        "end": DATES[START + READOUT - 1],
        "spend": np.r_[planned[:4], 2.0 * planned[4:]],
        "planned": planned,
        "readout": READOUT,
    } | change
    return GeoTest(**fields)


def _still(effect: float) -> np.ndarray:
    """10 geos over 52 weeks, each its level times one seasonal path, `effect` a geo-week in the
    treated geos over weeks 49-52, and no noise over the readout. The pre-period carries a wiggle
    in one treated geo that no reader's fit sees: it sums to 0 against the control sum and a
    constant over the pre-period, and against each control geo and a constant over the periods
    the synthetic control fits on, so the readers read their errors from it and not their lifts."""
    weeks = np.arange(WEEKS)
    season = 0.1 * np.sin(2.0 * np.pi * weeks / 13.0) + 0.05 * np.cos(2.0 * np.pi * weeks / 7.0)
    kpi = (100.0 + 3.0 * np.arange(10.0))[:, None] * (1.0 + season)
    kpi[0::2, START:] += effect
    fitted = START - 8
    rows = [
        np.r_[np.ones(START), np.zeros(WEEKS - START)],
        np.r_[kpi[1::2, :START].sum(0), np.zeros(WEEKS - START)],
    ]
    rows += [np.r_[np.ones(fitted), np.zeros(WEEKS - fitted)]]
    rows += [np.r_[kpi[g, :fitted], np.zeros(WEEKS - fitted)] for g in range(1, 10, 2)]
    a = np.array(rows)
    w = np.r_[np.random.default_rng(11).normal(size=START), np.zeros(WEEKS - START)]
    kpi[0] += w - a.T @ np.linalg.solve(a @ a.T, a @ w)
    return kpi


def _noisy(rng: np.random.Generator, effect: float = 0.0, sd: float = 2.0) -> np.ndarray:
    """10 geos over 52 weeks: levels about 100 of standard deviation 1, noise a first-order
    autoregression of coefficient 0.8 and standard deviation `sd`, and `effect` a geo-week in the
    treated geos over weeks 49-52."""
    rho = 0.8
    noise = np.empty((10, WEEKS))
    noise[:, 0] = rng.normal(scale=sd, size=10)
    for t in range(1, WEEKS):
        noise[:, t] = rho * noise[:, t - 1] + rng.normal(scale=sd * np.sqrt(1 - rho**2), size=10)
    kpi = 100.0 + rng.normal(size=10)[:, None] + noise
    kpi[0::2, START:] += effect
    return kpi


@pytest.mark.parametrize("method", READERS)
def test_each_reader_reads_the_true_lift_of_a_test_without_noise_over_its_readout(
    method: str,
) -> None:
    """10 geos over 52 weeks, +10 a geo-week over weeks 49-52 in 5 geos: the lift is 10 x 5 x 4 =
    200, read over the readout the design names, never over a split the data suggest."""
    estimate, reading = READERS[method](_panel(_still(10.0)), _design())
    assert reading.lift == pytest.approx(200.0, rel=1e-12, abs=0.0)
    assert estimate.lift == reading.lift
    assert estimate.readout == READOUT
    assert estimate.end == DATES[-1]
    assert estimate.geos == frozenset(TREATED)
    assert estimate.share == 1.0
    assert reading.method == method
    assert reading.verdict == "up"


@pytest.mark.parametrize("method", READERS)
def test_a_reading_without_an_effect_reads_a_lift_of_0(method: str) -> None:
    _, reading = READERS[method](_panel(_still(0.0)), _design())
    assert reading.lift == pytest.approx(0.0, abs=1e-9)
    assert reading.verdict == "inconclusive"


@functools.cache
def _null(method: str, seed: int = 2026) -> tuple[float, float]:
    """`method`'s rates of a rejection and of an "up" verdict at 5 % over panels of no effect,
    each reading's p-value and interval checked to agree on the way."""
    rng, rejected, up = np.random.default_rng(seed), 0, 0
    for _ in range(REPLICATIONS):
        _, reading = READERS[method](_panel(_noisy(rng)), _design(), alpha=ALPHA)
        assert (reading.p_value < ALPHA) == (reading.verdict != "inconclusive")
        rejected += reading.verdict != "inconclusive"
        up += reading.verdict == "up"
    return rejected / REPLICATIONS, up / REPLICATIONS


def _bound(rate: float) -> float:
    """`rate` plus 2 Monte Carlo standard errors over the replications."""
    return rate + 2.0 * math.sqrt(rate * (1.0 - rate) / REPLICATIONS)


@pytest.mark.parametrize("method", READERS)
def test_with_noise_carried_over_a_5_percent_test_rejects_a_true_null_5_percent_of_the_time(
    method: str,
) -> None:
    """10 geos over 52 weeks, no effect, geo levels of standard deviation 1 and a noise of 2 at a
    first-order autoregression of 0.8: the test holds its level, where an error that reads the
    periods as independent rejects far more often."""
    rejected, _ = _null(method)
    assert rejected <= _bound(ALPHA)


@pytest.mark.parametrize("method", READERS)
def test_under_no_effect_an_up_verdict_is_read_at_most_half_the_level(method: str) -> None:
    _, up = _null(method)
    assert up <= _bound(ALPHA / 2.0)


@pytest.mark.parametrize("method", READERS)
def test_a_significant_fall_reads_down_and_its_estimate_keeps_the_fall(method: str) -> None:
    """-2.5 a geo-week over 5 geos and 4 weeks, -50 in all: the reading is significant at 1 %
    and says "down", never a success, and the estimate a model reads keeps the lift below 0."""
    kpi = _noisy(np.random.default_rng(3), effect=-2.5, sd=1.0)
    estimate, reading = READERS[method](_panel(kpi), _design())
    assert reading.p_value < 0.01
    assert reading.verdict == "down"
    assert reading.lower < reading.upper < 0.0
    assert estimate.lift == reading.lift < 0.0


@pytest.mark.parametrize("method", READERS)
def test_a_kpi_a_thousand_times_larger_reads_a_lift_and_an_error_a_thousand_times_larger(
    method: str,
) -> None:
    kpi = _noisy(np.random.default_rng(5), effect=1.0)
    small, read_small = READERS[method](_panel(kpi), _design())
    large, read_large = READERS[method](_panel(1000.0 * kpi), _design())
    assert read_large.lift == pytest.approx(1000.0 * read_small.lift, rel=1e-12, abs=0.0)
    assert read_large.se == pytest.approx(1000.0 * read_small.se, rel=1e-12, abs=0.0)
    assert large.se == pytest.approx(1000.0 * small.se, rel=1e-12, abs=0.0)
    assert read_large.df == pytest.approx(read_small.df, rel=1e-12, abs=0.0)
    assert read_large.verdict == read_small.verdict


@pytest.mark.parametrize("method", READERS)
def test_the_estimates_normal_interval_at_95_percent_is_the_readings_t_interval(
    method: str,
) -> None:
    estimate, reading = READERS[method](_panel(_noisy(np.random.default_rng(8))), _design())
    t, z = stats.t.ppf(0.975, reading.df), stats.norm.ppf(0.975)
    assert estimate.se == pytest.approx(reading.se * t / z, rel=1e-12, abs=0.0)
    assert reading.upper - reading.lift == pytest.approx(t * reading.se, rel=1e-12, abs=0.0)
    assert reading.lift - reading.lower == pytest.approx(t * reading.se, rel=1e-12, abs=0.0)
    assert (reading.permutation_p is None) == (method != "synthetic_control")


def test_tbr_under_independent_errors_is_kerman_wang_and_vavers_closed_form() -> None:
    """Three pre-periods and a readout of two, in exact arithmetic: the fit of the treated geo
    on the control, the readout's cumulative effect, and its predictive variance `s^2 (R + R^2 /
    n + (X_R - R xbar)^2 / S_xx)`, read as a t at `n - 2 = 1` degree of freedom."""
    x = [Fraction(v) for v in (10, 12, 15, 13, 11)]
    y = [Fraction(v) for v in (21, 25, 29, 30, 27)]
    n, r = 3, 2
    xbar, ybar = sum(x[:n]) / n, sum(y[:n]) / n
    sxx = sum((v - xbar) ** 2 for v in x[:n])
    slope = sum((u - xbar) * (v - ybar) for u, v in zip(x[:n], y[:n], strict=True)) / sxx
    intercept = ybar - slope * xbar
    rss = sum((v - intercept - slope * u) ** 2 for u, v in zip(x[:n], y[:n], strict=True))
    lift = sum(y[n:]) - r * intercept - slope * sum(x[n:])
    variance = rss / (n - 2) * (r + Fraction(r * r, n) + (sum(x[n:]) - r * xbar) ** 2 / sxx)

    panel = _panel(np.array([[float(v) for v in y], [float(v) for v in x]]))
    design = _design(treated=GEOS[:1], control=GEOS[1:2], start=DATES[n], end=DATES[n], readout=r)
    _, reading = read_tbr(panel, design, variance="iid")
    assert reading.lift == pytest.approx(float(lift), rel=1e-12, abs=0.0)
    assert reading.se == pytest.approx(math.sqrt(variance), rel=1e-12, abs=0.0)
    assert reading.df == 1.0
    half = stats.t.ppf(0.975, 1) * math.sqrt(variance)
    assert reading.upper == pytest.approx(float(lift) + half, rel=1e-12, abs=0.0)
    p = 2.0 * stats.t.sf(abs(float(lift)) / math.sqrt(variance), 1)
    assert reading.p_value == pytest.approx(p, rel=1e-12, abs=0.0)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"treated": ()}, "at least one treated and one control"),
        ({"control": (GEOS[0], GEOS[1])}, "named twice"),
        ({"start": "2024-12-02"}, "start must be a date"),
        ({"end": DATES[START - 1]}, "before it starts"),
        ({"spend": np.full(8, 50.0)}, "changed no spend"),
    ],
)
def test_a_design_refuses_what_names_no_test(change: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _design(**change)


@pytest.mark.parametrize("method", READERS)
@pytest.mark.parametrize(
    ("change", "kpi", "match"),
    [
        ({"control": (*CONTROL, Geo("elsewhere"))}, None, "has no geos"),
        ({"channel": Channel("radio")}, None, "has no channel"),
        ({"start": DATES[START] + np.timedelta64(3, "D")}, None, "periods of the panel"),
        ({"readout": 2, "end": DATES[START + 2]}, None, "past its readout"),
        ({"start": DATES[50], "end": DATES[50]}, None, "ends past the panel"),
        ({"start": DATES[5], "end": DATES[5]}, None, "needs"),
        ({}, "missing", "has no KPI"),
        ({}, "alpha", "not a level"),
    ],
)
def test_a_reader_refuses_a_test_the_panel_cannot_read(
    method: str, change: dict[str, Any], kpi: str | None, match: str
) -> None:
    values = _noisy(np.random.default_rng(1))
    if kpi == "missing":
        values[1, 10] = np.nan  # g1, a control geo
    alpha = 1.0 if kpi == "alpha" else ALPHA
    with pytest.raises(ValueError, match=match):
        READERS[method](_panel(values), _design(**change), alpha=alpha)


def test_a_synthetic_control_refuses_fewer_than_3_donors() -> None:
    with pytest.raises(ValueError, match="at least 3 donors"):
        read_synthetic_control(_panel(_still(10.0)), _design(control=CONTROL[:2]))


def test_tbr_refuses_a_variance_it_does_not_read() -> None:
    with pytest.raises(ValueError, match="'placebo' or 'iid'"):
        read_tbr(_panel(_still(10.0)), _design(), variance="hac")  # ty: ignore[invalid-argument-type]


def test_a_placebo_standard_error_is_its_walk_forward_windows_as_the_module_states() -> None:
    """Each window of the readout's length with 13 periods or more before it, read as the test is
    on the periods before it: by the ratio of means for the difference in differences, by a
    regression for TBR. The error is the placebo lifts' root mean square, its degrees of freedom
    their effective count."""
    kpi = _noisy(np.random.default_rng(13))
    treated, control = kpi[0::2, : START + READOUT].sum(0), kpi[1::2, : START + READOUT].sum(0)
    did, tbr = [], []
    for f in range(13, START - READOUT + 1):
        x, y = control[:f], treated[:f]
        window = slice(f, f + READOUT)
        did.append(treated[window].sum() - y.mean() / x.mean() * control[window].sum())
        slope, intercept = np.polyfit(x, y, 1)
        tbr.append(np.sum(treated[window] - intercept - slope * control[window]))
    for method, lifts in (("did", did), ("tbr", tbr)):
        _, reading = READERS[method](_panel(kpi), _design())
        t = np.array(lifts)
        assert reading.se == pytest.approx(np.sqrt(np.mean(t**2)), rel=1e-9, abs=0.0)
        assert reading.df == pytest.approx(_effective_count(t, float(t @ t)), rel=1e-9, abs=0.0)


def test_a_synthetic_controls_standard_error_is_the_gaps_autoregression_as_the_module_states() -> (
    None
):
    """The treated geos' count times the error of the readout's gaps summed less their mean over
    the fitting periods, under a stationary first-order autoregression at the donors' coefficient,
    its innovations' variance their mean square over the 8 blank periods, read as a t at 8."""
    kpi = _noisy(np.random.default_rng(21))
    fitted, end = START - 8, START + READOUT
    units = np.vstack([kpi[0::2].mean(0), kpi[1::2]])[:, :end]
    units -= units[:, :fitted].mean(axis=1, keepdims=True)
    inference = blank_period_inference(units, 0, START, 8, serial="ar1")
    rho = inference.ar_coefficient
    assert rho is not None
    assert abs(rho) < 1.0
    gaps = units[0] - inference.estimate.weights @ units[1:]
    innovations = gaps[fitted:START] - rho * gaps[fitted - 1 : START - 1]
    lags = np.abs(np.subtract.outer(np.arange(end), np.arange(end)))
    covariance = np.mean(innovations**2) / (1.0 - rho**2) * rho**lags
    contrast = np.r_[np.full(fitted, -READOUT / fitted), np.zeros(8), np.ones(READOUT)]

    _, reading = read_synthetic_control(_panel(kpi), _design())
    assert reading.se == pytest.approx(
        5.0 * np.sqrt(contrast @ covariance @ contrast), rel=1e-9, abs=0.0
    )
    assert reading.lift == pytest.approx(5.0 * gaps[START:].sum(), rel=1e-9, abs=0.0)
    assert reading.df == 8.0
    assert reading.permutation_p == inference.p_value


@pytest.mark.parametrize("method", ["did", "tbr"])
@pytest.mark.parametrize("start", [14, 16, 17, 18])
def test_a_placebo_reading_needs_two_windows_after_its_warmup(method: str, start: int) -> None:
    """A warmup of 13 and a readout of 4: the windows start at periods 13 to `start - 4`, so 17
    periods before the test leave one window, which reads no spread, and 18 the first two."""
    panel = _panel(_noisy(np.random.default_rng(31)))
    design = _design(start=DATES[start], end=DATES[start + READOUT - 1])
    if start < 18:
        windows = max(start - 16, 0)
        with pytest.raises(ValueError, match=f"room for {windows} placebo window.*two need 18"):
            READERS[method](panel, design)
    else:
        _, reading = READERS[method](panel, design)
        assert 1.0 <= reading.df <= 2.0


# `_quarter(10)`'s lift and error as `read_synthetic_control` read them from the readout's own
# `blank_period_inference`, before the gaps were read apart from the permutations. One error for
# each of two x86-64 classes whose vector extensions differ (with and without AVX-512): numpy's
# kernels move its last bits. On a third class, record what origin/main reads there first.
LIFT_10, SE_10 = -32.13006605151425, {42.572777820089705, 42.5727778200897}


def _quarter(readout: int) -> tuple[MediaPanel, GeoTest]:
    """8 geos over 70 weeks, 2 of them treated from week 53, read over `readout` weeks."""
    rng = np.random.default_rng(37)
    kpi = 100.0 + rng.normal(size=(8, 1)) + 0.5 * rng.normal(size=(8, 70)).cumsum(axis=1)
    panel = _panel(kpi)
    planned = np.full(2 * readout, 50.0)
    design = GeoTest(
        channel=TV,
        treated=GEOS[:2],
        control=GEOS[2:8],
        start=panel.dates[52],
        end=panel.dates[52 + readout - 1],
        spend=np.r_[planned[:readout], 2.0 * planned[readout:]],
        planned=planned,
        readout=readout,
    )
    return panel, design


@pytest.mark.parametrize("readout", [12, 13])
def test_a_synthetic_control_reads_a_quarter_without_its_permutation_p_value(
    readout: int,
) -> None:
    """At 8 blank periods, a readout of 12 or 13 weeks has 1 511 640 or 2 645 370 arrangements,
    more than `chc.scm.blank_period_inference` enumerates: the lift and its error, which never
    use them, are read, and the permutation p-value is not."""
    panel, design = _quarter(readout)
    _, reading = read_synthetic_control(panel, design)
    assert reading.permutation_p is None
    assert math.isfinite(reading.lift)
    assert reading.se > 0.0
    assert reading.df == 8.0


def test_a_synthetic_control_reads_an_enumerable_readout_as_the_full_inference_does() -> None:
    """A readout of 10 weeks, 437 580 arrangements: the lift is the treated geos' count times the
    gaps of `blank_period_inference` on the test's own blank and readout periods, its p-value is
    that inference's, and the reading is the one it was before the gaps were read apart from the
    permutations, to the last bit."""
    panel, design = _quarter(10)
    units = np.vstack([panel.kpi[:2].mean(0), panel.kpi[2:]])[:, :62]
    units -= units[:, :44].mean(axis=1, keepdims=True)
    inference = blank_period_inference(units, 0, 52, 8, serial="ar1")
    _, reading = read_synthetic_control(panel, design)
    assert reading.lift == 2.0 * float(inference.estimate.att.sum())
    assert reading.permutation_p == inference.p_value
    assert reading.lift == LIFT_10
    assert reading.se in SE_10


# ------------------------------------------------------------- readings by the pairs' randomisation

PAIRED_WEEKS, PAIRED_START, PAIRED_READOUT = 60, 54, 6


def _paired(pairs: int, rng: np.random.Generator, swapped: int = 0) -> tuple[MediaPanel, Any, Any]:
    """`2 pairs` geos over 60 weeks, each a level and a random walk with noise, no effect: the
    panel, its pairs `(g0, g1), (g2, g3), ...`, and the design treating each pair's first geo but
    in the pairs whose bit is set in `swapped`, treated over the last 6 weeks."""
    geos = 2 * pairs
    kpi = (
        100.0
        + 10.0 * rng.standard_normal((geos, 1))
        + 0.5 * rng.standard_normal((geos, PAIRED_WEEKS)).cumsum(axis=1)
        + rng.standard_normal((geos, PAIRED_WEEKS))
    )
    names = tuple(Geo(f"g{g}") for g in range(geos))
    matched = [(names[2 * p], names[2 * p + 1]) for p in range(pairs)]
    return _panel_of(kpi, names), matched, _paired_design(matched, swapped)


def _panel_of(kpi: np.ndarray, names: tuple[Geo, ...]) -> MediaPanel:
    geos, weeks = kpi.shape
    return MediaPanel(
        dates=np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(weeks),
        cadence=Cadence.WEEK,
        geos=names,
        channels=(TV,),
        controls=(),
        kpi=kpi,
        kpi_unit=Revenue(),
        spend=np.zeros((geos, weeks, 1)),
        currency="EUR",
        control_values=np.zeros((geos, weeks, 0)),
        filled=np.zeros((geos, weeks), dtype=bool),
    )


def _paired_design(matched: list[tuple[Geo, Geo]], swapped: int) -> GeoTest:
    order = [(b, a) if swapped >> p & 1 else (a, b) for p, (a, b) in enumerate(matched)]
    dates = np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(PAIRED_WEEKS)
    planned = np.ones(PAIRED_WEEKS)
    return GeoTest(
        TV,
        tuple(a for a, _ in order),
        tuple(b for _, b in order),
        dates[PAIRED_START],
        dates[-1],
        np.r_[planned[:PAIRED_START], np.zeros(PAIRED_READOUT)],
        planned,
        PAIRED_READOUT,
    )


def test_the_defaults_read_as_they_did_before_the_randomisation_readings() -> None:
    """`read_did` and `read_tbr` gained `variance="unit_placebo"`, `pairs=` and `seed=`: at their
    defaults every reader reads a fixed panel to the bits it read before."""
    rng = np.random.default_rng(58)
    geos = 12
    kpi = (
        100
        + rng.standard_normal((geos, 1)) * 10
        + rng.standard_normal((geos, 60)).cumsum(axis=1) * 0.5
        + rng.standard_normal((geos, 60))
    )
    names = tuple(Geo(f"g{g}") for g in range(geos))
    panel = _panel_of(kpi, names)
    dates = panel.dates
    test = GeoTest(
        TV,
        names[0::2],
        names[1::2],
        dates[54],
        dates[-1],
        np.ones(60),
        np.r_[np.ones(54), np.zeros(6)],
        6,
    )
    # One reading for each of two x86-64 classes whose vector extensions differ (with and without
    # AVX-512): numpy's kernels move the last bits. `read_tbr` reads the same on both.
    before = {
        read_did: {
            (
                -96.8847023842046,
                121.16110978562087,
                -334.35611389092503,
                140.5867091225158,
                0.2672965572111591,
                2.6576977244844753,
            ),
            (
                -96.8847023842046,
                121.16110978562082,
                -334.3561138909249,
                140.58670912251574,
                0.2672965572111588,
                2.6576977244844744,
            ),
        },
        read_tbr: {
            (
                2.6147916046316624,
                36.99550507123424,
                -69.89506592485638,
                75.12464913411971,
                0.9088061069725402,
                2.578986293602999,
            ),
        },
        read_synthetic_control: {
            (
                -51.60610773367631,
                37.75179677211683,
                -125.59826975870077,
                22.386054291348145,
                0.14642932183706034,
                8.0,
            ),
            (
                -51.60610773367631,
                37.75179677211684,
                -125.59826975870078,
                22.38605429134816,
                0.14642932183706045,
                8.0,
            ),
        },
    }
    for reader, pinned in before.items():
        estimate, reading = reader(panel, test)
        read = (estimate.lift, estimate.se, reading.lower, reading.upper, reading.p_value)
        assert (*read, reading.df) in pinned


READINGS = {
    "did": lambda panel, design, pairs: read_did(
        panel, design, variance="unit_placebo", pairs=pairs
    ),
    "tbr": lambda panel, design, pairs: read_tbr(
        panel, design, variance="unit_placebo", pairs=pairs
    ),
    "pairs": read_pairs,
}


@pytest.mark.parametrize("method", READINGS)
def test_over_every_assignment_of_its_pairs_a_randomisation_reading_rejects_at_its_level(
    method: str,
) -> None:
    """One panel with no effect read under each of the 64 assignments of its 6 pairs, as many as
    the design could have drawn: each assignment reads the same set of assignments, so at most
    `ceil(0.05 x 64) - 1 = 3` reject, whatever the noise. The verdict is the p-value's at 5 %,
    and the sign flip's p-values are multiples of 2 / 64, a sign and its reverse alike."""
    panel, matched, _ = _paired(6, np.random.default_rng(7))
    rejected = 0
    for swapped in range(64):
        _, reading = READINGS[method](panel, _paired_design(matched, swapped), matched)
        assert reading.method == method
        assert (reading.p_value < 0.05) == (reading.verdict != "inconclusive")
        assert reading.lower <= reading.lift <= reading.upper
        assert math.isinf(reading.df)
        if method == "pairs":
            assert (reading.p_value * 32) == pytest.approx(round(reading.p_value * 32), abs=1e-9)
        rejected += reading.p_value < 0.05
    assert rejected <= 3


def test_the_sign_flip_reads_each_pair_s_difference_in_differences_summed() -> None:
    """The lift is the pairs' gaps, treated less control, over the readout less their pre-period
    mean, summed; with each geo a level and a path all geos share and a lift over the readout of
    0.5 to 3.5 a week in the treated geos, 2 a geo-week on average, it reads 2 x 7 x 6 = 84."""
    panel, matched, design = _paired(7, np.random.default_rng(3), swapped=0b0101101)
    kpi = panel.kpi
    treated = [panel.geos.index(g) for g in design.treated]
    control = [panel.geos.index(g) for g in design.control]
    order = [(t, c) for t, c in zip(treated, control, strict=True)]
    gaps = np.array([kpi[t] - kpi[c] for t, c in order])
    expected = float(np.sum(gaps[:, PAIRED_START:].sum(1) - 6 * gaps[:, :PAIRED_START].mean(1)))
    _, reading = read_pairs(panel, design, matched)
    assert reading.lift == pytest.approx(expected, rel=1e-12, abs=0.0)
    still = (100.0 + np.arange(14.0))[:, None] + np.sin(np.arange(PAIRED_WEEKS) / 5.0)[None]
    still[treated, PAIRED_START:] += (0.5 + 0.5 * np.arange(7.0))[:, None]
    estimate, reading = read_pairs(_panel_of(still, panel.geos), design, matched)
    assert reading.lift == pytest.approx(84.0, rel=1e-12, abs=0.0)
    assert estimate.lift == reading.lift


def test_the_sign_flip_interval_ends_where_its_test_of_a_shared_lift_rejects() -> None:
    """The interval inverts the sign-flip test of a lift `L` shared alike by the pairs: the KPI
    less `L / (P R)` in each treated geo-week of the readout reads a p-value of at least 5 % just
    inside either end and below 5 % just outside it. The estimate's error is the farther end of
    the 95 % interval from the lift over `z(0.975)`."""
    panel, matched, design = _paired(7, np.random.default_rng(11), swapped=0b0011010)
    estimate, reading = read_pairs(panel, design, matched)
    treated = [panel.geos.index(g) for g in design.treated]
    width = reading.upper - reading.lower
    for end, outward in ((reading.lower, -1.0), (reading.upper, 1.0)):
        for step, rejects in ((-1e-6, False), (1e-6, True)):
            shifted = panel.kpi.copy()
            lift = end + outward * step * width
            shifted[treated, PAIRED_START:] -= lift / (7 * PAIRED_READOUT)
            _, at = read_pairs(_panel_of(shifted, panel.geos), design, matched)
            assert (at.p_value < 0.05) == rejects
    farther = max(reading.upper - reading.lift, reading.lift - reading.lower)
    assert estimate.se == pytest.approx(farther / stats.norm.ppf(0.975), rel=1e-12, abs=0.0)


def test_unit_placebos_studentise_each_assignment_by_its_held_out_error() -> None:
    """The observed assignment's interval is its lift plus or minus its error over the last 6
    pre-period weeks, the difference in differences refitted on the 48 before them, times the
    `ceil(0.05 x 64) = 4`th largest studentised lift; the lift is `read_did`'s own."""
    panel, matched, design = _paired(6, np.random.default_rng(5), swapped=0b100110)
    _, default = read_did(panel, design)
    estimate, reading = read_did(panel, design, variance="unit_placebo", pairs=matched)
    assert reading.lift == pytest.approx(default.lift, rel=1e-12, abs=0.0)

    def studentised(swapped: int) -> tuple[float, float]:
        test = _paired_design(matched, swapped)
        t = panel.kpi[[panel.geos.index(g) for g in test.treated]].sum(0)[:PAIRED_START]
        c = panel.kpi[[panel.geos.index(g) for g in test.control]].sum(0)[:PAIRED_START]
        fit = PAIRED_START - PAIRED_READOUT
        gaps = t - t[:fit].mean() / c[:fit].mean() * c
        error = math.sqrt(np.mean((gaps[fit:] - gaps[:fit].mean()) ** 2))
        lift = read_did(panel, test)[1].lift
        return lift, error

    lift, error = studentised(0b100110)
    largest = sorted(abs(a / e) for a, e in map(studentised, range(64)))[64 - 4]
    half = largest * error
    assert reading.lower == pytest.approx(lift - half, rel=1e-9, abs=0.0)
    assert reading.upper == pytest.approx(lift + half, rel=1e-9, abs=0.0)
    assert estimate.se * stats.norm.ppf(0.975) == pytest.approx(half, rel=1e-9, abs=0.0)


@pytest.mark.skipif(shutil.which("maxima") is None, reason="maxima is not on PATH")
def test_maxima_checks_the_sign_flip_interval_s_ends_in_closed_form() -> None:
    """`validation/sign_flip_interval.mac`: an assignment under a shared lift, the end of the
    lifts at which it is as far from 0 as the observed one, and the count of such assignments
    over every sign of 4 pairs by the definition and by the ends, each a PASS."""
    script = Path(__file__).parents[1] / "validation" / "sign_flip_interval.mac"
    out = subprocess.run(
        ["maxima", "--very-quiet", "-b", str(script)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    ).stdout
    assert "Lisp error" not in out
    assert "(FAIL)" not in out
    # the verdict's own definition is echoed with its "(PASS)" in quotes
    assert out.count("(PASS)") - out.count('"(PASS)"') == 4


def test_pairs_past_the_count_to_enumerate_are_drawn_from_a_child_of_the_seed() -> None:
    """17 pairs have 131 072 assignments, past the 100 000 enumerated: the observed one and 10 000
    drawn by the seed are read, the same at one seed and not at another."""
    panel, matched, design = _paired(17, np.random.default_rng(2))
    one = read_pairs(panel, design, matched, seed=1)[1]
    assert one == read_pairs(panel, design, matched, seed=1)[1]
    assert one.p_value != read_pairs(panel, design, matched, seed=2)[1].p_value
    assert one.p_value * 10_001 == pytest.approx(round(one.p_value * 10_001), abs=1e-6)


def test_what_a_randomisation_reading_cannot_read_is_refused() -> None:
    """Five pairs' 32 assignments cannot reject a sign flip at 5 %: a sign and its reverse read
    alike, so the least p-value is 2 / 32. Unit placebos need the pairs, and pairs must each be
    one treated and one control geo of the test, all of them."""
    panel, matched, design = _paired(5, np.random.default_rng(1))
    with pytest.raises(ValueError, match="too few for the test to reject"):
        read_pairs(panel, design, matched)
    read_did(panel, design, variance="unit_placebo", pairs=matched)
    panel, matched, design = _paired(6, np.random.default_rng(1))
    with pytest.raises(ValueError, match="name them by pairs="):
        read_did(panel, design, variance="unit_placebo")
    with pytest.raises(ValueError, match="every geo of the test once"):
        read_pairs(panel, design, matched[:-1])
    crossed = [(matched[0][0], matched[1][0]), (matched[0][1], matched[1][1]), *matched[2:]]
    with pytest.raises(ValueError, match="one treated and one control"):
        read_tbr(panel, design, variance="unit_placebo", pairs=crossed)
    with pytest.raises(ValueError, match="variance must be"):
        read_did(panel, design, variance="iid")  # ty: ignore[invalid-argument-type]
