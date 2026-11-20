"""Media transforms: Robyn's numbers at Robyn's hyperparameters, `chc.response`'s channel equal to
the NumPy path, and the invariances a kernel kept in periods and a curve read off the window
promise."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mixdyn.transforms import (
    Geometric,
    Hill,
    WeibullCdf,
    carry_over,
    channel,
    half_saturation,
    robyn_weibull,
    saturate,
)

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "oracles" / "robyn_transforms.json").read_text()
)


def _close(ours: np.ndarray, theirs: list[float], tolerance: float) -> None:
    expected = np.array(theirs)
    assert np.abs(ours - expected).max() <= tolerance * np.abs(expected).max()


@pytest.mark.parametrize("case", FIXTURE["geometric"], ids=lambda c: f"{c['series']}-{c['theta']}")
def test_geometric_carryover_is_robyn_s(case: dict) -> None:
    spend = FIXTURE["series"][case["series"]]
    _close(carry_over(spend, Geometric(case["theta"])), case["decayed"], 1e-15)


@pytest.mark.parametrize(
    "case",
    FIXTURE["weibull"],
    ids=lambda c: f"{c['series']}-{c['shape']}-{c['scale']}",
)
def test_weibull_carryover_is_robyn_s_with_its_scale_a_share_of_the_series(case: dict) -> None:
    spend = FIXTURE["series"][case["series"]]
    adstock = robyn_weibull(case["shape"], case["scale"], len(spend))
    _close(carry_over(spend, adstock), case["decayed"], 1e-14)


def test_robyn_s_scale_rounds_half_to_even_and_a_zero_carries_nothing() -> None:
    assert robyn_weibull(2.0, 0.5, 60) == WeibullCdf(2.0, 30.0)
    assert robyn_weibull(2.0, 0.5, 62) == WeibullCdf(2.0, 32.0)
    assert robyn_weibull(0.0, 0.5, 60) == robyn_weibull(2.0, 0.0, 60) == Geometric(0.0)


@pytest.mark.parametrize(
    "case", FIXTURE["hill"]["runs"], ids=lambda c: f"{c['alpha']}-{c['gamma']}"
)
def test_the_hill_is_robyn_s_on_the_window_and_at_other_points(case: dict) -> None:
    hill = Hill(case["alpha"], case["gamma"])
    half = half_saturation(hill, FIXTURE["hill"]["window"])
    assert half == pytest.approx(case["inflexion"], rel=1e-15, abs=0.0)
    _close(saturate(FIXTURE["hill"]["window"], hill, half), case["saturated"], 1e-15)
    _close(saturate(FIXTURE["hill"]["marginal"], hill, half), case["marginal"], 1e-15)


@pytest.mark.parametrize("case", FIXTURE["hill"]["zero"], ids=lambda c: f"{c['gamma']}")
def test_the_hill_is_robyn_s_on_a_window_with_a_dark_period(case: dict) -> None:
    hill = Hill(case["alpha"], case["gamma"])
    spend = FIXTURE["series"]["long"]
    half = half_saturation(hill, spend)
    assert half == pytest.approx(case["inflexion"], rel=1e-15, abs=0.0)
    _close(saturate(spend, hill, half), case["saturated"], 1e-15)


@pytest.mark.parametrize(
    "adstock",
    [Geometric(0.0), Geometric(0.7), WeibullCdf(0.6, 3.0), WeibullCdf(4.0, 9.5)],
    ids=repr,
)
def test_chc_s_channel_is_the_numpy_path(adstock: Geometric | WeibullCdf) -> None:
    """The decisions read a fit through `chc.response`; the fit itself through NumPy."""
    spend = np.array(FIXTURE["series"]["long"])
    hill = Hill(2.5, 0.4)
    half = half_saturation(hill, carry_over(spend, adstock)[10:])
    planned = channel(adstock, hill, half, 3.5, spend.size)
    fitted = 3.5 * saturate(carry_over(spend, adstock), hill, half)
    np.testing.assert_allclose(np.asarray(planned(spend)), fitted, rtol=1e-14, atol=1e-15)


adstocks = st.one_of(
    st.builds(Geometric, st.floats(0.0, 0.99)),
    st.builds(WeibullCdf, st.floats(0.1, 10.0), st.floats(0.1, 40.0)),
)


@settings(max_examples=200, deadline=None)
@given(
    adstocks,
    st.floats(0.2, 5.0),
    st.floats(0.05, 1.0),
    st.floats(-6.0, 6.0),
    st.integers(0, 2**32 - 1),
)
def test_a_currency_moves_nothing_once_saturated(
    adstock: Geometric | WeibullCdf, alpha: float, gamma: float, power: float, seed: int
) -> None:
    spend = np.random.default_rng(seed).gamma(2.0, 100.0, 40)
    hill = Hill(alpha, gamma)
    rate = 10.0**power
    curves = []
    for series in (spend, rate * spend):
        adstocked = carry_over(series, adstock)
        curves.append(saturate(adstocked, hill, half_saturation(hill, adstocked)))
    np.testing.assert_allclose(curves[1], curves[0], rtol=1e-12, atol=1e-15)


@settings(max_examples=200, deadline=None)
@given(adstocks, st.integers(1, 40), st.integers(0, 2**32 - 1))
def test_a_kernel_in_periods_moves_no_earlier_value_when_the_series_grows(
    adstock: Geometric | WeibullCdf, cut: int, seed: int
) -> None:
    spend = np.random.default_rng(seed).gamma(2.0, 100.0, 40)
    np.testing.assert_allclose(
        carry_over(spend[:cut], adstock), carry_over(spend, adstock)[:cut], rtol=1e-13
    )


def test_robyn_s_share_moves_earlier_values_when_the_series_grows() -> None:
    spend = np.array(FIXTURE["series"]["long"])
    shorter = carry_over(spend[:40], robyn_weibull(2.0, 0.1, 40))
    longer = carry_over(spend, robyn_weibull(2.0, 0.1, 60))[:40]
    assert np.abs(shorter - longer).max() > 0.05 * np.abs(longer).max()


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: Geometric(1.0), "theta must be in"),
        (lambda: Geometric(-0.1), "theta must be in"),
        (lambda: WeibullCdf(0.0, 2.0), "positive and finite"),
        (lambda: WeibullCdf(1.0, np.inf), "positive and finite"),
        (lambda: Hill(0.0, 0.5), "alpha must be positive"),
        (lambda: Hill(1.0, 1.5), "gamma in"),
        (lambda: robyn_weibull(-1.0, 0.1, 10), "need shape >= 0"),
        (lambda: robyn_weibull(1.0, 1.5, 10), "need shape >= 0"),
        (lambda: robyn_weibull(1.0, 0.1, 0), "need shape >= 0"),
        (lambda: carry_over([1.0, -2.0], Geometric(0.5)), "nonnegative"),
        (lambda: carry_over([1.0, np.nan], Geometric(0.5)), "finite"),
        (lambda: carry_over(np.ones((3, 2)), Geometric(0.5)), r"\(T,\) series"),
        (lambda: carry_over([], Geometric(0.5)), "nonempty"),
        (lambda: half_saturation(Hill(1.0, 0.5), [0.0, 0.0]), "half-saturation point is 0"),
        (lambda: Hill(1.0, 0.0), "gamma in"),
    ],
)
def test_what_it_cannot_transform_is_refused(make, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        make()
