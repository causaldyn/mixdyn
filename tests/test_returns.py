"""A channel's returns, one definition each, read off `chc.response`: the revenue its spend earns
per currency unit with all of its carryover, the marginal and steady-state returns on that basis, a
window's contribution, and the share of a unit's adstock that is carried over."""

from __future__ import annotations

import functools
import math
from typing import Any

import numpy as np
import pytest
from chc import response

from mixdyn import NATIONAL, Cadence, Channel, MediaPanel, Revenue
from mixdyn.decisions import Returns, portfolio_roi, returns
from mixdyn.models import DlmSpec, Effect, Trend, decompose_dlm, fit_dlm
from mixdyn.models.bayes_mmm import BayesFit, BayesMedia, BayesSpec, fit_bayes

LONG = 200
"""A kernel long enough that a retention of 0.8 leaves `0.8^200`, 4e-20, of a unit uncounted."""
NONE = response.GeometricAdstock(0.0, length=1, normalized=False)
"""No carryover."""
LINEAR = response.Power(1.0, 1.0)


def geometric(retention: float) -> response.GeometricAdstock:
    """A geometric adstock, `a_t = x_t + r a_{t - 1}`, unnormalised."""
    return response.GeometricAdstock(retention, length=LONG, normalized=False)


def test_a_linear_channel_returns_its_coefficient_times_its_kernel_s_sum() -> None:
    """2 a unit through a geometric kernel of 0.5: all of a window's spend's carryover counted,
    each unit returns `2 / (1 - 0.5)`, at the margin, on average and at the steady state alike, and
    half of its adstock lands after the period it is spent in."""
    spend = np.random.default_rng(1).uniform(50.0, 150.0, 52)
    for window in (slice(None), slice(10, 30), slice(-13, None)):
        ours = returns(response.Channel(geometric(0.5), LINEAR, 2.0), spend, window)
        assert ours.spend == pytest.approx(spend[window].sum(), rel=1e-15)
        assert ours.roi == pytest.approx(4.0, rel=1e-12)
        assert ours.marginal_roi == pytest.approx(4.0, rel=1e-12)
        assert ours.steady_state_marginal_roi == pytest.approx(4.0, rel=1e-12)
        assert ours.carryover == pytest.approx(0.5, rel=1e-12, abs=0.0)


def test_a_window_s_contribution_is_what_the_decomposition_puts_in_its_periods() -> None:
    """The carryover into the window from the spend before it is counted, the carryover out of it
    is not: the contribution over a window and over the periods after it add up to the series'."""
    spend = np.random.default_rng(2).uniform(0.0, 100.0, 40)
    channel = response.Channel(geometric(0.7), response.Hill(80.0, 1.5), 3.0)
    whole = returns(channel, spend).contribution
    parts = returns(channel, spend, slice(0, 25)).contribution
    parts += returns(channel, spend, slice(25, None)).contribution
    assert parts == pytest.approx(whole, rel=1e-13)
    expected = 3.0 * np.sum(channel.curve(np.convolve(spend, 0.7 ** np.arange(LONG))[:40][25:]))
    assert returns(channel, spend, slice(25, None)).contribution == pytest.approx(
        expected, rel=1e-12
    )


def test_the_steady_state_is_held_at_the_window_s_mean_spend() -> None:
    """A Hill of half point 80 and slope 1 behind a geometric kernel of 0.5, whose sum is 2: one
    more unit a period, once the window's mean spend `s` has been held, returns `3 h'(2 s) 2`."""
    spend = np.r_[np.full(20, 10.0), np.full(20, 90.0)]
    channel = response.Channel(geometric(0.5), response.Hill(80.0, 1.0), 3.0)
    for window, level in ((slice(None, 20), 10.0), (slice(20, None), 90.0), (slice(None), 50.0)):
        expected = 3.0 * 80.0 / (2.0 * level + 80.0) ** 2 * 2.0
        assert returns(channel, spend, window).steady_state_marginal_roi == pytest.approx(
            expected, rel=1e-12, abs=0.0
        )


@pytest.mark.parametrize("retention", [0.0, 0.3, 0.8])
def test_the_carryover_is_the_share_of_a_unit_s_adstock_after_its_period(retention: float) -> None:
    """A geometric kernel puts `1 - r` of a unit's adstock in its own period and carries `r`."""
    channel = response.Channel(geometric(retention), response.Hill(50.0, 1.0), 1.0)
    assert returns(channel, np.full(10, 20.0)).carryover == pytest.approx(retention, abs=1e-12)


def test_returns_scale_with_revenue_per_kpi_and_not_with_the_currency() -> None:
    """Revenue per KPI scales every return and the contribution, and none of the shares; spend
    counted in cents, the curve's scale with it, returns a hundredth as much a cent."""
    spend = np.random.default_rng(3).uniform(10.0, 90.0, 30)
    channel = response.Channel(geometric(0.6), response.Hill(40.0, 2.0), 5.0)
    base = returns(channel, spend)
    scaled = returns(channel, spend, revenue_per_kpi=50.0)
    for name in ("contribution", "roi", "marginal_roi", "steady_state_marginal_roi"):
        assert getattr(scaled, name) == pytest.approx(
            50.0 * getattr(base, name), rel=1e-13, abs=0.0
        )
    assert scaled.carryover == base.carryover
    cents = returns(response.Channel(geometric(0.6), response.Hill(4000.0, 2.0), 5.0), 100 * spend)
    assert cents.roi == pytest.approx(base.roi / 100.0, rel=1e-12, abs=0.0)
    assert cents.contribution == pytest.approx(base.contribution, rel=1e-12)


@pytest.mark.parametrize(
    ("spend", "kwargs", "match"),
    [
        (np.ones((2, 5)), {}, r"non-empty vector, got shape \(2, 5\)"),
        (np.array([]), {}, "non-empty vector"),
        (np.array([1.0, np.nan]), {}, "finite and at or above 0"),
        (np.array([1.0, np.inf]), {}, "finite and at or above 0"),
        (np.array([1.0, -1.0]), {}, "finite and at or above 0"),
        (np.array([1.0, 0.0, 0.0]), {"window": slice(1, None)}, r"spends 0\.0"),
        (np.ones(3), {"revenue_per_kpi": 0.0}, "positive and finite, got 0.0"),
        (np.ones(3), {"revenue_per_kpi": math.inf}, "positive and finite"),
    ],
)
def test_a_return_that_cannot_be_is_refused(
    spend: np.ndarray, kwargs: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        returns(response.Channel(NONE, LINEAR, 1.0), spend, **kwargs)


def test_a_portfolio_returns_its_channels_revenue_over_their_spend() -> None:
    channels = [Returns(100.0, 0.0, 2.0, 0.0, 0.0, 0.0), Returns(300.0, 0.0, 1.0, 0.0, 0.0, 0.0)]
    assert portfolio_roi(channels) == pytest.approx(500.0 / 400.0, rel=1e-15, abs=0.0)
    with pytest.raises(ValueError, match=r"spend 0\.0"):
        portfolio_roi([Returns(0.0, 0.0, 2.0, 0.0, 0.0, 0.0)])


def test_a_return_is_on_the_spend() -> None:
    """100 a week for 52 weeks, `y = 1000 + 2 adstock(x, r)`. The return reads the spend, 5 200,
    not the adstock it builds, 10 200 at a retention of 0.5 and 24 000 at 0.8; the contribution
    over the 52 weeks is 3.92 and 9.23 a unit spent, and with all of the carryover counted each
    unit returns `2 / (1 - r)`, 4 and 10."""
    x = np.full(52, 100.0)
    for rate in (0.5, 0.8):
        channel = response.Channel(geometric(rate), LINEAR, 2.0)
        ours = returns(channel, x)
        stock = np.convolve(x, rate ** np.arange(x.size))[: x.size]
        assert ours.spend == 5200.0
        assert ours.contribution == pytest.approx(2.0 * stock.sum(), rel=1e-12)
        assert ours.roi == pytest.approx(2.0 / (1.0 - rate), rel=1e-12)
        assert stock.sum() > 1.9 * ours.spend


def test_the_marginal_return_is_on_the_curve() -> None:
    """A channel `2.5 s(x)`, `s` a Hill of slope 1 whose half point is 150, over weeks of 100, 200
    and 300: its return is 0.00683, and its marginal return 0.00295,
    `2.5 sum K x / (x + K)^2 / sum x`. The coefficient, 2.5, is neither, nor in their units."""
    x = np.array([100.0, 200.0, 300.0])
    ours = returns(response.Channel(NONE, response.Hill(150.0, 1.0), 2.5), x)
    roas = 2.5 * np.sum(x / (x + 150.0)) / x.sum()
    marginal = 2.5 * np.sum(150.0 * x / (x + 150.0) ** 2) / x.sum()
    assert ours.roi == pytest.approx(roas, rel=1e-12, abs=0.0)
    assert ours.marginal_roi == pytest.approx(marginal, rel=1e-12, abs=0.0)
    assert 800.0 * ours.marginal_roi < 2.5


def test_the_steady_state_counts_the_carryover() -> None:
    """A Hill of ceiling 1, half point 100 and slope 1 behind a geometric adstock of 0.8: held at
    100 a week, one more unit a week returns `h'(500) 5`, 0.001389. Differencing the curve at one
    period's spend, where the adstock would return its input, reads 0.00250 and drops the
    carryover. The marginal at 100 over the one at 1 is 3.06 %."""
    channel = response.Channel(geometric(0.8), response.Hill(100.0, 1.0), 1.0)
    at, floor = (
        returns(channel, np.full(52, level)).steady_state_marginal_roi for level in (100.0, 1.0)
    )
    assert at == pytest.approx(5.0 * 100.0 / 600.0**2, rel=1e-12, abs=0.0)
    assert 1.7 * at < 100.0 / 200.0**2
    assert 100.0 * at / floor == pytest.approx(3.0625, abs=1e-4)


def test_a_linear_channel_s_margin_is_its_average() -> None:
    """A linear channel, `y = 2x`: every unit returns 2, at the margin as on average, so its return
    and marginal return are both 2."""
    x = np.linspace(100.0, 200.0, 52)
    ours = returns(response.Channel(NONE, LINEAR, 2.0), x)
    assert ours.roi == pytest.approx(2.0, rel=1e-12)
    assert ours.marginal_roi == pytest.approx(ours.roi, rel=1e-12)


@functools.cache
def _rising() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The engine's draws of a window's return per unit spent, on a world whose coefficient rises
    from 1 to 3 over 104 weeks of spend that varies, and its truth and spend."""
    rng = np.random.default_rng(7)
    weeks = 104
    x = rng.uniform(50.0, 150.0, weeks)
    beta = np.linspace(1.0, 3.0, weeks)
    kpi = 500.0 + beta * x + rng.normal(0.0, 5.0, weeks)
    dates = np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(weeks)
    panel = MediaPanel(
        dates=dates,
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=(Channel("x"),),
        controls=(),
        kpi=kpi[None],
        kpi_unit=Revenue(),
        spend=x[None, :, None],
        currency="EUR",
        control_values=np.zeros((1, weeks, 0)),
        filled=np.zeros((1, weeks), dtype=bool),
    )
    fit = fit_dlm(panel, NATIONAL, DlmSpec((Effect("x", 0.98),), trend=Trend(1, 0.98)))
    window = decompose_dlm(fit, draws=1000, seed=7).window(dates[0], dates[-1])
    return window.contributions["x"] / x.sum(), beta, x


def test_a_window_s_interval_holds_its_point() -> None:
    """The coefficient rises from 1 to 3 over 104 weeks of spend that varies. The interval of the
    window's return is the window's draws': it holds its point and the world's return. The last
    week's coefficient interval, near 3, would hold neither."""
    draws, truth, spend = _rising()
    low, high = np.quantile(draws, [0.05, 0.95])
    assert low < draws.mean() < high
    assert low < truth @ spend / spend.sum() < high
    assert high < truth[-1]


def test_a_portfolio_s_return_is_its_revenue_over_its_spend() -> None:
    """Six channels, each 100 of revenue on 100 of spend, return 1. Two channels at returns of 2
    and 1.5 on 1 000 and 3 000 of spend return 1.625 between them."""
    six = [Returns(100.0, 100.0, 1.0, 0.0, 0.0, 0.0)] * 6
    assert portfolio_roi(six) == pytest.approx(1.0, rel=1e-15, abs=0.0)
    two = [Returns(1000.0, 0.0, 2.0, 0.0, 0.0, 0.0), Returns(3000.0, 0.0, 1.5, 0.0, 0.0, 0.0)]
    assert portfolio_roi(two) == pytest.approx(1.625, rel=1e-15, abs=0.0)


def test_the_carryover_share_is_the_kernel_s() -> None:
    """At a geometric retention of 0.5 each unit's adstock lands half in its own period and half
    after, so the carryover share is 0.5."""
    channel = response.Channel(geometric(0.5), LINEAR, 2.0)
    assert returns(channel, np.full(52, 100.0)).carryover == pytest.approx(0.5, rel=1e-12, abs=0.0)


@functools.cache
def _decay(carried: bool) -> tuple[np.ndarray, BayesFit]:
    """78 weeks of flighted spend and a fit of its channel's decay: the KPI carries the channel
    through a retention of 0.6 into a Hill of half point 100 and coefficient 300, or does not
    carry it at all."""
    rng = np.random.default_rng(11)
    weeks = 78
    x = np.where(np.arange(weeks) % 6 < 2, rng.uniform(150.0, 250.0, weeks), 0.0)
    effect = 0.0
    if carried:
        stock = np.convolve(x, 0.6 ** np.arange(weeks))[:weeks] * 0.4
        effect = 300.0 * stock / (stock + 100.0)
    kpi = 1000.0 + effect + rng.normal(0.0, 10.0, weeks)
    panel = MediaPanel(
        dates=np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(weeks),
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=(Channel("tv"),),
        controls=(),
        kpi=kpi[None],
        kpi_unit=Revenue(),
        spend=x[None, :, None],
        currency="EUR",
        control_values=np.zeros((1, weeks, 0)),
        filled=np.zeros((1, weeks), dtype=bool),
    )
    spec = BayesSpec((BayesMedia("tv"),))
    return x, fit_bayes(panel, NATIONAL, spec, chains=2, warmup=300, draws=300, seed=11)


def test_the_decay_is_as_identified_as_the_data_make_it() -> None:
    """The posterior on the decay narrows to a sixth of the prior's spread where the KPI carries
    the channel through a retention of 0.6, and keeps most of it where the KPI does not carry the
    channel."""
    prior = math.sqrt(3.0 / (4.0**2 * 5.0))
    carried, absent = (_decay(c)[1].draws["retention"][..., 0] for c in (True, False))
    assert carried.std() < prior / 6.0
    assert abs(np.median(carried) - 0.6) < 0.1
    assert absent.std() > prior / 2.0


def test_a_bayesian_fit_s_returns_at_each_draw_hold_the_world_s() -> None:
    """A return's posterior is `returns` at each draw of the fit (`BayesFit.channel_draws`). Where
    the KPI carries the channel through a retention of 0.6 into a Hill, the 90 % intervals of the
    window's return and of its marginal return, each with all of its carryover, hold the world's;
    and at every draw the model's return parameter, the window's contribution over its spend,
    sits below the return by the carryover past the window's end."""
    x, fit = _decay(True)
    kernel = response.GeometricAdstock(0.6, length=x.size, normalized=True)
    world = response.Channel(kernel, response.Hill(100.0, 1.0), 300.0)
    truth = returns(world, x)
    drawn = [returns(channels["tv"], x) for channels in fit.channel_draws(200)]
    for name in ("roi", "marginal_roi"):
        low, high = np.quantile([getattr(r, name) for r in drawn], [0.05, 0.95])
        assert low < getattr(truth, name) < high, name
    rows = np.linspace(0, fit.draws["roi"].size - 1, 200).round().astype(int)
    parameter = fit.draws["roi"].reshape(-1)[rows]
    assert np.all(parameter < np.array([r.roi for r in drawn]))
