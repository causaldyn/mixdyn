"""Effects held to a sign: the engine's posterior truncated to it.

The world is 156 weeks of `y = 100 - 0.5 x + N(0, 5^2)` with `x = max(0, N(50, 20))`, so the data
pull against a positive sign on x. Fitted without the sign, the base sits about 99 beside the
negative effect, where the KPI averages 76. The posterior truncated to the sign moves what the data
pull against it into the base.
"""

from __future__ import annotations

import functools
import math
from typing import Any

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
    decompose_dlm,
    diagnose_dlm,
    dlm_choice,
    fit_dlm,
    forecast_dlm,
    forecast_draws,
)
from mixdyn.models.dlm import _carried, _planned
from mixdyn.models.dlm_choice import _kept_fit

DATES = np.datetime64("2023-01-02") + np.timedelta64(7, "D") * np.arange(156)
NOISE = 5.0
_RNG = np.random.default_rng(5)
X = np.maximum(0.0, _RNG.normal(50.0, 20.0, DATES.size))
KPI = 100.0 - 0.5 * X + _RNG.normal(0.0, NOISE, DATES.size)


def world(kpi: np.ndarray, **columns: np.ndarray) -> MediaPanel:
    return read_panel(
        {"date": DATES, "kpi": kpi, **columns},
        date="date",
        kpi="kpi",
        spend={name: name for name in columns},
        cadence=Cadence.WEEK,
        kpi_unit=Count("units"),
        currency="USD",
    )


def spec(*effects: Effect) -> DlmSpec:
    """A level at 0.95 and each channel at 0.95."""
    return DlmSpec(effects, trend=Trend(1, 0.95))


def mean_base(fit: DlmFit) -> float:
    assert fit.base is not None
    return float(fit.base.mean())


@functools.cache
def fit(*effects: Effect, draws: int = 1000) -> DlmFit:
    return fit_dlm(world(KPI, x=X), NATIONAL, spec(*effects), draws=draws)


HELD = Effect("x", 0.95, sign="positive")


def test_a_sign_the_data_contradict_moves_the_channel_into_the_base() -> None:
    """The free fit's base averages 99.0 beside the effect of -0.5, and a fit without the channel
    averages the KPI, 76.2. Held positive, the base lands within a fifth of that distance of the
    channel-less one, at 72.0; the free fit's is the whole distance away."""
    target = mean_base(fit())
    reach = 0.25 * abs(mean_base(fit(Effect("x", 0.95))) - target)
    assert abs(mean_base(fit(HELD)) - target) < reach


def test_a_signed_fit_still_adds_up_to_the_kpi() -> None:
    """A fit's parts add up to its estimate, which averages the KPI: the signed fit misses that
    average by 0.03. Zeroing the wrong-signed contribution after a free fit instead leaves the
    free fit's base, which misses it by 22.8, nearly the channel's whole average effect, 0.5 times
    x's mean of 48.0."""
    assert abs(np.mean(fit(HELD).estimated - KPI)) < 0.1 * NOISE
    assert abs(np.mean(mean_base(fit(Effect("x", 0.95))) - KPI)) > 4.0 * NOISE


def test_every_draw_holds_the_sign_and_the_point_decomposition_is_their_mean() -> None:
    held = fit(HELD)
    assert held.signed is not None
    assert held.signed.mixed
    assert not [f for f in diagnose_dlm(held).findings if f.code is CheckCode.UNMIXED]
    every = held.signed.pooled.states.shape[0]
    draws = decompose_dlm(held, draws=every, seed=3).draws
    assert draws.contributions["x"].min() >= 0.0
    np.testing.assert_allclose(draws.contributions["x"].mean(axis=0), held.contributions["x"])
    assert draws.base is not None
    assert held.base is not None
    np.testing.assert_allclose(draws.base.mean(axis=0), held.base)


def test_only_the_signed_effect_is_held_to_its_sign() -> None:
    """Two channels whose effects are both -0.5, the second held positive: its contribution stays
    at or above zero, and the free one keeps its pull, near -0.5 times x's mean of 48.0."""
    other = X[::-1].copy()
    kpi = 100.0 - 0.5 * X - 0.5 * other + np.random.default_rng(7).normal(0.0, NOISE, X.size)
    effects = (Effect("x", 0.95), Effect("y", 0.95, sign="positive"))
    two = fit_dlm(world(kpi, x=X, y=other), NATIONAL, spec(*effects), draws=250, seed=1)
    assert two.contributions["y"].min() >= 0.0
    assert two.contributions["x"].mean() < -0.25 * X.mean()


def test_chains_that_have_not_mixed_are_a_finding_on_every_date() -> None:
    short = fit(HELD, draws=10)
    assert short.signed is not None
    assert not short.signed.mixed
    (finding,) = [f for f in diagnose_dlm(short).findings if f.code is CheckCode.UNMIXED]
    assert np.array_equal(finding.dates, short.dates)
    assert "have not mixed" in finding.message


def test_a_kept_spec_s_chains_are_drawn_again_at_twice_the_draws_until_they_mix() -> None:
    """Chains that mix are kept as drawn. Ten draws a chain cannot mix, since 400 effective draws
    are asked of 40, so the kept fit is drawn at 10, 20 and 40 a chain and keeps the last; a spec
    with no sign draws no chains."""
    data = world(KPI, x=X)
    mixed = _kept_fit(data, NATIONAL, spec(HELD))
    assert mixed.signed is not None
    assert mixed.signed.mixed
    assert mixed.signed.states.shape[:2] == (4, 1000)
    short = _kept_fit(data, NATIONAL, spec(HELD), draws=10)
    assert short.signed is not None
    assert not short.signed.mixed
    assert short.signed.states.shape[:2] == (4, 40)
    assert _kept_fit(data, NATIONAL, spec(Effect("x", 0.95)), draws=10).signed is None


def test_a_kept_spec_whose_sampler_stops_is_drawn_from_the_next_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CHC's exact trajectory stops with a `RuntimeError` where it meets more walls than its cap,
    which depends on where the chains start. A sampler that stops from seeds 0 and 1 is drawn from
    seed 2, and one that stops from every seed raises. The stop is put in place at `fit_dlm`: on
    the full grid's world it came from one start of one spec in 1 964, which no small world
    reproduces."""
    data = world(KPI, x=X)
    tried: list[int] = []

    def stopping(*args: Any, seed: int, **kwargs: Any) -> DlmFit:
        tried.append(seed)
        if seed < 2:
            raise RuntimeError(f"an exact Hamiltonian trajectory met many walls from seed {seed}")
        return fit_dlm(*args, seed=seed, **kwargs)

    monkeypatch.setattr(dlm_choice, "fit_dlm", stopping)
    drawn = _kept_fit(data, NATIONAL, spec(HELD))
    assert tried == [0, 1, 2]
    own = fit_dlm(data, NATIONAL, spec(HELD), seed=2)
    assert drawn.signed is not None
    assert own.signed is not None
    assert np.array_equal(drawn.signed.states, own.signed.states)
    tried.clear()

    def stopped(*args: Any, seed: int, **kwargs: Any) -> DlmFit:
        tried.append(seed)
        raise RuntimeError(f"an exact Hamiltonian trajectory met many walls from seed {seed}")

    monkeypatch.setattr(dlm_choice, "fit_dlm", stopped)
    with pytest.raises(RuntimeError, match="from seed 3"):
        _kept_fit(data, NATIONAL, spec(HELD))
    assert tried == [0, 1, 2, 3]


def test_a_signed_fit_s_forecast_is_drawn_not_student_t() -> None:
    with pytest.raises(ValueError, match="signed effects is drawn, not Student t"):
        forecast_dlm(fit(HELD, draws=10), 4, {"x": np.full(4, 50.0)})


def test_a_signed_fit_s_forecast_is_its_posterior_with_the_sign_held_a_week_on() -> None:
    """A week on, x planned at 1 000 so that its coefficient carries the forecast, against the
    model's law with the sign held over that week too: each of the fit's draws weighed by the
    chance its coefficient keeps the sign, `Phi(theta / sqrt(V w))`, the coefficient then a normal
    truncated to the sign, integrated over 400 midpoints of its quantiles, and the level and the
    noise normal about it. At the 99 percentiles the 40 000 draws keep within 0.004 of it, inside
    the 99 % Kolmogorov-Smirnov bound of 0.008; every draw weighed alike, as carrying each one
    until its coefficient keeps the sign would, lies 0.028 away."""
    signed = fit(HELD)
    assert signed.signed is not None
    planned = 1000.0
    drawn = forecast_draws(signed, 1, {"x": [planned]}, 40_000, seed=7)
    y = drawn.kpi[:, 0]
    filtered = signed.filtered
    # the evolution variances in units of the variance: a level and a coefficient, each a block
    # of its own at a discount of 0.95, which the planned spend keeps active
    w = (1.0 / 0.95 - 1.0) * np.diag(filtered.covariance[-1]) / filtered.scale[-1]
    level = signed.signed.states[:, :, -1, 0].reshape(-1)
    theta = signed.signed.states[:, :, -1, 1].reshape(-1)
    variance = signed.signed.variance.reshape(-1)
    step = np.sqrt(variance * w[1])
    kept = stats.norm.cdf(theta / step)
    below = stats.norm.cdf(-theta / step)
    u = (np.arange(400) + 0.5) / 400.0
    truncated = theta[:, None] + step[:, None] * stats.norm.ppf(
        below[:, None] + u[None] * (1.0 - below[:, None])
    )
    sd = np.sqrt(variance * (w[0] + 1.0))
    at = np.quantile(y, np.linspace(0.01, 0.99, 99))
    each = np.array(
        [
            stats.norm.cdf((c - level[:, None] - planned * truncated) / sd[:, None]).mean(axis=1)
            for c in at
        ]
    )
    exact = each @ kept / kept.sum()
    alike = each.mean(axis=1)
    seen = np.searchsorted(np.sort(y), at, side="right") / y.size
    bound = 1.63 / math.sqrt(y.size)
    assert np.abs(seen - exact).max() < bound
    assert np.abs(alike - exact).max() > 2.0 * bound
    assert drawn.acceptance == pytest.approx(kept.mean(), abs=4.0 * math.sqrt(0.25 / 8192))


def test_a_signed_fit_s_forecast_keeps_the_sign_at_every_week() -> None:
    """Over 13 weeks the share of paths drawn that keep the sign at every week, 0.64 here, is at
    least `C(26, 13) / 4^13`, 0.155, the chance a walk started at 0 keeps a sign so long (Sparre
    Andersen), and below 1; every path kept keeps it."""
    signed = fit(HELD)
    states, _, held, tried = _carried(
        signed, _planned(signed, 13, {"x": np.full(13, 400.0)}), 4000, np.random.default_rng(3)
    )
    assert states.shape == (4000, 13, 2)
    assert states[:, :, 1].min() >= 0.0
    assert math.comb(26, 13) / 4**13 <= held / tried < 1.0


def test_a_decomposition_asking_for_more_draws_than_the_fit_holds_is_refused() -> None:
    short = fit(HELD, draws=10)
    with pytest.raises(ValueError, match="from 1 to the 40 the signed fit holds, got 41"):
        decompose_dlm(short, draws=41, seed=0)


def test_a_sign_that_is_neither_is_refused() -> None:
    sign: Any = "up"
    with pytest.raises(ValueError, match="a sign is 'positive' or 'negative', got 'up'"):
        Effect("x", 0.95, sign=sign)


@pytest.mark.parametrize(
    "options",
    [{"form": "multiplicative"}, {"variance_discount": 0.99}],
    ids=["multiplicative", "variance discounted"],
)
def test_a_signed_effect_in_a_model_the_sampler_cannot_draw_is_refused(
    options: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match=r"signed effects \['x'\] need form='additive'"):
        DlmSpec((HELD,), trend=Trend(1, 0.95), **options)


def test_a_signed_fit_s_draws_and_its_decomposition_s_come_from_their_seeds() -> None:
    def signed(seed: int) -> DlmFit:
        return fit_dlm(world(KPI, x=X), NATIONAL, spec(HELD), draws=10, seed=seed)

    def states(seed: int) -> np.ndarray:
        draws = signed(seed).signed
        assert draws is not None
        return draws.states

    assert np.array_equal(states(4), states(4))
    assert not np.array_equal(states(4), states(5))
    short = signed(4)

    def contributions(seed: int) -> np.ndarray:
        return decompose_dlm(short, draws=20, seed=seed).draws.contributions["x"]

    assert np.array_equal(contributions(1), contributions(1))
    assert not np.array_equal(contributions(1), contributions(2))
