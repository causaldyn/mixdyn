"""Leave-one-out cross-validation and stacking: the smoothing is arviz-stats' on the cases
arviz-stats 1.3.3 recorded, the generalised Pareto's fit recovers its shape and scale,
PSIS-LOO recovers exact leave-one-out on a conjugate model, its diagnostic flags the observation
the posterior leans on, a precise test's bounded ratios pass that diagnostic while Kish's
effective sample collapses as Maxima's law says, and stacking finds the weights a closed form
gives."""

from __future__ import annotations

import json
import math
import sys
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from scipy import special, stats

from mixdyn.inference.diagnostics import _ess, _split
from mixdyn.inference.loo import Loo, _generalised_pareto, _log_quantile, loo, psis, stacking

ORACLES = Path(__file__).parent / "fixtures" / "oracles"
ARVIZ = json.loads((ORACLES / "psis_arviz.json").read_text())
ESS_LAW = json.loads((ORACLES / "voi_conjugate.json").read_text())["m3"]["points"]
BASE = np.random.default_rng(11).normal(0.0, 1.0, 29)


@pytest.mark.parametrize("name", sorted(ARVIZ["cases"]))
def test_the_smoothing_is_arviz_stats(name: str) -> None:
    """The normal mean model's log importance ratios at the recorded draws: each observation's
    fitted shape and the log predictive density its smoothed weights give, shapes from below 0 to
    above 1, tails of a fifth of the draws and of ``3 sqrt(S / r_eff)``."""
    case = ARVIZ["cases"][name]
    y, mu = np.array(case["y"]), np.array(case["mu"])
    ll = -0.5 * (y - mu[:, None]) ** 2 - 0.5 * math.log(2.0 * math.pi)
    for i in range(y.size):
        smoothed = psis(-ll[:, i], efficiency=case["efficiency"])
        assert smoothed.pareto_k == pytest.approx(case["pareto_k"][i], abs=1e-12)
        elpd = special.logsumexp(smoothed.log_weights + ll[:, i])
        assert elpd == pytest.approx(case["elpd"][i], abs=1e-12)


@pytest.mark.parametrize("shape", [-0.3, 0.0, 0.5, 0.9])
def test_the_generalised_pareto_fit_recovers_its_shape_and_scale(shape: float) -> None:
    """From 50 000 draws, within 4.5 standard errors of the maximum likelihood estimate's,
    ``(1 + k) / sqrt(n)`` for the shape and ``sqrt(2 (1 + k) / n)`` for the scale's ratio, the
    shape shrunk toward 1/2 as by ten prior observations."""
    n = 50_000
    x = np.sort(stats.genpareto(c=shape, scale=1.0).rvs(n, random_state=7))
    k, sigma = _generalised_pareto(x)
    assert abs(k - (n * shape + 5.0) / (n + 10.0)) <= 4.5 * (1.0 + shape) / math.sqrt(n)
    assert abs(sigma - 1.0) <= 4.5 * math.sqrt(2.0 * (1.0 + shape) / n)


@pytest.mark.parametrize("shape", [-0.4, 0.0, 0.3, 2.0])
def test_the_tails_log_quantiles_are_the_generalised_paretos(shape: float) -> None:
    """At a tail of 40's plotting positions, the exponential's at a shape of 0."""
    p = (np.arange(1, 41) - 0.5) / 40
    quantile = stats.genpareto.ppf(p, shape, scale=1.7)
    np.testing.assert_allclose(_log_quantile(p, shape, 1.7), np.log(quantile), rtol=0.0, atol=4e-15)


def _conjugate(y: np.ndarray, seed: int) -> tuple[Loo, np.ndarray, np.ndarray]:
    """``y_i ~ N(mu, 1)``, ``mu ~ N(0, 10^2)``: PSIS-LOO on 4000 draws of the posterior, and each
    observation's exact log predictive density left out and in sample."""
    n = y.size
    precision = 0.01 + n
    mu = np.random.default_rng(seed).normal(y.sum() / precision, 1.0 / math.sqrt(precision), 4000)
    result = loo(stats.norm.logpdf(y, mu[:, None], 1.0).reshape(4, 1000, n))
    left = precision - 1.0
    exact = stats.norm.logpdf(y, (y.sum() - y) / left, np.sqrt(1.0 + 1.0 / left))
    in_sample = stats.norm.logpdf(y, y.sum() / precision, math.sqrt(1.0 + 1.0 / precision))
    return result, exact, in_sample


def test_psis_loo_is_exact_leave_one_out_on_a_conjugate_model() -> None:
    """Twenty-nine standard normal draws and one at 4: each observation's log predictive density
    left out within 0.03 of the exact one, their sum within 0.05, and so the effective number of
    parameters; every shape below 0.5."""
    result, exact, in_sample = _conjugate(np.r_[BASE, 4.0], seed=0)
    np.testing.assert_allclose(result.pointwise, exact, atol=0.03)
    assert result.elpd == pytest.approx(exact.sum(), abs=0.05)
    assert result.p_loo == pytest.approx((in_sample - exact).sum(), abs=0.05)
    assert result.se == pytest.approx(math.sqrt(30 * exact.var()), rel=0.05)
    assert np.all(result.pareto_k < 0.5)


def test_the_observation_the_posterior_leans_on_is_flagged() -> None:
    """One observation at 20 among standard normal draws: leaving it out moves the posterior too
    far for its draws to stand in, and its shape is above the threshold, which 4000 draws put at
    0.7; every other shape is below 0.5."""
    result, _, _ = _conjugate(np.r_[BASE, 20.0], seed=0)
    assert result.k_threshold == 0.7
    assert result.pareto_k[-1] > result.k_threshold
    assert np.all(result.pareto_k[:-1] < 0.5)


def test_the_draws_set_the_shape_to_trust() -> None:
    """``min(1 - 1 / log10 S, 0.7)``: 0.5 for 100 draws."""
    draws = np.random.default_rng(3).normal(0.0, 1.0, (4, 25, 3))
    assert loo(draws).k_threshold == pytest.approx(0.5)


def test_correlated_chains_smooth_a_longer_tail() -> None:
    """Draws that follow one another, ``mu`` an autoregression of coefficient 1/2 in each of four
    chains: each observation is smoothed with the relative efficiency of its likelihood over the
    split chains, below 1, and so over a longer tail than independent draws'."""
    rng = np.random.default_rng(5)
    mu = np.zeros((4, 1000))
    for t in range(1, 1000):
        mu[:, t] = 0.5 * mu[:, t - 1] + math.sqrt(0.75) * rng.standard_normal(4) / 5.5
    y = np.r_[BASE, 4.0]
    ll = stats.norm.logpdf(y, mu[..., None], 1.0)
    result = loo(ll)
    flat = ll.reshape(4000, y.size)
    for i in range(y.size):
        efficiency = _ess(_split(np.exp(ll[:, :, i] - ll[:, :, i].max()))) / 4000
        assert 0.2 < efficiency < 0.6
        smoothed = psis(-flat[:, i], efficiency=efficiency)
        assert result.pareto_k[i] == smoothed.pareto_k
        assert result.pointwise[i] == special.logsumexp(smoothed.log_weights + flat[:, i])


def test_an_observation_no_draw_tells_apart_is_its_own_density() -> None:
    """Its likelihood one value at every draw: the weights are equal, so its density left out is
    that value; with no tail to fit, it has no shape to trust."""
    ll = np.random.default_rng(9).normal(-1.0, 0.1, (4, 100, 3))
    ll[..., 1] = -2.5
    result = loo(ll)
    assert result.pointwise[1] == pytest.approx(-2.5, abs=1e-14)
    assert result.pareto_k[1] == math.inf
    assert np.all(np.isfinite(result.pareto_k[[0, 2]]))


def test_chains_too_short_for_an_effective_sample_are_read_as_independent_draws() -> None:
    """Four draws a chain split into halves of two, which give no effective sample size: each
    observation's draws are smoothed as independent ones."""
    ll = np.random.default_rng(17).normal(-1.0, 0.3, (2, 4, 3))
    result = loo(ll)
    flat = ll.reshape(8, 3)
    for i in range(3):
        smoothed = psis(-flat[:, i], efficiency=1.0)
        assert result.pointwise[i] == special.logsumexp(smoothed.log_weights + flat[:, i])


def test_a_tail_tied_at_its_cutoff_is_left_as_it_is() -> None:
    """100 ratios, a tail of 20: six of it equal to the 21st largest, so its first quartile is 0
    over it, and Zhang and Stephens' grid has no scale. The weights are the raw ratios', the
    shape infinite."""
    log_ratios = np.r_[np.linspace(-3.0, -1.2, 79), np.full(7, -1.0), np.linspace(-0.9, 0.0, 14)]
    smoothed = psis(log_ratios)
    assert smoothed.pareto_k == math.inf
    np.testing.assert_allclose(
        smoothed.log_weights, log_ratios - special.logsumexp(log_ratios), rtol=1e-14
    )


def test_a_tail_spanning_hundreds_of_orders_of_magnitude_is_smoothed_in_logs() -> None:
    """400 ratios whose tail of 60 spans 700 nats, its cutoff at 1600: the tail fits a shape in the
    hundreds, not to be trusted, at which ``expm1(k t)`` passes the floats for the largest
    quantiles. Each smoothed ratio is its quantile over the cutoff, read to 40 digits, and clipped
    at the largest raw ratio."""
    log_ratios = np.r_[np.full(340, -1600.0), np.linspace(-700.0, 0.0, 60)]
    smoothed = psis(log_ratios)
    k, sigma = _generalised_pareto(np.exp(log_ratios[340:]))
    assert smoothed.pareto_k == k > 100.0 > smoothed.k_threshold
    with localcontext() as context:
        context.prec = 40
        cutoff = Decimal(-1600).exp()
        tail = []
        for z in range(1, 61):
            t = -(1 - (z - Decimal("0.5")) / 60).ln()
            tail.append((Decimal(sigma) * ((Decimal(k) * t).exp() - 1) / Decimal(k) + cutoff).ln())
        assert Decimal(k) * -(1 - Decimal("59.5") / 60).ln() > Decimal(sys.float_info.max).ln()
        raw = [Decimal(-1600)] * 340 + [min(value, Decimal(0)) for value in tail]
        total = sum((value.exp() for value in raw), Decimal(0)).ln()
        expected = np.array([float(value - total) for value in raw])
    np.testing.assert_allclose(smoothed.log_weights, expected, rtol=0.0, atol=1e-12)


def test_a_tail_too_wide_for_the_fits_grid_is_left_as_it_is() -> None:
    """100 ratios whose tail of 20 spans 920 nats: its first quartile over the cutoff, near
    ``exp(-726)``, scales Zhang and Stephens' grid past the floats. The weights are the raw
    ratios', the shape infinite."""
    log_ratios = np.r_[np.full(80, -1840.0), np.linspace(-920.0, 0.0, 20)]
    smoothed = psis(log_ratios)
    assert smoothed.pareto_k == math.inf
    np.testing.assert_allclose(
        smoothed.log_weights, log_ratios - special.logsumexp(log_ratios), rtol=1e-14
    )


@pytest.mark.parametrize(
    "point",
    [p for p in ESS_LAW if p["sigma"] == "1/30" and p["y - m"] in ("0", "1")],
    ids=lambda p: f"y-m={p['y - m']}",
)
def test_a_precise_tests_ratios_pass_the_shape_while_the_effective_sample_collapses(
    point: dict,
) -> None:
    """A test of standard error ``sigma = tau / 30`` reads a parameter whose 40 000 draws are
    ``N(m, tau^2)``: the log ratios are its normal log likelihood, bounded above, so their tail
    fits a negative shape, well under the threshold, while Kish's effective sample is the share
    Maxima's law gives, ``sigma sqrt(sigma^2 + 2 tau^2) / (sigma^2 + tau^2)`` at ``y = m``, about
    ``sqrt(2) sigma / tau``, and less as ``y`` leaves ``m``: under 5 % of the draws, within 10 %
    of the law."""
    sigma, tau, gap = (float(Fraction(point[key])) for key in ("sigma", "tau", "y - m"))
    draws = 40_000
    mu = np.random.default_rng(13).normal(0.0, tau, draws)
    smoothed = psis(stats.norm.logpdf(gap, mu, sigma))
    assert smoothed.pareto_k < 0.0 < smoothed.k_threshold
    share = point["ess share"]["value"]
    assert share < 0.05
    assert smoothed.ess / draws == pytest.approx(share, rel=0.1)


def test_equal_ratios_are_as_many_effective_draws_as_their_efficiency_allows() -> None:
    """1000 equal ratios, read as draws of relative efficiency 0.36: equal weights, no tail to
    fit, and Kish's effective sample ``0.36 * 1000``; the threshold is ``1 - 1 / log10 S``."""
    smoothed = psis(np.zeros(1000), efficiency=0.36)
    np.testing.assert_allclose(np.exp(smoothed.log_weights), 1e-3, rtol=1e-14)
    assert smoothed.pareto_k == math.inf
    assert smoothed.ess == pytest.approx(360.0, rel=1e-12)
    assert smoothed.k_threshold == pytest.approx(2.0 / 3.0, rel=1e-15, abs=0.0)
    assert psis([0.5]).k_threshold == -math.inf


def test_stacking_finds_the_closed_form_weights() -> None:
    """Densities 2 and 1 at one observation, 1 and 3 at the other: the weight ``w`` on the first
    sets ``1 / (1 + w) = 2 / (3 - 2 w)``, so ``w = 1/4``, at any common scale of the densities,
    however small. A copy of the second shares its weight."""
    np.testing.assert_allclose(stacking(np.log([[2.0, 1.0], [1.0, 3.0]])), [0.25, 0.75])
    np.testing.assert_allclose(stacking(np.log([[2.0, 1.0], [1.0, 3.0]]) - 1000.0), [0.25, 0.75])
    np.testing.assert_allclose(
        stacking(np.log([[2.0, 1.0], [1.0, 3.0], [1.0, 3.0]])), [0.25, 0.375, 0.375]
    )


def test_a_model_better_at_every_observation_takes_every_weight() -> None:
    weights = stacking(np.log([[2.0, 1.0, 5.0], [1.0, 0.5, 4.0]]))
    assert weights[0] == pytest.approx(1.0, abs=1e-9)
    assert weights.sum() == pytest.approx(1.0, abs=1e-15)


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: loo(np.zeros((4, 1000))), "chains, draws"),
        (lambda: loo(np.zeros((4, 3, 10))), "chains, draws"),
        (lambda: loo(np.full((4, 100, 10), np.nan)), "not finite"),
        (lambda: stacking(np.zeros(3)), "models, observations"),
        (lambda: stacking(np.zeros((0, 3))), "models, observations"),
        (lambda: stacking(np.array([[0.0, -np.inf], [0.0, 0.0]])), "not finite"),
        (lambda: psis(np.zeros((2, 3))), "log ratios"),
        (lambda: psis(np.float64(0.5)), "log ratios"),
        (lambda: psis(np.zeros(0)), "log ratios"),
        (lambda: psis(np.array([0.0, np.inf])), "not finite"),
        (lambda: psis(np.zeros(10), efficiency=0.0), "efficiency"),
        (lambda: psis(np.zeros(10), efficiency=math.nan), "efficiency"),
        (lambda: psis(np.zeros(10), efficiency=math.inf), "efficiency"),
    ],
)
def test_what_cannot_be_read_is_refused(call, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()
