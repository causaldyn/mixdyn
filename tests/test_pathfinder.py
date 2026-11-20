"""Pathfinder: its inverse Hessian is the BFGS recursion's and its diagonal Gilbert and
Lemaréchal's, its draws carry their normal's log density, it keeps the approximation with the
largest evidence lower bound, a normal posterior is approximated closely, and draws resampled from
two paths take each mode in proportion to its mass."""

from __future__ import annotations

import logging
import math

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from mixdyn.inference.loo import psis
from mixdyn.inference.pathfinder import (
    Path,
    _diagonal,
    _draw,
    _inverse_hessian,
    _memory,
    pathfinder,
    resample,
)

RNG = np.random.default_rng(5)
A = RNG.normal(size=(5, 5))
COVARIANCE = A @ A.T / 5 + 0.2 * np.eye(5)
PRECISION = np.linalg.inv(COVARIANCE)
MEAN = np.array([1.0, -2.0, 0.5, 3.0, -1.0])
LOG_NORMALISER = -0.5 * (5 * math.log(2.0 * math.pi) + np.linalg.slogdet(COVARIANCE)[1])


def gaussian(q):
    """The normal with ``MEAN`` and ``COVARIANCE``, normalised: its ELBO is minus a KL."""
    r = q - MEAN
    return LOG_NORMALISER - 0.5 * r @ jnp.asarray(PRECISION) @ r


def two_modes(q):
    """0.8 N((-6, 0), I) + 0.2 N((6, 0), I)."""
    left = math.log(0.8) - 0.5 * ((q[0] + 6.0) ** 2 + q[1] ** 2)
    right = math.log(0.2) - 0.5 * ((q[0] - 6.0) ** 2 + q[1] ** 2)
    return jnp.logaddexp(left, right)


def pairs(count: int, d: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Steps and the gradient changes a convex quadratic gives them, so every curvature is
    positive."""
    b = rng.normal(size=(d, d))
    hessian = b @ b.T + d * np.eye(d)
    s = rng.normal(size=(count, d))
    return s, s @ hessian


def test_the_inverse_hessian_is_the_bfgs_recursion_s() -> None:
    """The compact form over ``diag(alpha)`` is the BFGS update applied pair by pair, oldest
    first, to ``1e-12`` of its largest entry."""
    rng = np.random.default_rng(1)
    for count, d in [(1, 4), (3, 4), (5, 9), (6, 3)]:
        s, y = pairs(count, d, rng)
        alpha = rng.uniform(0.2, 3.0, d)
        h = np.diag(alpha)
        for si, yi in zip(s, y, strict=True):
            rho = 1.0 / (si @ yi)
            v = np.eye(d) - rho * np.outer(yi, si)
            h = v.T @ h @ v + rho * np.outer(si, si)
        compact = _inverse_hessian(s, y, alpha)
        np.testing.assert_allclose(compact, h, rtol=0, atol=1e-12 * np.abs(h).max())


def test_the_diagonal_is_gilbert_and_lemarechal_s() -> None:
    """Eq. 4.9: the inverse of the diagonal of the BFGS update of the Hessian ``diag(1 / alpha)``
    first scaled by ``(y . alpha y) / (y . s)``, Oren and Spedicato's factor."""
    rng = np.random.default_rng(2)
    for _ in range(20):
        s, y = pairs(1, 6, rng)
        s, y = s[0], y[0]
        alpha = rng.uniform(0.2, 3.0, 6)
        b = np.diag((y @ (alpha * y)) / (y @ s) / alpha)
        updated = b - np.outer(b @ s, b @ s) / (s @ b @ s) + np.outer(y, y) / (y @ s)
        np.testing.assert_allclose(_diagonal(alpha, s, y), 1.0 / np.diag(updated), rtol=1e-12)


def test_the_memory_keeps_the_last_steps_that_pass_the_curvature_test() -> None:
    """Seven steps on a convex quadratic but the third, whose gradient change is reversed, and the
    fifth, whose curvature ``|y|^2 / (s . y)`` is ``1e13``: at each iterate the memory holds the
    last three steps that passed, oldest first, and the diagonal is updated by them alone."""
    rng = np.random.default_rng(4)
    s, y = pairs(7, 3, rng)
    y[2] = -y[2]
    s[4], y[4] = np.array([1e-7, 0.0, 0.0]), np.array([1e6, 0.0, 0.0])
    iterates = np.vstack([np.zeros(3), np.cumsum(s, axis=0)])
    gradients = np.vstack([np.ones(3), np.ones(3) + np.cumsum(y, axis=0)])
    s, y = np.diff(iterates, axis=0), np.diff(gradients, axis=0)
    kept: list[int] = []
    alpha = np.ones(3)
    for i, (a, steps, changes) in enumerate(_memory(list(iterates), list(gradients), 3)):
        if i not in (2, 4):
            kept = [*kept, i][-3:]
            alpha = _diagonal(alpha, s[i], y[i])
        np.testing.assert_array_equal(steps, s[kept].reshape(-1, 3))
        np.testing.assert_array_equal(changes, y[kept].reshape(-1, 3))
        np.testing.assert_array_equal(a, alpha)
    assert kept == [3, 5, 6]


def test_each_draw_carries_its_normal_s_log_density() -> None:
    rng = np.random.default_rng(3)
    factor = np.linalg.cholesky(COVARIANCE)
    draws, log_q = _draw(rng, MEAN, factor, 50)
    expected = stats.multivariate_normal(MEAN, COVARIANCE).logpdf(draws)
    np.testing.assert_allclose(log_q, expected, rtol=1e-12)


@pytest.fixture(scope="module")
def normal_path() -> Path:
    return pathfinder(gaussian, np.full((1, 5), 2.0), seed=7, draws=4000)[0]


def test_the_approximation_kept_has_the_largest_elbo(normal_path: Path) -> None:
    """Every iterate's estimate is recorded, and the draws are the largest one's: their own
    estimate is within 4.5 standard errors of its 25 draws' (the last iterate's is 0.35 below)."""
    assert normal_path.elbo == np.max(normal_path.elbos)
    assert normal_path.iteration == int(np.argmax(normal_path.elbos)) + 1
    assert np.sum(np.isfinite(normal_path.elbos)) > 1
    ratios = normal_path.log_p - normal_path.log_q
    se = ratios.std() * math.sqrt(1.0 / ratios.size + 1.0 / 25)
    assert abs(ratios.mean() - normal_path.elbo) <= 4.5 * se


def test_a_normal_posterior_is_approximated_as_closely_as_by_stan(normal_path: Path) -> None:
    """Against a normalised target the draws' mean log density less the approximation's is minus
    the approximation's divergence from it: under 0.1 nats from this start, where the median of
    Stan's single paths on this target from 100 starts is 0.11."""
    assert 0.0 < float(np.mean(normal_path.log_q - normal_path.log_p)) < 0.1
    assert np.all(np.isfinite(normal_path.log_p))


def test_an_isotropic_normal_is_found_exactly_from_one_step() -> None:
    """For ``N(mu, c I)`` one step's pair sets the diagonal to ``c`` and the BFGS update keeps it,
    and the normal at ``x - H g`` is the target: at every iterate the approximation is exact, its
    draws' log density the target's to rounding."""
    mu, c = np.array([1.0, -2.0, 0.5]), 2.5

    def target(q):
        r = q - mu
        return -0.5 * (r @ r) / c - 1.5 * math.log(2.0 * math.pi * c)

    path = pathfinder(target, np.full((1, 3), 3.0), seed=1, draws=200)[0]
    assert np.all(np.isfinite(path.elbos))
    np.testing.assert_allclose(path.elbos, 0.0, atol=1e-12)
    np.testing.assert_allclose(path.log_p - path.log_q, 0.0, atol=1e-12)


def test_a_draw_outside_the_support_has_log_density_minus_infinity() -> None:
    """A standard normal cut at ``q_0 = -2.5``, not a number beyond: the draws there carry minus
    infinity, and resampling never takes them."""

    def cut(q):
        return jnp.where(q[0] > -2.5, -0.5 * q @ q, jnp.nan)

    path = pathfinder(cut, np.full((1, 3), 0.5), seed=2, draws=4000)[0]
    outside = path.draws[:, 0] <= -2.5
    assert outside.sum() > 0
    assert np.all(path.log_p[outside] == -np.inf)
    assert np.all(np.isfinite(path.log_p[~outside]))
    assert resample([path], 3000, seed=0).points[:, 0].min() > -2.5


def test_a_path_is_its_seed_s() -> None:
    first = pathfinder(gaussian, np.zeros((2, 5)), seed=4, draws=100)
    second = pathfinder(gaussian, np.zeros((2, 5)), seed=4, draws=100)
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a.draws, b.draws)
        np.testing.assert_array_equal(a.log_p, b.log_p)


@pytest.fixture(scope="module")
def two_paths() -> tuple[Path, ...]:
    return pathfinder(two_modes, np.array([[-4.0, 0.5], [4.0, 0.5]]), seed=0, draws=4000)


def test_draws_resampled_from_two_paths_take_each_mode_by_its_mass(
    two_paths: tuple[Path, ...],
) -> None:
    """A path into each mode; 400 of their 8000 draws resampled take the left mode 0.8 of the
    time, within 4.5 binomial standard errors, no draw twice."""
    assert two_paths[0].draws[:, 0].mean() < 0.0 < two_paths[1].draws[:, 0].mean()
    chosen = resample(two_paths, 400, seed=9)
    left = float(np.mean(chosen.points[:, 0] < 0.0))
    assert abs(left - 0.8) <= 4.5 * math.sqrt(0.8 * 0.2 / 400)
    assert chosen.points.shape == (400, 2)
    assert len({tuple(p) for p in chosen.points}) == 400


def test_the_resampling_weights_are_the_pooled_ratios_smoothed() -> None:
    """Log ratios with a Pareto tail of shape 0.7, split over two paths: the draws taken are the
    ones the Pareto-smoothed weights of the pooled ratios choose from the same seed, and the
    smoothing's shape is reported."""
    rng = np.random.default_rng(6)
    ratios = 0.7 * rng.standard_exponential(1000)
    paths = [
        Path(
            draws=np.arange(500.0)[:, None] + 500.0 * half,
            log_q=-ratios[500 * half : 500 * (half + 1)],
            log_p=np.zeros(500),
            elbo=0.0,
            iteration=1,
            elbos=np.zeros(1),
        )
        for half in (0, 1)
    ]
    smoothed = psis(ratios)
    weights = np.exp(smoothed.log_weights)
    expected = np.random.default_rng(9).choice(
        1000, size=100, replace=False, p=weights / weights.sum()
    )
    chosen = resample(paths, 100, seed=9)
    np.testing.assert_array_equal(chosen.points[:, 0], expected.astype(float))
    assert chosen.pareto_k == smoothed.pareto_k


def test_what_cannot_be_approximated_is_refused() -> None:
    with pytest.raises(ValueError, match="not finite at the start"):
        pathfinder(lambda q: jnp.log(q[0]), np.array([[-1.0], [-2.0]]), seed=0)
    with pytest.raises(ValueError, match="no approximation along the path of 0 iterations"):
        pathfinder(lambda q: jnp.where(q[0] == 1.0, 0.0, -jnp.inf), np.ones((1, 1)), seed=0)


def test_a_path_that_cannot_be_approximated_is_dropped(caplog: pytest.LogCaptureFixture) -> None:
    """A density finite above -5 only, its mass near 0: the path from -6 starts where it is not
    finite, and the one from 1.5 climbs to the mode; the first is dropped and the second kept, as
    Stan drops it."""
    with caplog.at_level(logging.WARNING, logger="mixdyn.inference.pathfinder"):
        paths = pathfinder(
            lambda q: jnp.log(q[0] + 5.0) - 0.5 * q[0] ** 2,
            np.array([[-6.0], [1.5]]),
            seed=0,
            draws=50,
        )
    assert len(paths) == 1
    assert np.isfinite(paths[0].elbo)
    assert "1 of 2 Pathfinder paths dropped: from [-6.0], the log density" in caplog.text
    flat = Path(
        draws=np.zeros((3, 1)),
        log_q=np.zeros(3),
        log_p=np.array([0.0, -np.inf, -np.inf]),
        elbo=0.0,
        iteration=1,
        elbos=np.zeros(1),
    )
    with pytest.raises(ValueError, match="finite log density"):
        resample([flat], 2, seed=0)
