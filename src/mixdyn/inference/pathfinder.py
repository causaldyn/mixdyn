"""Pathfinder (Zhang, Carpenter, Gelman and Vehtari 2022): normal approximations to a posterior
along a quasi-Newton path to its mode, the one nearest the posterior by its evidence lower bound
kept, and several paths' draws importance-resampled together.

**One path.** L-BFGS-B climbs the log density from a start, ``history`` corrections deep. At each
iterate ``x`` after the start, with ``g`` the gradient of the negative log density there, the
inverse Hessian is estimated in Byrd, Nocedal and Schnabel's (1994) compact form from the last
``history`` steps ``s`` and gradient changes ``y`` whose curvature ``s . y`` is positive with
``|y|^2 / (s . y)`` at most ``1e12``, over a diagonal that starts at ones and is updated at each
such step as Gilbert and Lemaréchal (1989, eq. 4.9) update it; a step that fails the test is not
kept. The approximation there is the normal at ``x - H g`` with covariance ``H``. ``elbo_draws``
draws estimate its evidence lower bound, the mean of the log density less the approximation's log
density, a draw where the log density is not finite counting as minus infinity; ``draws`` draws
are made from the approximation whose estimate is the largest.

**Several paths.** A path that cannot be approximated, from a start where the log density or its
gradient is not finite or with no iterate whose evidence lower bound is finite, is dropped, as
Stan's multi-path Pathfinder drops it. The rest's draws are pooled and weighted by the log density
less the log density of the approximation each came from; the weights are Pareto-smoothed
(``mixdyn.inference.loo``), and ``count`` draws are taken without replacement in proportion to
them.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from scipy import linalg, optimize

from mixdyn.inference.compiled import JaxDensity
from mixdyn.inference.loo import psis

type _Array = npt.NDArray[np.float64]

_CURVATURE = 1e12

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Path:
    """One path's draws from its best approximation, with the log densities of each under the
    approximation (``log_q``) and under the target (``log_p``, minus infinity where it is not
    finite). ``elbos`` holds each iterate's estimated evidence lower bound after the start, minus
    infinity where no approximation was made or its estimate is not finite; ``elbo`` is the
    largest, at iterate ``iteration``."""

    draws: _Array
    log_q: _Array
    log_p: _Array
    elbo: float
    iteration: int
    elbos: _Array


@dataclass(frozen=True, slots=True)
class Resampled:
    """Draws resampled from several paths, with the Pareto shape of their smoothed weights."""

    points: _Array
    pareto_k: float


def pathfinder(
    log_density: JaxDensity,
    starts: npt.ArrayLike,
    *,
    seed: int,
    history: int = 5,
    elbo_draws: int = 25,
    draws: int = 1000,
    max_iterations: int = 1000,
) -> tuple[Path, ...]:
    """One path from each row of ``starts`` on ``log_density``, a log density up to a constant on
    the unconstrained space, written in JAX, those that cannot be approximated dropped with a
    warning.

    Raises:
        ValueError: when no path can be approximated: each started where the log density or its
            gradient is not finite, or no approximation's evidence lower bound on it is finite.
    """
    value_and_grad = jax.jit(jax.value_and_grad(log_density))
    batch = jax.jit(jax.vmap(log_density))
    rng = np.random.default_rng(seed)
    paths: list[Path] = []
    failures: list[str] = []
    for x0 in np.atleast_2d(np.asarray(starts, dtype=float)):
        path = _path(value_and_grad, batch, x0, rng, history, elbo_draws, draws, max_iterations)
        if isinstance(path, Path):
            paths.append(path)
        else:
            failures.append(f"from {x0.tolist()}, {path}")
    if not paths:
        msg = f"no path can be approximated: {'; '.join(failures)}"
        raise ValueError(msg)
    if failures:
        log.warning(
            "%d of %d Pathfinder paths dropped: %s",
            len(failures),
            len(failures) + len(paths),
            "; ".join(failures),
        )
    return tuple(paths)


def resample(paths: Sequence[Path], count: int, *, seed: int) -> Resampled:
    """``count`` of ``paths``' pooled draws, taken without replacement in proportion to their
    Pareto-smoothed importance weights.

    Raises:
        ValueError: if fewer than ``count`` draws have a finite log density or carry weight.
    """
    draws = np.concatenate([p.draws for p in paths])
    log_ratios = np.concatenate([p.log_p - p.log_q for p in paths])
    finite = np.isfinite(log_ratios)
    if finite.sum() < count:
        msg = f"{finite.sum()} draws have a finite log density, fewer than the {count} asked for"
        raise ValueError(msg)
    smoothed = psis(log_ratios[finite])
    weights = np.exp(smoothed.log_weights)
    if np.count_nonzero(weights) < count:
        msg = f"{np.count_nonzero(weights)} draws carry weight, fewer than the {count} asked for"
        raise ValueError(msg)
    chosen = np.random.default_rng(seed).choice(
        weights.size, size=count, replace=False, p=weights / weights.sum()
    )
    return Resampled(draws[finite][chosen], smoothed.pareto_k)


def _path(
    value_and_grad: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    batch: Callable[[jax.Array], jax.Array],
    x0: _Array,
    rng: np.random.Generator,
    history: int,
    elbo_draws: int,
    draws: int,
    max_iterations: int,
) -> Path | str:
    """The path from ``x0``, or why it cannot be approximated."""
    seen: dict[bytes, tuple[float, _Array]] = {}

    def negative(x: _Array) -> tuple[float, _Array]:
        key = x.tobytes()
        if key not in seen:
            value, gradient = value_and_grad(jnp.asarray(x))
            seen[key] = (-float(value), -np.asarray(gradient, dtype=float))
        return seen[key]

    f0, g0 = negative(x0)
    if not (math.isfinite(f0) and np.all(np.isfinite(g0))):
        return "the log density or its gradient is not finite at the start"
    iterates = [x0]
    optimize.minimize(
        negative,
        x0,
        jac=True,
        method="L-BFGS-B",
        callback=lambda x: iterates.append(np.array(x, dtype=float)),
        options={
            "maxcor": history,
            "ftol": 1e4 * np.finfo(float).eps,
            "gtol": 1e-8,
            "maxiter": max_iterations,
        },
    )
    gradients = [negative(x)[1] for x in iterates]
    elbos = np.full(len(iterates) - 1, -np.inf)
    best: tuple[float, int, _Array, _Array] | None = None
    for i, (alpha, steps, changes) in enumerate(_memory(iterates, gradients, history), start=1):
        if not len(steps):
            continue
        covariance = _inverse_hessian(steps, changes, alpha)
        try:
            factor = linalg.cholesky(covariance, lower=True, check_finite=False)
        except linalg.LinAlgError:
            continue
        mean = iterates[i] - covariance @ gradients[i]
        sample, log_q = _draw(rng, mean, factor, elbo_draws)
        elbo = float(np.mean(_log_p(batch, sample) - log_q))
        if math.isfinite(elbo):
            elbos[i - 1] = elbo
            if best is None or elbo > best[0]:
                best = (elbo, i, mean, factor)
    if best is None:
        return f"no approximation along the path of {elbos.size} iterations has a finite ELBO"
    elbo, iteration, mean, factor = best
    sample, log_q = _draw(rng, mean, factor, draws)
    return Path(sample, log_q, _log_p(batch, sample), elbo, iteration, elbos)


def _memory(
    iterates: Sequence[_Array], gradients: Sequence[_Array], history: int
) -> Iterator[tuple[_Array, _Array, _Array]]:
    """At each iterate after the first, the diagonal and the last ``history`` steps and gradient
    changes, oldest first, that passed the curvature test; a step that fails it is not kept."""
    alpha = np.ones(iterates[0].size)
    steps: list[_Array] = []
    changes: list[_Array] = []
    for i in range(1, len(iterates)):
        s, y = iterates[i] - iterates[i - 1], gradients[i] - gradients[i - 1]
        curvature = float(s @ y)
        if curvature > 0.0 and float(y @ y) / curvature <= _CURVATURE:
            alpha = _diagonal(alpha, s, y)
            steps, changes = [*steps, s][-history:], [*changes, y][-history:]
        yield alpha, np.array(steps), np.array(changes)


def _diagonal(alpha: _Array, s: _Array, y: _Array) -> _Array:
    """Gilbert and Lemaréchal's (1989) eq. 4.9: the diagonal of the inverse Hessian updated by one
    step ``s`` and its gradient change ``y``."""
    y_alpha_y = float(y @ (alpha * y))
    s_alpha_s = float(s @ (s / alpha))
    return float(y @ s) / (y_alpha_y / alpha + y**2 - (y_alpha_y / s_alpha_s) * (s / alpha) ** 2)


def _inverse_hessian(s: _Array, y: _Array, alpha: _Array) -> _Array:
    """The BFGS inverse Hessian over ``diag(alpha)`` after the steps ``s`` and gradient changes
    ``y``, ``(history, d)`` oldest first, in Byrd, Nocedal and Schnabel's compact form."""
    sy = s @ y.T
    n = -linalg.solve_triangular(np.triu(sy), s)
    ya = y * alpha
    middle = ya @ y.T + np.diag(np.diag(sy))
    return np.diag(alpha) + ya.T @ n + n.T @ ya + n.T @ middle @ n


def _draw(
    rng: np.random.Generator, mean: _Array, factor: _Array, count: int
) -> tuple[_Array, _Array]:
    """``count`` draws of the normal with ``mean`` and covariance ``factor @ factor.T``, and the
    normal's log density at each."""
    u = rng.standard_normal((count, mean.size))
    log_q = -np.log(np.diag(factor)).sum() - 0.5 * (
        (u**2).sum(axis=1) + mean.size * math.log(2.0 * math.pi)
    )
    return mean + u @ factor.T, log_q


def _log_p(batch: Callable[[jax.Array], jax.Array], sample: _Array) -> _Array:
    values = np.asarray(batch(jnp.asarray(sample)), dtype=float)
    return np.where(np.isfinite(values), values, -np.inf)
