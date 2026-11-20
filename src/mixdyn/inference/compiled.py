"""The reference NUTS transition (`mixdyn.inference.nuts`) compiled by XLA, for a log density
written in JAX.

Where the reference recurses, a subtree is built here by iteration: its leaves in time order, and
after each leaf the completed subtrees merged with their left neighbours for as long as the
binary count of the leaves carries. The merges come in the reference's order, so each reads the
same one of ``Noise.subtree``, and given the same point, step size, metric and ``Noise`` a
transition makes the reference's decisions; its arithmetic differs from the reference's only by
rounding. Warm-up, its step size searches and the random numbers stay the reference's, in NumPy,
one compiled call a transition.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from mixdyn.inference.nuts import MAX_ENERGY_ERROR, Chains, Noise, Point, Transition, _chains

type JaxDensity = Callable[[jax.Array], jax.Array]
"""A log density on the unconstrained space, up to a constant, written in JAX."""


class _Z(NamedTuple):
    q: jax.Array
    p: jax.Array
    log_density: jax.Array
    gradient: jax.Array


class _Subtree(NamedTuple):
    """What a merge reads of a subtree, its last state aside: that is always the latest leaf."""

    first_p: jax.Array
    rho: jax.Array
    log_weight: jax.Array
    sample: _Z


class _Count(NamedTuple):
    leapfrogs: jax.Array
    metropolis: jax.Array
    divergent: jax.Array
    merges: jax.Array


def _energy(z: _Z, inverse_metric: jax.Array) -> jax.Array:
    h = -z.log_density + 0.5 * jnp.dot(z.p, inverse_metric * z.p)
    return jnp.where(jnp.isfinite(h), h, jnp.inf)


def _leapfrog(value_and_grad: Callable, z: _Z, step: jax.Array, inverse_metric: jax.Array) -> _Z:
    p = z.p + 0.5 * step * z.gradient
    q = z.q + step * inverse_metric * p
    value, gradient = value_and_grad(q)
    return _Z(q, p + 0.5 * step * gradient, value, gradient)


def _onward(sharp_first: jax.Array, sharp_last: jax.Array, rho: jax.Array) -> jax.Array:
    return (jnp.dot(sharp_first, rho) > 0.0) & (jnp.dot(sharp_last, rho) > 0.0)


def _pick(take: jax.Array, a, b):
    return jax.tree.map(lambda x, y: jnp.where(take, x, y), a, b)


def _build(
    value_and_grad: Callable,
    start: _Z,
    sign: jax.Array,
    depth: jax.Array,
    step: jax.Array,
    m: jax.Array,
    h0: jax.Array,
    subtree_noise: jax.Array,
    count: _Count,
    max_depth: int,
) -> tuple[jax.Array, _Subtree, _Z, _Count]:
    """``2^depth`` leapfrog steps on from ``start``: whether none diverged or turned back, the
    subtree, its last state, and the counts."""
    d = start.q.shape[0]
    # each level's subtree waiting for its right neighbour: its first and last momenta, its rho,
    # log weight and sample
    waiting = (
        jnp.zeros((max_depth, d)),
        jnp.zeros((max_depth, d)),
        jnp.zeros((max_depth, d)),
        jnp.zeros(max_depth),
        jax.tree.map(lambda x: jnp.zeros((max_depth, *x.shape)), start),
    )
    empty = _Subtree(start.p, start.p, jnp.zeros(()), start)

    def more(carry):
        i, _, ok, _, _, _ = carry
        return ok & (i < 2**depth)

    def leaf(carry):
        i, state, _, waiting, _, count = carry
        z = _leapfrog(value_and_grad, state, sign * step, m)
        h = _energy(z, m)
        divergent = h - h0 > MAX_ENERGY_ERROR
        count = count._replace(
            leapfrogs=count.leapfrogs + 1,
            metropolis=count.metropolis + jnp.where(h0 - h > 0.0, 1.0, jnp.exp(h0 - h)),
            divergent=divergent,
        )
        first_ps, last_ps, rhos, weights, samples = waiting

        def carries(merging):
            k, _, ok, _ = merging
            return ok & ((i >> k) & 1 == 1)

        def merge(merging):
            k, second, _, merges = merging
            log_weight = jnp.logaddexp(weights[k], second.log_weight)
            moved = (second.log_weight > log_weight) | (
                subtree_noise[merges] < jnp.exp(second.log_weight - log_weight)
            )
            rho = rhos[k] + second.rho
            ok = (
                _onward(m * first_ps[k], m * z.p, rho)
                & _onward(m * first_ps[k], m * second.first_p, rhos[k] + second.first_p)
                & _onward(m * last_ps[k], m * z.p, second.rho + last_ps[k])
            )
            first_sample = jax.tree.map(lambda x: x[k], samples)
            sample = _pick(moved, second.sample, first_sample)
            return k + 1, _Subtree(first_ps[k], rho, log_weight, sample), ok, merges + 1

        k, merged, ok, merges = jax.lax.while_loop(
            carries, merge, (jnp.array(0), _Subtree(z.p, z.p, h0 - h, z), ~divergent, count.merges)
        )
        waiting = (
            first_ps.at[k].set(merged.first_p),
            last_ps.at[k].set(z.p),
            rhos.at[k].set(merged.rho),
            weights.at[k].set(merged.log_weight),
            jax.tree.map(lambda x, y: x.at[k].set(y), samples, merged.sample),
        )
        return i + 1, z, ok, waiting, merged, count._replace(merges=merges)

    _, last, ok, _, built, count = jax.lax.while_loop(
        more, leaf, (jnp.array(0), start, jnp.array(True), waiting, empty, count)
    )
    return ok, built, last, count


def _transition(
    value_and_grad: Callable,
    q: jax.Array,
    log_density: jax.Array,
    gradient: jax.Array,
    step: jax.Array,
    m: jax.Array,
    momentum: jax.Array,
    direction: jax.Array,
    trajectory: jax.Array,
    subtree: jax.Array,
    *,
    max_depth: int,
):
    p0 = momentum / jnp.sqrt(m)
    z0 = _Z(q, p0, log_density, gradient)
    h0 = _energy(z0, m)
    count = _Count(jnp.array(0), jnp.array(0.0), jnp.array(False), jnp.array(0))

    def more(carry):
        depth, _, _, _, _, _, go, _ = carry
        return go & (depth < max_depth)

    def double(carry):
        depth, forward, backward, rho, sample, log_weight, _, count = carry
        ahead = direction[depth] > 0.5
        ok, built, last, count = _build(
            value_and_grad,
            _pick(ahead, forward, backward),
            jnp.where(ahead, 1.0, -1.0),
            depth,
            step,
            m,
            h0,
            subtree,
            count,
            max_depth,
        )
        moved = (built.log_weight > log_weight) | (
            trajectory[depth] < jnp.exp(built.log_weight - log_weight)
        )
        # the two parts in time order: the earlier, b, and the later, f
        rho_b = jnp.where(ahead, rho, built.rho)
        b_first = jnp.where(ahead, backward.p, last.p)
        b_last = jnp.where(ahead, forward.p, built.first_p)
        rho_f = jnp.where(ahead, built.rho, rho)
        f_first = jnp.where(ahead, built.first_p, backward.p)
        f_last = jnp.where(ahead, last.p, forward.p)
        onward = (
            _onward(m * b_first, m * f_last, rho_b + rho_f)
            & _onward(m * b_first, m * f_first, rho_b + f_first)
            & _onward(m * b_last, m * f_last, rho_f + b_last)
        )
        # after a doubling that fails, only the depth and the sample are read
        return (
            jnp.where(ok, depth + 1, depth),
            _pick(ahead, last, forward),
            _pick(ahead, backward, last),
            rho_b + rho_f,
            _pick(ok & moved, built.sample, sample),
            jnp.logaddexp(log_weight, built.log_weight),
            ok & onward,
            count,
        )

    depth, _, _, _, sample, _, _, count = jax.lax.while_loop(
        more, double, (jnp.array(0), z0, z0, p0, z0, jnp.array(0.0), jnp.array(True), count)
    )
    return (
        sample.q,
        sample.log_density,
        sample.gradient,
        count.metropolis / count.leapfrogs,
        depth,
        count.leapfrogs,
        count.divergent,
        _energy(sample, m),
    )


def _kernel(compiled: Callable, point: Point, step: float, inverse_metric, noise: Noise, _: int):
    q, value, gradient, accept, depth, leapfrogs, divergent, energy = compiled(
        point.position,
        point.log_density,
        point.gradient,
        step,
        inverse_metric,
        noise.momentum,
        noise.direction,
        noise.trajectory,
        noise.subtree,
    )
    return Point(np.asarray(q), float(value), np.asarray(gradient)), Transition(
        float(accept), int(depth), int(leapfrogs), bool(divergent), float(energy)
    )


def kernel(log_density: JaxDensity, max_depth: int):
    """``nuts.transition`` for ``log_density``, compiled: a function of a point, a step size, an
    inverse metric, a transition's ``Noise`` and ``max_depth``, which must be the one given here."""
    value_and_grad = jax.value_and_grad(log_density)
    compiled = jax.jit(functools.partial(_transition, value_and_grad, max_depth=max_depth))
    return functools.partial(_kernel, compiled)


def sample(
    log_density: JaxDensity,
    initial,
    *,
    seed: int,
    warmup: int = 1000,
    draws: int = 1000,
    max_depth: int = 10,
    target: float = 0.8,
) -> Chains:
    """``nuts.sample`` for a log density written in JAX, each transition compiled.

    Raises:
        ValueError: as ``nuts.sample``.
    """
    value_and_grad = jax.jit(jax.value_and_grad(log_density))

    def numpy_density(q):
        value, gradient = value_and_grad(jnp.asarray(q, dtype=float))
        return float(value), np.asarray(gradient)

    return _chains(
        numpy_density,
        kernel(log_density, max_depth),
        initial,
        seed=seed,
        warmup=warmup,
        draws=draws,
        max_depth=max_depth,
        target=target,
    )
