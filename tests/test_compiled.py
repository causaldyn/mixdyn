"""The compiled NUTS transition is the reference's: given the same point, step size, metric and
``Noise`` it makes the same decisions, and its sample, acceptance and energy agree to rounding. A
chain is sensitive to rounding, a start moved by ``1e-15`` ending warm-up ``1e-6`` away, so whole
chains are compared in distribution, not draw by draw."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mixdyn.inference import compiled
from mixdyn.inference.nuts import Noise, Point, transition

SCALES = np.array([0.5, 1.0, 2.0])


def banana(q):
    x, y = q[0], q[1]
    r = y - 0.8 * (x * x - 1.0)
    return -0.5 * (x * x + r * r)


def gaussian(q):
    return -0.5 * jnp.sum((q / SCALES) ** 2)


def cliff(q):
    """``1e155 tanh(q)``: a step off its edge overflows the kinetic energy, and diverges."""
    return 1e155 * jnp.tanh(q[0])


def edge(q):
    """Not a number past 1, as a density read outside its support is: a step there diverges."""
    return jnp.log1p(-q[0]) - 0.5 * q[0] ** 2


def numpy_density(log_density):
    value_and_grad = jax.jit(jax.value_and_grad(log_density))

    def density(q: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = value_and_grad(jnp.asarray(q))
        return float(value), np.asarray(gradient)

    return density


@pytest.mark.parametrize(
    ("log_density", "dimension", "spread"),
    [(banana, 2, 1.5), (gaussian, 3, 1.5), (cliff, 1, 0.01), (edge, 1, 0.3)],
    ids=["banana", "gaussian", "cliff", "edge"],
)
@pytest.mark.parametrize("max_depth", [3, 10])
def test_a_compiled_transition_makes_the_reference_s_decisions(
    log_density, dimension: int, spread: float, max_depth: int
) -> None:
    """From 150 random points, at step sizes from far too small to far too large and random
    metrics: the same tree depth, leapfrog steps and divergence, and the same sample, acceptance
    and energy to ``1e-12``."""
    density = numpy_density(log_density)
    kernel = compiled.kernel(log_density, max_depth)
    rng = np.random.default_rng(max_depth)
    depths, divergences = set(), 0
    for _ in range(150):
        q = rng.standard_normal(dimension) * spread
        step = float(rng.choice([0.05, 0.3, 0.9, 2.5]))
        metric = np.exp(rng.normal(0.0, 0.5, dimension))
        noise = Noise.draw(rng, dimension, max_depth)
        point = Point(q, *density(q))
        theirs, done = transition(density, point, step, metric, noise, max_depth)
        ours, made = kernel(point, step, metric, noise, max_depth)
        assert (made.depth, made.leapfrogs, made.divergent) == (
            done.depth,
            done.leapfrogs,
            done.divergent,
        )
        np.testing.assert_allclose(ours.position, theirs.position, rtol=1e-12, atol=1e-12)
        assert made.accept == pytest.approx(done.accept, rel=1e-12, abs=1e-12)
        assert made.energy == pytest.approx(done.energy, rel=1e-12, abs=1e-12)
        depths.add(done.depth)
        divergences += done.divergent
    if log_density in (cliff, edge):
        assert divergences > 0
    else:
        assert max(depths) == max_depth or len(depths) > 2


def test_compiled_chains_find_each_scale_and_the_acceptance_they_aim_at() -> None:
    """``nuts.sample``'s test of warm-up, on chains whose every transition is compiled."""
    scales = np.logspace(-2.0, 2.0, 5)
    chains = compiled.sample(
        lambda q: -0.5 * jnp.sum((q / scales) ** 2),
        np.full((2, 5), 0.01),
        seed=7,
        warmup=1000,
        draws=1000,
    )
    ratio = chains.inverse_metric / scales**2
    assert ratio.min() > 0.6
    assert ratio.max() < 1.6
    assert 0.8 <= chains.accept.mean() < 0.95
    assert not chains.divergent.any()
    z = chains.positions.reshape(-1, 5) / scales
    assert np.abs(z.mean(axis=0)).max() < 0.1
    assert np.abs(z.var(axis=0) - 1.0).max() < 0.15
