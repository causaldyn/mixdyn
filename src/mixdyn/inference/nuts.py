"""The No-U-Turn sampler, the reference: one transition at a time, its subtrees built by recursion,
in NumPy float64.

A transition draws a momentum and doubles a leapfrog trajectory in a random direction until the
trajectory turns back on itself (Hoffman and Gelman 2014). Its sample is drawn from the trajectory
in proportion to each point's ``exp(-H)``: within a subtree by uniform progressive sampling, and
between the trajectory and a new subtree by biased progressive sampling, which favours the new
subtree (Betancourt 2017, appendix A). The trajectory turns back where the generalised criterion
``rho . M^-1 p > 0`` fails at either end, ``rho`` the sum of its momenta. It is checked around each
merged pair of subtrees and across the pair, from each part to the nearer end of the other, which
catches a turn the check around the pair misses. A step whose energy rises by more than
``MAX_ENERGY_ERROR``, or overflows, diverges and ends the transition, its subtree discarded.

Warm-up is Stan's (Stan Reference Manual, "MCMC Sampling"): the step size by dual averaging towards
a mean acceptance probability ``target``, and a diagonal metric estimated over windows that double
from 25 transitions, after an initial 75 and before a terminal 50, each window's variance shrunk
towards ``1e-3`` by ``5 / (n + 5)``; the step size is searched for again at the end of each window,
and its dual averaging restarted there. Told the acceptance statistics and positions of a warm-up
CmdStan 2.38 ran, ``Adaptation`` gives each transition the step size Stan gave it, to the bit, and
leaves Stan's step size and metric. One difference: from 20 transitions to 149, Stan shrinks the
buffers to 15 % and 10 % of warm-up, as its manual says, but never closes the window between them,
so adapts no metric; this module closes it.

mixdyn owns its sampler: the engine cannot install Stan, NumPyro or BlackJAX (ADR 0001). Stan is
the oracle its models are checked against. Each transition's random numbers are drawn before it
(``Noise``), one for each decision it can make, so that another implementation given the same
numbers makes the same decisions.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

type _Array = npt.NDArray[np.float64]
type LogDensity = Callable[[_Array], tuple[float, _Array]]
"""A log density on the unconstrained space, up to a constant, and its gradient."""

MAX_ENERGY_ERROR = 1000.0
"""The rise in the Hamiltonian over a trajectory's start at which a leapfrog step diverges."""

_INIT_BUFFER, _WINDOW, _TERM_BUFFER = 75, 25, 50
_GAMMA, _T0, _KAPPA = 0.05, 10.0, 0.75


@dataclass(frozen=True, slots=True)
class Point:
    """A position with its log density and that density's gradient."""

    position: _Array
    log_density: float
    gradient: _Array


@dataclass(frozen=True, slots=True)
class Noise:
    """One transition's random numbers: a standard normal for each coordinate of the momentum, and a
    uniform for every decision the transition can make. ``direction[k]`` sends the ``k``-th doubling
    forward when above one half; ``trajectory[k]`` moves the sample to that doubling's subtree;
    ``subtree[i]`` moves a subtree's sample to its second half at the ``i``-th merge, counting the
    merges in the order they happen."""

    momentum: _Array
    direction: _Array
    trajectory: _Array
    subtree: _Array

    @classmethod
    def draw(cls, rng: np.random.Generator, dimension: int, max_depth: int) -> Noise:
        return cls(
            rng.standard_normal(dimension),
            rng.random(max_depth),
            rng.random(max_depth),
            rng.random(1 << max_depth),
        )


@dataclass(frozen=True, slots=True)
class Transition:
    """What a transition did: the mean Metropolis acceptance probability over all its leapfrog
    steps, a discarded subtree's included, which warm-up adapts the step size to; the doublings
    kept; the leapfrog steps taken; whether one diverged; and the Hamiltonian at the sample."""

    accept: float
    depth: int
    leapfrogs: int
    divergent: bool
    energy: float


@dataclass(frozen=True, slots=True)
class Chains:
    """Each chain's draws after warm-up, ``(chains, draws)`` leading, with what each transition did,
    and each chain's step size and inverse metric as warm-up left them."""

    positions: _Array
    log_density: _Array
    accept: _Array
    depth: npt.NDArray[np.int64]
    leapfrogs: npt.NDArray[np.int64]
    divergent: npt.NDArray[np.bool_]
    energy: _Array
    step: _Array
    inverse_metric: _Array


@dataclass(frozen=True, slots=True)
class _State:
    q: _Array
    p: _Array
    log_density: float
    gradient: _Array


@dataclass(frozen=True, slots=True)
class _Subtree:
    sample: _State
    log_weight: float
    rho: _Array
    first: _State
    last: _State


def _energy(z: _State, inverse_metric: _Array) -> float:
    """The Hamiltonian at ``z``, infinite where it is not finite: a diverging trajectory's momentum
    and energy can overflow, and the step that reached them diverges."""
    with np.errstate(over="ignore", invalid="ignore"):
        h = -z.log_density + 0.5 * float(z.p @ (inverse_metric * z.p))
    return h if math.isfinite(h) else math.inf


def _leapfrog(log_density: LogDensity, z: _State, step: float, inverse_metric: _Array) -> _State:
    with np.errstate(over="ignore", invalid="ignore"):
        p = z.p + 0.5 * step * z.gradient
        q = z.q + step * inverse_metric * p
    value, gradient = log_density(q)
    with np.errstate(over="ignore", invalid="ignore"):
        return _State(q, p + 0.5 * step * gradient, value, gradient)


def _onward(sharp_first: _Array, sharp_last: _Array, rho: _Array) -> bool:
    return float(sharp_first @ rho) > 0.0 and float(sharp_last @ rho) > 0.0


class _Walk:
    """One transition's trajectory: its counts, and the subtrees it builds."""

    def __init__(
        self,
        log_density: LogDensity,
        step: float,
        inverse_metric: _Array,
        h0: float,
        noise: Noise,
    ) -> None:
        self.log_density = log_density
        self.step = step
        self.inverse_metric = inverse_metric
        self.h0 = h0
        self.noise = noise
        self.leapfrogs = 0
        self.metropolis = 0.0
        self.divergent = False
        self.merges = 0

    def build(self, z: _State, sign: float, depth: int) -> _Subtree | None:
        """``2^depth`` leapfrog steps on from ``z``, or None if they diverge or turn back."""
        if depth == 0:
            leaf = _leapfrog(self.log_density, z, sign * self.step, self.inverse_metric)
            self.leapfrogs += 1
            h = _energy(leaf, self.inverse_metric)
            self.divergent = h - self.h0 > MAX_ENERGY_ERROR
            self.metropolis += 1.0 if self.h0 - h > 0.0 else math.exp(self.h0 - h)
            if self.divergent:
                return None
            return _Subtree(leaf, self.h0 - h, leaf.p, leaf, leaf)
        first = self.build(z, sign, depth - 1)
        if first is None:
            return None
        second = self.build(first.last, sign, depth - 1)
        if second is None:
            return None
        log_weight = float(np.logaddexp(first.log_weight, second.log_weight))
        u = self.noise.subtree[self.merges]
        self.merges += 1
        moved = second.log_weight > log_weight or u < math.exp(second.log_weight - log_weight)
        rho = first.rho + second.rho
        m = self.inverse_metric
        if not (
            _onward(m * first.first.p, m * second.last.p, rho)
            and _onward(m * first.first.p, m * second.first.p, first.rho + second.first.p)
            and _onward(m * first.last.p, m * second.last.p, second.rho + first.last.p)
        ):
            return None
        sample = second.sample if moved else first.sample
        return _Subtree(sample, log_weight, rho, first.first, second.last)


def transition(
    log_density: LogDensity,
    point: Point,
    step: float,
    inverse_metric: _Array,
    noise: Noise,
    max_depth: int,
) -> tuple[Point, Transition]:
    """One NUTS transition from ``point`` with a diagonal metric, its random numbers ``noise``."""
    m = inverse_metric
    p0 = noise.momentum / np.sqrt(m)
    z0 = _State(point.position, p0, point.log_density, point.gradient)
    walk = _Walk(log_density, step, m, _energy(z0, m), noise)
    backward = forward = z0
    rho = p0
    sample, log_weight, depth = z0, 0.0, 0
    while depth < max_depth:
        ahead = bool(noise.direction[depth] > 0.5)
        subtree = walk.build(forward if ahead else backward, 1.0 if ahead else -1.0, depth)
        if subtree is None:
            break
        if subtree.log_weight > log_weight or noise.trajectory[depth] < math.exp(
            subtree.log_weight - log_weight
        ):
            sample = subtree.sample
        log_weight = float(np.logaddexp(log_weight, subtree.log_weight))
        depth += 1
        # The two parts in time order: the earlier, b, and the later, f.
        if ahead:
            rho_b, b_first, b_last = rho, backward, forward
            rho_f, f_first, f_last = subtree.rho, subtree.first, subtree.last
            forward = subtree.last
        else:
            rho_b, b_first, b_last = subtree.rho, subtree.last, subtree.first
            rho_f, f_first, f_last = rho, backward, forward
            backward = subtree.last
        rho = rho_b + rho_f
        if not (
            _onward(m * b_first.p, m * f_last.p, rho)
            and _onward(m * b_first.p, m * f_first.p, rho_b + f_first.p)
            and _onward(m * b_last.p, m * f_last.p, rho_f + b_last.p)
        ):
            break
    moved = Point(sample.q, sample.log_density, sample.gradient)
    accept = walk.metropolis / walk.leapfrogs
    return moved, Transition(accept, depth, walk.leapfrogs, walk.divergent, _energy(sample, m))


def find_step(
    log_density: LogDensity,
    point: Point,
    step: float,
    inverse_metric: _Array,
    rng: np.random.Generator,
) -> float:
    """Stan's search from ``step``: double or halve it until one leapfrog step from a fresh momentum
    crosses an acceptance probability of 0.8. A step of 0, above ``1e7`` or not a number is kept.

    Raises:
        ValueError: the step grew past ``1e7``, so the density is improper, or shrank to 0.
    """
    if step == 0.0 or step > 1e7 or math.isnan(step):
        return step

    def gain(step: float) -> float:
        p = rng.standard_normal(point.position.size) / np.sqrt(inverse_metric)
        z = _State(point.position, p, point.log_density, point.gradient)
        return _energy(z, inverse_metric) - _energy(
            _leapfrog(log_density, z, step, inverse_metric), inverse_metric
        )

    up = gain(step) > math.log(0.8)
    while True:
        delta = gain(step)
        if (up and not delta > math.log(0.8)) or (not up and not delta < math.log(0.8)):
            return step
        step = step * 2.0 if up else step * 0.5
        if step > 1e7:
            msg = "the step size grew past 1e7: the density is improper"
            raise ValueError(msg)
        if step == 0.0:
            msg = "no step size is small enough: the density is not continuous"
            raise ValueError(msg)


class _DualAveraging:
    """Nesterov's dual averaging of the log step size (Hoffman and Gelman 2014, section 3.2)."""

    def __init__(self, mu: float) -> None:
        self.mu = mu
        self.counter = 0
        self.s_bar = 0.0
        self.x_bar = 0.0

    def learn(self, accept: float, target: float) -> float:
        self.counter += 1
        eta = 1.0 / (self.counter + _T0)
        self.s_bar = (1.0 - eta) * self.s_bar + eta * (target - min(accept, 1.0))
        x = self.mu - self.s_bar * math.sqrt(self.counter) / _GAMMA
        weight = self.counter**-_KAPPA
        self.x_bar = (1.0 - weight) * self.x_bar + weight * x
        return math.exp(x)


class _Windows:
    """Stan's windows for the metric: the positions' variance over each, shrunk at its end."""

    def __init__(self, warmup: int, dimension: int) -> None:
        init, window, term = _INIT_BUFFER, _WINDOW, _TERM_BUFFER
        if init + window + term > warmup:
            init, term = int(0.15 * warmup), int(0.1 * warmup)
            window = warmup - (init + term)
        self.active = warmup >= 20
        self.warmup, self.init, self.term = warmup, init, term
        self.counter = 0
        self.size = window
        self.end = init + window - 1
        self.count = 0
        self.mean = np.zeros(dimension)
        self.m2 = np.zeros(dimension)

    def learn(self, q: _Array, current: _Array) -> _Array | None:
        """Add a warm-up position; at a window's end, the variance of its positions shrunk, or
        ``current``, the inverse metric, shrunk as one position's."""
        if not self.active:
            return None
        if self.init <= self.counter < self.warmup - self.term:
            self.count += 1
            delta = q - self.mean
            self.mean = self.mean + delta / self.count
            self.m2 = self.m2 + delta * (q - self.mean)
        closing = self.counter == self.end
        self.counter += 1
        if not closing:
            return None
        n = float(self.count)
        variance = self.m2 / (n - 1.0) if self.count > 1 else current
        variance = (n / (n + 5.0)) * variance + 1e-3 * (5.0 / (n + 5.0))
        self._next()
        self.count = 0
        self.mean = np.zeros_like(self.mean)
        self.m2 = np.zeros_like(self.m2)
        return variance

    def _next(self) -> None:
        last = self.warmup - self.term - 1
        if self.end == last:
            return
        self.size *= 2
        self.end = self.counter - 1 + self.size
        if self.end == last:
            return
        if self.end + 2 * self.size >= self.warmup - self.term:
            self.end = last


type Search = Callable[[Point, float, _Array], float]
"""A step size search from a point, a step size and an inverse metric: ``find_step``, its density
and random numbers bound."""


class Adaptation:
    """Stan's warm-up without its transitions: told each warm-up transition's acceptance statistic
    and the point it moved to, it keeps the step size and inverse metric the next transition takes.

    The step size is searched for by ``search``: at the start, from 1 under a unit metric, and
    where a window closes, from the step size dual averaging reached, under the window's metric;
    dual averaging then starts again towards ten times the search's result. ``final_step`` is the
    step size warm-up leaves.
    """

    def __init__(self, warmup: int, dimension: int, target: float) -> None:
        self.target = target
        self.inverse_metric = np.ones(dimension)
        self._averaging = _DualAveraging(math.log(10.0))
        self._windows = _Windows(warmup, dimension)

    def start(self, point: Point, search: Search) -> float:
        """The first transition's step size, from ``point``."""
        return search(point, 1.0, self.inverse_metric)

    def learn(self, accept: float, point: Point, search: Search) -> float:
        """The next transition's step size, after one of acceptance statistic ``accept`` that
        moved to ``point``."""
        step = self._averaging.learn(accept, self.target)
        variance = self._windows.learn(point.position, self.inverse_metric)
        if variance is None:
            return step
        self.inverse_metric = variance
        step = search(point, step, variance)
        self._averaging = _DualAveraging(math.log(10.0 * step))
        return step

    @property
    def final_step(self) -> float:
        """The average of the log step sizes since the last search, exponentiated."""
        return math.exp(self._averaging.x_bar)


def sample(
    log_density: LogDensity,
    initial: _Array,
    *,
    seed: int,
    warmup: int = 1000,
    draws: int = 1000,
    max_depth: int = 10,
    target: float = 0.8,
) -> Chains:
    """NUTS chains from ``initial``, one row a chain, each adapting its step size and diagonal
    metric over ``warmup`` transitions and keeping the ``draws`` after; chain ``c`` draws its
    random numbers from ``numpy.random.default_rng([seed, c + 1])``, so that no chain reads
    ``default_rng(seed)``'s stream, which a caller may draw the starts from.

    Raises:
        ValueError: on settings out of range, a start where the log density or its gradient is not
            finite, or a density whose step size search fails.
    """
    return _chains(
        log_density,
        functools.partial(transition, log_density),
        initial,
        seed=seed,
        warmup=warmup,
        draws=draws,
        max_depth=max_depth,
        target=target,
    )


type Kernel = Callable[[Point, float, _Array, Noise, int], tuple[Point, Transition]]
"""``transition`` with its density bound, or another implementation of it."""


def _chains(
    log_density: LogDensity,
    kernel: Kernel,
    initial: _Array,
    *,
    seed: int,
    warmup: int,
    draws: int,
    max_depth: int,
    target: float,
) -> Chains:
    """``sample``'s chains, each transition taken by ``kernel``."""
    start = np.atleast_2d(np.asarray(initial, dtype=float))
    if warmup < 0 or draws < 1 or max_depth < 1 or not 0.0 < target < 1.0:
        msg = f"warmup {warmup}, draws {draws}, max_depth {max_depth}, target {target}"
        raise ValueError(msg)
    chains, dimension = start.shape
    positions = np.empty((chains, draws, dimension))
    log_densities, accept, energy = (np.empty((chains, draws)) for _ in range(3))
    depth = np.empty((chains, draws), dtype=np.int64)
    leapfrogs = np.empty((chains, draws), dtype=np.int64)
    divergent = np.empty((chains, draws), dtype=bool)
    steps = np.empty(chains)
    metrics = np.empty((chains, dimension))
    for c in range(chains):
        rng = np.random.default_rng([seed, c + 1])
        value, gradient = log_density(start[c])
        if not (math.isfinite(value) and np.isfinite(gradient).all()):
            msg = f"chain {c} starts where the log density is {value} and its gradient {gradient}"
            raise ValueError(msg)
        point = Point(start[c], value, gradient)
        adaptation = Adaptation(warmup, dimension, target)
        search = functools.partial(find_step, log_density, rng=rng)
        step = adaptation.start(point, search)
        for _ in range(warmup):
            noise = Noise.draw(rng, dimension, max_depth)
            point, done = kernel(point, step, adaptation.inverse_metric, noise, max_depth)
            step = adaptation.learn(done.accept, point, search)
        if warmup > 0:
            step = adaptation.final_step
        inverse_metric = adaptation.inverse_metric
        for i in range(draws):
            noise = Noise.draw(rng, dimension, max_depth)
            point, done = kernel(point, step, inverse_metric, noise, max_depth)
            positions[c, i] = point.position
            log_densities[c, i] = point.log_density
            accept[c, i], energy[c, i] = done.accept, done.energy
            depth[c, i], leapfrogs[c, i] = done.depth, done.leapfrogs
            divergent[c, i] = done.divergent
        steps[c], metrics[c] = step, inverse_metric
    return Chains(
        positions, log_densities, accept, depth, leapfrogs, divergent, energy, steps, metrics
    )
