"""The reference NUTS: a transition leaves its target exactly invariant, checked from exact draws of
targets that can be drawn exactly; its kernel is Stan's, statistic by statistic, at Stan's step and
metric; warm-up finds each scale and the acceptance it aims at; a centred funnel diverges where its
non-centred form does not; a constrained posterior is recovered through its transform."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from scipy import special, stats

from mixdyn.inference.nuts import Adaptation, Noise, Point, find_step, sample, transition

ORACLES = Path(__file__).parent / "fixtures" / "oracles"
STAN = json.loads((ORACLES / "nuts_stan_kernel.json").read_text())
WARMUP = json.loads((ORACLES / "nuts_stan_warmup.json").read_text())


def gaussian(scales: np.ndarray):
    def log_density(q: np.ndarray) -> tuple[float, np.ndarray]:
        return -0.5 * float(np.sum((q / scales) ** 2)), -q / scales**2

    return log_density


def banana(bend: float):
    """``x ~ N(0, 1)``, ``y | x ~ N(bend (x^2 - 1), 1)``: curved, and drawn exactly."""

    def log_density(q: np.ndarray) -> tuple[float, np.ndarray]:
        x, y = q
        r = y - bend * (x * x - 1.0)
        return -0.5 * (x * x + r * r), np.array([-x + 2.0 * bend * x * r, -r])

    return log_density


def exact_banana(bend: float, n: int, rng: np.random.Generator) -> np.ndarray:
    x = rng.standard_normal(n)
    return np.column_stack([x, bend * (x * x - 1.0) + rng.standard_normal(n)])


def after(log_density, starts: np.ndarray, step: float, metric: np.ndarray, moves: int, seed: int):
    """Each start moved ``moves`` transitions, independently, at a fixed step and metric."""
    rng = np.random.default_rng(seed)
    ends, depths = np.empty_like(starts), []
    for i, q in enumerate(starts):
        value, gradient = log_density(q)
        point = Point(q, value, gradient)
        for _ in range(moves):
            point, done = transition(
                log_density, point, step, metric, Noise.draw(rng, q.size, 10), 10
            )
            depths.append(done.depth)
        ends[i] = point.position
    return ends, np.array(depths)


@pytest.mark.parametrize(("step", "metric"), [(0.9, [1.0, 1.0, 1.0]), (0.35, [0.25, 1.0, 4.0])])
def test_transitions_from_exact_gaussian_draws_stay_exact(step: float, metric: list[float]) -> None:
    """Each end is an exact draw if the kernel leaves the target invariant, and the starts are
    independent, so each coordinate's ends pass a Kolmogorov-Smirnov test against its marginal."""
    scales = np.array([0.5, 1.0, 2.0])
    rng = np.random.default_rng(1)
    starts = rng.standard_normal((3000, 3)) * scales
    ends, depths = after(gaussian(scales), starts, step, np.array(metric), 2, seed=2)
    assert depths.max() >= 3
    for j in range(3):
        assert stats.kstest(ends[:, j] / scales[j], "norm").pvalue > 1e-3
    moved = np.mean(np.abs(ends - starts) > 1e-12)
    assert moved > 0.9


def test_transitions_from_exact_draws_of_a_curved_target_stay_exact() -> None:
    """The energy varies along a banana's trajectories, so a draw from a trajectory weighted wrongly
    shows; ``x`` and ``y``'s residual given ``x`` are independent standard normals."""
    bend = 0.8
    rng = np.random.default_rng(3)
    starts = exact_banana(bend, 3000, rng)
    ends, depths = after(banana(bend), starts, 0.45, np.ones(2), 3, seed=4)
    assert depths.max() >= 4
    residual = ends[:, 1] - bend * (ends[:, 0] ** 2 - 1.0)
    assert stats.kstest(ends[:, 0], "norm").pvalue > 1e-3
    assert stats.kstest(residual, "norm").pvalue > 1e-3
    assert abs(np.corrcoef(ends[:, 0] ** 2, residual)[0, 1]) < 4.0 / math.sqrt(3000)


def _lag1(x: np.ndarray) -> np.ndarray:
    c = x - x.mean()
    return c[1:] * c[:-1] / c.var()


def _batch_mean(x: np.ndarray, batches: int) -> tuple[float, float]:
    means = x[: x.size // batches * batches].reshape(batches, -1).mean(axis=1)
    return float(means.mean()), float(means.std(ddof=1) / math.sqrt(batches))


@pytest.mark.parametrize("name", ["gauss_unit", "gauss_matched", "banana"])
def test_the_kernel_is_stan_s(name: str) -> None:
    """20 000 transitions at the step and metric Stan ran, from its start: the tree depths are
    distributed as Stan's 200 000 (a chi-square test), and each statistic is within 4.5 joint
    standard errors of Stan's. A sampler valid but not Stan's fails: one that samples between the
    trajectory and a new subtree uniformly, or checks no turn across merged subtrees."""
    case = STAN["cases"][name]
    provenance = STAN["provenance"]
    data = case["data"]
    log_density = (
        gaussian(np.array(data["scales"])) if case["model"] == "gaussian" else banana(data["bend"])
    )
    metric = np.array(case["inverse_metric"])
    depth_limit = provenance["max_depth"]
    q = np.full(metric.size, provenance["start"])
    value, gradient = log_density(q)
    point = Point(q, value, gradient)
    rng = np.random.default_rng(12)
    rows, positions = [], []
    for _ in range(20_000):
        noise = Noise.draw(rng, metric.size, depth_limit)
        point, done = transition(log_density, point, case["step"], metric, noise, depth_limit)
        rows.append((done.depth, done.leapfrogs, done.accept, done.divergent))
        positions.append(point.position)
    depth, leapfrogs, accept, divergent = (
        np.array(c, dtype=float) for c in zip(*rows, strict=True)
    )
    q = np.array(positions)
    theirs = {int(k): v for k, v in case["depths"].items()}
    levels = sorted(set(theirs) | {int(d) for d in depth})
    table = [[theirs.get(k, 0) for k in levels], [int(np.sum(depth == k)) for k in levels]]
    assert stats.chi2_contingency(np.array(table)).pvalue > 1e-4
    ours = {"leapfrogs": leapfrogs, "accept": accept, "divergent": divergent}
    for j in range(metric.size):
        x = q[:, j]
        ours |= {f"q{j + 1}^2": x**2, f"lag1 q{j + 1}": _lag1(x), f"lag1 q{j + 1}^2": _lag1(x**2)}
    assert set(ours) == set(case["statistics"])
    for key, values in ours.items():
        mean, se = _batch_mean(values, provenance["batches"])
        stan = case["statistics"][key]
        assert abs(mean - stan["mean"]) <= 4.5 * math.hypot(se, stan["se"]), key


def test_a_transition_is_its_noise_s() -> None:
    log_density = banana(0.8)
    q = np.array([0.3, -0.2])
    value, gradient = log_density(q)
    noise = Noise.draw(np.random.default_rng(5), 2, 6)
    first = transition(log_density, Point(q, value, gradient), 0.4, np.ones(2), noise, 6)
    second = transition(log_density, Point(q, value, gradient), 0.4, np.ones(2), noise, 6)
    assert np.array_equal(first[0].position, second[0].position)
    assert first[1] == second[1]


@pytest.mark.parametrize(("height", "step"), [(1e155, 1.0), (1e308, 4.0)])
def test_a_trajectory_that_overflows_diverges(height: float, step: float) -> None:
    """A cliff, ``height tanh(q)``: one step off its edge, the kinetic energy (``1e155``) or the
    momentum itself (``1e308``) is too large for a float. That is a divergence, as in Stan, and no
    warning."""

    def cliff(q: np.ndarray) -> tuple[float, np.ndarray]:
        top = np.tanh(q)
        return height * float(top[0]), height * (1.0 - top**2)

    q = np.zeros(1)
    value, gradient = cliff(q)
    noise = Noise.draw(np.random.default_rng(8), 1, 4)
    _, done = transition(cliff, Point(q, value, gradient), step, np.ones(1), noise, 4)
    assert done.divergent
    assert (done.leapfrogs, done.depth, done.accept) == (1, 0, 0.0)


def test_a_transition_takes_at_most_its_depth_s_steps() -> None:
    log_density = gaussian(np.array([1e-3, 1.0]))
    q = np.array([0.0, 0.5])
    value, gradient = log_density(q)
    rng = np.random.default_rng(6)
    for _ in range(50):
        _, done = transition(
            log_density, Point(q, value, gradient), 1e-4, np.ones(2), Noise.draw(rng, 2, 3), 3
        )
        assert done.depth == 3
        assert done.leapfrogs == 7


def test_warm_up_finds_each_scale_and_the_acceptance_it_aims_at() -> None:
    """Scales over four orders of magnitude: the inverse metric is each variance, the mean
    acceptance at least the target, since the step size kept is the average of warm-up's log step
    sizes, and the draws' moments are the target's."""
    scales = np.logspace(-2.0, 2.0, 5)
    chains = sample(gaussian(scales), np.full((2, 5), 0.01), seed=7, warmup=1000, draws=1000)
    ratio = chains.inverse_metric / scales**2
    assert ratio.min() > 0.6
    assert ratio.max() < 1.6
    assert 0.8 <= chains.accept.mean() < 0.95
    assert not chains.divergent.any()
    z = chains.positions.reshape(-1, 5) / scales
    assert np.abs(z.mean(axis=0)).max() < 0.1
    assert np.abs(z.var(axis=0) - 1.0).max() < 0.15


@pytest.mark.parametrize("name", sorted(WARMUP["replay"]))
def test_warm_up_replays_stan_s_to_the_bit(name: str) -> None:
    """Told the acceptance statistics and positions of a warm-up CmdStan ran, adaptation gives each
    transition the step size Stan gave it, bit for bit; each search starts from a step size Stan's
    moved by a power of 2, the first from 1 under a unit metric, the last under the metric warm-up
    leaves; and warm-up leaves Stan's step size and inverse metric, also bit for bit."""
    case = WARMUP["replay"][name]
    steps, accept, position = case["step"], case["accept"], np.array(case["position"])
    dimension = position.shape[1]
    adaptation = Adaptation(case["warmup"], dimension, WARMUP["provenance"]["target"])
    t, starts, metrics = -1, [], []

    def search(point: Point, step: float, metric: np.ndarray) -> float:
        starts.append(step)
        metrics.append(metric.copy())
        power = math.log2(steps[t + 1] / step)
        assert power == round(power), t
        return steps[t + 1]

    def at(q: np.ndarray) -> Point:
        return Point(q, 0.0, np.zeros(dimension))

    step = adaptation.start(at(np.full(dimension, WARMUP["provenance"]["start"])), search)
    for t in range(case["warmup"]):
        assert step == steps[t], t
        step = adaptation.learn(accept[t], at(position[t]), search)
    assert starts[0] == 1.0
    assert len(starts) > 1
    assert np.array_equal(metrics[0], np.ones(dimension))
    assert np.array_equal(metrics[-1], case["inverse_metric"])
    assert adaptation.final_step == case["final_step"]
    assert np.array_equal(adaptation.inverse_metric, case["inverse_metric"])


@pytest.mark.parametrize("model", sorted(WARMUP["search"]))
def test_the_step_search_is_stan_s(model: str) -> None:
    """From 1 at Stan's start with a unit metric, the search's step sizes fall on the powers of 2
    as Stan's 4000 did (a chi-square test)."""
    case = WARMUP["search"][model]
    data = case["data"]
    log_density, dimension = (
        (gaussian(np.array(data["scales"])), len(data["scales"]))
        if model == "gaussian"
        else (banana(data["bend"]), 2)
    )
    q = np.full(dimension, case["start"])
    value, gradient = log_density(q)
    point, rng = Point(q, value, gradient), np.random.default_rng(13)
    ours = Counter(
        round(math.log2(find_step(log_density, point, 1.0, np.ones(dimension), rng)))
        for _ in range(case["chains"])
    )
    theirs = {int(k): n for k, n in case["powers"].items()}
    levels = sorted(set(theirs) | set(ours))
    table = [[theirs.get(k, 0) for k in levels], [ours[k] for k in levels]]
    assert stats.chi2_contingency(np.array(table)).pvalue > 1e-4


def test_an_extreme_step_is_kept_as_stan_keeps_it() -> None:
    log_density = gaussian(np.ones(2))
    q = np.full(2, 0.1)
    value, gradient = log_density(q)
    point, rng = Point(q, value, gradient), np.random.default_rng(14)
    assert find_step(log_density, point, 2e7, np.ones(2), rng) == 2e7
    assert math.isnan(find_step(log_density, point, math.nan, np.ones(2), rng))
    assert find_step(log_density, point, 2e6, np.ones(2), rng) < 10.0


def test_a_short_warm_up_adapts_the_metric() -> None:
    """At 100 transitions Stan adapts no metric; this sampler does, over its one window."""
    scales = np.array([0.1, 1.0, 10.0])
    chains = sample(gaussian(scales), np.full((2, 3), 0.1), seed=15, warmup=100, draws=10)
    assert (np.diff(np.log(chains.inverse_metric), axis=1) > 1.0).all()


def test_a_target_acceptance_moves_the_step_size() -> None:
    log_density = gaussian(np.ones(4))
    start = np.zeros((1, 4))
    low = sample(log_density, start, seed=8, warmup=400, draws=200, target=0.6)
    high = sample(log_density, start, seed=8, warmup=400, draws=200, target=0.95)
    assert low.step[0] > 1.5 * high.step[0]
    assert low.accept.mean() < high.accept.mean()


def funnel(centred: bool):
    """Neal's funnel, ``v ~ N(0, 3)``, ``x_i ~ N(0, exp(v / 2))``, four ``x``: as stated, or each
    ``x_i`` as ``exp(v / 2)`` times a standard normal."""

    def log_density(q: np.ndarray) -> tuple[float, np.ndarray]:
        v, x = q[0], q[1:]
        if not centred:
            return -v * v / 18.0 - 0.5 * float(x @ x), np.concatenate([[-v / 9.0], -x])
        w = np.exp(-v)
        value = -v * v / 18.0 - 0.5 * w * float(x @ x) - 0.5 * x.size * v
        dv = -v / 9.0 + 0.5 * w * float(x @ x) - 0.5 * x.size
        return value, np.concatenate([[dv], -w * x])

    return log_density


def test_a_centred_funnel_diverges_and_its_non_centred_form_does_not() -> None:
    start = np.zeros((2, 5))
    centred = sample(funnel(True), start, seed=9, warmup=500, draws=500)
    plain = sample(funnel(False), start, seed=9, warmup=500, draws=500)
    assert centred.divergent.sum() > 10
    assert not plain.divergent.any()
    v = plain.positions[:, :, 0].ravel()
    assert abs(v.mean()) < 0.5
    assert abs(v.std() / 3.0 - 1.0) < 0.15


def test_a_bounded_posterior_through_its_logit() -> None:
    """Beta(3, 7) on the logit scale, its log Jacobian added: the draws' mean and variance are the
    Beta's, 0.3 and 21 / 1100."""

    def log_density(q: np.ndarray) -> tuple[float, np.ndarray]:
        u = q[0]
        log_p, log_1p = -np.logaddexp(0.0, -u), -np.logaddexp(0.0, u)
        value = 3.0 * log_p + 7.0 * log_1p
        return float(value), np.array([3.0 - 10.0 * special.expit(u)])

    chains = sample(log_density, np.zeros((4, 1)), seed=10, warmup=500, draws=1000)
    p = special.expit(chains.positions.ravel())
    assert abs(p.mean() - 0.3) < 0.01
    assert abs(p.var() / (21.0 / 1100.0) - 1.0) < 0.1


def test_a_chain_is_its_seed_s() -> None:
    log_density = banana(0.5)
    first = sample(log_density, np.zeros((2, 2)), seed=11, warmup=100, draws=50)
    second = sample(log_density, np.zeros((2, 2)), seed=11, warmup=100, draws=50)
    assert np.array_equal(first.positions, second.positions)
    assert np.array_equal(first.step, second.step)


def test_a_start_outside_the_support_is_refused() -> None:
    def log_density(q: np.ndarray) -> tuple[float, np.ndarray]:
        return (-math.inf if q[0] < 0 else -q[0]), np.array([-1.0])

    with pytest.raises(ValueError, match="starts where the log density is -inf"):
        sample(log_density, np.array([[-1.0]]), seed=0, warmup=10, draws=10)


def test_a_start_where_the_gradient_is_not_finite_is_refused() -> None:
    def log_density(q: np.ndarray) -> tuple[float, np.ndarray]:
        return -abs(q[0]), np.array([math.nan if q[0] == 0.0 else -math.copysign(1.0, q[0])])

    with pytest.raises(ValueError, match="its gradient \\[nan\\]"):
        sample(log_density, np.array([[0.0]]), seed=0, warmup=10, draws=10)


def test_an_improper_density_is_refused() -> None:
    def flat(q: np.ndarray) -> tuple[float, np.ndarray]:
        return 0.0, np.zeros_like(q)

    with pytest.raises(ValueError, match="improper"):
        sample(flat, np.zeros((1, 2)), seed=0, warmup=10, draws=10)
