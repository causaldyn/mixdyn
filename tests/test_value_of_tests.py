"""mixdyn.experiments.value_of_tests and choose_tests: lift tests' value to the plan, alone and
together, by nested Monte Carlo over a posterior's draws, and the set to run, or none.

Three oracles hold the estimator. On the conjugate linear-Gaussian instance whose value Maxima
derived exactly (`voi_conjugate.json`, M4 and M5), the estimator's core with the closed-form plan
and either update reads EVSI 1/100 and EVPI 1/10 within 4.5 standard errors, the realised
reduction's 5 % and 95 % quantiles are Imhof's within binomial error, and the reweighted draws'
effective sample size averages the share Maxima's law gives; two tests on that instance (M6) read
their values apart and together, and the second's after the first; and refitted exactly, the
first test's value corrected by its refits reads EVSI within 4.5 standard errors by either update.
On two `tanh` channels whose test is linear in their coefficients (`voi_quadrature.json`),
`value_of_tests` itself reads the quadrature's EVSI and EVPI over CHC's exact plan, within 4.5
standard errors and the grid's bound, by either update. On a tiny world of two `tanh` channels
whose posterior CmdStan refitted at every outer draw of every affordable set of four tests
(`voi_bruteforce.json`), `choose_tests` reads each set's value by either update within 4.5
standard errors of its gap to the refits', and the refits' best set.

Besides: no outer draw's reduction passes its perfect information's; an outer draw's own chain is
left out of its posterior; the fits' weights are solved again with the tests' joint predictive
density, and given weights stay; an outer draw whose weights thin falls back to the Gaussian
update; every candidate reads the same normals in `value_of_tests`, and its own, the same in every
set, in `choose_tests`; the costs, the halves and the ranking are the arithmetic stated; the local
model is CHC's decision weight read as stated, of one test or two, and NaN on an S-shaped curve; a
refit reads each test's estimate of the reading the update read, by channel in the candidates'
order, at evenly spaced outer draws of each half, each half's mean is corrected by its own refits,
and the corrected mean's error is calibrated over replications; the affordable sets are every set
within half the market and the test budget, the greedy set grows as stated, and the verdict reads
the half no choice read, so that a test worth less than its cost is not run; and what cannot be
read is refused.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest
from chc import response
from chc.allocation import decision_weight
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from mixdyn import NATIONAL, Cadence, Channel, MediaPanel, Revenue
from mixdyn.calibration import LiftEstimate
from mixdyn.decisions import Range, Readings, worths
from mixdyn.decisions.posterior import _alike
from mixdyn.experiments import (
    Candidate,
    ExperimentChoice,
    ExperimentValue,
    Gaussian,
    Gross,
    Net,
    Refit,
    Reweight,
    choose_tests,
    predicted_lift,
    value_of_tests,
)
from mixdyn.experiments.value import (
    _affordable,
    _context,
    _corrected,
    _efficiency,
    _error,
    _fits_in,
    _greedy,
    _members,
    _Planner,
    _preposterior,
    _refitted,
    _reweighted,
    _Tests,
    _Updated,
    _updated,
)
from mixdyn.experiments.value import _value as _set_value
from mixdyn.inference.diagnostics import _ess, _split
from mixdyn.inference.loo import _stacking_of_two

ORACLES = Path(__file__).parent / "fixtures" / "oracles"
CONJUGATE = json.loads((ORACLES / "voi_conjugate.json").read_text())
QUADRATURE = json.loads((ORACLES / "voi_quadrature.json").read_text())
BRUTEFORCE = json.loads((ORACLES / "voi_bruteforce.json").read_text())
CHANNELS = (Channel("a"), Channel("b"))


@dataclass(frozen=True, slots=True, eq=False)
class _Chains:
    """Draws of one fit or more as the estimator reads them, without the channels' parameters."""

    fit: npt.NDArray[np.intp]
    chain: npt.NDArray[np.intp]
    stacking: npt.NDArray[np.float64]
    pointwise: npt.NDArray[np.float64] | None

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        counts = np.bincount(self.fit, minlength=self.stacking.size)
        return self.stacking[self.fit] / counts[self.fit]


def _one(
    lift: npt.NDArray[np.float64], share: float, se: float, noise: npt.NDArray[np.float64]
) -> _Tests:
    """One test, read with `noise` at the outer draws."""
    return _Tests(lift[:, None], np.array([share]), np.array([se]), noise[:, None])


def _value(key: str) -> npt.NDArray[np.float64]:
    return np.array(CONJUGATE["m4"][key]["value"], dtype=np.float64)


@functools.cache
def _conjugate() -> tuple[_Planner, _Chains, npt.NDArray[np.float64]]:
    """M4's instance as a posterior of two chains of 5000 exact draws: `V(a, θ) = θ'a - a'Ha/2` over
    `a_1 + a_2 = b`, planned on the posterior mean by `a*(m) = M m + h b`, and the test's lift
    `Jθ`, read with noise of variance `s2`."""
    m0, covariance = _value("m0")[:, 0], _value("S0")
    m, h, curvature = _value("M"), _value("h")[:, 0], _value("H")
    budget = float(_value("b"))
    theta = np.random.default_rng(20261005).multivariate_normal(m0, covariance, size=10_000)
    planner = _Planner(
        features=theta,
        plan=lambda mean: m @ mean + h * budget,
        worth=lambda i, plans: (
            plans @ theta[i] - 0.5 * np.einsum("pc,cd,pd->p", plans, curvature, plans)
        ),
    )
    draws = _Chains(np.zeros(10_000, dtype=np.intp), np.repeat([0, 1], 5000), np.ones(1), None)
    return planner, draws, theta @ _value("J")[0]


def _ess_share(tau_over_sigma: float) -> float:
    """Maxima's average share of draws a normal test's weights keep (M3), at the lift's spread over
    the test's standard error."""
    tau = tau_over_sigma
    return (2.0 * tau**2 + 1.0) / ((tau**2 + 1.0) * math.sqrt(4.0 * tau**2 + 1.0))


def test_the_ess_law_is_maximas() -> None:
    for point in CONJUGATE["m3"]["averages"]:
        assert _ess_share(point["tau / sigma"]) == pytest.approx(
            point["average ess share"]["value"], rel=1e-14, abs=0.0
        )


@pytest.mark.parametrize("update", [Gaussian(), Reweight()], ids=["gaussian", "reweight"])
def test_the_conjugate_test_is_worth_its_closed_form_and_no_more_than_perfect_information(
    update: Gaussian | Reweight,
) -> None:
    planner, draws, lift = _conjugate()
    se = math.sqrt(float(_value("s2")))
    outer = np.arange(lift.size)
    z = np.random.default_rng(11).standard_normal(outer.size)
    prior = _preposterior(planner, draws, outer, update)
    (reductions,), (updated,) = prior.reductions([_one(lift, 1.0, se, z)])
    perfect = prior.perfect()
    for values, exact in ((reductions, 1 / 100), (perfect, 1 / 10)):
        error = values.std(ddof=1) / math.sqrt(values.size)
        assert abs(values.mean() - exact) <= 4.5 * error
    assert np.all(reductions <= perfect + 1e-12)
    for quantile in CONJUGATE["m5"]["quantiles"]:
        level = quantile["level"]
        below = np.mean(reductions <= quantile["imhof"])
        assert abs(below - level) <= 4.5 * math.sqrt(level * (1.0 - level) / reductions.size)
    if isinstance(update, Reweight):
        tau = math.sqrt(float(np.var(lift)))
        share = np.mean(updated.ess) / 5000
        assert share == pytest.approx(_ess_share(tau / se), rel=0.05)


@pytest.mark.parametrize("update", [Gaussian(), Reweight()], ids=["gaussian", "reweight"])
def test_the_conjugate_tests_are_worth_their_closed_forms_apart_and_together(
    update: Gaussian | Reweight,
) -> None:
    """M6 on M4's instance: the sum of the two channels' parameters and their difference, each
    read with noise of variance 1/2, are worth 1/100 and 1/15 apart and 19/280 together, within 4.5
    standard errors; and the second is worth 81/1400 after the first, the pair's reduction less the
    first's at each outer draw, the first read with the same normals in both."""
    planner, draws, _ = _conjugate()
    m6 = CONJUGATE["m6"]
    lift = planner.features @ np.array(m6["J"]["value"]).T
    se = np.sqrt(np.diag(np.array(m6["N"]["value"])))
    outer = np.arange(lift.shape[0])
    z = np.random.default_rng(13).standard_normal((outer.size, 2))
    prior = _preposterior(planner, draws, outer, update)
    alone = [_Tests(lift[:, [t]], np.ones(1), se[[t]], z[:, [t]]) for t in range(2)]
    (first, second, pair), _ = prior.reductions([*alone, _Tests(lift, np.ones(2), se, z)])
    for values, key in (
        (first, "first"),
        (second, "second"),
        (pair, "pair"),
        (pair - first, "second after the first"),
    ):
        error = values.std(ddof=1) / math.sqrt(values.size)
        assert abs(values.mean() - m6[key]["value"]) <= 4.5 * error
    assert np.all(pair <= prior.perfect() + 1e-12)


@pytest.mark.parametrize("then", [Gaussian(), Reweight()], ids=["gaussian", "reweight"])
def test_an_exact_refit_corrects_the_conjugate_tests_value_to_its_closed_form(
    then: Gaussian | Reweight,
) -> None:
    """M4's test refitted exactly, the posterior mean `m0 + S0 J' (J S0 J' + s2)^-1 (y - J m0)`
    planned, at 200 outer draws of each half: each refit reads the reading the update read there,
    the update, all but exact here, misses the refit by under a tenth of its reductions' spread,
    and the corrected value reads EVSI 1/100 within 4.5 of its standard errors."""
    planner, draws, lift = _conjugate()
    s2 = float(_value("s2"))
    m0, covariance, j = _value("m0")[:, 0], _value("S0"), _value("J")[0]
    gain = covariance @ j / (j @ covariance @ j + s2)
    m, h, budget = _value("M"), _value("h")[:, 0], float(_value("b"))
    read: list[float] = []

    def plan(y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        read.append(float(y[0]))
        return m @ (m0 + gain * (y[0] - j @ m0)) + h * budget

    outer = np.arange(lift.size)
    z = np.random.default_rng(11).standard_normal(outer.size)
    tests = _one(lift, 1.0, math.sqrt(s2), z)
    prior = _preposterior(planner, draws, outer, then)
    (reductions,), _ = prior.reductions([tests])
    refitted = _refitted(prior, tests, 200, plan)
    places = np.flatnonzero(~np.isnan(refitted))
    assert places.size == 400
    np.testing.assert_array_equal(read, lift[places] + math.sqrt(s2) * z[places])
    gap = refitted[places] - reductions[places]
    assert np.std(gap) < 0.1 * np.std(refitted[places])
    estimate, error, _ = _corrected(reductions, refitted, draws.fit, draws.chain)
    assert abs(estimate - 1 / 100) <= 4.5 * error


def test_a_corrected_mean_reads_the_refits_mean_and_its_error_covers_it() -> None:
    """Reductions the update reads 0.5 above the refit's, with noise of the refit's spread: over
    2000 replications of 400 outer draws in two chains, every fourth refitted, the corrected mean
    is the mean and the mean gap, and its miss of the refits' mean, 1, over its standard error has
    mean 0 and variance 1, each within 4.5 of its standard error; the mean alone misses by the
    0.5."""
    rng = np.random.default_rng(41)
    n, replications = 400, 2000
    fit, chain = np.zeros(n, dtype=np.intp), np.repeat(np.arange(2), n // 2)
    places = np.arange(0, n, 4)
    scores, misses = [], []
    for _ in range(replications):
        refit = rng.normal(1.0, 1.0, n)
        base = refit + 0.5 + rng.normal(0.0, 1.0, n)
        refitted = np.full(n, np.nan)
        refitted[places] = refit[places]
        estimate, error, correction = _corrected(base, refitted, fit, chain)
        assert correction == float(np.mean(refit[places] - base[places]))
        assert estimate == float(base.mean()) + correction
        scores.append((estimate - 1.0) / error)
        misses.append(float(base.mean()) - 1.0)
    z = np.array(scores)
    assert abs(z.mean()) <= 4.5 / math.sqrt(replications)
    assert abs(z.var() - 1.0) <= 4.5 * math.sqrt(2.0 / replications)
    assert abs(np.mean(misses) - 0.5) <= 4.5 * np.std(misses) / math.sqrt(replications)


@functools.cache
def _two_tanh_channels(per: int) -> tuple[MediaPanel, Readings, Candidate, dict[Channel, Range]]:
    """The quadrature's world: two `tanh` channels of fixed kernels and curves, their coefficients
    `per` draws in each of two chains from the normal prior, the history of 104 weeks, and the
    go-dark of the first channel in a fifth of the market for four weeks of the history's last
    twelve, read over the twelve."""
    world, test = QUADRATURE["world"], QUADRATURE["test"]
    history = np.array(world["history"])
    weeks = history.shape[0]
    panel = MediaPanel(
        dates=np.datetime64("2023-01-02") + np.timedelta64(7, "D") * np.arange(weeks),
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=CHANNELS,
        controls=(),
        kpi=np.full((1, weeks), 2000.0),
        kpi_unit=Revenue(),
        spend=history[None],
        currency="EUR",
        control_values=np.zeros((1, weeks, 0)),
        filled=np.zeros((1, weeks), dtype=bool),
    )
    rng = np.random.default_rng(20261005)
    coefficient = rng.multivariate_normal(world["mean"], world["covariance"], size=2 * per)
    draws = 2 * per
    readings = Readings(
        channels=CHANNELS,
        retention=np.tile(world["retention"], (draws, 1)),
        half_saturation=np.tile(world["half_saturation"], (draws, 1)),
        slope=np.full((draws, 2), np.nan),
        coefficient=coefficient,
        tanh=np.ones(draws, dtype=bool),
        fit=np.zeros(draws, dtype=np.intp),
        chain=np.repeat([0, 1], per),
        stacking=np.ones(1),
        pointwise=None,
        length=world["length"],
    )
    share = test["share"]
    candidate = Candidate(
        CHANNELS[0],
        share * np.array(test["tested"]),
        share * np.array(test["spend"]),
        test["readout"][1] - test["readout"][0],
        test["se"],
        share,
    )
    limits = {
        c: Range(lo, hi) for c, lo, hi in zip(CHANNELS, world["lower"], world["upper"], strict=True)
    }
    return panel, readings, candidate, limits


def _grid_bound(panel: MediaPanel, readings: Readings, limits: dict[Channel, Range]) -> float:
    """How far a plan on the grid's envelopes can fall short of the best plan in expected worth,
    `sum_c h_c^2 max|m_c''| / 8`, `max|m''|` bounded by every draw's second differences: each
    posterior's mean worth is an average of the draws'."""
    world = QUADRATURE["world"]
    table = worths(panel, NATIONAL, readings, world["periods"], limits, grid=512).table
    return sum(float(np.max(np.abs(np.diff(table[:, c], 2, axis=-1)))) / 8.0 for c in range(2))


@pytest.mark.parametrize("update", [Gaussian(), Reweight()], ids=["gaussian", "reweight"])
def test_the_quadrature_test_is_worth_its_integral_end_to_end(
    update: Gaussian | Reweight,
) -> None:
    """The posterior planner on the grid, the predicted lift and the scoring, against the
    quadrature over CHC's exact plan: every outer draw's reduction, and perfect information's."""
    panel, readings, candidate, limits = _two_tanh_channels(1500)
    world = QUADRATURE["world"]
    np.testing.assert_allclose(
        predicted_lift(readings, candidate),
        readings.coefficient[:, 0] * QUADRATURE["test"]["gamma"],
        rtol=1e-13,
    )
    values = value_of_tests(
        panel,
        NATIONAL,
        readings,
        [candidate],
        world["budget"],
        world["periods"],
        limits,
        seed=3,
        update=update,
        draws=readings.fit.size,
    )
    bound = _grid_bound(panel, readings, limits)
    reductions = values.values[0].reductions
    error = reductions.std(ddof=1) / math.sqrt(reductions.size)
    assert abs(reductions.mean() - QUADRATURE["evsi"]["value"]) <= 4.5 * error + bound
    assert abs(values.evpi - QUADRATURE["evpi"]["value"]) <= 4.5 * values.evpi_error + bound
    assert values.draws == readings.fit.size


def _stan_world() -> tuple[MediaPanel, Readings, tuple[Candidate, ...], dict[Channel, Range]]:
    """The brute-force oracle's world: two `tanh` channels over 104 weeks, CmdStan's posterior of
    them in the kept draws, and each channel's go-dark and doubling."""
    world, posterior = BRUTEFORCE["world"], BRUTEFORCE["posterior"]
    kpi, spend = np.array(world["kpi"]), np.array(world["spend"])
    weeks = kpi.size
    channels = tuple(Channel(c) for c in world["channels"])
    panel = MediaPanel(
        dates=np.datetime64(world["start"]) + np.timedelta64(7, "D") * np.arange(weeks),
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=channels,
        controls=(),
        kpi=kpi[None],
        kpi_unit=Revenue(),
        spend=spend[None],
        currency="EUR",
        control_values=np.zeros((1, weeks, 0)),
        filled=np.zeros((1, weeks), dtype=bool),
    )
    retention = np.array(posterior["retention"])
    draws = retention.shape[0]
    readings = Readings(
        channels=channels,
        retention=retention,
        half_saturation=np.array(posterior["half_saturation"]),
        slope=np.full_like(retention, np.nan),
        coefficient=np.array(posterior["coefficient"]),
        tanh=np.ones(draws, dtype=bool),
        fit=np.zeros(draws, dtype=np.intp),
        chain=np.array(posterior["chain"], dtype=np.intp),
        stacking=np.ones(1),
        pointwise=None,
        length=posterior["length"],
    )
    candidates = tuple(
        Candidate(
            Channel(t["channel"]),
            np.array(t["spend"]),
            np.array(t["planned"]),
            t["readout"],
            t["se"],
            t["share"],
        )
        for t in BRUTEFORCE["tests"]
    )
    limits = {c: Range(*r) for c, r in zip(channels, BRUTEFORCE["plan"]["limits"], strict=True)}
    return panel, readings, candidates, limits


@functools.cache
def _stan_choice(update: str) -> tuple[tuple[Candidate, ...], ExperimentChoice]:
    panel, readings, candidates, limits = _stan_world()
    plan, provenance = BRUTEFORCE["plan"], BRUTEFORCE["provenance"]
    choice = choose_tests(
        panel,
        NATIONAL,
        readings,
        candidates,
        plan["budget"],
        plan["periods"],
        limits,
        test_budget=plan["test_budget"],
        seed=provenance["seed"],
        update={"gaussian": Gaussian(), "reweight": Reweight()}[update],
        draws=provenance["outer"],
    )
    return candidates, choice


def test_the_reweighted_reductions_on_the_world_by_stan_are_those_beside_its_refits() -> None:
    """The sets, their order and each outer draw's reduction under `Reweight` are the fixture's,
    so the refits it records were read at these outer draws with these normals."""
    candidates, choice = _stan_choice("reweight")
    sets = BRUTEFORCE["sets"]
    assert [
        [next(k for k, c in enumerate(candidates) if c is t) for t in value.candidates]
        for value in choice.values
    ] == [s["tests"] for s in sets]
    np.testing.assert_allclose(
        [value.reductions for value in choice.values],
        [s["reweight"] for s in sets],
        rtol=1e-9,
        atol=1e-9,
    )


@pytest.mark.parametrize("update", ["gaussian", "reweight"])
def test_every_set_on_the_world_by_stan_is_worth_what_its_refits_read(update: str) -> None:
    """Every affordable set of four tests on a tiny world whose posterior CmdStan refitted at
    every outer draw (`voi_bruteforce.json`): at the same outer draws and normals, each set's mean
    reduction by the update reads its refits' within 4.5 standard errors of their paired gap, and
    the update's best set is the refits' unless the refits' best two are within 2 standard errors
    of theirs."""
    _, choice = _stan_choice(update)
    refitted = np.array([s["refitted"] for s in BRUTEFORCE["sets"]])
    reductions = np.array([value.reductions for value in choice.values])
    count = refitted.shape[1]
    gaps = reductions - refitted
    misses = np.abs(gaps.mean(axis=1)) / (gaps.std(axis=1, ddof=1) / math.sqrt(count))
    assert np.all(misses <= 4.5), misses
    best, second = np.argsort(-refitted.mean(axis=1), kind="stable")[:2]
    apart = refitted[best] - refitted[second]
    close = apart.mean() <= 2.0 * apart.std(ddof=1) / math.sqrt(count)
    assert int(np.argmax(reductions.mean(axis=1))) == best or close


@functools.cache
def _ranked(sharpen: float, scale: float = 1.0, share: float = 1.0) -> npt.NDArray[np.float64]:
    """Simulation-based calibration of `Reweight`'s weights (Talts, Betancourt, Simpson, Vehtari
    and Gelman 2018) on the brute-force oracle's posterior: each draw in turn the truth, each test
    read at it with its standard error over `sharpen`, the other chains' draws weighed as the
    update weighs them, with the standard error times `scale` and the share times `share`. Each
    test's p-values, `(tests, 7)`, of the truth's weighted rank of the test's lift and of each
    channel's retention, half-saturation and coefficient being uniform over ten bins, by a
    chi-square whose variance is widened by a chain's draws over the weights' median effective
    size: the outer draws of a chain are ranked against the same draws."""
    _, readings, candidates, _ = _stan_world()
    count = readings.fit.size
    members = list(_members(readings))
    outer = np.arange(count)
    rng = np.random.default_rng(20261008)
    per = int(np.bincount(readings.chain).max())
    out = []
    for candidate in candidates:
        se = candidate.se / sharpen
        lift = predicted_lift(readings, candidate)
        z = rng.standard_normal(count)
        tests = _Tests(
            lift[:, None],
            np.array([candidate.share * share]),
            np.array([se * scale]),
            z[:, None],
        )
        y = candidate.share * lift[:, None] + se * z[:, None]
        weights, ess, _, _ = _reweighted(np.eye(count), readings, members, tests, outer, y)
        widen = 1.0 + per / float(np.median(ess))
        p = []
        for statistic in (lift, *readings.retention.T, *readings.half_saturation.T,
                          *readings.coefficient.T):  # fmt: skip
            rank = np.sum(weights * (statistic[None, :] < statistic[:, None]), axis=1)
            counts = np.bincount(np.minimum((rank * 10).astype(int), 9), minlength=10)
            chi_square = float(np.sum((counts - count / 10) ** 2 / (count / 10)))
            p.append(stats.chi2.sf(chi_square / widen, 9))
        out.append(p)
    return np.array(out)


@pytest.mark.parametrize("sharpen", [1.0, 4.0])
def test_the_update_ranks_each_truth_uniformly_on_the_world_by_stan(sharpen: float) -> None:
    """At the tests' own standard errors and at a quarter of them, where they move the posterior:
    no statistic's p-value is below 0.05 over the 28."""
    p = _ranked(sharpen)
    assert p.min() >= 0.05 / p.size, p


@pytest.mark.parametrize(
    ("scale", "share"),
    [(2.0, 1.0), (1.0, 1.2)],
    ids=["standard-error-doubled", "share-a-fifth-too-large"],
)
def test_the_ranks_turn_from_uniform_where_the_update_misreads_its_tests(
    scale: float, share: float
) -> None:
    """The calibration's power on this posterior: weights that read the tests' standard error
    doubled, or their share a fifth too large, rank some statistic away from uniform at a quarter
    of the tests' standard errors."""
    p = _ranked(4.0, scale, share)
    assert p.min() < 0.05 / p.size, p


def _two_chains(lift_mean: tuple[float, float], spread: float, per: int = 200) -> tuple:
    """Two chains of one fit whose features mark each draw's chain, and lifts about each chain's
    mean; a plan that is its mean features reads each chain's weight."""
    rng = np.random.default_rng(4)
    chain = np.repeat([0, 1], per)
    features = np.eye(2)[chain]
    lift = np.array(lift_mean)[chain] + spread * rng.standard_normal(2 * per)
    planner = _Planner(features, lambda mean: mean, lambda i, plans: np.zeros(plans.shape[0]))
    draws = _Chains(np.zeros(2 * per, dtype=np.intp), chain, np.ones(1), None)
    return planner, draws, lift


@pytest.mark.parametrize("update", [Gaussian(), Reweight()], ids=["gaussian", "reweight"])
def test_an_outer_draws_own_chain_is_left_out_of_its_posterior(
    update: Gaussian | Reweight,
) -> None:
    """A precise test whose lift sets the chains apart: each outer draw's posterior holds only the
    other chain, which its own would outweigh by far. Chains that disagree have a relative
    efficiency near 0, so the reweighted update falls back on every outer draw here."""
    planner, draws, lift = _two_chains((1.0, -1.0), 0.1)
    outer = np.array([0, 150, 250, 399])
    updated = _updated(planner, draws, _one(lift, 1.0, 0.5, np.zeros(outer.size)), outer, update)
    other = 1 - draws.chain[outer]
    np.testing.assert_allclose(updated.plans, np.eye(2)[other], rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("update", [Gaussian(), Reweight()], ids=["gaussian", "reweight"])
def test_an_outer_draw_is_scored_against_the_plan_of_its_posterior_without_its_chain(
    update: Gaussian | Reweight,
) -> None:
    """M4's draws as two fits of two chains each, weighed 0.3 and 0.7: each outer draw is scored
    against the plan of the mean of its posterior, its own chain left out and the fit's other chain
    taking the fit's weight, where every update starts. So a test of a standard error of 10^12,
    which reads nothing, is worth nothing at every outer draw."""
    planner, _, lift = _conjugate()
    fit, chain = np.repeat([0, 1, 0, 1], 2500), np.repeat([0, 0, 1, 1], 2500)
    draws = _Chains(fit, chain, np.array([0.3, 0.7]), None)
    outer = np.arange(0, 10_000, 25)
    prior = _preposterior(planner, draws, outer, update)
    theta = planner.features
    for j, i in enumerate(outer.tolist()):
        f, other = int(fit[i]), 1 - int(fit[i])
        own = theta[(fit == f) & (chain != chain[i])].mean(axis=0)
        rest = theta[fit == other].mean(axis=0)
        mean = draws.stacking[f] * own + draws.stacking[other] * rest
        expected = planner.worth(i, planner.plan(mean)[None])[0]
        assert prior.worth[j] == pytest.approx(expected, rel=1e-12, abs=0.0)
    blind = _one(lift, 1.0, 1e12, np.random.default_rng(3).standard_normal(outer.size))
    (reductions,), _ = prior.reductions([blind])
    assert np.max(np.abs(reductions)) < 1e-9


@pytest.mark.parametrize("update", [Gaussian(), Reweight()], ids=["gaussian", "reweight"])
def test_a_test_that_reads_nothing_is_worth_nothing_at_every_outer_draw(
    update: Gaussian | Reweight,
) -> None:
    """The cheap world's go-dark and doubling at standard errors of 10^12: each outer draw's plan
    after the update is the plan it is scored against, so every reduction is 0."""
    panel, readings, limits, cut, heavy_up = _cheap()
    world = QUADRATURE["world"]
    values = value_of_tests(
        panel,
        NATIONAL,
        readings,
        [dataclasses.replace(c, se=c.se * 1e12) for c in (cut, heavy_up)],
        world["budget"],
        world["periods"],
        limits,
        seed=5,
        draws=40,
        grid=64,
        update=update,
    )
    for value in values.values:
        np.testing.assert_array_equal(value.reductions, np.zeros(40))


def test_a_reweighted_outer_draw_weighs_nothing_of_its_own_chain() -> None:
    """Two chains of 1000 lifts of one law, read by a test of half their spread: the weights keep
    hundreds of effective draws, so no outer draw falls back, and each outer draw's plan, the
    weight its posterior gives each of four outer draws, holds none of its own chain's and some of
    the other's."""
    rng = np.random.default_rng(16)
    lift = rng.standard_normal(2000)
    outer = np.array([3, 700, 1100, 1999])
    planner = _Planner(
        (np.arange(2000)[:, None] == outer).astype(np.float64),
        lambda mean: mean,
        lambda i, plans: np.zeros(plans.shape[0]),
    )
    draws = _Chains(np.zeros(2000, dtype=np.intp), np.repeat([0, 1], 1000), np.ones(1), None)
    updated = _updated(
        planner, draws, _one(lift, 1.0, 0.5, np.zeros(outer.size)), outer, Reweight()
    )
    assert not updated.fell.any()
    same = draws.chain[outer][:, None] == draws.chain[outer][None, :]
    assert np.all(updated.plans[same] == 0.0)
    assert np.all(updated.plans[~same] > 0.0)


def _two_fits(pointwise: npt.NDArray[np.float64] | None) -> tuple:
    """Two fits of two chains each, the features marking each draw's fit, whose lifts lie far
    apart; their densities of the observations alike, so the observations weigh them evenly."""
    rng = np.random.default_rng(6)
    fit = np.repeat([0, 1], 800)
    chain = np.tile(np.repeat([0, 1], 400), 2)
    lift = np.where(fit == 0, 5.0, -5.0) + 0.1 * rng.standard_normal(1600)
    planner = _Planner(np.eye(2)[fit], lambda mean: mean, lambda i, plans: np.zeros(plans.shape[0]))
    return planner, _Chains(fit, chain, np.full(2, 0.5), pointwise), lift


@pytest.mark.parametrize("update", [Gaussian(), Reweight()], ids=["gaussian", "reweight"])
def test_the_fits_weights_are_solved_again_with_the_tests_density(
    update: Gaussian | Reweight,
) -> None:
    """A test that reads the first fit's lift gives it the whole weight where the weights were
    solved from the fits' densities, the reweighted draws enough to keep; and the given weights
    stay where they were given."""
    outer = np.array([0, 520])
    for pointwise, expected in ((np.zeros((2, 3)), [1.0, 0.0]), (None, [0.5, 0.5])):
        planner, draws, lift = _two_fits(pointwise)
        updated = _updated(planner, draws, _one(lift, 1.0, 0.1, np.zeros(2)), outer, update)
        np.testing.assert_allclose(updated.plans, [expected, expected], rtol=0.0, atol=1e-12)
        if pointwise is not None:
            assert not updated.fell.any()


@pytest.mark.parametrize("update", [Gaussian(), Reweight()], ids=["gaussian", "reweight"])
def test_each_fits_weight_is_solved_again_with_its_predictive_density_of_the_reading(
    update: Gaussian | Reweight,
) -> None:
    """Two fits of different counts, spreads and centres, whose densities of twenty observations
    hold neither at an end: each outer draw's weights are the stacking weights with the reading as
    one more observation, at each fit's predictive density, the mean of the test's likelihood over
    the fit's draws in the outer draw's posterior where they are reweighted, and the normal of
    their mean and variance with the test's noise where they are updated as Gaussian."""
    rng = np.random.default_rng(17)
    fit = np.repeat([0, 1], [600, 1200])
    chain = np.r_[np.repeat([0, 1], 300), np.repeat([0, 1], 600)]
    lift = np.where(fit == 0, 0.3 * rng.standard_normal(1800), 0.4 + rng.standard_normal(1800))
    pointwise = 0.5 * np.random.default_rng(18).standard_normal((2, 20))
    planner = _Planner(np.eye(2)[fit], lambda mean: mean, lambda i, plans: np.zeros(plans.shape[0]))
    share, se = 0.5, 0.3
    outer, z = np.array([5, 400, 700, 1500]), np.array([0.3, -1.2, 0.8, 0.1])
    draws = _Chains(fit, chain, np.full(2, 0.5), pointwise)
    updated = _updated(planner, draws, _one(lift, share, se, z), outer, update)
    assert not updated.fell.any()
    y = share * lift[outer] + se * z
    for r, i in enumerate(outer):
        density = []
        for k in range(2):
            mu = share * lift[(fit == k) & ((fit != fit[i]) | (chain != chain[i]))]
            if isinstance(update, Reweight):
                density.append(math.log(np.mean(stats.norm.pdf(y[r], mu, se))))
            else:
                spread = math.sqrt(np.var(mu) + se**2)
                density.append(stats.norm.logpdf(y[r], np.mean(mu), spread))
        expected = _stacking_of_two(np.column_stack([pointwise, density]))
        assert 0.5 < expected[0] < 0.95
        np.testing.assert_allclose(updated.plans[r], expected, rtol=0.0, atol=1e-9)


@pytest.mark.parametrize("update", [Gaussian(), Reweight()], ids=["gaussian", "reweight"])
def test_two_tests_weigh_each_fit_by_its_predictive_density_of_both_readings(
    update: Gaussian | Reweight,
) -> None:
    """The fits of the test above, read by a second test too, whose lift is the first's in part:
    each outer draw's weights are the stacking weights with both readings as one more observation,
    at the mean over the fit's draws of the product of the tests' likelihoods where they are
    reweighted, and at the bivariate normal of the lifts' mean and covariance with the tests' noise
    where they are updated as Gaussian."""
    rng = np.random.default_rng(17)
    fit = np.repeat([0, 1], [600, 1200])
    chain = np.r_[np.repeat([0, 1], 300), np.repeat([0, 1], 600)]
    first = np.where(fit == 0, 0.3 * rng.standard_normal(1800), 0.4 + rng.standard_normal(1800))
    second = np.where(
        fit == 0,
        -0.2 + 0.5 * rng.standard_normal(1800),
        0.6 * first + 0.5 * rng.standard_normal(1800),
    )
    lift = np.column_stack([first, second])
    pointwise = 0.5 * np.random.default_rng(18).standard_normal((2, 20))
    planner = _Planner(np.eye(2)[fit], lambda mean: mean, lambda i, plans: np.zeros(plans.shape[0]))
    share, se = np.array([0.5, 0.3]), np.array([0.3, 0.4])
    outer = np.array([5, 400, 700, 1500])
    z = np.array([[0.3, -0.5], [-1.2, 0.4], [0.8, 1.1], [0.1, -0.9]])
    draws = _Chains(fit, chain, np.full(2, 0.5), pointwise)
    updated = _updated(planner, draws, _Tests(lift, share, se, z), outer, update)
    assert not updated.fell.any()
    y = share * lift[outer] + se * z
    for r, i in enumerate(outer):
        density = []
        for k in range(2):
            mu = share * lift[(fit == k) & ((fit != fit[i]) | (chain != chain[i]))]
            if isinstance(update, Reweight):
                likelihood = np.prod(stats.norm.pdf(y[r], mu, se), axis=1)
                density.append(math.log(np.mean(likelihood)))
            else:
                spread = np.cov(mu, rowvar=False, bias=True) + np.diag(se**2)
                density.append(stats.multivariate_normal.logpdf(y[r], mu.mean(axis=0), spread))
        expected = _stacking_of_two(np.column_stack([pointwise, density]))
        assert 0.5 < expected[0] < 0.95
        np.testing.assert_allclose(updated.plans[r], expected, rtol=0.0, atol=1e-9)


def test_the_plan_without_a_test_weighs_each_fits_draws_by_its_weight() -> None:
    """Two fits of 400 and 800 draws weighed 1/4 and 3/4: the plan is theirs, not their counts'."""
    fit = np.repeat([0, 1], [400, 800])
    chain = np.r_[np.repeat([0, 1], 200), np.repeat([0, 1], 400)]
    planner = _Planner(np.eye(2)[fit], lambda mean: mean, lambda i, plans: np.zeros(plans.shape[0]))
    draws = _Chains(fit, chain, np.array([0.25, 0.75]), None)
    prior = _preposterior(planner, draws, np.array([0]), Gaussian())
    np.testing.assert_allclose(prior.plan, [0.25, 0.75], rtol=1e-13, atol=0.0)


@pytest.mark.parametrize("together", [False, True], ids=["alone", "beside-independent-lifts"])
def test_an_outer_draws_effective_sample_is_kishs_over_every_fit_times_the_lifts_efficiency(
    together: bool,
) -> None:
    """Two fits of the same AR(1) lifts of coefficient 0.5, weighed 1/4 and 3/4 as given, each draw
    its own feature, so that a plan is the weight it gives each draw: each outer draw's effective
    sample size is Kish's over those weights, `1 / sum q^2`, times the lift's relative efficiency
    over a fit's chains, here near a third; read beside a test of independent lifts, the least of
    the two lifts' efficiencies."""
    ar = _ar(np.random.default_rng(18), 0.5, 500).ravel()
    lift = np.tile(ar, 2)
    fit = np.repeat([0, 1], 1000)
    chain = np.tile(np.repeat([0, 1], 500), 2)
    planner = _Planner(np.eye(2000), lambda mean: mean, lambda i, plans: np.zeros(plans.shape[0]))
    draws = _Chains(fit, chain, np.array([0.25, 0.75]), None)
    outer = np.array([10, 1650])
    tests = _one(lift, 1.0, 1.0, np.zeros(2))
    efficiency = _efficiency(ar, chain[:1000])
    if together:
        other = np.random.default_rng(22).standard_normal(2000)
        for k in range(2):
            assert _efficiency(other[fit == k], chain[fit == k]) > 2.0 * efficiency
        tests = _Tests(np.column_stack([other, lift]), np.ones(2), np.ones(2), np.zeros((2, 2)))
    updated = _updated(planner, draws, tests, outer, Reweight())
    assert not updated.fell.any()
    assert 0.2 < efficiency < 0.5
    for r in range(outer.size):
        q = updated.plans[r]
        assert q[fit == 0].sum() == pytest.approx(0.25, rel=1e-13, abs=0.0)
        assert updated.ess[r] == pytest.approx(efficiency / np.sum(q**2), rel=1e-12, abs=0.0)


def test_an_outer_draw_whose_weights_thin_is_updated_as_gaussian() -> None:
    """With the lift's spread thirty times the test's standard error, the weights keep a thirtieth
    of the draws on average: the outer draws below 100 effective draws fall back to the Gaussian
    update, plan for plan, and the others do not."""
    planner, draws, lift = _conjugate()
    se = math.sqrt(float(np.var(lift))) / 30.0
    outer = np.arange(0, 10_000, 25)
    z = np.random.default_rng(12).standard_normal(outer.size)
    reweighted = _updated(planner, draws, _one(lift, 1.0, se, z), outer, Reweight())
    gaussian = _updated(planner, draws, _one(lift, 1.0, se, z), outer, Gaussian())
    assert 0.0 < reweighted.fell.mean() < 1.0
    assert np.all(reweighted.fell[reweighted.ess < 100.0])
    np.testing.assert_array_equal(
        reweighted.plans[reweighted.fell], gaussian.plans[reweighted.fell]
    )
    kept = ~reweighted.fell
    assert not np.array_equal(reweighted.plans[kept], gaussian.plans[kept])
    assert np.all(np.isnan(gaussian.ess))
    assert not gaussian.fell.any()


def test_an_outer_draw_whose_weights_tail_is_too_heavy_is_updated_as_gaussian() -> None:
    """Two chains of 5000 lifts, five at the test's reading and the rest 2.15 standard errors off
    it: the smoothed weights keep nearly every draw, but the tail they smoothed fits a shape past
    the threshold, so the outer draw is planned as `Gaussian` plans it."""
    rng = np.random.default_rng(15)
    lift = np.concatenate(
        [
            rng.permutation(np.r_[np.zeros(5), 2.15 + 0.001 * rng.standard_normal(4995)])
            for _ in range(2)
        ]
    )
    planner = _Planner(lift[:, None], lambda mean: mean, lambda i, plans: np.zeros(plans.shape[0]))
    draws = _Chains(np.zeros(10_000, dtype=np.intp), np.repeat([0, 1], 5000), np.ones(1), None)
    outer = np.flatnonzero(lift[:5000] == 0.0)[:1]
    reweighted = _updated(planner, draws, _one(lift, 1.0, 1.0, np.zeros(1)), outer, Reweight())
    gaussian = _updated(planner, draws, _one(lift, 1.0, 1.0, np.zeros(1)), outer, Gaussian())
    assert reweighted.ess[0] > 4900.0
    assert reweighted.pareto_k[0] > 0.7
    assert reweighted.fell[0]
    np.testing.assert_array_equal(reweighted.plans, gaussian.plans)


def _cheap() -> tuple[MediaPanel, Readings, dict[Channel, Range], Candidate, Candidate]:
    """The quadrature's world on 60 draws a chain, with its go-dark and a doubling of the second
    channel's last four weeks in a tenth of the market."""
    panel, readings, cut, limits = _two_tanh_channels(60)
    market = panel.spend[0, :, 1]
    doubled = market.copy()
    doubled[-12:-8] *= 2.0
    heavy_up = Candidate(CHANNELS[1], 0.1 * doubled, 0.1 * market, 12, 3.0, 0.1)
    return panel, readings, limits, cut, heavy_up


@pytest.mark.parametrize("cost", [Gross(), Net(1.0)], ids=["gross", "net"])
def test_the_costs_halves_and_ranking_are_the_arithmetic_stated(cost: Gross | Net) -> None:
    panel, readings, limits, cut, heavy_up = _cheap()
    world = QUADRATURE["world"]
    revenue, uses = 2.0, 3.0
    values = value_of_tests(
        panel,
        NATIONAL,
        readings,
        [cut, heavy_up],
        world["budget"],
        world["periods"],
        limits,
        seed=5,
        cost=cost,
        draws=40,
        level=0.8,
        revenue_per_kpi=revenue,
        uses=uses,
        grid=64,
    )
    assert (values.draws, values.level) == (40, 0.8)
    outer = _alike(readings, 40)
    fit, chain = readings.fit[outer][0::2], readings.chain[outer][0::2]
    for value, candidate in zip(values.values, (cut, heavy_up), strict=True):
        assert value.candidates == (candidate,)
        change = float(np.sum(candidate.spend - candidate.planned))
        lift = float(readings.weights @ predicted_lift(readings, candidate))
        opportunity = -revenue * candidate.share * lift
        assert value.spend_change == pytest.approx(change, rel=1e-15, abs=0.0)
        assert value.opportunity == pytest.approx(opportunity, rel=1e-14, abs=0.0)
        assert value.gross == max(value.spend_change, 0.0) + max(value.opportunity, 0.0)
        assert value.net == value.spend_change + value.opportunity
        assert value.cost == (value.gross if isinstance(cost, Gross) else max(value.net, 1.0))
        choice, held_out = value.reductions[0::2], value.reductions[1::2]
        assert value.reduction == float(choice.mean())
        assert value.error == _error(choice, fit, chain)
        assert value.predicted == float(held_out.mean())
        central = [(1.0 - values.level) / 2.0, (1.0 + values.level) / 2.0]
        assert value.interval == tuple(np.quantile(held_out, central).tolist())
        assert value.per_cost == revenue * value.reduction / value.cost
        assert value.net_value == uses * (revenue * value.reduction) - value.net
        assert value.predicted_net_value == uses * revenue * value.predicted - value.net
        assert value.fallback == float(np.mean(value.ess < 100.0))
    assert values.values[0].opportunity > 0.0 > values.values[0].spend_change
    assert values.values[1].spend_change > 0.0 > values.values[1].opportunity
    assert values.ranked == tuple(sorted(values.values, key=lambda v: -v.per_cost))


def test_the_monte_carlo_error_reads_each_fits_autocorrelation_and_not_the_gap_between_fits() -> (
    None
):
    """Two fits of two chains of 500: the first's draws a stationary AR(1) of coefficient 0.8 and
    variance 1, its mean's variance nine times its draws' over their count; the second's
    independent, of variance 1 about a mean 10 apart. Over 400 replications the error is the
    exact `sqrt(1000 * 9 + 1000) / 2000 = 0.05` on average, within 5 %, and the spread of the means
    is it within its sampling error; the first's autocorrelation ignored it would read 0.022, and
    the gap between the fits read as spread, 0.11."""
    rng = np.random.default_rng(14)
    fit = np.repeat([0, 1], 1000)
    chain = np.tile(np.repeat([0, 1], 500), 2)
    rho = 0.8
    errors, means = [], []
    for _ in range(400):
        values = np.r_[_ar(rng, rho, 500).ravel(), 10.0 + rng.standard_normal(1000)]
        errors.append(_error(values, fit, chain))
        means.append(values.mean())
    exact = math.sqrt(1000 * (1 + rho) / (1 - rho) + 1000) / 2000
    assert np.mean(errors) == pytest.approx(exact, rel=0.05)
    assert np.std(means, ddof=1) == pytest.approx(exact, rel=4.5 / math.sqrt(2 * 399))


@pytest.mark.parametrize("then", [Reweight(), Gaussian()], ids=["reweight", "gaussian"])
def test_a_refit_reads_each_tests_estimate_at_evenly_spaced_draws_of_each_half(
    then: Reweight | Gaussian, caplog: pytest.LogCaptureFixture
) -> None:
    """`Refit(3, fit)` on the cheap world over 40 outer draws: `fit` is called at outer draws 0, 20
    and 38 of the even half and 1, 21 and 39 of the odd, for each candidate in turn, with its
    estimate of the reading the update read there. A fit that returns the readings it was given
    plans as every draw does, so its reductions there are that plan's worth less the worth of the
    plan every update is scored against, and NaN elsewhere; the update's reductions and
    diagnostics are its `then`'s, and each half's mean, its error and the interval are corrected by
    its refits, which a log line reports."""
    panel, readings, limits, cut, heavy_up = _cheap()
    world = QUADRATURE["world"]
    calls: list[Mapping[Channel, Sequence[LiftEstimate]]] = []

    def fit(estimates: Mapping[Channel, Sequence[LiftEstimate]]) -> Readings:
        calls.append(estimates)
        return readings

    def read(update: Reweight | Gaussian | Refit) -> tuple[ExperimentValue, ...]:
        return value_of_tests(
            panel,
            NATIONAL,
            readings,
            [cut, heavy_up],
            world["budget"],
            world["periods"],
            limits,
            seed=5,
            draws=40,
            grid=64,
            update=update,
        ).values

    plain = read(then)
    with caplog.at_level(logging.INFO, logger="mixdyn.experiments.value"):
        refit = read(Refit(3, fit, then=then))
    outer = _alike(readings, 40)
    z = np.random.default_rng(np.random.SeedSequence(5).spawn(1)[0]).standard_normal(40)
    places = [0, 1, 20, 21, 38, 39]
    assert len(calls) == 2 * len(places)
    for k, candidate in enumerate((cut, heavy_up)):
        lift = predicted_lift(readings, candidate)
        for call, j in zip(calls[6 * k : 6 * k + 6], places, strict=True):
            ((channel, (estimate,)),) = call.items()
            assert channel == candidate.channel
            assert estimate.lift == candidate.share * lift[outer[j]] + candidate.se * z[j]
            assert (estimate.se, estimate.share, estimate.readout) == (
                candidate.se,
                candidate.share,
                candidate.readout,
            )
            np.testing.assert_array_equal(estimate.spend, candidate.spend)
            np.testing.assert_array_equal(estimate.planned, candidate.planned)
    fits, chains = readings.fit[outer], readings.chain[outer]
    prior = _context(
        panel,
        NATIONAL,
        readings,
        [cut, heavy_up],
        world["budget"],
        world["periods"],
        limits,
        "value_of_tests",
        seed=5,
        update=then,
        draws=40,
        level=0.9,
        revenue_per_kpi=1.0,
        uses=1.0,
        grid=64,
    ).prior
    expected = np.full(40, np.nan)
    expected[places] = [
        prior.planner.worth(int(outer[j]), prior.plan[None])[0] - prior.worth[j] for j in places
    ]
    for value, before in zip(refit, plain, strict=True):
        np.testing.assert_array_equal(value.reductions, before.reductions)
        np.testing.assert_array_equal(value.refitted, expected)
        np.testing.assert_array_equal(value.ess, before.ess)
        np.testing.assert_array_equal(value.pareto_k, before.pareto_k)
        np.testing.assert_equal(value.fallback, before.fallback)
        choice, held_out = value.reductions[0::2], value.reductions[1::2]
        even, odd = value.refitted[0::2][[0, 10, 19]], value.refitted[1::2][[0, 10, 19]]
        assert value.reduction == float(choice.mean()) + float(np.mean(even - choice[[0, 10, 19]]))
        assert (value.reduction, value.error) == _corrected(
            choice, value.refitted[0::2], fits[0::2], chains[0::2]
        )[:2]
        predicted, _, shift = _corrected(held_out, value.refitted[1::2], fits[1::2], chains[1::2])
        assert shift == float(np.mean(odd - held_out[[0, 10, 19]]))
        assert value.predicted == predicted
        central = np.quantile(held_out, [(1.0 - 0.9) / 2.0, (1.0 + 0.9) / 2.0]) + shift
        assert value.interval == tuple(central.tolist())
        assert value.predicted_net_value == value.predicted - value.net
        channels = [str(c.channel) for c in value.candidates]
        assert (
            f"6 refits of the tests of {channels} moved their reduction by "
            f"{value.reduction - float(choice.mean()):.3g}, its error {value.error:.3g}, and their "
            f"prediction by {shift:.3g}"
        ) in caplog.messages


def test_a_refit_of_a_set_reads_its_tests_estimates_by_channel_in_the_candidates_order() -> None:
    """`choose_tests` with `Refit(2, fit)` over a heavy-up, the go-dark and the heavy-up again on
    the cheap world, every set affordable: `fit` is called at outer draws 0 and 38 of the even half
    and 1 and 39 of the odd for each set, by their sizes and places, with each test's estimate
    under its channel, in the candidates' order, each read with its own column of normals. The fit
    returns the first chain of each fit, whose own draws and weights its plan reads."""
    panel, readings, limits, cut, heavy_up = _cheap()
    world = QUADRATURE["world"]
    candidates = (heavy_up, cut, heavy_up)
    calls: list[Mapping[Channel, Sequence[LiftEstimate]]] = []
    first = readings.chain == 0
    refitted = Readings(
        channels=readings.channels,
        retention=readings.retention[first],
        half_saturation=readings.half_saturation[first],
        slope=readings.slope[first],
        coefficient=readings.coefficient[first],
        tanh=readings.tanh[first],
        fit=readings.fit[first],
        chain=readings.chain[first],
        stacking=readings.stacking,
        pointwise=readings.pointwise,
        length=readings.length,
    )

    def fit(estimates: Mapping[Channel, Sequence[LiftEstimate]]) -> Readings:
        calls.append(estimates)
        return refitted

    choice = choose_tests(
        panel,
        NATIONAL,
        readings,
        candidates,
        world["budget"],
        world["periods"],
        limits,
        test_budget=1e6,
        seed=5,
        draws=40,
        grid=64,
        update=Refit(2, fit, then=Gaussian()),
    )
    sets = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
    places = [0, 1, 38, 39]
    assert len(calls) == len(sets) * len(places)
    outer = _alike(readings, 40)
    z = np.random.default_rng(np.random.SeedSequence(5).spawn(1)[0]).standard_normal((40, 3))
    lifts = [predicted_lift(readings, c) for c in candidates]
    for s, chosen in enumerate(sets):
        for call, j in zip(calls[4 * s : 4 * s + 4], places, strict=True):
            expected: dict[Channel, list[float]] = {}
            for t in chosen:
                c = candidates[t]
                reading = c.share * lifts[t][outer[j]] + c.se * z[j, t]
                expected.setdefault(c.channel, []).append(float(reading))
            assert {channel: [e.lift for e in read] for channel, read in call.items()} == expected
    assert choice.best is not None
    read = ~np.isnan(choice.best.refitted)
    assert np.flatnonzero(read).tolist() == places
    assert np.all(np.isfinite(choice.best.refitted[read]))


def test_a_refit_may_take_every_outer_draw_of_a_half_and_then_reads_the_refits_alone() -> None:
    """`Refit(4, fit)` over 8 outer draws refits every one, so each half's corrected mean is its
    refits' mean, whatever the update read there."""
    call = _refused({})
    readings = call["readings"]
    call["update"] = Refit(4, lambda _: readings)
    (value,) = value_of_tests(**call).values
    assert not np.isnan(value.refitted).any()
    assert np.any(value.reductions != value.refitted)
    for mean, refits in (
        (value.reduction, value.refitted[0::2]),
        (value.predicted, value.refitted[1::2]),
    ):
        assert mean == pytest.approx(float(refits.mean()), rel=1e-12, abs=1e-12)


def test_a_refits_halves_correct_their_own_means_and_the_interval_moves_with_the_odds() -> None:
    """Under `Refit`, the reduction and its error are the even half's corrected by its own refits,
    the prediction the odd half's by theirs, and the interval moves by the odd half's correction;
    the fallback is `then`'s, NaN under `Gaussian`. The halves' outer draws are of different fits:
    the even half's of one, the odd half's of two."""
    _, _, _, cut, _ = _cheap()
    rng = np.random.default_rng(43)
    reductions = rng.normal(1.0, 1.0, 40)
    refitted = np.full(40, np.nan)
    refitted[[0, 1, 20, 21, 38, 39]] = rng.normal(2.0, 1.0, 6)
    fit, chain = (np.arange(40) % 4 == 1).astype(np.intp), np.repeat(np.arange(2), 20)
    updated = _Updated(
        np.zeros((40, 2)), np.full(40, np.nan), np.full(40, np.nan), np.zeros(40, dtype=np.bool_)
    )
    value = _set_value(
        (cut,),
        (0.5,),
        reductions,
        (fit, chain),
        updated,
        Refit(3, _unread, then=Gaussian()),
        math.nan,
        refitted,
        Gross(),
        0.9,
        1.0,
        1.0,
    )
    choice, held_out = reductions[0::2], reductions[1::2]
    assert (value.reduction, value.error) == _corrected(
        choice, refitted[0::2], fit[0::2], chain[0::2]
    )[:2]
    predicted, _, shift = _corrected(held_out, refitted[1::2], fit[1::2], chain[1::2])
    assert value.predicted == predicted
    central = np.quantile(held_out, [(1.0 - 0.9) / 2.0, (1.0 + 0.9) / 2.0]) + shift
    assert value.interval == tuple(central.tolist())
    assert value.predicted_net_value == value.predicted - value.net
    np.testing.assert_array_equal(value.refitted, refitted)
    assert math.isnan(value.fallback)


def _ar(rng: np.random.Generator, rho: float, draws: int) -> npt.NDArray[np.float64]:
    """Two stationary AR(1) chains of coefficient `rho` and variance 1, `(2, draws)`."""
    shocks = rng.standard_normal((2, draws)) * math.sqrt(1.0 - rho**2)
    ar = np.empty((2, draws))
    ar[:, 0] = rng.standard_normal(2)
    for t in range(1, draws):
        ar[:, t] = rho * ar[:, t - 1] + shocks[:, t]
    return ar


def test_an_antithetic_fit_reads_as_independent_draws_and_a_lone_outer_draw_as_nothing() -> None:
    """Chains that alternate about their mean hold more than their count of effective draws, but
    the relative efficiency is at most 1, so the error reads them as independent draws, never as
    more; and a fit of one outer draw, whose spread cannot be read, adds nothing."""
    ar = _ar(np.random.default_rng(19), -0.8, 50)
    assert _ess(_split(ar)) > ar.size
    values = np.r_[ar.ravel(), 3.0]
    fit = np.r_[np.zeros(100, dtype=np.intp), 1]
    chain = np.r_[np.repeat([0, 1], 50), 0]
    assert _error(values, fit, chain) == math.sqrt(100 * np.var(ar.ravel(), ddof=1)) / 101


def test_every_candidate_reads_the_same_normals_and_a_seed_its_own() -> None:
    panel, readings, limits, cut, _ = _cheap()
    world = QUADRATURE["world"]

    def read(seed: int) -> list[npt.NDArray[np.float64]]:
        values = value_of_tests(
            panel,
            NATIONAL,
            readings,
            [cut, cut],
            world["budget"],
            world["periods"],
            limits,
            seed=seed,
            draws=20,
            grid=64,
            update=Gaussian(),
        )
        return [value.reductions for value in values.values]

    first, again = read(5), read(5)
    np.testing.assert_array_equal(first[0], first[1])
    np.testing.assert_array_equal(first[0], again[0])
    assert not np.array_equal(first[0], read(6)[0])


def test_the_normals_come_from_a_stream_spawned_from_the_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not from the seed's own stream, whose normals a world simulated from the same seed reads."""
    panel, readings, limits, cut, _ = _cheap()
    world = QUADRATURE["world"]
    seen: list[npt.NDArray[np.float64]] = []

    def spy(*arguments: Any) -> Any:
        seen.append(arguments[2].noise[:, 0])
        return _updated(*arguments)

    monkeypatch.setattr("mixdyn.experiments.value._updated", spy)
    value_of_tests(
        panel,
        NATIONAL,
        readings,
        [cut],
        world["budget"],
        world["periods"],
        limits,
        seed=5,
        draws=20,
        grid=64,
        update=Gaussian(),
    )
    (z,) = seen
    child = np.random.SeedSequence(5).spawn(1)[0]
    np.testing.assert_array_equal(z, np.random.default_rng(child).standard_normal(20))
    assert not np.any(z == np.random.default_rng(5).standard_normal(20))


def _two_fits_of_the_cheap_world() -> tuple[
    MediaPanel, Readings, dict[Channel, Range], Candidate, Candidate
]:
    """The cheap world's draws as two fits weighed 0.3 and 0.7, the first channel's half-saturation
    varying over the draws, with its go-dark and heavy-up."""
    panel, readings, limits, cut, heavy_up = _cheap()
    count = readings.fit.size
    jitter = 1.0 + 0.1 * np.random.default_rng(21).standard_normal(count)
    two = Readings(
        channels=CHANNELS,
        retention=readings.retention,
        half_saturation=readings.half_saturation * np.column_stack([jitter, np.ones(count)]),
        slope=readings.slope,
        coefficient=readings.coefficient,
        tanh=readings.tanh,
        fit=np.repeat([0, 1], count // 2),
        chain=np.tile(np.repeat([0, 1], count // 4), 2),
        stacking=np.array([0.3, 0.7]),
        pointwise=None,
        length=readings.length,
    )
    return panel, two, limits, cut, heavy_up


def _local_model(panel: MediaPanel, two: Readings, tests: list[Candidate]) -> float:
    """The local model's reduction of `tests` read together, each fit's term half the trace of CHC's
    decision weight at the fit's median channels over the covariance of the channels' parameters
    with the lifts, through the inverse of the lifts' predictive covariance."""
    world = QUADRATURE["world"]
    scale = two.half_saturation / math.atanh(0.5)
    lifts = np.column_stack([predicted_lift(two, c) for c in tests])
    noise = np.diag([(c.se / c.share) ** 2 for c in tests])
    expected = 0.0
    for k, share in enumerate(two.stacking):
        held = two.fit == k
        channels = [
            response.Channel(
                response.GeometricAdstock(r, length=world["length"], normalized=True),
                response.Tanh(float(np.median(scale[held, c]))),
                float(np.median(two.coefficient[held, c])),
            )
            for c, r in enumerate(world["retention"])
        ]
        weight = decision_weight(
            channels,
            world["budget"],
            world["periods"],
            lower=world["lower"],
            upper=world["upper"],
            history=panel.spend[0],
        )
        leaves = {}
        for c in range(2):
            leaves[f"{c}.kernel.retention"] = two.retention[held, c]
            leaves[f"{c}.curve.scale"] = scale[held, c]
            leaves[f"{c}.coefficient"] = two.coefficient[held, c]
        parameters = np.column_stack([leaves[name] for name in weight.parameters])
        mu = lifts[held]
        covariance = (parameters - parameters.mean(axis=0)).T @ (mu - mu.mean(axis=0)) / mu.shape[0]
        spread = np.cov(mu, rowvar=False, bias=True).reshape(noise.shape) + noise
        reach = weight.matrix @ covariance @ np.linalg.inv(spread) @ covariance.T
        expected += float(share) * 0.5 * float(np.trace(reach))
    return expected


def test_the_local_model_reads_chcs_decision_weight_at_each_fits_median_channels() -> None:
    """Over the two fits, each fit's term is half the covariance of its channels' parameters with
    the lift, a `tanh`'s scale the one at which CHC's curve reads 1/2 at the half-saturation,
    through CHC's decision weight at the fit's median channels, over the lift's predictive
    variance; the local model sums the terms by the fits' weights, and misses the reduction by its
    stated gap."""
    panel, two, limits, cut, _ = _two_fits_of_the_cheap_world()
    world = QUADRATURE["world"]
    values = value_of_tests(
        panel,
        NATIONAL,
        two,
        [cut],
        world["budget"],
        world["periods"],
        limits,
        seed=5,
        draws=8,
        grid=64,
        update=Gaussian(),
    )
    half = two.half_saturation[0, 0]
    at_half = response.Tanh(half / math.atanh(0.5))(jnp.asarray(half))
    assert float(at_half) == pytest.approx(0.5, rel=1e-15, abs=0.0)
    (value,) = values.values
    assert value.local == pytest.approx(_local_model(panel, two, [cut]), rel=1e-9)
    assert value.model_gap == pytest.approx(abs(value.local / value.reduction - 1.0), rel=1e-15)


def test_the_local_model_of_two_tests_reads_their_lifts_predictive_covariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The go-dark and the heavy-up read together: the lifts' predictive covariance, their
    covariance over the draws and each test's noise, inverted between the parameters' covariances
    with the two lifts."""
    panel, two, limits, cut, heavy_up = _two_fits_of_the_cheap_world()
    world = QUADRATURE["world"]
    made: list[ExperimentValue] = []

    def value(*arguments: Any) -> ExperimentValue:
        made.append(_set_value(*arguments))
        return made[-1]

    monkeypatch.setattr("mixdyn.experiments.value._value", value)
    choose_tests(
        panel,
        NATIONAL,
        two,
        [cut, heavy_up],
        world["budget"],
        world["periods"],
        limits,
        test_budget=1e9,
        seed=5,
        draws=8,
        grid=64,
        update=Gaussian(),
    )
    (pair,) = [v for v in made if len(v.candidates) == 2]
    alone = _local_model(panel, two, [cut])
    assert pair.local == pytest.approx(_local_model(panel, two, [cut, heavy_up]), rel=1e-9)
    assert pair.local > alone


def test_the_local_model_is_not_read_on_an_s_shaped_curve() -> None:
    panel, readings, limits, cut, _ = _cheap()
    world = QUADRATURE["world"]
    hill = Readings(
        channels=CHANNELS,
        retention=readings.retention,
        half_saturation=readings.half_saturation,
        slope=np.full(readings.retention.shape, 2.0),
        coefficient=readings.coefficient,
        tanh=np.zeros(readings.fit.size, dtype=bool),
        fit=readings.fit,
        chain=readings.chain,
        stacking=readings.stacking,
        pointwise=None,
        length=readings.length,
    )
    values = value_of_tests(
        panel,
        NATIONAL,
        hill,
        [cut],
        world["budget"],
        world["periods"],
        limits,
        seed=5,
        draws=8,
        grid=64,
        update=Gaussian(),
    )
    (value,) = values.values
    assert math.isnan(value.local)
    assert math.isnan(value.model_gap)


@settings(max_examples=25, deadline=None)
@given(
    st.integers(2, 3).flatmap(
        lambda p: st.tuples(
            st.just(p),
            st.lists(st.floats(-3.0, 3.0), min_size=p, max_size=p),
            st.lists(st.floats(0.2, 3.0), min_size=p, max_size=p),
            st.lists(st.floats(-2.0, 2.0), min_size=p, max_size=p),
            st.floats(0.05, 4.0),
            st.integers(0, 2**32 - 1),
        )
    ),
    st.sampled_from([Gaussian(), Reweight()]),
)
def test_no_outer_draw_is_worth_more_than_perfect_information(
    instance: tuple[Any, ...], update: Gaussian | Reweight
) -> None:
    """On conjugate instances of two or three channels, the plan after any outcome scores no more
    at the outer draw than the plan of the draw itself, the best there is."""
    p, mean, curvature, direction, se, seed = instance
    rng = np.random.default_rng(seed)
    root = rng.normal(size=(p, p))
    covariance = root @ root.T + 0.1 * np.eye(p)
    theta = rng.multivariate_normal(mean, covariance, size=400)
    inverse = np.diag(1.0 / np.array(curvature))
    ones = np.ones(p)
    share = inverse @ ones / (ones @ inverse @ ones)
    project = inverse - np.outer(inverse @ ones, ones @ inverse) / (ones @ inverse @ ones)
    planner = _Planner(
        theta,
        lambda m: project @ m + share,
        lambda i, plans: plans @ theta[i] - 0.5 * np.sum(plans**2 * np.array(curvature), axis=1),
    )
    draws = _Chains(np.zeros(400, dtype=np.intp), np.repeat([0, 1], 200), np.ones(1), None)
    lift = theta @ np.array(direction)
    outer = np.arange(0, 400, 8)
    z = rng.standard_normal(outer.size)
    prior = _preposterior(planner, draws, outer, update)
    (reductions,), _ = prior.reductions([_one(lift, 1.0, se, z)])
    perfect = prior.perfect()
    scale = 1.0 + np.max(np.abs(perfect))
    assert np.all(reductions <= perfect + 1e-12 * scale)


def _unread(estimates: Mapping[Channel, Sequence[LiftEstimate]]) -> Readings:
    raise AssertionError(f"a refused call refitted {dict(estimates)}")


def _refused(change: dict[str, Any]) -> dict[str, Any]:
    panel, readings, limits, cut, _ = _cheap()
    world = QUADRATURE["world"]
    call: dict[str, Any] = {
        "panel": panel,
        "geo": NATIONAL,
        "readings": readings,
        "candidates": [cut],
        "budget": world["budget"],
        "periods": world["periods"],
        "limits": limits,
        "seed": 5,
        "draws": 8,
        "grid": 64,
    }
    if change.pop("one chain", False):
        call["readings"] = Readings(
            channels=CHANNELS,
            retention=readings.retention,
            half_saturation=readings.half_saturation,
            slope=readings.slope,
            coefficient=readings.coefficient,
            tanh=readings.tanh,
            fit=readings.fit,
            chain=np.zeros(readings.fit.size, dtype=np.intp),
            stacking=readings.stacking,
            pointwise=None,
            length=readings.length,
        )
    if change.pop("radio", False):
        call["candidates"] = [Candidate(Channel("radio"), cut.spend, cut.planned, 12, 3.0, 0.2)]
    if change.pop("refit of other channels", False):
        other = Readings(
            channels=CHANNELS[::-1],
            retention=readings.retention[:, ::-1],
            half_saturation=readings.half_saturation[:, ::-1],
            slope=readings.slope[:, ::-1],
            coefficient=readings.coefficient[:, ::-1],
            tanh=readings.tanh,
            fit=readings.fit,
            chain=readings.chain,
            stacking=readings.stacking,
            pointwise=readings.pointwise,
            length=readings.length,
        )
        call["update"] = Refit(2, lambda _: other)
    return call | change


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"candidates": []}, "value_of_tests needs candidates"),
        ({"draws": 3}, "draws must be a whole number, at least 4, got 3"),
        ({"draws": True}, "draws must be a whole number, at least 4, got True"),
        ({"draws": 121}, r"fits \[0\] hold \[120\] draws, fewer than their shares of 121"),
        ({"seed": -1}, "seed must be a whole number, at least 0, got -1"),
        ({"level": 1.0}, r"level=1.0 is not in \(0, 1\)"),
        ({"level": float("nan")}, r"level=nan is not in \(0, 1\)"),
        ({"revenue_per_kpi": 0.0}, "revenue_per_kpi=0.0 is not positive and finite"),
        ({"uses": float("inf")}, "uses=inf is not positive and finite"),
        ({"one chain": True}, r"draws of two chains or more: fits \[0\] have one"),
        ({"radio": True}, "the readings have no channel 'radio'"),
        ({"budget": 2000.0}, "a budget of 2000.0 is outside what the ranges spend"),
        ({"grid": 1}, "grid must be a whole number, at least 2, got 1"),
        ({"update": Refit(5, _unread)}, "Refit.draws=5 is more than the 4 outer draws of a half"),
        (
            {"refit of other channels": True},
            r"a refit's readings have the channels \['b', 'a'\], not the readings' \['a', 'b'\]",
        ),
    ],
)
def test_value_of_tests_refuses_what_it_cannot_read(change: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        value_of_tests(**_refused(dict(change)))


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"candidates": []}, "choose_tests needs candidates"),
        ({"test_budget": -1.0}, "test_budget=-1.0 is not finite and at least 0"),
        ({"test_budget": float("inf")}, "test_budget=inf is not finite and at least 0"),
        ({"test_budget": float("nan")}, "test_budget=nan is not finite and at least 0"),
        ({"exhaustive": -1}, "exhaustive must be a whole number, at least 0, got -1"),
        ({"exhaustive": 2.5}, "exhaustive must be a whole number, at least 0, got 2.5"),
        ({"draws": 3}, "draws must be a whole number, at least 4, got 3"),
    ],
)
def test_choose_tests_refuses_what_it_cannot_read(change: dict[str, Any], match: str) -> None:
    call: dict[str, Any] = {"test_budget": 10.0}
    call.update(change)
    with pytest.raises(ValueError, match=match):
        choose_tests(**_refused(call))


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: Net(0.0), r"floor=0.0 is not positive and finite"),
        (lambda: Net(float("inf")), r"floor=inf is not positive and finite"),
        (lambda: Reweight(0.0), r"ess=0.0 is not positive and finite"),
        (lambda: Reweight(float("nan")), r"ess=nan is not positive and finite"),
        (lambda: Refit(1, _unread), "Refit.draws must be a whole number, at least 2, got 1"),
        (lambda: Refit(True, _unread), "Refit.draws must be a whole number, at least 2, got True"),
    ],
)
def test_a_cost_or_an_update_refuses_a_threshold_it_cannot_hold(make, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        make()


def _three() -> tuple[MediaPanel, Readings, dict[Channel, Range], tuple[Candidate, ...]]:
    """The cheap world's go-dark and heavy-up, and a go-dark of the second channel's same four weeks
    in three tenths of the market: each costs more than the regret it removes is worth."""
    panel, readings, limits, cut, heavy_up = _cheap()
    market = panel.spend[0, :, 1]
    dark = market.copy()
    dark[-12:-8] = 0.0
    return (
        panel,
        readings,
        limits,
        (cut, heavy_up, Candidate(CHANNELS[1], 0.3 * dark, 0.3 * market, 12, 3.0, 0.3)),
    )


def _choose(
    candidates: tuple[Candidate, ...], test_budget: float, uses: float, exhaustive: int = 256
) -> ExperimentChoice:
    panel, readings, limits, _ = _three()
    world = QUADRATURE["world"]
    return choose_tests(
        panel,
        NATIONAL,
        readings,
        candidates,
        world["budget"],
        world["periods"],
        limits,
        test_budget=test_budget,
        seed=5,
        draws=120,
        grid=64,
        uses=uses,
        exhaustive=exhaustive,
    )


def test_choose_tests_values_every_affordable_set_or_grows_one_each_test_on_its_own_normals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under a test budget of 60, three tests of gross costs near 38, 13 and 52 and shares 0.2, 0.1
    and 0.3 make four affordable sets of the seven: every one is valued, the choice holds them in
    the order valued, and the best and the best single are the largest net values among them.
    Grown greedily, the set chosen is worth no more than the best of all; with no test budget it
    grows from the third test, and the choice holds the single tests, then the pairs with it. Each
    test reads its own column of normals from the stream spawned from the seed, the same in every
    set it is in."""
    _, _, _, candidates = _three()
    made: list[ExperimentValue] = []
    noises: dict[float, list[npt.NDArray[np.float64]]] = {}

    def value(*arguments: Any) -> ExperimentValue:
        made.append(_set_value(*arguments))
        return made[-1]

    def updated(*arguments: Any) -> Any:
        tests = arguments[2]
        for t, share in enumerate(tests.share.tolist()):
            noises.setdefault(share, []).append(tests.noise[:, t])
        return _updated(*arguments)

    monkeypatch.setattr("mixdyn.experiments.value._value", value)
    monkeypatch.setattr("mixdyn.experiments.value._updated", updated)

    def places(chosen: ExperimentValue) -> tuple[int, ...]:
        return tuple(next(t for t, c in enumerate(candidates) if c is d) for d in chosen.candidates)

    every = _choose(candidates, 60.0, 1000.0)
    assert every.exhaustive
    assert [places(v) for v in made] == [(0,), (1,), (2,), (0, 1)]
    assert all(v is m for v, m in zip(every.values, made, strict=True))
    for v in made:
        assert sum(c.share for c in v.candidates) <= 0.5
        assert v.cost <= 60.0
    assert every.best is max(made, key=lambda v: v.net_value)
    assert every.single is max(made[:3], key=lambda v: v.net_value)
    z = np.random.default_rng(np.random.SeedSequence(5).spawn(1)[0]).standard_normal((120, 3))
    for t, candidate in enumerate(candidates):
        for column in noises[candidate.share]:
            np.testing.assert_array_equal(column, z[:, t])
    made.clear()
    grown = _choose(candidates, 60.0, 1000.0, exhaustive=0)
    assert not grown.exhaustive
    assert all(v is m for v, m in zip(grown.values, made, strict=True))
    assert grown.best is not None
    assert every.best is not None
    assert grown.best.net_value <= every.best.net_value
    assert places(grown.best) in [(0,), (1,), (2,), (0, 1)]
    made.clear()
    wide = _choose(candidates, 1e9, 1000.0, exhaustive=0)
    assert [places(v) for v in made] == [(0,), (1,), (2,), (0, 2), (1, 2)]
    assert all(v is m for v, m in zip(wide.values, made, strict=True))


@pytest.mark.parametrize(
    ("worth", "costs", "shares", "asked", "kept"),
    [
        pytest.param(
            {(0,): 5.0, (1,): 4.0, (2,): 1.0, (0, 1): 7.0, (0, 2): 5.5, (0, 1, 2): 7.2},
            [1.0, 1.0, 1.0],
            [0.1, 0.1, 0.1],
            [(0,), (1,), (2,), (0, 1), (0, 2), (0, 1, 2)],
            (0, 1),
            id="grows-while-it-pays",
        ),
        pytest.param(
            {(0,): 5.0, (1,): 4.0, (2,): 3.0, (0, 2): 6.5},
            [1.0, 1.0, 1.0],
            [0.3, 0.3, 0.1],
            [(0,), (1,), (2,), (0, 2)],
            (0, 2),
            id="half-the-market",
        ),
        pytest.param(
            {(0,): 5.0, (1,): 4.0},
            [1.0, 4.0],
            [0.1, 0.1],
            [(0,), (1,)],
            (0,),
            id="the-test-budget",
        ),
        pytest.param(
            {(0,): 2.0, (1,): 4.0, (2,): 0.5, (0, 1): 5.5, (1, 2): 4.6, (0, 1, 2): 5.9},
            [1.0, 2.0, 1.0],
            [0.1, 0.1, 0.1],
            [(0,), (1,), (2,), (0, 1), (1, 2), (0, 1, 2)],
            (0, 1),
            id="the-larger-addition-on-a-tie",
        ),
        pytest.param(
            {(0,): 2.0, (1,): 10.0, (0, 1): 10.2},
            [0.5, 3.0],
            [0.1, 0.1],
            [(0,), (1,), (0, 1)],
            (1,),
            id="the-best-single-test-where-it-is-worth-more",
        ),
        pytest.param(
            {(0,): 0.5, (1,): 0.5},
            [1.0, 1.0],
            [0.1, 0.1],
            [(0,), (1,)],
            (0,),
            id="no-test-pays-alone",
        ),
        pytest.param(
            {(0,): 3.0, (1,): 1.0, (0, 1): 4.0},
            [1.0, 1.0],
            [0.1, 0.1],
            [(0,), (1,), (0, 1)],
            (0,),
            id="an-addition-worth-its-cost-exactly-is-not-made",
        ),
        pytest.param(
            {(0,): 5.0, (1,): 1.0, (2,): 1.0, (0, 1): 7.0, (0, 2): 8.0, (0, 1, 2): 8.5},
            [1.0, 1.0, 2.0],
            [0.1, 0.1, 0.1],
            [(0,), (1,), (2,), (0, 1), (0, 2), (0, 1, 2)],
            (0, 1),
            id="the-cost-an-addition-adds",
        ),
    ],
)
def test_the_greedy_set_grows_by_the_most_value_a_unit_of_cost_while_it_pays(
    worth: dict[tuple[int, ...], float],
    costs: list[float],
    shares: list[float],
    asked: list[tuple[int, ...]],
    kept: tuple[int, ...],
) -> None:
    """Sets worth `worth`, each test costing `costs` gross and net, under a test budget of 4: the
    sets valued in turn, and the one kept. A pair worth only together is never reached: no test
    pays alone, so no set grows."""
    priced = [(c, 0.0, c, c) for c in costs]
    made: dict[int, tuple[int, ...]] = {}
    seen: list[tuple[int, ...]] = []

    def valued(sets: Any) -> list[ExperimentValue]:
        out = []
        for chosen in sets:
            seen.append(chosen)
            out.append(_stub(worth[chosen], math.fsum(costs[t] for t in chosen)))
            made[id(out[-1])] = chosen
        return out

    best, singles = _greedy(valued, priced, shares, Gross(), 4.0, 1.0, 1.0)
    assert seen == asked
    assert best is not None
    assert made[id(best)] == kept
    assert [made[id(v)] for v in singles] == [s for s in asked if len(s) == 1]


def _stub(reduction: float, cost: float, predicted: float | None = None) -> ExperimentValue:
    """A set's value as the choice reads it, of `uses` and `revenue_per_kpi` 1: its reduction, its
    prediction, and a cost, gross and net alike."""
    nan = math.nan
    held = reduction if predicted is None else predicted
    return ExperimentValue(
        candidates=(),
        reduction=reduction,
        error=nan,
        predicted=held,
        interval=(nan, nan),
        reductions=np.empty(0),
        spend_change=cost,
        opportunity=0.0,
        gross=cost,
        net=cost,
        cost=cost,
        per_cost=reduction / cost if cost > 0.0 else math.inf,
        net_value=reduction - cost,
        predicted_net_value=held - cost,
        local=nan,
        model_gap=nan,
        ess=np.empty(0),
        pareto_k=np.empty(0),
        fallback=nan,
        refitted=np.empty(0),
    )


_PRICE = st.tuples(st.sampled_from([-3.0, -1.0, 0.0, 0.5, 2.5]), st.sampled_from([-2.0, 0.0, 1.5]))


@settings(max_examples=300, deadline=None)
@given(
    st.lists(
        st.tuples(st.sampled_from([0.05, 0.1, 0.17, 0.2, 0.28, 0.3]), _PRICE),
        min_size=1,
        max_size=7,
    ),
    st.sampled_from([Gross(), Net(0.5)]),
    st.sampled_from([0.0, 1.0, 2.5, 4.0, 10.0]),
    st.integers(0, 40),
)
def test_the_affordable_sets_are_every_set_within_half_the_market_and_the_test_budget(
    tests: list[tuple[float, tuple[float, float]]],
    cost: Gross | Net,
    test_budget: float,
    limit: int,
) -> None:
    """Every set of tests, each a spend change and an opportunity priced as `value_of_tests` prices
    them, whose shares sum to at most 1/2 and whose cost is at most the test budget, by their
    sizes and places; none where there are more than `limit`. Tests of negative net cost let a set
    past the budget grow back into it."""
    shares = [share for share, _ in tests]
    priced = [(a, b, max(a, 0.0) + max(b, 0.0), a + b) for _, (a, b) in tests]

    def charged(chosen: tuple[int, ...]) -> float:
        net = math.fsum(priced[t][3] for t in chosen)
        if isinstance(cost, Gross):
            return math.fsum(priced[t][2] for t in chosen)
        return max(net, cost.floor)

    every = [
        chosen
        for size in range(1, len(tests) + 1)
        for chosen in itertools.combinations(range(len(tests)), size)
        if math.fsum(shares[t] for t in chosen) <= 0.5 and charged(chosen) <= test_budget
    ]
    found = _affordable(priced, shares, cost, test_budget, limit)
    assert found == (every if len(every) <= limit else None)
    for chosen in itertools.combinations(range(len(tests)), min(3, len(tests))):
        assert _fits_in(chosen, priced, shares, cost, test_budget) == (chosen in every)


def test_shares_of_005_017_028_take_half_the_market_in_any_order() -> None:
    """Summed in turn, they come to 0.5 in one order and past it in another."""
    shares = [0.05, 0.17, 0.28]
    assert len({(a + b) + c for a, b, c in itertools.permutations(shares)}) == 2
    priced = [(0.0, 0.0, 0.0, 0.0)] * 3
    for order in itertools.permutations(range(3)):
        assert _fits_in(order, priced, shares, Gross(), 0.0)
    assert _affordable(priced, shares, Gross(), 0.0, 10) == [
        (0,),
        (1,),
        (2,),
        (0, 1),
        (0, 2),
        (1, 2),
        (0, 1, 2),
    ]


@pytest.mark.parametrize(
    ("best", "verdict"),
    [
        (None, "plan now"),
        (_stub(3.0, 1.0, predicted=2.0), "test"),
        (_stub(3.0, 1.0, predicted=0.5), "plan now"),
        (_stub(0.5, 1.0, predicted=3.0), "test"),
        (_stub(1.0, 1.0, predicted=1.0), "plan now"),
    ],
    ids=["no-set", "both-halves-pay", "only-the-choices-half-pays", "only-the-other", "even"],
)
def test_the_verdict_is_the_best_sets_predicted_net_value_above_0(
    best: ExperimentValue | None, verdict: str
) -> None:
    """Read on the outer draws at odd places, which no choice read: the largest of noisy net values
    overstates the set it picks."""
    choice = ExperimentChoice(best, best, (), True, np.zeros(2), 0.0, 0.0, 4, 0.9)
    assert choice.verdict == verdict


def test_a_test_worth_less_than_its_cost_is_not_run() -> None:
    """Each test removes some regret, but less than it costs: the best set is the one that loses
    least, the heavy-up alone, and the verdict is to plan now, whatever the test budget. A plan
    made a thousand times on what they read would pay for them; and with no test affordable there
    is nothing to run."""
    _, _, _, candidates = _three()
    once = _choose(candidates, 1e9, 1.0)
    assert once.best is not None
    (chosen,) = once.best.candidates
    assert chosen is candidates[1]
    assert once.best.reduction > 0.0
    assert once.best.net_value < 0.0
    assert once.verdict == "plan now"
    often = _choose(candidates, 1e9, 1000.0)
    assert often.best is not None
    assert often.best.predicted_net_value > 0.0
    assert often.verdict == "test"
    nothing = _choose(candidates, 0.0, 1000.0)
    assert (nothing.best, nothing.single, nothing.values, nothing.exhaustive) == (
        None,
        None,
        (),
        True,
    )
    assert nothing.verdict == "plan now"
