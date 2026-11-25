"""The value of a lift test to the plan, read before it runs, over a posterior's draws.

A test is worth the regret it is expected to remove: over the posterior's draws `θ`, and the lift
`y` the test would read under each,

    ERR = E_θ E_{y|θ} [V(π(p(·|D, y)), θ) - V(a₀, θ)],

`V(a, θ)` the worth of a plan `a` under `θ`, `π` the plan for a posterior's expected worth and `a₀`
its plan without the test. That is the expected value of sample information (Raiffa and Schlaifer
1961), never below 0 and never above that of perfect information, `EVPI = E_θ[max_a V(a, θ) -
V(a₀, θ)]`. The plan is `mixdyn.decisions.plan_posterior`'s: each channel's mean worth on a grid of
spends across its range, and the best split of the means' interpolants.

`value_of_tests` reads it by nested Monte Carlo:

- *Outer draws.* `draws` of the readings' draws, each fit's share of them its weight's by largest
  remainders, evenly spaced over its own: each is a truth `θ_i`. Each takes one standard normal
  `z_i`, shared by every candidate, and a candidate reads `y_i = s μ(θ_i) + se z_i`, `μ` its
  `predicted_lift`, `s` its share and `se` its standard error.
- *Each outer draw's posterior* is the readings' draws but those of `θ_i`'s chain of `θ_i`'s fit:
  `θ_i` drew `y_i`, so it would weigh most in the plan it is then scored on, and chains are
  independent, so the others are independent of it.
- *The update.* `Reweight`, the default, weighs each fit's draws by the test's likelihood,
  Pareto-smoothed (`mixdyn.inference.psis`), and solves the fits' weights again with the test's
  predictive density under each as one more observation, where the readings hold the
  leave-one-out densities they were solved from; weights given stay as given. `Gaussian` moves
  each fit's mean worths by the Bayes-linear update, the covariance of the draws' worths with the
  lift over the lift's predictive variance, exact where the two are jointly normal, and weighs the
  fits by the normal predictive density. Under `Reweight` an outer draw whose weights' effective
  sample size, Kish's times the chains' relative efficiency, falls below `Reweight.ess`, or one of
  whose weighted fits' Pareto shape passes its threshold, is updated as `Gaussian`. A normal
  likelihood is bounded, so the shape stays low however few draws carry the weights: the
  effective sample size is the gate.
- *Refit.* `Refit` corrects its `then`'s estimate by refitting. At `draws` outer draws of each
  half, evenly spaced, the caller's `fit` refits the posterior with the tests' readings `y_i` as
  estimates, and its plan is scored at `θ_i` as above. Each half's mean moves by the mean gap
  between those reductions and `then`'s at the same draws, and its error adds the gap's variance
  over the refits and its covariance with `then`'s reductions: a multi-fidelity estimate, unbiased
  for the plan on a refitted posterior up to the refits' own error. A refit is a fresh fit, so it
  needs no chain left out.
- *Plan and score.* Each outer draw's posterior mean worths are planned, and the plan is scored at
  `θ_i` exactly, against `a₀` of the posterior the update starts from, `θ_i`'s chain left out:
  `R_i = V(a_i, θ_i) - V(a₀_i, θ_i)`. The plan of every draw would have `θ_i`'s own chain behind
  it, so a test that read nothing would lose what that chain gave. The plan of `θ_i`'s worths
  alone scores perfect information.
- *Halves.* The reduction a choice reads is the mean over the outer draws at even places; what a
  chosen test is then expected to remove, and the central `level` interval of its realised
  reduction, are read on those at odd places, which no choice by the first read.
- *Monte Carlo error.* The outer draws are a chain's draws in order, and each fit's count of them
  is fixed: a mean's variance is the sum over the fits of their count's share squared times their
  mean's, each fit's variance over its split-chain effective sample size
  (`mixdyn.inference.diagnostics`), at most its count.
- *Costs*, each an expectation over the posterior: the spend change, `sum(spend - planned)`, and
  the opportunity, the revenue the group is expected to lose, `-κ s E[μ]`, `κ` the revenue a unit
  of the KPI is worth. `Gross` counts each where it is positive, what the test puts at risk; `Net`
  adds them, which a cut of a channel returning less than it costs makes negative, and holds the
  sum at least at its floor. A test's net value is `uses κ ERR` less its net cost, `uses` the
  number of plans made on what it reads.
- *The local model*, reported and not ranked on: `c' W c / (2 (Var μ + (se/s)^2))` at each fit's
  median channels, `W` CHC's decision weight (`chc.allocation.decision_weight`) and `c` the
  covariance of the channels' parameters with `μ` over the fit's draws, summed over the fits by
  their weights. It is blind on a channel held at its bound, and not read on an S-shaped curve,
  where it is NaN; `model_gap` is its relative miss against the reduction, logged above 25 %, as
  `chc.experiment` logs its own.

A candidate's reweighted means are one matrix product a block of outer draws, `(outer, draws)` by
`(draws, channels x grid)`, and a plan an outer draw. The arithmetic is NumPy's, in double
precision whatever JAX's, but for the decision weight, which is CHC's.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np
import numpy.typing as npt
from chc.allocation import decision_weight
from scipy import special

from mixdyn.decisions.posterior import (
    Readings,
    _alike,
    _allocated,
    _fits,
    _inputs,
    _response,
    _spendable,
    _whole,
    _worth,
    worths,
)
from mixdyn.experiments.candidates import Candidate, predicted_lift
from mixdyn.inference.diagnostics import _ess, _split
from mixdyn.inference.loo import _stacking_of_two, psis, stacking
from mixdyn.models.bayes_mmm import _TANH_HALF

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from mixdyn.calibration import LiftEstimate
    from mixdyn.data import Channel, Geo, MediaPanel
    from mixdyn.decisions.geo_plan import Range

log = logging.getLogger(__name__)

type _Array = npt.NDArray[np.float64]
type _Indices = npt.NDArray[np.intp]

_BLOCK = 1 << 23
"""The most elements a block of outer draws' weights holds, `(outer draws, draws)`."""
_GAP_WARNING = 0.25
"""The local model's relative miss above which a call logs a warning, as `chc.experiment`'s."""
_LOG_ROOT_TAU = 0.5 * math.log(2.0 * math.pi)
_SHARE_LIMIT = 0.5
"""The most of the market a set of tests takes between its disjoint groups: the rest is the
control they are read against."""


@dataclass(frozen=True, slots=True)
class Gross:
    """A test's cost as what it puts at risk: a heavy-up's extra spend and the revenue a cut is
    expected to lose, each counted where it is positive."""


@dataclass(frozen=True, slots=True)
class Net:
    """A test's cost as its spend change plus the revenue it is expected to lose, held at least at
    `floor`: a cut of a channel returning less than it costs pays for itself, and a cost not above
    0 ranks nothing.

    Raises:
        ValueError: on a floor not positive and finite.
    """

    floor: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.floor) and self.floor > 0.0):
            msg = f"floor={self.floor!r} is not positive and finite"
            raise ValueError(msg)


type Cost = Gross | Net
"""How `value_of_tests` counts a test's cost: what it puts at risk (`Gross`) or its net cost
(`Net`)."""


@dataclass(frozen=True, slots=True)
class Reweight:
    """Each outer draw's posterior read by weighing the other chains' draws by the test's
    likelihood, and as `Gaussian` where the weights' effective sample size falls below `ess`.

    Raises:
        ValueError: on an `ess` not positive and finite.
    """

    ess: float = 100.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.ess) and self.ess > 0.0):
            msg = f"ess={self.ess!r} is not positive and finite"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Gaussian:
    """Each outer draw's posterior mean worths moved by the Bayes-linear update: each fit's by the
    covariance of its draws' worths with the lift over the lift's predictive variance."""


_REWEIGHT = Reweight()


@dataclass(frozen=True, slots=True)
class Refit:
    """Each set of tests valued by `then` at every outer draw, and its posterior refitted with the
    tests' simulated estimates at `draws` outer draws of each half, evenly spaced: the mean gap
    between the refitted reductions and `then`'s there corrects `then`'s, a multi-fidelity
    estimate. `fit` takes each channel's simulated estimates, to be read beside those the model
    behind the readings already reads, and returns the readings of the model refitted with them.

    Raises:
        ValueError: on `draws` not a whole number, at least 2.
    """

    draws: int
    fit: Callable[[Mapping[Channel, Sequence[LiftEstimate]]], Readings]
    then: Reweight | Gaussian = _REWEIGHT

    def __post_init__(self) -> None:
        _whole(self.draws, "Refit.draws", 2)


type Update = Reweight | Gaussian | Refit
"""How `value_of_tests` updates each outer draw's posterior with a test's simulated lift:
`Reweight`, `Gaussian` or `Refit`."""

_GROSS = Gross()


@dataclass(frozen=True, slots=True, eq=False)
class ExperimentValue:
    """The value to the plan of `candidates` run together, in disjoint groups, and its cost; one
    candidate's where `value_of_tests` reads it.

    `reduction` is the regret the tests are expected to remove, in the KPI's units, read on the
    outer draws at even places, and `error` its Monte Carlo standard error; `predicted` and
    `interval` are the mean and the central `level` interval of the realised reduction on those at
    odd places, which no choice by `reduction` read. `reductions` is each outer draw's, in their
    order. `spend_change` is the tests' spend less the plan's and `opportunity` the revenue they are
    expected to lose; `gross` and `net` are the costs they make, each the sum of the tests', and
    `cost` the one asked for. `per_cost` is the revenue the reduction is worth a unit of `cost`,
    infinite where the tests cost nothing, and `net_value` that revenue times the uses, less the net
    cost; `predicted_net_value` is the same of `predicted`. `local` is the local model's reduction,
    NaN where it is not read, and `model_gap` its relative miss against `reduction`. `ess` and
    `pareto_k` are each outer draw's weights' effective sample size and their weighted fits'
    largest Pareto shape, and `fallback` the share of outer draws updated as `Gaussian`, each NaN
    under `Gaussian`. Under `Refit`, `reductions` and the update's diagnostics are its `then`'s,
    `refitted` is the reduction of the refitted posterior's plan at each outer draw refitted and
    NaN at the others, and `reduction`, `error`, `predicted` and `interval` are corrected by the
    gap, the interval moved by its half's mean; `refitted` is NaN throughout under the others.
    """

    candidates: tuple[Candidate, ...]
    reduction: float
    error: float
    predicted: float
    interval: tuple[float, float]
    reductions: _Array
    spend_change: float
    opportunity: float
    gross: float
    net: float
    cost: float
    per_cost: float
    net_value: float
    predicted_net_value: float
    local: float
    model_gap: float
    ess: _Array
    pareto_k: _Array
    fallback: float
    refitted: _Array


@dataclass(frozen=True, slots=True, eq=False)
class ExperimentValues:
    """Each candidate's value, in the candidates' order, and the plan without a test: `plan`, its
    spend a period in the readings' channels' order. `evpi` is the expected value of perfect
    information over every outer draw, which no test's reduction passes, and `evpi_error` its Monte
    Carlo standard error; `draws` is the number of outer draws and `level` the intervals'."""

    values: tuple[ExperimentValue, ...]
    plan: _Array
    evpi: float
    evpi_error: float
    draws: int
    level: float

    @property
    def ranked(self) -> tuple[ExperimentValue, ...]:
        """The values by the revenue each reduction is worth a unit of its cost, the most first,
        the candidates' order on a tie."""
        return tuple(sorted(self.values, key=lambda value: -value.per_cost))


@dataclass(frozen=True, slots=True, eq=False)
class ExperimentChoice:
    """The affordable set of tests of the largest net value, and whether to run it.

    `best` is that set's value, None where no set is affordable, and `single` the best affordable
    single test's. `verdict` is to test where `best`'s predicted net value is above 0, read on the
    outer draws at odd places, which no choice read, and to plan now otherwise: the empty set is
    always a choice, and the largest of noisy net values overstates the set it picks.
    `values` is every set valued, in the order valued: by size and then by the candidates' places
    where every affordable set was valued, each step's sets in turn where the set was grown
    greedily; `exhaustive` is which. `plan`, `evpi`, `evpi_error`, `draws` and `level` are as
    `ExperimentValues` holds them.
    """

    best: ExperimentValue | None
    single: ExperimentValue | None
    values: tuple[ExperimentValue, ...]
    exhaustive: bool
    plan: _Array
    evpi: float
    evpi_error: float
    draws: int
    level: float

    @property
    def verdict(self) -> Literal["test", "plan now"]:
        """To test where the best set's predicted net value is above 0, to plan now otherwise."""
        best = self.best
        return "test" if best is not None and best.predicted_net_value > 0.0 else "plan now"


def value_of_tests(
    panel: MediaPanel,
    geo: Geo,
    readings: Readings,
    candidates: Sequence[Candidate],
    budget: float,
    periods: int,
    limits: Mapping[Channel, Range],
    *,
    seed: int,
    cost: Cost = _GROSS,
    update: Update = _REWEIGHT,
    draws: int = 1024,
    level: float = 0.9,
    revenue_per_kpi: float = 1.0,
    uses: float = 1.0,
    grid: int = 512,
) -> ExperimentValues:
    """Each candidate's value to the plan of `geo`'s channels that spends `budget` over `periods`
    periods within `limits`, over `draws` outer draws of `readings`, each candidate's lift read
    with the same standard normals, drawn from a stream apart from `seed`'s own, and each outer
    draw's posterior updated by `update`; each channel's mean worth is read at `grid` spends across
    its range.

    Raises:
        ValueError: on no candidates or a candidate of a channel the readings do not have; a
            `draws` not a whole number, at least 4, or more than a fit holds of its share; a
            `level` not in `(0, 1)`; a revenue per unit of the KPI or a number of uses not positive
            and finite; a seed not a whole number, at least 0; a fit with draws of one chain, whose
            outer draws would leave no draw of it; a `Refit` of more draws than an outer half
            holds, or whose fit returns readings of other channels or of the channels in another
            order; and as `plan_posterior` on the panel, the ranges, the periods, the grid and the
            budget.
        RuntimeError: on three fits or more whose weights, solved again, do not converge.
    """
    context = _context(
        panel,
        geo,
        readings,
        candidates,
        budget,
        periods,
        limits,
        "value_of_tests",
        seed=seed,
        update=update,
        draws=draws,
        level=level,
        revenue_per_kpi=revenue_per_kpi,
        uses=uses,
        grid=grid,
    )
    outer = context.prior.outer
    count = len(context.candidates)
    z = np.random.default_rng(context.stream).standard_normal(outer.size)
    singles = [(t,) for t in range(count)]
    values = context.values(
        singles, np.repeat(z[:, None], count, axis=1), cost, level, revenue_per_kpi, uses
    )
    missed = sum(value.model_gap > _GAP_WARNING for value in values)
    if missed:
        log.warning(
            "the local model misses the reduction by more than %.0f%% on %d of %d candidates; the "
            "values are the nested Monte Carlo's",
            100.0 * _GAP_WARNING,
            missed,
            len(values),
        )
    perfect = context.prior.perfect()
    return ExperimentValues(
        values=tuple(values),
        plan=context.prior.plan,
        evpi=float(perfect.mean()),
        evpi_error=_error(perfect, *context.strata),
        draws=int(outer.size),
        level=float(level),
    )


def choose_tests(
    panel: MediaPanel,
    geo: Geo,
    readings: Readings,
    candidates: Sequence[Candidate],
    budget: float,
    periods: int,
    limits: Mapping[Channel, Range],
    *,
    test_budget: float,
    seed: int,
    cost: Cost = _GROSS,
    update: Update = _REWEIGHT,
    draws: int = 1024,
    exhaustive: int = 256,
    level: float = 0.9,
    revenue_per_kpi: float = 1.0,
    uses: float = 1.0,
    grid: int = 512,
) -> ExperimentChoice:
    """The set of `candidates` to run together, in disjoint groups of at most half the market
    between them, whose cost is at most `test_budget` and whose net value is the largest, valued as
    `value_of_tests` values one test, its tests' likelihoods multiplied: each candidate's lift read
    with its own standard normals, the same in every set it is in, drawn from a stream apart from
    `seed`'s own. Every affordable set is valued where there are at most `exhaustive` of them;
    otherwise the set grows from none by the test whose value adds the most a unit of the cost it
    adds, while that addition is worth more than the net cost it adds, and is kept where it is
    worth more than the best affordable single test, else that test is. The values are read on the
    outer draws at even places, so the best set's prediction, on those at odd places, is not the
    largest of noisy estimates.

    Raises:
        ValueError: as `value_of_tests`; and on a test budget not finite and at least 0, or an
            `exhaustive` not a whole number, at least 0.
        RuntimeError: as `value_of_tests`.
    """
    if not (math.isfinite(test_budget) and test_budget >= 0.0):
        msg = f"test_budget={test_budget!r} is not finite and at least 0"
        raise ValueError(msg)
    _whole(exhaustive, "exhaustive", 0)
    context = _context(
        panel,
        geo,
        readings,
        candidates,
        budget,
        periods,
        limits,
        "choose_tests",
        seed=seed,
        update=update,
        draws=draws,
        level=level,
        revenue_per_kpi=revenue_per_kpi,
        uses=uses,
        grid=grid,
    )
    outer = context.prior.outer
    z = np.random.default_rng(context.stream).standard_normal((outer.size, len(context.candidates)))
    priced = [
        _price((c,), (m,), revenue_per_kpi)
        for c, m in zip(context.candidates, context.means, strict=True)
    ]
    shares = [c.share for c in context.candidates]
    affordable = _affordable(priced, shares, cost, test_budget, exhaustive)
    every: list[ExperimentValue] = []

    def valued(sets: Sequence[tuple[int, ...]]) -> list[ExperimentValue]:
        values = context.values(sets, z, cost, level, revenue_per_kpi, uses)
        every.extend(values)
        return values

    if affordable is not None:
        values = valued(affordable)
        singles = [v for v in values if len(v.candidates) == 1]
        best = max(values, key=lambda v: v.net_value, default=None)
    else:
        best, singles = _greedy(valued, priced, shares, cost, test_budget, revenue_per_kpi, uses)
    single = max(singles, key=lambda v: v.net_value, default=None)
    perfect = context.prior.perfect()
    return ExperimentChoice(
        best=best,
        single=single,
        values=tuple(every),
        exhaustive=affordable is not None,
        plan=context.prior.plan,
        evpi=float(perfect.mean()),
        evpi_error=_error(perfect, *context.strata),
        draws=int(outer.size),
        level=float(level),
    )


class _Draws(Protocol):
    """A posterior's draws as the estimator reads them, as `Readings` holds them: each draw's fit
    and chain, each fit's weight and each draw's, and the fits' leave-one-out densities where the
    weights were solved from them."""

    @property
    def fit(self) -> _Indices: ...

    @property
    def chain(self) -> _Indices: ...

    @property
    def stacking(self) -> _Array: ...

    @property
    def pointwise(self) -> _Array | None: ...

    @property
    def weights(self) -> _Array: ...


@dataclass(frozen=True, slots=True, eq=False)
class _Planner:
    """What a posterior's plan reads, and how a plan is scored: each draw's features, `(draws,
    F)`, whose weighted mean `plan` plans, and `worth`, a draw's worth of each of several plans,
    `(plans, channels)`."""

    features: _Array
    plan: Callable[[_Array], _Array]
    worth: Callable[[int, _Array], _Array]


@dataclass(frozen=True, slots=True, eq=False)
class _Tests:
    """Tests read together: each draw's lift of each, `(draws, tests)`, their shares and standard
    errors, `(tests,)`, and each outer draw's standard normal of each, `(outer, tests)`."""

    lift: _Array
    share: _Array
    se: _Array
    noise: _Array


@dataclass(frozen=True, slots=True, eq=False)
class _Updated:
    """A set of tests' plan at each outer draw after they read, `(outer, channels)`, and their
    update's diagnostics there."""

    plans: _Array
    ess: _Array
    pareto_k: _Array
    fell: npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True, eq=False)
class _Preposterior:
    """What every set of tests is valued against: the planner, the posterior's draws, the outer
    draws and the update; the plan without a test, and each outer draw's worth of the plan without
    a test of its posterior, its own chain left out."""

    planner: _Planner
    draws: _Draws
    outer: _Indices
    update: Update
    plan: _Array
    worth: _Array

    def reductions(self, sets: Sequence[_Tests]) -> tuple[_Array, tuple[_Updated, ...]]:
        """Each set's reduction at each outer draw, `(sets, outer)`, its tests read together, and
        each set's update."""
        updated = tuple(
            _updated(self.planner, self.draws, tests, self.outer, self.update) for tests in sets
        )
        return self._against([u.plans for u in updated]), updated

    def perfect(self) -> _Array:
        """Perfect information's reduction at each outer draw: the plan of its draw's worths."""
        plans = np.array([self.planner.plan(self.planner.features[i]) for i in self.outer])
        return self._against([plans])[0]

    def _against(self, plans: Sequence[_Array]) -> _Array:
        """Each of several plans' worth at each outer draw, `(plans, outer)`, less the plan's
        without a test."""
        scored = np.array(
            [
                self.planner.worth(int(i), np.vstack([p[j] for p in plans]))
                for j, i in enumerate(self.outer)
            ]
        )
        return scored.T - self.worth


def _preposterior(
    planner: _Planner, draws: _Draws, outer: _Indices, update: Update
) -> _Preposterior:
    """The plan without a test, from every draw by its weight; and each outer draw's worth of the
    plan without a test of the posterior its update starts from, its own chain left out. Scored
    against the plan of every draw, which its own chain moved towards it, a test that reads nothing
    would lose what that chain gave."""
    plan = planner.plan(draws.weights @ planner.features)
    fits, chains = draws.fit[outer].tolist(), draws.chain[outer].tolist()
    plans = {
        (f, c): planner.plan(_left_out(draws, f, c) @ planner.features)
        for f, c in set(zip(fits, chains, strict=True))
    }
    worth = np.array(
        [
            planner.worth(int(i), plans[f, c][None])[0]
            for i, f, c in zip(outer, fits, chains, strict=True)
        ]
    )
    return _Preposterior(planner, draws, outer, update, plan, worth)


def _left_out(draws: _Draws, fit: int, chain: int) -> _Array:
    """Each draw's weight with fit `fit`'s chain `chain` left out, the fit's other draws sharing
    its weight: the posterior an update of an outer draw of that chain starts from."""
    weights = draws.weights.copy()
    held = draws.fit == fit
    kept = held & (draws.chain != chain)
    weights[held] = 0.0
    weights[kept] = draws.stacking[fit] / np.count_nonzero(kept)
    return weights


@dataclass(frozen=True, slots=True, eq=False)
class _Context:
    """What a call values sets of its candidates in: the candidates, each one's lift at every draw
    and its mean, the posterior and the plan's inputs, the outer draws' fits and chains, the
    stream the tests' normals are drawn from, and the plan of a refit's readings."""

    candidates: tuple[Candidate, ...]
    lifts: tuple[_Array, ...]
    means: tuple[float, ...]
    readings: Readings
    history: _Array
    lower: _Array
    upper: _Array
    budget: float
    periods: int
    prior: _Preposterior
    strata: tuple[_Indices, _Indices]
    stream: np.random.SeedSequence
    replan: Callable[[Readings], _Array]

    def values(
        self,
        sets: Sequence[tuple[int, ...]],
        noise: _Array,
        cost: Cost,
        level: float,
        revenue_per_kpi: float,
        uses: float,
    ) -> list[ExperimentValue]:
        """Each set of candidates' value, by their places, its tests read together, the tests
        read with the columns of `noise`, `(outer, candidates)`, at their places."""
        if not sets:
            return []
        tests = [
            _Tests(
                np.column_stack([self.lifts[t] for t in chosen]),
                np.array([self.candidates[t].share for t in chosen]),
                np.array([self.candidates[t].se for t in chosen]),
                noise[:, list(chosen)],
            )
            for chosen in sets
        ]
        reductions, updated = self.prior.reductions(tests)
        local = _local(self, tests)
        refit = self.prior.update
        refitted = [
            _refitted(self.prior, t, refit.draws, self.refit_plan(refit, chosen))
            if isinstance(refit, Refit)
            else np.full(self.prior.outer.size, np.nan)
            for chosen, t in zip(sets, tests, strict=True)
        ]
        return [
            _value(
                tuple(self.candidates[t] for t in chosen),
                tuple(self.means[t] for t in chosen),
                reductions[s],
                self.strata,
                updated[s],
                self.prior.update,
                local[s],
                refitted[s],
                cost,
                level,
                revenue_per_kpi,
                uses,
            )
            for s, chosen in enumerate(sets)
        ]

    def refit_plan(self, refit: Refit, chosen: tuple[int, ...]) -> Callable[[_Array], _Array]:
        """The plan of the posterior `refit` refits with a set's readings, `(tests,)`, as its
        candidates' estimates; the candidates by their places."""

        def plan(readings: _Array) -> _Array:
            estimates: dict[Channel, list[LiftEstimate]] = {}
            for t, lift in zip(chosen, readings.tolist(), strict=True):
                candidate = self.candidates[t]
                estimates.setdefault(candidate.channel, []).append(candidate.estimate(lift))
            return self.replan(refit.fit(estimates))

        return plan


def _context(
    panel: MediaPanel,
    geo: Geo,
    readings: Readings,
    candidates: Sequence[Candidate],
    budget: float,
    periods: int,
    limits: Mapping[Channel, Range],
    caller: str,
    *,
    seed: int,
    update: Update,
    draws: int,
    level: float,
    revenue_per_kpi: float,
    uses: float,
    grid: int,
) -> _Context:
    """The context `caller` values its candidates in, its arguments checked."""
    candidates = tuple(candidates)
    if not candidates:
        msg = f"{caller} needs candidates"
        raise ValueError(msg)
    _whole(draws, "draws", 4)
    _whole(seed, "seed", 0)
    if not (math.isfinite(level) and 0.0 < level < 1.0):
        msg = f"level={level!r} is not in (0, 1)"
        raise ValueError(msg)
    for name, value in (("revenue_per_kpi", revenue_per_kpi), ("uses", uses)):
        if not (math.isfinite(value) and value > 0.0):
            msg = f"{name}={value!r} is not positive and finite"
            raise ValueError(msg)
    single = [
        k for k, held in enumerate(_fits(readings)) if np.unique(readings.chain[held]).size < 2
    ]
    if single:
        msg = (
            "an outer draw's chain is left out of the posterior it updates, so every fit needs "
            f"draws of two chains or more: fits {single} have one"
        )
        raise ValueError(msg)
    history, lower, upper = _inputs(panel, geo, readings, periods, limits)
    _whole(grid, "grid", 2)
    budget = _spendable(budget, periods, lower, upper)
    lifts = tuple(predicted_lift(readings, c) for c in candidates)
    table = worths(panel, geo, readings, periods, limits, grid=grid)
    planner = _grid(history, readings, table.rates, table.table, budget / periods, periods)
    outer = _alike(readings, draws)
    if isinstance(update, Refit) and update.draws > outer.size // 2:
        msg = f"Refit.draws={update.draws} is more than the {outer.size // 2} outer draws of a half"
        raise ValueError(msg)

    def replan(refitted: Readings) -> _Array:
        if refitted.channels != readings.channels:
            msg = (
                f"a refit's readings have the channels {list(refitted.channels)}, not the "
                f"readings' {list(readings.channels)}"
            )
            raise ValueError(msg)
        again = worths(panel, geo, refitted, periods, limits, grid=grid)
        mean = refitted.weights @ again.table.reshape(again.table.shape[0], -1)
        return _allocated(again.rates, mean.reshape(again.rates.shape), budget / periods)

    return _Context(
        candidates=candidates,
        lifts=lifts,
        means=tuple(float(readings.weights @ lift) for lift in lifts),
        readings=readings,
        history=history,
        lower=lower,
        upper=upper,
        budget=budget,
        periods=periods,
        prior=_preposterior(planner, readings, outer, update),
        strata=(readings.fit[outer], readings.chain[outer]),
        stream=np.random.SeedSequence(seed).spawn(1)[0],
        replan=replan,
    )


def _grid(
    history: _Array,
    readings: Readings,
    rates: _Array,
    table: _Array,
    target: float,
    periods: int,
) -> _Planner:
    """The posterior planner on each draw's worths at `rates`, `table`, spending `target` a
    period, its plans scored by each draw's worth at their spends exactly."""
    channels, grid = rates.shape

    def plan(mean: _Array) -> _Array:
        return _allocated(rates, mean.reshape(channels, grid), target)

    def worth(s: int, plans: _Array) -> _Array:
        out = np.zeros(plans.shape[0])
        draw = np.array([s])
        for c in range(channels):
            for block, value in _worth(history[:, c], readings, draw, c, periods, plans[:, c]):
                out[block] += value[:, 0]
        return out

    return _Planner(table.reshape(table.shape[0], channels * grid), plan, worth)


def _members(draws: _Draws) -> Iterator[_Indices]:
    """Each fit's draws, in order."""
    for k in range(draws.stacking.size):
        yield np.flatnonzero(draws.fit == k)


def _inner(draws: _Draws, members: _Indices, k: int, i: int) -> _Indices:
    """Fit `k`'s draws in outer draw `i`'s posterior: all, or all but `i`'s chain where `i` is
    one of them."""
    if draws.fit[i] != k:
        return members
    return members[draws.chain[members] != draws.chain[i]]


def _updated(
    planner: _Planner, draws: _Draws, tests: _Tests, outer: _Indices, update: Update
) -> _Updated:
    """The plan at each outer draw once the tests read `share lift + se noise` there, together; a
    `Refit`'s `then`'s."""
    if isinstance(update, Refit):
        update = update.then
    y = tests.share * tests.lift[outer] + tests.se * tests.noise
    members = list(_members(draws))
    gaussian = _gaussian(planner.features, draws, members, tests, outer, y)
    match update:
        case Gaussian():
            means = gaussian
            ess, pareto_k = np.full(outer.size, np.nan), np.full(outer.size, np.nan)
            fell = np.zeros(outer.size, dtype=np.bool_)
        case Reweight(ess=least):
            means, ess, pareto_k, trusted = _reweighted(
                planner.features, draws, members, tests, outer, y
            )
            fell = (ess < least) | ~trusted
            means[fell] = gaussian[fell]
    return _Updated(np.array([planner.plan(mean) for mean in means]), ess, pareto_k, fell)


def _refitted(
    prior: _Preposterior, tests: _Tests, draws: int, plan: Callable[[_Array], _Array]
) -> _Array:
    """The reduction at `draws` outer draws of each half, evenly spaced, of the plan `plan` makes
    of the tests' readings there, those the update read; NaN at the other outer draws."""
    out = np.full(prior.outer.size, np.nan)
    for j in _refit_places(prior.outer.size, draws):
        i = int(prior.outer[j])
        y = tests.share * tests.lift[i] + tests.se * tests.noise[j]
        out[j] = prior.planner.worth(i, plan(y)[None])[0] - prior.worth[j]
    return out


def _refit_places(count: int, draws: int) -> _Indices:
    """`draws` places of each half of `count` outer draws, those at even places and those at odd,
    evenly spaced over the half from its first to its last."""
    places = [
        half[np.unique(np.linspace(0, half.size - 1, draws).round().astype(np.intp))]
        for half in (np.arange(0, count, 2), np.arange(1, count, 2))
    ]
    return np.sort(np.concatenate(places))


def _gaussian(
    features: _Array,
    draws: _Draws,
    members: Sequence[_Indices],
    tests: _Tests,
    outer: _Indices,
    y: _Array,
) -> _Array:
    """Each outer draw's posterior mean features by the Bayes-linear update of each fit's draws in
    its posterior, `f + Cov(f, lift) S (S Var(lift) S + D)^-1 (y - S lift)` at their means, `S`
    the shares and `D` the tests' noise variances, the fits weighed by the lifts' normal predictive
    density under each. The outer draws whose posteriors hold the same draws, those of other fits
    and those of each chain of their own, share their moments."""
    count, k = y.shape
    noise = np.diag(tests.se**2)
    updated = np.empty((len(members), count, features.shape[1]))
    density = np.empty((count, len(members)))
    for f, held in enumerate(members):
        centre, middle = features[held].mean(axis=0), tests.lift[held].mean(axis=0)
        x, m = features[held] - centre, tests.lift[held] - middle
        chains = draws.chain[held]
        own = draws.fit[outer] == f
        groups = [(~own, np.ones(held.size, dtype=np.bool_))]
        groups += [(own & (draws.chain[outer] == j), chains != j) for j in np.unique(chains)]
        for rows, kept in groups:
            if not rows.any():
                continue
            n = int(kept.sum())
            mean_x, mean_m = x[kept].sum(axis=0) / n, m[kept].sum(axis=0) / n
            variance = m[kept].T @ m[kept] / n - np.outer(mean_m, mean_m)
            covariance = m[kept].T @ x[kept] / n - np.outer(mean_m, mean_x)
            spread = tests.share[:, None] * variance * tests.share + noise
            gap = y[rows] - tests.share * (middle + mean_m)
            solved = np.linalg.solve(spread, gap.T).T
            updated[f, rows] = centre + mean_x + (solved * tests.share) @ covariance
            _, log_det = np.linalg.slogdet(spread)
            density[rows, f] = (
                -0.5 * np.sum(gap * solved, axis=1) - 0.5 * log_det - k * _LOG_ROOT_TAU
            )
    weights = np.array([_restacked(draws, column) for column in density])
    return np.einsum("rk,krf->rf", weights, updated)


def _reweighted(
    features: _Array,
    draws: _Draws,
    members: Sequence[_Indices],
    tests: _Tests,
    outer: _Indices,
    y: _Array,
) -> tuple[_Array, _Array, _Array, npt.NDArray[np.bool_]]:
    """Each outer draw's posterior mean features by its posterior's draws weighed by the tests'
    likelihood, the product of theirs, Pareto-smoothed within each fit, the fits' weights solved
    again; and the weights' effective sample size, their weighted fits' largest Pareto shape, and
    whether every such shape is within its threshold. A fit's relative efficiency is the least of
    its tests' lifts'."""
    count, k = y.shape
    efficiency = [
        min(_efficiency(tests.lift[held, t], draws.chain[held]) for t in range(k))
        for held in members
    ]
    log_noise = float(np.sum(np.log(tests.se))) + k * _LOG_ROOT_TAU
    means = np.empty((count, features.shape[1]))
    ess, pareto_k = np.empty(count), np.empty(count)
    trusted = np.ones(count, dtype=np.bool_)
    rows = max(1, _BLOCK // tests.lift.shape[0])
    for start in range(0, count, rows):
        stop = min(count, start + rows)
        q = np.zeros((stop - start, tests.lift.shape[0]))
        for r in range(start, stop):
            i = int(outer[r])
            ratios = -0.5 * np.sum(((y[r] - tests.share * tests.lift) / tests.se) ** 2, axis=1)
            column = np.empty(len(members))
            smoothed = []
            for f, held in enumerate(members):
                inner = _inner(draws, held, f, i)
                column[f] = special.logsumexp(ratios[inner]) - math.log(inner.size) - log_noise
                smoothed.append((inner, psis(ratios[inner], efficiency=efficiency[f])))
            weights = _restacked(draws, column)
            spread, shape = 0.0, -math.inf
            for f, (inner, s) in enumerate(smoothed):
                if weights[f] > 0.0:
                    q[r - start, inner] = weights[f] * np.exp(s.log_weights)
                    spread += weights[f] ** 2 / s.ess
                    shape = max(shape, s.pareto_k)
                    trusted[r] &= s.pareto_k <= s.k_threshold
            ess[r], pareto_k[r] = 1.0 / spread, shape
        means[start:stop] = q @ features
    return means, ess, pareto_k, trusted


def _restacked(draws: _Draws, column: _Array) -> _Array:
    """The fits' weights solved again with `column`, the test's log predictive density under each
    fit, as one more observation; the weights as given where there are no densities to solve them
    from."""
    if draws.pointwise is None:
        return draws.stacking
    pointwise = np.column_stack([draws.pointwise, column])
    match draws.stacking.size:
        case 1:
            return np.ones(1)
        case 2:
            return _stacking_of_two(pointwise)
        case _:
            return stacking(pointwise)


def _efficiency(values: _Array, chain: _Indices) -> float:
    """The relative efficiency of `values` over their chains, each cut to the shortest: their
    split-chain effective sample size over their number, at most 1; and 1 where there is none, the
    split chains holding fewer than four values or the values one throughout."""
    held = [values[chain == j] for j in np.unique(chain)]
    n = min(h.size for h in held)
    x = np.stack([h[:n] for h in held])
    ess = _ess(_split(x))
    return min(1.0, ess / x.size) if math.isfinite(ess) else 1.0


def _local(context: _Context, sets: Sequence[_Tests]) -> list[float]:
    """Each set's reduction by the local model: at each fit's median channels, half the trace of
    the covariance of the channels' parameters with the tests' lifts through the decision weight,
    over the lifts' predictive covariance at the market's scale, summed over the fits by their
    weights; NaN where a fit of positive weight has a decision weight that is not read."""
    readings = context.readings
    local = np.zeros(len(sets))
    for k, held in enumerate(_fits(readings)):
        if readings.stacking[k] == 0.0:
            continue
        median = _median(readings, held)
        channels = [_response(median, 0, c) for c in range(len(readings.channels))]
        try:
            weight = decision_weight(
                channels,
                context.budget,
                context.periods,
                lower=context.lower,
                upper=context.upper,
                history=context.history,
            )
        except ValueError:
            local[:] = np.nan
            continue
        leaves = _leaves(readings, held, weight.parameters)
        leaves -= leaves.mean(axis=0)
        for t, tests in enumerate(sets):
            lift = tests.lift[held] - tests.lift[held].mean(axis=0)
            covariance = leaves.T @ lift / held.size
            spread = lift.T @ lift / held.size + np.diag((tests.se / tests.share) ** 2)
            reach = covariance.T @ weight.matrix @ covariance
            local[t] += readings.stacking[k] * 0.5 * np.trace(np.linalg.solve(spread, reach))
    return local.tolist()


def _median(readings: Readings, held: _Indices) -> Readings:
    """The fit of draws `held` as one draw, each parameter its median over them."""
    tanh = bool(readings.tanh[held[0]])
    channels = len(readings.channels)

    def median(values: _Array) -> _Array:
        return np.median(values[held], axis=0)[None]

    return Readings(
        channels=readings.channels,
        retention=median(readings.retention),
        half_saturation=median(readings.half_saturation),
        slope=np.full((1, channels), np.nan) if tanh else median(readings.slope),
        coefficient=median(readings.coefficient),
        tanh=np.array([tanh]),
        fit=np.zeros(1, dtype=np.intp),
        chain=np.zeros(1, dtype=np.intp),
        stacking=np.ones(1),
        pointwise=None,
        length=readings.length,
    )


def _leaves(readings: Readings, held: _Indices, names: Sequence[str]) -> _Array:
    """Draws `held`'s channel parameters named as the decision weight names them, `(draws,
    parameters)`: each channel's retention, curve scale (a `tanh`'s half-saturation point over
    `atanh(1/2)`), Hill slope and coefficient."""
    tanh = readings.tanh[held]
    columns = []
    for name in names:
        position, _, leaf = name.partition(".")
        c = int(position)
        match leaf:
            case "kernel.retention":
                columns.append(readings.retention[held, c])
            case "curve.scale":
                half = readings.half_saturation[held, c]
                columns.append(np.where(tanh, half / _TANH_HALF, half))
            case "curve.slope":
                columns.append(readings.slope[held, c])
            case "coefficient":
                columns.append(readings.coefficient[held, c])
            case _:
                msg = f"the decision weight reads a parameter the readings do not hold, {name!r}"
                raise ValueError(msg)
    return np.column_stack(columns)


def _value(
    candidates: tuple[Candidate, ...],
    mean_lifts: tuple[float, ...],
    reductions: _Array,
    strata: tuple[_Indices, _Indices],
    updated: _Updated,
    update: Update,
    local: float,
    refitted: _Array,
    cost: Cost,
    level: float,
    revenue_per_kpi: float,
    uses: float,
) -> ExperimentValue:
    """A set of candidates' value from its reduction at each outer draw, of fits and chains
    `strata`, and its cost; under `Refit`, each half's corrected by its refits."""
    choice, held_out = reductions[0::2], reductions[1::2]
    even, odd = (strata[0][0::2], strata[1][0::2]), (strata[0][1::2], strata[1][1::2])
    if isinstance(update, Refit):
        reduction, error, correction = _corrected(choice, refitted[0::2], *even)
        predicted, _, shift = _corrected(held_out, refitted[1::2], *odd)
        log.info(
            "%d refits of the tests of %s moved their reduction by %.3g, its error %.3g, and their "
            "prediction by %.3g",
            int(np.count_nonzero(~np.isnan(refitted))),
            [str(c.channel) for c in candidates],
            correction,
            error,
            shift,
        )
    else:
        reduction, error = float(choice.mean()), _error(choice, *even)
        predicted, shift = float(held_out.mean()), 0.0
    low, high = np.quantile(held_out, [(1.0 - level) / 2.0, (1.0 + level) / 2.0]) + shift
    spend_change, opportunity, gross, net = _price(candidates, mean_lifts, revenue_per_kpi)
    charged = _charged(cost, gross, net)
    base = update.then if isinstance(update, Refit) else update
    worth = revenue_per_kpi * reduction
    return ExperimentValue(
        candidates=candidates,
        reduction=reduction,
        error=error,
        predicted=predicted,
        interval=(float(low), float(high)),
        reductions=reductions,
        spend_change=spend_change,
        opportunity=opportunity,
        gross=gross,
        net=net,
        cost=charged,
        per_cost=worth / charged if charged > 0.0 else math.inf,
        net_value=uses * worth - net,
        predicted_net_value=uses * revenue_per_kpi * predicted - net,
        local=local,
        model_gap=_relative_gap(local, reduction),
        ess=updated.ess,
        pareto_k=updated.pareto_k,
        fallback=float(updated.fell.mean()) if isinstance(base, Reweight) else math.nan,
        refitted=refitted,
    )


def _corrected(
    base: _Array, refitted: _Array, fit: _Indices, chain: _Indices
) -> tuple[float, float, float]:
    """The mean of `base`, outer draws of fits `fit` and chains `chain` in order, corrected by the
    mean gap between `refitted` and `base` where `refitted` is read; the corrected mean's standard
    error, `sqrt(e^2 + Var(gap) / m + 2 Cov(base, gap) / n)`, `e` the error of `base`'s mean, `m`
    the refits and `n` the outer draws; and the correction."""
    held = ~np.isnan(refitted)
    gap = refitted[held] - base[held]
    correction = float(gap.mean())
    spread = (
        _error(base, fit, chain) ** 2
        + float(np.var(gap, ddof=1)) / gap.size
        + 2.0 * float(np.cov(base[held], gap)[0, 1]) / base.size
    )
    return float(base.mean()) + correction, math.sqrt(max(spread, 0.0)), correction


def _price(
    candidates: Sequence[Candidate], mean_lifts: Sequence[float], revenue_per_kpi: float
) -> tuple[float, float, float, float]:
    """The tests' spend change, their opportunity, their gross cost, each test's counted where it
    is positive, and their net cost: each the sum of the tests'."""
    changes = [float(np.sum(c.spend - c.planned)) for c in candidates]
    lost = [-revenue_per_kpi * c.share * m for c, m in zip(candidates, mean_lifts, strict=True)]
    spend_change, opportunity = sum(changes, 0.0), sum(lost, 0.0)
    gross = sum((max(a, 0.0) + max(b, 0.0) for a, b in zip(changes, lost, strict=True)), 0.0)
    return spend_change, opportunity, gross, spend_change + opportunity


def _charged(cost: Cost, gross: float, net: float) -> float:
    """The cost `cost` asks for."""
    match cost:
        case Gross():
            return gross
        case Net(floor=floor):
            return max(net, floor)


def _affordable(
    priced: Sequence[tuple[float, float, float, float]],
    shares: Sequence[float],
    cost: Cost,
    test_budget: float,
    limit: int,
) -> list[tuple[int, ...]] | None:
    """Every set of candidates, by their places in order, whose shares sum to at most half the
    market and whose cost is at most `test_budget`; None where there are more than `limit`. The
    candidates are tried by their net costs, the least first, and only an affordable set is grown:
    in that order a set's share and gross cost only grow, and its net cost falls only while all its
    tests' are negative, where `Net` charges its floor, so no set grown from one it costs too much
    is affordable.
    """
    order = sorted(range(len(shares)), key=lambda t: priced[t][3])
    found: list[tuple[int, ...]] = []
    stack: list[tuple[tuple[int, ...], int]] = [((), 0)]
    while stack:
        chosen, start = stack.pop()
        for place in range(start, len(order)):
            grown = (*chosen, order[place])
            if not _fits_in(grown, priced, shares, cost, test_budget):
                continue
            found.append(tuple(sorted(grown)))
            if len(found) > limit:
                return None
            stack.append((grown, place + 1))
    return sorted(found, key=lambda chosen: (len(chosen), chosen))


def _greedy(
    valued: Callable[[Sequence[tuple[int, ...]]], list[ExperimentValue]],
    priced: Sequence[tuple[float, float, float, float]],
    shares: Sequence[float],
    cost: Cost,
    test_budget: float,
    revenue_per_kpi: float,
    uses: float,
) -> tuple[ExperimentValue | None, list[ExperimentValue]]:
    """The set grown from none by the affordable test whose value adds the most a unit of the cost
    it adds, the most value on a tie, while what it adds is worth more than the net cost it adds,
    or the best affordable single test where that is worth more; and every affordable single
    test's value."""
    chosen: tuple[int, ...] = ()
    grown: ExperimentValue | None = None
    singles: list[ExperimentValue] = []
    while True:
        sets = [
            tuple(sorted((*chosen, t)))
            for t in range(len(shares))
            if t not in chosen and _fits_in((*chosen, t), priced, shares, cost, test_budget)
        ]
        if not sets:
            break
        values = valued(sets)
        if not chosen:
            singles = values
        reduction, net, charged = (
            (grown.reduction, grown.net, grown.cost) if grown is not None else (0.0, 0.0, 0.0)
        )
        pick, best = None, (-math.inf, -math.inf)
        for place, value in enumerate(values):
            added = revenue_per_kpi * (value.reduction - reduction)
            if uses * added - (value.net - net) <= 0.0:
                continue
            spent = value.cost - charged
            ranked = (added / spent if spent > 0.0 else math.inf, added)
            if ranked > best:
                pick, best = place, ranked
        if pick is None:
            break
        chosen, grown = sets[pick], values[pick]
    single = max(singles, key=lambda value: value.net_value, default=None)
    if grown is None or (single is not None and single.net_value > grown.net_value):
        return single, singles
    return grown, singles


def _fits_in(
    chosen: Sequence[int],
    priced: Sequence[tuple[float, float, float, float]],
    shares: Sequence[float],
    cost: Cost,
    test_budget: float,
) -> bool:
    """Whether a set of candidates takes at most half the market and costs at most the budget."""
    share, gross, net = _totals(chosen, priced, shares)
    return share <= _SHARE_LIMIT and _charged(cost, gross, net) <= test_budget


def _totals(
    chosen: Sequence[int],
    priced: Sequence[tuple[float, float, float, float]],
    shares: Sequence[float],
) -> tuple[float, float, float]:
    """A set of candidates' share of the market, gross cost and net cost, each the exact sum of its
    tests' rounded once: summed in turn, shares of 0.05, 0.17 and 0.28 come to 0.5 in one order and
    past it in another."""
    return (
        math.fsum(shares[t] for t in chosen),
        math.fsum(priced[t][2] for t in chosen),
        math.fsum(priced[t][3] for t in chosen),
    )


def _error(values: _Array, fit: _Indices, chain: _Indices) -> float:
    """The Monte Carlo standard error of the mean of `values`, outer draws of fits `fit` and chains
    `chain` in order: `sqrt(sum_k n_k s_k^2 / e_k) / n`, `n_k` fit `k`'s count, `s_k^2` its
    variance and `e_k` its relative efficiency; a fit of one outer draw adds what it cannot
    measure, a term of order `1 / n^2`, as 0."""
    variance = 0.0
    for k in np.unique(fit):
        held = fit == k
        count = int(held.sum())
        if count > 1:
            spread = float(np.var(values[held], ddof=1))
            variance += count * spread / _efficiency(values[held], chain[held])
    return math.sqrt(variance) / values.size


def _relative_gap(model: float, measured: float) -> float:
    """`|model / measured - 1|`, as `chc.experiment` reads its local model's miss: 0 where both
    are 0, infinite where only `measured` is, NaN where `model` is."""
    if math.isnan(model):
        return math.nan
    if measured == 0.0:
        return 0.0 if model == 0.0 else math.inf
    return abs(model / measured - 1.0)
