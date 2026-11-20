"""CHC's own planner as an arm: `chc.allocation.allocate` over curves the arm fits itself from the
export, by the simplest fit CHC offers for a sales history (ADR 0041).

The fit is `chc.dlm`'s forward filter of a static model: every discount 1, so the filter is the
conjugate regression over every week at once. The weekly sales are regressed on a level and a
slope, a yearly season of 3 harmonics, each channel's spend through a normalised geometric kernel
of the export's length and a `tanh` at a coefficient of 1, and the controls, under a weakly
informative prior on the series' own scale: each standard deviation ten times its coordinate's
part of the forecast, the sales' standard deviation `s` for the level and the season, `s / T` for
the slope and `s / rms(x)` for a column, the variance learned from one degree of freedom at `s²`,
as the engine's dynamic model's default prior states it. A column's prior so scales with it, and a
curve's scale moves the likelihood only through the fit.

Each channel's retention and `tanh` scale, the latter in multiples of its mean positive spend, are
chosen on `GRID` by the filter's log-likelihood, the criterion `chc.dlm` names for choosing a model.
The likelihood over the channels' curves has more than one peak, so the grid is searched as the
engine's ridge search searches its space: 32 starts, the grid's middle for every channel and 31
drawn at random from a child of the export's seed, and the 4 most likely polished, one channel at
a time, its whole grid with the others held, each in turn, a move taken only where it raises the
likelihood, until a sweep moves none or `SWEEPS` sweeps have run; the choice is the most likely
polished. The chosen model's coefficients are then drawn with every channel's held at or above 0
(`chc.dlm.constrained_sample`), 4 chains of 1000 draws after 1000 of warm-up from another child of
the export's seed, since `allocate` refuses a channel that returns less as it is spent; each
channel's coefficient is the mean of its draws, the truncated posterior's, not a clipped one.

The variant `chc` plans with `allocate` on those channels, against the export's box and budget,
the history carried in. The variant `chc cvar 0.1 vs equal` plans with
`chc.allocation.cvar_allocate` for the most mean gain over the bench's equal split
(`scorecard.arms.panel.equal_split`) in the worst tenth of `DRAWS` readings, each the chosen curves
at one draw's coefficients, evenly spaced over the pooled draws (ADR 0044). Each claims its plan's
gain over the status quo, the mean and the central 90 % interval of the gain over the readings.
`chc`'s record carries the response (`scorecard.arms.mapped`), the readings beside each channel's
part of each draw's mean over the history as the filter reads it (`chc.dlm.decompose`), and the
other names it. Neither forecasts: `chc.dlm.forecast` reads the posterior without the signs, not
the draws the plans read. Their flags are each channel's retention and scale, whether
each sits at an end of its grid, its chosen column's coefficient of variation and whether it is
`flat`, below `FLAT`, the chosen model's log-likelihood, the models the search filtered, the sweeps
the chosen polish ran and whether it ended by moving none, whether the draws did not mix by CHC's
check, and their least bulk effective sample size.

A flat column is one the history barely moves: a curve saturated where the spend is never 0, or a
spend that never moves. The fit reads the channel's coefficient from the column's movement, and the
return counts the column's whole height, down to no spend: the rest of it trades against the level
along a ridge the likelihood hardly sees. Below a coefficient of variation of 0.05 the column's
standard deviation is under a twentieth of its mean. On the refresh cell's pilot, 41 of 84
channel-fits chose such a column, and every one read its return above 3 times the truth (ADR 0041,
amended 2026-11-07); `FLAT` is the line that reading drew, fixed before any run reads the flag. The
variation is read over the periods from the first whose carryover holds `FULL` of its kernel's
weight: before it the column rises from the nothing the kernel assumes spent before the history,
which no spend moved. Over those periods a spend that never moves varies its column by at most 1 %
of it, so its column is always flat.

The fit reads the history alone, and CHC's fit of a channel to tests (`chc.lift.fit_lift`) reads a
test's gap period by period, which the export does not hold: the arm reads no lift test (`tests`),
and every record says so, under `n/a` and in its settings. It reads a national export alone: one
with a geo axis is refused, since the arm fits no channel for each geo. The harness hands it a geo
export's market instead (`scorecard.arms.panel.market`, ADR 0043): the geos' sums, planned one
weekly spend a channel.

A smoke run searches 3 retentions and 3 scales from 8 starts, polishes 1, and draws 2 chains of 50
after 50 of warm-up.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from chc import response
from chc.allocation import allocate, cvar_allocate
from chc.dlm import (
    DynamicLinearModel,
    Polynomial,
    PosteriorDraws,
    Prior,
    Regression,
    Seasonal,
    constrained_sample,
    decompose,
    forward_filter,
)

from scorecard.arms import panel as reading
from scorecard.arms.base import Planned
from scorecard.arms.mapped import held, mapped

if TYPE_CHECKING:
    from collections.abc import Callable

    from causaldyn_bench.scorecard.observe import Observation
    from chc.dlm import DLMFit

type Transforms = tuple[tuple[float, float], ...]
"""Each channel's retention and scale multiple, in the export's order of the channels."""


@dataclass(frozen=True, slots=True)
class Grid:
    """Each channel's retentions and `tanh` scales, the latter in multiples of its mean positive
    spend; a search starts at the middle of each."""

    retentions: tuple[float, ...]
    scales: tuple[float, ...]

    @property
    def start(self) -> tuple[float, float]:
        return self.retentions[len(self.retentions) // 2], self.scales[len(self.scales) // 2]


@dataclass(frozen=True, slots=True)
class Search:
    """`starts` points of a grid, its middle first and the rest drawn at random, and the `polish`
    most likely of them polished."""

    starts: int
    polish: int


@dataclass(frozen=True, slots=True)
class Sampler:
    """The sign-held draws' chains, and the draws of each after its warm-up."""

    chains: int
    warmup: int
    draws: int


GRID = Grid(tuple(round(0.1 * k, 1) for k in range(10)), (0.25, 0.5, 1.0, 2.0, 4.0))
SMOKE_GRID = Grid((0.2, 0.5, 0.8), (0.5, 1.0, 2.0))
SEARCH = Search(starts=32, polish=4)
SMOKE_SEARCH = Search(starts=8, polish=1)
SWEEPS = 10
SAMPLER = Sampler(chains=4, warmup=1000, draws=1000)
SMOKE_SAMPLER = Sampler(chains=2, warmup=50, draws=50)
HARMONICS, YEAR = 3, 52.18
LEADING = 2
"""The model's blocks before the channels': the level and slope, and the season."""
SPREAD, PRIOR_DOF = 100.0, 1.0
"""Each prior variance's multiple of its coordinate's scale squared, and the variance's prior
degrees of freedom."""
FLAT = 0.05
"""The chosen column's coefficient of variation below which a channel's flag `flat` is set."""
FULL = 0.99
"""The share of its kernel's weight a period's carryover holds from which a column's variation is
read."""
DRAWS = 400
LEVEL = 0.1
GAIN_LEVEL = 0.9
RESPONDS = "chc"
"""The variant whose record carries the fit's response."""
UNREAD = (
    "the arm's fit reads the history alone; CHC's fit of a channel to tests reads a test's gap"
    " period by period, which the export does not hold: every rung's plan is the plan of k = 0"
)
"""What a record says of the lift tests it did not read."""


@dataclass(frozen=True, slots=True, eq=False)
class Choice:
    """The search's choice: each channel's retention and scale multiple and the chosen model's
    fit; the sweeps its polish ran and whether the last moved none; and the models filtered."""

    transforms: Transforms
    fit: DLMFit
    sweeps: int
    converged: bool
    evaluations: int


@dataclass(frozen=True, slots=True, eq=False)
class Fitted:
    """The channels a plan reads, their readings, and what a record keeps of the fit."""

    channels: tuple[response.Channel, ...]
    readings: tuple[tuple[response.Channel, ...], ...]
    response: dict[str, Any]
    flags: dict[str, float | int | bool]
    settings: dict[str, Any]
    ess: float


class Chc:
    """The arm; see the module."""

    variants = (RESPONDS, "chc cvar 0.1 vs equal")
    tests = False
    cells = False

    def fit(self, observation: Observation, *, smoke: bool) -> Fitted:
        if observation.population is not None:
            msg = (
                f"the chc arm plans one national market, and this export has a geo axis of"
                f" {observation.population.size} geos: the arm fits no channel for each geo, which"
                " chc.allocation.allocate_geos would plan on"
            )
            raise ValueError(msg)
        grid, search, sampler = (
            (SMOKE_GRID, SMOKE_SEARCH, SMOKE_SAMPLER) if smoke else (GRID, SEARCH, SAMPLER)
        )
        seed, starts = (
            int(child.generate_state(1)[0])
            for child in np.random.SeedSequence(observation.seed).spawn(2)
        )
        means = _means(observation)
        choice = choose(observation, grid, means, search, starts)
        count = len(observation.channels)
        signed = constrained_sample(
            choice.fit,
            dict.fromkeys(range(count), "positive"),
            sampler.draws,
            seed,
            chains=sampler.chains,
            warmup=sampler.warmup,
        )
        pool = signed.pooled
        first = sum(b.size for b in choice.fit.model.blocks[:LEADING])
        coefficients = pool.states[:, -1, first : first + count]
        pick = np.unique(np.linspace(0, coefficients.shape[0] - 1, DRAWS).round().astype(int))

        def channels(at: np.ndarray) -> tuple[response.Channel, ...]:
            return tuple(
                _channel(retention, multiple * means[j], float(at[j]), observation.kernel_length)
                for j, (retention, multiple) in enumerate(choice.transforms)
            )

        readings = tuple(channels(c) for c in coefficients[pick])
        # each channel's part of each draw's mean: a Regression block's part per column, after a
        # part for each block before it
        parts = decompose(choice.fit, PosteriorDraws(pool.states[pick], pool.variance[pick]))
        split = parts[:, :, LEADING : LEADING + count]
        made = mapped(observation.channels, readings, split.sum(axis=1), split[held(len(pick))])
        variation = _variation(observation, means, choice.transforms)
        flags = _flags(choice, grid, observation.channels, variation) | {
            "fitted not mixed": int(not signed.mixed),
            "min ess bulk": signed.ess,
        }
        settings = {
            "retentions": list(grid.retentions),
            "scales": list(grid.scales),
            "search": asdict(search),
            "search_seed": starts,
            "sweeps": SWEEPS,
            "trend_order": 2,
            "season_harmonics": HARMONICS,
            "prior_spread": SPREAD,
            "prior_dof": PRIOR_DOF,
            "sampler": asdict(sampler),
            "seed": seed,
            "readings": len(readings),
            "gain_level": GAIN_LEVEL,
            "lift_rows_read": 0,
        }
        return Fitted(
            channels(coefficients.mean(axis=0)), readings, made, flags, settings, signed.ess
        )

    def plan(self, fit: Fitted, variant: str, observation: Observation) -> Planned:
        responds = variant == RESPONDS
        if responds:
            weekly = allocate(
                list(fit.channels),
                observation.budget,
                observation.planned,
                lower=observation.lower,
                upper=observation.upper,
                history=observation.spend,
            ).spend
            settings: dict[str, Any] = {"planner": "allocate"}
        else:
            weekly = cvar_allocate(
                fit.readings,
                observation.budget,
                observation.planned,
                level=LEVEL,
                against=reading.equal_split(observation),
                lower=observation.lower,
                upper=observation.upper,
                history=observation.spend,
            ).spend
            settings = {"planner": "cvar_allocate", "level": LEVEL, "against": "equal split"}
        gain = reading.gains(fit.readings, observation, weekly)
        low, high = np.quantile(gain, [(1.0 - GAIN_LEVEL) / 2.0, (1.0 + GAIN_LEVEL) / 2.0])
        absent = {
            "forecast": "chc.dlm's forecast reads the posterior without the plan's signs",
            "lift tests": UNREAD,
        }
        return Planned(
            weekly=np.asarray(weekly, dtype=float),
            gain=float(np.mean(gain)),
            gain_interval=(float(low), float(high)),
            response=fit.response if responds else None,
            flags=fit.flags,
            settings=fit.settings | settings,
            absent=absent if responds else absent | {"response": f"recorded under {RESPONDS!r}"},
            ess=fit.ess,
        )


def choose(
    observation: Observation, grid: Grid, means: np.ndarray, search: Search, seed: int
) -> Choice:
    """Each channel's retention and scale multiple on `grid` by the static model's log-likelihood,
    searched by `search` from starts `seed` draws; see the module. `means` is each channel's mean
    positive spend."""
    sales = np.asarray(observation.sales, dtype=float)
    spend = np.asarray(observation.spend, dtype=float)
    controls = [np.asarray(observation.controls[c], dtype=float) for c in observation.controls]
    columns: dict[tuple[int, float, float], np.ndarray] = {}
    likelihoods: dict[Transforms, float] = {}

    def column(j: int, retention: float, multiple: float) -> np.ndarray:
        key = (j, retention, multiple)
        if key not in columns:
            unit = _channel(retention, multiple * means[j], 1.0, observation.kernel_length)
            columns[key] = np.asarray(unit(spend[:, j]))
        return columns[key]

    def filtered(transforms: Transforms) -> DLMFit:
        x = np.column_stack([column(j, *t) for j, t in enumerate(transforms)] + controls)
        return forward_filter(_model(sales, x, len(transforms)), sales, x)

    def likelihood(transforms: Transforms) -> float:
        if transforms not in likelihoods:
            likelihoods[transforms] = filtered(transforms).log_likelihood
        return likelihoods[transforms]

    points = [(retention, multiple) for retention in grid.retentions for multiple in grid.scales]
    count, rng = len(observation.channels), np.random.default_rng(seed)
    starts = [(grid.start,) * count]
    starts += [
        tuple(points[i] for i in rng.integers(len(points), size=count))
        for _ in range(search.starts - 1)
    ]
    best_first = sorted(dict.fromkeys(starts), key=likelihood, reverse=True)
    polished = [_polish(start, points, likelihood) for start in best_first[: search.polish]]
    transforms, sweeps, converged = max(polished, key=lambda p: likelihood(p[0]))
    return Choice(transforms, filtered(transforms), sweeps, converged, len(likelihoods))


def _polish(
    start: Transforms, points: list[tuple[float, float]], likelihood: Callable[[Transforms], float]
) -> tuple[Transforms, int, bool]:
    """`start` moved one channel at a time to the most likely point of its grid, the others held,
    while a move raises the likelihood: where it ends, the sweeps it ran and whether the last moved
    none."""
    current = start
    for sweep in range(1, SWEEPS + 1):
        moved = False
        for j in range(len(current)):
            best = max(
                ((*current[:j], point, *current[j + 1 :]) for point in points), key=likelihood
            )
            if likelihood(best) > likelihood(current):
                current, moved = best, True
        if not moved:
            return current, sweep, True
    return current, SWEEPS, False


def _means(observation: Observation) -> np.ndarray:
    """Each channel's mean positive spend over the history.

    Raises:
        ValueError: on a channel that spends nothing over it.
    """
    spend = np.asarray(observation.spend, dtype=float)
    dark = [c for c, column in zip(observation.channels, spend.T, strict=True) if not column.any()]
    if dark:
        msg = f"the channels {dark} spend nothing over the history, so no curve of it is fitted"
        raise ValueError(msg)
    return np.array([float(column[column > 0.0].mean()) for column in spend.T])


def _channel(retention: float, scale: float, coefficient: float, length: int) -> response.Channel:
    kernel = response.GeometricAdstock(retention, length=length, normalized=True)
    return response.Channel(kernel, response.Tanh(scale), coefficient)


def _model(sales: np.ndarray, x: np.ndarray, channels: int) -> DynamicLinearModel:
    """The static model of `sales` on a level and a slope, the season and `x`'s columns, the first
    `channels` of them each a block of its own, under the prior the module states."""
    s2 = float(np.var(sales, ddof=1))
    blocks: list[Polynomial | Seasonal | Regression] = [
        Polynomial(2, 1.0),
        Seasonal(YEAR, tuple(range(1, HARMONICS + 1)), 1.0),
    ]
    blocks += [Regression(1, 1.0) for _ in range(channels)]
    if x.shape[1] > channels:
        blocks.append(Regression(x.shape[1] - channels, 1.0))
    squares = np.mean(x**2, axis=0)
    variance = [s2, s2 / sales.size**2, *[s2] * blocks[1].size]
    variance += [s2 / square if square > 0.0 else s2 for square in squares.tolist()]
    mean = np.zeros(len(variance))
    mean[0] = float(np.mean(sales))
    prior = Prior(mean, SPREAD * np.diag(variance), s2, PRIOR_DOF)
    # constrained_sample needs the additive form, and chc defaults to "directional" since 0.20.0
    return DynamicLinearModel(tuple(blocks), prior, form="additive")


def _flags(
    choice: Choice, grid: Grid, channels: tuple[str, ...], variation: list[float]
) -> dict[str, float | int | bool]:
    """What a record keeps of the search, each channel's column's coefficient of variation
    `variation` among it; see the module."""
    flags: dict[str, float | int | bool] = {}
    for name, (retention, multiple), cv in zip(channels, choice.transforms, variation, strict=True):
        flags[f"{name} retention"] = retention
        flags[f"{name} retention at an end"] = retention in _ends(grid.retentions)
        flags[f"{name} scale"] = multiple
        flags[f"{name} scale at an end"] = multiple in _ends(grid.scales)
        flags[f"{name} column cv"] = cv
        flags[f"{name} flat"] = cv < FLAT
    return flags | {
        "log likelihood": choice.fit.log_likelihood,
        "evaluations": choice.evaluations,
        "sweeps": choice.sweeps,
        "converged": choice.converged,
    }


def _ends(values: tuple[float, ...]) -> tuple[float, float]:
    return values[0], values[-1]


def _variation(observation: Observation, means: np.ndarray, transforms: Transforms) -> list[float]:
    """Each channel's column's coefficient of variation at `transforms`, over the periods from the
    first whose carryover holds `FULL` of its kernel's weight; `means` is each channel's mean
    positive spend."""
    spend = np.asarray(observation.spend, dtype=float)
    found = []
    for j, (retention, multiple) in enumerate(transforms):
        unit = _channel(retention, multiple * means[j], 1.0, observation.kernel_length)
        start = int(np.argmax(np.cumsum(np.asarray(unit.kernel.weights())) >= FULL))
        column = np.asarray(unit(spend[:, j]))[start:]
        found.append(float(np.std(column) / np.mean(column)))
    return found
