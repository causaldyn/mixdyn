"""Spend paths compared as what-ifs: each scenario's KPI, its difference from a baseline with the
carryover past the horizon counted, a budget-neutral move between geos, and response curves.

A scenario is a named spend path over a horizon after a fit's window, `Scenario`.
`compare_scenarios` reads every scenario of a static fit through its posterior predictive,
`forecast_bayes`, run over the horizon and the kernel's reach after it, `length - 1` periods: the
last period a spend of the horizon's moves. Past the horizon every scenario spends nothing and holds
the baseline's controls, so a scenario's difference from the baseline counts the carryover its spend
leaves after the horizon, and the controls and the calendar cancel there. A scenario's total is read
over the horizon alone, the KPI a planner would see in it; its difference over the horizon and the
carry-out, what its spend changes wherever the change falls. The two are read over different
windows, and a difference is not the difference of two totals.

Every scenario reads the same draws of the posterior and the same noise, common random numbers: a
difference is the difference of the two scenarios' means at each draw, so no noise enters it, and
two identical scenarios differ by exactly 0 at every draw. The mean is the KPI's expectation under
each of the model's noises, the log-normal's included (ADR 0017), so a difference of means is the
difference of expected KPIs.

The model's non-media terms, its intercept, calendar and controls, are additive in its mean in
every model the engine fits: the static model, each geo of the model over geos, and the nested
model's second stage. Two spend plans under the same controls then differ by their channels'
contributions alone, `media_difference`: each draw's `chc.response.Channel` run over the history
and each plan and the kernel's reach after it, as `GeoPlan.gain` reads a plan against spending
nothing, for any fit that hands `channel_draws`.

`move_between_geos` moves an amount of one channel's spend from one geo's cell to another's in a
plan of spend by geo and channel, in exact arithmetic on the two cells: what leaves the one is
what the other gains, to the last bit, so the total is unchanged. `response_curves` reads
`mixdyn.decisions.posterior.worths`' table as each channel's posterior mean worth and its weighted
band at each spend a period.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from fractions import Fraction
from types import MappingProxyType
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from mixdyn.decisions._quantile import weighted_quantiles
from mixdyn.models.bayes_mmm import LogNormal, Normal, forecast_bayes

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence

    from chc import response

    from mixdyn.data import Channel
    from mixdyn.decisions.posterior import Readings, Worths
    from mixdyn.models.bayes_mmm import BayesFit

type _Array = npt.NDArray[np.float64]

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Scenario:
    """A named spend path over a forecast's horizon: `spend`, each channel's spend at each period,
    and `controls`, each control's value at each, a control it does not name held at the
    baseline's."""

    name: str
    spend: Mapping[str, npt.ArrayLike]
    controls: Mapping[str, npt.ArrayLike] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True, slots=True, eq=False)
class ScenarioComparison:
    """Scenarios read against a baseline at the same draws, in the panel's units.

    `names` are the scenarios' in the order given, and `baseline` the one the others are read
    against. `totals` is each scenario's KPI summed over the horizon at each draw, drawn about its
    mean by the model's noise, `(scenarios, draws)`; `total_mean` its mean and `total_low` and
    `total_high` the central `level` interval of its draws. `differences` is each scenario's mean
    less the baseline's at each draw, summed over the horizon and the `carry_out` periods after
    it, `(scenarios, draws)`; `difference_mean`, `difference_low` and `difference_high` read it as
    the totals are read. The baseline's differences are 0.
    """

    names: tuple[str, ...]
    baseline: str
    horizon: int
    carry_out: int
    level: float
    totals: _Array
    total_mean: _Array
    total_low: _Array
    total_high: _Array
    differences: _Array
    difference_mean: _Array
    difference_low: _Array
    difference_high: _Array


@dataclass(frozen=True, slots=True, eq=False)
class ResponseCurves:
    """Each channel's worth at each spend a period, `(channels, grid)`, the channels in the
    readings' order: `rates`, the spends; `mean`, the posterior mean worth over the draws at their
    weights; and `low` and `high`, the weighted central `level` band. `current` is each channel's
    current spend a period and `at_current` the mean worth there, linear between the grid's
    points, or both None where no current spend was given."""

    channels: tuple[Channel, ...]
    level: float
    rates: _Array
    mean: _Array
    low: _Array
    high: _Array
    current: _Array | None
    at_current: _Array | None


def compare_scenarios(
    fit: BayesFit,
    scenarios: Sequence[Scenario],
    *,
    baseline: str,
    horizon: int,
    draws: int,
    seed: int,
    level: float = 0.9,
) -> ScenarioComparison:
    """Each of `scenarios` over the `horizon` periods after the fit's window, read against the one
    named `baseline` at `draws` of the posterior and the noise drawn from `seed`, both shared by
    every scenario.

    A scenario's total is its KPI over the horizon; its difference from the baseline is its mean
    less the baseline's over the horizon and the kernel's reach after it, `fit.length - 1`
    periods, in which every scenario spends nothing and holds the baseline's controls at their
    last value. Totals and differences are read over different windows: a difference counts the
    carryover the horizon's spend leaves after it, which no total holds. Under log-normal noise a
    mean must be above 0 over the horizon, where a total draws the KPI about it, and not over the
    carry-out, which only differences read; the totals then draw their noise over the horizon
    alone.

    Raises:
        ValueError: on no scenarios, two of one name, or no scenario named `baseline`; a level
            not inside (0, 1); a baseline that does not name every control or a scenario that
            names one the model does not have; and what `forecast_bayes` refuses of the horizon,
            the draws and each scenario's spend and controls, a log-normal mean not above 0 over
            the horizon among it.
    """
    scenarios = tuple(scenarios)
    names = tuple(s.name for s in scenarios)
    if not names or len(set(names)) != len(names) or baseline not in names:
        msg = (
            f"scenarios need distinct names, one of them the baseline {baseline!r}: got "
            f"{list(names)}"
        )
        raise ValueError(msg)
    _level(level)
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        msg = f"horizon must be a positive integer, got {horizon!r}"
        raise ValueError(msg)
    base = scenarios[names.index(baseline)]
    controls = [c.column for c in fit.spec.controls]
    lacking = [c for c in controls if c not in base.controls]
    if lacking:
        msg = f"the baseline must give every control's values: it lacks {lacking}"
        raise ValueError(msg)
    carry_out = fit.length - 1
    held = {}
    for c in controls:
        values = np.asarray(base.controls[c], dtype=np.float64)
        if values.shape != (horizon,):
            msg = f"the baseline's control {c!r} has shape {values.shape}, not ({horizon},)"
            raise ValueError(msg)
        held[c] = np.full(carry_out, values[-1])
    # a log-normal KPI has no density where its mean is not above 0, which a trend run on over the
    # carry-out can reach; the carry-out's means enter differences alone, where the non-media
    # terms cancel, so they are read as under normal noise, and the KPI over the horizon alone
    lognormal = isinstance(fit.spec.likelihood, LogNormal)
    carried = replace(fit, spec=replace(fit.spec, likelihood=Normal())) if lognormal else fit
    means = []
    kpis = []
    for s in scenarios:
        extra = sorted(set(s.controls) - set(controls))
        if extra:
            msg = f"scenario {s.name!r} names controls {extra} the model does not have"
            raise ValueError(msg)
        plan = {}
        for channel, spend in s.spend.items():
            plan[channel] = _extended(spend, horizon, carry_out, 0.0)
        for c in controls:
            values = s.controls[c] if c in s.controls else base.controls[c]
            plan[c] = np.concatenate([_series(values, horizon, f"{s.name}'s {c}"), held[c]])
        ahead = forecast_bayes(carried, horizon + carry_out, plan, draws, seed)
        means.append(ahead.mean)
        if lognormal:
            within = {name: series[:horizon] for name, series in plan.items()}
            kpis.append(forecast_bayes(fit, horizon, within, draws, seed).kpi.sum(axis=1))
        else:
            kpis.append(ahead.kpi[:, :horizon].sum(axis=1))
    reference = means[names.index(baseline)]
    differences = np.stack([(m - reference).sum(axis=1) for m in means])
    totals = np.stack(kpis)
    ends = ((1.0 - level) / 2.0, (1.0 + level) / 2.0)
    total_low, total_high = np.quantile(totals, ends, axis=1)
    difference_low, difference_high = np.quantile(differences, ends, axis=1)
    log.info(
        "scenarios compared",
        extra={"scenarios": len(names), "horizon": horizon, "carry_out": carry_out, "draws": draws},
    )
    return ScenarioComparison(
        names=names,
        baseline=baseline,
        horizon=horizon,
        carry_out=carry_out,
        level=level,
        totals=totals,
        total_mean=totals.mean(axis=1),
        total_low=total_low,
        total_high=total_high,
        differences=differences,
        difference_mean=differences.mean(axis=1),
        difference_low=difference_low,
        difference_high=difference_high,
    )


def media_difference(
    channel_draws: Sequence[Mapping[str, response.Channel]],
    history: Mapping[str, npt.ArrayLike],
    plan_a: Mapping[str, npt.ArrayLike],
    plan_b: Mapping[str, npt.ArrayLike],
    periods: int,
) -> _Array:
    """What `plan_b` adds to the KPI over `plan_a` at each draw, `(draws,)`, read through the
    channels alone: each draw's channel run over its `history` and each plan's `periods` periods
    of spend and its kernel's reach after them, `length - 1` periods of no spend, and the
    difference summed over the plan's periods and the reach.

    The models' non-media terms are additive in the mean and cancel between two plans under the
    same controls; under log-normal noise the mean is the KPI's expectation (ADR 0017), so the
    difference is one of expected KPIs there too. `channel_draws` is any fit's: `BayesFit`'s,
    `BayesGeoFit`'s for one geo with that geo's history, or `NestedFit`'s.

    Raises:
        ValueError: on JAX in single precision; no draws; a draw, a history or a plan whose
            channels are not the first draw's; a plan's series that is not `periods` finite values
            at or above 0, or a history's that is not finite and at or above 0.
    """
    if not jax.config.x64_enabled:
        msg = "a difference is read in double precision (JAX_ENABLE_X64=1)"
        raise ValueError(msg)
    if isinstance(periods, bool) or not isinstance(periods, int) or periods < 1:
        msg = f"periods must be a positive integer, got {periods!r}"
        raise ValueError(msg)
    draws = list(channel_draws)
    if not draws:
        msg = "a difference needs at least one draw"
        raise ValueError(msg)
    channels = sorted(draws[0])
    for name, given in (("history", history), ("plan_a", plan_a), ("plan_b", plan_b)):
        if sorted(given) != channels:
            msg = f"{name} must name the channels {channels}, got {sorted(given)}"
            raise ValueError(msg)
    if any(sorted(d) != channels for d in draws):
        msg = f"every draw must hold the channels {channels}"
        raise ValueError(msg)
    past = {c: _spend(history[c], None, f"history's {c}") for c in channels}
    a = {c: _spend(plan_a[c], periods, f"plan_a's {c}") for c in channels}
    b = {c: _spend(plan_b[c], periods, f"plan_b's {c}") for c in channels}
    out = np.zeros(len(draws))
    for i, draw in enumerate(draws):
        for c in channels:
            channel = draw[c]
            tail = np.zeros(channel.kernel.length - 1)
            before = past[c].size
            read = [
                np.asarray(channel(jnp.asarray(np.concatenate([past[c], plan[c], tail]))))[before:]
                for plan in (a, b)
            ]
            out[i] += float((read[1] - read[0]).sum())
    return out


def move_between_geos[G: Hashable, C: Hashable](
    plan: Mapping[G, Mapping[C, npt.ArrayLike]],
    channel: C,
    from_geo: G,
    to_geo: G,
    amount: npt.ArrayLike,
) -> dict[G, dict[C, _Array]]:
    """`plan`, each geo's spend by channel, with `amount` of `channel`'s spend moved from
    `from_geo`'s cell to `to_geo`'s, at each period where the cells are series; every other cell
    as it was.

    The move is exact on the two cells: the larger of them after it is rounded to its last bit
    and the smaller is the two cells' sum less it, so that what leaves the one is what the other
    gains, and their sum is the same real number before and after. Where the sum's last bit is
    finer than the smaller cell can hold, as moving 5000 from 9000.37 to 12.34 leaves it, no two
    floats near the move hold that real number, and their sum in double precision is the same
    float before and after instead, the real sum within one and a half units in the smaller
    cell's last place. The amount moved is within a unit in the larger cell's last place of
    `amount`; one equal to the source's moves all of it but what lies below that place.

    Raises:
        ValueError: on a geo or channel the plan does not have; the same geo twice; cells of
            other shapes, or with a value not finite or below 0; an amount that does not fit
            their shape, or is not finite or below 0; and a move larger than the source holds.
    """
    if from_geo == to_geo:
        msg = f"a move needs two geos, got {from_geo!r} twice"
        raise ValueError(msg)
    for geo in (from_geo, to_geo):
        if geo not in plan or channel not in plan[geo]:
            msg = f"the plan has no cell for geo {geo!r} and channel {channel!r}"
            raise ValueError(msg)
    source = _cell(plan[from_geo][channel], f"{from_geo!r}'s {channel!r}")
    target = _cell(plan[to_geo][channel], f"{to_geo!r}'s {channel!r}")
    if source.shape != target.shape:
        msg = f"the cells have shapes {source.shape} and {target.shape}"
        raise ValueError(msg)
    try:
        moved = np.broadcast_to(np.asarray(amount, dtype=np.float64), source.shape).copy()
    except ValueError:
        msg = f"the amount's shape {np.shape(amount)} does not fit the cells' {source.shape}"
        raise ValueError(msg) from None
    if not (np.isfinite(moved).all() and (moved >= 0.0).all()):
        msg = "the amount must be finite and at or above 0"
        raise ValueError(msg)
    if (moved > source).any():
        at = np.flatnonzero((moved > source).ravel()).tolist()
        msg = f"the move is larger than the source cell holds at {at}"
        raise ValueError(msg)
    after_source = np.empty_like(source)
    after_target = np.empty_like(target)
    for k in np.ndindex(source.shape):
        after_source[k], after_target[k] = _exact_move(
            float(source[k]), float(target[k]), float(moved[k])
        )
    out = {
        g: {c: _cell(v, f"{g!r}'s {c!r}") for c, v in cells.items()} for g, cells in plan.items()
    }
    out[from_geo][channel] = after_source
    out[to_geo][channel] = after_target
    return out


def response_curves(
    worths: Worths,
    readings: Readings,
    *,
    level: float,
    current: Mapping[Channel, float] | None = None,
) -> ResponseCurves:
    """Each channel's posterior mean worth and its central `level` band at each spend a period
    of `worths`, the draws at `readings.weights`: the band's ends are the weighted quantiles,
    each the least worth whose cumulative weight reaches its share, which at equal weights is
    `numpy.quantile`'s `inverted_cdf`. `current` marks each channel's current spend a period.

    Raises:
        ValueError: on a level not inside (0, 1); a table whose draws or channels are not the
            readings'; and a current spend missing for a channel, named for one the readings do
            not have, or off the channel's grid.
    """
    _level(level)
    table = np.asarray(worths.table, dtype=np.float64)
    rates = np.asarray(worths.rates, dtype=np.float64)
    weights = readings.weights
    channels = readings.channels
    if table.ndim != 3 or table.shape[:2] != (weights.size, len(channels)):
        msg = (
            f"the table has shape {table.shape}, not ({weights.size}, {len(channels)}, grid): "
            "the readings' draws and channels"
        )
        raise ValueError(msg)
    mean = np.einsum("d,dcg->cg", weights, table)
    ends = weighted_quantiles(table, weights, ((1.0 - level) / 2.0, (1.0 + level) / 2.0))
    marked = None
    at_current = None
    if current is not None:
        missing = [c for c in channels if c not in current]
        extra = [c for c in current if c not in channels]
        if missing or extra:
            msg = f"current spend lacks channels {missing} and names others {extra}"
            raise ValueError(msg)
        marked = np.array([float(current[c]) for c in channels])
        off = [
            c
            for c, x, low, high in zip(channels, marked, rates[:, 0], rates[:, -1], strict=True)
            if not low <= x <= high
        ]
        if off:
            msg = f"the current spend of {off} is off its grid"
            raise ValueError(msg)
        at_current = np.array(
            [np.interp(x, r, m) for x, r, m in zip(marked, rates, mean, strict=True)]
        )
    return ResponseCurves(
        channels=channels,
        level=level,
        rates=rates,
        mean=mean,
        low=ends[0],
        high=ends[1],
        current=marked,
        at_current=at_current,
    )


def _level(level: float) -> None:
    if not (isinstance(level, float | int) and 0.0 < level < 1.0):
        msg = f"level must be inside (0, 1), got {level!r}"
        raise ValueError(msg)


def _series(values: npt.ArrayLike, horizon: int, name: str) -> _Array:
    """`values` as `horizon` floats; what they hold is `forecast_bayes`' to refuse."""
    out = np.asarray(values, dtype=np.float64)
    if out.shape != (horizon,):
        msg = f"{name} has shape {out.shape}, not ({horizon},)"
        raise ValueError(msg)
    return out


def _extended(values: npt.ArrayLike, horizon: int, after: int, fill: float) -> _Array:
    """`horizon` values, then `after` of `fill`."""
    return np.concatenate([_series(values, horizon, "a scenario's spend"), np.full(after, fill)])


def _spend(values: npt.ArrayLike, periods: int | None, name: str) -> _Array:
    """A spend series, of `periods` values where given, finite and at or above 0."""
    out = np.asarray(values, dtype=np.float64)
    if out.ndim != 1 or (periods is not None and out.size != periods):
        want = f"({periods},)" if periods is not None else "one dimension"
        msg = f"{name} has shape {out.shape}, not {want}"
        raise ValueError(msg)
    if not (np.isfinite(out).all() and (out >= 0.0).all()):
        msg = f"{name} must be finite and at or above 0"
        raise ValueError(msg)
    return out


def _cell(values: npt.ArrayLike, name: str) -> _Array:
    out = np.array(values, dtype=np.float64)
    if not (np.isfinite(out).all() and (out >= 0.0).all()):
        msg = f"the cell {name} must be finite and at or above 0"
        raise ValueError(msg)
    return out


def _exact_move(source: float, target: float, amount: float) -> tuple[float, float]:
    """The two cells after `amount` leaves `source` for `target`, their sum kept.

    The larger cell after the move is the float nearest its exact value, at most the two cells'
    sum, and the smaller is the sum less it, computed exactly, where that is a float. Where it is
    not, the sum's last bit finer than the smaller cell's, no two floats near the move sum to it,
    and the smaller is the float nearest it, or one of its neighbours, whose double-precision sum
    with the larger is the cells' sum before the move.
    """
    whole = Fraction(source) + Fraction(target)
    wanted = (Fraction(source) - Fraction(amount), Fraction(target) + Fraction(amount))
    big = 0 if wanted[0] >= wanted[1] else 1
    larger = float(wanted[big])
    if Fraction(larger) > whole:
        larger = float(np.nextafter(larger, 0.0))
    rest = whole - Fraction(larger)
    smaller = float(rest)
    if Fraction(smaller) != rest:
        before = source + target
        near = (smaller, float(np.nextafter(smaller, 0.0)), float(np.nextafter(smaller, np.inf)))
        # the sum's rounding interval is at least twice the smaller's step, so one of the three
        # falls inside it
        smaller = next(x for x in near if x >= 0.0 and larger + x == before)
    return (larger, smaller) if big == 0 else (smaller, larger)
