"""A DLM's discounts chosen by one-step predictive likelihood, and intervals that carry the choice.

Every spec of a grid is filtered on the geo. The one-step likelihood's best is the point estimate,
and `chc.dlm.confidence_set` keeps the specs a likelihood-ratio test does not reject; only those
are fitted. An interval is the union of the kept fits' intervals (projection). The best fit's
interval alone is too narrow: an evolution variance is weakly identified, so the likelihood's best
point is a selection. CHC's `scripts/bench_dlm.py` measures the pick, the likelihood-weighted
mixture and the projection against the truth.

Where the effects carry transforms, the choice hands its channels to a planner: the best fit's at a
point (`DlmChoice.channels`), and draws over every kept fit (`DlmChoice.readings`), a mixture that
weighs each kept spec alike. The mixture's quantiles are not the union's interval: CHC's ADR 0041
measured the set's draws pooled below the union's coverage, so an interval is read off `bounds` or
`coefficient_bounds`.

Given lift tests (`mixdyn.calibration.LiftEstimate`), a spec's likelihood is its KPI's and its
tests' given the KPI (`DlmFit.log_likelihood`): the best spec, the kept set and so the mixture move
with the tests, and each fit's posterior is given them.

The effects share one discount unless the grid says `PerEffect`, when each effect's is chosen on
its own, one parameter an effect. A grid stops where its values stop: where a kept spec sits at an
end of the grid that the parameter goes on past, the set, and every interval over it, may stop
there short of the data's, so the choice names each such end (`DlmChoice.edges`) and logs it.
"""

from __future__ import annotations

import functools
import itertools
import logging
import math
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from chc import response
from scipy import special, stats

from mixdyn.data.panel import calendar
from mixdyn.models.dlm import (
    DlmForecastDraws,
    _carried,
    _drawable,
    _filter,
    _media,
    _planned,
    fit_dlm,
)
from mixdyn.models.dlm_decomposition import decompose_dlm
from mixdyn.models.dlm_lift import lift_reading, lifted

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from chc.dlm import DLMFit

    from mixdyn.calibration import LiftEstimate
    from mixdyn.data import Geo, MediaPanel
    from mixdyn.models.dlm import DlmFit, DlmSpec, Season, Trend
    from mixdyn.models.dlm_decomposition import Components, DlmDecomposition

log = logging.getLogger(__name__)

type _Array = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PerEffect:
    """Discounts each effect takes on its own: a grid tries every combination of `values` over the
    effects, the last effect's varying fastest, and counts one parameter an effect. The effects
    whose columns `held` names keep the spec's own discount, are no parameter and no block, as a
    calendar event's static coefficient (ADR 0069).

    Raises:
        ValueError: on no value, or a value named twice.
    """

    values: tuple[float, ...]
    held: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.values or len(set(self.values)) != len(self.values):
            msg = f"each effect's discounts must name at least one, none twice, got {self.values}"
            raise ValueError(msg)

    def varied(self, spec: DlmSpec) -> list[int]:
        """The indices of `spec`'s effects whose discounts the grid varies: all but the held."""
        return [j for j, e in enumerate(spec.effects) if e.column not in self.held]


@dataclass(frozen=True, slots=True)
class Edge:
    """An end of a grid that a kept spec reaches where the parameter goes on past it, so the set
    the likelihood ratio keeps may stop there short of the data's: `parameter` names it, such as
    `"trend discount"`, `"tv discount"` or `"tv half-saturation"`, `end` is the grid's value
    there, a half-saturation point's as a multiple of the channel's mean positive spend, and
    `best` says whether the likelihood's best spec sits there too."""

    parameter: str
    end: float
    best: bool


@dataclass(frozen=True, slots=True)
class DiscountGrid:
    """The discounts to try, per family of blocks: the trend's, the season's, and the effects',
    one shared by every effect or with `PerEffect` each effect's own. An empty family keeps the
    spec's own discount and is not a parameter.

    Raises:
        ValueError: on a grid with no family, or a family that names a discount twice.
    """

    trend: tuple[float, ...] = ()
    season: tuple[float, ...] = ()
    effects: tuple[float, ...] | PerEffect = ()

    def __post_init__(self) -> None:
        families = {"trend": self.trend, "season": self.season}
        if isinstance(self.effects, tuple):
            families["effects"] = self.effects
        if not any(families.values()) and not isinstance(self.effects, PerEffect):
            msg = "a discount grid needs at least one family's discounts"
            raise ValueError(msg)
        for name, values in families.items():
            if len(set(values)) != len(values):
                msg = f"the grid names a {name} discount twice: {values}"
                raise ValueError(msg)

    def parameters(self, spec: DlmSpec) -> int:
        """How many discounts the grid varies in `spec`, one a family it sets and with `PerEffect`
        one an effect: its likelihood ratio's degrees of freedom."""
        shared = sum(1 for values in (self.trend, self.season) if values)
        if isinstance(self.effects, PerEffect):
            return shared + len(self.effects.varied(spec))
        return shared + (1 if self.effects else 0)

    def specs(self, spec: DlmSpec) -> tuple[DlmSpec, ...]:
        """`spec` at every combination of the grid's discounts, the last family varying fastest,
        and with `PerEffect` the last effect's fastest of the effects'.

        Raises:
            ValueError: on a family the grid varies that `spec` does not have.
        """
        trends, seasons = self._shared(spec)
        effects = [spec.effects]
        if isinstance(self.effects, PerEffect):
            self._has_effects(spec)
            varied = self.effects.varied(spec)
            effects = []
            for chosen in itertools.product(self.effects.values, repeat=len(varied)):
                at = dict(zip(varied, chosen, strict=True))
                effects.append(
                    tuple(
                        replace(e, discount=at[j]) if j in at else e
                        for j, e in enumerate(spec.effects)
                    )
                )
        elif self.effects:
            self._has_effects(spec)
            effects = [tuple(replace(e, discount=d) for e in spec.effects) for d in self.effects]
        return tuple(
            replace(spec, trend=t, season=s, effects=e)
            for t, s, e in itertools.product(trends, seasons, effects)
        )

    def blocks(self, spec: DlmSpec) -> tuple[tuple[DlmSpec, ...], ...]:
        """The grid about `spec` in the blocks a search moves one at a time (`search_media`): the
        whole of `specs(spec)` as one; with `PerEffect`, every pair of the discounts it varies,
        the trend's, the season's and each effect's, at every combination of the pair's values
        with the rest held, or the one alone where it varies one. The product over the effects
        grows as the values to the power of their count and the pairs as its square; a pair
        moves together two discounts that trade off, where one at a time can stop short.

        Raises:
            ValueError: on a family the grid varies that `spec` does not have.
        """
        if not isinstance(self.effects, PerEffect):
            return (self.specs(spec),)
        self._has_effects(spec)
        trends, seasons = self._shared(spec)
        axes: list[list[Callable[[DlmSpec], DlmSpec]]] = []
        if self.trend:
            axes.append([functools.partial(replace, trend=t) for t in trends])
        if self.season:
            axes.append([functools.partial(replace, season=s) for s in seasons])
        axes += [
            [functools.partial(_with_discount, index=j, discount=d) for d in self.effects.values]
            for j in self.effects.varied(spec)
        ]
        if len(axes) == 1:
            return (tuple(move(spec) for move in axes[0]),)
        return tuple(
            tuple(second(first(spec)) for first, second in itertools.product(one, other))
            for one, other in itertools.combinations(axes, 2)
        )

    def edges(self, kept: Sequence[DlmSpec], best: DlmSpec) -> tuple[Edge, ...]:
        """Each end of the grid that a spec of `kept` reaches where the discount goes on past it:
        a family's lowest discount, and its highest where that is below 1, the discount's own
        bound, of a family of more than one; with `PerEffect` each effect's. `best` is the
        likelihood's best spec."""
        per_effect = isinstance(self.effects, PerEffect)
        axes = {"trend discount": self.trend, "season discount": self.season}
        if isinstance(self.effects, PerEffect):
            axes |= {
                f"{e.column} discount": self.effects.values
                for e in best.effects
                if e.column not in self.effects.held
            }
        else:
            axes["effects discount"] = self.effects
        readings = [_discounts(s, per_effect=per_effect) for s in kept]
        top = _discounts(best, per_effect=per_effect)
        edges: list[Edge] = []
        for parameter, values in axes.items():
            ends = _ends(values, 0.0, 1.0)
            if ends:
                edges += _edges(parameter, ends, [r[parameter] for r in readings], top[parameter])
        return tuple(edges)

    def _shared(self, spec: DlmSpec) -> tuple[Sequence[Trend | None], Sequence[Season | None]]:
        """The trends and the seasons the grid tries in `spec`, the spec's own where it varies
        none.

        Raises:
            ValueError: on a family the grid varies that `spec` does not have.
        """
        trend, trends = spec.trend, [spec.trend]
        if self.trend:
            if trend is None:
                msg = "the grid varies the trend's discount, and the spec has no trend"
                raise ValueError(msg)
            trends = [replace(trend, discount=d) for d in self.trend]
        season, seasons = spec.season, [spec.season]
        if self.season:
            if season is None:
                msg = "the grid varies the season's discount, and the spec has no season"
                raise ValueError(msg)
            seasons = [replace(season, discount=d) for d in self.season]
        return trends, seasons

    @staticmethod
    def _has_effects(spec: DlmSpec) -> None:
        if not spec.effects:
            msg = "the grid varies the effects' discount, and the spec has no effect"
            raise ValueError(msg)


def _with_discount(spec: DlmSpec, *, index: int, discount: float) -> DlmSpec:
    """`spec` with its effect at `index` at `discount`."""
    effects = list(spec.effects)
    effects[index] = replace(effects[index], discount=discount)
    return replace(spec, effects=tuple(effects))


def _discounts(spec: DlmSpec, *, per_effect: bool) -> dict[str, float]:
    """`spec`'s discounts by the parameter a grid varies: the trend's, the season's, and the
    effects' one or, `per_effect`, each effect's by its column."""
    found = {}
    if spec.trend is not None:
        found["trend discount"] = spec.trend.discount
    if spec.season is not None:
        found["season discount"] = spec.season.discount
    if per_effect:
        found |= {f"{e.column} discount": e.discount for e in spec.effects}
    elif spec.effects:
        found["effects discount"] = spec.effects[0].discount
    return found


def _ends(values: Sequence[float], low: float, high: float) -> tuple[float, ...]:
    """The lowest and the highest of an axis of more than one value, each where it is not the
    parameter's own bound, `low` or `high`: the ends a grid can cut a set short at."""
    if len(values) < 2:
        return ()
    return tuple(end for end, bound in ((min(values), low), (max(values), high)) if end != bound)


def _edges(
    parameter: str, ends: Sequence[float], kept: Sequence[float], best: float, unit: float = 1.0
) -> list[Edge]:
    """Each of `parameter`'s `ends`, in units of `unit`, that a kept spec's value of `kept`
    reaches, and whether the best's value, `best`, is there too."""
    return [Edge(parameter, end, best == end * unit) for end in ends if end * unit in kept]


def _warn(logger: logging.Logger, geo: Geo, edges: Sequence[Edge]) -> None:
    """A warning on `logger` naming each end of a grid the kept specs reach, where there is one."""
    if not edges:
        return
    logger.warning(
        "dlm grid cut short on geo %s: the kept specs reach %s, where the set may stop short of "
        "the data's",
        geo,
        "; ".join(f"{e.parameter} at {e.end:g}{', the best' if e.best else ''}" for e in edges),
        extra={"geo": str(geo), "edges": [(e.parameter, e.end, e.best) for e in edges]},
    )


@dataclass(frozen=True, slots=True)
class Bounds:
    """An interval's ends."""

    lower: float
    upper: float


@dataclass(frozen=True, slots=True, eq=False)
class ComponentBounds:
    """One interval per component, as `Components` holds one array of draws per component."""

    base: Bounds | None
    seasonality: Bounds | None
    contributions: Mapping[str, Bounds]


@dataclass(frozen=True, slots=True, eq=False)
class DlmChoice:
    """Every spec of a grid filtered on one geo, the one-step likelihood's best, and the specs the
    likelihood ratio keeps, each with its fit. `fit` gives the point estimates; `bounds` and
    `coefficient_bounds` the intervals; `channels` and `readings` what a planner reads.

    `filtered` holds each spec's filter, in the order of `specs`; `kept` maps the index of each
    kept spec to its fit. A spec the test rejects is not smoothed, and with a signed effect its
    posterior is not drawn; a kept spec's signed chains that have not mixed are drawn again at
    twice the draws, at most twice. A media search (`mixdyn.models.dlm_media.search_media`) holds
    only the specs it kept, and fits every one of them or a sample of them stratified by
    likelihood (ADR 0023): `kept` maps the index of each fitted spec to its fit.

    `edges` names each end of a grid that a kept spec reaches where the parameter goes on past it
    (`Edge`): there the set, and every interval over it, may stop short of the data's.

    `lifts`, where the choice read lift tests, is each spec's tests' log density given its KPI, in
    the order of `specs`, and empty where it read none: a spec's log-likelihood is its filter's
    plus its tests' (`log_likelihoods`).
    """

    specs: tuple[DlmSpec, ...]
    filtered: tuple[DLMFit, ...]
    best: int
    kept: Mapping[int, DlmFit]
    edges: tuple[Edge, ...] = ()
    lifts: tuple[float, ...] = ()

    @property
    def fit(self) -> DlmFit:
        """The fit whose log-likelihood is the highest."""
        return self.kept[self.best]

    @property
    def log_likelihoods(self) -> tuple[float, ...]:
        """Each spec's log-likelihood, in the order of `specs`: its filter's one-step, plus its
        tests' log density given its KPI where the choice read tests."""
        kpi = [f.log_likelihood for f in self.filtered]
        if not self.lifts:
            return tuple(kpi)
        return tuple(a + b for a, b in zip(kpi, self.lifts, strict=True))

    def bounds(
        self,
        view: Callable[[DlmDecomposition], Components],
        *,
        draws: int,
        seed: int,
        level: float = 0.9,
    ) -> ComponentBounds:
        """Each component's interval at `level` for a view of the decomposition, such as
        `lambda d: d.window(start, end)`: the union of the kept fits' intervals, each from
        `draws` of its posterior's draws, as `decompose_dlm` takes them.

        Raises:
            ValueError: on a `level` outside (0, 1), or what `view` or `decompose_dlm` refuses.
        """
        if not 0.0 < level < 1.0:
            msg = f"level must lie in (0, 1), got {level}"
            raise ValueError(msg)
        ends = [0.5 - level / 2.0, 0.5 + level / 2.0]
        per_fit = [
            view(decompose_dlm(fit, draws=draws, seed=seed + k)) for k, fit in self.kept.items()
        ]

        first = per_fit[0]
        return ComponentBounds(
            base=None
            if first.base is None
            else _union([c.base for c in per_fit if c.base is not None], ends),
            seasonality=None
            if first.seasonality is None
            else _union([c.seasonality for c in per_fit if c.seasonality is not None], ends),
            contributions=MappingProxyType(
                {
                    k: _union([c.contributions[k] for c in per_fit], ends)
                    for k in first.contributions
                }
            ),
        )

    def coefficient_bounds(self, level: float = 0.9) -> Mapping[str, Bounds]:
        """Each effect's coefficient at the last period, by its column: the union of the kept
        fits' central intervals holding `level`. A fit's interval is its Student t at the last
        period (`DlmFit.last`), the coefficient's posterior given every period and every test, or
        with a signed effect the quantiles of its draws of the posterior truncated to the signs
        there.

        Raises:
            ValueError: on a `level` outside (0, 1), or kept specs whose transforms differ, where
                each coefficient multiplies another column.
        """
        if not 0.0 < level < 1.0:
            msg = f"level must lie in (0, 1), got {level}"
            raise ValueError(msg)
        if len({tuple(e.transform for e in f.spec.effects) for f in self.kept.values()}) > 1:
            msg = (
                "the kept specs' transforms differ, so each coefficient multiplies another column "
                "in each, and no one interval bounds it"
            )
            raise ValueError(msg)
        ends = [0.5 - level / 2.0, 0.5 + level / 2.0]
        spec = self.fit.spec
        first = spec.trend.order if spec.trend else 0
        coordinates = list(range(first, first + len(spec.effects)))
        lows, highs = [], []
        for fit in self.kept.values():
            if fit.signed is None:
                state, covariance, dof, _ = fit.last()
                mean = state[coordinates]
                spread = np.sqrt(np.diag(covariance)[coordinates])
                half = stats.t.ppf(ends[1], dof) * spread
                lo, hi = mean - half, mean + half
            else:
                draws = fit.signed.pooled.states[:, -1][:, coordinates]
                lo, hi = np.quantile(draws, ends, axis=0)
            lows.append(lo)
            highs.append(hi)
        lower, upper = np.min(lows, axis=0), np.max(highs, axis=0)
        return MappingProxyType(
            {e.column: Bounds(float(lower[j]), float(upper[j])) for j, e in enumerate(spec.effects)}
        )

    def channels(self, at: int = -1) -> dict[str, response.Channel]:
        """The best fit's channels, `DlmFit.channels`: the plan at the likelihood's best point.

        Raises:
            ValueError: on what `DlmFit.channels` refuses.
        """
        return self.fit.channels(at)

    def readings(self, draws: int, seed: int) -> tuple[tuple[response.Channel, ...], ...]:
        """`draws` readings of the channels at the last period, as `chc.allocation` reads a
        posterior's draws: each a tuple of every transformed effect's channel, in the order of
        `channels()`, at one draw of their coefficients. `allocate` takes one reading at a time.

        The draws are spread over the fitted specs (`kept`) evenly, a mixture in which no spec the
        likelihood ratio keeps is ranked above another: each takes `draws // len(kept)`, and the
        rest go one each to specs drawn at random from `seed`. Each reading's kernels and curves
        are its spec's. A spec's coefficients are drawn from their posterior at the last period
        given every period and every test, its multivariate Student t there (`DlmFit.last`), or
        with a signed effect from its fit's draws of the posterior truncated to the signs, without
        replacement; each spec from a stream of `seed` and its index of its own. The readings come
        in an order drawn from `seed`, so any of them are a sample of the rest. Their quantiles are
        the mixture's, not the wider union of the kept fits' intervals that `bounds` and
        `coefficient_bounds` give.

        Raises:
            ValueError: on `draws` below 1, a choice with no transformed effect, a signed spec's
                share above the draws its fit holds, or JAX in single precision.
        """
        if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
            msg = f"draws must be a positive integer, got {draws!r}"
            raise ValueError(msg)
        rng = np.random.default_rng(seed)
        shares = np.full(len(self.kept), draws // len(self.kept))
        shares[rng.choice(len(self.kept), draws % len(self.kept), replace=False)] += 1
        readings: list[tuple[response.Channel, ...]] = []
        for share, (k, fit) in zip(shares.tolist(), self.kept.items(), strict=True):
            if not share:
                continue
            media = _media(fit.spec)
            parts = [(t.kernel(), t.curve()) for _, _, t in media]
            # numpy pads a seed with zeros, so `(seed, 0)` would be the stream that spreads the
            # draws and orders them
            coefficients = _coefficient_draws(
                fit, [i for i, _, _ in media], share, np.random.default_rng((seed, k + 1))
            )
            readings += [
                tuple(
                    response.Channel(kernel, curve, float(b))
                    for (kernel, curve), b in zip(parts, row, strict=True)
                )
                for row in coefficients
            ]
        return tuple(readings[i] for i in rng.permutation(draws))

    def forecast(
        self, horizon: int, plan: Mapping[str, npt.ArrayLike], draws: int, seed: int
    ) -> DlmForecastDraws:
        """`draws` draws of the KPI over the `horizon` periods after the data, the effects' columns
        taken from `plan`, each a draw of one fitted spec's forecast (`forecast_draws`), spread
        over the fitted specs as `readings` spreads its draws and in an order drawn from `seed`. A
        transformed effect's plan is its spend, which each spec carries over and saturates by its
        own transform. Each spec draws from a stream of `seed` and its index, apart from those
        `readings` takes for the same seed. `acceptance` is over every spec's paths.

        Raises:
            ValueError: on `draws` below 1, or what `forecast_draws` refuses.
        """
        if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
            msg = f"draws must be a positive integer, got {draws!r}"
            raise ValueError(msg)
        # numpy pads a seed with zeros, so a trailing 1 keeps every stream apart from `readings`'
        rng = np.random.default_rng((seed, 0, 1))
        shares = np.full(len(self.kept), draws // len(self.kept))
        shares[rng.choice(len(self.kept), draws % len(self.kept), replace=False)] += 1
        kpi: list[_Array] = []
        held = tried = 0
        for share, (k, fit) in zip(shares.tolist(), self.kept.items(), strict=True):
            if not share:
                continue
            _drawable(fit)
            x = _planned(fit, horizon, plan)
            _, rows, kept, drawn = _carried(fit, x, share, np.random.default_rng((seed, k + 1, 1)))
            kpi.append(rows)
            held, tried = held + kept, tried + drawn
        log.info(
            "dlm forecast drawn over %d fitted specs: %d draws over %d periods, %.3f of the "
            "paths kept every sign",
            len(kpi),
            draws,
            horizon,
            held / tried,
            extra={
                "specs": len(kpi),
                "draws": draws,
                "horizon": horizon,
                "acceptance": held / tried,
            },
        )
        dates = calendar(self.fit.dates[-1], horizon + 1, self.fit.cadence)[1:]
        return DlmForecastDraws(dates, np.concatenate(kpi)[rng.permutation(draws)], held / tried)


def _coefficient_draws(
    fit: DlmFit, coordinates: list[int], count: int, rng: np.random.Generator
) -> _Array:
    """`count` draws of the states at `coordinates` at the fit's last period given every period,
    `(count, coordinates)`: its multivariate Student t there (`DlmFit.last`), `n_T` degrees of
    freedom, location `m_T` and scale `C_T`, or a signed fit's own draws, taken without
    replacement.

    Raises:
        ValueError: on a signed fit holding fewer than `count` draws.
    """
    if fit.signed is not None:
        pool = fit.signed.pooled.states[:, -1][:, coordinates]
        if count > pool.shape[0]:
            msg = f"a share of {count} draws is above the {pool.shape[0]} the signed fit holds"
            raise ValueError(msg)
        return pool[rng.choice(pool.shape[0], count, replace=False)]
    state, covariance, dof, _ = fit.last()
    values, vectors = np.linalg.eigh(covariance[np.ix_(coordinates, coordinates)])
    root = vectors * np.sqrt(np.clip(values, 0.0, None))
    normal = rng.standard_normal((count, len(coordinates))) @ root.T
    if math.isinf(dof):
        return state[coordinates] + normal
    return state[coordinates] + normal / np.sqrt(rng.chisquare(dof, count) / dof)[:, None]


def _union(parts: list[_Array], ends: list[float]) -> Bounds:
    """From the lowest lower end to the highest upper end of each part's interval."""
    lo, hi = np.quantile(np.stack(parts), ends, axis=1)
    return Bounds(float(lo.min()), float(hi.max()))


def choose_dlm(
    panel: MediaPanel,
    geo: Geo,
    spec: DlmSpec,
    grid: DiscountGrid,
    *,
    level: float = 0.95,
    estimates: Mapping[str, Sequence[LiftEstimate]] | None = None,
) -> DlmChoice:
    """Filter `spec` on one geo at every discount of `grid` and choose by one-step likelihood;
    fit the specs a likelihood-ratio test at `level` does not reject, for `DlmChoice.bounds`.

    With a signed effect the likelihood is the model's without the signs, and only a kept spec's
    fit draws its posterior truncated to them. Each end of the grid that a kept spec reaches where
    the discount goes on past it is named in `DlmChoice.edges` and logged as a warning.

    `estimates`, each channel's lift tests, enter each spec's likelihood, its tests' log density
    given its KPI added to the filter's (`DlmChoice.lifts`), and each kept fit's posterior, as
    `fit_dlm` reads them. The kept specs are those within `chi2_p(level) / 2` of the best, as
    `chc.dlm.confidence_set` keeps them, on that likelihood.

    Raises:
        ValueError: on a `level` outside (0, 1), and what `DiscountGrid.specs`, `fit_dlm` or
            `mixdyn.models.dlm_lift` refuses.
        RuntimeError: on a kept spec's sampler that stops from every seed it starts from.
    """
    specs = grid.specs(spec)
    filtered = tuple(_filter(panel, geo, s) for s in specs)
    lifts = tuple(
        lifted(f, lift_reading(panel, geo, s, estimates)).log_density
        for s, f in zip(specs, filtered, strict=True)
        if estimates
    )
    loglik = np.array([f.log_likelihood for f in filtered]) + (lifts or 0.0)
    best = int(np.argmax(loglik))
    parameters = grid.parameters(spec)
    kept = within(loglik, parameters, level)
    edges = grid.edges([specs[k] for k in kept], specs[best])
    log.info(
        "dlm discounts chosen for geo %s: %d of %d specs kept at %d parameters, the best at %.4g",
        geo,
        len(kept),
        len(specs),
        parameters,
        loglik[best],
        extra={
            "geo": str(geo),
            "specs": len(specs),
            "kept": len(kept),
            "parameters": parameters,
            "best": best,
        },
    )
    _warn(log, geo, edges)
    fits = {k: _kept_fit(panel, geo, specs[k], estimates=estimates) for k in kept}
    return DlmChoice(specs, filtered, best, MappingProxyType(fits), edges, lifts)


def within(loglik: npt.ArrayLike, parameters: int, level: float) -> tuple[int, ...]:
    """The indices of the log-likelihoods within `chi2_parameters(level) / 2` of the highest,
    in order: the specs a likelihood-ratio test at `level` does not reject, as
    `chc.dlm.confidence_set` keeps fits, on any log-likelihood, the tests' with the KPI's.

    Raises:
        ValueError: on a `parameters` that is not a positive integer, or a `level` outside (0, 1).
    """
    if isinstance(parameters, bool) or not isinstance(parameters, int) or parameters < 1:
        msg = f"parameters must be a positive integer, got {parameters!r}"
        raise ValueError(msg)
    if not 0.0 < level < 1.0:
        msg = f"level must lie in (0, 1), got {level}"
        raise ValueError(msg)
    values = np.asarray(loglik, dtype=np.float64)
    margin = float(special.chdtri(parameters, 1.0 - level)) / 2.0
    return tuple(int(i) for i in np.flatnonzero(values >= values.max() - margin))


_REDRAWS = 2
"""How many times a kept spec whose signed chains have not mixed is drawn again."""
_SEEDS = 4
"""How many seeds a kept spec's sampler starts from before its failure is raised."""


def _kept_fit(
    panel: MediaPanel,
    geo: Geo,
    spec: DlmSpec,
    *,
    draws: int = 1000,
    estimates: Mapping[str, Sequence[LiftEstimate]] | None = None,
) -> DlmFit:
    """A kept spec's fit, `fit_dlm`'s at `draws` a chain (`_drawn`) given `estimates`; where its
    signed chains have not mixed, drawn again at twice the draws, at most `_REDRAWS` times
    (ADR 0022). A fit whose chains still have not mixed says so (`ConstrainedDraws.mixed`), and the
    choice keeps it.

    Raises:
        ValueError: on what `fit_dlm` refuses.
        RuntimeError: on a sampler that stops from every seed `_drawn` tries.
    """
    fit = _drawn(panel, geo, spec, draws, estimates)
    for _ in range(_REDRAWS):
        if fit.signed is None or fit.signed.mixed:
            break
        draws *= 2
        log.info(
            "dlm signed chains of a kept spec have not mixed: drawn again at %d a chain",
            draws,
            extra={"draws": draws, "rhat": fit.signed.rhat, "ess": fit.signed.ess},
        )
        fit = _drawn(panel, geo, spec, draws, estimates)
    return fit


def _drawn(
    panel: MediaPanel,
    geo: Geo,
    spec: DlmSpec,
    draws: int,
    estimates: Mapping[str, Sequence[LiftEstimate]] | None,
) -> DlmFit:
    """`fit_dlm`'s fit at `draws` a chain given `estimates`, from the first of `_SEEDS` seeds
    whose sampler runs to its end (ADR 0023). CHC's exact Hamiltonian trajectory stops with a
    `RuntimeError`, the only one `chc.dlm` raises, where it meets more walls than its cap; whether
    it does depends on where the chains start, which the next seed moves.

    Raises:
        ValueError: on what `fit_dlm` refuses.
        RuntimeError: on the sampler stopping from the last seed as well.
    """
    for seed in range(_SEEDS - 1):
        try:
            return fit_dlm(panel, geo, spec, draws=draws, seed=seed, estimates=estimates)
        except RuntimeError as error:
            log.warning(
                "dlm sampler of a kept spec stopped from seed %d: %s; drawn from seed %d",
                seed,
                error,
                seed + 1,
                extra={"seed": seed, "draws": draws, "error": str(error)},
            )
    return fit_dlm(panel, geo, spec, draws=draws, seed=_SEEDS - 1, estimates=estimates)
