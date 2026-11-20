"""The engine's model over geos: `chc.dlm.GeoDLM` on every geo of a media panel.

Each geo has its own trend and season. An effect's coefficient in a geo is the national
coefficient plus the geo's own deviation, static about it. How far the geos spread, the
deviations' prior variance, is chosen by the marginal likelihood and its uncertainty carried by
draws from its posterior (`chc.dlm.fit_geo_spread`). A geo's noise variance is in proportion to its
population when the panel has one, as a count's is, and the same in every geo otherwise; that is
the model's per-capita scaling, since in a model linear in its columns a fit per head is the fit in
totals with those weights. The smoothed states are mixed over the spread's draws, and the
decomposition and each geo's effects are read off the mixture, so a geo's contribution over its
column is its effect's mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
from chc.dlm import (
    GeoDLM,
    GeoSpread,
    Polynomial,
    Prior,
    Regression,
    Seasonal,
    SmoothedStates,
    fit_geo_spread,
    forward_filter_geos,
    smooth,
)
from scipy import stats

from mixdyn.data.panel import refuse_levers
from mixdyn.models.dlm import DlmSpec, Effect, Season, Trend

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mixdyn.data import Cadence, Geo, MediaPanel

type _Array = npt.NDArray[np.float64]

_SPREAD = 100.0
"""The vague prior's variance over its scale's square, as `vague_prior`'s."""
_PRIOR_DOF = 1.0


@dataclass(frozen=True, slots=True)
class GeoDlmSpec:
    """The model's structure over geos. An effect's discount is its national coefficient's, and a
    geo's deviation from it is static; `trend` and `season` are each geo's own. `form` and
    `variance_discount` are `chc.dlm`'s.

    Raises:
        ValueError: on what `DlmSpec` refuses, on a signed effect: the posterior truncated to a
            sign is not drawn over geos; and on a transformed effect: the model over geos reads
            its columns raw.
    """

    effects: tuple[Effect, ...]
    trend: Trend | None = None
    season: Season | None = None
    form: Literal["additive", "multiplicative"] = "additive"
    variance_discount: float = 1.0

    def __post_init__(self) -> None:
        one = self.one_geo()
        if one.signs():
            signed = [e.column for e in self.effects if e.sign]
            msg = f"signed effects {signed} are not fitted over geos"
            raise ValueError(msg)
        transformed = [e.column for e in self.effects if e.transform is not None]
        if transformed:
            msg = f"transformed effects {transformed} are not fitted over geos"
            raise ValueError(msg)

    def one_geo(self) -> DlmSpec:
        """The structure of one geo's model: `fit_dlm`'s spec of the same blocks."""
        return DlmSpec(self.effects, self.trend, self.season, self.form, self.variance_discount)

    def national(self) -> tuple[Regression, ...]:
        """The national blocks: a coefficient for each effect."""
        return tuple(Regression(1, e.discount, e.hold_when_idle) for e in self.effects)

    def regional(self) -> tuple[Polynomial | Regression | Seasonal, ...]:
        """Each geo's blocks: its trend, its deviations from the national coefficients, and its
        season, in that order."""
        trend = [Polynomial(self.trend.order, self.trend.discount)] if self.trend else []
        deviations = [Regression(len(self.effects), 1.0)] if self.effects else []
        season = []
        if self.season:
            harmonics = tuple(range(1, self.season.harmonics + 1))
            season = [Seasonal(self.season.period, harmonics, self.season.discount)]
        return (*trend, *deviations, *season)

    def deviations(self) -> range:
        """The deviations' coordinates within a geo's regional state."""
        start = self.trend.order if self.trend else 0
        return range(start, start + len(self.effects))


def _columns(panel: MediaPanel, spec: GeoDlmSpec) -> _Array:
    """Every geo's effects' columns, `(period, geo, effect)`, a missing value as zero."""
    columns = {str(c): panel.spend[:, :, j] for j, c in enumerate(panel.channels)}
    columns |= {c: panel.control_values[:, :, j] for j, c in enumerate(panel.controls)}
    missing = [e.column for e in spec.effects if e.column not in columns]
    if missing:
        msg = f"the panel has no channel or control named {missing}"
        raise ValueError(msg)
    refuse_levers(panel, [e.column for e in spec.effects], "the DLM over geos")
    if not spec.effects:
        return np.empty((panel.dates.size, len(panel.geos), 0))
    x = np.stack([columns[e.column] for e in spec.effects], axis=2).transpose(1, 0, 2)
    return np.where(np.isfinite(x), x, 0.0)


def _weights(panel: MediaPanel) -> _Array:
    """Each geo's noise variance relative to the others': its population's share of the mean."""
    if panel.population is None:
        return np.ones(len(panel.geos))
    return panel.population / panel.population.mean()


def _design(spec: GeoDlmSpec, periods: int, x: _Array) -> _Array:
    """One geo's least-squares design: its trend, season and effects' columns."""
    t = np.arange(periods, dtype=np.float64)
    columns = [np.ones(periods)]
    if spec.trend and spec.trend.order == 2:
        columns.append(t)
    if spec.season:
        for k in range(1, spec.season.harmonics + 1):
            angle = 2.0 * np.pi * k * t / spec.season.period
            columns += (
                [np.cos(angle)] if 2 * k == spec.season.period else [np.cos(angle), np.sin(angle)]
            )
    return np.column_stack([*columns, x])


def vague_geo_prior(panel: MediaPanel, spec: GeoDlmSpec) -> Prior:
    """The engine's default prior over geos: weakly informative, on each geo's own scale.

    A geo's level starts at its KPI's mean, and every other coordinate at zero. Each standard
    deviation is ten times the scale of its coordinate's part of the forecast, as `vague_prior`'s:
    the geo's KPI standard deviation `s_g` for its level and season, `s_g / T` for its slope, and
    `s / rms(x)` for a national coefficient and for every geo's deviation from it, `s` the geos'
    root mean square `s_g` and `rms(x)` over every geo. The deviations' variance is the vaguest
    spread `fit_geo_spread` considers.

    The variance is learned from one degree of freedom, its guess each geo's least-squares residual
    variance over its relative weight, averaged over the geos. A prior's covariance is stated at
    that guess, so a guess of the KPI's whole variance would tighten every coordinate's prior by the
    noise's share of it once the variance is learned, and with it the vaguest spread.

    Raises:
        ValueError: on a geo whose KPI does not vary over two observed periods, or no geo observed
            in more periods than its least-squares design has columns.
    """
    y = panel.kpi
    x = _columns(panel, spec)
    weights = _weights(panel)
    periods = panel.dates.size
    mean: list[float] = []
    variance: list[float] = []
    national = []
    for e in range(len(spec.effects)):
        square = float(np.mean(x[:, :, e][np.isfinite(y.T)] ** 2))
        national.append(square)
    scales, residuals = [], []
    for g, geo in enumerate(panel.geos):
        seen = np.isfinite(y[g])
        s2 = float(np.var(y[g][seen], ddof=1)) if seen.sum() >= 2 else 0.0
        if not s2 > 0.0:
            msg = f"geo {geo!r}'s KPI does not vary over two observed periods, so it sets no scale"
            raise ValueError(msg)
        scales.append(s2)
        design = _design(spec, periods, x[:, g, :])[seen]
        if seen.sum() > design.shape[1]:
            fitted = np.linalg.lstsq(design, y[g][seen], rcond=None)[0]
            residual = y[g][seen] - design @ fitted
            residuals.append(residual @ residual / (seen.sum() - design.shape[1]) / weights[g])
    if not residuals:
        msg = "no geo is observed in more periods than its trend, season and effects have columns"
        raise ValueError(msg)
    s2_all = float(np.mean(scales))
    vaguest = [s2_all / square if square > 0.0 else s2_all for square in national]
    mean += [0.0] * len(spec.effects)
    variance += vaguest
    seasonal = len(spec.one_geo().season_observed())
    for g in range(len(panel.geos)):
        seen = np.isfinite(y[g])
        if spec.trend:
            mean += [float(np.mean(y[g][seen])), 0.0][: spec.trend.order]
            variance += [scales[g], scales[g] / periods**2][: spec.trend.order]
        mean += [0.0] * len(spec.effects)
        variance += vaguest
        mean += [0.0] * seasonal
        variance += [scales[g]] * seasonal
    return Prior(np.array(mean), _SPREAD * np.diag(variance), float(np.mean(residuals)), _PRIOR_DOF)


@dataclass(frozen=True, slots=True, eq=False)
class GeoDlmFit:
    """The model fitted to every geo of a panel.

    `spread` is `chc.dlm.fit_geo_spread`'s: the deviations' spread, its intervals and draws, and
    the filter at the spread chosen; `smoothed` is the smoothed states mixed over its draws, each
    step's mean and covariance those of the mixture, a covariance the variance, and its degrees of
    freedom the draws' mean. Its marginals are mixtures of Student t, so it is not `exact`. The
    decomposition is from its means, each array `(geo, period)` in the panel's orientation: `base`
    (None without a trend), `seasonality` (None without a season), one contribution per effect, and
    `estimated`, their sum.
    """

    spec: GeoDlmSpec
    dates: npt.NDArray[np.datetime64]
    cadence: Cadence
    geos: tuple[Geo, ...]
    y: _Array
    x: _Array
    spread: GeoSpread
    smoothed: SmoothedStates
    base: _Array | None
    seasonality: _Array | None
    contributions: Mapping[str, _Array]
    estimated: _Array


def fit_geo_dlm(
    panel: MediaPanel,
    spec: GeoDlmSpec,
    prior: Prior | None = None,
    *,
    draws: int = 64,
    seed: int = 0,
    level: float = 0.9,
) -> GeoDlmFit:
    """Fit `spec` to every geo of `panel`, the geos' spread chosen by the marginal likelihood.

    `prior` is on the stacked state, the national coefficients and then each geo's regional
    coordinates in the order of `spec.regional()`; by default it is `vague_geo_prior`'s. `draws`
    spreads are drawn from their posterior with `seed`, and each spread's interval is at `level`.

    Raises:
        ValueError: on a panel of one geo, which has nothing to pool, and on what
            `vague_geo_prior` or `chc.dlm.fit_geo_spread` refuses.
    """
    if len(panel.geos) < 2:
        msg = f"a model over geos needs two geos or more; the panel has {list(panel.geos)}"
        raise ValueError(msg)
    if prior is None:
        prior = vague_geo_prior(panel, spec)
    x = _columns(panel, spec)
    model = GeoDLM(
        spec.national(),
        spec.regional(),
        len(panel.geos),
        prior,
        spec.form,
        spec.variance_discount,
        relative_variance=_weights(panel),
    )
    spread = fit_geo_spread(
        model, panel.kpi.T, x, list(spec.deviations()), draws, seed, level=level
    )
    smoothed = _mixture(spread)
    base, seasonality, contributions = _decompose(spec, len(panel.geos), smoothed.mean, x)
    shown = [base, seasonality, *contributions.values()]
    return GeoDlmFit(
        spec=spec,
        dates=panel.dates,
        cadence=panel.cadence,
        geos=panel.geos,
        y=panel.kpi,
        x=x.transpose(1, 0, 2),
        spread=spread,
        smoothed=smoothed,
        base=base,
        seasonality=seasonality,
        contributions=contributions,
        estimated=np.sum([part for part in shown if part is not None], axis=0),
    )


def _mixture(spread: GeoSpread) -> SmoothedStates:
    """The smoothed states at each of the spread's draws, mixed by its weights: the means' weighted
    mean, and the covariances' plus the means' own spread about it, by the law of total covariance.
    Each draw's means are kept for the second pass and its covariances only summed. The degrees of
    freedom are the draws' weighted mean: every draw has `n_T` at the last step, and with a
    variance discount below 1 the earlier steps' matched ones follow the draw's own variance."""
    live: list[int] = np.flatnonzero(spread.weights).tolist()
    weights = spread.weights[live] / spread.weights[live].sum()

    def at(j: int) -> SmoothedStates:
        return smooth(forward_filter_geos(spread.at(spread.draws[j]), spread.fit.y, spread.x))

    first = at(live[0])
    means = [first.mean]
    covariance = weights[0] * first.covariance
    cross = weights[0] * first.cross_covariance
    dof = weights[0] * first.dof
    for weight, j in zip(weights[1:], live[1:], strict=True):
        smoothed = at(j)
        means.append(smoothed.mean)
        covariance += weight * smoothed.covariance
        cross += weight * smoothed.cross_covariance
        dof += weight * smoothed.dof
    paths = np.array(means)
    mean = np.einsum("j,jtp->tp", weights, paths)
    apart = (paths - mean).transpose(1, 0, 2)
    weighted = weights[None, :, None] * apart
    between = weighted.transpose(0, 2, 1) @ apart
    covariance += 0.5 * (between + between.transpose(0, 2, 1))
    cross += weighted[:-1].transpose(0, 2, 1) @ apart[1:]
    return SmoothedStates(mean, covariance, cross, dof, exact=False)


def _sizes(spec: GeoDlmSpec) -> tuple[int, int]:
    """How many national coordinates the state has, and how many regional ones a geo."""
    return len(spec.effects), sum(b.size for b in spec.regional())


def _decompose(
    spec: GeoDlmSpec, geos: int, states: _Array, x: _Array
) -> tuple[_Array | None, _Array | None, dict[str, _Array]]:
    """Each geo's base, seasonality and contributions in a path of stacked state means,
    `(geo, period)` each."""
    national, regional = _sizes(spec)
    trend = spec.trend.order if spec.trend else 0
    own = states[:, national:].reshape(states.shape[0], geos, regional)
    base = own[:, :, 0].T.copy() if spec.trend else None
    seasonality = None
    if spec.season:
        observed = np.array(spec.one_geo().season_observed())
        seasonality = own[:, :, trend + len(spec.effects) :][:, :, observed].sum(axis=2).T
    contributions = {
        e.column: ((states[:, [i]] + own[:, :, trend + i]) * x[:, :, i]).T
        for i, e in enumerate(spec.effects)
    }
    return base, seasonality, contributions


@dataclass(frozen=True, slots=True, eq=False)
class GeoEffects:
    """Each geo's coefficient on each effect at the last period, `(geo, effect)`: its mean and
    the central interval holding `level` of it, mixed over the spread's draws."""

    effects: tuple[str, ...]
    level: float
    mean: _Array
    lower: _Array
    upper: _Array


def geo_effects(fit: GeoDlmFit, level: float = 0.9) -> GeoEffects:
    """Each geo's coefficients at the last period, the national one plus its deviation, with the
    spread's uncertainty: their mixture over the spread's draws, read as one Student t with the
    mixture's variance and the last step's degrees of freedom.

    Raises:
        ValueError: on a level outside (0, 1).
    """
    if not 0.0 < level < 1.0:
        msg = f"level must lie in (0, 1), got {level!r}"
        raise ValueError(msg)
    geos = len(fit.geos)
    national, regional = _sizes(fit.spec)
    effects = np.arange(len(fit.spec.effects))
    own = national + np.arange(geos)[:, None] * regional + np.array(fit.spec.deviations())[None, :]
    m, c = fit.smoothed.mean[-1], fit.smoothed.covariance[-1]
    mean = m[effects][None, :] + m[own]
    variance = c[effects, effects][None, :] + c[own, own] + 2.0 * c[effects[None, :], own]
    dof = float(fit.smoothed.dof[-1])
    half = stats.t.ppf(0.5 + level / 2.0, dof) * np.sqrt(variance * (1.0 - 2.0 / dof))
    return GeoEffects(
        effects=tuple(e.column for e in fit.spec.effects),
        level=level,
        mean=mean,
        lower=mean - half,
        upper=mean + half,
    )
