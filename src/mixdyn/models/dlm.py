"""The engine's dynamic linear model: `chc.dlm` on one geo of a media panel.

The state is a trend (`chc.dlm.Polynomial`), one random-walk coefficient per effect
(`chc.dlm.Regression`), and a season (`chc.dlm.Seasonal`), in that order. Time is the panel's
calendar: a period with no observation is a missing step, filtered without an update, and its
regressors enter as zero. The decomposition is read off the smoothed states: the trend's level,
the season's observed coordinates, and each effect's coefficient times its column. An effect may
be held to a sign: the states are then the posterior truncated to the signs, drawn by
`chc.dlm.constrained_sample`, and the decomposition is read off the draws' mean. A forecast is the
model's own, its effects' columns given by a plan, and splits the same way; a signed fit's is
drawn (`forecast_draws`), its coefficients held to their signs over the periods forecast too.

An effect may name one of the panel's calendar events (`MediaPanel.events`): its column is 1 in
the periods the event falls in, and at a discount of 1 its coefficient is the KPI the event adds
there, one for every year it recurs, beside the level and the season.

An effect may name a control the panel declares a lever (`mixdyn.data.Lever`). The model filters
its raw column, as any control's, so its states, its likelihood and its forecasts are those of the
same fit without the declaration; only the decomposition reads the lever against its
counterfactual level `b` (`MediaPanel.counterfactual`): its contribution is the coefficient times
its departure from `b`, `gamma_t (x_t - b)`, and the base takes `gamma_t b`, so the base is the KPI
at the counterfactual level at every period, its coefficient walking or not.

A channel's effect may carry a transform (`mixdyn.transforms.Transform`): its column is then the
channel's spend carried over and saturated, and its coefficient the channel's at each period, so
the fit hands the channel to a planner as `chc.response` reads it (`DlmFit.channels`).

A fit may read lift tests (`mixdyn.calibration.LiftEstimate`), each an observation of its channel's
coefficient path over its readout (`mixdyn.models.dlm_lift`): the posterior is then the KPI's
conditioned on them, and every reading of the fit, its decomposition, its channels, its draws and
its forecasts, is that posterior's. The filter stays the KPI's.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
from chc.dlm import (
    ConstrainedDraws,
    DLMFit,
    DynamicLinearModel,
    Polynomial,
    Prior,
    Regression,
    Seasonal,
    Sign,
    SmoothedStates,
    constrained_sample,
    forecast,
    forward_filter,
    smooth,
)
from scipy import stats

from mixdyn.data.panel import calendar
from mixdyn.models.dlm_lift import Lifted, lift_reading, lifted, signed_sample
from mixdyn.transforms.saturation import Transform

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from chc import response

    from mixdyn.calibration import LiftEstimate
    from mixdyn.data import Cadence, Geo, MediaPanel

type _Array = npt.NDArray[np.float64]

log = logging.getLogger(__name__)

_SPREAD = 100.0
"""The vague prior's variance over its scale's square: a standard deviation ten times the scale."""
_PRIOR_DOF = 1.0


@dataclass(frozen=True, slots=True)
class Trend:
    """A level (`order=1`), or a level and a slope (`order=2`), with one discount."""

    order: Literal[1, 2]
    discount: float


@dataclass(frozen=True, slots=True)
class Season:
    """Harmonics 1 to `harmonics` of a `period` counted in the panel's periods (52.18 weeks to a
    year is allowed), with one discount."""

    period: float
    harmonics: int
    discount: float


@dataclass(frozen=True, slots=True)
class Effect:
    """A channel or control of the panel, with a random-walk coefficient and its discount.

    While its column is zero the coefficient is held, not discounted (`hold_when_idle`), so a
    channel that goes dark does not come back with its variance grown by `discount ** -steps`.

    `sign` holds the coefficient to `"positive"` or `"negative"` at every step. The fit is then
    the posterior truncated to the sign, not a clipped one: what the data pull against the sign
    moves into the other states, the base first.

    `transform`, on a channel, carries its spend over and saturates it before the coefficient
    multiplies it: the column is the transform's (`Transform.column`), and the coefficient times
    it is the channel's response at the period.

    Raises:
        ValueError: on a sign that is neither, or a transform that is not a `Transform`.
    """

    column: str
    discount: float
    hold_when_idle: bool = True
    sign: Sign | None = None
    transform: Transform | None = None

    def __post_init__(self) -> None:
        if self.sign not in (None, "positive", "negative"):
            msg = f"effect {self.column!r}: a sign is 'positive' or 'negative', got {self.sign!r}"
            raise ValueError(msg)
        if self.transform is not None and not isinstance(self.transform, Transform):
            msg = f"effect {self.column!r}: a transform is a Transform, got {self.transform!r}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class DlmSpec:
    """The model's structure. `form` and `variance_discount` are `chc.dlm`'s.

    Raises:
        ValueError: on an effect named twice, a block `chc.dlm` refuses, or a signed effect in a
            model whose form is not additive or whose variance discount is not 1, which
            `chc.dlm.constrained_sample` does not draw.
    """

    effects: tuple[Effect, ...]
    trend: Trend | None = None
    season: Season | None = None
    form: Literal["additive", "multiplicative"] = "additive"
    variance_discount: float = 1.0

    def __post_init__(self) -> None:
        columns = [e.column for e in self.effects]
        twice = sorted({c for c in columns if columns.count(c) > 1})
        if twice:
            msg = f"effects name {twice} more than once"
            raise ValueError(msg)
        self.blocks()
        if self.signs() and (self.form != "additive" or self.variance_discount != 1.0):
            msg = (
                f"signed effects {[e.column for e in self.effects if e.sign]} need "
                "form='additive' and a variance discount of 1"
            )
            raise ValueError(msg)

    def signs(self) -> dict[int, Sign]:
        """Each signed effect's sign, keyed by its column of the design: `chc.dlm`'s `signs`."""
        return {i: e.sign for i, e in enumerate(self.effects) if e.sign is not None}

    def season_observed(self) -> list[bool]:
        """Per season coordinate, whether the observation reads it: the first of each harmonic's
        pair, and the single coordinate of a harmonic at the Nyquist frequency."""
        if self.season is None:
            return []
        observed: list[bool] = []
        for k in range(1, self.season.harmonics + 1):
            observed += [True] if 2 * k == self.season.period else [True, False]
        return observed

    def blocks(self) -> tuple[Polynomial | Regression | Seasonal, ...]:
        """The `chc.dlm` blocks, in state order."""
        trend = [Polynomial(self.trend.order, self.trend.discount)] if self.trend else []
        effects = [Regression(1, e.discount, e.hold_when_idle) for e in self.effects]
        season = []
        if self.season:
            harmonics = tuple(range(1, self.season.harmonics + 1))
            season = [Seasonal(self.season.period, harmonics, self.season.discount)]
        return (*trend, *effects, *season)


@dataclass(frozen=True, slots=True, eq=False)
class DlmFit:
    """The model fitted to one geo's calendar.

    `filtered` and `smoothed` are `chc.dlm`'s outputs. The decomposition is from the smoothed
    means: `base` (None without a trend), `seasonality` (None without a season), one contribution
    per effect, and `estimated`, their sum.

    With a signed effect, `signed` holds the draws of the posterior truncated to the signs and
    the decomposition is from their mean; `smoothed` stays the model's without the signs.
    `signed.mixed` says whether the draws can be read.

    `x` holds the effects' columns as the model reads them, a transformed effect's its channel's
    saturated adstock, and `channels` hands the transformed effects to a planner. `spend` holds each
    transformed effect's spend as the panel gave it, which a forecast carries over into the plan's.

    `lifted`, where the fit read lift tests, is the posterior given them too
    (`mixdyn.models.dlm_lift.Lifted`): `smoothed`, the decomposition and `signed` are then that
    posterior's, and `filtered` stays the KPI's filter, whose one-step errors the tests are not.

    `baselines` holds each effect the panel declares a lever at its counterfactual level, in the
    control's units: its contribution is its coefficient times its departure from that level and
    the base holds the coefficient times the level, here and in a forecast's decomposition.
    """

    spec: DlmSpec
    dates: npt.NDArray[np.datetime64]
    cadence: Cadence
    y: _Array
    x: _Array
    filtered: DLMFit
    smoothed: SmoothedStates
    signed: ConstrainedDraws | None
    base: _Array | None
    seasonality: _Array | None
    contributions: Mapping[str, _Array]
    estimated: _Array
    spend: Mapping[str, _Array]
    lifted: Lifted | None = None
    baselines: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))

    @property
    def log_likelihood(self) -> float:
        """The KPI filter's one-step log-likelihood, and with tests their log density given the
        KPI: the spec's log-likelihood of everything the fit read."""
        tests = 0.0 if self.lifted is None else self.lifted.log_density
        return self.filtered.log_likelihood + tests

    def last(self) -> tuple[_Array, _Array, float, float]:
        """The state at the last period given every period and every test the fit read, without
        the signs: Student t with the third's degrees of freedom about the first, its scale matrix
        the second, stated at the fourth, the variance's estimate. Without tests it is the
        filter's."""
        if self.lifted is None:
            f = self.filtered
            return f.mean[-1], f.covariance[-1], float(f.dof[-1]), float(f.scale[-1])
        t = self.lifted
        return t.mean, t.covariance, t.dof, t.scale

    def channels(self, at: int = -1) -> dict[str, response.Channel]:
        """Each effect with a transform as `chc.response` reads it, by its column: its transform's
        kernel and curve, and its coefficient at period `at` of the fit, the last by default.

        The coefficient is the state's posterior mean given the periods up to `at`, the filter's,
        so that a channel run over the panel's spend up to `at` returns there the coefficient times
        the effect's column. With a signed effect it is the mean of the draws of the posterior
        truncated to the signs, which read every period, and with tests the posterior's mean given
        them, which read their readouts' periods: only the last period is offered of either. A
        negative coefficient is handed on as it is; `chc.allocation` refuses to plan on it.

        Raises:
            ValueError: on a fit with no transformed effect, a period outside the fit, a period
                other than the last with a signed effect or tests, or JAX in single precision.
        """
        periods = self.dates.size
        if isinstance(at, bool) or not isinstance(at, int) or not -periods <= at < periods:
            msg = f"at must be a period of the fit's {periods}, from {-periods} to {periods - 1}"
            raise ValueError(msg)
        media = _media(self.spec)
        last = at % periods == periods - 1
        if self.signed is not None and last:
            mean = self.signed.pooled.states[:, -1].mean(axis=0)
        elif self.signed is not None:
            msg = (
                "a signed fit's coefficients are drawn given every period, so its channels are "
                f"read at the last, not at {at}"
            )
            raise ValueError(msg)
        elif self.lifted is None:
            mean = self.filtered.mean[at]
        elif last:
            mean = self.lifted.mean
        else:
            msg = (
                "a fit read with tests holds the posterior given them at the last period, so its "
                f"channels are read at the last, not at {at}"
            )
            raise ValueError(msg)
        return {column: t.channel(float(mean[i])) for i, column, t in media}


def _media(spec: DlmSpec) -> list[tuple[int, str, Transform]]:
    """Each effect with a transform: its coordinate in the state, its column and its transform.

    Raises:
        ValueError: on a spec with none.
    """
    first = spec.trend.order if spec.trend else 0
    media = [
        (first + i, e.column, e.transform)
        for i, e in enumerate(spec.effects)
        if e.transform is not None
    ]
    if not media:
        msg = "no effect carries a transform, so the fit has no channel to hand to a planner"
        raise ValueError(msg)
    return media


def _series(panel: MediaPanel, geo: Geo, spec: DlmSpec) -> tuple[_Array, _Array]:
    """The geo's KPI, NaN where missing, and its effects' columns, a missing value as zero; an
    effect with a transform reads its channel's spend carried over and saturated."""
    if geo not in panel.geos:
        msg = f"the panel has no geo {geo!r}; it has {list(panel.geos)}"
        raise ValueError(msg)
    g = panel.geos.index(geo)
    columns = {str(c): panel.spend[g, :, j] for j, c in enumerate(panel.channels)}
    columns |= {c: panel.control_values[g, :, j] for j, c in enumerate(panel.controls)}
    columns |= {e: panel.event_column(e) for e in panel.events}
    missing = [e.column for e in spec.effects if e.column not in columns]
    if missing:
        msg = f"the panel has no channel or control named {missing}"
        raise ValueError(msg)
    controls = [
        e.column
        for e in spec.effects
        if e.transform and (e.column in panel.controls or e.column in panel.events)
    ]
    if controls:
        msg = f"a transform carries a channel's spend over; {controls} are the panel's controls"
        raise ValueError(msg)
    x = np.column_stack(
        [
            e.transform.column(columns[e.column]) if e.transform else columns[e.column]
            for e in spec.effects
        ]
        or [np.empty((panel.dates.size, 0))]
    )
    return panel.kpi[g], np.where(np.isfinite(x), x, 0.0)


def vague_prior(panel: MediaPanel, geo: Geo, spec: DlmSpec) -> Prior:
    """The engine's default prior: weakly informative, on the series' own scale.

    Every coordinate has mean zero but the level, which starts at the KPI's mean. Each standard
    deviation is ten times the scale of its coordinate's part of the forecast: the KPI's standard
    deviation `s` for the level and the season, `s / T` for the slope over `T` periods, and
    `s / rms(x)` for an effect. The variance is learned from one degree of freedom at `s²`. The
    KPI's mean and standard deviation are the only numbers the data supply twice.

    Raises:
        ValueError: on fewer than two observed periods, or a KPI that does not vary.
    """
    y, x = _series(panel, geo, spec)
    observed = np.isfinite(y)
    if observed.sum() < 2:
        msg = f"geo {geo!r} has {int(observed.sum())} observed periods; a prior needs two"
        raise ValueError(msg)
    s2 = float(np.var(y[observed], ddof=1))
    if not s2 > 0.0:
        msg = f"geo {geo!r}'s KPI does not vary over its observed periods, so it sets no scale"
        raise ValueError(msg)
    mean: list[float] = []
    variance: list[float] = []
    if spec.trend:
        mean += [float(np.mean(y[observed])), 0.0][: spec.trend.order]
        variance += [s2, s2 / panel.dates.size**2][: spec.trend.order]
    for j in range(len(spec.effects)):
        square = float(np.mean(x[observed, j] ** 2))
        mean.append(0.0)
        variance.append(s2 / square if square > 0.0 else s2)
    seasonal = len(spec.season_observed())
    mean += [0.0] * seasonal
    variance += [s2] * seasonal
    return Prior(np.array(mean), _SPREAD * np.diag(variance), s2, _PRIOR_DOF)


def _filter(panel: MediaPanel, geo: Geo, spec: DlmSpec, prior: Prior | None = None) -> DLMFit:
    """`chc.dlm`'s forward filter of the geo through `spec`, from `prior` or `vague_prior`'s."""
    y, x = _series(panel, geo, spec)
    if prior is None:
        prior = vague_prior(panel, geo, spec)
    model = DynamicLinearModel(spec.blocks(), prior, spec.form, spec.variance_discount)
    return forward_filter(model, y, x)


def _baselines(panel: MediaPanel, spec: DlmSpec) -> Mapping[str, float]:
    """Each effect the panel declares a lever, at its counterfactual level.

    Raises:
        ValueError: on a lever in a model without a trend, whose base would hold its level.
    """
    levers = [e.column for e in spec.effects if e.column in panel.levers]
    if levers and spec.trend is None:
        msg = (
            f"the panel measures {levers} against a counterfactual level, which the base holds: "
            "a DLM that reads a lever needs a trend"
        )
        raise ValueError(msg)
    return MappingProxyType({c: panel.counterfactual(c) for c in levers})


def _decompose(
    spec: DlmSpec, states: _Array, x: _Array, baselines: Mapping[str, float]
) -> tuple[_Array | None, _Array | None, dict[str, _Array]]:
    """The base, the seasonality and each effect's contribution in a path of state means, a
    lever's against its level in `baselines`, which the base takes."""
    trend = spec.trend.order if spec.trend else 0
    base = None
    if spec.trend:
        base = states[:, 0].copy()
        for i, e in enumerate(spec.effects):
            if e.column in baselines:
                base = base + states[:, trend + i] * baselines[e.column]
    seasonality = None
    if spec.season:
        observed = np.array(spec.season_observed())
        seasonality = states[:, trend + len(spec.effects) :][:, observed].sum(axis=1)
    contributions = {
        e.column: states[:, trend + i] * (x[:, i] - baselines[e.column])
        if e.column in baselines
        else states[:, trend + i] * x[:, i]
        for i, e in enumerate(spec.effects)
    }
    return base, seasonality, contributions


def fit_dlm(
    panel: MediaPanel,
    geo: Geo,
    spec: DlmSpec,
    prior: Prior | None = None,
    *,
    draws: int = 1000,
    seed: int = 0,
    estimates: Mapping[str, Sequence[LiftEstimate]] | None = None,
) -> DlmFit:
    """Filter and smooth one geo of `panel` through the model `spec` describes.

    `prior` is on the state in the order of `spec.blocks()`; by default it is `vague_prior`'s.
    With a signed effect, four chains of `draws` each, after as many of warm-up, are drawn from
    the posterior truncated to the signs from `seed`; `draws` and `seed` are not read otherwise.

    `estimates`, each channel's lift tests, each placed by its readout's `end`, condition the
    posterior on their lifts as `mixdyn.models.dlm_lift` reads them; with a signed effect a test
    reads a signed channel. An empty mapping reads none.

    A control the panel declares a lever is read against its counterfactual level in the
    decomposition (`DlmFit.baselines`); the states are those of the same fit without the
    declaration.

    Raises:
        ValueError: on what `mixdyn.models.dlm_lift.lift_reading` and `lifted` refuse of the
            estimates, a test of a channel not held to its sign in a signed fit, and a lever in a
            model without a trend.
    """
    y, x = _series(panel, geo, spec)
    baselines = _baselines(panel, spec)
    filtered = _filter(panel, geo, spec, prior)
    reading = lift_reading(panel, geo, spec, estimates) if estimates else None
    tests = None if reading is None else lifted(filtered, reading)
    smoothed = smooth(filtered) if tests is None else tests.smoothed
    signed = None
    states = smoothed.mean
    if spec.signs():
        if reading is None:
            signed = constrained_sample(filtered, spec.signs(), draws, seed)
        else:
            signed = signed_sample(filtered, spec.signs(), reading, draws, seed)
        states = signed.pooled.states.mean(axis=0)

    base, seasonality, contributions = _decompose(spec, states, x, baselines)
    shown = [base, seasonality, *contributions.values()]
    g = panel.geos.index(geo)
    names = [str(c) for c in panel.channels]
    return DlmFit(
        spec=spec,
        dates=panel.dates,
        cadence=panel.cadence,
        y=y,
        x=x,
        filtered=filtered,
        smoothed=smoothed,
        signed=signed,
        base=base,
        seasonality=seasonality,
        contributions=MappingProxyType(contributions),
        estimated=np.sum([part for part in shown if part is not None], axis=0),
        spend=MappingProxyType(
            {
                e.column: panel.spend[g, :, names.index(e.column)].copy()
                for e in spec.effects
                if e.transform is not None
            }
        ),
        lifted=tests,
        baselines=baselines,
    )


@dataclass(frozen=True, slots=True, eq=False)
class DlmForecast:
    """The KPI over the periods after a fit's data, forecast 1 to `horizon` steps ahead.

    Each period's KPI is Student t with location `mean`, squared scale `scale` and `dof` degrees
    of freedom. `base`, `seasonality` and `contributions` split `mean` as `DlmFit`'s
    decomposition splits `estimated`.
    """

    dates: npt.NDArray[np.datetime64]
    mean: _Array
    scale: _Array
    dof: _Array
    base: _Array | None
    seasonality: _Array | None
    contributions: Mapping[str, _Array]

    def interval(self, level: float) -> tuple[_Array, _Array]:
        """Each period's central predictive interval holding `level` of its probability.

        Raises:
            ValueError: on a level outside (0, 1).
        """
        if not 0.0 < level < 1.0:
            msg = f"level must lie in (0, 1), got {level!r}"
            raise ValueError(msg)
        half = stats.t.ppf(0.5 + level / 2.0, self.dof) * np.sqrt(self.scale)
        return self.mean - half, self.mean + half


def forecast_dlm(fit: DlmFit, horizon: int, plan: Mapping[str, npt.ArrayLike]) -> DlmForecast:
    """Forecast the `horizon` periods after `fit`'s data, every effect's column taken from `plan`.

    The effects' future columns are given, not forecast: spend is a decision. A transformed
    effect's plan is its spend, carried over from the spend the fit read and saturated, as its
    column was. The first step's evolution variance is held for every later step, so a state's
    variance grows linearly with the horizon rather than as `discount ** -horizon`. A coefficient
    whose planned column is zero is held, as in the fit. A fit read with tests starts from the
    state given them (`DlmFit.last`) and evolves as the KPI's filter does, each step's evolution
    variance the filter's in units of the variance.

    Raises:
        ValueError: on a fit with a signed effect, whose future coefficients keep their signs,
            which a Student t forecast does not (`forecast_draws` draws it); on a horizon below 1,
            a plan whose names are not the effects', or a planned column that is not `horizon`
            finite values, a transformed effect's at or above 0.
    """
    if fit.spec.signs():
        msg = (
            "a forecast of a fit with signed effects is drawn, not Student t: their future "
            "coefficients keep their signs (forecast_draws)"
        )
        raise ValueError(msg)
    x = _planned(fit, horizon, plan)
    ahead = forecast(fit.filtered, horizon, x, evolution="constant")
    mean, scale, dof, states = ahead.mean, ahead.scale, ahead.dof, ahead.state_mean
    if fit.lifted is not None:
        mean, scale, dof, states = _from_tests(fit, ahead.state_scale, x)
    base, seasonality, contributions = _decompose(fit.spec, states, x, fit.baselines)
    return DlmForecast(
        dates=calendar(fit.dates[-1], horizon + 1, fit.cadence)[1:],
        mean=mean,
        scale=scale,
        dof=dof,
        base=base,
        seasonality=seasonality,
        contributions=MappingProxyType(contributions),
    )


def _from_tests(fit: DlmFit, kpi: _Array, x: _Array) -> tuple[_Array, _Array, _Array, _Array]:
    """The forecast's location, squared scale and degrees of freedom of the KPI, and the state's
    location, over the periods `x` covers, from the state at the last period given the tests,
    each step's evolution variance the KPI filter's, `kpi` its forecast's state scales."""
    filtered = fit.filtered
    g = _transition(filtered)
    state, covariance, dof, scale = fit.last()
    ratio = scale / float(filtered.scale[-1])
    first = fit.spec.trend.order if fit.spec.trend else 0
    design = np.tile(filtered.design[-1], (x.shape[0], 1))
    design[:, first : first + x.shape[1]] = x
    previous = filtered.covariance[-1]
    means, states = np.empty(x.shape[0]), np.empty((x.shape[0], state.size))
    scales = np.empty(x.shape[0])
    for k in range(x.shape[0]):
        evolved = g @ previous @ g.T
        state = g @ state
        covariance = g @ covariance @ g.T + ratio * (kpi[k] - 0.5 * (evolved + evolved.T))
        covariance = 0.5 * (covariance + covariance.T)
        previous = kpi[k]
        states[k], means[k] = state, design[k] @ state
        scales[k] = design[k] @ covariance @ design[k] + scale
    return means, scales, np.full(x.shape[0], dof), states


@dataclass(frozen=True, slots=True, eq=False)
class DlmForecastDraws:
    """Draws of the KPI over the periods after a fit's data, `kpi` `(draws, horizon)`, and the
    share of the paths drawn whose signed coefficients kept their signs at every period forecast,
    `acceptance`: how hard the signs bind over the horizon, 1 for a fit without them."""

    dates: npt.NDArray[np.datetime64]
    kpi: _Array
    acceptance: float


def forecast_draws(
    fit: DlmFit, horizon: int, plan: Mapping[str, npt.ArrayLike], draws: int, seed: int
) -> DlmForecastDraws:
    """`draws` draws of the KPI over the `horizon` periods after `fit`'s data, the effects'
    columns taken from `plan` as `forecast_dlm` takes them.

    Each draw starts from the states and the variance at the last period given every period: the
    filter's normal-gamma posterior there, or with a signed effect one of the fit's draws of the
    posterior truncated to the signs, taken with replacement. It is carried through the horizon
    by the model's evolution, the first step's evolution variance held for every later step as
    `forecast_dlm` holds it, and the KPI drawn about each period's mean at the draw's variance. A
    signed fit's path is kept only where every signed coefficient keeps its sign at every period
    forecast, so the draws are the model's with the signs held over the horizon too, a start the
    signs make unlikely to persist weighed down by just that chance. Without signs the draws
    follow `forecast_dlm`'s Student t.

    A start meets the signs and a signed coefficient walks on its own, so it keeps its sign over
    `h` periods with probability at least `C(2h, h) / 4^h` (Sparre Andersen), 0.155 at 13: the
    paths drawn for each one kept number at most 6.5 in expectation with one signed effect at 13
    periods, and 269 with three.

    Raises:
        ValueError: on `draws` below 1; on a fit without signs whose variance discount is below 1,
            whose variance moves over the horizon where a draw holds the last period's; or what
            `forecast_dlm` refuses of `horizon` and `plan`.
    """
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        msg = f"draws must be a positive integer, got {draws!r}"
        raise ValueError(msg)
    _drawable(fit)
    x = _planned(fit, horizon, plan)
    _, kpi, held, tried = _carried(fit, x, draws, np.random.default_rng(seed))
    log.info(
        "dlm forecast drawn: %d draws over %d periods, %.3f of the paths kept every sign",
        draws,
        horizon,
        held / tried,
        extra={"draws": draws, "horizon": horizon, "acceptance": held / tried},
    )
    return DlmForecastDraws(
        calendar(fit.dates[-1], horizon + 1, fit.cadence)[1:], kpi, held / tried
    )


_BATCH = 8192
"""The paths drawn at once while a signed fit's draws are kept."""


def _drawable(fit: DlmFit) -> None:
    """Refuse a fit whose forecast is not drawn from the last period's variance.

    Raises:
        ValueError: on a fit without signs whose variance discount is below 1.
    """
    if fit.signed is None and fit.spec.variance_discount != 1.0:
        msg = (
            "a forecast drawn from the last period's variance needs a variance discount of 1; "
            f"this fit's is {fit.spec.variance_discount} (forecast_dlm reads it)"
        )
        raise ValueError(msg)


def _planned(fit: DlmFit, horizon: int, plan: Mapping[str, npt.ArrayLike]) -> _Array:
    """The effects' columns over the `horizon` periods after the fit's data, `(horizon, effects)`;
    see `forecast_dlm`.

    Raises:
        ValueError: what `forecast_dlm` refuses of `horizon` and `plan`.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        msg = f"horizon must be a positive integer, got {horizon!r}"
        raise ValueError(msg)
    columns = [e.column for e in fit.spec.effects]
    missing = [c for c in columns if c not in plan]
    extra = sorted(set(plan) - set(columns))
    if missing or extra:
        msg = f"the plan must name the effects {columns}: it lacks {missing} and adds {extra}"
        raise ValueError(msg)
    x = np.zeros((horizon, len(columns)))
    for j, effect in enumerate(fit.spec.effects):
        values = np.asarray(plan[effect.column], dtype=np.float64)
        if values.shape != (horizon,):
            msg = f"the plan's {effect.column!r} has shape {values.shape}, not ({horizon},)"
            raise ValueError(msg)
        if not np.isfinite(values).all():
            steps = np.flatnonzero(~np.isfinite(values)).tolist()
            msg = f"the plan's {effect.column!r} is not finite at steps {steps}"
            raise ValueError(msg)
        if effect.transform is None:
            x[:, j] = values
            continue
        if (values < 0.0).any():
            steps = np.flatnonzero(values < 0.0).tolist()
            msg = f"the plan's {effect.column!r} spends below 0 at steps {steps}"
            raise ValueError(msg)
        carried = np.concatenate([fit.spend[effect.column], values])
        x[:, j] = effect.transform.column(carried)[-horizon:]
    return x


def _carried(
    fit: DlmFit, x: _Array, count: int, rng: np.random.Generator
) -> tuple[_Array, _Array, int, int]:
    """`count` draws of the states and the KPI over the periods `x` covers, `(count, horizon, p)`
    and `(count, horizon)`, with the paths that kept every sign and those drawn; see
    `forecast_draws`."""
    filtered = fit.filtered
    horizon = x.shape[0]
    ahead = forecast(filtered, horizon, x, evolution="constant")
    g = _transition(filtered)
    scale = float(filtered.scale[-1])
    roots = np.empty((horizon, *g.shape))
    previous = filtered.covariance[-1]
    for k in range(horizon):
        # the step's evolution variance, in units of the variance: forecast adds it to the
        # evolved covariance and symmetrises the sum, so a coordinate the model holds reads 0
        evolved = g @ previous @ g.T
        roots[k] = _root((ahead.state_scale[k] - 0.5 * (evolved + evolved.T)) / scale)
        previous = ahead.state_scale[k]
    first = fit.spec.trend.order if fit.spec.trend else 0
    design = np.tile(filtered.design[-1], (horizon, 1))
    design[:, first : first + x.shape[1]] = x
    signs = fit.spec.signs()
    signed = np.array([first + j for j in signs], dtype=np.intp)
    direction = np.array([1.0 if s == "positive" else -1.0 for s in signs.values()])

    states, kpi = [], []
    held = tried = 0
    while held < count:
        batch = _BATCH if signs else count - held
        theta, variance = _start(fit, batch, rng)
        sd = np.sqrt(variance)
        path = np.empty((batch, horizon, g.shape[0]))
        for k in range(horizon):
            theta = theta @ g.T + sd[:, None] * (rng.standard_normal(theta.shape) @ roots[k].T)
            path[:, k] = theta
        kept = np.all(path[:, :, signed] * direction >= 0.0, axis=(1, 2))
        tried += batch
        held += int(kept.sum())
        states.append(path[kept])
        noise = rng.standard_normal((int(kept.sum()), horizon))
        kpi.append(np.einsum("bkp,kp->bk", path[kept], design) + sd[kept, None] * noise)
    return np.concatenate(states)[:count], np.concatenate(kpi)[:count], held, tried


def _start(fit: DlmFit, count: int, rng: np.random.Generator) -> tuple[_Array, _Array]:
    """`count` draws of the states and the variance at the fit's last period given every period:
    a signed fit's own draws, with replacement, or the normal-gamma posterior there, the filter's
    or with tests the posterior given them (`DlmFit.last`)."""
    if fit.signed is not None:
        variance = fit.signed.variance.reshape(-1)
        pick = rng.integers(variance.size, size=count)
        last = fit.signed.states[:, :, -1]
        return last.reshape(-1, last.shape[-1])[pick], variance[pick]
    mean, covariance, dof, scale = fit.last()
    variance = np.full(count, scale) if math.isinf(dof) else dof * scale / rng.chisquare(dof, count)
    # the covariance is stated at the variance's estimate; given the variance it scales with it
    root = _root(covariance / scale)
    shock = rng.standard_normal((count, root.shape[0])) @ root.T
    return mean + np.sqrt(variance)[:, None] * shock, variance


def _transition(filtered: DLMFit) -> _Array:
    """`G`, the state's evolution, as `chc.dlm.forecast` applies it: its column `i` is the first
    step's mean from the unit state `e_i`."""
    p = filtered.mean.shape[1]
    x = np.zeros((1, filtered.model.regressors))
    return np.column_stack(
        [forecast(replace(filtered, mean=unit[None]), 1, x).state_mean[0] for unit in np.eye(p)]
    )


def _root(covariance: _Array) -> _Array:
    """A root `L` of a covariance, `L L' = covariance`, exactly 0 on the coordinates it holds
    still and rounding's negative eigenvalues read as 0."""
    live = np.flatnonzero(np.any(covariance != 0.0, axis=1))
    root = np.zeros_like(covariance)
    if live.size:
        values, vectors = np.linalg.eigh(covariance[np.ix_(live, live)])
        root[np.ix_(live, live)] = vectors * np.sqrt(np.clip(values, 0.0, None))
    return root
