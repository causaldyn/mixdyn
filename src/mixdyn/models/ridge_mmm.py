"""A static media-mix model fitted by bounded ridge regression, in Robyn's style (Runge et al.
2024).

    y_t = b0 + sum_m beta_m hill_m(adstock_m(spend_m))_t + sum_c gamma_c control_{c,t}
              + trend_t + season_t + e_t

over the periods of a window. Each channel's adstock runs from the panel's first period, so spend
before the window carries into it, and its Hill's half-saturation point is read off the window's
adstocked values (`mixdyn.transforms`). The coefficients minimise glmnet's ridge objective at
`penalty` (`ridge_path`), every column standardised and the intercept free and unpenalised, on the
window's observed periods. A channel's coefficient is at least 0, as Robyn holds every paid
channel; a control's bounds are its own, `lower <= 0 <= upper`, as glmnet's limits are. Robyn holds
prophet's components to `[0, 1]`, and its trend to `[-1, 0]` where the trend sums below 0.

`penalty` has no units: glmnet's `lambda` is it times the fitted KPI's standard deviation with
divisor n. Robyn's search scales its `lambda_hp` in `[0, 1]` onto glmnet's `lambda`; a model it
recreates from fixed hyperparameters reads `lambda` as glmnet's own.

The trend and the season are the model's own, in place of Robyn's prophet components: a line
through the window's periods, bent at knots spaced evenly inside it (`LinearTrend`), and harmonics
of a season (`FourierSeason`), their coefficients free and penalised as every other.

Robyn refits a model whose intercept comes out below 0 without one, under glmnet's standardisation
for that case. This engine keeps the intercept free and reports it.

A control the panel declares a lever (`mixdyn.data.Lever`) enters as its departure from its
counterfactual level (`MediaPanel.counterfactual`), with no adstock or curve: its contribution is
measured against that level, and the intercept is the KPI there. Its coefficient is the one its
values would take, since the design's columns are centred.

A control may name one of the panel's calendar events (`MediaPanel.events`): its column is the
event's, 1 in the periods it falls in, and its coefficient the KPI it adds there beside the trend
and the season.

Geo tests (`chc.lift.LiftTest`) calibrate it. A test's gap between its groups is the tested
channel's coefficient times the gap in its saturated spend, under the channel's own adstock and Hill
and the window's `K`, plus the groups' noise (`chc.lift`). So each tested channel's tests estimate
its coefficient alone, by least squares on their gaps, and the fit adds their rows to the KPI's:
the joint least squares of the KPI and the tests, each period weighted by its source's noise. The
tests' noise variance is their residuals' about those estimates, a degree of freedom taken off for
each tested channel; the KPI's is the residuals' of the fit without tests, the ridge's effective
degrees of freedom taken off, `sum d^2 / (d^2 + n penalty)` over the standardised design's singular
values. A test's rows enter as an anchor on its channel's coefficient (`ridge_path`), which is the
same objective less a constant.

The scores are Robyn's: `nrmse`, the root mean square error over the KPI's range on the fitted
periods, and `decomp_rssd`, the distance between the channels' shares of the effect and of the
spend over the window, raised by the share of channels whose effect share rounds to 0 at four
decimals.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from mixdyn.models.ridge import Anchor, ridge_path
from mixdyn.transforms import Adstock, Hill, carry_over, channel, half_saturation, saturate

if TYPE_CHECKING:
    from collections.abc import Sequence

    from chc import response
    from chc.lift import LiftTest

    from mixdyn.data import Geo, MediaPanel

type _Array = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Media:
    """A channel of the panel, its spend carried over by `adstock` and saturated by `hill`."""

    channel: str
    adstock: Adstock
    hill: Hill


@dataclass(frozen=True, slots=True)
class Control:
    """A control of the panel, its coefficient held to `lower <= 0 <= upper`."""

    column: str
    lower: float = -math.inf
    upper: float = math.inf

    def __post_init__(self) -> None:
        if not self.lower <= 0.0 <= self.upper:
            msg = (
                f"control {self.column!r}: need lower <= 0 <= upper, got {self.lower}, {self.upper}"
            )
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class LinearTrend:
    """A line through the window's periods, bent at `knots` points spaced evenly inside it: the
    period's place `t` in the window, `trend`, and a hinge `max(0, t - knot)` for each knot,
    `trend_bend_1` on.

    Raises:
        ValueError: on a negative number of knots.
    """

    knots: int = 0

    def __post_init__(self) -> None:
        if self.knots < 0:
            msg = f"a trend's knots are at least 0, got {self.knots}"
            raise ValueError(msg)

    def terms(self) -> tuple[str, ...]:
        """The trend's columns' names."""
        return ("trend", *(f"trend_bend_{k}" for k in range(1, self.knots + 1)))

    def columns(self, periods: int, *, ahead: int = 0) -> _Array:
        """The trend's columns over a window of `periods` periods, then over the `ahead` periods
        after it, where the line runs on from the window's last bend."""
        t = np.arange(periods + ahead, dtype=np.float64)
        knots = periods * np.arange(1, self.knots + 1) / (self.knots + 1)
        return np.column_stack([t, *(np.maximum(0.0, t - k) for k in knots)])


@dataclass(frozen=True, slots=True)
class FourierSeason:
    """Harmonics 1 to `harmonics` of a season `period` periods long (52.18 weeks to a year is
    allowed), counted from the window's first period: `season_cos_h` and `season_sin_h` each.

    Raises:
        ValueError: on a period of 2 or less, or a harmonic at or past half the period, which
            the periods cannot tell from a slower one.
    """

    period: float
    harmonics: int

    def __post_init__(self) -> None:
        if not (2.0 < self.period < math.inf and 1 <= self.harmonics < self.period / 2.0):
            msg = (
                f"need a finite period above 2 and harmonics from 1 to below half of it, got"
                f" {self.period} and {self.harmonics}"
            )
            raise ValueError(msg)

    def terms(self) -> tuple[str, ...]:
        """The season's columns' names."""
        return tuple(
            f"season_{wave}_{h}" for h in range(1, self.harmonics + 1) for wave in ("cos", "sin")
        )

    def columns(self, periods: int) -> _Array:
        """The season's columns over a window of `periods` periods."""
        angle = 2.0 * math.pi * np.arange(periods, dtype=np.float64) / self.period
        return np.column_stack(
            [f(h * angle) for h in range(1, self.harmonics + 1) for f in (np.cos, np.sin)]
        )


@dataclass(frozen=True, slots=True)
class RidgeSpec:
    """The model: its channels, its controls, its trend and season if any, and the ridge
    `penalty > 0`.

    Raises:
        ValueError: on a column named twice, a model with no channel, or a penalty that is not
            positive and finite.
    """

    media: tuple[Media, ...]
    controls: tuple[Control, ...]
    penalty: float
    trend: LinearTrend | None = None
    season: FourierSeason | None = None

    def __post_init__(self) -> None:
        columns = list(self.terms())
        twice = sorted({c for c in columns if columns.count(c) > 1})
        if twice:
            msg = f"the model names {twice} more than once"
            raise ValueError(msg)
        if not self.media:
            msg = "a media-mix model needs at least one channel"
            raise ValueError(msg)
        if not 0.0 < self.penalty < math.inf:
            msg = f"the penalty must be positive and finite, got {self.penalty!r}"
            raise ValueError(msg)

    def terms(self) -> tuple[str, ...]:
        """The design's columns: the channels, the controls, the trend's and the season's."""
        return (
            *(m.channel for m in self.media),
            *(c.column for c in self.controls),
            *(self.trend.terms() if self.trend else ()),
            *(self.season.terms() if self.season else ()),
        )

    def calendar(self, periods: int) -> _Array:
        """The trend's and the season's columns over a window of `periods` periods."""
        blocks = [np.empty((periods, 0))]
        blocks += [part.columns(periods) for part in (self.trend, self.season) if part]
        return np.column_stack(blocks)


@dataclass(frozen=True, slots=True, eq=False)
class Calibration:
    """How geo tests entered a fit: `estimates`, each tested channel's coefficient as its tests
    alone read it, and `weight`, a test period's weight against a KPI period, the KPI's noise
    variance over the tests'."""

    estimates: Mapping[str, float]
    weight: float


@dataclass(frozen=True, slots=True, eq=False)
class RidgeFit:
    """The model fitted to one geo over a window of the panel's calendar.

    Arrays run over the window's periods. `fitted` marks those the regression read, where the KPI
    was observed. `design` holds the terms' columns, each channel's saturated, and `contributions`
    each term's `coefficient * column`; `estimated` is their sum with the intercept, NaN where a
    control is missing. `half_saturation` is each channel's `K`,
    `bound` whether each coefficient sits on a bound, and `spend` each channel's spend over the
    window. `length` is the panel's number of periods, the length of the kernels `channels` hands
    on. `calibration` is how geo tests entered the fit, if any did.
    """

    spec: RidgeSpec
    dates: npt.NDArray[np.datetime64]
    y: _Array
    fitted: npt.NDArray[np.bool_]
    design: _Array
    intercept: float
    coefficients: Mapping[str, float]
    bound: Mapping[str, bool]
    half_saturation: Mapping[str, float]
    contributions: Mapping[str, _Array]
    estimated: _Array
    spend: Mapping[str, float]
    length: int
    calibration: Calibration | None = None

    def nrmse(self) -> float:
        """Robyn's NRMSE: the root mean square error over the KPI's range, on the fitted periods."""
        y, estimated = self.y[self.fitted], self.estimated[self.fitted]
        return float(np.sqrt(np.mean((y - estimated) ** 2)) / (y.max() - y.min()))

    def decomp_rssd(self) -> float:
        """Robyn's DECOMP.RSSD with its penalty on channels of no effect."""
        effect = np.array([np.sum(self.contributions[m.channel]) for m in self.spec.media])
        spend = np.array([self.spend[m.channel] for m in self.spec.media])
        effect_share, spend_share = effect / effect.sum(), spend / spend.sum()
        distance = float(np.sqrt(np.sum((effect_share - spend_share) ** 2)))
        return distance * (1.0 + float(np.mean(np.round(effect_share, 4) == 0.0)))

    def channels(self) -> dict[str, response.Channel]:
        """Each channel as `chc.response` reads it, for the decisions."""
        return {
            m.channel: channel(
                m.adstock,
                m.hill,
                self.half_saturation[m.channel],
                self.coefficients[m.channel],
                self.length,
            )
            for m in self.spec.media
        }


def _window(
    panel: MediaPanel, window: tuple[np.datetime64, np.datetime64] | None
) -> npt.NDArray[np.bool_]:
    if window is None:
        return np.ones(panel.dates.size, dtype=bool)
    start, end = (np.datetime64(d, "D") for d in window)
    inside = (panel.dates >= start) & (panel.dates <= end)
    if start not in panel.dates or end not in panel.dates or start > end:
        msg = f"the window {start} to {end} must run forward between dates of the panel's calendar"
        raise ValueError(msg)
    return inside


@dataclass(frozen=True, slots=True, eq=False)
class _Data:
    """What a spec's fit reads of one geo, read once for every spec of a search: each channel's
    spend over the whole calendar, and the window's KPI and controls."""

    dates: npt.NDArray[np.datetime64]
    spend: _Array
    inside: npt.NDArray[np.bool_]
    y: _Array
    fitted: npt.NDArray[np.bool_]
    controls: _Array


def _read(
    panel: MediaPanel,
    geo: Geo,
    spec: RidgeSpec,
    window: tuple[np.datetime64, np.datetime64] | None,
) -> _Data:
    if geo not in panel.geos:
        msg = f"the panel has no geo {geo!r}; it has {list(panel.geos)}"
        raise ValueError(msg)
    g = panel.geos.index(geo)
    channels = {str(c): j for j, c in enumerate(panel.channels)}
    controls = [c.column for c in spec.controls]
    missing = [m.channel for m in spec.media if m.channel not in channels]
    missing += [c for c in controls if c not in panel.controls and c not in panel.events]
    if missing:
        msg = f"the panel has no channel or control named {missing}"
        raise ValueError(msg)
    inside = _window(panel, window)
    y = panel.kpi[g, inside]
    return _Data(
        dates=panel.dates,
        spend=panel.spend[g][:, [channels[m.channel] for m in spec.media]],
        inside=inside,
        y=y,
        fitted=np.isfinite(y),
        controls=panel.regressors(controls, geo, inside) - panel.counterfactuals(controls),
    )


def _design(data: _Data, spec: RidgeSpec) -> tuple[_Array, dict[str, float]]:
    """The window's design under `spec`, each channel's column saturated, and each channel's `K`."""
    columns, halves = [], {}
    for j, m in enumerate(spec.media):
        adstocked = carry_over(data.spend[:, j], m.adstock)[data.inside]
        halves[m.channel] = half_saturation(m.hill, adstocked)
        columns.append(saturate(adstocked, m.hill, halves[m.channel]))
    calendar = spec.calendar(data.y.size)
    return np.column_stack([*columns, data.controls, calendar]), halves


def _bounds(spec: RidgeSpec) -> tuple[_Array, _Array]:
    free = len(spec.terms()) - len(spec.media) - len(spec.controls)
    lower = np.array(
        [0.0] * len(spec.media) + [c.lower for c in spec.controls] + [-math.inf] * free
    )
    upper = np.array(
        [math.inf] * len(spec.media) + [c.upper for c in spec.controls] + [math.inf] * free
    )
    return lower, upper


def _freedom(x: _Array, penalty: float) -> float:
    """The ridge's effective degrees of freedom on `x`, `sum d^2 / (d^2 + n penalty)` over the
    singular values of its standardised columns that vary."""
    centred = x - x.mean(axis=0)
    scale = np.sqrt(np.mean(centred**2, axis=0))
    live = scale > 0.0
    d = np.linalg.svd(centred[:, live] / scale[live], compute_uv=False)
    return float(np.sum(d**2 / (d**2 + x.shape[0] * penalty)))


def _gaps(media: Media, half: float, tests: Sequence[LiftTest]) -> tuple[_Array, _Array]:
    """A channel's tests read through its transforms: in each readout period the gap in saturated
    spend, treated less control, and the observed gap, both at the market's scale."""
    saturated, observed = [], []
    for test in tests:
        treated, control = (
            saturate(carry_over(arm.spend / arm.share, media.adstock), media.hill, half)[
                arm.history :
            ]
            for arm in (test.treated, test.control)
        )
        saturated.append(treated - control)
        observed.append(test.difference)
    return np.concatenate(saturated), np.concatenate(observed)


@dataclass(frozen=True, slots=True, eq=False)
class _Pull:
    """What the tests hold under a spec, read off spend and the tests alone: each term's estimate,
    its saturated gaps' sum of squares, 0 for a term no test reads, the tests' noise variance, and
    each tested term's saturated and observed gaps."""

    center: _Array
    curvature: _Array
    noise: float
    reads: tuple[tuple[int, _Array, _Array], ...]

    def anchor(self, x: _Array, y: _Array, spec: RidgeSpec) -> tuple[Anchor, float]:
        """The tests' rows as an anchor on a fit to `x` and `y`, whose objective is the mean of
        their squares, and the weight of a test period against a KPI period: the KPI's noise
        variance on those periods over the tests'.

        Raises:
            ValueError: on periods the ridge fits with no degree of freedom left.
        """
        free = ridge_path(x, y, [spec.penalty], *_bounds(spec))
        residual = y - free.intercepts[0] - x @ free.coefficients[0]
        left = y.size - 1.0 - _freedom(x, spec.penalty)
        if not left > 0.0:
            msg = f"{y.size} fitted periods leave no degree of freedom for the KPI's noise"
            raise ValueError(msg)
        weight = float(residual @ residual) / left / self.noise
        return Anchor(self.center, weight * self.curvature / y.size), weight

    def misses(self, coefficients: _Array) -> float:
        """The tests' squared errors under the terms' `coefficients`."""
        return sum(
            float(np.sum((observed - coefficients[j] * gap) ** 2))
            for j, gap, observed in self.reads
        )


def _pull(
    spec: RidgeSpec,
    terms: int,
    halves: Mapping[str, float],
    tests: Mapping[str, Sequence[LiftTest]],
) -> _Pull:
    center, curvature = np.zeros(terms), np.zeros(terms)
    squares, periods, reads = 0.0, 0, []
    for j, m in enumerate(spec.media):
        if m.channel not in tests:
            continue
        gap, observed = _gaps(m, halves[m.channel], tests[m.channel])
        curvature[j] = float(gap @ gap)
        if not curvature[j] > 0.0:
            msg = (
                f"the tests of {m.channel!r} move no saturated spend between their groups under its"
                " adstock and Hill, so they read nothing of its coefficient"
            )
            raise ValueError(msg)
        center[j] = float(observed @ gap) / curvature[j]
        squares += float(np.sum((observed - center[j] * gap) ** 2))
        periods += observed.size
        reads.append((j, gap, observed))
    spare = periods - len(tests)
    if spare < 1:
        msg = (
            f"the tests' {periods} readout periods leave none past their {len(tests)} channels'"
            " estimates to read their noise from, which a test's weight needs"
        )
        raise ValueError(msg)
    if not squares > 0.0:
        msg = "the tests' gaps fit their estimates exactly, so they have no noise to weigh them by"
        raise ValueError(msg)
    return _Pull(center, curvature, squares / spare, tuple(reads))


def _check_tests(spec: RidgeSpec, tests: Mapping[str, Sequence[LiftTest]]) -> None:
    channels = {m.channel for m in spec.media}
    unknown = sorted(set(tests) - channels)
    empty = sorted(c for c, t in tests.items() if not t)
    if unknown or empty:
        msg = f"tests name channels the model does not have, {unknown}, or none, {empty}"
        raise ValueError(msg)


def _fit(
    data: _Data, spec: RidgeSpec, tests: Mapping[str, Sequence[LiftTest]] | None = None
) -> RidgeFit:
    design, halves = _design(data, spec)
    x, y = design[data.fitted], data.y[data.fitted]
    anchor, calibration = None, None
    if tests:
        pull = _pull(spec, design.shape[1], halves, tests)
        anchor, weight = pull.anchor(x, y, spec)
        estimates = {m.channel: float(pull.center[j]) for j, m in enumerate(spec.media)}
        calibration = Calibration(MappingProxyType({c: estimates[c] for c in tests}), weight)
    path = ridge_path(x, y, [spec.penalty], *_bounds(spec), anchor)
    terms = spec.terms()
    coefficients = path.coefficients[0]
    contributions = design * coefficients
    spend = data.spend[data.inside].sum(axis=0)
    return RidgeFit(
        spec=spec,
        dates=data.dates[data.inside],
        y=data.y,
        fitted=data.fitted,
        design=design,
        intercept=float(path.intercepts[0]),
        coefficients=MappingProxyType(dict(zip(terms, coefficients.tolist(), strict=True))),
        bound=MappingProxyType(dict(zip(terms, path.bound[0].tolist(), strict=True))),
        half_saturation=MappingProxyType(halves),
        contributions=MappingProxyType(dict(zip(terms, contributions.T, strict=True))),
        estimated=path.intercepts[0] + contributions.sum(axis=1),
        spend=MappingProxyType(
            {m.channel: float(total) for m, total in zip(spec.media, spend, strict=True)}
        ),
        length=data.dates.size,
        calibration=calibration,
    )


def fit_ridge(
    panel: MediaPanel,
    geo: Geo,
    spec: RidgeSpec,
    window: tuple[np.datetime64, np.datetime64] | None = None,
    tests: Mapping[str, Sequence[LiftTest]] | None = None,
) -> RidgeFit:
    """Fit `spec` to one geo of `panel` over `window`, its first and last dates, by default the
    whole calendar, calibrated by `tests`, each channel's geo tests, if given.

    Raises:
        ValueError: on a geo or column the panel does not have, a window off its calendar, a
            channel dark over the whole window, or fewer than two fitted periods; tests of a
            channel the model does not have, tests that move no saturated spend, or tests with no
            noise left to weigh them by.
    """
    if tests:
        _check_tests(spec, tests)
    return _fit(_read(panel, geo, spec, window), spec, tests)
