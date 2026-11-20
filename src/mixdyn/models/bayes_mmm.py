"""A static media-mix model in Bayes' form, its posterior drawn by the engine's own NUTS
(`mixdyn.inference`).

    y_t = intercept + calendar_t . delta + controls_t . gamma
              + sum_m beta_m s_m(adstock_m(spend_m))_t + e_t

over the observed periods of one geo's window, `e_t` normal or Student-t; or the KPI log-normal
about that mean, `y_t ~ LogNormal(log mu_t - sigma^2 / 2, sigma)`, so that `mu_t` is its mean and
every identity below holds in it. The model reads the KPI in units of its mean over the window's
observed periods, and each channel's spend in units of its mean positive spend over the panel's
calendar, so that one set of priors serves every panel:

* a channel's spend is carried over from the panel's first period by a normalised geometric
  kernel, `w_l = r^l / sum_{l < L} r^l`, `L` the longest series the model reads or the length
  `BayesSpec.kernel` states, and saturated,
  with `z` the adstock over the half-saturation point `K`, by a Hill, `z^n / (1 + z^n)`, or by
  `tanh(atanh(1/2) z)`, PyMC-Marketing's logistic saturation (`chc.response`);
* its coefficient is not sampled but set by its return on the window's spend, `roi`, on which its
  prior is stated, as Meridian states its own: the contribution over the window is `roi` times the
  spend over it, whatever the carryover and the curve; or, where its prior `counts` the carried
  return, the window's spend's whole return is, its carryover past the window's end counted with
  nothing spent after the panel, as `mixdyn.decisions.returns` counts it;
* the trend's columns are the ridge model's over the window's length, so that a coefficient is a
  change over the window; the season's are the ridge model's; a control is standardised over the
  window, and held to its sign.

The priors, each on the scales above:

    log roi_m ~ N(mu_m, sigma_m)       (0.2 and 0.9 by default, Meridian's for a revenue KPI)
    or roi_m ~ HalfNormal(s_m)          (`HalfNormalRoi`, opt-in: a return that may be 0)
    r_m ~ Beta(1, 3)                    (PyMC-Marketing's for a geometric decay)
    K_m ~ LogNormal(0, 1)               (half-saturation at the mean positive spend, by median)
    n_m ~ LogNormal(a, b)               (a Hill's slope, `HillCurve`'s; 0 and 0.5 by default)
    intercept ~ N(1, 1)    delta ~ N(0, 1)    gamma ~ N(0, 1), held to its sign
    sigma ~ HalfNormal(1)               nu ~ Gamma(2, 0.1) (Juárez and Steel 2010), Student-t's

`sigma` is the log's scale under log-normal noise, where a point whose mean is not above 0 at an
observed period has no density.

Geo tests (`chc.lift.LiftTest`) enter the likelihood. A test's gap between its groups, at the
market's scale, is its channel's coefficient times the gap in its saturated spend, each group's
spend divided by its share of the market and carried over and saturated as the channel's own, plus
noise of one scale across every test's readout periods, `tau ~ HalfNormal(1)`.

Lift estimates (`mixdyn.calibration.LiftEstimate`), tests as their analyses reported them, enter it
too. An estimate's lift at the market's scale is normal about its channel's coefficient times the
gap in saturated spend between the spend as the test ran and as planned, summed over the readout,
with the estimate's own standard error; a lift at or below 0 is read as any other.

Lift series (`mixdyn.calibration.LiftSeries`), tests read period by period, enter it as one
observation each: the gaps at the market's scale are jointly normal about the coefficient times the
gap in saturated spend at each period of the readout, with the series' own covariance. A summed lift
reads one combination of a channel's return and decay; the gaps' shape reads the decay too.

A control may name one of the panel's calendar events (`MediaPanel.events`), whose column is 1 in
the periods it falls in: its coefficient is the KPI the event adds there.

A control the panel declares a lever (`mixdyn.data.Lever`) enters as any control does, with no
adstock or curve; the decomposition measures its contribution against its counterfactual level
(`MediaPanel.counterfactual`), the coefficient times the lever's departure from that level, and
the intercept's part takes the rest. `lever_contributions` reads each lever's contribution over the
window at every draw.

A fit's forecast (`forecast_bayes`) is the model's posterior predictive past its window: each
draw's mean there, the trend run on from the window's last bend and the season on its period, the
controls at the values planned and each channel's spend carried over from the window's into the
plan's, and the KPI drawn about it by the model's noise.

The parameters are sampled on the real line, a bounded one through Stan's transforms: the logit of
a decay, the logarithm of a scale, a control held at or above 0 as the logarithm of its
coefficient, one at or below 0 as the logarithm of its negative. The intercept is sampled as the
level, the intercept plus the channels' contribution averaged over the window, `sum_m roi_m
spend_m / periods`, or a carried return's share of it that falls in the window: the KPI's mean
pins the level, where the intercept and the returns trade against each other along a ridge, and the
map's Jacobian is 1, so the posterior is the same. The model is written a second time, as a Stan
program, in tests/fixtures/oracles/bayes_mmm_stan.json, which samples the intercept itself, and the
two are checked against each other.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from chc import response
from jax.scipy import stats as jstats
from scipy import linalg

from mixdyn.data.panel import calendar
from mixdyn.inference import compiled
from mixdyn.inference.diagnostics import Diagnostics, diagnose, e_bfmi
from mixdyn.inference.pathfinder import pathfinder, resample
from mixdyn.models.ridge_mmm import Control, FourierSeason, LinearTrend, _window

if TYPE_CHECKING:
    from chc.lift import LiftTest

    from mixdyn.calibration import LiftEstimate, LiftSeries
    from mixdyn.data import Cadence, Geo, MediaPanel

type _Array = npt.NDArray[np.float64]

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RoiPrior:
    """A channel's return on its spend over the window, `LogNormal(mu, sigma)`, in the KPI's units
    per currency unit. What it `counts`: `"window"`, its contribution over the window's periods,
    Meridian's ROI; or `"carried"`, the incremental return of the window's spend wherever it falls,
    its carryover past the window's end included, `mixdyn.decisions.returns`' `roi`. Only the
    one-geo model reads a carried return.

    Raises:
        ValueError: on a `mu` not finite, a `sigma` not positive and finite, or a count neither
            `"window"` nor `"carried"`.
    """

    mu: float = 0.2
    sigma: float = 0.9
    counts: Literal["window", "carried"] = "window"

    def __post_init__(self) -> None:
        if not (math.isfinite(self.mu) and 0.0 < self.sigma < math.inf):
            msg = f"need a finite mu and a positive, finite sigma, got {self.mu}, {self.sigma}"
            raise ValueError(msg)
        if self.counts not in ("window", "carried"):
            msg = f'a return counts "window" or "carried", got {self.counts!r}'
            raise ValueError(msg)


_HALF_NORMAL_SCALE = math.exp(0.2) / statistics.NormalDist().inv_cdf(0.75)
"""`HalfNormalRoi`'s default scale, 1.81085: its median, `Phi^-1(3/4)` of it, is `RoiPrior()`'s."""


@dataclass(frozen=True, slots=True)
class HalfNormalRoi:
    """A channel's return on its spend over the window, `HalfNormal(scale)`, in the KPI's units
    per currency unit, counting what `RoiPrior.counts` names: a return that may be 0. Its density
    is greatest at 0, so where the data read no return its posterior reaches 0, which
    `RoiPrior`'s log-normal, whose density vanishes there, never does. The default scale puts its
    median at `RoiPrior()`'s, `exp(0.2)`, so choosing it moves the prior's mass towards 0 and its
    tail in, not its centre. PyMC-Marketing's priors on a channel's coefficient are of this
    family. The model over geos does not read it.

    Raises:
        ValueError: on a `scale` not positive and finite, or a count neither `"window"` nor
            `"carried"`.
    """

    scale: float = _HALF_NORMAL_SCALE
    counts: Literal["window", "carried"] = "window"

    def __post_init__(self) -> None:
        if not 0.0 < self.scale < math.inf:
            msg = f"need a positive, finite scale, got {self.scale}"
            raise ValueError(msg)
        if self.counts not in ("window", "carried"):
            msg = f'a return counts "window" or "carried", got {self.counts!r}'
            raise ValueError(msg)


type ReturnPrior = RoiPrior | HalfNormalRoi
"""The priors a channel's return may have: log-normal, the default, or half-normal."""


@dataclass(frozen=True, slots=True)
class BayesMedia:
    """A channel of the panel and the prior on its return."""

    channel: str
    roi: ReturnPrior = field(default_factory=RoiPrior)


@dataclass(frozen=True, slots=True)
class Normal:
    """Normal noise."""


@dataclass(frozen=True, slots=True)
class StudentT:
    """Student-t noise, its degrees of freedom sampled with the rest."""


@dataclass(frozen=True, slots=True)
class LogNormal:
    """Log-normal noise about the mean, parametrised by it: the KPI's scale proportional to its
    level, the KPI above 0."""


type Likelihood = Normal | StudentT | LogNormal


@dataclass(frozen=True, slots=True)
class HillCurve:
    """Each channel saturated by a Hill, its slope sampled: Meridian's and Robyn's family,
    S-shaped where the slope passes 1. The slope's prior is `LogNormal(slope_location,
    slope_scale)`, its median `exp(slope_location)`.

    A slope above 1 is a curve whose margin exceeds its chord at low spend. Where lift tests read
    each channel's chord from no spend to the spend tested, the slope alone turns that chord into
    the margin a plan reads, so a prior with a location below 0 leans the curve concave unless
    the data say otherwise (ADR 0082).

    Raises:
        ValueError: on a location that is not finite or a scale not above 0 and finite.
    """

    slope_location: float = 0.0
    slope_scale: float = 0.5

    def __post_init__(self) -> None:
        if not (math.isfinite(self.slope_location) and 0.0 < self.slope_scale < math.inf):
            msg = (
                "a Hill's slope prior has a finite location and a finite scale above 0, not "
                f"{self.slope_location!r} and {self.slope_scale!r}"
            )
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class TanhCurve:
    """Each channel saturated by `tanh`: PyMC-Marketing's logistic saturation, concave from no
    spend."""


type Saturation = HillCurve | TanhCurve


@dataclass(frozen=True, slots=True)
class UniformStart:
    """Each chain starts where Stan's would, uniform in `[-2, 2]` on the sampling scale; under
    log-normal noise, where some period's mean would not be above 0, with the intercept raised
    until the least is the least KPI observed."""


@dataclass(frozen=True, slots=True)
class PathfinderStart:
    """Each chain starts at a draw of `_PATHS` Pathfinder paths (`mixdyn.inference.pathfinder`),
    each from a start uniform in `[-2, 2]` on the sampling scale, a path that cannot be
    approximated dropped, and the rest's draws resampled together,
    as Stan's multi-path Pathfinder initialises its chains. `fit_bayes`' default."""


type Start = UniformStart | PathfinderStart


@dataclass(frozen=True, slots=True)
class NaturalCoordinates:
    """The sampler moves on `BayesModel.coordinates` themselves. `fit_bayes`' default."""


@dataclass(frozen=True, slots=True)
class QRCoordinates:
    """The sampler moves the linear block, the level, the calendar's coefficients and the free
    controls', on the thin QR decomposition of their columns over the observed periods, `X = Q R`,
    Stan's QR reparameterisation: it samples `R* b`, `R* = R / sqrt(n - 1)`, where `b` is the
    block's coefficients. The map is linear, so the posterior is the same and its Jacobian a
    constant; the priors still read `b`. A control held to its sign keeps its own coordinate."""


type Coordinates = NaturalCoordinates | QRCoordinates

_UNIFORM_START = UniformStart()
_PATHFINDER_START = PathfinderStart()
_NATURAL = NaturalCoordinates()
_PATHS = 4

_TANH_HALF = math.atanh(0.5)
"""Where `tanh` reaches half its ceiling."""


@dataclass(frozen=True, slots=True)
class BayesSpec:
    """The model: its channels, its controls, its trend and season if any, its noise, the
    family every channel's curve is of, and its kernels' length in periods, `kernel`: by default
    the longest series the model reads, the panel's calendar or a test's, and otherwise the
    periods stated, the weights normalised over them and a spend's carryover cut after them
    (ADR 0082).

    Raises:
        ValueError: on a column named twice, a model with no channel, a control held to a bound
            other than its sign, or a kernel's length that is not a whole number of periods, at
            least 1.
    """

    media: tuple[BayesMedia, ...]
    controls: tuple[Control, ...] = ()
    trend: LinearTrend | None = None
    season: FourierSeason | None = None
    likelihood: Likelihood = field(default_factory=Normal)
    saturation: Saturation = field(default_factory=HillCurve)
    kernel: int | None = None

    def __post_init__(self) -> None:
        if self.kernel is not None and (
            isinstance(self.kernel, bool) or not isinstance(self.kernel, int) or self.kernel < 1
        ):
            msg = f"a kernel's length is a whole number of periods, at least 1, not {self.kernel!r}"
            raise ValueError(msg)
        columns = [*(m.channel for m in self.media), *(c.column for c in self.controls)]
        twice = sorted({c for c in columns if columns.count(c) > 1})
        if twice:
            msg = f"the model names {twice} more than once"
            raise ValueError(msg)
        if not self.media:
            msg = "a media-mix model needs at least one channel"
            raise ValueError(msg)
        unsigned = [
            c.column
            for c in self.controls
            if c.lower not in (-math.inf, 0.0)
            or c.upper not in (0.0, math.inf)
            or c.lower == c.upper
        ]
        if unsigned:
            msg = f"controls {unsigned} are held to bounds other than a sign"
            raise ValueError(msg)

    def calendar_terms(self) -> tuple[str, ...]:
        """The trend's and the season's columns' names."""
        return (
            *(self.trend.terms() if self.trend else ()),
            *(self.season.terms() if self.season else ()),
        )

    def calendar(self, periods: int, *, ahead: int = 0) -> _Array:
        """The trend's columns over the window's length and the season's, over `periods`, then
        over the `ahead` periods after them."""
        blocks = [np.empty((periods + ahead, 0))]
        if self.trend:
            blocks.append(self.trend.columns(periods, ahead=ahead) / periods)
        if self.season:
            blocks.append(self.season.columns(periods + ahead))
        return np.column_stack(blocks)


@dataclass(frozen=True, slots=True, eq=False)
class BayesFit:
    """The posterior of the model fitted to one geo over a window of the panel's calendar.

    Arrays run over the window's periods; `fitted` marks those the likelihood read. `draws` holds
    each parameter's draws in the panel's units, `(chains, draws)` leading: `roi`, `retention`,
    `half_saturation`, a Hill's `slope`, and `coefficient` of each channel, its contribution per
    unit of saturated spend; `intercept`; `calendar` and `control`, its coefficient per unit of the
    column, where the model has them; `sigma`, the log's scale under log-normal noise; and `nu` and
    `test_sigma` where the model has them.
    `log_likelihood` is each draw's log density of each observation in the panel's units,
    `(chains, draws, observations)`: the KPI at each fitted period, then each test's gap at its
    readout periods, then each lift estimate's lift, then each lift series' gaps jointly, the tests,
    the estimates and the series channel by channel in the model's order and each channel's in the
    order given. `contributions` and `estimated` are
    posterior means, a control's contribution its coefficient times its column, the intercept's the
    rest of `estimated`. `diagnostics` reads each parameter's draws; `divergent` counts each
    chain's divergent transitions, and `e_bfmi` is each chain's. `length` is the kernels' length,
    which `channels` hands on. `spend` is each channel's spend from the panel's first period to the
    window's last, which a forecast carries over into the plan's, and `cadence` the panel's.
    """

    spec: BayesSpec
    dates: npt.NDArray[np.datetime64]
    y: _Array
    fitted: npt.NDArray[np.bool_]
    draws: Mapping[str, _Array]
    log_likelihood: _Array
    contributions: Mapping[str, _Array]
    estimated: _Array
    diagnostics: Mapping[str, Diagnostics]
    divergent: npt.NDArray[np.int64]
    e_bfmi: _Array
    length: int
    spend: Mapping[str, _Array]
    cadence: Cadence

    def channels(self) -> dict[str, response.Channel]:
        """Each channel as `chc.response` reads it at its parameters' posterior medians."""
        return self._channels(lambda draws: float(np.median(draws)))

    def channel_draws(self, count: int) -> list[dict[str, response.Channel]]:
        """Each channel as `chc.response` reads it at `count` of the posterior's draws, evenly
        spaced over the chains' draws in order from the first to the last: a return's posterior is
        `mixdyn.decisions.returns` read at each.

        Raises:
            ValueError: on a count below 1 or above the fit's draws.
        """
        keep = _thinned(self.draws["roi"].shape, count)
        return [self._channels(lambda draws, i=i: float(draws.reshape(-1)[i])) for i in keep]

    def _channels(self, read: Callable[[_Array], float]) -> dict[str, response.Channel]:
        """Each channel at the value `read` takes of each of its parameters' `(chains, draws)`."""
        out = {}
        for j, m in enumerate(self.spec.media):
            retention, half, coefficient = (
                read(self.draws[k][..., j]) for k in ("retention", "half_saturation", "coefficient")
            )
            tanh = isinstance(self.spec.saturation, TanhCurve)
            slope = 1.0 if tanh else read(self.draws["slope"][..., j])
            out[m.channel] = self._channel(retention, half, slope, coefficient)
        return out

    def _channel(
        self,
        retention: float | jax.Array,
        half: float | jax.Array,
        slope: float | jax.Array,
        coefficient: float | jax.Array,
    ) -> response.Channel:
        """A channel at one value of each of its parameters in the panel's units; a `tanh` reads
        no slope."""
        kernel = response.GeometricAdstock(retention, length=self.length, normalized=True)
        if isinstance(self.spec.saturation, TanhCurve):
            curve = response.Tanh(half / _TANH_HALF)
        else:
            curve = response.Hill(half, slope)
        return response.Channel(kernel, curve, coefficient)


def _thinned(shape: tuple[int, ...], count: int) -> npt.NDArray[np.intp]:
    """`count` indices into a posterior's `(chains, draws)` flattened in chain order, evenly spaced
    from the first draw to the last."""
    total = shape[0] * shape[1]
    if not 1 <= count <= total:
        msg = f"count must be from 1 to the fit's {total} draws, got {count}"
        raise ValueError(msg)
    return np.linspace(0, total - 1, count).round().astype(np.intp)


@dataclass(frozen=True, slots=True, eq=False)
class _Arms:
    """One channel's tests at each of their readout periods, in test order: each group's spend at
    the market's scale over the model's spend unit, at the period and the `L - 1` before it, and
    the observed gap over the KPI's unit."""

    channel: int
    treated: _Array  # (readout periods, L)
    control: _Array
    gap: _Array


@dataclass(frozen=True, slots=True, eq=False)
class _Estimates:
    """One channel's lift estimates: the tested group's spend at the market's scale over the
    model's spend unit, as the test ran it and as planned, at each estimate's readout periods and
    the `L - 1` before each, in estimate order; the estimate each row reads; and each estimate's
    lift and standard error at the market's scale over the KPI's unit."""

    channel: int
    spend: _Array  # (readout periods, L)
    planned: _Array
    estimate: npt.NDArray[np.intp]  # (readout periods,)
    lift: _Array  # (estimates,)
    se: _Array


@dataclass(frozen=True, slots=True, eq=False)
class _Series:
    """One lift series: its channel; the tested group's spend at the market's scale over the
    model's spend unit, as the test ran it and as planned, at each readout period and the `L - 1`
    before it; its gaps at the market's scale over the KPI's unit; `whiten`, the inverse of the
    Cholesky factor of their covariance on that scale; and `constant`, the log of the normal
    density's constant there."""

    channel: int
    spend: _Array  # (readout, L)
    planned: _Array
    gaps: _Array  # (readout,)
    whiten: _Array  # (readout, readout)
    constant: float


@dataclass(frozen=True, slots=True, eq=False)
class _Tail:
    """One channel whose prior counts the carried return: its spend over the model's spend unit at
    each period from the window's first to `L - 1` past its last, with no spend past the panel,
    and the same with the window's spend removed, each at the period and the `L - 1` before it."""

    channel: int
    spend: _Array  # (window + L - 1, L)
    removed: _Array


@dataclass(frozen=True, slots=True, eq=False)
class BayesModel:
    """The model's posterior density on its sampling coordinates, with the data it reads.

    `log_density` takes a point of the `dimension` coordinates, named by `coordinates` in their
    order: each channel's log return, logit decay and log half-saturation point, each channel's
    log slope if its curve is a Hill, the level, the calendar's coefficients, the controls'
    (the log of a coefficient's size, if held to a sign), the log noise scale, and the log degrees
    of freedom and log tests' noise scale where the model has them. It returns the log density, up
    to a constant, with every transform's Jacobian, and its gradient. `parameters` maps points to
    the panel's units, as `BayesFit.draws` holds them, and `log_likelihood` to each observation's
    log density in the panel's units, as `BayesFit.log_likelihood` holds it.

    The data are on the model's scales, `kpi_unit` the KPI's and `spend_unit` each channel's:
    `lagged` each channel's spend over its unit at each of the window's periods and the `L - 1`
    before it, lag 0 first and zero before the panel's first period, `(M, window, L)`; `y` the KPI
    over its unit at the window's `observed` periods; `calendar` and `controls` the window's
    columns, the controls standardised by `control_mean` and `control_sd`; `window_spend` each
    channel's spend over the window, over the KPI's unit; `arms` the geo tests and `estimates` the
    lift estimates, channel by channel; `series` the lift series, channel by channel; `tails` the
    channels whose prior counts the carried return; `control_baseline`, where the panel declares
    some controls levers, each control's counterfactual level in its own units, 0 for one that is
    not a lever, which the decomposition measures its contribution against.
    """

    spec: BayesSpec
    lagged: _Array
    y: _Array
    observed: npt.NDArray[np.bool_]
    calendar: _Array
    controls: _Array
    window_spend: _Array
    arms: tuple[_Arms, ...]
    estimates: tuple[_Estimates, ...]
    kpi_unit: float
    spend_unit: _Array
    control_mean: _Array
    control_sd: _Array
    series: tuple[_Series, ...] = ()
    tails: tuple[_Tail, ...] = ()
    control_baseline: _Array | None = None
    _compiled: Callable[[jax.Array], tuple[jax.Array, jax.Array]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_compiled", jax.jit(jax.value_and_grad(self._log_density)))

    @property
    def _sizes(self) -> dict[str, int]:
        m = len(self.spec.media)
        return {
            "log_roi": m,
            "retention": m,
            "half": m,
            "slope": m if isinstance(self.spec.saturation, HillCurve) else 0,
            "level": 1,
            "calendar": self.calendar.shape[1],
            "control": self.controls.shape[1],
            "sigma": 1,
            "nu": int(isinstance(self.spec.likelihood, StudentT)),
            "test_sigma": int(bool(self.arms)),
        }

    @property
    def dimension(self) -> int:
        return sum(self._sizes.values())

    @property
    def coordinates(self) -> tuple[str, ...]:
        channels = [m.channel for m in self.spec.media]
        return (
            *(f"log_roi[{c}]" for c in channels),
            *(f"logit_retention[{c}]" for c in channels),
            *(f"log_half_saturation[{c}]" for c in channels),
            *(f"log_slope[{c}]" for c in channels if self._sizes["slope"]),
            "level",
            *(f"calendar[{t}]" for t in self.spec.calendar_terms()),
            *(f"control[{c.column}]" for c in self.spec.controls),
            "log_sigma",
            *(("log_nu",) if self._sizes["nu"] else ()),
            *(("log_test_sigma",) if self._sizes["test_sigma"] else ()),
        )

    def log_density(self, theta: _Array) -> tuple[float, _Array]:
        value, gradient = self._compiled(jnp.asarray(theta, dtype=float))
        return float(value), np.asarray(gradient)

    def qr_map(self) -> _Array:
        """The square matrix `A` that takes a point on `QRCoordinates`' sampling coordinates to
        one on `coordinates`, `theta = A u`: the identity but on the linear block, where it is
        `R*^-1`."""
        free = np.array(
            [c.lower == -math.inf and c.upper == math.inf for c in self.spec.controls], dtype=bool
        )
        names = self.coordinates
        level = names.index("level")
        block = [level, *range(level + 1, level + 1 + self._sizes["calendar"])]
        first = level + 1 + self._sizes["calendar"]
        block += [first + int(j) for j in np.flatnonzero(free)]
        columns = np.column_stack(
            [
                np.ones(int(self.observed.sum())),
                self.calendar[self.observed],
                self.controls[self.observed][:, free],
            ]
        )
        _, r = np.linalg.qr(columns)
        r = r * np.where(np.diag(r) < 0.0, -1.0, 1.0)[:, None]
        out = np.eye(self.dimension)
        out[np.ix_(block, block)] = np.linalg.inv(r / math.sqrt(columns.shape[0] - 1))
        return out

    def parameters(self, theta: _Array) -> dict[str, _Array]:
        """The parameters at each point of `theta`, `(points, dimension)`, in the panel's units."""
        raw, _, _ = _summary(self, np.atleast_2d(theta))
        return raw

    def log_likelihood(self, theta: _Array) -> _Array:
        """Each observation's log density in the panel's units at each point of `theta`, `(points,
        dimension)`: the KPI at each observed period, then each test's gap at its readout periods,
        then each lift estimate's lift, then each lift series' gaps jointly, in
        `BayesFit.log_likelihood`'s order."""
        _, _, pointwise = _summary(self, np.atleast_2d(theta))
        return pointwise

    def _constrained(self, theta: jax.Array) -> tuple[dict[str, jax.Array], jax.Array]:
        """Each block of `theta` on its own scale, and the log Jacobian of the transforms."""
        u, start = {}, 0
        for name, size in self._sizes.items():
            u[name] = theta[start : start + size]
            start += size
        signs = np.array(
            [
                0.0 if c.lower == -math.inf and c.upper == math.inf else 1.0
                for c in self.spec.controls
            ]
        )
        negative = np.array([c.upper == 0.0 for c in self.spec.controls], dtype=bool)
        signed = jnp.where(negative, -1.0, 1.0) * jnp.exp(jnp.where(signs > 0.0, u["control"], 0.0))
        out = {
            "log_roi": u["log_roi"],
            "retention": jax.nn.sigmoid(u["retention"]),
            "half": jnp.exp(u["half"]),
            "slope": jnp.exp(u["slope"]),
            "calendar": u["calendar"],
            "control": jnp.where(signs > 0.0, signed, u["control"]),
            "sigma": jnp.exp(u["sigma"]),
            "nu": jnp.exp(u["nu"]),
            "test_sigma": jnp.exp(u["test_sigma"]),
        }
        out["intercept"] = u["level"] - self._media_level(out)
        log_jacobian = (
            jnp.sum(jax.nn.log_sigmoid(u["retention"]) + jax.nn.log_sigmoid(-u["retention"]))
            + jnp.sum(u["half"])
            + jnp.sum(u["slope"])
            + jnp.sum(signs * u["control"])
            + jnp.sum(u["sigma"])
            + jnp.sum(u["nu"])
            + jnp.sum(u["test_sigma"])
        )
        return out, log_jacobian

    def _media_level(self, p: Mapping[str, jax.Array]) -> jax.Array:
        """The channels' contribution averaged over the window's periods: each one's return times
        its spend over the window, whatever its carryover and curve; where a return is carried,
        the share of it that falls in the window, which its decay and curve set."""
        if not self.tails:
            return (
                jnp.sum(jnp.exp(p["log_roi"]) * self.window_spend, axis=-1) / self.lagged.shape[1]
            )
        saturated, beta = self._terms(p)
        return jnp.sum(saturated @ beta) / self.lagged.shape[1]

    def _with_level(self, theta: jax.Array) -> jax.Array:
        """Points `theta`, `(..., dimension)`, whose `level` coordinate holds the intercept, as
        Stan's program samples it, with the level there instead."""
        level = self.coordinates.index("level")
        if not self.tails:
            roi = theta[..., : len(self.spec.media)]
            return theta.at[..., level].add(self._media_level({"log_roi": roi}))

        def one(t: jax.Array) -> jax.Array:
            p, _ = self._constrained(t)
            return t.at[level].add(self._media_level(p))

        flat = jnp.reshape(theta, (-1, theta.shape[-1]))
        return jnp.reshape(jax.vmap(one)(flat), theta.shape)

    def _saturated(self, p: Mapping[str, jax.Array], j: int, lagged: _Array) -> jax.Array:
        """Channel `j`'s spend at each row of `lagged`, `(rows, L)`, carried over and saturated."""
        kernel = response.GeometricAdstock(
            p["retention"][j], length=lagged.shape[1], normalized=True
        )
        if isinstance(self.spec.saturation, TanhCurve):
            curve = response.Tanh(p["half"][j] / _TANH_HALF)
        else:
            curve = response.Hill(p["half"][j], p["slope"][j])
        return curve(lagged @ kernel.weights())

    def _terms(self, p: Mapping[str, jax.Array]) -> tuple[jax.Array, jax.Array]:
        """The window's saturated spend, `(window, M)`, and each channel's coefficient: its return
        times the window's spend over the saturated spend that earns it, the window's or, where
        the return is carried, the window's spend's whole increment."""
        saturated = jnp.stack(
            [self._saturated(p, j, self.lagged[j]) for j in range(len(self.spec.media))], axis=1
        )
        earning = saturated.sum(axis=0)
        for t in self.tails:
            increment = self._saturated(p, t.channel, t.spend) - self._saturated(
                p, t.channel, t.removed
            )
            earning = earning.at[t.channel].set(increment.sum())
        beta = jnp.exp(p["log_roi"]) * self.window_spend / earning
        return saturated, beta

    def _log_density(self, theta: jax.Array) -> jax.Array:
        p, log_jacobian = self._constrained(theta)
        lp = log_jacobian
        lp += _log_roi_prior(self.spec.media, p["log_roi"])
        lp += jnp.sum(jstats.beta.logpdf(p["retention"], 1.0, 3.0))
        lp += jnp.sum(_log_normal(p["half"], 0.0, 1.0))
        if isinstance(self.spec.saturation, HillCurve):
            curve = self.spec.saturation
            lp += jnp.sum(_log_normal(p["slope"], curve.slope_location, curve.slope_scale))
        lp += jnp.sum(jstats.norm.logpdf(p["intercept"], 1.0, 1.0))
        lp += jnp.sum(jstats.norm.logpdf(p["calendar"]))
        lp += jnp.sum(jstats.norm.logpdf(p["control"]))
        lp += jnp.sum(jstats.norm.logpdf(p["sigma"]))
        lp += jnp.sum(jstats.gamma.logpdf(p["nu"], 2.0, scale=10.0))
        lp += jnp.sum(jstats.norm.logpdf(p["test_sigma"]))
        return lp + jnp.sum(self._pointwise(p, *self._terms(p)))

    def _pointwise(
        self, p: Mapping[str, jax.Array], saturated: jax.Array, beta: jax.Array
    ) -> jax.Array:
        """Each observation's log density on the model's scales, the KPI's, the tests', the lift
        estimates' and then the lift series'."""
        mean = self._mean(p, saturated, beta)
        if isinstance(self.spec.likelihood, StudentT):
            kpi = jstats.t.logpdf(self.y, p["nu"][0], mean, p["sigma"][0])
        elif isinstance(self.spec.likelihood, LogNormal):
            positive = mean > 0.0
            location = jnp.log(jnp.where(positive, mean, 1.0)) - p["sigma"][0] ** 2 / 2.0
            kpi = jnp.where(positive, _log_normal(self.y, location, p["sigma"][0]), -jnp.inf)
        else:
            kpi = jstats.norm.logpdf(self.y, mean, p["sigma"][0])
        tests = [
            jstats.norm.logpdf(
                arms.gap, beta[arms.channel] * self._tested(p, arms), p["test_sigma"][0]
            )
            for arms in self.arms
        ]
        lifts = [
            jstats.norm.logpdf(e.lift, beta[e.channel] * self._lifted(p, e), e.se)
            for e in self.estimates
        ]
        series = jnp.array([self._gapped(p, beta, s) for s in self.series]).reshape(-1)
        return jnp.concatenate([kpi, *tests, *lifts, series])

    @property
    def _units(self) -> _Array:
        """Each observation's count of the KPI's units, in `log_likelihood`'s order: a series'
        density is one over the KPI's unit to the power of its readout."""
        single = self.y.size + sum(a.gap.size for a in self.arms)
        single += sum(e.lift.size for e in self.estimates)
        return np.concatenate([np.ones(single), [float(s.gaps.size) for s in self.series]])

    def _mean(self, p: Mapping[str, jax.Array], saturated: jax.Array, beta: jax.Array) -> jax.Array:
        """The KPI's mean at each observed period on the model's scale."""
        return (
            p["intercept"][0]
            + self.calendar @ p["calendar"]
            + self.controls @ p["control"]
            + saturated @ beta
        )[self.observed]

    def _lift(self, theta: _Array) -> _Array:
        """`theta` moved into the support: under log-normal noise, where some period's mean is not
        above 0, the level raised until the least is the least KPI observed, which moves every
        period's mean alike; otherwise `theta` itself."""
        if not isinstance(self.spec.likelihood, LogNormal):
            return theta
        p, _ = self._constrained(jnp.asarray(theta))
        least = float(jnp.min(self._mean(p, *self._terms(p))))
        if least > 0.0:
            return theta
        lifted = np.array(theta, dtype=float)
        lifted[self.coordinates.index("level")] += float(self.y.min()) - least
        return lifted

    def _tested(self, p: Mapping[str, jax.Array], arms: _Arms) -> jax.Array:
        """The gap in saturated spend over each readout period of a channel's tests."""
        return self._saturated(p, arms.channel, arms.treated) - self._saturated(
            p, arms.channel, arms.control
        )

    def _gapped(self, p: Mapping[str, jax.Array], beta: jax.Array, s: _Series) -> jax.Array:
        """A lift series' log density: its gaps jointly normal about the coefficient times the gap
        in saturated spend at each readout period, under its covariance."""
        gap = self._saturated(p, s.channel, s.spend) - self._saturated(p, s.channel, s.planned)
        return s.constant - 0.5 * jnp.sum((s.whiten @ (s.gaps - beta[s.channel] * gap)) ** 2)

    def _lifted(self, p: Mapping[str, jax.Array], e: _Estimates) -> jax.Array:
        """The gap in saturated spend over each of a channel's estimates' readouts."""
        gap = self._saturated(p, e.channel, e.spend) - self._saturated(p, e.channel, e.planned)
        return jax.ops.segment_sum(gap, e.estimate, e.lift.size, indices_are_sorted=True)


def _log_roi_prior(media: Sequence[BayesMedia], log_roi: jax.Array) -> jax.Array:
    """The return priors' log density at the channels' log returns, the Jacobian of `exp` counted
    where a prior is on the return itself."""
    priors = [m.roi for m in media]
    log = np.array([isinstance(r, RoiPrior) for r in priors])
    mu = np.array([r.mu for r in priors if isinstance(r, RoiPrior)])
    sd = np.array([r.sigma for r in priors if isinstance(r, RoiPrior)])
    if log.all():
        return jnp.sum(jstats.norm.logpdf(log_roi, mu, sd))
    scale = np.array([r.scale for r in priors if isinstance(r, HalfNormalRoi)])
    x = log_roi[~log]
    on_return = jstats.norm.logpdf(jnp.exp(x), 0.0, scale) + math.log(2.0) + x
    return jnp.sum(jstats.norm.logpdf(log_roi[log], mu, sd)) + jnp.sum(on_return)


def _log_normal(
    x: jax.Array | _Array, mu: float | jax.Array, sigma: float | jax.Array
) -> jax.Array:
    return jstats.norm.logpdf(jnp.log(x), mu, sigma) - jnp.log(x)


def _lags(series: _Array, periods: npt.NDArray[np.bool_]) -> _Array:
    """`series` at each of `periods` and every period before it, lag 0 first and zero before the
    series, `(periods, series.size)`: a kernel's weights carry the series over as one product."""
    lag = np.flatnonzero(periods)[:, None] - np.arange(series.size)
    return np.where(lag >= 0, series[np.maximum(lag, 0)], 0.0)


def _arms(
    channel: int,
    tests: Sequence[LiftTest],
    length: int,
    spend_unit: float,
    kpi_unit: float,
) -> _Arms:
    treated, control, gap = [], [], []
    for test in tests:
        readout = np.zeros(length, dtype=bool)
        readout[length - test.treated.outcome.size :] = True
        for arm, rows in ((test.treated, treated), (test.control, control)):
            series = arm.spend / arm.share / spend_unit
            rows.append(_lags(np.concatenate([np.zeros(length - series.size), series]), readout))
        gap.append(test.difference / kpi_unit)
    return _Arms(channel, np.concatenate(treated), np.concatenate(control), np.concatenate(gap))


def _estimates(
    channel: int,
    estimates: Sequence[LiftEstimate],
    length: int,
    spend_unit: float,
    kpi_unit: float,
) -> _Estimates:
    spend, planned = [], []
    for e in estimates:
        readout = np.zeros(length, dtype=bool)
        readout[length - e.readout :] = True
        for series, rows in ((e.spend, spend), (e.planned, planned)):
            scaled = series / e.share / spend_unit
            rows.append(_lags(np.concatenate([np.zeros(length - scaled.size), scaled]), readout))
    return _Estimates(
        channel,
        np.concatenate(spend),
        np.concatenate(planned),
        np.repeat(np.arange(len(estimates)), [e.readout for e in estimates]),
        np.array([e.lift / e.share for e in estimates]) / kpi_unit,
        np.array([e.se / e.share for e in estimates]) / kpi_unit,
    )


def _series(
    channel: int,
    series: LiftSeries,
    length: int,
    spend_unit: float,
    kpi_unit: float,
) -> _Series:
    readout = np.zeros(length, dtype=bool)
    readout[length - series.readout :] = True
    spend, planned = (
        _lags(np.concatenate([np.zeros(length - x.size), x / series.share / spend_unit]), readout)
        for x in (series.spend, series.planned)
    )
    scale = series.share * kpi_unit
    factor = np.linalg.cholesky(series.covariance / scale**2)
    return _Series(
        channel,
        spend,
        planned,
        series.gaps / scale,
        linalg.solve_triangular(factor, np.eye(series.readout), lower=True),
        -float(np.log(np.diag(factor)).sum()) - 0.5 * series.readout * math.log(2.0 * math.pi),
    )


def bayes_model(
    panel: MediaPanel,
    geo: Geo,
    spec: BayesSpec,
    *,
    window: tuple[np.datetime64, np.datetime64] | None = None,
    tests: Mapping[str, Sequence[LiftTest]] | None = None,
    estimates: Mapping[str, Sequence[LiftEstimate]] | None = None,
    series: Mapping[str, Sequence[LiftSeries]] | None = None,
) -> BayesModel:
    """The model of `spec` on one geo of `panel` over `window`, its first and last dates, by
    default the whole calendar, its likelihood holding `tests`, each channel's geo tests,
    `estimates`, each channel's lift estimates, and `series`, each channel's lift series, if
    given.

    Raises:
        ValueError: on JAX in single precision; a geo or column the panel does not have, a window
            off its calendar, a channel that spends nothing over it, fewer than two observed
            periods or a mean KPI not above 0 over them, a KPI not above 0 at one of them under
            log-normal noise, a control that does not vary over it; tests, estimates or series of
            a channel the model does not have, or of none; and estimates or series in a geo other
            than `geo`.
    """
    return _model(panel, geo, spec, window, tests or {}, estimates or {}, 0, series or {})


def _model(
    panel: MediaPanel,
    geo: Geo,
    spec: BayesSpec,
    window: tuple[np.datetime64, np.datetime64] | None,
    tests: Mapping[str, Sequence[LiftTest]],
    estimates: Mapping[str, Sequence[LiftEstimate]],
    reads: int,
    series: Mapping[str, Sequence[LiftSeries]] | None = None,
) -> BayesModel:
    """`bayes_model`'s model, its kernels at least `reads` periods long: a model it is part of may
    read a longer series through them."""
    if not jax.config.x64_enabled:
        msg = "the model needs JAX in double precision (JAX_ENABLE_X64=1)"
        raise ValueError(msg)
    if geo not in panel.geos:
        msg = f"the panel has no geo {geo!r}; it has {list(panel.geos)}"
        raise ValueError(msg)
    g = panel.geos.index(geo)
    channels = {str(c): j for j, c in enumerate(panel.channels)}
    columns = [c.column for c in spec.controls]
    names = [m.channel for m in spec.media]
    missing = [c for c in names if c not in channels]
    missing += [c for c in columns if c not in panel.controls and c not in panel.events]
    if missing:
        msg = f"the panel has no channel or control named {missing}"
        raise ValueError(msg)
    series = series or {}
    for kind, given in (("tests", tests), ("estimates", estimates), ("series", series)):
        unknown = sorted(set(given) - set(names))
        empty = sorted(c for c, t in given.items() if not t)
        if unknown or empty:
            msg = f"{kind} name channels the model does not have, {unknown}, or none, {empty}"
            raise ValueError(msg)
    for kind, given in (("estimates", estimates), ("series", series)):
        elsewhere = sorted({g for es in given.values() for e in es for g in e.geos or ()} - {geo})
        if elsewhere:
            msg = f"{kind} name geos {elsewhere} that the model of {geo!r} does not read"
            raise ValueError(msg)
    inside = _window(panel, window)
    spend = panel.spend[g][:, [channels[c] for c in names]]
    window_spend = spend[inside].sum(axis=0)
    dark = [c for c, total in zip(names, window_spend, strict=True) if not total > 0.0]
    if dark:
        msg = f"channels {dark} spend nothing over the window"
        raise ValueError(msg)
    kpi = panel.kpi[g, inside]
    observed = np.isfinite(kpi)
    kpi_unit = float(np.mean(kpi[observed])) if observed.sum() >= 2 else math.nan
    if not kpi_unit > 0.0:
        msg = f"the window has {observed.sum()} observed periods of mean KPI {kpi_unit}"
        raise ValueError(msg)
    if isinstance(spec.likelihood, LogNormal) and not np.all(kpi[observed] > 0.0):
        at = np.flatnonzero(observed)[kpi[observed] <= 0.0].tolist()
        msg = f"log-normal noise needs a KPI above 0, and the window's periods {at} are not"
        raise ValueError(msg)
    spend_unit = np.array([float(np.mean(s[s > 0.0])) for s in spend.T])
    controls = panel.regressors(columns, geo, inside)
    control_mean, control_sd = controls.mean(axis=0), controls.std(axis=0)
    flat = [c.column for c, sd in zip(spec.controls, control_sd, strict=True) if not sd > 0.0]
    if flat:
        msg = f"controls {flat} do not vary over the window"
        raise ValueError(msg)
    longest = max(
        [spend.shape[0], reads]
        + [t.treated.spend.size for c in tests for t in tests[c]]
        + [e.spend.size for c in estimates for e in estimates[c]]
        + [e.spend.size for c in series for e in series[c]]
    )
    padded = np.concatenate([np.zeros((longest - spend.shape[0], len(names))), spend / spend_unit])
    periods = np.concatenate([np.zeros(longest - spend.shape[0], dtype=bool), inside])
    carried = [j for j, m in enumerate(spec.media) if m.roi.counts == "carried"]
    model = BayesModel(
        spec=spec,
        lagged=np.stack([_lags(column, periods) for column in padded.T]),
        y=kpi[observed] / kpi_unit,
        observed=observed,
        calendar=spec.calendar(int(inside.sum())),
        controls=(controls - control_mean) / control_sd,
        window_spend=window_spend / kpi_unit,
        arms=tuple(
            _arms(names.index(c), tests[c], longest, spend_unit[names.index(c)], kpi_unit)
            for c in names
            if c in tests
        ),
        estimates=tuple(
            _estimates(names.index(c), estimates[c], longest, spend_unit[names.index(c)], kpi_unit)
            for c in names
            if c in estimates
        ),
        kpi_unit=kpi_unit,
        spend_unit=spend_unit,
        control_mean=control_mean,
        control_sd=control_sd,
        series=tuple(
            _series(names.index(c), e, longest, spend_unit[names.index(c)], kpi_unit)
            for c in names
            for e in series.get(c, ())
        ),
        tails=tuple(_tail(j, padded[:, j], periods) for j in carried),
        control_baseline=(
            panel.counterfactuals([c.column for c in spec.controls])
            if any(c.column in panel.levers for c in spec.controls)
            else None
        ),
    )
    return model if spec.kernel is None else _cut(model, spec.kernel)


def _cut(model: BayesModel, kernel: int) -> BayesModel:
    """`model` with every kernel `kernel` periods long: each row of lags kept to its first
    `kernel`, lag 0 first, or run on with no spend before the panel where it is shorter, so that a
    spend carries over that many periods and the weights are normalised over them. A carried
    return's rows past the cut carry no gap, so they add nothing."""

    def cut(lags: _Array) -> _Array:
        short = max(0, kernel - lags.shape[-1])
        return np.pad(lags[..., :kernel], [(0, 0)] * (lags.ndim - 1) + [(0, short)])

    return dataclasses.replace(
        model,
        lagged=cut(model.lagged),
        arms=tuple(
            dataclasses.replace(a, treated=cut(a.treated), control=cut(a.control))
            for a in model.arms
        ),
        estimates=tuple(
            dataclasses.replace(e, spend=cut(e.spend), planned=cut(e.planned))
            for e in model.estimates
        ),
        series=tuple(
            dataclasses.replace(s, spend=cut(s.spend), planned=cut(s.planned)) for s in model.series
        ),
        tails=tuple(
            dataclasses.replace(t, spend=cut(t.spend), removed=cut(t.removed)) for t in model.tails
        ),
    )


def _tail(channel: int, spend: _Array, window: npt.NDArray[np.bool_]) -> _Tail:
    """Channel `channel`'s rows of a carried return: `spend` over the model's `L` periods, run on
    for `L - 1` with nothing spent, at each period from `window`'s first to `L - 1` past its last,
    as spent and with the window's spend removed."""
    length = spend.size
    run_on = np.concatenate([spend, np.zeros(length - 1)])
    inside = np.concatenate([window, np.zeros(length - 1, dtype=bool)])
    first, last = np.flatnonzero(window)[[0, -1]]
    rows = np.zeros(run_on.size, dtype=bool)
    rows[first : last + length] = True
    return _Tail(
        channel,
        _lags(run_on, rows)[:, :length],
        _lags(np.where(inside, 0.0, run_on), rows)[:, :length],
    )


def _summary(
    model: BayesModel, theta: _Array
) -> tuple[dict[str, _Array], dict[str, _Array], _Array]:
    """Each draw's parameters in the panel's units, `theta` being `(draws, dimension)`, the
    posterior mean of each term's contribution over the window and of the intercept, and each
    draw's log density of each observation in the panel's units."""
    calendar = model.spec.calendar_terms()
    log_scale = isinstance(model.spec.likelihood, LogNormal)
    raw_controls = jnp.asarray(model.controls * model.control_sd + model.control_mean)
    baseline = model.control_baseline
    if baseline is not None:
        raw_controls = raw_controls - jnp.asarray(baseline)
    calendar_columns = jnp.asarray(model.calendar)
    log_units = jnp.asarray(model._units * math.log(model.kpi_unit))

    def one(t: jax.Array) -> tuple[dict[str, jax.Array], dict[str, jax.Array], jax.Array]:
        p, _ = model._constrained(t)
        saturated, beta = model._terms(p)
        control = p["control"] / model.control_sd
        intercept = p["intercept"][0] - jnp.sum(control * model.control_mean)
        raw = {
            "roi": jnp.exp(p["log_roi"]),
            "retention": p["retention"],
            "half_saturation": p["half"] * model.spend_unit,
            "coefficient": beta * model.kpi_unit,
            "intercept": intercept * model.kpi_unit,
            "sigma": p["sigma"][0] * (1.0 if log_scale else model.kpi_unit),
        }
        if model._sizes["calendar"]:
            raw["calendar"] = p["calendar"] * model.kpi_unit
        if model._sizes["control"]:
            raw["control"] = control * model.kpi_unit
        base = intercept if baseline is None else intercept + jnp.sum(control * baseline)
        if model._sizes["slope"]:
            raw["slope"] = p["slope"]
        if model._sizes["nu"]:
            raw["nu"] = p["nu"][0]
        if model._sizes["test_sigma"]:
            raw["test_sigma"] = p["test_sigma"][0] * model.kpi_unit
        parts = {
            **{m.channel: saturated[:, j] * beta[j] for j, m in enumerate(model.spec.media)},
            **{
                c.column: raw_controls[:, j] * control[j] for j, c in enumerate(model.spec.controls)
            },
            **{name: calendar_columns[:, k] * p["calendar"][k] for k, name in enumerate(calendar)},
            "intercept": jnp.full(model.calendar.shape[0], base),
        }
        scaled = {k: v * model.kpi_unit for k, v in parts.items()}
        return raw, scaled, model._pointwise(p, saturated, beta) - log_units

    raw, parts, pointwise = jax.lax.map(one, jnp.asarray(theta), batch_size=256)
    return (
        {k: np.asarray(v) for k, v in raw.items()},
        {k: np.asarray(v).mean(axis=0) for k, v in parts.items()},
        np.asarray(pointwise),
    )


class _Density(Protocol):
    """What a chain's start reads of a model."""

    @property
    def dimension(self) -> int: ...

    def _log_density(self, theta: jax.Array) -> jax.Array: ...

    def _lift(self, theta: _Array) -> _Array: ...


def _uniform(model: _Density, rng: np.random.Generator, count: int) -> _Array:
    """`count` points uniform in `[-2, 2]` on the sampling coordinates, each lifted into the
    model's support."""
    return np.array([model._lift(x) for x in rng.uniform(-2.0, 2.0, (count, model.dimension))])


@dataclass(frozen=True, slots=True, eq=False)
class _Mapped:
    """`model`'s density on coordinates `u`, `theta = matrix u`, a linear map whose Jacobian is a
    constant the density leaves out."""

    model: BayesModel
    matrix: _Array

    @property
    def dimension(self) -> int:
        return self.model.dimension

    def _log_density(self, theta: jax.Array) -> jax.Array:
        return self.model._log_density(jnp.asarray(self.matrix) @ theta)

    def _lift(self, theta: _Array) -> _Array:
        return np.linalg.solve(self.matrix, self.model._lift(self.matrix @ theta))


def _initial(model: _Density, start: Start, chains: int, seed: int) -> _Array:
    rng = np.random.default_rng(seed)
    if isinstance(start, UniformStart):
        return _uniform(model, rng, chains)
    paths = pathfinder(
        model._log_density, _uniform(model, rng, _PATHS), seed=int(rng.integers(2**63))
    )
    resampled = resample(paths, chains, seed=int(rng.integers(2**63)))
    log.info(
        "chains started by Pathfinder: %d of %d paths, their largest ELBOs %s at iterations %s"
        " of %s, their draws resampled with Pareto shape %.2f",
        len(paths),
        _PATHS,
        [round(p.elbo, 2) for p in paths],
        [p.iteration for p in paths],
        [p.elbos.size for p in paths],
        resampled.pareto_k,
    )
    return resampled.points


def fit_bayes(
    panel: MediaPanel,
    geo: Geo,
    spec: BayesSpec,
    *,
    seed: int,
    window: tuple[np.datetime64, np.datetime64] | None = None,
    tests: Mapping[str, Sequence[LiftTest]] | None = None,
    estimates: Mapping[str, Sequence[LiftEstimate]] | None = None,
    series: Mapping[str, Sequence[LiftSeries]] | None = None,
    chains: int = 4,
    warmup: int = 1000,
    draws: int = 1000,
    start: Start = _PATHFINDER_START,
    coordinates: Coordinates = _NATURAL,
) -> BayesFit:
    """Draw the posterior of `spec` on one geo of `panel` over `window`, its first and last dates,
    by default the whole calendar, its likelihood holding `tests`, each channel's geo tests,
    `estimates`, each channel's lift estimates, and `series`, each channel's lift series, if given.
    Each of `chains` starts as `start` says, drawn from `seed`, on the sampler's `coordinates`.

    Raises:
        ValueError: on JAX in single precision; a geo or column the panel does not have, a window
            off its calendar, a channel that spends nothing over it, fewer than two observed
            periods or a mean KPI not above 0 over them, a KPI not above 0 at one of them under
            log-normal noise, a control that does not vary over it; tests, estimates or series of a
            channel the model does not have, or of none; estimates or series in a geo other than
            `geo`; and what the sampler, or Pathfinder, refuses.
    """
    model = bayes_model(
        panel, geo, spec, window=window, tests=tests, estimates=estimates, series=series
    )
    if isinstance(coordinates, NaturalCoordinates):
        initial = _initial(model, start, chains, seed)
        result = compiled.sample(model._log_density, initial, seed=seed, warmup=warmup, draws=draws)
        positions = result.positions
    else:
        mapped = _Mapped(model, model.qr_map())
        initial = _initial(mapped, start, chains, seed)
        result = compiled.sample(
            mapped._log_density, initial, seed=seed, warmup=warmup, draws=draws
        )
        positions = result.positions @ mapped.matrix.T
    return _fitted(panel, geo, window, model, positions, result.divergent, result.energy)


def _fitted(
    panel: MediaPanel,
    geo: Geo,
    window: tuple[np.datetime64, np.datetime64] | None,
    model: BayesModel,
    positions: _Array,
    divergent: npt.NDArray[np.bool_],
    energy: _Array,
) -> BayesFit:
    """The fit of `model`, built on one geo of `panel` over `window`, from a sampler's `positions`
    on its coordinates, `(chains, draws, dimension)`, and each transition's divergence and energy,
    `(chains, draws)`."""
    chains, draws, _ = positions.shape
    raw, parts, pointwise = _summary(model, positions.reshape(-1, model.dimension))
    contributions = {k: v for k, v in parts.items() if k != "intercept"}
    shaped = {k: v.reshape(chains, draws, *v.shape[1:]) for k, v in raw.items()}
    g = panel.geos.index(geo)
    inside = _window(panel, window)
    through = int(np.flatnonzero(inside)[-1]) + 1
    names = [str(c) for c in panel.channels]
    return BayesFit(
        spec=model.spec,
        dates=panel.dates[inside],
        y=panel.kpi[g, inside],
        fitted=model.observed,
        draws=MappingProxyType(shaped),
        log_likelihood=pointwise.reshape(chains, draws, -1),
        contributions=MappingProxyType(contributions),
        estimated=sum(parts.values(), np.zeros(model.calendar.shape[0])),
        diagnostics=MappingProxyType(
            {k: diagnose(v.reshape(chains, draws, -1)) for k, v in shaped.items()}
        ),
        divergent=divergent.sum(axis=1),
        e_bfmi=e_bfmi(energy),
        length=model.lagged.shape[2],
        spend=MappingProxyType(
            {
                m.channel: panel.spend[g, :through, names.index(m.channel)].copy()
                for m in model.spec.media
            }
        ),
        cadence=panel.cadence,
    )


def lever_contributions(fit: BayesFit, panel: MediaPanel, geo: Geo) -> dict[str, _Array]:
    """Each lever among the fit's controls, its contribution over the fit's window at every draw of
    the posterior, `(chains * draws,)` in the chains' order: the draw's coefficient times the sum
    over the window's periods of the lever's departure from its counterfactual level
    (`MediaPanel.counterfactual`). `panel` and `geo` are the fit's; the draws' mean is the sum of
    the decomposition's `contributions[lever]`, up to rounding.

    Raises:
        ValueError: on a geo the panel does not have, or a fit whose window is not on the panel's
            calendar.
    """
    if geo not in panel.geos:
        msg = f"the panel has no geo {geo!r}; it has {list(panel.geos)}"
        raise ValueError(msg)
    inside = np.isin(panel.dates, fit.dates)
    if int(inside.sum()) != fit.dates.size:
        msg = f"the fit's window, {fit.dates[0]} to {fit.dates[-1]}, is not on the panel's calendar"
        raise ValueError(msg)
    g = panel.geos.index(geo)
    out = {}
    for k, c in enumerate(fit.spec.controls):
        if c.column in panel.levers:
            values = panel.control_values[g, inside, panel.controls.index(c.column)]
            departure = float(np.sum(values - panel.counterfactual(c.column)))
            out[c.column] = fit.draws["control"][..., k].reshape(-1) * departure
    return out


@dataclass(frozen=True, slots=True, eq=False)
class BayesForecast:
    """Draws of the KPI over the periods after a fit's window in the panel's units, `(draws,
    horizon)`: `mean`, each draw's mean, and `kpi`, the KPI drawn about it."""

    dates: npt.NDArray[np.datetime64]
    mean: _Array
    kpi: _Array


def forecast_bayes(
    fit: BayesFit, horizon: int, plan: Mapping[str, npt.ArrayLike], draws: int, seed: int
) -> BayesForecast:
    """The model's posterior predictive over the `horizon` periods after the fit's window, `plan`
    giving each channel's spend and each control's value at each: `draws` draws, each at a draw of
    the posterior as `BayesFit.channel_draws` reads them, its noise drawn from `seed`.

    A channel's spend carries over from the window's into the plan's through the kernel the fit
    read; the trend runs on from the window's last bend and the season on its period.

    Raises:
        ValueError: on JAX in single precision; a horizon below 1, or a count of draws below 1 or
            above the fit's; a plan whose names are not the model's channels and controls, or a
            planned series that is not `horizon` finite values, a channel's at or above 0; and,
            under log-normal noise, a draw whose mean is not above 0 at a period, where the model
            has no KPI.
    """
    if not jax.config.x64_enabled:
        msg = "the model needs JAX in double precision (JAX_ENABLE_X64=1)"
        raise ValueError(msg)
    x = _planned(fit, horizon, plan)
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        msg = f"draws must be a positive integer, got {draws!r}"
        raise ValueError(msg)
    keep = _thinned(fit.draws["roi"].shape, draws)

    def at(name: str) -> _Array:
        every = fit.draws[name]
        return every.reshape(-1, *every.shape[2:])[keep]

    window = fit.dates.size
    mean = np.repeat(at("intercept")[:, None], horizon, axis=1)
    if "calendar" in fit.draws:
        mean += at("calendar") @ fit.spec.calendar(window, ahead=horizon)[window:].T
    if "control" in fit.draws:
        mean += at("control") @ np.column_stack([x[c.column] for c in fit.spec.controls]).T
    tanh = isinstance(fit.spec.saturation, TanhCurve)
    for j, m in enumerate(fit.spec.media):
        series = jnp.asarray(np.concatenate([fit.spend[m.channel], x[m.channel]]))

        def ahead(*parameters: jax.Array, series: jax.Array = series) -> jax.Array:
            return fit._channel(*parameters)(series)[-horizon:]

        slope = np.ones(draws) if tanh else at("slope")[:, j]
        mean += np.asarray(
            jax.vmap(ahead)(
                *(jnp.asarray(at(k)[:, j]) for k in ("retention", "half_saturation")),
                jnp.asarray(slope),
                jnp.asarray(at("coefficient")[:, j]),
            )
        )
    kpi = _noised(fit, mean, at, np.random.default_rng(seed))
    log.info(
        "the posterior predictive drawn",
        extra={
            "draws": draws,
            "horizon": horizon,
            "likelihood": type(fit.spec.likelihood).__name__,
        },
    )
    return BayesForecast(calendar(fit.dates[-1], horizon + 1, fit.cadence)[1:], mean, kpi)


def _noised(
    fit: BayesFit, mean: _Array, at: Callable[[str], _Array], rng: np.random.Generator
) -> _Array:
    """The KPI drawn about `mean`, `(draws, periods)`, by the model's noise at the draws `at` reads
    of each parameter.

    Raises:
        ValueError: under log-normal noise, a draw whose mean is not above 0 at a period, where
            the model has no KPI.
    """
    sigma = at("sigma")[:, None]
    if isinstance(fit.spec.likelihood, LogNormal):
        if not (mean > 0.0).all():
            low = mean <= 0.0
            msg = (
                f"under log-normal noise the mean is not above 0 at steps "
                f"{np.flatnonzero(low.any(axis=0)).tolist()} in {int(low.any(axis=1).sum())} of "
                f"the {mean.shape[0]} draws, where the model has no KPI"
            )
            raise ValueError(msg)
        return mean * np.exp(sigma * rng.standard_normal(mean.shape) - sigma**2 / 2.0)
    if isinstance(fit.spec.likelihood, StudentT):
        return mean + sigma * rng.standard_t(at("nu")[:, None], mean.shape)
    return mean + sigma * rng.standard_normal(mean.shape)


def _planned(fit: BayesFit, horizon: int, plan: Mapping[str, npt.ArrayLike]) -> dict[str, _Array]:
    """Each channel's spend and each control's value over the `horizon` periods after the fit's
    window; see `forecast_bayes`.

    Raises:
        ValueError: what `forecast_bayes` refuses of `horizon` and `plan`.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        msg = f"horizon must be a positive integer, got {horizon!r}"
        raise ValueError(msg)
    channels = [m.channel for m in fit.spec.media]
    names = [*channels, *(c.column for c in fit.spec.controls)]
    missing = [c for c in names if c not in plan]
    extra = sorted(set(plan) - set(names))
    if missing or extra:
        msg = (
            f"the plan must name the channels and controls {names}: it lacks {missing} and adds"
            f" {extra}"
        )
        raise ValueError(msg)
    out = {}
    for name in names:
        values = np.asarray(plan[name], dtype=np.float64)
        if values.shape != (horizon,):
            msg = f"the plan's {name!r} has shape {values.shape}, not ({horizon},)"
            raise ValueError(msg)
        if not np.isfinite(values).all():
            steps = np.flatnonzero(~np.isfinite(values)).tolist()
            msg = f"the plan's {name!r} is not finite at steps {steps}"
            raise ValueError(msg)
        if name in channels and (values < 0.0).any():
            steps = np.flatnonzero(values < 0.0).tolist()
            msg = f"the plan's {name!r} spends below 0 at steps {steps}"
            raise ValueError(msg)
        out[name] = values
    return out


type HoldoutKind = Literal["blocked", "interleaved"]
"""Which of a window's observed periods a holdout keeps out of the fit: its last ones,
`"blocked"`, or a draw from all of them, `"interleaved"`."""


@dataclass(frozen=True, slots=True, eq=False)
class BayesHoldout:
    """The model's forecast of periods held out of its fit: the holdout's `kind`; the `dates` held
    out, in order; the `kpi` observed at them; the posterior predictive's `draws` of the KPI there,
    `(draws, periods)`, one at each of the fit's draws; and `crps`, the predictive's continuous
    ranked probability score averaged over the periods, over the KPI's mean over the window's
    observed periods, so that one panel's holdout reads against another's. A factsheet records it
    as a `mixdyn.api.Holdout`: its first and last dates, its count of periods and its score."""

    kind: HoldoutKind
    dates: npt.NDArray[np.datetime64]
    kpi: _Array
    draws: _Array
    crps: float


def holdout_bayes(
    panel: MediaPanel,
    geo: Geo,
    spec: BayesSpec,
    *,
    periods: int,
    seed: int,
    kind: HoldoutKind = "blocked",
    window: tuple[np.datetime64, np.datetime64] | None = None,
    tests: Mapping[str, Sequence[LiftTest]] | None = None,
    estimates: Mapping[str, Sequence[LiftEstimate]] | None = None,
    series: Mapping[str, Sequence[LiftSeries]] | None = None,
    chains: int = 4,
    warmup: int = 1000,
    draws: int = 1000,
    start: Start = _PATHFINDER_START,
    coordinates: Coordinates = _NATURAL,
) -> BayesHoldout:
    """`spec` fitted to one geo of `panel` over `window` with `periods` of the window's observed
    periods held out of its likelihood, and its posterior predictive scored on them.

    `kind` picks the periods: `"blocked"`, the default, the window's last `periods` observed ones,
    as the model would forecast the periods after its window; `"interleaved"`, `periods` drawn
    from all of them at random. An interleaved holdout leaks: each held-out period keeps its
    neighbours in the fit, and where the KPI's noise is autocorrelated the terms that follow time,
    the trend and the season, are fitted to the noise a held-out period shares with its
    neighbours, so its score reads better than any forecast the model makes past its window (ADR
    0040, amended, and ADR 0079).

    A held-out period's KPI is masked, as a period the panel did not observe, so the window, its
    trend's bends, its season and every channel's carryover are the fit's own. The forecast at a
    held-out period is the model's mean there at each draw of the posterior, the controls and the
    spend as the panel holds them, and the KPI drawn about it by the model's noise; its score is
    the CRPS of those draws. The fit's seed, the interleaved periods' and the noise's are children
    of `seed`. The rest is `fit_bayes`'.

    Raises:
        ValueError: on a kind neither `"blocked"` nor `"interleaved"`; a geo the panel does not
            have; a count of periods below 1 or not below the window's observed periods; and what
            `fit_bayes` refuses.
    """
    if kind not in ("blocked", "interleaved"):
        msg = f'a holdout is "blocked" or "interleaved", got {kind!r}'
        raise ValueError(msg)
    if geo not in panel.geos:
        msg = f"the panel has no geo {geo!r}; it has {list(panel.geos)}"
        raise ValueError(msg)
    g = panel.geos.index(geo)
    inside = _window(panel, window)
    observed = np.flatnonzero(inside & np.isfinite(panel.kpi[g]))
    if (
        isinstance(periods, bool)
        or not isinstance(periods, int)
        or not 1 <= periods < observed.size
    ):
        msg = (
            f"a holdout keeps 1 to {observed.size - 1} of the window's observed periods, "
            f"not {periods}"
        )
        raise ValueError(msg)
    fitting, picking, noise = (
        int(child.generate_state(1)[0]) for child in np.random.SeedSequence(seed).spawn(3)
    )
    if kind == "blocked":
        held = observed[-periods:]
    else:
        held = np.sort(np.random.default_rng(picking).choice(observed, periods, replace=False))
    kpi = panel.kpi.copy()
    kpi[g, held] = np.nan
    fit = fit_bayes(
        dataclasses.replace(panel, kpi=kpi),
        geo,
        spec,
        seed=fitting,
        window=window,
        tests=tests,
        estimates=estimates,
        series=series,
        chains=chains,
        warmup=warmup,
        draws=draws,
        start=start,
        coordinates=coordinates,
    )
    controls = panel.regressors([c.column for c in spec.controls], geo, inside)
    places = held - int(np.flatnonzero(inside)[0])
    mean = _window_means(fit, controls)[:, places]
    predictive = _noised(fit, mean, partial(_every, fit), np.random.default_rng(noise))
    target = panel.kpi[g, held]
    scale = float(np.mean(panel.kpi[g, observed]))
    log.info(
        "a holdout scored",
        extra={"kind": kind, "periods": periods, "draws": predictive.shape[0]},
    )
    return BayesHoldout(
        kind, panel.dates[held], target, predictive, _crps(predictive, target) / scale
    )


def _window_means(fit: BayesFit, controls: _Array) -> _Array:
    """Each draw's mean of the KPI at each of the fit's window's periods in the panel's units,
    `(draws, window)`, the draws in chain order, `controls` the window's values of the spec's
    controls, `(window, controls)`: averaged over the draws, the fit's `estimated`."""
    every = partial(_every, fit)
    window = fit.dates.size
    mean = np.repeat(every("intercept")[:, None], window, axis=1)
    if "calendar" in fit.draws:
        mean += every("calendar") @ fit.spec.calendar(window).T
    if "control" in fit.draws:
        mean += every("control") @ controls.T
    tanh = isinstance(fit.spec.saturation, TanhCurve)
    for j, m in enumerate(fit.spec.media):
        spend = jnp.asarray(fit.spend[m.channel])

        def inside(*parameters: jax.Array, spend: jax.Array = spend) -> jax.Array:
            return fit._channel(*parameters)(spend)[-window:]

        slope = np.ones(mean.shape[0]) if tanh else every("slope")[:, j]
        mean += np.asarray(
            jax.vmap(inside)(
                *(jnp.asarray(every(k)[:, j]) for k in ("retention", "half_saturation")),
                jnp.asarray(slope),
                jnp.asarray(every("coefficient")[:, j]),
            )
        )
    return mean


def _every(fit: BayesFit, name: str) -> _Array:
    """Every draw of the fit's parameter `name`, the chains' draws in chain order leading."""
    drawn = fit.draws[name]
    return drawn.reshape(-1, *drawn.shape[2:])


def _crps(draws: _Array, target: _Array) -> float:
    """The CRPS of the draws' empirical distribution, `(draws, periods)`, at `target`, averaged
    over the periods: `E|X - y| - E|X - X'| / 2`, the second term read off the sorted draws."""
    n = draws.shape[0]
    spread = np.abs(draws - target).mean(axis=0)
    weights = (2.0 * np.arange(1, n + 1) - n - 1)[:, None]
    half = (weights * np.sort(draws, axis=0)).sum(axis=0) / n**2
    return float(np.mean(spread - half))


WEIGHT = 0.25
"""`refresh_bayes`' default power on the previous periods' evidence: they count 1.25 times in a
return's prior and likelihood together, and the refreshed mean lies at most a fifth of the way
from the full refit's towards their own reading (ADR 0079)."""


@dataclass(frozen=True, slots=True)
class ReturnDrift:
    """A channel's return on the previous fit's window's spend, as its prior counts it, in the
    KPI's units a currency unit: `before`, the previous posterior's mean, and `low` to `high` its
    central `level` interval; `after`, the refreshed posterior's mean of the same return."""

    channel: str
    before: float
    after: float
    low: float
    high: float
    level: float

    @property
    def change(self) -> float:
        """The refreshed mean less the previous one."""
        return self.after - self.before

    @property
    def leaves(self) -> bool:
        """Whether the refreshed mean lies outside the previous posterior's interval."""
        return not self.low <= self.after <= self.high


@dataclass(frozen=True, slots=True, eq=False)
class BayesRefresh:
    """A refreshed fit: `fit`, the model fitted again to the whole panel, its return priors
    tempered by the previous posterior; `weight`, the power the previous periods' evidence on each
    return was raised to; and `drift`, each channel's `ReturnDrift`, in the spec's order."""

    fit: BayesFit
    weight: float
    drift: tuple[ReturnDrift, ...]


def tempered_spec(spec: BayesSpec, previous: BayesFit, weight: float) -> BayesSpec:
    """`spec` with each channel's return prior times the evidence the previous fit's periods gave
    that return, raised to `weight`: a power prior.

    The evidence is read on the log return, as normal: the previous posterior's draws of it give a
    mean `m` and a variance `s^2`, and their normal over the previous prior's, `N(mu_p, sigma_p)`,
    is the evidence, of precision `1/s^2 - 1/sigma_p^2`, none where that is not above 0, as on a
    channel the data did not narrow. `spec`'s prior `N(mu, sigma)` times the evidence to the power
    `a` is normal again: its precision `1/sigma^2 + a (1/s^2 - 1/sigma_p^2)`, and its mean that
    over the precision `mu/sigma^2 + a (m/s^2 - mu_p/sigma_p^2)` (validation/refresh_prior.mac).
    Dividing by the previous fit's own prior, and multiplying into `spec`'s, keeps a refresh of a
    refresh from counting the earliest periods once more each time, where `spec` is the base
    model's. The previous return is on its window's spend and the refitted one on the panel's, the
    two the same return where the spend holds its level; every other prior is `spec`'s.

    Raises:
        ValueError: on a weight outside `[0, 1]`; a spec whose channels are not the previous
            fit's; or, at a weight above 0, a channel whose prior in `spec` or in the previous fit
            is not log-normal.
    """
    if not 0.0 <= weight <= 1.0:
        msg = f"a refresh's weight is from 0 to 1, got {weight}"
        raise ValueError(msg)
    names = [m.channel for m in spec.media]
    before = {m.channel: m for m in previous.spec.media}
    if sorted(names) != sorted(before):
        msg = f"the spec's channels {names} are not the previous fit's {list(before)}"
        raise ValueError(msg)
    if weight == 0.0:
        return spec
    unread = [
        m.channel
        for m in spec.media
        if not (isinstance(m.roi, RoiPrior) and isinstance(before[m.channel].roi, RoiPrior))
    ]
    if unread:
        msg = (
            f"a tempered prior is log-normal, and {unread} have another prior here or in the "
            "previous fit: refresh them at weight 0"
        )
        raise ValueError(msg)
    order = [m.channel for m in previous.spec.media]
    media = []
    for m in spec.media:
        prior, old = cast("RoiPrior", m.roi), cast("RoiPrior", before[m.channel].roi)
        log_roi = np.log(previous.draws["roi"][..., order.index(m.channel)]).reshape(-1)
        mean, variance = float(np.mean(log_roi)), float(np.var(log_roi, ddof=1))
        evidence = 1.0 / variance - 1.0 / old.sigma**2
        if not evidence > 0.0:
            media.append(m)
            continue
        precision = 1.0 / prior.sigma**2 + weight * evidence
        centre = prior.mu / prior.sigma**2 + weight * (mean / variance - old.mu / old.sigma**2)
        roi = RoiPrior(centre / precision, 1.0 / math.sqrt(precision), prior.counts)
        media.append(dataclasses.replace(m, roi=roi))
    return dataclasses.replace(spec, media=tuple(media))


def refresh_bayes(
    panel: MediaPanel,
    geo: Geo,
    previous: BayesFit,
    *,
    seed: int,
    weight: float = WEIGHT,
    level: float = 0.8,
    spec: BayesSpec | None = None,
    window: tuple[np.datetime64, np.datetime64] | None = None,
    tests: Mapping[str, Sequence[LiftTest]] | None = None,
    estimates: Mapping[str, Sequence[LiftEstimate]] | None = None,
    series: Mapping[str, Sequence[LiftSeries]] | None = None,
    chains: int = 4,
    warmup: int = 1000,
    draws: int = 1000,
    start: Start = _PATHFINDER_START,
    coordinates: Coordinates = _NATURAL,
) -> BayesRefresh:
    """`spec`, by default the previous fit's, fitted again to one geo of `panel` over `window`,
    which holds the previous fit's window and the periods added since, with each channel's return
    prior tempered by the previous posterior (`tempered_spec`); and each channel's drift.

    The refit reads every period once in its likelihood, the previous ones among them, so a prior
    that were the previous posterior itself would count them twice. Tempered at `weight` `a`, the
    previous periods' evidence on a return counts `1 + a` times: at 0 the refresh is the full
    refit with `spec`'s priors, draw for draw; at the default, 0.25, the refreshed mean of a log
    return lies at most `a / (1 + a)`, a fifth, of the way from the full refit's towards the
    previous periods' own reading, and its spread at least `1 / sqrt(1 + a)` of the full refit's,
    for normal evidence (validation/refresh_prior.mac). The weight is the stability a refresh
    buys, stated: the previous reading holds the return back by at most that much.

    The drift is each channel's return on the previous window's spend, as its prior counts it,
    under the previous posterior and the refreshed one, and whether the refreshed mean leaves the
    previous posterior's central `level` interval (`ReturnDrift`). The rest is `fit_bayes`'.

    Raises:
        ValueError: on a level outside `(0, 1)`; a window that does not hold the previous fit's;
            what `tempered_spec` refuses; and what `fit_bayes` refuses.
    """
    if not 0.0 < level < 1.0:
        msg = f"a drift's interval has a level inside (0, 1), got {level}"
        raise ValueError(msg)
    if geo not in panel.geos:
        msg = f"the panel has no geo {geo!r}; it has {list(panel.geos)}"
        raise ValueError(msg)
    inside = _window(panel, window)
    if not np.isin(previous.dates, panel.dates[inside]).all():
        msg = (
            f"a refresh's window holds the previous fit's, {previous.dates[0]} to "
            f"{previous.dates[-1]}"
        )
        raise ValueError(msg)
    base = previous.spec if spec is None else spec
    fit = fit_bayes(
        panel,
        geo,
        tempered_spec(base, previous, weight),
        seed=seed,
        window=window,
        tests=tests,
        estimates=estimates,
        series=series,
        chains=chains,
        warmup=warmup,
        draws=draws,
        start=start,
        coordinates=coordinates,
    )
    after = returns_on(fit, panel, geo, previous.dates)
    tail = (1.0 - level) / 2.0
    order = [m.channel for m in previous.spec.media]
    drift = []
    for j, m in enumerate(fit.spec.media):
        old = previous.draws["roi"][..., order.index(m.channel)].reshape(-1)
        low, high = np.quantile(old, [tail, 1.0 - tail])
        drift.append(
            ReturnDrift(
                m.channel,
                float(np.mean(old)),
                float(np.mean(after[:, j])),
                float(low),
                float(high),
                level,
            )
        )
    log.info(
        "a fit refreshed",
        extra={"weight": weight, "leaves": sum(d.leaves for d in drift)},
    )
    return BayesRefresh(fit, weight, tuple(drift))


def returns_on(
    fit: BayesFit, panel: MediaPanel, geo: Geo, dates: npt.NDArray[np.datetime64]
) -> _Array:
    """Each channel's return on its spend over `dates` of `panel`, consecutive periods of its
    calendar, at each draw of `fit`, `(draws, channels)` in the chains' and the spec's orders, as
    the channel's prior counts it (`RoiPrior.counts`): its contribution over those periods, or the
    whole return of their spend with its carryover, each over the spend. On the fit's own panel,
    geo and window it is the fit's `roi` draws.

    Raises:
        ValueError: on a geo the panel does not have, dates that are not consecutive periods of its
            calendar, or a channel that spends nothing over them.
    """
    if geo not in panel.geos:
        msg = f"the panel has no geo {geo!r}; it has {list(panel.geos)}"
        raise ValueError(msg)
    places = np.flatnonzero(np.isin(panel.dates, dates))
    if places.size != len(dates) or places.size == 0 or np.any(np.diff(places) != 1):
        msg = "the return's dates are consecutive periods of the panel's calendar"
        raise ValueError(msg)
    every = partial(_every, fit)
    g, channels = panel.geos.index(geo), [str(c) for c in panel.channels]
    first, last = int(places[0]), int(places[-1])
    tanh = isinstance(fit.spec.saturation, TanhCurve)
    out = []
    for j, m in enumerate(fit.spec.media):
        run_on = np.concatenate(
            [panel.spend[g, :, channels.index(m.channel)], np.zeros(fit.length - 1)]
        )
        spent = float(run_on[first : last + 1].sum())
        if not spent > 0.0:
            msg = f"{m.channel} spends nothing over the return's dates"
            raise ValueError(msg)
        if m.roi.counts == "carried":
            removed = run_on.copy()
            removed[first : last + 1] = 0.0
            rows = slice(first, last + fit.length)
        else:
            removed, rows = np.zeros_like(run_on), slice(first, last + 1)

        spend, without = jnp.asarray(run_on), jnp.asarray(removed)

        def earned(
            *parameters: jax.Array,
            spend: jax.Array = spend,
            without: jax.Array = without,
            rows: slice = rows,
        ) -> jax.Array:
            channel = fit._channel(*parameters)
            return jnp.sum((channel(spend) - channel(without))[rows])

        slope = np.ones(every("roi").shape[0]) if tanh else every("slope")[:, j]
        returned = jax.vmap(earned)(
            *(jnp.asarray(every(k)[:, j]) for k in ("retention", "half_saturation")),
            jnp.asarray(slope),
            jnp.asarray(every("coefficient")[:, j]),
        )
        out.append(np.asarray(returned) / spent)
    return np.stack(out, axis=1)
