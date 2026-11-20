"""A static media-mix model over geos in Bayes' form, each channel's effect partially pooled over
the geos, its posterior drawn by the engine's own NUTS (`mixdyn.inference`).

    y_gt = a_g (1 + calendar_t . delta) + controls_gt . gamma
               + sum_m beta_gm s_m(adstock_m(spend_gm))_t + e_gt

per head of each geo `g`, over the observed cells of the panel's window: a geo's KPI and spend are
divided by its population, and `e_gt` is normal or Student-t of scale `sigma / sqrt(w_g)`, `w_g`
the geo's population over the mean, so that a geo's noise variance in totals is in proportion to
its population (ADR 0006). A panel without populations has one head a geo. The scales are the
one-geo model's (`mixdyn.models.bayes_mmm`) pooled over the geos: the KPI's unit is its mean per
head over the observed cells, a channel's its mean positive spend per head over the panel's
calendar and geos.

* A channel's carryover and curve are the one-geo model's, shared by the geos: a normalised
  geometric kernel as long as the panel's calendar, then a Hill or `tanh`.
* A geo's base per head `a_g` multiplies the calendar, so the shared trend and season move each
  geo in proportion to its base.
* A channel's effect in a geo is its national effect times the geo's own factor:
  `beta_gm = b_m exp(eta_m z_gm)`. `b_m` is not sampled but set, as the one-geo coefficient is, by
  the channel's return on the window's spend over every geo, `roi_m`: the contribution over the
  window and the geos is `roi_m` times the spend over them. A shift of every `z_gm` of one channel
  would then move only `b_m`, unseen by the likelihood, so a channel's `z` sum to 0 over the geos:
  `z_m = H u_m`, `H` Helmert's orthonormal contrasts and `u_m ~ N(0, I)` over `G - 1`
  coordinates, which is `z_gm ~ N(0, 1)` with the shift integrated out.
* A channel's deviations `u_m` are sampled as they are, non-centred, unless the spec centres the
  channel: then its coordinates are `eta_m u_m`, `N(0, eta_m^2 I)`, the same model on coordinates
  that suit data which pin the channel's effect in every geo, where the non-centred ones narrow as
  `eta_m` grows.
* A control's coefficient is shared, the control standardised over the window's cells that were
  not filled; a filled cell, which has no KPI, reads each control at its mean.
* A national channel split by population has the same spend per head in every geo, so the geos
  hold no contrast in it: its effect is read from the weeks alone, as the one-geo model reads it,
  and its factors over the geos from how each geo follows the same series, under the pooling
  (ADR 0006).

Lift estimates (`mixdyn.calibration.LiftEstimate`) enter the likelihood as the one-geo model's do,
each read in the geos its test ran in, every geo where it names none. Its group's spend over its
share is spread over those geos' people, the same spend per head in each, carried over and
saturated as the channel's own; its lift over its share is normal about the gap in saturated spend
summed over the readout times those geos' effects, each weighted by its population, with the
estimate's own standard error.

The priors, on the scales above, are the one-geo model's where it has them:

    log roi_m ~ N(mu_m, sigma_m)    r_m ~ Beta(1, 3)    K_m ~ LogNormal(0, 1)
    n_m ~ LogNormal(0, 0.5)         eta_m ~ HalfNormal(spread), 1 by default, Meridian's
    a_g ~ LogNormal(0, 1)           delta ~ N(0, 1)     gamma ~ N(0, 1), held to its sign
    sigma ~ HalfNormal(1)           nu ~ Gamma(2, 0.1)

Three choices are opt-in, each a hypothesis about where a centred fit diverges (ADR 0059), and none
moves the default: `noise_prior="log-normal"`, `sigma ~ LogNormal(log 0.6745, 0.5)`, the
half-normal's median, held away from 0; `base_prior="normal"`, `a_g ~ N(1, 1)`, the one-geo
intercept's prior, a base allowed below 0 and sampled as it is; and `slope="fixed"`, every Hill's
slope 1 and not sampled.

The parameters are sampled on the real line through Stan's transforms, in the order of the Stan
program that tests/fixtures/oracles/bayes_geo_stan.json holds beside CmdStan's readings of it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from chc import response
from jax.scipy import stats as jstats

from mixdyn.data.panel import refuse_levers
from mixdyn.inference import compiled
from mixdyn.inference.diagnostics import Diagnostics, diagnose, e_bfmi
from mixdyn.models.bayes_mmm import (
    _TANH_HALF,
    _UNIFORM_START,
    BayesMedia,
    BayesSpec,
    HillCurve,
    Normal,
    RoiPrior,
    Saturation,
    Start,
    StudentT,
    TanhCurve,
    _initial,
    _lags,
    _log_normal,
    _thinned,
)
from mixdyn.models.ridge_mmm import Control, FourierSeason, LinearTrend, _window

if TYPE_CHECKING:
    from mixdyn.calibration import LiftEstimate
    from mixdyn.data import Geo, MediaPanel

type _Array = npt.NDArray[np.float64]

type GeoLikelihood = Normal | StudentT
"""The noises the model over geos reads; log-normal noise is the one-geo model's alone."""

_BATCH = 256
"""Draws summarised at once: the contributions are summed batch by batch, never held a draw at a
time."""
_NOISE_LOG_NORMAL = (math.log(0.6745), 0.5)
"""`noise_prior="log-normal"`'s location and scale: its median the half-normal's, `Phi^-1(3/4)`."""


@dataclass(frozen=True, slots=True)
class BayesGeoSpec:
    """The model over geos: the one-geo model's channels, controls, calendar, noise and curves; the
    scale of the half-normal prior on each channel's log-SD of its effect over the geos; the
    channels whose deviations are `centred`, sampled on the scale of that log-SD; and three opt-in
    choices, the noise scale's prior, `"half-normal"` or `"log-normal"`, the geos' bases',
    `"log-normal"`, above 0, or `"normal"`, and a Hill's slope, `"sampled"` or `"fixed"` at 1.

    Raises:
        ValueError: on what `BayesSpec` refuses, on noise neither normal nor Student-t, on a spread
            not positive and finite, on a centred channel the spec does not have, on a return
            prior that counts the carried return or is not log-normal, on a prior or a slope not
            named above, and on a slope fixed under `tanh` curves, which have none.
    """

    media: tuple[BayesMedia, ...]
    controls: tuple[Control, ...] = ()
    trend: LinearTrend | None = None
    season: FourierSeason | None = None
    likelihood: GeoLikelihood = field(default_factory=Normal)
    saturation: Saturation = field(default_factory=HillCurve)
    spread: float = 1.0
    centred: frozenset[str] = frozenset()
    noise_prior: Literal["half-normal", "log-normal"] = "half-normal"
    base_prior: Literal["log-normal", "normal"] = "log-normal"
    slope: Literal["sampled", "fixed"] = "sampled"

    def __post_init__(self) -> None:
        self.one_geo()
        if not isinstance(self.likelihood, Normal | StudentT):
            msg = f"the model over geos reads normal or Student-t noise, not {self.likelihood}"
            raise ValueError(msg)
        if not 0.0 < self.spread < math.inf:
            msg = f"need a positive, finite spread, got {self.spread}"
            raise ValueError(msg)
        unknown = sorted(self.centred - {m.channel for m in self.media})
        if unknown:
            msg = f"centred channels {unknown} are not among the spec's media"
            raise ValueError(msg)
        carried = [m.channel for m in self.media if m.roi.counts == "carried"]
        if carried:
            msg = f"the model over geos states each return on the window: {carried} carry it"
            raise ValueError(msg)
        other = [m.channel for m in self.media if not isinstance(m.roi, RoiPrior)]
        if other:
            msg = f"the model over geos pools log returns, so log-normal priors: {other} are not"
            raise ValueError(msg)
        for name, value, allowed in (
            ("noise prior", self.noise_prior, ("half-normal", "log-normal")),
            ("base prior", self.base_prior, ("log-normal", "normal")),
            ("slope", self.slope, ("sampled", "fixed")),
        ):
            if value not in allowed:
                msg = f"a {name} is one of {list(allowed)}, got {value!r}"
                raise ValueError(msg)
        if self.slope == "fixed" and isinstance(self.saturation, TanhCurve):
            msg = "a tanh curve has no slope to fix"
            raise ValueError(msg)

    def one_geo(self) -> BayesSpec:
        """`fit_bayes`' spec of the same channels, controls, calendar, noise and curves."""
        return BayesSpec(
            self.media, self.controls, self.trend, self.season, self.likelihood, self.saturation
        )


@dataclass(frozen=True, slots=True, eq=False)
class BayesGeoFit:
    """The posterior of the model fitted to every geo of a panel over a window of its calendar.

    Arrays run over the geos, then the window's periods; `fitted` marks the cells the likelihood
    read. `draws` holds each parameter's draws in the panel's units, `(chains, draws)` leading:
    each channel's `roi` over every geo, `retention`, `half_saturation` in spend per head, a Hill's
    `slope` where it is sampled, and `spread`, the log-SD of its effect over the geos;
    `coefficient`, `(geo, channel)`, a channel's contribution per head of a geo per unit of
    saturated spend; `base`, a geo's KPI per head before media and calendar, its controls at their
    means; `calendar`, each column's coefficient as a share of the base, and `control`, per head
    per unit of the control, where the model has them; `sigma`, the noise's scale per head at the
    mean population; and `nu` where the model has it.
    `log_likelihood` is each draw's log density of each observation in the panel's units,
    `(chains, draws, observations)`: each fitted cell's KPI, geo by geo, then each lift estimate's
    lift over its share, channel by channel in the model's order and each channel's in the order
    given. `contributions` and `estimated` are posterior means in the panel's totals, each `(geo,
    period)`: a channel's; a control's, its coefficient times its value; and a calendar column's,
    the base it moves. The base's is the rest of `estimated`.
    `diagnostics`, `divergent`, `e_bfmi` and `length` are as `BayesFit`'s; `population` is each
    geo's, one where the panel has none.
    """

    spec: BayesGeoSpec
    dates: npt.NDArray[np.datetime64]
    geos: tuple[Geo, ...]
    population: _Array
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

    def channels(self, geo: Geo) -> dict[str, response.Channel]:
        """Each channel in `geo` as `chc.response` reads it in the panel's totals, at its
        parameters' posterior medians: the half-saturation point and the coefficient per head,
        each times the geo's population.

        Raises:
            ValueError: on a geo the fit does not have.
        """
        return self._channels(geo, lambda draws: float(np.median(draws)))

    def channel_draws(self, geo: Geo, count: int) -> list[dict[str, response.Channel]]:
        """Each channel in `geo` as `channels` reads it, at `count` of the posterior's draws
        evenly spaced over the chains' draws in order from the first to the last, as
        `BayesFit.channel_draws` reads them.

        Raises:
            ValueError: on a geo the fit does not have, and a count below 1 or above the fit's
                draws.
        """
        keep = _thinned(self.draws["roi"].shape, count)
        return [self._channels(geo, lambda draws, i=i: float(draws.reshape(-1)[i])) for i in keep]

    def _channels(self, geo: Geo, read: Callable[[_Array], float]) -> dict[str, response.Channel]:
        """Each channel in `geo` at the value `read` takes of each of its parameters' `(chains,
        draws)`."""
        if geo not in self.geos:
            msg = f"the fit has no geo {geo!r}; it has {list(self.geos)}"
            raise ValueError(msg)
        g = self.geos.index(geo)
        people = float(self.population[g])
        out = {}
        for j, m in enumerate(self.spec.media):
            retention, half = (
                read(self.draws[k][..., j]) for k in ("retention", "half_saturation")
            )
            coefficient = read(self.draws["coefficient"][..., g, j])
            kernel = response.GeometricAdstock(retention, length=self.length, normalized=True)
            if isinstance(self.spec.saturation, TanhCurve):
                curve = response.Tanh(people * half / _TANH_HALF)
            else:
                fixed = self.spec.slope == "fixed"
                slope = 1.0 if fixed else read(self.draws["slope"][..., j])
                curve = response.Hill(people * half, slope)
            out[m.channel] = response.Channel(kernel, curve, people * coefficient)
        return out


@dataclass(frozen=True, slots=True, eq=False)
class _Estimates:
    """One channel's lift estimates: the spend per head each geo of a test's group saw, over the
    model's spend unit, as the test ran it and as planned, at each estimate's readout periods and
    the `L - 1` before each, in estimate order; the estimate each row reads; each geo's weight in
    each estimate, its population over the mean where the test ran and 0 elsewhere; and each
    estimate's lift and standard error over its share, over the KPI's unit and the mean
    population."""

    channel: int
    spend: _Array  # (readout periods, L)
    planned: _Array
    estimate: npt.NDArray[np.intp]  # (readout periods,)
    tested: _Array  # (estimates, G)
    lift: _Array  # (estimates,)
    se: _Array


@dataclass(frozen=True, slots=True, eq=False)
class BayesGeoModel:
    """The model's posterior density on its sampling coordinates, with the data it reads.

    `log_density` takes a point of the `dimension` coordinates, named by `coordinates` in their
    order: each channel's log return, logit decay, log half-saturation point, log slope if its
    curve is a Hill whose slope is sampled, and log spread; each channel's `G - 1` deviations,
    channel by channel, on the spread's scale for a centred channel; each geo's log base, or its
    base under a normal prior; the calendar's coefficients; the controls' (the log of a
    coefficient's size, if held to a sign); the log noise scale; and the log degrees of freedom
    where the model has them. It returns the log density, up to a constant, with every
    transform's Jacobian, and its gradient. `parameters` maps points to the panel's units, as
    `BayesGeoFit.draws` holds them, and `log_likelihood` to each observation's log density in the
    panel's units, as `BayesGeoFit.log_likelihood` holds it.

    The data are per head, on the model's scales, `kpi_unit` the KPI's and `spend_unit` each
    channel's: `spend` each geo's spend over its unit at every period of the calendar, `(G, T,
    M)`; `lags` each window period's lag behind every period of the calendar, negative after it,
    `(window, T)`; `y` the KPI over its unit at the `observed` cells, geo by geo; `weight` each
    geo's `population` over the mean; `calendar` the window's columns, and `controls` each geo's,
    `(G, window, C)`, standardised by `control_mean` and `control_sd`; `window_spend` each
    channel's spend over the window and the geos, over the KPI's unit and the mean population;
    `estimates` the lift estimates, channel by channel; and `length`, `L`, the kernels', as long as
    the longest series the model reads, the calendar or an estimate's.
    """

    spec: BayesGeoSpec
    geos: tuple[Geo, ...]
    spend: _Array
    lags: npt.NDArray[np.int64]
    y: _Array
    observed: npt.NDArray[np.bool_]
    weight: _Array
    calendar: _Array
    controls: _Array
    window_spend: _Array
    estimates: tuple[_Estimates, ...]
    length: int
    kpi_unit: float
    spend_unit: _Array
    population: _Array
    control_mean: _Array
    control_sd: _Array
    _compiled: Callable[[jax.Array], tuple[jax.Array, jax.Array]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_compiled", jax.jit(jax.value_and_grad(self._log_density)))

    @property
    def _sizes(self) -> dict[str, int]:
        m, g = len(self.spec.media), len(self.geos)
        return {
            "log_roi": m,
            "retention": m,
            "half": m,
            "slope": m if self._sampled_slope else 0,
            "spread": m,
            "deviation": (g - 1) * m,
            "base": g,
            "calendar": self.calendar.shape[1],
            "control": self.controls.shape[2],
            "sigma": 1,
            "nu": int(isinstance(self.spec.likelihood, StudentT)),
        }

    @property
    def dimension(self) -> int:
        return sum(self._sizes.values())

    @property
    def _sampled_slope(self) -> bool:
        return isinstance(self.spec.saturation, HillCurve) and self.spec.slope == "sampled"

    @property
    def _centred(self) -> npt.NDArray[np.bool_]:
        return np.array([m.channel in self.spec.centred for m in self.spec.media], dtype=bool)

    @property
    def coordinates(self) -> tuple[str, ...]:
        channels = [m.channel for m in self.spec.media]
        return (
            *(f"log_roi[{c}]" for c in channels),
            *(f"logit_retention[{c}]" for c in channels),
            *(f"log_half_saturation[{c}]" for c in channels),
            *(f"log_slope[{c}]" for c in channels if self._sizes["slope"]),
            *(f"log_spread[{c}]" for c in channels),
            *(f"deviation[{c},{k}]" for c in channels for k in range(1, len(self.geos))),
            *(
                f"log_base[{g}]" if self.spec.base_prior == "log-normal" else f"base[{g}]"
                for g in self.geos
            ),
            *(f"calendar[{t}]" for t in self.spec.one_geo().calendar_terms()),
            *(f"control[{c.column}]" for c in self.spec.controls),
            "log_sigma",
            *(("log_nu",) if self._sizes["nu"] else ()),
        )

    def log_density(self, theta: _Array) -> tuple[float, _Array]:
        value, gradient = self._compiled(jnp.asarray(theta, dtype=float))
        return float(value), np.asarray(gradient)

    def parameters(self, theta: _Array) -> dict[str, _Array]:
        """The parameters at each point of `theta`, `(points, dimension)`, in the panel's units."""
        raw, _, _ = _summary(self, np.atleast_2d(theta))
        return raw

    def log_likelihood(self, theta: _Array) -> _Array:
        """Each observation's log density in the panel's units at each point of `theta`, `(points,
        dimension)`: each fitted cell's, geo by geo, then each lift estimate's, in
        `BayesGeoFit.log_likelihood`'s order."""
        _, _, pointwise = _summary(self, np.atleast_2d(theta))
        return pointwise

    def _lift(self, theta: _Array) -> _Array:
        """`theta` itself: neither of the model's noises bounds the mean, so every point is in the
        support."""
        return theta

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
        spread = jnp.exp(u["spread"])
        deviation = u["deviation"].reshape(len(self.spec.media), -1)
        scale = jnp.where(self._centred, 1.0, spread)
        positive = self.spec.base_prior == "log-normal"
        out = {
            "log_roi": u["log_roi"],
            "retention": jax.nn.sigmoid(u["retention"]),
            "half": jnp.exp(u["half"]),
            "slope": jnp.exp(u["slope"]) if self._sampled_slope else jnp.ones(len(self.spec.media)),
            "spread": spread,
            "deviation": deviation,
            "effect": _helmert(len(self.geos)) @ (scale[:, None] * deviation).T,
            "base": jnp.exp(u["base"]) if positive else u["base"],
            "calendar": u["calendar"],
            "control": jnp.where(signs > 0.0, signed, u["control"]),
            "sigma": jnp.exp(u["sigma"]),
            "nu": jnp.exp(u["nu"]),
        }
        log_jacobian = (
            jnp.sum(jax.nn.log_sigmoid(u["retention"]) + jax.nn.log_sigmoid(-u["retention"]))
            + jnp.sum(u["half"])
            + jnp.sum(u["slope"])
            + jnp.sum(u["spread"])
            + (jnp.sum(u["base"]) if positive else 0.0)
            + jnp.sum(signs * u["control"])
            + jnp.sum(u["sigma"])
            + jnp.sum(u["nu"])
        )
        return out, log_jacobian

    def _weights(self, p: Mapping[str, jax.Array], j: int) -> jax.Array:
        """Channel `j`'s kernel, lag 0 first, `(L,)`."""
        return response.GeometricAdstock(
            p["retention"][j], length=self.length, normalized=True
        ).weights()

    def _curve(self, p: Mapping[str, jax.Array], j: int, adstock: jax.Array | _Array) -> jax.Array:
        """Channel `j`'s curve at `adstock`."""
        if isinstance(self.spec.saturation, TanhCurve):
            return response.Tanh(p["half"][j] / _TANH_HALF)(adstock)
        return response.Hill(p["half"][j], p["slope"][j])(adstock)

    def _saturated(self, p: Mapping[str, jax.Array], j: int) -> jax.Array:
        """Channel `j`'s spend in each geo at each window period, carried over and saturated,
        `(G, window)`."""
        weights = self._weights(p, j)
        kernel = jnp.where(self.lags >= 0, weights[np.maximum(self.lags, 0)], 0.0)
        return self._curve(p, j, self.spend[:, :, j] @ kernel.T)

    def _terms(self, p: Mapping[str, jax.Array]) -> tuple[jax.Array, jax.Array]:
        """The window's saturated spend, `(G, window, M)`, and each geo's channels' coefficients,
        `(G, M)`."""
        saturated = jnp.stack([self._saturated(p, j) for j in range(len(self.spec.media))], axis=2)
        factor = jnp.exp(p["effect"])
        reach = jnp.einsum("g,gm,gtm->m", self.weight, factor, saturated)
        return saturated, jnp.exp(p["log_roi"]) * self.window_spend / reach * factor

    def _mean(self, p: Mapping[str, jax.Array], saturated: jax.Array, beta: jax.Array) -> jax.Array:
        """The KPI's mean per head at each geo's window period, `(G, window)`."""
        base = p["base"][:, None] * (1.0 + self.calendar @ p["calendar"])
        return base + self.controls @ p["control"] + jnp.einsum("gtm,gm->gt", saturated, beta)

    def _log_density(self, theta: jax.Array) -> jax.Array:
        p, log_jacobian = self._constrained(theta)
        priors = [m.roi for m in self.spec.media if isinstance(m.roi, RoiPrior)]
        mu = np.array([r.mu for r in priors])
        sd = np.array([r.sigma for r in priors])
        lp = log_jacobian
        lp += jnp.sum(jstats.norm.logpdf(p["log_roi"], mu, sd))
        lp += jnp.sum(jstats.beta.logpdf(p["retention"], 1.0, 3.0))
        lp += jnp.sum(_log_normal(p["half"], 0.0, 1.0))
        if self._sampled_slope:
            curve = self.spec.saturation
            assert isinstance(curve, HillCurve)  # a sampled slope is a Hill's
            lp += jnp.sum(_log_normal(p["slope"], curve.slope_location, curve.slope_scale))
        lp += jnp.sum(jstats.norm.logpdf(p["spread"], 0.0, self.spec.spread))
        scale = jnp.where(self._centred, p["spread"], 1.0)
        lp += jnp.sum(jstats.norm.logpdf(p["deviation"], 0.0, scale[:, None]))
        if self.spec.base_prior == "log-normal":
            lp += jnp.sum(_log_normal(p["base"], 0.0, 1.0))
        else:
            lp += jnp.sum(jstats.norm.logpdf(p["base"], 1.0, 1.0))
        lp += jnp.sum(jstats.norm.logpdf(p["calendar"]))
        lp += jnp.sum(jstats.norm.logpdf(p["control"]))
        if self.spec.noise_prior == "half-normal":
            lp += jnp.sum(jstats.norm.logpdf(p["sigma"]))
        else:
            lp += jnp.sum(_log_normal(p["sigma"], *_NOISE_LOG_NORMAL))
        lp += jnp.sum(jstats.gamma.logpdf(p["nu"], 2.0, scale=10.0))
        return lp + jnp.sum(self._pointwise(p, *self._terms(p)))

    def _pointwise(
        self, p: Mapping[str, jax.Array], saturated: jax.Array, beta: jax.Array
    ) -> jax.Array:
        """Each observation's log density on the model's scales, each fitted cell's geo by geo and
        then each lift estimate's."""
        mean = self._mean(p, saturated, beta)[self.observed]
        scale = (
            p["sigma"][0]
            * np.broadcast_to(self.weight[:, None] ** -0.5, self.observed.shape)[self.observed]
        )
        if isinstance(self.spec.likelihood, StudentT):
            kpi = jstats.t.logpdf(self.y, p["nu"][0], mean, scale)
        else:
            kpi = jstats.norm.logpdf(self.y, mean, scale)
        lifts = [jstats.norm.logpdf(e.lift, self._lifted(p, beta, e), e.se) for e in self.estimates]
        return jnp.concatenate([kpi, *lifts])

    def _lifted(self, p: Mapping[str, jax.Array], beta: jax.Array, e: _Estimates) -> jax.Array:
        """Each of a channel's estimates' lift: the gap in saturated spend per head summed over
        its readout, times its geos' coefficients weighted by their populations."""
        weights = self._weights(p, e.channel)
        gap = self._curve(p, e.channel, e.spend @ weights) - self._curve(
            p, e.channel, e.planned @ weights
        )
        summed = jax.ops.segment_sum(gap, e.estimate, e.lift.size, indices_are_sorted=True)
        return summed * (e.tested @ beta[:, e.channel])


def _helmert(geos: int) -> _Array:
    """Helmert's contrasts over `geos` geos, `(geos, geos - 1)`: column `k` the first `k` geos'
    mean less the next geo's value, scaled to unit length, so the columns are orthonormal and each
    sums to 0."""
    basis = np.zeros((geos, geos - 1))
    for k in range(1, geos):
        basis[:k, k - 1] = 1.0 / math.sqrt(k * (k + 1))
        basis[k, k - 1] = -k / math.sqrt(k * (k + 1))
    return basis


def _estimates(
    channel: int,
    estimates: Sequence[LiftEstimate],
    geos: tuple[Geo, ...],
    length: int,
    population: _Array,
    spend_unit: float,
    kpi_unit: float,
) -> _Estimates:
    spend, planned, tested = [], [], []
    for e in estimates:
        where = np.array([e.geos is None or g in e.geos for g in geos])
        heads = e.share * population[where].sum()
        readout = np.zeros(length, dtype=bool)
        readout[length - e.readout :] = True
        for series, rows in ((e.spend, spend), (e.planned, planned)):
            scaled = series / heads / spend_unit
            rows.append(_lags(np.concatenate([np.zeros(length - scaled.size), scaled]), readout))
        tested.append(np.where(where, population / population.mean(), 0.0))
    scale = kpi_unit * population.mean()
    return _Estimates(
        channel,
        np.concatenate(spend),
        np.concatenate(planned),
        np.repeat(np.arange(len(estimates)), [e.readout for e in estimates]),
        np.array(tested),
        np.array([e.lift / e.share for e in estimates]) / scale,
        np.array([e.se / e.share for e in estimates]) / scale,
    )


def bayes_geo_model(
    panel: MediaPanel,
    spec: BayesGeoSpec,
    *,
    window: tuple[np.datetime64, np.datetime64] | None = None,
    estimates: Mapping[str, Sequence[LiftEstimate]] | None = None,
) -> BayesGeoModel:
    """The model of `spec` on every geo of `panel` over `window`, its first and last dates, by
    default the whole calendar, its likelihood holding `estimates`, each channel's lift estimates,
    if given.

    Raises:
        ValueError: on JAX in single precision; a panel of one geo, which has nothing to pool; a
            column the panel does not have, a window off its calendar, a channel that spends
            nothing over it in any geo, a geo with no KPI over it or a mean KPI per head not above
            0, and a control that does not vary over it; and estimates of a channel the model does
            not have, or of none, and estimates in a geo the panel does not have.
    """
    if not jax.config.x64_enabled:
        msg = "the model needs JAX in double precision (JAX_ENABLE_X64=1)"
        raise ValueError(msg)
    if len(panel.geos) < 2:
        msg = f"a model over geos needs two geos or more; the panel has {list(panel.geos)}"
        raise ValueError(msg)
    estimates = estimates or {}
    channels = {str(c): j for j, c in enumerate(panel.channels)}
    columns = {c: j for j, c in enumerate(panel.controls)}
    names = [m.channel for m in spec.media]
    missing = [c for c in names if c not in channels]
    missing += [c.column for c in spec.controls if c.column not in columns]
    if missing:
        msg = f"the panel has no channel or control named {missing}"
        raise ValueError(msg)
    refuse_levers(panel, [c.column for c in spec.controls], "the Bayesian model over geos")
    unknown = sorted(set(estimates) - set(names))
    empty = sorted(c for c, e in estimates.items() if not e)
    if unknown or empty:
        msg = f"estimates name channels the model does not have, {unknown}, or none, {empty}"
        raise ValueError(msg)
    named = {g for es in estimates.values() for e in es for g in e.geos or ()}
    nowhere = sorted(named - set(panel.geos))
    if nowhere:
        msg = f"estimates name geos {nowhere} that the panel does not have"
        raise ValueError(msg)
    inside = _window(panel, window)
    population = (
        np.ones(len(panel.geos)) if panel.population is None else np.asarray(panel.population)
    )
    spend = panel.spend[:, :, [channels[c] for c in names]] / population[:, None, None]
    window_spend = np.einsum("gtm,g->m", spend[:, inside], population)
    dark = [c for c, total in zip(names, window_spend, strict=True) if not total > 0.0]
    if dark:
        msg = f"channels {dark} spend nothing over the window"
        raise ValueError(msg)
    kpi = panel.kpi[:, inside] / population[:, None]
    observed = np.isfinite(kpi)
    blind = [str(g) for g, seen in zip(panel.geos, observed.any(axis=1), strict=True) if not seen]
    if blind:
        msg = f"geos {blind} have no KPI over the window"
        raise ValueError(msg)
    heads = np.broadcast_to(population[:, None], kpi.shape)[observed]
    kpi_unit = float(np.sum(kpi[observed] * heads) / np.sum(heads))
    if not kpi_unit > 0.0:
        msg = f"the window's mean KPI per head is {kpi_unit}"
        raise ValueError(msg)
    spend_unit = np.array([float(np.mean(s[s > 0.0])) for s in spend.reshape(-1, len(names)).T])
    values = panel.control_values[:, inside][:, :, [columns[c.column] for c in spec.controls]]
    seen = ~panel.filled[:, inside]
    control_mean, control_sd = values[seen].mean(axis=0), values[seen].std(axis=0)
    flat = [c.column for c, sd in zip(spec.controls, control_sd, strict=True) if not sd > 0.0]
    if flat:
        msg = f"controls {flat} do not vary over the window"
        raise ValueError(msg)
    length = max([panel.dates.size] + [e.spend.size for es in estimates.values() for e in es])
    return BayesGeoModel(
        spec=spec,
        geos=panel.geos,
        spend=spend / spend_unit,
        lags=np.flatnonzero(inside)[:, None] - np.arange(panel.dates.size),
        y=kpi[observed] / kpi_unit,
        observed=observed,
        weight=population / population.mean(),
        calendar=spec.one_geo().calendar(int(inside.sum())),
        controls=np.where(seen[:, :, None], (values - control_mean) / control_sd, 0.0),
        window_spend=window_spend / (kpi_unit * population.mean()),
        estimates=tuple(
            _estimates(
                names.index(c),
                estimates[c],
                panel.geos,
                length,
                population,
                spend_unit[names.index(c)],
                kpi_unit,
            )
            for c in names
            if c in estimates
        ),
        length=length,
        kpi_unit=kpi_unit,
        spend_unit=spend_unit,
        population=population,
        control_mean=control_mean,
        control_sd=control_sd,
    )


def _summary(
    model: BayesGeoModel, theta: _Array
) -> tuple[dict[str, _Array], dict[str, _Array], _Array]:
    """Each draw's parameters in the panel's units, `theta` being `(draws, dimension)`; the
    posterior mean, in totals, of each term's contribution at each geo's window period and of the
    fitted mean, `estimated`; and each draw's log density of each observation in the panel's
    units."""
    calendar = model.spec.one_geo().calendar_terms()
    raw_controls = jnp.asarray(model.controls * model.control_sd + model.control_mean)
    totals = model.population[:, None] * model.kpi_unit
    cells = np.broadcast_to(totals, model.observed.shape)[model.observed]
    lifts = np.full(
        sum(e.lift.size for e in model.estimates), model.kpi_unit * model.population.mean()
    )
    log_units = np.log(np.concatenate([cells, lifts]))

    def one(t: jax.Array) -> tuple[dict[str, jax.Array], dict[str, jax.Array], jax.Array]:
        p, _ = model._constrained(t)
        saturated, beta = model._terms(p)
        control = p["control"] / model.control_sd
        raw = {
            "roi": jnp.exp(p["log_roi"]),
            "retention": p["retention"],
            "half_saturation": p["half"] * model.spend_unit,
            "spread": p["spread"],
            "coefficient": beta * model.kpi_unit,
            "base": p["base"] * model.kpi_unit,
            "sigma": p["sigma"][0] * model.kpi_unit,
        }
        if model._sizes["calendar"]:
            raw["calendar"] = p["calendar"]
        if model._sizes["control"]:
            raw["control"] = control * model.kpi_unit
        if model._sizes["slope"]:
            raw["slope"] = p["slope"]
        if model._sizes["nu"]:
            raw["nu"] = p["nu"][0]
        parts = {
            **{
                m.channel: saturated[:, :, j] * beta[:, j, None]
                for j, m in enumerate(model.spec.media)
            },
            **{
                c.column: raw_controls[:, :, j] * control[j]
                for j, c in enumerate(model.spec.controls)
            },
            **{
                name: p["base"][:, None] * model.calendar[:, k] * p["calendar"][k]
                for k, name in enumerate(calendar)
            },
            "estimated": model._mean(p, saturated, beta),
        }
        scaled = {k: v * totals for k, v in parts.items()}
        return raw, scaled, model._pointwise(p, saturated, beta) - log_units

    batch = jax.jit(jax.vmap(one))
    count = theta.shape[0]
    padded = np.concatenate([theta, np.repeat(theta[-1:], -count % _BATCH, axis=0)])
    raws, pointwise, sums = [], [], {}
    for start in range(0, padded.shape[0], _BATCH):
        raw, parts, density = batch(jnp.asarray(padded[start : start + _BATCH]))
        keep = min(_BATCH, count - start)
        raws.append({k: np.asarray(v)[:keep] for k, v in raw.items()})
        pointwise.append(np.asarray(density)[:keep])
        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + np.asarray(v)[:keep].sum(axis=0)
    return (
        {k: np.concatenate([r[k] for r in raws]) for k in raws[0]},
        {k: v / count for k, v in sums.items()},
        np.concatenate(pointwise),
    )


def fit_bayes_geo(
    panel: MediaPanel,
    spec: BayesGeoSpec,
    *,
    seed: int,
    window: tuple[np.datetime64, np.datetime64] | None = None,
    estimates: Mapping[str, Sequence[LiftEstimate]] | None = None,
    chains: int = 4,
    warmup: int = 1000,
    draws: int = 1000,
    start: Start = _UNIFORM_START,
) -> BayesGeoFit:
    """Draw the posterior of `spec` on every geo of `panel` over `window`, its first and last
    dates, by default the whole calendar, its likelihood holding `estimates`, each channel's lift
    estimates, if given. Each of `chains` starts as `start` says, drawn from `seed`.

    Raises:
        ValueError: on what `bayes_geo_model` refuses, and on what the sampler, or Pathfinder,
            refuses.
    """
    model = bayes_geo_model(panel, spec, window=window, estimates=estimates)
    initial = _initial(model, start, chains, seed)
    result = compiled.sample(model._log_density, initial, seed=seed, warmup=warmup, draws=draws)
    raw, parts, pointwise = _summary(model, result.positions.reshape(-1, model.dimension))
    estimated = parts.pop("estimated")
    shaped = {k: v.reshape(chains, draws, *v.shape[1:]) for k, v in raw.items()}
    inside = _window(panel, window)
    return BayesGeoFit(
        spec=spec,
        dates=panel.dates[inside],
        geos=panel.geos,
        population=model.population,
        y=panel.kpi[:, inside],
        fitted=model.observed,
        draws=MappingProxyType(shaped),
        log_likelihood=pointwise.reshape(chains, draws, -1),
        contributions=MappingProxyType(parts),
        estimated=estimated,
        diagnostics=MappingProxyType(
            {k: diagnose(v.reshape(chains, draws, -1)) for k, v in shaped.items()}
        ),
        divergent=result.divergent.sum(axis=1),
        e_bfmi=e_bfmi(result.energy),
        length=model.length,
    )
