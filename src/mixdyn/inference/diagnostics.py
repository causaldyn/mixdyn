"""Convergence diagnostics for MCMC draws (Vehtari, Gelman, Simpson, Carpenter and Bürkner 2021).

Each chain is split in half, its middle draw dropped when the draws are odd in number, and the
halves are read as chains. The draws are rank-normalised: each replaced by the normal quantile of
its pooled rank, ties given their average rank, at ``(r - 3/8) / (S + 1/4)``.

* ``rhat``: the larger of the bulk R-hat, of the rank-normalised draws, and the tail R-hat, of the
  rank-normalised distances from the median, which a difference in the chains' scales moves where
  a difference in their locations does not.
* ``ess_bulk``: the effective sample size of the rank-normalised draws; ``ess_tail`` the smaller of
  those of the indicators of the draws at or below their 5 % quantile and at or above their 95 %,
  each quantile linear in the order statistics. Each from Geyer's initial monotone sequence over
  the chains' combined autocorrelations, its last even lag kept as a bias term that lowers the
  variance for antithetic chains, and capped at ``S log10 S``.
* ``e_bfmi``: the energy Bayesian fraction of missing information of a chain (Betancourt 2016),
  the mean squared change in the Hamiltonian between transitions over its variance. Below 0.3 the
  momentum resampling explores the energy too slowly.

A quantity whose draws are all one value has no R-hat or effective sample size: each is ``nan``.
Nor, as CmdStan has it, is there an effective sample size where the split chains hold fewer than
four draws, under eight a chain: Geyer's sequence has too few lags to read. The effective sample
size, which no scale moves, reads the draws scaled by a power of two to a range near 1: that
changes no bit but the exponents', and keeps the squared deviations of draws of a range of
``1e-200`` or ``1e200`` in the floats, which they would pass.
These are CmdStan 2.38's ``stansummary`` to rounding, but for one case: where a tail's indicator
is one value throughout a split chain, ``stansummary`` reads that indicator's effective sample size
as half the draws' number whatever the chains, and this module as any other series', far below
that on chains that disagree in location or scale.

A one-geo Bayesian fit's health card (`health`, ADR 0060) reads, beside the largest R-hat, the
posterior predictive p-value of the KPI's total, the probability of a negative baseline, and each
channel's return's shift from its prior and its place in the prior; `prior_predictive` draws the
media share and the returns a spec's priors imply. Each cell is a number, read against thresholds
the caller passes. `out_of_range` reads how far a plan's spend lies outside the range the history
logged, where any engine's curve is read past the spend that fitted it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from chc import response
from scipy import special, stats

if TYPE_CHECKING:
    from mixdyn.data import Geo, MediaPanel
    from mixdyn.models.bayes_mmm import BayesFit, BayesSpec

type _Array = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Diagnostics:
    """Per quantity, ``rhat``, ``ess_bulk`` and ``ess_tail``."""

    rhat: _Array
    ess_bulk: _Array
    ess_tail: _Array


def diagnose(draws: _Array) -> Diagnostics:
    """``draws`` is ``(chains, draws)`` for one quantity or ``(chains, draws, quantities)``.

    Raises:
        ValueError: on fewer than four draws a chain.
    """
    x = np.asarray(draws, dtype=float)
    x = x[:, :, None] if x.ndim == 2 else x
    if x.shape[1] < 4:
        msg = f"{x.shape[1]} draws a chain; the split chains need at least two each"
        raise ValueError(msg)
    quantities = x.shape[2]
    rhat, bulk, tail = (np.full(quantities, np.nan) for _ in range(3))
    for k in range(quantities):
        chains = _split(x[:, :, k])
        if np.ptp(chains) == 0.0:
            continue
        rhat[k] = max(
            _rhat(_rank_normal(chains)),
            _rhat(_rank_normal(np.abs(chains - np.median(chains)))),
        )
        bulk[k] = _ess(_rank_normal(chains))
        low, high = np.quantile(chains, (0.05, 0.95))
        tail[k] = min(_ess((chains <= low).astype(float)), _ess((chains >= high).astype(float)))
    return Diagnostics(rhat, bulk, tail)


def e_bfmi(energy: _Array) -> _Array:
    """Each chain's E-BFMI, ``energy`` being ``(chains, draws)``."""
    e = np.atleast_2d(np.asarray(energy, dtype=float))
    return np.sum(np.diff(e, axis=1) ** 2, axis=1) / np.sum(
        (e - e.mean(axis=1, keepdims=True)) ** 2, axis=1
    )


def _split(x: _Array) -> _Array:
    half = x.shape[1] // 2
    return np.concatenate([x[:, :half], x[:, x.shape[1] - half :]], axis=0)


def _rank_normal(x: _Array) -> _Array:
    ranks = stats.rankdata(x, method="average").reshape(x.shape)
    return special.ndtri((ranks - 0.375) / (x.size + 0.25))


def _rhat(x: _Array) -> float:
    _, n = x.shape
    within = x.var(axis=1, ddof=1).mean()
    between = n * x.mean(axis=1).var(ddof=1)
    if within == 0.0:
        return np.nan
    return float(np.sqrt(((n - 1.0) / n * within + between / n) / within))


def _autocovariance(x: _Array) -> _Array:
    """Each chain's biased autocovariance at lags ``0, ..., n - 1``."""
    n = x.shape[1]
    centred = x - x.mean(axis=1, keepdims=True)
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centred, n=size, axis=1)
    return np.fft.irfft(spectrum * np.conj(spectrum), n=size, axis=1)[:, :n] / n


def _ess(x: _Array) -> float:
    m, n = x.shape
    if n < 4 or np.ptp(x) == 0.0:
        return np.nan
    _, exponent = np.frexp(np.ptp(x))
    x = np.ldexp(x, -exponent)
    acov = _autocovariance(x)
    mean_var = acov[:, 0].mean() * n / (n - 1.0)
    var_plus = mean_var * (n - 1.0) / n + (x.mean(axis=1).var(ddof=1) if m > 1 else 0.0)
    rho = np.zeros(n)
    rho[0] = 1.0
    even, odd = 1.0, 1.0 - (mean_var - acov[:, 1].mean()) / var_plus
    rho[1] = odd
    t = 1
    while t < n - 4 and even + odd > 0.0:
        even = 1.0 - (mean_var - acov[:, t + 1].mean()) / var_plus
        odd = 1.0 - (mean_var - acov[:, t + 2].mean()) / var_plus
        if even + odd >= 0.0:
            rho[t + 1], rho[t + 2] = even, odd
        t += 2
    last = t
    if even > 0.0:
        rho[last + 1] = even
    for t in range(1, last - 1, 2):
        if rho[t + 1] + rho[t + 2] > rho[t - 1] + rho[t]:
            rho[t + 1] = rho[t + 2] = (rho[t - 1] + rho[t]) / 2.0
    total = m * n
    tau = -1.0 + 2.0 * rho[:last].sum() + rho[last + 1]
    return float(total / max(tau, 1.0 / np.log10(total)))


@dataclass(frozen=True, slots=True, eq=False)
class HealthCard:
    """The health of a Bayesian fit of one geo (`mixdyn.models.bayes_mmm`), each cell its number.

    * ``max_rhat``: the largest R-hat over every parameter of the fit, a constant's ``nan`` left
      out.
    * ``ppp``: the two-sided posterior predictive p-value of the KPI's total over the fitted
      periods, ``min(1, 2 min(P(T >= t), P(T <= t)))``, ``t`` the observed total and ``T`` a
      replicate's, one replicate at each of the posterior's draws, drawn about the draw's mean by
      the model's noise; ``ppp_se`` its Monte Carlo standard error from the draw count,
      ``2 sqrt(q (1 - q) / S)``, ``q`` the smaller tail's share of ``S`` replicates, which reads 0
      where no replicate is past the total: ``ppp`` is then below ``2 / S``. ``totals`` holds the
      replicates and ``total`` the observed total.
    * ``negative_baseline``: the share of draws whose baseline, the intercept, calendar and
      controls without the channels (the level less media, ADR 0036), is below 0 at some period of
      the window; ``negative_baseline_periods`` that share at each period.
    * per channel, of its return as ADR 0016 defines it, the window's spend's revenue with its
      carryover counted, read at draws of the prior and of the posterior: ``shift``, the posterior
      mean less the prior's over the prior's standard deviation; ``width``, the posterior's
      standard deviation over the prior's; and ``consistency``, the prior's quantile of the
      posterior median, the share of prior draws at or below it, near 0 or 1 where the posterior
      sits in the prior's tail.

    No cell is a verdict: ``flags`` reads each against a threshold the caller passes.
    """

    max_rhat: float
    ppp: float
    ppp_se: float
    total: float
    totals: _Array
    negative_baseline: float
    negative_baseline_periods: _Array
    shift: Mapping[str, float]
    width: Mapping[str, float]
    consistency: Mapping[str, float]

    def flags(
        self,
        *,
        max_rhat: float | None = None,
        min_ppp: float | None = None,
        max_negative_baseline: float | None = None,
        max_width: float | None = None,
        tail: float | None = None,
    ) -> dict[str, bool]:
        """Whether each cell passes the threshold given for it, keyed by the cell's name, a
        channel's cells as ``"width[tv]"`` and ``"consistency[tv]"``; a cell whose threshold is not
        given is not read. ``max_rhat``, ``max_negative_baseline`` and ``max_width`` pass a cell
        below them, ``min_ppp`` one at or above it, and ``tail`` a consistency inside
        ``[tail, 1 - tail]``.

        Meridian's model reviewer (Google, Meridian 2.x) reads its fits at a largest R-hat below
        1.2, a posterior predictive p-value of 0.05 or more and a probability of a negative
        baseline below 0.2: ``flags(max_rhat=1.2, min_ppp=0.05, max_negative_baseline=0.2)``
        reads this card at its thresholds, which are an example, not a default.
        """
        out: dict[str, bool] = {}
        if max_rhat is not None:
            out["rhat"] = bool(self.max_rhat < max_rhat)
        if min_ppp is not None:
            out["ppp"] = bool(self.ppp >= min_ppp)
        if max_negative_baseline is not None:
            out["negative_baseline"] = bool(self.negative_baseline < max_negative_baseline)
        if max_width is not None:
            out |= {f"width[{c}]": bool(w < max_width) for c, w in self.width.items()}
        if tail is not None:
            out |= {
                f"consistency[{c}]": bool(tail <= q <= 1.0 - tail)
                for c, q in self.consistency.items()
            }
        return out


@dataclass(frozen=True, slots=True, eq=False)
class PriorPredictive:
    """Draws of a model's prior over a window: ``share``, the channels' contribution over the
    window over the KPI's, `(draws,)`, the KPI's being its mean over the observed periods times
    the window's length; and ``roi``, each channel's return as ADR 0016 defines it, `(draws,)`."""

    share: _Array
    roi: Mapping[str, _Array]

    def quantiles(self, q: Sequence[float] = (0.05, 0.5, 0.95)) -> dict[str, _Array]:
        """The share's quantiles at ``q``, under ``"share"``, and each channel's return's."""
        levels = np.asarray(q, dtype=float)
        return {"share": np.quantile(self.share, levels)} | {
            c: np.quantile(r, levels) for c, r in self.roi.items()
        }


def posterior_predictive_p(totals: npt.ArrayLike, total: float) -> tuple[float, float]:
    """The two-sided tail probability of ``total`` among replicates ``totals``, and its Monte Carlo
    standard error; see `HealthCard`."""
    t = np.asarray(totals, dtype=float).reshape(-1)
    q = min(float(np.mean(t >= total)), float(np.mean(t <= total)))
    return min(1.0, 2.0 * q), 2.0 * math.sqrt(q * (1.0 - q) / t.size)


def negative_baseline(baseline: npt.ArrayLike) -> tuple[float, _Array]:
    """Of baselines ``(draws, periods)``, the share of draws below 0 at some period, and the share
    at each period."""
    below = np.asarray(baseline, dtype=float) < 0.0
    return float(below.any(axis=1).mean()), below.mean(axis=0)


def prior_shift(prior: npt.ArrayLike, posterior: npt.ArrayLike) -> tuple[float, float, float]:
    """Of one quantity's prior and posterior draws, the standardised shift of its mean, the ratio
    of its standard deviations and the prior's quantile of its posterior median; see
    `HealthCard`."""
    a = np.asarray(prior, dtype=float).reshape(-1)
    b = np.asarray(posterior, dtype=float).reshape(-1)
    sd = a.std(ddof=1)
    return (
        float((b.mean() - a.mean()) / sd),
        float(b.std(ddof=1) / sd),
        float(np.mean(a <= np.median(b))),
    )


def out_of_range(history: npt.ArrayLike, plan: npt.ArrayLike) -> tuple[float, _Array]:
    """How far a plan's spend, ``plan`` a spend a channel a period, `(periods, M)`, or a spend a
    channel, `(M,)`, lies outside the range ``history``, `(periods, M)`, logged, where the plan
    reads the channel's curve on spend that did not fit it. Each side counts its distance the same
    way: past a channel's most spend a period, the spend less that most; short of its least, that
    least less the spend. It returns the distances summed over the plan's periods and channels
    and divided by its whole spend, and each channel's part of that share, which sum to it. A plan
    inside every range reads 0, and one that doubles a channel's most logged spend half of that
    channel's spend. The share passes 1 where a plan spends further below the least logged levels
    than it spends.

    Raises:
        ValueError: on a history or a plan not finite and at or above 0, a history of no period,
            the two over different channels, and a plan that spends nothing.
    """
    logged = np.asarray(history, dtype=float)
    spend = np.asarray(plan, dtype=float)
    spend = spend[None] if spend.ndim == 1 else spend
    if logged.ndim != 2 or logged.shape[0] == 0 or spend.ndim != 2:
        msg = f"need a history (periods, M) and a plan (M,) or (periods, M), got {logged.shape}"
        raise ValueError(msg)
    if spend.shape[1] != logged.shape[1]:
        msg = f"the history has {logged.shape[1]} channels and the plan {spend.shape[1]}"
        raise ValueError(msg)
    for name, x in (("history", logged), ("plan", spend)):
        if not (np.isfinite(x).all() and (x >= 0.0).all()):
            msg = f"the {name}'s spend must be finite and at or above 0"
            raise ValueError(msg)
    total = float(spend.sum())
    if not total > 0.0:
        msg = "the plan spends nothing: no share of it is outside"
        raise ValueError(msg)
    above = np.maximum(spend - logged.max(axis=0), 0.0)
    below = np.maximum(logged.min(axis=0) - spend, 0.0)
    outside = above + below
    channels = outside.sum(axis=0) / total
    return float(channels.sum()), channels


def health(
    fit: BayesFit, panel: MediaPanel, geo: Geo, *, seed: int, prior_draws: int = 4000
) -> HealthCard:
    """The health card of ``fit``, fitted to ``geo`` of ``panel``: the posterior predictive at
    every draw of the fit, its noise drawn from ``seed``, and the returns' prior at
    ``prior_draws`` draws from a child of it.

    Raises:
        ValueError: on a geo the panel does not have, a fit whose window is not a run of the
            panel's dates, and fewer than two prior draws.
    """
    if prior_draws < 2:
        msg = f"need at least two prior draws, got {prior_draws}"
        raise ValueError(msg)
    data = _read(panel, geo, fit.spec, _fit_window(panel, fit))
    count = fit.draws["roi"].shape[0] * fit.draws["roi"].shape[1]
    at = {k: v.reshape(count, *v.shape[2:]) for k, v in fit.draws.items()}
    baseline = _baseline(fit, data, at)
    media, roi = _media(fit.spec, data, fit.length, at)
    mean = baseline + media.sum(axis=0)
    noise, prior = np.random.default_rng(seed).spawn(2)
    replicate = _replicate(fit.spec, mean[:, fit.fitted], at, noise)
    total = float(fit.y[fit.fitted].sum())
    ppp, se = posterior_predictive_p(replicate.sum(axis=1), total)
    anywhere, periods = negative_baseline(baseline)
    _, prior_roi = _prior(fit.spec, data, fit.length, prior_draws, prior)
    cells = {m.channel: prior_shift(prior_roi[j], roi[j]) for j, m in enumerate(fit.spec.media)}
    rhat = [d.rhat[np.isfinite(d.rhat)] for d in fit.diagnostics.values()]
    return HealthCard(
        max_rhat=float(np.concatenate(rhat).max(initial=-np.inf)),
        ppp=ppp,
        ppp_se=se,
        total=total,
        totals=replicate.sum(axis=1),
        negative_baseline=anywhere,
        negative_baseline_periods=periods,
        shift=MappingProxyType({c: v[0] for c, v in cells.items()}),
        width=MappingProxyType({c: v[1] for c, v in cells.items()}),
        consistency=MappingProxyType({c: v[2] for c, v in cells.items()}),
    )


def prior_predictive(
    panel: MediaPanel,
    geo: Geo,
    spec: BayesSpec,
    *,
    draws: int,
    seed: int,
    window: tuple[np.datetime64, np.datetime64] | None = None,
) -> PriorPredictive:
    """``draws`` draws of the prior of ``spec`` on ``geo`` of ``panel`` over ``window``, by
    default the whole calendar, from ``seed``: each channel's return, its decay, its curve's
    half-saturation point and slope drawn from the priors `mixdyn.models.bayes_mmm` states, its
    coefficient set by the return as the model sets it. The prior is the stated one, not its
    truncation to log-normal noise's support (ADR 0017).

    Raises:
        ValueError: on a geo the panel does not have, a window off its calendar, a channel that
            spends nothing over it, a window with no observed KPI, and fewer than two draws.
    """
    from mixdyn.models.ridge_mmm import _window  # the models import this module

    if draws < 2:
        msg = f"need at least two prior draws, got {draws}"
        raise ValueError(msg)
    data = _read(panel, geo, spec, _window(panel, window))
    share, roi = _prior(spec, data, panel.dates.size, draws, np.random.default_rng(seed))
    return PriorPredictive(
        share, MappingProxyType({m.channel: roi[j] for j, m in enumerate(spec.media)})
    )


@dataclass(frozen=True, slots=True, eq=False)
class _Window:
    """What the health card reads of one geo of a panel over a window: each channel's spend from
    the panel's first period to the window's last, `(M, through)`; its mean positive spend over
    the calendar; its spend over the window; the window's controls, `(window, C)`; its length; and
    the KPI's mean over its observed periods."""

    spend: _Array
    spend_unit: _Array
    window_spend: _Array
    controls: _Array
    periods: int
    kpi_unit: float


def _fit_window(panel: MediaPanel, fit: BayesFit) -> npt.NDArray[np.bool_]:
    inside = np.isin(panel.dates, fit.dates)
    where = np.flatnonzero(inside)
    if where.size != fit.dates.size or where.size == 0 or np.ptp(where) + 1 != where.size:
        msg = "the fit's window is not a run of the panel's dates"
        raise ValueError(msg)
    return inside


def _read(panel: MediaPanel, geo: Geo, spec: BayesSpec, inside: npt.NDArray[np.bool_]) -> _Window:
    if geo not in panel.geos:
        msg = f"the panel has no geo {geo!r}; it has {list(panel.geos)}"
        raise ValueError(msg)
    g = panel.geos.index(geo)
    channels = [str(c) for c in panel.channels]
    spend = panel.spend[g][:, [channels.index(m.channel) for m in spec.media]].T
    through = int(np.flatnonzero(inside)[-1]) + 1
    window_spend = spend[:, inside].sum(axis=1)
    if not np.all(window_spend > 0.0):
        msg = "every channel must spend over the window"
        raise ValueError(msg)
    kpi = panel.kpi[g, inside]
    if not np.isfinite(kpi).any():
        msg = "the window has no observed KPI"
        raise ValueError(msg)
    return _Window(
        spend=spend[:, :through],
        spend_unit=np.array([float(np.mean(s[s > 0.0])) for s in spend]),
        window_spend=window_spend,
        controls=panel.regressors([c.column for c in spec.controls], geo, inside),
        periods=int(inside.sum()),
        kpi_unit=float(np.nanmean(kpi)),
    )


def _baseline(fit: BayesFit, data: _Window, at: Mapping[str, _Array]) -> _Array:
    """Each draw's intercept, calendar and controls at each period of the window, `(draws,
    window)`."""
    out = np.repeat(at["intercept"][:, None], data.periods, axis=1)
    if "calendar" in at:
        out += at["calendar"] @ fit.spec.calendar(data.periods).T
    if "control" in at:
        out += at["control"] @ data.controls.T
    return out


def _channel(
    spec: BayesSpec, length: int
) -> Callable[[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]:
    """A channel's contribution at each period of a spend series, at one draw of its decay,
    half-saturation point, slope and coefficient, as `BayesFit` builds the channel."""
    from mixdyn.models.bayes_mmm import _TANH_HALF, TanhCurve

    tanh = isinstance(spec.saturation, TanhCurve)

    def contribution(
        series: jax.Array, retention: jax.Array, half: jax.Array, slope: jax.Array, coef: jax.Array
    ) -> jax.Array:
        kernel = response.GeometricAdstock(retention, length=length, normalized=True)
        curve = response.Tanh(half / _TANH_HALF) if tanh else response.Hill(half, slope)
        return response.Channel(kernel, curve, coef)(series)

    return contribution


def _media(
    spec: BayesSpec, data: _Window, length: int, at: Mapping[str, _Array]
) -> tuple[_Array, _Array]:
    """Each channel's contribution at each draw and period of the window, `(M, draws, window)`,
    and its return as ADR 0016 defines it at each draw, `(M, draws)`."""
    from mixdyn.models.bayes_mmm import TanhCurve

    draws = at["roi"].shape[0]
    slope = np.ones_like(at["retention"]) if isinstance(spec.saturation, TanhCurve) else at["slope"]
    return _read_channels(
        spec, data, length, at["retention"], at["half_saturation"], slope, at["coefficient"], draws
    )


def _read_channels(
    spec: BayesSpec,
    data: _Window,
    length: int,
    retention: _Array,
    half: _Array,
    slope: _Array,
    coefficient: _Array,
    draws: int,
) -> tuple[_Array, _Array]:
    """`_media` at draws of each channel's parameters, each `(draws, M)` in the panel's units."""
    contribution = _channel(spec, length)
    media, roi = (
        np.empty((len(spec.media), draws, data.periods)),
        np.empty((len(spec.media), draws)),
    )
    for j in range(len(spec.media)):
        series = data.spend[j]
        extended = jnp.asarray(np.concatenate([series, np.zeros(length - 1)]))
        inside = slice(series.size - data.periods, series.size)
        removed = extended.at[inside].set(0.0)

        def read(
            r: jax.Array,
            k: jax.Array,
            n: jax.Array,
            c: jax.Array,
            extended: jax.Array = extended,
            removed: jax.Array = removed,
            inside: slice = inside,
        ) -> tuple[jax.Array, jax.Array]:
            whole = contribution(extended, r, k, n, c)
            gain = jnp.sum(whole - contribution(removed, r, k, n, c))
            return whole[inside], gain

        window, gain = jax.vmap(read)(
            jnp.asarray(retention[:, j]),
            jnp.asarray(half[:, j]),
            jnp.asarray(slope[:, j]),
            jnp.asarray(coefficient[:, j]),
        )
        media[j] = np.asarray(window)
        roi[j] = np.asarray(gain) / data.window_spend[j]
    return media, roi


def _replicate(
    spec: BayesSpec, mean: _Array, at: Mapping[str, _Array], rng: np.random.Generator
) -> _Array:
    """The KPI drawn about each draw's ``mean``, `(draws, periods)`, by the model's noise at that
    draw."""
    from mixdyn.models.bayes_mmm import LogNormal, StudentT

    sigma = at["sigma"][:, None]
    if isinstance(spec.likelihood, LogNormal):
        return mean * np.exp(sigma * rng.standard_normal(mean.shape) - sigma**2 / 2.0)
    if isinstance(spec.likelihood, StudentT):
        return mean + sigma * rng.standard_t(at["nu"][:, None], mean.shape)
    return mean + sigma * rng.standard_normal(mean.shape)


def _prior(
    spec: BayesSpec, data: _Window, length: int, draws: int, rng: np.random.Generator
) -> tuple[_Array, _Array]:
    """``draws`` draws of the prior of ``spec``'s channels: the channels' contribution over the
    window over the KPI's, `(draws,)`, and each channel's return as ADR 0016 defines it, `(M,
    draws)`. A channel's coefficient is its return times its spend over the window over the
    saturated spend that earns it, as `BayesModel` sets it."""
    from mixdyn.models.bayes_mmm import RoiPrior, TanhCurve

    m = len(spec.media)
    priors = [c.roi for c in spec.media]
    log = np.array([isinstance(r, RoiPrior) for r in priors])
    mu = np.array([r.mu if isinstance(r, RoiPrior) else 0.0 for r in priors])
    sd = np.array([r.sigma if isinstance(r, RoiPrior) else r.scale for r in priors])
    drawn = rng.normal(mu, sd, (draws, m))
    if log.all():
        stated = np.exp(drawn)
    else:
        stated = np.where(log, np.exp(np.where(log, drawn, 0.0)), np.abs(drawn))
    retention = rng.beta(1.0, 3.0, (draws, m))
    half = np.exp(rng.normal(0.0, 1.0, (draws, m))) * data.spend_unit
    curve = spec.saturation
    slope = (
        np.ones((draws, m))
        if isinstance(curve, TanhCurve)
        else np.exp(rng.normal(curve.slope_location, curve.slope_scale, (draws, m)))
    )
    window, carried = _read_channels(
        spec, data, length, retention, half, slope, np.ones((draws, m)), draws
    )
    earning = np.where(
        np.array([c.roi.counts == "carried" for c in spec.media])[:, None],
        carried * data.window_spend[:, None],
        window.sum(axis=2),
    )
    coefficient = (stated * data.window_spend).T / earning
    contribution = (coefficient * window.sum(axis=2)).sum(axis=0)
    return contribution / (data.kpi_unit * data.periods), coefficient * carried
