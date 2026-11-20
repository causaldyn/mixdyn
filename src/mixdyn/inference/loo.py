"""Leave-one-out cross-validation by Pareto-smoothed importance sampling (Vehtari, Gelman and Gabry
2017; Vehtari, Simpson, Gelman, Yao and Gabry 2024), and the stacking of models' predictive
distributions on it (Yao, Vehtari, Simpson and Gelman 2018).

**PSIS-LOO.** The posterior's draws stand in for the posterior without observation ``i`` through
the importance ratios ``1 / p(y_i | theta_s)``. The largest ``M`` of the ``S`` ratios, ``M`` the
smaller of ``S / 5`` and ``3 sqrt(S / r_eff)`` rounded up, ``r_eff`` the effective sample size of
``p(y_i | theta)`` over the split chains as a share of ``S``, are replaced, in their order, by the
quantiles at ``(z - 1/2) / M``, ``z = 1, ..., M``, of a generalised Pareto fitted to their excess
over the next largest ratio, and no smoothed ratio exceeds the largest raw one; a tail all of one
value, or whose first quartile is the next largest ratio's, is left as it is, its shape infinite,
as R's ``loo`` leaves it, and so is one whose first quartile lies so near it that the fit's grid
passes the floats. The fit is Zhang and Stephens' (2009) profile empirical Bayes estimate on
``30 + floor(sqrt M)`` grid points, its shape ``k`` shrunk toward 1/2 as by ten prior
observations. The quantiles are read in logs: a tail spanning hundreds of orders of magnitude fits
a shape in the hundreds, at which ``expm1(k t)``, ``t`` a standard exponential's quantile, and at
times the quantile itself, pass the floats, and their logs do not. ``k`` is the diagnostic: above
``min(1 - 1 / log10 S, 0.7)`` the estimate of ``p(y_i | y_-i)`` is not to be trusted.

**The smoothing alone**, :func:`psis`, weights any draws by their log importance ratios, and
reports Kish's effective sample size of the smoothed weights beside the shape. The shape is the
wrong alarm for a bounded ratio, a normal likelihood's: its tail fits a negative shape while the
effective sample falls to a few per cent of the draws.

**Stacking.** The weights on the simplex that maximise ``sum_i log sum_k w_k p_k(y_i | y_-i)``. The
problem is a log-optimal portfolio's, and Cover's (1984) multiplicative iteration solves it: it
stops once its certificate puts the mean of the observations' logs within ``1e-12`` of its optimum.
Two models' problem is one-dimensional and concave, and `mixdyn.decisions.stacked` solves it by
bisection on the objective's slope instead, which ends where Cover's certificate holds and always
ends.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import special

from mixdyn.inference.diagnostics import _ess, _split

type _Array = npt.NDArray[np.float64]

_GAP = 1e-12
_ITERATIONS = 1_000_000


@dataclass(frozen=True, slots=True)
class Loo:
    """``pointwise`` is each observation's log predictive density left out, ``elpd`` their sum and
    ``se`` its standard error; ``p_loo`` the log predictive density in sample less ``elpd``, the
    effective number of parameters; ``pareto_k`` each observation's fitted shape, and
    ``k_threshold`` the shape above which its estimate is not to be trusted."""

    elpd: float
    se: float
    p_loo: float
    pointwise: _Array
    pareto_k: _Array
    k_threshold: float


def loo(log_likelihood: _Array) -> Loo:
    """``log_likelihood`` is each draw's log density of each observation, ``(chains, draws,
    observations)``.

    Raises:
        ValueError: on another shape, fewer than four draws a chain, or a log density not finite.
    """
    ll = np.asarray(log_likelihood, dtype=float)
    if ll.ndim != 3 or ll.shape[1] < 4 or ll.shape[2] < 1:
        msg = f"need (chains, draws >= 4, observations >= 1) log densities, got {ll.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(ll)):
        msg = f"{np.sum(~np.isfinite(ll))} log densities are not finite"
        raise ValueError(msg)
    chains, draws, n = ll.shape
    s = chains * draws
    flat = ll.reshape(s, n)
    pointwise, k = np.empty(n), np.empty(n)
    for i in range(n):
        efficiency = _ess(_split(np.exp(ll[:, :, i] - ll[:, :, i].max()))) / s
        smoothed = psis(-flat[:, i], efficiency=efficiency if efficiency > 0.0 else 1.0)
        k[i] = smoothed.pareto_k
        pointwise[i] = special.logsumexp(smoothed.log_weights + flat[:, i])
    in_sample = special.logsumexp(flat, axis=0) - math.log(s)
    return Loo(
        elpd=float(pointwise.sum()),
        se=math.sqrt(n * pointwise.var()),
        p_loo=float((in_sample - pointwise).sum()),
        pointwise=pointwise,
        pareto_k=k,
        k_threshold=_threshold(s),
    )


def stacking(pointwise: _Array) -> _Array:
    """The stacking weights of models, ``pointwise`` each one's log predictive density of each
    observation left out, ``(models, observations)``.

    Raises:
        ValueError: on another shape or a log density not finite.
    """
    lpd = np.asarray(pointwise, dtype=float)
    if lpd.ndim != 2 or 0 in lpd.shape:
        msg = f"need (models, observations) log densities, got {lpd.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(lpd)):
        msg = f"{np.sum(~np.isfinite(lpd))} log densities are not finite"
        raise ValueError(msg)
    density = np.exp(lpd - lpd.max(axis=0))
    weights = np.full(lpd.shape[0], 1.0 / lpd.shape[0])
    for _ in range(_ITERATIONS):
        ratio = (density / (weights @ density)).mean(axis=1)
        if ratio.max() - 1.0 <= _GAP:
            return weights
        weights = weights * ratio
        weights /= weights.sum()
    msg = f"the weights did not converge in {_ITERATIONS} iterations"
    raise RuntimeError(msg)


def _stacking_of_two(pointwise: _Array) -> _Array:
    """Two models' stacking weights, ``pointwise`` their finite ``(2, observations)`` log densities
    as :func:`stacking` takes them.

    The objective is concave in the first model's weight ``w``: its slope, ``mean_i (a_i - b_i) /
    (w a_i + (1 - w) b_i)``, ``a`` and ``b`` the two models' densities, falls as ``w`` rises. The
    weight is the end of ``[0, 1]`` whose slope does not point inward, or else the slope's root,
    which bisection closes to two adjacent floats, keeping the one below it. Where neither end's
    slope points inward, the slope is 0 throughout and the densities are equal at every
    observation: every weight is optimal, and the models share it evenly, as :func:`stacking`
    leaves them.
    """
    a, b = np.exp(pointwise - pointwise.max(axis=0))

    def slope(w: float) -> float:
        # each observation's larger density is 1, so a density that underflows to 0 at an end
        # makes the slope there infinite, never NaN
        with np.errstate(divide="ignore"):
            return float(np.mean((a - b) / (w * a + (1.0 - w) * b)))

    top, bottom = slope(1.0), slope(0.0)
    if top >= 0.0 and bottom <= 0.0:
        return np.array([0.5, 0.5])
    if top >= 0.0:
        return np.array([1.0, 0.0])
    if bottom <= 0.0:
        return np.array([0.0, 1.0])
    low, high = 0.0, 1.0
    middle = 0.5
    while low < middle < high:
        if slope(middle) > 0.0:
            low = middle
        else:
            high = middle
        middle = 0.5 * (low + high)
    return np.array([low, 1.0 - low])


@dataclass(frozen=True, slots=True)
class Psis:
    """Pareto-smoothed importance weights: ``log_weights`` normalised, ``pareto_k`` the fitted
    shape of the ratios' tail, infinite where there is none to fit, ``k_threshold`` the shape
    above which the weights are not to be trusted, and ``ess`` Kish's effective sample size of the
    weights, ``1 / sum w^2``, times the draws' relative efficiency."""

    log_weights: _Array
    pareto_k: float
    k_threshold: float
    ess: float


def psis(log_ratios: npt.ArrayLike, *, efficiency: float = 1.0) -> Psis:
    """Pareto-smoothed importance sampling of ``(S,)`` log importance ratios; ``efficiency`` is the
    draws' relative efficiency ``r_eff``, which sets the tail's length and scales the effective
    sample size.

    Raises:
        ValueError: on another shape, no ratio, a ratio not finite, or an efficiency not positive
            and finite.
    """
    ratios = np.asarray(log_ratios, dtype=float)
    if ratios.ndim != 1 or ratios.size < 1:
        msg = f"need (draws >= 1,) log ratios, got {ratios.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(ratios)):
        msg = f"{np.sum(~np.isfinite(ratios))} log ratios are not finite"
        raise ValueError(msg)
    if not (math.isfinite(efficiency) and efficiency > 0.0):
        msg = f"efficiency={efficiency!r} is not positive and finite"
        raise ValueError(msg)
    s = ratios.size
    lw = ratios - ratios.max()
    tail = math.ceil(min(0.2 * s, 3.0 * math.sqrt(s / efficiency)))
    k = math.inf
    if tail >= 5:
        order = np.argsort(lw, kind="stable")
        largest = order[s - tail :]
        log_cutoff = lw[order[s - tail - 1]]
        if np.ptp(lw[largest]) > 0.0:
            k, sigma = _generalised_pareto(np.exp(lw[largest]) - math.exp(log_cutoff))
            if math.isfinite(k):
                p = (np.arange(1, tail + 1) - 0.5) / tail
                lw[largest] = np.logaddexp(_log_quantile(p, k, sigma), log_cutoff)
    lw = np.minimum(lw, 0.0)
    log_weights = lw - special.logsumexp(lw)
    return Psis(
        log_weights=log_weights,
        pareto_k=k,
        k_threshold=_threshold(s),
        ess=efficiency / float(np.sum(np.exp(2.0 * log_weights))),
    )


def _threshold(draws: int) -> float:
    """The shape above which ``draws`` weighted draws are not to be trusted; with one draw, none."""
    return min(1.0 - 1.0 / math.log10(draws), 0.7) if draws > 1 else -math.inf


def _generalised_pareto(x: _Array) -> tuple[float, float]:
    """Zhang and Stephens' estimate of a generalised Pareto's shape and scale from ``x``, a sample
    in increasing order, the shape shrunk toward 1/2 as by ten prior observations; its grid is
    scaled by the first quartile, and with that at 0, or so near it that the grid's widest point,
    ``(sqrt(2 m) - 1) / 3`` over it, passes the floats, there is no estimate."""
    n = x.size
    m = 30 + math.floor(math.sqrt(n))
    quartile = x[math.floor(n / 4 + 0.5) - 1]
    if quartile <= (math.sqrt(2.0 * m) - 1.0) / 3.0 / sys.float_info.max:
        return math.inf, math.nan
    theta = 1.0 / x[-1] + (1.0 - np.sqrt(m / (np.arange(1, m + 1) - 0.5))) / (3.0 * quartile)
    shape = np.log1p(-theta[:, None] * x).mean(axis=1)
    profile = n * (np.log(-theta / shape) - shape - 1.0)
    estimate = float(np.sum(theta * np.exp(profile - special.logsumexp(profile))))
    k = float(np.log1p(-estimate * x).mean())
    sigma = -k / estimate
    k = (n * k + 10 * 0.5) / (n + 10)
    return (k, sigma) if math.isfinite(k) else (math.inf, sigma)


def _log_quantile(p: _Array, k: float, sigma: float) -> _Array:
    """The logs of a generalised Pareto's quantiles at ``p``, ``sigma expm1(k t) / k``, ``t`` the
    standard exponential's quantiles, ``-log1p(-p)``: for a positive shape, ``log expm1(k t)`` as
    ``k t + log(-expm1(-k t))``, which holds where ``expm1(k t)`` passes the floats."""
    t = -np.log1p(-p)
    if k == 0.0:
        return math.log(sigma) + np.log(t)
    if k > 0.0:
        return math.log(sigma) - math.log(k) + k * t + np.log(-np.expm1(-k * t))
    return math.log(sigma) - math.log(-k) + np.log(-np.expm1(k * t))
