"""Lift estimates in the dynamic linear model: each test an observation of its channel's coefficient
path over its readout.

A test of a channel (`mixdyn.calibration.LiftEstimate`) reads the KPI its spend moved over the
readout. In the DLM a channel's response at a period is its coefficient there times its column, the
spend carried over and saturated by the effect's transform. So the test's lift at the market's scale
is

    lift / share = sum over the readout of b_t [g(spend / share)_t - g(planned / share)_t] + e,

`b_t` the coefficient, `g` the column, `e ~ N(0, (se / share)^2)`: the static model's reading
(ADR 0015) with the coefficient at each period in place of one coefficient. Given a spec the
transform is fixed, so the reading is linear in the state path and nothing is linearised. The
transform's nonlinearity is read where the spec is chosen: each spec's likelihood holds its tests.

The discounts set each step's evolution variance through the KPI's filter, in units of the
observational variance `V` and from the design alone. With them the model is a Gaussian prior over
paths, as `chc.dlm.constrained_sample` reads it, and the tests are further observations of it. Their
reading is exact for that model: the KPI's smoothed path conditioned on the tests, with the tests'
covariance with every state carried by the smoother's gains. It is the posterior a cumulator would
give, a state summing each test's weights times the coefficient over its readout and observed once
at its end, in a filter whose evolution variances are the KPI filter's.

The one approximation is the variance. A test's error is the analysis's, in the KPI's units, while
the model's variance `V` is learned. A test enters at the variance `se^2 / S_T` times `V`, `S_T`
the KPI filter's estimate at the last period: at `V = S_T` that is its own error, and the model
stays Normal-Gamma, so every reading of a fit, its Student t at the last period and its draws, is
the conjugate one. With the variance known the reading is exact.

Tests are read jointly, each error apart from the others', so tests whose readouts overlap are each
an observation. A readout outside the panel's calendar, an estimate with no end or in another geo,
and a test of a control are refused.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from chc.dlm import (
    ConstrainedDraws,
    DLMFit,
    PosteriorDraws,
    SmoothedStates,
    _convergence,
    _path_covariance,
    _reflected_orbit,
    _structure,
    _transitions,
    _unit_filter,
    backward_sample,
)
from scipy import linalg, special

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from chc.dlm import Sign

    from mixdyn.calibration import LiftEstimate
    from mixdyn.data import Geo, MediaPanel
    from mixdyn.models.dlm import DlmSpec

type _Array = npt.NDArray[np.float64]

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, eq=False)
class LiftReading:
    """The tests a fit reads, each an observation of one coordinate of the state path: test `j`
    reads coordinate `coordinates[j]` at period `t` with weight `weights[j, t]`, zero outside its
    readout, and its lift `lift[j]` has standard error `se[j]`, both at the market's scale.
    `channels` names each test's channel."""

    channels: tuple[str, ...]
    coordinates: npt.NDArray[np.intp]  # (K,)
    weights: _Array  # (K, T)
    lift: _Array  # (K,)
    se: _Array  # (K,)


def lift_reading(
    panel: MediaPanel,
    geo: Geo,
    spec: DlmSpec,
    estimates: Mapping[str, Sequence[LiftEstimate]],
) -> LiftReading:
    """`estimates`, each channel's tests, as observations of `spec`'s state path on `geo`'s
    calendar: each test's weights are its channel's column at the tested spend less at the planned,
    each over its share, over the readout's periods, which end at its `end`.

    Raises:
        ValueError: on no estimate, a channel that is not one of `spec`'s effects or is a control,
            an estimate with no end, in a geo other than `geo`, or whose readout runs outside the
            panel's calendar.
    """
    effects = {e.column: (i, e) for i, e in enumerate(spec.effects)}
    unknown = sorted(c for c in estimates if c not in effects)
    empty = sorted(c for c, given in estimates.items() if not given)
    controls = sorted(c for c in estimates if c in panel.controls)
    if unknown or empty or controls or not estimates:
        msg = (
            f"estimates name channels the spec has no effect for, {unknown}, controls, {controls},"
            f" or none, {empty or ('every channel' if not estimates else [])}"
        )
        raise ValueError(msg)
    first = spec.trend.order if spec.trend else 0
    periods = panel.dates.size
    channels, coordinates, rows, lifts, errors = [], [], [], [], []
    for column, given in estimates.items():
        i, effect = effects[column]
        for estimate in given:
            if estimate.end is None:
                msg = (
                    f"an estimate of {column!r} has no end, and the DLM reads a test at the "
                    "periods of its readout"
                )
                raise ValueError(msg)
            if estimate.geos is not None and estimate.geos != {geo}:
                msg = f"an estimate of {column!r} ran in {sorted(estimate.geos)}, not in {geo!r}"
                raise ValueError(msg)
            at = np.flatnonzero(panel.dates == estimate.end)
            if at.size != 1 or at[0] + 1 < estimate.readout:
                msg = (
                    f"an estimate of {column!r} reads {estimate.readout} periods ending "
                    f"{estimate.end}, outside the panel's calendar from {panel.dates[0]} to "
                    f"{panel.dates[-1]}"
                )
                raise ValueError(msg)
            end = int(at[0]) + 1
            tested, planned = estimate.spend / estimate.share, estimate.planned / estimate.share
            if effect.transform is not None:
                tested, planned = effect.transform.column(tested), effect.transform.column(planned)
            row = np.zeros(periods)
            row[end - estimate.readout : end] = (tested - planned)[-estimate.readout :]
            channels.append(column)
            coordinates.append(first + i)
            rows.append(row)
            lifts.append(estimate.lift / estimate.share)
            errors.append(estimate.se / estimate.share)
    return LiftReading(
        tuple(channels),
        np.array(coordinates, dtype=np.intp),
        np.array(rows),
        np.array(lifts),
        np.array(errors),
    )


def _sym(m: _Array) -> _Array:
    return 0.5 * (m + np.swapaxes(m, -1, -2))


def _units(filtered: DLMFit) -> tuple[_Array, _Array, _Array]:
    """The KPI's smoothed means `(T, p)`, their covariances in units of `V` `(T, p, p)`, and the
    smoother's gains `B_t = C_t G' R_(t+1)^-1` `(T - 1, p, p)`: Rauch-Tung-Striebel with the
    variance discount 1, so that one `V` scales every step."""
    g = _structure(filtered.model).evolution
    m, c, a, r, s = (
        filtered.mean,
        filtered.covariance,
        filtered.prior_mean,
        filtered.prior_covariance,
        filtered.scale,
    )
    horizon = m.shape[0]
    mean, cov = m.copy(), np.empty_like(c)
    gains = np.empty((max(horizon - 1, 0), *c.shape[1:]))
    cov[-1] = c[-1] / s[-1]
    for t in range(horizon - 2, -1, -1):
        # C_t and R_(t+1) are both stated at S_t, so the gain is free of the scale
        b = np.linalg.solve(r[t + 1], g @ c[t]).T
        gains[t] = b
        mean[t] = m[t] + b @ (mean[t + 1] - a[t + 1])
        cov[t] = _sym(c[t] / s[t] + b @ (cov[t + 1] - r[t + 1] / s[t]) @ b.T)
    return mean, cov, gains


def _with_tests(cov: _Array, gains: _Array, reading: LiftReading) -> _Array:
    """Each state's covariance with each test's reading of the path, in units of `V`,
    `(T, p, K)`: the part from the readout's periods at or after the state's, run back by the
    smoother's gains, and the part from those before it, run forward by their transposes."""
    horizon, p, _ = cov.shape
    k = reading.lift.size
    picks = np.zeros((horizon, p, k))
    picks[:, reading.coordinates, np.arange(k)] = reading.weights.T
    later = np.empty((horizon, p, k))
    later[-1] = cov[-1] @ picks[-1]
    for t in range(horizon - 2, -1, -1):
        later[t] = gains[t] @ later[t + 1] + cov[t] @ picks[t]
    earlier = np.zeros((p, k))
    out = np.empty((horizon, p, k))
    for t in range(horizon):
        out[t] = later[t] + cov[t] @ earlier
        if t < horizon - 1:
            earlier = gains[t].T @ (earlier + picks[t])
    return out


@dataclass(frozen=True, slots=True, eq=False)
class Lifted:
    """A fit's posterior given its KPI and its tests.

    `log_density` is the tests' log density given the KPI, multivariate Student t, so a spec's
    log-likelihood is its KPI filter's plus it. At the last period the state is Student t with
    `dof` degrees of freedom about `mean`, its scale matrix `covariance` stated at `scale`, the
    variance's estimate; `smoothed` holds every period's, as `chc.dlm.smooth` holds the KPI's.
    `reading` is the tests, `units` each one's variance in units of `V`, `kpi` the KPI's smoothed
    means, `cross` each state's covariance with each test given the KPI in units of `V`, and
    `factor` the Cholesky factor of the tests' covariance in those units: what a draw of the path
    given the tests reads.
    """

    reading: LiftReading
    log_density: float
    mean: _Array
    covariance: _Array
    dof: float
    scale: float
    smoothed: SmoothedStates
    units: _Array
    kpi: _Array
    cross: _Array
    factor: tuple[_Array, bool]


def lifted(filtered: DLMFit, reading: LiftReading) -> Lifted:
    """The KPI filter's posterior conditioned on the tests `reading` holds; see the module.

    Raises:
        ValueError: on a variance discount below 1, whose variance moves along the path, so that
            no one variance states the tests' errors; or final degrees of freedom not above 2,
            whose smoothed covariances are infinite.
        numpy.linalg.LinAlgError: on tests whose covariance is not positive definite in floating
            point.
    """
    if filtered.model.variance_discount != 1.0:
        msg = (
            "tests are read under one observational variance, and a variance discount of "
            f"{filtered.model.variance_discount} moves it along the path"
        )
        raise ValueError(msg)
    mean, cov, gains = _units(filtered)
    cross = _with_tests(cov, gains, reading)
    k, coordinates, weights = reading.lift.size, reading.coordinates, reading.weights
    scale, dof = float(filtered.scale[-1]), float(filtered.dof[-1])
    units = reading.se**2 / scale
    paths = np.einsum("jt,tjk->jk", weights, cross[:, coordinates, :])
    tests = _sym(paths) + np.diag(units)
    factor = linalg.cho_factor(tests, lower=True)
    error = reading.lift - np.einsum("jt,tj->j", weights, mean[:, coordinates])
    quadratic = float(error @ linalg.cho_solve(factor, error))
    log_det = 2.0 * float(np.log(np.diag(factor[0])).sum()) + k * math.log(scale)
    if math.isinf(dof):
        log_density = -0.5 * (k * math.log(2.0 * math.pi) + log_det + quadratic / scale)
        dof_after, scale_after = math.inf, scale
    else:
        log_density = float(
            special.gammaln((dof + k) / 2.0)
            - special.gammaln(dof / 2.0)
            - 0.5 * k * math.log(dof * math.pi)
            - 0.5 * log_det
            - (dof + k) / 2.0 * math.log1p(quadratic / (scale * dof))
        )
        dof_after = dof + k
        scale_after = (dof * scale + quadratic) / dof_after
    if not dof_after > 2.0:
        msg = f"the final degrees of freedom is {dof_after:.3g}, not above 2"
        raise ValueError(msg)
    gain = linalg.cho_solve(factor, cross.reshape(-1, k).T).T.reshape(cross.shape)
    mean_after = mean + gain @ error
    cov_after = _sym(cov - np.einsum("tik,tjk->tij", gain, cross))
    lagged = np.einsum("tij,tjk->tik", gains, cov[1:]) - np.einsum(
        "tik,tjk->tij", gain[:-1], cross[1:]
    )
    expected = scale_after if math.isinf(dof_after) else scale_after * dof_after / (dof_after - 2)
    horizon = mean.shape[0]
    return Lifted(
        reading=reading,
        log_density=log_density,
        mean=mean_after[-1],
        covariance=scale_after * cov_after[-1],
        dof=dof_after,
        scale=scale_after,
        smoothed=SmoothedStates(
            mean_after,
            expected * cov_after,
            expected * lagged,
            np.full(horizon, dof_after),
            exact=True,
        ),
        units=units,
        kpi=mean,
        cross=cross,
        factor=factor,
    )


def lifted_draws(filtered: DLMFit, tests: Lifted, draws: int, seed: int) -> PosteriorDraws:
    """`draws` joint draws of the path and the variance given the KPI and the tests.

    Each is a draw of the KPI's posterior (`chc.dlm.backward_sample`) at a variance drawn from its
    posterior given the tests too, moved by the tests it would have read (Matheron's rule): the
    KPI's draw's departure from its mean is a draw of the path's covariance in units of `V` times
    the root of its variance, so it is scaled to the new one, and a test's error is drawn at it.

    Raises:
        ValueError: on `draws` below 1.
    """
    base = backward_sample(filtered, draws, seed)
    rng = np.random.default_rng((seed, 1))
    reading = tests.reading
    k = reading.lift.size
    if math.isinf(tests.dof):
        variance = np.full(draws, tests.scale)
    else:
        variance = tests.dof * tests.scale / rng.chisquare(tests.dof, draws)
    ratio = np.sqrt(variance / base.variance[:, -1])
    path = tests.kpi + ratio[:, None, None] * (base.states - tests.kpi)
    read = np.einsum("jt,dtj->dj", reading.weights, path[:, :, reading.coordinates])
    noise = np.sqrt(variance)[:, None] * rng.standard_normal((draws, k)) * np.sqrt(tests.units)
    moved = linalg.cho_solve(tests.factor, (reading.lift - read - noise).T).T
    path += np.einsum("tpk,dk->dtp", tests.cross, moved)
    return PosteriorDraws(path, np.repeat(variance[:, None], path.shape[1], axis=1))


def signed_sample(
    fit: DLMFit,
    signs: Mapping[int, Sign],
    reading: LiftReading | None,
    draws: int,
    seed: int,
    *,
    chains: int = 4,
    warmup: int | None = None,
) -> ConstrainedDraws:
    """`chc.dlm.constrained_sample`'s draws of the posterior truncated to `signs`, given the tests
    `reading` holds as well.

    The sampler is CHC's, its blocks unchanged but one: the constrained coefficients' distinct
    values given `V` are its Gaussian conditioned on the tests. A test reads only the values, so the
    other states given them are the KPI's, as CHC draws them. The values' Gaussian, conditioned
    once, is truncated to the signs by CHC's exact Hamiltonian trajectory; its velocity, a centred
    draw of it, is CHC's draw moved by the tests it would have read; and `V` given the values is
    the inverse gamma with the tests' errors counted, each test's variance `se^2 / S_T` times `V`.
    With `reading` None, the sampler is CHC's, draw for draw.

    Raises:
        ValueError: on a test of a coefficient that is not held to a sign; what
            `chc.dlm.constrained_sample` refuses is the caller's to have refused, as `DlmSpec`
            does.
        RuntimeError: on a trajectory that meets more walls than CHC's cap.
    """
    model = fit.model
    s = _structure(model)
    constrained = {
        int(s.regression[column]): 1.0 if sign == "positive" else -1.0
        for column, sign in signs.items()
    }
    warmup = draws if warmup is None else warmup
    rng = np.random.default_rng(seed)
    horizon, p = fit.mean.shape
    design, y = fit.design, fit.y
    observed = ~np.isnan(y)
    y0 = np.where(observed, y, 0.0)
    g = s.evolution
    first_cov, w = _transitions(fit)
    first_mean = fit.prior_mean[0]
    learned = not math.isinf(model.prior.dof)
    cs = sorted(constrained)
    us = [j for j in range(p) if j not in constrained]

    full_gain, full_prior, full_cov, full_back, full_root = _unit_filter(
        g, design, observed, first_cov, w
    )
    filtered = np.empty((horizon, p))
    a = first_mean
    for t in range(horizon):
        if t:
            a = g @ filtered[t - 1]
        filtered[t] = a + full_gain[t] * (y0[t] - design[t] @ a)
    smoothed, smoothed_cov = filtered.copy(), full_cov.copy()
    for t in range(horizon - 2, -1, -1):
        smoothed[t] += full_back[t] @ (smoothed[t + 1] - g @ filtered[t])
        smoothed_cov[t] = _sym(
            full_cov[t] + full_back[t] @ (smoothed_cov[t + 1] - full_prior[t + 1]) @ full_back[t].T
        )
    starts = [np.concatenate([[0], 1 + np.flatnonzero(w[:, j, j] > 0.0)]) for j in cs]
    times = np.unique(np.concatenate(starts))
    step = np.concatenate(starts)
    which = np.repeat(np.arange(len(cs)), [run.size for run in starts])
    columns = np.asarray(cs)[which]
    spot = np.searchsorted(times, step)
    value_cov = _path_covariance(full_back, smoothed_cov, cs, times)[
        spot[:, None], which[:, None], spot, which
    ]
    value_mean = smoothed[step, columns]
    value_sign = np.array([constrained[j] for j in columns])
    first_value = np.cumsum([0] + [run.size for run in starts[:-1]])
    fill = np.stack(
        [
            first_value[c] + np.searchsorted(run, np.arange(horizon), side="right") - 1
            for c, run in enumerate(starts)
        ],
        axis=1,
    )

    dof, rate_kpi = float(fit.dof[-1]), float(fit.dof[-1] * fit.scale[-1])
    if reading is not None:
        loose = sorted({int(j) for j in reading.coordinates} - set(cs))
        if loose:
            msg = (
                f"tests read the coefficients at state coordinates {loose}, which are not held "
                "to a sign; in a signed fit a test reads a signed channel"
            )
            raise ValueError(msg)
        # each test's weights summed over the periods a value spans
        k = reading.lift.size
        reads = np.zeros((k, value_mean.size))
        for j in range(k):
            np.add.at(reads[j], fill[:, cs.index(int(reading.coordinates[j]))], reading.weights[j])
        units = reading.se**2 / float(fit.scale[-1])
        tests = _sym(reads @ value_cov @ reads.T) + np.diag(units)
        factor = linalg.cho_factor(tests, lower=True)
        towards = linalg.cho_solve(factor, reads @ value_cov).T  # (values, K)
        error = reading.lift - reads @ value_mean
        value_mean = value_mean + towards @ error
        value_cov = _sym(value_cov - towards @ reads @ value_cov)
        if learned:
            dof += k
            rate_kpi += float(error @ linalg.cho_solve(factor, error))

    gu, fu = g[np.ix_(us, us)], design[:, us]
    shift = first_cov[np.ix_(us, cs)] @ np.linalg.inv(first_cov[np.ix_(cs, cs)])
    pu = len(us)
    if pu:
        r_first = first_cov[np.ix_(us, us)] - shift @ first_cov[np.ix_(cs, us)]
        gains, _, _, back, roots = _unit_filter(gu, fu, observed, r_first, w[:, us][:, :, us])

    value_root = np.linalg.cholesky(value_cov)
    shape = (dof + value_mean.size) / 2.0 if learned else math.inf

    start = backward_sample(fit, chains, seed=int(rng.integers(2**31)))
    theta = start.states.copy()
    theta[:, :, cs] = (value_sign * np.maximum(value_sign * theta[:, step, columns], 0.0))[:, fill]
    variance = start.variance[:, -1] if learned else np.full(chains, model.prior.scale)

    kept_states = np.empty((chains, draws, horizon, p))
    kept_variance = np.empty((chains, draws))
    bounces = 0
    for sweep in range(warmup + draws):
        if learned:
            white = linalg.solve_triangular(
                value_root, (theta[:, step, columns] - value_mean).T, lower=True
            )
            rate = 0.5 * (rate_kpi + (white**2).sum(axis=0))
            variance = 1.0 / rng.gamma(shape, 1.0 / rate)
        sd = np.sqrt(variance)[:, None]

        noise = rng.standard_normal((chains, horizon, p))
        path = np.empty((chains, horizon, p))
        path[:, -1] = noise[:, -1] @ full_root[-1].T
        for t in range(horizon - 2, -1, -1):
            path[:, t] = path[:, t + 1] @ full_back[t].T + noise[:, t] @ full_root[t].T
        velocity = path[:, step, columns]
        if reading is not None:
            missed = rng.standard_normal((chains, reading.lift.size)) * np.sqrt(units)
            velocity = velocity - (velocity @ reads.T + missed) @ towards.T
        moved, met = _reflected_orbit(
            value_mean,
            theta[:, step, columns] - value_mean,
            sd * velocity,
            value_sign,
            value_cov,
        )
        bounces += met
        values = value_sign * np.maximum(value_sign * (value_mean + moved), 0.0)
        theta[:, :, cs] = values[:, fill]

        if pu:
            offsets = y0 - np.einsum("tp,ctp->ct", design[:, cs], theta[:, :, cs])
            a = first_mean[us] + (theta[:, 0, cs] - first_mean[cs]) @ shift.T
            means = np.empty((chains, horizon, pu))
            ahead = np.empty((chains, horizon, pu))
            for t in range(horizon):
                if t:
                    a = means[:, t - 1] @ gu.T
                ahead[:, t] = a
                if observed[t]:
                    a = a + (offsets[:, t] - a @ fu[t])[:, None] * gains[t]
                means[:, t] = a
            noise = rng.standard_normal((chains, horizon, pu))
            u = means[:, -1] + sd * (noise[:, -1] @ roots[-1].T)
            theta[:, -1, us] = u
            for t in range(horizon - 2, -1, -1):
                u = (
                    means[:, t]
                    + (u - ahead[:, t + 1]) @ back[t].T
                    + sd * (noise[:, t] @ roots[t].T)
                )
                theta[:, t, us] = u

        if sweep >= warmup:
            kept_states[:, sweep - warmup] = theta
            kept_variance[:, sweep - warmup] = variance

    result = ConstrainedDraws(kept_states, kept_variance, *_convergence(kept_states, kept_variance))
    walls = bounces / (chains * (warmup + draws))
    log.log(
        logging.INFO if result.mixed else logging.WARNING,
        "dlm signed sample with %d tests: %d chains of %d draws, R-hat %.4f, bulk ESS %.0f, "
        "%.1f walls a trajectory%s",
        0 if reading is None else reading.lift.size,
        chains,
        draws,
        result.rhat,
        result.ess,
        walls,
        "" if result.mixed else "; the chains have not mixed and the draws are not to be read",
        extra={"rhat": result.rhat, "ess": result.ess, "walls": walls, "mixed": result.mixed},
    )
    return result
