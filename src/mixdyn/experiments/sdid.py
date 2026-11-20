"""A geo test read by synthetic difference in differences, as Arkhangelsky, Athey, Hirshberg, Imbens
and Wager (2021, American Economic Review 111(12), "Synthetic Difference-in-Differences") state it.

`synthetic_did` is their Algorithm 1 on a balanced panel `y` of `N` units by `T` periods: the
treated units, `N_tr` of them, are treated over the last `T_post` periods, and the `N_co` controls
and the first `T_pre` periods are read before it.

- The noise level is `sigma^2`, the variance of the controls' first differences over the pre-period,
  `sum (D_it - mean D)^2 / (N_co (T_pre - 1))` with `D_it = y_i(t+1) - y_it`.
- The unit weights `w` minimise over an intercept `w0` and `w` on the simplex
  `sum_{t <= T_pre} (w0 + sum_i w_i y_it - mean_tr y_t)^2 + zeta^2 T_pre ||w||^2`, at
  `zeta = (N_tr T_post)^(1/4) sigma`.
- The time weights `l` minimise over an intercept `l0` and `l` on the simplex
  `sum_{i <= N_co} (l0 + sum_t l_t y_it - mean_post y_i)^2 + zeta_l^2 N_co ||l||^2`, at
  `zeta_l = 1e-6 sigma`, the authors' package's ridge, there only to make the weights unique.
- The effect `tau` is the two-way fixed-effects regression of `y` on unit and period effects and the
  treatment, each cell weighted by its unit's weight times its period's: the treated units at
  `1 / N_tr`, the post-periods at `1 / T_post`. That is
  `(mean_post y_tr - sum_t l_t y_tr,t) - sum_i w_i (mean_post y_i - sum_t l_t y_it)`.

Each weight problem is a least squares over the simplex. The intercept is eliminated exactly, by
centring each column, and the ridge is one more block of rows, so the problem is
`min ||Z w - c||^2` over the simplex. With `Z - c 1'` and a row of ones appended it is one
non-negative least squares, which Lawson and Hanson's active set solves in finitely many steps: for
`u >= 0` minimising `||(Z - c 1') u||^2 + (1'u - 1)^2`, `u / 1'u` is the simplex optimum, the
reduction `chc.scm` documents for a synthetic control. The conditions of optimality, the gradient
level on the support and at or above that level off it, are then checked to `1e-10` of the
problem's scale, and weights that miss them raise rather than read on. The data are scaled by
`sigma` before either problem is set, and every term of both losses is in the KPI's units squared,
so the weights do not depend on the KPI's units, and the effect and its error scale with them.

The variance is the paper's placebo (Algorithm 4) by default: `N_tr` of the controls drawn as
placebo treated units, the estimator run whole on the controls alone, and the variance the
placebo effects'. It needs `N_co > N_tr` and holds with one treated unit; where there are no more
placebo assignments than `replications`, every one is read once. The jackknife (Algorithm 3)
leaves out each unit, controls included, with the weights held and renormalised over the units
left, and refuses fewer than two treated units or two controls: left out, one would leave none.

Design-aware unit placebos (`variance="unit_placebo"`, opt-in, ADR 0058) draw the placebo treated
sets as the test's own were drawn, `N_tr` of all `N` units, treated ones included, each read with
every other unit its control, and read them by their distribution instead of their variance. Every
set is read where there are at most `replications`, the observed one among them, and otherwise the
observed one and `replications` drawn from the same child stream. Each effect is studentised by
its pre-period out-of-sample error, the estimator run on the pre-period alone with its last
`T_post` periods as a pseudo post-period and the root mean square of the per-period gaps there,
`mean_tr y_t - sum_i w_i y_it` less its time-weighted pre-period. The p-value is the share of the
`A` studentised effects at least as far from 0 as the observed one, and the interval the lift
plus or minus its own error times the `ceil(alpha A)`-th largest of them in absolute value, so
that it holds 0 exactly where the p-value is at least `alpha`; the estimate's error is that
interval's half at 95 % over `z(0.975)`, and the reading's `se` the spread of the effects. With no
effect and a treated set drawn at random the test is exact whatever the noise; under a lift the
sets that hold treated units read part of it, so the interval is wider than the lift's own
spread. It needs a pre-period of at least `T_post + 2` periods, and enough sets to reject:
`alpha A > 1` at `alpha` and at 5 %.

The paper reads either variance against the normal. With few controls the variance is itself read
with error, and that error is larger than its count of placebos suggests: the time weights are fit
on the controls, `|S|` of them non-zero, so the controls' gaps keep `N_co - |S|` degrees of
freedom, as a regression's residuals keep their count less its parameters (an intercept and
`|S| - 1` free weights). The interval and the p-value are therefore a t's (`quantile="t"`) at
`nu_0 = max(N_co - |S|, (N_co - 1) / 2)`: with few controls `|S|` nears or passes `N_co` and the
count alone falls to 1, far below the variance's own (its spread over panels gives 2 to 8 at 5 to
10 controls), so the t loses all power; the floor at half a sample variance's `N_co - 1` held the
test's level at 5 to 10 controls where `N_co / 2` did not (ADR 0048). Random placebo draws add
their own error, a sample variance's of `B - 1` degrees of freedom, so
`1 / nu = 1 / nu_0 + 1 / (B - 1)` for `B` drawn, as Satterthwaite's count of two independent
sources. The jackknife's variance with few treated units rests on the `N_tr` leave-one-treated-out
terms, so its count is the smaller of `nu_0` and `N_tr - 1`. `quantile="normal"` reads the
paper's normal.

The `LiftEstimate` handed on carries the error the geo-test readers' do (`mixdyn.experiments.
reading`): at `quantile="t"` widened to `se t_nu(0.975) / z(0.975)`, so that the normal 95 %
interval a model reads of it is the reading's t interval; at `quantile="normal"` the plain error.
Random placebos are drawn from a child of `seed`'s stream, never from `default_rng(seed)` itself.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
from scipy import optimize, stats

from mixdyn.calibration import LiftEstimate
from mixdyn.experiments.candidates import _UPPER
from mixdyn.experiments.reading import _half

if TYPE_CHECKING:
    from collections.abc import Collection

    from mixdyn.data import Geo, MediaPanel

type _Array = npt.NDArray[np.float64]
type _Flags = npt.NDArray[np.bool_]
type Variance = Literal["placebo", "jackknife", "unit_placebo"]
type Quantile = Literal["t", "normal"]
type Verdict = Literal["up", "down", "inconclusive"]

_KKT = 1e-10  # the conditions of optimality's residual, relative to the problem's scale
_ZETA_TIME = 1e-6  # the time weights' ridge as a share of the noise level, the authors' package's


@dataclass(frozen=True, slots=True)
class SdidFit:
    """Synthetic difference in differences on one panel.

    `effect` is `tau`, the treated units' effect per unit and post-period. `unit_weights` are the
    controls', in their order in the panel, and `time_weights` the pre-periods'. `noise` is
    `sigma`, `zeta` the unit weights' regularisation. `kkt` is the larger of the two weight
    problems' residuals of the conditions of optimality, relative to each problem's scale.
    """

    effect: float
    unit_weights: _Array
    time_weights: _Array
    noise: float
    zeta: float
    kkt: float


@dataclass(frozen=True, slots=True)
class SdidReading:
    """A geo test as synthetic difference in differences read it.

    `lift` is the treated geos' KPI over the readout less its counterfactual, summed over the geos
    and the readout's periods, so `effect` times both counts, and `se` its standard error by
    `variance`'s method over `replications` estimates. `lower` and `upper` bound the interval at
    `alpha`, two-sided, a t's at `df` degrees of freedom (infinite for the normal), and `p_value`
    is the same test's of no lift. `verdict` is "up" where the interval lies above 0, "down" where
    below and "inconclusive" where it holds 0. `controls` and `pre` name what `unit_weights` and
    `time_weights` weigh.
    """

    method: Literal["sdid"]
    variance: Variance
    lift: float
    se: float
    lower: float
    upper: float
    p_value: float
    verdict: Verdict
    alpha: float
    df: float
    effect: float
    replications: int
    controls: tuple[Geo, ...]
    unit_weights: _Array
    pre: npt.NDArray[np.datetime64]
    time_weights: _Array


def _simplex_ls(z: _Array, c: _Array) -> tuple[_Array, float]:
    """The point `w` of the simplex minimising `||z w - c||^2`, and the relative residual of its
    conditions of optimality."""
    n = z.shape[1]
    shifted = z - c[:, None]
    scale = float(np.sqrt(np.mean(shifted**2)))
    if scale == 0.0:
        return np.full(n, 1.0 / n), 0.0
    design = np.vstack([shifted / scale, np.ones(n)])
    target = np.zeros(design.shape[0])
    target[-1] = 1.0
    u, _ = optimize.nnls(design, target, maxiter=50 * n)
    w = u / u.sum()
    return w, _kkt(shifted / scale, w)


def _kkt(z: _Array, w: _Array) -> float:
    """The residual of `w`'s conditions of optimality for `||z w||^2` over the simplex: the
    gradient level on the support, at or above that level off it, relative to the gradient's
    scale."""
    gradient = z.T @ (z @ w)
    support = w > 0.0
    level = float(gradient[support].mean())
    stationary = float(np.abs(gradient[support] - level).max())
    signs = float(np.maximum(level - gradient[~support], 0.0).max(initial=0.0))
    scale = max(float(np.abs(z.T @ z).max()), np.finfo(np.float64).tiny)
    return max(stationary, signs, abs(float(w.sum()) - 1.0) * scale) / scale


def _weights(a: _Array, b: _Array, ridge: float) -> tuple[_Array, float]:
    """The simplex weights `w` minimising over an intercept `w0`
    `||w0 + a w - b||^2 + ridge ||w||^2`, the intercept eliminated by centring."""
    a = a - a.mean(axis=0)
    b = b - b.mean()
    z = np.vstack([a, math.sqrt(ridge) * np.eye(a.shape[1])])
    return _simplex_ls(z, np.concatenate([b, np.zeros(a.shape[1])]))


def _check(y: _Array, treated: _Flags, post: int) -> None:
    if y.ndim != 2 or treated.shape != (y.shape[0],):
        msg = f"y must be units by periods with one flag a unit, got {y.shape} and {treated.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(y)):
        msg = "y must be finite: synthetic difference in differences reads a balanced panel"
        raise ValueError(msg)
    n_treated = int(treated.sum())
    if n_treated == 0:
        msg = "no unit is treated"
        raise ValueError(msg)
    if n_treated == treated.size:
        msg = "no unit is a control: every unit is treated"
        raise ValueError(msg)
    if not 1 <= post < y.shape[1]:
        msg = f"a post-period of {post} is not within the {y.shape[1]} periods"
        raise ValueError(msg)
    if y.shape[1] - post < 2:
        msg = f"a pre-period of {y.shape[1] - post} period reads no noise: it needs at least 2"
        raise ValueError(msg)


def synthetic_did(
    y: npt.ArrayLike,
    treated: npt.ArrayLike,
    post: int,
    *,
    weights: Literal["sdid", "did"] = "sdid",
) -> SdidFit:
    """Synthetic difference in differences on `y`, units by periods, the units flagged in
    `treated` treated over the last `post` periods.

    `weights="did"` holds every control's and every pre-period's weight equal, which is plain
    difference in differences.

    Raises:
        ValueError: on a `y` not finite and two-dimensional, flags not one a unit, no treated unit,
            no control, a post-period not within the periods, or a pre-period under 2 periods.
        RuntimeError: on weights that miss the conditions of optimality by more than `1e-10`.
    """
    return _fit(np.array(y, dtype=np.float64), np.array(treated, dtype=np.bool_), post, weights)


def _fit(y: _Array, treated: _Flags, post: int, weights: Literal["sdid", "did"]) -> SdidFit:
    _check(y, treated, post)
    pre = y.shape[1] - post
    controls = y[~treated]
    n_control, n_treated = controls.shape[0], int(treated.sum())
    steps = np.diff(controls[:, :pre], axis=1)
    noise = float(np.sqrt(np.mean((steps - steps.mean()) ** 2)))
    zeta = (n_treated * post) ** 0.25 * noise
    if weights == "did":
        omega, lam, kkt = np.full(n_control, 1.0 / n_control), np.full(pre, 1.0 / pre), 0.0
    else:
        scaled, unit_ridge, time_ridge = y, 0.0, 0.0
        if noise > 0.0:
            scaled = y / noise
            unit_ridge = (zeta / noise) ** 2 * pre
            time_ridge = _ZETA_TIME**2 * n_control
        treated_mean = scaled[treated, :pre].mean(axis=0)
        omega, unit_kkt = _weights(scaled[~treated, :pre].T, treated_mean, unit_ridge)
        post_mean = scaled[~treated, pre:].mean(axis=1)
        lam, time_kkt = _weights(scaled[~treated, :pre], post_mean, time_ridge)
        kkt = max(unit_kkt, time_kkt)
        if kkt > _KKT:
            msg = f"the weights miss their conditions of optimality by {kkt:.3g}, above {_KKT}"
            raise RuntimeError(msg)
    effect = _effect(y, treated, pre, omega, lam)
    return SdidFit(effect, omega, lam, noise, zeta, kkt)


def _effect(y: _Array, treated: _Flags, pre: int, omega: _Array, lam: _Array) -> float:
    """The double difference of the weighted means: the treated units' post-period mean less their
    time-weighted pre-period, less the same of the unit-weighted controls."""
    gaps = y[:, pre:].mean(axis=1) - y[:, :pre] @ lam
    return float(gaps[treated].mean() - omega @ gaps[~treated])


def _residual_df(treated: _Flags, fit: SdidFit) -> float:
    """The degrees of freedom the controls' gaps keep once the time weights are fit on them, at
    least half the `N_co - 1` of the controls' sample variance."""
    n_control = int((~treated).sum())
    kept = n_control - int(np.count_nonzero(fit.time_weights))
    return float(max(kept, (n_control - 1) / 2.0, 1.0))


def _placebo(
    y: _Array, treated: _Flags, post: int, fit: SdidFit, replications: int, seed: int
) -> tuple[float, int, float]:
    """The placebo variance, its count of placebos and its degrees of freedom."""
    controls = y[~treated]
    n_control, n_treated = controls.shape[0], int(treated.sum())
    if n_control <= n_treated:
        msg = f"a placebo needs more controls than treated units, got {n_control} and {n_treated}"
        raise ValueError(msg)
    df = _residual_df(treated, fit)
    draws, drawn = _draws(n_control, n_treated, replications, seed)
    if drawn:
        df = 1.0 / (1.0 / df + 1.0 / (replications - 1))
    effects = []
    for draw in draws:
        placebo = np.zeros(n_control, dtype=np.bool_)
        placebo[draw] = True
        effects.append(_fit(controls, placebo, post, "sdid").effect)
    return float(np.var(effects)), len(effects), df


def _draws(
    units: int, size: int, replications: int, seed: int
) -> tuple[list[list[int]] | list[npt.NDArray[np.intp]], bool]:
    """The placebo treated sets of `size` of `units` units, every one where there are at most
    `replications`, else `replications` drawn from a child of `seed`'s stream, and whether they
    were drawn."""
    if math.comb(units, size) <= replications:
        return [list(c) for c in itertools.combinations(range(units), size)], False
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    return [rng.choice(units, size, replace=False) for _ in range(replications)], True


def _held_out(y: _Array, treated: _Flags, post: int) -> float:
    """The estimator's pre-period out-of-sample error: run on the pre-period alone, its last
    `post` periods a pseudo post-period, the root mean square of its per-period gaps there."""
    pre = y[:, : y.shape[1] - post]
    fit = _fit(pre, treated, post, "sdid")
    path = pre[treated].mean(axis=0) - fit.unit_weights @ pre[~treated]
    first = pre.shape[1] - post
    gaps = path[first:] - path[:first] @ fit.time_weights
    return float(np.sqrt(np.mean(gaps * gaps)))


def _unit_placebo(
    y: _Array, treated: _Flags, post: int, fit: SdidFit, replications: int, seed: int
) -> tuple[_Array, _Array]:
    """The observed effect and the placebos', first the observed, each placebo `N_tr` of all the
    units, and each one's pre-period out-of-sample error."""
    n_units, n_treated = treated.size, int(treated.sum())
    if y.shape[1] - 2 * post < 2:
        msg = (
            f"a pre-period of {y.shape[1] - post} holds no pseudo post-period of {post} after "
            "2 periods: unit placebos need it"
        )
        raise ValueError(msg)
    effects, errors = [fit.effect], [_held_out(y, treated, post)]
    observed = frozenset(np.flatnonzero(treated).tolist())
    draws, drawn = _draws(n_units, n_treated, replications, seed)
    for draw in draws:
        if not drawn and frozenset(draw) == observed:
            continue
        placebo = np.zeros(n_units, dtype=np.bool_)
        placebo[draw] = True
        effects.append(_fit(y, placebo, post, "sdid").effect)
        errors.append(_held_out(y, placebo, post))
    return np.array(effects), np.array(errors)


def _jackknife(y: _Array, treated: _Flags, post: int, fit: SdidFit) -> tuple[float, int, float]:
    """The jackknife variance over every unit, its count of units and its degrees of freedom."""
    n_treated = int(treated.sum())
    if n_treated < 2:
        msg = "the jackknife needs at least two treated units: left out, one leaves none"
        raise ValueError(msg)
    if treated.size - n_treated < 2:
        msg = "the jackknife needs at least two control units: left out, one leaves none"
        raise ValueError(msg)
    pre = y.shape[1] - post
    omega = np.zeros(treated.size)
    omega[~treated] = fit.unit_weights
    effects = []
    for left in range(treated.size):
        keep = np.arange(treated.size) != left
        kept = omega[keep][~treated[keep]]
        total = kept.sum()
        kept = kept / total if total > 0.0 else np.full(kept.size, 1.0 / kept.size)
        effects.append(_effect(y[keep], treated[keep], pre, kept, fit.time_weights))
    n = treated.size
    variance = (n - 1) / n * float(np.sum((np.array(effects) - fit.effect) ** 2))
    return variance, n, min(_residual_df(treated, fit), float(n_treated - 1))


def read_sdid(
    panel: MediaPanel,
    treated: Collection[Geo],
    start: np.datetime64,
    end: np.datetime64,
    *,
    spend: npt.ArrayLike,
    planned: npt.ArrayLike,
    alpha: float = 0.05,
    variance: Variance = "placebo",
    quantile: Quantile = "t",
    replications: int = 200,
    seed: int = 0,
    control: Collection[Geo] | None = None,
) -> tuple[LiftEstimate, SdidReading]:
    """A geo test in `panel` read by synthetic difference in differences: the geos in `treated`
    treated from the date `start` to the date `end`, both read, the geos in `control` the controls
    (every other geo where it is None) and every period before `start` the pre-period. The periods
    after `end` and the geos in neither group are not read.

    `spend` and `planned` are the treated geos' spend on the tested channel as the test ran and as
    planned, `LiftEstimate`'s, the readout their last periods. `replications` caps the placebos
    drawn, from a child of `seed`'s stream. The estimate's standard error is the reading's widened
    to its t at `quantile="t"`, `se t_df(0.975) / z(0.975)`, and the reading's at "normal". At
    `variance="unit_placebo"` the placebos are read by their studentised distribution, its
    interval's half at 95 % over `z(0.975)` the estimate's error, and `quantile` is not read.

    Raises:
        ValueError: on an `alpha` outside `(0, 1)`, fewer than 2 replications, a treated or control
            geo not in the panel, a geo both treated and a control, dates not the panel's or `end`
            before `start`, a KPI missing over the geos and periods read, what `synthetic_did`
            refuses, a placebo with no more controls than treated geos, a jackknife with fewer
            than two treated geos or fewer than two control geos, unit placebos with a pre-period
            under the post-period plus 2, an assignment whose gaps over its pseudo post-period are
            all 0 or too few placebos to reject at `alpha` or at 5 %, an unknown variance or
            quantile, and what `LiftEstimate` refuses, a standard error of 0 among it.
    """
    if not (math.isfinite(alpha) and 0.0 < alpha < 1.0):
        msg = f"alpha={alpha!r} is not in (0, 1)"
        raise ValueError(msg)
    if replications < 2:
        msg = f"replications={replications!r} reads no variance: it needs at least 2"
        raise ValueError(msg)
    if quantile not in ("t", "normal"):
        msg = f"quantile={quantile!r} is not 't' or 'normal'"
        raise ValueError(msg)
    unknown = sorted(set(treated) - set(panel.geos))
    if unknown:
        msg = f"treated geos {unknown} are not in the panel"
        raise ValueError(msg)
    if control is not None:
        unknown = sorted(set(control) - set(panel.geos))
        if unknown:
            msg = f"control geos {unknown} are not in the panel"
            raise ValueError(msg)
        both = sorted(set(control) & set(treated))
        if both:
            msg = f"geos {both} are both treated and controls"
            raise ValueError(msg)
    dates = panel.dates
    first, last = np.flatnonzero(dates == start), np.flatnonzero(dates == end)
    if first.size != 1 or last.size != 1 or last[0] < first[0]:
        msg = f"start {start} and end {end} must be the panel's dates, in that order"
        raise ValueError(msg)
    rows = [i for i, g in enumerate(panel.geos) if g in treated or control is None or g in control]
    geos = tuple(panel.geos[i] for i in rows)
    y = panel.kpi[rows, : last[0] + 1]
    flags = np.array([g in treated for g in geos])
    post = int(last[0] - first[0] + 1)
    if not np.all(np.isfinite(y)):
        msg = "the KPI is missing over the periods read; the panel must be balanced over them"
        raise ValueError(msg)
    fit = _fit(y, flags, post, "sdid")
    cells = int(flags.sum()) * post
    randomised = None
    if variance == "placebo":
        var, count, df = _placebo(y, flags, post, fit, replications, seed)
    elif variance == "jackknife":
        var, count, df = _jackknife(y, flags, post, fit)
    elif variance == "unit_placebo":
        effects, errors = _unit_placebo(y, flags, post, fit, replications, seed)
        if not (errors > 0.0).all():
            msg = "an assignment's gaps over its pseudo post-period are all 0: nothing reads them"
            raise ValueError(msg)
        studentised = np.abs(effects / errors)
        count, df = studentised.size, math.inf
        if min(alpha, 2.0 * (1.0 - _UPPER)) * count <= 1.0:
            msg = (
                f"{count} assignments are too few for the test to reject at a level of "
                f"{1.0 - alpha} or of 0.95, the estimate's"
            )
            raise ValueError(msg)
        var = float(np.var(effects))
        error, lift = float(errors[0]) * cells, fit.effect * cells
        randomised = (
            _half(studentised, error, lift, alpha),
            _half(studentised, error, lift, 2.0 * (1.0 - _UPPER)),
            float(np.mean(studentised >= studentised[0])),
        )
    else:
        msg = f"variance={variance!r} is not 'placebo' or 'jackknife', nor 'unit_placebo'"
        raise ValueError(msg)
    if quantile == "normal":
        df = math.inf
    lift, se = fit.effect * cells, math.sqrt(var) * cells
    widened = se
    if randomised is not None:
        widened = randomised[1] / float(stats.norm.ppf(_UPPER))
    elif quantile == "t":
        widened = se * float(stats.t.ppf(_UPPER, df) / stats.norm.ppf(_UPPER))
    estimate = LiftEstimate(
        np.asarray(spend, dtype=np.float64),
        np.asarray(planned, dtype=np.float64),
        post,
        lift,
        widened,
        geos=frozenset(treated),
        end=np.datetime64(end, "D"),
    )
    if randomised is None:
        half = float(stats.t.ppf(1.0 - alpha / 2.0, df)) * se
        p_value = float(2.0 * stats.t.sf(abs(lift) / se, df))
    else:
        half, _, p_value = randomised
    lower, upper = lift - half, lift + half
    verdict: Verdict = "up" if lower > 0.0 else "down" if upper < 0.0 else "inconclusive"
    reading = SdidReading(
        "sdid",
        variance,
        lift,
        se,
        lower,
        upper,
        p_value,
        verdict,
        alpha,
        df,
        fit.effect,
        count,
        tuple(g for g, f in zip(geos, flags, strict=True) if not f),
        fit.unit_weights,
        dates[: y.shape[1] - post],
        fit.time_weights,
    )
    return estimate, reading
