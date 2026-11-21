"""How far a channel's return moves under a confounder the model does not see (ADR 0061).

Both forms bound; neither corrects. Each reads one number, the strength a confounder would need
to move the return, and reports what the return is at a strength the reader states.

**The ridge engine** (`ridge_sensitivity`): Cinelli and Hazlett (2020, JRSS-B 82(1)) on the least
squares of the fit's KPI on its design, the ridge's own columns at penalty 0, each channel's
saturated under its adstock and Hill. With `t` the channel's t-statistic and `df` the regression's
residual degrees of freedom, `f = |t| / sqrt(df)` is its partial Cohen's f and

    R2(Y ~ D | X) = t^2 / (t^2 + df),
    RV_q = (sqrt(f_q^4 + 4 f_q^2) - f_q^2) / 2,             f_q = q f,

the share of both the channel's and the KPI's residual variance a confounder must explain to move
the coefficient by `q` of itself. `RV_{q, alpha}` moves the interval's near end there instead:
with `f* = t_{df - 1, 1 - alpha / 2} / sqrt(df - 1)` it is 0 where `f_q <= f*`, the root above at
`f_q - f*` where `f* < f_q < 1 / f*`, and `(f_q^2 - f*^2) / (1 + f_q^2)` past it, where the KPI's
share that moves the interval most is below the treatment's. A confounder that explains `R2yu`
of the KPI's residual variance given the channel and the design, and `R2du` of the channel's given
the design, moves the coefficient by at most

    |bias| = se sqrt(df) sqrt(R2yu R2du / (1 - R2du)),

and the adjusted standard error is `se sqrt((1 - R2yu) / (1 - R2du)) sqrt(df / (df - 1))`. A
confounder `k_d` times as strong as an observed control on the channel and `k_y` times on the
KPI, and independent of it given the rest, has `R2du = k_d R2(D ~ X_j | X_-j) / (1 - R2(D ~ X_j |
X_-j))` and the bound on `R2yu` of their Section 6.1. `validation/sensitivity.mac` checks each
closed form.

The return on the window's spend is the coefficient times the channel's saturated spend over the
window over that spend, so a share `q` of the coefficient is a share `q` of the return. The
statistics are the unpenalised regression's: the ridge's coefficient is shrunk from it, and a
geo test's anchor pulls it, and neither has a t-statistic of its own. `RidgeSensitivity` keeps the
fit's coefficient beside the regression's estimate.

**The Bayesian engines** (`msm_sensitivity`): a marginal sensitivity model of level `Gamma` on the
channel's spend given a hidden confounder (Tan 2006; the continuous form of Jesson et al. 2022).
Its density ratio's lower side, `1 / Gamma`, holds the share of the channel's residual variance
the confounder explains to at most `(1 - 1/Gamma)^2`, a sharp share (Dorn and Guo 2023 for the
binary model's sharpness; `chc.ChannelSensitivity` since 0.22.0). Nothing bounds the KPI's side,
so `R2yu` is 1, and at each draw `s` of the posterior the coefficient moves by at most

    kappa(Gamma) sigma_s / sd_s(D | X),        kappa(Gamma) = (Gamma - 1) / sqrt(2 Gamma - 1),

`kappa` `chc.ChannelSensitivity.factor`, `sigma_s` the root mean square of the KPI less the draw's
mean over the fitted periods, and `sd_s(D | X)` that of the draw's saturated spend of the channel
less its least squares on the draw's other channels, the controls, the trend, the season and the
intercept. The return moves by the same share of itself. The bound at `Gamma` is the interval from
the `(1 - level) / 2` quantile of the draws' return less their allowance to the `(1 + level) / 2`
quantile of the return plus it; `Gamma*` is the least `Gamma` at which that interval's lower end
reaches 0, found by bisection on `kappa`, along which it falls, and read back by
`chc.ChannelSensitivity.level_at`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from chc import ChannelSensitivity
from scipy import stats

from mixdyn.models.bayes_mmm import TanhCurve

if TYPE_CHECKING:
    from mixdyn.models.bayes_mmm import BayesFit
    from mixdyn.models.ridge_mmm import RidgeFit

type _Array = npt.NDArray[np.float64]

_TANH_HALF = 0.5493061443340549
"""Where `tanh` reaches half its ceiling, `atanh(1/2) = ln(3)/2` correctly rounded. A literal
rather than `math.atanh(0.5)`, whose rounding is the platform libm's: glibc 2.39's is one ulp
low, and every number of a tanh curve moved with it there."""
_BISECTIONS = 200


def robustness_value(t: float, df: int, q: float = 1.0, alpha: float | None = None) -> float:
    """Cinelli and Hazlett's `RV_q` of a coefficient of t-statistic `t` on `df` residual degrees of
    freedom, or `RV_{q, alpha}` where `alpha` is given: the share of both residual variances a
    confounder must explain to move the estimate, or its `1 - alpha` interval's near end, by `q`
    of the estimate.

    Raises:
        ValueError: on a `t` not finite, `df` below 2, a `q` not positive, or an `alpha` outside
            (0, 1).
    """
    if not (math.isfinite(t) and df >= 2 and 0.0 < q < math.inf):
        msg = f"need a finite t, df >= 2 and a positive q, got {t}, {df} and {q}"
        raise ValueError(msg)
    fq = q * abs(t) / math.sqrt(df)
    if alpha is None:
        return _root(fq)
    if not 0.0 < alpha < 1.0:
        msg = f"alpha must lie in (0, 1), got {alpha}"
        raise ValueError(msg)
    critical = float(stats.t.ppf(1.0 - alpha / 2.0, df - 1)) / math.sqrt(df - 1)
    if fq <= critical:
        return 0.0
    if fq * critical >= 1.0:
        return (fq**2 - critical**2) / (1.0 + fq**2)
    return _root(fq - critical)


def _root(f: float) -> float:
    """The equal share `r` of `r^2 / (1 - r) = f^2`."""
    return 0.5 * (math.sqrt(f**4 + 4.0 * f**2) - f**2)


@dataclass(frozen=True, slots=True)
class BiasBound:
    """A confounder's reach on a coefficient: it explains `r2_outcome` of the KPI's residual
    variance given the channel and the design, and `r2_treatment` of the channel's given the
    design. `bias` is the most it moves the estimate, `estimate` the estimate moved by it toward
    0, `se` the adjusted standard error, and `lower` and `upper` the adjusted interval at the
    sensitivity's level. `label` names a benchmark's control and its multiple, if it is one."""

    r2_outcome: float
    r2_treatment: float
    bias: float
    estimate: float
    se: float
    lower: float
    upper: float
    label: str = ""


@dataclass(frozen=True, slots=True)
class Benchmark:
    """An observed control's partial R^2: with the channel given the rest of the design,
    `r2_treatment`, and with the KPI given the channel and the rest, `r2_outcome`."""

    control: str
    r2_treatment: float
    r2_outcome: float


@dataclass(frozen=True, slots=True, eq=False)
class RidgeSensitivity:
    """A ridge fit's channel under Cinelli and Hazlett's sensitivity analysis.

    `coefficient` is the ridge fit's; `estimate`, `se` and `dof` the least squares' on the same
    design, which the rest reads. `partial_r2` is `R2(Y ~ D | X)`; `robustness` is `RV_1`, the
    shares that bring the estimate to 0, and `robustness_half` `RV_{1/2}`, those that halve it;
    `robustness_interval` and `robustness_interval_half` are `RV_{1, alpha}` and
    `RV_{1/2, alpha}`, `alpha = 1 - level`. `benchmarks` holds each control's partial R^2.
    """

    channel: str
    coefficient: float
    estimate: float
    se: float
    dof: int
    level: float
    partial_r2: float
    robustness: float
    robustness_half: float
    robustness_interval: float
    robustness_interval_half: float
    benchmarks: tuple[Benchmark, ...]

    def bound(self, r2_outcome: float, r2_treatment: float, label: str = "") -> BiasBound:
        """The estimate and its interval under a confounder that explains `r2_outcome` of the
        KPI's residual variance and `r2_treatment` of the channel's.

        Raises:
            ValueError: on an `r2_outcome` outside [0, 1] or an `r2_treatment` outside [0, 1).
        """
        if not (0.0 <= r2_outcome <= 1.0 and 0.0 <= r2_treatment < 1.0):
            msg = (
                "a confounder's shares lie in [0, 1], its treatment's below 1, got"
                f" {r2_outcome} and {r2_treatment}"
            )
            raise ValueError(msg)
        bias = self.se * math.sqrt(self.dof * r2_outcome * r2_treatment / (1.0 - r2_treatment))
        estimate = math.copysign(abs(self.estimate) - bias, self.estimate)
        se = self.se * math.sqrt((1.0 - r2_outcome) / (1.0 - r2_treatment))
        se *= math.sqrt(self.dof / (self.dof - 1))
        half = float(stats.t.ppf(0.5 + 0.5 * self.level, self.dof - 1)) * se
        return BiasBound(
            r2_outcome=r2_outcome,
            r2_treatment=r2_treatment,
            bias=bias,
            estimate=estimate,
            se=se,
            lower=estimate - half,
            upper=estimate + half,
            label=label,
        )

    def benchmark(self, control: str, kd: float = 1.0, ky: float | None = None) -> BiasBound:
        """The bound under a confounder `kd` times as strong as `control` on the channel and `ky`
        times on the KPI, `kd` by default, and independent of it given the rest of the design;
        its shares are Cinelli and Hazlett's (2020, Section 6.1), the KPI's held to 1.

        Raises:
            ValueError: on a control the fit does not have, a multiple not positive, or one so
                large the channel would be the confounder's alone.
        """
        found = {b.control: b for b in self.benchmarks}
        ky = kd if ky is None else ky
        if control not in found or not (0.0 < kd < math.inf and 0.0 < ky < math.inf):
            msg = (
                f"need one of the controls {sorted(found)} and positive multiples, got {control!r}"
            )
            raise ValueError(msg)
        b = found[control]
        r2_treatment = kd * b.r2_treatment / (1.0 - b.r2_treatment)
        shared = kd * b.r2_treatment**2 / ((1.0 - kd * b.r2_treatment) * (1.0 - b.r2_treatment))
        if not (r2_treatment < 1.0 and 0.0 <= shared < 1.0):
            msg = f"{kd} times {control!r} would explain all of the channel's residual variance"
            raise ValueError(msg)
        r2_outcome = (math.sqrt(ky) + math.sqrt(shared)) ** 2 / (1.0 - shared)
        r2_outcome = min(1.0, r2_outcome * b.r2_outcome / (1.0 - b.r2_outcome))
        return self.bound(r2_outcome, r2_treatment, label=f"{kd:g}x {control}")


@dataclass(frozen=True, slots=True)
class _Regression:
    coefficients: _Array
    se: _Array
    dof: int


def _least_squares(y: _Array, x: _Array) -> _Regression:
    """The least squares of `y` on an intercept and the columns of `x`, of full rank."""
    design = np.column_stack([np.ones(y.size), x])
    rank = int(np.linalg.matrix_rank(design))
    if rank < design.shape[1]:
        msg = f"the design's {design.shape[1]} columns span only {rank}: a coefficient is not read"
        raise ValueError(msg)
    dof = y.size - rank
    if dof < 2:
        msg = f"{y.size} periods leave {dof} degrees of freedom; the analysis needs 2"
        raise ValueError(msg)
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    variance = float(residual @ residual) / dof * np.diag(np.linalg.inv(design.T @ design))
    return _Regression(coefficients[1:], np.sqrt(variance[1:]), dof)


def _partial_r2(t: float, dof: int) -> float:
    return t**2 / (t**2 + dof)


def ridge_sensitivity(fit: RidgeFit, channel: str, level: float = 0.9) -> RidgeSensitivity:
    """Cinelli and Hazlett's sensitivity of `channel`'s coefficient in `fit`, its intervals at
    `level`, read on the least squares of the fitted periods' KPI on the fit's design; a column
    that does not vary over them is left out with the intercept.

    Raises:
        ValueError: on a channel the fit does not have, a level outside (0, 1), a design not of
            full rank or fewer than two degrees of freedom left.
    """
    media = [m.channel for m in fit.spec.media]
    if channel not in media:
        msg = f"the fit has no channel {channel!r}; it has {media}"
        raise ValueError(msg)
    if not 0.0 < level < 1.0:
        msg = f"level must lie in (0, 1), got {level}"
        raise ValueError(msg)
    terms = list(fit.spec.terms())
    x, y = fit.design[fit.fitted], fit.y[fit.fitted]
    live = [j for j, name in enumerate(terms) if np.ptp(x[:, j]) > 0.0 or name == channel]
    names = [terms[j] for j in live]
    x = x[:, live]
    d = names.index(channel)
    full = _least_squares(y, x)
    t = float(full.coefficients[d] / full.se[d])
    rest = np.delete(x, d, axis=1)
    treatment = _least_squares(x[:, d], rest)
    benchmarks = []
    for control in (c.column for c in fit.spec.controls):
        if control not in names:
            continue
        k = names.index(control)
        k_rest = k - (k > d)
        benchmarks.append(
            Benchmark(
                control=control,
                r2_treatment=_partial_r2(
                    float(treatment.coefficients[k_rest] / treatment.se[k_rest]), treatment.dof
                ),
                r2_outcome=_partial_r2(float(full.coefficients[k] / full.se[k]), full.dof),
            )
        )
    alpha = 1.0 - level
    return RidgeSensitivity(
        channel=channel,
        coefficient=float(fit.coefficients[channel]),
        estimate=float(full.coefficients[d]),
        se=float(full.se[d]),
        dof=full.dof,
        level=level,
        partial_r2=_partial_r2(t, full.dof),
        robustness=robustness_value(t, full.dof),
        robustness_half=robustness_value(t, full.dof, 0.5),
        robustness_interval=robustness_value(t, full.dof, 1.0, alpha),
        robustness_interval_half=robustness_value(t, full.dof, 0.5, alpha),
        benchmarks=tuple(benchmarks),
    )


@dataclass(frozen=True, slots=True, eq=False)
class MsmSensitivity:
    """A Bayesian fit's channel under the marginal sensitivity model.

    `roi` is the posterior median of the channel's return on the window's spend, `lower` and
    `upper` its interval at `level` at each of `gammas`, the confounder's allowance included, and
    `allowance` the posterior median of how far the confounder can move the return at each.
    `scale` is each draw's allowance per unit of `kappa`, `sigma_s / sd_s(D | X)` times the
    draw's return over its coefficient. `gamma_star` is the least `Gamma` at which `lower` reaches
    0: 1 where the interval holds 0 without a confounder.
    """

    channel: str
    level: float
    roi: float
    gammas: _Array
    lower: _Array
    upper: _Array
    allowance: _Array
    scale: _Array
    gamma_star: float


def _saturated(fit: BayesFit, j: int) -> _Array:
    """Channel `j`'s saturated spend over the window's periods at each draw, `(draws, periods)`."""
    spec = fit.spec
    name = spec.media[j].channel
    spend = fit.spend[name]
    periods = fit.y.size
    rows = spend.size - periods + np.arange(periods)
    lag = rows[:, None] - np.arange(fit.length)
    lagged = np.where(lag >= 0, spend[np.maximum(lag, 0)], 0.0)
    retention = fit.draws["retention"][..., j].reshape(-1)
    powers = retention[:, None] ** np.arange(fit.length)
    weights = powers / powers.sum(axis=1, keepdims=True)
    adstock = weights @ lagged.T
    half = fit.draws["half_saturation"][..., j].reshape(-1)[:, None]
    if isinstance(spec.saturation, TanhCurve):
        return np.tanh(_TANH_HALF * adstock / half)
    slope = fit.draws["slope"][..., j].reshape(-1)[:, None]
    z = (adstock / half) ** slope
    return z / (1.0 + z)


def _draws_means(fit: BayesFit) -> tuple[list[_Array], _Array, _Array]:
    """Each channel's saturated spend at each draw, `(draws, periods)` each; the columns every draw
    shares, the controls' and the calendar's, `(periods, k)`; and each draw's mean, `(draws,
    periods)`."""
    spec = fit.spec
    periods = fit.y.size
    saturated = [_saturated(fit, j) for j in range(len(spec.media))]
    coefficient = fit.draws["coefficient"].reshape(-1, len(spec.media))
    mean = fit.draws["intercept"].reshape(-1, 1) + sum(
        (coefficient[:, j : j + 1] * s for j, s in enumerate(saturated)), np.zeros(periods)
    )
    shared = [np.empty((periods, 0))]
    if spec.controls:
        control = fit.draws["control"].reshape(-1, len(spec.controls))
        # a control's contribution is its column times its coefficient's posterior mean
        columns = np.column_stack(
            [
                fit.contributions[c.column] / control[:, k].mean()
                for k, c in enumerate(spec.controls)
            ]
        )
        shared.append(columns)
        mean = mean + control @ columns.T
    if spec.calendar_terms():
        columns = spec.calendar(periods)
        shared.append(columns)
        mean = mean + fit.draws["calendar"].reshape(-1, columns.shape[1]) @ columns.T
    return saturated, np.column_stack(shared), mean


def msm_sensitivity(
    posterior: BayesFit,
    channel: str,
    gammas: Sequence[float] | npt.ArrayLike = (1.0, 1.1, 1.5, 2.0, 3.0),
    level: float = 0.9,
) -> MsmSensitivity:
    """The marginal sensitivity model's bounds on `channel`'s return in `posterior` at each of
    `gammas`, its intervals at `level`, and `Gamma*`.

    Raises:
        ValueError: on a channel the fit does not have, a `Gamma` below 1 or not finite, a level
            outside (0, 1), or a channel whose saturated spend some draw's other columns span.
    """
    media = [m.channel for m in posterior.spec.media]
    if channel not in media:
        msg = f"the fit has no channel {channel!r}; it has {media}"
        raise ValueError(msg)
    gammas = np.atleast_1d(np.asarray(gammas, dtype=np.float64))
    if gammas.ndim != 1 or not (np.isfinite(gammas) & (gammas >= 1.0)).all():
        msg = f"each Gamma must be finite and at least 1, got {gammas!r}"
        raise ValueError(msg)
    if not 0.0 < level < 1.0:
        msg = f"level must lie in (0, 1), got {level}"
        raise ValueError(msg)
    j = media.index(channel)
    fitted = posterior.fitted
    saturated, shared, mean = _draws_means(posterior)
    y = posterior.y[fitted]
    sigma = np.sqrt(np.mean((y - mean[:, fitted]) ** 2, axis=1))
    spread = np.empty(sigma.size)
    for s in range(sigma.size):
        others = [saturated[k][s, fitted] for k in range(len(media)) if k != j]
        rest = np.column_stack([np.ones(y.size), shared[fitted], *others])
        d = saturated[j][s, fitted]
        residual = d - rest @ np.linalg.lstsq(rest, d, rcond=None)[0]
        spread[s] = math.sqrt(float(residual @ residual) / y.size)
    if not (spread > 0.0).all():
        msg = f"some draw's other columns span {channel!r}'s saturated spend"
        raise ValueError(msg)
    roi = posterior.draws["roi"][..., j].reshape(-1)
    coefficient = posterior.draws["coefficient"][..., j].reshape(-1)
    scale = sigma / spread * roi / coefficient
    low, high = 0.5 - 0.5 * level, 0.5 + 0.5 * level
    kappas = np.array([ChannelSensitivity.factor(float(g)) for g in gammas])
    return MsmSensitivity(
        channel=channel,
        level=level,
        roi=float(np.median(roi)),
        gammas=gammas,
        lower=np.array([np.quantile(roi - k * scale, low) for k in kappas]),
        upper=np.array([np.quantile(roi + k * scale, high) for k in kappas]),
        allowance=kappas * float(np.median(scale)),
        scale=scale,
        gamma_star=_gamma_star(roi, scale, low),
    )


def _gamma_star(roi: _Array, scale: _Array, low: float) -> float:
    """The least `Gamma` at which the `low` quantile of `roi - kappa(Gamma) scale` reaches 0."""

    def end(kappa: float) -> float:
        return float(np.quantile(roi - kappa * scale, low))

    if end(0.0) <= 0.0:
        return 1.0
    below, above = 0.0, 1.0
    while end(above) > 0.0:
        below, above = above, 2.0 * above
    for _ in range(_BISECTIONS):
        middle = 0.5 * (below + above)
        if middle in (below, above):
            break
        below, above = (middle, above) if end(middle) > 0.0 else (below, middle)
    return ChannelSensitivity.level_at(above)


__all__ = [
    "Benchmark",
    "BiasBound",
    "MsmSensitivity",
    "RidgeSensitivity",
    "msm_sensitivity",
    "ridge_sensitivity",
    "robustness_value",
]
