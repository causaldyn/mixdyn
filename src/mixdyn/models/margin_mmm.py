"""Each channel's margin read off a weekly history by regression: the KPI a unit of weekly spend
returns, with the margins' posterior covariance.

    y_t = sum_c beta_c a_c(t) + trend_t + season_t + gamma promotion_t + delta price_t
          + sum_e eta_e [t mod 52 = e] + u_t,        u_t = phi u_(t-1) + e_t

`a_c` is channel `c`'s spend through a geometric adstock of the kernel's length `L`, its weights
`alpha_c^j` over `j < L` normalised to sum 1, so `beta_c` is what a unit spent in one week returns
over the kernel: a plan that spends `p_c` a week over `n` weeks in place of `s_c` gains
`n sum_c beta_c (p_c - s_c)`, the history's carryover the same under both. The trend is a cubic
B-spline with interior knots every 26 weeks, its basis summing to 1, so there is no separate
intercept; the season is 3 harmonics of a 52-week year; the promotion flag and the price enter as
given; and each recurring promoted week of the year has a dummy: a phase `t mod 52` whose mean
price over the years seen lies below the price's median by more than 0.15 (`promoted_phases`).
A history without the flag leaves its column out, and one without the price leaves out its column
and the promoted weeks' dummies, which are read off it; the rest of the design is the same.

By default the regression is read under the noise mixture (below). **Without one**, `noise=None`,
the errors are AR(1). `phi` is read off the residuals of the same model fitted with independent
errors, their lag-1 coefficient `sum r_t r_(t-1) / sum r_(t-1)^2` clipped to `[0, 0.95]`, and the
model is fitted again on the data whitened by the Cholesky factor of the AR(1) covariance `S`,
Prais and Winsten's transform. The coefficients carry a conjugate Gaussian prior of precision
`1e-3 diag(X' S^-1 X)` over the noise variance, the noise variance a prior `1 / sigma^2`, so the
posterior mean is all but the generalised least squares' where the sales' level is small against
the noise, and not where it is large (below); the coefficients' covariance is read at the noise
variance's plug-in, the residual sum of squares over `n - p`. Each channel's retention
runs over {0.1, 0.3, 0.5, 0.7, 0.9}, every combination of the channels' fitted so, and each is
weighted by its evidence under that prior, the data's density with the coefficients and the noise
variance integrated out:

    log evidence = -1/2 log|S| + 1/2 log|Lambda| - 1/2 log|X' S^-1 X + Lambda| - n/2 log RSS

up to a constant every combination shares, `Lambda` the prior's precision and RSS
`y' S^-1 y - y' S^-1 X (X' S^-1 X + Lambda)^-1 X' S^-1 y`. The margins' posterior is the mixture's
mean and covariance over the combinations. The arithmetic follows the research code that found
the rule, step for step, so its margins are reproduced to rounding
(`tests/fixtures/margin_worlds.json`).

Where the data hold nothing that moves a channel apart from the other columns, the prior sets its
margin. Each channel's `prior_share` is the evidence weight of the combinations in which the
prior's precision on its coefficient, `Lambda`'s, exceeds the data's: the Schur complement
`1 / [(X' S^-1 X)^-1]_cc`, the whitened column's sum of squares left after its regression on every
other whitened column, so that a column the trend, the season, the controls, the events or the
other channels span counts as one no data identify. A spend that never moves is such a column:
the trend's basis sums to 1.

The kernel reaches `L - 1` weeks before each week. Each channel's spend before the history is
imputed at its mean over the history's first 13 weeks, and the first `min(L - 1, 13)` weeks are
the burn-in, left out of the regression. CHOICE: the research code left out the first `L - 1`
weeks, so a kernel as long as the history left one week and the AR(1) coefficient was 0/0; with
`L <= 14` the burn-in covers every week the imputation reaches, and the margins are the research
code's.

A fit the data leave degenerate raises `DegenerateFitError`, never a margin that is not a number:
a design with no residual degree of freedom, a residual whose lagged sum of squares is 0 or an
AR(1) coefficient that is not finite, a residual sum of squares that is not positive, and margins
that are not finite.

**The curved regression** (`fit_curved_margins`) is the same regression with each channel's
adstock column saturated, `g(a) = a / (1 + a / K_c)`, a Michaelis-Menten curve whose
half-saturation `K_c` is `HALF_SATURATION` times the channel's status quo `s_c`, so that at a
steady status quo its elasticity is `16 / 17` whatever the channel's level; `K_c` infinite is the
line, the columns `fit_margins` reads. Each retention combination is fitted and weighed as above.
Its margin at the status quo is what a unit of weekly spend more over the planned quarter returns
over the quarter and the kernel's tail, `beta_c G'_cr(s_c)`, with

    G_cr(p) = 1/n sum_h g(carry_crh + p reach_rh)

over the `n` planned weeks and the `L - 1` after them: `carry` the history's spend carried into
week `h` through the kernel of retention `r`, the spend before the history imputed as above, and
`reach` what a unit spent in every planned week reaches week `h`. The margins' posterior is the
mixture's mean and covariance over the combinations of `beta_c G'_cr(s_c)`; each channel's curve
for planning (`Curve`) is the evidence-weighted mixture over its retentions of `G_cr / G'_cr(s_c)`,
whose slope at the status quo is 1.

**Without a mixture the sales' level moves the margins.** The prior covers the trend's basis, which
sums to 1 and so carries the level, and pulls the level towards 0. A channel's column, its mean the
spend's, takes up part of it, the more the smoother the column: the evidence leans to the high
retentions, and the level left over is slow, which the AR(1) coefficient reads as persistence.
At one retention combination and one AR(1) coefficient, sales `y + c` give the coefficients of `y`
plus `c (e - (X' S^-1 X + Lambda)^-1 Lambda e)`, `e` the trend's coefficients of the constant: the
channels' move by the second term, which a prior on the trend makes nonzero. On the README's
history, its sales 200 noise deviations up, margins of 1.5, 3.0 and 0.5 are read as 4.48, 8.30 and
0.20 at retentions 0.90, 0.90 and 0.20; the same history at a level of 0 reads 1.41, 2.20 and 0.16,
and the noise mixture reads 1.44, 2.19 and 0.21 at either level. The readers frozen on this
regression keep it by `noise=None` (ADR 0097).

**The noise mixture** (`NoiseMixture`), the default. Under it `fit_margins` and
`fit_curved_margins` read the same mean under another prior and another error model; with
`noise=None` they fit as above, to the bit.

- The trend's interior knots lie every `knot_every` weeks, 52.
- The nuisance columns, the trend, the season, the controls and the promoted weeks' dummies, carry
  a flat prior, so the sales' level is never penalised. A nuisance column that is 0 on every week
  read is dropped: the flat prior leaves its coefficient free, and the data say nothing of it.
- Each channel's coefficient keeps a Gaussian prior. Its precision is `PRIOR` times the whitened
  sum of squares of the channel's straight adstock column times `g'(s_c)^2`, the squared slope of
  its curve at a steady status quo, `(kappa / (1 + kappa))^4` at a half-saturation of `kappa`
  status quos. So every curve puts the same prior on the margin there, the line's.
- The errors are a stationary AR(1) of coefficient `rho` plus white noise, the AR(1)'s share of
  their variance `omega`, and their variance on a promoted phase's weeks is `h` times the other
  weeks':

      cov(u_t, u_s) = sigma^2 d_t d_s (omega rho^|t - s| + (1 - omega) [t = s]),  d_t^2 = h or 1

  `(rho, omega)` runs over the pairs of two grids, with white errors once more, and `h` over its
  own grid. The regression is whitened by the Cholesky factor of each covariance, so each whitened
  week reads the weeks before it alone.

Every error model and every retention combination is weighted by its evidence under a uniform
prior over them, the nuisance coefficients and the noise's scale `sigma^2` integrated out:

    log evidence = -1/2 log|S| + 1/2 sum_c log Lambda_c - 1/2 log|X' S^-1 X + Lambda|
                   - (n - q)/2 log RSS

up to a constant every model shares, `q` the nuisance columns kept and `Lambda` the channels'
precisions, 0 on the nuisance columns. It is the same on any common scale of `S`
(`validation/margin_noise.mac`), so the error models are compared as their covariances are
written. The coefficients' covariance is read at the noise variance's plug-in, RSS over `n - p`.
The margins' posterior is the mixture's mean and covariance over the error models and the
combinations; `MarginFit.phi`, `share` and `promoted` are the posterior means of `rho`, `omega`
and `h`, and `top` the combination of the most weight over the error models. A channel's
`prior_share` weighs the models in which the prior's precision exceeds what the data add to it,
`1 / [(X' S^-1 X + Lambda)^-1]_cc - Lambda_c`: the Schur complement above with the other channels'
coefficients under their priors. Nuisance columns the weeks read do not tell apart, the price that
never moves beside the trend, leave the flat prior improper, and a model whose residual sum of
squares is not positive has no evidence: both raise `DegenerateFitError`. Columns the weeks barely
tell apart, or a channel that never spent, can still fail a factorisation, a
`numpy.linalg.LinAlgError`, which is a `ValueError`.

The fits are stacked, every error model's of every combination: under the default grids `124 5^C`
of them for `C` channels, whose covariances alone take `124 5^C C^2` doubles, 0.52 GiB at six
channels, 3.5 GiB at seven and 23 GiB at eight. A history of more channels is read under a smaller
grid, the memory falling with the error models' count, or with `noise=None`.

A channel whose columns the nuisance columns span, a spend that never moves beside the trend, has
under the flat prior a coefficient its prior alone sets: the data's part of its posterior is the
nuisance coefficients', so its posterior mean is exactly 0 in every model, and floating point
leaves a rounding residue of either sign in its place. Where every one of a channel's columns lies
within `SPANNED` of its length of the nuisance columns' span, its coefficient is 0 in every model
and its margin exactly 0, never a residue whose sign a reader could take for the data's;
`MarginFit.spanned` names the channel. Its variance is its prior's.

**The regression as a linear map** (`project`). At one retention combination, with the AR(1)
coefficient the sales give held, the posterior mean is linear in the sales:
`(X' S^-1 X + Lambda)^-1 X' S^-1 y`. Put a series in the sales' place and the map reads the
coefficients the regression would give it: with a curve of a channel's adstock there, the slope a
straight line through the history reads off that curve (`Projection.slope`).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.interpolate import BSpline
from scipy.linalg import cholesky, solve_triangular

type _Array = npt.NDArray[np.float64]

RETENTIONS = (0.1, 0.3, 0.5, 0.7, 0.9)
"""Each channel's retentions; every combination of the channels' is fitted and weighed."""
PRIOR = 1e-3
"""The prior's precision of each coefficient over the noise's, a share of its column's whitened
sum of squares; a column of none reads 1."""
YEAR = 52
"""The weeks of the season and of the promotions' cycle."""
HARMONICS = 3
"""The season's harmonics, a sine and a cosine each."""
KNOT_EVERY = 26
"""The weeks between the trend's interior knots."""
DEGREE = 3
"""The trend's B-spline degree."""
PRICE_DROP = 0.15
"""How far below the price's median a phase's mean price lies to be a promoted week."""
AR_MOST = 0.95
"""The AR(1) coefficient's upper bound; its lower is 0."""
BURN_IN = 13
"""The most weeks left out of the regression at the history's start."""
PRESAMPLE = 13
"""The history's first weeks, whose mean spend stands in for the spend before it."""
HALF_SATURATION = 16.0
"""The curved regression's half-saturation of each channel over its status quo."""
PERSISTENCE = (0.3, 0.6, 0.8, 0.9, 0.95, 0.98)
"""The noise mixture's AR(1) coefficients."""
SHARE = (0.5, 0.7, 0.85, 0.95, 1.0)
"""The noise mixture's shares of the errors' variance the AR(1) holds; the rest is white."""
PROMOTED = (1.0, 2.0, 4.0, 8.0)
"""The noise mixture's ratios of the errors' variance on a promoted week to another week's."""
MIXTURE_KNOT_EVERY = 52
"""The weeks between the trend's interior knots under a noise mixture."""
SPANNED = 1e-8
"""Under a noise mixture, how far apart from the nuisance columns' span a column the span holds
may lie, over its length: the square root of the rounding unit. Rounding leaves a column in the
span 1e-16 apart, and a column further apart than this has a coefficient the data set to 1e-8 of
it or better."""


class DegenerateFitError(ValueError):
    """The data leave the margin regression degenerate; the message says how."""


@dataclass(frozen=True, slots=True)
class NoiseMixture:
    """The regression's settings under a mixture of error models (the module): the weeks between
    the trend's interior knots, `knot_every`; the AR(1) coefficients `persistence` and the AR(1)'s
    shares of the errors' variance `share`, each pair of them an error model, with white errors
    once more where `white`; and the promoted weeks' variance ratios `promoted`, each crossed with
    every one of those.

    Raises:
        ValueError: on a knot spacing below a week, a coefficient outside (0, 1), a share outside
            (0, 1], a ratio that is not positive and finite, or no error model.
    """

    knot_every: int = MIXTURE_KNOT_EVERY
    persistence: tuple[float, ...] = PERSISTENCE
    share: tuple[float, ...] = SHARE
    promoted: tuple[float, ...] = PROMOTED
    white: bool = True

    def __post_init__(self) -> None:
        if (
            self.knot_every < 1
            or not all(0.0 < rho < 1.0 for rho in self.persistence)
            or not all(0.0 < omega <= 1.0 for omega in self.share)
            or not all(0.0 < h < math.inf for h in self.promoted)
            or not self.models()
        ):
            msg = (
                "a noise mixture has knots a week apart or more, AR(1) coefficients in (0, 1),"
                " shares in (0, 1], positive finite variance ratios and an error model, not"
                f" {self}"
            )
            raise ValueError(msg)

    def pairs(self) -> list[tuple[float, float]]:
        """Each error model's `(rho, omega)`: white errors first where `white`, then every
        coefficient with every share."""
        white = [(0.0, 1.0)] if self.white else []
        return white + [(rho, omega) for rho in self.persistence for omega in self.share]

    def models(self) -> list[tuple[float, float, float]]:
        """Every error model's `(rho, omega, h)`, in the order the fits are stacked."""
        return [(rho, omega, h) for rho, omega in self.pairs() for h in self.promoted]


MIXTURE = NoiseMixture()
"""The mixture of error models `fit_margins` and `fit_curved_margins` read under by default:
the regression's own grids."""


@dataclass(frozen=True, slots=True, eq=False)
class MarginFit:
    """The margins' posterior, the mixture over the retention combinations.

    `mean` is each channel's margin, the KPI a unit of weekly spend returns over the kernel, and
    `cov` their covariance, `(channels, channels)`. `retention` is each channel's posterior mean
    retention and `top` the combination of the most evidence, `top_weight` its weight. `phi` is
    the AR(1) coefficient's mean over the combinations, by weight. `phases` are the promoted
    phases, `t mod 52`, each a dummy. `rows` are the weeks the regression read, after `burn_in`,
    and `columns` its design's. `prior_share` is each channel's evidence weight of the
    combinations in which the prior's precision on its coefficient exceeds the data's (the module).
    `spanned` are the channels whose columns the nuisance columns span under a noise mixture, their
    margin exactly 0 (the module); without one, none.
    Under a noise mixture the weights run over the error models too: `phi` is the posterior mean
    of `rho`, `share` that of `omega`, the AR(1)'s share of the errors' variance, and `promoted`
    that of `h`, the promoted weeks' variance over the other weeks'; without one the errors are
    the AR(1) alone on every week, and both are 1.
    """

    mean: _Array
    cov: _Array
    retention: _Array
    top: tuple[float, ...]
    top_weight: float
    phi: float
    phases: tuple[int, ...]
    rows: int
    burn_in: int
    columns: int
    prior_share: _Array
    spanned: npt.NDArray[np.bool_]
    share: float = 1.0
    promoted: float = 1.0


@dataclass(frozen=True, slots=True, eq=False)
class Curve:
    """Each channel's curve for planning, `H_c(p) = sum_r w_cr G_cr(p) / G'_cr(s_c)` over its
    retentions `r` of evidence weight `w_cr` above 0 (the module), one term a pair of a channel
    and a retention: the pairs' `channel`, `carry` and `reach` over the horizon, `half_saturation`
    `(pairs, 1)`, and `scale`, `w_cr / G'_cr(s_c) / n`; `channels`, the channels' count."""

    channel: npt.NDArray[np.intp]
    carry: _Array
    reach: _Array
    half_saturation: _Array
    scale: _Array
    channels: int

    def slope(self, spend: npt.ArrayLike) -> _Array:
        """`H'_c(p_c)` for each channel at the weekly spends `spend`, `(channels,)`."""
        p = np.asarray(spend, dtype=float)
        a = self.carry + p[self.channel][:, None] * self.reach
        terms = self.scale * np.sum(self.reach * _saturated_slope(a, self.half_saturation), axis=1)
        return np.bincount(self.channel, weights=terms, minlength=self.channels)

    def rise(self, spend: npt.ArrayLike, status_quo: npt.ArrayLike) -> _Array:
        """`H_c(p_c) - H_c(s_c)` for each channel, from the status quo `status_quo` to the weekly
        spends `spend`, `(channels,)`."""
        p = np.asarray(spend, dtype=float)
        s = np.asarray(status_quo, dtype=float)
        at = self.carry + p[self.channel][:, None] * self.reach
        quo = self.carry + s[self.channel][:, None] * self.reach
        gained = _saturated(at, self.half_saturation) - _saturated(quo, self.half_saturation)
        terms = self.scale * np.sum(gained, axis=1)
        return np.bincount(self.channel, weights=terms, minlength=self.channels)


@dataclass(frozen=True, slots=True, eq=False)
class CurvedFit:
    """The curved regression (the module): `margins`, the margins at the status quo, the KPI a
    unit of weekly spend more over the planned quarter returns, with their covariance and the
    fit's diagnostics as `fit_margins` reads them; each channel's `half_saturation` `K_c`, infinite
    for a line; `weights`, each channel's evidence weight of each retention,
    `(channels, retentions)`; `adstock`, each channel's adstock over the history at each retention,
    `(channels, retentions, weeks)`; and `curve`, each channel's curve for planning."""

    margins: MarginFit
    half_saturation: _Array
    weights: _Array
    adstock: _Array
    curve: Curve

    @property
    def shaped(self) -> npt.NDArray[np.bool_]:
        """The channels whose curve is not a line: a finite half-saturation."""
        return np.isfinite(self.half_saturation)

    def planning_adstock(self) -> _Array:
        """Each channel's adstock over the history at its retention of the most evidence weight,
        the first of equal weights, `(weeks, channels)`."""
        top = np.argmax(self.weights, axis=1)
        return self.adstock[np.arange(top.size), top].T


@dataclass(frozen=True, slots=True, eq=False)
class _History:
    """What every retention combination's regression reads: the nuisance columns and the sales
    on the weeks read, each channel's adstock over every week at each retention, the spend and
    the spend imputed before it, and what a fit records of its design."""

    nuisance: _Array
    sales: _Array
    spend: _Array
    before: _Array
    adstocks: dict[tuple[int, float], _Array]
    rows: npt.NDArray[np.intp]
    phases: tuple[int, ...]
    burn_in: int
    columns: int


@dataclass(frozen=True, slots=True, eq=False)
class _Combination:
    """One retention combination's fit: the channels' coefficients and their covariance, its log
    evidence, its AR(1) coefficient, and the channels whose coefficient the prior sets."""

    mean: _Array
    cov: _Array
    evidence: float
    phi: float
    dominated: npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True, eq=False)
class _Models:
    """Every error model's fit of every retention combination, stacked an error model after
    another, the combinations in their order within each: the log `evidence`, `(fits,)`; the
    channels' coefficients `mean`, `(fits, channels)`, and their covariance `cov`; the channels
    whose coefficient the prior sets, `dominated`; each fit's error model, `(rho, omega, h)` a row
    of `errors`; and the channels whose columns the nuisance columns span, `spanned`,
    `(channels,)`."""

    evidence: _Array
    mean: _Array
    cov: _Array
    dominated: npt.NDArray[np.bool_]
    errors: _Array
    spanned: npt.NDArray[np.bool_]


def fit_margins(
    sales: npt.ArrayLike,
    spend: npt.ArrayLike,
    promotion: npt.ArrayLike | None,
    price: npt.ArrayLike | None,
    kernel: int,
    *,
    noise: NoiseMixture | None = MIXTURE,
) -> MarginFit:
    """Each channel's margin over the weekly history: `sales` `(weeks,)`, `spend`
    `(weeks, channels)`, the `promotion` flag and the `price` `(weeks,)`, either None where the
    history holds none, and the adstock's `kernel` length in weeks; under the mixture of error
    models `noise`, `MIXTURE` by default, and with None as the research code fitted it, whose
    margins the sales' level moves. See the module.

    Raises:
        ValueError: on arrays of other shapes, a value that is not finite, or a kernel shorter
            than a week; under `noise`, as `numpy.linalg.cholesky` raises too (the module).
        DegenerateFitError: where the data leave the fit degenerate (the module).
    """
    history = _read(sales, spend, promotion, price, kernel, noise)
    channels = history.spend.shape[1]
    carried = {key: series[history.rows] for key, series in history.adstocks.items()}
    combos = list(itertools.product(RETENTIONS, repeat=channels))
    if noise is not None:
        lines = np.array([[carried[c, a] for a in RETENTIONS] for c in range(channels)])
        models = _noise_fits(history, lines, lines, 1.0, noise)
        margins, _ = _noise_mixture(history, combos, models, models.mean, models.cov)
        return margins
    fits = [
        _combination(history, [carried[c, a] for c, a in enumerate(combo)], combo)
        for combo in combos
    ]
    margins, _ = _mixture(history, combos, fits, [f.mean for f in fits], [f.cov for f in fits])
    return margins


def fit_curved_margins(
    sales: npt.ArrayLike,
    spend: npt.ArrayLike,
    promotion: npt.ArrayLike | None,
    price: npt.ArrayLike | None,
    kernel: int,
    status_quo: npt.ArrayLike,
    planned: int,
    half_saturation: float = HALF_SATURATION,
    *,
    noise: NoiseMixture | None = MIXTURE,
) -> CurvedFit:
    """The curved regression over the weekly history, as `fit_margins` reads it, each channel's
    column saturated at `half_saturation` times its weekly `status_quo` `(channels,)`, infinity
    for the line, and its margin at the status quo over the `planned` weeks; under the mixture of
    error models `noise`, `MIXTURE` by default, and with None as the research code fitted it. See
    the module.

    Raises:
        ValueError: as `fit_margins` raises, and on a status quo of another shape or not positive
            and finite, no planned week, or a half-saturation not positive.
        DegenerateFitError: where the data leave the fit degenerate (the module).
    """
    history = _read(sales, spend, promotion, price, kernel, noise)
    channels = history.spend.shape[1]
    s = np.asarray(status_quo, dtype=float)
    if s.shape != (channels,) or not (np.all(np.isfinite(s)) and np.all(s > 0.0)):
        msg = f"the status quo is (channels,), positive and finite, not {s.tolist()}"
        raise ValueError(msg)
    if planned < 1 or not half_saturation > 0.0:
        msg = f"a planned week and a positive half-saturation, not {planned} and {half_saturation}"
        raise ValueError(msg)
    half = half_saturation * s
    curved = {
        (c, a): _saturated(series[history.rows], float(half[c]))
        for (c, a), series in history.adstocks.items()
    }
    combos = list(itertools.product(RETENTIONS, repeat=channels))
    fits, models = [], None
    if noise is None:
        fits = [
            _combination(history, [curved[c, a] for c, a in enumerate(combo)], combo)
            for combo in combos
        ]
    else:
        columns = np.array([[curved[c, a] for a in RETENTIONS] for c in range(channels)])
        lines = np.array(
            [[history.adstocks[c, a][history.rows] for a in RETENTIONS] for c in range(channels)]
        )
        steady = 1.0 if math.isinf(half_saturation) else half_saturation / (1.0 + half_saturation)
        models = _noise_fits(history, columns, lines, steady**4, noise)
    carry, reach = _quarter(history.spend, kernel, history.before, planned)
    path = carry + s[:, None, None] * reach[None]
    at_quo = np.sum(reach[None] * _saturated_slope(path, half[:, None, None]), axis=-1) / planned
    place = {a: r for r, a in enumerate(RETENTIONS)}
    index = np.array([[place[a] for a in combo] for combo in combos])
    scale = at_quo[np.arange(channels)[None, :], index]
    if models is None:
        margins, w = _mixture(
            history,
            combos,
            fits,
            [f.mean * d for f, d in zip(fits, scale, strict=True)],
            [f.cov * d[:, None] * d[None, :] for f, d in zip(fits, scale, strict=True)],
        )
    else:
        repeats = models.evidence.size // len(combos)
        each = np.tile(scale, (repeats, 1))
        margins, w = _noise_mixture(
            history,
            combos,
            models,
            models.mean * each,
            models.cov * each[:, :, None] * each[:, None, :],
        )
        index = np.tile(index, (repeats, 1))
    weights = np.zeros((channels, len(RETENTIONS)))
    for c in range(channels):
        np.add.at(weights[c], index[:, c], w)
    pairs, kept = np.nonzero(weights > 0.0)
    curve = Curve(
        channel=pairs,
        carry=carry[pairs, kept],
        reach=reach[kept],
        half_saturation=half[pairs][:, None],
        scale=weights[pairs, kept] / at_quo[pairs, kept] / planned,
        channels=channels,
    )
    adstocks = np.array([[history.adstocks[c, a] for a in RETENTIONS] for c in range(channels)])
    return CurvedFit(margins, half, weights, adstocks, curve)


@dataclass(frozen=True, slots=True, eq=False)
class Projection:
    """The regression at one retention combination as a linear map (the module): `rows`, the
    weeks read; `factor`, the Cholesky factor of the AR(1) covariance at the coefficient the sales
    give; `solve`, the map `(X' X + Lambda)^-1 X'` on whitened data, `(columns, rows)`; and
    `retention`, the combination."""

    rows: npt.NDArray[np.intp]
    factor: _Array
    solve: _Array
    retention: tuple[float, ...]

    def slope(self, channel: int, series: npt.ArrayLike) -> float:
        """The coefficient the regression reads on `channel`'s column off `series`, `(weeks,)`,
        in the sales' place."""
        whitened = solve_triangular(
            self.factor, np.asarray(series, dtype=float)[self.rows], lower=True
        )
        return float((self.solve @ whitened)[channel - len(self.retention)])


def project(
    sales: npt.ArrayLike,
    spend: npt.ArrayLike,
    promotion: npt.ArrayLike | None,
    price: npt.ArrayLike | None,
    kernel: int,
    retention: tuple[float, ...],
) -> Projection:
    """The regression `fit_margins` runs without a noise mixture, `noise=None`, at the retention
    combination `retention`, one of `RETENTIONS` a channel, as a linear map; see the module.

    Raises:
        ValueError: as `fit_margins` raises, and on a combination of another length or a
            retention not among `RETENTIONS`.
        DegenerateFitError: where the data leave the combination's fit degenerate.
    """
    history = _read(sales, spend, promotion, price, kernel)
    combo = tuple(float(a) for a in retention)
    if len(combo) != history.spend.shape[1] or not set(combo) <= set(RETENTIONS):
        msg = f"one retention of {RETENTIONS} a channel, not {retention}"
        raise ValueError(msg)
    columns = [history.adstocks[c, a][history.rows] for c, a in enumerate(combo)]
    design = np.column_stack([history.nuisance, np.column_stack(columns)])
    b, *_ = _ridge(design, history.sales)
    phi = _ar1(history.sales - design @ b)
    factor = np.linalg.cholesky(_ar_cov(design.shape[0], phi))
    xw = solve_triangular(factor, design, lower=True)
    gram = xw.T @ xw
    d = np.diag(gram).copy()
    d[d <= 0] = 1.0
    return Projection(history.rows, factor, np.linalg.solve(gram + np.diag(PRIOR * d), xw.T), combo)


def _read(
    sales: npt.ArrayLike,
    spend: npt.ArrayLike,
    promotion: npt.ArrayLike | None,
    price: npt.ArrayLike | None,
    kernel: int,
    noise: NoiseMixture | None = None,
) -> _History:
    """The history checked and read into what every combination's regression reads; under `noise`
    the trend's knots are its own and the nuisance columns that are 0 on the weeks read are
    dropped.

    Raises:
        ValueError: as `fit_margins` raises on its input.
        DegenerateFitError: on a design with no residual degree of freedom, and under `noise` on
            nuisance columns the weeks read do not tell apart.
    """
    y = np.asarray(sales, dtype=float)
    x = np.asarray(spend, dtype=float)
    flag = None if promotion is None else np.asarray(promotion, dtype=float)
    cost = None if price is None else np.asarray(price, dtype=float)
    _check(y, x, flag, cost, kernel)
    weeks, channels = x.shape
    phases = [] if cost is None else promoted_phases(cost)
    controls = sum(a is not None for a in (flag, cost))
    burn_in = min(kernel - 1, BURN_IN)
    n = weeks - burn_in
    every = KNOT_EVERY if noise is None else noise.knot_every
    knots = len(range(every, weeks - 1, every))
    p = knots + DEGREE + 1 + 2 * HARMONICS + controls + len(phases) + channels
    if noise is None and n - p < 1:
        raise _no_freedom(n, burn_in, p)
    rows = np.arange(burn_in, weeks)
    zr, yr = _nuisance(weeks, flag, cost, phases, every)[rows], y[rows]
    assert zr.shape[1] + channels == p  # the count above is the design's
    if noise is not None:
        zr = zr[:, np.any(zr != 0.0, axis=0)]
        p = zr.shape[1] + channels
        if n - p < 1:
            raise _no_freedom(n, burn_in, p)
        if np.linalg.matrix_rank(zr) < zr.shape[1]:
            msg = (
                f"the {n} weeks read do not tell the design's {zr.shape[1]} nuisance columns apart,"
                " and their flat prior leaves the fit improper"
            )
            raise DegenerateFitError(msg)
    before = x[:PRESAMPLE].mean(axis=0)
    adstocks = {
        (c, a): adstock(x[:, c], a, kernel, float(before[c]))
        for c in range(channels)
        for a in RETENTIONS
    }
    return _History(zr, yr, x, before, adstocks, rows, tuple(phases), burn_in, p)


def _no_freedom(n: int, burn_in: int, p: int) -> DegenerateFitError:
    return DegenerateFitError(
        f"{n} weeks after a burn-in of {burn_in} leave no residual degree of freedom for the"
        f" design's {p} columns"
    )


def _combination(
    history: _History, columns: list[_Array], combo: tuple[float, ...]
) -> _Combination:
    """The regression at one retention combination, the channels' `columns` beside the nuisance
    ones; see the module.

    Raises:
        DegenerateFitError: where the data leave the fit degenerate.
    """
    yr = history.sales
    channels = len(columns)
    design = np.column_stack([history.nuisance, np.column_stack(columns)])
    n, p = design.shape
    b, *_ = _ridge(design, yr)
    phi = _ar1(yr - design @ b)
    factor = np.linalg.cholesky(_ar_cov(n, phi))
    logdet = 2.0 * float(np.sum(np.log(np.diag(factor))))
    xw = solve_triangular(factor, design, lower=True)
    yw = solve_triangular(factor, yr, lower=True)
    b, chol, z, lam = _ridge(xw, yw)
    rss = float(yw @ yw - z @ z)
    if not rss > 0.0:
        msg = f"the whitened fit at retentions {combo} leaves a residual sum of squares of {rss}"
        raise DegenerateFitError(msg)
    evidence = (
        -0.5 * logdet
        + 0.5 * float(np.sum(np.log(lam)))
        - float(np.sum(np.log(np.diag(chol))))
        - 0.5 * n * math.log(rss)
    )
    inverse = solve_triangular(chol, np.eye(chol.shape[0]), lower=True)
    return _Combination(
        b[-channels:],
        rss / (n - p) * (inverse.T @ inverse)[-channels:, -channels:],
        evidence,
        phi,
        lam[-channels:] > _schur(xw, channels),
    )


def _mixture(
    history: _History,
    combos: list[tuple[float, ...]],
    fits: list[_Combination],
    means: list[_Array],
    covs: list[_Array],
) -> tuple[MarginFit, _Array]:
    """The mixture over the combinations of the margins `means` of covariances `covs`, each
    weighed by its fit's evidence, with the fits' diagnostics; and the weights.

    Raises:
        DegenerateFitError: on margins or a covariance that are not all finite.
    """
    w, mean, cov = _weigh(np.array([f.evidence for f in fits]), np.array(means), np.array(covs))
    top = int(np.argmax(w))
    fit = MarginFit(
        mean=mean,
        cov=cov,
        retention=w @ np.array(combos),
        top=combos[top],
        top_weight=float(w[top]),
        phi=float(w @ np.array([f.phi for f in fits])),
        phases=history.phases,
        rows=history.rows.size,
        burn_in=history.burn_in,
        columns=history.columns,
        prior_share=w @ np.array([f.dominated for f in fits], dtype=float),
        spanned=np.zeros(mean.size, dtype=bool),
    )
    return fit, w


def _weigh(logs: _Array, means: _Array, covs: _Array) -> tuple[_Array, _Array, _Array]:
    """The fits' weights by their log evidence `logs`, and the mixture's mean and covariance of
    their margins `means`, `(fits, channels)`, of covariances `covs`.

    Raises:
        DegenerateFitError: on margins or a covariance that are not all finite.
    """
    w = np.exp(logs - logs.max())
    w /= w.sum()
    mean = w @ means
    second = np.einsum("k,kij->ij", w, covs + np.einsum("ki,kj->kij", means, means))
    cov = second - np.outer(mean, mean)
    cov = 0.5 * (cov + cov.T)
    if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(cov))):
        msg = f"the margins {mean.tolist()} or their covariance are not all finite"
        raise DegenerateFitError(msg)
    return w, mean, cov


def _noise_fits(
    history: _History, columns: _Array, lines: _Array, steady: float, noise: NoiseMixture
) -> _Models:
    """Under each of `noise`'s error models, the regression at every retention combination: the
    channels' `columns`, `(channels, retentions, weeks read)`, beside the nuisance ones, each
    coefficient's prior precision `PRIOR` times `steady` times the whitened sum of squares of its
    straight column in `lines`. Each error model is whitened once, and every combination's
    regression is read off the whitened columns' cross-products. A channel whose columns the
    nuisance columns span has a coefficient of exactly 0 in every fit; see the module.

    Raises:
        DegenerateFitError: where a fit's residual sum of squares is not positive.
        numpy.linalg.LinAlgError: where a fit's cross-products are not positive definite.
    """
    channels, count, n = columns.shape
    q = history.nuisance.shape[1]
    p = q + channels
    design = np.column_stack(
        [history.nuisance, columns.reshape(-1, n).T, lines.reshape(-1, n).T, history.sales]
    )
    index = np.array(list(itertools.product(range(count), repeat=channels)))
    column = q + np.arange(channels)[None, :] * count + index
    line = column + channels * count
    kept = np.concatenate([np.broadcast_to(np.arange(q), (index.shape[0], q)), column], axis=1)
    sales = design.shape[1] - 1
    unit = np.zeros((p, channels))
    unit[q:] = np.eye(channels)
    week = np.arange(n)
    lag = np.abs(week[:, None] - week[None, :])
    promoted = np.isin(history.rows % YEAR, history.phases)
    spanned = _apart(history.nuisance, columns) <= SPANNED
    evidence, mean, cov, dominated = [], [], [], []
    for rho, omega in noise.pairs():
        factor = cholesky(omega * rho**lag + (1.0 - omega) * np.eye(n), lower=True)
        for h in noise.promoted:
            sd = np.where(promoted, math.sqrt(h), 1.0)
            logdet = 2.0 * float(np.sum(np.log(np.diag(factor)))) + 2.0 * float(np.sum(np.log(sd)))
            white = solve_triangular(factor, design / sd[:, None], lower=True)
            gram = white.T @ white
            lam = np.zeros(kept.shape)
            lam[:, q:] = PRIOR * gram[line, line] * steady
            a = gram[kept[:, :, None], kept[:, None, :]] + lam[:, :, None] * np.eye(p)[None]
            chol = np.linalg.cholesky(a)
            xy = gram[kept, sales]
            rhs = [xy[:, :, None], np.broadcast_to(unit, (kept.shape[0], p, channels))]
            solved = np.linalg.solve(a, np.concatenate(rhs, axis=2))
            b = solved[:, :, 0]
            rss = gram[sales, sales] - np.einsum("mi,mi->m", xy, b)
            if not np.all(rss > 0.0):
                msg = (
                    f"the whitened fit at the error model {(rho, omega, h)} leaves a residual sum"
                    f" of squares of {float(rss.min())}"
                )
                raise DegenerateFitError(msg)
            evidence.append(
                -0.5 * logdet
                + 0.5 * np.sum(np.log(lam[:, q:]), axis=1)
                - np.sum(np.log(np.diagonal(chol, axis1=1, axis2=2)), axis=1)
                - 0.5 * (n - q) * np.log(rss)
            )
            inverse = solved[:, q:, 1:]
            spread = (rss / (n - p))[:, None, None] * inverse
            mean.append(np.where(spanned, 0.0, b[:, q:]))
            cov.append(0.5 * (spread + np.transpose(spread, (0, 2, 1))))
            own = np.diagonal(inverse, axis1=1, axis2=2)
            dominated.append(lam[:, q:] > 1.0 / own - lam[:, q:])
    return _Models(
        np.concatenate(evidence),
        np.concatenate(mean),
        np.concatenate(cov),
        np.concatenate(dominated),
        np.repeat(np.array(noise.models()), index.shape[0], axis=0),
        spanned,
    )


def _apart(nuisance: _Array, columns: _Array) -> _Array:
    """How far each channel's `columns`, `(channels, retentions, weeks read)`, lie apart from the
    `nuisance` columns' span: the largest over its retentions of the column's residual after its
    least squares on the nuisance columns, over the column's length; 0 for a column of zeros. The
    span is the same under every error model, whose whitening maps the weeks one to one."""
    basis, _ = np.linalg.qr(nuisance / np.linalg.norm(nuisance, axis=0))
    left = columns - (columns @ basis) @ basis.T
    length = np.linalg.norm(columns, axis=2)
    apart = np.zeros(length.shape)
    np.divide(np.linalg.norm(left, axis=2), length, out=apart, where=length > 0.0)
    return apart.max(axis=1)


def _noise_mixture(
    history: _History,
    combos: list[tuple[float, ...]],
    models: _Models,
    means: _Array,
    covs: _Array,
) -> tuple[MarginFit, _Array]:
    """The mixture over `models`' fits of the margins `means` of covariances `covs`, each weighed
    by its evidence, with the fits' diagnostics; and the weights, `(fits,)`.

    Raises:
        DegenerateFitError: on margins or a covariance that are not all finite.
    """
    w, mean, cov = _weigh(models.evidence, means, covs)
    by_combo = w.reshape(-1, len(combos)).sum(axis=0)
    top = int(np.argmax(by_combo))
    rho, omega, h = w @ models.errors
    fit = MarginFit(
        mean=mean,
        cov=cov,
        retention=by_combo @ np.array(combos),
        top=combos[top],
        top_weight=float(by_combo[top]),
        phi=float(rho),
        phases=history.phases,
        rows=history.rows.size,
        burn_in=history.burn_in,
        columns=history.columns,
        prior_share=w @ models.dominated.astype(float),
        spanned=models.spanned,
        share=float(omega),
        promoted=float(h),
    )
    return fit, w


def adstock(spend: _Array, retention: float, length: int, before: float) -> _Array:
    """`spend`, `(weeks,)`, through the geometric kernel of `retention` over `length` weeks,
    normalised to sum 1, the spend before the first week `before` in each of the `length - 1`
    weeks the kernel reaches back. A week from the `length`-th on reads the history alone, to the
    bit."""
    w = retention ** np.arange(length)
    w = w / w.sum()
    out = np.convolve(spend, w)[: spend.size]
    head = min(length - 1, spend.size)
    if head > 0:
        out[:head] += before * np.cumsum(w[::-1])[::-1][1 : head + 1]
    return out


def _saturated(a: _Array, half: float | _Array) -> _Array:
    """The Michaelis-Menten curve `a / (1 + a / K)`: `a` itself where `K` is infinite."""
    return a / (1.0 + a / half)


def _saturated_slope(a: _Array, half: float | _Array) -> _Array:
    """The curve's slope, `1 / (1 + a / K)^2`."""
    return 1.0 / (1.0 + a / half) ** 2


def _quarter(spend: _Array, length: int, before: _Array, planned: int) -> tuple[_Array, _Array]:
    """Over the `planned` weeks and the kernel's `length - 1` after them, for each retention:
    each channel's history carried into each week through the kernel, the spend before the
    history `before`, `(channels, retentions, horizon)`, and what a unit spent in every planned
    week reaches each week, `(retentions, horizon)`."""
    weeks, channels = spend.shape
    horizon = planned + length - 1
    h = np.arange(horizon)[:, None]
    j = np.arange(length)[None, :]
    source = weeks + h - j
    during = ((h - j) >= 0) & ((h - j) < planned)
    carry = np.zeros((channels, len(RETENTIONS), horizon))
    reach = np.zeros((len(RETENTIONS), horizon))
    for r, a in enumerate(RETENTIONS):
        k = a ** np.arange(length)
        k = k / k.sum()
        reach[r] = (during * k[None, :]).sum(axis=1)
        for c in range(channels):
            spent = np.where(
                source < weeks,
                np.where(source >= 0, spend[np.clip(source, 0, weeks - 1), c], before[c]),
                0.0,
            )
            carry[c, r] = spent @ k
    return carry, reach


def promoted_phases(price: _Array) -> list[int]:
    """The phases `t mod 52` whose mean price over the years seen lies below the price's median
    by more than `PRICE_DROP`, ascending."""
    phase = np.arange(price.size) % YEAR
    median = float(np.median(price))
    return [
        p
        for p in range(YEAR)
        if (phase == p).any() and price[phase == p].mean() < median - PRICE_DROP
    ]


def _check(y: _Array, x: _Array, flag: _Array | None, price: _Array | None, kernel: int) -> None:
    given = [a for a in (flag, price) if a is not None]
    if (
        y.ndim != 1
        or x.ndim != 2
        or x.shape[0] != y.size
        or x.shape[1] < 1
        or any(a.shape != y.shape for a in given)
    ):
        flag_shape, price_shape = (None if a is None else a.shape for a in (flag, price))
        msg = (
            f"sales are (weeks,), spend (weeks, channels) and the promotion and the price (weeks,)"
            f" where given, not {y.shape}, {x.shape}, {flag_shape} and {price_shape}"
        )
        raise ValueError(msg)
    if kernel < 1:
        msg = f"a kernel is at least a week long, not {kernel}"
        raise ValueError(msg)
    if not all(np.all(np.isfinite(a)) for a in (y, x, *given)):
        msg = "the sales, the spend, the promotion and the price are finite"
        raise ValueError(msg)


def _nuisance(
    weeks: int,
    promotion: _Array | None,
    price: _Array | None,
    phases: list[int],
    knot_every: int = KNOT_EVERY,
) -> _Array:
    """The columns beside the channels': the trend, its interior knots `knot_every` weeks apart,
    the season, the flag and the price where given, and a dummy for each promoted phase."""
    phase = np.arange(weeks) % YEAR
    columns = [_trend(weeks, knot_every), _season(weeks)]
    columns += [a[:, None] for a in (promotion, price) if a is not None]
    columns += [(phase == p).astype(float)[:, None] for p in phases]
    return np.column_stack(columns)


def _trend(weeks: int, knot_every: int) -> _Array:
    t = np.arange(weeks, dtype=float)
    interior = [float(k) for k in range(knot_every, weeks - 1, knot_every)]
    knots = np.r_[[0.0] * (DEGREE + 1), interior, [float(weeks - 1)] * (DEGREE + 1)]
    return BSpline.design_matrix(t, knots, DEGREE).toarray()


def _season(weeks: int) -> _Array:
    t = np.arange(weeks, dtype=float)
    columns = []
    for k in range(1, HARMONICS + 1):
        columns += [np.sin(2 * math.pi * k * t / YEAR), np.cos(2 * math.pi * k * t / YEAR)]
    return np.column_stack(columns)


def _ridge(xw: _Array, yw: _Array) -> tuple[_Array, _Array, _Array, _Array]:
    """The conjugate prior's posterior mean on whitened data, the Cholesky factor of
    `X'X + Lambda`, the triangular solve of `X'y` by it, and `Lambda`'s diagonal."""
    gram = xw.T @ xw
    d = np.diag(gram).copy()
    d[d <= 0] = 1.0
    lam = PRIOR * d
    factor = np.linalg.cholesky(gram + np.diag(lam))
    z = solve_triangular(factor, xw.T @ yw, lower=True)
    return solve_triangular(factor.T, z, lower=False), factor, z, lam


def _schur(xw: _Array, channels: int) -> _Array:
    """Each of the last `channels` columns' sum of squares left after its least-squares regression
    on every other column of `xw`: the data's precision on its coefficient over the noise's, the
    Schur complement of `xw' xw` at it, 0 to rounding where the other columns span it."""
    p = xw.shape[1]
    out = np.empty(channels)
    for j, column in enumerate(range(p - channels, p)):
        others = np.delete(xw, column, axis=1)
        coef, *_ = np.linalg.lstsq(others, xw[:, column], rcond=None)
        rest = xw[:, column] - others @ coef
        out[j] = rest @ rest
    return out


def _ar1(residual: _Array) -> float:
    """The residual's lag-1 coefficient clipped to `[0, AR_MOST]`.

    Raises:
        DegenerateFitError: where the lagged residual's sum of squares is not positive or the
            coefficient is not finite.
    """
    lagged = np.dot(residual[:-1], residual[:-1])
    if not lagged > 0.0:
        msg = (
            f"the lagged residual's sum of squares is {float(lagged)}: the AR(1) coefficient is 0/0"
        )
        raise DegenerateFitError(msg)
    phi = float(np.clip(np.dot(residual[1:], residual[:-1]) / lagged, 0.0, AR_MOST))
    if not math.isfinite(phi):
        msg = f"the AR(1) coefficient is {phi}"
        raise DegenerateFitError(msg)
    return phi


def _ar_cov(n: int, phi: float) -> _Array:
    k = np.arange(n)
    return phi ** np.abs(k[:, None] - k[None, :]) / (1.0 - phi * phi)
