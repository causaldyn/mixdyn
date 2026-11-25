"""A closed-form model over geos fitted by empirical Bayes: each channel's curve of spend per head
shared by the geos, its margin in each geo pooled about the channel's, the geo tests in the
likelihood, and each channel's posterior over a grid of its carryover and curvature (ADR 0090).

For geo `g`, week `t` and channel `c` at the channel's grid point `theta_c = (r_c, kappa_c)`:

    y_gt = n_gt + sum_c b_gc X_gct(theta_c) + e_gt,        e_gt ~ N(0, sigma_g^2)
    X_gct = s_gc F(a_gct / s_gc),        F(u) = u / (1 + u / kappa) (1 + 1 / kappa)^2

`a_gct` is the channel's spend in the geo through a geometric kernel of the given length `L`, its
weights `r^l` over `l < L` normalised to sum 1, nothing spent before the first week
(`mixdyn.models.margin_mmm.adstock`); the first `L - 1` weeks are its burn-in and are not read.
`s_gc = p_g xbar_c` is the geo's population share `p_g` of the channel's national mean weekly
spend `xbar_c`, the sum over the geos of each geo's mean weekly spend. `F` is Michaelis and
Menten's curve scaled to a margin of 1 at `u = 1`, and the line `u` at `kappa = inf`, so `b_gc` is
what a unit of weekly spend returns in the geo at its share of the national mean spend, the same
spend per head in every geo. Each channel's grid is `POINTS`, `RETENTIONS` by `CURVATURES`.

`n_gt` is the geo's own intercept, `HARMONICS` harmonics of a `YEAR`-week year and a linear trend,
partialled out of the KPI and of every column geo by geo (Frisch, Waugh and Lovell). Each geo's
noise variance `sigma_g^2` is read once, off the reference fit, every channel at retention 0.45
on the line: its residual sum of squares over the weeks read less the nuisance's columns and the
channels'. It is held at every grid point. Its floor is `FLOOR` of the KPI's variance once the
nuisance is out, and no constant here is in the money's unit, so the KPI and the spend in another
unit give the same fit in it; a geo whose KPI does not move at all is refused.

A channel is identified by the movement of its spend that a geo's nuisance does not explain. One
whose reference column keeps less than `VARIATION` of its sum of squares in every geo, as a
channel never spent or one spent the same every week, is refused by name: its margin would be
rounding or noise, and the pooling's scales are read off it. A geo where the channel does not
move is no reason to refuse: its margin is pooled about the channel's, which the other geos
give.

The geos' margins are pooled about the channel's, `b_gc = mu_c + delta_gc`, with
`delta_gc ~ N(0, (SPREAD m_c)^2)` and `mu_c ~ N(0, (PRIOR_WIDTH max(max_g |bhat_gc|, m_c))^2)`:
`bhat_gc` is the reference fit's least squares and `m_c = sqrt(wmean_c^2 + se_c^2)`, their mean
over the geos weighted by each one's precision and that mean's standard error.

A lift estimate (`mixdyn.calibration.LiftEstimate`) is a normal observation, of its own standard
error, of the change in its geos' summed contribution over its readout, which is linear in their
`b_gc` given `theta_c`. Each geo the test names spends its own history changed in proportion to
the group's, the estimate's `planned` to its `spend`, so a go-dark sets each named geo's spend to
0 in the dark weeks; it is carried over and saturated as the geo's columns are. An estimate is
placed in the panel's calendar by its `end`, and its `planned` is the named geos' spend summed.
Its readout lies in the weeks read: in the burn-in the adstock misses the spend before the
panel, and a curve's change there is not the change the test read.

Given every channel's grid point the coefficients are Gaussian, and the marginal likelihood of the
KPI and the lifts is closed (`validation/geo_eb.mac`): per geo the least squares and the residual
sum of squares give the likelihood of its coefficients, and the coefficients, their pooling and the
tests are one Gaussian integral. The grid points are found channel by channel: `SWEEPS` sweeps of
coordinate ascent from the reference point, each step the channel's point of the most marginal
likelihood with the others held. Then each channel's posterior over its own grid, the others at
their points and a uniform prior on the grid: its weights above `KEPT`, renormalised, and under
each kept point the posterior mean of every coefficient.

A cell of a plan (`PooledGeoFit.cell`) is a geo's own channel, or a channel bought nationally as
one spend: split by the population shares, its geos' columns are one channel of the national spend,
of scale `xbar_c` and coefficient `sum_g p_g b_gc` (`validation/geo_eb.mac`). Its expected worth
under the posterior (`CellResponse`) is the mixture over the channel's kept points of its worth at
the posterior mean coefficients, as worth is linear in them;
`mixdyn.decisions.geo_plan.plan_mixture` plans the cells on it.

The arithmetic follows the code the rule was frozen in, step for step, so that its plans and their
regrets are reproduced to rounding (ADR 0090).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from mixdyn.data import Allotment
from mixdyn.models.margin_mmm import adstock

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mixdyn.calibration import LiftEstimate
    from mixdyn.data import Channel, Geo, MediaPanel

type _Array = npt.NDArray[np.float64]

RETENTIONS = (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9)
"""Each channel's retentions on its grid."""
CURVATURES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, math.inf)
"""Each channel's curvatures `kappa` on its grid, the line at infinity."""
POINTS = tuple((r, kappa) for kappa in CURVATURES for r in RETENTIONS)
"""A channel's grid, `(retention, curvature)`, curvature by curvature."""
REFERENCE = (0.45, math.inf)
"""The reference fit's point, every channel's: the noise and the pooling's scales are read there."""
YEAR = 52.0
"""The weeks of the season's year."""
HARMONICS = 2
"""The season's harmonics, a sine and a cosine each."""
NUISANCE = 2 + 2 * HARMONICS
"""Each geo's nuisance columns: its intercept, its season's and its trend."""
SPREAD = 0.4
"""The standard deviation of a geo's margin about its channel's, over the reference fit's `m_c`."""
PRIOR_WIDTH = 10.0
"""The channel's margin's prior standard deviation over the reference fit's largest margin."""
SWEEPS = 3
"""The sweeps of coordinate ascent over the channels."""
KEPT = 1e-4
"""The least posterior weight of a grid point kept in a channel's mixture."""
RIDGE = 1e-9
"""Each geo's least squares' ridge, over the mean of its Gram matrix's diagonal."""
FLOOR = 1e-12
"""The least noise variance of a geo, as a share of its KPI's variance once its nuisance is out:
rounding."""
VARIATION = 1e-6
"""The least share of a channel's reference column's sum of squares that a geo's nuisance leaves,
in some geo, for the panel to identify the channel: a column that moves by a thousandth of its
size."""
AGREE = 1e-9
"""How near an estimate's `planned` must lie to its geos' summed spend, relative to its largest."""


@dataclass(frozen=True, slots=True, eq=False)
class Mixture:
    """A channel's posterior over its grid, the other channels at their points: `points`, the
    kept points as indices into `POINTS`; `weights`, theirs, summing to 1; and `coefficients`,
    the posterior mean of the channel's coefficient in each geo under each point,
    `(points, geos)`."""

    points: npt.NDArray[np.intp]
    weights: _Array
    coefficients: _Array


@dataclass(frozen=True, slots=True, eq=False)
class CellResponse:
    """A cell's expected response to its weekly spend over a plan: a mixture of curves, one a kept
    grid point, each of its `weights`, `retentions`, `curvatures` and `coefficients`.

    A plan spends `w` a week over its `periods`; with the cell's `history`, its spend before them,
    the kernel's adstock at each of the plan's periods and the `length - 1` after them is
    `a_h = carry_h + w reach_h`, and a point's worth is `b s sum_h F(a_h / s)`, `s` the cell's
    `scale` (the module's). `worth` is the mixture's, and `slope` its derivative in `w`.

    Raises:
        ValueError: on component arrays of different shapes, a history shorter than the kernel's
            `length - 1` weeks it reaches back, or a scale not positive.
    """

    history: _Array
    scale: float
    length: int
    weights: _Array
    retentions: _Array
    curvatures: _Array
    coefficients: _Array

    def __post_init__(self) -> None:
        names = ("history", "weights", "retentions", "curvatures", "coefficients")
        for name in names:
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=float))
        shapes = {getattr(self, name).shape for name in names[1:]}
        if len(shapes) != 1 or self.weights.ndim != 1:
            msg = f"a cell's components are one (points,) array each, not {sorted(shapes)}"
            raise ValueError(msg)
        if self.history.ndim != 1 or self.length < 1 or self.history.size < self.length - 1:
            msg = (
                f"a cell's history of {self.history.shape} weeks is shorter than its kernel of"
                f" {self.length} reaches back"
            )
            raise ValueError(msg)
        if not self.scale > 0.0:
            msg = f"a cell's scale is positive, not {self.scale}"
            raise ValueError(msg)

    def slope(self, weekly: npt.ArrayLike, periods: int) -> _Array:
        """The expected worth's derivative at each weekly spend of `weekly` over `periods`."""
        x = np.asarray(weekly, dtype=float)
        out = np.zeros(x.shape)
        for w, b, carry, reach, kappa in self._components(periods):
            a = carry[None, :] + x[:, None] * reach[None, :]
            out += w * b * (reach[None, :] * _slope(a / self.scale, kappa)).sum(axis=1)
        return out

    def worth(self, weekly: npt.ArrayLike, periods: int) -> _Array:
        """The expected worth at each weekly spend of `weekly` over `periods` and the kernel's
        tail after them, the history's carryover counted in."""
        x = np.asarray(weekly, dtype=float)
        out = np.zeros(x.shape)
        for w, b, carry, reach, kappa in self._components(periods):
            a = carry[None, :] + x[:, None] * reach[None, :]
            out += w * b * (self.scale * _curve(a / self.scale, kappa)).sum(axis=1)
        return out

    def _components(self, periods: int) -> list[tuple[float, float, _Array, _Array, float]]:
        """Each point's weight, coefficient, carryover and reach at each horizon, and curvature."""
        size, length = self.history.size, self.length
        source = size + np.arange(periods + length - 1)[:, None] - np.arange(length)[None, :]
        before = source < size
        during = (source >= size) & (source < size + periods)
        out = []
        for w, b, r, kappa in zip(
            self.weights, self.coefficients, self.retentions, self.curvatures, strict=True
        ):
            kern = _kernel(float(r), length)
            carry = (np.where(before, self.history[np.minimum(source, size - 1)], 0.0) * kern).sum(
                axis=1
            )
            reach = (during * kern).sum(axis=1)
            out.append((float(w), float(b), carry, reach, float(kappa)))
        return out


@dataclass(frozen=True, slots=True, eq=False)
class PooledGeoFit:
    """The fit of a panel over geos (the module).

    `channels` and `geos` are the panel's, `national` its channels bought nationally, `length` the
    kernel's. `selected` is each channel's grid point at the end of the coordinate ascent, an index
    into `POINTS`, and `mixtures` each channel's posterior over its grid. `shares` are the geos'
    population shares, `scale` each geo's channel's `s_gc`, `(geos, channels)`, and `spend` the
    panel's, `(geos, weeks, channels)`. `sigma` is each geo's noise standard deviation, `margin`
    each channel's `m_c` and `prior_sd` its margin's prior standard deviation. `rows` are the weeks
    read after the burn-in, and `tests` the lift estimates read.
    """

    geos: tuple[Geo, ...]
    channels: tuple[Channel, ...]
    national: frozenset[Channel]
    length: int
    selected: tuple[int, ...]
    mixtures: tuple[Mixture, ...]
    shares: _Array
    scale: _Array
    spend: _Array
    sigma: _Array
    margin: _Array
    prior_sd: _Array
    rows: int
    tests: int

    def cell(self, geo: Geo | None, channel: Channel) -> CellResponse:
        """The expected response of a geo's own `channel` in `geo`, or, at `geo` None, of a
        channel bought nationally as one cell of its national spend (the module).

        Raises:
            ValueError: on a channel or a geo the fit does not have, a geo named for a channel
                bought nationally, or none for one that is not.
        """
        if channel not in self.channels:
            msg = f"the fit has no channel {channel!r}: {list(self.channels)}"
            raise ValueError(msg)
        if (geo is None) != (channel in self.national):
            msg = (
                f"a channel bought nationally is one cell of no geo, and each other channel a cell"
                f" of each geo: got {channel!r} in {geo!r}"
            )
            raise ValueError(msg)
        c = self.channels.index(channel)
        mixture = self.mixtures[c]
        if geo is None:
            history = self.spend[:, :, c].sum(axis=0)
            scale = float(self.scale[:, c].sum())
            coefficients = np.array([float((self.shares * b).sum()) for b in mixture.coefficients])
        else:
            if geo not in self.geos:
                msg = f"the fit has no geo {geo!r}: {list(self.geos)}"
                raise ValueError(msg)
            g = self.geos.index(geo)
            history = self.spend[g, :, c]
            scale = float(self.scale[g, c])
            coefficients = mixture.coefficients[:, g]
        picked = [POINTS[j] for j in mixture.points]
        return CellResponse(
            history=np.array(history),
            scale=scale,
            length=self.length,
            weights=mixture.weights,
            retentions=np.array([r for r, _ in picked]),
            curvatures=np.array([kappa for _, kappa in picked]),
            coefficients=np.array(coefficients),
        )


def fit_pooled_geos(
    panel: MediaPanel, estimates: Mapping[str, Sequence[LiftEstimate]], length: int
) -> PooledGeoFit:
    """The model over geos (the module) fitted to `panel` with each channel's lift `estimates`,
    its kernel `length` weeks long.

    Raises:
        ValueError: on a panel without populations, with controls, events, filled steps or a KPI
            that is not finite, a channel split nationally other than by population, a kernel
            shorter than a week or one that leaves no residual degree of freedom, a channel whose
            spend no geo's nuisance leaves `VARIATION` of, a geo whose KPI its nuisance leaves
            nothing of; and on an estimate of a channel or a geo the panel does not have, in a
            share of its geos' market, without its end in the panel's calendar, whose readout
            begins in the burn-in, whose `planned` is not its geos' spend summed, or that spends
            where its geos planned nothing.
        FloatingPointError: on a marginal likelihood that is not finite, as where the KPI's
            squares overflow.
    """
    return _Model(panel, estimates, length).fit()


class _Model:
    """One panel's columns, projections and Gram matrices at every grid point, its tests' rows, and
    the closed-form marginal likelihood of a choice of each channel's grid point."""

    def __init__(
        self, panel: MediaPanel, estimates: Mapping[str, Sequence[LiftEstimate]], length: int
    ) -> None:
        _check(panel, length)
        population = np.asarray(panel.population, dtype=float)
        spend = np.asarray(panel.spend, dtype=float)
        sales = np.asarray(panel.kpi, dtype=float)
        geos, weeks, channels = spend.shape
        self.panel, self.length = panel, length
        self.G, self.C, self.J = geos, channels, len(POINTS)
        self.spend = spend
        self.shares = population / population.sum()
        self.rows = np.arange(length - 1, weeks)
        nuisance = _nuisance(self.rows + 1)
        q = np.linalg.qr(nuisance)[0]
        self.dof = self.rows.size - nuisance.shape[1]
        national_mean = spend.mean(axis=1).sum(axis=0)
        scale = self.shares[:, None] * national_mean[None, :]
        # a channel never spent has no scale: its columns are 0 at any, and it is refused below
        self.scale = np.where(scale > 0.0, scale, 1.0)
        self.carried = {
            r: np.array(
                [
                    [adstock(spend[g, :, c], r, length, 0.0) for g in range(geos)]
                    for c in range(channels)
                ]
            )
            for r in RETENTIONS
        }
        x = np.empty((channels, self.J, geos, self.rows.size))
        for j, (r, kappa) in enumerate(POINTS):
            u = self.carried[r] / self.scale.T[:, :, None]
            x[:, j] = (self.scale.T[:, :, None] * _curve(u, kappa))[:, :, self.rows]
        at = POINTS.index(REFERENCE)
        whole = np.einsum("kgt,kgt->kg", x[:, at], x[:, at])
        x = x - np.einsum("tq,kjgq->kjgt", q, np.einsum("tq,kjgt->kjgq", q, x))
        _identified(panel, whole, np.einsum("kgt,kgt->kg", x[:, at], x[:, at]))
        y = sales[:, self.rows]
        y = y - (y @ q) @ q.T
        self.x, self.y = x, y
        self.yty = np.einsum("gt,gt->g", y, y)
        still = [str(g) for g, left in zip(panel.geos, self.yty, strict=True) if not left > 0.0]
        if still:
            msg = (
                f"the KPI of the geos {still} does not move once their intercept, season and trend"
                " are out, so it gives them no noise variance"
            )
            raise ValueError(msg)
        self.xy = np.einsum("kjgt,gt->kjg", x, y)
        self.gram: dict[tuple[int, int], _Array] = {}
        for a in range(channels):
            for b in range(a, channels):
                held = np.einsum("jgt,igt->gji", x[a], x[b])
                self.gram[a, b] = held
                self.gram[b, a] = held.transpose(0, 2, 1)
        self.tests = [
            self._row(channel, estimate) for channel, rows in estimates.items() for estimate in rows
        ]
        self.reference = np.full(channels, POINTS.index(REFERENCE))
        bh, rss, _ = self._ols(self.reference[None, :])
        self.sigma2 = np.maximum(rss[0] / (self.dof - channels), FLOOR * self.yty / self.dof)
        precision = np.einsum(
            "gkk->gk", self._gram(self.reference[None, :])[0] / self.sigma2[:, None, None]
        )
        wmean = (precision * bh[0]).sum(axis=0) / precision.sum(axis=0)
        se = 1.0 / np.sqrt(precision.sum(axis=0))
        self.margin = np.sqrt(wmean**2 + se**2)
        self.prior_sd = PRIOR_WIDTH * np.maximum(np.abs(bh[0]).max(axis=0), self.margin)

    def _row(self, channel: str, estimate: LiftEstimate) -> tuple[int, _Array, float, float]:
        """A lift estimate's channel, its `z` at every grid point in every geo, `(points, geos)`, 0
        in a geo it does not name, its lift and its standard error."""
        panel = self.panel
        if channel not in panel.channels:
            msg = (
                f"an estimate of {channel!r}, which the panel does not have: {list(panel.channels)}"
            )
            raise ValueError(msg)
        if estimate.share != 1.0:
            msg = (
                f"{channel}'s estimate is in a share {estimate.share} of its geos' market, which"
                " the model of the market reads; the model over geos reads a test in its geos"
            )
            raise ValueError(msg)
        named = panel.geos if estimate.geos is None else tuple(estimate.geos)
        unknown = sorted(set(named) - set(panel.geos))
        if unknown:
            msg = f"{channel}'s estimate names geos {unknown} the panel does not have"
            raise ValueError(msg)
        group = sorted(panel.geos.index(g) for g in named)
        dates = np.asarray(panel.dates)
        if estimate.end is None or not np.any(dates == estimate.end):
            msg = f"{channel}'s estimate ends on {estimate.end}, not a period of the panel"
            raise ValueError(msg)
        last = int(np.flatnonzero(dates == estimate.end)[0])
        first = last - estimate.spend.size + 1
        if first < 0:
            msg = f"{channel}'s estimate runs from before the panel's first period"
            raise ValueError(msg)
        begins = last - estimate.readout + 1
        if begins < self.length - 1:
            msg = (
                f"{channel}'s estimate ending on {estimate.end} reads from the panel's period"
                f" {begins + 1}, in the kernel's burn-in of {self.length - 1} periods, where the"
                " adstock misses the spend before the panel"
            )
            raise ValueError(msg)
        c = panel.channels.index(channel)
        summed = self.spend[group, first : last + 1, c].sum(axis=0)
        spent, planned = estimate.spend, estimate.planned
        if np.max(np.abs(planned - summed)) > AGREE * float(np.max(summed, initial=0.0)):
            msg = f"{channel}'s estimate plans spend other than its geos' over its periods"
            raise ValueError(msg)
        if np.any((planned == 0.0) & (spent != 0.0)):
            msg = (
                f"{channel}'s test spends where its geos planned nothing, so its spend has no split"
                " over them"
            )
            raise ValueError(msg)
        ratio = np.divide(spent, planned, out=np.ones_like(planned), where=planned != 0.0)
        tested = {}
        for g in group:
            series = self.spend[g, :, c].copy()
            series[first : last + 1] *= ratio
            tested |= {(r, g): adstock(series, r, self.length, 0.0) for r in RETENTIONS}
        readout = slice(begins, last + 1)
        z = np.zeros((self.J, self.G))
        for j, (r, kappa) in enumerate(POINTS):
            for g in group:
                s = self.scale[g, c]
                diff = s * (
                    _curve(tested[r, g] / s, kappa) - _curve(self.carried[r][c, g] / s, kappa)
                )
                z[j, g] = diff[readout].sum()
        return c, z, float(estimate.lift), float(estimate.se)

    def _gram(self, sel: npt.NDArray[np.intp]) -> _Array:
        """The Gram matrices of the selections `sel`, `(choices, channels)`: `(choices, geos,
        channels, channels)`."""
        out = np.empty((sel.shape[0], self.G, self.C, self.C))
        for a in range(self.C):
            for b in range(self.C):
                out[:, :, a, b] = self.gram[a, b][:, sel[:, a], sel[:, b]].T
        return out

    def _xy(self, sel: npt.NDArray[np.intp]) -> _Array:
        return np.stack([self.xy[k][sel[:, k]] for k in range(self.C)], axis=-1)

    def _ols(self, sel: npt.NDArray[np.intp]) -> tuple[_Array, _Array, _Array]:
        """Each geo's least squares at each selection, its residual sum of squares and its Gram
        matrix, the least squares under `RIDGE`."""
        xtx = self._gram(sel)
        xy = self._xy(sel)
        ridge = RIDGE * np.einsum("bgkk->bgk", xtx).mean(axis=-1)[..., None, None]
        bh = np.linalg.solve(xtx + ridge * np.eye(self.C), xy[..., None])[..., 0]
        rss = self.yty[None, :] - np.einsum("bgk,bgk->bg", xy, bh)
        return bh, rss, xtx

    def loglik(self, sel: npt.NDArray[np.intp]) -> tuple[_Array, _Array]:
        """The log marginal likelihood of each selection of `sel`, `(choices, channels)`, up to a
        constant every selection shares, and the posterior mean of the coefficients under each,
        `(choices, geos, channels)`."""
        choices = sel.shape[0]
        geos, channels = self.G, self.C
        bh, rss, xtx = self._ols(sel)
        vinv = xtx / self.sigma2[None, :, None, None]
        size = channels + geos * channels
        a = np.zeros((choices, size, size))
        h = np.zeros((choices, size))
        vb = np.einsum("bgkl,bgl->bgk", vinv, bh)
        c0 = np.einsum("bgk,bgk->b", bh, vb)
        a[:, :channels, :channels] = vinv.sum(axis=1)
        h[:, :channels] = vb.sum(axis=1)
        for g in range(geos):
            s = slice(channels + g * channels, channels + (g + 1) * channels)
            a[:, :channels, s] = vinv[:, g]
            a[:, s, :channels] = vinv[:, g]
            a[:, s, s] = vinv[:, g]
            h[:, s] = vb[:, g]
        omega2 = (np.full(channels, SPREAD) * self.margin) ** 2
        p0 = np.concatenate([1.0 / self.prior_sd**2, np.tile(1.0 / omega2, geos)])
        a[:, np.arange(size), np.arange(size)] += p0[None, :]
        for c, z, lift, se in self.tests:
            zz = z[sel[:, c]]
            v = np.zeros((choices, size))
            v[:, c] = zz.sum(axis=1)
            v[:, channels + np.arange(geos) * channels + c] = zz
            a += np.einsum("bi,bj->bij", v, v) / se**2
            h += v * lift / se**2
            c0 = c0 + lift**2 / se**2
        chol = np.linalg.cholesky(a)
        logdet = 2.0 * np.log(np.einsum("bii->bi", chol)).sum(axis=1)
        sol = np.linalg.solve(a, h[..., None])[..., 0]
        quad = np.einsum("bi,bi->b", h, sol)
        ll = (
            -0.5 * (rss / self.sigma2[None, :]).sum(axis=1)
            + 0.5 * np.log(p0).sum()
            - 0.5 * logdet
            - 0.5 * (c0 - quad)
        )
        if not np.all(np.isfinite(ll)):
            msg = (
                f"the marginal likelihood is not finite at {int((~np.isfinite(ll)).sum())} of"
                f" {choices} selections: the KPI's, the spend's or a test's scale is past what"
                " the arithmetic holds"
            )
            raise FloatingPointError(msg)
        mu = sol[:, :channels]
        delta = sol[:, channels:].reshape(choices, geos, channels)
        return ll, mu[:, None, :] + delta

    def _around(self, sel: npt.NDArray[np.intp], c: int) -> npt.NDArray[np.intp]:
        """`sel` with channel `c` at each of its grid points in turn, `(points, channels)`."""
        cand = np.repeat(sel[None, :], self.J, axis=0)
        cand[:, c] = np.arange(self.J)
        return cand

    def fit(self) -> PooledGeoFit:
        sel = self.reference.copy()
        for _ in range(SWEEPS):
            for c in range(self.C):
                cand = self._around(sel, c)
                ll, _ = self.loglik(cand)
                sel = cand[int(np.argmax(ll))]
        mixtures = []
        for c in range(self.C):
            ll, beta = self.loglik(self._around(sel, c))
            w = np.exp(ll - ll.max())
            w /= w.sum()
            keep = w > KEPT
            w = w[keep] / w[keep].sum()
            mixtures.append(Mixture(np.flatnonzero(keep), w, beta[keep][:, :, c]))
        panel = self.panel
        return PooledGeoFit(
            geos=panel.geos,
            channels=panel.channels,
            national=frozenset(panel.national),
            length=self.length,
            selected=tuple(int(j) for j in sel),
            mixtures=tuple(mixtures),
            shares=self.shares,
            scale=self.scale,
            spend=self.spend,
            sigma=np.sqrt(self.sigma2),
            margin=self.margin,
            prior_sd=self.prior_sd,
            rows=int(self.rows.size),
            tests=len(self.tests),
        )


def _check(panel: MediaPanel, length: int) -> None:
    if panel.population is None:
        msg = "the model over geos reads each geo's share of the population, and the panel has none"
        raise ValueError(msg)
    if panel.controls or panel.events:
        msg = (
            f"the model over geos reads no control or event, and the panel has controls"
            f" {list(panel.controls)} and events {list(panel.events)}"
        )
        raise ValueError(msg)
    if np.any(panel.filled) or not np.all(np.isfinite(panel.kpi)):
        msg = "the model over geos reads every geo's KPI in every period, and the panel misses some"
        raise ValueError(msg)
    split = [c for c, by in panel.national.items() if by is not Allotment.POPULATION]
    if split:
        msg = f"the channels {split} are split over the geos other than by population"
        raise ValueError(msg)
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        msg = f"a kernel is a whole number of weeks, at least 1, not {length!r}"
        raise ValueError(msg)
    weeks = panel.dates.size - length + 1
    if weeks - NUISANCE - len(panel.channels) < 1:
        msg = (
            f"{panel.dates.size} weeks after a burn-in of {length - 1} leave no residual degree of"
            f" freedom for {NUISANCE} nuisance columns and {len(panel.channels)} channels"
        )
        raise ValueError(msg)


def _identified(panel: MediaPanel, whole: _Array, left: _Array) -> None:
    """Refuse each channel whose reference column, its sum of squares `whole` in each geo and
    `left` once the geo's nuisance is out of it, `(channels, geos)`, keeps less than `VARIATION`
    of itself in every geo (the module)."""
    share = np.divide(left, whole, out=np.zeros_like(left), where=whole > 0.0)
    flat = [str(c) for c, kept in zip(panel.channels, share, strict=True) if kept.max() < VARIATION]
    if flat:
        msg = (
            f"the channels {flat} move in no geo: with a geo's intercept, season and trend out,"
            f" less than {VARIATION:g} of the reference column's sum of squares is left, so the"
            " panel does not identify their margins (a channel never spent, or spent the same"
            " every week)"
        )
        raise ValueError(msg)


def _nuisance(weeks: _Array) -> _Array:
    """A geo's intercept, its season's harmonics and its linear trend at `weeks`."""
    t = np.asarray(weeks, dtype=float)
    cols = [np.ones_like(t)]
    for k in range(1, HARMONICS + 1):
        cols += [np.sin(2 * np.pi * k * t / YEAR), np.cos(2 * np.pi * k * t / YEAR)]
    cols.append((t - t.mean()) / (t.std() + 1e-12))
    return np.column_stack(cols)


def _kernel(retention: float, length: int) -> _Array:
    w = retention ** np.arange(length, dtype=float)
    return w / w.sum()


def _curve(u: _Array, kappa: float) -> _Array:
    """`F(u)`, Michaelis and Menten's curve of curvature `kappa` scaled to a margin of 1 at 1."""
    if kappa == math.inf:
        return u
    return u / (1.0 + u / kappa) * (1.0 + 1.0 / kappa) ** 2


def _slope(u: _Array, kappa: float) -> _Array:
    """`F'(u)`."""
    if kappa == math.inf:
        return np.ones_like(u)
    return ((1.0 + 1.0 / kappa) / (1.0 + u / kappa)) ** 2
