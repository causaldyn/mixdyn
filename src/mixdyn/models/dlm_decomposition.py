"""A fitted DLM's decomposition with its uncertainty: every component, draw by draw.

`chc.dlm.backward_sample` draws the states' path given all the data, moved by the tests where the
fit read lift tests (`mixdyn.models.dlm_lift.lifted_draws`), or with a signed effect the fit holds
the draws of the posterior truncated to the signs, and `chc.dlm.decompose` splits each draw's mean
into its parts. A window's total, or a change between two windows, is computed draw by draw before
any quantile is taken, so its interval carries the dependence of the components across periods and
on each other: an interval of a sum is not a sum of intervals. Every number here is draws; an
interval is their quantiles, `np.quantile(d, [0.05, 0.95])`.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from chc.dlm import PosteriorDraws, backward_sample, decompose

from mixdyn.models.dlm_lift import lifted_draws

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from mixdyn.models.dlm import DlmFit

type _Array = npt.NDArray[np.float64]
type _Dates = npt.NDArray[np.datetime64]


@dataclass(frozen=True, slots=True, eq=False)
class Components:
    """One array of draws per component: the base (None without a trend), the seasonality (None
    without a season), and each effect's contribution. All share a shape."""

    base: _Array | None
    seasonality: _Array | None
    contributions: Mapping[str, _Array]

    def total(self) -> _Array:
        """The components' sum, draw by draw."""
        parts = [self.base, self.seasonality, *self.contributions.values()]
        return np.sum([p for p in parts if p is not None], axis=0)

    def _map(self, f: Callable[[_Array], _Array]) -> Components:
        return Components(
            None if self.base is None else f(self.base),
            None if self.seasonality is None else f(self.seasonality),
            MappingProxyType({k: f(v) for k, v in self.contributions.items()}),
        )


@dataclass(frozen=True, slots=True, eq=False)
class DlmDecomposition:
    """A fit's components over its periods, `(draws, periods)` each, from the posterior of its
    states. A draw's components add up to its mean; their averages over the draws are the fit's
    smoothed decomposition, up to Monte Carlo error."""

    dates: _Dates
    draws: Components

    def window(self, start: np.datetime64, end: np.datetime64) -> Components:
        """Each component's total over the periods from `start` to `end`, both included: the
        waterfall's steps, which add up to the window's total mean draw by draw.

        Raises:
            ValueError: on a window that holds no period of the fit.
        """
        inside = self._periods(start, end)
        return self.draws._map(lambda d: d[:, inside].sum(axis=1))

    def change(
        self,
        before: tuple[np.datetime64, np.datetime64],
        after: tuple[np.datetime64, np.datetime64],
    ) -> Components:
        """Each component's change in its mean per period from the window `before` to the window
        `after`, each given as its first and last date: the period-over-period view. Means per
        period let windows of different lengths compare.

        Raises:
            ValueError: on a window that holds no period of the fit.
        """
        first, second = self._periods(*before), self._periods(*after)
        return self.draws._map(lambda d: d[:, second].mean(axis=1) - d[:, first].mean(axis=1))

    def _periods(self, start: np.datetime64, end: np.datetime64) -> npt.NDArray[np.bool_]:
        inside = (self.dates >= np.datetime64(start)) & (self.dates <= np.datetime64(end))
        if not inside.any():
            msg = (
                f"no period of the fit lies from {start} to {end}; it runs from "
                f"{self.dates[0]} to {self.dates[-1]}"
            )
            raise ValueError(msg)
        return inside


def decompose_dlm(fit: DlmFit, *, draws: int, seed: int) -> DlmDecomposition:
    """Draw `fit`'s states given all its data, `draws` times, and split each draw's mean into the
    base, the seasonality and each effect's contribution.

    With a signed effect the draws are `draws` of the fit's own (`fit.signed`), chosen without
    replacement from `seed`: the decomposition and its point estimate then come from one run,
    the one `diagnose_dlm` checks for mixing. A fit read with tests draws its posterior given them.
    A lever's contribution is each draw's coefficient times its departure from its level
    (`DlmFit.baselines`), and the base takes the coefficient times the level.

    Raises:
        ValueError: on `draws` below 1, or above the draws a signed fit holds.
    """
    if fit.signed is not None:
        pool = fit.signed.pooled
        held = pool.states.shape[0]
        if isinstance(draws, bool) or not isinstance(draws, int) or not 1 <= draws <= held:
            msg = (
                f"draws must be an integer from 1 to the {held} the signed fit holds, got {draws!r}"
            )
            raise ValueError(msg)
        pick = np.random.default_rng(seed).choice(held, draws, replace=False)
        posterior = PosteriorDraws(pool.states[pick], pool.variance[pick])
    elif fit.lifted is not None:
        posterior = lifted_draws(fit.filtered, fit.lifted, draws, seed)
    else:
        posterior = backward_sample(fit.filtered, draws, seed)
    parts = decompose(fit.filtered, posterior)
    spec = fit.spec
    first = 1 if spec.trend else 0
    order = spec.trend.order if spec.trend else 0
    levels = fit.baselines
    base = None
    if spec.trend:
        base = parts[:, :, 0]
        for i, e in enumerate(spec.effects):
            if e.column in levels:
                base = base + posterior.states[:, :, order + i] * levels[e.column]
    contributions = {
        e.column: parts[:, :, first + i] - posterior.states[:, :, order + i] * levels[e.column]
        if e.column in levels
        else parts[:, :, first + i]
        for i, e in enumerate(spec.effects)
    }
    components = Components(
        base=base,
        seasonality=parts[:, :, -1] if spec.season else None,
        contributions=MappingProxyType(contributions),
    )
    return DlmDecomposition(fit.dates, components)
