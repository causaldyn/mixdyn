"""The weighted quantile every decision module reads an interval's ends by.

A share's quantile is the least draw whose cumulative weight, the draws sorted, reaches that share
of the total: the inverse of the weighted empirical distribution, Hyndman and Fan's (1996, *The
American Statistician* 50(4)) first definition with each draw at its weight. An end is always a
draw's own value, and a draw of weight 0 is never read.

The cumulative weights and the share are each rounded, by different sums, so a cumulative weight
whose exact value is the share can land a unit in the last place below it: 20 draws of weight 0.05
sum to a total one bit above 1, and the first five's 0.25 is one bit short of `0.25 * total`, so
the sixth draw would be read where the fifth reaches the quartile. A cumulative weight reaches a
share where it is within a relative `_SLACK` of it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

type _Array = npt.NDArray[np.float64]

_SLACK = 1e-12
"""How far, relative to the total, a cumulative weight may fall short of a share and reach it."""


def weighted_quantiles(
    values: npt.ArrayLike, weights: npt.ArrayLike, shares: Sequence[float]
) -> list[_Array]:
    """The weighted quantiles of `values`, `(draws, ...)`, at each of `shares` in `(0, 1)`, each
    `(...)`: the least draw of weight above 0 whose cumulative weight, the draws sorted, reaches
    the share of the total; NaN where any draw is NaN."""
    array = np.asarray(values, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    flat = array.reshape(array.shape[0], -1)
    missing = np.isnan(flat).any(axis=0)
    kept = weight > 0.0
    flat, weight = flat[kept], weight[kept]
    order = np.argsort(flat, axis=0, kind="stable")
    cumulative = np.cumsum(weight[order], axis=0)
    total = cumulative[-1]
    out = []
    for share in shares:
        reached = cumulative >= share * total * (1.0 - _SLACK)
        index = np.take_along_axis(order, reached.argmax(axis=0)[None], axis=0)
        end = np.take_along_axis(flat, index, axis=0)[0]
        end[missing] = np.nan
        out.append(end.reshape(array.shape[1:]))
    return out
