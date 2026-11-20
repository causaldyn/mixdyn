"""Ridge regression with bounds on the coefficients, in glmnet's objective, and an anchor if one is
given: the static engines' least-squares core.

    minimise over b0, b:  (1 / 2n) ||y - b0 - X b||^2 + (lam / 2) sum_j (s_j b_j)^2
                              + (1 / 2) sum_j r_j (b_j - m_j)^2
    subject to:           lower <= b <= upper

`s_j` is column j's standard deviation with divisor n, so the penalty reads each coefficient in
units of its column's spread, and a column's currency does not move the fit. `lam` has no units:
scaling `y` scales the solution and leaves `lam` alone. glmnet's gaussian `lambda` is this `lam`
times `y`'s standard deviation with divisor n, since glmnet standardises `y` and keeps its
`lambda` in `y`'s units. The intercept is free and unpenalised, so the problem is the centred one,
with `b0 = mean(y) - mean(X) b`. A column that does not vary has nothing to fit, and its
coefficient is 0.

Without a binding bound every penalty's solution is closed form from one SVD of the centred,
standardised design `Z = U D V'`: `c(lam) = V diag(d / (d^2 + n lam)) U' y_c`, with `b = c / s`.
Where a bound binds, bounded-variable least squares on the augmented system
`[Z / sqrt(n); sqrt(lam) I] c ~ [y_c / sqrt(n); 0]` (Stark and Parker 1995) is exact: an active-set
method that ends at the Karush-Kuhn-Tucker point, and the problem is strictly convex, so that point
is the solution. A coefficient within `1e-12` of the largest standardised coefficient, bounded or
free, from its bound is on it, and is returned as the bound itself.

The anchor (`Anchor`) is what a second source of evidence on the coefficients adds, Gaussian and
independent of `y`: a lift test's estimate `m_j` of a coefficient, with its precision `r_j` in the
objective's units, so that the fit is the joint least squares of the outcome and the tests. With
one the system gains the anchor's rows, `sqrt(q_j) c_j ~ sqrt(q_j) s_j m_j` with
`q_j = r_j / s_j^2`, and is solved by least squares, or by BVLS where a bound binds. A column that
does not vary has only its anchor to fit: its coefficient is `m_j` held to its bounds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import optimize

type _Array = npt.NDArray[np.float64]

_TOLERANCE = 1e-14
"""BVLS's stopping tolerance on its first-order optimality, in the standardised problem's units."""
_SNAP = 1e-12
"""How near its bound, as a share of the largest standardised coefficient, a value is on it."""


@dataclass(frozen=True, slots=True, eq=False)
class Anchor:
    """A pull of each coefficient toward `center` with `precision`, `(p,)` each on the columns' own
    scale: `sum_j precision_j (b_j - center_j)^2 / 2` added to the objective. A precision of 0
    leaves its coefficient to the data.

    Raises:
        ValueError: on a center or precision that is not finite, or a precision below 0.
    """

    center: _Array
    precision: _Array

    def __post_init__(self) -> None:
        center = np.atleast_1d(np.asarray(self.center, dtype=np.float64))
        precision = np.atleast_1d(np.asarray(self.precision, dtype=np.float64))
        if (
            not (np.isfinite(center).all() and np.isfinite(precision).all())
            or (precision < 0.0).any()
        ):
            msg = (
                "an anchor's centers must be finite and its precisions finite and at least 0, got"
                f" {center!r} and {precision!r}"
            )
            raise ValueError(msg)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "precision", precision)


@dataclass(frozen=True, slots=True, eq=False)
class RidgePath:
    """The solution at each penalty: `intercepts` `(k,)`, `coefficients` `(k, p)` on the columns'
    own scale, and `bound` `(k, p)`, whether each coefficient sits on one of its bounds."""

    penalties: _Array
    intercepts: _Array
    coefficients: _Array
    bound: npt.NDArray[np.bool_]


def ridge_path(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    penalties: npt.ArrayLike,
    lower: npt.ArrayLike | None = None,
    upper: npt.ArrayLike | None = None,
    anchor: Anchor | None = None,
) -> RidgePath:
    """Solve the bounded ridge problem of `x` `(n, p)` and `y` `(n,)` at each of `penalties`.

    `lower` and `upper` bound each coefficient on its column's own scale, `-inf` and `inf` by
    default; as glmnet's limits, each lower bound is at most 0 and each upper bound at least 0, so a
    coefficient of 0 is always allowed. `anchor` pulls coefficients toward values a second source
    of evidence holds.

    Raises:
        ValueError: on values that are not finite, shapes that do not agree, fewer than two rows, a
            penalty that is not positive, or a bound on the wrong side of 0.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    penalties = np.atleast_1d(np.asarray(penalties, dtype=np.float64))
    if x.ndim != 2 or y.shape != (x.shape[0],) or x.shape[0] < 2:
        msg = f"x must be (n, p) and y (n,) with n >= 2, got {x.shape} and {y.shape}"
        raise ValueError(msg)
    n, p = x.shape
    lo = np.full(p, -np.inf) if lower is None else np.asarray(lower, dtype=np.float64)
    hi = np.full(p, np.inf) if upper is None else np.asarray(upper, dtype=np.float64)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        msg = "x and y must be finite"
        raise ValueError(msg)
    if penalties.ndim != 1 or not (np.isfinite(penalties) & (penalties > 0.0)).all():
        msg = f"penalties must be positive and finite, got {penalties!r}"
        raise ValueError(msg)
    if lo.shape != (p,) or hi.shape != (p,) or (lo > 0.0).any() or (hi < 0.0).any():
        msg = f"lower must be {p} values at most 0 and upper {p} at least 0, got {lo!r}, {hi!r}"
        raise ValueError(msg)
    if anchor is not None and (anchor.center.shape != (p,) or anchor.precision.shape != (p,)):
        msg = (
            f"an anchor holds {p} centers and {p} precisions, got {anchor.center.shape} and"
            f" {anchor.precision.shape}"
        )
        raise ValueError(msg)
    means = x.mean(axis=0)
    centred = x - means
    scale = np.sqrt(np.mean(centred**2, axis=0))
    live = scale > 0.0
    z = centred[:, live] / scale[live]
    yc = y - y.mean()
    u, d, vt = np.linalg.svd(z, full_matrices=False)
    projected = u.T @ yc
    floor, ceiling = lo[live] * scale[live], hi[live] * scale[live]
    coefficients = np.zeros((penalties.size, p))
    bound = np.zeros((penalties.size, p), dtype=bool)
    # the anchor's rows, sqrt(r_j) (c_j / s_j - m_j), in the standardised coefficients c
    pull, toward = np.zeros((0, z.shape[1])), np.zeros(0)
    if anchor is not None and anchor.precision.any():
        root = np.sqrt(anchor.precision[live])
        pull, toward = np.diag(root / scale[live]), root * anchor.center[live]
        # a column with nothing to fit has only its anchor: its center, held to its bounds
        alone = ~live & (anchor.precision > 0.0)
        held = np.clip(anchor.center[alone], lo[alone], hi[alone])
        coefficients[:, alone] = held
        bound[:, alone] = (held == lo[alone]) | (held == hi[alone])

    def augmented(lam: float) -> tuple[_Array, _Array]:
        system = np.vstack([z / np.sqrt(n), np.sqrt(lam) * np.eye(z.shape[1]), pull])
        return system, np.concatenate([yc / np.sqrt(n), np.zeros(z.shape[1]), toward])

    for i, lam in enumerate(penalties.tolist()):
        if pull.size:
            free = np.linalg.lstsq(*augmented(lam), rcond=None)[0]
        else:
            free = vt.T @ (d / (d**2 + n * lam) * projected)
        c = free
        if (c < floor).any() or (c > ceiling).any():
            c = optimize.lsq_linear(
                *augmented(lam), bounds=(floor, ceiling), method="bvls", tol=_TOLERANCE
            ).x
        # BVLS stops once its cost no longer moves, and can leave a value that belongs on its bound
        # a rounding error off it, on either side. The error is the size of the sums it ran, which
        # the free solution sets: a solution all on bounds at 0 is no measure of it.
        size = max(np.abs(free).max(initial=0.0), np.abs(c).max(initial=0.0))
        snap = _SNAP * max(size, np.finfo(np.float64).tiny)
        at_floor, at_ceiling = c - floor <= snap, ceiling - c <= snap
        own = np.where(at_floor, lo[live], np.where(at_ceiling, hi[live], c / scale[live]))
        coefficients[i, live] = own
        bound[i, live] = at_floor | at_ceiling
    return RidgePath(
        penalties=penalties,
        intercepts=y.mean() - coefficients @ means,
        coefficients=coefficients,
        bound=bound,
    )
