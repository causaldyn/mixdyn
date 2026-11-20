"""An arm's fitted response as the bench maps a tool's (`causaldyn_bench.scorecard.mapping`): its
draws of each channel in `chc.response`'s terms, beside the fit's own account of what each channel
returned over the history, by which the bench checks the mapping before it reads a return off it.

Each draw's channel is a geometric kernel, a `tanh` or a Hill, and a coefficient: a number, or
where the fit's coefficient moves over the history's weeks, its path there, the channel then read
at a coefficient of 1 and the path times it, as the bench reads a coefficient that moves. A
channel's curves are listed draw by draw, so a fit whose draws mix curves maps as one whose draws
do not. The account is each draw's total over the history, and in each week for `HELD` draws.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from chc import response

if TYPE_CHECKING:
    from collections.abc import Sequence

HELD = 4
"""The draws whose every week a record keeps, evenly spaced over its draws."""


def held(draws: int) -> np.ndarray:
    """The indices of the draws whose every week a record keeps, `HELD` of `draws` evenly spaced,
    or every draw where there are fewer."""
    return np.unique(np.linspace(0, draws - 1, min(draws, HELD)).round().astype(int))


def mapped(
    names: Sequence[str],
    draws: Sequence[Sequence[response.Channel]],
    totals: np.ndarray,
    weekly: np.ndarray,
    paths: np.ndarray | None = None,
) -> dict[str, Any]:
    """The record's `response`: each draw's channels, in the order of `names`, beside what the fit
    says each returned over the history in each draw, `totals` `(draws, channels)`, and in each
    week of the draws `held` names, `weekly` `(held, weeks, channels)`. Where `paths` is given,
    `(draws, weeks, channels)`, each draw's coefficient is its path, and its channel's own, 1, is
    not read.

    Raises:
        ValueError: on arrays of other shapes, a kernel other than a geometric one of the first's
            length and normalisation, or a curve other than a `tanh` or a Hill.
    """
    count, kept, channels = len(draws), held(len(draws)), len(names)
    weeks = weekly.shape[1] if weekly.ndim == 3 else -1
    if (
        any(len(draw) != channels for draw in draws)
        or totals.shape != (count, channels)
        or weekly.shape != (kept.size, weeks, channels)
        or (paths is not None and paths.shape != (count, weeks, channels))
    ):
        msg = (
            f"{count} draws of {channels} channels need totals {(count, channels)} and weeks "
            f"{(kept.size, weeks, channels)}, not {totals.shape} and {weekly.shape}"
        )
        raise ValueError(msg)
    first = draws[0][0].kernel
    if not isinstance(first, response.GeometricAdstock):
        msg = f"a response maps geometric kernels, not {type(first).__name__}"
        raise ValueError(msg)
    curves: dict[str, list[str]] = {}
    parameters: dict[str, dict[str, list[Any]]] = {}
    for j, name in enumerate(names):
        kinds: list[str] = []
        retention: list[float] = []
        scale: list[float] = []
        slope: list[float | None] = []
        for draw in draws:
            kernel, curve = draw[j].kernel, draw[j].curve
            if not (
                isinstance(kernel, response.GeometricAdstock)
                and (kernel.length, kernel.normalized) == (first.length, first.normalized)
            ):
                msg = f"{name}'s kernels are not all geometric of the first's length and norm"
                raise ValueError(msg)
            if isinstance(curve, response.Hill):
                kinds.append("Hill")
                slope.append(float(curve.slope))
            elif isinstance(curve, response.Tanh):
                kinds.append("Tanh")
                slope.append(None)
            else:
                msg = (
                    f"{name}'s curve is a {type(curve).__name__}; a response maps a Tanh or a Hill"
                )
                raise ValueError(msg)
            retention.append(float(kernel.retention))
            scale.append(float(curve.scale))
        values: dict[str, list[Any]] = {"retention": retention, "scale": scale}
        if "Hill" in kinds:
            values["slope"] = slope
        values["coefficient"] = (
            [float(draw[j].coefficient) for draw in draws]
            if paths is None
            else paths[:, :, j].tolist()
        )
        curves[name] = kinds
        parameters[name] = values
    return {
        "kernel": "GeometricAdstock",
        "length": first.length,
        "normalized": first.normalized,
        "curve": curves,
        "parameters": parameters,
        "decomposition": {
            "total": {name: totals[:, j].tolist() for j, name in enumerate(names)},
            "held": kept.tolist(),
            "weekly": {name: weekly[:, :, j].tolist() for j, name in enumerate(names)},
        },
    }
