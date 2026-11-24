"""Fuzz `mixdyn.decisions.margin_plan`'s planners, `plan_corner` and `water_fill`, on any box.

    uv run --with-requirements fuzz/requirements.txt python fuzz/fuzz_plan.py -max_total_time=60

Each input decodes into up to six channels:

* margins: positive, zero, negative, tied, or far apart in size;
* boxes: points, wide, inverted, or not finite;
* a budget: inside the box's sum, on either end of it, or outside it;
* for `water_fill`, each channel's curve as the margin regressions plan it (`Curve`): a
  Michaelis-Menten saturation `H'(p) = s r / (1 + (c + p r) / K)^2`, its slope 1 at a status quo
  inside the box as the module asks, or a line, where `K` is infinite, whose slope is not read.

A planner refuses the box and budget with `ValueError`, or its plan keeps the promises of its
docstring, to rounding:

* the plan lies in the box and spends the budget;
* the plan is the most gain. Moving spend from a channel above its floor to one below its
  ceiling gains at most the difference of their marginal gains, the margin itself on a line and
  `m H'(p)` on a curve, times the spend moved: an upper bound on a concave program, and no move
  may gain more than `GAIN` of the largest marginal gain times the budget's size. Where curves
  are nearly lines, the price's bisection resolves their spends only to some billionths of the
  budget, and the module gives that rest by the margins' order, not the marginal gains'.

Anything else, a broken promise or another exception, is a crash with the input that caused it.
"""

from __future__ import annotations

import math
import sys

import atheris
import numpy as np
from atheris.import_hook import HookManager

# Branch coverage of the whole package, without atheris's comparison tracing: in atheris 3.1.0
# that tracing keeps each numpy array a traced comparison returns (41 MB over 3 000 inputs of
# the planners), and `instrument_imports` always turns it on.
with HookManager({"mixdyn"}, set(), enable_loader_override=True, trace_dataflow=False):
    from mixdyn.decisions.margin_plan import plan_corner, water_fill
    from mixdyn.models.margin_mmm import Curve

RELATIVE = 1e-9
"""Rounding allowed, relative to the largest bound or the budget, on the box and the sum."""
GAIN = 1e-9
"""The most a move of spend may gain, relative to the largest marginal gain times the budget's
size."""


def _size(provider: atheris.FuzzedDataProvider) -> float:
    kind = provider.ConsumeIntInRange(0, 5)
    if kind <= 2:
        return 10.0 ** provider.ConsumeFloatInRange(-3.0, 6.0)
    if kind == 3:
        return 0.0
    if kind == 4:
        return provider.ConsumeFloatInRange(-10.0, 10.0)
    return [math.nan, math.inf, -math.inf, 5e-324][provider.ConsumeIntInRange(0, 3)]


def _margins(provider: atheris.FuzzedDataProvider, channels: int) -> np.ndarray:
    margins = np.empty(channels)
    for c in range(channels):
        kind = provider.ConsumeIntInRange(0, 5)
        if kind <= 2 or c == 0:
            margins[c] = provider.ConsumeFloatInRange(-1.0, 5.0)
        elif kind == 3:
            margins[c] = margins[provider.ConsumeIntInRange(0, c - 1)]  # a tie
        elif kind == 4:
            margins[c] = 10.0 ** provider.ConsumeFloatInRange(-8.0, 4.0)
        else:
            margins[c] = _size(provider)
    return margins


def _box(
    provider: atheris.FuzzedDataProvider, channels: int
) -> tuple[np.ndarray, np.ndarray, float]:
    low = np.array([abs(_size(provider)) for _ in range(channels)])
    width = np.array([_size(provider) for _ in range(channels)])
    high = low + np.where(provider.ConsumeIntInRange(0, 3) == 0, 0.0, width)
    floor, ceiling = float(np.sum(low)), float(np.sum(high))
    kind = provider.ConsumeIntInRange(0, 6)
    if kind <= 3:
        budget = floor + provider.ConsumeProbability() * (ceiling - floor)
    elif kind == 4:
        budget = floor
    elif kind == 5:
        budget = ceiling
    else:
        budget = _size(provider)
    return low, high, budget


def _curve(provider: atheris.FuzzedDataProvider, low: np.ndarray, high: np.ndarray) -> Curve:
    channels = low.size

    def positive(lo: float, hi: float) -> np.ndarray:
        return np.array([10.0 ** provider.ConsumeFloatInRange(lo, hi) for _ in range(channels)])

    carry, reach, line = positive(-3.0, 4.0), positive(-2.0, 1.0), positive(-2.0, 2.0)
    carry[[provider.ConsumeBool() for _ in range(channels)]] = 0.0
    half = positive(-1.0, 5.0)
    half[[provider.ConsumeIntInRange(0, 3) == 0 for _ in range(channels)]] = math.inf
    quo = low + np.array([provider.ConsumeProbability() for _ in range(channels)]) * (high - low)
    with np.errstate(all="ignore"):  # a box that is not finite is refused before a slope is read
        at_quo = reach / (1.0 + (carry + quo * reach) / half) ** 2
        scale = np.where(np.isfinite(half), 1.0 / at_quo, line)
    return Curve(
        channel=np.arange(channels),
        carry=carry[:, None],
        reach=reach[:, None],
        half_saturation=half[:, None],
        scale=scale,
        channels=channels,
    )


def _keeps_promises(
    plan: np.ndarray, gain: np.ndarray, low: np.ndarray, high: np.ndarray, budget: float, case: str
) -> None:
    size = max(1.0, abs(budget), float(np.max(np.abs(low))), float(np.max(np.abs(high))))
    slack = RELATIVE * size * plan.size
    if plan.shape != low.shape or not np.all(np.isfinite(plan)):
        raise AssertionError(f"{case}: plan {plan}")
    if np.any(plan < low - slack) or np.any(plan > high + slack):
        raise AssertionError(f"{case}: plan {plan} outside the box [{low}, {high}]")
    if abs(float(np.sum(plan)) - budget) > slack:
        raise AssertionError(f"{case}: plan {plan} spends {np.sum(plan)}, not {budget}")
    rise = np.where(plan < high - slack, high - plan, 0.0)
    fall = np.where(plan > low + slack, plan - low, 0.0)
    move = (gain[:, None] - gain[None, :]) * np.minimum(rise[:, None], fall[None, :])
    to, away = np.unravel_index(int(np.argmax(move)), move.shape)
    if move[to, away] > GAIN * max(float(np.max(np.abs(gain))), 1e-300) * size:
        raise AssertionError(
            f"{case}: plan {plan}: moving spend from channel {away} (marginal gain {gain[away]}) "
            f"to channel {to} ({gain[to]}) gains up to {move[to, away]}"
        )


def check(data: bytes) -> None:
    provider = atheris.FuzzedDataProvider(data)
    channels = provider.ConsumeIntInRange(1, 6)
    margins = _margins(provider, channels)
    low, high, budget = _box(provider, channels)
    case = f"margins {margins.tolist()}, box [{low.tolist()}, {high.tolist()}], budget {budget}"
    if provider.ConsumeBool():
        try:
            plan = plan_corner(margins, low, high, budget)
        except ValueError:
            return
        _keeps_promises(plan, margins, low, high, budget, f"plan_corner: {case}")
        return
    curve = _curve(provider, low, high)
    shaped = np.isfinite(curve.half_saturation[:, 0])
    case = (
        f"water_fill: {case}, carry {curve.carry[:, 0].tolist()}, "
        f"reach {curve.reach[:, 0].tolist()}, K {curve.half_saturation[:, 0].tolist()}, "
        f"scale {curve.scale.tolist()}"
    )
    try:
        plan = water_fill(margins, curve.slope, shaped, low, high, budget)
    except ValueError:
        return
    curved = shaped & (margins > 0)
    gain = np.where(curved, margins * curve.slope(plan), margins)
    _keeps_promises(plan, gain, low, high, budget, case)


def main() -> None:
    atheris.Setup(sys.argv, check)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
