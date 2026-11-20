"""Budget plans for a goal in the budget's place: a target marginal return or a target return on
spend, beside the fixed-budget plans.

On family 12's true curves (the bench's Meridian home ground, recorded by
`scripts/family12_curves.py` in tests/fixtures/oracles/family12_curves.json), under Meridian's
default box of 30 % either side of each channel's share of the budget, `plan_geos` (CHC's goals) and
`plan_posterior` (the world's own curves as its one draw) plan the quarter's budget, a target
marginal return and a target return on spend. Each matches the outcome of Meridian's optimiser
within 1 %, its rule reproduced below from its documentation, and each is the exact plan that a
solver of the KKT conditions written here apart from the engine and CHC finds, `plan_geos` to
`1e-9` and `plan_posterior` to its grid's bound. The target marginal return's plan stops where each
channel's marginal return crosses the target, or at an end of the channel's range.

On S-shaped means `plan_posterior`'s target return on spend is the most budget a brute force over
the means' interpolants finds, and its search's bound holds every split in a box.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from chc.response import Channel as Response
from chc.response import GeometricAdstock, Hill
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.optimize import brentq

from mixdyn import NATIONAL, Cadence, Channel, MediaPanel, Revenue
from mixdyn.decisions import (
    GeoLimits,
    MarginalReturnTarget,
    Range,
    Readings,
    ReturnOnSpendTarget,
    ReturnTarget,
    plan_geos,
    plan_posterior,
)
from mixdyn.decisions.posterior import _charged, _on_spend, _rates, _searched

FAMILY = json.loads(
    (Path(__file__).parent / "fixtures" / "oracles" / "family12_curves.json").read_text()
)
PLANNED, LENGTH, WORLDS = FAMILY["planned"], FAMILY["length"], FAMILY["worlds"]
BOX = 0.3  # Meridian's default spend constraint either side of a channel's share, fixed budget
GRID = 2048


class Truth:
    """A world's channels as the quarter reads them, written apart from the engine and CHC: each
    channel's adstock over the quarter and the kernel's tail is its history's carryover plus the
    weekly rate times its reach, the kernel Meridian's normalised geometric one, and its gain at a
    rate is its coefficient times the Hill of slope 1 summed over those weeks, less the same at no
    spend."""

    def __init__(self, world: dict[str, Any]) -> None:
        history = np.array(world["history"])
        weeks = PLANNED + LENGTH - 1
        lag = np.arange(history.shape[0], history.shape[0] + weeks)[:, None] - np.arange(LENGTH)
        self.carry, self.reach = [], []
        for c, retention in enumerate(world["retention"]):
            kernel = retention ** np.arange(LENGTH)
            kernel /= kernel.sum()
            spent = np.concatenate([history[:, c], np.zeros(weeks)])
            self.carry.append((kernel * np.where(lag < history.shape[0], spent[lag], 0.0)).sum(1))
            self.reach.append(
                (kernel * ((lag >= history.shape[0]) & (lag < history.shape[0] + PLANNED))).sum(1)
            )
        self.inflexion = np.array(world["inflexion"])
        self.coefficient = np.array(world["coefficient"])
        self.least = (1.0 - BOX) * np.array(world["status_quo"])
        self.most = (1.0 + BOX) * np.array(world["status_quo"])

    def gain(self, c: int, rate: float | np.ndarray) -> np.ndarray:
        """Channel `c`'s gain at each weekly `rate`, against no spend in the quarter."""
        rate = np.asarray(rate, dtype=float)[..., None]
        at = self.carry[c] + rate * self.reach[c]
        hill = at / (at + self.inflexion[c]) - self.carry[c] / (self.carry[c] + self.inflexion[c])
        return self.coefficient[c] * hill.sum(-1)

    def marginal(self, c: int, rate: float) -> float:
        """Channel `c`'s marginal return at a weekly `rate`: its gain's slope a currency unit."""
        at = self.carry[c] + rate * self.reach[c]
        k = self.inflexion[c]
        return float(self.coefficient[c] * np.sum(self.reach[c] * k / (at + k) ** 2)) / PLANNED

    def at_price(self, price: float) -> np.ndarray:
        """Each channel's weekly rate where its marginal return meets `price`, or an end of its
        box: the KKT conditions' plan at that price."""
        rates = []
        for c in range(self.least.size):
            low, high = self.least[c], self.most[c]
            if self.marginal(c, low) <= price:
                rates.append(low)
            elif self.marginal(c, high) >= price:
                rates.append(high)
            else:
                rates.append(
                    brentq(lambda r, c=c: self.marginal(c, r) - price, low, high, xtol=1e-13)
                )
        return np.array(rates)

    def best(self, budget: float) -> np.ndarray:
        """The plan in the box that spends `budget` over the quarter for the most gain: the price
        at which the KKT conditions' plan spends it, by bisection."""
        top = max(self.marginal(c, self.least[c]) for c in range(self.least.size))
        price = brentq(lambda p: PLANNED * self.at_price(p).sum() - budget, 0.0, top, xtol=1e-15)
        return self.at_price(price)

    def total(self, rates: np.ndarray) -> float:
        return float(sum(self.gain(c, r) for c, r in enumerate(rates)))

    def on_spend(self, per_unit: float) -> np.ndarray:
        """The plan of the most budget whose best plan gains `per_unit` a currency unit or more:
        past the marginal target's budget the best gain less `per_unit` a unit falls, concavely."""
        peak = PLANNED * self.at_price(per_unit).sum()
        most = PLANNED * self.most.sum()

        def surplus(budget: float) -> float:
            return self.total(self.best(budget)) - per_unit * budget

        if surplus(most) >= 0.0:
            return self.most.copy()
        return self.best(brentq(surplus, peak, most, xtol=1e-12 * most))


def test_the_worlds_are_family_12s_and_each_channels_gain_is_the_benchs() -> None:
    """Where the bench is importable, which mixdyn's CI does not install: each recorded world is
    family 12's, and each channel's gain over the quarter at each end of its box is the bench's
    oracle's cell's, `causaldyn_bench.scorecard.truth.cells`, to `1e-12`."""
    pytest.importorskip("causaldyn_bench")
    from causaldyn_bench.mmm_decision import BOX as BENCH_BOX
    from causaldyn_bench.mmm_decision import Quarter
    from causaldyn_bench.scorecard.meridian import MERIDIAN
    from causaldyn_bench.scorecard.truth import cells

    for world in WORLDS:
        history = MERIDIAN.world("priors", world["seed"]).history
        quarter = Quarter.after(history, BENCH_BOX)
        np.testing.assert_array_equal(quarter.status_quo, world["status_quo"])
        truth = Truth(world)
        for c, cell in enumerate(cells(history, quarter)):
            for rate in (truth.least[c], truth.most[c]):
                bench = cell.worth(rate) - cell.worth(0.0)
                assert truth.gain(c, rate) == pytest.approx(bench, rel=1e-12, abs=0.0)


def _meridian(
    truth: Truth,
    budget: float,
    *,
    target_roi: float | None = None,
    target_mroi: float | None = None,
    gtol: float = 1e-4,
) -> np.ndarray:
    """Meridian's budget optimiser, `BudgetOptimizer.optimize` of Meridian 2.1
    (`meridian/analysis/optimizer.py`; https://developers.google.com/meridian/docs/user-guide/
    budget-optimization-scenarios), on channels whose incremental outcome is `truth`'s gain, each
    channel's spend over the quarter. Reproduced from its documented rule; Meridian is not
    installed:

    - each channel's spend is bounded by `(1 - spend_constraint_lower)` and
      `(1 + spend_constraint_upper)` times the budget times its historical share, 0.3 each by
      default, here for every scenario, both bounds rounded to the budget's rounding;
    - the grid steps by `10^n`, the budget rounded where `budget * gtol` (default `1e-4`) allows,
      one currency unit where that is below 1, from each bound to the other;
    - the search climbs the grid from every channel's lower bound: it takes, over every channel and
      every point of its grid past its spend, the step of the most incremental outcome per
      currency unit (each ratio rounded to 8 decimals), and stops before the step that breaks the
      scenario: a total spend above the budget (fixed budget); a total ROI below `target_roi` with
      the step's ROI below the total's (flexible, target ROI); or the step's ROI below
      `target_mroi` (flexible, target marginal ROI).

    Meridian computes its outcome by scaling each channel's historical media; here every solver
    reads the same channels, so what differs is the optimiser's rule alone."""
    tolerance = budget * gtol
    rounding = 0 if tolerance < 1.0 else -int(math.log10(tolerance)) - 1
    step = 10.0**-rounding
    allocation = PLANNED * (truth.least + truth.most) / 2.0
    lower = np.maximum((1.0 - BOX) * allocation, 0.0).round(rounding)
    upper = ((1.0 + BOX) * allocation).round(rounding)
    rows = int(np.max(upper - lower) // step) + 1
    channels = allocation.size
    spend_grid = np.full((rows, channels), np.nan)
    outcome_grid = np.full((rows, channels), np.nan)
    for c in range(channels):
        points = np.arange(lower[c], upper[c] + step, step)[:rows]
        spend_grid[: points.size, c] = points
        outcome_grid[: points.size, c] = truth.gain(c, points / PLANNED)
    spend, outcome = spend_grid[0].copy(), outcome_grid[0].copy()
    spend_grid, outcome_grid = spend_grid[1:], outcome_grid[1:]
    roi = np.round((outcome_grid - outcome) / (spend_grid - spend), 8)
    while True:
        optimal = spend.copy()
        if np.isnan(roi).all():
            break
        row, c = np.unravel_index(np.nanargmax(roi), roi.shape)
        spend[c], outcome[c] = spend_grid[row, c], outcome_grid[row, c]
        point = roi[row, c]
        if target_roi is not None:
            total = outcome.sum() / spend.sum()
            stop = total < target_roi and point < total
        elif target_mroi is not None:
            stop = point < target_mroi
        else:
            stop = spend.sum() > budget
        if stop:
            break
        roi[: row + 1, c] = np.nan
        roi[row + 1 :, c] = np.round(
            (outcome_grid[row + 1 :, c] - outcome_grid[row, c])
            / (spend_grid[row + 1 :, c] - spend_grid[row, c]),
            8,
        )
    return optimal / PLANNED


def _panel(world: dict[str, Any]) -> MediaPanel:
    history = np.array(world["history"])[None]
    weeks = history.shape[1]
    return MediaPanel(
        dates=np.datetime64("2026-01-05") + np.timedelta64(7, "D") * np.arange(weeks),
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=tuple(Channel(name) for name in world["channels"]),
        controls=(),
        kpi=np.ones((1, weeks)),
        kpi_unit=Revenue(),
        spend=history,
        currency="EUR",
        control_values=np.zeros((1, weeks, 0)),
        filled=np.zeros((1, weeks), dtype=bool),
    )


def _limits(world: dict[str, Any], truth: Truth) -> dict[Channel, Range]:
    return {
        Channel(name): Range(float(truth.least[c]), float(truth.most[c]))
        for c, name in enumerate(world["channels"])
    }


def _plug_in(world, truth, target):
    responses = {
        NATIONAL: {
            Channel(name): Response(
                GeometricAdstock(world["retention"][c], length=LENGTH, normalized=True),
                Hill(world["inflexion"][c], 1.0),
                world["coefficient"][c],
            )
            for c, name in enumerate(world["channels"])
        }
    }
    cells = {(NATIONAL, channel): box for channel, box in _limits(world, truth).items()}
    plan = plan_geos(_panel(world), responses, target, PLANNED, GeoLimits(cells=cells))
    return plan.spend[0]


def _posterior(world, truth, target):
    channels = len(world["channels"])
    readings = Readings(
        channels=tuple(Channel(name) for name in world["channels"]),
        retention=np.array([world["retention"]]),
        half_saturation=np.array([world["inflexion"]]),
        slope=np.ones((1, channels)),
        coefficient=np.array([world["coefficient"]]),
        tanh=np.array([False]),
        fit=np.array([0]),
        chain=np.array([0]),
        stacking=np.array([1.0]),
        pointwise=None,
        length=LENGTH,
    )
    plan = plan_posterior(
        _panel(world), NATIONAL, readings, target, PLANNED, _limits(world, truth), grid=GRID
    )
    return plan.spend


def _targets(truth: Truth, budget: float) -> tuple[float, float]:
    """A target marginal return at which some channel stops inside its box, the median channel's
    marginal return at the status quo; and a target return on spend met at about 1.15 times the
    budget, between the budget and the box's most."""
    status_quo = (truth.least + truth.most) / 2.0
    marginal = float(np.median([truth.marginal(c, r) for c, r in enumerate(status_quo)]))
    on_spend = truth.total(truth.best(1.15 * budget)) / (1.15 * budget)
    return marginal, on_spend


PLANNERS = {"plan_geos": _plug_in, "plan_posterior": _posterior}


def _exact(truth: Truth, planner: str, scenario: str, spend: np.ndarray, exact: np.ndarray) -> None:
    """`plan_geos` makes CHC's exact plan, to `1e-9` of each channel's spend. `plan_posterior`
    plans the interpolants of its grid, 2048 points across a range of 0.6 times the status quo: a
    budget's gain and a return on spend's budget within `1e-8` of the exact ones, measured within
    `3.2e-9` and `2.8e-9` on these worlds; the marginal target's spend on a point of each grid, so
    within one step of it of the exact spend."""
    if planner == "plan_geos":
        np.testing.assert_allclose(spend, exact, rtol=1e-9, atol=0.0)
    elif scenario == "budget":
        assert truth.total(spend) == pytest.approx(truth.total(exact), rel=1e-8, abs=0.0)
    elif scenario == "on spend":
        assert spend.sum() == pytest.approx(exact.sum(), rel=1e-8, abs=0.0)
    else:
        step = (truth.most - truth.least) / (GRID - 1)
        assert np.all(np.abs(spend - exact) <= step)


@pytest.mark.parametrize("planner", PLANNERS)
@pytest.mark.parametrize("world", WORLDS, ids=lambda w: str(w["seed"]))
def test_on_family_12_the_plans_match_meridians_optimiser_within_1_percent(
    world: dict[str, Any], planner: str
) -> None:
    """The plan's gain, and its budget, within 1 % of those of Meridian's optimiser at its default
    box and grid, on each scenario: measured within 5e-4 on these worlds, Meridian's own grid
    within 5e-4 of the exact plan."""
    truth = Truth(world)
    budget = world["budget"]
    marginal, on_spend = _targets(truth, budget)
    exact = {
        "budget": truth.best(budget),
        "marginal": truth.at_price(marginal),
        "on spend": truth.on_spend(on_spend),
    }
    meridian = {
        "budget": _meridian(truth, budget),
        "marginal": _meridian(truth, budget, target_mroi=marginal),
        "on spend": _meridian(truth, budget, target_roi=on_spend),
    }
    plan = PLANNERS[planner]
    ours = {
        "budget": plan(world, truth, budget),
        "marginal": plan(world, truth, MarginalReturnTarget(marginal)),
        "on spend": plan(world, truth, ReturnOnSpendTarget(on_spend)),
    }
    # the targets put the flexible plans strictly inside the box's budgets
    for scenario in ("marginal", "on spend"):
        assert truth.least.sum() < exact[scenario].sum() < truth.most.sum()
    for scenario, spend in ours.items():
        gain, theirs = truth.total(spend), truth.total(meridian[scenario])
        assert gain == pytest.approx(theirs, rel=0.01, abs=0.0), scenario
        assert spend.sum() == pytest.approx(meridian[scenario].sum(), rel=0.01, abs=0.0), scenario
        _exact(truth, planner, scenario, spend, exact[scenario])
        assert np.all((spend >= truth.least * (1 - 1e-12)) & (spend <= truth.most * (1 + 1e-12)))
    assert PLANNED * ours["budget"].sum() == pytest.approx(budget, rel=1e-12, abs=0.0)
    assert truth.total(ours["on spend"]) / ours["on spend"].sum() >= on_spend * (1 - 1e-9)


@pytest.mark.parametrize("planner", PLANNERS)
@pytest.mark.parametrize("world", WORLDS, ids=lambda w: str(w["seed"]))
def test_the_target_marginal_returns_plan_stops_where_each_channels_marginal_return_crosses_it(
    world: dict[str, Any], planner: str
) -> None:
    """Each channel inside its box sits where its marginal return is the target; at its least the
    marginal return there is at most the target, at its most at least it. `plan_geos` holds this to
    `1e-9` of the target; `plan_posterior` to the change in the channel's marginal return over one
    step of its grid either side, where the slope of the interpolant crosses the target."""
    truth = Truth(world)
    target, _ = _targets(truth, world["budget"])
    spend = PLANNERS[planner](world, truth, MarginalReturnTarget(target))
    inside = 0
    for c, rate in enumerate(spend):
        marginal = truth.marginal(c, rate)
        if planner == "plan_geos":
            tolerance = 1e-9 * target
        else:
            step = (truth.most[c] - truth.least[c]) / (GRID - 1)
            tolerance = truth.marginal(c, max(rate - step, truth.least[c])) - truth.marginal(
                c, min(rate + step, truth.most[c])
            )
        if rate <= truth.least[c] * (1 + 1e-12):
            assert marginal <= target + tolerance
        elif rate >= truth.most[c] * (1 - 1e-12):
            assert marginal >= target - tolerance
        else:
            inside += 1
            assert abs(marginal - target) <= tolerance
    assert inside >= 1


def test_a_channel_held_at_a_point_stays_there_under_either_goal() -> None:
    """An equality on a channel, its range a point, holds under each goal; the other channels still
    stop where their marginal returns cross the target."""
    world = WORLDS[0]
    truth = Truth(world)
    target, on_spend = _targets(truth, world["budget"])
    held = float(truth.most[0])
    limits = _limits(world, truth) | {Channel(world["channels"][0]): Range(held, held)}
    readings = Readings(
        channels=tuple(Channel(name) for name in world["channels"]),
        retention=np.array([world["retention"]]),
        half_saturation=np.array([world["inflexion"]]),
        slope=np.ones((1, 3)),
        coefficient=np.array([world["coefficient"]]),
        tanh=np.array([False]),
        fit=np.array([0]),
        chain=np.array([0]),
        stacking=np.array([1.0]),
        pointwise=None,
        length=LENGTH,
    )
    free = _posterior(world, truth, MarginalReturnTarget(target))
    for goal in (MarginalReturnTarget(target), ReturnOnSpendTarget(on_spend)):
        plan = plan_posterior(_panel(world), NATIONAL, readings, goal, PLANNED, limits, grid=GRID)
        assert plan.spend[0] == held
        if isinstance(goal, MarginalReturnTarget):
            np.testing.assert_array_equal(plan.spend[1:], free[1:])
        else:
            assert plan.gain / plan.budget >= on_spend * (1 - 1e-9)


def test_a_posterior_plan_refuses_a_return_target_and_a_return_on_spend_out_of_reach() -> None:
    world = WORLDS[0]
    truth = Truth(world)
    with pytest.raises(TypeError, match="ReturnOnSpendTarget"):
        _posterior(world, truth, ReturnTarget(1.0))
    with pytest.raises(ValueError, match="needs a finite one"):
        _posterior(world, truth, MarginalReturnTarget(math.nan))
    best = max(truth.marginal(c, truth.least[c]) for c in range(3))
    with pytest.raises(ValueError, match="no budget within the ranges returns"):
        _posterior(world, truth, ReturnOnSpendTarget(10.0 * best))


def _best_split(
    rates: np.ndarray | list[np.ndarray], means: np.ndarray | list[np.ndarray], target: float
) -> float:
    """The most worth on the interpolants of a split of `target` within the grids: every channel
    but one on a point of its grid, that one taking the rest."""
    best = -np.inf
    channels = len(rates)
    for free in range(channels):
        others = [c for c in range(channels) if c != free]
        for points in itertools.product(*(range(len(rates[c])) for c in others)):
            rest = target - sum(rates[c][i] for c, i in zip(others, points, strict=True))
            slack = 1e-12 * max(1.0, abs(target))
            if not rates[free][0] - slack <= rest <= rates[free][-1] + slack:
                continue
            rest = min(max(rest, rates[free][0]), rates[free][-1])
            worth = np.interp(rest, rates[free], means[free]) + sum(
                means[c][i] for c, i in zip(others, points, strict=True)
            )
            best = max(best, float(worth))
    return best


def _most_total(rates: np.ndarray, means: np.ndarray, idle: float, charge: float) -> float:
    """The most total of a split within the grids whose worth on the interpolants less `charge` a
    unit of the total is at least `idle`, by brute force: the worth less the charge is linear in
    each cell of the grids, so a split of the most total has every channel but one on a point of
    its grid, and that one at the rightmost spend its grid's interpolant allows."""
    best = -np.inf
    channels, grid = rates.shape
    surplus = means - charge * rates
    for free in range(channels):
        others = [c for c in range(channels) if c != free]
        for points in itertools.product(*(range(grid) for _ in others)):
            fixed = sum(rates[c, i] for c, i in zip(others, points, strict=True))
            need = idle - sum(surplus[c, i] for c, i in zip(others, points, strict=True))
            h, x = surplus[free], rates[free]
            for k in range(grid - 1, -1, -1):
                if h[k] >= need:
                    if k < grid - 1:
                        # the crossing inside the cell to the right of point k
                        share = (need - h[k]) / (h[k + 1] - h[k])
                        rest = x[k] + share * (x[k + 1] - x[k])
                    else:
                        rest = x[k]
                    best = max(best, fixed + rest)
                    break
    return best


@st.composite
def _s_shaped(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray]:
    """Two or three channels, each mean a Hill curve of slope 0.5 to 5 tilted by a line, on a grid
    small enough to search by brute force."""
    channels = draw(st.integers(2, 3))
    grid = 21 if channels == 2 else 9
    unit = st.floats(0.0, 1.0)
    lower = np.array([draw(unit) for _ in range(channels)])
    upper = lower + np.array([draw(st.floats(0.5, 3.0)) for _ in range(channels)])
    rates = _rates(lower, upper, grid)
    means = np.empty_like(rates)
    for c in range(channels):
        half, slope = draw(st.floats(0.3, 2.5)), draw(st.floats(0.5, 5.0))
        top, tilt = draw(st.floats(0.5, 3.0)), draw(st.floats(-0.05, 0.05))
        means[c] = top * rates[c] ** slope / (rates[c] ** slope + half**slope) + tilt * rates[c]
    return rates, means


@settings(max_examples=300, deadline=None)
@given(_s_shaped(), st.floats(0.0, 1.0), st.floats(0.0, 1.0))
def test_the_most_budget_on_spend_is_a_brute_forces_on_s_shaped_means(
    case: tuple[np.ndarray, np.ndarray], at: float, steep: float
) -> None:
    """A charge up to the means' steepest slope, and the worth of spending nothing set so that the
    best split of a total drawn within the grids meets the target with `1e-9` to spare: the most
    total the search finds is the brute force's, to a share `1e-9` of the grids' most."""
    rates, means = case
    least, most = float(rates[:, 0].sum()), float(rates[:, -1].sum())
    charge = steep * float(np.max(np.diff(means, axis=1) / np.diff(rates, axis=1)))
    total = least + at * (most - least)
    value = _best_split(rates, means, total) - charge * total
    idle = value - 1e-9 * max(1.0, abs(value))
    spend = _on_spend(rates, means, idle, charge)
    assert spend is not None
    best = _most_total(rates, means, idle, charge)
    assert best >= total - 1e-12 * most
    assert spend.sum() == pytest.approx(best, rel=0.0, abs=1e-9 * most)
    assert np.all((spend >= rates[:, 0] - 1e-12) & (spend <= rates[:, -1] + 1e-12))
    worth = sum(np.interp(spend[c], rates[c], means[c]) for c in range(spend.size))
    assert worth - charge * spend.sum() >= idle - 1e-9 * max(1.0, abs(idle))


def _charged_best(
    rates: list[np.ndarray], means: list[np.ndarray], charge: float, least: float, most: float
) -> float:
    """The most worth less `charge` a unit of the total of a split within the grids whose total is
    within `[least, most]`, by brute force: on the interpolants the worth less the charge is linear
    in each cell of the grids, so the best has its total at an end of the stretch, or every
    channel on a point of its grid."""
    ends = [_best_split(rates, means, total) - charge * total for total in (least, most)]
    points = [
        sum(means[c][i] - charge * rates[c][i] for c, i in enumerate(at))
        for at in itertools.product(*(range(len(r)) for r in rates))
        if least <= sum(rates[c][i] for c, i in enumerate(at)) <= most
    ]
    return max(ends + points)


@settings(max_examples=300, deadline=None)
@given(_s_shaped(), st.data())
def test_a_charged_boxs_fill_bounds_every_split_in_the_box_and_the_search_finds_the_best(
    case: tuple[np.ndarray, np.ndarray], data: st.DataObject
) -> None:
    """A charge up to the means' steepest slope, a box of each channel's grid points and a stretch
    of totals the box can spend: the fill down the box's envelopes at its best total in the
    stretch, less the charge, is at least every split's worth less the charge in the box with its
    total in the stretch. Over the whole grids, the search's plan is the brute force's best to
    `2e-9` of it, and its total is in the stretch."""
    rates, means = case
    channels, grid = rates.shape
    charge = data.draw(st.floats(0.0, 1.0)) * float(np.max(np.diff(means) / np.diff(rates)))
    box = []
    for _ in range(channels):
        lo = data.draw(st.integers(0, grid - 2))
        box.append((lo, data.draw(st.integers(lo + 1, grid - 1))))

    def stretch(low: float, high: float) -> tuple[float, float]:
        ends = [low + data.draw(st.floats(0.0, 1.0)) * (high - low) for _ in range(2)]
        return min(ends), max(ends)

    low = sum(rates[c, lo] for c, (lo, _) in enumerate(box))
    high = sum(rates[c, hi] for c, (_, hi) in enumerate(box))
    least, most = stretch(low, high)
    _, top, _ = _charged(rates, means, tuple(box), least, most, charge, {})
    inside = [slice(lo, hi + 1) for lo, hi in box]
    best = _charged_best(
        [rates[c, cells] for c, cells in enumerate(inside)],
        [means[c, cells] for c, cells in enumerate(inside)],
        charge,
        least,
        most,
    )
    assert top >= best - 1e-12 * max(1.0, abs(best))
    least, most = stretch(float(rates[:, 0].sum()), float(rates[:, -1].sum()))
    spend, worth = _searched(rates, means, least, most, charge, {})
    best = _charged_best(list(rates), list(means), charge, least, most)
    assert least - 1e-12 * max(1.0, most) <= spend.sum() <= most + 1e-12 * max(1.0, most)
    assert best - 2e-9 * max(1.0, abs(best)) <= worth <= best + 1e-12 * max(1.0, abs(best))
