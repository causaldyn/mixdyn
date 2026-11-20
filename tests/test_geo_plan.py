"""mixdyn.decisions.plan_geos: `chc.allocation`'s joint plan of every geo's channels, by name.

One geo is `chc.allocation.allocate`'s plan of its channels; a limit named for a geo or a channel
binds the cells it names; each goal is met on the plan's gain; a plan that holds every cell at its
current spend is the current spend and gains what the engine reads it to gain, observed periods
alone counted. A channel bought nationally is one spend split by population, as one geo's cell it
is that cell's plan, and over two it never returns more than its cells planned apart. On worlds
drawn by `chc.geo_world`, a plan made geo by geo never returns more than the joint plan. A plan
names each cell it moves whose spend `read_panel` flags as never moving, in its geo or in every geo,
and moves no number for it.
"""

from __future__ import annotations

import dataclasses

import jax
import numpy as np
import pytest
from chc.allocation import allocate
from chc.geo_world import GeoMediaMix, MediaChannel
from chc.response import (
    Channel as Response,
)
from chc.response import (
    DelayedAdstock,
    Exponential,
    GeometricAdstock,
    MichaelisMenten,
    Tanh,
)

from mixdyn import (
    Cadence,
    Channel,
    Geo,
    Issue,
    IssueCode,
    MediaPanel,
    Revenue,
    Severity,
    allot_national,
    read_panel,
)
from mixdyn.decisions import (
    GeoLimits,
    MarginalReturnTarget,
    Range,
    ReturnOnSpendTarget,
    ReturnTarget,
    plan_geos,
)

NORTH, SOUTH = Geo("north"), Geo("south")
SEARCH, SOCIAL, TV = Channel("search"), Channel("social"), Channel("tv")
GEOS, CHANNELS = (NORTH, SOUTH), (SEARCH, SOCIAL, TV)
PERIODS = 13
KERNELS = (
    GeometricAdstock(0.3, length=6, normalized=True),
    DelayedAdstock(0.6, 1.5, length=8, normalized=True),
    GeometricAdstock(0.7, length=10, normalized=True),
)
CURVES = (Tanh(250.0), MichaelisMenten(120.0), Exponential(300.0))
SIZES = ((1100.0, 900.0, 1400.0), (500.0, 700.0, 300.0))
RESPONSES = {
    geo: {
        channel: Response(kernel, curve, size)
        for channel, kernel, curve, size in zip(CHANNELS, KERNELS, CURVES, row, strict=True)
    }
    for geo, row in zip(GEOS, SIZES, strict=True)
}
SPEND = np.random.default_rng(7).gamma(4.0, 20.0, size=(2, 40, 3))
LOWER, UPPER = 10.0, 160.0
BOXES = {(g, c): Range(LOWER, UPPER) for g in GEOS for c in CHANNELS}
BUDGET = PERIODS * 320.0


def _panel(geos, channels, spend, filled=None, population=None):
    """A weekly panel of `spend`, `(geo, period, channel)`; a plan does not read the KPI."""
    n_geo, n_period, _ = spend.shape
    filled = np.zeros((n_geo, n_period), dtype=bool) if filled is None else filled
    return MediaPanel(
        dates=np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(n_period),
        cadence=Cadence.WEEK,
        geos=tuple(geos),
        channels=tuple(channels),
        controls=(),
        kpi=np.where(filled, np.nan, 0.0),
        kpi_unit=Revenue(),
        spend=spend,
        currency="EUR",
        control_values=np.where(filled[..., None], np.nan, np.zeros((n_geo, n_period, 0))),
        filled=filled,
        population=population,
    )


PANEL = _panel(GEOS, CHANNELS, SPEND)


def _plan(target=BUDGET, limits=None, panel=PANEL, responses=RESPONSES):
    return plan_geos(
        panel, responses, target, PERIODS, GeoLimits(cells=BOXES) if limits is None else limits
    )


def test_one_geo_is_chc_s_plan_of_its_channels():
    panel = _panel((NORTH,), CHANNELS, SPEND[:1])
    limits = GeoLimits(cells={(NORTH, c): BOXES[NORTH, c] for c in CHANNELS})
    plan = _plan(BUDGET / 2, limits, panel, {NORTH: RESPONSES[NORTH]})
    alone = allocate(
        [RESPONSES[NORTH][c] for c in CHANNELS],
        BUDGET / 2,
        PERIODS,
        lower=np.full(3, LOWER),
        upper=np.full(3, UPPER),
        history=SPEND[0],
    )
    np.testing.assert_allclose(plan.spend[0], alone.spend, rtol=1e-12)
    assert plan.gain == pytest.approx(alone.gain, rel=1e-12)
    assert plan.price == pytest.approx(alone.price, rel=1e-12)
    assert plan.shortfall == 0.0


def test_a_limit_named_for_a_geo_or_a_channel_binds_the_cells_it_names():
    free = _plan()
    cap = 0.85 * PERIODS * free.spend[0].sum()
    floor = 1.15 * PERIODS * free.spend[:, 2].sum()
    plan = _plan(
        limits=GeoLimits(cells=BOXES, geos={NORTH: Range(most=cap)}, channels={TV: Range(floor)})
    )
    assert PERIODS * plan.spend[0].sum() == pytest.approx(cap, rel=1e-12)
    assert PERIODS * plan.spend[:, 2].sum() == pytest.approx(floor, rel=1e-12)
    assert plan.geo_prices[NORTH] > 0.0
    assert plan.channel_prices[TV] < 0.0
    assert (
        plan.geo_prices[SOUTH] == plan.channel_prices[SEARCH] == plan.channel_prices[SOCIAL] == 0.0
    )
    assert plan.gain < free.gain
    assert plan.shortfall == 0.0


@pytest.mark.parametrize(
    "goal",
    [ReturnTarget(9000.0), MarginalReturnTarget(4.0), ReturnOnSpendTarget(4.2)],
    ids=["return", "marginal", "on-spend"],
)
def test_a_goal_is_met_on_the_plan_s_gain(goal):
    plan = _plan(goal)
    match goal:
        case ReturnTarget(amount=amount):
            assert plan.gain == pytest.approx(amount, rel=1e-12)
        case MarginalReturnTarget(per_unit=per_unit):
            assert plan.price == per_unit
        case ReturnOnSpendTarget(per_unit=per_unit):
            # the floors hold spend on poor cells, so the return on spend rises before it falls;
            # the target is met at the most budget
            assert plan.return_on_spend == pytest.approx(per_unit, rel=1e-12)
            assert _plan(plan.budget * (1 + 1e-9)).return_on_spend < per_unit
    assert np.all((plan.spend >= LOWER) & (plan.spend <= UPPER))


def test_a_goal_holds_a_channels_total_held_to_a_point():
    """An equality on TV's total over the plan, its range a point 20 % above the free plan's,
    holds under a target marginal return; the cells of the other channels, which it does not bind,
    still stop where the free plan stops them."""
    goal = MarginalReturnTarget(4.0)
    free = _plan(goal)
    held = 1.2 * PERIODS * free.spend[:, 2].sum()
    plan = _plan(goal, GeoLimits(cells=BOXES, channels={TV: Range(held, held)}))
    assert PERIODS * plan.spend[:, 2].sum() == pytest.approx(held, rel=1e-12, abs=0.0)
    assert plan.price == goal.per_unit
    np.testing.assert_allclose(plan.spend[:, :2], free.spend[:, :2], rtol=1e-9, atol=0.0)


def test_a_plan_held_at_the_current_spend_is_it_and_gains_what_it_gains():
    """The current spend is each cell's mean over the last `PERIODS` periods; the plan that holds
    every cell there gains, by `chc.allocation`, what the plan's own reading of the cells gives."""
    current = SPEND[:, -PERIODS:, :].mean(axis=1)
    held = {
        (g, c): Range(current[i, j], current[i, j])
        for i, g in enumerate(GEOS)
        for j, c in enumerate(CHANNELS)
    }
    plan = _plan(PERIODS * current.sum(), GeoLimits(cells=held))
    np.testing.assert_array_equal(plan.current, current)
    np.testing.assert_allclose(plan.spend, current, rtol=1e-12)
    assert plan.gain == pytest.approx(plan.current_gain, rel=1e-12)
    assert _plan(PERIODS * current.sum()).gain > plan.current_gain


def _read(spend):
    """`spend`, `(geo, period, channel)`, read by `read_panel` from a user's rows, one a geo and
    week, so that a cell whose spend never moves is flagged as it flags one."""
    n_geo, n_period, _ = spend.shape
    dates = np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(n_period)
    data = {
        "date": np.tile(dates, n_geo),
        "geo": np.repeat([str(g) for g in GEOS[:n_geo]], n_period),
        "sales": np.random.default_rng(3).uniform(900.0, 1100.0, n_geo * n_period),
    } | {str(c): spend[..., j].reshape(-1) for j, c in enumerate(CHANNELS)}
    return read_panel(
        data,
        date="date",
        kpi="sales",
        spend={str(c): str(c) for c in CHANNELS},
        cadence=Cadence.WEEK,
        kpi_unit=Revenue(),
        currency="EUR",
        geo="geo",
    )


def _same_numbers(plan, other):
    for name in ("spend", "current", "budget", "gain", "current_gain", "shortfall", "price"):
        np.testing.assert_array_equal(getattr(plan, name), getattr(other, name))
    assert dict(plan.geo_prices) == dict(other.geo_prices)
    assert dict(plan.channel_prices) == dict(other.channel_prices)


def test_a_plan_names_each_cell_it_moves_whose_spend_never_moved():
    """TV spends 80 every week in the north and moves in the south: `read_panel` flags the north's
    cell alone, and the plan, which moves TV off its current spend in both geos, names the north's
    and no other; every number is the plan's on the same panel without the flag, to the bit. Held
    at 80 by a range that is a point, the north's TV is not named. At 80 every week in both geos,
    TV is flagged in every geo, and both cells are named."""
    spend = SPEND.copy()
    spend[0, :, 2] = 80.0
    panel = _read(spend)
    flags = [(i.geo, i.column) for i in panel.issues if i.code is IssueCode.CONSTANT]
    assert flags == [("north", "tv")]
    plan = _plan(panel=panel)
    assert np.all(np.abs(plan.spend[:, 2] - plan.current[:, 2]) > 1.0)
    assert plan.unidentified == ((NORTH, TV),)
    unflagged = _plan(panel=dataclasses.replace(panel, issues=()))
    assert unflagged.unidentified == ()
    _same_numbers(plan, unflagged)
    held = _plan(panel=panel, limits=GeoLimits(cells={**BOXES, (NORTH, TV): Range(80.0, 80.0)}))
    assert held.spend[0, 2] == 80.0
    assert held.unidentified == ()
    spend[1, :, 2] = 80.0
    everywhere = _read(spend)
    flags = [(i.geo, i.column) for i in everywhere.issues if i.code is IssueCode.CONSTANT]
    assert flags == [(None, "tv")]
    assert _plan(panel=everywhere).unidentified == ((NORTH, TV), (SOUTH, TV))


def test_a_channel_bought_nationally_names_its_cells_where_it_moves_its_spend():
    """A channel bought nationally has no cell ranges: its cells are named against its national
    range's most. TV flagged in every geo, its national spend moved, both cells are named; held at
    its current national spend by a range that is that point, none is."""
    flag = Issue(
        IssueCode.CONSTANT, Severity.WARNING, "channel 'tv' never moves", column="tv", channel="tv"
    )
    panel = dataclasses.replace(_bought(), issues=(flag,))
    limits = GeoLimits(cells=_local_boxes(), national={TV: Range(2.0 * LOWER, 2.0 * UPPER)})
    plan = _plan(panel=panel, limits=limits)
    assert abs(plan.spend[:, 2].sum() - plan.current[:, 2].sum()) > 1.0
    assert plan.unidentified == ((NORTH, TV), (SOUTH, TV))
    _same_numbers(plan, _plan(panel=_bought(), limits=limits))
    rate = float(plan.current[:, 2].sum())
    held = _plan(
        panel=panel, limits=GeoLimits(cells=_local_boxes(), national={TV: Range(rate, rate)})
    )
    assert held.unidentified == ()


def test_the_current_spend_counts_observed_periods_only():
    filled = np.zeros((2, 40), dtype=bool)
    filled[0, -3] = True
    spend = SPEND.copy()
    spend[0, -3] = 0.0
    plan = _plan(panel=_panel(GEOS, CHANNELS, spend, filled))
    window = np.delete(spend[0, -PERIODS:], -3, axis=0)
    np.testing.assert_allclose(plan.current[0], window.mean(axis=0), rtol=1e-15)


POPULATION = np.array([3e5, 7e5])
SHARES = POPULATION / POPULATION.sum()


def _bought(geos=GEOS, population=POPULATION, spend=SPEND):
    """The panel with TV bought nationally, its history the two geos' TV spend summed, split by
    population."""
    local = _panel(geos, CHANNELS[:2], spend[:, :, :2], population=population)
    return allot_national(local, TV, spend[:, :, 2].sum(axis=0))


def _local_boxes(geos=GEOS):
    return {(g, c): BOXES[g, c] for g in geos for c in CHANNELS[:2]}


def test_a_channel_bought_nationally_is_one_spend_split_by_population():
    """TV is one national spend a period within its range, each geo's cell its population share;
    its cells planned apart, each boxed at its share of that range, may do better and never
    worse. Here they do: apart, north's TV reads its larger coefficient at 15 times its share of
    the national least, a plan no national buy places."""
    national = Range(2.0 * LOWER, 2.0 * UPPER)
    plan = _plan(panel=_bought(), limits=GeoLimits(cells=_local_boxes(), national={TV: national}))
    rate = plan.spend[:, 2].sum()
    np.testing.assert_allclose(plan.spend[:, 2], SHARES * rate, rtol=1e-12, atol=0.0)
    assert national.least <= rate <= national.most
    assert PERIODS * plan.spend.sum() == pytest.approx(BUDGET, rel=1e-12, abs=0.0)
    assert plan.shortfall == 0.0

    apart = _local_boxes() | {
        (g, TV): Range(s * national.least, s * national.most)
        for g, s in zip(GEOS, SHARES, strict=True)
    }
    free = _plan(panel=_panel(GEOS, CHANNELS, _bought().spend), limits=GeoLimits(cells=apart))
    assert plan.gain <= free.gain * (1 + 1e-12)
    assert free.gain - plan.gain > 0.01 * free.gain
    assert free.spend[0, 2] > 10.0 * SHARES[0] * rate


def test_a_goal_plans_a_channel_bought_nationally_as_one_spend():
    limits = GeoLimits(cells=_local_boxes(), national={TV: Range(2.0 * LOWER, 2.0 * UPPER)})
    plan = _plan(ReturnTarget(9000.0), limits, _bought())
    assert plan.gain == pytest.approx(9000.0, rel=1e-12, abs=0.0)
    np.testing.assert_allclose(
        plan.spend[:, 2], SHARES * plan.spend[:, 2].sum(), rtol=1e-12, atol=0.0
    )


def test_a_channel_bought_nationally_in_one_geo_is_its_cell_s_plan():
    panel = _bought((NORTH,), POPULATION[:1], SPEND[:1])
    limits = GeoLimits(cells=_local_boxes((NORTH,)), national={TV: BOXES[NORTH, TV]})
    plan = _plan(BUDGET / 2, limits, panel, {NORTH: RESPONSES[NORTH]})
    cells = GeoLimits(cells={(NORTH, c): BOXES[NORTH, c] for c in CHANNELS})
    alone = _plan(
        BUDGET / 2, cells, _panel((NORTH,), CHANNELS, SPEND[:1]), {NORTH: RESPONSES[NORTH]}
    )
    np.testing.assert_allclose(plan.spend, alone.spend, rtol=1e-9, atol=0.0)
    assert plan.gain == pytest.approx(alone.gain, rel=1e-12, abs=0.0)


@pytest.mark.parametrize(
    ("panel", "limits", "match"),
    [
        (
            "bought",
            GeoLimits(cells=BOXES, national={TV: Range(0.0, 100.0)}),
            r"the cells \[\('north', 'tv'\), \('south', 'tv'\)\] are of a channel bought",
        ),
        ("bought", GeoLimits(cells=_local_boxes()), r"missing \['tv'\]"),
        (
            "bought",
            GeoLimits(cells=_local_boxes(), national={TV: Range(0.0)}),
            r"no finite most \['tv'\]",
        ),
        (
            "local",
            GeoLimits(cells=BOXES, national={TV: Range(0.0, 100.0)}),
            r"not bought nationally \['tv'\]",
        ),
    ],
)
def test_it_refuses_a_national_channel_ranged_as_cells_or_not_at_all(panel, limits, match):
    with pytest.raises(ValueError, match=match):
        _plan(panel=_bought() if panel == "bought" else PANEL, limits=limits)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"responses": {NORTH: RESPONSES[NORTH]}}, r"the responses miss cells \[\('south'"),
        (
            {"limits": GeoLimits(cells={k: v for k, v in BOXES.items() if k != (SOUTH, TV)})},
            r"missing \[\('south', 'tv'\)\]",
        ),
        (
            {"limits": GeoLimits(cells=BOXES | {(SOUTH, TV): Range(0.0)})},
            r"no finite most \[\('south', 'tv'\)\]",
        ),
        (
            {"limits": GeoLimits(cells=BOXES, geos={Geo("west"): Range(most=1.0)})},
            r"geo totals name \['west'\]",
        ),
        ({"target": 1e9}, r"outside what the box spends .* \(geos in order \['north', 'south'\]"),
    ],
)
def test_it_refuses_a_plan_it_cannot_make_naming_what_is_wrong(change, match):
    arguments = {
        "target": BUDGET,
        "limits": GeoLimits(cells=BOXES),
        "responses": RESPONSES,
    } | change
    with pytest.raises(ValueError, match=match):
        _plan(**arguments)


def test_it_refuses_to_plan_in_single_precision():
    with jax.enable_x64(False), pytest.raises(ValueError, match="double precision"):
        _plan()
    with jax.enable_x64(False):
        single = Response(KERNELS[0], CURVES[0], 1100.0)
    responses = {NORTH: RESPONSES[NORTH] | {SEARCH: single}, SOUTH: RESPONSES[SOUTH]}
    with pytest.raises(ValueError, match=r"\[\('north', 'search'\)\] hold single-precision"):
        _plan(responses=responses)


def test_a_range_is_ordered():
    with pytest.raises(ValueError, match="a range needs"):
        Range(5.0, 1.0)
    with pytest.raises(ValueError, match="a range needs"):
        Range(-1.0)


DRAWN = GeoMediaMix(
    (
        MediaChannel("search", MichaelisMenten(2.0), 3.0, spread=0.4, retention=0.3, spend=1.5),
        MediaChannel("social", Exponential(1.0), 1.5, spread=0.5, retention=0.5, spend=0.8),
        MediaChannel("video", Tanh(3.0), 2.0, spread=0.3, retention=0.6, spend=1.2),
        MediaChannel("tv", Tanh(4.0), 2.0, spread=0.2, retention=0.7, spend=2.0, national=True),
    ),
    geos=8,
    weeks=52,
)


def _split(budget, weights, least, most):
    """`budget` in proportion to `weights`, each part clipped into `[least, most]`: the clipped
    parts' sum rises with the multiplier on the weights, so bisection finds the one that spends
    the budget."""
    low, high = 0.0, float(np.max(most / weights))
    for _ in range(200):
        middle = 0.5 * (low + high)
        if np.clip(middle * weights, least, most).sum() < budget:
            low = middle
        else:
            high = middle
    return np.clip(high * weights, least, most)


@pytest.mark.parametrize("seed", range(10))
def test_on_a_drawn_world_a_plan_made_geo_by_geo_never_returns_more(seed):
    """Eight geos drawn by `chc.geo_world`, three concave channels each buys and a national one
    held where it ran, every cell within half and twice its last quarter's spend a period. The
    budget split over the geos first, in proportion to each geo's average return on its current
    spend or to that spend, then each geo's share planned over its channels: one plan of the grid,
    never above the joint plan. The average reads where the margin decides and leaves more than a
    percent behind; the split as it stands, more than a tenth of one."""
    world = DRAWN.draw(seed)
    geos = tuple(Geo(f"geo {g}") for g in range(DRAWN.geos))
    channels = tuple(Channel(c.name) for c in DRAWN.channels)
    panel = _panel(geos, channels, world.spend)
    responses = {
        g: dict(zip(channels, row, strict=True)) for g, row in zip(geos, world.cells(), strict=True)
    }
    current = world.spend[:, -PERIODS:, :].mean(axis=1)
    held = np.array([c.national for c in DRAWN.channels])
    lower = np.where(held, current, 0.5 * current)
    upper = np.where(held, current, 2.0 * current)
    cells = {
        (g, c): Range(lower[i, j], upper[i, j])
        for i, g in enumerate(geos)
        for j, c in enumerate(channels)
    }
    budget = PERIODS * float(current.sum())
    joint = plan_geos(panel, responses, budget, PERIODS, GeoLimits(cells=cells))
    assert joint.shortfall == 0.0

    returns = np.array(
        [
            [float(cell.coefficient * cell.curve(current[i, j])) for j, cell in enumerate(row)]
            for i, row in enumerate(world.cells())
        ]
    )
    least = np.array([PERIODS * float(row.sum()) for row in lower])
    most = np.array([PERIODS * float(row.sum()) for row in upper])
    for weights, behind in (
        (returns.sum(axis=1) / current.sum(axis=1), 0.01),
        (current.sum(axis=1), 1e-3),
    ):
        totals = _split(budget, weights, least, most)
        limits = GeoLimits(
            cells=cells, geos={g: Range(t, t) for g, t in zip(geos, totals, strict=True)}
        )
        two = plan_geos(panel, responses, budget, PERIODS, limits)
        np.testing.assert_allclose(PERIODS * two.spend.sum(axis=1), totals, rtol=1e-12)
        assert two.gain <= joint.gain * (1 + 1e-12)
        assert joint.gain - two.gain > behind * joint.gain
