"""The margin plan (`mixdyn.decisions.margin_plan`): on six of M2's worlds the shrunk means are the
research code's and the corner is its plan exactly; the posterior's covariance at one `tau` is the
law of total variance over the common level (the identity checked in Maxima, exactly at rational
points and symbolically for two channels); no positive margin shrinks to nothing; the corner
spends the budget inside the box, fills the highest margin first and breaks ties by the channels'
order, and plans a channel it fills at its ceiling to the last bit; and the observed range cuts
each box to the spend its history logged, raising the typed error where the cut box holds no plan
on the budget. The time guard names a channel whose
weekly spend correlates with the week's number above 0.8 either way, by the definition's
correlation; a step and a ramp are named where their closed forms say (checked in Maxima); a
spend that never moves, or moves in its last bits alone, reads 0 and is not named; and a history
it does not read is refused. Water-filling meets its KKT conditions on
random Michaelis-Menten curves, at the closed form (checked in Maxima), and on random mixtures of
them; the box binds a channel whose margin outruns or trails the price; a margin at most 0 stays
at its floor; the positive channels' ceilings hand the rest to the corner; a line it fills is at its
ceiling to the last bit; and on lines it is the corner. The adstock range cuts the observed range
further, and falls back to it where the cut box holds no plan on the budget. Water-filling on a
table spends the budget inside the box within its table's steps of the exact plan, on
Michaelis-Menten curves and on mixtures with logistic ones; on margins that never fall it is the
corner, and a margin at most 0 stays at its floor."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from margin_worlds import PARITY, name
from mixdyn.decisions import margin_plan
from mixdyn.decisions.margin_plan import plan_corner, shrink_margins

RELATIVE = 1e-9


@pytest.mark.parametrize("world", PARITY, ids=name)
def test_the_shrunk_means_and_the_corner_are_the_research_codes(world):
    research = world["research"]
    shrunk = shrink_margins(research["mean"], research["cov"])
    assert shrunk is not None
    np.testing.assert_allclose(shrunk.mean, research["shrunk"], rtol=RELATIVE, atol=0.0)
    budget = float(np.sum(world["status_quo"]))
    plan = plan_corner(research["shrunk"], world["lower"], world["upper"], budget)
    np.testing.assert_array_equal(plan, research["plan"])


@pytest.mark.parametrize("world", PARITY, ids=name)
def test_the_corner_spends_the_budget_inside_the_box(world):
    lower, upper = np.asarray(world["lower"]), np.asarray(world["upper"])
    budget = float(np.sum(world["status_quo"]))
    plan = plan_corner(world["research"]["shrunk"], lower, upper, budget)
    assert np.all(plan >= lower)
    assert np.all(plan <= upper)
    assert plan.sum() == pytest.approx(budget, rel=1e-12, abs=0.0)
    assert world["planned"] * plan.sum() == pytest.approx(world["budget"], rel=1e-12, abs=0.0)


@pytest.mark.parametrize("world", PARITY, ids=name)
def test_the_posterior_is_symmetric_positive_and_its_tau_on_the_grid(world):
    shrunk = shrink_margins(world["research"]["mean"], world["research"]["cov"])
    assert shrunk is not None
    np.testing.assert_array_equal(shrunk.cov, shrunk.cov.T)
    assert np.all(np.linalg.eigvalsh(shrunk.cov) > 0.0)
    assert margin_plan.TAU_GRID[0] <= shrunk.tau <= margin_plan.TAU_GRID[-1]


@pytest.mark.parametrize("tau_rel", [0.05, 0.5, 2.0])
def test_at_one_tau_the_posterior_is_the_law_of_total_variance_over_the_common_level(
    tau_rel, monkeypatch
):
    """With `mu` flat, `mu | b ~ N(mu_hat, 1 / 1' S^-1 1)`, `S = V + tau^2 I`, and given `mu` the
    margins' mean is `mu 1 + tau^2 S^-1 (b - mu 1)` and their covariance `tau^2 I - tau^4 S^-1`."""
    monkeypatch.setattr(margin_plan, "TAU_GRID", np.array([tau_rel]))
    b = np.array([1.5, -0.25, 0.7])
    v = np.array([[0.4, 0.1, 0.05], [0.1, 0.3, -0.03], [0.05, -0.03, 0.2]])
    shrunk = shrink_margins(b, v)
    assert shrunk is not None
    tau = tau_rel * float(np.mean(b[b > 0]))
    one = np.ones(3)
    si = np.linalg.inv(v + tau**2 * np.eye(3))
    level = float(one @ si @ b) / float(one @ si @ one)
    lean = one - tau**2 * (si @ one)
    mean = level * one + tau**2 * (si @ (b - level * one))
    cov = tau**2 * np.eye(3) - tau**4 * si + np.outer(lean, lean) / float(one @ si @ one)
    np.testing.assert_allclose(shrunk.mean, mean, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(shrunk.cov, cov, rtol=1e-10, atol=1e-14)
    assert shrunk.tau == tau_rel


def test_no_positive_margin_shrinks_to_nothing():
    assert shrink_margins([-1.0, 0.0, -0.5], np.eye(3)) is None


def test_the_highest_margin_fills_to_its_ceiling_then_the_next():
    plan = plan_corner([0.5, 2.0, 1.0], [1.0, 1.0, 1.0], [3.0, 4.0, 5.0], 9.0)
    np.testing.assert_array_equal(plan, [1.0, 4.0, 4.0])


@pytest.mark.parametrize(
    ("margins", "plan"),
    [
        ([1.0, 1.0, 0.0], [5.0, 2.0, 1.0]),
        ([0.0, 1.0, 1.0], [1.0, 5.0, 2.0]),
        ([1.0, 0.0, 1.0], [5.0, 1.0, 2.0]),
        ([1.0, 1.0, 1.0], [5.0, 2.0, 1.0]),
    ],
)
def test_equal_margins_fill_in_the_channels_order(margins, plan):
    np.testing.assert_array_equal(plan_corner(margins, [1.0] * 3, [5.0] * 3, 8.0), plan)


@pytest.mark.parametrize(
    ("lower", "upper", "budget"),
    [([1.0, 1.0], [2.0, 2.0], 1.5), ([1.0, 1.0], [2.0, 2.0], 4.5), ([1.0, 3.0], [2.0, 2.0], 3.5)],
)
def test_a_box_no_plan_at_the_budget_fits_in_is_refused(lower, upper, budget):
    with pytest.raises(ValueError, match="no plan in the box"):
        plan_corner([1.0, 2.0], lower, upper, budget)


def test_margins_that_are_not_finite_are_refused():
    with pytest.raises(ValueError, match="finite"):
        plan_corner([1.0, np.nan], [1.0, 1.0], [2.0, 2.0], 3.0)


def test_a_channel_the_corner_fills_is_planned_at_its_ceiling_to_the_last_bit():
    """`0.7 + (2.9 - 0.7)` is `2.9000000000000004`: the floor plus the room above it misses the
    ceiling by a unit in the last place, and the plan left its box."""
    plan = plan_corner([2.0, 1.0], [0.7, 1.0], [2.9, 5.0], 4.0)
    assert plan[0] == 2.9
    assert plan[1] == pytest.approx(1.1, rel=1e-15, abs=0.0)


@pytest.mark.parametrize("seed", range(40))
def test_every_channel_the_corner_fills_is_at_its_ceiling_and_none_past_it(seed):
    """On random boxes, some of whose ceilings the floor plus the room misses either way, a
    channel within rounding of its ceiling is at it exactly, and none is past it."""
    rng = np.random.default_rng(seed)
    count = int(rng.integers(2, 6))
    quo, margins = rng.uniform(5.0, 50.0, count), rng.uniform(-0.5, 3.0, count)
    lower, upper = quo * rng.uniform(0.0, 0.9, count), quo * rng.uniform(1.1, 3.0, count)
    plan = plan_corner(margins, lower, upper, float(quo.sum()))
    near = np.abs(plan - upper) <= 1e-12 * upper
    np.testing.assert_array_equal(plan[near], upper[near])
    assert np.all(plan <= upper)


HISTORY = np.array([[4.0, 1.0, 6.0], [6.0, 3.0, 6.0], [5.0, 2.0, 6.0]])
"""Three weeks of three channels: the first logged 4 to 6, the second 1 to 3, the third 6 alone."""


def test_the_observed_range_cuts_each_box_to_the_spend_its_history_logged():
    """The first channel's box reaches past both ends of its range, the second's lies inside its
    range, and the third's holds its one level."""
    floor, ceiling = margin_plan.observed_range(HISTORY, [2.0, 1.5, 3.0], [8.0, 2.5, 12.0], 12.0)
    np.testing.assert_array_equal(floor, [4.0, 1.5, 6.0])
    np.testing.assert_array_equal(ceiling, [6.0, 2.5, 6.0])


@pytest.mark.parametrize(
    ("lower", "upper", "budget"),
    [
        ([2.0, 3.5, 3.0], [8.0, 5.0, 12.0], 12.0),
        ([2.0, 1.5, 3.0], [8.0, 2.5, 12.0], 11.0),
        ([2.0, 1.5, 3.0], [8.0, 2.5, 12.0], 14.6),
    ],
    ids=["a box past the range", "floors past the budget", "ceilings short of it"],
)
def test_an_observed_range_with_no_plan_on_the_budget_raises_the_typed_error(lower, upper, budget):
    with pytest.raises(margin_plan.ObservedRangeError, match="holds no plan of"):
        margin_plan.observed_range(HISTORY, lower, upper, budget)


@pytest.mark.parametrize(
    ("history", "match"),
    [
        (HISTORY[:, :2], "the history is \\(weeks, channels\\)"),
        (np.where(HISTORY == 5.0, np.nan, HISTORY), "finite"),
    ],
    ids=["shapes", "not finite"],
)
def test_an_observed_range_of_input_it_does_not_read_is_refused_untyped(history, match):
    with pytest.raises(ValueError, match=match) as raised:
        margin_plan.observed_range(history, [2.0, 1.5, 3.0], [8.0, 2.5, 12.0], 12.0)
    assert not isinstance(raised.value, margin_plan.ObservedRangeError)


def _correlated(weeks, r, seed=0):
    """A weekly spend about a level of 100 whose correlation with the week's number is `r`: the
    centred weeks and a draw made orthogonal to them and to the level, each of length 1, mixed."""
    week = np.arange(weeks, dtype=float)
    along = week - week.mean()
    along /= np.linalg.norm(along)
    across = np.random.default_rng(seed).standard_normal(weeks)
    across -= across.mean()
    across -= (across @ along) * along
    across /= np.linalg.norm(across)
    return 100.0 + 10.0 * (r * along + np.sqrt(1.0 - r * r) * across)


def test_the_time_guard_reads_each_channels_correlation_with_the_week_and_names_those_past_it():
    """A rising ramp, a falling one, a spend at a correlation of 0.5 and one that never moves:
    the correlation is the definition's, its sign dropped, and the ramps alone are named."""
    weeks = 156
    week = np.arange(weeks, dtype=float)
    history = np.column_stack(
        [50.0 + week, 300.0 - 2.0 * week, _correlated(weeks, -0.5), np.full(weeks, 40.0)]
    )
    correlation, held = margin_plan.time_held(history)
    np.testing.assert_allclose(correlation, [1.0, 1.0, 0.5, 0.0], rtol=0.0, atol=1e-12)
    assert correlation[3] == 0.0
    np.testing.assert_array_equal(held, [True, True, False, False])
    centred = history[:, :3] - history[:, :3].mean(axis=0)
    by_hand = np.abs(centred.T @ (week - week.mean())) / (
        np.linalg.norm(centred, axis=0) * np.linalg.norm(week - week.mean())
    )
    np.testing.assert_allclose(correlation[:3], by_hand, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("sign", [1.0, -1.0], ids=["rising", "falling"])
def test_the_time_guard_names_a_channel_just_above_its_threshold_and_not_one_just_below(sign):
    assert margin_plan.TIME_HELD == 0.8
    history = np.column_stack(
        [_correlated(104, sign * 0.7999), _correlated(104, sign * 0.8001, seed=1)]
    )
    correlation, held = margin_plan.time_held(history)
    np.testing.assert_allclose(correlation, [0.7999, 0.8001], rtol=0.0, atol=1e-12)
    np.testing.assert_array_equal(held, [False, True])


def test_a_spend_that_moves_in_its_last_bits_alone_is_a_constant_to_the_time_guard():
    """A level spend whose last bits drift with the week: its correlation with the week by the
    definition is 1, and the guard reads a constant, 0. A ramp a millionth of the level high is a
    ramp."""
    assert margin_plan.CONSTANT == 1e-9
    week = np.arange(156, dtype=float)
    drift = 100.0 * (1.0 + 5e-10 * week / 155.0)
    ramp = 100.0 * (1.0 + 1e-6 * week / 155.0)
    assert np.ptp(drift) > 0.0
    assert abs(np.corrcoef(drift, week)[0, 1]) > 0.99
    correlation, held = margin_plan.time_held(np.column_stack([drift, ramp, -drift]))
    np.testing.assert_array_equal(correlation[[0, 2]], [0.0, 0.0])
    assert correlation[1] == pytest.approx(1.0, rel=0.0, abs=1e-6)
    np.testing.assert_array_equal(held, [False, True, False])


@pytest.mark.parametrize("weeks", [12, 156])
def test_a_steps_correlation_with_the_week_is_its_closed_form_and_names_it_by_where_it_lies(
    weeks,
):
    """A spend that steps up at the week `k` of `T` has a squared correlation with the week of
    `3 k (T - k) / (T^2 - 1)` (`validation/time_guard.mac`), so it is named where that is above
    0.64: at 156 weeks, a step from the week 49 to the week 107."""
    week = np.arange(weeks)
    steps = np.arange(1, weeks)
    history = np.column_stack([np.where(week >= k, 150.0, 100.0) for k in steps])
    correlation, held = margin_plan.time_held(history)
    closed = 3.0 * steps * (weeks - steps) / (weeks**2 - 1.0)
    np.testing.assert_allclose(correlation**2, closed, rtol=1e-12, atol=0.0)
    np.testing.assert_array_equal(held, closed > 0.64)
    if weeks == 156:
        assert (steps[held].min(), steps[held].max()) == (49, 107)


def test_a_ramp_then_a_plateau_is_named_from_36_percent_of_the_history():
    """A spend that rises in a straight line over the first share `b` of a long history and is
    level after it: its squared correlation with the week tends to `b (3 - 2 b)^2 / (4 - 3 b)`
    (`validation/time_guard.mac`), past 0.64 from `b` of 0.36."""
    weeks = 20000
    week = np.arange(weeks, dtype=float)
    shares = np.array([0.2, 0.35, 0.37, 0.75, 1.0])
    history = np.column_stack([100.0 + 50.0 * np.minimum(week / (b * weeks), 1.0) for b in shares])
    correlation, held = margin_plan.time_held(history)
    closed = shares * (3.0 - 2.0 * shares) ** 2 / (4.0 - 3.0 * shares)
    np.testing.assert_allclose(correlation**2, closed, rtol=1e-3, atol=0.0)
    np.testing.assert_array_equal(held, [False, False, True, True, True])


@pytest.mark.parametrize(
    ("history", "match"),
    [
        (HISTORY[:, 0], "the history is \\(weeks, channels\\)"),
        (HISTORY[:1], "over 2 weeks or more"),
        (np.where(HISTORY == 5.0, np.inf, HISTORY), "finite"),
    ],
    ids=["shape", "one week", "not finite"],
)
def test_the_time_guard_refuses_a_history_it_does_not_read(history, match):
    with pytest.raises(ValueError, match=match):
        margin_plan.time_held(history)


@pytest.mark.skipif(shutil.which("maxima") is None, reason="maxima is not on PATH")
def test_maxima_proves_the_time_guards_reach_at_every_step():
    script = Path(__file__).parents[1] / "validation" / "time_guard.mac"
    out = subprocess.run(
        ["maxima", "--very-quiet", "-b", str(script)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    ).stdout
    assert "Lisp error" not in out
    verdicts = [line.strip() for line in out.splitlines() if line.startswith("STEP ")]
    assert len(verdicts) == 5
    assert all(line.endswith("(PASS)") for line in verdicts), verdicts


def _michaelis_menten(half, quo):
    """Each channel's Michaelis-Menten slope over its slope at the status quo,
    `((K + s) / (K + p))^2`."""
    return lambda p: ((half + quo) / (half + np.asarray(p, dtype=float))) ** 2


def _mixture(halves, weights, quo):
    """Each channel's mixture of Michaelis-Menten slopes, each 1 at the status quo, of half
    saturations `halves` and weights `weights`, `(channels, terms)`."""
    return lambda p: np.sum(
        weights * ((halves + quo[:, None]) / (halves + np.asarray(p, dtype=float)[:, None])) ** 2,
        axis=1,
    )


def _random_curves(seed):
    """A world of 2 to 5 channels: margins, status quo, a box holding it and the budget, and
    half-saturations from half the status quo to 20 times it."""
    rng = np.random.default_rng(seed)
    count = int(rng.integers(2, 6))
    quo = rng.uniform(5.0, 50.0, count)
    margins = rng.uniform(0.2, 3.0, count)
    lower = quo * rng.uniform(0.0, 0.9, count)
    upper = quo * rng.uniform(1.1, 3.0, count)
    return rng, quo, margins, lower, upper, float(quo.sum())


def _kkt(plan, marginal, lower, upper, budget):
    """The plan spends the budget inside the box, its free channels' marginal returns are equal,
    a channel at its floor returns at most that price and one at its ceiling at least; returns the
    free channels and the price."""
    assert plan.sum() == pytest.approx(budget, rel=1e-12, abs=0.0)
    assert np.all(plan >= lower)
    assert np.all(plan <= upper)
    free = (plan > lower) & (plan < upper)
    if free.any():
        price = float(np.mean(marginal[free]))
        np.testing.assert_allclose(marginal[free], price, rtol=1e-9, atol=0.0)
    else:
        price = float(np.max(marginal[plan == lower], initial=0.0))
    assert np.all(marginal[(plan == lower) & ~free] <= price * (1.0 + 1e-9))
    assert np.all(marginal[(plan == upper) & ~free] >= price * (1.0 - 1e-9))
    return free, price


@pytest.mark.parametrize("seed", range(20))
def test_water_filling_meets_its_kkt_conditions_on_michaelis_menten_curves(seed):
    """Each free channel spends where its curve `beta a / (K + a)`, through its margin `m` at the
    status quo `s` (`beta = m (K + s)^2 / K`), meets the price: `p = sqrt(beta K / lam) - K`
    (checked in Maxima)."""
    rng, quo, margins, lower, upper, budget = _random_curves(seed)
    half = quo * rng.uniform(0.5, 20.0, quo.size)
    slope = _michaelis_menten(half, quo)
    plan = margin_plan.water_fill(margins, slope, np.ones(quo.size, bool), lower, upper, budget)
    free, price = _kkt(plan, margins * slope(plan), lower, upper, budget)
    beta = margins * (half + quo) ** 2 / half
    np.testing.assert_allclose(
        plan[free], np.sqrt(beta[free] * half[free] / price) - half[free], rtol=1e-9, atol=0.0
    )


@pytest.mark.parametrize("seed", range(20))
def test_water_filling_meets_its_kkt_conditions_on_mixtures_of_curves(seed):
    rng, quo, margins, lower, upper, budget = _random_curves(seed)
    halves = quo[:, None] * rng.uniform(0.5, 20.0, (quo.size, 4))
    weights = rng.dirichlet(np.ones(4), quo.size)
    slope = _mixture(halves, weights, quo)
    np.testing.assert_allclose(slope(quo), 1.0, rtol=1e-12, atol=0.0)
    plan = margin_plan.water_fill(margins, slope, np.ones(quo.size, bool), lower, upper, budget)
    _kkt(plan, margins * slope(plan), lower, upper, budget)


def test_a_channel_whose_margin_outruns_the_others_at_its_ceiling_is_held_there():
    """The first channel's margin stays above the price across its box, the last's below it:
    the box binds both, and the middle channel takes the rest."""
    quo = np.array([10.0, 10.0, 10.0])
    slope = _michaelis_menten(16.0 * quo, quo)
    margins = np.array([50.0, 1.0, 0.01])
    lower, upper = np.array([5.0, 2.0, 4.0]), np.array([12.0, 30.0, 20.0])
    plan = margin_plan.water_fill(margins, slope, np.ones(3, bool), lower, upper, 30.0)
    assert (plan[0], plan[2]) == (upper[0], lower[2])
    assert plan[1] == pytest.approx(30.0 - upper[0] - lower[2], rel=1e-12, abs=0.0)
    _kkt(plan, margins * slope(plan), lower, upper, 30.0)


def test_a_channel_whose_margin_is_not_positive_stays_at_its_floor():
    quo = np.array([10.0, 10.0, 10.0])
    slope = _michaelis_menten(16.0 * quo, quo)
    margins = np.array([1.0, -0.5, 0.0])
    lower, upper = np.array([2.0, 3.0, 4.0]), np.array([40.0, 40.0, 40.0])
    plan = margin_plan.water_fill(margins, slope, np.ones(3, bool), lower, upper, 30.0)
    assert (plan[1], plan[2]) == (3.0, 4.0)
    assert plan[0] == pytest.approx(30.0 - 7.0, rel=1e-12, abs=0.0)


def test_where_the_positive_channels_ceilings_leave_budget_the_rest_goes_by_the_corner():
    """The one positive channel at its ceiling spends 12 of 30: the rest by the corner on the
    others' margins, the higher first."""
    quo = np.array([10.0, 10.0, 10.0])
    slope = _michaelis_menten(16.0 * quo, quo)
    margins = np.array([1.0, -0.5, -0.1])
    lower, upper = np.array([2.0, 3.0, 4.0]), np.array([12.0, 40.0, 40.0])
    plan = margin_plan.water_fill(margins, slope, np.ones(3, bool), lower, upper, 30.0)
    np.testing.assert_array_equal(plan, [12.0, 3.0, 15.0])


def test_a_line_is_planned_on_its_margin_whatever_its_slope_reads():
    """A line's marginal return is its margin: the price's bisection must start where the line
    drops to its floor, here above 1, not at twice the margins times the slopes at the floors,
    0.212, where the line still sits at its ceiling and the floors and it spend 25 of 22."""
    quo = np.array([10.0, 10.0])
    curve = _michaelis_menten(16.0 * quo, quo)

    def slope(p):
        return np.array([curve(p)[0], 0.1])

    margins = np.array([0.1, 1.0])
    lower, upper = np.array([5.0, 5.0]), np.array([20.0, 20.0])
    plan = margin_plan.water_fill(margins, slope, np.array([True, False]), lower, upper, 22.0)
    np.testing.assert_array_equal(plan, [5.0, 17.0])


def test_a_line_water_filling_fills_is_planned_at_its_ceiling_to_the_last_bit():
    """The curve at its ceiling, two lines at the price take what is left, the first to its
    ceiling, which its floor plus its room, `0.7 + (2.9 - 0.7)`, misses by a unit in the last
    place."""
    quo = np.array([10.0, 10.0, 10.0])
    slope = _michaelis_menten(16.0 * quo, quo)
    margins = np.array([2.0, 1.0, 1.0])
    lower, upper = np.array([5.0, 0.7, 0.7]), np.array([30.0, 2.9, 20.0])
    plan = margin_plan.water_fill(
        margins, slope, np.array([True, False, False]), lower, upper, 35.0
    )
    assert (plan[0], plan[1]) == (30.0, 2.9)
    assert plan[2] == pytest.approx(35.0 - 30.0 - 2.9, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("shaped", [False, True])
@pytest.mark.parametrize("seed", range(10))
def test_on_lines_water_filling_is_the_corner(seed, shaped):
    """Curves whose slope is 1 everywhere, read as lines or solved as curves, plan as the corner
    plans their margins, some of them not positive."""
    rng, quo, margins, lower, upper, budget = _random_curves(seed)
    margins = margins - rng.uniform(0.0, 1.0, quo.size)
    plan = margin_plan.water_fill(
        margins, np.ones_like, np.full(quo.size, shaped), lower, upper, budget
    )
    np.testing.assert_allclose(
        plan, plan_corner(margins, lower, upper, budget), rtol=1e-12, atol=1e-12
    )


ADSTOCK = np.array([[4.5, 1.2, 6.0], [5.5, 2.4, 6.0], [5.0, 2.0, 6.0]])
"""`HISTORY`'s adstock: a narrower range than its spend's on the first two channels."""


def test_the_adstock_range_cuts_the_observed_range_to_each_channels_adstock():
    floor, ceiling = margin_plan.adstock_range(
        HISTORY, ADSTOCK, [2.0, 1.5, 3.0], [8.0, 2.5, 12.0], 12.0
    )
    np.testing.assert_array_equal(floor, [4.5, 1.5, 6.0])
    np.testing.assert_array_equal(ceiling, [5.5, 2.4, 6.0])


def test_an_adstock_range_with_no_plan_on_the_budget_falls_back_to_the_observed_range():
    """The adstock's ceilings sum to 13.9 and the budget is 14.2: the observed range stands."""
    floor, ceiling = margin_plan.adstock_range(
        HISTORY, ADSTOCK, [2.0, 1.5, 3.0], [8.0, 2.5, 12.0], 14.2
    )
    np.testing.assert_array_equal(floor, [4.0, 1.5, 6.0])
    np.testing.assert_array_equal(ceiling, [6.0, 2.5, 6.0])


@pytest.mark.parametrize("toward", [0.0, 12.0])
def test_a_constant_channels_adstock_a_last_bit_off_its_spend_does_not_empty_the_cut_box(toward):
    """The third channel's spend never moves, so its adstock is its spend; computed, it lands a
    last bit to either side. That bit must not put the adstock's range outside the spend's and
    send every channel back to the observed range."""
    adstock = ADSTOCK.copy()
    adstock[:, 2] = np.nextafter(6.0, toward)
    floor, ceiling = margin_plan.adstock_range(
        HISTORY, adstock, [2.0, 1.5, 3.0], [8.0, 2.5, 12.0], 12.0
    )
    np.testing.assert_array_equal(floor, [4.5, 1.5, 6.0])
    np.testing.assert_array_equal(ceiling, [5.5, 2.4, 6.0])


def test_an_adstock_of_another_shape_is_refused():
    with pytest.raises(ValueError, match="the adstock is the history's shape"):
        margin_plan.adstock_range(HISTORY, ADSTOCK[:, :2], [2.0, 1.5, 3.0], [8.0, 2.5, 12.0], 12.0)


def _tabulated(margins, slope):
    """Each channel's margin curve as `water_fill_tabulated` reads it, from `water_fill`'s
    margins at the status quo and slopes."""

    def curve(c):
        def margin(p):
            spends = np.zeros((p.size, margins.size))
            spends[:, c] = p
            return margins[c] * np.array([slope(row)[c] for row in spends])

        return margin

    return [curve(c) for c in range(margins.size)]


def _near_the_exact_plan(plan, exact, lower, upper, budget):
    """The tabulated plan spends the budget inside the box. At its price each channel sits at
    most one of its table's steps below its exact demand, and the price lies below the exact one
    by no more than moves the channels by their steps summed: so each channel lies within its
    own step and the steps' sum of the exact plan."""
    assert plan.sum() == pytest.approx(budget, rel=1e-12, abs=0.0)
    assert np.all(plan >= lower)
    assert np.all(plan <= upper)
    step = (upper - lower) / (margin_plan.POINTS - 1)
    assert np.all(np.abs(plan - exact) <= step + step.sum())


@pytest.mark.parametrize("seed", range(20))
def test_water_filling_on_a_table_is_the_exact_plan_to_its_steps_on_michaelis_menten(seed):
    rng, quo, margins, lower, upper, budget = _random_curves(seed)
    half = quo * rng.uniform(0.5, 20.0, quo.size)
    slope = _michaelis_menten(half, quo)
    exact = margin_plan.water_fill(margins, slope, np.ones(quo.size, bool), lower, upper, budget)
    plan = margin_plan.water_fill_tabulated(_tabulated(margins, slope), lower, upper, budget)
    _near_the_exact_plan(plan, exact, lower, upper, budget)


@pytest.mark.parametrize("seed", range(20))
def test_water_filling_on_a_table_is_the_exact_plan_to_its_steps_on_mixtures(seed):
    """Each channel's slope mixes Michaelis-Menten curves, a logistic one,
    `(cosh(k s) / cosh(k p))^2`, and the line."""
    rng, quo, margins, lower, upper, budget = _random_curves(seed)
    halves = quo[:, None] * rng.uniform(0.5, 20.0, (quo.size, 3))
    rate = rng.uniform(0.1, 1.5, quo.size) / quo
    weights = rng.dirichlet(np.ones(5), quo.size)

    def slope(p):
        p = np.asarray(p, dtype=float)
        curved = np.sum(weights[:, :3] * ((halves + quo[:, None]) / (halves + p[:, None])) ** 2, 1)
        logistic = weights[:, 3] * (np.cosh(rate * quo) / np.cosh(rate * p)) ** 2
        return curved + logistic + weights[:, 4]

    np.testing.assert_allclose(slope(quo), 1.0, rtol=1e-12, atol=0.0)
    exact = margin_plan.water_fill(margins, slope, np.ones(quo.size, bool), lower, upper, budget)
    _kkt(exact, margins * slope(exact), lower, upper, budget)
    plan = margin_plan.water_fill_tabulated(_tabulated(margins, slope), lower, upper, budget)
    _near_the_exact_plan(plan, exact, lower, upper, budget)


@pytest.mark.parametrize("seed", range(10))
def test_on_margins_that_never_fall_water_filling_on_a_table_is_the_corner(seed):
    _, _, margins, lower, upper, budget = _random_curves(seed)
    flat = [lambda p, m=m: np.full(p.shape, m) for m in margins]
    plan = margin_plan.water_fill_tabulated(flat, lower, upper, budget)
    np.testing.assert_allclose(
        plan, plan_corner(margins, lower, upper, budget), rtol=1e-12, atol=1e-12 * budget
    )


def test_on_a_table_a_margin_at_most_0_stays_at_its_floor_and_a_high_one_at_its_ceiling():
    quo = np.array([10.0, 10.0, 10.0, 10.0])
    lower, upper = 0.5 * quo, 2.0 * quo
    half = 4.0 * quo
    levels = np.array([50.0, 1.0, 0.95, -0.3])

    def curve(c):
        if levels[c] <= 0:
            return lambda p: np.full(p.shape, levels[c])
        return lambda p: levels[c] * ((half[c] + quo[c]) / (half[c] + p)) ** 2

    plan = margin_plan.water_fill_tabulated([curve(c) for c in range(4)], lower, upper, 40.0)
    assert plan.sum() == pytest.approx(40.0, rel=1e-12, abs=0.0)
    assert (plan[0], plan[3]) == (upper[0], lower[3])
    assert lower[2] < plan[2] < plan[1] < upper[1]
    at = levels[1:3] * ((half[1:3] + quo[1:3]) / (half[1:3] + plan[1:3])) ** 2
    assert at[0] == pytest.approx(at[1], rel=1e-3, abs=0.0)


def test_water_filling_on_a_table_refuses_a_box_that_holds_no_plan():
    with pytest.raises(ValueError, match="no plan in the box"):
        margin_plan.water_fill_tabulated([lambda p: 1.0 / (1.0 + p)] * 2, [0, 0], [1, 1], 3.0)
