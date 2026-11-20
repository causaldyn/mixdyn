"""Scenarios read as differences: each draw's channels as `chc.response` reads them, run over the
history, the plan and the kernel's reach after it; every scenario at the same draws and noise; a
move between geos that neither makes nor loses money; and response curves at the draws' weights."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from fractions import Fraction
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest
from chc import response

from mixdyn import Channel, MediaPanel
from mixdyn.decisions.posterior import Readings, Worths
from mixdyn.decisions.scenarios import (
    Scenario,
    compare_scenarios,
    media_difference,
    move_between_geos,
    response_curves,
)
from mixdyn.models.bayes_mmm import BayesFit, LogNormal, fit_bayes, forecast_bayes
from test_bayes_mmm import NATIONAL, stan_world

LENGTH = 8
"""A kernel of 8 periods: a spend's carryover reaches 7 periods past it."""
HORIZON = 6
LINEAR = response.Power(1.0, 1.0)


def draws(seed: int, count: int = 5) -> list[dict[str, response.Channel]]:
    """`count` draws of two channels, Hill curves through normalised geometric kernels, their
    coefficients from 0.5 to 4, never 1, so that a coefficient applied twice is seen."""
    rng = np.random.default_rng(seed)
    return [
        {
            c: response.Channel(
                response.GeometricAdstock(rng.uniform(0.3, 0.8), length=LENGTH, normalized=True),
                response.Hill(rng.uniform(40.0, 120.0), rng.uniform(0.8, 2.5)),
                rng.choice([0.5, 1.5, 2.5, 4.0]),
            )
            for c in ("tv", "search")
        }
        for _ in range(count)
    ]


def history(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {c: rng.uniform(20.0, 150.0, 30) for c in ("tv", "search")}


def plan(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {c: rng.uniform(20.0, 150.0, HORIZON) for c in ("tv", "search")}


def contribution(channel: response.Channel, past: np.ndarray, spend: np.ndarray) -> float:
    """The channel's contribution over the plan's periods and its kernel's reach after them, the
    past's carryover counted in, as `chc.response` computes it."""
    series = np.concatenate([past, spend, np.zeros(channel.kernel.length - 1)])
    return response.contribution(channel, series, slice(past.size, None))


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_zeroing_a_channel_takes_away_its_contribution_and_its_carry_out(seed: int) -> None:
    """At the history's own spend run on, the plan that zeroes `tv` differs from it by `tv`'s
    contribution over the plan's periods and the 7 after them less what the history alone carries
    into them, at each draw, to 1e-12; `search` cancels."""
    channels, past, base = draws(seed), history(seed), plan(seed)
    zeroed = base | {"tv": np.zeros(HORIZON)}
    got = media_difference(channels, past, base, zeroed, HORIZON)
    expected = [
        contribution(d["tv"], past["tv"], np.zeros(HORIZON))
        - contribution(d["tv"], past["tv"], base["tv"])
        for d in channels
    ]
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0.0)
    assert (got < 0.0).all()


def test_with_no_history_zeroing_a_channel_takes_away_its_whole_contribution() -> None:
    channels, base = draws(4), plan(4)
    empty = {c: np.zeros(0) for c in base}
    got = media_difference(channels, empty, base, base | {"search": np.zeros(HORIZON)}, HORIZON)
    whole = [contribution(d["search"], np.zeros(0), base["search"]) for d in channels]
    np.testing.assert_allclose(got, -np.array(whole), rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("seed", [5, 6, 7])
def test_doubling_a_channel_s_spend_moves_the_difference_by_what_its_response_says(
    seed: int,
) -> None:
    """Twice `search`'s planned spend adds its contribution at twice the spend less at once, at
    each draw, to 1e-12: positive wherever the coefficient is, never 0."""
    channels, past, base = draws(seed), history(seed), plan(seed)
    doubled = base | {"search": 2.0 * base["search"]}
    got = media_difference(channels, past, base, doubled, HORIZON)
    expected = [
        contribution(d["search"], past["search"], doubled["search"])
        - contribution(d["search"], past["search"], base["search"])
        for d in channels
    ]
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0.0)
    assert (got > 0.0).all()


def test_the_carry_out_past_the_plan_counts() -> None:
    """One unit of a linear channel through a normalised kernel spent at the plan's last period
    returns its coefficient whole only with the 7 periods after it counted."""
    kernel = response.GeometricAdstock(0.6, length=LENGTH, normalized=True)
    channels = [{"tv": response.Channel(kernel, LINEAR, 3.0)}]
    spend = np.zeros(HORIZON)
    spend[-1] = 1.0
    got = media_difference(
        channels, {"tv": np.zeros(4)}, {"tv": np.zeros(HORIZON)}, {"tv": spend}, HORIZON
    )
    np.testing.assert_allclose(got, [3.0], rtol=1e-14, atol=0.0)


@pytest.mark.parametrize("seed", [8, 9])
def test_a_plan_identical_to_another_differs_from_it_by_exactly_zero(seed: int) -> None:
    channels, past, base = draws(seed), history(seed), plan(seed)
    copy = {c: v.copy() for c, v in base.items()}
    np.testing.assert_array_equal(media_difference(channels, past, base, copy, HORIZON), 0.0)


def test_a_difference_is_signed_from_the_first_plan_to_the_second() -> None:
    channels, past, base = draws(10), history(10), plan(10)
    more = base | {"tv": base["tv"] + 10.0}
    forth = media_difference(channels, past, base, more, HORIZON)
    back = media_difference(channels, past, more, base, HORIZON)
    np.testing.assert_allclose(forth, -back, rtol=1e-12, atol=0.0)
    assert (forth > 0.0).all()


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda a: a.update(periods=0), "periods"),
        (lambda a: a.update(channel_draws=[]), "at least one draw"),
        (lambda a: a.update(plan_b={"tv": np.ones(HORIZON)}), "plan_b must name"),
        (lambda a: a.update(history={"tv": np.ones(3)}), "history must name"),
        (lambda a: a["plan_a"].update(tv=np.ones(HORIZON + 1)), "shape"),
        (lambda a: a["plan_a"].update(tv=-np.ones(HORIZON)), "at or above 0"),
        (lambda a: a["history"].update(tv=np.full(3, np.nan)), "finite"),
    ],
)
def test_what_cannot_be_differenced_is_refused(
    change: Callable[[dict[str, Any]], None], match: str
) -> None:
    arguments: dict[str, Any] = {
        "channel_draws": draws(11, 2),
        "history": history(11),
        "plan_a": plan(11),
        "plan_b": plan(12),
        "periods": HORIZON,
    }
    change(arguments)
    with pytest.raises(ValueError, match=match):
        media_difference(**arguments)


def linear_geo(coefficient: float) -> list[dict[str, response.Channel]]:
    """A geo whose one channel returns `coefficient` a unit, with no carryover."""
    kernel = response.GeometricAdstock(0.0, length=1, normalized=True)
    return [{"tv": response.Channel(kernel, LINEAR, coefficient)}]


def kpi_change(
    before: dict[str, dict[str, np.ndarray]],
    after: dict[str, dict[str, np.ndarray]],
    responses: dict[str, list[dict[str, response.Channel]]],
) -> float:
    """What the move changes in the KPI, each geo's difference read through its own channels."""
    return sum(
        float(
            media_difference(
                responses[g], {"tv": np.zeros(0)}, before[g], after[g], before[g]["tv"].size
            )[0]
        )
        for g in before
    )


def total(plan: dict[str, dict[str, np.ndarray]]) -> float:
    return sum(float(v.sum()) for cells in plan.values() for v in cells.values())


@pytest.mark.parametrize(
    ("spends", "coefficients", "expected"),
    [((100.0, 1000.0), (2.0, 2.0), 0.0), ((100.0, 300.0), (2.0, 1.0), -20.0)],
)
def test_a_move_between_geos_neither_makes_nor_loses_money(
    spends: tuple[float, float], coefficients: tuple[float, float], expected: float
) -> None:
    """Two linear geos: moving 20 from the first to the second changes the KPI by 20 times the
    gap in their returns, exactly, and the total spend not at all; a move by multipliers would
    have read +360 and +20."""
    plan = {g: {"tv": np.full(1, s)} for g, s in zip(("north", "south"), spends, strict=True)}
    responses = {g: linear_geo(c) for g, c in zip(plan, coefficients, strict=True)}
    moved = move_between_geos(plan, "tv", "north", "south", 20.0)
    assert kpi_change(plan, moved, responses) == expected
    assert total(moved) == total(plan)
    assert moved["north"]["tv"][0] == spends[0] - 20.0
    assert moved["south"]["tv"][0] == spends[1] + 20.0


def test_a_move_keeps_the_two_cells_sum_to_the_last_bit() -> None:
    """Cells and amounts in cents, which no binary fraction holds, over cells of sizes up to 10^4
    times apart: each period's two cells sum to the same float before and after, to the same real
    number wherever two floats can hold it and within 1.5 units in the smaller's last place
    elsewhere, and the amount moved is within a unit in the last
    place of the larger cell of the amount asked."""
    rng = np.random.default_rng(13)
    exact = 0
    for _ in range(400):
        a, b = np.round(rng.uniform(0.0, 1e4, 2) * 10.0 ** rng.integers(-2, 3, 2), 2)
        amount = round(float(rng.uniform(0.0, a)), 2)
        plan = {"north": {"tv": np.array([a, 0.1])}, "south": {"tv": np.array([b, 0.2])}}
        moved = move_between_geos(plan, "tv", "north", "south", [amount, 0.1])
        for k in range(2):
            s, t = (float(moved[g]["tv"][k]) for g in plan)
            assert s >= 0.0
            assert s + t == plan["north"]["tv"][k] + plan["south"]["tv"][k]
        s, t = (float(moved[g]["tv"][0]) for g in plan)
        whole = Fraction(float(a)) + Fraction(float(b))
        exact += Fraction(s) + Fraction(t) == whole
        step = Fraction(float(np.spacing(min(s, t))))
        assert abs(Fraction(s) + Fraction(t) - whole) <= 3 * step / 2
        assert abs((a - s) - amount) <= np.spacing(max(s, t))
    assert exact > 200
    every = move_between_geos(plan, "tv", "north", "south", plan["north"]["tv"])
    left = every["north"]["tv"]
    assert (left >= 0.0).all()
    assert (left <= np.spacing(every["south"]["tv"])).all()
    integers = {"north": {"tv": np.array([7.0])}, "south": {"tv": np.array([64.0])}}
    whole = move_between_geos(integers, "tv", "south", "north", 64.0)
    np.testing.assert_array_equal([whole["north"]["tv"], whole["south"]["tv"]], [[71.0], [0.0]])


def test_a_move_leaves_every_other_cell_and_the_plan_as_they_were() -> None:
    plan = {
        "north": {"tv": np.array([50.0, 60.0]), "search": np.array([5.0, 6.0])},
        "south": {"tv": np.array([10.0, 10.0]), "search": np.array([1.0, 2.0])},
    }
    moved = move_between_geos(plan, "tv", "south", "north", np.array([10.0, 5.0]))
    np.testing.assert_array_equal(moved["north"]["tv"], [60.0, 65.0])
    np.testing.assert_array_equal(moved["south"]["tv"], [0.0, 5.0])
    for g in plan:
        np.testing.assert_array_equal(moved[g]["search"], plan[g]["search"])
    np.testing.assert_array_equal(plan["south"]["tv"], [10.0, 10.0])


@pytest.mark.parametrize(
    ("arguments", "match"),
    [
        (("tv", "north", "south", 100.5), "larger than the source"),
        (("tv", "north", "south", -1.0), "at or above 0"),
        (("tv", "north", "north", 1.0), "two geos"),
        (("radio", "north", "south", 1.0), "no cell"),
        (("tv", "north", "east", 1.0), "no cell"),
        (("tv", "north", "south", [1.0, 2.0, 3.0]), "does not fit"),
    ],
)
def test_what_cannot_be_moved_is_refused(arguments: tuple[Any, ...], match: str) -> None:
    plan = {"north": {"tv": np.array([100.0, 100.0])}, "south": {"tv": np.array([5.0, 5.0])}}
    with pytest.raises(ValueError, match=match):
        move_between_geos(plan, *arguments)


def readings() -> Readings:
    """Three draws of two channels, two of a fit weighted 1/4 and one of a fit weighted 3/4: the
    draws weigh 1/8, 1/8 and 3/4."""
    return Readings(
        channels=(Channel("tv"), Channel("search")),
        retention=np.full((3, 2), 0.5),
        half_saturation=np.ones((3, 2)),
        slope=np.ones((3, 2)),
        coefficient=np.ones((3, 2)),
        tanh=np.zeros(3, dtype=bool),
        fit=np.array([0, 0, 1]),
        chain=np.zeros(3, dtype=np.intp),
        stacking=np.array([0.25, 0.75]),
        pointwise=None,
        length=LENGTH,
    )


def table(seed: int) -> Worths:
    rng = np.random.default_rng(seed)
    rates = np.tile(np.linspace(0.0, 100.0, 5), (2, 1))
    return Worths(rates=rates, table=rng.uniform(0.0, 50.0, (3, 2, 5)))


@pytest.mark.parametrize("level", [0.5, 0.8, 0.9])
def test_a_curve_s_mean_and_band_are_the_draws_at_their_weights(level: float) -> None:
    """Weights 1/8, 1/8 and 3/4 read as the draws repeated 1, 1 and 6 times with equal weights:
    the mean and `numpy.quantile`'s `inverted_cdf` of the repeated draws."""
    worths = table(14)
    curves = response_curves(worths, readings(), level=level)
    repeated = worths.table[[0, 1, 2, 2, 2, 2, 2, 2]]
    np.testing.assert_allclose(curves.mean, repeated.mean(axis=0), rtol=1e-14, atol=0.0)
    ends = ((1.0 - level) / 2.0, (1.0 + level) / 2.0)
    low, high = np.quantile(repeated, ends, axis=0, method="inverted_cdf")
    np.testing.assert_array_equal(curves.low, low)
    np.testing.assert_array_equal(curves.high, high)
    np.testing.assert_array_equal(curves.rates, worths.rates)
    assert curves.current is None
    assert curves.at_current is None


def test_a_curve_marks_the_current_spend_on_its_mean() -> None:
    worths = table(15)
    current = {Channel("tv"): 30.0, Channel("search"): 0.0}
    curves = response_curves(worths, readings(), level=0.9, current=current)
    assert curves.at_current is not None
    np.testing.assert_array_equal(curves.current, [30.0, 0.0])
    tv = curves.mean[0, 1] + (30.0 - 25.0) / 25.0 * (curves.mean[0, 2] - curves.mean[0, 1])
    np.testing.assert_allclose(curves.at_current, [tv, curves.mean[1, 0]], rtol=1e-14, atol=0.0)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"level": 1.0}, "level"),
        ({"level": 0.0}, "level"),
        ({"current": {Channel("tv"): 30.0}}, "lacks"),
        ({"current": {Channel("tv"): 130.0, Channel("search"): 1.0}}, "off its grid"),
    ],
)
def test_what_cannot_be_drawn_as_a_curve_is_refused(change: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        response_curves(
            table(16), readings(), level=change.get("level", 0.9), current=change.get("current")
        )
    with pytest.raises(ValueError, match="readings' draws"):
        response_curves(Worths(np.zeros((2, 5)), np.zeros((4, 2, 5))), readings(), level=0.9)


@pytest.fixture(scope="module")
def fitted() -> tuple[MediaPanel, BayesFit]:
    """A short fit of the harness's world with its trend, season and controls, as
    tests/test_bayes_forecast.py fits its model of the channels alone."""
    panel, _, _, spec = stan_world("normal")
    return panel, fit_bayes(panel, NATIONAL, spec, seed=4, chains=2, warmup=50, draws=20)


AHEAD, DRAWS = 13, 40


def scenario(panel: MediaPanel, fit: BayesFit, name: str, scale: float = 1.0) -> Scenario:
    """The panel's last 13 weeks of spend and controls run on, each channel's spend times
    `scale`."""
    channels = [str(c) for c in panel.channels]
    controls = list(panel.controls)
    spend = {
        m.channel: scale * panel.spend[0, -AHEAD:, channels.index(m.channel)]
        for m in fit.spec.media
    }
    values = {
        c.column: panel.control_values[0, -AHEAD:, controls.index(c.column)]
        for c in fit.spec.controls
    }
    return Scenario(name, spend, values)


def test_a_scenario_s_difference_is_its_media_difference_with_its_carry_out(
    fitted: tuple[MediaPanel, BayesFit],
) -> None:
    """Spending 1.5 times the baseline differs from it by the channels' difference over the 13
    weeks and the kernel's reach after them at each draw `BayesFit.channel_draws` reads, to 1e-12
    of the KPI's mean over the 13 weeks; the trend, season and controls cancel. The baseline
    differs from itself, and from a copy of itself, by exactly 0."""
    panel, fit = fitted
    base = scenario(panel, fit, "base")
    more = scenario(panel, fit, "more", 1.5)
    copy = dataclasses.replace(base, name="copy")
    read = compare_scenarios(
        fit, [base, more, copy], baseline="base", horizon=AHEAD, draws=DRAWS, seed=3
    )
    assert read.names == ("base", "more", "copy")
    assert read.carry_out == fit.length - 1
    np.testing.assert_array_equal(read.differences[[0, 2]], 0.0)
    expected = media_difference(fit.channel_draws(DRAWS), fit.spend, base.spend, more.spend, AHEAD)
    scale = abs(float(read.total_mean[0]))
    np.testing.assert_allclose(read.differences[1], expected, rtol=0.0, atol=1e-12 * scale)
    assert (read.differences[1] != 0.0).all()


def test_under_log_normal_noise_a_difference_is_one_of_means(
    fitted: tuple[MediaPanel, BayesFit],
) -> None:
    """The fit with log-normal noise, each draw's log scale 0.3: its KPI drawn about each mean
    multiplies it by a noise of mean 1, so two scenarios' drawn KPIs differ by their means' gap
    times that noise, and the difference, read from the means, is still the channels' at each draw
    (ADR 0017)."""
    panel, fit = fitted
    noisy = dataclasses.replace(
        fit,
        spec=dataclasses.replace(fit.spec, likelihood=LogNormal()),
        draws=MappingProxyType(dict(fit.draws) | {"sigma": np.full(fit.draws["sigma"].shape, 0.3)}),
    )
    base, more = scenario(panel, fit, "base"), scenario(panel, fit, "more", 1.5)
    read = compare_scenarios(
        noisy, [base, more], baseline="base", horizon=AHEAD, draws=DRAWS, seed=5
    )
    expected = media_difference(fit.channel_draws(DRAWS), fit.spend, base.spend, more.spend, AHEAD)
    scale = abs(float(read.total_mean[0]))
    np.testing.assert_allclose(read.differences[1], expected, rtol=0.0, atol=1e-12 * scale)


def test_under_log_normal_noise_only_the_horizon_s_mean_must_be_above_0(
    fitted: tuple[MediaPanel, BayesFit],
) -> None:
    """The fit with log-normal noise, each draw's log scale 0.3, and every intercept lowered to
    halfway between the baseline's least mean over the 13 weeks and its least over the carry-out,
    which the trend and season run on for `fit.length - 1` periods: the horizon's means are all
    above 0, some of the carry-out's are not. The scenarios compare: the totals are the KPI drawn
    over the horizon alone, and the differences the channels', as at any other intercept."""
    panel, fit = fitted
    base, more = scenario(panel, fit, "base"), scenario(panel, fit, "more", 1.5)
    carry_out = fit.length - 1
    plan = {name: np.r_[series, np.zeros(carry_out)] for name, series in base.spend.items()}
    for name, values in base.controls.items():
        held = np.asarray(values, dtype=np.float64)
        plan[name] = np.r_[held, np.full(carry_out, held[-1])]
    mean = forecast_bayes(fit, AHEAD + carry_out, plan, DRAWS, 5).mean
    inside, after = float(mean[:, :AHEAD].min()), float(mean[:, AHEAD:].min())
    assert after < inside
    shift = -(inside + after) / 2.0
    noisy = dataclasses.replace(
        fit,
        spec=dataclasses.replace(fit.spec, likelihood=LogNormal()),
        draws=MappingProxyType(
            dict(fit.draws)
            | {
                "intercept": fit.draws["intercept"] + shift,
                "sigma": np.full(fit.draws["sigma"].shape, 0.3),
            }
        ),
    )
    with pytest.raises(ValueError, match="not above 0"):
        forecast_bayes(noisy, AHEAD + carry_out, plan, DRAWS, 5)
    read = compare_scenarios(
        noisy, [base, more], baseline="base", horizon=AHEAD, draws=DRAWS, seed=5
    )
    within = {name: series[:AHEAD] for name, series in plan.items()}
    drawn = forecast_bayes(noisy, AHEAD, within, DRAWS, 5).kpi.sum(axis=1)
    np.testing.assert_array_equal(read.totals[0], drawn)
    assert (read.totals > 0.0).all()
    expected = media_difference(fit.channel_draws(DRAWS), fit.spend, base.spend, more.spend, AHEAD)
    scale = abs(float(read.total_mean[0]))
    np.testing.assert_allclose(read.differences[1], expected, rtol=0.0, atol=1e-12 * scale)
    np.testing.assert_array_equal(read.differences[0], 0.0)


def test_a_scenario_s_controls_count_over_the_horizon_and_cancel_after_it(
    fitted: tuple[MediaPanel, BayesFit],
) -> None:
    """Raising the first control by 1 over the 13 weeks, the spend unchanged, moves the
    difference by its coefficient times 13 at each draw: past the horizon both hold the
    baseline's."""
    panel, fit = fitted
    base = scenario(panel, fit, "base")
    column = fit.spec.controls[0].column
    raised = Scenario("raised", base.spend, {column: np.asarray(base.controls[column]) + 1.0})
    read = compare_scenarios(
        fit, [base, raised], baseline="base", horizon=AHEAD, draws=DRAWS, seed=3
    )
    keep = np.linspace(0, fit.draws["control"].shape[0] * fit.draws["control"].shape[1] - 1, DRAWS)
    gamma = fit.draws["control"].reshape(-1, len(fit.spec.controls))[keep.round().astype(int), 0]
    scale = abs(float(read.total_mean[0]))
    np.testing.assert_allclose(read.differences[1], AHEAD * gamma, rtol=0.0, atol=1e-12 * scale)


def test_every_scenario_reads_the_same_draws_and_the_same_noise(
    fitted: tuple[MediaPanel, BayesFit],
) -> None:
    """A scenario's total is its forecast's KPI over the 13 weeks, each run over the 13 and the
    reach after them at one seed, so each draw's noise is the same in every scenario: a total less
    its mean is the same in both. Its interval is the totals' central 80 %."""
    panel, fit = fitted
    base, more = scenario(panel, fit, "base"), scenario(panel, fit, "more", 1.5)
    read = compare_scenarios(
        fit, [more, base], baseline="base", horizon=AHEAD, draws=DRAWS, seed=7, level=0.8
    )
    tail = fit.length - 1
    means = []
    for i, s in enumerate((more, base)):
        planned = {c: np.concatenate([v, np.zeros(tail)]) for c, v in s.spend.items()}
        planned |= {
            c: np.concatenate([v, np.full(tail, np.asarray(base.controls[c])[-1])])
            for c, v in s.controls.items()
        }
        ahead = forecast_bayes(fit, AHEAD + tail, planned, DRAWS, seed=7)
        np.testing.assert_array_equal(read.totals[i], ahead.kpi[:, :AHEAD].sum(axis=1))
        means.append(ahead.mean[:, :AHEAD].sum(axis=1))
    np.testing.assert_allclose(
        read.totals[0] - means[0], read.totals[1] - means[1], rtol=0.0, atol=1e-9
    )
    np.testing.assert_array_equal(read.differences[1], 0.0)
    # the ends as the reader forms them: (1 - 0.8) / 2 is 0.09999999999999998, not 0.1
    low, high = np.quantile(read.totals, ((1.0 - 0.8) / 2.0, (1.0 + 0.8) / 2.0), axis=1)
    np.testing.assert_array_equal(read.total_low, low)
    np.testing.assert_array_equal(read.total_high, high)
    np.testing.assert_array_equal(read.total_mean, read.totals.mean(axis=1))


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda s: s.update(baseline="none"), "baseline"),
        (lambda s: s.update(scenarios=[s["scenarios"][0]] * 2), "distinct"),
        (lambda s: s.update(level=1.5), "level"),
        (lambda s: s.update(horizon=0), "horizon"),
    ],
)
def test_what_cannot_be_compared_is_refused(
    fitted: tuple[MediaPanel, BayesFit],
    change: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    panel, fit = fitted
    arguments: dict[str, Any] = {
        "scenarios": [scenario(panel, fit, "base")],
        "baseline": "base",
        "horizon": AHEAD,
        "draws": 4,
        "seed": 1,
    }
    change(arguments)
    with pytest.raises(ValueError, match=match):
        compare_scenarios(fit, **arguments)
    no_controls = Scenario("base", scenario(panel, fit, "base").spend)
    with pytest.raises(ValueError, match="every control"):
        compare_scenarios(fit, [no_controls], baseline="base", horizon=AHEAD, draws=4, seed=1)
