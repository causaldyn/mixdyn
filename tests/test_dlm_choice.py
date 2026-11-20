"""The discount choice against the likelihood it reads, and its intervals by hand.

The world is `seasonal_world`'s history, 156 weeks of `y = 100 + s + 0.5 x + N(0, 1)`. Coverage
itself is CHC's `scripts/bench_dlm.py`'s to measure, over hundreds of series; one series can only
show the construction.
"""

from __future__ import annotations

import functools
import itertools
import logging

import numpy as np
import pytest
from chc.dlm import confidence_set

from mixdyn import NATIONAL, Cadence, Count, MediaPanel, read_panel
from mixdyn.models import (
    DiscountGrid,
    DlmChoice,
    DlmSpec,
    Edge,
    Effect,
    PerEffect,
    Season,
    Trend,
    choose_dlm,
    decompose_dlm,
)
from seasonal_world import DATES, HISTORY, history

SPEC = DlmSpec((Effect("x", 0.95),), trend=Trend(1, 0.95), season=Season(52.0, 2, 0.98))
# The world is static, so the likelihood ranks coarse discounts far apart; near 1 it keeps two.
GRID = DiscountGrid(trend=(0.98, 0.99, 1.0), effects=(0.98, 0.99, 1.0))


@functools.cache
def choice() -> DlmChoice:
    return choose_dlm(history(), NATIONAL, SPEC, GRID)


def drifting_world() -> tuple[MediaPanel, DlmSpec]:
    """156 weeks of `y = 100 + b_t x + 0.3 v + 0.2 w + N(0, 1)`, each regressor `max(0, N(50,
    15))` and `log b_t` a random walk of 0.05 a week from `log 0.5`: one effect that drifts beside
    two that do not, and a spec of a level and the three."""
    rng = np.random.default_rng(20261004)
    x, v, w = np.maximum(0.0, rng.normal(50.0, 15.0, (3, HISTORY)))
    drift = 0.5 * np.exp(np.cumsum(0.02 * rng.standard_normal(HISTORY)))
    kpi = 100.0 + drift * x + 0.3 * v + 0.2 * w + 10.0 * rng.standard_normal(HISTORY)
    panel = read_panel(
        {"date": DATES[:HISTORY], "kpi": kpi, "x": x, "v": v, "w": w},
        date="date",
        kpi="kpi",
        spend={c: c for c in ("x", "v", "w")},
        cadence=Cadence.WEEK,
        kpi_unit=Count("units"),
        currency="USD",
    )
    effects = tuple(Effect(c, 1.0) for c in ("x", "v", "w"))
    return panel, DlmSpec(effects, trend=Trend(1, 1.0))


TWO = DlmSpec(
    (Effect("x", 0.95), Effect("y", 0.9)), trend=Trend(1, 0.95), season=Season(52.0, 2, 0.98)
)


def test_a_grid_varies_its_families_and_counts_them_as_parameters() -> None:
    trends, effects = (0.9, 0.95), (0.97, 0.98, 0.99)
    grid = DiscountGrid(trend=trends, effects=effects)
    specs = grid.specs(TWO)
    expected = [(t, e, e) for t, e in itertools.product(trends, effects)]
    got = [
        (s.trend.discount, s.effects[0].discount, s.effects[1].discount) for s in specs if s.trend
    ]
    assert got == expected
    assert {s.season for s in specs} == {TWO.season}
    assert grid.parameters(TWO) == 2
    assert DiscountGrid(season=(0.95, 0.99)).parameters(TWO) == 1
    assert grid.blocks(TWO) == (specs,)


def test_a_grid_per_effect_tries_every_combination_and_counts_each_effect() -> None:
    """Each effect's discount on its own, the last effect's varying fastest, one parameter an
    effect; a search's blocks are every pair of the discounts the grid varies, the rest held, or
    the one alone."""
    trends, values = (0.9, 0.95), (0.97, 0.99)
    grid = DiscountGrid(trend=trends, effects=PerEffect(values))
    specs = grid.specs(TWO)
    got = [
        (s.trend.discount, s.effects[0].discount, s.effects[1].discount) for s in specs if s.trend
    ]
    assert got == list(itertools.product(trends, values, values))
    assert {s.season for s in specs} == {TWO.season}
    assert grid.parameters(TWO) == 3
    three = DlmSpec((*TWO.effects, Effect("z", 1.0)), trend=TWO.trend)
    assert grid.parameters(three) == 4
    assert len(grid.specs(three)) == 2 * 2**3
    trend_x, trend_y, x_y = grid.blocks(TWO)
    pairs = list(itertools.product(trends, values))
    assert [(s.trend.discount, s.effects[0].discount) for s in trend_x if s.trend] == pairs
    assert {s.effects[1] for s in trend_x} == {TWO.effects[1]}
    assert [(s.trend.discount, s.effects[1].discount) for s in trend_y if s.trend] == pairs
    assert {s.effects[0] for s in trend_y} == {TWO.effects[0]}
    together = [(s.effects[0].discount, s.effects[1].discount) for s in x_y]
    assert together == list(itertools.product(values, values))
    assert {s.trend for s in x_y} == {TWO.trend}
    alone = DiscountGrid(effects=PerEffect(values))
    (both,) = alone.blocks(TWO)
    assert set(both) == set(alone.specs(TWO))
    one = DlmSpec((Effect("x", 0.95),))
    assert alone.blocks(one) == (alone.specs(one),)


def test_an_effect_the_grid_holds_keeps_its_discount_and_is_no_parameter() -> None:
    """A held effect, as a recurring event's static coefficient (ADR 0085), keeps the spec's own
    discount in every spec and block of the grid, counts no parameter and names no end."""
    trends, values = (0.9, 0.95), (0.97, 0.99)
    grid = DiscountGrid(trend=trends, effects=PerEffect(values, held=("y",)))
    specs = grid.specs(TWO)
    assert {s.effects[1] for s in specs} == {TWO.effects[1]}
    got = [(s.trend.discount, s.effects[0].discount) for s in specs if s.trend]
    assert got == list(itertools.product(trends, values))
    assert grid.parameters(TWO) == 2
    (trend_x,) = grid.blocks(TWO)
    assert set(trend_x) == set(specs)
    mixed = DlmSpec((Effect("x", 0.97), Effect("y", 0.97)))
    each = DiscountGrid(effects=PerEffect((0.97, 0.99, 1.0), held=("y",)))
    assert each.edges([mixed], mixed) == (Edge("x discount", 0.97, best=True),)
    assert each.specs(mixed) == tuple(
        DlmSpec((Effect("x", d), Effect("y", 0.97))) for d in (0.97, 0.99, 1.0)
    )


def test_each_effect_s_discount_is_chosen_on_its_own() -> None:
    """On a world of three effects, one of them drifting: each effect's discount chosen on its
    own puts the drifting one's below the others', where one discount shared by all sits between
    them; the set counts a parameter an effect."""
    panel, spec = drifting_world()
    values = (0.9, 0.95, 0.97, 0.99, 1.0)
    each = choose_dlm(panel, NATIONAL, spec, DiscountGrid(effects=PerEffect(values)))
    shared = choose_dlm(panel, NATIONAL, spec, DiscountGrid(effects=values))
    assert each.specs == DiscountGrid(effects=PerEffect(values)).specs(spec)
    loglik = [f.log_likelihood for f in each.filtered]
    assert each.best == int(np.argmax(loglik))
    assert tuple(each.kept) == confidence_set(list(each.filtered), 3)
    drifting, *static = (e.discount for e in each.fit.spec.effects)
    assert drifting < shared.fit.spec.effects[0].discount <= min(static)


def test_an_end_the_kept_specs_reach_is_named_where_the_discount_goes_on_past_it() -> None:
    """A family's lowest discount is an end, and its highest where it is below 1, the discount's
    own bound; a family of one discount is set, not chosen, and has none. `best` says whether the
    likelihood's best spec is there."""
    grid = DiscountGrid(trend=(0.95, 0.99, 1.0), season=(0.98,), effects=(0.97, 0.99, 1.0))
    at = {
        (t, e): DlmSpec(
            (Effect("x", e), Effect("y", e)), trend=Trend(1, t), season=Season(52.0, 2, 0.98)
        )
        for t, e in itertools.product((0.95, 0.99), (0.97, 0.99, 1.0))
    }
    assert grid.edges([at[0.99, 0.99], at[0.99, 1.0]], at[0.99, 1.0]) == ()
    assert grid.edges([at[0.99, 1.0], at[0.95, 0.97]], at[0.99, 1.0]) == (
        Edge("trend discount", 0.95, best=False),
        Edge("effects discount", 0.97, best=False),
    )
    assert grid.edges([at[0.95, 0.99]], at[0.95, 0.99]) == (
        Edge("trend discount", 0.95, best=True),
    )
    below = DiscountGrid(effects=(0.9, 0.95))
    assert below.edges([at[0.99, 0.97]], at[0.99, 0.97]) == ()
    high = DlmSpec((Effect("x", 0.95), Effect("y", 0.95)))
    assert below.edges([high], high) == (Edge("effects discount", 0.95, best=True),)
    each = DiscountGrid(effects=PerEffect((0.97, 0.99, 1.0)))
    mixed = DlmSpec((Effect("x", 0.97), Effect("y", 1.0)))
    assert each.edges([mixed], mixed) == (Edge("x discount", 0.97, best=True),)


def test_a_choice_names_and_logs_each_end_of_its_grid_its_set_reaches(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The static world keeps discounts of 0.999 and 1 when the grid holds only those, so its set
    reaches the grid's floor and may stop there; with the grid's floor below the set, no end is
    named and nothing is logged."""
    with caplog.at_level(logging.WARNING, logger="mixdyn.models.dlm_choice"):
        narrow = choose_dlm(history(), NATIONAL, SPEC, DiscountGrid(effects=(0.999, 1.0)))
    assert narrow.edges == (Edge("effects discount", 0.999, best=False),)
    (record,) = [r for r in caplog.records if "edges" in r.__dict__]
    assert record.__dict__["edges"] == [("effects discount", 0.999, False)]
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="mixdyn.models.dlm_choice"):
        wide = choice()
    assert wide.edges == ()
    assert not [r for r in caplog.records if "edges" in r.__dict__]


def test_the_choice_is_the_likelihood_s_best_and_its_set_chc_s() -> None:
    chosen = choice()
    assert chosen.specs == GRID.specs(SPEC)
    loglik = [f.log_likelihood for f in chosen.filtered]
    assert chosen.best == int(np.argmax(loglik))
    assert tuple(chosen.kept) == confidence_set(list(chosen.filtered), 2)
    assert chosen.fit is chosen.kept[chosen.best]
    assert 1 < len(chosen.kept) < len(chosen.specs)


def test_only_the_kept_specs_are_fitted_and_a_signed_one_s_posterior_drawn() -> None:
    """A rejected spec is filtered for its likelihood and nothing more; a kept one is `fit_dlm`'s
    fit, on the filter the choice ranked, and with a signed effect it holds its draws."""
    signed = DlmSpec(
        (Effect("x", 0.95, sign="positive"),), trend=Trend(1, 0.95), season=Season(52.0, 2, 0.98)
    )
    chosen = choose_dlm(history(), NATIONAL, signed, GRID)
    assert len(chosen.filtered) == len(chosen.specs) == 9
    assert 1 < len(chosen.kept) < 9
    for k, fit in chosen.kept.items():
        assert fit.spec == chosen.specs[k]
        assert fit.filtered.log_likelihood == chosen.filtered[k].log_likelihood
        assert fit.signed is not None


def test_an_interval_is_the_union_of_the_kept_fits_intervals() -> None:
    chosen = choice()
    dates = chosen.fit.dates
    start, end = dates[-13], dates[-1]
    bounds = chosen.bounds(lambda d: d.window(start, end), draws=500, seed=7)
    lows, highs = [], []
    for k in chosen.kept:
        window = decompose_dlm(chosen.kept[k], draws=500, seed=7 + k).window(start, end)
        lo, hi = np.quantile(window.contributions["x"], [0.05, 0.95])
        lows.append(lo)
        highs.append(hi)
    assert bounds.contributions["x"].lower == pytest.approx(min(lows), rel=1e-14, abs=0.0)
    assert bounds.contributions["x"].upper == pytest.approx(max(highs), rel=1e-14, abs=0.0)
    assert len(lows) > 1
    assert bounds.base is not None
    assert bounds.seasonality is not None
    best = decompose_dlm(chosen.fit, draws=500, seed=7 + chosen.best).window(start, end)
    assert best.base is not None
    lo, hi = np.quantile(best.base, [0.05, 0.95])
    assert bounds.base.lower <= lo
    assert hi <= bounds.base.upper


def test_a_spec_without_a_season_has_no_season_bounds() -> None:
    spec = DlmSpec((Effect("x", 0.95),), trend=Trend(1, 0.95))
    chosen = choose_dlm(history(), NATIONAL, spec, DiscountGrid(effects=(0.95, 1.0)))
    dates = chosen.fit.dates
    bounds = chosen.bounds(lambda d: d.window(dates[0], dates[9]), draws=200, seed=1)
    assert bounds.seasonality is None
    assert bounds.base is not None


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda: DiscountGrid(), "at least one family"),
        (lambda: DiscountGrid(trend=(0.9, 0.9)), "names a trend discount twice"),
        (lambda: PerEffect(()), "must name at least one, none twice"),
        (lambda: PerEffect((0.9, 0.9)), "must name at least one, none twice"),
        (
            lambda: DiscountGrid(effects=PerEffect((0.9,))).specs(DlmSpec((), trend=Trend(1, 1.0))),
            "the spec has no effect",
        ),
        (
            lambda: DiscountGrid(effects=PerEffect((0.9,))).blocks(
                DlmSpec((), trend=Trend(1, 1.0))
            ),
            "the spec has no effect",
        ),
        (
            lambda: DiscountGrid(season=(0.95,)).specs(DlmSpec((Effect("x", 0.95),))),
            "the spec has no season",
        ),
        (
            lambda: DiscountGrid(trend=(0.95,)).specs(DlmSpec((Effect("x", 0.95),))),
            "the spec has no trend",
        ),
        (
            lambda: choice().bounds(
                lambda d: d.window(d.dates[0], d.dates[1]), draws=10, seed=1, level=1.0
            ),
            r"level must lie in \(0, 1\)",
        ),
    ],
)
def test_what_is_not_a_choice_is_refused(make, message) -> None:
    with pytest.raises(ValueError, match=message):
        make()
