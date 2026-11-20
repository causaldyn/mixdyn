"""The flat-spend world: a channel whose spend never moves, read by every engine.

104 weeks of one geo. Television and search ads spend as ADR 0014's test world draws them;
retail media spends `FLAT` every week, as it did before the panel began. Each channel's response is
a `tanh` of its adstock; retail's tops out at 200 a week and returns 100 at its spend, carried
over at 0.3. The noise is 25 a week.

Nothing in the history moves retail's spend, so no fit can read its return: the honest answer is
a wide interval, and a plan made without a test should leave retail where it is. What each part of
the engine does with it:

- `read_panel` flags it, `IssueCode.CONSTANT` on its column: its effect cannot be identified;
- the ridge, a point, holds its coefficient at its bound, 0, and states no interval;
- the DLM's interval of its coefficient holds 0 and the world's and is more than five times as wide
  as either other channel's;
- the Bayesian model's 90 % interval of its return keeps most of the prior's width, and the health
  card's `width` reads it;
- the plan made without a test, for the posterior's worst tenth against the plan in place (ADR
  0044), leaves retail's spend where it is, where the plan for the posterior's mean halves it and
  names it as a channel it moves on its prior alone; so do the plan for a worse share that moves
  it, the flighted plan at the posterior's median and the joint plan of the geos;
- the choice of tests (ADR 0031) is to test, and to test retail.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from chc.response import Channel as Response
from chc.response import GeometricAdstock, Tanh

from mixdyn import (
    NATIONAL,
    Cadence,
    Channel,
    IssueCode,
    MediaPanel,
    Revenue,
    Severity,
    read_panel,
)
from mixdyn.decisions import (
    CvarPlan,
    GeoLimits,
    GeoPlan,
    Range,
    plan_cvar,
    plan_flighting,
    plan_geos,
    plan_posterior,
    plan_robust,
    readings,
)
from mixdyn.experiments import After, GroupNoise, choose_tests, designs
from mixdyn.inference.diagnostics import health
from mixdyn.models import DiscountGrid, DlmSpec, Effect, Season, Trend, choose_dlm
from mixdyn.models.bayes_mmm import BayesFit, BayesMedia, BayesSpec, RoiPrior, TanhCurve, fit_bayes
from mixdyn.models.ridge_mmm import FourierSeason, Media, RidgeSpec, fit_ridge
from mixdyn.transforms import Geometric, Hill, TanhShape, Transform

WEEKS = 104
FLAT = 80.0
"""Retail's spend every week, before the panel too."""
CHANNELS = (Channel("tv"), Channel("search"), Channel("retail"))
TRUTH = {"tv": (300.0, 0.5, 100.0), "search": (150.0, 0.1, 60.0), "retail": (200.0, 0.3, 80.0)}
"""Each channel's ceiling a week, retention and half-saturation point in euros a week."""
NOISE = 25.0


def _response(spend: np.ndarray, ceiling: float, retention: float, half: float) -> np.ndarray:
    """A channel's response each week: its spend carried over by a normalised geometric kernel,
    from as far back as it spent, through `ceiling tanh(atanh(1/2) a / half)`."""
    before = np.full(400, spend[0]) if np.all(spend == spend[0]) else np.zeros(400)
    weights = retention ** np.arange(400)
    adstock = np.convolve(np.r_[before, spend], weights / weights.sum())[400 : 400 + spend.size]
    return ceiling * np.tanh(np.arctanh(0.5) * adstock / half)


def flat_world(seed: int = 20261106, retail: str = "retail") -> MediaPanel:
    """The world's history, read as a user reads one, retail's spend from the column `retail`."""
    rng = np.random.default_rng(seed)
    t = np.arange(WEEKS)
    spend = {
        "tv": rng.gamma(2.0, 50.0, WEEKS) * (rng.random(WEEKS) > 0.3),
        "search": rng.gamma(2.0, 30.0, WEEKS) * (rng.random(WEEKS) > 0.2),
        "retail": np.full(WEEKS, FLAT),
    }
    sales = 1500.0 + 40.0 * np.sin(2.0 * np.pi * t / 52.18) + NOISE * rng.standard_normal(WEEKS)
    for name, x in spend.items():
        sales += _response(x, *TRUTH[name])
    columns = {c: retail if c == "retail" else c for c in spend}
    return read_panel(
        {"date": np.datetime64("2023-01-02") + np.timedelta64(7, "D") * t, "sales": sales}
        | {columns[c]: x for c, x in spend.items()},
        date="date",
        kpi="sales",
        spend=columns,
        cadence=Cadence.WEEK,
        kpi_unit=Revenue(),
        currency="EUR",
    )


def test_reading_the_panel_names_the_flat_channel() -> None:
    panel = flat_world()
    (flag,) = [i for i in panel.issues if i.code is IssueCode.CONSTANT]
    assert flag.column == "retail"
    assert flag.severity is Severity.WARNING
    assert flag.message == (
        "channel 'retail' spends 80 in every period: its effect cannot be identified"
    )


def test_the_ridge_holds_the_flat_channel_at_its_bound() -> None:
    """The ridge's adstock starts from nothing before the panel, so a flat channel's column rises
    over the first weeks and is flat after: its one contrast is the panel's start, where sales,
    which the channel fed before the panel too, do not rise. The fit reads that as no effect and
    holds the coefficient at its bound, 0, where the world's channel returns 100 a week.
    A ridge fit is a point: it states no interval, and its `bound` is the only sign."""
    names = [str(c) for c in CHANNELS]
    spec = RidgeSpec(
        tuple(Media(c, Geometric(0.3), Hill(1.0, 0.5)) for c in names),
        (),
        penalty=1e-3,
        season=FourierSeason(52.18, 1),
    )
    fit = fit_ridge(flat_world(), NATIONAL, spec)
    assert fit.bound["retail"]
    assert fit.coefficients["retail"] == 0.0
    assert not fit.bound["tv"]
    assert not fit.bound["search"]


@pytest.mark.parametrize("retention", [0.0, 0.3])
def test_the_dlms_interval_of_the_flat_channel_holds_0_and_the_truth(retention: float) -> None:
    """Each channel through the world's own curve, carried over at `retention`, its coefficient
    fixed over the weeks, beside a level and a season: retail's 90 % interval at the last week
    holds 0 and the world's 200 and is more than five times as wide as either other channel's;
    a column flat over the weeks is the level's, and only the first weeks' rise from nothing
    before the panel, at a retention above 0, tells them apart."""
    names = [str(c) for c in CHANNELS]
    effects = tuple(
        Effect(c, 1.0, transform=Transform(retention, TRUTH[c][2], TanhShape(), WEEKS))
        for c in names
    )
    spec = DlmSpec(effects, trend=Trend(1, 1.0), season=Season(52.18, 1, 1.0))
    bounds = choose_dlm(flat_world(), NATIONAL, spec, DiscountGrid(trend=(0.99, 1.0)))
    interval = bounds.coefficient_bounds(0.9)
    retail = interval["retail"]
    assert retail.lower < 0.0 < TRUTH["retail"][0] < retail.upper
    width = retail.upper - retail.lower
    for other in ("tv", "search"):
        assert width > 5.0 * (interval[other].upper - interval[other].lower)


@pytest.fixture(scope="module")
def bayes() -> BayesFit:
    """The Bayesian model with `tanh` curves on the world, two chains of 300 warm-up iterations and
    200 draws from Pathfinder's starts, the settings of the nested model's tests."""
    spec = BayesSpec(
        tuple(BayesMedia(str(c)) for c in CHANNELS),
        season=FourierSeason(52.18, 1),
        saturation=TanhCurve(),
    )
    return fit_bayes(flat_world(), NATIONAL, spec, seed=3, chains=2, warmup=300, draws=200)


def test_the_bayesian_models_interval_of_the_flat_channel_keeps_most_of_the_priors_width(
    bayes: BayesFit,
) -> None:
    """Retail's 90 % interval of its return spans a ratio of 15 between its ends at this seed, 0.92
    of the prior's on the log scale, against 1.3 and 1.5 for television and search; the health
    card's `width`, the posterior's standard deviation over the prior's, is 0.45 for retail and
    below 0.09 for the others. The bounds hold each with a margin of at least 1.5."""
    roi = bayes.draws["roi"].reshape(-1, len(CHANNELS))
    low, high = np.quantile(roi, [0.05, 0.95], axis=0)
    spread = dict(zip([str(c) for c in CHANNELS], np.log(high / low), strict=True))
    prior = RoiPrior()
    assert spread["retail"] > 0.6 * 2.0 * 1.6449 * prior.sigma
    card = health(bayes, flat_world(), NATIONAL, seed=1)
    for other in ("tv", "search"):
        assert spread["retail"] > 4.0 * spread[other]
        assert card.width["retail"] > 3.0 * card.width[other]
    assert card.width["retail"] > 0.3


def test_the_plan_made_without_a_test_holds_the_flat_channel(bayes: BayesFit) -> None:
    """The plan of the posterior's worst tenth's gain over the plan in place, the level mixdyn
    plans at without a lift test (ADR 0044), spends retail's 80 a week, where the plan of the
    posterior's mean, the same plan at a level of 1, cuts it to the floor of its range: the
    clause, not the fit, holds it. As the level rises the plan moves television before retail: at
    the first of the levels 0.15, 0.2, 0.25, 0.3, 0.4 and 0.5 at which it moves television by more
    than a twentieth of its spend, it moves retail by less than a twentieth of its.

    Measured on the fixture's fit at seeds 0 to 9, and at seed 3 with XLA's code held to AVX and
    to SSE4.2, that level was 0.2 to 0.3 and retail's move there at most 0.013 of its spend; at
    0.1 no fit moved any channel, the earliest move coming at 0.15. Television's move at 0.25
    alone ran from 0.006 to 0.21 of its spend, and was 0 on the GitHub runners of CI's 3.12 to
    3.14 legs: no one level moves it at every posterior a seed or a machine's last bits draw."""
    panel = flat_world()
    planner = readings(bayes)
    current = {c: float(panel.spend[0, -13:, j].mean()) for j, c in enumerate(planner.channels)}
    limits = {c: Range(0.5 * v, 2.0 * v) for c, v in current.items()}
    against = np.array(list(current.values()))
    budget = 13 * float(against.sum())
    retail = planner.channels.index(Channel("retail"))
    tv = planner.channels.index(Channel("tv"))

    def plan(level: float) -> np.ndarray:
        found = plan_cvar(
            panel, NATIONAL, planner, budget, 13, limits, against=against, level=level
        )
        return found.spend

    assert plan(0.1)[retail] == pytest.approx(FLAT, rel=1e-6, abs=0.0)
    assert plan(1.0)[retail] == pytest.approx(0.5 * FLAT, rel=1e-6, abs=0.0)
    first = next(
        spend
        for spend in map(plan, (0.15, 0.2, 0.25, 0.3, 0.4, 0.5))
        if abs(spend[tv] - against[tv]) > 0.05 * against[tv]
    )
    assert abs(first[retail] - FLAT) < 0.05 * FLAT


def test_a_flat_channel_read_from_a_column_of_another_name_is_named(bayes: BayesFit) -> None:
    """Retail read from the column `retail_eur`: the panel's flag names the channel beside its
    column, and the plan of the posterior's mean and the plan for the least gain at 1.1, each of
    which halves retail's spend, name retail. A flag that names the column `retail` and no channel,
    as a constant control of that name raises, names nothing."""
    panel = flat_world(retail="retail_eur")
    (flag,) = [i for i in panel.issues if i.code is IssueCode.CONSTANT]
    assert (flag.column, flag.channel) == ("retail_eur", "retail")
    planner = readings(bayes)
    current = {c: float(panel.spend[0, -13:, j].mean()) for j, c in enumerate(planner.channels)}
    limits = {c: Range(0.5 * v, 2.0 * v) for c, v in current.items()}
    budget = 13 * sum(current.values())
    plan = plan_posterior(panel, NATIONAL, planner, budget, 13, limits)
    assert plan.unidentified == (Channel("retail"),)
    robust = plan_robust(panel, NATIONAL, planner, budget, 13, limits, scale=1.1)
    assert robust.unidentified == (Channel("retail"),)
    control = dataclasses.replace(flag, column="retail", channel=None)
    column_alone = dataclasses.replace(panel, issues=(control,))
    assert plan_posterior(column_alone, NATIONAL, planner, budget, 13, limits).unidentified == ()


def test_the_plan_of_the_posterior_s_mean_names_the_flat_channel_it_moves(bayes: BayesFit) -> None:
    """The plan of the posterior's mean halves retail's spend, which the panel flags as never
    moving: the plan names retail, and only it, in `unidentified`, and every number it holds is
    the plan's on the same panel without the flag, to the bit. Held at 80 by a range that is a
    point, retail is not named. The plan for the least gain with each effect off by up to a
    factor cuts retail by more than a quarter and names it at 1.1, and holds it at 80 and names
    nothing at 2.

    Measured on the fixture's fit at seeds 0 to 9, and at seed 3 with XLA's code held to AVX and
    to SSE4.2, the mean's plan put retail at 40 every time; at 1.1 the robust plan put it at 40 to
    45.4 (44.6 and 44.8 on two of CI's runner classes), and it kept the plan in place from a
    factor of 1.4 or 1.5, at seed 6 from one between 1.5 and 2, retail at 78.6 at 1.5."""
    panel = flat_world()
    planner = readings(bayes)
    current = {c: float(panel.spend[0, -13:, j].mean()) for j, c in enumerate(planner.channels)}
    limits = {c: Range(0.5 * v, 2.0 * v) for c, v in current.items()}
    budget = 13 * sum(current.values())
    retail = planner.channels.index(Channel("retail"))
    plan = plan_posterior(panel, NATIONAL, planner, budget, 13, limits)
    assert plan.spend[retail] == pytest.approx(0.5 * FLAT, rel=1e-6, abs=0.0)
    assert plan.unidentified == (Channel("retail"),)
    unflagged = plan_posterior(
        dataclasses.replace(panel, issues=()), NATIONAL, planner, budget, 13, limits
    )
    assert unflagged.unidentified == ()
    for field in ("spend", "current", "budget", "gain", "current_gain", "grid"):
        np.testing.assert_array_equal(getattr(plan, field), getattr(unflagged, field))
    held = {**limits, Channel("retail"): Range(FLAT, FLAT)}
    kept = plan_posterior(panel, NATIONAL, planner, budget, 13, held)
    assert kept.spend[retail] == FLAT
    assert kept.unidentified == ()
    robust = plan_robust(panel, NATIONAL, planner, budget, 13, limits, scale=1.1)
    assert robust.spend[retail] < 0.75 * FLAT
    assert robust.unidentified == (Channel("retail"),)
    robust = plan_robust(panel, NATIONAL, planner, budget, 13, limits, scale=2.0)
    assert robust.spend[retail] == pytest.approx(FLAT, rel=1e-9, abs=0.0)
    assert robust.unidentified == ()
    robust = plan_robust(panel, NATIONAL, planner, budget, 13, held, scale=1.1)
    assert robust.unidentified == ()


@pytest.mark.parametrize("column", ["retail", "retail_eur"])
def test_the_plan_for_the_worst_share_names_the_flat_channel_where_it_moves_it(
    bayes: BayesFit, column: str
) -> None:
    """At the level mixdyn plans at without a lift test, 0.1, the plan keeps retail's 80 to the bit
    and names nothing. At the first of the levels 0.25, 0.3, 0.4, 0.5 and 1 at which it moves
    retail by more than a thousandth of its spend, it names it, and only it; every number it holds
    is the plan's on the same panel without the flag, to the bit. Held at 80 by a range that is a
    point, retail is not named. Retail's spend read from a column of another name reads alike.

    Measured on the fixture's fit at seeds 0 to 9, no plan at 0.1 moved retail; at 0.25 the plan
    moved it by 0.0004 to 0.039 of its spend, and by none on the AMD EPYC 9V45 runner of CI's 3.12
    leg; at 1 it cut retail to 40 every time. Every move, down to 0.0004, was named."""
    panel = flat_world(retail=column)
    planner = readings(bayes)
    current = {c: float(panel.spend[0, -13:, j].mean()) for j, c in enumerate(planner.channels)}
    limits = {c: Range(0.5 * v, 2.0 * v) for c, v in current.items()}
    against = np.array(list(current.values()))
    budget = 13 * float(against.sum())
    retail = planner.channels.index(Channel("retail"))

    def plan(level: float, flagged: MediaPanel = panel, ranges: dict = limits) -> CvarPlan:
        return plan_cvar(
            flagged, NATIONAL, planner, budget, 13, ranges, against=against, level=level
        )

    kept = plan(0.1)
    assert kept.spend[retail] == FLAT
    assert kept.unidentified == ()
    level, moved = next(
        (level, found)
        for level in (0.25, 0.3, 0.4, 0.5, 1.0)
        if FLAT - (found := plan(level)).spend[retail] > 0.001 * FLAT
    )
    assert moved.unidentified == (Channel("retail"),)
    unflagged = plan(level, dataclasses.replace(panel, issues=()))
    assert unflagged.unidentified == ()
    for field in ("spend", "against", "budget", "level", "gain", "cvar", "bound"):
        np.testing.assert_array_equal(getattr(moved, field), getattr(unflagged, field))
    held = plan(level, ranges={**limits, Channel("retail"): Range(FLAT, FLAT)})
    assert held.spend[retail] == FLAT
    assert held.unidentified == ()


@pytest.mark.parametrize("column", ["retail", "retail_eur"])
def test_the_flighted_and_the_geos_plans_name_the_flat_channel_they_move(
    bayes: BayesFit, column: str
) -> None:
    """The flighted plan at the posterior's median cuts retail to 40 in every week, in its schedule
    and its rate a week, and names it; on the panel without the flag it names nothing and holds the
    same numbers, to the bit; held at 80, retail is not named. The joint plan of the geos, on the
    world's own curves, cuts retail to 40 and names its one cell; held at 80, none. Retail's spend
    read from a column of another name reads alike."""
    panel = flat_world(retail=column)
    planner = readings(bayes)
    current = {c: float(panel.spend[0, -13:, j].mean()) for j, c in enumerate(planner.channels)}
    limits = {c: Range(0.5 * v, 2.0 * v) for c, v in current.items()}
    held = {**limits, Channel("retail"): Range(FLAT, FLAT)}
    budget = 13 * sum(current.values())
    retail = planner.channels.index(Channel("retail"))
    flighted = plan_flighting(panel, NATIONAL, planner, budget, 13, limits)
    np.testing.assert_array_equal(flighted.schedule[:, retail], 0.5 * FLAT)
    assert flighted.continuous[retail] == 0.5 * FLAT
    assert flighted.unidentified == (Channel("retail"),)
    unflagged = plan_flighting(
        dataclasses.replace(panel, issues=()), NATIONAL, planner, budget, 13, limits
    )
    assert unflagged.unidentified == ()
    for field in ("schedule", "continuous", "current", "gains", "edge", "worth", "bound"):
        np.testing.assert_array_equal(getattr(flighted, field), getattr(unflagged, field))
    assert plan_flighting(panel, NATIONAL, planner, budget, 13, held).unidentified == ()

    responses = {
        NATIONAL: {
            Channel(name): Response(
                GeometricAdstock(retention, length=12, normalized=True),
                Tanh(half / math.atanh(0.5)),
                ceiling,
            )
            for name, (ceiling, retention, half) in TRUTH.items()
        }
    }

    def geos(ranges: dict) -> GeoPlan:
        cells = {(NATIONAL, c): r for c, r in ranges.items()}
        return plan_geos(panel, responses, budget, 13, GeoLimits(cells=cells))

    joint = geos(limits)
    assert joint.spend[0, retail] == pytest.approx(0.5 * FLAT, rel=1e-9, abs=0.0)
    assert joint.unidentified == ((NATIONAL, Channel("retail")),)
    assert geos(held).unidentified == ()


def test_the_test_chosen_for_the_plan_is_of_the_flat_channel(bayes: BayesFit) -> None:
    """Six candidates, each channel cut to none or doubled for 4 weeks in a fifth of the market
    after the history, and a test budget of 100: the choice (ADR 0031) is to test, its set holds a
    test of retail, and its best single test is retail's. Television's and search's returns are
    read to within a ratio of about 1.3 and 1.5 at 90 %, and a test of either removes, in absolute
    value, less than a hundredth of the regret the lesser of retail's two removes, which is more
    than 10: retail's cut removes 80 at this seed, its Monte Carlo standard error 38, read on 200
    outer draws.

    Measured on the fixture's fit at seeds 0 to 9, and at seed 3 with XLA's code held to AVX and
    to SSE4.2, retail's tests removed 17 to 147 and the others at most 0.23, a ratio of at most
    0.0016. At 4 of the 12 the set took search's doubling beside a test of retail, for the 2 its
    return nets over its spend: at a cost of 31 it fits the budget beside retail's doubling, 64,
    and beside retail's cut where the revenue the cut is expected to lose, 65 to 86 over the fits,
    is under 69. CI's 3.15 leg chose a test of search beside retail's too. So the set is held to
    hold a test of retail, not that test alone."""
    panel = flat_world()
    planner = readings(bayes)
    current = {c: float(panel.spend[0, -13:, j].mean()) for j, c in enumerate(planner.channels)}
    limits = {c: Range(0.5 * v, 2.0 * v) for c, v in current.items()}
    candidates = designs(
        panel,
        NATIONAL,
        list(planner.channels),
        GroupNoise(NOISE),
        when=After(current),
        multipliers=(0.0, 2.0),
        durations=(4,),
        shares=(0.2,),
    )
    choice = choose_tests(
        panel,
        NATIONAL,
        planner,
        candidates,
        13 * sum(current.values()),
        13,
        limits,
        test_budget=100.0,
        seed=4,
        draws=400,
    )
    assert choice.verdict == "test"
    assert choice.best is not None
    assert choice.single is not None
    assert Channel("retail") in {c.channel for c in choice.best.candidates}
    assert {c.channel for c in choice.single.candidates} == {Channel("retail")}
    singles = [v for v in choice.values if len(v.candidates) == 1]
    retail = [v.reduction for v in singles if v.candidates[0].channel == Channel("retail")]
    others = [v.reduction for v in singles if v.candidates[0].channel != Channel("retail")]
    assert len(retail) == 2
    assert len(others) == 4
    assert min(retail) > 10.0
    assert max(map(abs, others)) < 0.01 * min(retail)
