"""The twin conflation family (ADR 0073): two worlds of one history, a line whose slope drifts and
a fixed S-shaped curve (`twin_worlds`), whose best plans differ by more than 0.05 a euro, a ridge
fit that scores their held-out weeks alike, each engine's plan on both, and a seesaw test that
tells them apart where a go-dark cannot."""

from __future__ import annotations

import math

import numpy as np
import pytest
from chc.allocation import allocate
from scipy import stats

import twin_worlds as tw
from mixdyn import NATIONAL, Channel, MediaPanel
from mixdyn.decisions import Range, Readings, plan_posterior, readings, stacked_readings
from mixdyn.experiments import After, Candidate, GroupNoise, designs, predicted_lift, seesaw
from mixdyn.models import DiscountGrid, DlmSpec, Effect, PerEffect, Trend, choose_dlm
from mixdyn.models.bayes_mmm import BayesMedia, BayesSpec, HillCurve, TanhCurve, fit_bayes
from mixdyn.models.ridge_mmm import Media, RidgeSpec, fit_ridge
from mixdyn.transforms import Geometric, Hill, TanhShape, Transform

SEED = 20261106
HELD = 78
"""The history's first weeks a fit reads; the 26 after them are held out."""


@pytest.fixture(scope="module")
def panels() -> dict[tw.World, MediaPanel]:
    return tw.twins(SEED)


def test_each_twins_best_plan_costs_the_other_more_than_0_05_a_euro() -> None:
    line, curve = tw.best("line"), tw.best("curve")
    assert line - curve > 10.0
    assert tw.regret("curve", line) > 0.05
    assert tw.regret("line", curve) > 0.05
    assert tw.regret("line", line) == 0.0
    assert tw.regret("curve", curve) == 0.0


def _held_out(panel: MediaPanel) -> np.ndarray:
    """Each held-out week's log density under a ridge fitted to the weeks before it: television
    through a Hill of slope 3, half-saturated at 60, and search through one of slope 1, half at
    50, neither carried over; the noise the fit's residuals' root mean square."""
    tv, paid = panel.spend[0, :, 0], panel.spend[0, :, 1]
    spec = RidgeSpec(
        (
            Media("tv", Geometric(0.0), Hill(3.0, 60.0 / tv[:HELD].max())),
            Media("search", Geometric(0.0), Hill(1.0, 50.0 / paid[:HELD].max())),
        ),
        (),
        penalty=1e-3,
    )
    fit = fit_ridge(panel, NATIONAL, spec, window=(panel.dates[0], panel.dates[HELD - 1]))
    channels = fit.channels()
    mean = fit.intercept + np.sum(
        [np.asarray(channels[c](panel.spend[0, :, j])) for j, c in enumerate(("tv", "search"))],
        axis=0,
    )
    y = panel.kpi[0]
    sigma = math.sqrt(np.mean((y[:HELD] - mean[:HELD]) ** 2))
    return stats.norm.logpdf(y[HELD:], mean[HELD:], sigma)


def test_a_ridge_fit_scores_the_twins_held_out_weeks_alike(
    panels: dict[tw.World, MediaPanel],
) -> None:
    """The held-out log scores' difference, week by week, line less curve: its 95 % interval over
    the 26 weeks holds 0."""
    gap = _held_out(panels["line"]) - _held_out(panels["curve"])
    half = stats.norm.ppf(0.975) * gap.std(ddof=1) * math.sqrt(gap.size)
    assert abs(gap.sum()) < half


SEARCH = Transform(0.0, tw.SEARCH_SCALE * math.atanh(0.5), TanhShape(), tw.WEEKS)
"""Search's curve as the worlds hold it, which the line's model is handed."""
LIMITS = {Channel("tv"): Range(*tw.TV_RANGE), Channel("search"): Range(*tw.SEARCH_RANGE)}


def _engines(panel: MediaPanel) -> dict[str, float]:
    """Each engine's television spend a week in its plan on `panel`'s whole history.

    - `"ridge"`: the ridge of `_held_out` on every week, its curves planned by `chc.allocation`;
    - `"hill"`, `"tanh"`: the Bayesian model with each curve family (ADR 0011), two chains of 300
      warm-up iterations and 200 draws, planned for its posterior's expected worth;
    - `"stacked"`: the two by their stacking weights, planned the same way;
    - `"dlm line"`: the dynamic linear model whose television coefficient drifts on its spend, a
      line, its discounts chosen on a grid (`choose_dlm`), search through its own curve: planned at
      the best fit's coefficients at the last week.
    """
    tv, paid = panel.spend[0, :, 0], panel.spend[0, :, 1]
    out: dict[str, float] = {}
    spec = RidgeSpec(
        (
            Media("tv", Geometric(0.0), Hill(3.0, 60.0 / tv.max())),
            Media("search", Geometric(0.0), Hill(1.0, 50.0 / paid.max())),
        ),
        (),
        penalty=1e-3,
    )
    ridge = fit_ridge(panel, NATIONAL, spec).channels()
    plan = allocate(
        [ridge["tv"], ridge["search"]],
        tw.BUDGET * tw.PERIODS,
        tw.PERIODS,
        lower=np.array([tw.TV_RANGE[0], tw.SEARCH_RANGE[0]]),
        upper=np.array([tw.TV_RANGE[1], tw.SEARCH_RANGE[1]]),
        history=panel.spend[0],
    )
    out["ridge"] = float(plan.spend[0])
    fits = []
    for k, (name, curve) in enumerate((("hill", HillCurve()), ("tanh", TanhCurve()))):
        model = BayesSpec((BayesMedia("tv"), BayesMedia("search")), saturation=curve)
        fit = fit_bayes(panel, NATIONAL, model, seed=11 + k, chains=2, warmup=300, draws=200)
        fits.append(fit)
        planned = plan_posterior(
            panel, NATIONAL, readings(fit), tw.BUDGET * tw.PERIODS, tw.PERIODS, LIMITS
        )
        out[name] = float(planned.spend[0])
    planned = plan_posterior(
        panel, NATIONAL, stacked_readings(fits), tw.BUDGET * tw.PERIODS, tw.PERIODS, LIMITS
    )
    out["stacked"] = float(planned.spend[0])
    choice = choose_dlm(
        panel,
        NATIONAL,
        DlmSpec((Effect("tv", 1.0), Effect("search", 1.0, transform=SEARCH)), trend=Trend(1, 1.0)),
        DiscountGrid(effects=PerEffect((0.9, 0.95, 0.98, 1.0))),
    )
    state = choice.fit.last()[0]
    slope, height = float(state[1]), float(state[2])
    out["dlm line"] = tw.best_split(
        lambda x: slope * x + height * float(np.tanh((tw.BUDGET - x) / tw.SEARCH_SCALE))
    )
    return out


@pytest.fixture(scope="module")
def engines(panels: dict[tw.World, MediaPanel]) -> dict[tw.World, dict[str, float]]:
    return {world: _engines(panel) for world, panel in panels.items()}


def test_no_engine_tells_the_twins_apart_from_their_history(
    engines: dict[tw.World, dict[str, float]],
) -> None:
    """Each engine plans television within 5 a week alike on both twins, at most 2.2 apart at this
    seed (the Bayesian model with Hill curves), while the twins' best plans lie 16.1 apart: what an
    engine plans is the same whichever twin it is in, so it pays one twin's regret or the other's.
    """
    for name in engines["line"]:
        assert abs(engines["line"][name] - engines["curve"][name]) < 5.0, name


PLAN = After({Channel("tv"): 100.0, Channel("search"): 50.0})
"""The plan in place after the history: television at the level the ramp ends at."""
SHARE = 0.3


def _gaps(world: tw.World, candidate: Candidate) -> np.ndarray:
    """The group's KPI gap in `world` at each period of the candidate's readout: the share times
    the market's response to the spend as tested less as planned, each over the share. Neither
    world carries spend over."""
    s = candidate.share
    slope = float(tw.drift(tw.WEEKS - 1))
    respond = tw.curve if world == "curve" else (lambda x: slope * np.asarray(x))
    gap = s * (respond(candidate.spend / s) - respond(candidate.planned / s))
    return gap[-candidate.readout :]


def _separation(candidate: Candidate) -> float:
    """The distance between the twins' expected gaps in the gaps' own noise: the likelihood ratio
    between the twins, each as likely before the test, picks the twin that drew the reading with
    probability `Phi(d / 2)`."""
    sd = math.sqrt(GroupNoise(tw.NOISE).covariance(candidate.readout, candidate.share)[0, 0])
    return float(np.linalg.norm(_gaps("line", candidate) - _gaps("curve", candidate)) / sd)


def _picked(candidate: Candidate, draws: int, seed: int) -> float:
    """The share of `draws` readings drawn in each twin, independent normal gaps about its own,
    whose likelihood ratio picks the twin that drew them."""
    rng = np.random.default_rng(seed)
    sd = math.sqrt(GroupNoise(tw.NOISE).covariance(candidate.readout, candidate.share)[0, 0])
    mean = {w: _gaps(w, candidate) for w in ("line", "curve")}
    right = 0
    for truth, other in (("line", "curve"), ("curve", "line")):
        y = mean[truth] + sd * rng.standard_normal((draws, candidate.readout))
        right += int(np.sum(((y - mean[truth]) ** 2).sum(1) < ((y - mean[other]) ** 2).sum(1)))
    return right / (2 * draws)


@pytest.fixture(scope="module")
def tests_of_tv(panels: dict[tw.World, MediaPanel]) -> dict[str, Candidate]:
    """Television's tests in a group of 0.3 of the market after the history, each with a cooldown
    of 8 weeks at the plan: a seesaw of 2-week blocks at twice the plan and none over 16 weeks;
    the default seesaw, 2-week blocks at 1.5 and 0.5 times the plan over 16 weeks; a go-dark of 16
    weeks; and a heavy-up to twice the plan for 16 weeks."""
    panel, noise, tv = panels["curve"], GroupNoise(tw.NOISE), [Channel("tv")]

    def stretch(multiplier: float) -> Candidate:
        laid = designs(
            panel,
            NATIONAL,
            tv,
            noise,
            when=PLAN,
            multipliers=(multiplier,),
            durations=(16,),
            cooldown=8,
            shares=(SHARE,),
        )
        return laid[0]

    return {
        "seesaw": seesaw(
            panel, NATIONAL, tv, noise, when=PLAN, high=2.0, low=0.0, cooldown=8, shares=(SHARE,)
        )[0],
        "default seesaw": seesaw(
            panel, NATIONAL, tv, noise, when=PLAN, cooldown=8, shares=(SHARE,)
        )[0],
        "go-dark": stretch(0.0),
        "heavy-up": stretch(2.0),
    }


def test_a_go_dark_reads_what_the_twins_share(tests_of_tv: dict[str, Candidate]) -> None:
    """At the plan's spend the line's slope is the curve's average return, and at none both
    return nothing: the twins expect the same gaps of a go-dark, so no reading picks either."""
    dark = tests_of_tv["go-dark"]
    np.testing.assert_allclose(_gaps("line", dark), _gaps("curve", dark), rtol=1e-12, atol=0.0)
    assert _separation(dark) < 1e-9


def test_a_seesaw_sums_to_no_lift_under_the_line_and_separates_the_twins(
    tests_of_tv: dict[str, Candidate],
) -> None:
    """Under the line a seesaw's high and low blocks are equal and opposite, so its summed lift
    is 0 whatever the slope; under the curve it is not. The likelihood ratio on its gaps picks the
    twin that drew them with probability `Phi(d / 2)`: above 0.99 for the seesaw at twice the plan
    and none, above 0.95 for the default one. Over 4000 readings in each twin the default's share
    picked rightly is within 4 Monte Carlo standard errors of its rate, a false alarm below 1e-4."""
    for name in ("seesaw", "default seesaw"):
        swing = tests_of_tv[name]
        assert swing.spend.sum() == pytest.approx(swing.planned.sum(), rel=1e-15, abs=0.0)
        line = _gaps("line", swing)
        assert abs(line.sum()) < 1e-9 * np.abs(line).max()
        assert _gaps("curve", swing).sum() < -0.5 * SHARE * tw.NOISE
    assert stats.norm.cdf(_separation(tests_of_tv["seesaw"]) / 2.0) > 0.99
    default = tests_of_tv["default seesaw"]
    rate = float(stats.norm.cdf(_separation(default) / 2.0))
    assert rate > 0.95
    picked = _picked(default, 4000, seed=SEED)
    assert abs(picked - rate) < 4.0 * math.sqrt(rate * (1.0 - rate) / 8000)


def test_a_heavy_up_separates_the_twins_too_at_the_cost_of_the_spend_it_adds(
    tests_of_tv: dict[str, Candidate],
) -> None:
    """Doubling television for the 16 weeks reads the curve's flattening in every week of them,
    so it separates the twins at least as far as the seesaw, which doubles it in half of them;
    it spends 480 more in the group, the seesaw nothing."""
    heavy, swing = tests_of_tv["heavy-up"], tests_of_tv["seesaw"]
    assert _separation(heavy) >= _separation(swing)
    assert heavy.spend.sum() - heavy.planned.sum() == pytest.approx(
        16 * 100.0 * SHARE, rel=1e-12, abs=0.0
    )


def test_the_engine_predicts_the_seesaws_lift_under_the_curve_as_the_world_reads_it(
    tests_of_tv: dict[str, Candidate],
) -> None:
    """The curve's twin as one draw a planner reads, no carryover and a Hill of slope 3 half at
    60: `predicted_lift` times the share is the curve's summed gaps."""
    truth = Readings(
        channels=(Channel("tv"), Channel("search")),
        retention=np.zeros((1, 2)),
        half_saturation=np.array([[tw.HALF, 50.0]]),
        slope=np.array([[tw.SLOPE, 1.0]]),
        coefficient=np.array([[tw.HEIGHT, tw.SEARCH_HEIGHT]]),
        tanh=np.array([False]),
        fit=np.array([0]),
        chain=np.array([0]),
        stacking=np.ones(1),
        pointwise=None,
        length=tw.WEEKS,
    )
    swing = tests_of_tv["seesaw"]
    assert SHARE * float(predicted_lift(truth, swing)[0]) == pytest.approx(
        float(_gaps("curve", swing).sum()), rel=1e-12, abs=0.0
    )
