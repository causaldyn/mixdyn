"""The nested model on a world where television moves branded searches and searches move sales:
stage one is not the mediator explaining itself and carries the spend over, a channel's return on
the KPI is its direct return plus the mediator's coefficient times its return on the mediator, the
television's comes back where a one-stage fit that controls for the mediator blocks it, the
planner reads a mediated channel's curve through the mediator, each stage reads the lift
estimates its own parameters set, and a lift on sales of a channel that moves the searches reads
both, which are then drawn as one model, its log density and posterior Stan's on the worlds
CmdStan 2.38 recorded."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from chc import response
from scipy import stats

from mixdyn import Cadence, Channel, Geo, MediaPanel, Revenue
from mixdyn.calibration import LiftEstimate
from mixdyn.models import nested
from mixdyn.models.bayes_mmm import (
    BayesFit,
    BayesMedia,
    BayesSpec,
    HillCurve,
    RoiPrior,
    TanhCurve,
    fit_bayes,
)
from mixdyn.models.nested import NestedFit, NestedSpec, fit_nested, nested_model
from mixdyn.models.ridge_mmm import Control, FourierSeason
from test_bayes_mmm import bound, standardised

NATIONAL = Geo("national")
WEEKS = 104
TRUTH = {"tv_on_searches": 0.5, "searches_on_sales": 2.0, "search_on_sales": 1.5}
"""Television's return on searches, in searches a euro; a search's on sales, in euros; and the
search ads' on sales, in euros a euro: television's return on sales through the searches is 1.
Sales fall by 20 a euro of price, which moves the searches not at all."""


def _saturated(spend: np.ndarray, retention: float, half: float, slope: float) -> np.ndarray:
    weights = retention ** np.arange(spend.size)
    adstock = np.convolve(spend / spend[spend > 0].mean(), weights / weights.sum())[: spend.size]
    return adstock**slope / (adstock**slope + half**slope)


def mediation_world() -> MediaPanel:
    rng = np.random.default_rng(20261002)
    t = np.arange(WEEKS)
    tv = rng.gamma(2.0, 50.0, WEEKS) * (rng.random(WEEKS) > 0.3)
    search = rng.gamma(2.0, 30.0, WEEKS) * (rng.random(WEEKS) > 0.2)
    season = np.sin(2.0 * np.pi * t / 52.18)
    h = _saturated(tv, 0.7, 1.0, 1.0)
    searches = (
        500.0
        + 50.0 * season
        + TRUTH["tv_on_searches"] * tv.sum() * h / h.sum()
        + 15.0 * rng.standard_normal(WEEKS)
    )
    h = _saturated(search, 0.1, 0.8, 1.0)
    sales = (
        1000.0
        + 40.0 * season
        + TRUTH["searches_on_sales"] * searches
        + TRUTH["search_on_sales"] * search.sum() * h / h.sum()
        + 20.0 * rng.standard_normal(WEEKS)
    )
    price = 10.0 + 0.5 * rng.standard_normal(WEEKS)
    sales -= 20.0 * (price - 10.0)
    return MediaPanel(
        dates=np.datetime64("2023-01-02") + np.timedelta64(7, "D") * t,
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=(Channel("tv"), Channel("search")),
        controls=("price", "searches"),
        kpi=sales[None, :],
        kpi_unit=Revenue(),
        spend=np.column_stack([tv, search])[None],
        currency="EUR",
        control_values=np.column_stack([price, searches])[None],
        filled=np.zeros((1, WEEKS), dtype=bool),
    )


SEASON = FourierSeason(52.18, 1)
SPEC = NestedSpec(
    "searches",
    BayesSpec((BayesMedia("tv", RoiPrior(0.0, 1.0)),), season=SEASON),
    BayesSpec(
        (BayesMedia("search"),),
        (Control("price", upper=0.0), Control("searches", lower=0.0)),
        season=SEASON,
    ),
)
TV_ON_SEARCHES = LiftEstimate(
    np.r_[np.full(4, 20.0), np.zeros(6)], np.full(10, 20.0), 6, -35.0, 15.0, share=0.2
)
"""A go-dark of television in a fifth of the market for six weeks, read on the searches."""
SEARCH_ON_SALES = LiftEstimate(
    np.r_[np.full(4, 5.0), np.full(4, 10.0)], np.full(8, 5.0), 4, 8.0, 4.0, share=0.1
)
"""A heavy-up of the search ads in a tenth of the market for four weeks, read on sales."""
TV_ON_SALES = dataclasses.replace(TV_ON_SEARCHES, lift=-70.0, se=30.0)
"""Television's go-dark read on sales, which it moves through the searches."""
CROSSED = NestedSpec(
    "searches",
    BayesSpec((BayesMedia("search"), BayesMedia("tv", RoiPrior(0.0, 1.0))), season=SEASON),
    dataclasses.replace(SPEC.stage_two, media=(BayesMedia("tv"), BayesMedia("search"))),
)
"""Both channels on the searches and on sales directly, in the other order in each stage."""


@pytest.fixture(scope="module")
def fits() -> tuple[MediaPanel, NestedFit, BayesFit]:
    """The nested fit, and a one-stage fit of both channels that controls for the searches."""
    panel = mediation_world()
    nested = fit_nested(panel, NATIONAL, SPEC, seed=5, chains=2, warmup=300, draws=200)
    one_stage = fit_bayes(
        panel,
        NATIONAL,
        dataclasses.replace(SPEC.stage_two, media=(BayesMedia("tv"), BayesMedia("search"))),
        seed=6,
        chains=2,
        warmup=300,
        draws=200,
    )
    return panel, nested, one_stage


def test_stage_one_does_not_explain_the_mediator_by_itself(
    fits: tuple[MediaPanel, NestedFit, BayesFit],
) -> None:
    """The searches' fitted share of their variance is the share television and the season
    explain, not 1, which a stage one that regressed a function of its own regressors on them
    would read."""
    panel, nested, _ = fits
    searches = panel.control_values[0, :, 1]
    residual = searches - nested.stage_one.estimated
    share = 1.0 - residual.var() / searches.var()
    assert 0.5 < share < 0.99


def test_stage_one_carries_the_spend_over(
    fits: tuple[MediaPanel, NestedFit, BayesFit],
) -> None:
    """Television's carryover on the searches, 0.7 in the world, is read back against a prior
    whose mean is 0.25."""
    _, nested, _ = fits
    retention = float(np.median(nested.stage_one.draws["retention"][..., 0]))
    assert 0.5 < retention < 0.85


def test_a_channel_s_return_adds_its_direct_and_mediated_paths(
    fits: tuple[MediaPanel, NestedFit, BayesFit],
) -> None:
    """Draw by draw: television's, the searches' coefficient times its return on the searches;
    the search ads', stage two's own; and a channel in both stages, the sum."""
    _, nested, one_stage = fits
    gamma = nested.mediator_coefficient()
    roi = nested.roi()
    np.testing.assert_array_equal(roi["tv"], gamma * nested.stage_one.draws["roi"][..., 0])
    np.testing.assert_array_equal(roi["search"], nested.stage_two.draws["roi"][..., 0])
    both = NestedFit(
        dataclasses.replace(SPEC, stage_two=one_stage.spec),
        nested.stage_one,
        one_stage,
        nested.mediated_log_likelihood,
    )
    np.testing.assert_array_equal(
        both.roi()["tv"],
        both.mediator_coefficient() * nested.stage_one.draws["roi"][..., 0]
        + one_stage.draws["roi"][..., 0],
    )
    with pytest.raises(ValueError, match=r"\['tv'\] act directly and through the mediator"):
        both.channels()


def test_the_mediated_return_comes_back_where_one_stage_blocks_it(
    fits: tuple[MediaPanel, NestedFit, BayesFit],
) -> None:
    """Television moves sales only through the searches, by 1 a euro: the nested fit's 90 %
    interval holds it, while a one-stage fit that controls for the searches reads almost none."""
    _, nested, one_stage = fits
    low, high = np.quantile(nested.roi()["tv"], [0.05, 0.95])
    assert low < TRUTH["tv_on_searches"] * TRUTH["searches_on_sales"] < high
    assert np.median(one_stage.draws["roi"][..., 0]) < 0.3 < low


def test_the_planner_reads_a_mediated_channel_through_the_mediator(
    fits: tuple[MediaPanel, NestedFit, BayesFit],
) -> None:
    """Television's curve is stage one's at its medians, scaled by the searches' median
    coefficient; the search ads' is stage two's."""
    panel, nested, _ = fits
    spend = panel.spend[0, :, 0]
    gamma = float(np.median(nested.mediator_coefficient()))
    np.testing.assert_allclose(
        np.asarray(nested.channels()["tv"](spend)),
        gamma * np.asarray(nested.stage_one.channels()["tv"](spend)),
        rtol=1e-12,
    )
    spend = panel.spend[0, :, 1]
    np.testing.assert_array_equal(
        np.asarray(nested.channels()["search"](spend)),
        np.asarray(nested.stage_two.channels()["search"](spend)),
    )


def test_each_pair_of_draws_returns_the_nested_return_on_the_window(
    fits: tuple[MediaPanel, NestedFit, BayesFit],
) -> None:
    """At nine of the stages' draws, evenly spaced and paired: television through the searches
    contributes the mediator's coefficient times its return on them, the search ads their own,
    each times the window's spend, to `1e-12`. A channel in both stages is refused."""
    panel, nested, one_stage = fits
    roi = {name: draws.reshape(-1) for name, draws in nested.roi().items()}
    rows = np.linspace(0, roi["tv"].size - 1, 9).round().astype(int)
    for row, channels in zip(rows, nested.channel_draws(9), strict=True):
        for j, name in enumerate(("tv", "search")):
            spend = panel.spend[0, :, j]
            assert response.contribution(channels[name], spend) == pytest.approx(
                roi[name][row] * spend.sum(), rel=1e-12
            ), name
    both = NestedFit(
        dataclasses.replace(SPEC, stage_two=one_stage.spec),
        nested.stage_one,
        one_stage,
        nested.mediated_log_likelihood,
    )
    with pytest.raises(ValueError, match=r"\['tv'\] act directly and through the mediator"):
        both.channel_draws(1)


def test_both_stages_read_the_window_and_the_draws_asked_for() -> None:
    """And the mediator's coefficient is its own column's, not the first control's; drawn apart,
    the fit reads no lift through the mediator."""
    panel = mediation_world()
    window = (panel.dates[24], panel.dates[-1])
    nested = fit_nested(panel, NATIONAL, SPEC, seed=7, window=window, chains=2, warmup=50, draws=20)
    for stage in (nested.stage_one, nested.stage_two):
        np.testing.assert_array_equal(stage.dates, panel.dates[24:])
        assert stage.draws["roi"].shape == (2, 20, 1)
    np.testing.assert_array_equal(
        nested.mediator_coefficient(), nested.stage_two.draws["control"][..., 1]
    )
    assert np.all(nested.mediator_coefficient() > 0.0)
    assert np.all(nested.stage_two.draws["control"][..., 0] < 0.0)
    assert nested.mediated_log_likelihood.shape == (2, 20, 0)


DEFAULT = dataclasses.replace(
    SPEC, stage_one=dataclasses.replace(SPEC.stage_one, media=(BayesMedia("tv"),))
)
"""`SPEC` with television's return on the searches left at the engine's default prior."""


def test_stage_one_reads_the_default_return_prior_at_the_mediator_s_scale() -> None:
    """Television left at `RoiPrior()` on the searches: its median is the default's times the
    searches' mean over sales' over the window, a quarter here, and the model and a fit drawn
    apart both hold that prior; a prior stated otherwise is read as stated."""
    panel = mediation_world()
    window = (panel.dates[24], panel.dates[-1])
    searches = panel.control_values[0, 24:, panel.controls.index("searches")]
    ratio = searches.mean() / panel.kpi[0, 24:].mean()
    assert 0.2 < ratio < 0.3
    model = nested_model(panel, NATIONAL, DEFAULT, window=window)
    (media,) = model.stage_one.spec.media
    assert isinstance(media.roi, RoiPrior)
    assert media.roi.mu == pytest.approx(RoiPrior().mu + math.log(ratio), rel=1e-12, abs=0.0)
    assert (media.roi.sigma, media.roi.counts) == (RoiPrior().sigma, "window")
    fit = fit_nested(panel, NATIONAL, DEFAULT, seed=7, window=window, chains=2, warmup=50, draws=20)
    assert fit.spec == DEFAULT
    assert fit.stage_one.spec == model.stage_one.spec
    assert nested_model(panel, NATIONAL, SPEC, window=window).stage_one.spec == SPEC.stage_one


@pytest.mark.parametrize("unit", [1e-3, 1e3])
def test_the_nested_model_does_not_move_with_the_mediator_s_unit(unit: float) -> None:
    """The searches counted `unit` times over, in thousands or in thousandths, with a lift on them
    and one on sales through them: at ten points of both stages' coordinates, television's log
    return on the searches moved by the log of `unit`, the joint log density and its gradient
    under the default prior are those on the searches as counted, to `1e-12`, so the posterior
    on the stages' own scales and television's return on sales do not move. Under a prior stated
    in the searches' units, the density moves by that prior's own change, and only by it."""
    panel = mediation_world()
    values = panel.control_values.copy()
    values[..., panel.controls.index("searches")] *= unit
    counted = dataclasses.replace(panel, control_values=values)
    estimates = {"tv": [TV_ON_SALES], "search": [SEARCH_ON_SALES]}
    on_searches = TV_ON_SEARCHES
    on_counted = dataclasses.replace(
        on_searches, lift=unit * on_searches.lift, se=unit * on_searches.se
    )
    for spec in (DEFAULT, SPEC):
        model = nested_model(
            panel, NATIONAL, spec, estimates=estimates, mediator_estimates={"tv": [on_searches]}
        )
        moved = nested_model(
            counted, NATIONAL, spec, estimates=estimates, mediator_estimates={"tv": [on_counted]}
        )
        j = model.stage_one.coordinates.index("log_roi[tv]")
        for theta in np.random.default_rng(11).uniform(-1.5, 1.5, (10, model.dimension)):
            shifted = theta.copy()
            shifted[j] += math.log(unit)
            value, gradient = model.log_density(theta)
            there, slope = moved.log_density(shifted)
            gradient = gradient.copy()
            if spec is DEFAULT:
                assert there == pytest.approx(value, rel=1e-12, abs=0.0)
            else:
                change = stats.norm.logpdf(shifted[j]) - stats.norm.logpdf(theta[j])
                assert there - value == pytest.approx(change, rel=0.0, abs=1e-12 * abs(value))
                gradient[j] -= math.log(unit)
            np.testing.assert_allclose(slope, gradient, rtol=1e-11, atol=1e-11)


def _lift(parameters: dict[str, np.ndarray], j: int, length: int, e: LiftEstimate) -> float:
    """The lift over the test's share of the market at one draw of a stage's `parameters` in the
    panel's units, in the stage's KPI units: channel `j`'s coefficient times the gap in its curve,
    carried over by kernels `length` long, at the spend the share saw, over the readout."""
    retention, half, slope, coefficient = (
        parameters[k][j] for k in ("retention", "half_saturation", "slope", "coefficient")
    )
    weights = retention ** np.arange(length)
    gap = 0.0
    for series, sign in ((e.spend, 1.0), (e.planned, -1.0)):
        padded = np.r_[np.zeros(length - series.size), series / e.share]
        adstock = np.convolve(padded, weights / weights.sum())[:length][-e.readout :]
        gap += sign * np.sum(adstock**slope / (adstock**slope + half**slope))
    return float(coefficient * gap)


def _expected_lift(fit: BayesFit, j: int, e: LiftEstimate) -> np.ndarray:
    """Each draw's lift, as `_lift` reads it at the draw."""
    out = np.empty(fit.draws["roi"].shape[:2])
    for index in np.ndindex(out.shape):
        out[index] = _lift({k: v[index] for k, v in fit.draws.items()}, j, fit.length, e)
    return out


def test_each_stage_reads_the_lifts_its_own_parameters_set() -> None:
    """Television's lift on the searches is a term of stage one's likelihood, in searches, and the
    search ads' on sales one of stage two's, in euros: at each draw, the lift over its share is
    normal about the channel's coefficient times the gap in its curve, its standard error over
    its share."""
    panel = mediation_world()
    nested = fit_nested(
        panel,
        NATIONAL,
        SPEC,
        seed=8,
        estimates={"search": [SEARCH_ON_SALES]},
        mediator_estimates={"tv": [TV_ON_SEARCHES]},
        chains=2,
        warmup=50,
        draws=20,
    )
    for stage, e in ((nested.stage_one, TV_ON_SEARCHES), (nested.stage_two, SEARCH_ON_SALES)):
        assert stage.log_likelihood.shape == (2, 20, WEEKS + 1)
        np.testing.assert_allclose(
            stage.log_likelihood[..., -1],
            stats.norm.logpdf(e.lift / e.share, _expected_lift(stage, 0, e), e.se / e.share),
            rtol=1e-12,
        )


LONG = LiftEstimate(
    np.r_[np.full(WEEKS + 4, 20.0), np.zeros(6)], np.full(WEEKS + 10, 20.0), 6, -60.0, 25.0, 0.25
)
"""A go-dark of television read on sales, its spend recorded from ten weeks before the panel."""


@pytest.mark.parametrize(
    ("spec", "lifts"),
    [
        (
            SPEC,
            {
                "tv": [
                    TV_ON_SALES,
                    dataclasses.replace(TV_ON_SALES, lift=-50.0, se=20.0, share=0.3),
                    LONG,
                ]
            },
        ),
        (CROSSED, {"tv": [TV_ON_SALES, LONG], "search": [SEARCH_ON_SALES]}),
    ],
    ids=["mediated", "crossed"],
)
def test_a_lift_on_sales_through_the_searches_adds_its_term_to_both_stages(
    spec: NestedSpec, lifts: dict[str, list[LiftEstimate]]
) -> None:
    """At points of both stages' coordinates, the joint log density less each stage's is each
    lift's normal log density about the searches' coefficient times its channel's lift on them,
    plus its lift on sales where stage two has the channel, by hand in the panel's units, plus the
    log of sales' unit; `log_likelihood` is each lift's term in the panel's units, channel by
    channel in stage one's order. Stage one's kernels run over the longest lift's spend, and stage
    two's only where it has that lift's channel."""
    model = nested_model(mediation_world(), NATIONAL, spec, estimates=lifts)
    one, two, d = model.stage_one, model.stage_two, model.stage_one.dimension
    staged = [m.channel for m in spec.stage_one.media]
    direct = [m.channel for m in spec.stage_two.media]
    first, second = WEEKS + 10, (WEEKS + 10 if "tv" in direct else WEEKS)
    assert (one.lagged.shape[2], two.lagged.shape[2]) == (first, second)
    k = [c.column for c in spec.stage_two.controls].index("searches")
    for theta in np.random.default_rng(3).uniform(-1.5, 1.5, (5, model.dimension)):
        p1 = {name: v[0] for name, v in one.parameters(theta[:d]).items()}
        p2 = {name: v[0] for name, v in two.parameters(theta[d:]).items()}
        expected = np.array(
            [
                stats.norm.logpdf(
                    e.lift / e.share,
                    p2["control"][k] * _lift(p1, staged.index(c), first, e)
                    + (_lift(p2, direct.index(c), second, e) if c in direct else 0.0),
                    e.se / e.share,
                )
                for c in staged
                for e in lifts.get(c, [])
            ]
        )
        apart = one.log_density(theta[:d])[0] + two.log_density(theta[d:])[0]
        assert model.log_density(theta)[0] == pytest.approx(
            apart + np.sum(expected + np.log(two.kpi_unit)), rel=1e-13, abs=0.0
        )
        np.testing.assert_allclose(model.log_likelihood(theta)[0], expected, rtol=1e-12)


def test_a_lift_on_sales_through_the_searches_is_read_at_each_draw() -> None:
    """With it the stages are drawn as one model: each stage's likelihood holds its own lift as
    drawn apart, the lift on sales is normal at each draw about the searches' coefficient times
    television's lift on them, by hand from the draws, and both stages hold the one sampler's
    divergences and energies."""
    nested = fit_nested(
        mediation_world(),
        NATIONAL,
        SPEC,
        seed=8,
        estimates={"search": [SEARCH_ON_SALES], "tv": [TV_ON_SALES]},
        mediator_estimates={"tv": [TV_ON_SEARCHES]},
        chains=2,
        warmup=50,
        draws=20,
    )
    for stage, e in ((nested.stage_one, TV_ON_SEARCHES), (nested.stage_two, SEARCH_ON_SALES)):
        assert stage.log_likelihood.shape == (2, 20, WEEKS + 1)
        np.testing.assert_allclose(
            stage.log_likelihood[..., -1],
            stats.norm.logpdf(e.lift / e.share, _expected_lift(stage, 0, e), e.se / e.share),
            rtol=1e-12,
        )
    e = TV_ON_SALES
    assert nested.mediated_log_likelihood.shape == (2, 20, 1)
    np.testing.assert_allclose(
        nested.mediated_log_likelihood[..., 0],
        stats.norm.logpdf(
            e.lift / e.share,
            nested.mediator_coefficient() * _expected_lift(nested.stage_one, 0, e),
            e.se / e.share,
        ),
        rtol=1e-12,
    )
    np.testing.assert_array_equal(nested.stage_one.divergent, nested.stage_two.divergent)
    np.testing.assert_array_equal(nested.stage_one.e_bfmi, nested.stage_two.e_bfmi)


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (
            lambda: dataclasses.replace(
                SPEC,
                stage_one=dataclasses.replace(SPEC.stage_one, controls=(Control("searches"),)),
            ),
            "stage one reads the mediator 'searches'",
        ),
        (
            lambda: dataclasses.replace(
                SPEC, stage_one=dataclasses.replace(SPEC.stage_one, media=(BayesMedia("searches"),))
            ),
            "stage one reads the mediator",
        ),
        (
            lambda: dataclasses.replace(
                SPEC,
                stage_two=dataclasses.replace(
                    SPEC.stage_two, controls=(Control("price", upper=0.0),)
                ),
            ),
            "stage two must read the mediator 'searches'",
        ),
        (
            lambda: dataclasses.replace(
                SPEC,
                stage_two=dataclasses.replace(SPEC.stage_two, controls=(Control("searches"),)),
            ),
            "held at or above 0",
        ),
        (
            lambda: dataclasses.replace(
                SPEC,
                stage_two=dataclasses.replace(
                    SPEC.stage_two, controls=(Control("searches", upper=0.0),)
                ),
            ),
            "held at or above 0",
        ),
        (
            lambda: dataclasses.replace(
                SPEC, stage_one=dataclasses.replace(SPEC.stage_one, kernel=6)
            ),
            "kernels are as long as the longest series a stage reads",
        ),
        (
            lambda: dataclasses.replace(
                SPEC, stage_two=dataclasses.replace(SPEC.stage_two, kernel=6)
            ),
            "kernels are as long as the longest series a stage reads",
        ),
    ],
)
def test_a_nesting_that_cannot_be_is_refused(make, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        make()


@pytest.mark.parametrize(
    ("estimates", "mediator_estimates", "match"),
    [
        (
            None,
            {"search": [TV_ON_SEARCHES]},
            r"estimates name channels the model does not have, \['search'\]",
        ),
        (
            {"radio": [SEARCH_ON_SALES]},
            None,
            r"estimates name channels the model does not have, \['radio'\]",
        ),
        ({"search": []}, None, r"or none, \['search'\]"),
        ({"tv": []}, None, r"or none, \['tv'\]"),
        (
            {"tv": [dataclasses.replace(TV_ON_SALES, geos=frozenset({Geo("north")}))]},
            None,
            r"estimates name geos \['north'\]",
        ),
        (
            {"search": [dataclasses.replace(SEARCH_ON_SALES, geos=frozenset({Geo("north")}))]},
            None,
            r"estimates name geos \['north'\]",
        ),
    ],
    ids=[
        "stage one's channel",
        "stage two's channel",
        "none",
        "none through the mediator",
        "another geo",
        "another geo through the mediator",
    ],
)
def test_what_either_stage_refuses_is_refused_before_either_is_drawn(
    monkeypatch: pytest.MonkeyPatch,
    estimates: dict[str, list[LiftEstimate]] | None,
    mediator_estimates: dict[str, list[LiftEstimate]] | None,
    match: str,
) -> None:
    """Both stages' models are built before either is drawn, apart or as one, so a lift stage two
    refuses does not wait for stage one's posterior."""

    def drawn(*args: object, **kwargs: object) -> BayesFit:
        raise AssertionError("a stage was drawn before both were built")

    monkeypatch.setattr(nested, "fit_bayes", drawn)
    monkeypatch.setattr(nested, "_initial", drawn)
    with pytest.raises(ValueError, match=match):
        fit_nested(
            mediation_world(),
            NATIONAL,
            SPEC,
            seed=1,
            estimates=estimates,
            mediator_estimates=mediator_estimates,
        )


def test_a_panel_without_the_mediator_is_refused() -> None:
    visits = NestedSpec(
        "visits",
        SPEC.stage_one,
        dataclasses.replace(SPEC.stage_two, controls=(Control("visits", lower=0.0),)),
    )
    with pytest.raises(ValueError, match=r"no control 'visits'; it has \['price', 'searches'\]"):
        fit_nested(mediation_world(), NATIONAL, visits, seed=1)


STAN = json.loads((Path(__file__).parent / "fixtures" / "oracles" / "nested_stan.json").read_text())


def stan_world(
    name: str,
) -> tuple[MediaPanel, dict[str, list[LiftEstimate]], dict[str, list[LiftEstimate]], NestedSpec]:
    """The harness's world: its panel, its lifts on sales and on the searches as the fixture
    records them, and its model as the harness states it."""
    case = STAN["cases"][name]
    kpi = np.array(case["panel"]["kpi"])
    panel = MediaPanel(
        dates=np.datetime64("2023-01-02") + np.timedelta64(7, "D") * np.arange(kpi.size),
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=(Channel("tv"), Channel("search")),
        controls=("price", "searches"),
        kpi=kpi[None, :],
        kpi_unit=Revenue(),
        spend=np.array(case["panel"]["spend"])[None],
        currency="EUR",
        control_values=np.array(case["panel"]["controls"])[None],
        filled=np.zeros((1, kpi.size), dtype=bool),
    )
    estimates, mediator_estimates = (
        {
            channel: [
                LiftEstimate(e["spend"], e["planned"], e["readout"], e["lift"], e["se"], e["share"])
                for e in recorded
            ]
            for channel, recorded in case[key].items()
        }
        for key in ("estimates", "mediator_estimates")
    )
    spec = NestedSpec(
        "searches",
        BayesSpec(
            (BayesMedia("tv", RoiPrior(0.0, 1.0)),),
            season=SEASON,
            saturation={"HillCurve": HillCurve(), "TanhCurve": TanhCurve()}[
                case["stage_one_saturation"]
            ],
        ),
        BayesSpec(
            tuple(BayesMedia(c) for c in case["stage_two_media"]),
            (Control("price", upper=0.0), Control("searches", lower=0.0)),
            season=SEASON,
        ),
    )
    return panel, estimates, mediator_estimates, spec


def _stages(case: dict[str, Any]) -> dict[str, Any]:
    """Each stage's parameters' posterior moments, keyed by stage and parameter."""
    return {
        f"{stage}.{key}": moments
        for stage, parameters in case["posterior"].items()
        for key, moments in parameters.items()
    }


def _with_levels(model: nested.NestedModel, theta: jax.Array) -> jax.Array:
    """Stan's points, each stage's intercept coordinate moved to its level."""
    d = model.stage_one.dimension
    first, second = theta[..., :d], theta[..., d:]
    one, two = model.stage_one._with_level(first), model.stage_two._with_level(second)
    return jnp.concatenate([one, two], axis=-1)


@pytest.mark.parametrize("name", sorted(STAN["cases"]))
def test_the_joint_log_density_is_stan_s_up_to_a_constant(name: str) -> None:
    """At 25 points uniform in ``[-2, 2]`` on both stages' coordinates, each stage's intercept
    read as its level, a map whose Jacobian is 1: the log density, with every transform's
    Jacobian, differs from Stan's by one constant, and its gradient by the chain rule is Stan's."""
    case = STAN["cases"][name]
    panel, estimates, mediator_estimates, spec = stan_world(name)
    model = nested_model(
        panel, NATIONAL, spec, estimates=estimates, mediator_estimates=mediator_estimates
    )
    points = np.array(case["points"])
    assert model.dimension == points.shape[1]
    stan_s = jax.jit(jax.value_and_grad(lambda t: model._log_density(_with_levels(model, t))))
    ours = [(float(v), np.asarray(g)) for v, g in (stan_s(jnp.asarray(p)) for p in points)]
    offset = np.array([value for value, _ in ours]) - np.array(case["log_prob"])
    assert np.ptp(offset) <= 1e-12 * np.abs(case["log_prob"]).max()
    np.testing.assert_allclose(
        np.array([gradient for _, gradient in ours]),
        case["gradient"],
        rtol=1e-11,
        atol=1e-12 * np.abs(case["gradient"]).max(),
    )


@pytest.mark.parametrize("name", sorted(STAN["cases"]))
def test_each_observation_s_log_likelihood_is_stan_s(name: str) -> None:
    """At 25 of Stan's draws: stage one's searches and its lift on them, stage two's sales and its
    lift on them, then the lifts on sales through the searches, as Stan's generated quantities
    compute them on the model's scales: in the panel's units, less the log of each one's unit."""
    recorded = STAN["cases"][name]["log_lik_draws"]
    panel, estimates, mediator_estimates, spec = stan_world(name)
    model = nested_model(
        panel, NATIONAL, spec, estimates=estimates, mediator_estimates=mediator_estimates
    )
    theta = np.asarray(_with_levels(model, jnp.asarray(recorded["theta"])))
    one, two, d = model.stage_one, model.stage_two, model.stage_one.dimension
    np.testing.assert_allclose(
        np.concatenate(
            [
                one.log_likelihood(theta[:, :d]) + math.log(one.kpi_unit),
                two.log_likelihood(theta[:, d:]) + math.log(two.kpi_unit),
                model.log_likelihood(theta) + math.log(two.kpi_unit),
            ],
            axis=1,
        ),
        recorded["log_lik"],
        rtol=1e-11,
        atol=1e-11,
    )


JOINT_BOUND = bound(
    [{"posterior": _stages(case), "log_lik": case["log_lik"]} for case in STAN["cases"].values()],
    STAN["provenance"],
)


@pytest.mark.parametrize("name", sorted(STAN["cases"]))
def test_the_joint_posterior_is_stan_s(name: str) -> None:
    """Four chains of 1000 draws after 1000 of warm-up from Pathfinder's starts: each stage's
    parameters' posterior means and standard deviations, in the panel's units, and each
    observation's log-likelihood's, within ``JOINT_BOUND`` joint Monte Carlo standard errors of
    those of Stan's 40 000 draws, as `standardised` reads them; every R-hat below 1.01 and every
    chain's E-BFMI above 0.3; and divergent transitions no more frequent than Stan's, whose own
    chains diverge on both worlds: the engine's share of the two samplers' divergences is not above
    its share of their draws, by a one-sided binomial test at 0.001 (ADR 0025, ADR 0033)."""
    case = STAN["cases"][name]
    panel, estimates, mediator_estimates, spec = stan_world(name)
    fit = fit_nested(
        panel, NATIONAL, spec, seed=1, estimates=estimates, mediator_estimates=mediator_estimates
    )
    model = nested_model(
        panel, NATIONAL, spec, estimates=estimates, mediator_estimates=mediator_estimates
    )
    one, two = model.stage_one.kpi_unit, model.stage_two.kpi_unit
    ours = {
        **{f"stage_one.{k}": v for k, v in fit.stage_one.draws.items()},
        **{f"stage_two.{k}": v for k, v in fit.stage_two.draws.items()},
        "log_lik": np.concatenate(
            [
                fit.stage_one.log_likelihood + math.log(one),
                fit.stage_two.log_likelihood + math.log(two),
                fit.mediated_log_likelihood + math.log(two),
            ],
            axis=2,
        ),
    }
    expected = {**_stages(case), "log_lik": case["log_lik"]}
    assert set(ours) == set(expected)
    for key, stan in expected.items():
        mean, sd = standardised(ours[key].reshape(4, 1000, -1), stan, STAN["provenance"])
        assert np.all(np.abs(mean) <= JOINT_BOUND), key
        assert np.all(np.abs(sd) <= JOINT_BOUND), key
    for stage in (fit.stage_one, fit.stage_two):
        for key, diagnostics in stage.diagnostics.items():
            assert np.all(diagnostics.rhat < 1.01), key
    ours, theirs = int(fit.stage_one.divergent.sum()), case["divergent"]
    share = 4000 / (4000 + STAN["provenance"]["chains"] * STAN["provenance"]["draws"])
    assert stats.binomtest(ours, ours + theirs, share, alternative="greater").pvalue > 1e-3, ours
    assert np.all(fit.stage_one.e_bfmi > 0.3)
