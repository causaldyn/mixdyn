"""The health card of a Bayesian fit: each cell's number on worlds whose answer is known, the
prior predictive at the default priors, and the pooling structure held under a change of a
prior's family.

The worlds are drawn from the one-geo model itself, two channels over two years of weeks and the
KPI log-normal about its mean (ADR 0017): a well-specified one; one whose KPI falls to a
thousandth in six weeks of outage the model has no column for, a confounder that moves the total;
and one whose level is below 0, the channels carrying the KPI above it.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from chc import response
from jax.scipy import stats as jstats

from mixdyn import Cadence, Channel, Geo, MediaPanel, Revenue
from mixdyn.decisions.returns import returns
from mixdyn.inference import compiled
from mixdyn.inference.diagnostics import (
    HealthCard,
    _baseline,
    _media,
    _read,
    health,
    negative_baseline,
    out_of_range,
    posterior_predictive_p,
    prior_predictive,
    prior_shift,
)
from mixdyn.models.bayes_geo import BayesGeoModel, bayes_geo_model
from mixdyn.models.bayes_mmm import (
    _UNIFORM_START,
    BayesFit,
    BayesMedia,
    BayesModel,
    BayesSpec,
    LogNormal,
    RoiPrior,
    _initial,
    _log_normal,
    _thinned,
    bayes_model,
    fit_bayes,
)
from mixdyn.models.nested import NestedSpec, nested_model
from mixdyn.models.ridge_mmm import Control
from test_bayes_geo import stan_world as geo_world

NATIONAL = Geo("national")
WEEKS = 104
OUTAGE = (10, 20, 47, 71, 90, 99)
MERIDIAN = {"max_rhat": 1.2, "min_ppp": 0.05, "max_negative_baseline": 0.2}
"""Meridian's model reviewer's thresholds, the example the card's docstring cites."""


def world(
    seed: int, *, intercept: float = 855.0, scale: float = 1.0, outage: tuple[int, ...] = ()
) -> tuple[MediaPanel, BayesSpec]:
    """Two years of weeks from the one-geo model: tv flighted six weeks on and six off, search
    always on, each through a normalised geometric kernel and a Hill whose half-saturation point
    is in the spend's units, its coefficient set by its return over the window, 2 and 1.5; the KPI
    log-normal about the mean at a log scale of 0.05, then a thousandth of it in the `outage`
    weeks. `scale` multiplies the spend and the half-saturation points alike."""
    rng = np.random.default_rng(seed)
    week = np.arange(WEEKS)
    tv = np.where((week // 6) % 2 == 0, rng.uniform(60.0, 140.0, WEEKS), 0.0)
    spend = scale * np.column_stack([tv, rng.uniform(10.0, 50.0, WEEKS)])
    mean = np.full(WEEKS, intercept)
    for j, (retention, half, slope, roi) in enumerate(
        ((0.5, 60.0, 1.5, 2.0), (0.2, 30.0, 1.0, 1.5))
    ):
        kernel = response.GeometricAdstock(retention, length=WEEKS, normalized=True)
        unit = np.asarray(
            response.Channel(kernel, response.Hill(scale * half, slope), 1.0)(spend[:, j])
        )
        mean += roi * spend[:, j].sum() / unit.sum() * unit
    kpi = mean * np.exp(0.05 * rng.standard_normal(WEEKS) - 0.05**2 / 2.0)
    kpi[list(outage)] *= 0.001
    panel = MediaPanel(
        dates=np.datetime64("2023-01-02") + np.timedelta64(7, "D") * week,
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=(Channel("tv"), Channel("search")),
        controls=("visits",),
        kpi=kpi[None],
        kpi_unit=Revenue(),
        spend=spend[None],
        currency="EUR",
        control_values=(100.0 + 0.05 * spend[:, :1] + rng.normal(0.0, 1.0, (WEEKS, 1)))[None],
        filled=np.zeros((1, WEEKS), dtype=bool),
    )
    return panel, BayesSpec((BayesMedia("tv"), BayesMedia("search")), likelihood=LogNormal())


def _fit(panel: MediaPanel, spec: BayesSpec) -> BayesFit:
    return fit_bayes(panel, NATIONAL, spec, seed=1, chains=4, warmup=300, draws=250)


@pytest.fixture(scope="module")
def well() -> tuple[MediaPanel, BayesFit, HealthCard]:
    panel, spec = world(1)
    fit = _fit(panel, spec)
    return panel, fit, health(fit, panel, NATIONAL, seed=2)


def test_a_well_specified_world_passes_every_cell(
    well: tuple[MediaPanel, BayesFit, HealthCard],
) -> None:
    """At Meridian's thresholds, and each return's posterior median inside its prior's central
    98 %."""
    _, fit, card = well
    assert card.flags(**MERIDIAN, tail=0.01) == {
        "rhat": True,
        "ppp": True,
        "negative_baseline": True,
        "consistency[tv]": True,
        "consistency[search]": True,
    }
    assert card.negative_baseline == 0.0
    assert card.totals.shape == (fit.draws["roi"].size // 2,)
    assert card.total == pytest.approx(float(fit.y.sum()), rel=1e-15)


def test_a_missing_confounder_that_moves_the_total_fails_the_ppp() -> None:
    """Under log-normal noise the outage weeks widen the log's scale, and the mean the model reads
    about it overshoots the total: the observed total falls in the replicates' far left tail, its
    p-value more than three Monte Carlo errors below 0.05."""
    panel, spec = world(1, outage=OUTAGE)
    card = health(_fit(panel, spec), panel, NATIONAL, seed=2)
    assert card.ppp + 3.0 * card.ppp_se < 0.05
    assert np.mean(card.totals <= card.total) < 0.025
    assert card.flags(**MERIDIAN)["ppp"] is False


def test_a_level_below_zero_is_a_negative_baseline_and_one_above_is_not(
    well: tuple[MediaPanel, BayesFit, HealthCard],
) -> None:
    """The level at -300 with the channels' spend ten times the well-specified world's, which
    keeps the mean above 0; the well-specified world's level is 855."""
    panel, spec = world(1, intercept=-300.0, scale=10.0)
    card = health(_fit(panel, spec), panel, NATIONAL, seed=2)
    assert card.negative_baseline > 0.95
    assert np.all(card.negative_baseline_periods > 0.95)
    assert card.flags(**MERIDIAN)["negative_baseline"] is False
    assert well[2].negative_baseline < 0.05


def test_each_draw_s_mean_is_the_model_s(well: tuple[MediaPanel, BayesFit, HealthCard]) -> None:
    """The baseline plus the channels at each draw, averaged over the draws, is the fit's own
    posterior mean of its fitted mean, to 1e-12."""
    panel, fit, _ = well
    data = _read(panel, NATIONAL, fit.spec, np.ones(WEEKS, dtype=bool))
    at = {k: v.reshape(-1, *v.shape[2:]) for k, v in fit.draws.items()}
    media, _ = _media(fit.spec, data, fit.length, at)
    mean = _baseline(fit, data, at) + media.sum(axis=0)
    np.testing.assert_allclose(mean.mean(axis=0), fit.estimated, rtol=1e-12)


def test_each_draw_s_return_is_adr_0016_s(well: tuple[MediaPanel, BayesFit, HealthCard]) -> None:
    """At seven draws evenly spaced over the chains, each channel's return is
    `mixdyn.decisions.returns`' of `BayesFit.channel_draws`' channel, to 1e-12."""
    panel, fit, _ = well
    data = _read(panel, NATIONAL, fit.spec, np.ones(WEEKS, dtype=bool))
    at = {k: v.reshape(-1, *v.shape[2:]) for k, v in fit.draws.items()}
    _, roi = _media(fit.spec, data, fit.length, at)
    keep = _thinned(fit.draws["roi"].shape, 7)
    for i, channels in zip(keep, fit.channel_draws(7), strict=True):
        for j, m in enumerate(fit.spec.media):
            spend = fit.spend[m.channel]
            assert roi[j, i] == pytest.approx(returns(channels[m.channel], spend).roi, rel=1e-12)


def test_the_default_priors_put_the_media_share_inside_zero_and_one() -> None:
    """On the well-specified world, whose spend is 8 % to 9 % of its revenue, the media share's
    90 % prior interval is inside (0, 1), and each channel's return's has its median near the
    stated prior's, `exp(0.2)`, its carryover past the window the difference."""
    panel, spec = world(1)
    assert 0.08 < panel.spend.sum() / panel.kpi.sum() < 0.09
    prior = prior_predictive(panel, NATIONAL, spec, draws=4000, seed=0)
    low, high = prior.quantiles((0.05, 0.95))["share"]
    assert 0.0 < low < high < 1.0
    for channel in ("tv", "search"):
        assert prior.quantiles((0.5,))[channel][0] == pytest.approx(math.exp(0.2), rel=0.15)


def test_the_prior_s_coefficient_is_set_by_its_return() -> None:
    """A return counted over the window contributes it times the window's spend, so the share is
    the stated returns' sum over the KPI's, exactly; a carried return is the prior's draw itself."""
    panel, spec = world(1)
    rng = np.random.default_rng(0)
    stated = np.exp(rng.normal(0.2, 0.9, (500, 2)))
    prior = prior_predictive(panel, NATIONAL, spec, draws=500, seed=0)
    spend = panel.spend[0].sum(axis=0)
    np.testing.assert_allclose(prior.share, stated @ spend / panel.kpi.sum(), rtol=1e-12)
    carried = dataclasses.replace(
        spec, media=tuple(BayesMedia(m.channel, RoiPrior(counts="carried")) for m in spec.media)
    )
    prior = prior_predictive(panel, NATIONAL, carried, draws=500, seed=0)
    for j, channel in enumerate(("tv", "search")):
        np.testing.assert_allclose(prior.roi[channel], stated[:, j], rtol=1e-12)


def test_the_ppp_is_the_smaller_tail_doubled() -> None:
    totals = np.arange(1.0, 101.0)
    assert posterior_predictive_p(totals, 5.0) == (0.1, 2.0 * math.sqrt(0.05 * 0.95 / 100))
    assert posterior_predictive_p(totals, 96.0)[0] == pytest.approx(0.1, rel=1e-15)
    assert posterior_predictive_p(totals, 50.5)[0] == 1.0
    assert posterior_predictive_p(totals, 0.0) == (0.0, 0.0)


def test_the_ppp_s_standard_error_is_its_spread_over_replicates() -> None:
    """Over 400 sets of 500 normal replicates, the p-value of the 95 % quantile spreads as its
    standard error at the true tail, `2 sqrt(0.05 0.95 / 500)`, to four standard errors of a
    standard deviation read from 400 sets, `4 / sqrt(2 x 399)`."""
    rng = np.random.default_rng(7)
    ps = [posterior_predictive_p(rng.standard_normal(500), 1.6448536)[0] for _ in range(400)]
    assert np.std(ps, ddof=1) == pytest.approx(
        2.0 * math.sqrt(0.05 * 0.95 / 500), rel=4.0 / math.sqrt(2.0 * 399), abs=0.0
    )


def test_a_baseline_is_negative_where_any_period_is() -> None:
    baseline = np.array([[1.0, 2.0, 3.0], [1.0, -0.1, 3.0], [-1.0, -1.0, 2.0], [0.0, 0.0, 0.0]])
    anywhere, periods = negative_baseline(baseline)
    assert anywhere == 0.5
    np.testing.assert_array_equal(periods, [0.25, 0.5, 0.0])


def test_a_plan_s_spend_outside_the_logged_range_is_its_share_of_the_plan() -> None:
    """Two channels logged at 10 to 30 and 0 to 5 a period. A plan of 20 and 5 is inside both; one
    of 40 and 5 is 10 past the first's most, 10 of its 45; one of 4 and 1 is 6 short of the
    first's least, 6 of its 5, a share past 1. Over two periods the distances are summed over both
    before they are divided."""
    history = np.array([[10.0, 0.0], [30.0, 5.0], [20.0, 2.0]])
    share, channels = out_of_range(history, np.array([20.0, 5.0]))
    assert share == 0.0
    np.testing.assert_array_equal(channels, [0.0, 0.0])
    share, channels = out_of_range(history, np.array([40.0, 5.0]))
    assert share == pytest.approx(10.0 / 45.0, rel=1e-15, abs=0.0)
    np.testing.assert_allclose(channels, [10.0 / 45.0, 0.0], rtol=1e-15, atol=0.0)
    share, channels = out_of_range(history, np.array([4.0, 1.0]))
    assert share == pytest.approx(6.0 / 5.0, rel=1e-15, abs=0.0)
    np.testing.assert_allclose(channels, [6.0 / 5.0, 0.0], rtol=1e-15, atol=0.0)
    share, channels = out_of_range(history, np.array([[40.0, 5.0], [20.0, 7.0]]))
    assert share == pytest.approx(12.0 / 72.0, rel=1e-15, abs=0.0)
    np.testing.assert_allclose(channels, [10.0 / 72.0, 2.0 / 72.0], rtol=1e-15, atol=0.0)
    assert share == pytest.approx(float(channels.sum()), rel=1e-15, abs=0.0)


def test_the_out_of_range_distance_is_the_same_on_both_sides() -> None:
    """Two channels each logged at 10 to 30: a plan of 4 and 36 is 6 short of the first's least
    and 6 past the second's most, and reads the same 6 for each, not the 4 the first spends;
    swapping the sides moves neither part."""
    history = np.array([[10.0, 10.0], [30.0, 30.0]])
    share, channels = out_of_range(history, np.array([4.0, 36.0]))
    assert share == pytest.approx(12.0 / 40.0, rel=1e-15, abs=0.0)
    np.testing.assert_array_equal(channels, [6.0 / 40.0, 6.0 / 40.0])
    swapped, parts = out_of_range(history, np.array([36.0, 4.0]))
    assert swapped == share
    np.testing.assert_array_equal(parts, channels)


@pytest.mark.parametrize(
    ("history", "plan", "match"),
    [
        (np.ones((0, 2)), np.ones(2), "need a history"),
        (np.ones((3, 2)), np.ones(3), "2 channels and the plan 3"),
        (np.ones((3, 2)), np.array([1.0, -1.0]), "plan's spend must be finite"),
        (np.array([[1.0, np.nan]]), np.ones(2), "history's spend must be finite"),
        (np.ones((3, 2)), np.zeros(2), "spends nothing"),
    ],
)
def test_the_out_of_range_share_refuses_what_it_cannot_read(
    history: np.ndarray, plan: np.ndarray, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        out_of_range(history, plan)


def test_the_shift_the_width_and_the_consistency() -> None:
    prior = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    posterior = np.array([3.0, 3.5, 4.0])
    shift, width, consistency = prior_shift(prior, posterior)
    sd = math.sqrt(2.5)
    assert shift == pytest.approx((3.5 - 2.0) / sd, rel=1e-15)
    assert width == pytest.approx(0.5 / sd, rel=1e-15)
    assert consistency == 0.8


def test_flags_read_only_the_thresholds_given() -> None:
    card = HealthCard(
        max_rhat=1.2,
        ppp=0.05,
        ppp_se=0.01,
        total=1.0,
        totals=np.zeros(3),
        negative_baseline=0.2,
        negative_baseline_periods=np.zeros(2),
        shift={"tv": 0.0},
        width={"tv": 0.5},
        consistency={"tv": 0.995},
    )
    assert card.flags() == {}
    assert card.flags(**MERIDIAN) == {"rhat": False, "ppp": True, "negative_baseline": False}
    assert card.flags(max_width=0.5, tail=0.005) == {"width[tv]": False, "consistency[tv]": True}
    assert card.flags(tail=0.01) == {"consistency[tv]": False}


def test_a_fit_read_against_another_panel_is_refused(
    well: tuple[MediaPanel, BayesFit, HealthCard],
) -> None:
    panel, fit, _ = well
    later = dataclasses.replace(panel, dates=panel.dates + np.timedelta64(7, "D"))
    with pytest.raises(ValueError, match="not a run of the panel's dates"):
        health(fit, later, NATIONAL, seed=0)
    with pytest.raises(ValueError, match="no geo"):
        health(fit, panel, Geo("elsewhere"), seed=0)
    with pytest.raises(ValueError, match="two prior draws"):
        health(fit, panel, NATIONAL, seed=0, prior_draws=1)


def _structure(parameters: Mapping[str, np.ndarray], geos: int, channels: int) -> dict[str, tuple]:
    """Each parameter's axes past the draws', each named `geo` or `channel` by its length."""
    names = {geos: "geo", channels: "channel"}
    return {k: tuple(names.get(n, n) for n in v.shape[1:]) for k, v in parameters.items()}


class _OneGeoLogNormalSigma(BayesModel):
    """The one-geo model with its noise scale's half-normal prior swapped for `LogNormal(0, 1)`."""

    def _log_density(self, theta: jax.Array) -> jax.Array:
        sigma = self._constrained(theta)[0]["sigma"]
        swap = jnp.sum(_log_normal(sigma, 0.0, 1.0) - jstats.norm.logpdf(sigma))
        return super()._log_density(theta) + swap


class _GeoLogNormalSpread(BayesGeoModel):
    """The model over geos with each channel's spread's half-normal prior swapped for a
    log-normal of the same scale's log."""

    def _log_density(self, theta: jax.Array) -> jax.Array:
        spread = self._constrained(theta)[0]["spread"]
        scale = self.spec.spread
        swap = _log_normal(spread, math.log(scale), 1.0) - jstats.norm.logpdf(spread, 0.0, scale)
        return super()._log_density(theta) + jnp.sum(swap)


def _swapped(
    model: BayesModel | BayesGeoModel, kind: type[BayesModel] | type[BayesGeoModel]
) -> BayesModel | BayesGeoModel:
    return kind(**{f.name: getattr(model, f.name) for f in dataclasses.fields(model) if f.init})


def _drawn(model: BayesModel | BayesGeoModel) -> dict[str, np.ndarray]:
    """The parameters at a short run of the sampler on the model's own density."""
    initial = _initial(model, _UNIFORM_START, 1, 3)
    positions = compiled.sample(model._log_density, initial, seed=3, warmup=30, draws=10).positions
    return model.parameters(positions.reshape(-1, model.dimension))


def _moved(model: BayesModel | BayesGeoModel, other: BayesModel | BayesGeoModel) -> bool:
    """Whether two models' densities differ by more than a constant at two points."""
    a, b = _initial(model, _UNIFORM_START, 2, 0)
    return not math.isclose(
        model.log_density(a)[0] - model.log_density(b)[0],
        other.log_density(a)[0] - other.log_density(b)[0],
        abs_tol=1e-9,
    )


@pytest.mark.parametrize("engine", ["one geo", "geos", "nested"])
def test_a_prior_s_family_moves_no_pooling(engine: str) -> None:
    """A scale's prior swapped from a half-normal to a log-normal moves the posterior but not
    which parameters are shared, per geo or per channel: the sampler's coordinates and the
    parameters drawn on the swapped density are the model's."""
    if engine == "geos":
        panel, spec = geo_world("normal")
        models: list[tuple[BayesModel | BayesGeoModel, BayesModel | BayesGeoModel]] = []
        model = bayes_geo_model(panel, spec)
        models.append((model, _swapped(model, _GeoLogNormalSpread)))
        geos = len(panel.geos)
        assert geos != len(spec.media)
        expected = {
            "roi": ("channel",),
            "spread": ("channel",),
            "coefficient": ("geo", "channel"),
            "base": ("geo",),
            "sigma": (),
        }
    else:
        panel, spec = world(1)
        if engine == "one geo":
            stages = [bayes_model(panel, NATIONAL, spec)]
        else:
            nested = nested_model(
                panel,
                NATIONAL,
                NestedSpec(
                    "visits",
                    BayesSpec((BayesMedia("tv"),)),
                    dataclasses.replace(spec, controls=(Control("visits", lower=0.0),)),
                ),
            )
            stages = [nested.stage_one, nested.stage_two]
        models = [(m, _swapped(m, _OneGeoLogNormalSigma)) for m in stages]
        geos = 1
        expected = {"roi": ("channel",), "coefficient": ("channel",), "sigma": ()}
    for model, swapped in models:
        assert _moved(model, swapped)
        assert swapped.coordinates == model.coordinates
        channels = len(model.spec.media)
        before = _structure(_drawn(model), geos, channels)
        after = _structure(_drawn(swapped), geos, channels)
        assert after == before
        assert {k: before[k] for k in expected} == expected


def _indexed(coordinates: tuple[str, ...]) -> list[tuple[str, str]]:
    """Each coordinate's parameter and index, a transform's `log_` taken off its name."""
    out = []
    for name in coordinates:
        stem, _, index = name.partition("[")
        out.append((stem.removeprefix("log_"), index))
    return out


@pytest.mark.parametrize(
    "options",
    [
        {"noise_prior": "log-normal"},
        {"base_prior": "normal"},
        {"noise_prior": "log-normal", "base_prior": "normal"},
    ],
)
def test_the_geo_model_s_prior_options_move_no_pooling(options: dict[str, str]) -> None:
    """The model over geos' own prior options, the noise scale's family and the bases', move the
    posterior and keep every parameter's index set: the coordinates by parameter and index, a
    base sampled as it is rather than on its log, and the axes of every parameter drawn."""
    panel, spec = geo_world("normal")
    model = bayes_geo_model(panel, spec)
    other = bayes_geo_model(panel, dataclasses.replace(spec, **options))
    assert _moved(model, other)
    assert _indexed(other.coordinates) == _indexed(model.coordinates)
    geos, channels = len(panel.geos), len(spec.media)
    assert _structure(_drawn(other), geos, channels) == _structure(_drawn(model), geos, channels)
