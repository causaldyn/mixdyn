"""Levers (ADR 0068): a price, promotion or organic column the panel declares a lever is measured
against its counterfactual level by the ridge, the one-geo Bayesian model and the DLM, appears in
their decompositions and never in a plan, and is refused by the engines that would read it against
0."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mixdyn import (
    NATIONAL,
    Baseline,
    Cadence,
    Channel,
    Combine,
    Geo,
    Lever,
    LeverKind,
    MediaPanel,
    PanelError,
    Revenue,
    aggregate,
    read_panel,
)
from mixdyn.api.serialize import load, save
from mixdyn.data import IssueCode
from mixdyn.decisions import Range, plan_posterior, readings
from mixdyn.models import (
    DlmFit,
    DlmSpec,
    Effect,
    GeoDlmSpec,
    Trend,
    decompose_dlm,
    fit_dlm,
    fit_geo_dlm,
    forecast_dlm,
)
from mixdyn.models.bayes_geo import BayesGeoSpec, fit_bayes_geo
from mixdyn.models.bayes_mmm import (
    BayesFit,
    BayesMedia,
    BayesSpec,
    TanhCurve,
    _summary,
    bayes_model,
    fit_bayes,
    lever_contributions,
)
from mixdyn.models.ridge_mmm import Control, Media, RidgeSpec, fit_ridge
from mixdyn.transforms import Geometric, Hill

WEEKS = 80
LEVERS = {
    "price": Lever(LeverKind.PRICE, Baseline.MAX),
    "newsletter": Lever(LeverKind.ORGANIC, 0.0),
}


def world(levers: dict[str, Lever] | None = None, seed: int = 3) -> MediaPanel:
    """80 weeks of TV and search spend, a price of 10 or 8 in a promotion week, and newsletters
    sent: the KPI is 600 plus 1.5 a unit of TV and of search spend, less 40 a unit of price, plus
    0.4 a newsletter, plus normal noise of SD 20."""
    rng = np.random.default_rng(seed)
    tv = rng.gamma(2.0, 50.0, WEEKS) * (rng.random(WEEKS) < 0.7)
    search = 80.0 + rng.gamma(4.0, 5.0, WEEKS)
    price = np.where(rng.random(WEEKS) < 0.25, 8.0, 10.0) + rng.normal(0.0, 0.2, WEEKS)
    newsletter = rng.poisson(200.0, WEEKS).astype(float)
    kpi = 600.0 + 1.5 * (tv + search) - 40.0 * price + 0.4 * newsletter + rng.normal(0, 20, WEEKS)
    return MediaPanel(
        dates=np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(WEEKS),
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=(Channel("tv"), Channel("search")),
        controls=("price", "newsletter"),
        kpi=kpi[None],
        kpi_unit=Revenue(),
        spend=np.column_stack([tv, search])[None],
        currency="EUR",
        control_values=np.column_stack([price, newsletter])[None],
        filled=np.zeros((1, WEEKS), dtype=bool),
        levers=levers or {},
    )


def two_geos(levers: dict[str, Lever]) -> MediaPanel:
    """Two geos of 30 weeks, the second's week 5 filled, with a price and a promotion depth."""
    rng = np.random.default_rng(7)
    weeks = 30
    filled = np.zeros((2, weeks), dtype=bool)
    filled[1, 5] = True
    price = 10.0 + rng.normal(0.0, 1.0, (2, weeks))
    price[1, 5] = np.nan
    depth = rng.random((2, weeks))
    depth[1, 5] = np.nan
    kpi = 500.0 + rng.normal(0.0, 10.0, (2, weeks))
    kpi[1, 5] = np.nan
    spend = rng.gamma(2.0, 50.0, (2, weeks, 1))
    spend[1, 5] = 0.0
    return MediaPanel(
        dates=np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(weeks),
        cadence=Cadence.WEEK,
        geos=(Geo("north"), Geo("south")),
        channels=(Channel("tv"),),
        controls=("price", "depth"),
        kpi=kpi,
        kpi_unit=Revenue(),
        spend=spend,
        currency="EUR",
        control_values=np.stack([price, depth], axis=-1),
        filled=filled,
        levers=levers,
    )


@pytest.mark.parametrize(
    ("kind", "baseline"),
    [
        ("price", Baseline.MIN),
        (LeverKind.PRICE, "min"),
        (LeverKind.PRICE, np.nan),
        (LeverKind.PRICE, np.inf),
        (LeverKind.PROMOTION, True),
        (LeverKind.ORGANIC, None),
    ],
)
def test_a_lever_has_a_kind_and_a_finite_or_observed_level(kind: object, baseline: object) -> None:
    with pytest.raises(ValueError, match=r"kind is a LeverKind|baseline is a Baseline"):
        Lever(kind, baseline)  # ty: ignore[invalid-argument-type]


def test_a_lever_is_one_of_the_panel_s_controls() -> None:
    with pytest.raises(ValueError, match="lever 'cost' is not one of the panel's controls"):
        world({"cost": Lever(LeverKind.PRICE)})
    with pytest.raises(ValueError, match="is not a Lever"):
        world({"price": "max"})  # ty: ignore[invalid-argument-type]
    data = {"date": world().dates, "kpi": np.ones(WEEKS), "tv": np.ones(WEEKS)}
    data |= {"price": np.arange(WEEKS, dtype=float)}
    with pytest.raises(PanelError) as raised:
        read_panel(
            data,
            date="date",
            kpi="kpi",
            spend={"tv": "tv"},
            cadence=Cadence.WEEK,
            kpi_unit=Revenue(),
            currency="EUR",
            controls=("price",),
            levers={"cost": Lever(LeverKind.PRICE)},
        )
    assert [i.code for i in raised.value.issues] == [IssueCode.BAD_SPEC]
    panel = read_panel(
        data,
        date="date",
        kpi="kpi",
        spend={"tv": "tv"},
        cadence=Cadence.WEEK,
        kpi_unit=Revenue(),
        currency="EUR",
        controls=("price",),
        levers={"price": Lever(LeverKind.PRICE, Baseline.MAX)},
    )
    assert panel.levers == {"price": Lever(LeverKind.PRICE, Baseline.MAX)}
    assert panel.counterfactual("price") == WEEKS - 1.0


def test_the_counterfactual_is_the_least_the_most_or_the_level_given() -> None:
    panel = two_geos(
        {"price": Lever(LeverKind.PRICE, Baseline.MIN), "depth": Lever(LeverKind.PROMOTION)}
    )
    seen = panel.control_values[~panel.filled]
    assert panel.counterfactual("price") == float(seen[:, 0].min())
    assert panel.counterfactual("depth") == float(seen[:, 1].min())
    assert panel.counterfactual("depth") < panel.control_values[0, :, 1].min()
    most = two_geos({"price": Lever(LeverKind.PRICE, Baseline.MAX)})
    assert most.counterfactual("price") == float(seen[:, 0].max())
    np.testing.assert_array_equal(most.counterfactuals(["depth", "price"]), [0.0, seen[:, 0].max()])
    given = two_geos({"price": Lever(LeverKind.PRICE, 9)})
    assert given.counterfactual("price") == 9.0
    with pytest.raises(ValueError, match="'depth' is not one of the panel's levers"):
        given.counterfactual("depth")


def ridge_spec(levers: bool) -> RidgeSpec:
    controls = (Control("price"), Control("newsletter")) if levers else (Control("price"),)
    media = tuple(Media(c, Geometric(0.3), Hill(1.0, 1.0)) for c in ("tv", "search"))
    return RidgeSpec(media, controls, 1e-4)


def test_the_ridge_reads_a_lever_against_its_counterfactual() -> None:
    plain = fit_ridge(world(), NATIONAL, ridge_spec(levers=True))
    panel = world(LEVERS)
    fit = fit_ridge(panel, NATIONAL, ridge_spec(levers=True))
    price = panel.control_values[0, :, 0]
    level = panel.counterfactual("price")
    assert level == price.max()
    for name in plain.coefficients:
        assert fit.coefficients[name] == pytest.approx(plain.coefficients[name], rel=1e-9, abs=0.0)
    np.testing.assert_allclose(fit.estimated, plain.estimated, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(
        fit.contributions["price"], fit.coefficients["price"] * (price - level), rtol=1e-12
    )
    assert np.all(fit.contributions["price"] >= 0.0)
    shifted = plain.intercept + plain.coefficients["price"] * level
    assert fit.intercept == pytest.approx(shifted, rel=1e-9, abs=0.0)
    np.testing.assert_allclose(
        fit.contributions["newsletter"], plain.contributions["newsletter"], rtol=1e-9, atol=0.0
    )
    total = fit.intercept + np.sum(list(fit.contributions.values()), axis=0)
    np.testing.assert_allclose(total, fit.estimated, rtol=1e-12, atol=0.0)


def test_the_bayesian_decomposition_reads_a_lever_against_its_counterfactual() -> None:
    spec = BayesSpec(
        (BayesMedia("tv"), BayesMedia("search")),
        (Control("price"), Control("newsletter")),
        saturation=TanhCurve(),
    )
    panel = world(LEVERS)
    model = bayes_model(panel, NATIONAL, spec)
    plain = bayes_model(world(), NATIONAL, spec)
    theta = 0.3 * np.random.default_rng(1).normal(size=(6, model.dimension))
    raw, parts, _ = _summary(model, theta)
    raw_plain, parts_plain, _ = _summary(plain, theta)
    for name, draws in raw.items():
        np.testing.assert_array_equal(draws, raw_plain[name])
    for name in ("tv", "search", "newsletter"):
        np.testing.assert_array_equal(parts[name], parts_plain[name])
    level = panel.counterfactual("price")
    price = panel.control_values[0, :, 0]
    coefficient = raw["control"][:, 0]
    np.testing.assert_allclose(
        parts["price"], coefficient.mean() * (price - level), rtol=1e-9, atol=0.0
    )
    np.testing.assert_allclose(
        parts["intercept"], parts_plain["intercept"] + coefficient.mean() * level, rtol=1e-12
    )
    total = np.sum(list(parts.values()), axis=0)
    np.testing.assert_allclose(total, np.sum(list(parts_plain.values()), axis=0), rtol=1e-12)


@pytest.fixture(scope="module")
def bayes() -> tuple[MediaPanel, BayesFit]:
    panel = world(LEVERS)
    spec = BayesSpec(
        (BayesMedia("tv"), BayesMedia("search")),
        (Control("price"), Control("newsletter", lower=0.0)),
        saturation=TanhCurve(),
    )
    return panel, fit_bayes(panel, NATIONAL, spec, seed=2, chains=2, warmup=200, draws=200)


def test_each_lever_s_contribution_is_read_at_every_draw(
    bayes: tuple[MediaPanel, BayesFit],
) -> None:
    panel, fit = bayes
    draws = lever_contributions(fit, panel, NATIONAL)
    assert set(draws) == {"price", "newsletter"}
    for k, name in enumerate(("price", "newsletter")):
        departure = np.sum(panel.control_values[0, :, k] - panel.counterfactual(name))
        coefficient = fit.draws["control"][..., k].reshape(-1)
        np.testing.assert_array_equal(draws[name], coefficient * departure)
        summed = float(np.sum(fit.contributions[name]))
        assert draws[name].mean() == pytest.approx(summed, rel=1e-9, abs=0.0)
    assert np.all(draws["price"] > 0.0)
    with pytest.raises(ValueError, match="no geo 'south'"):
        lever_contributions(fit, panel, Geo("south"))


def test_an_organic_channel_is_in_the_decomposition_and_never_planned(
    bayes: tuple[MediaPanel, BayesFit],
) -> None:
    panel, fit = bayes
    assert "newsletter" in fit.contributions
    assert set(fit.channels()) == {"tv", "search"}
    reads = readings(fit)
    assert reads.channels == (Channel("tv"), Channel("search"))
    limits = {Channel("tv"): Range(0.0, 300.0), Channel("search"): Range(0.0, 300.0)}
    plan = plan_posterior(panel, NATIONAL, reads, 13 * 200.0, 13, limits, grid=64)
    assert plan.channels == (Channel("tv"), Channel("search"))
    assert plan.spend.sum() == pytest.approx(200.0, rel=1e-9, abs=0.0)
    ridge = fit_ridge(panel, NATIONAL, ridge_spec(levers=True))
    assert "newsletter" in ridge.contributions
    assert set(ridge.channels()) == {"tv", "search"}


def test_the_engines_that_read_a_control_against_0_refuse_a_lever() -> None:
    panel = world(LEVERS)
    with pytest.raises(ValueError, match=r"\['price'\].*a DLM that reads a lever needs a trend"):
        fit_dlm(panel, NATIONAL, DlmSpec((Effect("tv", 1.0), Effect("price", 1.0))))
    fit_dlm(world(), NATIONAL, DlmSpec((Effect("tv", 1.0), Effect("price", 1.0))))
    geos = two_geos({"depth": Lever(LeverKind.PROMOTION)})
    with pytest.raises(ValueError, match=r"the DLM over geos reads a control against 0"):
        fit_geo_dlm(geos, GeoDlmSpec((Effect("tv", 1.0), Effect("depth", 1.0)), Trend(1, 0.95)))
    with pytest.raises(ValueError, match="the Bayesian model over geos reads a control against 0"):
        fit_bayes_geo(geos, BayesGeoSpec((BayesMedia("tv"),), (Control("depth"),)), seed=0)


def test_a_panel_s_levers_survive_aggregation_and_a_file(tmp_path: Path) -> None:
    panel = two_geos({"price": Lever(LeverKind.PRICE, 9.5)})
    kept = aggregate(panel, ["north", "south"], {"price": Combine.SUM, "depth": Combine.SUM})
    assert kept.levers == panel.levers
    save(panel, tmp_path / "panel.mixdyn")
    loaded = load(tmp_path / "panel.mixdyn")
    assert loaded.levers == panel.levers
    assert loaded.counterfactual("price") == 9.5


DLM = DlmSpec(
    (Effect("tv", 1.0), Effect("search", 1.0), Effect("price", 1.0), Effect("newsletter", 0.98)),
    Trend(1, 0.95),
)


def _dlm_fits() -> tuple[MediaPanel, DlmFit, DlmFit]:
    panel = world(LEVERS)
    return panel, fit_dlm(panel, NATIONAL, DLM), fit_dlm(world(), NATIONAL, DLM)


def test_the_dlm_reads_a_lever_against_its_counterfactual() -> None:
    """The states are the undeclared fit's; each lever's contribution is its coefficient times its
    departure from its level, the base takes the coefficient times the level at every period, a
    walking coefficient's too, and the parts still sum to the fit."""
    panel, fit, plain = _dlm_fits()
    np.testing.assert_array_equal(fit.filtered.mean, plain.filtered.mean)
    np.testing.assert_array_equal(fit.smoothed.mean, plain.smoothed.mean)
    assert fit.baselines == {"price": panel.counterfactual("price"), "newsletter": 0.0}
    assert plain.baselines == {}
    shift = np.zeros(WEEKS)
    for k, name in enumerate(("price", "newsletter")):
        coefficient = fit.smoothed.mean[:, 3 + k]
        level = fit.baselines[name]
        departure = panel.control_values[0, :, k] - level
        np.testing.assert_array_equal(fit.contributions[name], coefficient * departure)
        shift += coefficient * level
    assert np.all(fit.contributions["price"] >= 0.0)
    for name in ("tv", "search"):
        np.testing.assert_array_equal(fit.contributions[name], plain.contributions[name])
    assert fit.base is not None
    assert plain.base is not None
    np.testing.assert_allclose(fit.base, plain.base + shift, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(fit.estimated, plain.estimated, rtol=1e-12, atol=0.0)


def test_a_dlm_forecast_reads_a_planned_lever_against_its_counterfactual() -> None:
    panel, fit, plain = _dlm_fits()
    plan = {"tv": np.full(4, 100.0), "search": np.full(4, 90.0)}
    plan |= {"price": np.array([10.0, 8.0, 10.0, 9.0]), "newsletter": np.full(4, 200.0)}
    ahead, before = forecast_dlm(fit, 4, plan), forecast_dlm(plain, 4, plan)
    np.testing.assert_array_equal(ahead.mean, before.mean)
    level = panel.counterfactual("price")
    coefficient = before.contributions["price"] / plan["price"]
    np.testing.assert_allclose(
        ahead.contributions["price"], coefficient * (plan["price"] - level), rtol=1e-12, atol=0.0
    )
    np.testing.assert_array_equal(
        ahead.contributions["newsletter"], before.contributions["newsletter"]
    )
    assert ahead.base is not None
    assert before.base is not None
    np.testing.assert_allclose(ahead.base, before.base + coefficient * level, rtol=1e-12, atol=0.0)


def test_each_draw_of_a_dlm_decomposition_reads_a_lever_against_its_counterfactual(
    tmp_path: Path,
) -> None:
    panel, fit, plain = _dlm_fits()
    drawn = decompose_dlm(fit, draws=50, seed=4).draws
    before = decompose_dlm(plain, draws=50, seed=4).draws
    level = panel.counterfactual("price")
    coefficient = before.contributions["price"] / panel.control_values[0, :, 0]
    np.testing.assert_allclose(
        drawn.contributions["price"],
        coefficient * (panel.control_values[0, :, 0] - level),
        rtol=1e-9,
        atol=1e-9,
    )
    np.testing.assert_array_equal(drawn.contributions["tv"], before.contributions["tv"])
    assert drawn.base is not None
    assert before.base is not None
    np.testing.assert_allclose(drawn.base, before.base + coefficient * level, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(drawn.total(), before.total(), rtol=1e-12, atol=0.0)
    save(fit, tmp_path / "fit.mixdyn")
    assert load(tmp_path / "fit.mixdyn").baselines == fit.baselines


COVERED = 40
"""Worlds the DLM's lever interval is read on."""


def _promoted(seed: int, *, lever: bool) -> tuple[MediaPanel, float]:
    """104 weeks of TV spend and a price of 10, or 8 in a quarter of the weeks, each with noise of
    SD 0.2, and a level that walks by 3 a week from 600: the KPI is the level plus 1.5 a unit of
    TV, less 40 a unit of price, plus noise of SD 20. With the price a lever measured against its
    most, the truth of its contribution is -40 times its summed departure from that most."""
    weeks = 104
    rng = np.random.default_rng(seed)
    tv = rng.gamma(2.0, 50.0, weeks) * (rng.random(weeks) < 0.7)
    price = np.where(rng.random(weeks) < 0.25, 8.0, 10.0) + rng.normal(0.0, 0.2, weeks)
    level = 600.0 + np.cumsum(rng.normal(0.0, 3.0, weeks))
    kpi = level + 1.5 * tv - 40.0 * price + rng.normal(0.0, 20.0, weeks)
    panel = MediaPanel(
        dates=np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(weeks),
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=(Channel("tv"),),
        controls=("price",),
        kpi=kpi[None],
        kpi_unit=Revenue(),
        spend=tv[None, :, None],
        currency="EUR",
        control_values=price[None, :, None],
        filled=np.zeros((1, weeks), dtype=bool),
        levers={"price": Lever(LeverKind.PRICE, Baseline.MAX)} if lever else {},
    )
    return panel, -40.0 * float(np.sum(price - price.max()))


def test_the_dlm_s_lever_interval_covers_its_truth() -> None:
    """Over worlds 0 to 39, the central 90 % interval of the price's contribution over the fit's
    weeks, 1000 draws of `decompose_dlm`, holds its truth in at least 30: a calibrated interval
    holds it in 29 or fewer with chance 0.0015 (PARI/GP, the binomial's sum), and over worlds 0 to
    199 it held it in 182. Read against 0, with the price not declared a lever, it holds the truth
    in none."""
    spec = DlmSpec((Effect("tv", 1.0), Effect("price", 1.0)), Trend(1, 0.95))
    held = {True: 0, False: 0}
    for seed in range(COVERED):
        for lever in held:
            panel, truth = _promoted(seed, lever=lever)
            fit = fit_dlm(panel, NATIONAL, spec)
            draws = decompose_dlm(fit, draws=1000, seed=seed).window(
                panel.dates[0], panel.dates[-1]
            )
            low, high = np.quantile(draws.contributions["price"], [0.05, 0.95])
            held[lever] += int(low <= truth <= high)
    assert held[True] >= 30
    assert held[False] == 0
