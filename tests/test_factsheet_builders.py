"""The factsheets of the model over geos, the DLM and the ridge (`mixdyn.api`), each read off its
own fit on a tiny world.

Each fills `FACTSHEET`'s one schema: every field is on its page, in its JSON document and read back
the same, as `tests/test_factsheet.py` reads the one-geo model's. Each states its own returns: the
geo model's over its draws, the DLM's over its decomposition's draws, the ridge's as the point it
is, with a limitation that says so.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from chc.dlm import Prior as DlmPrior

from mixdyn import (
    NATIONAL,
    Allotment,
    Baseline,
    Cadence,
    Channel,
    Geo,
    Lever,
    LeverKind,
    MediaPanel,
    Revenue,
)
from mixdyn.api import (
    FACTSHEET,
    Factsheet,
    PowerSummary,
    Unfilled,
    dlm_factsheet,
    dumps,
    geo_factsheet,
    loads,
    render,
    ridge_factsheet,
)
from mixdyn.api.serialize import load, save
from mixdyn.calibration import LiftEstimate
from mixdyn.models import (
    DlmDecomposition,
    DlmFit,
    DlmSpec,
    Effect,
    Trend,
    decompose_dlm,
    fit_dlm,
    vague_prior,
)
from mixdyn.models.bayes_geo import BayesGeoFit, BayesGeoSpec, fit_bayes_geo
from mixdyn.models.bayes_mmm import BayesMedia, RoiPrior, TanhCurve
from mixdyn.models.ridge_mmm import (
    Control,
    FourierSeason,
    LinearTrend,
    Media,
    RidgeFit,
    RidgeSpec,
    fit_ridge,
)
from mixdyn.transforms import Geometric, Hill
from mixdyn.transforms.saturation import TanhShape, Transform
from ridge_worlds import world_and_tests
from test_factsheet import _in_order, _keys, _labels

WEEKS = 40
DATES = np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(WEEKS)
PRICE = {"price": Lever(LeverKind.PRICE, Baseline.MAX)}


def one_geo() -> MediaPanel:
    """Forty weeks of TV and search spend and a price of 10, or 8 in a promotion week: the KPI is
    500 plus 2 a unit of TV's spend carried over at 0.4, plus 1 a unit of search, less 30 a unit
    of price, plus noise of SD 10, the price a lever against its most."""
    rng = np.random.default_rng(20261107)
    tv = rng.gamma(2.0, 40.0, WEEKS) * (rng.random(WEEKS) < 0.8)
    search = 60.0 + rng.gamma(3.0, 5.0, WEEKS)
    price = np.where(rng.random(WEEKS) < 0.25, 8.0, 10.0)
    carried = np.convolve(tv, 0.4 ** np.arange(WEEKS))[:WEEKS]
    kpi = 500.0 + 2.0 * carried + search - 30.0 * price + rng.normal(0.0, 10.0, WEEKS)
    return MediaPanel(
        dates=DATES,
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=(Channel("tv"), Channel("search")),
        controls=("price",),
        kpi=kpi[None],
        kpi_unit=Revenue(),
        spend=np.column_stack([tv, search])[None],
        currency="EUR",
        control_values=price[None, :, None],
        filled=np.zeros((1, WEEKS), dtype=bool),
        levers=PRICE,
    )


def three_geos() -> MediaPanel:
    """Three geos of 30 weeks and 1, 2 and 3 million people, TV bought nationally and split by
    population and search bought by geo, and a promotion's depth: each geo's KPI per head is 5,
    plus 0.02 a unit of TV and 0.01 a unit of search spend per head, plus 0.5 a unit of depth,
    plus noise; the north's fifth week has no row, and TV spends nothing that week."""
    rng = np.random.default_rng(20261108)
    weeks, people = 30, np.array([1.0, 2.0, 3.0])
    tv = rng.gamma(2.0, 60.0, weeks)[None, :] * (people / people.sum())[:, None]
    search = rng.gamma(2.0, 10.0, (3, weeks)) * people[:, None]
    depth = rng.random((3, weeks))
    tv[:, 4] = 0.0
    kpi = people[:, None] * (
        5.0 + 0.02 * tv / people[:, None] + 0.01 * search / people[:, None] + 0.5 * depth
    )
    kpi = kpi + rng.normal(0.0, 0.05, kpi.shape) * people[:, None]
    filled = np.zeros((3, weeks), dtype=bool)
    filled[0, 4] = True
    kpi[0, 4] = depth[0, 4] = np.nan
    search[0, 4] = 0.0
    return MediaPanel(
        dates=np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(weeks),
        cadence=Cadence.WEEK,
        geos=(Geo("north"), Geo("middle"), Geo("south")),
        channels=(Channel("tv"), Channel("search")),
        controls=("depth",),
        kpi=kpi,
        kpi_unit=Revenue(),
        spend=np.stack([tv, search], axis=-1),
        currency="EUR",
        control_values=depth[:, :, None],
        filled=filled,
        population=people,
        national={Channel("tv"): Allotment.POPULATION},
    )


@pytest.fixture(scope="module")
def geo_fit() -> tuple[MediaPanel, BayesGeoFit]:
    panel = three_geos()
    spec = BayesGeoSpec(
        (BayesMedia("tv"), BayesMedia("search", RoiPrior(0.0, 0.5))),
        (Control("depth", lower=0.0),),
        trend=LinearTrend(0),
        saturation=TanhCurve(),
    )
    return panel, fit_bayes_geo(panel, spec, seed=3, chains=2, warmup=150, draws=150)


DLM = DlmSpec(
    (
        Effect("tv", 1.0, sign="positive", transform=Transform(0.4, 80.0, TanhShape(), WEEKS)),
        Effect("search", 0.98),
        Effect("price", 1.0),
    ),
    Trend(1, 0.95),
)
TEST = LiftEstimate(
    np.r_[np.full(4, 0.0), np.full(2, 50.0)], np.full(6, 50.0), 6, -150.0, 40.0, end=DATES[30]
)


@pytest.fixture(scope="module")
def dlm_fit() -> tuple[MediaPanel, DlmFit, DlmDecomposition]:
    panel = one_geo()
    fit = fit_dlm(panel, NATIONAL, DLM, draws=200, seed=5, estimates={"tv": [TEST]})
    return panel, fit, decompose_dlm(fit, draws=400, seed=6)


RIDGE = RidgeSpec(
    (Media("tv", Geometric(0.4), Hill(1.0, 0.6)), Media("search", Geometric(0.1), Hill(1.2, 0.5))),
    (Control("price", upper=0.0),),
    0.01,
    LinearTrend(1),
    FourierSeason(13.0, 1),
)


def _checked(sheet: Factsheet, tmp_path: Path) -> None:
    """Every field of `sheet` is on its page in order under its name, is a key of its JSON
    document, and reads back the same from JSON and from a file."""
    page = render(sheet)
    assert page.splitlines()[0] == "factsheet"
    assert _in_order(page, _labels(sheet, 0)) == []
    text = dumps(sheet, FACTSHEET)
    assert _keys(sheet, json.loads(text)["data"], "data") == []
    assert loads(text, FACTSHEET) == sheet
    save(sheet, tmp_path / "sheet")
    assert load(tmp_path / "sheet") == sheet


def test_the_geo_model_s_factsheet_is_read_off_its_fit(
    geo_fit: tuple[MediaPanel, BayesGeoFit], tmp_path: Path
) -> None:
    panel, fit = geo_fit
    sheet = geo_factsheet(
        panel,
        fit,
        level=0.8,
        origins={"tv": "the agency's national invoices"},
        power=PowerSummary(12, 0.2, 0.6),
    )
    _checked(sheet, tmp_path)
    assert sheet.geos == ("north", "middle", "south")
    assert (sheet.first, sheet.last) == (
        DATES[0].item(),
        (DATES[0] + np.timedelta64(29 * 7, "D")).item(),
    )
    assert sheet.channels == ("tv", "search")
    for j, r in enumerate(sheet.returns):
        draws = fit.draws["roi"][..., j].reshape(-1)
        assert (r.mean, r.level) == (draws.mean(), 0.8)
        assert [r.low, r.high] == np.quantile(draws, [(1 - 0.8) / 2, (1 + 0.8) / 2]).tolist()
    sources = {s.column: s for s in sheet.sources}
    assert sources["kpi"].transforms[:2] == (
        "1 of the geos' weeks with no row in the input read as unobserved",
        "divided by each geo's population",
    )
    assert sources["tv"].transforms[0] == "bought nationally and split over the geos by population"
    assert sources["tv"].origin == "the agency's national invoices"
    assert sources["depth"].transforms[-1] == "its coefficient held at or above 0"
    priors = {p.parameter: p for p in sheet.priors}
    assert "Meridian" in priors["roi[tv]"].provenance
    assert priors["roi[search]"].provenance == "stated by the caller"
    assert {"spread", "base", "calendar", "control", "sigma"} <= set(priors)
    assert not {"slope", "nu"} & set(priors)
    assert any(line.startswith("tv bought nationally") for line in sheet.limitations)
    assert "the lift tests' power was not audited" not in sheet.limitations
    with pytest.raises(ValueError, match="not of this panel"):
        geo_factsheet(one_geo(), fit)
    with pytest.raises(ValueError, match="origins name columns"):
        geo_factsheet(panel, fit, origins={"price": "?"})


def test_the_dlm_s_factsheet_reads_its_returns_off_the_decomposition(
    dlm_fit: tuple[MediaPanel, DlmFit, DlmDecomposition], tmp_path: Path
) -> None:
    panel, fit, drawn = dlm_fit
    sheet = dlm_factsheet(panel, NATIONAL, fit, drawn, estimates={"tv": [TEST]})
    _checked(sheet, tmp_path)
    assert (sheet.geos, sheet.channels) == (("national",), ("tv", "search"))
    window = drawn.window(DATES[0], DATES[-1]).contributions
    for r in sheet.returns:
        spend = panel.spend[0, :, ("tv", "search").index(r.channel)].sum()
        draws = window[r.channel] / spend
        assert r.mean == draws.mean()
        assert [r.low, r.high] == np.quantile(draws, [(1 - 0.9) / 2, (1 + 0.9) / 2]).tolist()
    (test,) = sheet.tests
    assert (test.channel, test.change, test.periods) == ("tv", "withheld", 4)
    sources = {s.column: s for s in sheet.sources}
    assert sources["tv"].transforms == (
        "carried over by a normalised geometric kernel of retention 0.4 over 40 periods",
        "saturated by tanh, half its ceiling at 80 EUR",
        "its coefficient held still, a discount of 1",
        "its coefficient held at or above 0, the posterior truncated to it",
    )
    level = panel.counterfactual("price")
    assert sources["price"].transforms[0] == (
        f"a price lever, read against its counterfactual level {level:g}"
    )
    assert sources["search"].transforms[-1] == (
        "its coefficient a random walk of discount 0.98, held while its column is 0"
    )
    priors = {p.parameter: p for p in sheet.priors}
    assert priors["level"].provenance.startswith("the engine's default (vague_prior)")
    assert set(priors) >= {"coefficient[tv]", "variance", "evolution[level]", "evolution[search]"}
    assert priors["evolution[tv]"].distribution == "discount 1: held still"
    lines = sheet.limitations
    assert "no lift test calibrates search" in lines[0]
    assert any("held at its transform's values" in line for line in lines)
    assert any("the coefficients of search walk" in line for line in lines)
    mixed = fit.signed is not None and fit.signed.mixed
    assert mixed != any("have not mixed" in line for line in lines)


def test_the_dlm_s_factsheet_names_a_prior_the_caller_states(
    dlm_fit: tuple[MediaPanel, DlmFit, DlmDecomposition],
) -> None:
    panel, _, _ = dlm_fit
    spec = DlmSpec((Effect("search", 1.0), Effect("price", 1.0)), Trend(1, 0.95))
    vague = vague_prior(panel, NATIONAL, spec)
    prior = DlmPrior(vague.mean + 1.0, vague.covariance, vague.scale, 4.0)
    fit = fit_dlm(panel, NATIONAL, spec, prior)
    sheet = dlm_factsheet(panel, NATIONAL, fit, decompose_dlm(fit, draws=50, seed=1), level=0.5)
    priors = {p.parameter: p for p in sheet.priors}
    assert {p.provenance for p in sheet.priors if "evolution" not in p.parameter} == {
        "stated by the caller"
    }
    assert priors["variance"].distribution.endswith("from 4 degrees of freedom")
    assert not any("transform" in line for line in sheet.limitations)


def test_the_dlm_s_factsheet_refuses_another_fit_s_draws(
    dlm_fit: tuple[MediaPanel, DlmFit, DlmDecomposition],
) -> None:
    panel, fit, drawn = dlm_fit
    other = fit_dlm(panel, NATIONAL, DlmSpec((Effect("search", 1.0),), Trend(1, 0.95)))
    with pytest.raises(ValueError, match="not of this fit"):
        dlm_factsheet(panel, NATIONAL, fit, decompose_dlm(other, draws=10, seed=1))
    with pytest.raises(ValueError, match="no geo 'south'"):
        dlm_factsheet(panel, Geo("south"), fit, drawn)
    with pytest.raises(ValueError, match=r"level is inside \(0, 1\)"):
        dlm_factsheet(panel, NATIONAL, fit, drawn, level=1.0)
    with pytest.raises(ValueError, match="lift tests of"):
        dlm_factsheet(panel, NATIONAL, fit, drawn, estimates={"price": [TEST]})


def test_the_ridge_s_factsheet_states_its_point_as_one(tmp_path: Path) -> None:
    panel = one_geo()
    fit: RidgeFit = fit_ridge(panel, NATIONAL, RIDGE)
    sheet = ridge_factsheet(panel, NATIONAL, fit, origins={"season": "the shop's calendar"})
    _checked(sheet, tmp_path)
    for r in sheet.returns:
        point = float(np.sum(fit.contributions[r.channel])) / fit.spend[r.channel]
        assert (r.mean, r.low, r.high, r.level) == (point, point, point, 0.0)
    sources = {s.column: s for s in sheet.sources}
    level = panel.counterfactual("price")
    assert sources["price"].transforms == (
        f"a price lever, read as its departure from its counterfactual level {level:g}",
        "its coefficient held at or below 0",
    )
    half = np.format_float_positional(
        fit.half_saturation["tv"], precision=4, unique=False, fractional=False, trim="-"
    )
    assert sources["tv"].transforms[1] == (
        f"saturated by a Hill of alpha 1, half its ceiling at {half} EUR, gamma 0.6 of the most "
        "of the window's adstocked values"
    )
    assert sources["season"].origin == "the shop's calendar"
    assert [p.parameter for p in sheet.priors] == ["coefficients"]
    assert sheet.tests == ()
    assert any(line.startswith("the ridge fit is a point") for line in sheet.limitations)
    assert sheet.limitations[0] == (
        "no lift test calibrates tv, search: the return rests on the observational data and the "
        "penalty"
    )
    assert isinstance(sheet.holdout, Unfilled)


def test_a_calibrated_ridge_s_factsheet_names_its_anchors_and_tests(tmp_path: Path) -> None:
    panel, media, _, tests, _ = world_and_tests(20.0)
    fit = fit_ridge(panel, Geo("national"), RidgeSpec(tuple(media), (), 1e-3), tests=tests)
    sheet = ridge_factsheet(panel, Geo("national"), fit)
    _checked(sheet, tmp_path)
    assert fit.calibration is not None
    anchors = {p.parameter: p for p in sheet.priors if p.parameter.startswith("anchor")}
    assert set(anchors) == {"anchor[c0]", "anchor[c2]"}
    assert all(p.provenance == "the fit's geo tests, read alone" for p in anchors.values())
    assert sheet.limitations[0].startswith("no lift test calibrates c1:")
    assert any("geo tests of c0, c2" in line for line in sheet.limitations)
    sources = {s.column: s for s in sheet.sources}
    assert (
        sources["c1"].transforms[0]
        == "carried over by a Weibull adstock of shape 1.5 and scale 3 periods"
    )
    with pytest.raises(ValueError, match="not of 'national'"):
        ridge_factsheet(one_geo(), NATIONAL, fit)


def test_every_builder_s_factsheet_is_a_factsheet_of_the_one_schema(
    geo_fit: tuple[MediaPanel, BayesGeoFit],
    dlm_fit: tuple[MediaPanel, DlmFit, DlmDecomposition],
) -> None:
    """The four builders write one schema: the same fields, each of the declared type."""
    geo_panel, geo = geo_fit
    panel, fit, drawn = dlm_fit
    sheets: list[Any] = [
        geo_factsheet(geo_panel, geo),
        dlm_factsheet(panel, NATIONAL, fit, drawn),
        ridge_factsheet(panel, NATIONAL, fit_ridge(panel, NATIONAL, RIDGE)),
    ]
    names = [f.name for f in dataclasses.fields(Factsheet)]
    for sheet in sheets:
        assert type(sheet) is Factsheet
        assert [f.name for f in dataclasses.fields(sheet)] == names
        assert json.loads(dumps(sheet, FACTSHEET))["schema"] == "mixdyn.factsheet"
