"""Calendar events (ADR 0069): the panel holds each event's days inside its calendar and reads them
as a column of the periods they fall in; the DLM, the ridge and the one-geo Bayesian model read an
event by its name as a regressor beside the level and the season, and on a world whose spend rises
with its events the model with them holds the channel's return where the level alone does not."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import stats

from mixdyn import (
    NATIONAL,
    Cadence,
    Channel,
    Combine,
    Geo,
    MediaPanel,
    PanelError,
    Revenue,
    aggregate,
    read_panel,
)
from mixdyn.api.serialize import load, save
from mixdyn.data import IssueCode
from mixdyn.models import DlmSpec, Effect, Season, Trend, fit_dlm
from mixdyn.models.bayes_mmm import BayesMedia, BayesSpec, bayes_model
from mixdyn.models.ridge_mmm import Control, LinearTrend, Media, RidgeSpec, fit_ridge
from mixdyn.transforms import Geometric, Hill, carry_over, half_saturation, saturate
from study_events import EVENTS, RETURN, world

START = np.datetime64("2024-01-01")
"""A Monday."""


def panel(events: dict[str, Any], cadence: Cadence = Cadence.WEEK, periods: int = 10) -> MediaPanel:
    dates = (
        (START.astype("datetime64[M]") + np.arange(periods)).astype("datetime64[D]")
        if cadence is Cadence.MONTH
        else START + np.timedelta64(7 if cadence is Cadence.WEEK else 1, "D") * np.arange(periods)
    )
    rng = np.random.default_rng(0)
    return MediaPanel(
        dates=dates,
        cadence=cadence,
        geos=(Geo("north"), Geo("south")),
        channels=(Channel("tv"),),
        controls=("price",),
        kpi=100.0 + rng.normal(0.0, 1.0, (2, periods)),
        kpi_unit=Revenue(),
        spend=rng.gamma(2.0, 5.0, (2, periods, 1)),
        currency="EUR",
        control_values=rng.normal(10.0, 1.0, (2, periods, 1)),
        filled=np.zeros((2, periods), dtype=bool),
        events=events,
    )


def test_an_event_is_the_periods_its_days_fall_in() -> None:
    weekly = panel({"sale": ["2024-01-04", "2024-01-07", "2024-01-25", "2024-01-04"]})
    np.testing.assert_array_equal(
        weekly.events["sale"],
        np.array(["2024-01-04", "2024-01-07", "2024-01-25"], dtype="datetime64[D]"),
    )
    assert not weekly.events["sale"].flags.writeable
    np.testing.assert_array_equal(weekly.event_column("sale"), [1, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    last = panel({"end": ["2024-03-10"]})
    assert last.event_column("end")[-1] == 1.0
    monthly = panel({"summer": ["2024-07-15", "2024-08-31"]}, Cadence.MONTH, 12)
    np.testing.assert_array_equal(np.flatnonzero(monthly.event_column("summer")), [6, 7])
    daily = panel({"day": ["2024-01-03"]}, Cadence.DAY)
    np.testing.assert_array_equal(np.flatnonzero(daily.event_column("day")), [2])
    with pytest.raises(ValueError, match="'gone' is not one of the panel's events"):
        weekly.event_column("gone")


@pytest.mark.parametrize(
    ("events", "match"),
    [
        ({"sale": ["2023-12-31"]}, "1 days outside the calendar"),
        ({"sale": ["2024-03-11"]}, "1 days outside the calendar"),
        ({"sale": []}, "needs at least one day"),
        ({"sale": ["2024-01-04", "NaT"]}, "needs at least one day"),
        ({"sale": ["soon"]}, "are not dates"),
        ({"tv": ["2024-01-04"]}, "needs a name of its own"),
        ({"price": ["2024-01-04"]}, "needs a name of its own"),
        ({"": ["2024-01-04"]}, "needs a name of its own"),
    ],
)
def test_an_event_s_days_lie_inside_the_calendar_under_a_name_of_its_own(
    events: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        panel(events)


def test_a_model_reads_controls_and_events_by_name() -> None:
    p = panel({"sale": ["2024-01-04"], "fair": ["2024-02-14", "2024-02-15"]})
    some = np.arange(10) >= 3
    for g, geo in enumerate(p.geos):
        both = p.regressors(["fair", "price", "sale"], geo)
        assert both.shape == (10, 3)
        np.testing.assert_array_equal(both[:, 0], p.event_column("fair"))
        np.testing.assert_array_equal(both[:, 1], p.control_values[g, :, 0])
        np.testing.assert_array_equal(both[:, 2], p.event_column("sale"))
        np.testing.assert_array_equal(
            p.regressors(["sale", "price"], geo, some), both[some][:, [2, 1]]
        )
        np.testing.assert_array_equal(p.regressors(["price"], geo), p.control_values[g])
    with pytest.raises(ValueError, match=r"no control or event named \['rain'\]"):
        p.regressors(["price", "rain"], Geo("north"))
    with pytest.raises(ValueError, match="no geo 'east'"):
        p.regressors(["price"], Geo("east"))


def test_reading_holds_events_and_names_what_it_refuses() -> None:
    dates = START + np.timedelta64(7, "D") * np.arange(20)
    data = {"date": dates, "kpi": np.ones(20), "tv": np.ones(20)}

    def read(events: dict[str, Any]) -> MediaPanel:
        return read_panel(
            data,
            date="date",
            kpi="kpi",
            spend={"tv": "tv"},
            cadence=Cadence.WEEK,
            kpi_unit=Revenue(),
            currency="EUR",
            events=events,
        )

    np.testing.assert_array_equal(
        np.flatnonzero(read({"sale": ["2024-02-01"]}).event_column("sale")), [4]
    )
    with pytest.raises(PanelError) as raised:
        read({"sale": ["2025-02-01"], "tv": ["2024-02-01"]})
    assert [i.code for i in raised.value.issues] == [IssueCode.BAD_SPEC] * 2


def test_a_panel_s_events_survive_aggregation_and_a_file(tmp_path: Path) -> None:
    p = panel({"sale": ["2024-01-04", "2024-02-01"]})
    kept = aggregate(p, ["north", "south"], {"price": Combine.SUM})
    assert kept.events.keys() == p.events.keys()
    np.testing.assert_array_equal(kept.events["sale"], p.events["sale"])
    save(p, tmp_path / "panel.mixdyn")
    loaded = load(tmp_path / "panel.mixdyn")
    np.testing.assert_array_equal(loaded.events["sale"], p.events["sale"])
    np.testing.assert_array_equal(loaded.event_column("sale"), p.event_column("sale"))


def noiseless() -> MediaPanel:
    """104 weeks of `y = 1000 + 0.8 x + 300 sale + 150 fair`, `x` 100 a week and 250 in the weeks
    of the two events, plus a spread of SD 20 drawn once."""
    weeks = 104
    dates = START + np.timedelta64(7, "D") * np.arange(weeks)
    events = {
        "sale": np.array(["2024-11-29", "2025-11-28"], dtype="datetime64[D]"),
        "fair": np.array(["2024-05-01", "2025-05-01"], dtype="datetime64[D]"),
    }
    lift = {"sale": 300.0, "fair": 150.0}
    at = {e: np.searchsorted(dates, d, side="right") - 1 for e, d in events.items()}
    held = np.zeros(weeks, dtype=bool)
    for where in at.values():
        held[where] = True
    spend = np.where(held, 250.0, 100.0) + np.random.default_rng(4).normal(0.0, 20.0, weeks)
    kpi = 1000.0 + 0.8 * spend
    for e, where in at.items():
        kpi[where] += lift[e]
    return MediaPanel(
        dates=dates,
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=(Channel("tv"),),
        controls=(),
        kpi=kpi[None],
        kpi_unit=Revenue(),
        spend=spend[None, :, None],
        currency="EUR",
        control_values=np.zeros((1, weeks, 0)),
        filled=np.zeros((1, weeks), dtype=bool),
        events=events,
    )


def test_the_static_engines_read_an_event_as_a_column_of_their_design() -> None:
    linear = Media("tv", Geometric(0.0), Hill(1.0, 1.0))
    p = noiseless()
    adstocked = carry_over(p.spend[0, :, 0], linear.adstock)
    column = saturate(adstocked, linear.hill, half_saturation(linear.hill, adstocked))
    p = replace(p, kpi=(p.kpi[0] - 0.8 * p.spend[0, :, 0] + 400.0 * column)[None])
    fit = fit_ridge(p, NATIONAL, RidgeSpec((linear,), (Control("sale"), Control("fair")), 1e-9))
    np.testing.assert_array_equal(fit.design[:, 1], p.event_column("sale"))
    np.testing.assert_array_equal(fit.design[:, 2], p.event_column("fair"))
    assert fit.coefficients["sale"] == pytest.approx(300.0, rel=1e-3, abs=0.0)
    assert fit.coefficients["fair"] == pytest.approx(150.0, rel=1e-3, abs=0.0)
    blind = fit_ridge(p, NATIONAL, RidgeSpec((linear,), (), 1e-9, LinearTrend()))
    assert np.sqrt(np.mean((blind.estimated - p.kpi[0]) ** 2)) > 20.0
    model = bayes_model(p, NATIONAL, BayesSpec((BayesMedia("tv"),), (Control("sale"),)))
    column = model.controls[:, 0] * model.control_sd[0] + model.control_mean[0]
    np.testing.assert_allclose(column, p.event_column("sale"), rtol=0.0, atol=1e-12)
    with pytest.raises(ValueError, match=r"no channel or control named \['easter'\]"):
        bayes_model(p, NATIONAL, BayesSpec((BayesMedia("tv"),), (Control("easter"),)))


WORLDS = 40
"""Worlds 0 to 39 of the study's: at a true coverage of 0.9 the events model's falls below 30 of 40
with chance 0.0015, and the level's 90 % interval, centred about five of its widths above 0.8,
holds it in none."""


def returns_interval(p: MediaPanel, events: tuple[str, ...], level: float) -> tuple[float, float]:
    """The channel's return's central interval: its static coefficient's Student t given every
    period, as `DlmFit.last` states it."""
    effects = (Effect("tv", 1.0), *(Effect(e, 1.0) for e in events))
    fit = fit_dlm(p, NATIONAL, DlmSpec(effects, Trend(1, 0.98), Season(52.0, 2, 1.0)))
    mean, covariance, dof, _ = fit.last()
    half = float(stats.t.ppf((1 + level) / 2, dof)) * float(np.sqrt(covariance[1, 1]))
    return float(mean[1]) - half, float(mean[1]) + half


def test_with_its_events_the_dlm_holds_the_return_the_level_alone_misses() -> None:
    held = {"level": 0, "events": 0}
    for seed in range(WORLDS):
        p = world(seed)
        for name, events in (("level", ()), ("events", tuple(EVENTS))):
            low, high = returns_interval(p, events, 0.9)
            held[name] += low <= RETURN <= high
    assert held["level"] == 0
    assert held["events"] >= 30
