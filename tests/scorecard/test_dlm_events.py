"""The no-test arm `dlm-events` (ADR 0085): the DLM's search handed the recurring events the
export's flag and price mark in place of the flag, each event's coefficient static, a prior on each
channel's half-saturation point, and each plan's gain claimed at the CVaR's level; the M2 arm, as it
was."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import pytest

pytest.importorskip("causaldyn_bench")

from causaldyn_bench.scorecard.track_m2 import TRACK_M2

from mixdyn import NATIONAL
from mixdyn.data import recurring_events
from mixdyn.data.recurring import week_of_year
from mixdyn.models import DlmChoice, DlmSpec, Effect, PerEffect, Season, Trend, fit_dlm
from mixdyn.transforms import TanhShape, Transform
from scorecard import run
from scorecard.arms import dlm
from scorecard.arms import panel as reading

SEED = 905
PROMOTED = (2, 3, 4, 16, 28, 29, 40, 46, 47)
"""The weeks of the year the bench's promotions fall in (`causaldyn_bench.endogenous_mmm`)."""


@pytest.fixture(scope="module")
def observation():
    return TRACK_M2.observe(TRACK_M2.world("drawn", SEED), 0)


class Seen(Exception):
    """Raised once the search is called, so that nothing is searched."""


def searched(arm: dlm.Dlm, observation, monkeypatch) -> dict[str, Any]:
    """What `arm.fit` hands the search."""
    seen: dict[str, Any] = {}

    def spy(panel, geo, spec, grid, discounts, **kwargs):
        seen.update(panel=panel, spec=spec, discounts=discounts, **kwargs)
        raise Seen

    monkeypatch.setattr(dlm, "search_media", spy)
    with pytest.raises(Seen):
        arm.fit(observation, smoke=True)
    return seen


def test_the_arm_is_declared_by_its_name_with_four_variants() -> None:
    assert run.ARMS["dlm-events"] is dlm.DlmEvents
    assert dlm.DlmEvents.tests is False  # fitted once for every rung, on the export without tests
    assert dlm.DlmEvents.variants == (
        "dlm events plug-in",
        "dlm events mixture",
        "dlm events cvar 0.1",
        "dlm events cvar 0.1 vs equal",
    )


def test_the_search_reads_the_events_in_place_of_the_flag(observation, monkeypatch) -> None:
    """The events are the bench's promoted weeks read off the flag and the price; the flag is no
    effect, the price is; each event's discount is held at 1, the rest searched as the M2 arm
    searches them; and the half-saturation prior is the static model's scale, 1."""
    seen = searched(dlm.DlmEvents(), observation, monkeypatch)
    found = recurring_events(reading.panel(observation), dlm.INDICATORS)
    # on this world the flag and the price miss the autumn sale's week 40, a 30 % discount
    assert {w for weeks in found.weeks for w in weeks} == set(PROMOTED) - {40}
    spec = seen["spec"]
    assert isinstance(spec, DlmSpec)
    columns = [e.column for e in spec.effects]
    assert columns == [*observation.channels, "price", *found.names]
    events = [e for e in spec.effects if e.column in found.names]
    assert {e.discount for e in events} == {1.0}
    assert all(e.transform is None and e.sign is None for e in events)
    discounts = seen["discounts"]
    assert isinstance(dlm.DISCOUNTS.effects, PerEffect)
    assert discounts.effects == PerEffect(dlm.DISCOUNTS.effects.values, held=found.names)
    assert discounts.trend == dlm.DISCOUNTS.trend
    assert seen["half_saturation_prior"] == 1.0
    panel = seen["panel"]
    of_year = week_of_year(panel.dates)
    for name, weeks in zip(found.names, found.weeks, strict=True):
        np.testing.assert_array_equal(panel.event_column(name), np.isin(of_year, weeks))


def test_the_m2_arm_searches_as_it_did(observation, monkeypatch) -> None:
    """Every control an effect, no event, every effect's discount searched, no prior."""
    seen = searched(dlm.Dlm(), observation, monkeypatch)
    assert [e.column for e in seen["spec"].effects] == [
        *observation.channels,
        *observation.controls,
    ]
    assert seen["discounts"] is dlm.DISCOUNTS
    assert seen["half_saturation_prior"] is None
    assert seen["panel"].events == {}
    assert (dlm.Dlm.events, dlm.Dlm.half_saturation_prior, dlm.Dlm.claim) == (False, None, None)


@pytest.fixture(scope="module")
def fitted(observation) -> dlm.Fitted:
    """The events arm's fit of one spec, a `tanh` on every channel, drawn from 4 chains of 100."""
    data = reading.panel(observation)
    found = recurring_events(data, dlm.INDICATORS)
    data = replace(data, events=found.days(data.dates))
    spend = np.asarray(observation.spend, dtype=float)
    media = [
        Effect(
            name,
            0.99,
            sign="positive",
            transform=Transform(
                0.5,
                float(spend[spend[:, j] > 0.0, j].mean()),
                TanhShape(),
                observation.kernel_length,
            ),
        )
        for j, name in enumerate(observation.channels)
    ]
    spec = DlmSpec(
        effects=(*media, Effect("price", 0.99), *(Effect(e, 1.0) for e in found.names)),
        trend=Trend(1, 0.99),
        season=Season(dlm.YEAR, dlm.HARMONICS, 1.0),
    )
    fit = fit_dlm(data, NATIONAL, spec, draws=100, seed=0)
    choice = DlmChoice((spec,), (fit.filtered,), 0, MappingProxyType({0: fit}))
    ess = dlm._ess([fit])
    readings = choice.readings(dlm.DRAWS, observation.seed)
    flags = dlm._flags(choice, observation.channels)
    return dlm.Fitted(choice, readings, None, flags, {}, {}, ess, ess, found)


def test_a_claim_at_a_level_is_the_mean_of_the_worst_share_of_the_gains() -> None:
    gains = np.arange(20.0)[::-1]
    assert dlm.claimed(gains, 0.1) == 0.5
    assert dlm.claimed(gains, 1.0) == gains.mean()
    assert dlm.claimed(gains, 0.01) == 0.0
    assert dlm.claimed(np.array([3.0]), 0.1) == 3.0


@pytest.mark.parametrize("variant", dlm.DlmEvents.variants)
def test_each_plan_claims_its_gain_at_the_cvar_s_level(fitted, observation, variant) -> None:
    """The claim is the mean of the worst tenth of the readings' gains of the plan, at most their
    mean; the interval is their central 90 % as the M2 arm's."""
    planned = dlm.DlmEvents().plan(fitted, variant, observation)
    m2 = dlm.Dlm().plan(fitted, variant.replace("dlm events", "dlm"), observation)
    np.testing.assert_array_equal(m2.weekly, planned.weekly)
    assert m2.gain_interval == planned.gain_interval
    # the readings' gains as the arm reads them: the M2 arm's claim is their mean
    gains = reading.gains(fitted.readings, observation, planned.weekly)
    assert m2.gain == pytest.approx(gains.mean(), rel=1e-12, abs=1e-9)
    assert planned.gain == pytest.approx(dlm.claimed(gains, 0.1), rel=1e-12, abs=1e-9)
    assert planned.gain is not None
    assert m2.gain is not None
    assert planned.gain <= m2.gain


def test_the_forecast_reads_each_event_over_the_quarter(fitted, observation) -> None:
    """The quarter's plan names every effect of the spec: the channels at the status quo, the price
    at the quarter's, and each event 1 in its weeks of the year past the history's calendar."""
    seen = {}

    class Spy:
        fit = fitted.choice.fit

        def forecast(self, horizon, plan, draws, seed):
            seen.update(horizon=horizon, plan=plan)
            raise Seen

    with pytest.raises(Seen):
        dlm._forecast(cast("DlmChoice", Spy()), observation, fitted.events)
    plan = seen["plan"]
    found = fitted.events
    assert set(plan) == {e.column for e in fitted.choice.fit.spec.effects}
    weeks = observation.sales.shape[0]
    dates = reading.FIRST_WEEK + np.timedelta64(7, "D") * np.arange(
        weeks, weeks + observation.planned
    )
    for name, weeks_of_year in zip(found.names, found.weeks, strict=True):
        np.testing.assert_array_equal(plan[name], np.isin(week_of_year(dates), weeks_of_year))
    np.testing.assert_array_equal(plan["price"], observation.future_controls["price"])
