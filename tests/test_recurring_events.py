"""Recurring events read off a panel's noisy records of them (ADR 0085): the weeks of the year an
event falls in every year, found from a flag that misses some of its weeks and a price that drops
in them, and on a world whose spend rises with its events the model with the events read holds the
channel's return where the level, the flag and the price alone do not."""

from __future__ import annotations

import datetime
from dataclasses import replace

import numpy as np
import pytest

from mixdyn import NATIONAL, Cadence, Channel, Geo, MediaPanel, Revenue
from mixdyn.data import recurring_events
from mixdyn.data.recurring import week_of_year
from study_recurring_events import EVENTS, RETURN, read, world

START = np.datetime64("2023-01-02")
"""A Monday, the first day of ISO week 1 of 2023."""


def panel(flag: np.ndarray, price: np.ndarray, start: np.datetime64 = START) -> MediaPanel:
    weeks = flag.size
    rng = np.random.default_rng(1)
    return MediaPanel(
        dates=start + np.timedelta64(7, "D") * np.arange(weeks),
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=(Channel("tv"),),
        controls=("promotion", "price"),
        kpi=1000.0 + rng.normal(0.0, 10.0, (1, weeks)),
        kpi_unit=Revenue(),
        spend=rng.gamma(2.0, 5.0, (1, weeks, 1)),
        currency="EUR",
        control_values=np.column_stack([flag, price])[None],
        filled=np.zeros((1, weeks), dtype=bool),
    )


def recorded(
    weeks: tuple[int, ...], seed: int = 0, recall: float = 0.7, false: float = 0.15
) -> MediaPanel:
    """Three years of a panel whose events fall in `weeks` of every year, recorded by a flag of
    `recall` and `false` positives and by a price 0.4 below 1 in them, with noise of SD 0.08. The
    three years from 2023 are ISO years of 52 weeks."""
    rng = np.random.default_rng([81, seed])
    of_year = np.arange(156) % 52 + 1
    held = np.isin(of_year, weeks)
    caught = rng.random(held.size)
    flag = np.where(held, caught < recall, caught < false).astype(float)
    price = 1.0 - 0.4 * held + rng.normal(0.0, 0.08, held.size)
    return panel(flag, price)


def test_a_week_of_the_year_is_iso_8601_s_its_53rd_read_as_its_52nd() -> None:
    """Every day of 1999 to 2030, which hold ISO years of 53 weeks."""
    days = np.datetime64("1999-01-01") + np.arange(32 * 366)
    expected = [
        min(datetime.date.fromisoformat(str(d)).isocalendar()[1], 52) for d in days.tolist()
    ]
    np.testing.assert_array_equal(week_of_year(days), expected)


def test_the_events_are_the_weeks_of_the_year_the_records_mark_in_runs() -> None:
    """Weeks 2 to 4, 16 and 46 to 47, each run of consecutive weeks one event, named by its
    weeks; the posterior is near 1 in them and near 0 elsewhere."""
    found = recurring_events(recorded((2, 3, 4, 16, 46, 47)), {"promotion": 1, "price": -1})
    assert found.weeks == ((2, 3, 4), (16,), (46, 47))
    assert found.names == ("recurring weeks 2-4", "recurring week 16", "recurring weeks 46-47")
    assert found.separation > 0.0
    inside = np.isin(np.arange(1, 53), (2, 3, 4, 16, 46, 47))
    assert np.all(found.posterior[inside] > 0.99)
    assert np.all(found.posterior[~inside] < 0.01)


def test_a_run_that_spans_the_year_s_end_is_one_event() -> None:
    found = recurring_events(recorded((1, 2, 30, 51, 52)), {"promotion": 1, "price": -1})
    assert found.weeks == ((51, 52, 1, 2), (30,))
    assert found.names == ("recurring weeks 51-2", "recurring week 30")


def test_an_event_s_column_is_1_in_its_weeks_of_every_year_and_past_the_calendar() -> None:
    """Over the panel's calendar, the days `MediaPanel.events` holds read back as the column; past
    it, the weeks of the year carry on."""
    data = recorded((2, 3, 4, 16))
    found = recurring_events(data, {"promotion": 1, "price": -1})
    with_events = replace(data, events=found.days(data.dates))
    for name, column in found.columns(data.dates).items():
        np.testing.assert_array_equal(with_events.event_column(name), column)
    later = data.dates[-1] + np.timedelta64(7, "D") * np.arange(1, 53)
    columns = found.columns(later)
    np.testing.assert_array_equal(np.flatnonzero(columns["recurring weeks 2-4"]), [1, 2, 3])
    np.testing.assert_array_equal(np.flatnonzero(columns["recurring week 16"]), [15])


def test_records_that_do_not_recur_read_no_event() -> None:
    """A flag on at random a fifth of the weeks and a price of noise alone: one class reads better,
    on every seed tried."""
    for seed in range(10):
        rng = np.random.default_rng([82, seed])
        flag = (rng.random(156) < 0.2).astype(float)
        price = 1.0 + rng.normal(0.0, 0.08, 156)
        found = recurring_events(panel(flag, price), {"promotion": 1, "price": -1})
        assert found.names == ()
        assert found.weeks == ()
        assert found.separation <= 0.0


def test_a_clean_flag_alone_reads_the_events() -> None:
    """A flag of no miss and no false week reads the events by itself; the export's flag, of recall
    0.7 and false positives 0.15, over three years is weak evidence alone (ADR 0085)."""
    clean = recorded((10, 11, 30), recall=1.0, false=0.0)
    assert recurring_events(clean, {"promotion": 1}).weeks == ((10, 11), (30,))


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda p: recurring_events(p, {}), "at least one indicator"),
        (lambda p: recurring_events(p, {"sales": 1}), "'sales' is not one of the controls"),
        (lambda p: recurring_events(p, {"price": 2}), "direction 2, not 1 or -1"),
        (lambda p: recurring_events(p, {"price": True}), "direction True, not 1 or -1"),
        (lambda p: recurring_events(p, {"price": -1}, geo=Geo("north")), "no geo 'north'"),
    ],
)
def test_what_cannot_be_read_is_refused(make, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        make(recorded((2,)))


def test_a_panel_too_short_or_records_that_never_move_are_refused() -> None:
    short = panel(np.zeros(51), np.ones(51))
    with pytest.raises(ValueError, match="a year of observed weeks"):
        recurring_events(short, {"promotion": 1, "price": -1})
    still = panel(np.zeros(104), 1.0 + np.random.default_rng(0).normal(0.0, 0.1, 104))
    with pytest.raises(ValueError, match="indicators that vary"):
        recurring_events(still, {"promotion": 1, "price": -1})
    data = recorded((2,))
    daily = replace(
        data, dates=START + np.arange(data.dates.size).astype("timedelta64[D]"), cadence=Cadence.DAY
    )
    with pytest.raises(ValueError, match="weekly panel"):
        recurring_events(daily, {"promotion": 1})


def test_the_learned_events_hold_the_channel_s_return_where_the_level_alone_does_not() -> None:
    """The event terms' criterion on the study world (`scripts/study_recurring_events.py`), worlds
    0 to 39: the events read off the flag and the price are the world's on at least 38, the
    channel's 90 % interval holds its return of 0.8 in at least 36, and the level with the flag and
    the price as controls holds it in none (ADR 0085's 200 worlds: see the record)."""
    reads = [read(world(seed), seed) for seed in range(40)]
    assert sum(found for found, _ in reads) >= 38
    for name, least, most in (("learned", 36, 40), ("level", 0, 0)):
        low, high = np.array([r[name][0] for _, r in reads]).T
        held = int(np.sum((low <= RETURN) & (high >= RETURN)))
        assert least <= held <= most, (name, held)
    assert set(EVENTS) == {"clearance", "spring", "cyber"}
