"""mixdyn.calibration's tables: PyMC-Marketing's lift table and geo experiments' records read as
lift estimates, and written back.

The tables are written by hand in the shapes the formats document: PyMC-Marketing's lift table
(`MMM.add_lift_test_measurements`: `channel`, `x`, `delta_x`, `delta_y`, `sigma`, a column for the
model's geo dimension and an optional `date`, as its lift-test and geo-lift notebooks build them)
and a GeoX analysis' record of an experiment (its estimate, standard error and window), each with
a go-dark test and a heavy-up. Read and written back, each table is the one read: the lift and its
standard error are the record's to the bit, and the lift table's a changed period's times the
period count, to the bit where that count is a power of two and within one rounding otherwise. The
estimates name what each test estimated: a lift against zero spend where the test went dark, a
marginal change where it added spend.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mixdyn import Cadence, Channel, Geo, MediaPanel, Revenue
from mixdyn.api import estimands
from mixdyn.calibration import (
    LiftEstimate,
    experiment_table,
    lift_table,
    read_experiments,
    read_lift_table,
)

SEARCH, SOCIAL, TV = Channel("search"), Channel("social"), Channel("tv")
NORTH, SOUTH = Geo("north"), Geo("south")
WEEKS = 30
DATES = np.datetime64("2026-01-05") + np.timedelta64(7, "D") * np.arange(WEEKS)

LIFT_TABLE: dict[str, list[Any]] = {
    "channel": ["search", "social", "tv"],
    "x": [5200.0, 1830.5, 0.0],
    "delta_x": [-5200.0, 915.25, 12000.0],
    "delta_y": [-6104.3, 1207.8, 15480.25],
    "sigma": [1380.2, 402.6, 3911.0],
}
GEO_LIFT_TABLE: dict[str, list[Any]] = LIFT_TABLE | {
    "geo": ["north", "south", "north"],
    "date": ["2026-03-02", "2026-04-06", "2026-05-04"],
}
RECORDS: dict[str, list[Any]] = {
    "channel": ["search", "tv"],
    "estimate": [-24411.7, 61920.4],
    "standard_error": [5522.9, 15644.1],
    "start": ["2026-03-02", "2026-05-04"],
    "end": ["2026-03-23", "2026-05-25"],
    "spend": [0.0, 12000.0],
    "planned": [5200.0, 0.0],
    "cooldown": [4, 2],
    "geo": ["north", "south"],
    "share": [1.0, 0.5],
}


def _panel(cadence: Cadence = Cadence.WEEK, dates: np.ndarray = DATES) -> MediaPanel:
    geos, channels = (NORTH, SOUTH), (SEARCH, SOCIAL, TV)
    return MediaPanel(
        dates=dates,
        cadence=cadence,
        geos=geos,
        channels=channels,
        controls=(),
        kpi=np.ones((2, dates.size)),
        kpi_unit=Revenue(),
        spend=np.ones((2, dates.size, 3)),
        currency="EUR",
        control_values=np.zeros((2, dates.size, 0)),
        filled=np.zeros((2, dates.size), dtype=bool),
    )


PANEL = _panel()


def _frame(data: dict[str, list[Any]], library: str) -> Any:
    if library == "dict":
        return data
    return pd.DataFrame(data) if library == "pandas" else pl.DataFrame(data)


@pytest.mark.parametrize("library", ["dict", "pandas", "polars"])
@pytest.mark.parametrize("periods", [4, 3])
def test_a_lift_table_read_and_written_back_is_the_table(library: str, periods: int) -> None:
    """Each row's lift is `delta_y` times the periods the change held and its error `sigma`
    times them; written back, `delta_y` and `sigma` are the table's to the bit at 4 periods and
    within one rounding at 3, `x` to the bit, and `delta_x` within one rounding of `x`."""
    read = read_lift_table(PANEL, _frame(LIFT_TABLE, library), periods=periods, cooldown=2)
    assert list(read) == [SEARCH, SOCIAL, TV]
    for row, channel in enumerate(LIFT_TABLE["channel"]):
        (test,) = read[Channel(channel)]
        x, delta_x = LIFT_TABLE["x"][row], LIFT_TABLE["delta_x"][row]
        assert test.lift == LIFT_TABLE["delta_y"][row] * periods
        assert test.se == LIFT_TABLE["sigma"][row] * periods
        assert test.readout == periods + 2
        np.testing.assert_array_equal(test.spend, [x + delta_x] * periods + [x] * 2)
        np.testing.assert_array_equal(test.planned, [x] * (periods + 2))
        assert test.geos is None
        assert test.end is None
    written = lift_table(PANEL, read, periods=periods, cooldown=2)
    assert written.keys() == LIFT_TABLE.keys()
    assert written["channel"] == LIFT_TABLE["channel"]
    assert written["x"] == LIFT_TABLE["x"]
    rounding = 0.0 if periods == 4 else float(np.finfo(float).eps)
    for name in ("delta_y", "sigma"):
        np.testing.assert_allclose(
            np.array(written[name], dtype=float), LIFT_TABLE[name], rtol=rounding, atol=0.0
        )
    np.testing.assert_allclose(
        np.array(written["delta_x"], dtype=float),
        LIFT_TABLE["delta_x"],
        rtol=0.0,
        atol=float(np.finfo(float).eps) * max(LIFT_TABLE["x"]),
    )


def test_a_geo_lift_tables_geos_and_dates_place_each_test() -> None:
    """A row's geo is the test's group, and its date the first changed period: the readout ends
    `periods + cooldown - 1` weeks after it. Written back, the geos and dates are the table's."""
    read = read_lift_table(PANEL, GEO_LIFT_TABLE, periods=4, cooldown=2)
    ends = [np.datetime64("2026-04-06"), np.datetime64("2026-05-11"), np.datetime64("2026-06-08")]
    for row, channel in enumerate(GEO_LIFT_TABLE["channel"]):
        (test,) = read[Channel(channel)]
        assert test.geos == frozenset({Geo(GEO_LIFT_TABLE["geo"][row])})
        assert test.share == 1.0
        assert test.end == ends[row]
    written = lift_table(PANEL, read, periods=4, cooldown=2)
    assert written["geo"] == GEO_LIFT_TABLE["geo"]
    assert written["date"] == GEO_LIFT_TABLE["date"]
    assert written["delta_y"] == GEO_LIFT_TABLE["delta_y"]
    assert written["sigma"] == GEO_LIFT_TABLE["sigma"]


def test_each_test_read_from_a_table_names_its_estimand() -> None:
    """Search went dark: a lift against zero spend. Social added to a spend: a marginal change.
    TV spent from nothing: a lift against zero spend."""
    read = read_lift_table(PANEL, GEO_LIFT_TABLE, periods=4, cooldown=2)
    named = {e.channel: e for e in estimands(PANEL, read)}
    assert named["search"].reading == "lift against zero spend"
    assert named["search"].change == "withheld"
    assert named["social"].reading == "marginal change"
    assert named["social"].change == "added"
    assert named["tv"].reading == "lift against zero spend"
    first, last = named["search"].first, named["search"].last
    assert first is not None
    assert last is not None
    assert (last - first).days == 21
    assert named["search"].periods == 4
    assert named["search"].readout == 6


@pytest.mark.parametrize("library", ["dict", "pandas", "polars"])
def test_experiments_read_and_written_back_are_the_records(library: str) -> None:
    """Each record's estimate and standard error are the lift and its error to the bit, its
    window the periods its spend changed, read with its cooldown after; written back, every
    column is the record's."""
    read = read_experiments(PANEL, _frame(RECORDS, library))
    (search,), (tv,) = read[SEARCH], read[TV]
    assert (search.lift, search.se) == (-24411.7, 5522.9)
    assert (tv.lift, tv.se) == (61920.4, 15644.1)
    np.testing.assert_array_equal(search.spend, [0.0] * 4 + [5200.0] * 4)
    np.testing.assert_array_equal(tv.planned, [0.0] * 6)
    assert (search.readout, tv.readout) == (8, 6)
    assert search.end == np.datetime64("2026-04-20")
    assert tv.end == np.datetime64("2026-06-08")
    assert (search.geos, tv.geos, tv.share) == (frozenset({NORTH}), frozenset({SOUTH}), 0.5)
    assert experiment_table(PANEL, read) == RECORDS


def test_an_experiments_return_is_read_as_its_lift_over_the_spend_it_changed() -> None:
    """A return of 1.2 a euro withheld, 5200 a week over four weeks, is a lift of -24960; its
    error scales by the spend's size, 20800."""
    records = RECORDS | {"estimate": [1.2, 1.5], "standard_error": [0.25, 0.3]}
    read = read_experiments(PANEL, records, per_unit=True)
    (search,), (tv,) = read[SEARCH], read[TV]
    assert search.lift == pytest.approx(-1.2 * 4 * 5200.0, rel=1e-15, abs=0.0)
    assert search.se == pytest.approx(0.25 * 4 * 5200.0, rel=1e-15, abs=0.0)
    assert tv.lift == pytest.approx(1.5 * 4 * 12000.0, rel=1e-15, abs=0.0)


def test_a_monthly_panels_dates_step_by_months() -> None:
    months = np.datetime64("2025-01", "M") + np.arange(24)
    panel = _panel(Cadence.MONTH, months.astype("datetime64[D]"))
    table = LIFT_TABLE | {"date": ["2025-11-01", "2026-01-01", "2025-02-01"]}
    read = read_lift_table(panel, table, periods=2, cooldown=1)
    assert read[SEARCH][0].end == np.datetime64("2026-01-01")
    assert read[SOCIAL][0].end == np.datetime64("2026-03-01")
    assert lift_table(panel, read, periods=2, cooldown=1)["date"] == table["date"]


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"sigma": None}, "no columns \\['sigma'\\]"),
        ({"channel": ["search", "radio", "tv"]}, "channel 'radio'"),
        ({"geo": ["north", "east", "north"]}, "geo 'east'"),
        ({"date": ["2026-03-03", "2026-04-06", "2026-05-04"]}, "dates the panel's calendar"),
        ({"x": [5200.0, "a", 0.0]}, "row 1's x is not a number"),
        ({"sigma": [1380.2, 0.0, 3911.0]}, "row 1: the standard error must be positive"),
        ({"delta_x": [-6000.0, 915.25, 12000.0]}, "row 0: spend must be finite and at or above"),
        ({"delta_x": [-5200.0, 0.0, 12000.0]}, "row 1: the test changed no spend"),
    ],
)
def test_a_lift_table_refuses_what_it_cannot_read(change: dict[str, Any], match: str) -> None:
    table = {k: v for k, v in (GEO_LIFT_TABLE | change).items() if v is not None}
    with pytest.raises(ValueError, match=match):
        read_lift_table(PANEL, table, periods=4)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"end": ["2026-02-23", "2026-05-25"]}, "before it starts"),
        ({"cooldown": [1.5, 2]}, "row 0's cooldown"),
        ({"start": ["2026-03-04", "2026-05-04"]}, "dates the panel's calendar"),
        ({"share": [1.0, 0.0]}, "row 1: share=0.0"),
    ],
)
def test_experiments_refuse_what_they_cannot_read(change: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        read_experiments(PANEL, RECORDS | change)


def test_a_table_written_refuses_a_test_its_rows_cannot_hold() -> None:
    ramp = LiftEstimate(np.array([0.0, 1.0, 2.0]), np.full(3, 2.0), 3, -1.0, 1.0)
    with pytest.raises(ValueError, match="one changed level"):
        lift_table(PANEL, {SEARCH: (ramp,)}, periods=2)
    late = LiftEstimate(np.array([2.0, 0.0, 0.0]), np.full(3, 2.0), 3, -1.0, 1.0)
    with pytest.raises(ValueError, match="is not 2 changed periods"):
        lift_table(PANEL, {SEARCH: (late,)}, periods=2, cooldown=1)
    shared = LiftEstimate(np.zeros(2), np.full(2, 2.0), 2, -1.0, 1.0, share=0.5)
    with pytest.raises(ValueError, match="a row reads all"):
        lift_table(PANEL, {SEARCH: (shared,)}, periods=2)
    both = LiftEstimate(np.zeros(2), np.full(2, 2.0), 2, -1.0, 1.0, geos=frozenset({NORTH, SOUTH}))
    with pytest.raises(ValueError, match="a row names one geo"):
        lift_table(PANEL, {SEARCH: (both,)}, periods=2)
    with pytest.raises(ValueError, match="not placed in the calendar"):
        experiment_table(PANEL, {SEARCH: (LiftEstimate(np.zeros(2), np.ones(2), 2, -1.0, 1.0),)})
