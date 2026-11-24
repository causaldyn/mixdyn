from __future__ import annotations

import datetime as dt
from decimal import Decimal
from fractions import Fraction
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mixdyn import (
    NATIONAL,
    Cadence,
    Channel,
    Count,
    Gap,
    Issue,
    IssueCode,
    MediaPanel,
    PanelError,
    Revenue,
    Severity,
    read_panel,
)
from mixdyn.decisions import returns
from mixdyn.models import DlmSpec, Effect, fit_dlm
from mixdyn.models.ridge_mmm import Media, RidgeSpec, fit_ridge
from mixdyn.transforms import Geometric, Hill, HillShape, Transform

START = np.datetime64("2023-01-07")  # a Saturday
WEEK = np.timedelta64(7, "D")


def long_table(
    weeks: int = 8, geos: tuple[str, ...] = ("north", "south"), seed: int = 0
) -> dict[str, np.ndarray]:
    """One row per geo and week, two channels and a control, every value distinct."""
    rng = np.random.default_rng(seed)
    n = weeks * len(geos)
    return {
        "date": np.tile(START + WEEK * np.arange(weeks), len(geos)),
        "region": np.repeat(np.array(geos, dtype=object), weeks),
        "sales": rng.uniform(100, 200, n),
        "spend_tv": rng.uniform(1, 50, n),
        "spend_search": rng.uniform(1, 50, n),
        "price": rng.uniform(5, 10, n),
    }


def read(data: Any, **overrides: Any) -> MediaPanel:
    spec: dict[str, Any] = {
        "date": "date",
        "geo": "region",
        "kpi": "sales",
        "spend": {"tv": "spend_tv", "search": "spend_search"},
        "controls": ("price",),
        "cadence": Cadence.WEEK,
        "kpi_unit": Revenue(),
        "currency": "USD",
    }
    spec.update(overrides)
    return read_panel(data, **spec)


def without(data: dict[str, np.ndarray], rows: list[int]) -> dict[str, np.ndarray]:
    keep = np.ones(len(data["date"]), dtype=bool)
    keep[rows] = False
    return {name: values[keep] for name, values in data.items()}


def others(issues: tuple[Issue, ...]) -> tuple[Issue, ...]:
    """The findings but the warning that a table this small observes too few periods an effect."""
    return tuple(i for i in issues if i.code is not IssueCode.FEW_OBSERVATIONS)


def errors(excinfo: pytest.ExceptionInfo[PanelError]) -> list[Any]:
    return [i for i in excinfo.value.issues if i.severity is Severity.ERROR]


def test_reads_a_long_table_onto_one_calendar() -> None:
    data = long_table()
    panel = read(data)
    assert panel.geos == ("north", "south")
    assert panel.channels == ("tv", "search")
    assert panel.controls == ("price",)
    assert np.array_equal(panel.dates, START + WEEK * np.arange(8))
    assert panel.kpi.shape == (2, 8)
    assert panel.spend.shape == (2, 8, 2)
    assert not panel.filled.any()
    assert [i.code for i in panel.issues] == [IssueCode.FEW_OBSERVATIONS]
    np.testing.assert_array_equal(panel.kpi[1], data["sales"][8:])
    np.testing.assert_array_equal(panel.spend[0, :, 1], data["spend_search"][:8])
    np.testing.assert_array_equal(panel.control_values[1, :, 0], data["price"][8:])


def test_panel_arrays_are_read_only() -> None:
    panel = read(long_table())
    with pytest.raises(ValueError, match="read-only"):
        panel.spend[0, 0, 0] = 1.0


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 2**32 - 1))
def test_row_order_does_not_change_the_panel(seed: int) -> None:
    data = long_table(weeks=6, geos=("a", "b", "c"))
    order = np.random.default_rng(seed).permutation(len(data["date"]))
    shuffled = read({name: values[order] for name, values in data.items()})
    reference = read(data)
    assert shuffled.geos == reference.geos
    for field in ("dates", "kpi", "spend", "control_values", "filled"):
        np.testing.assert_array_equal(getattr(shuffled, field), getattr(reference, field))


def test_a_missing_week_is_an_error_naming_the_geo_and_the_date() -> None:
    data = without(long_table(), [3])
    with pytest.raises(PanelError) as excinfo:
        read(data)
    (issue,) = errors(excinfo)
    assert issue.code is IssueCode.MISSING_PERIOD
    assert issue.geo == "north"
    assert issue.dates == (dt.date(2023, 1, 28),)
    assert "geo 'north' has no row for 1 of 8 periods: 2023-01-28" in str(excinfo.value)


def test_a_missing_week_becomes_a_filled_step_under_gap_step() -> None:
    data = long_table()
    panel = read(without(data, [3]), gaps=Gap.STEP)
    assert np.array_equal(panel.dates, START + WEEK * np.arange(8))
    assert np.flatnonzero(panel.filled[0]).tolist() == [3]
    assert np.isnan(panel.kpi[0, 3])
    assert (panel.spend[0, 3] == 0).all()
    assert np.isnan(panel.control_values[0, 3]).all()
    np.testing.assert_array_equal(panel.kpi[0, 4:], data["sales"][4:8])
    (warning,) = others(panel.issues)
    assert warning.code is IssueCode.MISSING_PERIOD
    assert warning.severity is Severity.WARNING


@settings(max_examples=60, deadline=None)
@given(data=st.data())
def test_dropped_rows_leave_every_other_row_on_its_date(data: st.DataObject) -> None:
    table = long_table(weeks=10, geos=("a", "b"))
    n = len(table["date"])
    # keep the first and the last week in one geo, so the calendar's ends are fixed, and a row of
    # the other, since a table with none of a geo's rows does not name it
    droppable = [i for i in range(n) if i not in (0, 9)]
    dropped = data.draw(
        st.lists(st.sampled_from(droppable), unique=True, max_size=12).filter(
            lambda d: not set(range(10, n)) <= set(d)
        )
    )
    full = read(table)
    gapped = read(without(table, dropped), gaps=Gap.STEP)
    assert np.array_equal(gapped.dates, full.dates)
    expected_filled = np.zeros((2, 10), dtype=bool)
    for i in dropped:
        expected_filled[i // 10, i % 10] = True
    assert np.array_equal(gapped.filled, expected_filled)
    kept = ~expected_filled
    np.testing.assert_array_equal(gapped.kpi[kept], full.kpi[kept])
    np.testing.assert_array_equal(gapped.spend[kept], full.spend[kept])


def test_a_geo_that_starts_late_is_reported_as_one_run() -> None:
    data = without(long_table(), [8, 9, 10])
    with pytest.raises(
        PanelError,
        match=r"geo 'south' has no row for 3 of 8 periods: "
        r"2023-01-07 to 2023-01-21 \(3\)",
    ):
        read(data)


def test_duplicate_rows_are_errors() -> None:
    data = long_table()
    doubled = {name: np.concatenate([values, values[2:3]]) for name, values in data.items()}
    with pytest.raises(PanelError) as excinfo:
        read(doubled)
    (issue,) = errors(excinfo)
    assert issue.code is IssueCode.DUPLICATE
    assert issue.dates == (dt.date(2023, 1, 21), dt.date(2023, 1, 21))


def test_rows_per_geo_without_a_geo_column_point_at_the_geo_column() -> None:
    with pytest.raises(PanelError, match="name its geo column"):
        read(long_table(), geo=None)


def test_a_panel_without_a_geo_column_has_one_national_geo() -> None:
    data = {name: values[:8] for name, values in long_table().items() if name != "region"}
    panel = read(data, geo=None)
    assert panel.geos == (NATIONAL,)


def test_negative_spend_is_an_error_with_its_geo_dates_and_lowest_value() -> None:
    data = long_table()
    data["spend_tv"][[9, 12]] = [-5.0, -2.5]
    with pytest.raises(PanelError) as excinfo:
        read(data)
    (issue,) = errors(excinfo)
    assert issue.code is IssueCode.NEGATIVE_SPEND
    assert issue.geo == "south"
    assert issue.column == "spend_tv"
    assert issue.dates == (dt.date(2023, 1, 14), dt.date(2023, 2, 4))
    assert "(lowest -5)" in issue.message


def test_missing_spend_is_an_error_not_a_zero() -> None:
    data = long_table()
    data["spend_search"][4] = np.nan
    with pytest.raises(PanelError, match="write 0 where nothing was spent") as excinfo:
        read(data)
    assert [i.code for i in errors(excinfo)] == [IssueCode.NON_FINITE]


def test_a_missing_kpi_is_a_warning_and_stays_missing() -> None:
    data = long_table()
    data["sales"][5] = np.nan
    panel = read(data)
    (warning,) = others(panel.issues)
    assert warning.code is IssueCode.MISSING_KPI
    assert np.isnan(panel.kpi[0, 5])
    assert not panel.filled[0, 5]


def test_text_in_a_number_column_is_reported_once() -> None:
    data = long_table()
    sales = data["sales"].astype(object)
    sales[2] = "1,250"
    data["sales"] = sales
    with pytest.raises(PanelError) as excinfo:
        read(data)
    (issue,) = excinfo.value.issues
    assert issue.code is IssueCode.NOT_NUMERIC
    assert "'1,250'" in issue.message


@pytest.mark.parametrize("cell", ["1.250", "nan", "1_000", "inf", "1e400"])
def test_text_is_not_read_as_a_number_even_where_it_spells_one(cell: str) -> None:
    """`float` read "1.250" as 1.25, where a locale that writes thousands with a point means 1250;
    the text "nan" as a missing cell, "1_000" as 1000, "inf" as an infinite KPI and "1e400" as a
    number past float64's range."""
    data = long_table()
    sales = data["sales"].astype(object)
    sales[3] = cell
    data["sales"] = sales
    with pytest.raises(PanelError, match="column 'sales' holds text") as e:
        read(data)
    (issue,) = e.value.issues
    assert issue.code is IssueCode.NOT_NUMERIC
    assert repr(cell) in issue.message
    assert issue.message.endswith("convert it to numbers where its locale is known")


@pytest.mark.parametrize(
    "dtype",
    [object, np.str_, np.bytes_, np.dtypes.StringDType()],
    ids=["object", "str", "bytes", "StringDType"],
)
def test_a_column_of_text_is_not_read_as_numbers(dtype: Any) -> None:
    data = long_table()
    sales = np.array([str(v) for v in data["sales"].tolist()], dtype=dtype)
    data["sales"] = sales
    with pytest.raises(PanelError, match="column 'sales' holds text") as e:
        read(data)
    assert IssueCode.NOT_NUMERIC in {i.code for i in errors(e)}


def test_a_week_off_the_grid_is_an_error_naming_the_weekday() -> None:
    data = long_table()
    data["date"] = data["date"].copy()
    data["date"][5] += np.timedelta64(2, "D")
    with pytest.raises(PanelError, match="off the grid of weeks starting on Saturday") as excinfo:
        read(data)
    codes = {i.code for i in errors(excinfo)}
    assert IssueCode.OFF_GRID in codes


@pytest.mark.parametrize(
    ("dates", "match"),
    [
        (np.arange(16), "holds numbers, not dates"),
        (np.array(["2023-01-07T12:00"] * 16, dtype=object), "time of day"),
        (np.array(["2023-01-07"] * 15 + ["soon"], dtype=object), "not dates: 'soon' at 15"),
        (np.array(["2023-01-07"] * 15 + [None], dtype=object), "has no date at row 15"),
        # numpy reads "-4" as the year -4, which no Python date holds
        (["2023-01-07"] * 15 + [-4], "not dates: '-4' at 15"),
        (np.array(["2023-01-07"] * 15 + ["soon"]), "not dates: 'soon' at 15"),
    ],
)
def test_dates_that_are_not_calendar_days_are_refused(dates: Any, match: str) -> None:
    data = long_table()
    data["date"] = dates
    with pytest.raises(PanelError, match=match):
        read(data)


def test_a_day_no_python_date_holds_is_refused_where_an_issue_would_list_it() -> None:
    """Every issue lists its dates as Python's, years 1 to 9999: a row past them that another
    issue names, here a missing KPI, is refused as not a date, not listed."""
    data = long_table()
    data["date"] = data["date"].copy()
    data["date"][15] = np.datetime64("10000-01-01")
    data["sales"][15] = np.nan
    with pytest.raises(PanelError, match=r"not dates: .*10000-01-01.* at 15"):
        read(data)


def test_a_typo_in_a_year_is_refused_before_the_calendar_is_built() -> None:
    data = long_table()
    data["date"] = data["date"].copy()
    data["date"][7] = START + WEEK * 52 * 180  # on the weekly grid, 180 years on
    with pytest.raises(PanelError, match=r"the dates span .* check the earliest and the latest"):
        read(data)


def test_monthly_periods_are_labelled_by_the_first_of_the_month() -> None:
    month_ends = np.array(["2024-01-31", "2024-02-29", "2024-03-31"], dtype="datetime64[D]")
    data = {"date": month_ends, "sales": np.array([1.0, 2.0, 3.0]), "tv": np.ones(3)}
    panel = read(data, geo=None, spend={"tv": "tv"}, controls=(), cadence=Cadence.MONTH)
    assert panel.dates.astype(str).tolist() == ["2024-01-01", "2024-02-01", "2024-03-01"]


def test_two_dates_in_one_month_are_a_duplicate() -> None:
    dates = np.array(["2024-01-01", "2024-01-31", "2024-02-29"], dtype="datetime64[D]")
    data = {"date": dates, "sales": np.array([1.0, 2.0, 3.0]), "tv": np.array([1.0, 2.0, 3.0])}
    with pytest.raises(PanelError) as excinfo:
        read(data, geo=None, spend={"tv": "tv"}, controls=(), cadence=Cadence.MONTH)
    (issue,) = errors(excinfo)
    assert issue.code is IssueCode.DUPLICATE
    assert issue.dates == (dt.date(2024, 1, 1), dt.date(2024, 1, 31))


def test_daily_cadence_counts_days() -> None:
    dates = np.datetime64("2024-02-27") + np.timedelta64(1, "D") * np.array([0, 1, 3])
    data = {"date": dates, "sales": np.array([1.0, 2.0, 3.0]), "tv": np.array([1.0, 2.0, 3.0])}
    panel = read(
        data, geo=None, spend={"tv": "tv"}, controls=(), cadence=Cadence.DAY, gaps=Gap.STEP
    )
    assert panel.dates.astype(str).tolist() == [
        "2024-02-27",
        "2024-02-28",
        "2024-02-29",
        "2024-03-01",
    ]
    assert panel.filled[0].tolist() == [False, False, True, False]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"currency": "usd"}, "not a three-letter ISO 4217 code"),
        ({"kpi_unit": Count(" ")}, "needs the name of what it counts"),
        ({"kpi_unit": "revenue"}, "must be Revenue"),
        ({"controls": ("spend_tv",)}, "'spend_tv' is given more than one role"),
        ({"kpi": "revenue"}, "column 'revenue' is not in the data"),
        ({"spend": {"": "spend_tv"}}, "channel name '' is empty"),
    ],
)
def test_the_reading_spec_is_checked(overrides: dict[str, Any], match: str) -> None:
    with pytest.raises(PanelError, match=match):
        read(long_table(), **overrides)


def test_columns_of_different_lengths_are_refused() -> None:
    data = long_table()
    data["price"] = data["price"][:-1]
    with pytest.raises(PanelError, match="columns differ in length"):
        read(data)


def test_every_error_is_reported_at_once() -> None:
    data = without(long_table(), [3])
    data["spend_tv"][0] = -1.0
    data["price"][1] = np.inf
    with pytest.raises(PanelError) as excinfo:
        read(data)
    assert {i.code for i in errors(excinfo)} == {
        IssueCode.NEGATIVE_SPEND,
        IssueCode.NON_FINITE,
        IssueCode.MISSING_PERIOD,
    }
    assert str(excinfo.value).startswith("3 errors reading the panel:")


@pytest.mark.parametrize(
    ("geo_value", "match"),
    [(None, "has no geo at row 4"), ("north ", "spaces at an end")],
)
def test_blank_and_padded_geo_names_are_errors(geo_value: object, match: str) -> None:
    data = long_table()
    data["region"] = data["region"].copy()
    data["region"][4] = geo_value
    with pytest.raises(PanelError, match=match):
        read(data)


def test_a_constant_channel_warns_once_per_geo_or_once_for_all() -> None:
    data = long_table()
    data["spend_tv"][:8] = 0.0
    data["spend_search"][:] = 3.0
    panel = read(data)
    messages = sorted(i.message for i in others(panel.issues))
    assert messages == [
        "channel 'search' spends 3 in every period across all geos: its effect cannot be "
        "identified",
        "channel 'tv' has no spend in geo 'north': its effect cannot be identified from that geo",
    ]
    assert {i.code for i in others(panel.issues)} == {IssueCode.CONSTANT}


def test_a_constant_channel_is_named_by_the_channel_its_column_is_read_as() -> None:
    """Each channel is read from a column of another name: the warning names both, and a constant
    KPI's or control's names no channel, though the control's column bears a channel's name."""
    data = long_table()
    data["spend_tv"][:8] = 0.0
    data["spend_search"][:] = 3.0
    data["search"] = np.full(16, 7.0)
    data["sales"][:] = 150.0
    panel = read(data, controls=("price", "search"))
    assert {(i.channel, i.column, i.geo) for i in others(panel.issues)} == {
        ("tv", "spend_tv", "north"),
        ("search", "spend_search", None),
        (None, "search", None),
        (None, "sales", None),
    }


@pytest.mark.parametrize(
    ("column", "cell", "code"),
    [
        ("spend_tv", -1.0, IssueCode.NEGATIVE_SPEND),
        ("spend_tv", np.nan, IssueCode.NON_FINITE),
        ("spend_tv", "12", IssueCode.NOT_NUMERIC),
        ("spend_tv", 10**400, IssueCode.NON_FINITE),
        ("price", "12", IssueCode.NOT_NUMERIC),
        ("sales", np.inf, IssueCode.NON_FINITE),
    ],
    ids=["negative", "missing", "text", "past float64", "control's text", "infinite KPI"],
)
def test_an_error_in_a_channel_s_spend_names_the_channel(column: str, cell: Any, code: Any) -> None:
    """A spend's error names the channel beside its column; a control's or the KPI's names none."""
    data = long_table()
    values = data[column].astype(object)
    values[3] = cell
    data[column] = values
    with pytest.raises(PanelError) as excinfo:
        read(data)
    (issue,) = errors(excinfo)
    assert (issue.code, issue.column) == (code, column)
    assert issue.channel == ("tv" if column == "spend_tv" else None)


class Frame:
    """The two things a frame library offers that reading relies on, and nothing else."""

    def __init__(self, data: dict[str, np.ndarray]) -> None:
        self._data = data

    @property
    def columns(self) -> list[str]:
        return list(self._data)

    def __getitem__(self, name: str) -> np.ndarray:
        return self._data[name]

    def __iter__(self):  # iterating yields values, as polars does
        return iter(self._data.values())


def test_a_frame_is_read_by_its_columns() -> None:
    data = long_table()
    panel = read(Frame(data))
    np.testing.assert_array_equal(panel.kpi, read(data).kpi)


def test_something_that_is_neither_a_mapping_nor_a_frame_is_a_type_error() -> None:
    with pytest.raises(TypeError, match="got list"):
        read([1, 2, 3])


def frame(data: dict[str, np.ndarray], library: str) -> Any:
    """The table as a pandas or a polars frame, its text in the library's own text type."""
    if library == "pandas":
        return pd.DataFrame(data)
    return pl.DataFrame(
        {
            name: values.tolist() if values.dtype == object else values
            for name, values in data.items()
        }
    )


LIBRARIES = ["pandas", "polars"]


@pytest.mark.parametrize("library", LIBRARIES)
def test_a_pandas_or_polars_frame_reads_as_its_arrays_do(library: str) -> None:
    data = long_table()
    panel, expected = read(frame(data, library)), read(data)
    assert panel.geos == expected.geos
    np.testing.assert_array_equal(panel.dates, expected.dates)
    np.testing.assert_array_equal(panel.kpi, expected.kpi)
    np.testing.assert_array_equal(panel.spend, expected.spend)
    np.testing.assert_array_equal(panel.control_values, expected.control_values)


@pytest.mark.parametrize("library", LIBRARIES)
def test_text_in_a_frame_is_not_read_as_numbers(library: str) -> None:
    data = long_table()
    text = {**data, "sales": np.array([str(v) for v in data["sales"].tolist()], dtype=object)}
    with pytest.raises(PanelError, match="column 'sales' holds text"):
        read(frame(text, library))


@pytest.mark.parametrize("library", LIBRARIES)
def test_a_frames_missing_value_reads_as_missing(library: str) -> None:
    data = long_table()
    sales = [*data["sales"][:5].tolist(), None, *data["sales"][6:].tolist()]
    table = frame(data, library)
    if library == "pandas":
        table["sales"] = pd.array(sales, dtype="Float64")
    else:
        table = table.with_columns(pl.Series("sales", sales))
    panel = read(table)
    (warning,) = others(panel.issues)
    assert warning.code is IssueCode.MISSING_KPI
    assert np.isnan(panel.kpi[0, 5])


@pytest.mark.parametrize(
    ("column", "values"),
    [
        ("spend_tv", START + np.arange(16).astype("timedelta64[D]")),
        ("price", np.arange(16).astype("timedelta64[s]")),
        ("price", np.linspace(5.0, 10.0, 16) + 0j),
        ("spend_tv", np.array([np.timedelta64(5, "D")] * 16, dtype=object)),
        ("price", np.array([np.complex128(5.0 + 1.0j)] * 16, dtype=object)),
        ("sales", np.array([dt.date(2023, 1, 7)] * 16, dtype=object)),
        (
            "spend_tv",
            np.ma.masked_array(
                (START + np.arange(16).astype("timedelta64[D]")).astype("datetime64[ns]"),
                mask=np.arange(16) == 5,
            ),
        ),
    ],
    ids=[
        "dates",
        "durations",
        "complex numbers",
        "NumPy durations as objects",
        "NumPy complex numbers as objects",
        "Python dates",
        "masked nanosecond dates",
    ],
)
def test_dates_durations_and_complex_numbers_are_not_numbers(column: str, values: Any) -> None:
    # NumPy's cast to float64 read a date as its count of days since 1970, a duration as its count
    # of units and a complex number as its real part, the last with only a warning
    data = long_table()
    data[column] = values
    with pytest.raises(
        PanelError, match=f"column {column!r} holds values that are not numbers"
    ) as e:
        read(data)
    assert {i.code for i in errors(e)} == {IssueCode.NOT_NUMERIC}


def test_a_nanosecond_date_column_shows_its_cells_as_dates() -> None:
    # as Python objects, nanosecond dates are integers: the message showed numbers it refused
    data = long_table()
    cells = data["date"].astype("M8[ns]")
    data["spend_tv"] = cells
    with pytest.raises(PanelError) as e:
        read(data)
    text = " ".join(i.message for i in errors(e))
    assert repr(cells[0]) in text
    assert str(cells.astype(np.int64)[0]) not in text


@pytest.mark.parametrize("library", LIBRARIES)
def test_a_frames_date_column_is_not_spend(library: str) -> None:
    data = long_table()
    data["spend_tv"] = data["date"]
    with pytest.raises(PanelError, match="column 'spend_tv' holds values that are not numbers"):
        read(frame(data, library))


WIDER_THAN_A_DOUBLE = pytest.mark.skipif(
    bool(np.finfo(np.longdouble).max <= np.finfo(np.float64).max),
    reason="this platform's long double is a double",
)


@pytest.mark.parametrize(
    "value",
    [
        Decimal("1E+400"),
        10**400,
        Fraction(10**400),
        pytest.param(np.longdouble(1e300) * np.longdouble(1e100), marks=WIDER_THAN_A_DOUBLE),
    ],
    ids=["Decimal", "int", "Fraction", "long double"],
)
def test_a_number_past_float64s_range_is_an_error_of_its_own(value: object) -> None:
    # read as an infinity, the KPI was reported infinite; an int or a Fraction raised OverflowError
    data = long_table()
    sales = data["sales"].astype(object)
    sales[3] = value
    data["sales"] = sales
    with pytest.raises(PanelError, match="column 'sales' holds numbers past float64's range") as e:
        read(data)
    (issue,) = e.value.issues
    assert issue.code is IssueCode.NON_FINITE


@pytest.mark.parametrize(
    "value",
    [float("inf"), np.float64("-inf"), Decimal("Infinity")],
    ids=["float", "NumPy", "Decimal"],
)
def test_an_infinity_is_infinite_not_past_float64s_range(value: object) -> None:
    data = long_table()
    sales = data["sales"].astype(object)
    sales[3] = value
    data["sales"] = sales
    with pytest.raises(PanelError, match="KPI 'sales' is infinite") as e:
        read(data)
    (issue,) = e.value.issues
    assert issue.code is IssueCode.NON_FINITE


@WIDER_THAN_A_DOUBLE
def test_a_long_double_column_past_float64s_range_is_an_error_of_its_own() -> None:
    data = long_table()
    sales = data["sales"].astype(np.longdouble)
    sales[3] = np.longdouble(1e300) * np.longdouble(1e100)
    data["sales"] = sales
    with pytest.raises(PanelError, match="column 'sales' holds numbers past float64's range") as e:
        read(data)
    (issue,) = e.value.issues
    assert issue.code is IssueCode.NON_FINITE


@pytest.mark.parametrize(
    ("column", "match"),
    [
        ("spend_tv", "write 0 where nothing was spent"),
        ("date", "has no date at row 5"),
        ("region", "has no geo at row 5"),
    ],
)
def test_a_masked_cell_is_a_missing_one(column: str, match: str) -> None:
    # np.asarray dropped the mask, which read the value under it
    data = long_table()
    data["date"] = data["date"].astype("datetime64[ns]")  # as objects, integers
    data[column] = np.ma.masked_array(data[column], mask=np.arange(16) == 5)
    with pytest.raises(PanelError, match=match) as e:
        read(data)
    assert len(errors(e)) == 1


def test_a_masked_kpi_is_a_missing_kpi() -> None:
    data = long_table()
    data["sales"] = np.ma.masked_array(data["sales"], mask=np.arange(16) == 5)
    panel = read(data)
    (warning,) = others(panel.issues)
    assert warning.code is IssueCode.MISSING_KPI
    assert np.isnan(panel.kpi[0, 5])


# ------------------------------------------------------------------------- the input contracts


def _shop(weeks: int = 60, seed: int = 3) -> dict[str, np.ndarray]:
    """One market's weeks of three channels, a KPI moved by each, every spend distinct."""
    rng = np.random.default_rng(seed)
    spend = {f"spend_{c}": rng.gamma(2.0, 40.0, weeks) for c in ("tv", "search", "social")}
    sales = 500.0 + 0.8 * spend["spend_tv"] + 1.5 * spend["spend_search"] + rng.normal(0, 5, weeks)
    return {
        "date": START + WEEK * np.arange(weeks),
        "sales": sales + spend["spend_social"],
        **spend,
    }


def _shop_panel(data: Any, channels: tuple[str, ...]) -> MediaPanel:
    return read(data, geo=None, controls=(), spend={c: f"spend_{c}" for c in channels})


def _shop_returns(panel: MediaPanel) -> dict[str, Any]:
    """Each channel's returns under the ridge engine and the DLM, each model naming its channels
    in one order whatever the panel's."""
    names = ("tv", "search", "social")
    ridge = fit_ridge(
        panel,
        NATIONAL,
        RidgeSpec(tuple(Media(c, Geometric(0.3), Hill(1.0, 0.5)) for c in names), (), 1e-3),
    )
    transform = Transform(0.3, 60.0, HillShape(1.0), 4)
    dlm = fit_dlm(
        panel, NATIONAL, DlmSpec(tuple(Effect(c, 0.99, transform=transform) for c in names))
    )
    out: dict[str, Any] = {}
    for engine, channels in (("ridge", ridge.channels()), ("dlm", dlm.channels())):
        for name, channel in channels.items():
            spend = panel.spend[0, :, panel.channels.index(Channel(name))]
            out[f"{engine}:{name}"] = returns(channel, spend)
    return out


def test_the_data_s_column_order_moves_nothing() -> None:
    data = _shop()
    shuffled = {name: data[name] for name in reversed(list(data))}
    panel, again = (
        _shop_panel(data, ("tv", "search", "social")),
        _shop_panel(shuffled, ("tv", "search", "social")),
    )
    for name in ("dates", "kpi", "spend", "control_values", "filled"):
        assert getattr(panel, name).tobytes() == getattr(again, name).tobytes()
    assert again.channels == panel.channels


@pytest.mark.parametrize("order", [("social", "tv", "search"), ("search", "social", "tv")])
def test_permuting_the_spend_columns_leaves_every_return_unchanged(
    order: tuple[str, ...],
) -> None:
    data = _shop()
    panel, permuted = _shop_panel(data, ("tv", "search", "social")), _shop_panel(data, order)
    assert permuted.channels == order
    for c in order:
        j = permuted.channels.index(Channel(c))
        np.testing.assert_array_equal(permuted.spend[0, :, j], data[f"spend_{c}"])
    assert _shop_returns(permuted) == _shop_returns(panel)


def _daily(days: int = 90) -> dict[str, np.ndarray]:
    data = _shop(days)
    data["date"] = START + np.timedelta64(1, "D") * np.arange(days)
    return data


@pytest.mark.parametrize(
    ("data", "cadence", "match"),
    [
        (
            _daily(),
            Cadence.WEEK,
            "the dates are daily, 1 day apart at the median, finer than the weekly cadence: "
            "resample the data to weeks before reading it, summing the KPI and the spend over "
            "each of its weeks; reading does not resample",
        ),
        (_daily(), Cadence.MONTH, "finer than the monthly cadence: resample the data to months"),
        (_shop(), Cadence.MONTH, "the dates are weekly, .* finer than the monthly cadence"),
        (_shop(), Cadence.DAY, "weekly, 7 days apart .* coarser .* read them as Cadence.WEEK"),
    ],
    ids=["daily as weekly", "daily as monthly", "weekly as monthly", "weekly as daily"],
)
def test_dates_spaced_unlike_the_cadence_are_refused_not_resampled(
    data: dict[str, np.ndarray], cadence: Cadence, match: str
) -> None:
    with pytest.raises(PanelError, match=match) as e:
        read(data, geo=None, controls=(), spend={"tv": "spend_tv"}, cadence=cadence)
    (issue,) = e.value.issues
    assert issue.code is IssueCode.CADENCE
    assert issue.column == "date"


def test_a_weekly_table_with_a_stray_date_is_still_read_on_its_weeks() -> None:
    data = long_table(weeks=20)
    data["date"] = data["date"].copy()
    data["date"][3] += np.timedelta64(2, "D")
    with pytest.raises(PanelError) as e:
        read(data)
    assert {i.code for i in errors(e)} == {IssueCode.OFF_GRID, IssueCode.MISSING_PERIOD}


def test_a_population_column_is_read_once_a_geo() -> None:
    data = long_table(weeks=50)
    data["people"] = np.repeat([2e5, 3e5], 50)
    panel = read(data, population="people")
    assert panel.population is not None
    np.testing.assert_array_equal(panel.population, [2e5, 3e5])


def test_a_population_that_changes_within_a_geo_is_refused() -> None:
    data = long_table(weeks=50)
    data["people"] = np.repeat([2e5, 3e5], 50)
    data["people"][60] = 3.1e5
    with pytest.raises(PanelError, match="takes 2 values in geo 'south' \\(300000, 310000\\)") as e:
        read(data, population="people")
    (issue,) = errors(e)
    assert (issue.code, issue.geo, issue.column) == (IssueCode.POPULATION, "south", "people")
    assert "a geo's population is one number over the calendar" in issue.message


def test_a_population_given_as_a_series_is_refused() -> None:
    with pytest.raises(PanelError, match="'north' is given as 50 values: a geo's population") as e:
        read(long_table(weeks=50), population={"north": np.full(50, 2e5), "south": 3e5})
    (issue,) = errors(e)
    assert issue.code is IssueCode.POPULATION


def test_a_geo_whose_population_is_none_has_no_population() -> None:
    with pytest.raises(PanelError, match="geo 'north' has no population") as e:
        read(long_table(weeks=50), population={"north": None, "south": 3e5})
    (issue,) = errors(e)
    assert issue.code is IssueCode.POPULATION


def test_too_few_observations_an_effect_warn_for_the_panel_or_each_geo() -> None:
    effects = 3  # two channels and a control
    panel = read(long_table(weeks=15 * effects))
    assert others(panel.issues) == panel.issues
    data = long_table(weeks=15 * effects)
    data["sales"][: 15 * effects][:2] = np.nan
    (warning,) = others(read(data).issues) or (None,)
    assert warning is not None
    assert warning.code is IssueCode.MISSING_KPI
    (few,) = [i for i in read(data).issues if i.code is IssueCode.FEW_OBSERVATIONS]
    assert (few.severity, few.geo) == (Severity.WARNING, "north")
    assert few.message == (
        "geo 'north' observes 43 weeks of the KPI, fewer than the 45 that 15 for each of its 3 "
        "channels and controls take: its effects are barely identified, and their intervals lean "
        "on the priors"
    )
    (every,) = read(long_table(weeks=10)).issues
    assert every.geo is None
    assert every.message.startswith("every geo observes at most 10 weeks of the KPI, fewer than")
