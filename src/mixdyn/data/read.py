"""Read a long table, one row per geo and period and one column per channel, into a MediaPanel.

Reading is where data is checked. It reports every problem it finds rather than stopping at the
first: `PanelError` carries them all as `Issue`s, each naming its geo and dates. Problems it does
not refuse become warnings, both on the panel and in the log.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from mixdyn.data.issues import Issue, IssueCode, PanelError, Severity
from mixdyn.data.panel import (
    NATIONAL,
    Cadence,
    Channel,
    Geo,
    KpiUnit,
    Lever,
    MediaPanel,
    hierarchy_problems,
    read_events,
    unit_problems,
)

log = logging.getLogger(__name__)

_SHOWN = 6
"""How many dates, runs or values one message lists before it counts the rest."""

_SPAN_LIMIT = 10
"""A calendar longer than this many times the best-covered geo's row count is refused as a typo."""

_POINTS_PER_EFFECT = 15
"""The fewest observed periods per effect, channel or control, a geo is read without a warning."""

_FIRST_DAY = np.datetime64(dt.date.min, "D")
_LAST_DAY = np.datetime64(dt.date.max, "D")
"""The days a Python date holds, years 1 to 9999."""

_SPACING = {
    Cadence.DAY: (0, "daily", "days"),
    Cadence.WEEK: (1, "weekly", "weeks"),
    Cadence.MONTH: (2, "monthly", "months"),
}
"""Each cadence's rank from the finest, its adjective and its periods' name."""


class ColumnFrame(Protocol):
    """A frame with named columns, indexable by name: pandas, polars, pyarrow and the like."""

    @property
    def columns(self) -> Sequence[str]: ...

    def __getitem__(self, name: str, /) -> Any: ...


type ColumnData = Mapping[str, Any] | ColumnFrame
"""The data `read_panel` reads: a mapping of column name to values, or a frame with named
columns (`ColumnFrame`)."""


class Gap(Enum):
    """What a period becomes when a geo has no row for it.

    - `ERROR`: reading fails, naming each geo and the periods it lacks.
    - `STEP`: the period stays as a step with no KPI observation, zero spend in every channel and
      NaN controls, and reading warns. The zero is a claim about the missing period, and adstock
      carries it into the periods after: choose `STEP` only where nothing was spent, or where
      that does not matter.
    """

    ERROR = "error"
    STEP = "step"


def read_panel(
    data: ColumnData,
    *,
    date: str,
    kpi: str,
    spend: Mapping[str, str],
    cadence: Cadence,
    kpi_unit: KpiUnit,
    currency: str,
    geo: str | None = None,
    controls: Sequence[str] = (),
    hierarchy: Mapping[str, str] | None = None,
    population: Mapping[str, float] | str | None = None,
    gaps: Gap = Gap.ERROR,
    levers: Mapping[str, Lever] | None = None,
    events: Mapping[str, npt.ArrayLike] | None = None,
) -> MediaPanel:
    """Check one row per geo and period, then read it into a `MediaPanel`.

    `data` is a mapping of column name to values, or a frame with `.columns` (pandas, polars).
    `spend` maps each channel's name to its spend column; the panel keeps that order. Geos are
    sorted. Without `geo` the panel has one geo, `NATIONAL`. `hierarchy` maps every geo, and every
    region above one, to its parent. Nodes with no geo of the data below them are dropped, with a
    warning. `population` maps every geo to its population, one number a geo, and a geo not in the
    data is dropped, with a warning; or it names a column that holds each row's geo's population,
    which must be the same on every row of a geo. `levers` declares some of `controls` levers,
    each with the level its contribution is measured against (`Lever`). `events` names calendar
    events, holidays among them, each by its days, which must fall inside the calendar read.

    Channels are read by name: the order of the data's columns moves nothing, and the panel holds
    the channels in `spend`'s order with each channel's own column. The dates must be spaced as
    `cadence` says: daily dates read at a weekly cadence are refused, not resampled, so data finer
    than the cadence is summed to it before reading. A geo with fewer than 15 observed periods per
    channel and control warns that its effects are barely identified.

    Raises `PanelError` listing every error found. Warnings ride on the panel's `issues` and go to
    this module's logger.
    """
    found: list[Issue] = []
    people_column = population if isinstance(population, str) else None
    roles = [
        date,
        kpi,
        *([geo] if geo is not None else []),
        *spend.values(),
        *controls,
        *([people_column] if people_column is not None else []),
    ]
    found += [
        _error(IssueCode.BAD_SPEC, f"column {name!r} is given more than one role", column=name)
        for name, uses in Counter(roles).items()
        if uses > 1
    ]
    found += [
        _error(IssueCode.BAD_SPEC, f"channel name {name!r} is empty or not a string")
        for name in spend
        if not isinstance(name, str) or not name.strip()
    ]
    found += [_error(IssueCode.UNIT, p) for p in unit_problems(kpi_unit, currency)]
    found += [
        _error(IssueCode.BAD_SPEC, f"lever {name!r} is not one of the controls, {list(controls)}")
        for name in levers or {}
        if name not in controls
    ]
    found += [
        _error(IssueCode.BAD_SPEC, f"lever {name!r}: {lever!r} is not a Lever", column=name)
        for name, lever in (levers or {}).items()
        if not isinstance(lever, Lever)
    ]
    columns, absent = _select(data, list(dict.fromkeys(roles)))
    found += [
        _error(IssueCode.MISSING_COLUMN, f"column {name!r} is not in the data", column=name)
        for name in absent
    ]
    found += _shape_issues(columns)
    _raise_on_errors(found)

    days, date_issues = _read_dates(columns[date], date)
    if geo is None:
        geo_names = np.full(days.shape, NATIONAL, dtype=object)
        geo_issues = []
    else:
        geo_names, geo_issues = _read_geos(columns[geo], geo)
    found += date_issues + geo_issues
    _raise_on_errors(found)
    found += _cadence_issues(days, geo_names, cadence, column=date)
    _raise_on_errors(found)

    rows = _Rows(geo_names, days, national=geo is None)
    kpi_values, spend_values, control_values, value_issues = _read_values(
        columns, rows, kpi=kpi, spend=spend, controls=controls
    )
    found += value_issues

    keys, label, off_grid, grid_note = _period_keys(days, cadence)
    found += rows.issues(off_grid, IssueCode.OFF_GRID, f"dates are off the grid {grid_note}")
    on_grid = ~off_grid
    unique_geos, geo_index = np.unique(geo_names, return_inverse=True)
    geo_index = geo_index.reshape(-1)
    names = tuple(Geo(str(g)) for g in unique_geos)
    first, last = int(keys[on_grid].min()), int(keys[on_grid].max())
    span = last - first + 1
    busiest = int(np.bincount(geo_index[on_grid]).max())
    if span > _SPAN_LIMIT * busiest:
        found.append(
            _error(
                IssueCode.BAD_DATE,
                f"the dates span {span} {cadence.value} periods, from {_iso(label(first))} to "
                f"{_iso(label(last))}, but no geo has more than {busiest} rows: check the "
                "earliest and the latest dates",
                column=date,
            )
        )
        _raise_on_errors(found)
    cell = geo_index * span + (keys - first)
    _, inverse, counts = np.unique(cell[on_grid], return_inverse=True, return_counts=True)
    repeated = np.zeros(days.shape, dtype=bool)
    repeated[on_grid] = counts[inverse.reshape(-1)] > 1
    hint = "" if geo is not None else "; if the data has one row per geo, name its geo column"
    found += rows.issues(repeated, IssueCode.DUPLICATE, f"more than one row per period{hint}")

    present = np.zeros((len(names), span), dtype=bool)
    present[geo_index[on_grid], keys[on_grid] - first] = True
    dates = label(np.arange(first, last + 1))
    found += _missing_periods(present, dates, names, national=geo is None, gaps=gaps)
    event_days, event_problems = read_events(events or {}, (*spend, *controls), dates, cadence)
    found += [_error(IssueCode.BAD_SPEC, p) for p in event_problems]
    parent, hierarchy_issues = _read_hierarchy(names, hierarchy or {})
    found += hierarchy_issues
    given = population if not isinstance(population, str) else None
    if people_column is not None:
        given, column_issues = _population_column(
            columns[people_column], people_column, rows, names, geo_index
        )
        found += column_issues
    people, population_issues = _read_population(names, given)
    found += population_issues
    _raise_on_errors(found)

    filled = ~present
    t = keys - first
    kpi_grid = np.full(present.shape, np.nan)
    kpi_grid[geo_index, t] = kpi_values
    spend_grid = np.zeros((*present.shape, len(spend)))
    for c, values in enumerate(spend_values):
        spend_grid[geo_index, t, c] = values
    control_grid = np.full((*present.shape, len(controls)), np.nan)
    for k, values in enumerate(control_values):
        control_grid[geo_index, t, k] = values
    channels = tuple(Channel(name) for name in spend)
    found += _constant_warnings(
        kpi_grid, spend_grid, control_grid, present, names, kpi=kpi, spend=spend, controls=controls
    )
    found += _few_observations(kpi_grid, names, len(spend) + len(controls), cadence)
    for issue in found:
        log.warning("%s", issue.message)
    log.info(
        "read %d rows into %d geos x %d %s periods x %d channels; %d steps filled",
        days.size,
        len(names),
        span,
        cadence.value,
        len(channels),
        int(filled.sum()),
    )
    return MediaPanel(
        dates=dates,
        cadence=cadence,
        geos=names,
        channels=channels,
        controls=tuple(controls),
        kpi=kpi_grid,
        kpi_unit=kpi_unit,
        spend=spend_grid,
        currency=currency,
        control_values=control_grid,
        filled=filled,
        hierarchy=parent,
        population=people,
        issues=tuple(found),
        levers=dict(levers or {}),
        events=event_days,
    )


@dataclass(frozen=True, slots=True)
class _Rows:
    """Locates input rows by geo and date, to turn a row mask into one issue per geo."""

    geos: npt.NDArray[np.object_]
    days: npt.NDArray[np.datetime64]
    national: bool

    def issues(
        self,
        mask: npt.NDArray[np.bool_],
        code: IssueCode,
        what: str,
        *,
        severity: Severity = Severity.ERROR,
        column: str | None = None,
        channel: str | None = None,
        detail: Callable[[npt.NDArray[np.intp]], str] = lambda _: "",
    ) -> list[Issue]:
        hits = np.flatnonzero(mask)
        if hits.size == 0:
            return []
        hits = hits[np.argsort(self.geos[hits], kind="stable")]
        groups, starts = np.unique(self.geos[hits], return_index=True)
        found = []
        for name, group in zip(groups, np.split(hits, starts[1:]), strict=True):
            dates = tuple(sorted(_as_dates(self.days[group])))
            where = "" if self.national else f" in geo {name!r}"
            message = f"{what}{where} on {_listed(dates)}{detail(group)}"
            found.append(
                Issue(
                    code,
                    severity,
                    message,
                    geo=None if self.national else str(name),
                    column=column,
                    dates=dates,
                    channel=channel,
                )
            )
        return found


def _error(code: IssueCode, message: str, *, column: str | None = None) -> Issue:
    return Issue(code, Severity.ERROR, message, column=column)


def _raise_on_errors(found: list[Issue]) -> None:
    if any(issue.severity is Severity.ERROR for issue in found):
        raise PanelError(tuple(found))


def _select(data: ColumnData, names: list[str]) -> tuple[dict[str, np.ndarray], list[str]]:
    """The named columns as NumPy arrays, and the names the data does not have.

    A mapping is told from a frame by `isinstance`, not by duck typing: iterating a polars frame
    yields its columns as values, so mapping idioms on a frame hand back data where names were
    asked for instead of failing.
    """
    if isinstance(data, Mapping):
        available = {str(name) for name in data}
    else:
        frame_columns = getattr(data, "columns", None)
        if frame_columns is None:
            msg = (
                "expected a mapping of column name to values, or a frame with `.columns`, "
                f"got {type(data).__name__}"
            )
            raise TypeError(msg)
        available = {str(name) for name in frame_columns}
    absent = [name for name in names if name not in available]
    columns = {name: _unmasked(data[name]) for name in names if name in available}
    return columns, absent


def _unmasked(values: Any) -> np.ndarray:
    """The values as an array, a masked cell as a missing one: `np.asarray` drops the mask, which
    reads the value under it."""
    if not isinstance(values, np.ma.MaskedArray):
        return np.asarray(values)
    masked = np.ma.getmaskarray(values)
    data = np.ma.getdata(values)
    if not masked.any():
        return data
    if data.dtype.kind in "Mm":  # as objects, nanosecond dates would turn into integers
        cells = data.copy()
        cells[masked] = np.array("NaT", dtype=data.dtype)
        return cells
    cells = data.astype(object)
    cells[masked] = None
    return cells


def _shape_issues(columns: dict[str, np.ndarray]) -> list[Issue]:
    found = [
        _error(IssueCode.BAD_SPEC, f"column {name!r} is not one-dimensional", column=name)
        for name, values in columns.items()
        if values.ndim != 1
    ]
    lengths = {name: len(values) for name, values in columns.items() if values.ndim == 1}
    if len(set(lengths.values())) > 1:
        found.append(_error(IssueCode.BAD_SPEC, f"columns differ in length: {lengths}"))
    elif set(lengths.values()) == {0}:
        found.append(_error(IssueCode.BAD_SPEC, "the data has no rows"))
    return found


def _is_missing(value: object) -> bool:
    """None, a float NaN, or a frame library's missing marker, recognised without importing one."""
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return type(value).__name__ in {"NAType", "NaTType"}


def _read_dates(values: np.ndarray, column: str) -> tuple[npt.NDArray[np.datetime64], list[Issue]]:
    """Calendar days, refusing what is not a date, is missing, or carries a time of day."""
    if values.dtype.kind in "biuf":
        message = (
            f"column {column!r} holds numbers, not dates; convert them first (Excel serial days "
            "and yyyymmdd integers read differently)"
        )
        nat = np.full(values.shape, np.datetime64("NaT", "D"))
        return nat, [_error(IssueCode.BAD_DATE, message, column=column)]
    unparsed = np.zeros(values.shape, dtype=bool)
    try:
        fine = values if values.dtype.kind == "M" else values.astype("datetime64[us]")
    except (ValueError, TypeError, OverflowError):
        fine = np.full(values.shape, np.datetime64("NaT", "us"))
        for i, value in enumerate(values.tolist()):
            try:
                fine[i] = np.datetime64(value, "us")
            except (ValueError, TypeError, OverflowError):
                unparsed[i] = True
    days = fine.astype("datetime64[D]")
    # numpy reads years no Python date holds ("-4" is the year -4), and every issue lists its dates
    # as Python's: such a value is not a date here
    beyond = ~np.isnat(days) & ((days < _FIRST_DAY) | (days > _LAST_DAY))
    if beyond.any():
        unparsed |= beyond
        fine = np.where(beyond, np.array("NaT", dtype=fine.dtype), fine)
        days = fine.astype("datetime64[D]")
    found = []
    if unparsed.any():
        found.append(
            _error(
                IssueCode.BAD_DATE,
                f"column {column!r} has values that are not dates: {_positions(unparsed, values)}",
                column=column,
            )
        )
    blank = np.isnat(fine) & ~unparsed
    if blank.any():
        found.append(
            _error(
                IssueCode.BAD_DATE,
                f"column {column!r} has no date {_positions(blank)}",
                column=column,
            )
        )
    timed = ~np.isnat(fine) & (days.astype(fine.dtype) != fine)
    if timed.any():
        found.append(
            _error(
                IssueCode.BAD_DATE,
                f"column {column!r} carries a time of day, one date per period expected: "
                f"{_positions(timed, values)}",
                column=column,
            )
        )
    return days, found


def _read_geos(values: np.ndarray, column: str) -> tuple[npt.NDArray[np.object_], list[Issue]]:
    names = np.empty(values.shape, dtype=object)
    blank = np.zeros(values.shape, dtype=bool)
    padded = np.zeros(values.shape, dtype=bool)
    for i, value in enumerate(values.tolist()):
        text = "" if _is_missing(value) else str(value)
        names[i] = text
        blank[i] = not text.strip()
        padded[i] = text != text.strip()
    found = []
    if blank.any():
        found.append(
            _error(
                IssueCode.BAD_GEO,
                f"column {column!r} has no geo {_positions(blank)}",
                column=column,
            )
        )
    if (padded & ~blank).any():
        found.append(
            _error(
                IssueCode.BAD_GEO,
                f"column {column!r} has geo names with spaces at an end, which would read as "
                f"other geos: {_positions(padded & ~blank, values)}",
                column=column,
            )
        )
    return names, found


def _read_values(
    columns: dict[str, np.ndarray],
    rows: _Rows,
    *,
    kpi: str,
    spend: Mapping[str, str],
    controls: Sequence[str],
) -> tuple[
    npt.NDArray[np.float64],
    list[npt.NDArray[np.float64]],
    list[npt.NDArray[np.float64]],
    list[Issue],
]:
    """The KPI, spend and control values row by row, and what is wrong with them.

    A missing KPI is a warning; a missing spend or control is an error.
    """
    kpi_values, unread, found = _read_numbers(columns[kpi], kpi, rows)
    found += rows.issues(
        np.isinf(kpi_values), IssueCode.NON_FINITE, f"KPI {kpi!r} is infinite", column=kpi
    )
    found += rows.issues(
        np.isnan(kpi_values) & ~unread,
        IssueCode.MISSING_KPI,
        f"KPI {kpi!r} has no value",
        severity=Severity.WARNING,
        column=kpi,
    )
    spend_values = []
    for channel, column in spend.items():
        values, unread, issues = _read_numbers(columns[column], column, rows, channel=channel)
        spend_values.append(values)
        found += issues
        found += rows.issues(
            ~np.isfinite(values) & ~unread,
            IssueCode.NON_FINITE,
            f"spend on {channel!r} ({column!r}) is missing or infinite; write 0 where nothing "
            "was spent",
            column=column,
            channel=channel,
        )
        found += rows.issues(
            values < 0,
            IssueCode.NEGATIVE_SPEND,
            f"spend on {channel!r} ({column!r}) is negative",
            column=column,
            channel=channel,
            detail=lambda r, v=values: f" (lowest {v[r].min():g})",
        )
    control_values = []
    for column in controls:
        values, unread, issues = _read_numbers(columns[column], column, rows)
        control_values.append(values)
        found += issues
        found += rows.issues(
            ~np.isfinite(values) & ~unread,
            IssueCode.NON_FINITE,
            f"control {column!r} is missing or infinite",
            column=column,
        )
    return kpi_values, spend_values, control_values, found


def _constant_warnings(
    kpi_grid: npt.NDArray[np.float64],
    spend_grid: npt.NDArray[np.float64],
    control_grid: npt.NDArray[np.float64],
    present: npt.NDArray[np.bool_],
    geos: tuple[Geo, ...],
    *,
    kpi: str,
    spend: Mapping[str, str],
    controls: Sequence[str],
) -> list[Issue]:
    found = _constant(
        kpi_grid, present, geos, f"KPI {kpi!r}", "there is nothing to explain", column=kpi
    )
    for c, (channel, column) in enumerate(spend.items()):
        found += _constant(
            spend_grid[..., c],
            present,
            geos,
            f"channel {channel!r}",
            "its effect cannot be identified",
            column=column,
            channel=channel,
            describe=lambda v: "has no spend" if v == 0 else f"spends {v:g} in every period",
        )
    for k, column in enumerate(controls):
        found += _constant(
            control_grid[..., k],
            present,
            geos,
            f"control {column!r}",
            "it cannot be told apart from the level",
            column=column,
        )
    return found


def _read_numbers(
    values: np.ndarray, column: str, rows: _Rows, *, channel: str | None = None
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_], list[Issue]]:
    """float64 values, which cells held something other than a number, and the errors that makes.

    A missing value (None, NaN, NaT, a frame's NA) becomes NaN without an error here: whether the
    column may miss values is the caller's decision. Text is not a number, though it may spell one:
    where thousands are written with a point, "1.250" is 1250, which `float` reads as 1.25, so the
    caller converts text where its locale is known. A date, a duration and a complex number are not
    numbers, though NumPy's cast to float64 reads a date as its count of days since 1970, a duration
    as its count of units and a complex number as its real part. A number past float64's range is
    an error of its own, which the cast reads as an infinity and `float` refuses with
    `OverflowError`.
    """
    numbers = np.full(values.shape, np.nan)
    bad = np.zeros(values.shape, dtype=bool)
    text = np.zeros(values.shape, dtype=bool)
    past = np.zeros(values.shape, dtype=bool)
    kind = values.dtype.kind
    if kind in "biuf":
        with np.errstate(over="ignore"):  # an overflow is told apart below
            numbers = values.astype(np.float64)
        if kind == "f":  # a long double can hold what a double cannot
            past = np.isinf(numbers) & np.isfinite(values)
            numbers[past] = np.nan
    elif kind in "OSUT":
        for i, value in enumerate(values.tolist()):
            if _is_missing(value):
                continue
            if isinstance(value, str | bytes):
                text[i] = True
                continue
            if isinstance(value, np.complexfloating):  # float reads its real part
                bad[i] = True
                continue
            try:
                number = float(value)
            except OverflowError:
                past[i] = True
                continue
            except (ValueError, TypeError):
                bad[i] = True
                continue
            if math.isinf(number) and _finite_where_it_came_from(value):
                past[i] = True
            else:
                numbers[i] = number
    else:  # dates, durations, complex numbers and structured records
        bad[:] = True
    issues = rows.issues(
        bad,
        IssueCode.NOT_NUMERIC,
        f"column {column!r} holds values that are not numbers",
        column=column,
        channel=channel,
        detail=lambda r: f" ({_sample(values[r])})",
    )
    issues += rows.issues(
        text,
        IssueCode.NOT_NUMERIC,
        f"column {column!r} holds text",
        column=column,
        channel=channel,
        detail=lambda r: (
            f" ({_sample(values[r])}), which is not read as numbers: convert it to numbers where "
            "its locale is known"
        ),
    )
    issues += rows.issues(
        past,
        IssueCode.NON_FINITE,
        f"column {column!r} holds numbers past float64's range",
        column=column,
        channel=channel,
        detail=lambda r: f" ({_sample(values[r])})",
    )
    return numbers, bad | text | past, issues


def _finite_where_it_came_from(value: object) -> bool:
    """Whether a value that `float` read as an infinity is finite in its own type."""
    if isinstance(value, Decimal):
        return value.is_finite()
    if isinstance(value, np.floating):
        return bool(np.isfinite(value))
    return False


def _period_keys(
    days: npt.NDArray[np.datetime64], cadence: Cadence
) -> tuple[
    npt.NDArray[np.int64],
    Callable[[Any], npt.NDArray[np.datetime64]],
    npt.NDArray[np.bool_],
    str,
]:
    """Each date's period number, the label of a period number, which dates are off the grid, and
    the grid's description for messages.

    Weekly periods sit on the weekday most dates share; the smallest such weekday wins a tie, so
    the grid does not depend on row order.
    """
    epoch_days = days.astype(np.int64)
    if cadence is Cadence.MONTH:
        months = days.astype("datetime64[M]").astype(np.int64)
        return (
            months,
            lambda k: np.asarray(k).astype("datetime64[M]").astype("datetime64[D]"),
            np.zeros(days.shape, dtype=bool),
            "of months",
        )
    if cadence is Cadence.DAY:
        return (
            epoch_days,
            lambda k: np.asarray(k).astype("datetime64[D]"),
            np.zeros(days.shape, dtype=bool),
            "of days",
        )
    weekday = int(np.argmax(np.bincount(epoch_days % 7, minlength=7)))
    off_grid = epoch_days % 7 != weekday
    name = np.datetime64(weekday, "D").item().strftime("%A")
    return (
        (epoch_days - weekday) // 7,
        lambda k: (7 * np.asarray(k) + weekday).astype("datetime64[D]"),
        off_grid,
        f"of weeks starting on {name}",
    )


def _missing_periods(
    present: npt.NDArray[np.bool_],
    dates: npt.NDArray[np.datetime64],
    geos: tuple[Geo, ...],
    *,
    national: bool,
    gaps: Gap,
) -> list[Issue]:
    severity = Severity.ERROR if gaps is Gap.ERROR else Severity.WARNING
    outcome = "" if gaps is Gap.ERROR else "; kept as steps with no KPI and zero spend"
    found = []
    for g, name in enumerate(geos):
        lacking = np.flatnonzero(~present[g])
        if lacking.size == 0:
            continue
        subject = "the data" if national else f"geo {name!r}"
        found.append(
            Issue(
                IssueCode.MISSING_PERIOD,
                severity,
                f"{subject} has no row for {lacking.size} of {dates.size} periods: "
                f"{_runs(lacking, dates)}{outcome}",
                geo=None if national else name,
                dates=tuple(_as_dates(dates[lacking])),
            )
        )
    return found


def _cadence_issues(
    days: npt.NDArray[np.datetime64],
    geos: npt.NDArray[np.object_],
    cadence: Cadence,
    *,
    column: str,
) -> list[Issue]:
    """An error where the dates are spaced finer or coarser than `cadence`.

    The dates are taken to be spaced otherwise where most rows miss the cadence's grid: most
    dates are off the weekday the most share at a weekly cadence, share their geo's month with
    another row at a monthly one, or most of the days between a geo's first and last are missing
    at a daily one. Their spacing is then the median gap between consecutive distinct dates of a
    geo, over every geo: under 7 days daily, under 28 weekly, and monthly from there. A few
    missing or misplaced dates leave the grid's own checks to name them.
    """
    epoch = days.astype(np.int64)
    gaps, missed = [], 0
    for name in np.unique(geos):
        mine = epoch[geos == name]
        seen = np.unique(mine)
        gaps.append(np.diff(seen))
        if cadence is Cadence.MONTH:
            _, counts = np.unique(days[geos == name].astype("datetime64[M]"), return_counts=True)
            missed += int(counts[counts > 1].sum())
        elif cadence is Cadence.DAY:
            missed += int(seen[-1] - seen[0] + 1 - seen.size)
    if cadence is Cadence.WEEK:
        missed = epoch.size - int(np.bincount(epoch % 7, minlength=7).max())
    gap = np.concatenate(gaps)
    if gap.size == 0 or not 2 * missed > epoch.size:
        return []
    median = float(np.median(gap))
    rank, adjective, periods = _SPACING[cadence]
    seen_rank = 0 if median < 7 else 1 if median < 28 else 2
    if seen_rank == rank:
        return []
    found = next(c for c, (r, _, _) in _SPACING.items() if r == seen_rank)
    spacing = (
        f"the dates are {_SPACING[found][1]}, {median:g} day{'' if median == 1 else 's'} apart at "
        "the median"
    )
    if seen_rank < rank:
        message = (
            f"{spacing}, finer than the {adjective} cadence: resample the data to {periods} "
            f"before reading it, summing the KPI and the spend over each of its {periods}; "
            "reading does not resample"
        )
    else:
        message = f"{spacing}, coarser than the {adjective} cadence: read them as {found}"
    return [_error(IssueCode.CADENCE, message, column=column)]


def _population_column(
    values: np.ndarray,
    column: str,
    rows: _Rows,
    geos: tuple[Geo, ...],
    geo_index: npt.NDArray[np.intp],
) -> tuple[dict[str, float] | None, list[Issue]]:
    """Each geo's population as its rows give it, and the errors that makes: a value that is not
    a finite number, and a geo whose rows disagree, which are refused rather than averaged."""
    numbers, unread, found = _read_numbers(values, column, rows)
    found += rows.issues(
        ~np.isfinite(numbers) & ~unread,
        IssueCode.POPULATION,
        f"population {column!r} is missing or infinite",
        column=column,
    )
    people: dict[str, float] = {}
    for g, name in enumerate(geos):
        seen = np.unique(numbers[(geo_index == g) & np.isfinite(numbers)])
        if seen.size > 1:
            shown = ", ".join(f"{v:g}" for v in seen[:_SHOWN])
            more = "" if seen.size <= _SHOWN else f" and {seen.size - _SHOWN} more"
            message = (
                f"population {column!r} takes {seen.size} values in geo {name!r} ({shown}{more}): "
                "a geo's population is one number over the calendar; give each geo one, or read "
                "a population that changes as a control"
            )
            found.append(
                Issue(IssueCode.POPULATION, Severity.ERROR, message, geo=name, column=column)
            )
        elif seen.size == 1:
            people[name] = float(seen[0])
    if any(issue.severity is Severity.ERROR for issue in found):
        return None, found
    return people, found


def _few_observations(
    kpi: npt.NDArray[np.float64], geos: tuple[Geo, ...], effects: int, cadence: Cadence
) -> list[Issue]:
    """Warn where a geo observes the KPI in fewer than `_POINTS_PER_EFFECT` periods per effect,
    once for the panel where every geo does."""
    if effects == 0:
        return []
    observed = np.isfinite(kpi).sum(axis=1)
    few = [(name, int(n)) for name, n in zip(geos, observed, strict=True)]
    few = [(name, n) for name, n in few if n < _POINTS_PER_EFFECT * effects]
    if not few:
        return []
    _, _, periods = _SPACING[cadence]
    need = (
        f"fewer than the {_POINTS_PER_EFFECT * effects} that {_POINTS_PER_EFFECT} for each of its "
        f"{effects} channels and controls take: its effects are barely identified, and their "
        "intervals lean on the priors"
    )
    if len(few) == len(geos):
        most = max(n for _, n in few)
        where = "the data observes" if len(geos) == 1 else "every geo observes at most"
        message = f"{where} {most} {periods} of the KPI, {need}"
        return [Issue(IssueCode.FEW_OBSERVATIONS, Severity.WARNING, message)]
    return [
        Issue(
            IssueCode.FEW_OBSERVATIONS,
            Severity.WARNING,
            f"geo {name!r} observes {n} {periods} of the KPI, {need}",
            geo=name,
        )
        for name, n in few
    ]


def _read_hierarchy(
    geos: tuple[Geo, ...], raw: Mapping[str, str]
) -> tuple[dict[Geo, Geo], list[Issue]]:
    """The hierarchy without the nodes no geo of the data sits below, and what is wrong with it."""
    parent = {Geo(str(child)): Geo(str(up)) for child, up in raw.items()}
    kept: set[Geo] = set()
    for geo in geos:
        node = geo
        while node in parent and node not in kept:
            kept.add(node)
            node = parent[node]
    dropped = sorted(node for node in parent if node not in kept)
    pruned = {child: up for child, up in parent.items() if child in kept}
    found = [
        Issue(IssueCode.HIERARCHY, Severity.ERROR, problem)
        for problem in hierarchy_problems(geos, pruned)
    ]
    if dropped:
        found.append(
            Issue(
                IssueCode.HIERARCHY,
                Severity.WARNING,
                f"hierarchy nodes with no geo of the data below them are dropped: {dropped}",
            )
        )
    return pruned, found


def _read_population(
    geos: tuple[Geo, ...], raw: Mapping[str, float] | None
) -> tuple[npt.NDArray[np.float64] | None, list[Issue]]:
    """Each geo's population, in the panel's order, and what is wrong with it."""
    if raw is None:
        return None, []
    given = {Geo(str(name)): value for name, value in raw.items()}
    found = [
        Issue(
            IssueCode.POPULATION,
            Severity.ERROR,
            f"the population of geo {g!r} is given as {np.size(value)} values: a geo's population "
            "is one number over the calendar; give each geo one, or read a population that "
            "changes as a control",
            geo=g,
        )
        for g, value in given.items()
        if g in geos and isinstance(value, np.ndarray | Sequence) and not isinstance(value, str)
    ]
    if found:
        return None, found
    found += [
        Issue(IssueCode.POPULATION, Severity.ERROR, f"geo {g!r} has no population", geo=g)
        for g in geos
        if given.get(g) is None
    ]
    for g in geos:
        value = given.get(g)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int | float | np.number):
            message = f"the population of geo {g!r}, {value!r}, is not a number"
            found.append(Issue(IssueCode.POPULATION, Severity.ERROR, message, geo=g))
        elif not (math.isfinite(value) and value > 0):
            message = f"the population of geo {g!r}, {value!r}, is not finite and positive"
            found.append(Issue(IssueCode.POPULATION, Severity.ERROR, message, geo=g))
    dropped = sorted(name for name in given if name not in geos)
    if dropped:
        message = f"populations of geos not in the data are dropped: {dropped}"
        found.append(Issue(IssueCode.POPULATION, Severity.WARNING, message))
    if any(issue.severity is Severity.ERROR for issue in found):
        return None, found
    return np.array([float(given[g]) for g in geos]), found


def _constant(
    values: npt.NDArray[np.float64],
    present: npt.NDArray[np.bool_],
    geos: tuple[Geo, ...],
    subject: str,
    consequence: str,
    *,
    column: str,
    channel: str | None = None,
    describe: Callable[[float], str] = lambda v: f"is constant ({v:g})",
) -> list[Issue]:
    """Warn where a column does not vary within a geo, or within any geo at one shared value."""
    level: dict[Geo, float] = {}
    for g, name in enumerate(geos):
        seen = values[g][present[g] & ~np.isnan(values[g])]
        if seen.size >= 2 and seen.min() == seen.max():
            level[name] = float(seen[0])
    if not level:
        return []
    if len(level) == len(geos) and len(set(level.values())) == 1:
        where = "" if len(geos) == 1 else " across all geos"
        message = f"{subject} {describe(next(iter(level.values())))}{where}: {consequence}"
        return [
            Issue(IssueCode.CONSTANT, Severity.WARNING, message, column=column, channel=channel)
        ]
    return [
        Issue(
            IssueCode.CONSTANT,
            Severity.WARNING,
            f"{subject} {describe(v)} in geo {name!r}: {consequence} from that geo",
            geo=name,
            column=column,
            channel=channel,
        )
        for name, v in level.items()
    ]


def _as_dates(days: npt.NDArray[np.datetime64]) -> list[dt.date]:
    return days.astype("datetime64[D]").astype(object).tolist()


def _iso(day: Any) -> str:
    return str(np.asarray(day).astype("datetime64[D]"))


def _listed(dates: Sequence[dt.date]) -> str:
    shown = ", ".join(d.isoformat() for d in dates[:_SHOWN])
    return shown if len(dates) <= _SHOWN else f"{shown} and {len(dates) - _SHOWN} more"


def _runs(steps: npt.NDArray[np.intp], dates: npt.NDArray[np.datetime64]) -> str:
    """Consecutive steps as date ranges: `2023-01-07 to 2023-02-25 (8), 2023-08-19`."""
    breaks = np.flatnonzero(np.diff(steps) > 1) + 1
    runs = np.split(steps, breaks)
    parts = [
        _iso(dates[run[0]])
        if run.size == 1
        else f"{_iso(dates[run[0]])} to {_iso(dates[run[-1]])} ({run.size})"
        for run in runs[:_SHOWN]
    ]
    if len(runs) > _SHOWN:
        parts.append(f"and {len(runs) - _SHOWN} more runs")
    return ", ".join(parts)


def _positions(mask: npt.NDArray[np.bool_], values: np.ndarray | None = None) -> str:
    """Row positions (0-based) of a mask, with the values found there when given."""
    rows = np.flatnonzero(mask)
    if values is None:
        shown = ", ".join(str(r) for r in rows[:_SHOWN])
    else:
        shown = ", ".join(f"{_plain(values[r])!r} at {r}" for r in rows[:_SHOWN])
    more = "" if rows.size <= _SHOWN else f" and {rows.size - _SHOWN} more"
    return f"at row {shown}{more}" if values is None else f"{shown}{more}"


def _plain(value: Any) -> Any:
    """A numpy scalar as the Python value it holds, so a message shows `'soon'`, not
    `np.str_('soon')`. Dates and durations stay numpy's: `item` can turn them into integers."""
    if isinstance(value, np.generic) and value.dtype.kind not in "Mm":
        return value.item()
    return value


def _sample(values: np.ndarray) -> str:
    head = values[:_SHOWN]
    # `tolist` turns nanosecond dates and durations into integers, which would read as numbers
    cells = list(head) if head.dtype.kind in "Mm" else head.tolist()
    shown = ", ".join(repr(v) for v in cells)
    return shown if values.size <= _SHOWN else f"{shown}, ..."
