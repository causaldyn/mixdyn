"""Lift tests read from the tables other tools take them in, and written back to them.

PyMC-Marketing's lift table (`MMM.add_lift_test_measurements`) holds a row a test: the channel, the
spend a period `x` before it, the change `delta_x` the test made to it, the change `delta_y` in the
KPI a period, its standard error `sigma`, a column for each of the model's dimensions, such as a
geo, and optionally a date. It reads `delta_y` on the saturation curve at a steady spend, with no
carryover. A row says nothing of how long the test ran, so `read_lift_table` is told: the change
held for `periods` periods and the lift read over those and `cooldown` more at the spend before,
`delta_y` the lift over that readout a changed period. That is how an analyst reduces a test whose
readout holds the carryover the change cost to a row of a steady-state likelihood: the row's
`delta_y` times `periods` is the lift, and `sigma` times `periods` its standard error. A date names
the first changed period.

A geo experiment's analysis, as GeoX reports one and Meridian's calibration takes it, is a record:
the estimate, its standard error and the window the spend changed over, by its first and last
dates. `read_experiments` reads each as the lift over the window and the periods of cooldown after
it, or, where the estimates are returns, incremental KPI a currency unit of spend changed (GeoX's
iROAS, Meridian's ROI), as that return times the spend the window changed.

Either is read into `LiftEstimate`s, each channel's in the table's order, the readout ending in the
panel's calendar. What each estimated is named where `mixdyn.api.estimands` reads them: a lift
against zero spend where the test went dark or started from nothing, otherwise a marginal change.
`lift_table` and `experiment_table` write estimates back, so that one file of tests feeds every
tool.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from mixdyn.calibration.lift import LiftEstimate
from mixdyn.data import Cadence, Channel, Geo
from mixdyn.data.read import _read_dates, _select

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mixdyn.data import ColumnData, MediaPanel

type Estimates = dict[str, tuple[LiftEstimate, ...]]
"""Lift estimates by channel, as the models take them."""

_LIFT = ("channel", "x", "delta_x", "delta_y", "sigma")
_RECORD = ("channel", "estimate", "standard_error", "start", "end", "spend", "planned")


def read_lift_table(
    panel: MediaPanel,
    table: ColumnData,
    *,
    periods: int,
    cooldown: int = 0,
    geo: str = "geo",
    date: str = "date",
) -> Estimates:
    """PyMC-Marketing's lift table as lift estimates on `panel`: each row a test that changed the
    spend from `x` to `x + delta_x` a period for `periods` periods, its lift read over those and
    `cooldown` periods after at `x`, `delta_y` times `periods` and its standard error `sigma` times
    `periods`. A `geo` column names the geo each test ran in, the whole market where absent, and a
    `date` column its first changed period, on the panel's calendar.

    Raises:
        ValueError: on a column of the five missing or not one-dimensional, columns of different
            lengths or no row; `periods` not a whole number at least 1 or `cooldown` at least 0;
            a channel or a geo the panel does not have; a date that is not one of the panel's;
            and what `LiftEstimate` refuses of a row, the row named.
    """
    _whole(periods, "periods", 1)
    _whole(cooldown, "cooldown", 0)
    columns = _columns(table, _LIFT, (geo, date))
    starts = _dates(panel, columns[date], date) if date in columns else None
    out: dict[str, list[LiftEstimate]] = {}
    for row in range(len(columns["channel"])):
        channel = _channel(panel, columns["channel"][row], row)
        x, delta_x, delta_y, sigma = (_number(columns[name][row], name, row) for name in _LIFT[1:])
        spend = np.concatenate([np.full(periods, x + delta_x), np.full(cooldown, x)])
        planned = np.full(periods + cooldown, x)
        end = None if starts is None else _shift(starts[row], periods + cooldown - 1, panel)
        out.setdefault(channel, []).append(
            _estimate(
                row,
                spend,
                planned,
                periods + cooldown,
                delta_y * periods,
                sigma * periods,
                1.0,
                _geos(panel, columns, geo, row),
                end,
            )
        )
    return {channel: tuple(tests) for channel, tests in out.items()}


def lift_table(
    panel: MediaPanel,
    estimates: Mapping[str, Sequence[LiftEstimate]],
    *,
    periods: int,
    cooldown: int = 0,
) -> dict[str, list[object]]:
    """Lift estimates as PyMC-Marketing's lift table, the columns `read_lift_table` reads: each
    estimate's lift a changed period and its standard error a changed period, its spend before
    the test and the change. A `geo` column where any estimate ran in one geo, and a `date`
    column, its first changed period's, where any estimate is placed in the calendar.

    Raises:
        ValueError: on an estimate whose readout is not `periods` changed periods then `cooldown`
            at its planned spend, whose planned spend is not one level, whose changed spend is not
            one level, that ran in more than one geo or in a share of the market, or that is placed
            in the calendar beside one that is not.
    """
    _whole(periods, "periods", 1)
    _whole(cooldown, "cooldown", 0)
    rows: dict[str, list[object]] = {name: [] for name in _LIFT}
    geos: list[object] = []
    dates: list[object] = []
    for channel, tests in estimates.items():
        for test in tests:
            changed, level, plan = _levels(channel, test)
            length = periods + cooldown
            if changed != (0, periods - 1) or test.readout != length or test.spend.size != length:
                msg = (
                    f"a test of {channel!r} is not {periods} changed periods read with {cooldown} "
                    f"after: it changed periods {changed} of {test.spend.size}, its readout the "
                    f"last {test.readout}"
                )
                raise ValueError(msg)
            if test.share != 1.0:
                msg = f"a test of {channel!r} ran in {test.share} of its market: a row reads all"
                raise ValueError(msg)
            rows["channel"].append(str(channel))
            rows["x"].append(plan)
            rows["delta_x"].append(level - plan)
            rows["delta_y"].append(test.lift / periods)
            rows["sigma"].append(test.se / periods)
            geos.append(_geo_name(channel, test))
            dates.append(
                None
                if test.end is None
                else _iso(_shift(test.end, -(periods + cooldown - 1), panel))
            )
    return rows | _optional("geo", geos) | _optional("date", dates)


def read_experiments(
    panel: MediaPanel, records: ColumnData, *, per_unit: bool = False
) -> Estimates:
    """Geo experiments' records as lift estimates on `panel`: each record a test of `channel`
    whose group spent `spend` a period against `planned` over its window, the periods from
    `start` to `end` on the panel's calendar, its lift read over the window and `cooldown`
    periods after it at `planned`, none where the column is absent. Its `estimate` and
    `standard_error` are the lift, or, where `per_unit`, the incremental KPI a currency unit of the
    spend the window changed, which they are multiplied by. A `geo` column names the geo the group
    is, the whole market where absent, and a `share` column its share of that market, 1 where
    absent.

    Raises:
        ValueError: on a column of the seven missing or not one-dimensional, columns of different
            lengths or no row; a channel or a geo the panel does not have; a start or an end that
            is not one of the panel's dates, or an end before the start; a cooldown that is not a
            whole number at least 0; and what `LiftEstimate` refuses of a record, the row named.
    """
    columns = _columns(records, _RECORD, ("geo", "share", "cooldown"))
    starts, ends = _dates(panel, columns["start"], "start"), _dates(panel, columns["end"], "end")
    out: dict[str, list[LiftEstimate]] = {}
    for row in range(len(columns["channel"])):
        channel = _channel(panel, columns["channel"][row], row)
        first, last = _index(panel, starts[row]), _index(panel, ends[row])
        if last < first:
            msg = f"row {row} ends on {ends[row]}, before it starts on {starts[row]}"
            raise ValueError(msg)
        after = 0 if "cooldown" not in columns else _count(columns["cooldown"][row], row)
        length = last - first + 1
        level, plan = (_number(columns[name][row], name, row) for name in ("spend", "planned"))
        value, error = (
            _number(columns[name][row], name, row) for name in ("estimate", "standard_error")
        )
        if per_unit:
            changed = length * (level - plan)
            value, error = value * changed, error * abs(changed)
        share = 1.0 if "share" not in columns else _number(columns["share"][row], "share", row)
        out.setdefault(channel, []).append(
            _estimate(
                row,
                np.concatenate([np.full(length, level), np.full(after, plan)]),
                np.full(length + after, plan),
                length + after,
                value,
                error,
                share,
                _geos(panel, columns, "geo", row),
                _shift(ends[row], after, panel),
            )
        )
    return {channel: tuple(tests) for channel, tests in out.items()}


def experiment_table(
    panel: MediaPanel, estimates: Mapping[str, Sequence[LiftEstimate]]
) -> dict[str, list[object]]:
    """Lift estimates as experiments' records, the columns `read_experiments` reads with its
    estimate a lift: each estimate's lift and standard error, the window its spend changed over by
    its first and last dates, the periods read after it, its spend and planned spend a period over
    the window, its geo where it ran in one and its share of the market.

    Raises:
        ValueError: on an estimate not placed in the calendar, whose readout does not start where
            its spend changes, whose planned spend is not one level, whose changed spend is not one
            level over consecutive periods, or that ran in more than one geo.
    """
    columns = (*_RECORD, "cooldown", "share")
    rows: dict[str, list[object]] = {name: [] for name in columns}
    geos: list[object] = []
    for channel, tests in estimates.items():
        for test in tests:
            (first, last), level, plan = _levels(channel, test)
            size = test.spend.size
            if test.end is None or first != size - test.readout:
                msg = (
                    f"a test of {channel!r} is not placed in the calendar, or its readout does "
                    f"not start where its spend changes: an experiment's record names its window"
                )
                raise ValueError(msg)
            rows["channel"].append(str(channel))
            rows["estimate"].append(test.lift)
            rows["standard_error"].append(test.se)
            rows["start"].append(_iso(_shift(test.end, first - size + 1, panel)))
            rows["end"].append(_iso(_shift(test.end, last - size + 1, panel)))
            rows["spend"].append(level)
            rows["planned"].append(plan)
            rows["cooldown"].append(size - 1 - last)
            rows["share"].append(test.share)
            geos.append(_geo_name(channel, test))
    return rows | _optional("geo", geos)


def _columns(
    data: ColumnData, required: Sequence[str], optional: Sequence[str]
) -> dict[str, np.ndarray]:
    columns, absent = _select(data, [*required, *optional])
    missing = [name for name in required if name in absent]
    if missing:
        msg = f"the table has no columns {missing}: it needs {list(required)}"
        raise ValueError(msg)
    flat = [name for name, values in columns.items() if values.ndim != 1]
    lengths = {len(values) for values in columns.values() if values.ndim == 1}
    if flat or len(lengths) != 1 or lengths == {0}:
        msg = f"the table's columns must be one-dimensional, of one length and not empty: {flat}"
        raise ValueError(msg)
    return columns


def _whole(value: int, name: str, least: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < least:
        msg = f"{name} must be a whole number of periods, at least {least}, got {value!r}"
        raise ValueError(msg)


def _number(value: Any, name: str, row: int) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        msg = f"row {row}'s {name} is not a number: {value!r}"
        raise ValueError(msg) from None
    if not math.isfinite(number):
        msg = f"row {row}'s {name} is not finite: {value!r}"
        raise ValueError(msg)
    return number


def _count(value: Any, row: int) -> int:
    count = _number(value, "cooldown", row)
    if not (count.is_integer() and count >= 0.0):
        msg = f"row {row}'s cooldown, {value!r}, is not a whole number of periods, at least 0"
        raise ValueError(msg)
    return int(count)


def _channel(panel: MediaPanel, value: object, row: int) -> Channel:
    channel = Channel(str(value))
    if channel not in panel.channels:
        msg = f"row {row} names channel {channel!r}, which the panel does not have"
        raise ValueError(msg)
    return channel


def _geos(
    panel: MediaPanel, columns: Mapping[str, np.ndarray], column: str, row: int
) -> frozenset[Geo] | None:
    if column not in columns:
        return None
    geo = Geo(str(columns[column][row]))
    if geo not in panel.geos:
        msg = f"row {row} names geo {geo!r}, which the panel does not have"
        raise ValueError(msg)
    return frozenset({geo})


def _dates(panel: MediaPanel, values: np.ndarray, column: str) -> npt.NDArray[np.datetime64]:
    """The column's values as dates, each one of the panel's."""
    days, issues = _read_dates(values, column)
    if issues:
        msg = "; ".join(issue.message for issue in issues)
        raise ValueError(msg)
    off = [str(day) for day in days if day not in panel.dates]
    if off:
        msg = f"column {column!r} holds dates the panel's calendar does not: {off}"
        raise ValueError(msg)
    return days


def _index(panel: MediaPanel, day: np.datetime64) -> int:
    return int(np.flatnonzero(panel.dates == day)[0])


def _shift(day: np.datetime64, steps: int, panel: MediaPanel) -> np.datetime64:
    """The date `steps` periods of the panel's cadence after `day`, before it where negative."""
    if panel.cadence is Cadence.MONTH:
        return (np.datetime64(day, "M") + np.timedelta64(steps, "M")).astype("datetime64[D]")
    spacing = 7 if panel.cadence is Cadence.WEEK else 1
    return np.datetime64(day, "D") + np.timedelta64(spacing * steps, "D")


def _iso(day: np.datetime64) -> str:
    return str(np.datetime64(day, "D"))


def _estimate(
    row: int,
    spend: np.ndarray,
    planned: np.ndarray,
    readout: int,
    lift: float,
    se: float,
    share: float,
    geos: frozenset[Geo] | None,
    end: np.datetime64 | None,
) -> LiftEstimate:
    try:
        return LiftEstimate(spend, planned, readout, lift, se, share, geos, end)
    except ValueError as error:
        msg = f"row {row}: {error}"
        raise ValueError(msg) from error


def _levels(channel: str, test: LiftEstimate) -> tuple[tuple[int, int], float, float]:
    """The first and last periods a test changed, its spend a period over them and its planned
    spend a period, each one level."""
    changed = np.flatnonzero(test.spend != test.planned)
    first, last = int(changed[0]), int(changed[-1])
    levels = np.unique(test.spend[first : last + 1])
    plans = np.unique(test.planned)
    if levels.size != 1 or plans.size != 1:
        msg = (
            f"a test of {channel!r} is not one changed level over consecutive periods against one "
            f"planned level: spend {test.spend.tolist()}, planned {test.planned.tolist()}"
        )
        raise ValueError(msg)
    return (first, last), float(levels[0]), float(plans[0])


def _geo_name(channel: str, test: LiftEstimate) -> str | None:
    if test.geos is None:
        return None
    if len(test.geos) != 1:
        msg = f"a test of {channel!r} ran in {sorted(test.geos)}: a row names one geo"
        raise ValueError(msg)
    return str(next(iter(test.geos)))


def _optional(name: str, values: list[object]) -> dict[str, list[object]]:
    """The column where any row has a value, none where no row does."""
    if all(value is None for value in values):
        return {}
    if any(value is None for value in values):
        msg = f"some rows have a {name} and some do not: a table holds it for every row or none"
        raise ValueError(msg)
    return {name: values}
