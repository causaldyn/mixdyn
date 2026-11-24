"""Fuzz `mixdyn.data.read`'s `read_panel` on long tables of any content.

    uv run --with-requirements fuzz/requirements.txt python fuzz/fuzz_read.py -max_total_time=60

Each input decodes into a table of up to 12 rows over up to three geos, and a call to read it:

* the dates on the cadence, off it, repeated, missing, or of another type;
* the geos as text, empty, missing or a number;
* the KPI, two channels' spends and a control: any float (NaN, the infinities, negatives, zeros,
  huge ones), a missing value, a number written as text, a `Decimal`, a bool, or text that is not a
  number;
* the cadence, the rule for a missing period, the currency, and a population for each geo.

`read_panel` refuses the table with `PanelError`, or it returns a panel that keeps the promises of
its docstring and `MediaPanel`'s:

* its dates rise by one period of the cadence at a time, and its geos are sorted;
* each array is shaped by the panel's geos, periods, and channels or controls;
* spend is finite and never negative, and the controls are finite but on a filled step;
* a filled step has no KPI observation, zero spend and NaN controls;
* the channels are `spend`'s, in its order.

Anything else, a broken promise or another exception, is a crash with the input that caused it.
"""

from __future__ import annotations

import datetime as dt
import sys
from decimal import Decimal
from typing import Any

import atheris
import numpy as np
from atheris.import_hook import HookManager

# Branch coverage of the whole package, without atheris's comparison tracing: in atheris 3.1.0
# that tracing keeps each numpy array a traced comparison returns (41 MB over 3 000 inputs of
# the planners), and `instrument_imports` always turns it on.
with HookManager({"mixdyn"}, set(), enable_loader_override=True, trace_dataflow=False):
    from mixdyn.data.issues import PanelError
    from mixdyn.data.panel import Cadence, Count, Revenue
    from mixdyn.data.read import Gap, read_panel

START = dt.date(2023, 1, 2)  # a Monday: the daily and weekly grids start here
CADENCES = list(Cadence)
GAPS = list(Gap)
GEOS = ("north", "south", "east")
SPEND = {"tv": "spend_tv", "search": "spend_search"}
FLOATS = (0.0, -0.0, 1.0, -1.0, 1e-300, 5e-324, 1e300, float("nan"), float("inf"), float("-inf"))


def _step(cadence: Cadence, start: dt.date, k: int) -> dt.date:
    if cadence is Cadence.MONTH:
        month = start.month - 1 + k
        return dt.date(start.year + month // 12, month % 12 + 1, 1)
    return start + dt.timedelta(days=k * (7 if cadence is Cadence.WEEK else 1))


def _date(provider: atheris.FuzzedDataProvider, cadence: Cadence, period: int) -> Any:
    start = dt.date(2023, 1, 1) if cadence is Cadence.MONTH else START
    kind = provider.ConsumeIntInRange(0, 9)
    day = _step(cadence, start, period)
    if kind <= 4:
        return day
    if kind == 5:
        return day + dt.timedelta(days=provider.ConsumeIntInRange(-40, 40))
    if kind == 6:
        return np.datetime64(day, "D") if provider.ConsumeBool() else np.datetime64(day, "s")
    if kind == 7:
        return day.isoformat() if provider.ConsumeBool() else dt.datetime.combine(day, dt.time())
    if kind == 8:
        return None
    return provider.ConsumeIntInRange(-5, 5) if provider.ConsumeBool() else float("nan")


def _geo(provider: atheris.FuzzedDataProvider, index: int) -> Any:
    kind = provider.ConsumeIntInRange(0, 7)
    if kind <= 4:
        return GEOS[index]
    if kind == 5:
        return ""
    if kind == 6:
        return None
    return index


def _value(provider: atheris.FuzzedDataProvider) -> Any:
    kind = provider.ConsumeIntInRange(0, 11)
    if kind <= 4:
        return provider.ConsumeFloatInRange(0.0, 1000.0)
    if kind == 5:
        return FLOATS[provider.ConsumeIntInRange(0, len(FLOATS) - 1)]
    if kind == 6:
        return provider.ConsumeRegularFloat()
    if kind == 7:
        return None
    if kind == 8:
        return f"{provider.ConsumeFloatInRange(-10.0, 10.0)}"
    if kind == 9:
        return Decimal(provider.ConsumeIntInRange(-100, 100)) / 7
    if kind == 10:
        return provider.ConsumeBool()
    return provider.ConsumeUnicodeNoSurrogates(4)


def _column(provider: atheris.FuzzedDataProvider, rows: int) -> Any:
    values = [_value(provider) for _ in range(rows)]
    if provider.ConsumeIntInRange(0, 3) == 0:
        return values
    try:
        return np.asarray(values, dtype=float)
    except (TypeError, ValueError, ArithmeticError):
        return np.asarray(values, dtype=object)


def table(provider: atheris.FuzzedDataProvider) -> tuple[dict[str, Any], dict[str, Any]]:
    """A table and the keywords that read it, decoded from the input."""
    cadence = CADENCES[provider.ConsumeIntInRange(0, len(CADENCES) - 1)]
    geos = provider.ConsumeIntInRange(1, len(GEOS))
    periods = provider.ConsumeIntInRange(1, 12 // geos)
    keep = [provider.ConsumeIntInRange(0, 7) != 0 for _ in range(geos * periods)]
    cells = [(g, t) for g in range(geos) for t in range(periods)]
    cells = [cell for cell, kept in zip(cells, keep, strict=True) if kept]
    if provider.ConsumeBool():
        cells.reverse()
    rows = len(cells)
    data: dict[str, Any] = {
        "date": [_date(provider, cadence, t) for _, t in cells],
        "region": [_geo(provider, g) for g, _ in cells],
        "sales": _column(provider, rows),
        "spend_tv": _column(provider, rows),
        "spend_search": _column(provider, rows),
        "price": _column(provider, rows),
    }
    keywords: dict[str, Any] = {
        "date": "date",
        "kpi": "sales",
        "spend": SPEND,
        "controls": ("price",),
        "cadence": cadence,
        "kpi_unit": Revenue()
        if provider.ConsumeBool()
        else Count(["orders", " "][provider.ConsumeIntInRange(0, 1)]),
        "currency": ["USD", "EUR", "usd", ""][provider.ConsumeIntInRange(0, 3)],
        "gaps": GAPS[provider.ConsumeIntInRange(0, len(GAPS) - 1)],
    }
    if geos > 1 or provider.ConsumeBool():
        keywords["geo"] = "region"
    if provider.ConsumeIntInRange(0, 3) == 0:
        keywords["population"] = {g: _value(provider) for g in GEOS[:geos]}
    return data, keywords


def _promises(panel: Any, cadence: Cadence) -> None:
    dates = panel.dates.astype("datetime64[D]")
    if cadence is Cadence.MONTH:
        months = dates.astype("datetime64[M]")
        if np.any(months.astype("datetime64[D]") != dates) or np.any(
            np.diff(months) != np.timedelta64(1, "M")
        ):
            raise AssertionError(f"monthly dates {dates} are not consecutive first days")
    else:
        step = np.timedelta64(7 if cadence is Cadence.WEEK else 1, "D")
        if np.any(np.diff(dates) != step):
            raise AssertionError(f"dates {dates} do not rise by {step}")
    if list(panel.geos) != sorted(panel.geos):
        raise AssertionError(f"geos {panel.geos} are not sorted")
    shape = (len(panel.geos), dates.size)
    if panel.kpi.shape != shape or panel.filled.shape != shape:
        raise AssertionError(f"kpi {panel.kpi.shape} or filled {panel.filled.shape}, not {shape}")
    if panel.spend.shape != (*shape, len(SPEND)):
        raise AssertionError(f"spend {panel.spend.shape}, not {(*shape, len(SPEND))}")
    if panel.control_values.shape != (*shape, 1):
        raise AssertionError(f"controls {panel.control_values.shape}, not {(*shape, 1)}")
    if list(panel.channels) != list(SPEND):
        raise AssertionError(f"channels {panel.channels}, not {list(SPEND)}")
    if not np.all(np.isfinite(panel.spend)) or np.any(panel.spend < 0.0):
        raise AssertionError(f"spend {panel.spend} not finite and non-negative")
    filled = panel.filled
    if not np.all(np.isfinite(panel.control_values[~filled])):
        raise AssertionError(f"controls {panel.control_values} not finite off the filled steps")
    if np.any(np.isfinite(panel.kpi[filled])) or np.any(panel.spend[filled] != 0.0):
        raise AssertionError("a filled step has a KPI observation or spend")
    if not np.all(np.isnan(panel.control_values[filled])):
        raise AssertionError("a filled step has a control's value")
    if panel.population is not None and (
        panel.population.shape != (len(panel.geos),)
        or not np.all(np.isfinite(panel.population))
        or np.any(panel.population <= 0.0)
    ):
        raise AssertionError(f"population {panel.population} not finite and positive by geo")


def check(data: bytes) -> None:
    provider = atheris.FuzzedDataProvider(data)
    columns, keywords = table(provider)
    try:
        panel = read_panel(columns, **keywords)
    except PanelError:
        return
    _promises(panel, keywords["cadence"])


def main() -> None:
    atheris.Setup(sys.argv, check)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
