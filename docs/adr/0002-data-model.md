# 0002. The media panel: one regular calendar, checked at the boundary

Status: accepted, 2026-10-01; amended 2026-10-07: one rule reads a column's cells as numbers,
whatever array or frame holds them, and text is not one; amended 2026-11-06 by ADRs 0068 and 0069:
levers and calendar events are fields of the panel, each checked on construction; amended
2026-11-07: an issue about a channel's spend names the channel.

## Context

An engine that numbers periods by row, not by date, reads a weekly series missing one week as
though every later week were the week before it: each later week's trend and season is read off its
neighbour's.

The same design leaves several other things to convention:
- whether a table is long or wide;
- whether a sign or a mode is coded as a string or an integer;
- what unit a KPI is in.

Each convention is a place for two callers to disagree.

## Decision

- **The calendar is the invariant.**
  - `MediaPanel.dates` holds every period from the first to the last at the panel's cadence.
  - Position `t` of every array is `dates[t]` in every geo, so a period's index is fixed by its date.
  - Weekly periods sit on the weekday most dates share.
  - Monthly periods are labelled by the first of the month.
- **A period with no row is explicit** (`Gap`).
  - By default, reading fails, naming the geo and the dates.
  - `Gap.STEP` keeps the period as a filled step with no KPI observation, zero spend and NaN
    controls. The zero is a claim about that period, and adstock carries it forward, so this is
    opt-in.
- **Arrays, not frames.**
  - The KPI is a float64 array (geo, period); spend is (geo, period, channel). All are read-only.
  - Reading accepts a mapping or any frame with `.columns`, recognised structurally. No frame
    library is a dependency.
  - One rule reads a column's cells as numbers, whatever holds them (amended 2026-10-07):
    - a missing cell is missing: `None`, NaN, NaT, pandas' `NA`, a masked cell;
    - text is not a number (`NOT_NUMERIC`), though it may spell one: where thousands are written
      with a point, `"1.250"` is 1250, which `float` reads as 1.25, so the caller converts text
      where its locale is known;
    - a date, a duration, a complex number or a record is not a number (`NOT_NUMERIC`);
    - a number past float64's range is an error of its own (`NON_FINITE`), not an infinity;
      an infinity in the source is reported as one.
- **Units are a sum type.**
  - The KPI is `Revenue()`, in the spend's currency, or `Count(unit)`.
  - The currency is an ISO 4217 code. A panel cannot hold revenue and spend in two currencies.
- **A geo hierarchy is one tree whose leaves are the panel's geos.**
  - Nodes with no geo below them are dropped with a warning.
  - A geo with children is refused, since its rows would count theirs twice.
- **Checks run at the boundary, all at once.**
  - `read_panel` reports every problem as an `Issue` with a code, a severity, the geo and every
    date involved. Errors raise `PanelError`.
  - An issue about one channel's spend names the channel beside its column (amended 2026-11-07):
    a channel is read from a column of any name, and a consumer compares channels with channels
    (ADR 0063, amended 2026-11-07).
  - Warnings travel on `MediaPanel.issues` and to the log.
- **`MediaPanel` re-checks its invariants on construction.** A panel built directly, or decoded from
  a document, cannot break them either.

## Consequences

- Engines index periods by position and get dates for free. Trend and season come from the period
  index, never from a row count.
- An engine must treat a NaN KPI as a missing observation. The DLM steps without an update.
- Not in the panel yet:
  - exposure (impressions, GRPs);
  - population;
  - national-media flags;
  - calendar events.

  Each arrives with the first engine that reads it, as a field with its own checks.

## Alternatives considered

- **Carry a long data frame through the engine.** Rejected: it needs a frame library, and it cannot
  hold the calendar invariant; every consumer would re-derive it.
- **Rows indexed by position, with a date column alongside.** Rejected: it is the defect above.
- **Impute a missing week.** Rejected: it is silent. If anything fills a period, it is a stated
  policy.
- **Read text that spells a number as that number** (until 2026-10-07). Rejected: `float` knows one
  locale, so `"1.250"` written as 1250 read as 1.25, a thousandth of a spend with no error; the
  text `"nan"` read as a missing cell and `"1_000"` as 1000. Refusing only the ambiguous form, a
  point before three digits, would refuse 1.250 written in the reader's own locale and still read
  the rest. Text from an API or a spreadsheet is converted by the caller, who knows its locale.
