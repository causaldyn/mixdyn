# 0063. Versioned documents, each read by its declared type

Status: accepted, 2026-11-06; amended 2026-11-07: an issue names the channel whose spend it is
about, `ISSUES` changed in place at version 1

## Context

ADR 0062 gives Python callers one file format and leaves callers outside Python to the versioned
JSON documents of `mixdyn.api`, "which state their schema field by field". Those documents shipped
with the media panel on 2026-10-01 (`mixdyn.api.documents`) and gained a second schema with the
lift tests' estimands, but no record says what a document is, how a value is written into one,
what reading checks, or when a schema's version moves. A caller in another language needs all
four, and an engine that changes a schema needs the last.

## Decision

- **A document is `{"schema": name, "version": n, "data": value}`**, exactly those three keys.
  `Schema(name, version, data)` names a document type: its name, an integer version and the
  declared Python type of its data. `dumps(value, schema)` writes one and `loads(text, schema)`
  reads one; the caller names the schema on both sides, and the document carries it.
- **The schema is the declared type, not the value.** A value is encoded by the type its schema
  declares, never by inspecting the value, so a document's shape is fixed by the schema alone:
  - a dataclass as an object of its fields, those its `__init__` takes;
  - a union of dataclasses as that object plus `"kind"`, the class's name; an optional value as
    `null` when absent;
  - an enum member by its value; a date as ISO 8601; a string, an integer and a boolean as
    themselves, a boolean never read as an integer nor an integer as a boolean;
  - a float as a JSON number, NaN as `null`; an infinity has no form and is refused on writing;
  - a tuple as a list, of one item type (`tuple[X, ...]`) or of fixed length; a mapping with
    string keys as an object, any other key refused;
  - a NumPy array as `{"dtype", "shape", "data"}`, its data flattened in C order, its dtype one of
    `float64`, `int64`, `bool` and `datetime64[D]`; NaN and NaT as `null`;
  - a type alias or a `NewType` as what it stands for.
  The JSON text is written with `allow_nan=False`, so no non-standard token reaches a reader.
- **Reading checks every value against the declared type and names where it fails.** A document
  of another name or version is refused, naming both, never read as this one. An object must hold
  exactly its dataclass's fields: a missing field and an extra one are both refused, a field with
  a default included. A union reads the class its `"kind"` names. An array's dtype, shape and
  every element must fit. Each dataclass is rebuilt through its constructor, so its invariants run
  on a decoded value as on a built one: a panel whose spend is negative is refused at reading as at
  construction. The first failure raises `SchemaError`, a `ValueError`, whose message is the path
  of the value that does not fit (`data[0].dates[1]: expected an ISO date, got 'soon'`).
- **A new document enters as a schema constant** in `mixdyn.api.documents`, a docstring after it,
  listed in `mixdyn.api`'s `__all__` (ADR 0062), its declared type built only of the forms above,
  and a test that round-trips a value of it. No registry beyond the constants exists: a schema is
  data the caller passes to `dumps` and `loads`.
- **The documents at the time of writing**:
  - `ISSUES`, `"mixdyn.issues"` version 1, `tuple[Issue, ...]`: what reading a panel found
    (`MediaPanel.issues`, `PanelError.issues`), each issue's code, severity, message, geo, column,
    dates and channel (the channel since 2026-11-07; see the amendment below).
  - `LIFT_TESTS`, `"mixdyn.lift_tests"` version 1, `tuple[Estimand, ...]`: what each lift test a
    model reads estimated, built by `estimands(panel, estimates)` on the
    panel's calendar, cadence and currency. A test `withheld`, `added` or `moved` spend over its
    changed periods; where the spend or the plan is 0 over every changed period its lift is read
    `"lift against zero spend"`, otherwise as a `"marginal change"` between the two levels' means
    a period, which says nothing of the curve below the lower; `first`, `last` and `end` place it
    in the calendar, and are `null` for a test with no end, which is not placed. `estimand` states
    all of it in one line. A test whose changed spend or readout starts before the panel's first
    date, or whose end is not one of its dates, is refused.
- **When a version moves.** Reading is strict, so a document of one version is refused by a
  reader of another whenever the declared type changes what a document holds. A schema's version
  moves when a field is added, removed or renamed (with a default or without), when a field's
  type, units or meaning change, when an enum the document holds gains, loses or renames a value,
  or when a value is written in another form. A change to a type the schema does not reach moves
  nothing.
- **Before the first release a schema changes in place.** mixdyn has not been released (its
  version is 0.0.1 and no tag exists), so no document a released engine wrote exists: until the
  first release a schema's type may change at its version. `ISSUES` gained `CADENCE` and
  `FEW_OBSERVATIONS` on 2026-10-31 (ADR 0062) at version 1 under this rule. From the first release
  every change above moves the version, and the release that moves it reads the previous one
  through a converter for at least one minor release, as ADR 0062 asks of the file format. No
  converter exists today.

## Consequences

- A document is read the same in any language that reads JSON: its shape follows from the
  declared type alone, and a reader that checks a document by the rules above refuses what this
  engine refuses.
- The documents are stricter than the file format: ADR 0062 lets a field with a default enter
  without moving `FORMAT_VERSION`, because `load` rebuilds a class from the fields a file has; a
  document's object must hold every field, so the same change moves a schema's version.
- A field declared `float | None` writes NaN and `None` alike as `null` and reads `null` as
  `None`: a NaN in such a field does not survive a round trip. No schema at the time of writing
  declares one; a schema that needs both declares the float alone, whose `null` reads as NaN.
- A union of dataclasses names its member by the class's name, so renaming a class moves the
  version of every schema that holds it in a union.
- `tests/test_documents.py` holds the rules: `test_a_document_names_its_schema_and_version` (the
  three keys and the encoding of an issue), `test_issues_survive_a_round_trip` (any issues, drawn
  by Hypothesis), `test_a_document_that_does_not_fit_is_refused_with_its_path` (another schema,
  another version, a missing key, bad JSON, a bad date, an unknown enum value and a missing field,
  each with its path), `test_a_union_of_dataclasses_carries_the_kind_and_floats_keep_nan`,
  `test_arrays_survive_a_round_trip`, `test_values_without_a_json_form_are_refused_on_writing`,
  `test_array_data_that_does_not_fit_its_dtype_is_refused`,
  `test_a_whole_panel_survives_a_round_trip`, `test_reading_runs_the_dataclass_invariants`,
  `test_a_panel_s_population_and_national_split_survive_a_round_trip` (ADR 0006), and for
  `LIFT_TESTS` `test_a_national_go_dark_test_states_a_lift_against_zero_spend_over_its_dark_weeks`,
  `test_a_geo_heavy_up_states_a_marginal_change_between_its_two_levels`,
  `test_a_test_s_estimands_survive_a_round_trip` and
  `test_a_test_outside_the_calendar_or_an_unnamed_reading_is_refused`.

## Alternatives considered

- **Encoding by the value's runtime type**, as `json.dumps` with a default hook does. Not done: a
  reader could not know a document's shape without the value, a subclass would write fields the
  schema does not state, and an empty tuple and an empty array would read alike.
- **A lenient reader that ignores unknown fields and fills missing ones with defaults.** Not done:
  a document of another version would read as this one's, silently, which is what the version
  exists to prevent.
- **JSON Schema files beside the code.** Not written: the declared Python types are the schema,
  and a second statement of them could drift from the first. A caller outside Python reads the
  rules above and the dataclasses' fields.

## Amended 2026-11-07: an issue names the channel whose spend it is about

- **The defect.** `read_panel` reads each channel from a column of any name (`spend={"retail":
  "retail_eur"}`), and an `Issue` named only the column. The plans' flag of a channel moved on its
  prior alone (ADR 0044) matched the `CONSTANT` issue's `column` against the channel's name. So a
  channel read from a column of another name was never named, and a constant control whose column
  bore a channel's name named that channel. `SpendFeedback.issue()` (ADR 0074) wrote the channel's
  name into `column`, as no field held it.
- **Decision: `Issue.channel`**, the last field, default `None`: the channel whose spend the finding
  is about, by the panel's name for it. `read_panel` writes it on every finding about one channel's
  spend values: `CONSTANT`, `NEGATIVE_SPEND`, `NON_FINITE` and `NOT_NUMERIC`.
  `SpendFeedback.issue()` writes it on `ENDOGENOUS_SPEND`. A finding about a column as such names
  the column alone: one missing, given two roles or not one-dimensional. A consumer compares
  `channel` with a channel, never `column`. Chosen over a map from channel to column on
  `MediaPanel`: the issue then states its channel wherever it travels, in `PanelError` and in a
  document, with no panel beside it, and each consumer reads one field instead of a lookup.
- **`ISSUES` stays version 1**, changed in place under the clause above: no released engine wrote
  an issues document. No field changes its meaning, and the new one adds the channel; a version-1
  document written without it is refused, the field named.
- **Saved files.** `Issue` is in the file format's registry. A file written before this loads (ADR
  0062: a field with a default enters without moving `FORMAT_VERSION`), its issues naming no
  channel. A plan made on a panel loaded from such a file names no channel in `unidentified`; the
  panel read again from its data names it. `FORMAT_VERSION` stays 1.
- **Not changed.** `ENDOGENOUS_SPEND` still writes the channel's name into `column` as well:
  `spend_feedback` reads a panel, which keeps no column names, and the issue's `column` is an
  existing output.
- **Tests.**
  - `tests/test_read_panel.py`: a constant channel read from a column of another name is named by
    both, and a constant KPI's or control's names no channel, though the control's column bears a
    channel's name. A negative, missing, textual or out-of-range spend names its channel; a
    control's or the KPI's error names none.
  - `tests/test_feedback.py`: the `ENDOGENOUS_SPEND` warning names its channel.
  - `tests/test_documents.py`: an issue's channel survives a round trip (Hypothesis); a version-1
    document without the field is refused, the field named.
  - `tests/test_roundtrip.py`: an issue naming its channel is saved and loaded the same; a panel
    saved without the field loads, its issues naming none, every other field as saved.
  - `tests/test_flat_spend.py`: retail read from `retail_eur` is named by `plan_posterior` and by
    `plan_robust` at 1.1; a `CONSTANT` issue naming the column `retail` and no channel names
    nothing.
