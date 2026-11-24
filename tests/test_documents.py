from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass, replace

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from mixdyn import (
    Allotment,
    Cadence,
    Count,
    Gap,
    Geo,
    Issue,
    IssueCode,
    MediaPanel,
    Revenue,
    Severity,
    allot_national,
    read_panel,
)
from mixdyn.api import (
    ISSUES,
    LIFT_TESTS,
    Estimand,
    Schema,
    SchemaError,
    dumps,
    estimands,
    loads,
    sensitivity_line,
)
from mixdyn.calibration import LiftEstimate, LiftSeries
from mixdyn.data.panel import KpiUnit
from mixdyn.inference.sensitivity import Benchmark, MsmSensitivity, RidgeSensitivity

issues = st.lists(
    st.builds(
        Issue,
        code=st.sampled_from(IssueCode),
        severity=st.sampled_from(Severity),
        message=st.text(),
        geo=st.none() | st.text(),
        column=st.none() | st.text(),
        dates=st.lists(st.dates()).map(tuple),
        channel=st.none() | st.text(),
    )
).map(tuple)


@given(issues)
def test_issues_survive_a_round_trip(value: tuple[Issue, ...]) -> None:
    assert loads(dumps(value, ISSUES), ISSUES) == value


def test_a_document_names_its_schema_and_version() -> None:
    issue = Issue(
        IssueCode.DUPLICATE, Severity.ERROR, "twice", geo="north", dates=(dt.date(2024, 1, 6),)
    )
    document = json.loads(dumps((issue,), ISSUES))
    assert document == {
        "schema": "mixdyn.issues",
        "version": 1,
        "data": [
            {
                "code": "duplicate",
                "severity": "error",
                "message": "twice",
                "geo": "north",
                "column": None,
                "dates": ["2024-01-06"],
                "channel": None,
            }
        ],
    }


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ('{"schema": "mixdyn.fit", "version": 1, "data": []}', "is a 'mixdyn.fit', not"),
        ('{"schema": "mixdyn.issues", "version": 2, "data": []}', "this engine reads version 1"),
        ('{"schema": "mixdyn.issues", "data": []}', "exactly the keys"),
        ("[1, 2", "not JSON"),
        (
            '{"schema": "mixdyn.issues", "version": 1, "data": [{"code": "duplicate", '
            '"severity": "error", "message": "m", "geo": null, "column": null, '
            '"dates": ["2024-01-06", "soon"], "channel": null}]}',
            r"data\[0\]\.dates\[1\]: expected an ISO date, got 'soon'",
        ),
        (
            '{"schema": "mixdyn.issues", "version": 1, "data": [{"code": "nope", '
            '"severity": "error", "message": "m", "geo": null, "column": null, "dates": [], '
            '"channel": null}]}',
            r"data\[0\]\.code: 'nope' is not a IssueCode",
        ),
        (
            '{"schema": "mixdyn.issues", "version": 1, "data": [{"code": "duplicate", '
            '"severity": "error", "message": "m", "geo": null, "dates": [], "channel": null}]}',
            r"data\[0\]: Issue has the fields",
        ),
        (
            '{"schema": "mixdyn.issues", "version": 1, "data": [{"code": "constant", '
            '"severity": "warning", "message": "m", "geo": null, "column": "tv", "dates": []}]}',
            r"data\[0\]: Issue has the fields \[.*'channel'.*\], the document \[",
        ),
    ],
)
def test_a_document_that_does_not_fit_is_refused_with_its_path(text: str, match: str) -> None:
    with pytest.raises(SchemaError, match=match):
        loads(text, ISSUES)


def test_an_integer_python_s_parser_does_not_convert_is_refused_as_not_json() -> None:
    """Python's parser refuses an integer of more digits than it converts with `ValueError`, which
    is not `JSONDecodeError`."""
    text = '{"schema": "mixdyn.issues", "version": 1, "data": ' + "1" * 5_000 + "}"
    with pytest.raises(SchemaError, match="not JSON: Exceeds the limit"):
        loads(text, ISSUES)


def test_json_nested_past_the_parser_s_stack_is_refused_as_not_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The parser refuses JSON nested past its stack with `RecursionError`. From CPython 3.14 on,
    how deep that is depends on the C stack, so the refusal is made here; the test below reads a
    real one wherever the stack runs out first."""

    def parse(_: str) -> object:
        raise RecursionError("maximum recursion depth exceeded while decoding a JSON array")

    monkeypatch.setattr(json, "loads", parse)
    with pytest.raises(SchemaError, match="not JSON: maximum recursion depth"):
        loads("[]", ISSUES)


def test_json_nested_a_hundred_thousand_deep_is_refused() -> None:
    """Where the stack runs out the parser refuses it, and where it does not the document is a
    list, not an object: either way a `SchemaError`, never the parser's `RecursionError`."""
    with pytest.raises(SchemaError, match=r"not JSON|not a document"):
        loads("[" * 100_000 + "]" * 100_000, ISSUES)


def test_a_value_nested_past_what_repr_writes_is_named_in_short() -> None:
    """`json` reads a value nested deeper than `repr` writes; the refusal names it cut short. On
    Python 3.12 and 3.13 the parser stops first, and the text is not JSON."""
    nested = '{"a": [' * 30_000 + "1" + "]}" * 30_000
    text = (
        '{"schema": "mixdyn.issues", "version": 1, "data": [{"code": "duplicate", "severity": '
        f'"error", "message": {nested}, "geo": null, "column": null, "dates": [], "channel": null'
        "}]}"
    )
    with pytest.raises(SchemaError, match=r"data\[0\]\.message: expected str|not JSON") as error:
        loads(text, ISSUES)
    assert len(str(error.value)) < 300


def test_an_enum_s_value_nested_past_what_repr_writes_is_named_in_short() -> None:
    """An enum's own lookup names a value it does not hold with `repr`, which the decoder never
    lets it meet. On Python 3.12 and 3.13 the parser stops first, and the text is not JSON."""
    nested = '{"a": [' * 30_000 + "1" + "]}" * 30_000
    text = (
        '{"schema": "mixdyn.issues", "version": 1, "data": [{"code": "duplicate", "severity": '
        f'{nested}, "message": "", "geo": null, "column": null, "dates": [], "channel": null'
        "}]}"
    )
    with pytest.raises(SchemaError, match=r"data\[0\]\.severity: .* is not a Severity|not JSON"):
        loads(text, ISSUES)


@dataclass(frozen=True, slots=True)
class Priced:
    unit: KpiUnit
    value: float
    counts: np.ndarray


PRICED = Schema("test.priced", 1, tuple[Priced, ...])


def test_a_union_of_dataclasses_carries_the_kind_and_floats_keep_nan() -> None:
    value = (
        Priced(Revenue(), math.nan, np.array([[1.5, np.nan]])),
        Priced(Count("orders"), 2.0, np.array([True, False])),
    )
    document = json.loads(dumps(value, PRICED))
    assert document["data"][0]["unit"] == {"kind": "Revenue"}
    assert document["data"][1]["unit"] == {"kind": "Count", "unit": "orders"}
    assert document["data"][0]["counts"] == {
        "dtype": "float64",
        "shape": [1, 2],
        "data": [1.5, None],
    }
    back = loads(dumps(value, PRICED), PRICED)
    assert back[1].unit == Count("orders")
    assert math.isnan(back[0].value)
    np.testing.assert_array_equal(back[0].counts, value[0].counts)
    np.testing.assert_array_equal(back[1].counts, value[1].counts)


@pytest.mark.parametrize(
    "array",
    [
        np.array([], dtype=np.float64),
        np.arange(6, dtype=np.int64).reshape(2, 3),
        np.array(["2024-01-01", "NaT"], dtype="datetime64[D]"),
    ],
)
def test_arrays_survive_a_round_trip(array: np.ndarray) -> None:
    schema = Schema("test.array", 1, np.ndarray)
    back = loads(dumps(array, schema), schema)
    assert back.dtype == array.dtype
    np.testing.assert_array_equal(back, array)


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (np.array([1.0, np.inf]), "infinite value has no JSON form"),
        (np.array([1], dtype=np.int32), "arrays of int32 have no JSON form"),
    ],
)
def test_values_without_a_json_form_are_refused_on_writing(value: np.ndarray, match: str) -> None:
    with pytest.raises(SchemaError, match=match):
        dumps(value, Schema("test.array", 1, np.ndarray))


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ({"dtype": "int64", "shape": [2], "data": [1, True]}, "True is not a int64 value"),
        ({"dtype": "bool", "shape": [1], "data": [None]}, "None is not a bool value"),
        ({"dtype": "float64", "shape": [3], "data": [1.0]}, r"shape \[3\] needs 3 values"),
        ({"dtype": "float64", "shape": [1], "data": ["1"]}, "'1' is not a float64 value"),
        ({"dtype": "int64", "shape": [1], "data": [2**63]}, "9223372036854775808 is not a int64"),
        ({"dtype": "float64", "shape": [1], "data": [10**400]}, "is not a float64 value"),
        ({"dtype": "float64", "shape": [1], "data": [math.inf]}, "inf is not a float64 value"),
        ({"dtype": ["float64"], "shape": [1], "data": [1.0]}, r"unknown array dtype \['float64'\]"),
        ({"dtype": "float64", "shape": [10**400] * 11, "data": []}, "more values than an array"),
    ],
)
def test_array_data_that_does_not_fit_its_dtype_is_refused(data: dict, match: str) -> None:
    text = json.dumps({"schema": "test.array", "version": 1, "data": data})
    with pytest.raises(SchemaError, match=match):
        loads(text, Schema("test.array", 1, np.ndarray))


@pytest.mark.parametrize(
    "value",
    [math.inf, -math.inf, 10**400],
    ids=["Infinity", "-Infinity", "an integer past a float's range"],
)
def test_a_number_no_float_holds_short_of_an_infinity_is_refused(value: float) -> None:
    """A float is written as a finite number or as null for NaN. `Infinity`, which is not JSON but
    which Python's parser reads, and an integer past a float's range are refused, not read as an
    infinity that `dumps` refuses to write."""
    document = json.loads(dumps((Priced(Revenue(), 1.0, np.array([1.0])),), PRICED))
    document["data"][0]["value"] = value
    with pytest.raises(SchemaError, match=r"data\[0\]\.value: .* is past a float's finite range"):
        loads(json.dumps(document), PRICED)


PANEL = Schema("test.panel", 1, MediaPanel)


def a_panel() -> MediaPanel:
    weeks = 5
    return read_panel(
        {
            "date": np.tile(
                np.datetime64("2024-01-06") + np.timedelta64(7, "D") * np.arange(weeks), 2
            ),
            "geo": np.repeat(np.array(["a", "b"], dtype=object), weeks),
            "kpi": np.arange(10, dtype=float) + 1,
            "tv": np.arange(10, dtype=float) % 3,
        },
        date="date",
        geo="geo",
        kpi="kpi",
        spend={"tv": "tv"},
        cadence=Cadence.WEEK,
        kpi_unit=Count("orders"),
        currency="EUR",
        hierarchy={"a": "all", "b": "all"},
        gaps=Gap.STEP,
    )


def test_a_whole_panel_survives_a_round_trip() -> None:
    panel = a_panel()
    back = loads(dumps(panel, PANEL), PANEL)
    assert (back.geos, back.channels, back.cadence, back.kpi_unit) == (
        panel.geos,
        panel.channels,
        panel.cadence,
        panel.kpi_unit,
    )
    assert dict(back.hierarchy) == dict(panel.hierarchy)
    for field in ("dates", "kpi", "spend", "control_values", "filled"):
        np.testing.assert_array_equal(getattr(back, field), getattr(panel, field))


def test_reading_runs_the_dataclass_invariants() -> None:
    document = json.loads(dumps(a_panel(), PANEL))
    document["data"]["spend"]["data"][0] = -1.0
    with pytest.raises(SchemaError, match=r"(?s)MediaPanel refuses these values: .*non-negative"):
        loads(json.dumps(document), PANEL)


def test_a_panel_s_population_and_national_split_survive_a_round_trip() -> None:
    base = a_panel()
    panel = allot_national(
        replace(base, population=np.array([2.0e5, 3.0e5])), "radio", np.arange(5, dtype=float)
    )
    back = loads(dumps(panel, PANEL), PANEL)
    assert dict(back.national) == {"radio": Allotment.POPULATION}
    np.testing.assert_array_equal(back.population, panel.population)
    np.testing.assert_array_equal(back.spend, panel.spend)


def test_a_national_go_dark_test_states_a_lift_against_zero_spend_over_its_dark_weeks() -> None:
    """Two weeks dark from the third of a six-week series, read over its last four: the line names
    the weeks withheld, the spend against the plan and the readout's end. The series' first week
    falls before the panel's, where it changes nothing."""
    panel = a_panel()
    planned = np.full(6, 100.0)
    spend = planned.copy()
    spend[2:4] = 0.0
    dark = LiftEstimate(spend, planned, 4, -150.0, 30.0, end=panel.dates[4])
    (estimand,) = estimands(panel, {"tv": (dark,)})
    assert estimand.estimand == (
        "tv, the whole market: spend withheld over 2 weeks from 2024-01-13 to 2024-01-20, at 0 "
        "EUR a week against 100 planned; the lift is read over the 4 weeks of the readout to "
        "2024-02-03 as a lift against zero spend"
    )
    assert (estimand.change, estimand.reading, estimand.periods, estimand.readout) == (
        "withheld",
        "lift against zero spend",
        2,
        4,
    )
    assert (estimand.first, estimand.last, estimand.end) == (
        dt.date(2024, 1, 13),
        dt.date(2024, 1, 20),
        dt.date(2024, 2, 3),
    )


def test_a_test_in_an_unnamed_part_of_the_market_does_not_read_as_the_whole_market() -> None:
    """A test at a share below 1 whose geos are not named states its share, not the whole market."""
    panel = a_panel()
    planned = np.full(6, 100.0)
    spend = planned.copy()
    spend[2:4] = 0.0
    dark = LiftEstimate(spend, planned, 4, -150.0, 30.0, 0.25, None, panel.dates[4])
    (estimand,) = estimands(panel, {"tv": (dark,)})
    assert estimand.estimand.startswith("tv, in 0.25 of the market, its geos not named: ")
    assert "whole market" not in estimand.estimand


def test_a_geo_heavy_up_states_a_marginal_change_between_its_two_levels() -> None:
    """A test in one geo that doubles the spend over two weeks reads the curve between the
    planned level and the doubled one, not against zero spend."""
    panel = a_panel()
    planned = np.array([40.0, 40.0, 40.0, 60.0])
    spend = np.array([40.0, 40.0, 80.0, 120.0])
    heavy = LiftEstimate(spend, planned, 2, 35.0, 10.0, 0.25, frozenset({Geo("b")}), panel.dates[3])
    (estimand,) = estimands(panel, {"tv": (heavy,)})
    assert estimand.estimand == (
        "tv, in b (0.25 of the market): spend added over 2 weeks from 2024-01-20 to 2024-01-27, "
        "at 100 EUR a week against 50 planned; the lift is read over the 2 weeks of the readout "
        "to 2024-01-27 as a marginal change from 50 to 100 EUR a week"
    )
    assert (estimand.geos, estimand.share, estimand.change, estimand.reading) == (
        ("b",),
        0.25,
        "added",
        "marginal change",
    )
    assert (estimand.spend, estimand.planned) == (100.0, 50.0)


def test_a_test_s_estimands_survive_a_round_trip() -> None:
    panel = a_panel()
    planned = np.full(3, 10.0)
    tests = {
        "tv": (
            LiftEstimate(np.zeros(3), planned, 3, -9.0, 3.0, end=panel.dates[-1]),
            LiftSeries(np.array([10.0, 30.0, 30.0]), planned, np.ones(2), np.eye(2)),
        )
    }
    value = estimands(panel, tests)
    assert value[1].first is None
    assert "not placed in the panel's calendar" in value[1].estimand
    assert loads(dumps(value, LIFT_TESTS), LIFT_TESTS) == value


def test_a_test_outside_the_calendar_or_an_unnamed_reading_is_refused() -> None:
    panel = a_panel()
    early = LiftEstimate(np.zeros(6), np.ones(6), 2, -1.0, 1.0, end=panel.dates[2])
    undated = LiftEstimate(np.zeros(2), np.ones(2), 2, -1.0, 1.0, end=np.datetime64("2025-01-04"))
    for test in (early, undated):
        with pytest.raises(ValueError, match="outside the panel's calendar"):
            estimands(panel, {"tv": (test,)})
    with pytest.raises(ValueError, match="a test's change is one of"):
        Estimand("tv", (), 1.0, "halved", "marginal change", 1, 1, 1.0, 2.0, None, None, None, "")


# ------------------------------------------------------------------------- the sensitivity line


def _ridge_sensitivity(robustness_interval: float = 0.076) -> RidgeSensitivity:
    return RidgeSensitivity(
        channel="tv",
        coefficient=0.09,
        estimate=0.0973,
        se=0.0233,
        dof=783,
        level=0.9,
        partial_r2=0.022,
        robustness=0.139,
        robustness_half=0.072,
        robustness_interval=robustness_interval,
        robustness_interval_half=0.01,
        benchmarks=(Benchmark("price", 0.0092, 0.1246),),
    )


def _msm_sensitivity(gamma_star: float = 1.0146) -> MsmSensitivity:
    return MsmSensitivity(
        channel="tv",
        level=0.9,
        roi=2.5,
        gammas=np.array([1.0, 1.5]),
        lower=np.array([0.81, -0.4]),
        upper=np.array([4.2, 5.1]),
        allowance=np.array([0.0, 1.2]),
        scale=np.ones(4),
        gamma_star=gamma_star,
    )


def test_the_sensitivity_line_states_adr_0061_s_reading() -> None:
    """Both parts of ADR 0061's line, the ridge's robustness values with a control's benchmark and
    the Bayesian fit's `Gamma*` and its bound at 1.5, and the closing caveat; `Gamma*` to two
    decimals, so 1.0146 is not read as 1."""
    ridge = _ridge_sensitivity()
    moved = ridge.benchmark("price").estimate
    line = sensitivity_line(ridge, _msm_sensitivity(), control="price")
    assert line == (
        "*Sensitivity to an unmeasured confounder.* tv's return would reach 0 only under a"
        " confounder that explains 13.9% of both its residual spend and the KPI's (7.6% to bring"
        f" its 90% interval to 0); one as strong as price would move its coefficient from 0.0973"
        f" to {moved:.4g}. For the Bayesian fit: the return's 90% interval stays above 0 up to a"
        " confounding level of Gamma = 1.01; at Gamma = 1.5 it is [-0.4, 5.1]. These are bounds"
        " under a stated strength, not corrections."
    )


def test_the_sensitivity_line_writes_the_parts_it_is_given() -> None:
    ridge = sensitivity_line(_ridge_sensitivity(0.0))
    assert "its 90% interval holds 0 already" in ridge
    assert "Bayesian" not in ridge
    assert "as strong as" not in ridge
    bayes = sensitivity_line(bayes=_msm_sensitivity(1.0))
    assert "the return's 90% interval holds 0 with no confounder;" in bayes
    assert "would reach 0" not in bayes
    assert bayes.endswith("These are bounds under a stated strength, not corrections.")


def test_the_sensitivity_line_refuses_what_it_cannot_state() -> None:
    with pytest.raises(ValueError, match="needs a ridge result"):
        sensitivity_line()
    other = replace(_msm_sensitivity(), channel="search")
    with pytest.raises(ValueError, match="two channels"):
        sensitivity_line(_ridge_sensitivity(), other)
    with pytest.raises(ValueError, match="not 2"):
        sensitivity_line(bayes=_msm_sensitivity(), gamma=2.0)
    with pytest.raises(ValueError, match="controls"):
        sensitivity_line(_ridge_sensitivity(), control="season")
