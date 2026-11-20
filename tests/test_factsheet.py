"""The factsheet and the decision document (`mixdyn.api`).

Every field of each report the engine generates is named in its class's docstring, written in its
JSON document and read back the same, saved and loaded the same, and written on its page under
its name, an unfilled one with its reason. The decision document of a fixed tiny world is the
golden page `tests/golden/decision.txt`, its numbers at the four significant digits `render`
writes. After a change that moves the page on purpose, regenerate it from the tree's root:

    PYTHONPATH=src python tests/test_factsheet.py > tests/golden/decision.txt
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mixdyn import NATIONAL, Cadence, Channel, MediaPanel, Revenue
from mixdyn.api import (
    DECISION,
    FACTSHEET,
    Compute,
    Decision,
    Event,
    Factsheet,
    Holdout,
    PowerSummary,
    Reconciliation,
    Refresh,
    Schema,
    SchemaError,
    Unfilled,
    decision,
    dumps,
    factsheet,
    loads,
    reconcile,
    render,
    sensitivity_line,
)
from mixdyn.api.serialize import load, save
from mixdyn.calibration import LiftEstimate
from mixdyn.decisions import Range, Readings, plan_posterior
from mixdyn.experiments import GroupNoise, Within, choose_tests, designs
from mixdyn.inference.diagnostics import (
    HealthCard,
    diagnose,
    negative_baseline,
    posterior_predictive_p,
    prior_shift,
)
from mixdyn.inference.sensitivity import MsmSensitivity
from mixdyn.models.bayes_mmm import (
    BayesMedia,
    BayesSpec,
    HalfNormalRoi,
    HillCurve,
    LogNormal,
    RoiPrior,
    StudentT,
    TanhCurve,
)
from mixdyn.models.ridge_mmm import Control, FourierSeason, LinearTrend

GOLDEN = Path(__file__).parent / "golden" / "decision.txt"
WEEKS = 20
BASE = 1000.0
CHANNELS = (Channel("tv"), Channel("search"))
DATES = np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(WEEKS)


def _spend() -> np.ndarray:
    return np.random.default_rng(20261106).uniform(20.0, 80.0, (WEEKS, 2))


def _readings() -> Readings:
    """Sixteen Hill draws of one fit, two chains of eight, each weighing the same."""
    rng = np.random.default_rng(20261107)
    return Readings(
        channels=CHANNELS,
        retention=rng.uniform(0.2, 0.6, (16, 2)),
        half_saturation=rng.uniform(30.0, 60.0, (16, 2)),
        slope=rng.uniform(0.8, 1.2, (16, 2)),
        coefficient=rng.uniform(80.0, 160.0, (16, 2)),
        tanh=np.zeros(16, dtype=bool),
        fit=np.zeros(16, dtype=np.intp),
        chain=np.repeat([0, 1], 8),
        stacking=np.ones(1),
        pointwise=None,
        length=6,
    )


def _contributions(readings: Readings) -> dict[str, np.ndarray]:
    """Each channel's contribution at each week, the draws' mean: the spend carried over by the
    normalised geometric kernel of the readings' length and saturated by each draw's Hill."""
    spend, out = _spend(), {}
    for c, channel in enumerate(readings.channels):
        draws = []
        for s in range(readings.weights.size):
            kernel = readings.retention[s, c] ** np.arange(readings.length)
            adstock = np.convolve(spend[:, c], kernel / kernel.sum())[:WEEKS]
            z = (adstock / readings.half_saturation[s, c]) ** readings.slope[s, c]
            draws.append(readings.coefficient[s, c] * z / (1.0 + z))
        out[str(channel)] = np.average(draws, axis=0, weights=readings.weights)
    return out


def _panel() -> MediaPanel:
    contributions = _contributions(_readings())
    noise = 5.0 * np.random.default_rng(20261108).standard_normal(WEEKS)
    kpi = BASE + contributions["tv"] + contributions["search"] + noise
    price = np.random.default_rng(20261109).normal(10.0, 1.0, WEEKS)
    return MediaPanel(
        dates=DATES,
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=CHANNELS,
        controls=("price",),
        kpi=kpi[None, :],
        kpi_unit=Revenue(),
        spend=_spend()[None],
        currency="EUR",
        control_values=price[None, :, None],
        filled=np.zeros((1, WEEKS), dtype=bool),
    )


def _card() -> HealthCard:
    """A health card's cells read by the card's own arithmetic off draws of a fixed seed."""
    rng = np.random.default_rng(20261110)
    ppp, ppp_se = posterior_predictive_p(rng.normal(20_100.0, 150.0, 400), 20_000.0)
    share, periods = negative_baseline(rng.normal(900.0, 400.0, (400, WEEKS)))
    shifts = {
        c: prior_shift(rng.lognormal(0.2, 0.9, 400), rng.normal(m, 0.3, 400))
        for c, m in (("tv", 1.4), ("search", 2.6))
    }
    return HealthCard(
        max_rhat=float(np.nanmax(diagnose(rng.standard_normal((2, 100, 3))).rhat)),
        ppp=ppp,
        ppp_se=ppp_se,
        total=20_000.0,
        totals=rng.normal(20_100.0, 150.0, 400),
        negative_baseline=share,
        negative_baseline_periods=periods,
        shift={c: s[0] for c, s in shifts.items()},
        width={c: s[1] for c, s in shifts.items()},
        consistency={c: s[2] for c, s in shifts.items()},
    )


def world_decision() -> Decision:
    """The decision document of the tiny world: twenty weeks of two channels, a sixteen-draw
    posterior, its plan of the next four weeks within half and one and a half times the current
    spend, its health card read at Meridian's reviewer's thresholds, its contributions reconciled
    to the KPI, the next lift tests by their value and a recorded compute."""
    panel, readings = _panel(), _readings()
    current = panel.spend[0, -4:].mean(axis=0)
    limits = {
        c: Range(0.5 * float(x), 1.5 * float(x)) for c, x in zip(CHANNELS, current, strict=True)
    }
    budget = 4.0 * float(current.sum())
    plan = plan_posterior(panel, NATIONAL, readings, budget, 4, limits, grid=64)
    candidates = designs(
        panel,
        NATIONAL,
        CHANNELS,
        GroupNoise(5.0),
        when=Within(WEEKS - 8),
        multipliers=(0.0, 2.0),
        durations=(2,),
        cooldown=2,
        shares=(0.2,),
    )
    choice = choose_tests(
        panel,
        NATIONAL,
        readings,
        candidates,
        budget,
        4,
        limits,
        test_budget=1e9,
        seed=1,
        draws=16,
        grid=16,
    )
    parts = {"base": np.full(WEEKS, BASE), **_contributions(readings)}
    return decision(
        panel,
        NATIONAL,
        plan,
        title="the tiny world's next four weeks",
        card=_card(),
        thresholds={"max_rhat": 1.2, "min_ppp": 0.05, "max_negative_baseline": 0.2},
        reconciliation=reconcile(panel.dates, panel.kpi[0], parts),
        choice=choice,
        events=(
            Event(dt.date(2024, 5, 13), "fitted on twenty weeks"),
            Event(dt.date(2024, 5, 14), "planned the next four weeks"),
        ),
        compute=Compute("cpu", 12.5),
    )


def world_factsheet(*, filled: bool = True) -> Factsheet:
    """The tiny world's factsheet: filled, a model of Hill curves, a trend, a control and
    Student-t noise with a lift test of tv and every record given; or, unfilled, a model of
    `tanh` curves, a season and log-normal noise with no test and no record."""
    panel, rng = _panel(), np.random.default_rng(20261111)
    returns = {"tv": rng.lognormal(0.3, 0.2, 400), "search": rng.lognormal(0.9, 0.3, 400)}
    if not filled:
        spec = BayesSpec(
            (BayesMedia("tv"), BayesMedia("search")),
            season=FourierSeason(period=13.0, harmonics=2),
            likelihood=LogNormal(),
            saturation=TanhCurve(),
        )
        return factsheet(panel, NATIONAL, spec, returns)
    spec = BayesSpec(
        (BayesMedia("tv"), BayesMedia("search", RoiPrior(0.5, 0.4, "carried"))),
        controls=(Control("price", upper=0.0),),
        trend=LinearTrend(knots=1),
        likelihood=StudentT(),
    )
    planned = np.full(6, 50.0)
    test = LiftEstimate(
        np.r_[np.zeros(4), planned[4:]], planned, 6, -310.0, 40.0, end=np.datetime64("2024-04-15")
    )
    return factsheet(
        panel,
        NATIONAL,
        spec,
        returns,
        origins={"kpi": "the shop's orders summed by week", "tv": "the agency's invoices"},
        estimates={"tv": [test]},
        holdout=Holdout(
            dt.date(2024, 4, 22), dt.date(2024, 5, 13), 4, "mean absolute percentage error", 0.031
        ),
        power=PowerSummary(26, 120.0, 340.0),
        refresh=Refresh(4, "week", dt.date(2024, 5, 13)),
        limitations=("the world is twenty weeks long",),
    )


def _reports() -> list[tuple[Any, Schema]]:
    panel = _panel()
    plan = world_plan(panel)
    return [
        (world_factsheet(), FACTSHEET),
        (world_factsheet(filled=False), FACTSHEET),
        (world_decision(), DECISION),
        (decision(panel, NATIONAL, plan, title="a plan alone"), DECISION),
        (decision(panel, NATIONAL, _moving_search(plan), title="a plan on a prior"), DECISION),
    ]


def _moving_search(plan: Any) -> Any:
    """`plan` as one that names search as a channel it moves on its prior alone."""
    return dataclasses.replace(plan, unidentified=(CHANNELS[1],))


@pytest.fixture(scope="module")
def reports() -> list[tuple[Any, Schema]]:
    return _reports()


def _labels(value: Any, depth: int) -> list[str]:
    """The line each field of `value` starts on its page, in order, as the page's rules write it:
    a field under its name, indented two spaces a level, an item of a tuple after a dash, an
    unfilled field with its reason, and a reconciliation's periods under a table's head."""
    pad, out = "  " * depth, []
    for f in dataclasses.fields(value):
        item = getattr(value, f.name)
        if isinstance(item, Unfilled):
            out.append(f"{pad}{f.name}: unfilled: {item.reason}")
        elif isinstance(item, Reconciliation):
            head = ["date", "kpi", *item.parts, "total", "residual"]
            out += [f"{pad}{f.name}:", f"{pad}  parts:", f"{pad}  periods:"]
            out.append(f"{pad}    | {' | '.join(head)} |")
            out += [f"{pad}    | {p.date.isoformat()} |" for p in item.periods]
        elif dataclasses.is_dataclass(item):
            out += [f"{pad}{f.name}:", *_labels(item, depth + 1)]
        elif isinstance(item, tuple) and item and dataclasses.is_dataclass(item[0]):
            out.append(f"{pad}{f.name}:")
            for element in item:
                inner = _labels(element, depth + 2)
                out += [f"{pad}  - {inner[0].lstrip()}", *inner[1:]]
        elif isinstance(item, tuple | Mapping) and not item:
            out.append(f"{pad}{f.name}: none")
        elif isinstance(item, Mapping):
            out += [f"{pad}{f.name}:", *(f"{pad}  {k}: " for k in item)]
        elif isinstance(item, tuple):
            out += [f"{pad}{f.name}:", *(f"{pad}  - {x}" for x in item)]
        else:
            out.append(f"{pad}{f.name}: ")
    return out


def _in_order(page: str, labels: list[str]) -> list[str]:
    """The labels no line of `page` starts with, each after the line the label before it found."""
    lines, at, missing = page.splitlines(), 0, []
    for label in labels:
        found = next((i for i in range(at, len(lines)) if lines[i].startswith(label)), None)
        if found is None:
            missing.append(label)
        else:
            at = found + 1
    return missing


def _fields(value: Any) -> list[tuple[type, str]]:
    """Each class and field the dataclasses under `value` hold, depth first."""
    out = []
    for f in dataclasses.fields(value):
        out.append((type(value), f.name))
        item = getattr(value, f.name)
        for element in item if isinstance(item, tuple) else (item,):
            if dataclasses.is_dataclass(element):
                out += _fields(element)
    return out


def _keys(value: Any, data: Any, where: str) -> list[str]:
    """Where the JSON `data` of `value` misses a field the dataclass holds, or holds another."""
    names = [f.name for f in dataclasses.fields(value)]
    if not isinstance(data, dict) or set(data) - {"kind"} != set(names):
        return [where]
    out = []
    for name in names:
        item = getattr(value, name)
        if dataclasses.is_dataclass(item):
            out += _keys(item, data[name], f"{where}.{name}")
        elif isinstance(item, tuple):
            out += [
                miss
                for i, element in enumerate(item)
                if dataclasses.is_dataclass(element)
                for miss in _keys(element, data[name][i], f"{where}.{name}[{i}]")
            ]
    return out


def test_every_field_of_every_report_is_on_its_page_and_in_its_document(
    reports: list[tuple[Any, Schema]], tmp_path: Path
) -> None:
    for report, schema in reports:
        page = render(report)
        assert page.splitlines()[0] == type(report).__name__.lower()
        assert _in_order(page, _labels(report, 0)) == []
        text = dumps(report, schema)
        assert _keys(report, json.loads(text)["data"], "data") == []
        assert loads(text, schema) == report
        save(report, tmp_path / "report")
        assert load(tmp_path / "report") == report


def test_every_field_is_named_in_its_class_s_docstring(reports: list[tuple[Any, Schema]]) -> None:
    """Each class of this change states each of its fields; `Estimand`, `LIFT_TESTS`' since its
    first version, names its fields in its own words."""
    unnamed = {
        f"{cls.__name__}.{name}"
        for report, _ in reports
        for cls, name in _fields(report)
        if cls.__name__ != "Estimand" and f"`{name}`" not in (cls.__doc__ or "")
    }
    assert sorted(unnamed) == []


def test_the_reports_cover_every_field_both_filled_and_unfilled(
    reports: list[tuple[Any, Schema]],
) -> None:
    """Between them, the reports hold each union field both filled and unfilled, and every class
    the documents declare."""
    seen = {
        (cls, name, isinstance(getattr(obj, name), Unfilled)) for obj, cls, name in _walk(reports)
    }
    unions = {(cls, name) for cls, name, unfilled in seen if unfilled}
    assert all((cls, name, False) in seen for cls, name in unions)
    assert {cls.__name__ for cls, _, _ in seen} >= {
        "Factsheet",
        "Source",
        "Prior",
        "Estimand",
        "Holdout",
        "ReturnRange",
        "PowerSummary",
        "Refresh",
        "Decision",
        "Spend",
        "Health",
        "Reconciliation",
        "Period",
        "NextTest",
        "Event",
        "Compute",
    }
    assert {name for cls, name in unions} == {
        "holdout",
        "power",
        "refresh",
        "health",
        "reconciliation",
        "next_test",
        "compute",
    }


def _walk(reports: list[tuple[Any, Schema]]) -> list[tuple[Any, type, str]]:
    out: list[tuple[Any, type, str]] = []

    def visit(value: Any) -> None:
        for f in dataclasses.fields(value):
            out.append((value, type(value), f.name))
            item = getattr(value, f.name)
            for element in item if isinstance(item, tuple) else (item,):
                if dataclasses.is_dataclass(element):
                    visit(element)

    for report, _ in reports:
        visit(report)
    return out


def test_the_decision_document_of_the_tiny_world_is_the_golden_page() -> None:
    assert render(world_decision()) == GOLDEN.read_text(encoding="utf-8")


def test_the_decision_names_the_channels_its_plan_moves_on_their_prior_alone() -> None:
    """The decision names, by name, the channels its plan names in `unidentified`, and its page
    lists them under it; the tiny world's plan names none, and its page says so. The field joined
    the schema at version 1, before the first release (ADR 0063), so a document written without it
    is refused, the field named."""
    panel = _panel()
    plan = world_plan(panel)
    assert plan.unidentified == ()
    assert decision(panel, NATIONAL, plan, title="t").unidentified == ()
    assert "\nunidentified: none\n" in render(world_decision())
    named = decision(panel, NATIONAL, _moving_search(plan), title="t")
    assert named.unidentified == ("search",)
    assert render(named).endswith("\nunidentified:\n  - search\n")
    assert loads(dumps(named, DECISION), DECISION) == named
    document = json.loads(dumps(named, DECISION))
    del document["data"]["unidentified"]
    with pytest.raises(SchemaError, match=r"data: Decision has the fields .*'unidentified'"):
        loads(json.dumps(document), DECISION)


def test_the_reconciliation_sums_the_parts_to_the_kpi_by_period() -> None:
    decided = world_decision()
    table = decided.reconciliation
    assert isinstance(table, Reconciliation)
    assert table.parts == ("base", "tv", "search")
    for p in table.periods:
        assert p.total == (p.parts[0] + p.parts[1]) + p.parts[2]
        assert p.kpi - p.total == p.residual
    noise = 5.0 * np.random.default_rng(20261108).standard_normal(WEEKS)
    assert np.allclose([p.residual for p in table.periods], noise, rtol=0.0, atol=1e-9)


def test_the_health_checks_read_the_card_at_the_thresholds_given() -> None:
    card = _card()
    decided = world_decision()
    assert decided.health != Unfilled("no health card was read")
    health = decided.health
    assert not isinstance(health, Unfilled)
    assert health.flags == card.flags(max_rhat=1.2, min_ppp=0.05, max_negative_baseline=0.2)
    assert set(health.flags) == {"rhat", "ppp", "negative_baseline"}
    assert health.max_rhat == card.max_rhat
    assert health.shift == dict(card.shift)


def test_the_power_audit_is_unfilled_with_its_reason_until_one_is_given() -> None:
    sheet = world_factsheet(filled=False)
    assert isinstance(sheet.power, Unfilled)
    assert "power audit" in sheet.power.reason
    assert "the lift tests' power was not audited" in sheet.limitations
    assert isinstance(world_factsheet().power, PowerSummary)


def test_the_factsheet_states_the_models_priors_with_their_provenance() -> None:
    sheet = world_factsheet()
    priors = {p.parameter: p for p in sheet.priors}
    assert priors["roi[tv]"].distribution.startswith("LogNormal(0.2, 0.9)")
    assert "Meridian" in priors["roi[tv]"].provenance
    assert priors["roi[search]"].provenance == "stated by the caller"
    assert {"slope", "nu", "control", "calendar"} <= set(priors)
    assert [r.channel for r in sheet.returns] == ["tv", "search"]
    assert all(r.low <= r.mean <= r.high for r in sheet.returns)
    (test,) = sheet.tests
    assert (test.channel, test.change, test.periods) == ("tv", "withheld", 4)
    assert sheet.limitations[0] == "the world is twenty weeks long"
    assert any("no lift test calibrates search" in line for line in sheet.limitations)
    bare = {p.parameter for p in world_factsheet(filled=False).priors}
    assert not {"slope", "nu", "control"} & bare


def test_the_factsheet_states_a_hill_s_slope_prior_as_its_curve_states_it() -> None:
    panel, rng = _panel(), np.random.default_rng(20261116)
    returns = {"tv": rng.lognormal(0.3, 0.2, 400), "search": rng.lognormal(0.9, 0.3, 400)}
    media = (BayesMedia("tv"), BayesMedia("search"))
    default = {p.parameter: p for p in factsheet(panel, NATIONAL, BayesSpec(media), returns).priors}
    assert default["slope"].distribution == "LogNormal(0, 0.5), each channel's Hill's"
    assert default["slope"].provenance != "stated by the caller"
    spec = BayesSpec(media, saturation=HillCurve(-0.3, 0.4))
    priors = {p.parameter: p for p in factsheet(panel, NATIONAL, spec, returns).priors}
    assert priors["slope"].distribution == "LogNormal(-0.3, 0.4), each channel's Hill's"
    assert priors["slope"].provenance == "stated by the caller"


def test_the_factsheet_states_a_half_normal_return_prior_as_one() -> None:
    panel, rng = _panel(), np.random.default_rng(20261111)
    returns = {"tv": rng.lognormal(0.3, 0.2, 400), "search": rng.lognormal(0.9, 0.3, 400)}
    spec = BayesSpec((BayesMedia("tv", HalfNormalRoi()), BayesMedia("search")))
    priors = {p.parameter: p for p in factsheet(panel, NATIONAL, spec, returns).priors}
    assert priors["roi[tv]"].distribution == (
        "HalfNormal(1.81085), the window's contribution over its spend"
    )
    assert priors["roi[tv]"].provenance == "stated by the caller"
    assert priors["roi[search]"].distribution.startswith("LogNormal(0.2, 0.9)")


def _bound(channel: str, gamma_star: float) -> MsmSensitivity:
    """A marginal sensitivity result of `channel` read at `Gamma` 1 and 1.5, by hand."""
    return MsmSensitivity(
        channel=channel,
        level=0.9,
        roi=2.5,
        gammas=np.array([1.0, 1.5]),
        lower=np.array([0.81, -0.4]),
        upper=np.array([4.2, 5.1]),
        allowance=np.array([0.0, 1.2]),
        scale=np.ones(4),
        gamma_star=gamma_star,
    )


def test_the_factsheet_states_each_channel_s_sensitivity_to_a_confounder() -> None:
    """ADR 0061's line, with its `Gamma*`, for each channel a marginal sensitivity result is given
    of: after the engine's line on the confounding no column holds, in the model's order, and on
    the page. A result of a channel the model does not read, two of one channel, or one not read
    at `Gamma = 1.5` is refused."""
    panel, rng = _panel(), np.random.default_rng(20261111)
    returns = {"tv": rng.lognormal(0.3, 0.2, 400), "search": rng.lognormal(0.9, 0.3, 400)}
    spec = BayesSpec((BayesMedia("tv"), BayesMedia("search")))
    tv, search = _bound("tv", 1.0), _bound("search", 1.2345)
    plain = factsheet(panel, NATIONAL, spec, returns).limitations
    sheet = factsheet(panel, NATIONAL, spec, returns, sensitivity=(search, tv))
    at = next(i for i, line in enumerate(plain) if "no column holds" in line) + 1
    lines = (sensitivity_line(bayes=tv), sensitivity_line(bayes=search))
    assert sheet.limitations == (*plain[:at], *lines, *plain[at:])
    assert "up to a confounding level of Gamma = 1.23; at Gamma = 1.5 it is" in lines[1]
    assert "holds 0 with no confounder" in lines[0]
    assert f"  - {lines[1]}" in render(sheet).splitlines()
    for given in ((tv, tv), (_bound("radio", 1.1),)):
        with pytest.raises(ValueError, match="once at most for each of the model's channels"):
            factsheet(panel, NATIONAL, spec, returns, sensitivity=given)
    unread = dataclasses.replace(tv, gammas=np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match=r"not 1\.5"):
        factsheet(panel, NATIONAL, spec, returns, sensitivity=(unread,))


def test_a_report_refuses_what_it_cannot_state() -> None:
    panel, spec = _panel(), BayesSpec((BayesMedia("tv"), BayesMedia("search")))
    returns = {"tv": np.ones(4), "search": np.ones(4)}
    with pytest.raises(ValueError, match="returns are read for the model's channels"):
        factsheet(panel, NATIONAL, spec, {"tv": np.ones(4)})
    with pytest.raises(ValueError, match="each finite"):
        factsheet(panel, NATIONAL, spec, {"tv": np.ones(4), "search": np.array([np.nan])})
    with pytest.raises(ValueError, match="origins name columns"):
        factsheet(panel, NATIONAL, spec, returns, origins={"radio": "?"})
    with pytest.raises(ValueError, match="which the model does not read"):
        factsheet(panel, NATIONAL, spec, returns, estimates={"radio": []})
    with pytest.raises(ValueError, match="thresholds"):
        decision(panel, NATIONAL, world_plan(panel), title="t", thresholds={"rhat": 1.2})
    with pytest.raises(ValueError, match="as long as"):
        reconcile(panel.dates, panel.kpi[0], {"tv": np.ones(3)})
    with pytest.raises(ValueError, match="why it is unfilled"):
        Unfilled(" ")
    with pytest.raises(SchemaError, match=r"not a 'mixdyn\.decision'"):
        loads(dumps(world_factsheet(), FACTSHEET), DECISION)


def world_plan(panel: MediaPanel) -> Any:
    return plan_posterior(
        panel, NATIONAL, _readings(), 200.0, 4, {c: Range(10.0, 90.0) for c in CHANNELS}, grid=32
    )


if __name__ == "__main__":
    import jax

    jax.config.update("jax_enable_x64", True)
    sys.stdout.write(render(world_decision()))
