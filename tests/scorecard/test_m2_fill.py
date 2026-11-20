"""M2's fill: every slot of the draft filled from a pilot laid out as the w30 pilot's roots and from
the pins, the draft left as it was, and nothing written while a slot stays open."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("causaldyn_bench")

from causaldyn_bench.scorecard.scoring import ArmScore, WorldRecord

from scorecard.layout import slug
from scorecard.preregistration import m2_fill, m2_sizing

BENCH = "0123456789abcdef0123456789abcdef01234567"
WORLDS = 12  # two past the first ten, which the forecasts leave to the ablation
GIB = 2**30
FIT = {
    "fit_wall_seconds": 600.0,
    "fit_cpu_seconds": 590.0,
    "plan_wall_seconds": 10.0,
    "plan_cpu_seconds": 10.0,
}
TOOL = {"wall_seconds": 100.0, "cpu_seconds": 150.0}
DRAFT = json.loads((m2_fill.HERE / "m2.json").read_text(encoding="utf-8"))
pytestmark = pytest.mark.skipif(
    "pending" not in DRAFT, reason="the pre-registration is filled: its draft is in the history"
)


def _roots() -> list[tuple[str, int, list[str], int, list[str], list[str], int]]:
    """Each root as the w30 pilot lays them out: name, family, environments, tested rung, arms,
    tools and worlds an environment; a two-stage and a Robyn root beside each main one, and
    Meridian's on family 10 apart, on `geos-50` too, where the w10 pilot started it and the run
    does not."""
    out = []
    for name, family in DRAFT["families"].items():
        f = int(name)
        tools = [r for r in family["rivals"] if r not in m2_sizing.BUILT_IN]
        split = {0: [("f0", ["drawn"]), ("f0ref", ["reference"])]}
        split[7] = [("f7a", ["L-52"]), ("f7b", ["L-78", "L-104"])]
        for root, environments in split.get(f, [(f"f{f}", family["environments"])]):
            tested = (1 if environments == ["L-52"] else 2) if f == 7 else 4
            out.append((root, f, environments, tested, ["dlm"], tools, WORLDS))
            out.append((f"{root}-ts", f, environments, tested, ["dlm-two-stage"], [], WORLDS))
            out.append((f"{root}-robyn", f, environments, tested, ["chc"], ["robyn"], 2))
    geos = ["geos-3", "geos-8", "geos-50"]
    out.append(("f10-meridian", 10, geos, 4, ["chc"], ["meridian"], WORLDS))
    return out


def _peak(root: str, environment: str) -> int:
    """3 GiB a process, but Meridian's on `geos-50`, which no limit reads."""
    return (18 if (root, environment) == ("f10-meridian", "geos-50") else 3) * GIB


def _record(family, environment, index, k, variants) -> WorldRecord:
    """One world-rung, the built-in arms' regrets the same in every root that holds it, the
    claim's arms 0.02 below the status quo's and every tool 0.01 above."""
    rng = np.random.default_rng([family, sum(map(ord, environment)), index, k])
    base = 0.2 + 0.01 * rng.standard_normal()
    regrets = {"status quo": base, "equal split": base + 0.01}
    for v in variants:
        own = v.startswith(("dlm", "chc"))
        regrets[v] = base + (-0.02 if own else 0.01) + 0.005 * rng.standard_normal()
    arms = {
        name: ArmScore(
            regret,
            None,
            None,
            0,
            None,
            None,
            None,
            None,
            {"error": None} | ({} if name in m2_sizing.BUILT_IN else {"cost": _cost(name)}),
        )
        for name, regret in regrets.items()
    }
    seed = 100_000 * family + 1_000 * len(environment) + 10 * sum(map(ord, environment)) + index
    return WorldRecord(family, environment, seed, index, True, k, 1.0, 1.0, 3, 0, {}, arms)


def _cost(variant: str) -> dict[str, float]:
    if variant.startswith(("dlm", "chc")):
        return FIT | {"wall_seconds": 610.0, "cpu_seconds": 600.0}
    return TOOL


def _variants(arms: list[str], tools: list[str]) -> list[str]:
    named = {"dlm": "dlm plug-in", "dlm-two-stage": "dlm two-stage", "chc": "chc"}
    return [named[a] for a in arms] + tools


def _pilot(directory: Path, leave: str | None = None) -> Path:
    """A pilot of every root `_roots` lays out, but `leave`'s, and its peaks file."""
    peaks = []
    for root, family, environments, tested, arms, tools, worlds in _roots():
        if root == leave:
            continue
        path = directory / root
        path.mkdir(parents=True)
        rungs = [tested] if arms == ["dlm-two-stage"] else [0, tested]
        declared: dict[str, object] = {
            "family": family,
            "environments": environments,
            "worlds": worlds,
            "pilot": True,
            "rungs": rungs,
            "arms": arms,
            "external": tools,
        }
        (path / "run.json").write_text(json.dumps(declared), encoding="utf-8")
        records = [
            _record(family, e, i, k, _variants(arms, tools))
            for e in environments
            for i in range(worlds)
            for k in rungs
        ]
        lines = [json.dumps(r.as_json()) for r in records]
        (path / "scores.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        peaks += [
            {
                "log": f"{path}/logs/{slug(arm)}/{r.environment}-k{r.k}-{r.seed}.log",
                "peak_bytes": _peak(root, r.environment),
            }
            for r in records
            for arm in [*arms, *tools]
        ]
    lines = [json.dumps(p) for p in peaks]
    (directory / "peaks.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return directory


def _fill(monkeypatch, pilot: Path, out: Path) -> None:
    monkeypatch.setattr(m2_sizing, "RESAMPLES", 200)
    monkeypatch.setattr(m2_fill, "DRAWS", 2000)
    argv = ["m2_fill", str(pilot), "--bench", BENCH, "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    m2_fill.main()


@pytest.fixture(scope="module")
def filled(tmp_path_factory):
    """The fill of a whole pilot, the draft's bytes before it, and the pilot."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        base = tmp_path_factory.mktemp("m2")
        pilot = _pilot(base / "pilot")
        before = {n: (m2_fill.HERE / n).read_bytes() for n in ("m2.md", "m2.json")}
        _fill(monkeypatch, pilot, base / "out")
        out = base / "out"
        doc = json.loads((out / "m2.json").read_text(encoding="utf-8"))
        return doc, (out / "m2.md").read_text(encoding="utf-8"), before, pilot


def test_every_slot_is_filled_and_the_draft_is_left_as_it_was(filled):
    doc, text, before, _ = filled
    assert "{{" not in text
    assert "(draft)" not in text
    assert "**Draft.**" not in text
    assert list(m2_fill.open_slots(doc)) == []
    assert "pending" not in doc
    assert {n: (m2_fill.HERE / n).read_bytes() for n in before} == before


def test_the_pins_are_the_lock_s_the_bench_s_and_the_pilot_s(filled):
    doc, text, _, pilot = filled
    pins = doc["pins"]
    assert pins["causal-hybrid-control"] == importlib.metadata.version("causal-hybrid-control")
    assert pins["jax"] == importlib.metadata.version("jax")
    assert pins["bench"] == BENCH
    assert pins["forecast seed"] == m2_fill.SEED == doc["forecasts"]["seed"]
    digests = {
        p.parent.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in pilot.glob("*/scores.jsonl")
    }
    assert doc["pilot digests"] == digests
    assert all(f"- `{root}`: `{digest}`" in text for root, digest in digests.items())
    assert f"causal-hybrid-control {pins['causal-hybrid-control']}." in text
    assert f"The bench at `{BENCH}`" in text


def test_each_family_declares_its_sizing_and_family_15_its_100_files(filled):
    doc, text, _, _ = filled
    for name, family in doc["families"].items():
        held = name == "15"
        assert family["declared worlds"] == (100 if held else 240), name
        assert family["exploratory"] is held
        assert doc["worlds"][name] == family["worlds an environment"]
    assert "| 15 | 1 | 10 | 100 | 100 | yes |" in text
    assert doc["launch power"] == round(0.8**11, 6)


def test_each_claim_item_carries_its_comparison_s_sizing(filled):
    doc, _, _, _ = filled
    for entry in doc["items"]:
        item = entry["item"]
        if item["second"] == "robyn" or (
            item["group"] == "family 10" and item["second"] == "meridian"
        ):
            assert entry["pilot mean"] is None
            assert "not sized" in entry["n/a"]["pilot mean"]
            continue
        environments = DRAFT["families"][item["group"].removeprefix("family ")]["environments"]
        assert (entry["pilot n"], entry["pilot missing"]) == (WORLDS * len(environments), 0)
        assert math.isfinite(entry["pilot mean"])
        assert entry["pilot sd"] > 0.0
        if item["kind"] == "claim":
            assert entry["predicted failure"] is False
            assert entry["chance at a tie"] == pytest.approx(1.0, rel=0.0, abs=0.05)


def test_every_item_but_robyn_s_is_forecast_from_the_worlds_past_the_tenth(filled):
    doc, _, _, _ = filled
    made = doc["forecasts"]
    robyn = [e for e in made["items"] if e["item"]["second"] == "robyn"]
    assert all(e["predicted"] is None and "Robyn" in e["reason"] for e in robyn)
    others = [e for e in made["items"] if e["item"]["second"] != "robyn"]
    assert all(e["predicted"] is not None and e["pilot n"] >= 2 for e in others)
    assert made["forecast items"] == len(others) == len(DRAFT["items"]) - 24
    assert (made["from"], made["draws"]) == (m2_fill.FROM, 2000)
    meridian = [
        e for e in others if e["item"]["second"] == "meridian" and e["item"]["name"] == "F10"
    ]
    assert {tuple(e["item"]["environments"]) for e in meridian} == {("geos-3", "geos-8")}
    assert {e["scored"] for e in meridian} == {160}
    for entry in others:
        if entry not in meridian:
            family = DRAFT["families"][entry["item"]["name"].removeprefix("F")]
            assert entry["item"]["environments"] == family["environments"], entry["item"]


def test_limits_and_compute_follow_the_rule(filled):
    doc, _, _, _ = filled
    dlm = doc["limits"]["0"]["dlm"]
    assert (dlm["memory"], dlm["timeout"]) == ("5G", 3660.0)  # 1.5 x 3 GiB; 6 x 610 s
    assert doc["limits"]["1"]["pymc-marketing"]["timeout"] == 600.0
    hours = doc["compute"]["by arm"]
    worlds = 11 * 240 + 100
    assert hours["dlm-two-stage"] == pytest.approx(worlds * 600.0 / 3600.0, rel=1e-12, abs=0.0)
    environments = sum(len(f["environments"]) for f in DRAFT["families"].values())
    robyn = environments * 20 * 2 * 150.0 / 3600.0
    assert hours["robyn"] == pytest.approx(robyn, rel=1e-12, abs=0.0)
    meridian = doc["limits"]["10"]["meridian"]
    assert (meridian["memory"], meridian["processes"]) == ("5G", 2 * WORLDS * 2)
    planned = 10 * 240 + 100 + 2 * 80  # family 10's on geos-3 and geos-8 only
    assert hours["meridian"] == pytest.approx(planned * 2 * 150.0 / 3600.0, rel=1e-12, abs=0.0)


def test_a_pilot_without_a_family_s_main_root_writes_nothing(tmp_path, monkeypatch):
    pilot = _pilot(tmp_path / "pilot", leave="f14")
    with pytest.raises(m2_fill.OpenSlots, match="family 14"):
        _fill(monkeypatch, pilot, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_a_root_not_scored_yet_writes_nothing(tmp_path, monkeypatch):
    pilot = _pilot(tmp_path / "pilot")
    (pilot / "f3" / "scores.jsonl").unlink()
    with pytest.raises(m2_fill.OpenSlots, match="f3"):
        _fill(monkeypatch, pilot, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_an_arm_without_a_peak_leaves_its_limit_open_and_writes_nothing(tmp_path, monkeypatch):
    pilot = _pilot(tmp_path / "pilot")
    peaks = pilot / "peaks.jsonl"
    kept = [line for line in peaks.read_text().splitlines() if "/f2/logs/orbit/" not in line]
    peaks.write_text("\n".join(kept) + "\n")
    with pytest.raises(m2_fill.OpenSlots, match=r"limits\.2\.orbit"):
        _fill(monkeypatch, pilot, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("inside", ["scorecard/preregistration", "scorecard"])
def test_the_fill_never_writes_into_the_draft(tmp_path, monkeypatch, inside):
    before = {n: (m2_fill.HERE / n).read_bytes() for n in ("m2.md", "m2.json")}
    with pytest.raises(SystemExit):
        _fill(monkeypatch, tmp_path / "pilot", m2_fill.TREE / inside)
    assert {n: (m2_fill.HERE / n).read_bytes() for n in before} == before
