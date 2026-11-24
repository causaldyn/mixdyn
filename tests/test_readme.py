"""The README states what the engine was and was not validated on, and points to the pages a user
reads first; its example runs; the scope page states what the engine does not model.

Its record of the runs its claim was read on, M4, M5 and M3's geo verdicts, states the conflict of
interest, the error bars and the engine's place. Its tables of the claim, the worlds per cell, the
words, the losses, the tuned rivals, the versions and the compute are the ones
`scripts/readme_record.py` writes from the runs' records (`evidence/results/m4_record.json`,
`m5_record.json` and `m3_record.json`): a table typed into one of those blocks fails here. Each
cell of the claim's table is checked against its run's reading apart from the script that writes
it, and so is each count, interval and level the prose beside the tables states."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

import readme_record

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
# The README's text with every run of whitespace one space: a phrase reads the same wherever the
# prose wraps it.
FLAT = " ".join(README.split())
RECORDS = readme_record.records()
MINUS = "\u2212"


def sections(text: str) -> list[str]:
    return re.findall(r"^## (.+)$", text, flags=re.MULTILINE)


def test_the_readme_says_there_is_no_real_data_validation_and_names_the_protocol() -> None:
    assert "Real-data validation: none, and why" in sections(README)
    assert "`scripts/real_data_protocol.py`" in README
    assert (ROOT / "scripts" / "real_data_protocol.py").is_file()


def test_every_page_the_readme_points_to_exists() -> None:
    pages = ("SKILL.md", "docs/api/index.md", "docs/scope.md", "evidence/")
    assert [p for p in pages if f"`{p}`" not in README] == []
    assert [p for p in pages if not (ROOT / p).exists()] == []


def test_reach_and_frequency_are_stated_out_of_scope() -> None:
    scope = (ROOT / "docs" / "scope.md").read_text(encoding="utf-8")
    assert "Reach and frequency" in sections(scope)
    assert "Meridian" in scope


def test_its_example_runs_and_plans_inside_its_box_at_the_budget() -> None:
    code = re.findall(r"^```python\n(.*?)^```$", README, flags=re.DOTALL | re.MULTILINE)
    assert len(code) == 1
    namespace: dict[str, object] = {"__name__": "readme"}
    exec(compile(code[0], "README.md", "exec"), namespace)
    plan, low, high = (np.asarray(namespace[name]) for name in ("plan", "low", "high"))
    budget = float(np.asarray(namespace["status_quo"]).sum())
    assert np.all(low <= plan)
    assert np.all(plan <= high)
    assert abs(float(plan.sum()) - budget) <= 1e-9 * budget


def section(text: str, heading: str) -> str:
    """The text under `## heading`, up to the next heading of its level."""
    return text.split(f"\n## {heading}\n", 1)[1].split("\n## ", 1)[0]


def test_the_record_s_tables_are_the_ones_its_script_writes() -> None:
    """Run `python scripts/readme_record.py` when a record or the script changes."""
    assert readme_record.blocks(README) == readme_record.render(RECORDS)


def test_each_record_table_sits_in_the_section_that_names_it() -> None:
    placed = {
        "claim": "What has been measured",
        "cells": "What has been measured",
        "words": "What has been measured",
        "losses": "Where it loses",
        "exploratory": "Where it loses",
        "tuned": "The tuned rivals",
        "m4-versions": "Versions of every arm",
        "m5-versions": "Versions of every arm",
        "versions": "Versions of every arm",
        "m4-compute": "Compute",
        "m5-compute": "Compute",
        "compute": "Compute",
    }
    assert sorted(placed) == sorted(readme_record.RENDERERS)
    assert [b for b, h in placed.items() if f"<!-- record:{b} -->" not in section(README, h)] == []


def test_where_it_loses_sits_beside_the_claim_and_folds_nothing() -> None:
    assert (
        sections(README).index("Where it loses")
        == sections(README).index("What has been measured") + 1
    )
    assert "<details" not in README


def test_the_readme_states_its_conflict_of_interest_its_error_bars_and_its_place() -> None:
    assert "the bench that scores it" in section(README, "Conflict of interest")
    measured = section(README, "What has been measured")
    assert "**The error bars.**" in measured
    assert "The variability" in measured
    assert "**Where it sits.**" in README
    assert "multi-touch attribution (MTA)" in README


def test_best_is_said_only_where_a_claim_does_not_say_it() -> None:
    said = re.findall(r"[^.\n]*\bbest\b[^.\n]*", README, flags=re.IGNORECASE)
    assert [s.strip() for s in said] == [
        'Nothing here says "best"',
        "It does not say that mixdyn is the best MMM, and it does not say",
    ]


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, held in value.items():
            yield key
            yield from _strings(held)
    elif isinstance(value, list):
        for held in value:
            yield from _strings(held)


def test_the_records_name_no_local_path() -> None:
    local = re.compile(r"^[/~]|/home/|\.cache|\.internal")
    assert [s for s in _strings(RECORDS) if local.search(s)] == []


def test_the_core_hours_the_text_states_are_the_records() -> None:
    for run, record in RECORDS.items():
        hours = sum(
            f["core_hours"]["engine"] + f["core_hours"]["rivals"]
            for f in record["compute"]["families"].values()
        )
        stated = f"{round(hours):,}".replace(",", " ")
        assert f"{readme_record.RUNS[run].name} took {stated} CPU-hours" in README


def test_the_reader_s_lines_are_read_field_by_field() -> None:
    reading = "\n".join(
        (
            "  family 9, k = 4, geo eb less pymc-marketing         +0.0052 [+0.0016, +0.0089]; "
            "within 0.05; loses to, unadjusted (level 0.0012); 234 complete cases, the rival "
            "failed 6, the default 0; the status quo for a failed plan: +0.0031 [-0.0010, "
            "+0.0072], within; by Hall's reader, the interval Student's",
            "  family 19, k = 0, margin held less meridian tuned         not compared: the rival "
            "failed on most worlds; 0 complete cases, the rival failed 180, the default 0, not "
            "compared in flat-meta; on its cap, 60 worlds an environment; by Hall's reader, the "
            "interval Student's",
            "the claim by family and rung, each an intersection-union test:",
            "  family 9, k = 4: holds (level 1e-12; 5 rivals, each within unadjusted)",
            "  family 4, k = 0: not claimed; not claimed, predicted to fail: status quo, orbit",
            "no headline: some family or rung is not claimed whole or does not hold",
            "losses by Holm's step-down at familywise 0.025 over 7 comparisons: family 9 k 4 "
            "pymc-marketing, family 4 k 0 equal split",
            "",
            "secondary readings, reported beside the claim and never read by it:",
            "  every stage of every fit past the bounds at the status quo: 5 of 5 claimed "
            "comparisons within; not within: none",
            "  at the margin 0.01: 4 of 5 claimed comparisons within; not within: family 9 k 4 "
            "pymc-marketing",
        )
    )
    read = readme_record._reading(reading)
    first, second = read["comparisons"]
    assert (first["family"], first["k"], first["arm"], first["rival"]) == (
        9,
        4,
        "geo eb",
        "pymc-marketing",
    )
    assert (first["mean"], first["low"], first["high"], first["word"]) == (
        "+0.0052",
        "+0.0016",
        "+0.0089",
        "loses to",
    )
    assert (first["complete"], first["rival_failed"], first["cap"]) == (234, 6, None)
    assert first["loss level"] == "0.0012"
    assert (second["mean"], second["word"], second["rival_failed"], second["cap"]) == (
        None,
        "not compared",
        180,
        60,
    )
    assert second["loss level"] is None
    assert read["verdicts"] == [
        {"family": 9, "k": 4, "word": "holds", "level": "1e-12"},
        {"family": 4, "k": 0, "word": "not claimed", "level": None},
    ]
    assert read["headline"].startswith("no headline")
    assert read["holm"] == {
        "familywise": "0.025",
        "over": 7,
        "losses": [
            {"family": 9, "k": 4, "rival": "pymc-marketing"},
            {"family": 4, "k": 0, "rival": "equal split"},
        ],
    }
    assert read["secondary"] == {
        "every stage of every fit past the bounds at the status quo": {
            "within": 5,
            "of": 5,
            "not within": [],
        },
        "at the margin 0.01": {
            "within": 4,
            "of": 5,
            "not within": [{"family": 9, "k": 4, "rival": "pymc-marketing"}],
        },
    }


def _found(run: str) -> dict[tuple[int, int, str], dict[str, Any]]:
    return {(c["family"], c["k"], c["rival"]): c for c in RECORDS[run]["reading"]["comparisons"]}


def _verdicts(run: str) -> dict[tuple[int, int], dict[str, Any]]:
    return {(v["family"], v["k"]): v for v in RECORDS[run]["reading"]["verdicts"]}


def _holds(run: str, *, geo: bool | None = None) -> list[tuple[int, int]]:
    found = _found(run)
    return [
        fk
        for fk, v in _verdicts(run).items()
        if v["word"] == "holds" and geo in (None, found[*fk, "status quo"]["arm"] == "geo eb")
    ]


def _number(text: str) -> float:
    return float(text.replace(MINUS, "-"))


def test_the_claim_s_table_reads_as_the_runs_readings() -> None:
    """Each row is a verdict that holds in the run it names, M4's, M5's or M3's on the geo
    families; its worlds are the status quo's complete cases and the arm's failures, its level the
    verdict's, and each cell the reading's comparison as printed, its Holm's loss Holm's: read
    apart from the script that writes the table."""
    runs = {s.name: run for run, s in readme_record.RUNS.items()}
    rows = readme_record.blocks(README)["claim"].splitlines()[2:]
    seen, off = [], []
    for line in rows:
        cells = [c.strip() for c in line.strip("|").split("|")]
        run, family = runs[cells[3]], int(cells[0].split()[0])
        rung = {"0": 0, "4": 4, "1\u20132": 1}[cells[1]]
        found, verdict = _found(run), _verdicts(run)[family, rung]
        holm = {
            (h["family"], h["k"], h["rival"]) for h in RECORDS[run]["reading"]["holm"]["losses"]
        }
        seen.append((run, family, rung))
        quo = found[family, rung, "status quo"]
        if int(cells[4]) != quo["complete"] + quo["default_failed"]:
            off.append((family, rung, "worlds"))
        if (verdict["word"], verdict["level"]) != ("holds", cells[5]):
            off.append((family, rung, "level"))
        for rival, cell in zip(readme_record.RIVALS, cells[6:], strict=True):
            c = found[family, rung, rival]
            if c["mean"] is None:
                if not cell.startswith("not compared"):
                    off.append((family, rung, rival))
                continue
            typed = re.findall(r"[+\u2212]\d+\.\d+", cell)
            read = [float(c[k]) for k in ("mean", "low", "high")]
            if [_number(t) for t in typed] != read or c["arm"] != cells[2].strip("`"):
                off.append((family, rung, rival))
            if ("Holm's loss" in cell) != ((family, rung, rival) in holm):
                off.append((family, rung, rival, "holm"))
    assert off == []
    m4 = [(f, k) for run, f, k in seen if run == "m4"]
    m5 = [(f, k) for run, f, k in seen if run == "m5"]
    m3 = [(f, k) for run, f, k in seen if run == "m3"]
    assert sorted(m4) == sorted(_holds("m4"))
    assert sorted(m5) == sorted(_holds("m5"))
    assert sorted(m3) == sorted(_holds("m3", geo=True))
    assert (len(m4), len(m5), len(m3)) == (26, 5, 6)
    # No family-rung is claimed twice: M5 claims only what M4 left open.
    assert len(set(m4) | set(m5)) == len(m4) + len(m5)


def _holm(levels: list[float], familywise: float) -> int:
    """How many of the smallest `levels` Holm's step-down rejects at `familywise`."""
    held = sorted(levels)
    for i, level in enumerate(held):
        if level > familywise / (len(held) - i):
            return i
    return len(held)


def test_the_counts_the_text_states_are_the_readings() -> None:
    m4 = RECORDS["m4"]["reading"]
    holds4, geo3 = _holds("m4"), _holds("m3", geo=True)
    assert f"**M4**: {len(holds4)} verdicts on {len({f for f, _ in holds4})} national" in README
    assert f"All {len(holds4)} held; the" in README
    assert {v["word"] for v in m4["verdicts"]} == {"holds", "not claimed"}
    assert f"sequence of {len(_holds('m3'))} verdicts, all of which held" in README
    assert (len(geo3), sorted({f for f, _ in geo3})) == (6, [8, 9, 10])
    holds5 = _holds("m5")
    assert f"on {len(holds4) + len(holds5) + len(geo3)} pre-registered\nfamily-rungs" in README
    prereg = {
        r: s.preregistration.read_text(encoding="utf-8") for r, s in readme_record.RUNS.items()
    }
    assert "expected 25.23" in README
    assert "25.23 of 26" in prereg["m4"]
    assert "(16.94 expected)" in README
    assert "16.94 of 17" in prereg["m3"]
    assert f"**M5**, the run after it: {len(holds5)} verdicts on what M4 left open" in FLAT
    assert {v["word"] for v in RECORDS["m5"]["reading"]["verdicts"]} == {"holds"}
    assert f"All {len(holds5)} held; the pre-registration expected 4.997." in FLAT
    assert "4.997 of 5" in prereg["m5"]

    # A rival that failed on most worlds leaves the claim: family 19's Meridian.
    claimed = [c for c in m4["comparisons"] if c["claimed"] and c["mean"] is not None]
    beside = [c for c in m4["comparisons"] if not c["claimed"]]
    tight = m4["secondary"]["at the margin 0.01"]
    assert f"over {m4['holm']['over']} comparisons finds five losses" in README
    assert len(m4["holm"]["losses"]) == 5
    assert m4["holm"]["over"] == len(claimed) + len(beside) == 148
    assert f"the {len(claimed)} claimed and these {len(beside)}" in README
    assert f"{tight['within']} of M4's {tight['of']} claimed comparisons are within" in README
    assert tight["of"] == len(claimed)
    # Holm's step-down over the claimed comparisons alone, from the levels the reading prints.
    levels = [float(c["loss level"]) for c in claimed if c["loss level"] is not None]
    kept = {(h["family"], h["k"]) for h in m4["holm"]["losses"]} & {
        (c["family"], c["k"]) for c in claimed
    }
    assert _holm(levels, 0.025) == len(kept) == 3
    assert "the 128\n  claimed comparisons, Holm's losses are the same three" in README

    f4 = _found("m3")
    assert [f"{float(f4[4, 0, r]['mean']):+.3f}" for r in ("pymc-marketing", "orbit")] == [
        "+0.184",
        "+0.171",
    ]
    closest = max(_holds("m4"), key=lambda fk: float(_verdicts("m4")[fk]["level"]))
    assert closest == (6, 4)
    assert f"at level {_verdicts('m4')[closest]['level']}" in README
    assert f"(level\n  {_found('m4')[1, 4, 'pymc-marketing']['loss level']}, unadjusted)" in README


def test_m5_s_losses_the_text_states_are_its_reading() -> None:
    """M5's Holm's losses are all on family 15 and within δ, and they are the comparisons its
    secondary reading at 0.01 does not find within."""
    m5 = RECORDS["m5"]["reading"]
    claimed = [c for c in m5["comparisons"] if c["claimed"] and c["mean"] is not None]
    assert m5["holm"]["over"] == len(claimed) == len(m5["comparisons"]) == 25
    losses = {(h["family"], h["k"], h["rival"]) for h in m5["holm"]["losses"]}
    assert len(losses) == 4
    assert {f for f, _, _ in losses} == {15}
    assert f"over M5's {m5['holm']['over']} comparisons finds four losses" in FLAT
    levels = [float(c["loss level"]) for c in claimed if c["loss level"] is not None]
    assert _holm(levels, 0.025) == len(losses)
    found = _found("m5")
    assert all(found[fkr]["within"] and found[fkr]["word"] == "loses to" for fkr in losses)
    tight = m5["secondary"]["at the margin 0.01"]
    assert {(n["family"], n["k"], n["rival"]) for n in tight["not within"]} == losses
    assert (tight["of"], tight["of"] - tight["within"]) == (len(claimed), len(losses))
    assert f"the only ones of M5's {tight['of']} claimed comparisons not within 0.01" in FLAT
    assert m5["headline"].startswith("the headline holds")
    assert "prints its headline as holding over its own five verdicts only" in FLAT


def test_family_4_at_k_0_is_the_one_national_family_rung_in_no_claim() -> None:
    """M4 and M5 together read 32 national family-rungs; every one but family 4 at k = 0 is a
    verdict that holds, and that one is not within δ of PyMC-Marketing or Orbit."""
    national = {fk for run in ("m4", "m5") for fk in _verdicts(run)}
    held = set(_holds("m4")) | set(_holds("m5"))
    assert national - held == {(4, 0)}
    assert len(national) == 32
    assert f"Of the {len(national)} national family-rungs, one is in no claim" in FLAT
    assert (4, 0) not in _verdicts("m5")
    f4 = _found("m4")
    assert [f4[4, 0, r]["within"] for r in ("pymc-marketing", "orbit")] == [False, False]
    search = "PyMC-Marketing by +0.046 and Orbit by +0.054"
    prereg = readme_record.RUNS["m5"].preregistration.read_text(encoding="utf-8")
    assert search in " ".join(prereg.split())
    assert search in " ".join(section(README, "What has been measured").split())


def test_the_failures_the_text_states_are_the_readings() -> None:
    m4, m3 = _found("m4"), _found("m3")
    assert {m4[19, k, "meridian"]["rival_failed"] for k in (0, 4)} == {282}
    assert "Meridian failed on every world of family 19, 282 at each rung" in README
    orbit = sum(m4[18, k, "orbit"]["rival_failed"] for k in (0, 4))
    assert f"Orbit failed on {orbit} world-rungs of family 18" in README
    geo = sum(m3[f, k, "pymc-marketing"]["rival_failed"] for f, k in _holds("m3", geo=True))
    assert f"PyMC-Marketing failed on {geo} world-rungs of the geo families" in FLAT
    assert all(c["default_failed"] == 0 for c in m4.values())
    m5 = _found("m5")
    pymc = [m5[15, k, "pymc-marketing"]["rival_failed"] for k in (0, 4)]
    assert sum(c["rival_failed"] for c in m5.values()) == sum(pymc)
    assert (
        f"and in M5 on {sum(pymc)} world-rungs of family 15, {pymc[0]} at k = 0 and {pymc[1]} at "
        "the tested rung" in FLAT
    )
    assert all(c["default_failed"] == 0 for c in m5.values())
    assert "No claim arm failed in M4 or M5." in FLAT


def test_the_run_s_box_and_pins_the_text_states_are_the_record_s() -> None:
    record = RECORDS["m4"]
    box = record["compute"]["box"]
    assert (box["launches"], box["launched"][0]) == (3, "2026-11-21T06:27:48+00:00")
    assert "launched three times" in README
    assert "first launched, at 06:27:48 UTC on\n  2026-11-21" in README
    commits = record["commits"]
    stated = f"causal-hybrid-control {commits['causal-hybrid-control']} and jax {commits['jax']}"
    assert f"The run locked {stated}." in README
    assert commits["bench"] in README
    assert commits["mixdyn"][:7] in section(README, "Pre-registration and reproduction")

    record = RECORDS["m5"]
    box = record["compute"]["box"]
    assert (box["launches"], box["launched"]) == (1, ["2026-11-23T17:05:36+00:00"])
    assert "was launched, at 17:05:36 UTC the same day" in FLAT
    commits = record["commits"]
    code = section(README, "Pre-registration and reproduction").split("**M5's code.**", 1)[1]
    code = code.split("\n- ", 1)[0]
    stated = f"causal-hybrid-control {commits['causal-hybrid-control']} and jax {commits['jax']}"
    assert f"The run locked {stated}." in code
    assert f"The run ran mixdyn `{commits['mixdyn'][:7]}`" in code
    assert commits["bench"] in README


def test_the_pre_registrations_the_readme_names_are_the_files_the_runs_declared() -> None:
    for run, scored in readme_record.RUNS.items():
        digest = hashlib.sha256(scored.preregistration.read_bytes()).hexdigest()
        assert RECORDS[run]["commits"]["preregistration"] == digest
        assert f"`{digest}`" in section(README, "Pre-registration and reproduction")


def test_every_interval_typed_beside_the_tables_is_a_reading_s() -> None:
    """An interval the prose states is a comparison of one of the runs' readings, as printed."""
    prose = readme_record.BLOCK.sub("", README)
    typed = re.findall(
        r"([+\u2212]\d\.\d{4}) \[([+\u2212]\d\.\d{4}), ([+\u2212]\d\.\d{4})\]", prose
    )
    printed = {
        tuple(float(c[k]) for k in ("mean", "low", "high"))
        for record in RECORDS.values()
        for c in record["reading"]["comparisons"]
        if c["mean"] is not None
    }
    assert typed
    assert [t for t in typed if tuple(map(_number, t)) not in printed] == []
