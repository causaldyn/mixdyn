"""The README states what the engine was and was not validated on, and points to the pages a user
reads first; its example runs; the scope page states what the engine does not model.

Its record of the run its claim was read on states the conflict of interest, the error bars and the
engine's place, and its tables of worlds per cell, tuned rivals, versions and compute are the ones
`scripts/readme_record.py` writes from the run's records (`evidence/results/m3_record.json`): a
table typed into one of those blocks fails here. The claim's own table is typed; each of its cells
is checked against the run's reading."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

import readme_record

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
RECORD = json.loads(readme_record.RECORD.read_text(encoding="utf-8"))


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
    """Run `python scripts/readme_record.py` when the record or the script changes."""
    assert readme_record.blocks(README) == readme_record.render(RECORD)


def test_each_record_table_sits_in_the_section_that_names_it() -> None:
    placed = {
        "cells": "What has been measured",
        "tuned": "The tuned rivals",
        "versions": "Versions of every arm",
        "compute": "Compute",
    }
    assert sorted(placed) == sorted(readme_record.RENDERERS)
    assert [b for b, h in placed.items() if f"<!-- record:{b} -->" not in section(README, h)] == []


def test_the_readme_states_its_conflict_of_interest_its_error_bars_and_its_place() -> None:
    assert "the bench that scores it" in section(README, "Conflict of interest")
    measured = section(README, "What has been measured")
    assert "**The error bars.**" in measured
    assert "The variability" in measured
    assert "**Where it sits.**" in README
    assert "multi-touch attribution (MTA)" in README


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


def test_the_record_names_no_local_path() -> None:
    local = re.compile(r"^[/~]|/home/|\.cache|\.internal")
    assert [s for s in _strings(RECORD) if local.search(s)] == []


def test_the_core_hours_the_text_states_are_the_record_s() -> None:
    hours = sum(
        f["core_hours"]["engine"] + f["core_hours"]["rivals"]
        for f in RECORD["compute"]["families"].values()
    )
    stated = f"{round(hours):,}".replace(",", " ")
    assert f"The run took {stated} CPU-hours" in README


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
    assert (second["mean"], second["word"], second["rival_failed"], second["cap"]) == (
        None,
        "not compared",
        180,
        60,
    )
    assert read["verdicts"] == [{"family": 9, "k": 4, "word": "holds"}]


def _typed(cell: str) -> tuple[float, float, float]:
    numbers = re.findall(r"[+\u2212]\d+\.\d+", cell)
    mean, low, high = (float(n.replace("\u2212", "-")) for n in numbers)
    return mean, low, high


def test_the_claim_s_typed_table_reads_as_the_run_s_reading() -> None:
    """Each cell of the claim's table is the reading's comparison at three digits, and its worlds
    are the status quo's complete cases and the arm's failures: no cell drifts from the record."""
    found = {(c["family"], c["k"], c["rival"]): c for c in RECORD["reading"]["comparisons"]}
    rungs = {"0": 0, "4": 4, "1\u20132": 1}
    after = section(README, "What has been measured").split("| Family | Rung | Arm | Worlds |", 1)
    rows = after[1].split("\n\n", 1)[0].splitlines()[2:]
    off = []
    for row in rows:
        cells = [c.strip() for c in row.strip("|").split("|")]
        family, k = int(cells[0].split()[0]), rungs[cells[1]]
        quo = found[family, k, "status quo"]
        if int(cells[3]) != quo["complete"] + quo["default_failed"]:
            off.append((family, k, "worlds"))
        for rival, cell in zip(readme_record.RIVALS, cells[4:], strict=True):
            c = found[family, k, rival]
            read = (float(c["mean"]), float(c["low"]), float(c["high"]))
            if any(abs(t - r) > 0.0005 + 1e-12 for t, r in zip(_typed(cell), read, strict=True)):
                off.append((family, k, rival))
    assert len(rows) == 17
    assert off == []
