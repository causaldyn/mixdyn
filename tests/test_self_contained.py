"""Nothing the tree publishes points into the research repository the engine was measured in.

The engine, its pages, its tests and its scripts are to be read without that repository, which is
not public: a reader who follows a path into it finds nothing. A pointer that carries a result
names the public file that holds it instead, or states the result.

The harnesses some ADRs cite, and the script one fixture names, live in `evidence/`, which the
scan reads too. `PENDING` lists the lines still waiting on a public home, none since 2026-11-06. A
line outside it fails, and so does an entry whose line is gone, so the list only shrinks.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCANNED = ("src", "docs", "tests", "scripts", "evidence", "README.md", "SKILL.md")
PRIVATE_POINTER = re.compile(r"mixdyn-parity|plans/[0-9]|discoveries/|(?i:research repository)")
# A fixture's JSON is one line, so an entry keeps the start of its line only.
WIDTH = 80
PENDING: frozenset[str] = frozenset()


def _files() -> list[Path]:
    """Every scanned file but this one, whose pattern and `PENDING` spell what it looks for."""
    found: list[Path] = []
    for name in SCANNED:
        path = ROOT / name
        found += [path] if path.is_file() else sorted(path.rglob("*"))
    here = Path(__file__).resolve()
    return [p for p in found if p.is_file() and "__pycache__" not in p.parts and p != here]


def _pointers() -> set[str]:
    """``path:text`` of every line that names a private path or the research repository."""
    return {
        f"{path.relative_to(ROOT)}:{line.strip()[:WIDTH]}"
        for path in _files()
        for line in path.read_text(encoding="utf-8").splitlines()
        if PRIVATE_POINTER.search(line)
    }


def test_no_line_points_into_the_research_repository_but_the_pending_ones() -> None:
    assert sorted(_pointers() - PENDING) == []


def test_every_pending_line_still_points_there() -> None:
    assert sorted(PENDING - _pointers()) == []


def test_the_pattern_tells_a_private_pointer_from_prose() -> None:
    private = (
        "the harness `mixdyn-parity/bench/run.py` records the run",
        "as plans/9 states",
        "see discoveries/notes.md",
        "measured in the Research Repository's pilot",
    )
    public = (
        "the plan's table reads every return",
        "plans/ holds nothing here",
        "the discoveries of the pilot",
        "the repository's own fixtures",
        "a mixdyn parity test",
    )
    assert [text for text in private if not PRIVATE_POINTER.search(text)] == []
    assert [text for text in public if PRIVATE_POINTER.search(text)] == []
