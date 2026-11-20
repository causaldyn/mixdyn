"""`SKILL.md` is an Agent Skill whose commands run: its front matter names it and says when to use
it, and its Python blocks, run in order in one namespace, read, fit, check, plan and report on the
stand-in panel it builds, its health-card gate passing there.

The fit's seed is fixed, and the card passes with room: on 2026-11-06 the largest R-hat read 1.012
against 1.2, the predictive p-value 0.99 against 0.05 and no draw's baseline went negative against
0.2, so another machine's last bits do not move a flag."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKILL = (ROOT / "SKILL.md").read_text(encoding="utf-8")
STEPS = ("Read the panel", "Fit", "Diagnostics", "Tests", "Plan", "Report")


def front_matter() -> dict[str, str]:
    match = re.match(r"---\n(.*?)\n---\n", SKILL, flags=re.DOTALL)
    assert match is not None
    return dict(line.split(": ", 1) for line in match.group(1).splitlines())


def blocks() -> list[str]:
    return re.findall(r"^```python\n(.*?)^```$", SKILL, flags=re.DOTALL | re.MULTILINE)


def test_the_front_matter_names_the_skill_and_when_to_use_it() -> None:
    meta = front_matter()
    assert set(meta) == {"name", "description"}
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", meta["name"])
    assert len(meta["description"]) <= 1024
    assert "Use when" in meta["description"]


def test_the_steps_come_in_order_each_with_its_gate() -> None:
    headings = re.findall(r"^## \d\. (.+)$", SKILL, flags=re.MULTILINE)
    assert [h.split(":")[0] for h in headings] == list(STEPS)
    assert SKILL.count("**Gate") >= len(STEPS)


def test_its_blocks_run_in_order_on_the_stand_in() -> None:
    code = blocks()
    assert len(code) == len(STEPS)
    namespace: dict[str, object] = {"__name__": "skill"}
    for block in code:
        exec(compile(block, "SKILL.md", "exec"), namespace)
    report = namespace["report"]
    assert isinstance(report, dict)
    assert report["approved"] is False
    assert all(report["health card"].values())
