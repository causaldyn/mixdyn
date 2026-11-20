"""The README states what the engine was and was not validated on, and points to the pages a user
reads first; its example runs; the scope page states what the engine does not model."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


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
