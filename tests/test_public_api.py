"""The public surface's shape: no public callable or class takes, returns or holds a class of the
engine that no `__all__` lists, so a caller never reaches past the public names to build an
argument or to name a field's type (ADR 0062).

The annotations are read from the source's syntax tree, each name followed to the class it means
where the module that wrote it imports it, under `TYPE_CHECKING` too. A function's parameters and
return are read; a class's public fields and its public methods' parameters and returns.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re
from collections import defaultdict
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import pytest

from test_public_names import public

if TYPE_CHECKING:
    from collections.abc import Iterable


@cache
def _tree(module: str) -> ast.Module:
    path = Path(str(importlib.import_module(module).__file__))
    return ast.parse(path.read_text(encoding="utf-8"))


def _meant(module: str, name: str) -> object | None:
    """What `name` means where `module` writes it in an annotation: its own, or what it imports
    under that name from within the engine."""
    found = vars(importlib.import_module(module)).get(name)
    if found is not None:
        return found
    for node in ast.walk(_tree(module)):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("mixdyn"):
            for alias in node.names:
                if (alias.asname or alias.name) == name:
                    return getattr(importlib.import_module(node.module), alias.name, None)
    return None


def _names(annotation: ast.expr | None) -> set[str]:
    """The bare names an annotation is built of, a string annotation parsed first."""
    if annotation is None:
        return set()
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        annotation = ast.parse(annotation.value, mode="eval").body
    return {node.id for node in ast.walk(annotation) if isinstance(node, ast.Name)}


def _annotations(node: ast.FunctionDef, *, method: bool) -> list[ast.expr | None]:
    args = node.args
    taken = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    return [a.annotation for a in taken[1 if method else 0 :]] + [node.returns]


def _read(value: object) -> list[ast.expr | None]:
    """Every annotation of a public function or class of the engine; none for anything else."""
    if not (inspect.isclass(value) or inspect.isfunction(value)):
        return []
    module = getattr(value, "__module__", "")
    if not module.startswith("mixdyn"):
        return []
    for node in _tree(module).body:
        if isinstance(node, ast.FunctionDef) and node.name == value.__name__:
            return _annotations(node, method=False)
        if isinstance(node, ast.ClassDef) and node.name == value.__name__:
            found: list[ast.expr | None] = []
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    if not item.target.id.startswith("_"):
                        found.append(item.annotation)
                elif isinstance(item, ast.FunctionDef) and not item.name.startswith("_"):
                    static = any(
                        isinstance(d, ast.Name) and d.id == "staticmethod"
                        for d in item.decorator_list
                    )
                    found += _annotations(item, method=not static)
            return found
    return []


def leaks(listed: Iterable[tuple[ModuleType, str]]) -> dict[str, set[str]]:
    """Each of the names `listed`, each with the module that lists it, that takes, returns or holds
    a class of the engine none of them is, by `module:name`, with those classes."""
    values = [(m, name, getattr(m, name)) for m, name in listed]
    shown = {(v.__module__, v.__qualname__) for _, _, v in values if inspect.isclass(v)}
    found: dict[str, set[str]] = defaultdict(set)
    for m, name, value in values:
        module = getattr(value, "__module__", "")
        for annotation in _read(value):
            for ident in _names(annotation):
                meant = _meant(module, ident)
                if (
                    inspect.isclass(meant)
                    and meant.__module__.startswith("mixdyn")
                    and (meant.__module__, meant.__qualname__) not in shown
                ):
                    found[f"{m.__name__}:{name}"].add(f"{meant.__module__}:{meant.__qualname__}")
    return dict(found)


def test_no_public_name_takes_returns_or_holds_a_class_no_all_lists() -> None:
    assert leaks(public()) == {}


def test_the_check_finds_a_class_that_no_all_lists() -> None:
    """`anchor`, a part of a planning rule at a module path, returns `Anchored`, which no `__all__`
    lists: read as public, it leaks that class, and with the class beside it, nothing."""
    anchors = importlib.import_module("mixdyn.calibration.anchors")
    planner = [*public(), (anchors, "anchor")]
    assert leaks(planner) == {
        "mixdyn.calibration.anchors:anchor": {"mixdyn.calibration.anchors:Anchored"}
    }
    assert leaks([*planner, (anchors, "Anchored")]) == {}


@pytest.mark.parametrize(
    ("old", "new"), [("stacked", "stacked_readings"), ("lift_row_pick", "pick_by_lift_tests")]
)
def test_a_renamed_name_imports_until_the_release_it_leaves_in_and_warns(
    old: str, new: str
) -> None:
    """A renamed public name still imports for one release, and using it warns with its
    replacement and the release it leaves in (ADR 0062)."""
    decisions = importlib.import_module("mixdyn.decisions")
    said = re.escape(
        f"mixdyn.decisions.{old} is renamed mixdyn.decisions.{new}; the old name leaves in 1.0.0"
    )
    with pytest.warns(DeprecationWarning, match=said):
        found = getattr(decisions, old)
    assert found is getattr(decisions, new)
    assert old not in decisions.__all__
    with pytest.raises(AttributeError, match="no attribute 'stacking'"):
        decisions.stacking  # noqa: B018
