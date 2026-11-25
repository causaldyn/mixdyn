"""The public surface: every name a package's `__all__` lists imports and is documented, and every
name of the engine the pages cite imports.

A name is public where a package's `__all__` lists it (ADR 0062). It is documented where it carries
a docstring of its own: a class's or a function's, or, for a constant or a type alias, the string
right after it in the module that defines it, as the engine documents them; or where the pages
cite it. Every public name is documented. `docs/api/index.md` lists every public name, as
`scripts/api_index.py` writes it; it cites every name, so it documents none.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
import re
from functools import cache
from pathlib import Path
from types import ModuleType

import api_index
import mixdyn

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
INDEX = DOCS / "api" / "index.md"
CITED = re.compile(r"`(mixdyn(?:\.[A-Za-z_]\w*)+)")


@cache
def modules() -> tuple[ModuleType, ...]:
    """The package and every module under it."""
    names = [m.name for m in pkgutil.walk_packages(mixdyn.__path__, "mixdyn.")]
    return (mixdyn, *(importlib.import_module(name) for name in sorted(names)))


def public() -> list[tuple[ModuleType, str]]:
    """Every name an `__all__` lists, with the module that lists it."""
    return [(m, name) for m in modules() for name in getattr(m, "__all__", ())]


@cache
def _tree(module: ModuleType) -> ast.Module:
    return ast.parse(Path(str(module.__file__)).read_text(encoding="utf-8"))


def _source(module: ModuleType, name: str) -> tuple[ast.stmt | None, str | None]:
    """The statement that defines `name` as `module` imports it, and the string right after it,
    each `from ... import` within mixdyn followed to the module that defines it."""
    body = _tree(module).body
    for i, node in enumerate(body):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("mixdyn"):
            for alias in node.names:
                if (alias.asname or alias.name) == name:
                    return _source(importlib.import_module(node.module), alias.name)
        defined = (isinstance(node, ast.ClassDef | ast.FunctionDef) and node.name == name) or any(
            isinstance(target, ast.Name) and target.id == name
            for target in (
                node.targets
                if isinstance(node, ast.Assign)
                else [node.target]
                if isinstance(node, ast.AnnAssign)
                else [node.name]
                if isinstance(node, ast.TypeAlias)
                else []
            )
        )
        if defined:
            after = body[i + 1] if i + 1 < len(body) else None
            string = (
                after.value.value
                if isinstance(after, ast.Expr)
                and isinstance(after.value, ast.Constant)
                and isinstance(after.value.value, str)
                else None
            )
            return node, string
    return None, None


def documented(module: ModuleType, name: str) -> bool:
    """Whether `name`, as `module` exports it, has a docstring of its own or the pages cite it."""
    node, string = _source(module, name)
    if node is None:  # from outside the engine: its own docstring, as its package documents it
        value = getattr(module, name)
        return bool(vars(value).get("__doc__") if isinstance(value, type) else value.__doc__)
    if isinstance(node, ast.ClassDef | ast.FunctionDef):
        return ast.get_docstring(node) is not None
    return string is not None or f"`{name}`" in _prose()


@cache
def _pages() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(DOCS.rglob("*.md")))


@cache
def _prose() -> str:
    """The pages but the index, whose list of every public name documents none of them."""
    pages = sorted(p for p in DOCS.rglob("*.md") if p != INDEX)
    return "\n".join(p.read_text(encoding="utf-8") for p in pages)


def _resolves(dotted: str) -> bool:
    """Whether `dotted` names a module of the engine, or an attribute down from one."""
    parts = dotted.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            value: object = importlib.import_module(".".join(parts[:cut]))
        except ModuleNotFoundError:
            continue
        for part in parts[cut:]:
            if not hasattr(value, part):
                return False
            value = getattr(value, part)
        return True
    return False


def test_every_name_an_all_lists_imports() -> None:
    assert public()
    missing = [f"{m.__name__}:{name}" for m, name in public() if not hasattr(m, name)]
    assert missing == []
    for m in modules():
        if hasattr(m, "__all__"):
            namespace: dict[str, object] = {}
            exec(f"from {m.__name__} import *", namespace)
            assert set(m.__all__) <= set(namespace)


def test_no_name_is_listed_twice_or_private() -> None:
    for m in modules():
        names = list(getattr(m, "__all__", ()))
        assert len(set(names)) == len(names), m.__name__
        assert [n for n in names if n.startswith("_")] == [], m.__name__


def test_nothing_an_all_lists_is_undocumented() -> None:
    assert sorted(f"{m.__name__}:{name}" for m, name in public() if not documented(m, name)) == []


def test_every_name_of_the_engine_the_pages_cite_imports() -> None:
    cited = sorted(set(CITED.findall(_pages())))
    assert cited
    assert [name for name in cited if not _resolves(name)] == []


def test_the_api_index_lists_every_public_name() -> None:
    index = INDEX.read_text(encoding="utf-8")
    assert sorted({name for _, name in public() if f"`{name}`" not in index}) == []


def test_the_api_index_is_the_one_the_engine_writes() -> None:
    """Run `python scripts/api_index.py` when a public name or its docstring changes."""
    assert INDEX.read_text(encoding="utf-8") == api_index.render()


def test_the_rules_tell_a_cited_name_from_prose() -> None:
    assert _resolves("mixdyn.decisions.returns")
    assert _resolves("mixdyn.calibration.LiftEstimate")
    assert _resolves("mixdyn.models.dlm_media.search_media")
    assert not _resolves("mixdyn.decisions.no_such_plan")
    assert not _resolves("mixdyn.nowhere")
    assert CITED.findall("see `mixdyn.models.nested` and `mixdyn`") == ["mixdyn.models.nested"]
    assert documented(mixdyn, "NATIONAL")
    assert documented(mixdyn, "read_panel")
    assert documented(importlib.import_module("mixdyn.data"), "KpiUnit")
    assert not documented(importlib.import_module("mixdyn.data.panel"), "_CURRENCY")
