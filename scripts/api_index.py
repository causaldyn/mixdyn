"""Write `docs/api/index.md`, the public surface ADR 0062 promises, from the engine itself.

A name is public where a package's `__all__` lists it (ADR 0062); the page lists each, package by
package, with the first sentence of its own documentation: a class's or a function's docstring,
or, for a constant or a type alias, the string right after it in the module that defines it. A
name without one is listed bare. The rules the page states beside the list are ADR 0062's and ADR
0063's. `tests/test_public_names.py` fails when the page is not what this script writes.

    python scripts/api_index.py            # rewrite the page
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import re
import textwrap
from pathlib import Path
from types import ModuleType

import mixdyn

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "docs" / "api" / "index.md"
SENTENCE_END = re.compile(r"(?<=\.)\s+(?=[A-Z`(])")

HEAD = """# The public API

This page is written by `scripts/api_index.py` from each package's `__all__` and each name's own
documentation, and `tests/test_public_names.py` fails when the engine and the page disagree. Each
name is listed by the packages that list it, with the first sentence of its docstring; its fields,
methods and the rest are in its docstring. Decisions the names rest on are in `docs/adr/`.

## What is public

- **A name is public when an `__all__` of a package lists it**, and so are the fields, methods
  and properties of the classes these name that do not start with an underscore (ADR 0062). A
  name reachable only by its module path is not public, however much it is used.
- **Every public name imports and is documented**: by a docstring of its own, the string after a
  constant or an alias in the module that defines it, or a page of `docs/` that cites it. This
  page cites every name and so documents none: `tests/test_public_names.py` lists the names not
  documented yet, and that list only shrinks.

## Versions and deprecation

- **Before 1.0**, a minor release may change the public surface, and only as the removal rule
  below allows; a patch release never removes or changes a public name. **From 1.0**, a name
  leaves only in a major release.
- **How a public name is removed:** for one minor release it still imports, and using it raises a
  `DeprecationWarning` that names its replacement and the release it leaves in; it leaves in the
  minor release after, and the changelog names it under *Removed*. causal-hybrid-control removes
  its names the same way.
- A renamed class that `save` writes stays in the format's registry under its old key for as long
  as its old name imports, so files that name it still load.

## The file format and the documents

- **`save` and `load`** (`mixdyn.api.serialize`) write and read one format, a zip of a JSON
  manifest and NumPy arrays; nothing is pickled, and a loaded object equals the saved one to the
  bit (ADR 0062).
- **`FORMAT_VERSION` is 1.** It moves when a file the current version wrote would load
  differently or not at all: a registered class or a field renamed, moved or removed, a field
  added without a default, or a field's meaning or units changed. Adding a class to the registry,
  or a field with a default, does not move it. `load` reads its own version only and refuses any
  other, naming both. The release that moves the version reads the previous one too, through a
  converter, for at least one minor release.
- **The JSON documents** (`dumps`, `loads` and their schemas) are for callers outside Python. Each
  names its schema and an integer version, and reading refuses a document of another version.
  ADR 0063 states when a schema's version moves.
"""


def modules() -> list[ModuleType]:
    """The package and every module under it that has an `__all__`, by name."""
    names = sorted(m.name for m in pkgutil.walk_packages(mixdyn.__path__, "mixdyn."))
    found = [mixdyn, *(importlib.import_module(name) for name in names)]
    return [m for m in found if hasattr(m, "__all__")]


def _documentation(module: ModuleType, name: str) -> str | None:
    """`name`'s docstring, or the string after it, in the module of the engine that defines it;
    each `from mixdyn... import` followed there. A name from outside the engine: its own."""
    body = ast.parse(Path(str(module.__file__)).read_text(encoding="utf-8")).body
    for i, node in enumerate(body):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("mixdyn"):
            for alias in node.names:
                if (alias.asname or alias.name) == name:
                    return _documentation(importlib.import_module(node.module), alias.name)
        if isinstance(node, ast.ClassDef | ast.FunctionDef) and node.name == name:
            return ast.get_docstring(node)
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, ast.AnnAssign)
            else [node.name]
            if isinstance(node, ast.TypeAlias)
            else []
        )
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            after = body[i + 1] if i + 1 < len(body) else None
            if (
                isinstance(after, ast.Expr)
                and isinstance(after.value, ast.Constant)
                and isinstance(after.value.value, str)
            ):
                return after.value.value
            return None
    value = getattr(module, name)
    return vars(value).get("__doc__") if isinstance(value, type) else value.__doc__


def summary(module: ModuleType, name: str) -> str:
    """The first sentence of `name`'s documentation, on one line; empty where it has none."""
    text = _documentation(module, name)
    if not text:
        return ""
    paragraph = " ".join(inspect.cleandoc(text).split("\n\n")[0].split())
    first = SENTENCE_END.split(paragraph, maxsplit=1)[0]
    return first[:-1] + "." if first.endswith(":") else first


def render() -> str:
    """The page, as the engine's `__all__`s and docstrings make it now."""
    lines = [HEAD, "## The names, by package", ""]
    for module in modules():
        lines += [f"### `{module.__name__}`", ""]
        for name in sorted(module.__all__, key=str.casefold):
            text = summary(module, name)
            entry = f"- `{name}`" + (f": {text}" if text else "")
            lines += textwrap.wrap(
                entry, 100, subsequent_indent="  ", break_long_words=False, break_on_hyphens=False
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    INDEX.parent.mkdir(exist_ok=True)
    INDEX.write_text(render(), encoding="utf-8")


if __name__ == "__main__":
    main()
