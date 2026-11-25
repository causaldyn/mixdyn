"""The public surface (ADR 0062): every name an `__all__` lists, held to a snapshot.

- **The snapshot**, `tests/public_api.txt`, holds one line a public object: a function's signature,
  a class's decorators and bases with a line for each public field, class attribute, enum member
  and method, a constant's value, and the package a name of another package comes from. A name
  being removed holds a line naming its replacement and the release it leaves in. The lines are
  read from the source's syntax tree, so they are the same on every Python the CI runs. Write it
  with `PYTHONPATH=scripts uv run python tests/test_public_api.py`, which refuses to drop a name
  until the release its package's `_DEPRECATED` says it leaves in.
- **The removal rule**: a renamed name leaves its package's `__all__` for its `_DEPRECATED`, which
  names its replacement and the release it leaves in; until then it imports with a
  `DeprecationWarning` saying so. From 1.0 it leaves only in a major release.
- **What the last release held stays** (ADR 0103). Each line of the snapshot at the newest release
  tag below the checkout is still in the surface, or it changes as the rules allow: a name leaves
  as the removal rule says, a function or a method may gain parameters with defaults and a field a
  default, a default or a value moves only in a minor release, and anything else changes only in a
  major release from 1.0. A dataclass, an attrs class or a `NamedTuple` is read for the `__init__`
  it writes from its fields too: a field without a default, or one that moves the fields passed by
  position, is a change of that kind. Read with git; skipped where no release tag is reachable, as
  in a shallow clone.
- **No public name shadows a submodule** of its package, but those `SHADOWING` lists.
- **No public callable or class takes, returns or holds a class of the engine that no `__all__`
  lists**, so a caller never reaches past the public names to build an argument or to name a
  field's type. Each name in an annotation is followed to the class it means where the module that
  wrote it imports it, under `TYPE_CHECKING` too, and a `type` alias to the names of its value, in
  the module that defines it. A function's parameters and return are read; a class's public fields
  and its public methods' parameters and returns; a public alias's value.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import re
import subprocess
import sys
import warnings
from collections import defaultdict
from functools import cache
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, TypeAliasType

import pytest

from test_public_names import _source, modules, public

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


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


def _value(alias: TypeAliasType) -> tuple[str, ast.expr | None]:
    """The module that defines `alias`, and what `alias` stands for as that module writes it."""
    module = str(alias.__module__)
    for node in _tree(module).body:
        if isinstance(node, ast.TypeAlias) and node.name.id == alias.__name__:
            return module, node.value
    return module, None


def _classes(module: str, annotation: ast.expr | None, seen: set[TypeAliasType]) -> Iterator[type]:
    """Each class of the engine `annotation` names where `module` writes it, each `type` alias it
    names followed to the classes of its value, each alias once."""
    for ident in _names(annotation):
        meant = _meant(module, ident)
        if isinstance(meant, TypeAliasType):
            if meant not in seen:
                seen.add(meant)
                yield from _classes(*_value(meant), seen)
        elif inspect.isclass(meant) and meant.__module__.startswith("mixdyn"):
            yield meant


def _read(value: object) -> list[ast.expr | None]:
    """Every annotation of a public function or class of the engine, and the value of a public
    alias of it; none for anything else."""
    module = getattr(value, "__module__", "")
    if not module.startswith("mixdyn"):
        return []
    if isinstance(value, TypeAliasType):
        return [_value(value)[1]]
    if not (inspect.isclass(value) or inspect.isfunction(value)):
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
        seen: set[TypeAliasType] = set()
        for annotation in _read(value):
            for meant in _classes(module, annotation, seen):
                if (meant.__module__, meant.__qualname__) not in shown:
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


def test_the_check_follows_a_type_alias_to_the_classes_it_names() -> None:
    """`fit_bayes` takes a `Start`, an alias of `UniformStart | PathfinderStart`: with those two
    out of the `__all__`s, the check finds them through the alias."""
    hidden = {"UniformStart", "PathfinderStart"}
    found = leaks([(m, name) for m, name in public() if name not in hidden])
    assert found["mixdyn.models:fit_bayes"] == {f"mixdyn.models.bayes_mmm:{n}" for n in hidden}


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


SNAPSHOT = Path(__file__).with_name("public_api.txt")
SHADOWING = frozenset(
    {"mixdyn.decisions.plan_table", "mixdyn.decisions.returns", "mixdyn.inference.loo"}
)
"""The public functions named as a submodule of their package is, from before the rule: a caller
who imports the submodule by its path does not reach the function by that path. It only shrinks."""


def _signature(node: ast.FunctionDef) -> str:
    params = node.type_params
    kinds = f"[{', '.join(map(ast.unparse, params))}]" if params else ""
    returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"{kinds}({ast.unparse(node.args)}){returns}"


def _members(node: ast.ClassDef) -> list[str]:
    """A line for each public field, class attribute, enum member and method of a class, and its
    `__init__` and `__call__`, in the order the class writes them."""
    found = []
    for item in node.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            if not item.target.id.startswith("_"):
                default = f" = {ast.unparse(item.value)}" if item.value is not None else ""
                found.append(f"  .{item.target.id}: {ast.unparse(item.annotation)}{default}")
        elif isinstance(item, ast.Assign):
            names = [t.id for t in item.targets if isinstance(t, ast.Name)]
            found += [f"  .{n} = {ast.unparse(item.value)}" for n in names if not n.startswith("_")]
        elif isinstance(item, ast.FunctionDef) and (
            not item.name.startswith("_") or item.name in ("__init__", "__call__")
        ):
            decorated = "".join(f"@{ast.unparse(d)} " for d in item.decorator_list)
            found.append(f"  .{item.name} {decorated}def{_signature(item)}")
    return found


def _lines(module: ModuleType, name: str) -> list[str]:
    """The snapshot's lines of `name` as `module` lists it."""
    head = f"{module.__name__}:{name}"
    node, _ = _source(module, name)
    if node is None:
        return [f"{head} from {getattr(getattr(module, name), '__module__', '?')}"]
    if isinstance(node, ast.FunctionDef):
        return [f"{head} def{_signature(node)}"]
    if isinstance(node, ast.ClassDef):
        decorated = "".join(f"@{ast.unparse(d)} " for d in node.decorator_list)
        bases = ", ".join(map(ast.unparse, [*node.bases, *node.keywords]))
        return [f"{head} {decorated}class({bases})", *_members(node)]
    if isinstance(node, ast.TypeAlias):
        return [f"{head} type = {ast.unparse(node.value)}"]
    if isinstance(node, ast.AnnAssign):
        value = f" = {ast.unparse(node.value)}" if node.value is not None else ""
        return [f"{head}: {ast.unparse(node.annotation)}{value}"]
    assert isinstance(node, ast.Assign)
    return [f"{head} = {ast.unparse(node.value)}"]


def deprecated() -> dict[str, tuple[ModuleType, str, str]]:
    """Each name leaving a package, by `module:name`, with its package, its replacement and the
    release it leaves in."""
    return {
        f"{m.__name__}:{old}": (m, new, leaves)
        for m in modules()
        for old, (new, leaves) in getattr(m, "_DEPRECATED", {}).items()
    }


def surface() -> list[str]:
    """The snapshot's lines: each public name's, and each leaving name's, by module and name."""
    entries = [((m.__name__, name), _lines(m, name)) for m, name in public()]
    entries += [
        ((m.__name__, key.split(":")[1]), [f"{key} renamed {new}, leaves in {leaves}"])
        for key, (m, new, leaves) in deprecated().items()
    ]
    return [line for _, lines in sorted(entries, key=lambda e: e[0]) for line in lines]


def _heads(lines: Iterable[str]) -> set[str]:
    """The `module:name` each entry of the snapshot's `lines` opens with."""
    return {line.split(" ", 1)[0].rstrip(":") for line in lines if not line.startswith(" ")}


def _release(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", text)[:3])


_LEAVING = re.compile(r"(?P<head>\S+) renamed \S+, leaves in (?P<leaves>\S+)")
_DEF = re.compile(r"(?:^| )def(?=[(\[])")


def _entries(lines: Iterable[str]) -> dict[str, dict[str, str]]:
    """The snapshot's `lines` by entry, its `module:name`, each entry's lines by key: its own line
    under its `module:name`, a member's under `.` and the member's name."""
    found: dict[str, dict[str, str]] = {}
    head = ""
    for line in lines:
        if line.startswith(" "):
            found[head][line.split()[0].rstrip(":")] = line
        else:
            head = line.split(" ", 1)[0].rstrip(":")
            found[head] = {head: line}
    return found


def _text(node: ast.expr | None) -> str | None:
    return None if node is None else ast.unparse(node)


def _parameters(node: ast.FunctionDef) -> list[tuple[str, str, str | None, str | None]]:
    """Each parameter of `node`: its kind, its name, its annotation and its default."""
    args = node.args
    positional = [*args.posonlyargs, *args.args]
    defaults = [None] * (len(positional) - len(args.defaults)) + [*args.defaults]
    found = []
    for i, (a, default) in enumerate(zip(positional, defaults, strict=True)):
        kind = "positional only" if i < len(args.posonlyargs) else "positional"
        found.append((kind, a.arg, _text(a.annotation), _text(default)))
    if args.vararg:
        found.append(("*", args.vararg.arg, _text(args.vararg.annotation), None))
    found += [
        ("keyword", a.arg, _text(a.annotation), _text(d))
        for a, d in zip(args.kwonlyargs, args.kw_defaults, strict=True)
    ]
    if args.kwarg:
        found.append(("**", args.kwarg.arg, _text(args.kwarg.annotation), None))
    return found


def _extends(old: ast.FunctionDef, new: ast.FunctionDef) -> str:
    """How the signature `new` changes `old`: "extended" where it only adds parameters with
    defaults, keyword-only or after the old positional ones, or gives an old one a default;
    "revalued" where an old default moves besides; "changed" otherwise."""
    if (_text(old.returns), [*map(ast.unparse, old.type_params)]) != (
        _text(new.returns),
        [*map(ast.unparse, new.type_params)],
    ):
        return "changed"
    before, after = _parameters(old), _parameters(new)
    ordered = [p[:3] for p in before if p[0].startswith("positional")]
    if [p[:3] for p in after if p[0].startswith("positional")][: len(ordered)] != ordered:
        return "changed"
    kept = {p[1]: p for p in after}
    moved = False
    for kind, name, annotation, default in before:
        now = kept.pop(name, None)
        if now is None or now[:3] != (kind, name, annotation):
            return "changed"
        if default is not None and now[3] != default:
            if now[3] is None:
                return "changed"
            moved = True
    if any(default is None and kind not in ("*", "**") for kind, _, _, default in kept.values()):
        return "changed"
    return "revalued" if moved else "extended"


def _signed(line: str, found: re.Match[str]) -> ast.FunctionDef:
    """The function a snapshot's `def` line writes, `found` its `def`."""
    node = ast.parse(f"def _{line[found.end() :]}: ...").body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


_BUILDERS = frozenset({"dataclass", "define", "frozen", "mutable", "s"})
"""The decorators that write a class's `__init__` from its fields: `dataclass`, and attrs'
`define`, `frozen`, `mutable` and `attr.s`."""
_FIELDS = frozenset({"field", "ib", "attrib"})
_DEFAULTS = frozenset({"default", "default_factory", "factory"})


def _last(node: ast.expr) -> str:
    """The last name of `node`, a name or a dotted one; empty for anything else."""
    return (
        node.id
        if isinstance(node, ast.Name)
        else node.attr
        if isinstance(node, ast.Attribute)
        else ""
    )


def _keyword(call: ast.Call | None, name: str) -> object:
    """The constant `call` passes as `name`; None where it passes none."""
    for k in call.keywords if call else ():
        if k.arg == name and isinstance(k.value, ast.Constant):
            return k.value.value
    return None


def _init(lines: dict[str, str]) -> ast.FunctionDef | None:
    """The `__init__` that a dataclass, an attrs class or a `NamedTuple` writes from the fields of
    the snapshot's entry `lines`, a parameter for each field it takes, in order, each with whether
    it has a default, its annotation and its value left out; None for a class that writes none."""
    head = next(iter(lines.values()))
    decorated, written, bases = head.split(" ", 1)[-1].rpartition("class(")
    if not written or ".__init__" in lines:
        return None
    decorators = [f"@{d}" for d in decorated.strip().removeprefix("@").split(" @") if d]
    node = ast.parse("\n".join([*decorators, f"class _({bases}: ..."])).body[0]
    assert isinstance(node, ast.ClassDef)
    builds, keyword = any(_last(base) == "NamedTuple" for base in node.bases), False
    for decorator in node.decorator_list:
        call = decorator if isinstance(decorator, ast.Call) else None
        if _last(call.func if call else decorator) in _BUILDERS:
            if _keyword(call, "init") is False:
                return None
            builds, keyword = True, _keyword(call, "kw_only") is True
    if not builds:
        return None
    positional: list[str] = []
    named: list[str] = []
    for key, line in lines.items():
        if not (key.startswith(".") and line.split()[0].endswith(":")):
            continue
        field = ast.parse(line.strip().removeprefix(".")).body[0]
        assert isinstance(field, ast.AnnAssign)
        assert isinstance(field.target, ast.Name)
        kind = field.annotation
        if _last(kind.value if isinstance(kind, ast.Subscript) else kind) == "ClassVar":
            continue
        value = field.value
        call = value if isinstance(value, ast.Call) and _last(value.func) in _FIELDS else None
        if _keyword(call, "init") is False:
            continue
        if call is None:
            given = value is not None
        else:
            given = bool(call.args) or any(k.arg in _DEFAULTS for k in call.keywords)
        # A field without a default can follow one with a default only past a `KW_ONLY` marker,
        # which is private and so not in the snapshot: it and every field after it are keywords.
        keyword = keyword or (not given and any(p.endswith("=...") for p in positional))
        kept = named if keyword or _keyword(call, "kw_only") is True else positional
        kept.append(f"{field.target.id}=..." if given else field.target.id)
    built = ast.parse(f"def _({', '.join([*positional, *(['*', *named] if named else [])])}): ...")
    function = built.body[0]
    assert isinstance(function, ast.FunctionDef)
    return function


def _change(old: str, new: str) -> str:
    """How the snapshot's line `new` changes `old`, the line of the same entry and key it holds
    now: as `_extends` says for a function or a method; "revalued" where only a default or a value
    moves, "extended" where a field gains a default; "changed" otherwise."""
    defs = _DEF.search(old), _DEF.search(new)
    if defs[0] and defs[1]:
        if old[: defs[0].start()] != new[: defs[1].start()]:
            return "changed"
        return _extends(_signed(old, defs[0]), _signed(new, defs[1]))
    (before, _, was), (after, _, now) = old.partition(" = "), new.partition(" = ")
    if any(defs) or before != after or before.endswith(" type"):
        return "changed"
    if was and now:
        return "revalued"
    return "extended" if now else "changed"


def breaks(
    released: Iterable[str], was: tuple[int, ...], current: Iterable[str], now: tuple[int, ...]
) -> list[str]:
    """Each line of `released`, the snapshot of release `was`, that `current`, the surface at
    version `now`, drops or changes past what the rules allow (ADR 0062, ADR 0103).

    A name leaves only where `released` held it leaving, at the release it named, and from 1.0 in
    a major release. A function or a method may gain parameters with defaults in any release. A
    default or a value moves only in a minor release. Anything else that a still-public name drops
    or changes, a parameter, a type, a field, a member, a base, moves only in a major release from
    1.0; before 1.0 a name whose signature must change takes a new name, the old leaving."""
    minor, major = now[:2] > was[:2], was >= (1,) and now[0] > was[0]
    old, new = _entries(released), _entries(current)
    found = []
    for head, lines in old.items():
        leaving = _LEAVING.fullmatch(lines[head])
        if head not in new:
            if leaving is None:
                found.append(f"{head} left with no release in _DEPRECATED")
            elif now < _release(leaving["leaves"]):
                found.append(f"{head} left before {leaving['leaves']}")
            elif now >= (1,) and now[0] == was[0]:
                found.append(f"{head} left in a minor release")
            continue
        if leaving or _LEAVING.fullmatch(new[head][head]):
            continue
        reported = len(found)
        for key, line in lines.items():
            held = new[head].get(key)
            if held == line:
                continue
            change = "changed" if held is None else _change(line, held)
            if not (change == "extended" or (change == "revalued" and minor) or major):
                found.append(f"{line.strip()} -> {held}: {change}")
        was_init, now_init = _init(lines), _init(new[head])
        if (
            len(found) == reported
            and was_init
            and now_init
            and not major
            and _extends(was_init, now_init) == "changed"
        ):
            built = f"__init__({ast.unparse(was_init.args)}) -> ({ast.unparse(now_init.args)})"
            found.append(f"{head} {built}: changed")
    return found


def _last_release() -> tuple[tuple[int, ...], list[str]] | None:
    """The newest release tag the checkout's commit contains, a tag on that commit excepted, by
    version, and the snapshot it holds; None where git, such a tag or its snapshot is not there."""
    root = SNAPSHOT.parent.parent

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        command = ["git", "-C", str(root), *args]
        return subprocess.run(command, capture_output=True, text=True, check=False)

    try:
        here = set(git("tag", "--points-at", "HEAD").stdout.split())
        tags = git("tag", "--merged", "HEAD", "--list", "v[0-9]*").stdout.split()
        released = max(((_release(tag), tag) for tag in tags if tag not in here), default=None)
        if released is None:
            return None
        shown = git("show", f"{released[1]}:{SNAPSHOT.relative_to(root).as_posix()}")
    except FileNotFoundError:
        return None
    return (released[0], shown.stdout.splitlines()) if shown.returncode == 0 else None


def test_the_surface_is_the_snapshot() -> None:
    """Run `PYTHONPATH=scripts uv run python tests/test_public_api.py` when a public name or its
    signature changes."""
    assert surface() == SNAPSHOT.read_text(encoding="utf-8").splitlines()


def test_a_name_leaving_its_package_still_imports_and_warns_until_its_release() -> None:
    version = _release(metadata.version("mixdyn"))
    for key, (module, new, leaves) in deprecated().items():
        old = key.split(":")[1]
        assert old not in getattr(module, "__all__", ()), key
        assert new in getattr(module, "__all__", ()), key
        assert version < _release(leaves), f"{key} leaves in {leaves}: remove it"
        if version >= (1,):
            assert _release(leaves)[0] > version[0], f"{key}: from 1.0 a name leaves in a major"
        name = module.__name__
        said = f"{name}.{old} is renamed {name}.{new}; the old name leaves in {leaves}"
        with pytest.warns(DeprecationWarning, match=re.escape(said)):
            found = getattr(module, old)
        assert found is getattr(module, new), key


def test_each_line_the_last_release_held_stays_or_changes_as_the_rules_allow() -> None:
    released = _last_release()
    if released is None:
        pytest.skip("no release tag with a snapshot is reachable: in a shallow clone, fetch tags")
    was, lines = released
    assert breaks(lines, was, surface(), _release(metadata.version("mixdyn"))) == []


F = "m:f def(a: int, /, b: float = 1.0, *, c: str = 'c') -> float"
RELEASED = [
    "m:C @dataclass(frozen=True, slots=True) class()",
    "  .x: float",
    "  .y: float = 0.5",
    "  .scaled def(self, k: float) -> float",
    "m:K = 1",
    F,
    "m:old renamed new, leaves in 1.0.0",
]
"""A snapshot a release 1.2.0 held, for the rules' tests."""


@pytest.mark.parametrize(
    ("old", "new", "first", "change"),
    [
        (F, F.replace("*, c", "d: int = 0, *, c"), "", ""),
        (F, F.replace("'c')", "'c', e: int = 0)"), "", ""),
        (F, F.replace("*, c", "*args: int, c"), "", ""),
        (F, F.replace("a: int,", "a: int = 0,"), "", ""),
        ("  .x: float", "  .x: float = 0.0", "", ""),
        (F, F.replace("1.0", "2.0"), "1.3.0", "revalued"),
        ("  .y: float = 0.5", "  .y: float = 0.9", "1.3.0", "revalued"),
        ("m:K = 1", "m:K = 2", "1.3.0", "revalued"),
        (F, F.replace("'c')", "'c', e: int)"), "2.0.0", "changed"),
        (F, F.replace("b: float", "d: int = 0, b: float"), "2.0.0", "changed"),
        (F, F.replace("c: str", "d: str"), "2.0.0", "changed"),
        (F, F.replace(" = 'c'", ""), "2.0.0", "changed"),
        (F, F.replace("b: float", "b: int"), "2.0.0", "changed"),
        (F, F.replace("-> float", "-> int"), "2.0.0", "changed"),
        (RELEASED[3], RELEASED[3].replace("def", "@property def"), "2.0.0", "changed"),
        ("  .y: float = 0.5", "  .y: int = 0.5", "2.0.0", "changed"),
        ("  .x: float", None, "2.0.0", "changed"),
        (RELEASED[0], RELEASED[0].replace(", slots=True", ""), "2.0.0", "changed"),
    ],
)
def test_a_released_line_changes_only_in_the_release_its_change_allows(
    old: str, new: str | None, first: str, change: str
) -> None:
    """A function or a method gains parameters with defaults, and a field a default, in any
    release; a default or a value moves in a minor release; anything else changes in a major from
    1.0. `first` is the first release after 1.2.0 the change may come in, "" for any."""
    current = [line for line in RELEASED if line != old]
    if new is not None:
        current.insert(RELEASED.index(old), new)
    before = {"": (1, 2, 1), "1.3.0": (1, 2, 1), "2.0.0": (1, 9, 0)}[first]
    found = breaks(RELEASED, (1, 2, 0), current, before)
    assert [f.rsplit(": ", 1)[1] for f in found] == ([change] if change else [])
    if first:
        assert breaks(RELEASED, (1, 2, 0), current, _release(first)) == []


D = [
    "m:D @dataclass(frozen=True) class()",
    "  .x: float",
    "  .y: float = 0.5",
    "  .z def(self) -> float",
]
"""A dataclass a release 1.2.0 held, its `__init__` `(x, y=0.5)`."""
WIDE = [*D[:3], "  .w: int = 0", D[3]]
KW = [D[0].replace("frozen=True", "kw_only=True"), *D[1:]]
MARKED = ["m:M @dataclass() class()", "  .x: float = 0.0", "  .y: float"]
"""A dataclass whose `y` follows a private `KW_ONLY` marker, which the snapshot does not hold."""
NO_INIT = [D[0].replace("frozen=True", "init=False"), *D[1:]]
TUPLE = ["m:T class(NamedTuple)", "  .a: int"]
ATTRS = ["m:A @attrs.frozen class()", "  .a: int"]
PLAIN = ["m:P class()", "  .a: int"]


def _with(lines: list[str], at: int, *added: str) -> list[str]:
    return [*lines[:at], *added, *lines[at:]]


@pytest.mark.parametrize(
    ("released", "current", "change"),
    [
        pytest.param(D, _with(D, 3, "  .w: int = 0"), "", id="default-last"),
        pytest.param(
            D, _with(D, 3, "  .w: list[int] = field(default_factory=list)"), "", id="factory"
        ),
        pytest.param(D, _with(D, 3, "  .w: int = field(init=False)"), "", id="not-in-init"),
        pytest.param(D, _with(D, 3, "  .W: ClassVar[int]"), "", id="class-var"),
        pytest.param(D, _with(D, 3, "  .w: int"), "changed", id="required"),
        pytest.param(D, _with(D, 3, "  .w: int = field()"), "changed", id="required-field"),
        pytest.param(D, _with(D, 2, "  .w: int = 0"), "changed", id="default-between"),
        pytest.param(WIDE, [*WIDE[:2], WIDE[3], WIDE[2], WIDE[4]], "changed", id="reordered"),
        pytest.param(
            D, _with(D, 3, "  .w: int = field(kw_only=True)"), "changed", id="field-keyword"
        ),
        pytest.param(
            D,
            _with(D, 3, "  .w: int = field(default=0, kw_only=True)"),
            "",
            id="field-keyword-default",
        ),
        pytest.param(KW, _with(KW, 2, "  .w: int = 0"), "", id="keyword-between"),
        pytest.param(KW, _with(KW, 3, "  .w: int"), "changed", id="keyword-required"),
        pytest.param(NO_INIT, _with(NO_INIT, 3, "  .w: int"), "", id="no-init"),
        pytest.param(MARKED, [*MARKED, "  .w: int"], "changed", id="marked-required"),
        pytest.param(MARKED, _with(MARKED, 2, "  .w: int = 0"), "", id="marked-default"),
        pytest.param(TUPLE, [*TUPLE, "  .b: int = 0"], "", id="tuple-default"),
        pytest.param(TUPLE, [*TUPLE, "  .b: int"], "changed", id="tuple-required"),
        pytest.param(ATTRS, [*ATTRS, "  .b: int = attrs.field(default=0)"], "", id="attrs-default"),
        pytest.param(ATTRS, [*ATTRS, "  .b: int"], "changed", id="attrs-required"),
        pytest.param(ATTRS, [*ATTRS, "  .b: int = attrs.field()"], "changed", id="attrs-field"),
        pytest.param(PLAIN, [*PLAIN, "  .b: int"], "", id="plain-class"),
    ],
)
def test_a_class_built_from_its_fields_changes_them_only_as_its_init_allows(
    released: list[str], current: list[str], change: str
) -> None:
    """A dataclass, an attrs class or a `NamedTuple` writes its `__init__` from its fields, so a
    field that comes without a default, or that moves the fields a caller passes by position,
    breaks a caller who builds it: it comes only in a major release from 1.0. A field with a
    default after the others, a keyword-only one, one the `__init__` does not take, and a class
    that writes no `__init__`, break none."""
    found = breaks(released, (1, 2, 0), current, (1, 9, 0))
    assert [f.rsplit(": ", 1)[1] for f in found] == ([change] if change else [])
    assert breaks(released, (1, 2, 0), current, (2, 0, 0)) == []


def test_a_released_name_leaves_only_as_the_removal_rule_says() -> None:
    """A name leaves where the release held it leaving, at the release it named, and from 1.0 in a
    major release; a name may start leaving in any release. Before 1.0 nothing but a leaving name
    changes, so 1.0 holds what the last release before it held."""
    gone = [line for line in RELEASED if not line.startswith("m:old ")]
    assert breaks(RELEASED, (0, 3, 0), gone, (0, 3, 1)) == ["m:old left before 1.0.0"]
    assert breaks(RELEASED, (0, 3, 0), gone, (1, 0, 0)) == []
    soon = [line.replace("1.0.0", "1.3.0") for line in RELEASED]
    assert breaks(soon, (1, 2, 0), gone, (1, 3, 0)) == ["m:old left in a minor release"]
    left = [line for line in RELEASED if line != F]
    assert breaks(RELEASED, (1, 2, 0), left, (2, 0, 0)) == [
        "m:f left with no release in _DEPRECATED"
    ]
    leaving = [line if line != F else "m:f renamed g, leaves in 2.0.0" for line in RELEASED]
    assert breaks(RELEASED, (1, 2, 0), leaving, (1, 2, 1)) == []
    retyped = [line.replace("b: float", "b: int") for line in RELEASED]
    assert breaks(RELEASED, (0, 3, 0), retyped, (1, 0, 0)) != []


def test_no_public_name_shadows_a_submodule_of_its_package() -> None:
    found = {
        f"{m.__name__}.{name}"
        for m, name in public()
        if hasattr(m, "__path__") and importlib.util.find_spec(f"{m.__name__}.{name}") is not None
    }
    assert found == SHADOWING


def test_the_snapshot_is_not_written_without_a_name_before_the_release_it_leaves_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Writing the snapshot again does not hide a removal: a name dropped without a rename, or a
    renamed one before its release, stops it; a renamed name leaves at its release."""
    held = tmp_path / "public_api.txt"
    held.write_text(SNAPSHOT.read_text(encoding="utf-8"), encoding="utf-8")
    here = sys.modules[__name__]
    monkeypatch.setattr(here, "SNAPSHOT", held)
    lines = surface()
    for gone in ("mixdyn.decisions:plan_margins ", "mixdyn.decisions:stacked "):
        monkeypatch.setattr(
            here, "surface", lambda gone=gone: [line for line in lines if not line.startswith(gone)]
        )
        assert write() == 1
        assert gone.strip() in capsys.readouterr().err
        assert held.read_text(encoding="utf-8") == SNAPSHOT.read_text(encoding="utf-8")
    monkeypatch.setattr(metadata, "version", lambda _: "1.0.0")
    assert write() == 0
    assert "mixdyn.decisions:stacked " not in held.read_text(encoding="utf-8")


def write() -> int:
    """Write the snapshot; refuse where it would drop a name before the release it leaves in."""
    lines = surface()
    held = SNAPSHOT.read_text(encoding="utf-8").splitlines() if SNAPSHOT.exists() else []
    version = _release(metadata.version("mixdyn"))
    left = {
        found["head"]
        for line in held
        if (found := _LEAVING.fullmatch(line)) and version >= _release(found["leaves"])
    }
    dropped = sorted(_heads(held) - _heads(lines) - left)
    if dropped:
        print(
            f"not written: {dropped} would leave the surface; name each in its package's"
            " _DEPRECATED with its replacement and the release it leaves in (ADR 0062)",
            file=sys.stderr,
        )
        return 1
    SNAPSHOT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        raise SystemExit(write())
