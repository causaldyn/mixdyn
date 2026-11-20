"""Fail when a package the engine installs at run time is not on the allow-list (ADR 0001).

The runtime closure is read from uv.lock, not from pyproject.toml: a special-purpose package arrives
as often through a general package's dependency as through a direct one. The walk starts at the
project, follows its dependencies and every extra it declares, and on each edge follows only the
extras that edge requests. Development groups are never followed. Edges carrying an environment
marker are followed regardless of the marker, so a package any platform would install is checked.
"""

from __future__ import annotations

import re
import sys
import tomllib
from collections import defaultdict, deque
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ALLOWLIST = "allowed-runtime-packages.txt"


def normalize(name: str) -> str:
    """The PEP 503 form uv.lock writes names in."""
    return re.sub(r"[-_.]+", "-", name).lower()


def read_allowlist(text: str) -> frozenset[str]:
    """One package per line; `#` starts a comment."""
    names = (line.split("#", 1)[0].strip() for line in text.splitlines())
    return frozenset(normalize(name) for name in names if name)


def runtime_closure(lock: dict, root: str) -> dict[str, tuple[str, ...]]:
    """Every package `root` installs at run time, with the first path found that brings it in."""
    entries: defaultdict[str, list[dict]] = defaultdict(list)
    for entry in lock.get("package", []):
        entries[entry["name"]].append(entry)
    if root not in entries:
        msg = f"{root!r} is not a package in the lock"
        raise KeyError(msg)
    root_extras = {x for entry in entries[root] for x in entry.get("optional-dependencies", {})}
    paths: dict[str, tuple[str, ...]] = {root: (root,)}
    queue: deque[tuple[str, str | None]] = deque([(root, None), *((root, x) for x in root_extras)])
    seen: set[tuple[str, str | None]] = set()
    while queue:
        name, extra = queue.popleft()
        if (name, extra) in seen:
            continue
        seen.add((name, extra))
        for entry in entries[name]:
            if extra is None:
                edges = entry.get("dependencies", [])
            else:
                edges = entry.get("optional-dependencies", {}).get(extra, [])
            for edge in edges:
                child = edge["name"]
                paths.setdefault(child, (*paths[name], child))
                queue.append((child, None))
                queue.extend((child, x) for x in edge.get("extra", []))
    del paths[root]
    return paths


def outside_allowlist(
    closure: dict[str, tuple[str, ...]], allowed: frozenset[str]
) -> dict[str, tuple[str, ...]]:
    return {name: path for name, path in sorted(closure.items()) if name not in allowed}


def main(repo: Path = REPO) -> int:
    lock = tomllib.loads((repo / "uv.lock").read_text())
    project = normalize(tomllib.loads((repo / "pyproject.toml").read_text())["project"]["name"])
    closure = runtime_closure(lock, project)
    outside = outside_allowlist(closure, read_allowlist((repo / ALLOWLIST).read_text()))
    if outside:
        print(f"runtime packages not on {ALLOWLIST} (ADR 0001):", file=sys.stderr)
        for name, path in outside.items():
            print(f"  {name}  via {' -> '.join(path)}", file=sys.stderr)
        return 1
    print(f"{len(closure)} runtime packages, all allowed: {', '.join(sorted(closure))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
