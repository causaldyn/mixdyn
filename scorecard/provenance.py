"""Where a record was made: the engine's and the bench's commits and whether their trees were clean,
the packages' versions, and the settings a number depends on.

A scored run is pre-registered against the bench's commit, so it refuses to start unless the bench
is at that commit, both trees are clean and JAX runs in double precision (`require`). A pilot or a
smoke run records the same and refuses nothing but single precision. Any run that declares its
pre-registration also refuses to start once the file is gone or its hash moved since the run was
declared, before it spends compute on a design it no longer states. Every world's process computes
on one thread, whatever the run (`ONE_THREAD`).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import causaldyn_bench
import jax
from causaldyn_bench.scorecard.preregistration import Stamp, verify

import mixdyn
from scorecard.layout import TREE

ENVIRONMENT = (
    "JAX_ENABLE_X64",
    "JAX_PLATFORMS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "PJRT_NPROC",
    "XLA_FLAGS",
)
ONE_THREAD = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PJRT_NPROC": "1",
    "MKL_NUM_THREADS": "1",
}
"""What keeps a process's arithmetic on one thread: NumPy's BLAS by the first two, XLA's CPU pools
by `PJRT_NPROC`, and MKL where a tool's environment links it. XLA reads neither of the first two
and otherwise keeps a thread a CPU."""


def tree(module: Any) -> Path:
    """The root of the git tree a module is imported from, its sources under `src/`."""
    return Path(module.__file__).resolve().parents[2]


def checkout(module: Any) -> dict[str, Any]:
    """The commit of the git tree a module is imported from, and whether the tree is clean."""
    root = tree(module)

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    return {
        "path": str(root),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def provenance() -> dict[str, Any]:
    """The engine's and the bench's checkouts, and the environment a number depends on."""
    return {
        "mixdyn": checkout(mixdyn),
        "bench": checkout(causaldyn_bench),
        "environment": {name: os.environ.get(name) for name in ENVIRONMENT},
    }


def require(*, bench: str | None, preregistration: Stamp | None = None) -> dict[str, Any]:
    """The provenance, once the run may start: JAX in double precision always; the pre-registration
    the run declares, `preregistration`, still at its path, a relative one read from the engine's
    tree, and still at the SHA-256 declared; for a scored run, `bench` the commit it is
    pre-registered against, the bench there and both trees clean.

    Raises:
        RuntimeError: on what the run may not start with, every reason named.
    """
    source = provenance()
    problems = []
    if not jax.config.x64_enabled:
        problems.append("JAX runs in single precision; set JAX_ENABLE_X64=1")
    if preregistration is not None:
        try:
            verify(preregistration, TREE)
        except (FileNotFoundError, ValueError) as moved:  # the file is read here, a boundary
            problems.append(f"the pre-registration is not the one declared: {moved}")
    if bench is not None:
        if source["bench"]["commit"] != bench:
            problems.append(f"the bench is at {source['bench']['commit']}, not {bench}")
        problems += [
            f"{name}'s tree is not clean" for name in ("mixdyn", "bench") if source[name]["dirty"]
        ]
    if problems:
        raise RuntimeError("; ".join(problems))
    return source
