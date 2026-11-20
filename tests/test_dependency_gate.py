from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from check_runtime_deps import (
    REPO,
    main,
    normalize,
    outside_allowlist,
    read_allowlist,
    runtime_closure,
)

LOCK = """
version = 1
requires-python = ">=3.14"

[[package]]
name = "mixdyn"
version = "0.0.1"
source = { editable = "." }
dependencies = [{ name = "numpy" }, { name = "helper" }]

[package.optional-dependencies]
plots = [{ name = "matplotlib" }]

[package.dev-dependencies]
dev = [{ name = "pytest" }]

[[package]]
name = "numpy"
version = "2.5.0"

[[package]]
name = "helper"
version = "1.0"
dependencies = [{ name = "statsmodels" }, { name = "jax", extra = ["cpu"] }]

[[package]]
name = "statsmodels"
version = "0.14"
dependencies = [{ name = "patsy" }]

[[package]]
name = "patsy"
version = "1.0"

[[package]]
name = "jax"
version = "0.11"

[package.optional-dependencies]
cpu = [{ name = "jaxlib" }]
cuda = [{ name = "jax-cuda-plugin" }]

[[package]]
name = "jaxlib"
version = "0.11"

[[package]]
name = "jax-cuda-plugin"
version = "0.11"

[[package]]
name = "matplotlib"
version = "3.10"

[[package]]
name = "pytest"
version = "8.0"
"""


def test_a_special_package_planted_two_levels_down_is_caught_with_its_path() -> None:
    closure = runtime_closure(tomllib.loads(LOCK), "mixdyn")
    allowed = read_allowlist("numpy\nhelper\njax\njaxlib\nmatplotlib\n")
    assert outside_allowlist(closure, allowed) == {
        "patsy": ("mixdyn", "helper", "statsmodels", "patsy"),
        "statsmodels": ("mixdyn", "helper", "statsmodels"),
    }


def test_the_walk_follows_requested_extras_and_skips_development_groups() -> None:
    closure = runtime_closure(tomllib.loads(LOCK), "mixdyn")
    assert "jaxlib" in closure
    assert "matplotlib" in closure  # the project's own extras install at run time
    assert "jax-cuda-plugin" not in closure
    assert "pytest" not in closure


def test_the_allowlist_ignores_comments_and_normalises_names() -> None:
    assert read_allowlist("# general\nScikit_Learn  # the author's call\n\nnumpy\n") == {
        "scikit-learn",
        "numpy",
    }
    assert normalize("Typing.Extensions") == "typing-extensions"


def test_the_gate_fails_on_a_repository_with_a_planted_package(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "uv.lock").write_text(LOCK)
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "mixdyn"\n')
    (tmp_path / "allowed-runtime-packages.txt").write_text(
        (REPO / "allowed-runtime-packages.txt").read_text()
    )
    assert main(tmp_path) == 1
    reported = capsys.readouterr().err
    assert "statsmodels  via mixdyn -> helper -> statsmodels" in reported


def test_the_repository_passes_its_own_gate() -> None:
    assert main(REPO) == 0
