"""The wheel holds the package alone. hatchling builds it from `src/mixdyn` and nothing else, so
the tests, the evidence, the scorecard and the scripts never ship; this is read from the build's
configuration, without a build. `docs/release-check.md` records a build that confirmed it."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_the_wheel_is_built_from_the_package_alone() -> None:
    build = CONFIG["tool"]["hatch"]["build"]
    assert CONFIG["build-system"]["build-backend"] == "hatchling.build"
    assert set(build) == {"targets"}
    assert build["targets"]["wheel"] == {"packages": ["src/mixdyn"]}
