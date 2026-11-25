"""SECURITY.md's supported-versions table lists the release this checkout builds.

The table is a copy of the version that nothing else checks: 0.3.0 shipped with it still naming
0.2.x, so the policy supported no release that was being installed.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]


def test_the_security_policy_supports_the_current_minor() -> None:
    major, minor, *_ = VERSION.split(".")
    text = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    assert f"| {major}.{minor}.x | yes |" in text
    assert f"| < {major}.{minor} | no |" in text
