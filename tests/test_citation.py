"""CITATION.cff names the version and date the package was released as.

Zenodo archives a release with the metadata this file holds, so a citation that still names the
previous version would mint a DOI under the wrong number.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CITATION = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]


def _field(name: str) -> str:
    found = re.findall(rf'^{name}: "?([^"\n]+)"?$', CITATION, flags=re.MULTILINE)
    assert len(found) == 1, f"{name}: {found}"
    return found[0]


def test_the_citation_names_the_package_version() -> None:
    assert _field("version") == VERSION


def test_the_citation_date_is_the_changelog_date_of_that_version() -> None:
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    dates = re.findall(
        rf"^## \[{re.escape(VERSION)}\] — (\d{{4}}-\d{{2}}-\d{{2}})$", changelog, re.MULTILINE
    )
    assert dates == [_field("date-released")]
