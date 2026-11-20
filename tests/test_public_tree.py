"""What the tree holds can be traced from outside it.

Every fixture names a package and version that recorded it, one of the oracles the tests read, and
records no application's commit; no module is a legacy mode of another engine.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RECORDERS = frozenset(
    {
        "arviz-stats",
        "causal-hybrid-control",
        "causaldyn-bench",
        "cmdstan",
        "glmnet",
        "maxima",
        "robyn",
    }
)


def fixtures() -> list[Path]:
    return sorted(p for p in (ROOT / "tests" / "fixtures").rglob("*") if p.is_file())


def provenance(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if path.suffix == ".gz":
        data = gzip.decompress(data)
    return json.loads(data)["provenance"]


def test_every_fixture_names_the_package_and_version_that_recorded_it() -> None:
    unnamed = [
        str(p.relative_to(ROOT))
        for p in fixtures()
        if not any(
            isinstance(version, str) and version[:1].isdigit()
            for name, version in provenance(p).items()
            if name in RECORDERS
        )
    ]
    assert fixtures()
    assert unnamed == []


def test_no_fixture_records_an_application_s_commit() -> None:
    assert [str(p.relative_to(ROOT)) for p in fixtures() if "app_commit" in provenance(p)] == []


def test_no_module_is_a_legacy_mode() -> None:
    found = [
        str(p.relative_to(ROOT))
        for directory in ("src", "tests", "scripts", "scorecard")
        for p in (ROOT / directory).rglob("*.py")
        if "legacy" in p.stem.lower()
    ]
    assert found == []
