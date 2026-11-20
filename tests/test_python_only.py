"""The repository holds Python and its data, and no R.

Oracles run in R, glmnet's and Robyn's among them, are kept as JSON fixtures, each naming what
recorded it; the scripts that ran them stay outside the tree.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R_SUFFIXES = {".r", ".rmd", ".rnw", ".rproj", ".rds", ".rda", ".rdata"}
R_NAMES = {"renv.lock", ".rprofile"}


def r_files(root: Path) -> list[Path]:
    found: list[Path] = []
    for directory, subdirectories, files in root.walk():
        subdirectories[:] = [d for d in subdirectories if not d.startswith(".")]
        found += [
            directory / f
            for f in files
            if Path(f).suffix.lower() in R_SUFFIXES or f.lower() in R_NAMES
        ]
    return found


def test_the_repository_holds_no_r() -> None:
    assert [str(p.relative_to(ROOT)) for p in r_files(ROOT)] == []
