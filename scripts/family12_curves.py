"""Write the true curves of family 12's first pilot worlds, Meridian's home ground in the bench
(causaldyn-bench, `causaldyn_bench.scorecard.meridian`), for the budget plans' tests.

Each world's media are Meridian 2.1.0's default transform at draws of its default priors: each
channel's spend through a geometric adstock normalised over max_lag + 1 = 9 weeks, then a Hill of
slope 1 at the channel's inflexion, times its coefficient. Its quarter is the bench's: 13 weeks, the
budget 13 times last year's mean weekly spend over every channel, the status quo each channel at
that mean. A plan reads the history's last 8 weeks, all the carryover a 9-week kernel brings into
the quarter, so the record keeps those.

Run from the tree's root, the bench importable, to write the fixture:

    PYTHONPATH=src:<bench>/src python scripts/family12_curves.py \\
        > tests/fixtures/oracles/family12_curves.json
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import tomllib
from pathlib import Path

import causaldyn_bench
import numpy as np
from causaldyn_bench.mmm_decision import BOX, PLANNED, Quarter
from causaldyn_bench.scorecard.meridian import MAX_LAG, MERIDIAN

WORLDS = 8


def main() -> None:
    records = []
    for seed in list(MERIDIAN.pilots["priors"])[:WORLDS]:
        history = MERIDIAN.world("priors", seed).history
        quarter = Quarter.after(history, BOX)
        records.append(
            {
                "seed": seed,
                "channels": list(history.channels),
                "retention": list(history.retention),
                "inflexion": list(history.saturation),
                "coefficient": list(history.effect),
                "history": history.spend[-MAX_LAG:].tolist(),
                "status_quo": quarter.status_quo.tolist(),
                "budget": quarter.budget,
            }
        )
    tree = Path(causaldyn_bench.__file__).resolve().parents[2]
    project = tomllib.loads((tree / "pyproject.toml").read_text())["project"]
    commit = subprocess.run(
        ["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    document = {
        "provenance": {
            "generator": (
                "causaldyn-bench family 12 (Meridian's home ground), environment 'priors', its "
                f"first {WORLDS} pilot worlds, written by scripts/family12_curves.py"
            ),
            "causaldyn-bench": project["version"],
            "bench": {"commit": commit},
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
        "planned": PLANNED,
        "length": MAX_LAG + 1,
        "worlds": records,
    }
    json.dump(document, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
