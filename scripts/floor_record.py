"""Write the spend floor's reading on a scored run: at the tested rung, each family's paired
difference in regret between `anchored floored`, the claim's arm, and `anchored held`, the same rule
without the floor, both recorded on the same worlds (ADR 0102).

The reading is post hoc and unadjusted: no pre-registration declares it, and it changes no verdict.
A world counts where both arms planned. The difference is floored's regret less held's, a euro of
budget, so a positive mean is what the floor costs; its interval is the Student t's at `LEVEL`.

The records are not in this repository, so it runs once for each run:

    uv run python scripts/floor_record.py --run m4 STAGE

STAGE is the scored stage's directory, as `readme_record.py extract` reads it. Every root's
`scores.jsonl` is read, and each record above k = 0 that holds both arms is a world of its family.
It writes the run's reading into `evidence/results/floor_record.json`, beside the other run's, with
the SHA-256 of each `scores.jsonl` it read and no path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
WRITTEN = ROOT / "evidence" / "results" / "floor_record.json"
RUNS = {"m4": "M4", "m5": "M5"}
FLOORED, HELD = "anchored floored", "anchored held"
LEVEL = 0.95


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _regret(arm: dict[str, Any] | None) -> float | None:
    if arm is None or arm["failure"] is not None:
        return None
    return arm["regret"]


def extract(stage: Path) -> dict[str, Any]:
    """The floor's reading of the scored run under `stage` (see the module).

    Raises:
        ValueError: on a world recorded twice.
    """
    differences: dict[int, list[float]] = defaultdict(list)
    seen: set[tuple[int, str, int, int]] = set()
    sources = {}
    for scores in sorted(stage.glob("*/scores.jsonl")):
        read = False
        for line in scores.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            floored = _regret(record["arms"].get(FLOORED))
            held = _regret(record["arms"].get(HELD))
            if record["k"] == 0 or floored is None or held is None:
                continue
            key = (record["family"], record["environment"], record["seed"], record["k"])
            if key in seen:
                msg = f"the world {key} is recorded twice"
                raise ValueError(msg)
            seen.add(key)
            differences[record["family"]].append(floored - held)
            read = True
        if read:
            sources[scores.parent.name] = {"scores.jsonl": sha256(scores)}
    families = []
    for family, found in sorted(differences.items()):
        x = np.asarray(found)
        mean = float(x.mean())
        half = float(stats.t.ppf((1.0 + LEVEL) / 2.0, x.size - 1)) * float(x.std(ddof=1))
        half /= math.sqrt(x.size)
        families.append(
            {
                "family": family,
                "worlds": x.size,
                "mean": mean,
                "low": mean - half,
                "high": mean + half,
            }
        )
    return {"sources": {"roots": sources}, "families": families}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", choices=sorted(RUNS), required=True)
    parser.add_argument("stage", type=Path)
    args = parser.parse_args(argv)
    held = json.loads(WRITTEN.read_text(encoding="utf-8")) if WRITTEN.exists() else {"runs": {}}
    record = {
        "about": (
            "The spend floor's reading on the scored runs at the tested rung: each family's paired"
            f" difference in regret, `{FLOORED}`'s less `{HELD}`'s, a euro of budget, over the"
            " worlds where both planned, its mean and its Student t interval, written by"
            " `scripts/floor_record.py` from each run's scores; each file it read is named by the"
            " run's root and its SHA-256. Post hoc and unadjusted: no pre-registration declares"
            " it, and it changes no verdict."
        ),
        "post_hoc": True,
        "level": LEVEL,
        "runs": dict(sorted((held["runs"] | {args.run: extract(args.stage)}).items())),
    }
    text = json.dumps(record, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
    WRITTEN.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
