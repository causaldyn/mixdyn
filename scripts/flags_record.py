"""Write a scored run's flags reading: each claim arm's flags read as classifiers of a plan's regret
above the status quo's, and every in-sample R² the run's records hold read against the plan's
regret, family by rung (`scorecard.flags`), for the benchmark page.

The reading is post hoc: no pre-registration names the flags, their AUROC or `tau`, and it was
read after the run's verdicts. It changes no verdict. It holds only what reads no `tau`, the
relative error of a channel's ROI past which its return counts as wrong: `scorecard.flags.TAU` is a
placeholder no pre-registration has frozen, so a flag read against a wrong return is left out until
one does. The claim arms record no draws of the returns, so their flags read regret alone.

The records are not in this repository, so it runs in two steps, the first once for each run:

    PYTHONPATH=<causaldyn-bench>/src:. uv run python scripts/flags_record.py --run m5 STAGE
    uv run python scripts/readme_record.py

STAGE is the scored stage's directory, as `readme_record.py extract` reads it. Each root that
declares the claim's margin is read as the claim reads it, its joined root's arms joined
(`scorecard.claim.read_run`), and a family's roots together. Each rung is read over the look every
environment holds whole, as `report --flags` reads a root (`scorecard.report.main`), on one set of
resamples: the claim's arm at that rung, the run's default at k = 0 and its default above 0 at the
rung above it, by its flags against regret; and every arm that holds an R² by its rank correlation
with regret. It writes `evidence/results/m5_flags.json` (`--run m4`: `m4_flags.json`), with the
SHA-256 of each root's `scores.jsonl` and of `scorecard/flags.py`, and no path; the second command
writes the page's blocks from it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from causaldyn_bench.scorecard.scoring import BUILT_IN, look

from scorecard import claim, flags
from scorecard.layout import Layout
from scorecard.report import whole

if TYPE_CHECKING:
    from causaldyn_bench.scorecard.scoring import WorldRecord

    from scorecard.layout import Declaration

ROOT = Path(__file__).resolve().parent.parent
RUNS = {"m4": "M4", "m5": "M5"}
OUTCOMES: tuple[flags.Outcome, ...] = ("regret",)
"""The outcomes read: the one that needs no `tau`."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def written(run: str) -> Path:
    return ROOT / "evidence" / "results" / f"{run}_flags.json"


def _arm(declarations: list[Declaration], k: int) -> str:
    """The claim's arm at rung `k`, one for every root of the family that holds the rung."""
    arms = {d.default if k == 0 else d.default_above_0 for d in declarations if k in d.rungs}
    if len(arms) != 1 or None in arms:
        msg = f"the family's roots declare {sorted(map(str, arms))} as the claim's arm at k = {k}"
        raise ValueError(msg)
    return str(arms.pop())


def _reading(
    family: int, k: int, held: list[tuple[Path, Declaration]], records: list[WorldRecord]
) -> dict[str, Any]:
    at = [r for r in records if r.k == k]
    worlds = look(at, whole(at))
    arm = _arm([d for _, d in held], k)
    drawn = flags.resamples(len(worlds))
    arms = sorted({a for r in worlds for a in r.arms} - set(BUILT_IN))
    fits = [
        {
            "arm": other,
            "key": key,
            "rho": fit.value,
            "low": fit.low,
            "high": fit.high,
            "worlds": fit.worlds,
        }
        for other in arms
        for key, fit in sorted(flags.fit_against_regret(worlds, other, drawn).items())
    ]
    roots = [root.name for root, d in held if k in d.rungs]
    roots += [root.name + d.joined for root, d in held if k in d.rungs and k > 0 and d.joined]
    return {
        "family": family,
        "k": k,
        "arm": arm,
        "roots": sorted(roots),
        "worlds": len(worlds),
        "failed": {
            a: sum(1 for r in worlds if a in r.arms and r.arms[a].failure is not None) for a in arms
        },
        "flags": flags.read(worlds, arm, drawn=drawn, outcomes=OUTCOMES),
        "fits": fits,
    }


def extract(stage: Path, run: str) -> dict[str, Any]:
    """The flags reading of the scored run under `stage` (see the module)."""
    families: dict[int, list[tuple[Path, Declaration]]] = defaultdict(list)
    for root in sorted(p.parent for p in stage.glob("*/run.json")):
        d = Layout(root).read()
        if d.margin is not None:
            families[d.family].append((root, d))
    sources: dict[str, dict[str, str]] = {}
    reads = []
    for family, held in sorted(families.items()):
        records: list[WorldRecord] = []
        for root, d in held:
            records += claim.read_run(root)[1]
            sources[root.name] = {"scores.jsonl": sha256(root / "scores.jsonl")}
            if d.joined is not None:
                joined = root.with_name(root.name + d.joined)
                sources[joined.name] = {"scores.jsonl": sha256(joined / "scores.jsonl")}
        reads += [_reading(family, k, held, records) for k in sorted({r.k for r in records})]
    return {
        "about": (
            f"{RUNS[run]}'s claim arms' flags read as classifiers of a plan's regret above the "
            "status quo's, and every in-sample R² its records hold read against the plan's regret, "
            "family by rung, written by `scripts/flags_record.py` from the run's scores; each file "
            "it read is named by the run's root and its SHA-256. Post hoc: no pre-registration "
            "names the flags, and the reading changes no verdict."
        ),
        "post_hoc": True,
        "outcomes": list(OUTCOMES),
        "resamples": flags.RESAMPLES,
        "seed": flags.SEED,
        "level": flags.LEVEL,
        "code": {"scorecard/flags.py": sha256(ROOT / "scorecard" / "flags.py")},
        "sources": {"roots": dict(sorted(sources.items()))},
        "reads": reads,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", choices=sorted(RUNS), required=True)
    parser.add_argument("stage", type=Path)
    args = parser.parse_args(argv)
    record = extract(args.stage, args.run)
    text = json.dumps(record, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
    written(args.run).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
