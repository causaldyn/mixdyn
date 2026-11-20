"""The shipped geo design against the comparator on the bench's geo worlds: the minimum detectable
lift each reads over the backtests a world's second year holds, and how often its size is flagged.
Detectable lifts and sizes only, no wall time.

The worlds are causaldyn-bench's families 8 (ten geos, policies 0 and 0.5) and 10 (three, eight
and fifty small noisy geos), the first `--worlds` pilot worlds of each environment, each a history
of 104 weeks. Every design reads the first 52 weeks; every design is audited on the backtests whose
readouts lie in the last 52, a test of 4 weeks read after 26, 49 backtests a world:

    pairs    `assign(pair_geos(mean KPI over weeks 1-52), seed)`, the shipped design (ADR 0049)
    strata   `stratified_design(KPI over weeks 1-52, 4, pre=26, seed=seed)`, the comparator

each read by `read_did` on the groups' sums, `power`'s default (ADR 0055), and by `read_tbr`, the
comparator's reader: the shipped design is `pairs` read by `read_did`, the comparator `strata` read
by `read_tbr`, and the other two pairings tell the design's part from the reader's. A world's seed
is the design's.

Each audit is `audit_power`'s at lifts 0.001 to 1 in steps of 0.001, read off one reading of each
backtest with no lift: `read_did` and `read_tbr` are linear in the lift, which moves a backtest's
effect by the lift times the treated sum over its readout and leaves its error alone
(tests/test_geo_design.py holds it), and the first world of each environment checks the shortcut
against `audit_power` itself on every tenth lift. Each world writes one JSON line to `--out`;
`--summary` reads them back: each environment's median MDE by design and reader, the geometric mean
of the shipped design's MDE over the comparator's with its 95 % t interval over the worlds, the
same for the two designs under `read_did`; and by arm the mean size over every backtest, the mean
share of rejections over the 13 whose readouts do not overlap, the share of audits flagged by the
binomial test of that count, and the median MDE range.

ADR 0049's amendment of 2026-11-06 is this run and its summary:

    PYTHONPATH=<causaldyn-bench>/src:src JAX_PLATFORMS=cpu python scripts/study_geo_design.py \\
        --worlds 30 --out geo_design.jsonl
    python scripts/study_geo_design.py --out geo_design.jsonl --summary
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from mixdyn.experiments import GeoDesign, assign, audit_power, pair_geos, stratified_design
from mixdyn.experiments.geo_design import (
    Reader,
    _binomial_p_value,
    _detectable,
    _read_did,
    _read_tbr,
)

PRE, READOUT, DESIGN_WEEKS = 26, 4, 52
ENVIRONMENTS = ("policy-0", "policy-0.5", "geos-3", "geos-8", "geos-50")
LIFTS = np.arange(1, 1001) / 1000.0
READERS: dict[str, Reader] = {"did": _read_did, "tbr": _read_tbr}
ARMS = (("pairs", "did"), ("strata", "tbr"), ("pairs", "tbr"), ("strata", "did"))
Z = float(stats.norm.ppf(0.975))


def audit(history: np.ndarray, design: GeoDesign, read: Reader, lifts: np.ndarray) -> dict:
    """`audit_power(history, design, READOUT, lifts, pre=PRE, read=read)`, off one reading a
    backtest."""
    treated, control = list(design.treated), list(design.control)
    backtests = history.shape[1] - PRE - READOUT + 1
    rows = np.empty((backtests, 3))
    for b in range(backtests):
        window = history[:, b : b + PRE + READOUT]
        effect, se = read(window[treated], window[control], PRE)
        rows[b] = effect, se, window[treated][:, PRE:].sum()
    scaled = np.concatenate([[0.0], lifts])
    detected = np.abs(rows[:, :1] + scaled[None, :] * rows[:, 2:]) > Z * rows[:, 1:2]
    half = backtests // 2
    ends = [
        _detectable(scaled, detected[r].mean(axis=0))
        for r in (slice(0, half), slice(backtests - half, backtests))
    ]
    finite = [e for e in ends if math.isfinite(e)]
    size = float(detected[:, 0].mean())
    rejected = int(detected[::READOUT, 0].sum())
    disjoint = (backtests - 1) // READOUT + 1
    p_value = _binomial_p_value(rejected, disjoint)
    return {
        "backtests": backtests,
        "size": size,
        "disjoint": disjoint,
        "rejected": rejected,
        "flag_p_value": p_value,
        "flagged": p_value <= 0.01,
        "mde": _detectable(scaled, detected.mean(axis=0)),
        "low": min(finite) if finite else math.nan,
        "high": max(ends) if len(finite) == 2 else math.nan,
    }


def world(environment: str, seed: int, check: bool) -> dict[str, Any]:
    from causaldyn_bench.scorecard.few_geos import FEW_GEOS
    from causaldyn_bench.scorecard.geo import GEO

    family = GEO if environment.startswith("policy") else FEW_GEOS
    kpi = np.asarray(family.world(environment, seed).history.kpi, dtype=np.float64)
    record: dict[str, Any] = {"environment": environment, "seed": seed, "geos": kpi.shape[0]}
    designs = {"pairs": assign(pair_geos(kpi[:, :DESIGN_WEEKS].mean(axis=1)), seed)}
    try:
        chosen = stratified_design(kpi[:, :DESIGN_WEEKS], READOUT, pre=PRE, seed=seed)
        designs["strata"] = chosen.design
        record["candidates"], record["kept"] = chosen.candidates, chosen.kept
    except ValueError as error:
        record["refused"] = str(error)
    later = kpi[:, DESIGN_WEEKS - PRE :]
    for name, design in designs.items():
        record[name] = {"treated": list(design.treated)}
        for reader, read in READERS.items():
            record[name][reader] = audit(later, design, read, LIFTS)
            if check:
                coarse = LIFTS[9::10]
                full = audit_power(later, design, READOUT, coarse, pre=PRE, read=read)
                short = audit(later, design, read, coarse)
                same = (full.size, full.mde, full.mde_low, full.rejected, full.flagged) == (
                    short["size"],
                    short["mde"],
                    short["low"],
                    short["rejected"],
                    short["flagged"],
                )
                same &= full.mde_high == short["high"] or math.isnan(full.mde_high + short["high"])
                if not same:
                    msg = (
                        f"the shortcut is not audit_power's on {environment} {seed} {name} {reader}"
                    )
                    raise AssertionError(msg)
    return record


def _ratio(a: np.ndarray, b: np.ndarray) -> str:
    """The geometric mean of `a / b` over the worlds where both are finite, its 95 % t interval."""
    ok = np.isfinite(a) & np.isfinite(b)
    logs = np.log(a[ok] / b[ok])
    half = float(stats.t.ppf(0.975, logs.size - 1)) * logs.std(ddof=1) / math.sqrt(logs.size)
    mean = float(logs.mean())
    return f"{math.exp(mean):.3f} [{math.exp(mean - half):.3f}, {math.exp(mean + half):.3f}]"


def summary(path: Path) -> None:
    records = [json.loads(line) for line in path.read_text().splitlines()]
    groups = {"all": records} | {
        e: [r for r in records if r["environment"] == e]
        for e in dict.fromkeys(r["environment"] for r in records)
    }

    def value(r: dict, arm: tuple[str, str], field: str) -> float:
        return float(r[arm[0]][arm[1]][field]) if arm[0] in r else math.nan

    print(
        "| environment | worlds | pairs, did | strata, tbr | pairs, tbr | strata, did "
        "| shipped / comparator | pairs / strata, did |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, rs in groups.items():
        mde = {arm: np.array([value(r, arm, "mde") for r in rs]) for arm in ARMS}
        medians = " | ".join(f"{np.nanmedian(mde[arm]):.4f}" for arm in ARMS)
        refused = sum("refused" in r for r in rs)
        print(
            f"| {name} | {len(rs)}{f' ({refused} refused)' if refused else ''} | {medians} | "
            f"{_ratio(mde[ARMS[0]], mde[ARMS[1]])} | {_ratio(mde[ARMS[0]], mde[ARMS[3]])} |"
        )
    print()
    print("| environment | " + " | ".join(f"{d}, {r}" for d, r in ARMS) + " |")
    print("| --- | --- | --- | --- | --- |")
    for name, rs in groups.items():
        cells = []
        for arm in ARMS:
            size = np.array([value(r, arm, "size") for r in rs])
            flagged = np.array([value(r, arm, "flagged") for r in rs])
            disjoint = np.array([value(r, arm, "rejected") / value(r, arm, "disjoint") for r in rs])
            low, high = (np.array([value(r, arm, f) for r in rs]) for f in ("low", "high"))
            cells.append(
                f"{np.nanmean(size):.3f}, {np.nanmean(disjoint):.3f}, {np.nanmean(flagged):.3f}, "
                f"{np.nanmedian(low):.3f}-{np.nanmedian(high):.3f}"
            )
        print(f"| {name} | " + " | ".join(cells) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("environments", nargs="*", default=list(ENVIRONMENTS))
    parser.add_argument("--worlds", type=int, default=30)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.summary:
        summary(args.out)
        return
    from causaldyn_bench.scorecard.few_geos import FEW_GEOS
    from causaldyn_bench.scorecard.geo import GEO

    with args.out.open("a") as out:
        for environment in args.environments:
            family = GEO if environment.startswith("policy") else FEW_GEOS
            for i, seed in enumerate(list(family.pilots[environment])[: args.worlds]):
                out.write(json.dumps(world(environment, seed, check=i == 0)) + "\n")
                out.flush()


if __name__ == "__main__":
    main()
