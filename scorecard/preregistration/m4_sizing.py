"""The M4 run's worlds a family, the order of its verdicts and their predictives, sized from M3's
scored records by the rules `m4.md` states (Sizing): scope 2, `margin held` at k = 0 and round 8's
rule, `anchored floored`, at the tested rung.

    PYTHONPATH=<bench>/src:. python scorecard/preregistration/m4_sizing.py --stage M3_STAGE \\
        --orbit-f17 SCORES.jsonl --floored ROWS.pkl [--out FILE.json]

`--stage` is M3's scored stage: its `design.json`, which names each family's scored environments,
and each root's `scores.jsonl`. `--orbit-f17` is the `scores.jsonl` of the re-run of 2026-11-20
with the fixed Orbit arm on M3's worlds of family 17, where M3 recorded Orbit as failed on every
world. `--floored` is round 8's development rows: each of M3's worlds at the tested rung planned
again with round 8's rule, a pickled list of {family, env, seed, index, k, regret}, `regret`
holding the rule's (`floor:sp:0.9`) and `anchored held`'s (`T0`). The rule's regret is read as
`anchored floored`; `T0` must equal M3's stored `anchored held` on every matched world.

**The sizing records.** Every complete case of M3's scored roots: families 1, 4, 6 and 14 in their
`e0-f<n>` (k = 0) and `et-f<n>` (rung 4) roots, M2's scored worlds with M2's rival records; family
7 pooled over `f7a` (k = 0 and 1) and `f7b` (k = 0 and 2), with their `-ts` roots; every other
family in `f<n>` with `f<n>-ts`. Family 17's Orbit records are the re-run's.

**k = 0** (M3's rule). Each comparison of `margin held` with a rival (the status quo, the equal
split, Meridian, PyMC-Marketing and Orbit at their recommended settings) needs
N = (2.8016 sd / margin)^2 worlds, in tens, sd the paired difference's (ddof 1) over the family's
worlds, its environments pooled. The rung needs the largest N over its rivals; past the cap it is
exploratory and does not size its family.

**The tested rung.** The larger of round 8's recommendation (520 on families 1 and 13, 360 on
family 4, the floor on the rest) and the same N on `anchored floored`, at most the cap; every
tested rung is claimed.

**The family** declares the largest of its claimed rungs' worlds, at least the floor, split evenly
over its environments and rounded up to whole worlds an environment.

**The predictive** of each verdict at the family's worlds an environment:
`scorecard.development.predictive`, the two-stage bootstrap, one stratum an environment, a column
a rival, nan where the world is not one of that comparison's complete cases. A rival that failed on
more than half of the worlds of every environment is not compared. A comparison below `CLAIMED` is
a predicted failure; the verdict's predictive is then read again over its claimed comparisons.

**The order.** Each claimed verdict whose rung is within the cap, by its predictive over its claimed
comparisons, unrounded, highest first; ties by family, then rung. Family 7's tested rung is named
1. The expected number confirmed is the fixed sequence's, as M3's: the sum over the order of the
running product of the predictives, the verdicts taken as independent.

The margin, the floor, the cap, the resamples and the seed are `m4.json`'s, beside this file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from scorecard.development import CLAIMED, predictive

HERE = Path(__file__).resolve().parent
DOC = json.loads((HERE / "m4.json").read_text(encoding="utf-8"))
MARGIN = float(DOC["margin"])
FLOOR = int(DOC["sizing"]["floor"])
CAP = int(DOC["sizing"]["cap"])
RESAMPLES = int(DOC["sizing"]["predictive"]["resamples"])
SEED = int(DOC["sizing"]["predictive"]["seed"])
Z = 2.8016
"""z(0.975) + z(0.8), rounded to four places, as M3's."""
FAMILIES = (0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 17, 18, 19, 20)
RIVALS = ("status quo", "equal split", "meridian", "pymc-marketing", "orbit")
K0, TESTED, HELD = "margin held", "anchored floored", "anchored held"
RULE, T0 = "floor:sp:0.9", "T0"
ROUND_8 = {1: 520, 13: 520, 4: 360}
"""Round 8's recommended worlds at the tested rung; the floor on every other family."""
KEEP = {*RIVALS, K0, HELD}

World = tuple[str, int, int, int]
"""(environment, seed, index, k)"""
Arms = dict[str, tuple[float | None, str | None]]
"""Each arm's (regret, failure)."""


def sources(family: int) -> dict[str, list[tuple[str, int]]]:
    """The (root, k) pairs that hold a family's worlds at each rung."""
    if family in (1, 4, 6, 14):
        return {"0": [(f"e0-f{family}", 0)], "tested": [(f"et-f{family}", 4)]}
    if family == 7:
        return {
            "0": [("f7a", 0), ("f7b", 0)],
            "tested": [("f7a", 1), ("f7a-ts", 1), ("f7b", 2), ("f7b-ts", 2)],
        }
    return {
        "0": [(f"f{family}", 0)],
        "tested": [(f"f{family}", 4), (f"f{family}-ts", 4)],
    }


def read_root(path: Path) -> dict[World, Arms]:
    """Each world's kept arms in a root's `scores.jsonl`."""
    out: dict[World, Arms] = {}
    with path.open(encoding="utf-8") as held:
        for line in held:
            record = json.loads(line)
            key = (record["environment"], record["seed"], record["index"], record["k"])
            out[key] = {
                arm: (a.get("regret"), a.get("failure"))
                for arm, a in record["arms"].items()
                if arm in KEEP
            }
    return out


class Records:
    """M3's scored records, root by root as they are read, with family 17's Orbit re-run."""

    def __init__(self, stage: Path, orbit_f17: Path) -> None:
        self.stage = stage
        self.roots: dict[str, dict[World, Arms]] = {}
        self.orbit_f17 = read_root(orbit_f17)
        self.notes: list[str] = []

    def root(self, name: str) -> dict[World, Arms]:
        if name not in self.roots:
            self.roots[name] = read_root(self.stage / name / "scores.jsonl")
        return self.roots[name]

    def worlds(self, family: int, rung: str) -> dict[World, Arms]:
        """The family's worlds at the rung, each with its arms merged over the roots that hold it.

        Raises:
            ValueError: where two roots record one arm of one world apart, or where the Orbit
                re-run's other arms differ from M3's on the same world.
        """
        merged: dict[World, Arms] = defaultdict(dict)
        for name, k in sources(family)[rung]:
            for key, arms in self.root(name).items():
                if key[3] != k:
                    continue
                for arm, held in arms.items():
                    if merged[key].setdefault(arm, held) != held:
                        msg = f"f{family} {rung}: two roots record {arm} of {key} apart"
                        raise ValueError(msg)
        if family == 17:
            agree = 0
            for key, arms in merged.items():
                again = self.orbit_f17.get(key)
                if again is None or "orbit" not in again:
                    continue
                for arm in ("status quo", "equal split", K0, "meridian"):
                    if arm in arms and arm in again:
                        if arms[arm] != again[arm]:
                            msg = f"f17 {rung}: the re-run's {arm} differs from M3's on {key}"
                            raise ValueError(msg)
                        agree += 1
                arms["orbit"] = again["orbit"]
            self.notes.append(f"f17 rung {rung}: Orbit from the re-run; other arms agree {agree}")
        self.notes.append(f"f{family} rung {rung}: {len(merged)} worlds")
        return dict(merged)


def with_floored(tested: dict[World, Arms], family: int, rows: list[dict]) -> int:
    """Round 8's rule added to the tested worlds as `anchored floored`; the matched worlds.

    Raises:
        ValueError: where round 8's `T0` is not M3's stored `anchored held`, or a world of M3's
            has no row of round 8's.
    """
    by_key = {(r["env"], r["seed"], r["index"], r["k"]): r for r in rows if r["family"] == family}
    for key, arms in tested.items():
        row = by_key.get(key)
        if row is None:
            msg = f"f{family}: round 8 holds no row of {key}"
            raise ValueError(msg)
        held = arms.get(HELD)
        if held is None or held[1] is not None or held[0] != row["regret"][T0]:
            msg = f"f{family}: round 8's T0 is not M3's {HELD} on {key}"
            raise ValueError(msg)
        arms[TESTED] = (row["regret"][RULE], None)
    return len(tested)


def needed(sd: float) -> int:
    return 10 * math.ceil((Z * sd / MARGIN) ** 2 / 10)


def comparisons(ws: dict[World, Arms], arm: str) -> list[dict]:
    """The arm less each rival on their complete cases: pairs, mean, sd and N."""
    out = []
    for rival in RIVALS:
        diffs = []
        for arms in ws.values():
            a, b = arms.get(arm), arms.get(rival)
            if a is None or b is None or a[1] is not None or b[1] is not None:
                continue
            if a[0] is None or b[0] is None:
                continue
            diffs.append(a[0] - b[0])
        row: dict = {"rival": rival, "pairs": len(diffs)}
        if len(diffs) >= 3:
            sd = statistics.stdev(diffs)
            row |= {"mean": statistics.fmean(diffs), "sd": sd, "needed": needed(sd)}
        out.append(row)
    return out


def tie_rule(ws: dict[World, Arms], arm: str) -> dict:
    rows = comparisons(ws, arm)
    worst = max((r for r in rows if "needed" in r), key=lambda r: r["needed"])
    return {
        "arm": arm,
        "needed": worst["needed"],
        "binding": worst["rival"],
        "comparisons": rows,
    }


def regret(held: tuple[float | None, str | None]) -> float:
    """An arm's regret on a world, nan where its plan failed.

    Raises:
        ValueError: on a plan that did not fail and carries no regret.
    """
    value, failure = held
    if failure is not None:
        return np.nan
    if value is None:
        msg = "a plan that did not fail carries a regret"
        raise ValueError(msg)
    return value


def strata(ws: dict[World, Arms], arm: str) -> tuple[list[str], list[np.ndarray], list[str]]:
    """The predictive's strata: one matrix an environment, a world a row, a compared rival a
    column; and the rivals not compared."""
    by_env: dict[str, list[Arms]] = defaultdict(list)
    for key, arms in ws.items():
        by_env[key[0]].append(arms)
    rivals, dropped = [], []
    for rival in RIVALS:
        failed_most = all(
            2 * sum(1 for a in rows if a.get(rival) is None or a[rival][1] is not None) > len(rows)
            for rows in by_env.values()
        )
        (dropped if failed_most else rivals).append(rival)
    out = []
    for env in sorted(by_env):
        rows = []
        for a in by_env[env]:
            if any(a.get(x) is None for x in (arm, *rivals)):
                continue
            rows.append([regret(a[arm]) - regret(a[r]) for r in rivals])
        out.append(np.array(rows, dtype=float))
    return rivals, out, dropped


def expected(chances: list[float]) -> float:
    """The fixed sequence's expected number confirmed, the verdicts taken as independent."""
    running, total = 1.0, 0.0
    for chance in chances:
        running *= chance
        total += running
    return total


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", type=Path, required=True, help="M3's scored stage")
    parser.add_argument("--orbit-f17", type=Path, required=True, help="the re-run's scores")
    parser.add_argument("--floored", type=Path, required=True, help="round 8's development rows")
    parser.add_argument("--out", type=Path, help="the sizes, predictives and order, as JSON")
    args = parser.parse_args()

    records = Records(args.stage, args.orbit_f17)
    with args.floored.open("rb") as held:
        rows = pickle.load(held)
    design = json.loads((args.stage / "design.json").read_text(encoding="utf-8"))
    families: dict[str, dict] = {}
    verdicts: dict[tuple[int, int], dict] = {}
    for family in FAMILIES:
        e = len(design["families"][str(family)]["scored environments"])
        k0_worlds = records.worlds(family, "0")
        tested_worlds = records.worlds(family, "tested")
        matched = with_floored(tested_worlds, family, rows)
        k0 = tie_rule(k0_worlds, K0)
        k0["past the cap"] = k0["needed"] > CAP
        floored = tie_rule(tested_worlds, TESTED)
        recommended = ROUND_8.get(family, FLOOR)
        sized = min(CAP, max(recommended, floored["needed"]))
        declared = max([sized, FLOOR] + ([] if k0["past the cap"] else [k0["needed"]]))
        each = math.ceil(declared / e)
        families[str(family)] = {
            "environments": e,
            "declared": each * e,
            "worlds an environment": each,
            "tested worlds matched to round 8's rows": matched,
            "basis": {
                "0": k0,
                "tested": floored | {"round 8": recommended, "sized": sized, "past the cap": False},
            },
        }
        tested_rung = 1 if family == 7 else 4
        for rung, ws, arm in (("0", k0_worlds, K0), ("tested", tested_worlds, TESTED)):
            rivals, held_strata, dropped = strata(ws, arm)
            each_chance, together = predictive(
                held_strata, each, margin=MARGIN, resamples=RESAMPLES, seed=SEED
            )
            keep = [j for j, c in enumerate(each_chance) if c >= CLAIMED]
            if len(keep) == len(each_chance):
                claimed = together
            elif keep:
                claimed = predictive(
                    [s[:, keep] for s in held_strata],
                    each,
                    margin=MARGIN,
                    resamples=RESAMPLES,
                    seed=SEED,
                )[1]
            else:
                claimed = 0.0
            verdicts[(family, 0 if rung == "0" else tested_rung)] = {
                "arm": arm,
                "worlds an environment": each,
                "past the cap": families[str(family)]["basis"][rung]["past the cap"],
                "together": together,
                "together, claimed comparisons": claimed,
                "rivals": dict(zip(rivals, each_chance, strict=True)),
                "predicted to fail": [
                    r for r, c in zip(rivals, each_chance, strict=True) if c < CLAIMED
                ],
                "not compared": dropped,
            }
    claimed = {
        v: held
        for v, held in verdicts.items()
        if not held["past the cap"] and len(held["predicted to fail"]) < len(held["rivals"])
    }
    order = sorted(claimed, key=lambda v: (-claimed[v]["together, claimed comparisons"], *v))
    chances = [claimed[v]["together, claimed comparisons"] for v in order]
    out = {
        "inputs": {
            "orbit f17 sha256": digest(args.orbit_f17),
            "floored rows sha256": digest(args.floored),
        },
        "rule": {
            "margin": MARGIN,
            "floor": FLOOR,
            "cap": CAP,
            "z": Z,
            "round 8": ROUND_8,
        },
        "families": families,
        "verdicts": {f"{f} {k}": held for (f, k), held in verdicts.items()},
        "order": [list(v) for v in order],
        "predicted": [
            {"family": f, "rung": k, "rival": rival}
            for (f, k), held in verdicts.items()
            for rival in held["predicted to fail"]
        ],
        "expected confirmed": expected(chances),
        "predictives' sum": sum(chances),
    }
    print("\n".join(records.notes))
    print(
        "| family | E | k = 0 N | tested N | round 8 | sized | declared | worlds an environment |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for family, held in families.items():
        k0, tested = held["basis"]["0"], held["basis"]["tested"]
        past = " (past the cap)" if k0["past the cap"] else ""
        print(
            f"| {family} | {held['environments']} | {k0['needed']}{past} | {tested['needed']} | "
            f"{tested['round 8']} | {tested['sized']} | {held['declared']} | "
            f"{held['worlds an environment']} |"
        )
    for (family, rung), held in verdicts.items():
        print(
            f"f{family} k{rung} {held['arm']}: together {held['together']:.4f}, claimed "
            f"{held['together, claimed comparisons']:.4f}"
            + (", past the cap" if held["past the cap"] else "")
            + (
                f", predicted to fail {held['predicted to fail']}"
                if held["predicted to fail"]
                else ""
            )
            + (f", not compared {held['not compared']}" if held["not compared"] else "")
        )
    print(f"order: {out['order']}")
    print(
        f"verdicts claimed: {len(order)}; expected confirmed {out['expected confirmed']:.2f} "
        f"(the predictives' sum {out["predictives' sum"]:.2f})"
    )
    if args.out:
        args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
