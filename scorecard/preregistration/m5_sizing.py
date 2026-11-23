"""The M5 run's worlds a family, the order of its verdicts and their predictives, sized from M4's
scored records by the rule `m5.md` states (Sizing): each comparison sized from its own sizing
mean and SD, with the sizing mean's own error carried, so that a comparison the claim's arm wins by
far does not set the worlds.

    PYTHONPATH=<bench>/src:. python scorecard/preregistration/m5_sizing.py --stage M4_STAGE \\
        --m3-stage M3_STAGE --floored ROWS.pkl [--out FILE.json]

`--stage` is M4's scored stage: its `design.json`, which names each family's scored environments,
and each root's `scores.jsonl`. `--m3-stage` is M3's scored stage, read for family 15 alone: its
`e0-f15` (k = 0) and `et-f15` (rung 4) roots, M2's 100 recorded worlds with M2's rival records.
`--floored` is round 8's development rows, read for family 15 alone: a pickled list of {family,
env, seed, index, k, regret}, `regret` holding `anchored floored`'s (`floor:sp:0.9`) and
`anchored held`'s (`T0`), `T0` equal to M3's stored `anchored held` on every matched world.
The k = 0 claim's arm is `margin held` and the tested rung's `anchored floored`, both M4's.

**The sizing records.** Every complete case of M4's scored roots: `f<n>` at k = 0 and at the
tested rung joined with `f<n>-ts`; family 7 pooled over `f7a` (k = 0 and 1) and `f7b` (k = 0 and
2) with their `-ts` roots. Family 15: M3's `e0-f15` and `et-f15`, `anchored floored` from round
8's rows.

**A comparison** is the claim's arm less one rival (the status quo, the equal split, Meridian,
PyMC-Marketing and Orbit at their recommended settings) on its complete cases, the family's
environments pooled: n worlds, mean m and SD s (ddof 1).

**The rule (taken).** A comparison's calibrated predictive at N worlds, by the normal
approximation of the claim's one-look reader, a flat prior on the true mean carrying the sizing
mean's own error:

    P(N) = Phi((margin - m - z s / sqrt(N)) / (s sqrt(1 / n + 1 / N))),   z = z(1 - 0.025).

It needs the fewest N, in tens, with P(N) at least the power, q = z(power):

    N = ((z^2 - q^2) / (a z - q sqrt(a^2 + (z^2 - q^2) / n)))^2,   a = (margin - m) / s,

which exists where a > q / sqrt(n), the predictive's limit Phi(a sqrt(n)) above the power. A
comparison whose mean is at or past the margin (m >= margin) is not sized: its limit is at most
0.5, and the predictive at the declared worlds (below) names it a predicted failure. The rung needs
the largest N over its sized comparisons; a rung where a sized comparison needs more than the cap,
or none exists, is exploratory and does not size its family. With an exact sizing mean (n to
infinity) N is ((z + q) s / (margin - m))^2, and with m = 0 it is M3's and M4's tie rule.

**The reader's check.** The claim reads a comparison with Hall's reader, which corrects for the
differences' skew; the normal form above does not, and on a heavy-tailed comparison its N is
optimistic. So the rung's N, at least the floor, is checked with the predictive below on each sized
comparison alone and raised by ten worlds while one of them is below the power; a rung raised past
the cap is exploratory.

**The tie rule (M4's, printed beside it).** N = (2.8016 s / margin)^2 in tens, the largest over
the rung's comparisons; past the cap, exploratory.

**The family** declares the largest of its claimed rungs' worlds, at least the floor, split evenly
over its environments and rounded up to whole worlds an environment.

**The predictive** of each verdict at the family's worlds an environment:
`scorecard.development.predictive`, the two-stage bootstrap, one stratum an environment, a column
a rival, nan where the world is not one of that comparison's complete cases. A rival that failed on
more than half of the worlds of every environment is not compared. A comparison below `CLAIMED` is
a predicted failure; the verdict's predictive is then read again over its claimed comparisons.

**The order.** Each claimed verdict whose rung is within the cap, by its predictive over its claimed
comparisons, unrounded, highest first; ties by family, then rung. Family 7's tested rung is named
1. The expected number confirmed is the fixed sequence's: the sum over the order of the running
product of the predictives, the verdicts taken as independent.

The margin, the power, the floor, the cap, the resamples, the seed, the families and the rungs are
`m5.json`'s, beside this file.
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
from scipy.stats import norm

from scorecard.development import CLAIMED, predictive

HERE = Path(__file__).resolve().parent
DOC = json.loads((HERE / "m5.json").read_text(encoding="utf-8"))
MARGIN = float(DOC["margin"])
SIZING = DOC["sizing"]
FLOOR = int(SIZING["floor"])
CAP = int(SIZING["cap"])
POWER = float(SIZING["power"])
RESAMPLES = int(SIZING["predictive"]["resamples"])
SEED = int(SIZING["predictive"]["seed"])
Z = float(norm.isf(float(DOC["claim"]["alpha"])))
"""z(0.975): the one-sided level of each verdict in the fixed sequence."""
Q = float(norm.ppf(POWER))
TIE_Z = 2.8016
"""M3's and M4's z(0.975) + z(0.8), rounded to four places."""
FAMILIES = tuple(int(f) for f in DOC["sizing"]["families"])
TESTED_RUN = bool(DOC["sizing"]["tested rung run"])
RIVALS = ("status quo", "equal split", "meridian", "pymc-marketing", "orbit")
K0, TESTED, HELD = "margin held", "anchored floored", "anchored held"
RULE, T0 = "floor:sp:0.9", "T0"
KEEP = {*RIVALS, K0, TESTED, HELD}

World = tuple[str, int, int, int]
"""(environment, seed, index, k)"""
Arms = dict[str, tuple[float | None, str | None]]
"""Each arm's (regret, failure)."""


def sources(family: int) -> dict[str, list[tuple[str, int]]]:
    """The (root, k) pairs of M4's stage that hold a family's worlds at each rung."""
    if family == 7:
        return {
            "0": [("f7a", 0), ("f7b", 0)],
            "tested": [("f7a", 1), ("f7a-ts", 1), ("f7b", 2), ("f7b-ts", 2)],
        }
    if family == 15:
        return {"0": [("e0-f15", 0)], "tested": [("et-f15", 4)]}
    return {"0": [(f"f{family}", 0)], "tested": [(f"f{family}", 4), (f"f{family}-ts", 4)]}


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
    """The sizing records, root by root as they are read."""

    def __init__(self, stage: Path, m3_stage: Path) -> None:
        self.stage, self.m3_stage = stage, m3_stage
        self.roots: dict[str, dict[World, Arms]] = {}
        self.notes: list[str] = []

    def root(self, name: str) -> dict[World, Arms]:
        if name not in self.roots:
            base = self.m3_stage if name.endswith("-f15") else self.stage
            self.roots[name] = read_root(base / name / "scores.jsonl")
        return self.roots[name]

    def worlds(self, family: int, rung: str) -> dict[World, Arms]:
        """The family's worlds at the rung, each with its arms merged over the roots that hold it.

        Raises:
            ValueError: where two roots record one arm of one world apart.
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
        self.notes.append(f"f{family} rung {rung}: {len(merged)} worlds")
        return dict(merged)


def with_floored(ws: dict[World, Arms], family: int, rows: list[dict]) -> int:
    """Round 8's rule added to family 15's tested worlds as `anchored floored`.

    Raises:
        ValueError: where round 8's `T0` is not M3's stored `anchored held`, or a world of M3's
            has no row of round 8's.
    """
    by_key = {(r["env"], r["seed"], r["index"], r["k"]): r for r in rows if r["family"] == family}
    for key, arms in ws.items():
        row = by_key.get(key)
        if row is None:
            msg = f"f{family}: round 8 holds no row of {key}"
            raise ValueError(msg)
        held = arms.get(HELD)
        if held is None or held[1] is not None or held[0] != row["regret"][T0]:
            msg = f"f{family}: round 8's T0 is not M3's {HELD} on {key}"
            raise ValueError(msg)
        arms[TESTED] = (row["regret"][RULE], None)
    return len(ws)


def tens(n: float) -> int:
    return 10 * math.ceil(n / 10)


def tie_needed(sd: float) -> int:
    return tens((TIE_Z * sd / MARGIN) ** 2)


def calibrated(mean: float, sd: float, pairs: int, worlds: float) -> float:
    """A comparison's calibrated predictive at `worlds` worlds (the rule's P(N))."""
    if sd == 0.0:
        return 1.0 if mean < MARGIN else 0.0
    spread = sd * math.sqrt(1.0 / pairs + 1.0 / worlds)
    return float(norm.cdf((MARGIN - mean - Z * sd / math.sqrt(worlds)) / spread))


def predictive_needed(mean: float, sd: float, pairs: int) -> float | None:
    """The worlds (unrounded) at which the calibrated predictive reaches the power; None where it
    never does, or where the mean is at or past the margin."""
    if mean >= MARGIN:
        return None
    if sd == 0.0:
        return 2.0
    a = (MARGIN - mean) / sd
    if a <= Q / math.sqrt(pairs):
        return None
    width = Z * Z - Q * Q
    return (width / (a * Z - Q * math.sqrt(a * a + width / pairs))) ** 2


def comparisons(ws: dict[World, Arms], arm: str) -> list[dict]:
    """The arm less each rival on their complete cases: pairs, mean, sd, the tie rule's N and the
    rule's N."""
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
            mean, sd = statistics.fmean(diffs), statistics.stdev(diffs)
            raw = predictive_needed(mean, sd, len(diffs))
            row |= {
                "mean": mean,
                "sd": sd,
                "tie rule": tie_needed(sd),
                "sized": mean < MARGIN,
                "rule": None if raw is None else max(tens(raw), 10),
                "rule, unrounded": raw,
            }
        out.append(row)
    return out


def rung_size(rows: list[dict]) -> dict:
    """The rung's worlds under the tie rule and under the rule, each with its binding rival."""
    timed = [r for r in rows if "tie rule" in r]
    tie = max(timed, key=lambda r: r["tie rule"])
    sized = [r for r in timed if r["sized"]]
    if not sized:
        return {
            "tie rule": tie["tie rule"],
            "tie rule binding": tie["rival"],
            "rule": None,
            "rule binding": None,
            "past the cap": True,
            "comparisons": rows,
        }
    unreachable = [r for r in sized if r["rule"] is None]
    worst = unreachable[0] if unreachable else max(sized, key=lambda r: r["rule"])
    needed = None if unreachable else worst["rule"]
    return {
        "tie rule": tie["tie rule"],
        "tie rule binding": tie["rival"],
        "rule": needed,
        "rule binding": worst["rival"],
        "past the cap": needed is None or needed > CAP,
        "tie rule past the cap": tie["tie rule"] > CAP,
        "comparisons": rows,
    }


def checked(ws: dict[World, Arms], arm: str, basis: dict, e: int) -> dict:
    """The rule's N checked by the claim's own reader: the bootstrap predictive of each sized
    comparison at N, at least the floor, raised by ten worlds while one is below the power; past
    the cap the rung is exploratory. Hall's reader corrects for the differences' skew, which the
    normal form above leaves out: on a few heavy-tailed comparisons its N is optimistic.

    Returns:
        The rule's N after the check, whether it passes the cap, the raise, and each sized
        comparison's bootstrap predictive at that N.
    """
    sized = [r["rival"] for r in basis["comparisons"] if r.get("sized")]
    rivals, held_strata, _ = strata(ws, arm)
    columns = [j for j, rival in enumerate(rivals) if rival in sized]
    start = max(FLOOR, basis["rule"])
    needed = start
    while True:
        each = math.ceil(needed / e)
        chances = predictive(
            [s[:, columns] for s in held_strata],
            each,
            margin=MARGIN,
            resamples=RESAMPLES,
            seed=SEED,
        )[0]
        if min(chances, default=1.0) >= POWER or needed > CAP:
            break
        needed += 10
    return {
        "rule": needed,
        "rule, before the reader's check": basis["rule"],
        "raised by the reader's check": needed > start,
        "past the cap": needed > CAP,
        "reader's check": {rivals[j]: c for j, c in zip(columns, chances, strict=True)},
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


def environments(stage: Path, m3_stage: Path, family: int) -> int:
    design = json.loads(((m3_stage if family == 15 else stage) / "design.json").read_text())
    return len(design["families"][str(family)]["scored environments"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", type=Path, required=True, help="M4's scored stage")
    parser.add_argument("--m3-stage", type=Path, required=True, help="M3's scored stage (f15)")
    parser.add_argument("--floored", type=Path, required=True, help="round 8's rows (f15)")
    parser.add_argument("--out", type=Path, help="the sizes, predictives and order, as JSON")
    args = parser.parse_args()

    records = Records(args.stage, args.m3_stage)
    with args.floored.open("rb") as held:
        floored_rows = pickle.load(held)
    families: dict[str, dict] = {}
    verdicts: dict[tuple[int, int], dict] = {}
    for family in FAMILIES:
        e = environments(args.stage, args.m3_stage, family)
        rungs = {"0": records.worlds(family, "0")}
        if TESTED_RUN or family == 15:
            rungs["tested"] = records.worlds(family, "tested")
            if family == 15:
                with_floored(rungs["tested"], family, floored_rows)
        basis: dict[str, dict] = {}
        for rung, ws in rungs.items():
            arm = K0 if rung == "0" else TESTED
            basis[rung] = {"arm": arm} | rung_size(comparisons(ws, arm))
            if not basis[rung]["past the cap"]:
                basis[rung] |= checked(ws, arm, basis[rung], e)
        claimed = [b["rule"] for b in basis.values() if not b["past the cap"]]
        declared = max([FLOOR, *claimed])
        each = math.ceil(declared / e)
        tie_claimed = [b["tie rule"] for b in basis.values() if b["tie rule"] <= CAP]
        tie_each = math.ceil(max([FLOOR, *tie_claimed]) / e)
        families[str(family)] = {
            "environments": e,
            "declared": each * e,
            "worlds an environment": each,
            "tie rule: declared": tie_each * e,
            "basis": basis,
        }
        tested_rung = 1 if family == 7 else 4
        for rung, ws in rungs.items():
            arm = basis[rung]["arm"]
            rivals, held_strata, dropped = strata(ws, arm)
            each_chance, together = predictive(
                held_strata, each, margin=MARGIN, resamples=RESAMPLES, seed=SEED
            )
            keep = [j for j, c in enumerate(each_chance) if c >= CLAIMED]
            if len(keep) == len(each_chance):
                claimed_chance = together
            elif keep:
                claimed_chance = predictive(
                    [s[:, keep] for s in held_strata],
                    each,
                    margin=MARGIN,
                    resamples=RESAMPLES,
                    seed=SEED,
                )[1]
            else:
                claimed_chance = 0.0
            verdicts[(family, 0 if rung == "0" else tested_rung)] = {
                "arm": arm,
                "worlds an environment": each,
                "past the cap": basis[rung]["past the cap"],
                "together": together,
                "together, claimed comparisons": claimed_chance,
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
            "floored rows sha256": digest(args.floored),
            "k0 arm": K0,
            "tested arm": TESTED,
        },
        "rule": {
            "margin": MARGIN,
            "power": POWER,
            "z": Z,
            "q": Q,
            "floor": FLOOR,
            "cap": CAP,
            "tie rule z": TIE_Z,
        },
        "families": families,
        "verdicts": {f"{f} {k}": held for (f, k), held in verdicts.items()},
        "order": [list(v) for v in order],
        "exploratory": [list(v) for v, held in verdicts.items() if held["past the cap"]],
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
        "| family | E | k = 0: tie rule (binding) | k = 0: rule (binding) | tested: tie rule | "
        "tested: rule | declared | worlds an environment | tie rule: declared |"
    )
    print("|---|---|---|---|---|---|---|---|---|")

    def cell(b: dict, key: str) -> str:
        value = b[key]
        past = (key == "rule" and b["past the cap"]) or (key == "tie rule" and value > CAP)
        shown = "none" if value is None else f"{value}"
        if key == "rule" and "rule, before the reader's check" in b:
            shown = f"{b["rule, before the reader's check"]} -> {shown}"
        binding = b[f"{key} binding"]
        return f"{shown}{' (past the cap)' if past else ''} ({binding})"

    for family, held in families.items():
        k0 = held["basis"]["0"]
        tested = held["basis"].get("tested")
        print(
            f"| {family} | {held['environments']} | {cell(k0, 'tie rule')} | {cell(k0, 'rule')} | "
            + (f"{cell(tested, 'tie rule')} | {cell(tested, 'rule')} | " if tested else "- | - | ")
            + f"{held['declared']} | {held['worlds an environment']} | "
            f"{held['tie rule: declared']} |"
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
