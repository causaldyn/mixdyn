"""The M3 run's worlds a family and the order of its verdicts, sized from the development set's
records of the claim's arms by the rule `m3.md` states.

    python scorecard/preregistration/m3_sizing.py --stored ROOT ... [--pilot ROOT ...] \\
        [--fresh ROOT ...] --k0 VARIANT --tested VARIANT --geo VARIANT [--out FILE.json]

The roots are `scorecard.development select`'s: M2's stored roots, the roots that plan the claim's
arms on M2's worlds beside them, and the fresh roots of families 9, 12 and 13. The claim's arms are
the variants the claim reads at k = 0 and at the tested rung on the national families, and on the
geo families at both rungs.

**N.** For each family and rung, each comparison of the claim's arm with a rival (the status quo,
the equal split, Meridian, PyMC-Marketing and Orbit at their recommended settings) needs
N = (2.8016 sd / delta)^2 worlds, in tens, sd the paired difference's (the arm less the rival,
ddof 1) over the family's fit-split worlds at the rung, its environments pooled, complete cases
only. Families 9, 12 and 13 hold no fit split: their sd is read on every pilot world. A family
declares the largest N over its rungs and rivals, at least the floor, split evenly over its
environments and rounded up to whole worlds an environment. A family and rung whose N passes the
cap is exploratory, and a family whose largest N passes it declares the floor.

**The predictive** of each verdict, a family at a rung, at the worlds an environment above:
`scorecard.development.check_strata` and `predictive`, as `select` calls them for its winner, on
the check split (every pilot world of families 9, 12 and 13), against the rivals `select` reads
there. A comparison below `CLAIMED` is a predicted failure and is not claimed; the verdict's
predictive is then read again over its claimed comparisons alone, on the same draws.

**The order.** Each verdict that claims a comparison and whose N is within the cap, by the
predictive over its claimed comparisons, unrounded, highest first; ties by family, then rung.
Family 7's tested rung is named 1, as its pool's first run names it. The expected number confirmed
is the fixed sequence's: the sum over the order of the running product of the predictives, the
verdicts taken as independent.

The margin, the floor, the cap, the resamples and the seed are `m3.json`'s, beside this file.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scorecard.development import (
    CHECK,
    CLAIMED,
    FIT,
    FRESH,
    GEO,
    NATIONAL,
    Candidate,
    Development,
    check_strata,
    development,
    fresh,
    fresh_rivals,
    joined,
    predictive,
    score,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from causaldyn_bench.scorecard.scoring import WorldRecord

    from scorecard.development import Place

HERE = Path(__file__).resolve().parent
DOC = json.loads((HERE / "m3.json").read_text(encoding="utf-8"))
MARGIN: float = DOC["margin"]
FLOOR: int = DOC["sizing"]["floor"]
CAP: int = DOC["sizing"]["cap"]
RESAMPLES: int = DOC["sizing"]["predictive"]["resamples"]
SEED: int = DOC["sizing"]["predictive"]["seed"]
Z = 2.8016
"""z at one-sided 0.025 plus z at 80 % power."""
RIVALS = ("status quo", "equal split", "meridian", "pymc-marketing", "orbit")
"""The rivals the sizing reads: the built-in arms and each tool at its recommended settings."""
TESTED = {7: 1}
"""The tested rung's name where it is not 4: family 7's pool names it by its first run's, 1."""
RUNGS = ("0", "tested")


def needed(sd: float, margin: float = MARGIN) -> int:
    """The worlds a comparison of paired SD `sd` needs, in tens."""
    return 10 * math.ceil((Z * sd / margin) ** 2 / 10)


def _rung(k: int) -> str:
    return "0" if k == 0 else "tested"


def _difference(record: WorldRecord, arm: str, rival: str) -> float | None:
    """`arm`'s regret less `rival`'s on one world, None where either plan failed."""
    mine, theirs = record.arms[arm], record.arms[rival]
    if mine.failure is not None or theirs.failure is not None:
        return None
    if mine.regret is None or theirs.regret is None:
        return None
    return mine.regret - theirs.regret


def environments_of(family: int) -> list[str]:
    """The family's environments in the claim, as `m3.json` lists them."""
    return list(DOC["families"][str(family)]["environments"])


def sizes(
    records: Mapping[Place, WorldRecord],
    dev: Development,
    arms: Mapping[int, Mapping[str, str]],
) -> dict[str, dict[str, Any]]:
    """Each family's N, declared worlds and worlds an environment, with each rung's binding
    comparison; `arms` names the claim's arm of each family at each rung."""
    groups: dict[int, Collection[str] | None] = {
        **dict.fromkeys(NATIONAL),
        **GEO,
        **dict.fromkeys(FRESH),
    }
    out: dict[str, dict[str, Any]] = {}
    for family, environments in sorted(groups.items()):
        split = CHECK if family in FRESH else FIT
        basis: dict[str, dict[str, Any]] = {}
        for rung in RUNGS:
            arm = arms[family][rung]
            rows: list[dict[str, Any]] = []
            for rival in RIVALS:
                diffs = [
                    d
                    for r in records.values()
                    if r.family == family
                    and _rung(r.k) == rung
                    and (environments is None or r.environment in environments)
                    and dev.of(r.family, r.environment, r.seed) == split
                    and arm in r.arms
                    and rival in r.arms
                    and (d := _difference(r, arm, rival)) is not None
                ]
                if len(diffs) < 3:
                    msg = f"family {family} at rung {rung}: {len(diffs)} pairs of {arm} and {rival}"
                    raise ValueError(msg)
                sd = statistics.stdev(diffs)
                rows.append(
                    {
                        "rival": rival,
                        "pairs": len(diffs),
                        "mean": statistics.fmean(diffs),
                        "sd": sd,
                        "needed": needed(sd),
                    }
                )
            worst = max(rows, key=lambda row: row["needed"])
            basis[rung] = {
                "arm": arm,
                "needed": worst["needed"],
                "binding": worst["rival"],
                "past the cap": worst["needed"] > CAP,
                "comparisons": rows,
            }
        e = len(environments_of(family))
        n = max(v["needed"] for v in basis.values())
        declared = FLOOR if n > CAP else max(n, FLOOR)
        each = math.ceil(declared / e)
        out[str(family)] = {
            "needed": n,
            "declared": each * e,
            "worlds an environment": each,
            "past the cap": n > CAP,
            "read on": "pilot worlds" if family in FRESH else "fit split",
            "basis": basis,
        }
    return out


def verdicts(
    records: Mapping[Place, WorldRecord],
    dev: Development,
    names: Mapping[str, Mapping[str, str]],
    sized: Mapping[str, Mapping[str, Any]],
    claimed: Mapping[int, tuple[str, ...]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Each verdict's predictive at the sized worlds, and the predicted failures; `names` gives
    the claim's arm of each group (`national`, `geo`) at each rung, `claimed` each fresh family's
    rivals."""
    out: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    held = {f for f, _, _, _ in records if f in FRESH}
    for group, families in (("national", dict.fromkeys(NATIONAL)), ("geo", GEO)):
        for rung, name in names[group].items():
            scored = score(records, dev, Candidate(name, rung, 0, None), families)
            reads: list[tuple[int, Collection[str] | None, list[str]]] = [
                (f, e, sorted({r for g, r in scored.d if g == f})) for f, e in families.items()
            ]
            for family in sorted(f for f, g in FRESH.items() if g == group and f in held):
                rivals = sorted(
                    {
                        a
                        for r in records.values()
                        if r.family == family and _rung(r.k) == rung and name in r.arms
                        for a in r.arms
                        if a in claimed[family]
                    }
                )
                reads.append((family, None, rivals))
            for family, environments, rivals in reads:
                each_n = sized[str(family)]["worlds an environment"]
                strata = check_strata(records, dev, name, family, rung, rivals, environments)
                each, joint = predictive(
                    strata, each_n, margin=MARGIN, resamples=RESAMPLES, seed=SEED
                )
                kept = [i for i, c in enumerate(each) if c >= CLAIMED]
                together: float | None = joint
                if not kept:
                    together = None
                elif len(kept) < len(rivals):
                    _, together = predictive(
                        [s[:, kept] for s in strata],
                        each_n,
                        margin=MARGIN,
                        resamples=RESAMPLES,
                        seed=SEED,
                    )
                sizing = sized[str(family)]
                means = {c["rival"]: c for c in sizing["basis"][rung]["comparisons"]}
                for rival, chance in zip(rivals, each, strict=True):
                    if chance < CLAIMED:
                        failures.append(
                            {
                                "family": family,
                                "rung": 0 if rung == "0" else TESTED.get(family, 4),
                                "rival": rival,
                                "mean": means[rival]["mean"],
                                "pairs": means[rival]["pairs"],
                                "read on": sizing["read on"],
                                "predictive": chance,
                            }
                        )
                out.append(
                    {
                        "family": family,
                        "rung": 0 if rung == "0" else TESTED.get(family, 4),
                        "arm": name,
                        "worlds an environment": each_n,
                        "check worlds": [int(s.shape[0]) for s in strata],
                        "rivals": rivals,
                        "each": list(each),
                        "together": joint,
                        "together claimed": together,
                        "past the cap": sized[str(family)]["basis"][rung]["past the cap"],
                    }
                )
    return out, failures


def expected(chances: list[float]) -> float:
    """The fixed sequence's expected number confirmed, the verdicts taken as independent."""
    running, total = 1.0, 0.0
    for chance in chances:
        running *= chance
        total += running
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stored", type=Path, nargs="+", required=True)
    parser.add_argument("--pilot", type=Path, nargs="*", default=[])
    parser.add_argument("--fresh", type=Path, nargs="*", default=[])
    parser.add_argument("--k0", required=True, help="the claim's arm at k = 0")
    parser.add_argument("--tested", required=True, help="the claim's arm at the tested rung")
    parser.add_argument("--geo", required=True, help="the claim's arm on the geo families")
    parser.add_argument("--out", type=Path, help="the sizes, the verdicts and the order, as JSON")
    args = parser.parse_args()
    dev = development(args.stored)
    records = joined(args.stored, args.pilot)
    new = fresh(args.fresh)
    if {(f, e, s) for f, e, s, _ in new} & dev.splits.keys():
        msg = "a fresh root holds a world of the stored run"
        raise ValueError(msg)
    dev = Development({**dev.splits, **{(f, e, s): CHECK for f, e, s, _ in new}})
    records |= new
    national = {"0": args.k0, "tested": args.tested}
    geo = {"0": args.geo, "tested": args.geo}
    geo_families = {*GEO, *(f for f, g in FRESH.items() if g == "geo")}
    arms = {f: geo if f in geo_families else national for f in (*NATIONAL, *GEO, *FRESH)}
    sized = sizes(records, dev, arms)
    read, failures = verdicts(
        records, dev, {"national": national, "geo": geo}, sized, fresh_rivals(args.fresh)
    )
    entering = sorted(
        (v for v in read if v["together claimed"] is not None),
        key=lambda v: (-v["together claimed"], v["family"], v["rung"]),
    )
    order = [[v["family"], v["rung"]] for v in entering if not v["past the cap"]]
    past = [[v["family"], v["rung"]] for v in entering if v["past the cap"]]
    sequence = [v["together claimed"] for v in entering if not v["past the cap"]]
    held = {
        "rule": {
            "margin": MARGIN,
            "floor": FLOOR,
            "cap": CAP,
            "resamples": RESAMPLES,
            "seed": SEED,
        },
        "arms": {"k0": args.k0, "tested": args.tested, "geo": args.geo},
        "families": sized,
        "verdicts": read,
        "order": order,
        "past the cap": past,
        "predicted": failures,
        "expected": round(expected(sequence), 2),
        "expected, the cap not applied": round(
            expected([v["together claimed"] for v in entering]), 2
        ),
    }
    for family, s in sized.items():
        binding = "; ".join(
            f"rung {r}: {b['arm']} less {b['binding']} {b['needed']}"
            + (" (past the cap)" if b["past the cap"] else "")
            for r, b in s["basis"].items()
        )
        print(
            f"family {family}: N {s['needed']} ({binding}), declares {s['declared']}, "
            f"{s['worlds an environment']} worlds an environment"
        )
    for v in read:
        print(
            f"family {v['family']} rung {v['rung']}, {v['arm']}: together {v['together']:.3f}, "
            f"claimed {v['together claimed']}, "
            + ", ".join(f"{r} {c:.3f}" for r, c in zip(v["rivals"], v["each"], strict=True))
            + ("; past the cap" if v["past the cap"] else "")
        )
    print(f"order: {json.dumps(order)}")
    print(f"past the cap: {json.dumps(past)}")
    print(f"predicted failures: {json.dumps(failures)}")
    print(
        f"expected confirmed: {held['expected']} of {len(order)}; the cap not applied "
        f"{held['expected, the cap not applied']} of {len(entering)}"
    )
    if args.out:
        args.out.write_text(json.dumps(held, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
