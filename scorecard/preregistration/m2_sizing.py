"""The M2 run's worlds a family, sized from its pilot's scores by the rule `m2.md` states.

    python scorecard/preregistration/m2_sizing.py [--default NAME] [--out FILE] ROOT [ROOT ...]

Each `ROOT` is a pilot run's directory, holding `run.json` and `scores.jsonl`, or a directory of
such runs. A run is read by what it declares: a run with the `dlm` arm is a family's main run, a
run with `dlm-two-stage` alone holds the two-stage records at the main run's tested rung, and every
other run (Robyn's, a tool run apart) is not sized. A family's main runs pool; family 7's two runs
pool on their untested rung and on their one tested rung, whatever that rung's k.

**The default.** `claim`, the default, reads `dlm plug-in` at k = 0 and `dlm two-stage` at the
tested rung, the two-stage record paired with the main record by environment, seed and rung. Any
other name reads that variant at every rung.

**The comparisons.** The default less each rival at each rung, regret per euro as scored (a failed
plan at the status quo's regret): the status quo, the equal split and each tool the main run
declares, Robyn's runs apart. A world enters only where it holds every rung.

**The rule.** Each comparison's differences are centred at 0 and raised by the ratio of their SD's
80 % upper confidence limit to the SD, the larger of the chi-square limit and the
kurtosis-adjusted one (`ln s^2` with standard error `((kappa - (n - 3) / n) / (n - 1))^(1/2)`,
`kappa` the sample kurtosis). Worlds are resampled within each environment, the same worlds for
every comparison. A comparison passes where Hall's (1992) transform of its one-sided t,
`g(T) = T + a T^2 + a^2 T^3 / 3 + a / 2`, `a` the skewness of `margin - difference` over
`3 n^(1/2)`, is at least `z_0.975`. `needed` is the fewest worlds, in tens, at which every
comparison passes together with chance at least `POWER`. The run declares
`max(needed, FLOOR)` worlds, rounded up to whole worlds an environment; a family whose `needed`
passes `CAP` is exploratory and declares `FLOOR`. A family whose blocks hold fewer worlds an
environment than that (`HELD`) declares what they hold and is exploratory.

**The prediction.** At the worlds the run declares (at most `CAP`), the same bootstrap with the
differences not centred, each at its pilot mean, gives each comparison's chance to be within the
margin and the family's joint chance. A comparison whose chance is below a half is a predicted
failure.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import chi2, norm

if TYPE_CHECKING:
    from collections.abc import Iterator

MARGIN = 0.05
"""delta, regret per euro."""
POWER = 0.8
"""Each family's joint chance at a true tie."""
CONFIDENCE = 0.8
"""The SDs' upper confidence limit."""
ALPHA = 0.025
Z_BOUND = float(norm.isf(ALPHA))
RESAMPLES = 20_000
SEED = 1982
TENS = 10
FLOOR = 240
CAP = 520
HELD = {15: 100}
"""Worlds an environment a family's blocks hold, where fewer than the floor: siMMMulator's
files of family 15, which the bench reads and does not draw."""
MOST = 6000
BUILT_IN = ("status quo", "equal split")
CLAIM = "claim"
PLUG_IN = "dlm plug-in"
TWO_STAGE = "dlm two-stage"
UNTESTED, TESTED = "k0", "k>0"

Row = dict[str, float]


def runs(paths: list[Path]) -> list[Path]:
    """Every run directory under `paths`: each path itself where it holds `run.json`, else its
    children that do."""
    found: list[Path] = []
    for path in paths:
        if (path / "run.json").is_file():
            found.append(path)
        else:
            found += sorted(p.parent for p in path.glob("*/run.json"))
    return found


def _rung(k: int) -> str:
    return UNTESTED if k == 0 else TESTED


def _read(path: Path) -> Iterator[dict[str, Any]]:
    for line in (path / "scores.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def load(
    paths: list[Path],
) -> dict[int, tuple[dict[tuple[str, int], dict[str, Row]], list[str]]]:
    """Each family's worlds, `(environment, seed) -> rung -> arm -> regret`, and its tools."""
    main: dict[int, dict[tuple[str, int], dict[str, Row]]] = defaultdict(dict)
    tools: dict[int, list[str]] = {}
    staged: dict[tuple[int, str, int, str], float] = {}
    for path in runs(paths):
        declared = json.loads((path / "run.json").read_text(encoding="utf-8"))
        family, arms = declared["family"], declared["arms"]
        if "dlm" in arms:
            external = [t for t in declared.get("external", []) if t != "robyn"]
            if tools.setdefault(family, external) != external:
                msg = f"family {family}'s main runs declare different tools"
                raise ValueError(msg)
            for r in _read(path):
                regrets = {a: s["regret"] for a, s in r["arms"].items()}
                main[family].setdefault((r["environment"], r["seed"]), {})[_rung(r["k"])] = regrets
        elif arms == ["dlm-two-stage"]:
            for r in _read(path):
                key = (family, r["environment"], r["seed"], _rung(r["k"]))
                staged[key] = r["arms"][TWO_STAGE]["regret"]
    for (family, environment, seed, rung), regret in staged.items():
        world = main.get(family, {}).get((environment, seed), {})
        if rung in world:
            world[rung][TWO_STAGE] = regret
    return {f: (main[f], tools[f]) for f in sorted(main)}


def matrix(
    worlds: dict[tuple[str, int], dict[str, Row]], default: str, rivals: list[str]
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    """The differences, a row a world in (environment, seed) order, a column a (rival, rung)."""
    rungs = (UNTESTED, TESTED)
    columns = [(rival, rung) for rung in rungs for rival in rivals]

    def own(rung: str) -> str:
        if default != CLAIM:
            return default
        return PLUG_IN if rung == UNTESTED else TWO_STAGE

    keys = sorted(
        key
        for key, world in worlds.items()
        if all(rung in world and own(rung) in world[rung] for rung in rungs)
    )
    if not keys:
        msg = f"no world holds both rungs and the default {default!r}"
        raise ValueError(msg)
    m = np.array(
        [[worlds[k][g][own(g)] - worlds[k][g][rival] for rival, g in columns] for k in keys]
    )
    return m, np.array([k[0] for k in keys]), columns


def kurtosis(x: np.ndarray) -> float:
    centred = x - x.mean()
    m2 = float(np.mean(centred**2))
    return float(np.mean(centred**4)) / m2**2 if m2 > 0.0 else 3.0


def chi2_ratio(n: int) -> float:
    return math.sqrt((n - 1) / float(chi2.ppf(1.0 - CONFIDENCE, n - 1)))


def kurtosis_ratio(x: np.ndarray) -> float:
    n = x.size
    se = math.sqrt(max(kurtosis(x) - (n - 3) / n, 0.0) / (n - 1))
    return math.exp(float(norm.isf(1.0 - CONFIDENCE)) * se / 2.0)


def passes(
    strata: list[np.ndarray], scales: np.ndarray, *, centred: bool
) -> Iterator[tuple[int, np.ndarray]]:
    """Each count of worlds an environment, from 1, with whether each resample passes each
    comparison by Hall's reader; every count extends the draws of the count before it."""
    centre = np.mean([s.mean(axis=0) for s in strata], axis=0)
    strata = [scales * (s - centre) + (0.0 if centred else centre) for s in strata]
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(SEED).spawn(len(strata))]
    total = np.zeros((RESAMPLES, strata[0].shape[1]))
    squares = np.zeros_like(total)
    cubes = np.zeros_like(total)
    count = 0
    while True:
        count += 1
        for stratum, rng in zip(strata, streams, strict=True):
            drawn = stratum[rng.integers(0, stratum.shape[0], RESAMPLES)]
            total += drawn
            squares += drawn * drawn
            cubes += drawn * drawn * drawn
        n = count * len(strata)
        if n < 2:
            continue
        mean = total / n
        m2 = np.maximum(squares / n - mean * mean, 0.0)
        m3 = cubes / n - 3.0 * mean * squares / n + 2.0 * mean**3
        sigma = np.sqrt(m2)
        gap = MARGIN - mean
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(sigma > 0.0, gap / (sigma / math.sqrt(n)), np.inf * np.sign(gap))
            a = np.where(sigma > 0.0, -m3 / sigma**3, 0.0) / (3.0 * math.sqrt(n))
            g = np.where(np.isfinite(t), t + a * t**2 + a * a * t**3 / 3.0 + a / 2.0, t)
        yield count, g >= Z_BOUND


def needed(strata: list[np.ndarray], scales: np.ndarray) -> tuple[int, list[float], float] | None:
    """The fewest worlds, in tens, at which every comparison passes with joint chance `POWER` at a
    true tie, with each comparison's chance and the joint one; None past `MOST`."""
    e = len(strata)
    target = TENS
    for count, passed in passes(strata, scales, centred=True):
        while math.ceil(target / e) == count:
            joint = float(passed.all(axis=1).mean())
            if joint >= POWER:
                return target, passed.mean(axis=0).tolist(), joint
            target += TENS
        if target > MOST:
            return None
    raise AssertionError  # passes never ends


def predictive(
    strata: list[np.ndarray], scales: np.ndarray, count: int
) -> tuple[list[float], float]:
    for c, passed in passes(strata, scales, centred=False):
        if c == count:
            return passed.mean(axis=0).tolist(), float(passed.all(axis=1).mean())
    raise AssertionError  # passes never ends


def size(
    family: int, worlds: dict[tuple[str, int], dict[str, Row]], tools: list[str], default: str
) -> dict[str, Any]:
    rivals = [*BUILT_IN, *tools]
    m, places, columns = matrix(worlds, default, rivals)
    environments = sorted(set(places))
    strata = [m[places == e] for e in environments]
    e, n = len(strata), m.shape[0]
    scales = np.array([max(chi2_ratio(n), kurtosis_ratio(m[:, j])) for j in range(m.shape[1])])
    found = needed(strata, scales)
    need = None if found is None else found[0]
    final = None if need is None else e * math.ceil(max(need, FLOOR) / e)
    exploratory = need is None or need > CAP
    declared = e * math.ceil(FLOOR / e) if final is None or exploratory else final
    if family in HELD and declared > e * HELD[family]:
        declared, exploratory = e * HELD[family], True
    each_declared, joint_declared = predictive(strata, scales, declared // e)
    at_final = None if final is None else predictive(strata, scales, final // e)
    comparisons = []
    for j, (rival, rung) in enumerate(columns):
        x = m[:, j]
        sd = float(x.std(ddof=1))
        centred = x - x.mean()
        skew = float(np.mean(centred**3) / np.mean(centred**2) ** 1.5) if sd > 0 else 0.0
        comparisons.append(
            {
                "rival": rival,
                "rung": rung,
                "mean": float(x.mean()),
                "sd": sd,
                "skew": skew,
                "excess kurtosis": kurtosis(x) - 3.0,
                "sd ucl80 chi2": sd * chi2_ratio(n),
                "sd ucl80 kurtosis": sd * kurtosis_ratio(x),
                "scale": float(scales[j]),
                "chance at a tie": None if found is None else found[1][j],
                "predictive at declared": each_declared[j],
                "predicted failure": each_declared[j] < 0.5,
            }
        )
    return {
        "family": family,
        "default": default,
        "environments": environments,
        "pilot worlds": n,
        "needed": need,
        "final": final,
        "exploratory": exploratory,
        "declared": declared,
        "worlds an environment": declared // e,
        "joint at needed": None if found is None else found[2],
        "predictive joint at final": None if at_final is None else at_final[1],
        "predictive joint at declared": joint_declared,
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("roots", type=Path, nargs="+")
    parser.add_argument("--default", default=CLAIM, help=f"{CLAIM!r} or one variant's name")
    parser.add_argument("--out", type=Path, help="write every family's sizing here as JSON")
    args = parser.parse_args()
    sized = [size(f, w, t, args.default) for f, (w, t) in load(args.roots).items()]
    print("family  E  pilot  needed  final  declared  exploratory  joint  predictive")
    for s in sized:
        joint = s["joint at needed"]
        print(
            f"{s['family']:>6}  {len(s['environments'])}  {s['pilot worlds']:>5}  "
            f"{s['needed'] if s['needed'] is not None else '>' + str(MOST):>6}  "
            f"{s['final'] if s['final'] is not None else '-':>5}  {s['declared']:>8}  "
            f"{s['exploratory']!s:>11}  {'-' if joint is None else f'{joint:.3f}':>5}  "
            f"{s['predictive joint at declared']:.3f}"
        )
        for c in s["comparisons"]:
            if c["predicted failure"]:
                print(
                    f"        predicted failure: {c['rival']} {c['rung']} "
                    f"mean {c['mean']:+.4f} chance {c['predictive at declared']:.3f}"
                )
    if args.out is not None:
        args.out.write_text(json.dumps(sized, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
