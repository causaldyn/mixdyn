"""A scored run's worlds as the next run's development set: their split, the reading of a pilot's
new arms on them against the run's stored records, the rule that selects a candidate arm, and the
predictive of the claim the candidate would make on fresh worlds.

    PYTHONPATH=<bench>/src uv run --no-sync python -m scorecard.development split ROOT [ROOT ...]
    PYTHONPATH=<bench>/src uv run --no-sync python -m scorecard.development select \\
        --stored ROOT [ROOT ...] [--pilot ROOT ...] [--fresh ROOT ...] --candidates FILE \\
        [--worlds FAMILY=N ...] [--resamples B] [--seed S]

**The split** (`split`). Within each family and environment the stored run's scored worlds are
ranked by their seeds, ascending: a world whose rank leaves 2 when divided by 3 is in the `check`
split, and so is every world of a root beside a run named `<run>-robyn` (`ROBYN`), the worlds a
capped tool planned, so that its stored plans are read once, on worlds no candidate was tuned on.
Every other world is in the `fit` split. Candidates are tuned, selected and compared on the fit
split alone; the check split is read once a candidate rule, for the predictive.

**The reading** (`joined`). A pilot root plans new arms on the stored run's worlds. Each of its
worlds is paired with the stored world of the same family, environment, seed and rung, whose
records, the stored roots' arms joined as `claim read` joins them (`scorecard.claim.joined_runs`),
hold every rival's plan: no rival is fitted again. A pair is read only where the two exports hold
the same observation, by its digest (`causaldyn_bench.scorecard.observe.Observation.digest`), and
the built-in arms score alike in both; otherwise the reading is refused.

**The rule** (`select`), declared before the pilot ran. For each candidate and each confirmatory
family at a rung, the candidate less each rival, regret per euro, is averaged over the fit split's
worlds both arms were read on: `d(f, r, rival)`. A fit that failed plays the status quo, as the
bench scores it (`causaldyn_bench.scorecard.scoring`), and is counted, never dropped: a mean over
the worlds both planned would favour a candidate that fails where it would plan badly. A
candidate's score at a rung is the largest `d` over the families and rivals, its worst comparison.
The candidate with the smallest score wins the rung; candidates within `TIE` of it tie, and a tie
goes to the fewest CPU-seconds a fit, then to the fewest settings, then to the name. Before that, at
the tested rung, the S-curve guard (`guard`): the families whose truth is a Hill curve, `GUARDED`,
are not in the score, but a candidate whose mean regret per euro on the fit split of any of them
exceeds `REFERENCE`'s by more than `GUARD` is out. A rung is `0`,
or `tested`, every rung above 0 a family's worlds are read at (family 7 tests at 1 and 2). The geo
families are read by the same rule over their own families and candidates, apart.

**The predictive** (`predictive`). Each comparison the winner would make on fresh worlds is read on
the check split by a bootstrap of two stages, the pilot mean's own error carried: in each
environment the check split's worlds are drawn with replacement, as many as it holds, and from
those as many as the scored run will read there; on each such draw the comparison is read as the
claim reads it, on its complete cases by Hall's reader against the margin. A world where either
arm's fit failed is drawn and left out of the comparison, so the predictive reads as few complete
cases as the claim will, while the rule above counts the candidate's failures against it. Its
predictive is the share of draws that read it within; a family's at a rung is the share that read
every comparison within together. A comparison whose predictive is below `CLAIMED` is listed as
predicted to fail and stays out of the claim (`scorecard.claim`, the per-rung claim).

**Families the stored run never held** (`FRESH`): their pilot roots (`--fresh`) are read for the
winner's predictive alone, each of their worlds in the check split, since no candidate is chosen on
them; a family is read for the winner of its group, the geo candidates' on a geo family. Its
comparisons are the claim's (`fresh_rivals`): the pilot run's own rivals and each tool of a root
beside it, but for a tool the root caps and declares exploratory, as Robyn's root does. Such a
tool's comparison is printed by the claim and left out of it, and its few worlds are no reason to
read the others' predictive on them alone.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from causaldyn_bench.scorecard.gate import ALPHA, hall
from causaldyn_bench.scorecard.observe import read
from causaldyn_bench.scorecard.scoring import BUILT_IN, WorldRecord
from scipy.stats import norm

from scorecard.claim import _records, joined_runs
from scorecard.claim import rivals as run_rivals
from scorecard.composite import compose
from scorecard.layout import Layout
from scorecard.tools import TOOLS

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Mapping, Sequence

FIT = "fit"
CHECK = "check"
MOD = 3
"""Ranks are split by their remainder on division by `MOD`."""
CHECKED = 2
"""The remainder whose ranks are in the check split."""
ROBYN = "-robyn"
"""The suffix of the roots beside a run whose worlds are all in the check split."""
TIE = 0.005
"""Candidates whose scores lie within this of the smallest, regret per euro, tie."""
CLAIMED = 0.5
"""The least predictive at which a comparison enters the claim."""
MARGIN = 0.05
RESAMPLES = 4000
SEED = 20261116
NATIONAL = (0, 2, 3, 5, 7)
"""The confirmatory national families the rule reads."""
GEO = {8: None, 10: ("geos-3", "geos-8")}
"""The geo families the rule reads apart, each with the environments it reads (None: all)."""
FRESH = {9: "geo", 12: "national", 13: "national"}
"""The families the stored run never held, each with the group whose winner its pilot worlds are
read for: every one of their worlds is in the check split."""
GUARDED = (4, 14)
"""The families whose truth is a Hill curve: out of the score, read by the tested rung's guard."""
REFERENCE = "stacked"
"""The arm the guard reads each tested-rung candidate against."""
GUARD = 0.02
"""The most, regret per euro, a tested candidate may lose to `REFERENCE` on a guarded family."""

Place = tuple[int, str, int, int]
"""A world at a rung: its family, environment, seed and rung."""


def split(seeds: Iterable[int], robyn: Collection[int] = ()) -> dict[int, str]:
    """Each seed of one family's environment in the fit or the check split: by its rank among
    `seeds`, ascending, the check split where it leaves `CHECKED` on division by `MOD`, and every
    seed of `robyn`, the worlds a root beside the run named `<run>-robyn` holds.

    Raises:
        ValueError: on a seed held twice, or a seed of `robyn` that `seeds` do not hold.
    """
    ordered = sorted(seeds)
    if len(set(ordered)) != len(ordered) or not set(robyn) <= set(ordered):
        msg = f"a split ranks seeds held once, every one of Robyn's among them: {sorted(robyn)}"
        raise ValueError(msg)
    return {
        seed: CHECK if rank % MOD == CHECKED or seed in robyn else FIT
        for rank, seed in enumerate(ordered)
    }


@dataclass(frozen=True, slots=True)
class Development:
    """The split of every scored world a stored run holds, by family, environment and seed."""

    splits: Mapping[tuple[int, str, int], str]

    def of(self, family: int, environment: str, seed: int) -> str:
        """The split of one stored world.

        Raises:
            ValueError: on a world the stored run did not score.
        """
        held = self.splits.get((family, environment, seed))
        if held is None:
            msg = f"family {family} {environment} world {seed} is not a stored scored world"
            raise ValueError(msg)
        return held


def development(roots: Sequence[Path]) -> Development:
    """The split of the scored worlds `roots` hold: each root's worlds, the roots named
    `<run>-robyn` all in the check split.

    Raises:
        ValueError: on a pilot's root, which no development set holds, and as `split` does.
    """
    seeds: dict[tuple[int, str], set[int]] = defaultdict(set)
    robyn: dict[tuple[int, str], set[int]] = defaultdict(set)
    for root in roots:
        d = Layout(root).read()
        if not d.scored:
            msg = f"{root} holds no scored worlds: a development set is a scored run's"
            raise ValueError(msg)
        for environment in d.environments:
            held = robyn if root.name.endswith(ROBYN) else seeds
            held[d.family, environment].update(d.seeds(environment))
    out = {}
    for (family, environment), held in seeds.items():
        for seed, part in split(held, robyn.get((family, environment), set())).items():
            out[family, environment, seed] = part
    return Development(out)


def stored(roots: Sequence[Path]) -> tuple[dict[Place, WorldRecord], dict[tuple[int, str], Path]]:
    """Every world the stored run's `roots` hold at each rung, each run's records with its joined
    root's and the roots beside it joined as `claim read` joins them, at the world's own rung; and
    the run's root that holds each family's environment, whose exports the pilot's are held to.

    Raises:
        ValueError: as `scorecard.claim.joined_runs` does, and on a world two runs hold.
    """
    runs, _ = joined_runs(roots)
    out: dict[Place, WorldRecord] = {}
    homes: dict[tuple[int, str], Path] = {}
    for root, d, records in runs:
        homes |= {(d.family, e): root for e in d.environments}
        for r in records:
            place = (r.family, r.environment, r.seed, r.k)
            if place in out:
                msg = f"{place} is held by two of the stored runs"
                raise ValueError(msg)
            out[place] = r
    return out, homes


def _digest(path: Path) -> str:
    return read(path).digest()


def joined(stored_roots: Sequence[Path], pilot_roots: Sequence[Path]) -> dict[Place, WorldRecord]:
    """Every stored world, with the arms of the pilot's world of the same family, environment,
    seed and rung beside the stored ones where the pilot holds it, and each composite of a
    world's arms assembled (`scorecard.composite`): with no pilot root, the stored run's own
    arms, its recorded variants among them, are the candidates.

    Raises:
        ValueError: on a pilot world the stored run does not hold, one whose export is not the
            stored one's observation, one whose built-in arms score apart from the stored world's,
            or an arm both hold.
    """
    held, homes = stored(stored_roots)
    out = dict(held)
    for root in pilot_roots:
        layout = Layout(root)
        d = layout.read()
        for r in _records(layout):
            place = (r.family, r.environment, r.seed, r.k)
            theirs = held.get(place)
            if theirs is None or (r.family, r.environment) not in homes:
                msg = f"{root}: {place} is not a world the stored run holds"
                raise ValueError(msg)
            mine = _digest(layout.export(d.family, r.environment, r.k, r.seed))
            home = Layout(homes[r.family, r.environment])
            if mine != _digest(home.export(d.family, r.environment, r.k, r.seed)):
                msg = f"{root}: {place}'s export is not the stored run's observation"
                raise ValueError(msg)
            base = out[place]
            apart = [
                name
                for name in BUILT_IN
                if not math.isclose(r.arms[name].regret, theirs.arms[name].regret, rel_tol=1e-9)
            ]
            new = {name: s for name, s in r.arms.items() if name not in BUILT_IN}
            twice = sorted(set(new) & set(base.arms))
            if apart or twice:
                msg = f"{root}: {place} scores {apart} apart from the stored run, or holds {twice}"
                raise ValueError(msg)
            out[place] = replace(base, arms={**base.arms, **new})
    return {place: compose(r) for place, r in out.items()}


def fresh(roots: Sequence[Path]) -> dict[Place, WorldRecord]:
    """Every world the pilot's `roots` hold of a family the stored run never held, each run joined
    with the roots beside it as `claim read` joins them.

    Raises:
        ValueError: on a family not in `FRESH`, and as `scorecard.claim.joined_runs` does.
    """
    return _fresh(roots)[0]


def fresh_rivals(roots: Sequence[Path]) -> dict[int, tuple[str, ...]]:
    """Each family of the pilot's `roots` with the rivals the claim compares its arm with: each
    run's own (`scorecard.claim.rivals`), then each tool of a root beside it that the run does not
    hold, but for a tool that root caps where it declares its capped comparisons exploratory
    (`scorecard.claim.compare`).

    Raises:
        ValueError: as `fresh` does.
    """
    return _fresh(roots)[1]


def _fresh(
    roots: Sequence[Path],
) -> tuple[dict[Place, WorldRecord], dict[int, tuple[str, ...]]]:
    """`fresh` and `fresh_rivals` off one reading of `roots`."""
    runs, sides = joined_runs(roots)
    out: dict[Place, WorldRecord] = {}
    claimed: dict[int, list[str]] = defaultdict(list)
    for root, d, records in runs:
        if d.family not in FRESH:
            msg = f"{root}: family {d.family} is not one the stored run never held, {sorted(FRESH)}"
            raise ValueError(msg)
        out |= {(r.family, r.environment, r.seed, r.k): r for r in records}
        own = run_rivals(d)
        beside = [
            tool
            for side in sides[root].values()
            for tool in side.external
            if tool not in own and not (side.exploratory_caps and tool in (side.caps or {}))
        ]
        claimed[d.family] += [a for a in (*own, *beside) if a not in claimed[d.family]]
    return out, {family: tuple(names) for family, names in claimed.items()}


def rival(arm: str) -> bool:
    """Whether `arm` is a rival: a built-in arm or a tool's."""
    return arm in BUILT_IN or arm in TOOLS


def _rung(k: int) -> str:
    return "0" if k == 0 else "tested"


def _read(record: WorldRecord, arm: str) -> bool:
    """Whether `arm` was read on the world: it planned, or its fit failed and it played the status
    quo, as the bench scores a failure."""
    return arm in record.arms


def _failed(record: WorldRecord, arm: str) -> bool:
    return record.arms[arm].failure is not None


@dataclass(frozen=True, slots=True)
class Candidate:
    """A candidate arm: the name its records are kept under, the rung it is a candidate at, the
    settings it changes from the arm's defaults, and its CPU-seconds a fit where stated rather
    than measured on the pilot's records."""

    name: str
    rung: str
    settings: int = 0
    cpu: float | None = None


@dataclass(frozen=True, slots=True)
class Scored:
    """A candidate at a rung: `d`, its mean difference less each rival on each family, with the
    worlds read, (family, rival) -> (mean, worlds); its score, the largest mean, and where it
    binds; its CPU-seconds a fit; and on each family's fit split, its failed fits, each played as
    the status quo, and the worlds it was read on."""

    candidate: Candidate
    d: Mapping[tuple[int, str], tuple[float, int]]
    score: float
    binding: tuple[int, str]
    cpu: float
    failed: Mapping[int, tuple[int, int]] = field(default_factory=dict)


def differences(
    records: Iterable[WorldRecord],
    arm: str,
    rival_arm: str,
    keep: Callable[[WorldRecord], bool],
) -> np.ndarray:
    """`arm` less `rival_arm`, regret per euro, on the worlds `keep` keeps where both were read, a
    failed fit playing the status quo."""
    return np.array(
        [
            r.arms[arm].regret - r.arms[rival_arm].regret
            for r in records
            if keep(r) and _read(r, arm) and _read(r, rival_arm)
        ]
    )


def _cpu(records: Iterable[WorldRecord], arm: str) -> float:
    """The mean CPU-seconds of `arm`'s records that state theirs, a fit's and its plan's; nan where
    none does."""
    held = [
        float(s.record["cost"]["cpu_seconds"])
        for r in records
        if (s := r.arms.get(arm)) is not None and "cpu_seconds" in s.record.get("cost", {})
    ]
    return float(np.mean(held)) if held else math.nan


def score(
    records: Mapping[Place, WorldRecord],
    development: Development,
    candidate: Candidate,
    families: Mapping[int, Collection[str] | None],
) -> Scored:
    """The candidate's mean difference less every rival on every family at its rung, over the fit
    split's worlds, a failed fit playing the status quo, and its score.

    Raises:
        ValueError: where the candidate is read on no world of some family.
    """
    d: dict[tuple[int, str], tuple[float, int]] = {}
    failed: dict[int, tuple[int, int]] = {}
    for family, environments in families.items():

        def keep(r: WorldRecord, family: int = family, environments=environments) -> bool:
            return (
                r.family == family
                and _rung(r.k) == candidate.rung
                and (environments is None or r.environment in environments)
                and development.of(r.family, r.environment, r.seed) == FIT
            )

        mine = [r for r in records.values() if keep(r) and candidate.name in r.arms]
        if not mine:
            msg = f"{candidate.name} is read on no fit-split world of family {family}"
            raise ValueError(msg)
        failed[family] = (sum(_failed(r, candidate.name) for r in mine), len(mine))
        rivals = sorted({a for r in mine for a in r.arms if rival(a)})
        for name in rivals:
            x = differences(mine, candidate.name, name, keep)
            if x.size:
                d[family, name] = (float(x.mean()), int(x.size))
    binding = max(d, key=lambda key: (d[key][0], key))
    cpu = candidate.cpu
    if cpu is None:
        cpu = _cpu((r for r in records.values() if r.family in families), candidate.name)
    return Scored(candidate, d, d[binding][0], binding, cpu, failed)


def guard(
    records: Mapping[Place, WorldRecord],
    development: Development,
    name: str,
    *,
    reference: str = REFERENCE,
    families: Collection[int] = GUARDED,
) -> dict[int, tuple[float, int]]:
    """`name` less `reference` on each guarded family at the tested rung, (mean, worlds), regret
    per euro over the fit split's worlds both were read on, a failed fit playing the status quo.

    Raises:
        ValueError: where the two are read together on no fit-split world of some family.
    """
    out: dict[int, tuple[float, int]] = {}
    for family in families:

        def keep(r: WorldRecord, family: int = family) -> bool:
            return (
                r.family == family
                and _rung(r.k) == "tested"
                and development.of(r.family, r.environment, r.seed) == FIT
            )

        x = differences(records.values(), name, reference, keep)
        if not x.size:
            msg = (
                f"{name} and {reference} are read together on no fit-split world of family {family}"
            )
            raise ValueError(msg)
        out[family] = (float(x.mean()), int(x.size))
    return out


def screen(
    records: Mapping[Place, WorldRecord], development: Development, scored: Sequence[Scored]
) -> tuple[list[Scored], list[str]]:
    """The tested rung's candidates the guard keeps, and its lines: each candidate less
    `REFERENCE` on each guarded family, and whether it is out."""
    kept: list[Scored] = []
    lines: list[str] = []
    for s in scored:
        excess = guard(records, development, s.candidate.name)
        out = sorted(f for f, (m, _) in excess.items() if m > GUARD)
        kept += [] if out else [s]
        lines.append(
            f"  guard: {s.candidate.name:28s} "
            + "; ".join(
                f"family {f} less {REFERENCE} {m:+.4f} on {n} worlds"
                for f, (m, n) in sorted(excess.items())
            )
            + (f"; out (above {GUARD:g} on family {', '.join(map(str, out))})" if out else "")
        )
    return kept, lines


def select(scored: Sequence[Scored]) -> tuple[Scored, tuple[Scored, ...]]:
    """The rule's winner among one rung's `scored` candidates, and the candidates tied with the
    smallest score, the winner first: the fewest CPU-seconds a fit, then the fewest settings, then
    the name. A candidate whose CPU is not known ties last.

    Raises:
        ValueError: on no candidate.
    """
    if not scored:
        msg = "the rule selects among one candidate or more"
        raise ValueError(msg)
    least = min(s.score for s in scored)
    tied = sorted(
        (s for s in scored if s.score <= least + TIE),
        key=lambda s: (
            math.isnan(s.cpu),
            s.cpu if not math.isnan(s.cpu) else 0.0,
            s.candidate.settings,
            s.candidate.name,
        ),
    )
    return tied[0], tuple(tied)


def predictive(
    strata: Sequence[np.ndarray],
    worlds: int,
    *,
    margin: float = MARGIN,
    resamples: int = RESAMPLES,
    seed: int = SEED,
) -> tuple[tuple[float, ...], float]:
    """Each comparison's predictive to be read within `margin` on `worlds` worlds an environment,
    and every comparison's together, by the two-stage bootstrap of the check split's differences:
    `strata`, one matrix an environment, a row a world, a column a comparison, nan where the world
    is not one of the comparison's complete cases. Each draw reads each comparison on the draw's
    complete cases, as the claim does, and never within on fewer than two.

    Raises:
        ValueError: on an environment with no world, or comparisons that differ in number.
    """
    if not strata or any(s.ndim != 2 or s.shape[0] < 1 for s in strata):
        msg = "the predictive reads one matrix of worlds by comparisons an environment, each held"
        raise ValueError(msg)
    width = strata[0].shape[1]
    if any(s.shape[1] != width for s in strata):
        msg = "every environment reads the same comparisons"
        raise ValueError(msg)
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(len(strata))]
    z = float(norm.isf(ALPHA))
    within = np.zeros((resamples, width), dtype=bool)
    for b in range(resamples):
        drawn = []
        for stratum, rng in zip(strata, streams, strict=True):
            first = stratum[rng.integers(0, stratum.shape[0], stratum.shape[0])]
            drawn.append(first[rng.integers(0, first.shape[0], worlds)])
        x = margin - np.vstack(drawn)
        for j in range(width):
            within[b, j] = _within(x[~np.isnan(x[:, j]), j], z)
    return tuple(float(c) for c in within.mean(axis=0)), float(within.all(axis=1).mean())


def _within(x: np.ndarray, z: float) -> bool:
    """Whether Hall's one-look reader shows the mean of `x`, the margin less a comparison's
    differences on its complete cases, above 0 at `z`; never on fewer than two cases."""
    n = x.size
    if n < 2:
        return False
    mean, sigma = float(x.mean()), float(x.std())
    if sigma == 0.0:
        return mean > 0.0
    skew = float(((x - mean) ** 3).mean()) / sigma**3
    return hall(mean / (sigma / math.sqrt(n)), n, skew) >= z


def pairing(
    records: Mapping[Place, WorldRecord],
    development: Development,
    arm: str,
    family: int,
    rung: str,
    rivals: Sequence[str],
    environments: Collection[str] | None = None,
) -> dict[str, tuple[float, int]]:
    """What common random numbers give each comparison of `arm` with one of `rivals` on one family
    at one rung: the variance of their paired difference over the fit split's complete cases, less
    than or above the two arms' variances summed over the same worlds, the variance independent
    worlds would give the difference; (ratio, worlds), nan on fewer than two worlds or none
    unpaired."""
    out: dict[str, tuple[float, int]] = {}
    for name in rivals:
        pairs = np.array(
            [
                (r.arms[arm].regret, r.arms[name].regret)
                for r in records.values()
                if r.family == family
                and _rung(r.k) == rung
                and (environments is None or r.environment in environments)
                and development.of(r.family, r.environment, r.seed) == FIT
                and _read(r, arm)
                and _read(r, name)
                and not _failed(r, arm)
                and not _failed(r, name)
            ]
        ).reshape(-1, 2)
        n = pairs.shape[0]
        apart = float(pairs.var(axis=0, ddof=1).sum()) if n >= 2 else 0.0
        paired = float((pairs[:, 0] - pairs[:, 1]).var(ddof=1)) if n >= 2 else math.nan
        out[name] = (paired / apart if apart > 0.0 else math.nan, n)
    return out


def check_strata(
    records: Mapping[Place, WorldRecord],
    development: Development,
    arm: str,
    family: int,
    rung: str,
    rivals: Sequence[str],
    environments: Collection[str] | None = None,
) -> list[np.ndarray]:
    """`arm` less each of `rivals` on the check split of one family at one rung, a matrix an
    environment, a row each world where `arm` and every rival were read: nan where the world is not
    the comparison's complete case, where `arm`'s fit or the rival's failed."""
    by: dict[str, list[list[float]]] = defaultdict(list)
    for r in records.values():
        if (
            r.family == family
            and _rung(r.k) == rung
            and (environments is None or r.environment in environments)
            and development.of(r.family, r.environment, r.seed) == CHECK
            and all(_read(r, a) for a in (arm, *rivals))
        ):
            mine = math.nan if _failed(r, arm) else r.arms[arm].regret
            by[r.environment].append(
                [math.nan if _failed(r, a) else mine - r.arms[a].regret for a in rivals]
            )
    return [np.array(by[e]) for e in sorted(by)]


def _predicted(
    name: str,
    family: int,
    rung: str,
    rivals: Sequence[str],
    strata: Sequence[np.ndarray],
    worlds: int,
    *,
    resamples: int = RESAMPLES,
    seed: int = SEED,
) -> list[str]:
    """The lines `select` prints of `name`'s predictive on one family at one rung: every
    comparison's together, then each rival's, marked where it is below `CLAIMED`."""
    each, joint = predictive(strata, worlds, resamples=resamples, seed=seed)
    return [
        f"  predictive of {name}, family {family} at rung {rung}, "
        f"{worlds} worlds an environment: together {joint:.3f}",
        *(
            f"    less {r:22s} {c:.3f}"
            + ("" if c >= CLAIMED else "; not claimed: predicted to fail")
            for r, c in zip(rivals, each, strict=True)
        ),
    ]


def _candidates(path: Path) -> dict[str, list[Candidate]]:
    """The candidates file: `{"national": [...], "geo": [...]}`, each candidate an object with its
    `name`, `rung` ("0" or "tested"), and optionally `settings` and `cpu`."""
    held = json.loads(path.read_text())
    return {
        group: [
            Candidate(c["name"], str(c["rung"]), int(c.get("settings", 0)), c.get("cpu"))
            for c in held.get(group, [])
        ]
        for group in ("national", "geo")
    }


def _lines(
    rung: str, scored: Sequence[Scored], winner: Scored, tied: Sequence[Scored]
) -> list[str]:
    out = [f"rung {rung}: {len(scored)} candidates; the winner {winner.candidate.name}"]
    if len(tied) > 1:
        out[0] += f", tied within {TIE:g} with {', '.join(s.candidate.name for s in tied[1:])}"
    for s in sorted(scored, key=lambda s: (s.score, s.candidate.name)):
        family, name = s.binding
        out.append(
            f"  {s.candidate.name:28s} score {s.score:+.4f} (family {family} less {name}); "
            f"CPU-s a fit {s.cpu:.0f}; settings {s.candidate.settings}"
        )
        out += [
            f"    family {f:2d}: failed on {k} of {n} fit worlds, each played as the status quo"
            for f, (k, n) in sorted(s.failed.items())
        ]
        out += [
            f"    family {f:2d} less {r:22s} {m:+.4f} on {n} worlds"
            for (f, r), (m, n) in sorted(s.d.items())
        ]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    splitting = commands.add_parser("split", help="each stored world's split, counted")
    splitting.add_argument("roots", type=Path, nargs="+")
    selecting = commands.add_parser("select", help="the rule's winner at each rung")
    selecting.add_argument("--stored", type=Path, nargs="+", required=True)
    selecting.add_argument("--pilot", type=Path, nargs="*", default=[])
    selecting.add_argument(
        "--fresh",
        type=Path,
        nargs="*",
        default=[],
        help="pilot roots of the families the stored run never held, read for the predictive",
    )
    selecting.add_argument("--candidates", type=Path, required=True)
    selecting.add_argument(
        "--worlds",
        action="append",
        default=[],
        metavar="FAMILY=N",
        help="the scored run's worlds an environment, for the winner's predictive",
    )
    selecting.add_argument("--resamples", type=int, default=RESAMPLES)
    selecting.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.command == "split":
        held = development(args.roots)
        counts: dict[tuple[int, str, str], int] = defaultdict(int)
        for (family, environment, _), part in held.splits.items():
            counts[family, environment, part] += 1
        for family, environment in sorted({(f, e) for f, e, _ in counts}):
            fit, check = counts[family, environment, FIT], counts[family, environment, CHECK]
            print(f"family {family} {environment}: fit {fit}, check {check}")
        return
    dev = development(args.stored)
    records = joined(args.stored, args.pilot)
    new, claimed = _fresh(args.fresh)
    if {(f, e, s) for f, e, s, _ in new} & dev.splits.keys():
        msg = "a fresh root holds a world of the stored run"
        raise ValueError(msg)
    dev = Development({**dev.splits, **{(f, e, s): CHECK for f, e, s, _ in new}})
    records |= new
    held = {f for f, _, _, _ in new}
    worlds = {int(f): int(n) for f, _, n in (w.partition("=") for w in args.worlds)}
    groups = {"national": dict.fromkeys(NATIONAL), "geo": GEO}
    for group, candidates in _candidates(args.candidates).items():
        families: Mapping[int, Collection[str] | None] = groups[group]
        for rung in sorted({c.rung for c in candidates}):
            scored = [score(records, dev, c, families) for c in candidates if c.rung == rung]
            if group == "national" and rung == "tested":
                scored, guarded = screen(records, dev, scored)
                print(f"{group}, rung {rung}, the S-curve guard:\n" + "\n".join(guarded))
            winner, tied = select(scored)
            print(f"{group}, " + "\n".join(_lines(rung, scored, winner, tied)))
            name = winner.candidate.name
            reads = [
                (f, e, sorted({r for g, r in winner.d if g == f})) for f, e in families.items()
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
                if not rivals:
                    msg = f"{name} is read on no world of family {family} at rung {rung}"
                    raise ValueError(msg)
                reads.append((family, None, rivals))
            for family, environments, rivals in reads:
                if family not in held:
                    print(
                        f"  common random numbers, {name}, family {family} at rung {rung}: the "
                        "paired variance over the unpaired, fit split\n"
                        + "\n".join(
                            f"    less {r:22s} {ratio:.3f} on {n} worlds"
                            for r, (ratio, n) in pairing(
                                records, dev, name, family, rung, rivals, environments
                            ).items()
                        )
                    )
                if family not in worlds:
                    continue
                strata = check_strata(records, dev, name, family, rung, rivals, environments)
                print(
                    "\n".join(
                        _predicted(
                            name,
                            family,
                            rung,
                            rivals,
                            strata,
                            worlds[family],
                            resamples=args.resamples,
                            seed=args.seed,
                        )
                    )
                )


if __name__ == "__main__":
    main()
