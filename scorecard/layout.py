"""A run: what it is declared to hold, and where it keeps each file.

A run is one family's worlds, each environment's first `worlds` in the family's seeded order,
pilots or scored, at some rungs of the ladder, planned by some arms. It is declared once, in
`run.json`, and every later step reads the declaration, so a run cannot drift from what it said it
would hold. Under its directory:

* `exports/family-<f>/<environment>/k<k>/world_<seed>.npz`: what every arm reads of a world, as the
  bench archives it (`causaldyn_bench.scorecard.observe.export`);
* `records/<arm>/family-<f>/<environment>/k<k>/world_<seed>.json`: one arm's record of one world,
  this engine's arms' written by `scorecard.run`, a tool's by the bench's script for it;
* `inputs/<arm>/family-<f>/<environment>/k<k>/world_<seed>/`: the one world a tool's process reads,
  its export hard-linked, and where the process writes its record before the harness moves it into
  `records`;
* `scores.jsonl`: each world's score, every arm's beside the built-in ones, one line a world;
* `logs/`: each world's process's output, and `gate.lock`, which admits one process at a time.

A run names the pre-registration it runs under, the file and its SHA-256 when the run was declared
(`stamp`); the runner declares a scored run only with one. The report holds the file to that hash
(`Declaration.preregistered`), so a run whose design moved after its declaration is not reported.

Every file is written aside and moved into place, so a reader sees all of it or none, and a step
that stops is resumed by the files that are missing.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from causaldyn_bench.scorecard.collinear import COLLINEAR
from causaldyn_bench.scorecard.confounded import CONFOUNDED
from causaldyn_bench.scorecard.drift import DRIFT
from causaldyn_bench.scorecard.few_geos import FEW_GEOS
from causaldyn_bench.scorecard.flat import FLAT
from causaldyn_bench.scorecard.geo import GEO
from causaldyn_bench.scorecard.long_term import LONG_TERM
from causaldyn_bench.scorecard.meridian import MERIDIAN
from causaldyn_bench.scorecard.outside import OUTSIDE
from causaldyn_bench.scorecard.performance import PERFORMANCE
from causaldyn_bench.scorecard.preregistration import Stamp, digest, verify
from causaldyn_bench.scorecard.promotions import PROMOTIONS
from causaldyn_bench.scorecard.pymc_marketing import PYMC_MARKETING
from causaldyn_bench.scorecard.robyn import ROBYN
from causaldyn_bench.scorecard.s_curves import S_CURVES
from causaldyn_bench.scorecard.scoring import BUILT_IN
from causaldyn_bench.scorecard.short import SHORT
from causaldyn_bench.scorecard.simmmulator import Simmmulator
from causaldyn_bench.scorecard.spillover import SPILLOVER
from causaldyn_bench.scorecard.track_m2 import TRACK_M2
from causaldyn_bench.scorecard.twins import TWINS

if TYPE_CHECKING:
    from collections.abc import Mapping

    from causaldyn_bench.scorecard.family import Family

FAMILIES: dict[int, Family[Any] | type[Simmmulator]] = {
    0: TRACK_M2,
    1: DRIFT,
    2: CONFOUNDED,
    3: COLLINEAR,
    4: S_CURVES,
    5: PERFORMANCE,
    6: OUTSIDE,
    7: SHORT,
    8: GEO,
    9: SPILLOVER,
    10: FEW_GEOS,
    12: MERIDIAN,
    13: PYMC_MARKETING,
    14: ROBYN,
    15: Simmmulator,
    17: TWINS,
    18: LONG_TERM,
    19: FLAT,
    20: PROMOTIONS,
}
"""The bench's families a run may hold, by number. Family 15's worlds are files the bench reads
and does not draw, so its entry is its class, made on the directory a run declares
(`Declaration.world_directory`)."""
TREE = Path(__file__).resolve().parents[1]
"""The engine's tree: its pre-registrations are kept under it (`scorecard/preregistration/`), and a
pre-registration's relative path is read from it."""
READERS = ("student", "hall")
"""The readers of a comparison's one look: Student's t, or Hall's transform."""
UNITS = {"": 1, "K": 2**10, "M": 2**20, "G": 2**30, "T": 2**40}


def size(text: str) -> int:
    """A size in bytes, as systemd reads `MemoryMax`: a whole number of bytes, or of K, M, G or T,
    each 1024 of the one before.

    Raises:
        ValueError: on any other form.
    """
    found = re.fullmatch(r"([0-9]+)([KMGT]?)", text)
    if found is None:
        msg = f"a size is a whole number of bytes, K, M, G or T, as 8G, not {text!r}"
        raise ValueError(msg)
    return int(found[1]) * UNITS[found[2]]


@dataclass(frozen=True, slots=True)
class Limits:
    """How each world of one arm runs: in a scope capped at `memory` (`size`) with no swap,
    stopped after `timeout` seconds, `slots` worlds at once; `cpus`, the CPUs a stage's admission
    reserves for one of its processes, 1 unless the arm is measured to take more. Limits of one CPU
    are written as they were before the CPUs were declared.

    Raises:
        ValueError: on a size systemd does not read, or a timeout or slots below what runs at all.
    """

    memory: str
    timeout: float
    slots: int
    cpus: float = 1.0

    def __post_init__(self) -> None:
        if size(self.memory) < 1 or self.timeout <= 0 or self.slots < 1 or not self.cpus > 0:
            msg = f"a world needs some memory, time and a slot, not {self}"
            raise ValueError(msg)

    @property
    def cap(self) -> int:
        return size(self.memory)


@dataclass(frozen=True, slots=True)
class Declaration:
    """What a run holds: `family`'s `environments`, each one's first `worlds` worlds, pilots if
    `pilot`, the worlds no scored run read if `confirmatory`
    (`causaldyn_bench.scorecard.family.confirmatory`), at each rung of `rungs`; this engine's
    `arms` plan them and the records of the tools' `external` arms are read beside them. A
    `smoke` run checks the plumbing with reduced fits and is never a result. A scored run names
    `bench`, the bench's commit it is pre-registered against, and runs only there.

    How the run runs is declared with it, so that it says what it ran: each arm's `limits`
    (`Limits`); the memory gate's `reserve`, the size kept free beside a process's cap, and its
    `settle`, the seconds each start holds the gate so that its memory shows before the next is
    admitted; the `attempts` a world gets before it is recorded as a failure; Robyn's R library
    (`r_library`) and nevergrad's Python environment (`nevergrad`); and, for a family whose
    worlds are files, the directory they are read from (`world_directory`).

    A run the claim reads (`scorecard.claim`) declares the engine's `default` procedure, one of
    its arms' variants, with the claim's `margin` in regret per euro. A run whose family needs more
    worlds than the claim affords is `exploratory`: it is read, and left out of the claim. A tool
    too slow to plan every world is read on a prefix: `caps` gives each such tool of `external`
    its worlds an environment, the first of the seeded order, its comparisons read on as many of
    them as the run holds; with `exploratory_caps` its comparisons are read and left out of the
    claim. A pilot's caps are those its joint sizing reads (`scorecard.claim`).

    How the claim reads the run is declared with it too. Its `reader` reads each comparison's one
    look: Student's t, or Hall's (1992) transform of the studentised mean, which removes the
    skewness of the paired differences (`causaldyn_bench.scorecard.gate.Gate`). Its default
    procedure may change at the rungs above 0, where the worlds hold lift tests: there the claim
    reads `default_above_0`, and where the run's own arms do not plan it, `joined` names the root
    beside the run's that does, the run's directory name with it appended (`-ts` for the two-stage
    arm), whose worlds pair with the run's by environment, seed and rung. A family read over several
    runs names them all in `pool`, by their directories' names, and the claim reads them as one
    run over every environment they hold. A declaration without caps, a reader of its own, a
    default above 0 or a pool is written as one was before they existed, so a file written then
    reads and is declared again unchanged.

    A run may declare its claim read by family and rung (`per_rung`, ADR 0081): each rung's
    comparisons an intersection-union test of their own, the verdicts tested at 0.025 one at a time
    in the fixed sequence `order` declares, each a family and a rung, the first that fails ending
    it; the printed word Hall's where the reader is. The comparisons its pre-registration predicts
    to fail (`predicted`, each a rung and a rival; a rung above 0 named as a pool's first run names
    it) are read, printed and left out of the verdicts. A run that declares none of these reads as
    before.

    A run the claim reads may declare some of its rungs exploratory (`exploratory_rungs`; a rung
    above 0 named as a pool's first run names it): a rung whose family needs more worlds there than
    the claim affords, past the sizing cap (ADR 0100). Each is read and printed as an exploratory
    run is, and left out of the claim, of Holm's list and of the order of verdicts. A rung kept out
    of the claim this way is not predicted to fail. A run that declares none is written as one was
    before the field existed.

    A run the claim reads may name variants of its own arms its default at rung 0 is read against
    beside the claim, never in it (`default_less`): the default less each, paired, run by run and
    pooled over the confirmatory runs (`scorecard.claim.against_default`). A run that names none
    is written as one was before the field existed.

    A confirmatory run may be the blocks' second (`second_confirmatory`): it draws each
    environment's confirmatory worlds past those the first confirmatory run draws from, so that it
    reads no world of the first's (`causaldyn_bench.scorecard.family.confirmatory`). A run that is
    not is written as one was before the field existed. It may instead be the third
    (`third_confirmatory`): each environment's worlds from a block of its own in a region no
    earlier run or block holds, family 17's twins apart (ADR 0101).

    The pre-registration the run runs under is declared with it, `preregistration`: its path and
    the SHA-256 of the file when the run was declared (`stamp`). The runner requires one of a
    scored run (`scorecard.run.declare`); a run declared without one, as every run was before it
    existed, is written and read unchanged.

    Raises:
        ValueError: on a family the harness does not know, a run of pilots that is
            confirmatory, a directory of worlds for a family that draws its own or none for one
            that reads files, an environment the family does
            not have, fewer worlds than 1 or more than an environment draws, no rung or a rung
            below 0, no arm, an arm named twice, a scored run that names no commit of the bench,
            limits for an arm the run does not name, a reserve systemd does not read, a negative
            settle, no attempt, one of Robyn's two paths without the other, the default
            procedure without the margin or the other way round, a margin that is not a number
            above 0, a tool's arm as the default, a margin on fewer than 2 worlds an environment,
            an exploratory run without a margin, a cap on an arm that is not one of the run's tools
            or below 2 worlds, exploratory caps without a cap, a reader that is neither
            Student's t nor Hall's or one on a run the claim does not read, a second
            confirmatory run that is not confirmatory, a third confirmatory run that is not
            confirmatory or is also the second, a default above 0
            without the default or that is a tool's arm, a joined root without a default above 0
            or whose suffix is not a hyphen and a name, or a pool on a run the claim does not
            read, of fewer than 2 runs, with a run named twice or a name that is not a
            directory's; a claim by rung on a run the claim does not read or without an order of
            its verdicts, an order or a predicted failure without one, an order that names a
            family and rung twice, one the harness does not know or one of this run's the claim
            does not read (`reads`), or a malformed predicted failure; a variant read against the
            default on a run the claim does not read or without rung 0, named twice, or that is
            the default or a tool's arm; an exploratory rung on a run the claim does not read,
            named twice or that the run does not hold.
    """

    family: int
    environments: tuple[str, ...]
    worlds: int
    pilot: bool
    rungs: tuple[int, ...]
    arms: tuple[str, ...]
    external: tuple[str, ...] = ()
    smoke: bool = False
    bench: str | None = None
    limits: Mapping[str, Limits] = field(default_factory=dict)
    reserve: str = "2G"
    settle: float = 60.0
    attempts: int = 2
    r_library: str | None = None
    nevergrad: str | None = None
    world_directory: str | None = None
    default: str | None = None
    margin: float | None = None
    exploratory: bool = False
    caps: Mapping[str, int] | None = None
    exploratory_caps: bool = False
    reader: str = "student"
    default_above_0: str | None = None
    joined: str | None = None
    pool: tuple[str, ...] = ()
    preregistration: Stamp | None = None
    per_rung: bool = False
    predicted: tuple[tuple[int, str], ...] = ()
    confirmatory: bool = False
    order: tuple[tuple[int, int], ...] = ()
    default_less: tuple[str, ...] = ()
    second_confirmatory: bool = False
    exploratory_rungs: tuple[int, ...] = ()
    third_confirmatory: bool = False

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            msg = f"family {self.family} is not one of {sorted(FAMILIES)}"
            raise ValueError(msg)
        if self.pilot and self.confirmatory:
            msg = "a run draws pilots or confirmatory worlds, not both"
            raise ValueError(msg)
        if self.second_confirmatory and not self.confirmatory:
            msg = "a second confirmatory run is a confirmatory run: declare it confirmatory"
            raise ValueError(msg)
        if self.third_confirmatory and (not self.confirmatory or self.second_confirmatory):
            msg = "a third confirmatory run is a confirmatory run and not the second"
            raise ValueError(msg)
        samples = self.samples()
        unknown = [e for e in self.environments if e not in samples]
        if unknown or not self.environments:
            msg = f"{self.name()} draws {sorted(samples)}, not {unknown or 'no environment'}"
            raise ValueError(msg)
        short = {e: len(samples[e]) for e in self.environments if len(samples[e]) < self.worlds}
        if self.worlds < 1 or short:
            msg = f"a run reads at least 1 world an environment and at most {short or 'all'}"
            raise ValueError(msg)
        if not self.rungs or min(self.rungs) < 0:
            msg = f"a run reads at least one rung, each at or above 0, not {self.rungs}"
            raise ValueError(msg)
        names = [*self.arms, *self.external]
        if not self.arms or len(set(names)) != len(names):
            msg = f"a run names its arms once each, and at least one of this engine's: {names}"
            raise ValueError(msg)
        if not (self.pilot or self.smoke or self.bench):
            msg = "a scored run names the bench's commit it is pre-registered against"
            raise ValueError(msg)
        stray = sorted(set(self.limits) - set(names))
        if stray:
            msg = f"the run declares limits for {stray}, which it does not name"
            raise ValueError(msg)
        size(self.reserve)
        if self.settle < 0 or self.attempts < 1:
            msg = f"the gate settles for 0 s or more and a world has an attempt, not {self}"
            raise ValueError(msg)
        if (self.r_library is None) != (self.nevergrad is None):
            msg = "Robyn's R library and nevergrad's environment are declared together"
            raise ValueError(msg)
        if (self.default is None) != (self.margin is None):
            msg = "the default procedure and the claim's margin are declared together"
            raise ValueError(msg)
        if self.margin is not None:
            if not (math.isfinite(self.margin) and self.margin > 0.0):
                msg = f"a margin is a regret per euro above 0, not {self.margin}"
                raise ValueError(msg)
            if self.default in self.external or self.worlds < 2:
                msg = (
                    f"the claim reads one of this engine's arms, not {self.default}, on at least "
                    f"2 worlds an environment, not {self.worlds}"
                )
                raise ValueError(msg)
        elif self.exploratory:
            msg = "an exploratory run is one the claim reads: declare its margin"
            raise ValueError(msg)
        caps = self.caps or {}
        wrong = {
            tool: worlds
            for tool, worlds in caps.items()
            if tool not in self.external
            or isinstance(worlds, bool)
            or not isinstance(worlds, int)
            or worlds < 2
        }
        if wrong:
            msg = f"a cap is 2 worlds or more of one of the run's tools {self.external}: {wrong}"
            raise ValueError(msg)
        if self.exploratory_caps and not caps:
            msg = "exploratory caps read the capped tools: declare their caps"
            raise ValueError(msg)
        self._check_claim()

    def _check_claim(self) -> None:
        """The reader, the default above 0, the joined root, the pool and the variants read
        against the default; see the class."""
        if self.reader not in READERS:
            msg = f"the claim reads a look by one of {READERS}, not {self.reader!r}"
            raise ValueError(msg)
        claimed = self.margin is not None
        if self.reader != "student" and not claimed:
            msg = f"the {self.reader} reader reads the claim: declare its margin"
            raise ValueError(msg)
        if self.default_above_0 is not None and (
            not claimed or self.default_above_0 in self.external
        ):
            msg = (
                "a default above 0 is one of this engine's arms on a run the claim reads, not "
                f"{self.default_above_0!r}"
            )
            raise ValueError(msg)
        if self.joined is not None and (
            self.default_above_0 is None
            or re.fullmatch(r"-[a-z0-9]+(-[a-z0-9]+)*", self.joined) is None
        ):
            msg = (
                "a joined root holds the default above 0 and is named by a suffix as -ts, not "
                f"{self.joined!r}"
            )
            raise ValueError(msg)
        if self.pool and (
            not claimed
            or len(self.pool) < 2
            or len(set(self.pool)) != len(self.pool)
            or any(not name or name in (".", "..") or "/" in name for name in self.pool)
        ):
            msg = (
                "a pool names 2 runs or more the claim reads, each once by its directory: "
                f"{self.pool}"
            )
            raise ValueError(msg)
        if (self.per_rung and not claimed) or (self.predicted and not self.per_rung):
            msg = "a claim by rung is a claim the run declares, and only it predicts failures"
            raise ValueError(msg)
        odd = [
            p
            for p in self.predicted
            if len(p) != 2 or isinstance(p[0], bool) or not isinstance(p[0], int) or p[0] < 0
        ]
        if odd or len(set(self.predicted)) != len(self.predicted):
            msg = f"a predicted failure is a rung and a rival, each named once: {self.predicted}"
            raise ValueError(msg)
        if self.per_rung != bool(self.order):
            msg = "a claim by rung tests its verdicts in the order it declares, and only it has one"
            raise ValueError(msg)
        malformed = [
            p
            for p in self.order
            if len(p) != 2
            or any(isinstance(n, bool) or not isinstance(n, int) for n in p)
            or p[0] not in FAMILIES
            or p[1] < 0
        ]
        if malformed or len(set(self.order)) != len(self.order):
            msg = f"the order of verdicts names each family and rung once: {self.order}"
            raise ValueError(msg)
        unread = [k for f, k in self.order if f == self.family and not self.reads(k)]
        if unread:
            msg = (
                f"the order of verdicts names family {self.family} at rungs {unread}, which the "
                "claim does not read of this run: it or the rung is exploratory, it does not hold "
                "the rung, or every rival there is predicted to fail or capped exploratory"
            )
            raise ValueError(msg)
        rungs = self.exploratory_rungs
        if rungs and (
            not claimed
            or len(set(rungs)) != len(rungs)
            or any(
                isinstance(k, bool) or not isinstance(k, int) or not self.holds(k) for k in rungs
            )
        ):
            msg = (
                "an exploratory rung is a rung the run holds, named once, on a run the claim "
                f"reads: not {rungs}"
            )
            raise ValueError(msg)
        against = self.default_less
        if against and (
            not claimed
            or 0 not in self.rungs
            or len(set(against)) != len(against)
            or set(against) & {self.default, *self.external}
        ):
            msg = (
                "the default at rung 0 is read against variants of the run's own arms, each named "
                f"once, on a run the claim reads and that holds rung 0: not {against}"
            )
            raise ValueError(msg)

    def holds(self, k: int) -> bool:
        """Whether the run holds rung `k`: a pool's rung above 0 is named as its first run names
        it, so a run of a pool holds any rung above 0 where it holds one of its own."""
        return k in self.rungs or (bool(self.pool) and k > 0 and any(r > 0 for r in self.rungs))

    def reads(self, k: int) -> bool:
        """Whether the claim reads a verdict of this run at rung `k`: the run is not exploratory,
        nor is the rung (`exploratory_rungs`), it holds the rung (`holds`), and some rival of its
        own there, a built-in arm or a tool not capped exploratory, is not predicted to fail. A tool
        of a root beside the run is not one of its own."""
        held = self.holds(k) and k not in self.exploratory_rungs
        capped = set(self.caps or {}) if self.exploratory_caps else set()
        claimed = [
            rival
            for rival in (*BUILT_IN, *self.external)
            if rival not in capped and (k, rival) not in self.predicted
        ]
        return not self.exploratory and held and bool(claimed)

    def preregistered(self) -> Stamp | None:
        """The pre-registration the run declares, once the file at its path, a relative one read
        from the engine's tree (`TREE`), still hashes to the SHA-256 declared; None where the run
        declares none.

        Raises:
            FileNotFoundError: no file is at the declared path.
            ValueError: the file's SHA-256 is not the one declared.
        """
        return None if self.preregistration is None else verify(self.preregistration, TREE)

    def default_at(self, k: int) -> str | None:
        """The default procedure the claim reads at rung `k`."""
        if k > 0 and self.default_above_0 is not None:
            return self.default_above_0
        return self.default

    def worlds_of(self, arm: str) -> int:
        """The worlds an environment `arm`'s comparisons read: its cap, at most the run's."""
        return min(self.worlds, (self.caps or {}).get(arm, self.worlds))

    @property
    def scored(self) -> bool:
        """Whether the run is a result: neither a pilot nor a smoke run."""
        return not (self.pilot or self.smoke)

    def source(self) -> Family[Any]:
        """The family the run's worlds come from.

        Raises:
            ValueError: on a directory of worlds for a family that draws its own, or none for one
                that reads its worlds from files.
        """
        held = FAMILIES[self.family]
        if not isinstance(held, type):
            if self.world_directory is not None:
                msg = f"family {self.family} draws its worlds; it reads none from a directory"
                raise ValueError(msg)
            return held
        if self.world_directory is None:
            msg = f"family {self.family} reads its worlds from files: declare their directory"
            raise ValueError(msg)
        return held(Path(self.world_directory))

    def name(self) -> str:
        return self.source().name

    def samples(self) -> dict[str, range]:
        family = self.source()
        if self.confirmatory:
            # imported here so that every other run still reads on the bench M2 pinned, which
            # holds no confirmatory blocks
            from causaldyn_bench.scorecard.family import confirmatory

            if self.third_confirmatory:
                return confirmatory(family, third=True)
            if self.second_confirmatory:
                return confirmatory(family, second=True)
            # called as on the bench M3 pinned, whose function takes no `second`
            return confirmatory(family)
        return dict(family.pilots if self.pilot else family.scored)

    def seeds(self, environment: str) -> range:
        """The environment's worlds the run holds, a prefix of its seeded order."""
        return self.samples()[environment][: self.worlds]

    def jobs(self) -> list[tuple[str, int, int]]:
        """Every world of the run at every rung: environment, seed and rung."""
        return [
            (environment, seed, k)
            for environment in self.environments
            for seed in self.seeds(environment)
            for k in self.rungs
        ]


@dataclass(frozen=True, slots=True)
class Layout:
    """The run kept under `root`."""

    root: Path

    @property
    def declaration(self) -> Path:
        return self.root / "run.json"

    @property
    def scores(self) -> Path:
        return self.root / "scores.jsonl"

    def exports(self, family: int, environment: str, k: int) -> Path:
        return self.root / "exports" / f"family-{family}" / environment / f"k{k}"

    def export(self, family: int, environment: str, k: int, seed: int) -> Path:
        return self.exports(family, environment, k) / f"world_{seed}.npz"

    @property
    def gate(self) -> Path:
        return self.root / "logs" / "gate.lock"

    @property
    def attempts(self) -> Path:
        """Each world's process's attempts, a line as each starts and as each ends."""
        return self.root / "logs" / "attempts.jsonl"

    @property
    def costs(self) -> Path:
        """Each process's cost, a line its scope appends as it ends."""
        return self.root / "logs" / "costs.jsonl"

    @property
    def reused(self) -> Path:
        """Each record taken from another run's root (`scorecard.run.reuse`)."""
        return self.root / "logs" / "reused.jsonl"

    def records(self, arm: str, family: int, environment: str, k: int) -> Path:
        return self.root / "records" / slug(arm) / f"family-{family}" / environment / f"k{k}"

    def record(self, arm: str, family: int, environment: str, k: int, seed: int) -> Path:
        return self.records(arm, family, environment, k) / f"world_{seed}.json"

    def inputs(self, arm: str, family: int, environment: str, k: int, seed: int) -> Path:
        """The directory a tool's process reads one world from and writes its record to."""
        held = self.root / "inputs" / slug(arm) / f"family-{family}" / environment / f"k{k}"
        return held / f"world_{seed}"

    def declare(self, declaration: Declaration) -> Declaration:
        """Write `declaration` as the run's, or read it back if the run already holds it.

        Raises:
            RuntimeError: the run is declared otherwise.
        """
        held = _written(declaration)
        if self.declaration.exists():
            stored = json.loads(self.declaration.read_text())
            if _tuples(stored) != _tuples(held):
                msg = f"{self.root} is declared as {stored}, not {held}"
                raise RuntimeError(msg)
            return declaration
        write_json(self.declaration, held)
        return declaration

    def read(self) -> Declaration:
        """The run's declaration.

        Raises:
            FileNotFoundError: the run was never declared.
        """
        stored = _tuples(json.loads(self.declaration.read_text()))
        if "limits" in stored:
            stored["limits"] = {arm: Limits(**held) for arm, held in stored["limits"].items()}
        if stored.get("preregistration") is not None:
            stored["preregistration"] = Stamp.from_json(stored["preregistration"])
        return Declaration(**stored)


ADDED = {
    "caps": None,
    "exploratory_caps": False,
    "reader": "student",
    "default_above_0": None,
    "joined": None,
    "pool": (),
    "preregistration": None,
    "per_rung": False,
    "predicted": (),
    "confirmatory": False,
    "order": (),
    "default_less": (),
    "second_confirmatory": False,
    "exploratory_rungs": (),
    "third_confirmatory": False,
}
"""Fields a declaration gained after runs were declared, each with the value that leaves a run as
it was: a declaration holding that value is written without the field."""


def _written(declaration: Declaration) -> dict[str, Any]:
    held = asdict(declaration)
    for name, unchanged in ADDED.items():
        if held[name] == unchanged:
            del held[name]
    for limits in held["limits"].values():
        if limits["cpus"] == 1.0:
            del limits["cpus"]
    return held


def _tuples(value: dict[str, Any]) -> dict[str, Any]:
    return {k: _tuple(v) for k, v in value.items()}


def _tuple(value: Any) -> Any:
    """`value` with every list in it a tuple, as a declaration holds them."""
    if isinstance(value, list | tuple):
        return tuple(_tuple(v) for v in value)
    return value


def stamp(path: Path | str) -> Stamp:
    """The stamp a run declares of the pre-registration at `path`, read from the working
    directory: the SHA-256 of the file's bytes now, beside its path, relative to the engine's tree
    (`TREE`) where the file is in it, so that the stamp names the file as the tree does, and
    absolute where it is not.

    Raises:
        FileNotFoundError: no file is at `path`.
    """
    found = Path(path).resolve()
    named = found.relative_to(TREE) if found.is_relative_to(TREE) else found
    return Stamp(str(named), digest(found))


def slug(arm: str) -> str:
    """An arm's name as a directory: lower case, every run of other characters one hyphen.

    Raises:
        ValueError: on a name with no letter or digit.
    """
    out = re.sub(r"[^a-z0-9]+", "-", arm.lower()).strip("-")
    if not out:
        msg = f"an arm's name needs a letter or a digit, got {arm!r}"
        raise ValueError(msg)
    return out


def write_json(path: Path, value: Any) -> None:
    """`value` as strict JSON at `path`, written aside and moved into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    aside = path.with_name(f".{path.name}.{os.getpid()}")
    aside.write_text(json.dumps(value, allow_nan=False))
    aside.replace(path)
