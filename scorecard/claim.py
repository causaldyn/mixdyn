"""The claim a scored scorecard makes over its families, and the worlds each family needs for it.

    PYTHONPATH=<bench>/src uv run --no-sync python -m scorecard.claim read [--recorded] \\
        RUN [RUN ...]
    PYTHONPATH=<bench>/src uv run --no-sync python -m scorecard.claim size --default NAME \\
        --margin DELTA [--joint [--resamples B] [--seed S] [--confidence C] [--families F]
        [--cap TOOL=WORLDS ... [--exploratory-caps]]] PILOT [PILOT ...]

**A comparison.** A run that declares the engine's default procedure and a margin `delta`
(`Declaration.default`, `Declaration.margin`) is read at each of its rungs, the default against
each rival: every arm whose records are not this engine's, the built-in arms and the run's tools.
The engine's other variants are its own alternatives to the default, not rivals. A comparison reads
the default's regret per euro less the rival's, paired over every environment's declared worlds,
by the bench's gate (`causaldyn_bench.scorecard.gate.Gate`) at one look: it is within the margin
where the paired 95 % Student-t interval's upper end lies below `delta`. A run that does not yet
hold its declared worlds whole in every environment leaves its comparisons unread.

**The reader.** A run declares how the look is read (`Declaration.reader`): by Student's t, as
above, or by Hall's (1992) transform, `reader = "hall"`, the bench's one-look reader for skewed
differences (`Gate(looks, reader="hall")`). Hall reads `x`, the margin less each paired difference,
`T = sqrt(n) mean(x) / sigma`, and shows the comparison where
`g(T) = T + c T^2 + c^2 T^3 / 3 + c / 2`, `c = gamma / (3 sqrt(n))`, reaches `z_0.975`, `sigma` and
the skewness `gamma` of `x` on divisor `n`: the Edgeworth term of order `n^(-1/2)` that skewed
differences leave in Student's level is removed (`validation/hall_reader.mac`). The losses are read
by the same reader, each at the level of its `g(T)`. The printed interval stays the paired
Student-t interval; with Hall's reader the line says so.

**The default per rung.** A run may declare its default above rung 0, where the worlds hold lift
tests, apart from its default at 0 (`Declaration.default_above_0`): the claim's DLM arm is
`dlm plug-in` at k = 0 and `dlm two-stage` above. Where the run's own arms do not plan it, the run
names the root that does (`Declaration.joined`), its own directory's name with the suffix appended,
`<root>-ts`, which holds that arm's records of the run's worlds at its rungs above 0, or, where
the default above 0 is a composite (`scorecard.composite`), its router's and its routed arm's;
`read_run` pairs the two roots' worlds by environment, seed and rung, every world's built-in arms
scored alike in both. A world the joined root does not hold yet leaves the rung's comparisons
unread.

**A family over several runs.** A family sized pooled over its environments and run as several
runs, each holding some of them, declares the runs in `Declaration.pool`, by their directories'
names. `claim read` reads them as one run over every environment they hold (`pooled`), each
environment's first worlds paired as one run's are, so the comparison is the one the sizing
stratified by environment. The runs must run under one pre-registration and agree on everything
the claim reads, and `claim read` is handed all of them. They may differ in one more thing: runs
that each hold rung 0 and one rung above it may test at different rungs above 0, as family 7 tests
at 1 on `L-52` and at 2 on `L-78` and `L-104` (`tested_rungs`). The pool then reads each run's
records at its rung above 0 as the first run's rung, one comparison a rival over every environment,
and the line names each environment's own rung.

**The roots beside a run.** `claim read` is handed every root of a run. A root that declares no
margin is a run's joined root, read with it (`read_run`), or a root beside a run handed
(`read_runs`). Every arm the joined root plans and the run does not is joined to the run's worlds at
the rungs above 0: the default there, and the rest, which only the reading beside the claim reads
(`recorded`). A root beside a run is a directory beside the run's, named by the run's directory
name, a hyphen and a name of its own, as `<root>-robyn` and `f10-meridian`. It runs under the run's
pre-registration and holds the run's family, its pilots or scored worlds, its bench and directory of
worlds, some of its environments at every rung of the run, and the run's worlds or a cap on each arm
it adds (`read_beside`). Each arm it plans and the run does not is joined to the run's worlds by
environment, seed and rung (`join`); an arm both plan, as `chc`, is the run's. Each of its tools is
read against the default as a tool of the run is, on the root's environments and on its cap,
exploratory where the root declares its caps so: an environment the root does not hold is not
compared and holds none of the tool's failures. Its own arms, as `bayes-geo`'s, are joined and
compared with nothing in the claim. The roots beside a pool's runs are read as one root over their
environments, each run handed one of the same name.

**Reported regardless** (`recorded`, `claim read --recorded`). After the claim, and outside it and
Holm's list, `claim read --recorded` prints what the claim does not read. At each rung of each run
as the claim reads it, each recorded arm, an arm of this engine's, the run's, its joined root's at
the rungs above 0 or a root's beside it, that is not the rung's default, less every rival: the run's
(`rivals`) and the tools of the roots beside it. And at each rung above 0 where the run declares its
default there, that default less the engine's static arm (`STATIC`). Each is read as a comparison
is: on the complete cases of the environments both arms are read on, on a capped tool's cap, each
environment's own rung named where a pool's runs test apart. It has no margin and no reader: its
mean, its paired 95 % Student-t interval, its complete cases and the reading rule's verdict,
unadjusted.

**The default against a declared variant** (`against_default`). A run may name variants of its
own arms its default at rung 0 is read against (`Declaration.default_less`), as a check that the
claim's procedure was the better choice. Beside the claim, with the secondary readings, the
default less each is read run by run as `recorded` reads an arm, then pooled over the runs on
confirmatory worlds: every complete case one world, the mean's variance stratified by run (each
run's own paired variance, `sum_f n_f s_f^2 / N^2`), its 95 % interval on the normal quantile.
Neither enters the claim or Holm's list.

**Complete cases.** The bench scores an arm whose fit raised as the status quo, which mixes a tool's
quality with the status quo's. So a comparison is read on the worlds where both the default and the
rival made a plan (`complete_case`), pooled over the environments it compares, with each arm's
failures counted beside it. An environment where the rival failed on more than half of the look's
worlds (`HALF`) is not compared, and a rival not compared in any environment leaves the claim
(`Comparison.compared`). The reading with the status quo standing in for every failed plan stays
beside it as a secondary readout (`Comparison.fallback_mean`, `fallback_interval`,
`fallback_within`).

**The claim** is an intersection-union test (Berger 1982): it holds only where every comparison of
every run read, at every rung, is within the margin, so each comparison's one-sided 0.025 is the
claim's, with no adjustment for how many it needs. A run declared `exploratory` is read and printed,
and left out of the claim, and so is a rung the run declares exploratory
(`Declaration.exploratory_rungs`): past the sizing cap, not predicted to fail. A tool the run
declares capped (`Declaration.caps`) is read on its cap's prefix of each environment's worlds, once
the run holds the prefix whole, and its comparisons are left out of the claim where the run declares
its caps exploratory (`Declaration.exploratory_caps`).

**The losses.** Each comparison is read the other way too: the rival beats the default where the
interval of the default less the rival lies above 0. Every comparison's verdict is printed
unadjusted, and beside them the losses Holm's step-down (`gate.holm`) shows at familywise 0.025
over every comparison the claim reads, each at its sequential level (`Gate.level`): at one look,
the Student-t p-value of the rival beating the default. An unread comparison holds its place in
the step-down at level 1.

**The worlds a family needs** (`size`). At a true tie, a mean difference of 0, a comparison of
paired SD `sd` over `N` worlds is within `delta` with chance 0.8 where
`N = ((z_0.975 + z_0.8) sd / delta)^2`, and `z_0.975 + z_0.8 = 2.8016` (`Z`). A family's `N` is
the fewest worlds, in tens, that give every comparison that chance on the pilot's paired SDs, so the
largest SD over the rivals and the rungs binds. A family whose `N` passes `CAP` is exploratory.

**The worlds the claim needs on every comparison together** (`joint_size`, `--joint`). The claim
holds only where all its comparisons are within, so at a true tie its chance is that of all of them
together (Berger 1982), below the binding comparison's `POWER` wherever another comparison can
fail. The joint `N` is the fewest worlds, in tens, at which the pilot's paired differences, every
comparison's centred at 0 and resampled world by world within each environment, the same worlds
for every comparison and the same resamples for every candidate `N`, pass all the comparisons with
chance at least `POWER`, each read as the gate reads it at one look (Sozu, Sugimoto and Hamasaki
2010 size co-primary endpoints this way). It prints each comparison's chance and the joint one. It
is an option beside `size`, whose rule the declared runs follow.

A pilot's SDs are estimates: an `N` sized on them reaches `POWER` about half the time. Asked to
(`confidence`, `--confidence`), the joint sizing raises every comparison's centred differences by
the ratio of their SD's one-sided upper confidence limit to the SD, `upper_sd`, Browne's (1995)
rule: sized on the 80 % limit, a study reaches its planned power with chance at least 0.8 where the
differences are normal (Kieser and Wassmer 1996). The limit is normal theory: differences with
heavy tails spread their SD wider than it allows.

**The launch claim over families** holds only where every family's claim holds. The bench draws
each family's worlds from a block of seeds and a stream of its own
(`causaldyn_bench.scorecard.seeds`), and each arm fits a world from the world's own seed, so the
families' readings are independent by construction and at a true tie the launch claim's chance is
the product of the families' joint chances (`overall`): `F` families at `POWER` each give
`POWER^F`, 0.107 at `F = 10`. A claim that fails at a true tie is then the likely outcome, not
evidence of a loss; the losses are Holm's. `size --joint` prints the product over the families it
sizes. Asked to (`--families F`), it sizes each family to `POWER^(1/F)` (`each_family`), so that
`F` families together reach `POWER`.

**A capped tool** (`capped`, `--cap TOOL=WORLDS`) is one too slow to plan every world of the family:
it plans each environment's first `WORLDS` worlds, a prefix of the worlds the other arms plan, so
its comparisons pass with the chance of that many worlds whatever the family's `N`, and they bound
the family's joint chance from above. The joint sizing reads each capped comparison on the cap's
prefix of the same resamples. Asked to (`exploratory`, `--exploratory-caps`), the capped
comparisons are exploratory: printed with their chances, they leave the joint chance and the joint
`N`, and `JointSize.whole`, every comparison's chance together with the capped ones on their caps,
states what the cap costs the family. Otherwise they stay in the claim, each on its cap, and the
joint `N` grows to cover them, or is refused where no `N` can. A pilot that declares caps is sized
on them, and as exploratory where it declares them so; `--cap` adds a cap or replaces one.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from causaldyn_bench.external_arms import verdict
from causaldyn_bench.scorecard.gate import ALPHA, Gate, holm
from causaldyn_bench.scorecard.scoring import BUILT_IN, WorldRecord, look, paired
from scipy.stats import chi2, norm
from scipy.stats import t as student

from scorecard.composite import compose, parts
from scorecard.layout import Layout

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    from causaldyn_bench.scorecard.gate import Reader
    from causaldyn_bench.scorecard.preregistration import Stamp

    from scorecard.layout import Declaration

POWER = 0.8
"""The chance a comparison at a true tie is within the margin on the worlds `size` gives."""
Z = float(norm.isf(ALPHA) + norm.isf(1.0 - POWER))
"""z_0.975 + z_0.8: 2.8016 to four places."""
TENS = 10
CAP = 520
"""The most worlds a family's claim reads: the design's figure for 80 % at a true tie at the drawn
worlds' paired SD of 0.081 and a margin of 0.01, (2.8016 * 0.081 / 0.01)^2 = 515, in tens."""

RESAMPLES = 20_000
"""The bootstrap resamples `joint_size` reads: a joint chance near 0.8 to a Monte Carlo SE of
0.003."""
SEED = 1982
"""The bootstrap's seed: any fixed one, so that a pilot's joint `N` is a fact of the pilot."""
MOST = 5000
"""The most worlds `joint_size` searches."""

Key = tuple[int, int, str]
"""A comparison's family, rung and rival."""
HALF = 0.5
"""The share of an environment's worlds a rival may fail on and still be compared there."""
POOLED = "complete cases"
"""The one environment a comparison's complete cases are pooled into for the gate's look."""
AGREED = (
    "preregistration",
    "family",
    "worlds",
    "pilot",
    "smoke",
    "bench",
    "rungs",
    "arms",
    "external",
    "world_directory",
    "default",
    "margin",
    "exploratory",
    "caps",
    "exploratory_caps",
    "reader",
    "default_above_0",
    "joined",
    "pool",
    "per_rung",
    "predicted",
    "confirmatory",
    "order",
    "default_less",
    "second_confirmatory",
    "exploratory_rungs",
)
"""What the runs of one pool declare alike: the pre-registration they run under, and everything the
claim and the reading beside it (`recorded`) read but their environments and, where they test apart
(`tested_rungs`), their rungs."""
BESIDE = ("worlds", "arms", "external", "caps", "exploratory_caps")
"""What the roots of one name beside a pool's runs declare alike: what the claim and the reading
beside it (`recorded`) read of them but their environments."""
STATIC = "stacked"
NARROW = 0.01
"""The margin of the claim by rung's secondary reading (ADR 0081), beside the declared one."""
MERIDIAN = "meridian"
"""The tool whose gain over the status quo at rung 0 the claim by rung reports beside it."""
"""The engine's static Bayesian arm (`scorecard.arms.bayes`), which the default above rung 0 is
read less beside the claim (`recorded`): what a second stage adds over a static fit."""


def rivals(declaration: Declaration) -> tuple[str, ...]:
    """The arms the default procedure is read against: every arm whose records are not this
    engine's."""
    return (*BUILT_IN, *declaration.external)


@dataclass(frozen=True, slots=True)
class Comparison:
    """The default procedure less one rival at one rung of one run, regret per euro: its mean and
    paired 95 % interval, the reading rule's unadjusted verdict, whether the interval's upper end
    lies below the margin, and the level at which the rival beats the default, on the complete
    cases; all None where the run does not hold its worlds whole or no two complete cases are left.
    `cap` is a capped rival's worlds an environment, the prefix it is read on. `pairs` are the
    complete cases read, `failures` and `default_failures` the rival's and the default's failed
    plans over the look, `left_out` the environments not compared, and `compared` False where no
    environment is. The `fallback_` fields read every world of the look, the status quo standing in
    for a failed plan. `default` is the default procedure read at the rung, `reader` the look's
    reader, and `statistic` the look's statistic on the complete cases: `(delta - mean) / se` by
    Student's t, `g(T)` by Hall's. `environments` are those a tool of a root beside the run is
    read on, where they are not all the run's, and `tested` each environment's own rung above 0,
    where a pool's runs test apart (`tested_rungs`). `within_level` is the level at which the look
    shows the comparison within the margin, as `level` is the rival's beating the default's; and
    `claimed` is False where the run declares the comparison predicted to fail
    (`Declaration.predicted`), so that it is read and left out of the claim by rung (`verdicts`)."""

    family: int
    k: int
    rival: str
    exploratory: bool
    mean: float | None = None
    interval: tuple[float, float] | None = None
    reading: str | None = None
    within: bool | None = None
    level: float = 1.0
    cap: int | None = None
    pairs: int | None = None
    failures: int | None = None
    default_failures: int | None = None
    left_out: tuple[str, ...] = ()
    compared: bool = True
    fallback_mean: float | None = None
    fallback_interval: tuple[float, float] | None = None
    fallback_within: bool | None = None
    default: str | None = None
    reader: str = "student"
    statistic: float | None = None
    environments: tuple[str, ...] = ()
    tested: tuple[tuple[str, int], ...] = ()
    within_level: float = 1.0
    claimed: bool = True

    @property
    def key(self) -> Key:
        return (self.family, self.k, self.rival)


def _failed(record: WorldRecord, arm: str) -> bool:
    score = record.arms.get(arm)
    return score is None or score.failure is not None


def complete_case(
    records: Sequence[WorldRecord], default: str, rival: str, worlds: int
) -> tuple[list[WorldRecord], dict[str, int], dict[str, int], tuple[str, ...]]:
    """The look at `worlds` worlds of each environment cut to its complete cases: the worlds of the
    environments compared where neither `default` nor `rival` failed, pooled into one environment
    (`POOLED`) and numbered in their order; each environment's failures of the rival and of the
    default over the look; and the environments left out, where the rival failed on more than
    `HALF` of the look's worlds.

    Raises:
        ValueError: as `causaldyn_bench.scorecard.scoring.look` does.
    """
    seen = look(records, worlds)
    environments = sorted({r.environment for r in seen})
    failures = {e: sum(_failed(r, rival) for r in seen if r.environment == e) for e in environments}
    own = {e: sum(_failed(r, default) for r in seen if r.environment == e) for e in environments}
    left_out = tuple(e for e in environments if failures[e] > HALF * worlds)
    kept = [
        r
        for r in seen
        if r.environment not in left_out and not _failed(r, default) and not _failed(r, rival)
    ]
    pooled = [replace(r, environment=POOLED, index=i) for i, r in enumerate(kept)]
    return pooled, failures, own, left_out


def compare(
    records: Sequence[WorldRecord],
    declaration: Declaration,
    *,
    beside: Sequence[Declaration] = (),
    tested: Sequence[tuple[str, int]] = (),
) -> list[Comparison]:
    """Every comparison of the run whose `records` these are, rung by rung: its own rivals, then
    each tool of the roots `beside` it that the run does not hold, read on its root's environments
    (`read_runs`). `tested` names each environment's own rung above 0 where a pool's runs test at
    different rungs (`tested_rungs`).

    Raises:
        ValueError: on a run that declares no default procedure and margin.
    """
    d = declaration
    if d.default is None or d.margin is None:
        msg = "the run declares no default procedure and margin for the claim to read"
        raise ValueError(msg)
    out = []
    for k in d.rungs:
        default = d.default_at(k)
        assert default is not None  # the default is declared with the margin
        at = [r for r in records if r.k == k and r.environment in d.environments]
        if d.joined is not None and k > 0:
            # a world the joined root does not hold yet is not read
            at = [r for r in at if default in r.arms]
        named = tuple(tested) if k > 0 else ()
        out += [_compared(at, d, d, rival, k, named) for rival in rivals(d)]
        for side in beside:
            for tool in (t for t in side.external if t not in rivals(d)):
                # nor is a world the root beside the run does not hold yet
                held = [r for r in at if r.environment in side.environments and tool in r.arms]
                out.append(_compared(held, d, side, tool, k, named))
    return out


def _compared(
    at: Sequence[WorldRecord],
    d: Declaration,
    root: Declaration,
    rival: str,
    k: int,
    tested: tuple[tuple[str, int], ...],
) -> Comparison:
    """The default of the run `d` declares less `rival` at rung `k`, on its records there, `at`:
    `rival` one of the tools of `root`, the run's own declaration or a root's beside it, read on
    that root's environments and cap."""
    # `compare` reads a run that declares both
    default, margin = d.default_at(k), d.margin
    assert default is not None
    assert margin is not None
    reader: Reader = "hall" if d.reader == "hall" else "student"
    worlds = min(d.worlds, root.worlds_of(rival))
    cap = worlds if rival in (root.caps or {}) else None
    environments = tuple(e for e in d.environments if e in root.environments)
    base = Comparison(
        d.family,
        k,
        rival,
        d.exploratory or k in d.exploratory_rungs or (root.exploratory_caps and cap is not None),
        cap=cap,
        default=default,
        reader=d.reader,
        environments=() if environments == d.environments else environments,
        tested=tuple(place for place in tested if place[0] in environments),
        claimed=(k, rival) not in d.predicted,
    )
    gate = Gate((worlds,), reader=reader)
    held = {r.environment for r in at} == set(environments)
    looks = gate.read(at, default, rival, margin=margin) if held else ()
    if not looks:
        return base
    fallback = paired(at, default, rival, worlds)
    pooled, failures, own, left_out = complete_case(at, default, rival, worlds)
    base = replace(
        base,
        pairs=len(pooled),
        failures=sum(failures.values()),
        default_failures=sum(own.values()),
        left_out=left_out,
        compared=len(left_out) < len(failures),
        fallback_mean=fallback.difference.mean,
        fallback_interval=fallback.difference.interval,
        fallback_within=looks[-1].verdict == "shown",
    )
    if len(pooled) < 2:
        return base
    one = Gate((len(pooled),), reader=reader)
    shown = one.read(pooled, default, rival, margin=margin)
    loss = one.read(pooled, rival, default, margin=0.0)
    c = paired(pooled, default, rival, len(pooled))
    reading = verdict(c)
    if d.per_rung and reader == "hall":
        # one reader a printed verdict: the word read by Hall's, as the within and the level are
        beats = one.read(pooled, default, rival, margin=0.0)[-1].verdict == "shown"
        reading = "loses to" if loss[-1].verdict == "shown" else "beats" if beats else "ties"
    return replace(
        base,
        mean=c.difference.mean,
        interval=c.difference.interval,
        reading=reading,
        within=shown[-1].verdict == "shown",
        level=one.level(loss),
        statistic=shown[-1].statistic,
        within_level=one.level(shown),
    )


def join(
    records: Sequence[WorldRecord], joined: Sequence[WorldRecord], arm: str
) -> list[WorldRecord]:
    """`records` with `arm`'s score of each world the `joined` records hold, paired by
    environment, seed and rung; a world `joined` does not hold is left as it is.

    Raises:
        ValueError: on a joined world that does not score `arm`, a world held twice, an `arm`
            `records` score already, or a world whose built-in arms the two score apart.
    """
    held: dict[tuple[str, int, int], WorldRecord] = {}
    for r in joined:
        place = (r.environment, r.seed, r.k)
        if place in held or arm not in r.arms:
            msg = f"the joined root holds {place} twice or without {arm!r}"
            raise ValueError(msg)
        held[place] = r
    out = []
    for r in records:
        other = held.get((r.environment, r.seed, r.k))
        if other is None:
            out.append(r)
            continue
        if arm in r.arms:
            msg = f"{arm!r} is scored in both roots on {r.environment} world {r.seed} at k = {r.k}"
            raise ValueError(msg)
        apart = [
            name
            for name in BUILT_IN
            if not math.isclose(r.arms[name].regret, other.arms[name].regret, rel_tol=1e-9)
        ]
        if r.family != other.family or r.index != other.index or apart:
            msg = (
                f"{r.environment} world {r.seed} at k = {r.k} is not the joined root's: "
                f"{apart or 'its family or place differs'}"
            )
            raise ValueError(msg)
        out.append(replace(r, arms={**r.arms, arm: other.arms[arm]}))
    return out


def read_run(root: Path) -> tuple[Declaration, list[WorldRecord]]:
    """The run under `root`, its records joined with the declared joined root's, and each
    composite of their arms assembled (`scorecard.composite`). The joined root's arm that plans
    the default above 0 is joined, or where that default is a composite, its router and the arm
    its routed worlds play (`scorecard.composite.parts`); then every other arm of this engine's
    the joined root plans and the run does not, on the worlds that score it, which the claim does
    not read and the reading beside it does (`recorded`).

    Raises:
        ValueError: on a joined root declared otherwise than the run: under another
            pre-registration, another family, other environments or worlds, pilots against scored
            worlds, another bench, a rung above 0 it does not hold, or no arm that plans the
            default above 0, a composite's two arms where it is one; and as `join` does.
    """
    layout = Layout(root)
    d = layout.read()
    records = _records(layout)
    if d.joined is None:
        return d, [compose(r) for r in records]
    # the runner reads the claim, so it is read here only once both are loaded
    from scorecard.run import names

    other = Layout(root.with_name(root.name + d.joined))
    o = other.read()
    if o.preregistration != d.preregistration:
        msg = (
            f"{other.root} runs under the pre-registration {_named(o.preregistration)}, and "
            f"{root} under {_named(d.preregistration)}"
        )
        raise ValueError(msg)
    above = [k for k in d.rungs if k > 0]
    same = (
        o.family,
        o.environments,
        o.worlds,
        o.pilot,
        o.confirmatory,
        o.second_confirmatory,
        o.smoke,
        o.bench,
        o.world_directory,
    ) == (
        d.family,
        d.environments,
        d.worlds,
        d.pilot,
        d.confirmatory,
        d.second_confirmatory,
        d.smoke,
        d.bench,
        d.world_directory,
    )
    assert d.default_above_0 is not None  # a joined root is declared with it
    held = parts(d.default_above_0)
    if not same or not set(above) <= set(o.rungs) or not set(held) <= set(names(o)):
        msg = (
            f"{other.root} does not hold {d.default_above_0!r} of {root}'s worlds at its rungs "
            f"above 0, {above}"
        )
        raise ValueError(msg)
    theirs = [r for r in _records(other) if r.k > 0] if other.scores.exists() else []
    for arm in held:
        records = join(records, theirs, arm)
    own = names(d)
    for arm in (a for a in names(o) if a not in held and a not in own and a not in o.external):
        # read beside the claim only: a world that does not score it leaves it unread there
        records = join(records, [r for r in theirs if arm in r.arms], arm)
    return d, [compose(r) for r in records]


def _named(stamp: Stamp | None) -> str:
    return "none" if stamp is None else f"{stamp.path}, SHA-256 {stamp.sha256}"


def read_beside(
    root: Path, declaration: Declaration, records: Sequence[WorldRecord], side: Path
) -> tuple[Declaration, list[WorldRecord]]:
    """The root `side` beside the run under `root`, and the run's `records` with each arm the root
    plans and the run does not joined by environment, seed and rung (`join`); an arm both plan is
    the run's.

    Raises:
        ValueError: on a root under another pre-registration, or not beside the run: of another
            family, pilots against scored worlds, another bench or directory of worlds, an
            environment the run does not hold, a rung of the run it does not hold, fewer worlds
            than the run's for an arm it adds and does not cap, or no arm the run does not plan;
            and as `join` does.
    """
    # the runner reads the claim, so it is read here only once both are loaded
    from scorecard.run import names

    d = declaration
    layout = Layout(side)
    o = layout.read()
    if o.preregistration != d.preregistration:
        msg = (
            f"{side} runs under the pre-registration {_named(o.preregistration)}, and {root} "
            f"under {_named(d.preregistration)}"
        )
        raise ValueError(msg)
    own = names(d)
    new = [arm for arm in names(o) if arm not in own]
    short = [arm for arm in new if o.worlds < d.worlds and arm not in (o.caps or {})]
    same = (
        o.family,
        o.pilot,
        o.confirmatory,
        o.second_confirmatory,
        o.smoke,
        o.bench,
        o.world_directory,
    ) == (
        d.family,
        d.pilot,
        d.confirmatory,
        d.second_confirmatory,
        d.smoke,
        d.bench,
        d.world_directory,
    )
    if (
        not same
        or not set(o.environments) <= set(d.environments)
        or not set(d.rungs) <= set(o.rungs)
        or short
        or not new
    ):
        msg = (
            f"{side} is not a root beside {root}: one holds the run's family, its pilots or scored "
            "worlds, its bench and directory of worlds, some of its environments at every rung of "
            "the run, and an arm the run does not plan, on the run's worlds or on a cap of each "
            "such arm"
        )
        raise ValueError(msg)
    theirs = [r for r in _records(layout) if r.k in d.rungs] if layout.scores.exists() else []
    joined = list(records)
    for arm in new:
        joined = join(joined, theirs, arm)
    return o, joined


@dataclass(frozen=True, slots=True)
class Run:
    """A run as `claim read` reads it: its declaration, a pool's runs one run over every
    environment they hold (`pooled`); its records, with the arms of its joined root and of the
    roots beside it; the roots `beside` it, each declared over the environments it holds; where a
    pool's runs test at different rungs above 0, each environment's own rung (`tested_rungs`); and
    its `joined` root, declared over the environments it holds, where the run declares one."""

    declaration: Declaration
    records: tuple[WorldRecord, ...]
    beside: tuple[Declaration, ...] = ()
    tested: tuple[tuple[str, int], ...] = ()
    joined: Declaration | None = None

    def comparisons(self) -> list[Comparison]:
        """Every comparison of the run (`compare`)."""
        return compare(self.records, self.declaration, beside=self.beside, tested=self.tested)


def read_runs(roots: Sequence[Path]) -> list[Run]:
    """The runs under `roots` as the claim reads them: each root that declares a margin a run,
    with its joined root's records (`read_run`), and every other root a joined root or a root
    beside a run (`read_beside`), named by the run's directory name, a hyphen and a name of its
    own, the longest such run's; a pool's runs as one run (`pooled`), and the roots of one name
    beside them as one root.

    Raises:
        ValueError: on a root that declares no margin and is neither a run's joined root nor beside
            a run handed, on roots beside some runs of a pool and not the others or declared apart
            (`BESIDE`), and as `read_run`, `read_beside` and `pooled` do.
    """
    read, sides = joined_runs(roots)
    joins = {
        root: {}
        if d.joined is None
        else {d.joined: Layout(root.with_name(root.name + d.joined)).read()}
        for root, d, _ in read
    }
    out = []
    for members in _members(read):
        d, records = _pool(members)
        tested = tested_rungs([m[1] for m in members])
        joined = _pooled_beside(members, joins)
        out.append(Run(d, tuple(records), _pooled_beside(members, sides), tested, *joined))
    return out


def joined_runs(
    roots: Sequence[Path],
) -> tuple[list[tuple[Path, Declaration, list[WorldRecord]]], dict[Path, dict[str, Declaration]]]:
    """Each run under `roots` before its pool is read (`read_runs`): its root, its declaration and
    its records, its joined root's arms and the arms of the roots beside it joined, each world at
    its own rung, with each composite of its arms (`scorecard.composite`); and each run's roots
    beside it, by the name each adds to the run's.

    Raises:
        ValueError: as `read_runs` does, but on a pool.
    """
    declared = {root: Layout(root).read() for root in roots}
    runs = [root for root in roots if declared[root].margin is not None]
    suffixes = {run: declared[run].joined for run in runs}
    joined = {run.with_name(run.name + s).resolve() for run, s in suffixes.items() if s}
    beside: dict[Path, list[Path]] = {run: [] for run in runs}
    for root in roots:
        if root in beside or root.resolve() in joined:
            continue
        held = [
            run
            for run in runs
            if root.name.startswith(f"{run.name}-")
            and root.resolve().parent == run.resolve().parent
        ]
        if not held:
            msg = (
                f"{root} declares no margin and is beside none of the runs handed: a root beside "
                "a run is named by the run's directory name, a hyphen and a name of its own"
            )
            raise ValueError(msg)
        beside[max(held, key=lambda run: len(run.name))].append(root)
    read: list[tuple[Path, Declaration, list[WorldRecord]]] = []
    sides: dict[Path, dict[str, Declaration]] = {}
    for run in runs:
        d, records = read_run(run)
        sides[run] = {}
        for root in beside[run]:
            o, records = read_beside(run, d, records, root)
            sides[run][root.name.removeprefix(run.name)] = o
        read.append((run, d, [compose(r) for r in records]))
    return read, sides


def tested_rungs(declarations: Sequence[Declaration]) -> tuple[tuple[str, int], ...]:
    """Each environment of a pool's runs with its run's rung above 0, where each run holds rung 0
    and one rung above it and those rungs are not all one; () otherwise."""
    above = [tuple(k for k in d.rungs if k > 0) for d in declarations]
    each = all(len(d.rungs) == 2 and len(a) == 1 for d, a in zip(declarations, above, strict=True))
    if not each or len(set(above)) == 1:
        return ()
    return tuple(
        (e, a[0]) for d, a in zip(declarations, above, strict=True) for e in d.environments
    )


def _apart(declarations: Sequence[Declaration], agreed: Sequence[str]) -> list[str]:
    """The fields of `agreed` the declarations do not all declare alike."""
    first = declarations[0]
    return sorted(
        {name for d in declarations for name in agreed if getattr(d, name) != getattr(first, name)}
    )


def pooled(
    runs: Sequence[tuple[Path, Declaration, list[WorldRecord]]],
) -> list[tuple[Declaration, list[WorldRecord]]]:
    """The runs as the claim reads them: each run alone, and the runs of each pool as one run over
    every environment they hold, in the pool's order, their records together; where the pool's
    runs test at different rungs above 0 (`tested_rungs`), each run's records at its rung above 0
    read at the first run's.

    Raises:
        ValueError: on a pool `runs` do not hold whole, runs of a pool under two pre-registrations
            or that declare apart what the claim reads (`AGREED`) other than such rungs, or an
            environment two of them hold.
    """
    return [_pool(members) for members in _members(runs)]


def _members(
    runs: Sequence[tuple[Path, Declaration, list[WorldRecord]]],
) -> list[list[tuple[Path, Declaration, list[WorldRecord]]]]:
    """`runs` as the claim groups them: each run alone, then each pool's runs in the pool's order.

    Raises:
        ValueError: as `pooled` does.
    """
    out: list[list[tuple[Path, Declaration, list[WorldRecord]]]] = []
    pools: dict[tuple[str, ...], dict[str, tuple[Path, Declaration, list[WorldRecord]]]] = {}
    for root, d, records in runs:
        if not d.pool:
            out.append([(root, d, records)])
            continue
        if root.name not in d.pool:
            msg = f"{root} is not one of the runs its pool names, {d.pool}"
            raise ValueError(msg)
        pools.setdefault(d.pool, {})[root.name] = (root, d, records)
    for pool, held in pools.items():
        missing = sorted(set(pool) - set(held))
        if missing:
            msg = f"the pool {pool} is read whole: hand the claim {missing} too"
            raise ValueError(msg)
        members = [held[name] for name in pool]
        declared = [d for _, d, _ in members]
        apart = _apart(declared, AGREED)
        if tested_rungs(declared):
            apart.remove("rungs")
        environments = [e for d in declared for e in d.environments]
        if apart or len(set(environments)) != len(environments):
            msg = (
                f"the runs of the pool {pool} declare alike all but their environments, each its "
                f"own: they differ in {apart or 'none'}, and hold {environments}"
            )
            raise ValueError(msg)
        out.append(members)
    return out


def _pool(
    members: Sequence[tuple[Path, Declaration, list[WorldRecord]]],
) -> tuple[Declaration, list[WorldRecord]]:
    """A run alone as it is, or a pool's runs as one run over every environment they hold, their
    records together, each run's rung above 0 read at the first run's where they test apart."""
    if len(members) == 1:
        ((_, d, records),) = members
        return d, records
    declared = [d for _, d, _ in members]
    one = replace(declared[0], environments=tuple(e for d in declared for e in d.environments))
    if not tested_rungs(declared):
        return one, [r for _, _, records in members for r in records]
    # each run holds rung 0 and one rung above it, its largest
    label = max(declared[0].rungs)
    out: list[WorldRecord] = []
    for _, d, records in members:
        own = max(d.rungs)
        out += [replace(r, k=label) if r.k == own else r for r in records]
    return one, out


def _pooled_beside(
    members: Sequence[tuple[Path, Declaration, list[WorldRecord]]],
    sides: Mapping[Path, Mapping[str, Declaration]],
) -> tuple[Declaration, ...]:
    """The roots beside the runs `members` holds, by name, the roots of one name read as one root
    over every environment they hold.

    Raises:
        ValueError: on a name beside some runs of a pool and not the others, or roots of one name
            that declare apart what the claim reads of them (`BESIDE`).
    """
    named = list(dict.fromkeys(name for root, _, _ in members for name in sides[root]))
    missing = sorted(
        f"{root.name}{name}" for root, _, _ in members for name in named if name not in sides[root]
    )
    if missing:
        msg = f"the pool {members[0][1].pool} is read whole: hand the claim {missing} too"
        raise ValueError(msg)
    out = []
    for name in named:
        held = [sides[root][name] for root, _, _ in members]
        apart = _apart(held, BESIDE)
        if apart:
            msg = (
                f"the roots {[root.name + name for root, _, _ in members]} beside one pool's runs "
                f"declare alike all but their environments: they differ in {apart}"
            )
            raise ValueError(msg)
        environments = tuple(e for o in held for e in o.environments)
        out.append(held[0] if len(held) == 1 else replace(held[0], environments=environments))
    return tuple(out)


def losses(comparisons: Sequence[Comparison]) -> frozenset[Key]:
    """The losses Holm's step-down shows at familywise `ALPHA` over the claim's comparisons."""
    return holm({c.key: c.level for c in comparisons if not c.exploratory and c.compared})


def holds(comparisons: Sequence[Comparison]) -> bool:
    """Whether the claim holds: it has some comparison, and every one, an unread one too, is
    within."""
    claimed = [c for c in comparisons if not c.exploratory and c.compared]
    return bool(claimed) and all(c.within for c in claimed)


@dataclass(frozen=True, slots=True)
class Verdict:
    """The claim at one rung of one family (`verdicts`): an intersection-union test over the
    comparisons it claims, read at `level`, the largest of their levels within the margin; `within`
    where every comparison it claims is within, unadjusted; the rivals it leaves out, `predicted`
    to fail; `holds` where the fixed sequence reaches it and it is shown at `ALPHA`; and `stopped`,
    the family and rung the sequence stopped at, where that came before it."""

    family: int
    k: int
    rivals: tuple[str, ...]
    level: float
    within: bool
    holds: bool
    predicted: tuple[str, ...]
    stopped: tuple[int, int] | None = None


def verdicts(comparisons: Sequence[Comparison], order: Sequence[tuple[int, int]]) -> list[Verdict]:
    """The claim by family and rung (ADR 0081): at each rung of each family read, the comparisons
    in the claim that the run does not predict to fail, one intersection-union test (Berger 1982)
    whose level is the largest of theirs. The verdicts are tested in the fixed sequence the run
    declares, `order`, each at `ALPHA`: each holds where its level is at most `ALPHA` and every one
    before it held, and the first that does not ends the sequence, every one after it read and not
    confirmed. The familywise level is `ALPHA` with no adjustment, since a verdict is tested only
    once every one before it held. A verdict the order names that is not read or claims no
    comparison is not shown, and a comparison left unread holds the level 1, so neither can hold.
    The verdicts come in the order, then those that claim nothing and the order does not name.

    Raises:
        ValueError: on a verdict that claims a comparison and that `order` does not name.
    """
    groups: dict[tuple[int, int], list[Comparison]] = {}
    for c in comparisons:
        if not c.exploratory and c.compared:
            groups.setdefault((c.family, c.k), []).append(c)
    levels = {
        key: max(c.within_level for c in held if c.claimed)
        for key, held in groups.items()
        if any(c.claimed for c in held)
    }
    unnamed = sorted(set(levels) - set(order))
    if unnamed:
        named = ", ".join(f"family {f} rung {k}" for f, k in unnamed)
        msg = f"the declared order of verdicts does not name {named}, which the claim reads"
        raise ValueError(msg)

    def verdict(key: tuple[int, int], holds: bool, stopped: tuple[int, int] | None) -> Verdict:
        held = groups.get(key, [])
        return Verdict(
            key[0],
            key[1],
            tuple(c.rival for c in held if c.claimed),
            levels.get(key, 1.0),
            all(c.within for c in held if c.claimed) and key in levels,
            holds,
            tuple(c.rival for c in held if not c.claimed),
            stopped,
        )

    out = []
    stopped: tuple[int, int] | None = None
    for key in order:
        holds = stopped is None and levels.get(key, 1.0) <= ALPHA
        out.append(verdict(key, holds, stopped))
        if stopped is None and not holds:
            stopped = key
    return out + [verdict(key, False, None) for key in groups if key not in order]


def verdict_lines(
    comparisons: Sequence[Comparison], margin: float, order: Sequence[tuple[int, int]]
) -> list[str]:
    """The claim by family and rung: each verdict in the declared order, then those that claim
    nothing, then each family and rung read only as exploratory (an exploratory run's, or a rung
    the run declares exploratory), then the headline, which is said only where every family and
    rung read is claimed, none of them exploratory, and every verdict holds at `ALPHA`, an
    intersection-union test with no adjustment. A tool read on its cap and exploratory is read
    beside its verdict and does not stop the headline."""
    held = verdicts(comparisons, order)
    out = [
        f"the claim by family and rung, each an intersection-union test over its claimed rivals "
        f"within {margin:g}, the {len(order)} verdicts tested at {ALPHA:g} in the declared order, "
        "the first that does not hold ending the sequence:"
    ]
    for v in held:
        head = f"  family {v.family}, k = {v.k}: "
        left = f"; not claimed, predicted to fail: {', '.join(v.predicted)}" if v.predicted else ""
        word = "holds" if v.holds else "does not hold"
        if v.stopped is not None:
            word = (
                f"not confirmed: the sequence stopped at family {v.stopped[0]} rung {v.stopped[1]}"
            )
        elif not v.holds and (v.family, v.k) in order:
            left += "; the sequence stops here"
        if not v.rivals:
            after = f"; {word}" if v.stopped is not None else ""
            out.append(f"{head}{'not claimed' if v.predicted else 'unread'}{left}{after}")
            continue
        out.append(
            f"{head}{word} (level {v.level:.3g}; {len(v.rivals)} rivals, "
            f"{'each within' if v.within else 'not each within'} unadjusted){left}"
        )
    shown = {(v.family, v.k) for v in held}
    # a family and rung with no verdict whose comparisons are exploratory: the run or the rung is
    explored = [
        key
        for key in dict.fromkeys((c.family, c.k) for c in comparisons if c.exploratory)
        if key not in shown
    ]
    out += [f"  family {f}, k = {k}: exploratory, read and not claimed" for f, k in explored]
    whole = bool(held) and not explored and all(v.holds and not v.predicted for v in held)
    out.append(
        "the headline holds: within the margin of every rival on every family and rung"
        if whole
        else "no headline: some family or rung is not claimed whole or does not hold"
    )
    return out


def lines(
    comparisons: Sequence[Comparison], margin: float, *, order: Sequence[tuple[int, int]] = ()
) -> list[str]:
    """Every comparison's verdicts, the claim's and Holm's list of losses; with an `order` of
    verdicts, the claim by family and rung (`verdict_lines`) in place of the claim over every
    comparison."""
    out = []
    named = len({c.default for c in comparisons}) > 1
    for c in comparisons:
        head = f"  family {c.family}, k = {c.k}, less {c.rival:22s}"
        if named:
            head = f"  family {c.family}, k = {c.k}, {c.default} less {c.rival:22s}"
        on = f"; on {_listed(c.environments)} only" if c.environments else ""
        if c.cap is not None:
            on += f"; on its cap, {c.cap} worlds an environment"
        if c.reader == "hall":
            on += "; by Hall's reader, the interval Student's"
        if c.tested:
            on += f"; the tested rung: {_rungs(c.tested)}"
        if c.fallback_interval is None:
            out.append(f"{head} unread: the run does not hold its worlds whole{on}")
            continue
        cases = (
            f"; {c.pairs} complete cases, the rival failed {c.failures}, the default "
            f"{c.default_failures}"
            + (f", not compared in {', '.join(c.left_out)}" if c.left_out else "")
        )
        low, high = c.fallback_interval
        fallback = (
            f"; the status quo for a failed plan: {c.fallback_mean:+.4f} "
            f"[{low:+.4f}, {high:+.4f}], {'within' if c.fallback_within else 'not within'}"
        )
        if not c.compared:
            out.append(f"{head} not compared: the rival failed on most worlds{cases}{fallback}{on}")
            continue
        if c.interval is None:
            out.append(f"{head} unread: fewer than two complete cases{cases}{fallback}{on}")
            continue
        low, high = c.interval
        out.append(
            f"{head} {c.mean:+.4f} [{low:+.4f}, {high:+.4f}]; "
            f"{'within' if c.within else 'not within'} {margin:g}; {c.reading}, unadjusted "
            f"(level {c.level:.3g}){cases}{fallback}{on}"
            + ("; exploratory" if c.exploratory else "")
            + ("" if c.claimed else "; not claimed: predicted to fail")
        )
    claimed = [c for c in comparisons if not c.exploratory and c.compared]
    outside = [c for c in claimed if not c.within]
    if order:
        out += verdict_lines(comparisons, margin, order)
    elif holds(comparisons):
        out.append(
            f"the claim holds: {len(claimed)} of {len(claimed)} comparisons within {margin:g}"
        )
    else:
        named = ", ".join(f"family {c.family} k {c.k} {c.rival}" for c in outside) or "none read"
        out.append(f"the claim does not hold: {len(outside)} of {len(claimed)} not within: {named}")
    lost = sorted(losses(comparisons))
    named = ", ".join(f"family {f} k {k} {rival}" for f, k, rival in lost) or "none"
    out.append(
        f"losses by Holm's step-down at familywise {ALPHA:g} over {len(claimed)} comparisons: "
        f"{named}"
    )
    return out


def secondary_lines(runs: Sequence[Run]) -> list[str]:
    """What the claim by rung reports beside it and never reads (ADR 0081): its comparisons with
    every plan whose fit, at any stage, is past the bounds at the status quo's regret
    (`scorecard.report.gated`); its comparisons at the margin `NARROW`; and at rung 0 of each
    family the gain the worlds hold, the status quo's regret, which the best plan's (0) is less,
    and Meridian's plan less the status quo's, each on the worlds read; then the default less each
    variant a run names (`against_default`)."""
    # the report reads the claim, so it is read here only once both are loaded
    from causaldyn_bench.budget_regret import Mean

    from scorecard.report import STATUS_QUO, gated

    def mean(values: Sequence[float]) -> str:
        if len(values) < 2:
            return "unread: fewer than two worlds"
        m = Mean.of(np.asarray(values, dtype=float))
        low, high = m.interval
        return f"{m.mean:+.4f} [{low:+.4f}, {high:+.4f}] on {len(values)} worlds"

    out = ["secondary readings, reported beside the claim and never read by it:"]
    for name, change in (
        ("every stage of every fit past the bounds at the status quo", None),
        (f"at the margin {NARROW:g}", NARROW),
    ):
        read = []
        for run in runs:
            d = run.declaration if change is None else replace(run.declaration, margin=change)
            records = run.records if change is not None else tuple(gated(run.records))
            read += compare(records, d, beside=run.beside, tested=run.tested)
        claimed = [c for c in read if not c.exploratory and c.compared and c.claimed]
        outside = [f"family {c.family} k {c.k} {c.rival}" for c in claimed if not c.within]
        out.append(
            f"  {name}: {len(claimed) - len(outside)} of {len(claimed)} claimed comparisons "
            f"within; not within: {', '.join(outside) or 'none'}"
        )
    for run in runs:
        d = run.declaration
        if 0 not in d.rungs or d.exploratory:
            continue
        at = [r for r in run.records if r.k == 0 and r.environment in d.environments]
        # the best plan's regret is 0, so it less the status quo is the status quo's, negated
        best = [-r.arms[STATUS_QUO].regret for r in at if r.arms[STATUS_QUO].regret is not None]
        line = f"  family {d.family}, k = 0: the best plan less the status quo {mean(best)}"
        gains = [
            r.arms[MERIDIAN].regret - r.arms[STATUS_QUO].regret  # type: ignore[operator]
            for r in at
            if not _failed(r, MERIDIAN) and r.arms[MERIDIAN].regret is not None
        ]
        if any(MERIDIAN in r.arms for r in at):
            line += f"; {MERIDIAN} less the status quo {mean(gains)}"
        out.append(line)
    return out + against_default(runs)


def against_default(runs: Sequence[Run]) -> list[str]:
    """Each variant a run reads its default at rung 0 against (`Declaration.default_less`): the
    default less it on the complete cases, a line a run, read as `recorded` reads an arm; then,
    for each default and variant, a line pooled over the confirmatory runs that read it: the mean
    over their complete cases and its 95 % interval, the variance stratified by run."""
    # the runner reads the claim, so it is read here only once both are loaded
    from scorecard.run import names

    out: list[str] = []
    strata: dict[tuple[str, str], list[tuple[int, np.ndarray]]] = {}
    for run in runs:
        d = run.declaration
        default = d.default
        if default is None:  # a run that names variants declares its default
            continue
        planned = {
            arm: root
            for root in (d, *run.beside)
            for arm in names(root)
            if arm not in root.external
        }
        for variant in d.default_less:
            read, pairs = _recorded_cases(
                run, 0, (default, d), (variant, planned.get(variant, d)), ()
            )
            head = f"  family {d.family}, k = 0, {default} less {variant}"
            if read.pairs is None:
                out.append(f"{head}: unread: the run does not hold its worlds whole")
                continue
            cases = (
                f"{read.pairs} complete cases; {variant} failed {read.failures}, "
                f"{default} {read.arm_failures}"
            )
            if read.interval is None:
                out.append(f"{head}: unread: fewer than two complete cases, {cases}")
                continue
            low, high = read.interval
            out.append(f"{head}: {read.mean:+.4f} [{low:+.4f}, {high:+.4f}] on {cases}")
            if d.confirmatory:
                gaps = [_regret(r, default) - _regret(r, variant) for r in pairs]
                strata.setdefault((default, variant), []).append(
                    (d.family, np.asarray(gaps, dtype=float))
                )
    for (default, variant), held in strata.items():
        mean, low, high, n = stratified(held)
        families = _listed([str(f) for f in sorted({f for f, _ in held})])  # 7a and 7b are one
        out.append(
            f"  pooled over the confirmatory runs at k = 0 (families {families}), {default} "
            f"less {variant}: {mean:+.4f} [{low:+.4f}, {high:+.4f}] on {n} complete cases, "
            "the variance stratified by run"
        )
    return out


def stratified(strata: Sequence[tuple[int, np.ndarray]]) -> tuple[float, float, float, int]:
    """The mean of every value of `strata` (each a family and its paired differences, two or
    more), its 95 % interval on the normal quantile with the variance stratified by stratum,
    `sum_f n_f s_f^2 / N^2` (`s_f` on divisor `n_f - 1`), and `N`."""
    values = [v for _, v in strata]
    n = sum(len(v) for v in values)
    mean = float(sum(v.sum() for v in values) / n)
    se = math.sqrt(sum(len(v) * v.var(ddof=1) for v in values)) / n
    z = float(norm.isf(ALPHA))
    return mean, mean - z * se, mean + z * se, n


def _listed(names: Sequence[str]) -> str:
    """`names` as a list in prose: `a`, `a and b`, `a, b and c`."""
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _rungs(tested: Sequence[tuple[str, int]]) -> str:
    """Each environment's rung, the environments of one rung together in their order:
    `k = 1 on L-52, 2 on L-78 and L-104`."""
    at: dict[int, list[str]] = {}
    for environment, k in tested:
        at.setdefault(k, []).append(environment)
    return "k = " + ", ".join(f"{k} on {_listed(names)}" for k, names in at.items())


@dataclass(frozen=True, slots=True)
class Recorded:
    """One arm less another at one rung of one run, regret per euro, read outside the claim
    (`recorded`): the mean and paired 95 % Student-t interval of `arm` less `rival` over the
    complete cases of the environments both are read on, `pairs` of them, and the reading rule's
    verdict, unadjusted; None where the run does not hold the look whole or fewer than two complete
    cases are left. `failures` and `arm_failures` are the rival's and the arm's failed plans over
    the look, `left_out` the environments not compared (`complete_case`); `environments`, `cap` and
    `tested` are as a comparison's."""

    family: int
    k: int
    arm: str
    rival: str
    mean: float | None = None
    interval: tuple[float, float] | None = None
    reading: str | None = None
    pairs: int | None = None
    failures: int | None = None
    arm_failures: int | None = None
    left_out: tuple[str, ...] = ()
    cap: int | None = None
    environments: tuple[str, ...] = ()
    tested: tuple[tuple[str, int], ...] = ()


def recorded(run: Run) -> list[Recorded]:
    """What the claim does not read of the run, rung by rung: each recorded arm, an arm of this
    engine's, the run's, its joined root's at the rungs above 0 or a root's beside it, that is not
    the rung's default, less every rival, the run's (`rivals`) and the tools of the roots beside
    it; then, at each rung above 0 where the run declares its default there, that default less the
    engine's static arm (`STATIC`), where one of those arms is it. Each arm is read on its root's
    environments and worlds, a capped tool on its cap, as `compare` reads a tool of a root beside
    the run."""
    # the runner reads the claim, so it is read here only once both are loaded
    from scorecard.run import names

    d = run.declaration
    arms: dict[str, tuple[Declaration, tuple[int, ...]]] = {}
    against: dict[str, Declaration] = dict.fromkeys(rivals(d), d)
    for root in (d, *run.beside):
        arms |= {
            a: (root, d.rungs) for a in names(root) if a not in root.external and a not in arms
        }
        against |= {tool: root for tool in root.external if tool not in against}
    j = run.joined
    if j is not None:
        above = tuple(k for k in d.rungs if k > 0)
        arms |= {a: (j, above) for a in names(j) if a not in j.external and a not in arms}
    out = []
    for k in d.rungs:
        tested = run.tested if k > 0 else ()
        out += [
            _recorded(run, k, (arm, mine), (rival, theirs), tested)
            for arm, (mine, held) in arms.items()
            if k in held and arm != d.default_at(k)
            for rival, theirs in against.items()
            if set(mine.environments) & set(theirs.environments)
        ]
    if d.default_above_0 is not None and STATIC in arms:
        out += [
            _recorded(run, k, (d.default_above_0, d), (STATIC, arms[STATIC][0]), run.tested)
            for k in d.rungs
            if k > 0
        ]
    return out


def _recorded(
    run: Run,
    k: int,
    first: tuple[str, Declaration],
    second: tuple[str, Declaration],
    tested: tuple[tuple[str, int], ...],
) -> Recorded:
    """`first`'s arm less `second`'s at rung `k` of `run`, each given with the root that plans it,
    on the environments both roots hold, at the fewest worlds an environment either root reads."""
    return _recorded_cases(run, k, first, second, tested)[0]


def _recorded_cases(
    run: Run,
    k: int,
    first: tuple[str, Declaration],
    second: tuple[str, Declaration],
    tested: tuple[tuple[str, int], ...],
) -> tuple[Recorded, list[WorldRecord]]:
    """`_recorded`'s reading and the complete cases it is read on, none where it is unread."""
    d = run.declaration
    (arm, mine), (rival, theirs) = first, second
    worlds = min(d.worlds, mine.worlds_of(arm), theirs.worlds_of(rival))
    environments = tuple(
        e for e in d.environments if e in mine.environments and e in theirs.environments
    )
    base = Recorded(
        d.family,
        k,
        arm,
        rival,
        cap=worlds if rival in (theirs.caps or {}) else None,
        environments=() if environments == d.environments else environments,
        tested=tuple(place for place in tested if place[0] in environments),
    )
    at = [
        r
        for r in run.records
        if r.k == k and r.environment in environments and arm in r.arms and rival in r.arms
    ]
    held = {(r.environment, r.index) for r in at}
    if any((e, i) not in held for e in environments for i in range(worlds)):
        return base, []
    pooled, failures, own, left_out = complete_case(at, arm, rival, worlds)
    base = replace(
        base,
        pairs=len(pooled),
        failures=sum(failures.values()),
        arm_failures=sum(own.values()),
        left_out=left_out,
    )
    if len(pooled) < 2:
        return base, pooled
    c = paired(pooled, arm, rival, len(pooled))
    read = replace(base, mean=c.difference.mean, interval=c.difference.interval, reading=verdict(c))
    return read, pooled


def recorded_lines(readings: Sequence[Recorded]) -> list[str]:
    """The readings `recorded` makes, a line each, below a line that says what they are."""
    out = [
        "reported regardless, outside the claim and Holm's list: each recorded arm less every "
        f"rival, and the default above rung 0 less {STATIC}, regret per euro on the complete "
        "cases, unadjusted"
    ]
    for r in readings:
        head = f"  family {r.family}, k = {r.k}, {f'{r.arm} less {r.rival}':46s}"
        on = f"; on {_listed(r.environments)} only" if r.environments else ""
        if r.cap is not None:
            on += f"; on its cap, {r.cap} worlds an environment"
        if r.tested:
            on += f"; the tested rung: {_rungs(r.tested)}"
        if r.pairs is None:
            out.append(f"{head} unread: the run does not hold its worlds whole{on}")
            continue
        cases = (
            f"; {r.pairs} complete cases, the rival failed {r.failures}, the arm {r.arm_failures}"
            + (f", not compared in {', '.join(r.left_out)}" if r.left_out else "")
        )
        if r.interval is None:
            out.append(f"{head} unread: fewer than two complete cases{cases}{on}")
            continue
        low, high = r.interval
        out.append(
            f"{head} {r.mean:+.4f} [{low:+.4f}, {high:+.4f}]; {r.reading}, unadjusted{cases}{on}"
        )
    return out


@dataclass(frozen=True, slots=True)
class Size:
    """The worlds `family`'s claim needs: the largest paired SD on its pilot, `sd`, of the default
    less `rival` at rung `k`, and the fewest worlds in tens, `needed`, at which it reaches `POWER`
    at a true tie; over `environments` environments."""

    family: int
    environments: int
    rival: str
    k: int
    sd: float
    needed: int

    @property
    def exploratory(self) -> bool:
        return self.needed > CAP

    @property
    def worlds(self) -> int:
        """Each environment's worlds the run declares: the needed ones, at most `CAP`, shared."""
        return math.ceil(min(self.needed, CAP) / self.environments)

    def power(self, margin: float) -> float:
        """The chance the binding comparison is within `margin` at a true tie on the run's worlds,
        by the bench's gate at one look."""
        n = self.worlds * self.environments
        if self.sd == 0.0:
            return 1.0
        return Gate((n,)).crossings(margin * math.sqrt(n) / self.sd).shown[-1]


def needed(sd: float, margin: float) -> int:
    """The fewest worlds, in tens, at which a comparison of paired SD `sd` reaches `POWER` at a true
    tie against `margin`."""
    return TENS * max(1, math.ceil((Z * sd / margin) ** 2 / TENS))


def size(
    records: Sequence[WorldRecord], declaration: Declaration, default: str, margin: float
) -> Size:
    """The worlds the family of a pilot's `records` needs for the claim on `default` at `margin`.

    Raises:
        ValueError: on a pilot that misses a world it declares, or a rival a world does not score.
    """
    d = declaration
    spread: dict[tuple[str, int], float] = {}
    for k in d.rungs:
        at = [r for r in records if r.k == k and r.environment in d.environments]
        missing = sorted(set(d.environments) - {r.environment for r in at})
        if missing:
            msg = f"the pilot holds no world of {missing} at k = {k}"
            raise ValueError(msg)
        for rival in rivals(d):
            spread[rival, k] = paired(at, default, rival, d.worlds).difference.sd
    (rival, k), sd = max(spread.items(), key=lambda item: item[1])
    return Size(d.family, len(d.environments), rival, k, sd, needed(sd, margin))


@dataclass(frozen=True, slots=True)
class Differences:
    """A pilot's paired differences, the default less each rival at each rung, regret per euro:
    `comparisons`, each a (rival, rung), and for each environment a matrix of its worlds by the
    comparisons, a row a world."""

    comparisons: tuple[tuple[str, int], ...]
    strata: tuple[np.ndarray, ...]


def differences(
    records: Sequence[WorldRecord], declaration: Declaration, default: str
) -> Differences:
    """Every comparison's paired differences on a pilot's declared worlds, world by world.

    Raises:
        ValueError: on a pilot that misses a world it declares, or an arm a world does not score.
    """
    d = declaration
    comparisons = tuple((rival, k) for k in d.rungs for rival in rivals(d))
    columns: list[np.ndarray] = []
    places: list[str] = []
    for k in d.rungs:
        at = [r for r in records if r.k == k and r.environment in d.environments]
        missing = sorted(set(d.environments) - {r.environment for r in at})
        if missing:
            msg = f"the pilot holds no world of {missing} at k = {k}"
            raise ValueError(msg)
        worlds = look(at, d.worlds)
        places = [r.environment for r in worlds]
        mine = np.array([_regret(r, default) for r in worlds])
        columns += [mine - np.array([_regret(r, rival) for r in worlds]) for rival in rivals(d)]
    matrix = np.column_stack(columns)
    where = np.array(places)
    return Differences(comparisons, tuple(matrix[where == e] for e in sorted(set(d.environments))))


def _regret(record: WorldRecord, arm: str) -> float:
    score = record.arms.get(arm)
    if score is None or score.regret is None:
        msg = f"{arm}'s regret is not scored on {record.environment} world {record.index}"
        raise ValueError(msg)
    return score.regret


def upper_sd(sd: float, worlds: int, confidence: float = POWER) -> float:
    """The one-sided `confidence` upper confidence limit of an SD `sd` estimated on `worlds`
    values, normal theory: `sd sqrt((worlds - 1) / chi2_(1 - confidence, worlds - 1))`, the
    chi-square's lower `1 - confidence` quantile on `worlds - 1` degrees of freedom.

    Raises:
        ValueError: on fewer than 2 values or a confidence outside (0, 1).
    """
    if worlds < 2 or not 0.0 < confidence < 1.0:
        msg = (
            f"an SD's upper limit needs 2 values and a confidence in (0, 1): {worlds}, {confidence}"
        )
        raise ValueError(msg)
    return sd * math.sqrt((worlds - 1) / float(chi2.ppf(1.0 - confidence, worlds - 1)))


def _caps(pilot: Differences, capped: Mapping[str, int] | None) -> np.ndarray | None:
    """Each comparison's most worlds an environment, its rival's cap or infinity; None where no
    rival is capped.

    Raises:
        ValueError: on a cap for a rival the pilot does not read, or a cap below 2 worlds.
    """
    if not capped:
        return None
    unknown = sorted(set(capped) - {rival for rival, _ in pilot.comparisons})
    short = {tool: worlds for tool, worlds in capped.items() if worlds < 2}
    if unknown or short:
        msg = f"a cap is 2 worlds or more for a rival the pilot reads: {unknown or short}"
        raise ValueError(msg)
    return np.array([capped.get(rival, math.inf) for rival, _ in pilot.comparisons])


def _claimed(caps: np.ndarray | None, exploratory: bool) -> np.ndarray | slice:
    """The comparisons the joint chance reads: all, or the uncapped where the capped are
    exploratory.

    Raises:
        ValueError: where every comparison is capped and exploratory.
    """
    if caps is None or not exploratory:
        return slice(None)
    kept = np.isinf(caps)
    if not kept.any():
        msg = "every comparison is capped and exploratory: the claim reads none"
        raise ValueError(msg)
    return kept


def _passes(
    pilot: Differences,
    margin: float,
    resamples: int,
    seed: int,
    scale: float,
    caps: np.ndarray | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Each count of worlds an environment, from 1 on, with whether each resample passes each
    comparison there: within `margin` by the gate's one-look reading, Student's t on n - 1
    degrees of freedom. A count's resamples extend the count's before it, so every count reads the
    same draws; each environment draws from its own spawned stream. A comparison with a cap in
    `caps` keeps, past its cap, its reading on the cap's prefix of the draws."""
    centre = np.mean([stratum.mean(axis=0) for stratum in pilot.strata], axis=0)
    strata = [scale * (stratum - centre) for stratum in pilot.strata]
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(len(strata))]
    total = np.zeros((resamples, len(pilot.comparisons)))
    squares = np.zeros_like(total)
    held: np.ndarray | None = None
    count = 0
    while True:
        count += 1
        for stratum, rng in zip(strata, streams, strict=True):
            drawn = stratum[rng.integers(0, stratum.shape[0], resamples)]
            total += drawn
            squares += drawn * drawn
        n = count * len(strata)
        if n < 2:
            continue
        mean = total / n
        se = np.sqrt(np.maximum(squares - n * mean * mean, 0.0) / (n - 1) / n)
        gap = margin - mean
        with np.errstate(divide="ignore", invalid="ignore"):
            statistic = np.where(se > 0.0, gap / se, np.where(gap > 0.0, np.inf, 0.0))
        passed = statistic >= student.isf(ALPHA, n - 1)
        if caps is not None and held is not None:
            passed = np.where(count <= caps, passed, held)
        held = passed
        yield count, passed


def chances(
    pilot: Differences,
    margin: float,
    worlds: int,
    *,
    resamples: int = RESAMPLES,
    seed: int = SEED,
    scale: float = 1.0,
    capped: Mapping[str, int] | None = None,
    exploratory: bool = False,
) -> tuple[tuple[float, ...], float]:
    """Each comparison's chance at a true tie to be within `margin` on `worlds` worlds an
    environment, a capped rival's on its cap, and every comparison's together, the capped ones
    left out where they are `exploratory`, by `joint_size`'s bootstrap.

    Raises:
        ValueError: as `_caps` and `_claimed` do.
    """
    caps = _caps(pilot, capped)
    claimed = _claimed(caps, exploratory)
    for count, passed in _passes(pilot, margin, resamples, seed, scale, caps):
        if count >= worlds:
            return (
                tuple(float(c) for c in passed.mean(axis=0)),
                float(passed[:, claimed].all(axis=1).mean()),
            )
    raise AssertionError  # _passes never ends


@dataclass(frozen=True, slots=True)
class JointSize:
    """The worlds `family`'s claim needs on every comparison together: the fewest worlds in tens,
    `needed`, at which all `comparisons`, each a (rival, rung), are within the margin at a true tie
    with chance `joint` at least `POWER`, each with chance `chances`; over `environments`
    environments, the pilot's differences raised by `scale` (1 where they are read as they are).
    The `capped` tools, each with its worlds an environment, are read on their caps, left out of
    `joint` and `needed` where they are `exploratory_caps`; `whole` is every comparison's chance
    together at `needed`, the capped ones on their caps (`joint` where none is left out)."""

    family: int
    environments: int
    comparisons: tuple[tuple[str, int], ...]
    needed: int
    chances: tuple[float, ...]
    joint: float
    scale: float = 1.0
    capped: tuple[tuple[str, int], ...] = ()
    exploratory_caps: bool = False
    whole: float | None = None

    @property
    def exploratory(self) -> bool:
        return self.needed > CAP

    @property
    def worlds(self) -> int:
        """Each environment's worlds the run would declare: the needed ones, at most `CAP`."""
        return math.ceil(min(self.needed, CAP) / self.environments)


def joint_needed(
    pilot: Differences,
    margin: float,
    *,
    power: float = POWER,
    resamples: int = RESAMPLES,
    seed: int = SEED,
    scale: float = 1.0,
    capped: Mapping[str, int] | None = None,
    exploratory: bool = False,
) -> tuple[int, tuple[float, ...], float]:
    """The fewest worlds, in tens, at which every comparison of `pilot` is within `margin` with
    chance at least `power` at a true tie, with each comparison's chance and the joint one there;
    a capped rival's comparisons read on its cap, and left out of the joint chance where they are
    `exploratory`.

    Raises:
        ValueError: as `_caps` and `_claimed` do, and where no count up to `MOST` worlds reaches
            `power`.
    """
    environments = len(pilot.strata)
    caps = _caps(pilot, capped)
    claimed = _claimed(caps, exploratory)
    reached: tuple[int, tuple[float, ...], float] | None = None
    target = TENS
    for count, passed in _passes(pilot, margin, resamples, seed, scale, caps):
        while math.ceil(target / environments) == count:
            joint = float(passed[:, claimed].all(axis=1).mean())
            if joint >= power:
                reached = (target, tuple(float(c) for c in passed.mean(axis=0)), joint)
                break
            target += TENS
        if reached is not None:
            return reached
        if target > MOST:
            msg = f"no count up to {MOST} worlds passes every comparison with chance {power}"
            if caps is not None and not exploratory:
                bound = float(passed[:, np.isfinite(caps)].all(axis=1).mean())
                msg += f": the capped comparisons together pass with chance {bound:.3f} at most"
            raise ValueError(msg)
    raise AssertionError  # _passes never ends


def joint_size(
    records: Sequence[WorldRecord],
    declaration: Declaration,
    default: str,
    margin: float,
    *,
    power: float = POWER,
    resamples: int = RESAMPLES,
    seed: int = SEED,
    confidence: float | None = None,
    capped: Mapping[str, int] | None = None,
    exploratory: bool = False,
) -> JointSize:
    """The worlds the family of a pilot's `records` needs for every comparison of the claim on
    `default` at `margin` together; with `confidence`, its differences raised to their SDs'
    `confidence` upper limits on the pilot's worlds (`upper_sd`); the `capped` tools' comparisons
    read on their caps, `exploratory` or in the claim.

    Raises:
        ValueError: as `differences`, `upper_sd` and `joint_needed` do.
    """
    pilot = differences(records, declaration, default)
    worlds = sum(stratum.shape[0] for stratum in pilot.strata)
    scale = 1.0 if confidence is None else upper_sd(1.0, worlds, confidence)
    needed, each, joint = joint_needed(
        pilot,
        margin,
        power=power,
        resamples=resamples,
        seed=seed,
        scale=scale,
        capped=capped,
        exploratory=exploratory,
    )
    whole = joint
    if capped and exploratory:
        _, whole = chances(
            pilot,
            margin,
            math.ceil(needed / len(pilot.strata)),
            resamples=resamples,
            seed=seed,
            scale=scale,
            capped=capped,
        )
    return JointSize(
        declaration.family,
        len(pilot.strata),
        pilot.comparisons,
        needed,
        each,
        joint,
        scale,
        tuple(sorted((capped or {}).items())),
        bool(capped) and exploratory,
        whole,
    )


def each_family(families: int, power: float = POWER) -> float:
    """The joint chance each of `families` independent families needs for their claims together to
    reach `power`: `power^(1/families)`.

    Raises:
        ValueError: on fewer families than 1 or a power outside (0, 1).
    """
    if families < 1 or not 0.0 < power < 1.0:
        msg = f"a claim needs a family and a power in (0, 1): {families}, {power}"
        raise ValueError(msg)
    return power ** (1.0 / families)


def overall(joints: Iterable[float]) -> float:
    """The launch claim's chance at a true tie over families whose readings are independent, each
    with its joint chance in `joints`: their product."""
    return math.prod(joints)


def _records(layout: Layout) -> list[WorldRecord]:
    return [
        WorldRecord.from_json(json.loads(line)) for line in layout.scores.read_text().splitlines()
    ]


def _joint_lines(
    records: Sequence[WorldRecord], d: Declaration, args: argparse.Namespace, rule: Size
) -> tuple[list[str], float, float]:
    """The joint size's lines, its joint chance and the joint chance at the binding rule's `N`: the
    pilot's declared caps and `--cap`'s read, `--cap` replacing a declared cap of the same tool."""
    power = each_family(args.families)
    capped = {**(d.caps or {}), **args.cap}
    exploratory = args.exploratory_caps or d.exploratory_caps
    j = joint_size(
        records,
        d,
        args.default,
        args.margin,
        power=power,
        resamples=args.resamples,
        seed=args.seed,
        confidence=args.confidence,
        capped=capped,
        exploratory=exploratory,
    )
    pilot = differences(records, d, args.default)
    each = math.ceil(rule.needed / rule.environments)
    _, at_rule = chances(
        pilot,
        args.margin,
        each,
        resamples=args.resamples,
        seed=args.seed,
        capped=capped,
        exploratory=exploratory,
    )
    out = [
        f"family {j.family}: joint N {j.needed}"
        + (f", above {CAP}: exploratory" if j.exploratory else "")
        + f"; every comparison within {args.margin:g} at a true tie with chance {j.joint:.3f} "
        f"({args.resamples} resamples, seed {args.seed}); at the binding rule's "
        f"N {rule.needed} ({each * rule.environments} worlds), {at_rule:.3f}"
        + (
            ""
            if args.confidence is None
            else f"; the pilot's SDs raised by {j.scale:.4f}, to their {args.confidence:g} "
            "upper limits"
        )
        + (
            ""
            if args.families == 1
            else f"; sized to {power:.4f}, {POWER:g} over {args.families} families"
        )
        + (
            ""
            if not j.capped
            else "; "
            + ", ".join(f"{tool} on {worlds} worlds an environment" for tool, worlds in j.capped)
            + (
                f", exploratory: with them every comparison together {j.whole:.3f}"
                if j.exploratory_caps
                else ", in the claim"
            )
        )
    ]
    caps = dict(j.capped)
    out += [
        f"  less {rival:22s} k = {k}: {chance:.3f}"
        + (
            ""
            if rival not in caps
            else f" on {caps[rival]} worlds an environment"
            + ("; exploratory" if j.exploratory_caps else "")
        )
        for (rival, k), chance in zip(j.comparisons, j.chances, strict=True)
    ]
    return out, j.joint, at_rule


def cap_argument(text: str) -> tuple[str, int]:
    """A `TOOL=WORLDS` argument as a tool and its cap."""
    tool, _, worlds = text.rpartition("=")
    if not tool or not worlds.isdigit():
        msg = f"a cap is TOOL=WORLDS, as robyn=20, not {text!r}"
        raise argparse.ArgumentTypeError(msg)
    return tool, int(worlds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    reading = commands.add_parser("read", help="the claim over scored runs")
    reading.add_argument("runs", type=Path, nargs="+")
    reading.add_argument(
        "--recorded",
        action="store_true",
        help="after the claim, what it does not read: every recorded arm less every rival",
    )
    sizing = commands.add_parser("size", help="the worlds each family's claim needs")
    sizing.add_argument("pilots", type=Path, nargs="+")
    sizing.add_argument("--default", required=True, help="the engine's default procedure")
    sizing.add_argument("--margin", type=float, required=True, help="regret per euro")
    sizing.add_argument(
        "--joint", action="store_true", help="also the worlds every comparison needs together"
    )
    sizing.add_argument("--resamples", type=int, default=RESAMPLES, help="the joint bootstrap's")
    sizing.add_argument("--seed", type=int, default=SEED, help="the joint bootstrap's")
    sizing.add_argument(
        "--confidence", type=float, help="size the joint N on the SDs' upper limits at this level"
    )
    sizing.add_argument(
        "--families",
        type=int,
        default=1,
        help="size each family's joint N for this many families' claims together",
    )
    sizing.add_argument(
        "--cap",
        action="append",
        type=cap_argument,
        default=[],
        metavar="TOOL=WORLDS",
        help="a tool read on its first WORLDS worlds an environment; again for another",
    )
    sizing.add_argument(
        "--exploratory-caps",
        action="store_true",
        help="leave the capped tools' comparisons out of the joint chance and N",
    )
    args = parser.parse_args()
    if args.command == "size":
        args.cap = dict(args.cap)
    if args.command == "size" and (args.cap or args.exploratory_caps) and not args.joint:
        parser.error("--cap and --exploratory-caps size the joint N: they need --joint")
    if args.command == "size" and args.confidence is not None and not args.joint:
        parser.error("--confidence sizes the joint N: it needs --joint")
    if args.command == "size" and args.families != 1 and not args.joint:
        parser.error("--families sizes the joint N: it needs --joint")
    if args.command == "size" and args.families < 1:
        parser.error(f"--families counts the launch claim's families, 1 or more: {args.families}")
    if args.command == "size":
        families: set[int] = set()
        joints: list[float] = []
        at_rules: list[float] = []
        for root in args.pilots:
            layout = Layout(root)
            d = layout.read()
            if not d.pilot or d.family in families:
                parser.error(f"{root} is not a pilot of a family not sized already")
            families.add(d.family)
            if args.exploratory_caps and not (args.cap or d.caps):
                parser.error(
                    f"--exploratory-caps reads the capped tools: {root} declares none, so name them"
                    " by --cap"
                )
            s = size(_records(layout), d, args.default, args.margin)
            print(
                f"family {s.family}: N {s.needed}"
                + (f", above {CAP}: exploratory" if s.exploratory else "")
                + f"; {s.worlds} worlds in each of {s.environments} environments; binding "
                f"{s.rival} at k = {s.k}, sd {s.sd:.4f}; power at a true tie "
                f"{s.power(args.margin):.3f}"
            )
            if args.joint:
                out, joint, at_rule = _joint_lines(_records(layout), d, args, s)
                print("\n".join(out))
                joints.append(joint)
                at_rules.append(at_rule)
        if args.joint:
            print(
                f"the launch claim over the families sized here ({len(joints)}), at a true tie: "
                f"chance {overall(joints):.3f} at their joint N, {overall(at_rules):.3f} at the "
                "binding rule's; their worlds are drawn apart, so the chances multiply"
            )
        return
    declared = [Layout(root).read() for root in args.runs]
    # the roots beside a run declare none
    margins = sorted({d.margin for d in declared if d.margin is not None})
    if len(margins) != 1:
        parser.error(
            f"the claim reads one margin every run declares, not {[d.margin for d in declared]}"
        )
    by_rung = {d.per_rung for d in declared if d.margin is not None}
    if len(by_rung) != 1:
        parser.error("the runs read one claim: every run declares the claim by rung, or none")
    orders = {d.order for d in declared if d.margin is not None}
    if len(orders) != 1:
        parser.error("the runs read one claim: every run declares the same order of verdicts")
    try:
        runs = read_runs(args.runs)
    except ValueError as refused:
        parser.error(str(refused))
    comparisons = [c for run in runs for c in run.comparisons()]
    order = orders.pop()
    try:
        printed = lines(comparisons, margins[0], order=order)
    except ValueError as refused:
        parser.error(str(refused))
    print("\n".join(printed))
    if order:
        print("\n" + "\n".join(secondary_lines(runs)))
    if args.recorded:
        print("\n" + "\n".join(recorded_lines([r for run in runs for r in recorded(run)])))


if __name__ == "__main__":
    main()
