"""A run's report: for each rung, a table for each environment and one pooled over them, each read
over the look that every environment holds whole.

A look at `n` reads the first `n` worlds of each environment's seeded order
(`causaldyn_bench.scorecard.scoring.look`), so a report of a run still under way reads a prefix of
it, as a sequential reading does. Each table gives every arm's mean regret per euro with its 95 %
Student-t interval, its failures by kind, the plans moved into the box and, for an arm that
samples, the plans whose fits are past the bounds (`past`); where the arms claim them, the
claimed gain less the realised one per euro and how often the claim's interval held the realised
gain, and the forecast's CRPS; where the harness scored them on the truth, the flighted schedules
scored and refused and the switchback's reading a week on less its truth, relative to it, with
how often its interval held it (`scorecard.run.SCORED`); and where this engine's records hold it,
their compute as recorded.
Below it, each of this engine's arms less every other arm, paired over the look, with the worlds
where it is lower, tied and higher and the reading rule's verdict: the axis, the family and the
interval a claim names. Where a plan's fit is past the bounds, the table is read a second time
with every such plan at the status quo's regret (`gated`): a fit is scored on its plan, and the
second reading is reported beside it. The family's labels say what its worlds hand some arm by
construction; a pilot and a smoke run say what they are. A run that declares the claim's margin
ends with the claim read on its comparisons alone (`scorecard.claim`), by the reader it declares,
its default above rung 0's records joined from the root it declares them in
(`scorecard.claim.read_run`); a run of a pool is read alone here, and with its pool by
`claim read`. Asked with a `tau`, each rung ends with every arm's flags read as classifiers of a
return wrong by more than `tau` and of a regret above the status quo's, and each fit's R² against
its regret, over the look every environment holds whole (`scorecard.flags`); asked without, the
report is as it was before the flags were read.

A run that declares its pre-registration (`scorecard.layout.Declaration.preregistered`) prints it
below its first line, the path and the SHA-256 as the bench's stamp prints them
(`causaldyn_bench.scorecard.preregistration.Stamp`), and is not reported where the file is gone or
its hash moved since the run was declared; a run that declares none prints as it did before runs
declared one. Asked for the columns, the report ends with the bench's report of the run's family on
its records (`family_report`): each arm's columns at each rung, the compute table its records feed
and, for a scored run, the stamp; asked without, it is as it was before.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from typing import TYPE_CHECKING, Any

import numpy as np
from causaldyn_bench.budget_regret import Mean
from causaldyn_bench.external_arms import verdict
from causaldyn_bench.scorecard import report as bench_report
from causaldyn_bench.scorecard.scoring import BUILT_IN, WorldRecord, look, paired

from scorecard import claim, flags
from scorecard.arms.bayes import shares
from scorecard.arms.decisions import SCORE as SCHEDULE
from scorecard.arms.switchback import SCORE as SWITCHBACK
from scorecard.layout import TREE, write_json

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from causaldyn_bench.scorecard.scoring import ArmScore

    from scorecard.layout import Declaration, Layout

POOLED = "pooled"
STATUS_QUO = BUILT_IN[0]
RHAT = 1.01
"""The largest R-hat a fit's chains may reach: ADR 0024's ceiling on its normal and `tanh`
worlds, the noise of every arm here that samples being normal."""
DIVERGED = 27 / 4000
"""The share of a fit's draws that may diverge: ADR 0025's ceiling at its most, the 27 of 4 000
draws of Stan's run on its `tanh` world."""
GIB = 2.0**30


def past(record: Mapping[str, Any]) -> bool | None:
    """Whether a fit whose draws the record's plan reads is past the bounds, `RHAT` and
    `DIVERGED`; None where the plan reads no Markov chain's draws, as a failure's record, which
    holds no flags, does not.

    * The engine's static arm: each family whose draws the readings hold (`bayes.shares`).
    * The engine's DLM: a fitted spec whose chains did not mix by CHC's check, R-hat at most 1.01
      and 100 effective draws a chain, after ADR 0022 drew them again; Gibbs draws do not diverge.
      The CHC arm's one fit, by the same check (ADR 0041). The two-stage arm's first stage, the
      static arm's fits whose curves it reads, as the static arm's are (ADR 0081): every stage of
      an arm is read.
    * PyMC-Marketing's and Meridian's scripts: their one fit.
    * A record that names no R-hat reads no chain: Orbit's fit is variational, the ridge arm's is
      a point, and the built-in arms fit nothing.

    A sampler's flag that is not a number is past: its chains cannot be read as converged.

    Raises:
        ValueError: on a record that names an R-hat in a form the bounds are not stated for.
    """
    flags, settings = record.get("flags") or {}, record.get("settings") or {}
    if "fitted not mixed" in flags:
        mixed = flags["fitted not mixed"] != 0
        if "first stage" not in settings:
            return mixed
        return mixed or _beyond(_static(flags, settings))
    if "families" in settings:
        fits = _static(flags, settings)
    elif "max_rhat" in flags:
        sampling = settings.get("sampling")
        draws = (
            sampling["n_chains"] * sampling["n_keep"]
            if sampling
            else settings["chains"] * settings["draws"]
        )
        fits = [(flags["max_rhat"], flags["divergences"], draws)]
    elif any("rhat" in name for name in flags):
        msg = f"no bound is stated for a record whose flags are {sorted(flags)}"
        raise ValueError(msg)
    else:
        return None
    return _beyond(fits)


Fit = tuple[float | None, float | None, int]
"""One sampler's fit: its largest R-hat, its divergent draws and its draws."""


def _static(flags: Mapping[str, Any], settings: Mapping[str, Any]) -> list[Fit]:
    """The static arm's fits a record's readings hold, each family's (`bayes.shares`)."""
    sampler = settings["sampler"]
    draws = sampler["chains"] * sampler["draws"]
    counts = shares(flags["hill weight"], settings["readings"])
    return [
        (flags[f"{f} max rhat"], flags[f"{f} divergences"], draws)
        for f, count in counts.items()
        if count
    ]


def _beyond(fits: Sequence[Fit]) -> bool:
    return any(
        rhat is None or diverged is None or rhat > RHAT or diverged > DIVERGED * draws
        for rhat, diverged, draws in fits
    )


def gated(records: Sequence[WorldRecord]) -> list[WorldRecord]:
    """`records` with every plan whose fit is `past` the bounds at the status quo's regret."""
    return [
        dataclasses.replace(
            r,
            arms={
                name: dataclasses.replace(s, regret=r.arms[STATUS_QUO].regret)
                if past(s.record)
                else s
                for name, s in r.arms.items()
            },
        )
        for r in records
    ]


def whole(records: Sequence[WorldRecord]) -> int:
    """The largest look every environment of `records` holds whole: its first worlds in order."""
    held: dict[str, set[int]] = {}
    for r in records:
        held.setdefault(r.environment, set()).add(r.index)
    n = min(len(places) for places in held.values())
    return min(next((i for i in range(n + 1) if i not in places), n) for places in held.values())


def _mean(values: Sequence[float]) -> str:
    if len(values) < 2:
        return f"{values[0]:+.4f}" if values else "n/a"
    m = Mean.of(np.asarray(values, dtype=float))
    return f"{m.mean:+.4f} [{m.interval[0]:+.4f}, {m.interval[1]:+.4f}]"


def _failed(scores: Sequence[ArmScore]) -> str:
    """The failures, by the kind each record names; a tool's script names none."""
    kinds = Counter(s.record.get("failure_kind", "raised") for s in scores if s.failure is not None)
    shown = ", ".join(f"{kind} {count}" for kind, count in sorted(kinds.items()))
    return f"failed {kinds.total()}" + (f" ({shown})" if shown else "")


def _compute(scores: Sequence[ArmScore]) -> str | None:
    """The compute the records hold, as recorded under the run's load: not timings. An engine's
    record splits the fit's compiling and CPU from the plan's; a tool's process holds only its
    whole CPU."""
    costs = [s.record["cost"] for s in scores if "peak_rss_bytes" in s.record.get("cost", {})]
    if not costs:
        return None
    peak = np.array([c["peak_rss_bytes"] for c in costs]) / GIB
    line = (
        f"compute of {len(costs)}: peak RSS median {np.median(peak):.2f} GiB, most {peak.max():.2f}"
    )
    if all("fit_compile_seconds" in c for c in costs):
        line += (
            f"; the fit compiling median {np.median([c['fit_compile_seconds'] for c in costs]):.1f}"
            f" s, its CPU median {np.median([c['fit_cpu_seconds'] for c in costs]):.1f} s"
        )
    else:
        line += f"; the process's CPU median {np.median([c['cpu_seconds'] for c in costs]):.1f} s"
    rates = [c["ess_bulk_per_fit_cpu_second"] for c in costs if "ess_bulk_per_fit_cpu_second" in c]
    if rates:
        line += f"; least bulk ESS a fit CPU second, median {np.median(rates):.3g}"
    return line


def table(records: Sequence[WorldRecord], ours: Sequence[str]) -> list[str]:
    """The look's table, every arm's axes and this engine's arms paired against the rest, and
    again with every plan whose fit is past the bounds at the status quo, where one is."""
    n = whole(records)
    if n < 1:
        return ["  no environment holds its first world yet"]
    seen = look(records, n)
    lines = [f"  {len(seen)} worlds, the first {n} of each environment"]
    lines += _axes(seen) + _paired(seen, ours, n)
    if any(past(s.record) for r in seen for s in r.arms.values()):
        second = gated(seen)
        lines.append(
            f"  each plan whose fit is past R-hat {RHAT} or {DIVERGED * 4000:g} divergences in"
            " 4 000 draws at the status quo:"
        )
        lines += [
            f"  {arm:16s} regret {_mean([r.arms[arm].regret for r in second])}"
            for arm in seen[0].arms
        ]
        lines += _paired(second, ours, n)
    return lines


def _axes(seen: Sequence[WorldRecord]) -> list[str]:
    lines = []
    for arm in seen[0].arms:
        scores = [r.arms[arm] for r in seen]
        line = (
            f"  {arm:16s} regret {_mean([s.regret for s in scores])}; {_failed(scores)}, "
            f"moved {sum(s.moved is not None for s in scores)}"
        )
        sampled = [past(s.record) for s in scores]
        if any(p is not None for p in sampled):
            line += f", past the bounds {sum(p is True for p in sampled)}"
        uplift = [s.uplift for s in scores if s.uplift is not None]
        if uplift:
            line += f"; claimed less realised {_mean(uplift)}"
        covered = [s.covered for s in scores if s.covered is not None]
        if covered:
            line += f", held {sum(covered)} of {len(covered)}"
        crps = [s.crps for s in scores if s.crps is not None]
        if crps:
            line += f"; CRPS {_mean(crps)}"
        line += _beside(scores)
        lines.append(line)
        compute = _compute(scores)
        if compute:
            lines.append(f"    {compute}")
    return lines


def _beside(scores: Sequence[ArmScore]) -> str:
    """What the harness scored on the truth beside the bench's score, where it did."""
    shown = ""
    schedules = [s.record[SCHEDULE] for s in scores if SCHEDULE in s.record]
    if schedules:
        planned = sum("scored" in x for x in schedules)
        refused = sum("refused" in x for x in schedules)
        shown += (
            f"; schedule scored {len(schedules) - refused} ({planned} recommended, its regret the"
            f" variant's), refused {refused}"
        )
    read = [s.record[SWITCHBACK] for s in scores if "error" in s.record.get(SWITCHBACK, {})]
    if read:
        shown += (
            f"; switchback a week on less its truth, relative {_mean([r['error'] for r in read])},"
            f" held {sum(r['held'] for r in read)} of {len(read)}"
        )
    return shown


def _paired(seen: Sequence[WorldRecord], ours: Sequence[str], n: int) -> list[str]:
    lines: list[str] = []
    if n < 2:
        return lines
    arms = list(seen[0].arms)
    for first in (a for a in arms if a in ours):
        lines.append(f"  {first} less, regret per euro:")
        for second in (a for a in arms if a != first):
            c = paired(seen, first, second, n)
            low, high = c.difference.interval
            lines.append(
                f"    {second:16s} {c.difference.mean:+.4f} [{low:+.4f}, {high:+.4f}]; lower "
                f"{c.lower}, tied {c.tied}, higher {c.higher}; {verdict(c)}"
            )
    return lines


def family_report(d: Declaration, records: Sequence[WorldRecord]) -> dict[str, Any]:
    """The bench's report of the run `d` declares on its `records`, as JSON holds it
    (`causaldyn_bench.scorecard.report.report`): each arm's columns at each rung, the compute its
    records hold and, where the run declares it, the pre-registration, named by the declared path.

    Raises:
        ValueError: as the bench's report refuses, as scored worlds of a run that declares no
            pre-registration; and as `Declaration.preregistered` does.
        FileNotFoundError: as `Declaration.preregistered` does.
    """
    held = d.preregistered()
    data = bench_report.report(
        d.source(), records, preregistration=None if held is None else TREE / held.path
    )
    if held is not None:
        # the bench's report names the path it was handed; the run's stamp names the declared one
        data["preregistration"] = held.as_json()
    return data


def main(
    layout: Layout,
    ours: Sequence[str],
    *,
    tau: float | None = None,
    columns: bool = False,
    saved: Path | None = None,
) -> None:
    """Print the report of the run under `layout`, `ours` the names of this engine's records, and
    with `tau` each rung's flags read as classifiers, a channel's return wrong past `tau`; with
    `columns`, the bench's report of the run's family after it (`family_report`), and with `saved`
    that report written there as JSON and as markdown beside it, `saved` with `.md`.

    Raises:
        ValueError: on a `saved` path that is not a `.json` file's; and as
            `Declaration.preregistered` and `family_report` do.
        FileNotFoundError: as `Declaration.preregistered` does.
    """
    if saved is not None and saved.suffix != ".json":
        msg = f"the family's report is written to a .json path, its markdown beside it: {saved}"
        raise ValueError(msg)
    d = layout.read()
    preregistration = d.preregistered()
    family = d.source()
    kind = "smoke, not a result" if d.smoke else "pilot" if d.pilot else f"scored, bench {d.bench}"
    print(f"{layout.root}: {family.name}, {kind}; {d.worlds} worlds of each of {d.environments}")
    if preregistration is not None:
        print(preregistration.line)
    for label in family.labels:
        print(f"  by construction: {label}")
    if not layout.scores.exists():
        print("  no world is scored yet")
        return
    records = [
        WorldRecord.from_json(json.loads(line)) for line in layout.scores.read_text().splitlines()
    ]
    for k in d.rungs:
        at = [r for r in records if r.k == k]
        for environment in (*d.environments, POOLED):
            group = at if environment == POOLED else [r for r in at if r.environment == environment]
            print(f"\nk = {k}, {environment}:")
            print("\n".join(table(group, ours) if group else ["  no world is scored yet"]))
        n = whole(at) if at and tau is not None else 0
        if tau is not None and n >= 1:
            print(f"\nk = {k}, flags over the first {n} worlds of every environment:")
            print("\n".join(flags.lines(look(at, n), tau)))
    if d.margin is not None:
        default = f"{d.default}"
        if d.default_above_0 is not None:
            default = f"{d.default} at k = 0 and {d.default_above_0} above"
        by = ", by Hall's reader" if d.reader == "hall" else ""
        print(f"\nthe claim on this run alone, {default} less every rival{by}:")
        _, claimed = claim.read_run(layout.root)
        print("\n".join(claim.lines(claim.compare(claimed, d), d.margin)))
    if columns or saved is not None:
        data = family_report(d, records)
        written = bench_report.markdown(data)
        if columns:
            print("\n" + written, end="")
        if saved is not None:
            write_json(saved, data)
            saved.with_suffix(".md").write_text(written)
