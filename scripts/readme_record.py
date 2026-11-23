"""Write the README's record of the scored runs its claim was read on, M4 and M3, from their own
records.

The README states M4's claim, and M3's verdicts on the geo families, which M4 did not run and left
standing. Its tables of the claim, the worlds per cell, the losses, the tuned rivals, the versions
and the compute sit between `<!-- record:NAME -->` and `<!-- /record:NAME -->` markers. This script
writes each block, and `tests/test_readme.py` fails where a block is not what it writes: no number
in them is typed. The blocks named `m4-*` are M4's, with M3's geo rows where they say so; `tuned`,
`versions` and `compute` are M3's.

The records are not in this repository (M3's 25 GB, M4's 7.8 GB, not published yet), so it runs in
two steps, the first once for each run:

    uv run --no-project python -I scripts/readme_record.py extract --run m4 STAGE READING
    uv run python scripts/readme_record.py

`extract` reads one run where it lies. STAGE is the scored stage's directory: each root's
`run.json`, `scores.jsonl`, `logs/costs.jsonl` and `records/`, and the stage's `COMMITS`,
`plan.json`, `reuse.json` and `launches.log`. READING is the output of
`python -m scorecard.claim read --recorded` over the run's claim roots, the reading the run's
pre-registration declares. It writes what the
README prints, with the SHA-256 of every file the numbers come from and no path, to
`evidence/results/m4_record.json` (`--run m3`: `m3_record.json`): a reader with the records runs it
again and compares the file. The second command writes the README's blocks from those two files
alone.

What each block reads:

- **m4-claim**, **m4-cells**, **m4-words**: each verdict that holds, its comparisons in the reading,
  their complete cases, failures and words, and Holm's losses: M4's, then M3's on the geo families;
- **m4-losses**, **m4-exploratory**: each comparison of those verdicts, and of M4's rungs not
  claimed, that loses, is not within the margin, is one of Holm's losses or is not within 0.01;
- **tuned**: each comparison in M3's reading of the claim's arm with Meridian and Robyn, at their
  recommended settings and tuned;
- **m4-versions**, **versions**: each record's `versions`, `python` and `provenance` commits, and
  the `settings` of the arms the claim reads;
- **m4-compute**, **compute**: each process's line of `logs/costs.jsonl` (its CPU time, its peak
  memory and its wall), the CPU quota of its class in `plan.json`, and the `cpu_model` and
  `threads` its records' `cost` holds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import textwrap
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
GIB = 2.0**30
MINUS = chr(0x2212)  # the README's minus sign
DASH = chr(0x2013)
BUILT_IN = ("status quo", "equal split")
RIVALS = ("status quo", "equal split", "meridian", "pymc-marketing", "orbit")
TUNED = ("meridian", "meridian tuned", "robyn", "robyn tuned")
TOOLS = ("meridian", "meridian tuned", "pymc-marketing", "orbit", "robyn", "robyn tuned")
WORDS = ("beats", "ties", "loses to", "not compared")
NAMES = {
    "status quo": "Status quo",
    "equal split": "Equal split",
    "meridian": "Meridian",
    "meridian tuned": "Meridian tuned",
    "pymc-marketing": "PyMC-Marketing",
    "orbit": "Orbit",
    "robyn": "Robyn",
    "robyn tuned": "Robyn tuned",
}
PACKAGES = {
    "meridian": "google-meridian",
    "meridian tuned": "google-meridian",
    "pymc-marketing": "pymc-marketing",
    "orbit": "orbit-ml",
}
# The settings each row of the versions table prints, by the record's arm: a label, the path of
# keys into the record's `settings` and, where the setting is read at one rung only, that rung.
Setting = tuple[str, tuple[str, ...]] | tuple[str, tuple[str, ...], int]
SETTINGS: dict[str, tuple[Setting, ...]] = {
    "margin held": (
        ("planner", ("planner",)),
        ("retentions", ("retentions",)),
        ("kernel weeks", ("kernel_length",)),
        ("AR(1) grid", ("ar",)),
        ("time guard", ("time_held",)),
    ),
    "anchored held": (
        ("planner", ("planner",)),
        ("curves", ("curve families",)),
        ("lift rows read", ("lift_rows_read",)),
        ("routes a flagged world to", ("routes_to",)),
        ("time guard", ("time_held",)),
    ),
    # Its records carry `anchored held`'s `routes_to`, but the variant plays no route (ADR 0099).
    "anchored floored": (
        ("planner", ("planner",)),
        ("curves", ("curve families",)),
        ("lift rows read", ("lift_rows_read",)),
        ("floor, a share of the least weekly spend", ("spend_floor",)),
        ("time guard", ("time_held",)),
    ),
    "stacked": (
        ("sampler", ("sampler",)),
        ("curves", ("families",)),
        ("planner", ("planner",)),
    ),
    "geo eb": (
        ("planner", ("planner",)),
        ("retentions", ("retentions",)),
        ("curvatures", ("curvatures",)),
        ("sweeps", ("sweeps",)),
    ),
    "meridian": (
        ("sampling", ("sampling",)),
        ("automatic knot selection", ("enable_aks",)),
        ("ROI prior at k = 0: loc", ("roi_prior", "loc"), 0),
        ("scale", ("roi_prior", "scale"), 0),
        ("max lag", ("max_lag",)),
        ("dtype", ("float_dtype",)),
    ),
    "meridian tuned": (
        ("sampling", ("sampling",)),
        ("automatic knot selection", ("enable_aks",)),
        ("media prior", ("configuration", "media_prior_type")),
        ("knots, weeks apart at most", ("configuration", "knot_weeks")),
        ("max lag", ("max_lag",)),
        ("dtype", ("float_dtype",)),
    ),
    "pymc-marketing": (
        ("model", ("setting",)),
        ("chains", ("chains",)),
        ("draws", ("draws",)),
        ("adstock weeks", ("l_max",)),
        ("yearly harmonics", ("yearly_seasonality",)),
    ),
    "orbit": (
        ("model", ("model",)),
        ("samples", ("num_sample",)),
        ("steps", ("num_steps",)),
        ("holdout weeks", ("holdout_weeks",)),
        ("seasonality", ("seasonality",)),
    ),
    "robyn": (
        ("ranges", ("ranges",)),
        ("iterations", ("iterations",)),
        ("trials", ("trials",)),
        ("cores", ("cores",)),
        ("date range", ("date_range",)),
        ("calibration", ("calibration_scope",)),
    ),
    "robyn tuned": (
        ("ranges", ("ranges",)),
        ("iterations", ("iterations",)),
        ("trials", ("trials",)),
        ("cores", ("cores",)),
        ("date range", ("date_range",)),
        ("calibration", ("calibration_scope",)),
        ("ts_validation", ("ts_validation",)),
    ),
}


@dataclass(frozen=True, slots=True)
class Run:
    """A scored run: its record file, its pre-registration, the run whose records it could reuse,
    the arms its claim reads with their roles, and the processes that plan them."""

    name: str
    record: Path
    preregistration: Path
    reused_from: str
    roles: dict[str, str]
    claim_processes: frozenset[str]


RUNS = {
    "m3": Run(
        "M3",
        ROOT / "evidence" / "results" / "m3_record.json",
        ROOT / "scorecard" / "preregistration" / "m3.md",
        "M2",
        {
            "margin held": "the claim's arm at k = 0",
            "anchored held": "the claim's arm at the tested rung, `gs held`: its router",
            "stacked": "`gs held`'s routed plan, the Bayesian engine's",
            "geo eb": "the claim's arm on the geo families",
        },
        frozenset({"margin", "anchored", "bayes", "geo-eb"}),
    ),
    "m4": Run(
        "M4",
        ROOT / "evidence" / "results" / "m4_record.json",
        ROOT / "scorecard" / "preregistration" / "m4.md",
        "M3",
        {
            "margin held": "the claim's arm at k = 0",
            "anchored floored": "the claim's arm at the tested rung, not routed",
        },
        frozenset({"margin", "anchored"}),
    ),
}
LINE = re.compile(
    r"^  family (?P<family>\d+), k = (?P<k>\d+), (?P<arm>.+?) less (?P<rival>.+?)\s{2,}"
    r"(?P<rest>.+)$"
)
NUMBER = r"[-+]\d+\.\d+"
READ = re.compile(
    rf"^(?P<mean>{NUMBER}) \[(?P<low>{NUMBER}), (?P<high>{NUMBER})\]; "
    r"(?P<within>within|not within) [\d.]+; (?P<word>beats|ties|loses to), unadjusted"
    r"(?: \(level (?P<level>[^)]+)\))?"
)
CASES = re.compile(
    r"(?P<n>\d+) complete cases, the rival failed (?P<rival_failed>\d+), "
    r"the default (?P<default_failed>\d+)"
)
CAP = re.compile(r"on its cap, (?P<cap>\d+) worlds an environment")
VERDICT = re.compile(
    r"^  family (?P<family>\d+), k = (?P<k>\d+): (?P<word>holds|not claimed)"
    r"(?: \(level (?P<level>[^;)]+))?"
)
HOLM = re.compile(
    r"^losses by Holm's step-down at familywise (?P<familywise>[\d.]+) over (?P<over>\d+) "
    r"comparisons: (?P<losses>.*)$"
)
SECONDARY = re.compile(
    r"^  (?P<name>every stage of every fit past the bounds at the status quo"
    r"|at the margin [\d.]+): (?P<within>\d+) of (?P<of>\d+) claimed comparisons within; "
    r"not within: (?P<off>.*)$"
)
NAMED = re.compile(r"^family (?P<family>\d+) k (?P<k>\d+) (?P<rival>.+)$")
BLOCK = re.compile(
    r"<!-- record:(?P<name>[a-z0-9-]+) -->\n(?P<body>.*?)<!-- /record:(?P=name) -->", re.DOTALL
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --- extract: reads the run where it lies ---------------------------------------------------------


def _role(name: str, declared: dict[str, dict[str, Any]]) -> str:
    """A root's role: a claim root declares the margin; a root beside one (its joined `-ts` root,
    its Robyn root) is named after it; any other root records arms beside the claim."""
    if declared[name].get("margin") is not None:
        return "claim"
    claims = [c for c, d in declared.items() if d.get("margin") is not None]
    return "beside" if any(name.startswith(f"{c}-") for c in claims) else "side"


def _named(text: str) -> list[dict[str, Any]]:
    """The comparisons a reading's list names, `family F k K RIVAL, ...`, or none."""
    if text == "none":
        return []
    found = []
    for item in text.split(", "):
        m = NAMED.match(item)
        if m is None:
            msg = f"a comparison the reader does not name: {item[:80]}"
            raise ValueError(msg)
        found.append({"family": int(m["family"]), "k": int(m["k"]), "rival": m["rival"]})
    return found


def _reading(text: str) -> dict[str, Any]:
    """The claim's comparisons and verdicts, the headline, Holm's losses and the secondary
    readings of the claimed comparisons, as the reading prints them."""
    comparisons = []
    lines = text.splitlines()
    end = next(i for i, line in enumerate(lines) if line.startswith("the claim by family"))
    for line in lines[:end]:
        m = LINE.match(line)
        if m is None:
            msg = f"a claim line the reader does not print: {line[:80]}"
            raise ValueError(msg)
        rest = m["rest"]
        read, cases, cap = READ.match(rest), CASES.search(rest), CAP.search(rest)
        if cases is None:
            msg = f"a claim line without its complete cases: {line[:80]}"
            raise ValueError(msg)
        comparisons.append(
            {
                "family": int(m["family"]),
                "k": int(m["k"]),
                "arm": m["arm"],
                "rival": m["rival"],
                "mean": read["mean"] if read else None,
                "low": read["low"] if read else None,
                "high": read["high"] if read else None,
                "within": None if read is None else read["within"] == "within",
                "word": read["word"] if read else "not compared",
                "complete": int(cases["n"]),
                "rival_failed": int(cases["rival_failed"]),
                "default_failed": int(cases["default_failed"]),
                "cap": int(cap["cap"]) if cap else None,
                "exploratory": rest.endswith("; exploratory") or "; exploratory;" in rest,
                "claimed": "not claimed" not in rest,
                "tested": "the tested rung: " in rest,
                "loss level": read["level"] if read else None,
            }
        )
    verdicts = [
        {"family": int(v["family"]), "k": int(v["k"]), "word": v["word"], "level": v["level"]}
        for v in map(VERDICT.match, lines[end:])
        if v is not None
    ]
    headline = [line for line in lines[end:] if re.match(r"^(no )?headline", line)]
    holm = [m for m in map(HOLM.match, lines[end:]) if m is not None]
    secondary = [m for m in map(SECONDARY.match, lines[end:]) if m is not None]
    if len(headline) != 1 or len(holm) != 1:
        msg = "a reading without one headline line and one line of Holm's losses"
        raise ValueError(msg)
    return {
        "comparisons": comparisons,
        "verdicts": verdicts,
        "headline": headline[0],
        "holm": {
            "familywise": holm[0]["familywise"],
            "over": int(holm[0]["over"]),
            "losses": _named(holm[0]["losses"]),
        },
        "secondary": {
            s["name"]: {
                "within": int(s["within"]),
                "of": int(s["of"]),
                "not within": _named(s["off"]),
            }
            for s in secondary
        },
    }


def _versions(directory: Path) -> dict[str, Any]:
    """The distinct `python`, `versions` and provenance commits over a record directory's files."""
    seen: dict[str, int] = defaultdict(int)
    for path in sorted(directory.rglob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        provenance = record.get("provenance") or {}
        key = {
            "python": record.get("python"),
            "versions": record.get("versions"),
            "commits": {
                name: {"commit": held.get("commit"), "dirty": held.get("dirty")}
                for name, held in sorted(provenance.items())
                if isinstance(held, dict) and "commit" in held
            },
            "failed": record.get("error") is not None,
        }
        seen[json.dumps(key, sort_keys=True)] += 1
    return {"directory": directory.name, "seen": dict(seen)}


def _at(settings: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = settings
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _quota(classes: dict[str, Any], arm: str, root: str, role: str) -> float:
    """The CPUs a process of `arm` in `root` was given: its cost class's in the stage's plan, a
    class named for its root, or for a root beside the claim, before the arm's own."""
    name = arm.replace(" ", "-")
    held = (f"{name}@{root}", *((f"side-{name}",) if role == "side" else ()), name)
    found = next((c for c in held if c in classes), None)
    if found is None:
        msg = f"the stage's plan holds no cost class for {arm} in {root}"
        raise KeyError(msg)
    return float(classes[found]["cpus"])


def _summary(values: list[float]) -> dict[str, float]:
    return {"median": statistics.median(values), "most": max(values)}


def extract(run: Path, reading: Path, scored: Run, workers: int = 4) -> dict[str, Any]:
    roots = sorted(p.parent for p in run.glob("*/run.json"))
    declared = {r.name: json.loads((r / "run.json").read_text(encoding="utf-8")) for r in roots}
    roles = {name: _role(name, declared) for name in declared}
    reuse = json.loads((run / "reuse.json").read_text(encoding="utf-8"))
    classes = json.loads((run / "plan.json").read_text(encoding="utf-8"))["classes"]
    commits = {}
    for line in (run / "COMMITS").read_text(encoding="utf-8").splitlines():
        name, *values = line.split()
        commits[name] = values[-1]
    earlier = scored.reused_from

    processes: dict[tuple[int, str], list[dict[str, float]]] = defaultdict(list)
    side: dict[int, float] = defaultdict(float)
    cost_fields: dict[int, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    reused_fields: dict[int, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    settings: dict[str, dict[str, dict[str, set[int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(set))
    )
    rival_of: dict[str, set[str]] = defaultdict(set)
    sources: dict[str, dict[str, Any]] = {}
    for root in roots:
        name, run_json = root.name, declared[root.name]
        family = int(run_json["family"])
        reused = set(reuse.get(name, {}).get("tools", ()))
        sources[name] = {
            "family": family,
            "role": roles[name],
            "scores.jsonl": sha256(root / "scores.jsonl")
            if (root / "scores.jsonl").exists()
            else None,
            "logs/costs.jsonl": sha256(root / "logs" / "costs.jsonl"),
            f"reused from {earlier}": sorted(reused),
        }
        rival_of[name] = set(run_json.get("external", ()))
        for line in (root / "logs" / "costs.jsonl").read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            arm = entry["tag"]["arm"]
            processes[family, arm].append(
                {
                    "cpu_seconds": entry["cpu_usec"] / 1e6,
                    "peak_gib": entry["peak_bytes"] / GIB,
                    "wall_seconds": entry["ended"] - entry["began"],
                    "rival": arm in rival_of[name],
                    "cpus": _quota(classes, arm, name, roles[name]),
                    "rungs": len(entry["tag"]["rungs"]),
                }
            )
            if roles[name] == "side":
                side[family] += entry["cpu_usec"] / 1e6
        scores = root / "scores.jsonl"
        if not scores.exists():
            continue
        with scores.open(encoding="utf-8") as handle:
            for line in handle:
                world = json.loads(line)
                for arm, score in world["arms"].items():
                    record = score.get("record") or {}
                    cost = record.get("cost") or {}
                    held = reused_fields if arm in reused else cost_fields
                    for key in ("cpu_model", "threads"):
                        if cost.get(key) is not None:
                            held[family][key].add(str(cost[key]))
                    if arm in reused:
                        continue
                    for label, path, *rung in SETTINGS.get(arm, ()):
                        if rung and world["k"] != rung[0]:
                            continue
                        value = _at(record.get("settings") or {}, path)
                        if value is not None:
                            settings[arm][label][json.dumps(value, sort_keys=True)].add(family)

    directories = sorted(d for root in roots for d in (root / "records").iterdir() if d.is_dir())
    with ProcessPoolExecutor(max_workers=workers) as pool:
        scanned = list(pool.map(_versions, directories, chunksize=4))
    versions: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    tools = {t.replace(" ", "-"): t for t in TOOLS}
    for directory, found in zip(directories, scanned, strict=True):
        root = directory.parent.parent.name
        group = tools.get(found["directory"], "mixdyn")
        reused = group in set(reuse.get(root, {}).get("tools", ()))
        for key, count in found["seen"].items():
            held = versions[group].setdefault(
                key, {"records": 0, "families": set(), earlier: set()}
            )
            held["records"] += count
            held[earlier if reused else "families"].add(int(declared[root]["family"]))

    families: dict[str, dict[str, Any]] = {}
    for family in sorted({f for f, _ in processes}):
        rows = [p for (f, _), held in processes.items() if f == family for p in held]
        families[str(family)] = {
            "processes": len(rows),
            "core_hours": {
                "engine": sum(p["cpu_seconds"] for p in rows if not p["rival"]) / 3600.0,
                "rivals": sum(p["cpu_seconds"] for p in rows if p["rival"]) / 3600.0,
                "side": side[family] / 3600.0,
            },
            "peak_gib": _summary([p["peak_gib"] for p in rows]),
            "wall_seconds": _summary([p["wall_seconds"] for p in rows]),
            "cpu_model": sorted(cost_fields[family]["cpu_model"]),
            "threads": sorted(cost_fields[family]["threads"]),
            "reused": {
                "cpu_model": sorted(reused_fields[family]["cpu_model"]),
                "threads": sorted(reused_fields[family]["threads"]),
            },
        }
    arms: dict[str, dict[str, Any]] = {}
    for arm in sorted({a for _, a in processes}):
        rows = [p for (_, a), held in processes.items() if a == arm for p in held]
        arms[arm] = {
            "rival": any(p["rival"] for p in rows),
            "cpus": sorted({p["cpus"] for p in rows}),
            "rungs": sorted({p["rungs"] for p in rows}),
            "processes": len(rows),
            "families": sorted(f for f, a in processes if a == arm),
            "core_hours": sum(p["cpu_seconds"] for p in rows) / 3600.0,
            "peak_gib": _summary([p["peak_gib"] for p in rows]),
            "wall_seconds": _summary([p["wall_seconds"] for p in rows]),
        }
    launches = (run / "launches.log").read_text(encoding="utf-8")
    machine = re.search(r": (\d+) CPUs, (\d+) GiB;", launches)
    admitted = {int(w) for w in re.findall(r"box launch: scored, workers (\d+),", launches)}
    if machine is None or len(admitted) != 1:
        msg = "launches.log names no box, or several admissions"
        raise ValueError(msg)
    box = {
        "cpus": int(machine[1]),
        "memory_gib": int(machine[2]),
        "launches": launches.count("box launch: scored"),
        "admitted_cpus": admitted.pop(),
        "launched": re.findall(r"^(\S+) box launch: scored", launches, flags=re.MULTILINE),
    }
    text = reading.read_text(encoding="utf-8")
    return {
        "about": (
            f"{scored.name}'s record as the README prints it, written by "
            "`scripts/readme_record.py extract` from the run's records; every file it read is "
            "named by the run's root and its SHA-256."
        ),
        "commits": commits,
        "reading": {
            "command": "python -m scorecard.claim read --recorded",
            "roots": sorted(n for n, r in roles.items() if r != "side"),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            **_reading(text),
        },
        "sources": {
            "COMMITS": sha256(run / "COMMITS"),
            "plan.json": sha256(run / "plan.json"),
            "reuse.json": sha256(run / "reuse.json"),
            "launches.log": sha256(run / "launches.log"),
            "roots": sources,
        },
        "versions": {
            group: [
                {
                    **json.loads(key),
                    "records": held["records"],
                    "families": sorted(held["families"]),
                    earlier: sorted(held[earlier]),
                }
                for key, held in sorted(seen.items())
            ]
            for group, seen in sorted(versions.items())
        },
        "settings": {
            arm: {
                label: [
                    {"value": json.loads(value), "families": sorted(found)}
                    for value, found in sorted(held.items(), key=lambda v: min(v[1]))
                ]
                for label, held in labels.items()
            }
            for arm, labels in sorted(settings.items())
        },
        "compute": {"box": box, "families": families, "arms": arms},
    }


# --- render: the README's blocks from the records -------------------------------------------------

Records = dict[str, dict[str, Any]]


def _prose(text: str) -> str:
    """A generated sentence wrapped as the README's own prose is, at 100 columns."""
    return textwrap.fill(text, width=100, break_on_hyphens=False, break_long_words=False)


def _signed(value: str) -> str:
    return value.replace("-", MINUS)


def _interval(c: dict[str, Any]) -> str:
    return f"{_signed(c['mean'])} [{_signed(c['low'])}, {_signed(c['high'])}]"


def family_names(scored: Run) -> dict[int, str]:
    """Each family's name, as the run's pre-registration's table of families states it."""
    text = scored.preregistration.read_text(encoding="utf-8")
    table = text.split("## Families and environments", 1)[1].split("\n\n- ", 1)[0]
    names = {}
    for line in table.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and cells[0].isdigit():
            names[int(cells[0])] = cells[1]
    return names


def _rung(c: dict[str, Any]) -> str:
    return f"1{DASH}2" if c["tested"] else str(c["k"])


def _verdict_cells(record: dict[str, Any]) -> list[tuple[int, int]]:
    """The verdicts that hold, national families first and the geo families' after them."""
    comparisons = record["reading"]["comparisons"]
    arm = {(c["family"], c["k"]): c["arm"] for c in comparisons}
    held = [(v["family"], v["k"]) for v in record["reading"]["verdicts"] if v["word"] == "holds"]
    return sorted(held, key=lambda fk: (arm[fk] == "geo eb", fk))


@dataclass(frozen=True, slots=True)
class Row:
    """One family and rung of the claim: the run it was read in (a key of `RUNS`), its comparisons
    by rival and its verdict's level."""

    run: str
    family: int
    k: int
    found: dict[str, dict[str, Any]]
    level: str | None

    @property
    def first(self) -> dict[str, Any]:
        return next(iter(self.found.values()))


def _rows(records: Records, *, claimed: bool = True) -> list[Row]:
    """The claim's family-rungs: M4's verdicts that hold, then M3's verdicts on the geo families,
    which M4 did not run. With `claimed` false, M4's family-rungs not claimed instead."""
    rows = []
    for run in ("m4", "m3"):
        record = records[run]
        reading = record["reading"]
        found: dict[tuple[int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
        for c in reading["comparisons"]:
            found[c["family"], c["k"]][c["rival"]] = c
        for v in sorted(reading["verdicts"], key=lambda v: (v["family"], v["k"])):
            fk = (v["family"], v["k"])
            geo = found[fk][RIVALS[0]]["arm"] == "geo eb"
            if run == "m3" and (not geo or not claimed):
                continue
            if (v["word"] == "holds") == claimed:
                held = {r: found[fk][r] for r in RIVALS}
                rows.append(Row(run, *fk, held, v["level"]))
    return rows


def _holm(records: Records, row: Row) -> set[str]:
    """The rivals Holm's step-down finds a loss to on `row`'s family and rung."""
    found = records[row.run]["reading"]["holm"]["losses"]
    return {h["rival"] for h in found if (h["family"], h["k"]) == (row.family, row.k)}


def _family(row: Row) -> str:
    return f"{row.family} {family_names(RUNS[row.run])[row.family]}"


def _compared(c: dict[str, Any], holm: set[str], *, word: bool = False) -> str:
    if c["mean"] is None:
        return f"not compared: the rival failed {c['rival_failed']}"
    notes = [c["word"]] if word else []
    if c["within"] is False:
        notes.append("not within δ")
    if c["exploratory"]:
        notes.append("exploratory")
    if c["rival"] in holm:
        notes.append("Holm's loss")
    return _interval(c) + "".join(f", {n}" for n in notes)


def claim(records: Records) -> str:
    head = ["Family", "Rung", "Arm", "Run", "Worlds", "Level", *(NAMES[r] for r in RIVALS)]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for row in _rows(records):
        quo = row.found["status quo"]
        holm = _holm(records, row)
        lines.append(
            f"| {_family(row)} | {_rung(row.first)} | `{row.first['arm']}` | {RUNS[row.run].name} "
            f"| {quo['complete'] + quo['default_failed']} | {row.level} | "
            + " | ".join(_compared(row.found[r], holm) for r in RIVALS)
            + " |"
        )
    return "\n".join(lines) + "\n"


def _cases(c: dict[str, Any]) -> str:
    notes = []
    if c["rival_failed"]:
        notes.append(f"the rival failed {c['rival_failed']}")
    if c["default_failed"]:
        notes.append(f"the arm failed {c['default_failed']}")
    if c["exploratory"]:
        notes.append("exploratory")
    return f"{c['complete']}" + (f" ({', '.join(notes)})" if notes else "")


def m4_cells(records: Records) -> str:
    head = ["Family", "Rung", "Arm", "Run", *(NAMES[r] for r in RIVALS)]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += [
        f"| {_family(row)} | {_rung(row.first)} | `{row.first['arm']}` | {RUNS[row.run].name} "
        "| " + " | ".join(_cases(row.found[r]) for r in RIVALS) + " |"
        for row in _rows(records)
    ]
    return "\n".join(lines) + "\n"


def words(records: Records) -> str:
    """Hall's reader's words over the claim's verdicts, by run, arm and rung, for each rival."""
    groups: dict[tuple[str, str, str], list[Row]] = defaultdict(list)
    for row in _rows(records):
        groups[RUNS[row.run].name, row.first["arm"], "0" if row.k == 0 else "tested"].append(row)
    head = ["Run", "Arm", "Rung", "Verdicts", *(NAMES[r] for r in RIVALS)]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for (run, arm, rung), rows in groups.items():
        counts = []
        for rival in RIVALS:
            said = [r.found[rival]["word"] for r in rows]
            counts.append(", ".join(f"{w} {said.count(w)}" for w in WORDS if said.count(w)))
        lines.append(
            f"| {run} | `{arm}` | {'the tested rung' if rung == 'tested' else rung} "
            f"| {len(rows)} | " + " | ".join(counts) + " |"
        )
    return "\n".join(lines) + "\n"


def _yes(value: bool) -> str:
    return "yes" if value else "no"


def losses(records: Records) -> str:
    """Each comparison of the claim's verdicts, and of M4's rungs not claimed, that Hall's reader
    reads as a loss, that is not within δ, that is one of Holm's losses, or that is not within
    0.01 in the secondary reading."""
    head = ["Family", "Rung", "Arm", "Run", "Rival", "Mean [95 % interval]", "Hall's word"]
    head += ["Within δ = 0.05", "Holm's loss", "Within δ = 0.01"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    every = _rows(records) + _rows(records, claimed=False)
    for row in sorted(every, key=lambda r: (r.run != "m4", r.family, r.k)):
        tight = records[row.run]["reading"]["secondary"]["at the margin 0.01"]["not within"]
        holm = _holm(records, row)
        for rival in RIVALS:
            c = row.found[rival]
            off = {"family": row.family, "k": row.k, "rival": rival} in tight
            if c["mean"] is None or not (
                c["word"] == "loses to" or not c["within"] or rival in holm or off
            ):
                continue
            close = _yes(not off) if c["claimed"] else "not read"
            lines.append(
                f"| {_family(row)} | {_rung(c)} | `{c['arm']}` | {RUNS[row.run].name} "
                f"| {NAMES[rival]} "
                f"| {_interval(c)} | {c['word']} | {_yes(c['within'])} | {_yes(rival in holm)} "
                f"| {close} |"
            )
    return "\n".join(lines) + "\n"


def exploratory(records: Records) -> str:
    head = ["Family", "Rung", "Arm", "Worlds", *(NAMES[r] for r in RIVALS)]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for row in _rows(records, claimed=False):
        quo = row.found["status quo"]
        holm = _holm(records, row)
        lines.append(
            f"| {_family(row)} | {_rung(row.first)} | `{row.first['arm']}` "
            f"| {quo['complete'] + quo['default_failed']} | "
            + " | ".join(_compared(row.found[r], holm, word=True) for r in RIVALS)
            + " |"
        )
    return "\n".join(lines) + "\n"


def tuned(record: dict[str, Any]) -> str:
    names = family_names(RUNS["m3"])
    found = {(c["family"], c["k"], c["rival"]): c for c in record["reading"]["comparisons"]}
    rows = sorted(
        {(f, k) for f, k, r in found if r in ("meridian tuned", "robyn tuned")},
    )
    head = ["Family", "Rung", "Arm", *(NAMES[r] for r in TUNED)]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for family, k in rows:
        first = next(found[family, k, r] for r in TUNED if (family, k, r) in found)
        text = []
        for rival in TUNED:
            c = found.get((family, k, rival))
            if c is None:
                text.append("not run")
            elif c["mean"] is None:
                text.append(f"not compared: the rival failed {c['rival_failed']}")
            else:
                note = f", {c['complete']} worlds" if rival.startswith("robyn") else ""
                text.append(f"{_interval(c)}, {c['word']}{note}")
        lines.append(
            f"| {family} {names[family]} | {_rung(first)} | `{first['arm']}` | "
            + " | ".join(text)
            + " |"
        )
    counts = []
    for rival in TUNED:
        read = [found[f, k, rival] for f, k in rows if (f, k, rival) in found]
        said = {w: sum(c["word"] == w for c in read) for w in ("beats", "ties", "loses to")}
        said["is not compared"] = sum(c["mean"] is None for c in read)
        listed = ", ".join(f"{w} on {n}" for w, n in said.items() if n)
        counts.append(f"against {NAMES[rival]} it {listed}, of {len(read)}")
    lines += ["", _prose("Over these rows the claim's arm reads: " + "; ".join(counts) + ".")]
    return "\n".join(lines) + "\n"


def _value(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, list) and value and all(v == value[0] for v in value):
        return _value(value[0])
    if isinstance(value, list):
        return ", ".join(_value(v) for v in value)
    if isinstance(value, dict):
        return ", ".join(f"{k} {_value(v)}" for k, v in value.items())
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _settings(record: dict[str, Any], arm: str) -> str:
    parts = []
    for label, found in record["settings"].get(arm, {}).items():
        values = list(dict.fromkeys(_value(f["value"]) for f in found))
        listed = values[0] if len(values) == 1 else f"{', '.join(values[:-1])} or {values[-1]}"
        parts.append(f"{label} {listed}" + (" (by family)" if len(values) > 1 else ""))
    return "; ".join(parts)


def _families(numbers: list[int]) -> str:
    return ", ".join(map(str, numbers))


def _listed(numbers: list[int]) -> str:
    return f"{'family' if len(numbers) == 1 else 'families'} {_families(numbers)}"


def _ran(e: dict[str, Any], earlier: str) -> str:
    """The families whose records hold `e`'s versions: run in this run, and reused from the
    earlier one."""
    parts = []
    if e["families"]:
        parts.append(_listed(e["families"]))
    if e[earlier]:
        parts.append(f"{earlier}'s records reused on {_families(e[earlier])}")
    return "; ".join(parts)


def _tool_role(record: dict[str, Any], tool: str) -> str:
    """A tool's role: a rival the claim reads, or one read beside it on every comparison."""
    read = [c for c in record["reading"]["comparisons"] if c["rival"] == tool]
    beside = all(c["exploratory"] for c in read)
    kind = "tuned rival" if tool.endswith("tuned") else "rival"
    return f"{kind}, read beside the claim" if beside else kind


def _failures(entries: list[dict[str, Any]], earlier: str) -> str:
    failed = [e for e in entries if e["failed"]]
    if not failed:
        return ""
    families = sorted({f for e in failed for f in (*e["families"], *e[earlier])})
    count = sum(e["records"] for e in failed)
    return f"; {count} failure records, {_listed(families)}"


def versions(record: dict[str, Any], scored: Run) -> str:
    lines = [
        "| Arm | Role | Version, as each record states it | Run with | How it was configured |",
        "|---|---|---|---|---|",
    ]
    held = record["versions"]
    arms = record["compute"]["arms"]
    earlier = scored.reused_from
    for e in (e for e in held["mixdyn"] if not e["failed"]):
        built = ", ".join(
            f"{name} at `{c['commit'][:7]}`" + (", dirty" if c["dirty"] else "")
            for name, c in e["commits"].items()
        )
        named = ", ".join(f"{k} {v}" for k, v in e["versions"].items() if k != "mixdyn")
        for arm, role in scored.roles.items():
            families = sorted(
                {
                    f
                    for found in record["settings"][arm].values()
                    for v in found
                    for f in v["families"]
                }
            )
            lines.append(
                f"| `{arm}` | {role}; {_listed(families)} | {built} "
                f"| Python {e['python']}; {named} | {_settings(record, arm)} |"
            )
        others = sorted(
            a for a, row in arms.items() if not row["rival"] and a not in scored.claim_processes
        )
        families = sorted({f for a in others for f in arms[a]["families"]})
        lines.append(
            f"| {', '.join(f'`{a}`' for a in others)} | the engine's other arms, recorded beside "
            f"the claim; {_listed(families)} | {built} | Python {e['python']}; {named} "
            "| each record's own settings |"
        )
    for tool in TOOLS:
        entries = held.get(tool, [])
        for e in (e for e in entries if not e["failed"]):
            found = e["versions"]
            if tool.startswith("robyn"):
                packages = found["packages"]
                version = f"Robyn {packages['Robyn']}"
                rest = (
                    f"{found['r'].split(' (')[0]}; "
                    + ", ".join(f"{k} {v}" for k, v in packages.items() if k != "Robyn")
                    + f"; Python {found['python'].split()[0]}, nevergrad {found['nevergrad']}, "
                    f"numpy {found['numpy']}"
                )
            else:
                package = PACKAGES[tool]
                version = f"{NAMES[tool].removesuffix(' tuned')} {found[package]}"
                rest = f"Python {e['python']}; " + ", ".join(
                    f"{k} {v}" for k, v in found.items() if k != package
                )
            lines.append(
                f"| {NAMES[tool]} | {_tool_role(record, tool)}; {_ran(e, earlier)}"
                f"{_failures(entries, earlier)} | {version} | {rest} | {_settings(record, tool)} |"
            )
    return "\n".join(lines) + "\n"


def _hours(value: float) -> str:
    return f"{value:,.1f}".replace(",", " ")


def _count(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def _wall(seconds: float) -> str:
    return f"{seconds:.0f} s" if seconds < 600 else f"{seconds / 60:.0f} min"


def compute(record: dict[str, Any], scored: Run) -> str:
    names = family_names(scored)
    families, arms, box = (record["compute"][k] for k in ("families", "arms", "box"))
    lines = [
        _prose(
            f"One box of {box['cpus']} CPUs and {box['memory_gib']} GiB ran every process, "
            f"admitting up to {box['admitted_cpus']} CPUs of quota at a time. It was launched "
            f"{box['launches']} times: the stage resumed after each of {box['launches'] - 1} "
            "spot interruptions."
        ),
        "",
        "| Family | CPU model | Threads | Processes | Peak memory a process, median / most "
        "| Wall a process, median / most | Core-hours: engine | rivals | all |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    total = {"engine": 0.0, "rivals": 0.0, "processes": 0}
    for family, row in families.items():
        hours = row["core_hours"]
        total["engine"] += hours["engine"]
        total["rivals"] += hours["rivals"]
        total["processes"] += row["processes"]
        peak, wall = row["peak_gib"], row["wall_seconds"]
        lines.append(
            f"| {family} {names[int(family)]} | {', '.join(row['cpu_model'])} "
            f"| {', '.join(row['threads'])} | {_count(row['processes'])} "
            f"| {peak['median']:.2f} / {peak['most']:.2f} GiB "
            f"| {_wall(wall['median'])} / {_wall(wall['most'])} "
            f"| {_hours(hours['engine'])} | {_hours(hours['rivals'])} "
            f"| {_hours(hours['engine'] + hours['rivals'])} |"
        )
    lines.append(
        f"| all | | | {_count(total['processes'])} | | | {_hours(total['engine'])} "
        f"| {_hours(total['rivals'])} | {_hours(total['engine'] + total['rivals'])} |"
    )
    lines += [
        "",
        "| Arm | Role | CPU quota | Rungs a process | Families | Processes "
        "| Peak memory a process, median / most | Wall a process, median / most | Core-hours |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, row in sorted(arms.items(), key=lambda a: -a[1]["core_hours"]):
        peak, wall = row["peak_gib"], row["wall_seconds"]
        lines.append(
            f"| `{arm}` | {'rival' if row['rival'] else 'engine'} "
            f"| {', '.join(f'{c:g}' for c in row['cpus'])} "
            f"| {' or '.join(map(str, row['rungs']))} "
            f"| {_families(row['families'])} | {_count(row['processes'])} "
            f"| {peak['median']:.2f} / {peak['most']:.2f} GiB "
            f"| {_wall(wall['median'])} / {_wall(wall['most'])} | {_hours(row['core_hours'])} |"
        )
    reused = {
        f: row["reused"]["cpu_model"] for f, row in families.items() if row["reused"]["cpu_model"]
    }
    side = {
        f: row["core_hours"]["side"] for f, row in families.items() if row["core_hours"]["side"]
    }
    notes = []
    if reused:
        models = sorted({m for held in reused.values() for m in held})
        notes.append(
            f"Families {_families([int(f) for f in reused])} read the rivals' records of "
            f"{scored.reused_from}, whose compute is not counted here; those records name "
            f"{', '.join(models)}."
        )
    if side:
        notes.append(
            "The engine's core-hours include the arms recorded beside the claim, read by no "
            "comparison: " + ", ".join(f"{_hours(h)} on family {f}" for f, h in side.items()) + "."
        )
    return "\n".join(lines) + "\n" + ("\n" + _prose(" ".join(notes)) + "\n" if notes else "")


RENDERERS: dict[str, Callable[[Records], str]] = {
    "m4-claim": claim,
    "m4-cells": m4_cells,
    "m4-words": words,
    "m4-losses": losses,
    "m4-exploratory": exploratory,
    "m4-versions": lambda records: versions(records["m4"], RUNS["m4"]),
    "m4-compute": lambda records: compute(records["m4"], RUNS["m4"]),
    "tuned": lambda records: tuned(records["m3"]),
    "versions": lambda records: versions(records["m3"], RUNS["m3"]),
    "compute": lambda records: compute(records["m3"], RUNS["m3"]),
}


def records() -> Records:
    """Each run's record, as `extract` wrote it."""
    return {run: json.loads(s.record.read_text(encoding="utf-8")) for run, s in RUNS.items()}


def render(held: Records) -> dict[str, str]:
    """Each block the README holds, as this script writes it from the runs' records."""
    return {name: write(held) for name, write in RENDERERS.items()}


def blocks(text: str) -> dict[str, str]:
    """Each generated block of `text`, by its name."""
    return {m["name"]: m["body"] for m in BLOCK.finditer(text)}


def rewrite(text: str, written: dict[str, str]) -> str:
    def replace(m: re.Match[str]) -> str:
        name = m["name"]
        return f"<!-- record:{name} -->\n{written[name]}<!-- /record:{name} -->"

    return BLOCK.sub(replace, text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command")
    take = commands.add_parser("extract", help="read one run's records into its record file")
    take.add_argument("--run", choices=sorted(RUNS), required=True)
    take.add_argument("run_directory", type=Path, metavar="run")
    take.add_argument("reading", type=Path)
    take.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.command == "extract":
        scored = RUNS[args.run]
        record = extract(args.run_directory, args.reading, scored, args.workers)
        text = json.dumps(record, indent=1, ensure_ascii=False) + "\n"
        scored.record.write_text(text, encoding="utf-8")
        return 0
    text = README.read_text(encoding="utf-8")
    missing = sorted(set(RENDERERS) - set(blocks(text)))
    if missing:
        print(f"README.md holds no block for {', '.join(missing)}", file=sys.stderr)
        return 1
    README.write_text(rewrite(text, render(records())), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
