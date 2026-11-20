"""M2's pre-registration with every slot filled, from the pilot and the pins, written apart from
the draft.

    PYTHONPATH=<bench>/src python -m scorecard.preregistration.m2_fill --bench SHA --out DIR PILOT

`PILOT` holds the pilot's roots as the w30 pilot does: each a run's directory with `run.json` and
`scores.jsonl`, and `peaks.jsonl` beside them. The script reads `m2.md` and `m2.json` beside it,
fills every `{{W30}}` and `{{FREEZE}}` slot and every null of the draft, and writes both files to
`DIR`. It writes nothing while a slot stays open, and nothing into the draft's directory.

**From the pilot.**

* Each family's worlds and each comparison's sizing: `m2_sizing.py` on every root.
* The forecasts: `scorecard.forecast` on every item of `m2.json`, at the declared worlds an
  environment, `DRAWS` resamples, seed `SEED`, from each environment's worlds past its first
  `FROM`, which the ablation that chose the claim's arm did not read. A root that plans an arm the
  family's main roots do not, `dlm two-stage`, Robyn or Meridian, is joined to them by
  environment, seed and rung (`scorecard.claim.join`). Robyn's items carry no forecast: Robyn
  plans 20 worlds an environment, not the declared worlds the forecast reads.
* Each arm's limits on each family: its memory cap `MEMORY` times the largest peak of its
  processes in the peaks files, in whole GiB, and its timeout `TIMEOUT` times the 90th percentile
  of its processes' wall time as their records hold it, in whole minutes. A process's time is its
  fit's once and each variant's plan's.
* The compute: each arm's mean CPU a world, by family and environment, times the worlds the run
  declares there: Robyn's capped, and neither `bayes-geo` nor Meridian on family 10's `geos-50`
  (`NOT_RUN`), which no forecast and no limit reads either.

**From the pins.** causal-hybrid-control's and jax's versions in mixdyn's lock, the bench's commit,
each root's `scores.jsonl` SHA-256, and the forecast's seed.

A value that does not apply stays null, and the object that holds it names it and why under
`n/a`, as a record does.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import tomllib
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from causaldyn_bench.scorecard.scoring import BUILT_IN, WorldRecord
from scipy.stats import t as student

from scorecard.claim import join
from scorecard.forecast import DRAWS, Item, Pilot, forecast, pilot
from scorecard.layout import slug
from scorecard.preregistration import m2_sizing
from scorecard.run import ARMS

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

HERE = Path(__file__).resolve().parent
TREE = HERE.parents[1]
SEED = 20_261_110
"""The forecast's seed: the day it was fixed, before the w30 pilot's last record, as the drift
run's forecast's seed was the day of its own."""
FROM = 10
"""Each environment's first world, counted from 0, that the forecasts read."""
MEMORY = 1.5
TIMEOUT = 6.0
QUANTILE = 0.9
GIB = 2**30
CAPPED = "robyn"
NOT_RUN = frozenset({(10, "bayes-geo", "geos-50"), (10, "meridian", "geos-50")})
"""(family, arm, environment) that the run does not plan, though its pilot may have."""
TESTED = 1
"""The rung every forecast item above 0 reads: each environment's own tested rung."""
SLOT = re.compile(r"\{\{(?:W30|FREEZE|CALL):[^}]*\}\}")
DRAFT_NOTE = re.compile(r"\*\*Draft\.\*\*.*?\n\n", re.DOTALL)
LOG = re.compile(r"(?P<environment>.+)-k[0-9-]+-[0-9]+\.log")
IDENTITY = ("name", "k", "first", "second", "kind", "gated")

type Json = dict[str, Any]
type Records = list[WorldRecord]
type Roots = dict[str, tuple[Path, Json]]


class OpenSlots(Exception):
    """A slot the pilot or the pins leave open: nothing is written."""


def roots(pilot_dir: Path) -> Roots:
    """Each root of the pilot by its directory's name: its path and its declaration.

    Raises:
        OpenSlots: on a pilot with no root, or a root not scored yet.
    """
    found = {
        p.parent.name: (p.parent, json.loads(p.read_text(encoding="utf-8")))
        for p in sorted(pilot_dir.glob("*/run.json"))
    }
    unscored = sorted(n for n, (path, _) in found.items() if not (path / "scores.jsonl").is_file())
    if not found or unscored:
        msg = f"{pilot_dir} holds no root, or roots not scored yet: {unscored}"
        raise OpenSlots(msg)
    return found


def read(path: Path) -> Records:
    lines = (path / "scores.jsonl").read_text(encoding="utf-8").splitlines()
    return [WorldRecord.from_json(json.loads(line)) for line in lines if line.strip()]


def locked(name: str) -> str:
    """The version of `name` that mixdyn's lock holds."""
    lock = tomllib.loads((TREE / "uv.lock").read_text(encoding="utf-8"))
    (version,) = [p["version"] for p in lock["package"] if p["name"] == name]
    return version


def head() -> str:
    command = ["git", "-C", str(TREE), "rev-parse", "HEAD"]
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def _declared(declaration: Json) -> list[str]:
    return [*declaration["arms"], *declaration["external"]]


def _main(declaration: Json) -> bool:
    return "dlm" in declaration["arms"]


def families(found: Roots, held: dict[str, Records], read_: set[str]) -> dict[int, Records]:
    """Each family's records: its main roots', joined with each arm of `read_` that another of
    its roots plans and they do not."""
    out: dict[int, Records] = {}
    for family in sorted({d["family"] for _, d in found.values()}):
        names = [n for n, (_, d) in found.items() if d["family"] == family]
        main = [n for n in names if _main(found[n][1])]
        records = [r for n in main for r in held[n]]
        own = {v for r in records for v in r.arms}
        for name in names:
            if name not in main:
                theirs = held[name]
                for arm in sorted({v for r in theirs for v in r.arms} & read_ - own):
                    records = join(records, theirs, arm)
        out[family] = records
    return out


def sizing(found: Roots, default: str, doc: Json) -> dict[int, Json]:
    """`m2_sizing.py`'s sizing of each family the draft covers, with how many of its pilot worlds
    hold no comparison.

    Raises:
        OpenSlots: on a family the pilot does not size, or sizes on other environments.
    """
    loaded = m2_sizing.load([path for path, _ in found.values()])
    out = {}
    for name, family in doc["families"].items():
        if int(name) not in loaded:
            msg = f"the pilot holds no main root of family {name}"
            raise OpenSlots(msg)
        worlds, tools = loaded[int(name)]
        sized = m2_sizing.size(int(name), worlds, tools, default)
        if sized["environments"] != sorted(family["environments"]):
            msg = f"family {name}'s pilot holds {sized['environments']}, the draft {family}"
            raise OpenSlots(msg)
        out[int(name)] = sized | {"missing": len(worlds) - sized["pilot worlds"]}
    return out


def _family(item: Json) -> int:
    return int(item["group"].removeprefix("family "))


def _rung(k: int | str) -> str:
    return m2_sizing.UNTESTED if k == 0 else m2_sizing.TESTED


def forecasts(
    doc: Json, sized: dict[int, Json], records: dict[int, Records], default: str, start: int
) -> Json:
    """The forecast of every item of `doc` that the pilot past each environment's first `start`
    worlds can make, an entry an item in `doc`'s order, and the count of held forecasts."""
    capped = doc["robyn cap"]
    entries: list[Json] = []
    candidates: list[tuple[int, Item]] = []
    held = {
        f: [replace(r, k=0 if r.k == 0 else TESTED) for r in rs if r.index >= start]
        for f, rs in records.items()
    }
    for j, entry in enumerate(doc["items"]):
        i = entry["item"]
        family = _family(i)
        first = i["first"] if default == m2_sizing.CLAIM else default
        both = {first, i["second"]}
        seen = {r.environment for r in records[family] if both <= r.arms.keys()}
        environments = tuple(
            e
            for e in doc["families"][str(family)]["environments"]
            if e in seen and (family, i["second"], e) not in NOT_RUN
        )
        each = sized[family]["worlds an environment"]
        made: Json = {
            "item": {k: i[k] for k in IDENTITY} | {"first": first, "environments": environments},
            "scored": len(environments) * (min(each, capped) if i["second"] == CAPPED else each),
        }
        empty = [e for e in environments if not any(r.environment == e for r in held[family])]
        reason = None
        if i["second"] == CAPPED:
            reason = f"Robyn plans {capped} worlds an environment, not the declared worlds"
        elif not environments:
            reason = f"the pilot scores {first} and {i['second']} in no environment together"
        elif empty:
            reason = f"the pilot holds no world past the first {start} in {', '.join(empty)}"
        else:
            candidates.append((j, _item(i, environments, first)))
        entries.append(made if reason is None else made | {"predicted": None, "reason": reason})
    read_ = _stacked(held, [item for _, item in candidates])
    known = (~np.isnan(read_.values)).sum(axis=0) >= 2
    for (j, _), enough in zip(candidates, known, strict=True):
        if not enough:
            reason = f"fewer than 2 of its pilot worlds past the first {start} score it"
            entries[j] |= {"predicted": None, "reason": reason}
    eligible = [c for c, enough in zip(candidates, known, strict=True) if enough]
    scored = {e: s["worlds an environment"] for s in sized.values() for e in s["environments"]}
    made_all = forecast(
        Pilot(read_.environments, read_.values[:, known]),
        [item for _, item in eligible],
        scored,
        seed=SEED,
        draws=DRAWS,
    )
    for (j, _), own in zip(eligible, made_all.entries, strict=True):
        entries[j] |= {k: v for k, v in own.items() if k not in ("item", "scored")}
    low, high = made_all.interval
    return {
        "draws": made_all.draws,
        "seed": made_all.seed,
        "from": start,
        "forecast items": sum(e["predicted"] is not None for e in entries),
        "expected": made_all.expected,
        "interval": [low, high],
        "items": entries,
    }


def _item(i: Json, environments: tuple[str, ...], first: str) -> Item:
    return Item(
        name=f"{i['name']} {first} less {i['second']} at {i['k']}",
        k=0 if i["k"] == 0 else TESTED,
        environments=environments,
        axis=i["axis"],
        first=first,
        second=i["second"],
        level=i["level"],
        kind="clause" if i["kind"] == "clause" else "verdict",
        gated=i["gated"],
    )


def _stacked(held: dict[int, Records], items: Sequence[Item]) -> Pilot:
    """Every family's pilot as one, a row a world: an item reads only its own family's
    environments, so another family's rows hold no value of it."""
    parts = [pilot(rs, items) for rs in held.values() if rs]
    if not parts:
        return Pilot(np.array([], dtype=str), np.empty((0, len(items))))
    return Pilot(
        np.concatenate([p.environments for p in parts]), np.vstack([p.values for p in parts])
    )


def _arm(name: str) -> tuple[tuple[str, ...], bool]:
    """An arm's variants, and whether it runs a world at each rung in a process of its own, as
    every tool does."""
    if name not in ARMS:
        return (name,), True
    made = ARMS[name]()
    return tuple(made.variants), bool(made.tests)


type Process = tuple[str, int, int | None]


def processes(
    records: Records, variants: Sequence[str], per_rung: bool
) -> dict[Process, tuple[float, float]]:
    """Each process's wall and CPU seconds, by environment, seed and rung, the rung None where a
    process runs every rung: its fit's once and each variant's plan's, or what a tool's or a
    failure's record holds."""
    held: dict[Process, list[Json]] = defaultdict(list)
    for r in records:
        for v in variants:
            if v in r.arms:
                cost = r.arms[v].record.get("cost") or {}
                held[r.environment, r.seed, r.k if per_rung else None].append(cost)
    out = {}
    for key, costs in held.items():
        fits = [c for c in costs if "fit_wall_seconds" in c]
        if fits:
            wall = fits[0]["fit_wall_seconds"] + sum(c.get("plan_wall_seconds", 0.0) for c in costs)
            cpu = fits[0]["fit_cpu_seconds"] + sum(c.get("plan_cpu_seconds", 0.0) for c in costs)
        else:
            wall = max(c.get("wall_seconds", 0.0) for c in costs)
            cpu = max(c.get("cpu_seconds", 0.0) for c in costs)
        out[key] = (float(wall), float(cpu))
    return out


def peaks(paths: Sequence[Path], found: Roots) -> dict[tuple[int, str, str], int]:
    """The largest peak of each family's arm's processes in each environment, by the root, the
    arm and the world its log names: `<root>/logs/<arm>/<environment>-k<rungs>-<seed>.log`."""
    out: dict[tuple[int, str, str], int] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            parts = Path(entry["log"]).parts
            world = LOG.fullmatch(parts[-1])
            if len(parts) < 4 or parts[-3] != "logs" or parts[-4] not in found or world is None:
                continue
            key = (found[parts[-4]][1]["family"], parts[-2], world["environment"])
            out[key] = max(out.get(key, 0), int(entry["peak_bytes"]))
    return out


def costs(
    found: Roots,
    held: dict[str, Records],
    peak: dict[tuple[int, str, str], int],
    doc: Json,
    sized: dict[int, Json],
) -> tuple[Json, Json]:
    """Each family's arms' limits, and the run's compute in core-hours."""
    limits: Json = {}
    hours: dict[tuple[int, str, str], float] = {}
    for family in sorted(sized):
        names = [n for n, (_, d) in found.items() if d["family"] == family]
        own: Json = {}
        for arm in sorted({a for n in names for a in _declared(found[n][1])}):
            variants, per_rung = _arm(arm)
            runs = {
                n: {
                    key: spent
                    for key, spent in processes(held[n], variants, per_rung).items()
                    if (family, arm, key[0]) not in NOT_RUN
                }
                for n in names
            }
            walls = [wall for each in runs.values() for wall, _ in each.values()]
            tops = [
                b
                for (f, a, e), b in peak.items()
                if (f, a) == (family, slug(arm)) and (family, arm, e) not in NOT_RUN
            ]
            top = max(tops, default=None)
            if not walls or top is None:
                own[arm] = {"memory": None, "timeout": None}
                continue
            p90 = float(np.quantile(walls, QUANTILE))
            own[arm] = {
                "memory": f"{math.ceil(MEMORY * top / GIB)}G",
                "timeout": float(60 * math.ceil(TIMEOUT * p90 / 60)),
                "peak bytes": top,
                "p90 wall seconds": p90,
                "processes": len(walls),
            }
            mains = [n for n in names if _main(found[n][1]) and arm in _declared(found[n][1])]
            world: dict[tuple[str, int], float] = defaultdict(float)
            for n in mains or names:
                for (environment, seed, _), (_, cpu) in runs[n].items():
                    world[environment, seed] += cpu
            each = sized[family]["worlds an environment"]
            scored = min(each, doc["robyn cap"]) if arm == CAPPED else each
            for environment in sized[family]["environments"]:
                cpu = [c for (e, _), c in world.items() if e == environment]
                if cpu:
                    hours[family, arm, environment] = float(np.mean(cpu)) * scored / 3600.0
        limits[str(family)] = own
    by_family: dict[str, float] = defaultdict(float)
    by_arm: dict[str, float] = defaultdict(float)
    for (family, arm, _), h in hours.items():
        by_family[str(family)] += h
        by_arm[arm] += h
    return limits, {
        "core-hours": sum(hours.values()),
        "by family": dict(by_family),
        "by arm": dict(sorted(by_arm.items(), key=lambda kv: -kv[1])),
    }


def open_slots(value: Any, path: str = "") -> Iterator[str]:
    """Every null of `value` that the object holding it does not name under `n/a`, but a clause's
    margin, which it has none of, and a forecast's verdict that gives its reason."""
    if isinstance(value, dict):
        excused = set(value.get("n/a", {}))
        if value.get("kind") == "clause":
            excused.add("margin")
        if "reason" in value:
            excused.add("predicted")
        for key, held in value.items():
            if held is None and key not in excused:
                yield f"{path}.{key}".lstrip(".")
            else:
                yield from open_slots(held, f"{path}.{key}")
    elif isinstance(value, list):
        for j, held in enumerate(value):
            yield from open_slots(held, f"{path}[{j}]")


def fill_json(
    doc: Json,
    sized: dict[int, Json],
    made: Json,
    costed: tuple[Json, Json],
    pins: Json,
    digests: dict[str, str],
) -> Json:
    out = json.loads(json.dumps(doc))
    out["name"] = out["name"].removesuffix(" (draft)")
    for name, family in out["families"].items():
        s = sized[int(name)]
        family |= {
            "needed": s["needed"],
            "declared worlds": s["declared"],
            "worlds an environment": s["worlds an environment"],
            "exploratory": s["exploratory"],
            "joint at needed": s["joint at needed"],
            "predictive joint": s["predictive joint at declared"],
        }
        if s["needed"] is None:
            family["n/a"] = dict.fromkeys(("needed", "joint at needed"), f"past {m2_sizing.MOST}")
    out["worlds"] = {f: sized[int(f)]["worlds an environment"] for f in out["families"]}
    claimed = sum(not sized[int(f)]["exploratory"] for f in out["families"])
    out["launch power"] = round(m2_sizing.POWER**claimed, 6)
    for entry, own in zip(out["items"], made["items"], strict=True):
        _fill_item(entry, own, sized[_family(entry["item"])])
    out["forecasts"] = made
    out["limits"], out["compute"] = costed
    out["pilot digests"] = digests
    out["pins"] |= pins
    del out["pending"]
    return out


def _fill_item(entry: Json, made: Json, sized: Json) -> None:
    """One item's sizing, from its comparison in `sized`, and its forecast `made`."""
    i = entry["item"]
    if i["exploratory"] is None:
        i["exploratory"] = sized["exploratory"]
    rival = "status quo" if i["kind"] == "clause" else i["second"]
    found = [c for c in sized["comparisons"] if (c["rival"], c["rung"]) == (rival, _rung(i["k"]))]
    na: Json = {}
    sizing_only = ("scale", "chance at a tie", "predictive", "predicted failure")
    if found:
        (c,) = found
        n = sized["pilot worlds"]
        entry |= {
            "pilot n": n,
            "pilot missing": sized["missing"],
            "pilot mean": c["mean"],
            "pilot half-width": float(student.ppf(0.975, n - 1)) * c["sd"] / math.sqrt(n),
            "pilot sd": c["sd"],
            "pilot skew": c["skew"],
            "pilot excess kurtosis": c["excess kurtosis"],
            "pilot sd ucl80 chi2": c["sd ucl80 chi2"],
            "pilot sd ucl80 kurtosis": c["sd ucl80 kurtosis"],
        }
        if i["kind"] == "clause":
            na |= dict.fromkeys(sizing_only, "a clause reads its interval against 0, unsized")
        else:
            entry |= {
                "scale": c["scale"],
                "chance at a tie": c["chance at a tie"],
                "predictive": c["predictive at declared"],
                "predicted failure": c["predicted failure"],
            }
            if c["chance at a tie"] is None:
                na["chance at a tie"] = f"no size up to {m2_sizing.MOST} worlds"
    elif i["exploratory"] is True:
        unsized = [k for k, v in entry.items() if v is None and k.startswith("pilot ")]
        na |= dict.fromkeys([*unsized, *sizing_only], "an exploratory comparison, not sized")
    else:
        msg = f"family {_family(i)}'s sizing holds no comparison with {rival} at {i['k']}"
        raise OpenSlots(msg)
    entry["scored"] = made["scored"]
    if made["predicted"] is None:
        na |= dict.fromkeys(("predicted", "chances", "chance"), made["reason"])
    else:
        entry |= {k: made[k] for k in ("chances", "predicted", "chance")}
    if na:
        entry["n/a"] = na


def _number(x: float) -> str:
    """A whole number with a space between each three digits, as the draft writes them."""
    return f"{round(x):,}".replace(",", " ")


def _k(k: int | str) -> str:
    return "k_max(L)" if k == "kmax" else str(k)


def fill_md(text: str, doc: Json) -> str:
    """`text` with every slot filled from the filled `doc`.

    Raises:
        OpenSlots: on a slot of the draft that is not filled once.
    """
    families, pins, made = doc["families"], doc["pins"], doc["forecasts"]
    text = text.replace("# The M2 run: pre-registration (draft)", "# The M2 run: pre-registration")
    text = DRAFT_NOTE.sub("", text, count=1)

    def row(match: re.Match[str]) -> str:
        f = families[match[1]]
        needed = "> " + _number(m2_sizing.MOST) if f["needed"] is None else _number(f["needed"])
        cells = [needed, _number(f["declared worlds"]), str(f["worlds an environment"])]
        cells.append("yes" if f["exploratory"] else "no")
        return f"| {match[1]} | {match[2]} | {' | '.join(cells)} |"

    text, rows = re.subn(
        r"^\| (\d+) \| (\d+) \| \{\{W30: N, m2_sizing\.py\}\}.*$", row, text, flags=re.MULTILINE
    )
    if rows != len(families):
        msg = f"the draft's table holds {rows} rows, not {len(families)}"
        raise OpenSlots(msg)
    claimed = sum(not f["exploratory"] for f in families.values())
    listed = "\n".join(
        f"  - `{name}`: `{digest}`" for name, digest in sorted(doc["pilot digests"].items())
    )
    low, high = made["interval"]
    fills = {
        r"\{\{FREEZE: each pilot root's scores\.jsonl\s+digest\}\}, archived with the scored "
        r"run's records\.": f"as listed here, archived with the scored run's records:\n\n{listed}",
        r"\{\{FREEZE: the version mixdyn's lock holds\}\}": pins["causal-hybrid-control"],
        r"\{\{FREEZE: the bench's commit\}\}": f"`{pins['bench']}`",
        r"\{\{FREEZE: jax version\}\}": pins["jax"],
        r"\{\{FREEZE: the forecast's seed\}\}": str(pins["forecast seed"]),
        r"\{\{W30: 0\.8\^F, F the non-exploratory families\}\}": (
            f"0.8^{claimed} = {doc['launch power']:.3f}, for its {claimed} non-exploratory families"
        ),
        r"\{\{W30: m2_sizing\.py --out, every family's comparisons\}\}\.": _sizing_table(doc),
        r"\{\{W30: m2_sizing\.py, every comparison whose predictive chance is\s+below 0\.5\}\}\.": (
            _failures_table(doc)
        ),
        r"\{\{W30: the expected count of held\s+forecasts and its 95 % interval\}\}": (
            f"{made['expected']:.1f} of {made['forecast items']} (95 % interval {low} to {high})"
        ),
        r"\{\{W30: scorecard\.forecast, every item's forecast and chance\}\}\.": _forecast_table(
            doc
        ),
        r"\{\{W30: each arm's cap, peak memory \u00d7 1\.5, and timeout, 6 \u00d7 p90 wall a "
        r"world-rung\}\}": (_limits_line(doc["limits"])),
        r"\{\{W30: the run's core-hours on the declared worlds, from the w30 pilot's CPU a "
        r"world-rung\}\}\.": _compute_line(doc["compute"]),
    }
    for pattern, value in fills.items():
        text, count = re.subn(pattern, lambda _, v=value: v, text)
        if count != 1:
            msg = f"the draft holds {count} slots matching {pattern!r}, not 1"
            raise OpenSlots(msg)
    left = SLOT.findall(text)
    if left:
        msg = f"slots left open in m2.md: {left}"
        raise OpenSlots(msg)
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+\n", "\n", text))


def _rows(doc: Json, keep: Callable[[Json], bool]) -> Iterator[tuple[Json, Json]]:
    for entry in doc["items"]:
        if keep(entry):
            yield entry["item"], entry


def _sizing_table(doc: Json) -> str:
    lines = [
        "\n\n| # | Rung | Rival | n | Mean | SD | Skew | Excess kurtosis | SD UCL χ² | SD UCL κ "
        "| Scale | Chance at a tie | Predictive |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, e in _rows(doc, lambda e: e["item"]["kind"] == "claim" and e["pilot n"] is not None):
        tie = "-" if e["chance at a tie"] is None else f"{e['chance at a tie']:.3f}"
        lines.append(
            f"| {_family(i)} | {_k(i['k'])} | {i['second']} | {e['pilot n']} | "
            f"{e['pilot mean']:+.4f} | {e['pilot sd']:.4f} | {e['pilot skew']:+.2f} | "
            f"{e['pilot excess kurtosis']:.2f} | {e['pilot sd ucl80 chi2']:.4f} | "
            f"{e['pilot sd ucl80 kurtosis']:.4f} | {e['scale']:.3f} | {tie} | "
            f"{e['predictive']:.3f} |"
        )
    return "\n".join(lines)


def _failures_table(doc: Json) -> str:
    rows = list(_rows(doc, lambda e: e.get("predicted failure") is True))
    if not rows:
        return "none."
    lines = ["\n\n| # | Rung | Rival | Pilot mean | Chance within δ |", "|---|---|---|---|---|"]
    lines += [
        f"| {_family(i)} | {_k(i['k'])} | {i['second']} | {e['pilot mean']:+.4f} | "
        f"{e['predictive']:.3f} |"
        for i, e in rows
    ]
    return "\n".join(lines)


def _forecast_table(doc: Json) -> str:
    lines = [
        "\n\n| # | Rung | Less | Kind | Worlds read | Forecast | Chance |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, e in _rows(doc, lambda _: True):
        made = (
            f"none: {e['n/a']['predicted']} | -"
            if e["predicted"] is None
            else f"{e['predicted']} | {e['chance']:.3f}"
        )
        cells = [str(_family(i)), _k(i["k"]), i["second"], i["kind"], str(e["scored"]), made]
        lines.append(f"| {' | '.join(cells)} |")
    return "\n".join(lines)


def _limits_line(limits: Json) -> str:
    most: dict[str, tuple[int, float]] = {}
    for own in limits.values():
        for arm, held in own.items():
            gib, seconds = int(held["memory"].removesuffix("G")), held["timeout"]
            was = most.get(arm, (0, 0.0))
            most[arm] = (max(was[0], gib), max(was[1], seconds))
    shown = "; ".join(
        f"`{arm}` {gib}G and {_number(seconds)} s" for arm, (gib, seconds) in sorted(most.items())
    )
    return (
        f"{MEMORY:g} times the arm's largest peak memory on the family's pilot, in whole GiB, and "
        f"{TIMEOUT:g} times its 90th percentile of wall time a process, each family's in "
        f"`m2.json`'s `limits`; the most any family's arm gets: {shown}"
    )


def _compute_line(compute: Json) -> str:
    arms = ", ".join(f"`{arm}` {_number(h)}" for arm, h in list(compute["by arm"].items())[:4])
    return (
        "On the declared worlds, from the w30 pilot's recorded CPU a world: about "
        f"{_number(compute['core-hours'])} core-hours, the most in {arms}; each family's and "
        "arm's in `m2.json`'s `compute`."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pilot", type=Path)
    parser.add_argument("--bench", required=True, help="the bench's commit, all 40 hex digits")
    parser.add_argument("--out", type=Path, required=True, help="the directory to write into")
    parser.add_argument("--peaks", type=Path, action="append", help="peaks files beside PILOT's")
    parser.add_argument(
        "--default", default=m2_sizing.CLAIM, help="a stand-in's arm, at both rungs"
    )
    parser.add_argument("--from", dest="start", type=int, default=FROM, help="a stand-in's FROM")
    args = parser.parse_args()
    if re.fullmatch(r"[0-9a-f]{40}", args.bench) is None:
        parser.error(f"--bench takes a commit's 40 hex digits, not {args.bench!r}")
    out = args.out.resolve()
    if out.is_relative_to(TREE / "scorecard"):
        parser.error(f"the filled files are written apart from the draft, not into {out}")
    doc = json.loads((HERE / "m2.json").read_text(encoding="utf-8"))
    if "pending" not in doc:
        parser.error("m2.json holds no slot: the pre-registration is filled already")
    found = roots(args.pilot)
    held = {name: read(path) for name, (path, _) in found.items()}
    digests = {
        name: hashlib.sha256((path / "scores.jsonl").read_bytes()).hexdigest()
        for name, (path, _) in found.items()
    }
    sized = sizing(found, args.default, doc)
    read_ = {i["item"][side] for i in doc["items"] for side in ("first", "second")}
    joined = families(found, held, read_ - set(BUILT_IN) | {args.default})
    made = forecasts(doc, sized, joined, args.default, args.start)
    peak = peaks([args.pilot / "peaks.jsonl", *(args.peaks or [])], found)
    pins = {
        "mixdyn": f"the commit that adds this file, on {head()}",
        "causal-hybrid-control": locked("causal-hybrid-control"),
        "bench": args.bench,
        "jax": locked("jax"),
        "forecast seed": SEED,
    }
    filled = fill_json(doc, sized, made, costs(found, held, peak, doc, sized), pins, digests)
    left = list(open_slots(filled))
    if left:
        msg = f"slots left open in m2.json: {left[:20]}{' ...' if len(left) > 20 else ''}"
        raise OpenSlots(msg)
    text = fill_md((HERE / "m2.md").read_text(encoding="utf-8"), filled)
    out.mkdir(parents=True, exist_ok=True)
    for name, content in (("m2.json", json.dumps(filled, indent=1) + "\n"), ("m2.md", text)):
        aside = out / f".{name}.part"
        aside.write_text(content, encoding="utf-8")
        aside.replace(out / name)
    print(
        f"wrote {out / 'm2.md'} and {out / 'm2.json'}: {len(sized)} families, "
        f"{made['forecast items']} forecasts, {filled['compute']['core-hours']:.0f} core-hours"
    )


if __name__ == "__main__":
    main()
