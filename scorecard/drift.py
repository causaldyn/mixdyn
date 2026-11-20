"""The drift run's pre-registered reading: its predictions, P1 to P12, made from its pilot by one
stated rule, and the scored run read against them (`scorecard/preregistration/drift.md`).

    PYTHONPATH=<bench>/src uv run --no-sync python -m scorecard.drift predict PILOT OUT
    PYTHONPATH=<bench>/src uv run --no-sync python -m scorecard.drift read SCORES PREDICTIONS

`PILOT` and `SCORES` are runs' `scores.jsonl`. `predict` writes the predictions to `OUT` as JSON and
prints them as the pre-registration's tables; `read` prints each prediction beside what the scored
run reads, and how many held against how many were expected to.

**An item** is one reading of a run: a quantity of each world, over the worlds of one law, of the
five drifting laws or of every law, at one rung. A `verdict` item's quantity is a paired
difference, or a value less a level, and it is read by where its 95 % Student-t interval over the
worlds lies against nought: `below`, `across` or `above`. A `clause` item is a verdict item read
only as kept, its interval not above nought, or broken. A `share` item is the share of the
world-rungs where something holds. A world where either arm's quantity is not scored, as a failed
fit's CRPS, is left out of that item, and the reading says how many were.

**The rule.** A verdict item's prediction is the verdict most probable given the pilot, with that
chance. With the pilot's mean `m`, standard deviation `s` and `n` worlds, and the `N` worlds the
scored run reads, the scored mean is `m + s (1/n + 1/N)^(1/2) T`, `T` Student's t on `n - 1`
degrees of freedom: the predictive of a future mean under the prior uniform on the mean and on
the log of the spread. The scored interval's half-width is taken as `t(0.975, N - 1) s / N^(1/2)`,
the spread the pilot's. A share item's prediction is the central 90 % of the scored share's
beta-binomial predictive, under Jeffreys' prior on the pilot's count.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from causaldyn_bench.scorecard.scoring import WorldRecord, look
from scipy import stats

from scorecard.report import STATUS_QUO, gated, past

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

DRIFTING = ("walk", "trend", "step", "season", "revert")
LAWS = (*DRIFTING, "static")
GROUPS = {**{law: (law,) for law in LAWS}, "drifting": DRIFTING, "every law": LAWS}
"""Each group an item reads, by the laws whose worlds it pools."""
PRIMARY = "dlm mixture"
"""The engine's default procedure, the one the run's claim reads (`variant`)."""
VARIANTS = ("dlm plug-in", "dlm cvar 0.1")
TIME_VARYING = ("pymc-marketing shared", "pymc-marketing channel")
STATIC = ("stacked", "cvar 0.1", "pymc-marketing", "meridian")
EVERY = (STATUS_QUO, *STATIC[:3], *TIME_VARYING, "meridian", "orbit")
SAMPLING = ("stacked", PRIMARY, "pymc-marketing", *TIME_VARYING, "meridian")
"""An arm for each fit whose chains `past` reads: `cvar 0.1` plans on `stacked`'s fit, and the
DLM's variants on one fit."""
FITTING = (*SAMPLING, "orbit")
RESPONDING = ("stacked", PRIMARY, "pymc-marketing", *TIME_VARYING, "meridian")
FORECASTING = ("stacked", "pymc-marketing", *TIME_VARYING, "orbit")
CLAIMING = ("stacked", "cvar 0.1", *VARIANTS, "pymc-marketing", *TIME_VARYING, "meridian", "orbit")
LEVEL = 0.95
"""The level of a verdict's interval."""
COVERAGE = 0.9
"""The level of the arms' own intervals: a channel's ROI, the claimed gain."""
CENTRAL = 0.9
"""The share of a share item's predictive its prediction holds."""
FEWEST, MOST = 30, 100
"""The sample-size rule's bounds on each law's scored worlds (`worlds`)."""

QUARTER = 13
"""The quarter's weeks, against which P7 reads the multiplier's lengthscale."""

Kind = Literal["verdict", "clause", "share"]


@dataclass(frozen=True, slots=True)
class Item:
    """One reading: `axis` of `first`, less `second`'s or less `level`, over `group` at rung `k`;
    with every plan past its chains' bounds at the status quo's regret where `gated`."""

    name: str
    k: int
    group: str
    axis: str
    first: str
    second: str | None = None
    level: float = 0.0
    kind: Kind = "verdict"
    gated: bool = False

    def label(self) -> str:
        """The quantity, as the tables name it."""
        if self.kind == "share":
            return f"{self.axis}, {self.first}"
        against = self.second if self.second is not None else f"{self.level:g}"
        second = ", second reading" if self.gated else ""
        return f"{self.axis}, {self.first} less {against}{second}"


def items() -> tuple[Item, ...]:
    """The pre-registered readings, P1 to P12."""
    p = PRIMARY
    return (
        *(Item("P1", 0, "drifting", "regret", p, arm) for arm in STATIC),
        *(Item("P2", 0, "drifting", "regret", p, arm) for arm in TIME_VARYING),
        Item("P3", 0, "drifting", "regret", p, "orbit"),
        *(Item("P4", 0, law, "regret", p, arm) for law in LAWS for arm in STATIC),
        *(Item("P5", 0, g, "regret", p, STATUS_QUO, kind="clause") for g in (*LAWS, "drifting")),
        *(Item("P6", 4, "drifting", "regret", p, arm) for arm in EVERY),
        *(
            Item("P7", 0, "drifting", axis, arm, level=level)
            for arm in TIME_VARYING
            for axis, level in (("lengthscale", QUARTER), ("multiplier", 1.0))
        ),
        *(Item("P8", 0, "walk", "roi_covered", arm, level=COVERAGE) for arm in RESPONDING),
        *(Item("P9", 0, law, "crps", p, arm) for law in ("walk", "step") for arm in FORECASTING),
        *(Item("P10", 0, "drifting", "past", arm, kind="share") for arm in SAMPLING),
        *(Item("P10", 0, "drifting", "failure", arm, kind="share") for arm in FITTING),
        *(
            Item("P10", 0, "drifting", "regret", p, arm, gated=True)
            for arm in (*STATIC, *TIME_VARYING, "orbit")
        ),
        *(Item("P11", k, "every law", "moved", "meridian", kind="share") for k in (0, 4)),
        *(Item("P12", 0, law, "uplift", p) for law in LAWS),
        *(Item("P12", 0, law, "covered", p, kind="share") for law in LAWS),
        *(Item("P12", 0, "drifting", "uplift", arm) for arm in CLAIMING),
    )


def _multiplier(record: Mapping[str, Any], axis: str) -> float | None:
    """P7's quantities, from the `multiplier` block of a time-varying PyMC-Marketing record:
    its lengthscale's mean, in weeks, or the mean over the channels and the quarter's weeks of
    the draws' mean multiplier, whose mean over the history is 1 by construction. None where
    the record holds no block, as a static setting's or a failure's."""
    block = record.get("multiplier")
    if not block:
        return None
    if axis == "lengthscale":
        scale = block["lengthscale"]
        return None if scale is None else float(scale["mean"])
    return float(np.mean([c["mean"][1:] for c in block["channels"].values()]))


def _quantity(record: WorldRecord, arm: str, axis: str) -> float | None:
    """`arm`'s `axis` on one world-rung, or None where it is not scored."""
    score = record.arms.get(arm)
    if score is None:
        return None
    if axis == "past":
        read = past(score.record)
        return None if read is None else float(read)
    if axis == "failure":
        return float(score.failure is not None)
    if axis == "moved":
        return float(score.moved is not None)
    if axis in ("lengthscale", "multiplier"):
        return _multiplier(score.record, axis)
    value = getattr(score, axis)
    return None if value is None else float(value)


def seen(records: Sequence[WorldRecord], item: Item, worlds: int) -> list[WorldRecord]:
    """The look at `worlds` worlds of each law of `item`'s group at its rung, every plan past its
    chains' bounds at the status quo's regret where the item is `gated`.

    Raises:
        ValueError: where some law of the group misses a world of the look (`look`).
    """
    laws = GROUPS[item.group]
    at = [r for r in records if r.k == item.k and r.environment in laws]
    chosen = list(look(at, worlds))
    return gated(chosen) if item.gated else chosen


def values(records: Sequence[WorldRecord], item: Item, worlds: int) -> tuple[np.ndarray, int]:
    """`item`'s quantity on each world of the look where it is scored, and how many worlds it
    is not scored on."""
    read, missing = [], 0
    for record in seen(records, item, worlds):
        first = _quantity(record, item.first, item.axis)
        second = item.level if item.second is None else _quantity(record, item.second, item.axis)
        if first is None or second is None:
            missing += 1
        else:
            read.append(first if item.kind == "share" else first - second)
    return np.asarray(read, dtype=float), missing


def verdict(read: np.ndarray) -> str:
    """Where the 95 % Student-t interval of `read`'s mean lies against nought."""
    centre, half = _centre(read)
    return "below" if centre + half < 0.0 else "above" if centre - half > 0.0 else "across"


def _centre(read: np.ndarray) -> tuple[float, float]:
    """The mean of `read` and its 95 % Student-t half-width."""
    n = read.size
    if n < 2:
        msg = f"an interval reads at least two worlds, not {n}"
        raise ValueError(msg)
    sd = float(np.std(read, ddof=1))
    return float(np.mean(read)), float(stats.t.ppf(0.5 + LEVEL / 2.0, n - 1)) * sd / math.sqrt(n)


def chances(pilot: np.ndarray, scored: int) -> dict[str, float]:
    """The chance of each verdict over `scored` worlds, given the pilot's values (the rule)."""
    m, n = float(np.mean(pilot)), pilot.size
    s = float(np.std(pilot, ddof=1))
    if not s > 0.0:
        msg = "the rule reads a pilot whose values spread; one that does not is read as a share"
        raise ValueError(msg)
    half = float(stats.t.ppf(0.5 + LEVEL / 2.0, scored - 1)) * s / math.sqrt(scored)
    scale = s * math.sqrt(1.0 / n + 1.0 / scored)
    below = float(stats.t.cdf((-half - m) / scale, n - 1))
    above = float(stats.t.sf((half - m) / scale, n - 1))
    return {"below": below, "across": 1.0 - below - above, "above": above}


def share_interval(count: int, n: int, scored: int) -> tuple[int, int, float]:
    """The central `CENTRAL` of the count of `scored` world-rungs, given `count` of the pilot's
    `n`, under Jeffreys' prior, and the chance it holds the scored count."""
    law = stats.betabinom(scored, count + 0.5, n - count + 0.5)
    tail = (1.0 - CENTRAL) / 2.0
    low, high = int(law.ppf(tail)), int(law.ppf(1.0 - tail))
    return low, high, float(law.cdf(high) - law.cdf(low - 1))


def worlds(pilot: Sequence[WorldRecord], per_law: int) -> int:
    """Each law's scored worlds by the sample-size rule: for each law and each arm the primary is
    paired with at k = 0, the fewest worlds, in tens from `FEWEST` to `MOST`, at which the pilot's
    spread gives a 95 % half-width of at most half the pilot's difference; the most over every law
    and arm."""
    most = FEWEST
    for law in LAWS:
        for arm in (*EVERY, *VARIANTS):
            read, _ = values(pilot, Item("N", 0, law, "regret", PRIMARY, arm), per_law)
            m, s = abs(float(np.mean(read))), float(np.std(read, ddof=1))
            need = next(
                (
                    n
                    for n in range(FEWEST, MOST + 1, 10)
                    if float(stats.t.ppf(0.5 + LEVEL / 2.0, n - 1)) * s / math.sqrt(n) <= m / 2.0
                ),
                MOST,
            )
            most = max(most, need)
    return most


def variant(pilot: Sequence[WorldRecord], per_law: int) -> str:
    """The engine's default procedure by the rule "X unless the pilot's paired interval of X less
    Y lies above 0", X the DLM's expected-gain plan, pooled over the drifting laws at k = 0; where
    more than one Y passes, the one of least mean regret."""
    beaten = [
        arm
        for arm in VARIANTS
        if verdict(values(pilot, Item("V", 0, "drifting", "regret", PRIMARY, arm), per_law)[0])
        == "above"
    ]
    if not beaten:
        return PRIMARY
    regret = {
        arm: float(np.mean(values(pilot, Item("V", 0, "drifting", "regret", arm), per_law)[0]))
        for arm in beaten
    }
    return min(regret, key=regret.__getitem__)


def preferred(pilot: Sequence[WorldRecord], per_law: int) -> str:
    """PyMC-Marketing's time-varying setting of least mean regret on the pilot, pooled over the
    drifting laws at k = 0: the choice made in its favour."""
    regret = {
        arm: float(np.mean(values(pilot, Item("T", 0, "drifting", "regret", arm), per_law)[0]))
        for arm in TIME_VARYING
    }
    return min(regret, key=regret.__getitem__)


def _scored(item: Item, per_law: int) -> int:
    return per_law * len(GROUPS[item.group])


def predict(pilot: Sequence[WorldRecord], per_law: int) -> dict[str, Any]:
    """Every item's prediction from the pilot's look at `per_law` worlds a law, the scored run
    reading `worlds` of them."""
    n = worlds(pilot, per_law)
    out: dict[str, Any] = {
        "worlds": n,
        "pilot worlds": per_law,
        "primary": variant(pilot, per_law),
        "time-varying": preferred(pilot, per_law),
        "items": [],
    }
    for item in items():
        read, missing = values(pilot, item, per_law)
        scored = _scored(item, n)
        entry: dict[str, Any] = {"item": asdict(item), "pilot n": int(read.size)}
        entry["pilot missing"] = missing
        if item.kind == "share":
            count = int(read.sum())
            low, high, chance = share_interval(count, read.size, scored)
            entry |= {"pilot count": count, "scored": scored, "interval": [low, high]}
            entry |= {"predicted": f"{low}-{high} of {scored}", "chance": chance}
        else:
            centre, half = _centre(read)
            odds = chances(read, scored)
            if item.kind == "clause":
                odds = {"kept": odds["below"] + odds["across"], "broken": odds["above"]}
            best = max(odds, key=odds.__getitem__)
            entry |= {
                "pilot mean": centre,
                "pilot half-width": half,
                "pilot sd": float(np.std(read, ddof=1)),
            }
            entry |= {"scored": scored, "chances": odds, "predicted": best, "chance": odds[best]}
        out["items"].append(entry)
    return out


def outcome(entry: dict[str, Any], records: Sequence[WorldRecord], per_law: int) -> dict[str, Any]:
    """What the scored run reads for one prediction, and whether the prediction held."""
    item = Item(**entry["item"])
    read, missing = values(records, item, per_law)
    if item.kind == "share":
        count = int(read.sum())
        low, high = entry["interval"]
        held = low <= count <= high
        return {"read": f"{count} of {read.size}", "held": held, "missing": missing}
    reading = verdict(read)
    if item.kind == "clause":
        reading = "broken" if reading == "above" else "kept"
    centre, half = _centre(read)
    return {
        "read": f"{reading} {centre:+.4f} [{centre - half:+.4f}, {centre + half:+.4f}]",
        "held": reading == entry["predicted"],
        "missing": missing,
    }


def _records(path: Path) -> list[WorldRecord]:
    return [WorldRecord.from_json(json.loads(line)) for line in path.read_text().splitlines()]


def _table(predictions: dict[str, Any]) -> list[str]:
    """The predictions as the pre-registration's tables, one for each of P1 to P12."""
    lines: list[str] = []
    name = None
    for entry in predictions["items"]:
        item = Item(**entry["item"])
        if item.name != name:
            name = item.name
            head = "| Group | k | Quantity | Pilot | Scored | Prediction | Chance |"
            lines += ["", f"**{name}**", "", head, "|---|---|---|---|---|---|---|"]
        if item.kind == "share":
            pilot = f"{entry['pilot count']} of {entry['pilot n']}"
        else:
            m, h = entry["pilot mean"], entry["pilot half-width"]
            pilot = f"{m:+.3f} [{m - h:+.3f}, {m + h:+.3f}], n {entry['pilot n']}"
        if entry["pilot missing"]:
            pilot += f", {entry['pilot missing']} not scored"
        lines.append(
            f"| {item.group} | {item.k} | {item.label()} | {pilot} | {entry['scored']} | "
            f"{entry['predicted']} | {entry['chance']:.2f} |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    making = commands.add_parser("predict", help="the predictions from a pilot's scores")
    making.add_argument("pilot", type=Path)
    making.add_argument("out", type=Path)
    making.add_argument("--worlds", type=int, default=10, help="the pilot's worlds a law")
    reading = commands.add_parser("read", help="a scored run read against the predictions")
    reading.add_argument("scores", type=Path)
    reading.add_argument("predictions", type=Path)
    args = parser.parse_args()
    if args.command == "predict":
        made = predict(_records(args.pilot), args.worlds)
        args.out.write_text(json.dumps(made, indent=1, allow_nan=False) + "\n")
        print(
            f"worlds a law {made['worlds']}; primary {made['primary']}; time-varying "
            f"{made['time-varying']}"
        )
        print("\n".join(_table(made)))
        return
    made = json.loads(args.predictions.read_text())
    records = _records(args.scores)
    held, expected = 0, 0.0
    for entry in made["items"]:
        read = outcome(entry, records, made["worlds"])
        held += read["held"]
        expected += entry["chance"]
        mark = "held" if read["held"] else "not held"
        item = Item(**entry["item"])
        print(
            f"{item.name} k{item.k} {item.group}, {item.label()}: predicted {entry['predicted']} "
            f"({entry['chance']:.2f}); read {read['read']}; {mark}"
            + (f"; {read['missing']} not scored" if read["missing"] else "")
        )
    print(f"held {held} of {len(made['items'])}, against {expected:.1f} expected")


if __name__ == "__main__":
    main()
