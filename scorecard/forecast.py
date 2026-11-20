"""Forecasts for a run's pre-registration, from a bootstrap predictive of its pilot's worlds.

A run's own module names its items and reads them with these functions: `pilot` lays the pilot's
values out by world, `forecast` makes every item's forecast and the count of held forecasts' own,
and `values` and `outcome` read a scored run against them.

**An item** is one reading of a run: a quantity of each world over some of its environments at one
rung (`Item`). A `verdict` item's quantity is a paired difference of two arms, or an arm's value
less a level, read by where its 95 % Student-t interval over the worlds lies against nought:
`below`, `across` or `above`. A `clause` item is a verdict item read only as kept, its interval not
above nought, or broken. A `share` item counts the worlds where something holds. A world where an
item's quantity is not scored is left out of it. `quantity` reads the axes every record scores; a
run whose items read more passes its own reader.

**The rule.** `forecast` draws resamples of the pilot, each in two stages, from one seeded stream.
In each environment it draws the pilot's worlds with replacement, as many as the pilot holds: what
the worlds could be, given the pilot. From those it draws as many as the scored run reads there:
the scored run's own draw. Every item reads the same worlds of a resample at every rung, so the
items keep the correlation their shared worlds give them. On each resample an item reads as the
scored run would read it. A verdict's forecast is the verdict most resamples read, with that share
as its chance; a share's is the central 90 % of its counts, with the share of resamples inside it.
The two stages give the scored mean about the spread of the predictive the drift run's rule
states, `s^2 (1/n + 1/N)` for `n` pilot worlds and `N` scored ones; where that rule took the pilot's
values as normal, the bootstrap keeps their tails. It does not widen them as Student's t does: a
resample spreads as the pilot's worlds do, `(n - 1)/n` of `s^2`, so on a few pilot worlds an
environment its forecasts are the surer. The rule is meant for 30 or more.

**No forecast** is made on an item that reads an arm the pilot selected, as the engine's default
procedure or a tool's setting: the pilot that chose the arm for its showing would forecast that
showing. Nor on an item whose pilot values are all one value: every resample reads it alike, so its
chance would be 1 by construction, not by evidence.

**The count of held forecasts.** Each resample holds some of the forecasts. The mean of that count
is the sum of the chances; its central 95 % is the count's interval, which holds the items'
correlation where a sum of independent chances would not.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy import stats

from scorecard.report import gated, past

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping, Sequence

    from causaldyn_bench.scorecard.scoring import WorldRecord

LEVEL = 0.95
"""The level of a verdict's interval."""
CENTRAL = 0.9
"""The share of a share item's predictive its forecast holds."""
HELD = 0.95
"""The central share of the count of held forecasts its interval holds."""
DRAWS = 4000
FEWEST = 2000
"""The resamples a forecast draws, and the fewest it may."""
VERDICTS = ("below", "across", "above")

Kind = Literal["verdict", "clause", "share"]


@dataclass(frozen=True, slots=True)
class Item:
    """One reading: `axis` of `first`, less `second`'s or less `level`, over the worlds of
    `environments` at rung `k`; with every plan past its chains' bounds at the status quo's regret
    where `gated`."""

    name: str
    k: int
    environments: tuple[str, ...]
    axis: str
    first: str
    second: str | None = None
    level: float = 0.0
    kind: Kind = "verdict"
    gated: bool = False

    @property
    def arms(self) -> tuple[str, ...]:
        return (self.first,) if self.second is None else (self.first, self.second)

    def label(self) -> str:
        """The quantity, as a table names it."""
        if self.kind == "share":
            return f"{self.axis}, {self.first}"
        against = self.second if self.second is not None else f"{self.level:g}"
        return f"{self.axis}, {self.first} less {against}" + (", gated" if self.gated else "")


type Reader = Callable[[WorldRecord, Item], float | None]


def score(record: WorldRecord, arm: str, axis: str) -> float | None:
    """`arm`'s `axis` on one world-rung, or None where it is not scored: an axis the record
    scores, whether the fit is `past` its chains' bounds, or whether the plan `failure`d or was
    `moved` into the box."""
    held = record.arms.get(arm)
    if held is None:
        return None
    if axis == "past":
        read = past(held.record)
        return None if read is None else float(read)
    if axis == "failure":
        return float(held.failure is not None)
    if axis == "moved":
        return float(held.moved is not None)
    value = getattr(held, axis)
    return None if value is None else float(value)


def quantity(record: WorldRecord, item: Item) -> float | None:
    """`item`'s quantity on one world-rung (`score`), or None where it is not scored."""
    seen = gated([record])[0] if item.gated else record
    first = score(seen, item.first, item.axis)
    second = item.level if item.second is None else score(seen, item.second, item.axis)
    if first is None or second is None:
        return None
    return first if item.kind == "share" else first - second


@dataclass(frozen=True)
class Pilot:
    """A pilot's values by world: each world's environment, and its value of each item, not a
    number where the item is not scored on it or is not read there."""

    environments: np.ndarray
    values: np.ndarray


def pilot(
    records: Sequence[WorldRecord], items: Sequence[Item], reader: Reader = quantity
) -> Pilot:
    """The pilot's `records` as one row a world, the world's records at every rung on it.

    Raises:
        ValueError: on records of more than one family, or a world held twice at one rung.
    """
    held: dict[tuple[str, int, int], WorldRecord] = {}
    for r in records:
        if r.family != records[0].family or (r.environment, r.index, r.k) in held:
            msg = f"a pilot is one family's, each world once a rung: {r.environment} {r.index}"
            raise ValueError(msg)
        held[r.environment, r.index, r.k] = r
    worlds = sorted({(e, i) for e, i, _ in held})
    values = np.full((len(worlds), len(items)), np.nan)
    for row, (environment, index) in enumerate(worlds):
        for column, item in enumerate(items):
            record = held.get((environment, index, item.k))
            if record is None or environment not in item.environments:
                continue
            value = reader(record, item)
            if value is not None:
                values[row, column] = value
    return Pilot(np.array([e for e, _ in worlds]), values)


@dataclass(frozen=True)
class Forecasts:
    """Every item's forecast, the count of held forecasts on each resample, and the draw."""

    entries: tuple[dict[str, Any], ...]
    held: np.ndarray
    draws: int
    seed: int

    @property
    def expected(self) -> float:
        return float(np.mean(self.held))

    @property
    def interval(self) -> tuple[int, int]:
        tail = (1.0 - HELD) / 2.0
        low, high = np.quantile(self.held, (tail, 1.0 - tail), method="inverted_cdf")
        return int(low), int(high)

    def as_json(self) -> dict[str, Any]:
        return {
            "draws": self.draws,
            "seed": self.seed,
            "expected": self.expected,
            "interval": list(self.interval),
            "items": list(self.entries),
        }


def forecast(
    read: Pilot,
    items: Sequence[Item],
    scored: Mapping[str, int],
    *,
    seed: int,
    selected: Collection[str] = (),
    draws: int = DRAWS,
) -> Forecasts:
    """Every item's forecast from the pilot `read`, the scored run reading `scored[e]` worlds of
    each environment `e`, by `draws` resamples seeded by `seed`; none on an item that reads an
    arm of `selected`.

    Raises:
        ValueError: on fewer than `FEWEST` draws, or an environment an item reads that the pilot
            holds no world of or the scored run reads none of.
    """
    if draws < FEWEST:
        msg = f"a forecast draws at least {FEWEST} resamples, not {draws}"
        raise ValueError(msg)
    needed = sorted({e for item in items for e in item.environments})
    rows = {e: np.flatnonzero(read.environments == e) for e in needed}
    empty = [e for e in needed if rows[e].size == 0 or scored.get(e, 0) < 1]
    if empty:
        msg = f"the pilot and the scored run each hold worlds of every environment read: {empty}"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    drawn = {}
    for e in needed:
        outer = rng.integers(0, rows[e].size, size=(draws, rows[e].size))
        inner = rng.integers(0, rows[e].size, size=(draws, scored[e]))
        drawn[e] = rows[e][np.take_along_axis(outer, inner, axis=1)]
    held = np.zeros(draws, dtype=int)
    entries = []
    for column, item in enumerate(items):
        own = read.values[np.concatenate([rows[e] for e in item.environments]), column]
        known = own[~np.isnan(own)]
        entry: dict[str, Any] = {"item": asdict(item), "pilot n": int(known.size)}
        entry["pilot missing"] = int(own.size - known.size)
        entry["scored"] = sum(scored[e] for e in item.environments)
        if set(item.arms) & set(selected):
            entries.append(entry | {"predicted": None, "reason": "the pilot selected it"})
            continue
        if known.size < 2 or np.ptp(known) == 0.0:
            entries.append(entry | {"predicted": None, "reason": "its pilot values are one value"})
            continue
        resampled = read.values[
            np.concatenate([drawn[e] for e in item.environments], axis=1), column
        ]
        if item.kind == "share":
            counts = _counts(resampled)
            tail = (1.0 - CENTRAL) / 2.0
            low, high = np.quantile(counts, (tail, 1.0 - tail), method="inverted_cdf")
            hits = (counts >= low) & (counts <= high)
            entry |= {"pilot count": int(known.sum()), "interval": [int(low), int(high)]}
            entry |= {"predicted": f"{int(low)}-{int(high)} of {entry['scored']}"}
        else:
            codes = _codes(resampled)
            odds = {v: float(np.mean(codes == c)) for c, v in enumerate(VERDICTS)}
            if item.kind == "clause":
                odds = {"kept": odds["below"] + odds["across"], "broken": odds["above"]}
            best = max(odds, key=odds.__getitem__)
            hits = _reading(item.kind, codes) == best
            entry |= {"pilot mean": float(known.mean()), "pilot sd": float(known.std(ddof=1))}
            entry |= {"chances": odds, "predicted": best}
        entry["chance"] = float(np.mean(hits))
        held += hits
        entries.append(entry)
    return Forecasts(tuple(entries), held, draws, seed)


def _codes(values: np.ndarray) -> np.ndarray:
    """Each row's verdict, 0 `below`, 1 `across` or 2 `above`: where the 95 % Student-t interval
    of the mean of its values that are numbers lies against nought, strictly; `across` on a row of
    fewer than two."""
    seen = ~np.isnan(values)
    n = seen.sum(axis=1)
    ok = n >= 2
    mean = np.divide(np.where(seen, values, 0.0).sum(axis=1), n, out=np.zeros(n.size), where=ok)
    squares = np.where(seen, values - mean[:, None], 0.0) ** 2
    sd = np.sqrt(np.divide(squares.sum(axis=1), n - 1, out=np.zeros(n.size), where=ok))
    half = stats.t.ppf(0.5 + LEVEL / 2.0, np.maximum(n - 1, 1)) * sd / np.sqrt(np.maximum(n, 1))
    codes = np.ones(n.size, dtype=int)
    codes[ok & (mean + half < 0.0)] = 0
    codes[ok & (mean - half > 0.0)] = 2
    return codes


def _counts(values: np.ndarray) -> np.ndarray:
    """Each row's count of the worlds where a share item holds."""
    return np.rint(np.where(np.isnan(values), 0.0, values).sum(axis=1)).astype(int)


def _reading(kind: Kind, codes: np.ndarray) -> np.ndarray:
    """The verdicts `codes` give, read as a `kind` item reads them."""
    if kind == "clause":
        return np.where(codes == 2, "broken", "kept")
    return np.asarray(VERDICTS)[codes]


def values(
    records: Sequence[WorldRecord], item: Item, reader: Reader = quantity
) -> tuple[np.ndarray, int]:
    """`item`'s quantity on each of `records`' worlds it reads where it is scored, and how many
    of them it is not scored on."""
    read, missing = [], 0
    for record in records:
        if record.k != item.k or record.environment not in item.environments:
            continue
        value = reader(record, item)
        if value is None:
            missing += 1
        else:
            read.append(value)
    return np.asarray(read, dtype=float), missing


def outcome(entry: Mapping[str, Any], read: np.ndarray) -> tuple[str, bool]:
    """What a scored run's values `read` of an item read, and whether `entry`'s forecast held.

    Raises:
        ValueError: on an entry that carries no forecast.
    """
    if entry["predicted"] is None:
        msg = f"no forecast was made on {entry['item']}: {entry['reason']}"
        raise ValueError(msg)
    kind = entry["item"]["kind"]
    if kind == "share":
        (count,) = _counts(read[None, :])
        low, high = entry["interval"]
        return f"{count} of {read.size}", bool(low <= count <= high)
    (reading,) = _reading(kind, _codes(read[None, :]))
    return str(reading), bool(reading == entry["predicted"])
