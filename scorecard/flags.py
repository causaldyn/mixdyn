"""Each flag a run's records hold, read as a classifier of a bad outcome: its AUROC over the run's
worlds with a bootstrap interval; and the fit's in-sample R² against the plan's regret.

**A flag** is a number an arm's record gives about its fit or its plan that a reader would act
on, read in the direction that raises an alarm (`ALARMS`): the engine's health card (ADR 0060)
and sampler health, the share of a plan's spend outside the logged range; a tool's own, Robyn's
NRMSE, DECOMP.RSSD and R², Meridian's R-hat, its reviewer's statuses and their numbers (the
posterior predictive p-value, the probability of a negative baseline, R²), PyMC-Marketing's R-hat
and divergences, Orbit's ELBO drift. A status reads PASS, then REVIEW, then FAIL as more alarming.
The direction is stated here and never fitted to a run, so a flag that tells the outcomes apart
the wrong way round reads below 0.5. A record's number no entry names, as a fitted parameter, is
not a flag and is not read.

**The outcomes**, each read off a world's score (`causaldyn_bench.scorecard.scoring`):

* `return`: a channel's return is wrong, its ROI's relative error above `tau`: the mean of the
  arm's draws of its ROI against the world's, where the record's response is read. A flag of one
  channel, a key that names the channel as a word, is read against that channel's error; a flag
  of the fit, against some channel's error above `tau`.
* `regret`: the plan's regret per euro is above the status quo's. A channel's flag is read here at
  its most alarming channel.

**The AUROC** is Mann and Whitney's: the chance that a world with the outcome reads the flag more
alarming than one without, a tie counted half. Its interval is the percentile bootstrap over the
worlds, `RESAMPLES` resamples from `SEED`, a world's channels drawn together, so that the channels
of one fit are not read as apart; every flag is read on the same resamples. A flag whose AUROC is
below 0.5 is dropped, the reason printed; one whose interval lies above 0.5 tells the outcomes
apart. A flag every world of which reads one outcome, or one value, is not read.

**R² against regret.** Where a record holds the fit's R² in sample (`FITS`), it is read against
the plan's regret per euro by Spearman's rank correlation, with the same bootstrap's interval: a
fit that reads the history well need not plan well, and the correlation says how far it does.

A run is one family, read here rung by rung over its environments: pooled over families or rungs,
a flag that tells families apart would read as one that tells worlds apart.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from causaldyn_bench.scorecard.scoring import BUILT_IN
from scipy import stats

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from causaldyn_bench.scorecard.scoring import ArmScore, WorldRecord

type Outcome = Literal["return", "regret"]

TAU = 0.5
"""The relative error of a channel's ROI above which its return counts as wrong. A placeholder:
the value a scored run reads is the one its pre-registration freezes."""
RESAMPLES = 2000
SEED = 20261106
"""The bootstrap's seed: any fixed one, so that a run's intervals are facts of the run."""
LEVEL = 0.95
STATUS = {"PASS": 0.0, "REVIEW": 1.0, "FAIL": 2.0}
"""A reviewer's status, in the order of alarm."""
CHANNEL = "<channel>"
"""Where a channel's flag names its channel."""


def _higher(value: float) -> float:
    return value


def _lower(value: float) -> float:
    return -value


def _away(value: float) -> float:
    return abs(value)


def _tail(value: float) -> float:
    return abs(value - 0.5)


ALARMS: tuple[tuple[str, Callable[[float], float]], ...] = (
    ("max rhat", _higher),
    ("divergences", _higher),
    ("max treedepth", _higher),
    ("min ess bulk", _lower),
    ("min e-bfmi", _lower),
    ("pareto k above", _higher),
    ("fitted not mixed", _higher),
    ("converged", _lower),
    ("at an end", _higher),
    ("flat", _higher),
    ("on its bound", _higher),
    ("nrmse", _higher),
    ("decomp rssd", _higher),
    ("calibration mape", _higher),
    ("rsq train", _lower),
    ("r2", _lower),
    ("health score", _lower),
    ("ppp", _lower),
    ("negative baseline", _higher),
    ("prior shift", _away),
    ("prior width", _higher),
    ("prior consistency", _tail),
    ("out of range", _higher),
    ("elbo drift", _away),
    ("holdout rmse", _higher),
    ("prediction gap", _higher),
)
"""Each flag by the words its key ends with, and how its value reads as an alarm: as it is,
negated where a lower value alarms, its distance from 0 for a shift, from 0.5 for a quantile.
The width of a return's posterior over its prior's alarms high: the data moved the prior little."""
FITS = ("r2", "rsq train")
"""The flags that are the fit's R² in sample."""
REVIEW = {
    ("ConvergenceCheck", "rhat"): "review max rhat",
    ("BayesianPPPCheck", "bayesian_ppp"): "review ppp",
    ("BaselineCheck", "negative_baseline_prob"): "review negative baseline",
    ("GoodnessOfFitCheck", "r_squared"): "review r2",
}
"""The numbers of Meridian's reviewer's checks a record keeps, by check and detail, and the flag
each is read as."""


def _words(key: str) -> str:
    return re.sub(r"[\s_]+", " ", key.strip()).lower()


def alarm(key: str) -> Callable[[float], float] | None:
    """How flag `key`'s value reads as an alarm, by the longest entry of `ALARMS` its words end
    with; None where none does."""
    words = _words(key)
    for suffix, reads in sorted(ALARMS, key=lambda entry: -len(entry[0])):
        if words == suffix or words.endswith(" " + suffix):
            return reads
    return None


def named(record: Mapping[str, Any]) -> dict[str, Any]:
    """A record's flags as it holds them: its `flags` and the numbers of a reviewer's checks
    (`REVIEW`)."""
    found: dict[str, Any] = dict(record.get("flags") or {})
    review = record.get("review")
    checks = review.get("checks") if isinstance(review, dict) else None
    if isinstance(checks, dict):
        for (check, detail), name in REVIEW.items():
            details = (checks.get(check) or {}).get("details")
            if isinstance(details, dict) and detail in details:
                found[name] = details[detail]
    return found


def _finite(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))


def numbers(record: Mapping[str, Any]) -> dict[str, float]:
    """A record's flags (`named`), each read as an alarm, the higher the more: a status by
    `STATUS`, a number by `ALARMS`. A flag that is neither a finite number nor a status, or that
    no entry of `ALARMS` names, is left out."""
    found: dict[str, float] = {}
    for key, value in named(record).items():
        if isinstance(value, str) and value in STATUS:
            found[key] = STATUS[value]
            continue
        reads = alarm(key)
        if reads is not None and _finite(value):
            found[key] = reads(float(value))
    return found


def _named(key: str, channels: Sequence[str]) -> tuple[str, str | None]:
    """A flag's name with the channel it names, as a word, replaced by `CHANNEL`, and that
    channel; the key and None where it names no channel or more than one."""
    words = key.split(" ")
    at = [i for i, word in enumerate(words) if word in channels]
    if len(at) != 1:
        return key, None
    channel = words[at[0]]
    words[at[0]] = CHANNEL
    return " ".join(words), channel


def wrong(record: WorldRecord, score: ArmScore, tau: float) -> dict[str, bool] | None:
    """Each channel's return wrong by `tau` in `score`'s draws (see the module); None where the
    score reads no draws of the returns."""
    if score.returns is None or not record.returns:
        return None
    return {
        name: abs(score.returns[name]["roi"][0] - truth["roi"]) > tau * abs(truth["roi"])
        for name, truth in record.returns.items()
    }


@dataclass(frozen=True, slots=True)
class Units:
    """What one flag's AUROC against one outcome reads: each unit's world, by its place among
    the records read, its alarm and whether it has the outcome."""

    world: np.ndarray
    value: np.ndarray
    outcome: np.ndarray


def units(
    records: Sequence[WorldRecord], arm: str, outcome: Outcome, tau: float = TAU
) -> dict[str, Units]:
    """Each flag of `arm`'s records with the units its AUROC against `outcome` reads, a flag of a
    channel by its name with `CHANNEL`. A world whose plan failed holds no flag of the arm."""
    rows: dict[str, list[tuple[int, float, bool]]] = {}
    for w, record in enumerate(records):
        score = record.arms.get(arm)
        if score is None or score.failure is not None:
            continue
        flags = numbers(score.record)
        if not flags:
            continue
        channels = tuple(record.returns)
        if outcome == "regret":
            bad = score.regret > record.arms[BUILT_IN[0]].regret
            most: dict[str, float] = {}
            for key, value in flags.items():
                name, _ = _named(key, channels)
                most[name] = max(value, most.get(name, -math.inf))
            for name, value in most.items():
                rows.setdefault(name, []).append((w, value, bad))
            continue
        errors = wrong(record, score, tau)
        if errors is None:
            continue
        for key, value in flags.items():
            name, channel = _named(key, channels)
            bad = errors[channel] if channel is not None else any(errors.values())
            rows.setdefault(name, []).append((w, value, bad))
    return {
        name: Units(
            np.array([r[0] for r in found], dtype=np.int64),
            np.array([r[1] for r in found], dtype=float),
            np.array([r[2] for r in found], dtype=bool),
        )
        for name, found in rows.items()
    }


def auroc(value: np.ndarray, outcome: np.ndarray, weight: np.ndarray | None = None) -> float:
    """Mann and Whitney's AUROC of `value` at telling `outcome` from its absence, each unit
    counted `weight` times, a tie half; NaN where either class has no weight."""
    w = np.ones((1, value.size)) if weight is None else np.asarray(weight, dtype=float)[None]
    return float(_aurocs(value, outcome, w)[0])


def _aurocs(value: np.ndarray, outcome: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """`auroc` under each row of `weights`, `(resamples, units)`, at once."""
    levels, place = np.unique(value, return_inverse=True)
    rows, size = weights.shape[0], levels.size
    at = (place[None, :] + size * np.arange(rows)[:, None]).ravel()

    def summed(w: np.ndarray) -> np.ndarray:
        return np.bincount(at, w.ravel(), minlength=rows * size).reshape(rows, size)

    up, down = summed(weights * outcome), summed(weights * ~outcome)
    below = np.cumsum(down, axis=1) - down
    total = up.sum(axis=1) * down.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0.0, np.sum(up * (below + 0.5 * down), axis=1) / total, np.nan)


def resamples(worlds: int, count: int = RESAMPLES, seed: int = SEED) -> np.ndarray:
    """How often each of `worlds` worlds is drawn in each of `count` bootstrap resamples,
    `(count, worlds)`."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(worlds, np.full(worlds, 1.0 / worlds), size=count)


@dataclass(frozen=True, slots=True)
class Read:
    """A statistic over a run's worlds with its bootstrap interval: `worlds` read, `units` among
    them, `with_outcome` units having the outcome where it is an AUROC."""

    value: float
    low: float
    high: float
    worlds: int
    units: int
    with_outcome: int


def _interval(draws: np.ndarray) -> tuple[float, float]:
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return math.nan, math.nan
    tail = (1.0 - LEVEL) / 2.0
    low, high = np.quantile(finite, [tail, 1.0 - tail])
    return float(low), float(high)


def scored(u: Units, drawn: np.ndarray) -> Read | None:
    """`u`'s AUROC and its interval over the resamples `drawn` (`resamples`, over every world
    of the records `u` was read from); None where the units hold one outcome or one value."""
    if u.outcome.all() or not u.outcome.any() or np.ptp(u.value) == 0.0:
        return None
    draws = _aurocs(u.value, u.outcome, drawn[:, u.world].astype(float))
    low, high = _interval(draws)
    return Read(
        auroc(u.value, u.outcome),
        low,
        high,
        int(np.unique(u.world).size),
        int(u.value.size),
        int(u.outcome.sum()),
    )


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman's rank correlation of `x` and `y`, ties at their mean rank; NaN where either is
    one value."""
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return math.nan
    return float(np.corrcoef(stats.rankdata(x), stats.rankdata(y))[0, 1])


def fit_against_regret(
    records: Sequence[WorldRecord], arm: str, drawn: np.ndarray
) -> dict[str, Read]:
    """Each R² flag of `arm`'s records (`FITS`), its rank correlation with the plan's regret per
    euro over the worlds that hold it, and its interval over the resamples `drawn`."""
    found: dict[str, list[tuple[int, float, float]]] = {}
    for w, record in enumerate(records):
        score = record.arms.get(arm)
        if score is None or score.failure is not None:
            continue
        for key, value in named(score.record).items():
            words = _words(key)
            if _finite(value) and any(words == s or words.endswith(" " + s) for s in FITS):
                found.setdefault(key, []).append((w, float(value), score.regret))
    out = {}
    for key, rows in found.items():
        world = np.array([r[0] for r in rows], dtype=np.int64)
        r2, regret = np.array([r[1] for r in rows]), np.array([r[2] for r in rows])
        value = spearman(r2, regret)
        if not math.isfinite(value):
            continue
        draws = np.array(
            [
                spearman(np.repeat(r2, counts[world]), np.repeat(regret, counts[world]))
                for counts in drawn
            ]
        )
        low, high = _interval(draws)
        out[key] = Read(value, low, high, world.size, world.size, 0)
    return out


def lines(records: Sequence[WorldRecord], tau: float = TAU) -> list[str]:
    """The report's lines: every arm's flags read against each outcome, those below 0.5
    dropped with the reason, and the R² against regret."""
    if not records:
        return []
    drawn = resamples(len(records))
    kept: list[str] = []
    dropped: list[str] = []
    unread = 0
    fits: list[str] = []
    arms = [a for a in records[0].arms if a not in BUILT_IN]
    for arm in arms:
        for outcome in ("return", "regret"):
            for name, u in sorted(units(records, arm, outcome, tau).items()):
                read = scored(u, drawn)
                if read is None:
                    unread += 1
                    continue
                shown = (
                    f"{arm} {name!r} against {outcome}: {read.value:.3f} "
                    f"[{read.low:.3f}, {read.high:.3f}], {read.with_outcome} of {read.units}"
                    f" with it over {read.worlds} worlds"
                )
                if read.value < 0.5:
                    dropped.append(f"    {shown}: below 0.5, the flag alarms the wrong way")
                elif read.low > 0.5:
                    kept.append(f"    {shown}; tells them apart")
                else:
                    kept.append(f"    {shown}")
        for key, read in sorted(fit_against_regret(records, arm, drawn).items()):
            fits.append(
                f"    {arm} {key!r}: {read.value:+.3f} [{read.low:+.3f}, {read.high:+.3f}]"
                f" over {read.worlds} worlds"
            )
    out = [
        f"  flags as classifiers of a channel's ROI error above {tau:g} (return) and of regret"
        f" above the status quo's (regret), AUROC [{LEVEL * 100:g} % interval, bootstrap over"
        " worlds]:"
    ]
    out += kept or ["    no flag is read"]
    if dropped:
        out.append("  dropped:")
        out += dropped
    if unread:
        out.append(f"  {unread} flag readings not read: their worlds hold one outcome or one value")
    out.append(
        f"  in-sample R² against regret per euro, Spearman's rank correlation [{LEVEL * 100:g} %]:"
    )
    out += fits or ["    no record holds an R²"]
    return out
