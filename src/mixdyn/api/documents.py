"""Versioned JSON documents for what the engine returns.

A document is `{"schema": name, "version": n, "data": value}`. The value is encoded by its declared
type, not by inspecting it:

- a dataclass as an object of its fields;
- a union of dataclasses as that object plus a `"kind"` naming the class;
- an enum by its value;
- a date as ISO 8601;
- a float with NaN written as null;
- a NumPy array as `{"dtype", "shape", "data"}`, its data flattened in C order.

Reading checks every value against the declared type, rebuilds dataclasses through their
constructors (so their invariants run), and names the path of the first value that does not fit.
A document of another schema or version is refused, not guessed at.

`LIFT_TESTS` states what each lift test a model reads estimated (`estimands`): the spend it
withheld or added, over which periods, at which level against the plan, and whether its lift is
read against zero spend or as a marginal change between two levels on the channel's curve.

`FACTSHEET` states what a fitted model is and what it rests on: its data's sources and transforms,
its priors and where they come from, each test's estimand and weeks, the holdout, the return
ranges, the tests' power audit, the refresh cadence and the limitations. `factsheet` writes the
one-geo Bayesian model's, `geo_factsheet` the model over geos', `dlm_factsheet` the DLM's and
`ridge_factsheet` the ridge's, each read off its own fit into the one schema. `DECISION` is a
plan on one page (`decision`): the plan, the fit's health checks, the contributions reconciled to
the KPI period by period, the next lift tests by their value, the decision's progress and its
compute record. `render` writes either as plain text. A field with nothing to record holds
`Unfilled` and its reason, never a silent blank.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
import reprlib
import types
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeAliasType, Union, get_args, get_origin, get_type_hints

import numpy as np
import numpy.typing as npt

from mixdyn.calibration import LiftEstimate, LiftSeries
from mixdyn.data import Channel, Issue, Revenue
from mixdyn.models.bayes_mmm import HillCurve, LogNormal, RoiPrior, StudentT, TanhCurve
from mixdyn.models.dlm import vague_prior
from mixdyn.transforms import Geometric, WeibullCdf
from mixdyn.transforms.saturation import HillShape

if TYPE_CHECKING:
    from collections.abc import Sequence

    from _typeshed import DataclassInstance
    from chc.dlm import Prior as DlmPrior

    from mixdyn.data import Geo, MediaPanel
    from mixdyn.decisions import PosteriorPlan
    from mixdyn.experiments import ExperimentChoice
    from mixdyn.inference.diagnostics import HealthCard
    from mixdyn.inference.sensitivity import MsmSensitivity, RidgeSensitivity
    from mixdyn.models.bayes_geo import BayesGeoFit, BayesGeoSpec
    from mixdyn.models.bayes_mmm import BayesSpec, ReturnPrior
    from mixdyn.models.dlm import DlmFit, DlmSpec
    from mixdyn.models.dlm_decomposition import DlmDecomposition
    from mixdyn.models.ridge_mmm import Control, FourierSeason, LinearTrend, RidgeFit, RidgeSpec
    from mixdyn.transforms.saturation import Transform


class SchemaError(ValueError):
    """A value or document does not fit its schema; the message names where."""


@dataclass(frozen=True, slots=True)
class Schema:
    """A document type: its name, its version and the declared type of its data."""

    name: str
    version: int
    data: Any


ISSUES = Schema("mixdyn.issues", 1, tuple[Issue, ...])
"""What reading a panel found: `MediaPanel.issues`, or `PanelError.issues`."""

_CHANGES = ("withheld", "added", "moved")
_READINGS = ("lift against zero spend", "marginal change")


@dataclass(frozen=True, slots=True)
class Estimand:
    """What one lift test of `channel` estimated, as a model reads it.

    The test changed the spend of `geos`, a `share` of the market (unnamed where empty, the whole
    market at a share of 1), over `periods` periods of its series: it `withheld` spend, `added`
    it, or `moved` it both ways.
    `spend` is the group's mean spend a period over those periods as the test ran and `planned` as
    planned, in the panel's currency. Its lift is read over the `readout`'s last periods of the
    series. Where either level is zero over every changed period, the lift is the curve's
    response to the spend `against zero spend`; otherwise it is a `marginal change` from one
    level to the other, which says nothing of the response below the lower. `first` and `last`
    are the dates of the first and last changed periods and `end` the readout's last, where the
    test is placed in the calendar. `estimand` states all of it in one line.

    Raises:
        ValueError: on a change or reading not one of those named.
    """

    channel: str
    geos: tuple[str, ...]
    share: float
    change: str
    reading: str
    periods: int
    readout: int
    spend: float
    planned: float
    first: dt.date | None
    last: dt.date | None
    end: dt.date | None
    estimand: str

    def __post_init__(self) -> None:
        if self.change not in _CHANGES or self.reading not in _READINGS:
            msg = (
                f"a test's change is one of {_CHANGES} and its reading one of {_READINGS}, got "
                f"{self.change!r} and {self.reading!r}"
            )
            raise ValueError(msg)


LIFT_TESTS = Schema("mixdyn.lift_tests", 1, tuple[Estimand, ...])
"""What each lift test a model reads estimated: `estimands`."""


def estimands(
    panel: MediaPanel, estimates: Mapping[str, Sequence[LiftEstimate | LiftSeries]]
) -> tuple[Estimand, ...]:
    """Each test of `estimates`, each channel's in the order given, as the `Estimand` it reads
    on `panel`'s calendar, cadence and currency. A test with no `end`, as a `LiftSeries`, is not
    placed in the calendar.

    Raises:
        ValueError: on a test whose end is not one of the panel's dates, or whose changed spend
            or readout starts before the panel's first.
    """
    out = []
    for channel, given in estimates.items():
        for test in given:
            estimate = test.summed if isinstance(test, LiftSeries) else test
            out.append(_estimand(panel, str(channel), estimate))
    return tuple(out)


def _estimand(panel: MediaPanel, channel: str, e: LiftEstimate) -> Estimand:
    changed = np.flatnonzero(e.spend != e.planned)
    spend, planned = e.spend[changed], e.planned[changed]
    change = (
        "withheld" if np.all(spend < planned) else "added" if np.all(spend > planned) else "moved"
    )
    zero = bool(np.all(spend == 0.0) or np.all(planned == 0.0))
    reading = _READINGS[0] if zero else _READINGS[1]
    first = last = end = None
    if e.end is not None:
        at = np.flatnonzero(panel.dates == e.end)
        start = int(at[0]) + 1 - e.spend.size if at.size == 1 else -1
        if start + min(int(changed[0]), e.spend.size - e.readout) < 0:
            msg = (
                f"a test of {channel!r} changes its spend or reads its lift outside the panel's "
                f"calendar from {panel.dates[0]} to {panel.dates[-1]}, its series ending {e.end}"
            )
            raise ValueError(msg)
        first, last, end = (
            _date(panel.dates[start + i]) for i in (changed[0], changed[-1], e.spend.size - 1)
        )
    geos = tuple(sorted(e.geos)) if e.geos is not None else ()
    level, plan = float(spend.mean()), float(planned.mean())
    unit, currency = panel.cadence.value, panel.currency
    if geos:
        where = f"in {', '.join(geos)} ({e.share:.6g} of the market)"
    elif e.share < 1:
        where = f"in {e.share:.6g} of the market, its geos not named"
    else:
        where = "the whole market"
    when = (
        f"from {first} to {last}"
        if first is not None
        else "of its series, not placed in the panel's calendar"
    )
    to = f" to {end}" if end is not None else ""
    read = (
        "as a lift against zero spend"
        if zero
        else f"as a marginal change from {min(level, plan):.6g} to {max(level, plan):.6g} "
        f"{currency} a {unit}"
    )
    line = (
        f"{channel}, {where}: spend {change} over {_count(changed.size, unit)} {when}, at "
        f"{level:.6g} {currency} a {unit} against {plan:.6g} planned; the lift is read over the "
        f"{_count(e.readout, unit)} of the readout{to} {read}"
    )
    return Estimand(
        channel=channel,
        geos=geos,
        share=e.share,
        change=change,
        reading=reading,
        periods=int(changed.size),
        readout=e.readout,
        spend=level,
        planned=plan,
        first=first,
        last=last,
        end=end,
        estimand=line,
    )


def _date(value: np.datetime64) -> dt.date:
    return dt.date.fromisoformat(str(value.astype("datetime64[D]")))


def _count(n: int, unit: str) -> str:
    return f"{n} {unit}" if n == 1 else f"{n} {unit}s"


@dataclass(frozen=True, slots=True)
class Unfilled:
    """A field of a document with nothing to record, and the `reason` it has none.

    Raises:
        ValueError: on a reason that is blank.
    """

    reason: str

    def __post_init__(self) -> None:
        if not self.reason.strip():
            msg = "an unfilled field states why it is unfilled"
            raise ValueError(msg)


_NOT_STATED = "not stated by the caller"
"""The origin a `Source` reads where the caller states none."""
_ENGINE_MADE = "made by the engine"
"""The origin of a calendar column the caller states none for."""
_NO_HOLDOUT = Unfilled("no periods were held out of the fit: its forecast was not checked on any")
"""What a factsheet's `holdout` reads until the caller records one."""
_NO_POWER_AUDIT = Unfilled(
    "no power audit was given: read one with `mixdyn.experiments.audit_power` and pass its "
    "backtests and its minimum detectable effect's range, in the KPI's units"
)
"""What a factsheet's `power` reads until a power audit fills it."""
_NO_REFRESH = Unfilled("no refresh cadence was stated")
"""What a factsheet's `refresh` reads until the caller states one."""


@dataclass(frozen=True, slots=True)
class Source:
    """A column the model reads: its `column` name, its `role` (`"kpi"`, `"spend"`, `"control"` or
    `"calendar"`), its `origin`, where its data came from as the caller states it, and the
    `transforms` the reader and the model ran on it before the likelihood read it, in order."""

    column: str
    role: str
    origin: str
    transforms: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Prior:
    """A prior the model states: the `parameter` it is on, its `distribution` on the model's
    scales, and its `provenance`, where its numbers come from."""

    parameter: str
    distribution: str
    provenance: str


@dataclass(frozen=True, slots=True)
class Holdout:
    """The periods from `first` to `last`, `periods` of them, held out of the fit, and how far its
    forecast of the KPI missed them: `error` by the `metric` named.

    Raises:
        ValueError: on a last date before the first, or fewer than one period.
    """

    first: dt.date
    last: dt.date
    periods: int
    metric: str
    error: float

    def __post_init__(self) -> None:
        if self.last < self.first or self.periods < 1:
            msg = (
                f"a holdout runs from its first date to its last over at least one period, got "
                f"{self.first} to {self.last} over {self.periods}"
            )
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ReturnRange:
    """A `channel`'s return over a posterior's draws: their `mean` and the central `level`
    interval from `low` to `high`, in the KPI's units a currency unit, and `counts`, what the
    return counts (`RoiPrior.counts`): the window's contribution over its spend, or the window's
    spend's whole return with its carryover."""

    channel: str
    counts: str
    mean: float
    low: float
    high: float
    level: float


@dataclass(frozen=True, slots=True)
class PowerSummary:
    """How the tests' design was audited before they ran: `backtests`, the number of placebo
    windows of the history the design was read on, and `mde_low` to `mde_high`, the range of its
    minimum detectable effect over them, in the KPI's units.

    Raises:
        ValueError: on fewer than one backtest, or a range whose ends are not finite and in order.
    """

    backtests: int
    mde_low: float
    mde_high: float

    def __post_init__(self) -> None:
        if self.backtests < 1 or not (
            math.isfinite(self.mde_low) and self.mde_low <= self.mde_high < math.inf
        ):
            msg = (
                f"a power audit reads at least one backtest and a finite range in order, got "
                f"{self.backtests} and {self.mde_low} to {self.mde_high}"
            )
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Refresh:
    """How often the model is fitted again: every `every` periods of `cadence`, last on `last`.

    Raises:
        ValueError: on fewer than one period.
    """

    every: int
    cadence: str
    last: dt.date

    def __post_init__(self) -> None:
        if self.every < 1:
            msg = f"a model is fitted again every period or more, got {self.every}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Factsheet:
    """What a fitted media-mix model is and what it rests on, in fields that follow the Coalition
    for Innovative Media Measurement's (CIMM) roadmap for media-mix models.

    `model` names the model; `kpi` the KPI and its unit, `cadence` the periods', `first` and `last`
    the calendar's first and last dates, and `geos` and `channels` what it reads. `sources` holds
    each column the model reads with its origin and transforms; `priors` each prior with its
    provenance; `tests` each lift test as `estimands` reads it, its weeks and its estimand;
    `holdout` the periods held out of the fit; `returns` each channel's return range; `power` the
    tests' power audit; `refresh` the refit cadence; and `limitations` what the numbers do not
    say. A field with nothing to record is `Unfilled`, with its reason.
    """

    model: str
    kpi: str
    cadence: str
    first: dt.date
    last: dt.date
    geos: tuple[str, ...]
    channels: tuple[str, ...]
    sources: tuple[Source, ...]
    priors: tuple[Prior, ...]
    tests: tuple[Estimand, ...]
    holdout: Holdout | Unfilled
    returns: tuple[ReturnRange, ...]
    power: PowerSummary | Unfilled
    refresh: Refresh | Unfilled
    limitations: tuple[str, ...]


FACTSHEET = Schema("mixdyn.factsheet", 1, Factsheet)
"""What a fitted model is and what it rests on: `factsheet`."""


@dataclass(frozen=True, slots=True)
class Spend:
    """A `channel`'s spend a period, in the panel's currency: the `current` and the `planned`."""

    channel: str
    current: float
    planned: float


@dataclass(frozen=True, slots=True)
class Health:
    """The health checks of a fit (`mixdyn.inference.diagnostics.HealthCard`), each read against
    the threshold the caller passed: `max_rhat`, `ppp` and its Monte Carlo error `ppp_se`,
    `negative_baseline`, and each channel's `shift`, `width` and `consistency`, as the card holds
    them; `thresholds` the thresholds passed, by `HealthCard.flags`' keywords, and `flags` whether
    each cell passes, as `HealthCard.flags` reads it."""

    max_rhat: float
    ppp: float
    ppp_se: float
    negative_baseline: float
    shift: Mapping[str, float]
    width: Mapping[str, float]
    consistency: Mapping[str, float]
    thresholds: Mapping[str, float]
    flags: Mapping[str, bool]


@dataclass(frozen=True, slots=True)
class Period:
    """One period of a reconciliation: its `date`, the `kpi` observed, its `parts`, each part's
    contribution in the order `Reconciliation` names them, their `total` and the `residual`, the
    KPI less the total."""

    date: dt.date
    kpi: float
    parts: tuple[float, ...]
    total: float
    residual: float


@dataclass(frozen=True, slots=True)
class Reconciliation:
    """The contributions of `parts` to the KPI over the fitted `periods`, which sum to each
    period's total; the KPI is that total and the residual.

    Raises:
        ValueError: on a period with another number of parts.
    """

    parts: tuple[str, ...]
    periods: tuple[Period, ...]

    def __post_init__(self) -> None:
        if any(len(p.parts) != len(self.parts) for p in self.periods):
            msg = f"each period holds a contribution of each of the {len(self.parts)} parts"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class NextTest:
    """The next set of lift tests by its value to the plan (`ExperimentChoice`): the `verdict`, to
    `"test"` or to `"plan now"`; the set's `tests` by their labels and their `channels`; the regret
    they are expected to remove, `reduction`, and its Monte Carlo `error`, in the KPI's units; their
    `cost`, `net_value` and `predicted_net_value`, in revenue; and `evpi`, the value of perfect
    information, with its `evpi_error`, over `draws` outer draws.

    Raises:
        ValueError: on a verdict other than those named.
    """

    verdict: str
    tests: tuple[str, ...]
    channels: tuple[str, ...]
    reduction: float
    error: float
    cost: float
    net_value: float
    predicted_net_value: float
    evpi: float
    evpi_error: float
    draws: int

    def __post_init__(self) -> None:
        if self.verdict not in ("test", "plan now"):
            msg = f'a verdict is "test" or "plan now", got {self.verdict!r}'
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Event:
    """A step of the decision's progress: `what` happened `on` a date."""

    on: dt.date
    what: str


@dataclass(frozen=True, slots=True)
class Compute:
    """The compute a decision took as it was recorded: the `device` it ran on and the
    `wall_seconds` the caller measured. A record, not a claim of speed.

    Raises:
        ValueError: on seconds not finite and at or above 0.
    """

    device: str
    wall_seconds: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.wall_seconds < math.inf:
            msg = f"wall seconds are finite and at or above 0, got {self.wall_seconds}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Decision:
    """A plan on one page, with what it rests on.

    `title` names it. The plan (`PosteriorPlan`) is of `geo`'s channels over `periods` periods:
    `budget` over them, in `currency`; `gain` what its spend adds to the KPI over the periods and
    the carryover after them, and `current_gain` what the current spend adds, run the same way; and
    `spend`, each channel's current and planned spend a period. `health` holds the fit's health
    checks; `reconciliation` the contributions that sum to the KPI by period; `next_test` the next
    set of lift tests by its value; `events` the decision's progress, in order; and `compute` the
    compute record. A field with nothing to record is `Unfilled`, with its reason. `unidentified`
    names each channel the plan moves off its current spend whose spend the panel flags as never
    moving in the geo (`PosteriorPlan.unidentified`): its worth there is read from its prior alone.
    """

    title: str
    geo: str
    currency: str
    periods: int
    budget: float
    gain: float
    current_gain: float
    spend: tuple[Spend, ...]
    health: Health | Unfilled
    reconciliation: Reconciliation | Unfilled
    next_test: NextTest | Unfilled
    events: tuple[Event, ...]
    compute: Compute | Unfilled
    unidentified: tuple[str, ...] = ()


DECISION = Schema("mixdyn.decision", 1, Decision)
"""A plan on one page, with what it rests on: `decision`."""


def factsheet(
    panel: MediaPanel,
    geo: Geo,
    spec: BayesSpec,
    returns: Mapping[str, npt.ArrayLike],
    *,
    level: float = 0.9,
    origins: Mapping[str, str] | None = None,
    estimates: Mapping[str, Sequence[LiftEstimate | LiftSeries]] | None = None,
    holdout: Holdout | Unfilled = _NO_HOLDOUT,
    power: PowerSummary | Unfilled = _NO_POWER_AUDIT,
    refresh: Refresh | Unfilled = _NO_REFRESH,
    limitations: Sequence[str] = (),
    sensitivity: Sequence[MsmSensitivity] = (),
) -> Factsheet:
    """The factsheet of the one-geo Bayesian model `spec` (`mixdyn.models.bayes_mmm`) fitted to
    `geo` of `panel`.

    The engine fills what it knows: the panel's calendar, KPI and geo; each column the model reads
    as a `Source`, its origin the caller's in `origins` by column name (`"kpi"` for the KPI,
    `"trend"` and `"season"` for the calendar's) and its transforms the reader's and the model's;
    each prior the model states, with where its numbers come from; each test of `estimates` as
    `estimands` reads it; and each channel's `returns`, draws of its return as its prior counts it
    (`fit.draws["roi"][..., i]` of a `BayesFit`), read into a `ReturnRange` at `level`. `holdout`,
    `power` and `refresh` are the caller's records, each `Unfilled` with its reason by default:
    the engine runs no holdout, power audit or refresh schedule of its own. `limitations` are the
    caller's, then the engine's: a channel no test calibrates, the confounding no column holds
    and, for each channel of which `sensitivity` holds a result of `msm_sensitivity`
    (`mixdyn.inference.sensitivity`), how strong that confounding would have to be: ADR 0061's
    line, its `Gamma*` and its bound at `Gamma = 1.5` (`sensitivity_line`), in the model's order;
    then a holdout or a power audit not recorded.

    Raises:
        ValueError: on a geo, a channel or a control the panel does not have; returns not given
            for exactly the model's channels, or with no draw or one not finite; a level outside
            `(0, 1)`; an origin of a column, or a test of a channel, the model does not read; a
            sensitivity result of a channel the model does not read, two of one channel, or one
            not read at `Gamma = 1.5`; and what `estimands` refuses.
    """
    if geo not in panel.geos:
        msg = f"the panel has no geo {geo!r}: it has {list(panel.geos)}"
        raise ValueError(msg)
    channels = tuple(m.channel for m in spec.media)
    controls = tuple(c.column for c in spec.controls)
    unread = [c for c in channels if c not in panel.channels] + [
        c for c in controls if c not in panel.controls
    ]
    if unread:
        msg = f"the model reads {unread}, which the panel does not have"
        raise ValueError(msg)
    if set(returns) != set(channels):
        msg = f"returns are read for the model's channels {list(channels)}, got {list(returns)}"
        raise ValueError(msg)
    _level(level)
    calendar = (*(("trend",) if spec.trend else ()), *(("season",) if spec.season else ()))
    origins = _origins(origins, ("kpi", *channels, *controls, *calendar))
    estimates = _tested(estimates, channels)
    bounds = {s.channel: s for s in sensitivity}
    if len(bounds) < len(sensitivity) or not set(bounds) <= set(channels):
        msg = (
            f"a sensitivity result is read once at most for each of the model's channels"
            f" {list(channels)}, got {[s.channel for s in sensitivity]}"
        )
        raise ValueError(msg)
    return Factsheet(
        model=_model(spec),
        kpi=_kpi(panel),
        cadence=panel.cadence.value,
        first=_date(panel.dates[0]),
        last=_date(panel.dates[-1]),
        geos=(str(geo),),
        channels=tuple(str(c) for c in channels),
        sources=_sources(panel, geo, spec, origins),
        priors=_priors(spec),
        tests=estimands(panel, estimates),
        holdout=holdout,
        returns=tuple(
            _range(m.channel, m.roi.counts, returns[m.channel], level) for m in spec.media
        ),
        power=power,
        refresh=refresh,
        limitations=(
            *limitations,
            *_derived(channels, _calibrated(estimates), holdout, power, bounds=bounds),
        ),
    )


def _level(level: float) -> None:
    if not 0.0 < level < 1.0:
        msg = f"a return range's level is inside (0, 1), got {level}"
        raise ValueError(msg)


def _origins(origins: Mapping[str, str] | None, columns: Sequence[str]) -> dict[str, str]:
    """The caller's `origins`, each of a column the model reads.

    Raises:
        ValueError: on an origin of a column the model does not read.
    """
    given = dict(origins or {})
    stray = sorted(set(given) - set(columns))
    if stray:
        msg = f"origins name columns the model does not read: {stray}"
        raise ValueError(msg)
    return given


def _tested(
    estimates: Mapping[str, Sequence[LiftEstimate | LiftSeries]] | None, channels: Sequence[str]
) -> Mapping[str, Sequence[LiftEstimate | LiftSeries]]:
    """The caller's lift tests, each of a channel the model reads.

    Raises:
        ValueError: on a test of a channel the model does not read.
    """
    given = estimates or {}
    strays = sorted(set(given) - set(channels))
    if strays:
        msg = f"lift tests of {strays}, which the model does not read"
        raise ValueError(msg)
    return given


def _derived(
    channels: Sequence[str],
    tested: Collection[str],
    holdout: Holdout | Unfilled,
    power: PowerSummary | Unfilled,
    *,
    rests: str = "the observational data and the prior",
    bounds: Mapping[str, MsmSensitivity] | None = None,
) -> list[str]:
    """The limitations every factsheet's fields imply: a channel not among those a test calibrates,
    `tested`, its return resting on what `rests` names; the confounding no column holds, then how
    strong it would have to be for each channel of which `bounds` holds a sensitivity result
    (`sensitivity_line`), in the order of `channels`; and a holdout or a power audit not
    recorded."""
    bounds = bounds or {}
    derived = []
    untested = [c for c in channels if c not in tested]
    if untested:
        derived.append(
            f"no lift test calibrates {', '.join(untested)}: the return rests on {rests}"
        )
    derived.append(
        "the returns are read from observational data under the model's form: a confounder of "
        "spend and the KPI that no column holds biases them, which a sensitivity bound "
        "(mixdyn.inference.sensitivity) states and does not remove"
    )
    derived += [sensitivity_line(bayes=bounds[c]) for c in channels if c in bounds]
    if isinstance(holdout, Unfilled):
        derived.append("the fit's forecast was not checked on held-out periods")
    if isinstance(power, Unfilled):
        derived.append("the lift tests' power was not audited")
    return derived


def _calibrated(estimates: Mapping[str, Sequence[LiftEstimate | LiftSeries]]) -> set[str]:
    """The channels at least one of `estimates` tests."""
    return {c for c, given in estimates.items() if given}


def _kpi(panel: MediaPanel) -> str:
    return (
        f"revenue in {panel.currency}"
        if isinstance(panel.kpi_unit, Revenue)
        else f"a count of {panel.kpi_unit.unit}"
    )


def _model(spec: BayesSpec) -> str:
    curve = "Hill curves" if isinstance(spec.saturation, HillCurve) else "tanh curves"
    noise = (
        "Student-t"
        if isinstance(spec.likelihood, StudentT)
        else "log-normal"
        if isinstance(spec.likelihood, LogNormal)
        else "normal"
    )
    calendar = "".join([", a trend" if spec.trend else "", ", a season" if spec.season else ""])
    media, controls = _count(len(spec.media), "channel"), _count(len(spec.controls), "control")
    return (
        f"one-geo Bayesian media-mix model (mixdyn.models.bayes_mmm): {media} on {curve}, "
        f"{controls}{calendar}, {noise} noise"
    )


def _sources(
    panel: MediaPanel, geo: Geo, spec: BayesSpec, origins: Mapping[str, str]
) -> tuple[Source, ...]:
    filled = int(panel.filled[panel.geos.index(geo)].sum())
    kpi = ["read in units of its mean over the window's observed periods"]
    if filled:
        unread = _count(filled, panel.cadence.value)
        kpi.insert(0, f"{unread} with no row in the input read as unobserved")
    out = [Source("kpi", "kpi", origins.get("kpi", _NOT_STATED), tuple(kpi))]
    curve = (
        "saturated by a Hill, its half-saturation point and slope sampled"
        if isinstance(spec.saturation, HillCurve)
        else "saturated by tanh, its half-saturation point sampled"
    )
    for m in spec.media:
        steps = [
            "read in units of its mean positive spend over the panel's calendar",
            "carried over by a normalised geometric kernel, its retention sampled",
            curve,
        ]
        allotment = panel.national.get(Channel(m.channel))
        if allotment is not None:
            by = allotment.value
            steps.insert(0, f"bought nationally and split over the geos by {by}")
        out.append(Source(m.channel, "spend", origins.get(m.channel, _NOT_STATED), tuple(steps)))
    for c in spec.controls:
        steps = ("standardised over the window", *_held(c))
        out.append(Source(c.column, "control", origins.get(c.column, _NOT_STATED), steps))
    return (*out, *_calendar(spec.trend, spec.season, origins))


def _held(control: Control) -> tuple[str, ...]:
    """A control's coefficient's sign, as a step of its source."""
    if control.lower == 0.0:
        return ("its coefficient held at or above 0",)
    if control.upper == 0.0:
        return ("its coefficient held at or below 0",)
    return ()


def _calendar(
    trend: LinearTrend | None, season: FourierSeason | None, origins: Mapping[str, str], *also: str
) -> list[Source]:
    """The trend's and the season's sources, each with the steps `also` names."""
    out = []
    if trend:
        line = f"a line through the window's periods, bent at {_count(trend.knots, 'knot')}"
        out.append(Source("trend", "calendar", origins.get("trend", _ENGINE_MADE), (line, *also)))
    if season:
        line = f"harmonics 1 to {season.harmonics} of a season {season.period:g} periods long"
        out.append(Source("season", "calendar", origins.get("season", _ENGINE_MADE), (line, *also)))
    return out


def _priors(spec: BayesSpec) -> tuple[Prior, ...]:
    engine = "the engine's default"
    out = [
        Prior(
            f"roi[{m.channel}]",
            f"{_family(m.roi)}, the {_COUNTS[m.roi.counts]}",
            f"{engine}, Meridian's for a revenue KPI"
            if m.roi == RoiPrior()
            else "stated by the caller",
        )
        for m in spec.media
    ]
    out += [
        Prior(
            "retention",
            "Beta(1, 3), each channel's",
            f"{engine}, PyMC-Marketing's for a geometric decay",
        ),
        Prior(
            "half_saturation",
            "LogNormal(0, 1) in units of the channel's mean positive spend, each channel's",
            f"{engine}: half-saturation at the mean positive spend, by its median",
        ),
    ]
    if isinstance(spec.saturation, HillCurve):
        out.append(_slope(spec.saturation, engine))
    out.append(Prior("intercept", "Normal(1, 1) in units of the KPI's mean", engine))
    if spec.trend or spec.season:
        out.append(Prior("calendar", "Normal(0, 1), each column's coefficient", engine))
    if spec.controls:
        line = "Normal(0, 1), each standardised control's coefficient, held to its sign"
        out.append(Prior("control", line, engine))
    scale = (
        "the log's scale"
        if isinstance(spec.likelihood, LogNormal)
        else "in units of the KPI's mean"
    )
    out.append(Prior("sigma", f"HalfNormal(1), {scale}", engine))
    if isinstance(spec.likelihood, StudentT):
        line = "Gamma(2, 0.1), Student-t's degrees of freedom"
        out.append(Prior("nu", line, f"{engine}, Juárez and Steel (2010)"))
    return tuple(out)


def _family(prior: ReturnPrior) -> str:
    if isinstance(prior, RoiPrior):
        return f"LogNormal({prior.mu:g}, {prior.sigma:g})"
    return f"HalfNormal({prior.scale:g})"


_COUNTS = {
    "window": "window's contribution over its spend",
    "carried": "window's spend's whole return, its carryover counted",
}


def _range(channel: str, counts: str, draws: npt.ArrayLike, level: float) -> ReturnRange:
    x = np.asarray(draws, dtype=float).reshape(-1)
    if x.size == 0 or not np.all(np.isfinite(x)):
        msg = f"{channel!r}'s returns need at least one draw, each finite"
        raise ValueError(msg)
    low, high = np.quantile(x, [(1.0 - level) / 2.0, (1.0 + level) / 2.0])
    return ReturnRange(
        channel=channel,
        counts=_COUNTS[counts],
        mean=float(x.mean()),
        low=float(low),
        high=float(high),
        level=level,
    )


def geo_factsheet(
    panel: MediaPanel,
    fit: BayesGeoFit,
    *,
    level: float = 0.9,
    origins: Mapping[str, str] | None = None,
    estimates: Mapping[str, Sequence[LiftEstimate | LiftSeries]] | None = None,
    holdout: Holdout | Unfilled = _NO_HOLDOUT,
    power: PowerSummary | Unfilled = _NO_POWER_AUDIT,
    refresh: Refresh | Unfilled = _NO_REFRESH,
    limitations: Sequence[str] = (),
) -> Factsheet:
    """The factsheet of the Bayesian model over geos (`mixdyn.models.bayes_geo`), `fit`, fitted to
    every geo of `panel`, read off the fit as `factsheet` reads the one-geo model off its spec.

    The calendar is the fit's window, and the geos every geo. The sources hold the steps the model
    runs over the geos: a geo's KPI and spend each divided by its population where the panel has
    one, a channel's carryover and curve shared by the geos, its effect pooled over them. The
    priors are the model's, the spread's and the geos' bases' with them. Each channel's return is
    its return over the window and every geo at each of the fit's draws (`fit.draws["roi"]`), read
    into a `ReturnRange` at `level`. `estimates` are the lift tests the fit read, each as
    `estimands` reads it. `origins`, `holdout`, `power`, `refresh` and `limitations` are read as
    `factsheet` reads them; the engine's limitations add a channel bought nationally and split by
    population, whose effect the geos hold no contrast in.

    Raises:
        ValueError: on a fit of other geos or of dates off the panel's calendar, a level outside
            `(0, 1)`, an origin of a column, or a test of a channel, the model does not read, and
            what `estimands` refuses.
    """
    if fit.geos != panel.geos or not np.isin(fit.dates, panel.dates).all():
        msg = "the fit is not of this panel: its geos or its window's dates are not the panel's"
        raise ValueError(msg)
    spec = fit.spec
    channels = tuple(m.channel for m in spec.media)
    controls = tuple(c.column for c in spec.controls)
    _level(level)
    calendar = (*(("trend",) if spec.trend else ()), *(("season",) if spec.season else ()))
    origins = _origins(origins, ("kpi", *channels, *controls, *calendar))
    estimates = _tested(estimates, channels)
    national = [c for c in channels if Channel(c) in panel.national]
    split = (
        [
            f"{', '.join(national)} bought nationally and split over the geos by population: the "
            "geos hold no contrast in its spend, so its effect is read from the periods alone"
        ]
        if national and panel.population is not None
        else []
    )
    return Factsheet(
        model=_geo_model(spec, len(fit.geos)),
        kpi=_kpi(panel),
        cadence=panel.cadence.value,
        first=_date(fit.dates[0]),
        last=_date(fit.dates[-1]),
        geos=tuple(str(g) for g in fit.geos),
        channels=channels,
        sources=_geo_sources(panel, fit, origins),
        priors=_geo_priors(spec),
        tests=estimands(panel, estimates),
        holdout=holdout,
        returns=tuple(
            _range(c, "window", fit.draws["roi"][..., j], level) for j, c in enumerate(channels)
        ),
        power=power,
        refresh=refresh,
        limitations=(
            *limitations,
            *_derived(channels, _calibrated(estimates), holdout, power),
            *split,
        ),
    )


def _geo_model(spec: BayesGeoSpec, geos: int) -> str:
    curve = "tanh curves"
    if isinstance(spec.saturation, HillCurve):
        curve = "Hill curves of slope 1" if spec.slope == "fixed" else "Hill curves"
    noise = "Student-t" if isinstance(spec.likelihood, StudentT) else "normal"
    calendar = "".join([", a trend" if spec.trend else "", ", a season" if spec.season else ""])
    media, controls = _count(len(spec.media), "channel"), _count(len(spec.controls), "control")
    return (
        f"Bayesian media-mix model over geos (mixdyn.models.bayes_geo): {media} on {curve}, each "
        f"channel's effect pooled over {geos} geos, {controls}{calendar}, {noise} noise"
    )


def _geo_sources(
    panel: MediaPanel, fit: BayesGeoFit, origins: Mapping[str, str]
) -> tuple[Source, ...]:
    spec = fit.spec
    people = panel.population is not None
    head = " per head" if people else ""
    inside = np.isin(panel.dates, fit.dates)
    filled = int(panel.filled[:, inside].sum())
    kpi = [f"read in units of its mean{head} over the window's observed cells of every geo"]
    if people:
        kpi.insert(0, "divided by each geo's population")
    if filled:
        cells = f"{filled} of the geos' {panel.cadence.value}s"
        kpi.insert(0, f"{cells} with no row in the input read as unobserved")
    out = [Source("kpi", "kpi", origins.get("kpi", _NOT_STATED), tuple(kpi))]
    if isinstance(spec.saturation, TanhCurve):
        curve = "saturated by tanh, its half-saturation point sampled"
    elif spec.slope == "fixed":
        curve = "saturated by a Hill of slope 1, its half-saturation point sampled"
    else:
        curve = "saturated by a Hill, its half-saturation point and slope sampled"
    for m in spec.media:
        steps = [
            f"read in units of its mean positive spend{head} over the panel's calendar and geos",
            "carried over by a normalised geometric kernel, its retention sampled",
            curve,
            "its effect in each geo the channel's times the geo's own factor, pooled over the geos",
        ]
        if people:
            steps.insert(0, "divided by each geo's population")
        allotment = panel.national.get(Channel(m.channel))
        if allotment is not None:
            steps.insert(0, f"bought nationally and split over the geos by {allotment.value}")
        out.append(Source(m.channel, "spend", origins.get(m.channel, _NOT_STATED), tuple(steps)))
    for c in spec.controls:
        steps = ("standardised over the window's cells that were not filled", *_held(c))
        out.append(Source(c.column, "control", origins.get(c.column, _NOT_STATED), steps))
    moves = "moving each geo in proportion to its base"
    return (*out, *_calendar(spec.trend, spec.season, origins, moves))


def _geo_priors(spec: BayesGeoSpec) -> tuple[Prior, ...]:
    engine = "the engine's default"
    out = [
        Prior(
            f"roi[{m.channel}]",
            f"{_family(m.roi)}, the {_COUNTS['window']}, over every geo",
            f"{engine}, Meridian's for a revenue KPI"
            if m.roi == RoiPrior()
            else "stated by the caller",
        )
        for m in spec.media
    ]
    out += [
        Prior(
            "retention",
            "Beta(1, 3), each channel's",
            f"{engine}, PyMC-Marketing's for a geometric decay",
        ),
        Prior(
            "half_saturation",
            "LogNormal(0, 1) in units of the channel's mean positive spend per head, each "
            "channel's",
            f"{engine}: half-saturation at the mean positive spend, by its median",
        ),
    ]
    if isinstance(spec.saturation, HillCurve) and spec.slope == "sampled":
        out.append(_slope(spec.saturation, engine))
    out.append(
        Prior(
            "spread",
            f"HalfNormal({spec.spread:g}), each channel's log-SD of its effect over the geos",
            f"{engine}, Meridian's" if spec.spread == 1.0 else "stated by the caller",
        )
    )
    if spec.base_prior == "normal":
        line = "Normal(1, 1), each geo's base in units of the KPI's mean per head"
        out.append(Prior("base", line, "stated by the caller: the one-geo intercept's prior"))
    else:
        line = "LogNormal(0, 1), each geo's base in units of the KPI's mean per head"
        out.append(Prior("base", line, engine))
    if spec.trend or spec.season:
        line = "Normal(0, 1), each column's coefficient, a share of the geo's base"
        out.append(Prior("calendar", line, engine))
    if spec.controls:
        line = "Normal(0, 1), each standardised control's coefficient, held to its sign"
        out.append(Prior("control", line, engine))
    if spec.noise_prior == "log-normal":
        line = "LogNormal(log 0.6745, 0.5), in units of the KPI's mean per head"
        out.append(Prior("sigma", line, "stated by the caller: the half-normal's median"))
    else:
        line = "HalfNormal(1), in units of the KPI's mean per head"
        out.append(Prior("sigma", line, engine))
    if isinstance(spec.likelihood, StudentT):
        line = "Gamma(2, 0.1), Student-t's degrees of freedom"
        out.append(Prior("nu", line, f"{engine}, Juárez and Steel (2010)"))
    return tuple(out)


def dlm_factsheet(
    panel: MediaPanel,
    geo: Geo,
    fit: DlmFit,
    decomposition: DlmDecomposition,
    *,
    level: float = 0.9,
    origins: Mapping[str, str] | None = None,
    estimates: Mapping[str, Sequence[LiftEstimate | LiftSeries]] | None = None,
    holdout: Holdout | Unfilled = _NO_HOLDOUT,
    power: PowerSummary | Unfilled = _NO_POWER_AUDIT,
    refresh: Refresh | Unfilled = _NO_REFRESH,
    limitations: Sequence[str] = (),
) -> Factsheet:
    """The factsheet of the dynamic linear model (`mixdyn.models.dlm`), `fit`, fitted to `geo` of
    `panel`, its returns read off `decomposition`, draws of the fit's decomposition
    (`decompose_dlm`).

    A channel is an effect on one of the panel's channels: its spend carried over and saturated
    by its transform, or read as it is. Its return at each draw is its contribution over the fit's
    periods over its spend over them, read into a `ReturnRange` at `level`. The sources hold each
    effect's column with its transform, a lever's level, its coefficient's discount and sign. The
    priors are the state's and the variance's before the first period, `vague_prior`'s or the
    caller's, then each block's discount, the evolution the spec states. `estimates` are the lift
    tests the fit read, each as `estimands` reads it. `origins`, `holdout`, `power`, `refresh` and
    `limitations` are read as `factsheet` reads them; the engine's limitations add the transforms
    held at their values, coefficients that walk, and draws held to the signs that have not mixed.

    Raises:
        ValueError: on a geo the panel does not have, a fit of another series than the geo's KPI,
            a decomposition of another fit's periods or effects, a channel that spends nothing
            over the periods, a level outside `(0, 1)`, an origin of a column, or a test of a
            channel, the model does not read, and what `estimands` refuses.
    """
    if geo not in panel.geos:
        msg = f"the panel has no geo {geo!r}: it has {list(panel.geos)}"
        raise ValueError(msg)
    g = panel.geos.index(geo)
    if not (
        np.array_equal(fit.dates, panel.dates)
        and np.array_equal(fit.y, panel.kpi[g], equal_nan=True)
    ):
        msg = f"the fit is not of {geo!r}'s KPI over the panel's calendar"
        raise ValueError(msg)
    spec = fit.spec
    columns = tuple(e.column for e in spec.effects)
    if not np.array_equal(decomposition.dates, fit.dates) or set(
        decomposition.draws.contributions
    ) != set(columns):
        msg = "the decomposition is not of this fit: its periods or its effects are another's"
        raise ValueError(msg)
    names = [str(c) for c in panel.channels]
    channels = tuple(c for c in columns if c in names)
    spend = {c: float(panel.spend[g, :, names.index(c)].sum()) for c in channels}
    dark = [c for c in channels if not spend[c] > 0.0]
    if dark:
        msg = f"channels {dark} spend nothing over the fit's periods, so they have no return"
        raise ValueError(msg)
    _level(level)
    calendar = (*(("trend",) if spec.trend else ()), *(("season",) if spec.season else ()))
    origins = _origins(origins, ("kpi", *columns, *calendar))
    estimates = _tested(estimates, channels)
    window = decomposition.window(fit.dates[0], fit.dates[-1]).contributions
    return Factsheet(
        model=_dlm_model(spec),
        kpi=_kpi(panel),
        cadence=panel.cadence.value,
        first=_date(fit.dates[0]),
        last=_date(fit.dates[-1]),
        geos=(str(geo),),
        channels=channels,
        sources=_dlm_sources(panel, geo, spec, fit.baselines, origins),
        priors=_dlm_priors(panel, geo, fit),
        tests=estimands(panel, estimates),
        holdout=holdout,
        returns=tuple(_range(c, "window", window[c] / spend[c], level) for c in channels),
        power=power,
        refresh=refresh,
        limitations=(
            *limitations,
            *_derived(channels, _calibrated(estimates), holdout, power),
            *_dlm_limits(fit, channels),
        ),
    )


def _dlm_model(spec: DlmSpec) -> str:
    trend = ""
    if spec.trend:
        trend = "a level, " if spec.trend.order == 1 else "a level and a slope, "
    effects = _count(len(spec.effects), "effect")
    season = ", a season" if spec.season else ""
    variance = (
        f", the variance discounted by {spec.variance_discount:g} a period"
        if spec.variance_discount < 1.0
        else ""
    )
    return (
        f"dynamic linear model (mixdyn.models.dlm): {trend}{effects} each with its own coefficient"
        f"{season}, in {spec.form} form, its variance learned{variance}"
    )


def _dlm_sources(
    panel: MediaPanel,
    geo: Geo,
    spec: DlmSpec,
    baselines: Mapping[str, float],
    origins: Mapping[str, str],
) -> tuple[Source, ...]:
    filled = int(panel.filled[panel.geos.index(geo)].sum())
    kpi = ["read on its own scale, a period with no KPI a step filtered without an update"]
    if filled:
        unread = _count(filled, panel.cadence.value)
        kpi.insert(0, f"{unread} with no row in the input read as unobserved")
    out = [Source("kpi", "kpi", origins.get("kpi", _NOT_STATED), tuple(kpi))]
    for e in spec.effects:
        steps, role = [], "control"
        if e.column in panel.events:
            role = "calendar"
            steps.append("1 in the periods one of its days falls in, 0 elsewhere")
        elif e.column in panel.controls:
            lever = panel.levers.get(e.column)
            if lever is None:
                steps.append("read as it is")
            else:
                level = _number(baselines[e.column])
                steps.append(
                    f"a {lever.kind.value} lever, read against its counterfactual level {level}"
                )
        else:
            role = "spend"
            allotment = panel.national.get(Channel(e.column))
            if allotment is not None:
                steps.append(f"bought nationally and split over the geos by {allotment.value}")
            steps += _carried(e.transform, panel.currency) if e.transform else ["read as it is"]
        if e.discount < 1.0:
            idle = ", held while its column is 0" if e.hold_when_idle else ""
            steps.append(f"its coefficient a random walk of discount {e.discount:g}{idle}")
        else:
            steps.append("its coefficient held still, a discount of 1")
        if e.sign is not None:
            side = "above" if e.sign == "positive" else "below"
            steps.append(f"its coefficient held at or {side} 0, the posterior truncated to it")
        out.append(Source(e.column, role, origins.get(e.column, _NOT_STATED), tuple(steps)))
    if spec.trend:
        walks = "a level that walks" if spec.trend.order == 1 else "a level and a slope that walk"
        line = f"{walks}, discount {spec.trend.discount:g}"
        out.append(Source("trend", "calendar", origins.get("trend", _ENGINE_MADE), (line,)))
    if spec.season:
        s = spec.season
        line = f"harmonics 1 to {s.harmonics} of a season {s.period:g} periods long"
        steps = (line, f"discount {s.discount:g}")
        out.append(Source("season", "calendar", origins.get("season", _ENGINE_MADE), steps))
    return tuple(out)


def _carried(transform: Transform, currency: str) -> list[str]:
    retention, length = _number(transform.retention), _count(transform.length, "period")
    half = f"{_number(transform.half_saturation)} {currency}"
    shape = transform.shape
    curve = (
        f"saturated by a Hill of slope {_number(shape.slope)}, half its ceiling at {half}"
        if isinstance(shape, HillShape)
        else f"saturated by tanh, half its ceiling at {half}"
    )
    return [
        f"carried over by a normalised geometric kernel of retention {retention} over {length}",
        curve,
    ]


def _slope(curve: HillCurve, engine: str) -> Prior:
    """The Hill's slope prior as its curve states it."""
    default = (curve.slope_location, curve.slope_scale) == (0.0, 0.5)
    return Prior(
        "slope",
        f"LogNormal({curve.slope_location:g}, {curve.slope_scale:g}), each channel's Hill's",
        engine if default else "stated by the caller",
    )


def _dlm_priors(panel: MediaPanel, geo: Geo, fit: DlmFit) -> tuple[Prior, ...]:
    spec = fit.spec
    prior = fit.filtered.model.prior
    vague = vague_prior(panel, geo, spec)
    default = (
        np.array_equal(prior.mean, vague.mean)
        and np.array_equal(prior.covariance, vague.covariance)
        and prior.scale == vague.scale
        and prior.dof == vague.dof
    )
    engine = "the engine's default (vague_prior): "
    ten = "its scale ten times the KPI's standard deviation"
    named = []
    if spec.trend:
        named.append(("level", f"{engine}the KPI's mean over its observed periods, {ten}"))
        if spec.trend.order == 2:
            named.append(("slope", f"{engine}0, {ten} over the periods"))
    named += [
        (f"coefficient[{e.column}]", f"{engine}0, {ten} over its column's root mean square")
        for e in spec.effects
    ]
    out = [
        Prior(name, _state(prior, i), why if default else "stated by the caller")
        for i, (name, why) in enumerate(named)
    ]
    season = range(len(named), prior.mean.size)
    if season:
        why = f"{engine}0, {ten}" if default else "stated by the caller"
        first = _state(prior, season[0])
        if all(_state(prior, i) == first for i in season):
            out.append(Prior("season", f"each of its {len(season)} coordinates: {first}", why))
        else:
            out += [Prior(f"season[{k + 1}]", _state(prior, i), why) for k, i in enumerate(season)]
    variance = (
        f"known, {_number(prior.scale)}"
        if math.isinf(prior.dof)
        else f"{_number(prior.scale)} from {_freedom(prior.dof)}"
    )
    why = (
        f"{engine}the KPI's variance over its observed periods"
        if default
        else "stated by the caller"
    )
    out.append(Prior("variance", f"the KPI's noise variance: {variance}", why))
    stated = "the spec's, stated by the caller or chosen by a search"
    blocks = [("level", spec.trend.discount)] if spec.trend else []
    blocks += [(e.column, e.discount) for e in spec.effects]
    blocks += [("season", spec.season.discount)] if spec.season else []
    out += [Prior(f"evolution[{name}]", _evolution(discount), stated) for name, discount in blocks]
    if spec.variance_discount < 1.0:
        line = f"discount {spec.variance_discount:g} of the variance's degrees of freedom a period"
        out.append(Prior("evolution[variance]", line, stated))
    return tuple(out)


def _state(prior: DlmPrior, i: int) -> str:
    mean, scale = _number(prior.mean[i]), _number(math.sqrt(prior.covariance[i, i]))
    if math.isinf(prior.dof):
        return f"Normal about {mean}, its SD {scale}"
    return f"Student t of {_freedom(prior.dof)} about {mean}, its scale {scale}"


def _freedom(dof: float) -> str:
    return f"{_number(dof)} degree{'' if dof == 1.0 else 's'} of freedom"


def _evolution(discount: float) -> str:
    if discount == 1.0:
        return "discount 1: held still"
    return (
        f"discount {discount:g}: its variance grows by a factor {_number(1.0 / discount)} a period"
    )


def _dlm_limits(fit: DlmFit, channels: Sequence[str]) -> list[str]:
    spec, out = fit.spec, []
    if any(e.transform is not None for e in spec.effects):
        out.append(
            "each channel's carryover and curve are held at its transform's values: the return "
            "ranges leave out their uncertainty"
        )
    walking = [e.column for e in spec.effects if e.column in channels and e.discount < 1.0]
    if walking:
        out.append(
            f"the coefficients of {', '.join(walking)} walk: each return is over the fit's whole "
            "periods, and the last period's coefficient, which a plan reads, may differ from it"
        )
    if fit.signed is not None and not fit.signed.mixed:
        rhat, ess = _number(fit.signed.rhat), _number(fit.signed.ess)
        out.append(
            f"the draws held to the signs have not mixed (R-hat {rhat}, bulk effective sample "
            f"size {ess}): the returns are not to be read"
        )
    return out


def ridge_factsheet(
    panel: MediaPanel,
    geo: Geo,
    fit: RidgeFit,
    *,
    origins: Mapping[str, str] | None = None,
    holdout: Holdout | Unfilled = _NO_HOLDOUT,
    power: PowerSummary | Unfilled = _NO_POWER_AUDIT,
    refresh: Refresh | Unfilled = _NO_REFRESH,
    limitations: Sequence[str] = (),
) -> Factsheet:
    """The factsheet of the ridge model (`mixdyn.models.ridge_mmm`), `fit`, fitted to `geo` of
    `panel`.

    The calendar is the fit's window. The sources hold each column with its adstock, its Hill and
    its half-saturation point, a lever's level, and each coefficient's bounds. The ridge states no
    prior: its penalty, and a geo test's anchor where the fit read tests, are listed in its place.
    The fit is a point, with no draws: each channel's return is its contribution over the window
    over its spend over it, a `ReturnRange` whose low and high are that point, at a level of 0, and
    the limitations say so. The ridge reads geo tests (`chc.lift.LiftTest`), which `estimands` does
    not state, so `tests` holds none and the limitations name the channels they read. `origins`,
    `holdout`, `power`, `refresh` and `limitations` are read as `factsheet` reads them.

    Raises:
        ValueError: on a geo the panel does not have, a fit of another series than the geo's KPI
            over dates of the panel's calendar, and an origin of a column the model does not read.
    """
    if geo not in panel.geos:
        msg = f"the panel has no geo {geo!r}: it has {list(panel.geos)}"
        raise ValueError(msg)
    inside = np.isin(panel.dates, fit.dates)
    if not (
        inside.sum() == fit.dates.size
        and np.array_equal(fit.y, panel.kpi[panel.geos.index(geo), inside], equal_nan=True)
    ):
        msg = f"the fit is not of {geo!r}'s KPI over dates of the panel's calendar"
        raise ValueError(msg)
    spec = fit.spec
    channels = tuple(m.channel for m in spec.media)
    controls = tuple(c.column for c in spec.controls)
    calendar = (*(("trend",) if spec.trend else ()), *(("season",) if spec.season else ()))
    origins = _origins(origins, ("kpi", *channels, *controls, *calendar))
    tested = dict(fit.calibration.estimates) if fit.calibration else {}
    limits = [
        "the ridge fit is a point: each return range holds its point estimate alone, at a level of"
        " 0, with no interval"
    ]
    bound = [c for c in channels if fit.bound[c]]
    if bound:
        limits.append(
            f"{', '.join(bound)}: the coefficient sits on its bound at 0, so the fit reads no "
            "effect of the channel"
        )
    if tested:
        limits.append(
            f"the fit read geo tests of {', '.join(tested)} (chc.lift.LiftTest), which the "
            "factsheet's tests do not state"
        )
    return Factsheet(
        model=_ridge_model(spec),
        kpi=_kpi(panel),
        cadence=panel.cadence.value,
        first=_date(fit.dates[0]),
        last=_date(fit.dates[-1]),
        geos=(str(geo),),
        channels=channels,
        sources=_ridge_sources(panel, geo, fit, origins),
        priors=_ridge_priors(fit),
        tests=(),
        holdout=holdout,
        returns=tuple(
            _range(c, "window", float(np.sum(fit.contributions[c])) / fit.spend[c], 0.0)
            for c in channels
        ),
        power=power,
        refresh=refresh,
        limitations=(
            *limitations,
            *_derived(
                channels, tested, holdout, power, rests="the observational data and the penalty"
            ),
            *limits,
        ),
    )


def _ridge_model(spec: RidgeSpec) -> str:
    calendar = "".join([", a trend" if spec.trend else "", ", a season" if spec.season else ""])
    media, controls = _count(len(spec.media), "channel"), _count(len(spec.controls), "control")
    return (
        f"ridge media-mix model (mixdyn.models.ridge_mmm): {media} on Hill curves, "
        f"{controls}{calendar}, by bounded ridge regression at a penalty of {spec.penalty:g}"
    )


def _ridge_sources(
    panel: MediaPanel, geo: Geo, fit: RidgeFit, origins: Mapping[str, str]
) -> tuple[Source, ...]:
    spec = fit.spec
    inside = np.isin(panel.dates, fit.dates)
    filled = int(panel.filled[panel.geos.index(geo), inside].sum())
    kpi = ["read as it is, the window's observed periods fitted"]
    if filled:
        unread = _count(filled, panel.cadence.value)
        kpi.insert(0, f"{unread} with no row in the input read as unobserved")
    out = [Source("kpi", "kpi", origins.get("kpi", _NOT_STATED), tuple(kpi))]
    for m in spec.media:
        steps = []
        allotment = panel.national.get(Channel(m.channel))
        if allotment is not None:
            steps.append(f"bought nationally and split over the geos by {allotment.value}")
        match m.adstock:
            case Geometric(theta):
                steps.append(f"carried over by a geometric adstock of theta {theta:g}")
            case WeibullCdf(shape, scale):
                line = f"carried over by a Weibull adstock of shape {shape:g} and scale {scale:g}"
                steps.append(f"{line} periods")
        half = f"{_number(fit.half_saturation[m.channel])} {panel.currency}"
        steps += [
            f"saturated by a Hill of alpha {m.hill.alpha:g}, half its ceiling at {half}, gamma "
            f"{m.hill.gamma:g} of the most of the window's adstocked values",
            "its coefficient held at or above 0",
        ]
        out.append(Source(m.channel, "spend", origins.get(m.channel, _NOT_STATED), tuple(steps)))
    for c in spec.controls:
        role, steps = "control", []
        if c.column in panel.events:
            role = "calendar"
            steps.append("1 in the periods one of its days falls in, 0 elsewhere")
        elif c.column in panel.levers:
            level = _number(panel.counterfactual(c.column))
            kind = panel.levers[c.column].kind.value
            steps.append(
                f"a {kind} lever, read as its departure from its counterfactual level {level}"
            )
        steps += _held(c)
        if not (c.lower == 0.0 or c.upper == 0.0) and math.isfinite(c.lower + c.upper):
            steps.append(f"its coefficient held from {c.lower:g} to {c.upper:g}")
        out.append(
            Source(
                c.column,
                role,
                origins.get(c.column, _NOT_STATED),
                tuple(steps or ["read as it is"]),
            )
        )
    return (*out, *_calendar(spec.trend, spec.season, origins))


def _ridge_priors(fit: RidgeFit) -> tuple[Prior, ...]:
    spec = fit.spec
    out = [
        Prior(
            "coefficients",
            f"no prior: a ridge penalty of {spec.penalty:g} on each standardised coefficient, the "
            "intercept's free, glmnet's objective",
            "the spec's, stated by the caller or chosen by search_ridge",
        )
    ]
    if fit.calibration is not None:
        weight = _number(fit.calibration.weight)
        out += [
            Prior(
                f"anchor[{channel}]",
                f"the coefficient pulled to {_number(estimate)}, a test period weighing {weight} "
                "of a KPI period",
                "the fit's geo tests, read alone",
            )
            for channel, estimate in fit.calibration.estimates.items()
        ]
    return tuple(out)


def reconcile(
    dates: npt.ArrayLike, kpi: npt.ArrayLike, parts: Mapping[str, npt.ArrayLike]
) -> Reconciliation:
    """The table that sums `parts`, each part's contribution to the KPI at each of `dates`, to a
    total in the parts' order, beside the `kpi` observed and its residual, the KPI less the total.
    A `BayesFit`'s over its fitted periods: `reconcile(fit.dates[fit.fitted],
    fit.y[fit.fitted], {name: c[fit.fitted] for name, c in fit.contributions.items()})`, whose
    total is the fit's `estimated`.

    Raises:
        ValueError: on no parts, a part, the KPI or the dates of another length than the dates,
            and a value that is not finite.
    """
    days = np.asarray(dates, dtype="datetime64[D]")
    y = np.asarray(kpi, dtype=float)
    columns = [np.asarray(c, dtype=float) for c in parts.values()]
    if not columns or any(a.shape != (days.size,) for a in (days, y, *columns)):
        msg = (
            f"a reconciliation needs parts, each a series as long as the {days.size} dates, as is "
            f"the KPI"
        )
        raise ValueError(msg)
    if np.isnat(days).any() or not all(np.all(np.isfinite(a)) for a in (y, *columns)):
        msg = "a reconciliation reads dates and finite values at every period"
        raise ValueError(msg)
    total = columns[0]
    for column in columns[1:]:
        total = total + column
    return Reconciliation(
        parts=tuple(str(name) for name in parts),
        periods=tuple(
            Period(
                date=_date(days[t]),
                kpi=float(y[t]),
                parts=tuple(float(c[t]) for c in columns),
                total=float(total[t]),
                residual=float(y[t] - total[t]),
            )
            for t in range(days.size)
        ),
    )


_THRESHOLDS = ("max_rhat", "min_ppp", "max_negative_baseline", "max_width", "tail")


def decision(
    panel: MediaPanel,
    geo: Geo,
    plan: PosteriorPlan,
    *,
    title: str,
    card: HealthCard | None = None,
    thresholds: Mapping[str, float] | None = None,
    reconciliation: Reconciliation | None = None,
    choice: ExperimentChoice | None = None,
    events: Sequence[Event] = (),
    compute: Compute | None = None,
) -> Decision:
    """The decision document of `plan` (`plan_posterior`) for `geo` of `panel`, named `title`.

    It holds the plan and the channels it moves on their prior alone, `plan.unidentified`; the
    fit's health `card` read against `thresholds`, keyed by `HealthCard.flags`' keywords; a
    `reconciliation` (`reconcile`); the next set of lift tests by its value, from `choice`
    (`choose_tests`); the `events` of the decision's progress; and the `compute` record. What is
    not given is `Unfilled`, with its reason.

    Raises:
        ValueError: on a geo the panel does not have, and a threshold `HealthCard.flags` does not
            read.
    """
    if geo not in panel.geos:
        msg = f"the panel has no geo {geo!r}: it has {list(panel.geos)}"
        raise ValueError(msg)
    thresholds = dict(thresholds or {})
    unread = sorted(set(thresholds) - set(_THRESHOLDS))
    if unread:
        msg = f"a health card reads the thresholds {list(_THRESHOLDS)}, not {unread}"
        raise ValueError(msg)
    return Decision(
        title=title,
        geo=str(geo),
        currency=panel.currency,
        periods=plan.periods,
        budget=plan.budget,
        gain=plan.gain,
        current_gain=plan.current_gain,
        spend=tuple(
            Spend(str(c), float(now), float(then))
            for c, now, then in zip(plan.channels, plan.current, plan.spend, strict=True)
        ),
        health=Unfilled("no health card was read") if card is None else _health(card, thresholds),
        reconciliation=(
            Unfilled("no contributions were reconciled")
            if reconciliation is None
            else reconciliation
        ),
        next_test=Unfilled("no set of lift tests was valued") if choice is None else _next(choice),
        events=tuple(events),
        compute=Unfilled("no compute was recorded") if compute is None else compute,
        unidentified=tuple(str(c) for c in plan.unidentified),
    )


def _health(card: HealthCard, thresholds: Mapping[str, float]) -> Health:
    return Health(
        max_rhat=card.max_rhat,
        ppp=card.ppp,
        ppp_se=card.ppp_se,
        negative_baseline=card.negative_baseline,
        shift=dict(card.shift),
        width=dict(card.width),
        consistency=dict(card.consistency),
        thresholds={name: float(value) for name, value in thresholds.items()},
        flags=card.flags(**thresholds),
    )


def _next(choice: ExperimentChoice) -> NextTest | Unfilled:
    best = choice.best
    if best is None:
        evpi, error = _number(choice.evpi), _number(choice.evpi_error)
        return Unfilled(
            f"no set of lift tests is affordable, so the plan is made now; perfect information "
            f"would be worth {evpi} ± {error} in the KPI's units"
        )
    return NextTest(
        verdict=choice.verdict,
        tests=tuple(c.label or str(c.channel) for c in best.candidates),
        channels=tuple(str(c.channel) for c in best.candidates),
        reduction=best.reduction,
        error=best.error,
        cost=best.cost,
        net_value=best.net_value,
        predicted_net_value=best.predicted_net_value,
        evpi=choice.evpi,
        evpi_error=choice.evpi_error,
        draws=choice.draws,
    )


def render(document: Factsheet | Decision) -> str:
    """`document` as plain text: its kind on the first line, then each field on a line of its own
    under its name, a nested one indented below it, an item of a tuple after a dash, an `Unfilled`
    one as `unfilled:` and its reason, and a reconciliation's periods as a table, a period a row.
    Every number is written at four significant digits, without an exponent."""
    return "\n".join([type(document).__name__.lower(), *_lines(document, 0)]) + "\n"


def _lines(value: DataclassInstance, depth: int) -> list[str]:
    return [
        line
        for f in dataclasses.fields(value)
        for line in _field(f.name, getattr(value, f.name), depth)
    ]


def _field(name: str, value: Any, depth: int) -> list[str]:
    pad = "  " * depth
    if isinstance(value, Unfilled):
        return [f"{pad}{name}: unfilled: {value.reason}"]
    if isinstance(value, Reconciliation):
        head = ["date", "kpi", *value.parts, "total", "residual"]
        rows = [
            [p.date.isoformat(), *(_number(x) for x in (p.kpi, *p.parts, p.total, p.residual))]
            for p in value.periods
        ]
        return [
            f"{pad}{name}:",
            *_field("parts", value.parts, depth + 1),
            f"{pad}  periods:",
            *(f"{pad}    | {' | '.join(row)} |" for row in [head, ["---"] * len(head), *rows]),
        ]
    if dataclasses.is_dataclass(value):
        return [f"{pad}{name}:", *_lines(value, depth + 1)]
    if isinstance(value, tuple) and value:
        out = [f"{pad}{name}:"]
        for item in value:
            if dataclasses.is_dataclass(item):
                lines = _lines(item, depth + 2)
                out += [f"{pad}  - {lines[0].lstrip()}", *lines[1:]]
            else:
                out.append(f"{pad}  - {_scalar(item)}")
        return out
    if isinstance(value, Mapping) and value:
        return [f"{pad}{name}:", *(f"{pad}  {k}: {_scalar(v)}" for k, v in value.items())]
    if isinstance(value, tuple | Mapping):
        return [f"{pad}{name}: none"]
    return [f"{pad}{name}: {_scalar(value)}"]


def _scalar(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "pass" if value else "fail"
    if isinstance(value, float):
        return _number(value)
    if isinstance(value, dt.date):
        return value.isoformat()
    return str(value)


def _number(x: float) -> str:
    """`x` at four significant digits, without an exponent, `-0` as `0`."""
    if not math.isfinite(x):
        return str(x)
    return np.format_float_positional(
        x + 0.0, precision=4, unique=False, fractional=False, trim="-"
    )


def sensitivity_line(
    ridge: RidgeSensitivity | None = None,
    bayes: MsmSensitivity | None = None,
    *,
    control: str | None = None,
    gamma: float = 1.5,
) -> str:
    """ADR 0061's line on a channel's sensitivity to an unmeasured confounder: the ridge fit's
    robustness values and, where `control` is named, its coefficient under a confounder as strong
    as that control; the Bayesian fit's `Gamma*` and its bound at `gamma`. Each part is written
    where its result is given. `Gamma*` is written to two decimals, so that one just above 1 does
    not read as 1, the level at which the interval holds 0 with no confounder.

    Raises:
        ValueError: on neither result, results of two channels, a `control` the ridge result has
            no benchmark for, or a `gamma` the Bayesian result was not read at.
    """
    if ridge is None and bayes is None:
        msg = "a sensitivity line needs a ridge result, a Bayesian one or both"
        raise ValueError(msg)
    if ridge is not None and bayes is not None and ridge.channel != bayes.channel:
        msg = f"the results are of two channels, {ridge.channel!r} and {bayes.channel!r}"
        raise ValueError(msg)
    parts = ["*Sensitivity to an unmeasured confounder.*"]
    if ridge is not None:
        held = ridge.robustness_interval
        interval = (
            f"{held:.1%} to bring its {ridge.level:.0%} interval to 0"
            if held > 0.0
            else f"its {ridge.level:.0%} interval holds 0 already"
        )
        line = (
            f"{ridge.channel}'s return would reach 0 only under a confounder that explains"
            f" {ridge.robustness:.1%} of both its residual spend and the KPI's ({interval})"
        )
        if control is not None:
            moved = ridge.benchmark(control)
            line += (
                f"; one as strong as {control} would move its coefficient from"
                f" {ridge.estimate:.4g} to {moved.estimate:.4g}"
            )
        parts.append(line + ".")
    if bayes is not None:
        at = np.flatnonzero(bayes.gammas == gamma)
        if at.size == 0:
            msg = f"the Bayesian result was read at Gamma {bayes.gammas.tolist()}, not {gamma}"
            raise ValueError(msg)
        k = int(at[0])
        stays = (
            "holds 0 with no confounder"
            if bayes.gamma_star == 1.0
            else f"stays above 0 up to a confounding level of Gamma = {bayes.gamma_star:.2f}"
        )
        parts.append(
            f"For the Bayesian fit: the return's {bayes.level:.0%} interval {stays}; at Gamma ="
            f" {gamma:g} it is [{bayes.lower[k]:.3g}, {bayes.upper[k]:.3g}]."
        )
    parts.append("These are bounds under a stated strength, not corrections.")
    return " ".join(parts)


def dumps(value: Any, schema: Schema) -> str:
    document = {
        "schema": schema.name,
        "version": schema.version,
        "data": _encode(value, schema.data, "data"),
    }
    return json.dumps(document, allow_nan=False, ensure_ascii=False)


def loads(text: str, schema: Schema) -> Any:
    try:
        document = json.loads(text)
    except (ValueError, RecursionError) as error:
        msg = f"not JSON: {error}"
        raise SchemaError(msg) from error
    if not isinstance(document, dict) or set(document) != {"schema", "version", "data"}:
        msg = "not a document: expected exactly the keys schema, version and data"
        raise SchemaError(msg)
    if document["schema"] != schema.name:
        msg = f"the document is a {reprlib.repr(document['schema'])}, not a {schema.name!r}"
        raise SchemaError(msg)
    if document["version"] != schema.version:
        msg = (
            f"the document is {schema.name} version {reprlib.repr(document['version'])}; this "
            f"engine reads version {schema.version}"
        )
        raise SchemaError(msg)
    return _decode(document["data"], schema.data, "data")


_ARRAY_DTYPES = {"float64", "int64", "bool", "datetime64[D]"}


def _resolve(hint: Any) -> Any:
    """Unwrap `type X = ...` aliases, subscripted or not, and NewTypes down to what they stand for.

    NumPy's `NDArray[T]` is such an alias, of `ndarray[shape, dtype[T]]`.
    """
    while True:
        origin = get_origin(hint)
        if isinstance(hint, TypeAliasType):
            hint = hint.__value__
        elif isinstance(origin, TypeAliasType):
            hint = origin.__value__[get_args(hint)]
        elif hasattr(hint, "__supertype__"):
            hint = hint.__supertype__
        else:
            return hint


def _is_union(hint: Any) -> bool:
    return get_origin(hint) in (Union, types.UnionType)


def _is_array(hint: Any) -> bool:
    return hint is np.ndarray or get_origin(hint) is np.ndarray


def _is_number(value: object) -> bool:
    return isinstance(value, int | float | np.integer | np.floating) and not isinstance(
        value, bool | np.bool_
    )


def _encode(value: Any, hint: Any, path: str) -> Any:
    hint = _resolve(hint)
    if _is_union(hint):
        options = [_resolve(a) for a in get_args(hint)]
        if value is None and type(None) in options:
            return None
        options = [a for a in options if a is not type(None)]
        if len(options) == 1:
            return _encode(value, options[0], path)
        for option in options:
            if dataclasses.is_dataclass(option) and type(value) is option:
                return {"kind": option.__name__, **_encode_fields(value, option, path)}
        names = [getattr(a, "__name__", repr(a)) for a in options]
        msg = f"{path}: a {type(value).__name__} is none of {names}"
        raise SchemaError(msg)
    if dataclasses.is_dataclass(hint):
        if type(value) is not hint:
            msg = f"{path}: expected a {hint.__name__}, got a {type(value).__name__}"
            raise SchemaError(msg)
        return _encode_fields(value, hint, path)
    if isinstance(hint, type) and issubclass(hint, Enum):
        if not isinstance(value, hint):
            msg = f"{path}: expected a {hint.__name__}, got {value!r}"
            raise SchemaError(msg)
        return value.value
    if hint is dt.date:
        if type(value) is not dt.date:
            msg = f"{path}: expected a date, got {value!r}"
            raise SchemaError(msg)
        return value.isoformat()
    if hint is float:
        if not _is_number(value):
            msg = f"{path}: expected a number, got {value!r}"
            raise SchemaError(msg)
        number = float(value)
        if math.isinf(number):
            msg = f"{path}: {number} has no JSON form"
            raise SchemaError(msg)
        return None if math.isnan(number) else number
    if hint is int:
        if not isinstance(value, int | np.integer) or isinstance(value, bool | np.bool_):
            msg = f"{path}: expected an int, got {value!r}"
            raise SchemaError(msg)
        return int(value)
    if hint is bool:
        if not isinstance(value, bool | np.bool_):
            msg = f"{path}: expected a bool, got {value!r}"
            raise SchemaError(msg)
        return bool(value)
    if hint is str:
        if not isinstance(value, str):
            msg = f"{path}: expected a str, got {value!r}"
            raise SchemaError(msg)
        return value
    if get_origin(hint) is tuple:
        item, rest = _tuple_items(hint, path)
        if rest is not None and len(value) != len(rest):
            msg = f"{path}: expected {len(rest)} items, got {len(value)}"
            raise SchemaError(msg)
        hints = rest if rest is not None else [item] * len(value)
        return [
            _encode(v, h, f"{path}[{i}]") for i, (v, h) in enumerate(zip(value, hints, strict=True))
        ]
    if get_origin(hint) in (dict, Mapping):
        key_hint, value_hint = (_resolve(a) for a in get_args(hint))
        if key_hint is not str:
            msg = f"{path}: only string keys have a JSON form, not {key_hint!r}"
            raise SchemaError(msg)
        return {str(k): _encode(v, value_hint, f"{path}[{k!r}]") for k, v in value.items()}
    if _is_array(hint):
        return _encode_array(np.asarray(value), path)
    msg = f"{path}: no JSON form for {hint!r}"
    raise SchemaError(msg)


def _encode_fields(value: Any, cls: type[DataclassInstance], path: str) -> dict[str, Any]:
    hints = get_type_hints(cls)
    return {
        f.name: _encode(getattr(value, f.name), hints[f.name], f"{path}.{f.name}")
        for f in dataclasses.fields(cls)
        if f.init
    }


def _encode_array(array: np.ndarray, path: str) -> dict[str, Any]:
    dtype = str(array.dtype)
    if dtype not in _ARRAY_DTYPES:
        msg = (
            f"{path}: arrays of {dtype} have no JSON form; expected one of {sorted(_ARRAY_DTYPES)}"
        )
        raise SchemaError(msg)
    flat = array.reshape(-1)
    if dtype == "float64":
        if np.isinf(flat).any():
            msg = f"{path}: an infinite value has no JSON form"
            raise SchemaError(msg)
        data = [None if math.isnan(x) else x for x in flat.tolist()]
    elif dtype == "datetime64[D]":
        data = [None if np.isnat(x) else str(x) for x in flat]
    else:
        data = flat.tolist()
    return {"dtype": dtype, "shape": list(array.shape), "data": data}


def _tuple_items(hint: Any, path: str) -> tuple[Any, list[Any] | None]:
    """The item type of `tuple[X, ...]`, or the item types of a fixed-length tuple."""
    args = get_args(hint)
    if len(args) == 2 and args[1] is Ellipsis:
        return args[0], None
    if not args:
        msg = f"{path}: a bare tuple declares no item type"
        raise SchemaError(msg)
    return None, list(args)


def _decode(data: Any, hint: Any, path: str) -> Any:
    hint = _resolve(hint)
    if _is_union(hint):
        options = [_resolve(a) for a in get_args(hint)]
        if data is None and type(None) in options:
            return None
        options = [a for a in options if a is not type(None)]
        if len(options) == 1:
            return _decode(data, options[0], path)
        kind = data.get("kind") if isinstance(data, dict) else None
        for option in options:
            if dataclasses.is_dataclass(option) and option.__name__ == kind:
                return _decode_fields({k: v for k, v in data.items() if k != "kind"}, option, path)
        names = [getattr(a, "__name__", repr(a)) for a in options]
        msg = f"{path}: kind {reprlib.repr(kind)} is none of {names}"
        raise SchemaError(msg)
    if dataclasses.is_dataclass(hint):
        if not isinstance(data, dict):
            msg = f"{path}: expected an object for {hint.__name__}, got {reprlib.repr(data)}"
            raise SchemaError(msg)
        return _decode_fields(data, hint, path)
    if isinstance(hint, type) and issubclass(hint, Enum):
        try:
            return hint(data)
        except ValueError:
            msg = f"{path}: {reprlib.repr(data)} is not a {hint.__name__}"
            raise SchemaError(msg) from None
    if hint is dt.date:
        try:
            return dt.date.fromisoformat(data)
        except (TypeError, ValueError):
            msg = f"{path}: expected an ISO date, got {reprlib.repr(data)}"
            raise SchemaError(msg) from None
    if hint is float:
        if data is None:
            return math.nan
        if type(data) not in (int, float):
            msg = f"{path}: expected a number, got {reprlib.repr(data)}"
            raise SchemaError(msg)
        if not _floats(data):
            msg = f"{path}: {reprlib.repr(data)} is past a float's finite range"
            raise SchemaError(msg)
        return float(data)
    if hint in (bool, int, str):
        if type(data) is not hint:
            msg = f"{path}: expected {hint.__name__}, got {reprlib.repr(data)}"
            raise SchemaError(msg)
        return data
    if get_origin(hint) is tuple:
        if not isinstance(data, list):
            msg = f"{path}: expected a list, got {reprlib.repr(data)}"
            raise SchemaError(msg)
        item, rest = _tuple_items(hint, path)
        if rest is not None and len(data) != len(rest):
            msg = f"{path}: expected {len(rest)} items, got {len(data)}"
            raise SchemaError(msg)
        hints = rest if rest is not None else [item] * len(data)
        return tuple(
            _decode(v, h, f"{path}[{i}]") for i, (v, h) in enumerate(zip(data, hints, strict=True))
        )
    if get_origin(hint) in (dict, Mapping):
        if not isinstance(data, dict):
            msg = f"{path}: expected an object, got {reprlib.repr(data)}"
            raise SchemaError(msg)
        _, value_hint = get_args(hint)
        return {k: _decode(v, value_hint, f"{path}[{k!r}]") for k, v in data.items()}
    if _is_array(hint):
        return _decode_array(data, path)
    msg = f"{path}: no JSON form for {hint!r}"
    raise SchemaError(msg)


def _decode_fields(data: dict[str, Any], cls: type[DataclassInstance], path: str) -> Any:
    hints = get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls) if f.init}
    if set(data) != names:
        msg = f"{path}: {cls.__name__} has the fields {sorted(names)}, the document {sorted(data)}"
        raise SchemaError(msg)
    values = {name: _decode(data[name], hints[name], f"{path}.{name}") for name in names}
    try:
        return cls(**values)
    except (TypeError, ValueError) as error:
        msg = f"{path}: {cls.__name__} refuses these values: {error}"
        raise SchemaError(msg) from error


def _floats(number: float) -> bool:
    """Whether `number` reads as a float that is not infinite: a document has no form for an
    infinity, and no float holds an integer past a float's range. NaN reads, as null does."""
    try:
        return not math.isinf(number)
    except OverflowError:
        return False


def _decode_array(data: Any, path: str) -> np.ndarray:
    if not isinstance(data, dict) or set(data) != {"dtype", "shape", "data"}:
        msg = f"{path}: expected an array object with dtype, shape and data"
        raise SchemaError(msg)
    dtype, shape, flat = data["dtype"], data["shape"], data["data"]
    if type(dtype) is not str or dtype not in _ARRAY_DTYPES:
        msg = f"{path}: unknown array dtype {reprlib.repr(dtype)}"
        raise SchemaError(msg)
    if not isinstance(shape, list) or not all(type(n) is int and n >= 0 for n in shape):
        msg = f"{path}: bad shape {reprlib.repr(shape)}"
        raise SchemaError(msg)
    if not isinstance(flat, list) or len(flat) != math.prod(shape):
        msg = f"{path}: shape {shape} needs {math.prod(shape)} values"
        raise SchemaError(msg)
    fits = {
        "float64": lambda x: x is None or (type(x) in (int, float) and _floats(x)),
        "int64": lambda x: type(x) is int and -(2**63) <= x < 2**63,
        "bool": lambda x: type(x) is bool,
        "datetime64[D]": lambda x: x is None or type(x) is str,
    }[dtype]
    misfits = [x for x in flat if not fits(x)]
    if misfits:
        msg = f"{path}: {reprlib.repr(misfits[0])} is not a {dtype} value"
        raise SchemaError(msg)
    if dtype == "float64":
        values = [math.nan if x is None else x for x in flat]
    elif dtype == "datetime64[D]":
        values = ["NaT" if x is None else x for x in flat]
    else:
        values = flat
    try:
        return np.array(values, dtype=dtype).reshape(shape)
    except (TypeError, ValueError) as error:
        msg = f"{path}: values do not fit {dtype}: {error}"
        raise SchemaError(msg) from error
