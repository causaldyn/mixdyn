"""What this engine's arms read of an export: its panel, its lift tests as estimates, the market a
geo export sums to, and what a plan or a claim over the quarter is read against.

A national export (`causaldyn_bench.scorecard.observe.Observation`) is read as one national geo's
weekly panel, its first week Monday 2023-01-02, the date the bench's arm scripts give it, its
controls by the names the export gives them.

Each lift row is one go-dark test of its channel, reduced by the bench as an analyst reads a test
for a steady-state likelihood (`causaldyn_bench.budget_regret.lift_rows`): the channel's spend
`x` over the dark weeks, which went to 0 in the tested universe, and the gap summed over the dark
weeks and the cooldown after them, over the dark weeks, `delta_y`, its standard error `sigma`. A
test is two universes at the market's scale, so the gap is the whole market's. It is read as the
engine reads a test's analysis (`mixdyn.calibration.LiftEstimate`): the tested universe's spend
over the history up to the readout's end, the history's own with the test's dark weeks at 0;
what it would have spent, the history's own; a readout of the dark weeks and the cooldown; the
lift `delta_y` times the dark weeks, and its standard error `sigma` times them. The history before
the readout is all of it, since a model's kernel may run that long. Each row's `x` is the
history's mean spend over its dark weeks, checked, so a row is read against the weeks it was
taken from or not at all. CHOICE: each test zeroes its own dark weeks alone, though a channel's
tests ran in one universe, as each row describes one test.

A geo export, one with a geo axis, is read as a geo panel (`panel`): each geo of the export a geo
of the panel, `geo-<g>` by its place on the axis, every geo under one root, `mixdyn.NATIONAL`, and
its population the export's share of the market. Each geo's own channels are spent as the export
spends them, and each channel bought nationally is recorded as `mixdyn.allot_national` records it:
its spend summed over the geos and split by the shares. The export's own split must pass the
panel's check of a national channel, each geo's spend its share of the period's to `1e-9` of it,
or the export is refused. A geo export's controls are the market's, one series a week, which no
rule reads geo by geo, so an export that has them is refused.

A geo test's row (`causaldyn_bench.scorecard.ladder.geo_rung`) is one go-dark of its channel in
the geos it names, every other geo of the ring, read by difference in differences against the
others: `x` is the named geos' spend summed, its mean over the dark weeks; `delta_y` the named
geos' KPI less the others' scaled to them by the four weeks before, summed over the dark weeks and
the cooldown, over the dark weeks; `sigma` its standard error, read in time from placebo
readings. It is read as the engine reads a geo test's analysis (`LiftEstimate` with `geos`), as a
national row is but on the named geos: their spend summed, with the dark weeks at 0, against
their history's own; the lift `delta_y` times the dark weeks, and its standard error `sigma`
times them; the named geos as `geos`, and a share of 1, as the group is the whole of its geos'
market. The model over geos spreads a group's spend over its geos' people at one spend per head
(ADR 0018): a geo's own channel spent at its own, and the reading is their mean.

A geo export of the bench's version 4 holds, beside version 2's arrays, the universe each of its
tests ran in and the world's switchback test (`causaldyn_bench.scorecard.observe.Universes` and
`SwitchbackTest`): the panels the readers and the switchback arm read (`scorecard.arms.readers`,
`scorecard.arms.switchback`), which they take from the export itself. Every other reading of it is
of its version-2 arrays (`second`): its market, the export without its tests, and what a tool's
script is handed.

A geo export's market (`market`) is the export read as one national export of the geos' sums, by
`mixdyn.aggregate` to the root, for an arm that plans the market: one weekly spend a channel, which
the bench allots to the cells, a per-geo channel's across the geos as the status quo splits it and
a national channel's to its one cell. A channel's status quo is its cells' summed, and its box the
spends whose allotment keeps each of its cells in the cell's box. A test of the market is the geo
test's estimate with its group a share of the market, the named geos' share of the population,
in place of its geos (`LiftEstimate.share`): the model of the market reads the group's spend and
lift over its share, as if the whole market had gone dark at the group's spend per head. A
channel bought nationally spends the same per head in every geo, so on it that reading is exact;
on a per-geo channel it reads the group at its mean spend per head, as the model over geos does.
The market's own sums read the test otherwise, the market's spend less the group's against the
market's: on a concave curve that is the market's last euros, where the test took the group's
first ones, and it reads the lift as a steeper curve than the market's.

A plan's or a reading's worth is what its cells return over the quarter and each kernel's length
after it, the history's carryover counted in, as `chc.allocation` reads a channel; a gain is a
plan's worth less the status quo's, both on the same reading. A national export's cells are its
channels; a geo export's are each geo's own channels, geo by geo, then each national channel once,
its spend summed over the geos (`cells`).

The equal split the bench scores beside every arm is not in the export: it is rebuilt from the
export's budget and box by the bench's own rule (`equal_split`). A geo export's market reads it
as the bench's equal split of the export's cells summed by channel: each channel spends the total
the bench's split gives it. The bench splits a per-geo channel over the geos by population; the
market's allotment splits it as the status quo does, so the two equal splits share their channel
totals, not their cells.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from causaldyn_bench.lift_calibration import COOLDOWN, TEST
from causaldyn_bench.mmm_decision import Quarter
from causaldyn_bench.scorecard.observe import UNIVERSES, VERSION, Observation
from causaldyn_bench.scorecard.truth import GeoQuarter

from mixdyn import NATIONAL, Cadence, Channel, Geo, MediaPanel, Revenue, aggregate, allot_national
from mixdyn.calibration import LiftEstimate

if TYPE_CHECKING:
    from collections.abc import Sequence

    from chc import response

FIRST_WEEK = np.datetime64("2023-01-02")
AGREE = 1e-12
"""How near a lift row's `x` must lie to its dark weeks' mean spend, relative to it."""
ROUNDING = 1e-12
"""How far past the market's box the cells' equal split may sum, relative to the bound: rounding."""


@dataclass(frozen=True, eq=False, kw_only=True)
class Market(Observation):
    """A geo export read as its market (`market`): a national export of the geos' sums, and `geo`,
    the export it was read from, whose tests `estimates` reads as tests in a share of the market."""

    geo: Observation


def second(observation: Observation) -> Observation:
    """`observation` as version 2 holds it: a version-4 export without its tests' universes and its
    switchback test, every other array the same; an export of another version as it is."""
    if observation.version != UNIVERSES:
        return observation
    return dataclasses.replace(observation, version=VERSION, universes=None, switchback=None)


def geos(observation: Observation) -> tuple[Geo, ...]:
    """The geos of `observation`'s panel: `NATIONAL` for a national export, and `geo-<g>` for
    each geo of a geo export's axis."""
    if observation.population is None:
        return (NATIONAL,)
    return tuple(Geo(f"geo-{g}") for g in range(observation.population.size))


def cells(observation: Observation) -> tuple[tuple[int | None, str], ...]:
    """Each cell of `observation`'s plan, the geo it spends in by its place on the geo axis and its
    channel: a national export's channels, each in no geo; a geo export's own channels, geo by geo
    in the channels' order, then each channel bought nationally, in no geo."""
    if observation.population is None:
        return tuple((None, c) for c in observation.channels)
    own = [c for c in observation.channels if c not in observation.national]
    every = [(g, c) for g in range(observation.population.size) for c in own]
    return (*every, *((None, c) for c in observation.channels if c in observation.national))


def cell_spend(observation: Observation) -> np.ndarray:
    """Each cell's spend over the history, `(weeks, cells)` in `cells`' order: a cell in no geo's,
    its channel's spend summed over the geos."""
    spend = np.asarray(observation.spend, dtype=float)
    if observation.population is None:
        return spend
    columns = [
        spend[:, :, observation.channels.index(c)].sum(axis=1)
        if g is None
        else spend[:, g, observation.channels.index(c)]
        for g, c in cells(observation)
    ]
    return np.column_stack(columns)


def panel(observation: Observation) -> MediaPanel:
    """`observation`'s history as a weekly panel: one national geo's, or a geo export's geos'.

    Raises:
        ValueError: on a geo export with controls, or whose national channel's split is not its
            population's (`MediaPanel`).
    """
    weeks = observation.sales.shape[0]
    dates = FIRST_WEEK + np.timedelta64(7, "D") * np.arange(weeks)
    if observation.population is None:
        names = tuple(observation.controls)
        values = [np.asarray(observation.controls[name], dtype=float) for name in names]
        return MediaPanel(
            dates=dates,
            cadence=Cadence.WEEK,
            geos=(NATIONAL,),
            channels=tuple(Channel(c) for c in observation.channels),
            controls=names,
            kpi=np.asarray(observation.sales, dtype=float)[None, :],
            kpi_unit=Revenue(),
            spend=np.asarray(observation.spend, dtype=float)[None, :, :],
            currency="EUR",
            control_values=(np.column_stack(values) if values else np.zeros((weeks, 0)))[None],
            filled=np.zeros((1, weeks), dtype=bool),
        )
    if observation.controls:
        msg = (
            f"a geo export's controls {list(observation.controls)} are the market's, one series a"
            " week, and no rule reads them geo by geo"
        )
        raise ValueError(msg)
    named = geos(observation)
    spend = np.asarray(observation.spend, dtype=float).transpose(1, 0, 2)
    own = [c for c in observation.channels if c not in observation.national]
    made = MediaPanel(
        dates=dates,
        cadence=Cadence.WEEK,
        geos=named,
        channels=tuple(Channel(c) for c in own),
        controls=(),
        kpi=np.asarray(observation.sales, dtype=float).T,
        kpi_unit=Revenue(),
        spend=spend[:, :, [observation.channels.index(c) for c in own]],
        currency="EUR",
        control_values=np.zeros((len(named), weeks, 0)),
        filled=np.zeros((len(named), weeks), dtype=bool),
        hierarchy=dict.fromkeys(named, NATIONAL),
        population=np.asarray(observation.population, dtype=float),
    )
    for c in observation.national:
        made = allot_national(made, c, spend[:, :, observation.channels.index(c)].sum(axis=0))
    # the export's own split, held to the panel's check of a channel bought nationally
    dataclasses.replace(
        made, spend=spend[:, :, [observation.channels.index(c) for c in made.channels]]
    )
    return made


def estimates(observation: Observation) -> dict[str, tuple[LiftEstimate, ...]]:
    """Each channel's lift rows as the engine reads a test's analysis, by channel, each ending at
    its readout's last week in `panel`'s calendar; a channel with no row is left out. A geo
    export's estimates name their geos; a market's are in a share of it.

    Raises:
        ValueError: on a row whose readout runs past the history, or whose `x` is not the
            history's mean spend over its dark weeks in its geos, or whose change is not the whole
            of it.
    """
    if isinstance(observation, Market):
        return _in_share(observation)
    lift, spend = observation.lift, np.asarray(observation.spend, dtype=float)
    weeks = spend.shape[0]
    if observation.population is None:
        spend, groups, named = spend[:, None, :], ((0,),) * len(lift.channel), None
    else:
        groups, named = lift.geos or (), geos(observation)
    out: dict[str, list[LiftEstimate]] = {}
    for i, name in enumerate(lift.channel):
        first = lift.start[i] - 1
        end = first + TEST + COOLDOWN
        if first < 0 or end > weeks:
            msg = f"{name}'s test from week {lift.start[i]} reads past the history's {weeks} weeks"
            raise ValueError(msg)
        group = list(groups[i])
        planned = spend[:end, group, observation.channels.index(name)].sum(axis=1)
        x = float(lift.x[i])
        dark = float(np.mean(planned[first : first + TEST]))
        if abs(x - dark) > AGREE * abs(dark) or float(lift.delta_x[i]) != -x:
            msg = (
                f"{name}'s test from week {lift.start[i]} reads x {x} and a change of"
                f" {float(lift.delta_x[i])}, where the history spent {dark} a dark week"
            )
            raise ValueError(msg)
        tested = planned.copy()
        tested[first : first + TEST] = 0.0
        out.setdefault(name, []).append(
            LiftEstimate(
                spend=tested,
                planned=planned,
                readout=TEST + COOLDOWN,
                lift=float(lift.delta_y[i]) * TEST,
                se=float(lift.sigma[i]) * TEST,
                geos=None if named is None else frozenset(named[g] for g in group),
                end=FIRST_WEEK + np.timedelta64(7, "D") * (end - 1),
            )
        )
    return {name: tuple(rows) for name, rows in out.items()}


def equal_split(observation: Observation) -> np.ndarray:
    """The equal split the bench scores beside every arm (`causaldyn_bench.scorecard.scoring`):
    each channel an equal share of the budget, moved into the box at the budget by one shift
    (`causaldyn_bench.mmm_decision.Quarter.equal_split`), made from the export's budget and box.
    On a geo export's market (`market`) it is the bench's equal split of the geo export's cells
    (`causaldyn_bench.scorecard.truth.GeoQuarter.equal_split`) summed by channel: the market's plan
    that spends each channel's total as the bench's does, which the market's allotment then splits
    over the geos as the status quo does, not by population.

    Raises:
        ValueError: on an export with a geo axis, whose equal split the bench makes over its
            cells; its market (`market`) is read instead.
    """
    if isinstance(observation, Market):
        geo = observation.geo
        channel = [geo.channels.index(c) for _, c in cells(geo)]
        share = np.asarray(geo.population, dtype=float) / float(np.sum(geo.population))
        quarter = GeoQuarter(
            budget=geo.budget,
            lower=geo.lower,
            upper=geo.upper,
            status_quo=geo.status_quo,
            history=cell_spend(geo),
            channel=tuple(channel),
            people=np.array([1.0 if g is None else share[g] for g, _ in cells(geo)]),
        )
        total = np.bincount(channel, weights=quarter.equal_split(), minlength=len(geo.channels))
        # a channel whose cells all sit at their bound sums to the market's bound only in exact
        # arithmetic: `market` divides each cell's bound by its share, this sums the cells
        lower, upper = np.asarray(observation.lower), np.asarray(observation.upper)
        slack = ROUNDING * np.maximum(np.abs(lower), np.abs(upper))
        if np.any(total < lower - slack) or np.any(total > upper + slack):
            msg = f"the cells' equal split {total} is outside the market's box [{lower}, {upper}]"
            raise ValueError(msg)
        return np.clip(total, lower, upper)
    if observation.population is not None:
        msg = "the equal split of an export with a geo axis is made over its cells; read its market"
        raise ValueError(msg)
    quarter = Quarter(
        budget=observation.budget,
        lower=observation.lower,
        upper=observation.upper,
        status_quo=observation.status_quo,
        history=observation.spend,
    )
    return quarter.equal_split()


def market(observation: Observation) -> Market:
    """`observation`, a geo export, read as its market; see the module.

    Raises:
        ValueError: on a national export; what `panel` refuses; a cell whose status quo is not
            above 0, which a channel's plan allots nothing; and a channel no one spend of which
            keeps each of its cells in its box.
    """
    if observation.population is None:
        msg = "a national export is its own market"
        raise ValueError(msg)
    whole = aggregate(panel(observation), (NATIONAL,))
    order = [whole.channels.index(Channel(c)) for c in observation.channels]
    channel = np.array([observation.channels.index(c) for _, c in cells(observation)])
    quo = np.asarray(observation.status_quo, dtype=float)
    if not np.all(quo > 0.0):
        msg = f"cells {np.flatnonzero(~(quo > 0.0)).tolist()} hold a status quo of 0 or less"
        raise ValueError(msg)
    count = len(observation.channels)
    totals = np.bincount(channel, weights=quo, minlength=count)
    split = quo / totals[channel]
    lower, upper = np.full(count, -np.inf), np.full(count, np.inf)
    np.maximum.at(lower, channel, np.asarray(observation.lower, dtype=float) / split)
    np.minimum.at(upper, channel, np.asarray(observation.upper, dtype=float) / split)
    if not np.all(lower <= upper):
        shut = [observation.channels[c] for c in np.flatnonzero(~(lower <= upper))]
        msg = f"no one spend of {shut} keeps each of its cells in its box"
        raise ValueError(msg)
    plain = second(observation)
    fields = {f.name: getattr(plain, f.name) for f in dataclasses.fields(plain)}
    fields |= {
        "sales": whole.kpi[0],
        "spend": whole.spend[0][:, order],
        "lower": lower,
        "upper": upper,
        "status_quo": totals,
        "lift": dataclasses.replace(observation.lift, geos=None),
        "population": None,
        "national": (),
    }
    return Market(**fields, geo=observation)


def _in_share(observation: Market) -> dict[str, tuple[LiftEstimate, ...]]:
    """The market's tests: each of its geo export's, its group the named geos' share of the
    population in place of its geos."""
    population = np.asarray(observation.geo.population, dtype=float)
    named = geos(observation.geo)
    return {
        name: tuple(
            dataclasses.replace(
                row,
                geos=None,
                share=math.fsum(population[named.index(g)] for g in row.geos or ())
                / float(population.sum()),
            )
            for row in rows
        )
        for name, rows in estimates(observation.geo).items()
    }


def worth(channel: response.Channel, history: np.ndarray, weekly: float, periods: int) -> float:
    """What `channel` returns over `periods` weeks at `weekly` a week and its kernel's length after
    them, carried over from `history`, the channel's spend before them."""
    tail = int(channel.kernel.weights().size) - 1
    series = np.concatenate([history, np.full(periods, weekly), np.zeros(tail)])
    return float(np.sum(np.asarray(channel(series))[history.size :]))


def gains(
    readings: Sequence[Sequence[response.Channel]], observation: Observation, weekly: np.ndarray
) -> np.ndarray:
    """Each reading's worth of the plan `weekly` less its worth of the status quo, `(readings,)`,
    each reading a cell a column in `cells`' order."""
    spent = cell_spend(observation)
    out = np.empty(len(readings))
    for r, reading in enumerate(readings):
        out[r] = sum(
            worth(c, spent[:, j], float(weekly[j]), observation.planned)
            - worth(c, spent[:, j], float(observation.status_quo[j]), observation.planned)
            for j, c in enumerate(reading)
        )
    return out
