"""Each geo test of a rung read again by another of the engine's readers, and the static arm's
plan of the market fitted on those readings (ADR 0078).

A geo export's rows are the bench's readings of its tests by a difference in differences
(`causaldyn_bench.scorecard.ladder.geo_rung`). A reading arm reads each test in the universe it ran
in, its channel's KPI in every geo as the bench's version-4 export holds it
(`causaldyn_bench.scorecard.observe.Universes`), by one of `mixdyn.experiments`' readers, its
tested geos treated and every other geo a control, the test's dark weeks its change in spend and
its 16 weeks of readout, the dark weeks and the cooldown, read with it:

* `read tbr`: `read_tbr`, Kerman, Wang and Vaver's time-based regression, its error from placebo
  windows in time;
* `read sc`: `read_synthetic_control`, the demeaned synthetic control of the tested geos' mean,
  its error from the gaps' innovations over the 8 blank weeks before the test;
* `read sdid`: `read_sdid`, synthetic difference in differences over the readout, its error the
  jackknife's. CHOICE: the jackknife, since a placebo draws as many controls as there are tested
  geos and the ladder tests half of the geos, so no placebo is left.

Each reading's lift and error over the readout, per dark week as the bench writes a row, take the
row's place, and the static arm's fit and `stacked` plan read the market with them, each test in
its geos' share (`scorecard.arms.panel.market`); the harness runs each arm at each rung on its own.
A test the reader refuses keeps the bench's reading, and the record counts them (`reads`): the
time-based regression's placebo windows need `TBR_WARMUP` weeks and a readout before each, 30
weeks before the test, which the ladder's first test, from week 20, does not have; the synthetic
control needs `DONORS` control geos and SDID's jackknife `JACKKNIFE` treated geos and as many
controls, which the tests of a world of 3 geos (family 10's `geos-3`) do not have. The record's
flags count
the rows read by the reader and those kept, and its settings hold each row's reading beside the
bench's under `reader rows`, beside the static arm's own settings.

The arm's regret less the static arm's on the same market at the same rung is what reading the
same tests by that reader rather than the bench's difference in differences moves the plan by.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from causaldyn_bench.lift_calibration import COOLDOWN, TEST
from causaldyn_bench.scorecard.observe import UNIVERSES

from mixdyn import Channel
from mixdyn.experiments import GeoTest, read_sdid, read_synthetic_control, read_tbr
from scorecard.arms import bayes
from scorecard.arms import panel as reading

if TYPE_CHECKING:
    from causaldyn_bench.scorecard.observe import Observation

    from mixdyn.calibration import LiftEstimate
    from mixdyn.data import Geo, MediaPanel
    from scorecard.arms.base import Planned

TBR, SC, SDID = "read tbr", "read sc", "read sdid"
TBR_WARMUP = 13
"""`read_tbr`'s default warm-up before each placebo window."""
DONORS = 3
"""The fewest control geos `read_synthetic_control` reads a test against."""
JACKKNIFE = 2
"""The fewest treated geos, and control geos, `read_sdid`'s jackknife reads."""
READOUT = TEST + COOLDOWN


@dataclass(frozen=True, slots=True, eq=False)
class Fitted:
    """The static arm's fit of the market on the reader's rows, and what a record keeps of the
    readings."""

    fitted: bayes.Fitted
    flags: dict[str, float | int | bool]
    settings: dict[str, Any]


def reread(observation: Observation, variant: str) -> tuple[Observation, list[dict[str, Any]]]:
    """`observation`, a geo export of version 4, with each lift row's reading the reader's of
    `variant` where it reads the test, and each row's reading, the reader's and the bench's, the
    export read as version 2 (`scorecard.arms.panel.second`); see the module.

    Raises:
        ValueError: on an export that holds no universes, one of a version before 4.
    """
    lift = observation.lift
    universes = observation.universes
    if universes is None:
        msg = (
            f"a reading arm reads each test in the universe it ran in, which a geo export holds at"
            f" the bench's version {UNIVERSES} alone, not {observation.version}: export it again"
        )
        raise ValueError(msg)
    groups = lift.geos or ()
    named = reading.geos(observation)
    held = reading.estimates(observation)
    panel = reading.panel(observation)
    delta_y = np.asarray(lift.delta_y, dtype=float).copy()
    sigma = np.asarray(lift.sigma, dtype=float).copy()
    seen: dict[str, int] = {}
    kept = []
    for i, name in enumerate(lift.channel):
        estimate = held[name][seen.get(name, 0)]
        seen[name] = seen.get(name, 0) + 1
        first = lift.start[i] - 1
        row = {
            "channel": name,
            "start": int(lift.start[i]),
            "bench": [float(delta_y[i]), float(sigma[i])],
        }
        if not reads(variant, first, len(groups[i]), len(named) - len(groups[i])):
            row["reader"] = None
            kept.append(row)
            continue
        # the readers read the KPI alone; each test's spend is its design's
        kpi = np.ascontiguousarray(universes.sales[universes.channel.index(name)].T)
        universe = dataclasses.replace(panel, kpi=kpi)
        read = _read(universe, variant, Channel(name), estimate, groups[i], named, first)
        delta_y[i], sigma[i] = read.lift / TEST, read.se / TEST
        row["reader"] = [float(delta_y[i]), float(sigma[i])]
        kept.append(row)
    rows = dataclasses.replace(lift, delta_y=delta_y, sigma=sigma, gaps=None)
    return dataclasses.replace(reading.second(observation), lift=rows), kept


def reads(variant: str, first: int, treated: int, control: int) -> bool:
    """Whether the reader of `variant` reads a test from the 0-based week `first` of `treated`
    geos against `control` others; see the module."""
    if variant == TBR:
        return first >= TBR_WARMUP + READOUT + 1
    if variant == SC:
        return control >= DONORS
    return min(treated, control) >= JACKKNIFE


def _read(
    panel: MediaPanel,
    variant: str,
    channel: Channel,
    estimate: LiftEstimate,
    group: tuple[int, ...],
    named: tuple[Geo, ...],
    first: int,
) -> LiftEstimate:
    """The reader's estimate of one test on its universe's `panel`."""
    treated = tuple(named[g] for g in group)
    control = tuple(g for g in named if g not in treated)
    dates = panel.dates
    if variant == SDID:
        read, _ = read_sdid(
            panel,
            treated,
            dates[first],
            dates[first + READOUT - 1],
            spend=estimate.spend,
            planned=estimate.planned,
            variance="jackknife",
        )
        return read
    design = GeoTest(
        channel,
        treated,
        control,
        dates[first],
        dates[first + TEST - 1],
        estimate.spend,
        estimate.planned,
        READOUT,
    )
    read, _ = (read_tbr if variant == TBR else read_synthetic_control)(panel, design)
    return read


class Reader:
    """A reading arm of one reader, `variant` its name; see the module."""

    tests = True
    cells = False

    def __init__(self, variant: str) -> None:
        if variant not in (TBR, SC, SDID):
            msg = f"the readers are {TBR!r}, {SC!r} and {SDID!r}, not {variant!r}"
            raise ValueError(msg)
        self.variants = (variant,)

    def fit(self, observation: Observation, *, smoke: bool) -> Fitted:
        if not isinstance(observation, reading.Market):
            msg = (
                "a reading arm reads a geo export's tests in their universes, and plans its market"
            )
            raise ValueError(msg)
        (variant,) = self.variants
        geo, rows = reread(observation.geo, variant)
        market = reading.market(geo)
        sampler = bayes.SMOKE if smoke else bayes.SAMPLER
        fitted = bayes.fitted(market, bayes.family_fits(market, sampler), sampler)
        read = sum(row["reader"] is not None for row in rows)
        flags: dict[str, float | int | bool] = {"rows read": read, "rows kept": len(rows) - read}
        settings = {"reader": variant, "reader rows": rows}
        return Fitted(fitted, flags, settings)

    def plan(self, fit: Fitted, variant: str, observation: Observation) -> Planned:
        planned = bayes.Bayes().plan(fit.fitted, bayes.RESPONDS, observation)
        return dataclasses.replace(
            planned,
            flags=dict(planned.flags) | fit.flags,
            settings=dict(planned.settings) | fit.settings,
        )
