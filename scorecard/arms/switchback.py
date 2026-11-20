"""A switchback test of the national channel read beside a rung's geo tests, and the static arm's
plan of the market fitted on both (ADR 0078).

Each world of the bench's geo families (8, 9 and 10) carries a switchback test of its national
channel, TV, bought nationally and allotted to the geos by population: a channel no geo test can
cut without cutting the national buy. The bench's version-4 export holds it
(`causaldyn_bench.scorecard.observe.SwitchbackTest`): the history's last 12 blocks of the
kernel's length, 8 weeks, half of them on in an order the bench draws, TV off in every geo in the
others, and the universe it ran in, the world's base and the test's spend with noise from a
stream apart from the ladder's. The arm reads it as `mixdyn.experiments.SwitchbackDesign` with
the export's blocks and the first half of a block's weeks left out after a switch, a washout of
`block // 2`. CHOICE: a block as long as the kernel lets a block on reach its steady state, and a
washout of half of it leaves out the weeks the last block's carryover moves most; 12 blocks give
924 arrangements to the randomization reading, whose least p-value is then 1/924.

`mixdyn.experiments.read_switchback` reads it by randomization on the KPI summed over the geos: the
lift the weeks on gained over TV off, in the steady state, and its error. The estimate is the whole
market's, of share 1, and the static arm's fit of the market (`scorecard.arms.bayes`) reads it
beside the rung's geo tests, each in its geos' share; the harness runs the arm at each rung on its
own, the same switchback at every rung. Its one plan, `switchback`, is the static arm's `stacked`
on that fit. Its regret less the static arm's on the market at the same rung is what the
switchback adds to the rung's tests. Its flags hold the reading: the lift, its error and p-value,
whether its interval lies above 0, and the carryover check; its settings the design and the
reading's interval.

The harness scores the reading on the world's truth beside the bench's score of the plan
(`scored`): its lift a week on, the difference of the kept weeks' mean KPI on and off, against
the bench's lift a week on as a reader of that washout reads it (`Truth.switchback_steady`), the
test's whole lift beside it (`Truth.switchback`). The bench states the gap between the two: the
difference in means reads the steady truth less the lift the kept weeks off still carry, 0.956 to
0.996 of it on its 35 pilot worlds.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from causaldyn_bench.scorecard.observe import UNIVERSES

from mixdyn.experiments import SwitchbackDesign, read_switchback
from scorecard.arms import bayes
from scorecard.arms import panel as reading

if TYPE_CHECKING:
    from causaldyn_bench.scorecard.family import Truth
    from causaldyn_bench.scorecard.observe import Observation
    from causaldyn_bench.scorecard.scoring import ArmScore

    from scorecard.arms.base import Planned

VARIANT = "switchback"
STREAM = 7
"""The child of the export's seed the randomization reading draws its arrangements from, where
they are too many to enumerate, apart from the static arm's fits (0), forecasts (1, 2) and health
card (3)."""
SCORE = "switchback score"
"""The key the reading's score on the truth is kept under in the arm's score's record."""


@dataclass(frozen=True, slots=True, eq=False)
class Fitted:
    """The static arm's fit of the market with the switchback read, and what a record keeps."""

    fitted: bayes.Fitted
    flags: dict[str, float | int | bool]
    settings: dict[str, Any]


class Switchback:
    """The arm; see the module."""

    variants = (VARIANT,)
    tests = True
    cells = False

    def fit(self, observation: Observation, *, smoke: bool) -> Fitted:
        if not isinstance(observation, reading.Market) or not observation.geo.national:
            msg = "the switchback arm tests a geo export's national channel, and plans its market"
            raise ValueError(msg)
        geo = observation.geo
        test = geo.switchback
        if test is None:
            msg = (
                f"the switchback arm reads the test a geo export holds at the bench's version"
                f" {UNIVERSES} alone, not {geo.version}: export it again"
            )
            raise ValueError(msg)
        channel, block, blocks = test.channel, test.block, len(test.on)
        window = np.arange(test.first - 1, test.first - 1 + blocks * block)
        dates = reading.panel(observation).dates[window]
        child = np.random.SeedSequence(observation.seed).spawn(STREAM + 1)[STREAM]
        seed = int(child.generate_state(1)[0])
        design = SwitchbackDesign(dates, block, block // 2, test.on, sum(test.on) / blocks, seed)
        kpi = np.asarray(test.sales, dtype=float).T
        spend = np.asarray(test.spend, dtype=float).transpose(1, 0, 2)
        c = geo.channels.index(channel)
        estimate, read = read_switchback(kpi[:, window], spend[:, window, c], design)
        tests = dict(reading.estimates(observation))
        tests[channel] = (*tests.get(channel, ()), estimate)
        sampler = bayes.SMOKE if smoke else bayes.SAMPLER
        fits = bayes.family_fits(observation, sampler, estimates=tests)
        fitted = bayes.fitted(observation, fits, sampler)
        flags: dict[str, float | int | bool] = {
            "switchback lift": read.lift,
            "switchback se": read.se,
            "switchback p": read.p_value,
            "switchback up": read.verdict == "up",
            "switchback carryover": read.carryover,
        }
        settings = {
            "switchback": {
                "channel": channel,
                "weeks": [int(window[0]) + 1, int(window[-1]) + 1],
                "block": block,
                "washout": design.washout,
                "on": list(design.on),
                "seed": seed,
                "method": read.method,
                "arrangements": read.arrangements,
                "lower": read.lower,
                "upper": read.upper,
            },
        }
        return Fitted(fitted, flags, settings)

    def plan(self, fit: Fitted, variant: str, observation: Observation) -> Planned:
        planned = bayes.Bayes().plan(fit.fitted, bayes.RESPONDS, observation)
        return dataclasses.replace(
            planned,
            flags=dict(planned.flags) | fit.flags,
            settings=dict(planned.settings) | fit.settings,
        )


def scored(score: ArmScore, truth: Truth, observation: Observation) -> ArmScore:
    """`score`, the bench's of the arm's plan on the world `observation` was exported from, with
    the switchback's reading scored on the world's `truth` beside it, under `SCORE`: its lift a
    week on, the reading's lift over the weeks on, `block` times the blocks on; the bench's steady
    truth a week on, `Truth.switchback_steady`; their difference relative to it; whether the
    reading's interval, a week on, holds it; and the test's whole lift, `Truth.switchback`. The
    reading is scored only on the test the export holds: a record of another design, or one of a
    world with no switchback truth, says so in place of the score. A failed fit's score is as it
    is."""
    if score.failure is not None:
        return score
    held = score.record["settings"]["switchback"]
    test = observation.switchback
    steady = truth.switchback_steady
    if test is None or steady is None or "lower" not in held or list(held["on"]) != list(test.on):
        why = "the record read another switchback test than the export holds, or the world has none"
        return dataclasses.replace(score, record=score.record | {SCORE: {"unscored": why}})
    weeks = held["block"] * sum(held["on"])
    week = score.record["flags"]["switchback lift"] / weeks
    read = {
        "a week": week,
        "steady": steady,
        "error": (week - steady) / steady,
        "held": held["lower"] / weeks <= steady <= held["upper"] / weeks,
        "total": truth.switchback,
    }
    return dataclasses.replace(score, record=score.record | {SCORE: read})
