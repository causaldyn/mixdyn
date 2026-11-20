"""Three of the engine's decision tools as an arm: the static arm's posterior read by the flighted
planner, by scenarios compared at its draws, and by the plan table (ADR 0078).

The fit is the static arm's (`scorecard.arms.bayes`): its two fits of the export, every lift row
read, and their stacked posterior, so the harness runs the arm at each rung on its own. Each variant
plans one weekly spend a channel within the export's box at its budget, the history carried in,
and the bench scores it as it scores every arm's (`causaldyn_bench.scorecard.scoring`). On a geo
export the harness hands the arm the export's market, as it hands the static arm.

* `flighting`: `mixdyn.decisions.plan_flighting` over the quarter's weeks on the posterior of the
  family of the larger stacking weight, the Hill curve's on a tie, as many of its draws as the
  stacked posterior holds, each parameter at its weighted median. It plans a schedule, a spend in
  each week, and the best rate a week beside it, and recommends the schedule only where its gain
  over the rate holds across the draws. CHOICE: the planner takes one curve's median or one draw,
  and the two families' curves have no median together; the leading family's median is the
  planner's own reading of the posterior the stacking leans on, where one draw would be one
  sample of it. The record's plan is the rate, which the bench's scorer of a world reads; the
  schedule, the recommendation, the family and the schedule's edge over the rate, its weighted
  mean and central 90 % interval over the family's draws, are kept beside it. The harness scores
  the schedule on the world's truth beside the bench's score (`scored`), by the bench's scorer of
  a schedule (`causaldyn_bench.scorecard.schedule.score_schedule`): where the schedule is
  recommended, the variant's regret is the schedule's, else the rate's. Where the bench refuses
  the schedule, its cells not concave or not stating each week's reach, the regret is the rate's
  and the score says why.
* `scenarios`: `mixdyn.decisions.compare_scenarios` over four plans held for the quarter, the
  status quo (the baseline), the bench's equal split (`scorecard.arms.panel.equal_split`), the
  static arm's `stacked` and its `cvar 0.1`, the quarter's controls the export's. Each family's fit
  is read at its share of the static arm's readings (`scorecard.arms.bayes.shares`), the noise
  from a child of the export's seed; the scenarios' differences from the status quo, over the
  quarter and the kernel's reach after it, are pooled over the two families' draws. The plan is
  the scenario of the largest mean difference, the first on a tie, and the claim its difference's
  mean and central 90 % interval. An export that gives controls but not the quarter's has no
  scenario to read, and the variant fails.
* `plan table`: `mixdyn.decisions.plan_table` of the static arm's `stacked` plan against the status
  quo. The plan is `stacked`'s; the claim is the table's total change in the KPI, its posterior
  mean and central 90 % interval, which `plan_breakdown` reads over the quarter and the carryover
  after it, and the record keeps the table's rows.

`flighting` claims as the static arm claims, the mean and central 90 % interval of the plan's gain
over the status quo over its readings (`scorecard.arms.panel.gains`). Each record carries the
static arm's flags and settings; none carries a forecast or a response, which the static arm's
`stacked` record carries.

A smoke run samples as the static arm's smoke run does.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from causaldyn_bench.scorecard.schedule import oracle, score_schedule

from mixdyn import NATIONAL, Channel
from mixdyn.decisions import (
    Range,
    Scenario,
    compare_scenarios,
    plan_flighting,
    plan_table,
    readings,
)
from scorecard.arms import bayes
from scorecard.arms import panel as reading
from scorecard.arms.base import Planned

if TYPE_CHECKING:
    from collections.abc import Mapping

    from causaldyn_bench.scorecard.family import Truth
    from causaldyn_bench.scorecard.observe import Observation
    from causaldyn_bench.scorecard.scoring import ArmScore

    from mixdyn.models.bayes_mmm import BayesFit

FLIGHTING, SCENARIOS, TABLE = "flighting", "scenarios", "plan table"
BASELINE = "status quo"
STREAM = 6
"""The child of the export's seed the scenarios' noise is drawn from, apart from the static arm's
fits (0), forecasts (1, 2) and health card (3)."""
SCORE = "schedule score"
"""The key the schedule's score on the truth is kept under in `flighting`'s score's record."""
ABSENT = {
    "forecast": f"recorded under the static arm's {bayes.RESPONDS!r}",
    "response": f"recorded under the static arm's {bayes.RESPONDS!r}",
}


@dataclass(frozen=True, slots=True, eq=False)
class Fitted:
    """The static arm's fit, the two fits it stacked, and its `stacked` and `cvar 0.1` plans."""

    fitted: bayes.Fitted
    fits: Mapping[str, BayesFit]
    stacked: np.ndarray
    cvar: np.ndarray


class Decisions:
    """The arm; see the module."""

    variants = (FLIGHTING, SCENARIOS, TABLE)
    tests = True
    cells = False

    def fit(self, observation: Observation, *, smoke: bool) -> Fitted:
        sampler = bayes.SMOKE if smoke else bayes.SAMPLER
        fits = bayes.family_fits(observation, sampler)
        fitted = bayes.fitted(observation, fits, sampler)
        static = bayes.Bayes()
        stacked, cvar = (
            np.asarray(static.plan(fitted, variant, observation).weekly, dtype=float)
            for variant in (bayes.RESPONDS, "cvar 0.1")
        )
        return Fitted(fitted, fits, stacked, cvar)

    def plan(self, fit: Fitted, variant: str, observation: Observation) -> Planned:
        if variant == FLIGHTING:
            weekly, claim, flags, settings, absent = _flighting(fit, observation)
        elif variant == SCENARIOS:
            weekly, claim, flags, settings, absent = _scenarios(fit, observation)
        else:
            weekly, claim, flags, settings, absent = _table(fit, observation)
        gain, low, high = claim
        return Planned(
            weekly=weekly,
            gain=gain,
            gain_interval=(low, high),
            flags=fit.fitted.flags | flags,
            settings=fit.fitted.settings | settings,
            absent=fit.fitted.absent | ABSENT | absent,
            ess=fit.fitted.ess,
        )


type _Read = tuple[
    np.ndarray,
    tuple[float, float, float],
    dict[str, float | int | bool],
    dict[str, Any],
    dict[str, str],
]


def limits(observation: Observation) -> dict[Channel, Range]:
    """Each channel's box a week, by the engine's name for it."""
    return {
        Channel(c): Range(float(low), float(high))
        for c, low, high in zip(
            observation.channels, observation.lower, observation.upper, strict=True
        )
    }


def _flighting(fit: Fitted, observation: Observation) -> _Read:
    posterior = fit.fitted.posterior
    leading = int(np.argmax(posterior.stacking))
    family = list(fit.fits)[leading]
    one = readings(fit.fits[family], draws=int(np.bincount(posterior.fit)[leading]))
    flight = plan_flighting(
        reading.panel(observation),
        NATIONAL,
        one,
        observation.budget,
        observation.planned,
        limits(observation),
        level=bayes.GAIN_LEVEL,
    )
    weekly = np.asarray(flight.continuous, dtype=float)
    gain = reading.gains(fit.fitted.readings, observation, weekly)
    low, high = np.quantile(gain, [(1.0 - bayes.GAIN_LEVEL) / 2.0, (1.0 + bayes.GAIN_LEVEL) / 2.0])
    edge_low, edge_high = map(float, flight.edge_interval)
    flags: dict[str, float | int | bool] = {
        "flighted": flight.recommendation == "flighted",
        "edge": float(one.weights @ flight.edge),
        "edge low": edge_low,
        "edge high": edge_high,
        "schedule worth": flight.worth,
        "schedule bound": flight.bound,
    }
    settings = {
        "planner": "plan_flighting",
        "family": family,
        "reading": flight.reading,
        "recommendation": flight.recommendation,
        "schedule": flight.schedule.tolist(),
        "search": str(flight.stopped),
        "limit": None if flight.limit is None else str(flight.limit),
    }
    return weekly, (float(np.mean(gain)), float(low), float(high)), flags, settings, {}


def _scenarios(fit: Fitted, observation: Observation) -> _Read:
    if observation.controls and observation.future_controls is None:
        msg = "the scenarios read the quarter's controls, and the export gives none"
        raise ValueError(msg)
    plans = {
        BASELINE: np.asarray(observation.status_quo, dtype=float),
        "equal split": reading.equal_split(observation),
        bayes.RESPONDS: fit.stacked,
        "cvar 0.1": fit.cvar,
    }
    future = observation.future_controls or {}
    controls = {name: np.asarray(future[name], dtype=float) for name in observation.controls}
    scenarios = [
        Scenario(
            name,
            {
                c: np.full(observation.planned, float(w[j]))
                for j, c in enumerate(observation.channels)
            },
            controls,
        )
        for name, w in plans.items()
    ]
    weights = dict(zip(bayes.FAMILIES, map(float, fit.fitted.posterior.stacking), strict=True))
    counts = bayes.shares(weights["hill"], int(fit.fitted.settings["readings"]))
    seed = int(
        np.random.SeedSequence(observation.seed).spawn(STREAM + 1)[STREAM].generate_state(1)[0]
    )
    pooled = np.concatenate(
        [
            compare_scenarios(
                fit.fits[family],
                scenarios,
                baseline=BASELINE,
                horizon=observation.planned,
                draws=count,
                seed=seed,
                level=bayes.GAIN_LEVEL,
            ).differences
            for family, count in counts.items()
            if count
        ],
        axis=1,
    )
    means = pooled.mean(axis=1)
    names = list(plans)
    chosen = int(np.argmax(means))
    low, high = np.quantile(
        pooled[chosen], [(1.0 - bayes.GAIN_LEVEL) / 2.0, (1.0 + bayes.GAIN_LEVEL) / 2.0]
    )
    flags: dict[str, float | int | bool] = {
        f"{name} difference": float(m) for name, m in zip(names, means, strict=True)
    }
    settings = {
        "planner": "compare_scenarios",
        "scenarios": names,
        "baseline": BASELINE,
        "chosen": names[chosen],
        "draws": counts,
        "horizon": observation.planned,
    }
    claim = (float(means[chosen]), float(low), float(high))
    return plans[names[chosen]], claim, flags, settings, {}


def _table(fit: Fitted, observation: Observation) -> _Read:
    table = plan_table(
        reading.panel(observation),
        NATIONAL,
        fit.fitted.posterior,
        fit.stacked,
        observation.planned,
        against=observation.status_quo,
        level=bayes.GAIN_LEVEL,
        limits=limits(observation),
    )
    flags: dict[str, float | int | bool] = {
        "factor": table.factor,
        "factor low": table.factor_low,
        "factor high": table.factor_high,
    }
    # strict JSON holds no NaN: a column that is not a number is null
    rows = [
        {k: None if isinstance(v, float) and math.isnan(v) else v for k, v in row.items()}
        for row in table.rows()
    ]
    settings = {"planner": "plan_table", "plan": bayes.RESPONDS, "rows": rows}
    claim = (float(table.total_kpi.mean), float(table.total_kpi.low), float(table.total_kpi.high))
    return fit.stacked, claim, flags, settings, {}


def scored(score: ArmScore, truth: Truth, observation: Observation) -> ArmScore:
    """`score`, the bench's of `flighting`'s rate on the world `observation` was exported from,
    with the schedule of its record scored on the world's `truth` beside it, under `SCORE`: the
    bench's score of the schedule (`causaldyn_bench.scorecard.schedule.ScheduleScore`), its regret
    against the best schedule, on the realised path too where there is one, the best schedule's
    edge over the best plan and the schedule's move into the box, and the rate's regret.

    Where the schedule is recommended, the score's regret is the schedule's, its realised regret
    and its move too: what the schedule returns short of the best plan, per euro of the budget, as
    every arm's regret is read, so it is the schedule's regret less the edge and may be below 0.
    Elsewhere the regret stays the rate's, which the record recommends. Where the bench refuses the
    schedule, the score says why and the rate's regret stands. A failed fit's score is as it is.
    The claimed gain, and so its error, stay the rate's.
    """
    if score.failure is not None:
        return score
    settings = score.record["settings"]
    try:
        schedule = score_schedule(truth, np.asarray(settings["schedule"], dtype=float))
    except ValueError as refused:
        return dataclasses.replace(score, record=score.record | {SCORE: {"refused": str(refused)}})
    read = dataclasses.asdict(schedule) | {"rate regret": score.regret}
    if settings["recommendation"] != "flighted":
        return dataclasses.replace(score, record=score.record | {SCORE: read})
    budget = truth.quarter.budget
    realised = None
    if schedule.realised is not None and truth.realised is not None and truth.hindsight is not None:
        best = oracle(truth.realised, truth.quarter).worth
        realised = schedule.realised - (best - truth.hindsight.worth) / budget
    return dataclasses.replace(
        score,
        regret=schedule.regret - schedule.edge,
        realised=realised,
        moved=schedule.moved,
        record=score.record | {SCORE: read | {"scored": "the schedule"}},
    )
