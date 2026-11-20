"""The engine's static Bayesian model as an arm: a Hill's fit and a `tanh`'s of the export, and the
plans of their stacked posterior.

Each fit is `mixdyn.models.bayes_mmm.fit_bayes` on the export's panel (`scorecard.arms.panel`),
every lift row in its likelihood as an estimate: each channel's return, carryover and curve; a
promotion's coefficient held at or above 0 and every other control free; a trend of 3 knots and a
yearly season of 3 harmonics; normal noise; Meridian's default prior on each channel's return; four
chains of 1000 draws after 1000 of warm-up, from a child of the export's seed, whose own stream drew
the world. The two fits are stacked by the weights of their leave-one-out densities at every
observation their likelihoods read (`mixdyn.decisions.stacked`). This is the engine's static model
as Track M v2's harnesses scored it, and either fit failing fails every variant. On a geo export
the harness hands the arm the export's market (`scorecard.arms.panel.market`, ADR 0043): it fits
the geos' sums, reads each test in its geos' share of the market, and plans one weekly spend a
channel.

The variants:

* `stacked`: the plan for the stacked posterior's expected return,
  `mixdyn.decisions.plan_posterior` over `DRAWS` of its draws, on `GRID` weekly spends across
  each channel's box;
* `cvar 0.1`: the split with the most mean gain over the status quo in the worst tenth of
  `DRAWS` readings of the stacked posterior, each family's share its weight, by
  `chc.allocation.cvar_allocate`: the engine's plan where no test has been run, as its no-test
  pilot chose it;
* `cvar 0.1 vs equal`: the same over the bench's equal split (`scorecard.arms.panel.equal_split`),
  so the split does no worse than the equal split in the worst tenth of the readings (ADR 0044).

Each claims its plan's gain over the status quo, the mean and the central 90 % interval of the
gain over those readings (`scorecard.arms.panel.gains`). Its flags are each fit's sampler health
and leave-one-out reading, its health card's cells (`mixdyn.inference.diagnostics.health`, ADR
0060: the posterior predictive p-value of the KPI's total, the probability of a negative baseline,
each channel's return's shift, width and consistency against its prior, the card's draws from a
child of the export's seed apart from the fits' and the forecast's) and its R² in sample, and the
Hill's stacking weight. Its effective sample size is the least bulk one of the fits whose draws
the readings hold (`shares`).

`stacked`'s record carries the fit's forecast and its response (`scorecard.arms.mapped`), the
others' neither, one fit for the three. The forecast is the stacked posterior's predictive of the
quarter's weekly sales at the status quo, the controls at the quarter's
(`mixdyn.models.bayes_mmm.forecast_bayes`): a draw at each reading's draw of its family's
posterior, the trend run on from the window's last bend, the history's spend carried into the
quarter's, and the KPI drawn about the draw's mean by the model's noise, each family's from a
child of the export's seed apart from the fits'. The response is the readings, each channel at a
draw of its family's posterior, beside what the model
says each channel returned over the history in that draw, the draw's return times the history's
spend, the identity the model is parametrised by. The bench so holds the readings' kernels, curves
and coefficients to the model's own draws in total; the weeks of the draws it holds are the
readings' own, so their timing is checked against nothing but the total.

A smoke run samples 2 chains of 100 draws after 100 of warm-up.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from chc.allocation import cvar_allocate

from mixdyn import NATIONAL, Channel
from mixdyn.decisions import Range, plan_posterior, stacked
from mixdyn.inference import loo
from mixdyn.inference.diagnostics import health
from mixdyn.models.bayes_mmm import (
    BayesMedia,
    BayesSpec,
    HillCurve,
    TanhCurve,
    _thinned,
    fit_bayes,
    forecast_bayes,
)
from mixdyn.models.ridge_mmm import Control, FourierSeason, LinearTrend
from scorecard.arms import panel as reading
from scorecard.arms.base import Planned, r2
from scorecard.arms.mapped import held, mapped

if TYPE_CHECKING:
    from collections.abc import Mapping

    from causaldyn_bench.scorecard.observe import Observation
    from chc import response

    from mixdyn.calibration import LiftEstimate
    from mixdyn.data import MediaPanel
    from mixdyn.decisions import Readings
    from mixdyn.models.bayes_geo import BayesGeoFit
    from mixdyn.models.bayes_mmm import BayesFit, Saturation

FAMILIES = {"hill": HillCurve(), "tanh": TanhCurve()}
POSITIVE = ("promotion",)
"""Controls held at or above 0."""
KNOTS, HARMONICS, YEAR = 3, 3, 365.25 / 7.0
DRAWS = 400
"""The stacked posterior's draws a plan and a claim read, or every draw where there are fewer."""
GRID = 2048
LEVEL = 0.1
GAIN_LEVEL = 0.9
RESPONDS = "stacked"
"""The variant whose record carries the fit's forecast and response."""
EQUAL = "cvar 0.1 vs equal"
"""The variant whose worst tenth is read against the equal split, not the status quo."""


@dataclass(frozen=True, slots=True)
class Sampler:
    """Each fit's chains, and the draws of each after its warm-up."""

    chains: int
    warmup: int
    draws: int


SAMPLER = Sampler(chains=4, warmup=1000, draws=1000)
SMOKE = Sampler(chains=2, warmup=100, draws=100)


@dataclass(frozen=True, slots=True, eq=False)
class Fitted:
    """The two fits' stacked posterior as each variant reads it, and what a record keeps of it:
    `roi` is each reading's return per euro over the history, the model's own parameter,
    `(readings, channels)`, and `forecast` the quarter's sales drawn at each reading's draw."""

    posterior: Readings
    readings: tuple[tuple[response.Channel, ...], ...]
    roi: np.ndarray
    forecast: np.ndarray | None
    flags: dict[str, float | int | bool]
    settings: dict[str, Any]
    absent: dict[str, str]
    ess: float


def shares(hill: float, readings: int) -> dict[str, int]:
    """Each family's count of the stacked posterior's `readings`: the Hill's stacking weight
    `hill` times `readings`, rounded as Python rounds, and the `tanh` the rest."""
    count = round(hill * readings)
    return {"hill": count, "tanh": readings - count}


def least_ess(flags: Mapping[str, float | int | bool], counts: Mapping[str, int]) -> float:
    """The least bulk effective sample size, by `flags`, of the families whose draws the
    readings hold, `counts` of them each."""
    return float(np.min([flags[f"{f} min ess bulk"] for f, count in counts.items() if count]))


def family_fits(
    observation: Observation,
    sampler: Sampler,
    estimates: Mapping[str, tuple[LiftEstimate, ...]] | None = None,
    *,
    curves: Mapping[str, Saturation] = FAMILIES,
    kernel: int | None = None,
) -> dict[str, BayesFit]:
    """Each family's fit of the export, by `FAMILIES`' name, every lift row read, or `estimates`
    in their place where given: the fits the arm stacks, from a child of the export's seed. Or
    each of `curves`' fit, by its name, the kernels `kernel` periods long where given
    (`mixdyn.models.bayes_mmm.BayesSpec.kernel`): each fit is drawn from the same seed, so a curve
    fitted alone is the fit it is beside another."""
    data = reading.panel(observation)
    tests = reading.estimates(observation) if estimates is None else estimates
    spec = BayesSpec(
        media=tuple(BayesMedia(c) for c in observation.channels),
        controls=tuple(
            Control(c, lower=0.0) if c in POSITIVE else Control(c) for c in observation.controls
        ),
        trend=LinearTrend(KNOTS),
        season=FourierSeason(YEAR, HARMONICS),
        kernel=kernel,
    )
    seed = int(np.random.SeedSequence(observation.seed).spawn(1)[0].generate_state(1)[0])
    return {
        family: fit_bayes(
            data,
            NATIONAL,
            dataclasses.replace(spec, saturation=curve),
            seed=seed,
            estimates=tests or None,
            chains=sampler.chains,
            warmup=sampler.warmup,
            draws=sampler.draws,
        )
        for family, curve in curves.items()
    }


class Bayes:
    """The arm; see the module."""

    variants = ("stacked", "cvar 0.1", EQUAL)
    tests = True
    cells = False

    def fit(self, observation: Observation, *, smoke: bool) -> Fitted:
        sampler = SMOKE if smoke else SAMPLER
        return fitted(observation, family_fits(observation, sampler), sampler)

    def plan(self, fit: Fitted, variant: str, observation: Observation) -> Planned:
        if variant == "stacked":
            limits = {
                Channel(c): Range(float(low), float(high))
                for c, low, high in zip(
                    observation.channels, observation.lower, observation.upper, strict=True
                )
            }
            weekly = plan_posterior(
                reading.panel(observation),
                NATIONAL,
                fit.posterior,
                observation.budget,
                observation.planned,
                limits,
                grid=GRID,
            ).spend
            settings = {"planner": "plan_posterior", "grid": GRID}
        else:
            equal = variant == EQUAL
            weekly = cvar_allocate(
                fit.readings,
                observation.budget,
                observation.planned,
                level=LEVEL,
                against=reading.equal_split(observation) if equal else observation.status_quo,
                lower=observation.lower,
                upper=observation.upper,
                history=observation.spend,
            ).spend
            settings = {
                "planner": "cvar_allocate",
                "level": LEVEL,
                "against": "equal split" if equal else "status quo",
            }
        gain = reading.gains(fit.readings, observation, weekly)
        low, high = np.quantile(gain, [(1.0 - GAIN_LEVEL) / 2.0, (1.0 + GAIN_LEVEL) / 2.0])
        responds = variant == RESPONDS
        elsewhere = f"recorded under {RESPONDS!r}"
        return Planned(
            weekly=np.asarray(weekly, dtype=float),
            gain=float(np.mean(gain)),
            gain_interval=(float(low), float(high)),
            forecast=fit.forecast if responds else None,
            response=_response(fit, observation) if responds else None,
            flags=fit.flags,
            settings=fit.settings | settings,
            absent=fit.absent
            if responds
            else {"forecast": elsewhere} | fit.absent | {"response": elsewhere},
            ess=fit.ess,
        )


def fitted(
    observation: Observation,
    fits: dict[str, BayesFit],
    sampler: Sampler,
    *,
    given: np.ndarray | None = None,
) -> Fitted:
    """The arm's fit of `observation` from its two families' `fits`, `family_fits`' of it by
    `sampler`, in `FAMILIES`' order: their stacked posterior and what a record keeps of it; see
    the module. With `given`, one weight a family, the posterior weighs the families by them, not
    by stacking (`mixdyn.decisions.stacked`)."""
    data = reading.panel(observation)
    tests = reading.estimates(observation)
    draws = min(DRAWS, sampler.chains * sampler.draws)
    posterior = stacked(list(fits.values()), weights=given, draws=draws)
    weights = dict(zip(FAMILIES, map(float, posterior.stacking), strict=True))
    counts = shares(weights["hill"], draws)
    readings, roi = [], []
    for family, count in counts.items():
        if count:
            one = fits[family]
            readings += [
                tuple(d[c] for c in observation.channels) for d in one.channel_draws(count)
            ]
            # the draws `channel_draws` reads, its channels in the export's order as the spec's
            every = one.draws["roi"]
            roi.append(every.reshape(-1, every.shape[2])[_thinned(every.shape, count)])
    forecast, absent = None, {}
    if observation.future_controls is None and observation.controls:
        absent["forecast"] = "the export gives no controls over the quarter"
    else:
        forecast = _forecast(fits, counts, observation)
    flags: dict[str, float | int | bool] = {"hill weight": weights["hill"]}
    card = int(np.random.SeedSequence(observation.seed).spawn(4)[3].generate_state(1)[0])
    for family, fit in fits.items():
        flags |= {f"{family} {name}": value for name, value in _health(fit).items()}
        flags |= {f"{family} {name}": value for name, value in _card(fit, data, card).items()}
    ess = least_ess(flags, counts)
    settings = {
        "families": list(FAMILIES),
        "controls": {c: "at or above 0" if c in POSITIVE else "free" for c in data.controls},
        "trend_knots": KNOTS,
        "season_harmonics": HARMONICS,
        "lift_rows": sum(len(rows) for rows in tests.values()),
        "readings": draws,
        "gain_level": GAIN_LEVEL,
        "sampler": dataclasses.asdict(sampler),
    }
    return Fitted(
        posterior, tuple(readings), np.concatenate(roi), forecast, flags, settings, absent, ess
    )


def _forecast(
    fits: dict[str, BayesFit], counts: dict[str, int], observation: Observation
) -> np.ndarray:
    """The record's forecast; see the module. Its draws run as the readings do, each family's
    in turn."""
    future = observation.future_controls or {}
    plan = {
        name: np.full(observation.planned, float(observation.status_quo[j]))
        for j, name in enumerate(observation.channels)
    }
    plan |= {name: np.asarray(future[name], dtype=float) for name in observation.controls}
    children = np.random.SeedSequence(observation.seed).spawn(1 + len(counts))[1:]
    return np.concatenate(
        [
            forecast_bayes(
                fits[family],
                observation.planned,
                plan,
                count,
                int(child.generate_state(1)[0]),
            ).kpi
            for (family, count), child in zip(counts.items(), children, strict=True)
            if count
        ]
    )


def _response(fit: Fitted, observation: Observation) -> dict[str, Any]:
    """The record's response; see the module."""
    spend = np.asarray(observation.spend, dtype=float)
    weekly = [
        [np.asarray(channel(spend[:, j])) for j, channel in enumerate(fit.readings[i])]
        for i in held(len(fit.readings))
    ]
    return mapped(
        observation.channels,
        fit.readings,
        fit.roi * spend.sum(axis=0),
        np.asarray(weekly).transpose(0, 2, 1),
    )


def _card(fit: BayesFit, data: MediaPanel, seed: int) -> dict[str, float]:
    """A one-geo fit's health card's cells, its replicates and prior drawn from `seed`, and its R²
    in sample, of the posterior mean over the fitted periods."""
    card = health(fit, data, NATIONAL, seed=seed)
    cells = {
        "ppp": card.ppp,
        "negative baseline": card.negative_baseline,
        "r2": r2(fit.y[fit.fitted], fit.estimated[fit.fitted]),
    }
    for channel, shift in card.shift.items():
        cells |= {
            f"{channel} prior shift": shift,
            f"{channel} prior width": card.width[channel],
            f"{channel} prior consistency": card.consistency[channel],
        }
    return cells


def _health(fit: BayesFit | BayesGeoFit) -> dict[str, float | int]:
    """A fit's sampler health and its leave-one-out reading, of the model of one geo or over
    geos."""
    result = loo(fit.log_likelihood)
    return {
        "divergences": int(fit.divergent.sum()),
        "max rhat": max(float(np.max(d.rhat)) for d in fit.diagnostics.values()),
        "min ess bulk": min(float(np.min(d.ess_bulk)) for d in fit.diagnostics.values()),
        "min e-bfmi": float(np.min(fit.e_bfmi)),
        "elpd": float(result.elpd),
        "elpd se": float(result.se),
        "pareto k above": int(np.sum(result.pareto_k > result.k_threshold)),
    }
