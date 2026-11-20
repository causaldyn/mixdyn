"""The engine's calibration loop as an arm: the tests a rung adds valued before they are read, one
chosen, its reading read, and the plan made again from the posterior refitted with it (ADR 0077).

At rung `k` of the ladder each channel holds its latest `k` tests: the rung below's and one more,
run before them (`causaldyn_bench.scorecard.ladder`). The arm reads every test but each channel's
earliest as already run, the tests `before`, and each channel's earliest, the one the rung adds, as
a candidate not yet read. The export holds every candidate's reading; the arm reads only the
chosen one's. The tests before are the rung's own rows, read as the static arm reads a row
(`scorecard.arms.panel.estimates`); the bench reads each row's standard error from the pre-test
weeks of all of its channel's tests at the rung, so they can differ from the rung below's in their
errors alone, never in their readings.

1. *Before*: the static arm's two fits of the export with the tests before in their likelihoods
   (`scorecard.arms.bayes.family_fits`, `scorecard.arms.bayes.fitted`), stacked over every draw
   (`mixdyn.decisions.stacked`), their leave-one-out densities kept, so that a test's reading can
   solve the weights again.
2. *Values*: `mixdyn.experiments.value_of_tests` reads each candidate's expected regret reduction
   for the plan of the quarter, its interval and the expected value of perfect information over
   `OUTER` outer draws, each outer draw's posterior the other chains' draws reweighted by the test's
   likelihood (`Reweight`, the Gaussian rule below an effective sample size of 100), plans read on
   a grid of `GRID` spends a channel. The arm chooses the candidate of the largest reduction, read
   on the outer draws at even places, the first on a tie.
   `mixdyn.experiments.choose_tests` is asked beside it for the affordable set of the largest net
   value, at a revenue of 1 a unit of the KPI, the bench's sales being revenue, and one use, and
   for its verdict, to test or to plan now; both are recorded.
3. *Refit*: the static arm's fit again with the tests before and the chosen one, and its plan,
   `scorecard.arms.bayes.Bayes`'s `stacked`: the plan of the refitted posterior's expected return.

CHOICE: a candidate is its row's test read as a group of share `s` of the market whose spend, lift
and error are `s` times the row's, so that the model, which reads a group's spend and lift over its
share (`mixdyn.calibration.LiftEstimate`), reads the row's own likelihood whatever `s` is, and its
reduction does not move with `s`. The shares set the tests' gross costs alike: each candidate's
gross cost a unit of its share is the spend it adds and the sales the posterior before expects it
to lose, and the cost every candidate is given is half the market's go-dark of the cheapest,
`SHARE` times the least unit cost, each candidate's share that cost over its own unit cost, so that
`choose_tests` affords one test at a time: two cost twice the budget it is given, the cost. A
candidate the posterior expects to cost nothing runs at `SHARE`. CHOICE: the arm runs the test of
the largest reduction, not `choose_tests`' set, whatever the verdict: `choose_tests` ranks by the
net value, the reduction less the spend a test saves and plus the sales it loses, and the bench's
tests ran in universes of their own, so the quarter the scorecard reads pays for none of them and
the regret it reads after the refit rewards the reduction alone.

Meridian's channel-calibration score (Meridian's documentation, "Channel calibration
recommendation", updated 2026-09-11) is read beside it on the posterior before, each channel's
`0.5 implausible + 0.25 variance + 0.25 bias` in points of 100 (`calibration_scores`): the
implausible-ROI score is a bowl in the log of the readings' mean return about the geometric mean
of the bounds 0.5 and 20, its half-width at half height `log(sqrt(20 / 0.5) / share)` widening as
the channel's share of the history's spend falls; the high-variance score is 100 up to
`x = share * ratio = 0.5` and a bowl in `log(x / 0.5)` of half-width `log 2` past it, `ratio` the
80 % highest-density interval's width over the median, over the same of the ROI prior,
LogNormal(0.2, 0.9), 2.0687 (Maxima, `validation/meridian_calibration_score.mac`); the bias
score is `100 b^(log 0.5 / log 0.1)`, `b` the largest absolute correlation over the weeks of the
channel's spend with a control, 0 with no control, 50 at 0.1. A channel scoring below `CALIBRATION`
is recommended a test, the lowest first.

The variants, each planned as `stacked` is, its record the static arm's of its posterior:

* `voi`: the test of the largest reduction, refitted;
* `voi meridian`: the test of the channel Meridian's score recommends, refitted, and no test where
  every channel scores `CALIBRATION` or more; at rung 1 it is the test the score chooses against
  the one chosen by value, each read by its regret after the refit;
* `voi no test`: the posterior before, the plan the loop starts from: the realised reduction is its
  regret less `voi`'s on the same world.

Where the two choose one test the refit is one. At rung 0 no test is left to choose, and every
variant plans the posterior before. Each record carries the values beside its plan: each
candidate's share, reduction and its Monte Carlo error, each channel's calibration score, the
test cost, the expected value of perfect information, the chosen test's reduction, its
prediction and interval on the outer draws no choice read, and `choose_tests`' set, whether it is
the chosen test, and its verdict. The arm reads a
national export alone; a geo export's market is refused, since its tests are read in a share of
the market already.

A smoke run samples as the static arm's smoke run does and reads `SMOKE_OUTER` outer draws.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from mixdyn import NATIONAL, Channel
from mixdyn.decisions import Range, stacked
from mixdyn.experiments import Candidate, choose_tests, predicted_lift, value_of_tests
from scorecard.arms import bayes
from scorecard.arms import panel as reading

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping, Sequence

    from causaldyn_bench.scorecard.observe import Observation

    from mixdyn.decisions import Readings
    from mixdyn.experiments import ExperimentChoice, ExperimentValue, ExperimentValues
    from scorecard.arms.base import Planned

VOI, MERIDIAN, NO_TEST = "voi", "voi meridian", "voi no test"
SHARE = 0.5
"""The share of the market the cheapest candidate's test is priced at: half, the most a set of
tests may take beside its control (`choose_tests`)."""
OUTER, SMOKE_OUTER = 1024, 64
GRID = 512
STREAM = 5
"""The child of the export's seed the values are drawn from, apart from the static arm's fits
(0), forecasts (1, 2) and health card (3)."""
CALIBRATION = 67.5
"""Meridian's score below which a channel is recommended a test."""
ROI_BOUNDS = (0.5, 20.0)
"""Meridian's implausibly low and high returns."""
IDEAL_WIDTH, HIGH_WIDTH = 0.5, 1.0
"""Meridian's width ratio times the spend share at and below which the variance score is 100, and
the one it scores 50."""
PRIOR_WIDTH = 2.0687
"""The 80 % highest-density interval's width over the median of LogNormal(0.2, 0.9), Meridian's
default return prior and the static arm's (Maxima: 2.068704)."""
BIAS_CORRELATION = 0.1
"""Meridian's correlation of a channel's spend with a control at which the bias score is 50."""
HDI = 0.8


@dataclass(frozen=True, slots=True, eq=False)
class Posterior:
    """The static arm's fit of an export: `every` draw of its two fits stacked, their
    leave-one-out densities kept, and `fitted`, what the static arm plans from."""

    every: Readings
    fitted: bayes.Fitted


def posterior(observation: Observation, sampler: bayes.Sampler) -> Posterior:
    """The static arm's fit of `observation` by `sampler`, every lift row it holds read."""
    fits = bayes.family_fits(observation, sampler)
    return Posterior(stacked(list(fits.values())), bayes.fitted(observation, fits, sampler))


@dataclass(frozen=True, slots=True, eq=False)
class Fitted:
    """The posterior before, each chosen row's refit, each variant's row, None for no test, and why
    it is the one, and what every record keeps of the values."""

    before: bayes.Fitted
    after: Mapping[int, bayes.Fitted]
    chosen: Mapping[str, int | None]
    why: Mapping[str, str]
    flags: dict[str, float | int | bool]
    settings: dict[str, Any]


class Voi:
    """The arm; see the module. `posterior` fits an export, the static arm's fit by default."""

    variants = (VOI, MERIDIAN, NO_TEST)
    tests = True
    cells = False

    def __init__(
        self, posterior: Callable[[Observation, bayes.Sampler], Posterior] = posterior
    ) -> None:
        self.posterior = posterior

    def fit(self, observation: Observation, *, smoke: bool) -> Fitted:
        if isinstance(observation, reading.Market):
            msg = "the calibration loop reads a national export's tests, not a market's shares"
            raise ValueError(msg)
        sampler = bayes.SMOKE if smoke else bayes.SAMPLER
        new = newest(observation)
        old = [row for row in range(len(observation.lift.channel)) if row not in new]
        before = self.posterior(rows(observation, old), sampler)
        settings: dict[str, Any] = {"tests before": len(old)}
        if not new:
            why = dict.fromkeys(self.variants, "no test: the rung holds none to choose")
            return Fitted(before.fitted, {}, dict.fromkeys(self.variants), why, {}, settings)
        made, cost = candidates(observation, new, before.every)
        values, choice = _valued(observation, made, cost, before.every, smoke=smoke)
        # the first of the largest reductions, as `max` takes it
        best = max(range(len(made)), key=lambda t: values.values[t].reduction)
        voi = new[best]
        picked = None if choice.best is None else [c.label for c in choice.best.candidates]
        scores = calibration_scores(
            before.fitted.roi, np.asarray(observation.spend, dtype=float), _controls(observation)
        )
        proposed = recommended(scores)
        named = None if proposed is None else observation.channels[proposed]
        meridian = next((r for r in new if observation.lift.channel[r] == named), None)
        flags: dict[str, float | int | bool] = {
            "test cost": cost,
            "evpi": values.evpi,
            "evpi error": values.evpi_error,
            "test verdict": choice.verdict == "test",
            "choose_tests agrees": picked == [made[best].label],
        }
        for c, value in zip(made, values.values, strict=True):
            flags |= {
                f"{c.channel} share": c.share,
                f"{c.channel} reduction": value.reduction,
                f"{c.channel} reduction error": value.error,
            }
        for name, score in zip(observation.channels, scores, strict=True):
            flags[f"{name} calibration score"] = score["score"]
        flags |= _chosen(values.values[best])
        settings |= {
            "candidates": [c.label for c in made],
            "shares": {c.label: c.share for c in made},
            "outer draws": SMOKE_OUTER if smoke else OUTER,
            "grid": GRID,
            "update": "reweight, the Gaussian rule below an effective sample size of 100",
            "choose_tests": {"chosen": picked, "verdict": choice.verdict},
            # strict JSON holds no NaN: a part that is not a number is null
            "calibration scores": {
                name: {k: None if math.isnan(v) else v for k, v in score.items()}
                for name, score in zip(observation.channels, scores, strict=True)
            },
        }
        after = {
            row: self.posterior(rows(observation, [*old, row]), sampler).fitted
            for row in sorted({r for r in (voi, meridian) if r is not None})
        }
        why = {
            VOI: "the largest expected regret reduction",
            MERIDIAN: "Meridian's calibration score"
            if meridian is not None
            else f"no test: no channel scores below {CALIBRATION:g}",
            NO_TEST: "no test: the plan the loop starts from",
        }
        chosen = {VOI: voi, MERIDIAN: meridian, NO_TEST: None}
        return Fitted(before.fitted, after, chosen, why, flags, settings)

    def plan(self, fit: Fitted, variant: str, observation: Observation) -> Planned:
        row = fit.chosen[variant]
        posterior = fit.before if row is None else fit.after[row]
        planned = bayes.Bayes().plan(posterior, bayes.RESPONDS, observation)
        settings = fit.settings | {
            "chosen": None if row is None else label(observation, row),
            "chosen by": fit.why[variant],
        }
        return dataclasses.replace(
            planned,
            flags=dict(planned.flags) | fit.flags,
            settings=dict(planned.settings) | settings,
        )


def _valued(
    observation: Observation,
    made: Sequence[Candidate],
    cost: float,
    every: Readings,
    *,
    smoke: bool,
) -> tuple[ExperimentValues, ExperimentChoice]:
    """`value_of_tests`' values of the candidates `made` and `choose_tests`' choice among them at
    the test budget `cost`, both on the posterior `every`; see the module."""
    limits = {
        Channel(c): Range(float(low), float(high))
        for c, low, high in zip(
            observation.channels, observation.lower, observation.upper, strict=True
        )
    }
    child = np.random.SeedSequence(observation.seed).spawn(STREAM + 1)[STREAM]
    seed = int(child.generate_state(1)[0])
    draws = SMOKE_OUTER if smoke else OUTER
    panel = reading.panel(observation)
    args = (panel, NATIONAL, every, made, observation.budget, observation.planned, limits)
    values = value_of_tests(*args, seed=seed, draws=draws, grid=GRID)
    # the candidates' costs round apart in their last bits: one test fits, two cost twice as much
    budget = cost * (1.0 + 1e-9)
    choice = choose_tests(*args, test_budget=budget, seed=seed, draws=draws, grid=GRID)
    return values, choice


def newest(observation: Observation) -> tuple[int, ...]:
    """Each channel's test the rung adds, by its row: the channel's earliest; none at rung 0."""
    lift = observation.lift
    first: dict[str, int] = {}
    for row, name in enumerate(lift.channel):
        if name not in first or lift.start[row] < lift.start[first[name]]:
            first[name] = row
    return tuple(sorted(first.values()))


def rows(observation: Observation, kept: Collection[int]) -> Observation:
    """`observation` with the lift rows at `kept` alone, in their order."""
    lift = observation.lift
    at = sorted(kept)
    held = dataclasses.replace(
        lift,
        channel=tuple(lift.channel[i] for i in at),
        start=tuple(lift.start[i] for i in at),
        x=np.asarray(lift.x, dtype=float)[at],
        delta_x=np.asarray(lift.delta_x, dtype=float)[at],
        delta_y=np.asarray(lift.delta_y, dtype=float)[at],
        sigma=np.asarray(lift.sigma, dtype=float)[at],
        geos=None if lift.geos is None else tuple(lift.geos[i] for i in at),
        gaps=None if lift.gaps is None else np.asarray(lift.gaps, dtype=float)[at],
    )
    return dataclasses.replace(observation, lift=held)


def label(observation: Observation, row: int) -> str:
    return f"{observation.lift.channel[row]} from week {observation.lift.start[row]}"


def candidates(
    observation: Observation, new: Sequence[int], every: Readings
) -> tuple[tuple[Candidate, ...], float]:
    """Each row of `new` as a candidate at the share that prices it at the cost every candidate is
    given, and that cost; see the module."""
    whole = []
    for row in new:
        ((estimate,),) = reading.estimates(rows(observation, [row])).values()
        whole.append(
            Candidate(
                Channel(observation.lift.channel[row]),
                estimate.spend,
                estimate.planned,
                estimate.readout,
                estimate.se,
                1.0,
                label(observation, row),
            )
        )
    units = [unit_cost(every, c) for c in whole]
    priced = [u for u in units if u > 0.0]
    cost = SHARE * min(priced) if priced else 0.0
    made = tuple(_at(c, cost / u if u > 0.0 else SHARE) for c, u in zip(whole, units, strict=True))
    return made, cost


def unit_cost(every: Readings, candidate: Candidate) -> float:
    """`candidate`'s gross cost a unit of its share under the posterior `every`, at a revenue of 1
    a unit of the KPI: the spend it adds and the sales it is expected to lose, each where positive,
    both in proportion to the share (`mixdyn.experiments.Gross`)."""
    lost = -float(every.weights @ predicted_lift(every, candidate))
    added = float(np.sum(candidate.spend - candidate.planned)) / candidate.share
    return max(added, 0.0) + max(lost, 0.0)


def _at(candidate: Candidate, share: float) -> Candidate:
    """`candidate` as a group of `share` of the market, its spend, lift and error that times the
    whole market's, which the model reads as the same test."""
    scale = share / candidate.share
    return Candidate(
        candidate.channel,
        scale * candidate.spend,
        scale * candidate.planned,
        candidate.readout,
        scale * candidate.se,
        share,
        candidate.label,
    )


def _chosen(value: ExperimentValue) -> dict[str, float]:
    """What a record keeps of the chosen test's value."""
    low, high = value.interval
    return {
        "chosen reduction": value.reduction,
        "chosen reduction error": value.error,
        "chosen predicted": value.predicted,
        "chosen predicted low": low,
        "chosen predicted high": high,
        "chosen fallback": value.fallback,
        "chosen net value": value.net_value,
    }


def _controls(observation: Observation) -> np.ndarray:
    """The history's controls, `(weeks, controls)`, in the export's order."""
    weeks = np.asarray(observation.sales).shape[0]
    columns = [np.asarray(v, dtype=float) for v in observation.controls.values()]
    return np.column_stack(columns) if columns else np.zeros((weeks, 0))


def center_bowl(roi: float, share: float) -> float:
    """Meridian's implausible-ROI score of a channel's mean return `roi` at its share of the
    spend: 100 at the bounds' geometric mean, half that at `log(sqrt(20 / 0.5) / share)` from it in
    the log; 100 at a share of 0."""
    if share <= 0.0:
        return 100.0
    low, high = ROI_BOUNDS
    z = math.log(max(roi, 1e-9) / math.sqrt(low * high))
    d = max(math.log(math.sqrt(high / low) / share), 1e-9)
    return float(np.clip(100.0 * math.exp(-math.log(2.0) * (z / d) ** 2), 0.0, 100.0))


def half_bowl(ratio: float, share: float) -> float:
    """Meridian's high-variance score of a channel's width ratio at its share of the spend: 100 up
    to `IDEAL_WIDTH`, 50 at `HIGH_WIDTH`."""
    x = ratio * share
    if x <= IDEAL_WIDTH:
        return 100.0
    z = math.log(x / IDEAL_WIDTH)
    return float(100.0 * math.exp(-math.log(2.0) * (z / math.log(HIGH_WIDTH / IDEAL_WIDTH)) ** 2))


def bias_score(correlation: float) -> float:
    """Meridian's potential-bias score of a channel's largest absolute correlation with a control:
    0 at none, 50 at `BIAS_CORRELATION`, 100 at 1."""
    b = float(np.clip(correlation, 0.0, 1.0))
    if b <= 0.0:
        return 0.0
    return float(100.0 * b ** (math.log(0.5) / math.log(BIAS_CORRELATION)))


def hdi(values: np.ndarray, prob: float) -> tuple[float, float]:
    """The narrowest run of `floor(prob n) + 1` of the sorted `values`, as ArviZ reads a highest-
    density interval of draws."""
    x = np.sort(values)
    run = math.floor(prob * x.size) + 1
    if run >= x.size:
        return float(x[0]), float(x[-1])
    widths = x[run - 1 :] - x[: x.size - run + 1]
    start = int(np.argmin(widths))
    return float(x[start]), float(x[start + run - 1])


def calibration_scores(
    roi: np.ndarray, spend: np.ndarray, controls: np.ndarray
) -> list[dict[str, float]]:
    """Each channel's Meridian calibration score and its parts, from draws of its return `roi`,
    `(draws, channels)`, each draw weighed alike, the history's `spend`, `(weeks, channels)`, and
    `controls`, `(weeks, controls)`; see the module. The bias score is NaN where a channel's spend
    or every control holds still."""
    totals = spend.sum(axis=0)
    shares = totals / totals.sum() if totals.sum() > 0.0 else np.zeros_like(totals)
    out = []
    for c in range(roi.shape[1]):
        r = roi[:, c]
        median = float(np.median(r))
        low, high = hdi(r, HDI)
        width = (high - low) / abs(median) if median != 0.0 else 0.0
        correlations = [
            abs(float(np.corrcoef(spend[:, c], column)[0, 1]))
            for column in controls.T
            if np.ptp(column) > 0.0 and np.ptp(spend[:, c]) > 0.0
        ]
        if controls.shape[1] == 0:
            bias = 0.0
        elif correlations:
            bias = bias_score(max(correlations))
        else:
            bias = math.nan
        mean = float(np.mean(r))
        ratio = width / PRIOR_WIDTH
        implausible = center_bowl(mean, float(shares[c]))
        variance = half_bowl(ratio, float(shares[c]))
        out.append(
            {
                "share": float(shares[c]),
                "roi mean": mean,
                "width ratio": ratio,
                "correlation": max(correlations) if correlations else math.nan,
                "implausible roi": implausible,
                "high variance": variance,
                "potential bias": bias,
                "score": 0.5 * implausible + 0.25 * variance + 0.25 * bias,
            }
        )
    return out


def recommended(scores: Sequence[Mapping[str, float]]) -> int | None:
    """The channel Meridian's score recommends a test, by its place: the lowest of those under
    `CALIBRATION`, the first on a tie, or None."""
    under = [c for c, s in enumerate(scores) if s["score"] < CALIBRATION]
    return min(under, key=lambda c: scores[c]["score"]) if under else None
