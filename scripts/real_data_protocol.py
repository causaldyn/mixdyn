"""The protocol a partner runs on one real weekly panel: each engine predicts a lift test it did
not read, and the test scores the prediction.

No public panel carries the truth an engine is scored against, so the engine is validated on
simulated worlds only (README, "Real-data validation: none, and why"). A real panel with real lift
or geo tests carries the next best thing: a test's reading is an unbiased, noisy measurement of one
quantity every engine predicts. The protocol:

1. **The data.** One weekly panel of one market (`read_panel`) and at least two lift or geo tests
   of its channels, each read into a `LiftEstimate` with its standard error, its share of the
   market and the date its readout ends. A test the panel spans is one the market ran: the panel
   holds its spend.
2. **Withhold each test in turn.** Every arm fits the panel reading the other tests where it reads
   tests, and predicts the withheld test's lift, at the market's scale, over the test's readout:
   - `ridge`: `search_ridge`, its space the scorecard's ridge arm's, reading no test (it is
     calibrated by geo tests period by period, `chc.lift.LiftTest`, not by a read estimate); the
     lift its channel reads, `channel(spend) - channel(planned)` over the readout, each over the
     test's share. It reads no test, so it is fitted once.
   - `bayes`: `fit_bayes`, the engine's default curve, a linear trend and a yearly season, the
     other tests in its likelihood (ADR 0015); the median over its draws of `predicted_lift`.
   - `dlm`: `choose_dlm` over the `bayes` fit's transforms (`transforms_from_bayes`, ADR 0020),
     the other tests in its likelihood and posterior (ADR 0038); the test's reading of the best
     spec's smoothed state path (`mixdyn.models.dlm_lift.lift_reading`), so a coefficient that
     moved is read at the test's weeks.
3. **The cells**, per arm over the withheld tests: `|predicted - lift| / se`, each test's lift and
   standard error at the market's scale, its mean and median, and the share of tests whose 95 %
   interval, `lift +- 1.96 se`, holds the prediction. A prediction off by the test's own noise
   alone reads about 0.8 on average and is held about 95 % of the time; more says the engine
   misreads the channel.

`stand_in` draws a synthetic stand-in of the data, a national weekly panel of two channels and a
go-dark and a heavy-up test whose lifts are the world's plus noise of their standard error, so
that the protocol runs end to end without a partner's data.

    JAX_PLATFORMS=cpu python scripts/real_data_protocol.py [--weeks 156] [--seed 0]
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import jax
import numpy as np
from chc import response

from mixdyn import NATIONAL, Cadence, Channel, Geo, MediaPanel, Revenue
from mixdyn.calibration import LiftEstimate
from mixdyn.decisions import readings
from mixdyn.experiments import Candidate, predicted_lift
from mixdyn.models import DiscountGrid, DlmSpec, Effect, Season, Trend, choose_dlm
from mixdyn.models.bayes_mmm import BayesFit, BayesMedia, BayesSpec, fit_bayes
from mixdyn.models.dlm_lift import lift_reading
from mixdyn.models.dlm_media import transforms_from_bayes
from mixdyn.models.ridge_mmm import Control, FourierSeason, LinearTrend
from mixdyn.models.ridge_search import (
    Folds,
    GeometricSpace,
    Interval,
    MediaSpace,
    RidgeSpace,
    search_ridge,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

jax.config.update("jax_enable_x64", True)

YEAR = 365.25 / 7.0
Z = 1.959963984540054
"""The normal's 97.5 % quantile: a test's 95 % interval is its lift plus or minus Z of its SE."""
ARMS = ("ridge", "bayes", "dlm")


@dataclass(frozen=True, slots=True)
class Settings:
    """How hard each arm works: the Bayesian fit's chains and draws, the ridge search's Sobol
    starts and polished points, the season's harmonics (0 for none), and the seed of both."""

    chains: int = 4
    warmup: int = 1000
    draws: int = 1000
    starts: int = 256
    polish: int = 8
    harmonics: int = 2
    seed: int = 0


@dataclass(frozen=True, slots=True)
class Prediction:
    """One arm's prediction of one withheld test, and the test, at the market's scale."""

    arm: str
    test: int
    channel: str
    predicted: float
    lift: float
    se: float

    @property
    def cell(self) -> float:
        """`|predicted - lift| / se`."""
        return abs(self.predicted - self.lift) / self.se

    @property
    def held(self) -> bool:
        """Whether the test's 95 % interval holds the prediction."""
        return self.cell <= Z


@dataclass(frozen=True, slots=True)
class Cells:
    """One arm's cells over the tests it predicted."""

    arm: str
    tests: int
    mean: float
    median: float
    coverage: float


def cells(predictions: Sequence[Prediction]) -> tuple[Cells, ...]:
    """Each arm's cells, in `ARMS`' order, over its predictions."""
    out = []
    for arm in ARMS:
        mine = [p for p in predictions if p.arm == arm]
        if mine:
            scores = np.array([p.cell for p in mine])
            out.append(
                Cells(
                    arm,
                    len(mine),
                    float(scores.mean()),
                    float(np.median(scores)),
                    float(np.mean([p.held for p in mine])),
                )
            )
    return tuple(out)


def protocol(
    panel: MediaPanel,
    tests: Mapping[str, Sequence[LiftEstimate]],
    settings: Settings | None = None,
) -> tuple[Prediction, ...]:
    """Every arm's prediction of each test of `tests`, by channel, withheld in turn from the fit
    of `panel`'s one geo.

    Raises:
        ValueError: on a panel of more than one geo, fewer than two tests, and what the engines
            refuse of the panel or the tests.
    """
    if len(panel.geos) != 1:
        msg = f"the protocol reads one market's panel, not {len(panel.geos)} geos"
        raise ValueError(msg)
    flat = [(channel, e) for channel, given in tests.items() for e in given]
    if len(flat) < 2:
        msg = "the protocol withholds one test and reads another: it needs at least two"
        raise ValueError(msg)
    settings = settings or Settings()
    geo = panel.geos[0]
    ridge = _ridge(panel, geo, settings)
    out = []
    for k, (channel, withheld) in enumerate(flat):
        others: dict[str, tuple[LiftEstimate, ...]] = {}
        for j, (c, e) in enumerate(flat):
            if j != k:
                others[c] = (*others.get(c, ()), e)
        lift, se = withheld.lift / withheld.share, withheld.se / withheld.share
        bayes = fit_bayes(
            panel,
            geo,
            _bayes_spec(panel, settings),
            seed=settings.seed,
            estimates=others,
            chains=settings.chains,
            warmup=settings.warmup,
            draws=settings.draws,
        )
        candidate = Candidate(
            Channel(channel),
            withheld.spend,
            withheld.planned,
            withheld.readout,
            withheld.se,
            withheld.share,
        )
        predicted = {
            "ridge": _channel_lift(ridge[channel], withheld),
            "bayes": float(np.median(predicted_lift(readings(bayes), candidate))),
            "dlm": _dlm_lift(panel, geo, bayes, others, withheld, channel, settings),
        }
        out += [Prediction(arm, k, channel, predicted[arm], lift, se) for arm in ARMS]
    return tuple(out)


def _ridge(panel: MediaPanel, geo: Geo, settings: Settings) -> dict[str, response.Channel]:
    """The ridge search's fitted channels, as the scorecard's ridge arm searches."""
    space = RidgeSpace(
        media=tuple(
            MediaSpace(
                c, GeometricSpace(Interval(0.0, 0.9)), Interval(0.5, 3.0), Interval(0.3, 1.0)
            )
            for c in panel.channels
        ),
        controls=tuple(Control(c) for c in panel.controls),
        penalty=Interval(1e-4, 1e2),
        trend=LinearTrend(3),
        season=FourierSeason(YEAR, settings.harmonics) if settings.harmonics else None,
    )
    folds = Folds.within(panel.dates.size, most=4, horizon=13, least=26)
    search = search_ridge(
        panel,
        geo,
        space,
        folds,
        seed=settings.seed,
        starts=settings.starts,
        polish=settings.polish,
    )
    return search.fit.channels()


def _channel_lift(channel: response.Channel, test: LiftEstimate) -> float:
    """The lift `channel` reads of `test` over its readout at the market's scale, nothing spent
    before the test's series."""
    tested = np.asarray(channel(test.spend / test.share))
    planned = np.asarray(channel(test.planned / test.share))
    return float(np.sum((tested - planned)[-test.readout :]))


def _bayes_spec(panel: MediaPanel, settings: Settings) -> BayesSpec:
    return BayesSpec(
        tuple(BayesMedia(c) for c in panel.channels),
        controls=tuple(Control(c) for c in panel.controls),
        trend=LinearTrend(3),
        season=FourierSeason(YEAR, settings.harmonics) if settings.harmonics else None,
    )


def _dlm_lift(
    panel: MediaPanel,
    geo: Geo,
    bayes: BayesFit,
    others: Mapping[str, Sequence[LiftEstimate]],
    withheld: LiftEstimate,
    channel: str,
    settings: Settings,
) -> float:
    """The best DLM spec's smoothed state path read through `withheld`'s weights."""
    transforms = transforms_from_bayes(bayes)
    spec = DlmSpec(
        (
            *(Effect(c, 1.0, transform=transforms[c]) for c in panel.channels),
            *(Effect(c, 1.0) for c in panel.controls),
        ),
        trend=Trend(1, 1.0),
        season=Season(YEAR, settings.harmonics, 1.0) if settings.harmonics else None,
    )
    grid = DiscountGrid(trend=(0.95, 0.98, 1.0), effects=(0.9, 0.95, 0.98, 1.0))
    fit = choose_dlm(panel, geo, spec, grid, estimates=others).fit
    reading = lift_reading(panel, geo, fit.spec, {channel: (withheld,)})
    path = np.asarray(fit.smoothed.mean)[:, reading.coordinates[0]]
    return float(reading.weights[0] @ path)


def stand_in(
    *, weeks: int = 156, seed: int = 0
) -> tuple[MediaPanel, dict[str, tuple[LiftEstimate, ...]]]:
    """A synthetic stand-in for a partner's data: a national weekly panel of `tv` and `search`,
    each a geometric carryover and a `tanh` curve, a level and noise; `tv` dark for four weeks a
    third of the way in and `search` doubled for four weeks two thirds of the way in, the market's
    spend as the tests ran; each test read over its weeks and the four after, its lift the world's
    plus noise of its standard error, the noise's over its eight weeks."""
    rng = np.random.default_rng(seed)
    retention, half, coefficient, noise = (0.5, 0.2), (120.0, 40.0), (900.0, 500.0), 40.0
    planned = np.stack([rng.uniform(60.0, 180.0, weeks), rng.uniform(20.0, 60.0, weeks)], axis=1)
    spend = planned.copy()
    dark, doubled = weeks // 3, 2 * weeks // 3
    spend[dark : dark + 4, 0] = 0.0
    spend[doubled : doubled + 4, 1] *= 2.0
    truth = [
        response.Channel(
            response.GeometricAdstock(retention[j], length=8, normalized=True),
            response.Tanh(half[j] / math.atanh(0.5)),
            coefficient[j],
        )
        for j in range(2)
    ]
    kpi = 2000.0 + sum(np.asarray(truth[j](spend[:, j])) for j in range(2))
    kpi = kpi + rng.normal(0.0, noise, weeks)
    dates = np.datetime64("2023-01-02") + np.timedelta64(7, "D") * np.arange(weeks)
    panel = MediaPanel(
        dates=dates,
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=(Channel("tv"), Channel("search")),
        controls=(),
        kpi=kpi[None],
        kpi_unit=Revenue(),
        spend=spend[None],
        currency="EUR",
        control_values=np.zeros((1, weeks, 0)),
        filled=np.zeros((1, weeks), dtype=bool),
    )
    se = noise * math.sqrt(8.0)
    tests: dict[str, tuple[LiftEstimate, ...]] = {}
    for j, (name, start) in enumerate((("tv", dark), ("search", doubled))):
        end = start + 8
        tested = spend[:end, j]
        held = planned[:end, j]
        lift = float(np.sum((np.asarray(truth[j](tested)) - np.asarray(truth[j](held)))[-8:]))
        tests[name] = (
            LiftEstimate(tested, held, 8, lift + rng.normal(0.0, se), se, end=dates[end - 1]),
        )
    return panel, tests


def report(predictions: Sequence[Prediction]) -> str:
    """The cells and each prediction, as Markdown."""
    lines = [
        "| arm | tests | mean of abs(pred - lift) / se | median | held by the 95 % interval |",
        "| --- | --- | --- | --- | --- |",
        *(
            f"| {c.arm} | {c.tests} | {c.mean:.2f} | {c.median:.2f} | {c.coverage:.2f} |"
            for c in cells(predictions)
        ),
        "",
        "| arm | test | channel | predicted | lift | se |",
        "| --- | --- | --- | --- | --- | --- |",
        *(
            f"| {p.arm} | {p.test} | {p.channel} | {p.predicted:.1f} | {p.lift:.1f} | {p.se:.1f} |"
            for p in predictions
        ),
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weeks", type=int, default=156)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    panel, tests = stand_in(weeks=arguments.weeks, seed=arguments.seed)
    print(report(protocol(panel, tests, replace(Settings(), seed=arguments.seed))))


if __name__ == "__main__":
    main()
