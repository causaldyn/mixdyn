"""How a lift test is read when the channel's return has moved since it ran, and when it ran at a
spend far from the model's.

A test reads the channel over its own weeks and at its own spend. Two readings follow:

* *Staleness.* On a world whose return is still while a go-dark test runs at weeks 60 to 63 and
  then falls linearly by 40 % to week 104, the DLM (ADR 0038) reads the test at its weeks and the
  KPI after them at theirs, so its 90 % interval of the week-104 return keeps covering it. The
  static Bayesian model (ADR 0015) reads the test as a statement about its one coefficient over
  the whole history, and covers it less often.
* *A contrast at the test's spend.* A heavy-up at ten times the spend the model sees is read
  through the curve at the test's spend, `beta [g(spend) - g(planned)]` over the readout, never as
  a return at the model's spend times the spend it added. Where the curve is held, the test moves
  the return at the model's spend exactly as far as the contrast implies, in both models.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest
from chc import response
from scipy import optimize

from mixdyn import NATIONAL, Cadence, Channel, MediaPanel, Revenue
from mixdyn.calibration import LiftEstimate
from mixdyn.decisions.returns import returns
from mixdyn.models import DiscountGrid, DlmSpec, Effect, Trend, choose_dlm, fit_dlm
from mixdyn.models.bayes_mmm import BayesMedia, BayesSpec, TanhCurve, bayes_model, fit_bayes
from mixdyn.models.dlm_lift import lift_reading
from mixdyn.transforms import TanhShape, Transform

WEEKS = 104
DARK = slice(59, 63)  # weeks 60 to 63
STILL = 63  # the last week of the still return
RETURN = 2.0
DRIFT = 0.4
NOISE = 20.0
SEEDS = range(20)
LEVEL = 0.9
GRID = DiscountGrid(
    trend=(0.9, 0.95, 0.98, 1.0), effects=(0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0)
)


def a_panel(kpi: np.ndarray, spend: np.ndarray) -> MediaPanel:
    weeks = kpi.size
    return MediaPanel(
        dates=np.datetime64("2025-01-06") + np.timedelta64(7, "D") * np.arange(weeks),
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=(Channel("x"),),
        controls=(),
        kpi=kpi[None],
        kpi_unit=Revenue(),
        spend=spend[None, :, None],
        currency="EUR",
        control_values=np.zeros((1, weeks, 0)),
        filled=np.zeros((1, weeks), dtype=bool),
    )


def fading(seed: int) -> tuple[MediaPanel, LiftEstimate, float]:
    """A world of a level, one channel whose return per euro is `RETURN` to week 63 and falls
    linearly to `(1 - DRIFT) RETURN` at week 104, and noise; the channel dark at weeks 60 to 63,
    the test that reads it at the noise's error over its four weeks; and the return at week 104."""
    rng = np.random.default_rng(seed)
    planned = rng.uniform(20.0, 180.0, WEEKS)
    spend = planned.copy()
    spend[DARK] = 0.0
    week = np.arange(1, WEEKS + 1)
    path = RETURN * (1.0 - DRIFT * np.clip((week - STILL) / (WEEKS - STILL), 0.0, 1.0))
    kpi = 1000.0 + path * spend + rng.normal(0.0, NOISE, WEEKS)
    se = NOISE * 2.0  # the error of a sum of four weeks' noise
    lift = float(np.sum(path[DARK] * (spend[DARK] - planned[DARK]))) + rng.normal(0.0, se)
    panel = a_panel(kpi, spend)
    test = LiftEstimate(spend[:STILL], planned[:STILL], 4, lift, se, end=panel.dates[STILL - 1])
    return panel, test, float(path[-1])


def covered(lower: float, upper: float, truth: float) -> bool:
    return lower <= truth <= upper


@functools.cache
def dlm_coverage(*, tested: bool) -> tuple[bool, ...]:
    """Whether the DLM's 90 % interval of the coefficient at week 104 covers the return there:
    the union over the specs its likelihood keeps, the tests in that likelihood and in each fit.
    The effect is linear, so its coefficient is the return per euro of the week's spend."""
    hits = []
    for seed in SEEDS:
        panel, test, truth = fading(seed)
        spec = DlmSpec((Effect("x", 1.0),), Trend(1, 1.0))
        estimates = {"x": (test,)} if tested else None
        choice = choose_dlm(panel, NATIONAL, spec, GRID, estimates=estimates)
        bounds = choice.coefficient_bounds(LEVEL)["x"]
        hits.append(covered(bounds.lower, bounds.upper, truth))
    return tuple(hits)


@functools.cache
def static_coverage() -> tuple[bool, ...]:
    """Whether the static model's 90 % interval of week 104's contribution per euro spent there,
    `mixdyn.decisions.returns` at each draw, covers the return there, the test in its
    likelihood."""
    hits = []
    spec = BayesSpec((BayesMedia("x"),), saturation=TanhCurve())
    for seed in SEEDS:
        panel, test, truth = fading(seed)
        fit = fit_bayes(
            panel,
            NATIONAL,
            spec,
            seed=seed,
            estimates={"x": (test,)},
            chains=2,
            warmup=400,
            draws=400,
        )
        spend = panel.spend[0, :, 0]
        week = slice(WEEKS - 1, WEEKS)
        draws = [
            returns(c["x"], spend, week).contribution / spend[-1] for c in fit.channel_draws(400)
        ]
        lower, upper = np.quantile(draws, [(1.0 - LEVEL) / 2.0, (1.0 + LEVEL) / 2.0])
        hits.append(covered(float(lower), float(upper), truth))
    return tuple(hits)


def standard_error(p: float, n: int) -> float:
    return math.sqrt(p * (1.0 - p) / n)


def test_the_dlm_reads_a_stale_test_at_its_weeks_and_keeps_coverage_of_the_drifting_return() -> (
    None
):
    """17 of 20 at a nominal 0.9, within one Monte Carlo error of it, 0.067 over 20 worlds. The
    coverage asked of it, set before the run, is 0.85."""
    coverage = float(np.mean(dlm_coverage(tested=True)))
    assert coverage >= 0.85
    assert coverage >= LEVEL - 3.0 * standard_error(LEVEL, len(SEEDS))


def test_the_static_model_reads_the_test_over_the_whole_history_and_loses_the_drift() -> None:
    """Its coverage falls below the DLM's by more than three Monte Carlo errors of their
    difference, each world's coverage a draw at the nominal level."""
    dlm = float(np.mean(dlm_coverage(tested=True)))
    static = float(np.mean(static_coverage()))
    difference = math.sqrt(2.0) * standard_error(LEVEL, len(SEEDS))
    assert static < dlm - 3.0 * difference
    assert static < LEVEL - 3.0 * standard_error(LEVEL, len(SEEDS))


# ------------------------------------------------------------- a test far from the model's spend

HALF = 100.0
TEN = 10.0


def a_heavy_up(planned: np.ndarray) -> tuple[np.ndarray, int]:
    """The spend of a test at `TEN` times `planned` over its last four periods, and its readout."""
    spend = planned.copy()
    spend[-4:] *= TEN
    return spend, 4


def test_the_static_model_reads_a_heavy_up_at_its_spend_through_the_curve() -> None:
    """With the curve held at a point, the test's term alone is maximised over the return at the
    return whose coefficient times the curve's contrast at the test's spend is its lift: the
    true return, though the test spent ten times the model's spend. The plug-in reading, a return
    of the lift over the spend added, lies far below it on this saturating curve."""
    rng = np.random.default_rng(3)
    weeks = 52
    planned = rng.uniform(60.0, 140.0, weeks)
    panel = a_panel(1000.0 + rng.normal(0.0, 10.0, weeks), planned)
    spec = BayesSpec((BayesMedia("x"),), saturation=TanhCurve())
    spend, readout = a_heavy_up(planned[-8:])
    placeholder = LiftEstimate(spend, planned[-8:], readout, 1.0, 1.0)
    model = bayes_model(panel, NATIONAL, spec, estimates={"x": (placeholder,)})
    theta = np.zeros(model.dimension)
    theta[model.coordinates.index("log_half_saturation[x]")] = math.log(HALF / model.spend_unit[0])
    p = {k: float(np.asarray(v).reshape(-1)[0]) for k, v in model.parameters(theta).items()}
    channel = response.Channel(
        response.GeometricAdstock(p["retention"], length=model.lagged.shape[2], normalized=True),
        response.Tanh(p["half_saturation"] / math.atanh(0.5)),
        p["coefficient"],
    )
    window = slice(spend.size - readout, spend.size)
    contrast = response.contribution(channel, spend, window) - response.contribution(
        channel, planned[-8:], window
    )
    test = LiftEstimate(spend, planned[-8:], readout, contrast, 0.05 * contrast)
    model = bayes_model(panel, NATIONAL, spec, estimates={"x": (test,)})
    at = model.coordinates.index("log_roi[x]")

    def term(log_roi: float) -> float:
        point = theta.copy()
        point[at] = log_roi
        return -float(model.log_likelihood(point)[0, -1])

    best = optimize.minimize_scalar(term, bounds=(-4.0, 4.0), method="bounded").x
    assert math.exp(best) == pytest.approx(p["roi"], rel=1e-5, abs=0.0)
    plug_in = contrast / float(np.sum(spend - planned[-8:]))
    assert plug_in < 0.5 * p["roi"]


def test_the_dlm_reads_a_heavy_up_at_its_spend_through_the_transform() -> None:
    """The test's weights are the transform's column at ten times the spend less at the plan, at
    the readout's weeks: the lift the coefficient implies is the channel's contrast there, and a
    precise test whose lift is that contrast at the true coefficient moves the coefficient at the
    model's spend to the coefficient the contrast implies, within three of its errors. The plug-in
    coefficient, the lift over the column's change linearised at the plan, lies more than ten of
    the KPI's errors away."""
    rng = np.random.default_rng(5)
    weeks = 52
    transform = Transform(0.5, HALF, TanhShape(), 4)
    planned = rng.uniform(60.0, 140.0, weeks)
    spend, readout = a_heavy_up(planned)
    beta = 300.0
    kpi = 1000.0 + beta * transform.column(spend) + rng.normal(0.0, 10.0, weeks)
    panel = a_panel(kpi, spend)
    contrast = transform.column(spend) - transform.column(planned)
    lift = float(beta * contrast[-readout:].sum())
    test = LiftEstimate(spend, planned, readout, lift, 0.01 * lift, end=panel.dates[-1])
    spec = DlmSpec((Effect("x", 1.0, transform=transform),), Trend(1, 1.0))
    reading = lift_reading(panel, NATIONAL, spec, {"x": (test,)})
    expected = np.zeros(weeks)
    expected[-readout:] = contrast[-readout:]
    np.testing.assert_allclose(reading.weights[0], expected, rtol=1e-12, atol=0.0)
    channel = transform.channel(beta)
    window = slice(weeks - readout, weeks)
    assert lift == pytest.approx(
        response.contribution(channel, spend, window)
        - response.contribution(channel, planned, window),
        rel=1e-10,
        abs=0.0,
    )
    alone = fit_dlm(panel, NATIONAL, spec).last()
    given = fit_dlm(panel, NATIONAL, spec, estimates={"x": (test,)}).last()
    error = math.sqrt(alone[1][1, 1])
    implied = lift / float(reading.weights[0].sum())
    assert implied == pytest.approx(beta, rel=1e-12, abs=0.0)
    assert abs(given[0][1] - implied) < 3.0 * math.sqrt(given[1][1, 1])
    step = 1e-6
    linear = (
        transform.column(planned + step * (spend - planned)) - transform.column(planned)
    ) / step
    plug_in = lift / float(linear[-readout:].sum())
    assert abs(plug_in - beta) > 10.0 * error
