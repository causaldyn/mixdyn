"""Each arm's response as the bench reads it: its draws of each channel in `chc.response`'s terms,
held by the bench to the fit's own account of the history within its tolerance, and read for each
channel's return on the export's window."""

import dataclasses
import json
import math
from types import MappingProxyType

import numpy as np
import pytest

pytest.importorskip("causaldyn_bench")

from causaldyn_bench.scorecard.mapping import TOLERANCE, drawn, residual
from causaldyn_bench.scorecard.returns import drawn_returns
from causaldyn_bench.scorecard.scoring import crps
from causaldyn_bench.scorecard.track_m2 import TRACK_M2
from chc import response

from mixdyn import NATIONAL
from mixdyn.models import DlmChoice, DlmSpec, Effect, Season, Trend, fit_dlm
from mixdyn.transforms import HillShape, TanhShape, Transform
from scorecard.arms import bayes, dlm
from scorecard.arms import panel as reading
from scorecard.arms.base import Planned, _fields
from scorecard.arms.mapped import HELD, held, mapped


@pytest.fixture(scope="module")
def observation():
    return TRACK_M2.observe(TRACK_M2.world("drawn", 905), 0)


def _spec(observation, shapes, retention, multiple, *, sign="positive"):
    """The arm's model with each channel's transform at `retention`, its half-saturation point
    `multiple` times its mean positive spend, and its shape of `shapes`."""
    spend = np.asarray(observation.spend, dtype=float)
    media = [
        Effect(
            name,
            0.99,
            sign=sign,
            transform=Transform(
                retention,
                multiple * float(spend[spend[:, j] > 0.0, j].mean()),
                shapes[j],
                observation.kernel_length,
            ),
        )
        for j, name in enumerate(observation.channels)
    ]
    return DlmSpec(
        effects=(*media, *(Effect(c, 0.99) for c in observation.controls)),
        trend=Trend(1, 0.99),
        season=Season(dlm.YEAR, dlm.HARMONICS, 1.0),
    )


SHAPES = (
    (TanhShape(), TanhShape(), TanhShape()),
    (HillShape(2.0), HillShape(1.0), TanhShape()),
)


@pytest.fixture(scope="module")
def choice(observation):
    """Two fitted specs, one a `tanh` on every channel, the other a Hill on two of them, each
    holding 4 chains of 100 draws of its signed posterior."""
    data = reading.panel(observation)
    specs = (_spec(observation, SHAPES[0], 0.5, 1.0), _spec(observation, SHAPES[1], 0.3, 2.0))
    fits = [fit_dlm(data, NATIONAL, s, draws=100, seed=0) for s in specs]
    return DlmChoice(
        specs, tuple(f.filtered for f in fits), 0, MappingProxyType(dict(enumerate(fits)))
    )


@pytest.fixture(scope="module")
def made(choice, observation):
    return dlm._response(choice, observation)


def test_the_dlm_response_is_held_by_the_bench_to_the_filter_s_own_parts(made, observation):
    """Each draw's channel, its spec's transform at a coefficient of 1 times the coefficient's path,
    returns over the history what its part of the draw's mean says, week by week in the draws
    held: the transforms the planner reads are the columns the filter was fitted on."""
    assert residual(made, observation) <= TOLERANCE
    assert made["decomposition"]["held"] == held(dlm.DRAWS).tolist()
    json.dumps(made, allow_nan=False)


def test_the_dlm_response_spreads_its_draws_over_the_fitted_specs_each_with_its_curves(
    made, observation
):
    """Half the draws a spec, each channel's curve its spec's, and each coefficient a path over the
    history's weeks, no two alike: each spec's draws are taken without replacement."""
    parsed = drawn(made, observation.channels)
    weeks = observation.sales.shape[0]
    for j, name in enumerate(observation.channels):
        kinds = [type(shapes[j]).__name__.removesuffix("Shape") for shapes in SHAPES]
        counts = {kind: parsed[name].curves.count(kind) for kind in set(kinds)}
        assert counts == {kind: kinds.count(kind) * dlm.DRAWS // 2 for kind in set(kinds)}
        assert parsed[name].coefficient.shape == (dlm.DRAWS, weeks)
        assert len({tuple(path) for path in parsed[name].coefficient}) == dlm.DRAWS
        assert np.all(parsed[name].coefficient >= 0.0)  # drawn from the posterior held at 0


def test_the_dlm_response_is_read_for_each_channel_s_return_on_the_window(made, observation):
    returns = drawn_returns(made, observation)
    assert returns.window == observation.roi_window
    assert returns.roi.shape == returns.marginal.shape == (dlm.DRAWS, len(observation.channels))
    assert np.all(np.isfinite(returns.roi))
    assert np.all(np.isfinite(returns.marginal))


def test_the_dlm_response_is_drawn_from_the_export_s_seed_apart_from_the_readings(
    choice, made, observation
):
    """The same seed draws the same response; the readings' last week is another sample."""
    again = dlm._response(choice, observation)
    assert again == made
    last = {
        name: sorted(path[-1] for path in made["parameters"][name]["coefficient"])
        for name in observation.channels
    }
    read = choice.readings(dlm.DRAWS, observation.seed)
    for j, name in enumerate(observation.channels):
        assert last[name] != sorted(float(r[j].coefficient) for r in read)


def test_a_dlm_spec_whose_posterior_is_not_drawn_has_no_response(observation):
    data = reading.panel(observation)
    spec = _spec(observation, SHAPES[0], 0.5, 1.0, sign=None)
    fit = fit_dlm(data, NATIONAL, spec)
    alone = DlmChoice((spec,), (fit.filtered,), 0, MappingProxyType({0: fit}))
    with pytest.raises(ValueError, match="no draws of its posterior"):
        dlm._response(alone, observation)


def test_a_response_maps_geometric_kernels_and_tanh_or_hill_curves_alone():
    kernel = response.GeometricAdstock(0.5, length=4, normalized=True)
    tanh = response.Channel(kernel, response.Tanh(2.0), 1.0)
    one = np.zeros((1, 1))
    weeks = np.zeros((1, 3, 1))
    assert mapped(["a"], [(tanh,)], one, weeks)["curve"] == {"a": ["Tanh"]}
    other = response.Channel(
        response.GeometricAdstock(0.5, length=5, normalized=True), tanh.curve, 1.0
    )
    with pytest.raises(ValueError, match="first's length"):
        mapped(["a"], [(tanh,), (other,)], np.zeros((2, 1)), np.zeros((2, 3, 1)))
    exponential = response.Channel(kernel, response.Exponential(2.0), 1.0)
    with pytest.raises(ValueError, match="maps a Tanh or a Hill"):
        mapped(["a"], [(exponential,)], one, weeks)
    with pytest.raises(ValueError, match="need totals"):
        mapped(["a"], [(tanh,)], np.zeros((1, 2)), weeks)


def test_the_held_draws_are_evenly_spaced_and_at_most_held():
    assert held(400).tolist() == [0, 133, 266, 399]
    assert held(2).tolist() == [0, 1]
    assert HELD == 4


def test_a_record_carries_a_finite_response_and_says_why_it_has_none_otherwise(observation):
    status_quo = np.asarray(observation.status_quo, dtype=float)
    record = _fields(Planned(weekly=status_quo, response={"curve": {"a": [1.0]}}), observation)
    assert record["response"] == {"curve": {"a": [1.0]}}
    record = _fields(Planned(weekly=status_quo, response={"a": [[1.0, math.inf]]}), observation)
    assert "response" not in record
    assert record["n/a"]["response"] == "the response holds a number that is not finite"


def test_a_record_holds_the_share_of_its_plan_outside_the_logged_range(observation):
    """A plan at each channel's most logged spend is inside every range; doubling the first
    channel's puts half its spend outside, that channel's part of the plan's whole spend. A plan
    that spends nothing, or below 0 somewhere, records no share."""
    most = np.asarray(observation.spend, dtype=float).max(axis=0)
    inside = _fields(Planned(weekly=most), observation)["flags"]
    assert inside == {"out of range": 0.0} | {
        f"{c} out of range": 0.0 for c in observation.channels
    }
    doubled = most.copy()
    doubled[0] *= 2.0
    flags = _fields(Planned(weekly=doubled), observation)["flags"]
    share = most[0] / doubled.sum()
    first, *others = observation.channels
    assert flags["out of range"] == pytest.approx(share, rel=1e-15, abs=0.0)
    assert flags[f"{first} out of range"] == pytest.approx(share, rel=1e-15, abs=0.0)
    assert [flags[f"{c} out of range"] for c in others] == [0.0] * len(others)
    assert "out of range" not in _fields(Planned(weekly=0.0 * most), observation)["flags"]
    below = most.copy()
    below[0] = -1e-12
    assert "out of range" not in _fields(Planned(weekly=below), observation)["flags"]


@pytest.fixture(scope="module")
def stacked(observation):
    return bayes.Bayes().fit(observation, smoke=True)


def test_the_bayes_flags_hold_each_fit_s_health_card_and_r2(stacked, observation):
    """Each family's card, ADR 0060's cells, and its R² in sample, every channel's cells under its
    name: a p-value and a probability in [0, 1], a consistency a prior's quantile, a width above
    0, and an R² at most 1."""
    flags = stacked.flags
    for family in bayes.FAMILIES:
        assert 0.0 <= flags[f"{family} ppp"] <= 1.0
        assert 0.0 <= flags[f"{family} negative baseline"] <= 1.0
        assert flags[f"{family} r2"] <= 1.0
        for c in observation.channels:
            assert math.isfinite(flags[f"{family} {c} prior shift"])
            assert flags[f"{family} {c} prior width"] > 0.0
            assert 0.0 <= flags[f"{family} {c} prior consistency"] <= 1.0


def test_the_bayes_response_is_held_by_the_bench_to_each_draw_s_return(stacked, observation):
    """Each reading's channel returns over the history its draw's return per euro times the
    history's spend, the identity the model is parametrised by: the readings are the model's draws,
    paired with their returns, in the model's units."""
    made = bayes._response(stacked, observation)
    assert residual(made, observation) <= TOLERANCE
    parsed = drawn(made, observation.channels)
    hill = round(stacked.flags["hill weight"] * len(stacked.readings))
    for name in observation.channels:
        assert parsed[name].curves == ("Hill",) * hill + ("Tanh",) * (len(stacked.readings) - hill)
    json.dumps(made, allow_nan=False)


def test_a_bayes_reading_misread_by_one_draw_is_off_the_model_s_returns(stacked, observation):
    """The check bites: each reading paired with its neighbour's return is refused."""
    moved = dataclasses.replace(stacked, roi=np.roll(stacked.roi, 1, axis=0))
    assert residual(bayes._response(moved, observation), observation) > TOLERANCE


def test_a_bayes_fit_s_forecast_and_response_are_recorded_under_stacked_alone(stacked, observation):
    made = bayes.Bayes()
    planned = {v: made.plan(stacked, v, observation) for v in made.variants}
    response = planned[bayes.RESPONDS].response
    assert response is not None
    assert residual(response, observation) <= TOLERANCE
    assert planned[bayes.RESPONDS].forecast is stacked.forecast
    assert "forecast" not in planned[bayes.RESPONDS].absent
    assert planned["cvar 0.1"].response is None
    assert planned["cvar 0.1"].forecast is None
    assert planned["cvar 0.1"].absent["response"] == "recorded under 'stacked'"
    assert planned["cvar 0.1"].absent["forecast"] == "recorded under 'stacked'"


def test_the_bayes_plans_read_the_least_bulk_ess_of_the_fits_their_readings_hold(
    stacked, observation
):
    counts = bayes.shares(stacked.flags["hill weight"], len(stacked.readings))
    assert sum(counts.values()) == len(stacked.readings)
    read = [stacked.flags[f"{family} min ess bulk"] for family, n in counts.items() if n]
    assert read
    assert stacked.ess == min(read)
    arm = bayes.Bayes()
    assert {arm.plan(stacked, v, observation).ess for v in arm.variants} == {stacked.ess}


def test_the_least_ess_is_of_the_families_the_readings_hold():
    flags = {"hill min ess bulk": 300.0, "tanh min ess bulk": 900.0}
    assert bayes.least_ess(flags, {"hill": 100, "tanh": 300}) == 300.0
    assert bayes.least_ess(flags, {"hill": 0, "tanh": 400}) == 900.0
    assert math.isnan(
        bayes.least_ess(flags | {"tanh min ess bulk": math.nan}, {"hill": 1, "tanh": 399})
    )


def test_shares_round_the_hill_s_weight_as_python_rounds():
    assert bayes.shares(0.45, 400) == {"hill": 180, "tanh": 220}
    assert bayes.shares(0.00125, 400) == {"hill": 0, "tanh": 400}  # 0.5 rounds to the even 0
    assert bayes.shares(0.00375, 400) == {"hill": 2, "tanh": 398}  # 1.5 rounds to the even 2
    assert bayes.shares(1.0, 400) == {"hill": 400, "tanh": 0}


def test_the_bayes_forecast_is_the_quarter_s_sales_at_the_status_quo_as_the_bench_scores_it(
    stacked, observation
):
    """A draw of each of the quarter's weeks at each reading's draw, every channel at the status
    quo and the controls at the quarter's, which the bench's CRPS reads against the quarter's
    sales."""
    assert stacked.forecast is not None
    assert stacked.forecast.shape == (len(stacked.readings), observation.planned)
    assert np.isfinite(stacked.forecast).all()
    truth = TRACK_M2.truth(TRACK_M2.world("drawn", 905))
    assert crps(stacked.forecast, truth.target) >= 0.0


@pytest.mark.parametrize(
    ("arm", "mixture"),
    [(dlm.Dlm(), "dlm mixture"), (dlm.DlmUntested(), "dlm untested mixture")],
)
def test_a_dlm_fit_s_forecast_and_response_are_recorded_under_the_mixture_alone(
    arm, mixture, choice, made, observation
):
    forecast = dlm._forecast(choice, observation)
    readings = choice.readings(dlm.DRAWS, observation.seed)
    least, best = dlm._ess(list(choice.kept.values())), dlm._ess([choice.fit])
    fitted = dlm.Fitted(choice, readings, forecast.kpi, {}, {}, {}, least, best)
    planned = {v: arm.plan(fitted, v, observation) for v in arm.variants}
    plug_in, _, worst, over_equal = arm.variants
    # the plug-in's channels are the best spec's draws' means; the others read every fitted spec's
    assert least == min(f.signed.ess for f in choice.kept.values())
    assert {v: p.ess for v, p in planned.items()} == {
        plug_in: choice.fit.signed.ess,
        mixture: least,
        worst: least,
        over_equal: least,
    }
    assert {v: p.settings.get("level") for v, p in planned.items()} == {
        plug_in: None,
        mixture: 1.0,
        worst: dlm.CVAR,
        over_equal: dlm.CVAR,
    }
    assert planned[mixture].response == made
    assert planned[mixture].forecast is forecast.kpi
    assert "forecast" not in planned[mixture].absent
    for variant in set(arm.variants) - {mixture}:
        assert planned[variant].response is None
        assert planned[variant].forecast is None
        assert planned[variant].absent["response"] == f"recorded under {mixture!r}"
        assert planned[variant].absent["forecast"] == f"recorded under {mixture!r}"


def test_the_dlm_forecast_is_the_quarter_s_sales_at_the_status_quo_as_the_bench_scores_it(
    choice, observation
):
    """Draws of each of the quarter's weeks, every channel at the status quo and the controls at
    the quarter's, which the bench's CRPS reads against the quarter's sales."""
    forecast = dlm._forecast(choice, observation)
    assert forecast.kpi.shape == (dlm.DRAWS, observation.planned)
    assert np.isfinite(forecast.kpi).all()
    assert 0.0 < forecast.acceptance <= 1.0
    truth = TRACK_M2.truth(TRACK_M2.world("drawn", 905))
    assert crps(forecast.kpi, truth.target) >= 0.0
