"""The bootstrap's forecasts: the drift run's predictive where its normal law holds, the two-stage
law of a share's count in closed form, the worlds shared by every item and rung, the strata, the
items that carry no forecast, the seed, and a scored run read against the forecasts."""

import math

import numpy as np
import pytest

pytest.importorskip("causaldyn_bench")

from causaldyn_bench.scorecard.scoring import ArmScore, WorldRecord
from scipy import stats

from scorecard import drift
from scorecard.forecast import (
    FEWEST,
    Forecasts,
    Item,
    Pilot,
    _codes,
    forecast,
    outcome,
    pilot,
    quantity,
    values,
)
from scorecard.report import RHAT

SPREAD = 4.5  # Monte Carlo standard errors a resampled chance may lie from its law's


def _pilot(columns, environments):
    return Pilot(np.asarray(environments), np.column_stack(columns).astype(float))


def _standard(n, seed):
    z = np.random.default_rng(seed).standard_normal(n)
    return (z - z.mean()) / z.std(ddof=1)


def test_where_the_pilot_is_normal_and_many_a_verdict_s_chances_are_the_drift_rule_s():
    """300 pilot worlds of mean 0.2 and SD 1, 300 scored: the predictive's mean has SD
    sqrt(2 / 300), the pilot's and the scored run's spreads both, and the chances agree with the
    drift run's Student-t predictive to within the bootstrap's own error."""
    read = 0.2 + _standard(300, 1)
    item = Item("V", 0, ("walk",), "regret", "a", "b")
    made = forecast(_pilot([read], ["walk"] * 300), [item], {"walk": 300}, seed=11, draws=8000)
    (entry,) = made.entries
    rule = drift.chances(read, 300)
    for side, chance in rule.items():
        error = math.sqrt(chance * (1.0 - chance) / 8000)
        assert abs(entry["chances"][side] - chance) <= SPREAD * error + 0.005, side
    assert entry["predicted"] == "above"
    assert entry["chance"] == pytest.approx(entry["chances"]["above"], rel=0.0, abs=0.0)


def test_a_share_s_count_follows_the_two_stage_law_of_its_pilot():
    """6 of 40 pilot worlds hold, 200 are scored: the count is binomial on 200 at the share of a
    binomial resample of 40, and the forecast's interval holds that law's mass as its chance."""
    n, count, scored = 40, 6, 200
    read = np.zeros(n)
    read[:count] = 1.0
    item = Item("S", 0, ("walk",), "failure", "a", kind="share")
    draws = 20_000
    made = forecast(_pilot([read], ["walk"] * n), [item], {"walk": scored}, seed=5, draws=draws)
    (entry,) = made.entries
    low, high = entry["interval"]
    j = np.arange(n + 1)
    outer = stats.binom.pmf(j, n, count / n)
    inside = stats.binom.cdf(high, scored, j / n) - stats.binom.cdf(low - 1, scored, j / n)
    exact = float(outer @ inside)
    assert exact >= 0.88
    assert abs(entry["chance"] - exact) <= SPREAD * math.sqrt(exact * (1.0 - exact) / draws)
    assert entry["predicted"] == f"{low}-{high} of {scored}"
    mean = float(outer @ (j / n)) * scored
    assert low < mean < high


def test_every_item_reads_the_same_worlds_at_every_rung():
    """Three items of one quantity, two at one rung and one at another, hold or fail together on
    every resample: the count of held forecasts is 0 or 3, never between."""
    read = 0.05 + _standard(30, 2)
    items = [
        Item("A", 0, ("walk",), "regret", "a", "b"),
        Item("B", 0, ("walk",), "regret", "a", "c"),
        Item("C", 4, ("walk",), "regret", "a", "b"),
    ]
    made = forecast(_pilot([read] * 3, ["walk"] * 30), items, {"walk": 100}, seed=3)
    assert set(np.unique(made.held)) <= {0, 3}
    assert made.expected == pytest.approx(sum(e["chance"] for e in made.entries), rel=0.0, abs=1e-9)
    assert np.std(made.held) > np.sqrt(
        3 * made.entries[0]["chance"] * (1 - made.entries[0]["chance"])
    )


def test_each_environment_is_resampled_on_its_own():
    """A share that never holds in one environment and always in the other counts the other's
    scored worlds on every resample."""
    read = np.r_[np.zeros(10), np.ones(10)]
    item = Item("S", 0, ("walk", "trend"), "failure", "a", kind="share")
    environments = ["walk"] * 10 + ["trend"] * 10
    made = forecast(_pilot([read], environments), [item], {"walk": 30, "trend": 50}, seed=1)
    (entry,) = made.entries
    assert (entry["interval"], entry["chance"], entry["scored"]) == ([50, 50], 1.0, 80)


def test_an_item_the_pilot_selected_or_does_not_spread_on_carries_no_forecast():
    rng = np.random.default_rng(4)
    spread = rng.normal(0.3, 1.0, 40)
    items = [
        Item("A", 0, ("walk",), "regret", "chosen", "b"),
        Item("B", 0, ("walk",), "regret", "b", "chosen"),
        Item("C", 0, ("walk",), "regret", "a", "b"),
        Item("D", 0, ("walk",), "failure", "a", kind="share"),
        Item("E", 0, ("walk",), "regret", "a", "c", kind="clause"),
    ]
    columns = [spread, spread, spread, np.zeros(40), spread - 0.6]
    made = forecast(
        _pilot(columns, ["walk"] * 40), items, {"walk": 100}, seed=2, selected={"chosen"}
    )
    reasons = [e.get("reason") for e in made.entries]
    assert reasons == [
        "the pilot selected it",
        "the pilot selected it",
        None,
        "its pilot values are one value",
        None,
    ]
    assert [e["predicted"] is None for e in made.entries] == [True, True, False, True, False]
    clause = made.entries[4]
    assert set(clause["chances"]) == {"kept", "broken"}
    assert clause["chances"]["kept"] + clause["chances"]["broken"] == pytest.approx(1.0, abs=1e-12)
    forecast_chances = made.entries[2]["chance"] + clause["chance"]
    assert made.expected == pytest.approx(forecast_chances, rel=0.0, abs=1e-12)
    assert set(np.unique(made.held)) <= {0, 1, 2}


def test_the_seed_fixes_the_forecasts():
    read = 0.1 + _standard(30, 6)
    items = [Item("A", 0, ("walk",), "regret", "a", "b")]
    laid = _pilot([read], ["walk"] * 30)
    first = forecast(laid, items, {"walk": 60}, seed=8)
    again = forecast(laid, items, {"walk": 60}, seed=8)
    other = forecast(laid, items, {"walk": 60}, seed=9)
    assert first.entries == again.entries
    np.testing.assert_array_equal(first.held, again.held)
    assert not np.array_equal(first.held, other.held)
    assert first.as_json()["seed"] == 8
    assert first.as_json()["draws"] == 4000


def test_a_forecast_draws_at_least_its_fewest_resamples_and_reads_only_held_environments():
    laid = _pilot([np.arange(10.0)], ["walk"] * 10)
    item = Item("A", 0, ("walk",), "regret", "a", "b")
    with pytest.raises(ValueError, match=f"at least {FEWEST}"):
        forecast(laid, [item], {"walk": 50}, seed=1, draws=FEWEST - 1)
    with pytest.raises(ValueError, match="every environment read"):
        forecast(laid, [Item("A", 0, ("trend",), "regret", "a", "b")], {"trend": 50}, seed=1)
    with pytest.raises(ValueError, match="every environment read"):
        forecast(laid, [item], {"trend": 50}, seed=1)


@pytest.mark.parametrize("seed", range(6))
def test_a_resample_is_read_by_the_drift_run_s_reading_rule(seed):
    rng = np.random.default_rng(seed)
    rows = rng.normal(rng.normal(0.0, 0.4, (40, 1)), 1.0, (40, 12))
    codes = _codes(rows)
    assert [("below", "across", "above")[c] for c in codes] == [drift.verdict(r) for r in rows]


def test_a_resample_whose_interval_touches_nought_is_across():
    read = np.array([1.0, 3.0])
    half = stats.t.ppf(0.975, 1) * np.std(read, ddof=1) / np.sqrt(2)
    rows = np.array(
        [
            read - read.mean() + half,
            read - read.mean() + half * (1 + 1e-9),
            read - read.mean() - half,
            read - read.mean() - half * (1 + 1e-9),
            [0.5, np.nan],
        ]
    )
    assert _codes(rows).tolist() == [1, 2, 1, 0, 1]


def _score(regret, *, crps=0.1, record=None, failure=None):
    return ArmScore(
        regret=regret,
        failure=failure,
        moved=None,
        bounds=0,
        realised=None,
        uplift=None,
        covered=None,
        crps=crps,
        record=record if record is not None else {"error": None},
    )


def _world(law, index, arms, k=0):
    return WorldRecord(
        family=1,
        environment=law,
        seed=220_000 + index,
        index=index,
        pilot=True,
        k=k,
        budget=1000.0,
        best=2000.0,
        channels=3,
        oracle_bounds=0,
        covariates={},
        arms=arms,
    )


def test_the_pilot_is_laid_out_by_world_with_what_is_not_scored_left_out():
    past = {"error": None, "flags": {"max_rhat": RHAT + 0.01, "divergences": 0}}
    past["settings"] = {"chains": 4, "draws": 1000}
    records = []
    for i in range(6):
        arms = {
            "status quo": _score(0.9),
            "a": _score(0.5, crps=None if i == 2 else 0.2),
            "b": _score(0.1, record=past if i < 2 else None, failure="x" if i == 5 else None),
        }
        records += [_world("walk", i, arms), _world("walk", i, arms, k=4)]
    records.append(_world("trend", 0, {"a": _score(0.5), "b": _score(0.5)}))
    items = [
        Item("R", 0, ("walk",), "regret", "a", "b"),
        Item("G", 0, ("walk",), "regret", "a", "b", gated=True),
        Item("C", 4, ("walk",), "crps", "a", "b"),
        Item("F", 4, ("walk",), "failure", "b", kind="share"),
        Item("T", 0, ("trend",), "regret", "a", "b"),
    ]
    laid = pilot(records, items)
    assert laid.environments.tolist() == ["trend", *["walk"] * 6]
    np.testing.assert_allclose(laid.values[1:, 0], np.full(6, 0.4), rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(laid.values[1:, 1], [-0.4, -0.4, 0.4, 0.4, 0.4, 0.4], atol=1e-15)
    assert np.isnan(laid.values[3, 2])
    assert laid.values[1:, 3].tolist() == [0.0] * 5 + [1.0]
    assert laid.values[0, 4] == 0.0
    assert np.isnan(laid.values[0, :4]).all()
    assert np.isnan(laid.values[1:, 4]).all()
    read, missing = values(records, items[2])
    assert (read.size, missing) == (5, 1)
    assert quantity(records[0], Item("L", 0, ("walk",), "regret", "a", level=0.25)) == 0.25
    with pytest.raises(ValueError, match="each world once a rung"):
        pilot([*records, records[0]], items)


def test_a_scored_run_is_read_against_the_forecasts():
    item = Item("A", 0, ("walk",), "regret", "a", "b")
    share = Item("S", 0, ("walk",), "failure", "a", kind="share")
    clause = Item("K", 0, ("walk",), "regret", "a", "c", kind="clause")
    read = -0.5 + _standard(30, 4) * 0.3
    flags = np.r_[np.ones(3), np.zeros(27)]
    made = forecast(
        _pilot([read, flags, -read], ["walk"] * 30), [item, share, clause], {"walk": 100}, seed=4
    )
    verdict, counted, kept = made.entries
    assert (verdict["predicted"], kept["predicted"]) == ("below", "broken")
    assert outcome(verdict, -0.5 + _standard(100, 5) * 0.3) == ("below", True)
    assert outcome(verdict, _standard(100, 5) * 0.3) == ("across", False)
    assert outcome(kept, _standard(100, 5) * 0.3) == ("kept", False)
    low, high = counted["interval"]
    assert outcome(counted, np.r_[np.ones(low), np.zeros(100 - low)]) == (f"{low} of 100", True)
    assert outcome(counted, np.r_[np.ones(high + 1), np.zeros(99 - high)])[1] is False
    unmade = forecast(
        _pilot([read], ["walk"] * 30), [item], {"walk": 100}, seed=4, selected={"a"}
    ).entries[0]
    with pytest.raises(ValueError, match="no forecast was made"):
        outcome(unmade, read)


def test_the_held_count_s_interval_is_its_central_95_per_cent():
    """Of 100 equally likely counts 0 to 99, the least count whose share at or below it reaches
    0.025 is 2, and 0.975's is 97."""
    made = Forecasts((), np.arange(100), 4000, 1)
    assert made.interval == (2, 97)
    assert made.expected == 49.5
