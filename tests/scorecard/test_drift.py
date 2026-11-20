"""The drift run's pre-registered reading: the rule's chances against a simulation of the predictive
it states, the verdicts, the share intervals, the items' values with the unscored worlds left out
and the second reading, the sample-size rule and the default procedure's rule."""

from dataclasses import asdict

import numpy as np
import pytest

pytest.importorskip("causaldyn_bench")

from causaldyn_bench.scorecard.scoring import ArmScore, WorldRecord
from scipy import stats

from scorecard.drift import (
    DRIFTING,
    EVERY,
    LAWS,
    PRIMARY,
    VARIANTS,
    Item,
    chances,
    outcome,
    share_interval,
    values,
    variant,
    verdict,
    worlds,
)
from scorecard.report import RHAT

DRAWS = 400_000
SPREAD = 4.5  # Monte Carlo standard errors a simulated chance may lie from the rule's


def _score(regret, *, crps=0.1, record=None):
    return ArmScore(
        regret=regret,
        failure=None,
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
        seed=1000 * LAWS.index(law) + index,
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


def _pilot(differences, *, arm=EVERY[1], per_law=10):
    """Every law's `per_law` worlds, each arm's regret the primary's but `arm`'s, which is the
    primary's less `differences` at each world of every law."""
    rng = np.random.default_rng(7)
    records = []
    for law in LAWS:
        for i in range(per_law):
            base = float(rng.uniform(0.2, 0.8))
            arms = {name: _score(base) for name in (PRIMARY, *EVERY, *VARIANTS)}
            arms[arm] = _score(base - differences[i])
            records.append(_world(law, i, arms))
    return records


@pytest.mark.parametrize(
    ("pilot", "scored"),
    [([0.3, -0.1, 0.2, 0.05, -0.2, 0.15, 0.4, 0.0, 0.1, -0.05], 100), ([-0.4, 0.1, -0.3], 500)],
)
def test_the_rules_chances_are_the_predictive_of_a_future_mean(pilot, scored):
    """Under the prior uniform on the mean and on the log of the spread, the spread's law is
    (n - 1) s^2 / chi^2(n - 1), the mean's normal about the pilot's, and the scored mean's normal
    about it; the chance of each side of the scored half-width, drawn so, is the rule's."""
    read = np.asarray(pilot)
    n, m, s = read.size, float(read.mean()), float(read.std(ddof=1))
    rng = np.random.default_rng(11)
    sigma = np.sqrt((n - 1) * s**2 / rng.chisquare(n - 1, DRAWS))
    mean = rng.normal(m, sigma / np.sqrt(n))
    future = rng.normal(mean, sigma / np.sqrt(scored))
    half = stats.t.ppf(0.975, scored - 1) * s / np.sqrt(scored)
    simulated = {"below": np.mean(future < -half), "above": np.mean(future > half)}
    simulated["across"] = 1.0 - simulated["below"] - simulated["above"]
    rule = chances(read, scored)
    for side, chance in rule.items():
        error = np.sqrt(chance * (1.0 - chance) / DRAWS)
        assert abs(simulated[side] - chance) <= SPREAD * error, side
    assert sum(rule.values()) == pytest.approx(1.0, rel=0.0, abs=1e-12)


def test_the_rule_refuses_a_pilot_whose_values_do_not_spread():
    with pytest.raises(ValueError, match="spread"):
        chances(np.full(10, 0.1), 100)


@pytest.mark.parametrize(
    ("read", "expected"),
    [
        ([-1.0, -1.1, -0.9, -1.05], "below"),
        ([1.0, 1.1, 0.9, 1.05], "above"),
        ([-1.0, 1.0, -0.5, 0.5], "across"),
    ],
)
def test_a_verdict_reads_where_the_interval_lies(read, expected):
    assert verdict(np.asarray(read)) == expected


def test_a_verdict_whose_interval_touches_nought_is_across():
    """The ends are read strictly, as the bench's reading rule reads them."""
    read = np.array([1.0, 3.0])
    half = stats.t.ppf(0.975, 1) * np.std(read, ddof=1) / np.sqrt(2)
    assert verdict(read - read.mean() + half) == "across"
    assert verdict(read - read.mean() + half * (1 + 1e-9)) == "above"
    assert verdict(read - read.mean() - half) == "across"
    assert verdict(read - read.mean() - half * (1 + 1e-9)) == "below"


@pytest.mark.parametrize(("count", "n", "scored"), [(0, 50, 500), (7, 50, 500), (60, 60, 600)])
def test_the_share_interval_holds_its_central_share(count, n, scored):
    """The count's beta-binomial predictive, drawn as the share from Jeffreys' posterior and the
    count binomial at it, falls in the interval as often as the interval says."""
    low, high, chance = share_interval(count, n, scored)
    assert chance >= 0.9
    rng = np.random.default_rng(13)
    drawn = rng.binomial(scored, rng.beta(count + 0.5, n - count + 0.5, DRAWS))
    inside = np.mean((drawn >= low) & (drawn <= high))
    assert abs(inside - chance) <= SPREAD * np.sqrt(chance * (1.0 - chance) / DRAWS)


def test_an_items_values_leave_out_the_worlds_it_is_not_scored_on():
    records = _pilot(np.linspace(-0.2, 0.2, 10))
    records[3].arms[EVERY[1]] = _score(0.5, crps=None)
    read, missing = values(records, Item("P9", 0, "walk", "crps", PRIMARY, EVERY[1]), 10)
    assert (read.size, missing) == (9, 1)


def test_the_second_reading_puts_a_plan_past_its_bounds_at_the_status_quo():
    past = {"error": None, "flags": {"max_rhat": RHAT + 0.01, "divergences": 0}}
    past["settings"] = {"chains": 4, "draws": 1000}
    records = []
    for i in range(10):
        arms = {name: _score(0.5) for name in (PRIMARY, *EVERY)}
        arms["status quo"] = _score(0.9)
        arms["meridian"] = _score(0.1, record=past if i < 4 else None)
        records.append(_world("walk", i, arms))
    first = Item("P10", 0, "walk", "regret", PRIMARY, "meridian")
    second = Item("P10", 0, "walk", "regret", PRIMARY, "meridian", gated=True)
    np.testing.assert_array_equal(values(records, first, 10)[0], np.full(10, 0.4))
    expected = np.array([-0.4] * 4 + [0.4] * 6)
    np.testing.assert_allclose(values(records, second, 10)[0], expected, rtol=0.0, atol=1e-15)


def test_a_pooled_item_reads_every_drifting_law_and_no_other():
    """Each law's differences are its place among the laws, so the pooled reading names them."""
    records = _pilot(np.zeros(10))
    for record in records:
        place = LAWS.index(record.environment)
        record.arms[EVERY[1]] = _score(record.arms[PRIMARY].regret - place)
    read, _ = values(records, Item("P1", 0, "drifting", "regret", PRIMARY, EVERY[1]), 10)
    assert sorted(set(read.tolist())) == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert read.size == 50
    assert set(DRIFTING) == set(LAWS) - {"static"}


@pytest.mark.parametrize(("sd", "expected"), [(0.3, 40), (0.5, 100), (0.05, 30)])
def test_the_sample_size_rule_gives_the_fewest_tens_that_halve_the_difference(sd, expected):
    """A difference of mean 0.2: at sd 0.3, 30 worlds give a half-width of 0.112 and 40 of 0.096,
    half the difference being 0.1; at sd 0.5 no count up to 100 does; at 0.05 the floor holds."""
    z = np.linspace(-1.0, 1.0, 10)
    differences = 0.2 + sd * z / np.std(z, ddof=1)
    assert worlds(_pilot(-differences), 10) == expected


def test_the_default_procedure_changes_only_where_a_variant_beats_it():
    assert variant(_pilot(np.linspace(-0.2, 0.2, 10), arm=VARIANTS[0]), 10) == PRIMARY
    better = 0.3 + np.linspace(-0.05, 0.05, 10)
    assert variant(_pilot(better, arm=VARIANTS[0]), 10) == VARIANTS[0]


@pytest.mark.parametrize(
    ("shift", "kind", "predicted", "held"),
    [
        (-0.5, "verdict", "below", True),
        (-0.5, "verdict", "across", False),
        (0.5, "clause", "kept", False),
        (0.0, "clause", "kept", True),
    ],
)
def test_a_prediction_holds_where_the_scored_reading_is_the_predicted_one(
    shift, kind, predicted, held
):
    records = _pilot(shift + np.linspace(-0.1, 0.1, 10))
    item = Item("P5", 0, "walk", "regret", PRIMARY, EVERY[1], kind=kind)
    entry = {"item": asdict(item), "predicted": predicted}
    assert outcome(entry, records, 10)["held"] is held


def _block(scale, means):
    """A time-varying record's multiplier block: the lengthscale's mean, and each channel's mean
    over the history's last week and the quarter's 13."""
    channels = {
        name: {"mean": path, "interval": [[0.0, 9.0]] * 14, "correlation": [1.0] * 14}
        for name, path in means.items()
    }
    return {
        "weeks": 14,
        "lengthscale": {"mean": scale, "interval": [0.5, 9.0]},
        "channels": channels,
    }


def test_p7_reads_the_lengthscale_and_the_quarters_mean_multiplier_off_its_last_week():
    """The history's last week is left out of the quarter's mean, and a record without the block
    is not scored on P7."""
    records = _pilot(np.linspace(-0.2, 0.2, 10))
    walk = [r for r in records if r.environment == "walk"]
    for i, record in enumerate(walk[:9]):
        means = {"pla": [9.0] + [1.0 + 0.1 * i] * 13, "meta": [9.0] + [1.0 + 0.3 * i] * 13}
        block = {"error": None, "multiplier": _block(1.5 + i, means)}
        record.arms["pymc-marketing shared"] = _score(0.5, record=block)
    item = Item("P7", 0, "walk", "lengthscale", "pymc-marketing shared", level=13.0)
    scale, missing = values(records, item, 10)
    np.testing.assert_allclose(scale, 1.5 + np.arange(9) - 13.0, rtol=0.0, atol=1e-12)
    assert missing == 1
    item = Item("P7", 0, "walk", "multiplier", "pymc-marketing shared", level=1.0)
    mean, _ = values(records, item, 10)
    np.testing.assert_allclose(mean, 0.2 * np.arange(9), rtol=0.0, atol=1e-12)
