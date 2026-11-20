"""mixdyn.experiments.geo_design: a geo test's design, its balance and its power.

Pairs adjacent in a score's order are the least summed gap any pairing of the geos has, the geos
left over included, and blocks of three the least summed range; a seeded draw within the pairs is
reproduced by its seed and moved by another; over ten geos whose means run 1 to 10 the pairs'
designs are every one of the 32 the pairs allow, their difference in means is 0 on average and
their worst standardised difference is `1/sqrt(10)` (Maxima), where treating the top five reads
`sqrt(10)`; a control match is the least summed Mahalanobis distance of every one-to-one match,
where a greedy one is not; and on twenty geos of log-normal levels and 5 % weekly noise, ten
treated, a four-week test's power read from the history detects a lift below 5 %, its size is
the test's, and a longer readout or pre-period detects a smaller lift; where the noise is a
first-order autoregression of 0.8, the default reader, `read_did` on the groups' sums, rejects a
true null at no more than its level, and `log_did` more often on the same markets (ADR 0055).
A power audit reads every backtest once, flags a size outside [0.03, 0.08] and refuses a short
pre-period or fewer than 8 backtests; the comparator design stratifies on the geos' mean, CV and
slope and chooses among candidates by the time-based regression (ADRs 0049, 0055).
"""

from __future__ import annotations

import itertools
import math
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from mixdyn import Cadence, Channel, Geo, MediaPanel, Revenue
from mixdyn.experiments.geo_design import (
    Blocks,
    GeoDesign,
    _binomial_p_value,
    _detectable,
    _read_did,
    _read_tbr,
    _tbr_scores,
    assign,
    audit_power,
    balance,
    log_did,
    match_controls,
    pair_geos,
    power,
    stratified_design,
)
from mixdyn.experiments.reading import GeoTest, read_tbr

ALPHA = 0.05


def _least_range(values: list[float], size: int) -> float:
    """The least summed range of blocks of `size` over `values`, `len % size` of them left out,
    by enumerating every partition."""
    if len(values) < size:
        return 0.0
    if len(values) % size:
        return min(_least_range(values[:k] + values[k + 1 :], size) for k in range(len(values)))
    first, rest = values[0], values[1:]
    best = math.inf
    for others in itertools.combinations(range(len(rest)), size - 1):
        block = [first, *(rest[k] for k in others)]
        left = [v for k, v in enumerate(rest) if k not in others]
        best = min(best, max(block) - min(block) + _least_range(left, size))
    return best


@settings(max_examples=60, deadline=None)
@given(
    st.lists(st.integers(-50, 50), min_size=3, max_size=7),
    st.sampled_from([2, 3]),
)
def test_adjacent_blocks_are_the_least_summed_range(score: list[int], size: int) -> None:
    values = [float(v) for v in score]
    blocks = pair_geos(values, size=size)
    assert len(blocks.left_out) == len(values) % size
    assert all(len(b) == size for b in blocks.blocks)
    rows = sorted([g for b in blocks.blocks for g in b] + list(blocks.left_out))
    assert rows == list(range(len(values)))
    summed = sum(max(values[g] for g in b) - min(values[g] for g in b) for b in blocks.blocks)
    assert summed == _least_range(values, size)


def test_an_odd_count_leaves_out_the_geo_whose_leaving_costs_least() -> None:
    blocks = pair_geos([0.0, 1.0, 1.1, 5.0, 5.2])
    assert blocks == Blocks(((1, 2), (3, 4)), (0,))
    assert pair_geos([0.0, 0.1, 4.0, 4.2, 9.0]).left_out == (4,)


def test_a_tie_leaves_out_the_later_geo_in_the_order() -> None:
    assert pair_geos([3.0, 1.0, 2.0]) == Blocks(((1, 2),), (0,))
    assert pair_geos([1.0, 2.0, 3.0, 4.0, 5.0]).left_out == (4,)


TEN = np.arange(1.0, 11.0)
"""Ten geos whose mean KPI over the pre-period runs 1 to 10."""


def _std_difference_squared(treated: list[int], control: list[int]) -> Fraction:
    def var(a: list[int]) -> Fraction:
        m = Fraction(sum(a), len(a))
        return sum((Fraction(x) - m) ** 2 for x in a) / Fraction(len(a) - 1)

    gap = Fraction(sum(treated), len(treated)) - Fraction(sum(control), len(control))
    return gap**2 / ((var(treated) + var(control)) / 2)


def test_pairs_randomised_within_bound_the_imbalance_top_n_maximises() -> None:
    """Every design of the pairs (1, 2), ..., (9, 10): the difference in means averages to 0 and
    the largest squared standardised difference is 1/10, with all five of one side treated
    (Maxima); the top five geos treated read 10."""
    blocks = pair_geos(TEN)
    assert blocks.blocks == tuple((2 * k, 2 * k + 1) for k in range(5))
    differences, worst = [], Fraction(0)
    for coins in itertools.product((0, 1), repeat=5):
        treated = [2 * k + 1 + c for k, c in enumerate(coins)]
        control = [2 * k + 2 - c for k, c in enumerate(coins)]
        differences.append(Fraction(sum(treated) - sum(control), 5))
        worst = max(worst, _std_difference_squared(treated, control))
    assert sum(differences) == 0
    assert worst == Fraction(1, 10)
    assert _std_difference_squared([6, 7, 8, 9, 10], [1, 2, 3, 4, 5]) == 10

    designs = {assign(blocks, seed) for seed in range(400)}
    assert len(designs) == 32
    for design in designs:
        assert all(len(set(b) & set(design.treated)) == 1 for b in blocks.blocks)
        largest = balance(design, {"mean": TEN}).largest
        assert largest <= math.sqrt(0.1) + 1e-12
    drawn = [
        float(np.mean(TEN[list(d.treated)]) - np.mean(TEN[list(d.control)]))
        for d in (assign(blocks, seed) for seed in range(400))
    ]
    # each draw's difference is (2 heads - 5) / 5 of five fair coins: its variance 1/5
    assert abs(np.mean(drawn)) < 3.0 * math.sqrt(0.2 / 400)


def test_a_seed_draws_its_design_and_another_seed_another() -> None:
    blocks = pair_geos(np.random.default_rng(1).normal(size=20))
    assert assign(blocks, 7) == assign(blocks, 7)
    assert len({assign(blocks, seed) for seed in range(5)}) == 5


def test_a_share_of_each_block_is_treated() -> None:
    blocks = pair_geos(np.arange(12.0), size=4)
    design = assign(blocks, 3, share=0.25)
    assert len(design.treated) == 3
    assert all(len(set(b) & set(design.treated)) == 1 for b in blocks.blocks)
    with pytest.raises(ValueError, match="treats 0 of a block of 4"):
        assign(blocks, 3, share=0.1)


def test_half_a_block_of_odd_size_rounds_up() -> None:
    """At the default share a block of three treats two geos and a block of five three, where
    rounding half to even treats two of five."""
    for size, treated in ((3, 2), (5, 3)):
        blocks = pair_geos(np.arange(float(size)), size=size)
        for seed in range(4):
            design = assign(blocks, seed)
            assert (len(design.treated), len(design.control)) == (treated, size - treated)


def test_one_treated_geo_and_one_candidate_are_matched() -> None:
    design = match_controls([2], [0], {"level": [1.0, 5.0, 3.0]})
    assert design == GeoDesign((2,), (0,))
    with pytest.raises(ValueError, match="past the features' 3 geos"):
        match_controls([2], [3], {"level": [1.0, 5.0, 3.0]})


def test_the_optimal_match_beats_the_greedy_one() -> None:
    x = np.array([0.0, 1.0, 0.9, -2.0])
    design = match_controls([0, 1], [2, 3], {"level": x})
    assert design == GeoDesign((0, 1), (3, 2))
    total = sum(abs(x[t] - x[c]) for t, c in zip(design.treated, design.control, strict=True))
    assert total == pytest.approx(2.1, rel=1e-12, abs=0.0)


def test_the_match_is_the_least_summed_mahalanobis_distance() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(9, 2)) @ np.array([[1.0, 0.0], [0.8, 0.5]])
    treated, candidates = [0, 1, 2], [3, 4, 5, 6, 7, 8]
    centred = np.vstack(
        [x[treated] - x[treated].mean(axis=0), x[candidates] - x[candidates].mean(axis=0)]
    )
    inverse = np.linalg.inv(centred.T @ centred / 7)

    def distance(t: int, c: int) -> float:
        gap = x[t] - x[c]
        return math.sqrt(gap @ inverse @ gap)

    best = min(
        sum(distance(t, c) for t, c in zip(treated, chosen, strict=True))
        for chosen in itertools.permutations(candidates, 3)
    )
    design = match_controls(treated, candidates, {"a": x[:, 0], "b": x[:, 1]})
    got = sum(distance(t, c) for t, c in zip(design.treated, design.control, strict=True))
    assert got == pytest.approx(best, rel=1e-12, abs=0.0)


def test_a_singular_covariance_is_refused_naming_its_features() -> None:
    x = np.arange(6.0)
    with pytest.raises(ValueError, match=r"\['a', 'b'\] is singular"):
        match_controls([0, 1], [2, 3, 4, 5], {"a": x, "b": 2.0 * x + 1.0, "c": x**2})


def test_balance_reads_each_feature_over_the_pooled_sd() -> None:
    design = GeoDesign((0, 1), (2, 3))
    read = balance(design, {"up": [1.0, 3.0, 0.0, 2.0], "down": [0.0, 2.0, 3.0, 5.0]})
    # pooled SD sqrt((2 + 2) / 2): the gaps 1 and -3 over it
    assert read.differences == pytest.approx(
        {"up": 1.0 / math.sqrt(2.0), "down": -3.0 / math.sqrt(2.0)}, rel=1e-12, abs=0.0
    )
    assert read.largest == pytest.approx(3.0 / math.sqrt(2.0), rel=1e-12, abs=0.0)


def test_a_design_needs_a_control() -> None:
    with pytest.raises(ValueError, match="needs a treated and a control geo"):
        GeoDesign((0, 1), ())
    with pytest.raises(ValueError, match="named twice"):
        GeoDesign((0, 1), (1, 2))


GEOS, WEEKS, PRE = 20, 156, 52


def _market(seed: int) -> np.ndarray:
    """Twenty geos of log-normal levels, log-SD 0.5, each week's KPI its level times 5 % noise."""
    rng = np.random.default_rng(seed)
    level = 100.0 * np.exp(rng.normal(0.0, 0.5, GEOS))
    return level[:, None] * np.exp(rng.normal(0.0, 0.05, (GEOS, WEEKS)))


def _half_treated(history: np.ndarray) -> GeoDesign:
    return assign(pair_geos(history[:, :PRE].mean(axis=1)), 5)


def _binomial_se(p: float, reps: int, windows: int) -> float:
    """A rate's binomial standard error over the replications, which re-draw `windows` windows, so
    count at most that many apart."""
    return math.sqrt(max(p * (1.0 - p), 1.0 / 4.0 / min(reps, windows)) / min(reps, windows))


def test_a_four_week_test_detects_a_lift_below_five_percent() -> None:
    history = _market(20261028)
    design = _half_treated(history)
    lifts = [0.0233, 0.035, 0.05]
    read = power(history, design, 4, lifts, pre=PRE, reps=2000, seed=1)
    assert read.windows == WEEKS - PRE - 4 + 1
    se = [_binomial_se(p, 2000, read.windows) for p in read.power]
    assert read.power[0] + 3.0 * se[0] < 0.8
    assert read.power[2] - 3.0 * se[2] > 0.8
    assert 0.035 / 1.5 < read.mde < 0.05


def test_the_test_s_size_over_independent_markets_is_its_alpha() -> None:
    """One window of each of 2000 markets, so the rejections are independent: their rate is the
    test's size, alpha to three binomial standard errors."""
    markets, pre, readout = 2000, 26, 4
    rejected = 0.0
    for m in range(markets):
        rng = np.random.default_rng((20261028, m + 1))
        level = 100.0 * np.exp(rng.normal(0.0, 0.5, GEOS))
        history = level[:, None] * np.exp(rng.normal(0.0, 0.05, (GEOS, pre + readout)))
        design = assign(pair_geos(history[:, :pre].mean(axis=1)), m)
        rejected += power(history, design, readout, [0.01], pre=pre, reps=1, seed=m).size
    assert abs(rejected / markets - ALPHA) < 3.0 * math.sqrt(ALPHA * (1.0 - ALPHA) / markets)


def test_a_persistent_noise_s_size_is_read_by_the_reader_that_reads_the_test() -> None:
    """One window of each of 8000 markets whose log noise is a stationary first-order
    autoregression of 0.8, 5 % its standard deviation, over a 52-week pre-period: the default
    reader rejects a true null at no more than alpha and two binomial standard errors, and
    `log_did`, whose placebo windows read a baseline on both their sides, rejects more often on
    the same markets, by over three standard errors of the paired difference."""
    markets, pre, readout, rho = 8000, 52, 4, 0.8
    default, placebo = np.zeros(markets), np.zeros(markets)
    for m in range(markets):
        rng = np.random.default_rng((20261028, m + 1))
        level = 100.0 * np.exp(rng.normal(0.0, 0.5, GEOS))
        noise = np.empty((GEOS, pre + readout))
        noise[:, 0] = rng.normal(0.0, 0.05, GEOS)
        for t in range(1, pre + readout):
            step = rng.normal(0.0, 0.05 * math.sqrt(1.0 - rho**2), GEOS)
            noise[:, t] = rho * noise[:, t - 1] + step
        history = level[:, None] * np.exp(noise)
        design = assign(pair_geos(history[:, :pre].mean(axis=1)), m)
        default[m] = power(history, design, readout, [0.01], pre=pre, reps=1, seed=m).size
        placebo[m] = power(
            history, design, readout, [0.01], pre=pre, reps=1, seed=m, read=log_did
        ).size
    assert default.mean() < ALPHA + 2.0 * math.sqrt(ALPHA * (1.0 - ALPHA) / markets)
    gap = placebo - default
    assert gap.mean() > 3.0 * gap.std(ddof=1) / math.sqrt(markets)


def test_a_longer_readout_or_pre_period_detects_a_smaller_lift() -> None:
    history = _market(20261028)
    design = _half_treated(history)
    lifts = np.arange(1, 13) / 100.0
    by_readout = [
        power(history, design, r, lifts, pre=PRE, reps=600, seed=2).mde for r in (2, 4, 8)
    ]
    by_pre = [power(history, design, 4, lifts, pre=p, reps=600, seed=2).mde for p in (18, 26, 52)]
    assert by_readout[0] > by_readout[1] > by_readout[2]
    assert by_pre[0] > by_pre[1] > by_pre[2]


def test_a_reader_plugs_in_and_reads_the_window_with_its_lift() -> None:
    history = _market(3)
    design = _half_treated(history)
    seen: list[tuple[np.ndarray, np.ndarray, int]] = []

    def read(treated: np.ndarray, control: np.ndarray, pre: int) -> tuple[float, float]:
        seen.append((treated, control, pre))
        return 1.0, 0.0

    read_ = power(history, design, 3, [0.5], pre=10, reps=4, seed=0, read=read)
    assert (read_.size, read_.power.tolist()) == (1.0, [1.0])
    assert len(seen) == 8
    for (plain, control, pre), (lifted, same, again) in zip(seen[::2], seen[1::2], strict=True):
        assert plain.shape == control.shape == (10, 13)
        assert pre == again == 10
        np.testing.assert_array_equal(control, same)
        np.testing.assert_array_equal(lifted[:, :10], plain[:, :10])
        np.testing.assert_allclose(lifted[:, 10:], 1.5 * plain[:, 10:], rtol=1e-15, atol=0.0)


def test_the_default_reader_reads_a_lift_on_the_log_scale() -> None:
    rng = np.random.default_rng(4)
    control = np.exp(rng.normal(0.0, 0.01, (3, 30)))
    treated = control * 2.0
    treated[:, 26:] *= 1.1
    effect, se = log_did(treated, control, 26)
    assert effect == pytest.approx(math.log(1.1), rel=1e-12, abs=0.0)
    assert se < 1e-12


def test_a_heavy_tailed_placebo_widens_the_default_reader_s_error() -> None:
    """One spike in a 40-week pre-period: the placebo windows' kurtosis is far above 3, and the
    standard error is their root mean square widened to a t at `2 n^2 / ((kappa - 1) sum_{f, f'}
    rho(f - f')^2)` placebo windows, read here by direct sums."""
    pre, readout = 40, 2
    gap = 0.01 * np.cos(np.arange(pre + readout))
    gap[17] = 0.3
    control = np.ones((2, pre + readout))
    _, se = log_did(control * np.exp(gap), control, pre)
    count = pre - readout + 1
    inside = np.array([gap[f : f + readout].mean() for f in range(count)])
    outside = np.array([np.delete(gap[:pre], range(f, f + readout)).mean() for f in range(count)])
    t = inside - outside
    square = float(t @ t)
    kurtosis = count * float(np.sum(t**4)) / square**2
    assert kurtosis > 10.0
    rho = [float(t[: count - d] @ t[d:]) / square for d in range(count)]
    pairs = count + 2.0 * sum((count - d) * rho[d] ** 2 for d in range(1, count))
    nu = 2.0 * count**2 / ((kurtosis - 1.0) * pairs)
    rms = math.sqrt(square / count)
    expected = rms * stats.t.ppf(0.975, nu) / stats.norm.ppf(0.975)
    assert se == pytest.approx(expected, rel=1e-12, abs=0.0)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"readout": 110}, "156 periods holds no test of 110 periods after 52"),
        ({"lifts": [0.1, -1.0]}, "above -1"),
        ({"pre": 17, "readout": 4}, "17 holds 1 placebo windows of 4 after 13 periods, not 2"),
        ({"pre": 4, "readout": 4, "read": log_did}, "holds 1 placebo windows of 4, not 2"),
        ({"design": GeoDesign((0,), (25,))}, "past the history's 20"),
    ],
)
def test_power_refuses(change: dict[str, Any], match: str) -> None:
    history = _market(5)
    arguments: dict[str, Any] = {
        "design": _half_treated(history),
        "readout": 4,
        "lifts": [0.05],
        "pre": PRE,
        "reps": 5,
        "seed": 0,
    } | change
    with pytest.raises(ValueError, match=match):
        power(history, **arguments)


def _detections(history: np.ndarray, design: GeoDesign, readout: int, pre: int, read: Any):
    """Each backtest read once with no lift, its effect, error and treated sum over the readout:
    a lift `x` moves the effect by `x` times that sum and leaves the error, so its detection at
    any lift follows."""
    backtests = history.shape[1] - pre - readout + 1
    rows = []
    for b in range(backtests):
        window = history[:, b : b + pre + readout]
        treated, control = window[list(design.treated)], window[list(design.control)]
        effect, se = read(treated, control, pre)
        rows.append((effect, se, float(treated[:, pre:].sum())))
    return np.array(rows)


@pytest.mark.parametrize("read", [_read_did, _read_tbr], ids=["did", "tbr"])
def test_a_power_audit_reads_every_backtest_once_at_each_lift(read: Any) -> None:
    """Over a history of 78 weeks, a 4-week test after 26: 49 backtests, one a start week. Each
    reading is linear in the lift, so a backtest detects lift `x` where `|e + x S| > z se`, read
    off its one reading with none: the audit's power and size are those shares, its MDE their
    80 % crossing, and its range the same crossing over the first 24 backtests and the last 24.
    Its flag counts the 13 backtests 4 weeks apart, whose readouts do not overlap, and reads the
    exact binomial test of that count against 5 %, twice the smaller tail."""
    history = _market(20261106)[:, :78]
    design = _half_treated(history)
    lifts = np.arange(1, 21) / 200.0
    audit = audit_power(history, design, 4, lifts, pre=26, read=read)
    assert audit.backtests == 49
    rows = _detections(history, design, 4, 26, read)
    scaled = np.concatenate([[0.0], lifts])
    z = float(stats.norm.ppf(0.975))
    detected = np.abs(rows[:, :1] + scaled[None, :] * rows[:, 2:]) > z * rows[:, 1:2]
    np.testing.assert_array_equal(audit.power, detected[:, 1:].mean(axis=0))
    assert audit.size == detected[:, 0].mean()
    rejected = int(detected[::4, 0].sum())
    assert (audit.disjoint, audit.rejected) == (13, rejected)
    pmf = [math.comb(13, k) * 0.05**k * 0.95 ** (13 - k) for k in range(14)]
    tails = sum(pmf[: rejected + 1]), sum(pmf[rejected:])
    assert audit.flag_p_value == pytest.approx(min(1.0, 2.0 * min(tails)), rel=1e-12, abs=0.0)
    assert audit.flagged is (audit.flag_p_value <= 0.01)
    assert audit.mde == _detectable(scaled, detected.mean(axis=0))
    halves = [_detectable(scaled, detected[r].mean(axis=0)) for r in (slice(0, 24), slice(25, 49))]
    assert (audit.mde_low, audit.mde_high) == (min(halves), max(halves))
    assert audit.mde_low <= audit.mde_high


@pytest.mark.parametrize(("rejected", "flagged"), [(5, False), (6, True)])
def test_the_audit_flags_by_the_backtests_whose_readouts_do_not_overlap(
    rejected: int, flagged: bool
) -> None:
    """Over 100 backtests of a 4-week test, the 25 at starts 0, 4, ..., 96 have readouts that do
    not overlap. A reader that rejects in `rejected` of those and in every backtest between them
    reads a size of `(75 + rejected) / 100`, but the flag counts the 25 alone: 6 of 25 is the
    least count flagged (validation/geo_aa_flag.mac, STEP 2), 5 is not."""
    history = _market(7)[:, :125]
    calls: list[int] = []

    def read(treated: np.ndarray, control: np.ndarray, pre: int) -> tuple[float, float]:
        backtest = len(calls) // 2
        calls.append(backtest)
        between = backtest % 4 != 0
        return (3.0 if between or backtest // 4 < rejected else 0.0), 1.0

    audit = audit_power(history, _half_treated(history), 4, [0.5], pre=22, read=read)
    assert (audit.backtests, audit.disjoint, audit.rejected) == (100, 25, rejected)
    assert audit.size == (75 + rejected) / 100
    assert audit.flagged is flagged
    assert calls == [b for b in range(100) for _ in range(2)]


# validation/geo_aa_flag.mac, STEP 2: the least count of m flagged, for m below 104, where no
# count is too low to flag; at 104 a count of 0 is flagged too
_FLAGGED_FROM = {2: 2, 3: 3, 8: 4, 12: 4, 13: 4, 16: 5, 19: 5, 25: 6, 49: 8, 50: 8, 103: 13}


def test_the_flag_s_rejection_region_is_maxima_s() -> None:
    """The binomial test flags a count of `m` exactly where Maxima's exact rationals do: from
    `_FLAGGED_FROM[m]` up below 104, and from 13 up and at 0 at 104; nothing at 1."""
    for m, start in _FLAGGED_FROM.items():
        flagged = [k for k in range(m + 1) if _binomial_p_value(k, m) <= 0.01]
        assert flagged == list(range(start, m + 1))
    assert [k for k in range(105) if _binomial_p_value(k, 104) <= 0.01] == [0, *range(13, 105)]
    assert (_binomial_p_value(0, 1), _binomial_p_value(1, 1)) == pytest.approx((1.0, 0.1), abs=0.0)


@pytest.mark.skipif(shutil.which("maxima") is None, reason="maxima is not on PATH")
def test_maxima_checks_the_flag_s_rejection_region_in_exact_rationals() -> None:
    """`validation/geo_aa_flag.mac`: the flagged counts are runs whose chance under a size of 5 %
    is at most 1 % for every m to 150, the runs' starts the tests pin, nothing at 1, and a flag
    more likely the larger the size, each a PASS."""
    script = Path(__file__).parents[1] / "validation" / "geo_aa_flag.mac"
    out = subprocess.run(
        ["maxima", "--very-quiet", "-b", str(script)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    ).stdout
    assert "Lisp error" not in out
    assert "(FAIL)" not in out
    # the verdict's own definition is echoed with its "(PASS)" in quotes
    assert out.count("(PASS)") - out.count('"(PASS)"') == 4


def test_a_reader_of_size_5_percent_is_flagged_at_most_1_percent_of_the_time() -> None:
    """Over 4000 null audits of four geos of independent unit noise, a 4-week test after 12 over
    64 weeks (49 backtests, 13 whose readouts do not overlap), read by a reader whose effect is
    the readout's summed gap and its error that gap's known one: its size is exactly 5 % and its
    verdicts over disjoint readouts are independent. The flag fires within 3 binomial standard
    errors of its exact chance at 13, 0.0031 (validation/geo_aa_flag.mac, STEP 4). The same test
    over the 49 backtests one week apart, which share their readouts' weeks, fires more than 1 %
    of the time by over 3 standard errors (0.028 measured): the reason the flag counts the 13."""
    design = GeoDesign((0, 1), (2, 3))

    def read(treated: np.ndarray, control: np.ndarray, pre: int) -> tuple[float, float]:
        return float(treated[:, pre:].sum() - control[:, pre:].sum()), 4.0

    draws, flagged, overlapping = 4000, 0, 0
    for seed in range(draws):
        history = 1000.0 + np.random.default_rng((seed, 7)).normal(size=(4, 64))
        audit = audit_power(history, design, 4, [0.5], pre=12, read=read)
        assert (audit.backtests, audit.disjoint) == (49, 13)
        flagged += audit.flagged
        overlapping += _binomial_p_value(round(audit.size * 49), 49) <= 0.01
    exact = 0.0031029961680991212
    assert abs(flagged / draws - exact) <= 3.0 * math.sqrt(exact * (1.0 - exact) / draws)
    assert overlapping / draws - 3.0 * math.sqrt(0.01 * 0.99 / draws) > 0.01


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (
            {"pre": 11},
            "a pre-period of 11 periods is shorter than three times the test's 4: it needs 12",
        ),
        (
            {"readout": 8, "pre": 22},
            "a pre-period of 22 periods is shorter than three times the test's 8",
        ),
        (
            {"history": 40},
            "a history of 40 periods holds 2 backtests of a test of 4 periods after 30 whose "
            "readouts do not overlap, fewer than 8: it needs 62 periods",
        ),
        (
            {"history": 61},
            "a history of 61 periods holds 7 backtests of a test of 4 periods after 30 whose",
        ),
        ({"lifts": [0.1, -1.0]}, "above -1"),
        ({"design": GeoDesign((0,), (25,))}, "past the history's 20"),
    ],
)
def test_a_power_audit_refuses_a_short_pre_period_or_few_backtests(
    change: dict[str, Any], match: str
) -> None:
    """A pre-period shorter than three times the test is refused, naming both, even where the
    reader would read it (`read_did` reads 22 weeks before an 8-week test); and so is a history of
    fewer than 8 backtests whose readouts do not overlap, naming their count and the history it
    needs (61 weeks hold 28 backtests, 7 of them apart; 62 hold 8)."""
    history = _market(5)[:, : change.get("history", WEEKS)]
    arguments: dict[str, Any] = {
        "design": _half_treated(history),
        "readout": 4,
        "lifts": [0.05],
        "pre": 30,
    } | {k: v for k, v in change.items() if k != "history"}
    with pytest.raises(ValueError, match=match):
        audit_power(history, **arguments)
    if change == {"readout": 8, "pre": 22}:
        power(history, _half_treated(history), 8, [0.05], pre=22, reps=2, seed=0)


def _alike(seed: int) -> np.ndarray:
    """Six geos in three alike pairs, 0 and 3, 1 and 4, 2 and 5: a pair shares its level, its
    noise's size and its trend, and the pairs differ in all three. A season every geo shares gives
    the TBR fit a signal to read: with none, the controls predict nothing and which candidates
    pass the A/A test is the noise's draw (on seed 1, none of the four at or above the median R^2
    passed it). With the season, 79 of 80 draws (seeds 1-40, six and five geos) keep one."""
    rng = np.random.default_rng(seed)
    t = np.arange(40.0)
    level = np.array([100.0, 400.0, 1600.0])
    noise = np.array([0.02, 0.1, 0.05])
    trend = np.array([0.0, 0.004, -0.003])
    season = 1.0 + 0.2 * np.sin(2.0 * np.pi * t / 13.0)
    shapes = level[:, None] * (1.0 + trend[:, None] * t) * season
    pairs = np.vstack([shapes, shapes])
    return pairs * np.exp(rng.normal(0.0, 1.0, (6, 40)) * np.tile(noise, 2)[:, None])


def test_strata_join_the_geos_alike_in_mean_cv_and_slope() -> None:
    """The geos of each alike pair share a stratum, each with one geo treated, whatever the
    seed; a seed draws its design and the same seed draws it again."""
    chosen = stratified_design(_alike(1), 4, pre=36, seed=3)
    assert sorted(chosen.strata.blocks) == [(0, 3), (1, 4), (2, 5)]
    assert chosen.strata.left_out == ()
    assert chosen.candidates == 8
    assert 1 <= chosen.kept <= chosen.candidates
    for treated in (0, 1, 2):
        assert (treated in chosen.design.treated) != (treated + 3 in chosen.design.treated)
    assert stratified_design(_alike(1), 4, pre=36, seed=3) == chosen
    odd = stratified_design(_alike(1)[:5], 4, pre=36, seed=3)
    assert odd.strata.left_out == (2,)
    assert len(odd.design.treated) == len(odd.design.control) == 2


def test_the_comparator_chooses_the_least_mde_among_the_candidates_it_keeps() -> None:
    """Every distinct candidate drawn is scored; the ones at or above the median out-of-sample
    R^2 that pass the A/A test are kept, and the chosen one has their least MDE."""
    history = _market(20261106)[:, :60]
    chosen = stratified_design(history, 4, pre=26, seed=11, candidates=50)
    series = history[:, -26:]
    drawn = {}
    for child in np.random.SeedSequence(11).spawn(50):
        design = assign(chosen.strata, int(child.generate_state(1)[0]))
        drawn.setdefault(design.treated, design)
    scores = [_tbr_scores(series, d, 4) for d in drawn.values()]
    floor = float(np.median([r2 for r2, _, _ in scores]))
    kept = [mde for r2, passes, mde in scores if r2 >= floor and passes]
    assert (chosen.candidates, chosen.kept) == (len(drawn), len(kept))
    assert chosen.mde == min(kept)
    assert chosen.r2 >= floor


def test_the_comparator_s_aa_test_is_kerman_s_predictive_error() -> None:
    """The A/A test on the pre-period's last `readout` periods reads `read_tbr`'s own lift and its
    error under independent errors, fitted on the periods before them."""
    history = _market(4)[:, :30]
    design = _half_treated(history)
    rng = np.random.default_rng(0)
    for readout in (2, 4):
        series = history * np.exp(rng.normal(0.0, 0.05, history.shape))
        _, passes, _ = _tbr_scores(series, design, readout)
        geos = tuple(Geo(f"g{g}") for g in range(GEOS))
        dates = np.datetime64("2025-01-06") + np.timedelta64(7, "D") * np.arange(30)
        panel = MediaPanel(
            dates=dates,
            cadence=Cadence.WEEK,
            geos=geos,
            channels=(Channel("tv"),),
            controls=(),
            kpi=series,
            kpi_unit=Revenue(),
            spend=np.zeros((GEOS, 30, 1)),
            currency="EUR",
            control_values=np.zeros((GEOS, 30, 0)),
            filled=np.zeros((GEOS, 30), dtype=bool),
        )
        test = GeoTest(
            Channel("tv"),
            tuple(geos[g] for g in design.treated),
            tuple(geos[g] for g in design.control),
            dates[30 - readout],
            dates[-1],
            np.ones(30),
            np.r_[np.ones(30 - readout), np.zeros(readout)],
            readout,
        )
        _, reading = read_tbr(panel, test, variance="iid")
        t = float(stats.t.ppf(0.975, 30 - readout - 2))
        assert passes == (abs(reading.lift) <= t * reading.se)
        assert reading.df == 30 - readout - 2


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (
            {"pre": 11},
            "a pre-period of 11 periods is shorter than three times the test's 4: it needs 12",
        ),
        ({"pre": 61}, "a pre-period of 61 periods is longer than the history's 60"),
        ({"size": 21}, "20 geos hold no stratum of 21"),
        ({"share": 1.0}, "leaves no contrast"),
        ({"candidates": 0}, "candidates must be a whole number"),
    ],
)
def test_the_comparator_refuses_a_short_pre_period(change: dict[str, Any], match: str) -> None:
    arguments: dict[str, Any] = {"readout": 4, "pre": 26, "seed": 0} | change
    with pytest.raises(ValueError, match=match):
        stratified_design(_market(5)[:, :60], **arguments)


def test_the_comparator_refuses_where_no_candidate_passes_its_aa_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mixdyn.experiments import geo_design

    monkeypatch.setattr(
        geo_design, "_tbr_scores", lambda series, design, readout: (1.0, False, 0.1)
    )
    with pytest.raises(ValueError, match="none of the 8 candidate designs passes the A/A test"):
        stratified_design(_alike(1), 4, pre=36, seed=3)
