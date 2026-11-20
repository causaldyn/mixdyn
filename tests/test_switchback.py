"""A switchback test: its design randomised over blocks of periods, the washout after a switch
and the burn-in that replaces it, its reading by re-randomisation, by a Newey-West error or by
the half-panel jackknife into a `LiftEstimate`, and the two-epoch test of its carryover's order."""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest
from scipy import stats

from mixdyn.experiments.switchback import (
    _DRAWN,
    CarryoverOrder,
    SwitchbackDesign,
    _arrangements,
    _randomization,
    _reach,
    _rerandomised,
    carryover_order,
    read_switchback,
    switchback_design,
)

DAYS = np.datetime64("2025-01-06") + np.arange(48)

# validation/switchback_carryover.mac, STEP 3: the no-washout difference's bias over every
# arrangement of 12 blocks of 4 periods, a share of the effect a period, at retention 1/2 and 4/5
MAXIMA_BIAS = {
    Fraction(1, 2): Fraction(-12572548959742635, 49539595901075456),
    Fraction(4, 5): Fraction(
        -99496852002717160382188814954103237, 156319401867222040891647338867187500
    ),
}


def _world(
    seed: int,
    rho: float,
    retention: float,
    washout: int,
    effect: float,
    *,
    blocks: int = 12,
    block: int = 4,
) -> tuple[np.ndarray, np.ndarray, SwitchbackDesign]:
    """A national KPI over a switchback test drawn at `seed`: a level of 10, a stationary AR(1)
    noise of variance 1 at `rho`, and `effect` times the channel's adstock at `retention`, which
    reads `effect` a period once the channel has been on for good; the channel is off before the
    test and spends 100 a period on."""
    periods = blocks * block
    design = switchback_design(DAYS[0] + np.arange(periods), block, washout, seed=seed)
    u = np.random.default_rng((seed, 56)).normal(size=periods)
    noise = np.empty(periods)
    noise[0] = u[0]
    for t in range(1, periods):
        noise[t] = rho * noise[t - 1] + math.sqrt(1.0 - rho * rho) * u[t]
    adstock = np.empty(periods)
    level = 0.0
    for t, on in enumerate(design.assignment):
        level = retention * level + (1.0 - retention) * on
        adstock[t] = level
    return 10.0 + effect * adstock + noise, 100.0 * design.assignment, design


def _rate(
    cell: tuple[float, float, int, float],
    method: str,
    panels: int,
    *,
    blocks: int = 12,
    bandwidth: str = "newey-west",
) -> float:
    """The share of `panels` worlds of `cell` whose interval misses the lift the channel made."""
    misses = 0
    for seed in range(panels):
        kpi, spend, design = _world(seed, *cell, blocks=blocks)
        _, reading = read_switchback(kpi, spend, design, method=method, bandwidth=bandwidth)  # ty: ignore[invalid-argument-type]
        truth = cell[3] * design.block * design.on_blocks
        misses += not reading.lower <= truth <= reading.upper
    return misses / panels


@pytest.mark.parametrize(
    ("periods", "block", "washout", "share", "match"),
    [
        (DAYS, 4, 4, 0.5, "leaves nothing"),
        (DAYS, 4, 5, 0.5, "leaves nothing"),
        (DAYS[:12], 4, 0, 0.5, "at least 4 blocks"),
        (DAYS[:46], 4, 0, 0.5, "whole blocks"),
        (DAYS, 4, 0, 0.0, "share"),
        (DAYS, 4, 0, 1.0, "share"),
        (DAYS, 4, 0, math.nan, "share"),
        (DAYS[:16], 4, 0, 0.1, "no contrast"),
        (DAYS[:16], 4, 0, 0.9, "no contrast"),
        (np.arange(48), 4, 0, 0.5, "dates"),
        (DAYS[::-1], 4, 0, 0.5, "increasing"),
        (DAYS, 0, 0, 0.5, "block"),
        (DAYS, 4, -1, 0.5, "washout"),
    ],
)
def test_a_design_refuses_what_reads_nothing(
    periods: np.ndarray, block: int, washout: int, share: float, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        switchback_design(periods, block, washout, seed=0, share=share)


def test_a_design_refuses_a_negative_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        switchback_design(DAYS, 4, 0, seed=-1)


@pytest.mark.parametrize(
    ("blocks", "share", "on"), [(12, 0.5, 6), (5, 0.5, 3), (4, 0.375, 2), (5, 0.3, 2), (7, 0.2, 1)]
)
def test_a_design_turns_on_the_share_of_its_blocks_rounded_half_up(
    blocks: int, share: float, on: int
) -> None:
    for seed in range(20):
        design = switchback_design(DAYS[: blocks * 3], 3, 1, seed=seed, share=share)
        assert design.on_blocks == on
        assert design.assignment.sum() == 3 * on
        assert design.blocks == blocks


def test_a_design_draws_its_order_from_the_seeds_child_stream() -> None:
    for seed in (0, 1, 56, 2**40):
        rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
        expected = np.zeros(12, dtype=bool)
        expected[rng.permutation(12)[:6]] = True
        design = switchback_design(DAYS, 4, 1, seed=seed)
        assert design.on == tuple(expected.tolist())
        assert switchback_design(DAYS, 4, 1, seed=seed).on == design.on
    assert len({switchback_design(DAYS, 4, 1, seed=s).on for s in range(50)}) > 40


def test_a_design_keeps_its_dates_as_days_and_its_washout_after_each_switch() -> None:
    weeks = np.datetime64("2025-01-06") + np.timedelta64(7, "D") * np.arange(8)
    design = SwitchbackDesign(weeks, 2, 1, (True, True, False, True), 0.75, 0)
    assert design.dates.dtype == np.dtype("datetime64[D]")
    assert not design.dates.flags.writeable
    # the first block is read as after a switch; the second follows a block on and keeps all
    assert design.kept.tolist() == [False, True, True, True, False, True, False, True]
    assert design.assignment.tolist() == [True, True, True, True, False, False, True, True]
    with pytest.raises(ValueError, match="flag each"):
        SwitchbackDesign(weeks, 2, 1, (True, False, True), 0.5, 0)
    with pytest.raises(ValueError, match="no contrast"):
        SwitchbackDesign(weeks, 2, 1, (True,) * 4, 0.5, 0)


@pytest.mark.parametrize(
    ("share", "on", "refused"),
    [
        *((share, on, True) for share, on in ((0.5, 3), (0.5, 1), (0.7, 2), (0.3, 2))),
        *((share, on, False) for share, on in ((0.375, 2), (0.625, 3), (0.25, 1), (0.75, 3))),
    ],
)
def test_a_design_turns_on_the_share_it_names(share: float, on: int, refused: bool) -> None:
    """A design written by hand names the share it turns on: `round_half_up(share * blocks)` of
    its blocks are, as a drawn design's are, or it is refused."""
    weeks = np.datetime64("2025-01-06") + np.timedelta64(7, "D") * np.arange(8)
    flags = (True,) * on + (False,) * (4 - on)
    if refused:
        with pytest.raises(ValueError, match=f"turns {math.floor(share * 4 + 0.5)} of 4 blocks"):
            SwitchbackDesign(weeks, 2, 1, flags, share, 0)
    else:
        assert SwitchbackDesign(weeks, 2, 1, flags, share, 0).on_blocks == on


def test_the_randomization_p_value_is_the_one_counted_by_hand() -> None:
    """Four blocks of one period, the KPI 4, 0, 3, 1, the first and third on: the differences of
    the six arrangements are 0, 3, 1, -1, -3 and 0, two as far from 0 as the observed 3. Each
    arrangement's difference less `tau` times its flags' difference, 0, 1, 0, 0, -1, 0, is as far
    from 0 as `3 - tau` for four of six at `tau` in (2, 4) and for two beyond: at a level of 0.5
    the interval of the effect a period is [2, 4]."""
    design = SwitchbackDesign(DAYS[:4], 1, 0, (True, False, True, False), 0.5, 0)
    y = np.array([4.0, 0.0, 3.0, 1.0])
    p_value, spread, reaches, count, enumerated = _randomization(y, design, 3.0, (0.5, 0.05))
    assert p_value == pytest.approx(2 / 6, rel=0.0, abs=1e-15)
    assert spread == pytest.approx(math.sqrt(20 / 6), rel=1e-14, abs=0.0)
    assert reaches[0] == pytest.approx((1.0, 1.0), rel=1e-12, abs=0.0)
    assert reaches[1] == (math.inf, math.inf)
    assert (count, enumerated) == (6, True)
    with pytest.raises(ValueError, match="too few"):
        read_switchback(y, np.array([1.0, 0.0, 1.0, 0.0]), design)


def test_the_randomization_reading_matches_an_independent_enumeration() -> None:
    """A design of 8 blocks with a washout, every arrangement read by a loop over the periods:
    the p-value, and the interval's ends where the test of a constant effect turns."""
    kpi, spend, design = _world(3, 0.5, 0.5, 1, 1.0, blocks=8)
    _, reading = read_switchback(kpi, spend, design)

    def difference(on: tuple[bool, ...], y: np.ndarray) -> float:
        up, down = [], []
        for b, state in enumerate(on):
            switched = b == 0 or on[b - 1] != state
            for j in range(design.block):
                if switched and j < design.washout:
                    continue
                (up if state else down).append(y[b * design.block + j])
        return float(np.mean(up) - np.mean(down))

    arrangements = [
        tuple(b in chosen for b in range(8)) for chosen in itertools.combinations(range(8), 4)
    ]
    flags = design.assignment.astype(float)

    def p(tau: float) -> float:
        y = kpi - tau * flags
        observed = abs(difference(design.on, y))
        return float(np.mean([abs(difference(z, y)) >= observed - 1e-12 for z in arrangements]))

    assert reading.arrangements == 70
    assert reading.enumerated
    assert reading.p_value == pytest.approx(p(0.0), rel=0.0, abs=1e-15)
    count = design.block * design.on_blocks
    assert reading.lift == pytest.approx(count * difference(design.on, kpi), rel=1e-12, abs=0.0)
    for end in (reading.lower / count, reading.upper / count):
        inside = end + (1e-7 if end == reading.lower / count else -1e-7)
        outside = end - (1e-7 if end == reading.lower / count else -1e-7)
        assert p(inside) > 0.05
        assert p(outside) <= 0.05


def test_a_reading_sums_geos_and_counts_every_period_on() -> None:
    """The lift is the difference of the kept periods' means times every period on, washout
    included, and a KPI by geo reads as its sum."""
    design = SwitchbackDesign(DAYS[:8], 2, 1, (True, False, False, True), 0.5, 0)
    kpi = np.array([[5.0, 6.0, 1.0, 2.0, 3.0, 1.0, 9.0, 7.0], [1.0, 1.0, 0.0, 1.0, 0, 0, 1.0, 2.0]])
    spend = np.array([[8.0, 8.0, 1.0, 1.0, 1.0, 1.0, 8.0, 8.0]] * 2)
    kept_on, kept_off = [7.0, 9.0], [3.0, 3.0, 1.0]  # periods 1, 7 and 3, 4, 5
    estimate, reading = read_switchback(kpi, spend, design, method="hac")
    assert reading.lift == pytest.approx(4 * (np.mean(kept_on) - np.mean(kept_off)), abs=0.0)
    alone = read_switchback(kpi.sum(axis=0), spend.sum(axis=0), design, method="hac")[1]
    assert repr(alone) == repr(reading)
    assert math.isnan(reading.carryover)  # no block after a switch keeps two periods
    assert estimate.readout == 8
    assert estimate.geos is None
    assert estimate.end == DAYS[7]
    np.testing.assert_array_equal(estimate.spend, spend.sum(axis=0))
    np.testing.assert_array_equal(estimate.planned, np.full(8, 2.0))
    national, spend, design = _world(2, 0.5, 0.5, 1, 1.0)
    shares = np.random.default_rng(2).dirichlet(np.ones(3))
    _, alone = read_switchback(national, spend, design)
    _, summed = read_switchback(np.outer(shares, national), np.outer(shares, spend), design)
    assert summed.p_value == alone.p_value
    assert summed.lower == pytest.approx(alone.lower, rel=1e-9, abs=0.0)


ON = np.repeat(switchback_design(DAYS, 4, 1, seed=0).on, 4)


@pytest.mark.parametrize(
    ("kpi", "spend", "kwargs", "match"),
    [
        (np.ones(48), 100.0 * ON, {}, "same in every kept period"),
        (np.arange(47.0), 100.0 * ON, {}, "48 periods"),
        (np.arange(48.0), 100.0 * ON[:47], {}, "48 periods"),
        (np.r_[np.nan, np.arange(47.0)], 100.0 * ON, {}, "finite"),
        (np.arange(48.0), 100.0 * ON, {"level": 1.0}, "level"),
        (np.arange(48.0), 100.0 * ON, {"method": "ols"}, "method"),
        (np.arange(48.0), 100.0 * ON, {"method": "hac", "bandwidth": 1}, "bandwidth"),
        (np.arange(48.0), np.full(48, 5.0), {}, "changed no spend"),
        (5.0 + ON, 100.0 * ON, {"method": "hac"}, "blocks' noise"),
    ],
)
def test_a_reading_refuses_what_it_cannot_read(
    kpi: np.ndarray, spend: np.ndarray, kwargs: dict[str, object], match: str
) -> None:
    design = switchback_design(DAYS, 4, 1, seed=0)
    with pytest.raises(ValueError, match=match):
        read_switchback(kpi, spend, design, **kwargs)  # ty: ignore[invalid-argument-type]


def test_the_estimates_error_holds_the_readings_95_percent_interval() -> None:
    """The normal 95 % interval the likelihood reads is the reading's t interval for "hac", as
    the geo readers widen theirs, and holds the randomization interval, one end on it."""
    kpi, spend, design = _world(7, 0.5, 0.0, 1, 1.0)
    z = float(stats.norm.ppf(0.975))
    estimate, reading = read_switchback(kpi, spend, design, method="hac")
    assert reading.df == 10.0
    t = float(stats.t.ppf(0.975, 10.0))
    assert estimate.se == pytest.approx(reading.se * t / z, rel=1e-14, abs=0.0)
    assert estimate.lift + z * estimate.se == pytest.approx(reading.upper, rel=1e-14, abs=0.0)
    estimate, reading = read_switchback(kpi, spend, design)
    reach = max(reading.lift - reading.lower, reading.upper - reading.lift)
    assert estimate.se == pytest.approx(reach / z, rel=1e-14, abs=0.0)
    assert estimate.lift - z * estimate.se <= reading.lower + 1e-9
    assert estimate.lift + z * estimate.se >= reading.upper - 1e-9
    assert estimate.se > reading.se  # the interval is wider than its spread's normal one
    _, narrow = read_switchback(kpi, spend, design, level=0.8)
    estimate_80, _ = read_switchback(kpi, spend, design, level=0.8)
    assert narrow.upper - narrow.lower < reading.upper - reading.lower
    assert estimate_80.se == estimate.se  # the estimate reads 95 % whatever the reading's level


def test_the_verdict_is_keyed_on_the_intervals_sign() -> None:
    """Keyed on the interval's sign, which agrees with the p-value against 0 for both methods: an
    arrangement's slope in the effect lies in [-1, 1], so the effects the randomization test
    accepts have no hole (ADR 0056)."""
    for seed in range(30):
        kpi, spend, design = _world(seed, 0.0, 0.0, 0, 0.6)
        for method in ("randomization", "hac"):
            _, r = read_switchback(kpi, spend, design, method=method)
            expected = "up" if r.lower > 0 else "down" if r.upper < 0 else "inconclusive"
            assert r.verdict == expected
            assert (r.p_value <= 0.05) == (r.verdict != "inconclusive")


def test_the_verdict_follows_the_interval_where_the_p_value_holds_a_near_tie() -> None:
    """Eight blocks of one period, the KPI whole numbers moved by a few 1e-13: one arrangement
    falls short of the observed difference's distance from 0 by about 1e-13, inside the
    p-value's tie, a relative 1e-12, and far past rounding. The p-value counts it as far, 4 of 70
    arrangements, 0.057; the interval reads every effect without that tie and ends 5e-12 below 0,
    a thousand times rounding's reach on a lift of 12. The verdict is the interval's, down. The
    KPI's whole numbers alone put the end on 0 itself, where rounding would set the verdict."""
    on = (False, True, True, True, False, True, False, False)
    design = SwitchbackDesign(DAYS[:8], 1, 0, on, 0.5, 0)
    whole = np.array([5.0, 3.0, 2.0, 1.0, 5.0, 0.0, 5.0, 3.0])
    kpi = whole + 2e-13 * np.array([2.0, -3.0, -3.0, 1.0, -2.0, 0.0, 1.0, 3.0])
    _, reading = read_switchback(kpi, 100.0 * design.assignment, design)
    assert reading.arrangements == 70
    assert reading.p_value == pytest.approx(4 / 70, rel=1e-12, abs=0.0)
    assert -1e-11 < reading.upper < -1e-12
    assert reading.verdict == "down"
    _, tied = read_switchback(whole, 100.0 * design.assignment, design)
    assert abs(tied.upper) <= 1e-14


def test_the_carryover_check_reads_no_block_before_the_first() -> None:
    """Four blocks of 3 periods, on, off, on, off, no washout: the check reads blocks 1 to 3, each
    after a switch, and not block 0, which no block of the test precedes, though its state is not
    the last block's. Their first periods less their others' mean are -1 (block 2, on) and 2 and 0
    (blocks 1 and 3, off), so -1 - 1 = -2; block 0 would add its 6 to the blocks on."""
    design = SwitchbackDesign(DAYS[:12], 3, 0, (True, False, True, False), 0.5, 0)
    assert design.on[0] != design.on[-1]
    kpi = np.array([9.0, 3.0, 3.0, 4.0, 2.0, 2.0, 5.0, 6.0, 6.0, 1.0, 1.0, 1.0])
    spend = np.array([1.0, 0.0, 1.0, 0.0]).repeat(3)
    _, reading = read_switchback(kpi, spend, design, method="hac")
    assert reading.carryover == pytest.approx(-2.0, rel=0.0, abs=1e-14)


def test_the_carryover_check_reads_a_washout_too_short() -> None:
    """Without noise within a block, a channel without carryover leaves the check at 0, and one of
    retention 0.8 below 0, the less so the longer the washout."""
    checks = {}
    for retention, washout in itertools.product((0.0, 0.8), (0, 1, 2)):
        design = switchback_design(DAYS, 4, washout, seed=5)
        adstock = np.empty(design.periods)
        level = 0.0
        for t, on in enumerate(design.assignment):
            level = retention * level + (1.0 - retention) * on
            adstock[t] = level
        # a level a block moves no block's first period against its others
        levels = np.repeat(np.random.default_rng(5).normal(size=design.blocks), design.block)
        kpi = 10.0 + adstock + levels
        _, reading = read_switchback(kpi, 100.0 * design.assignment, design)
        checks[retention, washout] = reading.carryover
    for washout in (0, 1, 2):
        assert checks[0.0, washout] == pytest.approx(0.0, abs=1e-12)
    assert checks[0.8, 0] < checks[0.8, 1] < checks[0.8, 2] < 0.0


def _closed(retention: Fraction, blocks: int, block: int) -> Fraction:
    """The no-washout difference's bias over every arrangement, a share of the effect a period:
    `-(G / L) (1 + 1/B - r (1 - r^(B-1)) / (B (B-1) (1 - r)))`, `G = lam (1 - lam^L) / (1 -
    lam)` and `r = lam^L`; see validation/switchback_carryover.mac."""
    g = retention * (1 - retention**block) / (1 - retention)
    r = retention**block
    tail = r * (1 - r ** (blocks - 1)) / (blocks * (blocks - 1) * (1 - r))
    return -(g / block) * (1 + Fraction(1, blocks) - tail)


@pytest.mark.parametrize("retention", [Fraction(1, 2), Fraction(4, 5)])
def test_the_no_washout_bias_is_its_closed_form(retention: Fraction) -> None:
    """The closed form is Maxima's, exactly, and the simulation's mean difference of 2000 worlds
    with no washout lies within 3 Monte Carlo standard errors of it."""
    assert _closed(retention, 12, 4) == MAXIMA_BIAS[retention]
    shares = []
    for seed in range(2000):
        kpi, spend, design = _world(seed, 0.0, float(retention), 0, 1.0)
        _, reading = read_switchback(kpi, spend, design, method="hac")
        shares.append(reading.lift / (design.block * design.on_blocks) - 1.0)
    mean, se = np.mean(shares), np.std(shares, ddof=1) / math.sqrt(len(shares))
    assert abs(mean - float(MAXIMA_BIAS[retention])) < 3.0 * se
    assert mean < -10.0 * se  # the bias is far from 0 at this design


def test_the_closed_form_is_0_without_carryover() -> None:
    assert _closed(Fraction(0), 12, 4) == 0
    assert _closed(Fraction(4, 5), 12, 4) < _closed(Fraction(1, 2), 12, 4) < 0


# the binomial standard error of a 5 % rate over 2000 worlds
SIZE_SE = math.sqrt(0.05 * 0.95 / 2000)


@pytest.mark.parametrize(
    "cell",
    [
        pytest.param((0.0, 0.0, 0, 0.0), id="no-lift-white-noise"),
        pytest.param((0.8, 0.5, 2, 1.0), id="ar-0.8-carryover-0.5-washout-2"),
    ],
)
def test_the_randomization_reading_holds_its_size(cell: tuple[float, float, int, float]) -> None:
    """Over 2000 worlds, the 95 % randomization interval misses the lift the channel made within
    3 binomial standard errors of 5 %: with no lift and white noise, and with an AR(1) noise of
    0.8 and a carryover of retention 0.5 that a washout of 2 periods takes out (ADR 0056)."""
    assert abs(_rate(cell, "randomization", 2000) - 0.05) <= 3.0 * SIZE_SE


def test_the_newey_west_reading_over_rejects_with_few_blocks_and_persistent_noise() -> None:
    """Over the same 2000 worlds at an AR(1) of 0.95, no lift and a washout of 2, the Newey-West
    interval over 12 blocks misses 0 far more than 5 % of the time: why it is not the default."""
    assert _rate((0.95, 0.0, 2, 0.0), "hac", 2000) > 0.05 + 6.0 * SIZE_SE


def test_the_lag_0_bandwidth_holds_its_size_over_6_blocks() -> None:
    """Over 4000 worlds of 6 blocks of 4, no lift and white noise, the "hac" interval at lag 0, a
    pooled two-sample t on the block means, misses 0 within 3 binomial standard errors of 5 %;
    Newey and West's kernel, whose lag reads the centred scores' negative cross products, misses
    it far more (ADR 0056)."""
    se = math.sqrt(0.05 * 0.95 / 4000)
    cell = (0.0, 0.0, 0, 0.0)
    assert abs(_rate(cell, "hac", 4000, blocks=6, bandwidth="lag-0") - 0.05) <= 3.0 * se
    assert _rate(cell, "hac", 4000, blocks=6) > 0.05 + 6.0 * se


@pytest.mark.parametrize(("blocks", "washout"), [(4, 0), (5, 1), (12, 1)])
def test_the_lag_0_error_is_the_pooled_two_sample_error_of_the_block_means(
    blocks: int, washout: int
) -> None:
    """At lag 0 the error is the block means' pooled spread about their state's mean, each block
    weighted by its kept periods, over `blocks - 2` degrees of freedom, times `1 / N_on + 1 /
    N_off`; at half the blocks on and every block keeping as many periods, Newey and West's lag-0
    term times `blocks / (blocks - 2)` is the same."""
    kpi, spend, design = _world(13, 0.5, 0.0, washout, 0.0, blocks=blocks)
    kept = design.kept.reshape(blocks, design.block)
    y = kpi.reshape(blocks, design.block)
    on = np.array(design.on)
    sizes = kept.sum(axis=1)
    means = np.array([y[b][kept[b]].mean() for b in range(blocks)])
    centre = {s: y[on == s][kept[on == s]].mean() for s in (True, False)}
    pooled = sum(sizes[b] * (means[b] - centre[bool(on[b])]) ** 2 for b in range(blocks))
    variance = pooled / (blocks - 2) * (1 / sizes[on].sum() + 1 / sizes[~on].sum())
    _, reading = read_switchback(kpi, spend, design, method="hac", bandwidth="lag-0")
    count = design.block * design.on_blocks
    assert reading.se == pytest.approx(count * math.sqrt(variance), rel=1e-12, abs=0.0)
    assert reading.df == blocks - 2
    t = float(stats.t.ppf(0.975, blocks - 2))
    assert reading.upper - reading.lift == pytest.approx(t * reading.se, rel=1e-12, abs=0.0)
    if washout == 0 and blocks % 2 == 0:
        default = read_switchback(kpi, spend, design, method="hac")[1]
        with_lag_0 = math.sqrt(
            sum(((means[b] - centre[bool(on[b])]) / (blocks // 2)) ** 2 for b in range(blocks))
            * blocks
            / (blocks - 2)
        )
        assert reading.se == pytest.approx(count * with_lag_0, rel=1e-12, abs=0.0)
        assert default.se != reading.se  # the default's lag reads more than lag 0


def test_the_randomization_reading_samples_its_arrangements_past_100_000() -> None:
    """20 blocks of 4 with half on have 184 756 arrangements, more than are enumerated: the
    reading reads the observed one, first, and 10 000 drawn from the seed's second child stream,
    each with as many blocks on, the same draws for the same seed."""
    periods = np.datetime64("2025-01-06") + np.arange(80)
    design = switchback_design(periods, 4, 1, seed=8)
    assert math.comb(20, 10) == 184_756
    rows, enumerated = _arrangements(design)
    assert not enumerated
    assert rows.shape == (10_001, 20) == (_DRAWN + 1, 20)
    assert tuple(rows[0].tolist()) == design.on
    assert (rows.sum(axis=1) == 10).all()
    rng = np.random.default_rng(np.random.SeedSequence(8).spawn(2)[1])
    order = np.argsort(rng.random((10_000, 20)), axis=1)
    expected = np.zeros((10_000, 20), dtype=bool)
    np.put_along_axis(expected, order[:, :10], values=True, axis=1)
    np.testing.assert_array_equal(rows[1:], expected)
    u = np.random.default_rng(8).normal(size=80)
    kpi, spend = 10.0 + u + 0.5 * design.assignment, 100.0 * design.assignment
    _, reading = read_switchback(kpi, spend, design)
    assert (reading.arrangements, reading.enumerated) == (10_001, False)
    assert repr(read_switchback(kpi, spend, design)[1]) == repr(reading)
    other = switchback_design(periods, 4, 1, seed=9)
    assert not np.array_equal(_arrangements(other)[0][1:], rows[1:])
    differences, _ = _rerandomised(kpi, rows, design)
    observed = float(differences[0])
    assert reading.lift == pytest.approx(40 * observed, rel=1e-12, abs=0.0)
    assert reading.p_value == np.mean(np.abs(differences) >= abs(observed) * (1.0 - 1e-12))


def test_the_test_rejects_once_at_most_alpha_of_the_arrangements_are_as_far() -> None:
    """Twenty arrangements whose slope is 0, at distances 1 to 20 from 0: arrangement `z` is as
    far as the observed difference less `u` while `u <= z`, so past `u = 18` two of twenty are,
    a share of 0.1, and the test at 0.1 rejects there, not at 19 where one is."""
    a, c = np.arange(1.0, 21.0), np.zeros(20)
    assert _reach(a, c, 0.1) == 18.0
    assert _reach(a, c, 0.05) == 19.0
    assert _reach(-a, c, 0.1) == 18.0


def test_the_reach_is_the_least_effect_past_which_the_test_rejects() -> None:
    """Over arrangements of slopes in [-1, 1], as a switchback's are: just past the reach at most
    `alpha` of them are as far from 0 as the observed one, and just short of it more are."""
    rng = np.random.default_rng(56)
    for _ in range(200):
        n = int(rng.integers(20, 200))
        a, c = rng.normal(size=n), rng.uniform(-1.0, 1.0, size=n)
        alpha = float(rng.uniform(0.02, 0.3))
        u = _reach(a, c, alpha)
        far = [int(np.sum(np.abs(a - c * v) >= v)) for v in (u * (1.0 + 1e-9), u * (1.0 - 1e-9))]
        assert far[0] <= alpha * n < far[1]


def test_the_observed_arrangement_and_its_reverse_stay_as_far_at_every_effect() -> None:
    """With half the blocks on and no washout, the observed arrangement and its reverse are as
    far from 0 as the observed difference at every effect, so a test at a level below theirs, 2
    of 70, never rejects: the interval reaches without bound, rounding notwithstanding."""
    for seed in range(20):
        kpi, _, design = _world(seed, 0.5, 0.0, 0, 1.0, blocks=8)
        on, kept = design.assignment, design.kept
        observed = float(kpi[on & kept].mean() - kpi[~on & kept].mean())
        _, _, reaches, _, _ = _randomization(kpi, design, observed, (0.02, 0.05))
        assert reaches[0] == (math.inf, math.inf)
        assert all(math.isfinite(u) for u in reaches[1])
    # their slopes are ratios of counts, exactly 1 and -1 at any size and any KPI
    design = switchback_design(np.datetime64("2025-01-06") + np.arange(84), 7, 3, seed=1)
    rows, _ = _arrangements(design)
    kpi = (np.random.default_rng(1).normal(size=84) + 1e8) / 3.0
    _, slopes = _rerandomised(kpi, rows, design)
    flags = np.array(design.on)
    assert slopes[(rows == flags).all(axis=1)].tolist() == [1.0]
    assert slopes[(rows == ~flags).all(axis=1)].tolist() == [-1.0]
    assert np.abs(slopes).max() == 1.0


@pytest.mark.parametrize(("blocks", "lags"), [(4, 1), (12, 2), (32, 3)])
def test_the_newey_west_error_weighs_its_lags_by_bartletts_kernel(blocks: int, lags: int) -> None:
    """The error is Newey and West's over the blocks' scores, their weight a period on or off
    times their mean's distance from their state's, to lag `floor(4 (blocks / 100)^(2/9))` at
    Bartlett's weights, times `blocks / (blocks - 2)`."""
    assert math.floor(4.0 * (blocks / 100.0) ** (2.0 / 9.0)) == lags
    kpi, spend, design = _world(11, 0.8, 0.0, 1, 0.0, blocks=blocks)
    kept = design.kept.reshape(blocks, design.block)
    y = kpi.reshape(blocks, design.block)
    on = np.array(design.on)
    sizes = kept.sum(axis=1)
    means = np.array([y[b][kept[b]].mean() for b in range(blocks)])
    up = y[on][kept[on]].mean()
    down = y[~on][kept[~on]].mean()
    scores = [
        (sizes[b] / sizes[on].sum()) * (means[b] - up)
        if on[b]
        else -(sizes[b] / sizes[~on].sum()) * (means[b] - down)
        for b in range(blocks)
    ]
    variance = sum(
        max(0.0, 1.0 - abs(b - c) / (lags + 1)) * scores[b] * scores[c]
        for b in range(blocks)
        for c in range(blocks)
    )
    _, reading = read_switchback(kpi, spend, design, method="hac")
    count = design.block * design.on_blocks
    expected = count * math.sqrt(variance * blocks / (blocks - 2))
    assert reading.se == pytest.approx(expected, rel=1e-12, abs=0.0)
    assert reading.df == blocks - 2


def test_the_package_exports_the_design_and_its_reading() -> None:
    import mixdyn.experiments as experiments
    from mixdyn.experiments import switchback

    for name in (
        "CarryoverOrder",
        "SwitchbackDesign",
        "SwitchbackReading",
        "carryover_order",
        "read_switchback",
        "switchback_design",
    ):
        assert name in experiments.__all__
        assert getattr(experiments, name) is getattr(switchback, name)


def _lagged(
    seed: int,
    rho: float,
    kernel: tuple[float, ...],
    effect: float,
    *,
    blocks: int = 12,
    block: int = 4,
    washout: int = 1,
) -> tuple[np.ndarray, SwitchbackDesign]:
    """A national KPI over a switchback test drawn at `seed`: a level of 10, a stationary AR(1)
    noise of variance 1 at `rho`, and `effect` times a distributed lag of the channel's state
    weighted by `kernel`, which sums to 1: the carryover reaches `len(kernel) - 1` periods after
    a switch, and once the channel has been on that long it reads `effect` a period. The channel
    is off before the test."""
    periods = blocks * block
    design = switchback_design(DAYS[0] + np.arange(periods), block, washout, seed=seed)
    u = np.random.default_rng((seed, 2611)).normal(size=periods)
    noise = np.empty(periods)
    noise[0] = u[0]
    for t in range(1, periods):
        noise[t] = rho * noise[t - 1] + math.sqrt(1.0 - rho * rho) * u[t]
    state = design.assignment.astype(np.float64)
    lagged = np.zeros(periods)
    for j, weight in enumerate(kernel):
        lagged[j:] += weight * state[: periods - j]
    return 10.0 + effect * lagged + noise, design


ORDER_1 = (0.5, 0.5)
ORDER_2 = (1 / 3, 1 / 3, 1 / 3)


@pytest.mark.parametrize("method", ["randomization", "hac", "jackknife"])
def test_a_burn_in_reads_the_periods_a_washout_of_its_length_leaves(method: str) -> None:
    """A burn-in as long as the design's washout reads the test as the washout does, to the bit,
    and a longer one as a design of that washout, its blocks and order the same, does."""
    kpi, spend, design = _world(4, 0.5, 0.5, 1, 1.0)
    estimate, plain = read_switchback(kpi, spend, design, method=method)  # ty: ignore[invalid-argument-type]
    again, same = read_switchback(kpi, spend, design, method=method, burn_in=1)  # ty: ignore[invalid-argument-type]
    assert repr(same) == repr(plain)
    assert (again.lift, again.se) == (estimate.lift, estimate.se)
    longer = SwitchbackDesign(design.dates, 4, 2, design.on, design.share, design.seed)
    burnt = read_switchback(kpi, spend, design, method=method, burn_in=2)[1]  # ty: ignore[invalid-argument-type]
    assert repr(burnt) == repr(read_switchback(kpi, spend, longer, method=method)[1])  # ty: ignore[invalid-argument-type]
    assert burnt.lift != plain.lift


@pytest.mark.parametrize(
    ("burn_in", "match"),
    [
        (0, "reads periods the design's washout of 1 leaves out"),
        (4, "a burn-in of 4 periods leaves nothing of a block of 4"),
        (-1, "burn_in must be a whole number"),
        (1.5, "burn_in must be a whole number"),
    ],
)
def test_a_burn_in_refuses_less_than_the_washout_or_the_whole_block(
    burn_in: float, match: str
) -> None:
    kpi, spend, design = _world(4, 0.5, 0.5, 1, 1.0)
    with pytest.raises(ValueError, match=match):
        read_switchback(kpi, spend, design, burn_in=burn_in)  # ty: ignore[invalid-argument-type]


def test_the_jackknife_error_is_half_the_gap_between_the_halves_differences() -> None:
    """Five blocks of 2 periods, on, off, off, on, off, a washout of 1: the halves are blocks 0 to
    2 and 2 to 4, sharing block 2, each period kept as the whole test keeps it. The first half
    keeps period 1 on and 3, 4, 5 off, a difference of 5 - 2 = 3; the second keeps 7 on and 4, 5,
    9 off, 9 - 3 = 6; the whole keeps 1, 7 on and 3, 4, 5, 9 off, 7 - 2.5 = 4.5. Four periods
    on: the lift is 18 and its error 4 |3 - 6| / 2 = 6, read as a t at 1 degree of freedom."""
    weeks = np.datetime64("2025-01-06") + np.timedelta64(7, "D") * np.arange(10)
    design = SwitchbackDesign(weeks, 2, 1, (True, False, False, True, False), 0.4, 0)
    kpi = np.array([0.0, 5.0, 0.0, 1.0, 2.0, 3.0, 0.0, 9.0, 0.0, 4.0])
    estimate, reading = read_switchback(kpi, 100.0 * design.assignment, design, method="jackknife")
    assert reading.lift == pytest.approx(18.0, rel=1e-15, abs=0.0)
    assert reading.se == pytest.approx(6.0, rel=1e-15, abs=0.0)
    assert (reading.df, reading.arrangements, reading.enumerated) == (1.0, 0, False)
    t = float(stats.t.ppf(0.975, 1.0))
    assert t == pytest.approx(math.tan(19.0 * math.pi / 40.0), rel=1e-12, abs=0.0)
    assert reading.upper - reading.lift == pytest.approx(6.0 * t, rel=1e-14, abs=0.0)
    assert reading.p_value == pytest.approx(
        1.0 - 2.0 * math.atan(3.0) / math.pi, rel=1e-12, abs=0.0
    )
    assert reading.verdict == "inconclusive"
    assert estimate.se == pytest.approx(6.0 * t / float(stats.norm.ppf(0.975)), rel=1e-14, abs=0.0)


def test_the_jackknife_refuses_a_half_that_keeps_one_state() -> None:
    weeks = np.datetime64("2025-01-06") + np.timedelta64(7, "D") * np.arange(8)
    design = SwitchbackDesign(weeks, 2, 0, (True, True, False, False), 0.5, 0)
    kpi = np.arange(8.0)
    with pytest.raises(ValueError, match="the first half keeps no period off"):
        read_switchback(kpi, 100.0 * design.assignment, design, method="jackknife")
    flat = np.array([1.0, 2.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0])
    design = SwitchbackDesign(weeks, 2, 0, (True, False, True, False), 0.5, 0)
    with pytest.raises(ValueError, match="the same difference"):
        read_switchback(flat, 100.0 * design.assignment, design, method="jackknife")


def test_the_jackknife_reading_holds_its_size() -> None:
    """Over 2000 worlds with no lift and white noise, 12 blocks of 4 and no washout, the 95 %
    jackknife interval misses 0 within 3 binomial standard errors of 5 % (0.0146), the 2 of 924
    designs whose halves keep one state each refused and counted apart (ADR 0056)."""
    misses = refused = 0
    for seed in range(2000):
        kpi, spend, design = _world(seed, 0.0, 0.0, 0, 0.0)
        try:
            _, reading = read_switchback(kpi, spend, design, method="jackknife")
        except ValueError:
            refused += 1
            continue
        misses += not reading.lower <= 0.0 <= reading.upper
    assert refused <= 10
    assert abs(misses / (2000 - refused) - 0.05) <= 3.0 * SIZE_SE


def test_the_carryover_order_test_is_the_one_counted_by_hand() -> None:
    """Four blocks of 3 periods, on, off, on, off, order 0: each block's first period less its
    others' mean is 6, 2, -1 and 0. Over the six arrangements of two blocks on, the blocks on after
    a switch less those off after one read 0, -2, -2, 2, 2 and 0, the first block among neither,
    so four of six are as far from 0 as the observed -2: the carryover check's own value."""
    design = SwitchbackDesign(DAYS[:12], 3, 0, (True, False, True, False), 0.5, 0)
    kpi = np.array([9.0, 3.0, 3.0, 4.0, 2.0, 2.0, 5.0, 6.0, 6.0, 1.0, 1.0, 1.0])
    test = carryover_order(kpi, design)
    assert test == CarryoverOrder(0, test.statistic, test.p_value, 6, True)
    assert test.statistic == pytest.approx(-2.0, rel=0.0, abs=1e-14)
    assert test.p_value == pytest.approx(4 / 6, rel=0.0, abs=1e-15)
    _, reading = read_switchback(kpi, np.repeat([1.0, 0.0, 1.0, 0.0], 3), design, method="hac")
    assert test.statistic == pytest.approx(reading.carryover, rel=0.0, abs=1e-14)
    # at order 1 the epochs are each block's second period and its third, alike in every block
    assert carryover_order(kpi, design, order=1) == CarryoverOrder(1, 0.0, 1.0, 6, True)
    # a KPI by geo reads as its sum
    shares = np.array([[0.25], [0.75]])
    assert carryover_order(shares * kpi, design).p_value == test.p_value


def test_an_effect_of_the_order_tested_leaves_the_test_where_it_was() -> None:
    """The epochs read an effect whose carryover reaches no further than the order alike, in
    every block whatever its state or the one before it: adding an order-1 effect of any size to
    one KPI moves neither the statistic nor the p-value at order 1, which is why the test is
    exact under that null. An order-2 effect moves the statistic, noise-free by exactly twice its
    last weight, `-2 effect / 3` (validation/switchback_carryover.mac, STEP 5)."""
    plain, design = _lagged(7, 0.5, ORDER_1, 0.0)
    base = carryover_order(plain, design)
    for effect in (0.5, 5.0, -40.0):
        kpi, _ = _lagged(7, 0.5, ORDER_1, effect)
        moved = carryover_order(kpi, design)
        assert moved.statistic == pytest.approx(base.statistic, rel=1e-9, abs=1e-12)
        assert moved.p_value == base.p_value
        clean = kpi - plain + 10.0
        assert carryover_order(clean, design).statistic == pytest.approx(0.0, rel=0.0, abs=1e-12)
        lagged, _ = _lagged(7, 0.5, ORDER_2, effect)
        clean = lagged - plain + 10.0
        statistic = carryover_order(clean, design).statistic
        assert statistic == pytest.approx(-2.0 * effect / 3.0, rel=1e-12, abs=0.0)


def test_the_carryover_order_test_rejects_46_of_924_arrangements_whatever_the_noise() -> None:
    """Where the effect does not reach the epochs, each block's gap is the same under every
    arrangement, so taking each of the 924 arrangements of 6 of 12 blocks as the one run, the
    test rejects at 5 % on exactly 46 of them, the 23 pairs of an arrangement and its complement
    farthest from 0: its size is 46/924 = 0.0498 for any noise, not a Monte Carlo rate."""
    kpi = 10.0 + np.random.default_rng(20261106).normal(size=48)
    rejected = 0
    for on in itertools.combinations(range(12), 6):
        state = tuple(b in on for b in range(12))
        design = SwitchbackDesign(DAYS, 4, 1, state, 0.5, 0)
        test = carryover_order(kpi, design)
        assert (test.arrangements, test.enumerated) == (924, True)
        rejected += test.p_value <= 0.05
    assert rejected == 46


def test_the_carryover_order_test_holds_its_size_on_order_1_worlds() -> None:
    """Over 2000 worlds whose channel carries over one period, half its effect of 2 a period, at
    an AR(1) noise of 0.5, 12 blocks of 4 (924 arrangements, enumerated) read at order 1, the test
    rejects at 5 % within 0.02 of 0.05; the binomial standard error is 0.0049, so 0.02 is four of
    them (ADR 0056)."""
    rejected = 0
    for seed in range(2000):
        kpi, design = _lagged(seed, 0.5, ORDER_1, 2.0)
        rejected += carryover_order(kpi, design).p_value <= 0.05
    assert abs(rejected / 2000 - 0.05) <= 0.02


def test_the_carryover_order_test_finds_a_second_period_of_carryover() -> None:
    """Over 500 worlds whose channel carries over two periods, a third of its effect of 4 a
    period in each, white noise, 24 blocks of 4 (10 001 arrangements, drawn) read at order 1, the
    test rejects at 5 % more than 0.8 of the time by over three binomial standard errors (ADR
    0056)."""
    rejected = 0
    for seed in range(500):
        kpi, design = _lagged(seed, 0.0, ORDER_2, 4.0, blocks=24)
        test = carryover_order(kpi, design, order=1)
        assert (test.arrangements, test.enumerated) == (_DRAWN + 1, False)
        rejected += test.p_value <= 0.05
    rate = rejected / 500
    assert rate - 3.0 * math.sqrt(rate * (1.0 - rate) / 500) > 0.8


@pytest.mark.parametrize(
    ("order", "match"),
    [
        (3, "a block of 4 periods holds no second epoch after an order of 3"),
        (-1, "order must be a whole number"),
        (True, "order must be a whole number"),
    ],
)
def test_the_carryover_order_test_refuses_an_order_without_a_second_epoch(
    order: int, match: str
) -> None:
    kpi, design = _lagged(1, 0.0, ORDER_1, 1.0)
    with pytest.raises(ValueError, match=match):
        carryover_order(kpi, design, order=order)
    with pytest.raises(ValueError, match="48 periods"):
        carryover_order(kpi[:47], design)
