"""The endogenous-spend flag (ADR 0074): `gamma` is the least squares' and its standard error
Newey and West's; the flag fires where the spend reads the KPI's last period, Track M v2's paid
shopping among them, and stays quiet where the spend reads nothing of it."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mixdyn import NATIONAL, Cadence, Channel, IssueCode, MediaPanel, Revenue, Severity
from mixdyn.data.feedback import SpendFeedback, spend_feedback
from search_demand_world import world as search_world
from track_m2_worlds import KINDS, SEEDS, drawn, world, worlds


def _panel(kpi: np.ndarray, spend: np.ndarray, filled: np.ndarray | None = None) -> MediaPanel:
    weeks = kpi.size
    filled = np.zeros(weeks, dtype=bool) if filled is None else filled
    kpi, spend = np.where(filled, np.nan, kpi), np.where(filled[:, None], 0.0, spend)
    return MediaPanel(
        dates=np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(weeks),
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=tuple(Channel(f"c{j}") for j in range(spend.shape[1])),
        controls=(),
        kpi=kpi[None],
        kpi_unit=Revenue(),
        spend=spend[None],
        currency="EUR",
        control_values=np.zeros((1, weeks, 0)),
        filled=filled[None],
    )


def _reading(seed: int, weeks: int = 156, reads: float = 0.05) -> tuple[MediaPanel, np.ndarray]:
    """A KPI around 1000 whose noise is a stationary AR(1) of coefficient 0.8 and sd 50, and two
    spends: c0 is `20 + reads kpi_{t-1}` and normal noise of sd 2, c1 a Gamma(4, 10) drawn apart."""
    rng = np.random.default_rng(seed)
    u = np.empty(weeks)
    u[0] = rng.normal(0.0, 50.0)
    for t in range(1, weeks):
        u[t] = 0.8 * u[t - 1] + rng.normal(0.0, 50.0 * math.sqrt(1.0 - 0.8**2))
    kpi = 1000.0 + u
    c0 = 20.0 + reads * np.concatenate([[1000.0], kpi[:-1]]) + rng.normal(0.0, 2.0, weeks)
    c1 = rng.gamma(4.0, 10.0, weeks)
    return _panel(kpi, np.column_stack([c0, c1])), kpi


def test_gamma_is_the_least_squares_and_its_error_newey_west_s() -> None:
    """On a panel with a week missing, the coefficient is the least squares' on the spend's and
    the KPI's last periods and the calendar, over the periods observed with the one before; its
    standard error the Bartlett-weighted sum over pairs of periods both read, each residual over one
    less its leverage, written here as the double sum it is; and its interval's half-width that
    error times Kiefer and Vogelsang's fixed-b value at the level, 1.6449 + 2.1859 b + 0.3142 b^2 -
    0.3427 b^3 at 0.9."""
    weeks = 80
    rng = np.random.default_rng(3)
    kpi = 500.0 + np.cumsum(rng.normal(0.0, 5.0, weeks))
    spend = np.column_stack([rng.gamma(3.0, 10.0, weeks), np.full(weeks, 7.0)])
    filled = np.zeros(weeks, dtype=bool)
    filled[30] = True
    panel = _panel(kpi, spend, filled)
    (read,) = spend_feedback(panel, NATIONAL, level=0.9, harmonics=2)
    assert read.channel == "c0"  # c1 never moves, so it has no feedback to read
    rows = [t for t in range(1, weeks) if not filled[t] and not filled[t - 1]]
    assert read.periods == len(rows) == weeks - 3
    year = 365.25 / 7.0
    x = np.array(
        [
            [
                1.0,
                s / weeks,
                *(f(2.0 * math.pi * h * s / year) for h in (1, 2) for f in (math.cos, math.sin)),
                spend[s - 1, 0],
                kpi[s - 1],
            ]
            for s in rows
        ]
    )
    y = spend[rows, 0]
    bread = np.linalg.inv(x.T @ x)
    b = bread @ x.T @ y
    leverage = np.diag(x @ bread @ x.T)
    e = (y - x @ b) / (1.0 - leverage)
    n, k = x.shape
    lags = int(4.0 * (n / 100.0) ** (2.0 / 9.0))
    meat = np.zeros((k, k))
    for i, s in enumerate(rows):
        for j, r in enumerate(rows):
            if abs(s - r) <= lags:
                meat += (1.0 - abs(s - r) / (lags + 1.0)) * e[i] * e[j] * np.outer(x[i], x[j])
    se = math.sqrt((bread @ meat @ bread)[-1, -1])
    assert read.lags == lags == 3
    assert read.gamma == pytest.approx(b[-1], rel=1e-9, abs=0.0)
    assert read.se == pytest.approx(se, rel=1e-9, abs=0.0)
    fixed_b = (lags + 1) / n
    half = (1.6449 + 2.1859 * fixed_b + 0.3142 * fixed_b**2 - 0.3427 * fixed_b**3) * se
    assert (read.lower, read.upper) == pytest.approx((b[-1] - half, b[-1] + half), rel=1e-9)


def test_a_spend_that_reads_last_week_s_kpi_is_flagged() -> None:
    """c0 reads 0.05 of last week's KPI: `gamma` lies within five standard errors of it and its
    interval excludes 0, so the flag fires; c1, drawn apart, reads within five standard errors of
    0. Five standard errors keep the test's own false-red chance near 1e-6; c0's interval excludes
    0 by some 20 of them."""
    panel, _ = _reading(7)
    c0, c1 = spend_feedback(panel, NATIONAL)
    assert abs(c0.gamma - 0.05) < 5.0 * c0.se
    assert c0.follows
    issue = c0.issue()
    assert issue is not None
    assert (issue.code, issue.severity, issue.geo, issue.column, issue.channel) == (
        IssueCode.ENDOGENOUS_SPEND,
        Severity.WARNING,
        "national",
        "c0",
        "c0",
    )
    assert "'c0''s spend reads the KPI's last period" in issue.message
    assert abs(c1.gamma) < 5.0 * c1.se


def test_the_flag_fires_on_track_m2_s_reference_world() -> None:
    """Heusch's reference instance, as the fixture holds it at five seeds: meta's spend, raised
    ahead of each promotion and held back in it, reads a strong week's KPI with a fall, and the
    flag fires on it at every seed, as it did at 1000 of 1000 (ADR 0074); television's, which
    reads only the quarter's budget, is spared at every seed, as at 999 of 1000."""
    for seed in SEEDS:
        read = {f.channel: f for f in spend_feedback(world("reference", seed).panel(), NATIONAL)}
        assert read["meta"].follows, seed
        assert read["meta"].gamma < 0.0
        assert not read["tv"].follows, seed
        issue = read["meta"].issue()
        assert issue is not None
        assert issue.code is IssueCode.ENDOGENOUS_SPEND


def test_the_flag_reads_search_and_spares_television_on_the_search_demand_world() -> None:
    """On 200 search-demand worlds the flag holds 0 for television, flighted apart from the KPI, in
    at least 90 % of them, and fires for search, whose spend follows a demand that persists, in
    most. The share of seeds holding 0 is nominally 0.95; under 0.9 at 200 seeds is three of its
    Monte Carlo standard errors below it."""
    quiet, fired = 0, 0
    for seed in range(200):
        read = {f.channel: f for f in spend_feedback(search_world(seed).panel, NATIONAL)}
        quiet += int(not read["tv"].follows)
        fired += int(read["search"].follows)
    assert quiet >= 180
    assert fired >= 100


def test_the_flag_refuses_what_it_cannot_read() -> None:
    panel, _ = _reading(1)
    with pytest.raises(ValueError, match="no geo"):
        spend_feedback(panel, "north")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="level"):
        spend_feedback(panel, NATIONAL, level=1.0)
    with pytest.raises(ValueError, match="harmonics"):
        spend_feedback(panel, NATIONAL, harmonics=0)
    short, _ = _reading(1, weeks=12)
    with pytest.raises(ValueError, match="periods observed with the one before"):
        spend_feedback(short, NATIONAL)


def test_a_reading_is_one_value() -> None:
    read = SpendFeedback("national", "tv", 0.1, 0.02, 0.06, 0.14, 0.95, 155, 4)
    assert read.follows
    assert not SpendFeedback("national", "tv", 0.01, 0.02, -0.03, 0.05, 0.95, 155, 4).follows


# ------------------------------------------------------------------------- with the bench at hand


def test_the_fixture_is_the_bench_s() -> None:
    """The fixture is what the bench draws, so it cannot drift from it unseen. Equal to 1e-12 of
    each value, not to the bit, since numpy's float64 kernels move the last bits on some
    machines."""
    pytest.importorskip("causaldyn_bench")
    fresh = drawn()
    for held, again in zip(worlds(), fresh["worlds"], strict=True):
        assert (held.kind, held.seed, held.channels) == (
            again["kind"],
            again["seed"],
            tuple(again["channels"]),
        )
        for name in ("sales", "spend", "promotion", "price", "roas"):
            np.testing.assert_allclose(
                getattr(held, name), np.array(again[name]), rtol=1e-12, atol=0.0, err_msg=name
            )


def _fires(switches: dict[str, bool], seeds: int) -> dict[str, int]:
    """How many of `seeds` worlds of his generator at `switches` the flag fires on, channel by
    channel."""
    from causaldyn_bench.endogenous_mmm import EndogenousMediaMix

    from track_m2_worlds import TrackM2World

    fired = dict.fromkeys(("pla", "meta", "tv"), 0)
    for seed in range(seeds):
        w = EndogenousMediaMix(**switches).simulate(seed)
        panel = TrackM2World(
            "",
            seed,
            w.channels,
            w.sales,
            w.spend,
            w.observed_promotion.astype(float),
            w.observed_price,
            w.roas(),
        ).panel()
        for f in spend_feedback(panel, NATIONAL):
            fired[f.channel] += int(f.follows)
    return fired


def test_the_flag_is_quiet_on_track_m2_s_exogenous_worlds() -> None:
    """His generator with its four mechanisms off at 200 seeds: each channel's interval holds 0 in
    at least 90 % of them, three Monte Carlo standard errors below the nominal 0.95; at 1000 seeds
    it fired on 5.3 % to 5.7 % (ADR 0074)."""
    pytest.importorskip("causaldyn_bench")
    fired = _fires(KINDS["exogenous"], 200)
    assert max(fired.values()) <= 20, fired


def test_the_flag_reads_paid_shopping_s_bidding_rule() -> None:
    """The bidding rule alone, at 200 seeds: the flag fires on paid shopping at 190 or more, at
    1000 of 1000 in the study; at his reference instance, where anticipation pulls the other way,
    at 100 or more, at 648 of 1000 in the study, four Monte Carlo standard errors above 100."""
    pytest.importorskip("causaldyn_bench")
    alone = {"budget_feedback": False, "anticipatory": False, "bursts": False, "bidding": True}
    assert _fires(alone, 200)["pla"] >= 190
    assert _fires(KINDS["reference"], 200)["pla"] >= 100
