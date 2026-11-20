"""mixdyn.experiments: candidate lift tests, and the lift a posterior expects each to read.

On a world of two channels whose truth is known: each draw's predicted lift is the one the Bayesian
model's likelihood reads when the candidate's estimate is in it, to 1e-12, for both curves, a test
after the panel and one inside it, a cut and a heavy-up, and so is the lift readings of the
history's length predict of a test after it read over its cooldown alone, at points of the model
where a retention near 1 makes the kernel's last periods count; a group's standard error is the
spread of its lift over a market of units drawn one by one; where the noise carries over from one
period to the next, a placebo standard error read from the history's own noise covers the lift
where the group's does not, and on a long history it is the group's times Maxima's factor; the
designs lay out every channel's
candidates in order, each as the plan in place or the history runs it; a seesaw alternates the
plan's spend high and low in blocks and spends what the plan spends; and a test held out of the
fit is given a predictive distribution that covers it.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from scipy import signal, stats

from mixdyn import NATIONAL, Cadence, Channel, MediaPanel, Revenue
from mixdyn.calibration import LiftEstimate
from mixdyn.decisions import Readings, readings
from mixdyn.experiments import (
    After,
    Candidate,
    GroupNoise,
    PlaceboNoise,
    Within,
    designs,
    predicted_lift,
    seesaw,
)
from mixdyn.models.bayes_mmm import (
    BayesFit,
    BayesMedia,
    BayesSpec,
    HillCurve,
    TanhCurve,
    bayes_model,
    fit_bayes,
)

WEEKS = 104
NOISE = 20.0
TRUTH = {
    "tv": {"roi": 1.2, "retention": 0.6, "half": 1.0, "slope": 1.0},
    "search": {"roi": 2.0, "retention": 0.2, "half": 0.8, "slope": 1.0},
}
"""Each channel's return over the history, its retention, and its Hill's half-saturation point in
units of its mean positive spend and slope."""
CHANNELS = (Channel("tv"), Channel("search"))


def _spend() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20261005)
    return {
        "tv": rng.gamma(2.0, 50.0, WEEKS) * (rng.random(WEEKS) > 0.3),
        "search": rng.gamma(2.0, 30.0, WEEKS) * (rng.random(WEEKS) > 0.2),
    }


def _truth() -> Readings:
    """The world's channels as one draw a planner reads, in the panel's units."""
    spend = _spend()
    parameters: dict[str, list[float]] = {"retention": [], "half": [], "slope": [], "beta": []}
    for name, x in spend.items():
        t = TRUTH[name]
        weights = t["retention"] ** np.arange(WEEKS)
        unit = x[x > 0].mean()
        adstock = np.convolve(x, weights / weights.sum())[:WEEKS]
        half = t["half"] * unit
        h = adstock ** t["slope"] / (adstock ** t["slope"] + half ** t["slope"])
        parameters["retention"].append(t["retention"])
        parameters["half"].append(half)
        parameters["slope"].append(t["slope"])
        parameters["beta"].append(t["roi"] * x.sum() / h.sum())
    return Readings(
        channels=CHANNELS,
        retention=np.array([parameters["retention"]]),
        half_saturation=np.array([parameters["half"]]),
        slope=np.array([parameters["slope"]]),
        coefficient=np.array([parameters["beta"]]),
        tanh=np.array([False]),
        fit=np.array([0]),
        chain=np.array([0]),
        stacking=np.ones(1),
        pointwise=None,
        length=WEEKS,
    )


def world() -> MediaPanel:
    spend = _spend()
    truth = _truth()
    sales = 1000.0 + NOISE * np.random.default_rng(7).standard_normal(WEEKS)
    for c, name in enumerate(spend):
        x = spend[name]
        weights = truth.retention[0, c] ** np.arange(WEEKS)
        adstock = np.convolve(x, weights / weights.sum())[:WEEKS]
        half, slope = truth.half_saturation[0, c], truth.slope[0, c]
        sales += truth.coefficient[0, c] * adstock**slope / (adstock**slope + half**slope)
    return MediaPanel(
        dates=np.datetime64("2023-01-02") + np.timedelta64(7, "D") * np.arange(WEEKS),
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=CHANNELS,
        controls=(),
        kpi=sales[None, :],
        kpi_unit=Revenue(),
        spend=np.column_stack([spend["tv"], spend["search"]])[None],
        currency="EUR",
        control_values=np.zeros((1, WEEKS, 0)),
        filled=np.zeros((1, WEEKS), dtype=bool),
    )


def _fit(saturation: HillCurve | TanhCurve, estimates: dict[str, list[LiftEstimate]]) -> BayesFit:
    spec = BayesSpec(tuple(BayesMedia(c) for c in CHANNELS), saturation=saturation)
    return fit_bayes(
        world(), NATIONAL, spec, seed=3, estimates=estimates, chains=2, warmup=50, draws=20
    )


PLAN = After({Channel("tv"): 80.0, Channel("search"): 50.0})


@pytest.mark.parametrize(
    ("saturation", "when", "multiplier"),
    [(HillCurve(), PLAN, 2.0), (TanhCurve(), Within(60), 0.0), (HillCurve(), Within(30), 0.5)],
    ids=["hill after, doubled", "tanh within, dark", "hill within, halved"],
)
def test_each_draws_predicted_lift_is_the_one_the_model_reads(
    saturation: HillCurve | TanhCurve, when: After | Within, multiplier: float
) -> None:
    """With the candidate's estimate in the likelihood, its term at each draw is normal about the
    predicted lift, at the market's scale, with the standard error over the share; after the
    panel the kernel runs through the candidate's periods, as the model's does."""
    (candidate,) = designs(
        world(),
        NATIONAL,
        [Channel("tv")],
        GroupNoise(NOISE),
        when=when,
        multipliers=(multiplier,),
        durations=(4,),
        shares=(0.2,),
    )
    estimate = candidate.estimate(-25.0 if multiplier < 1.0 else 25.0)
    fit = _fit(saturation, {"tv": [estimate]})
    assert fit.length == max(WEEKS, candidate.spend.size)
    predicted = predicted_lift(readings(fit), candidate).reshape(2, 20)
    np.testing.assert_allclose(
        fit.log_likelihood[..., -1],
        stats.norm.logpdf(estimate.lift / estimate.share, predicted, estimate.se / estimate.share),
        rtol=1e-12,
    )


@pytest.mark.parametrize("saturation", [HillCurve(), TanhCurve()], ids=["hill", "tanh"])
def test_readings_of_the_history_predict_a_later_tests_carryover_as_the_model_reads_it(
    saturation: HillCurve | TanhCurve,
) -> None:
    """At points of the model whose likelihood holds a heavy-up after the panel read over its
    cooldown alone, the lift each point predicts from readings whose kernels run the history's
    length is the model's: the kernel runs through the test's periods, and the readout reads only
    what the test carries over. Television's retention is held at 0.97, so that the kernel's last
    periods count."""
    (heavy_up,) = designs(
        world(),
        NATIONAL,
        [Channel("tv")],
        GroupNoise(NOISE),
        when=PLAN,
        multipliers=(2.0,),
        durations=(4,),
        shares=(0.2,),
    )
    carryover = Candidate(
        heavy_up.channel, heavy_up.spend, heavy_up.planned, 8, heavy_up.se, heavy_up.share
    )
    estimate = carryover.estimate(10.0)
    spec = BayesSpec(tuple(BayesMedia(c) for c in CHANNELS), saturation=saturation)
    model = bayes_model(world(), NATIONAL, spec, estimates={"tv": [estimate]})
    theta = 0.3 * np.random.default_rng(11).standard_normal((5, model.dimension))
    theta[:, model.coordinates.index("logit_retention[tv]")] = math.log(0.97 / 0.03)
    p = model.parameters(theta)
    hill = isinstance(saturation, HillCurve)
    points = Readings(
        channels=CHANNELS,
        retention=p["retention"],
        half_saturation=p["half_saturation"],
        slope=p["slope"] if hill else np.full((5, 2), np.nan),
        coefficient=p["coefficient"],
        tanh=np.full(5, not hill),
        fit=np.zeros(5, dtype=np.intp),
        chain=np.zeros(5, dtype=np.intp),
        stacking=np.ones(1),
        pointwise=None,
        length=WEEKS,
    )
    np.testing.assert_allclose(
        model.log_likelihood(theta)[:, -1],
        stats.norm.logpdf(
            estimate.lift / estimate.share,
            predicted_lift(points, carryover),
            estimate.se / estimate.share,
        ),
        rtol=1e-12,
    )


def test_a_groups_standard_error_is_its_lifts_spread_over_a_market_of_units() -> None:
    """A market of 50 units, each with its own noise, a group of 10 of them read against the other
    40 scaled to its size over 6 periods: over 40 000 draws the lift's spread is the formula's to
    within 2 %, some 5.7 of its Monte Carlo errors."""
    units, members, periods, draws = 50, 10, 6, 40_000
    rng = np.random.default_rng(11)
    noise = GroupNoise(3.0)
    share = members / units
    lifts = []
    for _ in range(4):
        unit = rng.normal(0.0, noise.market_sd / np.sqrt(units), (draws // 4, periods, units))
        group = unit[..., :members].sum(axis=2)
        control = unit[..., members:].sum(axis=2) * share / (1.0 - share)
        lifts.append((group - control).sum(axis=1))
    spread = float(np.std(np.concatenate(lifts)))
    assert spread == pytest.approx(noise.se(periods, share), rel=0.02, abs=0.0)


def _autoregressive(
    rng: np.random.Generator, shape: tuple[int, int], phi: float, sd: float
) -> np.ndarray:
    """Rows of a stationary first-order autoregression of coefficient `phi` and standard deviation
    `sd` a period."""
    shocks = rng.standard_normal(shape) * sd * math.sqrt(1.0 - phi**2)
    shocks[:, 0] = rng.standard_normal(shape[0]) * sd
    return signal.lfilter([1.0], [1.0, -phi], shocks, axis=1)


_Z = float(stats.norm.ppf(0.975))
_FACTOR = {0.0: 1.0, 0.5: 10923 / 4096, 0.8: 287877519 / 48828125}
"""Maxima's `1 + 2 sum_{k<12} (1 - k/12) phi^k`, the variance of 12 periods' sum of a first-order
autoregression of coefficient `phi` over 12 periods' of independent noise."""


def _readouts(
    phi: float, histories: int, level: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """`histories` markets of a group of a fifth and the rest, each noise a first-order
    autoregression of coefficient `phi`, variances in proportion to their shares and the market's
    standard deviation a period `level`: the market's noise over the periods before a readout of
    12, and the lift's error over the readout, the rest scaled to the group's size."""
    rng = np.random.default_rng((20261013, seed))
    shape, share = (histories, level.size), 0.2
    group = _autoregressive(rng, shape, phi, math.sqrt(share)) * level
    rest = _autoregressive(rng, shape, phi, math.sqrt(1.0 - share)) * level
    error = (group[:, -12:] - share / (1.0 - share) * rest[:, -12:]).sum(axis=1)
    return (group + rest)[:, :-12], error


@pytest.mark.parametrize("phi", [0.0, 0.5, 0.8])
def test_under_noise_carried_over_a_placebo_standard_error_covers_where_the_groups_does_not(
    phi: float,
) -> None:
    """4000 markets of 104 periods and a readout of 12 after them, a group of a fifth, each noise a
    first-order autoregression: the placebo standard error's 95 % interval holds the lift within
    four Monte Carlo errors of 0.95, as an A/A test of the group rejects within four of 0.05. The
    group's standard error at the noise's true size does so only where the noise is independent:
    over 40 000 markets each, it holds 0.950, 0.770 and 0.582 at `phi` 0, 0.5 and 0.8, its error
    `sqrt(10923 / 4096) = 1.63` and `sqrt(287877519 / 48828125) = 2.43` times too small, and the
    placebo's 0.951, 0.952 and 0.948, a median 1.17, 1.22 and 1.31 times the error's spread."""
    histories = 4000
    residuals, error = _readouts(phi, histories, np.ones(116), int(10 * phi))
    placebo = np.array([PlaceboNoise(row).se(12, 0.2) for row in residuals])
    group = GroupNoise(1.0).se(12, 0.2)
    assert float(np.std(error)) / group == pytest.approx(math.sqrt(_FACTOR[phi]), rel=0.05, abs=0.0)
    within = 4.0 * math.sqrt(0.95 * 0.05 / histories)
    assert abs(float(np.mean(np.abs(error) < _Z * placebo)) - 0.95) <= within
    covered = float(np.mean(np.abs(error) < _Z * group))
    if phi == 0.0:
        assert abs(covered - 0.95) <= within
    else:
        assert covered < 0.95 - 10.0 * within


def test_under_a_noise_level_that_rises_a_studentised_placebo_still_covers() -> None:
    """The market's noise trebles its size over 116 periods, at `phi` 0.5: each placebo window is
    read in units of the noise just before it, and the test in those of the noise before it, so the
    interval holds the lift for at least 0.95 less four Monte Carlo errors, 0.960 over 40 000
    markets. Read in the history's mean units instead, the placebo's sums, it held 0.823."""
    histories = 4000
    residuals, error = _readouts(0.5, histories, np.linspace(1.0, 3.0, 116), 7)
    placebo = np.array([PlaceboNoise(row).se(12, 0.2) for row in residuals])
    covered = float(np.mean(np.abs(error) < _Z * placebo))
    assert 0.95 - 4.0 * math.sqrt(0.95 * 0.05 / histories) <= covered <= 0.975


@pytest.mark.parametrize("phi", [0.0, 0.5, 0.8])
def test_on_a_long_history_a_placebo_standard_error_is_the_groups_times_maximas_factor(
    phi: float,
) -> None:
    """Over a million periods, each window studentised by the 100 000 before it: the placebo
    standard error is the group's at the noise's size times the square root of Maxima's variance
    factor, within 2 %, some four of the studentising scale's own standard errors at 0.8."""
    (noise,) = _autoregressive(np.random.default_rng(int(10 * phi)), (1, 1_000_000), phi, 1.0)
    placebo = PlaceboNoise(noise, pre=100_000).se(12, 0.2)
    expected = GroupNoise(1.0).se(12, 0.2) * math.sqrt(_FACTOR[phi])
    assert placebo == pytest.approx(expected, rel=0.02, abs=0.0)


def _placebo_by_hand(residuals: np.ndarray, readout: int, share: float, pre: int) -> float:
    """`PlaceboNoise.se` window by window, as the module states it."""
    t = []
    for first in range(pre, residuals.size - readout + 1):
        scale = math.sqrt(sum(float(x) ** 2 for x in residuals[first - pre : first]) / pre)
        t.append(float(sum(residuals[first : first + readout])) / scale)
    n = len(t)
    square = sum(x * x for x in t)
    rho = [sum(t[f] * t[f + h] for f in range(n - h)) / square for h in range(n)]
    pairs = sum(rho[abs(f - g)] ** 2 for f in range(n) for g in range(n))
    kurtosis = max(n * sum(x**4 for x in t) / square**2, 3.0)
    nu = 2.0 * n * n / ((kurtosis - 1.0) * pairs)
    before = math.sqrt(sum(float(x) ** 2 for x in residuals[-pre:]) / pre)
    widened = float(stats.t.ppf(0.975, nu)) / _Z
    return before * math.sqrt(square / n) * widened * math.sqrt(share / (1.0 - share))


@pytest.mark.parametrize("draw", ["normal", "light"])
def test_a_placebo_standard_error_is_its_windows_studentised_sums_as_the_module_states(
    draw: str,
) -> None:
    """Window by window on 60 periods, normal ones whose studentised sums have a kurtosis of 3.34
    and alternating ones whose sums have 1.23, below a normal's 3, which the count takes instead:
    the standard error is the module's to 1e-12. It scales with the noise and with a group's share
    as the group's does, and copies the residuals it is given."""
    rng = np.random.default_rng(3)
    residuals = {
        "normal": rng.standard_normal(60),
        "light": np.tile([1.0, -1.0, 0.9, -1.1], 15) + 0.01 * rng.standard_normal(60),
    }[draw]
    noise = PlaceboNoise(residuals, pre=8)
    assert noise.se(5, 0.3) == pytest.approx(
        _placebo_by_hand(residuals, 5, 0.3, 8), rel=1e-12, abs=0.0
    )
    assert PlaceboNoise(-3.0 * residuals, pre=8).se(5, 0.3) == pytest.approx(
        3.0 * noise.se(5, 0.3), rel=1e-12, abs=0.0
    )
    group = GroupNoise(1.0)
    assert noise.se(5, 0.1) / noise.se(5, 0.3) == pytest.approx(
        group.se(5, 0.1) / group.se(5, 0.3), rel=1e-12, abs=0.0
    )
    residuals[0] += 1.0
    assert noise.residuals[0] == residuals[0] - 1.0
    with pytest.raises(ValueError, match="read-only"):
        noise.residuals[0] = 0.0


def test_the_designs_lay_out_every_channels_candidates_in_order() -> None:
    panel = world()
    noise = GroupNoise(NOISE)
    laid = designs(
        panel,
        NATIONAL,
        CHANNELS,
        noise,
        when=PLAN,
        multipliers=(0.0, 1.5),
        durations=(2, 4),
        cooldown=3,
        shares=(0.1, 0.3),
    )
    assert len(laid) == 2 * 2 * 2 * 2
    assert len({c.label for c in laid}) == len(laid)
    order = [(c.channel, c.readout, c.share) for c in laid]
    assert order[:4] == [("tv", 5, 0.1), ("tv", 5, 0.3), ("tv", 7, 0.1), ("tv", 7, 0.3)]
    assert [c.channel for c in laid[8:]] == ["search"] * 8
    heavy = laid[6 + 8]
    history = panel.spend[0, :, 1]
    assert heavy.label == "search x1.5 for 4 at 0.1"
    np.testing.assert_array_equal(heavy.planned, 0.1 * np.r_[history, np.full(7, 50.0)])
    np.testing.assert_array_equal(
        heavy.spend, 0.1 * np.r_[history, np.full(4, 75.0), np.full(3, 50.0)]
    )
    assert heavy.se == noise.se(7, 0.1)
    tv = panel.spend[0, :, 0]
    for start, end in ((80, 94), (90, 104)):
        inside = designs(
            panel,
            NATIONAL,
            [Channel("tv")],
            noise,
            when=Within(start),
            multipliers=(0.0,),
            durations=(4,),
            cooldown=10,
            shares=(0.5,),
        )[0]
        np.testing.assert_array_equal(inside.planned, 0.5 * tv[:end])
        expected = 0.5 * np.r_[tv[:start], np.zeros(4), tv[start + 4 : end]]
        np.testing.assert_array_equal(inside.spend, expected)
        assert inside.readout == 14


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"geo": "nowhere"}, "no geo 'nowhere'"),
        ({"channels": []}, r"designs need channels of the panel, each once: got \[\]"),
        ({"channels": ["tv", "radio"]}, r"got \['tv', 'radio'\]"),
        ({"channels": ["tv", "tv"]}, r"got \['tv', 'tv'\]"),
        ({"multipliers": (1.0,)}, r"not 1, got \[1.0\]"),
        ({"multipliers": (-0.5,)}, "at or above 0"),
        ({"multipliers": (float("inf"),)}, "must be finite"),
        ({"durations": (0,)}, "a duration must be a whole number, at least 1, got 0"),
        ({"durations": (2.0,)}, "a duration must be a whole number, at least 1, got 2.0"),
        ({"cooldown": -1}, "cooldown must be a whole number, at least 0, got -1"),
        ({"shares": (1.0,)}, r"share=1.0 is not in \(0, 1\)"),
        ({"shares": (0.0,)}, r"share=0.0 is not in \(0, 1\)"),
        ({"when": After({Channel("tv"): 80.0})}, "no spend for channel 'search'"),
        ({"when": After({Channel("tv"): 80.0, Channel("search"): -1.0})}, "spends -1.0 a period"),
        ({"when": Within(-1)}, "start must be a whole number, at least 0, got -1"),
        ({"when": Within(100)}, "from period 100 ends past the history's 104"),
        ({"when": After({Channel("tv"): 0.0, Channel("search"): 1.0})}, "'tv' spends nothing"),
    ],
)
def test_designs_refuse_what_cannot_be_tested(change: dict[str, Any], match: str) -> None:
    arguments: dict[str, Any] = {
        "geo": NATIONAL,
        "channels": list(CHANNELS),
        "when": PLAN,
        "multipliers": (0.5,),
        "durations": (4,),
        "cooldown": 8,
        "shares": (0.2,),
    } | change
    with pytest.raises(ValueError, match=match):
        designs(
            world(),
            arguments.pop("geo"),
            [Channel(c) for c in arguments.pop("channels")],
            GroupNoise(NOISE),
            **arguments,
        )


def test_a_seesaw_alternates_the_spend_about_the_plan_in_blocks_then_cools_down() -> None:
    """Blocks of 3 weeks at twice and at none of the plan's 50, from a high one, then 4 weeks at
    the plan: at a plan held alike it spends what the plan spends; within the history it moves the
    history's own spend."""
    panel = world()
    noise = GroupNoise(NOISE)
    laid = seesaw(
        panel,
        NATIONAL,
        CHANNELS[::-1],
        noise,
        when=PLAN,
        high=2.0,
        low=0.0,
        block=3,
        blocks=4,
        cooldown=4,
        shares=(0.1, 0.3),
    )
    assert [(c.channel, c.share) for c in laid] == [
        ("search", 0.1),
        ("search", 0.3),
        ("tv", 0.1),
        ("tv", 0.3),
    ]
    search = laid[1]
    history = panel.spend[0, :, 1]
    assert search.label == "search seesaw x2/x0 by 3 over 4 at 0.3"
    assert search.readout == 16
    assert search.se == noise.se(16, 0.3)
    np.testing.assert_array_equal(search.planned, 0.3 * np.r_[history, np.full(16, 50.0)])
    blocks = np.r_[np.full(3, 100.0), np.zeros(3), np.full(3, 100.0), np.zeros(3)]
    np.testing.assert_array_equal(search.spend, 0.3 * np.r_[history, blocks, np.full(4, 50.0)])
    assert search.spend.sum() == pytest.approx(search.planned.sum(), rel=1e-15, abs=0.0)
    tv = panel.spend[0, :, 0]
    inside = seesaw(
        panel,
        NATIONAL,
        [Channel("tv")],
        noise,
        when=Within(90),
        block=1,
        blocks=2,
        cooldown=2,
        shares=(0.5,),
    )[0]
    np.testing.assert_array_equal(inside.planned, 0.5 * tv[:94])
    expected = tv[:94].copy()
    expected[90:92] *= [1.5, 0.5]
    np.testing.assert_array_equal(inside.spend, 0.5 * expected)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"geo": "nowhere"}, "no geo 'nowhere'"),
        ({"channels": ["tv", "tv"]}, r"got \['tv', 'tv'\]"),
        ({"high": 1.0}, "a finite high above 1 and a low in"),
        ({"high": float("inf")}, "a finite high above 1"),
        ({"low": 1.0}, r"a low in \[0, 1\), got 1.5, 1.0"),
        ({"low": -0.5}, r"got 1.5, -0.5"),
        ({"block": 0}, "block must be a whole number, at least 1, got 0"),
        ({"blocks": 3}, "blocks must be even, each high block with a low one, got 3"),
        ({"blocks": 0}, "blocks must be a whole number, at least 2, got 0"),
        ({"cooldown": -1}, "cooldown must be a whole number, at least 0, got -1"),
        ({"shares": (1.0,)}, r"share=1.0 is not in \(0, 1\)"),
        ({"when": Within(100)}, "from period 100 ends past the history's 104"),
        ({"when": After({Channel("tv"): 0.0, Channel("search"): 1.0})}, "'tv' spends nothing"),
    ],
)
def test_a_seesaw_refuses_what_cannot_be_tested(change: dict[str, Any], match: str) -> None:
    arguments: dict[str, Any] = {
        "geo": NATIONAL,
        "channels": list(CHANNELS),
        "when": PLAN,
        "high": 1.5,
        "low": 0.5,
        "shares": (0.2,),
    } | change
    with pytest.raises(ValueError, match=match):
        seesaw(
            world(),
            arguments.pop("geo"),
            [Channel(c) for c in arguments.pop("channels")],
            GroupNoise(NOISE),
            **arguments,
        )


def test_a_candidate_reports_its_estimate_once_its_lift_is_read() -> None:
    candidate = Candidate(Channel("tv"), np.r_[10.0, 0.0, 0.0], np.full(3, 10.0), 2, 4.0, 0.25)
    estimate = candidate.estimate(-3.0)
    assert (estimate.readout, estimate.lift, estimate.se, estimate.share) == (2, -3.0, 4.0, 0.25)
    np.testing.assert_array_equal(estimate.spend, candidate.spend)
    assert estimate.geos is None
    assert candidate.estimate(-3.0, se=6.0).se == 6.0
    with pytest.raises(ValueError, match="read-only"):
        candidate.spend[0] = 1.0
    with pytest.raises(ValueError, match="the test changed no spend"):
        Candidate(Channel("tv"), np.full(3, 10.0), np.full(3, 10.0), 2, 4.0, 0.25)
    with pytest.raises(ValueError, match="the standard error must be positive"):
        Candidate(Channel("tv"), np.r_[10.0, 0.0, 0.0], np.full(3, 10.0), 2, 0.0, 0.25)


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: GroupNoise(0.0), r"market_sd=0.0 is not positive and finite"),
        (lambda: GroupNoise(float("nan")), "is not positive and finite"),
        (lambda: GroupNoise(1.0).se(0, 0.2), "readout must be a whole number, at least 1"),
        (lambda: GroupNoise(1.0).se(4, 1.0), r"share=1.0 is not in \(0, 1\)"),
    ],
)
def test_group_noise_refuses_what_has_no_spread(make, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        make()


_RESIDUALS = np.random.default_rng(4).standard_normal(40)
_FOUR: Any = 4.0


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: PlaceboNoise(np.ones((2, 20))), r"one-dimensional finite series, got shape"),
        (lambda: PlaceboNoise(np.r_[_RESIDUALS, np.nan]), "one-dimensional finite series"),
        (lambda: PlaceboNoise(_RESIDUALS, pre=1), "pre must be a whole number, at least 2"),
        (lambda: PlaceboNoise(_RESIDUALS, pre=_FOUR), "pre must be a whole number, at least 2"),
        (lambda: PlaceboNoise(_RESIDUALS, 8).se(0, 0.2), "readout must be a whole number"),
        (lambda: PlaceboNoise(_RESIDUALS, 8).se(4, 1.0), r"share=1.0 is not in \(0, 1\)"),
        (
            lambda: PlaceboNoise(_RESIDUALS, 30).se(11, 0.2),
            "40 periods of residuals hold no placebo window of 11 periods after 30 before it",
        ),
        (
            lambda: PlaceboNoise(np.r_[_RESIDUALS, np.zeros(8)], 8).se(4, 0.2),
            "the residuals are 0 over 8 periods",
        ),
        (
            lambda: PlaceboNoise(np.r_[np.zeros(8), _RESIDUALS], 8).se(4, 0.2),
            "the residuals are 0 over 8 periods",
        ),
        (
            lambda: PlaceboNoise(np.tile([1.0, -1.0], 20), 4).se(2, 0.2),
            "every placebo window of 2 periods sums to 0",
        ),
    ],
)
def test_placebo_noise_refuses_what_reads_no_spread(make, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        make()


def test_the_designs_read_a_placebo_standard_error_for_each_candidate() -> None:
    noise = PlaceboNoise(np.random.default_rng(6).standard_normal(104) * NOISE)
    laid = designs(
        world(),
        NATIONAL,
        CHANNELS,
        noise,
        when=PLAN,
        multipliers=(0.0,),
        durations=(2, 4),
        cooldown=8,
        shares=(0.1, 0.3),
    )
    assert len(laid) == 8
    for candidate in laid:
        assert candidate.se == noise.se(candidate.readout, candidate.share)


def test_a_lift_of_a_channel_the_readings_lack_is_refused() -> None:
    candidate = Candidate(Channel("radio"), np.r_[10.0, 0.0], np.full(2, 10.0), 1, 4.0, 0.25)
    with pytest.raises(ValueError, match="the readings have no channel 'radio'"):
        predicted_lift(_truth(), candidate)


def test_a_test_held_out_of_the_fit_is_given_a_predictive_distribution_that_covers_it() -> None:
    """A go-dark of television in a fifth of the market over four weeks of the history and eight
    after, held out of the fit: each draw predicts its lift, and the lift the truth predicts lies
    inside the central 95 % of the draws' and of the predictive distribution, each draw's share of
    it normal with the candidate's standard error."""
    (candidate,) = designs(
        world(),
        NATIONAL,
        [Channel("tv")],
        GroupNoise(NOISE),
        when=Within(70),
        multipliers=(0.0,),
        durations=(4,),
        shares=(0.2,),
    )
    fit = fit_bayes(
        world(),
        NATIONAL,
        BayesSpec(tuple(BayesMedia(c) for c in CHANNELS)),
        seed=5,
        chains=2,
        warmup=300,
        draws=200,
    )
    predicted = candidate.share * predicted_lift(readings(fit), candidate)
    assert predicted.shape == (400,)
    assert np.all(np.isfinite(predicted))
    truth = candidate.share * float(predicted_lift(_truth(), candidate)[0])
    assert truth < 0.0
    assert 0.025 < float(np.mean(predicted < truth)) < 0.975
    level = float(np.mean(stats.norm.cdf(truth, predicted, candidate.se)))
    assert 0.025 < level < 0.975
