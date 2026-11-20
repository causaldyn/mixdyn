"""A geo test read by synthetic difference in differences (`mixdyn.experiments.sdid`), as
Arkhangelsky, Athey, Hirshberg, Imbens and Wager (2021) state it.

The estimate does not move with the KPI's units, nor with each geo's level or each period's; its
weights are each problem's optimum over the simplex, checked against an exact rational one that
Maxima's enumeration of every support also found; with every weight equal it is plain difference in
differences, and both of the paper's forms of it agree. On a panel of factors without noise whose
post-period the pre-period's periods span, the effect is recovered. With few treated geos and no
effect, the placebo test rejects at its level, and the jackknife, which leaves out every geo,
controls included, at or below it.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy import stats

from mixdyn import Cadence, Channel, Geo, MediaPanel, Revenue
from mixdyn.calibration import LiftEstimate
from mixdyn.experiments.reading import GeoTest, read_did
from mixdyn.experiments.sdid import SdidReading, read_sdid, synthetic_did

if TYPE_CHECKING:
    from collections.abc import Iterable

PRE, POST, CONTROLS = 10, 5, 20
START = np.datetime64("2025-01-06") + np.timedelta64(7 * PRE, "D")
END = START + np.timedelta64(7 * (POST - 1), "D")
SPEND, PLANNED = np.full(POST, 2.0), np.full(POST, 1.0)


def panel(y: np.ndarray) -> MediaPanel:
    """A weekly panel of the geos `g00`, `g01`, ... with `y` as its KPI and one idle channel."""
    geos, weeks = y.shape
    return MediaPanel(
        dates=np.datetime64("2025-01-06") + np.timedelta64(7, "D") * np.arange(weeks),
        cadence=Cadence.WEEK,
        geos=tuple(Geo(f"g{i:02d}") for i in range(geos)),
        channels=(Channel("tv"),),
        controls=(),
        kpi=y,
        kpi_unit=Revenue(),
        spend=np.zeros((geos, weeks, 1)),
        currency="EUR",
        control_values=np.zeros((geos, weeks, 0)),
        filled=np.zeros((geos, weeks), dtype=bool),
    )


def null_panel(rng: np.random.Generator, treated: int, controls: int = CONTROLS) -> np.ndarray:
    """`controls + treated` geos over `PRE + POST` weeks: a level of sd 2 a geo, a shock a week
    and unit noise, the first `treated` geos treated to no effect."""
    geos, weeks = controls + treated, PRE + POST
    return (
        rng.normal(0.0, 2.0, geos)[:, None]
        + rng.normal(0.0, 1.0, weeks)[None]
        + rng.normal(0.0, 1.0, (geos, weeks))
    )


def flags(geos: int, treated: int) -> np.ndarray:
    return np.arange(geos) < treated


def treated_geos(treated: int) -> frozenset[Geo]:
    return frozenset(Geo(f"g{i:02d}") for i in range(treated))


def read(y: np.ndarray, treated: int, **kwargs) -> tuple[LiftEstimate, SdidReading]:
    return read_sdid(
        panel(y), treated_geos(treated), START, END, spend=SPEND, planned=PLANNED, **kwargs
    )


# ----------------------------------------------------------------------------- units and levels


@pytest.mark.parametrize(
    ("treated", "variance"), [(1, "placebo"), (3, "placebo"), (3, "jackknife")]
)
def test_the_estimate_and_its_error_scale_with_the_kpi_s_units(treated: int, variance) -> None:
    y = null_panel(np.random.default_rng(1), treated)
    y[:treated, PRE:] += 0.7
    one, reading = read(y, treated, variance=variance, replications=30)
    thousand, scaled = read(1000.0 * y, treated, variance=variance, replications=30)
    assert thousand.lift == pytest.approx(1000.0 * one.lift, rel=1e-9, abs=0.0)
    assert thousand.se == pytest.approx(1000.0 * one.se, rel=1e-9, abs=0.0)
    np.testing.assert_allclose(scaled.unit_weights, reading.unit_weights, rtol=0, atol=1e-9)
    np.testing.assert_allclose(scaled.time_weights, reading.time_weights, rtol=0, atol=1e-9)
    assert scaled.p_value == pytest.approx(reading.p_value, rel=1e-9, abs=0.0)


def test_the_estimate_does_not_move_with_a_geo_s_level_or_a_common_trend() -> None:
    """The intercepts take each geo's level and the time weights' intercept a trend all geos
    share. A trend's steps are equal, so the noise level, read from the steps about their mean,
    holds; a week's shock that is not a trend moves it, and with it `zeta`, as the paper reads
    it."""
    rng = np.random.default_rng(2)
    y = null_panel(rng, 2)
    shifted = y + rng.normal(0.0, 50.0, y.shape[0])[:, None] + 7.0 * np.arange(y.shape[1])
    fit, moved = (
        synthetic_did(y, flags(y.shape[0], 2), POST),
        synthetic_did(shifted, flags(y.shape[0], 2), POST),
    )
    assert moved.effect == pytest.approx(fit.effect, rel=0.0, abs=1e-9)
    np.testing.assert_allclose(moved.unit_weights, fit.unit_weights, rtol=0, atol=1e-9)
    np.testing.assert_allclose(moved.time_weights, fit.time_weights, rtol=0, atol=1e-9)


# ------------------------------------------------------------------------------------ the weights


def _optimal(a: np.ndarray, b: np.ndarray, ridge: float, w: np.ndarray) -> float:
    """The residual of `w`'s conditions of optimality, over an intercept and the simplex, for
    `||w0 + a w - b||^2 + ridge ||w||^2`, its intercept the best, relative to the gradient's
    scale: the gradient equal on the support and no lower off it."""
    residual = a @ w - b
    residual -= residual.mean()
    gradient = 2.0 * (a.T @ residual + ridge * w)
    support = w > 0.0
    level = gradient[support].mean()
    scale = 2.0 * (np.abs((a - a.mean(axis=0)).T @ (a - a.mean(axis=0))).max() + ridge)
    off = np.maximum(level - gradient[~support], 0.0).max(initial=0.0)
    return max(np.abs(gradient[support] - level).max(), off) / scale


@pytest.mark.parametrize("seed", range(6))
def test_each_weight_is_its_problem_s_optimum_over_the_simplex(seed: int) -> None:
    rng = np.random.default_rng(seed)
    treated = 1 + seed % 3
    y = null_panel(rng, treated) * 10.0 ** (seed - 3)
    tr = flags(y.shape[0], treated)
    fit = synthetic_did(y, tr, POST)
    controls = y[~tr, :PRE]
    for w in (fit.unit_weights, fit.time_weights):
        assert np.all(w >= 0.0)
        assert w.sum() == pytest.approx(1.0, rel=0.0, abs=1e-14)
    steps = np.diff(controls, axis=1).ravel()
    sigma = math.sqrt(np.sum((steps - steps.mean()) ** 2) / (CONTROLS * (PRE - 1)))
    zeta = (treated * POST) ** 0.25 * sigma
    assert (fit.noise, fit.zeta) == (
        pytest.approx(sigma, rel=1e-12, abs=0.0),
        pytest.approx(zeta, rel=1e-12, abs=0.0),
    )
    unit = _optimal(controls.T, y[tr, :PRE].mean(axis=0), zeta**2 * PRE, fit.unit_weights)
    time = _optimal(
        controls, y[~tr, PRE:].mean(axis=1), (1e-6 * sigma) ** 2 * CONTROLS, fit.time_weights
    )
    assert max(unit, time) < 1e-10
    assert fit.kkt < 1e-10
    assert np.count_nonzero(fit.unit_weights == 0.0) + np.count_nonzero(fit.time_weights == 0.0)


# One treated unit and four controls over four weeks before and one after, the noise level and both
# weight problems exact. Maxima 5.46.0, enumerating every support of each problem with an explicit
# intercept, found one point meeting the conditions of optimality in each, at these rationals; the
# same problems without their intercepts have other points.
TINY = [
    [-3, 4, 2, 4, 1],
    [2, -1, 3, -3, -3],
    [-1, 2, 0, 3, 4],
    [-1, -1, 0, 1, 0],
    [1, 4, 2, -2, -1],
]
MAXIMA_SIGMA2 = "341/36"
MAXIMA_UNIT = ["0", "594227/1202964", "721327/2405928", "496147/2405928"]
MAXIMA_TIME = ["0", "51750000000341/348750000000682", "0", "297000000000341/348750000000682"]
MAXIMA_TAU = "-45729841500088089881/13533345000026465208"


def _sum(values: Iterable[Fraction]) -> Fraction:
    return sum(values, Fraction(0))


def _exact_simplex(a: list[list[Fraction]], b: list[Fraction], ridge: Fraction) -> list[Fraction]:
    """The simplex optimum of `||w0 + a w - b||^2 + ridge ||w||^2` in exact arithmetic: the one
    support whose stationary point is positive on it with the multipliers off it at or above 0."""
    rows, n = len(a), len(a[0])
    found = []
    for size in range(1, n + 1):
        for support in combinations(range(n), size):
            # unknowns: w0, w_j on the support, mu; equations: d/dw0 = 0, d/dw_j = mu, sum w = 1
            k = len(support) + 2
            m = [[Fraction(0)] * (k + 1) for _ in range(k)]
            design = [[Fraction(1)] + [a[r][j] for j in support] for r in range(rows)]
            for i in range(k - 1):
                for c in range(k - 1):
                    m[i][c] = 2 * _sum(design[r][i] * design[r][c] for r in range(rows))
                    if i == c and i > 0:
                        m[i][c] += 2 * ridge
                m[i][k] = 2 * _sum(design[r][i] * b[r] for r in range(rows))
                if i > 0:
                    m[i][k - 1] = Fraction(-1)
            m[k - 1] = [Fraction(0)] + [Fraction(1)] * len(support) + [Fraction(0), Fraction(1)]
            solution = _solve(m)
            if solution is None:
                continue
            w = [Fraction(0)] * n
            for j, value in zip(support, solution[1:-1], strict=True):
                w[j] = value
            mu = solution[-1]
            residual = [
                solution[0] + _sum(a[r][j] * w[j] for j in range(n)) - b[r] for r in range(rows)
            ]
            gradient = [
                2 * _sum(a[r][j] * residual[r] for r in range(rows)) + 2 * ridge * w[j]
                for j in range(n)
            ]
            if all(w[j] > 0 for j in support) and all(
                gradient[j] - mu >= 0 for j in range(n) if j not in support
            ):
                found.append(w)
    assert len(found) == 1
    return found[0]


def _solve(m: list[list[Fraction]]) -> list[Fraction] | None:
    """Gauss-Jordan elimination of an augmented square system, or None where it is singular."""
    k = len(m)
    m = [row[:] for row in m]
    for col in range(k):
        pivot = next((r for r in range(col, k) if m[r][col] != 0), None)
        if pivot is None:
            return None
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(k):
            if r != col and m[r][col] != 0:
                factor = m[r][col] / m[col][col]
                m[r] = [x - factor * p for x, p in zip(m[r], m[col], strict=True)]
    return [m[r][k] / m[r][r] for r in range(k)]


def test_the_weights_are_the_exact_rational_optimum_maxima_found() -> None:
    y = [[Fraction(v) for v in row] for row in TINY]
    controls = y[1:]
    steps = [row[t + 1] - row[t] for row in controls for t in range(3)]
    mean = _sum(steps) / len(steps)
    sigma2 = _sum((s - mean) ** 2 for s in steps) / len(steps)
    assert sigma2 == Fraction(MAXIMA_SIGMA2)
    unit = _exact_simplex(
        [[controls[i][t] for i in range(4)] for t in range(4)],
        [y[0][t] for t in range(4)],
        4 * sigma2,
    )
    time = _exact_simplex(
        [row[:4] for row in controls], [row[4] for row in controls], sigma2 * 4 / 10**12
    )
    assert unit == [Fraction(v) for v in MAXIMA_UNIT]
    assert time == [Fraction(v) for v in MAXIMA_TIME]
    gaps = [row[4] - _sum(lt * v for lt, v in zip(time, row[:4], strict=False)) for row in y]
    tau = gaps[0] - _sum(w * g for w, g in zip(unit, gaps[1:], strict=True))
    assert tau == Fraction(MAXIMA_TAU)
    fit = synthetic_did(np.array(TINY, dtype=float), flags(5, 1), 1)
    np.testing.assert_allclose(fit.unit_weights, [float(w) for w in unit], rtol=0, atol=1e-12)
    np.testing.assert_allclose(fit.time_weights, [float(w) for w in time], rtol=0, atol=1e-12)
    assert fit.effect == pytest.approx(float(tau), rel=1e-12, abs=0.0)


@pytest.mark.skipif(shutil.which("maxima") is None, reason="maxima is not on PATH")
def test_maxima_s_enumeration_passes_every_step() -> None:
    script = Path(__file__).parents[1] / "validation" / "sdid_simplex_weights.mac"
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
    assert [f"STEP {k} (PASS)" in out for k in range(1, 6)] == [True] * 5


# ----------------------------------------------------------------------------- the estimate's forms


def _twfe(y: np.ndarray, treated: np.ndarray, unit: np.ndarray, period: np.ndarray) -> float:
    """The treatment's coefficient in the weighted regression on unit and period effects."""
    n, t = y.shape
    rows, weights = [], []
    for i in range(n):
        for s in range(t):
            row = np.zeros(n + t + 1)
            row[i] = row[n + s] = 1.0
            row[-1] = float(treated[i] and s >= t - POST)
            rows.append(row)
            weights.append(unit[i] * period[s])
    root = np.sqrt(np.array(weights))
    design = np.array(rows) * root[:, None]
    coefficients = np.linalg.lstsq(design, y.ravel() * root, rcond=None)[0]
    return float(coefficients[-1])


def _cell_weights(treated: np.ndarray, omega: np.ndarray, lam: np.ndarray) -> tuple:
    unit = np.where(treated, 1.0 / treated.sum(), 0.0)
    unit[~treated] = omega
    return unit, np.concatenate([lam, np.full(POST, 1.0 / POST)])


@pytest.mark.parametrize("treated", [1, 3])
def test_the_weighted_regression_is_the_double_difference(treated: int) -> None:
    y = null_panel(np.random.default_rng(3), treated)
    y[:treated, PRE:] += 1.5
    tr = flags(y.shape[0], treated)
    fit = synthetic_did(y, tr, POST)
    unit, period = _cell_weights(tr, fit.unit_weights, fit.time_weights)
    assert _twfe(y, tr, unit, period) == pytest.approx(fit.effect, rel=1e-10, abs=0.0)


def test_with_every_weight_equal_it_is_plain_difference_in_differences() -> None:
    y = null_panel(np.random.default_rng(4), 2)
    tr = flags(y.shape[0], 2)
    fit = synthetic_did(y, tr, POST, weights="did")
    plain = (y[tr, PRE:].mean() - y[tr, :PRE].mean()) - (y[~tr, PRE:].mean() - y[~tr, :PRE].mean())
    assert fit.effect == pytest.approx(plain, rel=0.0, abs=1e-12)
    unit, period = _cell_weights(tr, np.full(CONTROLS, 1 / CONTROLS), np.full(PRE, 1 / PRE))
    assert _twfe(y, tr, unit, period) == pytest.approx(plain, rel=0.0, abs=1e-12)


def test_on_factors_without_noise_the_effect_is_recovered() -> None:
    """Each geo a level and two factors' loadings; the post-weeks' factors are convex
    combinations of the pre-weeks', so some time weights balance every geo at once, and the treated
    geo is a convex combination of controls plus a level of its own."""
    rng = np.random.default_rng(5)
    pre_factors = rng.normal(0.0, 3.0, (PRE, 2))
    mix = rng.dirichlet(np.ones(PRE), POST)
    factors = np.vstack([pre_factors, mix @ pre_factors])
    controls = (
        rng.normal(0.0, 4.0, CONTROLS)[:, None] + rng.normal(0.0, 1.0, (CONTROLS, 2)) @ factors.T
    )
    combination = rng.dirichlet(np.ones(CONTROLS))
    treated = combination @ controls + 17.0
    treated[PRE:] += 2.5
    y = np.vstack([treated, controls])
    fit = synthetic_did(y, flags(y.shape[0], 1), POST)
    assert fit.effect == pytest.approx(2.5, rel=0.0, abs=1e-10)


# ------------------------------------------------------------------------------------- inference


def _mc_bound(replications: int) -> float:
    return 0.05 + 2.0 * math.sqrt(0.05 * 0.95 / replications)


@pytest.mark.parametrize("treated", [1, 2, 3, 5])
def test_with_no_effect_the_placebo_test_holds_its_size(treated: int) -> None:
    """20 controls, 10 weeks before and 5 after, a level of sd 2 a geo, a shock a week and unit
    noise: 2000 panels each, and placebos read at every assignment of one treated geo, else 50
    drawn."""
    replications = 2000
    rng = np.random.default_rng([6, treated])
    rejected = 0
    for _ in range(replications):
        _, reading = read(
            null_panel(rng, treated), treated, replications=50, seed=int(rng.integers(2**31))
        )
        rejected += reading.p_value < 0.05
    assert rejected / replications <= _mc_bound(replications)


def test_with_few_controls_the_placebo_test_rejects_at_its_level_not_below_it() -> None:
    """Seven controls and one treated geo, every placebo read: the time weights' support is near
    the controls' count in every panel, and the floor at half the controls' count less 1 keeps the
    test at its level rather than at the t of 1 degree of freedom, which rejected 0.019 of these
    panels. The band is the binomial's three standard deviations either side of 0.05 over 2000
    panels."""
    replications, controls = 2000, 7
    rng = np.random.default_rng(16)
    rejected = 0
    for _ in range(replications):
        _, reading = read(null_panel(rng, 1, controls), 1)
        assert reading.df >= (controls - 1) / 2
        rejected += reading.p_value < 0.05
    band = 3.0 * math.sqrt(0.05 * 0.95 / replications)
    assert 0.05 - band <= rejected / replications <= 0.05 + band


@pytest.mark.parametrize("treated", [2, 3, 5])
def test_with_no_effect_the_jackknife_rejects_at_or_below_its_level(treated: int) -> None:
    replications = 2000
    rng = np.random.default_rng([7, treated])
    rejected = 0
    for _ in range(replications):
        _, reading = read(null_panel(rng, treated), treated, variance="jackknife")
        rejected += reading.p_value < 0.05
    assert rejected / replications <= _mc_bound(replications)


def test_the_jackknife_leaves_out_every_geo_controls_included() -> None:
    """The paper's Algorithm 3 with the weights held: treated geos alike in every week leave the
    treated geos' own terms at 0, and the controls' terms still read the noise."""
    rng = np.random.default_rng(8)
    y = null_panel(rng, 3)
    y[1] = y[2] = y[0]
    tr = flags(y.shape[0], 3)
    fit = synthetic_did(y, tr, POST)
    omega = np.concatenate([np.zeros(3), fit.unit_weights])
    effects = []
    for left in range(y.shape[0]):
        keep = np.arange(y.shape[0]) != left
        w = omega[keep][~tr[keep]]
        gaps = y[keep, PRE:].mean(axis=1) - y[keep, :PRE] @ fit.time_weights
        effects.append(gaps[tr[keep]].mean() - (w / w.sum()) @ gaps[~tr[keep]])
    n = y.shape[0]
    expected = math.sqrt((n - 1) / n * np.sum((np.array(effects) - fit.effect) ** 2))
    _, reading = read(y, 3, variance="jackknife")
    assert reading.se == pytest.approx(3 * POST * expected, rel=1e-12, abs=0.0)
    assert reading.se > 0.0
    assert reading.replications == n


def test_with_one_treated_geo_every_placebo_is_read_once() -> None:
    y = null_panel(np.random.default_rng(9), 1)
    _, reading = read(y, 1, replications=200)
    effects = [synthetic_did(y[1:], np.arange(CONTROLS) == i, POST).effect for i in range(CONTROLS)]
    assert reading.replications == CONTROLS
    assert reading.se == pytest.approx(POST * float(np.std(effects)), rel=1e-12, abs=0.0)
    kept = CONTROLS - np.count_nonzero(reading.time_weights)
    assert reading.df == max(kept, (CONTROLS - 1) / 2)


def test_both_errors_are_pinned_on_a_small_panel() -> None:
    """Two treated geos and five controls: the ten placebos, every pair of controls, and the seven
    leave-one-out estimates, controls included, at the values the code read when they were set."""
    y = null_panel(np.random.default_rng(17), 2, 5)
    placebo = read(y, 2)[1]
    effects = [
        synthetic_did(y[2:], np.isin(np.arange(5), pair), POST).effect
        for pair in combinations(range(5), 2)
    ]
    assert placebo.replications == 10
    assert placebo.se == pytest.approx(2 * POST * float(np.std(effects)), rel=1e-12, abs=0.0)
    assert placebo.se == pytest.approx(4.302637091355319, rel=1e-12, abs=0.0)
    assert placebo.df == 2.0  # 5 controls less 4 time weights is 1, below (5 - 1) / 2
    jackknife = read(y, 2, variance="jackknife")[1]
    assert jackknife.replications == 7
    assert jackknife.se == pytest.approx(5.133518943059618, rel=1e-12, abs=0.0)
    assert jackknife.df == 1.0  # two treated geos leave one out of two


def test_random_placebos_are_drawn_from_a_child_of_the_seed_s_stream() -> None:
    """Three treated geos among 20 controls have 1140 assignments, so 200 are drawn: from
    `SeedSequence(seed).spawn(1)[0]`, never from `default_rng(seed)`, the caller's own stream."""
    y = null_panel(np.random.default_rng(11), 3)
    _, reading = read(y, 3, seed=5)
    rng = np.random.default_rng(np.random.SeedSequence(5).spawn(1)[0])
    effects = []
    for _ in range(200):
        placebo = np.zeros(CONTROLS, dtype=bool)
        placebo[rng.choice(CONTROLS, 3, replace=False)] = True
        effects.append(synthetic_did(y[3:], placebo, POST).effect)
    assert reading.replications == 200
    assert reading.se == pytest.approx(3 * POST * float(np.std(effects)), rel=1e-12, abs=0.0)
    # the panel's placebo error at the default seed, 4.3610 from the caller's own stream
    assert read(y, 3)[1].se == pytest.approx(4.931520045634167, rel=1e-12, abs=0.0)


def _held_out(y: np.ndarray, treated: np.ndarray) -> float:
    """The estimator on the pre-period alone, its last `POST` weeks a pseudo post-period: the
    root mean square of its per-period gaps there."""
    pre = y[:, :PRE]
    fit = synthetic_did(pre, treated, POST)
    path = pre[treated].mean(0) - fit.unit_weights @ pre[~treated]
    gaps = path[PRE - POST :] - path[: PRE - POST] @ fit.time_weights
    return math.sqrt(float(np.mean(gaps**2)))


def test_unit_placebos_studentise_each_effect_by_its_held_out_error() -> None:
    """One treated geo among 20 controls: the design could have treated any of the 21 geos, so
    each is read treated against the other 20, the observed one among them, each effect over its
    pre-period out-of-sample error; the p-value is the share at least as far from 0 as the
    observed one, and the interval the lift plus or minus its own error times the `ceil(0.05 x
    21) = 2`nd largest. The lift is the default reading's."""
    y = null_panel(np.random.default_rng(23), 1)
    estimate, reading = read(y, 1, variance="unit_placebo")
    effects, errors = [], []
    for i in range(CONTROLS + 1):
        placebo = np.arange(CONTROLS + 1) == i
        effects.append(synthetic_did(y, placebo, POST).effect)
        errors.append(_held_out(y, placebo))
    t = np.abs(np.array(effects) / np.array(errors))
    assert reading.replications == CONTROLS + 1
    assert reading.variance == "unit_placebo"
    assert math.isinf(reading.df)
    assert reading.lift == read(y, 1)[1].lift
    assert reading.p_value == pytest.approx(float(np.mean(t >= t[0])), rel=1e-12, abs=0.0)
    half = float(np.sort(t)[-2]) * errors[0] * POST
    assert reading.upper - reading.lift == pytest.approx(half, rel=1e-9, abs=0.0)
    assert reading.lift - reading.lower == pytest.approx(half, rel=1e-9, abs=0.0)
    assert estimate.se == pytest.approx(half / stats.norm.ppf(0.975), rel=1e-9, abs=0.0)
    assert reading.se == pytest.approx(POST * float(np.std(effects)), rel=1e-12, abs=0.0)


def test_over_every_treated_geo_it_could_have_drawn_unit_placebos_reject_at_their_level() -> None:
    """One panel with no effect, read with each of its 21 geos as the treated one: each reading
    reads the same 21 studentised effects, so at most `ceil(0.05 x 21) - 1 = 1` rejects, whatever
    the noise, and each verdict is its p-value's."""
    y = null_panel(np.random.default_rng(37), 1)
    rejected = 0
    for g in range(CONTROLS + 1):
        geo = frozenset({Geo(f"g{g:02d}")})
        _, reading = read_sdid(
            panel(y), geo, START, END, spend=SPEND, planned=PLANNED, variance="unit_placebo"
        )
        assert (reading.p_value < 0.05) == (reading.verdict != "inconclusive")
        rejected += reading.p_value < 0.05
    assert rejected <= 1


@pytest.mark.parametrize("shift", [-6.0, 0.0, 6.0])
def test_a_unit_placebo_verdict_is_its_interval_s_sign_and_its_p_value_s(shift: float) -> None:
    """Three treated geos among 20 controls, 200 placebos drawn: the interval holds 0 exactly
    where the p-value is at least 5 %, at no effect and at a fall or a rise of 6 a geo-week
    (read as one over 30 panels each)."""
    y = null_panel(np.random.default_rng(29), 3)
    y[:3, PRE:] += shift
    _, reading = read(y, 3, variance="unit_placebo", seed=3)
    assert reading.replications == 201
    assert (reading.p_value < 0.05) == (reading.verdict != "inconclusive")
    assert reading.verdict == {-6.0: "down", 0.0: "inconclusive", 6.0: "up"}[shift]


def test_unit_placebos_refuse_what_they_cannot_read() -> None:
    """One treated geo among 11 geos holds 11 assignments, too few to reject at 5 %; a
    pre-period of 6 weeks holds no pseudo post-period of 5 after 2 weeks."""
    y = null_panel(np.random.default_rng(31), 1, 10)
    with pytest.raises(ValueError, match="too few for the test to reject"):
        read(y, 1, variance="unit_placebo")
    short = null_panel(np.random.default_rng(31), 1)[:, PRE - 6 :]
    with pytest.raises(ValueError, match="no pseudo post-period"):
        read_sdid(
            panel(short),
            treated_geos(1),
            panel(short).dates[6],
            panel(short).dates[-1],
            spend=SPEND,
            planned=PLANNED,
            variance="unit_placebo",
        )


# ------------------------------------------------------------------------------- the handed-on test


def test_the_reading_is_handed_on_as_a_lift_estimate() -> None:
    y = null_panel(np.random.default_rng(10), 2)
    y[:2, PRE:] += 3.0
    estimate, reading = read(y, 2)
    fit = synthetic_did(y, flags(y.shape[0], 2), POST)
    assert reading.method == "sdid"
    assert reading.variance == "placebo"
    assert estimate.lift == reading.lift == pytest.approx(2 * POST * fit.effect, rel=1e-15, abs=0.0)
    widening = stats.t.ppf(0.975, reading.df) / stats.norm.ppf(0.975)
    assert estimate.se == pytest.approx(reading.se * widening, rel=1e-15, abs=0.0)
    assert estimate.readout == POST
    assert estimate.geos == treated_geos(2)
    assert estimate.end == END
    np.testing.assert_array_equal(estimate.spend, SPEND)
    assert reading.controls == tuple(Geo(f"g{i:02d}") for i in range(2, 2 + CONTROLS))
    np.testing.assert_array_equal(reading.pre, panel(y).dates[:PRE])
    assert reading.lower < reading.lift < reading.upper
    assert reading.verdict == "up"
    assert reading.p_value < 0.05


@pytest.mark.parametrize("variance", ["placebo", "jackknife"])
def test_the_estimate_s_error_is_widened_to_the_t_and_plain_at_the_normal(variance) -> None:
    """At the t the estimate's normal 95 % interval is the reading's t interval, as the geo-test
    readers' is; at the normal it is the reading's own."""
    y = null_panel(np.random.default_rng(10), 3)
    estimate, reading = read(y, 3, variance=variance)
    assert math.isfinite(reading.df)
    ratio = stats.t.ppf(0.975, reading.df) / stats.norm.ppf(0.975)
    assert estimate.se == pytest.approx(reading.se * ratio, rel=1e-15, abs=0.0)
    assert estimate.se > reading.se
    half = stats.norm.ppf(0.975) * estimate.se
    assert reading.upper - reading.lift == pytest.approx(half, rel=1e-12, abs=0.0)
    normal, plain = read(y, 3, variance=variance, quantile="normal")
    assert normal.se == plain.se == reading.se


def test_each_reader_hands_on_its_error_widened_to_its_own_t() -> None:
    """One panel, read by `read_did` and by `read_sdid`: each estimate's error is its reading's
    times `t_df(0.975) / z(0.975)` at its own degrees of freedom."""
    pre, geos = 30, 3 + CONTROLS
    rng = np.random.default_rng(14)
    y = (
        50.0
        + rng.normal(0.0, 2.0, geos)[:, None]
        + rng.normal(0.0, 1.0, pre + POST)[None]
        + rng.normal(0.0, 1.0, (geos, pre + POST))
    )
    dates = np.datetime64("2025-01-06") + np.timedelta64(7, "D") * np.arange(pre + POST)
    start, end = dates[pre], dates[-1]
    spend, planned = np.full(pre + POST, 2.0), np.full(pre + POST, 1.0)
    design = GeoTest(
        Channel("tv"),
        tuple(treated_geos(3)),
        tuple(Geo(f"g{i:02d}") for i in range(3, geos)),
        start,
        end,
        spend,
        planned,
        POST,
    )
    readings = [
        read_did(panel(y), design),
        read_sdid(panel(y), treated_geos(3), start, end, spend=spend, planned=planned),
    ]
    for estimate, reading in readings:
        ratio = stats.t.ppf(0.975, reading.df) / stats.norm.ppf(0.975)
        assert estimate.se == pytest.approx(reading.se * ratio, rel=1e-15, abs=0.0)


@pytest.mark.parametrize("shift", [-3.0, 0.0, 3.0])
def test_the_verdict_is_the_interval_s_sign_and_the_p_value_s(shift: float) -> None:
    y = null_panel(np.random.default_rng(11), 3)
    y[:3, PRE:] += shift
    for quantile in ("t", "normal"):
        _, reading = read(y, 3, quantile=quantile)
        expected = "up" if reading.lower > 0 else "down" if reading.upper < 0 else "inconclusive"
        assert reading.verdict == expected
        assert (reading.p_value < 0.05) == (reading.verdict != "inconclusive")
    assert read(y, 3, quantile="normal")[1].df == math.inf
    assert read(y, 3, quantile="normal")[1].upper < read(y, 3)[1].upper
    assert {0.0: "inconclusive", 3.0: "up", -3.0: "down"}[shift] == read(y, 3)[1].verdict


def test_named_controls_are_read_and_a_geo_in_neither_group_is_not() -> None:
    """A geo outside the test with a missing week refuses the default, which reads every untreated
    geo as a control, and is left unread with the controls named, as the panel without it reads."""
    y = null_panel(np.random.default_rng(15), 2)
    y[:2, PRE:] += 2.0
    gap = np.vstack([y, np.full((1, PRE + POST), np.nan)])
    gap[-1, 0] = 3.0
    with pytest.raises(ValueError, match="balanced"):
        read(gap, 2)
    controls = [Geo(f"g{i:02d}") for i in range(2, 2 + CONTROLS)]
    named, named_reading = read(gap, 2, control=controls[::-1])
    whole, whole_reading = read(y, 2)
    assert (named.lift, named.se) == (whole.lift, whole.se)
    assert named_reading.controls == whole_reading.controls == tuple(controls)
    fewer = read(y, 2, control=controls[:12], variance="jackknife")[1]
    assert fewer.controls == tuple(controls[:12])
    assert fewer.unit_weights.size == 12


# ------------------------------------------------------------------------------------- refusals


def test_what_cannot_be_read_is_refused() -> None:
    y = null_panel(np.random.default_rng(12), 2)
    with pytest.raises(ValueError, match="no unit is a control"):
        synthetic_did(y, np.ones(y.shape[0], dtype=bool), POST)
    with pytest.raises(ValueError, match="no unit is treated"):
        synthetic_did(y, np.zeros(y.shape[0], dtype=bool), POST)
    with pytest.raises(ValueError, match="pre-period of 1 period"):
        synthetic_did(y, flags(y.shape[0], 2), y.shape[1] - 1)
    with pytest.raises(ValueError, match="no unit is a control"):
        read_sdid(
            panel(y),
            frozenset(panel(y).geos),
            START,
            END,
            spend=SPEND,
            planned=PLANNED,
        )
    with pytest.raises(ValueError, match="pre-period of 1 period"):
        read_sdid(
            panel(y),
            treated_geos(2),
            panel(y).dates[1],
            END,
            spend=SPEND,
            planned=PLANNED,
        )
    with pytest.raises(ValueError, match="at least two treated units"):
        read(null_panel(np.random.default_rng(13), 1), 1, variance="jackknife")
    with pytest.raises(ValueError, match="at least two control units"):
        read(null_panel(np.random.default_rng(13), 2, controls=1), 2, variance="jackknife")
    with pytest.raises(ValueError, match="more controls than treated"):
        read(y[:4], 2)
    with pytest.raises(ValueError, match="not in"):
        read(y, 2, alpha=1.0)
    with pytest.raises(ValueError, match="not in the panel"):
        read_sdid(panel(y), {Geo("nowhere")}, START, END, spend=SPEND, planned=PLANNED)
    with pytest.raises(ValueError, match="control geos"):
        read(y, 2, control={Geo("nowhere")})
    with pytest.raises(ValueError, match="both treated and controls"):
        read(y, 2, control={Geo("g01"), Geo("g05")})
    with pytest.raises(ValueError, match="in that order"):
        read_sdid(panel(y), treated_geos(2), END, START, spend=SPEND, planned=PLANNED)
    with pytest.raises(ValueError, match="is not 'placebo' or 'jackknife'"):
        read(y, 2, variance="bootstrap")
    with pytest.raises(ValueError, match="is not 't' or 'normal'"):
        read(y, 2, quantile="z")
    missing = y.copy()
    missing[5, 3] = np.nan
    with pytest.raises(ValueError, match="balanced"):
        read(missing, 2)
