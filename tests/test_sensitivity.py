"""Sensitivity to an unobserved confounder (ADR 0061): Cinelli and Hazlett's statistics reproduce
their worked example, hold their closed forms and the in-sample bias identity on a ridge fit, and
the marginal sensitivity model's bound on a Bayesian fit's return is narrow on an exogenous world
and covers the return of a confounded one at the level that generated it, and on Track M v2's world
at its `Gamma*`."""

from __future__ import annotations

import functools
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from chc import ChannelSensitivity
from scipy import optimize, stats

from mixdyn import Cadence, Channel, Geo, MediaPanel, Revenue
from mixdyn.inference.sensitivity import (
    Benchmark,
    MsmSensitivity,
    RidgeSensitivity,
    _draws_means,
    _partial_r2,
    msm_sensitivity,
    ridge_sensitivity,
    robustness_value,
)
from mixdyn.models.bayes_mmm import BayesFit, BayesMedia, BayesSpec, TanhCurve, fit_bayes
from mixdyn.models.ridge_mmm import Control, FourierSeason, LinearTrend, RidgeSpec, fit_ridge
from mixdyn.transforms import carry_over, half_saturation
from ridge_worlds import NATIONAL, panel_of, saturated, world
from track_m2_worlds import world as track_m2_world

# ----------------------------------------------------------------------- the paper's worked example

# Cinelli and Hazlett (2020), Section 5: peacefactor on directlyharmed with age, farmer_dar,
# herder_dar, pastvoted, hhsize_darfur, female and village fixed effects, on the public Darfur
# survey of Hazlett (2020, Journal of Conflict Resolution). The t-statistics are the least squares'
# on those data: the treatment's and female's in the outcome's regression, and female's in the
# treatment's on the covariates.
DARFUR_ESTIMATE = 0.09731581928498363
DARFUR_SE = 0.023256537809860974
DARFUR_T = 4.184449984800441
DARFUR_DOF = 783
FEMALE_T_OUTCOME = -9.788837042333014
FEMALE_T_TREATMENT = -2.6804478643239458
# their bounds at 1, 2 and 3 times female, as printed to four decimals: R2dz.x, R2yz.dx, the
# adjusted estimate, its standard error, and its 95 % interval
DARFUR_BOUNDS = {
    1: (0.0092, 0.1246, 0.0752, 0.0219, 0.0323, 0.1182),
    2: (0.0183, 0.2493, 0.0529, 0.0204, 0.0130, 0.0929),
    3: (0.0275, 0.3741, 0.0304, 0.0187, -0.0062, 0.0670),
}


def _darfur() -> RidgeSensitivity:
    return RidgeSensitivity(
        channel="directlyharmed",
        coefficient=DARFUR_ESTIMATE,
        estimate=DARFUR_ESTIMATE,
        se=DARFUR_SE,
        dof=DARFUR_DOF,
        level=0.95,
        partial_r2=_partial_r2(DARFUR_T, DARFUR_DOF),
        robustness=robustness_value(DARFUR_T, DARFUR_DOF),
        robustness_half=robustness_value(DARFUR_T, DARFUR_DOF, 0.5),
        robustness_interval=robustness_value(DARFUR_T, DARFUR_DOF, 1.0, 0.05),
        robustness_interval_half=robustness_value(DARFUR_T, DARFUR_DOF, 0.5, 0.05),
        benchmarks=(
            Benchmark(
                "female",
                _partial_r2(FEMALE_T_TREATMENT, DARFUR_DOF + 1),
                _partial_r2(FEMALE_T_OUTCOME, DARFUR_DOF),
            ),
        ),
    )


def test_the_statistics_are_the_paper_s() -> None:
    """Their Table 2: R2(Y ~ D | X) 2.2 %, RV_1 13.9 %, RV_{1, 0.05} 7.6 %, each to its printed
    digit; and the estimate's t over its standard error."""
    darfur = _darfur()
    assert pytest.approx(DARFUR_ESTIMATE / DARFUR_SE, rel=1e-12, abs=0.0) == DARFUR_T
    assert round(100.0 * darfur.partial_r2, 1) == 2.2
    assert round(100.0 * darfur.robustness, 1) == 13.9
    assert round(100.0 * darfur.robustness_interval, 1) == 7.6


@pytest.mark.parametrize("k", [1, 2, 3])
def test_the_benchmark_bounds_are_the_paper_s(k: int) -> None:
    """Each bound at `k` times female within a unit of its last printed digit: the 3x interval's
    lower end reads -0.006253 here and is printed -0.0062."""
    bound = _darfur().benchmark("female", kd=k)
    got = (bound.r2_treatment, bound.r2_outcome, bound.estimate, bound.se, bound.lower, bound.upper)
    np.testing.assert_allclose(got, DARFUR_BOUNDS[k], rtol=0.0, atol=1e-4)
    assert bound.label == f"{k}x female"


# --------------------------------------------------------------------------------- the closed forms


@pytest.mark.parametrize("q", [1.0, 0.5])
@pytest.mark.parametrize(("t", "dof"), [(4.18, 783), (2.5, 40), (12.0, 30)])
def test_a_confounder_at_the_robustness_value_moves_the_estimate_by_q(
    t: float, dof: int, q: float
) -> None:
    """At equal shares of `RV_q` the estimate moves by `q` of itself, and at `RV_{q, alpha}` the
    interval's near end reaches `(1 - q)` of the estimate, where the equal shares bind."""
    se = 0.1
    rv = robustness_value(t, dof, q)
    s = RidgeSensitivity("d", t * se, t * se, se, dof, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0, ())
    assert s.bound(rv, rv).estimate == pytest.approx((1.0 - q) * t * se, rel=1e-10, abs=1e-12)
    rv_interval = robustness_value(t, dof, q, 0.05)
    critical = stats.t.ppf(0.975, dof - 1) / math.sqrt(dof - 1)
    if critical < q * t / math.sqrt(dof) < 1.0 / critical:
        lower = s.bound(rv_interval, rv_interval).lower
        assert lower == pytest.approx((1.0 - q) * t * se, rel=1e-10, abs=1e-12)


def _box_reach(t: float, dof: int, q: float, alpha: float) -> float:
    """`RV_{q, alpha}` by brute force: the least side `r` of the box of both shares within which
    some confounder brings the interval's near end to `(1 - q)` of the estimate."""
    critical = stats.t.ppf(1.0 - alpha / 2.0, dof - 1) * math.sqrt(dof / (dof - 1))

    def moved(r: float) -> float:
        # the treatment's share binds at the box's side; the outcome's is searched inside it
        def end(x: float) -> float:
            return -(
                math.sqrt(dof * x * r / (1.0 - r)) + critical * math.sqrt((1.0 - x) / (1.0 - r))
            )

        best = optimize.minimize_scalar(
            end, bounds=(0.0, r), method="bounded", options={"xatol": 1e-14}
        )
        return max(-best.fun, -end(r), -end(0.0)) - q * abs(t)

    return optimize.brentq(moved, 1e-12, 1.0 - 1e-12, xtol=1e-14)


@pytest.mark.parametrize(
    ("t", "dof", "q"),
    [(4.18, 783, 1.0), (4.18, 783, 0.5), (12.0, 30, 1.0), (30.0, 10, 1.0), (60.0, 12, 0.5)],
)
def test_the_interval_s_robustness_value_is_the_least_box_that_reaches_it(
    t: float, dof: int, q: float
) -> None:
    """Both branches of `RV_{q, alpha}`, the equal shares' and the one past `f_q = 1 / f*` where
    the outcome's share that moves the interval most is below the box's side, are the least box
    found by search."""
    assert robustness_value(t, dof, q, 0.05) == pytest.approx(
        _box_reach(t, dof, q, 0.05), rel=1e-6, abs=0.0
    )


def test_an_estimate_its_interval_does_not_clear_needs_no_confounder() -> None:
    assert robustness_value(1.5, 100, 1.0, 0.05) == 0.0


@pytest.mark.skipif(shutil.which("maxima") is None, reason="maxima is not on PATH")
def test_maxima_s_closed_forms_pass_every_step() -> None:
    script = Path(__file__).parents[1] / "validation" / "sensitivity.mac"
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


@pytest.mark.parametrize("gamma", [1.1, 1.5, 2.0, 3.0])
def test_chc_s_factor_is_the_bias_of_the_sharp_share(gamma: float) -> None:
    """`kappa(Gamma)`, the bias per unit of the residuals' spread ratio, is
    `sqrt(R2 / (1 - R2))` at the share `(1 - 1/Gamma)^2`, and `level_at` inverts it."""
    share = (1.0 - 1.0 / gamma) ** 2
    kappa = ChannelSensitivity.factor(gamma)
    assert kappa == pytest.approx(math.sqrt(share / (1.0 - share)), rel=1e-14, abs=0.0)
    assert ChannelSensitivity.level_at(kappa) == pytest.approx(gamma, rel=1e-12, abs=0.0)


# ----------------------------------------------------------------------------------- on a ridge fit


def _ridge_world(seed: int = 5) -> tuple[MediaPanel, RidgeSpec, np.ndarray]:
    """`ridge_worlds.world`'s three channels and a price control, with a confounder `u` the model
    does not see that moves both c0's saturated spend and the KPI: an AR(1) of coefficient 0.9,
    persistent as c0's adstock is, that c0's spend rises with."""
    spend, media, beta = world(seed)
    rng = np.random.default_rng(seed + 100)
    weeks = spend.shape[0]
    u = np.zeros(weeks)
    for t in range(1, weeks):
        u[t] = 0.9 * u[t - 1] + rng.normal(0.0, 0.5)
    spend[:, 0] *= np.exp(0.8 * u)
    price = rng.normal(10.0, 1.0, weeks) + 0.5 * u
    columns = [
        saturated(m, half_saturation(m.hill, carry_over(spend[:, j], m.adstock)), spend[:, j])
        for j, m in enumerate(media)
    ]
    kpi = 1000.0 + np.column_stack(columns) @ beta - 20.0 * price + 40.0 * u
    kpi += rng.normal(0.0, 30.0, weeks)
    panel = panel_of(kpi, spend, {"price": price})
    spec = RidgeSpec(media=tuple(media), controls=(Control("price"),), penalty=1e-4)
    return panel, spec, u


def test_the_ridge_s_statistics_are_the_least_squares_on_its_design() -> None:
    """The estimate, its standard error and the partial R^2 are the least squares' on the fit's
    own design, the channel's partial R^2 its residuals' share by Frisch-Waugh-Lovell, and the
    coefficient the ridge fit's."""
    panel, spec, _ = _ridge_world()
    fit = fit_ridge(panel, NATIONAL, spec)
    s = ridge_sensitivity(fit, "c0")
    x = np.column_stack([np.ones(fit.y.size), fit.design])
    coefficients, *_ = np.linalg.lstsq(x, fit.y, rcond=None)
    residual = fit.y - x @ coefficients
    dof = fit.y.size - x.shape[1]
    se = np.sqrt(residual @ residual / dof * np.diag(np.linalg.inv(x.T @ x)))
    assert s.coefficient == fit.coefficients["c0"]
    assert s.dof == dof
    assert s.estimate == pytest.approx(coefficients[1], rel=1e-10, abs=0.0)
    assert s.se == pytest.approx(se[1], rel=1e-10, abs=0.0)

    def partial(target: np.ndarray, column: int, others: np.ndarray) -> float:
        def left(v: np.ndarray) -> np.ndarray:
            return v - others @ np.linalg.lstsq(others, v, rcond=None)[0]

        a, b = left(target), left(x[:, column])
        return float((a @ b) ** 2 / ((a @ a) * (b @ b)))

    assert s.partial_r2 == pytest.approx(partial(fit.y, 1, np.delete(x, 1, axis=1)), rel=1e-10)
    price = 1 + len(spec.media)
    (b,) = s.benchmarks
    assert b.control == "price"
    assert b.r2_outcome == pytest.approx(
        partial(fit.y, price, np.delete(x, price, axis=1)), rel=1e-10
    )
    assert b.r2_treatment == pytest.approx(
        partial(x[:, 1], price, np.delete(x, [1, price], axis=1)), rel=1e-10
    )
    t = s.estimate / s.se
    assert s.robustness == robustness_value(t, dof)
    assert s.robustness_half == robustness_value(t, dof, 0.5)
    assert s.robustness_interval == robustness_value(t, dof, 1.0, 0.1)
    assert s.robustness_interval_half == robustness_value(t, dof, 0.5, 0.1)


def test_the_bound_at_the_confounder_s_own_shares_is_the_bias_it_made() -> None:
    """In sample the omitted-variable bias is the bound's own: the least squares with the
    confounder added moves the channel's coefficient by exactly the bias the bound gives at the
    confounder's partial R^2 with the KPI and with the channel."""
    panel, spec, u = _ridge_world()
    fit = fit_ridge(panel, NATIONAL, spec)
    s = ridge_sensitivity(fit, "c0")
    x = np.column_stack([np.ones(fit.y.size), fit.design])
    long = np.linalg.lstsq(np.column_stack([x, u]), fit.y, rcond=None)[0][1]

    def left(v: np.ndarray, others: np.ndarray) -> np.ndarray:
        return v - others @ np.linalg.lstsq(others, v, rcond=None)[0]

    def share(a: np.ndarray, b: np.ndarray) -> float:
        return float((a @ b) ** 2 / ((a @ a) * (b @ b)))

    r2_outcome = share(left(fit.y, x), left(u, x))
    r2_treatment = share(left(x[:, 1], np.delete(x, 1, axis=1)), left(u, np.delete(x, 1, axis=1)))
    bound = s.bound(r2_outcome, r2_treatment)
    assert abs(s.estimate - long) == pytest.approx(bound.bias, rel=1e-9, abs=0.0)
    assert bound.bias > 0.05 * abs(s.estimate)


def test_the_ridge_analysis_refuses_what_it_cannot_read() -> None:
    panel, spec, _ = _ridge_world()
    fit = fit_ridge(panel, NATIONAL, spec)
    with pytest.raises(ValueError, match="no channel"):
        ridge_sensitivity(fit, "price")
    with pytest.raises(ValueError, match="level"):
        ridge_sensitivity(fit, "c0", level=1.0)
    s = ridge_sensitivity(fit, "c0")
    with pytest.raises(ValueError, match="shares"):
        s.bound(0.5, 1.0)
    with pytest.raises(ValueError, match="controls"):
        s.benchmark("c1")


# -------------------------------------------------------------------------------- on a Bayesian fit

GEO = Geo("national")
_RETENTION, _HALF, _SLOPE, _BETA = (0.5, 0.3), (100.0, 80.0), (1.5, 1.0), (300.0, 150.0)
_ON = 0.7
_SPREAD = 0.6


def _confounded(seed: int, push: float, effect: float) -> tuple[MediaPanel, float, float]:
    """104 weeks of two channels in the Bayesian model's own form, normalised geometric adstocks
    into Hills, and a stationary AR(1) driver `u` of coefficient 0.8 and variance 1 the model does
    not see. tv runs in 70 % of the weeks, its log spend `push u` plus noise of standard deviation
    0.6; search's spend is flighted gamma. The KPI is the channels' effect, `effect u` and noise of
    standard deviation 5. Returns the panel, tv's return on its spend over the weeks, and the level
    `Gamma = 1 / (1 - rho)` the world confounds tv at, `rho` the maximal correlation of `u` and
    tv's spend: `sqrt(0.7)` times their logs' correlation in the weeks it runs, which are jointly
    normal (CHC's reading of a Gaussian lever)."""
    rng = np.random.default_rng(seed)
    weeks = 104
    u = np.empty(weeks)
    u[0] = rng.normal()
    for t in range(1, weeks):
        u[t] = 0.8 * u[t - 1] + rng.normal(0.0, 0.6)
    tv = 100.0 * np.exp(push * u + _SPREAD * rng.normal(size=weeks)) * (rng.random(weeks) < _ON)
    search = rng.gamma(4.0, 25.0, weeks) * (rng.random(weeks) < _ON)
    spend = np.column_stack([tv, search])
    columns = []
    for j in range(2):
        weights = _RETENTION[j] ** np.arange(weeks)
        stock = np.convolve(spend[:, j], weights / weights.sum())[:weeks]
        z = (stock / _HALF[j]) ** _SLOPE[j]
        columns.append(z / (1.0 + z))
    effects = np.column_stack(columns) @ np.array(_BETA)
    kpi = 1000.0 + effects + effect * u + rng.normal(0.0, 5.0, weeks)
    panel = MediaPanel(
        dates=np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(weeks),
        cadence=Cadence.WEEK,
        geos=(GEO,),
        channels=(Channel("tv"), Channel("search")),
        controls=(),
        kpi=kpi[None],
        kpi_unit=Revenue(),
        spend=spend[None],
        currency="EUR",
        control_values=np.zeros((1, weeks, 0)),
        filled=np.zeros((1, weeks), dtype=bool),
    )
    roi = _BETA[0] * columns[0].sum() / tv.sum()
    rho = math.sqrt(_ON) * push / math.hypot(push, _SPREAD)
    return panel, roi, 1.0 / (1.0 - rho)


@functools.cache
def _fit(seed: int, push: float, effect: float) -> tuple[BayesFit, float, float]:
    panel, roi, gamma = _confounded(seed, push, effect)
    spec = BayesSpec((BayesMedia("tv"), BayesMedia("search")))
    return fit_bayes(panel, GEO, spec, seed=seed, chains=2, warmup=300, draws=300), roi, gamma


def test_each_draw_s_mean_is_the_fit_s() -> None:
    """The draws' means the bound reads its noise from average to the fit's own estimate."""
    fit, _, _ = _fit(0, 0.0, 0.0)
    _, _, mean = _draws_means(fit)
    np.testing.assert_allclose(mean.mean(axis=0), fit.estimated, rtol=1e-12, atol=0.0)


def test_an_exogenous_world_s_return_is_robust() -> None:
    """Where nothing confounds tv, a confounder at `Gamma = 1.5` moves its return by less than a
    tenth of it, and only one past `Gamma = 3` brings the return's 90 % interval to 0."""
    fit, roi, _ = _fit(0, 0.0, 0.0)
    m = msm_sensitivity(fit, "tv", [1.0, 1.5])
    assert m.lower[0] < roi < m.upper[0]
    assert m.allowance[1] < 0.1 * m.roi
    assert m.gamma_star > 3.0
    assert msm_sensitivity(fit, "tv", [m.gamma_star]).lower[0] == pytest.approx(0.0, abs=1e-9)


def test_the_bound_at_the_world_s_level_covers_a_confounded_return() -> None:
    """Twenty worlds whose driver pushes tv's log spend by 0.6 of itself and the KPI by 25, at
    `Gamma = 2.45`: the bound at that level holds tv's return in at least 18 of them. The posterior
    interval without a confounder holds it in fewer, so the worlds do confound."""
    held, nominal = 0, 0
    for seed in range(20):
        fit, roi, gamma = _fit(seed, 0.6, 25.0)
        m: MsmSensitivity = msm_sensitivity(fit, "tv", [1.0, gamma])
        assert m.lower[1] <= m.lower[0]
        assert m.upper[1] >= m.upper[0]
        held += int(m.lower[1] <= roi <= m.upper[1])
        nominal += int(m.lower[0] <= roi <= m.upper[0])
    assert held >= 18
    assert nominal < held


def test_the_msm_analysis_refuses_what_it_cannot_read() -> None:
    fit, _, _ = _fit(0, 0.0, 0.0)
    with pytest.raises(ValueError, match="no channel"):
        msm_sensitivity(fit, "radio")
    with pytest.raises(ValueError, match="Gamma"):
        msm_sensitivity(fit, "tv", [0.9])
    with pytest.raises(ValueError, match="level"):
        msm_sensitivity(fit, "tv", level=0.0)


# --------------------------------------------------------------------------- on Track M v2's world


@functools.cache
def _track_m2_fit(seed: int) -> tuple[BayesFit, np.ndarray]:
    """His reference instance at `seed`, as the fixture holds it, fitted as the scorecard's arm
    reads an export, its tanh family, on 2 chains of 500 draws after 500 of warm-up."""
    w = track_m2_world("reference", seed)
    spec = BayesSpec(
        media=tuple(BayesMedia(c) for c in w.channels),
        controls=(Control("promotion", lower=0.0), Control("price")),
        trend=LinearTrend(3),
        season=FourierSeason(365.25 / 7.0, 3),
        saturation=TanhCurve(),
    )
    fit = fit_bayes(w.panel(), GEO, spec, seed=seed, chains=2, warmup=500, draws=500)
    return fit, w.roas


@pytest.mark.parametrize("seed", [0, 1])
def test_the_bound_at_gamma_star_holds_track_m2_s_returns(seed: int) -> None:
    """Heusch's reference instance, whose spend follows the sales over a sentiment no column
    holds: every channel's `Gamma*` is below 1.5, so the fit's returns survive only a weak
    confounder, and the bound at it holds each channel's true return, paid shopping's though its
    posterior median is twice it and at seed 1 its 90 % interval lies above it. The study of ADR
    0061's amendment reads ten seeds."""
    fit, roas = _track_m2_fit(seed)
    for channel, truth in zip(fit.spec.media, roas, strict=True):
        m = msm_sensitivity(fit, channel.channel, [1.0])
        assert 1.0 <= m.gamma_star < 1.5, channel
        (upper,) = msm_sensitivity(fit, channel.channel, [m.gamma_star]).upper
        assert 0.0 < truth <= upper, channel
