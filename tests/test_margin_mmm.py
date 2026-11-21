"""The margin regression (`mixdyn.models.margin_mmm`): on six of M2's worlds its margins, their
covariance and its diagnostics are the research code's, which found the rule; a kernel as long as
the history fits, with the spend before the history imputed and a burn-in of 13 weeks, where the
research code raised; the adstock with that imputation is the history's own from the kernel's
length on; a fit the data leave degenerate raises the typed error, never a margin that is not a
number; on a world made by the model, at a negligible prior, the fit finds its margins; a
history without the promotion or the price fits without their columns; and the prior, not the
data, sets the margin of a channel whose spend never moves, by the Schur complement of the
design's cross-products. The curved regression at an infinite half-saturation is the linear one;
each of its curves has slope 1 at the status quo, falls in the spend and rises by its integral;
its planning adstock is each channel's at its retention of the most weight; and it refuses a
status quo, a planned quarter or a half-saturation it does not read. Under a noise mixture the
curved margins, their covariance and the error models' posterior means are the research code's on
the seven worlds; each error model's fit is the generalised least squares under its covariance
inverted whole, and at an AR(1) alone Prais and Winsten's regression under a flat nuisance prior;
the evidence is the same on any common scale of the covariance; a nuisance column that is 0 on
the weeks read is dropped; the trend's knots are 52 weeks apart; a mixture of one error model
reports it, and at a negligible prior the mixture reads the errors a world was made with; nuisance
columns the weeks do not tell apart raise the typed error, a price that never moves, a price that
is a straight line in the week and a promotion flag that is its promoted weeks' dummies' sum
among them; a channel whose columns the nuisance
columns span has a margin of exactly 0, its prior's, within the square root of the rounding unit
of the span and not past it; without a mixture the fit is assembled as it was before one, to the
bit; by default the fit is the mixture's, and on the README's history its margins are the truth's
within two deviations and the same at any level of the sales, which moves the margins without a
mixture; and Maxima proves the error models' identities."""

from __future__ import annotations

import dataclasses
import itertools
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

from margin_worlds import LONG, NOISE, PARITY, arrays, name
from mixdyn.models import margin_mmm
from mixdyn.models.margin_mmm import (
    BURN_IN,
    HALF_SATURATION,
    PRESAMPLE,
    RETENTIONS,
    SPANNED,
    DegenerateFitError,
    NoiseMixture,
    adstock,
    fit_curved_margins,
    fit_margins,
    promoted_phases,
)

RELATIVE = 1e-9


@pytest.mark.parametrize("world", PARITY, ids=name)
def test_the_margins_and_their_covariance_are_the_research_codes(world):
    fit = fit_margins(*arrays(world), noise=None)
    research = world["research"]
    np.testing.assert_allclose(fit.mean, research["mean"], rtol=RELATIVE, atol=0.0)
    np.testing.assert_allclose(fit.cov, research["cov"], rtol=RELATIVE, atol=0.0)
    np.testing.assert_allclose(fit.retention, research["alpha_mean"], rtol=RELATIVE, atol=0.0)
    assert fit.top == tuple(research["alpha_top"])
    assert fit.top_weight == pytest.approx(research["w_top"], rel=RELATIVE, abs=0.0)
    assert fit.phi == pytest.approx(research["phi"], rel=RELATIVE, abs=0.0)
    assert (fit.rows, fit.columns, len(fit.phases)) == (
        research["n"],
        research["p"],
        research["events"],
    )
    assert fit.burn_in == world["kernel_length"] - 1


def test_a_kernel_as_long_as_the_history_fits_where_the_research_code_raised():
    """Family 14's kernel is its history's 156 weeks: the research code kept one week and its
    AR(1) coefficient was 0/0; here the burn-in is 13 weeks and the fit is finite."""
    sales, spend, promotion, price, kernel = arrays(LONG)
    assert kernel == sales.size == 156
    assert LONG["research"]["error"].startswith("ValueError")
    fit = fit_margins(sales, spend, promotion, price, kernel, noise=None)
    assert (fit.burn_in, fit.rows) == (BURN_IN, 156 - BURN_IN)
    assert np.all(np.isfinite(fit.mean))
    assert np.all(np.isfinite(fit.cov))
    assert np.all(np.linalg.eigvalsh(fit.cov) > 0.0)
    assert 0.0 <= fit.phi <= 0.95


def _world(
    weeks,
    *,
    seed=0,
    kernel=6,
    margins=(2.0, 0.5, 1.2),
    retention=(0.5, 0.3, 0.7),
    controls=True,
):
    """A world the model makes: three channels through the module's adstock, the spend before the
    history at its first 13 weeks' mean, a trend, a season, the promotion and the price (none of
    their effect without `controls`), and white noise."""
    rng = np.random.default_rng(seed)
    spend = rng.lognormal(np.log([100.0, 60.0, 40.0]), 0.3, size=(weeks, 3))
    flag = (rng.random(weeks) < 0.15).astype(float)
    price = 1.0 - 0.3 * flag + rng.normal(0.0, 0.02, weeks)
    before = spend[:PRESAMPLE].mean(axis=0)
    media = sum(
        m * adstock(spend[:, c], a, kernel, float(before[c]))
        for c, (a, m) in enumerate(zip(retention, margins, strict=True))
    )
    t = np.arange(weeks)
    calendar = 1000.0 + 0.5 * t + 50.0 * np.sin(2.0 * np.pi * t / 52.0)
    effect = 80.0 * flag - 100.0 * (price - 1.0) if controls else 0.0
    sales = calendar + effect + media + rng.normal(0.0, 5.0, weeks)
    return sales, spend, flag, price


@pytest.mark.parametrize("weeks", [104, 156])
def test_a_kernel_of_the_histories_length_fits(weeks):
    fit = fit_margins(*_world(weeks, kernel=weeks), weeks, noise=None)
    assert (fit.burn_in, fit.rows) == (BURN_IN, weeks - BURN_IN)
    assert np.all(np.isfinite(fit.mean))
    assert np.all(np.linalg.eigvalsh(fit.cov) > 0.0)


def test_at_a_negligible_prior_a_kernel_of_the_historys_length_finds_the_worlds_margins(
    monkeypatch,
):
    """The world is the regression's model, its adstock imputing the spend before the history.
    The prior pulls each column's contribution towards 0 by `PRIOR` of its squared norm, which on
    this world, its sales 200 noise deviations up, moves the margins by up to 4.8 of their
    deviations; at 1e-9 the retentions found are the world's and the margins lie within 3."""
    monkeypatch.setattr(margin_mmm, "PRIOR", 1e-9)
    margins = np.array([2.0, 0.5, 1.2])
    fit = fit_margins(*_world(156, kernel=156, margins=tuple(margins)), 156, noise=None)
    assert fit.top == (0.5, 0.3, 0.7)
    sd = np.sqrt(np.diag(fit.cov))
    assert np.all(np.abs(fit.mean - margins) < 3.0 * sd)
    assert np.all(sd < 0.1 * margins)


@pytest.mark.parametrize(("promotion", "price"), [(False, False), (True, False), (False, True)])
def test_a_history_without_a_control_fits_without_its_columns(promotion, price):
    """The flag and the price are a column each, and the promoted weeks, read off the price, a
    column each; a control the history lacks leaves out its own, and the rest of the design is the
    same."""
    sales, spend, flag, cost = _world(156)
    full = fit_margins(sales, spend, flag, cost, 6, noise=None)
    assert len(full.phases) > 0
    fit = fit_margins(
        sales, spend, flag if promotion else None, cost if price else None, 6, noise=None
    )
    lost = (not promotion) + (0 if price else 1 + len(full.phases))
    assert (fit.columns, fit.rows) == (full.columns - lost, full.rows)
    assert fit.phases == (full.phases if price else ())
    assert np.all(np.isfinite(fit.mean))
    assert np.all(np.linalg.eigvalsh(fit.cov) > 0.0)


def test_at_a_negligible_prior_a_history_without_the_controls_finds_the_worlds_margins(
    monkeypatch,
):
    """A world whose sales the promotion and the price do not move, read without them."""
    monkeypatch.setattr(margin_mmm, "PRIOR", 1e-9)
    margins = np.array([2.0, 0.5, 1.2])
    sales, spend, _, _ = _world(156, margins=tuple(margins), controls=False)
    fit = fit_margins(sales, spend, None, None, 6, noise=None)
    assert fit.phases == ()
    assert fit.top == (0.5, 0.3, 0.7)
    sd = np.sqrt(np.diag(fit.cov))
    assert np.all(np.abs(fit.mean - margins) < 3.0 * sd)
    assert np.all(sd < 0.1 * margins)


def test_the_prior_sets_the_margin_of_a_channel_whose_spend_never_moves_and_of_no_other():
    """Its column is the trend's, whose basis sums to 1: in every retention combination the
    prior's precision on its coefficient exceeds the data's, and in none on a channel whose spend
    moves by its coefficient of variation of about 0.3."""
    sales, spend, flag, price = _world(156)
    spend[:, 2] = spend[:, 2].mean()
    fit = fit_margins(sales, spend, flag, price, 6, noise=None)
    np.testing.assert_array_equal(fit.prior_share[:2], [0.0, 0.0])
    assert fit.prior_share[2] == pytest.approx(1.0, rel=0.0, abs=1e-12)


def test_the_datas_precision_is_the_schur_complement_of_the_cross_products():
    """The sum of squares a column keeps after its regression on the others is the inverse of
    the cross-products' inverse at it, where that inverse exists."""
    design = np.random.default_rng(3).normal(size=(60, 9))
    np.testing.assert_allclose(
        margin_mmm._schur(design, 3),
        1.0 / np.diag(np.linalg.inv(design.T @ design))[-3:],
        rtol=1e-9,
        atol=0.0,
    )


def test_the_spend_before_the_history_is_imputed_at_its_first_13_weeks_mean(monkeypatch):
    seen = []

    def spy(spend, retention, length, before):
        seen.append((spend, length, before))
        return adstock(spend, retention, length, before)

    monkeypatch.setattr(margin_mmm, "adstock", spy)
    sales, spend, flag, price = _world(104, kernel=104)
    fit_margins(sales, spend, flag, price, 104, noise=None)
    assert len(seen) == spend.shape[1] * len(margin_mmm.RETENTIONS)
    means = spend[:PRESAMPLE].mean(axis=0)
    for column, length, before in seen:
        (c,) = [j for j in range(spend.shape[1]) if np.array_equal(spend[:, j], column)]
        assert (length, before) == (104, means[c])


@pytest.mark.parametrize(("kernel", "burn_in"), [(1, 0), (6, 5), (14, 13), (15, 13), (104, 13)])
def test_the_burn_in_is_the_kernels_reach_back_at_most_13_weeks(kernel, burn_in):
    fit = fit_margins(*_world(104, kernel=kernel), kernel, noise=None)
    assert (fit.burn_in, fit.rows) == (burn_in, 104 - burn_in)


@pytest.mark.parametrize(("weeks", "length"), [(30, 6), (30, 14), (30, 30), (20, 45)])
def test_the_adstock_is_the_series_with_the_spend_before_it_at_its_level(weeks, length):
    """Against the history extended back by the kernel's reach at `before`: the same to rounding,
    and from the kernel's length on the history's own adstock to the bit, whatever `before`."""
    rng = np.random.default_rng(weeks + length)
    x, before = rng.lognormal(3.0, 0.5, weeks), 17.0
    for retention in (0.1, 0.5, 0.9):
        w = retention ** np.arange(length)
        w = w / w.sum()
        extended = np.convolve(np.r_[np.full(length - 1, before), x], w)[length - 1 :][:weeks]
        got = adstock(x, retention, length, before)
        np.testing.assert_allclose(got, extended, rtol=1e-12, atol=0.0)
        np.testing.assert_array_equal(
            got[length - 1 :], adstock(x, retention, length, 0.0)[length - 1 :]
        )
        np.testing.assert_array_equal(got[length - 1 :], np.convolve(x, w)[length - 1 : weeks])


def test_a_promoted_phase_is_a_week_of_the_year_priced_below_the_median_by_more_than_0_15():
    weeks = 156
    price = np.ones(weeks)
    price[np.arange(weeks) % 52 == 10] = 0.7
    price[np.arange(weeks) % 52 == 20] = 0.9  # 0.10 below: not promoted
    price[[30, 82]] = 0.6  # in two of the three years: their mean 0.733 is promoted
    price[40] = 0.6  # in one of three: its mean 0.867 is not
    assert promoted_phases(price) == [10, 30]


def test_a_history_too_short_for_the_design_raises_the_typed_error():
    """20 weeks leave 15 after the burn-in, and the design holds the spline's 4, the season's 6,
    the flag, the price, the events and the 3 channels."""
    sales, spend, flag, price = _world(20)
    with pytest.raises(DegenerateFitError, match="no residual degree of freedom"):
        fit_margins(sales, spend, flag, price, 6, noise=None)


def test_sales_the_design_fits_exactly_raise_the_typed_error_not_a_nan():
    """The research code's AR(1) coefficient was 0/0 on a residual of 0."""
    _, spend, flag, price = _world(104)
    with pytest.raises(DegenerateFitError, match="0/0"):
        fit_margins(np.zeros(104), spend, flag, price, 6, noise=None)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda a: (a[0][:-1], *a[1:]), "sales are"),
        (lambda a: (a[0], a[1][:, 0], *a[2:]), "sales are"),
        (lambda a: (a[0], a[1], a[2], a[3][:-1], a[4]), "sales are"),
        (lambda a: (np.where(np.arange(a[0].size) == 7, np.nan, a[0]), *a[1:]), "finite"),
        (lambda a: (*a[:4], 0), "at least a week"),
    ],
)
def test_input_the_regression_does_not_read_is_refused(change, match):
    sales, spend, flag, price = _world(104)
    with pytest.raises(ValueError, match=match) as raised:
        fit_margins(*change((sales, spend, flag, price, 6)))
    assert not isinstance(raised.value, DegenerateFitError)


def test_the_curved_fit_whose_curves_are_lines_is_the_linear_fit():
    """At an infinite half-saturation each column is its adstock to the bit: every combination's
    fit and weight are `fit_margins`', and each margin at the status quo is its coefficient times
    the quarter's mean slope of a line, 1 to rounding."""
    sales, spend, flag, price = _world(156)
    quo = spend[-52:].mean(axis=0)
    linear = fit_margins(sales, spend, flag, price, 6, noise=None)
    curved = fit_curved_margins(sales, spend, flag, price, 6, quo, 13, math.inf, noise=None)
    fit = curved.margins
    np.testing.assert_allclose(fit.mean, linear.mean, rtol=RELATIVE, atol=0.0)
    np.testing.assert_allclose(fit.cov, linear.cov, rtol=RELATIVE, atol=0.0)
    for field in ("retention", "top", "top_weight", "phi", "prior_share", "phases", "rows"):
        np.testing.assert_array_equal(getattr(fit, field), getattr(linear, field))
    assert not curved.shaped.any()
    spends = quo * np.array([0.3, 1.0, 2.5])
    np.testing.assert_allclose(curved.curve.slope(spends), 1.0, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(curved.curve.rise(spends, quo), spends - quo, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("kernel", [1, 6, 30])
def test_each_curve_has_slope_1_at_the_status_quo_falls_and_rises_by_its_integral(kernel):
    """`H'_c(s_c) = 1` (the margin at the status quo carries the level), `H'_c` falls in the
    spend, and `H_c(p) - H_c(s_c)` is the integral of `H'_c` from `s_c` to `p`."""
    sales, spend, flag, price = _world(156, kernel=kernel)
    quo = spend[-52:].mean(axis=0)
    curved = fit_curved_margins(sales, spend, flag, price, kernel, quo, 13, noise=None)
    assert curved.shaped.all()
    np.testing.assert_array_equal(curved.half_saturation, HALF_SATURATION * quo)
    np.testing.assert_allclose(curved.weights.sum(axis=1), 1.0, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(curved.curve.slope(quo), 1.0, rtol=1e-12, atol=0.0)
    grid = quo[None, :] * np.linspace(0.0, 3.0, 31)[:, None]
    slopes = np.array([curved.curve.slope(p) for p in grid])
    assert np.all(np.diff(slopes, axis=0) < 0.0)
    target = quo * np.array([0.2, 1.7, 2.9])
    for c in range(quo.size):

        def along(p, c=c):
            spends = quo.copy()
            spends[c] = p
            return curved.curve.slope(spends)[c]

        integral, _ = quad(along, quo[c], target[c], epsabs=0.0, epsrel=1e-12)
        assert curved.curve.rise(target, quo)[c] == pytest.approx(integral, rel=1e-10, abs=0.0)


def test_the_planning_adstock_is_each_channels_adstock_at_its_retention_of_most_weight():
    sales, spend, flag, price = _world(156)
    quo = spend[-52:].mean(axis=0)
    curved = fit_curved_margins(sales, spend, flag, price, 6, quo, 13, noise=None)
    before = spend[:PRESAMPLE].mean(axis=0)
    top = np.argmax(curved.weights, axis=1)
    expected = np.column_stack(
        [
            adstock(spend[:, c], margin_mmm.RETENTIONS[top[c]], 6, float(before[c]))
            for c in range(quo.size)
        ]
    )
    np.testing.assert_array_equal(curved.planning_adstock(), expected)


@pytest.mark.parametrize(
    ("quo", "planned", "half", "match"),
    [
        (np.array([1.0, 1.0]), 13, 16.0, "the status quo is"),
        (np.array([1.0, 0.0, 1.0]), 13, 16.0, "the status quo is"),
        (np.array([1.0, 1.0, 1.0]), 0, 16.0, "a planned week"),
        (np.array([1.0, 1.0, 1.0]), 13, 0.0, "a positive half-saturation"),
    ],
    ids=["shape", "zero", "no week", "half-saturation"],
)
def test_input_the_curved_regression_does_not_read_is_refused(quo, planned, half, match):
    sales, spend, flag, price = _world(104)
    with pytest.raises(ValueError, match=match):
        fit_curved_margins(sales, spend, flag, price, 6, quo, planned, half)


def _noise_world(weeks=156, *, seed=0, rho=0.0, share=1.0, ratio=1.0, phases=(8, 21, 34, 47)):
    """A world the noise mixture's model makes: `_world`'s channels and calendar, the price cut on
    the promoted `phases` of every year, and errors of deviation 5 that are an AR(1) of coefficient
    `rho` holding `share` of their variance plus white noise, their variance `ratio` times as wide
    on the promoted weeks. No promotion flag: it would be the promoted weeks' dummies' sum."""
    rng = np.random.default_rng(seed)
    spend = rng.lognormal(np.log([100.0, 60.0, 40.0]), 0.3, size=(weeks, 3))
    t = np.arange(weeks)
    promoted = np.isin(t % 52, phases)
    price = 1.0 - 0.3 * promoted + rng.normal(0.0, 0.02, weeks)
    before = spend[:PRESAMPLE].mean(axis=0)
    media = sum(
        m * adstock(spend[:, c], a, 6, float(before[c]))
        for c, (a, m) in enumerate(zip((0.5, 0.3, 0.7), (2.0, 0.5, 1.2), strict=True))
    )
    calendar = 1000.0 + 0.5 * t + 50.0 * np.sin(2.0 * np.pi * t / 52.0)
    shocks = rng.normal(size=weeks)
    persistent = np.empty(weeks)
    persistent[0] = shocks[0]
    for week in range(1, weeks):
        persistent[week] = rho * persistent[week - 1] + math.sqrt(1.0 - rho * rho) * shocks[week]
    errors = math.sqrt(share) * persistent + math.sqrt(1.0 - share) * rng.normal(size=weeks)
    wide = np.where(promoted, math.sqrt(ratio), 1.0)
    return calendar - 100.0 * (price - 1.0) + media + 5.0 * wide * errors, spend, price


def _pieces(sales, spend, flag, price, noise, half=HALF_SATURATION):
    """What the mixture's fits read of a history at a kernel of 6 weeks: the design, each
    channel's straight adstock column at each retention on the weeks read, the columns saturated
    at `half` status quos, and the fourth power of the curve's `half / (1 + half)`."""
    history = margin_mmm._read(sales, spend, flag, price, 6, noise)
    lines = np.array(
        [[history.adstocks[c, a][history.rows] for a in RETENTIONS] for c in range(spend.shape[1])]
    )
    quo = spend[-52:].mean(axis=0)
    columns = lines / (1.0 + lines / (half * quo)[:, None, None])
    return history, columns, lines, (half / (1.0 + half)) ** 4


def _one(rho, omega, h):
    """The mixture of the one error model `(rho, omega, h)`, white errors at `rho` 0."""
    if rho == 0.0:
        return NoiseMixture(persistence=(), share=(), promoted=(h,))
    return NoiseMixture(persistence=(rho,), share=(omega,), promoted=(h,), white=False)


@pytest.mark.parametrize("world", [*PARITY, LONG], ids=name)
def test_under_the_mixture_the_curved_margins_are_the_research_codes(world):
    """The frozen research rule's regression on the seven worlds, family 14's among them: the
    margins at the status quo, their covariance, the posterior means of the error model and the
    columns the design kept."""
    sales, spend, promotion, price, kernel = arrays(world)
    quo = np.asarray(world["status_quo"], dtype=float)
    fit = fit_curved_margins(
        sales, spend, promotion, price, kernel, quo, world["planned"], noise=NoiseMixture()
    ).margins
    research = NOISE[world["seed"]]
    np.testing.assert_allclose(fit.mean, research["mean"], rtol=RELATIVE, atol=0.0)
    np.testing.assert_allclose(fit.cov, research["cov"], rtol=RELATIVE, atol=0.0)
    assert fit.phi == pytest.approx(research["rho"], rel=RELATIVE, abs=0.0)
    assert fit.share == pytest.approx(research["omega"], rel=RELATIVE, abs=0.0)
    assert fit.promoted == pytest.approx(research["h"], rel=RELATIVE, abs=0.0)
    assert fit.columns == research["columns"]


def test_the_mixtures_grid_holds_white_errors_once_and_every_pair_with_every_ratio():
    noise = NoiseMixture()
    assert noise.knot_every == 52
    pairs = noise.pairs()
    assert pairs[0] == (0.0, 1.0)
    assert pairs[1:] == [
        (rho, omega)
        for rho in (0.3, 0.6, 0.8, 0.9, 0.95, 0.98)
        for omega in (0.5, 0.7, 0.85, 0.95, 1.0)
    ]
    assert noise.models() == [(rho, omega, h) for rho, omega in pairs for h in (1.0, 2.0, 4.0, 8.0)]
    assert len(noise.models()) == 124
    assert NoiseMixture(white=False).models()[0] == (0.3, 0.5, 1.0)


@pytest.mark.parametrize(
    "settings",
    [
        {"knot_every": 0},
        {"persistence": (0.0,)},
        {"persistence": (1.0,)},
        {"share": (0.0,)},
        {"share": (1.1,)},
        {"promoted": (0.0,)},
        {"promoted": (math.inf,)},
        {"promoted": ()},
        {"persistence": (), "white": False},
    ],
    ids=str,
)
def test_a_mixture_the_regression_does_not_read_is_refused(settings):
    with pytest.raises(ValueError, match="a noise mixture has"):
        NoiseMixture(**settings)


def _dense(history, columns, lines, steady, model):
    """Every retention combination's regression under the error model `(rho, omega, h)` by dense
    algebra, the errors' covariance inverted whole and the residual formed: the channels'
    coefficients, their covariance and the log evidence, a row a combination."""
    rho, omega, h = model
    y, z = history.sales, history.nuisance
    n, q = z.shape
    week = np.arange(n)
    d = np.where(np.isin(history.rows % 52, history.phases), math.sqrt(h), 1.0)
    cov = np.outer(d, d) * (
        omega * rho ** np.abs(week[:, None] - week[None, :]) + (1.0 - omega) * np.eye(n)
    )
    precision = np.linalg.inv(cov)
    means, covs, evidence = [], [], []
    for combo in itertools.product(range(columns.shape[1]), repeat=columns.shape[0]):
        x = np.column_stack([z, *(columns[c, r] for c, r in enumerate(combo))])
        prior = np.zeros(x.shape[1])
        prior[q:] = [
            margin_mmm.PRIOR * steady * (lines[c, r] @ precision @ lines[c, r])
            for c, r in enumerate(combo)
        ]
        a = x.T @ precision @ x + np.diag(prior)
        b = np.linalg.solve(a, x.T @ precision @ y)
        rest = y - x @ b
        rss = rest @ precision @ rest + prior @ b**2
        means.append(b[q:])
        covs.append(rss / (n - x.shape[1]) * np.linalg.inv(a)[q:, q:])
        evidence.append(
            -0.5 * np.linalg.slogdet(cov)[1]
            + 0.5 * np.sum(np.log(prior[q:]))
            - 0.5 * np.linalg.slogdet(a)[1]
            - 0.5 * (n - q) * math.log(rss)
        )
    return np.array(means), np.array(covs), np.array(evidence)


@pytest.mark.parametrize(
    "model", [(0.0, 1.0, 1.0), (0.3, 1.0, 2.0), (0.8, 0.7, 4.0), (0.98, 0.5, 8.0)], ids=str
)
def test_each_error_models_fit_is_the_generalised_least_squares_under_its_covariance(model):
    """The whitening by the covariance's Cholesky factor and the table of cross-products, against
    the covariance inverted whole: every retention combination's coefficients, their covariance
    and its log evidence, under white errors, an AR(1) alone and one beside white noise, each with
    its variance ratio on the 12 promoted weeks."""
    noise = _one(*model)
    assert noise.models() == [model]
    sales, spend, price = _noise_world()
    history, columns, lines, steady = _pieces(sales, spend, None, price, noise)
    assert np.isin(history.rows % 52, history.phases).sum() == 12
    fits = margin_mmm._noise_fits(history, columns, lines, steady, noise)
    means, covs, evidence = _dense(history, columns, lines, steady, model)
    np.testing.assert_allclose(fits.mean, means, rtol=1e-8, atol=0.0)
    np.testing.assert_allclose(fits.cov, covs, rtol=1e-8, atol=1e-8 * np.abs(covs).max())
    np.testing.assert_allclose(fits.evidence, evidence, rtol=0.0, atol=1e-7)
    np.testing.assert_array_equal(fits.errors, np.tile(model, (125, 1)))


def _prais_winsten(a, phi):
    """Prais and Winsten's transform of the weeks in `a`'s rows at the AR(1) coefficient `phi`:
    the first times `sqrt(1 - phi^2)`, each other less `phi` times the week before it."""
    out = np.array(a, dtype=float)
    out[1:] -= phi * np.asarray(a)[:-1]
    out[0] *= math.sqrt(1.0 - phi * phi)
    return out


@pytest.mark.parametrize("phi", [0.3, 0.9])
def test_at_an_ar1_alone_the_fit_is_prais_and_winstens_under_a_flat_nuisance_prior(phi):
    """One error model, an AR(1) holding all the variance of every week: the regression on the
    weeks differenced by Prais and Winsten's transform, no prior on a nuisance column, each
    channel's precision `PRIOR` of its differenced straight column's sum of squares times the
    curve's steady slope squared, and the evidence's degrees of freedom the weeks less the nuisance
    columns. The transform whitens an AR(1) of unit innovations and the mixture one of unit
    variance: the evidence is the same on both scales."""
    noise = _one(phi, 1.0, 1.0)
    sales, spend, flag, price = _world(156)
    history, columns, lines, steady = _pieces(sales, spend, flag, price, noise)
    fits = margin_mmm._noise_fits(history, columns, lines, steady, noise)
    n, q = history.nuisance.shape
    yw, zw = _prais_winsten(history.sales, phi), _prais_winsten(history.nuisance, phi)
    for m, combo in enumerate(itertools.product(range(len(RETENTIONS)), repeat=3)):
        xw = np.column_stack(
            [zw, *(_prais_winsten(columns[c, r], phi) for c, r in enumerate(combo))]
        )
        prior = np.zeros(q + 3)
        prior[q:] = [
            margin_mmm.PRIOR * steady * np.sum(_prais_winsten(lines[c, r], phi) ** 2)
            for c, r in enumerate(combo)
        ]
        a = xw.T @ xw + np.diag(prior)
        b = np.linalg.solve(a, xw.T @ yw)
        rss = np.sum((yw - xw @ b) ** 2) + prior @ b**2
        evidence = (
            0.5 * math.log(1.0 - phi * phi)
            + 0.5 * np.sum(np.log(prior[q:]))
            - 0.5 * np.linalg.slogdet(a)[1]
            - 0.5 * (n - q) * math.log(rss)
        )
        spread = rss / (n - q - 3) * np.linalg.inv(a)[q:, q:]
        np.testing.assert_allclose(fits.mean[m], b[q:], rtol=1e-8, atol=0.0)
        np.testing.assert_allclose(fits.cov[m], spread, rtol=1e-8, atol=1e-8 * np.abs(spread).max())
        assert fits.evidence[m] == pytest.approx(evidence, rel=0.0, abs=1e-7)


def test_the_evidence_is_the_same_on_any_common_scale_of_the_errors_covariance():
    """With every week read promoted the variance ratio scales the whole covariance: each fit's
    log evidence, coefficients and covariance are the same at every ratio
    (`validation/margin_noise.mac`). With the promoted weeks the history's own, the ratio moves
    the evidence of every fit."""
    noise = NoiseMixture()
    ratios = len(noise.promoted)
    sales, spend, price = _noise_world()
    history, columns, lines, steady = _pieces(sales, spend, None, price, noise)
    everywhere = dataclasses.replace(history, phases=tuple(range(52)))
    scaled = margin_mmm._noise_fits(everywhere, columns, lines, steady, noise)
    evidence = scaled.evidence.reshape(-1, ratios, 125)
    np.testing.assert_allclose(evidence - evidence[:, :1], 0.0, rtol=0.0, atol=1e-7)
    mean = scaled.mean.reshape(-1, ratios, 125, 3)
    np.testing.assert_allclose(mean / mean[:, :1], 1.0, rtol=1e-9, atol=0.0)
    cov = scaled.cov.reshape(-1, ratios, 125, 3, 3)
    np.testing.assert_allclose(cov - cov[:, :1], 0.0, rtol=0.0, atol=1e-9 * np.abs(cov).max())
    own = margin_mmm._noise_fits(history, columns, lines, steady, noise)
    assert np.all(np.ptp(own.evidence.reshape(-1, ratios, 125), axis=1) > 0.1)


def test_under_a_mixture_a_nuisance_column_that_is_0_on_the_weeks_read_is_dropped():
    """A phase promoted in the burn-in alone: its dummy is 0 on every week read, a flat prior
    leaves its coefficient free, and the column is dropped. The fit is that of the history whose
    burn-in week is not promoted, the weeks read being the same, to the bit; its design holds the
    trend's 5 columns, the season's 6, the flag, the price, one promoted phase and the channels.
    Without a mixture the ridge keeps the column."""
    sales, spend, flag, _ = _world(56)
    price = 1.0 + np.random.default_rng(5).normal(0.0, 0.02, 56)
    price[30] = 0.6
    early = price.copy()
    early[4] = 0.6
    assert (promoted_phases(price), promoted_phases(early)) == ([30], [4, 30])
    noise = NoiseMixture()
    plain = fit_margins(sales, spend, flag, price, 6, noise=noise)
    fit = fit_margins(sales, spend, flag, early, 6, noise=noise)
    assert (plain.phases, fit.phases) == ((30,), (4, 30))
    assert fit.columns == plain.columns == 5 + 6 + 1 + 1 + 1 + 3
    for field in ("mean", "cov", "retention", "prior_share"):
        np.testing.assert_array_equal(getattr(fit, field), getattr(plain, field))
    assert (fit.phi, fit.share, fit.promoted, fit.top, fit.top_weight) == (
        plain.phi,
        plain.share,
        plain.promoted,
        plain.top,
        plain.top_weight,
    )
    assert fit_margins(sales, spend, flag, early, 6, noise=None).columns == 6 + 6 + 1 + 1 + 2 + 3


def test_under_a_mixture_the_trends_knots_are_its_own_52_weeks_apart():
    """156 weeks hold interior knots at 26, 52, 78, 104 and 130, or at 52 and 104: three columns
    fewer; a mixture with knots every 26 weeks keeps them."""
    sales, spend, flag, price = _world(156)
    ridge = fit_margins(sales, spend, flag, price, 6, noise=None)
    fit = fit_margins(sales, spend, flag, price, 6, noise=NoiseMixture())
    assert (ridge.columns - fit.columns, fit.rows) == (3, ridge.rows)
    close = fit_margins(sales, spend, flag, price, 6, noise=_one(0.6, 0.85, 2.0))
    assert close.columns == fit.columns
    every = dataclasses.replace(_one(0.6, 0.85, 2.0), knot_every=26)
    assert fit_margins(sales, spend, flag, price, 6, noise=every).columns == ridge.columns


def test_a_mixture_of_one_error_model_reports_it_and_no_mixture_reports_an_ar1_alone():
    sales, spend, flag, price = _world(156)
    fit = fit_margins(sales, spend, flag, price, 6, noise=_one(0.6, 0.85, 2.0))
    assert (fit.phi, fit.share, fit.promoted) == pytest.approx((0.6, 0.85, 2.0), rel=1e-12, abs=0.0)
    mixed = fit_margins(sales, spend, flag, price, 6, noise=NoiseMixture())
    assert 0.0 < mixed.phi < 0.98
    assert 0.5 < mixed.share < 1.0
    assert 1.0 < mixed.promoted < 8.0
    assert mixed.top_weight < 1.0
    assert np.all((mixed.retention > 0.1) & (mixed.retention < 0.9))
    ridge = fit_margins(sales, spend, flag, price, 6, noise=None)
    assert (ridge.share, ridge.promoted) == (1.0, 1.0)


def test_at_a_negligible_prior_the_mixture_reads_the_errors_a_world_was_made_with(monkeypatch):
    """Three worlds of the mixture's own model: white errors, an AR(1) of coefficient 0.9, and
    white errors 8 times as wide in variance on the promoted weeks. The posterior means of the
    AR(1) coefficient and of the variance ratio tell them apart. The prior is made negligible
    because its width chooses among the error models too: a channel's precision is `PRIOR` of its
    whitened column's sum of squares, most of it the column's level, which a persistent error
    model discounts; at `PRIOR` itself these worlds' margins lie far outside the white model's
    prior, and the white world reads a coefficient near 0.98."""
    noise = NoiseMixture()

    def read(**made):
        sales, spend, price = _noise_world(**made)
        return fit_margins(sales, spend, None, price, 6, noise=noise)

    assert read().phi > 0.95
    monkeypatch.setattr(margin_mmm, "PRIOR", 1e-9)
    white, persistent, wide = read(), read(rho=0.9), read(ratio=8.0)
    assert white.phi < 0.3 < 0.8 < persistent.phi
    assert persistent.share > 0.95
    assert max(white.promoted, persistent.promoted) < 2.0 < 4.0 < wide.promoted
    for fit in (white, persistent, wide):
        sd = np.sqrt(np.diag(fit.cov))
        assert np.all(np.abs(fit.mean - [2.0, 0.5, 1.2]) < 3.0 * sd)


def test_the_curved_fit_under_a_mixture_whose_curves_are_lines_is_the_linear_fit_under_it():
    sales, spend, flag, price = _world(156)
    quo = spend[-52:].mean(axis=0)
    noise = NoiseMixture()
    linear = fit_margins(sales, spend, flag, price, 6, noise=noise)
    curved = fit_curved_margins(sales, spend, flag, price, 6, quo, 13, math.inf, noise=noise)
    fit = curved.margins
    np.testing.assert_allclose(fit.mean, linear.mean, rtol=RELATIVE, atol=0.0)
    np.testing.assert_allclose(fit.cov, linear.cov, rtol=RELATIVE, atol=0.0)
    for field in ("retention", "top", "top_weight", "phi", "share", "promoted", "prior_share"):
        np.testing.assert_array_equal(getattr(fit, field), getattr(linear, field))
    np.testing.assert_allclose(curved.weights.sum(axis=1), 1.0, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(curved.curve.slope(quo), 1.0, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("noise", [NoiseMixture(), _one(0.6, 0.85, 2.0)], ids=["grid", "one"])
def test_nuisance_columns_the_weeks_do_not_tell_apart_raise_the_typed_error(noise):
    """A price that never moves is the trend's columns' sum, and a flat prior leaves the fit
    improper: the typed error whatever the rounding of a factorisation, where the ridge fits."""
    sales, spend, flag, _ = _world(156)
    price = np.ones(156)
    with pytest.raises(DegenerateFitError, match="do not tell the design's 14 nuisance columns"):
        fit_margins(sales, spend, flag, price, 6, noise=noise)
    assert np.all(np.isfinite(fit_margins(sales, spend, flag, price, 6, noise=None).mean))


def _reproduced(control):
    """A history whose `control` the mixture's other nuisance columns reproduce on every week
    read, and the count of the nuisance columns: `_world`'s with a price that never moves or one
    that is a straight line in the week, or `_noise_world`'s with a promotion flag that is 1 on
    its promoted weeks alone."""
    week = np.arange(156)
    if control == "flag":
        sales, spend, price = _noise_world()
        assert promoted_phases(price) == [8, 21, 34, 47]
        return sales, spend, np.isin(week % 52, (8, 21, 34, 47)).astype(float), price, 18
    sales, spend, flag, _ = _world(156)
    return sales, spend, flag, np.ones(156) if control == "constant" else 1.0 + 0.0005 * week, 14


@pytest.mark.parametrize("control", ["constant", "line", "flag"])
def test_a_control_the_other_nuisance_columns_reproduce_raises_the_typed_error(control):
    """A price that never moves is the trend's columns' sum; a price that is a straight line in
    the week is a combination of them, a cubic spline's; and a promotion flag that is 1 on the
    promoted weeks alone is their dummies' sum. Under the flat nuisance prior each leaves the fit
    improper: the curved fit, the arm's, raises the typed error with the weeks read and the
    columns' count, where the ridge without a mixture fits."""
    sales, spend, flag, price, columns = _reproduced(control)
    quo = spend[-52:].mean(axis=0)
    message = (
        f"the 151 weeks read do not tell the design's {columns} nuisance columns apart, and their"
        " flat prior leaves the fit improper"
    )
    with pytest.raises(DegenerateFitError, match=message):
        fit_curved_margins(sales, spend, flag, price, 6, quo, 13, noise=NoiseMixture())
    ridge = fit_curved_margins(sales, spend, flag, price, 6, quo, 13, noise=None).margins
    assert np.all(np.isfinite(ridge.mean))
    assert not ridge.spanned.any()


def test_under_a_mixture_a_channel_the_nuisance_columns_span_has_a_margin_of_exactly_0():
    """A spend that never moves is a multiple of the trend's columns' sum. Under their flat prior
    its coefficient's posterior is its prior, of mean 0, in every model, and floating point left a
    residue of either sign there, 1e-12 of the other margins, which a reader of the margins' signs
    took for the data's. The fit names the channel spanned and reads its margin as exactly 0, of
    its prior's variance and independent of the others'; no other channel is spanned, and without a
    mixture none is: the ridge on the nuisance columns sets the constant channel's margin."""
    sales, spend, price = _noise_world()
    spend[:, 2] = spend[:, 2].mean()
    noise = NoiseMixture()
    quo = spend[-52:].mean(axis=0)
    history, columns, lines, steady = _pieces(sales, spend, None, price, noise)
    for block in (columns, lines):
        apart = margin_mmm._apart(history.nuisance, block)
        assert apart[2] < 1e-13
        assert np.all(apart[:2] > 0.1)
    fits = margin_mmm._noise_fits(history, columns, lines, steady, noise)
    np.testing.assert_array_equal(fits.spanned, [False, False, True])
    np.testing.assert_array_equal(fits.mean[:, 2], 0.0)
    assert np.all(fits.mean[:, :2] != 0.0)
    for fit in (
        fit_margins(sales, spend, None, price, 6, noise=noise),
        fit_curved_margins(sales, spend, None, price, 6, quo, 13, noise=noise).margins,
    ):
        np.testing.assert_array_equal(fit.spanned, [False, False, True])
        assert fit.mean[2] == 0.0
        assert np.all(fit.mean[:2] != 0.0)
        assert fit.prior_share[2] == pytest.approx(1.0, rel=0.0, abs=1e-12)
        sd = np.sqrt(np.diag(fit.cov))
        assert sd[2] > 5.0 * sd[:2].max()
        assert np.all(np.abs(fit.cov[2, :2]) < 1e-6 * sd[2] * sd[:2])
    plain = fit_margins(sales, spend, None, price, 6, noise=None)
    assert not plain.spanned.any()
    assert plain.mean[2] != 0.0


@pytest.mark.parametrize(("move", "spanned"), [(1e-10, True), (1e-9, True), (1e-6, False)])
def test_a_channel_is_spanned_within_the_square_root_of_the_rounding_unit_and_not_past_it(
    move, spanned
):
    """A constant spend moved by `move` of itself along a pattern: its columns lie that far from
    the span, less what the kernel averages out. Within `SPANNED` the margin is exactly 0 whichever
    way the spend was moved; past it the data set the margin, to a sign that follows the move."""
    assert SPANNED == 1e-8
    sales, spend, price = _noise_world()
    pattern = np.random.default_rng(1).standard_normal(156)
    pattern /= np.abs(pattern).max()
    level = spend[:, 2].mean()
    margins = []
    for sign in (1.0, -1.0):
        spend[:, 2] = level * (1.0 + sign * move * pattern)
        fit = fit_margins(sales, spend, None, price, 6, noise=NoiseMixture())
        assert bool(fit.spanned[2]) is spanned
        assert not fit.spanned[:2].any()
        margins.append(float(fit.mean[2]))
    if spanned:
        assert margins == [0.0, 0.0]
    else:
        assert margins[0] * margins[1] < 0.0


def test_under_a_mixture_a_history_too_short_for_the_columns_kept_raises_the_typed_error():
    """20 weeks leave 15 after the burn-in, and the design holds the spline's 4, the season's 6,
    the flag, the price, the events and the 3 channels."""
    sales, spend, flag, price = _world(20)
    with pytest.raises(DegenerateFitError, match="no residual degree of freedom"):
        fit_margins(sales, spend, flag, price, 6, noise=NoiseMixture())


@pytest.mark.parametrize("world", [*PARITY, LONG], ids=name)
def test_without_a_mixture_the_fit_is_assembled_as_before_one_to_the_bit(world, monkeypatch):
    """With `noise=None` no error model is fitted, and the fit is each retention combination's
    regression mixed by its evidence, in the arithmetic it had before the mixture, to the bit."""

    def refuse(*_args):
        raise AssertionError("an error model was fitted")

    monkeypatch.setattr(margin_mmm, "_noise_fits", refuse)
    sales, spend, promotion, price, kernel = arrays(world)
    fit = fit_margins(sales, spend, promotion, price, kernel, noise=None)
    history = margin_mmm._read(sales, spend, promotion, price, kernel)
    combos = list(itertools.product(RETENTIONS, repeat=spend.shape[1]))
    fits = [
        margin_mmm._combination(
            history, [history.adstocks[c, a][history.rows] for c, a in enumerate(combo)], combo
        )
        for combo in combos
    ]
    logs = np.array([f.evidence for f in fits])
    w = np.exp(logs - logs.max())
    w /= w.sum()
    means = np.array([f.mean for f in fits])
    mean = w @ means
    second = np.einsum(
        "k,kij->ij", w, np.array([f.cov for f in fits]) + np.einsum("ki,kj->kij", means, means)
    )
    cov = second - np.outer(mean, mean)
    np.testing.assert_array_equal(fit.mean, mean)
    np.testing.assert_array_equal(fit.cov, 0.5 * (cov + cov.T))
    np.testing.assert_array_equal(fit.retention, w @ np.array(combos))
    np.testing.assert_array_equal(
        fit.prior_share, w @ np.array([f.dominated for f in fits], dtype=float)
    )
    top = int(np.argmax(w))
    assert (fit.top, fit.top_weight) == (combos[top], float(w[top]))
    assert fit.phi == float(w @ np.array([f.phi for f in fits]))
    assert (fit.share, fit.promoted) == (1.0, 1.0)
    assert not fit.spanned.any()


def _readme_world(level):
    """The README's history at the sales' `level`: three channels of i.i.d. uniform weekly spend,
    each through an adstock of retention 0.5 over 8 weeks, margins of 1.5, 3.0 and 0.5, and white
    noise of deviation 25 over 104 weeks."""
    rng = np.random.default_rng(0)
    spend = rng.uniform([60.0, 20.0, 15.0], [180.0, 80.0, 55.0], size=(104, 3))
    carried = np.column_stack([adstock(s, 0.5, 8, s[:13].mean()) for s in spend.T])
    return level + carried @ np.array([1.5, 3.0, 0.5]) + rng.normal(0.0, 25.0, 104), spend


def test_by_default_the_fit_is_the_mixtures():
    """`fit_margins` and `fit_curved_margins` without `noise` fit under `NoiseMixture()`."""
    sales, spend = _readme_world(5000.0)
    quo = spend[-13:].mean(axis=0)
    pairs = (
        (
            fit_margins(sales, spend, None, None, 8),
            fit_margins(sales, spend, None, None, 8, noise=NoiseMixture()),
        ),
        (
            fit_curved_margins(sales, spend, None, None, 8, quo, 13).margins,
            fit_curved_margins(sales, spend, None, None, 8, quo, 13, noise=NoiseMixture()).margins,
        ),
    )
    for got, want in pairs:
        for field in ("mean", "cov", "retention", "prior_share", "spanned"):
            np.testing.assert_array_equal(getattr(got, field), getattr(want, field))
        assert (got.phi, got.share, got.promoted, got.top) == (
            want.phi,
            want.share,
            want.promoted,
            want.top,
        )


def test_by_default_the_sales_level_moves_no_margin_and_the_truth_is_read():
    """The README's history, its sales 200 noise deviations up. The mixture's flat prior leaves
    the trend's level free: at a level of 0 and of 5000 the margins and their covariance agree to
    rounding and lie within two deviations of the truth. Without a mixture the ridge pulls the
    level towards 0 and the channels' columns take it up: the level moves the first two margins by
    more than two deviations, to about three times the truth."""
    truth = np.array([1.5, 3.0, 0.5])
    low, high = (fit_margins(*_readme_world(level), None, None, 8) for level in (0.0, 5000.0))
    np.testing.assert_allclose(high.mean, low.mean, rtol=1e-9, atol=0.0)
    np.testing.assert_allclose(high.cov, low.cov, rtol=1e-8, atol=0.0)
    np.testing.assert_allclose(high.retention, low.retention, rtol=1e-9, atol=0.0)
    sd = np.sqrt(np.diag(high.cov))
    assert np.all(np.abs(high.mean - truth) < 2.0 * sd)
    ridge_low, ridge_high = (
        fit_margins(*_readme_world(level), None, None, 8, noise=None) for level in (0.0, 5000.0)
    )
    ridge_sd = np.sqrt(np.diag(ridge_high.cov))
    assert np.all(ridge_high.mean[:2] - ridge_low.mean[:2] > 2.0 * ridge_sd[:2])
    assert np.all(ridge_high.mean[:2] > 2.5 * truth[:2])


@pytest.mark.skipif(shutil.which("maxima") is None, reason="maxima is not on PATH")
def test_maxima_proves_the_error_models_identities_at_every_step():
    script = Path(__file__).parents[1] / "validation" / "margin_noise.mac"
    out = subprocess.run(
        ["maxima", "--very-quiet", "-b", str(script)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    ).stdout
    assert "Lisp error" not in out
    verdicts = [line.strip() for line in out.splitlines() if line.startswith("STEP ")]
    assert len(verdicts) == 7
    assert all(line.endswith("(PASS)") for line in verdicts), verdicts
