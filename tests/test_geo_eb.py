"""mixdyn.models.geo_eb and mixdyn.decisions.geo_plan.plan_mixture: the closed-form model over geos.

The identities the fit relies on pass in Maxima (`validation/geo_eb.mac`). On a small panel the
marginal likelihood of a choice of grid points and the coefficients' posterior mean equal the
brute force's: the Gaussian density of every geo's KPI and the lift, their columns and the test's
row built here by plain loops. A lift read almost without error is reproduced. The plan spends the
budget within the box and meets the conditions of the best plan: one price for every cell inside
its box, a cell at its floor no steeper, one at its ceiling no flatter. A channel bought
nationally is one cell whose worth is its geos' summed.
"""

from __future__ import annotations

import dataclasses
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from mixdyn import Cadence, Channel, Geo, MediaPanel, Revenue, allot_national
from mixdyn.calibration import LiftEstimate
from mixdyn.decisions.geo_plan import BISECTIONS, FEASIBLE, GRID_POINTS, plan_mixture
from mixdyn.models import geo_eb
from mixdyn.models.geo_eb import POINTS, CellResponse, fit_pooled_geos

NORTH, SOUTH, WEST = Geo("north"), Geo("south"), Geo("west")
GEOS = (NORTH, SOUTH, WEST)
SEARCH, SOCIAL, TV = Channel("search"), Channel("social"), Channel("tv")
WEEKS, LENGTH, PERIODS = 60, 4, 13
POPULATION = np.array([5.0, 3.0, 2.0])
FIRST = np.datetime64("2024-01-01")
DARK, READOUT = (30, 34), 8
"""The test's dark weeks, `[first, last)`, and its readout: the dark weeks and four after."""


def _mm(u, kappa):
    return u if kappa == math.inf else u / (1.0 + u / kappa) * (1.0 + 1.0 / kappa) ** 2


def _carried(series, retention, length):
    """The geometric adstock by its definition: week `t` reads its `length` last weeks."""
    w = np.array([retention**lag for lag in range(length)])
    w = w / w.sum()
    return np.array(
        [sum(w[lag] * series[t - lag] for lag in range(length) if t - lag >= 0)
         for t in range(series.size)]
    )  # fmt: skip


def _world(seed=3):
    """A panel of three geos, two channels of their own and one bought nationally, drawn from the
    model: search at retention 0.3 and curvature 2, social on the line, the national channel at
    retention 0.6 and curvature 4."""
    rng = np.random.default_rng(seed)
    share = POPULATION / POPULATION.sum()
    own = rng.gamma(4.0, 25.0, size=(3, WEEKS, 2)) * share[:, None, None]
    national = rng.gamma(4.0, 60.0, size=WEEKS)
    spend = np.concatenate([own, (share[:, None] * national[None, :])[:, :, None]], axis=2)
    truth = ((0.3, 2.0), (0.0, math.inf), (0.6, 4.0))
    margins = np.array([[1.4, 0.7, 0.9], [1.1, 0.8, 1.0], [1.6, 0.6, 0.8]])
    scale = share[:, None] * spend.mean(axis=1).sum(axis=0)[None, :]
    t = np.arange(1, WEEKS + 1)
    kpi = np.empty((3, WEEKS))
    for g in range(3):
        kpi[g] = share[g] * (900.0 + 60.0 * np.sin(2 * np.pi * t / 52.0) + 1.5 * t)
        for c, (r, kappa) in enumerate(truth):
            a = _carried(spend[g, :, c], r, LENGTH)
            kpi[g] += margins[g, c] * scale[g, c] * _mm(a / scale[g, c], kappa)
        kpi[g] += rng.normal(0.0, 4.0 * math.sqrt(share[g]), size=WEEKS)
    panel = MediaPanel(
        dates=FIRST + np.timedelta64(7, "D") * np.arange(WEEKS),
        cadence=Cadence.WEEK,
        geos=GEOS,
        channels=(SEARCH, SOCIAL),
        controls=(),
        kpi=kpi,
        kpi_unit=Revenue(),
        spend=own,
        currency="EUR",
        control_values=np.zeros((3, WEEKS, 0)),
        filled=np.zeros((3, WEEKS), dtype=bool),
        population=POPULATION,
    )
    return allot_national(panel, TV, national)


def _estimate(panel, channel, named, lift, se, dark=DARK, readout=READOUT):
    """A go-dark of `channel` in the geos `named` over the weeks `dark`, read over `readout`
    weeks from the first dark one."""
    end = dark[0] + readout
    group = [panel.geos.index(g) for g in named]
    planned = panel.spend[group, :end, panel.channels.index(channel)].sum(axis=0)
    tested = planned.copy()
    tested[dark[0] : dark[1]] = 0.0
    return LiftEstimate(
        spend=tested,
        planned=planned,
        readout=readout,
        lift=lift,
        se=se,
        geos=frozenset(named),
        end=panel.dates[end - 1],
    )


PANEL = _world()
ESTIMATES = {"search": (_estimate(PANEL, SEARCH, (NORTH, WEST), -260.0, 40.0),)}
DARK_ALONE = {"search": (_estimate(PANEL, SEARCH, (NORTH, WEST), -150.0, 40.0, readout=4),)}
"""A test read over its four dark weeks alone: the readout's first and last week both move."""


def _columns(panel, point, c):
    """Channel `c`'s column in every geo at the grid point `point`, `(geos, weeks)`, and each
    geo's scale, by the module's definition."""
    r, kappa = point
    share = panel.population / panel.population.sum()
    scale = share * panel.spend[:, :, c].mean(axis=1).sum()
    return np.array(
        [
            scale[g] * _mm(_carried(panel.spend[g, :, c], r, LENGTH) / scale[g], kappa)
            for g in range(len(panel.geos))
        ]
    ), scale


def _row(panel, estimate, point, c):
    """The estimate's row at `point`: in each geo it names, the change in the geo's column summed
    over the readout, the geo's spend dark in the dark weeks; 0 in every other geo."""
    r, kappa = point
    history, scale = _columns(panel, point, c)
    z = np.zeros(len(panel.geos))
    end = estimate.spend.size
    went_dark = np.flatnonzero((estimate.spend == 0.0) & (estimate.planned != 0.0))
    for g, geo in enumerate(panel.geos):
        if geo in (estimate.geos or ()):
            dark = panel.spend[g, :, c].copy()
            dark[went_dark] = 0.0
            tested = scale[g] * _mm(_carried(dark, r, LENGTH) / scale[g], kappa)
            z[g] = (tested - history[g])[end - estimate.readout : end].sum()
    return z


def _brute(panel, estimates, model, sel):
    """The log density of every geo's KPI and the lifts at the selection `sel`, and the
    coefficients' posterior mean, `(geos, channels)`, by one Gaussian of all of them: the nuisance
    projected out by a basis of its complement, the model's noise and prior scales given."""
    geos, channels = len(panel.geos), len(panel.channels)
    rows = np.arange(LENGTH - 1, WEEKS)
    t = (rows + 1).astype(float)
    nuisance = np.column_stack(
        [np.ones_like(t)]
        + [f(2 * np.pi * k * t / 52.0) for k in (1, 2) for f in (np.sin, np.cos)]
        + [t]
    )
    complement = np.linalg.qr(nuisance, mode="complete")[0][:, nuisance.shape[1] :]
    kept = complement.shape[1]
    size = channels + geos * channels
    design = np.zeros((geos * kept + sum(len(v) for v in estimates.values()), size))
    data = np.zeros(design.shape[0])
    noise = np.zeros(design.shape[0])
    for g in range(geos):
        block = slice(g * kept, (g + 1) * kept)
        data[block] = complement.T @ panel.kpi[g, rows]
        noise[block] = model.sigma2[g]
        for c in range(channels):
            column = complement.T @ _columns(panel, POINTS[sel[c]], c)[0][g, rows]
            design[block, c] = column
            design[block, channels + g * channels + c] = column
    at = geos * kept
    for name, tests in estimates.items():
        c = panel.channels.index(name)
        for estimate in tests:
            z = _row(panel, estimate, POINTS[sel[c]], c)
            design[at, c] = z.sum()
            design[at, channels + np.arange(geos) * channels + c] = z
            data[at], noise[at] = estimate.lift, estimate.se**2
            at += 1
    prior = np.concatenate([model.prior_sd**2, np.tile((geo_eb.SPREAD * model.margin) ** 2, geos)])
    covariance = design @ np.diag(prior) @ design.T + np.diag(noise)
    density = stats.multivariate_normal(np.zeros(data.size), covariance).logpdf(data)
    mean = prior * (design.T @ np.linalg.solve(covariance, data))
    constant = -0.5 * np.sum(np.log(2 * np.pi * noise))
    return density - constant, mean[:channels][None, :] + mean[channels:].reshape(geos, channels)


@pytest.mark.skipif(shutil.which("maxima") is None, reason="maxima is not on PATH")
def test_maxima_checks_the_identities_the_fit_and_the_plan_rely_on():
    """`validation/geo_eb.mac`: Frisch, Waugh and Lovell; the least-squares split; the Gaussian
    integral; the closed form assembled against the brute force, exactly; the curve scaled to its
    margin; a national channel as one cell and a cell's slope: each a PASS."""
    script = Path(__file__).parents[1] / "validation" / "geo_eb.mac"
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
    assert out.count("(PASS)") - out.count('"(PASS)"') == 15


@pytest.mark.parametrize(
    "estimates",
    [{}, ESTIMATES, DARK_ALONE],
    ids=["no test", "one test", "a test read over its dark weeks alone"],
)
def test_the_marginal_likelihood_and_the_posterior_mean_are_the_brute_force_gaussians(estimates):
    """At the reference point, at a point of every channel curved and at one mixed, with and
    without a lift row: the closed form's log marginal likelihood is the joint Gaussian's log
    density less a constant no selection moves, and its posterior mean the Gaussian's conditional
    mean. The least squares' ridge of 1e-9 is the distance allowed."""
    model = geo_eb._Model(PANEL, estimates, LENGTH)
    assert len(model.tests) == sum(len(v) for v in estimates.values())
    selections = np.array(
        [
            [POINTS.index(geo_eb.REFERENCE)] * 3,
            [POINTS.index((0.3, 2.0)), POINTS.index((0.6, 0.5)), POINTS.index((0.9, 8.0))],
            [POINTS.index((0.0, 0.25)), POINTS.index((0.45, math.inf)), POINTS.index((0.6, 4.0))],
        ]
    )
    ll, beta = model.loglik(selections)
    for i, sel in enumerate(selections):
        density, mean = _brute(PANEL, estimates, model, sel)
        assert ll[i] == pytest.approx(density, rel=1e-7, abs=0.0)
        np.testing.assert_allclose(beta[i], mean, rtol=1e-6, atol=0.0)
    assert len({round(float(v), 6) for v in ll}) == 3


def test_the_noise_and_the_pooling_scales_are_the_reference_fits():
    """Each geo's noise variance is its residual sum of squares at the reference point over its
    residual degrees of freedom, by least squares on the nuisance and the channels together; the
    channel's `m` and its prior's width follow the module's formulas from those coefficients."""
    fit = fit_pooled_geos(PANEL, {}, LENGTH)
    rows = np.arange(LENGTH - 1, WEEKS)
    t = (rows + 1).astype(float)
    nuisance = np.column_stack(
        [np.ones_like(t)]
        + [f(2 * np.pi * k * t / 52.0) for k in (1, 2) for f in (np.sin, np.cos)]
        + [t]
    )
    coefficients, precisions = np.empty((3, 3)), np.empty((3, 3))
    for g in range(3):
        columns = np.column_stack(
            [_columns(PANEL, geo_eb.REFERENCE, c)[0][g, rows] for c in range(3)]
        )
        design = np.column_stack([nuisance, columns])
        solved, residual = np.linalg.lstsq(design, PANEL.kpi[g, rows], rcond=None)[:2]
        variance = float(residual[0]) / (rows.size - design.shape[1])
        assert fit.sigma[g] ** 2 == pytest.approx(variance, rel=1e-6, abs=0.0)
        partial = columns - nuisance @ np.linalg.lstsq(nuisance, columns, rcond=None)[0]
        coefficients[g] = solved[nuisance.shape[1] :]
        precisions[g] = np.einsum("tc,tc->c", partial, partial) / variance
    mean = (precisions * coefficients).sum(axis=0) / precisions.sum(axis=0)
    margin = np.sqrt(mean**2 + 1.0 / precisions.sum(axis=0))
    np.testing.assert_allclose(fit.margin, margin, rtol=1e-6, atol=0.0)
    width = geo_eb.PRIOR_WIDTH * np.maximum(np.abs(coefficients).max(axis=0), margin)
    np.testing.assert_allclose(fit.prior_sd, width, rtol=1e-6, atol=0.0)
    assert fit.rows == rows.size
    assert fit.tests == 0


def test_the_fit_finds_the_curved_channels_and_keeps_a_posterior_over_each_grid():
    fit = fit_pooled_geos(PANEL, {}, LENGTH)
    assert fit.channels == (SEARCH, SOCIAL, TV)
    assert fit.national == frozenset({TV})
    for mixture in fit.mixtures:
        assert mixture.weights.sum() == pytest.approx(1.0, rel=1e-12, abs=0.0)
        assert np.all(mixture.weights > geo_eb.KEPT)
        assert mixture.coefficients.shape == (mixture.points.size, 3)
    for c, j in enumerate(fit.selected):
        assert j in fit.mixtures[c].points
    # search bends and social is on the line, so search's curvature reads the lower of the two
    assert POINTS[fit.selected[0]][1] < POINTS[fit.selected[1]][1]


def test_a_channels_mixture_is_its_grids_posterior_at_the_other_channels_selection():
    """With the other channels at their selected points, a channel's weights are its grid's
    marginal likelihoods normalised, those above `KEPT` of the whole kept and normalised again,
    and its coefficients under a kept point are that channel's posterior means there."""
    fit = fit_pooled_geos(PANEL, ESTIMATES, LENGTH)
    model = geo_eb._Model(PANEL, ESTIMATES, LENGTH)
    sizes = []
    for c, mixture in enumerate(fit.mixtures):
        around = np.repeat(np.array(fit.selected)[None, :], len(POINTS), axis=0)
        around[:, c] = np.arange(len(POINTS))
        ll, beta = model.loglik(around)
        share = np.exp(ll - ll.max())
        share /= share.sum()
        np.testing.assert_array_equal(mixture.points, np.flatnonzero(share > geo_eb.KEPT))
        kept = ll[mixture.points]
        np.testing.assert_allclose(
            np.log(mixture.weights / mixture.weights.max()), kept - kept.max(), rtol=0.0, atol=1e-9
        )
        np.testing.assert_array_equal(mixture.coefficients, beta[mixture.points][:, :, c])
        assert not np.array_equal(mixture.coefficients, beta[mixture.points][:, :, (c + 1) % 3])
        sizes.append(mixture.points.size)
    assert max(sizes) > 1


def test_the_rules_constants_are_the_frozen_ones():
    """The plans the rule was frozen with were made at these: a constant moved is another rule."""
    assert geo_eb.RETENTIONS == (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9)
    assert list(geo_eb.CURVATURES) == [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, math.inf]
    assert list(geo_eb.REFERENCE) == [0.45, math.inf]
    assert (geo_eb.YEAR, geo_eb.HARMONICS, geo_eb.NUISANCE) == (52.0, 2, 6)
    assert (geo_eb.SPREAD, geo_eb.PRIOR_WIDTH, geo_eb.SWEEPS, geo_eb.KEPT) == (0.4, 10.0, 3, 1e-4)
    assert (geo_eb.RIDGE, geo_eb.FLOOR, geo_eb.AGREE) == (1e-9, 1e-12, 1e-9)
    assert geo_eb.VARIATION == 1e-6
    assert (GRID_POINTS, BISECTIONS, FEASIBLE) == (401, 200, 1e-12)


def test_a_lift_read_almost_without_error_is_the_fits_own_reading_of_the_test():
    """With a standard error of a thousandth of the lift, under each grid point the channel's
    posterior keeps, the tested geos' coefficients on the test's row give the lift."""
    tight = {"search": (_estimate(PANEL, SEARCH, (NORTH, WEST), -260.0, 0.26),)}
    fit = fit_pooled_geos(PANEL, tight, LENGTH)
    assert fit.tests == 1
    mixture = fit.mixtures[0]
    for j, b in zip(mixture.points, mixture.coefficients, strict=True):
        z = _row(PANEL, tight["search"][0], POINTS[j], 0)
        assert z[1] == 0.0
        assert float(z @ b) == pytest.approx(-260.0, rel=5e-3, abs=0.0)
    loose = fit_pooled_geos(PANEL, ESTIMATES, LENGTH).mixtures[0]
    j = int(loose.points[0])
    z = _row(PANEL, ESTIMATES["search"][0], POINTS[j], 0)
    assert abs(float(z @ loose.coefficients[0]) + 260.0) > 0.26


def test_what_the_fit_cannot_read_is_refused():
    with pytest.raises(ValueError, match="the panel has none"):
        fit_pooled_geos(dataclasses.replace(_bare(), population=None), {}, LENGTH)
    with pytest.raises(ValueError, match="no residual degree of freedom"):
        fit_pooled_geos(PANEL, {}, WEEKS - 8)
    with pytest.raises(ValueError, match="a whole number of weeks"):
        fit_pooled_geos(PANEL, {}, 0)
    row = ESTIMATES["search"][0]
    with pytest.raises(ValueError, match="which the panel does not have"):
        fit_pooled_geos(PANEL, {"radio": (row,)}, LENGTH)
    with pytest.raises(ValueError, match=r"in a share 0\.5 of its geos' market"):
        fit_pooled_geos(PANEL, {"search": (dataclasses.replace(row, share=0.5),)}, LENGTH)
    with pytest.raises(ValueError, match="not a period of the panel"):
        fit_pooled_geos(PANEL, {"search": (dataclasses.replace(row, end=None),)}, LENGTH)
    with pytest.raises(ValueError, match="plans spend other than its geos'"):
        fit_pooled_geos(PANEL, {"social": (row,)}, LENGTH)
    still = PANEL.kpi.copy()
    still[1] = 0.0
    with pytest.raises(ValueError, match=r"the KPI of the geos \['south'\] does not move"):
        fit_pooled_geos(dataclasses.replace(PANEL, kpi=still), {}, LENGTH)


def test_a_channel_that_moves_in_no_geo_is_refused_by_name():
    """A channel never spent, or spent the same every week in every geo, leaves a reference
    column of 0 or of rounding once each geo's nuisance is out: its margin is 0 over 0 or noise
    over noise, and the pooling's scales are read off it. It is refused by name, own or bought
    nationally, and so is one that moves by less than a thousandth of its size."""
    own = _bare()
    never = own.spend.copy()
    never[:, :, 0] = 0.0
    with pytest.raises(ValueError, match=r"the channels \['search'\] move in no geo"):
        fit_pooled_geos(dataclasses.replace(own, spend=never), {}, LENGTH)
    flat = own.spend.copy()
    flat[:, :, 1] = flat[:, :, 1].mean(axis=1, keepdims=True)
    with pytest.raises(ValueError, match=r"the channels \['social'\] move in no geo"):
        fit_pooled_geos(dataclasses.replace(own, spend=flat), {}, LENGTH)
    both = never.copy()
    both[:, :, 1] = flat[:, :, 1]
    with pytest.raises(ValueError, match=r"the channels \['search', 'social'\] move in no"):
        fit_pooled_geos(dataclasses.replace(own, spend=both), {}, LENGTH)
    for national in (np.full(WEEKS, 240.0), np.zeros(WEEKS)):
        with pytest.raises(ValueError, match=r"the channels \['tv'\] move in no geo"):
            fit_pooled_geos(allot_national(own, TV, national), {}, LENGTH)
    rng = np.random.default_rng(8)
    for jitter, refused in ((1e-4, True), (1e-1, False)):
        near = own.spend.copy()
        near[:, :, 0] = flat[:, :, 0].mean(axis=1, keepdims=True) * np.exp(
            jitter * rng.standard_normal((3, WEEKS))
        )
        if refused:
            with pytest.raises(ValueError, match="move in no geo"):
                fit_pooled_geos(dataclasses.replace(own, spend=near), {}, LENGTH)
        else:
            assert np.all(
                np.isfinite(
                    fit_pooled_geos(dataclasses.replace(own, spend=near), {}, LENGTH).margin
                )
            )


def test_a_channel_that_does_not_move_in_one_geo_takes_the_channels_margin_there():
    """A geo where a channel is never spent, or never moves, is no reason to refuse: its column
    there holds nothing, so its coefficient there is the channel's pooled margin, which the
    other geos give. The closed form is the brute force's on such a panel too."""
    for level in (0.0, 35.0):
        spend = PANEL.spend.copy()
        spend[1, :, 0] = level
        panel = dataclasses.replace(PANEL, spend=spend)
        model = geo_eb._Model(panel, {}, LENGTH)
        sel = np.array([POINTS.index(geo_eb.REFERENCE)] * 3)
        ll, beta = model.loglik(sel[None, :])
        density, mean = _brute(panel, {}, model, sel)
        assert ll[0] == pytest.approx(density, rel=1e-7, abs=0.0)
        np.testing.assert_allclose(beta[0], mean, rtol=1e-5, atol=0.0)
        others = beta[0][[0, 2], 0]
        assert others.min() < beta[0][1, 0] < others.max()
        fit = fit_pooled_geos(panel, {}, LENGTH)
        assert all(m.points.size >= 1 for m in fit.mixtures)
        assert np.all(np.isfinite(fit.margin))


def test_a_likelihood_that_is_not_finite_is_refused():
    """A KPI whose squares overflow is finite, so the panel holds it, and every likelihood is
    NaN: the fit says so, where it would have kept no grid point and planned the status quo."""
    with (
        pytest.raises(FloatingPointError, match="not finite at 63 of 63 selections"),
        np.errstate(over="ignore", invalid="ignore"),
    ):
        fit_pooled_geos(dataclasses.replace(PANEL, kpi=PANEL.kpi * 1e155), {}, LENGTH)


def test_an_estimate_read_in_the_burn_in_is_refused_by_its_row():
    """The kernel's first `LENGTH - 1` weeks are not read: the adstock there misses the spend
    before the panel. An estimate whose readout begins in them is refused by its channel and
    its end, and one that begins at the first week read is read."""
    early = _estimate(PANEL, SEARCH, (NORTH, WEST), -90.0, 40.0, dark=(2, 4), readout=4)
    with pytest.raises(
        ValueError,
        match=r"search's estimate ending on 2024-02-05 reads from the panel's period 3, in the"
        r" kernel's burn-in of 3 periods",
    ):
        fit_pooled_geos(PANEL, {"search": (early,)}, LENGTH)
    first = _estimate(PANEL, SEARCH, (NORTH, WEST), -90.0, 40.0, dark=(3, 5), readout=4)
    assert fit_pooled_geos(PANEL, {"search": (first,)}, LENGTH).tests == 1
    assert fit_pooled_geos(PANEL, {"search": (early,)}, 3).tests == 1


@pytest.mark.parametrize("unit", [1e-15, 1e-9, 1e9])
def test_the_fit_and_the_plan_are_the_same_in_another_unit_of_money(unit):
    """The KPI, the spend, the test's lift and standard error, the box and the budget in a unit
    `unit` times as large: the same grid points selected and kept, the same weights, each geo's
    noise the same in the unit, and the plan the same in it."""

    def planned(unit):
        panel = dataclasses.replace(PANEL, kpi=PANEL.kpi * unit, spend=PANEL.spend * unit)
        row = ESTIMATES["search"][0]
        row = dataclasses.replace(
            row,
            spend=row.spend * unit,
            planned=row.planned * unit,
            lift=row.lift * unit,
            se=row.se * unit,
        )
        fit = fit_pooled_geos(panel, {"search": (row,)}, LENGTH)
        cells = [fit.cell(g, c) for c in (SEARCH, SOCIAL) for g in GEOS] + [fit.cell(None, TV)]
        quo = np.array([cell.history[-PERIODS:].mean() for cell in cells])
        plan = plan_mixture(cells, 0.3 * quo, 2.5 * quo, quo.sum() * PERIODS, PERIODS)
        return fit, plan.weekly / unit, quo / unit

    fit, weekly, quo = planned(1.0)
    assert np.abs(weekly - quo).max() > 0.1 * quo.max()
    other, scaled, _ = planned(unit)
    assert other.selected == fit.selected
    for theirs, ours in zip(other.mixtures, fit.mixtures, strict=True):
        np.testing.assert_array_equal(theirs.points, ours.points)
        np.testing.assert_allclose(theirs.weights, ours.weights, rtol=1e-8, atol=0.0)
    np.testing.assert_allclose(other.sigma / unit, fit.sigma, rtol=1e-10, atol=0.0)
    np.testing.assert_allclose(scaled, weekly, rtol=1e-10, atol=0.0)


def _bare():
    """`PANEL` without its national channel, which a panel without populations cannot hold."""
    return dataclasses.replace(
        PANEL, channels=(SEARCH, SOCIAL), spend=PANEL.spend[:, :, :2], national={}
    )


def _cells(seed=11, lines=()):
    """Six cells of two components each, histories drawn; the cells in `lines` are lines."""
    rng = np.random.default_rng(seed)
    cells = []
    for i in range(6):
        line = i in lines
        cells.append(
            CellResponse(
                history=rng.gamma(4.0, 20.0, size=30),
                scale=float(rng.uniform(60.0, 120.0)),
                length=LENGTH,
                weights=np.array([1.0]) if line else np.array([0.6, 0.4]),
                retentions=np.array([0.3]) if line else np.array([0.3, 0.6]),
                curvatures=np.array([math.inf]) if line else np.array([1.0, 4.0]),
                coefficients=rng.uniform(0.6, 1.6, size=1 if line else 2),
            )
        )
    return cells


LOWER, UPPER = np.full(6, 20.0), np.full(6, 200.0)


def test_a_cells_slope_is_its_worths_derivative():
    cell = _cells()[0]
    w, step = np.array([35.0, 90.0, 170.0]), 1e-4
    numeric = (cell.worth(w + step, PERIODS) - cell.worth(w - step, PERIODS)) / (2 * step)
    np.testing.assert_allclose(cell.slope(w, PERIODS), numeric, rtol=1e-8, atol=0.0)
    # with no carryover and a kernel of one week, a line's worth is its margin times the spend
    line = CellResponse(
        np.zeros(5),
        50.0,
        1,
        np.array([1.0]),
        np.array([0.0]),
        np.array([math.inf]),
        np.array([1.3]),
    )
    assert line.worth(np.array([70.0]), PERIODS)[0] == pytest.approx(
        1.3 * 70.0 * PERIODS, rel=1e-12, abs=0.0
    )


@pytest.mark.parametrize("rate", [260.0, 480.0, 900.0])
def test_the_plan_spends_the_budget_in_the_box_at_one_price(rate):
    """Concave cells: every cell inside its box spends where its slope is the price, a cell at
    its floor is no steeper and one at its ceiling no flatter, to the grid's reading of a slope;
    and no feasible plan drawn at random is worth more."""
    cells = _cells()
    plan = plan_mixture(cells, LOWER, UPPER, rate * PERIODS, PERIODS)
    weekly = plan.weekly
    assert weekly.sum() == pytest.approx(rate, rel=1e-12, abs=0.0)
    assert np.all(weekly >= LOWER)
    assert np.all(weekly <= UPPER)
    slopes = np.array([float(c.slope(weekly[i : i + 1], PERIODS)[0]) for i, c in enumerate(cells)])
    inside = (weekly > LOWER) & (weekly < UPPER)
    assert inside.sum() >= 2
    np.testing.assert_allclose(slopes[inside], plan.price, rtol=1e-4, atol=0.0)
    assert np.all(slopes[weekly == LOWER] <= plan.price * (1 + 1e-4))
    assert np.all(slopes[weekly == UPPER] >= plan.price * (1 - 1e-4))

    def worth(spend):
        return sum(float(c.worth(spend[i : i + 1], PERIODS)[0]) for i, c in enumerate(cells))

    rng = np.random.default_rng(5)
    best = worth(weekly)
    for _ in range(200):
        other = LOWER + rng.dirichlet(np.ones(6)) * (rate - LOWER.sum())
        if np.all(other <= UPPER):
            assert worth(other) <= best * (1 + 1e-9)


def test_lines_take_the_budget_steepest_first_and_the_last_one_takes_the_rest():
    """Every cell a line: its slope is flat, so the plan fills the steepest cells to their
    ceilings and the cell at the price takes what is left, the budget spent exactly."""
    cells = _cells(lines=range(6))
    rate = 610.0
    plan = plan_mixture(cells, LOWER, UPPER, rate * PERIODS, PERIODS)
    slopes = np.array([float(c.slope(np.array([50.0]), PERIODS)[0]) for c in cells])
    expected = LOWER.copy()
    left = rate - expected.sum()
    for i in np.argsort(-slopes):
        step = min(UPPER[i] - expected[i], left)
        expected[i] += step
        left -= step
    np.testing.assert_allclose(plan.weekly, expected, rtol=1e-9, atol=1e-9)
    assert plan.weekly.sum() == pytest.approx(rate, rel=1e-12, abs=0.0)


def test_a_cell_that_is_not_concave_gets_a_plan_on_the_budget_that_is_not_the_best():
    """A cell whose mixture keeps a point with a coefficient below 0, beside a concave one: its
    slope falls from 10.83 at no spend to 0.99 at 21, rises to 2.23 at 80.25 and falls again.
    The planner reads it as the least at or below each spend, so the cell stops where its slope
    first meets the price, at 14.18, short of the rise. The plan spends the budget in the box;
    the best plan with the cell at one of the planner's grid spends, 93.75, is worth 0.0027 of
    the budget more."""
    bent = CellResponse(
        history=np.zeros(1),
        scale=100.0,
        length=1,
        weights=np.full(3, 1.0 / 3.0),
        retentions=np.zeros(3),
        curvatures=np.array([0.25, 0.5, 1.0]),
        coefficients=np.array([0.5, -2.0, 2.0]),
    )
    concave = CellResponse(
        history=np.zeros(1),
        scale=100.0,
        length=1,
        weights=np.array([1.0]),
        retentions=np.array([0.0]),
        curvatures=np.array([1.0]),
        coefficients=np.array([0.3]),
    )
    low, high, rate = np.zeros(2), np.full(2, 300.0), 260.0
    grid = np.linspace(low[0], high[0], GRID_POINTS)
    slopes = bent.slope(grid, PERIODS)
    turns = np.flatnonzero(np.diff(np.sign(np.diff(slopes))) != 0.0) + 1
    np.testing.assert_allclose(grid[turns], [21.0, 80.25], rtol=1e-12, atol=0.0)
    assert slopes[0] == pytest.approx(10.833, rel=1e-4, abs=0.0)
    np.testing.assert_allclose(slopes[turns], [0.9941, 2.2318], rtol=1e-4, atol=0.0)

    plan = plan_mixture([bent, concave], low, high, rate * PERIODS, PERIODS)
    assert plan.weekly.sum() == pytest.approx(rate, rel=1e-12, abs=0.0)
    assert np.all(plan.weekly >= low)
    assert np.all(plan.weekly <= high)
    np.testing.assert_allclose(plan.weekly, [14.18116, 245.81884], rtol=1e-6, atol=0.0)
    assert slopes[turns[0]] < plan.price < slopes[turns[1]]

    def worth(first):
        return bent.worth(first, PERIODS) + concave.worth(rate - first, PERIODS)

    on_grid = grid[(rate - grid >= low[1]) & (rate - grid <= high[1])]
    values = worth(on_grid)
    assert on_grid[values.argmax()] == pytest.approx(93.75, rel=1e-12, abs=0.0)
    gap = (values.max() - worth(plan.weekly[:1])[0]) / (rate * PERIODS)
    assert gap == pytest.approx(0.002737, rel=1e-3, abs=0.0)


def test_a_plan_at_the_boxs_ends_and_what_no_plan_spends():
    cells = _cells()
    low = plan_mixture(cells, LOWER, UPPER, LOWER.sum() * PERIODS, PERIODS)
    np.testing.assert_allclose(low.weekly, LOWER, rtol=1e-12, atol=0.0)
    high = plan_mixture(cells, LOWER, UPPER, UPPER.sum() * PERIODS, PERIODS)
    np.testing.assert_allclose(high.weekly, UPPER, rtol=1e-12, atol=0.0)
    with pytest.raises(ValueError, match="no plan in the box"):
        plan_mixture(cells, LOWER, UPPER, 1.01 * UPPER.sum() * PERIODS, PERIODS)
    with pytest.raises(ValueError, match="no plan in the box"):
        plan_mixture(cells, UPPER, LOWER, 500.0 * PERIODS, PERIODS)
    with pytest.raises(ValueError, match="a cell's"):
        plan_mixture(cells, LOWER[:5], UPPER[:5], 500.0 * PERIODS, PERIODS)
    with pytest.raises(ValueError, match="a whole number of periods"):
        plan_mixture(cells, LOWER, UPPER, 500.0, 0)
    assert GRID_POINTS == 401


def test_a_channel_bought_nationally_is_one_cell_worth_its_geos_summed():
    """The national cell's worth at a national spend `W` is the sum over the geos of each geo's
    worth at its population share of `W`, under each kept grid point and its posterior-mean
    coefficients; its scale is the channel's national mean spend."""
    fit = fit_pooled_geos(PANEL, ESTIMATES, LENGTH)
    cell = fit.cell(None, TV)
    c = fit.channels.index(TV)
    share = POPULATION / POPULATION.sum()
    national = PANEL.spend[:, :, c].sum(axis=0)
    assert cell.scale == pytest.approx(float(national.mean()), rel=1e-12, abs=0.0)
    np.testing.assert_allclose(cell.history, national, rtol=1e-12, atol=0.0)
    spends = np.array([0.0, 150.0, 240.0, 700.0])
    mixture = fit.mixtures[c]
    expected = np.zeros(spends.size)
    for n, j in enumerate(mixture.points):
        r, kappa = POINTS[j]
        weight = mixture.weights[n]
        for g in range(3):
            scale = share[g] * national.mean()
            for i, spend in enumerate(spends):
                series = np.concatenate(
                    [PANEL.spend[g, :, c], np.full(PERIODS, share[g] * spend), np.zeros(LENGTH - 1)]
                )
                carried = _carried(series, r, LENGTH)[WEEKS:]
                margin = mixture.coefficients[n, g]
                expected[i] += weight * margin * scale * _mm(carried / scale, kappa).sum()
    np.testing.assert_allclose(cell.worth(spends, PERIODS), expected, rtol=1e-10, atol=0.0)
    own = fit.cell(SOUTH, SEARCH)
    np.testing.assert_array_equal(own.history, PANEL.spend[1, :, 0])
    np.testing.assert_array_equal(own.coefficients, fit.mixtures[0].coefficients[:, 1])
    with pytest.raises(ValueError, match="one cell of no geo"):
        fit.cell(NORTH, TV)
    with pytest.raises(ValueError, match="one cell of no geo"):
        fit.cell(None, SEARCH)
    with pytest.raises(ValueError, match="no channel"):
        fit.cell(NORTH, Channel("radio"))
    with pytest.raises(ValueError, match="no geo"):
        fit.cell(Geo("east"), SEARCH)
