"""Lift tests as anchors (`mixdyn.calibration.anchors`): on eight of M2's worlds each channel's
return per euro and slope statistic, the regression's bias, the anchored margins, their shrunk means
and the corner are the research code's, which found the rule, and so are the tests alone shrunk and
planned; the anchored posterior is the generalised least-squares form of the same posterior; tests
whose returns rise with the spend read a positive slope and tests at one spend read none; and lift
rows the reading does not take are refused. The curve columns: a logistic column's rate and a
Michaelis-Menten column's half-saturation read their elasticity (closed forms checked in Maxima);
tests and a regression margin made exactly by one column put the posterior's most weight on it and
its shape on the channel's margin curve; with the line's prior at 1 the curve is a straight line,
and with it at 0 the line weighs nothing; a curve's rise is the integral of its margin; and rows
the curves do not take are refused."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad

from gs_worlds import WORLDS, name
from mixdyn.calibration.anchors import (
    ELASTICITIES,
    FLAT,
    anchor,
    logistic_rate,
    read_curves,
    read_tests,
)
from mixdyn.decisions.margin_plan import plan_corner, shrink_margins
from mixdyn.models.margin_mmm import adstock

RELATIVE = 1e-9


def _read(world):
    lift = world["lift"]
    return read_tests(
        tuple(world["channels"]),
        world["status_quo"],
        tuple(lift["channel"]),
        lift["x"],
        lift["delta_y"],
        lift["sigma"],
    )


@pytest.mark.parametrize("world", WORLDS, ids=name)
def test_each_channels_return_and_slope_statistic_are_the_research_codes(world):
    read = _read(world)
    research = world["research"]
    np.testing.assert_allclose(read.roi, research["roi"], rtol=RELATIVE, atol=0.0)
    np.testing.assert_allclose(read.roi_se, research["roi_se"], rtol=RELATIVE, atol=0.0)
    np.testing.assert_allclose(read.slope_t, research["t"], rtol=RELATIVE, atol=0.0)
    assert read.count == tuple(world["lift"]["channel"].count(c) for c in world["channels"])


@pytest.mark.parametrize("world", WORLDS, ids=name)
def test_the_anchored_margins_their_shrunk_means_and_the_corner_are_the_research_codes(world):
    research = world["research"]
    anchored = anchor(research["mean"], research["cov"], research["roi"], research["roi_se"])
    assert anchored.tau == pytest.approx(research["tau_b"], rel=RELATIVE, abs=0.0)
    np.testing.assert_allclose(anchored.mean, research["anchored"], rtol=RELATIVE, atol=0.0)
    np.testing.assert_allclose(anchored.cov, research["anchored_cov"], rtol=RELATIVE, atol=0.0)
    shrunk = shrink_margins(anchored.mean, anchored.cov)
    assert shrunk is not None
    np.testing.assert_allclose(shrunk.mean, research["shrunk"], rtol=RELATIVE, atol=0.0)
    budget = float(np.sum(world["status_quo"]))
    plan = plan_corner(research["shrunk"], world["lower"], world["upper"], budget)
    np.testing.assert_array_equal(plan, research["plan"])


@pytest.mark.parametrize("world", WORLDS, ids=name)
def test_the_tests_alone_shrunk_and_planned_are_the_research_codes(world):
    research = world["research"]
    alone = shrink_margins(research["roi"], np.diag(np.square(research["roi_se"])))
    assert alone is not None
    np.testing.assert_allclose(
        alone.mean, research["tests_alone"]["shrunk"], rtol=RELATIVE, atol=0.0
    )
    budget = float(np.sum(world["status_quo"]))
    plan = plan_corner(alone.mean, world["lower"], world["upper"], budget)
    np.testing.assert_array_equal(plan, research["tests_alone"]["plan"])


def test_the_family_2_world_the_fixture_holds_for_the_bias_reads_one():
    """The fixture holds a world whose regression and tests disagree past their errors."""
    assert any(w["research"]["tau_b"] > 0.0 for w in WORLDS)


@pytest.mark.parametrize("tau", [0.0, 0.7, 3.0])
def test_the_anchored_posterior_is_the_tests_moved_towards_the_regression_by_a_gain(tau):
    """The precision form `P^-1 ((V + tau^2 I)^-1 b + S^-1 r)`, `P = (V + tau^2 I)^-1 + S^-1`, is
    the generalised least-squares form `r + K (b - r)`, `K = S (S + V + tau^2 I)^-1`, of
    covariance `S - S (S + V + tau^2 I)^-1 S`, `S = diag(se^2)`; `tau` is set by giving the margins
    a disagreement with the tests whose moments read it."""
    v = np.array([[0.4, 0.1, 0.05], [0.1, 0.3, -0.03], [0.05, -0.03, 0.2]])
    se = np.array([0.5, 0.8, 0.3])
    r = np.array([1.0, 2.0, 0.5])
    d = np.sqrt(np.diag(v) + se**2 + tau**2) * np.array([1.0, -1.0, 1.0])
    anchored = anchor(r + d, v, r, se)
    assert anchored.tau == pytest.approx(tau, rel=1e-12, abs=1e-12)
    s = np.diag(se**2)
    whole = np.linalg.inv(s + v + anchored.tau**2 * np.eye(3))
    np.testing.assert_allclose(anchored.mean, r + s @ whole @ d, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(anchored.cov, s - s @ whole @ s, rtol=1e-10, atol=1e-14)


def test_the_bias_reads_0_where_the_margins_agree_with_the_tests_within_their_errors():
    v = np.diag([0.4, 0.3, 0.2])
    anchored = anchor([1.1, 1.9, 0.6], v, [1.0, 2.0, 0.5], [0.5, 0.5, 0.5])
    assert anchored.tau == 0.0


def _tests(levels, rho, se=0.1, s=1.0):
    """One channel's tests at spends `levels`, each reading the return `rho` at its spend."""
    x = np.asarray(levels, dtype=float)
    return read_tests(
        ("a",),
        [s],
        ("a",) * x.size,
        x,
        -np.asarray(rho, dtype=float) * x,
        np.full(x.size, se) * x,
    )


def test_tests_whose_return_rises_with_the_spend_read_a_positive_slope():
    rising = _tests([0.5, 1.0, 2.0, 4.0], [1.0, 1.3, 1.6, 1.9])
    assert rising.slope_t[0] > 1.645
    falling = _tests([0.5, 1.0, 2.0, 4.0], [1.9, 1.6, 1.3, 1.0])
    assert falling.slope_t[0] < -1.645


def test_tests_at_one_spend_read_no_slope():
    flat = _tests([2.0, 2.0, 2.0 * (1.0 + FLAT / 10.0)], [1.0, 3.0, 2.0])
    assert flat.slope_t[0] == 0.0
    assert flat.roi[0] == pytest.approx(2.0, rel=1e-12, abs=0.0)
    assert flat.roi_se[0] == pytest.approx(0.1 / np.sqrt(3.0), rel=1e-12, abs=0.0)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda a: ([1.0, 2.0, 3.0], *a[1:]), "are \\(channels,\\)"),
        (lambda a: (a[0], np.eye(3), *a[2:]), "are \\(channels,\\)"),
        (lambda a: (*a[:2], [1.0], a[3]), "are \\(channels,\\)"),
        (lambda a: (a[0], [[0.4, np.inf], [0.1, 0.3]], *a[2:]), "are finite"),
        (lambda a: (*a[:2], [1.0, np.nan], a[3]), "are finite"),
        (lambda a: (*a[:3], [0.5, 0.0]), "standard errors > 0"),
    ],
)
def test_margins_and_returns_the_anchoring_does_not_take_are_refused(change, match):
    good = ([1.0, 2.0], [[0.4, 0.1], [0.1, 0.3]], [1.5, 1.0], [0.5, 0.8])
    anchor(*good)
    with pytest.raises(ValueError, match=match):
        anchor(*change(good))


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda a: (*a[:2], ("a", "c"), *a[3:]), "each of them has a row"),
        (lambda a: (*a[:2], ("a", "a"), *a[3:]), "each of them has a row"),
        (lambda a: (*a[:3], [1.0, 0.0], *a[4:]), "positive and finite"),
        (lambda a: (*a[:5], [1.0, -1.0]), "positive and finite"),
        (lambda a: (*a[:4], [1.0, np.nan], a[5]), "positive and finite"),
        (lambda a: (a[0], [1.0, 0.0], *a[2:]), "positive and finite"),
        (lambda a: (*a[:3], [1.0], *a[4:]), "equal lengths"),
    ],
)
def test_lift_rows_the_reading_does_not_take_are_refused(change, match):
    good = (("a", "b"), [1.0, 1.0], ("a", "b"), [1.0, 2.0], [-1.0, -2.0], [0.1, 0.2])
    read_tests(*good)
    with pytest.raises(ValueError, match=match):
        read_tests(*change(good))


def test_a_logistic_columns_rate_reads_its_elasticity():
    """`tanh(k a)` has the elasticity `2 k s / sinh(2 k s)` at `s` (checked in Maxima)."""
    for e in ELASTICITIES:
        u = logistic_rate(float(e))
        assert 2.0 * u / math.sinh(2.0 * u) == pytest.approx(e, rel=1e-12, abs=0.0)
    assert logistic_rate(1.0) == 0.0


KERNEL, DARK, READOUT = 5, 4, 12
RETENTION = (0.5, 0.3)
QUO = np.array([10.0, 4.0])
STARTS = (21, 41, 61, 25, 45, 65)
ROWS = ("a", "a", "a", "b", "b", "b")


def _history():
    rng = np.random.default_rng(7)
    return QUO * rng.uniform(0.3, 2.5, (90, 2))


def _slope(history):
    """A regression read as a linear map: least squares on an intercept and the two channels'
    adstocks, the coefficient on a channel's."""
    carried = [adstock(history[:, c], RETENTION[c], KERNEL, 0.0) for c in range(2)]
    design = np.column_stack([np.ones(history.shape[0]), *carried])
    solve = np.linalg.pinv(design)
    return lambda c, series: float((solve @ series)[1 + c])


def _made_by(history, curve, margin):
    """The regression's margins and the lift rows a world whose channels follow `curve(c, a)`,
    of margin `margin[c]` at the status quo, gives exactly: the regression's reading of the curve
    over its reading of the adstock, and each test's exact lift over its dark spend."""
    slope = _slope(history)
    mean, x, delta_y = [], [], []
    for c in range(2):
        carried = adstock(history[:, c], RETENTION[c], KERNEL, 0.0)
        step = 1e-6 * QUO[c]
        unit = (curve(c, QUO[c] + step) - curve(c, QUO[c] - step)) / (2.0 * step)
        mean.append(margin[c] * slope(c, curve(c, carried)) / unit / slope(c, carried))
        for start in STARTS[3 * c : 3 * c + 3]:
            first = start - 1
            darkened = history[:, c].copy()
            darkened[first : first + DARK] = 0.0
            window = slice(first, first + READOUT)
            lift = np.sum(
                curve(c, carried[window])
                - curve(c, adstock(darkened, RETENTION[c], KERNEL, 0.0)[window])
            )
            spent = float(history[first : first + DARK, c].mean())
            x.append(spent)
            delta_y.append(-margin[c] * lift / unit / DARK)
    return np.array(mean), np.array(x), np.array(delta_y)


def _curves(
    history, mean, x, delta_y, *, start=STARTS, spend=None, retention=RETENTION, line_prior=0.7
):
    return read_curves(
        ("a", "b"),
        QUO,
        ROWS,
        x,
        delta_y,
        1e-3 * np.abs(delta_y),
        start,
        mean,
        np.diag((1e-3 * mean) ** 2),
        history if spend is None else spend,
        _slope(history),
        retention=retention,
        kernel=KERNEL,
        dark=DARK,
        readout=READOUT,
        line_prior=line_prior,
    )


def test_tests_and_a_margin_one_column_made_put_the_most_weight_on_it():
    """Channel `a` follows a Michaelis-Menten curve `a / (K + a)`, `K = s e / (1 - e)`, and
    channel `b` a logistic `tanh(k a)`, each at a grid elasticity: the column that made the
    readings fits them with no residual, so it carries the posterior and its shape."""
    history = _history()
    at = (20, 30)
    half = QUO[0] * ELASTICITIES[at[0]] / (1.0 - ELASTICITIES[at[0]])
    k = logistic_rate(float(ELASTICITIES[at[1]])) / QUO[1]

    def curve(c, a):
        return a / (half + a) if c == 0 else np.tanh(k * a)

    margin = np.array([2.0, 0.7])
    curves = _curves(history, *_made_by(history, curve, margin))
    count = ELASTICITIES.size
    assert curves.half[0, at[0]] == pytest.approx(half, rel=1e-12, abs=0.0)
    assert curves.half[0, at[0]] / (curves.half[0, at[0]] + QUO[0]) == pytest.approx(
        ELASTICITIES[at[0]], rel=1e-12, abs=0.0
    )
    assert curves.rate[1, at[1]] == pytest.approx(k, rel=1e-12, abs=0.0)
    assert (int(np.argmax(curves.weights[0])), int(np.argmax(curves.weights[1]))) == (
        at[0],
        count + at[1],
    )
    np.testing.assert_allclose(curves.weights.sum(axis=1), 1.0, rtol=1e-12, atol=0.0)
    assert np.all(curves.weights[[0, 1], [at[0], count + at[1]]] > 0.99)
    spends = np.linspace(0.5, 2.0, 7)
    np.testing.assert_allclose(
        curves.margin(0, 1.0)(QUO[0] * spends) / curves.margin(0, 1.0)(QUO[:1]),
        ((half + QUO[0]) / (half + QUO[0] * spends)) ** 2,
        rtol=1e-2,
        atol=0.0,
    )
    np.testing.assert_allclose(
        curves.margin(1, 1.0)(QUO[1] * spends) / curves.margin(1, 1.0)(QUO[1:]),
        (np.cosh(k * QUO[1]) / np.cosh(k * QUO[1] * spends)) ** 2,
        rtol=1e-2,
        atol=0.0,
    )
    np.testing.assert_allclose(curves.elasticity(), ELASTICITIES[list(at)], rtol=1e-2, atol=0.0)


def _noisy(history):
    """Readings no column makes exactly: a square-root curve's, moved by noise."""
    rng = np.random.default_rng(11)
    mean, x, delta_y = _made_by(history, lambda _, a: np.sqrt(a), np.array([1.5, 0.9]))
    return mean * rng.uniform(0.8, 1.2, 2), x, delta_y * rng.uniform(0.8, 1.2, 6)


def test_with_the_lines_prior_at_1_each_margin_curve_is_a_straight_line():
    history = _history()
    curves = _curves(history, *_noisy(history), line_prior=1.0)
    np.testing.assert_array_equal(curves.weights[:, -1], 1.0)
    np.testing.assert_array_equal(curves.weights[:, :-1], 0.0)
    np.testing.assert_array_equal(curves.ratio[:, -1], 1.0)
    spends = QUO[0] * np.linspace(0.5, 2.0, 5)
    np.testing.assert_array_equal(curves.margin(0, 1.7)(spends), 1.7)
    np.testing.assert_array_equal(curves.rise(2.0 * QUO, [1.7, 0.4]), QUO)
    np.testing.assert_array_equal(curves.elasticity(), 1.0)


def test_with_the_lines_prior_at_0_the_line_weighs_nothing_and_the_margin_falls():
    history = _history()
    curves = _curves(history, *_noisy(history), line_prior=0.0)
    np.testing.assert_array_equal(curves.weights[:, -1], 0.0)
    np.testing.assert_allclose(curves.weights.sum(axis=1), 1.0, rtol=1e-12, atol=0.0)
    assert np.all(curves.elasticity() < 1.0)
    for c in range(2):
        falling = curves.margin(c, 1.0)(QUO[c] * np.linspace(0.5, 2.0, 50))
        assert np.all(np.diff(falling) < 0.0)


@pytest.mark.parametrize("to", [0.5, 0.8, 1.3, 2.0])
def test_a_curves_rise_is_the_integral_of_its_margin(to):
    """Each column's rise, `(K + s) (p - s) / (K + p)` and
    `cosh(k s)^2 (tanh(k p) - tanh(k s)) / k`, is its shape's integral (checked in Maxima)."""
    history = _history()
    curves = _curves(history, *_noisy(history))
    rise = curves.rise(to * QUO, [1.0, 1.0])
    for c in range(2):
        margin = curves.margin(c, 1.0)
        integral, _ = quad(lambda p, m=margin: float(m(np.array([p]))[0]), QUO[c], to * QUO[c])
        assert rise[c] == pytest.approx(integral, rel=1e-9, abs=0.0)
    # a level at most 0 is a line through its margin at the status quo
    flat = curves.rise(to * QUO, [-1.0, 1.0])
    assert flat[0] == pytest.approx(
        curves.scale()[0].sum() * (to - 1.0) * QUO[0], rel=1e-12, abs=0.0
    )
    assert flat[1] == rise[1]
    np.testing.assert_array_equal(
        curves.margin(0, -1.0)(QUO[0] * np.array([0.5, 2.0])), -curves.scale()[0].sum()
    )


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"start": (21.5, 41, 61, 25, 45, 65)}, "dark weeks lie in the history"),
        ({"start": (0, 41, 61, 25, 45, 65)}, "dark weeks lie in the history"),
        ({"start": (88, 41, 61, 25, 45, 65)}, "dark weeks lie in the history"),
        ({"start": (21, 41, 61)}, "dark weeks lie in the history"),
        ({"line_prior": 1.5}, "a probability"),
        ({"retention": (0.5,)}, "each channel has a retention"),
        ({"spend": np.ones(90)}, "each channel has a retention"),
    ],
)
def test_rows_the_curves_do_not_take_are_refused(change, match):
    history = _history()
    with pytest.raises(ValueError, match=match):
        _curves(history, *_noisy(history), **change)
