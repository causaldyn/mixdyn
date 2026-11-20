"""A lift series holds a test's gap period by period with the covariance of its errors, and the
Bayesian model reads it as one joint normal term of its likelihood: about the channel's gaps as
`chc.response` computes them, at the series' covariance and share, one period's series an estimate
and independent periods the sum of their one-period terms, and nothing else of the model moved.
On a world with no noise a summed lift is fitted as well at every decay, its return moving along a
ridge, where the gaps peak at the world's decay. On the gaps alone the model's most likely channel
is `chc.lift.fit_lift`'s least-squares fit, to `1e-8`. The noise models give the gaps' covariance:
the group's independent periods, and the history's autocorrelation scaled to the placebo error of
the sum.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from chc import response
from chc.lift import GeoArm, LiftTest, fit_lift
from scipy import linalg, optimize, signal, stats

from mixdyn import NATIONAL, Cadence, Channel, Geo, MediaPanel, Revenue
from mixdyn.calibration import LiftEstimate, LiftSeries
from mixdyn.experiments import GroupNoise, PlaceboNoise
from mixdyn.models.bayes_mmm import (
    _TANH_HALF,
    BayesMedia,
    BayesModel,
    BayesSpec,
    HillCurve,
    TanhCurve,
    bayes_model,
    fit_bayes,
)

WEEKS, READOUT, SPAN, SHARE = 104, 12, 30, 0.2
RETENTION, HALF, ROI = 0.5, 1.2, 2.0
"""The worlds' decay, half-saturation point in units of the mean positive spend, and return."""
PLANNED = SHARE * np.full(SPAN, 100.0)


def _run(multiplier: float, weeks: int = 4) -> np.ndarray:
    """The group's spend with the planned spend times `multiplier` over the readout's first
    `weeks`."""
    run = PLANNED.copy()
    run[SPAN - READOUT : SPAN - READOUT + weeks] *= multiplier
    return run


def _ar1(size: int, phi: float) -> np.ndarray:
    """A first-order autoregression's correlation matrix over `size` periods."""
    return phi ** np.abs(np.subtract.outer(np.arange(size), np.arange(size)))


def world(curve: HillCurve | TanhCurve) -> tuple[MediaPanel, response.Channel, BayesSpec]:
    """104 weeks of television, its spend about 100, and the KPI 1000 plus its return, carried
    over at 0.5 by the normalised kernel as long as the panel and saturated at 1.2 times its mean
    positive spend, by a Hill of slope 1 or a tanh, its return on the history's spend 2; and the
    channel as `chc.response` reads it, in the panel's units."""
    rng = np.random.default_rng(20261013)
    spend = np.maximum(100.0 + rng.normal(0.0, 30.0, WEEKS), 0.0)
    half = HALF * float(spend[spend > 0.0].mean())
    kernel = response.GeometricAdstock(RETENTION, length=WEEKS, normalized=True)
    shape = (
        response.Tanh(half / _TANH_HALF)
        if isinstance(curve, TanhCurve)
        else response.Hill(half, 1.0)
    )
    saturated = np.asarray(shape(kernel(spend)))
    channel = response.Channel(kernel, shape, ROI * spend.sum() / saturated.sum())
    panel = MediaPanel(
        dates=np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(WEEKS),
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=(Channel("tv"), Channel("search")),
        controls=(),
        kpi=(1000.0 + np.asarray(channel(spend)) + rng.normal(0.0, 20.0, WEEKS))[None],
        kpi_unit=Revenue(),
        spend=np.stack([spend, np.maximum(50.0 + rng.normal(0.0, 10.0, WEEKS), 0.0)], -1)[None],
        currency="EUR",
        control_values=np.zeros((1, WEEKS, 0)),
        filled=np.zeros((1, WEEKS), dtype=bool),
    )
    spec = BayesSpec((BayesMedia("tv"), BayesMedia("search")), saturation=curve)
    return panel, channel, spec


def _gaps(channel: response.Channel, run: np.ndarray) -> np.ndarray:
    """The group's gap at each readout period, nothing spent before the series, as `chc.response`
    reads it."""
    pad = np.zeros(WEEKS - 1 + READOUT - SPAN)
    gap = channel(np.r_[pad, run] / SHARE) - channel(np.r_[pad, PLANNED] / SHARE)
    return SHARE * np.asarray(gap)[-READOUT:]


def _point(model: BayesModel, **values: float) -> np.ndarray:
    theta = np.zeros(model.dimension)
    for name, value in values.items():
        theta[model.coordinates.index(name)] = value
    return theta


def test_a_series_holds_read_only_copies_and_sums_to_an_estimate() -> None:
    gaps, covariance = np.linspace(-3.0, 1.0, READOUT), 4.0 * _ar1(READOUT, 0.5)
    series = LiftSeries(_run(0.0), PLANNED, gaps, covariance, SHARE)
    gaps[0], covariance[0, 0] = 1e6, 1e6
    assert series.gaps[0] == -3.0
    assert series.covariance[0, 0] == 4.0
    for array in (series.gaps, series.covariance, series.spend):
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 0.0
    summed = series.summed
    assert series.readout == summed.readout == READOUT
    assert summed.lift == pytest.approx(float(np.linspace(-3.0, 1.0, READOUT).sum()), abs=1e-14)
    assert summed.se == pytest.approx(math.sqrt(4.0 * _ar1(READOUT, 0.5).sum()), rel=1e-15, abs=0)
    np.testing.assert_array_equal(summed.spend, series.spend)
    assert summed.share == SHARE


def test_a_nearly_symmetric_covariance_is_held_as_its_symmetric_part() -> None:
    factor = np.random.default_rng(3).normal(size=(READOUT, READOUT))
    covariance = factor @ factor.T + np.eye(READOUT)
    covariance[0, 1] *= 1.0 + 1e-14
    series = LiftSeries(_run(0.0), PLANNED, np.zeros(READOUT), covariance)
    np.testing.assert_array_equal(series.covariance, series.covariance.T)
    assert series.covariance[1, 0] == (covariance[0, 1] + covariance[1, 0]) / 2.0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"gaps": np.zeros((1, READOUT))}, r"non-empty vector of finite numbers, got shape \(1,"),
        ({"gaps": []}, "non-empty vector"),
        ({"gaps": np.r_[np.zeros(READOUT - 1), np.nan]}, "finite numbers"),
        ({"covariance": np.eye(READOUT - 1)}, r"finite and 12 square, got \(11, 11\)"),
        ({"covariance": np.where(np.eye(READOUT) > 0, np.inf, 0.0)}, "finite and 12 square"),
        ({"covariance": np.eye(READOUT) + np.triu(np.ones((READOUT,) * 2), 1)}, "symmetric"),
        ({"covariance": np.ones((READOUT, READOUT))}, "positive definite"),
        ({"covariance": -np.eye(READOUT)}, "positive definite"),
        ({"gaps": np.zeros(SPAN + 1), "covariance": np.eye(SPAN + 1)}, "readout of 31 periods"),
        ({"spend": PLANNED}, "changed no spend"),
        ({"planned": PLANNED[1:]}, "spend runs 30 periods and planned 29"),
        ({"share": 0.0}, "not a share of the market"),
        ({"geos": frozenset()}, "at least one geo"),
    ],
)
def test_a_series_that_cannot_be_is_refused(kwargs: dict[str, Any], match: str) -> None:
    call: dict[str, Any] = {
        "spend": _run(0.0),
        "planned": PLANNED,
        "gaps": np.zeros(READOUT),
        "covariance": np.eye(READOUT),
    }
    with pytest.raises(ValueError, match=match):
        LiftSeries(**(call | kwargs))


@pytest.mark.parametrize(
    ("curve", "name", "span"),
    [(HillCurve(), "tv", SPAN), (TanhCurve(), "search", SPAN), (HillCurve(), "search", 140)],
    ids=["hill-tv", "tanh-search", "hill-search-longer-than-the-panel"],
)
def test_a_series_is_jointly_normal_about_the_channel_s_gaps(
    curve: HillCurve | TanhCurve, name: str, span: int
) -> None:
    """At points across the support, each series' term is the multivariate normal density of its
    gaps over the share, in the panel's units, about the gaps `chc.response` gives the point's
    channel, under the covariance over the share squared; the kernel as long as the longer of the
    panel and the series, nothing spent before either."""
    panel, _, spec = world(curve)
    c = [m.channel for m in spec.media].index(name)
    covariance = 9.0 * _ar1(READOUT, 0.6) + np.eye(READOUT)
    gaps = np.random.default_rng(4).normal(-5.0, 3.0, (2, READOUT))
    planned = SHARE * np.full(span, 100.0)
    runs = [planned.copy(), planned.copy()]
    runs[0][span - READOUT : span - READOUT + 4] = 0.0
    runs[1][span - READOUT : span - READOUT + 4] *= 2.5
    series = [LiftSeries(run, planned, gaps[j], covariance, SHARE) for j, run in enumerate(runs)]
    model = bayes_model(panel, NATIONAL, spec, series={name: series})
    length = max(WEEKS, span)
    assert model.lagged.shape[2] == length
    theta = np.random.default_rng(5).uniform(-1.5, 1.5, (4, model.dimension))
    terms = model.log_likelihood(theta)[:, -2:]
    draws = model.parameters(theta)
    pad = np.zeros(length - span)
    for k in range(theta.shape[0]):
        kernel = response.GeometricAdstock(draws["retention"][k, c], length=length, normalized=True)
        half = draws["half_saturation"][k, c]
        shape = (
            response.Tanh(half / _TANH_HALF)
            if isinstance(curve, TanhCurve)
            else response.Hill(half, draws["slope"][k, c])
        )
        channel = response.Channel(kernel, shape, draws["coefficient"][k, c])
        for j, s in enumerate(series):
            gap = channel(np.r_[pad, s.spend] / SHARE) - channel(np.r_[pad, planned] / SHARE)
            mean = np.asarray(gap)[-READOUT:]
            expected = stats.multivariate_normal(mean, covariance / SHARE**2).logpdf(s.gaps / SHARE)
            assert terms[k, j] == pytest.approx(expected, rel=1e-11, abs=0)


def test_independent_gaps_read_as_one_period_estimates() -> None:
    """A series whose covariance is diagonal reads, at every point and with its gradient, as the
    estimates of each one period long, each over the spend through its period, at its own error."""
    panel, _, spec = world(HillCurve())
    gaps = np.random.default_rng(6).normal(-4.0, 2.0, READOUT)
    sd = np.linspace(1.0, 3.0, READOUT)
    run = _run(0.0)
    one = [
        LiftEstimate(run[: end + 1], PLANNED[: end + 1], 1, float(gaps[t]), float(sd[t]), SHARE)
        for t, end in enumerate(range(SPAN - READOUT, SPAN))
    ]
    joint = bayes_model(
        panel,
        NATIONAL,
        spec,
        series={"tv": [LiftSeries(run, PLANNED, gaps, np.diag(sd**2), SHARE)]},
    )
    apart = bayes_model(panel, NATIONAL, spec, estimates={"tv": one})
    assert joint.coordinates == apart.coordinates
    theta = np.random.default_rng(7).uniform(-1.5, 1.5, (5, joint.dimension))
    np.testing.assert_allclose(
        joint.log_likelihood(theta)[:, -1],
        apart.log_likelihood(theta)[:, -READOUT:].sum(axis=1),
        rtol=1e-12,
    )
    for point in theta:
        (value, gradient), (other, other_gradient) = (
            joint.log_density(point),
            apart.log_density(point),
        )
        assert value == pytest.approx(other, rel=1e-12, abs=0)
        np.testing.assert_allclose(
            gradient, other_gradient, rtol=1e-11, atol=1e-11 * np.abs(gradient).max()
        )


def test_a_series_adds_its_own_term_and_nothing_else() -> None:
    """Television's series adds one term to the log density at every point and leaves the rest of
    it as it was: every other observation's term, and the gradient in search's coordinates."""
    panel, _, spec = world(HillCurve())
    series = LiftSeries(_run(0.0), PLANNED, np.full(READOUT, -3.0), 4.0 * _ar1(READOUT, 0.5), SHARE)
    tested = bayes_model(panel, NATIONAL, spec, series={"tv": [series]})
    untested = bayes_model(panel, NATIONAL, spec)
    assert tested.coordinates == untested.coordinates
    theta = np.random.default_rng(8).uniform(-2.0, 2.0, (5, tested.dimension))
    terms = tested.log_likelihood(theta)
    np.testing.assert_array_equal(terms[:, :-1], untested.log_likelihood(theta))
    search = [k for k, name in enumerate(tested.coordinates) if name.endswith("[search]")]
    for k in range(theta.shape[0]):
        (value, gradient), (base, base_gradient) = (
            tested.log_density(theta[k]),
            untested.log_density(theta[k]),
        )
        assert value - base == pytest.approx(
            terms[k, -1] + READOUT * math.log(tested.kpi_unit), rel=1e-12
        )
        np.testing.assert_allclose(gradient[search], base_gradient[search], rtol=1e-13)


def test_the_sum_reads_a_ridge_and_the_gaps_read_the_decay() -> None:
    """On a world with no noise in its test, at the world's curve, each read's best fit over the
    return at each decay from 0.1 to 0.9: the summed lift is fitted exactly at every decay, the
    return taking up what the decay moves, so its density is its top at each; the gaps are fitted
    exactly at the world's decay alone, and at every other their best falls short by more than 5
    in the log, a likelihood ratio past a chi-squared's 99.8th percentile on one degree."""
    panel, channel, spec = world(HillCurve())
    covariance = 4.0 * _ar1(READOUT, 0.5)
    series = LiftSeries(_run(0.0), PLANNED, _gaps(channel, _run(0.0)), covariance, SHARE)
    rates = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    best, top = {}, {}
    models = {
        "series": bayes_model(panel, NATIONAL, spec, series={"tv": [series]}),
        "sum": bayes_model(panel, NATIONAL, spec, estimates={"tv": [series.summed]}),
    }
    for name, model in models.items():
        theta = jnp.asarray(_point(model, **{"log_half_saturation[tv]": math.log(HALF)}))
        roi = model.coordinates.index("log_roi[tv]")
        decay = model.coordinates.index("logit_retention[tv]")

        def term(
            u: jax.Array,
            logit: float,
            model: BayesModel = model,
            theta: jax.Array = theta,
            roi: int = roi,
            decay: int = decay,
        ) -> jax.Array:
            p, _ = model._constrained(theta.at[roi].set(u[0]).at[decay].set(logit))
            return -model._pointwise(p, *model._terms(p))[-1]

        own = []
        for logit in np.log(rates / (1.0 - rates)):
            value = jax.jit(jax.value_and_grad(lambda u, x=float(logit): term(u, x)))
            fitted = optimize.minimize(
                lambda u, value=value: tuple(
                    np.asarray(x, dtype=float) for x in value(jnp.asarray(u))
                ),
                np.array([math.log(ROI)]),
                jac=True,
                method="BFGS",
                options={"gtol": 1e-12},
            )
            own.append(-fitted.fun)
        best[name] = np.array(own)
        scale = (SHARE * model.kpi_unit) ** 2
        size = covariance.sum() if name == "sum" else covariance
        top[name] = -0.5 * np.linalg.slogdet(np.atleast_2d(2.0 * math.pi * size / scale))[1]
    np.testing.assert_allclose(best["sum"], top["sum"], rtol=1e-10)
    assert best["series"][2] == pytest.approx(top["series"], rel=1e-10)
    assert np.delete(top["series"] - best["series"], 2).min() > 5.0


def test_the_gaps_fit_as_chc_s_fit_lift_fits_them() -> None:
    """The oracle. Three tests of television, a go-dark and two heavy-ups, each read period by
    period with independent noise of one size. On the gaps alone the model's most likely channel is
    `chc.lift.fit_lift`'s least-squares fit to the same tests to `1e-8`, and there the series'
    terms are the normal density of fit_lift's least cost at the noise's size, to `1e-12`."""
    panel, channel, spec = world(TanhCurve())
    rng = np.random.default_rng(9)
    sd = 2.0
    series, tests = [], []
    pad = np.zeros(WEEKS - 1 + READOUT - SPAN)
    for multiplier in (0.0, 2.0, 3.5):
        run = _run(multiplier)
        gaps = _gaps(channel, run) + rng.normal(0.0, sd, READOUT)
        series.append(LiftSeries(run, PLANNED, gaps, sd**2 * np.eye(READOUT), SHARE))
        tests.append(
            LiftTest(
                GeoArm(np.r_[pad, run], gaps, SHARE),
                GeoArm(np.r_[pad, PLANNED], np.zeros(READOUT), SHARE),
            )
        )
    template = response.Channel(
        response.GeometricAdstock(0.3, length=WEEKS, normalized=True), response.Tanh(100.0), 1.0
    )
    fit = fit_lift(tests, template, serial="independent")
    assert fit.parameters == ("kernel.retention", "curve.scale", "coefficient")
    model = bayes_model(panel, NATIONAL, spec, series={"tv": series})
    names = ("log_roi[tv]", "logit_retention[tv]", "log_half_saturation[tv]")
    where = jnp.array([model.coordinates.index(n) for n in names])
    theta = jnp.zeros(model.dimension)

    def gaps_only(u: jax.Array) -> jax.Array:
        p, _ = model._constrained(theta.at[where].set(u))
        return -jnp.sum(model._pointwise(p, *model._terms(p))[-3:])

    value = jax.jit(jax.value_and_grad(gaps_only))
    fitted = optimize.minimize(
        lambda u: tuple(np.asarray(x, dtype=float) for x in value(jnp.asarray(u))),
        np.zeros(3),
        jac=True,
        method="BFGS",
        options={"gtol": 1e-11},
    )
    point = np.asarray(theta.at[where].set(fitted.x))
    draws = model.parameters(point)
    ours = np.array(
        [
            draws["retention"][0, 0],
            draws["half_saturation"][0, 0] / _TANH_HALF,
            draws["coefficient"][0, 0],
        ]
    )
    np.testing.assert_allclose(ours, fit.estimate, rtol=1e-8)
    difference = np.concatenate([t.difference for t in tests])
    predicted = np.concatenate(
        [
            np.asarray(fit.channel(t.treated.spend / SHARE) - fit.channel(t.control.spend / SHARE))[
                -READOUT:
            ]
            for t in tests
        ]
    )
    cost = float(np.sum((difference - predicted) ** 2))
    scale = sd / SHARE
    expected = -0.5 * cost / scale**2 - 1.5 * READOUT * math.log(2.0 * math.pi * scale**2)
    assert model.log_likelihood(point[None])[0, -3:].sum() == pytest.approx(
        expected, rel=1e-12, abs=0
    )


def test_a_fit_reads_its_series() -> None:
    """A short fit holds each series' term as its last observation, at each draw the model's own."""
    panel, channel, spec = world(TanhCurve())
    series = LiftSeries(_run(0.0), PLANNED, _gaps(channel, _run(0.0)), 4.0 * np.eye(READOUT), SHARE)
    fit = fit_bayes(
        panel, NATIONAL, spec, seed=3, chains=2, warmup=40, draws=10, series={"tv": [series]}
    )
    model = bayes_model(panel, NATIONAL, spec, series={"tv": [series]})
    assert fit.log_likelihood.shape == (2, 10, WEEKS + 1)
    assert np.isfinite(fit.log_likelihood).all()
    plain = fit_bayes(panel, NATIONAL, spec, seed=3, chains=2, warmup=40, draws=10, series={})
    assert plain.log_likelihood.shape == (2, 10, WEEKS)
    assert model.log_likelihood(np.zeros((1, model.dimension))).shape == (1, WEEKS + 1)


@pytest.mark.parametrize(
    ("series", "match"),
    [
        ({"radio": "one"}, r"series name channels the model does not have, \['radio'\]"),
        ({"tv": "none"}, r"or none, \['tv'\]"),
        ({"tv": "elsewhere"}, r"series name geos \['north'\] that the model of 'national'"),
    ],
)
def test_series_the_model_cannot_read_are_refused(series: dict[str, Any], match: str) -> None:
    panel, _, spec = world(HillCurve())
    one = LiftSeries(_run(0.0), PLANNED, np.zeros(READOUT), np.eye(READOUT), SHARE)
    north = LiftSeries(
        _run(0.0), PLANNED, np.zeros(READOUT), np.eye(READOUT), SHARE, frozenset({Geo("north")})
    )
    given = {
        c: {"one": [one], "elsewhere": [north], "none": []}.get(v, v) for c, v in series.items()
    }
    with pytest.raises(ValueError, match=match):
        bayes_model(panel, NATIONAL, spec, series=given)


def test_a_group_s_gaps_are_independent_at_its_period_variance() -> None:
    noise = GroupNoise(30.0)
    covariance = noise.covariance(READOUT, SHARE)
    np.testing.assert_array_equal(covariance, np.diag(np.diag(covariance)))
    assert covariance[0, 0] == pytest.approx(noise.se(1, SHARE) ** 2, rel=1e-15, abs=0)
    assert covariance.sum() == pytest.approx(noise.se(READOUT, SHARE) ** 2, rel=1e-14, abs=0)
    with pytest.raises(ValueError, match=r"share=1\.0 is not in"):
        noise.covariance(READOUT, 1.0)


_RESIDUALS = np.random.default_rng(11).normal(0.0, 5.0, 104)


def test_a_placebo_covariance_sums_to_the_placebo_error() -> None:
    """The covariance is a Toeplitz matrix of the residuals' own uncentred autocorrelation, positive
    definite, whose sum is the placebo error of the sum squared."""
    noise = PlaceboNoise(_RESIDUALS)
    covariance = noise.covariance(READOUT, SHARE)
    e = _RESIDUALS
    rho = np.array([e[: e.size - k] @ e[k:] for k in range(READOUT)]) / (e @ e)
    np.testing.assert_allclose(covariance / covariance[0, 0], linalg.toeplitz(rho), rtol=1e-13)
    assert covariance.sum() == pytest.approx(noise.se(READOUT, SHARE) ** 2, rel=1e-13, abs=0)
    assert np.linalg.eigvalsh(covariance).min() > 0.0
    with pytest.raises(ValueError, match="no placebo window"):
        noise.covariance(104, SHARE)


@pytest.mark.parametrize("phi", [0.0, 0.5, 0.8])
def test_a_placebo_covariance_reads_a_long_history_s_dependence(phi: float) -> None:
    """On 200 000 periods of a first-order autoregression the covariance's correlations are the
    autoregression's, `phi^|i - j|`, to 0.01."""
    shocks = np.random.default_rng(12).normal(size=200_000)
    noise = signal.lfilter([1.0], [1.0, -phi], shocks)
    covariance = PlaceboNoise(noise, pre=200).covariance(READOUT, SHARE)
    scale = np.sqrt(np.diag(covariance))
    np.testing.assert_allclose(covariance / np.outer(scale, scale), _ar1(READOUT, phi), atol=0.01)
