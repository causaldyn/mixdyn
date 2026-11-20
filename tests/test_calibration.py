"""A lift estimate holds what a test's analysis reported, and the Bayesian model reads each one as
a term of its likelihood: every test kept, each at its own standard error, and nothing else of the
model moved. A fit reads its own data and estimates under its spec's prior, whatever was fitted
before it."""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mixdyn import Cadence, Channel, Geo, MediaPanel, Revenue
from mixdyn.calibration import LiftEstimate
from mixdyn.models.bayes_mmm import (
    BayesFit,
    BayesMedia,
    BayesModel,
    BayesSpec,
    bayes_model,
    fit_bayes,
)

SPEND = np.array([50.0, 50.0, 0.0, 0.0, 50.0, 50.0])
PLANNED = np.full(6, 50.0)
NATIONAL = Geo("national")
LIFTS, ERRORS = np.array([10.0, -8.0, 2.0]), np.array([1.0, 100.0, 1.0])
"""Three tests of television, each moving 10 of spend in one week and read over that week."""


def test_an_estimate_holds_read_only_copies_of_its_spend() -> None:
    spend = SPEND.copy()
    estimate = LiftEstimate(spend, PLANNED, 4, -30.0, 12.0)
    spend[0] = 1e6
    assert estimate.spend[0] == 50.0
    with pytest.raises(ValueError, match="read-only"):
        estimate.planned[0] = 0.0


def test_a_lift_at_or_below_zero_is_kept() -> None:
    assert LiftEstimate(SPEND, PLANNED, 4, 0.0, 12.0).lift == 0.0
    assert LiftEstimate(PLANNED, SPEND, 4, -5.0, 12.0).lift == -5.0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"spend": SPEND[None]}, r"non-empty vector, got shape \(1, 6\)"),
        ({"spend": []}, "non-empty vector"),
        ({"spend": np.r_[SPEND[:-1], np.nan]}, "finite and at or above 0"),
        ({"planned": np.r_[PLANNED[:-1], -1.0]}, "finite and at or above 0"),
        ({"planned": PLANNED[:-1]}, "spend runs 6 periods and planned 5"),
        ({"spend": PLANNED}, "changed no spend"),
        ({"readout": 0}, "readout of 0 periods"),
        ({"readout": 7}, "readout of 7 periods"),
        ({"lift": np.inf}, "lift must be finite"),
        ({"se": 0.0}, "positive and finite, got 0.0"),
        ({"se": np.nan}, "positive and finite"),
        ({"share": 0.0}, r"not a share of the market in \(0, 1\]"),
        ({"share": 1.5}, "not a share of the market"),
        ({"geos": frozenset()}, "at least one geo; geos=None reads the whole market"),
    ],
)
def test_an_estimate_that_cannot_be_is_refused(kwargs: dict[str, Any], match: str) -> None:
    call: dict[str, Any] = {
        "spend": SPEND,
        "planned": PLANNED,
        "readout": 4,
        "lift": -30.0,
        "se": 12.0,
    }
    call |= kwargs
    with pytest.raises(ValueError, match=match):
        LiftEstimate(**call)


def flat_model(lifts: list[tuple[float, float]] | None = None) -> BayesModel:
    """52 weeks of television at 10 and search at 30 every week and the KPI flat at 1000, with
    television's tests as estimates of `(lift, standard error)`."""
    weeks, spend = 52, {"tv": 10.0, "search": 30.0}
    panel = MediaPanel(
        dates=np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(weeks),
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=tuple(Channel(c) for c in spend),
        controls=(),
        kpi=np.full((1, weeks), 1000.0),
        kpi_unit=Revenue(),
        spend=np.tile(list(spend.values()), (1, weeks, 1)),
        currency="EUR",
        control_values=np.zeros((1, weeks, 0)),
        filled=np.zeros((1, weeks), dtype=bool),
    )
    estimates = None
    if lifts is not None:
        estimates = {
            "tv": [
                LiftEstimate(np.array([10.0]), np.array([0.0]), 1, lift, se) for lift, se in lifts
            ]
        }
    spec = BayesSpec(tuple(BayesMedia(c) for c in spend))
    return bayes_model(panel, NATIONAL, spec, estimates=estimates)


def test_every_lift_enters_at_its_precision() -> None:
    """Lifts 10, -8 and 2. Their likelihood is that of one test at their precision-weighted mean
    lift with their summed precision, at every point and up to a constant, so the evidence centres
    at a lift of 4/3 when the errors are equal and at 6.00 when they are 1, 100 and 1. Dropping the
    negative lift instead centres it at 6 under equal errors, 4.5 times as far."""
    points = np.random.default_rng(5).uniform(-2.0, 2.0, (6, flat_model().dimension))

    def offsets(a: BayesModel, b: BayesModel) -> tuple[np.ndarray, np.ndarray]:
        """`a`'s log density less `b`'s at each point, and its gradient's, each over the largest
        of `a`'s."""
        ours, theirs = ([m.log_density(p) for p in points] for m in (a, b))
        values, gradients = (np.array([o[k] for o in ours]) for k in (0, 1))
        return (
            (values - [v for v, _ in theirs]) / np.abs(values).max(),
            (gradients - [g for _, g in theirs]) / np.abs(gradients).max(),
        )

    centres = []
    for se in (np.ones(3), ERRORS):
        weight = se**-2.0
        centre, pooled = weight @ LIFTS / weight.sum(), weight.sum() ** -0.5
        three = flat_model(list(zip(LIFTS, se, strict=True)))
        values, gradients = offsets(three, flat_model([(centre, pooled)]))
        assert np.ptp(values) <= 1e-12
        assert np.abs(gradients).max() <= 1e-12
        centres.append(centre)
    assert centres[0] == pytest.approx(4.0 / 3.0)
    assert centres[1] == pytest.approx(6.0, abs=1e-3)
    positive = float(LIFTS[LIFTS > 0].mean())
    values, _ = offsets(
        flat_model(list(zip(LIFTS, np.ones(3), strict=True))),
        flat_model([(positive, 3.0**-0.5)]),
    )
    assert np.ptp(values) > 1e-3


def test_a_channel_s_lifts_add_their_own_terms() -> None:
    """Television's three lift estimates add one term each to the log density, at every point, and
    leave the rest of it, every prior and every week's term, as it was: the gradient in search's
    coordinates does not move."""
    tested = flat_model(list(zip(LIFTS, ERRORS, strict=True)))
    untested = flat_model()
    assert tested.coordinates == untested.coordinates
    theta = np.random.default_rng(6).uniform(-2.0, 2.0, (6, tested.dimension))
    terms = tested.log_likelihood(theta)
    np.testing.assert_array_equal(terms[:, :-3], untested.log_likelihood(theta))
    search = [k for k, name in enumerate(tested.coordinates) if name.endswith("[search]")]
    for k in range(theta.shape[0]):
        point, own = theta[k], terms[k, -3:]
        (value, gradient), (base, base_gradient) = (
            tested.log_density(point),
            untested.log_density(point),
        )
        assert value - base == pytest.approx(np.sum(own + math.log(tested.kpi_unit)), rel=1e-12)
        np.testing.assert_allclose(gradient[search], base_gradient[search], rtol=1e-13)


WEEKS, RETENTION, HALF, ROI, SHARE = 104, 0.5, 1.0, 2.0, 0.1
DARK, READOUT, SERIES = slice(10, 14), 12, slice(70, 92)


def _saturated(spend: np.ndarray, unit: float) -> np.ndarray:
    """`spend` over `unit`, carried over by the normalised kernel as long as the panel, zero
    before the series, and saturated."""
    padded = np.r_[np.zeros(WEEKS - spend.size), spend / unit]
    weights = RETENTION ** np.arange(WEEKS)
    adstock = np.convolve(padded, weights / weights.sum())[:WEEKS]
    return adstock / (adstock + HALF)


def adstock_world() -> tuple[MediaPanel, LiftEstimate]:
    """104 weeks of one channel, its spend AR(1) about 100 at 0.7, carried over by the normalised
    geometric kernel at 0.5 as long as the panel and saturated by a Hill of slope 1 at the mean
    positive spend, its return on the window's spend 2, and the KPI 1000 plus that plus normal
    noise of SD 50. A go-dark test runs in a tenth of the market: the group's planned spend over
    22 weeks, the last 12 the readout, dark over the readout's first 4; its lift is the world's,
    without noise, its error a quarter of it."""
    rng = np.random.default_rng(20261003)
    spend = np.empty(WEEKS)
    spend[0] = 100.0
    for t in range(1, WEEKS):
        spend[t] = 100.0 + 0.7 * (spend[t - 1] - 100.0) + rng.normal(0.0, 30.0)
    spend = np.maximum(spend, 0.0)
    unit = float(spend[spend > 0.0].mean())
    h = _saturated(spend, unit)
    beta = ROI * spend.sum() / h.sum()
    kpi = 1000.0 + beta * h + rng.normal(0.0, 50.0, WEEKS)
    planned = SHARE * spend[SERIES]
    run = planned.copy()
    run[DARK] = 0.0
    gap = _saturated(run / SHARE, unit) - _saturated(planned / SHARE, unit)
    lift = SHARE * beta * gap[-READOUT:].sum()
    panel = MediaPanel(
        dates=np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(WEEKS),
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=(Channel("tv"),),
        controls=(),
        kpi=kpi[None],
        kpi_unit=Revenue(),
        spend=spend[None, :, None],
        currency="EUR",
        control_values=np.zeros((1, WEEKS, 0)),
        filled=np.zeros((1, WEEKS), dtype=bool),
    )
    return panel, LiftEstimate(run, planned, READOUT, lift, abs(lift) / 4.0, SHARE)


def test_a_test_peaks_at_the_world_s_decay() -> None:
    """On a world of the model's family whose decay is 0.5, a go-dark test in a tenth of the market
    read as a lift estimate: at the world's return, half-saturation point and slope, its log
    density over the decays from 0 to 0.9 is largest at the world's, where it is a normal's at its
    mean."""
    panel, estimate = adstock_world()
    model = bayes_model(
        panel, NATIONAL, BayesSpec((BayesMedia("tv"),)), estimates={"tv": [estimate]}
    )
    rates = np.clip(np.linspace(0.0, 0.9, 10), 1e-3, 1.0 - 1e-3)
    theta = np.zeros((rates.size, model.dimension))
    for name, value in (
        ("log_roi[tv]", math.log(ROI)),
        ("logit_retention[tv]", np.log(rates / (1.0 - rates))),
        ("log_half_saturation[tv]", math.log(HALF)),
        ("log_slope[tv]", 0.0),
    ):
        theta[:, model.coordinates.index(name)] = value
    own = model.log_likelihood(theta)[:, -1]
    assert rates[np.argmax(own)] == pytest.approx(RETENTION)
    assert own.max() == pytest.approx(
        -math.log(estimate.se / SHARE) - 0.5 * math.log(2.0 * math.pi), rel=1e-12
    )


def test_a_fit_reads_its_own_data_and_estimates_whatever_was_fitted_before() -> None:
    """The prior is the spec's, never an earlier fit's posterior: the adstock world fitted with its
    estimate, then its first 80 weeks fitted without it, then the first fit again from the same
    seed, draws the first's posterior bit for bit, so no fit counts another's data again."""
    panel, estimate = adstock_world()
    spec = BayesSpec((BayesMedia("tv"),))

    def fit(**kwargs: Any) -> BayesFit:
        return fit_bayes(panel, NATIONAL, spec, seed=7, chains=2, warmup=50, draws=20, **kwargs)

    first = fit(estimates={"tv": [estimate]})
    fit(window=(panel.dates[0], panel.dates[79]))
    again = fit(estimates={"tv": [estimate]})
    assert first.draws.keys() == again.draws.keys()
    for name, value in first.draws.items():
        np.testing.assert_array_equal(again.draws[name], value)


@pytest.mark.skipif(shutil.which("maxima") is None, reason="maxima is not on PATH")
def test_maxima_proves_the_lift_term_s_shape_at_every_step() -> None:
    script = Path(__file__).parents[1] / "validation" / "lift_logconcave.mac"
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
    assert len(verdicts) == 17
    assert all(line.endswith("(PASS)") for line in verdicts)
    assert {line.split()[1] for line in verdicts} == {str(k) for k in range(1, 8)}


def test_a_weakly_identified_channel_s_lift_keeps_its_mass_off_0() -> None:
    """Television spends 10 every week of two years and search 30, the KPI 1000 plus normal noise
    of SD 50: the history cannot tell either channel's effect from the level. One test doubled
    television's spend for 8 weeks and read a lift of 30 over them at an error of 10. The posterior
    of the model's lift, television's coefficient times its gap in saturated spend summed over the
    readout, holds under 5 % of its mass below 3, a tenth of the tested lift: the term is normal,
    log-concave in that lift (validation/lift_logconcave.mac), and adds no mode near 0. Each draw's
    lift is the one the likelihood read at that draw, to rounding.

    Four chains of 250 draws: 0.4 % of them fell below 3 on this world (0.2 % of 2 000 at another
    run's length), so the 5 % bar is passed by ten times the Monte Carlo error. With the term left
    out, the prior's lift puts 4.7 % below 3, which is why the test's lift sits inside the prior's
    bulk: a test far above it, a lift of 150 at an error of 50, left 4.9 % below 15, a long tail
    toward the prior and no second mode."""
    weeks, spend = 104, {"tv": 10.0, "search": 30.0}
    rng = np.random.default_rng(0)
    panel = MediaPanel(
        dates=np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(weeks),
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=tuple(Channel(c) for c in spend),
        controls=(),
        kpi=(1000.0 + rng.normal(0.0, 50.0, weeks))[None],
        kpi_unit=Revenue(),
        spend=np.tile(list(spend.values()), (1, weeks, 1)),
        currency="EUR",
        control_values=np.zeros((1, weeks, 0)),
        filled=np.zeros((1, weeks), dtype=bool),
    )
    run, planned = np.full(8, 20.0), np.full(8, 10.0)
    lift, se = 30.0, 10.0
    fit = fit_bayes(
        panel,
        NATIONAL,
        BayesSpec(tuple(BayesMedia(c) for c in spend)),
        seed=11,
        estimates={"tv": [LiftEstimate(run, planned, 8, lift, se)]},
        chains=4,
        warmup=300,
        draws=250,
    )
    count = fit.draws["roi"].shape[0] * fit.draws["roi"].shape[1]

    def padded(x: np.ndarray) -> np.ndarray:
        return np.concatenate([np.zeros(fit.length - x.size), x])

    lifts = np.array(
        [
            float(np.sum((np.asarray(d["tv"](padded(run))) - d["tv"](padded(planned)))[-8:]))
            for d in fit.channel_draws(count)
        ]
    )
    term = fit.log_likelihood.reshape(count, -1)[:, -1]
    read = -math.log(se) - 0.5 * math.log(2.0 * math.pi) - (lift - lifts) ** 2 / (2.0 * se**2)
    np.testing.assert_allclose(term, read, rtol=0.0, atol=1e-9)
    assert np.mean(lifts < 0.1 * lift) < 0.05
