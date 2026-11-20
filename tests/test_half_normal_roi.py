"""A return prior that may be 0: under `HalfNormalRoi(scale)` a channel's return is half-normal,
its log density on the sampling scale the half-normal's at the return plus the log return, the
Jacobian of `exp`. Its default scale puts its median at the log-normal default's; the health
card's prior draws read it; the model over geos refuses it; the nested model reads its default at
the mediator's scale, as it reads `RoiPrior()`'s; and a file holds it."""

from __future__ import annotations

import dataclasses
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from mixdyn.inference.diagnostics import prior_predictive
from mixdyn.models.bayes_geo import BayesGeoSpec
from mixdyn.models.bayes_mmm import BayesMedia, HalfNormalRoi, RoiPrior, bayes_model
from mixdyn.models.nested import nested_model
from test_bayes_mmm import NATIONAL, stan_world
from test_diagnostics_health import world
from test_nested import DEFAULT, SPEC, mediation_world


@pytest.mark.parametrize("name", ["normal", "tanh"])
def test_the_half_normal_density_is_its_formula(name: str) -> None:
    """Television's and social's returns half-normal, of scales 0.5 and 3, search's log-normal as
    stated: at ten points, the log density and its gradient less those of the model with every
    return at its stated log-normal prior are each half-normal channel's `log HalfNormal(e^u; s) +
    u - log N(u; mu, sigma)`, by SciPy, at its log return `u`, and its derivative, to `1e-12`."""
    panel, tests, estimates, spec = stan_world(name)
    scales = {"tv": 0.5, "social": 3.0}
    half = dataclasses.replace(
        spec,
        media=tuple(
            BayesMedia(m.channel, HalfNormalRoi(scales[m.channel])) if m.channel in scales else m
            for m in spec.media
        ),
    )
    stated = bayes_model(panel, NATIONAL, spec, tests=tests, estimates=estimates)
    model = bayes_model(panel, NATIONAL, half, tests=tests, estimates=estimates)
    rng = np.random.default_rng(20261107)
    for theta in rng.uniform(-2.0, 2.0, (10, model.dimension)):
        value, gradient = model.log_density(theta)
        base, slope = stated.log_density(theta)
        change, moved = 0.0, np.zeros(model.dimension)
        for j, m in enumerate(spec.media):
            if m.channel not in scales:
                continue
            assert isinstance(m.roi, RoiPrior)
            u, s = theta[j], scales[m.channel]
            change += stats.halfnorm.logpdf(math.exp(u), scale=s) + u
            change -= stats.norm.logpdf(u, m.roi.mu, m.roi.sigma)
            moved[j] = 1.0 - math.exp(2.0 * u) / s**2 + (u - m.roi.mu) / m.roi.sigma**2
        assert value - base == pytest.approx(change, rel=0.0, abs=1e-12 * abs(value))
        np.testing.assert_allclose(
            gradient - slope, moved, rtol=0.0, atol=1e-12 * np.abs(slope).max()
        )


def test_the_default_scale_puts_the_median_at_the_log_normal_default_s() -> None:
    assert stats.halfnorm.median(scale=HalfNormalRoi().scale) == pytest.approx(
        math.exp(RoiPrior().mu), rel=1e-12, abs=0.0
    )


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: HalfNormalRoi(0.0), "positive, finite scale, got 0.0"),
        (lambda: HalfNormalRoi(-1.0), "positive, finite scale, got -1.0"),
        (lambda: HalfNormalRoi(math.inf), "positive, finite scale, got inf"),
        (lambda: HalfNormalRoi(math.nan), "positive, finite scale, got nan"),
        (lambda: HalfNormalRoi(counts="all"), 'counts "window" or "carried", got \'all\''),  # ty: ignore[invalid-argument-type]
        (
            lambda: BayesGeoSpec((BayesMedia("tv", HalfNormalRoi()), BayesMedia("search"))),
            r"log-normal priors: \['tv'\] are not",
        ),
    ],
)
def test_a_prior_that_cannot_be_is_refused(make, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        make()


def test_the_health_card_s_prior_draws_read_a_half_normal_return() -> None:
    """The prior's draws of a carried return are the stated prior's: television's log-normal as
    stated and search's half-normal of scale 2, the absolute value of a normal draw from the same
    stream, to `1e-12`; half of search's lie below its median, `2 Phi^-1(3/4)`."""
    panel, spec = world(1)
    carried = dataclasses.replace(
        spec,
        media=(
            BayesMedia("tv", RoiPrior(counts="carried")),
            BayesMedia("search", HalfNormalRoi(2.0, counts="carried")),
        ),
    )
    drawn = np.random.default_rng(0).normal([0.2, 0.0], [0.9, 2.0], (2000, 2))
    prior = prior_predictive(panel, NATIONAL, carried, draws=2000, seed=0)
    np.testing.assert_allclose(prior.roi["tv"], np.exp(drawn[:, 0]), rtol=1e-12)
    np.testing.assert_allclose(prior.roi["search"], np.abs(drawn[:, 1]), rtol=1e-12)
    assert np.all(prior.roi["search"] >= 0.0)


@pytest.mark.parametrize("stated", [False, True], ids=["default", "stated"])
def test_stage_one_reads_the_half_normal_default_at_the_mediator_s_scale(stated: bool) -> None:
    """Television's return on the searches half-normal: at its default scale, the scale times
    the searches' mean over sales'; at a scale stated, that scale, in searches a euro."""
    panel = mediation_world()
    prior = HalfNormalRoi(0.7) if stated else HalfNormalRoi()
    spec = dataclasses.replace(
        DEFAULT, stage_one=dataclasses.replace(SPEC.stage_one, media=(BayesMedia("tv", prior),))
    )
    searches = panel.control_values[0, :, panel.controls.index("searches")]
    ratio = 1.0 if stated else searches.mean() / panel.kpi[0].mean()
    (media,) = nested_model(panel, NATIONAL, spec).stage_one.spec.media
    assert isinstance(media.roi, HalfNormalRoi)
    assert media.roi.scale == pytest.approx(prior.scale * ratio, rel=1e-12, abs=0.0)


@pytest.mark.skipif(shutil.which("maxima") is None, reason="maxima is not on PATH")
def test_maxima_proves_the_prior_s_density_scale_and_unit_at_every_step() -> None:
    script = Path(__file__).parents[1] / "validation" / "half_normal_roi.mac"
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
    assert len(verdicts) == 13
    assert all(line.endswith("(PASS)") for line in verdicts)
    assert {line.split()[1] for line in verdicts} == {str(k) for k in range(1, 6)}
