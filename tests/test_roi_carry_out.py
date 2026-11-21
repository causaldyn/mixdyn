"""A return prior that counts the carry-out: under `RoiPrior(counts="carried")` the one-geo model's
`roi` is `mixdyn.decisions.returns`' at every draw, its density is the formula's written again in
numpy, the level still means the intercept and the channels' mean contribution, and at no carryover
it is the window's; the default model is the one recorded before the option; the models over geos
and the nested one refuse it."""

from __future__ import annotations

import dataclasses
import hashlib
import math
from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest
from scipy import special, stats

import dispatch
from dispatch import Dispatch
from mixdyn import Cadence, Channel, MediaPanel, Revenue
from mixdyn.decisions import returns
from mixdyn.models.bayes_geo import BayesGeoSpec
from mixdyn.models.bayes_mmm import (
    BayesMedia,
    BayesModel,
    BayesSpec,
    HillCurve,
    RoiPrior,
    Saturation,
    TanhCurve,
    bayes_model,
    fit_bayes,
)
from mixdyn.models.nested import NestedSpec
from mixdyn.models.ridge_mmm import Control, FourierSeason, LinearTrend
from test_bayes_mmm import CHANNELS, NATIONAL, STAN, stan_world

CARRIED = RoiPrior(counts="carried")


def _digest(*arrays: npt.ArrayLike) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


ZEN3 = Dispatch(
    numpy="exp=X86_V3 log=X86_V3 tanh=X86_V3 expm1=baseline(X86_V2) log1p=baseline(X86_V2)",
    openblas="numpy=Haswell scipy=Haswell",
    xla="znver3 avx avx2 f16c fma sse sse2 sse3 sse4.1 sse4.2 sse4a ssse3",
    libc="glibc 2.43",
)
RECORDED = {
    ZEN3: {
        "density lognormal": "60771b0fb8c6ad50",
        "density normal": "5bb51c95ae380da9",
        "density student": "1a928b6ccd2aff13",
        "density tanh": "af9f6191c4dd4f9d",
        "fit normal": "632eb313674de615",
        "fit tanh": "36c34a2beb7a71c3",
    },
    dataclasses.replace(ZEN3, libc="glibc 2.39"): {
        "density lognormal": "60771b0fb8c6ad50",
        "density normal": "5bb51c95ae380da9",
        "density student": "1a928b6ccd2aff13",
        "density tanh": "3772bc24e339e794",
        "fit normal": "632eb313674de615",
        "fit tanh": "a7693601c7f123dc",
    },
}
"""Each case's digest under each machine's dispatch (`dispatch.key`): the default model's density
and gradient at Stan's 25 points of each world, and a short seeded fit's draws. A new key's
digests go in after checking that the tree before the change under review draws the same ones
there, so that a record holds the machine's bits and not a change's.

The first key is an AVX2 Zen 3 machine's with glibc 2.43, natively and in Ubuntu 26.04; the second
is the same machine's in Ubuntu 24.04, whose glibc 2.39 returns `atanh(1/2)`, the tanh curve's
constant, one ulp below its correctly rounded value, so only tanh moves. That row's tanh
digests are the ones GitHub's ubuntu-24.04 runners drew on 2026-11-01, at an EPYC 7763's speed. On
the Zen 3, XLA held to AVX or to SSE4.2 moves all six digests and OpenBLAS held to its Sandy Bridge
kernels both fits'; numpy on its baseline loops, OpenBLAS held to Zen's and two CPUs move none."""

BEFORE_KEYS = {
    "a cloud sandbox with AVX-512 in Ubuntu 24.04 on 2026-10-28": {
        "density lognormal": "39fa47895fe7446e",
        "density normal": "0cf8950637769a9e",
        "density student": "967797f3df106e39",
        "density tanh": "1f63e0f572fb054e",
        "fit normal": "35837240149fdcfc",
        "fit tanh": "ccbcd9530d92475d",
    },
    "GitHub's ubuntu-24.04 runners of three legs on 2026-11-07": {
        "fit normal": "263b0c220c35b762",
        "fit tanh": "def58ee14e153863",
    },
    "an AMD EPYC 9V45 on GitHub's ubuntu-26.04 on 2026-11-21": {"fit tanh": "dbddb83d18fdc8c5"},
    "an Intel Xeon Platinum 8573C on GitHub's ubuntu-26.04 on 2026-11-21": {
        "density tanh": "5477b9d7d6cbfb36",
        "fit tanh": "af7c22b607e4cf7e",
    },
}
"""Digests drawn before digests were keyed, on machines with AVX-512 whose dispatch no log names
(the runners of 2026-11-01, an AVX2 class, drew the second key's). Each drew, in the cases not
listed, one of the digests recorded then, which one is unknown. A key not in `RECORDED` fails
naming the machines here that drew the same digest; it goes under the key the failure prints."""


def _held(case: str, drawn: str) -> None:
    """Holds `drawn` to the digest `RECORDED` holds for `case` under this machine's dispatch."""
    key = dispatch.key()
    recorded = RECORDED.get(key, {})
    if case not in recorded:
        before = [where for where, held in BEFORE_KEYS.items() if held.get(case) == drawn]
        drew = f", the digest {' and '.join(before)} drew" if before else ""
        pytest.fail(
            f"no {case!r} digest is recorded under this machine's dispatch: it drew {drawn!r}"
            f"{drew}; record it as RECORDED[{key!r}][{case!r}]"
        )
    assert drawn == recorded[case], f"{case!r} under {key!r}"


@pytest.mark.parametrize("name", sorted(STAN["cases"]))
def test_the_default_density_and_gradient_are_the_ones_recorded_before(name: str) -> None:
    """At Stan's 25 points of each world, bit for bit."""
    panel, tests, estimates, spec = stan_world(name)
    model = bayes_model(panel, NATIONAL, spec, tests=tests, estimates=estimates)
    out = [model.log_density(p) for p in np.array(STAN["cases"][name]["points"])]
    _held(f"density {name}", _digest([v for v, _ in out], *[g for _, g in out]))


@pytest.mark.parametrize("name", ["normal", "tanh"])
def test_a_default_fit_draws_what_it_drew_before(name: str) -> None:
    """A short seeded fit over the last 80 weeks: every draw and each observation's log
    likelihood, bit for bit."""
    panel, tests, estimates, spec = stan_world(name)
    fit = fit_bayes(
        panel,
        NATIONAL,
        spec,
        seed=5,
        window=(panel.dates[24], panel.dates[-1]),
        tests=tests,
        estimates=estimates,
        chains=2,
        warmup=60,
        draws=20,
    )
    _held(f"fit {name}", _digest(*(fit.draws[k] for k in sorted(fit.draws)), fit.log_likelihood))


def test_a_digest_is_held_to_its_own_dispatch_s_record(monkeypatch: pytest.MonkeyPatch) -> None:
    """A recorded key holds its digest and refuses another key's; a key with no record fails
    with the key and the digest, naming the machine that drew it before keys."""
    monkeypatch.setattr(dispatch, "key", lambda: ZEN3)
    _held("fit tanh", "36c34a2beb7a71c3")
    with pytest.raises(AssertionError, match="'fit tanh' under Dispatch"):
        _held("fit tanh", "a7693601c7f123dc")
    forged = dataclasses.replace(ZEN3, openblas="numpy=SkylakeX scipy=SkylakeX")
    monkeypatch.setattr(dispatch, "key", lambda: forged)
    with pytest.raises(pytest.fail.Exception) as failed:
        _held("fit tanh", "dbddb83d18fdc8c5")
    assert str(failed.value) == (
        "no 'fit tanh' digest is recorded under this machine's dispatch: it drew "
        "'dbddb83d18fdc8c5', the digest an AMD EPYC 9V45 on GitHub's ubuntu-26.04 on 2026-11-21 "
        f"drew; record it as RECORDED[{forged!r}]['fit tanh']"
    )


def _carried(spec: BayesSpec, *channels: str) -> BayesSpec:
    return dataclasses.replace(
        spec,
        media=tuple(
            dataclasses.replace(m, roi=CARRIED) if m.channel in channels else m for m in spec.media
        ),
    )


def test_each_draw_s_return_is_the_return_mixdyn_reports() -> None:
    """Over the panel's first 80 of 104 weeks, every channel's return carried: at each of two
    chains' 100 draws, the draw's `roi` is `returns` of the channel `channel_draws` hands on, to
    `1e-12`, and the window's contribution falls short of it by the carry-out."""
    panel, _, _, spec = stan_world("normal")
    window = slice(0, 80)
    fit = fit_bayes(
        panel,
        NATIONAL,
        _carried(spec, *CHANNELS),
        seed=3,
        window=(panel.dates[0], panel.dates[79]),
        chains=2,
        warmup=150,
        draws=100,
    )
    roi = fit.draws["roi"].reshape(-1, len(CHANNELS))
    for i, channels in enumerate(fit.channel_draws(roi.shape[0])):
        for j, (name, channel) in enumerate(channels.items()):
            spend = panel.spend[0, :, j]
            read = returns(channel, spend, window)
            assert read.roi == pytest.approx(roi[i, j], rel=1e-12, abs=0.0), name
            assert read.contribution < roi[i, j] * read.spend, name


def _numpy_log_density(
    panel: MediaPanel, spec: BayesSpec, model: BayesModel, window: slice, theta: np.ndarray
) -> float:
    """The model's log density at `theta` written again from its formula: one free control, the
    calendar's columns read from `model`, normal noise, each channel's coefficient by the window's
    identity or the carried one."""
    m = len(spec.media)
    hill = isinstance(spec.saturation, HillCurve)
    blocks = [m, m, m, m if hill else 0, 1, model.calendar.shape[1], 1, 1]
    log_roi, u_ret, u_half, u_slope, (level,), delta, (u_control,), (u_sigma,) = np.split(
        theta, np.cumsum(blocks)[:-1]
    )
    retention, half, sigma = special.expit(u_ret), np.exp(u_half), math.exp(u_sigma)
    slope = np.exp(u_slope) if hill else np.ones(m)
    length = panel.dates.size
    kpi = panel.kpi[0, window]
    kpi_unit = kpi.mean()

    def curve(adstock: np.ndarray, j: int) -> np.ndarray:
        if hill:
            return adstock ** slope[j] / (adstock ** slope[j] + half[j] ** slope[j])
        return np.tanh(math.atanh(0.5) * adstock / half[j])

    def adstocked(series: np.ndarray, j: int) -> np.ndarray:
        weights = retention[j] ** np.arange(length)
        return np.convolve(series, weights / weights.sum())[: series.size]

    saturated, beta = np.empty((kpi.size, m)), np.empty(m)
    for j, media in enumerate(spec.media):
        spend = panel.spend[0, :, j]
        x = spend / spend[spend > 0.0].mean()
        saturated[:, j] = curve(adstocked(x, j), j)[window]
        if media.roi.counts == "carried":
            run_on = np.r_[x, np.zeros(length)]
            removed = run_on.copy()
            removed[window] = 0.0
            earning = np.sum(curve(adstocked(run_on, j), j) - curve(adstocked(removed, j), j))
        else:
            earning = saturated[:, j].sum()
        beta[j] = math.exp(log_roi[j]) * spend[window].sum() / kpi_unit / earning
    intercept = level - np.sum(saturated @ beta) / kpi.size
    control = panel.control_values[0, window, 1]
    control = (control - control.mean()) / control.std()
    mean = intercept + model.calendar @ delta + control * u_control + saturated @ beta
    lp = np.sum(special.log_expit(u_ret) + special.log_expit(-u_ret))
    lp += u_half.sum() + u_slope.sum() + u_sigma
    priors = [r.roi for r in spec.media]
    assert all(isinstance(r, RoiPrior) for r in priors)
    lp += stats.norm.logpdf(log_roi, [r.mu for r in priors if isinstance(r, RoiPrior)], 0.9).sum()
    lp += stats.beta.logpdf(retention, 1.0, 3.0).sum()
    lp += stats.lognorm.logpdf(half, 1.0).sum() + stats.lognorm.logpdf(slope, 0.5).sum() * hill
    lp += stats.norm.logpdf(intercept, 1.0, 1.0) + stats.norm.logpdf(delta).sum()
    lp += stats.norm.logpdf(u_control) + stats.norm.logpdf(sigma)
    return float(lp + stats.norm.logpdf(kpi / kpi_unit, mean, sigma).sum())


@pytest.mark.parametrize("saturation", [HillCurve(), TanhCurve()], ids=["hill", "tanh"])
def test_the_carried_density_is_its_formula(saturation: Saturation) -> None:
    """At ten points uniform in `[-2, 2]`, two channels carried and one not, over weeks 24 to 79
    of 104, so that spend before the window carries in and spend after it stays: the density is
    the formula's in numpy, to `1e-12`."""
    panel, _, _, _ = stan_world("normal")
    spec = BayesSpec(
        (BayesMedia("tv", CARRIED), BayesMedia("search"), BayesMedia("social", CARRIED)),
        (Control("price"),),
        LinearTrend(1),
        FourierSeason(52.18, 2),
        saturation=saturation,
    )
    window = slice(24, 80)
    model = bayes_model(panel, NATIONAL, spec, window=(panel.dates[24], panel.dates[79]))
    for point in np.random.default_rng(11).uniform(-2.0, 2.0, (10, model.dimension)):
        value, gradient = model.log_density(point)
        assert np.all(np.isfinite(gradient))
        expected = _numpy_log_density(panel, spec, model, window, point)
        assert value == pytest.approx(expected, rel=1e-12, abs=0.0)


def test_the_level_is_the_intercept_and_the_channels_mean_contribution_when_carried() -> None:
    """At ten points, the level coordinate is the intercept plus the mean over the window of the
    model's own channel terms, and `_with_level` maps a point holding the intercept back to it,
    one point or many at once."""
    panel, tests, estimates, spec = stan_world("normal")
    model = bayes_model(
        panel,
        NATIONAL,
        _carried(spec, "tv", "social"),
        window=(panel.dates[24], panel.dates[79]),
        tests=tests,
        estimates=estimates,
    )
    level = model.coordinates.index("level")
    points = np.random.default_rng(12).uniform(-2.0, 2.0, (10, model.dimension))
    held = points.copy()
    for k, point in enumerate(points):
        p, _ = model._constrained(jnp.asarray(point))
        saturated, beta = model._terms(p)
        media = float(jnp.mean(saturated @ beta))
        assert float(p["intercept"][0]) + media == pytest.approx(
            points[k, level], rel=1e-12, abs=1e-12
        )
        held[k, level] = float(p["intercept"][0])
    np.testing.assert_allclose(model._with_level(jnp.asarray(held)), points, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        model._with_level(jnp.asarray(held[3])), points[3], rtol=0.0, atol=1e-12
    )


def test_with_no_carryover_a_carried_return_is_the_window_s() -> None:
    """Over weeks 24 to 79 of 104, spend after the window: at a retention of 0 the coefficients
    are the same, and at a logit decay of -40, `4e-18`, so is the density to `1e-12`."""
    panel, _, _, spec = stan_world("tanh")
    window = (panel.dates[24], panel.dates[79])
    plain = bayes_model(panel, NATIONAL, spec, window=window)
    carried = bayes_model(panel, NATIONAL, _carried(spec, *CHANNELS), window=window)
    m = len(CHANNELS)
    p = {
        "log_roi": jnp.array([0.3, -0.2, 0.1]),
        "retention": jnp.zeros(m),
        "half": jnp.array([0.5, 1.0, 2.0]),
        "slope": jnp.ones(m),
    }
    np.testing.assert_allclose(carried._terms(p)[1], plain._terms(p)[1], rtol=1e-14, atol=0.0)
    for point in np.random.default_rng(13).uniform(-2.0, 2.0, (5, plain.dimension)):
        point[m : 2 * m] = -40.0
        assert carried.log_density(point)[0] == pytest.approx(
            plain.log_density(point)[0], rel=1e-12, abs=0.0
        )


def _constant_panel(weeks: int) -> MediaPanel:
    return MediaPanel(
        dates=np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(weeks),
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=(Channel("tv"),),
        controls=(),
        kpi=np.full((1, weeks), 1000.0),
        kpi_unit=Revenue(),
        spend=np.full((1, weeks, 1), 100.0),
        currency="EUR",
        control_values=np.empty((1, weeks, 0)),
        filled=np.zeros((1, weeks), dtype=bool),
    )


def _carry_out(retention: Fraction, weeks: int) -> Fraction:
    """The share of a constant spend's return over `weeks` that lands after them, through a
    geometric kernel normalised over `weeks` lags and a linear curve, in exact arithmetic."""
    weights = [retention**lag for lag in range(weeks)]
    total = Fraction(sum(weights))
    inside = Fraction(sum(sum(weights[: weeks - t]) for t in range(weeks)))
    return 1 - inside / (weeks * total)


def test_a_constant_spend_s_carry_out_is_the_share_the_window_drops() -> None:
    """100 a week for 52 weeks at a retention of 0.8, on a curve linear to `1e-13` at the spend:
    the window's coefficient over the carried one is one less the carry-out, 7.7 %, the exact
    rational's `0.0769139...` to `1e-9`."""
    panel = _constant_panel(52)
    p = {
        "log_roi": jnp.zeros(1),
        "retention": jnp.array([0.8]),
        "half": jnp.array([1e6]),
        "slope": jnp.ones(1),
    }
    window, carried = (
        bayes_model(panel, NATIONAL, BayesSpec((BayesMedia("tv", roi),), saturation=TanhCurve()))
        for roi in (RoiPrior(), CARRIED)
    )
    share = 1.0 - float(carried._terms(p)[1][0] / window._terms(p)[1][0])
    exact = _carry_out(Fraction(4, 5), 52)
    assert share == pytest.approx(float(exact), rel=1e-9, abs=0.0)
    assert round(share, 3) == 0.077


def test_the_carried_identity_holds_in_exact_arithmetic() -> None:
    """On a linear curve and a normalised geometric kernel of length `L`, a window `[0, T)` run on
    for `L - 1` with nothing spent earns its coefficient times its spend, for any spend, and the
    carry-out of a constant spend has its closed form, `1 - (T - r (1 - r^T) / (1 - r)) /
    (T (1 - r^T))` at `T = L`: the checks Maxima ran symbolically, at rational points."""
    for retention in (Fraction(1, 3), Fraction(4, 5)):
        for weeks, length in ((3, 3), (4, 6), (5, 5)):
            spend = [Fraction(k + 1, 7) for k in range(weeks)]
            raw = [retention**lag for lag in range(length)]
            weights = [w / sum(raw) for w in raw]
            earned = sum(
                weights[lag] * spend[t - lag]
                for t in range(weeks + length - 1)
                for lag in range(length)
                if 0 <= t - lag < weeks
            )
            assert earned == sum(spend)
        closed = 1 - (52 - retention * (1 - retention**52) / (1 - retention)) / (
            52 * (1 - retention**52)
        )
        assert _carry_out(retention, 52) == closed


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (
            lambda: BayesGeoSpec((BayesMedia("tv", CARRIED), BayesMedia("search"))),
            r"model over geos states each return on the window: \['tv'\] carry it",
        ),
        (
            lambda: NestedSpec(
                "searches",
                BayesSpec((BayesMedia("tv"),)),
                BayesSpec((BayesMedia("search", CARRIED),), (Control("searches", lower=0.0),)),
            ),
            r"nested model states each return on the window: \['search'\] carry it",
        ),
        (lambda: RoiPrior(counts="spend"), 'counts "window" or "carried", got \'spend\''),  # ty: ignore[invalid-argument-type]
    ],
    ids=["geo", "nested", "count"],
)
def test_what_counts_no_carried_return_refuses_one(make, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        make()
