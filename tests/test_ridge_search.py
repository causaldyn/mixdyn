"""The ridge hyperparameter search: the truth found in a world without noise and beaten on its own
folds in a world with noise, the score as stated, calibrated or not, the space mapped onto its
intervals, and the currencies and the seed moving nothing."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from chc.lift import GeoArm, LiftTest
from scipy.stats import qmc

from mixdyn.models.ridge import ridge_path
from mixdyn.models.ridge_mmm import (
    Control,
    FourierSeason,
    LinearTrend,
    Media,
    RidgeSpec,
    fit_ridge,
)
from mixdyn.models.ridge_search import (
    Folds,
    GeometricSpace,
    Interval,
    MediaSpace,
    Optimum,
    RidgeSpace,
    WeibullSpace,
    search_ridge,
)
from mixdyn.transforms import Geometric, Hill, WeibullCdf, carry_over, saturate
from ridge_worlds import NATIONAL, panel_of, world, world_and_tests

TRUTH = (0.9, 2.0, 0.5, 1.5, 3.0, 1.0, 0.3, 0.2, 3.0, 0.8)
"""`world`'s hyperparameters: c0's theta, alpha, gamma; c1's shape, scale, alpha, gamma; c2's."""


def _kpi(
    noise: float, seed: int = 7, beta: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spend, media, truth = world()
    beta = truth if beta is None else beta
    columns = [
        saturate(a := carry_over(spend[:, j], m.adstock), m.hill, m.hill.gamma * a.max())
        for j, m in enumerate(media)
    ]
    trend = np.arange(120.0)
    noise_draws = np.random.default_rng(seed).normal(0.0, noise, 120)
    return spend, 1000.0 + np.column_stack(columns) @ beta + 2.0 * trend + noise_draws, trend


def _space(penalty: Interval) -> RidgeSpace:
    return RidgeSpace(
        media=(
            MediaSpace(
                "c0", GeometricSpace(Interval(0.5, 0.95)), Interval(1.0, 3.0), Interval(0.3, 1.0)
            ),
            MediaSpace(
                "c1",
                WeibullSpace(Interval(1.0, 2.0), Interval(1.0, 6.0)),
                Interval(0.5, 2.0),
                Interval(0.2, 0.6),
            ),
            MediaSpace(
                "c2", GeometricSpace(Interval(0.0, 0.5)), Interval(1.0, 4.0), Interval(0.5, 1.0)
            ),
        ),
        controls=(Control("trend"),),
        penalty=penalty,
    )


def _fixed(spec: RidgeSpec) -> RidgeSpace:
    """`spec` as a space whose every interval is fixed."""
    media = []
    for m in spec.media:
        match m.adstock:
            case Geometric(theta):
                adstock = GeometricSpace(Interval(theta, theta))
            case WeibullCdf(shape, scale):
                adstock = WeibullSpace(Interval(shape, shape), Interval(scale, scale))
        alpha, gamma = m.hill.alpha, m.hill.gamma
        media.append(MediaSpace(m.channel, adstock, Interval(alpha, alpha), Interval(gamma, gamma)))
    return RidgeSpace(tuple(media), spec.controls, Interval(spec.penalty, spec.penalty))


def _hyperparameters(spec: RidgeSpec) -> np.ndarray:
    values = []
    for m in spec.media:
        match m.adstock:
            case Geometric(theta):
                values.append(theta)
            case WeibullCdf(shape, scale):
                values += [shape, scale]
        values += [m.hill.alpha, m.hill.gamma]
    return np.array(values)


def test_the_truth_is_found_in_a_world_without_noise() -> None:
    spend, kpi, trend = _kpi(noise=0.0)
    search = search_ridge(
        panel_of(kpi, spend, {"trend": trend}),
        NATIONAL,
        _space(Interval(1e-10, 1e-6)),
        Folds(4, 8),
        seed=3,
        starts=32,
        polish=1,
    )
    np.testing.assert_allclose(_hyperparameters(search.fit.spec), TRUTH, rtol=1e-5)
    np.testing.assert_allclose(
        [search.fit.coefficients[c] for c in ("c0", "c1", "c2")], world()[2], rtol=1e-5
    )
    assert search.score < 1e-7


def test_with_noise_the_choice_predicts_its_folds_better_than_the_truth() -> None:
    """The truth is in the space, at every penalty of the interval; the choice beats it."""
    spend, kpi, trend = _kpi(noise=10.0)
    panel = panel_of(kpi, spend, {"trend": trend})
    space = _space(Interval(1e-4, 1e2))
    search = search_ridge(panel, NATIONAL, space, Folds(4, 8), seed=3, starts=32, polish=2)
    _, media, _ = world()
    for penalty in (1e-4, 1e-2, 1.0):
        truth = _fixed(RidgeSpec(tuple(media), (Control("trend"),), penalty))
        assert search.score < search_ridge(panel, NATIONAL, truth, Folds(4, 8), seed=0).score
    assert [o.score for o in search.optima] == sorted(o.score for o in search.optima)
    assert search.optima[0] == Optimum(search.fit.spec, search.score)


def test_the_score_is_the_pooled_validation_nrmse_and_the_fit_the_whole_window_s() -> None:
    """Each block is predicted by the model fitted to the window's periods before it, with the
    design read off the whole window and the coefficients within their bounds; a missing week is
    neither fitted nor predicted."""
    spend, kpi, trend = _kpi(noise=10.0, beta=np.array([400.0, 250.0, -120.0]))
    kpi[[40, 105]] = np.nan
    panel = panel_of(kpi, spend, {"trend": trend})
    _, media, _ = world()
    spec = RidgeSpec(tuple(media), (Control("trend"),), 0.05)
    window = (panel.dates[10], panel.dates[-1])
    search = search_ridge(panel, NATIONAL, _fixed(spec), Folds(3, 10), seed=0, window=window)
    fit = fit_ridge(panel, NATIONAL, spec, window)
    periods = np.arange(fit.y.size)
    residuals, bound = [], []
    for start in (80, 90, 100):
        train = fit.fitted & (periods < start)
        held = fit.fitted & (periods >= start) & (periods < start + 10)
        path = ridge_path(fit.design[train], fit.y[train], [0.05], [0.0, 0.0, 0.0, -np.inf], None)
        residuals.append(fit.y[held] - path.intercepts[0] - fit.design[held] @ path.coefficients[0])
        bound.append(path.bound[0, 2])
    assert all(bound)
    pooled = np.concatenate(residuals)
    assert pooled.size == 29
    observed = fit.y[fit.fitted]
    nrmse = np.sqrt(np.mean(pooled**2)) / (observed.max() - observed.min())
    assert search.score == pytest.approx(nrmse, rel=1e-12, abs=0.0)
    assert search.evaluations == 1
    assert search.optima == (Optimum(spec, search.score),)
    assert search.fit.coefficients == fit.coefficients
    assert search.fit.intercept == fit.intercept
    weighted = search_ridge(
        panel, NATIONAL, _fixed(spec), Folds(3, 10), seed=0, window=window, rssd_weight=0.3
    )
    assert weighted.score == pytest.approx(nrmse + 0.3 * fit.decomp_rssd(), rel=1e-12, abs=0.0)


def test_a_calibrated_score_joins_the_folds_errors_and_the_tests_each_over_its_noise() -> None:
    """Each block is predicted by `fit_ridge`'s calibrated fit with the KPI from the block on
    blanked out, so the tests' anchor is weighed by the KPI's noise over the training periods
    alone. The tests' errors under the whole window's calibrated fit join the folds', a test
    period's at the search without tests' validation mean square error over the tests' mean square
    gap before their groups' spend parts."""
    panel, media, _, tests, reads = world_and_tests(test_noise=10.0)
    spec = RidgeSpec(tuple(media), (), 0.05)
    search = search_ridge(panel, NATIONAL, _fixed(spec), Folds(3, 10), seed=0, tests=tests)
    plain = search_ridge(panel, NATIONAL, _fixed(spec), Folds(3, 10), seed=0)
    whole = fit_ridge(panel, NATIONAL, spec, tests=tests)
    periods = np.arange(whole.y.size)
    residuals = []
    for start in (90, 100, 110):
        blanked = panel.kpi.copy()
        blanked[0, start:] = np.nan
        fold = fit_ridge(dataclasses.replace(panel, kpi=blanked), NATIONAL, spec, tests=tests)
        held = whole.fitted & (periods >= start) & (periods < start + 10)
        residuals.append(whole.y[held] - fold.estimated[held])
    pooled = np.concatenate(residuals)
    observed = whole.y[whole.fitted]
    span = observed.max() - observed.min()
    # a test's gap in saturated spend is 0 exactly until its groups' spend parts, whatever the
    # transforms; a zero week in the plan keeps it 0 past the change
    quiet = np.concatenate(
        [gap[: np.flatnonzero(saturated)[0]] for read in reads.values() for saturated, gap in read]
    )
    ratio = (plain.score * span) ** 2 / np.mean(quiet**2)
    misses = sum(
        np.sum((gap - whole.coefficients[channel] * saturated) ** 2)
        for channel, read in reads.items()
        for saturated, gap in read
    )
    score = np.sqrt((pooled @ pooled + ratio * misses) / pooled.size) / span
    assert search.score == pytest.approx(score, rel=1e-12, abs=0.0)
    assert search.evaluations == 2
    assert search.fit.coefficients == whole.coefficients
    assert search.fit.calibration is not None
    assert whole.calibration is not None
    assert search.fit.calibration.weight == whole.calibration.weight
    weighted = search_ridge(
        panel, NATIONAL, _fixed(spec), Folds(3, 10), seed=0, rssd_weight=0.3, tests=tests
    )
    assert weighted.score == pytest.approx(score + 0.3 * whole.decomp_rssd(), rel=1e-12, abs=0.0)


def test_a_search_refuses_tests_of_a_channel_its_space_does_not_have() -> None:
    panel, media, _, tests, _ = world_and_tests(test_noise=10.0)
    space = _fixed(RidgeSpec(tuple(media), (), 0.05))
    with pytest.raises(ValueError, match=r"does not have, \['c9'\]"):
        search_ridge(panel, NATIONAL, space, Folds(3, 10), seed=0, tests={"c9": tests["c0"]})


def test_a_search_refuses_tests_read_only_after_their_groups_spend_parted() -> None:
    """The score reads the tests' noise where no transform puts a gap; without such a period it has
    none to read."""
    panel, media, _, tests, reads = world_and_tests(test_noise=10.0)
    late = {
        channel: [
            LiftTest(
                *(
                    GeoArm(arm.spend, arm.outcome[np.flatnonzero(saturated)[0] :], arm.share)
                    for arm in (test.treated, test.control)
                )
            )
            for test, (saturated, _) in zip(channel_tests, reads[channel], strict=True)
        ]
        for channel, channel_tests in tests.items()
    }
    space = _fixed(RidgeSpec(tuple(media), (), 0.05))
    with pytest.raises(ValueError, match="0 readout periods before their groups' spend parts"):
        search_ridge(panel, NATIONAL, space, Folds(3, 10), seed=0, tests=late)


def test_a_search_refuses_tests_whose_groups_read_alike_until_their_spend_parts() -> None:
    """Their gaps there are 0, which would weigh every test period's error without bound."""
    panel, media, _, tests, reads = world_and_tests(test_noise=10.0)
    alike: dict[str, list[LiftTest]] = {}
    for channel, channel_tests in tests.items():
        for test, (saturated, _) in zip(channel_tests, reads[channel], strict=True):
            if test.treated.share != test.control.share:
                continue
            parts = np.flatnonzero(saturated)[0]
            outcome = np.concatenate([test.treated.outcome[:parts], test.control.outcome[parts:]])
            control = GeoArm(test.control.spend, outcome, test.control.share)
            alike.setdefault(channel, []).append(LiftTest(test.treated, control))
    space = _fixed(RidgeSpec(tuple(media), (), 0.05))
    with pytest.raises(ValueError, match="before their groups' spend parts hold no gap"):
        search_ridge(panel, NATIONAL, space, Folds(3, 10), seed=0, tests=alike)


def test_a_currency_and_the_kpi_s_scale_move_no_score() -> None:
    """The penalty reads standardised columns, the Hill a share of the window's most, and the
    NRMSE the KPI's range: none of them sees a unit, so no point of the space scores otherwise.
    The polish's path does, by rounding, and in a flat valley ends elsewhere along it."""
    spend, kpi, trend = _kpi(noise=10.0)
    base = panel_of(kpi, spend, {"trend": trend})
    moved = panel_of(3.0 * kpi, spend * np.array([1e-3, 1.0, 40.0]), {"trend": 100.0 * trend})
    space = _space(Interval(1e-4, 1e2))
    for point in np.random.default_rng(6).random((5, space.dimension)):
        fixed = _fixed(space.spec(point))
        scores = [
            search_ridge(p, NATIONAL, fixed, Folds(4, 8), seed=0).score for p in (base, moved)
        ]
        assert scores[1] == pytest.approx(scores[0], rel=1e-10)


def test_the_seed_draws_the_starts_and_the_best_are_polished() -> None:
    """The seed scrambles the Sobol sample and the polish starts from its best points, so one seed
    repeats bit for bit, another starts elsewhere, and the choice scores no worse than any start."""
    spend, kpi, trend = _kpi(noise=10.0)
    panel = panel_of(kpi, spend, {"trend": trend})
    space = _space(Interval(1e-4, 1e2))
    first, again, other = (
        search_ridge(panel, NATIONAL, space, Folds(2, 8), seed=seed, starts=8, polish=1)
        for seed in (9, 9, 10)
    )
    assert first.optima == again.optima
    assert first.evaluations == again.evaluations
    assert first.optima != other.optima
    sample = qmc.Sobol(space.dimension, scramble=True, seed=9).random(8)
    scores = [
        search_ridge(panel, NATIONAL, _fixed(space.spec(p)), Folds(2, 8), seed=0).score
        for p in sample
    ]
    assert first.score <= min(scores)


def test_a_point_of_the_space_maps_onto_its_intervals() -> None:
    """A fixed interval takes no coordinate; the penalty's runs on its logarithm."""
    space = RidgeSpace(
        media=(
            MediaSpace(
                "c0", GeometricSpace(Interval(0.125, 0.625)), Interval(0.5, 3.0), Interval(0.4, 0.4)
            ),
            MediaSpace(
                "c1",
                WeibullSpace(Interval(1.0, 2.0), Interval(2.5, 2.5)),
                Interval(1.0, 1.0),
                Interval(0.25, 0.75),
            ),
        ),
        controls=(Control("price"),),
        penalty=Interval(1e-4, 1e2),
        trend=LinearTrend(2),
        season=FourierSeason(52.18, 3),
    )
    assert space.dimension == 5
    low, high = space.spec(np.zeros(5)), space.spec(np.ones(5))
    assert low.media == (
        Media("c0", Geometric(0.125), Hill(0.5, 0.4)),
        Media("c1", WeibullCdf(1.0, 2.5), Hill(1.0, 0.25)),
    )
    assert high.media == (
        Media("c0", Geometric(0.625), Hill(3.0, 0.4)),
        Media("c1", WeibullCdf(2.0, 2.5), Hill(1.0, 0.75)),
    )
    assert (low.penalty, high.penalty) == (pytest.approx(1e-4), pytest.approx(1e2))
    assert space.spec(np.full(5, 0.5)).penalty == pytest.approx(0.1)
    assert space.spec([0.5, 0.0, 0.0, 0.0, 0.0]).media[0].adstock == Geometric(0.375)
    assert (low.controls, low.trend, low.season) == (
        (Control("price"),),
        LinearTrend(2),
        FourierSeason(52.18, 3),
    )


def test_folds_hold_out_the_blocks_that_close_the_window() -> None:
    fitted = np.ones(20, dtype=bool)
    fitted[[3, 12]] = False
    (train0, held0), (train1, held1) = Folds(2, 5).masks(fitted)
    assert list(np.flatnonzero(train0)) == [0, 1, 2, 4, 5, 6, 7, 8, 9]
    assert list(np.flatnonzero(held0)) == [10, 11, 13, 14]
    assert list(np.flatnonzero(train1)) == [0, 1, 2, *range(4, 12), 13, 14]
    assert list(np.flatnonzero(held1)) == [15, 16, 17, 18, 19]


@pytest.mark.parametrize(
    ("periods", "count"), [(156, 4), (104, 4), (78, 4), (77, 3), (52, 2), (51, 1), (39, 1)]
)
def test_a_short_window_drops_its_earliest_folds_and_keeps_the_least_to_fit(
    periods: int, count: int
) -> None:
    """At most 4 blocks of 13, as many as leave the first fold 26 periods: 52 periods hold 2."""
    folds = Folds.within(periods, most=4, horizon=13, least=26)
    assert folds == Folds(count, 13)
    (train, _), *_ = folds.masks(np.ones(periods, dtype=bool))
    assert train.sum() == periods - 13 * count >= 26


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: Interval(2.0, 1.0), "low <= high"),
        (lambda: Interval(np.nan, 1.0), "finite ends"),
        (lambda: GeometricSpace(Interval(0.0, 1.0)), r"within \[0, 1\)"),
        (lambda: GeometricSpace(Interval(-0.1, 0.5)), r"within \[0, 1\)"),
        (lambda: WeibullSpace(Interval(0.0, 1.0), Interval(1.0, 2.0)), "positive values"),
        (lambda: WeibullSpace(Interval(1.0, 1.0), Interval(0.0, 2.0)), "positive values"),
        (
            lambda: MediaSpace(
                "c0", GeometricSpace(Interval(0, 0)), Interval(0, 1), Interval(1, 1)
            ),
            "alpha > 0",
        ),
        (
            lambda: MediaSpace(
                "c0", GeometricSpace(Interval(0, 0)), Interval(1, 1), Interval(0, 1)
            ),
            r"gamma in \(0, 1\]",
        ),
        (
            lambda: MediaSpace(
                "c0", GeometricSpace(Interval(0, 0)), Interval(1, 1), Interval(0.5, 1.5)
            ),
            r"gamma in \(0, 1\]",
        ),
        (
            lambda: RidgeSpace(_space(Interval(1, 1)).media, (), Interval(0.0, 1.0)),
            "penalty must be searched over positive values",
        ),
        (
            lambda: RidgeSpace((*_space(Interval(1, 1)).media, _space(Interval(1, 1)).media[0])),
            "more than once",
        ),
        (lambda: _space(Interval(1e-4, 1.0)).spec(np.zeros(3)), "is 11 coordinates"),
        (lambda: _space(Interval(1e-4, 1.0)).spec(np.full(11, 1.5)), r"in \[0, 1\]"),
        (lambda: _space(Interval(1e-4, 1.0)).spec(np.full(11, np.nan)), r"in \[0, 1\]"),
        (lambda: Folds(0, 4), "at least 1"),
        (lambda: Folds(2, 0), "at least 1"),
        (lambda: Folds(3, 6).masks(np.ones(19, dtype=bool)), "fits 1 observed periods"),
        (lambda: Folds(1, 4).masks(np.r_[np.ones(6), np.zeros(4)].astype(bool)), "predicts 0"),
        (lambda: Folds.within(38, most=4, horizon=13, least=26), "holds no block of 13"),
        (lambda: Folds.within(52, most=0, horizon=13, least=26), "at least 1"),
        (lambda: Folds.within(52, most=4, horizon=0, least=26), "at least 1"),
        (lambda: Folds.within(52, most=4, horizon=13, least=0), "at least 1"),
    ],
)
def test_what_cannot_be_searched_is_refused(make, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        make()


@pytest.mark.parametrize(
    ("change", "folds", "rssd_weight", "starts", "polish", "match"),
    [
        ("dark", Folds(2, 8), 0.0, 8, 1, r"\['c2'\] spend nothing in the window"),
        ("constant", Folds(2, 8), 0.0, 8, 1, "constant over the window"),
        (None, Folds(2, 8), -1.0, 8, 1, "nonnegative and finite"),
        (None, Folds(2, 8), np.inf, 8, 1, "nonnegative and finite"),
        (None, Folds(2, 8), 0.0, 8, 0, "1 <= polish <= starts"),
        (None, Folds(2, 8), 0.0, 4, 5, "1 <= polish <= starts"),
        (None, Folds(15, 8), 0.0, 8, 1, "leave none of the window's 100 periods"),
        (None, Folds(11, 9), 0.0, 8, 1, "fold 1 of 11, periods 2 to 10 of the window's 100"),
    ],
)
def test_a_search_it_cannot_run_is_refused(
    change: str | None, folds: Folds, rssd_weight: float, starts: int, polish: int, match: str
) -> None:
    spend, kpi, trend = _kpi(noise=10.0)
    if change == "dark":
        spend[20:, 2] = 0.0
    if change == "constant":
        kpi = np.full(120, 5.0)
    panel = panel_of(kpi, spend, {"trend": trend})
    with pytest.raises(ValueError, match=match):
        search_ridge(
            panel,
            NATIONAL,
            _space(Interval(1e-4, 1.0)),
            folds,
            seed=0,
            window=(panel.dates[20], panel.dates[-1]),
            rssd_weight=rssd_weight,
            starts=starts,
            polish=polish,
        )
