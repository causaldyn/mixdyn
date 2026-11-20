"""The bounded ridge core: glmnet's coefficients at glmnet's penalties, the Karush-Kuhn-Tucker
conditions on random problems with and without an anchor, and the invariances its standardisation
promises."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from mixdyn.models.ridge import Anchor, ridge_path

FIXTURE = Path(__file__).parent / "fixtures" / "oracles" / "ridge_glmnet.json"


def _spread(values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((values - values.mean(axis=0)) ** 2, axis=0))


@pytest.mark.parametrize("case", ["free", "bounded", "capped"])
def test_the_path_is_glmnet_s_at_glmnet_s_penalties(case: str) -> None:
    """glmnet's `lambda` is ours times y's standard deviation; its coordinate descent stops within
    `1e-9` of the solution in standardised units, which this solver reaches exactly."""
    fixture = json.loads(FIXTURE.read_text())
    x, y = np.array(fixture["x"]), np.array(fixture["y"])
    run = next(r for r in fixture["runs"] if r["name"] == case)
    lower = np.array([-np.inf if v is None else v for v in run["lower"]])
    upper = np.array([np.inf if v is None else v for v in run["upper"]])
    path = ridge_path(x, y, np.array(fixture["lambda"]) / _spread(y), lower, upper)
    scale = _spread(y)
    standardised = (path.coefficients - np.array(run["coefficients"])) * _spread(x)
    assert np.abs(standardised).max() < 1e-8 * scale
    assert np.abs(path.intercepts - np.array(run["intercept"])).max() < 1e-8 * scale
    assert path.bound.any() == (case != "free")


@st.composite
def problems(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    n = draw(st.integers(3, 40))
    p = draw(st.integers(1, 7))
    seed = draw(st.integers(0, 2**32 - 1))
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p)) * np.exp(rng.uniform(-3.0, 6.0, p)) + rng.uniform(-5, 5, p)
    y = x @ rng.standard_normal(p) * 3.0 + rng.standard_normal(n) * np.exp(rng.uniform(-2.0, 3.0))
    kinds = draw(
        st.lists(st.sampled_from(["free", "above", "below", "capped"]), min_size=p, max_size=p)
    )
    lower = np.array([0.0 if k in ("above", "capped") else -np.inf for k in kinds])
    upper = np.array([0.0 if k == "below" else np.inf for k in kinds])
    caps = np.abs(rng.standard_normal(p)) / _spread(x)
    upper = np.where(np.array(kinds) == "capped", caps, upper)
    penalty = float(np.exp(draw(st.floats(-9.0, 5.0))))
    return x, y, penalty, lower, upper


@settings(max_examples=300, deadline=None)
@given(problems())
# Every coefficient on a bound at 0, and BVLS a rounding error off one: no measure of its error.
@example(
    (
        np.array([[9.92083657, -4.39905678], [3.89758507, -3.98230866], [0.6657207, -3.90847821]]),
        np.array([6.83356432, -3.92973041, -12.44505458]),
        0.01831563888873418,
        np.array([-np.inf, 0.0]),
        np.array([0.0, 1.10455524]),
    )
)
def test_the_solution_meets_the_karush_kuhn_tucker_conditions(
    problem: tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray],
) -> None:
    """The problem is strictly convex, so the point that meets them is the solution."""
    x, y, penalty, lower, upper = problem
    path = ridge_path(x, y, [penalty], lower, upper)
    b = path.coefficients[0]
    assert (b >= lower).all()
    assert (b <= upper).all()
    at_floor, at_ceiling = b == lower, b == upper
    np.testing.assert_array_equal(path.bound[0], at_floor | at_ceiling)
    scale = _spread(x)
    z = (x - x.mean(axis=0)) / scale
    yc = y - y.mean()
    c = b * scale
    gradient = z.T @ (z @ c - yc) / len(y) + penalty * c
    tolerance = 1e-9 * (np.abs(z.T @ yc).max() / len(y) + penalty * np.abs(c).max() + 1e-300)
    inside = ~(at_floor | at_ceiling)
    assert (np.abs(gradient[inside]) <= tolerance).all()
    assert (gradient[at_floor] >= -tolerance).all()
    assert (gradient[at_ceiling] <= tolerance).all()
    np.testing.assert_allclose(path.intercepts[0], y.mean() - x.mean(axis=0) @ b, rtol=1e-12)


def _problem(seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.gamma(3.0, 1.0, (50, 4)) * np.array([1e3, 1.0, 20.0, 0.1])
    y = x @ np.array([0.002, 5.0, -0.3, 40.0]) + rng.standard_normal(50)
    return x, y


def test_a_column_s_currency_moves_its_coefficient_and_nothing_else() -> None:
    x, y = _problem()
    lower = np.array([0.0, 0.0, 0.0, -np.inf])
    penalties = [1e-3, 0.1, 3.0]
    base = ridge_path(x, y, penalties, lower)
    rescaled = ridge_path(x * np.array([1e-3, 7.0, 1.0, 1.0]), y, penalties, lower)
    np.testing.assert_allclose(
        rescaled.coefficients * np.array([1e-3, 7.0, 1.0, 1.0]), base.coefficients, rtol=1e-10
    )
    np.testing.assert_allclose(rescaled.intercepts, base.intercepts, rtol=1e-10)


def test_a_level_added_to_a_column_or_to_the_outcome_moves_the_intercept_alone() -> None:
    """Sales in a currency's units sit far from 0 and vary little about their level; the level
    must not cost the coefficients the digits the data keep. Both fits read the same rounded
    values, since `y + 1e9` holds `y` only to `1e9`'s spacing."""
    x, y = _problem()
    lower = np.array([0.0, 0.0, 0.0, -np.inf])
    penalties = [1e-3, 0.1, 3.0]
    shift = np.array([5.0, 0.0, -2.0, 0.0])
    moved, level = x + shift, y + 1e9
    base = ridge_path(moved - shift, level - 1e9, penalties, lower)
    shifted = ridge_path(moved, level, penalties, lower)
    np.testing.assert_allclose(shifted.coefficients, base.coefficients, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(
        shifted.intercepts,
        base.intercepts - base.coefficients @ shift + 1e9,
        rtol=0.0,
        atol=8 * np.spacing(1e9),
    )


def test_scaling_the_outcome_scales_the_solution_at_the_same_penalty() -> None:
    x, y = _problem()
    upper = np.array([np.inf, 4.0, np.inf, np.inf])
    base = ridge_path(x, y, [0.01, 1.0], upper=upper)
    scaled = ridge_path(x, 250.0 * y, [0.01, 1.0], upper=250.0 * upper)
    np.testing.assert_allclose(scaled.coefficients, 250.0 * base.coefficients, rtol=1e-10)
    np.testing.assert_allclose(scaled.intercepts, 250.0 * base.intercepts, rtol=1e-10)
    np.testing.assert_array_equal(scaled.bound, base.bound)


def test_without_a_binding_bound_the_path_is_the_augmented_least_squares() -> None:
    x, y = _problem()
    path = ridge_path(x, y, [1e-4, 0.5])
    scale = _spread(x)
    z = (x - x.mean(axis=0)) / scale
    for lam, coefficients in zip(path.penalties, path.coefficients, strict=True):
        system = np.vstack([z / np.sqrt(len(y)), np.sqrt(lam) * np.eye(4)])
        target = np.concatenate([(y - y.mean()) / np.sqrt(len(y)), np.zeros(4)])
        solved = np.linalg.lstsq(system, target, rcond=None)[0]
        np.testing.assert_allclose(coefficients * scale, solved, rtol=1e-11)
    assert not path.bound.any()


def test_a_column_that_does_not_vary_gets_nothing() -> None:
    x, y = _problem()
    with_constant = np.column_stack([x[:, :2], np.full(50, 3.0), x[:, 2:]])
    path = ridge_path(with_constant, y, [0.05], np.array([0.0, 0.0, 0.0, 0.0, -np.inf]))
    without = ridge_path(x, y, [0.05], np.array([0.0, 0.0, 0.0, -np.inf]))
    assert path.coefficients[0, 2] == 0.0
    np.testing.assert_allclose(np.delete(path.coefficients[0], 2), without.coefficients[0])
    np.testing.assert_allclose(path.intercepts, without.intercepts)


@st.composite
def anchored(
    draw: st.DrawFn,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, Anchor]:
    x, y, penalty, lower, upper = draw(problems())
    p = x.shape[1]
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    pulled = np.array(draw(st.lists(st.booleans(), min_size=p, max_size=p)))
    # a pull from a fiftieth to fifty times the data's own curvature in each coefficient, s_j^2
    precision = np.where(pulled, _spread(x) ** 2 * np.exp(rng.uniform(-4.0, 4.0, p)), 0.0)
    center = rng.standard_normal(p) * 3.0 * _spread(y) / _spread(x)
    return x, y, penalty, lower, upper, Anchor(center, precision)


@settings(max_examples=300, deadline=None)
@given(anchored())
def test_an_anchored_solution_meets_the_karush_kuhn_tucker_conditions(
    problem: tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, Anchor],
) -> None:
    """The anchor adds `r_j (b_j - m_j)` to each coefficient's gradient, on its column's scale."""
    x, y, penalty, lower, upper, anchor = problem
    path = ridge_path(x, y, [penalty], lower, upper, anchor)
    b = path.coefficients[0]
    assert (b >= lower).all()
    assert (b <= upper).all()
    at_floor, at_ceiling = b == lower, b == upper
    np.testing.assert_array_equal(path.bound[0], at_floor | at_ceiling)
    xc, yc, n = x - x.mean(axis=0), y - y.mean(), len(y)
    data = xc.T @ (xc @ b - yc) / n
    ridge = penalty * _spread(x) ** 2 * b
    pull = anchor.precision * (b - anchor.center)
    gradient = data + ridge + pull
    # each coefficient's gradient against the size of the terms that make it, in its own units
    size = np.abs(xc.T @ yc) / n + np.abs(ridge) + np.abs(anchor.precision * anchor.center)
    tolerance = 1e-8 * (size + np.abs(pull) + 1e-300)
    inside = ~(at_floor | at_ceiling)
    assert (np.abs(gradient[inside]) <= tolerance[inside]).all()
    assert (gradient[at_floor] >= -tolerance[at_floor]).all()
    assert (gradient[at_ceiling] <= tolerance[at_ceiling]).all()
    np.testing.assert_allclose(path.intercepts[0], y.mean() - x.mean(axis=0) @ b, rtol=1e-12)


def test_an_anchor_of_no_precision_is_no_anchor() -> None:
    x, y = _problem()
    lower = np.array([0.0, 0.0, 0.0, -np.inf])
    base = ridge_path(x, y, [1e-3, 0.3], lower)
    none = ridge_path(x, y, [1e-3, 0.3], lower, anchor=Anchor(np.full(4, 7.0), np.zeros(4)))
    np.testing.assert_array_equal(none.coefficients, base.coefficients)
    np.testing.assert_array_equal(none.intercepts, base.intercepts)


def test_as_its_precision_grows_an_anchor_holds_its_coefficient_at_its_center() -> None:
    x, y = _problem()
    lower = np.array([0.0, 0.0, 0.0, -np.inf])
    upper = np.array([np.inf, np.inf, np.inf, 30.0])
    center = np.array([0.004, 2.0, 0.5, 45.0])  # the last past its cap, the third past its floor
    precision = 1e10 * _spread(x) ** 2
    path = ridge_path(x, y, [0.01], lower, upper, Anchor(center, precision))
    np.testing.assert_allclose(path.coefficients[0], [0.004, 2.0, 0.5, 30.0], rtol=1e-7)
    np.testing.assert_array_equal(path.bound[0], [False, False, False, True])


def test_a_column_that_does_not_vary_takes_its_anchor_held_to_its_bounds() -> None:
    x, y = _problem()
    with_constant = np.column_stack([x[:, :2], np.full(50, 3.0), x[:, 2:]])
    lower = np.array([0.0, 0.0, 0.0, 0.0, -np.inf])
    upper = np.array([np.inf, np.inf, 0.5, np.inf, np.inf])
    without = ridge_path(x, y, [0.05], np.delete(lower, 2), np.delete(upper, 2))
    for center, held in [(0.2, 0.2), (0.9, 0.5), (-1.0, 0.0)]:
        anchor = Anchor(np.array([0.0, 0.0, center, 0.0, 0.0]), np.array([0.0, 0.0, 1.0, 0.0, 0.0]))
        path = ridge_path(with_constant, y, [0.05], lower, upper, anchor)
        assert path.coefficients[0, 2] == held
        assert path.bound[0, 2] == (held != center)
        np.testing.assert_allclose(np.delete(path.coefficients[0], 2), without.coefficients[0])
        np.testing.assert_allclose(path.intercepts, without.intercepts - 3.0 * held)


@pytest.mark.parametrize(
    ("center", "precision", "match"),
    [
        ([0.0, np.nan, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], "centers must be finite"),
        ([0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 1.0, 1.0], "precisions finite and at least 0"),
        ([0.0, 0.0, 0.0, 0.0], [1.0, np.inf, 1.0, 1.0], "precisions finite and at least 0"),
        ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], "an anchor holds 4 centers and 4 precisions"),
    ],
)
def test_an_anchor_it_cannot_read_is_refused(
    center: list[float], precision: list[float], match: str
) -> None:
    x, y = _problem()
    with pytest.raises(ValueError, match=match):
        ridge_path(x, y, [1.0], anchor=Anchor(np.array(center), np.array(precision)))


@pytest.mark.parametrize(
    ("rows", "outcome", "penalty", "lower", "upper", "match"),
    [
        (50, 49, 1.0, None, None, r"x must be \(n, p\) and y \(n,\)"),
        (1, 1, 1.0, None, None, r"with n >= 2"),
        (50, 50, 0.0, None, None, "penalties must be positive"),
        (50, 50, np.inf, None, None, "penalties must be positive"),
        (50, 50, 1.0, [0.5, 0.0, 0.0, 0.0], None, "lower must be 4 values at most 0"),
        (50, 50, 1.0, None, [-1.0, 0.0, 0.0, 0.0], "upper 4 at least 0"),
        (50, 50, 1.0, [0.0, 0.0, 0.0], None, "lower must be 4 values"),
    ],
)
def test_what_it_cannot_solve_is_refused(
    rows: int,
    outcome: int,
    penalty: float,
    lower: list[float] | None,
    upper: list[float] | None,
    match: str,
) -> None:
    x, y = _problem()
    with pytest.raises(ValueError, match=match):
        ridge_path(x[:rows], y[:outcome], [penalty], lower, upper)


def test_a_value_that_is_not_finite_is_refused() -> None:
    x, y = _problem()
    with pytest.raises(ValueError, match="must be finite"):
        ridge_path(x, np.r_[y[:-1], np.nan], [1.0])
