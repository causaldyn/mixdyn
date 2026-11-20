"""The null-panel audit of the experiment readers (`validation/aa_audit.py`, ADR 0058), reduced.

The full table reads 200 null panels a cell; these tests read 3 of its cells over their first 60
panels, the band widened to 3 Monte Carlo standard errors of 0.05 at 60 draws, and pin what the
harness draws: the noise laws, the seeds, and each cell's count of rejections, so that a change to
the harness or to a reader that moves the table moves these first.
"""

from __future__ import annotations

import math
from typing import get_args

import numpy as np
import pytest

from mixdyn.experiments import switchback
from validation import aa_audit

REDUCED = 60
WIDE = 3.0 * math.sqrt(0.05 * 0.95 / REDUCED)

# Each cell's rejections over its first 60 panels, as the harness read them when this was set.
CELLS = {
    ("did", "ar1", 10): 1,
    ("sdid", "trend", 10): 0,
    ("switchback:randomization", "shift", 30): 4,
}


@pytest.mark.parametrize(("reader", "noise", "geos"), CELLS)
def test_a_reduced_cell_rejects_within_three_standard_errors_of_its_level(
    reader: str, noise: aa_audit.Noise, geos: int
) -> None:
    cell = aa_audit.audit(reader, noise, geos, REDUCED)
    assert cell.read + cell.refused == REDUCED
    assert cell.refused == 0
    assert abs(cell.fpr - 0.05) <= WIDE
    assert round(cell.fpr * cell.read) == CELLS[reader, noise, geos]
    # every reader here keys its interval on its p-value, so the coverage is the rate's complement
    assert cell.coverage == pytest.approx(1.0 - cell.fpr, rel=0.0, abs=1e-12)
    assert cell.se == pytest.approx(math.sqrt(0.05 * 0.95 / REDUCED), rel=1e-12, abs=0.0)


def test_the_band_is_the_level_plus_or_minus_two_standard_errors_at_200_draws_rounded() -> None:
    """At 200 draws a 5 % rate's Monte Carlo error is 0.0154; 0.05 + 2 SE = 0.0808 rounds to the
    band's top, 0.08; its floor, 0.03, was set before the run, tighter than the 0.0192 two errors
    give."""
    se = math.sqrt(0.05 * 0.95 / aa_audit.DRAWS)
    assert round(se, 4) == 0.0154
    assert round(0.05 + 2.0 * se, 2) == aa_audit.BAND[1]
    assert aa_audit.BAND[0] == 0.03 > 0.05 - 2.0 * se


def test_the_noise_laws_add_a_trend_or_a_shift_to_the_same_autoregression() -> None:
    """At one seed the three laws share the autoregression, the levels and the seasonality: the
    trend adds `0.05 mu_g b_g t / 52`, linear in the week in each geo, and the shift `0.05 mu_g x
    2` from one week on, the same week in every geo."""
    ar1 = aa_audit.null_panel("ar1", 10, np.random.default_rng(1))
    trend = aa_audit.null_panel("trend", 10, np.random.default_rng(1)) - ar1
    shift = aa_audit.null_panel("shift", 10, np.random.default_rng(1)) - ar1
    assert ar1.shape == (10, aa_audit.WEEKS)
    assert (ar1 > 0.0).all()
    weeks = np.arange(aa_audit.WEEKS)
    for g in range(10):
        slope = trend[g, 1] - trend[g, 0]
        np.testing.assert_allclose(trend[g], slope * weeks, rtol=0.0, atol=1e-9)
    level = (ar1 / (1.0 + 0.1 * np.sin(2.0 * np.pi * weeks / 52.0))).mean(axis=1)
    step = shift.max(axis=1)
    np.testing.assert_allclose(step / level, 0.1, rtol=0.05)
    starts = {int(np.flatnonzero(shift[g] > 0.0)[0]) for g in range(10)}
    assert len(starts) == 1
    assert 1 <= starts.pop() < aa_audit.WEEKS
    with pytest.raises(ValueError, match="noise must be one of"):
        aa_audit.null_panel("walk", 10, np.random.default_rng(1))  # ty: ignore[invalid-argument-type]


def test_a_cell_s_first_draws_are_the_full_table_s_first_draws() -> None:
    """Each draw is a child of its cell's child of `SeedSequence(ENTROPY)`: a reduced run reads the
    full run's first panels and designs, and two cells read different panels."""
    few = aa_audit._draws("trend", 10, 3)
    more = aa_audit._draws("trend", 10, 5)
    for (a, seed_a), (b, seed_b) in zip(few, more[:3], strict=True):
        assert seed_a == seed_b
        np.testing.assert_array_equal(a, b)
    other = aa_audit._draws("trend", 30, 1)[0][0]
    assert other.shape == (30, aa_audit.WEEKS)
    assert not np.array_equal(other[:10], few[0][0])


def test_every_switchback_method_is_audited_and_a_refused_draw_is_counted_apart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The switchback is read at every method `read_switchback` offers; a draw a reader refuses
    is counted, and the rates are read over the rest."""
    methods = get_args(switchback.Method.__value__)
    assert {f"switchback:{m}" for m in methods} <= set(aa_audit.READERS)
    calls = iter(range(10))

    def flaky(kpi: np.ndarray, seed: int) -> aa_audit.Read:
        if next(calls) % 2:
            msg = "refused"
            raise ValueError(msg)
        return aa_audit.Read(0.01, 1.0, 2.0)

    monkeypatch.setitem(aa_audit.READERS, "flaky", flaky)
    cell = aa_audit.audit("flaky", "ar1", 10, 4)
    assert (cell.read, cell.refused, cell.fpr, cell.coverage) == (2, 2, 1.0, 0.0)
    assert not cell.passes
