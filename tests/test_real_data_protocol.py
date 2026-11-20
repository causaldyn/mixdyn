"""The real-data protocol (`scripts/real_data_protocol.py`) runs end to end on a small synthetic
stand-in: every arm predicts every withheld test, and the cells are what the protocol defines.

The stand-in and the fits are seeded; nothing here bounds a prediction's error, so no draw can turn
a test red."""

from __future__ import annotations

import math

import numpy as np
import pytest

import real_data_protocol as protocol

SMALL = protocol.Settings(chains=2, warmup=150, draws=150, starts=16, polish=2, harmonics=0)


def test_a_cell_is_the_miss_in_the_test_s_errors_and_holds_within_the_quantile() -> None:
    near = protocol.Prediction("bayes", 0, "tv", 109.7, 100.0, 5.0)
    far = protocol.Prediction("bayes", 1, "tv", 90.0, 100.0, 5.0)
    assert near.cell == pytest.approx(1.94, rel=1e-12, abs=0.0)
    assert (near.held, far.held) == (True, False)
    (cells,) = protocol.cells([near, far])
    assert (cells.arm, cells.tests, cells.coverage) == ("bayes", 2, 0.5)
    assert cells.mean == pytest.approx((1.94 + 2.0) / 2.0, rel=1e-12, abs=0.0)


def test_fewer_than_two_tests_are_refused_before_any_fit() -> None:
    panel, tests = protocol.stand_in(weeks=104, seed=0)
    with pytest.raises(ValueError, match="at least two"):
        protocol.protocol(panel, {"tv": tests["tv"]}, SMALL)


def test_every_arm_predicts_every_withheld_test_on_the_stand_in() -> None:
    panel, tests = protocol.stand_in(weeks=104, seed=0)
    predictions = protocol.protocol(panel, tests, SMALL)
    assert [(p.arm, p.test) for p in predictions] == [
        (arm, k) for k in range(2) for arm in protocol.ARMS
    ]
    assert all(math.isfinite(p.predicted) and p.se > 0.0 for p in predictions)
    for cells in protocol.cells(predictions):
        mine = [p for p in predictions if p.arm == cells.arm]
        assert cells.tests == 2
        assert cells.mean == pytest.approx(np.mean([p.cell for p in mine]), rel=1e-12, abs=0.0)
    print(protocol.report(predictions))
