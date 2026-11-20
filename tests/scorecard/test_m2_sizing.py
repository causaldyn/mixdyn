"""M2's sizing script declares a family's sized worlds, rounded up to whole worlds an environment;
the floor where the family is exploratory; and what its blocks hold where they hold fewer."""

from __future__ import annotations

import numpy as np
import pytest

from scorecard.preregistration import m2_sizing

RUNGS = (m2_sizing.UNTESTED, m2_sizing.TESTED)


def _worlds(
    environments: int, n: int, sd: float = 0.01, level: float = 0.0
) -> dict[tuple[str, int], dict[str, dict[str, float]]]:
    """`n` worlds in each of `environments`: the default's regret 0, and each built-in arm's at each
    rung `level` plus `sd` times draws standardised over the environment's worlds."""
    rng = np.random.default_rng(3)
    worlds: dict[tuple[str, int], dict[str, dict[str, float]]] = {}
    for e in range(environments):
        z = rng.standard_normal((len(RUNGS), len(m2_sizing.BUILT_IN), n))
        z = (z - z.mean(axis=2, keepdims=True)) / z.std(axis=2, ddof=1, keepdims=True)
        for seed in range(n):
            worlds[f"e{e}", seed] = {
                rung: {"dlm plug-in": 0.0}
                | {arm: level + sd * z[r, a, seed] for a, arm in enumerate(m2_sizing.BUILT_IN)}
                for r, rung in enumerate(RUNGS)
            }
    return worlds


@pytest.fixture(autouse=True)
def _few_resamples(monkeypatch):
    monkeypatch.setattr(m2_sizing, "RESAMPLES", 400)


def _sized(monkeypatch, family, environments, need, sd=0.01, level=0.0):
    """The family's sizing where the bootstrap finds `need` worlds, or none below `MOST`."""
    found = None if need is None else (need, [0.9] * 2 * len(m2_sizing.BUILT_IN), 0.9)
    monkeypatch.setattr(m2_sizing, "needed", lambda strata, scales: found)
    return m2_sizing.size(family, _worlds(environments, 10, sd, level), [], "dlm plug-in")


def test_a_family_sized_below_the_floor_declares_the_floor_in_whole_worlds_an_environment(
    monkeypatch,
):
    sized = _sized(monkeypatch, 2, 7, 120)
    assert (sized["needed"], sized["final"], sized["declared"]) == (120, 245, 245)
    assert (sized["worlds an environment"], sized["exploratory"]) == (35, False)


def test_a_family_sized_within_the_cap_declares_its_size(monkeypatch):
    sized = _sized(monkeypatch, 2, 3, 380)
    assert (sized["final"], sized["declared"], sized["exploratory"]) == (381, 381, False)


@pytest.mark.parametrize("need", [530, 2260, None])
def test_a_family_past_the_cap_or_unsized_is_exploratory_and_declares_the_floor(monkeypatch, need):
    sized = _sized(monkeypatch, 1, 6, need)
    assert (sized["declared"], sized["worlds an environment"], sized["exploratory"]) == (
        240,
        40,
        True,
    )
    assert sized["final"] == (None if need is None else 6 * -(-need // 6))


def test_a_family_whose_blocks_hold_fewer_worlds_declares_them_and_is_exploratory(monkeypatch):
    assert m2_sizing.HELD == {15: 100}
    held = _sized(monkeypatch, 15, 1, 20)
    assert (held["final"], held["declared"], held["exploratory"]) == (240, 100, True)
    drawn = _sized(monkeypatch, 14, 1, 20)
    assert (drawn["declared"], drawn["exploratory"]) == (240, False)


def test_the_prediction_reads_the_declared_worlds_not_the_sized_ones(monkeypatch):
    """Family 15 declares 100 worlds where its sizing says 240: each chance is the bootstrap's at
    100, which differences 0.05 below nought, of SD 0.25, tell from its chances at 240."""
    held = _sized(monkeypatch, 15, 1, 20, sd=0.25, level=0.05)
    worlds = _worlds(1, 10, 0.25, 0.05)
    m, _, _ = m2_sizing.matrix(worlds, "dlm plug-in", list(m2_sizing.BUILT_IN))
    scales = np.array([c["scale"] for c in held["comparisons"]])
    each, joint = m2_sizing.predictive([m], scales, 100)
    assert [c["predictive at declared"] for c in held["comparisons"]] == each
    assert held["predictive joint at declared"] == joint
    assert 0.0 < joint < held["predictive joint at final"]
