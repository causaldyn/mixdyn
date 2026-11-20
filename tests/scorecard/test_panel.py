"""What the arms read of an export: its lift rows read back as the tests the world ran, and a plan's
worth as `chc.allocation` reads it."""

import dataclasses

import numpy as np
import pytest

pytest.importorskip("causaldyn_bench")

from causaldyn_bench.lift_calibration import COOLDOWN, PRE, TEST
from causaldyn_bench.observational_check import true_channel
from causaldyn_bench.scorecard import ladder
from causaldyn_bench.scorecard.track_m2 import TRACK_M2
from chc.allocation import allocate

from mixdyn import NATIONAL
from mixdyn.models import DlmSpec, Effect
from mixdyn.models.dlm_lift import lift_reading
from scorecard.arms import panel


@pytest.fixture(scope="module")
def world():
    return TRACK_M2.world("drawn", 905)


@pytest.fixture(scope="module")
def observed(world):
    return TRACK_M2.observe(world, 4)


def test_each_lift_estimate_holds_the_gap_the_world_made_over_its_readout(world, observed):
    history = world.history
    read = panel.estimates(observed)
    assert sum(len(rows) for rows in read.values()) == len(observed.lift.channel) > 0
    for name, rows in read.items():
        truth = true_channel(world.generator, name, history.kernel_length)
        lift = observed.lift
        starts = [s for c, s in zip(lift.channel, lift.start, strict=True) if c == name]
        for start, row in zip(starts, rows, strict=True):
            readout = slice(start - 1, start - 1 + TEST + COOLDOWN)
            gap = history.geo_test(name, (start,), test=TEST).true_gap[readout].sum()
            ours = np.asarray(truth(row.spend) - truth(row.planned))[-row.readout :].sum()
            assert ours == pytest.approx(gap, rel=1e-12, abs=0.0)
            assert row.readout == TEST + COOLDOWN
            assert row.share == 1.0


def test_each_lift_estimate_reads_its_tests_gap_summed_over_the_readout(world, observed):
    tests = ladder.rung(world.history, world.seed, 4)
    lift = observed.lift
    for name, rows in panel.estimates(observed).items():
        # a test's readout opens PRE weeks before its first dark week
        by_start = {t.control.history + PRE + 1: t for t in tests[name]}
        starts = [s for c, s in zip(lift.channel, lift.start, strict=True) if c == name]
        noise = np.sqrt(np.mean([np.var(t.difference[:PRE], ddof=1) for t in tests[name]]))
        for start, row in zip(starts, rows, strict=True):
            gap = float(by_start[start].difference[PRE:].sum())
            assert row.lift == pytest.approx(gap, rel=1e-12, abs=0.0)
            assert row.se == pytest.approx(noise * np.sqrt(TEST + COOLDOWN), rel=1e-12, abs=0.0)


def test_each_lift_estimate_ends_at_its_readout_s_last_week_and_the_dlm_reads_its_dark_weeks(
    observed,
):
    """Without a transform, the DLM reads a go-dark at the weeks it went dark, each weighted by
    the history's spend there taken away, and nowhere else."""
    data = panel.panel(observed)
    lift = observed.lift
    spec = DlmSpec(tuple(Effect(c, 1.0) for c in observed.channels))
    read = panel.estimates(observed)
    assert sum(len(rows) for rows in read.values()) > 0
    for name, rows in read.items():
        j = observed.channels.index(name)
        starts = [s for c, s in zip(lift.channel, lift.start, strict=True) if c == name]
        for start, row in zip(starts, rows, strict=True):
            assert row.end == data.dates[start - 2 + TEST + COOLDOWN]
            weights = lift_reading(data, NATIONAL, spec, {name: (row,)}).weights[0]
            dark = slice(start - 1, start - 1 + TEST)
            np.testing.assert_array_equal(weights[dark], -observed.spend[dark, j])
            assert not np.delete(weights, np.arange(dark.start, dark.stop)).any()


def test_a_lift_row_whose_spend_is_not_the_historys_is_refused(observed):
    lift = dataclasses.replace(observed.lift, x=observed.lift.x * (1.0 + 1e-9))
    with pytest.raises(ValueError, match="where the history spent"):
        panel.estimates(dataclasses.replace(observed, lift=lift))


def test_a_plans_worth_is_chcs_and_the_status_quo_gains_nothing(world, observed):
    channels = [
        true_channel(world.generator, name, world.history.kernel_length)
        for name in observed.channels
    ]
    weekly = observed.status_quo * np.array([1.3, 0.8, 0.9])[: observed.status_quo.size]
    periods = observed.planned
    held = allocate(
        channels,
        periods * float(weekly.sum()),
        periods,
        lower=weekly,
        upper=weekly,
        history=observed.spend,
    )
    ours = sum(
        panel.worth(c, observed.spend[:, j], float(weekly[j]), periods)
        for j, c in enumerate(channels)
    )
    assert ours == pytest.approx(held.worth, rel=1e-12, abs=0.0)
    assert panel.gains([channels], observed, observed.status_quo)[0] == 0.0
