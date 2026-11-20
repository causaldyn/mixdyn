"""The DLM arm's flags: the best spec's discounts, the trend's and each effect's, whether the best
sits at an end of a grid its parameter goes on past, and how many such ends the kept set reaches;
and the lift tests it hands its search, which its ablation is not handed."""

from types import MappingProxyType

import numpy as np
import pytest

pytest.importorskip("causaldyn_bench")

from causaldyn_bench.scorecard.track_m2 import TRACK_M2

from mixdyn import NATIONAL
from mixdyn.models import DlmChoice, DlmSpec, Edge, Effect, Season, Trend, fit_dlm
from mixdyn.transforms import TanhShape, Transform
from scorecard.arms import dlm
from scorecard.arms import panel as reading
from scorecard.arms.base import records


def test_the_dlm_arm_flags_the_ends_its_best_sits_at_and_counts_those_its_set_reaches() -> None:
    observation = TRACK_M2.observe(TRACK_M2.world("drawn", 905), 0)
    spend = np.asarray(observation.spend, dtype=float)
    discounts = (0.9, 0.95, 0.99)
    media = [
        Effect(
            name,
            discounts[j],
            transform=Transform(
                0.5,
                float(spend[spend[:, j] > 0.0, j].mean()),
                TanhShape(),
                observation.kernel_length,
            ),
        )
        for j, name in enumerate(observation.channels)
    ]
    spec = DlmSpec(
        effects=(*media, *(Effect(c, 1.0) for c in observation.controls)),
        trend=Trend(1, 0.98),
        season=Season(dlm.YEAR, dlm.HARMONICS, 1.0),
    )
    fit = fit_dlm(reading.panel(observation), NATIONAL, spec)
    first, second, _ = observation.channels
    edges = (
        Edge("trend discount", 0.98, best=True),
        Edge(f"{first} discount", 0.8, best=False),
        Edge(f"{first} retention", 0.9, best=True),
        Edge(f"{second} half-saturation", 4.0, best=True),
    )
    choice = DlmChoice((spec,), (fit.filtered,), 0, MappingProxyType({0: fit}), edges)
    flags = dlm._flags(choice, observation.channels)
    assert flags["trend discount"] == 0.98
    assert {c: flags[f"{c} discount"] for c in (*observation.channels, *observation.controls)} == {
        e.column: e.discount for e in spec.effects
    }
    ends = {key: value for key, value in flags.items() if key.endswith("at an end")}
    assert {key for key, value in ends.items() if value} == {
        "trend discount at an end",
        f"{first} retention at an end",
        f"{second} half-saturation at an end",
    }
    assert len(ends) == 1 + len(spec.effects) + 2 * len(observation.channels)
    assert flags["ends the kept set reaches"] == 4
    assert (flags["kept"], flags["fitted"], flags["fitted not mixed"]) == (1, 1, 0)


class Searched(Exception):
    """What a search that only records its arguments raises, before any fit."""


@pytest.mark.parametrize(("arm", "read"), [(dlm.Dlm(), True), (dlm.DlmUntested(), False)])
def test_the_dlm_arm_hands_its_search_the_export_s_tests_and_its_ablation_none(
    arm, read, monkeypatch
) -> None:
    observation = TRACK_M2.observe(TRACK_M2.world("drawn", 905), 4)
    expected = reading.estimates(observation)
    assert sum(len(rows) for rows in expected.values()) > 0
    handed = []

    def search(*args, **kwargs):
        handed.append(kwargs["estimates"])
        raise Searched

    monkeypatch.setattr(dlm, "search_media", search)
    made = records(arm, observation, smoke=True)
    assert all("Searched" in record["error"] for record in made.values())
    assert arm.tests is read
    if not read:
        assert handed == [None]
        return
    (estimates,) = handed
    assert list(estimates) == list(expected)
    for name, rows in expected.items():
        assert [(r.lift, r.se, r.end, r.readout) for r in estimates[name]] == [
            (r.lift, r.se, r.end, r.readout) for r in rows
        ]
