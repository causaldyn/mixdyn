"""The simulation-based calibration harness (`validation.sbc`, ADR 0059): it runs end to end on the
smallest engine, its ranks are in range, and its uniformity test does not reject there; a rank, the
bins and the test read as stated; and what it cannot simulate is refused. The test pins the
harness: whether an engine is calibrated is the records' question, at 200 simulations."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from validation.sbc import (
    THINNED,
    Geos,
    bins,
    histogram,
    markdown,
    rank,
    run,
    uniformity,
    world,
)


@pytest.fixture(scope="module")
def document() -> dict[str, Any]:
    return run("dlm", "small", 12, 0, command="python -m validation.sbc --engine dlm")


def test_the_harness_runs_on_the_smallest_engine(document: dict[str, Any]) -> None:
    summary = document["summary"]
    assert (summary["simulations"], summary["fitted"], summary["failed"]) == (12, 12, 0)
    assert document["coordinates"] == [
        "level",
        "coefficient[tv]",
        "coefficient[search]",
        "log_variance",
    ]
    for record in document["records"]:
        assert len(record["true"]) == len(record["ranks"]) == 4
        assert record["divergent"] is None
        assert record["compute"]["failure_kind"] is None
    assert json.loads(json.dumps(document)) == document
    text = markdown(document)
    assert "python -m validation.sbc --engine dlm" in text
    assert document["date"] in text


def test_its_ranks_are_in_range(document: dict[str, Any]) -> None:
    ranks = np.array([r["ranks"] for r in document["records"]])
    assert ranks.dtype.kind == "i"
    assert ranks.min() >= 0
    assert ranks.max() <= THINNED


def test_its_uniformity_test_does_not_reject(document: dict[str, Any]) -> None:
    """At 0.001, block by block: on 12 simulations the test has little power, and the check is
    that the harness, its ranks and its test, does not break calibration it has."""
    blocks = document["summary"]["uniformity"]["blocks"]
    assert set(blocks) == {"level", "coefficient", "log_variance"}
    assert all(b["p"] > 0.001 for b in blocks.values())


def test_a_rank_counts_the_thinned_draws_below_the_truth() -> None:
    """Among `THINNED` draws evenly spaced over the chains in order: a truth below every draw
    ranks 0, above every draw `THINNED`, and between the thinned draws by how many lie below."""
    draws = np.arange(4 * 100, dtype=float).reshape(4, 100, 1)
    assert rank(np.array([-1.0]), draws).tolist() == [0]
    assert rank(np.array([1e9]), draws).tolist() == [THINNED]
    kept = np.linspace(0, 399, THINNED).round()
    assert rank(np.array([200.5]), draws).tolist() == [int((kept < 200.5).sum())]


@pytest.mark.parametrize(
    ("simulations", "count"), [(1, 2), (12, 2), (40, 8), (200, 32), (10**6, 64)]
)
def test_the_bins_expect_five_simulations_each(simulations: int, count: int) -> None:
    assert bins(simulations) == count


def test_a_histogram_folds_the_64_ranks_into_equal_bins() -> None:
    assert histogram(np.arange(THINNED + 1), 8).tolist() == [8] * 8
    assert histogram(np.array([0, 7, 8, 63]), 8).tolist() == [2, 1, 0, 0, 0, 0, 0, 1]


def test_uniformity_reads_each_coordinate_and_bounds_each_block() -> None:
    """Uniform ranks are not rejected; ranks piled at 0 are, and a block's p-value is its least
    coordinate's times its size, at most 1."""
    uniform = np.tile(np.arange(THINNED + 1), 5)
    piled = np.zeros_like(uniform)
    ranks = np.column_stack([uniform, piled, uniform])
    read = uniformity(ranks, ["a[x]", "a[y]", "b"])
    assert read["bins"] == bins(uniform.size)
    assert read["coordinates"]["a[x]"] == pytest.approx(1.0)
    assert read["coordinates"]["a[y]"] < 1e-12
    assert read["blocks"]["a"]["p"] == 2 * read["coordinates"]["a[y]"]
    assert read["blocks"]["b"] == {"p": pytest.approx(1.0), "coordinates": ["b"]}


@pytest.mark.parametrize(
    ("engine", "name", "knobs", "match"),
    [
        ("stan", "small", {}, "an engine is one of"),
        ("bayes_mmm", "large", {}, "a world one of"),
        ("bayes_mmm", "centred-geo", {}, "the centred-geo world is bayes_geo's"),
        ("dlm", "small", {"slope": "fixed"}, "knobs are bayes_geo's, not dlm's"),
        ("bayes_geo", "small", {"spread": "2"}, r"not \['spread'\]"),
    ],
)
def test_a_world_the_harness_does_not_have_is_refused(
    engine: str, name: str, knobs: dict[str, str], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        world(engine, name, knobs)


def test_a_knob_sets_the_geo_spec_and_the_coordinates_follow() -> None:
    centred, none = (
        world("bayes_geo", "centred-geo"),
        world("bayes_geo", "centred-geo", {"centred": ""}),
    )
    assert isinstance(centred, Geos)
    assert isinstance(none, Geos)
    assert centred.model.spec.centred == {"tv", "search"}
    assert none.model.spec.centred == set()
    normal = world("bayes_geo", "small", {"base_prior": "normal"}).coordinates
    assert [c for c in normal if "base" in c] == [f"base[geo{g}]" for g in range(4)]
    fixed = world("bayes_geo", "small", {"slope": "fixed"}).coordinates
    assert not [c for c in fixed if "slope" in c]
