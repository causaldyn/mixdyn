"""The report's reading of a record's chains: which fits are past the R-hat and divergence bounds,
for the engine's arms and each tool's record, and the second reading, with each such plan at the
status quo's regret."""

import dataclasses

import pytest

pytest.importorskip("causaldyn_bench")

from causaldyn_bench.scorecard.scoring import ArmScore, WorldRecord

from scorecard.report import DIVERGED, RHAT, gated, past, table

SAMPLER = {"chains": 4, "warmup": 1000, "draws": 1000}


def _static(hill=0.5, **flags):
    """A record of the engine's static arm, both fits within the bounds unless `flags` say."""
    every = {"hill weight": hill}
    for family in ("hill", "tanh"):
        every |= {f"{family} max rhat": 1.005, f"{family} divergences": 3}
    every |= {key.replace("_", " "): value for key, value in flags.items()}
    settings = {"families": ["hill", "tanh"], "sampler": SAMPLER, "readings": 400}
    return {"error": None, "flags": every, "settings": settings}


def _tool(rhat=1.005, divergences=3, *, meridian=False):
    if meridian:
        settings = {"sampling": {"n_chains": 10, "n_adapt": 2000, "n_burnin": 500, "n_keep": 1000}}
    else:
        settings = {"chains": 4, "draws": 1000}
    flags = {"divergences": divergences, "max_rhat": rhat, "min_ess_bulk": 900.0}
    return {"error": None, "flags": flags, "settings": settings}


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        (_static(), False),
        (_static(hill_max_rhat=RHAT), False),
        (_static(hill_max_rhat=1.0101), True),
        (_static(tanh_divergences=27), False),
        (_static(tanh_divergences=28), True),
        # a family the readings hold no draw of is not read: round(0.001 * 400) is 0
        (_static(hill=0.001, hill_max_rhat=1.5, hill_divergences=900), False),
        (_static(hill=1.0, tanh_max_rhat=1.5), False),
        (_static(hill=0.002, hill_max_rhat=1.5), True),  # round(0.8) is 1 reading
        (_static(tanh_max_rhat=None), True),
        (_tool(), False),
        (_tool(divergences=27), False),
        (_tool(divergences=28), True),
        (_tool(rhat=1.02), True),
        (_tool(rhat=None), True),
        (_tool(divergences=67, meridian=True), False),  # 27 in 4 000 of 10 000 draws is 67.5
        (_tool(divergences=68, meridian=True), True),
        ({"error": None, "flags": {"fitted not mixed": 0, "kept": 40}, "settings": {}}, False),
        ({"error": None, "flags": {"fitted not mixed": 2, "kept": 40}, "settings": {}}, True),
        ({"error": None, "flags": {"fitted not mixed": None}, "settings": {}}, True),
        ({"error": None, "flags": {"elbo": -32.5, "elbo_drift": -0.01}, "settings": {}}, None),
        ({"error": None, "flags": {"odd": None}, "settings": {}}, None),
        ({"error": None}, None),  # a built-in arm
        ({"error": "ValueError: no fit", "failure_kind": "fit"}, None),
    ],
)
def test_a_fit_is_past_the_bounds_where_a_chain_it_reads_is(record, expected):
    assert past(record) is expected


def test_a_record_whose_chains_the_bounds_do_not_name_is_refused():
    with pytest.raises(ValueError, match="no bound is stated"):
        past({"error": None, "flags": {"r_hat": 1.0, "rhat_max": 1.0}, "settings": {}})


def test_the_bounds_are_adr_0024_s_and_adr_0025_s():
    assert RHAT == 1.01
    assert DIVERGED * 4000 == 27


def _score(regret, record, failure=None):
    return ArmScore(regret, failure, None, 0, None, None, None, None, record)


def _world(index, ours, tool):
    arms = {
        "status quo": _score(0.5 + index, {"error": None}),
        "equal split": _score(0.7 + index, {"error": None}),
        "ours": _score(0.1 + index, ours),
        "tool": _score(0.3 + index, tool),
    }
    return WorldRecord(0, "walk", 100 + index, index, True, 0, 1.0, 1.0, 3, 0, {}, arms)


def test_the_second_reading_puts_each_plan_past_the_bounds_at_the_status_quo():
    worlds = [_world(0, _static(), _tool(rhat=1.5)), _world(1, _static(), _tool())]
    second = gated(worlds)
    assert second[0].arms["tool"] == dataclasses.replace(worlds[0].arms["tool"], regret=0.5)
    for name in ("status quo", "equal split", "ours"):
        assert second[0].arms[name] == worlds[0].arms[name]
    assert second[1] == worlds[1]


def test_the_table_reads_twice_where_a_plan_is_past_the_bounds_and_counts_failures_by_kind():
    failed = {"error": "the world's process ended", "failure_kind": "timeout"}
    worlds = [_world(i, _static(), _tool(rhat=1.5 if i == 0 else 1.0)) for i in range(3)]
    worlds[2] = dataclasses.replace(
        worlds[2],
        arms=worlds[2].arms
        | {
            "ours": _score(2.5, failed, failure=failed["error"]),
            "tool": _score(2.5, {"error": "x"}, failure="ValueError: x"),
        },
    )
    shown = "\n".join(table(worlds, ["ours"]))
    assert "failed 1 (timeout 1), moved 0, past the bounds 0" in shown
    assert "failed 1 (raised 1), moved 0, past the bounds 1" in shown
    assert "at the status quo:" in shown
    assert shown.count("ours less, regret per euro") == 2
    calm = [_world(i, _static(), _tool()) for i in range(3)]
    assert "at the status quo:" not in "\n".join(table(calm, ["ours"]))
