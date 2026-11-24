"""The flags read as classifiers (`scorecard.flags`), on records whose AUROC is known: a flag that
alarms on every bad world and no other reads 1, one arranged to tell nothing exactly 0.5 inside
its interval, a random one near 0.5, an inverted one below 0.5 and dropped with the reason; a
channel's flag read against that channel's error; a reviewer's statuses and numbers read as
flags; and the R² against regret."""

import json
import math

import numpy as np
import pytest
from scipy import stats

pytest.importorskip("causaldyn_bench")

from causaldyn_bench.scorecard.scoring import ArmScore, WorldRecord

from scorecard import flags as f

CHANNELS = ("tv", "search")


def _score(regret, record, returns=None, failure=None):
    return ArmScore(regret, failure, None, 0, None, None, None, None, record, returns=returns)


def _world(index, regret, numbers, *, errors=(0.0, 0.0), status_quo=1.0, extra=None):
    """A world whose arm `ours` plans at `regret` against the status quo's `status_quo`, its
    record's flags `numbers`, each channel's ROI 1 and its draws' mean `1 + error`."""
    returns = {
        c: {"roi": [1.0 + e, 0.5, 2.0], "mroi": [1.0, 0.5, 2.0]}
        for c, e in zip(CHANNELS, errors, strict=True)
    }
    record = {"error": None, "flags": numbers} | (extra or {})
    arms = {
        "status quo": _score(status_quo, {"error": None}),
        "equal split": _score(status_quo, {"error": None}),
        "ours": _score(regret, record, returns),
    }
    truth = {c: {"roi": 1.0, "mroi": 1.0} for c in CHANNELS}
    return WorldRecord(0, "walk", index, index, False, 0, 1.0, 1.0, 2, 0, {}, arms, truth)


def test_a_flag_that_alarms_on_every_bad_plan_reads_1_and_one_inverted_0() -> None:
    worlds = [
        _world(
            i,
            2.0 if i % 3 == 0 else 0.5,
            {"max rhat": 1.2 if i % 3 == 0 else 1.0, "min ess bulk": 50.0 if i % 3 else 900.0},
        )
        for i in range(30)
    ]
    drawn = f.resamples(len(worlds))
    perfect = f.scored(f.units(worlds, "ours", "regret")["max rhat"], drawn)
    assert perfect is not None
    assert (perfect.value, perfect.low, perfect.high) == (1.0, 1.0, 1.0)
    assert (perfect.worlds, perfect.units, perfect.with_outcome) == (30, 30, 10)
    inverted = f.scored(f.units(worlds, "ours", "regret")["min ess bulk"], drawn)
    assert inverted is not None
    assert (inverted.value, inverted.high) == (0.0, 0.0)
    shown = "\n".join(f.lines(worlds))
    kept, dropped = shown.split("  dropped:")
    assert "ours 'max rhat' against regret: 1.000 [1.000, 1.000], 10 of 30" in kept
    assert "tells them apart" in kept
    assert "'min ess bulk' against regret" not in kept
    assert "ours 'min ess bulk' against regret: 0.000" in dropped
    assert "below 0.5, the flag alarms the wrong way" in dropped


def test_a_flag_arranged_to_tell_nothing_reads_one_half_inside_its_interval() -> None:
    """Blocks of four worlds, the flag 4b + 1 to 4b + 4 in block b and the second and third plans
    bad: within a block a bad plan beats a good one in half the pairs, and across blocks the
    higher block's plans beat the lower's whatever their outcome, so the AUROC is 1/2 exactly."""
    worlds = [_world(i, 2.0 if i % 4 in (1, 2) else 0.5, {"nrmse": float(i)}) for i in range(40)]
    read = f.scored(f.units(worlds, "ours", "regret")["nrmse"], f.resamples(40))
    assert read is not None
    assert read.value == 0.5
    assert read.low < 0.5 < read.high


def test_a_random_flag_reads_near_one_half() -> None:
    """400 worlds, half the plans bad, the flag uniform apart from them: the AUROC's standard
    error is `sqrt(401 / (12 x 200 x 200))` = 0.029, so 0.1 from 1/2 is 3.4 of it and a false
    red is 1 in 1 400 over seeds; the interval holds the AUROC and is about 4 errors wide."""
    rng = np.random.default_rng(3)
    bad = rng.permutation(np.arange(400) < 200)
    worlds = [
        _world(i, 2.0 if bad[i] else 0.5, {"decomp_rssd": float(rng.uniform())}) for i in range(400)
    ]
    read = f.scored(f.units(worlds, "ours", "regret")["decomp_rssd"], f.resamples(400))
    assert read is not None
    assert abs(read.value - 0.5) < 0.1
    assert read.low < read.value < read.high
    assert 0.07 < read.high - read.low < 0.16


def _auroc(units, drawn):
    read = f.scored(units, drawn)
    assert read is not None
    return read.value


def test_a_channel_s_flag_reads_its_own_channel_s_error() -> None:
    """tv's flag alarms where tv's ROI is off by more than tau and search's where search's is: each
    reads 1 against its own channel. A flag of the fit reads against some channel being off; a
    channel's flag against regret reads at its most alarming channel."""
    worlds = []
    for i in range(24):
        tv, search = i % 2 == 0, i % 3 == 0
        numbers = {
            "tv retention at an end": tv,
            "search retention at an end": search,
            "max rhat": 1.1 if tv or search else 1.0,
        }
        errors = (0.9 if tv else 0.1, 0.9 if search else 0.1)
        worlds.append(_world(i, 2.0 if tv or search else 0.5, numbers, errors=errors))
    drawn = f.resamples(24)
    by_channel = f.units(worlds, "ours", "return")
    own = by_channel["<channel> retention at an end"]
    assert own.value.size == 48
    assert _auroc(own, drawn) == 1.0
    assert _auroc(by_channel["max rhat"], drawn) == 1.0
    regret = f.units(worlds, "ours", "regret")["<channel> retention at an end"]
    assert regret.value.size == 24
    assert _auroc(regret, drawn) == 1.0


def test_a_record_written_before_the_flat_flag_reads_as_it_did() -> None:
    """Half the worlds' records predate the `flat` flag: the flag is read over the other half's
    channels alone, and every record's other flags as before."""
    worlds = []
    for i in range(24):
        tv, search = i % 2 == 0, i % 3 == 0
        numbers: dict[str, float | bool] = {
            "tv retention at an end": tv,
            "search retention at an end": search,
        }
        if i >= 12:
            numbers |= {"tv flat": tv, "search flat": search}
        errors = (0.9 if tv else 0.1, 0.9 if search else 0.1)
        worlds.append(_world(i, 0.5, numbers, errors=errors))
    by_channel = f.units(worlds, "ours", "return")
    assert by_channel["<channel> retention at an end"].value.size == 48
    flat = by_channel["<channel> flat"]
    assert flat.value.size == 24
    assert set(flat.world.tolist()) == set(range(12, 24))
    assert _auroc(flat, f.resamples(24)) == 1.0


def test_tau_sets_which_returns_are_wrong() -> None:
    world = _world(0, 0.5, {"max rhat": 1.0}, errors=(0.3, -0.6))
    assert f.wrong(world, world.arms["ours"], 0.5) == {"tv": False, "search": True}
    assert f.wrong(world, world.arms["ours"], 0.25) == {"tv": True, "search": True}
    assert f.wrong(world, world.arms["status quo"], 0.5) is None


@pytest.mark.parametrize(
    ("key", "value", "read"),
    [
        ("hill max rhat", 1.02, 1.02),
        ("min_ess_bulk", 400.0, -400.0),
        ("hill tv prior consistency", 0.98, 0.48),
        ("hill tv prior shift", -2.5, 2.5),
        ("ppp", 0.03, -0.03),
        ("rsq_train", 0.9, -0.9),
        ("tv retention at an end", True, 1.0),
        ("tv flat", True, 1.0),
        ("tv flat", False, 0.0),
        ("tv column cv", 0.01, None),
        ("converged", False, -0.0),
        ("BayesianPPPCheck", "REVIEW", 1.0),
        ("review", "FAIL", 2.0),
        ("tv retention", 0.4, None),
        ("convergence", ["DECOMP.RSSD converged"], None),
        ("max rhat", None, None),
        ("max rhat", math.nan, None),
    ],
)
def test_each_flag_reads_as_an_alarm_in_its_stated_direction(key, value, read) -> None:
    assert f.numbers({"flags": {key: value}}).get(key) == read


def test_a_reviewer_s_numbers_are_flags() -> None:
    review = {
        "checks": {
            "BayesianPPPCheck": {"status": "PASS", "details": {"bayesian_ppp": 0.2}},
            "GoodnessOfFitCheck": {"status": "PASS", "details": {"r_squared": 0.97}},
            "PotentialBiasCheck": {"status": "REVIEW", "details": "not a mapping"},
        }
    }
    numbers = f.numbers({"flags": {"GoodnessOfFitCheck": "PASS"}, "review": review})
    assert numbers == {"GoodnessOfFitCheck": 0.0, "review ppp": -0.2, "review r2": -0.97}


def test_r2_against_regret_is_their_rank_correlation() -> None:
    """R² falling as regret rises reads -1 over every resample; a failed plan holds no R²."""
    worlds = [_world(i, 0.1 * i, {"r2": 1.0 - 0.01 * i, "rsq_train": 0.5}) for i in range(20)]
    worlds.append(
        WorldRecord(
            0,
            "walk",
            20,
            20,
            False,
            0,
            1.0,
            1.0,
            2,
            0,
            {},
            worlds[0].arms | {"ours": _score(5.0, {"error": "x"}, failure="x")},
            worlds[0].returns,
        )
    )
    reads = f.fit_against_regret(worlds, "ours", f.resamples(len(worlds)))
    assert list(reads) == ["r2"]  # rsq_train is one value: no correlation
    read = reads["r2"]
    assert [read.value, read.low, read.high] == pytest.approx([-1.0] * 3, rel=1e-15, abs=0.0)
    assert reads["r2"].worlds == 20
    shown = "\n".join(f.lines(worlds))
    assert "ours 'r2': -1.000 [-1.000, -1.000] over 20 worlds" in shown


def test_the_auroc_is_mann_whitney_s_with_weights_as_copies() -> None:
    rng = np.random.default_rng(5)
    value = rng.integers(0, 4, 50).astype(float)
    outcome = rng.uniform(size=50) < 0.4
    stats_u = stats.mannwhitneyu(value[outcome], value[~outcome]).statistic
    u = f.auroc(value, outcome)
    assert u == pytest.approx(stats_u / (outcome.sum() * (~outcome).sum()), rel=1e-14, abs=0.0)
    weight = rng.integers(0, 3, 50)
    copied = f.auroc(np.repeat(value, weight), np.repeat(outcome, weight))
    assert f.auroc(value, outcome, weight) == pytest.approx(copied, rel=1e-14, abs=0.0)
    assert math.isnan(f.auroc(value, np.ones(50, dtype=bool)))


def test_the_report_prints_the_rows_read_gives() -> None:
    """`read`'s rows are the report's: each row with an AUROC a line of `lines` with its numbers,
    each without one counted as not read with its reason, and every number the one `units` and
    `scored` give apart from it. No world's return is wrong, so every flag against a wrong return
    reads one outcome; `nrmse` is one value on every world."""
    worlds = [
        _world(
            i,
            2.0 if i % 3 == 0 else 0.5,
            {
                "max rhat": 1.2 if i % 3 == 0 else 1.0,
                "min ess bulk": 50.0 if i % 3 else 900.0,
                "decomp_rssd": float(i % 5),
                "nrmse": 0.3,
            },
        )
        for i in range(30)
    ]
    rows = f.read(worlds, "ours")
    assert json.loads(json.dumps(rows, allow_nan=False)) == rows
    unread = {(r["outcome"], r["flag"]): r["unread"] for r in rows if "unread" in r}
    assert unread == {
        ("return", "decomp_rssd"): ["one outcome"],
        ("return", "max rhat"): ["one outcome"],
        ("return", "min ess bulk"): ["one outcome"],
        ("return", "nrmse"): ["one value", "one outcome"],
        ("regret", "nrmse"): ["one value"],
    }
    drawn = f.resamples(len(worlds))
    shown = f.lines(worlds)
    printed = [line for line in shown if line.startswith("    ours '") and " against " in line]
    assert len(printed) == len(rows) - len(unread) == 3
    assert f"  {len(unread)} flag readings not read" in "\n".join(shown)
    for row in rows:
        u = f.units(worlds, "ours", row["outcome"])[row["flag"]]
        assert (row["units"], row["with_outcome"], row["worlds"]) == (30, int(u.outcome.sum()), 30)
        assert row["above_zero"] == int((u.value > 0).sum())
        assert row["largest"] == u.value.max()
        found = f.scored(u, drawn)
        if found is None:
            assert "auroc" not in row
            continue
        assert (row["auroc"], row["low"], row["high"]) == (found.value, found.low, found.high)
        line = (
            f"    ours {row['flag']!r} against regret: {found.value:.3f} [{found.low:.3f}, "
            f"{found.high:.3f}], {found.with_outcome} of {found.units} with it over "
            f"{found.worlds} worlds"
        )
        assert [p for p in printed if p.startswith(line)] != []
    assert f.read(worlds, "ours", outcomes=("regret",)) == [
        r for r in rows if r["outcome"] == "regret"
    ]
