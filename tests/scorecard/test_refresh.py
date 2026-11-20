"""The refresh cell's harness: an arm fitted on every cut of a world in turn, each cut's records
kept once, a stopped run resumed at the cut it stopped at, and every complete world read by the
bench's swing or rolling origin; a tool's arm on each cut as the runner runs a tool's world."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("causaldyn_bench")

from scorecard import refresh
from scorecard.arms.base import Planned
from scorecard.layout import FAMILIES, Limits
from scorecard.tools import TOOLS, Tool

FAKE = Path(__file__).with_name("fake_tool.py")


class Still:
    """An arm that plays the status quo and counts its fits."""

    variants = ("still", "also still")
    tests = False
    cells = False

    def __init__(self):
        self.fits = 0

    def fit(self, observation, *, smoke):
        self.fits += 1
        return observation.status_quo

    def plan(self, fit, variant, observation):
        return Planned(weekly=np.asarray(fit, dtype=float))


def test_a_swing_records_every_cut_once_and_resumes_where_it_stopped(tmp_path):
    arm = Still()
    refresh.fit(tmp_path, "swing", 0, "drawn", 900, 0, arm, smoke=True)
    assert arm.fits == 7
    place = refresh.directory(tmp_path, "swing", 0, "drawn", 900, 0)
    held = sorted(p.relative_to(place).as_posix() for p in place.rglob("*.json"))
    assert held == [f"{v}/{i}.json" for v in ("also-still", "still") for i in range(7)]
    record = json.loads((place / "still" / "3.json").read_text())
    assert (record["variant"], record["cut"], record["weeks"]) == ("still", 3, 144)
    (place / "still" / "5.json").unlink()
    refresh.fit(tmp_path, "swing", 0, "drawn", 900, 0, arm, smoke=True)
    assert arm.fits == 8
    rows = refresh.read(tmp_path)
    assert [(r["kind"], r["arm"], r["smoke"]) for r in rows] == [
        ("swing", "also still", True),
        ("swing", "still", True),
    ]
    read = rows[0]["reading"]
    assert read["fits"] == 7
    assert read["unread"] == 7
    assert read["largest move"] == 0.0


def test_a_rolling_origin_reads_family_7_s_window_as_it_rolls(tmp_path):
    environment = "L-52"
    seed = FAMILIES[7].pilots[environment][0]
    refresh.fit(tmp_path, "rolling", 7, environment, seed, 0, Still(), smoke=True)
    data = refresh.report(tmp_path)
    (row,) = [r for r in data["rows"] if r["arm"] == "still"]
    assert (row["kind"], row["family"], row["environment"], row["seed"]) == (
        "rolling",
        7,
        environment,
        seed,
    )
    assert row["reading"]["window"] == [105, 132]
    assert row["reading"]["crps"] == [None] * 7
    assert (tmp_path / "refresh.md").read_text().startswith("## Refresh swing")
    assert json.loads((tmp_path / "refresh.json").read_text())["provenance"]["mixdyn"]


def test_a_cut_is_a_swing_or_a_rolling_origin():
    family = FAMILIES[0]
    with pytest.raises(ValueError, match="refresh is one of"):
        refresh.cuts("sliding", family, family.world("drawn", 900), 0)


def test_a_world_short_of_a_cut_is_not_read(tmp_path):
    refresh.fit(tmp_path, "swing", 0, "drawn", 900, 0, Still(), smoke=True)
    place = refresh.directory(tmp_path, "swing", 0, "drawn", 900, 0)
    (place / "still" / "6.json").unlink()
    assert [r["arm"] for r in refresh.read(tmp_path)] == ["also still"]


def _scopes() -> bool:
    if shutil.which("systemd-run") is None:
        return False
    command = ["systemd-run", "--user", "--scope", "-q", "true"]
    return subprocess.run(command, capture_output=True, check=False).returncode == 0


scopes = pytest.mark.skipif(not _scopes(), reason="no user systemd to run a scope")


@pytest.mark.parametrize(("mode", "ended"), [("record", None), ("exit", "exit")])
@scopes
def test_a_tool_is_fitted_on_every_cut_as_the_runner_runs_a_world(
    tmp_path, monkeypatch, mode, ended
):
    modes = tmp_path / "modes.json"
    modes.write_text(json.dumps({"900": mode}))
    monkeypatch.setitem(TOOLS, "fake", Tool((sys.executable, str(FAKE), "--modes", str(modes))))
    limits = Limits(memory="512M", timeout=60.0, slots=1)
    refresh.fit_tool(tmp_path, "swing", 0, "drawn", 900, 0, "fake", limits, smoke=True)
    place = refresh.directory(tmp_path, "swing", 0, "drawn", 900, 0)
    observations = refresh.cuts("swing", FAMILIES[0], FAMILIES[0].world("drawn", 900), 0)
    for i, observation in enumerate(observations):
        record = json.loads((place / "fake" / f"{i}.json").read_text())
        assert (record["variant"], record["cut"]) == ("fake", i)
        assert record["weeks"] == observation.sales.shape[0]
        assert record["digest"] == observation.digest()
        assert record.get("failure_kind") == ended
    starts = len((tmp_path / "modes.starts" / "900").read_text().splitlines())
    assert starts == 7 * (1 if mode == "record" else refresh.ATTEMPTS)
    refresh.fit_tool(tmp_path, "swing", 0, "drawn", 900, 0, "fake", limits, smoke=True)
    assert len((tmp_path / "modes.starts" / "900").read_text().splitlines()) == starts
    (row,) = refresh.read(tmp_path)
    assert (row["arm"], row["reading"]["fits"]) == ("fake", 7)
    assert row["reading"]["failed"] == (0 if mode == "record" else 7)


def test_a_cell_runs_each_cut_in_a_scope_of_its_arm_s_limits_and_keeps_a_dead_one(
    tmp_path, monkeypatch
):
    started = []

    def wait(command, log_path, timeout, **held):
        def given(flag):
            return command[command.index(flag) + 1]

        arm, i = given("--arm"), int(given("--cut"))
        started.append((arm, i, command, log_path, timeout, held["cap"]))
        if arm == "still" and i == 6:
            return "exit code 1"
        where = (given("--kind"), int(given("--family")), given("--environment"))
        place = refresh.directory(tmp_path, *where, int(given("--world")), int(given("--k")))
        for variant in Still.variants if arm == "still" else (arm,):
            path = place / variant.replace(" ", "-") / f"{i}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
        return "exit code 0"

    monkeypatch.setattr(refresh, "wait", wait)
    monkeypatch.setitem(refresh.ARMS, "still", Still)
    plan = {
        "jobs": [
            ["swing", 0, "drawn", 900, 4, "still"],
            ["rolling", 7, "L-52", 340000, 0, "robyn"],
        ],
        "limits": {"still": ["1G", 600], "robyn": ["8G", 100]},
        "r_library": "/r",
        "nevergrad": "/ng",
    }
    ended = refresh.cell(tmp_path, plan, 2, smoke=True)
    assert ended == ["6 of 7 cuts recorded", "7 of 7 cuts recorded"]
    still = [s for s in started if s[0] == "still"]
    assert [s[1] for s in still] == [0, 1, 2, 3, 4, 5, 6, 6]
    _, _, command, log_path, timeout, cap = still[0]
    assert command[command.index("MemoryMax=1G") + 1 :][:3] == [
        "-p",
        "MemorySwapMax=0",
        sys.executable,
    ]
    assert command[command.index("run") :] == [
        *("run", "--root", str(tmp_path), "--kind", "swing", "--family", "0"),
        *("--environment", "drawn", "--world", "900", "--k", "4", "--arm", "still", "--smoke"),
        *("--cut", "0"),
    ]
    assert log_path == tmp_path / "logs/cell/swing/family-0/drawn/k4/900-still.log"
    assert (timeout, cap) == (600.0, 2**30)
    place = refresh.directory(tmp_path, "swing", 0, "drawn", 900, 4)
    observations = refresh.cuts("swing", FAMILIES[0], FAMILIES[0].world("drawn", 900), 4)
    for variant in Still.variants:
        dead = json.loads((place / variant.replace(" ", "-") / "6.json").read_text())
        assert (dead["variant"], dead["cut"], dead["failure_kind"]) == (variant, 6, "exit")
        assert dead["digest"] == observations[6].digest()
        assert dead["weeks"] == observations[6].sales.shape[0]
    robyn = [s for s in started if s[0] == "robyn"]
    assert [s[1] for s in robyn] == list(range(7))
    _, _, command, _, timeout, cap = robyn[0]
    assert command[command.index("--memory") :] == [
        *("--memory", "8G", "--timeout", "100.0", "--r-library", "/r", "--nevergrad", "/ng"),
        *("--cut", "0"),
    ]
    assert (timeout, cap) == (100.0 * refresh.ATTEMPTS + refresh.EXPORT_SECONDS, 8 * 2**30)
    assert refresh.cell(tmp_path, plan, 1) == ["7 of 7 cuts recorded"] * 2
    assert len(started) == 15
    with pytest.raises(ValueError, match=r"no limits for \['still'\]"):
        refresh.cell(tmp_path, plan | {"limits": {"robyn": ["8G", 100]}}, 1)
    with pytest.raises(ValueError, match=r"runs no arm \['nothing'\]"):
        refresh.cell(tmp_path, plan | {"limits": plan["limits"] | {"nothing": ["1G", 1]}}, 1)


def test_a_run_fits_one_cut_when_asked(tmp_path):
    arm = Still()
    refresh.fit(tmp_path, "swing", 0, "drawn", 900, 0, arm, smoke=True, only=3)
    place = refresh.directory(tmp_path, "swing", 0, "drawn", 900, 0)
    assert arm.fits == 1
    assert sorted(p.name for p in (place / "still").iterdir()) == ["3.json"]
    with pytest.raises(ValueError, match="cuts 0 to 6, not 7"):
        refresh.fit(tmp_path, "swing", 0, "drawn", 900, 0, arm, smoke=True, only=7)
