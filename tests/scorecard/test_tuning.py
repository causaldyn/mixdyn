"""The tuned rivals: the development set's split, the tuning's worlds and its fits, the rule that
chooses each rival's configuration by the rival's own criterion, and the tuned arm that runs it."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("causaldyn_bench")

from scorecard import tools, tuning
from scorecard.layout import Declaration, Layout, Limits
from scorecard.run import declare
from scorecard.tools import GRIDS, RECOMMENDED, TOOLS, Tool

FAKE = Path(__file__).with_name("fake_tool.py")


def _development(tmp_path, held):
    """A development set of empty exports: `held` maps a root to its family, environment and
    seeds at rung 0."""
    for root, (family, environment, seeds) in held.items():
        place = tmp_path / root / "exports" / f"family-{family}" / environment / "k0"
        place.mkdir(parents=True)
        for seed in seeds:
            (place / f"world_{seed}.npz").write_bytes(b"")
    return tmp_path


def test_the_split_reads_ranks_within_an_environment_and_robyn_s_worlds(tmp_path):
    development = _development(
        tmp_path,
        {
            "f0": (0, "drawn", range(100, 109)),
            "f0-robyn": (0, "drawn", range(100, 102)),
            "f0ref": (0, "reference", range(200, 206)),
        },
    )
    held = tuning.split(development)
    check = sorted(w for w, s in held.items() if s == "check")
    assert check == [
        (0, "drawn", 100),
        (0, "drawn", 101),
        (0, "drawn", 102),
        (0, "drawn", 105),
        (0, "drawn", 108),
        (0, "reference", 202),
        (0, "reference", 205),
    ]
    assert len(held) == 15


def test_the_tuning_fits_the_first_fit_worlds_of_each_national_environment(tmp_path, monkeypatch):
    development = _development(
        tmp_path,
        {"f0": (0, "drawn", range(100, 109)), "f0-robyn": (0, "drawn", range(100, 102))},
    )
    monkeypatch.setattr(tuning, "ROOTS", ("f0",))
    assert [j.seed for j in tuning.jobs(development, 3)] == [103, 104, 106]
    with pytest.raises(ValueError, match="holds 4 fit worlds, not 5"):
        tuning.jobs(development, 5)


def test_each_rival_s_tuning_fits_its_own_arm():
    meridian = tuning.command("meridian", "mroi knots-8").command
    assert meridian[meridian.index("python") :] == (
        "python",
        "scripts/meridian_arm.py",
        "--media-prior-type",
        "mroi",
        "--knot-weeks",
        "8",
        "--holdout",
        "0.1",
    )
    assert "scripts/meridian_arm.txt" in meridian
    pymc = tuning.command("pymc-marketing", "logistic varying").command
    assert pymc[pymc.index("python") :] == (
        "python",
        "scripts/pymc_marketing_cv.py",
        "--intercept",
        "varying",
    )
    assert "scripts/pymc_marketing_arm.txt" in pymc
    # past the cost cap: Hill with a varying intercept fits at 7.7 times the recommended settings
    assert list(GRIDS["pymc-marketing"]) == ["logistic static", "logistic varying", "hill static"]
    for rival, grid in GRIDS.items():
        assert next(iter(grid)) == RECOMMENDED[rival]
        assert grid[RECOMMENDED[rival]] == ()


def _meridian(passes, health, out):
    return {"failed": False, "pass": passes, "health": health, "out": out}


def _worlds(values):
    return {(0, "drawn", s): v for s, v in enumerate(values)}


def test_meridian_s_card_comes_first_then_its_fit_out_of_sample():
    records = {c: _worlds([_meridian(True, 100.0, 0.10)] * 4) for c in GRIDS["meridian"]}
    # better out of sample everywhere, but its card fails half its fits
    records["mroi aks"] = _worlds([_meridian(i % 2 == 0, 100.0, 0.01) for i in range(4)])
    # clearly better out of sample, card as good
    records["roi knots-8"] = _worlds([_meridian(True, 100.0, 0.05 + 0.001 * i) for i in range(4)])
    chosen = tuning.choose("meridian", records)
    assert chosen["candidate"] == "roi knots-8"
    assert chosen["lost"]["mroi aks"] == "the health card"
    assert chosen["lost"]["roi aks"].startswith("out of sample")


def test_a_tie_goes_to_the_recommended_then_to_fewer_settings_moved():
    records = {
        c: _worlds([_meridian(True, 99.0, 0.10 + 0.01 * (i % 2)) for i in range(6)])
        for c in GRIDS["meridian"]
    }
    assert tuning.choose("meridian", records)["candidate"] == "roi aks"
    records["roi aks"] = _worlds([_meridian(True, 50.0, 0.10)] * 6)
    chosen = tuning.choose("meridian", records)
    assert chosen["candidate"] in {"roi knots-8", "roi knots-13", "mroi aks", "contribution aks"}
    assert tuning._moved("meridian", chosen["candidate"]) == 1


def test_pymc_marketing_s_candidates_fail_out_or_tie_by_one_standard_error():
    def crps(values):
        return _worlds([{"failed": v is None, "mixed": v is not None, "out": v} for v in values])

    records = {
        "logistic static": crps([0.10, 0.12, 0.11, 0.13]),
        "logistic varying": crps([0.09, 0.11, 0.10, 0.12]),
        "hill static": crps([0.05, None, 0.05, 0.05]),
    }
    chosen = tuning.choose("pymc-marketing", records)
    assert chosen["candidate"] == "logistic varying"
    assert chosen["lost"]["hill static"].startswith("failed on more than")
    assert chosen["summary"]["logistic static"]["difference"] == pytest.approx(0.01, abs=1e-12)


def _cv(out, rhat=1.01, divergences=10, kept=2000, folds=4):
    """A cross-validation's record, every fold's sampler flags alike."""
    flags = {"max_rhat": rhat, "divergences": divergences, "kept_draws": kept}
    return {"error": None, "mean_test_crps_scaled": out, "folds": [{"flags": flags}] * folds}


def test_a_world_mixes_where_every_fold_is_within_the_r_hat_and_the_divergences():
    assert tuning._criterion("pymc-marketing", _cv(0.1))["mixed"]
    assert not tuning._criterion("pymc-marketing", _cv(0.1, rhat=1.11))["mixed"]
    assert tuning._criterion("pymc-marketing", _cv(0.1, divergences=200))["mixed"]
    assert not tuning._criterion("pymc-marketing", _cv(0.1, divergences=201))["mixed"]
    one = _cv(0.1) | {"folds": [*_cv(0.1)["folds"][:3], _cv(0.1, rhat=1.6)["folds"][0]]}
    assert not tuning._criterion("pymc-marketing", one)["mixed"]
    raised = {"error": "RuntimeError: x"}
    assert tuning._criterion("pymc-marketing", raised) == {
        "failed": True,
        "mixed": False,
        "out": None,
    }


def test_a_candidate_whose_sampler_does_not_mix_is_out_as_meridian_s_card_holds_one_out():
    def read(rhats, outs):
        return _worlds(
            [
                tuning._criterion("pymc-marketing", _cv(o, rhat=r))
                for r, o in zip(rhats, outs, strict=True)
            ]
        )

    worlds = 20
    good = [1.01] * worlds
    records = {
        # one world in twenty past R-hat 1.1, as the recommended settings' fits are
        "logistic static": read([1.2, *good[1:]], [0.10] * worlds),
        # forecasts best, but a fifth of its worlds do not mix
        "logistic varying": read([*[1.6] * 4, *good[4:]], [0.05] * worlds),
        # one world more than the best past R-hat: within the gate
        "hill static": read(
            [1.2, 1.2, *good[2:]], [0.10 + (0.001 if i % 2 else -0.001) for i in range(worlds)]
        ),
    }
    chosen = tuning.choose("pymc-marketing", records)
    assert chosen["lost"]["logistic varying"] == "the sampler's mixing"
    assert chosen["summary"]["logistic varying"]["mixed"] == pytest.approx(0.8, abs=0.0)
    assert chosen["candidate"] == "logistic static"
    assert "hill static" in chosen["tied"]
    assert "| mixed |" in tuning.markdown(chosen)


def test_a_candidate_without_records_stops_the_choice():
    records = {c: {} for c in GRIDS["pymc-marketing"]}
    with pytest.raises(ValueError, match="holds no record"):
        tuning.choose("pymc-marketing", records)


def _scopes() -> bool:
    if shutil.which("systemd-run") is None:
        return False
    command = ["systemd-run", "--user", "--scope", "-q", "true"]
    return subprocess.run(command, capture_output=True, check=False).returncode == 0


@pytest.mark.skipif(not _scopes(), reason="no user systemd to run a scope")
def test_the_tuning_fits_each_candidate_on_each_world_once(tmp_path, monkeypatch):
    from causaldyn_bench.scorecard.observe import export

    from scorecard.layout import FAMILIES

    development = tmp_path / "development"
    family = FAMILIES[0]
    for seed in (900, 901):
        place = development / "f0" / "exports" / "family-0" / "drawn" / "k0"
        export(family.observe(family.world("drawn", seed), 0), place)
    modes = tmp_path / "modes.json"
    modes.write_text(json.dumps({"900": "record", "901": "exit"}))
    fake = Tool((sys.executable, str(FAKE), "--modes", str(modes)))
    monkeypatch.setattr(tuning, "ROOTS", ("f0",))
    monkeypatch.setattr(tuning, "command", lambda rival, candidate: fake)
    root = tmp_path / "tuning"
    limits = Limits(memory="512M", timeout=60.0, slots=2)
    tuning.fit(root, "pymc-marketing", development, limits, per_environment=2, smoke=True)
    for candidate in GRIDS["pymc-marketing"]:
        made = [
            json.loads(tuning.record_path(root, "pymc-marketing", candidate, job).read_text())
            for job in tuning.jobs(development, 2)
        ]
        assert [r.get("failure_kind") for r in made] == [None, "exit"]
    starts = {
        s: len((tmp_path / "modes.starts" / s).read_text().splitlines()) for s in ("900", "901")
    }
    grid = len(GRIDS["pymc-marketing"])
    assert starts == {"900": grid, "901": grid * tuning.ATTEMPTS}
    tuning.fit(root, "pymc-marketing", development, limits, per_environment=2, smoke=True)
    again = {
        s: len((tmp_path / "modes.starts" / s).read_text().splitlines()) for s in ("900", "901")
    }
    assert again == starts


def test_select_writes_the_choice_the_tuned_arm_reads(tmp_path, monkeypatch):
    root = tmp_path / "tuning"
    for c in GRIDS["pymc-marketing"]:
        for seed, value in enumerate((0.10, 0.12, 0.11)):
            out = value - (0.05 if c == "hill static" else 0.0)
            path = tuning.record_path(root, "pymc-marketing", c, tuning.Job("f0", 0, "drawn", seed))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(_cv(out)))
    monkeypatch.setattr(tools, "TUNED", tmp_path / "tuned.json")
    monkeypatch.setattr(tuning, "TUNED", tmp_path / "tuned.json")
    chosen = tuning.select(root, "pymc-marketing", write=True)
    assert chosen["candidate"] == "hill static"
    assert (root / "tuning-pymc-marketing.md").read_text().startswith("## pymc-marketing")
    held = json.loads((tmp_path / "tuned.json").read_text())
    assert held["pymc-marketing"]["options"] == ["--saturation", "hill"]
    arm = tools.tuned("pymc-marketing", TOOLS["pymc-marketing"])
    assert arm.command == (*TOOLS["pymc-marketing"].command, "--saturation", "hill")
    assert arm.unset is None


def test_a_tuned_arm_the_tuning_has_not_chosen_is_refused_where_a_run_declares_it(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(tools, "TUNED", tmp_path / "absent.json")
    monkeypatch.setitem(TOOLS, "meridian tuned", tools.tuned("meridian", TOOLS["meridian"]))
    declared = Declaration(
        family=0,
        environments=("drawn",),
        worlds=1,
        pilot=True,
        rungs=(0,),
        arms=("ridge",),
        external=("meridian tuned",),
        limits={
            "ridge": Limits(memory="1G", timeout=60.0, slots=1),
            "meridian tuned": Limits(memory="8G", timeout=60.0, slots=1),
        },
    )
    with pytest.raises(ValueError, match="has chosen no configuration of meridian"):
        declare(Layout(tmp_path / "run"), declared)


def test_a_tuned_arm_the_tuning_chose_the_recommended_settings_for_is_refused(
    tmp_path, monkeypatch
):
    # its column would fit the recommended column again: the recommended column is the tuned one
    chosen = tmp_path / "tuned.json"
    chosen.write_text(json.dumps({"pymc-marketing": {"candidate": "logistic static"}}))
    monkeypatch.setattr(tools, "TUNED", chosen)
    assert GRIDS["pymc-marketing"]["logistic static"] == ()
    assert tools.recommended("pymc-marketing")
    assert not tools.recommended("meridian")
    arm = tools.tuned("pymc-marketing", TOOLS["pymc-marketing"])
    assert arm.command == ()
    monkeypatch.setitem(TOOLS, "pymc-marketing tuned", arm)
    declared = Declaration(
        family=0,
        environments=("drawn",),
        worlds=1,
        pilot=True,
        rungs=(0,),
        arms=("ridge",),
        external=("pymc-marketing tuned",),
        limits={
            "ridge": Limits(memory="1G", timeout=60.0, slots=1),
            "pymc-marketing tuned": Limits(memory="8G", timeout=60.0, slots=1),
        },
    )
    with pytest.raises(ValueError, match="recommended settings, logistic static"):
        declare(Layout(tmp_path / "run"), declared)
    chosen.write_text(json.dumps({"pymc-marketing": {"candidate": "hill static"}}))
    assert not tools.recommended("pymc-marketing")
    assert tools.tuned("pymc-marketing", TOOLS["pymc-marketing"]).unset is None
    assert TOOLS["robyn tuned"].command[-3:] == ("--ranges", "genre", "--ts-validation")
    assert TOOLS["robyn tuned"].r
