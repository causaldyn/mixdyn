"""A scored run's worlds as the next run's development set: the split, the pilot's worlds read
against the stored run's by export digest, the selection rule and its predictive."""

import dataclasses
import json

import numpy as np
import pytest

pytest.importorskip("causaldyn_bench")

from causaldyn_bench.scorecard.scoring import WorldRecord
from test_claim import _score

from scorecard import development
from scorecard.development import (
    CHECK,
    FIT,
    TIE,
    Candidate,
    Development,
    Scored,
    joined,
    predictive,
    score,
    select,
    split,
)
from scorecard.layout import Declaration, Layout

BUILT_IN = ("status quo", "equal split")


def test_the_check_split_is_every_third_rank_from_the_third_and_every_world_robyn_planned():
    seeds = [507, 501, 503, 502, 504, 506, 505, 500]
    held = split(seeds, robyn={500})
    ranked = sorted(seeds)
    assert [held[s] for s in ranked] == [CHECK, FIT, CHECK, FIT, FIT, CHECK, FIT, FIT]
    assert split(seeds) == {s: CHECK if i % 3 == 2 else FIT for i, s in enumerate(ranked)}


def test_a_split_refuses_a_seed_held_twice_or_robyn_s_world_the_run_does_not_hold():
    with pytest.raises(ValueError, match="held once"):
        split([1, 1, 2])
    with pytest.raises(ValueError, match="held once"):
        split([1, 2], robyn={3})


def _declaration(family, environments, worlds, **change):
    held = Declaration(
        family=family,
        environments=environments,
        worlds=worlds,
        pilot=False,
        bench="316fada",
        rungs=(0, 4),
        arms=("dlm",),
        external=("orbit",),
        default="dlm plug-in",
        margin=0.05,
    )
    return dataclasses.replace(held, **change)


def test_the_development_set_ranks_each_environment_and_holds_robyn_s_worlds_out(tmp_path):
    main = _declaration(2, ("kappa-0.5", "kappa-1"), 9)
    robyn = Declaration(
        family=2,
        environments=("kappa-0.5", "kappa-1"),
        worlds=2,
        pilot=False,
        bench="316fada",
        rungs=(0, 4),
        arms=("chc",),
        external=("robyn",),
        caps={"robyn": 2},
        exploratory_caps=True,
    )
    Layout(tmp_path / "f2").declare(main)
    Layout(tmp_path / "f2-robyn").declare(robyn)
    held = development.development([tmp_path / "f2", tmp_path / "f2-robyn"])
    for environment in main.environments:
        seeds = list(main.seeds(environment))
        parts = [held.of(2, environment, s) for s in seeds]
        assert parts == [CHECK, CHECK, CHECK, FIT, FIT, CHECK, FIT, FIT, CHECK]
    with pytest.raises(ValueError, match="not a stored scored world"):
        held.of(2, "kappa-2", main.seeds("kappa-0.5")[0])
    # the rule is the rank's, with Robyn's worlds added: without them the first two are fit
    alone = development.development([tmp_path / "f2"])
    assert alone.of(2, "kappa-1", main.seeds("kappa-1")[0]) == FIT


def _world(family, environment, seed, index, k, regrets):
    return WorldRecord(
        family=family,
        environment=environment,
        seed=seed,
        index=index,
        pilot=False,
        k=k,
        budget=1000.0,
        best=2000.0,
        channels=3,
        oracle_bounds=0,
        covariates={},
        arms={arm: _score(regret) for arm, regret in regrets.items()},
    )


def _write(root, declaration, records):
    layout = Layout(root)
    layout.declare(declaration)
    layout.scores.write_text("".join(json.dumps(r.as_json()) + "\n" for r in records))
    for r in records:
        path = layout.export(r.family, r.environment, r.k, r.seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{r.environment}/{r.seed}/{r.k}")
    return layout


@pytest.fixture
def digests(monkeypatch):
    """Each export's digest, the text the test writes there in place of an archive."""
    monkeypatch.setattr(development, "_digest", lambda path: path.read_text())


def _stored_and_pilot(tmp_path, *, pilot_regret=0.1):
    d = _declaration(3, ("r-0.9",), 4)
    seeds = list(d.seeds("r-0.9"))
    stored = [
        _world(3, "r-0.9", s, i, k, {**dict.fromkeys(BUILT_IN, 0.3), "orbit": 0.2})
        for i, s in enumerate(seeds)
        for k in (0, 4)
    ]
    mine = [
        dataclasses.replace(
            r, arms={**{a: r.arms[a] for a in BUILT_IN}, "new arm": _score(pilot_regret)}
        )
        for r in stored
        if r.k == 0
    ]
    _write(tmp_path / "f3", d, stored)
    pilot = dataclasses.replace(d, rungs=(0,), default=None, margin=None)
    return _write(tmp_path / "p3", pilot, mine), stored


def test_the_pilot_s_arms_are_joined_to_the_stored_world_of_its_place(tmp_path, digests):
    pilot, stored = _stored_and_pilot(tmp_path)
    records = joined([tmp_path / "f3"], [pilot.root])
    assert len(records) == len(stored)
    at_0 = [r for r in records.values() if r.k == 0]
    assert all(r.arms["new arm"].regret == 0.1 and "orbit" in r.arms for r in at_0)
    assert not any("new arm" in r.arms for r in records.values() if r.k == 4)
    # with no pilot, the stored worlds as they are
    assert joined([tmp_path / "f3"], []) == {
        (r.family, r.environment, r.seed, r.k): r for r in stored
    }


def test_a_pilot_world_whose_export_is_another_observation_is_refused(tmp_path, digests):
    pilot, stored = _stored_and_pilot(tmp_path)
    r = stored[0]
    pilot.export(r.family, r.environment, r.k, r.seed).write_text("another observation")
    with pytest.raises(ValueError, match="not the stored run's observation"):
        joined([tmp_path / "f3"], [pilot.root])


def test_a_pilot_world_whose_built_in_arms_score_apart_is_refused(tmp_path, digests):
    pilot, _ = _stored_and_pilot(tmp_path)
    lines = pilot.scores.read_text().splitlines()
    first = json.loads(lines[0])
    first["arms"]["status quo"]["regret"] = 0.31
    pilot.scores.write_text("\n".join([json.dumps(first), *lines[1:]]) + "\n")
    with pytest.raises(ValueError, match="apart from the stored run"):
        joined([tmp_path / "f3"], [pilot.root])


def test_a_family_the_stored_run_never_held_is_read_on_its_pilot_worlds_all_checked(tmp_path):
    """Family 12's pilot root joins no stored world: its worlds are read as they are, for the
    winner's predictive alone; a root of a family the stored run held is refused."""
    d = _declaration(12, ("priors",), 4, pilot=True)
    records = [
        _world(12, "priors", s, i, 0, {**dict.fromkeys(BUILT_IN, 0.3), "orbit": 0.2, "a": 0.1})
        for i, s in enumerate(d.seeds("priors"))
    ]
    _write(tmp_path / "f12", d, records)
    held = development.fresh([tmp_path / "f12"])
    assert held == {(12, "priors", r.seed, 0): r for r in records}
    checked = Development({(f, e, s): CHECK for f, e, s, _ in held})
    (stratum,) = development.check_strata(held, checked, "a", 12, "0", ["orbit"])
    np.testing.assert_allclose(stratum, [[-0.1]] * 4, rtol=0.0, atol=1e-12)
    _write(tmp_path / "f3", _declaration(3, ("r-0.9",), 4), [])
    with pytest.raises(ValueError, match="not one the stored run never held"):
        development.fresh([tmp_path / "f3"])


def test_a_capped_exploratory_tool_beside_a_fresh_run_is_no_rival_of_the_predictive(tmp_path):
    """Family 12's pilot root holds 6 worlds and its rival `orbit`; a root beside it holds a
    capped tool, declared exploratory, on 2 of them. The claim prints that tool's comparison and
    leaves it out, and so does the predictive: its rivals, the other rivals' strata and the lines
    `select` prints are those of the pilot root alone. A tool beside the run that is not declared
    exploratory is a rival, as it is in the claim."""
    d = _declaration(12, ("priors",), 6, pilot=True)
    seeds = list(d.seeds("priors"))
    regrets = {**dict.fromkeys(BUILT_IN, 0.3), "orbit": 0.2}
    records = [
        _world(12, "priors", s, i, 0, {**regrets, "a": 0.1 + 0.03 * i}) for i, s in enumerate(seeds)
    ]
    _write(tmp_path / "f12", d, records)
    beside = Declaration(
        family=12,
        environments=("priors",),
        worlds=2,
        pilot=True,
        bench="316fada",
        rungs=(0, 4),
        arms=d.arms,
        external=("robyn",),
        caps={"robyn": 2},
        exploratory_caps=True,
    )
    capped = [
        _world(12, "priors", s, i, 0, {**dict.fromkeys(BUILT_IN, 0.3), "robyn": 0.5})
        for i, s in enumerate(seeds[:2])
    ]
    _write(tmp_path / "f12-robyn", beside, capped)
    alone, both = [tmp_path / "f12"], [tmp_path / "f12", tmp_path / "f12-robyn"]
    rivals = development.fresh_rivals(alone)
    assert rivals == {12: (*BUILT_IN, "orbit")}
    assert development.fresh_rivals(both) == rivals
    held = development.fresh(both)
    assert sum("robyn" in r.arms for r in held.values()) == 2
    checked = Development({(f, e, s): CHECK for f, e, s, _ in held})

    def lines(records, names):
        strata = development.check_strata(records, checked, "a", 12, "0", names)
        return strata, development._predicted("a", 12, "0", names, strata, 30, resamples=200)

    (stratum,), printed = lines(held, rivals[12])
    (expected,), expected_lines = lines(development.fresh(alone), rivals[12])
    assert stratum.shape == (6, 3)
    np.testing.assert_array_equal(stratum, expected)
    assert printed == expected_lines
    assert printed[0].startswith("  predictive of a, family 12 at rung 0, 30 worlds")
    # every arm with a record as a rival read the two worlds the capped tool planned, no more
    (cut,), _ = lines(held, [*rivals[12], "robyn"])
    assert cut.shape == (2, 4)
    # declared without the exploratory caps, the tool is in the claim and in the predictive
    claimed = [tmp_path / "claimed" / "f12", tmp_path / "claimed" / "f12-robyn"]
    _write(claimed[0], d, records)
    _write(claimed[1], dataclasses.replace(beside, exploratory_caps=False), capped)
    assert development.fresh_rivals(claimed) == {12: (*BUILT_IN, "orbit", "robyn")}


def _records(arms_by_family, worlds=6):
    """Worlds of each family at rung 0 and 4, each arm's regret its constant plus the world's
    place, so that every difference of two arms is the difference of their constants."""
    out = {}
    for family, arms in arms_by_family.items():
        for i in range(worlds):
            for k in (0, 4):
                regrets = {a: c + 0.01 * i for a, c in arms.items()}
                r = _world(family, "e", 1000 * family + i, i, k, regrets)
                out[family, "e", r.seed, k] = r
    return out


def _everything(records, part=FIT):
    return Development({(f, e, s): part for f, e, s, _ in records})


def test_a_candidate_s_score_is_its_worst_mean_difference_over_families_and_rivals():
    records = _records(
        {
            0: {"status quo": 0.3, "equal split": 0.2, "orbit": 0.25, "a": 0.21, "b": 0.24},
            2: {"status quo": 0.3, "equal split": 0.2, "orbit": 0.15, "a": 0.21, "b": 0.16},
        }
    )
    scored = {
        name: score(records, _everything(records), Candidate(name, "0"), {0: None, 2: None})
        for name in ("a", "b")
    }
    assert scored["a"].score == pytest.approx(0.06, abs=1e-12)
    assert scored["a"].binding == (2, "orbit")
    assert scored["b"].score == pytest.approx(0.04, abs=1e-12)
    assert scored["b"].binding == (0, "equal split")
    assert scored["b"].d[0, "status quo"] == (pytest.approx(-0.06, abs=1e-12), 6)
    winner, tied = select(list(scored.values()))
    assert winner.candidate.name == "b"
    assert tied == (winner,)
    # the check split is never read by the rule
    with pytest.raises(ValueError, match="no fit-split world"):
        score(records, _everything(records, CHECK), Candidate("a", "0"), {0: None})


def _failing(records, arm, worlds):
    """`records` with `arm`'s fit failed on the worlds of `worlds`, each playing the status quo
    there, as the bench scores a failure."""
    out = {}
    for place, r in records.items():
        if r.index in worlds:
            failed = dataclasses.replace(
                r.arms["status quo"], failure="the fit raised", record={"error": "the fit raised"}
            )
            r = dataclasses.replace(r, arms={**r.arms, arm: failed})
        out[place] = r
    return out


def test_a_failed_candidate_fit_plays_the_status_quo_and_is_counted_in_the_rule():
    """`flaky` plans 0.01 below `steady` on the worlds it plans and fails on the other half, which
    play the status quo, 0.1 above: on the worlds both planned it would win, counted it loses."""
    records = _records(
        {0: {"status quo": 0.3, "equal split": 0.2, "orbit": 0.25, "steady": 0.21, "flaky": 0.2}}
    )
    records = _failing(records, "flaky", {1, 3, 5})
    dev = _everything(records)
    flaky = score(records, dev, Candidate("flaky", "0"), {0: None})
    steady = score(records, dev, Candidate("steady", "0"), {0: None})
    # (3 * (0.2 - 0.25) + 3 * (0.3 - 0.25)) / 6 = 0, against orbit, on every world
    assert flaky.d[0, "orbit"] == (pytest.approx(0.0, abs=1e-12), 6)
    assert flaky.failed == {0: (3, 6)}
    assert steady.d[0, "orbit"] == (pytest.approx(-0.04, abs=1e-12), 6)
    assert steady.failed == {0: (0, 6)}
    winner, _ = select([flaky, steady])
    assert winner.candidate.name == "steady"
    assert "    family  0: failed on 3 of 6 fit worlds, each played as the status quo" in (
        development._lines("0", [flaky, steady], winner, (winner,))
    )


def test_the_predictive_reads_each_comparison_on_its_complete_cases_as_the_claim_does():
    """A check world where the candidate failed is no comparison's complete case, and one where a
    rival failed is not that rival's: each is drawn and left out. Far inside the margin on its
    complete cases, the candidate is predicted within, though its failures at the status quo's
    regret would put it far outside; on fewer than two complete cases, never."""
    records = _records({0: {"status quo": 0.9, "equal split": 0.2, "orbit": 0.25, "flaky": 0.1}})
    records = _failing(records, "flaky", {0, 2})
    records |= _failing({p: r for p, r in records.items() if r.index == 1}, "orbit", {1})
    (stratum,) = development.check_strata(
        records, _everything(records, CHECK), "flaky", 0, "0", ["equal split", "orbit"]
    )
    assert stratum.shape == (6, 2)
    assert np.isnan(stratum[[0, 2]]).all()
    assert np.isnan(stratum[1, 1])
    np.testing.assert_allclose(stratum[1, 0], -0.1, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(stratum[3:], [[-0.1, -0.15]] * 3, rtol=0.0, atol=1e-12)

    x = np.random.default_rng(2).normal(-0.1, 0.05, (40, 2))
    x[0::2] = np.nan
    x[1, 1] = np.nan
    each, joint = predictive([x], 40, resamples=200)
    assert each == (1.0, 1.0)
    assert joint == 1.0
    counted = np.where(np.isnan(x), 0.7, x)
    assert predictive([counted], 40, resamples=200) == ((0.0, 0.0), 0.0)
    x[3:, 1] = np.nan
    assert predictive([x], 40, resamples=200)[0] == (1.0, 0.0)


def test_the_s_curve_guard_puts_out_a_tested_candidate_that_loses_to_stacked_on_a_hill_family():
    """`sharp` has the best score on the claim's family but loses 0.03 to `stacked` on family 14:
    out, though it is fine on family 4; `mild` loses 0.019 there, inside the guard: kept."""
    records = _records(
        {
            0: {"status quo": 0.3, "equal split": 0.2, "stacked": 0.22, "sharp": 0.18, "mild": 0.2},
            4: {"status quo": 0.3, "equal split": 0.2, "stacked": 0.1, "sharp": 0.1, "mild": 0.1},
            14: {
                "status quo": 0.3,
                "equal split": 0.2,
                "stacked": 0.1,
                "sharp": 0.13,
                "mild": 0.119,
            },
        }
    )
    dev = _everything(records)
    scored = [score(records, dev, Candidate(n, "tested"), {0: None}) for n in ("sharp", "mild")]
    assert scored[0].score < scored[1].score
    excess = development.guard(records, dev, "sharp")
    assert excess[4] == (pytest.approx(0.0, abs=1e-12), 6)
    assert excess[14] == (pytest.approx(0.03, abs=1e-12), 6)
    kept, lines = development.screen(records, dev, scored)
    assert [s.candidate.name for s in kept] == ["mild"]
    assert lines[0].endswith("; out (above 0.02 on family 14)")
    assert "out" not in lines[1].rsplit(";", 1)[-1]
    # the check split is never read by the guard
    with pytest.raises(ValueError, match="no fit-split world of family 4"):
        development.guard(records, _everything(records, CHECK), "sharp")


def test_the_pairing_reads_the_variance_common_random_numbers_take_off_a_comparison():
    """`with` moves with the rival from world to world, so their difference does not move: the
    paired variance is none of the unpaired; `against` moves against it, twice the unpaired."""
    records = {}
    for i in range(6):
        regrets = {
            "status quo": 0.5,
            "equal split": 0.25,
            "orbit": 0.2 + 0.01 * i,
            "with": 0.1 + 0.01 * i,
            "against": 0.1 - 0.01 * i,
        }
        r = _world(0, "e", i, i, 0, regrets)
        records[0, "e", i, 0] = r
    dev = _everything(records)
    ((ratio, n),) = development.pairing(records, dev, "with", 0, "0", ["orbit"]).values()
    assert n == 6
    assert ratio == pytest.approx(0.0, abs=1e-12)
    ((ratio, _),) = development.pairing(records, dev, "against", 0, "0", ["orbit"]).values()
    assert ratio == pytest.approx(2.0, abs=1e-12)
    # a rival that does not move: pairing takes nothing off; neither moves: no ratio
    ((ratio, _),) = development.pairing(records, dev, "with", 0, "0", ["equal split"]).values()
    assert ratio == pytest.approx(1.0, abs=1e-12)
    ((ratio, _),) = development.pairing(
        records, dev, "status quo", 0, "0", ["equal split"]
    ).values()
    assert np.isnan(ratio)
    # the check split is not read
    unread = development.pairing(records, _everything(records, CHECK), "with", 0, "0", ["orbit"])
    assert unread["orbit"][1] == 0


def _scored(name, value, cpu, settings=0):
    return Scored(Candidate(name, "0", settings), {}, value, (0, "orbit"), cpu)


def test_a_tie_within_the_tie_goes_to_the_fewest_cpu_seconds_then_the_fewest_settings():
    least = _scored("least", 0.010, 900.0)
    near = _scored("near", 0.010 + TIE, 300.0, settings=2)
    nearer = _scored("nearer", 0.012, 300.0, settings=1)
    far = _scored("far", 0.010 + TIE + 1e-9, 10.0)
    winner, tied = select([least, near, nearer, far])
    assert winner is nearer
    assert tied == (nearer, near, least)
    assert select([least, far])[0] is least


def test_the_predictive_is_one_far_inside_the_margin_and_none_far_outside():
    rng = np.random.default_rng(0)
    inside = [rng.normal(-0.1, 0.05, (40, 2)) for _ in range(2)]
    each, joint = predictive(inside, 100, resamples=200)
    assert each == (1.0, 1.0)
    assert joint == 1.0
    outside = [rng.normal(0.2, 0.05, (40, 1))]
    assert predictive(outside, 100, resamples=200) == ((0.0,), 0.0)


def test_the_predictive_carries_the_pilot_mean_s_error_and_reads_comparisons_together():
    """At a check mean on the margin, a scored run of any size reads within about half the time:
    the first stage moves the mean as its own error does; and the comparisons together read within
    no more often than the least of them."""
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 0.1, (60, 2))
    x = x - x.mean(axis=0) + np.array([0.05, -0.05])
    each, joint = predictive([x], 2000, resamples=1000)
    assert 0.3 < each[0] < 0.7
    assert each[1] == 1.0
    assert joint <= min(each)
