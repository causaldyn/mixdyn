"""The calibration loop as an arm: it reads the rung below's tests, values each test the rung adds
by `value_of_tests`, refits with the one of the largest reduction whatever `choose_tests` picks,
and plans again; Meridian's calibration score chooses beside it, and the plan the loop starts from
is a variant of its own. Each candidate is its row's test in a share the model reads alike, every
candidate priced at one cost. The score's parts hold Meridian's anchors and the default prior's
width ratio Maxima reads. The arm runs a world end to end under the harness and is scored.

The arm's two fits are a stand-in here (`Fake`): synthetic draws on the export's channels, each
call's rows kept, so that a world runs in seconds; the loop's own steps are the arm's."""

import dataclasses
import json
import math
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import optimize, stats

pytest.importorskip("causaldyn_bench")

from causaldyn_bench.scorecard.geo import GEO
from causaldyn_bench.scorecard.track_m2 import TRACK_M2
from fake_posterior import Fake

from mixdyn.experiments import predicted_lift
from scorecard import report, run
from scorecard.arms import bayes, voi
from scorecard.arms import panel as reading
from scorecard.layout import Declaration, Layout, Limits

LIMITS = Limits(memory="4G", timeout=600.0, slots=1)


@pytest.fixture(scope="module")
def world():
    return TRACK_M2.world("drawn", TRACK_M2.pilots["drawn"][0])


@pytest.fixture(scope="module")
def observed(world):
    return {k: TRACK_M2.observe(world, k) for k in (0, 1, 2)}


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    """A smoke run of family 0's first pilot world at rungs 1 and 2, the arm's fits the stand-in,
    scored and reported; the stand-in's calls at each rung, and the report's lines."""
    layout = Layout(tmp_path_factory.mktemp("voi") / "run")
    run.declare(
        layout,
        Declaration(
            family=0,
            environments=("drawn",),
            worlds=1,
            pilot=True,
            rungs=(1, 2),
            arms=("voi",),
            smoke=True,
            limits={"voi": LIMITS},
        ),
    )
    run.export_all(layout)
    (seed,) = layout.read().seeds("drawn")
    calls = {}
    with pytest.MonkeyPatch.context() as patch:
        for k in (1, 2):
            fake = Fake()
            patch.setitem(run.ARMS, "voi", lambda fake=fake: voi.Voi(fake))
            run.one(layout, "voi", "drawn", seed, (k,))
            calls[k] = fake.calls
    run.score(layout)
    with pytest.MonkeyPatch.context() as patch:
        shown = []
        patch.setattr("builtins.print", lambda *a, **_: shown.append(" ".join(map(str, a))))
        report.main(layout, list(voi.Voi.variants))
    held = {
        (variant, k): json.loads(layout.record(variant, 0, "drawn", k, seed).read_text())
        for variant in voi.Voi.variants
        for k in (1, 2)
    }
    scores = [json.loads(line) for line in layout.scores.read_text().splitlines()]
    return SimpleNamespace(records=held, scores=scores, calls=calls, shown="\n".join(shown))


def test_the_arm_runs_a_world_end_to_end_and_is_scored_and_reported(ran):
    """Each variant's record at each rung holds a plan and no error, the scorer reads its regret,
    and the report names every variant."""
    for (variant, k), record in ran.records.items():
        assert record["error"] is None, (variant, k, record.get("traceback"))
        assert record["k"] == k
    assert sorted(line["k"] for line in ran.scores) == [1, 2]
    for line in ran.scores:
        for variant in voi.Voi.variants:
            assert line["arms"][variant]["failure"] is None
            assert math.isfinite(line["arms"][variant]["regret"])
    for variant in voi.Voi.variants:
        assert f"  {variant:16s} regret" in ran.shown


def test_the_loop_fits_the_rung_below_then_each_chosen_test_beside_it(ran, world):
    """At each rung the first fit reads every row but each channel's earliest, the rung below's
    tests, and each refit those rows and one chosen candidate's."""
    for k, calls in ran.calls.items():
        lift = TRACK_M2.observe(world, k).lift
        rows = list(zip(lift.channel, lift.start, strict=True))
        earliest = {c: min(s for d, s in rows if d == c) for c in set(lift.channel)}
        new = {(c, s) for c, s in rows if s == earliest[c]}
        before, *after = calls
        assert set(before) == set(rows) - new
        assert 1 <= len(after) <= 2
        for refit in after:
            (added,) = set(refit) - set(before)
            assert added in new
            assert set(refit) == set(before) | {added}


def test_voi_refits_the_candidate_of_the_largest_reduction(ran):
    """The test `voi` reads is the candidate of the largest expected regret reduction, the record
    keeps every candidate's, and `voi no test` plans the posterior before, as `voi meridian` does
    where it chooses no test."""
    for k in (1, 2):
        record = ran.records[voi.VOI, k]
        flags, settings = record["flags"], record["settings"]
        reductions = {
            c: flags[f"{c} reduction"]
            for c in {label.split()[0] for label in settings["candidates"]}
        }
        best = max(reductions, key=reductions.__getitem__)
        assert settings["chosen"].split()[0] == best
        assert flags["chosen reduction"] == reductions[best]
        assert settings["chosen by"] == "the largest expected regret reduction"
        assert flags["choose_tests agrees"] == (
            settings["choose_tests"]["chosen"] == [settings["chosen"]]
        )
        start = ran.records[voi.NO_TEST, k]
        assert start["settings"]["chosen"] is None
        meridian = ran.records[voi.MERIDIAN, k]
        if meridian["settings"]["chosen"] is None:
            assert meridian["weekly"] == start["weekly"]
        assert record["weekly"] != start["weekly"]


def test_a_candidate_is_its_rows_test_at_a_share_the_model_reads_alike(observed):
    """Each candidate's predicted lift at the market's scale and its error over its share are its
    row's whole-market test's, to rounding, and every candidate's gross cost, its share times its
    unit cost, is the one cost, half the cheapest's whole-market go-dark."""
    observation = observed[2]
    every = Fake()(observation, bayes.SMOKE).every
    new = voi.newest(observation)
    made, cost = voi.candidates(observation, new, every)
    assert cost > 0.0
    units = []
    for row, candidate in zip(new, made, strict=True):
        ((estimate,),) = reading.estimates(voi.rows(observation, [row])).values()
        whole = dataclasses.replace(
            candidate, share=1.0, spend=estimate.spend, planned=estimate.planned, se=estimate.se
        )
        assert candidate.label == voi.label(observation, row)
        assert 0.0 < candidate.share <= voi.SHARE
        np.testing.assert_allclose(
            predicted_lift(every, candidate), predicted_lift(every, whole), rtol=1e-12, atol=0.0
        )
        assert candidate.se / candidate.share == pytest.approx(estimate.se, rel=1e-12, abs=0.0)
        unit = voi.unit_cost(every, whole)
        units.append(unit)
        assert candidate.share * unit == pytest.approx(cost, rel=1e-12, abs=0.0)
    assert cost == pytest.approx(voi.SHARE * min(units), rel=1e-12, abs=0.0)


def _valued(reductions, picked):
    """A stand-in for the arm's valuation: the candidates' `reductions`, and `choose_tests`'
    set the candidate at `picked`."""

    def valued(observation, made, cost, every, *, smoke):
        values = [
            SimpleNamespace(
                reduction=r,
                error=0.1,
                predicted=r,
                interval=(r - 1.0, r + 1.0),
                fallback=0.0,
                net_value=r - cost,
            )
            for r in reductions
        ]
        best = SimpleNamespace(candidates=[made[picked]])
        return (
            SimpleNamespace(values=values, evpi=max(reductions), evpi_error=0.1),
            SimpleNamespace(best=best, verdict="test"),
        )

    return valued


@pytest.mark.parametrize("reductions", [(1.0, 5.0, 3.0), (7.0, 2.0, 7.0), (0.0, -1.0, 4.0)])
def test_the_choice_is_the_largest_reduction_whatever_choose_tests_picks(
    observed, monkeypatch, reductions
):
    """`choose_tests` picks the candidate of the least reduction; `voi` refits the largest, the
    first of a tie, and says the two disagree."""
    observation = observed[2]
    new = voi.newest(observation)
    assert len(new) == len(reductions)
    monkeypatch.setattr(voi, "_valued", _valued(reductions, int(np.argmin(reductions))))
    fit = voi.Voi(Fake()).fit(observation, smoke=True)
    assert fit.chosen[voi.VOI] == new[int(np.argmax(reductions))]
    assert fit.chosen[voi.NO_TEST] is None
    assert fit.flags["choose_tests agrees"] is False
    assert set(fit.after) == {r for r in fit.chosen.values() if r is not None}


def test_voi_meridian_refits_the_channel_its_score_recommends(observed, monkeypatch):
    """A channel scored below 67.5, the lowest, is the one `voi meridian` refits with its newest
    test; none below, and it plans the posterior before."""
    observation = observed[2]
    new = voi.newest(observation)
    channels = observation.channels
    monkeypatch.setattr(voi, "_valued", _valued((1.0,) * len(new), 0))
    for scores, named in (((90.0, 60.0, 50.0), channels[2]), ((90.0, 80.0, 70.0), None)):
        monkeypatch.setattr(
            voi, "calibration_scores", lambda *_, s=scores: [{"score": x} for x in s]
        )
        fit = voi.Voi(Fake()).fit(observation, smoke=True)
        row = fit.chosen[voi.MERIDIAN]
        if named is None:
            assert row is None
            assert fit.why[voi.MERIDIAN] == "no test: no channel scores below 67.5"
        else:
            assert row in new
            assert observation.lift.channel[row] == named
            assert row in fit.after


def test_rung_0_plans_the_posterior_before_in_every_variant(observed):
    """With no test to choose, the loop fits once and every variant says why it read none."""
    fake = Fake()
    fit = voi.Voi(fake).fit(observed[0], smoke=True)
    assert fake.calls == [()]
    assert fit.after == {}
    assert set(fit.chosen.values()) == {None}
    assert set(fit.why.values()) == {"no test: the rung holds none to choose"}


def test_the_arm_refuses_a_geo_exports_market():
    market = reading.market(GEO.observe(GEO.world("policy-0", GEO.pilots["policy-0"][0]), 4))
    with pytest.raises(ValueError, match="national export"):
        voi.Voi(Fake()).fit(market, smoke=True)


def test_meridians_scores_hold_their_anchors():
    """The implausible-ROI bowl is 100 at the bounds' geometric mean and 50 at either bound at a
    share of 1, wider at a smaller share; the variance score 100 to 0.5 and 50 at 1; the bias
    score 0 at no correlation, 50 at 0.1 and 100 at 1."""
    centre = math.sqrt(0.5 * 20.0)
    assert voi.center_bowl(centre, 1.0) == pytest.approx(100.0, rel=1e-12, abs=0.0)
    for bound in voi.ROI_BOUNDS:
        assert voi.center_bowl(bound, 1.0) == pytest.approx(50.0, rel=1e-12, abs=0.0)
        assert voi.center_bowl(bound, 0.2) > 50.0
    assert voi.half_bowl(0.5 / 0.4, 0.4) == 100.0
    assert voi.half_bowl(1.0 / 0.4, 0.4) == pytest.approx(50.0, rel=1e-12, abs=0.0)
    assert voi.half_bowl(4.0, 0.5) < 50.0
    assert voi.bias_score(0.0) == 0.0
    assert voi.bias_score(0.1) == pytest.approx(50.0, rel=1e-12, abs=0.0)
    assert voi.bias_score(1.0) == pytest.approx(100.0, rel=1e-12, abs=0.0)


def test_a_score_is_its_parts_weighed_and_the_lowest_under_the_bar_is_recommended():
    """On draws of known spread: the width ratio is the 80 % narrowest interval's width over the
    median over the prior's, the bias reads the largest correlation with a control, and the score
    is half the first part and a quarter each of the others."""
    rng = np.random.default_rng(20261107)
    weeks = 104
    spend = np.column_stack([rng.uniform(1.0, 2.0, weeks), rng.uniform(0.0, 1.0, weeks)])
    controls = np.column_stack([spend[:, 0] + rng.normal(0.0, 0.1, weeks), rng.normal(size=weeks)])
    roi = np.column_stack([rng.lognormal(0.5, 0.2, 2000), rng.lognormal(3.5, 1.0, 2000)])
    scores = voi.calibration_scores(roi, spend, controls)
    for c, score in enumerate(scores):
        low, high = voi.hdi(roi[:, c], voi.HDI)
        width = (high - low) / np.median(roi[:, c]) / voi.PRIOR_WIDTH
        assert score["width ratio"] == pytest.approx(width, rel=1e-12, abs=0.0)
        correlation = max(abs(np.corrcoef(spend[:, c], x)[0, 1]) for x in controls.T)
        assert score["correlation"] == pytest.approx(correlation, rel=1e-12, abs=0.0)
        parts = (score["implausible roi"], score["high variance"], score["potential bias"])
        assert score["score"] == pytest.approx(
            0.5 * parts[0] + 0.25 * parts[1] + 0.25 * parts[2], rel=1e-12, abs=0.0
        )
    assert scores[0]["share"] + scores[1]["share"] == pytest.approx(1.0, rel=1e-12, abs=0.0)
    assert voi.recommended([{"score": 90.0}, {"score": 60.0}, {"score": 40.0}]) == 2
    assert voi.recommended([{"score": 60.0}, {"score": 60.0}]) == 0
    assert voi.recommended([{"score": 67.5}, {"score": math.nan}]) is None
    assert voi.calibration_scores(roi, spend, np.zeros((weeks, 0)))[0]["potential bias"] == 0.0


def test_the_narrowest_interval_is_the_narrowest_run_of_draws():
    values = np.array([0.0, 1.0, 1.1, 1.2, 1.3, 5.0, 9.0, 9.5, 20.0, 30.0])
    assert voi.hdi(values, 0.3) == (1.0, 1.3)
    assert voi.hdi(values, 0.95) == (0.0, 30.0)


def test_the_default_priors_width_ratio_is_the_narrowest_80_interval_over_the_median():
    """By SciPy: LogNormal(0.2, 0.9)'s narrowest interval of mass 0.8 over its median is the
    arm's 2.0687, to its four decimals."""
    prior = stats.lognorm(s=0.9, scale=math.exp(0.2))
    found = optimize.minimize_scalar(
        lambda p: prior.ppf(p + 0.8) - prior.ppf(p),
        bounds=(1e-9, 0.2 - 1e-9),
        method="bounded",
        options={"xatol": 1e-12},
    )
    width = (prior.ppf(found.x + 0.8) - prior.ppf(found.x)) / prior.median()
    assert width == pytest.approx(voi.PRIOR_WIDTH, rel=5e-5 / 2.0687, abs=0.0)


@pytest.mark.skipif(shutil.which("maxima") is None, reason="maxima is not on PATH")
def test_maxima_proves_the_scores_width_ratio_and_anchors_at_every_step():
    script = Path(__file__).parents[2] / "validation" / "meridian_calibration_score.mac"
    out = subprocess.run(
        ["maxima", "--very-quiet", "-b", str(script)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    ).stdout
    assert "Lisp error" not in out
    verdicts = [line.strip() for line in out.splitlines() if line.startswith("STEP ")]
    assert len(verdicts) == 11
    assert all(line.endswith("(PASS)") for line in verdicts)
    assert {line.split()[1] for line in verdicts} == {str(k) for k in range(1, 5)}
