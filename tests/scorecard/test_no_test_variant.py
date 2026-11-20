"""The no-test variants: each arm that holds more than one reading of its fit plans the worst tenth
of them against the bench's equal split, so its split never does worse than the equal split there.
The equal split is the one the bench scores, rebuilt from the export. The variant follows the
readings: it is the equal split where they disagree about every move, and the plan for their mean
where they coincide."""

import dataclasses
import json
import math
from types import MappingProxyType

import numpy as np
import pytest

pytest.importorskip("causaldyn_bench")

from causaldyn_bench.scorecard.drift import DRIFT
from causaldyn_bench.scorecard.observe import read
from causaldyn_bench.scorecard.scoring import score_world
from causaldyn_bench.scorecard.track_m2 import TRACK_M2
from chc import response
from chc.allocation import allocate, cvar_allocate

from mixdyn import NATIONAL
from mixdyn.models import DlmChoice, DlmSpec, Effect, Season, Trend, fit_dlm
from mixdyn.transforms import TanhShape, Transform
from scorecard import run
from scorecard.arms import bayes, chc_arm, dlm
from scorecard.arms import panel as reading
from scorecard.arms.base import _fields, records
from scorecard.layout import Declaration, Layout, Limits, slug

SEED = 905
ARMS = {
    "bayes": (bayes.Bayes(), "cvar 0.1 vs equal", bayes),
    "dlm": (dlm.Dlm(), "dlm cvar 0.1 vs equal", dlm),
    "dlm untested": (dlm.DlmUntested(), "dlm untested cvar 0.1 vs equal", dlm),
    "chc": (chc_arm.Chc(), "chc cvar 0.1 vs equal", chc_arm),
}
RESPONDS = {
    "bayes": "stacked",
    "dlm": "dlm mixture",
    "dlm untested": "dlm untested mixture",
    "chc": "chc",
}
"""The variant of each arm whose record carries the fit's response."""
COMMON = {"version", "seed", "digest", "family", "environment", "k", "python", "versions"}
PLANNED = {"weekly", "flags", "settings", "error", "gain", "gain_interval", "n/a"}
DISAGREE = (((3.0, 1.0, 1.0),) * 14) + (((1.0, 2.0, 1.0),) * 3) + (((1.0, 1.0, 2.0),) * 3)
"""Coefficients of 20 readings of three like channels: under each, one channel returns more a euro
than the others, and each channel is the one in at least a tenth of them."""


@pytest.fixture(scope="module")
def observation():
    return TRACK_M2.observe(TRACK_M2.world("drawn", SEED), 0)


@pytest.fixture(scope="module")
def fitted(observation):
    """Each arm's fit of the world: the static model's and CHC's smoke fits, and the DLM's readings
    of one spec, a `tanh` on every channel, drawn from 4 chains of 100, which both DLM arms read."""
    data = reading.panel(observation)
    spend = np.asarray(observation.spend, dtype=float)
    media = [
        Effect(
            name,
            0.99,
            sign="positive",
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
        effects=(*media, *(Effect(c, 0.99) for c in observation.controls)),
        trend=Trend(1, 0.99),
        season=Season(dlm.YEAR, dlm.HARMONICS, 1.0),
    )
    fit = fit_dlm(data, NATIONAL, spec, draws=100, seed=0)
    choice = DlmChoice((spec,), (fit.filtered,), 0, MappingProxyType({0: fit}))
    ess = dlm._ess([fit])
    flags = dlm._flags(choice, observation.channels)
    readings = choice.readings(dlm.DRAWS, observation.seed)
    one = dlm.Fitted(choice, readings, None, flags, {}, {}, ess, ess)
    return {
        "bayes": bayes.Bayes().fit(observation, smoke=True),
        "dlm": one,
        "dlm untested": one,
        "chc": chc_arm.Chc().fit(observation, smoke=True),
    }


@pytest.fixture(scope="module")
def planned(fitted, observation):
    """Each arm's no-test variant's plan of the world."""
    return {
        name: arm.plan(fitted[name], variant, observation)
        for name, (arm, variant, _) in ARMS.items()
    }


def _box(observation, weekly):
    weekly = np.asarray(weekly, dtype=float)
    assert np.all(weekly >= observation.lower * (1.0 - 1e-9))
    assert np.all(weekly <= observation.upper * (1.0 + 1e-9))
    assert weekly.sum() * observation.planned == pytest.approx(
        observation.budget, rel=1e-9, abs=0.0
    )


@pytest.mark.parametrize(
    ("family", "environment", "seed"),
    [(TRACK_M2, "drawn", SEED), (TRACK_M2, "reference", 901), (DRIFT, "walk", 220000)],
)
def test_the_equal_split_is_the_one_the_bench_scores_beside_every_arm(family, environment, seed):
    """Rebuilt from the export, it is the bench's to the bit, and a record that plays it is scored
    at the equal split's regret."""
    world = family.world(environment, seed)
    observation = family.observe(world, 0)
    equal = reading.equal_split(observation)
    np.testing.assert_array_equal(equal, family.truth(world).quarter.equal_split())
    _box(observation, equal)
    record = {
        "version": observation.version,
        "digest": observation.digest(),
        "error": None,
        "weekly": equal.tolist(),
    }
    scored = score_world(family, environment, seed, 0, {"equal again": record}, pilot=True)
    assert scored.arms["equal again"].regret == scored.arms["equal split"].regret


def test_the_equal_split_is_moved_into_the_box_at_the_budget(observation):
    """An equal share, 54.06 a week, below the first channel's floor of 70 is raised to it, and
    the other two share the rest of the budget alike."""
    lower = np.array([70.0, 10.0, 10.0])
    boxed = dataclasses.replace(observation, lower=lower, upper=np.full(3, 1000.0))
    rate = observation.budget / observation.planned
    expected = [70.0, (rate - 70.0) / 2.0, (rate - 70.0) / 2.0]
    np.testing.assert_allclose(reading.equal_split(boxed), expected, rtol=1e-10, atol=0.0)


def test_the_equal_split_of_an_export_with_a_geo_axis_is_refused(observation):
    """Two equal geos, each with its own cells and no test."""
    geos = dataclasses.replace(
        observation,
        sales=np.tile(observation.sales[:, None], (1, 2)) / 2,
        spend=np.tile(observation.spend[:, None, :], (1, 2, 1)) / 2,
        population=np.full(2, 0.5),
        lower=np.tile(observation.lower, 2) / 2,
        upper=np.tile(observation.upper, 2) / 2,
        status_quo=np.tile(observation.status_quo, 2) / 2,
        lift=dataclasses.replace(observation.lift, geos=()),
    )
    with pytest.raises(ValueError, match="geo axis"):
        reading.equal_split(geos)


@pytest.mark.parametrize(
    ("name", "variant", "level", "against"),
    [
        ("bayes", "cvar 0.1 vs equal", 0.1, "equal split"),
        ("bayes", "cvar 0.1", 0.1, "status quo"),
        ("dlm", "dlm cvar 0.1 vs equal", 0.1, "equal split"),
        ("dlm", "dlm cvar 0.1", 0.1, "status quo"),
        ("dlm", "dlm mixture", 1.0, "status quo"),
        ("dlm untested", "dlm untested cvar 0.1 vs equal", 0.1, "equal split"),
        ("chc", "chc cvar 0.1 vs equal", 0.1, "equal split"),
    ],
)
def test_each_cvar_variant_reads_its_readings_against_its_reference_in_the_box(
    fitted, observation, monkeypatch, name, variant, level, against
):
    """The no-test variants read the equal split, the others the status quo; each at its level,
    over the fit's readings, within the export's box and budget, the history carried in."""
    arm, _, module = ARMS[name]
    seen = []

    class Seen(Exception):
        """Raised once the call is seen, so that no plan is searched."""

    def spy(readings, budget, periods, **kwargs):
        seen.append((readings, budget, periods, kwargs))
        raise Seen

    monkeypatch.setattr(module, "cvar_allocate", spy)
    with pytest.raises(Seen):
        arm.plan(fitted[name], variant, observation)
    ((readings, budget, periods, kwargs),) = seen
    assert readings is fitted[name].readings
    assert (budget, periods) == (observation.budget, observation.planned)
    assert kwargs.keys() == {"level", "against", "lower", "upper", "history"}
    assert kwargs["level"] == level
    reference = {"equal split": reading.equal_split(observation)}
    np.testing.assert_array_equal(kwargs["against"], reference.get(against, observation.status_quo))
    assert kwargs["lower"] is observation.lower
    assert kwargs["upper"] is observation.upper
    assert kwargs["history"] is observation.spend


@pytest.mark.parametrize("name", ARMS)
def test_each_variant_gains_at_least_0_over_the_equal_split_in_its_worst_tenth(
    fitted, planned, observation, name
):
    """The certificate `cvar_allocate` returns, its plan's, read again on the readings: each
    reading's worth of the plan less its worth of the equal split, the worst tenth's mean."""
    readings, weekly = fitted[name].readings, planned[name].weekly
    equal = reading.equal_split(observation)
    gain = reading.gains(readings, observation, weekly) - reading.gains(
        readings, observation, equal
    )
    worst = float(np.mean(np.sort(gain)[: gain.size // 10]))
    certified = cvar_allocate(
        readings,
        observation.budget,
        observation.planned,
        level=0.1,
        against=equal,
        lower=observation.lower,
        upper=observation.upper,
        history=observation.spend,
    )
    np.testing.assert_array_equal(weekly, np.asarray(certified.spend))
    history = np.asarray(observation.spend, dtype=float)
    scale = max(
        sum(
            reading.worth(c, history[:, j], float(equal[j]), observation.planned)
            for j, c in enumerate(r)
        )
        for r in readings
    )
    assert certified.cvar >= 0.0
    assert worst == pytest.approx(certified.cvar, rel=0.0, abs=1e-9 * scale)
    assert worst >= -1e-9 * scale
    _box(observation, weekly)


def _flat(observation):
    """`observation` with three like channels: each spent 50 a week over the history, boxed in
    [25, 100] a week, the budget 150 a week, so that the equal split, 50 each, is inside the box;
    the status quo spends it otherwise."""
    weeks = observation.sales.shape[0]
    return dataclasses.replace(
        observation,
        spend=np.full((weeks, 3), 50.0),
        status_quo=np.array([60.0, 50.0, 40.0]),
        lower=np.full(3, 25.0),
        upper=np.full(3, 100.0),
        budget=observation.planned * 150.0,
    )


def _reading(observation, coefficients):
    kernel = response.GeometricAdstock(0.5, length=observation.kernel_length, normalized=True)
    return tuple(response.Channel(kernel, response.Tanh(60.0), c) for c in coefficients)


@pytest.mark.parametrize("name", ARMS)
def test_the_variant_is_the_equal_split_where_the_readings_disagree_about_every_move(
    fitted, observation, name
):
    """Every move from the equal split loses, to first order, under the readings in which a
    channel it takes from returns most, a tenth of them at least; so the worst tenth gains only by
    keeping it, though the readings' mean favours the first channel and the status quo is
    elsewhere."""
    arm, variant, _ = ARMS[name]
    flat = _flat(observation)
    readings = tuple(_reading(flat, c) for c in DISAGREE)
    plan = arm.plan(dataclasses.replace(fitted[name], readings=readings), variant, flat)
    np.testing.assert_allclose(plan.weekly, [50.0, 50.0, 50.0], rtol=0.0, atol=1e-9)
    mean = tuple(np.mean(DISAGREE, axis=0))
    moved = allocate(
        list(_reading(flat, mean)),
        flat.budget,
        flat.planned,
        lower=flat.lower,
        upper=flat.upper,
        history=flat.spend,
    ).spend
    assert np.asarray(moved)[0] > 70.0


@pytest.mark.parametrize("name", ARMS)
def test_the_variant_is_the_plan_for_the_readings_mean_where_they_coincide(
    fitted, observation, name
):
    """Every reading the same: the worst tenth's gain is the reading's, so the split is the one
    that reading returns most on, `allocate`'s, to the search's tolerance on the return, far from
    the equal split."""
    arm, variant, _ = ARMS[name]
    flat = _flat(observation)
    one = _reading(flat, DISAGREE[0])
    plan = arm.plan(dataclasses.replace(fitted[name], readings=(one,) * 20), variant, flat)
    best = np.asarray(
        allocate(
            list(one),
            flat.budget,
            flat.planned,
            lower=flat.lower,
            upper=flat.upper,
            history=flat.spend,
        ).spend
    )
    np.testing.assert_allclose(plan.weekly, best, rtol=1e-4, atol=0.0)
    assert best[0] > 80.0

    def worth(weekly):
        return sum(
            reading.worth(c, flat.spend[:, j], float(weekly[j]), flat.planned)
            for j, c in enumerate(one)
        )

    assert worth(plan.weekly) == pytest.approx(worth(best), rel=1e-9, abs=0.0)


@pytest.mark.parametrize("name", ARMS)
def test_each_variant_records_its_plan_and_claim_and_names_where_its_response_is(
    planned, fitted, observation, name
):
    """A plan in the box on the budget, its gain over the status quo with its interval over the
    readings, the planner and its reference in the settings, and the forecast and the response
    named as recorded under the variant that carries them."""
    plan = planned[name]
    record = _fields(plan, observation)
    assert set(record) == PLANNED
    assert record["error"] is None
    _box(observation, record["weekly"])
    gain = reading.gains(fitted[name].readings, observation, plan.weekly)
    assert record["gain"] == pytest.approx(float(np.mean(gain)), rel=1e-12, abs=0.0)
    assert record["gain_interval"] == pytest.approx(
        list(np.quantile(gain, [0.05, 0.95])), rel=1e-12, abs=0.0
    )
    settings = record["settings"]
    assert (settings["planner"], settings["level"], settings["against"]) == (
        "cvar_allocate",
        0.1,
        "equal split",
    )
    elsewhere = f"recorded under {RESPONDS[name]!r}"
    assert record["n/a"]["response"] == elsewhere
    if name != "chc":
        assert record["n/a"]["forecast"] == elsewhere
    assert plan.ess == fitted[name].ess
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize("name", ARMS)
def test_each_variant_plans_alike_from_one_fit(planned, fitted, observation, name):
    arm, variant, _ = ARMS[name]
    again = arm.plan(fitted[name], variant, observation)
    np.testing.assert_array_equal(again.weekly, planned[name].weekly)
    assert (again.gain, again.gain_interval) == (planned[name].gain, planned[name].gain_interval)


def test_the_variants_keep_records_of_their_own():
    """Every arm's variants keep records of their own, but `bayes-geo-plug-in`'s: that arm is
    `bayes-geo` planning one of its variants alone, kept under the variant's name (ADR 0083), and a
    run declares one of the two (`scorecard.run.names`)."""
    made = {name: make() for name, make in run.ARMS.items()}
    alone = made.pop("bayes-geo-plug-in")
    held = [v for arm in made.values() for v in arm.variants]
    ours = [variant for _, variant, _ in ARMS.values()]
    assert set(ours) <= set(held)
    assert len({slug(v) for v in held}) == len(held)
    assert set(alone.variants) < set(made["bayes-geo"].variants)
    assert "ridge cvar 0.1 vs equal" not in held


def test_the_chc_variant_repeats_itself_at_a_seed_and_is_scored_at_every_rung(tmp_path):
    """Fitted once for both rungs, as the arm reads no test, recorded the same twice over at the
    export's seed, and scored beside the status quo and the equal split at each rung."""
    variant = ARMS["chc"][1]
    layout = Layout(tmp_path / "run")
    run.declare(
        layout,
        Declaration(
            family=0,
            environments=("drawn",),
            worlds=1,
            pilot=True,
            rungs=(0, 4),
            arms=("chc",),
            smoke=True,
            limits={"chc": Limits(memory="4G", timeout=600.0, slots=1)},
        ),
    )
    run.export_all(layout)
    (seed,) = layout.read().seeds("drawn")
    run.one(layout, "chc", "drawn", seed, (0, 4))
    first, second = (
        json.loads(layout.record(variant, 0, "drawn", k, seed).read_text()) for k in (0, 4)
    )
    assert set(first) == COMMON | PLANNED | {"smoke", "cost", "provenance"}
    assert second["weekly"] == first["weekly"]
    observation = read(layout.export(0, "drawn", 0, seed))
    again = records(chc_arm.Chc(), observation, smoke=True)[variant]
    assert {k: v for k, v in again.items() if k != "cost"} == {
        k: v for k, v in first.items() if k not in ("cost", "provenance")
    }
    run.score(layout)
    for line in layout.scores.read_text().splitlines():
        arms = json.loads(line)["arms"]
        assert arms[variant]["failure"] is None
        assert math.isfinite(arms[variant]["regret"])
        assert {"status quo", "equal split", "chc"} <= set(arms)
