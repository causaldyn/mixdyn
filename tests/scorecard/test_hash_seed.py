"""The arms' plans must not read Python's string-hash randomisation. One world each of `margin` at
k = 0, `anchored` at the tested rung and `geo eb` on family 8 is planned in two fresh interpreters
under two `PYTHONHASHSEED`s, and every variant's record but its cost must be the same bytes in
both. So must the shares of the market that a geo export's tests are read in
(`scorecard.arms.panel._in_share`), which the arms that plan a geo export's market and read its
tests fit on: `bayes`, the readers and `switchback`. The oracle is the other interpreter. That the
comparison can see a set's order is shown on the same world: a left fold over each test's geos, a
frozenset of their names, in the set's own order differs between the two hash seeds."""

import json
import os
import subprocess
import sys
from collections.abc import Iterable

import numpy as np
import pytest

pytest.importorskip("causaldyn_bench")

from causaldyn_bench.scorecard.geo import GEO
from test_anchored_arm import observation as gs_export
from test_margin_arm import observation as margin_export

from gs_worlds import WORLDS as TESTED
from margin_worlds import PARITY
from scorecard.arms import anchored, geo_eb, margin
from scorecard.arms import panel as reading
from scorecard.arms.base import records

# Under these two the tests' geo sets of family 8's pilot world iterate in orders whose left folds
# differ in the last bit; under 0 and 12345 they happen to fold alike.
SEEDS = ("0", "1")
ENVIRONMENT = "policy-0"


def _folded(values: Iterable[float]) -> float:
    """A left fold, which reads its values' order."""
    total = 0.0
    for value in values:
        total += float(value)
    return total


def plans() -> dict[str, str]:
    """Each arm's records of one world, without their costs; the market's lift shares; and each
    test's population summed by a left fold in its geo set's order; each as canonical JSON."""
    geo = GEO.observe(GEO.world(ENVIRONMENT, GEO.pilots[ENVIRONMENT][0]), 4)
    population, named = np.asarray(geo.population, dtype=float), reading.geos(geo)
    tests = [e for es in reading.estimates(geo).values() for e in es]
    folds = [float(_folded(population[named.index(g)] for g in e.geos or ())).hex() for e in tests]
    read = reading.estimates(reading.market(geo))
    shares = [float(e.share).hex() for es in read.values() for e in es]
    out = {"folds": json.dumps(folds), "shares": json.dumps(shares)}
    made = {
        "margin": records(margin.Margin(), margin_export(PARITY[0]), smoke=False),
        "anchored": records(anchored.Anchored(), gs_export(TESTED[0]), smoke=False),
        "geo eb": records(geo_eb.GeoEb(), geo, smoke=False),
    }
    for arm, variants in made.items():
        kept = {v: {k: x for k, x in r.items() if k != "cost"} for v, r in variants.items()}
        out[arm] = json.dumps(kept, sort_keys=True)
    return out


@pytest.fixture(scope="module")
def planned() -> list[dict[str, str]]:
    """`plans` in a fresh interpreter under each of `SEEDS`, the two run at once."""
    code = "import json, test_hash_seed as t; print(json.dumps(t.plans()))"
    env = os.environ | {"PYTHONPATH": os.pathsep.join(sys.path), "JAX_ENABLE_X64": "1"}
    started = [
        subprocess.Popen(
            [sys.executable, "-c", code],
            env=env | {"PYTHONHASHSEED": seed},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for seed in SEEDS
    ]
    out = []
    for process in started:
        stdout, stderr = process.communicate(timeout=900)
        assert process.returncode == 0, stderr[-2000:]
        out.append(json.loads(stdout.splitlines()[-1]))
    return out


@pytest.mark.parametrize("arm", ["margin", "anchored", "geo eb"])
def test_an_arms_records_are_the_same_bytes_under_two_hash_seeds(planned, arm):
    first, second = planned
    assert json.loads(first[arm])
    assert first[arm] == second[arm]


def test_the_markets_test_shares_are_the_same_bytes_under_two_hash_seeds(planned):
    first, second = planned
    assert json.loads(first["shares"])
    assert first["shares"] == second["shares"]


def test_a_fold_in_a_geo_sets_order_tells_the_two_hash_seeds_apart(planned):
    first, second = planned
    assert first["folds"] != second["folds"]
