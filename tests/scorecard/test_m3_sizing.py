"""M3's sizing script: its constants are the pre-registration's and the selection rule's, its N is
the stated formula in tens, its expected count is the fixed sequence's, and the SHA-256 the
pre-registration names is the file's."""

import hashlib
import json
import math
import statistics

import pytest

pytest.importorskip("causaldyn_bench")

from scorecard import development
from scorecard.layout import TREE
from scorecard.preregistration import m3_sizing

HERE = TREE / "scorecard" / "preregistration"
DOC = json.loads((HERE / "m3.json").read_text(encoding="utf-8"))


def test_its_constants_are_the_pre_registration_s_and_the_selection_rule_s():
    assert m3_sizing.MARGIN == DOC["margin"] == development.MARGIN
    assert (m3_sizing.FLOOR, m3_sizing.CAP) == (240, 520)
    assert m3_sizing.RESAMPLES == development.RESAMPLES
    assert m3_sizing.SEED == development.SEED
    z = statistics.NormalDist().inv_cdf
    assert abs(m3_sizing.Z - (z(0.975) + z(0.8))) < 5e-5


def test_n_is_the_stated_formula_in_tens():
    def sd_for(n: float) -> float:
        return m3_sizing.MARGIN * math.sqrt(n) / m3_sizing.Z

    assert m3_sizing.needed(sd_for(345)) == 350
    assert m3_sizing.needed(sd_for(350.5)) == 360
    assert m3_sizing.needed(sd_for(1)) == 10


def test_the_expected_count_is_the_fixed_sequence_s():
    assert m3_sizing.expected([1.0, 1.0, 1.0]) == 3.0
    assert m3_sizing.expected([0.5, 0.5]) == 0.75
    assert m3_sizing.expected([0.0, 1.0]) == 0.0


def test_the_hash_the_pre_registration_names_is_the_file_s():
    digest = hashlib.sha256((HERE / "m3_sizing.py").read_bytes()).hexdigest()
    assert DOC["sizing"]["script"] == "scorecard/preregistration/m3_sizing.py"
    assert DOC["sizing"]["script sha256"] == digest
    assert f"SHA-256 `{digest}`" in (HERE / "m3.md").read_text(encoding="utf-8")
