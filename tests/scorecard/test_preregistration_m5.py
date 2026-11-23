"""M5's pre-registration, a draft until its k = 0 arm is frozen: marked so, no slot left but its
two, the sizing script it names the one beside it, and what it states the code it names: the
declaration's fields, the order of its verdicts, the third confirmatory run's seeds and the sizing
rule's arithmetic."""

import dataclasses
import hashlib
import json
import math
import re

import pytest

pytest.importorskip("causaldyn_bench")

from causaldyn_bench.scorecard.family import confirmatory

from scorecard import development
from scorecard.layout import FAMILIES, TREE, Declaration
from scorecard.preregistration import m5_sizing
from tests.test_self_contained import PRIVATE_POINTER

HERE = TREE / "scorecard" / "preregistration"
TEXT = (HERE / "m5.md").read_text(encoding="utf-8")
DOC = json.loads((HERE / "m5.json").read_text(encoding="utf-8"))
SLOTS = {"{{k0 claim}}", "{{at the freeze}}"}
FROZEN = not DOC["status"].startswith("draft")


def _private(text: str) -> bool:
    """Whether `text` points where the docs test forbids, or into a scratch directory."""
    return bool(PRIVATE_POINTER.search(text)) or "chc-scratch" in text


def test_the_draft_says_so_and_leaves_only_its_slots():
    for text in (TEXT, json.dumps(DOC, ensure_ascii=False)):
        held = set(re.findall(r"\{\{[^}]*\}\}", text))
        assert held <= (set() if FROZEN else SLOTS)
        assert "<<" not in text
        assert ">>" not in text
        assert not _private(text)
    if not FROZEN:
        assert "**Draft, 2026-11-23: not frozen.**" in TEXT
        assert DOC["status"].startswith("draft, 2026-11-23: not frozen")


def test_the_sizing_script_it_names_is_the_one_beside_it():
    script = TREE / DOC["sizing"]["script"]
    assert script == HERE / "m5_sizing.py"
    assert not _private(script.read_text(encoding="utf-8"))
    held = hashlib.sha256(script.read_bytes()).hexdigest()
    named = DOC["sizing"]["script sha256"]
    assert named == held or (not FROZEN and named == "{{at the freeze}}")
    assert f"`{script.name}` beside this file (SHA-256 `{named}`)" in " ".join(TEXT.split())


def test_the_declaration_it_names_is_one_the_harness_reads():
    fields = {f.name for f in dataclasses.fields(Declaration)}
    held = dict(DOC["claim"]["declaration"])
    assert set(held) <= fields
    held.pop("order")
    assert held == {
        "per_rung": True,
        "reader": "hall",
        "confirmatory": True,
        "third_confirmatory": True,
    }
    assert DOC["margin"] == development.MARGIN == m5_sizing.MARGIN
    assert DOC["claim"]["claimed if predictive at least"] == development.CLAIMED


def test_the_order_names_every_verdict_the_claim_reads_once():
    order = [tuple(pair) for pair in DOC["claim"]["declaration"]["order"]]
    past = {tuple(pair) for pair in DOC["exploratory"]["past the cap"]}
    rungs = {(int(f), k) for f, held in DOC["families"].items() for k in held["rungs"]}
    assert len(order) == len(set(order))
    assert set(order) == rungs - past
    assert sorted(int(f) for f in DOC["families"]) == DOC["sizing"]["families"]
    assert all(isinstance(held["worlds"], int) for held in DOC["families"].values())


def test_each_family_names_its_environments_in_the_bench_s_order():
    # family 1 leaves out `cpm` and family 2 `search-demand`, as M3 and M4 did; a seed's block
    # counts the environment in the bench's order, the ones left out included
    for f, held in DOC["families"].items():
        named = list(FAMILIES[int(f)].scored)
        assert held["environments"] == [e for e in named if e in held["environments"]]
        assert set(held["environments"]) <= set(named)


def test_the_third_run_s_seeds_are_the_bench_s():
    def named(family: int) -> dict[str, list[int]]:
        held = FAMILIES[family]
        source = held(TREE) if isinstance(held, type) else held
        third = confirmatory(source, third=True)
        return {e: [r.start, r.stop - 1] for e, r in third.items()}

    seeds = DOC["seeds"]
    assert (
        named(0)
        == seeds["family 0"]
        == {
            "drawn": [1_000_000, 1_000_999],
            "reference": [1_001_000, 1_001_999],
        }
    )
    assert named(15) == seeds["family 15"] == {"demo": [1_300_000, 1_300_999]}
    assert (
        named(17)
        == seeds["family 17"]
        == {
            "curve": [1_340_000, 1_340_999],
            "line": [1_341_000, 1_341_999],
        }
    )
    dash = "\N{EN DASH}"
    for text in ("1 000 000", "1 000 999", "1 300 000", "1 300 999", "1 340 000", "1 341 999"):
        assert text in TEXT
    assert f"`line` 1 341 000{dash}1 341 999" in " ".join(TEXT.split())


def test_its_constants_are_the_pre_registration_s():
    sizing = DOC["sizing"]
    assert (m5_sizing.FLOOR, m5_sizing.CAP, m5_sizing.POWER) == (240, 520, 0.8)
    assert (sizing["floor"], sizing["cap"], sizing["power"]) == (240, 520, 0.8)
    assert m5_sizing.RESAMPLES == sizing["predictive"]["resamples"] == 4000
    assert sizing["predictive"]["seed"] == m5_sizing.SEED
    assert tuple(sizing["families"]) == m5_sizing.FAMILIES
    assert m5_sizing.TESTED_RUN is sizing["tested rung run"] is False
    assert abs(m5_sizing.Z + m5_sizing.Q - m5_sizing.TIE_Z) < 5e-5


# (margin, mean, sd, pairs) and the unrounded N Maxima's closed form gives there at fpprec 40
MAXIMA = [
    ((0.05, 0.0, 0.1, 240), 32.64998100868319),
    ((0.05, -0.18, 0.41, 240), 25.73031003042146),
    ((0.05, -0.37, 0.69, 240), 21.751992644323735),
    ((0.05, 0.02, 0.1, 240), 97.18969919544371),
    ((0.05, 0.03, 0.05, 520), 50.462433302799205),
    ((0.05, -0.005, 0.12, 120), 41.001569170675616),
]


@pytest.mark.parametrize(("point", "expected"), MAXIMA)
def test_a_comparison_s_n_is_the_closed_form_maxima_checked(point, expected):
    _, mean, sd, pairs = point
    held = m5_sizing.predictive_needed(mean, sd, pairs)
    assert held is not None
    assert held == pytest.approx(expected, rel=1e-12, abs=0.0)
    assert m5_sizing.calibrated(mean, sd, pairs, held) == pytest.approx(0.8, rel=1e-12, abs=0.0)


def test_an_exact_sizing_mean_gives_the_power_formula_and_at_a_tie_the_tie_rule():
    z, q, margin = m5_sizing.Z, m5_sizing.Q, m5_sizing.MARGIN
    for mean, sd in ((0.0, 0.41), (-0.18, 0.41), (0.02, 0.1)):
        exact = ((z + q) * sd / (margin - mean)) ** 2
        held = m5_sizing.predictive_needed(mean, sd, 10**12)
        assert held == pytest.approx(exact, rel=1e-6, abs=0.0)
    at_a_tie = m5_sizing.predictive_needed(0.0, 0.41, 10**12)
    assert at_a_tie is not None
    assert m5_sizing.tens(at_a_tie) == m5_sizing.tie_needed(0.41)


def test_a_comparison_it_cannot_size_is_not_sized():
    margin, q = m5_sizing.MARGIN, m5_sizing.Q
    assert m5_sizing.predictive_needed(margin, 0.1, 240) is None
    assert m5_sizing.predictive_needed(margin + 0.01, 0.1, 240) is None
    sd = 0.1
    at_the_bound = margin - q * sd / math.sqrt(240)
    assert m5_sizing.predictive_needed(at_the_bound + 1e-6, sd, 240) is None
    assert m5_sizing.predictive_needed(at_the_bound - 1e-4, sd, 240) is not None


def test_the_tie_rule_is_m4_s_in_tens():
    def sd_for(n: float) -> float:
        return m5_sizing.MARGIN * math.sqrt(n) / m5_sizing.TIE_Z

    assert m5_sizing.tie_needed(sd_for(345)) == 350
    assert m5_sizing.tie_needed(sd_for(350.5)) == 360


def test_the_expected_count_is_the_fixed_sequence_s():
    assert m5_sizing.expected([1.0, 1.0, 1.0]) == 3.0
    assert m5_sizing.expected([0.5, 0.5]) == 0.75
    assert m5_sizing.expected([0.0, 1.0]) == 0.0
