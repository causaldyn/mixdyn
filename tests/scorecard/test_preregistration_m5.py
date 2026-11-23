"""M5's pre-registration as frozen on the decision the author delegated: marked so, no slot left,
the sizing script it names the one beside it, and what it states the code it names: the
declaration's fields, the order of its verdicts, the third confirmatory run's seeds and the sizing
rule's arithmetic."""

import dataclasses
import hashlib
import json
import math

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


def _private(text: str) -> bool:
    """Whether `text` points where the docs test forbids, or into a scratch directory."""
    return bool(PRIVATE_POINTER.search(text)) or "chc-scratch" in text


def test_the_frozen_file_says_so_and_leaves_no_slot():
    frozen = (
        "on 2026-11-23 on the decision the author delegated, before any confirmatory world of the "
        "run is fitted"
    )
    assert f"**Frozen {frozen}.**" in " ".join(TEXT.split())
    assert DOC["status"] == f"frozen {frozen}"
    assert "draft" not in TEXT[:400].lower()
    assert "}}" not in TEXT
    for text in (TEXT, json.dumps(DOC, ensure_ascii=False)):
        for slot in ("{{", "<<", ">>", "__"):
            assert slot not in text
        assert not _private(text)


def test_the_sizing_script_it_names_is_the_one_beside_it():
    script = TREE / DOC["sizing"]["script"]
    assert script == HERE / "m5_sizing.py"
    assert not _private(script.read_text(encoding="utf-8"))
    held = hashlib.sha256(script.read_bytes()).hexdigest()
    assert DOC["sizing"]["script sha256"] == held
    assert f"`{script.name}` beside this file (SHA-256 `{held}`)" in " ".join(TEXT.split())


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
    chances = [DOC["families"][str(f)]["sizing"][str(k)]["predictive"] for f, k in order]
    assert chances == sorted(chances, reverse=True)
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
    for f, held in DOC["families"].items():
        bench = named(int(f))
        assert seeds[f"family {f}"] == {e: bench[e] for e in held["environments"]}
    assert seeds["family 1"]["walk"] == [1_020_000, 1_020_999]
    assert seeds["family 1"]["static"] == [1_025_000, 1_025_999]
    assert seeds["family 6"]["both"] == [1_122_000, 1_122_999]
    assert seeds["family 14"] == {"genre": [1_280_000, 1_280_999]}
    assert seeds["family 15"] == {"demo": [1_300_000, 1_300_999]}
    dash = "\N{EN DASH}"
    text = " ".join(TEXT.split())
    for held in (
        f"`walk` 1 020 000{dash}1 020 999",
        f"`static` 1 025 000{dash}1 025 999",
        f"`cliff` 1 120 000{dash}1 120 999",
        f"`both` 1 122 000{dash}1 122 999",
        f"`genre` 1 280 000{dash}1 280 999",
        f"`demo` 1 300 000{dash}1 300 999",
    ):
        assert held in text


def test_every_declared_world_is_within_its_block():
    for held in DOC["families"].values():
        assert held["worlds"] * len(held["environments"]) == held["declared"]
        assert m5_sizing.FLOOR <= held["declared"] <= 1_000 * len(held["environments"])


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
