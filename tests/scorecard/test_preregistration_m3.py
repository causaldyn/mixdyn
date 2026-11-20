"""M3's pre-registration as frozen on the author's confirmation: marked so, no slot left, and
what it states the code it names: the declaration's fields and order, the confirmatory
blocks, the selection rule's tie and the split. The pilot's pre-registration is whole: no slot, no
draft."""

import dataclasses
import json

import pytest

pytest.importorskip("causaldyn_bench")

from causaldyn_bench.scorecard.seeds import TRACK_M2_CONFIRMATORY, confirmatory

from scorecard import development
from scorecard.layout import TREE, Declaration
from tests.test_self_contained import PRIVATE_POINTER

HERE = TREE / "scorecard" / "preregistration"
TEXT = (HERE / "m3.md").read_text(encoding="utf-8")
DOC = json.loads((HERE / "m3.json").read_text(encoding="utf-8"))


def _private(text: str) -> bool:
    """Whether `text` points where the docs test forbids, or into a scratch directory."""
    return bool(PRIVATE_POINTER.search(text)) or "chc-scratch" in text


def test_the_frozen_file_says_so_and_leaves_no_slot():
    frozen = (
        "on 2026-11-17 on the author's confirmation, before any confirmatory world of the run is "
        "fitted"
    )
    assert f"**Frozen {frozen}.**" in " ".join(TEXT.split())
    assert DOC["status"] == f"frozen {frozen}"
    assert "draft" not in TEXT[:400].lower()
    assert "{{" not in TEXT
    assert "}}" not in TEXT
    assert "{{" not in json.dumps(DOC)
    assert not _private(TEXT)
    assert not _private(json.dumps(DOC))
    # the pilot's file is committed whole before its first root is declared
    pilot = (HERE / "m3-pilot.md").read_text(encoding="utf-8")
    assert "**Draft.**" not in pilot
    assert "{{" not in pilot
    assert "}}" not in pilot
    assert not _private(pilot)
    assert DOC["development"]["pilot preregistration"] == "scorecard/preregistration/m3-pilot.md"


def test_the_declaration_it_names_is_one_the_harness_reads():
    fields = {f.name for f in dataclasses.fields(Declaration)}
    held = dict(DOC["claim"]["declaration"])
    assert set(held) <= fields
    order = held.pop("order")
    assert held == {"per_rung": True, "reader": "hall", "confirmatory": True}
    # the order names each verdict once, a family the claim holds at a rung it reads, none past
    # the sizing cap
    pairs = [tuple(pair) for pair in order]
    assert len(pairs) == len(set(pairs)) == 17
    assert not set(pairs) & {tuple(pair) for pair in DOC["exploratory"]["past the cap"]}
    for family, rung in pairs:
        held = DOC["families"][str(family)]
        assert not held["in the claim"].startswith("no")
        assert rung in held["rungs"] or (family, rung) == (7, 1)
    assert "fixed sequence" in DOC["claim"]["multiplicity"]
    assert DOC["margin"] == 0.05
    assert DOC["claim"]["claimed if predictive at least"] == development.CLAIMED


def test_the_seeds_and_the_rule_it_states_are_the_code_s():
    assert confirmatory(3, 2).start == 200_000 + 20_000 * 3 + 1_000 * (2 + 10)
    assert {e: [r.start, r.stop - 1] for e, r in TRACK_M2_CONFIRMATORY.items()} == DOC["seeds"][
        "family 0"
    ]
    dash = "\N{EN DASH}"
    assert f"12 000{dash}12 999" in TEXT
    assert f"22 000{dash}22 999" in TEXT
    assert DOC["development"]["selection"]["tie"] == development.TIE
    assert development.split([5, 3, 4]) == {
        3: development.FIT,
        4: development.FIT,
        5: development.CHECK,
    }
