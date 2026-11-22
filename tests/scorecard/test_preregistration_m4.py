"""M4's pre-registration as frozen on the author's confirmation: marked so, no slot left, the
sizing script it names the one beside it, and what it states the code it names: the declaration's
fields, the order of its verdicts, the second confirmatory run's seeds, the claim's margin and the
predictive's bar."""

import dataclasses
import hashlib
import json

import pytest

pytest.importorskip("causaldyn_bench")

from causaldyn_bench.scorecard.family import confirmatory
from causaldyn_bench.scorecard.seeds import SECOND_RUN, TRACK_M2_CONFIRMATORY
from causaldyn_bench.scorecard.track_m2 import TRACK_M2

from scorecard import development
from scorecard.layout import TREE, Declaration
from tests.test_self_contained import PRIVATE_POINTER

HERE = TREE / "scorecard" / "preregistration"
TEXT = (HERE / "m4.md").read_text(encoding="utf-8")
DOC = json.loads((HERE / "m4.json").read_text(encoding="utf-8"))


def _private(text: str) -> bool:
    """Whether `text` points where the docs test forbids, or into a scratch directory."""
    return bool(PRIVATE_POINTER.search(text)) or "chc-scratch" in text


def test_the_frozen_file_says_so_and_leaves_no_slot():
    frozen = (
        "on 2026-11-21 on the author's confirmation, before any confirmatory world of the run is "
        "fitted"
    )
    assert f"**Frozen {frozen}.**" in " ".join(TEXT.split())
    assert DOC["status"] == f"frozen {frozen}"
    assert "draft" not in TEXT[:400].lower()
    assert "}}" not in TEXT
    for text in (TEXT, json.dumps(DOC, ensure_ascii=False)):
        assert "<<" not in text
        assert ">>" not in text
        assert "{{" not in text
        assert not _private(text)


def test_the_sizing_script_it_names_is_the_one_beside_it():
    script = TREE / DOC["sizing"]["script"]
    assert script.parent == HERE
    held = hashlib.sha256(script.read_bytes()).hexdigest()
    assert DOC["sizing"]["script sha256"] == held
    assert f"`{script.name}` beside this file (SHA-256 `{held}`)" in " ".join(TEXT.split())
    assert not _private(script.read_text(encoding="utf-8"))


def test_the_declaration_it_names_is_one_the_harness_reads():
    fields = {f.name for f in dataclasses.fields(Declaration)}
    held = dict(DOC["claim"]["declaration"])
    assert set(held) <= fields
    held.pop("order")
    assert held == {
        "per_rung": True,
        "reader": "hall",
        "confirmatory": True,
        "second_confirmatory": True,
    }
    assert DOC["margin"] == 0.05
    assert DOC["claim"]["claimed if predictive at least"] == development.CLAIMED


def test_the_order_names_every_verdict_the_claim_reads_once():
    order = [tuple(pair) for pair in DOC["claim"]["declaration"]["order"]]
    past = {tuple(pair) for pair in DOC["exploratory"]["past the cap"]}
    rungs = {(int(f), k) for f in DOC["families"] for k in (0, 1 if f == "7" else 4)}
    assert len(order) == len(set(order))
    assert set(order) == rungs - past
    assert all(isinstance(held["worlds"], int) for held in DOC["families"].values())


def test_the_second_run_s_seeds_are_the_bench_s():
    second = confirmatory(TRACK_M2, second=True)
    assert {e: [r.start, r.stop - 1] for e, r in second.items()} == DOC["seeds"]["family 0"]
    assert all(r.start == TRACK_M2_CONFIRMATORY[e].start + SECOND_RUN for e, r in second.items())
    assert f"first {SECOND_RUN} seeds" in " ".join(TEXT.split())
    dash = "\N{EN DASH}"
    assert f"12 300{dash}12 999" in TEXT
    assert f"22 300{dash}22 999" in TEXT
