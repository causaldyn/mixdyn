"""The long-term null worlds of `validation/long_term_null.py` (ADR 0014, amended 2026-11-06),
reduced: nothing television builds lasts in sales, so its long-term share is 0.

The record, `validation/long_term_null.md`, reads 100 worlds whose brand measure sales do not read
under the nested model, the first 20 of them under a dual adstock too, and 10 whose brand measure
television does not build. These tests read the first world of each kind and check the record.
The criterion, that the nested model's 90 % interval of the share covers 0 on at least 90 % of the
first kind, is met where the rate's 95 % Clopper-Pearson interval holds 0.9 or lies above it: 91
of 100, [0.836, 0.958]. A dual adstock, the practice the engine does not ship, covers 0 on none of
its 20, and the nested model on none of the second kind, the limit the amendment records.

The record's second table reads the nested model with stage one's return prior half-normal (ADR
0076) on 20 worlds of the first kind and 100 of the second, against the criterion set for it: that
the share covers 0 on at least 90 % of the second kind, read as the first's is. It is met: 87 of
100, [0.788, 0.929]. On the first kind it covers 0 on 18 of 20.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from validation import long_term_null as ltn

RECORD = Path(__file__).resolve().parent.parent / "validation" / "long_term_null.md"


def test_the_cuts_differ_only_in_the_path_through_the_searches() -> None:
    """Both worlds of a seed draw the same spend and price; television moves the searches only
    where sales do not read them."""
    data, _ = ltn.seeds(1)[0]
    unread, unbuilt = ltn.world(data, "unread"), ltn.world(data, "unbuilt")
    np.testing.assert_array_equal(unread.spend, unbuilt.spend)
    np.testing.assert_array_equal(unread.control_values[..., 0], unbuilt.control_values[..., 0])
    tv = unread.spend[0, :, 0]
    gap = unread.control_values[0, :, 1] - unbuilt.control_values[0, :, 1]
    assert np.all(gap >= 0.0)
    assert np.corrcoef(gap, tv)[0, 1] > 0.5


@pytest.fixture(scope="module")
def first() -> dict[tuple[str, str], ltn.Reading]:
    data, seed = ltn.seeds(1)[0]
    found = [
        *ltn.read(0, data, seed, "unread"),
        *ltn.read(0, data, seed, "unbuilt", nested=("nested", "nested half-normal")),
    ]
    return {(r.model, r.cut): r for r in found}


def test_where_sales_do_not_read_the_brand_the_nested_share_covers_0(
    first: dict[tuple[str, str], ltn.Reading],
) -> None:
    """The criterion on the test's own one world. At the rate the record measures, 0.91, a world
    drawn at random would miss with probability 0.09, and at the criterion's 0.9 with 0.1; the
    world is fixed, and its lower end, 0.0053 in the record, is a quarter of `NIL`, so draws that
    move in their last bits on another CPU class do not flip it."""
    nested = first["nested", "unread"]
    assert nested.covers
    assert nested.low < 0.5 * ltn.NIL


def test_where_sales_do_not_read_the_brand_a_dual_adstock_reads_a_long_term_share(
    first: dict[tuple[str, str], ltn.Reading],
) -> None:
    dual = first["dual adstock", "unread"]
    assert not dual.covers
    assert dual.low > 2.0 * ltn.NIL


def test_where_television_does_not_build_the_brand_the_nested_share_stays_off_0(
    first: dict[tuple[str, str], ltn.Reading],
) -> None:
    """Stage one's log-normal return prior has no mass near 0, so the share it carries stays
    above `NIL` where the data read no return on the searches: the limit ADR 0014 records."""
    assert not first["nested", "unbuilt"].covers


def test_where_television_does_not_build_the_brand_a_half_normal_stage_one_covers_0(
    first: dict[tuple[str, str], ltn.Reading],
) -> None:
    """Stage one's return prior half-normal, whose density is greatest at 0: on the same world
    the share's interval reaches below `NIL`, its lower end 0.0060 in the record, under a third of
    it, so draws that move in their last bits on another CPU class do not flip it."""
    half = first["nested half-normal", "unbuilt"]
    assert half.covers
    assert half.low < 0.5 * ltn.NIL


def _readings() -> list[ltn.Reading]:
    return ltn.rows(RECORD.read_text(encoding="utf-8"))


def test_the_record_reads_what_the_amendment_states() -> None:
    """Each model's worlds from the first, each once: 100 under the nested model where sales do not
    read the brand, 20 under the dual adstock, 10 where television does not build it; and the
    counts ADR 0014's amendment states, 91 of the 100, none of the 20 and none of the 10, which
    are the summary lines the record prints from its rows. With a half-normal stage one, 20 and
    100 worlds, and the counts ADR 0076 states: 18 of the 20 and 87 of the 100."""
    readings = _readings()
    worlds = {
        (m, cut): sorted(r.world for r in readings if (r.model, r.cut) == (m, cut))
        for m, cut in {(r.model, r.cut) for r in readings}
    }
    assert worlds == {
        ("nested", "unread"): list(range(ltn.WORLDS)),
        ("dual adstock", "unread"): list(range(ltn.DUAL_WORLDS)),
        ("nested", "unbuilt"): list(range(ltn.UNBUILT)),
        ("nested half-normal", "unread"): list(range(20)),
        ("nested half-normal", "unbuilt"): list(range(100)),
    }
    covered = {key: sum(r.covers for r in readings if (r.model, r.cut) == key) for key in worlds}
    assert covered == {
        ("nested", "unread"): 91,
        ("dual adstock", "unread"): 0,
        ("nested", "unbuilt"): 0,
        ("nested half-normal", "unread"): 18,
        ("nested half-normal", "unbuilt"): 87,
    }
    lines = [line for line in RECORD.read_text(encoding="utf-8").splitlines() if line[:2] == "- "]
    assert lines == ltn.summary(readings)


def test_the_record_meets_the_criterion_where_sales_do_not_read_the_brand() -> None:
    """The nested share's rate of covering 0 where sales do not read the searches has a 95 %
    Clopper-Pearson interval that holds 0.9 or lies above it: [0.836, 0.958] on 100 worlds. The
    dual adstock's lies wholly below it, and so does the nested share's where television does not
    build the searches."""
    readings = _readings()

    def rate(model: str, cut: str) -> tuple[float, float]:
        mine = [r for r in readings if (r.model, r.cut) == (model, cut)]
        return ltn.interval(sum(r.covers for r in mine), len(mine))

    assert rate("nested", "unread")[1] >= 0.9
    assert rate("dual adstock", "unread")[1] < 0.9
    assert rate("nested", "unbuilt")[1] < 0.9


def test_the_record_meets_the_criterion_where_television_does_not_build_the_brand() -> None:
    """With a half-normal stage one, the share's rate of covering 0 where television does not build
    the searches has a 95 % Clopper-Pearson interval that holds 0.9 or lies above it: [0.788,
    0.929] on 100 worlds, the log-normal's [0, 0.308] on 10 lying below it."""
    readings = _readings()
    mine = [r for r in readings if (r.model, r.cut) == ("nested half-normal", "unbuilt")]
    assert ltn.interval(sum(r.covers for r in mine), len(mine))[1] >= 0.9
