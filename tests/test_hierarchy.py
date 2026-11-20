from __future__ import annotations

import numpy as np
import pytest

from mixdyn import Cadence, IssueCode, MediaPanel, PanelError, Revenue, Severity, read_panel
from mixdyn.data.panel import Geo, hierarchy_problems
from test_read_panel import others

GEOS = (Geo("berlin"), Geo("munich"), Geo("paris"))


def table() -> dict[str, np.ndarray]:
    weeks = 4
    return {
        "date": np.tile(
            np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(weeks), len(GEOS)
        ),
        "city": np.repeat(np.array(GEOS, dtype=object), weeks),
        "sales": np.arange(12, dtype=float) + 1,
        "tv": np.arange(12, dtype=float) % 5 + 1,
    }


def read(hierarchy: dict[str, str]) -> MediaPanel:
    return read_panel(
        table(),
        date="date",
        geo="city",
        kpi="sales",
        spend={"tv": "tv"},
        cadence=Cadence.WEEK,
        kpi_unit=Revenue(),
        currency="EUR",
        hierarchy=hierarchy,
    )


TREE = {"berlin": "DE", "munich": "DE", "paris": "FR", "DE": "EU", "FR": "EU"}


def test_a_tree_over_the_geos_is_kept() -> None:
    panel = read(TREE)
    assert dict(panel.hierarchy) == TREE
    assert others(panel.issues) == ()


def test_nodes_with_no_geo_below_them_are_dropped_with_a_warning() -> None:
    panel = read({**TREE, "lyon": "FR", "rome": "IT", "IT": "EU"})
    assert dict(panel.hierarchy) == TREE
    (warning,) = others(panel.issues)
    assert warning.severity is Severity.WARNING
    assert "['IT', 'lyon', 'rome']" in warning.message


@pytest.mark.parametrize(
    ("hierarchy", "match"),
    [
        ({"berlin": "DE", "munich": "DE", "DE": "EU"}, "geo 'paris' has no parent"),
        ({**TREE, "DE": "berlin"}, "cycle: "),
        ({"berlin": "DE", "munich": "DE", "paris": "FR"}, "2 roots, \\['DE', 'FR'\\]"),
        ({**TREE, "munich": "berlin"}, "geo 'berlin' is in the data and is the parent of"),
    ],
)
def test_a_hierarchy_that_is_not_one_tree_over_the_geos_is_an_error(
    hierarchy: dict[str, str], match: str
) -> None:
    with pytest.raises(PanelError, match=match) as excinfo:
        read(hierarchy)
    assert all(i.code is IssueCode.HIERARCHY for i in excinfo.value.issues)


def test_the_panel_itself_refuses_a_stray_node() -> None:
    tree = {Geo(child): Geo(up) for child, up in TREE.items()}
    problems = hierarchy_problems(GEOS, {**tree, Geo("lyon"): Geo("FR")})
    assert problems == [
        "the hierarchy names 'lyon', which is neither a geo in the data nor anyone's parent"
    ]


def test_a_cycle_is_reported_once() -> None:
    problems = hierarchy_problems(
        (Geo("a"),), {Geo("a"): Geo("b"), Geo("b"): Geo("c"), Geo("c"): Geo("b")}
    )
    assert [p for p in problems if "cycle" in p] == ["the hierarchy has a cycle: b -> c -> b"]
