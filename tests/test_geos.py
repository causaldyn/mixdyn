"""Media bought nationally and split by population, and the panel at cuts of its hierarchy: each
node the sum of the geos below it, so every level adds up to the next."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mixdyn import (
    Allotment,
    Cadence,
    Channel,
    Combine,
    Geo,
    IssueCode,
    MediaPanel,
    PanelError,
    Revenue,
    Severity,
    aggregate,
    allot_national,
    read_panel,
)
from test_read_panel import others

GEOS = tuple(Geo(g) for g in ("berlin", "munich", "paris", "lyon", "rome"))
TREE = {
    Geo("berlin"): Geo("DE"),
    Geo("munich"): Geo("DE"),
    Geo("paris"): Geo("FR"),
    Geo("lyon"): Geo("FR"),
    Geo("rome"): Geo("EU"),
    Geo("DE"): Geo("EU"),
    Geo("FR"): Geo("EU"),
}
POPULATION = np.array([3.7e6, 1.5e6, 2.1e6, 0.5e6, 2.8e6])
WEEKS = 6


def panel(
    seed: int = 0,
    *,
    controls: tuple[str, ...] = ("price", "visits"),
    filled: np.ndarray | None = None,
    **overrides: Any,
) -> MediaPanel:
    rng = np.random.default_rng(seed)
    gone = np.zeros((len(GEOS), WEEKS), dtype=bool) if filled is None else filled
    kpi = rng.gamma(2.0, 50.0, (len(GEOS), WEEKS))
    spend = rng.gamma(2.0, 5.0, (len(GEOS), WEEKS, 2))
    values = rng.gamma(2.0, 1.0, (len(GEOS), WEEKS, len(controls)))
    kpi[gone] = np.nan
    spend[gone] = 0.0
    values[gone] = np.nan
    fields: dict[str, Any] = {
        "dates": np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(WEEKS),
        "cadence": Cadence.WEEK,
        "geos": GEOS,
        "channels": (Channel("search"), Channel("social")),
        "controls": controls,
        "kpi": kpi,
        "kpi_unit": Revenue(),
        "spend": spend,
        "currency": "EUR",
        "control_values": values,
        "filled": gone,
        "hierarchy": TREE,
        "population": POPULATION,
    }
    fields.update(overrides)
    return MediaPanel(**fields)


RULES = {"price": Combine.POPULATION_MEAN, "visits": Combine.SUM}


def test_national_spend_is_split_by_population_and_the_split_is_stated() -> None:
    total = np.array([100.0, 0.0, 50.0, 80.0, 0.0, 10.0])
    out = allot_national(panel(), "tv", total)
    assert out.channels[-1] == "tv"
    assert dict(out.national) == {"tv": Allotment.POPULATION}
    tv = out.spend[:, :, -1]
    np.testing.assert_allclose(tv, POPULATION[:, None] / POPULATION.sum() * total, rtol=1e-15)
    np.testing.assert_allclose(tv.sum(axis=0), total, rtol=1e-14)
    np.testing.assert_array_equal(out.spend[:, :, :2], panel().spend)


@pytest.mark.parametrize(
    ("make", "channel", "total", "match"),
    [
        (panel, "search", np.ones(WEEKS), "already has a channel 'search'"),
        (panel, "tv", np.ones(WEEKS + 1), r"shape \(7,\), expected \(6,\)"),
        (panel, "tv", -np.ones(WEEKS), "finite and non-negative"),
        (panel, "tv", np.full(WEEKS, np.nan), "finite and non-negative"),
        (lambda: panel(population=None), "tv", np.ones(WEEKS), "the panel has none"),
        (
            lambda: panel(filled=np.eye(len(GEOS), WEEKS, dtype=bool)),
            "tv",
            np.ones(WEEKS),
            r"falls in 5 periods a geo has no row for, from berlin 2024-01-01",
        ),
    ],
)
def test_national_spend_that_cannot_be_split_is_refused(
    make: Any, channel: str, total: np.ndarray, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        allot_national(make(), channel, total)


def test_national_spend_in_a_week_no_geo_lacks_is_split_around_the_week_one_does() -> None:
    gone = np.zeros((len(GEOS), WEEKS), dtype=bool)
    gone[1, 2] = True
    total = np.array([1.0, 2.0, 0.0, 4.0, 5.0, 6.0])
    out = allot_national(panel(filled=gone), "tv", total)
    assert out.spend[1, 2, -1] == 0.0


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"population": POPULATION[:3]}, r"population has shape \(3,\), expected \(5,\)"),
        ({"population": -POPULATION}, "finite and positive"),
        ({"national": {Channel("tv"): Allotment.POPULATION}}, "'tv' is not one of"),
        ({"national": {Channel("search"): "population"}}, "is not an Allotment"),
        (
            {"national": {Channel("search"): Allotment.POPULATION}, "population": None},
            "split by a population the panel lacks",
        ),
        (
            {"national": {Channel("search"): Allotment.POPULATION}},
            r"'search' is not split by population in 6 periods, from 2024-01-01",
        ),
    ],
)
def test_a_panel_states_a_national_split_only_where_it_holds(
    overrides: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        panel(**overrides)


def test_each_node_is_the_sum_of_the_geos_below_it() -> None:
    base = panel()
    out = aggregate(base, ["DE", "FR", "rome"], RULES)
    assert out.geos == ("DE", "FR", "rome")
    assert dict(out.hierarchy) == {"DE": "EU", "FR": "EU", "rome": "EU"}
    for node, rows in (("DE", [0, 1]), ("FR", [2, 3]), ("rome", [4])):
        n = out.geos.index(Geo(node))
        np.testing.assert_allclose(out.kpi[n], base.kpi[rows].sum(axis=0), rtol=1e-15)
        np.testing.assert_allclose(out.spend[n], base.spend[rows].sum(axis=0), rtol=1e-15)
        assert out.population is not None
        assert out.population[n] == pytest.approx(POPULATION[rows].sum(), rel=1e-15)
        weights = POPULATION[rows] / POPULATION[rows].sum()
        np.testing.assert_allclose(
            out.control_values[n, :, 0], weights @ base.control_values[rows, :, 0]
        )
        np.testing.assert_allclose(
            out.control_values[n, :, 1], base.control_values[rows, :, 1].sum(axis=0)
        )


def test_a_geo_may_be_a_node_of_its_own_and_the_root_leaves_no_hierarchy() -> None:
    base = panel(controls=())
    leaves = aggregate(base, list(GEOS))
    np.testing.assert_array_equal(leaves.kpi, base.kpi)
    assert dict(leaves.hierarchy) == TREE
    country = aggregate(base, ["EU"])
    assert country.geos == ("EU",)
    assert dict(country.hierarchy) == {}
    np.testing.assert_allclose(country.spend[0], base.spend.sum(axis=0), rtol=1e-14)


def test_a_node_misses_its_kpi_where_any_geo_below_it_does_and_is_filled_where_all_are() -> None:
    gone = np.zeros((len(GEOS), WEEKS), dtype=bool)
    gone[0, 1] = True
    gone[[0, 1], 3] = True
    base = panel(controls=(), filled=gone)
    out = aggregate(base, ["DE", "FR", "rome"])
    de = out.geos.index(Geo("DE"))
    assert np.isnan(out.kpi[de, [1, 3]]).all()
    assert np.isfinite(np.delete(out.kpi[de], [1, 3])).all()
    assert out.filled[de].tolist() == [False, False, False, True, False, False]
    np.testing.assert_allclose(out.spend[de, 1], base.spend[1, 1])


def test_national_media_stay_split_by_population_at_every_level() -> None:
    base = allot_national(panel(), "tv", np.arange(WEEKS, dtype=float) * 10.0)
    regions = aggregate(base, ["DE", "FR", "rome"], RULES)
    assert dict(regions.national) == {"tv": Allotment.POPULATION}
    assert regions.population is not None
    share = regions.population / regions.population.sum()
    np.testing.assert_allclose(regions.spend[:, :, -1], share[:, None] * np.arange(WEEKS) * 10.0)


@st.composite
def panels(draw: st.DrawFn) -> MediaPanel:
    gone = np.array(
        draw(st.lists(st.booleans(), min_size=len(GEOS) * WEEKS, max_size=len(GEOS) * WEEKS))
    )
    return panel(
        draw(st.integers(0, 2**32 - 1)), controls=(), filled=gone.reshape(len(GEOS), WEEKS)
    )


@settings(max_examples=60, deadline=None)
@given(panels())
def test_two_steps_up_the_hierarchy_are_one(base: MediaPanel) -> None:
    direct = aggregate(base, ["EU"])
    stepped = aggregate(aggregate(base, ["DE", "FR", "rome"]), ["EU"])
    np.testing.assert_allclose(stepped.kpi, direct.kpi, rtol=1e-13)
    np.testing.assert_allclose(stepped.spend, direct.spend, rtol=1e-13)
    np.testing.assert_array_equal(stepped.filled, direct.filled)
    assert stepped.population is not None
    assert direct.population is not None
    np.testing.assert_allclose(stepped.population, direct.population, rtol=1e-15)
    np.testing.assert_allclose(direct.spend[0], base.spend.sum(axis=0), rtol=1e-13)


@pytest.mark.parametrize(
    ("make", "nodes", "rules", "match"),
    [
        (panel, ["DE", "berlin", "FR", "rome"], RULES, r"'berlin' sits under \['berlin', 'DE'\]"),
        (panel, ["DE", "FR"], RULES, "'rome' sits under none"),
        (panel, ["DE", "FR", "rome", "UK"], RULES, r"\['UK'\] are neither geos"),
        (panel, ["DE", "DE"], RULES, "distinct"),
        (panel, [], RULES, "at least one"),
        (panel, ["EU"], {"price": Combine.SUM}, r"none for \['visits'\]"),
        (panel, ["EU"], {**RULES, "rain": Combine.SUM}, r"rules for no control \['rain'\]"),
        (lambda: panel(population=None), ["EU"], RULES, "needs the panel's population"),
        (
            lambda: panel(filled=np.eye(len(GEOS), WEEKS, dtype=bool)),
            ["DE", "FR", "rome"],
            RULES,
            "node 'DE': some of its geos have no row in 2 periods, from 2024-01-01",
        ),
    ],
)
def test_what_does_not_add_up_is_refused(
    make: Any, nodes: list[str], rules: dict[str, Combine], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        aggregate(make(), nodes, rules)


def read(population: dict[str, Any]) -> MediaPanel:
    weeks, cities = 3, ("berlin", "munich")
    return read_panel(
        {
            "date": np.tile(
                np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(weeks), 2
            ),
            "city": np.repeat(np.array(cities, dtype=object), weeks),
            "sales": np.arange(6, dtype=float) + 1,
            "tv": np.arange(6, dtype=float) % 4 + 1,
        },
        date="date",
        geo="city",
        kpi="sales",
        spend={"tv": "tv"},
        cadence=Cadence.WEEK,
        kpi_unit=Revenue(),
        currency="EUR",
        population=population,
    )


def test_populations_are_read_in_the_panel_s_order_and_strays_dropped_with_a_warning() -> None:
    out = read({"munich": 1.5e6, "berlin": 3.7e6, "paris": 2.1e6})
    assert out.population is not None
    np.testing.assert_array_equal(out.population, [3.7e6, 1.5e6])
    (warning,) = others(out.issues)
    assert warning.code is IssueCode.POPULATION
    assert warning.severity is Severity.WARNING
    assert "['paris']" in warning.message


@pytest.mark.parametrize(
    ("population", "match"),
    [
        ({"berlin": 3.7e6}, "geo 'munich' has no population"),
        (
            {"berlin": 3.7e6, "munich": 0.0},
            "population of geo 'munich', 0.0, is not finite and positive",
        ),
        ({"berlin": np.inf, "munich": 1.0}, "is not finite and positive"),
        ({"berlin": "many", "munich": 1.0}, "population of geo 'berlin', 'many', is not a number"),
        ({"berlin": True, "munich": 1.0}, "is not a number"),
    ],
)
def test_a_population_that_is_missing_or_not_positive_is_an_error(
    population: dict[str, Any], match: str
) -> None:
    with pytest.raises(PanelError, match=match) as excinfo:
        read(population)
    assert {i.code for i in excinfo.value.issues} == {IssueCode.POPULATION}
