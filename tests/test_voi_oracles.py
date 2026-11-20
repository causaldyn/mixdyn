"""The value-of-information oracles, Maxima's closed form and a quadrature, hold together: the
conjugate one's exact rationals are its floats, every identity it checked held and every control
failed, the test it records is worth its closed form, no more than perfect information, and two
tests on its instance are worth theirs, apart and together, in rationals computed by hand; the
quadrature one's test, over CHC's exact plan, is worth no more than perfect information either, its
integrals are the sums of their pieces, each piece ends where the plan's first-order condition puts
a bound, and its Monte Carlo cross-checks lie within 4.5 of their standard errors; and the brute
force's CmdStan fits converged, and it refitted every outer draw of the four tests and their six
pairs."""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ORACLES = Path(__file__).parent / "fixtures" / "oracles"
CONJUGATE = json.loads((ORACLES / "voi_conjugate.json").read_text())
QUADRATURE = json.loads((ORACLES / "voi_quadrature.json").read_text())
BRUTEFORCE = json.loads((ORACLES / "voi_bruteforce.json").read_text())


def _rationals(record: object) -> Iterator[tuple[str, float]]:
    """Every exact entry Maxima wrote as a rational, beside its float."""
    if isinstance(record, dict):
        if "exact" in record and "value" in record:
            exact, value = record["exact"], record["value"]
            if isinstance(exact, list):
                for row, values in zip(exact, value, strict=True):
                    yield from zip(row, values, strict=True)
            elif "sqrt" not in exact and "%e" not in exact:
                yield exact, value
            return
        for entry in record.values():
            yield from _rationals(entry)
    elif isinstance(record, list):
        for entry in record:
            yield from _rationals(entry)


def test_the_conjugate_oracles_rationals_are_its_floats() -> None:
    pairs = list(_rationals({key: CONJUGATE[key] for key in ("m4", "m5", "m6")}))
    assert len(pairs) > 40
    for exact, value in pairs:
        assert float(Fraction(exact)) == value, exact


def test_every_conjugate_identity_held_and_every_control_failed() -> None:
    assert CONJUGATE["checks"]
    assert all(check["holds"] for check in CONJUGATE["checks"])
    assert CONJUGATE["controls"]
    assert all(control["fails"] for control in CONJUGATE["controls"])


def test_the_conjugate_test_is_worth_its_closed_form_and_no_more_than_perfect_information() -> None:
    """EVSI 1/100 and EVPI 1/10 on the rational instance; the realised reduction's mean is EVSI,
    and its 5 % and 95 % quantiles by Imhof's inversion are the distribution's read directly."""
    m4, m5 = CONJUGATE["m4"], CONJUGATE["m5"]
    evsi, evpi = Fraction(m4["evsi"]["exact"]), Fraction(m4["evpi"]["exact"])
    assert (evsi, evpi) == (Fraction(1, 100), Fraction(1, 10))
    assert 0 < evsi <= evpi
    assert Fraction(m5["mean"]["exact"]) == evsi
    for quantile in m5["quantiles"]:
        assert abs(quantile["imhof"] - quantile["direct"]) <= 1e-10


type _Matrix = list[list[Fraction]]


def _product(a: _Matrix, b: _Matrix) -> _Matrix:
    return [
        [
            sum((x * y for x, y in zip(row, column, strict=True)), Fraction(0))
            for column in zip(*b, strict=True)
        ]
        for row in a
    ]


def _transpose(a: _Matrix) -> _Matrix:
    return [list(column) for column in zip(*a, strict=True)]


def _sum(a: _Matrix, b: _Matrix, sign: int = 1) -> _Matrix:
    return [[x + sign * y for x, y in zip(p, q, strict=True)] for p, q in zip(a, b, strict=True)]


def _inverse(a: _Matrix) -> _Matrix:
    """The inverse of a matrix of one or two rows, by its adjugate."""
    if len(a) == 1:
        return [[1 / a[0][0]]]
    (p, q), (r, s) = a
    det = p * s - q * r
    return [[s / det, -q / det], [-r / det, p / det]]


def _fractions(key: str, record: dict[str, Any]) -> _Matrix:
    return [[Fraction(x) for x in row] for row in record[key]["exact"]]


def test_two_conjugate_tests_are_worth_their_closed_forms_in_rationals_by_hand() -> None:
    """M6's values of the sum and the difference of M4's parameters, each read with noise of
    variance 1/2, apart and together, are `tr(W S0 J' (J S0 J' + N)^-1 J S0) / 2` in exact
    rationals computed apart from Maxima; the pair's posterior covariance is its `S1`, and the
    second's value after the first is the pair's less the first's."""
    m4, m6 = CONJUGATE["m4"], CONJUGATE["m6"]
    prior, weight = _fractions("S0", m4), _fractions("W", m4)
    rows, noise = _fractions("J", m6), _fractions("N", m6)

    def worth(chosen: list[int]) -> tuple[Fraction, _Matrix]:
        j = [rows[t] for t in chosen]
        gain = _product(prior, _transpose(j))
        spread = _sum(_product(j, gain), [[noise[a][b] for b in chosen] for a in chosen])
        drop = _product(_product(gain, _inverse(spread)), _transpose(gain))
        reach = _product(weight, drop)
        return (reach[0][0] + reach[1][1]) / 2, _sum(prior, drop, -1)

    (first, _), (second, _), (pair, posterior) = worth([0]), worth([1]), worth([0, 1])
    assert (first, second, pair) == (Fraction(1, 100), Fraction(1, 15), Fraction(19, 280))
    for key, value in (("first", first), ("second", second), ("pair", pair)):
        assert Fraction(m6[key]["exact"]) == value
    assert Fraction(m6["second after the first"]["exact"]) == pair - first == Fraction(81, 1400)
    assert _fractions("S1", m6) == posterior
    assert max(first, second) < pair < first + second


def test_the_quadrature_test_is_worth_no_more_than_perfect_information() -> None:
    """The test's direction is ``S0 J' / sqrt(J S0 J' + se^2)`` with ``J = (s gamma, 0)``, and its
    worth 0 < EVSI <= EVPI, each the sum of its pieces to their errors."""
    world, test = QUADRATURE["world"], QUADRATURE["test"]
    covariance = np.array(world["covariance"])
    J = np.array([test["share"] * test["gamma"], 0.0])
    assert test["J"] == pytest.approx(J.tolist(), rel=1e-15, abs=0.0)
    direction = covariance @ J / math.sqrt(float(J @ covariance @ J) + test["se"] ** 2)
    assert test["direction"] == pytest.approx(direction.tolist(), rel=1e-12)
    evsi, evpi = QUADRATURE["evsi"], QUADRATURE["evpi"]
    assert 0.0 < evsi["value"] <= evpi["value"]
    for integral in (evsi, evpi):
        assert sum(p["value"] for p in integral["pieces"]) == pytest.approx(
            integral["value"], abs=integral["error"]
        )
        assert integral["error"] <= 1e-9 * integral["value"]


def test_the_quadratures_pieces_end_where_a_bound_starts_to_bind() -> None:
    """Each breakpoint bisected on CHC's plan is the first-order condition's root at the bound the
    plan reaches there, and the pieces of each integral run between them."""
    breakpoints = QUADRATURE["breakpoints"]
    bounds = {QUADRATURE["world"]["lower"][0], QUADRATURE["world"]["upper"][0]}
    for line, integral in (("z", "evsi"), ("rho", "evpi")):
        found = breakpoints[line]
        assert len(found) == 2
        for b in found:
            assert b["at"] == pytest.approx(b["kkt"], abs=1e-9)
            assert b["spend"] in bounds
            assert b["from"] != b["to"]
        ends = [p["to"] for p in QUADRATURE[integral]["pieces"][:-1]]
        assert ends == [b["at"] for b in found]


def test_the_quadratures_monte_carlo_lies_within_its_errors() -> None:
    estimates = [
        (QUADRATURE["evsi"]["value"], mc) for mc in QUADRATURE["evsi"]["monte_carlo"].values()
    ]
    estimates.append((QUADRATURE["evpi"]["value"], QUADRATURE["evpi"]["monte_carlo"]))
    for exact, mc in estimates:
        assert abs(mc["mean"] - exact) <= 4.5 * mc["error"]


def test_the_brute_forces_fits_converged() -> None:
    """CmdStan's base fit has no divergence, and no refit diverges on more than 1 % of its draws;
    every fit's largest R-hat is at most 1.01."""
    refits, provenance = BRUTEFORCE["refits"], BRUTEFORCE["provenance"]
    assert refits["base"]["divergent"] == 0
    assert max(refits["divergent"]) <= 0.01 * provenance["chains"] * provenance["draws"]
    assert max(refits["base"]["rhat"], *refits["rhat"]) <= 1.01


def test_the_brute_force_refitted_every_outer_draw_of_the_four_tests_and_their_six_pairs() -> None:
    sets, outer = BRUTEFORCE["sets"], BRUTEFORCE["provenance"]["outer"]
    pairs = [list(p) for p in itertools.combinations(range(4), 2)]
    assert [s["tests"] for s in sets] == [[t] for t in range(4)] + pairs
    assert len(BRUTEFORCE["refits"]["rhat"]) == len(sets) * outer
    for record in sets:
        assert len(record["refitted"]) == len(record["reweight"]) == outer
        assert np.all(np.isfinite(record["refitted"]))
