"""Convergence diagnostics: CmdStan's ``stansummary`` on seven sets of draws, each a case the
estimators treat apart, and the E-BFMI's definition."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mixdyn.inference.diagnostics import _ess, _split, diagnose, e_bfmi

STAN = json.loads(
    (Path(__file__).parent / "fixtures" / "oracles" / "stan_diagnostics.json").read_text()
)


def _tails(draws: np.ndarray) -> list[tuple[np.ndarray, bool]]:
    """Each tail's indicator over the draws, at or below the split chains' 5 % quantile and at or
    above their 95 %, and whether it is one value throughout some split chain."""
    half = draws.shape[1] // 2
    chains = np.concatenate([draws[:, :half], draws[:, draws.shape[1] - half :]])
    low, high = np.quantile(chains, (0.05, 0.95))
    out = []
    for split, full in ((chains <= low, draws <= low), (chains >= high, draws >= high)):
        out.append((full.astype(float), any(len(set(row)) == 1 for row in split)))
    return out


def _constant_tail(draws: np.ndarray) -> bool:
    return any(constant for _, constant in _tails(draws))


@pytest.mark.parametrize("name", sorted(STAN["cases"]))
def test_the_diagnostics_are_stansummary_s(name: str) -> None:
    """R-hat and the bulk effective sample size in every case. The tail's is the smaller of its two
    indicators' bulk sizes, the rank-normalised indicator being an affine map of it; where an
    indicator is one value throughout a split chain, ``stansummary`` reads that indicator as half
    the draws' number."""
    case = STAN["cases"][name]
    draws = np.array(case["draws"])
    stan = case["stansummary"]
    ours = diagnose(draws)
    assert ours.rhat[0] == pytest.approx(stan["R_hat"], rel=1e-12)
    assert ours.ess_bulk[0] == pytest.approx(stan["ESS_bulk"], rel=1e-10)
    sizes = [diagnose(indicator).ess_bulk[0] for indicator, _ in _tails(draws)]
    assert ours.ess_tail[0] == pytest.approx(min(sizes), rel=1e-12)
    half_draws = draws.shape[0] * (draws.shape[1] // 2)
    theirs = [
        half_draws if constant else size
        for (_, constant), size in zip(_tails(draws), sizes, strict=True)
    ]
    assert stan["ESS_tail"] == pytest.approx(min(theirs), rel=1e-10)


def test_chains_that_disagree_have_few_effective_draws_in_their_tails() -> None:
    for name in ("shifted", "scaled"):
        draws = np.array(STAN["cases"][name]["draws"])
        assert _constant_tail(draws)
        assert diagnose(draws).ess_tail[0] < 0.1 * draws.size


def test_the_cases_cover_what_the_estimators_treat_apart() -> None:
    """The cap on antithetic chains, ties, an odd number of draws and a constant tail indicator."""
    cases = {name: np.array(c["draws"]) for name, c in STAN["cases"].items()}
    capped = cases["ar_neg"].size * np.log10(cases["ar_neg"].size)
    assert STAN["cases"]["ar_neg"]["stansummary"]["ESS_bulk"] == pytest.approx(capped)
    assert len(np.unique(cases["ties"])) < 20
    assert cases["odd"].shape[1] % 2 == 1
    assert sum(_constant_tail(x) for x in cases.values()) >= 2
    assert sum(not _constant_tail(x) for x in cases.values()) >= 3


def test_several_quantities_are_each_their_own() -> None:
    a = np.array(STAN["cases"]["ar_pos"]["draws"])
    b = np.array(STAN["cases"]["cauchy"]["draws"])
    both = diagnose(np.stack([a, b], axis=2))
    for k, x in enumerate((a, b)):
        alone = diagnose(x)
        assert both.rhat[k] == alone.rhat[0]
        assert both.ess_bulk[k] == alone.ess_bulk[0]
        assert both.ess_tail[k] == alone.ess_tail[0]


def test_a_constant_quantity_has_no_diagnostics() -> None:
    d = diagnose(np.ones((4, 100)))
    assert np.isnan(d.rhat[0])
    assert np.isnan(d.ess_bulk[0])
    assert np.isnan(d.ess_tail[0])


@pytest.mark.parametrize("draws", [4, 5, 6, 7, 8])
def test_split_chains_of_fewer_than_four_draws_have_no_effective_sample_size(draws: int) -> None:
    """Under eight draws a chain the split chains hold fewer than four, too few lags for Geyer's
    sequence, and CmdStan reads no effective sample size; the R-hat is read whatever the length."""
    d = diagnose(np.random.default_rng(draws).standard_normal((4, draws)))
    assert np.isfinite(d.rhat[0])
    for ess in (d.ess_bulk[0], d.ess_tail[0]):
        assert np.isnan(ess) == (draws < 8)


@pytest.mark.parametrize("power", [-900, -600, 600, 900])
def test_the_effective_sample_size_reads_no_scale(power: int) -> None:
    """Draws scaled by a power of two read the effective sample size of the draws themselves, bit
    for bit, though at these scales their squared deviations pass the floats, under or over; and
    a scale not a power of two, to rounding."""
    x = _split(np.random.default_rng(abs(power)).standard_normal((4, 200)))
    assert _ess(np.ldexp(x, power)) == _ess(x)
    assert _ess(10.0 ** (power / 4.0) * x) == pytest.approx(_ess(x), rel=1e-12)


def test_too_few_draws_are_refused() -> None:
    with pytest.raises(ValueError, match="3 draws a chain"):
        diagnose(np.zeros((4, 3)))


def test_e_bfmi_is_the_squared_steps_over_the_variance() -> None:
    """A random walk's energy steps are small against its spread; independent draws' are twice
    their variance."""
    rng = np.random.default_rng(0)
    walk = np.cumsum(rng.standard_normal(10_000))
    white = rng.standard_normal(10_000)
    assert e_bfmi(walk)[0] < 0.01
    assert e_bfmi(white)[0] == pytest.approx(2.0, rel=0.05)
    two = e_bfmi(np.stack([walk, white]))
    assert two[0] == e_bfmi(walk)[0]
    assert two[1] == e_bfmi(white)[0]
