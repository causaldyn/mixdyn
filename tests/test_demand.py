"""A demand proxy in the one role a model gives it (ADR 0074): the queries a control where they
confound search, left out where they mediate the other media, and what each channel's return then
is, on a world whose search buys against the demand the queries measure."""

from __future__ import annotations

import functools

import numpy as np
import pytest

from mixdyn import NATIONAL
from mixdyn.models.bayes_mmm import BayesFit, BayesMedia, BayesSpec, TanhCurve, fit_bayes
from mixdyn.models.demand import DemandProxy
from mixdyn.models.ridge_mmm import Control, FourierSeason
from search_demand_world import CHANNELS, SearchDemandWorld, world


def test_the_role_sets_what_the_model_reads_and_what_each_return_is() -> None:
    confounder = DemandProxy("queries", "confounder", ("search",))
    mediator = DemandProxy("queries", "mediator", ("search",))
    assert confounder.controls() == (Control("queries", lower=0.0),)
    assert mediator.controls() == ()
    assert [confounder.reading(c) for c in CHANNELS] == ["adjusted", "direct"]
    assert [mediator.reading(c) for c in CHANNELS] == ["confounded", "total"]


def test_a_proxy_states_its_role_and_what_it_confounds() -> None:
    with pytest.raises(TypeError):
        DemandProxy("queries", searched=("search",))  # ty: ignore[missing-argument]
    with pytest.raises(ValueError, match="role"):
        DemandProxy("queries", "control", ("search",))  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="at least one channel"):
        DemandProxy("queries", "confounder", ())
    with pytest.raises(ValueError, match="at least one channel"):
        DemandProxy("queries", "confounder", ("queries",))


@functools.cache
def _fit(seed: int, role: str) -> tuple[BayesFit, SearchDemandWorld]:
    w = world(seed)
    proxy = DemandProxy("queries", role, ("search",))  # ty: ignore[invalid-argument-type]
    spec = BayesSpec(
        media=tuple(BayesMedia(c) for c in CHANNELS),
        controls=proxy.controls(),
        season=FourierSeason(52.0, 1),
        saturation=TanhCurve(),
    )
    return fit_bayes(w.panel, NATIONAL, spec, seed=seed, chains=2, warmup=500, draws=500), w


def _interval(fit: BayesFit, channel: str) -> tuple[float, float, float]:
    roi = fit.draws["roi"][..., CHANNELS.index(channel)].reshape(-1)
    low, median, high = np.quantile(roi, [0.05, 0.5, 0.95])
    return float(low), float(median), float(high)


def test_the_queries_as_a_control_hold_search_s_return_and_left_out_inflate_it() -> None:
    """Seed 0 of the search-demand world, fitted with the queries in each role. As the confounder,
    a control, search's 90 % interval holds its true return. As the mediator, left out, search is
    credited with the demand it bought against: its interval lies above the truth at 10 of 10
    seeds in the study (ADR 0074) and above twice it at 8, its lower end 4.6 times the truth
    here."""
    adjusted, w = _fit(0, "confounder")
    low, _, high = _interval(adjusted, "search")
    assert low <= w.search <= high
    confounded, _ = _fit(0, "mediator")
    low, _, _ = _interval(confounded, "search")
    assert low > 2.0 * w.search
