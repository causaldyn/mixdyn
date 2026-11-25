"""Decisions on a panel's media: each channel's returns and costs per incremental KPI, and budget
plans over every geo's channels, or for a posterior's expected return, its worst share or its least
over effects off by a common factor, and the factor that erases a plan's gain; a plan read as a
table of channels, spend scenarios compared with their carry-out, and a flighted schedule scored at
every draw."""

import warnings

from chc.allocation import MarginalReturnTarget, ReturnOnSpendTarget, ReturnTarget

from mixdyn.decisions.flighting import FlightPlan, Recommendation, plan_flighting
from mixdyn.decisions.geo_plan import GeoLimits, GeoPlan, Goal, Range, plan_geos
from mixdyn.decisions.plan_table import PlanTable, Spread, plan_table, ranges_from_current
from mixdyn.decisions.posterior import (
    Breakdown,
    CvarPlan,
    PosteriorPlan,
    Readings,
    RobustPlan,
    Worths,
    breakdown,
    pick_by_lift_tests,
    plan_breakdown,
    plan_cvar,
    plan_posterior,
    plan_robust,
    readings,
    stacked_readings,
    worths,
)
from mixdyn.decisions.returns import Costs, Returns, Undefined, costs, portfolio_roi, returns
from mixdyn.decisions.scenarios import (
    ResponseCurves,
    Scenario,
    ScenarioComparison,
    compare_scenarios,
    media_difference,
    move_between_geos,
    response_curves,
)
from mixdyn.decisions.scored import (
    MarginPlan,
    PooledGeoPlan,
    plan_anchored_margins,
    plan_margins,
    plan_pooled_geos,
)

__all__ = [
    "Breakdown",
    "Costs",
    "CvarPlan",
    "FlightPlan",
    "GeoLimits",
    "GeoPlan",
    "Goal",
    "MarginPlan",
    "MarginalReturnTarget",
    "PlanTable",
    "PooledGeoPlan",
    "PosteriorPlan",
    "Range",
    "Readings",
    "Recommendation",
    "ResponseCurves",
    "ReturnOnSpendTarget",
    "ReturnTarget",
    "Returns",
    "RobustPlan",
    "Scenario",
    "ScenarioComparison",
    "Spread",
    "Undefined",
    "Worths",
    "breakdown",
    "compare_scenarios",
    "costs",
    "media_difference",
    "move_between_geos",
    "pick_by_lift_tests",
    "plan_anchored_margins",
    "plan_breakdown",
    "plan_cvar",
    "plan_flighting",
    "plan_geos",
    "plan_margins",
    "plan_pooled_geos",
    "plan_posterior",
    "plan_robust",
    "plan_table",
    "portfolio_roi",
    "ranges_from_current",
    "readings",
    "response_curves",
    "returns",
    "stacked_readings",
    "worths",
]

_DEPRECATED = {
    "lift_row_pick": ("pick_by_lift_tests", "1.0.0"),
    "stacked": ("stacked_readings", "1.0.0"),
}
"""Each name that is leaving the package, with the name that replaces it and the release it leaves
in. It still imports until then, with a `DeprecationWarning`."""


def __getattr__(name: str) -> object:
    if name in _DEPRECATED:
        replacement, leaves = _DEPRECATED[name]
        warnings.warn(
            f"mixdyn.decisions.{name} is renamed mixdyn.decisions.{replacement}; the old name"
            f" leaves in {leaves}",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[replacement]
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
