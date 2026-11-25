"""Decisions on a panel's media: each channel's returns and costs per incremental KPI, and budget
plans over every geo's channels, or for a posterior's expected return, its worst share or its least
over effects off by a common factor, and the factor that erases a plan's gain; a plan read as a
table of channels, spend scenarios compared with their carry-out, and a flighted schedule scored at
every draw."""

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
    lift_row_pick,
    plan_breakdown,
    plan_cvar,
    plan_posterior,
    plan_robust,
    readings,
    stacked,
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
    "lift_row_pick",
    "media_difference",
    "move_between_geos",
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
    "stacked",
    "worths",
]
