"""The tested rung's candidates for the claim: the static Bayesian arm's `stacked` plan, its two
families weighed, its Hill's slope prior and its kernels' length as each candidate states (ADR
0081).

`dlm two-stage` took its curves from the static arm's two fits, the family of the larger stacking
weight, and read the lift tests a second time in a DLM; it planned as `stacked` plans, and its
intervals covered less. A candidate reads the tests once, in the static fits, and plans as
`stacked` does (`scorecard.arms.bayes.Bayes`): the plan of the posterior's expected return,
`mixdyn.decisions.plan_posterior` over `bayes.DRAWS` of its draws on `bayes.GRID` spends a
channel. A candidate (`Candidate`) states three settings, each by default the `bayes` arm's:

* `rule`, how the two families' fits are weighed: `"stacking"`, by stacking over every
  observation the fits read; or `"lift rows"`, the whole weight on the family whose leave-one-out
  density of the lift rows is the larger (`mixdyn.decisions.lift_row_pick`), by stacking where
  the export holds no test;
* `slope`, the Hill's slope prior's location and scale (`mixdyn.models.bayes_mmm.HillCurve`);
* `kernel`, the kernels' length in weeks (`mixdyn.models.bayes_mmm.BayesSpec.kernel`), None the
  panel's.

An arm (`Tested`) is handed its candidates and draws each fit they read once, each from the seed
the `bayes` arm draws its fits from, so a candidate's plan is the same whichever candidates share
its arm. The runner holds each candidate in one arm, as it holds each variant in one arm:
`bayes-concave` holds both concave candidates, which read the same two fits, so the lift rows'
pick costs a plan and not a fit, and `bayes-lift-pick` holds the pick over the default prior; a
screen that compares candidates on one fit of each hands them to one arm. A variant's record is
`stacked`'s, its forecast, response, flags and effective sample size its own posterior's. Its
settings add the candidate's, the rule it read, and each fit it reads with the CPU seconds the fit
took, so that a candidate's cost alone is its fits' and its plan's; its flags add the Hill's
stacking weight and each family's leave-one-out density of the lift rows.

A smoke run samples as the `bayes` arm's does.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from mixdyn.decisions import lift_row_pick, stacked
from mixdyn.inference import loo
from mixdyn.models.bayes_mmm import HillCurve, TanhCurve
from scorecard.arms import bayes
from scorecard.arms.base import Planned

if TYPE_CHECKING:
    from collections.abc import Sequence

    from causaldyn_bench.scorecard.observe import Observation

    from mixdyn.models.bayes_mmm import BayesFit, Saturation

type Rule = Literal["stacking", "lift rows"]
RULES: tuple[Rule, ...] = ("stacking", "lift rows")
SLOPE = (0.0, 0.5)
"""The Hill's slope prior's location and scale by default, the `bayes` arm's."""
CONCAVE = (-0.3, 0.4)
"""A concave-leaning slope prior: median 0.74, 23 % of its mass above a slope of 1, 3.9 % above
1.5 and 0.7 % above 2."""


@dataclass(frozen=True, slots=True)
class Fit:
    """One fit a candidate reads: its family, `"hill"` or `"tanh"`, the Hill's slope prior, None
    for the `tanh`, and its kernels' length, None the panel's."""

    family: str
    slope: tuple[float, float] | None
    kernel: int | None

    def curve(self) -> Saturation:
        return TanhCurve() if self.slope is None else HillCurve(*self.slope)

    def label(self) -> str:
        """The fit's name in a record: its family, its slope prior and its kernel."""
        prior = "" if self.slope is None else f" LogNormal({self.slope[0]:g}, {self.slope[1]:g})"
        kernel = "the panel's" if self.kernel is None else f"{self.kernel} weeks"
        return f"{self.family}{prior}, kernel {kernel}"


@dataclass(frozen=True, slots=True)
class Candidate:
    """A `stacked` plan of the static arm under its settings; see the module. `name` is its
    variant's.

    Raises:
        ValueError: on a rule not in `RULES`, a slope prior `HillCurve` refuses, or a kernel's
            length that is not a whole number of weeks, at least 1.
    """

    name: str
    rule: Rule = "stacking"
    slope: tuple[float, float] = SLOPE
    kernel: int | None = None

    def __post_init__(self) -> None:
        if self.rule not in RULES:
            msg = f"a candidate weighs the families by one of {RULES}, not {self.rule!r}"
            raise ValueError(msg)
        HillCurve(*self.slope)
        if self.kernel is not None and (
            isinstance(self.kernel, bool) or not isinstance(self.kernel, int) or self.kernel < 1
        ):
            msg = f"a kernel's length is a whole number of weeks, at least 1, not {self.kernel!r}"
            raise ValueError(msg)

    def fits(self) -> dict[str, Fit]:
        """The fits it reads, by family, in `bayes.FAMILIES`' order."""
        return {
            "hill": Fit("hill", self.slope, self.kernel),
            "tanh": Fit("tanh", None, self.kernel),
        }

    def settings(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "slope prior": list(self.slope),
            "kernel": "the panel's" if self.kernel is None else self.kernel,
        }


LIFT_PICK = Candidate("stacked, lift-row pick", rule="lift rows")
CONCAVE_HILL = Candidate("stacked, concave hill", slope=CONCAVE)
CONCAVE_PICK = Candidate("stacked, concave hill, lift-row pick", rule="lift rows", slope=CONCAVE)
CANDIDATES = (LIFT_PICK, CONCAVE_HILL, CONCAVE_PICK)
"""The tested rung's candidates, each held by one arm of `scorecard.run.ARMS`."""


@dataclass(frozen=True, slots=True, eq=False)
class Fitted:
    """Each fit the arm's candidates read, the CPU seconds each took, and the sampler that drew
    them."""

    fits: dict[Fit, BayesFit]
    cpu: dict[Fit, float]
    sampler: bayes.Sampler


class Tested:
    """The arm of `candidates`; see the module.

    Raises:
        ValueError: on no candidate, or two of one name.
    """

    tests = True
    cells = False

    def __init__(self, candidates: Sequence[Candidate]) -> None:
        names = [c.name for c in candidates]
        if not names or len(set(names)) != len(names):
            msg = f"an arm holds one candidate or more, each named once: {names}"
            raise ValueError(msg)
        self.candidates = {c.name: c for c in candidates}
        self.variants = tuple(names)

    def fit(self, observation: Observation, *, smoke: bool) -> Fitted:
        sampler = bayes.SMOKE if smoke else bayes.SAMPLER
        fits, cpu = {}, {}
        for candidate in self.candidates.values():
            for fit in candidate.fits().values():
                if fit in fits:
                    continue
                began = time.process_time()
                drawn = bayes.family_fits(
                    observation, sampler, curves={fit.family: fit.curve()}, kernel=fit.kernel
                )
                fits[fit], cpu[fit] = drawn[fit.family], time.process_time() - began
        return Fitted(fits, cpu, sampler)

    def plan(self, fit: Fitted, variant: str, observation: Observation) -> Planned:
        candidate = self.candidates[variant]
        read = candidate.fits()
        pair = {family: fit.fits[f] for family, f in read.items()}
        picked = lift_row_pick(list(pair.values())) if candidate.rule == "lift rows" else None
        made = bayes.fitted(observation, pair, fit.sampler, given=picked)
        planned = bayes.Bayes().plan(made, bayes.RESPONDS, observation)
        stacking = stacked(list(pair.values()), draws=1).stacking
        flags = dict(planned.flags) | {"stacking hill weight": float(stacking[0])}
        flags |= {f"{family} lift-row elpd": value for family, value in _lift_rows(pair).items()}
        settings = dict(planned.settings) | {
            "candidate": candidate.settings(),
            "rule read": "stacking" if picked is None else "lift rows",
            "fits": {
                f.label(): {"family": family, "cpu_seconds": fit.cpu[f]}
                for family, f in read.items()
            },
        }
        return dataclasses.replace(planned, flags=flags, settings=settings)


def _lift_rows(fits: dict[str, BayesFit]) -> dict[str, float]:
    """Each fit's leave-one-out log predictive density of the observations past the KPI's, by
    PSIS-LOO; none where the fits read none."""
    out = {}
    for family, one in fits.items():
        kpi = int(one.fitted.sum())
        if one.log_likelihood.shape[2] > kpi:
            out[family] = loo(one.log_likelihood[:, :, kpi:]).elpd
    return out
