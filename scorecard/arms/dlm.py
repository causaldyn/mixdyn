"""The engine's dynamic model over media as an arm: one search of the export's panel for each
channel's carryover and curve and the discounts, and the plans made from the specs it keeps.

The model (`mixdyn.models.dlm_media`, ADR 0020) is a level whose discount the search chooses, a
yearly season of 3 harmonics held still, each control an effect, and each channel an effect whose
coefficient walks, held at or above 0, on its spend carried over and saturated by its transform,
the kernel the export's length. `search_media` chooses each channel's retention, half-saturation
point and curve on `TRANSFORMS` and the discounts on `DISCOUNTS`, the trend's and each effect's on
its own (ADR 0026), by the one-step log-likelihood, refined once, keeps every spec a
likelihood-ratio test at 0.95 does not reject, and fits `FITTED` of them at evenly spaced ranks
of their likelihood (ADR 0023). CHOICE: the search starts at a retention of 0.5, the
half-saturation point at the channel's mean positive spend and a `tanh`, so the arm reads no other
fit. It reads the export's lift tests (`tests`), each row as the static arm reads it
(`scorecard.arms.panel.estimates`) and placed at its readout's weeks, in every spec's likelihood and
every fitted spec's posterior (`search_media`'s `estimates`, ADR 0038), so the harness runs it at
each rung on its own. `DlmUntested` is the same arm handed no test, fitted once for every rung:
what reading the tests is worth is the one's regret less the other's. On a geo export the harness
hands either the export's market (`scorecard.arms.panel.market`, ADR 0043): it fits the geos'
sums, reads each test in its geos' share of the market, and plans one weekly spend a channel.

The variants, each planned against the export's box and budget, the history carried in:

* `dlm plug-in`: `chc.allocation.allocate` on the best spec's channels at the last week, the
  filter's coefficients there;
* `dlm mixture`: `chc.allocation.cvar_allocate` at level 1, the most expected gain over the
  status quo across `DRAWS` readings spread evenly over the fitted specs (`DlmChoice.readings`);
* `dlm cvar 0.1`: the same readings at level 0.1;
* `dlm cvar 0.1 vs equal`: the same readings at level 0.1 over the bench's equal split
  (`scorecard.arms.panel.equal_split`), not the status quo (ADR 0044).

Each claims its plan's gain over the status quo, the mean and the central 90 % interval of the
gain over those readings. Its flags are the specs kept, those fitted and how many of these drew
their signed coefficients from chains that did not mix; the best spec's discounts, the trend's and
each effect's; whether the best sits at an end of a grid where its parameter goes on past it
(`DlmChoice.edges`), each discount and each channel's retention and half-saturation point, which
has found a limit there and not a point; how many such ends the kept set reaches; the least bulk
effective sample size of the fitted specs' signed chains; and the share of the forecast's paths
that kept every sign. A plan's effective sample size is that of the chains it reads: the best
spec's for the plug-in, whose channels are the means of its draws, and the fitted specs' least for
the others, whose readings every fitted spec's draws share.

`dlm mixture`'s record carries the fit's forecast and its response, the other variants' neither,
one fit for the four. The forecast is `DRAWS` draws of the quarter's weekly sales at the status
quo, the controls at the quarter's, spread over the fitted specs as the readings are
(`DlmChoice.forecast`): each spec's draws of its posterior at the last week carried through the
quarter, each channel's coefficient held at or above 0 at every week, and the history's spend
carried into each channel's column. The response (`scorecard.arms.mapped`) is `DRAWS` draws spread
over the fitted specs as the readings are, each a
draw of its spec's posterior truncated to the signs, its channels its spec's transforms and each
coefficient its path over the history's weeks. Beside them is each draw's mean split into its parts
as the filter reads it (`chc.dlm.decompose`): a channel's part is its coefficient times the column
the spec was fitted on, which the bench holds the transforms to. The draws come from a stream of
the export's seed apart from the readings'.

A smoke run searches a grid of two retentions, two half-saturation points and the `tanh`.

`DlmEvents`, `dlm-events`, is the arm for no lift test (ADR 0085): the same search on the export's
panel with the recurring events `mixdyn.data.recurring_events` reads off the promotion flag and the
price (`INDICATORS`) in place of the flag, each event a static effect whose discount the search
holds at 1 (`PerEffect.held`), the price a control as before; and the search's prior on each
channel's half-saturation point, log-normal of scale `HALF_SATURATION_PRIOR` about its mean positive
spend. Where no event is read, the flag stays. Its four variants plan as `Dlm`'s, and each claims as
its gain the mean of the worst tenth of its readings' gains (`claimed`), not their mean. It reads no
test, so the harness fits it once for every rung, and its plan at a rung with tests is k = 0's.
Its flags add the events read, their weeks and the classes' separation; its settings name the
indicators, each event's weeks of the year, the control they replace, the prior and the claim.

`DlmTwoStage`, `dlm two-stage`, takes the curves from the static Bayesian arm instead of the
search (`mixdyn.models.dlm_media`'s two stages). Its first stage is the static arm's two fits of the
export, every lift row read (`scorecard.arms.bayes.family_fits`), stacked as that arm stacks them;
each channel's transform is the higher-weighted family's posterior medians
(`mixdyn.models.transforms_from_bayes`), the Hill's where the weights tie; or, where its `rule` is
`"lift rows"`, the family whose leave-one-out density of the lift rows is the larger
(`mixdyn.decisions.lift_row_pick`, ADR 0082), the higher-weighted where the export holds no test.
`DlmTwoStageLiftPick`, `dlm two-stage, lift-row pick`, is that arm. Its second stage is the
DLM on those transforms, every effect's discount 1 and the trend's chosen on `TWO_STAGE_DISCOUNTS`
by `mixdyn.models.choose_dlm`, the lift tests read in its likelihood and posterior as above. CHOICE:
every effect's discount is 1, since a coefficient that walks lets its last week drift from what the
tests pinned, and the search's grid lets its argmax land on curves a go-dark test cannot reject.
Its one plan is the plug-in's on the best spec at the last week, and its record carries the
forecast and the response as `dlm mixture`'s does, over the specs `choose_dlm` keeps. Its flags add
the first stage's sampler health and leave-one-out reading by family and the Hill's weight; its
settings name the family read and each channel's transform.

`DlmTwoStageLogged`, declared as `dlm-two-stage-logged` and off unless a run declares it, fits the
export as `DlmTwoStage` does and plans the secondary variant `dlm two-stage, logged range`, whose
curves are read only where the history spent; each variant keeps records of its own, so the arm
does not plan `dlm two-stage` again. Its box is the export's cut to each channel's logged
spend range, from the least to the most a week of the history spent on it (`logged_range`), so the
plan never extrapolates a curve past the spend that fitted it. Where the box already lies inside
that range, the two plans are one. Where the cut box cannot spend the quarter's budget, the variant
plans in the export's box, and its settings say so.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from chc.allocation import allocate, cvar_allocate
from chc.dlm import PosteriorDraws, decompose

from mixdyn import NATIONAL
from mixdyn.data import RecurringEvents, recurring_events
from mixdyn.decisions import lift_row_pick, stacked
from mixdyn.models import (
    DiscountGrid,
    DlmSpec,
    Effect,
    PerEffect,
    Season,
    Trend,
    choose_dlm,
    transforms_from_bayes,
)
from mixdyn.models.dlm_media import TransformGrid, search_media
from mixdyn.transforms import HillShape, TanhShape, Transform
from scorecard.arms import bayes
from scorecard.arms import panel as reading
from scorecard.arms.base import Planned
from scorecard.arms.mapped import held, mapped

if TYPE_CHECKING:
    from collections.abc import Sequence

    from causaldyn_bench.scorecard.observe import Observation
    from chc import response

    from mixdyn.models import DlmChoice, DlmFit, DlmForecastDraws

TRANSFORMS = TransformGrid(
    retentions=tuple(round(0.1 * k, 1) for k in range(10)),
    half_saturations=(0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0),
    shapes=(TanhShape(), HillShape(1.0), HillShape(2.0)),
)
SMOKE = TransformGrid((0.2, 0.6), (0.5, 2.0), (TanhShape(),))
DISCOUNTS = DiscountGrid(
    trend=(0.98, 0.99, 1.0), effects=PerEffect((0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99, 1.0))
)
REFINEMENTS, LEVEL = 1, 0.95
FITTED = 32
HARMONICS, YEAR = 3, 52.18
START_RETENTION = 0.5
DRAWS = 400
CVAR = 0.1
GAIN_LEVEL = 0.9
RESPONDS = "dlm mixture"
"""The variant of `Dlm` whose record carries the fit's forecast and response."""
TWO_STAGE = "dlm two-stage"
TWO_STAGE_DISCOUNTS = DiscountGrid(trend=(0.98, 0.99, 1.0))
"""The discounts `DlmTwoStage` chooses among: the trend's, as `DISCOUNTS` tries it."""
LOGGED = "dlm two-stage, logged range"
"""The variant of `DlmTwoStageLogged` planned within the logged spend range."""
LIFT_PICK = "dlm two-stage, lift-row pick"
"""The variant of `DlmTwoStageLiftPick`, its first stage's family picked by the lift rows."""
INDICATORS = {"promotion": 1, "price": -1}
"""The export's controls `DlmEvents` reads its recurring events off, each with its direction:
the promotion flag rises in a promoted week, the price falls."""
FLAG = "promotion"
"""The control `DlmEvents`' events replace: the flag's one coefficient is what they read apart."""
HALF_SATURATION_PRIOR = 1.0
"""`DlmEvents`' prior on each channel's half-saturation point: log-normal of this scale about
the channel's mean positive spend, as the static Bayesian model's `K_m`."""


@dataclass(frozen=True, slots=True, eq=False)
class Fitted:
    """The search's choice, its readings, and what a record keeps of it."""

    choice: DlmChoice
    readings: tuple[tuple[response.Channel, ...], ...]
    forecast: np.ndarray | None
    flags: dict[str, float | int | bool]
    settings: dict[str, Any]
    absent: dict[str, str]
    ess: float
    best_ess: float
    events: RecurringEvents | None = None


class Dlm:
    """The arm; see the module. Its variants are named plug-in, mixture, CVaR and CVaR over the
    equal split in that order, and the second's record carries the forecast and the response."""

    variants = ("dlm plug-in", RESPONDS, "dlm cvar 0.1", "dlm cvar 0.1 vs equal")
    tests = True
    cells = False
    events = False
    """Whether the arm reads recurring events in place of the promotion flag (`DlmEvents`)."""
    half_saturation_prior: float | None = None
    """The search's prior on each channel's half-saturation point, or None (`search_media`)."""
    claim: float | None = None
    """The level at which a plan claims its gain: the mean of that share of the readings' worst
    gains; None claims their mean."""

    def fit(self, observation: Observation, *, smoke: bool) -> Fitted:
        data = reading.panel(observation)
        tests = reading.estimates(observation)
        controls = tuple(observation.controls)
        found = None
        if self.events:
            indicators = {n: d for n, d in INDICATORS.items() if n in controls}
            if indicators:
                found = recurring_events(data, indicators)
            if found is not None and found.names:
                data = dataclasses.replace(data, events=found.days(data.dates))
                controls = tuple(c for c in controls if c != FLAG)
        spend = np.asarray(observation.spend, dtype=float)
        media = []
        for j, name in enumerate(observation.channels):
            column = spend[:, j]
            start = Transform(
                START_RETENTION,
                float(column[column > 0.0].mean()),
                TanhShape(),
                observation.kernel_length,
            )
            media.append(Effect(name, 0.99, sign="positive", transform=start))
        named = found.names if found is not None else ()
        spec = DlmSpec(
            effects=(
                *media,
                *(Effect(c, 0.99) for c in controls),
                *(Effect(e, 1.0) for e in named),
            ),
            trend=Trend(1, 0.99),
            season=Season(YEAR, HARMONICS, 1.0),
        )
        discounts = DISCOUNTS
        if named:
            assert isinstance(DISCOUNTS.effects, PerEffect)  # each effect's, as the arm searches
            # an event recurs once a year, so its coefficient is static (ADR 0069), not searched
            held = PerEffect(DISCOUNTS.effects.values, held=named)
            discounts = dataclasses.replace(DISCOUNTS, effects=held)
        grid = SMOKE if smoke else TRANSFORMS
        choice = search_media(
            data,
            NATIONAL,
            spec,
            grid,
            discounts,
            refinements=REFINEMENTS,
            level=LEVEL,
            fitted=FITTED,
            estimates=tests or None,
            half_saturation_prior=self.half_saturation_prior,
        )
        readings = choice.readings(DRAWS, observation.seed)
        flags = _flags(choice, observation.channels)
        ess = _ess(list(choice.kept.values()))
        flags["min ess bulk"] = ess
        forecast, absent = None, {}
        if observation.future_controls is None and controls:
            absent["forecast"] = "the export gives no controls over the quarter"
        else:
            drawn = _forecast(choice, observation, found)
            forecast = drawn.kpi
            flags["forecast acceptance"] = drawn.acceptance
        settings = {
            "transforms": _grid(grid),
            "discounts": dataclasses.asdict(DISCOUNTS),
            "refinements": REFINEMENTS,
            "level": LEVEL,
            "fitted": FITTED,
            "season_harmonics": HARMONICS,
            "start_retention": START_RETENTION,
            "readings": DRAWS,
            "gain_level": GAIN_LEVEL,
            "lift_rows_read": sum(len(rows) for rows in tests.values()),
        }
        if self.events:
            flags |= {
                "events": len(named),
                "event weeks": sum(len(w) for w in found.weeks) if found is not None else 0,
                "events separation": found.separation if found is not None else 0.0,
            }
            settings |= {
                "indicators": {n: d for n, d in INDICATORS.items() if n in observation.controls},
                "events": {n: list(w) for n, w in zip(named, found.weeks, strict=True)}
                if found is not None
                else {},
                "replaced": FLAG if named else None,
            }
        if self.half_saturation_prior is not None:
            settings["half_saturation_prior"] = self.half_saturation_prior
        if self.claim is not None:
            settings["claim"] = f"the mean of the readings' worst {self.claim:g} of gains"
        return Fitted(
            choice, readings, forecast, flags, settings, absent, ess, _ess([choice.fit]), found
        )

    def plan(self, fit: Fitted, variant: str, observation: Observation) -> Planned:
        box = {"lower": observation.lower, "upper": observation.upper}
        plug_in, mixture, _, over_equal = self.variants
        if variant == plug_in:
            channels = fit.choice.channels()
            weekly = allocate(
                [channels[c] for c in observation.channels],
                observation.budget,
                observation.planned,
                history=observation.spend,
                **box,
            ).spend
            settings: dict[str, Any] = {"planner": "allocate"}
            ess = fit.best_ess
        else:
            level = 1.0 if variant == mixture else CVAR
            equal = variant == over_equal
            weekly = cvar_allocate(
                fit.readings,
                observation.budget,
                observation.planned,
                level=level,
                against=reading.equal_split(observation) if equal else observation.status_quo,
                history=observation.spend,
                **box,
            ).spend
            settings = {
                "planner": "cvar_allocate",
                "level": level,
                "against": "equal split" if equal else "status quo",
            }
            ess = fit.ess
        gain = reading.gains(fit.readings, observation, weekly)
        low, high = np.quantile(gain, [(1.0 - GAIN_LEVEL) / 2.0, (1.0 + GAIN_LEVEL) / 2.0])
        responds = variant == mixture
        elsewhere = f"recorded under {mixture!r}"
        return Planned(
            weekly=np.asarray(weekly, dtype=float),
            gain=float(np.mean(gain)) if self.claim is None else claimed(gain, self.claim),
            gain_interval=(float(low), float(high)),
            forecast=fit.forecast if responds else None,
            response=_response(fit.choice, observation) if responds else None,
            flags=fit.flags,
            settings=fit.settings | settings,
            absent=fit.absent
            if responds
            else {"forecast": elsewhere} | fit.absent | {"response": elsewhere},
            ess=ess,
        )


class DlmEvents(Dlm):
    """The arm for no lift test: recurring events in place of the promotion flag, a prior on each
    channel's half-saturation point, and a claim at the CVaR's level (ADR 0085)."""

    variants = (
        "dlm events plug-in",
        "dlm events mixture",
        "dlm events cvar 0.1",
        "dlm events cvar 0.1 vs equal",
    )
    tests = False
    events = True
    half_saturation_prior = HALF_SATURATION_PRIOR
    claim = CVAR


class DlmUntested(Dlm):
    """The arm handed no lift test: the ablation that measures what reading them is worth."""

    variants = (
        "dlm untested plug-in",
        "dlm untested mixture",
        "dlm untested cvar 0.1",
        "dlm untested cvar 0.1 vs equal",
    )
    tests = False


class DlmTwoStage:
    """The arm on the static Bayesian arm's curves; see the module. `rule` is how its first stage
    picks a family: `"stacking"`, the higher stacking weight, or `"lift rows"`."""

    variants = (TWO_STAGE,)
    tests = True
    cells = False
    rule = "stacking"

    def fit(self, observation: Observation, *, smoke: bool) -> Fitted:
        data = reading.panel(observation)
        tests = reading.estimates(observation)
        sampler = bayes.SMOKE if smoke else bayes.SAMPLER
        fits = bayes.family_fits(observation, sampler)
        draws = min(bayes.DRAWS, sampler.chains * sampler.draws)
        posterior = stacked(list(fits.values()), draws=draws)
        weights = dict(zip(bayes.FAMILIES, map(float, posterior.stacking), strict=True))
        family = max(weights, key=weights.__getitem__)
        picked = lift_row_pick(list(fits.values())) if self.rule == "lift rows" else None
        if picked is not None:
            family = list(fits)[int(picked.argmax())]
        transforms = transforms_from_bayes(fits[family])
        media = [
            Effect(c, 1.0, sign="positive", transform=transforms[c]) for c in observation.channels
        ]
        spec = DlmSpec(
            effects=(*media, *(Effect(c, 1.0) for c in observation.controls)),
            trend=Trend(1, 1.0),
            season=Season(YEAR, HARMONICS, 1.0),
        )
        choice = choose_dlm(
            data, NATIONAL, spec, TWO_STAGE_DISCOUNTS, level=LEVEL, estimates=tests or None
        )
        readings = choice.readings(DRAWS, observation.seed)
        flags = _flags(choice, observation.channels) | {"hill weight": weights["hill"]}
        for name, one in fits.items():
            flags |= {f"{name} {key}": value for key, value in bayes._health(one).items()}
        ess = _ess(list(choice.kept.values()))
        flags["min ess bulk"] = ess
        forecast, absent = None, {}
        if observation.future_controls is None and observation.controls:
            absent["forecast"] = "the export gives no controls over the quarter"
        else:
            drawn = _forecast(choice, observation)
            forecast = drawn.kpi
            flags["forecast acceptance"] = drawn.acceptance
        settings = {
            "first stage": family,
            "weights": weights,
            "sampler": dataclasses.asdict(sampler),
            "transforms": {
                c: {
                    "retention": t.retention,
                    "half_saturation": t.half_saturation,
                    "shape": repr(t.shape),
                }
                for c, t in transforms.items()
            },
            "effects discount": 1.0,
            "discounts": dataclasses.asdict(TWO_STAGE_DISCOUNTS),
            "level": LEVEL,
            "season_harmonics": HARMONICS,
            "readings": DRAWS,
            "gain_level": GAIN_LEVEL,
            "lift_rows_read": sum(len(rows) for rows in tests.values()),
        }
        if self.rule != "stacking":
            settings["family rule"] = (
                self.rule if picked is not None else "stacking: the export holds no test"
            )
        return Fitted(choice, readings, forecast, flags, settings, absent, ess, _ess([choice.fit]))

    def plan(self, fit: Fitted, variant: str, observation: Observation) -> Planned:
        channels = fit.choice.channels()
        lower, upper = observation.lower, observation.upper
        settings: dict[str, Any] = {"planner": "allocate"}
        if variant == LOGGED:
            cut = logged_range(observation)
            settings["spend range"] = (
                "logged" if cut is not None else "the box: the logged range cannot spend the budget"
            )
            if cut is not None:
                lower, upper = cut
        weekly = allocate(
            [channels[c] for c in observation.channels],
            observation.budget,
            observation.planned,
            history=observation.spend,
            lower=lower,
            upper=upper,
        ).spend
        gain = reading.gains(fit.readings, observation, weekly)
        low, high = np.quantile(gain, [(1.0 - GAIN_LEVEL) / 2.0, (1.0 + GAIN_LEVEL) / 2.0])
        return Planned(
            weekly=np.asarray(weekly, dtype=float),
            gain=float(np.mean(gain)),
            gain_interval=(float(low), float(high)),
            forecast=fit.forecast,
            response=_response(fit.choice, observation),
            flags=fit.flags,
            settings=fit.settings | settings,
            absent=fit.absent,
            ess=fit.best_ess,
        )


class DlmTwoStageLiftPick(DlmTwoStage):
    """The two-stage arm, its first stage's family picked by the lift rows; see the module."""

    variants = (LIFT_PICK,)
    rule = "lift rows"


class DlmTwoStageLogged(DlmTwoStage):
    """The two-stage arm's secondary variant, planned within the logged spend range; see the
    module."""

    variants = (LOGGED,)


def claimed(gain: np.ndarray, level: float) -> float:
    """The gain a plan claims at `level`: the mean of the `ceil(level n)` worst of the `n`
    readings' gains, the lower tail `cvar_allocate` plans on at that level."""
    worst = np.sort(np.asarray(gain, dtype=float))[: max(1, int(np.ceil(level * len(gain))))]
    return float(worst.mean())


def logged_range(observation: Observation) -> tuple[np.ndarray, np.ndarray] | None:
    """The export's box cut to each channel's logged spend range: from the larger of its lower
    bound and the least a week of the history spent on it to the smaller of its upper bound and
    the most; None where that box is empty for a channel or cannot spend the quarter's budget at
    one rate a week."""
    spend = np.asarray(observation.spend, dtype=float)
    lower = np.maximum(np.asarray(observation.lower, dtype=float), spend.min(axis=0))
    upper = np.minimum(np.asarray(observation.upper, dtype=float), spend.max(axis=0))
    weekly = observation.budget / observation.planned
    if np.any(lower > upper) or not lower.sum() <= weekly <= upper.sum():
        return None
    return lower, upper


def _ess(fits: Sequence[DlmFit]) -> float:
    """The least bulk effective sample size of the fits' chains (`chc.dlm.ConstrainedDraws.ess`).

    Raises:
        ValueError: on a fit whose posterior is not drawn from chains, its effects not signed.
    """
    unsigned = [f.spec for f in fits if f.signed is None]
    if unsigned:
        msg = f"specs {unsigned} have no signed effect, so no chains"
        raise ValueError(msg)
    return float(np.min([f.signed.ess for f in fits if f.signed is not None]))


def _forecast(
    choice: DlmChoice, observation: Observation, events: RecurringEvents | None = None
) -> DlmForecastDraws:
    """The record's forecast; see the module. A recurring event's column over the quarter is 1 in
    its weeks of the year, read off the history's calendar carried on."""
    future = observation.future_controls or {}
    plan = {
        name: np.full(observation.planned, float(observation.status_quo[j]))
        for j, name in enumerate(observation.channels)
    }
    effects = {e.column for e in choice.fit.spec.effects}
    plan |= {
        name: np.asarray(future[name], dtype=float)
        for name in observation.controls
        if name in effects
    }
    if events is not None and events.names:
        weeks = observation.sales.shape[0]
        dates = reading.FIRST_WEEK + np.timedelta64(7, "D") * np.arange(
            weeks, weeks + observation.planned
        )
        plan |= events.columns(dates)
    return choice.forecast(observation.planned, plan, DRAWS, observation.seed)


def _response(choice: DlmChoice, observation: Observation) -> dict[str, Any]:
    """The record's response; see the module.

    Raises:
        ValueError: on a fitted spec whose effects are not signed, so whose posterior is not drawn.
    """
    names = observation.channels
    # a child spawned from the seed, its stream apart from the readings', each of them the seed's
    # own or the seed's with one number beside it
    rng = np.random.default_rng(np.random.SeedSequence(observation.seed).spawn(1)[0])
    kept = list(choice.kept.items())
    shares = np.full(len(kept), DRAWS // len(kept))
    shares[rng.choice(len(kept), DRAWS % len(kept), replace=False)] += 1
    draws: list[tuple[response.Channel, ...]] = []
    paths, parts = [], []
    for share, (k, fit) in zip(shares.tolist(), kept, strict=True):
        if not share:
            continue
        if fit.signed is None:
            msg = f"fitted spec {k} has no draws of its posterior to read a response from"
            raise ValueError(msg)
        pool = fit.signed.pooled
        pick = rng.choice(pool.states.shape[0], share, replace=False)
        posterior = PosteriorDraws(pool.states[pick], pool.variance[pick])
        spec = fit.spec
        index = [next(i for i, e in enumerate(spec.effects) if e.column == n) for n in names]
        channels = []
        for i in index:
            transform = spec.effects[i].transform
            assert transform is not None  # every channel's effect carries one, as the spec does
            channels.append(transform.channel(1.0))
        draws += [tuple(channels)] * share
        # a coefficient's coordinate in the state follows the trend's; its part, the trend's one
        first = spec.trend.order if spec.trend else 0
        paths.append(posterior.states[:, :, [first + i for i in index]])
        split = decompose(fit.filtered, posterior)
        parts.append(split[:, :, [(1 if spec.trend else 0) + i for i in index]])
    path, part = np.concatenate(paths), np.concatenate(parts)
    return mapped(names, draws, part.sum(axis=1), part[held(len(draws))], path)


def _flags(choice: DlmChoice, channels: Sequence[str]) -> dict[str, float | int | bool]:
    """What a record keeps of the choice; see the module."""
    best = choice.fit.spec
    flags: dict[str, float | int | bool] = {
        "kept": len(choice.specs),
        "fitted": len(choice.kept),
        "fitted not mixed": sum(
            1 for f in choice.kept.values() if f.signed is not None and not f.signed.mixed
        ),
        "trend discount": best.trend.discount if best.trend else 1.0,
    }
    flags |= {f"{e.column} discount": e.discount for e in best.effects}
    parameters = ["trend discount", *(f"{e.column} discount" for e in best.effects)]
    parameters += [
        f"{name} {axis}" for name in channels for axis in ("retention", "half-saturation")
    ]
    at_best = {e.parameter for e in choice.edges if e.best}
    flags |= {f"{parameter} at an end": parameter in at_best for parameter in parameters}
    flags["ends the kept set reaches"] = len(choice.edges)
    return flags


def _grid(grid: TransformGrid) -> dict[str, Any]:
    return {
        "retentions": list(grid.retentions),
        "half_saturations": list(grid.half_saturations),
        "shapes": [repr(s) for s in grid.shapes],
    }
