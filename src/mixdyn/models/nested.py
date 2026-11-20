"""A nested media-mix model: upper-funnel channels move a mediating series, branded searches or
visits, and the mediator with the other channels moves the KPI. Each stage is the Bayesian model of
`mixdyn.models.bayes_mmm`:

    mediator_t = stage one's model of its channels, its calendar and its controls
    KPI_t      = stage two's model of its channels, its calendar and its controls,
                 the mediator among them, `gamma * mediator_t`

The mediator is one of the panel's controls. Stage one reads it as its KPI and never as an input,
so its fit is not the series explaining itself; stage two reads it as a control held at or above
0, its coefficient `gamma` the KPI's rise per unit of the mediator. The stages share no parameter
and the mediator is observed, so their joint posterior is the product of the two: each is drawn by
`fit_bayes` from a seed of its own, and a draw of one paired with the same draw of the other is a
draw of both.

A channel's return on the KPI over the window is its return in stage two, where it acts directly,
plus `gamma` times its return on the mediator in stage one: by stage one's return identity, its
contribution to the mediator over the window is that return times its spend, and the mediator
enters the KPI in proportion. Reading the mediator as stage two's control blocks the channels'
path through it, so a channel that acts only through the mediator has no return in a one-stage fit
that controls for it; that is what the nested model is for. Like any mediation, it reads the
mediator's coefficient as causal: whatever moves the mediator and the KPI together and is not in
stage two's calendar or controls biases `gamma`.

Stage one's returns are in the mediator's units a currency unit, so the engine's default return
prior, `RoiPrior()`, Meridian's for a revenue KPI, would state a median of 1.2 of the mediator's
units a currency unit whatever they count. A stage-one channel left at it reads it at the
mediator's scale instead: its median times the mediator's mean over the KPI's, each over the
window's observed periods, so that the prior states the share of the mediator's mean a currency
unit buys as the default states the KPI's, and the stage's posterior on its own scale does not
move with the mediator's unit. `HalfNormalRoi()`, the opt-in prior for a channel that may not
move the mediator at all, is read at that scale the same way. A prior stated otherwise is read as
stated. The fit's `stage_one.spec` holds the prior it was drawn under.

Stage one holds the terms its spec gives it and no others. A mediator that drifts over the window,
as demand does, needs a slow term of its own there, a trend. A channel's kernel runs from the
panel's first period with nothing spent before it, so at a decay near 1 its carryover rises as the
periods fill the kernel. Without a trend it is the one term that can follow the drift, and the
drift is read as that channel's return on the mediator (ADR 0014).

Lift estimates (`mixdyn.calibration.LiftEstimate`) enter the stage whose parameters alone set
their lift: a channel's lift on the mediator is stage one's, and its lift on the KPI stage two's
where it acts on the KPI only directly. A lift on the KPI of a channel that moves the mediator
reads both stages: at the market's scale,

    lift / share ~ N(gamma l_1 + l_2, se / share),

`l_1` the channel's coefficient in stage one times the gap in its curve there between the spend as
the test ran and as planned, summed over the readout, in the mediator's units, and `l_2` the same
of stage two, in the KPI's, where stage two has the channel, and 0 where it does not. The
posterior then no longer factorises, and both stages are drawn as one model, its log density the
sum of the stages' and each such lift's term, on stage one's coordinates and then stage two's.
Each stage's kernels are as long as the longest series its parameters read, such a lift's among
them, and each stage's fit is summarised from its own coordinates of each draw as `fit_bayes`
summarises a fit, the one sampler's divergences and energies in both.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from chc import response
from jax.scipy import stats as jstats

from mixdyn.data.panel import Count
from mixdyn.inference import compiled
from mixdyn.models.bayes_mmm import (
    _PATHFINDER_START,
    BayesFit,
    BayesModel,
    BayesSpec,
    HalfNormalRoi,
    ReturnPrior,
    RoiPrior,
    Start,
    _Estimates,
    _estimates,
    _fitted,
    _initial,
    _model,
    _thinned,
    fit_bayes,
)

if TYPE_CHECKING:
    from mixdyn.calibration import LiftEstimate
    from mixdyn.data import Geo, MediaPanel

type _Array = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class NestedSpec:
    """The two stages and the control of the panel that mediates between them. A stage-one
    channel left at the engine's default return prior, `RoiPrior()`, or at `HalfNormalRoi()`,
    reads it at the mediator's scale, as the module states. The default is told apart by
    equality, so `RoiPrior()` or `HalfNormalRoi()` stated by hand is moved too; a prior meant in
    the mediator's own units is stated with numbers of its own.

    Raises:
        ValueError: on a stage one that reads the mediator, as a channel or a control, on a stage
            two that does not read it as a control held at or above 0, on a return prior that
            counts the carried return, and on a stage that states its kernels' length, since each
            stage's kernels are as long as the longest series either reads.
    """

    mediator: str
    stage_one: BayesSpec
    stage_two: BayesSpec

    def __post_init__(self) -> None:
        read = [m.channel for m in self.stage_one.media] + [
            c.column for c in self.stage_one.controls
        ]
        if self.mediator in read:
            msg = f"stage one reads the mediator {self.mediator!r}: it would explain itself"
            raise ValueError(msg)
        held = [c for c in self.stage_two.controls if c.column == self.mediator and c.lower == 0.0]
        if not held:
            msg = (
                f"stage two must read the mediator {self.mediator!r} as a control held at or above"
                " 0, Control(mediator, lower=0.0)"
            )
            raise ValueError(msg)
        media = (*self.stage_one.media, *self.stage_two.media)
        carried = [m.channel for m in media if m.roi.counts == "carried"]
        if carried:
            msg = f"the nested model states each return on the window: {carried} carry it"
            raise ValueError(msg)
        if self.stage_one.kernel is not None or self.stage_two.kernel is not None:
            msg = "the nested model's kernels are as long as the longest series a stage reads"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True, eq=False)
class NestedFit:
    """Both stages' posteriors over one geo's window: `stage_one` the mediator's, its returns in
    the mediator's units per currency unit, its spec's priors those it was drawn under, and
    `stage_two` the KPI's, the mediator's coefficient among its controls'. Their draws pair index
    by index.

    `mediated_log_likelihood` is each draw's log density of each lift on the KPI of a channel of
    stage one, which reads both stages, in the KPI's units, `(chains, draws, lifts)`: channel by
    channel in stage one's order, and each channel's in the order given. Where the fit read such a
    lift, the stages were drawn as one model, and each stage's `divergent` and `e_bfmi` are its
    sampler's.
    """

    spec: NestedSpec
    stage_one: BayesFit
    stage_two: BayesFit
    mediated_log_likelihood: _Array

    def mediator_coefficient(self) -> _Array:
        """The KPI's rise per unit of the mediator at each draw, `(chains, draws)`."""
        k = [c.column for c in self.stage_two.spec.controls].index(self.spec.mediator)
        return self.stage_two.draws["control"][..., k]

    def roi(self) -> dict[str, _Array]:
        """Each channel's return on the KPI over the window at each draw, `(chains, draws)`: its
        direct return, where stage two has the channel, plus the mediator's coefficient times its
        return on the mediator, where stage one has it."""
        gamma = self.mediator_coefficient()
        out: dict[str, _Array] = {}
        for j, m in enumerate(self.stage_one.spec.media):
            out[m.channel] = gamma * self.stage_one.draws["roi"][..., j]
        for j, m in enumerate(self.stage_two.spec.media):
            out[m.channel] = out.get(m.channel, 0.0) + self.stage_two.draws["roi"][..., j]
        return out

    def channels(self) -> dict[str, response.Channel]:
        """Each channel's return on the KPI as `chc.response` reads it, at its parameters'
        posterior medians: a channel of stage two as that stage reads it, and one of stage one as
        stage one reads it, its coefficient times the mediator's.

        Raises:
            ValueError: on a channel in both stages, whose return is a sum of two curves, which
                the planner does not read.
        """
        gamma = float(np.median(self.mediator_coefficient()))
        return _composed(self.stage_one.channels(), self.stage_two.channels(), gamma)

    def channel_draws(self, count: int) -> list[dict[str, response.Channel]]:
        """Each channel's return on the KPI as `channels` composes it, at `count` draws of both
        stages, evenly spaced as `BayesFit.channel_draws` reads them and paired index by index: a
        stage-one channel's coefficient times the mediator's at the same draw.

        Raises:
            ValueError: on a channel in both stages, and a count below 1 or above the fit's draws.
        """
        gamma = self.mediator_coefficient()
        paired = zip(
            self.stage_one.channel_draws(count),
            self.stage_two.channel_draws(count),
            gamma.reshape(-1)[_thinned(gamma.shape, count)],
            strict=True,
        )
        return [_composed(mediated, direct, float(g)) for mediated, direct, g in paired]


def _composed(
    mediated: dict[str, response.Channel], direct: dict[str, response.Channel], gamma: float
) -> dict[str, response.Channel]:
    """Stage two's channels as they are, and stage one's with their coefficients times `gamma`."""
    both = sorted(set(mediated) & set(direct))
    if both:
        msg = f"channels {both} act directly and through the mediator: two curves a channel"
        raise ValueError(msg)
    out = dict(direct)
    for name, channel in mediated.items():
        out[name] = response.Channel(
            channel.kernel, channel.curve, gamma * float(channel.coefficient)
        )
    return out


@dataclass(frozen=True, slots=True, eq=False)
class _Mediated:
    """One channel's lifts on the KPI where stage one has the channel: laid out for stage one's
    kernels and, where stage two has the channel too, for stage two's, each lift and its standard
    error over its share on stage two's scale."""

    one: _Estimates
    two: _Estimates | None


@dataclass(frozen=True, slots=True, eq=False)
class NestedModel:
    """Both stages' models, each as `bayes_model` builds it, its kernels as long as the longest
    series its parameters read, with the lifts on the KPI that read both: `mediator` is the
    mediator's column among stage two's controls, and `lifts` each lift on the KPI of a channel of
    stage one, channel by channel in stage one's order.

    `log_density` takes a point of the `dimension` coordinates, stage one's and then stage two's,
    each in its model's order, and returns the joint posterior's log density, up to a constant,
    with every transform's Jacobian, and its gradient: the stages' and each lift's term.
    `log_likelihood` maps points to each lift's log density in the KPI's units, as
    `NestedFit.mediated_log_likelihood` holds it.
    """

    stage_one: BayesModel
    stage_two: BayesModel
    mediator: int
    lifts: tuple[_Mediated, ...]
    _compiled: Callable[[jax.Array], tuple[jax.Array, jax.Array]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_compiled", jax.jit(jax.value_and_grad(self._log_density)))

    @property
    def dimension(self) -> int:
        return self.stage_one.dimension + self.stage_two.dimension

    def log_density(self, theta: _Array) -> tuple[float, _Array]:
        value, gradient = self._compiled(jnp.asarray(theta, dtype=float))
        return float(value), np.asarray(gradient)

    def log_likelihood(self, theta: _Array) -> _Array:
        """Each lift's log density in the KPI's units at each point of `theta`, `(points,
        dimension)`."""
        d = self.stage_one.dimension
        pointwise = jax.lax.map(
            lambda t: self._pointwise(t[:d], t[d:]),
            jnp.asarray(np.atleast_2d(theta), dtype=float),
            batch_size=256,
        )
        return np.asarray(pointwise) - math.log(self.stage_two.kpi_unit)

    def _log_density(self, theta: jax.Array) -> jax.Array:
        first, second = theta[: self.stage_one.dimension], theta[self.stage_one.dimension :]
        return (
            self.stage_one._log_density(first)
            + self.stage_two._log_density(second)
            + jnp.sum(self._pointwise(first, second))
        )

    def _lift(self, theta: _Array) -> _Array:
        """Each stage's coordinates moved into its support, as its model moves them."""
        d = self.stage_one.dimension
        return np.concatenate([self.stage_one._lift(theta[:d]), self.stage_two._lift(theta[d:])])

    def _pointwise(self, first: jax.Array, second: jax.Array) -> jax.Array:
        """Each lift's log density on stage two's scale, at stage one's coordinates `first` and
        stage two's `second`."""
        if not self.lifts:
            return jnp.zeros(0)
        lift = np.concatenate([lifts.one.lift for lifts in self.lifts])
        se = np.concatenate([lifts.one.se for lifts in self.lifts])
        return jstats.norm.logpdf(lift, self._means(first, second), se)

    def _means(self, first: jax.Array, second: jax.Array) -> jax.Array:
        """Each lift's mean on stage two's scale, at stage one's coordinates `first` and stage
        two's `second`."""
        one, two = self.stage_one, self.stage_two
        p1, _ = one._constrained(first)
        p2, _ = two._constrained(second)
        _, beta1 = one._terms(p1)
        _, beta2 = two._terms(p2)
        # the KPI's rise, on stage two's scale, per unit of the mediator on stage one's
        gamma = p2["control"][self.mediator] / two.control_sd[self.mediator] * one.kpi_unit
        out = []
        for lifts in self.lifts:
            mean = gamma * beta1[lifts.one.channel] * one._lifted(p1, lifts.one)
            if lifts.two is not None:
                mean = mean + beta2[lifts.two.channel] * two._lifted(p2, lifts.two)
            out.append(mean)
        return jnp.concatenate(out)


def _mediator(panel: MediaPanel, spec: NestedSpec) -> MediaPanel:
    """The panel with the mediator as its KPI, which stage one reads.

    Raises:
        ValueError: on a panel without the mediator among its controls.
    """
    if spec.mediator not in panel.controls:
        msg = f"the panel has no control {spec.mediator!r}; it has {list(panel.controls)}"
        raise ValueError(msg)
    k = panel.controls.index(spec.mediator)
    return dataclasses.replace(
        panel, kpi=panel.control_values[:, :, k], kpi_unit=Count(spec.mediator)
    )


def _at_scale(stage: BayesSpec, ratio: float) -> BayesSpec:
    """Stage one's spec with each channel left at a default return prior read at the mediator's
    scale, `ratio` the mediator's mean over the KPI's: the return times `ratio`."""

    def scaled(prior: ReturnPrior) -> ReturnPrior:
        if prior == RoiPrior():
            return dataclasses.replace(prior, mu=prior.mu + math.log(ratio))
        if prior == HalfNormalRoi():
            return dataclasses.replace(prior, scale=prior.scale * ratio)
        return prior

    media = tuple(dataclasses.replace(m, roi=scaled(m.roi)) for m in stage.media)
    return stage if media == stage.media else dataclasses.replace(stage, media=media)


def nested_model(
    panel: MediaPanel,
    geo: Geo,
    spec: NestedSpec,
    *,
    window: tuple[np.datetime64, np.datetime64] | None = None,
    estimates: Mapping[str, Sequence[LiftEstimate]] | None = None,
    mediator_estimates: Mapping[str, Sequence[LiftEstimate]] | None = None,
) -> NestedModel:
    """The nested model of `spec` on one geo of `panel` over `window`, its first and last dates,
    by default the whole calendar: stage one with the mediator as the KPI, a channel left at the
    default return prior reading it at the mediator's scale, its likelihood holding
    `mediator_estimates`, each channel's lift estimates on the mediator, in the mediator's units;
    stage two with the panel's own, its likelihood holding the lifts in `estimates`, each channel's
    on the KPI, of channels stage one does not have; and the lifts on the KPI of channels it has.

    Raises:
        ValueError: on a panel without the mediator among its controls; on a lift on the KPI of a
            channel of stage one in a geo other than `geo`, or a channel given none; and on what
            `bayes_model` refuses for either stage.
    """
    mediator = _mediator(panel, spec)
    staged = [m.channel for m in spec.stage_one.media]
    direct = [m.channel for m in spec.stage_two.media]
    given = estimates or {}
    mediated = {c: es for c, es in given.items() if c in staged}
    empty = sorted(c for c, es in mediated.items() if not es)
    if empty:
        msg = f"estimates name channels the model does not have, [], or none, {empty}"
        raise ValueError(msg)
    elsewhere = sorted({g for es in mediated.values() for e in es for g in e.geos or ()} - {geo})
    if elsewhere:
        msg = f"estimates name geos {elsewhere} that the model of {geo!r} does not read"
        raise ValueError(msg)
    one = _model(
        mediator,
        geo,
        spec.stage_one,
        window,
        {},
        mediator_estimates or {},
        max((e.spend.size for es in mediated.values() for e in es), default=0),
    )
    two = _model(
        panel,
        geo,
        spec.stage_two,
        window,
        {},
        {c: es for c, es in given.items() if c not in staged},
        max((e.spend.size for c in mediated if c in direct for e in mediated[c]), default=0),
    )
    scaled = _at_scale(spec.stage_one, one.kpi_unit / two.kpi_unit)
    if scaled is not spec.stage_one:
        one = dataclasses.replace(one, spec=scaled)
    return NestedModel(
        stage_one=one,
        stage_two=two,
        mediator=[c.column for c in spec.stage_two.controls].index(spec.mediator),
        lifts=tuple(
            _Mediated(
                _estimates(j, mediated[c], one.lagged.shape[2], one.spend_unit[j], two.kpi_unit),
                _estimates(
                    direct.index(c),
                    mediated[c],
                    two.lagged.shape[2],
                    two.spend_unit[direct.index(c)],
                    two.kpi_unit,
                )
                if c in direct
                else None,
            )
            for j, c in enumerate(staged)
            if c in mediated
        ),
    )


def fit_nested(
    panel: MediaPanel,
    geo: Geo,
    spec: NestedSpec,
    *,
    seed: int,
    window: tuple[np.datetime64, np.datetime64] | None = None,
    estimates: Mapping[str, Sequence[LiftEstimate]] | None = None,
    mediator_estimates: Mapping[str, Sequence[LiftEstimate]] | None = None,
    chains: int = 4,
    warmup: int = 1000,
    draws: int = 1000,
    start: Start = _PATHFINDER_START,
) -> NestedFit:
    """Draw both stages' posteriors on one geo of `panel` over `window`, its first and last dates,
    by default the whole calendar, as `nested_model` reads `estimates`, each channel's lift
    estimates on the KPI, and `mediator_estimates`, each channel's on the mediator: apart, each
    stage from a seed spawned from `seed`, or, where a lift on the KPI is of a channel of stage
    one, as one model from a third. The models are built before anything is drawn, so what either
    refuses is refused before any sampling.

    Raises:
        ValueError: on what `nested_model` refuses, and on what `fit_bayes` refuses for either
            stage.
    """
    model = nested_model(
        panel,
        geo,
        spec,
        window=window,
        estimates=estimates,
        mediator_estimates=mediator_estimates,
    )
    mediator = _mediator(panel, spec)
    one, two, joint = (int(s) for s in np.random.SeedSequence(seed).generate_state(3))
    if not model.lifts:
        return NestedFit(
            spec=spec,
            stage_one=fit_bayes(
                mediator,
                geo,
                model.stage_one.spec,
                seed=one,
                window=window,
                estimates=mediator_estimates,
                chains=chains,
                warmup=warmup,
                draws=draws,
                start=start,
            ),
            stage_two=fit_bayes(
                panel,
                geo,
                spec.stage_two,
                seed=two,
                window=window,
                estimates=estimates,
                chains=chains,
                warmup=warmup,
                draws=draws,
                start=start,
            ),
            mediated_log_likelihood=np.empty((chains, draws, 0)),
        )
    initial = _initial(model, start, chains, joint)
    result = compiled.sample(model._log_density, initial, seed=joint, warmup=warmup, draws=draws)
    d = model.stage_one.dimension
    positions, divergent, energy = result.positions, result.divergent, result.energy
    return NestedFit(
        spec=spec,
        stage_one=_fitted(
            mediator, geo, window, model.stage_one, positions[..., :d], divergent, energy
        ),
        stage_two=_fitted(
            panel, geo, window, model.stage_two, positions[..., d:], divergent, energy
        ),
        mediated_log_likelihood=model.log_likelihood(
            positions.reshape(-1, model.dimension)
        ).reshape(chains, draws, -1),
    )
