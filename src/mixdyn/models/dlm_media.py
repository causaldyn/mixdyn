"""The DLM over media: the carryover and curve of each channel a dynamic fit reads.

A channel's effect carries a `mixdyn.transforms.Transform`, and the dynamic model's coefficient
multiplies the channel's spend carried over and saturated (`mixdyn.models.dlm`). The transforms
are chosen one of two ways:

* in two stages: `transforms_from_bayes` reads a static Bayesian fit's posterior medians, and
  `choose_dlm` chooses the discounts with the transforms held;
* by a search: `search_media` chooses each channel's retention, half-saturation point and curve
  with the discounts by the dynamic model's one-step log-likelihood, and keeps every spec it
  filtered that a likelihood-ratio test does not reject, as `chc.dlm.confidence_set` keeps
  discounts.

A grid's product is too large to filter whole: ten retentions, eight half-saturation points and
three curves are 240 transforms a channel, and three channels 240³ specs before the discounts. The
search is a block coordinate ascent over the grid, then over the grid's intervals halved, each to a
fixed point. It filters the discounts' block and one channel's block at a time, and it stops at a
spec no block improves: the likelihood's best over the grid wherever no other spec is that.
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
from scipy import special

from mixdyn.models.bayes_mmm import TanhCurve
from mixdyn.models.dlm import _filter, _media
from mixdyn.models.dlm_choice import DlmChoice, _edges, _ends, _kept_fit, _warn, within
from mixdyn.models.dlm_lift import lift_reading, lifted
from mixdyn.transforms.saturation import HillShape, Shape, TanhShape, Transform

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from chc.dlm import DLMFit

    from mixdyn.calibration import LiftEstimate
    from mixdyn.data import Geo, MediaPanel
    from mixdyn.models.bayes_mmm import BayesFit
    from mixdyn.models.dlm import DlmSpec
    from mixdyn.models.dlm_choice import DiscountGrid, Edge
    from mixdyn.models.dlm_lift import LiftReading

log = logging.getLogger(__name__)


def transforms_from_bayes(fit: BayesFit) -> dict[str, Transform]:
    """Each channel's transform at its parameters' posterior medians, by channel, as
    `BayesFit.channels` reads them: its retention, its half-saturation point, its curve, a Hill's
    slope at its median, and the fit's kernel length."""
    transforms = {}
    for j, media in enumerate(fit.spec.media):
        retention, half = (
            float(np.median(fit.draws[k][..., j])) for k in ("retention", "half_saturation")
        )
        shape: Shape = TanhShape()
        if not isinstance(fit.spec.saturation, TanhCurve):
            shape = HillShape(float(np.median(fit.draws["slope"][..., j])))
        transforms[media.channel] = Transform(retention, half, shape, fit.length)
    return transforms


@dataclass(frozen=True, slots=True)
class TransformGrid:
    """The transforms a search tries for each channel: `retentions`, `half_saturations` as
    multiples of the channel's mean positive spend on the geo, the Bayesian model's unit for them,
    and `shapes`.

    Raises:
        ValueError: on an axis with no value; retentions not ascending within `[0, 1]`;
            half-saturation multiples not ascending, positive and finite; or a shape named twice
            or not a `TanhShape` or `HillShape`.
    """

    retentions: tuple[float, ...]
    half_saturations: tuple[float, ...]
    shapes: tuple[Shape, ...]

    def __post_init__(self) -> None:
        problems = []
        retentions, halves = np.array(self.retentions), np.array(self.half_saturations)
        if not (retentions.size and np.all(np.diff(retentions) > 0.0)) or not (
            retentions.min() >= 0.0 and retentions.max() <= 1.0
        ):
            problems.append(f"retentions must ascend within [0, 1], got {self.retentions}")
        if not (halves.size and np.all(np.diff(halves) > 0.0)) or not (
            halves.min() > 0.0 and halves.max() < math.inf
        ):
            problems.append(
                f"half-saturation multiples must ascend, positive and finite, "
                f"got {self.half_saturations}"
            )
        if not self.shapes or len(set(self.shapes)) != len(self.shapes):
            problems.append(f"shapes must name at least one, none twice, got {self.shapes}")
        if not all(isinstance(s, TanhShape | HillShape) for s in self.shapes):
            problems.append(f"a shape is TanhShape() or HillShape(slope), got {self.shapes}")
        if problems:
            raise ValueError("; ".join(problems))

    @property
    def parameters(self) -> int:
        """How many of a channel's parameters the grid varies, one for each axis of more than one
        value: a channel's degrees of freedom in a likelihood ratio."""
        axes = (self.retentions, self.half_saturations, self.shapes)
        return sum(1 for values in axes if len(values) > 1)

    def refined(self, level: int) -> TransformGrid:
        """The grid with each interval between neighbouring values cut into `2 ** level` equal
        parts, the half-saturation's on a log scale; the shapes as they are. A value of a coarser
        level is the same number at every finer one.

        Raises:
            ValueError: on a level that is not a whole number at or above 0.
        """
        if isinstance(level, bool) or not isinstance(level, int) or level < 0:
            msg = f"level must be a whole number at or above 0, got {level!r}"
            raise ValueError(msg)
        return TransformGrid(
            _subdivide(self.retentions, level, geometric=False),
            _subdivide(self.half_saturations, level, geometric=True),
            self.shapes,
        )


def _subdivide(axis: tuple[float, ...], level: int, *, geometric: bool) -> tuple[float, ...]:
    """`axis` with each interval cut into `2 ** level` parts; the fractions are dyadic, so a
    point's value is the same at every level that has it."""
    parts = 2**level
    values = []
    for a, b in itertools.pairwise(axis):
        for m in range(parts):
            share = m / parts
            if m == 0:
                values.append(a)
            elif geometric:
                values.append(a * (b / a) ** share)
            else:
                values.append(a + (b - a) * share)
    values.append(axis[-1])
    return tuple(values)


@dataclass(frozen=True, slots=True)
class _Point:
    """A channel's place on a level's grid: its retention's, half-saturation's and shape's
    indices."""

    retention: int
    half: int
    shape: int


@dataclass(slots=True)
class _Search:
    """The specs filtered so far, each once: their log-likelihoods, the KPI's and with
    `estimates` their tests' given it (`lifts`), and with `spread` each channel's half-saturation
    point's log prior density (`search_media`), and the filters of those within the likelihood
    ratio's margin of the best so far. A reading of the tests is made once for the transforms of
    the channels they test, whatever the discounts."""

    panel: MediaPanel
    geo: Geo
    margin: float
    scores: dict[DlmSpec, float]
    near: dict[DlmSpec, DLMFit]
    top: float = -math.inf
    estimates: Mapping[str, Sequence[LiftEstimate]] | None = None
    lifts: dict[DlmSpec, float] = field(default_factory=dict)
    readings: dict[tuple[Transform | None, ...], LiftReading] = field(default_factory=dict)
    scales: Mapping[str, float] = field(default_factory=dict)
    spread: float | None = None

    def score(self, spec: DlmSpec) -> float:
        if spec in self.scores:
            return self.scores[spec]
        filtered = _filter(self.panel, self.geo, spec)
        loglik = filtered.log_likelihood
        if self.spread is not None:
            loglik += sum(
                -0.5 * (math.log(t.half_saturation / self.scales[column]) / self.spread) ** 2
                for _, column, t in _media(spec)
            )
        if self.estimates:
            tested = tuple(e.transform for e in spec.effects if e.column in self.estimates)
            if tested not in self.readings:
                self.readings[tested] = lift_reading(self.panel, self.geo, spec, self.estimates)
            self.lifts[spec] = lifted(filtered, self.readings[tested]).log_density
            loglik += self.lifts[spec]
        self.scores[spec] = loglik
        if loglik > self.top:
            self.top = loglik
            self.near = {
                s: f for s, f in self.near.items() if self.scores[s] >= loglik - self.margin
            }
        if loglik >= self.top - self.margin:
            self.near[spec] = filtered
        return loglik

    def best(self, current: DlmSpec, options: Sequence[DlmSpec]) -> DlmSpec:
        """The option of the highest log-likelihood, the first of a tie; `current` where it is an
        option and ties the best."""
        scores = [self.score(o) for o in options]
        top = max(scores)
        if current in options and self.score(current) >= top:
            return current
        return options[scores.index(top)]

    def moved(self, current: DlmSpec, chosen: DlmSpec, *, stage: int, column: str | None) -> bool:
        """Whether a block's best `chosen` moves the search off `current`, the block a channel's
        or with no `column` the discounts'. Each move is logged with what it set, the channel's
        transform or the discounts, and the log-likelihoods it goes between, `-inf` before for a
        start not yet filtered."""
        if chosen == current:
            return False
        before, after = self.scores.get(current, -math.inf), self.scores[chosen]
        block = "the discounts" if column is None else f"channel {column!r}"
        moved_from, moved_to = _setting(current, column), _setting(chosen, column)
        log.debug(
            "media search on geo %s: stage %d moved %s from %s to %s, its log-likelihood from "
            "%.6g to %.6g",
            self.geo,
            stage,
            block,
            moved_from,
            moved_to,
            before,
            after,
            extra={
                "geo": str(self.geo),
                "stage": stage,
                "block": block,
                "moved_from": moved_from,
                "moved_to": moved_to,
                "before": before,
                "after": after,
            },
        )
        return True


def _setting(spec: DlmSpec, column: str | None) -> Transform | tuple[float, ...] | None:
    """What a block sets in `spec`: the channel's transform, or with no channel the discounts, the
    trend's and the season's where the spec has them, then each effect's."""
    if column is None:
        blocks = (b.discount for b in (spec.trend, spec.season) if b is not None)
        return (*blocks, *(e.discount for e in spec.effects))
    return next(e.transform for e in spec.effects if e.column == column)


def search_media(
    panel: MediaPanel,
    geo: Geo,
    spec: DlmSpec,
    grid: TransformGrid,
    discounts: DiscountGrid,
    *,
    refinements: int = 1,
    level: float = 0.95,
    fitted: int | None = None,
    estimates: Mapping[str, Sequence[LiftEstimate]] | None = None,
    half_saturation_prior: float | None = None,
) -> DlmChoice:
    """Choose each transformed effect's transform on `grid` and the discounts on `discounts` by
    the one-step log-likelihood on one geo, and fit every spec filtered that a likelihood-ratio
    test at `level` does not reject, or with `fitted` at most that many of them.

    With `half_saturation_prior`, each channel's half-saturation point is given a log-normal prior
    of that scale and median the channel's mean positive spend, the grid's unit, as the static
    Bayesian model's `K_m` (`mixdyn.models.bayes_mmm`): its log density on the log of the multiple,
    `-(log m)^2 / (2 s^2)`, is added to every spec's score, so the search, its best and its kept
    set read the posterior's mode and not the likelihood's where the likelihood is flat (ADR 0085).
    The grid is even in log `m`, so the density is that of `log m` and needs no Jacobian.

    `estimates`, each channel's lift tests, enter every spec's likelihood, their log density given
    the KPI added to its one-step log-likelihood (`DlmChoice.lifts`), so the search, its best and
    its kept set read them, and every fitted spec's posterior is given them (`fit_dlm`).

    Each effect that carries a transform in `spec` is a channel the search moves; its kernel keeps
    the transform's length. The search starts at the grid point nearest each channel's transform,
    the nearest retention, the nearest half-saturation point by ratio and the transform's shape if
    the grid has it, else the grid's first; and it moves in blocks, each to its best spec:

    1. the discounts, over the whole discount grid or with `PerEffect` in its blocks
       (`DiscountGrid.blocks`), then each channel in turn over the grid's product of retentions,
       half-saturations and shapes, the rest held, until a pass moves nothing;
    2. `refinements` times, each interval of the grid halved again (`TransformGrid.refined`), and
       the passes repeated with each channel's block the three retentions and three
       half-saturation points about its point, nine at every shape.

    A move raises the likelihood and a level's grid is finite, so each stage ends, at a point no
    block improves. The kept specs are the filtered ones within `chi2_p(level) / 2` of the best
    log-likelihood, `p` counting each axis the grid varies once a channel and each discount the
    discount grid varies (`DiscountGrid.parameters`). A spec the search did not filter is not
    kept, however near the best. Each end of either grid that a kept spec reaches where the
    parameter goes on past it, a discount's, a retention's or a half-saturation point's, is named
    in `DlmChoice.edges` and logged as a warning.
    With `fitted`, the kept specs fitted are those at evenly spaced ranks of their log-likelihood,
    rank `floor(i (N - 1) / (fitted - 1))` of the `N` kept for `i` from 0, the best first and the
    last kept last (ADR 0023): the choice holds every kept spec and its filter, and its readings
    are a sample of their mixture, stratified by likelihood.
    Each move is logged at debug level with the log-likelihoods it goes between; each stage's end
    and the kept set at info.

    Raises:
        ValueError: on refinements that are not a whole number at or above 0, a `level` outside
            (0, 1), a `fitted` that is not a whole number at or above 1, a prior's scale that is
            not positive and finite, a spec with no transformed effect, a channel with no positive
            spend on the geo, and what `DiscountGrid.specs`, `fit_dlm` or
            `mixdyn.models.dlm_lift` refuses.
        RuntimeError: on a fitted spec's sampler that stops from every seed it starts from.
    """
    if isinstance(refinements, bool) or not isinstance(refinements, int) or refinements < 0:
        msg = f"refinements must be a whole number at or above 0, got {refinements!r}"
        raise ValueError(msg)
    if not 0.0 < level < 1.0:
        msg = f"level must lie in (0, 1), got {level}"
        raise ValueError(msg)
    if fitted is not None and (
        isinstance(fitted, bool) or not isinstance(fitted, int) or fitted < 1
    ):
        msg = f"fitted must be a whole number at or above 1, got {fitted!r}"
        raise ValueError(msg)
    if half_saturation_prior is not None and not 0.0 < half_saturation_prior < math.inf:
        msg = f"a prior's scale must be positive and finite, got {half_saturation_prior!r}"
        raise ValueError(msg)
    media = _media(spec)
    discounts.blocks(spec)
    scales = {column: _positive_mean(panel, geo, column) for _, column, _ in media}
    parameters = grid.parameters * len(media) + discounts.parameters(spec)
    search = _Search(
        panel,
        geo,
        float(special.chdtri(parameters, 1.0 - level)) / 2.0,
        scores={},
        near={},
        estimates=estimates,
        scales=scales,
        spread=half_saturation_prior,
    )
    points = {column: _nearest(grid, t, scales[column]) for _, column, t in media}
    lengths = {column: t.length for _, column, t in media}

    def place(current: DlmSpec, at: dict[str, _Point], on: TransformGrid) -> DlmSpec:
        transforms = {
            column: Transform(
                on.retentions[p.retention],
                on.half_saturations[p.half] * scales[column],
                on.shapes[p.shape],
                lengths[column],
            )
            for column, p in at.items()
        }
        effects = tuple(
            replace(e, transform=transforms[e.column]) if e.column in transforms else e
            for e in current.effects
        )
        return replace(current, effects=effects)

    current = place(spec, points, grid)
    for stage in range(refinements + 1):
        on = grid.refined(stage)
        if stage:
            points = {
                column: _Point(2 * p.retention, 2 * p.half, p.shape) for column, p in points.items()
            }
        passes = 0
        while True:
            passes += 1
            moved = False
            for k in range(len(discounts.blocks(current))):
                # each block about the spec the last one moved to: one built before it would hold
                # the last block's discount where it was, and could move the search back down
                chosen = search.best(current, discounts.blocks(current)[k])
                moved = search.moved(current, chosen, stage=stage, column=None) or moved
                current = chosen
            for column in points:
                options = _block(points[column], on, coarse=not stage)
                specs = [place(current, points | {column: p}, on) for p in options]
                chosen = search.best(current, specs)
                if search.moved(current, chosen, stage=stage, column=column):
                    points[column] = options[specs.index(chosen)]
                    current, moved = chosen, True
            if not moved:
                break
        log.info(
            "media search on geo %s: stage %d at a fixed point after %d passes, %d specs filtered",
            geo,
            stage,
            passes,
            len(search.scores),
            extra={
                "geo": str(geo),
                "stage": stage,
                "passes": passes,
                "filtered": len(search.scores),
                "loglik": search.scores[current],
            },
        )

    specs = tuple(search.near)
    filtered = tuple(search.near.values())
    loglik = np.array([search.scores[s] for s in specs])
    kept = within(loglik, parameters, level)
    best = int(np.argmax(loglik))
    log.info(
        "media search on geo %s: %d of %d specs filtered kept at %d parameters, the best at %.4g",
        geo,
        len(kept),
        len(search.scores),
        parameters,
        loglik[best],
        extra={
            "geo": str(geo),
            "filtered": len(search.scores),
            "kept": len(kept),
            "parameters": parameters,
            "margin": search.margin,
        },
    )
    held = [specs[k] for k in kept]
    edges = (
        *discounts.edges(held, specs[best]),
        *_transform_edges(grid, scales, held, specs[best]),
    )
    _warn(log, geo, edges)
    ranked = np.argsort(-loglik[list(kept)], kind="stable")
    count = len(held) if fitted is None else min(fitted, len(held))
    chosen = sorted(int(ranked[i * (len(held) - 1) // max(count - 1, 1)]) for i in range(count))
    if count < len(held):
        log.info(
            "media search on geo %s: %d of %d kept specs fitted, at evenly spaced ranks",
            geo,
            count,
            len(held),
            extra={"geo": str(geo), "fitted": count, "kept": len(held)},
        )
    return DlmChoice(
        specs=tuple(held),
        filtered=tuple(filtered[k] for k in kept),
        best=held.index(specs[best]),
        kept=MappingProxyType(
            {i: _kept_fit(panel, geo, held[i], estimates=estimates) for i in chosen}
        ),
        edges=edges,
        lifts=tuple(search.lifts[s] for s in held) if estimates else (),
    )


def _transform_edges(
    grid: TransformGrid, scales: Mapping[str, float], kept: Sequence[DlmSpec], best: DlmSpec
) -> list[Edge]:
    """Each end of the grid that a kept spec's transform reaches where the parameter goes on past
    it: a retention's ends but 0 and 1, its bounds, and either end of the half-saturation points,
    each a multiple of the channel's mean positive spend, `scales`. A grid refined keeps its ends,
    the same numbers."""
    held = [{column: t for _, column, t in _media(s)} for s in kept]
    top = {column: t for _, column, t in _media(best)}
    edges = []
    for column, scale in scales.items():
        edges += _edges(
            f"{column} retention",
            _ends(grid.retentions, 0.0, 1.0),
            [h[column].retention for h in held],
            top[column].retention,
        )
        edges += _edges(
            f"{column} half-saturation",
            _ends(grid.half_saturations, 0.0, math.inf),
            [h[column].half_saturation for h in held],
            top[column].half_saturation,
            scale,
        )
    return edges


def _block(point: _Point, on: TransformGrid, *, coarse: bool) -> list[_Point]:
    """A channel's options: on the coarse grid its whole product; on a finer one its point and the
    neighbours on either side in retention and in half-saturation, at every shape."""
    shapes = range(len(on.shapes))
    if coarse:
        retentions, halves = range(len(on.retentions)), range(len(on.half_saturations))
    else:
        retentions = range(
            max(point.retention - 1, 0), min(point.retention + 2, len(on.retentions))
        )
        halves = range(max(point.half - 1, 0), min(point.half + 2, len(on.half_saturations)))
    return [_Point(r, h, s) for r, h, s in itertools.product(retentions, halves, shapes)]


def _nearest(grid: TransformGrid, transform: Transform, scale: float) -> _Point:
    """The grid point nearest `transform`: its retention by distance, its half-saturation point
    by ratio, and its shape if the grid has it, else the grid's first."""
    retention = int(np.argmin(np.abs(np.array(grid.retentions) - transform.retention)))
    ratios = np.log(np.array(grid.half_saturations) * scale / transform.half_saturation)
    shape = grid.shapes.index(transform.shape) if transform.shape in grid.shapes else 0
    return _Point(retention, int(np.argmin(np.abs(ratios))), shape)


def _positive_mean(panel: MediaPanel, geo: Geo, column: str) -> float:
    """The channel's mean spend over the geo's periods with spend, the unit of its half-saturation
    point on a grid.

    Raises:
        ValueError: on a geo or channel the panel does not have, or a channel with no spend on it.
    """
    if geo not in panel.geos:
        msg = f"the panel has no geo {geo!r}; it has {list(panel.geos)}"
        raise ValueError(msg)
    names = [str(c) for c in panel.channels]
    if column not in names:
        msg = f"a transform carries a channel's spend over, and {column!r} is not a channel"
        raise ValueError(msg)
    spend = panel.spend[panel.geos.index(geo), :, names.index(column)]
    if not (spend > 0.0).any():
        msg = (
            f"channel {column!r} spends nothing on geo {geo!r}, so its half-saturation has no unit"
        )
        raise ValueError(msg)
    return float(spend[spend > 0.0].mean())
