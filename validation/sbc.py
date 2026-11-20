"""Simulation-based calibration of the engine's Bayesian models (Talts, Betancourt, Simpson,
Vehtari and Gelman 2018; Modrák, Moon, Kim, Bürkner, Huurre, Faltejsková, Gelman and Vehtari 2023).

    python -m validation.sbc --engine bayes_geo --sims 40 --seed 0 --world centred-geo

Each simulation draws a point from the engine's prior on its sampling coordinates, simulates the
data from it with the engine's own mean and noise, fits them with the engine's sampler at its
defaults, and records the rank of each true coordinate among `L = 63` posterior draws thinned
evenly from the chains' draws, so that a rank is one of 64 values, uniform where the fit is
calibrated. A coordinate is a one-to-one map of a parameter, or of a function of the parameters
(the one-geo model's level), so its rank is the parameter's. Beside the ranks it records the
fit's divergent transitions, each coordinate's R-hat and bulk and tail ESS
(`mixdyn.inference.diagnostics`), the mean of the draws at the divergent transitions, and the
compute record of ADR 0035: each fit's wall, CPU and compiling seconds, the process's peak resident
set and the least bulk ESS per CPU second; and a failure's kind and message.

The engines' models read the KPI over its mean, so their priors are stated on a scale the data set.
Simulated data would move that scale, and the posterior would no longer be the one the prior and
the likelihood make: the harness simulates on the model's own scales, a model built once from the
world's panel with its KPI replaced by the simulated one. What is calibrated is the density and the
sampler; the scaling is a map of the data, read the same way by every fit.

* `bayes_mmm`: `mixdyn.models.bayes_mmm`, its chains started by Pathfinder.
* `bayes_geo`: `mixdyn.models.bayes_geo`, its chains started uniform; on the `centred-geo` world
  every channel is centred. `--knob` sets one of `BayesGeoSpec`'s opt-in choices, `noise_prior`,
  `base_prior`, `slope`, or `centred`, a comma-separated list of channels, empty for none.
* `nested`: `mixdyn.models.nested` without lift estimates, two stages drawn apart from seeds
  spawned as `fit_nested` spawns them: the mediator simulated by stage one, then read by stage two
  as its control.
* `dlm`: `mixdyn.models.dlm.fit_dlm` with its effects held to their signs, the part of the DLM a
  sampler draws (`chc.dlm.constrained_sample`). The prior is `vague_prior`'s on the world's panel,
  fixed, and every discount is 1: a discount below 1 makes the evolution's variance a function of
  the filter, and a regression's prior over its path is then not one the harness can draw. The
  ranks are the last step's states and the log variance; its chains are Gibbs chains, which do not
  diverge.

The uniformity test is Pearson's chi-square over `B` equal bins of the 64 ranks, `B` the most of
2, 4, ..., 64 that leaves at least five simulations a bin; a block's p-value is the least of its
coordinates' times their number (Bonferroni), which holds whatever the ranks' dependence within a
simulation. A simulation whose fit raised has no ranks: it is counted, and left out of the test.
The records are written to `validation/sbc/<engine>-<world>[-<knob>-<value>].json` and `.md`.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import functools
import importlib.metadata
import json
import math
import resource
import shlex
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import monitoring
from scipy import special, stats

from mixdyn import NATIONAL, Cadence, Channel, Geo, MediaPanel, Revenue
from mixdyn.inference import compiled
from mixdyn.inference.diagnostics import diagnose
from mixdyn.models.bayes_geo import _NOISE_LOG_NORMAL, BayesGeoModel, BayesGeoSpec, bayes_geo_model
from mixdyn.models.bayes_mmm import (
    _PATHFINDER_START,
    _UNIFORM_START,
    BayesMedia,
    BayesModel,
    BayesSpec,
    Normal,
    Start,
    StudentT,
    _initial,
    _model,
    bayes_model,
)
from mixdyn.models.dlm import DlmSpec, Effect, Trend, _filter, fit_dlm, vague_prior
from mixdyn.models.nested import NestedSpec, nested_model
from mixdyn.models.ridge_mmm import Control

if TYPE_CHECKING:
    from collections.abc import Generator

    from chc.dlm import Prior

type _Array = npt.NDArray[np.float64]

THINNED = 63
"""Posterior draws a rank is read among, so that a rank is one of 64 values."""
LEAST_A_BIN = 5
"""Simulations a bin of the uniformity test expects at least."""
CHAINS = 4
WARMUP = 1000
DRAWS = 1000
"""The engines' defaults, which every fit here keeps."""
ENGINES = ("bayes_mmm", "bayes_geo", "nested", "dlm")
WORLDS = ("small", "centred-geo")
KNOBS = ("noise_prior", "base_prior", "slope", "centred")
"""`BayesGeoSpec`'s opt-in choices a run may set, one at a time."""
COMPILES = (
    "/jax/core/compile/jaxpr_trace_duration",
    "/jax/core/compile/jaxpr_to_mlir_module_duration",
    "/jax/core/compile/backend_compile_duration",
)
"""JAX's events for tracing, lowering and compiling a function, as ADR 0035 sums them."""
PACKAGES = ("mixdyn", "causal-hybrid-control", "jax", "numpy", "scipy")
RECORDS = Path(__file__).resolve().parent / "sbc"

_WEEKS = 52
_POPULATION = np.array([1.0, 2.0, 0.5, 1.5]) * 1e5


@dataclass(frozen=True, slots=True, eq=False)
class Fitted:
    """A fit's draws on the coordinates, `(chains, draws, coordinates)`, and its divergent
    transitions, `(chains, draws)`, or None for a sampler that has none."""

    draws: _Array
    divergent: npt.NDArray[np.bool_] | None


class Engine(Protocol):
    """A world of one engine: the coordinates it ranks, a simulation of its prior predictive, and
    the engine's fit of simulated data."""

    @property
    def coordinates(self) -> tuple[str, ...]: ...

    def simulate(self, rng: np.random.Generator) -> tuple[_Array, Any]:
        """A point drawn from the prior, on the coordinates, and data drawn given it."""
        ...

    def fit(self, data: Any, seed: int) -> Fitted: ...


def _weekly(rng: np.random.Generator, geos: int, channels: int) -> _Array:
    """Spend over `_WEEKS` weeks, `(geos, weeks, channels)`: each channel on in flights, about a
    quarter of its weeks dark, at a level that varies over the geos."""
    level = rng.lognormal(0.0, 0.5, (geos, 1, channels))
    on = rng.random((1, _WEEKS, channels)) > 0.25
    return level * rng.gamma(2.0, 1.0, (geos, _WEEKS, channels)) * on


def _panel(
    spend: _Array,
    channels: Sequence[str],
    *,
    geos: tuple[Geo, ...] = (NATIONAL,),
    controls: Mapping[str, _Array] | None = None,
    population: _Array | None = None,
) -> MediaPanel:
    """A weekly panel of `spend`, `(geos, weeks, channels)`, whose KPI is 1 a head throughout: the
    models' scales are read from it, and every simulation replaces it."""
    controls = controls or {}
    heads = np.ones(len(geos)) if population is None else population
    weeks = spend.shape[1]
    return MediaPanel(
        dates=np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(weeks),
        cadence=Cadence.WEEK,
        geos=geos,
        channels=tuple(Channel(c) for c in channels),
        controls=tuple(controls),
        kpi=np.broadcast_to(heads[:, None], (len(geos), weeks)).copy(),
        kpi_unit=Revenue(),
        spend=spend,
        currency="EUR",
        control_values=np.stack([*controls.values()], axis=2)
        if controls
        else np.empty((len(geos), weeks, 0)),
        filled=np.zeros((len(geos), weeks), dtype=bool),
        population=population,
    )


def _controls(controls: Sequence[Control], rng: np.random.Generator) -> _Array:
    """The controls' coordinates drawn from `N(0, 1)`, a control held to its sign as the log of the
    size of a draw."""
    z = rng.standard_normal(len(controls))
    signed = np.array([not (c.lower == -math.inf and c.upper == math.inf) for c in controls])
    return np.where(signed, np.log(np.abs(z)), z)


def _log_roi(media: Sequence[BayesMedia], rng: np.random.Generator) -> _Array:
    return rng.normal([b.roi.mu for b in media], [b.roi.sigma for b in media])


def _noise(likelihood: object, nu: _Array, rng: np.random.Generator, size: int) -> _Array:
    if isinstance(likelihood, StudentT):
        return rng.standard_t(float(nu[0]), size)
    return rng.standard_normal(size)


def _nuts(model: Any, start: Start, seed: int) -> Fitted:
    """The engine's sampler at its defaults, from the engine's starts."""
    initial = _initial(model, start, CHAINS, seed)
    result = compiled.sample(model._log_density, initial, seed=seed, warmup=WARMUP, draws=DRAWS)
    return Fitted(result.positions, result.divergent)


@dataclass(frozen=True, slots=True, eq=False)
class OneGeo:
    """`bayes_mmm` on one geo: its model on the world's panel."""

    model: BayesModel

    @property
    def coordinates(self) -> tuple[str, ...]:
        return self.model.coordinates

    def simulate(self, rng: np.random.Generator) -> tuple[_Array, BayesModel]:
        point = _mmm_point(self.model, rng)
        return point, _mmm_data(self.model, point, rng)

    def fit(self, data: BayesModel, seed: int) -> Fitted:
        return _nuts(data, _PATHFINDER_START, seed)


def _mmm_point(model: BayesModel, rng: np.random.Generator) -> _Array:
    """A point of `model`'s prior on its coordinates: the level is the intercept drawn from its
    prior plus the channels' contribution its returns set."""
    spec, sizes = model.spec, model._sizes
    if model.arms or model.estimates or model.series or model.tails:
        msg = "the harness draws no geo test, lift estimate or carried return"
        raise ValueError(msg)
    if not isinstance(spec.likelihood, Normal | StudentT):
        msg = f"the harness simulates normal or Student-t noise, not {spec.likelihood}"
        raise ValueError(msg)
    m = len(spec.media)
    log_roi = _log_roi(spec.media, rng)
    level = rng.normal(1.0, 1.0) + float(model._media_level({"log_roi": jnp.asarray(log_roi)}))
    blocks = {
        "log_roi": log_roi,
        "retention": special.logit(rng.beta(1.0, 3.0, m)),
        "half": rng.normal(0.0, 1.0, m),
        "slope": rng.normal(0.0, 0.5, sizes["slope"]),
        "level": np.array([level]),
        "calendar": rng.standard_normal(sizes["calendar"]),
        "control": _controls(spec.controls, rng),
        "sigma": np.log(np.abs(rng.standard_normal(1))),
        "nu": np.log(rng.gamma(2.0, 10.0, sizes["nu"])),
        "test_sigma": np.zeros(0),
    }
    return np.concatenate([blocks[name] for name in sizes])


def _mmm_data(model: BayesModel, point: _Array, rng: np.random.Generator) -> BayesModel:
    p, _ = model._constrained(jnp.asarray(point))
    mean = np.asarray(model._mean(p, *model._terms(p)))
    nu = np.asarray(p["nu"])
    y = mean + float(p["sigma"][0]) * _noise(model.spec.likelihood, nu, rng, mean.size)
    return dataclasses.replace(model, y=y)


@dataclass(frozen=True, slots=True, eq=False)
class Geos:
    """`bayes_geo` over the world's geos."""

    model: BayesGeoModel

    @property
    def coordinates(self) -> tuple[str, ...]:
        return self.model.coordinates

    def simulate(self, rng: np.random.Generator) -> tuple[_Array, BayesGeoModel]:
        model, spec, sizes = self.model, self.model.spec, self.model._sizes
        if model.estimates:
            msg = "the harness draws no lift estimate"
            raise ValueError(msg)
        m, g = len(spec.media), len(model.geos)
        spread = np.abs(rng.normal(0.0, spec.spread, m))
        deviation = rng.standard_normal((m, g - 1))
        centred = np.array([b.channel in spec.centred for b in spec.media])
        deviation = np.where(centred[:, None], spread[:, None] * deviation, deviation)
        if spec.noise_prior == "half-normal":
            sigma = abs(rng.standard_normal())
        else:
            sigma = rng.lognormal(*_NOISE_LOG_NORMAL)
        blocks = {
            "log_roi": _log_roi(spec.media, rng),
            "retention": special.logit(rng.beta(1.0, 3.0, m)),
            "half": rng.normal(0.0, 1.0, m),
            "slope": rng.normal(0.0, 0.5, sizes["slope"]),
            "spread": np.log(spread),
            "deviation": deviation.reshape(-1),
            "base": rng.normal(0.0 if spec.base_prior == "log-normal" else 1.0, 1.0, g),
            "calendar": rng.standard_normal(sizes["calendar"]),
            "control": _controls(spec.controls, rng),
            "sigma": np.log([sigma]),
            "nu": np.log(rng.gamma(2.0, 10.0, sizes["nu"])),
        }
        point = np.concatenate([blocks[name] for name in sizes])
        p, _ = model._constrained(jnp.asarray(point))
        mean = np.asarray(model._mean(p, *model._terms(p)))[model.observed]
        scale = sigma * np.broadcast_to(model.weight[:, None] ** -0.5, model.observed.shape)
        noise = _noise(spec.likelihood, np.asarray(p["nu"]), rng, mean.size)
        return point, dataclasses.replace(model, y=mean + scale[model.observed] * noise)

    def fit(self, data: BayesGeoModel, seed: int) -> Fitted:
        return _nuts(data, _UNIFORM_START, seed)


@dataclass(frozen=True, slots=True, eq=False)
class Nested:
    """`nested` on one geo: stage one's model on the world's panel, and the panel, whose mediator
    each simulation replaces before stage two's model is built on it."""

    panel: MediaPanel
    spec: NestedSpec
    one: BayesModel
    two: tuple[str, ...]

    @property
    def coordinates(self) -> tuple[str, ...]:
        return (*(f"one.{c}" for c in self.one.coordinates), *(f"two.{c}" for c in self.two))

    def simulate(self, rng: np.random.Generator) -> tuple[_Array, tuple[BayesModel, BayesModel]]:
        first = _mmm_point(self.one, rng)
        one = _mmm_data(self.one, first, rng)
        k = self.panel.controls.index(self.spec.mediator)
        values = self.panel.control_values.copy()
        values[0, :, k] = one.y * one.kpi_unit
        panel = dataclasses.replace(self.panel, control_values=values)
        model = _model(panel, NATIONAL, self.spec.stage_two, None, {}, {}, 0)
        second = _mmm_point(model, rng)
        return np.concatenate([first, second]), (one, _mmm_data(model, second, rng))

    def fit(self, data: tuple[BayesModel, BayesModel], seed: int) -> Fitted:
        one, two, _ = (int(s) for s in np.random.SeedSequence(seed).generate_state(3))
        first, second = (
            _nuts(data[0], _PATHFINDER_START, one),
            _nuts(data[1], _PATHFINDER_START, two),
        )
        assert first.divergent is not None
        assert second.divergent is not None
        return Fitted(
            np.concatenate([first.draws, second.draws], axis=2),
            first.divergent | second.divergent,
        )


@dataclass(frozen=True, slots=True, eq=False)
class Dlm:
    """`fit_dlm` with signed effects on one geo, its prior fixed at `vague_prior`'s on the world's
    panel, every discount 1; `design` is each step's row of the observation's design."""

    panel: MediaPanel
    spec: DlmSpec
    prior: Prior
    design: _Array

    @property
    def coordinates(self) -> tuple[str, ...]:
        trend = ["level", "slope"][: self.spec.trend.order] if self.spec.trend else []
        effects = [f"coefficient[{e.column}]" for e in self.spec.effects]
        return (*trend, *effects, "log_variance")

    def simulate(self, rng: np.random.Generator) -> tuple[_Array, MediaPanel]:
        prior = self.prior
        variance = 1.0 / rng.gamma(prior.dof / 2.0, 2.0 / (prior.dof * prior.scale))
        root = np.linalg.cholesky(prior.covariance * variance / prior.scale)
        state = prior.mean + root @ rng.standard_normal(prior.mean.size)
        first = self.spec.trend.order if self.spec.trend else 0
        signed = {first + j: sign for j, sign in self.spec.signs().items()}
        for i, sign in signed.items():
            state[i] = abs(state[i]) if sign == "positive" else -abs(state[i])
        y = self.design @ state + math.sqrt(variance) * rng.standard_normal(self.design.shape[0])
        point = np.append(state, math.log(variance))
        return point, dataclasses.replace(self.panel, kpi=y[None])

    def fit(self, data: MediaPanel, seed: int) -> Fitted:
        fit = fit_dlm(data, NATIONAL, self.spec, self.prior, draws=DRAWS, seed=seed)
        assert fit.signed is not None
        draws = np.concatenate(
            [fit.signed.states[:, :, -1, :], np.log(fit.signed.variance)[..., None]], axis=2
        )
        return Fitted(draws, None)


def _dlm(spec: DlmSpec, panel: MediaPanel) -> Dlm:
    """The DLM's world, its signed coordinates' prior independent of the rest and centred on 0, so
    that the prior truncated to the signs is each one's size given its sign."""
    prior = vague_prior(panel, NATIONAL, spec)
    first = spec.trend.order if spec.trend else 0
    for j in spec.signs():
        i = first + j
        others = np.delete(prior.covariance[i], i)
        if prior.mean[i] != 0.0 or np.any(others != 0.0):
            msg = f"the harness draws a signed coordinate whose prior is N(0, v) alone, not {i}"
            raise ValueError(msg)
    blocks = [b.discount for b in spec.blocks()]
    if any(d != 1.0 for d in blocks):
        msg = f"the harness draws a DLM whose discounts are all 1, not {blocks}"
        raise ValueError(msg)
    design = _filter(panel, NATIONAL, spec, prior).design
    return Dlm(panel, spec, prior, design)


def world(engine: str, name: str, knobs: Mapping[str, str] | None = None) -> Engine:
    """The world `name` of `engine`, with `knobs` set on `bayes_geo`'s spec.

    Raises:
        ValueError: on an engine, a world or a knob the harness does not have.
    """
    knobs = dict(knobs or {})
    if engine not in ENGINES or name not in WORLDS:
        msg = f"an engine is one of {list(ENGINES)} and a world one of {list(WORLDS)}"
        raise ValueError(msg)
    if name == "centred-geo" and engine != "bayes_geo":
        msg = "the centred-geo world is bayes_geo's"
        raise ValueError(msg)
    if knobs and engine != "bayes_geo":
        msg = f"knobs are bayes_geo's, not {engine}'s"
        raise ValueError(msg)
    unknown = sorted(set(knobs) - set(KNOBS))
    if unknown:
        msg = f"knobs are {list(KNOBS)}, not {unknown}"
        raise ValueError(msg)
    rng = np.random.default_rng(2026)
    if engine == "bayes_mmm":
        panel = _panel(_weekly(rng, 1, 2), ("tv", "search"))
        spec = BayesSpec((BayesMedia("tv"), BayesMedia("search")))
        return OneGeo(bayes_model(panel, NATIONAL, spec))
    if engine == "bayes_geo":
        geos = tuple(Geo(f"geo{g}") for g in range(_POPULATION.size))
        panel = _panel(
            _weekly(rng, len(geos), 2), ("tv", "search"), geos=geos, population=_POPULATION
        )
        centred = frozenset({"tv", "search"} if name == "centred-geo" else ())
        if "centred" in knobs:
            centred = frozenset(c for c in knobs.pop("centred").split(",") if c)
        spec = BayesGeoSpec((BayesMedia("tv"), BayesMedia("search")), centred=centred, **knobs)  # ty: ignore[invalid-argument-type]
        return Geos(bayes_geo_model(panel, spec))
    if engine == "nested":
        spend = _weekly(rng, 1, 2)
        panel = _panel(spend, ("tv", "search"), controls={"brand": 1.0 + rng.random((1, _WEEKS))})
        spec = NestedSpec(
            "brand",
            BayesSpec((BayesMedia("tv"),)),
            BayesSpec((BayesMedia("search"),), (Control("brand", lower=0.0),)),
        )
        model = nested_model(panel, NATIONAL, spec)
        return Nested(panel, spec, model.stage_one, model.stage_two.coordinates)
    spend = _weekly(rng, 1, 2)
    panel = dataclasses.replace(
        _panel(spend, ("tv", "search")), kpi=10.0 + spend.sum(axis=2) + rng.standard_normal(_WEEKS)
    )
    spec = DlmSpec(
        (Effect("tv", 1.0, sign="positive"), Effect("search", 1.0, sign="positive")),
        Trend(1, 1.0),
    )
    return _dlm(spec, panel)


def rank(truth: _Array, draws: _Array) -> npt.NDArray[np.int64]:
    """Each coordinate's rank among `THINNED` of `draws`, `(chains, draws, coordinates)`, evenly
    spaced over the chains' draws in order: how many are below the truth, from 0 to `THINNED`."""
    flat = draws.reshape(-1, draws.shape[2])
    keep = np.linspace(0, flat.shape[0] - 1, THINNED).round().astype(np.intp)
    return (flat[keep] < truth).sum(axis=0)


def bins(simulations: int) -> int:
    """The most of 2, 4, ..., `THINNED + 1` equal bins that expect `LEAST_A_BIN` simulations each,
    2 at least."""
    out = 2
    while out < THINNED + 1 and simulations / (2 * out) >= LEAST_A_BIN:
        out *= 2
    return out


def histogram(ranks: npt.NDArray[np.int64], count: int) -> npt.NDArray[np.int64]:
    """The ranks' counts in `count` equal bins of the `THINNED + 1` values."""
    return np.bincount(ranks * count // (THINNED + 1), minlength=count)


def uniformity(ranks: npt.NDArray[np.int64], coordinates: Sequence[str]) -> dict[str, Any]:
    """Pearson's chi-square p-value of each coordinate's ranks, `(simulations, coordinates)`, over
    `bins`' bins, and each block's: the least of its coordinates' times their number, at most 1. A
    block is the coordinates whose names agree up to their first `[`."""
    count = bins(ranks.shape[0])
    each = {
        c: float(stats.chisquare(histogram(ranks[:, k], count)).pvalue)
        for k, c in enumerate(coordinates)
    }
    blocks: dict[str, list[str]] = {}
    for c in coordinates:
        blocks.setdefault(c.split("[")[0], []).append(c)
    return {
        "bins": count,
        "coordinates": each,
        "blocks": {
            b: {"p": min(1.0, len(cs) * min(each[c] for c in cs)), "coordinates": cs}
            for b, cs in blocks.items()
        },
    }


@contextmanager
def _compiling() -> Generator[Callable[[], float]]:
    """The seconds JAX has spent compiling since entry (`COMPILES`), read when called."""
    spent = [0.0]

    def heard(event: str, duration_secs: float, **_: str | int) -> None:
        if event in COMPILES:
            spent[0] += duration_secs

    monitoring.register_event_duration_secs_listener(heard)
    try:
        yield lambda: spent[0]
    finally:
        monitoring.unregister_event_duration_listener(heard)


def _cpu() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def _peak_rss() -> int:
    """The process's peak resident set so far, in bytes: `VmHWM`, which Linux keeps in kB."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
    msg = "/proc/self/status holds no VmHWM"
    raise RuntimeError(msg)


def _finite(values: _Array) -> list[float | None]:
    """`values` as JSON holds them: a number that is not finite is null."""
    return [float(v) if math.isfinite(v) else None for v in np.asarray(values, dtype=float)]


@functools.cache
def _world(engine: str, name: str, knobs: tuple[tuple[str, str], ...]) -> Engine:
    return world(engine, name, dict(knobs))


def simulation(
    engine: str, name: str, knobs: tuple[tuple[str, str], ...], seed: int, index: int
) -> dict[str, Any]:
    """Simulation `index` of a run from `seed`: its own streams, `SeedSequence([seed, index])`'s,
    so a record does not depend on which process ran it or in what order.

    Raises:
        ValueError: on what `world` refuses.
    """
    jax.config.update("jax_enable_x64", True)
    model = _world(engine, name, knobs)
    streams = np.random.SeedSequence([seed, index])
    simulated, fitting = streams.spawn(2)
    fit_seed = int(fitting.generate_state(1)[0])
    truth, data = model.simulate(np.random.default_rng(simulated))
    record: dict[str, Any] = {"index": index, "fit_seed": fit_seed, "true": _finite(truth)}
    with _compiling() as compiling:
        began, cpu = time.perf_counter(), _cpu()
        try:
            fitted: Fitted | None = model.fit(data, fit_seed)
            failure = None
        except Exception as error:  # a fit that raises is a result of the run, recorded
            fitted, failure = None, f"{type(error).__name__}: {error}"
        wall, cpu, compile_seconds = time.perf_counter() - began, _cpu() - cpu, compiling()
    compute: dict[str, Any] = {
        "fit_wall_seconds": wall,
        "fit_cpu_seconds": cpu,
        "fit_compile_seconds": compile_seconds,
        "fit_peak_rss_bytes": _peak_rss(),
        "failure_kind": None if fitted is not None else "fit",
    }
    if fitted is None:
        return record | {"error": failure, "compute": compute}
    health = diagnose(fitted.draws)
    least = float(np.nanmin(health.ess_bulk)) if np.isfinite(health.ess_bulk).any() else math.nan
    if math.isfinite(least) and cpu > 0.0:
        compute["ess_bulk_per_fit_cpu_second"] = least / cpu
    divergent = fitted.divergent
    at = None
    if divergent is not None and divergent.any():
        at = _finite(fitted.draws[divergent].mean(axis=0))
    return record | {
        "error": None,
        "ranks": rank(truth, fitted.draws).tolist(),
        "divergent": None if divergent is None else int(divergent.sum()),
        "divergent_mean": at,
        "rhat": _finite(health.rhat),
        "ess_bulk": _finite(health.ess_bulk),
        "ess_tail": _finite(health.ess_tail),
        "compute": compute,
    }


def summary(records: Sequence[Mapping[str, Any]], coordinates: Sequence[str]) -> dict[str, Any]:
    """The run's uniformity, divergences and health over its records."""
    fitted = [r for r in records if r["error"] is None]
    out: dict[str, Any] = {"simulations": len(records), "fitted": len(fitted)}
    out["failed"] = len(records) - len(fitted)
    if not fitted:
        return out
    ranks = np.array([r["ranks"] for r in fitted], dtype=np.int64)
    out["uniformity"] = uniformity(ranks, coordinates)
    counts = [r["divergent"] for r in fitted if r["divergent"] is not None]
    if counts:
        out["divergence_share"] = sum(c > 0 for c in counts) / len(counts)
        out["divergent_transitions"] = sum(counts)
        out["divergent_per_fit"] = {"median": float(np.median(counts)), "most": max(counts)}
    rhat = [max((v for v in r["rhat"] if v is not None), default=math.inf) for r in fitted]
    ess = [min((v for v in r["ess_bulk"] if v is not None), default=0.0) for r in fitted]
    out["rhat_above_1_01"] = sum(v > 1.01 for v in rhat)
    out["ess_bulk_below_400"] = sum(v < 400.0 for v in ess)
    return out


def traced(records: Sequence[Mapping[str, Any]], coordinates: Sequence[str]) -> dict[str, Any]:
    """Where the divergences fall, coordinate by coordinate: the share of fits that diverged among
    the simulations whose true value is below the coordinate's median over the run and among those
    above it; Spearman's correlation of a fit's divergent transitions with the true value, which
    still reads where every fit diverged; and over the fits that diverged, the mean of each
    coordinate's draws at the divergent transitions less its true value."""
    fitted = [r for r in records if r["error"] is None and r["divergent"] is not None]
    if not any(r["divergent"] for r in fitted):
        return {}
    truth = np.array([[np.nan if v is None else v for v in r["true"]] for r in fitted])
    counts = np.array([r["divergent"] for r in fitted])
    diverged = counts > 0
    at = np.array(
        [
            [np.nan if v is None else v for v in r["divergent_mean"]]
            for r in fitted
            if r["divergent"]
        ]
    )
    out = {}
    for k, c in enumerate(coordinates):
        low = truth[:, k] <= np.nanmedian(truth[:, k])
        out[c] = {
            "share_below_median": float(diverged[low].mean()) if low.any() else None,
            "share_above_median": float(diverged[~low].mean()) if (~low).any() else None,
            "spearman": float(stats.spearmanr(counts, truth[:, k]).statistic),
            "divergent_less_true": float(np.nanmean(at[:, k] - truth[diverged, k])),
        }
    return out


def _bar(counts: npt.NDArray[np.int64], expected: float) -> str:
    """Each bin's count as a character of `.:-=+*#%@`, its height in quarters of `expected`."""
    levels = ".:-=+*#%@"
    return "".join(levels[min(len(levels) - 1, int(4 * c / expected))] for c in counts)


def markdown(document: Mapping[str, Any]) -> str:
    """The run's record as text: its command and date, each block's uniformity p-value, each
    coordinate's rank histogram, the divergences and health, and where the divergences fall."""
    s = document["summary"]
    title = f"{document['engine']}, world {document['world']}"
    if document["knobs"]:
        title += ", " + ", ".join(f"{k} = {v!r}" for k, v in document["knobs"].items())
    lines = [
        f"# Simulation-based calibration: {title}",
        "",
        f"Command: `{document['command']}`, run {document['date']}. Protocol: ADR 0059.",
        "",
        f"{s['simulations']} simulations, {s['fitted']} fitted, {s['failed']} failed; ranks among"
        f" {document['thinned']} thinned draws of {CHAINS} chains of {DRAWS} after {WARMUP} of"
        " warm-up.",
    ]
    if "uniformity" not in s:
        return "\n".join([*lines, ""])
    u = s["uniformity"]
    lines += [
        "",
        "## Uniformity",
        "",
        f"Pearson's chi-square over {u['bins']} bins of the 64 ranks; a block's p-value is the"
        " least of its coordinates' times their number. The pass rule, at 200 simulations, is"
        " p > 0.01 for every block.",
        "",
        "| block | coordinates | p |",
        "|---|---|---|",
    ]
    for block, b in u["blocks"].items():
        lines.append(f"| `{block}` | {len(b['coordinates'])} | {b['p']:.3g} |")
    expected = s["fitted"] / u["bins"]
    lines += [
        "",
        "## Rank histograms",
        "",
        f"Each bin's count in quarters of the {expected:g} a uniform rank expects:"
        " `.` none, `:` a quarter, `-` a half, `=` three quarters, `+` as expected, and on to `@`"
        " at twice.",
        "",
        "```",
    ]
    ranks = np.array([r["ranks"] for r in document["records"] if r["error"] is None])
    width = max(len(c) for c in document["coordinates"])
    for k, c in enumerate(document["coordinates"]):
        counts = histogram(ranks[:, k], u["bins"])
        p = u["coordinates"][c]
        lines.append(f"{c:<{width}}  {_bar(counts, expected)}  p {p:.3f}  {counts.tolist()}")
    lines += ["```", "", "## Divergences and health", ""]
    if "divergence_share" in s:
        d = s["divergent_per_fit"]
        lines.append(
            f"Fits with a divergent transition: {s['divergence_share']:.0%}"
            f" ({round(s['divergence_share'] * s['fitted'])} of {s['fitted']});"
            f" {s['divergent_transitions']} divergent transitions in all, a fit's median"
            f" {d['median']:g} and most {d['most']}."
        )
    else:
        lines.append("The sampler has no divergent transitions (Gibbs chains).")
    lines.append(
        f"Fits with an R-hat above 1.01: {s['rhat_above_1_01']}; with a bulk ESS below 400:"
        f" {s['ess_bulk_below_400']}."
    )
    trace = document.get("traced") or {}
    if trace:
        lines += [
            "",
            "## Where the divergences fall",
            "",
            "Share of fits that diverged among simulations whose true value is at or below the"
            " coordinate's median over the run, and above it; Spearman's correlation of a fit's"
            " divergent transitions with the true value; and the mean of the draws at the"
            " divergent transitions less the true value, over the fits that diverged.",
            "",
            "| coordinate | diverged, true at or below median | above | Spearman |"
            " divergent less true |",
            "|---|---|---|---|---|",
        ]
        for c, t in trace.items():
            lines.append(
                f"| `{c}` | {t['share_below_median']:.0%} | {t['share_above_median']:.0%} |"
                f" {t['spearman']:+.2f} | {t['divergent_less_true']:+.2f} |"
            )
    failures = [r for r in document["records"] if r["error"] is not None]
    if failures:
        lines += ["", "## Failures", ""]
        lines += [f"- simulation {r['index']}: {r['error']}" for r in failures]
    return "\n".join([*lines, ""])


def stem(engine: str, name: str, knobs: Mapping[str, str]) -> str:
    """The records' file name without its suffix."""
    return "-".join([engine, name, *(f"{k}-{v or 'none'}" for k, v in knobs.items())])


def run(
    engine: str,
    name: str,
    sims: int,
    seed: int,
    *,
    knobs: Mapping[str, str] | None = None,
    workers: int = 1,
    command: str = "",
) -> dict[str, Any]:
    """The run's document: its settings, each simulation's record, and their summary.

    Raises:
        ValueError: on fewer than one simulation or worker, and on what `world` refuses.
    """
    if sims < 1 or workers < 1:
        msg = f"need a simulation and a worker at least, got {sims} and {workers}"
        raise ValueError(msg)
    knobs = dict(knobs or {})
    frozen = tuple(sorted(knobs.items()))
    coordinates = world(engine, name, knobs).coordinates
    tasks = [(engine, name, frozen, seed, i) for i in range(sims)]
    if workers == 1:
        records = [simulation(*t) for t in tasks]
    else:
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
            records = list(pool.map(simulation, *zip(*tasks, strict=True)))
    return {
        "engine": engine,
        "world": name,
        "knobs": knobs,
        "command": command,
        "date": datetime.date.today().isoformat(),
        "seed": seed,
        "thinned": THINNED,
        "chains": CHAINS,
        "warmup": WARMUP,
        "draws": DRAWS,
        "versions": {p: importlib.metadata.version(p) for p in PACKAGES},
        "coordinates": list(coordinates),
        "summary": summary(records, coordinates),
        "traced": traced(records, coordinates),
        "records": records,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m validation.sbc", description=__doc__)
    parser.add_argument("--engine", choices=ENGINES, required=True)
    parser.add_argument("--sims", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--world", choices=WORLDS, default="small")
    parser.add_argument(
        "--knob", action="append", default=[], metavar="NAME=VALUE", help="bayes_geo's, one"
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, default=RECORDS)
    given = sys.argv[1:] if argv is None else list(argv)
    args = parser.parse_args(given)
    if len(args.knob) > 1 or any("=" not in k for k in args.knob):
        parser.error("one --knob NAME=VALUE at most: the knobs are tested one at a time")
    knobs = dict(k.split("=", 1) for k in args.knob)
    jax.config.update("jax_enable_x64", True)
    document = run(
        args.engine,
        args.world,
        args.sims,
        args.seed,
        knobs=knobs,
        workers=args.workers,
        command=shlex.join(["python", "-m", "validation.sbc", *given]),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / stem(args.engine, args.world, knobs)
    path.with_suffix(".json").write_text(json.dumps(document, indent=1) + "\n")
    path.with_suffix(".md").write_text(markdown(document))
    print(path.with_suffix(".md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
